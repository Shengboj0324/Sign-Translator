"""Part-wise residual-VQ motion tokenizer and its depth-autoregressive text prior (tokens v2).

**Why.** The v1 tokenizer (``motion_tokens.MotionVQ``) quantises the whole signer —
body, 30 finger joints, face, gaze — to one of K codes per token. The joint
configuration space is a product over parts, so a single codebook leaves a large
quantisation error: on How2Sign v1 plateaued at ≈24° mean joint error and ≈10 cm
hand-point error while its commitment term stayed high. Two structural changes:

* **Parts.** Body (+ torso and head translation), left hand, right hand and face (+
  head rotation, gaze, blink) get their own encoders and quantisers, so a handshape
  never shares a code with a brow raise; one shared decoder reconstructs all
  channels from all parts' latents, so hands stay coherent with the arms.
* **Residual quantisation** (Zeghidour et al. 2021; Lee et al. 2022). Stage q
  quantises the residual of the stages before it, ``r_q = r_{q−1} − e_q``, ``ẑ = Σ_q e_q``;
  for Q stages of K codes the representable set grows as Kᴽ while each stage stays
  a nearest-neighbour lookup. Every codebook is learnt by exponential moving
  averages of the residuals assigned to it, and unused codes are re-seeded.
  *Quantiser dropout*: with probability ``quantizer_dropout`` a batch uses only its
  first q ~ U{1..Q} stages, so stage 1 alone is a usable coarse code.

The straight-through estimator ``z + sg(ẑ − z)`` carries the reconstruction gradient
to the encoders; ``commitment · ‖z − sg(ẑ)‖²`` keeps them near their codes.

**Loss.** The canonical per-channel loss (as v1), plus the arm-and-hand skeleton
loss of ``signing_space.skeleton_motion_loss`` — position, velocity and
acceleration of all 44 arm and hand points, which weighs rotation errors by their
lever arm and holds reconstructions to the reference's speed and smoothness.

**Prior** (:class:`RVQPrior`). Codes of step t are ordered stage-major
(stage 1 of every part, then stage 2, …), ``c_t = (c_t^1 … c_t^J)``, J = P·Q, and the
sequence is factorised exactly as in the RQ-Transformer,

    p(c | text) = Π_t Π_j p(c_t^j | c_t^{<j}, h_t),  h_t = Temporal(text, c_{<t}),

a causal temporal transformer over steps (input: the summed embeddings of the
previous step's codes plus its clock) and a small causal *depth* transformer over the
J codes of a step, with one output head per code position (each has its own
codebook). Training is teacher-forced cross entropy summed over j; generation
samples j = 1..J in turn at every step, with classifier-free guidance on the logits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F

from ..data.multichannel import MotionBatch
from .canonical_state import _supervision_weights, channel_loss_per_example
from .motion_tokens import _ResBlock, _empty_like, token_seconds
from .signing_space import applicable, skeleton_motion_loss
from .text_to_motion import (
    PAD, VOCABULARY, LossScales, TextToMotionConfig, assemble_state, batch_seconds, diffusion_layout,
    diffusion_targets, diffusion_valid, empty_loci, encode_texts, generation_loci, grammar_projection, locus_projection, native_values,
    sinusoid, training_loci, unit_to_raw, utterance_duration,
)
from ..grammar_motion.text import token_features

PARTS = (("body", ("body", "root_translation", "head_translation")),
         ("left_hand", ("left_hand",)),
         ("right_hand", ("right_hand",)),
         ("face", ("face", "head_rotation", "left_gaze", "right_gaze", "blink")))


# --------------------------------------------------------------------------------------
# Tokenizer


@dataclass(frozen=True)
class MotionRVQConfig:
    motion: TextToMotionConfig
    hidden: int = 384
    code_dim: int = 128
    codebook: int = 512
    stages: int = 4
    down_layers: int = 1
    res_blocks: int = 2
    base_blocks: int = 0                # residual blocks at full frame rate (before downsampling / after upsampling);
                                        # with down_layers = 0 they are the network's only depth
    commitment: float = .25
    ema: float = .99
    quantizer_dropout: float = .5
    # round 3d (opt-in): the codebooks' EMA rate follows the learning rate (rate lr/peak, no dead-code re-seeding below
    # a quarter of the peak), so codebooks stop drifting while the network can no longer follow; False: earlier behaviour
    codebook_follows_lr: bool = False
    skeleton_weights: dict = field(default_factory=lambda: dict(position=1., velocity=.5, acceleration=.25))
    # round 3c (opt-in): per-channel velocity matching, {channel: weight}. Empty keeps the earlier tokenizers' loss.
    velocity_weights: dict = field(default_factory=dict)
    # round 10 (opt-in): a velocity term above this value counts as cap * (1 + log(term / cap)): log-compressed, so one
    # runaway channel cannot blow the update up (t2xb diverged at step 8900 when head-translation velocity, whose real
    # energy is tiny, reached 1.9e7). 0 keeps every earlier tokenizer's loss and SHA.
    velocity_cap: float = 0.

    def to_dict(self) -> dict:
        data = asdict(self)
        data["motion"] = self.motion.to_dict()
        if not data.get("velocity_cap"):
            data.pop("velocity_cap", None)
        if not data.get("velocity_weights"):          # absent when unused: earlier tokenizers' configs (and SHAs) unchanged
            data.pop("velocity_weights", None)
        if not data.get("codebook_follows_lr"):       # likewise
            data.pop("codebook_follows_lr", None)
        return data

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "MotionRVQConfig":
        data = dict(data)
        data["motion"] = TextToMotionConfig.from_dict(data["motion"])
        return cls(**data)

    @property
    def step(self) -> int:
        return 2 ** self.down_layers

    @property
    def width(self) -> int:
        return sum(entry[3] for entry in diffusion_layout(self.motion))

    def part_slices(self) -> list[tuple[str, list[int]]]:
        """(part, unit-vector columns) for every part that has a trained channel."""
        offset, columns = 0, {}
        for name, _, _, width in diffusion_layout(self.motion):
            columns[name] = list(range(offset, offset + width))
            offset += width
        owner = {c: part for part, channels in PARTS for c in channels}
        out = {}
        for name, cols in columns.items():
            out.setdefault(owner.get(name, "body"), []).extend(cols)
        return [(part, out[part]) for part, _ in PARTS if part in out]

    @property
    def codes_per_step(self) -> int:
        return len(self.part_slices()) * self.stages


class ResidualQuantizer(nn.Module):
    """Q EMA codebooks applied to successive residuals."""

    def __init__(self, stages: int, codebook: int, dim: int, ema: float) -> None:
        super().__init__()
        self.stages, self.codebook, self.ema = stages, codebook, ema
        self.register_buffer("codes", torch.randn(stages, codebook, dim) * .1)
        self.register_buffer("code_mass", torch.ones(stages, codebook))
        self.register_buffer("code_sum", self.codes.clone())

    @staticmethod
    def _nearest(x: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        d = x.pow(2).sum(-1, keepdim=True) - 2 * x @ codes.T + codes.pow(2).sum(-1)[None]
        return d.argmin(-1)

    def forward(self, z: torch.Tensor, stages: int | None = None):
        """(ẑ = Σ e_q, indices (…, Q), residual inputs per stage); stages beyond ``stages`` are not used."""
        used = self.stages if stages is None else stages
        flat = z.reshape(-1, z.shape[-1])
        residual, total, indices, inputs = flat, torch.zeros_like(flat), [], []
        for q in range(self.stages):
            index = self._nearest(residual, self.codes[q])
            indices.append(index)
            if q < used:
                e = self.codes[q][index]
                inputs.append(residual)
                total = total + e
                residual = residual - e
        idx = torch.stack(indices, -1).reshape(*z.shape[:-1], self.stages)
        return total.reshape(z.shape), idx, inputs

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """(…, Q) indices -> (…, dim) Σ_q codes[q][indices[..., q]]."""
        return sum(self.codes[q][indices[..., q]] for q in range(self.stages))

    @torch.no_grad()
    def update(self, inputs, indices: torch.Tensor, valid: torch.Tensor) -> None:
        flat_valid = valid.reshape(-1)
        for q, x in enumerate(inputs):
            x = x[flat_valid]
            index = indices.reshape(-1, self.stages)[flat_valid, q]
            if not len(x):
                continue
            onehot = F.one_hot(index, self.codebook).to(x.dtype)
            self.code_mass[q].mul_(self.ema).add_(onehot.sum(0), alpha=1 - self.ema)
            self.code_sum[q].mul_(self.ema).add_(onehot.T @ x, alpha=1 - self.ema)
            self.codes[q].copy_(self.code_sum[q] / self.code_mass[q].clamp_min(1e-5)[:, None])
            dead = self.code_mass[q] < 1e-3
            if bool(dead.any()):
                pick = torch.randint(len(x), (int(dead.sum()),), device=x.device)
                self.codes[q][dead] = x[pick]
                self.code_sum[q][dead] = x[pick]
                self.code_mass[q][dead] = 1.


def _encoder(width: int, config: MotionRVQConfig) -> nn.Sequential:
    h = config.hidden
    layers = [nn.Conv1d(width, h, 3, padding=1), *(_ResBlock(h) for _ in range(config.base_blocks))]
    for _ in range(config.down_layers):
        layers += [nn.Conv1d(h, h, 4, stride=2, padding=1), *(_ResBlock(h) for _ in range(config.res_blocks))]
    layers.append(nn.Conv1d(h, config.code_dim, 1))
    return nn.Sequential(*layers)


class MotionRVQ(nn.Module):
    def __init__(self, config: MotionRVQConfig) -> None:
        super().__init__()
        self.config = config
        self.parts = config.part_slices()
        self.encoders = nn.ModuleDict({part: _encoder(len(cols), config) for part, cols in self.parts})
        self.quantizers = nn.ModuleDict({part: ResidualQuantizer(config.stages, config.codebook, config.code_dim,
                                                                 config.ema) for part, _ in self.parts})
        h = config.hidden
        decoder = [nn.Conv1d(len(self.parts) * config.code_dim, h, 3, padding=1)]
        for _ in range(config.down_layers):
            decoder += [*(_ResBlock(h) for _ in range(config.res_blocks)), nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv1d(h, h, 3, padding=1)]
        decoder += [*(_ResBlock(h) for _ in range(config.base_blocks)), nn.SiLU(), nn.Conv1d(h, config.width, 3, padding=1)]
        self.decoder = nn.Sequential(*decoder)
        for part, cols in self.parts:
            self.register_buffer(f"columns_{part}", torch.tensor(cols, dtype=torch.long), persistent=False)
        # velocity matching's normaliser per channel: a running mean of the real per-pair motion energy (0 = not yet seen)
        for name in (config.velocity_weights or {}):
            self.register_buffer(f"velocity_scale_{name}", torch.zeros((), dtype=torch.float64), persistent=False)
        # the codebook update's rate (1: the configured EMA) and whether dead codes are re-seeded; the trainer sets
        # them each step when config.codebook_follows_lr (runtime state, not saved)
        self.codebook_rate, self.codebook_reseed = 1., True

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        frames = x.shape[1]
        padded = math.ceil(frames / self.config.step) * self.config.step
        return F.pad(x, (0, 0, 0, padded - frames)), frames

    def token_valid(self, frame_valid: torch.Tensor) -> torch.Tensor:
        return F.max_pool1d(self._pad(frame_valid[..., None].float())[0].transpose(1, 2),
                            self.config.step).transpose(1, 2)[..., 0] > 0

    def encode(self, unit: torch.Tensor) -> dict:
        """part -> (B, K, code_dim) latents."""
        padded, _ = self._pad(unit)
        return {part: self.encoders[part](padded[..., getattr(self, f"columns_{part}")].transpose(1, 2)).transpose(1, 2)
                for part, _ in self.parts}

    def _codebook_buffers(self, name: str) -> torch.Tensor:
        """A quantiser buffer of every part, stacked: (P, Q, …)."""
        return torch.stack([getattr(self.quantizers[part], name) for part, _ in self.parts])

    def quantize(self, latents: dict, stages: int | None = None):
        """(part -> ẑ, (B, K, J) stage-major code indices, residual inputs per used stage (P, N, code_dim)).

        All parts share the codebook shape, so each stage is one batched nearest-neighbour
        search over the parts — the same arithmetic as :class:`ResidualQuantizer` per part.
        """
        names = [part for part, _ in self.parts]
        used = self.config.stages if stages is None else stages
        z = torch.stack([latents[part] for part in names])                             # (P, B, K, D)
        flat = z.reshape(len(names), -1, z.shape[-1])
        codes = self._codebook_buffers("codes")                                         # (P, Q, C, D)
        residual, total, indices, inputs = flat, torch.zeros_like(flat), [], []
        for q in range(self.config.stages):
            book = codes[:, q]
            d = (residual.pow(2).sum(-1, keepdim=True) - 2 * residual @ book.transpose(1, 2)
                 + book.pow(2).sum(-1)[:, None])
            index = d.argmin(-1)                                                        # (P, N)
            indices.append(index)
            if q < used:
                e = torch.gather(book, 1, index[..., None].expand(-1, -1, book.shape[-1]))
                inputs.append(residual)
                total = total + e
                residual = residual - e
        quantized = dict(zip(names, total.reshape(z.shape).unbind(0)))
        grid = torch.stack(indices, -1).permute(1, 2, 0)                                # (N, Q, P)
        return quantized, grid.reshape(*z.shape[1:-1], -1), inputs                      # j = q·P + p

    @torch.no_grad()
    def update_codebooks(self, inputs, codes: torch.Tensor, valid: torch.Tensor) -> None:
        """The EMA codebook update of every part's :meth:`ResidualQuantizer.update`, batched over parts.

        The valid tokens are selected once for all parts and stages (one wait for the GPU).
        The assigned-residual sums stay one matrix product per part: cuBLAS's batched GEMM
        sums in another order, and these sums are what makes the update identical to the
        per-part one. Dead codes are re-seeded part by part and stage by stage, drawing
        random numbers in the same order.
        """
        if self.codebook_rate <= 0:
            return
        names, size = [part for part, _ in self.parts], self.config.codebook
        ema = 1. - (1. - self.config.ema) * min(float(self.codebook_rate), 1.)
        rows = valid.reshape(-1).nonzero()[:, 0]
        if not len(rows) or not inputs:
            return
        grid = codes.reshape(-1, self.config.stages, len(names))[rows]                 # (N valid, Q, P)
        mass, total, book = (self._codebook_buffers(n) for n in ("code_mass", "code_sum", "codes"))
        selected = []
        for q, x in enumerate(inputs):
            x = x[:, rows]                                                              # (P, N valid, D)
            selected.append(x)
            onehot = x.new_zeros(*x.shape[:2], size).scatter_(2, grid[:, q].T[..., None], 1.)
            mass[:, q].mul_(ema).add_(onehot.sum(1), alpha=1 - ema)
            sums = torch.stack([onehot[k].T @ x[k] for k in range(len(names))])
            total[:, q].mul_(ema).add_(sums, alpha=1 - ema)
            book[:, q] = total[:, q] / mass[:, q].clamp_min(1e-5)[..., None]
        for k, part in enumerate(names):
            quantizer = self.quantizers[part]
            quantizer.code_mass.copy_(mass[k]); quantizer.code_sum.copy_(total[k]); quantizer.codes.copy_(book[k])
        if not self.codebook_reseed:
            return
        dead = mass[:, :len(inputs)] < 1e-3                                             # (P, used, C)
        if not bool(dead.any()):
            return
        for k, part in enumerate(names):
            quantizer = self.quantizers[part]
            for q, x in enumerate(selected):
                if not bool(dead[k, q].any()):
                    continue
                pick = torch.randint(x.shape[1], (int(dead[k, q].sum()),), device=x.device)
                quantizer.codes[q][dead[k, q]] = x[k][pick]
                quantizer.code_sum[q][dead[k, q]] = x[k][pick]
                quantizer.code_mass[q][dead[k, q]] = 1.

    def split_codes(self, codes: torch.Tensor) -> dict:
        """(B, K, J) stage-major indices -> part -> (B, K, Q)."""
        p = len(self.parts)
        grid = codes.reshape(*codes.shape[:-1], self.config.stages, p)
        return {part: grid[..., k] for k, (part, _) in enumerate(self.parts)}

    def decode_latents(self, quantized: dict, frames: int) -> torch.Tensor:
        x = torch.cat([quantized[part] for part, _ in self.parts], dim=-1)
        return self.decoder(x.transpose(1, 2)).transpose(1, 2)[:, :frames]

    def decode_codes(self, codes: torch.Tensor, frames: int) -> torch.Tensor:
        per_part = self.split_codes(codes)
        return self.decode_latents({part: self.quantizers[part].lookup(per_part[part]) for part, _ in self.parts}, frames)

    def forward(self, unit: torch.Tensor, frame_valid: torch.Tensor, stages: int | None = None):
        latents = self.encode(unit)
        quantized, codes, inputs = self.quantize(latents, stages)
        valid = self.token_valid(frame_valid)
        if self.training:
            self.update_codebooks(inputs, codes, valid)
        straight = {part: latents[part] + (quantized[part] - latents[part]).detach() for part, _ in self.parts}
        return self.decode_latents(straight, unit.shape[1]), latents, quantized, codes, valid


def velocity_matching_loss(prediction: torch.Tensor, channel, frame_valid: torch.Tensor,
                           scale: torch.Tensor | None = None, momentum: float = .99) -> torch.Tensor | None:
    """Relative frame-to-frame velocity error of one channel, in its own parameterisation (``B,T,J,D``):

        L = Σ_{supported t,t+1} ||(x̂_{t+1} - x̂_t) - (x_{t+1} - x_t)||² / Σ ||x_{t+1} - x_t||²,

    over consecutive frames both supervised. Scale-free (1 when the reconstruction is static, 0 when its motion is
    exact), so it penalises damping (hands too slow) and added jitter (a near-still head turned shaky) alike: the two
    tokenizer faults the round-3c fidelity check found, which per-frame reconstruction loss does not see. ``None``
    without a supported pair."""
    w = _supervision_weights(channel, frame_valid, allow_inferred=True) > 0          # (B, T, J)
    pair = w[:, 1:] & w[:, :-1]
    if not bool(pair.any()):
        return None
    target = channel.values.to(prediction.dtype)
    dp = prediction[:, 1:] - prediction[:, :-1]
    dt = torch.where(pair[..., None], target[:, 1:] - target[:, :-1], torch.zeros_like(dp))
    err = torch.where(pair[..., None], dp - dt, torch.zeros_like(dp))
    if scale is None:                                   # the batch's own motion energy (s8v; unstable on still batches)
        return (err ** 2).sum() / ((dt ** 2).sum() + 1e-8)
    # stable form: normalise by a running mean of the real per-pair energy, updated only in training. One batch of
    # near-still clips (a still head, rest poses) can no longer drive the denominator to zero (s8v diverged at step
    # 22000 with the per-batch form).
    pairs = pair.sum()
    energy = ((dt ** 2).sum() / pairs).detach().to(scale.dtype)
    if prediction.requires_grad:
        scale.copy_(energy if float(scale) == 0. else momentum * scale + (1. - momentum) * energy)
    denominator = (scale if float(scale) > 0. else energy).to(err.dtype)
    return (err ** 2).sum() / pairs / (denominator + 1e-12)


def rvq_loss(model: MotionRVQ, motion: MotionBatch, *, stages: int | None = None):
    """Per-channel reconstruction loss + skeleton position/velocity/acceleration + commitment.

    In training, quantiser dropout draws the number of stages; ``stages`` forces it.
    The terms are detached 0-dim tensors (``stages`` a float): reading them waits for the
    GPU, so a training loop converts them only when it logs.
    """
    config = model.config
    if stages is None and model.training and torch.rand(()) < config.quantizer_dropout:
        stages = int(torch.randint(1, config.stages + 1, ()))
    unit = diffusion_targets(config.motion, motion)
    reconstruction, latents, quantized, codes, valid = model(unit, motion.frame_valid, stages)
    raw = unit_to_raw(config.motion, reconstruction.double())
    scales = LossScales(**config.motion.loss_config)
    channels = []
    for name in config.motion.trained_channels:
        values, support, _ = channel_loss_per_example(name, raw[name], motion.channels[name], motion.frame_valid,
                                                      config=scales, timestamps=motion.timestamps, allow_inferred=True)
        channels.append((name, values, support))
    total = torch.zeros((), dtype=torch.float64, device=unit.device)
    terms = {}
    # which channels have support in this batch: one host sync for all of them
    present = torch.stack([support.any() for _, _, support in channels]).tolist() if channels else []
    for (name, values, support), keep in zip(channels, present):
        if keep:
            term = torch.where(support, values, torch.zeros_like(values)).sum() / support.sum()
            terms[name] = term.detach()
            total = total + term
    if applicable(config.motion.bindings, config.motion.trained_channels):
        # float64 was needed for the SO(3) projection (unit_to_raw); the projected 6D columns are
        # orthonormal, so the forward kinematics runs in float32 (GPUs run float64 at a fraction of the rate).
        skeleton = skeleton_motion_loss({n: raw[n].float() for n in ("body", "left_hand", "right_hand")}, motion)
        for key, value in skeleton.items():
            terms[f"skeleton_{key}"] = value.detach()
            total = total + config.skeleton_weights[key] * value.to(total.dtype)
    for name, weight in (config.velocity_weights or {}).items():           # round 3c (opt-in)
        if name in raw and name in motion.channels:
            value = velocity_matching_loss(raw[name], motion.channels[name], motion.frame_valid,
                                           scale=getattr(model, f"velocity_scale_{name}", None))
            if value is not None:
                terms[f"velocity_{name}"] = value.detach()
                cap = getattr(config, "velocity_cap", 0.)
                if cap > 0 and bool(value > cap):
                    value = cap * (1. + torch.log(value / cap))
                total = total + float(weight) * value.to(total.dtype)
    names = [p for p, _ in model.parts]
    z, zq = torch.stack([latents[p] for p in names]), torch.stack([quantized[p] for p in names])   # (P, B, K, D)
    error = ((z - zq.detach()) ** 2).mean(-1)
    commitment = (torch.where(valid, error, torch.zeros_like(error)).sum((1, 2)) / valid.sum()).mean()
    terms["commitment"] = commitment.detach()
    terms["stages"] = float(stages or config.stages)
    # distinct stage-1 codes per part on valid tokens (index ``codebook`` collects the padding)
    first = torch.where(valid[..., None], codes[..., :len(names)], config.codebook).reshape(-1, len(names)).T
    seen = torch.zeros(len(names), config.codebook + 1, dtype=torch.bool, device=codes.device).scatter_(1, first, True)
    for p, count in zip(names, seen[:, :config.codebook].sum(1).unbind()):
        terms[f"codes_used_{p}"] = count
    return total + config.commitment * commitment.to(total.dtype), terms


# --------------------------------------------------------------------------------------
# Depth-autoregressive prior


@dataclass(frozen=True)
class RVQPriorConfig:
    tokenizer: MotionRVQConfig
    d_model: int = 384
    heads: int = 8
    text_layers: int = 4
    temporal_layers: int = 6
    depth_layers: int = 2
    feedforward: int = 1536
    dropout: float = .1
    max_text_bytes: int = 256
    cfg_dropout: float = .1
    duration_weight: float = .1
    grammar_features: bool = True
    locus_conditioning: bool = False
    locus_dropout: float = .3
    # round 4 (opt-in): a frozen pretrained text encoder (Hugging Face name, e.g. sentence-transformers/all-MiniLM-L6-v2
    # or google/flan-t5-base) whose last hidden states, projected to d_model, replace the byte embedding as the input of
    # the trainable text layers. None keeps the byte encoder (and every earlier prior's config and SHA).
    text_model: str | None = None
    text_model_revision: str | None = None
    # round 7 (opt-in, training only): the previous-code inputs of each sequence are replaced by random codes at a
    # rate drawn uniformly in [0, input_corruption] per sequence (T2M-GPT's corruption), so the prior cannot coast on
    # motion continuity and must read the text. Round 6's audit: the text moved the prediction by about 0.01 nats.
    # 0 keeps every earlier prior's config and SHA.
    input_corruption: float = 0.
    # round 8 candidate (opt-in, training only): scheduled sampling against exposure bias (identification test 2:
    # one-step samples from the real history read 0.82-0.87, free run 0.017). A first, gradient-free pass draws every
    # step's codes from the prior given the real history (each code given the real codes before it in its step: one
    # parallel depth pass); each sequence then replaces its history by those model codes
    # at a rate drawn uniformly in [0, scheduled_sampling], so training sees histories like its own samples.
    # 0 keeps every earlier prior's config and SHA.
    scheduled_sampling: float = 0.
    # round 9 candidates against the copycat shortcut (2026-10-27 research report: at 24 fps the full previous frame
    # nearly determines the next, so the text matters only where history cannot predict, the sign's onset).
    # history_dropout (training only): with this probability per sequence, every history step before a random cut
    # is replaced by the start code (index K), as at generation's first step, so the prior must introduce the sign
    # from the text at any point. history_stages: the temporal transformer sums only the first n residual stages of
    # each part of the previous step (training and generation alike; 0 = all), removing the fine-detail copy path.
    # 0 keeps every earlier prior's config and SHA.
    history_dropout: float = 0.
    history_stages: int = 0
    # round 10 (opt-in): a code's cross entropy counts only where its part was observed (at least half of the part's
    # unit columns valid in the token's frames). Untracked hands or arms are filled with the neutral pose by
    # diffusion_targets, so without this the prior learns to emit "flat hand" codes there (2026-10-27 audit;
    # WLASL's untracked arm frames). False keeps every earlier prior's config and SHA.
    part_valid_loss: bool = False
    # round 13 candidates against sentences that do not step through their text (opt-in, training only, text-conditioned
    # rows only). guided_attention: weight of Tachibana et al.'s (2017, arXiv 1710.08969) guided-attention loss on the
    # temporal decoder's cross-attention (see guided_attention_loss), width g = guided_attention_width.
    # alignment_supervision: weight of a cross entropy pulling the layer-averaged cross-attention of each step toward the
    # bytes of the word being signed there, on rows that carry per-frame word spans (see alignment_loss).
    # 0 keeps every earlier prior's config and SHA (the width is dropped from the dict while the weight is 0).
    guided_attention: float = 0.
    guided_attention_width: float = .2
    alignment_supervision: float = 0.

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tokenizer"] = self.tokenizer.to_dict()
        for key in ("text_model", "text_model_revision"):
            if data.get(key) is None:
                data.pop(key, None)
        if not data.get("guided_attention"):
            data.pop("guided_attention_width", None)
        for key in ("input_corruption", "scheduled_sampling", "history_dropout", "history_stages", "part_valid_loss",
                    "guided_attention", "alignment_supervision"):
            if not data.get(key):
                data.pop(key, None)
        return data

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "RVQPriorConfig":
        data = dict(data)
        data["tokenizer"] = MotionRVQConfig.from_dict(data["tokenizer"])
        return cls(**data)


def load_text_model(name: str, revision: str | None = None):
    """(frozen model, tokenizer) of a Hugging Face text encoder; T5-style models are used through their encoder."""
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if "t5" in name.lower():
        from transformers import T5EncoderModel
        model = T5EncoderModel.from_pretrained(name, revision=revision)
    else:
        model = AutoModel.from_pretrained(name, revision=revision)
    return model, tokenizer


def cross_attention_probabilities(attention: nn.MultiheadAttention, query: torch.Tensor, memory: torch.Tensor,
                                  key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    """(B, S, L) head-averaged softmax weights of a batch-first ``attention`` (packed in-projection) from ``query``
    (B, S, d) to ``memory`` (B, L, d), without attention dropout: what ``attention(query, memory, memory,
    need_weights=True, average_attn_weights=True)`` returns in eval mode (tested). Padded keys get weight 0."""
    d, H = attention.embed_dim, attention.num_heads
    w, b = attention.in_proj_weight, attention.in_proj_bias
    q = F.linear(query, w[:d], None if b is None else b[:d])
    k = F.linear(memory, w[d:2 * d], None if b is None else b[d:2 * d])
    B, S, L = q.shape[0], q.shape[1], k.shape[1]
    q = q.reshape(B, S, H, d // H).transpose(1, 2)
    k = k.reshape(B, L, H, d // H).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / math.sqrt(d // H)                                  # (B, H, S, L)
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask[:, None, None], -math.inf)
    return scores.softmax(-1).mean(1)


class AttentionRecordingDecoderLayer(nn.TransformerDecoderLayer):
    """``nn.TransformerDecoderLayer`` that, while ``record`` is set, also stores its cross-attention weights
    (B, S, L) in ``cross_attention``. The layer's output is computed by the parent's unchanged code path (the
    weights are a side computation from the same normalised input, query and key projections), so outputs are
    bit-identical with or without recording, and the submodules and state_dict keys are the parent's: every
    earlier checkpoint loads unchanged."""

    record = False
    cross_attention: torch.Tensor | None = None

    def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal=False):
        if self.record:
            if attn_mask is not None:
                raise ValueError("attention recording supports key padding masks only")
            self.cross_attention = cross_attention_probabilities(self.multihead_attn, x, mem, key_padding_mask)
        return super()._mha_block(x, mem, attn_mask, key_padding_mask, is_causal)


def _ranks(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(position index among the row's valid entries, count of valid entries per row) of a (B, L) mask."""
    mask = mask.bool()
    return (mask.long().cumsum(1) - 1).clamp_min(0), mask.sum(1)


def guided_attention_weights(step_valid: torch.Tensor, text_valid: torch.Tensor, width: float = .2) -> torch.Tensor:
    """(B, S, L) penalty W[t, n] = 1 − exp(−(n/N − t/T)² / (2 g²)) of Tachibana et al. (2017), with t, n the ranks of
    step t and text position n among the row's valid steps and valid text positions (T and N of them); 0 elsewhere.
    All non-PAD memory positions count, BOS and EOS included: BOS (n = 0) anchors the start of the diagonal and EOS
    (n = N − 1) its end, so a prior that steps through the text attends BOS at the first step and EOS at the last."""
    t, T = _ranks(step_valid)
    n, N = _ranks(text_valid)
    x = t.float() / T.clamp_min(1)[:, None].float()                                         # (B, S)
    y = n.float() / N.clamp_min(1)[:, None].float()                                         # (B, L)
    W = 1 - torch.exp(-(y[:, None, :] - x[:, :, None]) ** 2 / (2 * width ** 2))
    return W * (step_valid.bool()[:, :, None] & text_valid.bool()[:, None, :])


def guided_attention_loss(attention: torch.Tensor, step_valid: torch.Tensor, text_valid: torch.Tensor,
                          width: float = .2) -> torch.Tensor:
    """Mean of A[t, n]·W[t, n] over the valid (step, text position) pairs of every row and over layers (Tachibana
    et al.'s 1/(NT) normalisation, pooled over the batch). ``attention``: (B, S, L) or (B, layers, S, L)."""
    if attention.dim() == 3:
        attention = attention[:, None]
    W = guided_attention_weights(step_valid, text_valid, width).to(attention.dtype)        # (B, S, L)
    pairs = (step_valid.sum(1) * text_valid.sum(1)).sum().clamp_min(1) * attention.shape[1]
    return (attention * W[:, None]).sum() / pairs


def alignment_targets(word_spans: torch.Tensor, text_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(target (B, S, L) distribution, has-target (B, S) mask) from per-step word spans (B, S, 2) of text positions
    [start, end) in the encoded (BOS-first) memory; rows/steps with start < 0 or an empty span have no target.
    The target is uniform over the span's positions."""
    start, end = word_spans[..., 0], word_spans[..., 1]
    has = (start >= 0) & (end > start)
    position = torch.arange(text_length, device=word_spans.device)
    target = ((position >= start[..., None]) & (position < end[..., None]) & has[..., None]).float()
    has = has & (target.sum(-1) > 0)                                       # spans cut off by truncation have none
    return target / target.sum(-1, keepdim=True).clamp_min(1), has


def alignment_loss(attention: torch.Tensor, word_spans: torch.Tensor, step_valid: torch.Tensor | None = None,
                   eps: float = 1e-6) -> torch.Tensor | None:
    """Cross entropy −Σ_n q[t, n] log Ā[t, n] of the layer-averaged cross-attention Ā (``attention`` (B, S, L) or
    (B, layers, S, L)) against q = uniform over the bytes of the word signed at step t (``word_spans`` (B, S, 2), see
    ``alignment_targets``), averaged over the steps that have a span (and are valid). Equals KL(q ‖ Ā) + H(q).
    None when no step has a span (the caller adds nothing)."""
    if attention.dim() == 4:
        attention = attention.mean(1)
    target, has = alignment_targets(word_spans, attention.shape[-1])
    if step_valid is not None:
        has = has & step_valid.bool()
    if not bool(has.any()):
        return None
    per_step = -(target * (attention.clamp_min(0) + eps).log()).sum(-1)                   # (B, S)
    return per_step[has].mean()


def word_byte_spans(text: str, *, max_bytes: int = 256) -> list[tuple[int, int]]:
    """[start, end) memory positions (BOS = position 0, as ``encode_texts``) of each whitespace-separated word of
    ``text``, in the NFC-normalised, stripped UTF-8 bytes ``encode_texts`` encodes; words past the ``max_bytes``
    truncation are cut back or (if wholly truncated) given as empty spans."""
    data = unicodedata.normalize("NFC", text.strip()).encode("utf-8")
    limit = max_bytes - 2
    spans, offset = [], 0
    for word in data.split():
        start = data.index(word, offset)
        offset = start + len(word)
        spans.append((min(start, limit) + 1, min(offset, limit) + 1))
    return spans


def _word_key(word: str) -> str | None:
    """Comparable base gloss of a text word: plan-line loci (``1>GIVE>2``) and annotation decoration removed;
    None for plan-line marker brackets (``[yn``, ``]``), which no sign spans."""
    from ..planning.asllrp_gloss import parse_surface_token
    if word.startswith("[") or word == "]":
        return None
    word = re.sub(r"^\d+>", "", re.sub(r">\d+$", "", word))
    return parse_surface_token(word).base


def frame_word_spans(text: str, signs, timestamps, *, max_bytes: int = 256):
    """(T, 2) int64 per-frame [start, end) memory positions of the text word being signed (−1, −1 where none).

    ``signs``: the utterance's time-segmented signs in order, as (gloss label, start seconds, end seconds) relative
    to the state's first frame (end inclusive of its last frame); ``timestamps``: the state's frame times (T,), after
    any resampling. Text words (whitespace tokens of the text ``encode_texts`` encodes: a gloss line or a plan line)
    are matched to signs by an order-preserving alignment of their base glosses (``difflib.SequenceMatcher``), so a
    plan line whose surface-tier glosses differ from the segmented signs keeps only the agreeing ones; a frame inside
    no matched sign has no span. See docs/research/alignment-supervision-design.md."""
    import difflib
    import numpy as np
    from ..planning.asllrp_gloss import parse_surface_token
    times = np.asarray(timestamps, dtype=np.float64)
    times = times - times[0] if len(times) else times
    out = np.full((len(times), 2), -1, dtype=np.int64)
    words = [w.decode("utf-8", errors="ignore") for w in unicodedata.normalize("NFC", text.strip()).encode().split()]
    spans = word_byte_spans(text, max_bytes=max_bytes)
    keys = [_word_key(w) for w in words]
    word_index = [i for i, k in enumerate(keys) if k is not None]
    sign_keys = [parse_surface_token(label).base for label, _, _ in signs]
    matcher = difflib.SequenceMatcher(a=[keys[i] for i in word_index], b=sign_keys, autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            start, end = spans[word_index[block.a + k]]
            if end <= start:
                continue
            _, t0, t1 = signs[block.b + k]
            inside = (times >= float(t0) - 1e-6) & (times <= float(t1) + 1e-6)
            out[inside] = (start, end)
    return out


class RVQPrior(nn.Module):
    def __init__(self, config: RVQPriorConfig) -> None:
        super().__init__()
        self.config = config
        d, J, K = config.d_model, config.tokenizer.codes_per_step, config.tokenizer.codebook
        self.J, self.K = J, K
        self.embed = nn.Embedding(VOCABULARY, d, padding_idx=PAD)
        if config.grammar_features:
            self.grammar = grammar_projection(d)
        if config.locus_conditioning:
            self.loci = locus_projection(d)
        layer = lambda: nn.TransformerEncoderLayer(d, config.heads, config.feedforward, config.dropout,
                                                   batch_first=True, norm_first=True)
        self.text_encoder = nn.TransformerEncoder(layer(), config.text_layers, enable_nested_tensor=False)
        if config.text_model:
            # frozen and kept outside the module tree: not in state_dict, the optimizer or the weight EMA
            lm, tok = load_text_model(config.text_model, config.text_model_revision)
            object.__setattr__(self, "_lm", lm.eval().requires_grad_(False))
            object.__setattr__(self, "_lm_tokenizer", tok)
            self.text_in = nn.Linear(lm.config.hidden_size if hasattr(lm.config, "hidden_size") else lm.config.d_model, d)
        self.duration = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # code j of a step has its own embedding table (index K = "start" for the first step)
        self.code_embed = nn.Parameter(torch.randn(J, K + 1, d) * .02)
        self.time = nn.Linear(2 * d, d)
        self.temporal = nn.TransformerDecoder(AttentionRecordingDecoderLayer(
            d, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True),
            config.temporal_layers)
        self.depth_position = nn.Parameter(torch.randn(J, d) * .02)
        self.depth = nn.TransformerEncoder(layer(), config.depth_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.heads_out = nn.Parameter(torch.randn(J, d, K) * .02)
        self.heads_bias = nn.Parameter(torch.zeros(J, K))

    def encode(self, tokens, mask, loci=None):
        """(hidden (B, L, d), log duration (B,), memory mask (B, L)) of byte-encoded texts. With a pretrained text model
        the memory is its tokenisation of the same texts, so its length and mask differ from the byte input's."""
        if self.config.text_model:
            inputs, mask = self._pretrained_inputs(tokens)
        else:
            positions = sinusoid(torch.arange(tokens.shape[1], device=tokens.device)[None].expand_as(tokens),
                                 self.config.d_model, max_period=1024.)
            inputs = self.embed(tokens) + positions
        if self.config.grammar_features:
            inputs = inputs + self.grammar(token_features(tokens).to(inputs.dtype))[:, None]
        if self.config.locus_conditioning and loci is not None:
            inputs = inputs + self.loci(loci.to(inputs.dtype))[:, None]
        hidden = self.text_encoder(inputs, src_key_padding_mask=~mask)
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return hidden, self.duration(pooled)[:, 0], mask

    def _pretrained_inputs(self, tokens):
        """The frozen text model's last hidden states of the byte rows' texts, projected and position-coded."""
        texts = [bytes(int(b) for b in row if int(b) < 256).decode("utf-8", errors="ignore") for row in tokens.cpu()]
        batch = self._lm_tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        device = tokens.device
        if next(self._lm.parameters()).device != device:
            object.__setattr__(self, "_lm", self._lm.to(device))
        with torch.no_grad():
            lm = self._lm.get_encoder() if hasattr(self._lm, "get_encoder") else self._lm
            states = lm(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device)).last_hidden_state
        mask = batch["attention_mask"].to(device).bool()
        positions = sinusoid(torch.arange(states.shape[1], device=device)[None].expand(states.shape[0], -1),
                             self.config.d_model, max_period=1024.)
        return self.text_in(states.to(self.text_in.weight.dtype)) + positions, mask

    def _embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """(B, S, J) indices (K = start) -> (B, S, J, d) per-position embeddings."""
        j = torch.arange(self.J, device=codes.device)
        return self.code_embed[j, codes]

    def temporal_states(self, hidden, text_mask, previous: torch.Tensor, seconds, duration, *,
                        return_attention: bool = False):
        """h_t for every step from the previous steps' codes ``previous`` (B, S, J) — start row first. With
        ``return_attention``, also every temporal layer's head-averaged cross-attention to the text (B, layers, S, L);
        h is the same either way."""
        d = self.config.d_model
        phase = seconds / duration[:, None].clamp_min(1e-3)
        embedded = self._embed_codes(previous)                                 # (B, S, J, d); code j = q·P + p
        if self.config.history_stages:
            embedded = embedded[:, :, :self.config.history_stages * (self.J // self.config.tokenizer.stages)]
        x = embedded.sum(2) + self.time(torch.cat(
            (sinusoid(seconds, d, max_period=60.), sinusoid(phase, d, max_period=4.)), -1))
        causal = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), 1)
        if not return_attention:
            return self.temporal(x, hidden, tgt_mask=causal, memory_key_padding_mask=~text_mask)
        layers = list(self.temporal.layers)
        for layer in layers:
            layer.record = True
        try:
            h = self.temporal(x, hidden, tgt_mask=causal, memory_key_padding_mask=~text_mask)
            attention = torch.stack([layer.cross_attention for layer in layers], 1)
        finally:
            for layer in layers:
                layer.record, layer.cross_attention = False, None
        return h, attention

    def depth_logits(self, h: torch.Tensor, codes_so_far: torch.Tensor) -> torch.Tensor:
        """Logits (N, n+1, K) of code positions 0..n given h (N, d) and the step's first n codes (N, n)."""
        n = codes_so_far.shape[1]
        j = torch.arange(n, device=h.device)
        inputs = [h[:, None]]
        if n:
            inputs.append(self.code_embed[j, codes_so_far])
        x = torch.cat(inputs, 1) + self.depth_position[:n + 1]
        causal = torch.triu(torch.ones(n + 1, n + 1, dtype=torch.bool, device=h.device), 1)
        y = self.norm(self.depth(x, mask=causal))
        return torch.einsum("nld,ldk->nlk", y, self.heads_out[:n + 1]) + self.heads_bias[:n + 1]


@torch.no_grad()
def part_code_weights(tokenizer: MotionRVQ, config: RVQPriorConfig, motion: MotionBatch, valid: torch.Tensor):
    """(N, J) 0/1 weights of the valid tokens' codes: 1 where the code's part was observed (at least half of the part's
    unit columns valid in the token's frames). Code j belongs to part j mod P."""
    observed = diffusion_valid(config.tokenizer.motion, motion)                            # (B, T, W)
    part_ok = torch.stack([tokenizer.token_valid(observed[..., getattr(tokenizer, f"columns_{part}")].float().mean(-1)
                                                 >= .5) for part, _ in tokenizer.parts], -1)  # (B, S, P)
    J = config.tokenizer.codes_per_step
    return part_ok[valid][:, torch.arange(J, device=part_ok.device) % len(tokenizer.parts)].float()


def rvq_prior_loss(model: RVQPrior, tokenizer: MotionRVQ, tokens, text_mask, motion: MotionBatch, loci=None, *,
                   train: bool = True, word_spans: torch.Tensor | None = None):
    """Σ_j cross entropy of the frozen tokenizer's codes, plus the duration term (text dropped only if ``train``).

    Opt-in attention terms (training only, text-conditioned rows only): ``guided_attention`` adds Tachibana et al.'s
    guided-attention loss on the temporal decoder's cross-attention; ``alignment_supervision`` adds
    ``alignment_loss`` against ``word_spans`` (B, T, 2): per *frame*, the [start, end) memory positions of the word
    signed there (start < 0: none; see ``frame_word_spans``). Each step takes its first frame's span. Rows without
    spans contribute nothing."""
    config = model.config
    with torch.no_grad():
        tokenizer.eval()
        unit = diffusion_targets(config.tokenizer.motion, motion)
        _, codes, _ = tokenizer.quantize(tokenizer.encode(unit))                     # (B, S, J)
        valid = tokenizer.token_valid(motion.frame_valid)
    conditioned = (torch.rand(len(tokens), device=tokens.device) >= config.cfg_dropout) if train else \
        torch.ones(len(tokens), dtype=torch.bool, device=tokens.device)
    tokens = torch.where(conditioned[:, None], tokens, _empty_like(tokens))
    text_mask = tokens != PAD
    hidden, log_duration, text_mask = model.encode(tokens, text_mask,
                                                   training_loci(config, loci, conditioned) if train else loci)
    start = torch.full_like(codes[:, :1], model.K)
    previous = torch.cat((start, codes[:, :-1]), 1)
    seconds = batch_seconds(motion).float()
    step_seconds = seconds[:, ::config.tokenizer.step][:, :codes.shape[1]]
    step_seconds = F.pad(step_seconds, (0, codes.shape[1] - step_seconds.shape[1]))
    duration = (seconds * motion.frame_valid).amax(1)
    if train and config.scheduled_sampling > 0:        # round 8 candidate: histories drawn from the model itself
        with torch.no_grad():
            h0 = model.temporal_states(hidden, text_mask, previous, step_seconds, duration)
            logits0 = model.depth_logits(h0.reshape(-1, h0.shape[-1]), codes.reshape(-1, model.J)[:, :-1])
            drawn = torch.multinomial(torch.softmax(logits0.float(), -1).reshape(-1, model.K), 1).reshape(codes.shape)
        own = torch.cat((start, drawn[:, :-1]), 1)
        rate = torch.rand(len(previous), 1, device=previous.device) * config.scheduled_sampling
        hit = torch.rand(previous.shape[:2], device=previous.device) < rate
        hit[:, 0] = False                               # the start token stays
        previous = torch.where(hit[..., None], own, previous)
    if train and config.history_dropout > 0:           # round 9 candidate: whole history spans hidden
        drop = torch.rand(len(previous), device=previous.device) < config.history_dropout
        cut = (torch.rand(len(previous), device=previous.device) * previous.shape[1]).long() + 1     # 1..S
        hidden_steps = (torch.arange(previous.shape[1], device=previous.device)[None] < cut[:, None]) & drop[:, None]
        previous = torch.where(hidden_steps[..., None], torch.full_like(previous, model.K), previous)
    if train and config.input_corruption > 0:          # round 7: corrupt the conditioning history, not the targets
        rate = torch.rand(len(previous), 1, device=previous.device) * config.input_corruption
        hit = torch.rand(previous.shape[:2], device=previous.device) < rate
        hit[:, 0] = False                               # the start token stays
        noise = torch.randint_like(previous, model.K)
        previous = torch.where(hit[..., None], noise, previous)
    supervise = train and bool(conditioned.any()) and (config.guided_attention > 0 or (
        config.alignment_supervision > 0 and word_spans is not None))
    if supervise and config.text_model and config.alignment_supervision > 0 and word_spans is not None:
        raise ValueError("alignment supervision needs the byte text encoder (word spans are byte positions)")
    if supervise:
        h, attention = model.temporal_states(hidden, text_mask, previous, step_seconds, duration,
                                             return_attention=True)                    # (B, layers, S, L)
    else:
        h = model.temporal_states(hidden, text_mask, previous, step_seconds, duration)   # (B, S, d)
    target = codes[valid]                                                                  # (N, J)
    logits = model.depth_logits(h[valid], target[:, :-1])                                  # (N, J, K)
    if config.part_valid_loss:
        weight = part_code_weights(tokenizer, config, motion, valid)
        per_code = F.cross_entropy(logits.reshape(-1, model.K), target.reshape(-1), reduction="none")
        cross = (per_code * weight.reshape(-1)).sum() / weight.sum().clamp_min(1.)
    else:
        cross = F.cross_entropy(logits.reshape(-1, model.K), target.reshape(-1))
    accuracy = (logits.argmax(-1) == target).float()
    terms = {"tokens": float(cross.detach()), "token_accuracy": float(accuracy.mean()),
             "stage1_accuracy": float(accuracy[:, :len(tokenizer.parts)].mean())}
    total = cross
    if bool(conditioned.any()):
        target_duration = utterance_duration(motion).log().to(log_duration.dtype)
        duration_term = F.smooth_l1_loss(log_duration[conditioned], target_duration[conditioned])
        terms["log_duration"] = float(duration_term.detach())
        total = total + config.duration_weight * duration_term
    if supervise and config.guided_attention > 0:
        rows = conditioned
        guided = guided_attention_loss(attention[rows], valid[rows], text_mask[rows], config.guided_attention_width)
        terms["guided_attention"] = float(guided.detach())
        total = total + config.guided_attention * guided
    if supervise and config.alignment_supervision > 0 and word_spans is not None:
        spans = word_spans.to(codes.device).long()[:, ::config.tokenizer.step][:, :codes.shape[1]]
        spans = F.pad(spans, (0, 0, 0, codes.shape[1] - spans.shape[1]), value=-1)        # (B, S, 2)
        rows = conditioned
        aligned = alignment_loss(attention[rows], spans[rows], valid[rows])
        if aligned is not None:
            terms["alignment"] = float(aligned.detach())
            total = total + config.alignment_supervision * aligned
    return total, terms


@torch.no_grad()
def rvq_sequence_log_likelihood(model: RVQPrior, tokenizer: MotionRVQ, tokens, text_mask, motion: MotionBatch,
                                loci=None) -> torch.Tensor:
    """(B,) summed log-probability of each row's valid tokenizer codes under teacher forcing: ``rvq_prior_loss``'s
    evaluation path (no text dropout, no corruption), not averaged. Word retrieval scores one clip's codes under many
    candidate texts with it (round 7, amendment 1)."""
    config = model.config
    with torch.no_grad():
        tokenizer.eval()
        unit = diffusion_targets(config.tokenizer.motion, motion)
        _, codes, _ = tokenizer.quantize(tokenizer.encode(unit))                     # (B, S, J)
        valid = tokenizer.token_valid(motion.frame_valid)
        hidden, _, text_mask = model.encode(tokens, text_mask, loci)
        start = torch.full_like(codes[:, :1], model.K)
        previous = torch.cat((start, codes[:, :-1]), 1)
        seconds = batch_seconds(motion).float()
        step_seconds = seconds[:, ::config.tokenizer.step][:, :codes.shape[1]]
        step_seconds = F.pad(step_seconds, (0, codes.shape[1] - step_seconds.shape[1]))
        duration = (seconds * motion.frame_valid).amax(1)
        h = model.temporal_states(hidden, text_mask, previous, step_seconds, duration)
        target = codes[valid]                                                              # (N, J)
        logits = model.depth_logits(h[valid], target[:, :-1])                              # (N, J, K)
        log_p = logits.log_softmax(-1).gather(-1, target[..., None])[..., 0].sum(1)        # (N,)
        row = torch.arange(len(codes), device=codes.device)[:, None].expand_as(valid)[valid]
        return torch.zeros(len(codes), dtype=log_p.dtype, device=log_p.device).index_add_(0, row, log_p)


@torch.no_grad()
def generate_rvq_motion(prior: RVQPrior, tokenizer: MotionRVQ, texts, *, fps: float, guidance: float = 1.,
                        temperature: float | dict = 1., top_k: int = 0, seed: int = 0, durations=None,
                        sample_prefix: str = "tokens-v2", loci=None):
    """Generated canonical states (observed=False, zero confidence) from sampled residual codes.

    ``temperature`` is one value or a mapping part -> value (parts of ``tokenizer.parts``;
    missing parts sample at 1): code position j = q·P + p belongs to part p. Per-part
    temperatures let the hands sample more freely than the face, whose head-rotation codes
    carry the headshakes (v2b + loci at T = 1.2: hands within the articulation gate, headshake
    rate 3.0× the reference).
    """
    prior.eval(); tokenizer.eval()
    config = prior.config
    device = next(prior.parameters()).device
    encoded, mask = encode_texts(texts, max_bytes=config.max_text_bytes)
    encoded, mask = encoded.to(device), mask.to(device)
    hidden, log_duration, mask = prior.encode(encoded, mask, generation_loci(config, texts, loci, device))
    seconds_out = (log_duration.exp() if durations is None else
                   torch.as_tensor(durations, dtype=log_duration.dtype, device=device)).clamp(1 / fps, 60.)
    empty = _empty_like(encoded[:1, :2])
    empty_hidden, _, empty_mask = prior.encode(empty, empty != PAD, empty_loci(config, device))
    generator = torch.Generator().manual_seed(seed)
    names = [part for part, _ in tokenizer.parts]
    if isinstance(temperature, dict):
        unknown = set(temperature) - set(names)
        if unknown:
            raise ValueError(f"unknown parts in temperature: {sorted(unknown)}")
        per_code = [float(temperature.get(names[j % len(names)], 1.)) for j in range(prior.J)]
    else:
        per_code = [float(temperature)] * prior.J
    states = []
    for row, text in enumerate(texts):
        frames = max(int(round(float(seconds_out[row]) * fps)), 1)
        count = math.ceil(frames / config.tokenizer.step)
        clock = token_seconds(config, count, fps, device)
        sequence = torch.full((1, 1, prior.J), prior.K, dtype=torch.long, device=device)
        for k in range(count):
            h = prior.temporal_states(hidden[row:row + 1], mask[row:row + 1], sequence, clock[:, :k + 1],
                                      seconds_out[row:row + 1])[:, -1]
            h_free = prior.temporal_states(empty_hidden, empty_mask, sequence, clock[:, :k + 1],
                                           seconds_out[row:row + 1])[:, -1] if guidance != 1. else None
            step_codes = torch.zeros((1, 0), dtype=torch.long, device=device)
            for j in range(prior.J):
                logits = prior.depth_logits(h, step_codes)[:, -1]
                if h_free is not None:
                    free = prior.depth_logits(h_free, step_codes)[:, -1]
                    logits = free + guidance * (logits - free)
                logits = logits / max(per_code[j], 1e-6)
                if top_k:
                    threshold = logits.topk(top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, -math.inf)
                choice = torch.multinomial(torch.softmax(logits.float(), -1).cpu(), 1, generator=generator).to(device)
                step_codes = torch.cat((step_codes, choice), 1)
            sequence = torch.cat((sequence, step_codes[:, None]), 1)
        unit = tokenizer.decode_codes(sequence[:, 1:], frames)
        raw = unit_to_raw(config.tokenizer.motion, unit.double())
        native = native_values(config.tokenizer.motion, {key: v[0] for key, v in raw.items()})
        states.append(assemble_state(config.tokenizer.motion, text, native, frames, fps, f"{sample_prefix}:{row}",
                                     f"{sample_prefix}-clock", method="motion_tokens_v2"))
    return states


@torch.no_grad()
def generate_rvq_motion_batch(prior: RVQPrior, tokenizer: MotionRVQ, texts, *, fps: float, durations,
                              guidance: float = 1., temperature: float | dict = 1., top_k: int = 0, seed: int = 0,
                              sample_prefix: str = "tokens-v2", loci=None):
    """``generate_rvq_motion`` for many texts at once: every step and every depth position is one batched call
    instead of one per text (evaluation sets of hundreds of clips are otherwise bound by Python and leave the GPU
    idle). Same model, same per-row duration, clock, guidance, temperatures and top-k; rows shorter than the longest
    are generated to the common length and cut back to their own. With ``top_k=1`` it returns exactly the
    per-text function's states (tested); with sampling, the random draws follow a different order, so individual
    samples differ while their distribution does not."""
    prior.eval(); tokenizer.eval()
    config = prior.config
    device = next(prior.parameters()).device
    B = len(texts)
    encoded, mask = encode_texts(texts, max_bytes=config.max_text_bytes)
    encoded, mask = encoded.to(device), mask.to(device)
    hidden, _, mask = prior.encode(encoded, mask, generation_loci(config, texts, loci, device))
    seconds_out = torch.as_tensor(durations, dtype=hidden.dtype, device=device).clamp(1 / fps, 60.)
    empty = _empty_like(encoded[:1, :2])
    empty_hidden, _, empty_mask = prior.encode(empty, empty != PAD, empty_loci(config, device))
    empty_hidden, empty_mask = empty_hidden.expand(B, -1, -1), empty_mask.expand(B, -1)
    generator = torch.Generator().manual_seed(seed)
    names = [part for part, _ in tokenizer.parts]
    if isinstance(temperature, dict):
        unknown = set(temperature) - set(names)
        if unknown:
            raise ValueError(f"unknown parts in temperature: {sorted(unknown)}")
        per_code = [float(temperature.get(names[j % len(names)], 1.)) for j in range(prior.J)]
    else:
        per_code = [float(temperature)] * prior.J
    frames = [max(int(round(float(s) * fps)), 1) for s in seconds_out]
    counts = [math.ceil(f / config.tokenizer.step) for f in frames]
    K = max(counts)
    clock = token_seconds(config, K, fps, device).expand(B, -1)
    sequence = torch.full((B, 1, prior.J), prior.K, dtype=torch.long, device=device)
    for k in range(K):
        h = prior.temporal_states(hidden, mask, sequence, clock[:, :k + 1], seconds_out)[:, -1]
        h_free = (prior.temporal_states(empty_hidden, empty_mask, sequence, clock[:, :k + 1], seconds_out)[:, -1]
                  if guidance != 1. else None)
        step_codes = torch.zeros((B, 0), dtype=torch.long, device=device)
        for j in range(prior.J):
            logits = prior.depth_logits(h, step_codes)[:, -1]
            if h_free is not None:
                free = prior.depth_logits(h_free, step_codes)[:, -1]
                logits = free + guidance * (logits - free)
            logits = logits / max(per_code[j], 1e-6)
            if top_k:
                threshold = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -math.inf)
            choice = torch.multinomial(torch.softmax(logits.float(), -1).cpu(), 1, generator=generator).to(device)
            step_codes = torch.cat((step_codes, choice), 1)
        sequence = torch.cat((sequence, step_codes[:, None]), 1)
    states = []
    for row, text in enumerate(texts):
        unit = tokenizer.decode_codes(sequence[row:row + 1, 1:counts[row] + 1], frames[row])
        raw = unit_to_raw(config.tokenizer.motion, unit.double())
        native = native_values(config.tokenizer.motion, {key: v[0] for key, v in raw.items()})
        states.append(assemble_state(config.tokenizer.motion, text, native, frames[row], fps, f"{sample_prefix}:{row}",
                                     f"{sample_prefix}-clock", method="motion_tokens_v2"))
    return states


__all__ = ["PARTS", "MotionRVQConfig", "ResidualQuantizer", "MotionRVQ", "rvq_loss", "RVQPriorConfig", "RVQPrior",
           "rvq_prior_loss", "generate_rvq_motion", "generate_rvq_motion_batch", "AttentionRecordingDecoderLayer",
           "cross_attention_probabilities", "guided_attention_weights", "guided_attention_loss", "alignment_targets",
           "alignment_loss", "word_byte_spans", "frame_word_spans"]
