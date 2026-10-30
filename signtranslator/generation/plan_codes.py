"""Round 17: coarse plan codes over the 24 Hz tokenizer (docs/NEXT_PHASE_PLAN.md, "Round 17 design").

**Why.** End-to-end generation reads at blank level while one-step prediction is good (rounds 7–16): the
autoregressive prior at 24 steps/s × 32 codes cannot start or sustain a sentence. Coarser tokenizers failed on
fidelity (t2x/t4x: hand speed 0.45 of real). LMR-style, the faithful d2 tokenizer and its decoder stay, and a short
*plan* sequence, generated first, carries the sentence-level structure:

    p(c, π | text) = p(π | text) · p(c | π, text),

where π is the plan (one plan step per W tokenizer steps, P·Q_π codes each) and c the d2 codes.

**Plan tokenizer** (:class:`PlanVQ`). Input: the frozen d2 tokenizer's *continuous* per-part latents z_t (before
quantisation; (B, S, D) per part, S tokenizer steps). Per part, a small convolutional encoder at step rate, a pooling
of each window of W steps (masked mean, or a learned masked softmax over the window's steps) and residual VQ
(``ResidualQuantizer``: EMA codebooks, dead-code re-seeding, Q_π stages, straight-through estimator, commitment).
Plan step i covers steps [i·W, (i+1)·W); the last window of a ragged clip holds its remaining valid steps and plan
step i is valid iff its window holds a valid step (valid steps are a prefix, so plan steps are too). One shared
decoder (as d2's) maps every part's plan latents back to every part's per-step latents: convolutions at plan rate,
nearest upsampling by W plus a learned within-window position, convolutions at step rate. Invalid steps and plan
steps are zeroed before every convolution, so a clip's codes do not depend on the batch's padding (tested).

**Loss.** latent_weight · relative latent error (per part Σ‖ẑ − z‖² / Σ‖z − z̄‖² over valid steps: 1 = predicting
each part's mean latent, 0 = exact) + motion_weight · the d2 decoder's reconstruction loss of the motion decoded from
ẑ (the same per-channel and skeleton terms as ``rvq_loss``; the frozen decoder passes the gradient) + commitment.
The motion term targets the fidelity gate (hand speed through d2's decoder); latent error alone regresses each
window to its mean and damps motion.

**Codes** are stage-major as d2's: plan code j = q·P + p belongs to part p, stage q.

**Plan prior** (:class:`PlanPriorConfig`): an ``RVQPrior`` whose "tokenizer" is the plan tokenizer's config
(codes_per_step = P·Q_π, codebook, stages, step = W·d2.step frames), so the temporal transformer runs over plan steps
and the depth transformer over the plan codes of a step; text encoder warm-started from W14b (``copy_text_encoder``).

**Detail prior**: ``RVQPrior`` with ``plan_window`` set (motion_rvq.py): plan embeddings added at step rate and a
plan cross-attention reader after every temporal layer, both zero-initialised, so a warm start from a plan-less
prior reproduces it exactly at step 0 (tested); plan dropout in training.

**Generation** (:func:`generate_plan_detail_batch`): plan prior → plan codes → detail prior given the plan → d2
decoder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..data.multichannel import MotionBatch
from .canonical_state import channel_loss_per_example
from .motion_rvq import (
    MotionRVQ, MotionRVQConfig, RVQPrior, RVQPriorConfig, ResidualQuantizer, padded_plan, part_temperatures,
    rvq_prior_loss, sample_code_sequence,
)
from .motion_tokens import _ResBlock, _empty_like, token_seconds
from .signing_space import applicable, skeleton_motion_loss
from .text_to_motion import (
    PAD, LossScales, assemble_state, diffusion_targets, diffusion_valid, empty_loci, encode_texts, generation_loci,
    native_values, unit_to_raw,
)

PLAN_KIND = "plan_vq_tokenizer"
PLAN_PRIOR_KIND = "motion_plan_prior"
POOLINGS = ("mean", "learned")
PLAN_SEED_OFFSET = 104_729          # the plan's sampler draws from seed + this, the detail's from seed


# --------------------------------------------------------------------------------------
# Windows


def plan_steps(steps: int, window: int) -> int:
    """Plan steps covering ``steps`` tokenizer steps: ceil(steps / W)."""
    return -(-steps // window)


def window_valid(valid: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """((B, S_plan, W) step-valid mask per window, (B, S_plan) plan-step valid) of a (B, S) step mask; plan step i
    covers steps [i·W, (i+1)·W) and is valid iff one of them is."""
    B, S = valid.shape
    n = plan_steps(S, window)
    inside = F.pad(valid.bool(), (0, n * window - S)).reshape(B, n, window)
    return inside, inside.any(-1)


def pool_windows(x: torch.Tensor, valid: torch.Tensor, window: int, scores: torch.Tensor | None = None):
    """((B, S_plan, C) pooled windows, (B, S_plan) plan valid) of per-step features ``x`` (B, S, C): the masked mean
    of each window's valid steps, or with ``scores`` (B, S) their softmax-weighted sum over the window's valid steps.
    Windows without a valid step pool to 0."""
    B, S, C = x.shape
    inside, plan_valid = window_valid(valid, window)
    n = inside.shape[1]
    grouped = F.pad(x, (0, 0, 0, n * window - S)).reshape(B, n, window, C)
    if scores is None:
        weight = inside.to(x.dtype)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1.)
    else:
        s = F.pad(scores, (0, n * window - S)).reshape(B, n, window).masked_fill(~inside, -math.inf)
        weight = torch.where(plan_valid[..., None], s.softmax(-1), torch.zeros_like(s)).to(x.dtype)
    return (grouped * weight[..., None]).sum(2), plan_valid


def upsample_windows(y: torch.Tensor, window: int, steps: int) -> torch.Tensor:
    """(B, steps, C) per-step features from plan-step features (B, S_plan, C): step t takes plan step t // W. The plan
    decoder upsamples with it; the detail prior's plan input uses the same rule as an index (t // W)."""
    if y.shape[1] * window < steps:
        raise ValueError(f"{y.shape[1]} plan steps of {window} cannot cover {steps} steps")
    return y.repeat_interleave(window, dim=1)[:, :steps]


# --------------------------------------------------------------------------------------
# Plan tokenizer


@dataclass(frozen=True)
class PlanVQConfig:
    tokenizer: MotionRVQConfig          # the frozen detail tokenizer (d2): parts, latent width, step
    window: int = 4                     # W tokenizer steps per plan step (d2: 24 Hz -> 6 Hz)
    hidden: int = 256
    code_dim: int = 64
    codebook: int = 512
    stages: int = 2
    blocks: int = 2                     # residual blocks at each rate, in the encoders and the decoder
    pooling: str = "mean"
    commitment: float = .25
    ema: float = .99
    quantizer_dropout: float = 0.
    latent_weight: float = 1.
    motion_weight: float = 1.

    def __post_init__(self) -> None:
        if self.window < 1 or self.stages < 1 or self.codebook < 2 or self.blocks < 0:
            raise ValueError("a plan tokenizer needs window >= 1, stages >= 1, codebook >= 2 and blocks >= 0")
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        if self.latent_weight < 0 or self.motion_weight < 0 or self.latent_weight + self.motion_weight <= 0:
            raise ValueError("the reconstruction weights must be non-negative and not both 0")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tokenizer"] = self.tokenizer.to_dict()
        return data

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "PlanVQConfig":
        data = dict(data)
        data["tokenizer"] = MotionRVQConfig.from_dict(data["tokenizer"])
        return cls(**data)

    # the interface RVQPrior and rvq_prior_loss read from a prior's ``tokenizer`` config (the plan prior's)
    @property
    def motion(self):
        return self.tokenizer.motion

    @property
    def step(self) -> int:
        """Frames per plan step."""
        return self.window * self.tokenizer.step

    @property
    def part_names(self) -> list[str]:
        return [part for part, _ in self.tokenizer.part_slices()]

    @property
    def codes_per_step(self) -> int:
        return len(self.part_names) * self.stages


class PlanVQ(nn.Module):
    """Per-part window encoders and residual VQ over the frozen tokenizer's latents; a shared upsampling decoder."""

    def __init__(self, config: PlanVQConfig) -> None:
        super().__init__()
        self.config = config
        self.parts = config.part_names
        D, h, W = config.tokenizer.code_dim, config.hidden, config.window
        blocks = lambda: nn.Sequential(*(_ResBlock(h) for _ in range(config.blocks)))
        self.encoder_in = nn.ModuleDict({p: nn.Conv1d(D, h, 3, padding=1) for p in self.parts})
        self.encoder_steps = nn.ModuleDict({p: blocks() for p in self.parts})
        if config.pooling == "learned":
            self.pool_score = nn.ModuleDict({p: nn.Conv1d(h, 1, 1) for p in self.parts})
        self.encoder_plan = nn.ModuleDict({p: blocks() for p in self.parts})
        self.encoder_out = nn.ModuleDict({p: nn.Conv1d(h, config.code_dim, 1) for p in self.parts})
        self.quantizers = nn.ModuleDict({p: ResidualQuantizer(config.stages, config.codebook, config.code_dim,
                                                              config.ema) for p in self.parts})
        P = len(self.parts)
        self.decoder_in = nn.Conv1d(P * config.code_dim, h, 3, padding=1)
        self.decoder_plan = blocks()
        self.window_position = nn.Parameter(torch.zeros(W, h))
        self.decoder_steps = blocks()
        self.decoder_out = nn.Sequential(nn.SiLU(), nn.Conv1d(h, P * D, 1))

    @staticmethod
    def _conv(module, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``module`` (one convolution or one residual block) over (B, L, C) with invalid positions zeroed before it,
        so its outputs at valid positions do not depend on the padding."""
        x = torch.where(valid[..., None], x, torch.zeros_like(x))
        return module(x.transpose(1, 2)).transpose(1, 2)

    @classmethod
    def _stack(cls, blocks: nn.Sequential, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """A stack of residual blocks, re-masking before every block: a block's outputs at invalid positions are
        not zero, so masking only the stack's input lets the padding reach valid positions from the second block on
        (review of c37036b: max |Δz| = 16 at blocks = 2)."""
        for block in blocks:
            x = cls._conv(block, x, valid)
        return x

    def encode(self, latents: dict, valid: torch.Tensor) -> tuple[dict, torch.Tensor]:
        """(part -> (B, S_plan, code_dim) continuous plan latents, (B, S_plan) plan valid) of the frozen tokenizer's
        per-part latents (B, S, D) and its (B, S) step-valid mask."""
        out, plan_valid = {}, None
        for p in self.parts:
            x = self._conv(self.encoder_in[p], latents[p], valid)
            x = self._stack(self.encoder_steps[p], x, valid)
            scores = self._conv(self.pool_score[p], x, valid)[..., 0] if self.config.pooling == "learned" else None
            pooled, plan_valid = pool_windows(x, valid, self.config.window, scores)
            pooled = self._stack(self.encoder_plan[p], pooled, plan_valid)
            out[p] = self._conv(self.encoder_out[p], pooled, plan_valid)
        return out, plan_valid

    def quantize(self, plan_latents: dict, stages: int | None = None):
        """(part -> ẑ_π, (B, S_plan, P·Q) stage-major codes, part -> residual inputs per used stage)."""
        quantized, indices, inputs = {}, [], {}
        for p in self.parts:
            quantized[p], idx, inputs[p] = self.quantizers[p](plan_latents[p], stages)
            indices.append(idx)                                                  # (B, S_plan, Q)
        grid = torch.stack(indices, -1)                                           # (B, S_plan, Q, P)
        return quantized, grid.reshape(*grid.shape[:2], -1), inputs               # j = q·P + p

    def split_codes(self, codes: torch.Tensor) -> dict:
        """(B, S_plan, P·Q) stage-major codes -> part -> (B, S_plan, Q)."""
        grid = codes.reshape(*codes.shape[:-1], self.config.stages, len(self.parts))
        return {p: grid[..., k] for k, p in enumerate(self.parts)}

    def lookup(self, codes: torch.Tensor) -> dict:
        per_part = self.split_codes(codes)
        return {p: self.quantizers[p].lookup(per_part[p]) for p in self.parts}

    def decode(self, quantized: dict, plan_valid: torch.Tensor, steps: int) -> dict:
        """part -> (B, steps, D) per-step latents of the tokenizer from the plan latents (B, S_plan, code_dim)."""
        x = torch.cat([quantized[p] for p in self.parts], -1)
        x = self._conv(self.decoder_in, x, plan_valid)
        x = self._stack(self.decoder_plan, x, plan_valid)
        W, n = self.config.window, x.shape[1]
        x = upsample_windows(x, W, n * W) + self.window_position.repeat(n, 1)[None]
        step_valid = upsample_windows(plan_valid[..., None], W, n * W)[..., 0]
        x = self._stack(self.decoder_steps, x, step_valid)
        x = self._conv(self.decoder_out, x, step_valid)[:, :steps]
        D = self.config.tokenizer.code_dim
        return {p: x[..., k * D:(k + 1) * D] for k, p in enumerate(self.parts)}

    def decode_codes(self, codes: torch.Tensor, plan_valid: torch.Tensor, steps: int) -> dict:
        return self.decode(self.lookup(codes), plan_valid, steps)

    @torch.no_grad()
    def encode_latents(self, latents: dict, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(codes (B, S_plan, P·Q), plan valid (B, S_plan)): the plan of the tokenizer's latents, all stages. Plan
        steps that are not valid hold code 0 (never read: masked in the prior's loss and memory)."""
        z, plan_valid = self.encode(latents, valid)
        _, codes, _ = self.quantize(z)
        return torch.where(plan_valid[..., None], codes, torch.zeros_like(codes)), plan_valid

    @torch.no_grad()
    def reconstruct(self, latents: dict, valid: torch.Tensor) -> dict:
        """part -> per-step latents decoded from the plan codes of ``latents`` (what the d2 decoder is given in the
        fidelity round trip)."""
        codes, plan_valid = self.encode_latents(latents, valid)
        return self.decode_codes(codes, plan_valid, valid.shape[1])

    def forward(self, latents: dict, valid: torch.Tensor, stages: int | None = None):
        """(reconstructed latents, plan latents, quantized, codes, plan valid); training updates the codebooks."""
        z, plan_valid = self.encode(latents, valid)
        quantized, codes, inputs = self.quantize(z, stages)
        if self.training:
            for p in self.parts:
                self.quantizers[p].update(inputs[p], codes.reshape(*codes.shape[:2], self.config.stages,
                                                                   len(self.parts))[..., self.parts.index(p)],
                                          plan_valid)
        straight = {p: z[p] + (quantized[p] - z[p]).detach() for p in self.parts}
        return self.decode(straight, plan_valid, valid.shape[1]), z, quantized, codes, plan_valid


@torch.no_grad()
def tokenizer_latents(tokenizer: MotionRVQ, motion: MotionBatch) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """(part -> (B, S, D) continuous latents, (B, S) step valid, (B, T, W) unit targets) of the frozen tokenizer."""
    tokenizer.eval()
    unit = diffusion_targets(tokenizer.config.motion, motion)
    return tokenizer.encode(unit), tokenizer.token_valid(motion.frame_valid), unit


def encode_plan(plan: PlanVQ, tokenizer: MotionRVQ, motion: MotionBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """(plan codes (B, S_plan, P·Q), plan valid (B, S_plan)) of a motion batch: plan step i covers tokenizer steps
    [i·W, (i+1)·W)."""
    latents, valid, _ = tokenizer_latents(tokenizer, motion)
    return plan.eval().encode_latents(latents, valid)


def decoded_motion_loss(tokenizer: MotionRVQ, reconstruction: torch.Tensor, motion: MotionBatch):
    """(total, terms) of a unit-space reconstruction (B, T, W) against the batch: ``rvq_loss``'s per-channel and
    skeleton terms with the tokenizer's weights (no velocity matching, no commitment)."""
    config = tokenizer.config
    raw = unit_to_raw(config.motion, reconstruction.double())
    scales = LossScales(**config.motion.loss_config)
    total = torch.zeros((), dtype=torch.float64, device=reconstruction.device)
    terms = {}
    for name in config.motion.trained_channels:
        values, support, _ = channel_loss_per_example(name, raw[name], motion.channels[name], motion.frame_valid,
                                                      config=scales, timestamps=motion.timestamps, allow_inferred=True)
        if bool(support.any()):
            term = torch.where(support, values, torch.zeros_like(values)).sum() / support.sum()
            terms[name] = term.detach()
            total = total + term
    if applicable(config.motion.bindings, config.motion.trained_channels):
        skeleton = skeleton_motion_loss({n: raw[n].float() for n in ("body", "left_hand", "right_hand")}, motion)
        for key, value in skeleton.items():
            terms[f"skeleton_{key}"] = value.detach()
            total = total + config.skeleton_weights[key] * value.to(total.dtype)
    return total, terms


@torch.no_grad()
def part_step_observed(tokenizer: MotionRVQ, motion: MotionBatch) -> dict:
    """part -> (B, S) bool: the tokenizer steps at which the part was observed (``part_code_weights``' rule: at least
    half of the part's unit columns valid in one of the step's frames). Untracked parts are filled with the neutral
    pose by ``diffusion_targets``; their latents encode the fill, not motion."""
    observed = diffusion_valid(tokenizer.config.motion, motion)
    return {part: tokenizer.token_valid(observed[..., getattr(tokenizer, f"columns_{part}")].float().mean(-1) >= .5)
            for part, _ in tokenizer.parts}


def relative_latent_error(reconstruction: dict, latents: dict, valid) -> torch.Tensor:
    """Mean over parts of Σ_m ‖ẑ − z‖² / Σ_m ‖z − z̄‖² over each part's mask m (z̄: the part's mean latent over m, the
    denominator detached): 1 = predicting the mean, 0 = exact. ``valid`` is one (B, S) mask for every part or a
    part -> (B, S) mask (the steps where the part was observed); parts without a masked step are left out."""
    out = []
    for p, z in latents.items():
        m = (valid[p] if isinstance(valid, dict) else valid)[..., None].to(z.dtype)
        count = m.sum()
        if float(count) == 0:
            continue
        mean = (z * m).sum((0, 1)) / count
        energy = (((z - mean) ** 2) * m).sum().detach()
        out.append((((reconstruction[p] - z) ** 2) * m).sum() / (energy + 1e-12))
    if not out:
        return sum(r.sum() for r in reconstruction.values()) * 0.
    return torch.stack(out).mean()


def plan_tokenizer_loss(plan: PlanVQ, tokenizer: MotionRVQ, motion: MotionBatch, *, stages: int | None = None):
    """(total, terms): latent_weight · relative latent error + motion_weight · the frozen decoder's motion loss on
    the motion decoded from the reconstructed latents + commitment. Quantiser dropout as ``rvq_loss``."""
    config = plan.config
    if stages is None and plan.training and config.quantizer_dropout > 0 and torch.rand(()) < config.quantizer_dropout:
        stages = int(torch.randint(1, config.stages + 1, ()))
    latents, valid, _ = tokenizer_latents(tokenizer, motion)
    reconstruction, z, quantized, codes, plan_valid = plan(latents, valid, stages)
    terms = {}
    total = torch.zeros((), dtype=torch.float64, device=valid.device)
    observed = part_step_observed(tokenizer, motion)              # (review fix 7) observed steps of each part only
    latent = relative_latent_error(reconstruction, latents, {p: valid & observed[p] for p in plan.parts})
    terms["latent_relative"] = latent.detach()
    if config.latent_weight:
        total = total + config.latent_weight * latent.to(total.dtype)
    if config.motion_weight:
        unit = tokenizer.decode_latents(reconstruction, motion.frame_valid.shape[1])
        motion_total, motion_terms = decoded_motion_loss(tokenizer, unit, motion)
        terms.update({f"motion_{k}": v for k, v in motion_terms.items()})
        terms["motion"] = motion_total.detach()
        total = total + config.motion_weight * motion_total
    zs, zq = torch.stack([z[p] for p in plan.parts]), torch.stack([quantized[p] for p in plan.parts])
    error = ((zs - zq.detach()) ** 2).mean(-1)                                    # (P, B, S_plan)
    commitment = (torch.where(plan_valid, error, torch.zeros_like(error)).sum((1, 2)) / plan_valid.sum()).mean()
    terms["commitment"] = commitment.detach()
    terms["stages"] = float(stages or config.stages)
    P = len(plan.parts)
    first = torch.where(plan_valid[..., None], codes[..., :P], config.codebook).reshape(-1, P).T
    seen = torch.zeros(P, config.codebook + 1, dtype=torch.bool, device=codes.device).scatter_(1, first, True)
    for p, count in zip(plan.parts, seen[:, :config.codebook].sum(1).unbind()):
        terms[f"codes_used_{p}"] = count
    # stage-1 perplexity per part over this batch's valid plan steps (exp of the code distribution's entropy; codebook
    # size = every code equally used; averaged over the validation batches it is the declared "mean per-batch stage-1
    # perplexity on validation") and the dead codes re-seeded so far (the trainer restores the count on resume)
    counts = torch.zeros(P, config.codebook + 1, device=codes.device).scatter_add_(1, first, torch.ones_like(first,
                                                                                                         dtype=torch.float))
    freq = counts[:, :config.codebook] / counts[:, :config.codebook].sum(1, keepdim=True).clamp_min(1.)
    perplexity = torch.exp(-(freq * freq.clamp_min(1e-12).log()).sum(1))
    for k, p in enumerate(plan.parts):
        terms[f"stage1_perplexity_{p}"] = perplexity[k]
        terms[f"reseeded_{p}"] = float(plan.quantizers[p].reseeded)
    return total + config.commitment * commitment.to(total.dtype), terms


# --------------------------------------------------------------------------------------
# Checkpoints


def plan_checkpoint_bytes(plan: PlanVQ, tokenizer_state: dict, *, tokenizer_key: str, tokenizer_sha256: str,
                          manifest_sha256: str, extra: dict | None = None) -> bytes:
    """A plan tokenizer checkpoint (``torch.save``; the same content gives the same bytes, tested): its kind, config
    and config SHA, its weights, and the frozen tokenizer it reads (key, SHA-256 and weights), so one plan checkpoint
    fully defines the plan's round trip."""
    buffer = io.BytesIO()
    torch.save(dict(kind=PLAN_KIND, model=plan.state_dict(), config=plan.config.to_dict(),
                    config_sha256=plan.config.sha256(), manifest_sha256=manifest_sha256, tokenizer_key=tokenizer_key,
                    tokenizer_sha256=tokenizer_sha256, tokenizer_state=tokenizer_state, **(extra or {})), buffer)
    return buffer.getvalue()


def load_plan_tokenizer(payload: bytes, expected_sha256: str, *, tokenizer_sha256: str | None = None,
                        device="cpu") -> tuple[PlanVQ, MotionRVQ, dict]:
    """(plan tokenizer, its frozen tokenizer, the checkpoint) from checkpoint bytes, both in eval mode; refuses bytes
    whose SHA-256 is not ``expected_sha256``, a checkpoint of another kind or config SHA, and (when given) one built
    on another tokenizer checkpoint than ``tokenizer_sha256``. The EMA weights are used when the checkpoint has
    them."""
    sha = hashlib.sha256(payload).hexdigest()
    if sha != expected_sha256:
        raise ValueError(f"plan tokenizer checkpoint: SHA-256 {sha}, expected {expected_sha256}")
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if saved.get("kind") != PLAN_KIND:
        raise ValueError(f"not a plan tokenizer checkpoint (kind {saved.get('kind')!r})")
    config = PlanVQConfig.from_dict(saved["config"])
    if config.sha256() != saved["config_sha256"]:
        raise ValueError("plan tokenizer checkpoint: its config does not hash to its config_sha256")
    if tokenizer_sha256 is not None and saved["tokenizer_sha256"] != tokenizer_sha256:
        raise ValueError(f"the plan tokenizer reads tokenizer {saved['tokenizer_sha256']}, not {tokenizer_sha256}")
    plan = PlanVQ(config)
    plan.load_state_dict(saved.get("ema") or saved["model"])
    tokenizer = MotionRVQ(config.tokenizer)
    tokenizer.load_state_dict(saved["tokenizer_state"])
    for m in (plan, tokenizer):
        m.to(device).eval().requires_grad_(False)
    return plan, tokenizer, saved


# --------------------------------------------------------------------------------------
# Plan prior


@dataclass(frozen=True)
class PlanPriorConfig(RVQPriorConfig):
    """An ``RVQPriorConfig`` over plan codes: ``tokenizer`` is the plan tokenizer's config (codes_per_step,
    codebook, stages and step = W·d2.step frames per plan step), so the temporal transformer runs over plan steps.
    A plan prior is not itself plan-conditioned."""
    tokenizer: PlanVQConfig

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.tokenizer, PlanVQConfig):
            raise ValueError("a plan prior's tokenizer is a PlanVQConfig")
        if self.plan_window:
            raise ValueError("a plan prior is not plan-conditioned")

    @classmethod
    def from_dict(cls, data: dict) -> "PlanPriorConfig":
        data = dict(data)
        data["tokenizer"] = PlanVQConfig.from_dict(data["tokenizer"])
        return cls(**data)


@torch.no_grad()
def plan_part_weights(tokenizer: MotionRVQ, plan_config: PlanVQConfig, motion: MotionBatch,
                      plan_valid: torch.Tensor) -> torch.Tensor:
    """(N, P·Q) 0/1 weights of the valid plan steps' codes: 1 where the code's part was observed in one of the
    window's tokenizer steps (``part_code_weights``' rule per step: at least half of the part's unit columns valid)."""
    observed = diffusion_valid(tokenizer.config.motion, motion)
    part_ok = torch.stack([tokenizer.token_valid(observed[..., getattr(tokenizer, f"columns_{part}")].float().mean(-1)
                                                 >= .5) for part, _ in tokenizer.parts], -1)    # (B, S, P)
    B, S, P = part_ok.shape
    n, W = plan_valid.shape[1], plan_config.window
    pooled = F.pad(part_ok, (0, 0, 0, n * W - S)).reshape(B, n, W, P).any(2)                       # (B, S_plan, P)
    J = plan_config.codes_per_step
    return pooled[plan_valid][:, torch.arange(J, device=pooled.device) % P].float()


def plan_prior_loss(model: RVQPrior, tokenizer: MotionRVQ, plan: PlanVQ, tokens, text_mask, motion: MotionBatch,
                    loci=None, *, train: bool = True, word_spans: torch.Tensor | None = None):
    """``rvq_prior_loss`` with the plan codes of the batch as targets (text dropout, history options, duration,
    attention terms and part-valid weights as there; steps are plan steps)."""
    if not isinstance(model.config, PlanPriorConfig):
        raise ValueError("plan_prior_loss needs a plan prior (PlanPriorConfig)")
    with torch.no_grad():
        latents, valid, _ = tokenizer_latents(tokenizer, motion)
        codes, plan_valid = plan.eval().encode_latents(latents, valid)
        weight = plan_part_weights(tokenizer, plan.config, motion, plan_valid) if model.config.part_valid_loss else None
    return rvq_prior_loss(model, tokenizer, tokens, text_mask, motion, loci, train=train, word_spans=word_spans,
                          targets=(codes, plan_valid, weight))


TEXT_PREFIXES = ("embed.", "grammar.", "loci.", "text_encoder.", "duration.", "text_in.")
TEXT_FIELDS = ("d_model", "heads", "text_layers", "feedforward", "max_text_bytes", "grammar_features",
               "locus_conditioning", "text_model", "text_model_revision")


def copy_text_encoder(model: RVQPrior, source_state: dict, source_config: dict) -> list[str]:
    """Load the text encoder (byte embedding, grammar and locus projections, text layers, duration head) of another
    prior's weights into ``model``; returns the keys copied. Refuses a source whose text-side config differs or whose
    text keys or shapes do not match ``model``'s exactly (the plan prior warm-starts its text side from W14b)."""
    config = model.config.to_dict()
    differ = {k: (source_config.get(k), config.get(k)) for k in TEXT_FIELDS if source_config.get(k) != config.get(k)}
    if differ:
        raise ValueError(f"the text encoders differ in config: {differ}")
    state = model.state_dict()
    mine = {k for k in state if k.startswith(TEXT_PREFIXES)}
    theirs = {k for k in source_state if k.startswith(TEXT_PREFIXES)}
    if mine != theirs:
        raise ValueError(f"text keys differ: {sorted(mine ^ theirs)[:6]}")
    bad = [k for k in mine if tuple(state[k].shape) != tuple(source_state[k].shape)]
    if bad:
        raise ValueError(f"text weights differ in shape: {bad[:6]}")
    state.update({k: source_state[k].to(state[k].dtype) for k in mine})
    model.load_state_dict(state)
    return sorted(mine)


# --------------------------------------------------------------------------------------
# Generation


def _encoded(prior: RVQPrior, texts, loci, device):
    config = prior.config
    encoded, mask = encode_texts(texts, max_bytes=config.max_text_bytes)
    encoded, mask = encoded.to(device), mask.to(device)
    hidden, log_duration, mask = prior.encode(encoded, mask, generation_loci(config, texts, loci, device))
    empty = _empty_like(encoded[:1, :2])
    empty_hidden, _, empty_mask = prior.encode(empty, empty != PAD, empty_loci(config, device))
    B = len(texts)
    return hidden, mask, log_duration, empty_hidden.expand(B, -1, -1), empty_mask.expand(B, -1)


@torch.no_grad()
def generate_plan_codes_batch(plan_prior: RVQPrior, texts, *, fps: float, durations, guidance: float = 1.,
                              temperature: float | dict = 1., top_k: int = 0, seed: int = 0, loci=None):
    """One (ceil(steps / W), P·Q) plan code tensor per text (CPU), sampled by the plan prior at each text's
    duration (steps: the detail tokenizer's steps of round(duration · fps) frames)."""
    plan_prior.eval()
    config = plan_prior.config
    if not isinstance(config, PlanPriorConfig):
        raise ValueError("generate_plan_codes_batch needs a plan prior")
    device = next(plan_prior.parameters()).device
    hidden, mask, _, empty_hidden, empty_mask = _encoded(plan_prior, texts, loci, device)
    seconds_out = torch.as_tensor(durations, dtype=hidden.dtype, device=device).clamp(1 / fps, 60.)
    frames = [max(int(round(float(s) * fps)), 1) for s in seconds_out]
    counts = [plan_steps(math.ceil(f / config.tokenizer.tokenizer.step), config.tokenizer.window) for f in frames]
    K = max(counts)
    clock = token_seconds(config, K, fps, device).expand(len(texts), -1)            # plan step i: frame i·W·step
    per_code = part_temperatures(plan_prior, config.tokenizer.part_names, temperature)
    generator = torch.Generator().manual_seed(seed)
    sequence = sample_code_sequence(plan_prior, hidden, mask, empty_hidden, empty_mask, seconds_out, clock, K,
                                    per_code, guidance=guidance, top_k=top_k, generator=generator)
    return [sequence[row, 1:counts[row] + 1].cpu() for row in range(len(texts))]


def check_plan_pair(plan_prior: RVQPrior, detail_prior: RVQPrior) -> None:
    """Refuse a plan prior and a detail prior that do not share the plan layout, the plan tokenizer (its config
    SHA-256, which the detail config records) and the detail tokenizer. The checkpoints' plan-tokenizer *weights*
    (their SHA-256) are compared where the checkpoints are loaded (``evaluate_text_to_motion.load_plan_prior``)."""
    plan_config, detail = plan_prior.config.tokenizer, detail_prior.config
    if not isinstance(plan_prior.config, PlanPriorConfig) or not detail.plan_window:
        raise ValueError("expected a plan prior and a plan-conditioned detail prior")
    if (plan_config.window, plan_config.codes_per_step, plan_config.codebook) != (
            detail.plan_window, detail.plan_codes, detail.plan_codebook):
        raise ValueError("the plan prior's plan layout differs from the detail prior's")
    if plan_config.tokenizer.sha256() != detail.tokenizer.sha256():
        raise ValueError("the plan prior and the detail prior read different tokenizers")
    if plan_config.sha256() != detail.plan_tokenizer:
        raise ValueError("the plan prior and the detail prior were built on different plan tokenizers")


@torch.no_grad()
def generate_plan_detail_batch(plan_prior: RVQPrior, detail_prior: RVQPrior, tokenizer: MotionRVQ, texts, *,
                               fps: float, durations=None, guidance: float = 1., temperature: float | dict = 1.,
                               top_k: int = 0, plan_guidance: float | None = None,
                               plan_temperature: float | dict | None = None, plan_top_k: int | None = None,
                               seed: int = 0, sample_prefix: str = "tokens-v2-plan", loci=None, plans=None,
                               return_codes: bool = False):
    """Two-stage generation: the plan prior samples each text's plan (seed + PLAN_SEED_OFFSET; its own guidance,
    temperature and top-k, by default the detail's), then the plan-conditioned detail prior samples the codes given
    that fixed plan (seed), then the frozen tokenizer decodes. ``durations`` None: the detail prior's predicted
    duration. ``plans`` (one (ceil(steps / W), P·Q) tensor per text) skips the plan prior: generation from a fixed
    plan. ``return_codes``: also (plans, detail codes)."""
    check_plan_pair(plan_prior, detail_prior)
    detail_prior.eval(); tokenizer.eval()
    device = next(detail_prior.parameters()).device
    if durations is None:
        _, _, log_duration, _, _ = _encoded(detail_prior, texts, loci, device)
        durations = log_duration.exp().clamp(1 / fps, 60.).tolist()
    if plans is None:
        plans = generate_plan_codes_batch(
            plan_prior, texts, fps=fps, durations=durations,
            guidance=guidance if plan_guidance is None else plan_guidance,
            temperature=temperature if plan_temperature is None else plan_temperature,
            top_k=top_k if plan_top_k is None else plan_top_k, seed=seed + PLAN_SEED_OFFSET, loci=loci)
    from .motion_rvq import generate_rvq_motion_batch
    out = generate_rvq_motion_batch(detail_prior, tokenizer, texts, fps=fps, durations=durations, guidance=guidance,
                                    temperature=temperature, top_k=top_k, seed=seed, sample_prefix=sample_prefix,
                                    loci=loci, return_codes=return_codes, plan=plans)
    if return_codes:
        states, codes = out
        return states, ([p.cpu() for p in plans], codes)
    return out


@torch.no_grad()
def decode_plan_states(plan: PlanVQ, tokenizer: MotionRVQ, plans, texts, *, fps: float, durations,
                       sample_prefix: str = "plan-route"):
    """Canonical states decoded from plan codes alone (no detail prior): each plan (n, P·Q) -> the plan decoder's
    per-step latents -> the frozen tokenizer's decoder, at round(duration · fps) frames. With the plan prior's plans
    this is the plan-only route (what the plan carries by itself); with a clip's own plan, its plan round trip."""
    plan.eval(); tokenizer.eval()
    device = next(tokenizer.parameters()).device
    states = []
    for row, (codes, text, seconds) in enumerate(zip(plans, texts, durations)):
        frames = max(int(round(float(min(max(seconds, 1 / fps), 60.)) * fps)), 1)
        steps = math.ceil(frames / tokenizer.config.step)
        codes = codes.to(device)[None]
        if codes.shape[1] != plan_steps(steps, plan.config.window):
            raise ValueError(f"plan {row}: {codes.shape[1]} plan steps for {steps} steps")
        latents = plan.decode_codes(codes, torch.ones(codes.shape[:2], dtype=torch.bool, device=device), steps)
        unit = tokenizer.decode_latents(latents, frames)
        raw = unit_to_raw(tokenizer.config.motion, unit.double())
        native = native_values(tokenizer.config.motion, {key: v[0] for key, v in raw.items()})
        states.append(assemble_state(tokenizer.config.motion, text, native, frames, fps, f"{sample_prefix}:{row}",
                                     f"{sample_prefix}-clock", method="motion_plan_codes"))
    return states


@torch.no_grad()
def oracle_plans(plan: PlanVQ, tokenizer: MotionRVQ, states, seconds, fps: float) -> tuple[list[torch.Tensor], int]:
    """(one plan per state, the number of plans whose length was adjusted): each reference state's own plan codes
    (the oracle-plan diagnostic: the detail prior given the true plan), fitted to the plan steps that generation at
    ``seconds`` covers (ceil(ceil(round(seconds · fps) / step) / W)): cut, or extended by repeating the last plan
    step, when a state's frame count differs from round(seconds · fps) (a state within 1% of the rate is not
    resampled). The states are encoded one at a time, so no clip's codes depend on another's padding."""
    from ..data.multichannel import collate_multichannel
    device = next(tokenizer.parameters()).device
    out, adjusted = [], 0
    for state, s in zip(states, seconds):
        batch = collate_multichannel([state])
        if device.type != "cpu":
            from dataclasses import replace
            batch = replace(batch, timestamps=batch.timestamps.to(device), frame_valid=batch.frame_valid.to(device),
                            channels={n: replace(c, values=c.values.to(device), valid=c.valid.to(device),
                                                 observed=c.observed.to(device), confidence=c.confidence.to(device))
                                      for n, c in batch.channels.items()})
        codes, valid = encode_plan(plan, tokenizer, batch)
        codes = codes[0, valid[0]].cpu()
        frames = max(int(round(float(s) * fps)), 1)
        need = plan_steps(math.ceil(frames / tokenizer.config.step), plan.config.window)
        if len(codes) != need:
            adjusted += 1
            codes = codes[:need] if len(codes) > need else torch.cat((codes, codes[-1:].expand(need - len(codes), -1)))
        out.append(codes)
    return out, adjusted


__all__ = ["oracle_plans", "PLAN_KIND", "PLAN_PRIOR_KIND", "PlanVQConfig", "PlanVQ", "PlanPriorConfig", "plan_steps", "window_valid",
           "pool_windows", "upsample_windows", "tokenizer_latents", "encode_plan", "plan_tokenizer_loss",
           "relative_latent_error", "decoded_motion_loss", "plan_checkpoint_bytes", "load_plan_tokenizer",
           "plan_part_weights", "plan_prior_loss", "copy_text_encoder", "generate_plan_codes_batch",
           "generate_plan_detail_batch", "check_plan_pair", "padded_plan"]
