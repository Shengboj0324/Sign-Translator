"""Motion tokens: a VQ-VAE tokenizer of canonical motion and a text-conditioned prior.

The second generative decoder (after diffusion in ``text_to_motion``), in the style
of T2M-GPT. Generating discrete tokens instead of regressing frames lets the prior
choose *one* plausible realisation per step, so it does not average modes either.

**Tokenizer.** The trained channels are mapped to the same unit-scale vector as the
diffusion objective. A 1D convolutional encoder downsamples time by
``2**down_layers`` to latent vectors ``z``; each is replaced by its nearest codebook
entry ``e_k`` (straight-through gradient ``z + sg(e_k − z)``); a mirrored decoder
reconstructs the frames. Codes are learned by exponential moving averages of their
assigned vectors (decay ``ema``); unused codes are re-seeded from the batch. The
loss is the canonical per-channel loss on the reconstruction (geodesic for
rotations, Huber for coefficients, Bernoulli KL for bounded states) plus
``commitment · ||z − sg(e)||²``.

**Prior.** A byte-level text encoder (with the duration head) conditions a causal
transformer that predicts token ``k`` from tokens ``< k`` and the token's time
(``k · step`` seconds and phase), trained by cross entropy. The number of tokens is
``ceil(frames / step)`` for the predicted duration. Text is dropped with
probability ``cfg_dropout`` for classifier-free guidance on the logits; sampling is
seeded, with temperature and top-k.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..data.multichannel import MotionBatch
from .canonical_state import channel_loss_per_example
from .text_to_motion import (
    BOS, EOS, PAD, VOCABULARY, LossScales, TextToMotionConfig, assemble_state, batch_seconds, diffusion_layout,
    diffusion_targets, encode_texts, native_values, sinusoid, unit_to_raw, utterance_duration,
)


# --------------------------------------------------------------------------------------
# Tokenizer


@dataclass(frozen=True)
class MotionVQConfig:
    motion: TextToMotionConfig          # bindings, trained channels and unit-space norms
    hidden: int = 256
    code_dim: int = 128
    codebook: int = 512
    down_layers: int = 2                # time is downsampled by 2 ** down_layers
    res_blocks: int = 2
    commitment: float = .25
    ema: float = .99

    def to_dict(self) -> dict:
        data = asdict(self)
        data["motion"] = self.motion.to_dict()
        return data

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "MotionVQConfig":
        data = dict(data)
        data["motion"] = TextToMotionConfig.from_dict(data["motion"])
        return cls(**data)

    @property
    def step(self) -> int:
        return 2 ** self.down_layers

    @property
    def width(self) -> int:
        return sum(entry[3] for entry in diffusion_layout(self.motion))


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.SiLU(), nn.Conv1d(channels, channels, 3, padding=1), nn.SiLU(),
                                  nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        return x + self.body(x)


class MotionVQ(nn.Module):
    def __init__(self, config: MotionVQConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden
        encoder = [nn.Conv1d(config.width, h, 3, padding=1)]
        for _ in range(config.down_layers):
            encoder += [nn.Conv1d(h, h, 4, stride=2, padding=1), *(_ResBlock(h) for _ in range(config.res_blocks))]
        encoder.append(nn.Conv1d(h, config.code_dim, 1))
        self.encoder = nn.Sequential(*encoder)
        decoder = [nn.Conv1d(config.code_dim, h, 3, padding=1)]
        for _ in range(config.down_layers):
            decoder += [*(_ResBlock(h) for _ in range(config.res_blocks)), nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv1d(h, h, 3, padding=1)]
        decoder += [nn.SiLU(), nn.Conv1d(h, config.width, 3, padding=1)]
        self.decoder = nn.Sequential(*decoder)
        self.register_buffer("codes", torch.randn(config.codebook, config.code_dim) * .1)
        self.register_buffer("code_mass", torch.ones(config.codebook))
        self.register_buffer("code_sum", self.codes.clone())

    # Frames are padded to a multiple of the step; the decoder output is cropped back.
    def _pad(self, unit: torch.Tensor) -> tuple[torch.Tensor, int]:
        frames = unit.shape[1]
        padded = math.ceil(frames / self.config.step) * self.config.step
        return F.pad(unit, (0, 0, 0, padded - frames)), frames

    def encode(self, unit: torch.Tensor) -> torch.Tensor:
        """(B, T, W) unit-space motion -> (B, ceil(T/step), code_dim) latents."""
        padded, _ = self._pad(unit)
        return self.encoder(padded.transpose(1, 2)).transpose(1, 2)

    def quantize(self, latents: torch.Tensor):
        """Nearest codes, their indices, and the straight-through latents."""
        flat = latents.reshape(-1, latents.shape[-1])
        distances = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codes.T + self.codes.pow(2).sum(1)[None])
        index = distances.argmin(1)
        quantized = self.codes[index].reshape(latents.shape)
        return quantized, index.reshape(latents.shape[:-1]), latents + (quantized - latents).detach()

    def decode_codes(self, quantized: torch.Tensor, frames: int) -> torch.Tensor:
        """(B, K, code_dim) -> (B, frames, W) unit-space motion."""
        return self.decoder(quantized.transpose(1, 2)).transpose(1, 2)[:, :frames]

    def decode_indices(self, index: torch.Tensor, frames: int) -> torch.Tensor:
        return self.decode_codes(self.codes[index], frames)

    @torch.no_grad()
    def _update_codes(self, latents: torch.Tensor, index: torch.Tensor, valid: torch.Tensor) -> None:
        flat, index = latents.reshape(-1, latents.shape[-1])[valid.reshape(-1)], index.reshape(-1)[valid.reshape(-1)]
        if not len(flat):
            return
        onehot = F.one_hot(index, self.config.codebook).to(flat.dtype)
        decay = self.config.ema
        self.code_mass.mul_(decay).add_(onehot.sum(0), alpha=1 - decay)
        self.code_sum.mul_(decay).add_(onehot.T @ flat, alpha=1 - decay)
        mass = self.code_mass.clamp_min(1e-5)
        self.codes.copy_(self.code_sum / mass[:, None])
        dead = self.code_mass < 1e-3                        # re-seed unused codes from the batch
        if bool(dead.any()):
            pick = torch.randint(len(flat), (int(dead.sum()),), device=flat.device)
            self.codes[dead] = flat[pick]
            self.code_sum[dead] = flat[pick]
            self.code_mass[dead] = 1.

    def forward(self, unit: torch.Tensor, frame_valid: torch.Tensor):
        latents = self.encode(unit)
        quantized, index, straight = self.quantize(latents)
        token_valid = F.max_pool1d(self._pad(frame_valid[..., None].float())[0].transpose(1, 2),
                                   self.config.step).transpose(1, 2)[..., 0] > 0
        if self.training:
            self._update_codes(latents.detach(), index, token_valid)
        return self.decode_codes(straight, unit.shape[1]), latents, quantized, index, token_valid


def tokenizer_loss(model: MotionVQ, motion: MotionBatch):
    """Canonical per-channel reconstruction loss plus the commitment term."""
    config = model.config
    unit = diffusion_targets(config.motion, motion)
    reconstruction, latents, quantized, index, token_valid = model(unit, motion.frame_valid)
    # float64 for the SO(3) projection: an untrained decoder emits near-degenerate 6D
    # columns, which float32 Gram-Schmidt cannot orthonormalise to the 1e-5 tolerance.
    raw = unit_to_raw(config.motion, reconstruction.double())
    scales = LossScales(**config.motion.loss_config)
    total = torch.zeros((), dtype=torch.float64, device=unit.device)
    terms = {}
    for name in config.motion.trained_channels:
        values, support, _ = channel_loss_per_example(name, raw[name], motion.channels[name], motion.frame_valid,
                                                      config=scales, timestamps=motion.timestamps, allow_inferred=True)
        if bool(support.any()):
            term = values[support].mean()
            terms[name] = float(term.detach())
            total = total + term
    commitment = ((latents - quantized.detach()) ** 2).mean(-1)[token_valid].mean()
    terms["commitment"] = float(commitment.detach())
    used = index[token_valid].unique().numel()
    terms["codes_used"] = float(used)
    return total + config.commitment * commitment.to(total.dtype), terms


# --------------------------------------------------------------------------------------
# Prior over tokens


@dataclass(frozen=True)
class TokenPriorConfig:
    tokenizer: MotionVQConfig
    d_model: int = 256
    heads: int = 8
    text_layers: int = 4
    token_layers: int = 6
    feedforward: int = 1024
    dropout: float = .1
    max_text_bytes: int = 256
    cfg_dropout: float = .1
    duration_weight: float = .1

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tokenizer"] = self.tokenizer.to_dict()
        return data

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "TokenPriorConfig":
        data = dict(data)
        data["tokenizer"] = MotionVQConfig.from_dict(data["tokenizer"])
        return cls(**data)


class TokenPrior(nn.Module):
    def __init__(self, config: TokenPriorConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.embed = nn.Embedding(VOCABULARY, d, padding_idx=PAD)
        self.text_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True),
            config.text_layers, enable_nested_tensor=False)
        self.duration = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        codebook = config.tokenizer.codebook
        self.token_embed = nn.Embedding(codebook + 1, d)          # last index = start of sequence
        self.time = nn.Linear(2 * d, d)
        self.token_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True),
            config.token_layers)
        self.norm = nn.LayerNorm(d)
        self.logits = nn.Linear(d, codebook)

    def encode(self, tokens, mask):
        positions = sinusoid(torch.arange(tokens.shape[1], device=tokens.device)[None].expand_as(tokens),
                             self.config.d_model, max_period=1024.)
        hidden = self.text_encoder(self.embed(tokens) + positions, src_key_padding_mask=~mask)
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return hidden, self.duration(pooled)[:, 0]

    def next_logits(self, hidden, text_mask, previous: torch.Tensor, seconds: torch.Tensor, duration: torch.Tensor):
        """Logits of tokens 0..K-1 given ``previous`` = [start, t_0 .. t_{K-2}] (B, K)."""
        d = self.config.d_model
        phase = seconds / duration[:, None].clamp_min(1e-3)
        queries = self.token_embed(previous) + self.time(torch.cat(
            (sinusoid(seconds, d, max_period=60.), sinusoid(phase, d, max_period=4.)), dim=-1))
        causal = torch.triu(torch.ones(previous.shape[1], previous.shape[1], dtype=torch.bool,
                                       device=previous.device), 1)
        states = self.token_decoder(queries, hidden, tgt_mask=causal, memory_key_padding_mask=~text_mask)
        return self.logits(self.norm(states))


def _empty_like(tokens: torch.Tensor) -> torch.Tensor:
    empty = torch.full_like(tokens, PAD)
    empty[:, 0], empty[:, 1] = BOS, EOS
    return empty


def token_seconds(config: TokenPriorConfig, count: int, fps: float, device) -> torch.Tensor:
    """Clock of token k: the first frame it covers, in seconds."""
    return (torch.arange(count, device=device, dtype=torch.float32) * config.tokenizer.step / fps)[None]


def prior_loss(model: TokenPrior, tokenizer: MotionVQ, tokens, text_mask, motion: MotionBatch):
    """Cross entropy of the tokenizer's codes (frozen tokenizer) plus the duration term."""
    config = model.config
    with torch.no_grad():
        tokenizer.eval()
        unit = diffusion_targets(config.tokenizer.motion, motion)
        _, index, _ = tokenizer.quantize(tokenizer.encode(unit))
        frames = F.max_pool1d(tokenizer._pad(motion.frame_valid[..., None].float())[0].transpose(1, 2),
                              config.tokenizer.step).transpose(1, 2)[..., 0] > 0
    conditioned = torch.rand(len(tokens), device=tokens.device) >= config.cfg_dropout
    tokens = torch.where(conditioned[:, None], tokens, _empty_like(tokens))
    text_mask = tokens != PAD
    hidden, log_duration = model.encode(tokens, text_mask)
    start = torch.full_like(index[:, :1], config.tokenizer.codebook)
    previous = torch.cat((start, index[:, :-1]), dim=1)
    seconds = batch_seconds(motion).float()
    step_seconds = seconds[:, ::config.tokenizer.step][:, :index.shape[1]]
    step_seconds = F.pad(step_seconds, (0, index.shape[1] - step_seconds.shape[1]))
    duration = (seconds * motion.frame_valid).amax(1)
    logits = model.next_logits(hidden, text_mask, previous, step_seconds, duration)
    cross = F.cross_entropy(logits[frames], index[frames])
    terms = {"tokens": float(cross.detach()),
             "token_accuracy": float((logits[frames].argmax(-1) == index[frames]).float().mean())}
    total = cross
    if bool(conditioned.any()):
        target = utterance_duration(motion).log().to(log_duration.dtype)
        duration_term = F.smooth_l1_loss(log_duration[conditioned], target[conditioned])
        terms["log_duration"] = float(duration_term.detach())
        total = total + config.duration_weight * duration_term
    return total, terms


@torch.no_grad()
def generate_tokens_motion(prior: TokenPrior, tokenizer: MotionVQ, texts, *, fps: float, guidance: float = 1.,
                           temperature: float = 1., top_k: int = 0, seed: int = 0, durations=None,
                           sample_prefix: str = "tokens"):
    """Generated canonical states from sampled motion tokens (observed=False, zero confidence)."""
    prior.eval()
    tokenizer.eval()
    config = prior.config
    device = next(prior.parameters()).device
    encoded, mask = encode_texts(texts, max_bytes=config.max_text_bytes)
    encoded, mask = encoded.to(device), mask.to(device)
    hidden, log_duration = prior.encode(encoded, mask)
    seconds_out = (log_duration.exp() if durations is None else
                   torch.as_tensor(durations, dtype=log_duration.dtype, device=device)).clamp(1 / fps, 60.)
    empty = _empty_like(encoded[:1, :2])
    empty_hidden, _ = prior.encode(empty, empty != PAD)
    generator = torch.Generator().manual_seed(seed)
    states = []
    for row, text in enumerate(texts):
        frames = max(int(round(float(seconds_out[row]) * fps)), 1)
        count = math.ceil(frames / config.tokenizer.step)
        clock = token_seconds(config, count, fps, device)
        sequence = torch.full((1, 1), config.tokenizer.codebook, dtype=torch.long, device=device)
        for k in range(count):
            logits = prior.next_logits(hidden[row:row + 1], mask[row:row + 1], sequence, clock[:, :k + 1],
                                       seconds_out[row:row + 1])[:, -1]
            if guidance != 1.:
                free = prior.next_logits(empty_hidden, empty != PAD, sequence, clock[:, :k + 1],
                                         seconds_out[row:row + 1])[:, -1]
                logits = free + guidance * (logits - free)
            logits = logits / max(temperature, 1e-6)
            if top_k:
                threshold = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -math.inf)
            choice = torch.multinomial(torch.softmax(logits.float(), -1).cpu(), 1, generator=generator).to(device)
            sequence = torch.cat((sequence, choice), dim=1)
        unit = tokenizer.decode_indices(sequence[:, 1:], frames)
        raw = unit_to_raw(config.tokenizer.motion, unit.double())
        native = native_values(config.tokenizer.motion, {k: v[0] for k, v in raw.items()})
        states.append(assemble_state(config.tokenizer.motion, text, native, frames, fps, f"{sample_prefix}:{row}",
                                     f"{sample_prefix}-clock", method="motion_tokens"))
    return states


__all__ = ["MotionVQConfig", "MotionVQ", "tokenizer_loss", "TokenPriorConfig", "TokenPrior", "prior_loss",
           "generate_tokens_motion", "token_seconds"]
