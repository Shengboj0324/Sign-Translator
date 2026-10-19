"""English text to continuous canonical motion: model, objective and generation.

Trained on sentence-aligned continuous signing (How2Sign English translations with
video-estimated canonical states). Text is read as UTF-8 bytes, so no tokenizer or
external vocabulary is downloaded. The model predicts the utterance duration from
the text, then decodes every frame non-autoregressively from queries built on the
**true clock in seconds** (and the normalised phase ``t / duration``), cross-
attending to the text. Each canonical channel has its own head in its own
parameterisation, initialised to the neutral value (identity rotation, zero
coefficient, 0.5 bounded state), so training starts from a valid pose.

The objective is :func:`channel_loss_per_example` from the canonical-state model —
``sin²(θ/2)`` for rotations, Huber for coefficients and translations, Bernoulli KL
for bounded states — over supported cells only, plus a Huber loss on log duration.
Video-estimated targets are inferred, so training declares ``allow_inferred=True``.
Generated states are emitted ``observed=False`` with zero confidence: generated
motion can never re-enter training as observation.

**Diffusion objective** (``objective="diffusion"``). A deterministic regressor trained
on these losses predicts the conditional *mean* motion, which under-articulates
signing where several realisations are plausible. With diffusion the decoder instead
denoises: the trained channels are mapped to a unit-scale vector per frame
(6D rotations as they are, coefficients divided by ``coefficient_norm``, bounded
states as ``2p − 1``, unit vectors as they are), noised as
``x_t = √ᾱ(t)·x₀ + √(1−ᾱ(t))·ε`` on the cosine schedule
``ᾱ(t) = cos²(π/2·(t+s)/(1+s)) / cos²(π/2·s/(1+s))``, and the decoder — given ``x_t``
and ``t`` on top of its clock queries — predicts ``x₀`` through the same heads and the
same per-channel losses (x₀ parameterisation). The noisy-input and noise-level layers
start at zero, so an untrained diffusion model equals the regressor. The text is
replaced by the empty text with probability ``cfg_dropout`` so that sampling can use
classifier-free guidance, ``x̂₀ = x̂₀(∅) + w·(x̂₀(text) − x̂₀(∅))``; sampling is DDIM.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
import unicodedata

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..data.multichannel import MotionBatch
from ..pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from ..pose.rotations import matrix_to_rotation_6d, rotation_6d_to_matrix
from .canonical_state import CanonicalChannelBinding, channel_loss_per_example

PAD, BOS, EOS = 256, 257, 258
VOCABULARY = 259
_IDENTITY_6D = (1., 0., 0., 0., 1., 0.)
_FORWARD = (0., 0., 1.)


def encode_texts(texts, *, max_bytes: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """NFC-normalised UTF-8 bytes with BOS/EOS, padded; returns (tokens, mask)."""
    rows = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("every text must be a nonempty string")
        data = unicodedata.normalize("NFC", text.strip()).encode("utf-8")[:max_bytes - 2]
        rows.append([BOS, *data, EOS])
    width = max(len(r) for r in rows)
    tokens = torch.full((len(rows), width), PAD, dtype=torch.long)
    for i, row in enumerate(rows):
        tokens[i, :len(row)] = torch.tensor(row)
    return tokens, tokens != PAD


def sinusoid(values: torch.Tensor, dimensions: int, *, max_period: float) -> torch.Tensor:
    """Sinusoidal features of real values (B, T) -> (B, T, dimensions)."""
    half = dimensions // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=values.device) / half)
    angles = values[..., None].float() * frequencies * 2 * math.pi
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


@dataclass(frozen=True)
class TextToMotionConfig:
    bindings: tuple[CanonicalChannelBinding, ...]
    trained_channels: tuple[str, ...]
    d_model: int = 256
    heads: int = 8
    text_layers: int = 4
    motion_layers: int = 6
    feedforward: int = 1024
    dropout: float = .1
    max_text_bytes: int = 256
    max_seconds: float = 30.
    duration_weight: float = .1
    objective: str = "regression"          # or "diffusion"
    cfg_dropout: float = .1                # diffusion: probability of training on the empty text
    coefficient_norm: float = .5           # diffusion: coefficient values are divided by this
    loss_config: dict = field(default_factory=lambda: dict(
        translation_scale_metres=.05, coefficient_scale=.1, velocity_weight=0.))

    def __post_init__(self):
        names = [b.name for b in self.bindings]
        if names != list(CHANNELS):
            raise ValueError("bindings must cover every canonical channel in order")
        if not set(self.trained_channels) <= set(names) or not self.trained_channels:
            raise ValueError("trained channels must be a nonempty subset of the bindings")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.objective not in ("regression", "diffusion"):
            raise ValueError("objective must be 'regression' or 'diffusion'")
        if not 0 <= self.cfg_dropout < 1 or self.coefficient_norm <= 0:
            raise ValueError("cfg_dropout must lie in [0, 1) and coefficient_norm be positive")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bindings"] = [b.to_dict() for b in self.bindings]
        return data

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "TextToMotionConfig":
        data = dict(data)
        data["bindings"] = tuple(CanonicalChannelBinding(b["name"], tuple(b["labels"]), b["coordinate_frame"],
                                                         b["units"], b["convention"]) for b in data["bindings"])
        data["trained_channels"] = tuple(data["trained_channels"])
        return cls(**data)


class TextToMotion(nn.Module):
    def __init__(self, config: TextToMotionConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.embed = nn.Embedding(VOCABULARY, d, padding_idx=PAD)
        self.text_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True),
            config.text_layers, enable_nested_tensor=False)
        self.duration = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.query = nn.Linear(2 * d, d)
        self.motion_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True),
            config.motion_layers)
        self.norm = nn.LayerNorm(d)
        self.heads_out = nn.ModuleDict()
        for binding in config.bindings:
            head = nn.Linear(d, binding.elements * binding.width)
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.copy_(self._neutral(binding))
            self.heads_out[binding.name] = head
        if config.objective == "diffusion":
            self.layout = diffusion_layout(config)
            width = sum(entry[3] for entry in self.layout)
            self.noisy_in = nn.Linear(width, d)
            self.level = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
            for layer in (self.noisy_in, self.level[2]):          # start as the regressor
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _neutral(binding: CanonicalChannelBinding) -> torch.Tensor:
        if binding.kind == "rotation_6d_columns":
            pattern = torch.tensor(_IDENTITY_6D)
        elif binding.kind == "unit_vector":
            pattern = torch.tensor(_FORWARD)
        else:
            pattern = torch.zeros(binding.width)
        return pattern.repeat(binding.elements)

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor):
        positions = sinusoid(torch.arange(tokens.shape[1], device=tokens.device)[None].expand_as(tokens),
                             self.config.d_model, max_period=1024.)
        hidden = self.text_encoder(self.embed(tokens) + positions, src_key_padding_mask=~mask)
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        log_duration = self.duration(pooled)[:, 0]
        return hidden, log_duration

    def decode(self, hidden, text_mask, seconds: torch.Tensor, frame_mask: torch.Tensor, duration: torch.Tensor,
               noisy: torch.Tensor | None = None, level: torch.Tensor | None = None):
        d = self.config.d_model
        phase = seconds / duration[:, None].clamp_min(1e-3)
        queries = self.query(torch.cat((sinusoid(seconds, d, max_period=2 * self.config.max_seconds),
                                        sinusoid(phase, d, max_period=4.)), dim=-1))
        if self.config.objective == "diffusion":
            if noisy is None or level is None:
                raise ValueError("the diffusion decoder needs the noisy motion and its noise level")
            queries = queries + self.noisy_in(noisy) + self.level(sinusoid(level[:, None], d, max_period=2.))
        states = self.norm(self.motion_decoder(queries, hidden, tgt_key_padding_mask=~frame_mask,
                                               memory_key_padding_mask=~text_mask))
        out = {}
        for binding in self.config.bindings:
            raw = self.heads_out[binding.name](states)
            out[binding.name] = raw.reshape(*raw.shape[:2], binding.elements, binding.width)
        return out

    def forward(self, tokens, text_mask, seconds, frame_mask):
        hidden, log_duration = self.encode(tokens, text_mask)
        observed = (seconds * frame_mask).amax(1)  # teacher forcing: the real clock
        return self.decode(hidden, text_mask, seconds, frame_mask, observed), log_duration


@dataclass(frozen=True)
class LossScales:
    """The scales :func:`channel_loss_per_example` reads; the velocity term is off."""

    translation_scale_metres: float = .05
    coefficient_scale: float = .1
    velocity_weight: float = 0.
    max_gap_seconds: float = 1.


def batch_seconds(motion: MotionBatch) -> torch.Tensor:
    """Seconds since each example's first frame; zero on padding."""
    first = motion.timestamps[:, :1]
    return torch.where(motion.frame_valid, motion.timestamps - first, torch.zeros_like(motion.timestamps))


def utterance_duration(motion: MotionBatch) -> torch.Tensor:
    """Last minus first frame time plus one median frame interval, per example."""
    seconds = batch_seconds(motion)
    last = seconds.gather(1, (motion.lengths - 1)[:, None])[:, 0]
    steps = torch.diff(motion.timestamps, dim=1)
    valid = motion.frame_valid[:, 1:] & motion.frame_valid[:, :-1]
    interval = torch.stack([steps[i][valid[i]].median() if valid[i].any() else torch.tensor(1 / 24., dtype=steps.dtype)
                            for i in range(len(steps))])
    return last + interval


# ---------------------------------------------------------------------------------
# Diffusion: unit-scale per-frame vector of the trained channels, cosine schedule.

_TRANSLATION_NORM = .1   # metres per diffusion unit


def diffusion_layout(config: TextToMotionConfig) -> tuple[tuple[str, str, int, int], ...]:
    """(channel, kind, elements, flat width) of every trained channel, in binding order."""
    return tuple((b.name, b.kind, b.elements, b.elements * b.width)
                 for b in config.bindings if b.name in config.trained_channels)


def alpha_bar(t: torch.Tensor, s: float = .008) -> torch.Tensor:
    """Cosine schedule ᾱ(t), t ∈ [0, 1]; ᾱ(0) = 1, clamped away from 0 at t = 1."""
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2 / math.cos(s / (1 + s) * math.pi / 2) ** 2
    return f.clamp(1e-5, 1.)


def _neutral_native(kind: str, width: int, like: torch.Tensor) -> torch.Tensor:
    pattern = {"rotation_6d_columns": _IDENTITY_6D, "unit_vector": _FORWARD}.get(kind)
    if pattern is not None:
        return torch.tensor(pattern, dtype=like.dtype, device=like.device)
    fill = .5 if kind == "bounded_scalar" else 0.
    return torch.full((width,), fill, dtype=like.dtype, device=like.device)


def _native_to_unit(kind: str, values: torch.Tensor, config: TextToMotionConfig) -> torch.Tensor:
    if kind == "coefficient":
        return values / config.coefficient_norm
    if kind == "bounded_scalar":
        return 2 * values - 1
    if kind == "translation":
        return values / _TRANSLATION_NORM
    return values                                        # 6D rotation columns, unit vectors


def _unit_to_native(kind: str, values: torch.Tensor, config: TextToMotionConfig) -> torch.Tensor:
    """Inverse map, projected onto each channel's valid set (SO(3), sphere, [0, 1])."""
    if kind == "rotation_6d_columns":
        return matrix_to_rotation_6d(rotation_6d_to_matrix(values))
    if kind == "unit_vector":
        return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(1e-9)
    if kind == "coefficient":
        return values * config.coefficient_norm
    if kind == "bounded_scalar":
        return ((values + 1) / 2).clamp(0., 1.)
    return values * _TRANSLATION_NORM


def diffusion_targets(config: TextToMotionConfig, motion: MotionBatch) -> torch.Tensor:
    """x₀ (B, T, W) from the batch targets; unsupported cells hold the neutral value."""
    parts = []
    for name, kind, elements, _ in diffusion_layout(config):
        channel = motion.channels[name]
        values = channel.values.float()
        valid = (channel.valid & motion.frame_valid[..., None])[..., None]
        neutral = _neutral_native(kind, values.shape[-1], values)
        values = torch.where(valid, values, neutral.expand_as(values))
        parts.append(_native_to_unit(kind, values, config).reshape(*values.shape[:2], -1))
    return torch.cat(parts, dim=-1) * motion.frame_valid[..., None]


def heads_to_unit(config: TextToMotionConfig, prediction: dict) -> torch.Tensor:
    """The decoder's x₀ estimate (head outputs) in the diffusion vector space."""
    parts = []
    for name, kind, _, _ in diffusion_layout(config):
        raw = prediction[name]
        native = torch.sigmoid(raw) if kind == "bounded_scalar" else raw
        if kind in ("rotation_6d_columns", "unit_vector"):
            native = _unit_to_native(kind, native, config)       # project onto SO(3) / the sphere
        parts.append(_native_to_unit(kind, native, config).reshape(*raw.shape[:2], -1))
    return torch.cat(parts, dim=-1)


def _empty_text_like(tokens: torch.Tensor) -> torch.Tensor:
    empty = torch.full_like(tokens, PAD)
    empty[:, 0], empty[:, 1] = BOS, EOS
    return empty


def _prediction(model: TextToMotion, tokens, text_mask, motion: MotionBatch):
    """Head outputs, log duration and the rows whose duration is supervised."""
    config = model.config
    seconds = batch_seconds(motion).float()
    observed = (seconds * motion.frame_valid).amax(1)       # teacher forcing: the real clock
    if config.objective == "regression":
        prediction, log_duration = model(tokens, text_mask, seconds, motion.frame_valid)
        return prediction, log_duration, torch.ones(len(tokens), dtype=torch.bool, device=tokens.device)
    conditioned = torch.rand(len(tokens), device=tokens.device) >= config.cfg_dropout
    tokens = torch.where(conditioned[:, None], tokens, _empty_text_like(tokens))
    text_mask = tokens != PAD
    hidden, log_duration = model.encode(tokens, text_mask)
    x0 = diffusion_targets(config, motion)
    t = torch.rand(len(tokens), device=tokens.device)
    ab = alpha_bar(t)[:, None, None]
    noisy = (ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)) * motion.frame_valid[..., None]
    prediction = model.decode(hidden, text_mask, seconds, motion.frame_valid, observed, noisy=noisy, level=t)
    return prediction, log_duration, conditioned


def training_loss(model: TextToMotion, tokens, text_mask, motion: MotionBatch):
    """Mean supported loss over trained channels plus the duration term.

    For ``objective="diffusion"`` the same losses score the decoder's estimate of the
    clean motion from a randomly noised copy (x₀ parameterisation); the duration term
    is computed on the rows that kept their text.
    """
    config = model.config
    prediction, log_duration, duration_rows = _prediction(model, tokens, text_mask, motion)
    loss_config = LossScales(**config.loss_config)
    terms, diagnostics = {}, {}
    total = torch.zeros((), dtype=torch.float64, device=tokens.device)
    for name in config.trained_channels:
        values, support, info = channel_loss_per_example(
            name, prediction[name], motion.channels[name], motion.frame_valid, config=loss_config,
            timestamps=motion.timestamps, allow_inferred=True)
        if bool(support.any()):
            term = values[support].mean()
            terms[name] = float(term.detach())
            total = total + term
            diagnostics.update({f"{name}/{k}": v for k, v in info.items()})
    if bool(duration_rows.any()):
        target = utterance_duration(motion).log().to(log_duration.dtype)
        duration_term = F.smooth_l1_loss(log_duration[duration_rows], target[duration_rows])
        terms["log_duration"] = float(duration_term.detach())
        total = total + config.duration_weight * duration_term.to(total.dtype)
    return total, terms, diagnostics


def _native_values(model: TextToMotion, raw: dict) -> dict:
    """Head outputs -> each channel's valid native values (one example, (T, J, D))."""
    values = {}
    for binding in model.config.bindings:
        r = raw[binding.name].double()
        if binding.kind == "rotation_6d_columns":
            values[binding.name] = matrix_to_rotation_6d(rotation_6d_to_matrix(r))
        elif binding.kind == "unit_vector":
            values[binding.name] = r / torch.linalg.vector_norm(r, dim=-1, keepdim=True).clamp_min(1e-9)
        elif binding.kind == "bounded_scalar":
            values[binding.name] = torch.sigmoid(r)
        else:
            values[binding.name] = r
    return values


@torch.no_grad()
def _sample_unit(model: TextToMotion, hidden, mask, empty_hidden, empty_mask, seconds, duration, *,
                 steps: int, guidance: float, eta: float, generator: torch.Generator) -> torch.Tensor:
    """DDIM from pure noise to x₀ in the diffusion space, with classifier-free guidance."""
    config = model.config
    width = sum(entry[3] for entry in model.layout)
    frames = torch.ones_like(seconds, dtype=torch.bool)
    x = torch.randn((1, seconds.shape[1], width), generator=generator, device="cpu").to(seconds.device)
    grid = torch.linspace(1., 0., steps + 1, device=seconds.device)
    x0 = x
    for i in range(steps):
        t, t_prev = grid[i:i + 1], grid[i + 1:i + 2]
        conditioned = heads_to_unit(config, model.decode(hidden, mask, seconds, frames, duration, noisy=x, level=t))
        if guidance != 1.:
            free = heads_to_unit(config, model.decode(empty_hidden, empty_mask, seconds, frames, duration,
                                                      noisy=x, level=t))
            x0 = free + guidance * (conditioned - free)
        else:
            x0 = conditioned
        ab, ab_prev = alpha_bar(t), (alpha_bar(t_prev) if float(t_prev) > 0 else torch.ones_like(t))
        eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        sigma = eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).clamp_min(0).sqrt()
        noise = torch.randn(x.shape, generator=generator, device="cpu").to(x.device)
        x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp_min(0).sqrt() * eps + sigma * noise
    return x0


def _unit_to_raw(model: TextToMotion, unit: torch.Tensor) -> dict:
    """Split a diffusion vector (1, T, W) into per-channel native values; untrained -> neutral."""
    raw, offset = {}, 0
    trained = {entry[0]: entry for entry in model.layout}
    for binding in model.config.bindings:
        if binding.name in trained:
            _, kind, elements, flat = trained[binding.name]
            piece = unit[..., offset:offset + flat].reshape(*unit.shape[:2], elements, binding.width)
            native = _unit_to_native(kind, piece, model.config)
            raw[binding.name] = (torch.logit(native.clamp(1e-6, 1 - 1e-6)) if kind == "bounded_scalar"
                                 else native)
            offset += flat
        else:
            raw[binding.name] = model._neutral(binding).to(unit).reshape(1, 1, binding.elements, binding.width
                                                                          ).expand(1, unit.shape[1], -1, -1)
    return raw


@torch.no_grad()
def generate(model: TextToMotion, texts, *, fps: float, sample_prefix: str = "generated", steps: int = 50,
             guidance: float = 2., eta: float = 0., seed: int = 0) -> list[MultichannelMotion]:
    """Generated canonical states: observed=False, zero confidence, on a declared clock.

    Regression models decode once; diffusion models sample with ``steps`` DDIM steps,
    classifier-free ``guidance`` and stochasticity ``eta`` from a seeded generator.
    """
    model.eval()
    tokens, mask = encode_texts(texts, max_bytes=model.config.max_text_bytes)
    device = next(model.parameters()).device
    tokens, mask = tokens.to(device), mask.to(device)
    hidden, log_duration = model.encode(tokens, mask)
    durations = log_duration.exp().clamp(1 / fps, model.config.max_seconds)
    generator = torch.Generator().manual_seed(seed)
    if model.config.objective == "diffusion":
        empty = _empty_text_like(tokens[:1, :2])
        empty_hidden, _ = model.encode(empty, empty != PAD)
    states = []
    for row, text in enumerate(texts):
        frames = max(int(round(float(durations[row]) * fps)), 1)
        seconds = (torch.arange(frames, device=device, dtype=torch.float32) / fps)[None]
        if model.config.objective == "diffusion":
            unit = _sample_unit(model, hidden[row:row + 1], mask[row:row + 1], empty_hidden, empty != PAD, seconds,
                                durations[row:row + 1], steps=steps, guidance=guidance, eta=eta, generator=generator)
            out = _unit_to_raw(model, unit)
        else:
            out = model.decode(hidden[row:row + 1], mask[row:row + 1], seconds,
                               torch.ones_like(seconds, dtype=torch.bool), durations[row:row + 1])
        native = _native_values(model, {k: v[0] for k, v in out.items()})
        channels = {}
        for binding in model.config.bindings:
            trained = binding.name in model.config.trained_channels
            valid = np.full((frames, binding.elements), trained)
            channels[binding.name] = MotionChannel(
                values=np.where(valid[..., None], native[binding.name].cpu().numpy(), 0.), valid=valid,
                observed=np.zeros_like(valid), confidence=np.zeros(valid.shape), labels=binding.labels,
                coordinate_frame=binding.coordinate_frame, units=binding.units,
                source_id="text_to_motion", source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                convention=binding.convention,
                inference_method=f"text_to_motion config {model.config.sha256()[:16]}" if trained else None)
        states.append(MultichannelMotion(np.arange(frames, dtype=np.float64) / fps, f"{sample_prefix}-clock",
                                         f"{sample_prefix}:{row}", channels))
    return states


__all__ = ["encode_texts", "TextToMotionConfig", "TextToMotion", "training_loss", "generate", "batch_seconds",
           "utterance_duration", "alpha_bar", "diffusion_layout", "diffusion_targets", "heads_to_unit"]
