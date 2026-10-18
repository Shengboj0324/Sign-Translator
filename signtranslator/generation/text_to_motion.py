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

    def decode(self, hidden, text_mask, seconds: torch.Tensor, frame_mask: torch.Tensor, duration: torch.Tensor):
        d = self.config.d_model
        phase = seconds / duration[:, None].clamp_min(1e-3)
        queries = self.query(torch.cat((sinusoid(seconds, d, max_period=2 * self.config.max_seconds),
                                        sinusoid(phase, d, max_period=4.)), dim=-1))
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


def training_loss(model: TextToMotion, tokens, text_mask, motion: MotionBatch):
    """Mean supported loss over trained channels plus the duration term."""
    config = model.config
    seconds = batch_seconds(motion).float()
    prediction, log_duration = model(tokens, text_mask, seconds, motion.frame_valid)
    loss_config = LossScales(**config.loss_config)
    terms, diagnostics = {}, {}
    total = torch.zeros((), dtype=torch.float64, device=seconds.device)
    for name in config.trained_channels:
        values, support, info = channel_loss_per_example(
            name, prediction[name], motion.channels[name], motion.frame_valid, config=loss_config,
            timestamps=motion.timestamps, allow_inferred=True)
        if bool(support.any()):
            term = values[support].mean()
            terms[name] = float(term.detach())
            total = total + term
            diagnostics.update({f"{name}/{k}": v for k, v in info.items()})
    target = utterance_duration(motion).log().to(log_duration.dtype)
    duration_term = F.smooth_l1_loss(log_duration, target)
    terms["log_duration"] = float(duration_term.detach())
    total = total + config.duration_weight * duration_term.to(total.dtype)
    return total, terms, diagnostics


@torch.no_grad()
def generate(model: TextToMotion, texts, *, fps: float, sample_prefix: str = "generated") -> list[MultichannelMotion]:
    """Generated canonical states: observed=False, zero confidence, on a declared clock."""
    model.eval()
    tokens, mask = encode_texts(texts, max_bytes=model.config.max_text_bytes)
    device = next(model.parameters()).device
    tokens, mask = tokens.to(device), mask.to(device)
    hidden, log_duration = model.encode(tokens, mask)
    durations = log_duration.exp().clamp(1 / fps, model.config.max_seconds)
    states = []
    for row, text in enumerate(texts):
        frames = max(int(round(float(durations[row]) * fps)), 1)
        seconds = (torch.arange(frames, device=device, dtype=torch.float32) / fps)[None]
        out = model.decode(hidden[row:row + 1], mask[row:row + 1], seconds, torch.ones_like(seconds, dtype=torch.bool),
                           durations[row:row + 1])
        channels = {}
        for binding in model.config.bindings:
            raw = out[binding.name][0].double()
            trained = binding.name in model.config.trained_channels
            if binding.kind == "rotation_6d_columns":
                values = matrix_to_rotation_6d(rotation_6d_to_matrix(raw))
            elif binding.kind == "unit_vector":
                values = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-9)
            elif binding.kind == "bounded_scalar":
                values = torch.sigmoid(raw)
            else:
                values = raw
            valid = np.full((frames, binding.elements), trained)
            channels[binding.name] = MotionChannel(
                values=np.where(valid[..., None], values.cpu().numpy(), 0.), valid=valid,
                observed=np.zeros_like(valid), confidence=np.zeros(valid.shape), labels=binding.labels,
                coordinate_frame=binding.coordinate_frame, units=binding.units,
                source_id="text_to_motion", source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                convention=binding.convention,
                inference_method=f"text_to_motion config {model.config.sha256()[:16]}" if trained else None)
        states.append(MultichannelMotion(np.arange(frames, dtype=np.float64) / fps, f"{sample_prefix}-clock",
                                         f"{sample_prefix}:{row}", channels))
    return states


__all__ = ["encode_texts", "TextToMotionConfig", "TextToMotion", "training_loss", "generate", "batch_seconds",
           "utterance_duration"]
