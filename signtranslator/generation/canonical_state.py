"""Conditional canonical-state motion from reviewed SIR on a declared clock.

The model consumes a :class:`~signtranslator.data.governed_corpus.GovernedMotionBatch`
and predicts every canonical channel in that channel's own geometry: 6D rotation
columns resolved on SO(3), translations in metres, unitless facial coefficients,
unit-norm gaze directions, and bounded blink/contact states. Supervision uses
only observed, valid cells with their declared reliability weights; an
unavailable channel yields zero branch support rather than a measured zero loss.

Mathematics, per channel kind, with per-cell reliability weight ``w`` and the
example mean ``sum(w * l) / sum(w)`` over that example's supported cells:

* ``rotation_6d_columns``. Both prediction and target become rotations by
  Gram-Schmidt. With ``theta`` the geodesic angle of ``R^T Rhat``,

      ||R - Rhat||_F^2 = 6 - 2 tr(R^T Rhat) = 4 (1 - cos theta) = 8 sin^2(theta/2),

  so ``l = ||R - Rhat||_F^2 / 8 = sin^2(theta/2)`` lies in ``[0, 1]``, is zero
  exactly at ``R = Rhat``, and is strictly increasing in ``theta`` on
  ``[0, pi]``. It is smooth everywhere, unlike ``arccos``, whose derivative is
  unbounded at ``theta in {0, pi}``; the geodesic angle itself is reported as a
  diagnostic rather than optimised.
* ``translation`` and ``coefficient``. Dimensionless Huber on
  ``r = (prediction - target) / scale``: ``r^2 / 2`` for ``|r| <= 1``, else
  ``|r| - 1/2``. Translation scale is in metres; both scales are declared
  modelling choices, never reviewer tolerances. Translation channels add an
  optional clock-aware term on first differences divided by the actual elapsed
  seconds between two supported frames, so a nonuniform source clock is not
  silently treated as a frame index.
* ``unit_vector``. The head's raw 3-vector is normalised and scored by
  ``l = (1 - cos angle) / 2`` in ``[0, 1]``, zero exactly at equal direction.
  Gaze magnitude carries no information and is not supervised.
* ``bounded_scalar``. ``l`` is the Kullback-Leibler divergence from the target
  Bernoulli to the predicted one, computed from logits as
  ``BCEWithLogits(t, z) - H(t)``, whose exact minimum ``0`` is attained only at
  ``sigmoid(z) = t``. A plain squared error would also be minimised there but
  would not keep the prediction a calibrated Bernoulli parameter.

Conditioning never reads the motion values it predicts. Each frame attends the
reviewed events whose half-open interval contains that frame's clock value --
the membership the governed corpus derives from the reviewed SIR and the source
clock -- plus one always-attendable null event so a frame inside no event still
has a defined attention distribution. Event timing, labels, referents and loci
are conditioning; the channels are targets.

Scope. This is the W4 conditioned-regression baseline the roadmap requires
before a tokenised or generative alternative may be preferred (blocker B44). It
is not a diffusion model, not calibrated, and not evidence of ASL adequacy.
Branch terms declare per-example support membership but no cell-level target
declarations: dense motion cells would grow the optimizer-exposure ledger in
proportion to frames times elements per step, while the annotation and corpus
hashes already bind the exact target bytes. That omission is recorded as an
unrecorded declaration, which the exposure audit keeps distinct from an empty
one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..data.governed_corpus import GovernedMotionBatch
from ..data.multichannel import ChannelBatch, MotionBatch
from ..inference_context import preserving_eval_mode
from ..planning.label_vocabulary import GovernedLabelVocabulary, encode_governed_labels
from ..planning.loci import LocusAlphabet
from ..planning.tensors import EVENT_KINDS
from ..pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from ..pose.rotations import (
    geodesic_distance, matrix_to_rotation_6d, rotation_6d_to_matrix,
)
from ..pose.temporal import supported_time_intervals, validate_max_gap_seconds
from ..training.objectives import SupportedObjective, SupportedTerm

# Branch names are the canonical channel names, in the canonical order.
CANONICAL_BRANCHES = tuple(CHANNELS)
_SHA256_HEX = 64
_IDENTITY_6D = (1., 0., 0., 0., 1., 0.)
_NONDEGENERATE_DIRECTION = (0., 0., 1.)


def _positive_finite(value, label):
    if (type(value) not in (int, float) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{label} must be a finite positive real number")
    return float(value)


def _bounded_int(value, label, *, low, high):
    if type(value) is not int or isinstance(value, bool) or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


@dataclass(frozen=True)
class CanonicalChannelBinding:
    """The exact channel layout this model is constructed against.

    Binding is declared from already admitted data, never inferred at training
    time: a changed label order, coordinate frame, unit or convention must fail
    rather than silently remap a channel.
    """

    name: str
    labels: tuple[str, ...]
    coordinate_frame: str
    units: str
    convention: str

    def __post_init__(self) -> None:
        if self.name not in CHANNELS:
            raise ValueError("binding must name a canonical channel")
        kind, width, units = CHANNELS[self.name]
        if (type(self.labels) is not tuple or not self.labels
                or any(type(label) is not str or not label.strip() for label in self.labels)
                or len(set(self.labels)) != len(self.labels)):
            raise ValueError(f"{self.name}: distinct nonempty channel labels are required")
        if self.units != units:
            raise ValueError(f"{self.name}: units must be {units}")
        for field in ("coordinate_frame", "convention"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{self.name}: {field} must be explicitly named")
        single = {"root_translation", "head_rotation", "head_translation",
                  "left_gaze", "right_gaze"}
        if self.name in single and len(self.labels) != 1:
            raise ValueError(f"{self.name}: exactly one channel element is required")
        del kind, width

    @property
    def kind(self) -> str:
        return CHANNELS[self.name][0]

    @property
    def width(self) -> int:
        return CHANNELS[self.name][1]

    @property
    def elements(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict:
        return dict(name=self.name, labels=list(self.labels),
                    coordinate_frame=self.coordinate_frame, units=self.units,
                    convention=self.convention)


def channel_bindings_from_state(state: MultichannelMotion) -> tuple[CanonicalChannelBinding, ...]:
    """Derive bindings from one admitted state, in canonical channel order."""
    if not isinstance(state, MultichannelMotion):
        raise ValueError("bindings require a typed canonical motion state")
    state.validate()
    return tuple(CanonicalChannelBinding(
        name, state.channels[name].labels, state.channels[name].coordinate_frame,
        state.channels[name].units, state.channels[name].convention)
        for name in CANONICAL_BRANCHES)


def _bindings_from_batch(motion: MotionBatch) -> tuple[CanonicalChannelBinding, ...]:
    return tuple(CanonicalChannelBinding(
        name, motion.channels[name].layout.labels,
        motion.channels[name].layout.coordinate_frame,
        motion.channels[name].layout.units,
        motion.channels[name].layout.convention)
        for name in CANONICAL_BRANCHES)


@dataclass(frozen=True)
class CanonicalStateConfig:
    """Versioned architecture, capacity, scale and vocabulary binding."""

    hidden_dim: int
    event_dim: int
    attention_heads: int
    frame_frequencies: int
    translation_scale_metres: float
    coefficient_scale: float
    velocity_weight: float
    max_frames: int
    max_events: int
    lexicon_sha256: str
    convention_sha256: str
    locus_count: int
    bindings: tuple[CanonicalChannelBinding, ...]
    max_gap_seconds: float | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported canonical-state model schema")
        _bounded_int(self.hidden_dim, "hidden_dim", low=1, high=4096)
        _bounded_int(self.event_dim, "event_dim", low=1, high=4096)
        _bounded_int(self.attention_heads, "attention_heads", low=1, high=64)
        _bounded_int(self.frame_frequencies, "frame_frequencies", low=1, high=64)
        _bounded_int(self.max_frames, "max_frames", low=2, high=1 << 20)
        _bounded_int(self.max_events, "max_events", low=1, high=1 << 16)
        _bounded_int(self.locus_count, "locus_count", low=1, high=4096)
        if self.event_dim % self.attention_heads:
            raise ValueError("event_dim must be divisible by attention_heads")
        _positive_finite(self.translation_scale_metres, "translation_scale_metres")
        _positive_finite(self.coefficient_scale, "coefficient_scale")
        if (type(self.velocity_weight) not in (int, float) or isinstance(self.velocity_weight, bool)
                or not math.isfinite(self.velocity_weight) or self.velocity_weight < 0):
            raise ValueError("velocity_weight must be a finite non-negative real number")
        for field in ("lexicon_sha256", "convention_sha256"):
            value = getattr(self, field)
            if (type(value) is not str or len(value) != _SHA256_HEX
                    or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        if (type(self.bindings) is not tuple
                or tuple(binding.name for binding in self.bindings) != CANONICAL_BRANCHES
                or any(not isinstance(binding, CanonicalChannelBinding) for binding in self.bindings)):
            raise ValueError("every canonical channel requires exactly one declared binding")
        validate_max_gap_seconds(self.max_gap_seconds)

    @property
    def binding_map(self) -> dict[str, CanonicalChannelBinding]:
        return {binding.name: binding for binding in self.bindings}

    def to_dict(self) -> dict:
        return dict(
            schema_version=self.schema_version, hidden_dim=self.hidden_dim,
            event_dim=self.event_dim, attention_heads=self.attention_heads,
            frame_frequencies=self.frame_frequencies,
            translation_scale_metres=self.translation_scale_metres,
            coefficient_scale=self.coefficient_scale,
            velocity_weight=self.velocity_weight, max_frames=self.max_frames,
            max_events=self.max_events, lexicon_sha256=self.lexicon_sha256,
            convention_sha256=self.convention_sha256, locus_count=self.locus_count,
            max_gap_seconds=self.max_gap_seconds,
            bindings=[binding.to_dict() for binding in self.bindings])

    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# per-channel objectives
# ---------------------------------------------------------------------------
def _supervision_weights(channel: ChannelBatch, frame_valid: torch.Tensor,
                         *, allow_inferred: bool) -> torch.Tensor:
    """Reliability weights restricted to real frames of the padded batch."""
    weights = channel.supervision_weights(allow_inferred=allow_inferred)
    if weights.shape != channel.valid.shape:
        raise ValueError("channel support shape disagrees with its validity mask")
    if frame_valid.shape != weights.shape[:2]:
        raise ValueError("frame validity must cover the padded batch frames")
    weights = torch.where(frame_valid[..., None], weights, torch.zeros_like(weights))
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("supervision weights must be finite and non-negative")
    return weights


def _example_mean(cell_loss: torch.Tensor, weights: torch.Tensor):
    """Weighted mean per example plus the per-example support indicator."""
    if cell_loss.shape != weights.shape:
        raise ValueError("per-cell loss and weight shapes must agree")
    flattened = weights.reshape(weights.shape[0], -1)
    totals = flattened.sum(dim=1)
    support = totals > 0
    numerator = (weights * torch.where(weights > 0, cell_loss,
                                       torch.zeros_like(cell_loss))
                 ).reshape(weights.shape[0], -1).sum(dim=1)
    values = numerator / torch.where(support, totals, torch.ones_like(totals))
    values = torch.where(support, values, torch.zeros_like(values))
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("canonical-state branch loss is not finite")
    return values, support


def _rotation_cells(prediction: torch.Tensor, target: torch.Tensor):
    """``sin^2(theta/2)`` per cell, with the geodesic angle as a diagnostic."""
    predicted = rotation_6d_to_matrix(prediction)
    observed = rotation_6d_to_matrix(target)
    difference = predicted - observed
    chordal = difference.square().sum(dim=(-1, -2))
    loss = chordal / 8.
    with torch.no_grad():
        angle = geodesic_distance(observed.detach(), predicted.detach())
    return loss, angle


def _huber(residual: torch.Tensor) -> torch.Tensor:
    absolute = residual.abs()
    return torch.where(absolute <= 1, .5 * absolute.clamp_max(1).square(), absolute - .5)


def _direction_cells(prediction: torch.Tensor, target: torch.Tensor):
    norm = torch.linalg.vector_norm(prediction, dim=-1, keepdim=True)
    if bool((norm <= 32 * torch.finfo(prediction.dtype).eps).any()):
        raise ValueError("predicted gaze direction is numerically unresolved")
    cosine = (prediction / norm * target).sum(dim=-1)
    return (1. - cosine) / 2.


def _bounded_cells(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL(Bernoulli(target) || Bernoulli(sigmoid(logits))), zero only at equality."""
    if bool(((target < 0) | (target > 1)).any()):
        raise ValueError("bounded-channel targets must lie in [0, 1]")
    cross = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    # Target entropy in nats, with the exact 0 log 0 = 0 convention.
    safe = target.clamp(min=torch.finfo(target.dtype).tiny, max=1.)
    complement = (1. - target).clamp(min=torch.finfo(target.dtype).tiny, max=1.)
    entropy = -(target * safe.log() + (1. - target) * complement.log())
    return (cross - entropy).clamp_min(0.)


def channel_loss_per_example(name: str, prediction: torch.Tensor, channel: ChannelBatch,
                             frame_valid: torch.Tensor, *, config: CanonicalStateConfig,
                             timestamps: torch.Tensor | None = None,
                             allow_inferred: bool = False):
    """Per-example weighted branch loss, support flags and channel diagnostics.

    ``prediction`` is ``B,T,J,D`` in the channel's own parameterisation and may
    carry gradients. Targets, masks and reliability weights come from the
    admitted batch. Unsupported examples contribute exactly zero with support
    ``False``; they are never reported as a measured zero.
    """
    if name not in CHANNELS:
        raise ValueError("unknown canonical channel")
    if not isinstance(channel, ChannelBatch):
        raise ValueError("typed canonical channel batch required")
    kind, width, _ = CHANNELS[name]
    target = channel.values
    if (not isinstance(prediction, torch.Tensor) or prediction.shape != target.shape
            or prediction.ndim != 4 or prediction.shape[-1] != width):
        raise ValueError(f"{name}: prediction must match the channel target shape")
    if not prediction.is_floating_point() or not bool(torch.isfinite(prediction).all()):
        raise ValueError(f"{name}: predictions must be finite floating values")
    weights = _supervision_weights(channel, frame_valid, allow_inferred=allow_inferred)
    dtype = target.dtype if target.dtype == torch.float64 else prediction.dtype
    predicted = prediction.to(dtype)
    observed = target.to(dtype)
    supported = weights > 0
    # Unsupported cells can hold padding or rejected measurements; replace them
    # with a nondegenerate constant so a shared kernel cannot raise on padding.
    diagnostics: dict[str, float] = {}
    if kind == "rotation_6d_columns":
        filler = torch.tensor(_IDENTITY_6D, dtype=dtype, device=observed.device)
        safe_target = torch.where(supported[..., None].expand_as(observed), observed, filler)
        safe_prediction = torch.where(supported[..., None].expand_as(predicted), predicted,
                                      filler.expand_as(predicted))
        cells, angle = _rotation_cells(safe_prediction, safe_target)
        if bool(supported.any()):
            diagnostics["geodesic_degrees_mean"] = float(
                angle[supported].mean() * 180. / math.pi)
            diagnostics["geodesic_degrees_max"] = float(angle[supported].max() * 180. / math.pi)
    elif kind in ("translation", "coefficient"):
        scale = (config.translation_scale_metres if kind == "translation"
                 else config.coefficient_scale)
        residual = (predicted - observed) / scale
        if not bool(torch.isfinite(residual[supported[..., None].expand_as(residual)]).all()):
            raise FloatingPointError(f"{name}: residual overflow")
        cells = _huber(residual).mean(dim=-1)
    elif kind == "unit_vector":
        filler = torch.tensor(_NONDEGENERATE_DIRECTION, dtype=dtype, device=observed.device)
        safe_target = torch.where(supported[..., None].expand_as(observed), observed, filler)
        safe_prediction = torch.where(supported[..., None].expand_as(predicted), predicted,
                                      filler.expand_as(predicted))
        cells = _direction_cells(safe_prediction, safe_target)
    elif kind == "bounded_scalar":
        clean = torch.where(supported[..., None].expand_as(observed), observed,
                            torch.zeros_like(observed))
        cells = _bounded_cells(predicted, clean).mean(dim=-1)
    else:  # pragma: no cover - CHANNELS is closed and fully covered above
        raise ValueError(f"{name}: unsupported channel kind {kind!r}")

    values, support = _example_mean(cells, weights)
    if kind == "translation" and config.velocity_weight > 0:
        if timestamps is None:
            raise ValueError(f"{name}: a clock is required for the velocity term")
        intervals, pairs = supported_time_intervals(
            timestamps, frame_valid, max_gap_seconds=config.max_gap_seconds)
        pair_weights = torch.minimum(weights[:, 1:], weights[:, :-1])
        pair_weights = torch.where(pairs[..., None], pair_weights,
                                   torch.zeros_like(pair_weights))
        pair_supported = pair_weights > 0
        elapsed = intervals.to(dtype)[:, :, None, None]
        step = torch.where(pair_supported[..., None].expand_as(predicted[:, 1:]),
                           predicted[:, 1:] - predicted[:, :-1],
                           torch.zeros_like(predicted[:, 1:])) / elapsed
        reference = torch.where(pair_supported[..., None].expand_as(observed[:, 1:]),
                                observed[:, 1:] - observed[:, :-1],
                                torch.zeros_like(observed[:, 1:])) / elapsed
        velocity_residual = (step - reference) / config.translation_scale_metres
        velocity_cells = _huber(velocity_residual).mean(dim=-1)
        velocity_values, velocity_support = _example_mean(velocity_cells, pair_weights)
        values = values + config.velocity_weight * velocity_values
        diagnostics["velocity_supported_examples"] = float(int(velocity_support.sum()))
    return values, support, diagnostics


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GeneratedStateProvenance:
    """Identity of the model that produced a generated canonical state."""

    model_sha256: str
    config_sha256: str
    inference_method: str

    def __post_init__(self) -> None:
        for field in ("model_sha256", "config_sha256"):
            value = getattr(self, field)
            if (type(value) is not str or len(value) != _SHA256_HEX
                    or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        if type(self.inference_method) is not str or not self.inference_method.strip():
            raise ValueError("inference_method must be explicitly named")


class CanonicalStateSIRDecoder(nn.Module):
    """Reviewed SIR and a declared clock to canonical multichannel motion.

    Every canonical channel has its own head in its own parameterisation. Heads
    start at an exact, nondegenerate constant: rotation heads emit the identity
    rotation, gaze heads a fixed nondegenerate axis, bounded heads ``0.5``, and
    translation and coefficient heads zero. That removes degenerate Gram-Schmidt
    and zero-norm failures at initialisation without hiding them later -- a
    degenerate prediction during training still fails explicitly.
    """

    governed_batch_schema_version = 1
    governed_objective_schema_version = 1

    def __init__(self, vocabulary: GovernedLabelVocabulary, *,
                 locus_alphabet: LocusAlphabet,
                 bindings: tuple[CanonicalChannelBinding, ...],
                 hidden_dim: int = 64, event_dim: int = 64, attention_heads: int = 4,
                 frame_frequencies: int = 6,
                 translation_scale_metres: float = .05,
                 coefficient_scale: float = .5, velocity_weight: float = 1.,
                 max_frames: int = 4096, max_events: int = 128,
                 max_gap_seconds: float | None = None) -> None:
        super().__init__()
        if not isinstance(vocabulary, GovernedLabelVocabulary):
            raise ValueError("a governed label vocabulary is required")
        if not isinstance(locus_alphabet, LocusAlphabet) or locus_alphabet.convention != vocabulary.convention:
            raise ValueError("locus alphabet must match the governed vocabulary convention")
        self.vocabulary = vocabulary
        self.locus_alphabet = locus_alphabet
        self.model_cfg = CanonicalStateConfig(
            hidden_dim=hidden_dim, event_dim=event_dim, attention_heads=attention_heads,
            frame_frequencies=frame_frequencies,
            translation_scale_metres=translation_scale_metres,
            coefficient_scale=coefficient_scale, velocity_weight=velocity_weight,
            max_frames=max_frames, max_events=max_events,
            lexicon_sha256=vocabulary.lexicon.sha256,
            convention_sha256=vocabulary.convention.sha256,
            locus_count=len(locus_alphabet.identities), bindings=tuple(bindings),
            max_gap_seconds=max_gap_seconds)

        cfg = self.model_cfg
        # Event conditioning: kind, governed label, referent presence, locus and
        # the reviewed interval expressed in the declared extent.
        self.kind_embedding = nn.Embedding(len(EVENT_KINDS), cfg.event_dim)
        self.label_embedding = nn.Embedding(len(vocabulary.entries), cfg.event_dim)
        self.locus_embedding = nn.Embedding(cfg.locus_count + 1, cfg.event_dim)
        self.referent_embedding = nn.Embedding(2, cfg.event_dim)
        self.event_interval = nn.Linear(3, cfg.event_dim)
        self.event_projection = nn.Linear(cfg.event_dim, cfg.event_dim)
        self.null_event = nn.Parameter(torch.zeros(cfg.event_dim))

        frame_features = 1 + 2 * cfg.frame_frequencies
        self.frame_projection = nn.Linear(frame_features, cfg.event_dim)
        self.query = nn.Linear(cfg.event_dim, cfg.event_dim)
        self.key = nn.Linear(cfg.event_dim, cfg.event_dim)
        self.value = nn.Linear(cfg.event_dim, cfg.event_dim)
        self.temporal = nn.GRU(2 * cfg.event_dim, cfg.hidden_dim,
                               batch_first=True, bidirectional=True)
        self.heads = nn.ModuleDict()
        for binding in cfg.bindings:
            head = nn.Linear(2 * cfg.hidden_dim, binding.elements * binding.width)
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.copy_(self._head_bias(binding))
            self.heads[binding.name] = head

    @staticmethod
    def _head_bias(binding: CanonicalChannelBinding) -> torch.Tensor:
        if binding.kind == "rotation_6d_columns":
            pattern = torch.tensor(_IDENTITY_6D)
        elif binding.kind == "unit_vector":
            pattern = torch.tensor(_NONDEGENERATE_DIRECTION)
        else:
            pattern = torch.zeros(binding.width)
        return pattern.repeat(binding.elements)

    # -- binding checks -----------------------------------------------------
    def _check_binding(self) -> None:
        if (self.vocabulary.lexicon.sha256 != self.model_cfg.lexicon_sha256
                or self.vocabulary.convention.sha256 != self.model_cfg.convention_sha256
                or self.locus_alphabet.convention != self.vocabulary.convention
                or len(self.locus_alphabet.identities) != self.model_cfg.locus_count):
            raise ValueError("model vocabulary or locus binding changed")

    def _check_batch(self, batch: GovernedMotionBatch) -> None:
        if not isinstance(batch, GovernedMotionBatch):
            raise ValueError("typed governed training batch required")
        self._check_binding()
        targets = batch.sir_targets
        if (targets.lexicon.sha256 != self.model_cfg.lexicon_sha256
                or targets.convention.sha256 != self.model_cfg.convention_sha256):
            raise ValueError("batch SIR targets do not match the model binding")
        if _bindings_from_batch(batch.motion) != self.model_cfg.bindings:
            raise ValueError("batch channel layout differs from the declared binding")
        frames = int(batch.motion.timestamps.shape[1])
        if not 1 <= frames <= self.model_cfg.max_frames:
            raise ValueError("batch exceeds the declared frame capacity")
        if int(targets.event_valid.shape[1]) > self.model_cfg.max_events:
            raise ValueError("batch exceeds the declared event capacity")
        if len(batch.event_frame_membership) != len(batch.annotations):
            raise ValueError("event membership cardinality mismatch")
        # Membership rows and SIR target rows must describe the same events in
        # the same order; a silent reordering would mis-condition every frame.
        for row, mask in enumerate(batch.event_frame_membership):
            declared = tuple(batch.event_ids[row])
            if mask.shape != (len(declared), frames):
                raise ValueError("event membership shape disagrees with the batch")
            count = int(targets.event_valid[row].sum())
            if (count != len(declared)
                    or tuple(targets.event_ids[row, :count].tolist()) != declared):
                raise ValueError("event membership order disagrees with SIR targets")

    # -- conditioning -------------------------------------------------------
    def _frame_features(self, batch: GovernedMotionBatch) -> torch.Tensor:
        """Normalised clock position inside the declared extent, plus harmonics."""
        device = self.frame_projection.weight.device
        timestamps = batch.motion.timestamps.to(device)
        frame_valid = batch.motion.frame_valid.to(device)
        extents = torch.tensor(batch.annotation_extents, dtype=torch.float64, device=device)
        origin, end = extents[:, 0:1], extents[:, 1:2]
        span = end - origin
        if bool((span <= 0).any()) or not bool(torch.isfinite(span).all()):
            raise ValueError("declared annotation extents must have positive duration")
        position = torch.where(frame_valid, (timestamps - origin) / span,
                               torch.zeros_like(timestamps))
        if bool((position[frame_valid] < 0).any()) or bool((position[frame_valid] > 1).any()):
            raise ValueError("frame clock values fall outside the declared extent")
        position = position.to(torch.float32)
        harmonics = [position]
        for index in range(self.model_cfg.frame_frequencies):
            angle = position * math.pi * float(1 << index)
            harmonics.extend((torch.sin(angle), torch.cos(angle)))
        features = torch.stack(harmonics, dim=-1)
        return torch.where(frame_valid[..., None], features, torch.zeros_like(features))

    def _event_features(self, batch: GovernedMotionBatch) -> torch.Tensor:
        device = self.event_projection.weight.device
        targets = batch.sir_targets
        valid = targets.event_valid.to(device)
        kinds = targets.event_kinds.to(device).clamp_min(0)
        # Reuse the governed encoder: it keys on (kind, label_id), so the four
        # event kinds stay distinct namespaces, and it raises on any uncovered
        # active label rather than substituting a nearest class.
        encoded = encode_governed_labels(batch.annotations, self.vocabulary).to(device)
        if encoded.shape != valid.shape:
            raise ValueError("governed label encoding disagrees with the SIR targets")
        label_index = torch.where(valid, encoded, torch.zeros_like(encoded))
        locus = torch.where(targets.locus_present.to(device),
                            targets.locus_ids.to(device).clamp_min(0) + 1,
                            torch.zeros_like(targets.locus_ids.to(device)))
        if bool((locus > self.model_cfg.locus_count).any()):
            raise ValueError("event locus falls outside the convention alphabet")
        referent = targets.referent_present.to(device).to(torch.int64)
        extents = torch.tensor(batch.annotation_extents, dtype=torch.float64, device=device)
        origin, span = extents[:, 0:1], extents[:, 1:2] - extents[:, 0:1]
        intervals = targets.intervals.to(device)
        start = (intervals[..., 0] - origin) / span
        end = (intervals[..., 1] - origin) / span
        geometry = torch.stack((start, end, end - start), dim=-1).to(torch.float32)
        geometry = torch.where(valid[..., None], geometry, torch.zeros_like(geometry))
        features = (self.kind_embedding(kinds) + self.label_embedding(label_index)
                    + self.locus_embedding(locus) + self.referent_embedding(referent)
                    + self.event_interval(geometry))
        features = self.event_projection(torch.tanh(features))
        return torch.where(valid[..., None], features, torch.zeros_like(features))

    def _membership(self, batch: GovernedMotionBatch) -> torch.Tensor:
        device = self.event_projection.weight.device
        events = int(batch.sir_targets.event_valid.shape[1])
        frames = int(batch.motion.timestamps.shape[1])
        mask = torch.zeros((len(batch.annotations), events, frames), dtype=torch.bool,
                           device=device)
        for row, membership in enumerate(batch.event_frame_membership):
            mask[row, :membership.shape[0]] = membership.to(device)
        return mask

    def frame_states(self, batch: GovernedMotionBatch) -> torch.Tensor:
        """Conditioned per-frame features; reads no channel values."""
        cfg = self.model_cfg
        frames = self._frame_features(batch)
        events = self._event_features(batch)
        membership = self._membership(batch)
        frame_tokens = self.frame_projection(frames)
        batch_size, frame_count, _ = frame_tokens.shape
        null = self.null_event.expand(batch_size, 1, cfg.event_dim)
        keys_source = torch.cat((null, events), dim=1)
        # The null event is always attendable, so every frame -- including one
        # inside no reviewed event -- has a defined attention distribution.
        attendable = torch.cat(
            (torch.ones((batch_size, 1, frame_count), dtype=torch.bool, device=membership.device),
             membership), dim=1).transpose(1, 2)
        heads = cfg.attention_heads
        width = cfg.event_dim // heads

        def split(value):
            return value.reshape(*value.shape[:2], heads, width).transpose(1, 2)

        query = split(self.query(frame_tokens))
        key = split(self.key(keys_source))
        value = split(self.value(keys_source))
        scores = query @ key.transpose(-1, -2) / math.sqrt(width)
        scores = scores.masked_fill(~attendable[:, None, :, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        if not bool(torch.isfinite(weights).all()):
            raise FloatingPointError("event attention produced a nonfinite distribution")
        context = (weights @ value).transpose(1, 2).reshape(batch_size, frame_count, cfg.event_dim)
        joined = torch.cat((frame_tokens, context), dim=-1)
        lengths = batch.motion.lengths.to("cpu")
        packed = pack_padded_sequence(joined, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.temporal(packed)
        states, _ = pad_packed_sequence(output, batch_first=True, total_length=frame_count)
        return torch.where(batch.motion.frame_valid.to(states.device)[..., None], states,
                           torch.zeros_like(states))

    def predict(self, batch: GovernedMotionBatch) -> dict[str, torch.Tensor]:
        """Raw per-channel predictions in each channel's parameterisation."""
        self._check_batch(batch)
        states = self.frame_states(batch)
        predictions = {}
        for binding in self.model_cfg.bindings:
            raw = self.heads[binding.name](states)
            predictions[binding.name] = raw.reshape(
                *raw.shape[:2], binding.elements, binding.width)
        return predictions

    # -- objective ----------------------------------------------------------
    def training_step(self, batch: GovernedMotionBatch, *, weights: dict) -> SupportedObjective:
        if (not isinstance(weights, dict) or set(weights) != set(CANONICAL_BRANCHES)
                or any(type(weight) not in (int, float) or isinstance(weight, bool)
                       or not math.isfinite(weight) or weight <= 0
                       for weight in weights.values())):
            raise ValueError("explicit positive weights for all canonical channels required")
        predictions = self.predict(batch)
        frame_valid = batch.motion.frame_valid
        timestamps = batch.motion.timestamps
        terms = {}
        self.last_diagnostics: dict[str, dict[str, float]] = {}
        for name in CANONICAL_BRANCHES:
            values, support, diagnostics = channel_loss_per_example(
                name, predictions[name], batch.motion.channels[name], frame_valid,
                config=self.model_cfg, timestamps=timestamps)
            self.last_diagnostics[name] = diagnostics
            terms[name] = SupportedTerm(
                values.sum(), int(support.sum()),
                tuple(bool(flag) for flag in support.detach().cpu().tolist()), None)
        return SupportedObjective(terms, weights, len(batch.annotations))

    # -- generation ---------------------------------------------------------
    @torch.no_grad()
    def generate_canonical_state(self, batch: GovernedMotionBatch, *,
                                 inference_method: str) -> tuple[MultichannelMotion, ...]:
        """Generate one canonical state per batch row on that row's own clock.

        Returned channels are marked ``observed=False`` with zero confidence and
        a declared inference method, so the existing supervision weights are
        exactly zero: generated motion can be inspected, saved and rendered, but
        never admitted as observation or consumed as a training target. Channel
        source identity names this model, not a recording.
        """
        if type(inference_method) is not str or not inference_method.strip():
            raise ValueError("an explicit inference method must be declared")
        with preserving_eval_mode(self):
            predictions = self.predict(batch)
            provenance = GeneratedStateProvenance(
                model_sha256=self.state_sha256(),
                config_sha256=self.model_cfg.sha256(),
                inference_method=inference_method)
            states = []
            lengths = batch.motion.lengths.tolist()
            for row, count in enumerate(lengths):
                clock = batch.motion.timestamps[row, :count].detach().cpu().to(torch.float64)
                channels = {}
                for binding in self.model_cfg.bindings:
                    raw = predictions[binding.name][row, :count].detach().cpu().to(torch.float64)
                    values = self._canonicalize(binding, raw)
                    shape = values.shape[:2]
                    channels[binding.name] = MotionChannel(
                        values=values.numpy(),
                        valid=np.ones(shape, dtype=np.bool_),
                        observed=np.zeros(shape, dtype=np.bool_),
                        confidence=np.zeros(shape, dtype=np.float64),
                        labels=binding.labels,
                        coordinate_frame=binding.coordinate_frame,
                        units=binding.units,
                        source_id=f"model:{provenance.config_sha256[:16]}",
                        source_sha256=provenance.model_sha256,
                        convention=binding.convention,
                        inference_method=(f"{provenance.inference_method}"
                                          f";model={provenance.model_sha256}"
                                          f";config={provenance.config_sha256}"))
                states.append(MultichannelMotion(
                    timestamps=clock.numpy(), clock_id=batch.motion.clock_ids[row],
                    sample_id=f"generated:{batch.motion.sample_ids[row]}", channels=channels))
            return tuple(states)

    @staticmethod
    def _canonicalize(binding: CanonicalChannelBinding, raw: torch.Tensor) -> torch.Tensor:
        """Project a raw head output onto its channel's declared manifold."""
        if binding.kind == "rotation_6d_columns":
            return matrix_to_rotation_6d(rotation_6d_to_matrix(raw))
        if binding.kind == "unit_vector":
            norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
            if bool((norm <= 32 * torch.finfo(raw.dtype).eps).any()):
                raise ValueError("generated gaze direction is numerically unresolved")
            return raw / norm
        if binding.kind == "bounded_scalar":
            return torch.sigmoid(raw)
        return raw

    def state_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            header = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                                separators=(",", ":")).encode()
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()

    def with_velocity_weight(self, value: float) -> "CanonicalStateSIRDecoder":
        """Declare a different velocity coupling; configuration stays explicit."""
        self.model_cfg = replace(self.model_cfg, velocity_weight=value)
        self.model_cfg.__post_init__()
        return self
