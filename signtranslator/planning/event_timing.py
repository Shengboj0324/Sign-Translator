"""Continuous event timing in an explicitly supplied annotation clock.

No frame quantization, event serialization-as-time assumption, minimum-duration
clamp or sorting repair. Overlap and simultaneous onsets are permitted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch.nn import functional as F

from .label_vocabulary import GovernedLabelVocabulary
from .supervision import GovernedSIRAnnotation
from .tensors import tensorize_sir_annotations


def event_intervals(raw: torch.Tensor, origins: torch.Tensor,
                    valid: torch.Tensor) -> torch.Tensor:
    """Map unconstrained onset/duration parameters to float64 seconds.

    start = origin + softplus(onset); end = start + softplus(duration).
    An onset may round to the declared origin. A duration that vanishes through
    underflow or endpoint rounding fails, rather than being replaced by epsilon.
    Inactive slots have exactly zero endpoints and zero gradient.
    """
    if (not isinstance(raw, torch.Tensor) or raw.dtype not in (torch.float32, torch.float64)
            or raw.ndim != 3 or raw.shape[-1] != 2 or not bool(torch.isfinite(raw).all())):
        raise ValueError('timing parameters must be finite float32/64 B,E,2')
    if (not isinstance(origins, torch.Tensor) or origins.dtype != torch.float64
            or origins.shape != (raw.shape[0],) or not bool(torch.isfinite(origins).all())):
        raise ValueError('timing requires one finite float64 clock origin per example')
    if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.shape != raw.shape[:2]):
        raise ValueError('timing requires an explicit B,E validity mask')
    origins, valid = origins.to(raw.device), valid.to(raw.device)
    positive = F.softplus(raw.to(torch.float64))
    start = origins[:, None] + positive[..., 0]
    end = start + positive[..., 1]
    if (not bool(torch.isfinite(start[valid]).all())
            or not bool(torch.isfinite(end[valid]).all())
            or bool((end[valid] <= start[valid]).any())):
        raise ValueError('predicted interval is nonfinite or has no representable duration')
    intervals = torch.stack((start, end), dim=-1)
    return torch.where(valid[..., None], intervals, torch.zeros_like(intervals))


@dataclass(frozen=True)
class AlignedEventIntervals:
    values: torch.Tensor              # B,E,2 float64 seconds; zero padding
    vocabulary_sha256: str
    annotation_sha256: tuple[str, ...]


def aligned_timing_loss(prediction: AlignedEventIntervals,
                        annotations: Sequence[GovernedSIRAnnotation],
                        vocabulary: GovernedLabelVocabulary, *,
                        scale_seconds: float) -> torch.Tensor:
    """Mean per-example, per-event, per-endpoint dimensionless Huber loss.

    Residual r=(prediction-target)/scale_seconds; rho(r)=r²/2 for |r|<=1,
    otherwise |r|-1/2. The declared physical scale is fixed in model config,
    never derived from held-out labels or longest event. STOP has no timing loss.
    """
    if (type(scale_seconds) not in (int, float) or not math.isfinite(scale_seconds)
            or scale_seconds <= 0):
        raise ValueError('timing scale_seconds must be finite and positive')
    if not isinstance(prediction, AlignedEventIntervals) or not isinstance(vocabulary, GovernedLabelVocabulary):
        raise ValueError('typed timing prediction and governed vocabulary required')
    annotations = tuple(annotations)
    target = tensorize_sir_annotations(annotations, expected_lexicon=vocabulary.lexicon,
                                       expected_convention=vocabulary.convention)
    if (prediction.vocabulary_sha256 != vocabulary.lexicon.sha256
            or prediction.annotation_sha256 != target.annotation_sha256):
        raise ValueError('timing vocabulary or annotation order mismatch')
    values = prediction.values
    if (not isinstance(values, torch.Tensor) or values.dtype != torch.float64
            or values.shape != target.intervals.shape or not bool(torch.isfinite(values).all())):
        raise ValueError('timing endpoints must be finite float64 of exact B,E,2 shape')
    valid = target.event_valid.to(values.device)
    if (bool((values[..., 1][valid] <= values[..., 0][valid]).any())
            or bool((values[~valid] != 0).any())):
        raise ValueError('timing endpoints require positive intervals and exact zero padding')
    residual = (values - target.intervals.to(values.device)) / scale_seconds
    if not bool(torch.isfinite(residual).all()):
        raise ValueError('timing residual overflow')
    absolute = residual.abs()
    # Clamp only the quadratic branch argument, not predictions or residuals.
    # This avoids computing a huge unused square in torch.where's other branch.
    rho = torch.where(absolute <= 1, .5 * absolute.clamp_max(1).square(), absolute - .5)
    endpoint_mean = rho.mean(dim=-1)
    per_example = torch.where(valid, endpoint_mean, torch.zeros_like(endpoint_mean)).sum(dim=1)
    per_example = per_example / valid.sum(dim=1)
    loss = per_example.mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError('timing objective overflow')
    return loss
