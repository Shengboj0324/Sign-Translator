"""Data quality inspection and cleaning for 3D keypoint corpora.

Real motion-capture / pose-estimation output is messy: trackers drop joints
(emitting NaN or exact zeros), emit spikes when they lose the subject, and
produce frozen segments when tracking fails. Training on this silently degrades
a model, so this module (a) *measures* the defects and (b) *repairs* what is
repairable while flagging what is not.

Everything operates on ``(C, T, V)`` or ``(N, C, T, V)`` float tensors.

Design: inspection never mutates; cleaning returns new tensors plus a report of
exactly what was changed, so data transformations are auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------
@dataclass
class QualityReport:
    """Per-corpus defect measurements. All rates are fractions in [0, 1]."""

    num_samples: int
    nan_rate: float
    inf_rate: float
    zero_joint_rate: float          # fraction of (sample, frame, joint) fully zero
    dead_joint_rate: float          # joints with ~no motion across a whole clip
    outlier_rate: float             # values beyond the robust z threshold
    max_abs_value: float
    frozen_frame_rate: float        # consecutive duplicate frames
    duplicate_sample_rate: float
    issues: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        lines = ["Data quality report", "=" * 46,
                 f"  samples                  {self.num_samples}",
                 f"  nan_rate                 {self.nan_rate:.5f}",
                 f"  inf_rate                 {self.inf_rate:.5f}",
                 f"  zero_joint_rate          {self.zero_joint_rate:.5f}",
                 f"  dead_joint_rate          {self.dead_joint_rate:.5f}",
                 f"  outlier_rate             {self.outlier_rate:.5f}",
                 f"  frozen_frame_rate        {self.frozen_frame_rate:.5f}",
                 f"  duplicate_sample_rate    {self.duplicate_sample_rate:.5f}",
                 f"  max_abs_value            {self.max_abs_value:.3f}",
                 "-" * 46]
        if self.issues:
            lines.append("  ISSUES:")
            lines.extend(f"    - {i}" for i in self.issues)
        else:
            lines.append("  no issues detected")
        return "\n".join(lines)


def _as_batched(pose: torch.Tensor) -> torch.Tensor:
    if pose.dim() == 3:
        return pose.unsqueeze(0)
    if pose.dim() != 4:
        raise ValueError("pose must be (C, T, V) or (N, C, T, V)")
    return pose


def robust_stats(pose: torch.Tensor, eps: float = 1e-6):
    """Return ``(median, scale)`` per channel using the median absolute deviation.

    MAD rather than the standard deviation, so the spikes we are trying to
    detect do not inflate the scale they are measured against. Scaled by 1.4826
    so MAD estimates sigma for Gaussian data.

    Detection and clipping share this helper so that a value clipped at
    ``|z| = k`` is guaranteed not to be re-flagged at threshold ``k``.
    """
    x = _as_batched(pose)
    def _med_cv(a):  # median over N, T, V -> shape (1, C, 1, 1)
        return (a.median(dim=0, keepdim=True).values
                 .median(dim=2, keepdim=True).values
                 .median(dim=3, keepdim=True).values)
    med = _med_cv(x)
    scale = 1.4826 * _med_cv((x - med).abs()) + eps
    return med, scale


def robust_zscore(pose: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Median/MAD-based z-score, computed per channel."""
    x = _as_batched(pose)
    med, scale = robust_stats(x, eps)
    return (x - med) / scale


def inspect_pose(pose: torch.Tensor, outlier_z: float = 12.0,
                 dead_joint_std: float = 1e-4,
                 max_abs_allowed: float = 1e4) -> QualityReport:
    """Measure defects without modifying the data."""
    x = _as_batched(pose)
    n, c, t, v = x.shape
    total = x.numel()

    nan_mask = torch.isnan(x)
    inf_mask = torch.isinf(x)
    finite = x.masked_fill(nan_mask | inf_mask, 0.0)

    nan_rate = float(nan_mask.float().mean())
    inf_rate = float(inf_mask.float().mean())

    # A joint reported as exactly zero across all channels is the usual
    # "tracker dropped this keypoint" sentinel.
    zero_joint = (finite.abs().sum(dim=1) == 0)                    # (N, T, V)
    zero_joint_rate = float(zero_joint.float().mean())

    # Dead joints: (near-)zero temporal variance across a whole clip.
    joint_std = finite.std(dim=2)                                   # (N, C, V)
    dead_joint_rate = float((joint_std < dead_joint_std).float().mean())

    # Outliers via robust z-score (ignores NaN by using the finite copy).
    z = robust_zscore(finite)
    outlier_rate = float((z.abs() > outlier_z).float().mean())
    max_abs_z = float(z.abs().max()) if total else 0.0
    max_abs_value = float(finite.abs().max()) if total else 0.0

    # Frozen frames: consecutive identical frames indicate tracking stalls.
    if t > 1:
        same = (finite[:, :, 1:] - finite[:, :, :-1]).abs().sum(dim=(1, 3)) == 0
        frozen_frame_rate = float(same.float().mean())
    else:
        frozen_frame_rate = 0.0

    # Duplicate samples (exact repeats bias training and inflate val scores).
    flat = finite.reshape(n, -1)
    duplicate = 0
    if n > 1:
        seen = set()
        for i in range(n):
            key = hash(flat[i].numpy().tobytes())
            if key in seen:
                duplicate += 1
            seen.add(key)
    duplicate_sample_rate = duplicate / max(1, n)

    issues: List[str] = []
    if nan_rate > 0:
        issues.append(f"{nan_rate:.4%} NaN values")
    if inf_rate > 0:
        issues.append(f"{inf_rate:.4%} infinite values")
    if zero_joint_rate > 0.01:
        issues.append(f"{zero_joint_rate:.2%} dropped (all-zero) keypoints")
    if dead_joint_rate > 0.05:
        issues.append(f"{dead_joint_rate:.2%} dead (motionless) joint-channels")
    # Flag on rate OR on severity: a single 1000-sigma spike is a defect even
    # though its rate is negligible.
    if outlier_rate > 0.001:
        issues.append(f"{outlier_rate:.4%} extreme outliers (|z|>{outlier_z})")
    elif max_abs_z > 4 * outlier_z:
        issues.append(f"severe outlier spike (max |z| = {max_abs_z:.1f})")
    if max_abs_value > max_abs_allowed:
        issues.append(f"max |value| = {max_abs_value:.1f} exceeds {max_abs_allowed}")
    if frozen_frame_rate > 0.05:
        issues.append(f"{frozen_frame_rate:.2%} frozen (duplicate) frames")
    if duplicate_sample_rate > 0:
        issues.append(f"{duplicate_sample_rate:.2%} duplicate samples")

    return QualityReport(
        num_samples=n, nan_rate=nan_rate, inf_rate=inf_rate,
        zero_joint_rate=zero_joint_rate, dead_joint_rate=dead_joint_rate,
        outlier_rate=outlier_rate, max_abs_value=max_abs_value,
        frozen_frame_rate=frozen_frame_rate,
        duplicate_sample_rate=duplicate_sample_rate, issues=issues,
    )


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
@dataclass
class CleaningReport:
    filled_missing: int = 0
    clipped_outliers: int = 0
    dropped_samples: List[int] = field(default_factory=list)

    def summary(self) -> str:
        return (f"cleaned: filled {self.filled_missing} missing values, "
                f"clipped {self.clipped_outliers} outliers, "
                f"dropped {len(self.dropped_samples)} samples")


def interpolate_missing(pose: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
    """Fill masked entries by linear interpolation along time.

    ``missing`` is a boolean mask broadcastable to ``pose``. Gaps at the start or
    end are filled with the nearest valid value (edge hold); a channel that is
    missing for the entire clip is filled with zeros.
    """
    x = _as_batched(pose).clone()
    m = _as_batched(missing.expand_as(_as_batched(pose)))
    n, c, t, v = x.shape

    # Work on (N*C*V, T) so interpolation is a simple 1-D pass per series.
    series = x.permute(0, 1, 3, 2).reshape(-1, t)
    mask = m.permute(0, 1, 3, 2).reshape(-1, t)

    for i in range(series.shape[0]):
        bad = mask[i]
        if not bool(bad.any()):
            continue
        good_idx = (~bad).nonzero(as_tuple=True)[0]
        if good_idx.numel() == 0:
            series[i] = 0.0
            continue
        bad_idx = bad.nonzero(as_tuple=True)[0]
        series[i, bad_idx] = torch.from_numpy(
            _np_interp(bad_idx.numpy(), good_idx.numpy(),
                       series[i, good_idx].numpy())).to(series.dtype)

    out = series.reshape(n, c, v, t).permute(0, 1, 3, 2).contiguous()
    return out.squeeze(0) if pose.dim() == 3 else out


def _np_interp(x, xp, fp):
    import numpy as np
    return np.interp(x, xp, fp)


def clean_pose(pose: torch.Tensor, outlier_z: float = 12.0,
               treat_zero_joint_as_missing: bool = True,
               drop_if_missing_rate_above: float = 0.5,
               ) -> tuple:
    """Repair a pose batch; returns ``(clean_pose, kept_index, CleaningReport)``.

    Steps, in order:
      1. NaN/Inf (and optionally all-zero "dropped keypoint" frames) are marked
         missing and linearly interpolated over time.
      2. Remaining extreme values are clipped at the robust z threshold rather
         than deleted, preserving sequence length.
      3. Samples whose missing rate exceeds ``drop_if_missing_rate_above`` are
         dropped as unrecoverable, and their indices reported.
    """
    x = _as_batched(pose).clone()
    n = x.shape[0]
    report = CleaningReport()

    missing = torch.isnan(x) | torch.isinf(x)
    if treat_zero_joint_as_missing:
        zero_joint = (x.nan_to_num(0.0).abs().sum(dim=1, keepdim=True) == 0)
        missing = missing | zero_joint.expand_as(x)

    per_sample_missing = missing.reshape(n, -1).float().mean(dim=1)
    keep = per_sample_missing <= drop_if_missing_rate_above
    report.dropped_samples = [i for i in range(n) if not bool(keep[i])]

    report.filled_missing = int(missing.sum())
    x = x.nan_to_num(0.0, posinf=0.0, neginf=0.0)
    if bool(missing.any()):
        x = _as_batched(interpolate_missing(x, missing))

    # Clip outliers using *the same* robust statistics the detector uses, so a
    # cleaned corpus provably contains no values beyond the threshold.
    med, scale = robust_stats(x)
    over = ((x - med) / scale).abs() > outlier_z
    if bool(over.any()):
        report.clipped_outliers = int(over.sum())
        limit = outlier_z * scale
        x = torch.max(torch.min(x, med + limit), med - limit)

    kept_index = keep.nonzero(as_tuple=True)[0]
    x = x[kept_index]
    if pose.dim() == 3:
        x = x.squeeze(0)
    return x, kept_index, report
