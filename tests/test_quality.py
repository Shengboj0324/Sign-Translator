"""Fierce tests for data quality inspection and cleaning.

Every defect is *injected deliberately* and the detector must find it, and the
cleaner must repair it without corrupting the good data.
"""

import numpy as np
import torch

from signtranslator.data.quality import (
    inspect_pose, clean_pose, interpolate_missing, robust_zscore,
)


def _clean_batch(n=8, c=3, t=16, v=27, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Smooth signals (not iid noise) so "dead joint"/"frozen frame" checks are
    # meaningful and the robust scale is well defined.
    ramp = torch.linspace(0, 1, t).view(1, 1, t, 1)
    base = torch.randn(n, c, 1, v, generator=g)
    return base * torch.sin(6.28 * ramp) + 0.01 * torch.randn(n, c, t, v, generator=g)


def test_clean_data_reports_no_issues():
    rep = inspect_pose(_clean_batch())
    assert rep.is_clean, rep.issues
    assert rep.nan_rate == 0.0 and rep.inf_rate == 0.0


def test_detects_nan_and_inf():
    x = _clean_batch()
    x[0, 0, 3, 5] = float("nan")
    x[1, 1, 4, 6] = float("inf")
    rep = inspect_pose(x)
    assert rep.nan_rate > 0 and rep.inf_rate > 0
    assert any("NaN" in i for i in rep.issues)
    assert any("infinite" in i for i in rep.issues)


def test_detects_dropped_zero_keypoints():
    x = _clean_batch()
    x[:, :, :, 7] = 0.0                     # tracker dropped joint 7 everywhere
    rep = inspect_pose(x)
    assert rep.zero_joint_rate > 0.01
    assert any("dropped" in i for i in rep.issues)


def test_detects_dead_joints():
    x = _clean_batch()
    x[:, :, :, 3] = 1.234                   # constant => no motion
    rep = inspect_pose(x)
    assert rep.dead_joint_rate > 0


def test_detects_outlier_spikes():
    x = _clean_batch()
    x[0, 0, 5, 2] = 5000.0
    rep = inspect_pose(x)
    assert rep.outlier_rate > 0
    assert any("outlier" in i for i in rep.issues)


def test_detects_frozen_frames():
    x = _clean_batch()
    for f in range(4, 12):
        x[:, :, f] = x[:, :, 3]             # tracking stalled
    rep = inspect_pose(x)
    assert rep.frozen_frame_rate > 0.05
    assert any("frozen" in i for i in rep.issues)


def test_detects_duplicate_samples():
    x = _clean_batch()
    x[5] = x[0]
    rep = inspect_pose(x)
    assert rep.duplicate_sample_rate > 0
    assert any("duplicate" in i for i in rep.issues)


def test_robust_zscore_resists_outliers():
    """A single huge spike must not mask itself by inflating the scale."""
    x = _clean_batch()
    x[0, 0, 5, 2] = 1e4
    z = robust_zscore(x)
    assert float(z[0, 0, 5, 2].abs()) > 50


# ---- cleaning --------------------------------------------------------------
def test_interpolate_missing_recovers_linear_signal():
    t = 10
    ramp = torch.linspace(0.0, 9.0, t).view(1, t, 1).repeat(3, 1, 2)  # (C,T,V)
    missing = torch.zeros_like(ramp, dtype=torch.bool)
    missing[:, 4:7, :] = True
    corrupted = ramp.clone()
    corrupted[missing] = 0.0
    filled = interpolate_missing(corrupted, missing)
    assert torch.allclose(filled, ramp, atol=1e-4)   # exact on a linear ramp


def test_interpolate_edge_gaps_hold_nearest():
    t = 8
    sig = torch.arange(t, dtype=torch.float32).view(1, t, 1)
    missing = torch.zeros_like(sig, dtype=torch.bool)
    missing[:, :2, :] = True                          # leading gap
    out = interpolate_missing(sig.clone(), missing)
    assert torch.allclose(out[0, :2, 0], torch.tensor([2.0, 2.0]), atol=1e-4)


def test_clean_pose_removes_nans_and_keeps_shape():
    x = _clean_batch()
    x[0, 0, 3, 5] = float("nan")
    x[1, :, 7, 9] = 0.0                                # dropped keypoint
    cleaned, kept, rep = clean_pose(x)
    assert torch.isfinite(cleaned).all()
    assert cleaned.shape[1:] == x.shape[1:]
    assert rep.filled_missing > 0


def test_clean_pose_clips_outliers():
    x = _clean_batch()
    x[0, 0, 5, 2] = 1e5
    cleaned, _, rep = clean_pose(x)
    assert rep.clipped_outliers > 0
    assert float(cleaned.abs().max()) < 1e4


def test_clean_pose_drops_unrecoverable_samples():
    x = _clean_batch(n=6)
    x[2] = float("nan")                                # entirely missing sample
    cleaned, kept, rep = clean_pose(x, drop_if_missing_rate_above=0.5)
    assert 2 in rep.dropped_samples
    assert cleaned.shape[0] == 5 and 2 not in kept.tolist()
    assert torch.isfinite(cleaned).all()


def test_clean_pose_is_idempotent_on_clean_data():
    x = _clean_batch()
    cleaned, kept, rep = clean_pose(x)
    assert rep.filled_missing == 0 and rep.dropped_samples == []
    assert torch.allclose(cleaned, x, atol=1e-5)


def test_cleaning_then_inspection_is_clean():
    """End-to-end: inject several defects, clean, and the result must pass."""
    x = _clean_batch(n=10)
    x[0, 0, 3, 5] = float("nan")
    x[1, 1, 4, 6] = float("inf")
    x[2, 0, 5, 2] = 1e5
    cleaned, _, _ = clean_pose(x)
    rep = inspect_pose(cleaned)
    assert rep.nan_rate == 0.0 and rep.inf_rate == 0.0
    assert rep.outlier_rate == 0.0
