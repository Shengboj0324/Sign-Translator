"""Verification of pose evaluation metrics.

Kabsch alignment is proved to recover a known similarity transform (with a
reflection guard keeping det R = +1); geodesic rotation error uses the SO(3)
metric; and a fixed fingertip displacement is shown to move the fingertip-weighted
error far more than an unweighted MPJPE -- the quantified reason MPJPE alone is
inadequate for signing.
"""

import math

import pytest
import torch

from signtranslator.pose.rotations import axis_angle_to_matrix
from signtranslator.pose.metrics import (
    mpjpe, kabsch, pa_mpjpe, mean_geodesic_rotation_error, v2v,
    fingertip_weighted_mpjpe,
)


# ---------------------------------------------------------------------------
# MPJPE
# ---------------------------------------------------------------------------
def test_mpjpe_zero_on_identical():
    j = torch.randn(8, 3, dtype=torch.float64)
    assert mpjpe(j, j.clone()).item() == pytest.approx(0.0, abs=1e-12)


def test_mpjpe_root_alignment_is_translation_invariant():
    j = torch.randn(8, 3, dtype=torch.float64)
    shifted = j + torch.tensor([3.0, -1.0, 2.0], dtype=torch.float64)
    assert mpjpe(j, shifted, root=0).item() == pytest.approx(0.0, abs=1e-12)
    # without root alignment the translation shows up
    assert mpjpe(j, shifted, root=None).item() > 1.0


# ---------------------------------------------------------------------------
# Kabsch / PA-MPJPE
# ---------------------------------------------------------------------------
def test_kabsch_recovers_known_similarity():
    torch.manual_seed(0)
    A = torch.randn(20, 3, dtype=torch.float64)
    R_true = axis_angle_to_matrix(torch.tensor([0.4, -0.2, 0.7], dtype=torch.float64))
    s_true, t_true = 1.7, torch.tensor([2.0, -3.0, 1.0], dtype=torch.float64)
    B = s_true * (A @ R_true.T) + t_true

    s, R, t, A_aligned = kabsch(A, B)
    assert s.item() == pytest.approx(s_true, abs=1e-9)
    assert torch.allclose(R, R_true, atol=1e-9)
    assert torch.allclose(t, t_true, atol=1e-8)
    assert torch.allclose(A_aligned, B, atol=1e-8)
    assert torch.linalg.det(R).item() == pytest.approx(1.0, abs=1e-9)  # no reflection


def test_pa_mpjpe_zero_after_similarity_but_mpjpe_nonzero():
    torch.manual_seed(1)
    gt = torch.randn(15, 3, dtype=torch.float64)
    R = axis_angle_to_matrix(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64))
    pred = 2.0 * (gt @ R.T) + torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    assert pa_mpjpe(pred, gt).item() == pytest.approx(0.0, abs=1e-8)
    assert mpjpe(pred, gt, root=0).item() > 0.1              # position-space differs


def test_pa_mpjpe_never_exceeds_root_aligned_mpjpe():
    torch.manual_seed(2)
    gt = torch.randn(15, 3, dtype=torch.float64)
    pred = gt + 0.1 * torch.randn(15, 3, dtype=torch.float64)
    assert pa_mpjpe(pred, gt).item() <= mpjpe(pred, gt, root=None).item() + 1e-9


# ---------------------------------------------------------------------------
# geodesic rotation error
# ---------------------------------------------------------------------------
def test_geodesic_rotation_error_zero_and_known():
    R = axis_angle_to_matrix(torch.randn(6, 3, dtype=torch.float64))
    assert mean_geodesic_rotation_error(R, R).item() == pytest.approx(0.0, abs=1e-7)
    R1 = torch.eye(3, dtype=torch.float64).expand(4, 3, 3)
    R2 = axis_angle_to_matrix(torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64)).expand(4, 3, 3)
    assert mean_geodesic_rotation_error(R1, R2).item() == pytest.approx(0.5, abs=1e-9)


def test_geodesic_catches_orientation_error_invisible_to_position():
    """A joint at the same position but wrong orientation -> geodesic > 0."""
    R_gt = torch.eye(3, dtype=torch.float64)[None]
    R_pred = axis_angle_to_matrix(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))[None]
    assert mean_geodesic_rotation_error(R_pred, R_gt).item() > 0.9


# ---------------------------------------------------------------------------
# V2V
# ---------------------------------------------------------------------------
def test_v2v_zero_identical_positive_else():
    v = torch.randn(30, 3, dtype=torch.float64)
    assert v2v(v, v.clone()).item() == pytest.approx(0.0, abs=1e-12)
    assert v2v(v, v + 0.01).item() > 0.0


# ---------------------------------------------------------------------------
# fingertip-weighted error -- the point of the whole section
# ---------------------------------------------------------------------------
def test_fingertip_error_dominates_when_fingertips_move():
    J = 12
    gt = torch.randn(J, 3, dtype=torch.float64)
    fingertips = [9, 10, 11]
    # perturb ONLY the fingertips
    pred = gt.clone()
    pred[fingertips] += torch.tensor([0.02, 0.0, 0.0], dtype=torch.float64)

    plain = mpjpe(pred, gt, root=0).item()
    weighted = fingertip_weighted_mpjpe(pred, gt, fingertips,
                                        fingertip_weight=10.0, root=0).item()
    # the same displacement registers much more strongly when fingertips are
    # up-weighted -- small fingertip errors "change meaning"
    assert weighted > 3.0 * plain


def test_fingertip_weight_localises_to_fingertips():
    J = 12
    gt = torch.randn(J, 3, dtype=torch.float64)
    fingertips = [9, 10, 11]
    torso = [1, 2, 3]
    disp = torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64)

    pred_finger = gt.clone(); pred_finger[fingertips] += disp
    pred_torso = gt.clone(); pred_torso[torso] += disp

    w_finger = fingertip_weighted_mpjpe(pred_finger, gt, fingertips, 10.0, root=0)
    w_torso = fingertip_weighted_mpjpe(pred_torso, gt, fingertips, 10.0, root=0)
    # identical displacement magnitude, but on fingertips it costs far more
    assert w_finger.item() > 5.0 * w_torso.item()


def test_metric_shape_validation():
    with pytest.raises(ValueError):
        mpjpe(torch.zeros(3, 3), torch.zeros(4, 3))
    with pytest.raises(ValueError):
        pa_mpjpe(torch.zeros(3, 3, 3), torch.zeros(3, 3, 3))
