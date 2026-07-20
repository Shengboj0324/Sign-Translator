"""Verification of the rigged-mesh track (LBS + retargeting + blendshapes).

Proves LBS rest identity and rigid equivariance, the handedness certificate
(alignment is always a proper rotation and a mirror-imaged target is refused),
blendshape linearity, and finger/wrist priority ordering.
"""

import pytest
import torch

from signtranslator.pose.rotations import axis_angle_to_matrix
from signtranslator.avatar_render.rigged import (
    apply_lbs, RetargetMap, build_retarget, retarget_residual, prioritized_joints,
    correct_joint_limits, apply_blendshapes,
)


def _rigid(R, t):
    M = torch.eye(4, dtype=torch.float64)
    M[:3, :3] = R; M[:3, 3] = t
    return M


# ---------------------------------------------------------------------------
# LBS
# ---------------------------------------------------------------------------
def test_lbs_rest_identity():
    v = torch.randn(5, 3, dtype=torch.float64)
    w = torch.softmax(torch.randn(5, 2, dtype=torch.float64), dim=-1)
    g_rel = torch.eye(4, dtype=torch.float64).expand(2, 4, 4)
    assert torch.allclose(apply_lbs(v, w, g_rel), v, atol=1e-12)


def test_lbs_applies_shared_rigid_transform_once():
    v = torch.randn(5, 3, dtype=torch.float64)
    w = torch.softmax(torch.randn(5, 2, dtype=torch.float64), dim=-1)   # partition of unity
    R = axis_angle_to_matrix(torch.tensor([0.3, -0.5, 0.2], dtype=torch.float64))
    t = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    g_rel = _rigid(R, t).expand(2, 4, 4)
    out = apply_lbs(v, w, g_rel)
    assert torch.allclose(out, v @ R.T + t, atol=1e-10)     # applied exactly once


# ---------------------------------------------------------------------------
# handedness-certified retargeting
# ---------------------------------------------------------------------------
def test_retarget_recovers_known_rotation_with_det_plus_one():
    torch.manual_seed(0)
    src = torch.randn(6, 3, dtype=torch.float64)
    R_true = axis_angle_to_matrix(torch.tensor([0.4, 0.1, -0.6], dtype=torch.float64))
    tgt = src @ R_true.T + torch.tensor([2.0, 0.0, 1.0], dtype=torch.float64)
    corr = {i: i for i in range(6)}
    rmap = build_retarget(src, tgt, corr)
    assert abs(float(torch.linalg.det(rmap.align)) - 1.0) < 1e-9
    assert rmap.preserves_handedness
    assert retarget_residual(rmap, src, tgt) < 1e-9          # fits a proper rotation


def test_retarget_refuses_to_mirror():
    """A reflected (mirror-imaged) target cannot be fit by a proper rotation, so the
    certified retarget leaves a large residual -- the mirroring error is refused,
    not silently produced."""
    torch.manual_seed(1)
    src = torch.randn(6, 3, dtype=torch.float64)
    tgt = src.clone(); tgt[:, 0] = -tgt[:, 0]                # reflect across x (mirror)
    corr = {i: i for i in range(6)}
    rmap = build_retarget(src, tgt, corr)
    assert abs(float(torch.linalg.det(rmap.align)) - 1.0) < 1e-9   # still proper
    assert retarget_residual(rmap, src, tgt) > 0.5          # cannot mirror -> big error


def test_retarget_map_rejects_reflection_alignment():
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))  # det -1
    with pytest.raises(ValueError):
        RetargetMap(correspondence={0: 0}, align=reflection)


def test_prioritized_joints_puts_fingers_and_wrist_first():
    importance = {0: "TORSO", 1: "FINGERS", 2: "ARMS", 3: "WRIST", 4: "FACE"}
    order = prioritized_joints(importance)
    assert order[0] == 1 and order[1] == 3                   # FINGERS, WRIST first
    assert order[-1] == 0                                    # TORSO last


# ---------------------------------------------------------------------------
# joint limits + blendshapes
# ---------------------------------------------------------------------------
def test_joint_limit_correction_feasible():
    angles = torch.tensor([2.0, -3.0, 0.5], dtype=torch.float64)
    out = correct_joint_limits(angles, theta_max=1.0)
    assert torch.all(out.abs() <= 1.0 + 1e-12)


def test_blendshapes_are_linear():
    Nv, E = 8, 4
    mean = torch.randn(Nv, 3, dtype=torch.float64)
    basis = torch.randn(Nv, 3, E, dtype=torch.float64)
    e1 = torch.randn(E, dtype=torch.float64)
    e2 = torch.randn(E, dtype=torch.float64)
    lhs = apply_blendshapes(mean, basis, e1 + e2) - mean
    rhs = (apply_blendshapes(mean, basis, e1) - mean) + (apply_blendshapes(mean, basis, e2) - mean)
    assert torch.allclose(lhs, rhs, atol=1e-12)             # affine in expression
    assert torch.allclose(apply_blendshapes(mean, basis, torch.zeros(E, dtype=torch.float64)),
                          mean, atol=1e-12)                  # zero expr -> mean face
