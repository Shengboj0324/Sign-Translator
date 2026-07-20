"""Verification of dual-quaternion skinning.

Proves the transform<->DQ round-trip, that a DQ applies the correct rigid
transform, DLB reproduces a single transform, unit-norm, antipodality handling,
and the decisive property: DLB preserves length where LBS collapses it.
"""

import math

import pytest
import torch

from signtranslator.pose.rotations import axis_angle_to_matrix
from signtranslator.avatar_render.dqs import (
    quat_mul, quat_conj, dq_from_transform, transform_from_dq, apply_dq_to_point,
    dq_normalize, dlb,
)


def _R(aa):
    return axis_angle_to_matrix(torch.tensor(aa, dtype=torch.float64))


# ---------------------------------------------------------------------------
# quaternion algebra
# ---------------------------------------------------------------------------
def test_quat_mul_identity():
    q = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(quat_mul(q, ident), q, atol=1e-12)
    assert torch.allclose(quat_mul(ident, q), q, atol=1e-12)


# ---------------------------------------------------------------------------
# transform <-> DQ round trip
# ---------------------------------------------------------------------------
def test_transform_dq_round_trip():
    R = _R([0.4, -0.7, 0.3])
    t = torch.tensor([1.5, -2.0, 0.5], dtype=torch.float64)
    dq = dq_from_transform(R, t)
    R2, t2 = transform_from_dq(dq)
    assert torch.allclose(R2, R, atol=1e-9)
    assert torch.allclose(t2, t, atol=1e-9)


def test_dq_applies_correct_rigid_transform():
    R = _R([0.2, 0.5, -0.3])
    t = torch.tensor([3.0, 1.0, -1.0], dtype=torch.float64)
    dq = dq_from_transform(R, t)
    p = torch.randn(3, dtype=torch.float64)
    assert torch.allclose(apply_dq_to_point(dq, p), R @ p + t, atol=1e-9)


def test_dq_is_unit_after_normalize():
    R = _R([1.0, 0.2, 0.3])
    t = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    dq = dq_normalize(dq_from_transform(R, t) * 3.7)         # scale then normalise
    assert abs(float(torch.linalg.norm(dq[:4])) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# DLB
# ---------------------------------------------------------------------------
def test_dlb_single_weight_reproduces_transform():
    R = _R([0.6, -0.2, 0.4])
    t = torch.tensor([2.0, 0.0, 1.0], dtype=torch.float64)
    dq = dq_from_transform(R, t)
    blended = dlb(torch.tensor([1.0], dtype=torch.float64), dq.unsqueeze(0))
    R2, t2 = transform_from_dq(blended)
    assert torch.allclose(R2, R, atol=1e-9) and torch.allclose(t2, t, atol=1e-9)


def test_dlb_of_identical_transforms_is_that_transform():
    R = _R([0.3, 0.3, 0.3]); t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    dq = dq_from_transform(R, t)
    blended = dlb(torch.tensor([0.5, 0.5], dtype=torch.float64),
                  torch.stack([dq, dq]))
    R2, t2 = transform_from_dq(blended)
    assert torch.allclose(R2, R, atol=1e-9) and torch.allclose(t2, t, atol=1e-9)


def test_dlb_handles_antipodal_quaternions():
    """q̂ and −q̂ are the same transform; DLB must sign-align so they don't cancel."""
    R = _R([0.5, 0.5, 0.5]); t = torch.zeros(3, dtype=torch.float64)
    dq = dq_from_transform(R, t)
    blended = dlb(torch.tensor([0.5, 0.5], dtype=torch.float64),
                  torch.stack([dq, -dq]))                    # one antipodal copy
    R2, _ = transform_from_dq(blended)
    assert torch.allclose(R2, R, atol=1e-9)                 # recovers R, no cancellation


# ---------------------------------------------------------------------------
# THE property: DQS preserves length where LBS collapses it
# ---------------------------------------------------------------------------
def test_dqs_preserves_length_where_lbs_collapses():
    # two rotations about z: 0 and 160 degrees (a large twist), no translation
    R0 = torch.eye(3, dtype=torch.float64)
    R1 = _R([0.0, 0.0, math.radians(160.0)])
    p = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)   # unit vector to skin

    # LBS: linearly average the transform matrices -> shrinks (candy-wrapper)
    M_lbs = 0.5 * R0 + 0.5 * R1
    lbs_len = float((M_lbs @ p).norm())
    assert lbs_len < 0.5                                      # dramatic volume collapse

    # DQS: blend on the rotation manifold -> a proper ~80-degree rotation
    dq0 = dq_from_transform(R0, torch.zeros(3, dtype=torch.float64))
    dq1 = dq_from_transform(R1, torch.zeros(3, dtype=torch.float64))
    blended = dlb(torch.tensor([0.5, 0.5], dtype=torch.float64), torch.stack([dq0, dq1]))
    R_dqs, _ = transform_from_dq(blended)
    dqs_len = float((R_dqs @ p).norm())
    assert abs(dqs_len - 1.0) < 1e-9                          # length preserved exactly
    # the DQS blend is a proper rotation (det +1)
    assert abs(float(torch.linalg.det(R_dqs)) - 1.0) < 1e-9
