"""Verification of wrist-relative geometry and the contact field.

Translation invariance and rotation+translation invariance are proved exactly in
float64; the contact field is proved symmetric, bounded in (0,1), and monotone
non-increasing in distance.
"""

import pytest
import torch

from signtranslator.pose.rotations import axis_angle_to_matrix, is_rotation_matrix
from signtranslator.hand_graph.hetero_graph import HAND_LANDMARKS
from signtranslator.hand_graph.geometry import (
    estimate_velocity, wrist_relative, wrist_frame_from_landmarks,
    wrist_frame_relative, ContactField, hard_contact,
)


def _hand(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(HAND_LANDMARKS, 3, generator=g, dtype=torch.float64)


# ---------------------------------------------------------------------------
# velocity
# ---------------------------------------------------------------------------
def test_estimate_velocity_finite_difference():
    x = torch.tensor([[[0.0, 0, 0]], [[1.0, 0, 0]], [[3.0, 0, 0]]], dtype=torch.float64)
    v = estimate_velocity(x, dt=1.0)
    assert torch.allclose(v[1], torch.tensor([[1.0, 0, 0]], dtype=torch.float64))
    assert torch.allclose(v[2], torch.tensor([[2.0, 0, 0]], dtype=torch.float64))
    assert torch.allclose(v[0], v[1])                        # first frame held, not fabricated


# ---------------------------------------------------------------------------
# translation invariance
# ---------------------------------------------------------------------------
def test_wrist_relative_is_translation_invariant():
    x = _hand(1)
    wrist_of = torch.zeros(HAND_LANDMARKS, dtype=torch.long)  # wrist = node 0
    t = torch.tensor([5.0, -3.0, 2.0], dtype=torch.float64)
    rel = wrist_relative(x, wrist_of)
    rel_shift = wrist_relative(x + t, wrist_of)
    assert torch.allclose(rel, rel_shift, atol=1e-12)
    assert torch.allclose(rel[0], torch.zeros(3, dtype=torch.float64))  # wrist maps to 0


# ---------------------------------------------------------------------------
# wrist frame is a valid rotation, equivariant to global rotation
# ---------------------------------------------------------------------------
def test_wrist_frame_is_special_orthogonal():
    x = _hand(2)
    R = wrist_frame_from_landmarks(x)
    assert is_rotation_matrix(R, atol=1e-10)


def test_wrist_frame_relative_is_rotation_and_translation_invariant():
    x = _hand(3)
    wrist_of = torch.zeros(HAND_LANDMARKS, dtype=torch.long)
    frame = wrist_frame_from_landmarks(x).expand(HAND_LANDMARKS, 3, 3).contiguous()
    xhat = wrist_frame_relative(x, wrist_of, frame)

    # apply a global rotation + translation
    R = axis_angle_to_matrix(torch.tensor([0.4, -0.2, 0.9], dtype=torch.float64))
    t = torch.tensor([2.0, 1.0, -1.0], dtype=torch.float64)
    x2 = x @ R.T + t
    frame2 = wrist_frame_from_landmarks(x2).expand(HAND_LANDMARKS, 3, 3).contiguous()
    xhat2 = wrist_frame_relative(x2, wrist_of, frame2)

    assert torch.allclose(xhat, xhat2, atol=1e-9)            # invariant under R, t


# ---------------------------------------------------------------------------
# contact field
# ---------------------------------------------------------------------------
def test_contact_probability_in_unit_interval():
    cf = ContactField(feat_dim=4).double()
    hi, hj = torch.randn(10, 4, dtype=torch.float64), torch.randn(10, 4, dtype=torch.float64)
    xi, xj = torch.randn(10, 3, dtype=torch.float64), torch.randn(10, 3, dtype=torch.float64)
    p = cf(hi, hj, xi, xj)
    assert torch.all(p > 0) and torch.all(p < 1)


def test_contact_is_symmetric():
    cf = ContactField(feat_dim=4).double()
    with torch.no_grad():
        cf.w_s.copy_(torch.tensor(0.7, dtype=torch.float64))
        cf.bias.copy_(torch.tensor(-0.3, dtype=torch.float64))
    hi, hj = torch.randn(6, 4, dtype=torch.float64), torch.randn(6, 4, dtype=torch.float64)
    xi, xj = torch.randn(6, 3, dtype=torch.float64), torch.randn(6, 3, dtype=torch.float64)
    vi, vj = torch.randn(6, 3, dtype=torch.float64), torch.randn(6, 3, dtype=torch.float64)
    p_ij = cf(hi, hj, xi, xj, vi, vj)
    p_ji = cf(hj, hi, xj, xi, vj, vi)
    assert torch.allclose(p_ij, p_ji, atol=1e-12)


def test_contact_is_monotone_non_increasing_in_distance():
    cf = ContactField(feat_dim=4).double()
    with torch.no_grad():
        cf.theta_d.copy_(torch.tensor(1.0, dtype=torch.float64))  # w_d = -softplus(1) < 0
    h = torch.zeros(1, 4, dtype=torch.float64)
    origin = torch.zeros(1, 3, dtype=torch.float64)
    dists = torch.linspace(0.0, 5.0, 50, dtype=torch.float64)
    ps = []
    for dd in dists:
        xj = torch.tensor([[float(dd), 0.0, 0.0]], dtype=torch.float64)
        ps.append(cf(h, h, origin, xj).detach().item())
    ps = torch.tensor(ps)
    assert torch.all(ps[1:] - ps[:-1] <= 1e-12)             # non-increasing


def test_contact_distance_gradient_sign_is_nonpositive():
    cf = ContactField(feat_dim=4).double()
    with torch.no_grad():
        cf.theta_d.copy_(torch.tensor(0.5, dtype=torch.float64))
    h = torch.zeros(1, 4, dtype=torch.float64)
    xi = torch.zeros(1, 3, dtype=torch.float64)
    xj = torch.tensor([[1.5, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    p = cf(h, h, xi, xj)
    p.sum().backward()
    # increasing distance (moving xj along +x) cannot increase contact prob
    assert float(xj.grad[0, 0]) <= 1e-12


def test_hard_contact_threshold():
    xi = torch.zeros(3, 3)
    xj = torch.tensor([[0.5, 0, 0], [2.0, 0, 0], [0.9, 0, 0]])
    lbl = hard_contact(xi, xj, rho=1.0)
    assert lbl.tolist() == [1.0, 0.0, 1.0]


def test_contact_field_gradients_flow():
    cf = ContactField(feat_dim=4).double()
    hi, hj = torch.randn(5, 4, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)
    xi, xj = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    cf(hi, hj, xi, xj).sum().backward()
    for p in cf.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
