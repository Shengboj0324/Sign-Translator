"""Verification of rotation representations and conversions.

Every produced matrix must be special-orthogonal; every conversion must
round-trip; the 6D encoding must be continuous exactly where the canonical
quaternion jumps; gradients must flow. float64 is used for the exact-identity
tests so the tolerance measures the algebra, not float32 noise.
"""

import math

import pytest
import torch

from signtranslator.pose.rotations import (
    rotation_6d_to_matrix, matrix_to_rotation_6d,
    axis_angle_to_matrix, matrix_to_axis_angle,
    quaternion_to_matrix, matrix_to_quaternion,
    geodesic_distance, is_rotation_matrix,
)


def _random_matrices(n, seed=0, dtype=torch.float64):
    """n uniformly-random rotations via QR of random Gaussians (det fixed to +1)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, 3, 3, generator=g, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    # make R's diagonal positive so Q is a proper, deterministic rotation
    d = torch.diagonal(R, dim1=-2, dim2=-1)
    Q = Q * torch.sign(d)[..., None, :]
    # ensure det = +1 (flip one column if it is -1)
    det = torch.linalg.det(Q)
    Q[..., :, 0] = Q[..., :, 0] * torch.sign(det)[..., None]
    return Q


# ---------------------------------------------------------------------------
# 6D <-> matrix
# ---------------------------------------------------------------------------
def test_6d_to_matrix_is_special_orthogonal():
    torch.manual_seed(1)
    d6 = torch.randn(256, 6, dtype=torch.float64)
    R = rotation_6d_to_matrix(d6)
    assert is_rotation_matrix(R, atol=1e-10).all()


def test_matrix_to_6d_to_matrix_round_trips_exactly():
    R = _random_matrices(256, seed=2)
    back = rotation_6d_to_matrix(matrix_to_rotation_6d(R))
    assert torch.allclose(back, R, atol=1e-10)


def test_6d_gram_schmidt_is_identity_on_orthonormal_pair():
    """Feeding two already-orthonormal columns back must not change them."""
    R = _random_matrices(64, seed=3)
    d6 = matrix_to_rotation_6d(R)
    R2 = rotation_6d_to_matrix(d6)
    assert torch.allclose(R2[..., :, 0], R[..., :, 0], atol=1e-10)
    assert torch.allclose(R2[..., :, 1], R[..., :, 1], atol=1e-10)


# ---------------------------------------------------------------------------
# axis-angle <-> matrix
# ---------------------------------------------------------------------------
def test_axis_angle_known_rotation_90deg_z():
    aa = torch.tensor([0.0, 0.0, math.pi / 2], dtype=torch.float64)
    R = axis_angle_to_matrix(aa)
    expected = torch.tensor([[0.0, -1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(R, expected, atol=1e-12)


def test_axis_angle_zero_is_identity_with_finite_gradient():
    aa = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    R = axis_angle_to_matrix(aa)
    assert torch.allclose(R, torch.eye(3, dtype=torch.float64), atol=1e-12)
    R.sum().backward()
    assert torch.isfinite(aa.grad).all()


def test_axis_angle_round_trip_generic():
    torch.manual_seed(4)
    # angles bounded away from 0 and pi so the log map is in its generic branch
    axis = torch.nn.functional.normalize(torch.randn(200, 3, dtype=torch.float64), dim=-1)
    angle = torch.empty(200, 1, dtype=torch.float64).uniform_(0.2, math.pi - 0.2)
    aa = axis * angle
    back = matrix_to_axis_angle(axis_angle_to_matrix(aa))
    assert torch.allclose(back, aa, atol=1e-9)


def test_axis_angle_round_trip_near_pi():
    """The phi ~ pi branch (2 sin phi -> 0) must still recover the rotation."""
    torch.manual_seed(5)
    axis = torch.nn.functional.normalize(torch.randn(50, 3, dtype=torch.float64), dim=-1)
    aa = axis * (math.pi - 1e-9)
    R = axis_angle_to_matrix(aa)
    aa_back = matrix_to_axis_angle(R)
    # recovered aa may differ by axis sign (pi and -pi rotations coincide); compare
    # via the rotation it induces, which must match
    assert torch.allclose(axis_angle_to_matrix(aa_back), R, atol=1e-6)


def test_matrix_to_axis_angle_matches_geodesic_angle():
    R = _random_matrices(128, seed=6)
    aa = matrix_to_axis_angle(R)
    ang = torch.linalg.norm(aa, dim=-1)
    geo = geodesic_distance(torch.eye(3, dtype=torch.float64).expand(R.shape), R)
    assert torch.allclose(ang, geo, atol=1e-9)


# ---------------------------------------------------------------------------
# quaternion <-> matrix
# ---------------------------------------------------------------------------
def test_quaternion_to_matrix_is_special_orthogonal():
    torch.manual_seed(7)
    q = torch.randn(256, 4, dtype=torch.float64)
    R = quaternion_to_matrix(q)
    assert is_rotation_matrix(R, atol=1e-10).all()


def test_matrix_to_quaternion_round_trips():
    R = _random_matrices(256, seed=8)
    q = matrix_to_quaternion(R)
    R2 = quaternion_to_matrix(q)
    assert torch.allclose(R2, R, atol=1e-9)


def test_quaternion_double_cover_canonicalised():
    """q and -q map to the same R; matrix_to_quaternion returns the w>=0 rep."""
    torch.manual_seed(9)
    q = torch.nn.functional.normalize(torch.randn(64, 4, dtype=torch.float64), dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)                    # canonical target
    R = quaternion_to_matrix(q)
    q_back = matrix_to_quaternion(R)
    assert (q_back[..., 0] >= 0).all()
    assert torch.allclose(q_back, q, atol=1e-9)


# ---------------------------------------------------------------------------
# geodesic metric
# ---------------------------------------------------------------------------
def test_geodesic_zero_iff_equal_and_symmetric():
    R = _random_matrices(64, seed=10)
    assert torch.allclose(geodesic_distance(R, R), torch.zeros(64, dtype=torch.float64),
                          atol=1e-7)
    S = _random_matrices(64, seed=11)
    assert torch.allclose(geodesic_distance(R, S), geodesic_distance(S, R), atol=1e-12)


def test_geodesic_known_angle():
    R1 = torch.eye(3, dtype=torch.float64)
    R2 = axis_angle_to_matrix(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))  # 1 rad about z
    assert abs(geodesic_distance(R1, R2).item() - 1.0) < 1e-12


def test_geodesic_triangle_inequality():
    R = _random_matrices(64, seed=12)
    S = _random_matrices(64, seed=13)
    T = _random_matrices(64, seed=14)
    d_rt = geodesic_distance(R, T)
    d_rs = geodesic_distance(R, S)
    d_st = geodesic_distance(S, T)
    assert (d_rt <= d_rs + d_st + 1e-9).all()


# ---------------------------------------------------------------------------
# THE continuity claim (Zhou et al.): 6D is continuous where quaternion jumps
# ---------------------------------------------------------------------------
def test_6d_is_continuous_where_canonical_quaternion_jumps():
    # sweep a rotation about z through angle = pi, where cos(theta/2) changes sign
    thetas = torch.linspace(math.pi - 0.1, math.pi + 0.1, 401, dtype=torch.float64)
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    aa = axis[None, :] * thetas[:, None]
    R = axis_angle_to_matrix(aa)

    d6 = matrix_to_rotation_6d(R)
    q = matrix_to_quaternion(R)                               # canonical w >= 0

    d6_jumps = torch.linalg.norm(d6[1:] - d6[:-1], dim=-1).max().item()
    q_jumps = torch.linalg.norm(q[1:] - q[:-1], dim=-1).max().item()

    step = (thetas[1] - thetas[0]).item()
    # 6D varies smoothly: adjacent change is O(step)
    assert d6_jumps < 10 * step
    # the canonical quaternion has an O(1) discontinuity at theta = pi
    assert q_jumps > 1.0
    assert q_jumps > 100 * d6_jumps


# ---------------------------------------------------------------------------
# differentiability through the representation used for regression
# ---------------------------------------------------------------------------
def test_gradients_flow_through_6d_to_matrix():
    d6 = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
    R = rotation_6d_to_matrix(d6)
    target = _random_matrices(8, seed=15)
    loss = geodesic_distance(R, target).sum()
    loss.backward()
    assert d6.grad is not None and torch.isfinite(d6.grad).all()
    assert d6.grad.abs().sum() > 0


def test_input_validation():
    with pytest.raises(ValueError):
        rotation_6d_to_matrix(torch.zeros(5))
    with pytest.raises(ValueError):
        axis_angle_to_matrix(torch.zeros(4))
    with pytest.raises(ValueError):
        quaternion_to_matrix(torch.zeros(3))
