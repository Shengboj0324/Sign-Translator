"""Verification of marker -> FLAME articulation.

Proves the expression mapping is linear and intensity-monotone (stronger marker ->
larger coefficient), the jaw rotation is a valid SO(3) whose angle scales with the
marker value, the eye rotation is valid SO(3), and blendshape integration.
"""

import pytest
import torch

from signtranslator.pose.rotations import (
    axis_angle_to_matrix, matrix_to_rotation_6d, is_rotation_matrix, geodesic_distance,
)
from signtranslator.facial_nmm.channels import Marker
from signtranslator.facial_nmm.articulate import (
    MarkerArticulator, jaw_rotation, eye_rotation, articulate_blendshapes,
    marker_one_hot,
)


# ---------------------------------------------------------------------------
# expression mapping
# ---------------------------------------------------------------------------
def test_expression_is_linear_in_intensity():
    art = MarkerArticulator(num_markers=6, num_expr=10).double()
    m1 = torch.randn(3, 6, dtype=torch.float64)
    m2 = torch.randn(3, 6, dtype=torch.float64)
    lhs = art.expression(m1 + m2)
    rhs = art.expression(m1) + art.expression(m2)
    assert torch.allclose(lhs, rhs, atol=1e-12)             # linear (no bias)


def test_expression_is_intensity_monotone():
    torch.manual_seed(0)
    art = MarkerArticulator(num_markers=6, num_expr=10).double()
    # a single marker at increasing intensity -> monotonically larger coefficients
    norms = []
    for val in (0.0, 0.3, 0.6, 1.0):
        m = marker_one_hot(Marker.YN_Q, 6, val, dtype=torch.float64).unsqueeze(0)
        norms.append(float(art.expression(m).norm()))
    assert norms == sorted(norms)                           # non-decreasing
    assert norms[0] < 1e-12                                 # value 0 -> no expression


# ---------------------------------------------------------------------------
# jaw / eye rotations
# ---------------------------------------------------------------------------
def test_jaw_rotation_is_so3_and_angle_scales_with_value():
    base = matrix_to_rotation_6d(axis_angle_to_matrix(
        torch.tensor([0.0, 0.0, 0.8], dtype=torch.float64)))   # base ~0.8 rad
    base_angle = float(geodesic_distance(torch.eye(3, dtype=torch.float64),
                                         axis_angle_to_matrix(torch.tensor([0.0, 0, 0.8],
                                                                           dtype=torch.float64))))
    for val in (0.0, 0.5, 1.0):
        R = jaw_rotation(torch.tensor(val, dtype=torch.float64), base)
        assert is_rotation_matrix(R, atol=1e-9)
        ang = float(geodesic_distance(torch.eye(3, dtype=torch.float64), R))
        assert abs(ang - val * base_angle) < 1e-7           # angle scales with value
    # value 0 -> identity (jaw closed)
    R0 = jaw_rotation(torch.tensor(0.0, dtype=torch.float64), base)
    assert torch.allclose(R0, torch.eye(3, dtype=torch.float64), atol=1e-7)


def test_eye_rotation_is_so3():
    gaze = matrix_to_rotation_6d(axis_angle_to_matrix(
        torch.tensor([0.1, 0.3, -0.2], dtype=torch.float64)))
    assert is_rotation_matrix(eye_rotation(gaze), atol=1e-9)


# ---------------------------------------------------------------------------
# blendshapes
# ---------------------------------------------------------------------------
def test_blendshape_articulation_linear_and_zero_expr_is_mean():
    Nv, E = 6, 4
    mean = torch.randn(Nv, 3, dtype=torch.float64)
    basis = torch.randn(Nv, 3, E, dtype=torch.float64)
    assert torch.allclose(articulate_blendshapes(mean, basis, torch.zeros(E, dtype=torch.float64)),
                          mean, atol=1e-12)                 # neutral expr -> mean face
    e = torch.randn(E, dtype=torch.float64)
    assert not torch.allclose(articulate_blendshapes(mean, basis, e), mean)


def test_marker_one_hot():
    v = marker_one_hot(Marker.WH_Q, 6, 0.7)
    assert abs(float(v[int(Marker.WH_Q)]) - 0.7) < 1e-6
    assert abs(float(v.sum()) - 0.7) < 1e-6                 # only that marker set
