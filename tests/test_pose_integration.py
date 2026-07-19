"""Integration + whole-chain cycle stress for the 3D representation layer.

Verifies the SMPL-X -> 27-joint skeleton bridge is faithful (correct shapes,
convex-hull containment, feeds the ST-GCN layout the existing encoder expects) and
that the body model's rigid equivariance survives the linear bridge. A 150-case
stress loop checks determinism and finiteness across random motions.
"""

import pytest
import torch

from signtranslator.skeleton.graph import NUM_DEFAULT_JOINTS, SkeletonGraph, DEFAULT_EDGES
from signtranslator.pose.state import SMPLXLayout, MotionSequence
from signtranslator.pose.body_model import SMPLXBodyModel, make_toy_model, rest_pose_sequence
from signtranslator.pose.rotations import (
    rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix,
)
from signtranslator.pose.integration import (
    build_joint_map, smplx_joints_to_skeleton, motion_to_skeleton, to_stgcn_layout,
)


@pytest.fixture
def layout():
    return SMPLXLayout(n_body=6, n_hand=4, n_expr=4, n_shape=4)


@pytest.fixture
def model(layout):
    return SMPLXBodyModel(make_toy_model(layout, num_vertices=32, seed=7))


@pytest.fixture
def joint_map(layout):
    return build_joint_map(layout.n_joints, seed=1)


def _random_seq(layout, T=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return MotionSequence(
        gamma=torch.randn(T, 3, generator=g, dtype=torch.float64),
        rot6d=torch.randn(T, layout.n_joints, 6, generator=g, dtype=torch.float64),
        expr=torch.randn(T, layout.n_expr, generator=g, dtype=torch.float64),
        beta=torch.randn(layout.n_shape, generator=g, dtype=torch.float64),
        layout=layout)


# ---------------------------------------------------------------------------
# faithful bridge
# ---------------------------------------------------------------------------
def test_bridge_produces_27_joint_skeleton(model, layout, joint_map):
    seq = _random_seq(layout, T=5, seed=2)
    skel = motion_to_skeleton(model, seq, joint_map)
    assert skel.shape == (5, NUM_DEFAULT_JOINTS, 3)
    assert torch.isfinite(skel).all()


def test_joint_map_is_row_stochastic(joint_map):
    assert torch.all(joint_map >= 0)
    assert torch.allclose(joint_map.sum(-1),
                          torch.ones(NUM_DEFAULT_JOINTS, dtype=torch.float64), atol=1e-12)


def test_skeleton_joints_lie_in_convex_hull_of_smplx_joints(model, layout, joint_map):
    seq = _random_seq(layout, T=1, seed=3)
    j = model(seq).joints[0]                                  # (J, 3)
    skel = smplx_joints_to_skeleton(model(seq).joints, joint_map)[0]  # (27, 3)
    # each coordinate is a convex combo -> within the per-axis min/max box
    lo, hi = j.min(0).values, j.max(0).values
    assert torch.all(skel >= lo - 1e-9) and torch.all(skel <= hi + 1e-9)


def test_stgcn_layout_matches_encoder_expectation(model, layout, joint_map):
    seq = _random_seq(layout, T=6, seed=4)
    skel = motion_to_skeleton(model, seq, joint_map)
    x = to_stgcn_layout(skel)
    assert x.shape == (1, 3, 6, NUM_DEFAULT_JOINTS)
    # the skeleton graph is defined on exactly this many joints
    g = SkeletonGraph(NUM_DEFAULT_JOINTS, DEFAULT_EDGES)
    assert g.num_nodes == x.shape[-1]


# ---------------------------------------------------------------------------
# equivariance survives the bridge
# ---------------------------------------------------------------------------
def test_translation_equivariance_through_bridge(model, layout, joint_map):
    seq = _random_seq(layout, T=3, seed=5)
    skel1 = motion_to_skeleton(model, seq, joint_map)
    delta = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    seq2 = MotionSequence(gamma=seq.gamma + delta, rot6d=seq.rot6d, expr=seq.expr,
                          beta=seq.beta, layout=layout)
    skel2 = motion_to_skeleton(model, seq2, joint_map)
    # rows sum to 1 -> a global translation passes straight through
    assert torch.allclose(skel2, skel1 + delta, atol=1e-10)


def test_global_rotation_equivariance_through_bridge(model, layout, joint_map):
    seq = _random_seq(layout, T=2, seed=6)
    seq = MotionSequence(gamma=torch.zeros(2, 3, dtype=torch.float64), rot6d=seq.rot6d,
                         expr=seq.expr, beta=seq.beta, layout=layout)
    skel1 = motion_to_skeleton(model, seq, joint_map)

    Rg = axis_angle_to_matrix(torch.tensor([0.2, 0.4, -0.3], dtype=torch.float64))
    rot_new = seq.rot6d.clone()
    rot_new[:, 0] = matrix_to_rotation_6d(Rg @ rotation_6d_to_matrix(seq.rot6d[:, 0]))
    seq2 = MotionSequence(gamma=seq.gamma, rot6d=rot_new, expr=seq.expr, beta=seq.beta,
                          layout=layout)
    skel2 = motion_to_skeleton(model, seq2, joint_map)

    shaped = model.t.template + torch.einsum("vcs,s->vc", model.t.shape_dirs, seq.beta)
    J0 = (model.t.joint_regressor @ shaped)[0]
    expected = torch.einsum("ab,tvb->tva", Rg, skel1 - J0) + J0  # rotate about pelvis
    assert torch.allclose(skel2, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# whole-chain cycle stress
# ---------------------------------------------------------------------------
def test_cycle_stress_over_many_motions(model, layout, joint_map):
    for seed in range(150):
        seq = _random_seq(layout, T=2, seed=1000 + seed)
        skel_a = motion_to_skeleton(model, seq, joint_map)
        skel_b = motion_to_skeleton(model, seq, joint_map)
        assert torch.equal(skel_a, skel_b)                   # deterministic
        assert torch.isfinite(skel_a).all()
        x = to_stgcn_layout(skel_a)
        assert x.shape == (1, 3, 2, NUM_DEFAULT_JOINTS)
