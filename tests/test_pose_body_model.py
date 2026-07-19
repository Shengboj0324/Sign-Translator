"""Verification of the SMPL-X forward model on the controllable toy body.

Every property that does not depend on the specific learned bases is proved
exactly in float64: rest-pose identity, blendshape linearity at rest, global
rigid equivariance (rotation about the pelvis), translation equivariance,
partition-of-unity skinning, and differentiability.
"""

import pytest
import torch

from signtranslator.pose.state import SMPLXLayout, MotionSequence
from signtranslator.pose.rotations import (
    rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix,
)
from signtranslator.pose.body_model import (
    SMPLXBodyModel, make_toy_model, rest_pose_sequence,
)


@pytest.fixture
def layout():
    # a smaller layout keeps the test fast while exercising every part
    return SMPLXLayout(n_body=6, n_hand=4, n_expr=5, n_shape=5)


@pytest.fixture
def model(layout):
    return SMPLXBodyModel(make_toy_model(layout, num_vertices=40, seed=1))


def _random_seq(layout, T=3, seed=0, gamma=True):
    g = torch.Generator().manual_seed(seed)
    J = layout.n_joints
    rot6d = torch.randn(T, J, 6, generator=g, dtype=torch.float64)
    return MotionSequence(
        gamma=(torch.randn(T, 3, generator=g, dtype=torch.float64) if gamma
               else torch.zeros(T, 3, dtype=torch.float64)),
        rot6d=rot6d,
        expr=torch.randn(T, layout.n_expr, generator=g, dtype=torch.float64),
        beta=torch.randn(layout.n_shape, generator=g, dtype=torch.float64),
        layout=layout,
    )


# ---------------------------------------------------------------------------
# 1. rest-pose identity
# ---------------------------------------------------------------------------
def test_rest_pose_returns_template_and_regressed_joints(model, layout):
    seq = rest_pose_sequence(layout, T=2)
    out = model(seq)
    template = model.t.template
    assert torch.allclose(out.vertices[0], template, atol=1e-12)
    J_rest = model.t.joint_regressor @ template
    assert torch.allclose(out.joints[0], J_rest, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. blendshape linearity at rest
# ---------------------------------------------------------------------------
def test_blendshapes_are_linear_at_rest(model, layout):
    seq = rest_pose_sequence(layout, T=1)
    beta = torch.randn(layout.n_shape, dtype=torch.float64)
    expr = torch.randn(1, layout.n_expr, dtype=torch.float64)
    seq = MotionSequence(gamma=seq.gamma, rot6d=seq.rot6d, expr=expr, beta=beta,
                         layout=layout)
    out = model(seq)
    expected = (model.t.template
                + torch.einsum("vcs,s->vc", model.t.shape_dirs, beta)
                + torch.einsum("vce,e->vc", model.t.expr_dirs, expr[0]))
    assert torch.allclose(out.vertices[0], expected, atol=1e-12)


def test_shape_blendshape_superposition(model, layout):
    """M(beta1+beta2) - template = [M(beta1)-t] + [M(beta2)-t] at rest (linearity)."""
    t = model.t.template
    b1 = torch.randn(layout.n_shape, dtype=torch.float64)
    b2 = torch.randn(layout.n_shape, dtype=torch.float64)

    def rest_verts(beta):
        s = rest_pose_sequence(layout, T=1)
        s = MotionSequence(gamma=s.gamma, rot6d=s.rot6d, expr=s.expr, beta=beta,
                           layout=layout)
        return model(s).vertices[0]

    lhs = rest_verts(b1 + b2) - t
    rhs = (rest_verts(b1) - t) + (rest_verts(b2) - t)
    assert torch.allclose(lhs, rhs, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. global rigid equivariance: prepending R_g to global orient rotates the
#    whole body about the pelvis J_0
# ---------------------------------------------------------------------------
def test_global_orient_is_rotation_about_pelvis(model, layout):
    seq = _random_seq(layout, T=2, seed=3, gamma=False)
    out1 = model(seq)

    Rg = axis_angle_to_matrix(torch.tensor([0.3, -0.7, 0.2], dtype=torch.float64))
    R0 = rotation_6d_to_matrix(seq.rot6d[:, 0])               # (T,3,3)
    R0_new = Rg @ R0
    rot6d_new = seq.rot6d.clone()
    rot6d_new[:, 0] = matrix_to_rotation_6d(R0_new)
    seq2 = MotionSequence(gamma=seq.gamma, rot6d=rot6d_new, expr=seq.expr,
                          beta=seq.beta, layout=layout)
    out2 = model(seq2)

    # pelvis rest location J_0 (depends on beta only)
    shaped = model.t.template + torch.einsum("vcs,s->vc", model.t.shape_dirs, seq.beta)
    J0 = (model.t.joint_regressor @ shaped)[0]               # (3,)

    def rot_about(x):
        return torch.einsum("ab,tvb->tva", Rg, x - J0) + J0

    assert torch.allclose(out2.vertices, rot_about(out1.vertices), atol=1e-10)
    assert torch.allclose(out2.joints, rot_about(out1.joints), atol=1e-10)


# ---------------------------------------------------------------------------
# 4. translation equivariance
# ---------------------------------------------------------------------------
def test_gamma_translates_everything(model, layout):
    seq = _random_seq(layout, T=3, seed=4)
    out1 = model(seq)
    delta = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    seq2 = MotionSequence(gamma=seq.gamma + delta, rot6d=seq.rot6d, expr=seq.expr,
                          beta=seq.beta, layout=layout)
    out2 = model(seq2)
    assert torch.allclose(out2.vertices, out1.vertices + delta, atol=1e-11)
    assert torch.allclose(out2.joints, out1.joints + delta, atol=1e-11)


# ---------------------------------------------------------------------------
# 5. partition of unity
# ---------------------------------------------------------------------------
def test_skinning_weights_are_partition_of_unity(model):
    w = model.t.weights
    assert torch.all(w >= 0)
    assert torch.allclose(w.sum(-1), torch.ones(w.shape[0], dtype=torch.float64),
                          atol=1e-12)


def test_joint_regressor_rows_sum_to_one(model):
    jr = model.t.joint_regressor
    assert torch.allclose(jr.sum(-1), torch.ones(jr.shape[0], dtype=torch.float64),
                          atol=1e-12)


# ---------------------------------------------------------------------------
# 6. differentiability
# ---------------------------------------------------------------------------
def test_gradients_flow_to_all_motion_and_identity(model, layout):
    seq = _random_seq(layout, T=2, seed=5)
    seq.rot6d.requires_grad_(True)
    seq.beta.requires_grad_(True)
    seq.expr.requires_grad_(True)
    seq.gamma.requires_grad_(True)
    out = model(seq)
    out.vertices.pow(2).sum().backward()
    for name, tns in (("rot6d", seq.rot6d), ("beta", seq.beta),
                      ("expr", seq.expr), ("gamma", seq.gamma)):
        assert tns.grad is not None, name
        assert torch.isfinite(tns.grad).all(), name
        assert tns.grad.abs().sum() > 0, name


# ---------------------------------------------------------------------------
# 7. pose correctives do not depend on global orientation
# ---------------------------------------------------------------------------
def test_pose_correctives_exclude_root(model, layout):
    """The pose-corrective feature must be invariant to the global orientation:
    it is built from joints 1..J-1 only. This is the precise design fact that
    makes global-rigid-equivariance hold (a change of global orient injects no
    extra pose-blendshape deformation). Tested directly on the feature, not via
    a cdist round trip whose sqrt amplifies float noise near zero."""
    seq = _random_seq(layout, T=1, seed=6, gamma=False)
    Rg = axis_angle_to_matrix(torch.tensor([0.5, 0.1, -0.4], dtype=torch.float64))
    rot6d_new = seq.rot6d.clone()
    rot6d_new[:, 0] = matrix_to_rotation_6d(Rg @ rotation_6d_to_matrix(seq.rot6d[:, 0]))

    feat1 = model._rot_features_no_root(rotation_6d_to_matrix(seq.rot6d))
    feat2 = model._rot_features_no_root(rotation_6d_to_matrix(rot6d_new))
    assert torch.equal(feat1, feat2)                         # root change invisible

    # and a change to a NON-root joint DOES change the feature (guard: not vacuous)
    rot6d_body = seq.rot6d.clone()
    rot6d_body[:, 1] = matrix_to_rotation_6d(Rg @ rotation_6d_to_matrix(seq.rot6d[:, 1]))
    feat3 = model._rot_features_no_root(rotation_6d_to_matrix(rot6d_body))
    assert not torch.equal(feat1, feat3)
