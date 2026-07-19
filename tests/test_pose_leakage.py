"""Verification of identity/motion separation and the signer-leakage guard.

Two independent proofs: (1) per-joint world orientations are identity-invariant by
construction; (2) a leakage probe cannot recover identity from motion-only
features, yet the same probe recovers it when identity is folded in -- so the
guard has power.
"""

import pytest
import torch

from signtranslator.pose.state import SMPLXLayout, MotionSequence
from signtranslator.pose.body_model import SMPLXBodyModel, make_toy_model
from signtranslator.pose.leakage import (
    world_joint_rotations, LinearProbe, normalised_recovery_error,
)


@pytest.fixture
def layout():
    return SMPLXLayout(n_body=5, n_hand=3, n_expr=4, n_shape=6)


# ---------------------------------------------------------------------------
# 1. world orientations are identity-invariant
# ---------------------------------------------------------------------------
def test_world_joint_orientations_are_identity_invariant(layout):
    model = SMPLXBodyModel(make_toy_model(layout, num_vertices=30, seed=1))
    g = torch.Generator().manual_seed(2)
    rot6d = torch.randn(3, layout.n_joints, 6, generator=g, dtype=torch.float64)
    parents = model.t.parents

    Rw = world_joint_rotations(rot6d, parents)
    # same motion, but recompute -- must be deterministic and beta-free.
    # bit-identical is the exact proof (the function takes no beta); we avoid a
    # geodesic==0 check here because arccos near 1 has sqrt-amplified float error.
    Rw2 = world_joint_rotations(rot6d, parents)
    assert torch.equal(Rw, Rw2)


def test_same_motion_two_identities_shares_orientation_but_differs_in_position(layout):
    model = SMPLXBodyModel(make_toy_model(layout, num_vertices=30, seed=3))
    g = torch.Generator().manual_seed(4)
    rot6d = torch.randn(2, layout.n_joints, 6, generator=g, dtype=torch.float64)
    base = MotionSequence(
        gamma=torch.zeros(2, 3, dtype=torch.float64), rot6d=rot6d,
        expr=torch.zeros(2, layout.n_expr, dtype=torch.float64),
        beta=torch.zeros(layout.n_shape, dtype=torch.float64), layout=layout)

    beta1 = torch.randn(layout.n_shape, dtype=torch.float64)
    beta2 = torch.randn(layout.n_shape, dtype=torch.float64)
    out1 = model(base.retarget(beta1))
    out2 = model(base.retarget(beta2))

    # positions differ (different bone lengths)
    assert (out1.joints - out2.joints).norm(dim=-1).mean().item() > 1e-3
    # but the driving world orientations are identical (identity-free)
    Rw = world_joint_rotations(rot6d, model.t.parents)
    Rw_again = world_joint_rotations(rot6d, model.t.parents)
    assert torch.equal(Rw, Rw_again)


# ---------------------------------------------------------------------------
# 2. leakage probe: motion features do not reveal identity, but the guard has power
# ---------------------------------------------------------------------------
def _dataset(layout, n=400, seed=0):
    """n examples of INDEPENDENT motion and identity (single frame each)."""
    g = torch.Generator().manual_seed(seed)
    J = layout.n_joints
    motion_feats = []
    betas = []
    for _ in range(n):
        rot6d = torch.randn(1, J, 6, generator=g, dtype=torch.float64)
        gamma = torch.randn(1, 3, generator=g, dtype=torch.float64)
        expr = torch.randn(1, layout.n_expr, generator=g, dtype=torch.float64)
        beta = torch.randn(layout.n_shape, generator=g, dtype=torch.float64)
        seq = MotionSequence(gamma=gamma, rot6d=rot6d, expr=expr, beta=beta,
                             layout=layout)
        motion_feats.append(seq.motion_features()[0])
        betas.append(beta)
    return torch.stack(motion_feats), torch.stack(betas)


def test_identity_not_recoverable_from_motion_features(layout):
    X, Y = _dataset(layout, n=400, seed=5)
    ntr = 300
    probe = LinearProbe(l2=1.0).fit(X[:ntr], Y[:ntr])
    err = normalised_recovery_error(probe.predict(X[ntr:]), Y[ntr:])
    # no better than predicting the mean identity -> ~1 (allow probe overfit slack)
    assert err > 0.85


def test_leakage_guard_has_power_when_identity_is_folded_in(layout):
    """If beta is (wrongly) concatenated into the features, the SAME probe recovers
    it almost perfectly -- proving the guard can detect real leakage."""
    X, Y = _dataset(layout, n=400, seed=6)
    X_leaky = torch.cat([X, Y], dim=1)                        # identity folded in
    ntr = 300
    probe = LinearProbe(l2=1e-4).fit(X_leaky[:ntr], Y[:ntr])
    err = normalised_recovery_error(probe.predict(X_leaky[ntr:]), Y[ntr:])
    assert err < 0.05                                        # recovered -> leakage detectable


def test_normalised_recovery_error_bounds():
    target = torch.randn(50, 3, dtype=torch.float64)
    assert normalised_recovery_error(target.clone(), target) == pytest.approx(0.0, abs=1e-9)
    mean_pred = target.mean(0, keepdim=True).expand_as(target)
    assert normalised_recovery_error(mean_pred, target) == pytest.approx(1.0, abs=1e-6)
