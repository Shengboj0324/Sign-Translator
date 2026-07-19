"""Verification of the SMPL-X representation contract.

The linguistically-critical property is identity/motion separation: the motion
feature vector must not contain identity, and retargeting to a new identity must
leave motion bit-identical.
"""

import pytest
import torch

from signtranslator.pose.state import SMPLXLayout, MotionSequence, ROT_DIM


def _seq(T=4, layout=None, seed=0):
    layout = layout or SMPLXLayout()
    g = torch.Generator().manual_seed(seed)
    return MotionSequence(
        gamma=torch.randn(T, 3, generator=g),
        rot6d=torch.randn(T, layout.n_joints, ROT_DIM, generator=g),
        expr=torch.randn(T, layout.n_expr, generator=g),
        beta=torch.randn(layout.n_shape, generator=g),
        layout=layout,
    )


def test_layout_joint_count_and_slices_partition():
    L = SMPLXLayout()
    assert L.n_joints == 22 + 1 + 2 + 15 + 15                 # 55
    slices = L.part_slices()
    covered = []
    for name in ("global_orient", "body", "jaw", "eye", "lhand", "rhand"):
        s = slices[name]
        covered.extend(range(s.start, s.stop))
    assert covered == list(range(L.n_joints))                 # exact, ordered partition


def test_motion_dim_matches_feature_length():
    L = SMPLXLayout()
    seq = _seq(T=3, layout=L)
    assert seq.motion_features().shape == (3, L.motion_dim)


def test_motion_features_round_trip():
    seq = _seq(T=5)
    feats = seq.motion_features()
    rebuilt = MotionSequence.from_motion_features(feats, seq.beta, seq.layout)
    assert torch.equal(rebuilt.gamma, seq.gamma)
    assert torch.equal(rebuilt.rot6d, seq.rot6d)
    assert torch.equal(rebuilt.expr, seq.expr)


def test_motion_features_exclude_identity():
    """beta must NOT be recoverable from the motion feature vector."""
    seq = _seq(T=6, seed=1)
    feats = seq.motion_features()
    # two sequences with identical motion but different beta produce identical feats
    other = seq.retarget(seq.beta + 3.0)
    assert torch.equal(feats, other.motion_features())        # identity invisible


def test_retarget_preserves_motion_bit_identically():
    seq = _seq(T=4, seed=2)
    new = seq.retarget(torch.zeros(seq.layout.n_shape))
    assert seq.motion_equal(new)
    assert not torch.equal(seq.beta, new.beta)                # identity changed


def test_part_accessor_shapes():
    L = SMPLXLayout()
    seq = _seq(T=3, layout=L)
    assert seq.part("global_orient").shape == (3, 1, ROT_DIM)
    assert seq.part("body").shape == (3, L.n_body - 1, ROT_DIM)
    assert seq.part("lhand").shape == (3, L.n_hand, ROT_DIM)
    assert seq.part("rhand").shape == (3, L.n_hand, ROT_DIM)
    assert seq.part("jaw").shape == (3, 1, ROT_DIM)
    assert seq.part("eye").shape == (3, 2, ROT_DIM)


def test_shape_validation_rejects_bad_tensors():
    L = SMPLXLayout()
    with pytest.raises(ValueError):
        MotionSequence(gamma=torch.zeros(4, 2), rot6d=torch.zeros(4, L.n_joints, 6),
                       expr=torch.zeros(4, L.n_expr), beta=torch.zeros(L.n_shape))
    with pytest.raises(ValueError):
        MotionSequence(gamma=torch.zeros(4, 3), rot6d=torch.zeros(4, L.n_joints, 6),
                       expr=torch.zeros(4, L.n_expr), beta=torch.zeros(L.n_shape + 1))


def test_from_motion_features_validates_dim():
    with pytest.raises(ValueError):
        MotionSequence.from_motion_features(torch.zeros(3, 5), torch.zeros(10))


def test_retarget_validates_beta_dim():
    seq = _seq()
    with pytest.raises(ValueError):
        seq.retarget(torch.zeros(seq.layout.n_shape + 2))
