"""Verification of hand-specific evaluation metrics.

Fingertip error is proved scale-invariant; contact F1 matches a hand-computed
confusion; collision rate brackets at 0 and 1; mirror/handedness behaviour and
left/right consistency are proved exactly.
"""

import pytest
import torch

from signtranslator.hand_graph.hetero_graph import HAND_LANDMARKS, FINGERTIPS
from signtranslator.hand_graph.metrics import (
    hand_scale, fingertip_error_in_hand_scale, handshape_accuracy,
    contact_prf, collision_rate, mirror_hand, left_right_consistency,
)


def _hand(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(HAND_LANDMARKS, 3, generator=g, dtype=torch.float64)


# ---------------------------------------------------------------------------
# fingertip error in hand scale
# ---------------------------------------------------------------------------
def test_fingertip_error_zero_when_identical():
    x = _hand(1)
    assert fingertip_error_in_hand_scale(x, x.clone()).item() < 1e-12


def test_fingertip_error_is_scale_invariant():
    gt = _hand(2)
    pred = gt.clone()
    pred[FINGERTIPS[0]] += torch.tensor([0.05, 0, 0], dtype=torch.float64)
    e1 = fingertip_error_in_hand_scale(pred, gt).item()
    # scale the whole scene by 10x: absolute error grows 10x, hand size grows 10x
    e2 = fingertip_error_in_hand_scale(pred * 10.0, gt * 10.0).item()
    assert abs(e1 - e2) < 1e-9


# ---------------------------------------------------------------------------
# handshape accuracy
# ---------------------------------------------------------------------------
def test_handshape_accuracy_masked():
    logits = torch.tensor([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0], [2.0, 0, 0]])
    targets = torch.tensor([0, 1, 0, 0])                    # sample 2 wrong
    acc_all = handshape_accuracy(logits, targets)
    assert abs(float(acc_all) - 0.75) < 1e-12
    # mask out the wrong sample -> accuracy 1.0
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
    assert abs(float(handshape_accuracy(logits, targets, mask)) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# contact F1
# ---------------------------------------------------------------------------
def test_contact_prf_matches_hand_computation():
    pred = torch.tensor([1, 1, 0, 0, 1])
    true = torch.tensor([1, 0, 0, 1, 1])
    # TP=2 (idx0,4), FP=1 (idx1), FN=1 (idx3)
    r = contact_prf(pred, true)
    assert abs(r.precision - 2 / 3) < 1e-12
    assert abs(r.recall - 2 / 3) < 1e-12
    assert abs(r.f1 - 2 / 3) < 1e-12


def test_contact_prf_handles_empty_predictions():
    r = contact_prf(torch.zeros(4), torch.tensor([1, 0, 1, 0]))
    assert r.precision == 0.0 and r.recall == 0.0 and r.f1 == 0.0


# ---------------------------------------------------------------------------
# collision rate
# ---------------------------------------------------------------------------
def test_collision_rate_brackets():
    radii = torch.tensor([1.0, 1.0], dtype=torch.float64)
    # frame 0 separated (dist 10), frame 1 penetrating (dist 0.5 < 2)
    joints = torch.tensor([[[0.0, 0, 0], [10.0, 0, 0]],
                           [[0.0, 0, 0], [0.5, 0, 0]]], dtype=torch.float64)
    assert collision_rate(joints, radii) == 0.5
    far = torch.tensor([[[0.0, 0, 0], [10.0, 0, 0]]], dtype=torch.float64)
    assert collision_rate(far, radii) == 0.0


# ---------------------------------------------------------------------------
# mirror / handedness + left-right consistency
# ---------------------------------------------------------------------------
def test_mirror_is_an_involution():
    x = _hand(3)
    assert torch.allclose(mirror_hand(mirror_hand(x)), x, atol=1e-12)
    # mirroring negates exactly the chosen axis
    m = mirror_hand(x, axis=0)
    assert torch.allclose(m[..., 0], -x[..., 0], atol=1e-12)
    assert torch.allclose(m[..., 1:], x[..., 1:], atol=1e-12)


def test_left_right_consistency_zero_for_symmetric_hands():
    right = _hand(4)
    left = mirror_hand(right, axis=0)
    assert left_right_consistency(left, right).item() < 1e-12
    # break it -> positive
    left2 = left.clone(); left2[8] += torch.tensor([0.2, 0, 0], dtype=torch.float64)
    assert left_right_consistency(left2, right).item() > 0


def test_hand_scale_positive():
    x = _hand(5)
    assert hand_scale(x).item() > 0
