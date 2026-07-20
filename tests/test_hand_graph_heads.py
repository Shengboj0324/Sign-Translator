"""Verification of auxiliary heads and annotation-masked losses.

The decisive discipline (from Docs 03/04): unlabelled samples are EXCLUDED, never
given a fabricated target; a batch with no labels contributes exactly 0 (no NaN).
Each loss is also checked to be minimised by the correct prediction.
"""

import math

import pytest
import torch

from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d
from signtranslator.hand_graph.heads import (
    masked_cross_entropy, masked_bce_with_logits,
    HandshapeHead, SelectedFingersHead, PalmOrientationHead,
    contact_loss, mirror_points, symmetry_loss,
)


# ---------------------------------------------------------------------------
# masked primitives
# ---------------------------------------------------------------------------
def test_masked_ce_ignores_unlabelled_and_matches_plain_when_all_labelled():
    torch.manual_seed(0)
    logits = torch.randn(4, 3, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 1])
    full = masked_cross_entropy(logits, targets, torch.ones(4, dtype=torch.float64))
    ref = torch.nn.functional.cross_entropy(logits, targets)
    assert abs(float(full) - float(ref)) < 1e-12
    # mask out sample 3 with an INVALID placeholder target -> must not error or leak
    bad_targets = torch.tensor([0, 1, 2, -1])
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    part = masked_cross_entropy(logits, bad_targets, mask)
    ref3 = torch.nn.functional.cross_entropy(logits[:3], targets[:3])
    assert abs(float(part) - float(ref3)) < 1e-12


def test_masked_losses_are_zero_with_no_labels_no_nan():
    logits = torch.randn(5, 4, dtype=torch.float64)
    ce = masked_cross_entropy(logits, torch.zeros(5, dtype=torch.long),
                              torch.zeros(5, dtype=torch.float64))
    bce = masked_bce_with_logits(torch.randn(5, 5, dtype=torch.float64),
                                 torch.zeros(5, 5, dtype=torch.float64),
                                 torch.zeros(5, dtype=torch.float64))
    assert float(ce) == 0.0 and float(bce) == 0.0
    assert torch.isfinite(torch.tensor([float(ce), float(bce)])).all()


def test_masked_bce_is_per_label_independent():
    # two labels; a sample labelled only on label 0 (mask per-entry)
    logits = torch.zeros(1, 2, dtype=torch.float64)          # p = 0.5 each
    targets = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])                        # only label 0 supervised
    loss = masked_bce_with_logits(logits, targets, mask)
    assert abs(float(loss) - math.log(2)) < 1e-12           # -log(0.5) on the one entry


# ---------------------------------------------------------------------------
# heads minimised by correct prediction
# ---------------------------------------------------------------------------
def test_handshape_head_learns_toward_target():
    torch.manual_seed(1)
    head = HandshapeHead(dim=8, num_classes=6).double()
    h = torch.randn(16, 8, dtype=torch.float64)
    targets = torch.randint(0, 6, (16,))
    mask = torch.ones(16, dtype=torch.float64)
    opt = torch.optim.Adam(head.parameters(), lr=0.1)
    l0 = head.loss(h, targets, mask).item()
    for _ in range(200):
        opt.zero_grad(); loss = head.loss(h, targets, mask); loss.backward(); opt.step()
    assert head.loss(h, targets, mask).item() < 0.1 * l0     # fits the labels


def test_palm_orientation_geodesic_zero_at_target():
    head = PalmOrientationHead(dim=6).double()
    with torch.no_grad():                                    # make the head output R exactly
        head.fc.weight.zero_();
    # target = identity; a zeroed linear (+bias) gives 6D=bias; set bias to identity 6D
    ident6d = torch.tensor([1.0, 0, 0, 0, 1.0, 0], dtype=torch.float64)
    with torch.no_grad():
        head.fc.bias.copy_(ident6d)
    h = torch.randn(4, 6, dtype=torch.float64)
    target_R = torch.eye(3, dtype=torch.float64).expand(4, 3, 3)
    loss = head.loss(h, target_R, torch.ones(4, dtype=torch.float64))
    assert float(loss) < 1e-9


def test_selected_fingers_multilabel_masking():
    head = SelectedFingersHead(dim=8, num_fingers=5).double()
    h = torch.randn(3, 8, dtype=torch.float64)
    targets = torch.tensor([[1.0, 0, 1, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)  # sample 1 unlabelled
    loss = head.loss(h, targets, mask)
    assert torch.isfinite(loss) and float(loss) > 0


# ---------------------------------------------------------------------------
# contact + symmetry
# ---------------------------------------------------------------------------
def test_contact_loss_zero_at_perfect_prediction():
    p = torch.tensor([0.999999, 1e-6, 0.999999], dtype=torch.float64)
    labels = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    assert contact_loss(p, labels).item() < 1e-4


def test_symmetry_loss_zero_for_mirror_symmetric_hands():
    torch.manual_seed(2)
    right = torch.randn(3, 21, 3, dtype=torch.float64)
    left = mirror_points(right, axis=0)                      # perfectly mirror-symmetric
    loss = symmetry_loss(left, right, torch.ones(3, dtype=torch.float64))
    assert float(loss) < 1e-12
    # break symmetry on one sample -> positive
    left2 = left.clone(); left2[0, 5] += 1.0
    assert symmetry_loss(left2, right, torch.ones(3, dtype=torch.float64)).item() > 0


def test_symmetry_loss_respects_mask():
    right = torch.randn(2, 21, 3, dtype=torch.float64)
    left = torch.randn(2, 21, 3, dtype=torch.float64)        # not symmetric
    # only sample 0 is a symmetric sign
    masked = symmetry_loss(left, right, torch.tensor([1.0, 0.0], dtype=torch.float64))
    only0 = symmetry_loss(left[:1], right[:1], torch.ones(1, dtype=torch.float64))
    assert abs(float(masked) - float(only0)) < 1e-12


def test_head_gradients_flow():
    head = SelectedFingersHead(dim=8).double()
    h = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    head.loss(h, torch.ones(4, 5, dtype=torch.float64),
              torch.ones(4, dtype=torch.float64)).backward()
    assert torch.isfinite(head.fc.weight.grad).all()
    assert h.grad is not None and torch.isfinite(h.grad).all()
