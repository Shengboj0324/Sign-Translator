"""Verification of the non-manual loss suite.

Proves the boundary target extraction (onset/offset of active runs), boundary loss
minimised by correct predictions, scope containment (reuse Doc-03), temporal
smoothness (zero on constant), and the assembled total.
"""

import pytest
import torch

from signtranslator.facial_nmm.losses import (
    nmm_bce, boundary_targets, boundary_loss, scope_loss, temporal_smoothness,
    total_nmm_loss, NMMWeights,
)


# ---------------------------------------------------------------------------
# boundary targets
# ---------------------------------------------------------------------------
def test_boundary_targets_mark_run_onset_and_offset():
    # one channel, active on frames 1,2 of [0,1,1,0]
    labels = torch.tensor([[[0.0], [1.0], [1.0], [0.0]]])   # (1, 4, 1)
    b = boundary_targets(labels)
    assert b.squeeze(-1).squeeze(0).tolist() == [0.0, 1.0, 1.0, 0.0]   # onset f1, offset f2


def test_boundary_targets_single_frame_run():
    labels = torch.tensor([[[1.0], [0.0], [1.0]]])          # two singleton runs
    b = boundary_targets(labels).squeeze(-1).squeeze(0)
    assert b.tolist() == [1.0, 0.0, 1.0]                    # each singleton is a boundary


def test_boundary_loss_minimised_by_correct_prediction():
    labels = torch.tensor([[[0.0], [1.0], [1.0], [0.0]]])
    tgt = boundary_targets(labels)
    good = (tgt * 20.0 - 10.0)                              # large logit where boundary
    bad = torch.zeros_like(tgt)
    assert boundary_loss(good, labels).item() < boundary_loss(bad, labels).item()


# ---------------------------------------------------------------------------
# scope containment (reuse Doc-03)
# ---------------------------------------------------------------------------
def test_scope_loss_zero_when_marker_contains_unit():
    ms = torch.tensor([0.0], dtype=torch.float64)
    me = torch.tensor([4.0], dtype=torch.float64)
    us = torch.tensor([1.0], dtype=torch.float64)
    ue = torch.tensor([2.0], dtype=torch.float64)
    assert scope_loss(ms, me, us, ue).item() < 1e-9         # marker [0,4] contains unit [1,2]
    # a marker that does not contain the unit -> positive
    assert scope_loss(torch.tensor([1.5]), torch.tensor([2.0]), us, ue).item() > 0


# ---------------------------------------------------------------------------
# temporal smoothness
# ---------------------------------------------------------------------------
def test_temporal_smoothness_zero_on_constant_positive_on_flicker():
    const = torch.ones(1, 6, 4)
    assert temporal_smoothness(const).item() == 0.0
    flicker = torch.zeros(1, 6, 4); flicker[:, ::2] = 1.0    # alternating -> high flicker
    assert temporal_smoothness(flicker).item() > 0.5


# ---------------------------------------------------------------------------
# total
# ---------------------------------------------------------------------------
def test_total_nmm_loss_assembles_and_weights():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 5)
    targets = (torch.rand(2, 6, 5) > 0.5).float()
    bnd = torch.randn(2, 6, 5)
    probs = torch.sigmoid(logits)
    total = total_nmm_loss(logits, targets, bnd, probs,
                           weights=NMMWeights(bce=2.0, boundary=1.0, smooth=0.5))
    assert torch.isfinite(total) and total.item() > 0
