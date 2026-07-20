"""Verification of the non-manual evaluation batteries.

Proves minimal-pair distinguishing (identical hands, changed marker -> different
meaning), scope boundary error, gaze/locus cosine agreement, head-manual
synchronisation, and channel ablation drop.
"""

import math

import pytest
import torch

from signtranslator.facial_nmm.channels import Marker
from signtranslator.facial_nmm.evaluation import (
    minimal_pair_distinguishes, minimal_pair_accuracy, scope_boundary_error,
    gaze_locus_agreement, head_manual_offset, synchronisation_rate,
    channel_ablation_drop,
)


# ---------------------------------------------------------------------------
# minimal-pair comprehension
# ---------------------------------------------------------------------------
def test_minimal_pair_distinguishes_grammatical_markers():
    # a reader that maps each marker to its own grammatical label
    predict = lambda m: int(m)
    assert minimal_pair_distinguishes(Marker.YN_Q, Marker.WH_Q, predict)   # different meaning
    # a blind reader that ignores the marker fails to distinguish
    blind = lambda m: 0
    assert not minimal_pair_distinguishes(Marker.YN_Q, Marker.WH_Q, blind)


def test_minimal_pair_accuracy():
    pairs = [(Marker.YN_Q, Marker.WH_Q), (Marker.NEG, Marker.TOPIC),
             (Marker.COND, Marker.YN_Q)]
    assert minimal_pair_accuracy(pairs, lambda m: int(m)) == 1.0     # all distinguished
    assert minimal_pair_accuracy(pairs, lambda m: 0) == 0.0          # blind reader


# ---------------------------------------------------------------------------
# scope boundary error
# ---------------------------------------------------------------------------
def test_scope_boundary_error():
    assert scope_boundary_error(1.0, 3.0, 1.0, 3.0) == 0.0           # exact
    assert abs(scope_boundary_error(1.1, 2.8, 1.0, 3.0) - (0.1 + 0.2)) < 1e-9


# ---------------------------------------------------------------------------
# gaze / locus agreement
# ---------------------------------------------------------------------------
def test_gaze_locus_agreement_cosine():
    gaze = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    at_locus = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)     # same direction
    assert abs(float(gaze_locus_agreement(gaze, at_locus)) - 1.0) < 1e-9
    away = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)
    assert abs(float(gaze_locus_agreement(gaze, away)) + 1.0) < 1e-9  # opposite -> -1
    ortho = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    assert abs(float(gaze_locus_agreement(gaze, ortho))) < 1e-9       # orthogonal -> 0


# ---------------------------------------------------------------------------
# head-manual synchronisation
# ---------------------------------------------------------------------------
def test_head_manual_synchronisation():
    assert head_manual_offset(1.02, 1.0) == pytest.approx(0.02)
    offsets = [0.01, -0.02, 0.10, 0.005]
    assert synchronisation_rate(offsets, tolerance=0.03) == 0.75      # 3 of 4 within 30ms


# ---------------------------------------------------------------------------
# channel ablation
# ---------------------------------------------------------------------------
def test_channel_ablation_drop():
    full = 0.9
    ablated = {"face": 0.5, "gaze": 0.8, "torso": 0.88}
    drop = channel_ablation_drop(full, ablated)
    assert abs(drop["face"] - 0.4) < 1e-9                             # face carries the most
    assert drop["face"] > drop["gaze"] > drop["torso"]               # face most important
