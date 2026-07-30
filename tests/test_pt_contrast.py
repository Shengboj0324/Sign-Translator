"""Adversarial tests for cross-modal contrast + retrieval (Doc-11, stage 11c)."""

import math

import pytest
import torch

from signtranslator.pretraining.contrast import (
    info_nce_loss, l2_normalize, recall_at_k, retrieval_recall,
    info_nce_against_negatives,
)

torch.manual_seed(0)


def test_symmetric_infonce_zero_for_aligned_at_low_temp():
    z = l2_normalize(torch.randn(8, 16))
    loss, _ = info_nce_loss(z, z.clone(), temperature=0.01)
    assert loss.detach().item() < 1e-3            # identical -> perfectly aligned


def test_recall_at_k_perfect_identity():
    sim = torch.eye(6) * 10.0
    assert recall_at_k(sim, 1) == 1.0


def test_recall_at_k_monotone_in_k():
    torch.manual_seed(1)
    sim = torch.randn(20, 20)
    r1 = recall_at_k(sim, 1)
    r5 = recall_at_k(sim, 5)
    assert r5 >= r1                                # more slots never hurt


def test_retrieval_recall_aligned_pairs():
    z = l2_normalize(torch.randn(10, 12))
    out = retrieval_recall(z, z.clone(), ks=(1,))
    assert out["a2b_recall@1"] == 1.0 and out["b2a_recall@1"] == 1.0


def test_recall_k_out_of_range():
    with pytest.raises(ValueError):
        recall_at_k(torch.eye(3), 4)


def test_infonce_negatives_zero_when_positive_dominates():
    # anchor == positive, negatives orthogonal => loss -> 0 as tau -> 0.
    d = 8
    a = torch.eye(4, d)                            # 4 orthonormal anchors
    negs = torch.eye(4, d, dtype=torch.float32)
    negs = torch.roll(negs, shifts=4, dims=1)      # disjoint dims from anchors
    loss = info_nce_against_negatives(a, a.clone(), negs, temperature=0.05)
    assert loss.detach().item() < 1e-3


def test_infonce_negatives_high_when_negative_equals_positive():
    # a hard negative identical to the positive makes the task ~chance (2 classes).
    a = l2_normalize(torch.randn(5, 8))
    loss = info_nce_against_negatives(a, a.clone(), a.clone(), temperature=1.0)
    # denominator = exp(s)+sum_j exp(s_ij); with one neg == positive per row plus
    # cross terms, loss is strictly positive and bounded below by ~log2 behaviour.
    assert loss.detach().item() > 0.3


def test_infonce_negatives_rejects_bad_temperature():
    a = torch.randn(3, 4)
    with pytest.raises(ValueError):
        info_nce_against_negatives(a, a, a, temperature=0.0)
