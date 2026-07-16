"""Tests for evaluation metrics against closed-form expected values."""

import torch

from signtranslator.eval import (
    retrieval_recall_at_k, mean_per_joint_position_error, top1_accuracy,
    word_error_rate,
)


def test_recall_perfect_when_diagonal_dominant():
    sim = torch.eye(10) * 5.0  # correct match strongest for every query
    r = retrieval_recall_at_k(sim, ks=(1, 5))
    assert r[1] == 1.0 and r[5] == 1.0


def test_recall_zero_when_diagonal_is_worst():
    n = 6
    sim = torch.ones(n, n)
    sim.fill_diagonal_(-1.0)  # correct match always ranked last
    r = retrieval_recall_at_k(sim, ks=(1,))
    assert r[1] == 0.0


def test_recall_at_k_counts_topk_membership():
    # Query 0: correct index 0 gets 2nd-highest score -> in top-2 but not top-1.
    sim = torch.tensor([
        [0.9, 1.0, 0.1],   # best is col 1, correct col 0 is 2nd
        [0.1, 0.9, 0.2],   # correct col 1 is best
        [0.1, 0.2, 0.9],   # correct col 2 is best
    ])
    r = retrieval_recall_at_k(sim, ks=(1, 2))
    assert abs(r[1] - (2 / 3)) < 1e-6   # rows 1,2 correct at top-1
    assert r[2] == 1.0                   # all correct within top-2


def test_mpjpe_zero_for_identical():
    x = torch.randn(2, 3, 8, 27)
    assert mean_per_joint_position_error(x, x) == 0.0


def test_mpjpe_known_offset():
    x = torch.zeros(1, 3, 4, 5)
    y = torch.zeros(1, 3, 4, 5)
    y[:, 0] = 3.0
    y[:, 1] = 4.0  # per-joint L2 = sqrt(3^2 + 4^2) = 5
    assert abs(mean_per_joint_position_error(x, y) - 5.0) < 1e-5


def test_top1_accuracy_known():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = torch.tensor([1, 0, 0])  # third is wrong
    assert abs(top1_accuracy(logits, labels) - (2 / 3)) < 1e-6


def test_word_error_rate_reexport():
    assert word_error_rate([[1, 2, 3]], [[1, 2, 3]]) == 0.0
