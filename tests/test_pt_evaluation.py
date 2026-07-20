"""Adversarial tests for the evidence battery (Doc-11, stage 11f)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from signtranslator.pretraining.evaluation import (
    chance_accuracy, linear_probe_accuracy, low_resource_scaling_curve,
    cross_signer_retrieval_recall, signer_leakage_accuracy, is_leaky,
    loss_usefulness_dissociation,
)

torch.manual_seed(0)


def test_chance_accuracy():
    assert chance_accuracy([0, 0, 0, 1]) == 0.75


def test_probe_recovers_linearly_decodable_label():
    C, n = 4, 200
    y = torch.randint(0, C, (n,))
    X = F.one_hot(y, C).float() + 0.05 * torch.randn(n, C)     # decodable
    acc = linear_probe_accuracy(X[:100], y[:100], X[100:], y[100:], C)
    assert acc > 0.9


def test_probe_fails_on_label_independent_features():
    C, n = 4, 200
    y = torch.randint(0, C, (n,))
    X = torch.randn(n, 8)                                       # independent of y
    acc = linear_probe_accuracy(X[:100], y[:100], X[100:], y[100:], C)
    assert acc < chance_accuracy(y.tolist()) + 0.15            # ~chance


def test_scaling_curve_is_monotone_ish_and_covers_sizes():
    C, n = 3, 180
    y = torch.randint(0, C, (n,))
    X = F.one_hot(y, C).float() + 0.1 * torch.randn(n, C)
    curve = low_resource_scaling_curve(X[:120], y[:120], X[120:], y[120:], C,
                                       sizes=[6, 30, 120])
    sizes = [s for s, _ in curve]
    assert sizes == [6, 30, 120]
    assert curve[-1][1] >= curve[0][1]                        # more data -> >= acc


def test_cross_signer_retrieval_content_generalises():
    # content-encoding embeddings retrieve across signers; signer-encoding do not.
    C = 3
    content = np.array([c for c in range(C) for _ in range(4)] * 1)   # 12 items
    signer = np.array([s for _ in range(C) for s in range(4)])
    content_emb = F.one_hot(torch.tensor(content), C).float()
    r_content = cross_signer_retrieval_recall(content_emb, content, signer, k=1)
    assert r_content == 1.0
    signer_emb = F.one_hot(torch.tensor(signer), 4).float()
    r_signer = cross_signer_retrieval_recall(signer_emb, content, signer, k=1)
    assert r_signer < 1.0                                     # signer code fails


def test_signer_leakage_detected_and_flagged():
    S, n = 3, 180
    signer = torch.randint(0, S, (n,))
    leaky = F.one_hot(signer, S).float() + 0.05 * torch.randn(n, S)   # encodes signer
    acc = signer_leakage_accuracy(leaky, signer.tolist(), S)
    assert is_leaky(acc, chance_accuracy(signer.tolist()))

    clean = torch.randn(n, 6)                                  # no signer info
    acc_clean = signer_leakage_accuracy(clean, signer.tolist(), S)
    assert not is_leaky(acc_clean, chance_accuracy(signer.tolist()))


def test_equal_loss_unequal_usefulness():
    r = loss_usefulness_dissociation(n=240, num_classes=4, seed=1)
    # identical reconstruction loss...
    assert abs(r["recon_loss_A"] - r["recon_loss_B"]) < 1e-9
    # ...but very different linguistic-probe accuracy.
    assert r["probe_acc_A"] > 0.9
    assert r["probe_acc_B"] < r["chance"] + 0.15
    assert r["probe_acc_A"] - r["probe_acc_B"] > 0.5
