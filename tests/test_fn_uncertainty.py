"""Verification of focal loss, class balancing, and heteroscedastic uncertainty.

Proves focal down-weights easy examples, class-balanced weights are larger for
rarer classes, the Gaussian NLL is minimised at σ²=(y−p)², and low inter-annotator
agreement maps to higher predicted uncertainty.
"""

import math

import pytest
import torch

from signtranslator.facial_nmm.uncertainty import (
    focal_loss, focal_modulation, class_balanced_weights, heteroscedastic_nll,
    agreement_to_target_logvar,
)


# ---------------------------------------------------------------------------
# focal loss
# ---------------------------------------------------------------------------
def test_focal_modulation_shrinks_for_easy_examples():
    pt = torch.tensor([0.0, 0.5, 0.9, 0.99], dtype=torch.float64)
    m = focal_modulation(pt, gamma=2.0)
    assert torch.all(m[1:] < m[:-1])                        # decreasing in p_t
    assert abs(float(m[0]) - 1.0) < 1e-12                   # hardest -> factor 1
    assert float(m[-1]) < 1e-3                              # easy -> ~0


def test_focal_suppresses_easy_more_than_bce():
    # one easy positive (logit 8) and one hard (logit 0), both y=1
    logits = torch.tensor([8.0, 0.0], dtype=torch.float64)
    y = torch.tensor([1.0, 1.0], dtype=torch.float64)
    fl = focal_loss(logits, y, gamma=2.0, alpha=0.5, reduction="none")
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    supp = fl / bce                                         # per-example suppression
    assert float(supp[0]) < float(supp[1])                 # easy suppressed more
    assert float(supp[0]) < 1e-3


# ---------------------------------------------------------------------------
# class balancing
# ---------------------------------------------------------------------------
def test_class_balanced_weights_larger_for_rarer():
    counts = torch.tensor([1000.0, 100.0, 10.0])
    w = class_balanced_weights(counts, beta=0.999)
    assert w[2] > w[1] > w[0]                               # rarer -> larger weight
    assert abs(float(w.mean()) - 1.0) < 1e-9               # normalised to mean 1


# ---------------------------------------------------------------------------
# heteroscedastic uncertainty
# ---------------------------------------------------------------------------
def test_nll_minimised_at_true_variance():
    r = 0.4                                                 # fixed residual (y - p)
    pred = torch.tensor([0.0], dtype=torch.float64)
    target = torch.tensor([r], dtype=torch.float64)
    s_star = math.log(r ** 2)                               # optimum log-variance
    nll_star = heteroscedastic_nll(pred, target, torch.tensor([s_star], dtype=torch.float64))
    for ds in (-1.0, -0.5, 0.5, 1.0):
        nll = heteroscedastic_nll(pred, target, torch.tensor([s_star + ds], dtype=torch.float64))
        assert float(nll) > float(nll_star) - 1e-12         # s* is the minimiser


def test_low_agreement_maps_to_higher_uncertainty():
    kappa = torch.tensor([0.9, 0.5, 0.1], dtype=torch.float64)   # high -> low agreement
    s = agreement_to_target_logvar(kappa, base=0.0, slope=2.0)
    assert torch.all(s[1:] > s[:-1])                        # lower agreement -> higher logvar
    assert float(s[0]) < float(s[-1])                       # 0.9 agreement -> narrowest
