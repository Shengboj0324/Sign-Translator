"""Verification of linguistic/affect disentanglement.

Proves the gradient-reversal mechanism (adversary gradient is the negation of a
plain classifier's), and the leakage certification: affect is NOT recoverable from
an affect-independent z_ling (normalised error ~1) but IS recoverable when folded
in (~0), so the guard has power.
"""

import pytest
import torch
import torch.nn.functional as F

from signtranslator.facial_nmm.disentangle import (
    grad_reverse, AffectAdversary, affect_leakage,
)


# ---------------------------------------------------------------------------
# gradient reversal
# ---------------------------------------------------------------------------
def test_grad_reverse_is_identity_forward_negates_backward():
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    y = grad_reverse(x, lambd=2.0)
    assert torch.equal(y, x)                                # identity forward
    y.sum().backward()
    assert torch.allclose(x.grad, -2.0 * torch.ones_like(x), atol=1e-12)  # -lambd


def test_adversary_gradient_is_reversed():
    torch.manual_seed(0)
    adv = AffectAdversary(dim=6, num_affect=3, lambd=1.0).double()
    affect = torch.randint(0, 3, (5,))
    z1 = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)
    z2 = z1.detach().clone().requires_grad_(True)
    # plain classifier (no reversal)
    g_plain = torch.autograd.grad(F.cross_entropy(adv.fc(z1), affect), z1)[0]
    # adversary (with reversal)
    g_adv = torch.autograd.grad(F.cross_entropy(adv(z2), affect), z2)[0]
    assert torch.allclose(g_adv, -g_plain, atol=1e-10)      # reversed gradient


# ---------------------------------------------------------------------------
# leakage certification
# ---------------------------------------------------------------------------
def _data(n=400, d=8, a=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    z_ling = torch.randn(n, d, generator=g, dtype=torch.float64)
    affect = torch.randn(n, a, generator=g, dtype=torch.float64)   # independent affect
    return z_ling, affect


def test_affect_not_recoverable_from_independent_z_ling():
    z, a = _data(seed=1)
    err = affect_leakage(z, a, ntrain=300, l2=1.0)
    assert err > 0.85                                       # ~1 -> disentangled


def test_leakage_probe_has_power_when_affect_folded_in():
    z, a = _data(seed=2)
    z_leaky = torch.cat([z, a], dim=1)                     # affect folded into z
    err = affect_leakage(z_leaky, a, ntrain=300, l2=1e-4)
    assert err < 0.05                                      # recovered -> leakage detectable
