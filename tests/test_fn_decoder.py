"""Verification of the multi-channel non-manual decoder.

Proves per-channel INDEPENDENT Bernoulli outputs (co-occurring channels, not a
softmax), conditioning dependence, span decoding, and gradient flow.
"""

import pytest
import torch

from signtranslator.facial_nmm.decoder import MultiChannelDecoder
from signtranslator.grammar.nonmanual import multilabel_scope_bce


def _dec(cond_dim=8, hidden=16, K=5, seed=0):
    torch.manual_seed(seed)
    return MultiChannelDecoder(cond_dim, hidden, num_channels=K, num_layers=2)


def test_output_shape_and_valid_probabilities():
    dec = _dec()
    cond = torch.randn(2, 7, 8)
    p = dec.probabilities(cond)
    assert p.shape == (2, 7, 5)
    assert torch.all(p > 0) and torch.all(p < 1)             # valid Bernoulli probs


def test_channels_are_independent_not_softmax():
    """Two channels can BOTH be ~1 at the same frame -- impossible under a softmax."""
    dec = _dec()
    with torch.no_grad():
        dec.head.bias[0] = 20.0                              # force channels 0 and 1 high
        dec.head.bias[1] = 20.0
    p = dec.probabilities(torch.randn(1, 4, 8))
    assert torch.all(p[..., 0] > 0.99) and torch.all(p[..., 1] > 0.99)
    assert torch.all(p[..., :2].sum(-1) > 1.5)               # sum > 1 => not a softmax


def test_output_depends_on_conditioning():
    dec = _dec(seed=1)
    dec.eval()
    a = dec.probabilities(torch.randn(1, 6, 8))
    b = dec.probabilities(torch.randn(1, 6, 8))
    assert not torch.allclose(a, b, atol=1e-4)


def test_decode_spans_from_activations():
    dec = _dec()
    # a hand-built activation: channel 2 active on units 1..2
    probs = torch.zeros(4, 5)
    probs[1:3, 2] = 1.0
    starts = [0.0, 1.0, 2.0, 3.0]; ends = [1.0, 2.0, 3.0, 4.0]
    spans = dec.decode(probs, starts, ends)
    assert len(spans) == 1
    assert spans[0].marker == 2 and spans[0].start == 1.0 and spans[0].end == 3.0


def test_bce_and_gradient_flow():
    dec = _dec(seed=2)
    cond = torch.randn(2, 6, 8, requires_grad=True)
    logits = dec(cond)
    targets = (torch.rand(2, 6, 5) > 0.5).float()
    loss = multilabel_scope_bce(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert cond.grad is not None and torch.isfinite(cond.grad).all()
    assert dec.head.weight.grad is not None and dec.head.weight.grad.abs().sum() > 0
