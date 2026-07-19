"""Verification of non-manual multilabel interval prediction.

The two load-bearing properties: markers are multilabel (co-occurring markers are
representable, which a softmax could not be), and a marker's span must *contain*
the manual units it scopes -- non-manual scope is co-temporal, not punctuation.
"""

import pytest
import torch
import torch.nn.functional as F

from signtranslator.grammar.nonmanual import (
    multilabel_scope_bce, spans_from_activations, scope_containment_loss,
    NonmanualScopeHead, MarkerSpan,
)


def _t(x):
    return torch.tensor(float(x), dtype=torch.float64)


# ---------------------------------------------------------------------------
# Multilabel BCE
# ---------------------------------------------------------------------------
def test_bce_matches_manual_computation():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, dtype=torch.float64)
    targets = (torch.rand(2, 3, 4) > 0.5).double()
    got = multilabel_scope_bce(logits, targets)
    manual = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(got, manual, atol=1e-9)


def test_bce_respects_the_position_mask():
    torch.manual_seed(1)
    logits = torch.randn(1, 4, 3, dtype=torch.float64)
    targets = (torch.rand(1, 4, 3) > 0.5).double()
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])          # last two padded
    masked = multilabel_scope_bce(logits, targets, mask)
    # equals BCE over only the first two positions
    ref = F.binary_cross_entropy_with_logits(logits[:, :2], targets[:, :2])
    assert torch.allclose(masked, ref, atol=1e-9)


def test_bce_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        multilabel_scope_bce(torch.randn(2, 3, 4), torch.randn(2, 3, 5))


def test_multilabel_allows_co_occurring_markers():
    """Two markers can be perfectly predicted active at the same position.

    A softmax head cannot achieve zero loss here; a multilabel head can.
    """
    # target: at position 0, markers 0 AND 1 are both active
    targets = torch.zeros(1, 1, 3)
    targets[0, 0, 0] = 1.0
    targets[0, 0, 1] = 1.0
    logits = torch.tensor([[[20.0, 20.0, -20.0]]])       # both active, third off
    loss = multilabel_scope_bce(logits, targets)
    assert float(loss) < 1e-4                             # near-zero achievable


# ---------------------------------------------------------------------------
# Span decoding
# ---------------------------------------------------------------------------
def test_spans_from_contiguous_activation_run():
    # marker 0 active over positions 1,2 ; marker 1 active at position 0
    probs = torch.tensor([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.7, 0.1],
    ])
    unit_starts = [0.0, 1.0, 2.0]
    unit_ends = [1.0, 2.0, 3.0]
    spans = spans_from_activations(probs, unit_starts, unit_ends, threshold=0.5)
    by_marker = {s.marker: s for s in spans}
    assert by_marker[0].covered_units == (1, 2)
    assert by_marker[0].start == 1.0 and by_marker[0].end == 3.0
    assert by_marker[1].covered_units == (0,)


def test_spans_split_on_gaps():
    """An inactive position between two active runs yields two separate spans."""
    probs = torch.tensor([[0.9], [0.1], [0.9]])          # active, off, active
    spans = spans_from_activations(probs, [0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    marker0 = [s for s in spans if s.marker == 0]
    assert len(marker0) == 2
    assert marker0[0].covered_units == (0,) and marker0[1].covered_units == (2,)


def test_no_activation_yields_no_spans():
    probs = torch.zeros(3, 2)
    assert spans_from_activations(probs, [0, 1, 2], [1, 2, 3]) == []


def test_spans_rejects_bad_rank():
    with pytest.raises(ValueError):
        spans_from_activations(torch.zeros(3), [0], [1])


# ---------------------------------------------------------------------------
# Scope containment coupling (the "not punctuation" property)
# ---------------------------------------------------------------------------
def test_scope_containing_the_unit_has_zero_loss():
    # marker [0, 3) contains unit [1, 2)
    loss = scope_containment_loss(_t(0.0), _t(3.0), _t(1.0), _t(2.0))
    assert float(loss) <= 1e-6


def test_scope_not_containing_the_unit_is_penalised():
    # marker [1, 2) does NOT contain unit [0, 3)
    loss = scope_containment_loss(_t(1.0), _t(2.0), _t(0.0), _t(3.0))
    assert float(loss) > 0.0


def test_scope_containment_is_trainable():
    """Optimise a marker span to cover a fixed manual unit."""
    torch.manual_seed(0)
    ms = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    me = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
    us, ue = _t(0.0), _t(2.0)                             # unit to be covered
    opt = torch.optim.Adam([ms, me], lr=0.05)
    for _ in range(500):
        loss = scope_containment_loss(ms, me, us, ue)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(ms) < float(us) and float(me) > float(ue)   # now contains it


# ---------------------------------------------------------------------------
# Head module
# ---------------------------------------------------------------------------
def test_head_shapes_and_overfit():
    torch.manual_seed(0)
    head = NonmanualScopeHead(d_model=16, num_markers=3)
    x = torch.randn(1, 5, 16)
    # target: marker 0 on positions 0-1, marker 2 on position 4
    targets = torch.zeros(1, 5, 3)
    targets[0, 0, 0] = targets[0, 1, 0] = 1.0
    targets[0, 4, 2] = 1.0
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    first = float(head.loss(x, targets))
    for _ in range(300):
        loss = head.loss(x, targets)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < first * 0.1
    # decode and check the learned spans match the targets
    probs = torch.sigmoid(head(x))[0]
    spans = spans_from_activations(probs, list(range(5)), list(range(1, 6)))
    covered = {(s.marker, s.covered_units) for s in spans}
    assert (0, (0, 1)) in covered and (2, (4,)) in covered


def test_head_rejects_bad_rank():
    with pytest.raises(ValueError):
        NonmanualScopeHead(16, 3)(torch.randn(5, 16))
