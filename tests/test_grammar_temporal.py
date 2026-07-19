"""Proofs for Allen's interval algebra and the differentiable temporal losses.

For every loss the four properties are checked: non-negativity, zero-iff-relation,
sub-differentiability, and (the operational statement) that gradient descent on
the endpoints drives the loss to zero and the Allen relation to satisfaction.
"""

import itertools

import pytest
import torch

from signtranslator.grammar.temporal import (
    AllenRelation, classify_relation, intervals_intersect,
    validity_loss, precedence_loss, meets_loss, contains_loss, during_loss,
    overlap_loss, equals_loss, sir_temporal_loss, DEFAULT_EPS,
)


def _t(x):
    return torch.tensor(float(x), dtype=torch.float64)


# ---------------------------------------------------------------------------
# Allen classification
# ---------------------------------------------------------------------------
def test_classification_of_the_canonical_thirteen():
    cases = {
        AllenRelation.BEFORE:      ((0, 1), (2, 3)),
        AllenRelation.AFTER:       ((2, 3), (0, 1)),
        AllenRelation.MEETS:       ((0, 1), (1, 2)),
        AllenRelation.MET_BY:      ((1, 2), (0, 1)),
        AllenRelation.OVERLAPS:    ((0, 2), (1, 3)),
        AllenRelation.OVERLAPPED_BY:((1, 3), (0, 2)),
        AllenRelation.STARTS:      ((0, 1), (0, 3)),
        AllenRelation.STARTED_BY:  ((0, 3), (0, 1)),
        AllenRelation.DURING:      ((1, 2), (0, 3)),
        AllenRelation.CONTAINS:    ((0, 3), (1, 2)),
        AllenRelation.FINISHES:    ((2, 3), (0, 3)),
        AllenRelation.FINISHED_BY: ((0, 3), (2, 3)),
        AllenRelation.EQUALS:      ((0, 3), (0, 3)),
    }
    for expected, (x, y) in cases.items():
        assert classify_relation(x, y) is expected, (expected, x, y)


def test_classification_is_exhaustive_over_a_grid():
    """Every integer-endpoint interval pair classifies to some relation."""
    pts = range(5)
    for xs, xe, ys, ye in itertools.product(pts, repeat=4):
        if xs < xe and ys < ye:
            rel = classify_relation((xs, xe), (ys, ye), eps=0.5)
            assert isinstance(rel, AllenRelation)


def test_before_and_after_are_inverses():
    assert classify_relation((0, 1), (5, 6)) is AllenRelation.BEFORE
    assert classify_relation((5, 6), (0, 1)) is AllenRelation.AFTER


def test_intervals_intersect_predicate():
    assert intervals_intersect((0, 2), (1, 3))
    assert not intervals_intersect((0, 1), (2, 3))
    assert not intervals_intersect((0, 1), (1, 2))       # meeting is not overlap


# ---------------------------------------------------------------------------
# Generic loss properties, applied to every loss
# ---------------------------------------------------------------------------
_SATISFIED = {
    "precedence": ((0.0, 1.0), (2.0, 3.0)),
    "meets":      ((0.0, 1.0), (1.0, 2.0)),
    "contains":   ((0.0, 3.0), (1.0, 2.0)),
    "during":     ((1.0, 2.0), (0.0, 3.0)),
    "overlap":    ((0.0, 2.0), (1.0, 3.0)),
    "equals":     ((0.0, 3.0), (0.0, 3.0)),
}
_VIOLATED = {
    "precedence": ((2.0, 3.0), (0.0, 1.0)),
    "meets":      ((0.0, 1.0), (2.0, 3.0)),
    "contains":   ((1.0, 2.0), (0.0, 3.0)),
    "during":     ((0.0, 3.0), (1.0, 2.0)),
    "overlap":    ((0.0, 1.0), (2.0, 3.0)),
    "equals":     ((0.0, 1.0), (2.0, 3.0)),
}
_LOSSES = {
    "precedence": precedence_loss, "meets": meets_loss, "contains": contains_loss,
    "during": during_loss, "overlap": overlap_loss, "equals": equals_loss,
}


@pytest.mark.parametrize("name", list(_LOSSES))
def test_loss_is_zero_when_satisfied(name):
    fn = _LOSSES[name]
    (xs, xe), (ys, ye) = _SATISFIED[name]
    val = fn(_t(xs), _t(xe), _t(ys), _t(ye))
    assert float(val) <= 1e-6, f"{name} not zero when satisfied: {float(val)}"


@pytest.mark.parametrize("name", list(_LOSSES))
def test_loss_is_positive_when_violated(name):
    fn = _LOSSES[name]
    (xs, xe), (ys, ye) = _VIOLATED[name]
    assert float(fn(_t(xs), _t(xe), _t(ys), _t(ye))) > 0.0


@pytest.mark.parametrize("name", list(_LOSSES))
def test_loss_is_nonnegative_everywhere(name):
    fn = _LOSSES[name]
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        xs, xe, ys, ye = (torch.rand(1, generator=g, dtype=torch.float64) * 10 for _ in range(4))
        assert float(fn(xs, xe, ys, ye)) >= -1e-9


@pytest.mark.parametrize("name", list(_LOSSES))
def test_gradient_descent_satisfies_the_relation(name):
    """The operational proof: optimising the loss makes the relation hold.

    LR decay is used because several of these are L1 losses (|.|): under a
    constant step, Adam's +/-1 subgradient makes the endpoints dither within
    O(lr) of the optimum. Decaying the LR shrinks that band, giving genuine
    convergence rather than a residual bounded by the step size.
    """
    torch.manual_seed(0)
    fn = _LOSSES[name]
    (xs0, xe0), (ys0, ye0) = _VIOLATED[name]              # violating start
    params = [torch.tensor(v, dtype=torch.float64, requires_grad=True)
              for v in (xs0, xe0, ys0, ye0)]
    opt = torch.optim.Adam(params, lr=0.05)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=150, gamma=0.3)
    for _ in range(600):
        loss = fn(*params)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    assert float(fn(*params)) < 1e-3, f"{name} did not converge"


# ---------------------------------------------------------------------------
# Precedence -- the document's own claim, checked directly
# ---------------------------------------------------------------------------
def test_precedence_loss_zero_iff_strictly_before():
    eps = DEFAULT_EPS
    # exactly at the margin boundary: xe = ys - eps -> loss 0
    assert float(precedence_loss(_t(0.0), _t(1.0), _t(1.0 + eps), _t(2.0))) <= 1e-9
    # just inside: xe = ys -> loss = eps > 0
    val = float(precedence_loss(_t(0.0), _t(1.0), _t(1.0), _t(2.0)))
    assert abs(val - eps) < 1e-9


def test_precedence_loss_grows_linearly_with_violation():
    base = float(precedence_loss(_t(0.0), _t(2.0), _t(1.0), _t(3.0)))
    more = float(precedence_loss(_t(0.0), _t(3.0), _t(1.0), _t(3.0)))
    assert more - base == pytest.approx(1.0, abs=1e-9)   # +1 to xe -> +1 loss


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------
def test_validity_loss_zero_for_ordered_interval():
    assert float(validity_loss(_t(0.0), _t(1.0))) <= 1e-9
    assert float(validity_loss(_t(1.0), _t(0.0))) > 0.0   # start after end


# ---------------------------------------------------------------------------
# Containment / scope directionality
# ---------------------------------------------------------------------------
def test_contains_and_during_are_symmetric_forms():
    x, y = (0.0, 3.0), (1.0, 2.0)                          # X contains Y
    assert float(contains_loss(_t(x[0]), _t(x[1]), _t(y[0]), _t(y[1]))) <= 1e-6
    assert float(during_loss(_t(y[0]), _t(y[1]), _t(x[0]), _t(x[1]))) <= 1e-6
    # a non-containing pair is penalised
    assert float(contains_loss(_t(1.0), _t(2.0), _t(0.0), _t(3.0))) > 0.0


# ---------------------------------------------------------------------------
# SIR-level temporal objective
# ---------------------------------------------------------------------------
def test_sir_temporal_loss_zero_for_a_consistent_graph():
    # two events, 0 precedes 1; both valid intervals
    starts = torch.tensor([0.0, 2.0], dtype=torch.float64)
    ends = torch.tensor([1.0, 3.0], dtype=torch.float64)
    edges = [(0, 1, "precedence")]
    assert float(sir_temporal_loss(starts, ends, edges)) <= 1e-6


def test_sir_temporal_loss_penalises_precedence_violation():
    starts = torch.tensor([2.0, 0.0], dtype=torch.float64)
    ends = torch.tensor([3.0, 1.0], dtype=torch.float64)
    edges = [(0, 1, "precedence")]                         # but 0 is after 1
    assert float(sir_temporal_loss(starts, ends, edges)) > 0.0


def test_sir_temporal_loss_is_trainable_end_to_end():
    """Optimise event endpoints so a whole SIR becomes temporally consistent."""
    torch.manual_seed(0)
    starts = torch.tensor([2.0, 0.0, 1.5], dtype=torch.float64, requires_grad=True)
    ends = torch.tensor([3.0, 1.0, 0.5], dtype=torch.float64, requires_grad=True)
    edges = [(0, 1, "precedence"), (1, 2, "precedence")]   # 0<1<2, all violated
    opt = torch.optim.Adam([starts, ends], lr=0.05)
    first = float(sir_temporal_loss(starts, ends, edges))
    for _ in range(600):
        loss = sir_temporal_loss(starts, ends, edges)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(sir_temporal_loss(starts, ends, edges)) < 1e-2 < first


def test_unknown_edge_type_contributes_zero_temporal_loss():
    starts = torch.tensor([0.0, 1.0], dtype=torch.float64)
    ends = torch.tensor([1.0, 2.0], dtype=torch.float64)
    # coref/locus are not temporal, so only validity contributes (which is 0)
    assert float(sir_temporal_loss(starts, ends, [(0, 1, "coref")])) <= 1e-6
