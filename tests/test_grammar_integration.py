"""Integration: doc-02 typed SignPlan -> doc-03 SIR -> gloss projection.

Verifies the bridge is *faithful*: it neither drops nor invents structure.
The decisive round-trip property is that the manual gloss projected out of the
SIR equals the plan's manual-unit order -- because the gloss is exactly one
linearisation of the manual sub-DAG the plan defined.
"""

import pytest

from signtranslator.planning.schema import (
    SignPlan, SemanticFrame, NonmanualSpan, DEFAULT_VOCAB, validate_plan,
)
from signtranslator.grammar.sir import (
    EventKind, EdgeType, gloss_projection, validate_sir, is_topological_order,
)
from signtranslator.grammar.integration import plan_to_sir, plan_manual_units
from signtranslator.grammar.temporal import sir_temporal_loss


def _plan(**kw):
    frame = kw.pop("frame", SemanticFrame(predicate=0, args=[(0, 1), (1, 2)]))
    base = dict(frame=frame, referents=[1, 2], manual_units=[10, 11, 12],
                loci={1: 0, 2: 1})
    base.update(kw)
    return SignPlan(**base)


# ---------------------------------------------------------------------------
# Faithful round trip
# ---------------------------------------------------------------------------
def test_gloss_projection_equals_plan_manual_order():
    plan = _plan(manual_units=[10, 11, 12, 13])
    sir = plan_to_sir(plan)
    assert gloss_projection(sir) == plan_manual_units(plan)


def test_nonmanual_spans_become_scoped_events():
    plan = _plan(manual_units=[10, 11, 12],
                 nonmanual=[NonmanualSpan(marker=0, start=0, end=2)])
    sir = plan_to_sir(plan)
    nm = sir.nonmanual_events()
    assert len(nm) == 1 and nm[0].label == 0
    # SCOPE edges reach every covered manual unit (inclusive end -> 3 units)
    scope = [e for e in sir.edges if e.type is EdgeType.SCOPE]
    assert len(scope) == 3
    # and the non-manual event does not appear in the manual gloss
    assert gloss_projection(sir) == [10, 11, 12]


def test_nonmanual_span_time_covers_its_units():
    plan = _plan(manual_units=[10, 11, 12, 13],
                 nonmanual=[NonmanualSpan(marker=1, start=1, end=2)])
    sir = plan_to_sir(plan, unit_time=1.0)
    nm = sir.nonmanual_events()[0]
    assert nm.t_start == 1.0 and nm.t_end == 3.0        # units [1..2] inclusive


def test_fingerspelled_units_are_fingerspell_events():
    plan = _plan(manual_units=[10, 99, 12], fingerspelling=[1])
    sir = plan_to_sir(plan)
    kinds = {e.label: e.kind for e in sir.events if e.kind.is_manual}
    assert kinds[10] == EventKind.MANUAL
    assert kinds[99] == EventKind.FINGERSPELL
    # still projected -- covered, not dropped
    assert gloss_projection(sir) == [10, 99, 12]


def test_no_information_is_dropped():
    """Every manual unit and every non-manual span survives the mapping."""
    plan = _plan(manual_units=[10, 11, 12, 13, 14],
                 nonmanual=[NonmanualSpan(0, 0, 1), NonmanualSpan(1, 3, 4)])
    sir = plan_to_sir(plan)
    assert len(sir.manual_events()) == 5
    assert len(sir.nonmanual_events()) == 2


# ---------------------------------------------------------------------------
# Structural validity of the produced SIR
# ---------------------------------------------------------------------------
def test_produced_sir_is_structurally_valid():
    plan = _plan(manual_units=[10, 11, 12],
                 nonmanual=[NonmanualSpan(0, 0, 2)])
    sir = plan_to_sir(plan)
    assert validate_sir(sir, num_loci=DEFAULT_VOCAB.num_loci) == []


def test_produced_sir_intervals_are_temporally_consistent():
    plan = _plan(manual_units=[10, 11, 12],
                 nonmanual=[NonmanualSpan(0, 0, 2)])
    sir = plan_to_sir(plan)
    import torch
    n = len(sir.events)
    starts = torch.zeros(n)
    ends = torch.zeros(n)
    for e in sir.events:
        starts[e.id] = e.t_start
        ends[e.id] = e.t_end
    edges = [(e.source, e.target, e.type.value) for e in sir.edges]
    loss = sir_temporal_loss(starts, ends, edges)
    # Validity is exactly 0 (all intervals valid). The residual loss is entirely
    # the hinge margins that boundary-aligned intervals pay:
    #   * 2 precedence edges: adjacent signs *meet* (end == next start), so each
    #     pays the strict-precedence margin eps.
    #   * scope containment [0,3) over units [0,1],[1,2],[2,3]: the nm marker
    #     starts exactly at the first unit's start and ends exactly at the last
    #     unit's end, so those two boundary units each pay one eps margin; the
    #     interior unit pays 0.
    # Total = (2 + 2) * DEFAULT_EPS -- fully accounted for, no unexplained slack.
    from signtranslator.grammar.temporal import DEFAULT_EPS
    assert loss.detach().item() == pytest.approx(4 * DEFAULT_EPS, abs=1e-6)


def test_projection_is_a_topological_order_of_manual_dag():
    plan = _plan(manual_units=[10, 11, 12, 13])
    sir = plan_to_sir(plan)
    # reconstruct the id order that produced the projection
    manual = sorted(sir.manual_events(), key=lambda e: e.t_start)
    assert is_topological_order([e.id for e in manual], sir)


# ---------------------------------------------------------------------------
# Optional referent/locus annotation (only when caller supplies the mapping)
# ---------------------------------------------------------------------------
def test_loci_annotated_only_when_mapping_supplied():
    plan = _plan(manual_units=[10, 11, 12], loci={1: 3, 2: 5})
    # without a unit->referent map, no locus is invented
    plain = plan_to_sir(plan)
    assert all(e.locus is None for e in plain.manual_events())
    # with a map, the locus flows through
    annotated = plan_to_sir(plan, unit_referents={0: 1, 2: 2})
    by_lex = {e.label: e for e in annotated.manual_events()}
    assert by_lex[10].referent == 1 and by_lex[10].locus == 3
    assert by_lex[12].referent == 2 and by_lex[12].locus == 5
    assert by_lex[11].locus is None                      # unit 1 unmapped


def test_coref_edges_link_repeated_referent():
    plan = _plan(manual_units=[10, 11, 10], loci={1: 2})
    sir = plan_to_sir(plan, unit_referents={0: 1, 2: 1})   # units 0 and 2 = ref 1
    coref = [e for e in sir.edges if e.type is EdgeType.COREF]
    assert len(coref) == 1                               # one edge chaining the two


# ---------------------------------------------------------------------------
# Whole-chain stress: many random-ish plans round-trip cleanly
# ---------------------------------------------------------------------------
def test_whole_chain_stress_over_many_plans():
    import random
    rng = random.Random(20260719)
    for _ in range(200):
        n = rng.randint(1, DEFAULT_VOCAB.max_units)
        units = [rng.randrange(DEFAULT_VOCAB.num_lexemes) for _ in range(n)]
        # 0..2 non-manual spans within range
        spans = []
        for _ in range(rng.randint(0, 2)):
            a = rng.randrange(n)
            b = rng.randrange(a, n)
            spans.append(NonmanualSpan(rng.randrange(DEFAULT_VOCAB.num_nonmanual),
                                       a, b))
        fs = sorted(rng.sample(range(n), rng.randint(0, min(2, n))))
        plan = SignPlan(frame=SemanticFrame(predicate=0, args=[]),
                        manual_units=units, nonmanual=spans, fingerspelling=fs)
        sir = plan_to_sir(plan)
        # (1) faithful manual order
        assert gloss_projection(sir) == units
        # (2) structurally valid
        assert validate_sir(sir, num_loci=DEFAULT_VOCAB.num_loci) == []
        # (3) every span survived
        assert len(sir.nonmanual_events()) == len(spans)
