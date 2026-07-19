"""Verification of the SIR temporal graph, validation, and gloss projection."""

import pytest

from signtranslator.grammar.sir import (
    EventKind, EdgeType, SIREvent, SIREdge, SIRGraph,
    validate_sir, gloss_projection, is_topological_order,
)


def _event(i, kind, label, ts, te, referent=None, locus=None):
    return SIREvent(id=i, kind=kind, label=label, t_start=ts, t_end=te,
                    referent=referent, locus=locus)


def _valid_graph():
    """Two manual signs in sequence, with a negation scoping the second."""
    events = [
        _event(0, EventKind.MANUAL, 5, 0.0, 1.0, referent=1, locus=0),
        _event(1, EventKind.MANUAL, 9, 1.0, 2.0, referent=2, locus=1),
        _event(2, EventKind.NONMANUAL, 0, 0.9, 2.1),      # NEG over event 1
    ]
    edges = [
        SIREdge(0, 1, EdgeType.PRECEDENCE),
        SIREdge(2, 1, EdgeType.SCOPE),
    ]
    return SIRGraph(events=events, edges=edges)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------
def test_valid_graph_passes():
    assert validate_sir(_valid_graph(), num_loci=7) == []


def test_manual_and_nonmanual_partition():
    g = _valid_graph()
    assert len(g.manual_events()) == 2
    assert len(g.nonmanual_events()) == 1
    assert EventKind.CLASSIFIER.is_manual and EventKind.FINGERSPELL.is_manual
    assert not EventKind.NONMANUAL.is_manual


def test_event_duration_and_time_overlap():
    a = _event(0, EventKind.MANUAL, 0, 0.0, 1.0)
    b = _event(1, EventKind.MANUAL, 1, 0.5, 1.5)
    c = _event(2, EventKind.MANUAL, 2, 2.0, 3.0)
    assert a.duration == 1.0
    assert a.overlaps_time(b) and not a.overlaps_time(c)


# ---------------------------------------------------------------------------
# Validation rules, each fired
# ---------------------------------------------------------------------------
def test_rule_invalid_interval():
    g = _valid_graph()
    g.events[0].t_end = g.events[0].t_start        # zero-length
    g.rebuild_index()
    assert "invalid_interval" in validate_sir(g)


def test_rule_edge_to_missing_node():
    g = _valid_graph()
    g.edges.append(SIREdge(0, 99, EdgeType.PRECEDENCE))
    assert "edge_references_missing_node" in validate_sir(g)


def test_rule_precedence_cycle():
    g = _valid_graph()
    g.edges.append(SIREdge(1, 0, EdgeType.PRECEDENCE))   # 0->1 and 1->0
    assert "precedence_cycle" in validate_sir(g)


def test_rule_scope_source_must_be_nonmanual():
    g = _valid_graph()
    g.edges.append(SIREdge(0, 1, EdgeType.SCOPE))        # manual source
    assert "scope_source_not_nonmanual" in validate_sir(g)


def test_rule_scope_target_must_be_manual():
    events = [
        _event(0, EventKind.NONMANUAL, 0, 0.0, 1.0),
        _event(1, EventKind.NONMANUAL, 1, 0.0, 1.0),
    ]
    g = SIRGraph(events=events, edges=[SIREdge(0, 1, EdgeType.SCOPE)])
    assert "scope_target_not_manual" in validate_sir(g)


def test_rule_coref_referent_mismatch():
    g = _valid_graph()
    g.edges.append(SIREdge(0, 1, EdgeType.COREF))        # refs 1 vs 2 differ
    assert "coref_referent_mismatch" in validate_sir(g)


def test_rule_locus_out_of_range():
    g = _valid_graph()
    g.events[0].locus = 99
    g.rebuild_index()
    assert "locus_out_of_range" in validate_sir(g, num_loci=7)


def test_rule_locus_collision():
    g = _valid_graph()
    g.events[1].locus = 0                                # same locus as event 0,
    g.events[1].referent = 2                             # different referent
    g.rebuild_index()
    assert "locus_collision" in validate_sir(g, num_loci=7)


def test_rule_duplicate_event_id():
    events = [_event(0, EventKind.MANUAL, 5, 0.0, 1.0),
              _event(0, EventKind.MANUAL, 6, 1.0, 2.0)]
    g = SIRGraph(events=events, edges=[])
    assert "duplicate_event_id" in validate_sir(g)


def test_hallucination_rule_with_lexicon():
    class _Lex:
        def __init__(self, entries): self.entries = set(entries)
        def contains(self, x): return x in self.entries

    g = _valid_graph()   # labels 5 and 9
    assert "hallucinated_manual_event" in validate_sir(g, lexicon=_Lex({5}))
    assert "hallucinated_manual_event" not in validate_sir(g, lexicon=_Lex({5, 9}))


def test_fingerspelled_event_is_not_hallucinated():
    class _Lex:
        def contains(self, x): return False              # nothing in lexicon
    events = [_event(0, EventKind.FINGERSPELL, 5, 0.0, 1.0)]
    g = SIRGraph(events=events, edges=[])
    assert "hallucinated_manual_event" not in validate_sir(g, lexicon=_Lex())


# ---------------------------------------------------------------------------
# Gloss projection = topological order
# ---------------------------------------------------------------------------
def test_gloss_projection_respects_precedence():
    g = _valid_graph()
    gloss = gloss_projection(g)
    assert gloss == [5, 9]                                # event 0 before 1


def test_gloss_projection_is_a_valid_topological_order():
    events = [_event(i, EventKind.MANUAL, 10 + i, float(i), float(i) + 1)
              for i in range(5)]
    # a diamond: 0 -> {1,2} -> 3, plus 3 -> 4
    edges = [SIREdge(0, 1, EdgeType.PRECEDENCE), SIREdge(0, 2, EdgeType.PRECEDENCE),
             SIREdge(1, 3, EdgeType.PRECEDENCE), SIREdge(2, 3, EdgeType.PRECEDENCE),
             SIREdge(3, 4, EdgeType.PRECEDENCE)]
    g = SIRGraph(events=events, edges=edges)
    gloss = gloss_projection(g)
    # recover the order of ids to check topological validity
    label_to_id = {e.label: e.id for e in events}
    order = [label_to_id[l] for l in gloss]
    assert is_topological_order(order, g)
    assert order[0] == 0 and order[-1] == 4              # source first, sink last


def test_gloss_projection_excludes_nonmanual_events():
    """The gloss contains exactly the manual events, never the non-manual ones."""
    # give the non-manual event a label that also appears manually would be
    # ambiguous, so use a label unique to the non-manual event and assert it
    # is absent from the gloss.
    events = [
        _event(0, EventKind.MANUAL, 5, 0.0, 1.0),
        _event(1, EventKind.MANUAL, 9, 1.0, 2.0),
        _event(2, EventKind.NONMANUAL, 42, 0.0, 2.0),     # unique label 42
    ]
    g = SIRGraph(events=events, edges=[SIREdge(0, 1, EdgeType.PRECEDENCE)])
    gloss = gloss_projection(g)
    assert 42 not in gloss                                # non-manual excluded
    assert gloss == [5, 9]
    assert len(gloss) == len(g.manual_events())


def test_gloss_projection_is_deterministic():
    g = _valid_graph()
    assert gloss_projection(g) == gloss_projection(g)


def test_gloss_projection_raises_on_cyclic_precedence():
    events = [_event(0, EventKind.MANUAL, 5, 0.0, 1.0),
              _event(1, EventKind.MANUAL, 9, 1.0, 2.0)]
    edges = [SIREdge(0, 1, EdgeType.PRECEDENCE), SIREdge(1, 0, EdgeType.PRECEDENCE)]
    with pytest.raises(ValueError):
        gloss_projection(SIRGraph(events=events, edges=edges))


def test_is_topological_order_detects_violation():
    g = _valid_graph()
    assert is_topological_order([0, 1], g)
    assert not is_topological_order([1, 0], g)           # 0 must precede 1
