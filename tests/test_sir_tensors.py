"""Exact graph target round-trips with fictional governed annotations."""
from dataclasses import replace

import pytest
import torch

from signtranslator.grammar.sir import (
    EdgeType, EventKind, SIREdge, SIREvent, SIRGraph, sir_sha256,
)
from signtranslator.planning.supervision import GovernedSIRAnnotation
from signtranslator.planning.tensors import EVENT_KINDS, EDGE_TYPES, tensorize_sir_annotations
from test_phase3_governance import _annotation, _sample


def annotation(graph):
    base = _annotation(0, _sample(0))
    return GovernedSIRAnnotation.create(
        annotation_id=base.annotation_id, origin=base.origin, source=base.source,
        convention=base.convention, lexicon=base.lexicon,
        lexicon_convention_sha256=base.lexicon_convention_sha256, graph=graph,
        review=replace(base.review, reviewed_sir_sha256=sir_sha256(graph)),
        created_at=base.created_at)


def graph_fixture():
    return SIRGraph([
        SIREvent(42, EventKind.MANUAL, 999, 0., .5, referent=0, locus=0),
        SIREvent(7, EventKind.CLASSIFIER, 123, .5, 1., referent=0, locus=0),
        SIREvent(80, EventKind.FINGERSPELL, 88, .25, .75, referent=1, locus=1),
        SIREvent(15, EventKind.NONMANUAL, 9, 0., 1.),
    ], [SIREdge(42, 7, EdgeType.PRECEDENCE), SIREdge(42, 80, EdgeType.OVERLAP),
        SIREdge(15, 42, EdgeType.SCOPE), SIREdge(42, 7, EdgeType.COREF),
        SIREdge(42, 7, EdgeType.LOCUS)])


def encode(annotations):
    return tensorize_sir_annotations(annotations, expected_lexicon=annotations[0].lexicon,
                                     expected_convention=annotations[0].convention)


def decode_row(targets, row):
    events = []
    for index in torch.nonzero(targets.event_valid[row]).flatten().tolist():
        events.append(SIREvent(
            int(targets.event_ids[row, index]), EVENT_KINDS[int(targets.event_kinds[row, index])],
            int(targets.label_ids[row, index]), *targets.intervals[row, index].tolist(),
            referent=int(targets.referent_ids[row, index]) if targets.referent_present[row, index] else None,
            locus=int(targets.locus_ids[row, index]) if targets.locus_present[row, index] else None))
    edges = []
    for index in torch.nonzero(targets.edge_valid[row]).flatten().tolist():
        source, target = targets.edge_indices[row, index].tolist()
        edges.append(SIREdge(events[source].id, events[target].id,
                             EDGE_TYPES[int(targets.edge_types[row, index])]))
    return SIRGraph(events, edges)


def test_roundtrip_all_fields_event_kinds_and_edge_relations_with_padding():
    a = annotation(graph_fixture())
    b = _annotation(1, _sample(1))
    targets = encode([a, b])
    assert targets.intervals.dtype == torch.float64
    for row, expected in enumerate((a, b)):
        assert sir_sha256(decode_row(targets, row)) == expected.sir_payload_sha256
    assert targets.event_ids[0].tolist() == [7, 15, 42, 80]
    assert targets.label_ids[0].tolist() == [123, 9, 999, 88]
    assert targets.referent_ids[0, 0] == 0 and targets.referent_present[0, 0]
    assert targets.referent_ids[0, 1] == -1 and not targets.referent_present[0, 1]
    assert targets.event_ids[1].tolist() == [1, -1, -1, -1]
    assert not targets.event_valid[1, 1:].any()
    assert not targets.edge_valid[1].any()
    assert (targets.edge_indices[1] == -1).all()
    assert targets.annotation_sha256 == (a.content_sha256(), b.content_sha256())


def test_no_edges_is_a_valid_zero_length_axis():
    targets = encode([_annotation(0, _sample(0))])
    assert targets.edge_indices.shape == (1, 0, 2)
    assert targets.edge_valid.shape == (1, 0)


@pytest.mark.parametrize('field', ['id', 'label', 'referent', 'locus'])
def test_identifiers_outside_int64_are_rejected_without_truncation(field):
    event = SIREvent(0, EventKind.MANUAL, 0, 0., 1.)
    setattr(event, field, 2**63)
    a = annotation(SIRGraph([event]))
    with pytest.raises(ValueError, match='int64'):
        encode([a])


def test_maximum_int64_identity_is_retained_exactly():
    maximum = 2**63 - 1
    a = annotation(SIRGraph([SIREvent(maximum, EventKind.MANUAL, maximum, 0., 1.)]))
    targets = encode([a])
    assert targets.event_ids.item() == maximum
    assert targets.label_ids.item() == maximum


def test_vocabulary_mismatch_cannot_remap_labels():
    a = _annotation(0, _sample(0))
    b = _annotation(1, _sample(1), lexicon_name='different')
    with pytest.raises(ValueError, match='binding'):
        encode([a, b])
    with pytest.raises(ValueError, match='binding'):
        tensorize_sir_annotations([a], expected_lexicon=b.lexicon, expected_convention=a.convention)
    with pytest.raises(ValueError, match='typed'):
        tensorize_sir_annotations([a], expected_lexicon=a.convention, expected_convention=a.convention)


def test_overflowing_duration_is_rejected():
    a = annotation(SIRGraph([SIREvent(0, EventKind.MANUAL, 0, -1e308, 1e308)]))
    with pytest.raises(ValueError, match='duration'):
        encode([a])


def test_empty_annotations_cannot_make_a_target_batch():
    a = _annotation(0, _sample(0))
    with pytest.raises(ValueError, match='nonempty'):
        tensorize_sir_annotations([], expected_lexicon=a.lexicon, expected_convention=a.convention)
