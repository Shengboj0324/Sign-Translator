from dataclasses import replace
from fractions import Fraction
import json

import pytest
import torch

from signtranslator.grammar.sir import EdgeType, sir_from_dict
from signtranslator.planning.graph_decode import DiagnosticRelationThresholds, decode_diagnostic_graph
from signtranslator.planning.text_timing import TimedLabel, TemporalSequenceCandidate
from signtranslator.planning.text_relations import RelationalSequenceCandidate
from signtranslator.planning.text_referents import ReferentSequenceCandidate
from signtranslator.planning.text_loci import LocusSequenceCandidate
from signtranslator.planning.tensors import EDGE_TYPES
from test_loci import fixture as locus_fixture


def fixture(tmp_path):
    vocab, _, alphabet, _ = locus_fixture(tmp_path)
    temporal = TemporalSequenceCandidate(tuple(TimedLabel(label, 0., 1.) for label in vocab.entries),
                                         'terminated', (), vocab.lexicon.sha256, 0.)
    edges = torch.full((3, 3, 5), -4.)
    edges[1, 0, EDGE_TYPES.index(EdgeType.SCOPE)] = 4.
    relational = RelationalSequenceCandidate(temporal, edges, ~torch.eye(3, dtype=torch.bool))
    refs = torch.tensor([[0., 4., -4.], [4., 0., -4.], [-4., -4., 0.]])
    reference = ReferentSequenceCandidate(relational, refs, ~torch.eye(3, dtype=torch.bool))
    candidate = LocusSequenceCandidate(reference, torch.tensor([[8., 0.], [8., 0.], [0., 8.]]),
                                       alphabet.convention.sha256, alphabet.identities)
    return candidate, vocab, alphabet


def decode(c, v, a, **kwargs):
    options = dict(relation_thresholds=DiagnosticRelationThresholds((-1.,) * 5, (1.,) * 5),
                   place=(True, False, True), source_extent=(0., 1.),
                   referent_weight=Fraction(1), locus_weight=Fraction(1), max_events=10, max_work=10000)
    options.update(kwargs)
    return decode_diagnostic_graph(c, v, a, **options)


def test_full_candidate_graph_and_immutable_output(tmp_path):
    c, v, a = fixture(tmp_path)
    result = decode(c, v, a)
    assert result.status == 'uncalibrated_structural_candidate'
    graph = sir_from_dict(json.loads(result.graph_json))
    assert [e.referent for e in graph.events] == [0, 0, 1]
    assert [e.locus for e in graph.events] == [0, None, 1]
    assert len(graph.edges) == 1 and graph.edges[0].type is EdgeType.SCOPE
    before = result.graph_json
    c.referential.relational.relation_logits.fill_(float('nan'))
    assert result.graph_json == before
    with pytest.raises(ValueError):
        decode(c, v, a)


@pytest.mark.parametrize('boundary', [-1., 0., 1.])
def test_relation_ambiguity_including_exact_boundaries_never_returns_graph(tmp_path, boundary):
    c, v, a = fixture(tmp_path)
    c.referential.relational.relation_logits[0, 2, 0] = boundary
    result = decode(c, v, a)
    assert result.status == 'relation_ambiguous' and result.graph_json is None
    assert result.ambiguous_relations == 1


def test_positive_contradiction_is_refused_not_silently_deleted(tmp_path):
    c, v, a = fixture(tmp_path)
    c.referential.relational.relation_logits[0, 2, EDGE_TYPES.index(EdgeType.PRECEDENCE)] = 4.
    result = decode(c, v, a)
    assert result.status == 'structure_rejected' and result.graph_json is None
    assert 'precedence_time_contradiction' in result.violations
    assert c.referential.relational.relation_logits[0, 2, EDGE_TYPES.index(EdgeType.PRECEDENCE)] == 4.


def test_float64_threshold_is_not_rounded_to_float32_score(tmp_path):
    c, v, a = fixture(tmp_path)
    scores = c.referential.relational.relation_logits
    scores[0, 2, EDGE_TYPES.index(EdgeType.OVERLAP)] = 1.
    threshold = 1. - 2.**-30
    result = decode(c, v, a, relation_thresholds=DiagnosticRelationThresholds((-1.,) * 5, (threshold,) * 5))
    assert result.status == 'uncalibrated_structural_candidate'
    assert len(sir_from_dict(json.loads(result.graph_json)).edges) == 2


def test_spatial_budget_scope_and_threshold_contracts(tmp_path):
    c, v, a = fixture(tmp_path)
    assert decode(c, v, a, max_work=1).status == 'spatial_search_exhausted'
    assert decode(c, v, a, source_extent=(0., .5)).status == 'structure_rejected'
    with pytest.raises(ValueError):
        decode(replace(c, convention_sha256='f'*64), v, a)
    for lo, hi in [((1.,)*5, (1.,)*5), ((-1.,)*4, (1.,)*5), ((float('nan'),)*5, (1.,)*5)]:
        with pytest.raises(ValueError):
            DiagnosticRelationThresholds(lo, hi)


def test_diagnostic_graph_path_accepts_proven_128_event_search(tmp_path):
    c, v, a = fixture(tmp_path)
    n = 128
    temporal = replace(c.referential.relational.temporal,
                       events=(TimedLabel(v.entries[0], 0., 1.),) * n)
    relational = replace(c.referential.relational, temporal=temporal,
                         relation_logits=torch.full((n, n, 5), -4.),
                         relation_valid=~torch.eye(n, dtype=torch.bool))
    reference = replace(c.referential, relational=relational,
                        equality_logits=torch.ones(n, n) - torch.eye(n),
                        pair_valid=~torch.eye(n, dtype=torch.bool))
    c = replace(c, referential=reference, locus_logits=torch.zeros(n, 2))
    result = decode(c, v, a, place=(False,) * n, max_events=128, max_work=10000)
    assert result.status == 'uncalibrated_structural_candidate'
    graph = sir_from_dict(json.loads(result.graph_json))
    assert len(graph.events) == n and not graph.edges
    assert all(e.referent == 0 and e.locus is None for e in graph.events)
