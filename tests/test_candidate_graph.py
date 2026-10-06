from dataclasses import replace
import json

import pytest
import torch

from signtranslator.grammar.sir import EdgeType, sir_from_dict
from signtranslator.planning.candidate_graph import assemble_candidate_graph
from signtranslator.planning.text_relations import RelationalSequenceCandidate
from signtranslator.planning.text_timing import TemporalSequenceCandidate, TimedLabel
from signtranslator.planning.tensors import EDGE_TYPES
from test_sir_label_vocabulary import setup


def fixture():
    vocab, _ = setup()
    label = vocab.entries[0]
    temporal = TemporalSequenceCandidate((TimedLabel(label, 0., 1.), TimedLabel(label, 1., 2.)),
                                         'terminated', (), vocab.lexicon.sha256, 0.)
    candidate = RelationalSequenceCandidate(temporal, torch.zeros(2, 2, 5), ~torch.eye(2, dtype=torch.bool))
    decisions = torch.zeros(2, 2, 5, dtype=torch.bool)
    return vocab, candidate, decisions


def run(vocab, candidate, decisions, **kwargs):
    options = dict(referents=(0, 0), loci=(0, 0), num_loci=2, source_extent=(0., 2.))
    options.update(kwargs)
    return assemble_candidate_graph(candidate, vocab, selected_edges=decisions, **options)


def test_explicit_graph_preserves_decisions_and_detaches_mutable_inputs():
    v, c, d = fixture()
    for relation in (EdgeType.PRECEDENCE, EdgeType.COREF, EdgeType.LOCUS):
        d[0, 1, EDGE_TYPES.index(relation)] = True
    result = run(v, c, d)
    assert result.violations == ()
    graph = sir_from_dict(json.loads(result.graph_json))
    assert len(graph.edges) == 3
    assert [e.referent for e in graph.events] == [0, 0]
    saved = result.graph_json
    d.fill_(False)
    c.relation_logits.fill_(float('nan'))
    assert result.graph_json == saved


@pytest.mark.parametrize('relation,kwargs,violation', [
    (EdgeType.OVERLAP, {}, 'overlap_time_contradiction'),
    (EdgeType.COREF, {'referents': (0, None)}, 'coref_referent_mismatch'),
    (EdgeType.LOCUS, {'loci': (0, None)}, 'locus_target_missing'),
    (EdgeType.PRECEDENCE, {'loci': (0, 2)}, 'locus_out_of_range'),
    (EdgeType.PRECEDENCE, {'referents': (0, 1)}, 'locus_collision'),
    (EdgeType.PRECEDENCE, {'source_extent': (0., 1.5)}, 'event_outside_source_extent'),
])
def test_inconsistent_proposal_refused_without_repair(relation, kwargs, violation):
    v, c, d = fixture()
    d[0, 1, EDGE_TYPES.index(relation)] = True
    result = run(v, c, d, **kwargs)
    assert result.graph_json is None and violation in result.violations
    assert d.sum() == 1


def test_unknown_fields_are_preserved_without_inventing_edges():
    v, c, d = fixture()
    result = run(v, c, d, referents=(None, None), loci=(None, None))
    graph = sir_from_dict(json.loads(result.graph_json))
    assert not graph.edges and all(e.referent is None and e.locus is None for e in graph.events)


def test_malformed_and_mutated_contracts_fail():
    v, c, d = fixture()
    for kwargs in ({'referents': (False, 0)}, {'num_loci': True}, {'source_extent': (1., 2.)}):
        with pytest.raises(ValueError):
            run(v, c, d, **kwargs)
    for bad in (replace(c, relation_types=tuple(reversed(EDGE_TYPES))),
                replace(c, relation_valid=torch.ones(2, 2, dtype=torch.bool)),
                replace(c, temporal=replace(c.temporal, vocabulary_sha256='f'*64))):
        with pytest.raises(ValueError):
            run(v, bad, d)
    with pytest.raises(ValueError):
        run(v, c, d.float())
    c.relation_logits[0, 1, 0] = float('nan')
    with pytest.raises(ValueError):
        run(v, c, d)


def test_failed_decoding_and_self_edges_have_no_graph():
    v, c, d = fixture()
    assert run(v, replace(c, temporal=replace(c.temporal, status='capacity_exceeded')), d).graph_json is None
    d[0, 0, 0] = True
    assert run(v, c, d).violations == ('self_edge',)
