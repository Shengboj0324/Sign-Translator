from dataclasses import replace
import math

import pytest
import torch

from signtranslator.planning.referent_evaluation import evaluate_referent_sequences
from signtranslator.planning.text_referents import ReferentSequenceCandidate
from signtranslator.planning.text_relations import RelationalSequenceCandidate
from signtranslator.planning.text_timing import TemporalSequenceCandidate, TimedLabel
from test_text_sir_labels import setup


def fixture(tmp_path, referents=(0, 0, 1)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    v, corpus = setup(tmp_path, configurations=[{}], referents=referents)
    a = (corpus._records[0].annotation,)
    graph = a[0].graph()
    temporal = TemporalSequenceCandidate(tuple(TimedLabel(label, event.t_start, event.t_end)
                                              for label, event in zip(v.entries, graph.events)),
                                         'terminated', v.entries, v.lexicon.sha256, 0.)
    relational = RelationalSequenceCandidate(temporal, None, None)
    candidate = ReferentSequenceCandidate(relational, torch.zeros(3, 3), ~torch.eye(3, dtype=torch.bool))
    return v, a, (candidate,)


def evaluate(v, a, c, budget=9):
    return evaluate_referent_sequences(c, a, v, annotation_sha256=tuple(x.content_sha256() for x in a),
                                       max_cells=100, max_pair_cells=budget)


def test_unordered_polarities_and_no_coref_edge_substitution(tmp_path):
    v, a, c = fixture(tmp_path)
    report = evaluate(v, a, c)
    data = report.to_dict()
    score = data['conditional_pair_weighted']
    assert (score['positive'], score['negative'], score['unknown'], score['known']) == (1, 2, 0, 3)
    assert score['nll'] == pytest.approx(math.log(2))
    assert score['brier'] == .25
    assert data['supported_example_coverage'] == dict(numerator=1, denominator=1)
    data['rows'].clear()
    assert len(report.to_dict()['rows']) == 1
    c[0].equality_logits.fill_(40.)
    changed = evaluate(v, a, c).to_dict()['conditional_pair_weighted']
    assert 0 < changed['positive_brier'] < 1e-30
    assert changed['negative_brier'] == 1.
    assert report.to_dict()['conditional_pair_weighted'] == score


def test_unknown_pairs_are_unavailable_and_label_mismatch_excluded(tmp_path):
    v, a, c = fixture(tmp_path, referents=(None, None, None))
    data = evaluate(v, a, c).to_dict()
    assert data['conditional_pair_weighted']['unknown'] == 3
    assert data['conditional_pair_weighted']['nll'] is None
    assert data['supported_example_coverage'] == dict(numerator=0, denominator=1)
    assert data['label_coverage'] == dict(numerator=1, denominator=1)
    temporal = c[0].relational.temporal
    changed = replace(temporal, events=tuple(replace(e, label=v.entries[2]) for e in temporal.events),
                      diagnostic_prefix=(v.entries[2],)*3)
    candidate = replace(c[0], relational=replace(c[0].relational, temporal=changed))
    assert evaluate(v, a, (candidate,)).to_dict()['rows'][0]['referent_equality'] is None


def test_unknown_endpoint_mask_and_renaming_invariance(tmp_path):
    v, a, c = fixture(tmp_path / 'first', referents=(0, 0, None))
    x = evaluate(v, a, c).to_dict()['conditional_pair_weighted']
    v2, a2, c2 = fixture(tmp_path / 'second', referents=(1234, 1234, None))
    assert evaluate(v2, a2, c2).to_dict()['conditional_pair_weighted'] == x
    assert (x['positive'], x['negative'], x['unknown']) == (1, 0, 2)


def test_masks_symmetry_budget_and_failed_generation(tmp_path):
    v, a, c = fixture(tmp_path)
    with pytest.raises(ValueError, match='budget'):
        evaluate(v, a, c, budget=8)
    asymmetric = c[0].equality_logits.clone(); asymmetric[0, 1] = 1.
    mask = c[0].pair_valid.clone(); mask[0, 1] = mask[1, 0] = False
    for bad in (replace(c[0], equality_logits=asymmetric), replace(c[0], pair_valid=mask)):
        with pytest.raises(ValueError):
            evaluate(v, a, (bad,))
    temporal = replace(c[0].relational.temporal, events=(), diagnostic_prefix=(), status='empty_prediction')
    bad = replace(c[0], relational=replace(c[0].relational, temporal=temporal))
    with pytest.raises(ValueError, match='failed generation'):
        evaluate(v, a, (bad,))
    failed = replace(bad, equality_logits=None, pair_valid=None)
    data = evaluate(v, a, (failed,)).to_dict()
    assert data['label_coverage'] == dict(numerator=0, denominator=1)
    assert data['conditional_pair_weighted']['nll'] is None
