from dataclasses import replace

import torch

from signtranslator.planning.graph_decode import DiagnosticRelationThresholds
from signtranslator.planning.relation_evaluation import evaluate_relation_sequences
from signtranslator.planning.relations import relation_targets
from signtranslator.planning.tensors import EDGE_TYPES
from test_relation_evaluation import fixture


def test_threshold_counts_match_scalar_oracle_with_abstention_and_unknowns(tmp_path):
    v, a, candidates = fixture(tmp_path)
    thresholds = DiagnosticRelationThresholds((-0.25,) * 5, (0.25,) * 5)
    changed = []
    for c in candidates:
        scores = c.relation_logits.double().clone()
        for i in range(len(scores)):
            for j in range(len(scores)):
                for k in range(5):
                    scores[i, j, k] = (-1., -.25, 0., .25, 1.)[(i * 3 + j + k) % 5]
        changed.append(replace(c, relation_logits=scores))
    result = evaluate_relation_sequences(tuple(changed), a, v,
        annotation_sha256=tuple(x.content_sha256() for x in a), max_cells=100,
        max_relation_cells=50, relation_thresholds=thresholds).to_dict()
    targets = relation_targets(a, v)
    assert result['schema_version'] == 2
    for k, kind in enumerate(EDGE_TYPES):
        counts = result['threshold_diagnostic']['conditional_cell_weighted'][kind.value]
        expected = {key: 0 for key in counts if key not in ('known_decision_coverage', 'conditional_error')}
        for row, c in enumerate(changed):
            for i in range(len(c.relation_logits)):
                for j in range(len(c.relation_logits)):
                    if i == j:
                        continue
                    value = float(c.relation_logits[i, j, k])
                    predicted = 1 if value > .25 else 0 if value < -.25 else None
                    known = bool(targets.known[row, i, j, k])
                    positive = bool(targets.positive[row, i, j, k])
                    if not known:
                        key = {1: 'unknown_positive', 0: 'unknown_negative', None: 'unknown_undecided'}[predicted]
                    elif predicted is None:
                        key = 'undecided_positive' if positive else 'undecided_negative'
                    else:
                        key = ('true_' if bool(predicted) == positive else 'false_') + ('positive' if predicted else 'negative')
                    expected[key] += 1
        assert all(counts[key] == value for key, value in expected.items())
        decided = sum(expected[x] for x in ('true_positive', 'true_negative', 'false_positive', 'false_negative'))
        assert counts['known_decision_coverage'] == dict(numerator=decided,
            denominator=decided + expected['undecided_positive'] + expected['undecided_negative'])
        assert counts['conditional_error'] == dict(numerator=expected['false_positive'] + expected['false_negative'], denominator=decided)


def test_all_boundary_equal_scores_abstain_and_default_report_stays_schema_one(tmp_path):
    v, a, c = fixture(tmp_path)
    kwargs = dict(annotation_sha256=tuple(x.content_sha256() for x in a), max_cells=100, max_relation_cells=50)
    baseline = evaluate_relation_sequences(c, a, v, **kwargs).to_dict()
    assert baseline['schema_version'] == 1 and 'threshold_diagnostic' not in baseline
    report = evaluate_relation_sequences(c, a, v, **kwargs,
        relation_thresholds=DiagnosticRelationThresholds((-1.,) * 5, (0.,) * 5)).to_dict()
    assert report['conditional_cell_weighted'] == baseline['conditional_cell_weighted']
    for counts in report['threshold_diagnostic']['conditional_cell_weighted'].values():
        assert counts['conditional_error'] == dict(numerator=0, denominator=0)
        assert counts['known_decision_coverage']['numerator'] == 0


def test_float32_scores_are_compared_to_unrounded_float64_thresholds(tmp_path):
    v, a, candidates = fixture(tmp_path)
    candidates = tuple(replace(c, relation_logits=torch.full_like(c.relation_logits, .1)) for c in candidates)
    value = float(torch.tensor(.1, dtype=torch.float32))
    report = evaluate_relation_sequences(candidates, a, v,
        annotation_sha256=tuple(x.content_sha256() for x in a), max_cells=100, max_relation_cells=50,
        relation_thresholds=DiagnosticRelationThresholds((-1.,) * 5, (value - 1e-12,) * 5)).to_dict()
    for kind in EDGE_TYPES:
        target = report['conditional_cell_weighted'][kind.value]
        counts = report['threshold_diagnostic']['conditional_cell_weighted'][kind.value]
        assert counts['true_positive'] == target['positive']
        assert counts['false_positive'] == target['negative']
        assert counts['unknown_positive'] == target['unknown']
        assert counts['known_decision_coverage'] == dict(numerator=target['known'], denominator=target['known'])


def test_failed_generations_have_unavailable_decision_rows(tmp_path):
    v, a, candidates = fixture(tmp_path)
    failed = tuple(replace(c, temporal=replace(c.temporal, events=(), diagnostic_prefix=(), status='empty_prediction'),
                           relation_logits=None, relation_valid=None) for c in candidates)
    report = evaluate_relation_sequences(failed, a, v,
        annotation_sha256=tuple(x.content_sha256() for x in a), max_cells=100, max_relation_cells=50,
        relation_thresholds=DiagnosticRelationThresholds((-1.,) * 5, (1.,) * 5)).to_dict()
    assert report['threshold_diagnostic']['rows'] == [None, None]
    assert report['coverage'] == dict(numerator=0, denominator=2)
    for counts in report['threshold_diagnostic']['conditional_cell_weighted'].values():
        assert counts['known_decision_coverage'] == dict(numerator=0, denominator=0)
