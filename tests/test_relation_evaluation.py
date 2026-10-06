from dataclasses import replace
import math

import pytest
import torch

from signtranslator.planning.relation_evaluation import evaluate_relation_sequences, _scores
from signtranslator.planning.text_relations import RelationalSequenceCandidate
from signtranslator.planning.relations import relation_targets
from signtranslator.planning.tensors import EDGE_TYPES
from test_temporal_evaluation import fixture as temporal_fixture


def fixture(tmp_path):
    v, a, temporal = temporal_fixture(tmp_path)
    candidates = tuple(RelationalSequenceCandidate(t, torch.zeros(len(t.events), len(t.events), 5),
                         ~torch.eye(len(t.events), dtype=torch.bool)) for t in temporal)
    return v, a, candidates


def evaluate(v, a, c, budget=50):
    return evaluate_relation_sequences(c, a, v, annotation_sha256=tuple(x.content_sha256() for x in a),
                                       max_cells=100, max_relation_cells=budget)


def test_known_unknown_polarities_and_analytical_zero_scores(tmp_path):
    v, a, c = fixture(tmp_path)
    report = evaluate(v, a, c)
    data = report.to_dict()
    assert data['coverage'] == dict(numerator=2, denominator=2)
    targets = relation_targets(a, v)
    for k, relation in enumerate(EDGE_TYPES):
        row = data['conditional_cell_weighted'][relation.value]
        assert row['known'] == int(targets.known[..., k].sum())
        assert row['positive'] == int(targets.positive[..., k].sum())
        assert row['negative'] + row['positive'] == row['known']
        assert row['unknown'] + row['known'] == 6
        if row['known']:
            assert row['nll'] == pytest.approx(math.log(2))
            assert row['brier'] == .25
        else:
            assert row['nll'] is row['brier'] is None
    data['rows'].clear()
    assert len(report.to_dict()['rows']) == 2
    # Unknown logits cannot affect any scored metric.
    changed = c[0].relation_logits.clone()
    changed[~targets.known[0, :3, :3]] = 500.
    other = evaluate(v, a, (replace(c[0], relation_logits=changed), c[1])).to_dict()
    assert other['conditional_cell_weighted'] == report.to_dict()['conditional_cell_weighted']


def test_failure_mismatch_and_budget_do_not_create_supported_accuracy(tmp_path):
    v, a, c = fixture(tmp_path)
    failed = replace(c[0], temporal=replace(c[0].temporal, events=(), diagnostic_prefix=(), status='empty_prediction'),
                     relation_logits=None, relation_valid=None)
    t = c[1].temporal
    mismatch = replace(c[1], temporal=replace(t, events=(replace(t.events[0], label=v.entries[2]),),
                                            diagnostic_prefix=(v.entries[2],)))
    result = evaluate(v, a, (failed, mismatch)).to_dict()
    assert result['coverage'] == dict(numerator=0, denominator=2)
    assert all(row['nll'] is None for row in result['conditional_cell_weighted'].values())
    with pytest.raises(ValueError, match='budget'):
        evaluate(v, a, c, budget=49)
    bad_mask = c[0].relation_valid.clone(); bad_mask[0, 1] = False
    for changed in (replace(c[0], relation_valid=bad_mask),
                    replace(c[0], relation_logits=torch.full((3, 3, 5), float('nan'))),
                    replace(c[0], relation_types=EDGE_TYPES[::-1])):
        with pytest.raises(ValueError):
            evaluate(v, a, (changed, c[1]))


def test_stable_extreme_and_analytical_nonzero_scores():
    assert _scores(1e308, True) == (0., 0.)
    assert _scores(-1e308, False) == (0., 0.)
    assert _scores(-1e308, True) == (1e308, 1.)
    assert _scores(1e308, False) == (1e308, 1.)
    assert _scores(math.log(3), True) == pytest.approx((math.log(4/3), 1/16))
    assert _scores(math.log(3), False) == pytest.approx((math.log(4), 9/16))


@pytest.mark.parametrize('logit', [-350., -40., -20., 0., 20., 40., 350.])
@pytest.mark.parametrize('positive', [True, False])
def test_brier_matches_high_precision_oracle_and_sign_complement(logit, positive):
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 400
        probability = 1 / (1 + (-Decimal(logit)).exp())
        expected = float((probability - int(positive)) ** 2)
    actual = _scores(logit, positive)[1]
    assert math.isclose(actual, expected, rel_tol=3e-15, abs_tol=0.)
    assert actual == _scores(-logit, not positive)[1]
    if (logit >= 0) == positive:
        assert actual > 0.


def test_nonnegative_means_preserve_subnormal_values_and_avoid_overflow():
    from signtranslator.planning.relation_evaluation import _summary
    smallest = math.nextafter(0., 1.)
    tiny = _summary([(1e-320, smallest)] * 100, [], 0)
    assert tiny['nll'] == 1e-320
    assert tiny['brier'] == smallest
    largest = float.fromhex('0x1.fffffffffffffp+1023')
    large = _summary([(largest, 1.)] * 100, [], 0)
    assert large['nll'] == largest
    assert large['brier'] == 1.
    mixed = _summary([(largest, 1.)], [(0., 0.)], 0)
    assert mixed['nll'] == largest / 2
    assert mixed['brier'] == .5
