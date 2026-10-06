from dataclasses import replace
from fractions import Fraction

import pytest

from signtranslator.planning.temporal_evaluation import evaluate_temporal_sequences
from signtranslator.planning.text_timing import TemporalSequenceCandidate, TimedLabel
from signtranslator.planning.label_vocabulary import encode_governed_labels
from signtranslator.planning.tensors import tensorize_sir_annotations
from test_text_sir_labels import setup


def fixture(tmp_path):
    v, corpus = setup(tmp_path, configurations=[{}, {}], single_event_indices=(1,))
    a = tuple(r.annotation for r in corpus._records)
    ids = encode_governed_labels(a, v)
    t = tensorize_sir_annotations(a, expected_lexicon=v.lexicon, expected_convention=v.convention)
    c = tuple(TemporalSequenceCandidate(
        tuple(TimedLabel(v.entries[int(label)], *t.intervals[row, col].tolist())
              for col, label in enumerate(ids[row]) if label >= 0),
        'terminated', tuple(v.entries[int(i)] for i in ids[row] if i >= 0), v.lexicon.sha256, -10.)
        for row in range(2))
    return v, a, c


def evaluate(v, a, c):
    return evaluate_temporal_sequences(c, a, v, annotation_sha256=tuple(x.content_sha256() for x in a),
                                       origins_seconds=(-10.,) * len(c), max_cells=100)


def fraction(value):
    return Fraction(value['numerator'], value['denominator'])


def test_exact_intervals_and_conditional_failure_coverage(tmp_path):
    v, a, c = fixture(tmp_path)
    report = evaluate(v, a, c)
    data = report.to_dict()
    assert data['coverage'] == dict(numerator=2, denominator=2)
    assert fraction(data['conditional_event_weighted']['mean_interval_iou']) == 1
    assert fraction(data['conditional_event_weighted']['endpoint_mae_seconds']) == 0
    data['rows'].clear()
    assert len(report.to_dict()['rows']) == 2
    fail = replace(c[0], events=(), diagnostic_prefix=(), status='empty_prediction')
    changed = replace(c[1], events=(replace(c[1].events[0], label=v.entries[2]),), diagnostic_prefix=(v.entries[2],))
    data = evaluate(v, a, (fail, changed)).to_dict()
    assert data['coverage'] == dict(numerator=0, denominator=2)
    assert data['conditional_event_weighted'] is None
    assert all(row['timing'] is None for row in data['rows'])


def test_event_weighting_exact_shift_and_disjoint_iou(tmp_path):
    v, a, c = fixture(tmp_path)
    shifted = replace(c[1], events=tuple(replace(e, start_seconds=e.start_seconds+100.,
                                               end_seconds=e.end_seconds+100.) for e in c[1].events))
    data = evaluate(v, a, (c[0], shifted)).to_dict()
    # Three exact events and one translated event; aggregate is event-weighted.
    assert fraction(data['conditional_event_weighted']['endpoint_mae_seconds']) == 25
    assert fraction(data['conditional_event_weighted']['mean_interval_iou']) == Fraction(3, 4)


def test_origin_interval_and_binding_fail_closed(tmp_path):
    v, a, c = fixture(tmp_path)
    for bad in (replace(c[0], origin_seconds=0.),
                replace(c[0], events=(replace(c[0].events[0], end_seconds=float('nan')),)),
                replace(c[0], events=(replace(c[0].events[0], start_seconds=-11.),)),
                replace(c[0], events=(replace(c[0].events[0], end_seconds=-100.),))):
        with pytest.raises(ValueError):
            evaluate(v, a, (bad, c[1]))
    with pytest.raises(ValueError, match='binding'):
        evaluate_temporal_sequences(c, a, v, annotation_sha256=('0'*64,)*2,
                                    origins_seconds=(-10., -10.), max_cells=100)


@pytest.mark.parametrize('start,end,iou,mae', [
    (.5, 1.5, Fraction(1, 3), Fraction(1, 2)),
    (.25, .75, Fraction(1, 2), Fraction(1, 4)),
    (1., 2., Fraction(0), Fraction(1)),
    (2., 3., Fraction(0), Fraction(2)),
    (-1., 2., Fraction(1, 3), Fraction(1)),
])
def test_overlap_containment_touching_and_disjoint_intervals(tmp_path, start, end, iou, mae):
    v, a, c = fixture(tmp_path)
    changed = replace(c[1], events=(replace(c[1].events[0], start_seconds=start, end_seconds=end),))
    data = evaluate(v, a[1:], (changed,)).to_dict()['conditional_event_weighted']
    assert fraction(data['mean_interval_iou']) == iou
    assert fraction(data['endpoint_mae_seconds']) == mae


def test_finite_extreme_coordinates_do_not_overflow_aggregate(tmp_path):
    v, a, c = fixture(tmp_path)
    changed = replace(c[1], events=(replace(c[1].events[0], start_seconds=1e308, end_seconds=1.7e308),))
    data = evaluate(v, a[1:], (changed,)).to_dict()['conditional_event_weighted']
    assert fraction(data['endpoint_mae_seconds']) == (Fraction(1e308) + Fraction(1.7e308) - 1) / 2
    assert fraction(data['mean_interval_iou']) == 0
