"""Historical relation types must not borrow support from another type."""
from copy import deepcopy
import hashlib

import pytest

from signtranslator.planning.relation_exposure import summarize_relation_exposure
from signtranslator.reproducibility import canonical_json_bytes


WEIGHTS = {'sir_relations': 1., 'sir_sequence': 1.}
IDENTITIES = {'a': 'a' * 64, 'b': 'b' * 64}


def records():
    a = [[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 0]]
    b = [[0, 1, 2, 1], [1, 0, 2, 0]]
    result = []
    for step, (ids, examples) in enumerate([(['a', 'b'], [a, b]), (['a'], [a]),
                                            (['b'], [[]]), (['a'], None)], 1):
        declaration = (dict(axes=['source_event', 'target_event', 'relation_type'],
                            class_count=2, examples=deepcopy(examples)) if examples is not None else None)
        result.append(dict(step=step, sample_ids=ids, annotation_sha256=[IDENTITIES[x] for x in ids],
            support={'sir_relations': sum(bool(x) for x in examples) if examples is not None else 1,
                     'sir_sequence': len(ids)}, weights=WEIGHTS,
            target_cells={'sir_relations': declaration, 'sir_sequence': None}))
    return result


def summarize(rows, **kwargs):
    options = dict(global_step=len(rows), weights=WEIGHTS, identities=IDENTITIES, max_events=3,
        expected_ledger_sha256=hashlib.sha256(canonical_json_bytes(rows)).hexdigest())
    options.update(kwargs)
    return summarize_relation_exposure(iter(rows), **options)


def test_polarities_types_repeated_samples_and_missing_declarations_stay_distinct():
    result = summarize(records())
    assert result['recorded_example_presentations'] == 4
    assert result['empty_recorded_example_presentations'] == 1
    assert result['unrecorded_example_presentations'] == 1
    types = result['relation_types']
    assert [(x['negative_cell_presentations'], x['positive_cell_presentations']) for x in types] == [
        (2, 2), (2, 0), (1, 1), (0, 0), (0, 0)]
    assert [(x['distinct_negative_samples'], x['distinct_positive_samples']) for x in types] == [
        (1, 1), (1, 0), (1, 1), (0, 0), (0, 0)]
    assert [x['recorded_polarities'] for x in types] == [
        ['negative', 'positive'], ['negative'], ['negative', 'positive'], [], []]
    assert not result['phase_exit_approved']
    assert summarize(records()[:1])['relation_types'][0]['positive_cell_presentations'] == 1


def test_no_steps_or_count_only_history_does_not_invent_type_support():
    empty = summarize([])
    assert empty['recorded_optimizer_steps'] == 0
    assert all(row['recorded_polarities'] == [] for row in empty['relation_types'])
    old = records()
    for row in old:
        del row['target_cells']
    result = summarize(old)
    assert result['recorded_example_presentations'] == 0
    assert result['unrecorded_example_presentations'] == 5
    assert all(row['recorded_polarities'] == [] for row in result['relation_types'])


@pytest.mark.parametrize('fault', ['axes', 'classes', 'self', 'event', 'type', 'hash', 'steps'])
def test_invalid_domain_or_ledger_identity_is_refused(fault):
    rows = records()
    declaration = rows[0]['target_cells']['sir_relations']
    options = {}
    if fault == 'axes':
        declaration['axes'][0] = 'unknown_axis'
    elif fault == 'classes':
        declaration['class_count'] = 3
    elif fault == 'self':
        declaration['examples'][0][0][1] = 0
    elif fault == 'event':
        declaration['examples'][0][-1][0] = 3
    elif fault == 'type':
        declaration['examples'][0][-1][2] = 5
    elif fault == 'hash':
        options['expected_ledger_sha256'] = '0' * 64
    else:
        options['global_step'] = 3
    with pytest.raises(ValueError):
        summarize(rows, **options)


def test_projection_uses_model_capacity_not_graph_search_cap():
    rows = records()[:1]
    rows[0]['target_cells']['sir_relations']['examples'][0][-1][0] = 200
    assert summarize(rows, max_events=256)['relation_types'][0]['negative_cell_presentations'] == 1
    with pytest.raises(ValueError, match='nonself domain'):
        summarize(rows, max_events=200)
