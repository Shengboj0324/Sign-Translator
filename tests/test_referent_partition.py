from dataclasses import replace
from fractions import Fraction
from itertools import product

import pytest
import torch

from signtranslator.planning.referent_partition import decode_referent_partition
from signtranslator.planning.text_referents import ReferentSequenceCandidate
from test_candidate_graph import fixture


def candidate(matrix):
    _, base, _ = fixture()
    n = len(matrix)
    temporal = replace(base.temporal, events=(base.temporal.events[0],) * n)
    relational = replace(base, temporal=temporal)
    return ReferentSequenceCandidate(relational, torch.tensor(matrix, dtype=torch.float64),
                                     ~torch.eye(n, dtype=torch.bool))


def decode(c, **kwargs):
    return decode_referent_partition(c, max_events=kwargs.get('max_events', 10),
                                    max_search_nodes=kwargs.get('max_search_nodes', 1_000_000))


def test_transitivity_can_override_positive_pair_scores():
    c = candidate([[0, 5, -20], [5, 0, 4], [-20, 4, 0]])
    r = decode(c)
    assert r.status == 'unique_optimum_candidate' and r.referents == (0, 0, 1)
    assert r.best_gain == 5 and r.runner_up_gain == 4 and r.objective_gap == 1
    assert 2 <= r.evaluated_partitions <= 5


def test_exact_tie_and_tiny_nonzero_gap_are_not_rounded_away():
    r = decode(candidate([[0, 0], [0, 0]]))
    assert r.status == 'ambiguous' and r.referents is None and r.objective_gap == 0
    tiny = float.fromhex('0x0.0000000000001p-1022')
    r = decode(candidate([[0, tiny], [tiny, 0]]))
    assert r.referents == (0, 0) and r.objective_gap == Fraction.from_float(tiny)
    r = decode(candidate([[0, 1e308, 1e308], [1e308, 0, 1e308], [1e308, 1e308, 0]]))
    assert r.referents == (0, 0, 0) and r.best_gain == 3 * Fraction.from_float(1e308)


def test_exhaustive_oracle_and_bell_counts():
    for n, bell in [(1, 1), (2, 2), (3, 5), (4, 15), (5, 52)]:
        torch.manual_seed(n)
        raw = torch.randint(-5, 6, (n, n))
        scores = (raw + raw.T).tolist()
        r = decode(candidate(scores))
        # Independent oracle over all n^n assignments, deduplicated by pair equivalence.
        objectives = {}
        for labels in product(range(n), repeat=n):
            equal = tuple(labels[i] == labels[j] for i in range(n) for j in range(i+1, n))
            objectives[equal] = sum(scores[i][j] for i in range(n) for j in range(i+1, n) if labels[i] == labels[j])
        values = sorted(objectives.values(), reverse=True)
        assert len(objectives) == bell
        assert 1 <= r.evaluated_partitions <= bell
        assert r.best_gain == values[0]
        assert r.runner_up_gain == (values[1] if len(values) > 1 else None)
        assert (r.referents is None) == (len(values) > 1 and values[0] == values[1])


def test_resource_exhaustion_never_returns_partial_optimum():
    c = candidate([[0, 5, -20], [5, 0, 4], [-20, 4, 0]])
    full = decode(c)
    for budget in range(1, full.visited_nodes):
        r = decode(c, max_search_nodes=budget)
        assert r.status == 'search_exhausted'
        assert r.referents is None and r.best_gain is None
        assert r.visited_nodes == budget
    assert decode(c, max_search_nodes=full.visited_nodes) == full
    assert decode(c, max_events=2).status == 'capacity_exceeded'


def test_malformed_mutable_inputs_and_failed_generation():
    c = candidate([[0, 1], [1, 0]])
    for bad in (replace(c, equality_logits=torch.ones(3, 3)),
                replace(c, equality_logits=torch.tensor([[0., 1.], [2., 0.]])),
                replace(c, pair_valid=torch.ones(2, 2, dtype=torch.bool))):
        with pytest.raises(ValueError):
            decode(bad)
    with pytest.raises(ValueError):
        decode(c, max_events=True)
    before = decode(c)
    c.equality_logits.fill_(float('nan'))
    assert before.referents == (0, 0)
    with pytest.raises(ValueError):
        decode(c)
    failed = replace(c, relational=replace(c.relational, temporal=replace(c.relational.temporal, status='capacity_exceeded')))
    assert decode(failed).status == 'label_decoding_not_terminated'


def test_large_structured_cases_have_proven_best_and_runner_up():
    n = 128
    c = candidate([[0. if i == j else 1. for j in range(n)] for i in range(n)])
    r = decode(c, max_events=128)
    assert r.status == 'unique_optimum_candidate' and r.referents == (0,) * n
    assert r.best_gain == n * (n - 1) // 2
    # Cheapest nontrivial cut isolates one event from the complete positive graph.
    assert r.objective_gap == n - 1
    assert r.evaluated_partitions < 100 and r.visited_nodes < 20000
    n = 32
    c = candidate([[0. if i == j else -1. for j in range(n)] for i in range(n)])
    r = decode(c, max_events=128)
    assert r.referents == tuple(range(n)) and r.best_gain == 0 and r.runner_up_gain == -1
    r = decode(candidate([[0.] * n for _ in range(n)]), max_events=128)
    assert r.status == 'ambiguous' and r.referents is None and r.objective_gap == 0


def test_pruning_matches_random_exhaustive_full_assignment_oracle():
    n = 5
    for seed in range(20):
        torch.manual_seed(seed + 900)
        raw = torch.randint(-20, 21, (n, n))
        scores = (raw + raw.T).tolist()
        states = {}
        for labels in product(range(n), repeat=n):
            equal = tuple(labels[i] == labels[j] for i in range(n) for j in range(i+1, n))
            states[equal] = sum(scores[i][j] for i in range(n) for j in range(i+1, n) if labels[i] == labels[j])
        expected = sorted(states.values(), reverse=True)
        r = decode(candidate(scores))
        assert r.best_gain == expected[0] and r.runner_up_gain == expected[1]
        assert (r.referents is None) == (expected[0] == expected[1])
