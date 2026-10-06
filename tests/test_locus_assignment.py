from dataclasses import replace
from fractions import Fraction
from itertools import permutations
import hashlib
import json

import pytest
import torch

from signtranslator.planning.loci import LocusAlphabet
from signtranslator.planning.locus_assignment import decode_locus_assignment
from signtranslator.planning.text_loci import LocusSequenceCandidate
from test_referent_partition import candidate as reference_candidate
from test_sir_label_vocabulary import setup


def fixture(scores):
    v, _ = setup()
    payload = json.dumps({'locus_schema_version': 1, 'loci': [
        {'id': j, 'identity': f'fictional-{j}'} for j in range(len(scores[0]))]}).encode()
    alphabet = LocusAlphabet(replace(v.convention, sha256=hashlib.sha256(payload).hexdigest()), payload)
    n = len(scores)
    c = LocusSequenceCandidate(reference_candidate([[0.] * n for _ in range(n)]),
                               torch.tensor(scores, dtype=torch.float64), alphabet.convention.sha256,
                               alphabet.identities)
    return c, alphabet


def decode(c, a, refs=None, place=None, budget=1000000):
    n = len(c.locus_logits)
    return decode_locus_assignment(c, a, referents=tuple(range(n)) if refs is None else refs,
                                   place=(True,) * n if place is None else place, max_work=budget)


def test_shared_referent_aggregation_and_collision_avoidance():
    c, a = fixture([[9, 8], [0, 7], [8, 0]])
    r = decode(c, a, refs=(91, 91, 4))
    assert r.status == 'unique_optimum_candidate' and r.loci == (1, 1, 0)
    assert r.best_gain == 23 and r.runner_up_gain == 9
    assert decode(c, a, refs=(0, 0, 100)).loci == r.loci


def test_rectangular_assignment_matches_independent_permutation_oracle():
    for rows in range(1, 5):
        for columns in range(rows, 6):
            torch.manual_seed(rows * 9 + columns)
            values = torch.randint(-4, 5, (rows, columns)).tolist()
            c, a = fixture(values)
            r = decode(c, a)
            expected = sorted((sum(values[i][j] for i, j in enumerate(p)), p)
                              for p in permutations(range(columns), rows))
            assert r.best_gain == expected[-1][0]
            assert r.runner_up_gain == (expected[-2][0] if len(expected) > 1 else None)
            if len(expected) > 1 and expected[-1][0] == expected[-2][0]:
                assert r.status == 'ambiguous' and r.loci is None
            else:
                assert r.loci == expected[-1][1]


def test_tiny_gap_huge_gain_and_tie_are_exact():
    tiny = float.fromhex('0x0.0000000000001p-1022')
    c, a = fixture([[tiny, 0.]])
    assert decode(c, a).objective_gap == Fraction.from_float(tiny)
    c, a = fixture([[1e308, -1e308], [1e308, -1e308]])
    r = decode(c, a, refs=(0, 0))
    assert r.best_gain == 2 * Fraction.from_float(1e308) and r.loci == (0, 0)
    c, a = fixture([[0, 0], [0, 0]])
    assert decode(c, a).status == 'ambiguous'


def test_budget_exhaustion_including_runner_up_never_returns_assignment():
    c, a = fixture([[8, 3, -2], [2, 9, 0]])
    full = decode(c, a)
    for budget in range(1, full.work):
        r = decode(c, a, budget=budget)
        assert r.status == 'search_exhausted' and r.loci is None and r.best_gain is None
        assert r.work == budget
    assert decode(c, a, budget=full.work) == full


def test_explicit_placement_mask_capacity_and_input_bindings():
    c, a = fixture([[8, 3], [1, 9], [10, 2]])
    assert decode(c, a).status == 'infeasible_locus_capacity'
    assert decode(c, a, refs=(0, None, 1), place=(True, False, True)).loci == (1, None, 0)
    r = decode(c, a, refs=(None,) * 3, place=(False,) * 3)
    assert r.status == 'no_placements_requested' and r.loci == (None,) * 3
    for kwargs in ({'refs': (0, None, 1)}, {'place': (1, True, True)}, {'budget': True}):
        with pytest.raises(ValueError):
            decode(c, a, **kwargs)
    with pytest.raises(ValueError):
        decode(replace(c, convention_sha256='f'*64), a)
    c.locus_logits[0, 0] = float('nan')
    with pytest.raises(ValueError):
        decode(c, a)
