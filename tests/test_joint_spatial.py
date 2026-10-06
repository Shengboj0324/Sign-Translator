from dataclasses import replace
from fractions import Fraction
from itertools import product, permutations

import pytest
import torch

from signtranslator.planning.joint_spatial import decode_joint_spatial
from signtranslator.planning.referent_partition import decode_referent_partition
from test_locus_assignment import fixture


def candidate(loci, references):
    c, a = fixture(loci)
    return replace(c, referential=replace(c.referential,
                   equality_logits=torch.tensor(references, dtype=torch.float64))), a


def decode(c, a, *, place=None, budget=1000000, rw=Fraction(1), lw=Fraction(1)):
    return decode_joint_spatial(c, a, place=(True,) * len(c.locus_logits) if place is None else place,
                                referent_weight=rw, locus_weight=lw, max_events=10, max_work=budget)


def test_locus_evidence_can_change_the_optimal_referent_partition():
    c, a = candidate([[10, 0], [0, 10]], [[0, 3], [3, 0]])
    assert decode_referent_partition(c.referential, max_events=10, max_search_nodes=100).referents == (0, 0)
    r = decode(c, a)
    assert r.referents == (0, 1) and r.loci == (0, 1) and r.best_gain == 20
    assert r.runner_up_gain == 13
    assert decode(c, a, rw=Fraction(4)).status == 'ambiguous'  # either shared locus has gain 22
    c, a = candidate([[5], [4]], [[0, -3], [-3, 0]])
    assert decode(c, a).referents == (0, 0)  # separate placed referents cannot fit


def test_joint_search_matches_independent_assignment_oracle_with_partial_placement():
    for n in range(1, 5):
        torch.manual_seed(n)
        refs = torch.randint(-4, 5, (n, n)); refs = (refs + refs.T).tolist()
        loci = torch.randint(-4, 5, (n, 3)).tolist()
        c, a = candidate(loci, refs)
        for place in [(True,) * n, tuple(i % 2 == 0 for i in range(n)), (False,) * n]:
            states = {}
            for labels in product(range(n), repeat=n):
                mapping = {}
                canonical = tuple(mapping.setdefault(x, len(mapping)) for x in labels)
                groups = tuple(dict.fromkeys(canonical[i] for i in range(n) if place[i]))
                if len(groups) > 3:
                    continue
                for selected in permutations(range(3), len(groups)):
                    assignment = dict(zip(groups, selected))
                    output = tuple(assignment[canonical[i]] if place[i] else None for i in range(n))
                    ref_gain = sum(refs[i][j] for i in range(n) for j in range(i+1, n) if canonical[i] == canonical[j])
                    locus_gain = sum(loci[i][output[i]] for i in range(n) if place[i])
                    states[(canonical, output)] = Fraction(2, 3) * ref_gain + Fraction(3, 2) * locus_gain
            expected = sorted(states.values(), reverse=True)
            result = decode(c, a, place=place, rw=Fraction(2, 3), lw=Fraction(3, 2))
            assert result.best_gain == expected[0]
            assert result.runner_up_gain == (expected[1] if len(expected) > 1 else None)
            if len(expected) > 1 and expected[1] == expected[0]:
                assert result.status == 'ambiguous' and result.referents is None and result.loci is None
            else:
                assert states[(result.referents, result.loci)] == expected[0]


def test_global_budget_exhaustion_never_returns_early_winner():
    c, a = candidate([[10, 0], [0, 10]], [[0, 3], [3, 0]])
    full = decode(c, a)
    for budget in range(1, full.work):
        result = decode(c, a, budget=budget)
        assert result.status == 'search_exhausted'
        assert result.best_gain is None and result.referents is None and result.loci is None
        assert result.work <= budget
    assert decode(c, a, budget=full.work) == full


def test_contracts_weights_nonfinite_and_alphabet_are_checked():
    c, a = candidate([[2, 0]], [[0]])
    for kwargs in ({'rw': 1.}, {'lw': Fraction(0)}, {'place': (1,)}, {'budget': True}):
        with pytest.raises(ValueError):
            decode(c, a, **kwargs)
    with pytest.raises(ValueError):
        decode(replace(c, convention_sha256='f'*64), a)
    c.locus_logits[0, 0] = float('nan')
    with pytest.raises(ValueError):
        decode(c, a)


def test_large_joint_search_with_capacity_and_relaxed_bound_proofs():
    n = 128
    c, a = candidate([[2.] for _ in range(n)],
                     [[0. if i == j else 1. for j in range(n)] for i in range(n)])
    r = decode_joint_spatial(c, a, place=(True,) * n, referent_weight=Fraction(1),
                             locus_weight=Fraction(1), max_events=128, max_work=100000)
    assert r.status == 'unique_optimum_candidate'
    assert r.referents == (0,) * n and r.loci == (0,) * n
    assert r.best_gain == 8128 + 256 and r.runner_up_gain is None
    assert r.evaluated_partitions == 1 and r.work < 10000
    # Without placement, the joint solver must prove the reference runner-up.
    r = decode_joint_spatial(c, a, place=(False,) * n, referent_weight=Fraction(2, 3),
                             locus_weight=Fraction(1), max_events=128, max_work=100000)
    assert r.referents == (0,) * n and r.loci == (None,) * n
    assert r.best_gain == Fraction(2, 3) * 8128 and r.objective_gap == Fraction(2, 3) * 127
    assert r.evaluated_partitions < 10 and r.work < 10000


def test_mixed_sign_bounds_match_more_exhaustive_joint_oracles():
    n = 4
    for seed in range(10):
        torch.manual_seed(100 + seed)
        raw = torch.randint(-9, 10, (n, n))
        refs = (raw + raw.T).tolist()
        loci = torch.randint(-9, 10, (n, 2)).tolist()
        c, a = candidate(loci, refs)
        place = (True, False, True, True)
        states = {}
        for labels in product(range(n), repeat=n):
            mapping = {}
            canonical = tuple(mapping.setdefault(x, len(mapping)) for x in labels)
            groups = tuple(dict.fromkeys(canonical[i] for i in range(n) if place[i]))
            for selected in permutations(range(2), len(groups)):
                assignment = dict(zip(groups, selected))
                output = tuple(assignment[canonical[i]] if place[i] else None for i in range(n))
                ref_gain = sum(refs[i][j] for i in range(n) for j in range(i+1, n) if canonical[i] == canonical[j])
                locus_gain = sum(loci[i][output[i]] for i in range(n) if place[i])
                states[(canonical, output)] = Fraction(3, 7) * ref_gain + Fraction(5, 2) * locus_gain
        expected = sorted(states.values(), reverse=True)
        r = decode(c, a, place=place, rw=Fraction(3, 7), lw=Fraction(5, 2))
        assert r.best_gain == expected[0] and r.runner_up_gain == expected[1]
        assert (r.referents is None) == (expected[0] == expected[1])


@pytest.mark.parametrize('place', [(True, True, True), (True, False, True), (False, False, False)])
def test_integer_bound_scale_preserves_extreme_binary_scores_and_rational_weights(place):
    tiny = float.fromhex('0x0.0000000000001p-1022')
    refs = [[0., 1e308, -1e308], [1e308, 0., tiny], [-1e308, tiny, 0.]]
    loci = [[tiny, 0.], [1e308, -1e308], [0., tiny]]
    c, a = candidate(loci, refs)
    rw, lw = Fraction(2, 7), Fraction(3, 11)
    states = {}
    for labels in product(range(3), repeat=3):
        mapping = {}
        canonical = tuple(mapping.setdefault(label, len(mapping)) for label in labels)
        groups = tuple(dict.fromkeys(canonical[i] for i in range(3) if place[i]))
        for selected in permutations(range(2), len(groups)):
            assignment = dict(zip(groups, selected))
            placed = tuple(assignment[canonical[i]] if place[i] else None for i in range(3))
            reference_gain = sum((Fraction(refs[i][j]) for i in range(3) for j in range(i+1, 3)
                                  if canonical[i] == canonical[j]), Fraction(0))
            locus_gain = sum((Fraction(loci[i][placed[i]]) for i in range(3) if place[i]), Fraction(0))
            states[(canonical, placed)] = rw * reference_gain + lw * locus_gain
    expected = sorted(states.values(), reverse=True)
    result = decode(c, a, place=place, rw=rw, lw=lw)
    assert result.best_gain == expected[0] and result.runner_up_gain == expected[1]
    if expected[0] == expected[1]:
        assert result.status == 'ambiguous' and result.referents is None
    else:
        assert states[(result.referents, result.loci)] == expected[0]
