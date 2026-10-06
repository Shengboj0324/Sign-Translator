"""Independent pair-allocation and joint assignment oracles."""
from itertools import product
from fractions import Fraction
import random

from signtranslator.planning.partition_bounds import minimum_added_pairs
from test_joint_spatial import candidate, decode


def test_minimum_added_pairs_matches_exhaustive_allocations():
    rng = random.Random(2031)
    for capacity in range(1, 5):
        for _ in range(40):
            sizes = [rng.randrange(5) for _ in range(rng.randrange(capacity + 1))]
            remaining = rng.randrange(6)
            initial = sizes + [0] * (capacity - len(sizes))
            expected = None
            for allocation in product(range(capacity), repeat=remaining):
                final = initial.copy()
                for group in allocation:
                    final[group] += 1
                gain = sum(b*(b-1)//2-a*(a-1)//2 for a,b in zip(initial,final))
                expected = gain if expected is None else min(expected,gain)
            assert minimum_added_pairs(sizes, remaining, capacity) == expected


def test_negative_placed_pairs_preserve_exact_joint_optimum_and_runner_up():
    for n in range(3, 7):
        refs = [[0 if i == j else -(i+j+1) for j in range(n)] for i in range(n)]
        loci = [[i-3, 2-i] for i in range(n)]
        c, alphabet = candidate(loci, refs)
        scores = []
        for assignment in product(range(2), repeat=n):
            ref = sum(refs[i][j] for i in range(n) for j in range(i+1,n)
                      if assignment[i] == assignment[j])
            spatial = sum(loci[i][assignment[i]] for i in range(n))
            scores.append(Fraction(2,3)*ref + Fraction(3,2)*spatial)
        scores.sort(reverse=True)
        result = decode(c, alphabet, rw=Fraction(2,3), lw=Fraction(3,2))
        assert (result.best_gain, result.runner_up_gain) == tuple(scores[:2])
        assert (result.status == 'ambiguous') == (scores[0] == scores[1])


def test_capacity_bound_relaxes_unplaced_event_interactions_without_double_counting():
    from itertools import permutations
    refs = [[0,-2,-3,7],[-2,0,-4,-6],[-3,-4,0,2],[7,-6,2,0]]
    loci = [[-1,3],[2,-5],[4,1],[99,-99]]
    place = (True,True,True,False)
    c, alphabet = candidate(loci, refs)
    states = {}
    for labels in product(range(4), repeat=4):
        mapping = {}
        canonical = tuple(mapping.setdefault(x,len(mapping)) for x in labels)
        groups = tuple(dict.fromkeys(canonical[i] for i in range(3)))
        for columns in permutations(range(2),len(groups)):
            assigned = dict(zip(groups,columns))
            vector = tuple(assigned[canonical[i]] if place[i] else None for i in range(4))
            ref = sum(refs[i][j] for i in range(4) for j in range(i+1,4) if canonical[i]==canonical[j])
            spatial = sum(loci[i][vector[i]] for i in range(3))
            states[(canonical,vector)] = Fraction(2,3)*ref+Fraction(3,2)*spatial
    expected = sorted(states.values(),reverse=True)
    result = decode(c,alphabet,place=place,rw=Fraction(2,3),lw=Fraction(3,2))
    assert (result.best_gain,result.runner_up_gain)==tuple(expected[:2])
