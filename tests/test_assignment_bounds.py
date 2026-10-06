"""Independent exhaustive assignment oracle, including negative gains."""
from itertools import permutations
import random

from signtranslator.planning.assignment_bounds import injective_assignment_upper_bound


def test_collision_bound_is_stricter_and_preserves_negative_gains():
    assert injective_assignment_upper_bound([]) == 0
    assert injective_assignment_upper_bound([[10, 0], [10, 0]]) == 10
    assert injective_assignment_upper_bound([[-3, -5], [-3, -5]]) == -8
    huge = 1 << 2000
    assert injective_assignment_upper_bound([[huge, -huge], [huge, -huge]]) == 0


def test_assignment_bounds_dominate_exhaustive_optimum():
    rng = random.Random(4619)
    for columns in range(1, 6):
        for count in range(1, min(columns, 4) + 1):
            for _ in range(30):
                rows = [[rng.randint(-20, 20) for _ in range(columns)] for _ in range(count)]
                optimum = max(sum(rows[i][column] for i, column in enumerate(assignment))
                              for assignment in permutations(range(columns), count))
                bound = injective_assignment_upper_bound(rows)
                assert optimum <= bound <= sum(max(row) for row in rows)
