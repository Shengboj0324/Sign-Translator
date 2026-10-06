import random

from signtranslator.planning.partition_bounds import partition_suffix_upper_bounds


def partitions(n):
    if n == 0:
        yield ()
        return
    def extend(prefix):
        if len(prefix) == n:
            yield prefix
            return
        for label in range(max(prefix) + 2):
            yield from extend(prefix + (label,))
    yield from extend((0,))


def test_triangle_conflict_deduction_is_exact_for_each_negative_edge_position():
    edges = [(0, 1), (0, 2), (1, 2)]
    for negative in edges:
        scores = {edge: -7 if edge == negative else 5 for edge in edges}
        assert partition_suffix_upper_bounds(3, scores)[0] == 5
    assert partition_suffix_upper_bounds(0, {}) == [0]
    assert partition_suffix_upper_bounds(1, {}) == [0, 0]


def test_each_suffix_bound_dominates_every_exhaustive_partition():
    rng = random.Random(1207)
    for n in range(2, 7):
        for seed in range(30):
            pair = {(i, j): rng.randint(-9, 9) for i in range(n) for j in range(i + 1, n)}
            bound = partition_suffix_upper_bounds(n, pair)
            for start in range(n):
                loose = sum(max(0, pair[i, j]) for i in range(start, n) for j in range(i + 1, n))
                best = max(sum(pair[i, j] for i in range(start, n) for j in range(i + 1, n)
                               if labels[i-start] == labels[j-start]) for labels in partitions(n-start))
                assert best <= bound[start] <= loose


def test_overlapping_conflicts_are_not_double_charged():
    pair = {(i, j): (10 if i == 0 else -10) for i in range(4) for j in range(i + 1, 4)}
    # Three apparent conflicts share positive star edges. Subtracting all three
    # penalties would produce zero, below the feasible one-pair score of ten.
    assert partition_suffix_upper_bounds(4, pair)[0] == 20
