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


def test_weighted_capacity_penalty_is_stronger_and_includes_zero_as_free():
    from signtranslator.planning.partition_bounds import placed_pair_penalty_tables

    pair = {(0, 1): -1, (0, 2): -5, (0, 3): -9,
            (1, 2): 0, (1, 3): 3, (2, 3): -7}
    table = placed_pair_penalty_tables(4, pair, (True,) * 4)
    assert table.free[0] == 2
    # Five forced equalities require at least three distinct negative pairs.
    assert table.penalty(0, 5) == 1 + 5 + 7
    assert table.free[2] == 2 and [table.penalty(2, k + 2) for k in range(4)] == [0, 5, 12, 21]
    assert table.free[4] == 0 and table.penalty(4, 0) == 0


def test_mixed_capacity_bound_dominates_every_prefix_completion():
    from signtranslator.planning.partition_bounds import minimum_added_pairs, placed_pair_penalty_tables

    rng = random.Random(60106)
    for n in range(2, 7):
        for trial in range(12):
            pair = {(i, j): rng.randint(-11, 5) for i in range(n) for j in range(i + 1, n)}
            place = tuple(bool(rng.randrange(2)) for _ in range(n))
            table = placed_pair_penalty_tables(n, pair, place)
            capacity = rng.randint(1, 3)
            for labels in partitions(n):
                if len({labels[i] for i in range(n) if place[i]}) > capacity:
                    continue
                actual = sum(value for (i, j), value in pair.items() if labels[i] == labels[j])
                for k in range(n + 1):
                    sizes = {}
                    for i in range(k):
                        if place[i]:
                            sizes[labels[i]] = sizes.get(labels[i], 0) + 1
                    forced = minimum_added_pairs(tuple(sizes.values()), sum(place[k:]), capacity)
                    exact_prefix = sum(value for (i, j), value in pair.items()
                                       if j < k and labels[i] == labels[j])
                    remaining_positive = sum(max(0, value) for (i, j), value in pair.items() if j >= k)
                    upper = exact_prefix + remaining_positive - table.penalty(k, forced)
                    assert actual <= upper


def test_persistent_penalties_match_sorted_remaining_costs_for_every_query():
    from signtranslator.planning.partition_bounds import placed_pair_penalty_tables

    rng = random.Random(61007)
    for n in range(11):
        for trial in range(10):
            pair = {(i, j): rng.choice([-2**2001, -19, -7, -7, -1, 0, 4])
                    for i in range(n) for j in range(i + 1, n)}
            place = tuple(bool(rng.randrange(2)) for _ in range(n))
            table = placed_pair_penalty_tables(n, pair, place)
            for prefix in range(n + 1):
                remaining = [value for (i, j), value in pair.items()
                             if j >= prefix and place[i] and place[j]]
                free = sum(value >= 0 for value in remaining)
                costs = sorted(-value for value in remaining if value < 0)
                assert table.free[prefix] == free
                for forced in range(len(remaining) + 1):
                    assert table.penalty(prefix, forced) == sum(costs[:max(0, forced - free)])
            # Neither later source edits nor queries may alter saved versions.
            before = table.penalty(0, sum(place) * (sum(place) - 1) // 2)
            pair.clear()
            assert table.penalty(0, sum(place) * (sum(place) - 1) // 2) == before


def test_penalty_queries_refuse_impossible_or_coerced_counts():
    import pytest
    from signtranslator.planning.partition_bounds import placed_pair_penalty_tables

    table = placed_pair_penalty_tables(2, {(0, 1): -3}, (True, True))
    for prefix, forced in [(-1, 0), (3, 0), (True, 0), (0, True), (0, -1), (0, 2), (2, 1)]:
        with pytest.raises(ValueError):
            table.penalty(prefix, forced)
