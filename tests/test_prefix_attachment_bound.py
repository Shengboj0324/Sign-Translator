"""Exhaustive completion checks for forced placed-prefix attachment."""
from itertools import product
import random

from signtranslator.planning.partition_bounds import prefix_attachment_upper_bound


def partitions(count):
    if count == 0:
        yield ()
    else:
        for prefix in partitions(count - 1):
            for group in range(max(prefix, default=-1) + 2):
                yield prefix + (group,)


def test_full_capacity_forces_negative_attachment_but_unplaced_remains_free():
    pair = {(0, 1): -1, (0, 2): -8, (1, 2): -3}
    assert prefix_attachment_upper_bound((0, 1), 3, pair, (True, True, True), 2) == -3
    assert prefix_attachment_upper_bound((0, 1), 3, pair, (True, True, False), 2) == 0
    assert prefix_attachment_upper_bound((0, 1), 3, pair, (True, False, True), 2) == 0
    # An unplaced fixed group cannot be activated after capacity is full.
    pair = {(0, 1): 0, (0, 2): -4, (1, 2): 100}
    assert prefix_attachment_upper_bound((0, 1), 3, pair, (True, False, True), 1) == -4


def test_bound_dominates_all_small_feasible_completions():
    rng = random.Random(20261006)
    checked = 0
    for count in range(1, 7):
        completions = list(partitions(count))
        for repeat in range(8):
            capacity = rng.randint(1, 3)
            place = tuple(bool(rng.getrandbits(1)) for _ in range(count))
            pair = {(i, j): rng.randint(-20, 20) for i in range(count) for j in range(i + 1, count)}
            if repeat == 0:
                pair = {key: value * 2**2000 for key, value in pair.items()}
            feasible = [labels for labels in completions
                        if len({labels[i] for i in range(count) if place[i]}) <= capacity]
            for length in range(1, count + 1):
                for prefix in {labels[:length] for labels in feasible}:
                    bound = prefix_attachment_upper_bound(prefix, count, pair, place, capacity)
                    old = sum(max(0, max(sum(pair[past, future] for past, label in enumerate(prefix)
                                            if label == group) for group in set(prefix)))
                              for future in range(length, count))
                    assert bound <= old
                    actual = max(sum(pair[i, j] for i in range(length) for j in range(length, count)
                                     if labels[i] == labels[j])
                                 for labels in feasible if labels[:length] == prefix)
                    assert actual <= bound
                    checked += 1
    assert checked > 1000


def test_shared_child_bounds_exactly_equal_independent_prefix_bounds():
    from signtranslator.planning.partition_bounds import child_attachment_upper_bounds
    rng = random.Random(20261008)
    checked = 0
    for count in range(2, 8):
        for repeat in range(12):
            capacity = rng.randint(1, 4)
            place = tuple(bool(rng.getrandbits(1)) for _ in range(count))
            pair = {(i, j): rng.randint(-20, 20) * (2**2000 if repeat == 0 else 1)
                    for i in range(count) for j in range(i + 1, count)}
            for length in range(1, count):
                for prefix in partitions(length):
                    if len({prefix[i] for i in range(length) if place[i]}) > capacity:
                        continue
                    shared = child_attachment_upper_bounds(prefix, count, pair, place, capacity)
                    expected = []
                    for label in range(max(prefix) + 2):
                        child = prefix + (label,)
                        if len({child[i] for i in range(length + 1) if place[i]}) > capacity:
                            expected.append(None)
                        else:
                            expected.append(prefix_attachment_upper_bound(child, count, pair, place, capacity))
                    assert shared == tuple(expected)
                    checked += len(shared)
    assert checked > 10000


def test_child_maxima_exhaustive_ties_signs_and_placement():
    """Every small gain/mask combination agrees with independent recomputation.

    Tied maxima must retain two distinct groups; an unplaced group must never
    become available merely because its score exceeds the placed maximum.
    """
    from signtranslator.planning.partition_bounds import child_attachment_upper_bounds

    edges = tuple((i, j) for i in range(4) for j in range(i + 1, 4))
    for values in product((-1, 0, 1), repeat=len(edges)):
        pair = dict(zip(edges, values))
        for place in product((False, True), repeat=4):
            for prefix in ((0,), (0, 0), (0, 1)):
                for capacity in (1, 2, 3):
                    if len({label for i, label in enumerate(prefix) if place[i]}) > capacity:
                        continue
                    expected = []
                    for label in range(max(prefix) + 2):
                        child = prefix + (label,)
                        if len({group for i, group in enumerate(child) if place[i]}) > capacity:
                            expected.append(None)
                        else:
                            expected.append(prefix_attachment_upper_bound(child, 4, pair, place, capacity))
                    assert child_attachment_upper_bounds(prefix, 4, pair, place, capacity) == tuple(expected)
