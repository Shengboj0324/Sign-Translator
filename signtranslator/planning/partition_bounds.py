"""Exact optimistic bounds for weighted equivalence-partition objectives."""
from dataclasses import dataclass


def prefix_attachment_upper_bound(prefix, count, pair, place, capacity):
    """Relax undecided prefix-to-suffix reference gains independently.

    If placed prefix groups already occupy every locus, any later placed event
    must join one of those groups: fixed groups cannot merge, and activating a
    new placed group would violate injectivity. Its best prefix gain can then
    be negative. Otherwise a fresh group with zero prefix gain remains allowed.
    Future pair relations and mutual attachment constraints are relaxed here.
    Inputs are validated integer gains and a capacity-feasible canonical prefix.
    """
    groups = max(prefix) + 1
    placed = {label for event, label in enumerate(prefix) if place[event]}
    full = len(placed) == capacity
    bound = 0
    for future in range(len(prefix), count):
        gains = [0] * groups
        for past, label in enumerate(prefix):
            gains[label] += pair[past, future]
        bound += (max(gains[label] for label in placed) if full and place[future]
                  else max(0, max(gains)))
    return bound


def partition_suffix_upper_bounds(count, pair):
    """Bound each suffix using positive gains minus disjoint triangle conflicts.

    Inputs are complete upper-triangle integer gains. For a triple with two
    positive edges and one negative edge, equivalence transitivity forces at
    least one preference violation. Its loss versus the all-positive-edge bound
    is at least the minimum absolute edge gain. Edge-disjoint triples charge
    disjoint objective terms, so these deductions may safely be summed.

    Greedy packing is a bound, not an optimal triangle packing. Processing new
    minimum vertices backwards retains a valid edge-disjoint packing in every
    suffix. Precomputation is O(count**3), bounded by the decoders' event cap.
    """
    bounds = [0] * (count + 1)
    # With fewer than two positive edges or no negative edges, no deduction
    # is possible. Preserve the quadratic preprocessing cost for these controls.
    if sum(value > 0 for value in pair.values()) < 2 or not any(value < 0 for value in pair.values()):
        for i in range(count - 1, -1, -1):
            bounds[i] = bounds[i + 1] + sum(max(0, pair[i, j]) for j in range(i + 1, count))
        return bounds
    used = set()
    for i in range(count - 1, -1, -1):
        bound = bounds[i + 1] + sum(max(0, pair[i, j]) for j in range(i + 1, count))
        conflicts = []
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                if (j, k) in used:
                    continue
                a, b, c = pair[i, j], pair[i, k], pair[j, k]
                if (a > 0) + (b > 0) + (c > 0) == 2 and (a < 0 or b < 0 or c < 0):
                    conflicts.append((min(abs(a), abs(b), abs(c)), ((i, j), (i, k), (j, k))))
        for deduction, edges in sorted(conflicts, reverse=True):
            if not any(edge in used for edge in edges):
                used.update(edges)
                bound -= deduction
        bounds[i] = bound
    return bounds


def minimum_added_pairs(sizes, remaining, capacity):
    """Minimum new within-group pairs after adding events to fixed groups.

    Callers supply nonnegative integer sizes, remaining >= 0 and capacity >=
    len(sizes). Empty groups are available up to capacity. Adding an event to a
    group of size k costs k new pairs. These marginal costs are increasing, so
    repeatedly taking the smallest available cost minimizes their sum. This
    relaxes all identity/score constraints and cannot overstate forced pairs.
    """
    from heapq import heapify, heapreplace

    if remaining == 0:
        return 0
    heap = list(sizes) + [0] * (capacity - len(sizes))
    heapify(heap)
    pairs = 0
    for _ in range(remaining):
        smallest = heap[0]
        pairs += smallest
        heapreplace(heap, smallest + 1)
    return pairs


@dataclass(frozen=True)
class PlacedPairPenalties:
    """Immutable suffix versions of a counted negative-cost order-statistic tree."""

    free: tuple[int, ...]
    roots: tuple[int, ...]
    values: tuple[int, ...]
    # Node fields are left index, right index, multiplicity, exact weighted sum.
    nodes: tuple[tuple[int, int, int, int], ...]

    def penalty(self, prefix, added_pairs):
        if (type(prefix) is not int or not 0 <= prefix < len(self.roots)
                or type(added_pairs) is not int or added_pairs < 0):
            raise ValueError('valid prefix and nonnegative integer forced-pair count required')
        needed = max(0, added_pairs - self.free[prefix])
        root = self.roots[prefix]
        if needed > self.nodes[root][2]:
            raise ValueError('forced-pair count exceeds remaining placed pairs')
        if needed == 0:
            return 0
        lower, upper, total = 0, len(self.values), 0
        while upper - lower > 1:
            left, right, _, _ = self.nodes[root]
            middle = (lower + upper) // 2
            left_count = self.nodes[left][2]
            if needed <= left_count:
                root, upper = left, middle
            else:
                total += self.nodes[left][3]
                needed -= left_count
                root, lower = right, middle
        return total + needed * self.values[lower]


def placed_pair_penalty_tables(count, pair, place):
    """Relax remaining placed-pair costs, including mixed signs and zero gains.

    At prefix length k only pairs with second endpoint >= k remain undecided.
    Nonnegative pairs can satisfy forced equalities without a negative cost.
    If m additional equalities are forced, at least max(0, m - free[k]) must
    use distinct negative pairs. The sum of that many cheapest remaining costs
    is a lower bound, even when those pairs cannot jointly form a partition.

    Inputs have the decoder's validated integer gain/placement domains. Persistent
    counted trees share unchanged cost ranges across suffixes. With E placed
    pairs and D distinct negative costs, construction uses O(count**2 + E log(D+1))
    time and O(count + E log(D+1)) space. Queries use O(1 + log(D+1)) time.
    """
    from collections import Counter

    free = [0] * (count + 1)
    roots = [0] * (count + 1)
    values = tuple(sorted({-value for (i, j), value in pair.items()
                           if place[i] and place[j] and value < 0}))
    ranks = {value: rank for rank, value in enumerate(values)}
    nodes = [(0, 0, 0, 0)]

    def insert(root, lower, upper, rank, multiplicity):
        left, right, old_count, old_sum = nodes[root]
        if upper - lower > 1:
            middle = (lower + upper) // 2
            if rank < middle:
                left = insert(left, lower, middle, rank, multiplicity)
            else:
                right = insert(right, middle, upper, rank, multiplicity)
        nodes.append((left, right, old_count + multiplicity,
                      old_sum + multiplicity * values[rank]))
        return len(nodes) - 1

    for j in range(count - 1, -1, -1):
        free[j] = free[j + 1]
        root = roots[j + 1]
        negative = Counter()
        if place[j]:
            for i in range(j):
                if place[i]:
                    value = pair[i, j]
                    if value < 0:
                        negative[-value] += 1
                    else:
                        free[j] += 1
        for value, multiplicity in negative.items():
            root = insert(root, 0, len(values), ranks[value], multiplicity)
        roots[j] = root
    return PlacedPairPenalties(tuple(free), tuple(roots), values, tuple(nodes))
