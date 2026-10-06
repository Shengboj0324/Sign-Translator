"""Exact optimistic bounds for weighted equivalence-partition objectives."""


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
