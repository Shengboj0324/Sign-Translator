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
    used = set()
    for i in range(count - 1, -1, -1):
        bound = bounds[i + 1] + sum(max(0, pair[i, j]) for j in range(i + 1, count))
        conflicts = []
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                if (j, k) in used:
                    continue
                edges = ((i, j), (i, k), (j, k))
                values = tuple(pair[edge] for edge in edges)
                if sum(value > 0 for value in values) == 2 and any(value < 0 for value in values):
                    conflicts.append((min(abs(value) for value in values), edges))
        for deduction, edges in sorted(conflicts, reverse=True):
            if not any(edge in used for edge in edges):
                used.update(edges)
                bound -= deduction
        bounds[i] = bound
    return bounds
