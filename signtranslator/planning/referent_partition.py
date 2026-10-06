"""Exact bounded decoding of annotation-local referent partitions.

For pair logit z and equality decision y, Bernoulli log likelihood is
    y*z - softplus(z).
The second term is constant across partitions, so maximize sum(z for equal
unordered pairs). This is a pairwise composite objective, not a calibrated
probability of a partition. Transitivity is imposed by searching partitions.
Binary floating-point inputs are compared as exact rational numbers; cancellation,
overflow or underflow cannot create or conceal an objective tie.
"""
from dataclasses import dataclass
from fractions import Fraction

import torch

from .text_referents import ReferentSequenceCandidate


@dataclass(frozen=True)
class ReferentPartitionResult:
    status: str
    referents: tuple[int, ...] | None
    best_gain: Fraction | None
    runner_up_gain: Fraction | None
    visited_nodes: int
    evaluated_partitions: int

    @property
    def objective_gap(self) -> Fraction | None:
        if self.best_gain is None or self.runner_up_gain is None:
            return None
        return self.best_gain - self.runner_up_gain


def decode_referent_partition(candidate: ReferentSequenceCandidate, *,
                               max_events: int, max_search_nodes: int) -> ReferentPartitionResult:
    """Exact branch-and-bound over restricted-growth partitions within resource caps.

    IDs are local first-occurrence cluster numbers, not external entity IDs.
    A unique optimum returns an uncalibrated candidate. A tie, failed label
    decoding, capacity excess or incomplete search returns no usable assignment.
    The implementation supports at most 128 events; never truncates or switches
    to a greedy approximation. Locus placement and absent-reference decisions
    are outside this decoder's hypothesis space.
    """
    if type(max_events) is not int or not 1 <= max_events <= 128:
        raise ValueError('explicit max_events between 1 and 128 required')
    if type(max_search_nodes) is not int or not 1 <= max_search_nodes <= 1_000_000:
        raise ValueError('explicit search budget between 1 and 1000000 required')
    if not isinstance(candidate, ReferentSequenceCandidate):
        raise ValueError('typed referent candidate required')
    temporal = candidate.relational.temporal
    def refused(status, nodes=0, leaves=0):
        return ReferentPartitionResult(status, None, None, None, nodes, leaves)
    if temporal.status != 'terminated':
        return refused('label_decoding_not_terminated')
    count = len(temporal.events)
    if count == 0:
        return refused('empty_candidate')
    if count > max_events:
        return refused('capacity_exceeded')
    scores, valid = candidate.equality_logits, candidate.pair_valid
    if (not isinstance(scores, torch.Tensor) or scores.dtype not in (torch.float32, torch.float64)
            or scores.shape != (count, count) or not bool(torch.isfinite(scores).all())
            or not torch.equal(scores, scores.T)):
        raise ValueError('finite symmetric equality scores of exact event shape required')
    domain = ~torch.eye(count, dtype=torch.bool, device=scores.device)
    if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.shape != domain.shape or not torch.equal(valid.to(scores.device), domain)):
        raise ValueError('reference domain must exclude exactly self pairs')
    # Snapshot once; each Python float exactly represents the float32/64 input.
    values = scores.detach().to(device='cpu', copy=True).tolist()
    rational = {(i, j): float(values[i][j]).as_integer_ratio()
                for i in range(count) for j in range(i + 1, count)}
    denominator = max((d for _, d in rational.values()), default=1)
    gains = {(i, j): numerator * (denominator // d)
             for (i, j), (numerator, d) in rational.items()}
    # Optimistic future/future contribution: omit every negative pair. This
    # ignores consistency constraints and therefore cannot underestimate a maximum.
    future_positive = [0] * (count + 1)
    for i in range(count - 1, -1, -1):
        future_positive[i] = future_positive[i + 1] + sum(
            max(0, gains[(i, j)]) for j in range(i + 1, count))

    def upper_bound(prefix, maximum, score):
        index = len(prefix)
        bound = score + future_positive[index]
        # A future event may join one fixed prefix cluster or a new cluster.
        # Maximize each such choice independently for a valid relaxed upper bound.
        for future in range(index, count):
            by_cluster = [0] * (maximum + 1)
            for past, label in enumerate(prefix):
                by_cluster[label] += gains[(past, future)]
            bound += max(0, max(by_cluster))
        return bound

    best = second = None
    best_partition = None
    nodes = leaves = 0
    exhausted = False

    def visit(prefix, maximum, score):
        nonlocal best, second, best_partition, nodes, leaves, exhausted
        if nodes == max_search_nodes:
            exhausted = True
            return
        nodes += 1
        index = len(prefix)
        if index == count:
            leaves += 1
            if best is None or score > best:
                second, best, best_partition = best, score, prefix
            elif second is None or score > second:
                second = score
            return
        # The runner-up, not only the winner, must be proved. Prune only when
        # this subtree cannot improve the already observed second-best score.
        # Equality is safe: a best-score tie is already represented if best==second.
        if second is not None and upper_bound(prefix, maximum, score) <= second:
            return
        choices = []
        for label in range(maximum + 2):
            added = sum(gains[(j, index)] for j, previous in enumerate(prefix) if previous == label)
            choices.append((added, label))
        for added, label in sorted(choices, key=lambda item: (-item[0], item[1])):
            visit(prefix + (label,), max(maximum, label), score + added)
            if exhausted:
                return

    visit((0,), 0, 0)
    if exhausted:
        return refused('search_exhausted', nodes, leaves)
    status = 'ambiguous' if second == best else 'unique_optimum_candidate'
    return ReferentPartitionResult(status, best_partition if status == 'unique_optimum_candidate' else None,
                                   Fraction(best, denominator),
                                   None if second is None else Fraction(second, denominator), nodes, leaves)
