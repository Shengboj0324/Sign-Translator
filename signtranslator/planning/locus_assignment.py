"""Exact injective locus assignment conditional on explicit placement decisions.

For fixed events/placement mask, categorical log normalizers are independent of
assignment. Summing selected-event logits for each referent/locus therefore gives
an exact maximum-weight assignment objective. A shared referent uses one locus;
distinct placed referents use different loci. Unknown/unplaced events stay None.
Neither the placement mask nor referent partition is inferred by this stage.
"""
from dataclasses import dataclass
from fractions import Fraction

import torch

from .loci import LocusAlphabet
from .text_loci import LocusSequenceCandidate


@dataclass(frozen=True)
class LocusAssignmentResult:
    status: str
    loci: tuple[int | None, ...] | None
    best_gain: Fraction | None
    runner_up_gain: Fraction | None
    work: int

    @property
    def objective_gap(self):
        if self.best_gain is None or self.runner_up_gain is None:
            return None
        return self.best_gain - self.runner_up_gain


class _BudgetExhausted(Exception):
    pass


def decode_locus_assignment(candidate: LocusSequenceCandidate, alphabet: LocusAlphabet, *,
                            referents: tuple[int | None, ...], place: tuple[bool, ...],
                            max_work: int) -> LocusAssignmentResult:
    """Exact rectangular assignment and runner-up using integer Hungarian solves.

    Work counts examined row/column pairs across all solves. Runner-up search
    excludes each edge of one optimum in turn; every different assignment must
    omit at least one such edge. No partial optimum is returned on exhaustion.
    Gain/gap are uncalibrated exact rational scores, not confidence estimates.
    """
    if type(max_work) is not int or not 1 <= max_work <= 10_000_000:
        raise ValueError('explicit max_work between 1 and 10000000 required')
    if not isinstance(candidate, LocusSequenceCandidate) or not isinstance(alphabet, LocusAlphabet):
        raise ValueError('typed locus candidate and alphabet required')
    if (candidate.convention_sha256 != alphabet.convention.sha256
            or candidate.locus_identities != alphabet.identities):
        raise ValueError('locus candidate alphabet identity mismatch')
    temporal = candidate.referential.relational.temporal
    work = 0
    def refused(status):
        return LocusAssignmentResult(status, None, None, None, work)
    if temporal.status != 'terminated':
        return refused('label_decoding_not_terminated')
    n, columns = len(temporal.events), len(alphabet.identities)
    if n == 0:
        return refused('empty_candidate')
    if (not isinstance(referents, tuple) or len(referents) != n
            or any(r is not None and (type(r) is not int or r < 0) for r in referents)
            or not isinstance(place, tuple) or len(place) != n or any(type(x) is not bool for x in place)):
        raise ValueError('explicit per-event references and boolean placement decisions required')
    if any(place[i] and referents[i] is None for i in range(n)):
        raise ValueError('placed events require explicit referent identities')
    scores = candidate.locus_logits
    if (not isinstance(scores, torch.Tensor) or scores.dtype not in (torch.float32, torch.float64)
            or scores.shape != (n, columns) or not bool(torch.isfinite(scores).all())):
        raise ValueError('finite E,L locus scores matching candidate and alphabet required')
    groups = tuple(dict.fromkeys(referents[i] for i in range(n) if place[i]))
    rows = len(groups)
    if rows > columns:
        return refused('infeasible_locus_capacity')
    if rows == 0:
        return LocusAssignmentResult('no_placements_requested', (None,) * n, Fraction(0), None, 0)
    index = {ref: i for i, ref in enumerate(groups)}
    values = scores.detach().to(device='cpu', copy=True).tolist()
    rational = {(i, j): float(values[i][j]).as_integer_ratio()
                for i in range(n) if place[i] for j in range(columns)}
    denominator = max(d for _, d in rational.values())
    gains = [[0] * columns for _ in groups]
    for (i, j), (num, den) in rational.items():
        gains[index[referents[i]]][j] += num * (denominator // den)

    def solve(forbidden=None):
        nonlocal work
        # One-based rectangular Hungarian primal/dual augmentations.
        u, v = [0] * (rows + 1), [0] * (columns + 1)
        p, way = [0] * (columns + 1), [0] * (columns + 1)
        for row in range(1, rows + 1):
            p[0], j0 = row, 0
            minimum, used = [None] * (columns + 1), [False] * (columns + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = j1 = None
                for j in range(1, columns + 1):
                    if used[j]:
                        continue
                    if work == max_work:
                        raise _BudgetExhausted
                    work += 1
                    if forbidden != (i0 - 1, j - 1):
                        reduced = -gains[i0 - 1][j - 1] - u[i0] - v[j]
                        if minimum[j] is None or reduced < minimum[j]:
                            minimum[j], way[j] = reduced, j0
                    if minimum[j] is not None and (delta is None or minimum[j] < delta):
                        delta, j1 = minimum[j], j
                if delta is None:
                    return None
                for j in range(columns + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    elif minimum[j] is not None:
                        minimum[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while j0:
                previous = way[j0]
                p[j0] = p[previous]
                j0 = previous
        assignment = [None] * rows
        for j in range(1, columns + 1):
            if p[j]:
                assignment[p[j] - 1] = j - 1
        return tuple(assignment)

    try:
        best = solve()
        if best is None:
            return refused('infeasible_assignment')
        best_gain = sum(gains[i][j] for i, j in enumerate(best))
        second_gain = None
        for i, j in enumerate(best):
            alternative = solve((i, j))
            if alternative is not None:
                value = sum(gains[r][c] for r, c in enumerate(alternative))
                second_gain = value if second_gain is None else max(second_gain, value)
    except _BudgetExhausted:
        return refused('search_exhausted')
    status = 'ambiguous' if second_gain == best_gain else 'unique_optimum_candidate'
    loci = tuple(best[index[referents[i]]] if place[i] else None for i in range(n))
    return LocusAssignmentResult(status, loci if status == 'unique_optimum_candidate' else None,
                                 Fraction(best_gain, denominator),
                                 None if second_gain is None else Fraction(second_gain, denominator), work)
