"""Exact bounded joint referent partitions and conditional locus assignments.

For a fixed placement mask, optimize positive rational weights times (1) the
sum of equality logits within referent clusters and (2) the sum of assigned locus
logits. Omitted Bernoulli/categorical normalizers are constant across this search.
This composite objective is not a calibrated joint probability.
"""
from dataclasses import dataclass, replace
from fractions import Fraction

import torch

from .loci import LocusAlphabet
from .locus_assignment import decode_locus_assignment
from .text_loci import LocusSequenceCandidate


@dataclass(frozen=True)
class JointSpatialResult:
    status: str
    referents: tuple[int, ...] | None
    loci: tuple[int | None, ...] | None
    best_gain: Fraction | None
    runner_up_gain: Fraction | None
    work: int
    evaluated_partitions: int

    @property
    def objective_gap(self):
        if self.best_gain is None or self.runner_up_gain is None:
            return None
        return self.best_gain - self.runner_up_gain


def decode_joint_spatial(candidate: LocusSequenceCandidate, alphabet: LocusAlphabet, *,
                         place: tuple[bool, ...], referent_weight: Fraction, locus_weight: Fraction,
                         max_events: int, max_work: int) -> JointSpatialResult:
    """Exact branch-and-bound over partitions and their best two placements.

    One work unit is a partition prefix visit or a Hungarian row/column probe.
    Only a completed search with a unique best full assignment yields usable IDs.
    All-placed partitions exceeding locus capacity are skipped, never repaired.
    """
    if any(type(w) is not Fraction or w <= 0 for w in (referent_weight, locus_weight)):
        raise ValueError('explicit positive Fraction objective weights required')
    if type(max_events) is not int or not 1 <= max_events <= 128:
        raise ValueError('explicit max_events between 1 and 128 required')
    if type(max_work) is not int or not 1 <= max_work <= 10_000_000:
        raise ValueError('explicit max_work between 1 and 10000000 required')
    if not isinstance(candidate, LocusSequenceCandidate) or not isinstance(alphabet, LocusAlphabet):
        raise ValueError('typed locus candidate and alphabet required')
    temporal = candidate.referential.relational.temporal
    work = partitions = 0
    def refused(status):
        return JointSpatialResult(status, None, None, None, None, work, partitions)
    if temporal.status != 'terminated':
        return refused('label_decoding_not_terminated')
    n = len(temporal.events)
    if n == 0:
        return refused('empty_candidate')
    if n > max_events:
        return refused('capacity_exceeded')
    if not isinstance(place, tuple) or len(place) != n or any(type(x) is not bool for x in place):
        raise ValueError('explicit boolean placement mask matching events required')
    if (candidate.convention_sha256 != alphabet.convention.sha256
            or candidate.locus_identities != alphabet.identities):
        raise ValueError('candidate locus alphabet mismatch')
    scores, valid = candidate.referential.equality_logits, candidate.referential.pair_valid
    if (not isinstance(scores, torch.Tensor) or scores.dtype not in (torch.float32, torch.float64)
            or scores.shape != (n, n) or not bool(torch.isfinite(scores).all())
            or not torch.equal(scores, scores.T)):
        raise ValueError('finite symmetric event equality scores required')
    domain = ~torch.eye(n, dtype=torch.bool, device=scores.device)
    if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.shape != domain.shape or not torch.equal(valid.to(scores.device), domain)):
        raise ValueError('reference pair domain must exclude exactly self pairs')
    loci = candidate.locus_logits
    if (not isinstance(loci, torch.Tensor) or loci.dtype not in (torch.float32, torch.float64)
            or loci.shape != (n, len(alphabet.identities)) or not bool(torch.isfinite(loci).all())):
        raise ValueError('finite event locus scores matching alphabet required')
    # Snapshot both branches for one coherent search, avoiding repeated mutable reads.
    snapshot = replace(candidate, locus_logits=loci.detach().to(device='cpu', copy=True))
    values = scores.detach().to(device='cpu', copy=True).tolist()
    pair = {(i, j): Fraction.from_float(float(values[i][j]))
            for i in range(n) for j in range(i + 1, n)}
    locus_values = [[Fraction.from_float(float(x)) for x in row]
                    for row in snapshot.locus_logits.tolist()]
    future_positive = [Fraction(0)] * (n + 1)
    future_locus = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        future_positive[i] = future_positive[i + 1] + sum(
            (max(Fraction(0), pair[(i, j)]) for j in range(i + 1, n)), Fraction(0))
        future_locus[i] = future_locus[i + 1] + (max(locus_values[i]) if place[i] else Fraction(0))

    def upper_bound(prefix, maximum, ref_gain):
        index = len(prefix)
        reference_bound = ref_gain + future_positive[index]
        for future in range(index, n):
            cluster_gains = [Fraction(0)] * (maximum + 1)
            for past, label in enumerate(prefix):
                cluster_gains[label] += pair[(past, future)]
            reference_bound += max(Fraction(0), max(cluster_gains))
        # Retain prefix persistence but relax inter-cluster injectivity and all
        # future attachment constraints. These relaxations can only increase gain.
        placed_clusters = {}
        for event, label in enumerate(prefix):
            if place[event]:
                aggregate = placed_clusters.setdefault(label, [Fraction(0)] * len(alphabet.identities))
                for locus, value in enumerate(locus_values[event]):
                    aggregate[locus] += value
        locus_bound = future_locus[index] + sum((max(row) for row in placed_clusters.values()), Fraction(0))
        return referent_weight * reference_bound + locus_weight * locus_bound

    best = second = None
    best_refs = best_loci = None
    exhausted = False

    def offer(gain, refs, assigned):
        nonlocal best, second, best_refs, best_loci
        if best is None or gain > best:
            second, best = best, gain
            best_refs, best_loci = refs, assigned
        elif second is None or gain > second:
            second = gain

    def visit(prefix, maximum, ref_gain):
        nonlocal work, partitions, exhausted
        if work == max_work:
            exhausted = True
            return
        work += 1
        i = len(prefix)
        groups = len({prefix[j] for j in range(i) if place[j]})
        # Distinct fixed prefix clusters cannot merge in a completion.
        if groups > len(alphabet.identities):
            return
        if second is not None and upper_bound(prefix, maximum, ref_gain) <= second:
            return
        if i == n:
            partitions += 1
            if groups and work == max_work:
                exhausted = True
                return
            assignment = decode_locus_assignment(snapshot, alphabet, referents=prefix, place=place,
                                                  max_work=max(1, max_work - work))
            work += assignment.work
            if assignment.status == 'search_exhausted':
                exhausted = True
                return
            if assignment.best_gain is None:
                raise RuntimeError('unexpected infeasible assignment after capacity check')
            gain = referent_weight * ref_gain + locus_weight * assignment.best_gain
            offer(gain, prefix, assignment.loci)
            if assignment.runner_up_gain is not None:
                offer(referent_weight * ref_gain + locus_weight * assignment.runner_up_gain, None, None)
            return
        choices = []
        for label in range(maximum + 2):
            added = sum((pair[(j, i)] for j, previous in enumerate(prefix) if previous == label), Fraction(0))
            choices.append((added, label))
        for added, label in sorted(choices, key=lambda item: (-item[0], item[1])):
            visit(prefix + (label,), max(maximum, label), ref_gain + added)
            if exhausted:
                return

    visit((0,), 0, Fraction(0))
    if exhausted:
        return refused('search_exhausted')
    if best is None:
        return refused('infeasible_assignment')
    status = 'ambiguous' if second == best else 'unique_optimum_candidate'
    return JointSpatialResult(status, best_refs if status == 'unique_optimum_candidate' else None,
                              best_loci if status == 'unique_optimum_candidate' else None,
                              best, second, work, partitions)
