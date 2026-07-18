"""Evaluation harness: the acceptance criteria of the speech foundation layer.

The source document requires: WER/CER, timestamp error, expected calibration
error, revision rate and downstream degradation; a transcript-only /
acoustic-only / fused ablation across clean, noisy, accented, code-switched and
long-form speech; and streaming reported with *explicit* chunk size and right
context plus median and p95 emission latency.

Two deliberate choices:

**Error rates carry their substitution/deletion/insertion breakdown.** A bare
WER hides the failure mode. Stage 3 found that this recogniser degrades by
*deleting* tokens rather than substituting them -- two systems with identical
WER can therefore need opposite fixes, and a scalar cannot tell them apart.

**Conditions are characterised empirically before they are used.** Also from
Stage 3: perturbation levels chosen by intuition produced a vacuous evaluation
(18 tokens instead of 90) because the model collapsed rather than degraded. The
harness exposes :func:`characterise_condition` so a level can be checked to lie
in the informative band before any conclusion rests on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Edit distance with an operation breakdown
# ---------------------------------------------------------------------------
@dataclass
class EditOps:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0

    @property
    def distance(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        """(S + D + I) / N_ref. May exceed 1 when insertions dominate."""
        return self.distance / self.reference_length if self.reference_length else 0.0

    def __add__(self, other: "EditOps") -> "EditOps":
        return EditOps(self.substitutions + other.substitutions,
                       self.deletions + other.deletions,
                       self.insertions + other.insertions,
                       self.reference_length + other.reference_length)

    def summary(self) -> str:
        return (f"ER={self.error_rate:.4f} (S={self.substitutions} "
                f"D={self.deletions} I={self.insertions} N={self.reference_length})")


def edit_ops(hypothesis: Sequence, reference: Sequence) -> EditOps:
    """Levenshtein distance with a backtrace, giving S/D/I counts.

    Deletion = a reference symbol the hypothesis omitted.
    Insertion = a hypothesis symbol absent from the reference.
    """
    h, r = list(hypothesis), list(reference)
    n, m = len(h), len(r)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if h[i - 1] == r[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,        # insertion (extra in hyp)
                           dp[i][j - 1] + 1,        # deletion  (missing from hyp)
                           dp[i - 1][j - 1] + cost)
    ops = EditOps(reference_length=m)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if h[i - 1] == r[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost:
                    ops.substitutions += 1
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.insertions += 1
            i -= 1
            continue
        ops.deletions += 1
        j -= 1
    return ops


def word_error_rate(hypotheses: Sequence[Sequence], references: Sequence[Sequence]
                    ) -> EditOps:
    """Corpus WER: errors and reference length are pooled, not averaged.

    Pooling is the standard definition and is not the same as the mean of
    per-utterance rates, which over-weights short utterances.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must align")
    total = EditOps()
    for h, r in zip(hypotheses, references):
        total = total + edit_ops(h, r)
    return total


def character_error_rate(hypotheses: Sequence[Sequence[int]],
                         references: Sequence[Sequence[int]],
                         spelling: Dict[int, str]) -> EditOps:
    """CER over spelled-out tokens.

    Requires a real orthography: without it "character error rate" over integer
    ids would be identical to WER and would measure nothing new. The same
    spelling table is what a fingerspelling fallback would use.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must align")

    def spell(seq: Sequence[int]) -> str:
        missing = [t for t in seq if t not in spelling]
        if missing:
            raise KeyError(f"no spelling for token(s) {missing}")
        return " ".join(spelling[t] for t in seq)

    total = EditOps()
    for h, r in zip(hypotheses, references):
        total = total + edit_ops(spell(h), spell(r))
    return total


# ---------------------------------------------------------------------------
# Timestamp error
# ---------------------------------------------------------------------------
@dataclass
class TimestampError:
    mean_start_error_s: float
    mean_end_error_s: float
    max_start_error_s: float
    count: int

    def summary(self) -> str:
        return (f"start {self.mean_start_error_s * 1000:.1f} ms | "
                f"end {self.mean_end_error_s * 1000:.1f} ms | "
                f"max start {self.max_start_error_s * 1000:.1f} ms (n={self.count})")


def timestamp_error(predicted: Sequence[Tuple[float, float]],
                    reference: Sequence[Tuple[float, float]]) -> TimestampError:
    """Absolute boundary error over positionally-matched tokens.

    Only the first ``min(len(pred), len(ref))`` pairs are scored: a timestamp for
    a token that was never spoken has no reference to be measured against, and
    inventing one would flatter the metric. Recognition errors are accounted for
    separately by WER.
    """
    n = min(len(predicted), len(reference))
    if n == 0:
        return TimestampError(0.0, 0.0, 0.0, 0)
    starts = [abs(p[0] - r[0]) for p, r in zip(predicted[:n], reference[:n])]
    ends = [abs(p[1] - r[1]) for p, r in zip(predicted[:n], reference[:n])]
    return TimestampError(sum(starts) / n, sum(ends) / n, max(starts), n)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    """One evaluation condition and how it perturbs the signal.

    ``noise`` adds Gaussian noise; ``pitch_scale`` multiplies every fundamental
    (a crude stand-in for an accent / different speaker); ``vocabulary`` selects
    which token inventory an utterance is drawn from (``"mixed"`` produces
    code-switching); ``words`` sets utterance length.
    """

    name: str
    noise: float = 0.0
    pitch_scale: float = 1.0
    vocabulary: str = "primary"      # "primary" | "secondary" | "mixed"
    words: int = 3
    is_baseline: bool = False        # a baseline is *expected* to sit at ceiling

    def describe(self) -> str:
        return (f"{self.name}: noise={self.noise} pitch x{self.pitch_scale} "
                f"vocab={self.vocabulary} words={self.words}")


# Severity levels below were **measured, not guessed**. A sweep against the
# reference recogniser gave:
#
#   noise 0.005 -> acc 0.79 | 0.010 -> 0.43 | 0.015 -> 0.25 | >=0.020 -> total
#                  collapse (nothing decoded at all)
#   pitch 1.08/1.15 -> acc 1.00 (no effect) | 1.25 -> 0.81 | 1.4 -> 0.33
#
# So the usable noise band is roughly [0.005, 0.012] and an "accent" must shift
# pitch by >=25% before this model notices. Levels chosen by intuition (noise
# 0.06, pitch 1.12) produced a *vacuous* evaluation: zero tokens decoded, or no
# degradation at all.
#
# These numbers are specific to this model and this synthetic audio. Any real
# system must be re-characterised with `characterise_condition` before its
# results mean anything.
STANDARD_CONDITIONS: Tuple[Condition, ...] = (
    Condition("clean", is_baseline=True),
    Condition("noisy", noise=0.010),
    Condition("accented", pitch_scale=1.25),
    Condition("code_switched", vocabulary="mixed"),
    Condition("long_form", words=8),
)


@dataclass
class ConditionProfile:
    """Empirical characterisation of a condition, for sanity before analysis."""

    condition: str
    tokens_decoded: int
    tokens_expected: int
    accuracy: float
    is_baseline: bool = False

    @property
    def is_informative(self) -> bool:
        """Neither trivially perfect nor a total collapse.

        A perturbed condition outside this band cannot support a conclusion: at
        ceiling it distinguishes nothing, at floor the model emits nothing to
        measure. A *baseline* is exempt -- clean speech is supposed to sit at
        ceiling, and flagging that as a defect would be wrong.
        """
        if self.is_baseline:
            return True
        coverage = self.tokens_decoded / max(self.tokens_expected, 1)
        return coverage >= 0.3 and 0.05 <= self.accuracy <= 0.999

    def summary(self) -> str:
        return (f"{self.condition:14s} decoded {self.tokens_decoded}/"
                f"{self.tokens_expected} acc={self.accuracy:.3f} "
                f"{'informative' if self.is_informative else 'DEGENERATE'}")


def characterise_condition(name: str, hypotheses: Sequence[Sequence[int]],
                           references: Sequence[Sequence[int]],
                           is_baseline: bool = False) -> ConditionProfile:
    """Measure whether a condition actually produces usable evaluation data."""
    decoded = sum(len(h) for h in hypotheses)
    expected = sum(len(r) for r in references)
    correct = sum(1 for h, r in zip(hypotheses, references)
                  for i, tok in enumerate(h) if i < len(r) and r[i] == tok)
    acc = correct / decoded if decoded else 0.0
    return ConditionProfile(condition=name, tokens_decoded=decoded,
                            tokens_expected=expected, accuracy=acc,
                            is_baseline=is_baseline)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class ArmResult:
    """Metrics for one input arm under one condition."""

    arm: str
    condition: str
    wer: EditOps
    cer: Optional[EditOps] = None
    timestamps: Optional[TimestampError] = None
    ece: Optional[float] = None
    revision_rate: Optional[float] = None
    downstream_accuracy: Optional[float] = None

    def summary(self) -> str:
        parts = [f"{self.arm:>14s}/{self.condition:<14s} {self.wer.summary()}"]
        if self.cer is not None:
            parts.append(f"CER={self.cer.error_rate:.4f}")
        if self.timestamps is not None and self.timestamps.count:
            parts.append(f"ts={self.timestamps.mean_start_error_s * 1000:.0f}ms")
        if self.ece is not None:
            parts.append(f"ECE={self.ece:.4f}")
        if self.revision_rate is not None:
            parts.append(f"rev={self.revision_rate:.4f}")
        if self.downstream_accuracy is not None:
            parts.append(f"downstream={self.downstream_accuracy:.3f}")
        return " | ".join(parts)


@dataclass
class EvaluationReport:
    results: List[ArmResult] = field(default_factory=list)
    profiles: List[ConditionProfile] = field(default_factory=list)
    streaming_config: Optional[str] = None
    latency_median_s: Optional[float] = None
    latency_p95_s: Optional[float] = None

    def add(self, result: ArmResult) -> None:
        self.results.append(result)

    def by_arm(self, arm: str) -> List[ArmResult]:
        return [r for r in self.results if r.arm == arm]

    def by_condition(self, condition: str) -> List[ArmResult]:
        return [r for r in self.results if r.condition == condition]

    def degenerate_conditions(self) -> List[str]:
        return [p.condition for p in self.profiles if not p.is_informative]

    def summary(self) -> str:
        lines = ["Speech-layer evaluation", "=" * 78]
        if self.streaming_config:
            lines.append(f"  streaming: {self.streaming_config}")
        if self.latency_median_s is not None:
            lines.append(f"  emission latency: median "
                         f"{self.latency_median_s * 1000:.1f} ms | p95 "
                         f"{self.latency_p95_s * 1000:.1f} ms")
        if self.profiles:
            lines.append("-" * 78)
            lines.extend("  " + p.summary() for p in self.profiles)
        lines.append("-" * 78)
        lines.extend("  " + r.summary() for r in self.results)
        degenerate = self.degenerate_conditions()
        if degenerate:
            lines.append("-" * 78)
            lines.append(f"  WARNING: degenerate conditions (conclusions unsafe): "
                         f"{degenerate}")
        return "\n".join(lines)
