"""Held-out falsification tests for direct paired text/video learning.

This module evaluates already-produced per-example scores. It neither trains a
model nor assigns linguistic labels. Passing shows sensitivity to the declared
video intervention under a narrow statistical contract; it does not establish
ASL correctness, causal understanding, translation quality, or deployment
readiness.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
REQUIRED_VIDEO_INTERVENTIONS = (
    "blank_video", "shuffled_video", "order_corrupted_video", "text_only",
)


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite_tuple(name: str, values: object, expected_length: int) -> tuple[float, ...]:
    if not isinstance(values, tuple) or len(values) != expected_length:
        raise ValueError(f"{name} must be a tuple of length {expected_length}")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} must contain only finite numeric scores")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class DependenceTestConfig:
    """Preregistered decision rule shared by all four interventions."""

    familywise_alpha: float
    minimum_effective_pairs: int
    minimum_median_score_drop: float
    tie_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        scalars = (
            self.familywise_alpha, self.minimum_median_score_drop,
            self.tie_tolerance,
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in scalars):
            raise ValueError("dependence-test scalar parameters must be finite")
        if not 0 < self.familywise_alpha < 1:
            raise ValueError("familywise_alpha must lie strictly inside (0, 1)")
        if isinstance(self.minimum_effective_pairs, bool) \
                or not isinstance(self.minimum_effective_pairs, int) \
                or self.minimum_effective_pairs < 1:
            raise ValueError("minimum_effective_pairs must be a positive integer")
        if self.minimum_median_score_drop <= 0:
            raise ValueError(
                "minimum_median_score_drop must be positive and preregistered")
        if self.tie_tolerance < 0:
            raise ValueError("tie_tolerance cannot be negative")

    @property
    def per_intervention_alpha(self) -> float:
        """Bonferroni control of the four-test family-wise error rate."""
        return self.familywise_alpha / len(REQUIRED_VIDEO_INTERVENTIONS)


@dataclass(frozen=True)
class InterventionAudit:
    """Immutable provenance and source-disjointness data for one evaluation."""

    model_sha256: str
    preregistration_sha256: str
    split_certificate_sha256: str
    aligned_input_manifest_sha256: str
    blank_video_manifest_sha256: str
    shuffled_video_manifest_sha256: str
    order_corruption_manifest_sha256: str
    text_only_manifest_sha256: str
    training_source_ids: tuple[str, ...]
    training_signer_id_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "model_sha256", "preregistration_sha256", "split_certificate_sha256",
            "aligned_input_manifest_sha256", "blank_video_manifest_sha256",
            "shuffled_video_manifest_sha256", "order_corruption_manifest_sha256",
            "text_only_manifest_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in ("training_source_ids", "training_signer_id_hashes"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values \
                    or len(values) != len(set(values)) \
                    or any(not isinstance(value, str) or _ID_RE.fullmatch(value) is None
                           for value in values):
                raise ValueError(f"{name} must contain unique restricted identifiers")


@dataclass(frozen=True)
class HeldOutInterventionScores:
    """Per-example scores, with higher values declared better before evaluation."""

    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    signer_id_hashes: tuple[str, ...]
    shuffle_permutation: tuple[int, ...]
    aligned: tuple[float, ...]
    blank_video: tuple[float, ...]
    shuffled_video: tuple[float, ...]
    order_corrupted_video: tuple[float, ...]
    text_only: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sample_ids, tuple):
            raise ValueError("sample_ids must be a tuple")
        n = len(self.sample_ids)
        if n < 2 or len(set(self.sample_ids)) != n:
            raise ValueError("sample_ids must contain at least two unique held-out IDs")
        for name in ("sample_ids", "source_ids", "signer_id_hashes"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != n \
                    or any(not isinstance(value, str) or _ID_RE.fullmatch(value) is None
                           for value in values):
                raise ValueError(f"{name} must contain {n} restricted identifiers")
        if not isinstance(self.shuffle_permutation, tuple) \
                or any(isinstance(index, bool) or not isinstance(index, int)
                       for index in self.shuffle_permutation) \
                or tuple(sorted(self.shuffle_permutation)) != tuple(range(n)):
            raise ValueError("shuffle_permutation must be an exact permutation")
        if any(index == shuffled for index, shuffled
               in enumerate(self.shuffle_permutation)):
            raise ValueError("shuffle_permutation must be a derangement")
        if any(self.source_ids[index] == self.source_ids[shuffled]
               for index, shuffled in enumerate(self.shuffle_permutation)):
            raise ValueError("shuffled videos must come from different source recordings")
        for name in (
            "aligned", "blank_video", "shuffled_video",
            "order_corrupted_video", "text_only",
        ):
            _finite_tuple(name, getattr(self, name), n)


@dataclass(frozen=True)
class InterventionResult:
    name: str
    total_pairs: int
    effective_pairs: int
    aligned_wins: int
    ties: int
    median_score_drop: float
    one_sided_sign_pvalue: float
    one_sided_sign_log10_pvalue: float
    corrected_alpha: float
    passed: bool


@dataclass(frozen=True)
class VideoDependenceCertificate:
    passed: bool
    source_disjoint: bool
    signer_disjoint: bool
    results: tuple[InterventionResult, ...]
    limitations: tuple[str, ...] = (
        "sensitivity_to_declared_interventions_is_not_linguistic_validity",
        "paired_score_evidence_is_not_a_translation_quality_measure",
        "observational_evaluation_does_not_prove_causal_understanding",
    )


def _exact_one_sided_sign_log_probability(wins: int, effective_pairs: int) -> float:
    """Natural log of P[Binomial(n, 0.5) >= wins] via log-sum-exp.

    Ties must be removed before calling this function. The binomial tail is the
    exact null distribution; the returned numerical evaluation remains finite
    even when its ordinary floating-point probability would underflow to zero.
    """
    if isinstance(wins, bool) or isinstance(effective_pairs, bool) \
            or not isinstance(wins, int) or not isinstance(effective_pairs, int) \
            or effective_pairs < 0 or not 0 <= wins <= effective_pairs:
        raise ValueError("require integer 0 <= wins <= effective_pairs")
    if effective_pairs == 0:
        return 0.0
    log_terms = [
        math.lgamma(effective_pairs + 1) - math.lgamma(k + 1)
        - math.lgamma(effective_pairs - k + 1) - effective_pairs * math.log(2.0)
        for k in range(wins, effective_pairs + 1)
    ]
    largest = max(log_terms)
    return min(0.0, largest + math.log(math.fsum(
        math.exp(value - largest) for value in log_terms)))


def exact_one_sided_sign_pvalue(wins: int, effective_pairs: int) -> float:
    """Readable floating-point form of the exact one-sided sign-test tail.

    Extremely small values may be represented as zero; certification uses the
    corresponding finite log probability instead.
    """
    return math.exp(_exact_one_sided_sign_log_probability(wins, effective_pairs))


def _evaluate_one(
    name: str,
    aligned: Sequence[float],
    intervention: Sequence[float],
    config: DependenceTestConfig,
) -> InterventionResult:
    differences = tuple(float(left) - float(right)
                        for left, right in zip(aligned, intervention))
    effective = tuple(value for value in differences
                      if abs(value) > config.tie_tolerance)
    wins = sum(value > 0 for value in effective)
    log_pvalue = _exact_one_sided_sign_log_probability(wins, len(effective))
    pvalue = math.exp(log_pvalue)
    median_drop = float(statistics.median(differences))
    passed = (
        len(effective) >= config.minimum_effective_pairs
        and median_drop >= config.minimum_median_score_drop
        and log_pvalue <= math.log(config.per_intervention_alpha)
    )
    return InterventionResult(
        name=name,
        total_pairs=len(differences),
        effective_pairs=len(effective),
        aligned_wins=wins,
        ties=len(differences) - len(effective),
        median_score_drop=median_drop,
        one_sided_sign_pvalue=pvalue,
        one_sided_sign_log10_pvalue=log_pvalue / math.log(10.0),
        corrected_alpha=config.per_intervention_alpha,
        passed=passed,
    )


def evaluate_video_dependence(
    scores: HeldOutInterventionScores,
    audit: InterventionAudit,
    config: DependenceTestConfig,
) -> VideoDependenceCertificate:
    """Evaluate all required interventions under one preregistered decision rule."""
    if not isinstance(scores, HeldOutInterventionScores) \
            or not isinstance(audit, InterventionAudit) \
            or not isinstance(config, DependenceTestConfig):
        raise TypeError("scores, audit, and config must use the governed types")
    training_sources = set(audit.training_source_ids)
    training_signers = set(audit.training_signer_id_hashes)
    source_disjoint = training_sources.isdisjoint(scores.source_ids)
    signer_disjoint = training_signers.isdisjoint(scores.signer_id_hashes)
    interventions: Mapping[str, tuple[float, ...]] = {
        "blank_video": scores.blank_video,
        "shuffled_video": scores.shuffled_video,
        "order_corrupted_video": scores.order_corrupted_video,
        "text_only": scores.text_only,
    }
    results = tuple(
        _evaluate_one(name, scores.aligned, interventions[name], config)
        for name in REQUIRED_VIDEO_INTERVENTIONS
    )
    return VideoDependenceCertificate(
        passed=source_disjoint and signer_disjoint
        and all(result.passed for result in results),
        source_disjoint=source_disjoint,
        signer_disjoint=signer_disjoint,
        results=results,
    )


__all__ = [
    "REQUIRED_VIDEO_INTERVENTIONS", "DependenceTestConfig", "InterventionAudit",
    "HeldOutInterventionScores", "InterventionResult", "VideoDependenceCertificate",
    "exact_one_sided_sign_pvalue", "evaluate_video_dependence",
]
