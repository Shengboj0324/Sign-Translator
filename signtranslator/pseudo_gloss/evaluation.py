"""Pre-registered falsification and source-group uncertainty analysis."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .contracts import canonical_json_bytes, sha256_bytes


REQUIRED_FALSIFICATION_TESTS = (
    "text_only",
    "blank_video",
    "shuffled_video",
    "order_corruption",
    "candidate_deletion",
    "vocabulary_holdout",
    "source_holdout",
    "counterfactual_sentences",
    "human_reference",
)


@dataclass(frozen=True)
class InterventionSpecification:
    name: str
    minimum_mean_decline: float
    null_hypothesis: str
    effect_measure: str
    stop_rule: str
    confidence_level: float = 0.95
    bootstrap_replicates: int = 10_000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_FALSIFICATION_TESTS:
            raise ValueError("unknown falsification test")
        if isinstance(self.minimum_mean_decline, bool) \
                or not isinstance(self.minimum_mean_decline, (int, float)) \
                or not math.isfinite(self.minimum_mean_decline) \
                or self.minimum_mean_decline < 0:
            raise ValueError("minimum decline must be finite and non-negative")
        if isinstance(self.confidence_level, bool) \
                or not isinstance(self.confidence_level, (int, float)) \
                or not math.isfinite(self.confidence_level) \
                or not 0 < self.confidence_level < 1 \
                or isinstance(self.bootstrap_replicates, bool) \
                or not isinstance(self.bootstrap_replicates, int) \
                or self.bootstrap_replicates < 100:
            raise ValueError("invalid confidence level or bootstrap count")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("falsification seed must be an integer")
        if not self.null_hypothesis or not self.effect_measure or not self.stop_rule:
            raise ValueError("falsification null, effect measure, and stop rule are required")

    def content_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes({
            "name": self.name,
            "minimum_mean_decline": self.minimum_mean_decline,
            "null_hypothesis": self.null_hypothesis,
            "effect_measure": self.effect_measure,
            "stop_rule": self.stop_rule,
            "confidence_level": self.confidence_level,
            "bootstrap_replicates": self.bootstrap_replicates,
            "seed": self.seed,
        }))


@dataclass(frozen=True)
class InterventionResult:
    name: str
    sample_count: int
    source_group_count: int
    mean_decline: float
    lower_confidence_bound: float
    upper_confidence_bound: float
    required_decline: float
    specification_sha256: str
    passed: bool

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_FALSIFICATION_TESTS:
            raise ValueError("unknown falsification result")
        if self.sample_count < 1 or self.source_group_count < 2:
            raise ValueError("falsification result has insufficient observations")
        values = (self.mean_decline, self.lower_confidence_bound,
                  self.upper_confidence_bound, self.required_decline)
        if any(not math.isfinite(value) for value in values) or self.required_decline < 0:
            raise ValueError("falsification result statistics must be finite")
        if self.lower_confidence_bound > self.upper_confidence_bound:
            raise ValueError("falsification confidence interval is reversed")
        if len(self.specification_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in self.specification_sha256):
            raise ValueError("falsification result specification hash is invalid")
        if not isinstance(self.passed, bool) \
                or self.passed != (self.lower_confidence_bound > self.required_decline):
            raise ValueError("falsification pass flag contradicts its confidence bound")


@dataclass(frozen=True)
class TokenErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        if self.reference_length < 1:
            raise ValueError("token error rate requires a non-empty reference")
        return self.errors / self.reference_length


@dataclass(frozen=True)
class HumanReferenceCase:
    sample_id: str
    source_id: str
    reference_tokens: tuple[str, ...]
    candidate_sequences: tuple[tuple[str, ...], ...]
    accepted_tokens: tuple[str, ...] | None
    construction_tags: tuple[str, ...]
    construction_acceptability: Mapping[str, bool]

    def __post_init__(self) -> None:
        if not self.sample_id or not self.source_id or not self.reference_tokens:
            raise ValueError("human-reference case requires IDs and reference tokens")
        if not self.candidate_sequences or any(not sequence for sequence in self.candidate_sequences):
            raise ValueError("human-reference case requires a non-empty candidate lattice")
        if len(set(self.candidate_sequences)) != len(self.candidate_sequences):
            raise ValueError("human-reference candidate sequences must be unique")
        if len(set(self.construction_tags)) != len(self.construction_tags):
            raise ValueError("construction tags must be unique")
        if set(self.construction_acceptability) != set(self.construction_tags) \
                or any(not isinstance(value, bool)
                       for value in self.construction_acceptability.values()):
            raise ValueError("every construction tag requires a boolean human judgment")


@dataclass(frozen=True)
class HumanReferenceCaseResult:
    sample_id: str
    source_id: str
    candidate_recall: bool
    accepted: bool
    exact_match: bool
    token_errors: TokenErrorCounts | None
    order_error_rate: float | None


@dataclass(frozen=True)
class ConstructionSlice:
    name: str
    count: int
    acceptable_count: int
    acceptability_rate: float


@dataclass(frozen=True)
class HumanReferenceEvaluation:
    case_count: int
    source_group_count: int
    candidate_recall: float
    coverage: float
    exact_match_rate_among_accepted: float | None
    token_error_rate_among_accepted: float | None
    order_error_rate_among_accepted: float | None
    substitutions: int
    deletions: int
    insertions: int
    cases: tuple[HumanReferenceCaseResult, ...]
    construction_slices: tuple[ConstructionSlice, ...]


def token_error_counts(reference: Sequence[str], hypothesis: Sequence[str]
                       ) -> TokenErrorCounts:
    """Levenshtein S/D/I counts with a declared deterministic tie rule.

    Equal-distance paths minimize substitutions, then deletions, then insertions.
    This makes the decomposition reproducible; total edit distance is invariant to
    the tie rule while individual S/D/I counts need not be.
    """
    reference_tokens = tuple(reference)
    hypothesis_tokens = tuple(hypothesis)
    if not reference_tokens or any(not token for token in reference_tokens + hypothesis_tokens):
        raise ValueError("token error analysis requires non-empty valid reference tokens")
    # State tuple is (total errors, substitutions, deletions, insertions).
    table: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(len(hypothesis_tokens) + 1)]
        for _ in range(len(reference_tokens) + 1)
    ]
    for row in range(1, len(reference_tokens) + 1):
        table[row][0] = (row, 0, row, 0)
    for column in range(1, len(hypothesis_tokens) + 1):
        table[0][column] = (column, 0, 0, column)
    for row, reference_token in enumerate(reference_tokens, start=1):
        for column, hypothesis_token in enumerate(hypothesis_tokens, start=1):
            if reference_token == hypothesis_token:
                table[row][column] = table[row - 1][column - 1]
                continue
            previous_substitution = table[row - 1][column - 1]
            previous_deletion = table[row - 1][column]
            previous_insertion = table[row][column - 1]
            candidates = (
                (previous_substitution[0] + 1, previous_substitution[1] + 1,
                 previous_substitution[2], previous_substitution[3]),
                (previous_deletion[0] + 1, previous_deletion[1],
                 previous_deletion[2] + 1, previous_deletion[3]),
                (previous_insertion[0] + 1, previous_insertion[1],
                 previous_insertion[2], previous_insertion[3] + 1),
            )
            table[row][column] = min(
                candidates, key=lambda value: (value[0], value[1], value[2], value[3]))
    _, substitutions, deletions, insertions = table[-1][-1]
    return TokenErrorCounts(
        substitutions, deletions, insertions, len(reference_tokens))


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for column, right_token in enumerate(right, start=1):
            current.append(previous[column - 1] + 1 if left_token == right_token
                           else max(previous[column], current[-1]))
        previous = current
    return previous[-1]


def evaluate_human_references(cases: Sequence[HumanReferenceCase]
                              ) -> HumanReferenceEvaluation:
    """Evaluate candidate recall and accepted-sequence errors against human gloss."""
    if not cases:
        raise ValueError("human-reference evaluation requires at least one case")
    sample_ids = [case.sample_id for case in cases]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("human-reference sample IDs must be unique")
    results = []
    accepted_error_counts = []
    order_errors = []
    construction_values: dict[str, list[bool]] = {}
    for case in cases:
        recalled = case.reference_tokens in case.candidate_sequences
        if case.accepted_tokens is None:
            result = HumanReferenceCaseResult(
                case.sample_id, case.source_id, recalled, False, False, None, None)
        else:
            if case.accepted_tokens not in case.candidate_sequences:
                raise ValueError("accepted tokens must identify a candidate in the lattice")
            counts = token_error_counts(case.reference_tokens, case.accepted_tokens)
            order_error = 1.0 - _lcs_length(
                case.reference_tokens, case.accepted_tokens) / len(case.reference_tokens)
            accepted_error_counts.append(counts)
            order_errors.append(order_error)
            result = HumanReferenceCaseResult(
                case.sample_id, case.source_id, recalled, True,
                case.accepted_tokens == case.reference_tokens, counts, order_error)
        results.append(result)
        for tag, acceptable in case.construction_acceptability.items():
            construction_values.setdefault(tag, []).append(acceptable)
    accepted_count = len(accepted_error_counts)
    total_reference_tokens = sum(item.reference_length for item in accepted_error_counts)
    substitutions = sum(item.substitutions for item in accepted_error_counts)
    deletions = sum(item.deletions for item in accepted_error_counts)
    insertions = sum(item.insertions for item in accepted_error_counts)
    slices = tuple(ConstructionSlice(
        name, len(values), sum(values), sum(values) / len(values))
        for name, values in sorted(construction_values.items()))
    return HumanReferenceEvaluation(
        case_count=len(cases), source_group_count=len({case.source_id for case in cases}),
        candidate_recall=sum(result.candidate_recall for result in results) / len(results),
        coverage=accepted_count / len(cases),
        exact_match_rate_among_accepted=(
            sum(result.exact_match for result in results if result.accepted) / accepted_count
            if accepted_count else None),
        token_error_rate_among_accepted=(
            (substitutions + deletions + insertions) / total_reference_tokens
            if total_reference_tokens else None),
        order_error_rate_among_accepted=(
            math.fsum(order_errors) / accepted_count if accepted_count else None),
        substitutions=substitutions, deletions=deletions, insertions=insertions,
        cases=tuple(results), construction_slices=slices,
    )


def paired_source_bootstrap(baseline_scores: Sequence[float],
                            intervention_scores: Sequence[float],
                            source_ids: Sequence[str],
                            specification: InterventionSpecification
                            ) -> InterventionResult:
    """Paired cluster bootstrap; complete source groups are resampled."""
    if len(baseline_scores) != len(intervention_scores) or len(source_ids) != len(baseline_scores) \
            or not baseline_scores:
        raise ValueError("paired intervention inputs must align and be non-empty")
    arrays = np.asarray([baseline_scores, intervention_scores], dtype=np.float64)
    if not np.isfinite(arrays).all() or any(
            not isinstance(source_id, str) or not source_id for source_id in source_ids):
        raise ValueError("intervention scores and source IDs must be valid")
    differences = arrays[0] - arrays[1]
    groups: dict[str, list[int]] = {}
    for index, source_id in enumerate(source_ids):
        groups.setdefault(source_id, []).append(index)
    ordered_groups = sorted(groups)
    if len(ordered_groups) < 2:
        raise ValueError("cluster uncertainty requires at least two source groups")
    # Stable seed binds both the declared seed and test name.
    seed_bytes = hashlib.sha256(
        f"{specification.seed}\x1f{specification.name}".encode("utf-8")).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
    replicates = np.empty(specification.bootstrap_replicates, dtype=np.float64)
    for replicate in range(specification.bootstrap_replicates):
        sampled = rng.choice(ordered_groups, size=len(ordered_groups), replace=True)
        indices = [index for group in sampled for index in groups[group]]
        replicates[replicate] = differences[indices].mean()
    tail = (1.0 - specification.confidence_level) / 2.0
    bounds = np.asarray(np.quantile(replicates, [tail, 1.0 - tail]),
                        dtype=np.float64).reshape(2)
    lower, upper = float(bounds[0]), float(bounds[1])
    mean = float(differences.mean())
    return InterventionResult(
        name=specification.name, sample_count=len(differences),
        source_group_count=len(ordered_groups), mean_decline=mean,
        lower_confidence_bound=lower, upper_confidence_bound=upper,
        required_decline=specification.minimum_mean_decline,
        specification_sha256=specification.content_sha256(),
        passed=lower > specification.minimum_mean_decline,
    )


@dataclass(frozen=True)
class FalsificationSuiteReport:
    results: tuple[InterventionResult, ...]
    human_reference_completed: bool
    video_reliance_certified: bool
    linguistic_validation_certified: bool = False


def build_falsification_report(results: Sequence[InterventionResult], *,
                               human_reference_completed: bool,
                               source_holdout_completed: bool) -> FalsificationSuiteReport:
    names = [result.name for result in results]
    if len(names) != len(set(names)):
        raise ValueError("falsification report contains duplicate tests")
    video_tests = {"blank_video", "shuffled_video", "order_corruption"}
    by_name = {result.name: result for result in results}
    video_certified = video_tests <= set(by_name) and all(
        by_name[name].passed for name in video_tests)
    # Human reference completion is necessary but not sufficient for a linguistic
    # claim; certification requires an external reviewed protocol not represented
    # by this automatic report.
    linguistic = False
    if source_holdout_completed and "source_holdout" not in by_name:
        raise ValueError("source holdout was claimed without a recorded result")
    return FalsificationSuiteReport(
        results=tuple(results), human_reference_completed=human_reference_completed,
        video_reliance_certified=video_certified,
        linguistic_validation_certified=linguistic,
    )


def certify_vocabulary_holdout(train_tokens: Sequence[Sequence[str]],
                               held_out_tokens: Sequence[Sequence[str]],
                               lexical_family: Mapping[str, str]) -> bool:
    """Ensure no declared lexical family crosses a vocabulary holdout."""
    train = {token for sequence in train_tokens for token in sequence}
    held = {token for sequence in held_out_tokens for token in sequence}
    if not train or not held or train & held:
        return False
    if any(token not in lexical_family for token in train | held):
        raise ValueError("every token requires an explicit lexical-family assignment")
    return {lexical_family[token] for token in train}.isdisjoint(
        {lexical_family[token] for token in held})


def deterministic_source_derangement(source_ids: Sequence[str],
                                     strata: Sequence[str], *, seed: int
                                     ) -> tuple[int, ...]:
    """Map each sample to another source inside its declared comparison stratum.

    This constructs a video-shuffling intervention without permitting a sample to
    retain its own source. A stratum with fewer than two distinct sources is
    unidentifiable and therefore rejected instead of silently cross-stratifying.
    """
    if len(source_ids) != len(strata) or not source_ids:
        raise ValueError("source IDs and strata must align and be non-empty")
    if any(not value for value in source_ids) or any(not value for value in strata):
        raise ValueError("source IDs and strata must be non-empty strings")
    indices_by_stratum: dict[str, list[int]] = {}
    for index, stratum in enumerate(strata):
        indices_by_stratum.setdefault(stratum, []).append(index)
    result = [-1] * len(source_ids)
    for stratum in sorted(indices_by_stratum):
        indices = indices_by_stratum[stratum]
        unique_sources = sorted({source_ids[index] for index in indices})
        if len(unique_sources) < 2:
            raise ValueError(f"stratum {stratum!r} has fewer than two distinct sources")
        ordered_sources = sorted(unique_sources, key=lambda source: hashlib.sha256(
            f"{seed}\x1f{stratum}\x1f{source}".encode("utf-8")).digest())
        donor_for_source = {
            source: ordered_sources[(position + 1) % len(ordered_sources)]
            for position, source in enumerate(ordered_sources)
        }
        first_index = {
            source: min(index for index in indices if source_ids[index] == source)
            for source in unique_sources
        }
        for index in indices:
            result[index] = first_index[donor_for_source[source_ids[index]]]
    if any(index < 0 for index in result):
        raise RuntimeError("derangement did not assign every sample")
    if any(source_ids[index] == source_ids[donor]
           or strata[index] != strata[donor] for index, donor in enumerate(result)):
        raise RuntimeError("derangement violated source or stratum constraints")
    return tuple(result)


def candidate_deletion_abstains(selected_annotation_id: str | None,
                                remaining_annotation_ids: Sequence[str]) -> bool:
    """Certificate for the candidate-deletion intervention.

    Deleting the selected candidate must lead to abstention. The system may not
    silently relabel a lower-ranked survivor as the original decision.
    """
    if selected_annotation_id is None:
        raise ValueError("candidate-deletion test requires a selected baseline candidate")
    if not remaining_annotation_ids or any(not value for value in remaining_annotation_ids):
        raise ValueError("remaining candidate IDs must be non-empty")
    if len(set(remaining_annotation_ids)) != len(remaining_annotation_ids):
        raise ValueError("remaining candidate IDs must be unique")
    return selected_annotation_id not in remaining_annotation_ids
