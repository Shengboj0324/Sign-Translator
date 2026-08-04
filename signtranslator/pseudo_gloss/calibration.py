"""Reference-gated acceptance calibration and fail-closed abstention."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .contracts import UNKNOWN_TOKEN, canonical_json_bytes


@dataclass(frozen=True)
class CalibrationCertificate:
    reference_set_sha256: str
    protocol_id: str
    reference_count: int
    source_disjoint: bool
    qualified_asl_reference: bool

    def __post_init__(self) -> None:
        if (not isinstance(self.reference_set_sha256, str)
                or len(self.reference_set_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.reference_set_sha256)
                or not isinstance(self.protocol_id, str) or not self.protocol_id):
            raise ValueError("calibration reference hash and protocol are required")
        if isinstance(self.reference_count, bool) \
                or not isinstance(self.reference_count, int) or self.reference_count < 1:
            raise ValueError("calibration reference set must be non-empty")
        if not isinstance(self.source_disjoint, bool) \
                or not isinstance(self.qualified_asl_reference, bool):
            raise TypeError("calibration certificate gates must be boolean")


@dataclass(frozen=True)
class AcceptanceFeatures:
    top_posterior: float
    posterior_margin: float
    normalized_video_log_probability: float
    normalized_path_entropy: float
    mean_blank_posterior: float
    dropped_text_mass: float

    def as_array(self) -> np.ndarray:
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in asdict(self).values()):
            raise TypeError("acceptance features must be exact numeric values")
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("acceptance features must be finite")
        if not 0 <= self.top_posterior <= 1 or not 0 <= self.posterior_margin <= 1 \
                or not 0 <= self.mean_blank_posterior <= 1 \
                or not 0 <= self.dropped_text_mass <= 1:
            raise ValueError("probability-like acceptance features must lie in [0,1]")
        if self.normalized_path_entropy < 0:
            raise ValueError("normalized path entropy cannot be negative")
        return values


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    empirical_accuracy: float | None
    absolute_gap: float | None


@dataclass(frozen=True)
class CalibrationSlice:
    source_id: str
    count: int
    brier_score: float
    log_loss: float
    mean_probability: float
    empirical_accuracy: float


@dataclass(frozen=True)
class CalibrationEvaluation:
    count: int
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    maximum_calibration_error: float
    bins: tuple[ReliabilityBin, ...]
    source_slices: tuple[CalibrationSlice, ...]


@dataclass(frozen=True)
class CalibrationUncertainty:
    confidence_level: float
    bootstrap_replicates: int
    source_group_count: int
    brier_interval: tuple[float, float]
    expected_calibration_error_interval: tuple[float, float]


def _validate_calibration_inputs(probabilities: Sequence[float], labels: Sequence[bool],
                                 source_ids: Sequence[str], bin_edges: Sequence[float]
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(probabilities) != len(labels) or len(source_ids) != len(probabilities) \
            or not probabilities:
        raise ValueError("calibration inputs must align and be non-empty")
    probability = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("calibration probabilities must be finite and in [0,1]")
    if any(not isinstance(label, (bool, np.bool_)) for label in labels):
        raise TypeError("calibration labels must be boolean")
    if any(not isinstance(source_id, str) or not source_id for source_id in source_ids):
        raise ValueError("calibration source IDs must be non-empty strings")
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() \
            or edges[0] != 0 or edges[-1] != 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("calibration bin edges must strictly partition [0,1]")
    return probability, target, edges


def _binary_log_loss(probability: np.ndarray, target: np.ndarray) -> float:
    if np.any((target == 1) & (probability == 0)) \
            or np.any((target == 0) & (probability == 1)):
        return math.inf
    terms = np.empty_like(probability)
    positive = target == 1
    terms[positive] = -np.log(probability[positive])
    terms[~positive] = -np.log1p(-probability[~positive])
    return float(terms.mean())


def evaluate_calibration(probabilities: Sequence[float], labels: Sequence[bool],
                         source_ids: Sequence[str], bin_edges: Sequence[float]
                         ) -> CalibrationEvaluation:
    """Evaluate held-out calibration without clipping exact 0/1 probabilities.

    A confidently wrong boundary probability therefore produces infinite log loss,
    which is reported rather than silently repaired.
    """
    probability, target, edges = _validate_calibration_inputs(
        probabilities, labels, source_ids, bin_edges)
    bins = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        membership = ((probability >= lower) & (probability < upper)
                      if index < len(edges) - 2
                      else (probability >= lower) & (probability <= upper))
        count = int(membership.sum())
        if count:
            observed_mean = float(probability[membership].mean())
            observed_accuracy = float(target[membership].mean())
            observed_gap = abs(observed_mean - observed_accuracy)
            weighted_gap += count * observed_gap
            maximum_gap = max(maximum_gap, observed_gap)
            mean_probability_value: float | None = observed_mean
            empirical_accuracy_value: float | None = observed_accuracy
            gap_value: float | None = observed_gap
        else:
            mean_probability_value = empirical_accuracy_value = gap_value = None
        bins.append(ReliabilityBin(
            float(lower), float(upper), count, mean_probability_value,
            empirical_accuracy_value, gap_value))
    slices = []
    source_array = np.asarray(source_ids, dtype=object)
    for source_id in sorted(set(source_ids)):
        membership = source_array == source_id
        source_probability = probability[membership]
        source_target = target[membership]
        slices.append(CalibrationSlice(
            source_id=source_id, count=int(membership.sum()),
            brier_score=float(np.mean((source_probability - source_target) ** 2)),
            log_loss=_binary_log_loss(source_probability, source_target),
            mean_probability=float(source_probability.mean()),
            empirical_accuracy=float(source_target.mean()),
        ))
    return CalibrationEvaluation(
        count=len(probability),
        brier_score=float(np.mean((probability - target) ** 2)),
        log_loss=_binary_log_loss(probability, target),
        expected_calibration_error=weighted_gap / len(probability),
        maximum_calibration_error=maximum_gap,
        bins=tuple(bins), source_slices=tuple(slices),
    )


def source_cluster_calibration_uncertainty(
    probabilities: Sequence[float], labels: Sequence[bool], source_ids: Sequence[str],
    bin_edges: Sequence[float], *, confidence_level: float = 0.95,
    bootstrap_replicates: int = 10_000, seed: int = 0,
) -> CalibrationUncertainty:
    """Paired source-cluster bootstrap intervals for Brier score and ECE."""
    probability, target, _ = _validate_calibration_inputs(
        probabilities, labels, source_ids, bin_edges)
    if not 0 < confidence_level < 1 or bootstrap_replicates < 100:
        raise ValueError("invalid calibration uncertainty configuration")
    groups: dict[str, list[int]] = {}
    for index, source_id in enumerate(source_ids):
        groups.setdefault(source_id, []).append(index)
    ordered_groups = sorted(groups)
    if len(ordered_groups) < 2:
        raise ValueError("calibration uncertainty requires at least two source groups")
    seed_material = hashlib.sha256(
        f"{seed}\x1fcalibration-cluster-bootstrap".encode("utf-8")).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_material, "big"))
    brier = np.empty(bootstrap_replicates, dtype=np.float64)
    ece = np.empty(bootstrap_replicates, dtype=np.float64)
    for replicate in range(bootstrap_replicates):
        sampled = rng.choice(ordered_groups, size=len(ordered_groups), replace=True)
        indices = [index for group in sampled for index in groups[group]]
        sampled_sources = [source_ids[index] for index in indices]
        evaluation = evaluate_calibration(
            probability[indices].tolist(), target[indices].astype(bool).tolist(),
            sampled_sources, bin_edges)
        brier[replicate] = evaluation.brier_score
        ece[replicate] = evaluation.expected_calibration_error
    tail = (1.0 - confidence_level) / 2.0
    brier_interval = np.asarray(np.quantile(brier, [tail, 1.0 - tail])).reshape(2)
    ece_interval = np.asarray(np.quantile(ece, [tail, 1.0 - tail])).reshape(2)
    return CalibrationUncertainty(
        confidence_level=confidence_level,
        bootstrap_replicates=bootstrap_replicates,
        source_group_count=len(ordered_groups),
        brier_interval=(float(brier_interval[0]), float(brier_interval[1])),
        expected_calibration_error_interval=(
            float(ece_interval[0]), float(ece_interval[1])),
    )


class LogisticAcceptanceCalibrator:
    """Deterministic ridge logistic calibration fitted on qualified references.

    Newton updates optimize mean binary log loss plus L2 regularization. The
    intercept is not regularized. No fitted certificate means no confidence.
    """

    def __init__(self, ridge: float = 1e-4) -> None:
        if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) \
                or not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        self.ridge = float(ridge)
        self.coefficients: np.ndarray | None = None
        self.certificate: CalibrationCertificate | None = None

    def fit(self, features: Sequence[AcceptanceFeatures], labels: Sequence[bool],
            certificate: CalibrationCertificate, *, max_iterations: int = 100,
            tolerance: float = 1e-10) -> "LogisticAcceptanceCalibrator":
        if not certificate.qualified_asl_reference:
            raise PermissionError("qualified ASL references are required for calibration")
        if not certificate.source_disjoint:
            raise PermissionError("calibration references must be source-disjoint")
        if len(features) != len(labels) or len(features) != certificate.reference_count:
            raise ValueError("calibration features, labels, and certificate count must align")
        if any(not isinstance(value, (bool, np.bool_)) for value in labels):
            raise TypeError("calibration correctness labels must be boolean")
        if len(features) < 2 or len(set(bool(value) for value in labels)) != 2:
            raise ValueError("calibration requires both correct and incorrect references")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) \
                or isinstance(tolerance, bool) \
                or not isinstance(tolerance, (int, float)) \
                or not math.isfinite(tolerance) or max_iterations < 1 or tolerance <= 0:
            raise ValueError("calibration optimizer bounds must be positive")
        matrix = np.stack([feature.as_array() for feature in features])
        design = np.concatenate((np.ones((matrix.shape[0], 1)), matrix), axis=1)
        target = np.asarray(labels, dtype=np.float64)
        coefficients = np.zeros(design.shape[1], dtype=np.float64)
        ridge_diagonal = np.diag([0.0] + [self.ridge] * matrix.shape[1])

        def objective(parameters: np.ndarray) -> float:
            linear = design @ parameters
            data_term = np.logaddexp(0.0, linear) - target * linear
            penalty = 0.5 * float(parameters @ ridge_diagonal @ parameters)
            return float(data_term.mean()) + penalty

        for _ in range(max_iterations):
            linear = design @ coefficients
            probability = np.empty_like(linear)
            positive = linear >= 0
            probability[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
            exp_linear = np.exp(linear[~positive])
            probability[~positive] = exp_linear / (1.0 + exp_linear)
            gradient = design.T @ (probability - target) / len(target)
            gradient += ridge_diagonal @ coefficients
            curvature = probability * (1.0 - probability)
            hessian = (design.T * curvature) @ design / len(target) + ridge_diagonal
            # A minute intercept ridge handles exactly separated finite samples
            # without materially changing the declared objective.
            hessian[0, 0] += np.finfo(np.float64).eps
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError as error:
                raise ValueError("calibration Hessian is singular") from error
            current_objective = objective(coefficients)
            scale = 1.0
            while scale >= 2.0 ** -20:
                proposal = coefficients - scale * step
                if objective(proposal) <= current_objective:
                    coefficients = proposal
                    break
                scale *= 0.5
            else:
                raise FloatingPointError("calibration Newton step failed descent search")
            if not np.isfinite(coefficients).all():
                raise FloatingPointError("calibration coefficients became non-finite")
            if scale * np.linalg.norm(step, ord=np.inf) <= tolerance:
                break
        self.coefficients = coefficients
        self.certificate = certificate
        return self

    def predict(self, feature: AcceptanceFeatures) -> float:
        if self.coefficients is None or self.certificate is None:
            raise RuntimeError("acceptance calibrator has no qualified-reference fit")
        vector = np.concatenate(([1.0], feature.as_array()))
        linear = float(np.clip(vector @ self.coefficients, -40.0, 40.0))
        return 1.0 / (1.0 + math.exp(-linear))

    def state(self) -> dict:
        if self.coefficients is None or self.certificate is None:
            raise RuntimeError("cannot serialize an unfitted calibrator")
        return {
            "ridge": self.ridge,
            "coefficients": self.coefficients.tolist(),
            "certificate": asdict(self.certificate),
        }

    def state_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.state())).hexdigest()

    @classmethod
    def from_state(cls, value: dict) -> "LogisticAcceptanceCalibrator":
        if not isinstance(value, dict) \
                or set(value) != {"ridge", "coefficients", "certificate"}:
            raise ValueError("calibrator state has unexpected fields")
        if not isinstance(value["certificate"], dict) \
                or set(value["certificate"]) != set(CalibrationCertificate.__dataclass_fields__):
            raise ValueError("calibrator certificate schema mismatch")
        if not isinstance(value["coefficients"], list) \
                or len(value["coefficients"]) != 7 \
                or any(isinstance(coefficient, bool)
                       or not isinstance(coefficient, (int, float))
                       for coefficient in value["coefficients"]):
            raise ValueError("calibrator coefficients must be exact JSON numbers")
        calibrator = cls(value["ridge"])
        certificate = CalibrationCertificate(**value["certificate"])
        coefficients = np.asarray(value["coefficients"], dtype=np.float64)
        if coefficients.shape != (7,) or not np.isfinite(coefficients).all():
            raise ValueError("calibrator coefficients must be a finite length-seven vector")
        calibrator.coefficients = coefficients
        calibrator.certificate = certificate
        return calibrator


@dataclass(frozen=True)
class AbstentionConfig:
    acceptance_threshold: float
    minimum_top_posterior: float
    maximum_dropped_text_mass: float

    def __post_init__(self) -> None:
        for name in (
            "acceptance_threshold", "minimum_top_posterior", "maximum_dropped_text_mass",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0,1]")


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    reason: str
    calibrated_probability: float | None


def decide_acceptance(tokens: Sequence[str], features: AcceptanceFeatures,
                      calibrator: LogisticAcceptanceCalibrator | None,
                      config: AbstentionConfig) -> AcceptanceDecision:
    if not tokens:
        return AcceptanceDecision(False, "empty_candidate", None)
    if UNKNOWN_TOKEN in tokens:
        return AcceptanceDecision(False, "candidate_contains_UNKNOWN", None)
    if features.dropped_text_mass > config.maximum_dropped_text_mass:
        return AcceptanceDecision(False, "excessive_dropped_text_mass", None)
    if features.top_posterior < config.minimum_top_posterior:
        return AcceptanceDecision(False, "diffuse_candidate_posterior", None)
    if calibrator is None:
        return AcceptanceDecision(False, "uncalibrated_acceptance", None)
    probability = calibrator.predict(features)
    if probability < config.acceptance_threshold:
        return AcceptanceDecision(False, "calibrated_risk_above_bound", probability)
    return AcceptanceDecision(True, "accepted_under_calibrated_policy", probability)
