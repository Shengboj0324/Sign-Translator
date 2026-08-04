"""Exact probability and selective-risk mathematics for candidate lattices."""

from __future__ import annotations

import math
from numbers import Integral
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class NoisyLabelObjectiveCertificate:
    reference_set_sha256: str
    calibration_artifact_sha256: str
    falsification_report_sha256: str
    qualified_asl_reference: bool
    source_disjoint: bool
    cross_fitted: bool
    all_required_falsification_tests_passed: bool
    confidence_calibrated: bool

    def __post_init__(self) -> None:
        for name in (
            "reference_set_sha256", "calibration_artifact_sha256",
            "falsification_report_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef"
                                       for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name in (
            "qualified_asl_reference", "source_disjoint", "cross_fitted",
            "all_required_falsification_tests_passed", "confidence_calibrated",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")

    def require_approved(self) -> None:
        if not all((self.qualified_asl_reference, self.source_disjoint,
                    self.cross_fitted, self.all_required_falsification_tests_passed,
                    self.confidence_calibrated)):
            raise PermissionError(
                "noisy-label objective lacks qualified, calibrated, cross-fit evidence")


def _integral_labels(target: Sequence[int]) -> tuple[int, ...]:
    labels: list[int] = []
    for value in target:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("CTC targets must contain integral non-boolean class IDs")
        labels.append(int(value))
    return tuple(labels)


def ctc_minimum_frames(target: Sequence[int]) -> int:
    labels = _integral_labels(target)
    return len(labels) + sum(left == right for left, right in zip(labels, labels[1:]))


def _validate_ctc(log_probs: torch.Tensor, target: Sequence[int], blank: int) -> tuple[int, ...]:
    if log_probs.ndim != 2 or log_probs.shape[0] < 1 or log_probs.shape[1] < 2:
        raise ValueError("log_probs must have shape (T, C) with T>=1 and C>=2")
    if not torch.isfinite(log_probs).all():
        raise ValueError("log_probs must be finite")
    if blank < 0 or blank >= log_probs.shape[1]:
        raise ValueError("blank index is outside the class dimension")
    normalization = torch.logsumexp(log_probs.detach(), dim=-1)
    tolerance = 1e-6 if log_probs.dtype == torch.float64 else 2e-5
    if not torch.allclose(normalization, torch.zeros_like(normalization), atol=tolerance, rtol=0):
        raise ValueError("log_probs must be normalized log-probabilities")
    labels = _integral_labels(target)
    if any(value == blank or value < 0 or value >= log_probs.shape[1] for value in labels):
        raise ValueError("CTC targets must be valid non-blank class indices")
    if log_probs.shape[0] < ctc_minimum_frames(labels):
        raise ValueError("CTC target is infeasible for the encoder output length")
    return labels


def ctc_log_probability(log_probs: torch.Tensor, target: Sequence[int],
                        blank: int = 0) -> torch.Tensor:
    """Exact log CTC probability by a differentiable forward dynamic program."""
    labels = _validate_ctc(log_probs, target, blank)
    time_steps = log_probs.shape[0]
    if not labels:
        return log_probs[:, blank].sum()

    extended = [blank]
    for label in labels:
        extended.extend((label, blank))
    states = len(extended)
    alpha = log_probs.new_full((states,), float("-inf"))
    alpha[0] = log_probs[0, blank]
    alpha[1] = log_probs[0, labels[0]]
    for time in range(1, time_steps):
        next_alpha = []
        for state, label in enumerate(extended):
            incoming = [alpha[state]]
            if state > 0:
                incoming.append(alpha[state - 1])
            if state > 1 and label != blank and label != extended[state - 2]:
                incoming.append(alpha[state - 2])
            # ``logsumexp([-inf, ...])`` has the correct forward value but an
            # undefined 0/0 softmax derivative in PyTorch. Unreachable states
            # are a structural property of the CTC graph (all supplied frame
            # probabilities are finite), so omit those predecessors exactly.
            reachable = [value for value in incoming if bool(torch.isfinite(value.detach()))]
            if reachable:
                total = torch.logsumexp(torch.stack(reachable), dim=0)
                next_alpha.append(total + log_probs[time, label])
            else:
                next_alpha.append(log_probs.new_tensor(float("-inf")))
        alpha = torch.stack(next_alpha)
    result = torch.logaddexp(alpha[-1], alpha[-2])
    if not torch.isfinite(result):
        raise ValueError("CTC forward probability is zero or non-finite")
    return result


@dataclass(frozen=True)
class CTCAlignmentDiagnostics:
    log_probability: float
    path_entropy_nats: float
    mean_blank_posterior: float
    time_steps: int
    target_length: int
    minimum_frames: int


def ctc_alignment_diagnostics(log_probs: torch.Tensor, target: Sequence[int],
                              blank: int = 0) -> CTCAlignmentDiagnostics:
    """Compute exact path entropy and mean blank posterior using CTC marginals.

    For path posterior ``p(pi|G,V)``, ``H = log Z - E[log p(pi|V)]``. The
    gradient of ``log Z`` with respect to each frame/class log probability is
    the corresponding posterior occupancy, so the expectation is exact up to
    floating-point roundoff.
    """
    # Diagnostics need the derivative of log Z even when called from a frozen
    # outer inference context. This local graph is isolated from model weights.
    with torch.enable_grad():
        working = log_probs.detach().to(dtype=torch.float64).clone().requires_grad_(True)
        value = ctc_log_probability(working, target, blank)
        occupancy, = torch.autograd.grad(value, working)
        expected_log_path = (occupancy * working).sum()
        entropy = value - expected_log_path
    entropy_value = float(entropy.detach())
    if entropy_value < -1e-9:
        raise RuntimeError("computed negative CTC path entropy")
    return CTCAlignmentDiagnostics(
        log_probability=float(value.detach()),
        path_entropy_nats=max(0.0, entropy_value),
        mean_blank_posterior=float(occupancy[:, blank].mean().detach()),
        time_steps=int(working.shape[0]),
        target_length=len(tuple(target)),
        minimum_frames=ctc_minimum_frames(target),
    )


@dataclass(frozen=True)
class FusedCandidate:
    tokens: tuple[str, ...]
    text_log_probability: float
    video_log_probability: float
    cost: float
    unnormalized_log_score: float
    posterior_log_probability: float


def fuse_candidate_lattice(
    candidates: Sequence[tuple[Sequence[str], float, float, float]], *,
    alpha: float, beta: float, penalty_weight: float,
) -> tuple[FusedCandidate, ...]:
    """Normalize the documented finite log-linear candidate lattice exactly."""
    if not candidates:
        raise ValueError("candidate lattice cannot be empty")
    coefficients = (alpha, beta, penalty_weight)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in coefficients):
        raise ValueError("fusion coefficients must be finite and non-negative")
    token_sequences: list[tuple[str, ...]] = []
    raw_scores: list[float] = []
    validated: list[tuple[float, float, float]] = []
    for tokens, text_log_probability, video_log_probability, cost in candidates:
        sequence = tuple(tokens)
        values = (float(text_log_probability), float(video_log_probability), float(cost))
        if not sequence or any(not token for token in sequence):
            raise ValueError("candidate sequences must be non-empty")
        if any(not math.isfinite(value) for value in values) or cost < 0:
            raise ValueError("candidate scores must be finite and cost non-negative")
        if text_log_probability > 0 or video_log_probability > 0:
            raise ValueError("log probabilities cannot exceed zero")
        token_sequences.append(sequence)
        validated.append(values)
        raw_scores.append(
            alpha * text_log_probability + beta * video_log_probability
            - penalty_weight * cost)
    if len(set(token_sequences)) != len(token_sequences):
        raise ValueError("candidate lattice contains duplicate token sequences")
    tensor = torch.tensor(raw_scores, dtype=torch.float64)
    normalizer = float(torch.logsumexp(tensor, dim=0))
    result = []
    for sequence, values, raw_score in zip(token_sequences, validated, raw_scores):
        text_lp, video_lp, cost = values
        result.append(FusedCandidate(
            tokens=sequence, text_log_probability=text_lp,
            video_log_probability=video_lp, cost=cost,
            unnormalized_log_score=raw_score,
            posterior_log_probability=raw_score - normalizer,
        ))
    return tuple(sorted(result, key=lambda item: (-item.posterior_log_probability,
                                                   item.tokens)))


def multi_candidate_marginal_loss(
    candidate_log_probabilities: torch.Tensor,
    candidate_weights: torch.Tensor,
    certificate: NoisyLabelObjectiveCertificate,
) -> torch.Tensor:
    """``-log sum_g w(g) P(g|V)`` for one finite candidate lattice."""
    certificate.require_approved()
    if candidate_log_probabilities.ndim != 1 or candidate_weights.ndim != 1:
        raise ValueError("candidate inputs must be one-dimensional")
    if candidate_log_probabilities.shape != candidate_weights.shape \
            or candidate_log_probabilities.numel() < 1:
        raise ValueError("candidate probabilities and weights must align and be non-empty")
    if not torch.isfinite(candidate_log_probabilities).all() \
            or not torch.isfinite(candidate_weights).all():
        raise ValueError("candidate inputs must be finite")
    if torch.any(candidate_log_probabilities > 1e-7):
        raise ValueError("candidate log probabilities cannot exceed zero")
    if torch.any(candidate_weights < 0):
        raise ValueError("candidate weights cannot be negative")
    weight_sum = candidate_weights.sum()
    if not torch.isfinite(weight_sum) or float(weight_sum.detach()) <= 0:
        raise ValueError("candidate weights must have positive mass")
    normalized_log_weights = torch.log(candidate_weights / weight_sum)
    return -torch.logsumexp(normalized_log_weights + candidate_log_probabilities, dim=0)


def confidence_weighted_loss(
    losses: torch.Tensor, confidences: torch.Tensor,
    certificate: NoisyLabelObjectiveCertificate,
) -> torch.Tensor:
    certificate.require_approved()
    if losses.ndim != 1 or confidences.ndim != 1 or losses.shape != confidences.shape:
        raise ValueError("losses and confidences must be aligned vectors")
    if losses.numel() < 1 or not torch.isfinite(losses).all() \
            or not torch.isfinite(confidences).all():
        raise ValueError("losses and confidences must be non-empty and finite")
    if torch.any((confidences < 0) | (confidences > 1)):
        raise ValueError("confidences must lie in [0, 1]")
    denominator = confidences.sum()
    if float(denominator.detach()) <= 0:
        raise ValueError("confidence mass must be positive")
    return (losses * confidences).sum() / denominator


def selective_risk_curve(scores: Sequence[float], losses: Sequence[float],
                         thresholds: Sequence[float]) -> tuple[dict[str, float | int], ...]:
    """Compute coverage and conditional risk for predeclared thresholds."""
    if len(scores) != len(losses) or not scores:
        raise ValueError("scores and losses must align and be non-empty")
    if any(not math.isfinite(value) for value in (*scores, *losses, *thresholds)):
        raise ValueError("selective-risk inputs must be finite")
    if any(loss < 0 for loss in losses):
        raise ValueError("loss values must be non-negative")
    rows = []
    count = len(scores)
    for threshold in thresholds:
        accepted = [index for index, score in enumerate(scores) if score >= threshold]
        rows.append({
            "threshold": float(threshold),
            "accepted": len(accepted),
            "coverage": len(accepted) / count,
            "selective_risk": (sum(losses[index] for index in accepted) / len(accepted)
                               if accepted else math.nan),
        })
    return tuple(rows)
