"""Leakage controls and label policy for future pseudo-gloss model fitting."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, Union

import torch
import torch.nn.functional as F

from ..data_engineering.exporter import LandmarkTrack
from .contracts import (
    HumanGlossAnnotation,
    LabelType,
    ReviewStatus,
    WeakGlossCandidateRecord,
)
from .model import (
    NeuralTextProposalModel,
    VideoCTCEvidenceModel,
    prepare_openpose_features,
    state_dict_sha256,
)
from .mathematics import (
    NoisyLabelObjectiveCertificate,
    confidence_weighted_loss,
    ctc_log_probability,
    multi_candidate_marginal_loss,
)
from .security import InputSecurityPolicy


TRAINING_LABEL_ALLOWLIST = frozenset({
    LabelType.OFFICIAL_HUMAN,
    LabelType.PROJECT_HUMAN,
    LabelType.HUMAN_CORRECTED_PSEUDO,
})


TrainingAnnotation = Union[HumanGlossAnnotation, WeakGlossCandidateRecord]


@dataclass(frozen=True)
class WeakSupervisionEvidence:
    annotation_id: str
    candidate_model_weight_sha256: str
    candidate_text_state_sha256: str
    candidate_video_state_sha256: str
    generator_training_source_ids: tuple[str, ...]
    generator_held_out_source_ids: tuple[str, ...]
    reference_set_sha256: str
    falsification_report_sha256: str
    calibrator_state_sha256: str
    qualified_asl_reference: bool
    all_required_falsification_tests_passed: bool
    confidence_calibrated: bool

    def __post_init__(self) -> None:
        if not self.annotation_id:
            raise ValueError("weak-supervision evidence requires an annotation ID")
        for name in (
            "candidate_model_weight_sha256", "candidate_text_state_sha256",
            "candidate_video_state_sha256", "reference_set_sha256",
            "falsification_report_sha256", "calibrator_state_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef"
                                       for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        training = self.generator_training_source_ids
        held_out = self.generator_held_out_source_ids
        if not training or not held_out or len(set(training)) != len(training) \
                or len(set(held_out)) != len(held_out) or set(training) & set(held_out) \
                or any(not source for source in training + held_out):
            raise ValueError("weak-supervision source lineage must be non-empty and disjoint")
        for name in (
            "qualified_asl_reference", "all_required_falsification_tests_passed",
            "confidence_calibrated",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")


def validate_training_annotation(
    record: TrainingAnnotation, *, source_group_id: str | None = None,
    weak_evidence: WeakSupervisionEvidence | None = None,
    current_text_state_sha256: str | None = None,
    current_video_state_sha256: str | None = None,
) -> None:
    """Reject machine-only candidates as supervised truth."""
    if not isinstance(record, (HumanGlossAnnotation, WeakGlossCandidateRecord)):
        raise TypeError("training annotation has an unsupported type")
    if record.label_type not in TRAINING_LABEL_ALLOWLIST:
        raise PermissionError("unreviewed pseudo labels cannot be supervised training targets")
    if record.review_status is not ReviewStatus.APPROVED:
        raise PermissionError("training annotation requires approved review status")
    if isinstance(record, WeakGlossCandidateRecord):
        if not record.human_annotator_pseudonym or not record.human_review_protocol:
            raise PermissionError("training annotation requires human review provenance")
        if weak_evidence is None:
            raise PermissionError(
                "pseudo-derived training requires explicit weak-supervision evidence")
        if weak_evidence.annotation_id != record.annotation_id \
                or weak_evidence.candidate_model_weight_sha256 \
                != record.provenance.model_weight_sha256:
            raise ValueError("weak-supervision evidence does not bind the candidate")
        if not weak_evidence.qualified_asl_reference \
                or not weak_evidence.all_required_falsification_tests_passed \
                or not weak_evidence.confidence_calibrated:
            raise PermissionError("weak-supervision evidence has not passed activation gates")
        if source_group_id is None \
                or source_group_id not in weak_evidence.generator_held_out_source_ids \
                or source_group_id in weak_evidence.generator_training_source_ids:
            raise PermissionError("candidate was not generated under source-disjoint cross-fitting")
        if current_text_state_sha256 is not None \
                and current_text_state_sha256 == weak_evidence.candidate_text_state_sha256:
            raise PermissionError("text model cannot self-train on candidates it generated")
        if current_video_state_sha256 is not None \
                and current_video_state_sha256 == weak_evidence.candidate_video_state_sha256:
            raise PermissionError("video model cannot self-train on candidates it generated")
    elif not record.annotator_pseudonym or not record.review_protocol:
        raise PermissionError("training annotation requires human review provenance")


@dataclass(frozen=True)
class CrossFitAssignment:
    source_id: str
    fold: int


def assign_source_folds(source_ids: Sequence[str], *, folds: int, seed: int
                        ) -> tuple[CrossFitAssignment, ...]:
    """Deterministically assign whole source groups to cross-fitting folds."""
    if folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    if not source_ids or any(not isinstance(value, str) or not value for value in source_ids):
        raise ValueError("source IDs must be non-empty strings")
    unique = sorted(set(source_ids))
    if len(unique) < folds:
        raise ValueError("cross-fitting has fewer source groups than folds")
    ordered = sorted(unique, key=lambda value: hashlib.sha256(
        f"{seed}\x1f{value}".encode("utf-8")).digest())
    source_to_fold = {source_id: index % folds for index, source_id in enumerate(ordered)}
    return tuple(CrossFitAssignment(source_id, source_to_fold[source_id])
                 for source_id in source_ids)


def certify_cross_fit(assignments: Sequence[CrossFitAssignment]) -> bool:
    """Prove each source maps to exactly one held-out fold."""
    observed: dict[str, int] = {}
    for assignment in assignments:
        if assignment.fold < 0:
            return False
        previous = observed.setdefault(assignment.source_id, assignment.fold)
        if previous != assignment.fold:
            return False
    return bool(observed) and len(set(observed.values())) >= 2


def training_indices_for_fold(assignments: Sequence[CrossFitAssignment], held_out_fold: int
                              ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if held_out_fold < 0:
        raise ValueError("held-out fold must be non-negative")
    train = tuple(index for index, item in enumerate(assignments)
                  if item.fold != held_out_fold)
    held_out = tuple(index for index, item in enumerate(assignments)
                     if item.fold == held_out_fold)
    if not train or not held_out:
        raise ValueError("cross-fit fold must have non-empty train and held-out partitions")
    train_sources = {assignments[index].source_id for index in train}
    held_sources = {assignments[index].source_id for index in held_out}
    if train_sources & held_sources:
        raise RuntimeError("source leakage across cross-fit partitions")
    return train, held_out


@dataclass(frozen=True)
class TextTrainingExample:
    # source_id is the recording-level group used for leakage control; sample_id
    # is the exact sentence/clip identifier bound by the annotation.
    source_id: str
    sample_id: str
    transcript: str
    annotation: TrainingAnnotation


@dataclass(frozen=True)
class VideoTrainingExample:
    source_id: str
    sample_id: str
    track: LandmarkTrack
    annotation: TrainingAnnotation


@dataclass(frozen=True)
class VideoCandidateLatticeTrainingExample:
    source_id: str
    sample_id: str
    track: LandmarkTrack
    candidates: tuple[WeakGlossCandidateRecord, ...]
    candidate_weights: tuple[float, ...]
    calibrated_confidence: float

    def __post_init__(self) -> None:
        if not self.source_id or not self.sample_id or not self.candidates:
            raise ValueError("candidate-lattice training example is incomplete")
        if len(self.candidates) != len(self.candidate_weights):
            raise ValueError("candidate lattice and weights must align")
        if len({candidate.annotation_id for candidate in self.candidates}) \
                != len(self.candidates):
            raise ValueError("candidate-lattice annotation IDs must be unique")
        if any(candidate.provenance.source_sample_id != self.sample_id
               for candidate in self.candidates):
            raise ValueError("candidate lattice does not bind the example sample ID")
        if any(not math.isfinite(weight) or weight < 0 for weight in self.candidate_weights) \
                or math.fsum(self.candidate_weights) <= 0:
            raise ValueError("candidate weights must be finite, non-negative, and nonzero")
        if not math.isfinite(self.calibrated_confidence) \
                or not 0 <= self.calibrated_confidence <= 1:
            raise ValueError("calibrated confidence must lie in [0,1]")


@dataclass(frozen=True)
class OptimizationConfig:
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    batch_size: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (self.epochs, self.batch_size)) \
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value)
                       for value in (self.learning_rate, self.weight_decay,
                                     self.gradient_clip)) \
                or self.learning_rate <= 0 or self.weight_decay < 0 \
                or self.gradient_clip <= 0 \
                or isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("optimization configuration is invalid")


@dataclass(frozen=True)
class TrainingHistory:
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]


@dataclass(frozen=True)
class CrossFitTrainedFold:
    fold: int
    train_indices: tuple[int, ...]
    held_out_indices: tuple[int, ...]
    model: torch.nn.Module
    history: TrainingHistory


@dataclass(frozen=True)
class TextInitializationEvidence:
    pretrained: bool
    model_id: str
    model_license: str
    checkpoint_sha256: str
    loaded_state_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.pretrained, bool):
            raise TypeError("pretrained evidence flag must be boolean")
        if not isinstance(self.model_id, str) or not self.model_id \
                or not isinstance(self.model_license, str) or not self.model_license \
                or any(not isinstance(value, str) or len(value) != 64 or any(
                    character not in "0123456789abcdef" for character in value)
                    for value in (self.checkpoint_sha256, self.loaded_state_sha256)):
            raise ValueError("text initialization evidence is incomplete")


def _validate_partitions(train_source_ids: Sequence[str], validation_source_ids: Sequence[str]
                         ) -> None:
    train_sources = set(train_source_ids)
    validation_sources = set(validation_source_ids)
    if not train_sources or not validation_sources:
        raise ValueError("training and validation partitions must be non-empty")
    overlap = train_sources & validation_sources
    if overlap:
        raise ValueError(f"source leakage between train and validation: {sorted(overlap)[:3]}")


def _finite_gradient_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                          loss: torch.Tensor, gradient_clip: float) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError("training loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not parameters or len(gradients) != len(parameters) \
            or any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("training gradients are missing or non-finite")
    norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise FloatingPointError("gradient norm is non-finite")
    optimizer.step()


def _text_example_loss(model: NeuralTextProposalModel, example: TextTrainingExample,
                       policy: InputSecurityPolicy,
                       weak_evidence: WeakSupervisionEvidence | None = None) -> torch.Tensor:
    validate_training_annotation(
        example.annotation, source_group_id=example.source_id,
        weak_evidence=weak_evidence,
        current_text_state_sha256=state_dict_sha256(model))
    example.annotation.validate_against(model.lexicon)
    source_sample_id = (example.annotation.provenance.source_sample_id
                        if isinstance(example.annotation, WeakGlossCandidateRecord)
                        else example.annotation.source_sample_id)
    if not source_sample_id or source_sample_id != example.sample_id:
        raise ValueError("annotation source sample ID does not match the training example")
    source = torch.tensor(
        [model.tokenizer.encode(example.transcript, policy)], dtype=torch.long,
        device=next(model.parameters()).device)
    tokens = (example.annotation.candidate_tokens
              if isinstance(example.annotation, WeakGlossCandidateRecord)
              else example.annotation.tokens)
    target = model.lexicon.encode(tokens)
    decoder_input = torch.tensor(
        [[model.target_bos, *target]], dtype=torch.long, device=source.device)
    expected = torch.tensor(
        [[*target, 0]], dtype=torch.long, device=source.device)
    logits = model(source, decoder_input)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), expected.reshape(-1))


def fit_text_model(model: NeuralTextProposalModel,
                   train_examples: Sequence[TextTrainingExample],
                   validation_examples: Sequence[TextTrainingExample],
                   config: OptimizationConfig,
                   initialization: TextInitializationEvidence,
                   policy: InputSecurityPolicy = InputSecurityPolicy(),
                   weak_supervision: Mapping[str, WeakSupervisionEvidence] | None = None,
                   ) -> TrainingHistory:
    """Fit only on approved human-backed annotations, with source holdout."""
    if not initialization.pretrained:
        raise PermissionError("from-scratch text-to-gloss training is prohibited")
    if state_dict_sha256(model) != initialization.loaded_state_sha256:
        raise PermissionError(
            "text model weights do not match the declared pretrained initialization")
    _validate_partitions([item.source_id for item in train_examples],
                         [item.source_id for item in validation_examples])
    torch.manual_seed(config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    train_history, validation_history = [], []
    for _ in range(config.epochs):
        model.train()
        train_losses = []
        for example in train_examples:
            evidence = (weak_supervision or {}).get(example.annotation.annotation_id)
            loss = _text_example_loss(model, example, policy, evidence)
            _finite_gradient_step(model, optimizer, loss, config.gradient_clip)
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_losses = [float(_text_example_loss(
                model, example, policy,
                (weak_supervision or {}).get(example.annotation.annotation_id)))
                                 for example in validation_examples]
        train_mean = math.fsum(train_losses) / len(train_losses)
        validation_mean = math.fsum(validation_losses) / len(validation_losses)
        if not math.isfinite(train_mean) or not math.isfinite(validation_mean):
            raise FloatingPointError("epoch loss is non-finite")
        train_history.append(train_mean)
        validation_history.append(validation_mean)
    return TrainingHistory(tuple(train_history), tuple(validation_history))


def _video_example_loss(model: VideoCTCEvidenceModel,
                        example: VideoTrainingExample,
                        weak_evidence: WeakSupervisionEvidence | None = None) -> torch.Tensor:
    validate_training_annotation(
        example.annotation, source_group_id=example.source_id,
        weak_evidence=weak_evidence,
        current_video_state_sha256=state_dict_sha256(model))
    example.annotation.validate_against(model.lexicon)
    source_sample_id = (example.annotation.provenance.source_sample_id
                        if isinstance(example.annotation, WeakGlossCandidateRecord)
                        else example.annotation.source_sample_id)
    if not source_sample_id or source_sample_id != example.sample_id:
        raise ValueError("annotation source sample ID does not match the training example")
    features, frame_validity = prepare_openpose_features(example.track)
    device = next(model.parameters()).device
    tokens = (example.annotation.candidate_tokens
              if isinstance(example.annotation, WeakGlossCandidateRecord)
              else example.annotation.tokens)
    target = torch.tensor(model.lexicon.encode(tokens),
                          dtype=torch.long, device=device)
    input_length = torch.tensor([int(frame_validity.sum())], dtype=torch.long, device=device)
    target_length = torch.tensor([target.numel()], dtype=torch.long, device=device)
    return model.loss(
        features.unsqueeze(0).to(device), target, target_length, input_length,
        frame_validity.unsqueeze(0).to(device),
    )


def _validate_lattice_candidate(
    record: WeakGlossCandidateRecord, example: VideoCandidateLatticeTrainingExample,
    evidence: WeakSupervisionEvidence | None, current_video_state_sha256: str,
) -> None:
    if evidence is None:
        raise PermissionError("candidate lattice lacks weak-supervision evidence")
    if evidence.annotation_id != record.annotation_id \
            or evidence.candidate_model_weight_sha256 != record.provenance.model_weight_sha256:
        raise ValueError("weak-supervision evidence does not bind the lattice candidate")
    if not evidence.qualified_asl_reference \
            or not evidence.all_required_falsification_tests_passed \
            or not evidence.confidence_calibrated:
        raise PermissionError("candidate lattice has not passed weak-supervision gates")
    if example.source_id not in evidence.generator_held_out_source_ids \
            or example.source_id in evidence.generator_training_source_ids:
        raise PermissionError("candidate lattice was not generated by source cross-fitting")
    if current_video_state_sha256 == evidence.candidate_video_state_sha256:
        raise PermissionError("video model cannot optimize a lattice it generated")


def candidate_lattice_video_loss(
    model: VideoCTCEvidenceModel,
    examples: Sequence[VideoCandidateLatticeTrainingExample],
    weak_supervision: Mapping[str, WeakSupervisionEvidence],
    objective_certificate: NoisyLabelObjectiveCertificate,
) -> torch.Tensor:
    """Exact confidence-weighted multi-candidate CTC objective for one minibatch."""
    objective_certificate.require_approved()
    if not examples:
        raise ValueError("candidate-lattice minibatch cannot be empty")
    device = next(model.parameters()).device
    current_video_state = state_dict_sha256(model)
    example_losses = []
    confidences = []
    for example in examples:
        features, frame_validity = prepare_openpose_features(example.track)
        log_probs = model(features.unsqueeze(0).to(device))[0]
        log_probs = log_probs[frame_validity.to(device)]
        candidate_log_probabilities = []
        for record in example.candidates:
            record.validate_against(model.lexicon)
            evidence = weak_supervision.get(record.annotation_id)
            _validate_lattice_candidate(record, example, evidence, current_video_state)
            target = model.lexicon.encode(record.candidate_tokens)
            candidate_log_probabilities.append(ctc_log_probability(log_probs, target))
        log_probability_tensor = torch.stack(candidate_log_probabilities)
        weight_tensor = torch.tensor(
            example.candidate_weights, dtype=log_probability_tensor.dtype, device=device)
        example_losses.append(multi_candidate_marginal_loss(
            log_probability_tensor, weight_tensor, objective_certificate))
        confidences.append(example.calibrated_confidence)
    return confidence_weighted_loss(
        torch.stack(example_losses),
        torch.tensor(confidences, dtype=example_losses[0].dtype, device=device),
        objective_certificate)


def fit_video_model_on_candidate_lattices(
    model: VideoCTCEvidenceModel,
    train_examples: Sequence[VideoCandidateLatticeTrainingExample],
    validation_examples: Sequence[VideoCandidateLatticeTrainingExample],
    config: OptimizationConfig,
    weak_supervision: Mapping[str, WeakSupervisionEvidence],
    objective_certificate: NoisyLabelObjectiveCertificate,
) -> TrainingHistory:
    """Optimize the dossier's noisy-label objective under all mandatory evidence gates."""
    _validate_partitions([item.source_id for item in train_examples],
                         [item.source_id for item in validation_examples])
    torch.manual_seed(config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    train_history, validation_history = [], []
    active_train = [example for example in train_examples
                    if example.calibrated_confidence > 0]
    active_validation = [example for example in validation_examples
                         if example.calibrated_confidence > 0]
    if not active_train or not active_validation:
        raise ValueError("candidate-lattice partitions require positive confidence mass")
    for _ in range(config.epochs):
        model.train()
        minibatch_losses = []
        for start in range(0, len(active_train), config.batch_size):
            minibatch = active_train[start:start + config.batch_size]
            loss = candidate_lattice_video_loss(
                model, minibatch, weak_supervision, objective_certificate)
            _finite_gradient_step(model, optimizer, loss, config.gradient_clip)
            minibatch_losses.append((
                float(loss.detach()),
                math.fsum(example.calibrated_confidence for example in minibatch)))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for start in range(0, len(active_validation), config.batch_size):
                minibatch = active_validation[start:start + config.batch_size]
                loss = candidate_lattice_video_loss(
                    model, minibatch, weak_supervision, objective_certificate)
                validation_losses.append((
                    float(loss),
                    math.fsum(example.calibrated_confidence for example in minibatch)))
        train_mass = math.fsum(mass for _, mass in minibatch_losses)
        validation_mass = math.fsum(mass for _, mass in validation_losses)
        train_mean = math.fsum(loss * mass for loss, mass in minibatch_losses) / train_mass
        validation_mean = math.fsum(
            loss * mass for loss, mass in validation_losses) / validation_mass
        if not math.isfinite(train_mean) or not math.isfinite(validation_mean):
            raise FloatingPointError("candidate-lattice epoch loss is non-finite")
        train_history.append(train_mean)
        validation_history.append(validation_mean)
    return TrainingHistory(tuple(train_history), tuple(validation_history))


def fit_video_model(model: VideoCTCEvidenceModel,
                    train_examples: Sequence[VideoTrainingExample],
                    validation_examples: Sequence[VideoTrainingExample],
                    config: OptimizationConfig,
                    weak_supervision: Mapping[str, WeakSupervisionEvidence] | None = None,
                    ) -> TrainingHistory:
    """Fit visual CTC evidence without exposing transcripts to the video model."""
    _validate_partitions([item.source_id for item in train_examples],
                         [item.source_id for item in validation_examples])
    torch.manual_seed(config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    train_history, validation_history = [], []
    for _ in range(config.epochs):
        model.train()
        train_losses = []
        for example in train_examples:
            evidence = (weak_supervision or {}).get(example.annotation.annotation_id)
            loss = _video_example_loss(model, example, evidence)
            _finite_gradient_step(model, optimizer, loss, config.gradient_clip)
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_losses = [float(_video_example_loss(
                model, example,
                (weak_supervision or {}).get(example.annotation.annotation_id)))
                                 for example in validation_examples]
        train_mean = math.fsum(train_losses) / len(train_losses)
        validation_mean = math.fsum(validation_losses) / len(validation_losses)
        if not math.isfinite(train_mean) or not math.isfinite(validation_mean):
            raise FloatingPointError("epoch loss is non-finite")
        train_history.append(train_mean)
        validation_history.append(validation_mean)
    return TrainingHistory(tuple(train_history), tuple(validation_history))


def fit_cross_fold_video_models(
    examples: Sequence[VideoTrainingExample], *, folds: int,
    model_factory: Callable[[], VideoCTCEvidenceModel],
    config: OptimizationConfig,
    weak_supervision: Mapping[str, WeakSupervisionEvidence] | None = None,
) -> tuple[CrossFitTrainedFold, ...]:
    """Train one video model per fold, excluding every held-out source group."""
    assignments = assign_source_folds(
        [example.source_id for example in examples], folds=folds, seed=config.seed)
    if not certify_cross_fit(assignments):
        raise RuntimeError("cross-fit source assignment failed certification")
    results = []
    for fold in range(folds):
        train_indices, held_out_indices = training_indices_for_fold(assignments, fold)
        torch.manual_seed(config.seed + fold)
        model = model_factory()
        fold_config = OptimizationConfig(
            epochs=config.epochs, learning_rate=config.learning_rate,
            weight_decay=config.weight_decay, gradient_clip=config.gradient_clip,
            seed=config.seed + fold)
        history = fit_video_model(
            model, [examples[index] for index in train_indices],
            [examples[index] for index in held_out_indices], fold_config,
            weak_supervision)
        results.append(CrossFitTrainedFold(
            fold, train_indices, held_out_indices, model, history))
    covered = sorted(index for result in results for index in result.held_out_indices)
    if covered != list(range(len(examples))):
        raise RuntimeError("cross-fitting did not score every example exactly once")
    return tuple(results)


def fit_cross_fold_text_models(
    examples: Sequence[TextTrainingExample], *, folds: int,
    model_factory: Callable[[], NeuralTextProposalModel],
    config: OptimizationConfig, initialization: TextInitializationEvidence,
    policy: InputSecurityPolicy = InputSecurityPolicy(),
    weak_supervision: Mapping[str, WeakSupervisionEvidence] | None = None,
) -> tuple[CrossFitTrainedFold, ...]:
    """Fine-tune pretrained text proposals without source leakage."""
    assignments = assign_source_folds(
        [example.source_id for example in examples], folds=folds, seed=config.seed)
    if not certify_cross_fit(assignments):
        raise RuntimeError("cross-fit source assignment failed certification")
    results = []
    for fold in range(folds):
        train_indices, held_out_indices = training_indices_for_fold(assignments, fold)
        torch.manual_seed(config.seed + fold)
        model = model_factory()
        fold_config = OptimizationConfig(
            epochs=config.epochs, learning_rate=config.learning_rate,
            weight_decay=config.weight_decay, gradient_clip=config.gradient_clip,
            seed=config.seed + fold)
        history = fit_text_model(
            model, [examples[index] for index in train_indices],
            [examples[index] for index in held_out_indices], fold_config,
            initialization, policy, weak_supervision)
        results.append(CrossFitTrainedFold(
            fold, train_indices, held_out_indices, model, history))
    covered = sorted(index for result in results for index in result.held_out_indices)
    if covered != list(range(len(examples))):
        raise RuntimeError("cross-fitting did not score every example exactly once")
    return tuple(results)
