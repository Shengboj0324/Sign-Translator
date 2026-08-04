"""End-to-end hybrid candidate-lattice inference with mandatory abstention."""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable

import torch

from ..data_engineering.exporter import LandmarkTrack
from .calibration import (
    AbstentionConfig,
    AcceptanceDecision,
    AcceptanceFeatures,
    LogisticAcceptanceCalibrator,
    decide_acceptance,
)
from .contracts import (
    CandidateProvenance,
    LabelType,
    ReviewStatus,
    WeakGlossCandidateRecord,
    canonical_json_bytes,
    sha256_bytes,
)
from .mathematics import ctc_alignment_diagnostics, fuse_candidate_lattice
from .model import (
    CandidateLatticeProposal,
    NeuralTextProposalModel,
    VideoCTCEvidenceModel,
    prepare_openpose_features,
    state_dict_sha256,
)
from .security import InputSecurityPolicy, runtime_environment, validate_transcript


_DETERMINISM_LOCK = threading.RLock()


@dataclass(frozen=True)
class FusionConfig:
    alpha: float
    beta: float
    alignment_entropy_penalty: float
    candidate_limit: int = 16

    def __post_init__(self) -> None:
        values = (self.alpha, self.beta, self.alignment_entropy_penalty)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("fusion weights must be finite and non-negative")
        if self.alpha == 0 and self.beta == 0:
            raise ValueError("at least one evidence exponent must be positive")
        if isinstance(self.candidate_limit, bool) \
                or not isinstance(self.candidate_limit, int) \
                or self.candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")


@dataclass(frozen=True)
class CandidateGenerationResult:
    records: tuple[WeakGlossCandidateRecord, ...]
    decision: AcceptanceDecision
    selected_annotation_id: str | None
    dropped_text_probability_mass: float
    machine_only: bool = True
    translation_claim: bool = False
    linguistic_validation_claim: bool = False


class HybridPseudoGlossPipeline:
    """Fuse a frozen text lattice with independent frozen visual CTC evidence."""

    def __init__(self, text_model: NeuralTextProposalModel,
                 video_model: VideoCTCEvidenceModel,
                 fusion: FusionConfig,
                 abstention: AbstentionConfig,
                 security: InputSecurityPolicy = InputSecurityPolicy(),
                 calibrator: LogisticAcceptanceCalibrator | None = None) -> None:
        if text_model.lexicon != video_model.lexicon:
            raise ValueError("text and video models must use the exact same lexicon")
        if fusion.candidate_limit > security.max_candidates:
            raise ValueError("fusion candidate limit exceeds security policy")
        self.text_model = text_model
        self.video_model = video_model
        self.fusion = fusion
        self.abstention = abstention
        self.security = security
        self.calibrator = calibrator

    def freeze_for_inference(self) -> None:
        self.text_model.freeze_for_generation()
        self.video_model.eval()
        for parameter in self.video_model.parameters():
            parameter.requires_grad_(False)

    def _assert_frozen(self) -> None:
        self.text_model._assert_frozen()
        if self.video_model.training or any(
                parameter.requires_grad for parameter in self.video_model.parameters()):
            raise RuntimeError("video evidence model must be frozen and in eval mode")

    def generate(self, *, transcript: str, track: LandmarkTrack,
                 source_sample_id: str, source_video_sha256: str,
                 generator_model_id: str, code_revision: str, random_seed: int,
                 created_at: str | None = None) -> CandidateGenerationResult:
        """Generate under a process-wide deterministic-algorithm critical section."""
        with _DETERMINISM_LOCK:
            previous_enabled = torch.are_deterministic_algorithms_enabled()
            previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
            torch.use_deterministic_algorithms(True, warn_only=False)
            try:
                return self._generate_deterministic(
                    transcript=transcript, track=track,
                    source_sample_id=source_sample_id,
                    source_video_sha256=source_video_sha256,
                    generator_model_id=generator_model_id,
                    code_revision=code_revision, random_seed=random_seed,
                    created_at=created_at)
            finally:
                torch.use_deterministic_algorithms(
                    previous_enabled, warn_only=previous_warn_only)

    @torch.no_grad()
    def _generate_deterministic(self, *, transcript: str, track: LandmarkTrack,
                                source_sample_id: str, source_video_sha256: str,
                                generator_model_id: str, code_revision: str,
                                random_seed: int,
                                created_at: str | None = None,
                                ) -> CandidateGenerationResult:
        self._assert_frozen()
        if not source_sample_id or not generator_model_id or not code_revision:
            raise ValueError("source, model, and code identifiers are required")
        if (len(source_video_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in source_video_sha256)):
            raise ValueError("source_video_sha256 must be a lowercase SHA-256 digest")
        transcript_bytes = validate_transcript(transcript, self.security)
        proposal = self.text_model.propose(
            transcript, self.security, candidate_limit=self.fusion.candidate_limit)
        if not proposal.candidates:
            return CandidateGenerationResult(
                records=(), decision=AcceptanceDecision(False, "empty_lattice", None),
                selected_annotation_id=None,
                dropped_text_probability_mass=proposal.dropped_probability_mass,
            )

        try:
            visual_features, frame_validity = prepare_openpose_features(track)
        except ValueError as error:
            if "no frame with a valid visual observation" not in str(error):
                raise
            return CandidateGenerationResult(
                records=(), decision=AcceptanceDecision(False, "no_visual_evidence", None),
                selected_annotation_id=None,
                dropped_text_probability_mass=proposal.dropped_probability_mass,
            )
        device = next(self.video_model.parameters()).device
        visual_batch = visual_features.unsqueeze(0).to(device=device)
        video_log_probs = self.video_model(visual_batch)[0]
        # A wholly missing pose frame supplies no visual evidence and therefore
        # cannot create an extra CTC alignment slot from network bias alone.
        video_log_probs = video_log_probs[frame_validity.to(device)]

        scored = []
        diagnostics_by_tokens = {}
        for candidate in proposal.candidates:
            target = self.text_model.lexicon.encode(candidate.tokens)
            try:
                diagnostics = ctc_alignment_diagnostics(
                    video_log_probs, target, blank=0)
            except ValueError:
                # Infeasible candidates are rejected, never shortened or repaired.
                continue
            entropy_per_frame = diagnostics.path_entropy_nats / diagnostics.time_steps
            scored.append((
                candidate.tokens, candidate.text_log_probability,
                diagnostics.log_probability, entropy_per_frame,
            ))
            diagnostics_by_tokens[candidate.tokens] = diagnostics
        if not scored:
            return CandidateGenerationResult(
                records=(), decision=AcceptanceDecision(False, "all_candidates_ctc_infeasible", None),
                selected_annotation_id=None,
                dropped_text_probability_mass=proposal.dropped_probability_mass,
            )

        fused = fuse_candidate_lattice(
            scored, alpha=self.fusion.alpha, beta=self.fusion.beta,
            penalty_weight=self.fusion.alignment_entropy_penalty)
        posterior = [math.exp(item.posterior_log_probability) for item in fused]
        top = fused[0]
        top_diagnostics = diagnostics_by_tokens[top.tokens]
        features = AcceptanceFeatures(
            top_posterior=posterior[0],
            posterior_margin=(posterior[0] - posterior[1] if len(posterior) > 1
                              else posterior[0]),
            normalized_video_log_probability=(
                top.video_log_probability / top_diagnostics.time_steps),
            normalized_path_entropy=(
                top_diagnostics.path_entropy_nats / top_diagnostics.time_steps),
            mean_blank_posterior=top_diagnostics.mean_blank_posterior,
            dropped_text_mass=proposal.dropped_probability_mass,
        )
        decision = decide_acceptance(top.tokens, features, self.calibrator, self.abstention)

        feature_bytes = visual_features.contiguous().numpy().tobytes()
        decoding_hash = sha256_bytes(canonical_json_bytes({
            "abstention": asdict(self.abstention),
            "fusion": asdict(self.fusion),
            "security": asdict(self.security),
        }))
        no_prompt_hash = sha256_bytes(b"LOCAL_TOOL_FREE_TRANSFORMER_NO_PROMPT_V1")
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        provenance = CandidateProvenance(
            source_sample_id=source_sample_id,
            source_video_sha256=source_video_sha256,
            transcript_sha256=hashlib.sha256(transcript_bytes).hexdigest(),
            visual_feature_sha256=hashlib.sha256(feature_bytes).hexdigest(),
            generator_model_id=generator_model_id,
            model_weight_sha256=sha256_bytes(canonical_json_bytes({
                "text": state_dict_sha256(self.text_model),
                "video": state_dict_sha256(self.video_model),
            })),
            tokenizer_sha256=self.text_model.tokenizer.source_sha256,
            prompt_or_template_sha256=no_prompt_hash,
            decoding_config_sha256=decoding_hash,
            environment_sha256=sha256_bytes(canonical_json_bytes(runtime_environment())),
            code_revision=code_revision, random_seed=random_seed, created_at=timestamp,
        )
        records = []
        for rank, fused_candidate in enumerate(fused, start=1):
            identifier_payload = canonical_json_bytes({
                "provenance": asdict(provenance), "rank": rank,
                "tokens": list(fused_candidate.tokens),
            })
            record = WeakGlossCandidateRecord(
                annotation_id=f"pseudo-{sha256_bytes(identifier_payload)}",
                label_type=LabelType.UNREVIEWED_PSEUDO,
                review_status=ReviewStatus.UNREVIEWED,
                lexicon_id=self.text_model.lexicon.lexicon_id,
                convention_id=self.text_model.lexicon.convention_id,
                candidate_tokens=fused_candidate.tokens,
                candidate_log_score=fused_candidate.posterior_log_probability,
                candidate_rank=rank, provenance=provenance,
                limitations=(
                    "machine-only candidate; not authentic gloss",
                    "not linguistically validated",
                    "must not enter gloss_tokens without qualified human correction",
                ),
            )
            record.validate_against(self.text_model.lexicon)
            records.append(record)
        selected = records[0].annotation_id if decision.accepted else None
        return CandidateGenerationResult(
            records=tuple(records), decision=decision,
            selected_annotation_id=selected,
            dropped_text_probability_mass=proposal.dropped_probability_mass,
        )


def apply_video_intervention(track: LandmarkTrack, intervention: str,
                             *, permutation: torch.Tensor | None = None) -> LandmarkTrack:
    """Create declared falsification inputs without changing transcript data."""
    import numpy as np

    values = np.asarray(track.values).copy()
    confidence = np.asarray(track.confidence).copy()
    validity = np.asarray(track.validity_mask).copy()
    timestamps = np.asarray(track.timestamps).copy()
    if intervention == "blank":
        values.fill(0)
        confidence.fill(0)
        validity.fill(False)
    elif intervention == "reverse":
        values = values[:, ::-1].copy()
        confidence = confidence[::-1].copy()
        validity = validity[::-1].copy()
        # Retain the original increasing clock: only observation order changes.
    elif intervention == "permute_frames":
        if permutation is None or permutation.ndim != 1 \
                or len(permutation) != values.shape[1]:
            raise ValueError("frame permutation must cover the complete sequence")
        order = permutation.detach().cpu().numpy()
        if sorted(order.tolist()) != list(range(values.shape[1])):
            raise ValueError("frame permutation must be bijective")
        values = values[:, order].copy()
        confidence = confidence[order].copy()
        validity = validity[order].copy()
    else:
        raise ValueError(f"unknown video intervention: {intervention}")
    return LandmarkTrack(values=values, confidence=confidence,
                         validity_mask=validity, timestamps=timestamps)
