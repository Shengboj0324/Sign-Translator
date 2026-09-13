"""Fail-closed governance for human text/video-to-SIR supervision.

This module does not produce SIR annotations. It defines the evidence boundary
that a separately collected official-human or project-human annotation must
cross before research training may consume it. English transcripts are bound by
hash but are never copied into a gloss, lexeme, or SIR field.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence

from ..data_engineering.schema import (
    ConsentState,
    Sample,
    validate_authorization,
    validate_sample,
)
from ..data_engineering.splitting import certify_no_group_leakage
from ..grammar.sir import SIRGraph, sir_from_dict, sir_sha256, sir_to_dict


PHASE3_ANNOTATION_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


def _require_id(name: str, value: object) -> None:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a non-empty restricted identifier")


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_timestamp(name: str, value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _strict_json_loads(payload: bytes, *, max_bytes: int = 1_048_576) -> Any:
    if not isinstance(payload, bytes):
        raise TypeError("SIR payload must be immutable bytes")
    if not payload or len(payload) > max_bytes:
        raise ValueError("SIR payload is empty or exceeds the byte limit")
    text = payload.decode("utf-8", errors="strict")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")

    return json.loads(
        text, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


class SIRAnnotationOrigin(str, Enum):
    OFFICIAL_HUMAN = "official_human"
    PROJECT_HUMAN = "project_human"


class ArtifactKind(str, Enum):
    ASL_CONVENTION = "asl_convention"
    SIR_LEXICON = "sir_lexicon"
    ANNOTATION_PROTOCOL = "annotation_protocol"
    REVIEW_PROTOCOL = "review_protocol"


@dataclass(frozen=True)
class GovernedArtifact:
    """Content identity for one locally preserved governance artifact."""

    kind: ArtifactKind
    artifact_id: str
    version: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ArtifactKind):
            raise ValueError("artifact kind must be typed")
        _require_id("artifact_id", self.artifact_id)
        _require_id("artifact version", self.version)
        _require_sha256("artifact sha256", self.sha256)

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GovernedArtifact":
        required = {"kind", "artifact_id", "version", "sha256"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"artifact fields must be exactly {sorted(required)}")
        try:
            kind = ArtifactKind(value["kind"])
        except (TypeError, ValueError) as error:
            raise ValueError("unknown governed artifact kind") from error
        return cls(kind=kind, artifact_id=value["artifact_id"],
                   version=value["version"], sha256=value["sha256"])


@dataclass(frozen=True)
class SourceEvidenceBinding:
    """Immutable identity of the media/transcript pair being annotated."""

    sample_id: str
    source_recording_id: str
    signer_id_hash: str
    video_sha256: str
    transcript_sha256: str
    sample_provenance: str
    authorization_evidence_sha256: str

    def __post_init__(self) -> None:
        for name in ("sample_id", "source_recording_id", "signer_id_hash"):
            _require_id(name, getattr(self, name))
        if not isinstance(self.sample_provenance, str) or not self.sample_provenance:
            raise ValueError("sample_provenance is required")
        for name in (
            "video_sha256", "transcript_sha256", "authorization_evidence_sha256",
        ):
            _require_sha256(name, getattr(self, name))

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "source_recording_id": self.source_recording_id,
            "signer_id_hash": self.signer_id_hash,
            "video_sha256": self.video_sha256,
            "transcript_sha256": self.transcript_sha256,
            "sample_provenance": self.sample_provenance,
            "authorization_evidence_sha256": self.authorization_evidence_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceEvidenceBinding":
        required = {
            "sample_id", "source_recording_id", "signer_id_hash", "video_sha256",
            "transcript_sha256", "sample_provenance",
            "authorization_evidence_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"source binding fields must be exactly {sorted(required)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class IndependentReviewAttestation:
    """Human-review evidence; self-review and non-video review are forbidden."""

    annotator_pseudonym: str
    reviewer_pseudonym: str
    annotator_qualified_asl: bool
    reviewer_qualified_asl: bool
    annotator_viewed_source_video: bool
    reviewer_viewed_source_video: bool
    annotation_protocol: GovernedArtifact
    review_protocol: GovernedArtifact
    subject_annotation_id: str
    reviewed_sir_sha256: str
    reviewed_video_sha256: str
    reviewed_transcript_sha256: str
    reviewed_convention_sha256: str
    reviewed_lexicon_sha256: str
    annotator_qualification_evidence_sha256: str
    reviewer_qualification_evidence_sha256: str
    independence_evidence_sha256: str
    attestation_sha256: str
    approved_at: str

    def __post_init__(self) -> None:
        _require_id("annotator_pseudonym", self.annotator_pseudonym)
        _require_id("reviewer_pseudonym", self.reviewer_pseudonym)
        if self.annotator_pseudonym == self.reviewer_pseudonym:
            raise ValueError("annotator and independent reviewer must be distinct")
        flags = (
            self.annotator_qualified_asl, self.reviewer_qualified_asl,
            self.annotator_viewed_source_video, self.reviewer_viewed_source_video,
        )
        if any(value is not True for value in flags):
            raise ValueError("approval requires qualified annotator/reviewer video review")
        if self.annotation_protocol.kind is not ArtifactKind.ANNOTATION_PROTOCOL:
            raise ValueError("annotation_protocol has the wrong artifact kind")
        if self.review_protocol.kind is not ArtifactKind.REVIEW_PROTOCOL:
            raise ValueError("review_protocol has the wrong artifact kind")
        _require_id("subject_annotation_id", self.subject_annotation_id)
        for name in (
            "reviewed_sir_sha256", "reviewed_video_sha256",
            "reviewed_transcript_sha256", "reviewed_convention_sha256",
            "reviewed_lexicon_sha256", "annotator_qualification_evidence_sha256",
            "reviewer_qualification_evidence_sha256", "independence_evidence_sha256",
            "attestation_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_timestamp("approved_at", self.approved_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotator_pseudonym": self.annotator_pseudonym,
            "reviewer_pseudonym": self.reviewer_pseudonym,
            "annotator_qualified_asl": self.annotator_qualified_asl,
            "reviewer_qualified_asl": self.reviewer_qualified_asl,
            "annotator_viewed_source_video": self.annotator_viewed_source_video,
            "reviewer_viewed_source_video": self.reviewer_viewed_source_video,
            "annotation_protocol": self.annotation_protocol.to_dict(),
            "review_protocol": self.review_protocol.to_dict(),
            "subject_annotation_id": self.subject_annotation_id,
            "reviewed_sir_sha256": self.reviewed_sir_sha256,
            "reviewed_video_sha256": self.reviewed_video_sha256,
            "reviewed_transcript_sha256": self.reviewed_transcript_sha256,
            "reviewed_convention_sha256": self.reviewed_convention_sha256,
            "reviewed_lexicon_sha256": self.reviewed_lexicon_sha256,
            "annotator_qualification_evidence_sha256": (
                self.annotator_qualification_evidence_sha256),
            "reviewer_qualification_evidence_sha256": (
                self.reviewer_qualification_evidence_sha256),
            "independence_evidence_sha256": self.independence_evidence_sha256,
            "attestation_sha256": self.attestation_sha256,
            "approved_at": self.approved_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IndependentReviewAttestation":
        required = {
            "annotator_pseudonym", "reviewer_pseudonym", "annotator_qualified_asl",
            "reviewer_qualified_asl", "annotator_viewed_source_video",
            "reviewer_viewed_source_video", "annotation_protocol",
            "review_protocol", "subject_annotation_id", "reviewed_sir_sha256",
            "reviewed_video_sha256", "reviewed_transcript_sha256",
            "reviewed_convention_sha256", "reviewed_lexicon_sha256",
            "annotator_qualification_evidence_sha256",
            "reviewer_qualification_evidence_sha256",
            "independence_evidence_sha256", "attestation_sha256", "approved_at",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"review fields must be exactly {sorted(required)}")
        for name in (
            "annotator_qualified_asl", "reviewer_qualified_asl",
            "annotator_viewed_source_video", "reviewer_viewed_source_video",
        ):
            if not isinstance(value[name], bool):
                raise ValueError(f"{name} must be boolean")
        annotation_protocol = value["annotation_protocol"]
        review_protocol = value["review_protocol"]
        if not isinstance(annotation_protocol, Mapping) \
                or not isinstance(review_protocol, Mapping):
            raise ValueError("review protocols must be artifact objects")
        return cls(
            annotator_pseudonym=value["annotator_pseudonym"],
            reviewer_pseudonym=value["reviewer_pseudonym"],
            annotator_qualified_asl=value["annotator_qualified_asl"],
            reviewer_qualified_asl=value["reviewer_qualified_asl"],
            annotator_viewed_source_video=value["annotator_viewed_source_video"],
            reviewer_viewed_source_video=value["reviewer_viewed_source_video"],
            annotation_protocol=GovernedArtifact.from_dict(annotation_protocol),
            review_protocol=GovernedArtifact.from_dict(review_protocol),
            subject_annotation_id=value["subject_annotation_id"],
            reviewed_sir_sha256=value["reviewed_sir_sha256"],
            reviewed_video_sha256=value["reviewed_video_sha256"],
            reviewed_transcript_sha256=value["reviewed_transcript_sha256"],
            reviewed_convention_sha256=value["reviewed_convention_sha256"],
            reviewed_lexicon_sha256=value["reviewed_lexicon_sha256"],
            annotator_qualification_evidence_sha256=(
                value["annotator_qualification_evidence_sha256"]),
            reviewer_qualification_evidence_sha256=(
                value["reviewer_qualification_evidence_sha256"]),
            independence_evidence_sha256=value["independence_evidence_sha256"],
            attestation_sha256=value["attestation_sha256"],
            approved_at=value["approved_at"],
        )


@dataclass(frozen=True)
class GovernedSIRAnnotation:
    """An immutable, independently approved human SIR annotation."""

    schema_version: int
    annotation_id: str
    origin: SIRAnnotationOrigin
    source: SourceEvidenceBinding
    convention: GovernedArtifact
    lexicon: GovernedArtifact
    lexicon_convention_sha256: str
    sir_payload: bytes
    sir_payload_sha256: str
    review: IndependentReviewAttestation
    created_at: str
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version \
                != PHASE3_ANNOTATION_SCHEMA_VERSION:
            raise ValueError("unsupported Phase 3 annotation schema")
        _require_id("annotation_id", self.annotation_id)
        if not isinstance(self.origin, SIRAnnotationOrigin):
            raise ValueError("annotation origin must be official-human or project-human")
        if self.convention.kind is not ArtifactKind.ASL_CONVENTION:
            raise ValueError("convention has the wrong artifact kind")
        if self.lexicon.kind is not ArtifactKind.SIR_LEXICON:
            raise ValueError("lexicon has the wrong artifact kind")
        _require_sha256("lexicon_convention_sha256", self.lexicon_convention_sha256)
        if self.lexicon_convention_sha256 != self.convention.sha256:
            raise ValueError("lexicon is not bound to the declared ASL convention")
        _require_sha256("sir_payload_sha256", self.sir_payload_sha256)
        graph = self.graph()
        if not graph.events or not graph.manual_events():
            raise ValueError("approved SIR must contain at least one manual event")
        if sir_sha256(graph) != self.sir_payload_sha256:
            raise ValueError("SIR payload hash mismatch")
        if self.sir_payload != _canonical_json_bytes(sir_to_dict(graph)):
            raise ValueError("SIR payload is not canonical JSON")
        review_bindings = (
            (self.review.subject_annotation_id, self.annotation_id),
            (self.review.reviewed_sir_sha256, self.sir_payload_sha256),
            (self.review.reviewed_video_sha256, self.source.video_sha256),
            (self.review.reviewed_transcript_sha256, self.source.transcript_sha256),
            (self.review.reviewed_convention_sha256, self.convention.sha256),
            (self.review.reviewed_lexicon_sha256, self.lexicon.sha256),
        )
        if any(reviewed != declared for reviewed, declared in review_bindings):
            raise ValueError("review attestation is not bound to the declared annotation")
        created = _require_timestamp("created_at", self.created_at)
        approved = _require_timestamp("approved_at", self.review.approved_at)
        if approved < created:
            raise ValueError("review approval cannot precede annotation creation")
        if not isinstance(self.limitations, tuple) or any(
                not isinstance(item, str) or not item.strip() for item in self.limitations):
            raise ValueError("limitations must be a tuple of non-empty strings")

    @classmethod
    def create(
        cls,
        *,
        annotation_id: str,
        origin: SIRAnnotationOrigin,
        source: SourceEvidenceBinding,
        convention: GovernedArtifact,
        lexicon: GovernedArtifact,
        lexicon_convention_sha256: str,
        graph: SIRGraph,
        review: IndependentReviewAttestation,
        created_at: str,
        limitations: Sequence[str] = (),
    ) -> "GovernedSIRAnnotation":
        payload = _canonical_json_bytes(sir_to_dict(graph))
        return cls(
            schema_version=PHASE3_ANNOTATION_SCHEMA_VERSION,
            annotation_id=annotation_id,
            origin=origin,
            source=source,
            convention=convention,
            lexicon=lexicon,
            lexicon_convention_sha256=lexicon_convention_sha256,
            sir_payload=payload,
            sir_payload_sha256=sir_sha256(graph),
            review=review,
            created_at=created_at,
            limitations=tuple(limitations),
        )

    def graph(self) -> SIRGraph:
        value = _strict_json_loads(self.sir_payload)
        if not isinstance(value, Mapping):
            raise ValueError("SIR payload root must be an object")
        return sir_from_dict(value)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "annotation_id": self.annotation_id,
            "origin": self.origin.value,
            "source": self.source.to_dict(),
            "convention": self.convention.to_dict(),
            "lexicon": self.lexicon.to_dict(),
            "lexicon_convention_sha256": self.lexicon_convention_sha256,
            "sir": sir_to_dict(self.graph()),
            "sir_payload_sha256": self.sir_payload_sha256,
            "review": self.review.to_dict(),
            "created_at": self.created_at,
            "limitations": list(self.limitations),
        }

    def content_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_manifest())).hexdigest()

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "GovernedSIRAnnotation":
        required = {
            "schema_version", "annotation_id", "origin", "source", "convention",
            "lexicon", "lexicon_convention_sha256", "sir", "sir_payload_sha256",
            "review", "created_at", "limitations",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"annotation fields must be exactly {sorted(required)}")
        for name in ("source", "convention", "lexicon", "sir", "review"):
            if not isinstance(value[name], Mapping):
                raise ValueError(f"{name} must be an object")
        if not isinstance(value["limitations"], list):
            raise ValueError("limitations must be a list")
        try:
            origin = SIRAnnotationOrigin(value["origin"])
        except (TypeError, ValueError) as error:
            raise ValueError("unknown SIR annotation origin") from error
        sir_payload = _canonical_json_bytes(value["sir"])
        return cls(
            schema_version=value["schema_version"],
            annotation_id=value["annotation_id"],
            origin=origin,
            source=SourceEvidenceBinding.from_dict(value["source"]),
            convention=GovernedArtifact.from_dict(value["convention"]),
            lexicon=GovernedArtifact.from_dict(value["lexicon"]),
            lexicon_convention_sha256=value["lexicon_convention_sha256"],
            sir_payload=sir_payload,
            sir_payload_sha256=value["sir_payload_sha256"],
            review=IndependentReviewAttestation.from_dict(value["review"]),
            created_at=value["created_at"],
            limitations=tuple(value["limitations"]),
        )


@dataclass(frozen=True)
class SupervisionBatchCertificate:
    """Narrow certificate: governed records are trainable for a declared use."""

    approved_for_research_training: bool
    annotation_count: int
    signer_count: int
    source_recording_count: int
    violations: tuple[str, ...]


def validate_annotation_against_sample(
    annotation: GovernedSIRAnnotation,
    sample: Sample,
    *,
    observed_video_sha256: str,
    observed_transcript_sha256: str,
) -> tuple[str, ...]:
    """Validate source identity and research-training authority without I/O."""
    violations: list[str] = []
    _require_sha256("observed_video_sha256", observed_video_sha256)
    _require_sha256("observed_transcript_sha256", observed_transcript_sha256)
    violations.extend(f"sample_schema:{reason}" for reason in validate_sample(sample))
    if sample.target_language != "ASL":
        violations.append("target_language_not_asl")
    source = annotation.source
    comparisons = (
        (source.sample_id, sample.sample_id, "sample_id_mismatch"),
        (source.source_recording_id, sample.source_id, "source_recording_mismatch"),
        (source.signer_id_hash, sample.signer_id_hash, "signer_id_mismatch"),
        (source.sample_provenance, sample.provenance, "sample_provenance_mismatch"),
        (source.video_sha256, observed_video_sha256, "video_sha256_mismatch"),
        (source.transcript_sha256, observed_transcript_sha256,
         "transcript_sha256_mismatch"),
    )
    violations.extend(reason for actual, expected, reason in comparisons
                      if actual != expected)
    if sample.consent is ConsentState.WITHDRAWN:
        violations.append("consent_withdrawn")
    if sample.authorization is None:
        violations.append("missing_authorization")
    else:
        if source.authorization_evidence_sha256 \
                != sample.authorization.evidence_sha256.lower():
            violations.append("authorization_evidence_mismatch")
        violations.extend(validate_authorization(
            sample.authorization, sample.consent, sample.intended_use,
            requested_actions=("create_derivatives", "model_training"),
        ))
    return tuple(dict.fromkeys(violations))


def certify_supervision_batch(
    annotations: Sequence[GovernedSIRAnnotation],
    samples: Mapping[str, Sample],
    observed_video_sha256: Mapping[str, str],
    observed_transcript_sha256: Mapping[str, str],
) -> SupervisionBatchCertificate:
    """Certify a complete batch, including declared signer/source split isolation.

    The split is taken from each canonical ``Sample``. No split is inferred or
    repaired here. This certificate does not establish linguistic validity of
    the SIR schema or industrial deployment readiness.
    """
    violations: list[str] = []
    if isinstance(annotations, (str, bytes)) or not isinstance(annotations, Sequence):
        raise TypeError("annotations must be a sequence")
    if not annotations:
        return SupervisionBatchCertificate(False, 0, 0, 0,
                                           ("no_governed_annotations",))
    if any(not isinstance(item, GovernedSIRAnnotation) for item in annotations):
        return SupervisionBatchCertificate(
            False, len(annotations), 0, 0, ("invalid_annotation_type",))
    ids = [item.annotation_id for item in annotations]
    if len(ids) != len(set(ids)):
        violations.append("duplicate_annotation_id")
    sample_ids = [item.source.sample_id for item in annotations]
    if len(sample_ids) != len(set(sample_ids)):
        violations.append("multiple_annotations_for_sample_require_adjudication")
    conventions = {
        (item.convention.artifact_id, item.convention.version, item.convention.sha256)
        for item in annotations
    }
    lexicons = {
        (item.lexicon.artifact_id, item.lexicon.version, item.lexicon.sha256)
        for item in annotations
    }
    if len(conventions) != 1:
        violations.append("mixed_asl_conventions")
    if len(lexicons) != 1:
        violations.append("mixed_sir_lexicons")

    ordered_samples: list[Sample] = []
    for annotation in annotations:
        sample_id = annotation.source.sample_id
        sample = samples.get(sample_id)
        video_digest = observed_video_sha256.get(sample_id)
        transcript_digest = observed_transcript_sha256.get(sample_id)
        if sample is None:
            violations.append(f"{sample_id}:missing_sample")
            continue
        if video_digest is None:
            violations.append(f"{sample_id}:missing_observed_video_sha256")
            continue
        if transcript_digest is None:
            violations.append(f"{sample_id}:missing_observed_transcript_sha256")
            continue
        ordered_samples.append(sample)
        violations.extend(
            f"{sample_id}:{reason}" for reason in validate_annotation_against_sample(
                annotation, sample,
                observed_video_sha256=video_digest,
                observed_transcript_sha256=transcript_digest,
            )
        )

    if len(ordered_samples) == len(annotations):
        assignment = {index: sample.split for index, sample in enumerate(ordered_samples)}
        try:
            leakage = certify_no_group_leakage(ordered_samples, assignment)
        except ValueError as error:
            violations.append(f"invalid_split_assignment:{error}")
        else:
            violations.extend(
                f"signer_split_leakage:{value}" for value in leakage.offending_signers)
            violations.extend(
                f"source_split_leakage:{value}" for value in leakage.offending_sources)

    unique_violations = tuple(dict.fromkeys(violations))
    return SupervisionBatchCertificate(
        approved_for_research_training=not unique_violations,
        annotation_count=len(annotations),
        signer_count=len({item.source.signer_id_hash for item in annotations}),
        source_recording_count=len({
            item.source.source_recording_id for item in annotations}),
        violations=unique_violations,
    )


PHASE3_UNSOLVED_BLOCKERS = (
    "qualified_text_video_to_sir_supervision_absent",
    "direct_paired_learning_falsification_not_passed",
    "human_approved_lexical_motion_library_absent",
    "canonical_phase2_multichannel_state_absent",
    "independent_qualified_asl_validation_absent",
    "commercial_training_and_deployment_rights_absent",
)


@dataclass(frozen=True)
class Phase3ReadinessReport:
    research_exit_approved: bool
    industrial_path_approved: bool
    unsolved_blockers: tuple[str, ...]


def current_phase3_readiness() -> Phase3ReadinessReport:
    """Return the checked-in fail-closed baseline; no evidence is synthesized."""
    return Phase3ReadinessReport(
        research_exit_approved=False,
        industrial_path_approved=False,
        unsolved_blockers=PHASE3_UNSOLVED_BLOCKERS,
    )


__all__ = [
    "PHASE3_ANNOTATION_SCHEMA_VERSION", "SIRAnnotationOrigin", "ArtifactKind",
    "GovernedArtifact", "SourceEvidenceBinding", "IndependentReviewAttestation",
    "GovernedSIRAnnotation", "SupervisionBatchCertificate",
    "validate_annotation_against_sample", "certify_supervision_batch",
    "PHASE3_UNSOLVED_BLOCKERS", "Phase3ReadinessReport",
    "current_phase3_readiness",
]
