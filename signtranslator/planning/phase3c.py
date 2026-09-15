"""Phase 3C evidence integration and lexical-motion governance.

Phase 3C is an evidence gate, not a label generator.  This module provides the
software contracts that can be completed before the external linguistic,
motion, rights, and review artifacts exist:

* exact, independently verified identities for external evidence;
* a versioned library of human-approved lexical motion in the canonical
  Phase-2 representation only;
* explicit abstention for absent or ambiguous lexical coverage; and
* a readiness assessment that distinguishes implemented software from work
  which cannot truthfully be executed before its real artifacts arrive.

No English string, OpenPose track, synthetic motion, or model output is accepted
as a human-approved lexical-motion entry by this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence

from ..pretraining.dependence import VideoDependenceCertificate
from .supervision import SupervisionBatchCertificate


PHASE3C_SCHEMA_VERSION = 1
CANONICAL_PHASE2_REPRESENTATION = "canonical_phase2_multichannel_v1"
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_LIBRARY_ENTRIES = 100_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


def _require_id(name: str, value: object) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a non-empty restricted identifier")
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


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
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _strict_json_loads(payload: bytes, *, max_bytes: int) -> Any:
    if not isinstance(payload, bytes):
        raise TypeError("Phase 3C payload must be immutable bytes")
    if not payload or len(payload) > max_bytes:
        raise ValueError("Phase 3C payload is empty or exceeds its byte limit")
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

    try:
        return json.loads(
            text,
            object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed Phase 3C JSON: {error}") from error


class EvidenceRole(str, Enum):
    """External evidence roles which are deliberately non-substitutable."""

    CANONICAL_PHASE2_STATE = "canonical_phase2_state"
    QUALIFIED_ASL_VALIDATION = "qualified_asl_validation"
    RESEARCH_TRAINING_RIGHTS = "research_training_rights"
    COMMERCIAL_DEPLOYMENT_RIGHTS = "commercial_deployment_rights"


@dataclass(frozen=True)
class ExternalEvidenceAttestation:
    """Identity and independent verification record for one external artifact.

    This verifies internal consistency and content binding.  It cannot prove
    that a human's qualification or a legal claim is substantively correct.
    """

    schema_version: int
    evidence_id: str
    role: EvidenceRole
    artifact_sha256: str
    issuer_pseudonym: str
    verifier_pseudonym: str
    verifier_qualification_evidence_sha256: str
    verification_protocol_sha256: str
    attestation_sha256: str
    issued_at: str
    verified_at: str

    def __post_init__(self) -> None:
        if (not isinstance(self.schema_version, int)
                or isinstance(self.schema_version, bool)
                or self.schema_version != PHASE3C_SCHEMA_VERSION):
            raise ValueError("unsupported Phase 3C evidence schema")
        if not isinstance(self.role, EvidenceRole):
            raise ValueError("external evidence role must be typed")
        for name in ("evidence_id", "issuer_pseudonym", "verifier_pseudonym"):
            _require_id(name, getattr(self, name))
        if self.issuer_pseudonym == self.verifier_pseudonym:
            raise ValueError("external evidence issuer and verifier must be distinct")
        for name in (
            "artifact_sha256",
            "verifier_qualification_evidence_sha256",
            "verification_protocol_sha256",
            "attestation_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        evidence_digests = (
            self.artifact_sha256,
            self.verifier_qualification_evidence_sha256,
            self.verification_protocol_sha256,
            self.attestation_sha256,
        )
        if len(set(evidence_digests)) != len(evidence_digests):
            raise ValueError("external evidence artifact roles must be distinct")
        issued = _require_timestamp("issued_at", self.issued_at)
        verified = _require_timestamp("verified_at", self.verified_at)
        if verified <= issued:
            raise ValueError("external evidence verification must follow issuance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "role": self.role.value,
            "artifact_sha256": self.artifact_sha256,
            "issuer_pseudonym": self.issuer_pseudonym,
            "verifier_pseudonym": self.verifier_pseudonym,
            "verifier_qualification_evidence_sha256": (
                self.verifier_qualification_evidence_sha256),
            "verification_protocol_sha256": self.verification_protocol_sha256,
            "attestation_sha256": self.attestation_sha256,
            "issued_at": self.issued_at,
            "verified_at": self.verified_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExternalEvidenceAttestation":
        required = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                f"external evidence fields must be exactly {sorted(required)}")
        try:
            role = EvidenceRole(value["role"])
        except (TypeError, ValueError) as error:
            raise ValueError("unknown external evidence role") from error
        return cls(
            schema_version=value["schema_version"],
            evidence_id=value["evidence_id"],
            role=role,
            artifact_sha256=value["artifact_sha256"],
            issuer_pseudonym=value["issuer_pseudonym"],
            verifier_pseudonym=value["verifier_pseudonym"],
            verifier_qualification_evidence_sha256=value[
                "verifier_qualification_evidence_sha256"],
            verification_protocol_sha256=value["verification_protocol_sha256"],
            attestation_sha256=value["attestation_sha256"],
            issued_at=value["issued_at"],
            verified_at=value["verified_at"],
        )

    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())


def load_external_evidence_attestation(
    payload: bytes,
    *,
    max_bytes: int = 65_536,
) -> ExternalEvidenceAttestation:
    """Load one exact evidence attestation without accepting JSON aliases."""
    if (not isinstance(max_bytes, int) or isinstance(max_bytes, bool)
            or not 1 <= max_bytes <= 65_536):
        raise ValueError("max_bytes must be a positive bounded integer")
    value = _strict_json_loads(payload, max_bytes=max_bytes)
    if not isinstance(value, Mapping):
        raise ValueError("external evidence root must be an object")
    evidence = ExternalEvidenceAttestation.from_dict(value)
    if evidence.canonical_bytes() != payload:
        raise ValueError("external evidence bytes are not canonical")
    return evidence


@dataclass(frozen=True)
class LexicalMotionReview:
    """Independent human review bound to one exact meaning and motion pair."""

    creator_pseudonym: str
    reviewer_pseudonym: str
    creator_qualified_asl: bool
    reviewer_qualified_asl: bool
    reviewer_viewed_motion: bool
    reviewed_lexeme_id: int
    reviewed_asl_convention_sha256: str
    reviewed_sir_lexicon_sha256: str
    reviewed_phase2_schema_sha256: str
    reviewed_motion_manifest_sha256: str
    reviewed_motion_payload_sha256: str
    meaning_review_sha256: str
    articulation_review_sha256: str
    creator_qualification_evidence_sha256: str
    reviewer_qualification_evidence_sha256: str
    independence_evidence_sha256: str
    review_protocol_sha256: str
    approved_at: str

    def __post_init__(self) -> None:
        for name in ("creator_pseudonym", "reviewer_pseudonym"):
            _require_id(name, getattr(self, name))
        if self.creator_pseudonym == self.reviewer_pseudonym:
            raise ValueError("lexical-motion creator and reviewer must be distinct")
        if any(value is not True for value in (
                self.creator_qualified_asl,
                self.reviewer_qualified_asl,
                self.reviewer_viewed_motion,
        )):
            raise ValueError(
                "lexical motion requires qualified creators and direct motion review")
        if (not isinstance(self.reviewed_lexeme_id, int)
                or isinstance(self.reviewed_lexeme_id, bool)
                or self.reviewed_lexeme_id < 0):
            raise ValueError("reviewed_lexeme_id must be a non-negative integer")
        for name in (
            "reviewed_asl_convention_sha256",
            "reviewed_sir_lexicon_sha256",
            "reviewed_phase2_schema_sha256",
            "reviewed_motion_manifest_sha256",
            "reviewed_motion_payload_sha256",
            "meaning_review_sha256",
            "articulation_review_sha256",
            "creator_qualification_evidence_sha256",
            "reviewer_qualification_evidence_sha256",
            "independence_evidence_sha256",
            "review_protocol_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        review_digests = (
            self.reviewed_asl_convention_sha256,
            self.reviewed_sir_lexicon_sha256,
            self.reviewed_phase2_schema_sha256,
            self.reviewed_motion_manifest_sha256,
            self.reviewed_motion_payload_sha256,
            self.meaning_review_sha256,
            self.articulation_review_sha256,
            self.creator_qualification_evidence_sha256,
            self.reviewer_qualification_evidence_sha256,
            self.independence_evidence_sha256,
            self.review_protocol_sha256,
        )
        if len(set(review_digests)) != len(review_digests):
            raise ValueError("lexical-motion review artifact roles must be distinct")
        _require_timestamp("approved_at", self.approved_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LexicalMotionReview":
        required = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                f"lexical-motion review fields must be exactly {sorted(required)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class LexicalMotionEntry:
    """One reviewed lexical form bound to canonical Phase-2 motion bytes."""

    schema_version: int
    entry_id: str
    lexeme_id: int
    form_id: str
    asl_convention_sha256: str
    sir_lexicon_sha256: str
    representation: str
    phase2_schema_sha256: str
    motion_manifest_sha256: str
    motion_payload_sha256: str
    source_recording_id: str
    signer_id_hash: str
    authorization_evidence_sha256: str
    permitted_actions: tuple[str, ...]
    review: LexicalMotionReview

    def __post_init__(self) -> None:
        if (not isinstance(self.schema_version, int)
                or isinstance(self.schema_version, bool)
                or self.schema_version != PHASE3C_SCHEMA_VERSION):
            raise ValueError("unsupported lexical-motion entry schema")
        for name in ("entry_id", "form_id", "source_recording_id", "signer_id_hash"):
            _require_id(name, getattr(self, name))
        if (not isinstance(self.lexeme_id, int) or isinstance(self.lexeme_id, bool)
                or self.lexeme_id < 0):
            raise ValueError("lexeme_id must be a non-negative integer")
        if self.representation != CANONICAL_PHASE2_REPRESENTATION:
            raise ValueError(
                "lexical motion must use the canonical Phase-2 multichannel state")
        for name in (
            "asl_convention_sha256",
            "sir_lexicon_sha256",
            "phase2_schema_sha256",
            "motion_manifest_sha256",
            "motion_payload_sha256",
            "authorization_evidence_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        entry_digests = (
            self.asl_convention_sha256,
            self.sir_lexicon_sha256,
            self.phase2_schema_sha256,
            self.motion_manifest_sha256,
            self.motion_payload_sha256,
            self.authorization_evidence_sha256,
        )
        if len(set(entry_digests)) != len(entry_digests):
            raise ValueError("lexical-motion entry artifact roles must be distinct")
        allowed = {"research_training", "commercial_deployment"}
        if (not isinstance(self.permitted_actions, tuple)
                or not self.permitted_actions
                or tuple(sorted(self.permitted_actions)) != self.permitted_actions
                or len(set(self.permitted_actions)) != len(self.permitted_actions)
                or any(action not in allowed for action in self.permitted_actions)):
            raise ValueError("permitted_actions must be a canonical non-empty scope")
        if not isinstance(self.review, LexicalMotionReview):
            raise ValueError("lexical motion requires a typed independent review")
        review_bindings = (
            (self.review.reviewed_lexeme_id, self.lexeme_id),
            (self.review.reviewed_asl_convention_sha256,
             self.asl_convention_sha256),
            (self.review.reviewed_sir_lexicon_sha256, self.sir_lexicon_sha256),
            (self.review.reviewed_phase2_schema_sha256, self.phase2_schema_sha256),
            (self.review.reviewed_motion_manifest_sha256,
             self.motion_manifest_sha256),
            (self.review.reviewed_motion_payload_sha256,
             self.motion_payload_sha256),
        )
        if any(reviewed != declared for reviewed, declared in review_bindings):
            raise ValueError("lexical-motion review is not bound to the entry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entry_id": self.entry_id,
            "lexeme_id": self.lexeme_id,
            "form_id": self.form_id,
            "asl_convention_sha256": self.asl_convention_sha256,
            "sir_lexicon_sha256": self.sir_lexicon_sha256,
            "representation": self.representation,
            "phase2_schema_sha256": self.phase2_schema_sha256,
            "motion_manifest_sha256": self.motion_manifest_sha256,
            "motion_payload_sha256": self.motion_payload_sha256,
            "source_recording_id": self.source_recording_id,
            "signer_id_hash": self.signer_id_hash,
            "authorization_evidence_sha256": self.authorization_evidence_sha256,
            "permitted_actions": list(self.permitted_actions),
            "review": self.review.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LexicalMotionEntry":
        required = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                f"lexical-motion entry fields must be exactly {sorted(required)}")
        if not isinstance(value["permitted_actions"], list) \
                or not isinstance(value["review"], Mapping):
            raise ValueError("lexical-motion entry collections have invalid types")
        return cls(
            schema_version=value["schema_version"],
            entry_id=value["entry_id"],
            lexeme_id=value["lexeme_id"],
            form_id=value["form_id"],
            asl_convention_sha256=value["asl_convention_sha256"],
            sir_lexicon_sha256=value["sir_lexicon_sha256"],
            representation=value["representation"],
            phase2_schema_sha256=value["phase2_schema_sha256"],
            motion_manifest_sha256=value["motion_manifest_sha256"],
            motion_payload_sha256=value["motion_payload_sha256"],
            source_recording_id=value["source_recording_id"],
            signer_id_hash=value["signer_id_hash"],
            authorization_evidence_sha256=value["authorization_evidence_sha256"],
            permitted_actions=tuple(value["permitted_actions"]),
            review=LexicalMotionReview.from_dict(value["review"]),
        )

    def content_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()


class MotionLookupStatus(str, Enum):
    RESOLVED = "resolved"
    UNKNOWN_LEXEME = "unknown_lexeme"
    AMBIGUOUS_FORM = "ambiguous_form"
    ACTION_NOT_AUTHORIZED = "action_not_authorized"


@dataclass(frozen=True)
class MotionLookupResult:
    status: MotionLookupStatus
    lexeme_id: int
    requested_action: str
    candidate_entry_ids: tuple[str, ...]
    selected_entry_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, MotionLookupStatus):
            raise ValueError("motion lookup status must be typed")
        if (not isinstance(self.lexeme_id, int) or isinstance(self.lexeme_id, bool)
                or self.lexeme_id < 0):
            raise ValueError("motion lookup lexeme_id must be non-negative")
        if self.requested_action not in {
                "research_training", "commercial_deployment"}:
            raise ValueError("motion lookup action is invalid")
        if (not isinstance(self.candidate_entry_ids, tuple)
                or len(set(self.candidate_entry_ids))
                != len(self.candidate_entry_ids)):
            raise ValueError("motion lookup candidates must be a unique tuple")
        for entry_id in self.candidate_entry_ids:
            _require_id("motion lookup candidate", entry_id)
        if self.status is MotionLookupStatus.RESOLVED:
            if (len(self.candidate_entry_ids) != 1
                    or self.selected_entry_id != self.candidate_entry_ids[0]):
                raise ValueError("resolved motion lookup must select its sole candidate")
        elif self.selected_entry_id is not None:
            raise ValueError("abstained motion lookup cannot select an entry")
        if (self.status is MotionLookupStatus.UNKNOWN_LEXEME
                and self.candidate_entry_ids):
            raise ValueError("unknown lexeme lookup cannot contain candidates")

    @property
    def abstained(self) -> bool:
        return self.status is not MotionLookupStatus.RESOLVED


@dataclass(frozen=True)
class LexicalMotionLibrary:
    """Non-empty, versioned collection of independently reviewed motion forms."""

    schema_version: int
    library_id: str
    version: str
    asl_convention_sha256: str
    sir_lexicon_sha256: str
    phase2_schema_sha256: str
    entries: tuple[LexicalMotionEntry, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.schema_version, int)
                or isinstance(self.schema_version, bool)
                or self.schema_version != PHASE3C_SCHEMA_VERSION):
            raise ValueError("unsupported lexical-motion library schema")
        _require_id("library_id", self.library_id)
        _require_id("library version", self.version)
        for name in (
            "asl_convention_sha256", "sir_lexicon_sha256", "phase2_schema_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (not isinstance(self.entries, tuple) or not self.entries
                or len(self.entries) > _MAX_LIBRARY_ENTRIES
                or any(not isinstance(entry, LexicalMotionEntry)
                       for entry in self.entries)):
            raise ValueError("lexical-motion library must contain bounded typed entries")
        if tuple(sorted(
                self.entries, key=lambda item: (item.lexeme_id, item.form_id, item.entry_id)
        )) != self.entries:
            raise ValueError("lexical-motion entries must be canonically ordered")
        if len({entry.entry_id for entry in self.entries}) != len(self.entries):
            raise ValueError("lexical-motion entry identifiers must be unique")
        if len({(entry.lexeme_id, entry.form_id) for entry in self.entries}) \
                != len(self.entries):
            raise ValueError("lexeme and form pairs must be unique")
        if len({entry.motion_payload_sha256 for entry in self.entries}) \
                != len(self.entries):
            raise ValueError("one motion payload cannot be registered as multiple forms")
        for entry in self.entries:
            if (
                entry.asl_convention_sha256 != self.asl_convention_sha256
                or entry.sir_lexicon_sha256 != self.sir_lexicon_sha256
                or entry.phase2_schema_sha256 != self.phase2_schema_sha256
            ):
                raise ValueError("lexical-motion entry uses another governed schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "phase3c_lexical_motion_library",
            "library_id": self.library_id,
            "version": self.version,
            "asl_convention_sha256": self.asl_convention_sha256,
            "sir_lexicon_sha256": self.sir_lexicon_sha256,
            "phase2_schema_sha256": self.phase2_schema_sha256,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())

    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def lookup(self, lexeme_id: int, requested_action: str) -> MotionLookupResult:
        if (not isinstance(lexeme_id, int) or isinstance(lexeme_id, bool)
                or lexeme_id < 0):
            raise ValueError("lexeme_id must be a non-negative integer")
        if requested_action not in {"research_training", "commercial_deployment"}:
            raise ValueError("unknown lexical-motion action")
        candidates = tuple(
            entry for entry in self.entries if entry.lexeme_id == lexeme_id
        )
        candidate_ids = tuple(entry.entry_id for entry in candidates)
        if not candidates:
            return MotionLookupResult(
                MotionLookupStatus.UNKNOWN_LEXEME,
                lexeme_id,
                requested_action,
                (),
                None,
            )
        authorized = tuple(
            entry for entry in candidates if requested_action in entry.permitted_actions
        )
        if not authorized:
            return MotionLookupResult(
                MotionLookupStatus.ACTION_NOT_AUTHORIZED,
                lexeme_id,
                requested_action,
                candidate_ids,
                None,
            )
        if len(authorized) != 1:
            return MotionLookupResult(
                MotionLookupStatus.AMBIGUOUS_FORM,
                lexeme_id,
                requested_action,
                tuple(entry.entry_id for entry in authorized),
                None,
            )
        return MotionLookupResult(
            MotionLookupStatus.RESOLVED,
            lexeme_id,
            requested_action,
            (authorized[0].entry_id,),
            authorized[0].entry_id,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LexicalMotionLibrary":
        required = {
            "schema_version", "kind", "library_id", "version",
            "asl_convention_sha256", "sir_lexicon_sha256",
            "phase2_schema_sha256", "entries",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                f"lexical-motion library fields must be exactly {sorted(required)}")
        if value["kind"] != "phase3c_lexical_motion_library":
            raise ValueError("lexical-motion library kind is invalid")
        if not isinstance(value["entries"], list):
            raise ValueError("lexical-motion library entries must be a list")
        if len(value["entries"]) > _MAX_LIBRARY_ENTRIES:
            raise ValueError("lexical-motion library exceeds its entry limit")
        if any(not isinstance(item, Mapping) for item in value["entries"]):
            raise ValueError("lexical-motion library entry must be an object")
        return cls(
            schema_version=value["schema_version"],
            library_id=value["library_id"],
            version=value["version"],
            asl_convention_sha256=value["asl_convention_sha256"],
            sir_lexicon_sha256=value["sir_lexicon_sha256"],
            phase2_schema_sha256=value["phase2_schema_sha256"],
            entries=tuple(LexicalMotionEntry.from_dict(item)
                          for item in value["entries"]),
        )


def load_lexical_motion_library(
    payload: bytes,
    *,
    max_bytes: int = _MAX_MANIFEST_BYTES,
) -> LexicalMotionLibrary:
    """Load canonical bytes and reject reformatting, duplicate keys, or drift."""
    if (not isinstance(max_bytes, int) or isinstance(max_bytes, bool)
            or not 1 <= max_bytes <= _MAX_MANIFEST_BYTES):
        raise ValueError("max_bytes must be a positive bounded integer")
    value = _strict_json_loads(payload, max_bytes=max_bytes)
    if not isinstance(value, Mapping):
        raise ValueError("lexical-motion library root must be an object")
    library = LexicalMotionLibrary.from_dict(value)
    if library.canonical_bytes() != payload:
        raise ValueError("lexical-motion library bytes are not canonical")
    return library


class Phase3CWorkState(str, Enum):
    SOFTWARE_COMPLETE = "software_complete"
    EXTERNAL_ARTIFACT_BLOCKED = "external_artifact_blocked"
    EMPIRICAL_EXECUTION_BLOCKED = "empirical_execution_blocked"
    PREDECESSOR_PHASE_BLOCKED = "predecessor_phase_blocked"


@dataclass(frozen=True)
class Phase3CWorkItem:
    component_id: str
    software_contract_implemented: bool
    state: Phase3CWorkState
    required_artifacts: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        _require_id("component_id", self.component_id)
        if not isinstance(self.software_contract_implemented, bool):
            raise ValueError("software_contract_implemented must be boolean")
        if not isinstance(self.state, Phase3CWorkState):
            raise ValueError("Phase 3C work state must be typed")
        if (not isinstance(self.required_artifacts, tuple)
                or not self.required_artifacts
                or len(set(self.required_artifacts)) != len(self.required_artifacts)
                or any(not isinstance(item, str) or not item.strip()
                       for item in self.required_artifacts)):
            raise ValueError("blocked work must name unique required artifacts")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("blocked work must include a reason")
        if self.state is Phase3CWorkState.SOFTWARE_COMPLETE:
            raise ValueError("current work matrix records unresolved evidence work only")


@dataclass(frozen=True)
class Phase3CReadinessReport:
    preartifact_software_boundary_complete: bool
    research_exit_approved: bool
    industrial_path_approved: bool
    blockers: tuple[str, ...]
    work_items: tuple[Phase3CWorkItem, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(value, bool) for value in (
                self.preartifact_software_boundary_complete,
                self.research_exit_approved,
                self.industrial_path_approved,
        )):
            raise ValueError("Phase 3C readiness flags must be boolean")
        if self.industrial_path_approved and not self.research_exit_approved:
            raise ValueError("industrial approval requires research-exit approval")
        if (not isinstance(self.blockers, tuple)
                or len(set(self.blockers)) != len(self.blockers)
                or any(not isinstance(item, str) or not item
                       for item in self.blockers)):
            raise ValueError("Phase 3C blockers must be unique identifiers")
        if (not isinstance(self.work_items, tuple) or not self.work_items
                or any(not isinstance(item, Phase3CWorkItem)
                       for item in self.work_items)):
            raise ValueError("Phase 3C readiness requires typed work items")
        if self.research_exit_approved and any(
                blocker != "commercial_training_and_deployment_rights_absent"
                and blocker != "lexical_motion_not_commercially_authorized"
                for blocker in self.blockers):
            raise ValueError("research approval contradicts a research blocker")


_CURRENT_PHASE3C_WORK = (
    Phase3CWorkItem(
        "governed_corpus_population",
        True,
        Phase3CWorkState.EXTERNAL_ARTIFACT_BLOCKED,
        (
            "accepted_phase3b_sir_records",
            "qualified_asl_validation_attestation",
            "research_training_rights_evidence",
        ),
        "The certification contract exists, but authentic accepted records do not.",
    ),
    Phase3CWorkItem(
        "direct_paired_model_training",
        False,
        Phase3CWorkState.EXTERNAL_ARTIFACT_BLOCKED,
        (
            "governed_text_video_sir_corpus",
            "frozen_phase2_state_schema",
            "source_and_signer_disjoint_split",
        ),
        "Selecting and training the final learner before these inputs exist would freeze invented data semantics.",
    ),
    Phase3CWorkItem(
        "direct_paired_falsification_execution",
        True,
        Phase3CWorkState.EMPIRICAL_EXECUTION_BLOCKED,
        (
            "trained_direct_paired_model",
            "heldout_intervention_score_artifacts",
            "preregistered_effect_threshold",
        ),
        "The exact statistical evaluator exists, but no eligible model scores exist.",
    ),
    Phase3CWorkItem(
        "lexical_motion_library_population",
        True,
        Phase3CWorkState.EXTERNAL_ARTIFACT_BLOCKED,
        (
            "canonical_phase2_motion_segments",
            "human_meaning_reviews",
            "human_articulation_reviews",
            "motion_use_authorizations",
        ),
        "The registry and abstention path exist; no admissible reviewed motion entry exists.",
    ),
    Phase3CWorkItem(
        "canonical_phase2_motion_export",
        False,
        Phase3CWorkState.PREDECESSOR_PHASE_BLOCKED,
        (
            "authorized_multichannel_3d_source",
            "frozen_phase2_observability_contract",
            "validated_source_to_state_roundtrip",
        ),
        "The current frontal 2D observations cannot identify the required production state.",
    ),
    Phase3CWorkItem(
        "industrial_training_and_deployment",
        False,
        Phase3CWorkState.EXTERNAL_ARTIFACT_BLOCKED,
        (
            "commercial_training_rights",
            "commercial_motion_and_rig_rights",
            "independent_qualified_asl_product_validation",
        ),
        "Noncommercial research permission and software tests cannot authorize industrial deployment.",
    ),
)


def assess_phase3c_readiness(
    *,
    supervision: SupervisionBatchCertificate | None = None,
    video_dependence: VideoDependenceCertificate | None = None,
    lexical_motion_library: LexicalMotionLibrary | None = None,
    external_evidence: Sequence[ExternalEvidenceAttestation] = (),
) -> Phase3CReadinessReport:
    """Evaluate supplied evidence without inferring, repairing, or substituting it."""
    if supervision is not None and not isinstance(
            supervision, SupervisionBatchCertificate):
        raise TypeError("supervision must be a governed batch certificate or null")
    if video_dependence is not None and not isinstance(
            video_dependence, VideoDependenceCertificate):
        raise TypeError("video_dependence must be a governed certificate or null")
    if lexical_motion_library is not None and not isinstance(
            lexical_motion_library, LexicalMotionLibrary):
        raise TypeError("lexical_motion_library must be governed or null")
    if isinstance(external_evidence, (str, bytes)) or not isinstance(
            external_evidence, Sequence):
        raise TypeError("external_evidence must be a bounded sequence")
    if len(external_evidence) > len(EvidenceRole):
        raise ValueError("external evidence contains too many role records")
    if any(not isinstance(item, ExternalEvidenceAttestation)
           for item in external_evidence):
        raise TypeError("external evidence records must use the governed type")
    roles = [item.role for item in external_evidence]
    if len(set(roles)) != len(roles):
        raise ValueError("external evidence roles must be unique")
    artifact_hashes = [item.artifact_sha256 for item in external_evidence]
    if len(set(artifact_hashes)) != len(artifact_hashes):
        raise ValueError("one artifact cannot satisfy multiple evidence roles")

    available_roles = set(roles)
    blockers: list[str] = []
    if supervision is None or not supervision.approved_for_research_training:
        blockers.append("qualified_text_video_to_sir_supervision_absent")
    if video_dependence is None or not video_dependence.passed:
        blockers.append("direct_paired_learning_falsification_not_passed")
    if lexical_motion_library is None:
        blockers.append("human_approved_lexical_motion_library_absent")
    elif not any(
            "research_training" in entry.permitted_actions
            for entry in lexical_motion_library.entries):
        blockers.append("lexical_motion_library_not_authorized_for_research")
    phase2_evidence = next((
        item for item in external_evidence
        if item.role is EvidenceRole.CANONICAL_PHASE2_STATE
    ), None)
    if phase2_evidence is None:
        blockers.append("canonical_phase2_multichannel_state_absent")
    elif (lexical_motion_library is not None
          and phase2_evidence.artifact_sha256
          != lexical_motion_library.phase2_schema_sha256):
        blockers.append("lexical_motion_phase2_schema_mismatch")
    if EvidenceRole.QUALIFIED_ASL_VALIDATION not in available_roles:
        blockers.append("independent_qualified_asl_validation_absent")
    if EvidenceRole.RESEARCH_TRAINING_RIGHTS not in available_roles:
        blockers.append("research_training_rights_absent")
    research_approved = not blockers
    industrial_blockers = list(blockers)
    if EvidenceRole.COMMERCIAL_DEPLOYMENT_RIGHTS not in available_roles:
        industrial_blockers.append("commercial_training_and_deployment_rights_absent")
    if (lexical_motion_library is not None
            and any("commercial_deployment" not in entry.permitted_actions
                    for entry in lexical_motion_library.entries)):
        industrial_blockers.append("lexical_motion_not_commercially_authorized")

    return Phase3CReadinessReport(
        preartifact_software_boundary_complete=True,
        research_exit_approved=research_approved,
        industrial_path_approved=not industrial_blockers,
        blockers=tuple(dict.fromkeys(industrial_blockers)),
        work_items=_CURRENT_PHASE3C_WORK,
    )


__all__ = [
    "PHASE3C_SCHEMA_VERSION", "CANONICAL_PHASE2_REPRESENTATION",
    "EvidenceRole", "ExternalEvidenceAttestation",
    "load_external_evidence_attestation", "LexicalMotionReview",
    "LexicalMotionEntry", "MotionLookupStatus", "MotionLookupResult",
    "LexicalMotionLibrary", "load_lexical_motion_library",
    "Phase3CWorkState", "Phase3CWorkItem", "Phase3CReadinessReport",
    "assess_phase3c_readiness",
]
