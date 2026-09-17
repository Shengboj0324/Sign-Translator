"""Governed Phase 3B source-to-SIR review and adjudication.

This module records independently authored human judgments.  It never converts
publisher labels, English text, or machine output into SIR.  A source annotation
is first bound to a validated EAF manifest and unchanged source files.  Primary
and reviewer SIR submissions are then kept separate so disagreement remains
observable rather than being silently overwritten.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..data_engineering.eaf import (
    EAFDocument,
    load_eaf_manifest,
    parse_eaf_bytes,
)
from ..data_engineering.how2sign_audit import (
    StableFileDigest,
    assert_file_unchanged,
    stable_sha256,
)
from ..grammar.sir import (
    EventKind,
    SIREvent,
    SIRGraph,
    sir_from_dict,
    sir_sha256,
    sir_to_dict,
)
from .supervision import ArtifactKind, GovernedArtifact


PHASE3B_SOURCE_BINDING_SCHEMA_VERSION = 1
PHASE3B_SUBMISSION_SCHEMA_VERSION = 1
PHASE3B_SIR_TIME_UNIT = "milliseconds"
PHASE3B_SIR_TIME_ORIGIN = "eaf_time_order"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_BIDI_CONTROLS = frozenset(
    chr(value) for value in (
        0x061C, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
        0x2066, 0x2067, 0x2068, 0x2069,
    )
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_id(name: str, value: object) -> None:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a non-empty restricted identifier")


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


def _require_source_text(name: str, value: object, *, maximum: int = 1_000_000) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"{name} must be text within the declared size limit")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{name} must use NFC-normalized Unicode")
    if any(character in _BIDI_CONTROLS for character in value):
        raise ValueError(f"{name} contains a forbidden bidirectional control")
    if any(
        unicodedata.category(character) in {"Cc", "Cs"}
        and character not in "\t\n\r"
        for character in value
    ):
        raise ValueError(f"{name} contains a forbidden control or surrogate")
    return value


def _strict_json_loads(payload: bytes, *, max_bytes: int = 16 * 1024 * 1024) -> Any:
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("Phase 3B JSON must be non-empty immutable bytes")
    if (not isinstance(max_bytes, int) or isinstance(max_bytes, bool)
            or max_bytes < 1 or len(payload) > max_bytes):
        raise ValueError("Phase 3B JSON exceeds the configured byte limit")

    def unique_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value is forbidden: {value}")

    try:
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise ValueError(f"malformed Phase 3B JSON: {error}") from error


def _validate_limits(name: str, values: tuple[str, ...], *, maximum: int = 32) -> None:
    if not isinstance(values, tuple) or len(values) > maximum:
        raise ValueError(f"{name} must contain unique, bounded, non-empty strings")
    if any(not isinstance(item, str) or not item.strip() or len(item) > 512
           for item in values):
        raise ValueError(f"{name} must contain unique, bounded, non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique, bounded, non-empty strings")
    for item in values:
        _require_source_text(name, item, maximum=512)


def _validate_source_interval(
    begin_ms: object,
    end_ms: object,
    *,
    context: str,
) -> None:
    for name, value in (("begin_ms", begin_ms), ("end_ms", end_ms)):
        if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ValueError(f"{context} {name} must be a non-negative integer or null")
    if begin_ms is not None and end_ms is not None and begin_ms >= end_ms:
        raise ValueError(f"{context} source interval must have positive duration")


def _finite_binary64(value: object) -> float:
    """Reject non-finite or lossy integer conversion at a binary64 time boundary."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("time value must be a finite binary64 number")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("time value must be a finite binary64 number") from error
    if (not math.isfinite(converted)
            or isinstance(value, int) and int(converted) != value):
        raise ValueError("time value is not exactly representable in binary64")
    return converted


def _require_graph_within_source_interval(
    graph: SIRGraph,
    begin_ms: int | None,
    end_ms: int | None,
    *,
    context: str,
) -> None:
    if begin_ms is None or end_ms is None:
        raise ValueError(f"{context} cannot contain SIR when source timing is unavailable")
    if any(event.t_start < begin_ms or event.t_end > end_ms for event in graph.events):
        raise ValueError(f"{context} SIR event lies outside its bound source interval")


def _identity_digest(value: Mapping[str, Any]) -> StableFileDigest:
    required = {"relative_path", "sha256", "size", "device", "inode", "mtime_ns"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("source-file identity fields are invalid")
    _require_sha256("source-file sha256", value["sha256"])
    numbers = (value["size"], value["device"], value["inode"], value["mtime_ns"])
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0
           for item in numbers):
        raise ValueError("source-file identity numbers must be non-negative integers")
    return StableFileDigest(
        value["sha256"], value["size"], value["device"], value["inode"],
        value["mtime_ns"],
    )


def _resolve_manifest_file(
    root: Path,
    identity: Mapping[str, Any],
) -> tuple[Path, StableFileDigest, bool]:
    relative = identity.get("relative_path")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("manifest source path must be a POSIX relative path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or any(
            part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError("manifest source path escapes the source root")
    candidate = root / relative_path
    expected = _identity_digest(identity)
    observed = stable_sha256(candidate, root)
    if observed.sha256 != expected.sha256 or observed.size != expected.size:
        raise RuntimeError(f"source artifact differs from the EAF manifest: {relative}")
    return candidate.resolve(strict=True), observed, observed == expected


@dataclass(frozen=True)
class EAFAnnotationBinding:
    """Exact publisher annotation selected for independent human analysis."""

    schema_version: int
    eaf_manifest_sha256: str
    eaf_file_sha256: str
    eaf_schema_sha256: str
    media_descriptor_order: int
    source_video_sha256: str
    tier_id: str
    source_annotation_id: str
    source_annotation_kind: str
    source_order: int
    source_value: str
    source_value_sha256: str
    begin_time_slot_ref: str | None
    end_time_slot_ref: str | None
    begin_ms: int | None
    end_ms: int | None

    def __post_init__(self) -> None:
        if (not isinstance(self.schema_version, int)
                or isinstance(self.schema_version, bool)
                or self.schema_version != PHASE3B_SOURCE_BINDING_SCHEMA_VERSION):
            raise ValueError("unsupported Phase 3B source-binding schema")
        for name in (
            "eaf_manifest_sha256", "eaf_file_sha256", "eaf_schema_sha256",
            "source_video_sha256", "source_value_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (not isinstance(self.media_descriptor_order, int)
                or isinstance(self.media_descriptor_order, bool)
                or self.media_descriptor_order < 0):
            raise ValueError("media_descriptor_order must be a non-negative integer")
        if (not isinstance(self.source_order, int) or isinstance(self.source_order, bool)
                or self.source_order < 0):
            raise ValueError("source_order must be a non-negative integer")
        for name in ("tier_id", "source_annotation_id"):
            value = _require_source_text(name, getattr(self, name), maximum=16_384)
            if not value or any(character.isspace() for character in value):
                raise ValueError(f"{name} must be a non-empty source identifier")
        if self.source_annotation_kind not in {"alignable", "reference"}:
            raise ValueError("source_annotation_kind is invalid")
        value = _require_source_text("source_value", self.source_value)
        if hashlib.sha256(value.encode("utf-8")).hexdigest() != self.source_value_sha256:
            raise ValueError("source annotation value hash mismatch")
        for name in ("begin_time_slot_ref", "end_time_slot_ref"):
            item = getattr(self, name)
            if item is not None:
                checked = _require_source_text(name, item, maximum=16_384)
                if not checked or any(character.isspace() for character in checked):
                    raise ValueError(f"{name} must be a valid source identifier or null")
        if self.source_annotation_kind == "alignable" and (
                self.begin_time_slot_ref is None or self.end_time_slot_ref is None):
            raise ValueError("alignable source annotations require both time-slot references")
        for name in ("begin_ms", "end_ms"):
            item = getattr(self, name)
            if item is not None and (
                    not isinstance(item, int) or isinstance(item, bool) or item < 0):
                raise ValueError(f"{name} must be a non-negative integer or null")
        if self.begin_ms is not None and self.end_ms is not None \
                and self.begin_ms >= self.end_ms:
            raise ValueError("known source interval must have positive duration")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "eaf_manifest_sha256": self.eaf_manifest_sha256,
            "eaf_file_sha256": self.eaf_file_sha256,
            "eaf_schema_sha256": self.eaf_schema_sha256,
            "media_descriptor_order": self.media_descriptor_order,
            "source_video_sha256": self.source_video_sha256,
            "tier_id": self.tier_id,
            "source_annotation_id": self.source_annotation_id,
            "source_annotation_kind": self.source_annotation_kind,
            "source_order": self.source_order,
            "source_value": self.source_value,
            "source_value_sha256": self.source_value_sha256,
            "begin_time_slot_ref": self.begin_time_slot_ref,
            "end_time_slot_ref": self.end_time_slot_ref,
            "begin_ms": self.begin_ms,
            "end_ms": self.end_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EAFAnnotationBinding":
        required = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"source-binding fields must be exactly {sorted(required)}")
        return cls(**dict(value))

    def content_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()


@dataclass(frozen=True)
class EAFSourceCatalog:
    """Validated in-memory view of one immutable EAF source bundle."""

    manifest_sha256: str
    eaf_file_sha256: str
    eaf_schema_sha256: str
    document: EAFDocument
    media_sha256_by_order: tuple[str, ...]
    storage_identity_drift_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("manifest_sha256", "eaf_file_sha256", "eaf_schema_sha256"):
            _require_sha256(name, getattr(self, name))
        if not isinstance(self.document, EAFDocument):
            raise ValueError("catalog document must be a validated EAFDocument")
        if (not isinstance(self.media_sha256_by_order, tuple)
                or len(self.media_sha256_by_order) != len(
                    self.document.media_descriptors)
                or not self.media_sha256_by_order):
            raise ValueError("catalog media identities do not match its descriptors")
        for digest in self.media_sha256_by_order:
            _require_sha256("catalog media sha256", digest)
        _validate_limits(
            "storage_identity_drift_paths", self.storage_identity_drift_paths,
            maximum=128,
        )

    def bind(
        self,
        *,
        tier_id: str,
        source_annotation_id: str,
        media_descriptor_order: int = 0,
    ) -> EAFAnnotationBinding:
        if (not isinstance(media_descriptor_order, int)
                or isinstance(media_descriptor_order, bool)
                or not 0 <= media_descriptor_order < len(self.media_sha256_by_order)):
            raise ValueError("media descriptor order is outside the validated catalog")
        matches = [
            annotation
            for tier in self.document.tiers if tier.tier_id == tier_id
            for annotation in tier.annotations
            if annotation.annotation_id == source_annotation_id
        ]
        if len(matches) != 1:
            raise ValueError("source tier and annotation ID must identify exactly one record")
        annotation = matches[0]
        value_hash = hashlib.sha256(annotation.value.encode("utf-8")).hexdigest()
        return EAFAnnotationBinding(
            schema_version=PHASE3B_SOURCE_BINDING_SCHEMA_VERSION,
            eaf_manifest_sha256=self.manifest_sha256,
            eaf_file_sha256=self.eaf_file_sha256,
            eaf_schema_sha256=self.eaf_schema_sha256,
            media_descriptor_order=media_descriptor_order,
            source_video_sha256=self.media_sha256_by_order[media_descriptor_order],
            tier_id=tier_id,
            source_annotation_id=annotation.annotation_id,
            source_annotation_kind=annotation.kind,
            source_order=annotation.source_order,
            source_value=annotation.value,
            source_value_sha256=value_hash,
            begin_time_slot_ref=annotation.begin_time_slot_ref,
            end_time_slot_ref=annotation.end_time_slot_ref,
            begin_ms=annotation.begin_ms,
            end_ms=annotation.end_ms,
        )


def load_eaf_source_catalog(
    manifest_payload: bytes,
    source_root: str | Path,
) -> EAFSourceCatalog:
    """Validate current source bytes and expose bindable publisher annotations."""
    manifest = load_eaf_manifest(manifest_payload)
    root = Path(source_root)
    if root.is_symlink():
        raise ValueError("source root must not be a symlink")
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("source root must be a directory")

    identities: list[Mapping[str, Any]] = [
        manifest["eaf_file"], manifest["license_evidence_file"],
        *manifest["auxiliary_evidence_files"],
        *(binding["file"] for binding in manifest["media_bindings"]),
    ]
    expected_paths = {identity["relative_path"] for identity in identities}
    if len(expected_paths) != len(identities):
        raise ValueError("EAF manifest source roles must reference distinct files")

    def require_exact_inventory() -> None:
        discovered: set[str] = set()
        for path in resolved_root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"symlinked source entry is forbidden: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"unsupported source entry: {path}")
            discovered.add(path.relative_to(resolved_root).as_posix())
            if len(discovered) > 128:
                raise ValueError("source root exceeds the Phase 3B file-count limit")
        if discovered != expected_paths:
            raise ValueError("current source-root inventory differs from the EAF manifest")

    require_exact_inventory()

    resolved_records = [
        _resolve_manifest_file(resolved_root, identity) for identity in identities
    ]
    eaf_path = resolved_records[0][0]
    eaf_payload = eaf_path.read_bytes()
    if hashlib.sha256(eaf_payload).hexdigest() != manifest["eaf_file"]["sha256"]:
        raise RuntimeError("EAF changed between catalog hashing and parsing")
    document = parse_eaf_bytes(eaf_payload)
    if document.root.to_dict() != manifest["root"]:
        raise ValueError("EAF manifest tree differs from the bound source file")
    for path, observed, _ in resolved_records:
        assert_file_unchanged(path, resolved_root, observed)
    require_exact_inventory()
    media_sha256 = tuple(
        item["file"]["sha256"] for item in manifest["media_bindings"]
    )
    return EAFSourceCatalog(
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        eaf_file_sha256=manifest["eaf_file"]["sha256"],
        eaf_schema_sha256=manifest["eaf_schema_sha256"],
        document=document,
        media_sha256_by_order=media_sha256,
        storage_identity_drift_paths=tuple(
            identity["relative_path"]
            for identity, (_, _, identity_matches) in zip(
                identities, resolved_records, strict=True)
            if not identity_matches
        ),
    )


class SubmissionRole(str, Enum):
    PRIMARY = "primary"
    INDEPENDENT_REVIEWER = "independent_reviewer"


class SubmissionDecision(str, Enum):
    SIR = "sir"
    ABSTAIN = "abstain"


ABSTENTION_REASON_CODES = frozenset({
    "insufficient_visual_evidence",
    "source_misalignment",
    "lexicon_gap",
    "out_of_scope",
    "uncertain_analysis",
    "other_documented",
})


@dataclass(frozen=True)
class HumanSIRSubmission:
    """One independent human judgment; never an approved training record."""

    schema_version: int
    submission_id: str
    case_id: str
    source_binding_sha256: str
    source_interval_begin_ms: int | None
    source_interval_end_ms: int | None
    sir_time_unit: str
    sir_time_origin: str
    role: SubmissionRole
    author_pseudonym: str
    author_qualified_asl: bool
    source_video_reviewed: bool
    independently_created: bool
    qualification_evidence_sha256: str
    independence_evidence_sha256: str
    attestation_sha256: str
    protocol: GovernedArtifact
    decision: SubmissionDecision
    sir_payload: bytes | None
    sir_payload_sha256: str | None
    submitted_at: str
    reason_codes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.schema_version, int)
                or isinstance(self.schema_version, bool)
                or self.schema_version != PHASE3B_SUBMISSION_SCHEMA_VERSION):
            raise ValueError("unsupported Phase 3B submission schema")
        _require_id("submission_id", self.submission_id)
        _require_id("case_id", self.case_id)
        _require_id("author_pseudonym", self.author_pseudonym)
        _require_sha256("source_binding_sha256", self.source_binding_sha256)
        _validate_source_interval(
            self.source_interval_begin_ms,
            self.source_interval_end_ms,
            context="submission",
        )
        if (self.sir_time_unit != PHASE3B_SIR_TIME_UNIT
                or self.sir_time_origin != PHASE3B_SIR_TIME_ORIGIN):
            raise ValueError("submission uses an unsupported SIR timebase")
        _require_sha256("qualification_evidence_sha256",
                        self.qualification_evidence_sha256)
        _require_sha256("independence_evidence_sha256",
                        self.independence_evidence_sha256)
        _require_sha256("attestation_sha256", self.attestation_sha256)
        if len({
                self.qualification_evidence_sha256,
                self.independence_evidence_sha256,
                self.attestation_sha256,
        }) != 3:
            raise ValueError("submission evidence roles must use distinct artifacts")
        if not isinstance(self.role, SubmissionRole):
            raise ValueError("submission role is invalid")
        if not isinstance(self.decision, SubmissionDecision):
            raise ValueError("submission decision is invalid")
        if any(flag is not True for flag in (
                self.author_qualified_asl, self.source_video_reviewed,
                self.independently_created)):
            raise ValueError(
                "submission requires qualified, video-based, independent human work")
        expected_protocol = (
            ArtifactKind.ANNOTATION_PROTOCOL
            if self.role is SubmissionRole.PRIMARY
            else ArtifactKind.REVIEW_PROTOCOL
        )
        if not isinstance(self.protocol, GovernedArtifact) \
                or self.protocol.kind is not expected_protocol:
            raise ValueError("submission protocol has the wrong artifact kind")
        _require_timestamp("submitted_at", self.submitted_at)
        _validate_limits("reason_codes", self.reason_codes)
        _validate_limits("limitations", self.limitations)
        unknown_reasons = set(self.reason_codes) - ABSTENTION_REASON_CODES
        if unknown_reasons:
            raise ValueError(f"unknown submission reason codes: {sorted(unknown_reasons)}")
        if self.decision is SubmissionDecision.ABSTAIN:
            if self.sir_payload is not None or self.sir_payload_sha256 is not None:
                raise ValueError("abstention cannot contain a SIR payload")
            if not self.reason_codes:
                raise ValueError("abstention requires at least one declared reason code")
        else:
            if self.reason_codes:
                raise ValueError("SIR submission cannot contain abstention reason codes")
            if not isinstance(self.sir_payload, bytes) or not self.sir_payload:
                raise ValueError("SIR submission requires immutable canonical payload bytes")
            _require_sha256("sir_payload_sha256", self.sir_payload_sha256)
            graph = self.graph()
            if not graph.events or not graph.manual_events():
                raise ValueError("human SIR submission requires a manual event")
            _require_graph_within_source_interval(
                graph,
                self.source_interval_begin_ms,
                self.source_interval_end_ms,
                context="submission",
            )
            if sir_sha256(graph) != self.sir_payload_sha256:
                raise ValueError("human SIR submission hash mismatch")
            if self.sir_payload != _canonical_json_bytes(sir_to_dict(graph)):
                raise ValueError("human SIR submission is not canonical JSON")

    @classmethod
    def create(
        cls,
        *,
        submission_id: str,
        case_id: str,
        source: EAFAnnotationBinding,
        role: SubmissionRole,
        author_pseudonym: str,
        author_qualified_asl: bool,
        source_video_reviewed: bool,
        independently_created: bool,
        qualification_evidence_sha256: str,
        independence_evidence_sha256: str,
        attestation_sha256: str,
        protocol: GovernedArtifact,
        decision: SubmissionDecision,
        submitted_at: str,
        graph: SIRGraph | None = None,
        reason_codes: Sequence[str] = (),
        limitations: Sequence[str] = (),
    ) -> "HumanSIRSubmission":
        if not isinstance(source, EAFAnnotationBinding):
            raise TypeError("submission source must be an EAF annotation binding")
        if graph is not None and not isinstance(graph, SIRGraph):
            raise TypeError("submission graph must be a SIR graph or null")
        payload = None if graph is None else _canonical_json_bytes(sir_to_dict(graph))
        digest = None if graph is None else sir_sha256(graph)
        return cls(
            schema_version=PHASE3B_SUBMISSION_SCHEMA_VERSION,
            submission_id=submission_id,
            case_id=case_id,
            source_binding_sha256=source.content_sha256(),
            source_interval_begin_ms=source.begin_ms,
            source_interval_end_ms=source.end_ms,
            sir_time_unit=PHASE3B_SIR_TIME_UNIT,
            sir_time_origin=PHASE3B_SIR_TIME_ORIGIN,
            role=role,
            author_pseudonym=author_pseudonym,
            author_qualified_asl=author_qualified_asl,
            source_video_reviewed=source_video_reviewed,
            independently_created=independently_created,
            qualification_evidence_sha256=qualification_evidence_sha256,
            independence_evidence_sha256=independence_evidence_sha256,
            attestation_sha256=attestation_sha256,
            protocol=protocol,
            decision=decision,
            sir_payload=payload,
            sir_payload_sha256=digest,
            submitted_at=submitted_at,
            reason_codes=tuple(reason_codes),
            limitations=tuple(limitations),
        )

    def graph(self) -> SIRGraph:
        if self.decision is not SubmissionDecision.SIR or self.sir_payload is None:
            raise ValueError("abstained submission has no SIR graph")
        value = _strict_json_loads(self.sir_payload)
        if not isinstance(value, Mapping):
            raise ValueError("SIR submission root must be an object")
        return sir_from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "submission_id": self.submission_id,
            "case_id": self.case_id,
            "source_binding_sha256": self.source_binding_sha256,
            "source_interval_begin_ms": self.source_interval_begin_ms,
            "source_interval_end_ms": self.source_interval_end_ms,
            "sir_time_unit": self.sir_time_unit,
            "sir_time_origin": self.sir_time_origin,
            "role": self.role.value,
            "author_pseudonym": self.author_pseudonym,
            "author_qualified_asl": self.author_qualified_asl,
            "source_video_reviewed": self.source_video_reviewed,
            "independently_created": self.independently_created,
            "qualification_evidence_sha256": self.qualification_evidence_sha256,
            "independence_evidence_sha256": self.independence_evidence_sha256,
            "attestation_sha256": self.attestation_sha256,
            "protocol": self.protocol.to_dict(),
            "decision": self.decision.value,
            "sir": None if self.sir_payload is None else sir_to_dict(self.graph()),
            "sir_payload_sha256": self.sir_payload_sha256,
            "submitted_at": self.submitted_at,
            "reason_codes": list(self.reason_codes),
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HumanSIRSubmission":
        required = {
            "schema_version", "submission_id", "case_id", "source_binding_sha256",
            "source_interval_begin_ms", "source_interval_end_ms",
            "sir_time_unit", "sir_time_origin",
            "role", "author_pseudonym", "author_qualified_asl",
            "source_video_reviewed", "independently_created",
            "qualification_evidence_sha256", "independence_evidence_sha256",
            "attestation_sha256", "protocol",
            "decision", "sir", "sir_payload_sha256", "submitted_at",
            "reason_codes", "limitations",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"submission fields must be exactly {sorted(required)}")
        if not isinstance(value["protocol"], Mapping):
            raise ValueError("submission protocol must be an object")
        if not isinstance(value["reason_codes"], list) \
                or not isinstance(value["limitations"], list):
            raise ValueError("submission reasons and limitations must be lists")
        sir = value["sir"]
        if sir is not None and not isinstance(sir, Mapping):
            raise ValueError("submission SIR must be an object or null")
        try:
            role = SubmissionRole(value["role"])
            decision = SubmissionDecision(value["decision"])
        except (TypeError, ValueError) as error:
            raise ValueError("unknown submission role or decision") from error
        return cls(
            schema_version=value["schema_version"],
            submission_id=value["submission_id"],
            case_id=value["case_id"],
            source_binding_sha256=value["source_binding_sha256"],
            source_interval_begin_ms=value["source_interval_begin_ms"],
            source_interval_end_ms=value["source_interval_end_ms"],
            sir_time_unit=value["sir_time_unit"],
            sir_time_origin=value["sir_time_origin"],
            role=role,
            author_pseudonym=value["author_pseudonym"],
            author_qualified_asl=value["author_qualified_asl"],
            source_video_reviewed=value["source_video_reviewed"],
            independently_created=value["independently_created"],
            qualification_evidence_sha256=value["qualification_evidence_sha256"],
            independence_evidence_sha256=value["independence_evidence_sha256"],
            attestation_sha256=value["attestation_sha256"],
            protocol=GovernedArtifact.from_dict(value["protocol"]),
            decision=decision,
            sir_payload=None if sir is None else _canonical_json_bytes(sir),
            sir_payload_sha256=value["sir_payload_sha256"],
            submitted_at=value["submitted_at"],
            reason_codes=tuple(value["reason_codes"]),
            limitations=tuple(value["limitations"]),
        )

    def content_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()


@dataclass(frozen=True)
class EventCorrespondence:
    """Reviewer-declared one-to-one event pairing; no graph matching is inferred."""

    pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pairs, tuple) or any(
                not isinstance(pair, tuple) or len(pair) != 2
                or any(not isinstance(item, int) or isinstance(item, bool) or item < 0
                       for item in pair)
                for pair in self.pairs):
            raise ValueError("event correspondence must contain non-negative integer pairs")
        primary = [pair[0] for pair in self.pairs]
        reviewer = [pair[1] for pair in self.pairs]
        if len(set(primary)) != len(primary) or len(set(reviewer)) != len(reviewer):
            raise ValueError("event correspondence must be one-to-one")
        if tuple(sorted(self.pairs)) != self.pairs:
            raise ValueError("event correspondence must use canonical primary-ID order")


@dataclass(frozen=True)
class TemporalAgreement:
    primary_event_id: int
    reviewer_event_id: int
    temporal_iou: float
    reviewer_minus_primary_onset: float
    reviewer_minus_primary_offset: float

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in (self.primary_event_id, self.reviewer_event_id)):
            raise ValueError("temporal agreement event IDs must be non-negative integers")
        if (not isinstance(self.temporal_iou, float)
                or not math.isfinite(self.temporal_iou)
                or not 0.0 <= self.temporal_iou <= 1.0):
            raise ValueError("temporal agreement IoU must be a finite float in [0, 1]")
        for value in (
                self.reviewer_minus_primary_onset,
                self.reviewer_minus_primary_offset):
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError("temporal agreement deltas must be finite floats")


@dataclass(frozen=True)
class FieldAgreement:
    field: str
    agreements: int
    support: int
    rate: float | None
    primary_present: int
    reviewer_present: int
    both_present: int
    both_absent: int
    co_present_agreements: int
    co_present_rate: float | None

    def __post_init__(self) -> None:
        if (not isinstance(self.field, str)
                or self.field not in {"kind", "label", "referent", "locus"}):
            raise ValueError("unknown Phase 3B agreement field")
        if (not isinstance(self.agreements, int) or isinstance(self.agreements, bool)
                or not isinstance(self.support, int) or isinstance(self.support, bool)
                or not 0 <= self.agreements <= self.support):
            raise ValueError("field agreement counts are invalid")
        expected = None if self.support == 0 else self.agreements / self.support
        if self.rate is not None and (
                not isinstance(self.rate, float) or not math.isfinite(self.rate)):
            raise ValueError("field agreement rate must be a finite float or null")
        if self.rate != expected:
            raise ValueError("field agreement rate does not match its counts")
        counts = (
            self.primary_present, self.reviewer_present, self.both_present,
            self.both_absent, self.co_present_agreements,
        )
        if any(not isinstance(count, int) or isinstance(count, bool)
               or not 0 <= count <= self.support for count in counts):
            raise ValueError("field presence counts are invalid")
        if (self.both_present > min(self.primary_present, self.reviewer_present)
                or self.both_absent != self.support - self.primary_present
                - self.reviewer_present + self.both_present
                or self.co_present_agreements > self.both_present
                or self.agreements != self.both_absent
                + self.co_present_agreements):
            raise ValueError("field presence counts contradict agreement counts")
        if self.field in {"kind", "label"} and (
                self.primary_present != self.support
                or self.reviewer_present != self.support):
            raise ValueError("mandatory agreement fields must be present")
        expected_co_present = (
            None if self.both_present == 0
            else self.co_present_agreements / self.both_present
        )
        if self.co_present_rate is not None and (
                not isinstance(self.co_present_rate, float)
                or not math.isfinite(self.co_present_rate)):
            raise ValueError("co-present field agreement rate must be finite or null")
        if self.co_present_rate != expected_co_present:
            raise ValueError("co-present field agreement rate does not match its counts")


@dataclass(frozen=True)
class SubmissionAgreementReport:
    comparison_available: bool
    unavailable_reason: str | None
    exact_sir_hash_match: bool
    primary_event_count: int
    reviewer_event_count: int
    paired_event_count: int
    primary_unpaired_event_count: int
    reviewer_unpaired_event_count: int
    field_agreement: tuple[FieldAgreement, ...]
    kind_confusion: tuple[tuple[str, str, int], ...]
    label_confusion: tuple[tuple[int, int, int], ...]
    temporal: tuple[TemporalAgreement, ...]
    median_temporal_iou: float | None
    median_absolute_onset_difference: float | None
    median_absolute_offset_difference: float | None
    primary_edge_count: int
    reviewer_edge_count: int
    comparable_edge_intersection: int
    comparable_edge_union: int
    comparable_edge_jaccard: float | None


def _validate_submission_pair(
    primary: HumanSIRSubmission,
    reviewer: HumanSIRSubmission,
) -> None:
    """Reject invalid review pairs before metrics or adjudication are created."""
    if not isinstance(primary, HumanSIRSubmission) \
            or not isinstance(reviewer, HumanSIRSubmission):
        raise TypeError("submission pair must contain two human SIR submissions")
    if primary.role is not SubmissionRole.PRIMARY:
        raise ValueError("primary argument does not have the primary role")
    if reviewer.role is not SubmissionRole.INDEPENDENT_REVIEWER:
        raise ValueError("reviewer argument does not have the reviewer role")
    if primary.case_id != reviewer.case_id:
        raise ValueError("submissions belong to different review cases")
    if primary.source_binding_sha256 != reviewer.source_binding_sha256:
        raise ValueError("submissions refer to different source annotations")
    if (primary.source_interval_begin_ms != reviewer.source_interval_begin_ms
            or primary.source_interval_end_ms != reviewer.source_interval_end_ms
            or primary.sir_time_unit != reviewer.sir_time_unit
            or primary.sir_time_origin != reviewer.sir_time_origin):
        raise ValueError("submissions use different source timing contracts")
    if primary.author_pseudonym == reviewer.author_pseudonym:
        raise ValueError("primary annotator and reviewer must be distinct")
    if primary.submission_id == reviewer.submission_id:
        raise ValueError("primary and reviewer submission identifiers must be distinct")
    if {
            primary.qualification_evidence_sha256,
            primary.independence_evidence_sha256,
            primary.attestation_sha256,
    } & {
            reviewer.qualification_evidence_sha256,
            reviewer.independence_evidence_sha256,
            reviewer.attestation_sha256,
    }:
        raise ValueError("primary and reviewer evidence artifacts must be distinct")
    primary_time = _require_timestamp("primary submitted_at", primary.submitted_at)
    reviewer_time = _require_timestamp("reviewer submitted_at", reviewer.submitted_at)
    if reviewer_time <= primary_time:
        raise ValueError("review submission must follow primary completion")


def temporal_iou(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Intersection over union of two finite, positive-duration intervals."""
    if len(first) != 2 or len(second) != 2:
        raise ValueError("temporal intervals must contain two endpoints")
    values = (*first, *second)
    try:
        first = (_finite_binary64(values[0]), _finite_binary64(values[1]))
        second = (_finite_binary64(values[2]), _finite_binary64(values[3]))
    except ValueError as error:
        raise ValueError("temporal intervals require finite exact binary64 endpoints") \
            from error
    values = (*first, *second)
    if first[0] >= first[1] or second[0] >= second[1]:
        raise ValueError("temporal intervals must have positive duration")
    # The enclosing span equals the set union whenever intersection is positive;
    # for disjoint intervals the numerator is zero regardless of their gap.
    left = max(first[0], second[0])
    right = min(first[1], second[1])
    if left >= right:
        return 0.0
    outer_left = min(first[0], second[0])
    outer_right = max(first[1], second[1])
    union = outer_right - outer_left
    if math.isfinite(union):
        intersection = right - left
        if not math.isfinite(intersection):
            raise ValueError("temporal intersection is not representable")
        result = intersection / union
    else:
        # Finite endpoints can have an unrepresentable enclosing span. Preserve
        # a representable overlap difference *before* scaling: subtracting two
        # already-scaled, nearly equal endpoints can erase small valid overlap.
        scale = max(abs(value) for value in values)
        overlap = right - left
        intersection = (
            overlap / scale if math.isfinite(overlap)
            else right / scale - left / scale
        )
        denominator = outer_right / scale - outer_left / scale
        result = intersection / denominator
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("temporal IoU is not representable in [0, 1]")
    if result == 0.0 or (result == 1.0 and first != second):
        raise ValueError("temporal IoU is not representable without endpoint collapse")
    return result


def _finite_nonnegative_median(values: Sequence[float]) -> float:
    """Median without overflowing the midpoint of two finite metrics."""
    if not values or any(
            not isinstance(value, float) or not math.isfinite(value)
            or value < 0.0 for value in values):
        raise ValueError("median requires nonempty finite non-negative floats")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    lower, upper = ordered[middle - 1], ordered[middle]
    return float((Fraction.from_float(lower) + Fraction.from_float(upper)) / 2)


def _field_agreement(name: str, values: Sequence[tuple[object, object]]) -> FieldAgreement:
    support = len(values)
    agreements = sum(first == second for first, second in values)
    primary_present = sum(first is not None for first, _ in values)
    reviewer_present = sum(second is not None for _, second in values)
    both_present = sum(first is not None and second is not None
                       for first, second in values)
    both_absent = sum(first is None and second is None
                      for first, second in values)
    co_present_agreements = agreements - both_absent
    return FieldAgreement(
        field=name,
        agreements=agreements,
        support=support,
        rate=None if support == 0 else agreements / support,
        primary_present=primary_present,
        reviewer_present=reviewer_present,
        both_present=both_present,
        both_absent=both_absent,
        co_present_agreements=co_present_agreements,
        co_present_rate=(None if both_present == 0
                         else co_present_agreements / both_present),
    )


def _unavailable_agreement(
    reason: str,
) -> SubmissionAgreementReport:
    return SubmissionAgreementReport(
        comparison_available=False,
        unavailable_reason=reason,
        exact_sir_hash_match=False,
        primary_event_count=0,
        reviewer_event_count=0,
        paired_event_count=0,
        primary_unpaired_event_count=0,
        reviewer_unpaired_event_count=0,
        field_agreement=(),
        kind_confusion=(),
        label_confusion=(),
        temporal=(),
        median_temporal_iou=None,
        median_absolute_onset_difference=None,
        median_absolute_offset_difference=None,
        primary_edge_count=0,
        reviewer_edge_count=0,
        comparable_edge_intersection=0,
        comparable_edge_union=0,
        comparable_edge_jaccard=None,
    )


def compare_submissions(
    primary: HumanSIRSubmission,
    reviewer: HumanSIRSubmission,
    correspondence: EventCorrespondence,
) -> SubmissionAgreementReport:
    """Report graph agreement without choosing a pass threshold or alignment."""
    _validate_submission_pair(primary, reviewer)
    if not isinstance(correspondence, EventCorrespondence):
        raise TypeError("correspondence must be an event correspondence")
    if primary.decision is SubmissionDecision.ABSTAIN \
            or reviewer.decision is SubmissionDecision.ABSTAIN:
        if correspondence.pairs:
            raise ValueError("abstained submissions cannot declare event correspondence")
        reason = (
            "both_abstained" if primary.decision is reviewer.decision
            else "one_submission_abstained"
        )
        return _unavailable_agreement(reason)

    first_graph = primary.graph()
    second_graph = reviewer.graph()
    first_events = {event.id: event for event in first_graph.events}
    second_events = {event.id: event for event in second_graph.events}
    pairs = correspondence.pairs
    identical_sir = primary.sir_payload_sha256 == reviewer.sir_payload_sha256
    if identical_sir and pairs:
        raise ValueError(
            "identical SIR submissions require implicit complete correspondence")
    if identical_sir:
        pairs = tuple((event_id, event_id) for event_id in sorted(first_events))
    if any(first not in first_events or second not in second_events
           for first, second in pairs):
        raise ValueError("event correspondence references an absent event")

    event_pairs: list[tuple[SIREvent, SIREvent]] = [
        (first_events[first], second_events[second]) for first, second in pairs
    ]
    field_values: dict[str, list[tuple[object, object]]] = {
        "kind": [], "label": [], "referent": [], "locus": [],
    }
    temporal: list[TemporalAgreement] = []
    kind_confusion: Counter[tuple[str, str]] = Counter()
    label_confusion: Counter[tuple[int, int]] = Counter()
    for first, second in event_pairs:
        field_values["kind"].append((first.kind.value, second.kind.value))
        field_values["label"].append((first.label, second.label))
        field_values["referent"].append((first.referent, second.referent))
        field_values["locus"].append((first.locus, second.locus))
        kind_confusion[(first.kind.value, second.kind.value)] += 1
        label_confusion[(first.label, second.label)] += 1
        temporal.append(TemporalAgreement(
            primary_event_id=first.id,
            reviewer_event_id=second.id,
            temporal_iou=temporal_iou(
                (first.t_start, first.t_end), (second.t_start, second.t_end)),
            reviewer_minus_primary_onset=second.t_start - first.t_start,
            reviewer_minus_primary_offset=second.t_end - first.t_end,
        ))

    second_to_first = {second: first for first, second in pairs}
    paired_first = set(second_to_first.values())
    paired_second = set(second_to_first)
    first_edges = {
        (edge.source, edge.target, edge.type.value)
        for edge in first_graph.edges
        if edge.source in paired_first and edge.target in paired_first
    }
    second_edges = {
        (second_to_first[edge.source], second_to_first[edge.target], edge.type.value)
        for edge in second_graph.edges
        if edge.source in paired_second and edge.target in paired_second
    }
    intersection = len(first_edges & second_edges)
    union = len(first_edges | second_edges)
    temporal_ious = [item.temporal_iou for item in temporal]
    onset_differences = [abs(item.reviewer_minus_primary_onset) for item in temporal]
    offset_differences = [abs(item.reviewer_minus_primary_offset) for item in temporal]
    return SubmissionAgreementReport(
        comparison_available=True,
        unavailable_reason=None,
        exact_sir_hash_match=identical_sir,
        primary_event_count=len(first_events),
        reviewer_event_count=len(second_events),
        paired_event_count=len(pairs),
        primary_unpaired_event_count=len(first_events) - len(pairs),
        reviewer_unpaired_event_count=len(second_events) - len(pairs),
        field_agreement=tuple(
            _field_agreement(name, field_values[name])
            for name in ("kind", "label", "referent", "locus")
        ),
        kind_confusion=tuple(
            (first, second, count)
            for (first, second), count in sorted(kind_confusion.items())
        ),
        label_confusion=tuple(
            (first, second, count)
            for (first, second), count in sorted(label_confusion.items())
        ),
        temporal=tuple(temporal),
        median_temporal_iou=(
            None if not temporal_ious else _finite_nonnegative_median(temporal_ious)),
        median_absolute_onset_difference=(
            None if not onset_differences
            else _finite_nonnegative_median(onset_differences)),
        median_absolute_offset_difference=(
            None if not offset_differences
            else _finite_nonnegative_median(offset_differences)),
        primary_edge_count=len(first_graph.edges),
        reviewer_edge_count=len(second_graph.edges),
        comparable_edge_intersection=intersection,
        comparable_edge_union=union,
        comparable_edge_jaccard=None if union == 0 else intersection / union,
    )


PHASE3B_ADJUDICATION_SCHEMA_VERSION = 1
PHASE3B_CASE_SCHEMA_VERSION = 1
PHASE3B_EVENT_SCHEMA_VERSION = 1
PHASE3B_BATCH_REPORT_SCHEMA_VERSION = 2


class AdjudicationOutcome(str, Enum):
    ACCEPT_PRIMARY = "accept_primary"
    ACCEPT_REVIEWER = "accept_reviewer"
    REVISED = "revised"
    REJECTED = "rejected"
    ABSTAINED = "abstained"


class WorkflowState(str, Enum):
    DRAFT = "draft"
    PRIMARY_COMPLETE = "primary_complete"
    BLIND_REVIEWED = "blind_reviewed"
    ADJUDICATED = "adjudicated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ABSTAINED = "abstained"

    @property
    def terminal(self) -> bool:
        return self in {
            WorkflowState.ACCEPTED,
            WorkflowState.REJECTED,
            WorkflowState.ABSTAINED,
        }


ADJUDICATION_REASON_CODES = frozenset({
    "event_inventory_disagreement",
    "lexical_label_disagreement",
    "event_kind_disagreement",
    "timing_disagreement",
    "referent_disagreement",
    "spatial_locus_disagreement",
    "edge_relation_disagreement",
    "source_evidence_insufficient",
    "convention_or_lexicon_gap",
    "other_documented",
})


@dataclass(frozen=True)
class HumanAdjudication:
    """Third-person resolution of a non-identical primary/reviewer result."""

    schema_version: int
    adjudication_id: str
    case_id: str
    source_binding_sha256: str
    source_interval_begin_ms: int | None
    source_interval_end_ms: int | None
    sir_time_unit: str
    sir_time_origin: str
    primary_submission_sha256: str
    reviewer_submission_sha256: str
    adjudicator_pseudonym: str
    adjudicator_qualified_asl: bool
    source_video_reviewed: bool
    independent_of_submitters: bool
    qualification_evidence_sha256: str
    independence_evidence_sha256: str
    attestation_sha256: str
    protocol: GovernedArtifact
    outcome: AdjudicationOutcome
    final_sir_payload: bytes | None
    final_sir_payload_sha256: str | None
    submitted_at: str
    reason_codes: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.schema_version, int)
                or isinstance(self.schema_version, bool)
                or self.schema_version != PHASE3B_ADJUDICATION_SCHEMA_VERSION):
            raise ValueError("unsupported Phase 3B adjudication schema")
        for name in ("adjudication_id", "case_id", "adjudicator_pseudonym"):
            _require_id(name, getattr(self, name))
        for name in (
            "source_binding_sha256", "primary_submission_sha256",
            "reviewer_submission_sha256", "qualification_evidence_sha256",
            "independence_evidence_sha256", "attestation_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if len({
                self.qualification_evidence_sha256,
                self.independence_evidence_sha256,
                self.attestation_sha256,
        }) != 3:
            raise ValueError("adjudication evidence roles must use distinct artifacts")
        _validate_source_interval(
            self.source_interval_begin_ms,
            self.source_interval_end_ms,
            context="adjudication",
        )
        if (self.sir_time_unit != PHASE3B_SIR_TIME_UNIT
                or self.sir_time_origin != PHASE3B_SIR_TIME_ORIGIN):
            raise ValueError("adjudication uses an unsupported SIR timebase")
        if any(flag is not True for flag in (
                self.adjudicator_qualified_asl, self.source_video_reviewed,
                self.independent_of_submitters)):
            raise ValueError(
                "adjudication requires qualified, video-based, independent human work")
        if not isinstance(self.protocol, GovernedArtifact) \
                or self.protocol.kind is not ArtifactKind.ADJUDICATION_PROTOCOL:
            raise ValueError("adjudication protocol has the wrong artifact kind")
        if not isinstance(self.outcome, AdjudicationOutcome):
            raise ValueError("adjudication outcome is invalid")
        _require_timestamp("adjudication submitted_at", self.submitted_at)
        _validate_limits("adjudication reason_codes", self.reason_codes)
        _validate_limits("adjudication limitations", self.limitations)
        if not self.reason_codes:
            raise ValueError("adjudication requires at least one reason code")
        unknown_reasons = set(self.reason_codes) - ADJUDICATION_REASON_CODES
        if unknown_reasons:
            raise ValueError(f"unknown adjudication reason codes: {sorted(unknown_reasons)}")

        resolves_to_sir = self.outcome in {
            AdjudicationOutcome.ACCEPT_PRIMARY,
            AdjudicationOutcome.ACCEPT_REVIEWER,
            AdjudicationOutcome.REVISED,
        }
        if not resolves_to_sir:
            if self.final_sir_payload is not None \
                    or self.final_sir_payload_sha256 is not None:
                raise ValueError("rejected or abstained adjudication cannot contain SIR")
            return
        if not isinstance(self.final_sir_payload, bytes) or not self.final_sir_payload:
            raise ValueError("accepted adjudication requires immutable canonical SIR bytes")
        _require_sha256("final_sir_payload_sha256", self.final_sir_payload_sha256)
        graph = self.final_graph()
        if not graph.events or not graph.manual_events():
            raise ValueError("accepted adjudication requires at least one manual event")
        _require_graph_within_source_interval(
            graph,
            self.source_interval_begin_ms,
            self.source_interval_end_ms,
            context="adjudication",
        )
        if sir_sha256(graph) != self.final_sir_payload_sha256:
            raise ValueError("adjudicated SIR hash mismatch")
        if self.final_sir_payload != _canonical_json_bytes(sir_to_dict(graph)):
            raise ValueError("adjudicated SIR is not canonical JSON")

    @classmethod
    def create(
        cls,
        *,
        adjudication_id: str,
        case_id: str,
        source: EAFAnnotationBinding,
        primary: HumanSIRSubmission,
        reviewer: HumanSIRSubmission,
        adjudicator_pseudonym: str,
        adjudicator_qualified_asl: bool,
        source_video_reviewed: bool,
        independent_of_submitters: bool,
        qualification_evidence_sha256: str,
        independence_evidence_sha256: str,
        attestation_sha256: str,
        protocol: GovernedArtifact,
        outcome: AdjudicationOutcome,
        submitted_at: str,
        final_graph: SIRGraph | None = None,
        reason_codes: Sequence[str] = (),
        limitations: Sequence[str] = (),
    ) -> "HumanAdjudication":
        if not isinstance(source, EAFAnnotationBinding):
            raise TypeError("adjudication source must be an EAF annotation binding")
        _require_id("case_id", case_id)
        _require_id("adjudicator_pseudonym", adjudicator_pseudonym)
        if final_graph is not None and not isinstance(final_graph, SIRGraph):
            raise TypeError("adjudication graph must be a SIR graph or null")
        _validate_submission_pair(primary, reviewer)
        if primary.case_id != case_id:
            raise ValueError("adjudication submissions belong to another case")
        source_sha256 = source.content_sha256()
        if (primary.source_binding_sha256 != source_sha256
                or reviewer.source_binding_sha256 != source_sha256):
            raise ValueError("adjudication submissions bind another source annotation")
        expected_timing = (
            source.begin_ms,
            source.end_ms,
            PHASE3B_SIR_TIME_UNIT,
            PHASE3B_SIR_TIME_ORIGIN,
        )
        if any((
                submission.source_interval_begin_ms,
                submission.source_interval_end_ms,
                submission.sir_time_unit,
                submission.sir_time_origin,
        ) != expected_timing for submission in (primary, reviewer)):
            raise ValueError("adjudication submissions use another timing contract")
        if adjudicator_pseudonym in {
                primary.author_pseudonym, reviewer.author_pseudonym}:
            raise ValueError("adjudicator must be distinct from both submitters")
        if {
                qualification_evidence_sha256,
                independence_evidence_sha256,
                attestation_sha256,
        } & {
                primary.qualification_evidence_sha256,
                primary.independence_evidence_sha256,
                primary.attestation_sha256,
                reviewer.qualification_evidence_sha256,
                reviewer.independence_evidence_sha256,
                reviewer.attestation_sha256,
        }:
            raise ValueError("adjudicator evidence artifacts must be role-distinct")
        payload = (
            None if final_graph is None
            else _canonical_json_bytes(sir_to_dict(final_graph))
        )
        digest = None if final_graph is None else sir_sha256(final_graph)
        if outcome is AdjudicationOutcome.ACCEPT_PRIMARY \
                and digest != primary.sir_payload_sha256:
            raise ValueError("accept_primary must contain the exact primary SIR")
        if outcome is AdjudicationOutcome.ACCEPT_REVIEWER \
                and digest != reviewer.sir_payload_sha256:
            raise ValueError("accept_reviewer must contain the exact reviewer SIR")
        if outcome is AdjudicationOutcome.REVISED and digest in {
                primary.sir_payload_sha256, reviewer.sir_payload_sha256}:
            raise ValueError("revised adjudication must contain a distinct SIR")
        return cls(
            schema_version=PHASE3B_ADJUDICATION_SCHEMA_VERSION,
            adjudication_id=adjudication_id,
            case_id=case_id,
            source_binding_sha256=source.content_sha256(),
            source_interval_begin_ms=source.begin_ms,
            source_interval_end_ms=source.end_ms,
            sir_time_unit=PHASE3B_SIR_TIME_UNIT,
            sir_time_origin=PHASE3B_SIR_TIME_ORIGIN,
            primary_submission_sha256=primary.content_sha256(),
            reviewer_submission_sha256=reviewer.content_sha256(),
            adjudicator_pseudonym=adjudicator_pseudonym,
            adjudicator_qualified_asl=adjudicator_qualified_asl,
            source_video_reviewed=source_video_reviewed,
            independent_of_submitters=independent_of_submitters,
            qualification_evidence_sha256=qualification_evidence_sha256,
            independence_evidence_sha256=independence_evidence_sha256,
            attestation_sha256=attestation_sha256,
            protocol=protocol,
            outcome=outcome,
            final_sir_payload=payload,
            final_sir_payload_sha256=digest,
            submitted_at=submitted_at,
            reason_codes=tuple(reason_codes),
            limitations=tuple(limitations),
        )

    def final_graph(self) -> SIRGraph:
        if self.final_sir_payload is None:
            raise ValueError("non-accepted adjudication has no final SIR")
        value = _strict_json_loads(self.final_sir_payload)
        if not isinstance(value, Mapping):
            raise ValueError("adjudicated SIR root must be an object")
        return sir_from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adjudication_id": self.adjudication_id,
            "case_id": self.case_id,
            "source_binding_sha256": self.source_binding_sha256,
            "source_interval_begin_ms": self.source_interval_begin_ms,
            "source_interval_end_ms": self.source_interval_end_ms,
            "sir_time_unit": self.sir_time_unit,
            "sir_time_origin": self.sir_time_origin,
            "primary_submission_sha256": self.primary_submission_sha256,
            "reviewer_submission_sha256": self.reviewer_submission_sha256,
            "adjudicator_pseudonym": self.adjudicator_pseudonym,
            "adjudicator_qualified_asl": self.adjudicator_qualified_asl,
            "source_video_reviewed": self.source_video_reviewed,
            "independent_of_submitters": self.independent_of_submitters,
            "qualification_evidence_sha256": self.qualification_evidence_sha256,
            "independence_evidence_sha256": self.independence_evidence_sha256,
            "attestation_sha256": self.attestation_sha256,
            "protocol": self.protocol.to_dict(),
            "outcome": self.outcome.value,
            "final_sir": (
                None if self.final_sir_payload is None
                else sir_to_dict(self.final_graph())
            ),
            "final_sir_payload_sha256": self.final_sir_payload_sha256,
            "submitted_at": self.submitted_at,
            "reason_codes": list(self.reason_codes),
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HumanAdjudication":
        required = {
            "schema_version", "adjudication_id", "case_id",
            "source_binding_sha256", "source_interval_begin_ms",
            "source_interval_end_ms", "sir_time_unit", "sir_time_origin",
            "primary_submission_sha256",
            "reviewer_submission_sha256", "adjudicator_pseudonym",
            "adjudicator_qualified_asl", "source_video_reviewed",
            "independent_of_submitters", "qualification_evidence_sha256",
            "independence_evidence_sha256", "attestation_sha256", "protocol",
            "outcome", "final_sir", "final_sir_payload_sha256", "submitted_at",
            "reason_codes", "limitations",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"adjudication fields must be exactly {sorted(required)}")
        if not isinstance(value["protocol"], Mapping):
            raise ValueError("adjudication protocol must be an object")
        if not isinstance(value["reason_codes"], list) \
                or not isinstance(value["limitations"], list):
            raise ValueError("adjudication reasons and limitations must be lists")
        final_sir = value["final_sir"]
        if final_sir is not None and not isinstance(final_sir, Mapping):
            raise ValueError("adjudicated final SIR must be an object or null")
        try:
            outcome = AdjudicationOutcome(value["outcome"])
        except (TypeError, ValueError) as error:
            raise ValueError("unknown adjudication outcome") from error
        return cls(
            schema_version=value["schema_version"],
            adjudication_id=value["adjudication_id"],
            case_id=value["case_id"],
            source_binding_sha256=value["source_binding_sha256"],
            source_interval_begin_ms=value["source_interval_begin_ms"],
            source_interval_end_ms=value["source_interval_end_ms"],
            sir_time_unit=value["sir_time_unit"],
            sir_time_origin=value["sir_time_origin"],
            primary_submission_sha256=value["primary_submission_sha256"],
            reviewer_submission_sha256=value["reviewer_submission_sha256"],
            adjudicator_pseudonym=value["adjudicator_pseudonym"],
            adjudicator_qualified_asl=value["adjudicator_qualified_asl"],
            source_video_reviewed=value["source_video_reviewed"],
            independent_of_submitters=value["independent_of_submitters"],
            qualification_evidence_sha256=value["qualification_evidence_sha256"],
            independence_evidence_sha256=value["independence_evidence_sha256"],
            attestation_sha256=value["attestation_sha256"],
            protocol=GovernedArtifact.from_dict(value["protocol"]),
            outcome=outcome,
            final_sir_payload=(
                None if final_sir is None else _canonical_json_bytes(final_sir)
            ),
            final_sir_payload_sha256=value["final_sir_payload_sha256"],
            submitted_at=value["submitted_at"],
            reason_codes=tuple(value["reason_codes"]),
            limitations=tuple(value["limitations"]),
        )

    def content_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()


@dataclass(frozen=True)
class WorkflowEvent:
    """One content-addressed event in an append-only case history."""

    schema_version: int
    case_id: str
    sequence: int
    state: WorkflowState
    actor_pseudonym: str
    artifact_sha256: str
    previous_event_sha256: str | None
    occurred_at: str

    def __post_init__(self) -> None:
        if (not isinstance(self.schema_version, int)
                or isinstance(self.schema_version, bool)
                or self.schema_version != PHASE3B_EVENT_SCHEMA_VERSION):
            raise ValueError("unsupported Phase 3B event schema")
        _require_id("workflow case_id", self.case_id)
        if (not isinstance(self.sequence, int) or isinstance(self.sequence, bool)
                or self.sequence < 0):
            raise ValueError("workflow sequence must be a non-negative integer")
        if not isinstance(self.state, WorkflowState):
            raise ValueError("workflow state is invalid")
        _require_id("workflow actor_pseudonym", self.actor_pseudonym)
        _require_sha256("workflow artifact_sha256", self.artifact_sha256)
        if self.sequence == 0:
            if self.previous_event_sha256 is not None:
                raise ValueError("first workflow event cannot have a predecessor")
        else:
            _require_sha256("workflow previous_event_sha256",
                            self.previous_event_sha256)
        _require_timestamp("workflow occurred_at", self.occurred_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "sequence": self.sequence,
            "state": self.state.value,
            "actor_pseudonym": self.actor_pseudonym,
            "artifact_sha256": self.artifact_sha256,
            "previous_event_sha256": self.previous_event_sha256,
            "occurred_at": self.occurred_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkflowEvent":
        required = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"workflow event fields must be exactly {sorted(required)}")
        try:
            state = WorkflowState(value["state"])
        except (TypeError, ValueError) as error:
            raise ValueError("unknown workflow event state") from error
        return cls(**{**dict(value), "state": state})

    def content_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()


def _new_event(
    events: tuple[WorkflowEvent, ...],
    *,
    case_id: str,
    state: WorkflowState,
    actor_pseudonym: str,
    artifact_sha256: str,
    occurred_at: str,
) -> WorkflowEvent:
    return WorkflowEvent(
        schema_version=PHASE3B_EVENT_SCHEMA_VERSION,
        case_id=case_id,
        sequence=len(events),
        state=state,
        actor_pseudonym=actor_pseudonym,
        artifact_sha256=artifact_sha256,
        previous_event_sha256=(
            None if not events else events[-1].content_sha256()),
        occurred_at=occurred_at,
    )


def _review_artifact_sha256(
    reviewer: HumanSIRSubmission,
    correspondence: EventCorrespondence,
) -> str:
    if not isinstance(reviewer, HumanSIRSubmission):
        raise TypeError("reviewer must be a human SIR submission")
    if not isinstance(correspondence, EventCorrespondence):
        raise TypeError("correspondence must be an event correspondence")
    value = {
        "reviewer_submission_sha256": reviewer.content_sha256(),
        "event_correspondence": [list(pair) for pair in correspondence.pairs],
    }
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _draft_artifact_sha256(
    *,
    case_id: str,
    source: EAFAnnotationBinding,
    convention: GovernedArtifact,
    lexicon: GovernedArtifact,
    lexicon_convention_sha256: str,
    annotation_protocol: GovernedArtifact,
    review_protocol: GovernedArtifact,
    adjudication_protocol: GovernedArtifact,
    sampling_plan: GovernedArtifact,
    metrics_preregistration: GovernedArtifact,
) -> str:
    specification = {
        "schema_version": PHASE3B_CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "source": source.to_dict(),
        "convention": convention.to_dict(),
        "lexicon": lexicon.to_dict(),
        "lexicon_convention_sha256": lexicon_convention_sha256,
        "annotation_protocol": annotation_protocol.to_dict(),
        "review_protocol": review_protocol.to_dict(),
        "adjudication_protocol": adjudication_protocol.to_dict(),
        "sampling_plan": sampling_plan.to_dict(),
        "metrics_preregistration": metrics_preregistration.to_dict(),
    }
    return hashlib.sha256(_canonical_json_bytes(specification)).hexdigest()


@dataclass(frozen=True)
class Phase3BReviewCase:
    """Immutable snapshot of one governed review case and its full history."""

    schema_version: int
    case_id: str
    source: EAFAnnotationBinding
    convention: GovernedArtifact
    lexicon: GovernedArtifact
    lexicon_convention_sha256: str
    annotation_protocol: GovernedArtifact
    review_protocol: GovernedArtifact
    adjudication_protocol: GovernedArtifact
    sampling_plan: GovernedArtifact
    metrics_preregistration: GovernedArtifact
    creator_pseudonym: str
    created_at: str
    primary: HumanSIRSubmission | None
    reviewer: HumanSIRSubmission | None
    correspondence: EventCorrespondence
    adjudication: HumanAdjudication | None
    terminal_state: WorkflowState | None
    events: tuple[WorkflowEvent, ...]
    source_training_target_authorized: bool = False
    commercial_use_authorized: bool = False
    project_linguistic_validation_complete: bool = False

    def __post_init__(self) -> None:
        if (not isinstance(self.schema_version, int)
                or isinstance(self.schema_version, bool)
                or self.schema_version != PHASE3B_CASE_SCHEMA_VERSION):
            raise ValueError("unsupported Phase 3B case schema")
        _require_id("case_id", self.case_id)
        _require_id("creator_pseudonym", self.creator_pseudonym)
        _require_timestamp("case created_at", self.created_at)
        if not isinstance(self.source, EAFAnnotationBinding):
            raise ValueError("case source must be an EAF annotation binding")
        expected_kinds = (
            (self.convention, ArtifactKind.ASL_CONVENTION),
            (self.lexicon, ArtifactKind.SIR_LEXICON),
            (self.annotation_protocol, ArtifactKind.ANNOTATION_PROTOCOL),
            (self.review_protocol, ArtifactKind.REVIEW_PROTOCOL),
            (self.adjudication_protocol, ArtifactKind.ADJUDICATION_PROTOCOL),
            (self.sampling_plan, ArtifactKind.SAMPLING_PLAN),
            (self.metrics_preregistration, ArtifactKind.METRICS_PREREGISTRATION),
        )
        if any(not isinstance(artifact, GovernedArtifact)
               or artifact.kind is not expected
               for artifact, expected in expected_kinds):
            raise ValueError("case governance artifact has the wrong kind")
        _require_sha256("lexicon_convention_sha256", self.lexicon_convention_sha256)
        if self.lexicon_convention_sha256 != self.convention.sha256:
            raise ValueError("case lexicon is not bound to the declared convention")
        if any(value is not False for value in (
                self.source_training_target_authorized,
                self.commercial_use_authorized,
                self.project_linguistic_validation_complete)):
            raise ValueError(
                "Phase 3B EAF cases cannot claim training, commercial, "
                "or validation approval")
        if not isinstance(self.correspondence, EventCorrespondence):
            raise ValueError("case correspondence must be typed")
        self._validate_participants_and_state()
        self._validate_events()

    def _validate_participants_and_state(self) -> None:
        source_sha = self.source.content_sha256()
        created = _require_timestamp("case created_at", self.created_at)
        if self.primary is None:
            if any(item is not None for item in (
                    self.reviewer, self.adjudication, self.terminal_state)):
                raise ValueError("draft case cannot contain downstream review state")
            if self.correspondence.pairs:
                raise ValueError("draft case cannot contain event correspondence")
            return
        if self.primary.role is not SubmissionRole.PRIMARY:
            raise ValueError("case primary submission has the wrong role")
        if self.primary.case_id != self.case_id:
            raise ValueError("primary submission belongs to another case")
        if self.primary.source_binding_sha256 != source_sha:
            raise ValueError("case primary submission is bound to another source")
        if (self.primary.source_interval_begin_ms != self.source.begin_ms
                or self.primary.source_interval_end_ms != self.source.end_ms
                or self.primary.sir_time_unit != PHASE3B_SIR_TIME_UNIT
                or self.primary.sir_time_origin != PHASE3B_SIR_TIME_ORIGIN):
            raise ValueError("case primary submission has another timing contract")
        if self.primary.protocol != self.annotation_protocol:
            raise ValueError("case primary submission uses another protocol")
        primary_time = _require_timestamp("primary submitted_at", self.primary.submitted_at)
        if primary_time <= created:
            raise ValueError("primary submission must follow case creation")
        if self.reviewer is None:
            if self.adjudication is not None or self.terminal_state is not None:
                raise ValueError("unreviewed case cannot be adjudicated or terminal")
            if self.correspondence.pairs:
                raise ValueError("unreviewed case cannot contain event correspondence")
            return
        if self.reviewer.role is not SubmissionRole.INDEPENDENT_REVIEWER:
            raise ValueError("case reviewer submission has the wrong role")
        if self.reviewer.case_id != self.case_id:
            raise ValueError("reviewer submission belongs to another case")
        if self.reviewer.source_binding_sha256 != source_sha:
            raise ValueError("case reviewer submission is bound to another source")
        if (self.reviewer.source_interval_begin_ms != self.source.begin_ms
                or self.reviewer.source_interval_end_ms != self.source.end_ms
                or self.reviewer.sir_time_unit != PHASE3B_SIR_TIME_UNIT
                or self.reviewer.sir_time_origin != PHASE3B_SIR_TIME_ORIGIN):
            raise ValueError("case reviewer submission has another timing contract")
        if self.reviewer.protocol != self.review_protocol:
            raise ValueError("case reviewer submission uses another protocol")
        if self.primary.author_pseudonym == self.reviewer.author_pseudonym:
            raise ValueError("case primary annotator and reviewer must be distinct")
        if self.primary.submission_id == self.reviewer.submission_id:
            raise ValueError("case submissions must have distinct identifiers")
        reviewer_time = _require_timestamp(
            "reviewer submitted_at", self.reviewer.submitted_at)
        if reviewer_time <= primary_time:
            raise ValueError("review submission must follow primary completion")
        comparison = compare_submissions(
            self.primary, self.reviewer, self.correspondence)
        identical_sir = comparison.exact_sir_hash_match
        both_abstained = (
            self.primary.decision is SubmissionDecision.ABSTAIN
            and self.reviewer.decision is SubmissionDecision.ABSTAIN
        )
        if self.adjudication is None:
            allowed_terminal = (
                WorkflowState.ACCEPTED if identical_sir
                else WorkflowState.ABSTAINED if both_abstained
                else None
            )
            if self.terminal_state not in {None, allowed_terminal}:
                raise ValueError("non-adjudicated case has an invalid terminal state")
            if self.terminal_state is not None and allowed_terminal is None:
                raise ValueError("disagreement requires independent adjudication")
            return

        if identical_sir or both_abstained:
            raise ValueError("identical judgments must not be rewritten by adjudication")
        adjudication = self.adjudication
        if adjudication.case_id != self.case_id \
                or adjudication.source_binding_sha256 != source_sha \
                or adjudication.primary_submission_sha256 \
                != self.primary.content_sha256() \
                or adjudication.reviewer_submission_sha256 \
                != self.reviewer.content_sha256():
            raise ValueError("adjudication is not bound to the exact case submissions")
        if (adjudication.source_interval_begin_ms != self.source.begin_ms
                or adjudication.source_interval_end_ms != self.source.end_ms
                or adjudication.sir_time_unit != PHASE3B_SIR_TIME_UNIT
                or adjudication.sir_time_origin != PHASE3B_SIR_TIME_ORIGIN):
            raise ValueError("case adjudication has another timing contract")
        if adjudication.protocol != self.adjudication_protocol:
            raise ValueError("case adjudication uses another protocol")
        if adjudication.adjudicator_pseudonym in {
                self.primary.author_pseudonym, self.reviewer.author_pseudonym}:
            raise ValueError("adjudicator must be distinct from both submitters")
        if {
                adjudication.qualification_evidence_sha256,
                adjudication.independence_evidence_sha256,
                adjudication.attestation_sha256,
        } & {
                self.primary.qualification_evidence_sha256,
                self.primary.independence_evidence_sha256,
                self.primary.attestation_sha256,
                self.reviewer.qualification_evidence_sha256,
                self.reviewer.independence_evidence_sha256,
                self.reviewer.attestation_sha256,
        }:
            raise ValueError("case adjudicator evidence artifacts must be role-distinct")
        if adjudication.adjudication_id in {
                self.primary.submission_id, self.reviewer.submission_id}:
            raise ValueError("adjudication and submission identifiers must be distinct")
        adjudication_time = _require_timestamp(
            "adjudication submitted_at", adjudication.submitted_at)
        if adjudication_time <= reviewer_time:
            raise ValueError("adjudication must follow blind review")
        if adjudication.outcome is AdjudicationOutcome.ACCEPT_PRIMARY \
                and adjudication.final_sir_payload_sha256 \
                != self.primary.sir_payload_sha256:
            raise ValueError("accept_primary does not contain the primary SIR")
        if adjudication.outcome is AdjudicationOutcome.ACCEPT_REVIEWER \
                and adjudication.final_sir_payload_sha256 \
                != self.reviewer.sir_payload_sha256:
            raise ValueError("accept_reviewer does not contain the reviewer SIR")
        if adjudication.outcome is AdjudicationOutcome.REVISED \
                and adjudication.final_sir_payload_sha256 in {
                    self.primary.sir_payload_sha256,
                    self.reviewer.sir_payload_sha256,
                }:
            raise ValueError("revised adjudication must contain a distinct SIR")
        expected_terminal = {
            AdjudicationOutcome.ACCEPT_PRIMARY: WorkflowState.ACCEPTED,
            AdjudicationOutcome.ACCEPT_REVIEWER: WorkflowState.ACCEPTED,
            AdjudicationOutcome.REVISED: WorkflowState.ACCEPTED,
            AdjudicationOutcome.REJECTED: WorkflowState.REJECTED,
            AdjudicationOutcome.ABSTAINED: WorkflowState.ABSTAINED,
        }[adjudication.outcome]
        if self.terminal_state not in {None, expected_terminal}:
            raise ValueError("terminal state contradicts adjudication outcome")

    @property
    def state(self) -> WorkflowState:
        if self.terminal_state is not None:
            return self.terminal_state
        if self.adjudication is not None:
            return WorkflowState.ADJUDICATED
        if self.reviewer is not None:
            return WorkflowState.BLIND_REVIEWED
        if self.primary is not None:
            return WorkflowState.PRIMARY_COMPLETE
        return WorkflowState.DRAFT

    def _expected_event_records(self) -> tuple[tuple[WorkflowState, str, str, str], ...]:
        records: list[tuple[WorkflowState, str, str, str]] = [(
            WorkflowState.DRAFT,
            self.creator_pseudonym,
            _draft_artifact_sha256(
                case_id=self.case_id,
                source=self.source,
                convention=self.convention,
                lexicon=self.lexicon,
                lexicon_convention_sha256=self.lexicon_convention_sha256,
                annotation_protocol=self.annotation_protocol,
                review_protocol=self.review_protocol,
                adjudication_protocol=self.adjudication_protocol,
                sampling_plan=self.sampling_plan,
                metrics_preregistration=self.metrics_preregistration,
            ),
            self.created_at,
        )]
        if self.primary is not None:
            records.append((
                WorkflowState.PRIMARY_COMPLETE,
                self.primary.author_pseudonym,
                self.primary.content_sha256(),
                self.primary.submitted_at,
            ))
        if self.reviewer is not None:
            records.append((
                WorkflowState.BLIND_REVIEWED,
                self.reviewer.author_pseudonym,
                _review_artifact_sha256(self.reviewer, self.correspondence),
                self.reviewer.submitted_at,
            ))
        if self.adjudication is not None:
            records.append((
                WorkflowState.ADJUDICATED,
                self.adjudication.adjudicator_pseudonym,
                self.adjudication.content_sha256(),
                self.adjudication.submitted_at,
            ))
        if self.terminal_state is not None:
            if self.adjudication is not None:
                actor = self.adjudication.adjudicator_pseudonym
                occurred_at = self.adjudication.submitted_at
                artifact = (
                    self.adjudication.final_sir_payload_sha256
                    if self.terminal_state is WorkflowState.ACCEPTED
                    else self.adjudication.content_sha256()
                )
            else:
                assert self.reviewer is not None
                actor = self.reviewer.author_pseudonym
                occurred_at = self.reviewer.submitted_at
                artifact = (
                    self.reviewer.sir_payload_sha256
                    if self.terminal_state is WorkflowState.ACCEPTED
                    else self.reviewer.content_sha256()
                )
            if artifact is None:
                raise ValueError("terminal workflow event has no content identity")
            records.append((self.terminal_state, actor, artifact, occurred_at))
        return tuple(records)

    def _validate_events(self) -> None:
        if not isinstance(self.events, tuple):
            raise ValueError("workflow events must be an immutable tuple")
        expected = self._expected_event_records()
        if len(self.events) != len(expected):
            raise ValueError("workflow event count does not match case state")
        previous: WorkflowEvent | None = None
        for index, (event, record) in enumerate(zip(self.events, expected, strict=True)):
            state, actor, artifact, occurred_at = record
            if (event.sequence != index or event.state is not state
                    or event.case_id != self.case_id
                    or event.actor_pseudonym != actor
                    or event.artifact_sha256 != artifact
                    or event.occurred_at != occurred_at):
                raise ValueError("workflow event does not match the immutable case history")
            expected_previous = None if previous is None else previous.content_sha256()
            if event.previous_event_sha256 != expected_previous:
                raise ValueError("workflow event hash chain is broken")
            previous = event

    @classmethod
    def create_draft(
        cls,
        *,
        case_id: str,
        source: EAFAnnotationBinding,
        convention: GovernedArtifact,
        lexicon: GovernedArtifact,
        lexicon_convention_sha256: str,
        annotation_protocol: GovernedArtifact,
        review_protocol: GovernedArtifact,
        adjudication_protocol: GovernedArtifact,
        sampling_plan: GovernedArtifact,
        metrics_preregistration: GovernedArtifact,
        creator_pseudonym: str,
        created_at: str,
    ) -> "Phase3BReviewCase":
        draft_sha256 = _draft_artifact_sha256(
            case_id=case_id,
            source=source,
            convention=convention,
            lexicon=lexicon,
            lexicon_convention_sha256=lexicon_convention_sha256,
            annotation_protocol=annotation_protocol,
            review_protocol=review_protocol,
            adjudication_protocol=adjudication_protocol,
            sampling_plan=sampling_plan,
            metrics_preregistration=metrics_preregistration,
        )
        first_event = _new_event(
            (), case_id=case_id, state=WorkflowState.DRAFT,
            actor_pseudonym=creator_pseudonym,
            artifact_sha256=draft_sha256, occurred_at=created_at,
        )
        return cls(
            schema_version=PHASE3B_CASE_SCHEMA_VERSION,
            case_id=case_id,
            source=source,
            convention=convention,
            lexicon=lexicon,
            lexicon_convention_sha256=lexicon_convention_sha256,
            annotation_protocol=annotation_protocol,
            review_protocol=review_protocol,
            adjudication_protocol=adjudication_protocol,
            sampling_plan=sampling_plan,
            metrics_preregistration=metrics_preregistration,
            creator_pseudonym=creator_pseudonym,
            created_at=created_at,
            primary=None,
            reviewer=None,
            correspondence=EventCorrespondence(()),
            adjudication=None,
            terminal_state=None,
            events=(first_event,),
        )

    def with_primary(self, primary: HumanSIRSubmission) -> "Phase3BReviewCase":
        if self.state is not WorkflowState.DRAFT:
            raise ValueError("primary submission can only be added to a draft case")
        if not isinstance(primary, HumanSIRSubmission):
            raise TypeError("primary must be a human SIR submission")
        event = _new_event(
            self.events, case_id=self.case_id,
            state=WorkflowState.PRIMARY_COMPLETE,
            actor_pseudonym=primary.author_pseudonym,
            artifact_sha256=primary.content_sha256(),
            occurred_at=primary.submitted_at,
        )
        return replace(self, primary=primary, events=(*self.events, event))

    def with_reviewer(
        self,
        reviewer: HumanSIRSubmission,
        correspondence: EventCorrespondence,
    ) -> "Phase3BReviewCase":
        if self.state is not WorkflowState.PRIMARY_COMPLETE:
            raise ValueError("review submission requires completed primary work")
        if not isinstance(reviewer, HumanSIRSubmission):
            raise TypeError("reviewer must be a human SIR submission")
        if not isinstance(correspondence, EventCorrespondence):
            raise TypeError("correspondence must be an event correspondence")
        event = _new_event(
            self.events, case_id=self.case_id,
            state=WorkflowState.BLIND_REVIEWED,
            actor_pseudonym=reviewer.author_pseudonym,
            artifact_sha256=_review_artifact_sha256(reviewer, correspondence),
            occurred_at=reviewer.submitted_at,
        )
        return replace(
            self, reviewer=reviewer, correspondence=correspondence,
            events=(*self.events, event),
        )

    def with_adjudication(
        self,
        adjudication: HumanAdjudication,
    ) -> "Phase3BReviewCase":
        if self.state is not WorkflowState.BLIND_REVIEWED:
            raise ValueError("adjudication requires a completed blind review")
        if not isinstance(adjudication, HumanAdjudication):
            raise TypeError("adjudication must be a human adjudication")
        event = _new_event(
            self.events, case_id=self.case_id,
            state=WorkflowState.ADJUDICATED,
            actor_pseudonym=adjudication.adjudicator_pseudonym,
            artifact_sha256=adjudication.content_sha256(),
            occurred_at=adjudication.submitted_at,
        )
        return replace(
            self, adjudication=adjudication, events=(*self.events, event),
        )

    def finalize(self) -> "Phase3BReviewCase":
        if self.state is WorkflowState.BLIND_REVIEWED:
            assert self.primary is not None and self.reviewer is not None
            if (self.primary.decision is SubmissionDecision.SIR
                    and self.reviewer.decision is SubmissionDecision.SIR
                    and self.primary.sir_payload_sha256
                    == self.reviewer.sir_payload_sha256):
                terminal = WorkflowState.ACCEPTED
                artifact = self.reviewer.sir_payload_sha256
            elif (self.primary.decision is SubmissionDecision.ABSTAIN
                  and self.reviewer.decision is SubmissionDecision.ABSTAIN):
                terminal = WorkflowState.ABSTAINED
                artifact = self.reviewer.content_sha256()
            else:
                raise ValueError("non-identical submissions require adjudication")
            actor = self.reviewer.author_pseudonym
            occurred_at = self.reviewer.submitted_at
        elif self.state is WorkflowState.ADJUDICATED:
            assert self.adjudication is not None
            terminal = {
                AdjudicationOutcome.ACCEPT_PRIMARY: WorkflowState.ACCEPTED,
                AdjudicationOutcome.ACCEPT_REVIEWER: WorkflowState.ACCEPTED,
                AdjudicationOutcome.REVISED: WorkflowState.ACCEPTED,
                AdjudicationOutcome.REJECTED: WorkflowState.REJECTED,
                AdjudicationOutcome.ABSTAINED: WorkflowState.ABSTAINED,
            }[self.adjudication.outcome]
            artifact = (
                self.adjudication.final_sir_payload_sha256
                if terminal is WorkflowState.ACCEPTED
                else self.adjudication.content_sha256()
            )
            actor = self.adjudication.adjudicator_pseudonym
            occurred_at = self.adjudication.submitted_at
        else:
            raise ValueError("case is not ready for finalization")
        if artifact is None:
            raise ValueError("finalized case has no terminal artifact identity")
        event = _new_event(
            self.events, case_id=self.case_id, state=terminal,
            actor_pseudonym=actor,
            artifact_sha256=artifact, occurred_at=occurred_at,
        )
        return replace(
            self, terminal_state=terminal, events=(*self.events, event),
        )

    def agreement_report(self) -> SubmissionAgreementReport:
        if self.primary is None or self.reviewer is None:
            raise ValueError("agreement is unavailable before blind review")
        return compare_submissions(
            self.primary, self.reviewer, self.correspondence)

    def final_graph(self) -> SIRGraph:
        if self.state is not WorkflowState.ACCEPTED:
            raise ValueError("only accepted cases have a final SIR")
        assert self.primary is not None and self.reviewer is not None
        if self.adjudication is not None:
            return self.adjudication.final_graph()
        return self.primary.graph()

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "phase3b_governed_source_to_sir_case",
            "case_id": self.case_id,
            "source": self.source.to_dict(),
            "convention": self.convention.to_dict(),
            "lexicon": self.lexicon.to_dict(),
            "lexicon_convention_sha256": self.lexicon_convention_sha256,
            "annotation_protocol": self.annotation_protocol.to_dict(),
            "review_protocol": self.review_protocol.to_dict(),
            "adjudication_protocol": self.adjudication_protocol.to_dict(),
            "sampling_plan": self.sampling_plan.to_dict(),
            "metrics_preregistration": self.metrics_preregistration.to_dict(),
            "creator_pseudonym": self.creator_pseudonym,
            "created_at": self.created_at,
            "primary": None if self.primary is None else self.primary.to_dict(),
            "reviewer": None if self.reviewer is None else self.reviewer.to_dict(),
            "correspondence": [list(pair) for pair in self.correspondence.pairs],
            "adjudication": (
                None if self.adjudication is None else self.adjudication.to_dict()
            ),
            "terminal_state": (
                None if self.terminal_state is None else self.terminal_state.value
            ),
            "events": [event.to_dict() for event in self.events],
            "source_training_target_authorized": (
                self.source_training_target_authorized),
            "commercial_use_authorized": self.commercial_use_authorized,
            "project_linguistic_validation_complete": (
                self.project_linguistic_validation_complete),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_manifest())

    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "Phase3BReviewCase":
        required = {
            "schema_version", "kind", "case_id", "source", "convention",
            "lexicon", "lexicon_convention_sha256", "annotation_protocol",
            "review_protocol", "adjudication_protocol", "sampling_plan",
            "metrics_preregistration", "creator_pseudonym", "created_at",
            "primary", "reviewer", "correspondence", "adjudication",
            "terminal_state", "events", "source_training_target_authorized",
            "commercial_use_authorized", "project_linguistic_validation_complete",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"Phase 3B case fields must be exactly {sorted(required)}")
        if value["kind"] != "phase3b_governed_source_to_sir_case":
            raise ValueError("Phase 3B case kind is invalid")
        object_fields = (
            "source", "convention", "lexicon", "annotation_protocol",
            "review_protocol", "adjudication_protocol", "sampling_plan",
            "metrics_preregistration",
        )
        if any(not isinstance(value[name], Mapping) for name in object_fields):
            raise ValueError("Phase 3B case governance fields must be objects")
        for name in ("primary", "reviewer", "adjudication"):
            if value[name] is not None and not isinstance(value[name], Mapping):
                raise ValueError(f"Phase 3B {name} must be an object or null")
        pairs = value["correspondence"]
        events = value["events"]
        if not isinstance(pairs, list) or not isinstance(events, list):
            raise ValueError("Phase 3B correspondence and events must be lists")
        if any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs):
            raise ValueError("Phase 3B correspondence entries must be pairs")
        if any(not isinstance(event, Mapping) for event in events):
            raise ValueError("Phase 3B workflow events must be objects")
        terminal_value = value["terminal_state"]
        try:
            terminal = (
                None if terminal_value is None else WorkflowState(terminal_value)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("unknown Phase 3B terminal state") from error
        return cls(
            schema_version=value["schema_version"],
            case_id=value["case_id"],
            source=EAFAnnotationBinding.from_dict(value["source"]),
            convention=GovernedArtifact.from_dict(value["convention"]),
            lexicon=GovernedArtifact.from_dict(value["lexicon"]),
            lexicon_convention_sha256=value["lexicon_convention_sha256"],
            annotation_protocol=GovernedArtifact.from_dict(
                value["annotation_protocol"]),
            review_protocol=GovernedArtifact.from_dict(value["review_protocol"]),
            adjudication_protocol=GovernedArtifact.from_dict(
                value["adjudication_protocol"]),
            sampling_plan=GovernedArtifact.from_dict(value["sampling_plan"]),
            metrics_preregistration=GovernedArtifact.from_dict(
                value["metrics_preregistration"]),
            creator_pseudonym=value["creator_pseudonym"],
            created_at=value["created_at"],
            primary=(
                None if value["primary"] is None
                else HumanSIRSubmission.from_dict(value["primary"])
            ),
            reviewer=(
                None if value["reviewer"] is None
                else HumanSIRSubmission.from_dict(value["reviewer"])
            ),
            correspondence=EventCorrespondence(
                tuple(tuple(pair) for pair in pairs)),
            adjudication=(
                None if value["adjudication"] is None
                else HumanAdjudication.from_dict(value["adjudication"])
            ),
            terminal_state=terminal,
            events=tuple(WorkflowEvent.from_dict(event) for event in events),
            source_training_target_authorized=value[
                "source_training_target_authorized"],
            commercial_use_authorized=value["commercial_use_authorized"],
            project_linguistic_validation_complete=value[
                "project_linguistic_validation_complete"],
        )


def load_phase3b_case(
    payload: bytes,
    *,
    max_bytes: int = 16 * 1024 * 1024,
) -> Phase3BReviewCase:
    """Load one canonical case snapshot and revalidate its complete history."""
    value = _strict_json_loads(payload, max_bytes=max_bytes)
    if not isinstance(value, Mapping):
        raise ValueError("Phase 3B case root must be an object")
    case = Phase3BReviewCase.from_manifest(value)
    if case.canonical_bytes() != payload:
        raise ValueError("Phase 3B case bytes are not canonical")
    return case


@dataclass(frozen=True)
class Phase3BBatchReport:
    """Threshold-free accounting and agreement summary for one frozen batch.

    A complete software workflow is not linguistic validation and is never an
    authorization to train on the EAF-derived records.
    """

    schema_version: int
    expected_case_count: int
    expected_case_ids: tuple[str, ...]
    observed_case_count: int
    unique_source_annotation_count: int
    source_manifest_count: int
    state_counts: tuple[tuple[str, int], ...]
    case_sha256_by_id: tuple[tuple[str, str], ...]
    complete_accounting: bool
    all_cases_terminal: bool
    software_workflow_complete: bool
    violations: tuple[str, ...]
    adjudicated_case_count: int
    comparison_available_count: int
    comparison_unavailable_count: int
    exact_sir_match_count: int
    comparable_primary_event_count: int
    comparable_reviewer_event_count: int
    paired_event_count: int
    primary_event_pair_coverage: float | None
    reviewer_event_pair_coverage: float | None
    field_agreement: tuple[FieldAgreement, ...]
    kind_confusion: tuple[tuple[str, str, int], ...]
    label_confusion: tuple[tuple[int, int, int], ...]
    temporal_pair_count: int
    median_temporal_iou: float | None
    median_absolute_onset_difference: float | None
    median_absolute_offset_difference: float | None
    comparable_edge_intersection: int
    comparable_edge_union: int
    comparable_edge_jaccard: float | None
    no_acceptance_threshold_selected: bool = True
    approved_for_research_training: bool = False
    commercial_use_authorized: bool = False
    project_linguistic_validation_complete: bool = False

    def __post_init__(self) -> None:
        if (not isinstance(self.schema_version, int)
                or isinstance(self.schema_version, bool)
                or self.schema_version != PHASE3B_BATCH_REPORT_SCHEMA_VERSION):
            raise ValueError("unsupported Phase 3B batch-report schema")
        count_names = (
            "expected_case_count", "observed_case_count",
            "unique_source_annotation_count", "source_manifest_count",
            "adjudicated_case_count", "comparison_available_count",
            "comparison_unavailable_count", "exact_sir_match_count",
            "comparable_primary_event_count", "comparable_reviewer_event_count",
            "paired_event_count", "temporal_pair_count",
            "comparable_edge_intersection", "comparable_edge_union",
        )
        if any(not isinstance(getattr(self, name), int)
               or isinstance(getattr(self, name), bool)
               or getattr(self, name) < 0 for name in count_names):
            raise ValueError("Phase 3B batch counts must be non-negative integers")
        if self.expected_case_count < 1:
            raise ValueError("Phase 3B batch must declare at least one expected case")
        if (not isinstance(self.expected_case_ids, tuple)
                or len(self.expected_case_ids) != self.expected_case_count
                or any(not isinstance(case_id, str)
                       for case_id in self.expected_case_ids)):
            raise ValueError("expected Phase 3B case index is inconsistent")
        if (tuple(sorted(self.expected_case_ids)) != self.expected_case_ids
                or len(set(self.expected_case_ids)) != self.expected_case_count):
            raise ValueError("expected Phase 3B case index is inconsistent")
        for case_id in self.expected_case_ids:
            _require_id("expected batch case_id", case_id)
        expected_states = tuple(state.value for state in WorkflowState)
        if (not isinstance(self.state_counts, tuple)
                or tuple(name for name, _ in self.state_counts) != expected_states
                or any(not isinstance(count, int) or isinstance(count, bool)
                       or count < 0 for _, count in self.state_counts)
                or sum(count for _, count in self.state_counts)
                != self.observed_case_count):
            raise ValueError("Phase 3B state counts are incomplete or inconsistent")
        if (not isinstance(self.case_sha256_by_id, tuple)
                or len(self.case_sha256_by_id) != self.observed_case_count
                or any(not isinstance(item, tuple) or len(item) != 2
                       or not isinstance(item[0], str) or not isinstance(item[1], str)
                       for item in self.case_sha256_by_id)):
            raise ValueError("Phase 3B case hash index is inconsistent")
        if (tuple(sorted(self.case_sha256_by_id)) != self.case_sha256_by_id
                or len({case_id for case_id, _ in self.case_sha256_by_id})
                != self.observed_case_count):
            raise ValueError("Phase 3B case hash index is inconsistent")
        for case_id, digest in self.case_sha256_by_id:
            _require_id("batch case_id", case_id)
            _require_sha256("batch case sha256", digest)
        expected_complete_accounting = (
            set(self.expected_case_ids)
            == {case_id for case_id, _ in self.case_sha256_by_id}
        )
        if self.complete_accounting is not expected_complete_accounting:
            raise ValueError("complete-accounting claim contradicts the case indexes")
        nonterminal_states = {
            WorkflowState.DRAFT.value,
            WorkflowState.PRIMARY_COMPLETE.value,
            WorkflowState.BLIND_REVIEWED.value,
            WorkflowState.ADJUDICATED.value,
        }
        expected_all_terminal = (
            self.observed_case_count > 0
            and all(count == 0 for state, count in self.state_counts
                    if state in nonterminal_states)
        )
        if self.all_cases_terminal is not expected_all_terminal:
            raise ValueError("all-cases-terminal claim contradicts state counts")
        _validate_limits("Phase 3B batch violations", self.violations, maximum=4096)
        if self.unique_source_annotation_count > self.observed_case_count:
            raise ValueError("unique source count exceeds observed case count")
        if self.source_manifest_count > self.unique_source_annotation_count:
            raise ValueError("source manifest count exceeds source annotation count")
        if self.adjudicated_case_count > self.observed_case_count:
            raise ValueError("adjudicated case count exceeds observed case count")
        if (self.comparison_available_count + self.comparison_unavailable_count
                > self.observed_case_count):
            raise ValueError("comparison count exceeds observed case count")
        counts_by_state = dict(self.state_counts)
        reviewed_count = sum(
            counts_by_state[state.value] for state in (
                WorkflowState.BLIND_REVIEWED,
                WorkflowState.ADJUDICATED,
                WorkflowState.ACCEPTED,
                WorkflowState.REJECTED,
                WorkflowState.ABSTAINED,
            )
        )
        if (self.comparison_available_count
                + self.comparison_unavailable_count != reviewed_count):
            raise ValueError("comparison count contradicts reviewed state count")
        if not (counts_by_state[WorkflowState.ADJUDICATED.value]
                + counts_by_state[WorkflowState.REJECTED.value]
                <= self.adjudicated_case_count <= reviewed_count):
            raise ValueError("adjudication count contradicts workflow states")
        if self.exact_sir_match_count > self.comparison_available_count:
            raise ValueError("exact-match count exceeds available comparisons")
        if self.exact_sir_match_count > self.paired_event_count:
            raise ValueError("exact-match count exceeds paired-event support")
        if self.comparison_available_count > min(
                self.comparable_primary_event_count,
                self.comparable_reviewer_event_count):
            raise ValueError("available comparisons exceed comparable event support")
        if self.paired_event_count > min(
                self.comparable_primary_event_count,
                self.comparable_reviewer_event_count):
            raise ValueError("paired event count exceeds comparable event support")
        self._validate_rate(
            "primary_event_pair_coverage", self.primary_event_pair_coverage,
            self.paired_event_count, self.comparable_primary_event_count)
        self._validate_rate(
            "reviewer_event_pair_coverage", self.reviewer_event_pair_coverage,
            self.paired_event_count, self.comparable_reviewer_event_count)
        expected_fields = ("kind", "label", "referent", "locus")
        if (not isinstance(self.field_agreement, tuple)
                or any(not isinstance(item, FieldAgreement)
                       for item in self.field_agreement)
                or tuple(item.field for item in self.field_agreement)
                != expected_fields):
            raise ValueError("Phase 3B field agreement is incomplete or out of order")
        for item in self.field_agreement:
            if (not isinstance(item.agreements, int)
                    or isinstance(item.agreements, bool)
                    or not isinstance(item.support, int)
                    or isinstance(item.support, bool)
                    or not 0 <= item.agreements <= item.support):
                raise ValueError("Phase 3B field agreement counts are invalid")
            self._validate_rate(
                f"{item.field} agreement", item.rate,
                item.agreements, item.support)
            if item.support != self.paired_event_count:
                raise ValueError("field agreement support must equal paired events")
        self._validate_confusions()
        if self.temporal_pair_count != self.paired_event_count:
            raise ValueError("temporal support must equal paired event support")
        temporal_values = (
            self.median_temporal_iou,
            self.median_absolute_onset_difference,
            self.median_absolute_offset_difference,
        )
        if self.temporal_pair_count == 0:
            if any(value is not None for value in temporal_values):
                raise ValueError("zero temporal support requires unavailable medians")
        else:
            if any(not isinstance(value, float) or not math.isfinite(value)
                   for value in temporal_values):
                raise ValueError("temporal medians require finite binary64 floats")
            if not 0.0 <= self.median_temporal_iou <= 1.0:
                raise ValueError("median temporal IoU must be in [0, 1]")
            if (self.median_absolute_onset_difference < 0.0
                    or self.median_absolute_offset_difference < 0.0):
                raise ValueError("median absolute timing differences cannot be negative")
        if not 0 <= self.comparable_edge_intersection <= self.comparable_edge_union:
            raise ValueError("comparable edge counts are inconsistent")
        self._validate_rate(
            "comparable_edge_jaccard", self.comparable_edge_jaccard,
            self.comparable_edge_intersection, self.comparable_edge_union)
        expected_complete = (
            self.complete_accounting and self.all_cases_terminal
            and not self.violations and self.observed_case_count > 0
        )
        if self.software_workflow_complete is not expected_complete:
            raise ValueError("software workflow completion claim is inconsistent")
        if any(value is not expected for value, expected in (
                (self.no_acceptance_threshold_selected, True),
                (self.approved_for_research_training, False),
                (self.commercial_use_authorized, False),
                (self.project_linguistic_validation_complete, False),
        )):
            raise ValueError("Phase 3B batch contains a prohibited approval claim")

    @staticmethod
    def _validate_rate(
        name: str,
        value: float | None,
        numerator: int,
        denominator: int,
    ) -> None:
        expected = None if denominator == 0 else numerator / denominator
        if value is not None and (
                not isinstance(value, float) or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite float or null")
        if value != expected:
            raise ValueError(f"{name} does not match its numerator and denominator")

    def _validate_confusions(self) -> None:
        if (not isinstance(self.kind_confusion, tuple)
                or any(not isinstance(item, tuple) or len(item) != 3
                       for item in self.kind_confusion)
                or not isinstance(self.label_confusion, tuple)
                or any(not isinstance(item, tuple) or len(item) != 3
                       for item in self.label_confusion)):
            raise ValueError("confusion matrices require canonical triples")
        allowed_kinds = {kind.value for kind in EventKind}
        if any(not isinstance(first, str) or first not in allowed_kinds
               or not isinstance(second, str) or second not in allowed_kinds
               or not isinstance(count, int) or isinstance(count, bool) or count < 1
               for first, second, count in self.kind_confusion):
            raise ValueError("kind confusion entries are invalid")
        if any(not isinstance(first, int) or isinstance(first, bool)
               or first < 0 or not isinstance(second, int)
               or isinstance(second, bool) or second < 0
               or not isinstance(count, int) or isinstance(count, bool) or count < 1
               for first, second, count in self.label_confusion):
            raise ValueError("label confusion entries are invalid")
        if (len({(first, second) for first, second, _ in self.kind_confusion})
                != len(self.kind_confusion)
                or len({(first, second) for first, second, _
                        in self.label_confusion}) != len(self.label_confusion)):
            raise ValueError("confusion matrix cells must be unique")
        if (tuple(sorted(self.kind_confusion)) != self.kind_confusion
                or tuple(sorted(self.label_confusion)) != self.label_confusion
                or sum(count for _, _, count in self.kind_confusion)
                != self.paired_event_count
                or sum(count for _, _, count in self.label_confusion)
                != self.paired_event_count):
            raise ValueError("confusion matrices do not match paired event support")
        kind_matches = sum(
            count for first, second, count in self.kind_confusion
            if first == second
        )
        label_matches = sum(
            count for first, second, count in self.label_confusion
            if first == second
        )
        if (kind_matches != self.field_agreement[0].agreements
                or label_matches != self.field_agreement[1].agreements):
            raise ValueError(
                "confusion matrix diagonals contradict field agreement")
        if self.exact_sir_match_count > min(kind_matches, label_matches):
            raise ValueError(
                "exact SIR matches exceed kind or label agreement support")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "phase3b_threshold_free_batch_report",
            "expected_case_count": self.expected_case_count,
            "expected_case_ids": list(self.expected_case_ids),
            "observed_case_count": self.observed_case_count,
            "unique_source_annotation_count": self.unique_source_annotation_count,
            "source_manifest_count": self.source_manifest_count,
            "state_counts": [
                {"state": state, "count": count}
                for state, count in self.state_counts
            ],
            "case_sha256_by_id": [
                {"case_id": case_id, "sha256": digest}
                for case_id, digest in self.case_sha256_by_id
            ],
            "complete_accounting": self.complete_accounting,
            "all_cases_terminal": self.all_cases_terminal,
            "software_workflow_complete": self.software_workflow_complete,
            "violations": list(self.violations),
            "adjudicated_case_count": self.adjudicated_case_count,
            "comparison_available_count": self.comparison_available_count,
            "comparison_unavailable_count": self.comparison_unavailable_count,
            "exact_sir_match_count": self.exact_sir_match_count,
            "comparable_primary_event_count": self.comparable_primary_event_count,
            "comparable_reviewer_event_count": self.comparable_reviewer_event_count,
            "paired_event_count": self.paired_event_count,
            "primary_event_pair_coverage": self.primary_event_pair_coverage,
            "reviewer_event_pair_coverage": self.reviewer_event_pair_coverage,
            "field_agreement": [
                {
                    "field": item.field,
                    "agreements": item.agreements,
                    "support": item.support,
                    "rate": item.rate,
                    "primary_present": item.primary_present,
                    "reviewer_present": item.reviewer_present,
                    "both_present": item.both_present,
                    "both_absent": item.both_absent,
                    "co_present_agreements": item.co_present_agreements,
                    "co_present_rate": item.co_present_rate,
                }
                for item in self.field_agreement
            ],
            "kind_confusion": [list(item) for item in self.kind_confusion],
            "label_confusion": [list(item) for item in self.label_confusion],
            "temporal_pair_count": self.temporal_pair_count,
            "median_temporal_iou": self.median_temporal_iou,
            "median_absolute_onset_difference": (
                self.median_absolute_onset_difference),
            "median_absolute_offset_difference": (
                self.median_absolute_offset_difference),
            "comparable_edge_intersection": self.comparable_edge_intersection,
            "comparable_edge_union": self.comparable_edge_union,
            "comparable_edge_jaccard": self.comparable_edge_jaccard,
            "no_acceptance_threshold_selected": (
                self.no_acceptance_threshold_selected),
            "approved_for_research_training": self.approved_for_research_training,
            "commercial_use_authorized": self.commercial_use_authorized,
            "project_linguistic_validation_complete": (
                self.project_linguistic_validation_complete),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())

    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def require_software_workflow_complete(self) -> None:
        if not self.software_workflow_complete:
            raise RuntimeError(
                f"Phase 3B software workflow is incomplete: {self.violations}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Phase3BBatchReport":
        required = {
            "schema_version", "kind", "expected_case_count",
            "expected_case_ids", "observed_case_count",
            "unique_source_annotation_count", "source_manifest_count",
            "state_counts", "case_sha256_by_id", "complete_accounting",
            "all_cases_terminal", "software_workflow_complete", "violations",
            "adjudicated_case_count", "comparison_available_count",
            "comparison_unavailable_count", "exact_sir_match_count",
            "comparable_primary_event_count", "comparable_reviewer_event_count",
            "paired_event_count", "primary_event_pair_coverage",
            "reviewer_event_pair_coverage", "field_agreement", "kind_confusion",
            "label_confusion", "temporal_pair_count", "median_temporal_iou",
            "median_absolute_onset_difference",
            "median_absolute_offset_difference", "comparable_edge_intersection",
            "comparable_edge_union", "comparable_edge_jaccard",
            "no_acceptance_threshold_selected", "approved_for_research_training",
            "commercial_use_authorized", "project_linguistic_validation_complete",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                f"Phase 3B batch-report fields must be exactly {sorted(required)}")
        if value["kind"] != "phase3b_threshold_free_batch_report":
            raise ValueError("Phase 3B batch-report kind is invalid")
        list_fields = (
            "expected_case_ids", "state_counts", "case_sha256_by_id",
            "violations", "field_agreement", "kind_confusion", "label_confusion",
        )
        if any(not isinstance(value[name], list) for name in list_fields):
            raise ValueError("Phase 3B batch-report collection fields must be lists")
        state_counts: list[tuple[str, int]] = []
        for item in value["state_counts"]:
            if not isinstance(item, Mapping) or set(item) != {"state", "count"}:
                raise ValueError("Phase 3B state-count entry is invalid")
            state_counts.append((item["state"], item["count"]))
        case_hashes: list[tuple[str, str]] = []
        for item in value["case_sha256_by_id"]:
            if not isinstance(item, Mapping) or set(item) != {"case_id", "sha256"}:
                raise ValueError("Phase 3B case-hash entry is invalid")
            case_hashes.append((item["case_id"], item["sha256"]))
        fields: list[FieldAgreement] = []
        for item in value["field_agreement"]:
            if (not isinstance(item, Mapping)
                    or set(item) != {
                        "field", "agreements", "support", "rate",
                        "primary_present", "reviewer_present", "both_present",
                        "both_absent", "co_present_agreements", "co_present_rate",
                    }):
                raise ValueError("Phase 3B field-agreement entry is invalid")
            fields.append(FieldAgreement(**dict(item)))

        def triples(name: str) -> tuple[tuple[Any, Any, Any], ...]:
            collection = value[name]
            if any(not isinstance(item, list) or len(item) != 3
                   for item in collection):
                raise ValueError(f"Phase 3B {name} entries must be triples")
            return tuple(tuple(item) for item in collection)

        return cls(
            schema_version=value["schema_version"],
            expected_case_count=value["expected_case_count"],
            expected_case_ids=tuple(value["expected_case_ids"]),
            observed_case_count=value["observed_case_count"],
            unique_source_annotation_count=value["unique_source_annotation_count"],
            source_manifest_count=value["source_manifest_count"],
            state_counts=tuple(state_counts),
            case_sha256_by_id=tuple(case_hashes),
            complete_accounting=value["complete_accounting"],
            all_cases_terminal=value["all_cases_terminal"],
            software_workflow_complete=value["software_workflow_complete"],
            violations=tuple(value["violations"]),
            adjudicated_case_count=value["adjudicated_case_count"],
            comparison_available_count=value["comparison_available_count"],
            comparison_unavailable_count=value["comparison_unavailable_count"],
            exact_sir_match_count=value["exact_sir_match_count"],
            comparable_primary_event_count=value[
                "comparable_primary_event_count"],
            comparable_reviewer_event_count=value[
                "comparable_reviewer_event_count"],
            paired_event_count=value["paired_event_count"],
            primary_event_pair_coverage=value["primary_event_pair_coverage"],
            reviewer_event_pair_coverage=value["reviewer_event_pair_coverage"],
            field_agreement=tuple(fields),
            kind_confusion=triples("kind_confusion"),
            label_confusion=triples("label_confusion"),
            temporal_pair_count=value["temporal_pair_count"],
            median_temporal_iou=value["median_temporal_iou"],
            median_absolute_onset_difference=value[
                "median_absolute_onset_difference"],
            median_absolute_offset_difference=value[
                "median_absolute_offset_difference"],
            comparable_edge_intersection=value["comparable_edge_intersection"],
            comparable_edge_union=value["comparable_edge_union"],
            comparable_edge_jaccard=value["comparable_edge_jaccard"],
            no_acceptance_threshold_selected=value[
                "no_acceptance_threshold_selected"],
            approved_for_research_training=value[
                "approved_for_research_training"],
            commercial_use_authorized=value["commercial_use_authorized"],
            project_linguistic_validation_complete=value[
                "project_linguistic_validation_complete"],
        )


def audit_phase3b_batch(
    cases: Sequence[Phase3BReviewCase],
    expected_case_ids: Sequence[str],
) -> Phase3BBatchReport:
    """Reconcile a frozen case batch and pool only explicitly supported metrics."""
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise TypeError("cases must be a bounded sequence of Phase 3B cases")
    if (isinstance(expected_case_ids, (str, bytes))
            or not isinstance(expected_case_ids, Sequence)):
        raise TypeError("expected_case_ids must be a bounded sequence of identifiers")
    if len(cases) > 1_000_000 or not 0 < len(expected_case_ids) <= 1_000_000:
        raise ValueError("Phase 3B batch exceeds its declared case-count bounds")
    if any(not isinstance(case, Phase3BReviewCase) for case in cases):
        raise TypeError("batch contains a non-Phase3BReviewCase value")
    expected_ids = tuple(expected_case_ids)
    for case_id in expected_ids:
        _require_id("expected case_id", case_id)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected Phase 3B case identifiers must be unique")
    observed_ids = [case.case_id for case in cases]
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError("observed Phase 3B case identifiers must be unique")
    source_annotation_ids = [
        (
            case.source.eaf_file_sha256,
            case.source.tier_id,
            case.source.source_annotation_id,
        )
        for case in cases
    ]
    if len(set(source_annotation_ids)) != len(source_annotation_ids):
        raise ValueError("Phase 3B source annotations must not be double-counted")

    expected_set = set(expected_ids)
    observed_set = set(observed_ids)
    violations = [
        *(f"missing_case:{case_id}" for case_id in sorted(
            expected_set - observed_set)),
        *(f"unexpected_case:{case_id}" for case_id in sorted(
            observed_set - expected_set)),
        *(f"nonterminal_case:{case.case_id}" for case in sorted(
            cases, key=lambda item: item.case_id) if not case.state.terminal),
    ]
    governance = {
        "asl_convention": {case.convention for case in cases},
        "sir_lexicon": {case.lexicon for case in cases},
        "annotation_protocol": {case.annotation_protocol for case in cases},
        "review_protocol": {case.review_protocol for case in cases},
        "adjudication_protocol": {case.adjudication_protocol for case in cases},
        "sampling_plan": {case.sampling_plan for case in cases},
        "metrics_preregistration": {
            case.metrics_preregistration for case in cases},
    }
    violations.extend(
        f"mixed_governance:{name}"
        for name, identities in governance.items() if len(identities) > 1
    )
    violations_tuple = tuple(sorted(violations))
    complete_accounting = expected_set == observed_set
    all_terminal = bool(cases) and all(case.state.terminal for case in cases)

    reports = [
        case.agreement_report() for case in cases if case.reviewer is not None
    ]
    available = [report for report in reports if report.comparison_available]
    field_count_names = (
        "agreements", "support", "primary_present", "reviewer_present",
        "both_present", "both_absent", "co_present_agreements",
    )
    field_counts = {
        name: {count_name: 0 for count_name in field_count_names}
        for name in ("kind", "label", "referent", "locus")
    }
    kind_confusion: Counter[tuple[str, str]] = Counter()
    label_confusion: Counter[tuple[int, int]] = Counter()
    temporal_ious: list[float] = []
    onset_differences: list[float] = []
    offset_differences: list[float] = []
    for report in available:
        for item in report.field_agreement:
            for count_name in field_count_names:
                field_counts[item.field][count_name] += getattr(item, count_name)
        for first, second, count in report.kind_confusion:
            kind_confusion[(first, second)] += count
        for first, second, count in report.label_confusion:
            label_confusion[(first, second)] += count
        temporal_ious.extend(item.temporal_iou for item in report.temporal)
        onset_differences.extend(
            abs(item.reviewer_minus_primary_onset) for item in report.temporal)
        offset_differences.extend(
            abs(item.reviewer_minus_primary_offset) for item in report.temporal)

    primary_events = sum(report.primary_event_count for report in available)
    reviewer_events = sum(report.reviewer_event_count for report in available)
    paired_events = sum(report.paired_event_count for report in available)
    edge_intersection = sum(
        report.comparable_edge_intersection for report in available)
    edge_union = sum(report.comparable_edge_union for report in available)
    state_counter = Counter(case.state.value for case in cases)
    return Phase3BBatchReport(
        schema_version=PHASE3B_BATCH_REPORT_SCHEMA_VERSION,
        expected_case_count=len(expected_ids),
        expected_case_ids=tuple(sorted(expected_ids)),
        observed_case_count=len(cases),
        unique_source_annotation_count=len(set(source_annotation_ids)),
        source_manifest_count=len({
            case.source.eaf_manifest_sha256 for case in cases}),
        state_counts=tuple(
            (state.value, state_counter[state.value]) for state in WorkflowState),
        case_sha256_by_id=tuple(sorted(
            (case.case_id, case.content_sha256()) for case in cases)),
        complete_accounting=complete_accounting,
        all_cases_terminal=all_terminal,
        software_workflow_complete=(
            complete_accounting and all_terminal and not violations_tuple),
        violations=violations_tuple,
        adjudicated_case_count=sum(
            case.adjudication is not None for case in cases),
        comparison_available_count=len(available),
        comparison_unavailable_count=len(reports) - len(available),
        exact_sir_match_count=sum(
            report.exact_sir_hash_match for report in available),
        comparable_primary_event_count=primary_events,
        comparable_reviewer_event_count=reviewer_events,
        paired_event_count=paired_events,
        primary_event_pair_coverage=(
            None if primary_events == 0 else paired_events / primary_events),
        reviewer_event_pair_coverage=(
            None if reviewer_events == 0 else paired_events / reviewer_events),
        field_agreement=tuple(
            FieldAgreement(
                field=name,
                agreements=field_counts[name]["agreements"],
                support=field_counts[name]["support"],
                rate=(None if field_counts[name]["support"] == 0
                      else field_counts[name]["agreements"]
                      / field_counts[name]["support"]),
                primary_present=field_counts[name]["primary_present"],
                reviewer_present=field_counts[name]["reviewer_present"],
                both_present=field_counts[name]["both_present"],
                both_absent=field_counts[name]["both_absent"],
                co_present_agreements=(
                    field_counts[name]["co_present_agreements"]),
                co_present_rate=(
                    None if field_counts[name]["both_present"] == 0
                    else field_counts[name]["co_present_agreements"]
                    / field_counts[name]["both_present"]),
            )
            for name in ("kind", "label", "referent", "locus")
        ),
        kind_confusion=tuple(
            (first, second, count)
            for (first, second), count in sorted(kind_confusion.items())),
        label_confusion=tuple(
            (first, second, count)
            for (first, second), count in sorted(label_confusion.items())),
        temporal_pair_count=len(temporal_ious),
        median_temporal_iou=(
            None if not temporal_ious else _finite_nonnegative_median(temporal_ious)),
        median_absolute_onset_difference=(
            None if not onset_differences
            else _finite_nonnegative_median(onset_differences)),
        median_absolute_offset_difference=(
            None if not offset_differences
            else _finite_nonnegative_median(offset_differences)),
        comparable_edge_intersection=edge_intersection,
        comparable_edge_union=edge_union,
        comparable_edge_jaccard=(
            None if edge_union == 0 else edge_intersection / edge_union),
    )


def load_phase3b_batch_report(
    payload: bytes,
    *,
    max_bytes: int = 64 * 1024 * 1024,
) -> Phase3BBatchReport:
    """Load a canonical aggregate; case verification remains a separate step."""
    value = _strict_json_loads(payload, max_bytes=max_bytes)
    if not isinstance(value, Mapping):
        raise ValueError("Phase 3B batch-report root must be an object")
    report = Phase3BBatchReport.from_dict(value)
    if report.canonical_bytes() != payload:
        raise ValueError("Phase 3B batch-report bytes are not canonical")
    return report


def verify_phase3b_batch_report(
    report: Phase3BBatchReport,
    cases: Sequence[Phase3BReviewCase],
) -> None:
    """Recompute a loaded report from the exact case snapshots it identifies."""
    if not isinstance(report, Phase3BBatchReport):
        raise TypeError("report must be a Phase3BBatchReport")
    recomputed = audit_phase3b_batch(cases, report.expected_case_ids)
    if recomputed.canonical_bytes() != report.canonical_bytes():
        raise RuntimeError("Phase 3B batch report does not match its case snapshots")


__all__ = [
    "PHASE3B_SOURCE_BINDING_SCHEMA_VERSION", "PHASE3B_SUBMISSION_SCHEMA_VERSION",
    "PHASE3B_SIR_TIME_UNIT", "PHASE3B_SIR_TIME_ORIGIN",
    "EAFAnnotationBinding", "EAFSourceCatalog", "load_eaf_source_catalog",
    "SubmissionRole", "SubmissionDecision", "ABSTENTION_REASON_CODES",
    "HumanSIRSubmission", "EventCorrespondence", "TemporalAgreement",
    "FieldAgreement", "SubmissionAgreementReport", "temporal_iou",
    "compare_submissions", "PHASE3B_ADJUDICATION_SCHEMA_VERSION",
    "PHASE3B_CASE_SCHEMA_VERSION", "PHASE3B_EVENT_SCHEMA_VERSION",
    "AdjudicationOutcome", "WorkflowState", "ADJUDICATION_REASON_CODES",
    "HumanAdjudication", "WorkflowEvent", "Phase3BReviewCase",
    "load_phase3b_case", "PHASE3B_BATCH_REPORT_SCHEMA_VERSION",
    "Phase3BBatchReport", "audit_phase3b_batch", "load_phase3b_batch_report",
    "verify_phase3b_batch_report",
]
