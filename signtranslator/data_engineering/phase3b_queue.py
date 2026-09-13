"""Immutable Phase 3B review-queue preparation without linguistic labels.

The queue binds exact publisher-native annotations, media, and governance
artifacts.  It emits only draft review cases.  No source string is converted to
SIR, gloss, a lexeme identifier, or a training target.
"""

from __future__ import annotations

import hashlib
import html
import json
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..planning.adjudication import (
    EAFSourceCatalog,
    Phase3BReviewCase,
    load_eaf_source_catalog,
    load_phase3b_case,
)
from ..planning.supervision import ArtifactKind, GovernedArtifact
from ..reproducibility import canonical_json_bytes, package_implementation_identity
from .eaf import load_eaf_manifest
from .how2sign_audit import StableFileDigest, assert_file_unchanged, stable_sha256


PHASE3B_QUEUE_SCHEMA_VERSION = 1
PHASE3B_SAMPLING_PLAN_SCHEMA_VERSION = 1
_MAX_ARTIFACT_INDEX_BYTES = 1024 * 1024
_MAX_QUEUE_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_SAMPLING_PLAN_BYTES = 16 * 1024 * 1024
_MAX_CASE_JSON_BYTES = 16 * 1024 * 1024
_MAX_CASE_JSONL_BYTES = 512 * 1024 * 1024
_MAX_REVIEW_HTML_BYTES = 256 * 1024 * 1024
_QUEUE_FILES = frozenset({
    "artifact-index.json",
    "phase3b-draft-cases.jsonl",
    "phase3b-review-queue.html",
    "queue-manifest.json",
})
_QUEUE_PAYLOAD_LIMITS = {
    "phase3b-draft-cases.jsonl": _MAX_CASE_JSONL_BYTES,
    "phase3b-review-queue.html": _MAX_REVIEW_HTML_BYTES,
    "queue-manifest.json": _MAX_QUEUE_MANIFEST_BYTES,
}
_REQUIRED_GOVERNANCE_KINDS = (
    ArtifactKind.ASL_CONVENTION,
    ArtifactKind.SIR_LEXICON,
    ArtifactKind.ANNOTATION_PROTOCOL,
    ArtifactKind.REVIEW_PROTOCOL,
    ArtifactKind.ADJUDICATION_PROTOCOL,
    ArtifactKind.SAMPLING_PLAN,
    ArtifactKind.METRICS_PREREGISTRATION,
)


def _require_timestamp(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("queue timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("queue timestamp must be an ISO-8601 string") from error
    if parsed.tzinfo is None:
        raise ValueError("queue timestamp must include a timezone")


def _require_text(name: str, value: object, *, maximum: int = 16_384) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be non-empty bounded text")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{name} must use NFC-normalized Unicode")
    if any(unicodedata.category(character) in {"Cc", "Cs"}
           for character in value):
        raise ValueError(f"{name} contains a forbidden control or surrogate")
    return value


def _strict_json(payload: bytes, *, max_bytes: int = 64 * 1024 * 1024) -> Any:
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("queue JSON must be non-empty immutable bytes")
    if (not isinstance(max_bytes, int) or isinstance(max_bytes, bool)
            or max_bytes < 1 or len(payload) > max_bytes):
        raise ValueError("queue JSON exceeds the byte limit")

    def unique_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate queue JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite queue JSON value is forbidden: {value}")

    try:
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise ValueError(f"malformed queue JSON: {error}") from error


def _read_bounded_file(path: Path, *, max_bytes: int, name: str) -> bytes:
    if (not isinstance(max_bytes, int) or isinstance(max_bytes, bool)
            or max_bytes < 1):
        raise ValueError("file byte limit must be a positive integer")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a non-symlink regular file")
    before = path.stat()
    if before.st_size > max_bytes:
        raise ValueError(f"{name} exceeds the byte limit")
    with path.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    after = path.stat()
    before_identity = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or len(payload) != after.st_size:
        raise RuntimeError(f"{name} changed while being read")
    if len(payload) > max_bytes:
        raise ValueError(f"{name} exceeds the byte limit")
    return payload


def _relative_path(value: object) -> str:
    path_text = _require_text("relative_path", value)
    if "\\" in path_text:
        raise ValueError("governance relative path must use POSIX separators")
    path = Path(path_text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("governance relative path escapes its root")
    return path.as_posix()


@dataclass(frozen=True)
class GovernanceFileSpec:
    """Operator-declared identity and location of one governance document."""

    kind: ArtifactKind
    artifact_id: str
    version: str
    relative_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ArtifactKind):
            raise ValueError("governance kind must be typed")
        object.__setattr__(self, "relative_path", _relative_path(self.relative_path))
        GovernedArtifact(self.kind, self.artifact_id, self.version, "0" * 64)

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GovernanceFileSpec":
        required = {"kind", "artifact_id", "version", "relative_path"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("governance file specification fields are invalid")
        try:
            kind = ArtifactKind(value["kind"])
        except (TypeError, ValueError) as error:
            raise ValueError("unknown governance file kind") from error
        return cls(
            kind=kind,
            artifact_id=value["artifact_id"],
            version=value["version"],
            relative_path=value["relative_path"],
        )


@dataclass(frozen=True)
class QueueSelection:
    """Explicit source annotation and media view selected by a sampling plan."""

    tier_id: str
    source_annotation_id: str
    media_descriptor_order: int = 0

    def __post_init__(self) -> None:
        _require_text("tier_id", self.tier_id)
        _require_text("source_annotation_id", self.source_annotation_id)
        if (not isinstance(self.media_descriptor_order, int)
                or isinstance(self.media_descriptor_order, bool)
                or self.media_descriptor_order < 0):
            raise ValueError("media_descriptor_order must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_id": self.tier_id,
            "source_annotation_id": self.source_annotation_id,
            "media_descriptor_order": self.media_descriptor_order,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QueueSelection":
        required = {"tier_id", "source_annotation_id", "media_descriptor_order"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("queue selection fields are invalid")
        return cls(**dict(value))


@dataclass(frozen=True)
class Phase3BQueueReceipt:
    """Trusted root identity returned when a queue is atomically published."""

    path: Path
    queue_manifest_sha256: str
    case_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("queue receipt path must be absolute")
        if (not isinstance(self.queue_manifest_sha256, str)
                or len(self.queue_manifest_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.queue_manifest_sha256)):
            raise ValueError("queue receipt manifest identity must be lowercase SHA-256")
        if (not isinstance(self.case_count, int) or isinstance(self.case_count, bool)
                or self.case_count < 1):
            raise ValueError("queue receipt case count must be a positive integer")


def _resolve_root(root: str | Path, name: str) -> Path:
    path = Path(root)
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{name} must be a directory")
    return resolved


def _verify_governance(
    root: Path,
    specs: Sequence[GovernanceFileSpec],
) -> tuple[
    dict[ArtifactKind, GovernedArtifact],
    list[dict[str, Any]],
    list[tuple[Path, StableFileDigest]],
]:
    if (isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence)
            or len(specs) != len(_REQUIRED_GOVERNANCE_KINDS)
            or any(not isinstance(spec, GovernanceFileSpec) for spec in specs)):
        raise ValueError("queue requires exactly seven typed governance file specs")
    kinds = [spec.kind for spec in specs]
    paths = [spec.relative_path for spec in specs]
    if set(kinds) != set(_REQUIRED_GOVERNANCE_KINDS) or len(set(kinds)) != len(kinds):
        raise ValueError("queue governance kinds are missing or duplicated")
    if len(set(paths)) != len(paths):
        raise ValueError("queue governance files must be distinct")
    discovered: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlinked governance entry is forbidden: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported governance entry: {path}")
        discovered.add(path.relative_to(root).as_posix())
        if len(discovered) > 64:
            raise ValueError("governance directory exceeds its file-count limit")
    if discovered != set(paths):
        raise ValueError("governance directory inventory differs from its specification")

    artifacts: dict[ArtifactKind, GovernedArtifact] = {}
    records: list[dict[str, Any]] = []
    observed_files: list[tuple[Path, StableFileDigest]] = []
    for spec in sorted(specs, key=lambda item: item.kind.value):
        path = root / spec.relative_path
        digest = stable_sha256(path, root)
        artifact = GovernedArtifact(
            spec.kind, spec.artifact_id, spec.version, digest.sha256)
        artifacts[spec.kind] = artifact
        records.append({
            "spec": spec.to_dict(),
            "sha256": digest.sha256,
            "size": digest.size,
        })
        observed_files.append((path.resolve(strict=True), digest))
    return artifacts, records, observed_files


def _validate_selections(
    selections: Sequence[QueueSelection],
) -> tuple[QueueSelection, ...]:
    if (isinstance(selections, (str, bytes)) or not isinstance(selections, Sequence)
            or not 0 < len(selections) <= 100_000
            or any(not isinstance(item, QueueSelection) for item in selections)):
        raise ValueError("queue selections must be a non-empty bounded sequence")
    source_records = [
        (item.tier_id, item.source_annotation_id)
        for item in selections
    ]
    if len(set(source_records)) != len(source_records):
        raise ValueError(
            "queue selections must not repeat a source annotation across media views")
    return tuple(sorted(
        selections,
        key=lambda item: (
            item.tier_id, item.source_annotation_id, item.media_descriptor_order),
    ))


def _validate_sampling_plan(
    payload: bytes,
    *,
    source_manifest_sha256: str,
    selections: tuple[QueueSelection, ...],
) -> None:
    value = _strict_json(payload, max_bytes=16 * 1024 * 1024)
    required = {"schema_version", "kind", "source_manifest_sha256", "selections"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Phase 3B sampling plan schema is invalid")
    if (value["schema_version"] != PHASE3B_SAMPLING_PLAN_SCHEMA_VERSION
            or value["kind"] != "phase3b_exact_source_sampling_plan"):
        raise ValueError("Phase 3B sampling plan identity is invalid")
    if canonical_json_bytes(value) != payload:
        raise ValueError("Phase 3B sampling plan must be canonical JSON")
    if value["source_manifest_sha256"] != source_manifest_sha256:
        raise ValueError("Phase 3B sampling plan binds another source manifest")
    if not isinstance(value["selections"], list):
        raise ValueError("Phase 3B sampling plan selections must be a list")
    planned: list[QueueSelection] = []
    for item in value["selections"]:
        if not isinstance(item, Mapping):
            raise ValueError("Phase 3B sampling plan selection must be an object")
        planned.append(QueueSelection.from_dict(item))
    canonical_planned = _validate_selections(planned)
    if [item.to_dict() for item in canonical_planned] != value["selections"]:
        raise ValueError("Phase 3B sampling plan selections are not canonically ordered")
    if canonical_planned != selections:
        raise ValueError("queue selections differ from the frozen sampling plan")


def _draft_cases(
    catalog: EAFSourceCatalog,
    selections: tuple[QueueSelection, ...],
    artifacts: Mapping[ArtifactKind, GovernedArtifact],
    creator_pseudonym: str,
    created_at: str,
) -> tuple[Phase3BReviewCase, ...]:
    cases: list[Phase3BReviewCase] = []
    for selection in selections:
        source = catalog.bind(
            tier_id=selection.tier_id,
            source_annotation_id=selection.source_annotation_id,
            media_descriptor_order=selection.media_descriptor_order,
        )
        case_id = f"p3b:{source.content_sha256()}"
        cases.append(Phase3BReviewCase.create_draft(
            case_id=case_id,
            source=source,
            convention=artifacts[ArtifactKind.ASL_CONVENTION],
            lexicon=artifacts[ArtifactKind.SIR_LEXICON],
            lexicon_convention_sha256=artifacts[
                ArtifactKind.ASL_CONVENTION].sha256,
            annotation_protocol=artifacts[ArtifactKind.ANNOTATION_PROTOCOL],
            review_protocol=artifacts[ArtifactKind.REVIEW_PROTOCOL],
            adjudication_protocol=artifacts[
                ArtifactKind.ADJUDICATION_PROTOCOL],
            sampling_plan=artifacts[ArtifactKind.SAMPLING_PLAN],
            metrics_preregistration=artifacts[
                ArtifactKind.METRICS_PREREGISTRATION],
            creator_pseudonym=creator_pseudonym,
            created_at=created_at,
        ))
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("source selections produced duplicate review case identities")
    return tuple(sorted(cases, key=lambda case: case.case_id))


def _media_paths(manifest: Mapping[str, Any], source_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for binding in manifest["media_bindings"]:
        relative = _relative_path(binding["file"]["relative_path"])
        path = source_root / relative
        if path.is_symlink():
            raise ValueError("queue media path must not be a symlink")
        resolved = path.resolve(strict=True)
        if source_root not in resolved.parents or not resolved.is_file():
            raise ValueError("queue media path escapes the source root")
        paths.append(resolved)
    return tuple(paths)


def _render_html(
    cases: Sequence[Phase3BReviewCase],
    media: tuple[Path, ...],
) -> bytes:
    rows: list[str] = []
    for case in cases:
        source = case.source
        video_uri = media[source.media_descriptor_order].as_uri()
        interval = (
            "unavailable" if source.begin_ms is None or source.end_ms is None
            else f"{source.begin_ms}&ndash;{source.end_ms} ms"
        )
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(case.case_id)}</code></td>"
            f"<td>{html.escape(source.tier_id)}</td>"
            f"<td>{html.escape(source.source_annotation_id)}</td>"
            f"<td>{interval}</td>"
            f"<td>{html.escape(source.source_value)}</td>"
            f"<td><a href=\"{html.escape(video_uri, quote=True)}\">source video</a></td>"
            "<td>empty; qualified human entry required</td>"
            "</tr>"
        )
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Phase 3B governed review queue</title>
<style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #bbb;padding:.4rem;text-align:left;vertical-align:top}
code{overflow-wrap:anywhere}.warning{padding:1rem;background:#fff3cd}</style></head><body>
<h1>Phase 3B governed review queue</h1>
<p class="warning">Publisher-native source text is evidence, not project SIR. No project
label is pre-populated. This queue is not authorized for training, commercial use, or a
claim of linguistic validation.</p>
<table><thead><tr><th>Case</th><th>Tier</th><th>Source annotation</th><th>Interval</th>
<th>Publisher value</th><th>Media</th><th>Project SIR</th></tr></thead>
<tbody>__ROWS__</tbody></table></body></html>
""".replace("__ROWS__", "".join(rows))
    return document.encode("utf-8")


def _write_html(
    path: Path,
    cases: Sequence[Phase3BReviewCase],
    media: tuple[Path, ...],
) -> None:
    path.write_bytes(_render_html(cases, media))


def write_phase3b_review_queue(
    *,
    source_manifest_payload: bytes,
    source_root: str | Path,
    governance_root: str | Path,
    governance_specs: Sequence[GovernanceFileSpec],
    selections: Sequence[QueueSelection],
    output_dir: str | Path,
    creator_pseudonym: str,
    created_at: str,
) -> Phase3BQueueReceipt:
    """Atomically write a review-ready, label-empty Phase 3B queue bundle."""
    _require_text("creator_pseudonym", creator_pseudonym, maximum=256)
    _require_timestamp(created_at)
    source = _resolve_root(source_root, "source_root")
    governance = _resolve_root(governance_root, "governance_root")
    if (source == governance or source in governance.parents
            or governance in source.parents):
        raise ValueError("source and governance roots must be disjoint")
    catalog = load_eaf_source_catalog(source_manifest_payload, source)
    source_manifest = load_eaf_manifest(source_manifest_payload)
    artifacts, governance_records, observed_files = _verify_governance(
        governance, governance_specs)
    selected = _validate_selections(selections)
    sampling_spec = next(
        spec for spec in governance_specs
        if spec.kind is ArtifactKind.SAMPLING_PLAN
    )
    sampling_plan_payload = _read_bounded_file(
        governance / sampling_spec.relative_path,
        max_bytes=_MAX_SAMPLING_PLAN_BYTES,
        name="Phase 3B sampling plan",
    )
    _validate_sampling_plan(
        sampling_plan_payload,
        source_manifest_sha256=catalog.manifest_sha256,
        selections=selected,
    )
    cases = _draft_cases(
        catalog, selected, artifacts, creator_pseudonym, created_at)
    media = _media_paths(source_manifest, source)

    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Phase 3B queue output already exists")
    parent = output.parent.resolve(strict=True)
    if parent == source or source in parent.parents:
        raise ValueError("Phase 3B queue output must be outside the source root")
    if parent == governance or governance in parent.parents:
        raise ValueError("Phase 3B queue output must be outside the governance root")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        cases_path = temporary / "phase3b-draft-cases.jsonl"
        total_case_bytes = 0
        with cases_path.open("xb") as stream:
            for case in cases:
                case_payload = case.canonical_bytes()
                if len(case_payload) > _MAX_CASE_JSON_BYTES:
                    raise ValueError("Phase 3B case exceeds its byte limit")
                total_case_bytes += len(case_payload) + 1
                if total_case_bytes > _MAX_CASE_JSONL_BYTES:
                    raise ValueError("Phase 3B case JSONL exceeds its byte limit")
                stream.write(case_payload)
                stream.write(b"\n")
        _write_html(temporary / "phase3b-review-queue.html", cases, media)
        manifest = {
            "schema_version": PHASE3B_QUEUE_SCHEMA_VERSION,
            "kind": "phase3b_label_empty_review_queue",
            "source_manifest_sha256": catalog.manifest_sha256,
            "governance_files": governance_records,
            "selections": [selection.to_dict() for selection in selected],
            "cases": [
                {
                    "case_id": case.case_id,
                    "source_binding_sha256": case.source.content_sha256(),
                    "case_sha256": case.content_sha256(),
                }
                for case in cases
            ],
            "creator_pseudonym": creator_pseudonym,
            "created_at": created_at,
            "implementation": package_implementation_identity(),
            "project_labels_prepopulated": False,
            "source_training_target_authorized": False,
            "commercial_use_authorized": False,
            "project_linguistic_validation_complete": False,
        }
        manifest_bytes = canonical_json_bytes(manifest)
        if len(manifest_bytes) > _MAX_QUEUE_MANIFEST_BYTES:
            raise ValueError("Phase 3B queue manifest exceeds its byte limit")
        (temporary / "queue-manifest.json").write_bytes(manifest_bytes)
        indexed_files = []
        for path in sorted(temporary.iterdir(), key=lambda item: item.name):
            observed = stable_sha256(path, temporary)
            if observed.size > _QUEUE_PAYLOAD_LIMITS[path.name]:
                raise ValueError(f"Phase 3B queue payload exceeds limit: {path.name}")
            indexed_files.append({
                "name": path.name,
                "sha256": observed.sha256,
                "size": observed.size,
            })
        index = {
            "schema_version": PHASE3B_QUEUE_SCHEMA_VERSION,
            "kind": "phase3b_review_queue_artifact_index",
            "queue_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "files": indexed_files,
        }
        (temporary / "artifact-index.json").write_bytes(
            canonical_json_bytes(index))
        for path, digest in observed_files:
            assert_file_unchanged(path, governance, digest)
        verify_phase3b_review_queue(
            temporary,
            source_manifest_payload=source_manifest_payload,
            source_root=source,
            governance_root=governance,
            expected_queue_manifest_sha256=index["queue_manifest_sha256"],
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Phase3BQueueReceipt(
        path=output.resolve(strict=True),
        queue_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        case_count=len(cases),
    )


def verify_phase3b_review_queue(
    queue_dir: str | Path,
    *,
    source_manifest_payload: bytes,
    source_root: str | Path,
    governance_root: str | Path,
    expected_queue_manifest_sha256: str,
) -> tuple[Phase3BReviewCase, ...]:
    """Reload and independently verify a queue against current source bytes."""
    queue = _resolve_root(queue_dir, "queue_dir")
    source = _resolve_root(source_root, "source_root")
    governance = _resolve_root(governance_root, "governance_root")
    if (source == governance or source in governance.parents
            or governance in source.parents):
        raise ValueError("source and governance roots must be disjoint")
    discovered: set[str] = set()
    for path in queue.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("queue entries must be non-symlink regular files")
        discovered.add(path.name)
    if discovered != _QUEUE_FILES:
        raise ValueError("queue file inventory is incomplete or contains extras")

    manifest_payload = _read_bounded_file(
        queue / "queue-manifest.json",
        max_bytes=_MAX_QUEUE_MANIFEST_BYTES,
        name="queue manifest",
    )
    if (not isinstance(expected_queue_manifest_sha256, str)
            or len(expected_queue_manifest_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in expected_queue_manifest_sha256)):
        raise ValueError("expected queue manifest identity must be lowercase SHA-256")
    if hashlib.sha256(manifest_payload).hexdigest() \
            != expected_queue_manifest_sha256:
        raise RuntimeError("queue manifest does not match its trusted receipt")

    index_payload = _read_bounded_file(
        queue / "artifact-index.json",
        max_bytes=_MAX_ARTIFACT_INDEX_BYTES,
        name="queue artifact index",
    )
    index = _strict_json(index_payload, max_bytes=_MAX_ARTIFACT_INDEX_BYTES)
    if not isinstance(index, Mapping) or set(index) != {
            "schema_version", "kind", "queue_manifest_sha256", "files"}:
        raise ValueError("queue artifact index schema is invalid")
    if (index["schema_version"] != PHASE3B_QUEUE_SCHEMA_VERSION
            or index["kind"] != "phase3b_review_queue_artifact_index"
            or not isinstance(index["files"], list)):
        raise ValueError("queue artifact index identity is invalid")
    if canonical_json_bytes(index) != index_payload:
        raise ValueError("queue artifact index is not canonical JSON")
    indexed_names: set[str] = set()
    indexed_observations: dict[str, StableFileDigest] = {}
    for record in index["files"]:
        if not isinstance(record, Mapping) \
                or set(record) != {"name", "sha256", "size"}:
            raise ValueError("queue artifact index entry is invalid")
        name = record["name"]
        digest = record["sha256"]
        size = record["size"]
        if (not isinstance(name, str) or name in indexed_names
                or name not in _QUEUE_FILES - {"artifact-index.json"}
                or not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not isinstance(size, int) or isinstance(size, bool) or size < 0
                or size > _QUEUE_PAYLOAD_LIMITS.get(name, -1)):
            raise ValueError("queue artifact index entry is invalid")
        indexed_names.add(name)
        payload_path = queue / name
        if payload_path.stat().st_size > _QUEUE_PAYLOAD_LIMITS[name]:
            raise ValueError(f"queue payload exceeds the byte limit: {name}")
        observed = stable_sha256(payload_path, queue)
        indexed_observations[name] = observed
        if size != observed.size or digest != observed.sha256:
            raise RuntimeError("queue artifact differs from its hash index")
    if indexed_names != _QUEUE_FILES - {"artifact-index.json"}:
        raise ValueError("queue artifact index does not cover every payload")

    manifest = _strict_json(
        manifest_payload, max_bytes=_MAX_QUEUE_MANIFEST_BYTES)
    required = {
        "schema_version", "kind", "source_manifest_sha256", "governance_files",
        "selections", "cases", "creator_pseudonym", "created_at",
        "implementation", "project_labels_prepopulated",
        "source_training_target_authorized", "commercial_use_authorized",
        "project_linguistic_validation_complete",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise ValueError("queue manifest schema is invalid")
    if canonical_json_bytes(manifest) != manifest_payload:
        raise ValueError("queue manifest is not canonical JSON")
    if (manifest["schema_version"] != PHASE3B_QUEUE_SCHEMA_VERSION
            or manifest["kind"] != "phase3b_label_empty_review_queue"):
        raise ValueError("queue manifest identity is invalid")
    if index["queue_manifest_sha256"] != hashlib.sha256(manifest_payload).hexdigest():
        raise RuntimeError("queue manifest differs from the artifact index")
    if any(manifest[name] is not False for name in (
            "project_labels_prepopulated", "source_training_target_authorized",
            "commercial_use_authorized", "project_linguistic_validation_complete")):
        raise ValueError("queue manifest contains a prohibited approval claim")
    _require_text(
        "queue creator_pseudonym", manifest["creator_pseudonym"], maximum=256)
    _require_timestamp(manifest["created_at"])
    implementation = manifest["implementation"]
    implementation_fields = {
        "schema_version", "package_version", "identity_kind", "git_revision",
        "git_worktree_clean", "dependency_lock_sha256", "implementation_sha256",
        "sources",
    }
    if (not isinstance(implementation, Mapping)
            or set(implementation) != implementation_fields
            or not isinstance(implementation["sources"], list)):
        raise ValueError("queue implementation identity schema is invalid")
    catalog = load_eaf_source_catalog(source_manifest_payload, source)
    source_manifest = load_eaf_manifest(source_manifest_payload)
    media = _media_paths(source_manifest, source)
    if manifest["source_manifest_sha256"] != catalog.manifest_sha256:
        raise RuntimeError("queue references another source manifest")

    governance_records = manifest["governance_files"]
    if not isinstance(governance_records, list):
        raise ValueError("queue governance records must be a list")
    specs: list[GovernanceFileSpec] = []
    for record in governance_records:
        if (not isinstance(record, Mapping)
                or set(record) != {"spec", "sha256", "size"}
                or not isinstance(record["spec"], Mapping)):
            raise ValueError("queue governance record is invalid")
        specs.append(GovernanceFileSpec.from_dict(record["spec"]))
    artifacts, current_records, _ = _verify_governance(governance, specs)
    if current_records != governance_records:
        raise RuntimeError("queue governance artifacts have changed")

    selection_records = manifest["selections"]
    if not isinstance(selection_records, list):
        raise ValueError("queue selections must be a list")
    parsed_selections = []
    for record in selection_records:
        if not isinstance(record, Mapping):
            raise ValueError("queue selection entry must be an object")
        parsed_selections.append(QueueSelection.from_dict(record))
    selected = _validate_selections(parsed_selections)
    if [item.to_dict() for item in selected] != selection_records:
        raise ValueError("queue selections are not in canonical order")
    sampling_spec = next(
        spec for spec in specs if spec.kind is ArtifactKind.SAMPLING_PLAN)
    sampling_plan_payload = _read_bounded_file(
        governance / sampling_spec.relative_path,
        max_bytes=_MAX_SAMPLING_PLAN_BYTES,
        name="Phase 3B sampling plan",
    )
    _validate_sampling_plan(
        sampling_plan_payload,
        source_manifest_sha256=catalog.manifest_sha256,
        selections=selected,
    )

    cases_path = queue / "phase3b-draft-cases.jsonl"
    case_items: list[Phase3BReviewCase] = []
    with cases_path.open("rb") as stream:
        while True:
            framed = stream.readline(_MAX_CASE_JSON_BYTES + 2)
            if not framed:
                break
            if (not framed.endswith(b"\n") or len(framed) > _MAX_CASE_JSON_BYTES + 1
                    or len(framed) == 1):
                raise ValueError("queue case JSONL framing is invalid")
            case_items.append(load_phase3b_case(
                framed[:-1], max_bytes=_MAX_CASE_JSON_BYTES))
            if len(case_items) > len(selected):
                raise ValueError("queue case JSONL has more cases than selections")
    assert_file_unchanged(
        cases_path,
        queue,
        indexed_observations["phase3b-draft-cases.jsonl"],
    )
    if not case_items:
        raise ValueError("queue case JSONL framing is invalid")
    cases = tuple(case_items)
    if any(case.state.value != "draft" or case.primary is not None for case in cases):
        raise ValueError("queue contains a pre-populated or non-draft case")
    case_records = manifest["cases"]
    if not isinstance(case_records, list) or len(case_records) != len(cases):
        raise ValueError("queue case index has the wrong size")
    expected_records = [
        {
            "case_id": case.case_id,
            "source_binding_sha256": case.source.content_sha256(),
            "case_sha256": case.content_sha256(),
        }
        for case in cases
    ]
    if case_records != expected_records:
        raise RuntimeError("queue case index does not match its draft snapshots")
    reconstructed = _draft_cases(
        catalog,
        selected,
        artifacts,
        manifest["creator_pseudonym"],
        manifest["created_at"],
    )
    if tuple(case.canonical_bytes() for case in reconstructed) != tuple(
            case.canonical_bytes() for case in cases):
        raise RuntimeError("queue drafts do not match the selected source evidence")
    expected_html = _render_html(reconstructed, media)
    observed_html = indexed_observations["phase3b-review-queue.html"]
    if observed_html.size != len(expected_html):
        raise RuntimeError("queue HTML does not match the verified source evidence")
    html_payload = _read_bounded_file(
        queue / "phase3b-review-queue.html",
        max_bytes=_MAX_REVIEW_HTML_BYTES,
        name="queue review HTML",
    )
    if html_payload != expected_html:
        raise RuntimeError("queue HTML does not match the verified source evidence")
    return cases


__all__ = [
    "PHASE3B_QUEUE_SCHEMA_VERSION", "PHASE3B_SAMPLING_PLAN_SCHEMA_VERSION",
    "GovernanceFileSpec", "QueueSelection", "Phase3BQueueReceipt",
    "write_phase3b_review_queue", "verify_phase3b_review_queue",
]
