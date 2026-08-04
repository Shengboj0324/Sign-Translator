"""Resumable, fail-closed corpus inference after full activation approval."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import (
    _verify_candidate_batch_preverified,
    sha256_file,
    verify_bundle,
)
from .contracts import canonical_json_bytes
from .inference import (
    FileSnapshot,
    InferenceRequest,
    capture_regular_file,
    load_inference_context,
    run_inference,
    verify_identity_unchanged,
    verify_unchanged,
)
from .readiness import assess_activation, load_activation_charter
from .security import strict_json_loads


CORPUS_INFERENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CorpusInferenceRecord:
    sample_id: str
    source_id: str
    transcript_file: Path
    source_video: Path
    landmark_track: Path
    created_at: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorpusInferenceRecord":
        required = {
            "sample_id", "source_id", "transcript_file", "source_video",
            "landmark_track", "created_at",
        }
        if not isinstance(value, Mapping) or set(value) != required \
                or any(not isinstance(value[name], str) or not value[name]
                       for name in required):
            raise ValueError("corpus inference record schema mismatch")
        paths = (value["transcript_file"], value["source_video"],
                 value["landmark_track"])
        if any(not Path(path).is_absolute() for path in paths):
            raise ValueError("corpus inference input paths must be absolute")
        request = InferenceRequest(
            transcript_file=Path(paths[0]), source_video=Path(paths[1]),
            landmark_track=Path(paths[2]), sample_id=value["sample_id"],
            created_at=value["created_at"], output=Path("unused"))
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}",
                        value["source_id"]) is None:
            raise ValueError("source_id contains prohibited characters")
        return cls(
            request.sample_id, value["source_id"], request.transcript_file,
            request.source_video, request.landmark_track, request.created_at)


def load_corpus_manifest(path: Path) -> tuple[FileSnapshot, tuple[CorpusInferenceRecord, ...]]:
    snapshot = capture_regular_file(path.absolute())
    records = []
    with snapshot.path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank corpus manifest line: {line_number}")
            records.append(CorpusInferenceRecord.from_dict(
                strict_json_loads(line, max_bytes=1_048_576)))
    if not records:
        raise ValueError("corpus inference manifest is empty")
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("corpus inference manifest contains duplicate sample IDs")
    verify_unchanged(snapshot)
    return snapshot, tuple(sorted(records, key=lambda record: record.sample_id))


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("stale corpus-state temporary file requires manual inspection")
    with temporary.open("xb") as stream:
        stream.write(canonical_json_bytes(state) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _initial_state(*, manifest_sha256: str, charter_sha256: str,
                   model_manifest_sha256: str, authorization_sha256: str,
                   record_count: int) -> dict[str, Any]:
    return {
        "schema_version": CORPUS_INFERENCE_SCHEMA_VERSION,
        "input_manifest_sha256": manifest_sha256,
        "activation_charter_sha256": charter_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "dataset_authorization_sha256": authorization_sha256,
        "record_count": record_count,
        "completed": {},
        "complete": False,
    }


def _validate_state(value: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("corpus inference state schema mismatch")
    if isinstance(value["schema_version"], bool) \
            or not isinstance(value["schema_version"], int) \
            or isinstance(value["record_count"], bool) \
            or not isinstance(value["record_count"], int):
        raise ValueError("corpus state integer fields have invalid types")
    for name in expected:
        if name not in {"completed", "complete"} and value[name] != expected[name]:
            raise ValueError(f"corpus resume binding mismatch: {name}")
    if not isinstance(value["completed"], dict) or any(
            not isinstance(sample_id, str) or not isinstance(digest, str)
            or len(digest) != 64 or any(character not in "0123456789abcdef"
                                        for character in digest)
            for sample_id, digest in value["completed"].items()):
        raise ValueError("corpus completed-checkpoint map is invalid")
    if not isinstance(value["complete"], bool):
        raise ValueError("corpus complete flag must be boolean")
    if len(value["completed"]) > value["record_count"] \
            or value["complete"] and len(value["completed"]) != value["record_count"]:
        raise ValueError("corpus completion count contradicts its state")
    return value


def run_corpus_inference(*, input_manifest: str | Path, output: str | Path,
                         model_bundle: str | Path, dataset_authorization: str | Path,
                         activation_charter: str | Path, resume: bool = False) -> Path:
    """Generate one hash-bound candidate batch per record with deterministic resume."""
    if not isinstance(resume, bool):
        raise TypeError("resume must be boolean")
    charter_path = Path(activation_charter).absolute()
    charter_snapshot = capture_regular_file(charter_path)
    charter = load_activation_charter(charter_path)
    readiness = assess_activation(charter)
    if not readiness.activation_approved:
        failed = [check.name for check in readiness.checks if not check.passed]
        raise PermissionError(f"pseudo-gloss activation is not approved: {failed}")
    context = load_inference_context(model_bundle, dataset_authorization)
    if Path(charter.model_bundle.path).absolute() != context.bundle \
            or charter.model_bundle.sha256 != sha256_file(context.bundle / "manifest.json"):
        raise ValueError("corpus model bundle does not match the activation charter")
    if Path(charter.dataset_authorization.path).absolute() != context.authorization_path \
            or charter.dataset_authorization.sha256 != context.authorization_sha256:
        raise ValueError("corpus authorization does not match the activation charter")

    input_snapshot, records = load_corpus_manifest(Path(input_manifest))
    root = Path(output).absolute()
    current = root
    while current != current.parent:
        if current.is_symlink():
            raise ValueError("corpus output cannot have a symlinked path component")
        current = current.parent
    state_path = root / "corpus_state.json"
    sample_root = root / "samples"
    expected = _initial_state(
        manifest_sha256=input_snapshot.sha256,
        charter_sha256=charter_snapshot.sha256,
        model_manifest_sha256=charter.model_bundle.sha256,
        authorization_sha256=context.authorization_sha256,
        record_count=len(records))
    if resume:
        if not root.is_dir() or state_path.is_symlink() or not state_path.is_file() \
                or sample_root.is_symlink() or not sample_root.is_dir():
            raise ValueError("resume requires an intact corpus output directory")
        if {path.name for path in root.iterdir()} != {"corpus_state.json", "samples"}:
            raise ValueError("corpus output root contains undeclared artifacts")
        state = _validate_state(strict_json_loads(state_path.read_bytes()), expected)
    else:
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise FileExistsError("refusing non-empty corpus output without resume")
        root.mkdir(parents=True, exist_ok=True)
        sample_root.mkdir()
        state = expected
        _write_state(state_path, state)

    record_ids = {record.sample_id for record in records}
    completed = dict(state["completed"])
    if set(completed) - record_ids:
        raise ValueError("checkpoint contains a sample absent from the input manifest")
    unexpected = {path.name for path in sample_root.iterdir()} - record_ids
    if unexpected:
        raise ValueError(f"corpus output contains unexpected sample directories: {sorted(unexpected)}")

    for record in records:
        destination = sample_root / record.sample_id
        request = InferenceRequest(
            transcript_file=record.transcript_file,
            source_video=record.source_video, landmark_track=record.landmark_track,
            sample_id=record.sample_id, created_at=record.created_at,
            output=destination)
        if destination.exists():
            batch_manifest = _verify_candidate_batch_preverified(
                destination, model_bundle=context.bundle,
                dataset_authorization=context.authorization_path,
                verified_model_manifest=context.manifest)
            transcript_snapshot = capture_regular_file(record.transcript_file)
            video_snapshot = capture_regular_file(record.source_video)
            track_snapshot = capture_regular_file(record.landmark_track)
            if batch_manifest["source_sample_id"] != record.sample_id \
                    or batch_manifest["transcript_sha256"] != transcript_snapshot.sha256 \
                    or batch_manifest["source_video_sha256"] != video_snapshot.sha256 \
                    or batch_manifest["landmark_track_sha256"] != track_snapshot.sha256:
                raise ValueError("existing candidate batch does not bind the resume record")
            for snapshot in (transcript_snapshot, video_snapshot, track_snapshot):
                verify_unchanged(snapshot)
            batch_hash = sha256_file(destination / "manifest.json")
            if record.sample_id in completed and completed[record.sample_id] != batch_hash:
                raise ValueError("completed candidate batch hash drifted")
        else:
            run_inference(context, request, verify_model_hashes_after=False)
            batch_hash = sha256_file(destination / "manifest.json")
        if completed.get(record.sample_id) != batch_hash:
            completed[record.sample_id] = batch_hash
            state = {**expected, "completed": completed, "complete": False}
            _write_state(state_path, state)
        verify_unchanged(input_snapshot)
        verify_unchanged(charter_snapshot)
        for snapshot in context.bundle_files:
            verify_identity_unchanged(snapshot)

    verify_unchanged(input_snapshot)
    verify_unchanged(charter_snapshot)
    for snapshot in context.bundle_files:
        verify_unchanged(snapshot)
    if verify_bundle(context.bundle) != context.manifest:
        raise RuntimeError("model bundle failed final corpus verification")
    if not assess_activation(load_activation_charter(charter_path)).activation_approved:
        raise RuntimeError("activation evidence failed final corpus verification")
    state = {**expected, "completed": completed, "complete": True}
    _write_state(state_path, state)
    return state_path
