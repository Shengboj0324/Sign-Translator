"""Stable-input inference shared by the single-record and corpus CLIs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from ..data_engineering.exporter import decode_landmark_npz
from .artifacts import (
    load_pipeline_bundle,
    sha256_file,
    verify_bundle,
    write_candidate_batch,
)
from .pipeline import HybridPseudoGlossPipeline
from .readiness import validate_dataset_authorization_artifact


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    sha256: str
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class InferenceContext:
    bundle: Path
    manifest: Mapping[str, Any]
    pipeline: HybridPseudoGlossPipeline
    authorization_path: Path
    authorization_sha256: str
    authorization_evidence: FileSnapshot
    bundle_files: tuple[FileSnapshot, ...]


@dataclass(frozen=True)
class InferenceRequest:
    transcript_file: Path
    source_video: Path
    landmark_track: Path
    sample_id: str
    created_at: str
    output: Path

    def __post_init__(self) -> None:
        if any(not isinstance(path, Path) for path in (
                self.transcript_file, self.source_video, self.landmark_track,
                self.output)):
            raise TypeError("inference paths must be pathlib.Path values")
        if not isinstance(self.sample_id, str) or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", self.sample_id) is None:
            raise ValueError("sample_id contains prohibited path or identifier characters")
        if not isinstance(self.created_at, str):
            raise ValueError("created_at must be an ISO-8601 timestamp string")
        try:
            timestamp = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("created_at must be an ISO-8601 timestamp") from error
        if timestamp.tzinfo is None:
            raise ValueError("created_at must include a timezone")


def capture_regular_file(path: Path) -> FileSnapshot:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input must be a non-symlink regular file: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"input mutated while hashing: {path}")
    return FileSnapshot(path, digest, identity_after)


def verify_unchanged(snapshot: FileSnapshot) -> None:
    current = capture_regular_file(snapshot.path)
    if current.identity != snapshot.identity or current.sha256 != snapshot.sha256:
        raise RuntimeError(f"input mutated during inference: {snapshot.path}")


def verify_identity_unchanged(snapshot: FileSnapshot) -> None:
    if snapshot.path.is_symlink() or not snapshot.path.is_file():
        raise RuntimeError(f"input identity changed during inference: {snapshot.path}")
    stat = snapshot.path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity != snapshot.identity:
        raise RuntimeError(f"input identity changed during inference: {snapshot.path}")


def load_inference_context(model_bundle: str | Path,
                           dataset_authorization: str | Path) -> InferenceContext:
    bundle = Path(model_bundle).absolute()
    manifest = verify_bundle(bundle)
    pipeline = load_pipeline_bundle(bundle)
    authorization_path = Path(dataset_authorization).absolute()
    authorization_snapshot = capture_regular_file(authorization_path)
    authorization = validate_dataset_authorization_artifact(
        authorization_path, requested_actions=("create_derivatives",))
    evidence_snapshot = capture_regular_file(Path(authorization.evidence_uri).absolute())
    verify_unchanged(authorization_snapshot)
    bundle_files = tuple(capture_regular_file(path) for path in sorted(bundle.iterdir())
                         if path.is_file() and not path.is_symlink())
    if not bundle_files:
        raise ValueError("model bundle contains no regular files")
    return InferenceContext(
        bundle=bundle, manifest=manifest, pipeline=pipeline,
        authorization_path=authorization_path,
        authorization_sha256=authorization_snapshot.sha256,
        authorization_evidence=evidence_snapshot,
        bundle_files=bundle_files,
    )


def run_inference(context: InferenceContext, request: InferenceRequest, *,
                  verify_model_hashes_after: bool = True) -> Path:
    transcript_snapshot = capture_regular_file(request.transcript_file.absolute())
    transcript_bytes = transcript_snapshot.path.read_bytes()
    if hashlib.sha256(transcript_bytes).hexdigest() != transcript_snapshot.sha256:
        raise RuntimeError("transcript mutated during stable read")
    transcript = transcript_bytes.decode("utf-8", errors="strict")
    source_video_snapshot = capture_regular_file(request.source_video.absolute())
    track_snapshot = capture_regular_file(request.landmark_track.absolute())
    track = decode_landmark_npz(track_snapshot.path)
    governance = context.manifest["model_governance"]
    generator_model_id = f"{governance['text_model_id']}+{governance['video_model_id']}"
    result = context.pipeline.generate(
        transcript=transcript, track=track, source_sample_id=request.sample_id,
        source_video_sha256=source_video_snapshot.sha256,
        generator_model_id=generator_model_id,
        code_revision=context.manifest["code_revision"],
        random_seed=context.manifest["seed"], created_at=request.created_at,
    )
    decision = {
        **asdict(result.decision),
        "selected_annotation_id": result.selected_annotation_id,
        "dropped_text_probability_mass": result.dropped_text_probability_mass,
    }
    for snapshot in (transcript_snapshot, source_video_snapshot, track_snapshot,
                     context.authorization_evidence):
        verify_unchanged(snapshot)
    authorization_snapshot = capture_regular_file(context.authorization_path)
    if authorization_snapshot.sha256 != context.authorization_sha256:
        raise RuntimeError("dataset authorization mutated during inference")
    validate_dataset_authorization_artifact(
        context.authorization_path, requested_actions=("create_derivatives",))
    for snapshot in context.bundle_files:
        verify_identity_unchanged(snapshot)
    if verify_model_hashes_after and verify_bundle(context.bundle) != context.manifest:
        raise RuntimeError("model bundle changed during inference")
    return write_candidate_batch(
        request.output, records=result.records, decision=decision,
        model_bundle_manifest_sha256=sha256_file(context.bundle / "manifest.json"),
        dataset_authorization_sha256=context.authorization_sha256,
        source_sample_id=request.sample_id,
        transcript_sha256=transcript_snapshot.sha256,
        source_video_sha256=source_video_snapshot.sha256,
        landmark_track_sha256=track_snapshot.sha256,
        code_revision=context.manifest["code_revision"],
    )
