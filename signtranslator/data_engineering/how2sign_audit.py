"""Resumable, fail-closed audit of a local How2Sign frontal-view corpus.

This module is deliberately label-free.  It inventories immutable source bytes,
validates the exact video/OpenPose structure, computes 2D observation diagnostics,
and creates a review queue.  It never creates gloss, signer identity, depth, 3D
motion, corrected landmarks, or train/validation/test assignments.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import html
import json
import math
import os
import platform
import random
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence
from urllib.parse import quote

import numpy as np

from .exporter import LandmarkTrack, decode_video_clock
from .how2sign import (
    OPENPOSE_JOINT_NAMES,
    OPENPOSE_LANDMARK_PARTS,
    How2SignRow,
    decode_how2sign_openpose,
    inspect_how2sign_root,
    read_how2sign_metadata,
)


AUDIT_SCHEMA_VERSION = 1
DEFAULT_THRESHOLDS = (
    0.00, 0.05, 0.10, 0.20, 0.30, 0.40,
    0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
)
STATUS_VALUES = {
    "valid", "quality_warning", "structural_failure",
    "missing_source", "unjoinable_artifact",
}


@dataclass(frozen=True)
class How2SignAuditConfig:
    """Scientific configuration; execution limits are intentionally excluded."""

    schema_version: int = AUDIT_SCHEMA_VERSION
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    review_per_category: int = 8
    random_seed: int = 20260803

    def validate(self) -> None:
        if self.schema_version != AUDIT_SCHEMA_VERSION:
            raise ValueError("unsupported audit schema version")
        if self.review_per_category < 1:
            raise ValueError("review_per_category must be positive")
        if not self.thresholds:
            raise ValueError("at least one threshold is required")
        if any(not math.isfinite(value) or not 0 <= value <= 1
               for value in self.thresholds):
            raise ValueError("thresholds must be finite and in [0, 1]")
        if tuple(sorted(set(self.thresholds))) != self.thresholds:
            raise ValueError("thresholds must be unique and strictly increasing")


@dataclass(frozen=True)
class StableFileDigest:
    sha256: str
    size: int
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True)
class DerivativeSummary:
    velocity_count: int
    acceleration_count: int
    velocity_median: float | None
    velocity_scaled_mad: float | None
    acceleration_median: float | None
    acceleration_scaled_mad: float | None


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _resolved_regular_file(path: Path, root: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"symlinked source artifact is forbidden: {path}")
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"source path escapes dataset root: {path}")
    if not resolved.is_file():
        raise ValueError(f"source artifact is not a regular file: {path}")
    return resolved


def stable_sha256(path: str | os.PathLike[str], root: str | os.PathLike[str],
                  *, block_size: int = 1024 * 1024) -> StableFileDigest:
    """Hash a regular in-root file and fail if its identity changes while read."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    source_root = Path(root)
    resolved = _resolved_regular_file(Path(path), source_root)
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    after = resolved.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"source artifact changed while hashing: {resolved}")
    return StableFileDigest(digest.hexdigest(), after.st_size, after.st_dev,
                            after.st_ino, after.st_mtime_ns)


def assert_file_unchanged(path: str | os.PathLike[str],
                          root: str | os.PathLike[str],
                          expected: StableFileDigest) -> None:
    """Verify identity metadata after a separate decode/validation operation."""
    resolved = _resolved_regular_file(Path(path), Path(root))
    current = resolved.stat()
    identity = (current.st_size, current.st_dev, current.st_ino, current.st_mtime_ns)
    expected_identity = (expected.size, expected.device, expected.inode,
                         expected.mtime_ns)
    if identity != expected_identity:
        raise RuntimeError(f"source artifact changed after hashing: {resolved}")


def hierarchical_digest(
    named_digests: Iterable[tuple[str, StableFileDigest]],
) -> str:
    """Hash an explicitly ordered sequence of names and content digests."""
    digest = hashlib.sha256()
    previous: str | None = None
    for name, item in named_digests:
        if previous is not None and name <= previous:
            raise ValueError("hierarchical digest entries must be strictly ordered")
        previous = name
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(item.sha256))
        digest.update(item.size.to_bytes(8, "big"))
    return digest.hexdigest()


def scaled_mad(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    median = float(np.median(finite))
    mad = float(1.4826 * np.median(np.abs(finite - median)))
    return median, mad


def nonuniform_derivative_summary(track: LandmarkTrack) -> DerivativeSummary:
    """Compute speed/acceleration only over fully valid temporal supports."""
    values = np.asarray(track.values, dtype=np.float64)
    valid = np.asarray(track.validity_mask, dtype=np.bool_)
    times = np.asarray(track.timestamps, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError("quality derivatives require 2D values shaped (2,T,V)")
    if valid.shape != values.shape[1:] or times.shape != (values.shape[1],):
        raise ValueError("track arrays are not aligned")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")

    dt = np.diff(times)
    pair_valid = valid[:-1] & valid[1:]
    displacement = values[:, 1:] - values[:, :-1]
    speed = np.linalg.norm(displacement, axis=0) / dt[:, None]
    speeds = speed[pair_valid]

    if times.size >= 3:
        triple_valid = valid[:-2] & valid[1:-1] & valid[2:]
        v_prev = (values[:, 1:-1] - values[:, :-2]) / dt[:-1][None, :, None]
        v_next = (values[:, 2:] - values[:, 1:-1]) / dt[1:][None, :, None]
        accel = 2.0 * (v_next - v_prev) / (dt[:-1] + dt[1:])[None, :, None]
        accel_magnitude = np.linalg.norm(accel, axis=0)
        accelerations = accel_magnitude[triple_valid]
    else:
        accelerations = np.empty(0, dtype=np.float64)

    velocity_median, velocity_mad = scaled_mad(speeds)
    acceleration_median, acceleration_mad = scaled_mad(accelerations)
    return DerivativeSummary(
        velocity_count=int(speeds.size),
        acceleration_count=int(accelerations.size),
        velocity_median=velocity_median,
        velocity_scaled_mad=velocity_mad,
        acceleration_median=acceleration_median,
        acceleration_scaled_mad=acceleration_mad,
    )


def longest_false_run(mask: np.ndarray) -> int:
    """Longest consecutive run containing no true value in each frame."""
    frame_has_observation = np.asarray(mask, dtype=np.bool_).any(axis=1)
    longest = current = 0
    for observed in frame_has_observation:
        if observed:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def threshold_metrics(track: LandmarkTrack,
                      thresholds: Sequence[float]) -> list[dict]:
    """Evaluate declared thresholds without selecting or ranking an optimum."""
    config = How2SignAuditConfig(thresholds=tuple(thresholds))
    config.validate()
    base = np.asarray(track.validity_mask, dtype=np.bool_)
    confidence = np.asarray(track.confidence, dtype=np.float64)
    total = int(base.size)
    rows: list[dict] = []
    previous: np.ndarray | None = None
    for threshold in config.thresholds:
        retained = base & (confidence >= threshold)
        if previous is not None and np.any(retained & ~previous):
            raise AssertionError("threshold retention sets are not nested")
        previous = retained
        pair_count = int((retained[:-1] & retained[1:]).sum())
        triple_count = int((retained[:-2] & retained[1:-1] & retained[2:]).sum()) \
            if retained.shape[0] >= 3 else 0
        region_coverage = {}
        for region, indices in OPENPOSE_LANDMARK_PARTS.items():
            region_mask = retained[:, indices]
            region_coverage[region] = float(region_mask.mean()) if region_mask.size else 0.0
        rows.append({
            "threshold": float(threshold),
            "legacy_adapter_default": bool(abs(threshold - 0.30) < 1e-12),
            "coverage": float(retained.sum() / total) if total else 0.0,
            "region_coverage": region_coverage,
            "complete_frame_coverage": float(retained.all(axis=1).mean())
            if retained.shape[0] else 0.0,
            "longest_fully_missing_run": longest_false_run(retained),
            "derivative_valid_pairs": pair_count,
            "derivative_valid_triples": triple_count,
            "unusable": bool(not retained.any()),
        })
    return rows


def _region_metrics(track: LandmarkTrack) -> dict[str, dict[str, float | int | None]]:
    result = {}
    for region, indices in OPENPOSE_LANDMARK_PARTS.items():
        validity = track.validity_mask[:, indices]
        confidences = track.confidence[:, indices][validity]
        median, mad = scaled_mad(confidences)
        result[region] = {
            "coverage": float(validity.mean()) if validity.size else 0.0,
            "valid_observations": int(validity.sum()),
            "confidence_median": median,
            "confidence_scaled_mad": mad,
            "longest_fully_missing_run": longest_false_run(validity),
        }
    return result


def _landmark_metrics(track: LandmarkTrack) -> dict[str, list[float | int | None]]:
    coverage = []
    confidence_median = []
    confidence_scaled_mad = []
    longest_missing = []
    for joint in range(track.validity_mask.shape[1]):
        validity = track.validity_mask[:, joint]
        confidences = track.confidence[:, joint][validity]
        median, mad = scaled_mad(confidences)
        coverage.append(float(validity.mean()) if validity.size else 0.0)
        confidence_median.append(median)
        confidence_scaled_mad.append(mad)
        longest_missing.append(longest_false_run(validity[:, None]))
    return {
        "coverage": coverage,
        "confidence_median": confidence_median,
        "confidence_scaled_mad": confidence_scaled_mad,
        "longest_missing_run": longest_missing,
    }


def _edge_discontinuity_summary(track: LandmarkTrack) -> dict[str, float | int | None]:
    """Report projection-sensitive 2D edge changes; never an anatomy verdict."""
    from .how2sign import OPENPOSE_HOLISTIC_EDGES

    lengths = []
    changes = []
    for left, right in OPENPOSE_HOLISTIC_EDGES:
        observed = track.validity_mask[:, left] & track.validity_mask[:, right]
        distance = np.linalg.norm(track.values[:, :, left] - track.values[:, :, right], axis=0)
        lengths.extend(distance[observed].tolist())
        consecutive = observed[:-1] & observed[1:]
        changes.extend(np.abs(np.diff(distance))[consecutive].tolist())
    length_median, length_mad = scaled_mad(np.asarray(lengths))
    change_median, change_mad = scaled_mad(np.asarray(changes))
    return {
        "edge_length_count": len(lengths),
        "edge_length_median": length_median,
        "edge_length_scaled_mad": length_mad,
        "edge_change_count": len(changes),
        "edge_change_median": change_median,
        "edge_change_scaled_mad": change_mad,
    }


def _left_right_swap_signal(track: LandmarkTrack) -> dict[str, float | int | None]:
    # BODY_25 indices 4 and 7 are right and left wrist respectively.
    right, left = 4, 7
    observed = (track.validity_mask[:-1, right] & track.validity_mask[1:, right]
                & track.validity_mask[:-1, left] & track.validity_mask[1:, left])
    if not observed.any():
        return {"comparable_pairs": 0, "swap_advantage_fraction": None,
                "swap_advantage_median": None}
    previous_right = track.values[:, :-1, right]
    previous_left = track.values[:, :-1, left]
    current_right = track.values[:, 1:, right]
    current_left = track.values[:, 1:, left]
    direct = (np.linalg.norm(current_right - previous_right, axis=0)
              + np.linalg.norm(current_left - previous_left, axis=0))
    swapped = (np.linalg.norm(current_right - previous_left, axis=0)
               + np.linalg.norm(current_left - previous_right, axis=0))
    advantage = direct[observed] - swapped[observed]
    return {
        "comparable_pairs": int(advantage.size),
        "swap_advantage_fraction": float((advantage > 0).mean()),
        "swap_advantage_median": float(np.median(advantage)),
    }


def _torso_relative_summary(track: LandmarkTrack) -> dict[str, float | int | None]:
    """Projection-sensitive dimensionless wrist separation / shoulder width."""
    right_shoulder, left_shoulder = 2, 5
    right_wrist, left_wrist = 4, 7
    observed = (track.validity_mask[:, right_shoulder]
                & track.validity_mask[:, left_shoulder]
                & track.validity_mask[:, right_wrist]
                & track.validity_mask[:, left_wrist])
    shoulder_width = np.linalg.norm(
        track.values[:, :, right_shoulder] - track.values[:, :, left_shoulder], axis=0)
    stable_scale = shoulder_width > 1e-8
    evaluable = observed & stable_scale
    if not evaluable.any():
        return {"count": 0, "wrist_separation_over_shoulder_median": None,
                "wrist_separation_over_shoulder_scaled_mad": None}
    wrist_separation = np.linalg.norm(
        track.values[:, :, right_wrist] - track.values[:, :, left_wrist], axis=0)
    ratio = wrist_separation[evaluable] / shoulder_width[evaluable]
    median, mad = scaled_mad(ratio)
    return {"count": int(ratio.size),
            "wrist_separation_over_shoulder_median": median,
            "wrist_separation_over_shoulder_scaled_mad": mad}


def _repeated_frame_summary(track: LandmarkTrack) -> dict[str, float | int]:
    if track.values.shape[1] < 2:
        return {"comparable_pairs": 0, "exact_repeated_pairs": 0,
                "exact_repeated_fraction": 0.0}
    comparable = track.validity_mask[:-1] & track.validity_mask[1:]
    displacement = np.linalg.norm(track.values[:, 1:] - track.values[:, :-1], axis=0)
    comparable_pairs = comparable.any(axis=1)
    repeated = comparable_pairs & np.all((~comparable) | (displacement == 0), axis=1)
    count = int(comparable_pairs.sum())
    return {"comparable_pairs": count, "exact_repeated_pairs": int(repeated.sum()),
            "exact_repeated_fraction": float(repeated.sum() / count) if count else 0.0}


def pose_quality_metrics(track: LandmarkTrack) -> dict:
    derivatives = nonuniform_derivative_summary(track)
    valid_values = np.moveaxis(track.values, 0, -1)[track.validity_mask]
    frozen = False
    if track.values.shape[1] > 1:
        consecutive = track.validity_mask[:-1] & track.validity_mask[1:]
        displacement = np.linalg.norm(track.values[:, 1:] - track.values[:, :-1], axis=0)
        frozen = bool(consecutive.any() and np.all(displacement[consecutive] == 0))
    return {
        "frame_count": int(track.values.shape[1]),
        "joint_count": int(track.values.shape[2]),
        "valid_observations": int(track.validity_mask.sum()),
        "coverage": float(track.validity_mask.mean()),
        "coordinate_min": float(valid_values.min()) if valid_values.size else None,
        "coordinate_max": float(valid_values.max()) if valid_values.size else None,
        "regions": _region_metrics(track),
        "landmarks": _landmark_metrics(track),
        "derivatives": asdict(derivatives),
        "edges": _edge_discontinuity_summary(track),
        "left_right_swap_signal": _left_right_swap_signal(track),
        "torso_relative_2d": _torso_relative_summary(track),
        "repeated_frames": _repeated_frame_summary(track),
        "fully_frozen_on_comparable_pairs": frozen,
    }


def _implementation_identity() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "-C", os.fspath(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("audit requires a readable git revision") from error
    digest = hashlib.sha256()
    for source in (Path(__file__), Path(__file__).with_name("how2sign.py"),
                   Path(__file__).with_name("exporter.py")):
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    return {"git_revision": revision, "implementation_sha256": digest.hexdigest()}


def _meta_payload(root: Path, config: How2SignAuditConfig) -> dict:
    return {
        "root": os.fspath(root.resolve(strict=True)),
        "config": asdict(config),
        "implementation": _implementation_identity(),
    }


def _connect_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS clips (
            sample_id TEXT PRIMARY KEY,
            video_id TEXT,
            sentence_id TEXT,
            sentence_name TEXT,
            filename_code TEXT,
            status TEXT NOT NULL,
            error TEXT,
            raw_uri TEXT,
            rendered_uri TEXT,
            raw_sha256 TEXT,
            rendered_sha256 TEXT,
            openpose_sha256 TEXT,
            frame_count INTEGER,
            duration REAL,
            quality_json TEXT,
            diagnostics_json TEXT
        );
        CREATE TABLE IF NOT EXISTS thresholds (
            sample_id TEXT NOT NULL REFERENCES clips(sample_id) ON DELETE CASCADE,
            threshold REAL NOT NULL,
            metrics_json TEXT NOT NULL,
            PRIMARY KEY(sample_id, threshold)
        );
    """)
    return connection


def _initialize_database(connection: sqlite3.Connection, payload: dict,
                         *, resume: bool) -> None:
    encoded = _canonical_json(payload).decode("utf-8")
    row = connection.execute("SELECT value_json FROM meta WHERE key='identity'").fetchone()
    if row is None:
        if resume:
            raise ValueError("cannot resume an uninitialized audit database")
        connection.execute("INSERT INTO meta(key,value_json) VALUES('identity',?)", (encoded,))
        connection.commit()
    elif not resume:
        raise FileExistsError("audit database already exists; pass --resume explicitly")
    elif row[0] != encoded:
        raise ValueError("resume identity differs in root, config, revision, or implementation")


def _json_frame_paths(directory: Path, clip_name: str) -> list[Path]:
    # The strict decoder validates naming and contiguity.  We repeat only the
    # deterministic ordering needed by the byte-level provenance root.
    paths = sorted(directory.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() for path in paths):
        raise ValueError(f"{clip_name}: symlinked OpenPose entry is forbidden")
    expected_suffix = "_keypoints.json"
    if any(not path.is_file() or not path.name.startswith(clip_name + "_")
           or not path.name.endswith(expected_suffix) for path in paths):
        raise ValueError(f"{clip_name}: unsupported OpenPose directory entry")
    return paths


def _hash_openpose_tree(directory: Path, root: Path, clip_name: str
                        ) -> tuple[str, tuple[tuple[Path, StableFileDigest], ...]]:
    paths = _json_frame_paths(directory, clip_name)
    snapshots = tuple((path, stable_sha256(path, root)) for path in paths)
    root_digest = hierarchical_digest((path.name, digest)
                                      for path, digest in snapshots)
    return root_digest, snapshots


def _quality_warning(quality: Mapping, diagnostics: Mapping) -> bool:
    if any(int(diagnostics.get(key, 0)) > 0 for key in (
        "empty_person_frames", "invalid_confidences", "out_of_frame_coordinates",
    )):
        return True
    if quality.get("fully_frozen_on_comparable_pairs"):
        return True
    return any(region["coverage"] == 0 for region in quality["regions"].values())


def _compute_row(root_value: str, row: How2SignRow,
                 config: How2SignAuditConfig) -> dict:
    """Pure worker computation: source reads only, no shared database state."""
    root = Path(root_value)
    raw = root / "raw_videos" / f"{row.sentence_name}.mp4"
    rendered = root / "openpose_output" / "video" / f"{row.sentence_name}.mp4"
    keypoints = root / "openpose_output" / "json" / row.sentence_name
    base: dict = {
        "row": row, "thresholds": [],
        "raw_uri": os.fspath(raw.resolve(strict=False)) if raw.exists() else None,
        "rendered_uri": os.fspath(rendered.resolve(strict=False))
        if rendered.exists() else None,
    }
    missing = [os.fspath(path) for path in (raw, rendered, keypoints) if not path.exists()]
    raw_digest = rendered_digest = None
    openpose_digest = None
    try:
        if raw.exists():
            raw_digest = stable_sha256(raw, root)
            base["raw_sha256"] = raw_digest.sha256
        if rendered.exists():
            rendered_digest = stable_sha256(rendered, root)
            base["rendered_sha256"] = rendered_digest.sha256
        openpose_snapshots: tuple[tuple[Path, StableFileDigest], ...] = ()
        if keypoints.exists():
            if keypoints.is_symlink() or not keypoints.is_dir():
                raise ValueError("OpenPose artifact must be a real directory")
            openpose_digest, openpose_snapshots = _hash_openpose_tree(
                keypoints, root, row.sentence_name)
            base["openpose_sha256"] = openpose_digest
        if missing:
            return {
                **base, "status": "missing_source",
                "error": "missing required artifacts: " + ", ".join(missing),
            }
        assert raw_digest is not None and rendered_digest is not None
        track, diagnostics = decode_how2sign_openpose(raw, keypoints,
                                                       confidence_threshold=0.0)
        # Decode rendered media too: a hash alone cannot establish a playable review artifact.
        rendered_clock = decode_video_clock(rendered)
        if rendered_clock.timestamps.shape[0] != track.timestamps.shape[0]:
            raise ValueError("rendered/video frame-count mismatch")
        if not np.all(np.diff(rendered_clock.timestamps) > 0):
            raise ValueError("rendered video timestamps are not strictly increasing")
        assert_file_unchanged(raw, root, raw_digest)
        assert_file_unchanged(rendered, root, rendered_digest)
        for frame_path, frame_digest in openpose_snapshots:
            assert_file_unchanged(frame_path, root, frame_digest)
        quality = pose_quality_metrics(track)
        diagnostic_value = asdict(diagnostics)
        threshold_rows = threshold_metrics(track, config.thresholds)
        status = "quality_warning" if _quality_warning(quality, diagnostic_value) else "valid"
        duration = float(track.timestamps[-1] - track.timestamps[0]) \
            if track.timestamps.size > 1 else 0.0
        return {
            **base, "status": status, "error": None,
            "raw_uri": os.fspath(raw.resolve()),
            "rendered_uri": os.fspath(rendered.resolve()),
            "raw_sha256": raw_digest.sha256,
            "rendered_sha256": rendered_digest.sha256,
            "openpose_sha256": openpose_digest,
            "frame_count": int(track.timestamps.size), "duration": duration,
            "quality": quality, "diagnostics": diagnostic_value,
            "thresholds": threshold_rows,
        }
    except (OSError, UnicodeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        return {
            **base, "status": "structural_failure",
            "error": f"{type(error).__name__}: {error}",
        }


def _insert_scan_result(connection: sqlite3.Connection, result: Mapping) -> None:
    row = result["row"]
    connection.execute("""
        INSERT INTO clips(
            sample_id,video_id,sentence_id,sentence_name,filename_code,status,error,
            raw_uri,rendered_uri,raw_sha256,rendered_sha256,openpose_sha256,
            frame_count,duration,quality_json,diagnostics_json
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        row.sentence_name, row.video_id, row.sentence_id, row.sentence_name,
        row.filename_code, result["status"], result.get("error"),
        result.get("raw_uri"), result.get("rendered_uri"), result.get("raw_sha256"),
        result.get("rendered_sha256"), result.get("openpose_sha256"),
        result.get("frame_count"), result.get("duration"),
        json.dumps(result["quality"], sort_keys=True) if result.get("quality") else None,
        json.dumps(result["diagnostics"], sort_keys=True)
        if result.get("diagnostics") else None,
    ))
    connection.executemany(
        "INSERT INTO thresholds(sample_id,threshold,metrics_json) VALUES(?,?,?)",
        ((row.sentence_name, item["threshold"], json.dumps(item, sort_keys=True))
         for item in result["thresholds"]),
    )


def _scan_row(connection: sqlite3.Connection, root: Path, row: How2SignRow,
              config: How2SignAuditConfig) -> None:
    _insert_scan_result(connection, _compute_row(os.fspath(root), row, config))


def _record_orphans(connection: sqlite3.Connection, inventory, root: Path) -> None:
    categories = (
        ("raw_without_metadata", inventory.raw_without_metadata,
         lambda name: root / "raw_videos" / f"{name}.mp4", "raw"),
        ("openpose_json_without_raw", inventory.openpose_json_without_raw,
         lambda name: root / "openpose_output" / "json" / name, "openpose"),
        ("rendered_video_without_raw", inventory.rendered_video_without_raw,
         lambda name: root / "openpose_output" / "video" / f"{name}.mp4", "rendered"),
        ("unexpected_raw_entry", inventory.unexpected_raw_entries,
         lambda name: root / "raw_videos" / name, "raw"),
    )
    for category, names, locate, kind in categories:
        for name in names:
            sample_id = f"artifact:{category}:{name}"
            path = locate(name)
            fields = {"raw_uri": None, "rendered_uri": None, "raw_sha256": None,
                      "rendered_sha256": None, "openpose_sha256": None}
            error = category
            try:
                if kind == "openpose":
                    if path.is_symlink() or not path.is_dir():
                        raise ValueError("orphan OpenPose artifact is not a real directory")
                    digest, snapshots = _hash_openpose_tree(path, root, name)
                    for frame_path, frame_digest in snapshots:
                        assert_file_unchanged(frame_path, root, frame_digest)
                    fields["openpose_sha256"] = digest
                else:
                    digest = stable_sha256(path, root)
                    assert_file_unchanged(path, root, digest)
                    fields[f"{kind}_uri"] = os.fspath(path.resolve())
                    fields[f"{kind}_sha256"] = digest.sha256
            except (OSError, ValueError, RuntimeError) as exception:
                error += f"; hash_error={type(exception).__name__}: {exception}"
            connection.execute("""
                INSERT OR IGNORE INTO clips(
                    sample_id,sentence_name,status,error,raw_uri,rendered_uri,
                    raw_sha256,rendered_sha256,openpose_sha256
                ) VALUES(?,?,'unjoinable_artifact',?,?,?,?,?,?)
            """, (sample_id, name, error, fields["raw_uri"], fields["rendered_uri"],
                  fields["raw_sha256"], fields["rendered_sha256"],
                  fields["openpose_sha256"]))


def _duration_stratum(duration: float | None) -> str:
    if duration is None:
        return "unknown"
    if duration < 2:
        return "short"
    if duration < 6:
        return "medium"
    return "long"


def _load_review_candidates(connection: sqlite3.Connection) -> list[dict]:
    rows = connection.execute("""
        SELECT sample_id,video_id,sentence_name,filename_code,status,error,raw_uri,
               rendered_uri,frame_count,duration,quality_json,diagnostics_json
        FROM clips ORDER BY sample_id
    """).fetchall()
    names = ("sample_id", "video_id", "sentence_name", "filename_code", "status",
             "error", "raw_uri", "rendered_uri", "frame_count", "duration",
             "quality_json", "diagnostics_json")
    result = []
    for raw in rows:
        item = dict(zip(names, raw))
        item["quality"] = json.loads(item.pop("quality_json")) if item["quality_json"] else {}
        item["diagnostics"] = (json.loads(item.pop("diagnostics_json"))
                               if item["diagnostics_json"] else {})
        result.append(item)
    return result


def _review_selection(candidates: Sequence[dict], config: How2SignAuditConfig) -> list[dict]:
    rng = random.Random(config.random_seed)
    chosen: dict[str, set[str]] = {}

    def add(reason: str, values: Sequence[dict]) -> None:
        for item in values[:config.review_per_category]:
            chosen.setdefault(item["sample_id"], set()).add(reason)

    by_status: dict[str, list[dict]] = {}
    for item in candidates:
        by_status.setdefault(item["status"], []).append(item)
    for status, values in sorted(by_status.items()):
        add(f"status:{status}", values)

    valid_quality = [item for item in candidates if item["quality"]]
    for region in ("body", "left_hand", "right_hand", "face"):
        ranked = sorted(valid_quality,
                        key=lambda item: item["quality"]["regions"][region]["coverage"])
        add(f"lowest_{region}_coverage", ranked)
    ranked_swap = sorted(
        valid_quality,
        key=lambda item: item["quality"]["left_right_swap_signal"]
        .get("swap_advantage_fraction") or -1,
        reverse=True,
    )
    add("largest_left_right_swap_signal", ranked_swap)
    ranked_acceleration = sorted(
        valid_quality,
        key=lambda item: item["quality"]["derivatives"].get("acceleration_median")
        if item["quality"]["derivatives"].get("acceleration_median") is not None
        else -1,
        reverse=True,
    )
    add("largest_2d_acceleration", ranked_acceleration)
    ranked_repeated = sorted(
        valid_quality,
        key=lambda item: item["quality"]["repeated_frames"]["exact_repeated_fraction"],
        reverse=True,
    )
    add("largest_exact_repeated_frame_fraction", ranked_repeated)

    coverage_values = sorted(item["quality"]["coverage"] for item in valid_quality)

    def quality_decile(item: dict) -> str:
        if not coverage_values:
            return "unknown"
        rank = int(np.searchsorted(coverage_values, item["quality"]["coverage"],
                                   side="right"))
        return str(min(9, (10 * rank) // max(1, len(coverage_values))))

    strata: dict[tuple[str, str, str], list[dict]] = {}
    by_source: dict[str, list[dict]] = {}
    for item in valid_quality:
        key = (item.get("filename_code") or "unknown",
               _duration_stratum(item.get("duration")), quality_decile(item))
        strata.setdefault(key, []).append(item)
        by_source.setdefault(item.get("video_id") or "unknown", []).append(item)
    for key, values in sorted(strata.items()):
        pool = list(values)
        rng.shuffle(pool)
        add("stratum:" + "|".join(key), pool)
    source_representatives = []
    for source, values in sorted(by_source.items()):
        pool = list(values)
        rng.shuffle(pool)
        source_representatives.append(pool[0])
    rng.shuffle(source_representatives)
    add("deterministic_source_sample", source_representatives)

    output = []
    by_id = {item["sample_id"]: item for item in candidates}
    for sample_id in sorted(chosen):
        item = dict(by_id[sample_id])
        item["reasons"] = sorted(chosen[sample_id])
        output.append(item)
    return output


def _media_href(path: str | None, root: Path) -> str:
    if not path:
        return ""
    resolved = Path(path).resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    if root_resolved not in resolved.parents:
        raise ValueError("review media path escapes dataset root")
    return "file://" + quote(os.fspath(resolved), safe="/")


def write_review_queue(connection: sqlite3.Connection, root: Path, output: Path,
                       config: How2SignAuditConfig) -> int:
    selected = _review_selection(_load_review_candidates(connection), config)
    jsonl_path = output / "review_queue.jsonl"
    csv_path = output / "review_queue.csv"
    html_path = output / "review_queue.html"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for item in selected:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    columns = ("sample_id", "video_id", "sentence_name", "status", "reasons",
               "raw_uri", "rendered_uri", "review_decision", "review_notes")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in selected:
            writer.writerow({
                "sample_id": item["sample_id"], "video_id": item.get("video_id"),
                "sentence_name": item.get("sentence_name"), "status": item["status"],
                "reasons": ";".join(item["reasons"]), "raw_uri": item.get("raw_uri"),
                "rendered_uri": item.get("rendered_uri"), "review_decision": "",
                "review_notes": "",
            })
    rows = []
    for item in selected:
        raw_href = _media_href(item.get("raw_uri"), root) if item.get("raw_uri") else ""
        rendered_href = (_media_href(item.get("rendered_uri"), root)
                         if item.get("rendered_uri") else "")
        links = []
        if raw_href:
            links.append(f'<a href="{html.escape(raw_href, quote=True)}">source</a>')
        if rendered_href:
            links.append(f'<a href="{html.escape(rendered_href, quote=True)}">OpenPose</a>')
        rows.append(
            "<tr><td>" + html.escape(str(item["sample_id"])) + "</td><td>"
            + html.escape(str(item["status"])) + "</td><td>"
            + html.escape(", ".join(item["reasons"])) + "</td><td>"
            + " / ".join(links) + "</td><td>"
            + html.escape(json.dumps(item.get("quality", {}), sort_keys=True))
            + "</td><td>☐ source ☐ landmarks ☐ uncertainty ☐ notes</td></tr>"
        )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>How2Sign review queue</title>"
        "<h1>How2Sign technical review queue</h1>"
        "<p>This is a queue, not qualified signer attestation or linguistic approval.</p>"
        "<table border='1'><thead><tr><th>Sample</th><th>Status</th><th>Reasons</th>"
        "<th>Media</th><th>Metrics</th><th>Checklist</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>", encoding="utf-8",
    )
    return len(selected)


def _write_threshold_summary(connection: sqlite3.Connection, output: Path,
                             config: How2SignAuditConfig) -> None:
    path = output / "threshold_sweep.csv"
    columns = ("threshold", "legacy_adapter_default", "clip_count", "coverage_mean",
               "body_coverage_mean", "left_hand_coverage_mean",
               "right_hand_coverage_mean", "face_coverage_mean",
               "complete_frame_coverage_mean", "max_missing_run", "pair_count",
               "triple_count", "unusable_clips")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for threshold in config.thresholds:
            values = [json.loads(row[0]) for row in connection.execute(
                "SELECT metrics_json FROM thresholds WHERE threshold=? ORDER BY sample_id",
                (threshold,),
            )]
            count = len(values)
            mean = lambda items: float(np.mean(items)) if items else 0.0
            writer.writerow({
                "threshold": threshold,
                "legacy_adapter_default": abs(threshold - 0.30) < 1e-12,
                "clip_count": count,
                "coverage_mean": mean([item["coverage"] for item in values]),
                "body_coverage_mean": mean([item["region_coverage"]["body"] for item in values]),
                "left_hand_coverage_mean": mean([item["region_coverage"]["left_hand"] for item in values]),
                "right_hand_coverage_mean": mean([item["region_coverage"]["right_hand"] for item in values]),
                "face_coverage_mean": mean([item["region_coverage"]["face"] for item in values]),
                "complete_frame_coverage_mean": mean(
                    [item["complete_frame_coverage"] for item in values]),
                "max_missing_run": max(
                    (item["longest_fully_missing_run"] for item in values), default=0),
                "pair_count": sum(item["derivative_valid_pairs"] for item in values),
                "triple_count": sum(item["derivative_valid_triples"] for item in values),
                "unusable_clips": sum(bool(item["unusable"]) for item in values),
            })


def _write_source_groups(connection: sqlite3.Connection, output: Path) -> None:
    groups: dict[str, list[str]] = {}
    rows = connection.execute("""
        SELECT video_id,sample_id FROM clips
        WHERE video_id IS NOT NULL AND status IN ('valid','quality_warning')
        ORDER BY video_id,sample_id
    """)
    for video_id, sample_id in rows:
        groups.setdefault(video_id, []).append(sample_id)
    duplicate_hashes: dict[str, list[str]] = {}
    rows = connection.execute("""
        SELECT raw_sha256,sample_id FROM clips WHERE raw_sha256 IS NOT NULL
        ORDER BY raw_sha256,sample_id
    """)
    for digest, sample_id in rows:
        duplicate_hashes.setdefault(digest, []).append(sample_id)
    duplicate_hashes = {key: value for key, value in duplicate_hashes.items()
                        if len(value) > 1}
    payload = {
        "grouping_key": "official VIDEO_ID",
        "source_groups": groups,
        "duplicate_media_constraints": duplicate_hashes,
        "filename_code_is_signer_identity": False,
        "final_split_created": False,
        "signer_leakage_certificate": None,
        "blocker": "authoritative signer identity mapping is absent",
    }
    (output / "source_groups.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_landmark_summary(connection: sqlite3.Connection, output: Path) -> None:
    clip_count = int(connection.execute(
        "SELECT COUNT(*) FROM clips WHERE quality_json IS NOT NULL").fetchone()[0])
    joint_count = len(OPENPOSE_JOINT_NAMES)
    coverage_matrix = np.empty((clip_count, joint_count), dtype=np.float32)
    confidence_matrix = np.full((clip_count, joint_count), np.nan, dtype=np.float32)
    longest_matrix = np.empty((clip_count, joint_count), dtype=np.int32)
    for row_index, (encoded,) in enumerate(connection.execute(
            "SELECT quality_json FROM clips WHERE quality_json IS NOT NULL ORDER BY sample_id")):
        landmarks = json.loads(encoded)["landmarks"]
        coverage_matrix[row_index] = landmarks["coverage"]
        confidence_matrix[row_index] = [
            np.nan if value is None else value for value in landmarks["confidence_median"]
        ]
        longest_matrix[row_index] = landmarks["longest_missing_run"]
    columns = ("joint_index", "joint_name", "clip_count", "coverage_mean",
               "coverage_median", "coverage_q05", "coverage_q95",
               "confidence_median_of_clip_medians", "longest_missing_run_max")
    with (output / "landmark_summary.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, joint_name in enumerate(OPENPOSE_JOINT_NAMES):
            coverage = coverage_matrix[:, index].astype(np.float64)
            confidences = confidence_matrix[:, index].astype(np.float64)
            confidences = confidences[np.isfinite(confidences)]
            writer.writerow({
                "joint_index": index, "joint_name": joint_name,
                "clip_count": clip_count,
                "coverage_mean": float(coverage.mean()) if coverage.size else None,
                "coverage_median": float(np.median(coverage)) if coverage.size else None,
                "coverage_q05": float(np.quantile(coverage, 0.05)) if coverage.size else None,
                "coverage_q95": float(np.quantile(coverage, 0.95)) if coverage.size else None,
                "confidence_median_of_clip_medians": float(np.median(confidences))
                if confidences.size else None,
                "longest_missing_run_max": int(longest_matrix[:, index].max())
                if clip_count else 0,
            })


def _database_digest(path: Path) -> StableFileDigest:
    # Root is the output directory; the same stable-file rules apply.
    return stable_sha256(path, path.parent)


def _content_digest_manifest(digest: StableFileDigest) -> dict[str, str | int]:
    """Generated artifacts bind content, not incidental filesystem timestamps."""
    return {"sha256": digest.sha256, "size": digest.size}


def finalize_audit(connection: sqlite3.Connection, root: Path, output: Path,
                   config: How2SignAuditConfig, metadata_rows: int,
                   inventory) -> dict:
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    status_counts = dict(connection.execute(
        "SELECT status,COUNT(*) FROM clips GROUP BY status ORDER BY status").fetchall())
    accounted_metadata = int(connection.execute(
        "SELECT COUNT(*) FROM clips WHERE video_id IS NOT NULL").fetchone()[0])
    if accounted_metadata != metadata_rows:
        raise RuntimeError(
            f"metadata reconciliation failed: {accounted_metadata} != {metadata_rows}")
    if any(status not in STATUS_VALUES for status in status_counts):
        raise RuntimeError("database contains an unsupported status")
    review_count = write_review_queue(connection, root, output, config)
    _write_threshold_summary(connection, output, config)
    _write_source_groups(connection, output)
    _write_landmark_summary(connection, output)
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db_digest = _database_digest(output / "audit.sqlite3")
    identity_row = connection.execute(
        "SELECT value_json FROM meta WHERE key='identity'").fetchone()
    if identity_row is None:
        raise RuntimeError("audit identity disappeared before finalization")
    identity = json.loads(identity_row[0])
    generated_paths = (
        output / "audit.sqlite3",
        output / "review_queue.jsonl",
        output / "review_queue.csv",
        output / "review_queue.html",
        output / "threshold_sweep.csv",
        output / "source_groups.json",
        output / "landmark_summary.csv",
    )
    generated_artifacts = {
        path.name: _content_digest_manifest(stable_sha256(path, output))
        for path in generated_paths
    }
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_complete": True,
        "translation_ready": False,
        "stage_b_approved": False,
        "stage_c_approved": False,
        "metadata_rows": metadata_rows,
        "accounted_metadata_rows": accounted_metadata,
        "status_counts": status_counts,
        "review_queue_records": review_count,
        "thresholds": list(config.thresholds),
        "threshold_selection_performed": False,
        "known_blockers": [
            "authentic gloss annotations are absent",
            "authoritative signer identities are absent",
            "qualified target-language signer review is absent",
        ],
        "audit_database": _content_digest_manifest(db_digest),
        "generated_artifacts": generated_artifacts,
        "identity": identity,
        "filesystem_inventory": asdict(inventory),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    manifest_path = output / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    return manifest


def run_how2sign_audit(
    root: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    config: How2SignAuditConfig = How2SignAuditConfig(),
    resume: bool = False,
    max_rows: int | None = None,
    stop_after: int | None = None,
    selected_sample_ids: Sequence[str] | None = None,
    workers: int = 1,
) -> dict:
    """Run or resume an audit; return progress or the final manifest.

    ``max_rows`` is a bounded pilot control. ``stop_after`` simulates an
    interruption and is intended for adversarial testing. Neither is part of the
    scientific identity, so a pilot can resume into the complete scan.
    """
    config.validate()
    if workers < 1 or workers > 8:
        raise ValueError("workers must be in [1, 8]")
    if workers > 1 and stop_after is not None:
        raise ValueError("deliberate interruption testing requires workers=1")
    source_root = Path(root).resolve(strict=True)
    destination = Path(output)
    destination_resolved = destination.resolve(strict=False)
    if source_root == destination_resolved:
        raise ValueError("audit output must not replace the source dataset")
    protected = (
        source_root / "raw_videos",
        source_root / "openpose_output",
    )
    if any(path.resolve(strict=True) == destination_resolved
           or path.resolve(strict=True) in destination_resolved.parents
           for path in protected):
        raise ValueError("audit output must not be inside a source-media directory")
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("audit output must be a real directory")
        entries = list(destination.iterdir())
        if entries and not resume:
            raise FileExistsError("refusing non-empty audit output without explicit resume")
    else:
        if resume:
            raise FileNotFoundError("resume output directory does not exist")
        destination.mkdir(parents=True)

    metadata_path = source_root / "how2sign_realigned_train.csv"
    evidence_path = source_root / "LICENSE-HOW2SIGN-EVIDENCE.md"
    metadata_digest = stable_sha256(metadata_path, source_root)
    evidence_digest = stable_sha256(evidence_path, source_root)
    rows = read_how2sign_metadata(metadata_path)
    inventory = inspect_how2sign_root(source_root)
    payload = _meta_payload(source_root, config)
    payload["metadata_sha256"] = metadata_digest.sha256
    payload["license_evidence_sha256"] = evidence_digest.sha256

    connection = _connect_database(destination / "audit.sqlite3")
    try:
        _initialize_database(connection, payload, resume=resume)
        completed = {row[0] for row in connection.execute("SELECT sample_id FROM clips")}
        processed_now = 0
        selected = set(selected_sample_ids) if selected_sample_ids is not None else None
        if selected is not None:
            available = {row.sentence_name for row in rows}
            unknown = sorted(selected - available)
            if unknown:
                raise ValueError(f"selected sample IDs are absent from metadata: {unknown}")
        scan_rows = [row for row in rows if selected is None or row.sentence_name in selected]
        pending = [row for row in scan_rows if row.sentence_name not in completed]
        if max_rows is not None:
            if max_rows < 0:
                raise ValueError("max_rows must be non-negative")
            pending = pending[:max_rows]
        if workers == 1:
            for row in pending:
                with connection:
                    _scan_row(connection, source_root, row, config)
                processed_now += 1
                if stop_after is not None and processed_now >= stop_after:
                    raise InterruptedError("deliberate audit interruption")
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                results = executor.map(
                    _compute_row,
                    (os.fspath(source_root) for _ in pending), pending,
                    (config for _ in pending), chunksize=1,
                )
                for result in results:
                    with connection:
                        _insert_scan_result(connection, result)
                    processed_now += 1
        accounted = int(connection.execute(
            "SELECT COUNT(*) FROM clips WHERE video_id IS NOT NULL").fetchone()[0])
        if accounted == len(rows):
            with connection:
                _record_orphans(connection, inventory, source_root)
            return finalize_audit(connection, source_root, destination, config, len(rows),
                                  inventory)
        return {
            "audit_complete": False,
            "metadata_rows": len(rows),
            "accounted_metadata_rows": accounted,
            "remaining_metadata_rows": len(rows) - accounted,
            "selected_rows_complete": bool(selected is not None and all(
                row.sentence_name in {
                    item[0] for item in connection.execute("SELECT sample_id FROM clips")
                } for row in scan_rows)),
        }
    finally:
        connection.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable, label-free How2Sign structural and 2D quality audit.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_how2sign_audit(
        args.root, args.output, resume=args.resume, max_rows=args.max_rows,
        selected_sample_ids=args.sample_ids, workers=args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("audit_complete") else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
