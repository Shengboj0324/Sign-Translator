"""Adversarial verification of the resumable, label-free How2Sign audit."""

from __future__ import annotations

import builtins
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import pytest

from signtranslator.data_engineering.exporter import LandmarkTrack
from signtranslator.data_engineering.how2sign import (
    OPENPOSE_PARTS, decode_how2sign_openpose,
)
from signtranslator.data_engineering.how2sign_audit import (
    DEFAULT_THRESHOLDS,
    How2SignAuditConfig,
    assert_file_unchanged,
    hierarchical_digest,
    nonuniform_derivative_summary,
    pose_quality_metrics,
    run_how2sign_audit,
    scaled_mad,
    stable_sha256,
    threshold_metrics,
)


CLIP = "source_1-5-rgb_front"


def _write_video(path: Path, frames: int = 4) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), index * 10, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _person(score: float = 0.9) -> dict:
    person = {key: [] for key in (
        "pose_keypoints_3d", "face_keypoints_3d",
        "hand_left_keypoints_3d", "hand_right_keypoints_3d",
    )}
    person["person_id"] = [-1]
    for key, count in OPENPOSE_PARTS:
        person[key] = [item for joint in range(count)
                       for item in (4.0 + joint % 4, 5.0 + joint % 3, score)]
    return person


def _dataset(root: Path, *, sentence: str = "Hello", frames: int = 4) -> None:
    raw = root / "raw_videos"
    rendered = root / "openpose_output" / "video"
    keypoints = root / "openpose_output" / "json" / CLIP
    raw.mkdir(parents=True)
    rendered.mkdir(parents=True)
    keypoints.mkdir(parents=True)
    metadata = (
        "VIDEO_ID\tVIDEO_NAME\tSENTENCE_ID\tSENTENCE_NAME\tSTART_REALIGNED\t"
        "END_REALIGNED\tSENTENCE\n"
        f"source\tsource-5-rgb_front\tsource_1\t{CLIP}\t0\t1\t{sentence}\n"
    )
    (root / "how2sign_realigned_train.csv").write_text(metadata, encoding="utf-8")
    (root / "LICENSE-HOW2SIGN-EVIDENCE.md").write_text("CC BY-NC 4.0 evidence",
                                                         encoding="utf-8")
    _write_video(raw / f"{CLIP}.mp4", frames)
    _write_video(rendered / f"{CLIP}.mp4", frames)
    for index in range(frames):
        payload = {"version": 1.3, "people": [_person()]}
        (keypoints / f"{CLIP}_{index:012d}_keypoints.json").write_text(
            json.dumps(payload), encoding="utf-8")


def _track() -> LandmarkTrack:
    times = np.array([0.0, 0.1, 0.3, 0.6], dtype=np.float64)
    values = np.zeros((2, 4, 137), dtype=np.float32)
    values[0] = np.array([0.0, 0.1, 0.4, 1.0])[:, None]
    values[1] = 0.25
    valid = np.ones((4, 137), dtype=np.bool_)
    valid[1, 30] = False
    confidence = np.full((4, 137), 0.8, dtype=np.float32)
    confidence[~valid] = 0
    return LandmarkTrack(values, confidence, valid, times)


def test_nonuniform_derivatives_and_zero_mad_are_explicit():
    summary = nonuniform_derivative_summary(_track())
    # Declared non-uniform clock: speeds 1, 1.5, 2 and accelerations 10/3, 2.
    assert summary.velocity_count == 3 * 137 - 2
    assert summary.acceleration_count == 2 * 137 - 2
    assert summary.velocity_median == pytest.approx(1.5)
    assert summary.acceleration_median == pytest.approx(8.0 / 3.0, abs=2e-6)
    median, mad = scaled_mad(np.ones(20))
    assert median == 1.0 and mad == 0.0
    assert scaled_mad(np.array([np.nan])) == (None, None)


def test_threshold_sets_are_nested_monotone_and_never_choose_optimum():
    rows = threshold_metrics(_track(), DEFAULT_THRESHOLDS)
    coverage = [row["coverage"] for row in rows]
    assert coverage == sorted(coverage, reverse=True)
    assert all("best_threshold" not in row and "recommended" not in row for row in rows)
    assert sum(row["legacy_adapter_default"] for row in rows) == 1


def test_quality_metrics_never_create_3d_or_anatomical_verdict():
    metrics = pose_quality_metrics(_track())
    assert metrics["joint_count"] == 137
    assert len(metrics["landmarks"]["coverage"]) == 137
    assert metrics["torso_relative_2d"]["count"] >= 0
    assert 0 <= metrics["repeated_frames"]["exact_repeated_fraction"] <= 1
    encoded = json.dumps(metrics)
    assert "3d" not in encoded.lower()
    assert "anatom" not in encoded.lower()


def test_stable_hash_rejects_symlink_and_mid_read_mutation(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"a" * 32)
    link = tmp_path / "link.bin"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        stable_sha256(link, tmp_path)

    real_open = builtins.open

    class MutatingReader:
        def __init__(self, stream):
            self.stream = stream
            self.changed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size=-1):
            block = self.stream.read(size)
            if block and not self.changed:
                self.changed = True
                with real_open(source, "ab") as target:
                    target.write(b"changed")
            return block

    original_path_open = Path.open

    def altered_open(path, *args, **kwargs):
        stream = original_path_open(path, *args, **kwargs)
        return MutatingReader(stream) if path == source.resolve() else stream

    monkeypatch.setattr(Path, "open", altered_open)
    with pytest.raises(RuntimeError, match="changed while hashing"):
        stable_sha256(source, tmp_path, block_size=4)


def test_post_decode_identity_check_closes_hash_validation_race(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    digest = stable_sha256(source, tmp_path)
    source.write_bytes(b"modified after hash")
    with pytest.raises(RuntimeError, match="changed after hashing"):
        assert_file_unchanged(source, tmp_path, digest)


def test_hierarchical_hash_requires_order_and_is_deterministic(tmp_path):
    left = tmp_path / "a"; right = tmp_path / "b"
    left.write_bytes(b"left"); right.write_bytes(b"right")
    a = stable_sha256(left, tmp_path); b = stable_sha256(right, tmp_path)
    assert hierarchical_digest([("a", a), ("b", b)]) == hierarchical_digest(
        [("a", a), ("b", b)])
    with pytest.raises(ValueError, match="strictly ordered"):
        hierarchical_digest([("b", b), ("a", a)])


def test_interrupted_audit_resumes_to_byte_stable_complete_manifest(tmp_path):
    root = tmp_path / "dataset"; root.mkdir()
    _dataset(root)
    output = tmp_path / "audit"
    with pytest.raises(InterruptedError):
        run_how2sign_audit(root, output, stop_after=1)
    result = run_how2sign_audit(root, output, resume=True)
    assert result["audit_complete"] is True
    assert result["accounted_metadata_rows"] == 1
    assert result["stage_b_approved"] is False
    assert result["threshold_selection_performed"] is False
    manifest_bytes = (output / "audit_manifest.json").read_bytes()
    repeated = run_how2sign_audit(root, output, resume=True)
    assert repeated["audit_complete"] is True
    assert (output / "audit_manifest.json").read_bytes() == manifest_bytes
    assert (output / "review_queue.html").is_file()
    assert len((output / "landmark_summary.csv").read_text().splitlines()) == 138
    source_groups = json.loads((output / "source_groups.json").read_text())
    assert source_groups["filename_code_is_signer_identity"] is False
    assert source_groups["final_split_created"] is False


def test_resume_refuses_configuration_drift_and_bad_output_scope(tmp_path):
    root = tmp_path / "dataset"; root.mkdir()
    _dataset(root)
    output = tmp_path / "audit"
    progress = run_how2sign_audit(root, output, max_rows=0)
    assert progress["audit_complete"] is False
    changed = How2SignAuditConfig(review_per_category=9)
    with pytest.raises(ValueError, match="resume identity differs"):
        run_how2sign_audit(root, output, config=changed, resume=True)
    with pytest.raises(ValueError, match="source-media"):
        run_how2sign_audit(root, root / "raw_videos" / "audit")


def test_parallel_module_cli_is_spawn_safe_and_complete(tmp_path):
    root = tmp_path / "dataset"; root.mkdir()
    _dataset(root)
    output = tmp_path / "audit"
    completed = subprocess.run([
        sys.executable, "-W", "error", "-m",
        "signtranslator.data_engineering.how2sign_audit",
        "--root", str(root), "--output", str(output), "--workers", "2",
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["audit_complete"] is True


def test_unexpected_schema_and_frame_gap_are_recorded_not_silenced(tmp_path):
    root = tmp_path / "dataset"; root.mkdir()
    _dataset(root)
    keypoints = root / "openpose_output" / "json" / CLIP
    first = keypoints / f"{CLIP}_{0:012d}_keypoints.json"
    payload = json.loads(first.read_text())
    payload["injected"] = "<script>alert(1)</script>"
    first.write_text(json.dumps(payload), encoding="utf-8")
    result = run_how2sign_audit(root, tmp_path / "audit")
    assert result["status_counts"]["structural_failure"] == 1
    with sqlite3.connect(tmp_path / "audit" / "audit.sqlite3") as connection:
        error = connection.execute("SELECT error FROM clips WHERE sample_id=?", (CLIP,)).fetchone()[0]
    assert "unexpected OpenPose top-level schema" in error


def test_nan_infinity_unicode_and_count_mismatch_fail_closed(tmp_path):
    root = tmp_path / "dataset"; root.mkdir()
    _dataset(root, frames=2)
    video = root / "raw_videos" / f"{CLIP}.mp4"
    keypoints = root / "openpose_output" / "json" / CLIP
    first = keypoints / f"{CLIP}_{0:012d}_keypoints.json"
    payload = json.loads(first.read_text())
    payload["people"][0]["pose_keypoints_2d"][2] = float("nan")
    payload["people"][0]["pose_keypoints_2d"][5] = float("inf")
    first.write_text(json.dumps(payload), encoding="utf-8")
    track, diagnostics = decode_how2sign_openpose(video, keypoints, confidence_threshold=0)
    assert diagnostics.invalid_confidences == 2
    assert not track.validity_mask[0, :2].any()

    payload["people"][0]["pose_keypoints_2d"][0] = "not-a-number-雪"
    first.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric"):
        decode_how2sign_openpose(video, keypoints, confidence_threshold=0)

    first.write_text(json.dumps({"version": 1.3, "people": [_person()]}),
                     encoding="utf-8")
    (keypoints / f"{CLIP}_{1:012d}_keypoints.json").unlink()
    with pytest.raises(ValueError, match="frame-count mismatch"):
        decode_how2sign_openpose(video, keypoints, confidence_threshold=0)


def test_rendered_frame_count_mismatch_is_structural_failure(tmp_path):
    root = tmp_path / "dataset"; root.mkdir()
    _dataset(root, frames=2)
    rendered = root / "openpose_output" / "video" / f"{CLIP}.mp4"
    rendered.unlink()
    _write_video(rendered, frames=3)
    result = run_how2sign_audit(root, tmp_path / "audit")
    assert result["status_counts"]["structural_failure"] == 1
    with sqlite3.connect(tmp_path / "audit" / "audit.sqlite3") as connection:
        error = connection.execute("SELECT error FROM clips WHERE sample_id=?",
                                   (CLIP,)).fetchone()[0]
    assert "rendered/video frame-count mismatch" in error


def test_missing_and_orphan_records_retain_every_obtainable_hash(tmp_path):
    root = tmp_path / "dataset"; root.mkdir()
    _dataset(root)
    (root / "raw_videos" / f"{CLIP}.mp4").unlink()
    orphan = root / "raw_videos" / "unjoinable.bin"
    orphan.write_bytes(b"orphan bytes")
    result = run_how2sign_audit(root, tmp_path / "audit")
    assert result["status_counts"]["missing_source"] == 1
    assert result["status_counts"]["unjoinable_artifact"] == 3
    with sqlite3.connect(tmp_path / "audit" / "audit.sqlite3") as connection:
        missing = connection.execute(
            "SELECT raw_sha256,rendered_sha256,openpose_sha256 FROM clips "
            "WHERE sample_id=?", (CLIP,)).fetchone()
        orphan_hash = connection.execute(
            "SELECT raw_sha256 FROM clips WHERE sample_id=?",
            ("artifact:unexpected_raw_entry:unjoinable.bin",)).fetchone()[0]
    assert missing[0] is None and all(missing[index] for index in (1, 2))
    assert orphan_hash == stable_sha256(orphan, root).sha256


def test_long_track_quality_is_linear_memory():
    frames = 20_000
    values = np.zeros((2, frames, 2), dtype=np.float32)
    values[0, :, 0] = np.linspace(0, 1, frames)
    valid = np.ones((frames, 2), dtype=np.bool_)
    confidence = np.ones((frames, 2), dtype=np.float32)
    track = LandmarkTrack(values, confidence, valid,
                          np.arange(frames, dtype=np.float64) / 30.0)
    summary = nonuniform_derivative_summary(track)
    assert summary.velocity_count == (frames - 1) * 2
    assert summary.acceleration_count == (frames - 2) * 2
