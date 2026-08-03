"""Adversarial tests for the quarantined How2Sign 2D motion experiment."""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import av
import numpy as np
import pytest
import torch

from signtranslator.pretraining.how2sign_motion import (
    Pose2DExperimentConfig,
    Pose2DMaskedReconstructor,
    build_masked_input,
    coordinate_mean_baseline,
    last_observation_baseline,
    make_artificial_mask,
    masked_coordinate_error,
    source_disjoint_partition,
    temporal_interpolation_baseline,
    run_pose2d_experiment,
)
from signtranslator.data_engineering.how2sign import OPENPOSE_PARTS
from signtranslator.data_engineering.how2sign_audit import (
    hierarchical_digest, stable_sha256,
)
from signtranslator.pretraining.masking import (
    random_point_mask, span_mask, typical_masked_floor,
)


def _batch(frames=12):
    torch.manual_seed(0)
    values = torch.rand(2, 2, frames, 137)
    confidence = torch.rand(2, frames, 137)
    validity = torch.ones(2, frames, 137, dtype=torch.bool)
    validity[:, 3, 10] = False
    confidence[~validity] = 0
    return values, confidence, validity


def test_hidden_target_identity_cannot_enter_model_input_or_prediction():
    values, confidence, validity = _batch()
    mask = make_artificial_mask(validity, "span", seed=4, span_length=4)
    features_a, targets = build_masked_input(values, confidence, validity, mask)
    altered = values.clone()
    altered.masked_fill_(targets[:, None], 0.999)
    features_b, targets_b = build_masked_input(altered, confidence, validity, mask)
    assert torch.equal(features_a, features_b)
    assert torch.equal(targets, targets_b)
    model = Pose2DMaskedReconstructor(hidden_channels=8).eval()
    with torch.no_grad():
        assert torch.equal(model(features_a), model(features_b))


def test_genuine_missing_points_have_exactly_zero_loss_gradient():
    values, confidence, validity = _batch()
    mask = make_artificial_mask(validity, "point", seed=2, ratio=0.4)
    _, targets = build_masked_input(values, confidence, validity, mask)
    assert not targets[:, 3, 10].any()
    prediction = torch.rand_like(values, requires_grad=True)
    loss = masked_coordinate_error(prediction, values, targets)
    loss.backward()
    assert torch.all(prediction.grad[:, :, 3, 10] == 0)
    gradient_by_point = prediction.grad.permute(0, 2, 3, 1)
    assert gradient_by_point[targets].abs().sum() > 0


def test_masks_are_reproducible_and_spans_defeat_interpolation_more_than_points():
    validity = torch.ones(1, 40, 137, dtype=torch.bool)
    first = make_artificial_mask(validity, "span", seed=7, span_length=8)
    second = make_artificial_mask(validity, "span", seed=7, span_length=8)
    assert torch.equal(first, second)
    point = random_point_mask(40, 4, 0.25, seed=7)
    span = span_mask(40, 4, span_len=8, num_spans=2, seed=7)
    assert typical_masked_floor(span) > typical_masked_floor(point)


def test_region_tube_masks_only_declared_region():
    validity = torch.ones(1, 10, 137, dtype=torch.bool)
    mask = make_artificial_mask(validity, "left_hand_tube", seed=0)
    assert mask[:, :, 25:46].all()
    assert not mask[:, :, :25].any()
    assert not mask[:, :, 46:].any()


def test_model_loss_gradients_are_finite_and_checkpoint_exact():
    values, confidence, validity = _batch(frames=8)
    mask = make_artificial_mask(validity, "span", seed=3, span_length=3)
    features, targets = build_masked_input(values, confidence, validity, mask)
    model = Pose2DMaskedReconstructor(hidden_channels=8).eval()
    prediction = model(features)
    loss = masked_coordinate_error(prediction, values, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    reloaded = Pose2DMaskedReconstructor(hidden_channels=8).eval()
    reloaded.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        assert torch.equal(model(features), reloaded(features))


def test_tiny_continuous_reconstruction_overfits_without_language():
    values, confidence, validity = _batch(frames=8)
    values = values[:1]
    confidence = confidence[:1]
    validity = validity[:1]
    mask = make_artificial_mask(validity, "span", seed=5, span_length=3)
    features, targets = build_masked_input(values, confidence, validity, mask)
    model = Pose2DMaskedReconstructor(hidden_channels=12)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    initial = None
    for step in range(100):
        prediction = model(features)
        loss = masked_coordinate_error(prediction, values, targets)
        if step == 0:
            initial = float(loss.detach())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    final = float(masked_coordinate_error(model(features), values, targets).detach())
    assert final < 0.65 * initial


def test_baselines_never_claim_unsupported_predictions():
    values = torch.zeros(1, 2, 5, 1)
    values[0, 0, :, 0] = torch.arange(5)
    visible = torch.tensor([[[True], [False], [False], [True], [False]]])
    interpolated, supported = temporal_interpolation_baseline(values, visible)
    assert supported[0, 1:3, 0].all() and not supported[0, 4, 0]
    assert torch.equal(interpolated[0, 0, 1:3, 0], torch.tensor([1.0, 2.0]))
    last, last_supported = last_observation_baseline(values, visible)
    assert last_supported[0, 1:3, 0].all() and last_supported[0, 4, 0]
    mean, mean_supported = coordinate_mean_baseline(values, visible)
    assert mean_supported.all()
    no_visible = torch.zeros_like(visible)
    for baseline in (temporal_interpolation_baseline, last_observation_baseline,
                     coordinate_mean_baseline):
        _, support = baseline(values, no_visible)
        assert not support.any()


def test_source_partition_is_total_deterministic_and_not_a_signer_claim():
    sources = [f"video-{index}" for index in range(20)]
    first = source_disjoint_partition(sources, seed=11)
    second = source_disjoint_partition(list(reversed(sources)), seed=11)
    assert first == second
    assert set(first) == set(sources)
    assert set(first.values()) == {"train", "val", "test"}
    with pytest.raises(ValueError, match="unique"):
        source_disjoint_partition(["same", "same"], seed=0)


def test_invalid_shapes_masks_and_empty_targets_fail_closed():
    values, confidence, validity = _batch()
    with pytest.raises(ValueError):
        build_masked_input(values[:, :, :, :-1], confidence, validity,
                           torch.zeros_like(validity))
    with pytest.raises(ValueError, match="no evaluable"):
        masked_coordinate_error(values, values, torch.zeros_like(validity))
    with pytest.raises(ValueError, match="unknown masking"):
        make_artificial_mask(validity, "invented", seed=0)


def _write_video(path: Path, frames: int) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width = 16; stream.height = 16; stream.pix_fmt = "yuv420p"
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), index * 8, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _person(frame: int) -> dict:
    person = {key: [] for key in (
        "pose_keypoints_3d", "face_keypoints_3d",
        "hand_left_keypoints_3d", "hand_right_keypoints_3d",
    )}
    person["person_id"] = [-1]
    for key, count in OPENPOSE_PARTS:
        person[key] = [coordinate for joint in range(count)
                       for coordinate in (4 + 0.1 * frame + joint % 2,
                                          5 + 0.05 * frame + joint % 3, 0.9)]
    return person


def _fake_completed_audit(tmp_path: Path, count: int = 8, frames: int = 8):
    root = tmp_path / "dataset"
    raw_root = root / "raw_videos"
    json_root = root / "openpose_output" / "json"
    raw_root.mkdir(parents=True); json_root.mkdir(parents=True)
    audit = tmp_path / "audit"; audit.mkdir()
    connection = sqlite3.connect(audit / "audit.sqlite3")
    connection.execute("""
        CREATE TABLE clips(
            sample_id TEXT, video_id TEXT, filename_code TEXT, raw_uri TEXT,
            raw_sha256 TEXT, openpose_sha256 TEXT, quality_json TEXT,
            duration REAL, status TEXT
        )
    """)
    for index in range(count):
        sample = f"sample_{index}-5-rgb_front"
        video = raw_root / f"{sample}.mp4"
        directory = json_root / sample; directory.mkdir()
        _write_video(video, frames)
        for frame in range(frames):
            path = directory / f"{sample}_{frame:012d}_keypoints.json"
            path.write_text(json.dumps({"version": 1.3, "people": [_person(frame)]}))
        tree = hierarchical_digest(
            (path.name, stable_sha256(path, root)) for path in sorted(directory.iterdir()))
        quality = {"frame_count": frames, "coverage": 0.8 + index * 0.01}
        connection.execute("INSERT INTO clips VALUES(?,?,?,?,?,?,?,?,?)", (
            sample, f"video-{index}", str(1 + index % 3), str(video.resolve()),
            stable_sha256(video, root).sha256, tree, json.dumps(quality), 1.0 + index,
            "valid",
        ))
    connection.commit(); connection.close()
    database_hash = stable_sha256(audit / "audit.sqlite3", audit).sha256
    (audit / "audit_manifest.json").write_text(json.dumps({
        "audit_complete": True, "audit_database": {"sha256": database_hash},
    }))
    return root, audit


def test_bounded_runner_writes_reloadable_nonlinguistic_artifacts(tmp_path):
    root, audit = _fake_completed_audit(tmp_path)
    output = tmp_path / "experiment"
    config = Pose2DExperimentConfig(
        sample_count=8, window_frames=8, epochs=1, batch_size=2,
        hidden_channels=8, span_length=3, tiny_overfit_steps=80,
    )
    result = run_pose2d_experiment(audit, root, output, config=config)
    assert result["experiment_complete"] is True
    assert result["translation_claim"] is False
    assert result["linguistic_claim"] is False
    assert result["three_dimensional_claim"] is False
    assert set(result["split_counts"]) == {"train", "val", "test"}
    assert (output / "checkpoint.pt").is_file()
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert "checkpoint.pt" in manifest["artifacts"]
    assert len(json.loads((output / "selected_samples.json").read_text())) == 8


def test_runner_rejects_tampered_audit_database(tmp_path):
    root, audit = _fake_completed_audit(tmp_path)
    with sqlite3.connect(audit / "audit.sqlite3") as connection:
        connection.execute("UPDATE clips SET raw_sha256='0' || substr(raw_sha256,2)")
        connection.commit()
    config = Pose2DExperimentConfig(
        sample_count=8, window_frames=8, epochs=1, batch_size=2,
        hidden_channels=8, span_length=3, tiny_overfit_steps=1,
    )
    with pytest.raises(ValueError, match="database hash"):
        run_pose2d_experiment(audit, root, tmp_path / "experiment", config=config)


def test_runner_rejects_media_that_no_longer_matches_audit(tmp_path):
    root, audit = _fake_completed_audit(tmp_path)
    with sqlite3.connect(audit / "audit.sqlite3") as connection:
        media = Path(connection.execute("SELECT raw_uri FROM clips LIMIT 1").fetchone()[0])
    with media.open("ab") as stream:
        stream.write(b"tampered")
    config = Pose2DExperimentConfig(
        sample_count=8, window_frames=8, epochs=1, batch_size=2,
        hidden_channels=8, span_length=3, tiny_overfit_steps=1,
    )
    with pytest.raises(ValueError, match="no longer matches"):
        run_pose2d_experiment(audit, root, tmp_path / "experiment", config=config)
