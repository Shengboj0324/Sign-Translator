"""Strict How2Sign filesystem and OpenPose ingestion tests."""

import json

import numpy as np
import pytest
import torch

from signtranslator.config import DiffusionConfig, ModelConfig
from signtranslator.data_engineering import (
    OPENPOSE_JOINT_NAMES, OPENPOSE_LANDMARK_PARTS, decode_how2sign_openpose,
    inspect_how2sign_root, openpose_holistic_graph, read_how2sign_metadata,
)
from signtranslator.data_engineering.how2sign import OPENPOSE_PARTS
from signtranslator.models import SignTranslator


CLIP = "source_1-5-rgb_front"


def _write_video(path, frames=2):
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            image = np.full((16, 16, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _person(score=0.9):
    person = {key: [] for key in (
        "pose_keypoints_3d", "face_keypoints_3d",
        "hand_left_keypoints_3d", "hand_right_keypoints_3d",
    )}
    for key, count in OPENPOSE_PARTS:
        person[key] = [coordinate for _ in range(count) for coordinate in (8, 8, score)]
    return person


def _write_frame(directory, index, people):
    payload = {"version": 1.3, "people": people}
    path = directory / f"{CLIP}_{index:012d}_keypoints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_metadata(path, rows=None):
    header = "\t".join((
        "VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE_NAME",
        "START_REALIGNED", "END_REALIGNED", "SENTENCE",
    ))
    rows = rows or [f"source\tsource-5-rgb_front\tsource_1\t{CLIP}\t1.0\t2.0\tHello"]
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _layout(tmp_path, frames=2):
    raw = tmp_path / "raw_videos"
    rendered = tmp_path / "openpose_output" / "video"
    keypoints = tmp_path / "openpose_output" / "json" / CLIP
    raw.mkdir()
    rendered.mkdir(parents=True)
    keypoints.mkdir(parents=True)
    _write_metadata(tmp_path / "how2sign_realigned_train.csv")
    _write_video(raw / f"{CLIP}.mp4", frames)
    _write_video(rendered / f"{CLIP}.mp4", frames)
    return raw, rendered, keypoints


def test_metadata_is_tab_delimited_unique_and_interval_checked(tmp_path):
    path = tmp_path / "metadata.csv"
    _write_metadata(path)
    row = read_how2sign_metadata(path)[0]
    assert row.sentence_name == CLIP
    assert row.filename_code == "5"
    assert (row.start_realigned, row.end_realigned) == (1.0, 2.0)

    _write_metadata(path, [
        f"source\tsource-5-rgb_front\tsource_1\t{CLIP}\t1\t2\tOne",
        f"source\tsource-5-rgb_front\tsource_1\t{CLIP}\t2\t3\tTwo",
    ])
    with pytest.raises(ValueError, match="duplicate SENTENCE_NAME"):
        read_how2sign_metadata(path)

    _write_metadata(path, [f"source\tsource-5-rgb_front\tsource_1\t{CLIP}\t2\t1\tBad"])
    with pytest.raises(ValueError, match="START_REALIGNED"):
        read_how2sign_metadata(path)


def test_inventory_uses_exact_names_and_keeps_gate_failures_visible(tmp_path):
    raw, _, keypoints = _layout(tmp_path)
    _write_frame(keypoints, 0, [_person()])
    _write_frame(keypoints, 1, [_person()])
    (raw / "orphan").write_bytes(b"not silently accepted")
    inventory = inspect_how2sign_root(tmp_path)
    assert inventory.complete_clip_joins == 1
    assert not inventory.structurally_ready
    assert not inventory.translation_ready
    assert inventory.unexpected_raw_entries == ("orphan",)
    assert inventory.gloss_artifacts == ()
    assert inventory.license_artifacts == ()


def test_openpose_decode_is_honest_2d_and_invalidates_without_clipping(tmp_path):
    raw, _, keypoints = _layout(tmp_path)
    first = _person()
    first["pose_keypoints_2d"][2] = 1.1       # invalid confidence
    first["pose_keypoints_2d"][5] = 0.2       # below threshold
    first["pose_keypoints_2d"][6] = 16.0      # outside x bound
    _write_frame(keypoints, 0, [first])
    _write_frame(keypoints, 1, [_person()])

    track, diagnostics = decode_how2sign_openpose(raw / f"{CLIP}.mp4", keypoints)
    assert track.values.shape == (2, 2, len(OPENPOSE_JOINT_NAMES))
    assert set(OPENPOSE_LANDMARK_PARTS) == {"body", "left_hand", "right_hand", "face"}
    assert sum(map(len, OPENPOSE_LANDMARK_PARTS.values())) == len(OPENPOSE_JOINT_NAMES)
    assert np.array_equal(track.timestamps, np.array([0.0, 0.04]))
    assert diagnostics.invalid_confidences == 1
    assert diagnostics.below_threshold_observations == 1
    assert diagnostics.out_of_frame_coordinates == 1
    assert diagnostics.valid_observations == 2 * len(OPENPOSE_JOINT_NAMES) - 3
    assert not track.validity_mask[0, :3].any()
    assert np.all(track.values[:, 0, :3] == 0)
    assert np.all(track.confidence[0, :3] == 0)
    assert np.allclose(track.values[:, 1, 0], 0.5)
    assert np.all((track.values[:, track.validity_mask] >= 0)
                  & (track.values[:, track.validity_mask] <= 1))


def test_openpose_rejects_frame_gaps_counts_and_ambiguous_people(tmp_path):
    raw, _, keypoints = _layout(tmp_path)
    _write_frame(keypoints, 0, [_person()])
    _write_frame(keypoints, 2, [_person()])
    with pytest.raises(ValueError, match="not contiguous"):
        decode_how2sign_openpose(raw / f"{CLIP}.mp4", keypoints)

    for path in keypoints.iterdir():
        path.unlink()
    _write_frame(keypoints, 0, [_person(), _person()])
    _write_frame(keypoints, 1, [_person()])
    with pytest.raises(ValueError, match="multiple people"):
        decode_how2sign_openpose(raw / f"{CLIP}.mp4", keypoints)


def test_openpose_empty_person_frame_remains_all_invalid(tmp_path):
    raw, _, keypoints = _layout(tmp_path, frames=1)
    _write_frame(keypoints, 0, [])
    track, diagnostics = decode_how2sign_openpose(raw / f"{CLIP}.mp4", keypoints)
    assert diagnostics.empty_person_frames == 1
    assert diagnostics.valid_observations == 0
    assert not track.validity_mask.any()
    assert not track.values.any()
    assert not track.confidence.any()


def test_openpose_holistic_graph_is_connected_and_matches_joint_order():
    graph = openpose_holistic_graph()
    assert graph.num_nodes == len(OPENPOSE_JOINT_NAMES)
    assert len(graph.edges) == len(set(tuple(sorted(edge)) for edge in graph.edges))
    assert graph.hop.min() == 0
    assert graph.hop.max() > 0
    assert np.isfinite(graph.adjacency()).all()


def test_2d_holistic_graph_completes_model_forward_and_backward():
    model_config = ModelConfig(
        num_joints=len(OPENPOSE_JOINT_NAMES), in_channels=2, num_frames=4,
        stgcn_channels=(8,), text_embed_dim=8, text_layers=1, text_heads=2,
        latent_dim=8, speech_input_dim=4,
    )
    diffusion_config = DiffusionConfig(
        num_timesteps=4, denoiser_dim=8, denoiser_layers=1, denoiser_heads=2,
    )
    model = SignTranslator(
        model_config, diffusion_config, graph=openpose_holistic_graph(),
    )
    pose = torch.rand(2, 2, 4, len(OPENPOSE_JOINT_NAMES))
    tokens = torch.tensor([[3, 4], [4, 3]])
    result = model(pose, tokens)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
