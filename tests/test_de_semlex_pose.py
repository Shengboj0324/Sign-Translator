"""Sem-Lex Holistic adapter: layout, aspect, pseudo-world mapping, face geometry port, refusals, declared gaps."""
from dataclasses import replace
import os
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semlex_synthetic import FakeBlendshapes, T, body_y_up, fake_resources, face_points, rotation, semlex_array  # noqa: E402

from signtranslator.data_engineering import semlex_pose as sp  # noqa: E402
from signtranslator.data_engineering.video_landmarks import canonical_from_landmarks, square_crop  # noqa: E402
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402
from signtranslator.pose.rotations import rotation_6d_to_matrix  # noqa: E402

FACE_TASK = Path(os.path.expanduser("~/.sign-translator-ops/models/face_landmarker.task"))


def convert(array, **kw):
    kw.setdefault("blendshape_model", FakeBlendshapes())
    kw.setdefault("pose_z_scale", 1.)
    kw.setdefault("hand_z_scale", 1.)
    return sp.convert_array(array, 1000. * len(array) / 30., kw.pop("resources", fake_resources()), video_id="v",
                            source_sha256="a" * 64, **kw)


def test_layout_partitions_the_553_points():
    blocks = [sp.FACE, sp.POSE, sp.LEFT_HAND, sp.RIGHT_HAND]
    covered = np.concatenate([np.arange(553)[b] for b in blocks])
    assert sorted(covered.tolist()) == list(range(553)) and sp.POINTS == 553
    assert [len(range(553)[b]) for b in blocks] == [478, 33, 21, 21]
    assert len(sp.BLENDSHAPE_SUBSET) == 146 == len(set(sp.BLENDSHAPE_SUBSET))
    assert set(range(468, 478)) <= set(sp.BLENDSHAPE_SUBSET) and max(sp.BLENDSHAPE_SUBSET) == 477
    assert len(sp.BLENDSHAPE_NAMES) == 52 and sp.BLENDSHAPE_NAMES[9:11] == ("eyeBlinkLeft", "eyeBlinkRight")


def _varint(n):
    out = b""
    while True:
        byte, n = n & 0x7F, n >> 7
        out += bytes([byte | (0x80 if n else 0)])
        if not n:
            return out


def _field(number, wire, payload):
    tag = _varint(number << 3 | wire)
    return tag + (_varint(len(payload)) + payload if wire == 2 else payload)


@pytest.mark.parametrize("packed", [False, True])
def test_geometry_metadata_parser(packed):
    rng = np.random.default_rng(0)
    vertices = rng.normal(size=(468, 5)).astype(np.float32)
    floats = vertices.ravel()
    mesh = _field(1, 0, _varint(0)) + _field(2, 0, _varint(0))
    if packed:
        mesh += _field(3, 2, struct.pack(f"<{len(floats)}f", *floats))
    else:
        mesh += b"".join(_field(3, 5, struct.pack("<f", f)) for f in floats)
    mesh += _field(4, 0, _varint(7))
    basis = b"".join(_field(2, 2, _field(1, 0, _varint(i)) + _field(2, 5, struct.pack("<f", .5 + i)))
                     for i in (3, 10, 467))
    canonical, weights = sp.parse_geometry_metadata(_field(1, 2, mesh) + basis + _field(3, 0, _varint(1)))
    np.testing.assert_array_equal(canonical, vertices[:, :3].astype(np.float64))
    assert np.flatnonzero(weights).tolist() == [3, 10, 467] and weights[10] == pytest.approx(10.5)
    with pytest.raises(ValueError):
        sp.parse_geometry_metadata(_field(1, 2, _field(3, 5, struct.pack("<f", 1.))) + basis)


@pytest.mark.skipif(not FACE_TASK.exists(), reason="MediaPipe face_landmarker.task not installed")
def test_the_pinned_face_bundle_holds_the_canonical_face():
    resources = sp.load_face_resources(FACE_TASK)
    assert resources.canonical.shape == (468, 3) and len(resources.basis) == 33
    assert resources.weights.sum() == pytest.approx(1.055, abs=1e-3)
    # the canonical face is metric (cm), about 15 cm wide, y up, its nose tip (vertex 1) in front (z > 0)
    assert 10 < np.ptp(resources.canonical[:, 0]) < 20 and resources.canonical[1, 2] > resources.canonical[:, 2].mean()


def test_pseudo_world_is_x_y_over_aspect_and_scaled_z():
    xyz = np.array([[.5, .3, -.02], [.25, .9, .1]])
    np.testing.assert_allclose(sp.pseudo_world(xyz, 4 / 3, .3), [[.5, .3 * .75, -.006], [.25, .9 * .75, .03]])


@pytest.mark.parametrize("shape", ["16:9", "4:3", "1:1", "9:16"])
def test_aspect_is_recovered_from_the_face_whatever_the_head_turn(shape):
    resources = fake_resources()
    a = sp.STANDARD_ASPECTS[shape]
    size = (int(round(480 * a)), 480)
    noses = np.tile([size[0] / 2, 160.], (12, 1))
    face = face_points(resources, noses, size, frames=12, yaw=-25., pitch=15.)
    estimate = sp.estimate_aspect(face[..., :2], np.ones(12, bool), resources)
    assert estimate["snapped_to"] == shape and estimate["aspect"] == a
    assert estimate["raw"] == pytest.approx(size[0] / size[1], rel=1e-3)


def test_aspect_refuses_too_few_faces_and_unusual_shapes():
    resources = fake_resources()
    face = face_points(resources, np.tile([320., 160.], (12, 1)), (640, 480), frames=12)
    few = sp.estimate_aspect(face[..., :2], np.r_[np.ones(3, bool), np.zeros(9, bool)], resources)
    assert few["aspect"] is None and "face frames" in few["reason"]
    wide = face[..., :2].copy()
    wide[..., 1] *= 480 / (640 / 2.4)                   # the same pixels in a frame of height 640 / 2.4 (2.4:1)
    odd = sp.estimate_aspect(wide, np.ones(12, bool), resources)
    assert odd["aspect"] is None and odd["raw"] == pytest.approx(2.4, rel=1e-3)


def _reference(array, body, hands, size, **flags):
    """The solver on the true 3D points with the adapter's visibility rule and options."""
    xy = array[:, 478:511, :2]
    visible = ((xy >= 0) & (xy <= 1)).all(-1)
    clip, _ = sp.holistic_clip(array, 1000. * len(array) / 30., fake_resources(), image_size=size,
                               blendshape_model=FakeBlendshapes(), pose_z_scale=1., hand_z_scale=1.)
    points = dict(world=body, visible=visible, hands={s: (h, np.ones(len(h), bool)) for s, h in hands.items()})
    return canonical_from_landmarks(clip, sample_id="r", source_id="r", source_sha256="a" * 64, clock_id="r",
                                    inference_method="r", points=points, upright_chest=True, elbow_in_image=True,
                                    upright_root=True)[0]


@pytest.mark.parametrize("size", [(640, 640), (640, 480), (854, 480)])
def test_rotations_equal_the_solver_on_the_true_3d_points(size):
    array, body, hands = semlex_array(size)
    state, report = convert(array, image_size=size)
    reference = _reference(array, body, hands, size)
    for name in ("body", "left_hand", "right_hand"):
        a, r = state.channels[name], reference.channels[name]
        np.testing.assert_array_equal(a.valid, r.valid, err_msg=name)
        np.testing.assert_allclose(a.values, r.values, atol=1e-9, err_msg=name)
    body_valid = dict(zip(state.channels["body"].labels, state.channels["body"].valid.T))
    assert body_valid["Chest"].all() and body_valid["Pelvis"].all()          # upright chest and upright root
    assert not any(body_valid[j].any() for j in ("LeftThigh", "LeftShin", "RightThigh", "RightShin"))
    elbows_in = ((array[:, 478 + 13, :2] >= 0) & (array[:, 478 + 13, :2] <= 1)).all(-1)
    np.testing.assert_array_equal(body_valid["LeftUpperArm"], elbows_in)    # the elbow gate decides the arm
    assert report["hand_assignment"]["Left"] == dict(holistic=T, assigned_same_side=T, assigned_other_side=0, dropped=0)
    assert state.channels["left_hand"].valid.all() and state.channels["right_hand"].valid.all()


def test_square_frames_keep_the_arms_and_short_frames_lose_them():
    tall, *_ = semlex_array((640, 640))
    short, *_ = semlex_array((640, 400))
    arm = lambda s: s.channels["body"].valid[:, s.channels["body"].labels.index("RightForearm")]
    assert arm(convert(tall, image_size=(640, 640))[0]).all()
    assert not arm(convert(short, image_size=(640, 400))[0]).any()           # elbows below the frame


def test_the_depth_gain_scales_only_depth():
    array, body, hands = semlex_array((640, 640))
    gained, _ = convert(array, image_size=(640, 640), pose_z_scale=.3, hand_z_scale=1.5)
    scaled = {"Left": hands["Left"] * [1, 1, 1.5], "Right": hands["Right"] * [1, 1, 1.5]}
    reference = _reference(array, body * [1, 1, .3], scaled, (640, 640))
    for name in ("body", "left_hand", "right_hand"):
        np.testing.assert_allclose(gained.channels[name].values, reference.channels[name].values, atol=1e-9)


def test_estimated_aspect_matches_the_known_frame_size():
    array, *_ = semlex_array((640, 480))
    known, _ = convert(array, image_size=(640, 480))
    estimated, report = convert(array)
    assert report["aspect"]["snapped_to"] == "4:3"
    for name in ("body", "left_hand", "right_hand", "head_rotation"):
        np.testing.assert_allclose(estimated.channels[name].values, known.channels[name].values, atol=1e-9)


def _project_perspective(points_cm, width, height):
    """MediaPipe screen landmarks of camera-space points (y up, z toward the viewer, in front at z < 0): the 63 deg
    camera's normalised image x, y and the relative depth z that the geometry pipeline unprojects exactly."""
    h_near = 2 * sp.NEAR * np.tan(np.radians(sp.FOV_DEGREES) / 2)
    w_near = width * h_near / height
    depth = -points_cm[:, 2]
    x = (sp.NEAR * points_cm[:, 0] / depth + w_near / 2) / w_near
    y = 1 - (sp.NEAR * points_cm[:, 1] / depth + h_near / 2) / h_near
    z = (depth - depth.mean()) * sp.NEAR / (depth.mean() * w_near)
    return np.c_[x, y, z]


@pytest.mark.parametrize("width,height,offset", [(256, 256, (0., 0.)), (256, 200, (6., -4.)), (300, 256, (-8., 3.))])
def test_face_geometry_recovers_a_known_head_rotation(width, height, offset):
    resources = fake_resources()
    truth = rotation((0, 1, 0), 25.) @ rotation((1, 0, 0), -12.) @ rotation((0, 0, 1), 7.)
    camera = resources.canonical @ truth.T + [offset[0], offset[1], -60.]
    matrix = sp.face_transform(_project_perspective(camera, width, height), width, height, resources)
    u, _, vt = np.linalg.svd(matrix[:3, :3])
    np.testing.assert_allclose(u @ vt, truth, atol=1e-9)
    frontal = sp.face_transform(_project_perspective(resources.canonical + [0., 0., -60.], width, height),
                                width, height, resources)
    np.testing.assert_allclose(frontal[:3, :3] / np.linalg.norm(frontal[:3, 0]), np.eye(3), atol=1e-9)


def test_weighted_procrustes_recovers_a_similarity():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(3, 40))
    truth = rotation((1, 2, 3), 40.)
    target = 2.5 * truth @ source + np.array([[1.], [-2.], [3.]])
    m = sp.weighted_procrustes(source, target, rng.uniform(.1, 1., 40))
    np.testing.assert_allclose(m[:3, :3], 2.5 * truth, atol=1e-10)
    np.testing.assert_allclose(m[:3, 3], [1., -2., 3.], atol=1e-10)


@pytest.mark.parametrize("size", [(640, 480), (853, 480), (1280, 720)])
def test_the_rebuilt_face_crop_is_extract_clips_crop(size):
    rng = np.random.default_rng(1)
    image = np.zeros((size[1], size[0], 3), np.uint8)
    for _ in range(20):
        nose, shoulders = rng.uniform(0, 1, 2), rng.uniform(0, 1, (2, 2))
        pixel = np.array(size, dtype=np.float64)
        crop, offset = square_crop(image, nose * pixel, .9 * np.linalg.norm((shoulders[0] - shoulders[1]) * pixel))
        mine = sp.face_crop(nose, shoulders, size)
        if crop is None:
            assert mine is None
        else:
            assert mine == (int(offset[0]), int(offset[1]), crop.shape[1], crop.shape[0])


def test_missing_parts_are_invalid_never_filled():
    array, *_ = semlex_array((640, 640))
    array[2, 511:532] = np.nan                     # left hand not detected
    array[3, 532:553] = 0.                         # an all-zero right hand block
    array[4, :478] = np.nan                        # no face
    array[5] = np.nan                              # nothing at all
    state, report = convert(array, image_size=(640, 640))
    assert not state.channels["left_hand"].valid[2].any() and not state.channels["right_hand"].valid[3].any()
    assert not state.channels["head_rotation"].valid[4].any() and not state.channels["face"].valid[4].any()
    assert not any(c.valid[5].any() for c in state.channels.values())
    assert report["pose_frames"] == T - 1 and report["face_frames"] == T - 2
    for name in ("root_translation", "head_translation", "left_gaze", "right_gaze", "contact"):
        assert not state.channels[name].valid.any(), name
    assert state.channels["face"].valid[0].all() and state.channels["blink"].valid[0].all()


def test_holistic_hands_on_the_wrong_wrist_follow_anatomy():
    array, *_ = semlex_array((640, 640))
    swapped = array.copy()
    swapped[:, 511:532], swapped[:, 532:553] = array[:, 532:553], array[:, 511:532]
    a, _ = convert(array, image_size=(640, 640))
    b, report = convert(swapped, image_size=(640, 640))
    for name in ("left_hand", "right_hand"):
        np.testing.assert_allclose(a.channels[name].values, b.channels[name].values, atol=1e-12)
    assert report["hand_assignment"]["Left"] == dict(holistic=T, assigned_same_side=0, assigned_other_side=T, dropped=0)


@pytest.mark.parametrize("mutate,reason", [
    (lambda a: a[:, :552], "shape"),
    (lambda a: a[:1], "fewer than 2"),
    (lambda a: np.where(np.arange(553)[None, :, None] == 3, np.inf, a), "infinite"),
    (lambda a: a.astype(np.int16), "floating"),
])
def test_malformed_arrays_are_refused(mutate, reason):
    array, *_ = semlex_array((640, 480))
    with pytest.raises(sp.Refusal, match=reason):
        convert(mutate(array), image_size=(640, 480))


def test_implausible_rates_and_faceless_clips_are_refused():
    array, *_ = semlex_array((640, 480))
    with pytest.raises(sp.Refusal, match="fps"):
        sp.convert_array(array, 10., fake_resources(), video_id="v", source_sha256="a" * 64,
                         blendshape_model=FakeBlendshapes())
    faceless = array.copy()
    faceless[:, :478] = np.nan
    with pytest.raises(sp.Refusal, match="face frames"):
        convert(faceless)
    state, _ = convert(faceless, aspect=4 / 3)            # with a known aspect the body survives; the face is unavailable
    assert not state.channels["face"].valid.any() and not state.channels["head_rotation"].valid.any()
    assert state.channels["right_hand"].valid.any()


def test_clock_is_uniform_over_the_duration_and_the_state_round_trips(tmp_path):
    array, *_ = semlex_array((640, 480))
    state, report = sp.convert_array(array.astype(np.float16), 400., fake_resources(), video_id="v",
                                     source_sha256="a" * 64, blendshape_model=FakeBlendshapes())
    np.testing.assert_allclose(state.timestamps, np.arange(T) * .4 / T)
    assert report["fps"] == pytest.approx(T / .4)
    assert state.channels["body"].convention == "mediapipe-tasks-swing-ik-y-up-v1"
    assert state.channels["head_rotation"].coordinate_frame == "mediapipe-face-geometry-roi-camera-v1"
    assert "semlex-holistic-pseudo-world-v1" in state.channels["body"].inference_method
    digest = state.save(tmp_path / "s.npz")
    loaded = MultichannelMotion.load(tmp_path / "s.npz", expected_sha256=digest)
    for name, channel in state.channels.items():
        np.testing.assert_array_equal(loaded.channels[name].values, channel.values)


def test_pose_files_load_without_pickle():
    import io
    array = np.zeros((3, 553, 3), np.float16)
    buffer = io.BytesIO()
    np.save(buffer, array)
    assert sp.load_pose_file(buffer.getvalue()).shape == (3, 553, 3)
    buffer = io.BytesIO()
    np.save(buffer, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(sp.Refusal):
        sp.load_pose_file(buffer.getvalue())


@pytest.mark.skipif(not FACE_TASK.exists(), reason="MediaPipe face_landmarker.task not installed")
def test_the_blendshape_model_runs_and_is_similarity_invariant():
    pytest.importorskip("cv2")
    resources = sp.load_face_resources(FACE_TASK)
    model = sp.BlendshapeModel(resources)
    face = face_points(resources, np.tile([320., 200.], (2, 1)), (640, 480), frames=2)[..., :2] * [640, 480]
    scores = model(face)
    assert scores.shape == (2, 52) and (scores >= 0).all() and (scores <= 1).all()
    np.testing.assert_allclose(model(face * 1.7 + [40., -25.]), scores, atol=1e-4)
