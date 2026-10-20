"""Gaze model: exact geometry, recovery of a known blendshape-to-angle map, JSON round trip, crop rays."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from signtranslator.data_engineering.gaze import (
    FEATURES, SIDES, GazeModel, angle_degrees, design, direction, eye_features, fit_eye,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
NAMES = tuple(f + s for s in ("Left", "Right") for f in FEATURES) + ("jawOpen",)


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, .6, size=(n, len(NAMES)))
    head = Rotation.from_rotvec(rng.normal(scale=.25, size=(n, 3))).as_matrix()
    truth = {"Left": (np.array([.02, -.9, .8, 0, 0, 0, 0, 0]), np.array([-.05, 0, 0, .7, -.6, .1, 0, 0])),
             "Right": (np.array([-.02, .85, -.75, 0, 0, 0, 0, 0]), np.array([-.04, 0, 0, .65, -.55, 0, .1, 0]))}
    gaze = {}
    for side in SIDES:
        x = np.c_[np.ones(n), eye_features(NAMES, scores, side)]
        u = direction(x @ truth[side][0], x @ truth[side][1])
        gaze[side] = np.einsum("tij,tj->ti", head, u)
    return scores, head, gaze, truth


def test_direction_is_unit_and_forward_at_zero():
    np.testing.assert_allclose(direction(np.zeros(1), np.zeros(1)), [[0, 0, 1]])
    d = direction(np.linspace(-1, 1, 9), np.linspace(-.5, .5, 9))
    np.testing.assert_allclose(np.linalg.norm(d, axis=-1), 1)
    assert angle_degrees(np.array([1., 0, 0]), np.array([0., 0, 1])) == pytest.approx(90)


def test_fit_recovers_a_known_map_and_predicts_in_the_camera_frame():
    scores, head, gaze, truth = synthetic()
    model = GazeModel({"Left": "Left", "Right": "Right"},
                      *[{s: fit_eye(design(NAMES, scores, head, s, s, "monocular"), head, gaze[s], ridge=1e-9)[k]
                         for s in SIDES} for k in (0, 1)], {})
    for side in SIDES:
        np.testing.assert_allclose(model.yaw[side], truth[side][0], atol=1e-4)
        np.testing.assert_allclose(model.pitch[side], truth[side][1], atol=1e-4)
    predicted = model.predict(NAMES, scores, head)
    cyclopean = (gaze["Left"] + gaze["Right"]) / np.linalg.norm(gaze["Left"] + gaze["Right"], axis=-1, keepdims=True)
    for side in SIDES:                                   # both channels carry the eyes' mean direction
        assert angle_degrees(predicted[side], cyclopean).max() < 1e-2


def test_fit_is_robust_to_gross_failures():
    scores, head, gaze, truth = synthetic(seed=1)
    corrupted = gaze["Left"].copy()
    corrupted[::20] = -corrupted[::20]                           # 5% looking backwards: detector failures
    yaw, pitch = fit_eye(design(NAMES, scores, head, "Left", "Right", "monocular"), head, corrupted)
    np.testing.assert_allclose(yaw, truth["Left"][0], atol=.03)
    np.testing.assert_allclose(pitch, truth["Left"][1], atol=.03)


def test_json_round_trip_and_feature_guard():
    scores, head, gaze, _ = synthetic(n=400)
    model = GazeModel({"Left": "Right", "Right": "Left"}, {s: np.arange(17.) / 40 for s in SIDES},
                      {s: -np.arange(17.) / 80 for s in SIDES}, {"note": 1}, "binocular+head",
                      ((-.2, .2), (-.1, .3)))
    again = GazeModel.from_json(model.to_json())
    for side in SIDES:
        np.testing.assert_allclose(again.predict(NAMES, scores, head)[side], model.predict(NAMES, scores, head)[side])
    assert again.suffix == {"Left": "Right", "Right": "Left"} and again.variant == "binocular+head"
    np.testing.assert_allclose(again.head_range, ((-.2, .2), (-.1, .3)))
    # the head term saturates outside the calibrated range
    turned = Rotation.from_euler("y", np.array([[40.], [60.]]), degrees=True).as_matrix()
    z = again.design("Left", NAMES, scores[:2], turned)
    np.testing.assert_allclose(z[:, -2], .2)
    import json
    bad = json.loads(model.to_json()); bad["features"] = ["x"]
    with pytest.raises(ValueError):
        GazeModel.from_json(json.dumps(bad))


def test_crop_ray_rotation_takes_the_axis_to_the_crop_centre():
    from fit_gaze_model import ray_rotation
    ray = np.array([.2, -.1, 1.])
    R = ray_rotation(ray)
    np.testing.assert_allclose(R @ [0, 0, 1], ray / np.linalg.norm(ray), atol=1e-12)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1)
    np.testing.assert_allclose(ray_rotation(np.array([0, 0, 2.])), np.eye(3))


def test_two_view_state_gains_gaze_from_the_monocular_face_and_head():
    from dataclasses import replace
    import tests.test_de_two_view_states as tv
    from tests.test_de_video_landmarks import clip as mp_clip, convert
    from signtranslator.data_engineering.two_view_states import triangulated_state
    front, side, _ = tv.clips()
    times, fp, fc, sp, sc = tv.aligned(front, side, 2)
    calibration = tv.calibrate_session([(fp, fc, sp, sc)])
    points, usable = tv.triangulated_points(calibration, fp, fc, sp, sc)
    inputs = tv.rotation_inputs(points, usable)
    rotations, hands = tv.solve_rotations(inputs["world"], inputs["visible"], inputs["hands"])
    base = mp_clip()
    n = len(front.times)
    tile = lambda a: np.repeat(a[:1], n, axis=0)
    blend = tile(base.blendshapes).copy()
    eye = [f + side for side in SIDES for f in FEATURES if f != "eyeBlink"]
    names = tuple(base.blendshape_names[:3]) + tuple(eye) + tuple(base.blendshape_names[3 + len(eye):])
    base = replace(base, blendshape_names=names)
    blend[5, names.index("eyeBlinkLeft")] = .9                     # left eye closed on frame 5
    monocular, _ = convert(replace(base, times=np.arange(n) / 30., pose_world=tile(base.pose_world),
                                   pose_visibility=tile(base.pose_visibility), pose_image=tile(base.pose_image),
                                   pose_found=tile(base.pose_found), hands_world=tile(base.hands_world),
                                   hands_image=tile(base.hands_image), hands_found=tile(base.hands_found),
                                   blendshapes=blend, face_matrix=tile(base.face_matrix), face_found=tile(base.face_found)))
    model = GazeModel({"Left": "Left", "Right": "Right"}, {s: np.zeros(8) for s in SIDES},
                      {s: np.zeros(8) for s in SIDES}, {})                 # eyes straight ahead in the head
    state = triangulated_state(monocular, slice(0, len(times)), rotations, hands, source_id="tv", source_sha256="c" * 64,
                               inference_method="two-view", gaze_model=model, gaze_method="test gaze")
    state.validate()
    head = monocular.channels["head_rotation"]
    from signtranslator.pose.rotations import rotation_6d_to_matrix
    import torch
    forward = rotation_6d_to_matrix(torch.from_numpy(head.values[0, 0])).numpy()[:, 2]
    for name in ("left_gaze", "right_gaze"):
        channel = state.channels[name]
        assert channel.coordinate_frame == head.coordinate_frame and channel.inference_method == "test gaze"
        np.testing.assert_allclose(channel.values[channel.valid[:, 0], 0], np.broadcast_to(forward, (channel.valid.sum(), 3)),
                                   atol=1e-9)
    assert not state.channels["left_gaze"].valid[5, 0] and state.channels["right_gaze"].valid[5, 0]
    assert np.all(state.channels["left_gaze"].values[5] == 0)
