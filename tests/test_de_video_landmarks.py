"""Video landmark adapter: exact kinematics, anatomical hand assignment, declared gaps."""
from dataclasses import replace

import numpy as np
import torch

from signtranslator.data_engineering.landmark_kinematics import unit
from signtranslator.data_engineering.video_landmarks import (
    BODY_JOINTS, Y_UP, ClipLandmarks, assign_hands, canonical_from_landmarks, finger_joints,
)
from signtranslator.pose.rotations import rotation_6d_to_matrix

T = 5
NAMES = ("_neutral", "eyeBlinkLeft", "eyeBlinkRight", *(f"bs{i}" for i in range(49)))


def clip(*, swap_hands=False, face=True):
    rng = np.random.default_rng(3)
    pose_y_up = rng.normal(scale=.05, size=(T, 33, 3))
    anchors = {11: [.2, .5, 0], 12: [-.2, .5, 0], 13: [.45, .4, .05], 14: [-.45, .35, .1], 15: [.6, .25, .2],
               16: [-.55, .2, .25], 23: [.1, 0, 0], 24: [-.1, 0, 0], 25: [.12, -.45, .02], 26: [-.12, -.45, 0],
               27: [.12, -.9, 0], 28: [-.12, -.9, .01]}
    for k, v in anchors.items():
        pose_y_up[:, k] = np.asarray(v) + .01 * np.arange(T)[:, None] * rng.normal(size=3)
    pose_world = pose_y_up @ Y_UP  # back to MediaPipe axes (Y_UP is its own inverse)
    image = np.zeros((T, 33, 2))
    image[:, 15] = [.7, .6]  # subject's left wrist appears on the image right
    image[:, 16] = [.3, .6]
    hands_world = rng.normal(scale=.03, size=(T, 2, 21, 3))
    hands_world[:, :, 9] += [0., -.08, 0.]  # middle MCP away from wrist along the palm
    hands_image = np.zeros((T, 2, 21, 2))
    hands_image[:, 0, 0] = [.31, .61]  # detection 0 is near the right wrist
    hands_image[:, 1, 0] = [.69, .59]  # detection 1 is near the left wrist
    if swap_hands:
        hands_world, hands_image = hands_world[:, ::-1].copy(), hands_image[:, ::-1].copy()
    blend = np.tile(np.linspace(0, 1, 52), (T, 1))
    matrix = np.tile(np.eye(4), (T, 1, 1))
    matrix[:, :3, 3] = [1., 2., -50.]
    return ClipLandmarks(np.arange(T) / 30., pose_world, np.ones((T, 33)), image, np.ones(T, bool), hands_world,
                         hands_image, np.ones((T, 2), bool), blend, NAMES, matrix, np.full(T, face))


def convert(c):
    return canonical_from_landmarks(c, sample_id="s", source_id="src", source_sha256="a" * 64, clock_id="c",
                                    inference_method="test")


def test_body_rotations_reproduce_every_observed_bone_direction():
    c = clip()
    state, report = convert(c)
    body = state.channels["body"]
    mats = rotation_6d_to_matrix(torch.from_numpy(body.values)).numpy()
    world = c.pose_world @ Y_UP.T
    pelvis = mats[:, body.labels.index("Pelvis")]
    chest = pelvis @ mats[:, body.labels.index("Chest")]
    frames = {"Pelvis": pelvis, "Chest": chest}
    for joint in BODY_JOINTS:
        frames[joint.name] = frames[joint.parent] @ mats[:, body.labels.index(joint.name)]
        bone, _ = unit(world[:, joint.end] - world[:, joint.start])
        rest, _ = unit(np.asarray(joint.rest))
        np.testing.assert_allclose(np.einsum("tij,j->ti", frames[joint.name], rest), bone, atol=1e-6)
    assert body.valid.all() and not body.observed.any()


def test_hands_are_assigned_by_anatomy_not_detection_order():
    a, b = assign_hands(clip()), assign_hands(clip(swap_hands=True))
    for side in ("Left", "Right"):
        assert a[side][1].all()
        np.testing.assert_array_equal(a[side][0], b[side][0])
    np.testing.assert_array_equal(a["Left"][0], clip().hands_world[:, 1])
    far = clip()
    far = replace(far, hands_image=far.hands_image + .5)  # beyond the declared radius
    assert not assign_hands(far)["Left"][1].any()


def test_face_head_and_declared_gaps():
    state, _ = convert(clip())
    assert state.channels["face"].labels == NAMES and state.channels["face"].values.shape == (T, 52, 1)
    np.testing.assert_allclose(state.channels["blink"].values[0, :, 0], np.linspace(0, 1, 52)[[1, 2]])
    np.testing.assert_allclose(state.channels["head_translation"].values[0, 0], [.01, .02, -.5])
    for name in ("root_translation", "left_gaze", "right_gaze", "contact"):
        assert not state.channels[name].valid.any()
    missing, _ = convert(clip(face=False))
    for name in ("face", "blink", "head_rotation", "head_translation"):
        assert not missing.channels[name].valid.any()
    assert finger_joints("Left")[0].name == "LeftHandThumb1" and len(finger_joints("Right")) == 15
    assert np.isclose(np.linalg.det(Y_UP), 1.)
