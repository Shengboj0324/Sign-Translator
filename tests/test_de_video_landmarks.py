"""Video landmark adapter: exact kinematics, anatomical hand assignment, declared gaps."""
from dataclasses import replace

import numpy as np
import pytest
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


def test_upright_chest_keeps_the_arms_when_the_hips_are_out_of_frame():
    from signtranslator.data_engineering.landmark_kinematics import frame_from_axes
    from signtranslator.data_engineering.video_landmarks import assign_hands, solve_rotations
    c = clip()
    visibility = c.pose_visibility.copy()
    visibility[:, 23:] = 0.                                     # webcam framing: hips and legs unseen
    c = replace(c, pose_visibility=visibility)
    default, _ = convert(c)
    assert not default.channels["body"].valid.any()             # the hip-based chest drops every arm joint
    state, report = canonical_from_landmarks(c, sample_id="s", source_id="src", source_sha256="a" * 64,
                                             clock_id="c", inference_method="test", upright_chest=True)
    body = state.channels["body"]
    valid = dict(zip(body.labels, body.valid.T))
    for name in ("LeftUpperArm", "LeftForearm", "RightUpperArm", "RightForearm", "LeftHand", "RightHand"):
        assert valid[name].all(), name
    for name in ("Pelvis", "Chest", "LeftThigh", "LeftShin", "RightThigh", "RightShin"):
        assert not valid[name].any(), name
    assert report["upright_chest"] and report["arm_frames"] == {"LeftForearm": T, "RightForearm": T}
    world = c.pose_world @ Y_UP.T
    hands = {side: (points @ Y_UP.T, found) for side, (points, found) in assign_hands(c).items()}
    rotations, _ = solve_rotations(world, visibility >= .5, hands, upright_chest=True)
    chest, _ = frame_from_axes(world[:, 11] - world[:, 12], np.broadcast_to([0., 1., 0.], (T, 3)))
    frames = {"Chest": chest}
    for joint in BODY_JOINTS[:4]:                               # the arm bones are reproduced exactly
        frames[joint.name] = frames[joint.parent] @ rotations[joint.name][0]
        bone, _ = unit(world[:, joint.end] - world[:, joint.start])
        rest, _ = unit(np.asarray(joint.rest))
        np.testing.assert_allclose(np.einsum("tij,j->ti", frames[joint.name], rest), bone, atol=1e-6)


def _solve(c, **options):
    from signtranslator.data_engineering.video_landmarks import solve_rotations
    world = c.pose_world @ Y_UP.T
    hands = {side: (points @ Y_UP.T, found) for side, (points, found) in assign_hands(c).items()}
    visible = (c.pose_visibility >= .5) & c.pose_found[:, None]
    return solve_rotations(world, visible, hands, pose_image=c.pose_image, **options)


def _with_elbows(c, left, right):
    image = c.pose_image.copy()
    image[:, 13], image[:, 14] = left, right
    return replace(c, pose_image=image)


ARM_GATED = ("UpperArm", "Forearm", "Hand")


def test_elbow_in_image_gates_exactly_the_upper_arm_forearm_and_wrist_of_that_side():
    from signtranslator.data_engineering.video_landmarks import elbows_in_image, solve_rotations
    left = np.array([[.5, .5], [1., 0.], [1.2, .5], [.5, 1.01], [.5, .5]])     # bounds included; out on frames 2, 3
    right = np.array([[.4, -.1], [.4, .7], [.4, .7], [np.nan, .7], [0., 1.]])   # out on frames 0 and 3 (non-finite)
    c = _with_elbows(clip(), left, right)
    inside = elbows_in_image(c.pose_image)
    np.testing.assert_array_equal(inside["Left"], [True, True, False, False, True])
    np.testing.assert_array_equal(inside["Right"], [False, True, True, False, True])
    base, base_hands = _solve(c)
    gated, gated_hands = _solve(c, elbow_in_image=True)
    assert all(base[n][1].all() for n in base)                                 # every joint valid without the gate
    for name in base:
        side = "Left" if name.startswith("Left") else "Right" if name.startswith("Right") else None
        if side and name[len(side):] in ARM_GATED:
            np.testing.assert_array_equal(gated[name][1], inside[side], err_msg=name)
            ok = inside[side]                                                  # where valid, the values are the same
            np.testing.assert_array_equal(gated[name][0][ok], base[name][0][ok], err_msg=name)
        else:                                                                  # pelvis, chest, legs: untouched
            np.testing.assert_array_equal(gated[name][1], base[name][1], err_msg=name)
            np.testing.assert_array_equal(gated[name][0], base[name][0], err_msg=name)
    for side in ("Left", "Right"):                                             # fingers are rooted at the palm
        for name in base_hands[side]:
            np.testing.assert_array_equal(gated_hands[side][name][0], base_hands[side][name][0])
            np.testing.assert_array_equal(gated_hands[side][name][1], base_hands[side][name][1])
    state, report = canonical_from_landmarks(c, sample_id="s", source_id="src", source_sha256="a" * 64, clock_id="c",
                                             inference_method="test", elbow_in_image=True)
    assert report["elbow_in_image"] and report["elbow_in_image_frames"] == {"Left": 3, "Right": 3}
    assert report["arm_frames"] == {"LeftForearm": 3, "RightForearm": 3}
    default_state, _ = convert(c)
    for name in ("left_hand", "right_hand"):
        np.testing.assert_array_equal(state.channels[name].valid, default_state.channels[name].valid)
        np.testing.assert_array_equal(state.channels[name].values, default_state.channels[name].values)
    with pytest.raises(ValueError):
        solve_rotations(c.pose_world, c.pose_visibility >= .5, {}, elbow_in_image=True)


def test_the_elbow_gate_is_off_by_default_and_bit_identical():
    c = _with_elbows(clip(), [5., 5.], [-3., 2.])                              # both elbows far outside the image
    in_frame = _with_elbows(clip(), [.5, .5], [.5, .5])
    default, default_hands = _solve(in_frame)
    off, off_hands = _solve(c, elbow_in_image=False)                           # the elbow image points are ignored
    inside, inside_hands = _solve(in_frame, elbow_in_image=True)
    for other, other_hands in ((off, off_hands), (inside, inside_hands)):      # all elbows inside: no change at all
        assert other.keys() == default.keys()
        for name in default:
            np.testing.assert_array_equal(other[name][0], default[name][0])
            np.testing.assert_array_equal(other[name][1], default[name][1])
        for side in default_hands:
            for name in default_hands[side]:
                np.testing.assert_array_equal(other_hands[side][name][0], default_hands[side][name][0])
    _, report = convert(c)
    assert "elbow_in_image" not in report and "elbow_in_image_frames" not in report


def test_upright_chest_with_the_elbow_gate_keeps_in_frame_arms_without_hips():
    c = _with_elbows(clip(), [[.6, .5]] * 3 + [[.6, 1.3]] * 2, [[.4, .5]] * 5)  # left elbow below the frame at the end
    visibility = c.pose_visibility.copy()
    visibility[:, 23:] = 0.                                                    # chest-up framing: no hips, no legs
    c = replace(c, pose_visibility=visibility)
    state, report = canonical_from_landmarks(c, sample_id="s", source_id="src", source_sha256="a" * 64, clock_id="c",
                                             inference_method="test", upright_chest=True, elbow_in_image=True)
    valid = dict(zip(state.channels["body"].labels, state.channels["body"].valid.T))
    for part in ARM_GATED:
        np.testing.assert_array_equal(valid["Left" + part], [True, True, True, False, False])
        assert valid["Right" + part].all()
    for name in ("Pelvis", "Chest", "LeftThigh", "LeftShin", "RightThigh", "RightShin"):
        assert not valid[name].any(), name
    assert report["arm_frames"] == {"LeftForearm": 3, "RightForearm": 5}
    assert report["upright_chest"] and report["elbow_in_image_frames"] == {"Left": 3, "Right": 5}


def test_raw_landmarks_round_trip_and_re_solve():
    import hashlib
    from signtranslator.data_engineering.video_landmarks import load_raw_landmarks, save_raw_landmarks
    c = replace(clip(), other_person=np.linspace(0., .9, T), roi=True)
    raw = save_raw_landmarks(c)
    back = load_raw_landmarks(raw, expected_sha256=hashlib.sha256(raw).hexdigest())
    np.testing.assert_array_equal(back.times, c.times)                         # the clock is exact
    for name, atol in (("pose_world", 1e-3), ("pose_visibility", 1e-3), ("hands_world", 1e-4), ("blendshapes", 1e-3),
                       ("other_person", 1e-3), ("pose_image", 1e-6), ("hands_image", 1e-6), ("face_matrix", 1e-5)):
        np.testing.assert_allclose(getattr(back, name), getattr(c, name), atol=atol, err_msg=name)
        assert getattr(back, name).dtype == np.float64
    for name in ("pose_found", "hands_found", "face_found"):
        np.testing.assert_array_equal(getattr(back, name), getattr(c, name))
    assert back.blendshape_names == NAMES and back.roi
    again = load_raw_landmarks(save_raw_landmarks(back))                      # a second round trip changes nothing
    for name in ("pose_world", "pose_image", "hands_world", "blendshapes", "face_matrix"):
        np.testing.assert_array_equal(getattr(again, name), getattr(back, name))
    a, _ = canonical_from_landmarks(back, sample_id="s", source_id="src", source_sha256="a" * 64, clock_id="c",
                                    inference_method="test", upright_chest=True, elbow_in_image=True)
    b, _ = canonical_from_landmarks(again, sample_id="s", source_id="src", source_sha256="a" * 64, clock_id="c",
                                    inference_method="test", upright_chest=True, elbow_in_image=True)
    for name in a.channels:                                                    # re-solving reproduces the state
        np.testing.assert_array_equal(a.channels[name].values, b.channels[name].values)
        np.testing.assert_array_equal(a.channels[name].valid, b.channels[name].valid)
    assert load_raw_landmarks(save_raw_landmarks(replace(c, other_person=None))).other_person is None
    with pytest.raises(ValueError):
        load_raw_landmarks(raw, expected_sha256="0" * 64)
