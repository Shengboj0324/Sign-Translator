"""Canonical forward kinematics: rest pose layout and rotation of a whole arm."""
from dataclasses import replace

import numpy as np
import torch

from signtranslator.data_engineering.video_landmarks import BODY_LABELS
from signtranslator.generation.canonical_skeleton import skeleton_positions
from signtranslator.generation.skeleton_preview import render_preview_frames
from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d
from signtranslator.data_engineering.video_landmarks import finger_joints

IDENTITY = np.array([1., 0., 0., 0., 1., 0.])


def rest_state(frames=3):
    channels = {}
    labels = {"body": BODY_LABELS, "left_hand": tuple(j.name for j in finger_joints("Left")),
              "right_hand": tuple(j.name for j in finger_joints("Right"))}
    for name, (kind, width, units) in CHANNELS.items():
        names = labels.get(name, (name,))
        values = np.tile(IDENTITY, (frames, len(names), 1)) if kind == "rotation_6d_columns" else np.zeros((frames, len(names), width))
        valid = np.ones((frames, len(names)), bool) if name in labels else np.zeros((frames, len(names)), bool)
        channels[name] = MotionChannel(np.where(valid[..., None], values, 0.), valid, np.zeros_like(valid),
                                       valid.astype(float), names, "f", units, "src", "a" * 64, "c",
                                       inference_method="estimate" if valid.any() else None)
    return MultichannelMotion(np.arange(frames) / 24., "clock", "s", channels)


def test_rest_pose_puts_arms_out_sideways_and_fingers_up():
    joints, positions, bones = skeleton_positions(rest_state())
    at = {j: positions[0, i] for i, j in enumerate(joints)}
    np.testing.assert_allclose(at["LeftForeArm"] - at["LeftShoulder"], [.28, 0, 0], atol=1e-12)
    np.testing.assert_allclose(at["RightHand"] - at["RightForeArm"], [-.25, 0, 0], atol=1e-12)
    np.testing.assert_allclose(at["LeftHandIndex4"] - at["LeftHandIndex1"], [0, .045 + .028 + .022, 0], atol=1e-12)
    assert len(bones) == len(set(bones)) and all(a in joints and b in joints for a, b in bones)


def test_rotating_the_chest_rotates_the_whole_upper_body_rigidly():
    state = rest_state()
    body = state.channels["body"]
    values = body.values.copy()
    values[:, BODY_LABELS.index("Chest")] = matrix_to_rotation_6d(
        axis_angle_to_matrix(torch.tensor([0., 0., np.pi / 2], dtype=torch.float64))).numpy()
    state.channels["body"] = replace(body, values=values)
    joints, positions, _ = skeleton_positions(state)
    at = {j: positions[0, i] for i, j in enumerate(joints)}
    np.testing.assert_allclose(at["LeftForeArm"] - at["LeftShoulder"], [0, .28, 0], atol=1e-12)
    rest_joints, rest_positions, rest_bones = skeleton_positions(rest_state())
    frames = render_preview_frames(rest_positions, rest_joints, rest_bones, panel=64)
    assert len(frames) == 3 and frames[0].shape == (128, 128, 3)


def _hand_landmark_state():
    """Both hands from mirror-image 21-point landmarks through the project's own adapter."""
    from signtranslator.data_engineering.video_landmarks import ClipLandmarks, canonical_from_landmarks
    world = np.zeros((1, 33, 3))
    for index, point in ((11, (.18, .45, 0.)), (12, (-.18, .45, 0.)), (13, (.3, .25, .05)), (14, (-.3, .25, .05)),
                         (15, (.3, .3, .3)), (16, (-.3, .3, .3)), (23, (.1, 0., 0.)), (24, (-.1, 0., 0.))):
        world[0, index] = point
    visible = np.zeros((1, 33), bool)
    visible[0, [11, 12, 13, 14, 15, 16, 23, 24]] = True
    left = np.zeros((21, 3))
    left[0] = (.3, .3, .3)
    for finger, (dx, start) in enumerate(((.035, 1), (.02, 5), (0., 9), (-.02, 13), (-.04, 17))):
        for k in range(4):
            left[start + k] = left[0] + (dx * (1.3 if finger == 0 else 1.), .02 + .07 * (finger > 0) + .025 * k, .01 * k)
    right = left * [-1, 1, 1]                                    # the mirror image across the body midline
    hands = {"Left": (left[None], np.ones(1, bool)), "Right": (right[None], np.ones(1, bool))}
    clip = ClipLandmarks(times=np.zeros(1), pose_world=np.zeros((1, 33, 3)), pose_visibility=np.ones((1, 33)),
                         pose_image=np.zeros((1, 33, 2)), pose_found=np.ones(1, bool),
                         hands_world=np.zeros((1, 0, 21, 3)), hands_image=np.zeros((1, 0, 21, 2)),
                         hands_found=np.zeros((1, 0), bool), blendshapes=np.zeros((1, 0)), blendshape_names=(),
                         face_matrix=np.eye(4)[None], face_found=np.zeros(1, bool))
    state, _ = canonical_from_landmarks(clip, sample_id="s", source_id="src", source_sha256="a" * 64, clock_id="c",
                                        inference_method="synthetic landmarks",
                                        points=dict(world=world, visible=visible, hands=hands))
    return state, {"Left": left, "Right": right}


def test_knuckle_bases_lie_on_the_landmark_side_for_both_hands():
    state, landmarks = _hand_landmark_state()
    joints, positions, _ = skeleton_positions(state)
    at = {j: positions[0, i] for i, j in enumerate(joints)}
    for side, points in landmarks.items():
        index_side = points[5] - points[17]                       # landmark index MCP minus pinky MCP
        assert (at[f"{side}HandIndex1"] - at[f"{side}HandPinky1"]) @ index_side > 0, side
        assert (at[f"{side}HandThumb1"] - at[f"{side}HandPinky1"]) @ index_side > 0, side
        # the two hands are mirror images of each other, knuckles included
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        np.testing.assert_allclose(at[f"RightHand{finger}1"] - at["RightHand"],
                                   (at[f"LeftHand{finger}1"] - at["LeftHand"]) * [-1, 1, 1], atol=1e-9)
