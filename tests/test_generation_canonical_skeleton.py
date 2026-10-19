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
