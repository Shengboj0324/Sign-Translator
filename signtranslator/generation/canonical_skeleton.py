"""Forward kinematics of canonical states, for previews of real and generated motion.

Canonical body and hand channels hold swing-IK local rotations
(``video_landmarks``): a joint's world frame is its parent's world frame times its
local rotation, and its bone points along ``world(parent) · R_j · rest_j``. Metric
scale is not part of the canonical state, so nominal bone lengths are used — the
result is for inspection (does the motion look like signing?), not measurement.
Joint names follow the preview renderer's convention (``Spine2``, ``LeftShoulder``,
``LeftHand``, ``LeftHandIndex1`` …).
"""
from __future__ import annotations

import numpy as np

from ..data_engineering.video_landmarks import BODY_JOINTS, finger_joints
from ..pose.multichannel import MultichannelMotion
from .motion_metrics import _matrices

_LENGTH = {"UpperArm": .28, "Forearm": .25}
_FINGER_LENGTHS = (.045, .028, .022)
_KNUCKLES = {"Thumb": (.025, .02), "Index": (.02, .085), "Middle": (0., .09), "Ring": (-.02, .085),
             "Pinky": (-.04, .075)}
_CHEST_HEIGHT, _SHOULDER_HALF, _NECK = .45, .18, .12


def _rotations(state: MultichannelMotion, name: str) -> dict[str, np.ndarray]:
    channel = state.channels[name]
    return {label: _matrices(channel.values[:, k]) for k, label in enumerate(channel.labels)}


def skeleton_positions(state: MultichannelMotion):
    """(joint names, (T, J, 3) positions, bones) of the upper body and both hands."""
    body = _rotations(state, "body")
    frames = len(state.timestamps)
    pelvis_world = body["Pelvis"]
    chest_world = pelvis_world @ body["Chest"]
    hips = np.zeros((frames, 3))
    spine = hips + (chest_world @ np.array([0., _CHEST_HEIGHT, 0.]))
    positions = {"Hips": hips, "Spine2": spine, "Head": spine + chest_world @ np.array([0., _NECK + .1, 0.])}
    world = {"Chest": chest_world}
    bones = [("Hips", "Spine2"), ("Spine2", "Head")]
    for side, sign in (("Left", 1.), ("Right", -1.)):
        shoulder = spine + chest_world @ np.array([sign * _SHOULDER_HALF, 0., 0.])
        positions[f"{side}Shoulder"] = shoulder
        bones.append(("Spine2", f"{side}Shoulder"))
        start, previous = shoulder, f"{side}Shoulder"
        for joint, part, point in ((f"{side}UpperArm", "UpperArm", f"{side}ForeArm"),
                                   (f"{side}Forearm", "Forearm", f"{side}Hand")):
            spec = next(j for j in BODY_JOINTS if j.name == joint)
            frame = world[spec.parent] @ body[joint]
            world[joint] = frame
            end = start + _LENGTH[part] * (frame @ np.asarray(spec.rest))
            positions[point] = end
            bones.append((previous, point))
            start, previous = end, point
        hand_world = world[f"{side}Forearm"] @ body[f"{side}Hand"]
        fingers = _rotations(state, f"{side.lower()}_hand")
        wrist = positions[f"{side}Hand"]
        for spec in finger_joints(side):
            finger = spec.name[len(f"{side}Hand"):-1]
            segment = int(spec.name[-1])
            parent_world = hand_world if spec.parent == "Hand" else world[spec.parent]
            frame = parent_world @ fingers[spec.name]
            world[spec.name] = frame
            if segment == 1:
                x, y = _KNUCKLES[finger]
                # The hand frame's x runs pinky -> index MCP on both hands (video_landmarks builds it
                # from landmarks 5 - 17), so knuckle offsets are not mirrored by side.
                base = wrist + hand_world @ np.array([x, y, 0.])
                positions[f"{side}Hand{finger}1"] = base
                bones.append((f"{side}Hand", f"{side}Hand{finger}1"))
            tip = positions[f"{side}Hand{finger}{segment}"] + _FINGER_LENGTHS[segment - 1] * (frame @ np.asarray(spec.rest))
            positions[f"{side}Hand{finger}{segment + 1}"] = tip
            bones.append((f"{side}Hand{finger}{segment}", f"{side}Hand{finger}{segment + 1}"))
    joints = tuple(positions)
    return joints, np.stack([positions[j] for j in joints], axis=1), tuple(bones)


__all__ = ["skeleton_positions"]
