"""Synthetic UWB YouTube-ASL clip JSON with known 3D geometry (shared by the UWB adapter, converter and gate tests).

The real format (verified on 120 clips, see ``uwb_keypoints``): ``size`` is ``[height, width]``; every point is
``[x, y]`` in source pixels; an undetected part is an empty list."""
from __future__ import annotations

import json

import numpy as np

from semlex_synthetic import T, semlex_array

from signtranslator.data_engineering.video_landmarks import Y_UP
from signtranslator.data_engineering.world_regressor import FEATURES_XY


def uwb_clip_json(size=(1280, 720), *, frames: int = T, seed: int = 1, missing_left=(), missing_right=(),
                  missing_pose=(), missing_face=(), swap_hand_blocks: bool = False):
    """(JSON-ready dict, body (T, 33, 3) y-up, hands {side: (T, 21, 3) y-up}) for a ``size`` = (W, H) frame."""
    width, height = size
    array, body, hands = semlex_array(size, frames=frames, seed=seed)
    pixels = array[..., :2] * [width, height]

    def block(sl, t, missing):
        return [] if t in missing else [[round(float(x), 3), round(float(y), 3)] for x, y in pixels[t, sl]]
    keypoints = []
    for t in range(frames):
        left, right = block(slice(511, 532), t, missing_left), block(slice(532, 553), t, missing_right)
        if swap_hand_blocks:
            left, right = right, left
        keypoints.append(dict(pose_landmarks=block(slice(478, 511), t, missing_pose),
                              left_hand_landmarks=left, right_hand_landmarks=right,
                              face_landmarks=block(slice(0, 478), t, missing_face)))
    data = dict(keypoints=keypoints, cropped_keypoints=keypoints, bbox_left_hand=[[]] * frames,
                bbox_right_hand=[[]] * frames, bbox_face=[[]] * frames, yolo_sign_space=[0, 0, width, height],
                sign_space=[0, 0, width, width], size=[height, width], cropped_size=[512, 512])
    return data, body, hands


def uwb_payload(**kw) -> bytes:
    return json.dumps(uwb_clip_json(**kw)[0]).encode()


class TrueWorldXY:
    """A z-free stand-in regressor returning the clip's true 3D points (MediaPipe axes) and a set visibility."""

    sha256 = "e" * 64
    features = FEATURES_XY

    def __init__(self, body, hands, visibility=None):
        self.body = body @ Y_UP
        self.hand = {s: h @ Y_UP for s, h in hands.items()}
        if visibility is None:
            visibility = np.ones(body.shape[:2])
            visibility[:, 23:] = 0.
        self.visibility = visibility

    def pose_world(self, image, aspect):
        return self.body[:len(image)], self.visibility[:len(image)], np.ones(len(image), bool)

    def hands(self, image, aspect, side):
        h = self.hand[side][:len(image)]
        return h - h[:, :1], np.ones(len(image), bool)
