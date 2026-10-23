"""A shared 2D keypoint layout for real video landmarks and canonical (real or generated) motion (plan item 7 fallback).

The YouTube-ASL Clip Keypoint Dataset (UWB, CC BY 4.0) gives MediaPipe Holistic 2D landmarks in pixels. Canonical
states hold rotations. To let one "second-opinion" back-translator read both, each is mapped to the same 48 points:
shoulders, elbows and wrists, then the 21 MediaPipe hand points of each hand (wrist, thumb CMC→tip, index MCP→tip,
middle, ring, pinky). Then both are normalised the same way.

- **Video** (``from_holistic``): Holistic's ``left_hand_landmarks`` is the signer's left hand (its wrist coincides
  with pose landmark 15). Image y points down.
- **Canonical motion** (``from_state``): forward kinematics (``generation.canonical_skeleton``, nominal bone lengths)
  then an orthographic front projection, x and y of the camera-aligned y-up frame. The signer's left is +x, as in
  the image of a signer facing the camera.
- **Normalisation** (``normalise``, SignSpace-like): per clip, the median shoulder midpoint becomes the origin and
  the median shoulder width the unit, with y up. Translation inside the clip is kept, since where the hands are in
  signing space carries meaning.

Faces are left out. 2D face landmarks and blendshapes have no common space, so face meaning stays with the other
instruments.
"""
from __future__ import annotations

import numpy as np

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
ARM = ("LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow", "LeftWrist", "RightWrist")
HAND = ("Wrist",) + tuple(f"{f}{k}" for f in FINGERS for k in range(1, 5))
POINTS = ARM + tuple(f"LeftHand{h}" for h in HAND) + tuple(f"RightHand{h}" for h in HAND)
assert len(set(POINTS)) == len(POINTS) == 48
HOLISTIC_POSE = (11, 12, 13, 14, 15, 16)   # MediaPipe pose: shoulders, elbows, wrists (left, right)
_FK_ARM = ("LeftShoulder", "RightShoulder", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand")
FEATURES = 3 * len(POINTS)                 # x, y, observed


def from_holistic(frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(T, 48, 2) pixel points and (T, 48) validity of MediaPipe Holistic frames (``pose_landmarks``,
    ``left_hand_landmarks``, ``right_hand_landmarks``; an absent part is ``None`` or empty)."""
    T = len(frames)
    points, valid = np.zeros((T, len(POINTS), 2), np.float32), np.zeros((T, len(POINTS)), bool)
    for t, f in enumerate(frames):
        pose = f.get("pose_landmarks")
        if pose:
            points[t, :6] = np.asarray(pose, np.float32)[list(HOLISTIC_POSE), :2]
            valid[t, :6] = True
        for k, part in enumerate(("left_hand_landmarks", "right_hand_landmarks")):
            hand = f.get(part)
            if hand:
                a = 6 + 21 * k
                points[t, a:a + 21] = np.asarray(hand, np.float32)[:, :2]
                valid[t, a:a + 21] = True
    return points, valid


def from_state(state) -> tuple[np.ndarray, np.ndarray]:
    """(T, 48, 2) front-projected points (x right, y up, nominal metres) and (T, 48) validity of a canonical state."""
    from ..generation.canonical_skeleton import skeleton_positions
    joints, positions, _ = skeleton_positions(state)
    index = {j: k for k, j in enumerate(joints)}
    names = list(_FK_ARM) + [f"{s}Hand" if h == "Wrist" else f"{s}Hand{h}" for s in ("Left", "Right") for h in HAND]
    points = positions[:, [index[n] for n in names], :2].astype(np.float32)
    body = state.channels["body"]
    arm = {label: body.valid[:, k] for k, label in enumerate(body.labels)}
    both = state.frame_valid if hasattr(state, "frame_valid") else np.ones(len(state.timestamps), bool)
    chest = arm.get("Chest", both)
    valid = np.zeros(points.shape[:2], bool)
    valid[:, 0] = valid[:, 1] = chest
    for k, side in ((0, "Left"), (1, "Right")):
        upper = chest & arm.get(f"{side}UpperArm", chest)
        fore = upper & arm.get(f"{side}Forearm", upper)
        valid[:, 2 + k] = upper
        valid[:, 4 + k] = fore
        hand = state.channels[f"{side.lower()}_hand"].valid.all(axis=1) & fore
        valid[:, 6 + 21 * k:27 + 21 * k] = hand[:, None]
    return points, valid


def normalise(points: np.ndarray, valid: np.ndarray, *, y_down: bool, hands_local: bool = True) -> np.ndarray:
    """Points in shoulder units around the clip's median shoulder midpoint, y up; zeros where invalid. A clip whose
    shoulders are never seen returns all zeros.

    With ``hands_local`` (default) each hand's 20 non-wrist points are taken relative to that hand's wrist and
    scaled by the clip's median palm width (index to pinky knuckle), so hand shape and orientation are compared
    independently of hand size. On 20 clips each, forward kinematics' nominal hands came out about 0.65× the video
    hands in shoulder units (hand length 0.25 vs 0.39), while the arms agreed (forearm 0.58 vs 0.58). The wrist
    point keeps its place in signing space."""
    p = points.astype(np.float64).copy()
    if y_down:
        p[..., 1] = -p[..., 1]
    both = valid[:, 0] & valid[:, 1]
    if not both.any():
        return np.zeros_like(points, dtype=np.float32)
    mid = np.median((p[both, 0] + p[both, 1]) / 2, axis=0)
    width = float(np.median(np.linalg.norm(p[both, 0] - p[both, 1], axis=-1)))
    out = (p - mid) / max(width, 1e-6)
    if hands_local:
        for k in range(2):
            a = 6 + 21 * k
            wrist, index1, pinky1 = a, a + 5, a + 17
            ok = valid[:, wrist] & valid[:, index1] & valid[:, pinky1]
            if ok.any():
                palm = float(np.median(np.linalg.norm(p[ok, index1] - p[ok, pinky1], axis=-1)))
                out[:, a + 1:a + 21] = (p[:, a + 1:a + 21] - p[:, wrist:wrist + 1]) / max(palm, 1e-6)
    out[~valid] = 0.
    return out.astype(np.float32)


def resample(points: np.ndarray, valid: np.ndarray, *, fps: float, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear resampling in time; a target frame is valid where both neighbouring source frames are."""
    T = len(points)
    if T == 0 or abs(fps - target_fps) < 1e-6:
        return points, valid
    t = np.arange(int(np.floor((T - 1) * target_fps / fps)) + 1) * fps / target_fps
    lo = np.clip(np.floor(t).astype(int), 0, T - 1)
    hi = np.clip(lo + 1, 0, T - 1)
    w = (t - lo)[:, None, None]
    out = points[lo] * (1 - w) + points[hi] * w
    ok = valid[lo] & valid[hi]
    out[~ok] = 0.
    return out.astype(np.float32), ok


def features(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(T, 144): x, y and an observed flag per point."""
    return np.concatenate([points, valid[..., None].astype(np.float32)], axis=-1).reshape(len(points), -1)


__all__ = ["POINTS", "FEATURES", "HOLISTIC_POSE", "from_holistic", "from_state", "normalise", "resample", "features"]
