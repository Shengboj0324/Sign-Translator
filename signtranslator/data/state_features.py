"""Per-frame input of instrument v2, the state-to-text back-translator (docs/NEXT_PHASE_PLAN.md, "Instrument v2: design
and declared gates", 2026-10-30 09:10 UTC).

Instrument v1 (``keypoints2d``) reads 48 forward-kinematics points projected to 2D and drops depth and every native
channel. Instrument v2 reads the generator's own state, conformed to the ``how2sign-two-view-text-v1`` template, as
four blocks per frame (``state_features``):

1. **3D points** (48 x 3): v1's 48 points (shoulders, elbows, wrists, both hands' 21 points) by forward kinematics
   (``generation.canonical_skeleton``, nominal bone lengths) with z kept, normalised as v1 normalises them but in 3D
   (``normalise_nd``): the clip's median shoulder midpoint is the origin, its median shoulder width the unit, and
   each hand's 20 non-wrist points are taken relative to that hand's wrist in units of its median palm width. The
   canonical frame is y up, so nothing is flipped. Zeros where a point is invalid.
2. **Point validity** (48): v1's validity rule (``point_validity``, the same as ``keypoints2d.from_state``).
3. **Native channels**: every channel the instrument reads (``StateToT5Config.motion``'s trained channels), in the
   diffusion vector space of ``text_to_motion`` (6D rotations and gaze as stored, face coefficients / 0.5, blink
   2v - 1, translations / 0.1 m). Root and head translation are first centred on the clip's median over its valid
   frames, so only movement within the clip is read. Invalid cells are zero.
4. **Element validity** (one flag per channel element: a joint, a face coefficient, the head): valid and on a real
   frame.

A blank state (every channel invalid) therefore gives all-zero features for its duration: only its length remains.
"""
from __future__ import annotations

import numpy as np

from .keypoints2d import HAND, POINTS

FEATURE_VERSION = "state-features-v1"
TRANSLATION_CHANNELS = ("root_translation", "head_translation")   # centred on the clip's median
NATIVE_CHANNELS = ("body", "left_hand", "right_hand", "head_rotation", "head_translation", "root_translation", "face",
                   "blink", "left_gaze", "right_gaze")              # every canonical channel but contact
_FK_ARM = ("LeftShoulder", "RightShoulder", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand")
_FK_NAMES = list(_FK_ARM) + [f"{s}Hand" if h == "Wrist" else f"{s}Hand{h}" for s in ("Left", "Right") for h in HAND]


def point_validity(state) -> np.ndarray:
    """(T, 48) validity of the 48 points: ``keypoints2d.from_state``'s rule (chest for the shoulders, the upper arm
    for the elbow, the forearm for the wrist, every finger joint of the hand for its 21 points)."""
    body = state.channels["body"]
    arm = {label: body.valid[:, k] for k, label in enumerate(body.labels)}
    both = state.frame_valid if hasattr(state, "frame_valid") else np.ones(len(state.timestamps), bool)
    chest = arm.get("Chest", both)
    valid = np.zeros((len(state.timestamps), len(POINTS)), bool)
    valid[:, 0] = valid[:, 1] = chest
    for k, side in ((0, "Left"), (1, "Right")):
        upper = chest & arm.get(f"{side}UpperArm", chest)
        fore = upper & arm.get(f"{side}Forearm", upper)
        valid[:, 2 + k] = upper
        valid[:, 4 + k] = fore
        hand = state.channels[f"{side.lower()}_hand"].valid.all(axis=1) & fore
        valid[:, 6 + 21 * k:27 + 21 * k] = hand[:, None]
    return valid


def from_state_3d(state) -> tuple[np.ndarray, np.ndarray]:
    """(T, 48, 3) forward-kinematics points (x to the signer's left, y up, z towards the camera; nominal metres) and
    (T, 48) validity. Columns 0 and 1 are exactly ``keypoints2d.from_state``'s points."""
    from ..generation.canonical_skeleton import skeleton_positions
    joints, positions, _ = skeleton_positions(state)
    index = {j: k for k, j in enumerate(joints)}
    points = positions[:, [index[n] for n in _FK_NAMES], :].astype(np.float32)
    points = np.where(np.isfinite(points), points, 0.).astype(np.float32)
    return points, point_validity(state)


def normalise_nd(points: np.ndarray, valid: np.ndarray, *, hands_local: bool = True) -> np.ndarray:
    """``keypoints2d.normalise`` for points of any dimension, without a y flip (canonical states are y up): shoulder
    units around the clip's median shoulder midpoint, each hand's non-wrist points relative to its wrist in units of
    the clip's median palm width (index to pinky knuckle), zeros where invalid; all zeros when the shoulders are never
    valid. For 2D points it equals ``keypoints2d.normalise(points, valid, y_down=False)``."""
    p = points.astype(np.float64).copy()
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


def _to_unit(kind: str, values: np.ndarray, coefficient_norm: float) -> np.ndarray:
    """``text_to_motion._native_to_unit`` in NumPy."""
    from ..generation.text_to_motion import _TRANSLATION_NORM
    if kind == "coefficient":
        return values / coefficient_norm
    if kind == "bounded_scalar":
        return 2 * values - 1
    if kind == "translation":
        return values / _TRANSLATION_NORM
    return values


def native_features(state, motion) -> tuple[np.ndarray, np.ndarray]:
    """((T, W) native values in the diffusion space, (T, E) element validity) of the trained channels of ``motion``
    (a ``TextToMotionConfig``), in its binding order. The state's layouts must be the bindings' (conform first)."""
    from ..generation.text_to_motion import diffusion_layout
    bindings = {b.name: b for b in motion.bindings}
    T = len(state.timestamps)
    values_out, flags_out = [], []
    for name, kind, elements, width in diffusion_layout(motion):
        c, b = state.channels[name], bindings[name]
        if (c.labels, c.coordinate_frame, c.units, c.convention) != (b.labels, b.coordinate_frame, b.units,
                                                                      b.convention):
            raise ValueError(f"{name}: the state's layout is not the instrument's (conform it to the template first)")
        values = c.values.astype(np.float64)
        valid = c.valid
        if name in TRANSLATION_CHANNELS:
            values = values.copy()
            for j in range(values.shape[1]):
                if valid[:, j].any():
                    values[:, j] -= np.median(values[valid[:, j], j], axis=0)
        unit = np.where(valid[..., None], _to_unit(kind, values, motion.coefficient_norm), 0.)
        values_out.append(unit.reshape(T, width))
        flags_out.append(valid.astype(np.float32))
    return (np.concatenate(values_out, axis=-1).astype(np.float32),
            np.concatenate(flags_out, axis=-1).astype(np.float32))


def feature_width(motion) -> int:
    """Width of ``state_features`` for an instrument whose native channels are ``motion``'s trained channels."""
    from ..generation.text_to_motion import diffusion_layout
    layout = diffusion_layout(motion)
    return 4 * len(POINTS) + sum(width for *_, width in layout) + sum(elements for _, _, elements, _ in layout)


def state_features(state, motion) -> np.ndarray:
    """(T, ``feature_width(motion)``) float32: normalised 3D points, point validity, native values, element
    validity (module docstring)."""
    points, valid = from_state_3d(state)
    native, flags = native_features(state, motion)
    out = np.concatenate([normalise_nd(points, valid).reshape(len(points), -1), valid.astype(np.float32), native,
                          flags], axis=-1)
    if not np.isfinite(out).all():
        raise ValueError(f"{state.sample_id}: non-finite features")
    return out.astype(np.float32)


__all__ = ["FEATURE_VERSION", "TRANSLATION_CHANNELS", "NATIVE_CHANNELS", "point_validity", "from_state_3d",
           "normalise_nd", "native_features", "feature_width", "state_features"]
