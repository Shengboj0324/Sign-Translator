"""Chest-centred body frame for spatial grammar.

Canonical states hold local joint rotations (swing IK, ``video_landmarks``), so
positions come from the nominal-length forward kinematics of
``generation.canonical_skeleton`` — signer-normalised, which is what spatial
grammar needs (Lu & Huenerfauth express loci in shoulder widths). With ``C(t)``
the chest's world rotation (Pelvis · Chest) and ``s(t)`` the chest point
(``Spine2``), a point maps to the body frame as ``p_B = C(t)ᵀ (p − s(t))``: x toward
the signer's left, y up, z forward (toward the addressee), the axes of the
camera-aligned y-up frames of both the monocular and two-view states.

Azimuth ``φ = atan2(x, z)`` (positive to the signer's left) and elevation
``θ = atan2(y, √(x² + z²))``. Lengths are in units of the nominal shoulder width.
"""
from __future__ import annotations

import numpy as np

from ..generation.canonical_skeleton import _SHOULDER_HALF, skeleton_positions
from ..generation.motion_metrics import _matrices
from ..pose.multichannel import MultichannelMotion

SHOULDER_WIDTH = 2 * _SHOULDER_HALF          # nominal metres; unit of body-frame lengths


def chest_rotation(state: MultichannelMotion) -> np.ndarray:
    """(T, 3, 3) world rotation of the chest (identity where the body is invalid)."""
    body = state.channels["body"]
    labels = list(body.labels)
    pelvis, chest = _matrices(body.values[:, labels.index("Pelvis")]), _matrices(body.values[:, labels.index("Chest")])
    ok = body.valid[:, labels.index("Pelvis")] & body.valid[:, labels.index("Chest")]
    return np.where(ok[:, None, None], pelvis @ chest, np.eye(3))


def body_frame_positions(state: MultichannelMotion):
    """(joint names, (T, J, 3) chest-frame positions in shoulder widths, (T, J) validity)."""
    joints, positions, _ = skeleton_positions(state)
    chest = chest_rotation(state)
    spine = positions[:, joints.index("Spine2")]
    local = np.einsum("tji,tpj->tpi", chest, positions - spine[:, None]) / SHOULDER_WIDTH
    body = state.channels["body"]
    labels = list(body.labels)
    body_ok = body.valid[:, labels.index("Pelvis")] & body.valid[:, labels.index("Chest")]
    valid = np.repeat(body_ok[:, None], len(joints), axis=1)
    for side in ("Left", "Right"):
        arm_ok = body.valid[:, labels.index(f"{side}UpperArm")] & body.valid[:, labels.index(f"{side}Forearm")]
        hand = state.channels[f"{side.lower()}_hand"]
        for k, name in enumerate(joints):
            if name.startswith(f"{side}Hand") or name in (f"{side}ForeArm",):
                valid[:, k] &= arm_ok
                if name.startswith(f"{side}Hand") and name != f"{side}Hand":
                    valid[:, k] &= hand.valid.all(axis=1)
    return joints, local, valid


def to_body_frame(state: MultichannelMotion, vectors: np.ndarray) -> np.ndarray:
    """(T, 3) world directions (e.g. gaze in the camera-aligned y-up frame) rotated into the body frame."""
    return np.einsum("tji,tj->ti", chest_rotation(state), vectors)


def azimuth_elevation(vectors: np.ndarray) -> np.ndarray:
    """(..., 2) azimuth and elevation in radians of body-frame vectors."""
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    return np.stack((np.arctan2(x, z), np.arctan2(y, np.hypot(x, z))), axis=-1)


def dominant_side(joints, positions: np.ndarray, valid: np.ndarray) -> str:
    """'Left' or 'Right': the hand whose wrist travels further (path length) over the clip."""
    travel = {}
    for side in ("Left", "Right"):
        k = joints.index(f"{side}Hand")
        p, ok = positions[:, k], valid[:, k]
        both = ok[1:] & ok[:-1]
        travel[side] = float(np.linalg.norm(np.diff(p, axis=0), axis=1)[both].sum())
    return max(travel, key=travel.get)


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(n > 1e-12, n, 1.)


__all__ = ["SHOULDER_WIDTH", "chest_rotation", "body_frame_positions", "to_body_frame", "azimuth_elevation",
           "dominant_side", "unit"]
