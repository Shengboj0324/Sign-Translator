"""Left-right mirror of a canonical state (swing-IK convention ``mediapipe-tasks-swing-ik-y-up-v1``).

Mirroring the signer across the sagittal plane turns a left-dominant signer into a right-dominant one. In the
camera-aligned y-up frames x is lateral (toward the signer's left), so the reflection is ``M = diag(-1, 1, 1)``.
Each channel's rule follows from how ``video_landmarks`` builds its frames:

- **Pelvis, chest, arms, legs and the head:** frames whose x axis is lateral and whose rest directions mirror
  across sides (``(+1,0,0)`` for the left arm, ``(-1,0,0)`` for the right, ``(0,-1,0)`` for both legs). A mirrored
  local rotation is ``M R M``, and the left and right labels swap.
- **Wrists (body ``LeftHand``/``RightHand``):** the palm frame is ``frame_from_axes(p5 - p17, p9 - p0)`` on both
  hands. Reflected points give the frame ``M F N`` with ``N = diag(1, 1, -1)``, so the stored wrist rotation
  (forearm-relative) becomes ``M R N``.
- **Fingers:** they are children of the palm frame, and every finger rest direction and knuckle offset lies in
  its xy plane (z = 0). A mirrored finger rotation is ``N R N``, and the two hand channels swap.
- **Translations** (root, head) negate x. **Gaze** swaps eyes and negates x.
- **Face blendshapes and blinks** swap their ``Left``/``Right`` pairs; the others are unchanged.

Validity, observation and confidence masks move with their values. ``tests/test_pose_mirror.py`` checks the
result against forward kinematics: every mirrored joint position is ``M`` times its counterpart's.
"""
from __future__ import annotations

from dataclasses import replace
import re

import numpy as np
import torch

from .multichannel import MotionChannel, MultichannelMotion
from .rotations import matrix_to_rotation_6d, rotation_6d_to_matrix

M = np.diag([-1., 1., 1.])
N = np.diag([1., 1., -1.])
CONVENTION = "mediapipe-tasks-swing-ik-y-up-v1"
_SIDE = re.compile(r"Left|Right")


def swap_side(label: str) -> str:
    """``LeftForearm`` <-> ``RightForearm``, ``browDownLeft`` <-> ``browDownRight``; other labels unchanged."""
    return _SIDE.sub(lambda m: "Right" if m.group(0) == "Left" else "Left", label)


def _rotate(values: np.ndarray, valid: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """6D rotations R -> left @ R @ right where valid; invalid payload stays zero."""
    if not valid.any():
        return values
    out = values.copy()
    mats = rotation_6d_to_matrix(torch.from_numpy(values[valid].astype(np.float64))).numpy()
    out[valid] = matrix_to_rotation_6d(torch.from_numpy(np.ascontiguousarray(left @ mats @ right))).numpy()
    return out


def _permuted(channel: MotionChannel, order: list[int], labels: tuple[str, ...], values: np.ndarray) -> MotionChannel:
    return replace(channel, values=np.ascontiguousarray(values[:, order]), valid=channel.valid[:, order].copy(),
                   observed=channel.observed[:, order].copy(), confidence=channel.confidence[:, order].copy(),
                   labels=labels)


def _swap_within(channel: MotionChannel, values: np.ndarray | None = None) -> MotionChannel:
    """Reorder so each label's new value is its counterpart's (labels keep their positions)."""
    labels = list(channel.labels)
    order = [labels.index(swap_side(l)) if swap_side(l) in labels else k for k, l in enumerate(labels)]
    return _permuted(channel, order, channel.labels, channel.values if values is None else values)


def mirror_state(state: MultichannelMotion, *, sample_id: str | None = None) -> MultichannelMotion:
    """The state of the mirrored signer (see the module docstring for each channel's rule)."""
    ch = state.channels
    for name in ("body", "left_hand", "right_hand"):
        if ch[name].valid.any() and ch[name].convention != CONVENTION:
            raise ValueError(f"{name}: mirroring is defined for {CONVENTION}, not {ch[name].convention}")
    out = dict(ch)
    body = ch["body"]
    rules = {label: ((M, N) if label in ("LeftHand", "RightHand") else (M, M)) for label in body.labels}
    values = body.values.copy()
    for k, label in enumerate(body.labels):
        left, right = rules[label]
        values[:, k] = _rotate(body.values[:, k], body.valid[:, k], left, right)
    out["body"] = _swap_within(body, values)
    for name, other in (("left_hand", "right_hand"), ("right_hand", "left_hand")):
        src = ch[other]
        flipped = np.stack([_rotate(src.values[:, k], src.valid[:, k], N, N) for k in range(src.values.shape[1])], 1) \
            if src.values.shape[1] else src.values
        labels = tuple(swap_side(l) for l in src.labels)
        out[name] = replace(src, values=flipped, labels=labels)
    head = ch["head_rotation"]
    out["head_rotation"] = replace(head, values=_rotate(head.values[:, 0], head.valid[:, 0], M, M)[:, None])
    for name in ("root_translation", "head_translation"):
        c = ch[name]
        out[name] = replace(c, values=c.values * np.where(c.valid[..., None], np.array([-1., 1., 1.]), 0.) + 0.)
    for name, other in (("left_gaze", "right_gaze"), ("right_gaze", "left_gaze")):
        src, own = ch[other], ch[name]
        if src.valid.any():
            flipped = src.values * np.where(src.valid[..., None], np.array([-1., 1., 1.]), 0.) + 0.
            out[name] = replace(src, values=flipped, labels=tuple(swap_side(l) for l in src.labels))
        else:   # the counterpart eye is unavailable, so this eye is too
            out[name] = replace(own, values=np.zeros_like(own.values), valid=np.zeros_like(own.valid),
                                observed=np.zeros_like(own.observed), confidence=np.zeros_like(own.confidence),
                                inference_method=None)
    for name in ("face", "blink"):
        out[name] = _swap_within(ch[name])
    return MultichannelMotion(state.timestamps, state.clock_id, sample_id or f"{state.sample_id}-mirrored", out,
                              state.schema_version)


__all__ = ["M", "N", "mirror_state", "swap_side"]
