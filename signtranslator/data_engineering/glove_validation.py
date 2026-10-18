"""Independent validation of solved finger rotations against StretchSense glove output.

3D-LEX's ``Handshape Data`` archives hold, per sign and hand, the StretchSense
glove export: raw capacitances, a 120 Hz device timecode slaved to the Vicon
master clock, and per-joint Euler angles in degrees from the glove's own solve.
The canonical hand channels come from a different pipeline (Shogun Post, FBX,
FBX2glTF). Agreement between the two corroborates the conversion and the solve;
it does not make either an observation, and both stay ``observed=False``.

**Signal.** For finger joints 1–3 of the index, middle, ring and little fingers,
the glove reports flexion as its local z angle (x and y are zero to ``1e-14`` at
the distal joints). On the skeleton side the comparable quantity is the twist of
the joint's rotation relative to its first valid frame about the declared local
axis ``+x``: with ``Q = R(t_0)^T R(t)`` written as a unit quaternion ``(w, v)``
with ``w >= 0``, the twist is ``2 atan2(v . a, w)``. If ``R(t) = R_rest exp(phi(t)
[a]_x)``, then ``Q = exp((phi(t) - phi(t_0)) [a]_x)`` and the twist equals
``phi(t) - phi(t_0)`` exactly; the comparison is centred, so ``phi(t_0)`` and the
glove's zero both cancel. The axis is declared, not fitted, so no sign or
direction is chosen from the data. A 20-sign pilot fixed the axis and side
pairing (glove ``L`` = skeleton ``Left``) before the full run; thumbs were
excluded there because their flexion is not a rotation about local ``x``.

**Clock.** Device timecode ``HHMMSSFFF`` counts 120 Hz frames (``FFF < 120``), so
both gloves of a take sit on one exact frame grid. Repeated timecodes with
identical values are one sample; repeated timecodes with different values are
ambiguous and both are dropped; rows whose labels fall inside a counter restart's
overlap interval are dropped likewise, and a file needing more than 10% of rows
dropped is refused. One offset on that grid, searched exhaustively
with the rate fixed at 120 Hz, must explain every joint of both hands.

**Rule.** The thresholds are the ones already fixed for elbows from the marker
pilot, reused unchanged: per joint, centred error at most 10 degrees, and
correlation at least 0.90 wherever the skeleton joint moves at least 5 degrees.
A hand is validated only if every one of its twelve joints passes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping

import numpy as np
import torch

from ..pose.multichannel import MultichannelMotion
from ..pose.rotations import matrix_to_quaternion, rotation_6d_to_matrix
from .marker_validation import align_joint

GLOVE_RATE_HZ = 120
MIN_CORRELATION = .90
MAX_CENTRED_MAE_DEGREES = 10.
STATIC_RANGE_DEGREES = 5.
MAX_DISORDERED_FRACTION = .1
FINGERS = ("Index", "Middle", "Ring", "Pinky")
VALIDATED_JOINTS = tuple((finger, k) for finger in FINGERS for k in (1, 2, 3))
FLEXION_AXIS = (1., 0., 0.)
_GLOVE_JOINTS = ("hand", *(f"{f}_0{k}" for f in ("index", "middle", "pinky", "ring")
                           for k in (0, 1, 2, 3)), "thumb_01", "thumb_02", "thumb_03")
GLOVE_COLUMNS = (*(f"Capacitance {i} (pF)" for i in range(1, 33)), *(f"IMU {i}" for i in range(1, 11)),
                 "Timecode(device)", "Timer(device)", "Timecode(master)",
                 *(f"{joint}_{axis}" for joint in _GLOVE_JOINTS for axis in "xyz"))
_ANGLES = GLOVE_COLUMNS.index("hand_x")


@dataclass(frozen=True)
class GloveRecording:
    side: str                      # "Left" or "Right", from the metadata
    frames: np.ndarray             # (T,) int64 120 Hz frame index of the device timecode
    angles: np.ndarray             # (T, 3*len(_GLOVE_JOINTS)) degrees, NaN where ambiguous
    ambiguous_frames: int
    collapsed_duplicates: int

    def column(self, joint: str, axis: str) -> np.ndarray:
        return self.angles[:, _GLOVE_JOINTS.index(joint) * 3 + "xyz".index(axis)]


def _device_frame(value: str) -> int:
    if not value.isdigit():
        raise ValueError(f"device timecode {value!r} is not an integer")
    code = int(value)
    hours, rest = divmod(code, 10 ** 7)
    minutes, rest = divmod(rest, 10 ** 5)
    seconds, frame = divmod(rest, 10 ** 3)
    if minutes > 59 or seconds > 59 or frame >= GLOVE_RATE_HZ:
        raise ValueError(f"device timecode {value!r} is out of range")
    return ((hours * 60 + minutes) * 60 + seconds) * GLOVE_RATE_HZ + frame


def parse_glove_metadata(payload: bytes) -> dict:
    meta = json.loads(payload)
    if not isinstance(meta, dict) or meta.get("side") not in ("Left", "Right"):
        raise ValueError("glove metadata must name a Left or Right side")
    if meta.get("fpsRate") != GLOVE_RATE_HZ:
        raise ValueError(f"glove rate {meta.get('fpsRate')!r} is not the declared {GLOVE_RATE_HZ} Hz")
    if meta.get("masterTimeCodeSource") != "Vicon":
        raise ValueError("glove timecode is not slaved to the Vicon master")
    return meta


def parse_glove_csv(payload: bytes, *, side: str) -> GloveRecording:
    """Strictly parse one StretchSense export onto its exact 120 Hz frame index."""
    if side not in ("Left", "Right"):
        raise ValueError("side must be Left or Right")
    lines = payload.decode("utf-8").splitlines()
    if not lines or tuple(c.strip() for c in lines[0].split(",")) != GLOVE_COLUMNS:
        raise ValueError("header is not the declared StretchSense column set")
    frames, values = [], []
    for number, line in enumerate(lines[1:], start=2):
        fields = line.split(",")
        if len(fields) != len(GLOVE_COLUMNS):
            raise ValueError(f"line {number}: expected {len(GLOVE_COLUMNS)} fields")
        frames.append(_device_frame(fields[GLOVE_COLUMNS.index("Timecode(device)")].strip()))
        values.append([float(item) for item in fields[_ANGLES:]])
    if len(frames) < 2:
        raise ValueError("fewer than two glove samples")
    frames = np.asarray(frames, dtype=np.int64)
    angles = np.asarray(values, dtype=np.float64)
    if not np.isfinite(angles).all():
        raise ValueError("non-finite glove angle")
    # The device counter sometimes restarts a second's frame field (000 001 002
    # 000 002 ...). Either occurrence of an overlapping label may be the wrong one,
    # so every row whose label lies in an overlap interval [c_i, max_{j<i} c_j] is
    # ambiguous and dropped from both sides; no order is assumed.
    running = np.maximum.accumulate(frames)
    overlap = np.zeros(len(frames), dtype=bool)
    for i in np.flatnonzero(frames[1:] < running[:-1]) + 1:
        overlap |= (frames >= frames[i]) & (frames <= running[i - 1])
    if overlap.sum() > MAX_DISORDERED_FRACTION * len(frames):
        raise ValueError("glove timecode is disordered beyond the declared limit")
    frames, angles = frames[~overlap], angles[~overlap]
    if np.any(np.diff(frames) < 0):
        raise ValueError("glove timecode decreases outside any resolvable overlap")
    keep, ambiguous, collapsed = [], int(overlap.sum()), 0
    for frame in np.unique(frames):
        rows = np.flatnonzero(frames == frame)
        if len(rows) == 1:
            keep.append((frame, angles[rows[0]]))
        elif np.all(angles[rows] == angles[rows[0]]):
            keep.append((frame, angles[rows[0]]))
            collapsed += len(rows) - 1
        else:
            ambiguous += len(rows)
    if len(keep) < 2:
        raise ValueError("fewer than two unambiguous glove samples")
    return GloveRecording(side, np.array([f for f, _ in keep]), np.array([a for _, a in keep]),
                          ambiguous, collapsed)


def twist_degrees(rotations: np.ndarray, valid: np.ndarray, axis=FLEXION_AXIS) -> np.ndarray:
    """Twist about ``axis`` of each rotation relative to the first valid one, in degrees.

    ``rotations`` is (T, 3, 3); invalid frames return NaN. The series is unwrapped
    across consecutive valid frames, so it is continuous rather than confined to
    (-180, 180].
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    axis = np.asarray(axis, dtype=np.float64)
    if not math.isclose(float(np.linalg.norm(axis)), 1., abs_tol=1e-12):
        raise ValueError("twist axis must be a unit vector")
    out = np.full(len(rotations), np.nan)
    if not valid.any():
        return out
    reference = rotations[np.flatnonzero(valid)[0]]
    relative = np.einsum("ji,tjk->tik", reference, rotations[valid])
    q = matrix_to_quaternion(torch.from_numpy(relative)).numpy()  # (w, x, y, z), w >= 0
    wrapped = np.degrees(2 * np.arctan2(q[:, 1:] @ axis, q[:, 0]))
    # Twist is defined modulo 360 degrees. Consecutive valid frames are unwrapped,
    # which assumes less than half a turn between them (180 degrees per frame
    # interval: thousands of degrees per second for a finger at 24 fps).
    out[valid] = np.unwrap(wrapped, period=360.)
    return out


@dataclass(frozen=True)
class HandValidation:
    status: str                                  # validated | failed | no_alignment | no_glove
    offset_frames: int | None
    joints: dict                                 # label -> per-joint record

    def to_dict(self) -> dict:
        return dict(status=self.status, offset_frames=self.offset_frames, joints=self.joints)


def validate_hands(state: MultichannelMotion, gloves: Mapping[str, GloveRecording], *,
                   min_samples: int = 8) -> dict[str, HandValidation]:
    """One 120 Hz offset for both hands; every finger joint 1-3 must agree."""
    if set(gloves) - {"Left", "Right"}:
        raise ValueError("gloves are keyed by Left/Right")
    for side, glove in gloves.items():
        if glove.side != side:
            raise ValueError("glove metadata side differs from its key")
    result = {side: HandValidation("no_glove", None, {}) for side in ("Left", "Right")
              if side not in gloves}
    if not gloves:
        return result
    start = min(int(g.frames[0]) for g in gloves.values())
    stop = max(int(g.frames[-1]) for g in gloves.values())
    references, markers, keys = [], [], []
    for side, glove in sorted(gloves.items()):
        channel = state.channels["left_hand" if side == "Left" else "right_hand"]
        index = {label: i for i, label in enumerate(channel.labels)}
        # Unavailable cells carry the canonical zero payload, which is not a
        # rotation; only valid cells are converted, the rest stay identity and
        # are excluded by the validity mask.
        matrices = np.broadcast_to(np.eye(3), channel.valid.shape + (3, 3)).copy()
        matrices[channel.valid] = rotation_6d_to_matrix(
            torch.from_numpy(np.ascontiguousarray(channel.values[channel.valid]))).numpy()
        for finger, k in VALIDATED_JOINTS:
            element = index[f"{side}Hand{finger}{k}"]
            references.append(twist_degrees(matrices[:, element], channel.valid[:, element]))
            grid = np.full(stop - start + 1, np.nan)
            grid[glove.frames - start] = glove.column(f"{finger.lower()}_0{k}", "z")
            markers.append(grid)
            keys.append((side, f"{side}Hand{finger}{k}"))
    alignment = align_joint(state.timestamps, references, markers, rates=(float(GLOVE_RATE_HZ),),
                            min_samples=min_samples)
    if alignment is None:
        result.update({side: HandValidation("no_alignment", None, {}) for side in gloves})
        return result
    passed = alignment.passes(min_correlation=MIN_CORRELATION,
                              max_centred_mae_degrees=MAX_CENTRED_MAE_DEGREES,
                              static_range_degrees=STATIC_RANGE_DEGREES)
    for side in gloves:
        joints = {}
        for i, (owner, label) in enumerate(keys):
            if owner == side:
                r = alignment.correlations[i]
                joints[label] = dict(correlation=None if math.isnan(r) else round(r, 6),
                                     centred_mae_degrees=round(alignment.centred_mae_degrees[i], 4),
                                     reference_range_degrees=round(alignment.reference_ranges_degrees[i], 3),
                                     samples=alignment.samples[i], passed=bool(passed[i]))
        status = "validated" if all(j["passed"] for j in joints.values()) else "failed"
        result[side] = HandValidation(status, alignment.offset_frames, joints)
    return result


__all__ = ["GLOVE_RATE_HZ", "GLOVE_COLUMNS", "VALIDATED_JOINTS", "FLEXION_AXIS", "MIN_CORRELATION",
           "MAX_CENTRED_MAE_DEGREES", "STATIC_RANGE_DEGREES", "GloveRecording", "HandValidation",
           "parse_glove_metadata", "parse_glove_csv", "twist_degrees", "validate_hands"]
