"""Absolute shake and articulation measures of a posed clip, in metres, degrees and rad/s.

Nothing here reads a text, a model or an arm label: every number is a function of one retargeted clip.
Used to choose the render filter (:mod:`.motion_filter`) and to report it; it decides nothing about signing.

**Shake.** ``jitter`` is the RMS distance between a trajectory and its own zero-phase 6 Hz low-pass (a position in
millimetres, a rotation in degrees), so it is an amplitude a viewer sees, not a ratio against real. ``hf_share`` is the
share of velocity power at or above 6 Hz; white noise at 24 fps gives about 0.5, and signing is mostly below 3 Hz.

**Articulation.** Finger speed (rad/s) raw and at the 6 Hz and 3 Hz bands, finger bend range (degrees), peak wrist speed
(m/s) and wrist holds. A hold is at least ``HOLD_MIN_SECONDS`` with 6 Hz wrist speed under ``HOLD_SPEED``. These are read
before and after a filter on the same clip.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .makehuman import SOURCE_METERS_PER_UNIT

JITTER_HZ = 6.
HOLD_SPEED = .15            # m/s of a wrist, at the 6 Hz band
HOLD_MIN_SECONDS = .17
SIDES = ("L", "R")
FINGER_COUNT = 5


def lowpass(values: np.ndarray, fps: float, hz: float, *, axis: int = 0) -> np.ndarray:
    """Zero-phase 4th-order Butterworth, odd padding inside ``sosfiltfilt``; a clip too short for it is returned as is."""
    if hz >= fps / 2:
        return values.copy()
    sos = butter(2, hz / (fps / 2), output="sos")          # 2nd order twice over (forward, backward) = 4th
    n = values.shape[axis]
    pad = min(n - 1, int(round(3 * fps / hz)))
    if pad < 1:
        return values.copy()
    return sosfiltfilt(sos, values, axis=axis, padlen=pad)


def hf_share(positions: np.ndarray, fps: float, hz: float = JITTER_HZ) -> float:
    """Share of velocity power at or above ``hz``, over all elements of (T, ..., D)."""
    velocity = np.diff(positions, axis=0).reshape(len(positions) - 1, -1)
    if len(velocity) < 8:
        return float("nan")
    velocity = velocity - velocity.mean(0)
    power = (np.abs(np.fft.rfft(velocity, axis=0)) ** 2).sum(1)
    f = np.fft.rfftfreq(len(velocity), 1. / fps)
    total = power[f > 0].sum()
    return float(power[f >= hz].sum() / total) if total > 0 else 0.


def position_jitter_mm(positions: np.ndarray, fps: float, hz: float = JITTER_HZ) -> float:
    """RMS distance (mm) from the trajectory's own low-pass, for (T, 3) in source units."""
    residual = positions - lowpass(positions, fps, hz)
    return float(np.sqrt((residual ** 2).sum(-1).mean()) * SOURCE_METERS_PER_UNIT * 1000.)


def _rotation_columns(rotations: np.ndarray) -> np.ndarray:
    """Two columns (the state's own 6D form), continuous under low-pass, unlike angles or log maps."""
    return np.concatenate((rotations[..., 0], rotations[..., 1]), axis=-1)


def _gram_schmidt(values: np.ndarray) -> np.ndarray:
    a, b = values[..., :3], values[..., 3:]
    r1 = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-9)
    b = b - (r1 * b).sum(-1, keepdims=True) * r1
    r2 = b / np.linalg.norm(b, axis=-1, keepdims=True).clip(1e-9)
    return np.stack((r1, r2, np.cross(r1, r2)), axis=-1)


def geodesic_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(a, -1, -2) @ b
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.) / 2., -1., 1.)
    return np.degrees(np.arccos(cosine))


def rotation_jitter_degrees(rotations: np.ndarray, fps: float, hz: float = JITTER_HZ) -> float:
    """RMS geodesic angle (degrees) between (T, 3, 3) rotations and their own low-pass."""
    smooth = _gram_schmidt(lowpass(_rotation_columns(rotations), fps, hz))
    return float(np.sqrt((geodesic_degrees(rotations, smooth) ** 2).mean()))


def rotation_hf_share(rotations: np.ndarray, fps: float, hz: float = JITTER_HZ) -> float:
    return hf_share(_rotation_columns(rotations), fps, hz)


def angular_speed(rotations: np.ndarray, fps: float) -> np.ndarray:
    """Geodesic speed (rad/s), (T-1,) from (T, 3, 3)."""
    return np.radians(geodesic_degrees(rotations[:-1], rotations[1:])) * fps


def forward_kinematics(rig, local_rotations: np.ndarray) -> np.ndarray:
    """(T, B, 4, 4) bone globals from (T, B, 3, 3) locals, batched; the same arithmetic as ``MakeHumanRig.pose``."""
    rest = rig.rest_globals.detach().cpu().numpy()
    inverse = np.linalg.inv(rest)
    frames, bones = local_rotations.shape[:2]
    out = np.zeros((frames, bones, 4, 4))
    out[..., 3, 3] = 1.
    local = out.copy()
    local[..., :3, :3] = local_rotations
    for i, parent in enumerate(rig.parents):
        relative = rest[i] if parent < 0 else inverse[parent] @ rest[i]
        out[:, i] = relative @ local[:, i] if parent < 0 else out[:, parent] @ relative @ local[:, i]
    return out


def bone_index(rig) -> dict[str, int]:
    return {name: i for i, name in enumerate(rig.bone_names)}


def fingertip_positions(rig, globals_: np.ndarray, side: str) -> np.ndarray:
    """(T, 5, 3) tips: each last phalanx's head plus its rest length along its own y axis."""
    index = bone_index(rig)
    rest = rig.rest_globals.detach().cpu().numpy()
    out = []
    for k in range(1, FINGER_COUNT + 1):
        i, parent = index[f"finger{k}-3.{side}"], index[f"finger{k}-2.{side}"]
        length = float(np.linalg.norm(rest[i, :3, 3] - rest[parent, :3, 3]))     # the middle phalanx stands in for the last
        out.append(globals_[:, i, :3, 3] + .8 * length * globals_[:, i, :3, 1])
    return np.stack(out, axis=1)


POINTS = {"head": "head", "neck": "neck01", "chest": "spine02", "shoulder.L": "upperarm01.L", "shoulder.R": "upperarm01.R",
          "elbow.L": "lowerarm01.L", "elbow.R": "lowerarm01.R", "wrist.L": "wrist.L", "wrist.R": "wrist.R"}
ROTATIONS = {"head": "head", "chest": "spine02", "upperarm.L": "upperarm01.L", "upperarm.R": "upperarm01.R",
             "forearm.L": "lowerarm01.L", "forearm.R": "lowerarm01.R", "hand.L": "wrist.L", "hand.R": "wrist.R"}


def _bands(positions: np.ndarray, fps: float) -> dict:
    """RMS amplitude (mm) of a (T, 3) trajectory in three bands."""
    low3 = lowpass(positions, fps, 3.)
    low6 = lowpass(positions, fps, 6.)
    rms = lambda x: float(np.sqrt(((x - x.mean(0)) ** 2).sum(-1).mean()) * SOURCE_METERS_PER_UNIT * 1000.)
    return dict(below_3hz=rms(low3), from_3_to_6hz=rms(low6 - low3), above_6hz=rms(positions - low6))


def shake(rig, local_rotations: np.ndarray, fps: float, *, screen_pixels_per_unit: float | None = None) -> dict:
    """Shake of one clip: per world point (mm, bands, share), per bone frame (degrees, share), fingertips (mm)."""
    globals_ = forward_kinematics(rig, local_rotations)
    index = bone_index(rig)
    points, rotations = {}, {}
    for name, bone in POINTS.items():
        trajectory = globals_[:, index[bone], :3, 3]
        points[name] = dict(jitter_mm=position_jitter_mm(trajectory, fps), hf_share=hf_share(trajectory, fps),
                            **_bands(trajectory, fps))
        if screen_pixels_per_unit is not None:
            residual = trajectory - lowpass(trajectory, fps, JITTER_HZ)
            points[name]["jitter_px"] = float(np.sqrt((residual[:, :2] ** 2).sum(-1).mean()) * screen_pixels_per_unit)
    for name, bone in ROTATIONS.items():
        frames = globals_[:, index[bone], :3, :3]
        rotations[name] = dict(jitter_deg=rotation_jitter_degrees(frames, fps), hf_share=rotation_hf_share(frames, fps))
    tips = {}
    for side in SIDES:
        trajectory = fingertip_positions(rig, globals_, side)
        residual = trajectory - lowpass(trajectory, fps, JITTER_HZ)
        tips[side] = dict(jitter_mm=float(np.sqrt((residual ** 2).sum(-1).mean()) * SOURCE_METERS_PER_UNIT * 1000.),
                          hf_share=hf_share(trajectory, fps))
    return dict(points=points, rotations=rotations, fingertips=tips)


def _finger_chain(rig, side: str):
    index = bone_index(rig)
    return [(index[f"finger{k}-{s}.{side}"], k, s) for k in range(1, FINGER_COUNT + 1) for s in (1, 2, 3)]


def _bends(globals_: np.ndarray, chain, parents) -> np.ndarray:
    """Angle (degrees) between each finger bone's y axis and its parent's, (T, J): no twist, so no estimator twist noise."""
    out = []
    for bone, _, _ in chain:
        a = globals_[:, bone, :3, 1]
        b = globals_[:, parents[bone], :3, 1]
        out.append(np.degrees(np.arccos(np.clip((a * b).sum(-1), -1., 1.))))
    return np.stack(out, axis=1)


def _holds(speed: np.ndarray, fps: float) -> np.ndarray:
    """Mask of frames in a run of at least ``HOLD_MIN_SECONDS`` under ``HOLD_SPEED``."""
    below = speed < HOLD_SPEED
    mask = np.zeros(len(below), dtype=bool)
    need = int(np.ceil(HOLD_MIN_SECONDS * fps))
    start = None
    for t in range(len(below) + 1):
        if t < len(below) and below[t]:
            start = t if start is None else start
        elif start is not None:
            if t - start >= need:
                mask[start:t] = True
            start = None
    return mask


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def articulation(rig, local_rotations: np.ndarray, fps: float) -> dict:
    """Finger speed and range, wrist peaks and holds of one clip; every value per side, for before/after reads."""
    globals_ = forward_kinematics(rig, local_rotations)
    index = bone_index(rig)
    out = {}
    for side in SIDES:
        chain = _finger_chain(rig, side)
        bends = _bends(globals_, chain, rig.parents)
        bend_speed = {band: np.abs(np.diff(x, axis=0)).mean(1) * fps * np.pi / 180.
                      for band, x in (("raw", bends), ("6hz", lowpass(bends, fps, 6.)), ("3hz", lowpass(bends, fps, 3.)))}
        wrist = globals_[:, index[f"wrist.{side}"], :3, 3] * SOURCE_METERS_PER_UNIT
        speed_raw = np.linalg.norm(np.diff(wrist, axis=0), axis=-1) * fps
        speed_6 = np.linalg.norm(np.diff(lowpass(wrist, fps, 6.), axis=0), axis=-1) * fps
        holds = _holds(speed_6, fps)
        residual = wrist - lowpass(wrist, fps, 6.)
        inside = float(np.sqrt((residual[:-1][holds] ** 2).sum(-1).mean()) * 1000.) if holds.any() else None
        finger_step = np.abs(np.diff(bends, axis=0)).mean(1) * fps * np.pi / 180.
        out[side] = dict(
            finger_speed_6hz_series=bend_speed["6hz"], finger_speed_raw_series=finger_step,
            wrist_speed_6hz_series=speed_6,
            finger_speed_rad_s={band: float(v.mean()) for band, v in bend_speed.items()},
            finger_speed_peak_rad_s={band: float(np.percentile(v, 95)) for band, v in bend_speed.items()},
            finger_range_deg=float((np.percentile(bends, 95, axis=0) - np.percentile(bends, 5, axis=0)).mean()),
            wrist_speed_m_s=dict(mean_raw=float(speed_raw.mean()), mean_6hz=float(speed_6.mean()),
                                 p95_raw=float(np.percentile(speed_raw, 95)), p95_6hz=float(np.percentile(speed_6, 95)),
                                 max_raw=float(speed_raw.max()), max_6hz=float(speed_6.max())),
            holds=dict(count=len(_runs(holds)), frames=int(holds.sum()), share=float(holds.mean()),
                       mask=holds, jitter_inside_mm=inside),
            bends=bends, wrist=wrist)
    return out


def preservation(before: dict, after: dict, fps: float) -> dict:
    """After / before on one clip, per side: the numbers a filter must keep near 1 (and the holds' overlap)."""
    out = {}
    for side in SIDES:
        a, b = before[side], after[side]
        ratio = lambda x, y: float(y / x) if x > 1e-9 else None
        ma, mb = a["holds"]["mask"], b["holds"]["mask"]
        union = (ma | mb).sum()
        lag_bend = _lag(a["bends"], b["bends"])
        hold_shift = None
        if ma.any():
            displacement = np.linalg.norm(a["wrist"][:-1][ma] - b["wrist"][:-1][ma], axis=-1)
            hold_shift = float(np.percentile(displacement, 95) * 1000.)
        out[side] = dict(
            finger_speed_raw=ratio(a["finger_speed_rad_s"]["raw"], b["finger_speed_rad_s"]["raw"]),
            finger_speed_6hz=ratio(a["finger_speed_rad_s"]["6hz"], b["finger_speed_rad_s"]["6hz"]),
            finger_speed_3hz=ratio(a["finger_speed_rad_s"]["3hz"], b["finger_speed_rad_s"]["3hz"]),
            finger_peak_6hz=ratio(a["finger_speed_peak_rad_s"]["6hz"], b["finger_speed_peak_rad_s"]["6hz"]),
            finger_range=ratio(a["finger_range_deg"], b["finger_range_deg"]),
            wrist_p95_6hz=ratio(a["wrist_speed_m_s"]["p95_6hz"], b["wrist_speed_m_s"]["p95_6hz"]),
            wrist_max_6hz=ratio(a["wrist_speed_m_s"]["max_6hz"], b["wrist_speed_m_s"]["max_6hz"]),
            wrist_max_raw=ratio(a["wrist_speed_m_s"]["max_raw"], b["wrist_speed_m_s"]["max_raw"]),
            hold_iou=float((ma & mb).sum() / union) if union else None,
            hold_count_before=a["holds"]["count"], hold_count_after=b["holds"]["count"],
            hold_share_before=a["holds"]["share"], hold_share_after=b["holds"]["share"],
            hold_jitter_before_mm=a["holds"]["jitter_inside_mm"], hold_jitter_after_mm=b["holds"]["jitter_inside_mm"],
            hold_position_shift_p95_mm=hold_shift, finger_bend_lag_frames=lag_bend,
            finger_active_6hz=_active_ratio(a["finger_speed_6hz_series"], b["finger_speed_6hz_series"]),
            wrist_active_6hz=_active_ratio(a["wrist_speed_6hz_series"], b["wrist_speed_6hz_series"]),
            finger_quiet_raw_before=float(np.percentile(a["finger_speed_raw_series"], 10)),
            finger_quiet_raw_after=float(np.percentile(b["finger_speed_raw_series"], 10)))
    return out


def _active_ratio(before: np.ndarray, after: np.ndarray, top: float = 70.) -> float | None:
    """Mean speed after / before on the frames where the unfiltered 6 Hz speed is in its top 30%: the movements."""
    mask = before >= np.percentile(before, top)
    total = before[mask].mean()
    return float(after[mask].mean() / total) if total > 1e-9 else None


def _lag(a: np.ndarray, b: np.ndarray, reach: int = 4) -> int:
    """Whole-frame lag of ``b`` behind ``a`` that best correlates their bend velocities; 0 for a zero-phase filter."""
    va, vb = np.diff(a, axis=0), np.diff(b, axis=0)
    best, lag = -np.inf, 0
    for k in range(-reach, reach + 1):
        x, y = (va[:len(va) - k], vb[k:]) if k >= 0 else (va[-k:], vb[:k])
        score = float((x * y).sum())
        if score > best:
            best, lag = score, k
    return lag


def strip_arrays(record: dict) -> dict:
    """An :func:`articulation` record without its arrays, for JSON."""
    out = {}
    for side, values in record.items():
        kept = {k: v for k, v in values.items() if k not in ("bends", "wrist") and not k.endswith("_series")}
        kept["holds"] = {k: v for k, v in values["holds"].items() if k != "mask"}
        out[side] = kept
    return out


__all__ = ["shake", "articulation", "preservation", "forward_kinematics", "position_jitter_mm", "rotation_jitter_degrees",
           "hf_share", "lowpass", "strip_arrays", "geodesic_degrees", "bone_index", "JITTER_HZ"]
