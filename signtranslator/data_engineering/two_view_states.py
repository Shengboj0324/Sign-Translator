"""Session pipeline: synchronise views, self-calibrate, triangulate, gate, convert.

Implements the decision in ``docs/DATA_QUALITY_DECISION.md``. For one How2Sign
recording session (one source video, two cameras), Sapiens keypoints from the frontal
and lateral clips are synchronised, the camera pair is self-calibrated with the
bone-constrained bundle adjustment, every clip is triangulated, and canonical states
are built by the same swing IK as the monocular path. A session is admitted only if
it passes the declared gate; the gate's thresholds were fixed before any real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sapiens_keypoints import HAND21, to_hand21, to_pose33
from .two_view_geometry import Calibration, TwoViewObservations, bone_length_cv, calibrate, triangulate_session, view_offset
from ..pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from ..pose.rotations import matrix_to_rotation_6d
from .video_landmarks import BODY_LABELS, Y_UP, finger_joints

CONFIDENCE = .3
MAX_REPROJECTION_PX = 2.
MIN_SYNC_CORRELATION = .8
MAX_LAG_FRAMES = 15

# Rigid segments in Goliath indices: arms, shoulder and hip widths, legs, finger phalanges.
_BODY_BONES = ((5, 7), (7, 62), (6, 8), (8, 41), (5, 6), (9, 10), (9, 11), (11, 13), (10, 12), (12, 14))


def rigid_bones() -> tuple[tuple[int, int], ...]:
    bones = list(_BODY_BONES)
    for side in ("Left", "Right"):
        hand = HAND21[side]
        for finger in range(5):
            chain = hand[1 + 4 * finger: 5 + 4 * finger]
            bones += list(zip(chain, chain[1:]))
    return tuple(bones)


def motion_signal(points: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Mean vertical image velocity of the wrists and elbows; NaN where unconfident.

    Both cameras share the vertical direction closely, so this signal is comparable
    across the frontal and lateral views.
    """
    joints = [62, 41, 7, 8]
    y = np.where(confidence[:, joints] >= CONFIDENCE, points[:, joints, 1], np.nan)
    return np.nanmean(np.diff(y, axis=0), axis=1) if len(y) > 1 else np.zeros(0)


@dataclass(frozen=True)
class ClipKeypoints:
    name: str
    times: np.ndarray        # (T,)
    points: np.ndarray       # (T, 308, 2) image pixels
    confidence: np.ndarray   # (T, 308)


def synchronise(front: ClipKeypoints, side: ClipKeypoints) -> dict:
    lag, r = view_offset(motion_signal(front.points, front.confidence),
                         motion_signal(side.points, side.confidence), max_lag=MAX_LAG_FRAMES)
    return dict(lag_frames=int(lag), correlation=float(r), passed=bool(r >= MIN_SYNC_CORRELATION))


def aligned(front: ClipKeypoints, side: ClipKeypoints, lag: int):
    """Frame-aligned (front, side) arrays for ``side[t + lag] ↔ front[t]``."""
    start, stop = max(0, -lag), min(len(front.times), len(side.times) - lag)
    if stop - start < 2:
        raise ValueError("views do not overlap after alignment")
    f = slice(start, stop)
    s = slice(start + lag, stop + lag)
    return (front.times[f], front.points[f], front.confidence[f], side.points[s], side.confidence[s])


def calibrate_session(pairs, *, size=(1280, 720), max_frames: int = 3000, seed: int = 0) -> Calibration:
    """Calibrate from frame-aligned pairs [(front_pts, front_conf, side_pts, side_conf), ...]."""
    v1 = np.concatenate([p[0] for p in pairs])
    c1 = np.concatenate([p[1] for p in pairs])
    v2 = np.concatenate([p[2] for p in pairs])
    c2 = np.concatenate([p[3] for p in pairs])
    if len(v1) > max_frames:
        pick = np.sort(np.random.default_rng(seed).choice(len(v1), max_frames, replace=False))
        v1, c1, v2, c2 = v1[pick], c1[pick], v2[pick], c2[pick]
    usable = (c1 >= CONFIDENCE) & (c2 >= CONFIDENCE)
    return calibrate(TwoViewObservations(v1, v2, usable, size, size, rigid_bones()), seed=seed)


def triangulated_points(calibration: Calibration, front_pts, front_conf, side_pts, side_conf, *, size=(1280, 720)):
    usable = (front_conf >= CONFIDENCE) & (side_conf >= CONFIDENCE)
    obs = TwoViewObservations(front_pts, side_pts, usable, size, size, rigid_bones())
    return triangulate_session(calibration, obs), usable


def rotation_inputs(points3d: np.ndarray, usable: np.ndarray) -> dict:
    """Inputs for :func:`video_landmarks.solve_rotations` from camera-frame Goliath points."""
    y_up = np.where(np.isfinite(points3d), points3d, 0.) @ Y_UP.T
    world, visible = to_pose33(y_up, usable.astype(float), .5)
    hands = {side: to_hand21(y_up, usable.astype(float), side, .5) for side in ("Left", "Right")}
    return dict(world=world, visible=visible, hands=hands)


def symmetry_error(points3d: np.ndarray) -> float:
    """Median relative left/right difference of upper-arm and forearm lengths."""
    def length(a, b):
        return np.linalg.norm(points3d[:, a] - points3d[:, b], axis=1)
    ratios = []
    for (la, lb), (ra, rb) in (((5, 7), (6, 8)), ((7, 62), (8, 41))):
        left, right = length(la, lb), length(ra, rb)
        ok = np.isfinite(left) & np.isfinite(right) & (left + right > 0)
        if ok.any():
            ratios.append(np.median(np.abs(left[ok] - right[ok]) / ((left[ok] + right[ok]) / 2)))
    return float(np.median(ratios)) if ratios else float("nan")


def session_gate(calibration: Calibration, sync: list[dict], triangulated_cv: float,
                 monocular_cv: float | None) -> dict:
    reasons = []
    if not calibration.converged:
        reasons.append("bundle adjustment did not converge")
    if max(calibration.median_reprojection_px) > MAX_REPROJECTION_PX:
        reasons.append(f"median reprojection {max(calibration.median_reprojection_px):.2f} px > {MAX_REPROJECTION_PX}")
    synced = [s for s in sync if s["passed"]]
    if not synced:
        reasons.append("no clip pair passed the synchronisation threshold")
    lags = {s["lag_frames"] for s in synced}
    if len(lags) > 1:
        reasons.append(f"inconsistent view lags within the session: {sorted(lags)}")
    if monocular_cv is not None and not triangulated_cv < monocular_cv:
        reasons.append(f"triangulated bone-length CV {triangulated_cv:.4f} is not below monocular {monocular_cv:.4f}")
    return dict(passed=not reasons, reasons=reasons, thresholds=dict(
        max_reprojection_px=MAX_REPROJECTION_PX, min_sync_correlation=MIN_SYNC_CORRELATION,
        confidence=CONFIDENCE, quality="triangulated bone-length CV below monocular"))


def monocular_bone_cv(pose_world: np.ndarray, visibility: np.ndarray) -> float:
    """Bone-length CV of MediaPipe world landmarks on the arm bones, for comparison."""
    pts = np.where((visibility >= .5)[..., None], pose_world, np.nan)
    return bone_length_cv(pts, ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 24)))


def triangulated_bone_cv(points3d: np.ndarray) -> float:
    return bone_length_cv(points3d, ((5, 7), (7, 62), (6, 8), (8, 41), (5, 6), (9, 10)))


def triangulated_state(monocular: MultichannelMotion, frames: slice, rotations: dict, hand_rotations: dict, *,
                       source_id: str, source_sha256: str, inference_method: str) -> MultichannelMotion:
    """Body and hands from triangulation; face, head and blink from the monocular state.

    Both are indexed by the frontal camera's frames, so ``frames`` (the aligned span)
    puts them on one clock. Channels neither source provides stay unavailable.
    """
    import torch
    times = monocular.timestamps[frames]
    count = len(times)

    def rotation_channel(names, table, labels):
        mats = np.stack([table[n][0] for n in names], axis=1)
        valid = np.stack([table[n][1] for n in names], axis=1)
        if mats.shape[0] != count:
            raise ValueError("triangulated rotations must cover the aligned frames")
        values = matrix_to_rotation_6d(torch.from_numpy(np.ascontiguousarray(mats))).numpy()
        values = np.where(valid[..., None], values, 0.)
        return MotionChannel(values=values, valid=valid, observed=np.zeros_like(valid),
                             confidence=np.where(valid, 1., 0.), labels=tuple(labels),
                             coordinate_frame="two-view-triangulated-camera1-y-up", units="unitless",
                             source_id=source_id, source_sha256=source_sha256,
                             convention="sapiens-two-view-swing-ik-y-up-v1",
                             inference_method=inference_method if valid.any() else None)

    channels = {"body": rotation_channel(BODY_LABELS, rotations, BODY_LABELS)}
    for side, name in (("Left", "left_hand"), ("Right", "right_hand")):
        labels = tuple(j.name for j in finger_joints(side))
        channels[name] = rotation_channel(labels, hand_rotations[side], labels)
    for name in CHANNELS:
        if name in channels:
            continue
        source = monocular.channels[name]
        if name in ("face", "blink", "head_rotation") and source.valid.any():
            channels[name] = MotionChannel(
                values=source.values[frames], valid=source.valid[frames], observed=source.observed[frames],
                confidence=source.confidence[frames], labels=source.labels, coordinate_frame=source.coordinate_frame,
                units=source.units, source_id=source.source_id, source_sha256=source.source_sha256,
                convention=source.convention,
                inference_method=source.inference_method if source.valid[frames].any() else None)
        else:
            _, width, units = CHANNELS[name]
            channels[name] = MotionChannel(
                values=np.zeros((count, 1, width)), valid=np.zeros((count, 1), dtype=bool),
                observed=np.zeros((count, 1), dtype=bool), confidence=np.zeros((count, 1)),
                labels=(f"{name}-unavailable",), coordinate_frame="not-estimated", units=units,
                source_id=source_id, source_sha256=source_sha256, convention="sapiens-two-view-swing-ik-y-up-v1")
    return MultichannelMotion(np.asarray(times, dtype=np.float64), monocular.clock_id,
                              monocular.sample_id + ":two-view", channels)


__all__ = ["triangulated_state", "rigid_bones", "motion_signal", "ClipKeypoints", "synchronise", "aligned", "calibrate_session",
           "triangulated_points", "rotation_inputs", "symmetry_error", "session_gate", "monocular_bone_cv",
           "triangulated_bone_cv", "CONFIDENCE", "MAX_REPROJECTION_PX", "MIN_SYNC_CORRELATION"]
