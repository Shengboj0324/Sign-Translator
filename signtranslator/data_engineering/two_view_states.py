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
from .skeleton_fit import Skeleton, fit_skeleton
from .two_view_geometry import (
    Calibration, TwoViewObservations, bone_length_cv, calibrate, calibrate_jointly, ransac_fundamental, sampson,
    triangulate_session,
    view_offset,
)
from ..pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from ..pose.rotations import matrix_to_rotation_6d
from .video_landmarks import BODY_LABELS, Y_UP, finger_joints

CONFIDENCE = .3
MAX_REPROJECTION_PX = 2.
MIN_SYNC_CORRELATION = .8
CALIBRATION_ANCHORS = (0,)    # nose: on no bone but view-consistent (1.1 px median two-view error on Panoptic)
MAX_LAG_FRAMES = 30          # ±1 s at How2Sign's 30 fps (15 left 18% of sessions on the search edge)
MAX_SYNC_POINTS = 20_000     # correspondences per candidate lag used to fit and score the epipolar curve
# Epipolar synchronisation and calibration seeding (decided on the first real How2Sign
# session, before any training; docs/DATA_QUALITY_DECISION.md §7). Nose, shoulders,
# elbows and wrists are the view-consistent keypoints on How2Sign (0.75-2 px median
# Sampson distance; hand keypoints in the profile view ~4.8 px), so they seed the
# calibration. Timing needs fast motion, which the hands carry: body+hands give a sharp
# single minimum over lags (ratio 0.54 to lags >= 2 frames away), body alone a flat one (0.88).
CALIBRATION_SEED_JOINTS = (0, 5, 6, 7, 8, 62, 41)
SYNC_JOINTS = CALIBRATION_SEED_JOINTS + HAND21["Left"] + HAND21["Right"]
SYNC_CONFIDENCE = .5
MIN_SYNC_OBSERVATIONS = 200        # pooled correspondences needed to decide a session's lag
SYNC_SHARPNESS = .8                # best lag's error must be < 0.8x that of every lag >= 2 frames away
MAX_CLIP_EPIPOLAR_RATIO = 2.       # a clip whose error at the session lag exceeds 2x the session's is dropped

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


def in_image(points: np.ndarray, size) -> np.ndarray:
    """Whether each (…, 2) keypoint lies inside the image; outside points are extrapolations."""
    width, height = size
    return (points[..., 0] >= 0) & (points[..., 0] < width) & (points[..., 1] >= 0) & (points[..., 1] < height)


def _shifted(front: ClipKeypoints, side: ClipKeypoints, lag: int, joints, size):
    """Correspondences ``front[t] ↔ side[t + lag]`` of ``joints`` seen confidently in both views."""
    start, stop = max(0, -lag), min(len(front.points), len(side.points) - lag)
    if stop - start < 1:
        return np.zeros((0, 2)), np.zeros((0, 2))
    fp, sp = front.points[start:stop][:, joints], side.points[start + lag:stop + lag][:, joints]
    ok = ((front.confidence[start:stop][:, joints] >= SYNC_CONFIDENCE) &
          (side.confidence[start + lag:stop + lag][:, joints] >= SYNC_CONFIDENCE) &
          in_image(fp, size) & in_image(sp, size))
    return fp[ok], sp[ok]


def time_scale(pairs, *, min_frames: int = 24, tolerance: float = .02) -> float:
    """Session ratio ``lateral frames / frontal frames`` (1 when equal within ``tolerance``).

    How2Sign take-5 sessions are defective in the release: every lateral clip holds ¾ of
    the frontal clip's frames (sampled: 84 of 85 take-5 sessions, ratio 0.748–0.751; all
    155 sampled sessions of other takes, 1.00) and shows a different moment of the
    recording — no time scale, offset, or linear placement on the recording aligns them
    (measured: the epipolar error stays at 15–16 px for every candidate, against ~4 px
    for synchronised sessions). The ratio is constant within a session and is the
    median over its clips of at least ``min_frames`` frames.
    """
    ratios = [len(side.times) / len(front.times) for _, front, side in pairs if len(front.times) >= min_frames]
    if not ratios:
        return 1.
    scale = float(np.median(ratios))
    return 1. if abs(scale - 1.) < tolerance else scale


def epipolar_synchronise(pairs, *, size=(1280, 720), max_lag: int = MAX_LAG_FRAMES, seed: int = 0) -> dict:
    """One view lag for a whole session, from epipolar consistency of view-consistent joints.

    ``pairs`` is [(sentence, front, side), ...] from one recording session: both clips of
    a sentence were cut from the same two continuous recordings, so the lag is a property
    of the session. For each candidate integer lag the clips' correspondences are pooled,
    a fundamental matrix is fitted robustly and scored by the median Sampson distance;
    in time-misaligned frames every moving joint violates the epipolar constraint, so the
    true lag minimises it. The lag is accepted only if the minimum is sharp and lies
    inside the searched range (a minimum on the edge means the true lag is outside it);
    then each clip is kept unless its own error at that lag exceeds
    ``MAX_CLIP_EPIPOLAR_RATIO`` times the session's (a clip cut differently from the rest).
    Each lag's curve value uses at most ``MAX_SYNC_POINTS`` correspondences drawn with a
    fixed seed, so the median Sampson distance is estimated, not exhausted.
    """
    joints = list(SYNC_JOINTS)
    curve, fits = {}, {}
    for lag in range(-max_lag, max_lag + 1):
        gathered = [_shifted(f, s, lag, joints, size) for _, f, s in pairs]
        x1 = np.concatenate([g[0] for g in gathered])
        x2 = np.concatenate([g[1] for g in gathered])
        if len(x1) < MIN_SYNC_OBSERVATIONS:
            continue
        if len(x1) > MAX_SYNC_POINTS:
            pick = np.random.default_rng(seed + lag + max_lag).choice(len(x1), MAX_SYNC_POINTS, replace=False)
            x1, x2 = x1[pick], x2[pick]
        f, _ = ransac_fundamental(x1, x2, threshold=3., iterations=500, seed=seed)
        curve[lag], fits[lag] = float(np.median(sampson(f, x1, x2))), f
    if not curve:
        return dict(lag_frames=None, passed=False, reason="too few confident correspondences", curve={}, clips=[])
    lag = min(curve, key=curve.get)
    far = [v for k, v in curve.items() if abs(k - lag) >= 2]
    edge = abs(lag) >= max_lag
    sharp = bool(far) and curve[lag] < SYNC_SHARPNESS * min(far) and not edge
    clips = []
    for sentence, front, side in pairs:
        x1, x2 = _shifted(front, side, lag, joints, size)
        error = float(np.median(sampson(fits[lag], x1, x2))) if len(x1) else float("inf")
        clips.append(dict(sentence=sentence, sampson_px=error, observations=int(len(x1)),
                          passed=bool(sharp and error <= MAX_CLIP_EPIPOLAR_RATIO * curve[lag])))
    return dict(lag_frames=int(lag), session_sampson_px=curve[lag], sharp=sharp, passed=sharp,
                reason=None if sharp else ("the epipolar minimum lies on the edge of the searched lags" if edge
                                           else "the epipolar error has no sharp minimum over lags"),
                curve={int(k): v for k, v in curve.items()}, clips=clips)


def aligned(front: ClipKeypoints, side: ClipKeypoints, lag: int):
    """Frame-aligned (front, side) arrays for ``side[t + lag] ↔ front[t]``."""
    start, stop = max(0, -lag), min(len(front.times), len(side.times) - lag)
    if stop - start < 2:
        raise ValueError("views do not overlap after alignment")
    f = slice(start, stop)
    s = slice(start + lag, stop + lag)
    return (front.times[f], front.points[f], front.confidence[f], side.points[s], side.confidence[s])


def session_observations(pairs, *, size=(1280, 720), max_frames: int = 3000, seed: int = 0) -> TwoViewObservations:
    """Pooled observations of frame-aligned pairs [(front_pts, front_conf, side_pts, side_conf), ...]."""
    v1 = np.concatenate([p[0] for p in pairs])
    c1 = np.concatenate([p[1] for p in pairs])
    v2 = np.concatenate([p[2] for p in pairs])
    c2 = np.concatenate([p[3] for p in pairs])
    if len(v1) > max_frames:
        pick = np.sort(np.random.default_rng(seed).choice(len(v1), max_frames, replace=False))
        v1, c1, v2, c2 = v1[pick], c1[pick], v2[pick], c2[pick]
    usable = (c1 >= CONFIDENCE) & (c2 >= CONFIDENCE) & in_image(v1, size) & in_image(v2, size)
    return TwoViewObservations(v1, v2, usable, size, size, rigid_bones())


# Radial distortion is off by default: with the signer in the image centre the
# division-model coefficient trades off against the focal lengths (on undistorted
# synthetic sessions the joint fit invents λ ≈ -0.27 and focal lengths 7% low), and
# How2Sign's lenses are unknown. It helped on Panoptic only because those lenses are
# strongly distorted (k1 ≈ -0.26); see docs/DATA_QUALITY_DECISION.md §7.
DISTORTION = False


def calibrate_session(pairs, *, size=(1280, 720), max_frames: int = 3000, seed: int = 0,
                      distortion: bool = DISTORTION, intrinsics=None) -> Calibration:
    """Calibrate from frame-aligned pairs [(front_pts, front_conf, side_pts, side_conf), ...].

    With ``intrinsics`` (f1, f2, λ1, λ2) shared by the cameras — from
    :func:`calibrate_cameras` over many sessions — only the session's pose is fitted.
    """
    obs = session_observations(pairs, size=size, max_frames=max_frames, seed=seed)
    return calibrate(obs, seed=seed, distortion=distortion, anchors=CALIBRATION_ANCHORS,
                     ransac_joints=CALIBRATION_SEED_JOINTS, intrinsics=intrinsics)


def calibrate_cameras(sessions, *, size=(1280, 720), max_frames: int = 600, seed: int = 0,
                      distortion: bool = DISTORTION) -> list[Calibration]:
    """Joint calibration of many sessions (each a list of aligned pairs) with shared intrinsics."""
    observations = [session_observations(pairs, size=size, max_frames=max_frames, seed=seed) for pairs in sessions]
    return calibrate_jointly(observations, seed=seed, distortion=distortion, anchors=CALIBRATION_ANCHORS,
                             ransac_joints=CALIBRATION_SEED_JOINTS)


def triangulated_points(calibration: Calibration, front_pts, front_conf, side_pts, side_conf, *, size=(1280, 720)):
    usable = ((front_conf >= CONFIDENCE) & (side_conf >= CONFIDENCE) &
              in_image(front_pts, size) & in_image(side_pts, size))
    obs = TwoViewObservations(front_pts, side_pts, usable, size, size, rigid_bones())
    return triangulate_session(calibration, obs), usable


HAND_SMOOTHNESS = 20.    # tuned on CMU Panoptic 171026_pose3; see docs/DATA_QUALITY_DECISION.md
HAND_HINGE_WEIGHT = 20.  # coplanar-phalanx prior, tuned on 171026_pose3 (plateau 20-100); DATA_QUALITY_DECISION §6


def finger_hinges(side: str) -> tuple[tuple[int, int, int], ...]:
    """Per finger, the child joints of its last three bones (1-DOF hinged phalanges)."""
    hand = HAND21[side]
    return tuple(tuple(hand[2 + 4 * finger: 5 + 4 * finger]) for finger in range(5))


def hand_skeleton(side: str) -> Skeleton:
    """Wrist-rooted finger chains of one hand, in Goliath indices."""
    hand = HAND21[side]
    joints, parents = [hand[0]], [-1]
    for finger in range(5):
        chain = (hand[0], *hand[1 + 4 * finger: 5 + 4 * finger])
        for a, b in zip(chain, chain[1:]):
            joints.append(b)
            parents.append(joints.index(a))
    return Skeleton(tuple(joints), tuple(parents))


def refine_hands(points3d: np.ndarray, view1: np.ndarray, view2: np.ndarray, conf1: np.ndarray, conf2: np.ndarray,
                 p1: np.ndarray, p2: np.ndarray, *, smoothness: float = HAND_SMOOTHNESS,
                 hinge_weight: float = HAND_HINGE_WEIGHT) -> np.ndarray:
    """Finger joints re-estimated by a rigid-bone, temporally smooth two-view hand fit.

    Arms stay as triangulated (fitting them as well cost accuracy on Panoptic); each
    hand's fitted finger offsets are re-attached to the triangulated wrist, so only
    finger directions change. Views must be undistorted pixels for the pinhole
    ``p1``/``p2``. Joints not triangulated in a frame stay NaN.
    """
    out = points3d.copy()
    for side in ("Left", "Right"):
        skeleton = hand_skeleton(side)
        try:
            fitted, _ = fit_skeleton(points3d, view1, view2, conf1, conf2, p1, p2, skeleton, smoothness=smoothness,
                                     hinges=finger_hinges(side), hinge_weight=hinge_weight)
        except ValueError:
            continue                                   # this hand is never triangulated
        wrist, fingers = skeleton.joints[0], list(skeleton.joints[1:])
        refined = fitted[:, fingers] - fitted[:, [wrist]] + points3d[:, [wrist]]
        keep = np.isfinite(points3d[:, fingers]).all(-1, keepdims=True)
        out[:, fingers] = np.where(keep, refined, np.nan)
    return out


def refined_points(calibration: Calibration, points3d, usable, front_pts, front_conf, side_pts, side_conf, *,
                   size=(1280, 720), smoothness: float = HAND_SMOOTHNESS) -> np.ndarray:
    """:func:`refine_hands` for a self-calibrated session clip."""
    view1, view2 = calibration.undistort(front_pts, side_pts, size, size)
    p1, p2 = calibration.matrices(size, size)
    return refine_hands(points3d, view1, view2, front_conf * usable, side_conf * usable, p1, p2, smoothness=smoothness)


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
        max_reprojection_px=MAX_REPROJECTION_PX, sync="epipolar, session lag", sync_sharpness=SYNC_SHARPNESS,
        max_clip_epipolar_ratio=MAX_CLIP_EPIPOLAR_RATIO, confidence=CONFIDENCE,
        quality="triangulated bone-length CV below monocular"))


def monocular_bone_cv(pose_world: np.ndarray, visibility: np.ndarray) -> float:
    """Bone-length CV of MediaPipe world landmarks on the arm bones, for comparison."""
    pts = np.where((visibility >= .5)[..., None], pose_world, np.nan)
    return bone_length_cv(pts, ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 24)))


def triangulated_bone_cv(points3d: np.ndarray) -> float:
    return bone_length_cv(points3d, ((5, 7), (7, 62), (6, 8), (8, 41), (5, 6), (9, 10)))


TRANSLATION_LABELS = {"root_translation": ("torso_anchor",), "head_translation": ("cyclopean_eye",)}
TRANSLATION_FRAME = "two-view-session-world-y-up-metres"


def triangulated_state(monocular: MultichannelMotion, frames: slice, rotations: dict, hand_rotations: dict, *,
                       source_id: str, source_sha256: str, inference_method: str,
                       translations: dict | None = None, gaze_model=None,
                       gaze_method: str | None = None) -> MultichannelMotion:
    """Body and hands from triangulation; face, head and blink from the monocular state.

    Both are indexed by the frontal camera's frames, so ``frames`` (the aligned span)
    puts them on one clock. ``translations`` maps ``root_translation`` and
    ``head_translation`` to ((T, 3) metres, (T,) valid) in the session-world frame of
    :mod:`body_space`. With ``gaze_model`` (:class:`gaze.GazeModel`), ``left_gaze`` and
    ``right_gaze`` come from the monocular face blendshapes and head rotation on the
    same frames (``gaze_method`` is their inference method). Channels no source
    provides stay unavailable.
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
    for name, (values, valid) in (translations or {}).items():
        if name not in TRANSLATION_LABELS:
            raise ValueError(f"{name} is not a translation channel")
        values, valid = np.asarray(values, dtype=np.float64), np.asarray(valid, dtype=bool)
        if values.shape != (count, 3) or valid.shape != (count,):
            raise ValueError(f"{name} must be (T, 3) metres with a (T,) mask over the aligned frames")
        valid = valid & np.isfinite(values).all(axis=1)
        channels[name] = MotionChannel(
            values=np.where(valid[:, None], values, 0.)[:, None], valid=valid[:, None],
            observed=np.zeros((count, 1), dtype=bool), confidence=np.where(valid, 1., 0.)[:, None],
            labels=TRANSLATION_LABELS[name], coordinate_frame=TRANSLATION_FRAME, units="m",
            source_id=source_id, source_sha256=source_sha256, convention="body-space-anthropometric-scale-v1",
            inference_method=inference_method if valid.any() else None)
    if gaze_model is not None:
        face, head = monocular.channels["face"], monocular.channels["head_rotation"]
        from .gaze import FEATURES, SIDES as EYES, gaze_channels
        named = all(f + side in face.labels for f in FEATURES for side in EYES)
        if named and face.valid[frames].any() and head.valid[frames].any():
            from dataclasses import replace
            window = lambda c: replace(c, values=c.values[frames], valid=c.valid[frames], observed=c.observed[frames],
                                       confidence=c.confidence[frames])
            channels.update(gaze_channels(window(face), window(head), gaze_model, source_id=source_id,
                                          source_sha256=source_sha256,
                                          inference_method=gaze_method or "calibrated MediaPipe eye blendshapes"))
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


__all__ = ["triangulated_state", "TRANSLATION_LABELS", "TRANSLATION_FRAME", "rigid_bones", "motion_signal", "ClipKeypoints", "synchronise", "aligned", "calibrate_session",
           "epipolar_synchronise", "in_image", "SYNC_JOINTS", "CALIBRATION_SEED_JOINTS", "session_observations",
           "time_scale",
           "calibrate_cameras",
           "triangulated_points", "rotation_inputs", "symmetry_error", "session_gate", "monocular_bone_cv",
           "triangulated_bone_cv", "CONFIDENCE", "MAX_REPROJECTION_PX", "MIN_SYNC_CORRELATION", "CALIBRATION_ANCHORS",
           "HAND_SMOOTHNESS", "HAND_HINGE_WEIGHT", "finger_hinges", "hand_skeleton", "refine_hands", "refined_points"]
