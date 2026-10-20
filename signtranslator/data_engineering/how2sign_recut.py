"""Recover How2Sign take-5 sessions by re-cutting lateral keypoints from the raw recordings.

In take-5 sessions the released lateral clips are mis-cut (audit B80): each holds ¾
of its frontal clip's frames and shows another moment, so they cannot be synchronised.
The raw 30 fps recordings of both cameras are intact, and the released frontal clips
are exact runs of raw frontal frames: clip frame ``m`` is raw frame ``start + m`` with
``start`` ≈ 40 × the clip's START in the original CSV (validated frame by frame on
``a4uz0W33REs-5``; the release cut the lateral clips at 30 × START instead).

The recovery, per session:

1. place every released frontal clip in the raw frontal recording by frame
   differences of small grey thumbnails, searching around the nominal start
   (:func:`place_clips`);
2. run Sapiens on the raw lateral recording (the caller does this);
3. find the camera offset: for each lag, frontal clip frame ``m`` is paired with raw
   lateral frame ``start + m + lag`` and the median Sampson distance of a robust
   fundamental matrix over the pooled clips is scored, as in
   :func:`two_view_states.epipolar_synchronise`. The integer minimum must be sharp
   (below ``SYNC_SHARPNESS`` × every lag ≥ ``FAR_LAG`` frames away, lag 0 included) and
   lie inside the search range; a 0.25-frame grid around it, with the lateral keypoints
   linearly interpolated, and a least-squares parabola give the sub-frame offset, whose
   median epipolar error must not exceed ``MAX_SAMPSON_PX``;
4. re-cut: lateral keypoints at raw frame ``start + m + offset`` for each frontal frame.

Validated on ``a4uz0W33REs-5`` before this module existed: offset 9.35 frames, constant
over eight windows (SD 0.39 frames), epipolar error 4.1–5.9 px against 13.8 px at lag 0.
"""

from __future__ import annotations

import math

import numpy as np

from .two_view_geometry import ransac_fundamental, sampson
from .two_view_states import SYNC_CONFIDENCE, SYNC_JOINTS, SYNC_SHARPNESS, MIN_SYNC_OBSERVATIONS, ClipKeypoints, in_image

RAW_FPS = 30.
FRONT_FRAMES_PER_UNIT = 40.      # released frontal clip = raw frontal frames from ~40 × START (original CSV)
PLACEMENT_WINDOW = 8             # frames searched either side of the nominal start
PLACEMENT_SHARPNESS = .9         # unique placement: best error < 0.9x every other start's (a4uz0W33REs-5:
                                 # moving clips 0.40-0.87, still clips 0.99-1.00)
MATCH_RATIO = 1.25               # a clip's best error may exceed the session's placed-clip median by 25 %
MAX_OFFSET_FRAMES = 60           # camera offsets searched: ±2 s at 30 fps
FINE_STEP = .25                  # sub-frame grid (frames) around the integer minimum
FINE_HALF_WIDTH = 1.5
MAX_SAMPSON_PX = 8.               # accepted offset's median epipolar error: synchronised How2Sign sessions
                                 # ~4 px, a4uz0W33REs-5 4.1-5.9 px; misaligned views 12-28 px (B80)
FAR_LAG = 3                      # sharpness is judged against lags >= 3 frames from the integer minimum:
                                 # the true offset is fractional, so a lag 2 away can be 1.5 frames from it
                                 # (a4uz0W33REs-5: ratio 0.76 against lags 2 away, 0.63 against 3 away)
THUMBNAIL = (64, 36)             # (width, height) grey thumbnails for placement


def nominal_start(start: float) -> int:
    """Raw frontal frame index where a released frontal clip nominally starts."""
    return int(math.ceil(FRONT_FRAMES_PER_UNIT * start - 1e-6))


def _placement_errors(raw: np.ndarray, clip: np.ndarray, nominal: int, window: int) -> dict:
    return {s: float(np.mean((raw[s:s + len(clip)] - clip) ** 2))
            for s in range(nominal - window, nominal + window + 1) if s >= 0 and s + len(clip) <= len(raw)}


def place_clips(raw: np.ndarray, clips: dict, nominals: dict, *, window: int = PLACEMENT_WINDOW) -> dict:
    """Start of each released frontal clip in its raw recording, from grey thumbnails.

    ``raw`` is (N, h, w); ``clips`` maps a name to (n, h, w); ``nominals`` to its
    :func:`nominal_start`. Every start within ``window`` frames of the nominal one is
    scored by the mean squared difference over all the clip's frames. Re-encoding
    leaves a noise floor (the matching start's error); a 1-frame shift adds motion
    differences on top. A clip is

    - ``placed`` at its best start when that start is unique (error below
      ``PLACEMENT_SHARPNESS`` × every other candidate's);
    - ``still`` at its nominal start when its frames are too still to tell neighbouring
      starts apart but the nominal start is as good as the best (within 2 %): the
      keypoints cannot then depend on which of those starts is used;
    - refused otherwise, and also when its best error is above ``MATCH_RATIO`` × the
      session's median for placed clips (it does not show this recording).
    """
    raw = np.asarray(raw, np.float32)
    errors = {name: _placement_errors(raw, np.asarray(clip, np.float32), nominals[name], window)
              for name, clip in clips.items()}
    out = {}
    for name, errs in errors.items():
        if not errs:
            out[name] = dict(start=None, nominal=nominals[name], status="refused", error=None, runner_up=None,
                             reason="the clip does not fit in the raw recording near its nominal start")
            continue
        best = min(errs, key=errs.get)
        others = [v for k, v in errs.items() if k != best]
        runner_up = min(others) if others else None
        unique = runner_up is not None and errs[best] < PLACEMENT_SHARPNESS * runner_up
        out[name] = dict(start=int(best), nominal=nominals[name], status="placed" if unique else None,
                         error=errs[best], runner_up=runner_up, reason=None)
    floors = [p["error"] for p in out.values() if p["status"] == "placed"]
    floor = float(np.median(floors)) if floors else None
    for name, placement in out.items():
        if placement["error"] is None:
            continue
        if floor is None or placement["error"] > MATCH_RATIO * floor:
            placement.update(status="refused", start=None, reason="the clip does not match the raw recording"
                             if floor is not None else "no clip of the session could be placed to the frame")
        elif placement["status"] is None:
            nominal_error = errors[name].get(placement["nominal"])
            if nominal_error is not None and nominal_error <= 1.02 * placement["error"]:
                placement.update(status="still", start=placement["nominal"])
            else:
                placement.update(status="refused", start=None,
                                 reason="no unique frame placement and the nominal start is not among the best")
    return out


def sample(points: np.ndarray, confidence: np.ndarray, positions: np.ndarray):
    """Keypoints at fractional frame ``positions`` by linear interpolation of neighbouring frames.

    Confidence is the lower of the two neighbours'; positions outside the recording get
    confidence 0 and point (0, 0).
    """
    positions = np.asarray(positions, dtype=np.float64)
    count = len(points)
    inside = (positions >= 0) & (positions <= count - 1)
    clipped = np.clip(positions, 0, count - 1)
    low = np.minimum(np.floor(clipped).astype(int), count - 1)
    high = np.minimum(low + 1, count - 1)
    weight = (clipped - low)[:, None, None]
    out = (1 - weight) * points[low] + weight * points[high]
    conf = np.minimum(confidence[low], confidence[high]).astype(np.float64)
    out[~inside], conf[~inside] = 0., 0.
    return out, conf


def correspondences(placed, side_points, side_confidence, lag: float, *, joints=SYNC_JOINTS, size=(1280, 720)):
    """Per clip, confident pairs (front frame m of SYNC joints ↔ raw side at start + m + lag)."""
    joints = list(joints)
    out = []
    for start, front in placed:
        positions = start + np.arange(len(front.times)) + lag
        sp, sc = sample(side_points[:, joints], side_confidence[:, joints], positions)
        fp, fc = front.points[:, joints], front.confidence[:, joints]
        ok = (fc >= SYNC_CONFIDENCE) & (sc >= SYNC_CONFIDENCE) & in_image(fp, size) & in_image(sp, size)
        out.append((fp[ok], sp[ok]))
    return out


def _score(placed, side_points, side_confidence, lag, *, size, seed):
    pairs = correspondences(placed, side_points, side_confidence, lag, size=size)
    x1 = np.concatenate([p[0] for p in pairs]) if pairs else np.zeros((0, 2))
    x2 = np.concatenate([p[1] for p in pairs]) if pairs else np.zeros((0, 2))
    if len(x1) < MIN_SYNC_OBSERVATIONS:
        return None, None, pairs
    f, _ = ransac_fundamental(x1, x2, threshold=3., iterations=500, seed=seed)
    return float(np.median(sampson(f, x1, x2))), f, pairs


def estimate_offset(placed, side_points: np.ndarray, side_confidence: np.ndarray, *,
                    max_offset: int = MAX_OFFSET_FRAMES, size=(1280, 720), seed: int = 0) -> dict:
    """Sub-frame camera offset of a session from placed frontal clips and raw lateral keypoints.

    ``placed`` is [(raw start, frontal ClipKeypoints), ...]; ``side_points`` (N, 308, 2)
    and ``side_confidence`` (N, 308) cover the raw lateral recording (confidence 0 where
    not computed). Raw lateral frame ``start + m + offset`` shows frontal clip frame ``m``.
    """
    curve = {}
    for lag in range(-max_offset, max_offset + 1):
        error, _, _ = _score(placed, side_points, side_confidence, lag, size=size, seed=seed)
        if error is not None:
            curve[lag] = error
    result = dict(offset_frames=None, passed=False, curve=curve, fine_curve={}, sampson_px=None, sharpness=None,
                  zero_lag_ratio=None, clips=[])
    if not curve:
        return dict(result, reason="too few confident correspondences")
    best = min(curve, key=curve.get)
    far = [v for k, v in curve.items() if abs(k - best) >= FAR_LAG]
    result.update(integer_offset=int(best), sharpness=curve[best] / min(far) if far else None,
                  zero_lag_ratio=curve[best] / curve[0] if 0 in curve else None)
    if not far or curve[best] >= SYNC_SHARPNESS * min(far):
        return dict(result, reason=f"the epipolar error has no sharp minimum over offsets (ratio {result['sharpness']})")
    if abs(best) >= 2 and not (0 in curve and curve[best] < SYNC_SHARPNESS * curve[0]):
        return dict(result, reason="the epipolar minimum is not clearly below lag 0")
    if abs(best) >= max_offset:
        return dict(result, reason="the epipolar minimum lies at the edge of the searched offsets")
    lags = best + np.arange(-FINE_HALF_WIDTH, FINE_HALF_WIDTH + FINE_STEP / 2, FINE_STEP)
    fine = {}
    for lag in lags:
        error, _, _ = _score(placed, side_points, side_confidence, float(lag), size=size, seed=seed)
        if error is not None:
            fine[float(lag)] = error
    # The fine curve carries RANSAC noise of a few tenths of a pixel, so its minimum is the
    # vertex of a least-squares parabola over the whole grid rather than of three points.
    keys = np.array(sorted(fine))
    values = np.array([fine[k] for k in keys])
    offset = float(keys[np.argmin(values)])
    if len(keys) >= 3:
        a, b, _ = np.polyfit(keys - best, values, 2)
        if a > 0:
            offset = float(best + np.clip(-b / (2 * a), -FINE_HALF_WIDTH, FINE_HALF_WIDTH))
    error, f, pairs = _score(placed, side_points, side_confidence, offset, size=size, seed=seed)
    if error is None or error > MAX_SAMPSON_PX:
        return dict(result, fine_curve={round(k, 4): v for k, v in fine.items()}, sampson_px=error,
                    reason=f"the epipolar error at the best offset is {error} px, above {MAX_SAMPSON_PX} px")
    clips = [dict(start=int(start), observations=int(len(x1)),
                  sampson_px=float(np.median(sampson(f, x1, x2))) if len(x1) else None)
             for (start, _), (x1, x2) in zip(placed, pairs)]
    return dict(result, offset_frames=float(offset), passed=True, reason=None, sampson_px=error,
                fine_curve={round(k, 4): v for k, v in fine.items()}, clips=clips)


def recut(front: ClipKeypoints, start: int, offset: float, side_points: np.ndarray, side_confidence: np.ndarray,
          name: str) -> ClipKeypoints:
    """Lateral keypoints on the frontal clip's frames: raw lateral frame ``start + m + offset``."""
    points, confidence = sample(side_points, side_confidence, start + np.arange(len(front.times)) + offset)
    return ClipKeypoints(name, np.array(front.times, dtype=np.float64), points, confidence)


def needed_frames(placed_spans, count: int, *, max_offset: int = MAX_OFFSET_FRAMES) -> tuple[int, int]:
    """Contiguous raw lateral frame range [first, stop) the offset search and re-cut can touch."""
    first = min(s for s, n in placed_spans) - max_offset - 2
    stop = max(s + n for s, n in placed_spans) + max_offset + 2
    return max(0, first), min(count, stop)


__all__ = ["RAW_FPS", "FRONT_FRAMES_PER_UNIT", "PLACEMENT_WINDOW", "PLACEMENT_SHARPNESS", "MATCH_RATIO", "MAX_OFFSET_FRAMES",
           "FINE_STEP", "FAR_LAG", "MAX_SAMPSON_PX", "THUMBNAIL", "nominal_start", "place_clips", "sample", "correspondences", "estimate_offset",
           "recut", "needed_frames"]
