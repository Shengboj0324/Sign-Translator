"""Paired validation of the UWB lifter: UWB-lifted states against our own Tasks-extracted states of the same cues.

For YouTube-ASL videos that we extracted ourselves (``<youtube prefix>/videos/<id>.json``: caption cues with start,
end, text and state key) **and** that the UWB keypoint dataset also holds (clip ids ``<id>.<first>-<last>``):

1. **Frame rate, per video.** UWB clip names are source frame ranges at an unpublished rate. UWB clips and our cues
   come from the same caption track, so cues whose normalised text matches exactly once on both sides give rows
   ``(start, first)`` and ``(end, last)``. A Theil-Sen line (median of pairwise slopes, robust to mismatched cues)
   starts the fit; rows within 3 frames of it are refitted by least squares (``frame = fps * t + c``), and the rate is
   snapped to a standard rate within 0.03% (``STANDARD_FPS``). The estimate is refused when the offset ``c`` lies
   outside [-1, 0] frames (UWB frames are ``floor(t * fps)``: c is about -0.5) or the rate outside [10, 70] fps.
   With fewer than ``--min-text-matches`` text rows, each standard rate is scored by how many UWB clips land on a cue
   (both ends within 2 frames); the best is taken only if it places >= 50% of the video's clips and beats the
   runner-up. ``--fps-table-out`` writes the per-video rates for ``convert_uwb_clips.py``; after a full evaluation,
   videos whose lag drifts by more than 1 frame/min are dropped from it.
2. **Pairs.** Each UWB clip is paired with the cue of largest time overlap (IoU >= 0.5, one-to-one).
   ``--overlap-table-out`` (fps-only or all-video runs) writes every pair (clip -> our sample id) and every video we
   extracted with our channel-hashed partition (``build_youtube_asl_manifest.monitor_channel``), for the converter's
   dedup; videos we hold but could not time are marked ``whole-video`` (all their UWB clips count as duplicates).
3. **Temporal alignment.** UWB frame ``j`` is at ``(first + j + L) / fps`` and is matched to our nearest frame within
   half a frame. ``L`` (|L| <= ``--max-lag``) minimises the image-plane angle between UWB's raw 2D arm bones and our
   projected world arm directions (independent of the lifter); a non-zero lag must beat lag 0 by ``--lag-margin``
   degrees (the minimum is shallow). Clips without our arms keep ``L = 0`` (method ``none``). Lags are reported by
   frame-count difference (UWB clips hold ``last - first + 1`` or ``last - first`` frames) and drift per video.
4. **Errors** (degrees): upper arms and forearms (world bone directions by forward kinematics: convention-free),
   wrists (palm frame in the camera frame), fingers (bone directions in the palm frame), and arms and wrists after one
   best global rotation per clip. **The primary metric is the declared lifter on common frames**: frames held by our
   state and by every compared method. Methods: the lifter under the declared visibility rule (``--visibility``,
   required) and under each candidate rule (``--candidate-visibility``), and the geometric pseudo-world
   ``(x, y / a, 0)`` with the inside-image rule. Medians, p75, p90, means, and a 95% CI of the median from 2,000
   resamples of videos. Strata: hips visible in the UWB frame or not, fps about 30 or not, UWB stream rate equal to
   our download's or not, frame-count difference 0 / -1 / other.
5. **Validity on all frames**, stratified by framing (both hips inside the UWB frame or not): hand-valid agreement with
   ours per side (counts, agreement, Cohen's kappa, UWB-only rate) on every aligned frame; forearm survival (share of
   frames each rule keeps) on every frame; forearm agreement only where our chest is valid (our original solver needs
   both hips for the chest, so elsewhere our arms are absent by construction). Survival against MediaPipe's own
   visibility on chest-up framings, and the accuracy of what a rule keeps or drops outside the frame, are measured
   on How2Sign two-view truth by ``scripts/uwb_outside_frame_accuracy.py``.
6. **Inputs.** Every regressor input, standardised by the bundle's own statistics, on a sample of UWB frames: p99 |z|
   per input (the proposed bar is <= 4 for every input).
7. A **mirror check** (the arm error if UWB were the mirror image of ours) and, with ``--head affine``, the
   scaled-orthographic head rotation against our face-geometry head rotation.

``--exclude-videos`` (a file of video ids) keeps development videos out of the evaluation (the smoke runs' videos are
listed in ``reports/uwb_gate_smoke_videos.txt``).

**Gate v2 options** (plan "UWB lifter v2 and gate v2", 2026-10-30 11:12 UTC; without them every output is exactly as
before): ``--split-file`` (``build_uwb_lifter_split.py``; a local path, store key or ``KEY@SHA256``) restricts the
population to the split's held-out videos and refuses to evaluate any other video (the split's SHA-256 is recorded);
``--lag-curves`` stores each pair's per-lag alignment cost curve (``alignment.curve``: lag -> [median angle, bone-frames],
for the pooled-lag timing row); ``--secondary-visibility`` scores further rules (``head``) as reported methods that do
not enter the common frames. The summary carries ``proposal_check``: the coordinator's proposed
gate evaluated where this script can compute it (it is a proposal until declared). Reads only; writes the report
locally (``--out``) and optionally the fps and overlap tables.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import time
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

UWB_PREFIX = "acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1"
YOUTUBE_PREFIX = "acquisition/2026-10-17-v1/youtube-asl-v1"
STANDARD_FPS = (12., 15., 20., 23.976, 24., 25., 29.97, 30., 48., 50., 59.94, 60.)
SNAP_TOLERANCE = 3e-4
OUTLIER_FRAMES = 3.
PLAUSIBLE_FPS = (10., 70.)
OFFSET_RANGE = (-1., 0.)
GRID_MIN_SHARE = .5
DRIFT_LIMIT = 1.                   # frames per minute
DRIFT_MIN_CUES, DRIFT_MIN_SPAN = 8, 180.   # a drift is measured only from >= 8 arm-aligned cues spanning >= 3 min
ELBOW_BINS = (("inside", 0., 0.), ("0-0.1", 0., .1), ("0.1-0.3", .1, .3), ("0.3-inf", .3, np.inf))
ARM_BONES = {"LeftUpperArm": (11, 13), "LeftForearm": (13, 15), "RightUpperArm": (12, 14), "RightForearm": (14, 16)}
GROUPS = ("upper_arms", "forearms", "wrists", "fingers")
CELLS = ("both", "ours_only", "uwb_only", "neither")
Z_FRAMES = 8                       # UWB frames per pair whose standardised regressor inputs are kept
BOOTSTRAP = 2000
# The coordinator's proposed gate (reports/uwb_lifter_dev.md), checked where this script can compute it.
PROPOSAL = dict(median=dict(upper_arms=12., forearms=18., wrists=15., fingers=16.), ci_slack=1.,
                p90=dict(upper_arms=25., forearms=40.), rotation_gain=3., stratum_pairs=200, stratum_slack=3.,
                hand_agreement=.88, forearm_agreement=.85, uwb_only=.10, input_z=4., mirror_share=.03,
                lag_within_1=.90, drift_share=.95)
_WORKER: dict = {}


# ----------------------------------------------------------------------------------------------------------------------
# Frame rate and pairing

def normalise_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower()))


def snap_fps(raw: float) -> tuple[float, str | None]:
    best = min(STANDARD_FPS, key=lambda f: abs(raw / f - 1))
    return (best, f"{best:g}") if abs(raw / best - 1) <= SNAP_TOLERANCE else (raw, None)


def text_pairs(clips: list[dict], cues: list[dict]) -> np.ndarray:
    """(N, 2) rows (time, frame) from cues and clips whose normalised text occurs exactly once on each side."""
    ours = collections.defaultdict(list)
    for c in cues:
        ours[normalise_text(c["text"])].append(c)
    theirs = collections.defaultdict(list)
    for c in clips:
        theirs[normalise_text(c.get("text") or "")].append(c)
    rows = []
    for text, mine in ours.items():
        if text and len(mine) == 1 and len(theirs.get(text, ())) == 1:
            u = theirs[text][0]
            rows += [(mine[0]["start"], u["first"]), (mine[0]["end"], u["last"])]
    return np.array(rows, dtype=np.float64).reshape(-1, 2)


def theil_sen(rows: np.ndarray, *, max_pairs: int = 20000, min_dt: float = 1., seed: int = 0):
    """(slope, intercept): the median of pairwise slopes over rows at least ``min_dt`` seconds apart, and the median
    residual intercept; None when no such pair exists."""
    n = len(rows)
    i, j = np.triu_indices(n, 1)
    if len(i) > max_pairs:
        pick = np.random.default_rng(seed).choice(len(i), max_pairs, replace=False)
        i, j = i[pick], j[pick]
    dt = rows[j, 0] - rows[i, 0]
    keep = np.abs(dt) >= min_dt
    if not keep.any():
        return None
    slope = float(np.median((rows[j, 1] - rows[i, 1])[keep] / dt[keep]))
    return slope, float(np.median(rows[:, 1] - slope * rows[:, 0]))


def fit_rate(rows: np.ndarray) -> dict | None:
    """Theil-Sen start, then least squares on the rows within OUTLIER_FRAMES of the current line (at most 4 refits)."""
    start = theil_sen(rows) if len(rows) >= 4 else None
    if start is None:
        return None
    fps, offset = start
    keep = np.abs(rows[:, 1] - (fps * rows[:, 0] + offset)) <= OUTLIER_FRAMES
    for _ in range(4):
        if keep.sum() < 4 or np.ptp(rows[keep, 0]) <= 0:
            return None
        design = np.c_[rows[keep, 0], np.ones(keep.sum())]
        (fps, offset), *_ = np.linalg.lstsq(design, rows[keep, 1], rcond=None)
        residual = rows[:, 1] - (fps * rows[:, 0] + offset)
        new = np.abs(residual) <= OUTLIER_FRAMES
        if (new == keep).all():
            break
        keep = new
    r = np.abs(residual[keep])
    return dict(raw=float(fps), offset_frames=round(float(offset), 3), rows=int(keep.sum()),
                outliers=int((~keep).sum()), residual_median=round(float(np.median(r)), 3),
                residual_p90=round(float(np.percentile(r, 90)), 3), theil_sen_start=round(start[0], 4))


def grid_rate(clips: list[dict], cues: list[dict], tolerance: float = 2.) -> dict:
    """Standard rate under which the most UWB clips land on a cue (both ends within ``tolerance`` frames)."""
    starts = np.array([c["start"] for c in cues]) if cues else np.zeros(0)
    ends = np.array([c["end"] for c in cues]) if cues else np.zeros(0)
    scores = {}
    for fps in STANDARD_FPS:
        hits = 0
        for c in clips:
            hits += bool(((np.abs(c["first"] - starts * fps) <= tolerance)
                          & (np.abs(c["last"] - ends * fps) <= tolerance)).any())
        scores[fps] = hits
    order = sorted(scores, key=lambda f: -scores[f])
    return dict(best=order[0], hits=scores[order[0]], runner_up=order[1], runner_up_hits=scores[order[1]],
                clips=len(clips))


def estimate_fps(clips: list[dict], cues: list[dict], *, min_text_matches: int = 3) -> dict:
    """Per-video rate estimate: ``fps`` (snapped or raw) or None with ``refused`` saying why."""
    rows = text_pairs(clips, cues)
    out = dict(text_matches=len(rows) // 2, fps=None, method=None)
    if len(rows) // 2 >= min_text_matches:
        fit = fit_rate(rows)
        if fit is not None:
            out.update(fit, method="text-matched-fit")
            if not PLAUSIBLE_FPS[0] <= fit["raw"] <= PLAUSIBLE_FPS[1]:
                return dict(out, refused=f"implausible rate {fit['raw']:.3f}")
            if not OFFSET_RANGE[0] <= fit["offset_frames"] <= OFFSET_RANGE[1]:
                return dict(out, refused=f"offset {fit['offset_frames']} frames outside {list(OFFSET_RANGE)}")
            fps, snapped = snap_fps(fit["raw"])
            return dict(out, fps=fps, snapped_to=snapped)
    grid = grid_rate(clips, cues)
    out["grid"] = grid
    if (grid["hits"] >= max(min_text_matches, GRID_MIN_SHARE * len(clips))
            and grid["hits"] > grid["runner_up_hits"] and PLAUSIBLE_FPS[0] <= grid["best"] <= PLAUSIBLE_FPS[1]):
        out.update(fps=grid["best"], snapped_to=f"{grid['best']:g}", method="time-grid")
    else:
        out["refused"] = out.get("refused") or "too little timing evidence"
    return out


def pair_clips(clips: list[dict], cues: list[dict], fps: float, *, min_iou: float = .5) -> list[tuple[dict, dict, float]]:
    """One-to-one (clip, cue, IoU) pairs by largest time overlap, IoU >= ``min_iou``."""
    candidates = []
    for c in clips:
        a, b = c["first"] / fps, (c["last"] + 1) / fps
        for q in cues:
            inter = min(b, q["end"]) - max(a, q["start"])
            if inter <= 0:
                continue
            iou = inter / (max(b, q["end"]) - min(a, q["start"]))
            if iou >= min_iou:
                candidates.append((iou, c["clip_id"], q["sample_id"], c, q))
    used_c, used_q, out = set(), set(), []
    for iou, cid, qid, c, q in sorted(candidates, key=lambda x: (-x[0], x[1], x[2])):
        if cid in used_c or qid in used_q:
            continue
        used_c.add(cid)
        used_q.add(qid)
        out.append((c, q, round(float(iou), 4)))
    return sorted(out, key=lambda x: x[0]["first"])


# ----------------------------------------------------------------------------------------------------------------------
# State quantities and temporal alignment

def state_items(state) -> dict:
    """label -> (values, valid): world arm directions (T, 3), wrist frames (T, 3, 3), finger bone directions in the palm
    frame (T, 3)."""
    from validate_semlex_adapter import ARM_REST, matrices, world_frames
    from signtranslator.data_engineering.video_landmarks import finger_joints
    world = world_frames(state)
    out = {}
    for joint, rest in ARM_REST.items():
        w, v = world[joint]
        out[joint] = (w @ np.asarray(rest), v)
    for joint in ("LeftHand", "RightHand"):
        out[joint] = world[joint]
    for side, name in (("Left", "left_hand"), ("Right", "right_hand")):
        ch = state.channels[name]
        local = matrices(ch)
        chain, chain_ok = None, None
        for joint in finger_joints(side):
            j = ch.labels.index(joint.name)
            if joint.parent == "Hand":
                chain, chain_ok = local[:, j], ch.valid[:, j].astype(bool)
            else:
                chain, chain_ok = chain @ local[:, j], chain_ok & ch.valid[:, j]
            out[joint.name] = (chain @ np.asarray(joint.rest), chain_ok.copy())
    return out


def group_of(label: str) -> str | None:
    if label.endswith("UpperArm"):
        return "upper_arms"
    if label.endswith("Forearm"):
        return "forearms"
    if label in ("LeftHand", "RightHand"):
        return "wrists"
    if "Hand" in label and label[-1].isdigit():
        return "fingers"
    return None


def error_of(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from validate_semlex_adapter import angles, direction_angles
    return direction_angles(a, b) if a.ndim == 2 else angles(a, b)


def image_arm_directions(pose_px: np.ndarray, found: np.ndarray) -> dict:
    """Raw UWB 2D arm bone vectors (image x right, y down; pixels) per bone, and validity."""
    return {name: (pose_px[:, b] - pose_px[:, a], found & np.isfinite(pose_px[:, [a, b]]).all((1, 2)))
            for name, (a, b) in ARM_BONES.items()}


def angle_2d(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return np.degrees(np.abs(np.arctan2(cross, (u * v).sum(-1))))


def frame_map(times: np.ndarray, first: int, frames: int, fps: float, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """(our index, UWB index) pairs: UWB frame j at (first + j + lag) / fps matched to our nearest frame within half a
    frame period."""
    uwb_times = (first + np.arange(frames) + lag) / fps
    pos = np.clip(np.searchsorted(times, uwb_times), 1, max(len(times) - 1, 1))
    left = np.clip(pos - 1, 0, len(times) - 1)
    right = np.clip(pos, 0, len(times) - 1)
    nearest = np.where(np.abs(times[left] - uwb_times) <= np.abs(times[right] - uwb_times), left, right)
    ok = np.abs(times[nearest] - uwb_times) <= .5 / fps + 1e-9
    j = np.flatnonzero(ok)
    return nearest[j], j


def choose_lag(times, first, frames, fps, ours: dict, uwb2d: dict, max_lag: int, min_frames: int = 10,
               margin: float = 0., curve: bool = False) -> dict:
    """The lag minimising the median image-plane arm angle between UWB's 2D bones and our projected directions; a
    non-zero lag must beat lag 0 by ``margin`` degrees. ``ours``: label -> ((T, 3) y-up world directions, valid);
    ``uwb2d``: label -> ((F, 2) image vectors, valid). ``curve``: also return every scored lag's
    [median angle (3 dp), bone-frames] (an arm-aligned result only)."""
    scores = {}
    for lag in range(-max_lag, max_lag + 1):
        i, j = frame_map(times, first, frames, fps, lag)
        errors = []
        for label in ARM_BONES:
            d, ok = ours[label]
            v, okv = uwb2d[label]
            projected = np.stack((d[i, 0], -d[i, 1]), -1)              # y-up world -> image (y down)
            keep = ok[i] & okv[j] & (np.linalg.norm(projected, axis=-1) > .3) & (np.linalg.norm(v[j], axis=-1) > 1e-6)
            errors += angle_2d(projected[keep], v[j][keep]).tolist()
        if len(errors) >= min_frames:
            scores[lag] = (float(np.median(errors)), len(errors))
    if not scores:
        return dict(lag=0, method="none", score=None)
    best = min(scores, key=lambda k: (scores[k][0], abs(k)))
    lag = 0 if 0 in scores and scores[0][0] - scores[best][0] < margin else best
    out = dict(lag=int(lag), method="arm-2d", score=round(scores[lag][0], 3), frames=scores[lag][1],
               best_lag=int(best), best_score=round(scores[best][0], 3),
               lag0_score=round(scores[0][0], 3) if 0 in scores else None)
    if curve:
        out["curve"] = {str(k): [round(v[0], 3), int(v[1])] for k, v in sorted(scores.items())}
    return out


def kabsch(pairs: list) -> np.ndarray | None:
    if len(pairs) < 3:
        return None
    from validate_semlex_adapter import best_rotation
    return best_rotation(np.asarray(pairs))


def compare(ours: dict, other: dict, i: np.ndarray, j: np.ndarray, support: dict | None = None) -> dict:
    """group -> errors (deg) on aligned frames where both hold the quantity (and ``support`` when given)."""
    out = collections.defaultdict(list)
    for label, (value, ok) in ours.items():
        group = group_of(label)
        if group is None or label not in other:
            continue
        value_b, ok_b = other[label]
        keep = ok[i] & ok_b[j]
        if support is not None:
            keep &= support[label]
        out[group] += error_of(value[i][keep], value_b[j][keep]).tolist()
    return out


def rotated_items(items: dict, rotation: np.ndarray) -> dict:
    out = dict(items)
    for label, (value, ok) in items.items():
        group = group_of(label)
        if group in ("upper_arms", "forearms"):
            out[label] = (value @ rotation.T, ok)
        elif group == "wrists":
            out[label] = (rotation @ value, ok)
    return out


def mirrored_arms(items: dict) -> dict:
    """The arm directions of the mirror image: left and right swap, x reflected."""
    reflect = np.array([-1., 1., 1.])
    out = {}
    for label in ARM_BONES:
        other = label.replace("Left", "@").replace("Right", "Left").replace("@", "Right")
        value, ok = items[other]
        out[label] = (value * reflect, ok)
    return out


def confusion(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return dict(both=int((a & b).sum()), ours_only=int((a & ~b).sum()), uwb_only=int((~a & b).sum()),
                neither=int((~a & ~b).sum()))


def kappa(cells: dict) -> float | None:
    """Cohen's kappa of a 2x2 agreement table (None when undefined)."""
    n = sum(cells[k] for k in CELLS)
    if not n:
        return None
    po = (cells["both"] + cells["neither"]) / n
    ours = (cells["both"] + cells["ours_only"]) / n
    uwb = (cells["both"] + cells["uwb_only"]) / n
    pe = ours * uwb + (1 - ours) * (1 - uwb)
    return None if pe >= 1. else round((po - pe) / (1 - pe), 4)


def hips_inside(uwb_clip) -> np.ndarray:
    """(F,) both hip landmarks inside the UWB frame (the framing stratum)."""
    pose = uwb_clip.pose
    xy = pose[:, [23, 24]]
    with np.errstate(invalid="ignore"):
        return (np.isfinite(xy).all((1, 2)) & (xy[..., 0] >= 0).all(1) & (xy[..., 0] <= uwb_clip.width).all(1)
                & (xy[..., 1] >= 0).all(1) & (xy[..., 1] <= uwb_clip.height).all(1))


def hand_agreement(ours_state, uwb_state, i, j, strata=None) -> dict:
    """{stratum: {side: confusion}} of hand validity on aligned frames (stratum "all" plus the given masks)."""
    strata = {"all": np.ones(len(j), bool), **(strata or {})}
    out = {}
    for name, mask in strata.items():
        out[name] = {}
        for side, channel in (("Left", "left_hand"), ("Right", "right_hand")):
            a = ours_state.channels[channel].valid.any(axis=1)[i][mask]
            b = uwb_state.channels[channel].valid.any(axis=1)[j][mask]
            out[name][side] = confusion(a, b)
    return out


def arm_agreement(ours_state, uwb_state, i, j) -> dict:
    """Forearm validity per side on aligned frames where our chest is valid (our original solver needs both hips for
    the chest, so elsewhere our arms are absent by construction, not by visibility)."""
    a, b = ours_state.channels["body"], uwb_state.channels["body"]
    chest = a.valid[i, a.labels.index("Chest")].astype(bool)
    out = {}
    for side in ("Left", "Right"):
        x = a.valid[i, a.labels.index(f"{side}Forearm")].astype(bool)[chest]
        y = b.valid[j, b.labels.index(f"{side}Forearm")].astype(bool)[chest]
        out[side] = confusion(x, y)
    return out


def forearm_survival(uwb_state, frames: np.ndarray, strata: dict) -> dict:
    """{stratum: {side: [kept, frames]}}: forearm-valid frames of a UWB state among its pose frames ``frames``."""
    body = uwb_state.channels["body"]
    out = {}
    for name, mask in {"all": np.ones(len(frames), bool), **strata}.items():
        out[name] = {}
        for side in ("Left", "Right"):
            valid = body.valid[:, body.labels.index(f"{side}Forearm")].astype(bool)[frames][mask]
            out[name][side] = [int(valid.sum()), int(mask.sum())]
    return out


def head_errors(ours_state, uwb_state, i, j) -> list:
    from validate_semlex_adapter import angles, matrices
    a, b = ours_state.channels["head_rotation"], uwb_state.channels["head_rotation"]
    if not a.valid.any() or not b.valid.any():
        return []
    keep = a.valid[i, 0] & b.valid[j, 0]
    return angles(matrices(a)[i, 0][keep], matrices(b)[j, 0][keep]).tolist()


def _capped(values, cap: int, seed: int) -> np.ndarray:
    """float32 errors, at most ``cap`` of them (a seeded subsample: medians and percentiles survive, memory does too)."""
    v = np.asarray(values, dtype=np.float32)
    if cap and len(v) > cap:
        v = v[np.sort(np.random.default_rng(seed).choice(len(v), cap, replace=False))]
    return v


def zlib_seed(text: str) -> int:
    import zlib
    return zlib.crc32(text.encode())


def evaluate_pair(ours_state, uwb_states: dict, uwb_clip, fps: float, max_lag: int, head: bool = False,
                  cap: int = 0, lag_margin: float = 0., lag_curves: bool = False, secondary=()) -> dict:
    """Errors and validity of each UWB method's state against ours on the aligned frames. ``uwb_states``: method ->
    state. Every error list is a float32 array, subsampled to ``cap`` values per group when ``cap`` > 0.
    ``secondary``: methods scored and reported that do not define the common frames; ``lag_curves``: the alignment
    carries its per-lag cost curve."""
    from signtranslator.data_engineering.uwb_keypoints import _found
    times = ours_state.timestamps
    ours = state_items(ours_state)
    pose_found = _found(uwb_clip.pose)
    alignment = choose_lag(times, uwb_clip.first, uwb_clip.frames, fps, ours,
                           image_arm_directions(uwb_clip.pose, pose_found), max_lag, margin=lag_margin,
                           curve=lag_curves)
    i, j = frame_map(times, uwb_clip.first, uwb_clip.frames, fps, alignment["lag"])
    hips = hips_inside(uwb_clip)
    aligned_strata = {"hips_visible": hips[j], "hips_not_visible": ~hips[j]}
    pose_frames = np.flatnonzero(pose_found)
    pose_strata = {"hips_visible": hips[pose_frames], "hips_not_visible": ~hips[pose_frames]}
    items = {m: state_items(s) for m, s in uwb_states.items()}
    support = {label: np.all([items[m][label][1][j] for m in items if m not in secondary], axis=0) for label in ours}
    result = dict(alignment=alignment, aligned_frames=int(len(i)), our_frames=int(len(times)),
                  uwb_frames=int(uwb_clip.frames), hips_visible_share=round(float(hips[pose_found].mean()), 4)
                  if pose_found.any() else None, frame_count_difference=int(uwb_clip.frame_count_difference),
                  methods={})
    seed = zlib_seed(uwb_clip.clip_id)
    for method, other in items.items():
        own = compare(ours, other, i, j)
        common = compare(ours, other, i, j, support)
        pairs = []
        for label in ARM_BONES:
            keep = ours[label][1][i] & other[label][1][j]
            pairs += np.c_[other[label][0][j][keep], ours[label][0][i][keep]].tolist()
        rotation = kabsch(pairs)
        aligned = compare(ours, rotated_items(other, rotation), i, j) if rotation is not None else {}
        mirror = compare(ours, mirrored_arms(other), i, j)
        result["methods"][method] = dict(
            own={g: _capped(v, cap, seed) for g, v in own.items()},
            common={g: _capped(v, cap, seed) for g, v in common.items()},
            rotated={g: _capped(v, cap, seed) for g, v in aligned.items() if g in ("upper_arms", "forearms", "wrists")},
            arms=_capped(own.get("upper_arms", []) + own.get("forearms", []), cap, seed),
            mirror_arms=_capped(mirror.get("upper_arms", []) + mirror.get("forearms", []), cap, seed),
            hands=hand_agreement(ours_state, uwb_states[method], i, j, aligned_strata),
            forearms_valid=arm_agreement(ours_state, uwb_states[method], i, j),
            forearm_survival=forearm_survival(uwb_states[method], pose_frames, pose_strata))
        if head:
            result["methods"][method]["head"] = _capped(head_errors(ours_state, uwb_states[method], i, j), cap, seed)
    return result


def input_sample(regressors, uwb_clip, frames: int = Z_FRAMES, seed: int = 0) -> dict:
    """Standardised regressor inputs (float16) of up to ``frames`` UWB frames: {"pose": (n, D), "Left": ..., "Right"}."""
    from signtranslator.data_engineering.uwb_keypoints import _found
    from signtranslator.data_engineering.world_regressor import input_zscores
    found = _found(uwb_clip.pose)
    rows = np.flatnonzero(found)
    if not len(rows):
        return {}
    rows = np.sort(np.random.default_rng(seed).choice(rows, min(frames, len(rows)), replace=False))
    scale = np.array([uwb_clip.width, uwb_clip.height], dtype=np.float64)
    pose = uwb_clip.pose[rows] / scale
    hands = {}
    for k, side in enumerate(("Left", "Right")):
        ok = _found(uwb_clip.hands[rows, k])
        hands[side] = uwb_clip.hands[rows, k][ok] / scale
    z = input_zscores(regressors, pose, hands, uwb_clip.aspect)
    return {k: np.asarray(v, dtype=np.float16) for k, v in z.items()}


# ----------------------------------------------------------------------------------------------------------------------
# Summaries

def stat(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), median=round(float(np.median(v)), 2), p75=round(float(np.percentile(v, 75)), 2),
                p90=round(float(np.percentile(v, 90)), 2), mean=round(float(v.mean()), 2))


def clustered_median_ci(by_video: dict, *, resamples: int = BOOTSTRAP, seed: int = 0) -> list | None:
    """95% interval of the pooled median, resampling videos with replacement. Exact and identical to concatenating the
    resampled videos and taking np.median (same draws), but faster: the values are sorted once, and each resample
    weights them by its video counts and reads the median off the cumulative weights."""
    keys = [k for k, v in by_video.items() if len(v)]
    if len(keys) < 2 or resamples <= 0:
        return None
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(by_video[k], dtype=np.float64) for k in keys]
    values = np.concatenate(arrays)
    owner = np.concatenate([np.full(len(a), i) for i, a in enumerate(arrays)])
    order = np.argsort(values, kind="stable")
    values, owner = values[order], owner[order]
    medians = []
    for _ in range(resamples):
        pick = rng.integers(0, len(keys), len(keys))
        cum = np.cumsum(np.bincount(pick, minlength=len(keys))[owner])
        n = int(cum[-1])
        if n % 2:
            medians.append(values[np.searchsorted(cum, (n + 1) // 2)])
        else:
            medians.append((values[np.searchsorted(cum, n // 2)] + values[np.searchsorted(cum, n // 2 + 1)]) / 2)
    return [round(float(np.percentile(medians, 2.5)), 2), round(float(np.percentile(medians, 97.5)), 2)]


def lag_drift(rows: list[tuple[float, int]], *, min_cues: int = DRIFT_MIN_CUES,
              min_span: float = DRIFT_MIN_SPAN) -> float | None:
    """Least-squares slope of lag (frames) against cue time, in frames per minute; None (unchecked) with fewer than
    ``min_cues`` arm-aligned cues or a span under ``min_span`` seconds: a few cues cannot tell drift from lag noise."""
    if len(rows) < min_cues:
        return None
    t, lag = np.array(rows, dtype=np.float64).T
    if np.ptp(t) < min_span:
        return None
    return round(float(np.polyfit(t, lag, 1)[0] * 60.), 3)


def pooled_video_lags(pairs: list[dict], *, min_pairs: int = 3) -> dict:
    """Gate v2's timing measurement: per video with >= ``min_pairs`` evaluated, stream-match (``stratum_of``),
    arm-aligned pairs that carry a lag curve, the pooled curve is the mean of the pairs' per-lag median angles over the
    lags every one of them scored (the summed cost curves, in degrees per pair); the pooled lag is its minimum (ties to
    the smaller |lag|, then the earlier lag). Returns {video: {pairs, lag, gain_over_lag0, curve}}; ``gain_over_lag0``
    is None when lag 0 is not among the common lags."""
    by_video = collections.defaultdict(list)
    for p in pairs:
        a = p.get("alignment") or {}
        if (p.get("status") == "evaluated" and a.get("method") == "arm-2d" and a.get("curve")
                and stratum_of(p)["stream_match"] == "yes"):
            by_video[p["video_id"]].append(a["curve"])
    out = {}
    for video, curves in sorted(by_video.items()):
        if len(curves) < min_pairs:
            continue
        lags = sorted(set.intersection(*(set(c) for c in curves)), key=int)
        if not lags:
            out[video] = dict(pairs=len(curves), lag=None, gain_over_lag0=None, curve={})
            continue
        pooled = {k: float(np.mean([c[k][0] for c in curves])) for k in lags}
        best = min(lags, key=lambda k: (pooled[k], abs(int(k)), int(k)))
        out[video] = dict(pairs=len(curves), lag=int(best), curve={k: round(v, 4) for k, v in pooled.items()},
                          gain_over_lag0=round(pooled["0"] - pooled[best], 4) if "0" in pooled else None)
    return out


ARM_POINTS = {"Left": (11, 13, 15), "Right": (12, 14, 16)}      # shoulder, elbow, wrist


def arm_distance_counts(uwb_clip) -> dict:
    """{side: {bone: {bin: frames}}} on the clip's pose frames: a bone's distance outside the UWB frame (frame
    heights) is the larger of its two end points' (forearm: elbow and wrist; upper arm: shoulder and elbow), since the
    swing joint of a bone needs both ends."""
    from signtranslator.data_engineering.uwb_keypoints import _found
    found = _found(uwb_clip.pose)
    xy = uwb_clip.pose[found] / np.array([uwb_clip.width, uwb_clip.height], dtype=np.float64)
    out = {}
    for side, (shoulder, elbow, wrist) in ARM_POINTS.items():
        dist = {k: outside_distance(xy[:, k], uwb_clip.aspect) for k in (shoulder, elbow, wrist)}
        out[side] = {}
        for bone, (a, b) in (("Forearm", (elbow, wrist)), ("UpperArm", (shoulder, elbow))):
            d = np.maximum(dist[a], dist[b])
            out[side][bone] = {name: int(((d == 0) if hi == 0 else ((d > lo) & (d <= hi))).sum())
                               for name, lo, hi in ELBOW_BINS}
    return out


def outside_distance(xy: np.ndarray, aspect: float) -> np.ndarray:
    """Distance (frame heights) of normalised image points from the image rectangle; 0 inside."""
    dx = np.maximum(np.maximum(-xy[..., 0], xy[..., 0] - 1.), 0.) * aspect
    dy = np.maximum(np.maximum(-xy[..., 1], xy[..., 1] - 1.), 0.)
    return np.hypot(dx, dy)


def group_stats(rows: list[dict], method: str, scope: str, resamples: int) -> dict:
    out = {}
    for g in GROUPS:
        by_video = collections.defaultdict(list)
        for r in rows:
            by_video[r["video_id"]].append(np.asarray(r["methods"][method][scope].get(g, ()), dtype=np.float64))
        arrays = {k: np.concatenate(v) for k, v in by_video.items()}
        pooled = np.concatenate(list(arrays.values())) if arrays else np.zeros(0)
        if len(pooled):
            out[g] = dict(stat(pooled), median_ci95_by_video=clustered_median_ci(arrays, resamples=resamples))
    return out


def agreement_summary(counts: dict) -> dict:
    total = sum(counts[k] for k in CELLS)
    return dict(counts, frames=total, agreement=round((counts["both"] + counts["neither"]) / total, 4) if total else None,
                kappa=kappa(counts), uwb_only_rate=round(counts["uwb_only"] / total, 4) if total else None)


def stratum_of(r: dict) -> dict:
    fps, meta = r["fps"], r.get("meta_fps")
    share = r.get("hips_visible_share")
    return dict(hips=None if share is None else ("visible" if share >= .5 else "not_visible"),
                fps30="yes" if abs(fps - 30.) < .05 or abs(fps - 29.97) < .01 else "no",
                stream_match=None if not meta else ("yes" if abs(fps / meta - 1) < .002 else "no"),
                fc_diff={0: "0", -1: "-1"}.get(r.get("frame_count_difference"), "other"))


def input_summary(results: list[dict], names: dict | None) -> dict:
    out = {}
    for model in ("pose", "Left", "Right"):
        arrays = [np.asarray(r["inputs"][model], dtype=np.float64) for r in results if len(r.get("inputs", {}).get(model, ()))]
        if not arrays:
            continue
        z = np.abs(np.concatenate(arrays))
        p99 = np.percentile(z, 99, axis=0)
        labels = (names or {}).get("pose" if model == "pose" else "hand") or [str(k) for k in range(len(p99))]
        out[model] = dict(frames=int(len(z)), max_p99_abs_z=round(float(p99.max()), 3),
                          inputs_over_4=[labels[k] for k in np.flatnonzero(p99 > 4.)],
                          p99_abs_z={labels[k]: round(float(p99[k]), 3) for k in range(len(p99))},
                          median_abs_z={labels[k]: round(float(v), 3) for k, v in enumerate(np.median(z, axis=0))})
    return out


def names_for(results: list[dict]) -> dict | None:
    """Input names of the bundle that produced the sampled inputs (its pose width tells xy-v1 from xy-v2)."""
    from signtranslator.data_engineering.world_regressor import FEATURES_XY, FEATURES_XY2, input_names
    for r in results:
        pose = r.get("inputs", {}).get("pose")
        if pose is not None and len(pose):
            width = np.asarray(pose).shape[1]
            for version in (FEATURES_XY, FEATURES_XY2):
                if len(input_names(version)["pose"]) == width:
                    return input_names(version)
    return None


def summarise(results: list[dict], *, primary: str | None = None, resamples: int = BOOTSTRAP,
              input_names: dict | None = None) -> dict:
    ok = [r for r in results if r.get("status") == "evaluated"]
    methods = sorted({m for r in ok for m in r["methods"]})
    summary = dict(pairs=len(ok), videos=len({r["video_id"] for r in ok}), primary=primary, methods={})
    for m in methods:
        started = time.monotonic()
        entry = {scope: group_stats(ok, m, scope, resamples) for scope in ("common", "own", "rotated")}
        print(f"summarise {m}: angle statistics {time.monotonic() - started:.0f} s", flush=True)
        hands = collections.defaultdict(collections.Counter)
        for r in ok:
            for stratum, sides in r["methods"][m]["hands"].items():
                for side, cells in sides.items():
                    hands[f"{stratum}.{side}"].update(cells)
        entry["hand_valid"] = {key: agreement_summary({k: hands[key][k] for k in CELLS}) for key in sorted(hands)}
        arms = collections.defaultdict(collections.Counter)
        for r in ok:
            for side, cells in r["methods"][m]["forearms_valid"].items():
                arms[side].update(cells)
        entry["forearm_valid_where_our_chest_is"] = {side: agreement_summary({k: arms[side][k] for k in CELLS})
                                                     for side in sorted(arms)}
        survival = collections.defaultdict(lambda: [0, 0])
        for r in ok:
            for stratum, sides in r["methods"][m]["forearm_survival"].items():
                for side, (kept, total) in sides.items():
                    survival[f"{stratum}.{side}"][0] += kept
                    survival[f"{stratum}.{side}"][1] += total
        entry["forearm_survival"] = {k: dict(kept=v[0], frames=v[1], share=round(v[0] / v[1], 4) if v[1] else None)
                                     for k, v in sorted(survival.items())}
        both = [r for r in ok if len(r["methods"][m]["arms"]) and len(r["methods"][m]["mirror_arms"])]
        normal = [float(np.median(r["methods"][m]["arms"])) for r in both]
        mirror = [float(np.median(r["methods"][m]["mirror_arms"])) for r in both]
        entry["mirror_check"] = dict(clips=len(both), mirror_better=int(sum(b < a for a, b in zip(normal, mirror))),
                                     median_of_clip_median_arm_error=stat(normal).get("median"),
                                     median_of_clip_median_arm_error_if_mirrored=stat(mirror).get("median"))
        heads = [np.asarray(r["methods"][m]["head"]) for r in ok if len(r["methods"][m].get("head", ()))]
        if heads:
            entry["head"] = stat(np.concatenate(heads))
        summary["methods"][m] = entry
    if primary in summary["methods"]:
        strata = {}
        for kind in ("hips", "fps30", "stream_match", "fc_diff"):
            values = collections.defaultdict(list)
            for r in ok:
                values[stratum_of(r)[kind]].append(r)
            strata[kind] = {str(v): dict(pairs=len(rows), common=group_stats(rows, primary, "common", 0))
                            for v, rows in sorted(values.items(), key=lambda x: str(x[0]))}
        summary["strata"] = strata
    aligned = [r for r in ok if r["alignment"]["method"] == "arm-2d"]
    by_fc = collections.defaultdict(collections.Counter)
    for r in aligned:
        by_fc[stratum_of(r)["fc_diff"]][r["alignment"]["lag"]] += 1
    summary["alignment"] = dict(methods=dict(collections.Counter(r["alignment"]["method"] for r in ok)),
                                lags={str(k): v for k, v in sorted(collections.Counter(
                                    r["alignment"]["lag"] for r in ok).items())},
                                lags_by_frame_count_difference={fc: {str(k): v for k, v in sorted(c.items())}
                                                                for fc, c in sorted(by_fc.items())},
                                within_1_share=round(sum(abs(r["alignment"]["lag"]) <= 1 for r in aligned)
                                                     / len(aligned), 4) if aligned else None)
    by_video = collections.defaultdict(list)
    for r in aligned:
        by_video[r["video_id"]].append((r["cue_start"], r["alignment"]["lag"]))
    drifts = {v: lag_drift(rows) for v, rows in by_video.items()}
    summary["lag_drift_frames_per_minute"] = {v: d for v, d in drifts.items() if d is not None}
    summary["drift_status"] = {v: ("unchecked" if drifts.get(v) is None else
                                   "dropped" if abs(drifts[v]) > DRIFT_LIMIT else "ok")
                               for v in sorted({r["video_id"] for r in ok})}
    arms = collections.defaultdict(collections.Counter)
    for r in ok:
        for side, bones in r.get("arm_distance", {}).items():
            for bone, bins in bones.items():
                arms[(side, bone)].update(bins)
    summary["uwb_arm_distance"] = {}
    for (side, bone), c in sorted(arms.items()):
        summary["uwb_arm_distance"].setdefault(side, {})[bone] = dict(
            frames=sum(c.values()), share={name: round(c[name] / max(sum(c.values()), 1), 4) for name, *_ in ELBOW_BINS})
    summary["inputs"] = input_summary(ok, input_names)
    summary["refusals"] = dict(collections.Counter(r.get("reason", "?")[:60] for r in results
                                                   if r.get("status") != "evaluated"))
    if primary in summary["methods"]:
        summary["proposal_check"] = proposal_check(summary, primary)
    return summary


def proposal_check(summary: dict, primary: str, bar: dict = PROPOSAL) -> dict:
    """The proposed gate's criteria that this script can compute (True/False/None per item; not a declared verdict)."""
    e = summary["methods"][primary]
    out = {}
    for g, limit in bar["median"].items():
        s = e["common"].get(g, {})
        ci = s.get("median_ci95_by_video")
        out[f"median.{g}<={limit}"] = None if not s.get("n") else bool(s["median"] <= limit)
        out[f"median_ci_upper.{g}<={limit + bar['ci_slack']}"] = None if not ci else bool(ci[1] <= limit + bar["ci_slack"])
    for g, limit in bar["p90"].items():
        s = e["common"].get(g, {})
        out[f"p90.{g}<={limit}"] = None if not s.get("n") else bool(s["p90"] <= limit)
    for g in ("upper_arms", "forearms"):
        a, b = e["own"].get(g, {}), e["rotated"].get(g, {})
        out[f"rotation_gain.{g}<={bar['rotation_gain']}"] = (None if not a.get("n") or not b.get("n")
                                                             else bool(a["median"] - b["median"] <= bar["rotation_gain"]))
    worst = []
    for kind, values in summary.get("strata", {}).items():
        for value, s in values.items():
            if s["pairs"] < bar["stratum_pairs"]:
                continue
            for g, limit in bar["median"].items():
                m = s["common"].get(g, {}).get("median")
                if m is not None and m > limit + bar["stratum_slack"]:
                    worst.append(f"{kind}={value}:{g}={m}")
    out["strata_within_3deg_of_bar"] = not worst
    out["strata_failures"] = worst
    for side in ("Left", "Right"):
        h = e["hand_valid"].get(f"all.{side}", {})
        out[f"hand_agreement.{side}>={bar['hand_agreement']}"] = (None if h.get("agreement") is None
                                                                  else bool(h["agreement"] >= bar["hand_agreement"]))
        f = e["forearm_valid_where_our_chest_is"].get(side, {})
        out[f"forearm_agreement_hips_visible.{side}>={bar['forearm_agreement']}"] = (
            None if f.get("agreement") is None else bool(f["agreement"] >= bar["forearm_agreement"]))
        rates = [x.get("uwb_only_rate") for x in (h, f) if x.get("uwb_only_rate") is not None]
        out[f"uwb_only_rate.{side}<={bar['uwb_only']}"] = None if not rates else bool(max(rates) <= bar["uwb_only"])
    out["chest_up_forearm_survival_vs_mediapipe"] = "measured by scripts/uwb_outside_frame_accuracy.py"
    z = [v["max_p99_abs_z"] for v in summary.get("inputs", {}).values()]
    out["inputs_diagnostic"] = dict(max_p99_abs_z=max(z) if z else None, reference=bar["input_z"],
                                    note="a reported diagnostic (summary.inputs has every input), not a gate item")
    mc = e["mirror_check"]
    out[f"mirror_better_share<={bar['mirror_share']}"] = (None if not mc["clips"]
                                                          else bool(mc["mirror_better"] / mc["clips"] <= bar["mirror_share"]))
    w = summary["alignment"]["within_1_share"]
    out[f"lag_within_1_share>={bar['lag_within_1']}"] = None if w is None else bool(w >= bar["lag_within_1"])
    drifts = list(summary["lag_drift_frames_per_minute"].values())
    out[f"drift_below_1_share>={bar['drift_share']}"] = (None if not drifts else
                                                        bool(np.mean(np.abs(drifts) < DRIFT_LIMIT) >= bar["drift_share"]))
    out["drift_checked_videos"] = len(drifts)
    out["relative_to_how2sign_val"] = "not computed here (needs the bundle's How2Sign-val direction errors)"
    out["downstream_back_translator"] = "not computed here"
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Data access

class _Ranged(io.RawIOBase):
    """Seekable view of one store object through ranged GETs (random member access: small buffer)."""

    def __init__(self, store, key: str):
        self.store, self.key, self.pos = store, key, 0
        self.size = store.s3.head_object(Bucket=store.bucket, Key=key)["ContentLength"]

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        data = self.store._call(lambda: self.store.s3.get_object(
            Bucket=self.store.bucket, Key=self.key, Range=f"bytes={self.pos}-{end}"))["Body"].read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_member_zip(store, key: str) -> zipfile.ZipFile:
    if hasattr(store, "s3"):
        return zipfile.ZipFile(io.BufferedReader(_Ranged(store, key), buffer_size=1 << 20))
    return zipfile.ZipFile(store.root / key)


def members_index(store, prefix: str, cache: Path | None) -> dict:
    """clip id -> zip name, from every zip's central directory (cached locally as {zip: [members]})."""
    if cache is not None and cache.exists():
        listing = json.loads(cache.read_text())
    else:
        listing = {}
        for key in sorted(k for k in store.list(prefix + "/") if re.search(r"raw_keypoints_\d+\.zip$", k)):
            with open_member_zip(store, key) as archive:
                listing[Path(key).name] = archive.namelist()
        if cache is not None:
            cache.write_text(json.dumps(listing))
    return {Path(n).name[:-5]: z for z, names in listing.items() for n in names if n.endswith(".json")}


def uwb_clips_of(entry: dict) -> list[dict]:
    from signtranslator.data_engineering.uwb_keypoints import Refusal, parse_clip_name
    out = []
    for clip in entry.get("clip_order", []):
        try:
            clip_id, video, first, last = parse_clip_name(clip)
        except Refusal:
            continue
        out.append(dict(clip_id=clip_id, first=first, last=last, text=(entry.get(clip) or {}).get("translation")))
    return out


def read_video_list(path: str | None) -> set:
    """Video ids from a JSON list or a text file (one per line, '#' comments)."""
    if not path:
        return set()
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        return set(json.loads(text))
    return {line.split("#")[0].strip() for line in text.splitlines() if line.split("#")[0].strip()}


def method_rules(declared: str, candidates: list[str]) -> dict:
    """method name -> visibility rule: the declared lifter first, then the candidate rules."""
    from signtranslator.data_engineering.uwb_keypoints import visibility_margin
    rules = [declared] + [c for c in candidates if c and c != declared]
    for rule in rules:
        visibility_margin(rule)
    return {f"lifter[{rule}]": rule for rule in dict.fromkeys(rules)}


def secondary_rules(args) -> dict:
    """method name -> rule of ``--secondary-visibility`` (reported methods outside the common frames)."""
    from signtranslator.data_engineering.uwb_keypoints import visibility_margin
    rules = [r for r in (getattr(args, "secondary_visibility", "") or "").split(",") if r]
    declared = method_rules(args.visibility, args.candidate_visibility.split(","))
    out = {}
    for rule in rules:
        visibility_margin(rule)
        if f"lifter[{rule}]" in declared:
            raise ValueError(f"secondary rule {rule!r} is already the declared or a candidate rule")
        out[f"lifter[{rule}]"] = rule
    return out


def _init(args):
    import torch
    torch.set_num_threads(1)
    from train_text_to_motion import make_store
    from signtranslator.data_engineering.world_regressor import WorldRegressors, Z_FREE_FEATURES
    store = make_store(args.store)
    payload = Path(args.regressors).read_bytes() if Path(args.regressors).exists() else store.get(args.regressors)
    regressors = WorldRegressors.load(payload, expected_sha256=args.regressors_sha256)
    if regressors.features not in Z_FREE_FEATURES:
        raise ValueError(f"the bundle's features are {regressors.features!r}; UWB needs one of {sorted(Z_FREE_FEATURES)}")
    secondary = secondary_rules(args)
    _WORKER.update(args=args, store=store, archives={}, regressors=regressors,
                   rules={**method_rules(args.visibility, args.candidate_visibility.split(",")), **secondary},
                   secondary=frozenset(secondary))
    from signtranslator.data_engineering.semlex_pose import BlendshapeModel, load_face_resources
    resources = load_face_resources(Path(args.models_dir) / "face_landmarker.task") if args.head == "affine" else None
    _WORKER["resources"] = resources
    _WORKER["blendshapes"] = BlendshapeModel(resources) if resources is not None else None


def _evaluate(job: dict) -> dict:
    from signtranslator.data_engineering import uwb_keypoints as uk
    from signtranslator.pose.multichannel import MultichannelMotion
    args, store = _WORKER["args"], _WORKER["store"]
    base = {k: job.get(k) for k in ("video_id", "clip_id", "sample_id", "cue_start", "iou", "fps", "meta_fps", "zip")}
    try:
        archive = _WORKER["archives"].get(job["zip"])
        if archive is None:
            archive = _WORKER["archives"][job["zip"]] = open_member_zip(store, f"{args.uwb_prefix}/{job['zip']}")
        payload = archive.read(job["clip_id"] + ".json")
        clip = uk.parse_clip(payload, job["clip_id"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.npz"
            path.write_bytes(store.get(job["state_key"]))
            ours = MultichannelMotion.load(path, expected_sha256=job["state_sha256"])
        sha = hashlib.sha256(payload).hexdigest()
        common = dict(source_sha256=sha, fps_source="estimated", blendshape_model=_WORKER["blendshapes"],
                      head=args.head, face_resources=_WORKER["resources"], flags="generator")
        states = {m: uk.convert_clip(clip, job["fps"], regressors=_WORKER["regressors"], visibility=rule, **common)[0]
                  for m, rule in _WORKER["rules"].items()}
        states["pseudo"] = uk.convert_clip(clip, job["fps"], world="pseudo", visibility="inside", **common)[0]
        result = evaluate_pair(ours, states, clip, job["fps"], args.max_lag, head=args.head == "affine",
                               cap=args.max_values_per_pair, lag_margin=args.lag_margin,
                               lag_curves=getattr(args, "lag_curves", False), secondary=_WORKER.get("secondary", ()))
        result["inputs"] = input_sample(_WORKER["regressors"], clip, seed=zlib_seed(job["clip_id"]))
        result["arm_distance"] = arm_distance_counts(clip)
    except (uk.Refusal, ValueError, KeyError) as error:
        return dict(base, status="refused", reason=f"{type(error).__name__}: {error}"[:200])
    return dict(base, status="evaluated", member_sha256=sha, **result)


def _load_report(job):
    spec, key = job
    from train_text_to_motion import make_store
    store = _WORKER.get("report_store") or _WORKER.setdefault("report_store", make_store(spec))
    try:
        return json.loads(store.get(key))
    except Exception:  # noqa: BLE001 - an unreadable report: that video is skipped and counted
        return None


def summarise_fps(rows: dict) -> dict:
    """Agreement of the estimates with yt-dlp's rate in our reports, and the methods used."""
    methods = collections.Counter(r["method"] for r in rows.values())
    with_meta = [(r["fps"], r["meta_fps"]) for r in rows.values() if r["fps"] is not None and r.get("meta_fps")]
    # yt-dlp reports NTSC rates rounded (30 for 29.97, 24 for 23.976): agreement within 0.2% covers that, not 25 vs 24
    agree = sum(abs(f / m - 1) < .002 for f, m in with_meta)
    exact = sum(abs(f - m) < .01 for f, m in with_meta)
    estimated = [r["fps"] for r in rows.values() if r["fps"] is not None]
    counts = collections.Counter(f"{f:g}" for f in estimated)
    meta = collections.Counter(f"{r['meta_fps']:g}" for r in rows.values() if r.get("meta_fps"))
    residuals = [r["residual_median"] for r in rows.values() if r.get("residual_median") is not None]
    refused = collections.Counter(re.sub(r"[-\d.]+", "N", r["refused"]) for r in rows.values() if r.get("refused"))
    return dict(videos=len(rows), estimated=len(estimated), methods={str(k): v for k, v in methods.items()},
                refused=dict(refused), with_meta=len(with_meta), agree_with_meta_within_ntsc=agree, equal_to_meta=exact,
                estimated_rates=dict(counts.most_common()), meta_rates=dict(meta.most_common()),
                share_not_about_30=round(sum(abs(f - 30) > .1 and abs(f - 29.97) > .01 for f in estimated)
                                         / max(len(estimated), 1), 4),
                residual_median_of_medians=round(float(np.median(residuals)), 3) if residuals else None)


def cue_overlaps(clip: dict, cues: list[dict], fps: float, *, keep: int = 3, min_cover: float = .05) -> list[dict]:
    """Our cues overlapping a UWB clip in time, largest cover first: ``cover`` = intersection / the UWB clip's span,
    ``iou`` = intersection / union; at most ``keep`` with cover >= ``min_cover``."""
    a, b = clip["first"] / fps, (clip["last"] + 1) / fps
    out = []
    for q in cues:
        inter = min(b, q["end"]) - max(a, q["start"])
        if inter <= 0 or b <= a:
            continue
        cover = inter / (b - a)
        if cover >= min_cover:
            out.append(dict(sample_id=q["sample_id"], cover=round(float(cover), 4),
                            iou=round(float(inter / (max(b, q["end"]) - min(a, q["start"]))), 4)))
    return sorted(out, key=lambda o: (-o["cover"], o["sample_id"]))[:keep]


def overlap_table(fps_rows: dict, pairs_by_video: dict, reports: dict, monitor_share: float,
                  clips_by_video: dict | None = None) -> dict:
    """The converter's dedup table (schema ``uwb-overlap-table-v2``).

    - ``videos``: every video we extracted that UWB holds: our ``channel_id``, ``dedup`` (``cue`` when timed,
      ``whole-video`` otherwise: then every UWB clip of it is ``duplicate_of_our_video``), the informational hashed
      ``partition`` (manifest builders read partitions from our manifest records, ``uwb_manifest``), and the counts of
      paired, unpaired and unpaired-but-overlapping UWB clips;
    - ``clips``: every UWB clip of a timed video that overlaps one of our cues: the one-to-one pair (IoU >= 0.5) if
      any, and up to 3 overlapping cues with cover and IoU (near-duplicates the pairing does not catch)."""
    from build_youtube_asl_manifest import monitor_channel
    clips, videos = {}, {}
    totals = collections.Counter()
    for video_id, report in sorted(reports.items()):
        meta = report.get("meta") or {}
        fps = fps_rows.get(video_id, {}).get("fps")
        pairs = {clip["clip_id"]: (cue["sample_id"], iou) for clip, cue, iou in pairs_by_video.get(video_id, ())}
        entry = dict(channel_id=meta.get("channel_id"), dedup="cue" if fps is not None else "whole-video",
                     partition="monitor" if monitor_channel(meta.get("channel_id"), monitor_share) else "train",
                     paired=len(pairs), unpaired=0, unpaired_overlapping=0,
                     uwb_clips=len((clips_by_video or {}).get(video_id, ())))
        if fps is not None:
            for clip in (clips_by_video or {}).get(video_id, ()):
                overlaps = cue_overlaps(clip, report.get("segments", []), fps)
                pair = pairs.get(clip["clip_id"])
                if pair is None:
                    entry["unpaired"] += 1
                    entry["unpaired_overlapping"] += bool(overlaps)
                if pair is not None or overlaps:
                    clips[clip["clip_id"]] = dict(our_sample_id=pair and pair[0], iou=pair and pair[1],
                                                  overlaps=overlaps)
        videos[video_id] = entry
        totals.update({k: entry[k] for k in ("paired", "unpaired", "unpaired_overlapping", "uwb_clips")})
        totals["whole_video_clips"] += entry["uwb_clips"] if fps is None else 0
    return dict(schema="uwb-overlap-table-v2", monitor_share=monitor_share, clips=clips, videos=videos,
                totals=dict(totals, videos=len(videos), whole_videos=sum(v["dedup"] == "whole-video"
                                                                          for v in videos.values())),
                method="uwb_lifter_gate.pair_clips (IoU >= 0.5, one-to-one) and cue_overlaps (cover >= 0.05) at the "
                       "estimated rate; partitions are informational (read them from our manifest records)")


def load_split(store, spec: str) -> tuple[dict, str]:
    """(split, SHA-256) of a ``build_uwb_lifter_split.py`` output; refused unless its parts are disjoint by video and
    by channel."""
    from uwb_downstream_check import read_spec
    payload, sha = read_spec(store, spec)
    split = json.loads(payload)
    if split.get("schema") != "uwb-lifter-split-v1":
        raise SystemExit(f"{spec}: not a uwb-lifter-split-v1 file")
    parts = ("train", "lifter_val", "held_out")
    for kind in ("videos", "channels"):
        for a in range(3):
            for b in range(a + 1, 3):
                both = set(split[parts[a]][kind]) & set(split[parts[b]][kind])
                if both:
                    raise SystemExit(f"{spec}: {parts[a]} and {parts[b]} share {len(both)} {kind}")
    return split, sha


def check_population(jobs: list[dict], split: dict) -> None:
    """Refuses (SystemExit) any job whose video is not a held-out video of the split."""
    held = set(split["held_out"]["videos"])
    other = set(split["train"]["videos"]) | set(split["lifter_val"]["videos"])
    bad = sorted({j["video_id"] for j in jobs if j["video_id"] not in held or j["video_id"] in other})
    if bad:
        raise SystemExit(f"--split-file: {len(bad)} videos to evaluate are not held out, e.g. {bad[:3]}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default="r2")
    p.add_argument("--uwb-prefix", default=UWB_PREFIX)
    p.add_argument("--youtube-prefix", default=YOUTUBE_PREFIX)
    p.add_argument("--translations", default="YT.translations.all.json", help="UWB translations file under the prefix")
    p.add_argument("--regressors", help="z-free bundle (local path or store key); not needed with --fps-only")
    p.add_argument("--regressors-sha256")
    p.add_argument("--visibility", help="the declared visibility rule of the lifter (primary method)")
    p.add_argument("--candidate-visibility", default="learned,inside,margin:0.25",
                   help="comma-separated rules also scored (secondary)")
    p.add_argument("--models-dir", default=os.path.expanduser("~/.sign-translator-ops/models"))
    p.add_argument("--head", choices=("none", "affine"), default="none")
    p.add_argument("--members-cache", help="local JSON cache of the zips' member lists")
    p.add_argument("--exclude-videos", help="file of video ids kept out of the evaluation (development videos)")
    p.add_argument("--videos", type=int, default=0, help="evaluate pairs from at most this many videos (0: all)")
    p.add_argument("--pairs-per-video", type=int, default=5)
    p.add_argument("--max-lag", type=int, default=6)
    p.add_argument("--max-values-per-pair", type=int, default=600, help="per group and scope (seeded subsample)")
    p.add_argument("--lag-margin", type=float, default=.5,
                   help="degrees a non-zero lag must gain over lag 0 (the image-plane arm angle)")
    p.add_argument("--min-text-matches", type=int, default=3)
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    p.add_argument("--monitor-share", type=float, default=.01, help="as build_youtube_asl_manifest.py")
    p.add_argument("--fps-only", action="store_true", help="estimate rates for every overlapping video; no states")
    p.add_argument("--fps-table-out", help="write {video: fps} for convert_uwb_clips.py")
    p.add_argument("--overlap-table-out", help="write the dedup/partition table for convert_uwb_clips.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split-file", help="gate v2: build_uwb_lifter_split.py output (path, key or KEY@SHA256); only its "
                                         "held-out videos are evaluated")
    p.add_argument("--lag-curves", action="store_true", help="gate v2: store each pair's per-lag cost curve")
    p.add_argument("--secondary-visibility", default="",
                   help="gate v2: comma-separated rules scored as reported methods outside the common frames (head)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if not args.fps_only:
        if not (args.regressors and args.regressors_sha256):
            raise SystemExit("--regressors and --regressors-sha256 are required (or --fps-only)")
        if not args.visibility:
            raise SystemExit("--visibility (the declared rule) is required for an evaluation")
        try:
            method_rules(args.visibility, args.candidate_visibility.split(","))
            secondary_rules(args)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    if args.overlap_table_out and args.videos:
        raise SystemExit("--overlap-table-out needs every video: drop --videos")
    if args.split_file and (args.fps_only or args.overlap_table_out or args.fps_table_out):
        raise SystemExit("--split-file evaluates held-out videos only: no --fps-only, fps or overlap tables")
    excluded = read_video_list(args.exclude_videos)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    split = held_out = None
    if args.split_file:
        split, split_sha = load_split(store, args.split_file)
        held_out = set(split["held_out"]["videos"])
    uwb = json.loads(store.get(f"{args.uwb_prefix}/{args.translations}"))
    keys = sorted(k for k in store.list(f"{args.youtube_prefix}/videos/") if k.endswith(".json"))
    ids = [Path(k).name[:-5] for k in keys if Path(k).name[:-5] in uwb]
    if held_out is not None:
        ids = [v for v in ids if v in held_out]
    rng = random.Random(args.seed)
    if args.videos and not args.fps_only:          # a sample: read 3x as many reports as videos wanted, seeded
        population = [v for v in ids if v not in excluded]
        ids = sorted(rng.sample(population, min(len(population), 3 * args.videos)))
    fps_rows, jobs, pairs_by_video, converted, clips_by_video = {}, [], {}, {}, {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        loaded = list(pool.map(_load_report, [(args.store, f"{args.youtube_prefix}/videos/{v}.json") for v in ids],
                               chunksize=16))
    for video_id, report in zip(ids, loaded):
        if report is None or report.get("status") != "converted":
            continue
        converted[video_id] = report
        clips = uwb_clips_of(uwb[video_id])
        clips_by_video[video_id] = clips
        cues = report["segments"]
        estimate = estimate_fps(clips, cues, min_text_matches=args.min_text_matches)
        meta = (report.get("meta") or {}).get("fps")
        estimate.update(meta_fps=meta, uwb_clips=len(clips), our_cues=len(cues))
        fps_rows[video_id] = estimate
        if estimate["fps"] is None:
            continue
        pairs = pair_clips(clips, cues, estimate["fps"])
        pairs_by_video[video_id] = pairs
        estimate["pairs"] = len(pairs)
        for clip, cue, iou in pairs:
            jobs.append(dict(video_id=video_id, clip_id=clip["clip_id"], sample_id=cue["sample_id"],
                             cue_start=cue["start"], iou=iou, fps=estimate["fps"], meta_fps=meta,
                             state_key=cue["state_key"], state_sha256=cue["state_sha256"]))
    fps_summary = summarise_fps(fps_rows)
    print(json.dumps(fps_summary), flush=True)
    if args.overlap_table_out:
        table = overlap_table(fps_rows, pairs_by_video, converted, args.monitor_share, clips_by_video)
        Path(args.overlap_table_out).write_text(json.dumps(table, indent=1, sort_keys=True))
    out = dict(settings={k: v for k, v in vars(args).items()}, fps=dict(summary=fps_summary, videos=fps_rows),
               candidate_pairs=len(jobs), excluded_videos=sorted(excluded))
    if split is not None:
        out["split"] = dict(spec=args.split_file, sha256=split_sha, held_out_videos=len(held_out),
                            held_out_channels=len(split["held_out"]["channels"]))
    for key in ("split_file", "lag_curves", "secondary_visibility"):     # gate v1 reports stay exactly as before
        if not out["settings"][key]:
            del out["settings"][key]
    dropped, drift_status = {}, {}
    if not args.fps_only:
        by_video = collections.defaultdict(list)
        for job in jobs:
            if job["video_id"] not in excluded:
                by_video[job["video_id"]].append(job)
        videos = sorted(by_video)
        rng.shuffle(videos)
        videos = videos[:args.videos or None]
        index = members_index(store, args.uwb_prefix, Path(args.members_cache) if args.members_cache else None)
        chosen = []
        for v in videos:
            picked = rng.sample(by_video[v], min(args.pairs_per_video, len(by_video[v])))
            chosen += [dict(job, zip=index[job["clip_id"]]) for job in picked if job["clip_id"] in index]
        if held_out is not None:
            check_population(chosen, split)
        print(json.dumps(dict(videos=len(videos), pairs=len(chosen), excluded=len(excluded))), flush=True)
        with ProcessPoolExecutor(max_workers=max(1, args.workers), initializer=_init, initargs=(args,)) as pool:
            results, started = [], time.monotonic()
            for result in pool.map(_evaluate, chosen, chunksize=1):
                results.append(result)
                if len(results) % 250 == 0 or len(results) == len(chosen):
                    print(f"pairs {len(results)}/{len(chosen)} ({time.monotonic() - started:.0f} s)", flush=True)
        primary = f"lifter[{args.visibility}]"
        out["regressors_sha256"] = args.regressors_sha256
        out["summary"] = summarise(results, primary=primary, resamples=args.bootstrap, input_names=names_for(results))
        out["pairs"] = [{k: v for k, v in r.items() if k not in ("methods", "inputs")} for r in results]
        dropped = {v: d for v, d in out["summary"]["lag_drift_frames_per_minute"].items() if abs(d) > DRIFT_LIMIT}
        out["fps_dropped_for_drift"] = dropped
        drift_status = out["summary"]["drift_status"]
        print(json.dumps(out["summary"], indent=1, default=str), flush=True)
    if args.fps_table_out:
        table = {v: r["fps"] for v, r in sorted(fps_rows.items()) if r["fps"] is not None and v not in dropped}
        Path(args.fps_table_out).write_text(json.dumps(dict(
            schema="uwb-fps-table-v1", method="uwb_lifter_gate.estimate_fps (text-matched caption cues, Theil-Sen + "
                                             "least squares, snapped; offset in [-1, 0]; 10-70 fps)",
            drift_checked=not args.fps_only, dropped_for_drift=sorted(dropped),
            drift={v: drift_status.get(v, "unchecked") for v in table}, drift_rule=dict(
                min_cues=DRIFT_MIN_CUES, min_span_seconds=DRIFT_MIN_SPAN, limit_frames_per_minute=DRIFT_LIMIT),
            fps=table), indent=1, sort_keys=True))
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
