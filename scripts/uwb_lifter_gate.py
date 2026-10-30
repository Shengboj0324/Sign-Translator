"""Paired validation of the UWB lifter: UWB-lifted states against our own Tasks-extracted states of the same cues.

For YouTube-ASL videos that we extracted ourselves (``<youtube prefix>/videos/<id>.json``: caption cues with start,
end, text and state key) **and** that the UWB keypoint dataset also holds (clip ids ``<id>.<first>-<last>``):

1. **Frame rate, per video.** UWB clip names are source frame ranges at an unpublished rate. UWB clips and our cues
   come from the same caption track, so cues whose normalised text matches exactly once on both sides give pairs
   ``(start, first)`` and ``(end, last)``; a least-squares line ``frame = fps * t + c`` with outliers (> 3 frames)
   dropped gives the rate, snapped to a standard rate within 0.03% (``STANDARD_FPS``). With fewer than
   ``--min-text-matches`` text pairs, each standard rate is scored by how many UWB clips land on a cue
   (``|first - start * fps| <= 2`` and ``|last - end * fps| <= 2``). The estimate, its residuals and yt-dlp's
   rate from our report are reported, and ``--fps-table-out`` writes the per-video rates for ``convert_uwb_clips.py``.
2. **Pairs.** Each UWB clip is paired with the cue of largest time overlap (IoU >= 0.5, one-to-one).
3. **Temporal alignment.** UWB frame ``j`` is at ``(first + j + L) / fps`` and is matched to our nearest frame within
   half a frame. The lag ``L`` (|L| <= ``--max-lag``) is chosen by a lifter-independent signal: the image-plane angle
   between UWB's raw 2D arm bones (shoulder->elbow, elbow->wrist) and our state's world arm directions projected on the
   image plane; a non-zero lag must beat lag 0 by ``--lag-margin`` degrees (bone directions change slowly, so the
   minimum is shallow: on the first real smoke the best lag gained < 1 deg over lag 0 on most clips). Clips without
   our arms keep ``L = 0`` (method ``none``). Per video, the lag's drift against cue time (frames per minute) exposes a
   wrong rate.
4. **Errors**, on aligned frames where both states hold the quantity, in degrees: upper arms and forearms (world
   bone directions by forward kinematics of the stored rotations: convention-free, since the swing IK reproduces
   every bone direction exactly whatever the chest frame), wrists (the palm frame in the camera frame), fingers (bone
   directions in the palm frame, chained from the stored local rotations); also arms and wrists after one best
   global rotation per clip. Medians, p75, p90, means, and a 95% CI of the median by resampling videos.
   Methods: the z-free regressors with their visibility head (``learned``) and with the inside-image visibility rule
   (``learned-inside``), and the geometric pseudo-world ``(x, y / a, 0)`` with the inside-image rule (``pseudo``), on
   each method's own valid frames and on frames all of them and ours hold (``common``).
5. **Validity agreement** per side (both / ours only / UWB only / neither) for hands, and for forearms on frames
   where our chest is valid; and a **mirror check**: the arm error if UWB's video were the mirror image of ours (UWB
   left arm against our right arm reflected in x).
6. Head rotation (``--head affine``): the scaled-orthographic estimate against our face-geometry head rotation.

Our states were solved by the original pipeline (chest from both hips; no upright chest): their arms exist only where
the hips are seen, so arm errors come from the wider-framed videos. That is a property of the reference, recorded.
Reads only; writes the report locally (``--out``) and optionally the fps table.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
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
ARM_BONES = {"LeftUpperArm": (11, 13), "LeftForearm": (13, 15), "RightUpperArm": (12, 14), "RightForearm": (14, 16)}
GROUPS = ("upper_arms", "forearms", "wrists", "fingers")
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


def fit_rate(rows: np.ndarray) -> dict | None:
    """Least-squares ``frame = fps t + c`` with outliers (> OUTLIER_FRAMES) dropped, at most 3 refits."""
    keep = np.ones(len(rows), bool)
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
                residual_p90=round(float(np.percentile(r, 90)), 3))


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
    return dict(best=order[0], hits=scores[order[0]], runner_up=order[1], runner_up_hits=scores[order[1]])


def estimate_fps(clips: list[dict], cues: list[dict], *, min_text_matches: int = 3) -> dict:
    """Per-video rate estimate: ``fps`` (snapped or raw) or None, with how it was found."""
    rows = text_pairs(clips, cues)
    out = dict(text_matches=len(rows) // 2, fps=None, method=None)
    if len(rows) // 2 >= min_text_matches:
        fit = fit_rate(rows)
        if fit is not None:
            fps, snapped = snap_fps(fit["raw"])
            return dict(out, **fit, fps=fps, snapped_to=snapped, method="text-matched-fit")
    grid = grid_rate(clips, cues)
    out["grid"] = grid
    if grid["hits"] >= min_text_matches and grid["hits"] > grid["runner_up_hits"]:
        out.update(fps=grid["best"], snapped_to=f"{grid['best']:g}", method="time-grid")
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
        for k, joint in enumerate(finger_joints(side)):
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
               margin: float = 0.) -> dict:
    """The lag minimising the median image-plane arm angle between UWB's 2D bones and our projected directions; a
    non-zero lag must beat lag 0 by ``margin`` degrees (bone directions change slowly, so the minimum is shallow and
    a small gain is noise). ``ours``: label -> ((T, 3) y-up world directions, valid); ``uwb2d``: label -> ((F, 2) image
    vectors, valid)."""
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
    return dict(lag=int(lag), method="arm-2d", score=round(scores[lag][0], 3), frames=scores[lag][1],
                best_lag=int(best), best_score=round(scores[best][0], 3),
                lag0_score=round(scores[0][0], 3) if 0 in scores else None)


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


def hand_agreement(ours_state, uwb_state, i, j) -> dict:
    out = {}
    for side, name in (("Left", "left_hand"), ("Right", "right_hand")):
        a = ours_state.channels[name].valid.any(axis=1)[i]
        b = uwb_state.channels[name].valid.any(axis=1)[j]
        out[side] = dict(both=int((a & b).sum()), ours_only=int((a & ~b).sum()), uwb_only=int((~a & b).sum()),
                         neither=int((~a & ~b).sum()))
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
        out[side] = dict(both=int((x & y).sum()), ours_only=int((x & ~y).sum()), uwb_only=int((~x & y).sum()),
                         neither=int((~x & ~y).sum()))
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


def evaluate_pair(ours_state, uwb_states: dict, uwb_clip, fps: float, max_lag: int, head: bool = False,
                  cap: int = 0, lag_margin: float = 0.) -> dict:
    """Errors of each UWB method's state against ours on the aligned frames. ``uwb_states``: method -> state.
    Every error list is a float32 array, subsampled to ``cap`` values per group when ``cap`` > 0."""
    from signtranslator.data_engineering.uwb_keypoints import _found
    times = ours_state.timestamps
    ours = state_items(ours_state)
    pose_found = _found(uwb_clip.pose)
    alignment = choose_lag(times, uwb_clip.first, uwb_clip.frames, fps, ours,
                           image_arm_directions(uwb_clip.pose, pose_found), max_lag, margin=lag_margin)
    i, j = frame_map(times, uwb_clip.first, uwb_clip.frames, fps, alignment["lag"])
    items = {m: state_items(s) for m, s in uwb_states.items()}
    support = {label: np.all([items[m][label][1][j] for m in items], axis=0) for label in ours}
    result = dict(alignment=alignment, aligned_frames=int(len(i)), our_frames=int(len(times)),
                  uwb_frames=int(uwb_clip.frames), methods={})
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
        seed = zlib_seed(uwb_clip.clip_id)
        result["methods"][method] = dict(
            own={g: _capped(v, cap, seed) for g, v in own.items()},
            common={g: _capped(v, cap, seed) for g, v in common.items()},
            rotated={g: _capped(v, cap, seed) for g, v in aligned.items() if g in ("upper_arms", "forearms", "wrists")},
            arms=_capped(own.get("upper_arms", []) + own.get("forearms", []), cap, seed),
            mirror_arms=_capped(mirror.get("upper_arms", []) + mirror.get("forearms", []), cap, seed),
            hands=hand_agreement(ours_state, uwb_states[method], i, j),
            forearms_valid=arm_agreement(ours_state, uwb_states[method], i, j))
        if head:
            result["methods"][method]["head"] = _capped(head_errors(ours_state, uwb_states[method], i, j), cap, seed)
    return result


def zlib_seed(text: str) -> int:
    import zlib
    return zlib.crc32(text.encode())


# ----------------------------------------------------------------------------------------------------------------------
# Summaries

def stat(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), median=round(float(np.median(v)), 2), p75=round(float(np.percentile(v, 75)), 2),
                p90=round(float(np.percentile(v, 90)), 2), mean=round(float(v.mean()), 2))


def clustered_median_ci(by_video: dict, *, resamples: int = 200, seed: int = 0) -> list | None:
    """95% interval of the pooled median, resampling videos with replacement."""
    keys = [k for k, v in by_video.items() if len(v)]
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    medians = []
    for _ in range(resamples):
        pick = rng.integers(0, len(keys), len(keys))
        medians.append(np.median(np.concatenate([by_video[keys[k]] for k in pick])))
    return [round(float(np.percentile(medians, 2.5)), 2), round(float(np.percentile(medians, 97.5)), 2)]


def lag_drift(rows: list[tuple[float, int]]) -> float | None:
    """Least-squares slope of lag (frames) against cue time, in frames per minute; None with fewer than 3 cues."""
    if len(rows) < 3:
        return None
    t, lag = np.array(rows, dtype=np.float64).T
    if np.ptp(t) <= 0:
        return None
    return round(float(np.polyfit(t, lag, 1)[0] * 60.), 3)


def summarise(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("status") == "evaluated"]
    methods = sorted({m for r in ok for m in r["methods"]})
    summary = dict(pairs=len(ok), videos=len({r["video_id"] for r in ok}), methods={})
    for m in methods:
        entry = {}
        for scope in ("own", "common", "rotated"):
            entry[scope] = {}
            for g in GROUPS:
                by_video = collections.defaultdict(list)
                for r in ok:
                    by_video[r["video_id"]].append(np.asarray(r["methods"][m][scope].get(g, ()), dtype=np.float64))
                arrays = {k: np.concatenate(v) for k, v in by_video.items()}
                pooled = np.concatenate(list(arrays.values())) if arrays else np.zeros(0)
                if len(pooled):
                    entry[scope][g] = dict(stat(pooled), median_ci95_by_video=clustered_median_ci(arrays))
        for field, name in (("hands", "hand_valid"), ("forearms_valid", "forearm_valid_where_our_chest_is")):
            counts = collections.Counter()
            for r in ok:
                for side, cells in r["methods"][m][field].items():
                    for k, v in cells.items():
                        counts[f"{side}.{k}"] += v
            for side in ("Left", "Right"):
                total = sum(counts[f"{side}.{k}"] for k in ("both", "ours_only", "uwb_only", "neither"))
                entry.setdefault(name, {})[side] = dict(
                    {k: counts[f"{side}.{k}"] for k in ("both", "ours_only", "uwb_only", "neither")},
                    agreement=round((counts[f"{side}.both"] + counts[f"{side}.neither"]) / total, 4) if total else None)
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
    lags = collections.Counter(r["alignment"]["lag"] for r in ok)
    summary["alignment"] = dict(methods=dict(collections.Counter(r["alignment"]["method"] for r in ok)),
                                lags={str(k): v for k, v in sorted(lags.items())})
    by_video = collections.defaultdict(list)
    for r in ok:
        if r["alignment"]["method"] == "arm-2d":
            by_video[r["video_id"]].append((r["cue_start"], r["alignment"]["lag"]))
    drifts = {v: lag_drift(rows) for v, rows in by_video.items()}
    summary["lag_drift_frames_per_minute"] = {v: d for v, d in drifts.items() if d is not None}
    summary["refusals"] = dict(collections.Counter(r.get("reason", "?")[:60] for r in results
                                                   if r.get("status") != "evaluated"))
    return summary


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


def _init(args):
    import torch
    torch.set_num_threads(1)
    from train_text_to_motion import make_store
    from signtranslator.data_engineering.world_regressor import FEATURES_XY, WorldRegressors
    store = make_store(args.store)
    payload = Path(args.regressors).read_bytes() if Path(args.regressors).exists() else store.get(args.regressors)
    regressors = WorldRegressors.load(payload, expected_sha256=args.regressors_sha256)
    if regressors.features != FEATURES_XY:
        raise ValueError(f"the bundle's features are {regressors.features!r}; UWB needs {FEATURES_XY!r}")
    _WORKER.update(args=args, store=store, archives={}, regressors=regressors)
    from signtranslator.data_engineering.semlex_pose import BlendshapeModel, load_face_resources
    resources = load_face_resources(Path(args.models_dir) / "face_landmarker.task") if args.head == "affine" else None
    _WORKER["resources"] = resources
    _WORKER["blendshapes"] = BlendshapeModel(resources) if resources is not None else None


def _evaluate(job: dict) -> dict:
    from signtranslator.data_engineering import uwb_keypoints as uk
    from signtranslator.pose.multichannel import MultichannelMotion
    args, store = _WORKER["args"], _WORKER["store"]
    base = {k: job[k] for k in ("video_id", "clip_id", "sample_id", "cue_start", "iou", "fps", "zip")}
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
        states = {"learned": uk.convert_clip(clip, job["fps"], regressors=_WORKER["regressors"], **common)[0],
                  "learned-inside": uk.convert_clip(clip, job["fps"], regressors=_WORKER["regressors"],
                                                    visibility="inside", **common)[0],
                  "pseudo": uk.convert_clip(clip, job["fps"], world="pseudo", visibility="inside", **common)[0]}
        result = evaluate_pair(ours, states, clip, job["fps"], args.max_lag, head=args.head == "affine",
                               cap=args.max_values_per_pair, lag_margin=args.lag_margin)
    except (uk.Refusal, ValueError, KeyError) as error:
        return dict(base, status="refused", reason=f"{type(error).__name__}: {error}"[:200])
    return dict(base, status="evaluated", member_sha256=sha, **result)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default="r2")
    p.add_argument("--uwb-prefix", default=UWB_PREFIX)
    p.add_argument("--youtube-prefix", default=YOUTUBE_PREFIX)
    p.add_argument("--translations", default="YT.translations.all.json", help="UWB translations file under the prefix")
    p.add_argument("--regressors", help="z-free bundle (local path or store key); not needed with --fps-only")
    p.add_argument("--regressors-sha256")
    p.add_argument("--models-dir", default=os.path.expanduser("~/.sign-translator-ops/models"))
    p.add_argument("--head", choices=("none", "affine"), default="none")
    p.add_argument("--members-cache", help="local JSON cache of the zips' member lists")
    p.add_argument("--videos", type=int, default=0, help="evaluate pairs from at most this many videos (0: all)")
    p.add_argument("--pairs-per-video", type=int, default=5)
    p.add_argument("--max-lag", type=int, default=6)
    p.add_argument("--max-values-per-pair", type=int, default=600, help="per group and scope (seeded subsample)")
    p.add_argument("--lag-margin", type=float, default=.5,
                   help="degrees a non-zero lag must gain over lag 0 (the image-plane arm angle)")
    p.add_argument("--min-text-matches", type=int, default=3)
    p.add_argument("--fps-only", action="store_true", help="estimate rates for every overlapping video; no states")
    p.add_argument("--fps-table-out", help="write {video: fps} for convert_uwb_clips.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if not args.fps_only and not (args.regressors and args.regressors_sha256):
        raise SystemExit("--regressors and --regressors-sha256 are required (or --fps-only)")
    from train_text_to_motion import make_store
    store = make_store(args.store)
    uwb = json.loads(store.get(f"{args.uwb_prefix}/{args.translations}"))
    reports = sorted(k for k in store.list(f"{args.youtube_prefix}/videos/") if k.endswith(".json"))
    ids = [Path(k).name[:-5] for k in reports if Path(k).name[:-5] in uwb]
    rng = random.Random(args.seed)
    if args.videos and not args.fps_only:          # a sample: read 3x as many reports as videos wanted, seeded
        ids = sorted(rng.sample(ids, min(len(ids), 3 * args.videos)))
    fps_rows, jobs = {}, []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        loaded = list(pool.map(_load_report, [(args.store, f"{args.youtube_prefix}/videos/{v}.json") for v in ids],
                               chunksize=16))
    for video_id, report in zip(ids, loaded):
        if report is None or report.get("status") != "converted":
            continue
        clips = uwb_clips_of(uwb[video_id])
        cues = report["segments"]
        estimate = estimate_fps(clips, cues, min_text_matches=args.min_text_matches)
        meta = (report.get("meta") or {}).get("fps")
        estimate.update(meta_fps=meta, uwb_clips=len(clips), our_cues=len(cues))
        fps_rows[video_id] = estimate
        if estimate["fps"] is None:
            continue
        pairs = pair_clips(clips, cues, estimate["fps"])
        estimate["pairs"] = len(pairs)
        for clip, cue, iou in pairs:
            jobs.append(dict(video_id=video_id, clip_id=clip["clip_id"], sample_id=cue["sample_id"],
                             cue_start=cue["start"], iou=iou, fps=estimate["fps"], state_key=cue["state_key"],
                             state_sha256=cue["state_sha256"]))
    fps_summary = summarise_fps(fps_rows)
    print(json.dumps(fps_summary), flush=True)
    if args.fps_table_out:
        table = {v: r["fps"] for v, r in sorted(fps_rows.items()) if r["fps"] is not None}
        Path(args.fps_table_out).write_text(json.dumps(dict(
            schema="uwb-fps-table-v1", method="uwb_lifter_gate.estimate_fps (text-matched caption cues, snapped)",
            fps=table), indent=1, sort_keys=True))
    out = dict(settings={k: v for k, v in vars(args).items()}, fps=dict(summary=fps_summary, videos=fps_rows),
               candidate_pairs=len(jobs))
    if not args.fps_only:
        by_video = collections.defaultdict(list)
        for job in jobs:
            by_video[job["video_id"]].append(job)
        videos = sorted(by_video)
        rng.shuffle(videos)
        videos = videos[:args.videos or None]
        index = members_index(store, args.uwb_prefix, Path(args.members_cache) if args.members_cache else None)
        chosen = []
        for v in videos:
            picked = rng.sample(by_video[v], min(args.pairs_per_video, len(by_video[v])))
            for job in picked:
                if job["clip_id"] in index:
                    chosen.append(dict(job, zip=index[job["clip_id"]]))
        print(json.dumps(dict(videos=len(videos), pairs=len(chosen))), flush=True)
        with ProcessPoolExecutor(max_workers=max(1, args.workers), initializer=_init, initargs=(args,)) as pool:
            results = list(pool.map(_evaluate, chosen, chunksize=1))
        out["regressors_sha256"] = args.regressors_sha256
        out["summary"] = summarise(results)
        out["pairs"] = [{k: v for k, v in r.items() if k != "methods"} for r in results]
        print(json.dumps(out["summary"], indent=1), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


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
    return dict(videos=len(rows), estimated=len(estimated), methods={str(k): v for k, v in methods.items()},
                with_meta=len(with_meta), agree_with_meta_within_ntsc=agree, equal_to_meta=exact,
                estimated_rates=dict(counts.most_common()), meta_rates=dict(meta.most_common()),
                share_not_about_30=round(sum(abs(f - 30) > .1 and abs(f - 29.97) > .01 for f in estimated)
                                         / max(len(estimated), 1), 4),
                residual_median_of_medians=round(float(np.median(residuals)), 3) if residuals else None)


if __name__ == "__main__":
    raise SystemExit(main())
