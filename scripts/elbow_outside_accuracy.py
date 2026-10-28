"""How accurate is the arm our extraction solves when MediaPipe places the elbow outside the image? (local, CPU)

Clips: How2Sign val front clips with How2Sign two-view states (the truth; Sapiens two-view triangulation), cropped to
Sem-Lex framing (``validate_semlex_adapter.py`` crops at 4:3, 16:9 and 1:1: nose at 0.35, shoulders at 0.67 of the
crop height) plus tighter crops (``extract`` stage: nose at 0.30, shoulders at 0.80, 4:3 and 16:9) that put the elbow
below the frame more often.

Our path: ``extract_clip`` landmarks (Tasks trackers, ROI mode) solved by ``solve_rotations`` with ``upright_chest``
and MediaPipe's own visibility. To measure arms the visibility rule would drop (category iii), the arm joints are also
solved with the elbow and wrist taken as visible ("ungated"); on frames where MediaPipe's visibility is >= 0.5 this is
the stored arm exactly.

Per frame and side, against the two-view state (camera-aligned frames by forward kinematics): angle errors of the upper
arm and forearm directions and of the wrist (palm) frame, split by the elbow's image landmark and visibility:
(i) inside the crop with visibility >= 0.5; (ii) outside with visibility >= 0.5 (exactly the frames the elbow gate
removes); (iii) inside with visibility < 0.5. Category (ii) is also split by how far outside the elbow lies (in frame
heights: 0-0.1, 0.1-0.3, > 0.3), with the share of its frames whose wrist landmark is inside the crop and whose hand
was detected. The (ii) - (i) differences of medians carry bootstrap 95% intervals that resample source videos.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

TIGHT = dict(nose_y=.30, shoulder_y=.80)
SIDES = {"Left": (13, 15), "Right": (14, 16)}            # elbow, wrist (BlazePose)
QUANTITIES = ("UpperArm", "Forearm", "Hand")
DISTANCE_BINS = ((0., .1), (.1, .3), (.3, np.inf))


def extract_tight(job):
    video, shape, out_dir, models_dir = job
    from signtranslator.data_engineering.video_landmarks import extract_clip, verify_models
    from validate_semlex_adapter import CLIP_FIELDS, SHAPES, framing, write_crop
    name = f"{Path(video).stem}__{shape.replace(':', 'x')}-tight"
    target = Path(out_dir) / f"{name}.npz"
    if target.exists():
        return dict(name=name, status="present")
    models = verify_models(Path(models_dir))
    box = framing(Path(video), models, SHAPES[shape], **TIGHT)
    if box is None:
        return dict(name=name, status="framing-does-not-fit")
    with tempfile.TemporaryDirectory() as t:
        cropped = Path(t) / "crop.mp4"
        write_crop(Path(video), box, cropped)
        clip = extract_clip(cropped, models, roi=True)
    meta = dict(name=name, video=Path(video).name, shape=f"{shape}-tight", box=list(box),
                size=[int(box[2]), int(box[3])], names=list(clip.blendshape_names))
    np.savez_compressed(target, meta=np.array(json.dumps(meta)),
                        **{f"ref_{k}": np.asarray(getattr(clip, k)) for k in CLIP_FIELDS})
    return dict(name=name, status="extracted", frames=len(clip.times))


def outside_distance(xy: np.ndarray, aspect: float) -> np.ndarray:
    """Distance (frame heights) of normalised image points from the image rectangle; 0 inside."""
    dx = np.maximum(np.maximum(-xy[..., 0], xy[..., 0] - 1.), 0.) * aspect
    dy = np.maximum(np.maximum(-xy[..., 1], xy[..., 1] - 1.), 0.)
    return np.hypot(dx, dy)


def solved_world(clip, *, ungated: bool):
    """World frames of the arm joints and the hand frames (y-up, camera-aligned) from our path, upright chest."""
    from signtranslator.data_engineering.video_landmarks import (VISIBILITY_THRESHOLD, Y_UP, assign_hands,
                                                                 solve_rotations)
    from signtranslator.data_engineering.landmark_kinematics import frame_from_axes
    world = clip.pose_world @ Y_UP.T
    visible = (clip.pose_visibility >= VISIBILITY_THRESHOLD) & clip.pose_found[:, None]
    if ungated:
        visible = visible.copy()
        visible[:, [13, 14, 15, 16]] = clip.pose_found[:, None]
    hands = {s: (lm @ Y_UP.T, ok) for s, (lm, ok) in assign_hands(clip).items()}
    rotations, _ = solve_rotations(world, visible, hands, upright_chest=True)
    # forward kinematics from the chest's world frame (upright chest: the chest frame is its world frame)
    from signtranslator.data_engineering.video_landmarks import L_SHOULDER, R_SHOULDER
    chest, chest_ok = frame_from_axes(world[:, L_SHOULDER] - world[:, R_SHOULDER],
                                      np.broadcast_to([0., 1., 0.], world[:, 0].shape))
    chest_ok &= visible[:, [L_SHOULDER, R_SHOULDER]].all(1)
    out = {}
    for side in SIDES:
        upper, upper_ok = chest @ rotations[f"{side}UpperArm"][0], chest_ok & rotations[f"{side}UpperArm"][1]
        fore, fore_ok = upper @ rotations[f"{side}Forearm"][0], upper_ok & rotations[f"{side}Forearm"][1]
        hand, hand_ok = fore @ rotations[f"{side}Hand"][0], fore_ok & rotations[f"{side}Hand"][1]
        rest = np.array([1., 0., 0.]) if side == "Left" else np.array([-1., 0., 0.])
        out[side] = dict(UpperArm=(upper @ rest, upper_ok), Forearm=(fore @ rest, fore_ok), Hand=(hand, hand_ok),
                         hand_detected=hands[side][1])
    return out


def truth_world(state):
    from validate_semlex_adapter import world_frames
    w = world_frames(state)
    out = {}
    for side in SIDES:
        rest = np.array([1., 0., 0.]) if side == "Left" else np.array([-1., 0., 0.])
        out[side] = dict(UpperArm=(w[f"{side}UpperArm"][0] @ rest, w[f"{side}UpperArm"][1]),
                         Forearm=(w[f"{side}Forearm"][0] @ rest, w[f"{side}Forearm"][1]),
                         Hand=w[f"{side}Hand"])
    return out


def clip_rows(meta, clip, tv, frames) -> list[dict]:
    """One row per (frame, side) with the category, distance, errors (deg, NaN where unsolved) and context."""
    from validate_semlex_adapter import angles, direction_angles
    aspect = meta["size"][0] / meta["size"][1]
    gated, ungated, truth = solved_world(clip, ungated=False), solved_world(clip, ungated=True), truth_world(tv)
    rows = []
    for side, (elbow, wrist) in SIDES.items():
        xy = clip.pose_image[frames]
        vis = clip.pose_visibility[frames, elbow]
        distance = outside_distance(xy[:, elbow], aspect)
        wrist_inside = outside_distance(xy[:, wrist], aspect) == 0
        found = clip.pose_found[frames]
        errors = {}
        for q in QUANTITIES:
            ours, ok = ungated[side][q]
            truth_value, truth_ok = truth[side][q]
            ours, ok = ours[frames], ok[frames]
            fn = direction_angles if q != "Hand" else angles
            err = np.where(ok & truth_ok, fn(ours, truth_value), np.nan)
            errors[q] = err
            gated_ok = gated[side][q][1][frames]
            errors[f"{q}_stored"] = gated_ok
        for t in range(len(frames)):
            if not found[t]:
                continue
            inside = distance[t] == 0
            category = ("i" if inside else "ii") if vis[t] >= .5 else ("iii" if inside else "iv")
            rows.append(dict(video=meta["video"], clip=meta["name"], side=side, category=category,
                             distance=float(distance[t]), wrist_inside=bool(wrist_inside[t]),
                             hand_detected=bool(gated[side]["hand_detected"][frames][t]),
                             **{q: float(errors[q][t]) for q in QUANTITIES},
                             **{f"{q}_stored": bool(errors[f"{q}_stored"][t]) for q in QUANTITIES}))
    return rows


def stat(v) -> dict:
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), median=round(float(np.median(v)), 2), p90=round(float(np.percentile(v, 90)), 2))


def difference_ci(rows, quantity, a="ii", b="i", draws=2000, seed=0):
    """Bootstrap 95% interval of median(a) - median(b), resampling source videos with replacement."""
    rng = np.random.default_rng(seed)
    videos = sorted({r["video"] for r in rows})
    by = {v: [r for r in rows if r["video"] == v] for v in videos}

    def med(sample, c):
        v = [r[quantity] for r in sample if r["category"] == c and np.isfinite(r[quantity])]
        return np.median(v) if v else np.nan
    point = med(rows, a) - med(rows, b)
    diffs = []
    for _ in range(draws):
        sample = [r for v in rng.choice(videos, len(videos)) for r in by[v]]
        d = med(sample, a) - med(sample, b)
        if np.isfinite(d):
            diffs.append(d)
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan)
    return dict(difference=round(float(point), 2), ci=[round(float(lo), 2), round(float(hi), 2)], draws=len(diffs))


def summarise(rows) -> dict:
    out = dict(rows=len(rows), clips=len({r["clip"] for r in rows}), videos=len({r["video"] for r in rows}))
    categories = {}
    for c in ("i", "ii", "iii", "iv"):
        sel = [r for r in rows if r["category"] == c]
        categories[c] = dict(frames=len(sel), **{q: stat([r[q] for r in sel]) for q in QUANTITIES},
                             stored_share={q: round(float(np.mean([r[f"{q}_stored"] for r in sel])), 4) if sel else None
                                           for q in QUANTITIES})
    out["categories"] = categories
    ii = [r for r in rows if r["category"] == "ii"]
    out["ii_by_distance"] = {f"{lo}-{hi}": dict(frames=len(s), wrist_inside=round(float(np.mean([r["wrist_inside"] for r in s])), 4)
                                                if s else None,
                                                hand_detected=round(float(np.mean([r["hand_detected"] for r in s])), 4)
                                                if s else None, **{q: stat([r[q] for r in s]) for q in QUANTITIES})
                             for lo, hi in DISTANCE_BINS
                             for s in [[r for r in ii if lo < r["distance"] <= hi]]}
    out["ii_context"] = dict(wrist_inside=round(float(np.mean([r["wrist_inside"] for r in ii])), 4) if ii else None,
                             hand_detected=round(float(np.mean([r["hand_detected"] for r in ii])), 4) if ii else None)
    out["ii_minus_i_median"] = {q: difference_ci(rows, q) for q in QUANTITIES}
    out["iii_minus_i_median"] = {q: difference_ci(rows, q, a="iii") for q in QUANTITIES}
    out["verdict_ii_within_5deg_of_i_at_median"] = {q: bool(abs(out["ii_minus_i_median"][q]["difference"]) <= 5.)
                                                    for q in QUANTITIES}
    return out


def analyse(args) -> dict:
    from signtranslator.pose.multichannel import MultichannelMotion
    from validate_semlex_adapter import load_npz, two_view_frames
    index = json.loads((Path(args.two_view_dir) / "index.json").read_text())
    tv = {f"{k}-rgb_front.mp4": MultichannelMotion.load(Path(args.two_view_dir) / f"{k}.npz", expected_sha256=e["sha256"])
          for k, e in index.items()}
    full, crops = {}, []
    for directory in args.crop_dirs:
        for path in sorted(Path(directory).glob("*.npz")):
            meta, clip, _ = load_npz(path)
            (full.__setitem__(meta["video"], clip) if meta["shape"] == "full" else crops.append((meta, clip)))
    rows, skipped = [], []
    for meta, clip in crops:
        state, mf = tv.get(meta["video"]), full.get(meta["video"])
        if state is None or mf is None or len(mf.times) != len(clip.times):
            skipped.append(meta["name"])
            continue
        frames = two_view_frames(state, mf.times)
        if frames is None:
            skipped.append(meta["name"])
            continue
        rows.extend(clip_rows(meta, clip, state, frames))
    report = dict(all=summarise(rows), skipped=skipped,
                  by_framing={name: summarise([r for r in rows if r["clip"].endswith(name)])
                              for name in ("tight", "4x3", "16x9", "1x1")})
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=("extract", "analyse"))
    p.add_argument("--videos", nargs="*", default=[])
    p.add_argument("--out-dir")
    p.add_argument("--crop-dirs", nargs="*", default=[])
    p.add_argument("--two-view-dir")
    p.add_argument("--models-dir", default=os.path.expanduser("~/.sign-translator-ops/models"))
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--report")
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    if args.stage == "extract":
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        jobs = [(v, s, args.out_dir, args.models_dir) for v in args.videos for s in ("4:3", "16:9")]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(extract_tight, jobs):
                print(json.dumps(r), flush=True)
        return 0
    report = analyse(args)
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report["all"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
