"""Visibility rules for the UWB lifter, measured on How2Sign two-view truth with arms cut by the frame (local, CPU).

The question (review of e186d62, item 1): YouTube framings cut the arms, so a visibility rule decides how many arm
frames survive. Which rule keeps arms that are accurate, and how many does it keep compared with MediaPipe's own
visibility?

**Data** (the ``elbow_outside_accuracy.py`` set): How2Sign val front clips with How2Sign two-view states (the truth),
cropped to Sem-Lex framing (``validate_semlex_adapter.py``: 4:3, 16:9, 1:1) and to tighter crops (nose at 0.30,
shoulders at 0.80 of the height), whose stored Tasks image landmarks stand in for UWB's 2D points (pixels = normalised
x size; a Tasks-vs-Holistic difference remains). Only ``--signers`` are read (default How2Sign 1 and 2: the regressors'
validation signers, never trained on); **signer 9 (the generator's final-test signer) is refused outright**.

**Per crop**, the lifter (``uwb_keypoints.convert_clip``, generator flags) runs once per rule and once with every
point visible (``all``: the ungated arm). Per two-view frame and side:

- the elbow's distance outside the crop (frame heights; 0 inside), from the crop's image landmarks;
- MediaPipe's own keep decision: shoulder, elbow and wrist visibility >= 0.5 (our extraction's rule on that crop);
- each rule's keep decision (the forearm is valid in that rule's state);
- the ungated lifter's errors against the two-view state: upper arm and forearm directions, wrist (palm) frame.

**Report per rule**: survival (kept share) overall, inside and outside the frame, and relative to MediaPipe's survival
(the proposed gate asks >= 80% on chest-up framings); median / p90 errors of the frames it keeps and of those it
drops, inside and outside, and by distance outside (0-0.1, 0.1-0.3, > 0.3 frame heights). MediaPipe's rule is scored
the same way for reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

SIDES = {"Left": (11, 13, 15), "Right": (12, 14, 16)}          # shoulder, elbow, wrist (BlazePose)
QUANTITIES = ("UpperArm", "Forearm", "Hand")
DISTANCE_BINS = ((0., .1), (.1, .3), (.3, np.inf))
EXCLUDED_SIGNERS = ("9",)


def signer_of(name: str) -> str | None:
    match = re.search(r"-(\d+)-rgb_front", name)
    return match.group(1) if match else None


def uwb_clip_from_landmarks(clip, size, name: str):
    """A ``UwbClip`` holding a ``ClipLandmarks``' image points in pixels (NaN where not detected); no face."""
    from signtranslator.data_engineering.uwb_keypoints import UwbClip
    width, height = int(size[0]), int(size[1])
    scale = np.array([width, height], dtype=np.float64)
    frames = len(clip.times)
    pose = np.where(clip.pose_found[:, None, None], np.asarray(clip.pose_image)[..., :2] * scale, np.nan)
    hands = np.full((frames, 2, 21, 2), np.nan)
    for k in range(min(2, clip.hands_found.shape[1])):
        hands[:, k] = np.where(clip.hands_found[:, k, None, None], clip.hands_image[:, k] * scale, np.nan)
    return UwbClip(clip_id=f"{name}.000000-{frames - 1:06d}", video_id=name, first=0, last=frames - 1, width=width,
                   height=height, pose=pose, hands=hands, face=np.full((frames, 478, 2), np.nan))


def outside_distance(xy: np.ndarray, aspect: float) -> np.ndarray:
    """Distance (frame heights) of normalised image points from the image rectangle; 0 inside."""
    dx = np.maximum(np.maximum(-xy[..., 0], xy[..., 0] - 1.), 0.) * aspect
    dy = np.maximum(np.maximum(-xy[..., 1], xy[..., 1] - 1.), 0.)
    return np.hypot(dx, dy)


def lifted_world(state):
    """Per side: UpperArm and Forearm world directions and the Hand world frame, with validity, from a state."""
    from validate_semlex_adapter import ARM_REST, world_frames
    w = world_frames(state)
    out = {}
    for side in SIDES:
        rest = np.asarray(ARM_REST[f"{side}UpperArm"])
        out[side] = dict(UpperArm=(w[f"{side}UpperArm"][0] @ rest, w[f"{side}UpperArm"][1]),
                         Forearm=(w[f"{side}Forearm"][0] @ rest, w[f"{side}Forearm"][1]),
                         Hand=w[f"{side}Hand"])
    return out


def clip_rows(meta, clip, truth_state, frames, regressors, rules) -> list[dict]:
    """One row per (two-view frame, side): distance outside, MediaPipe's and each rule's keep decision, ungated
    lifter errors (NaN where the lifter or the truth has no value)."""
    from signtranslator.data_engineering import uwb_keypoints as uk
    from validate_semlex_adapter import angles, direction_angles
    from elbow_outside_accuracy import truth_world
    size = meta["size"]
    uwb = uwb_clip_from_landmarks(clip, size, meta["name"])
    fps = 1. / float(np.median(np.diff(clip.times)))
    common = dict(source_sha256="0" * 64, fps_source="measured", regressors=regressors)
    ungated = lifted_world(uk.convert_clip(uwb, fps, visibility="all", **common)[0])
    kept = {}
    for rule in rules:
        body = uk.convert_clip(uwb, fps, visibility=rule, **common)[0].channels["body"]
        kept[rule] = {side: body.valid[:, body.labels.index(f"{side}Forearm")].astype(bool) for side in SIDES}
    truth = truth_world(truth_state)
    aspect = size[0] / size[1]
    rows = []
    for side, (shoulder, elbow, wrist) in SIDES.items():
        distance = outside_distance(np.asarray(clip.pose_image)[frames, elbow], aspect)
        vis = np.asarray(clip.pose_visibility)[frames][:, [shoulder, elbow, wrist]]
        mediapipe = (vis >= .5).all(1) & np.asarray(clip.pose_found)[frames]
        errors = {}
        for q in QUANTITIES:
            value, ok = ungated[side][q]
            true_value, true_ok = truth[side][q]
            fn = direction_angles if q != "Hand" else angles
            errors[q] = np.where(ok[frames] & true_ok, fn(value[frames], true_value), np.nan)
        for t in range(len(frames)):
            if not clip.pose_found[frames[t]]:
                continue
            rows.append(dict(video=meta["video"], clip=meta["name"], side=side, distance=float(distance[t]),
                             mediapipe=bool(mediapipe[t]), **{f"rule:{r}": bool(kept[r][side][frames[t]]) for r in rules},
                             **{q: float(errors[q][t]) for q in QUANTITIES}))
    return rows


def stat(v) -> dict:
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), median=round(float(np.median(v)), 2), p90=round(float(np.percentile(v, 90)), 2))


def rule_summary(rows: list[dict], key: str, reference: str = "mediapipe") -> dict:
    """Survival and the accuracy of kept and dropped frames for one keep decision."""
    def part(sel):
        if not sel:
            return dict(frames=0)
        keep = [r for r in sel if r[key]]
        drop = [r for r in sel if not r[key]]
        out = dict(frames=len(sel), kept=len(keep), survival=round(len(keep) / len(sel), 4),
                   kept_errors={q: stat([r[q] for r in keep]) for q in QUANTITIES},
                   dropped_errors={q: stat([r[q] for r in drop]) for q in QUANTITIES})
        ref = sum(r[reference] for r in sel)
        out["survival_vs_mediapipe"] = round(len(keep) / ref, 4) if ref else None
        return out
    inside = [r for r in rows if r["distance"] == 0]
    outside = [r for r in rows if r["distance"] > 0]
    return dict(all=part(rows), elbow_inside=part(inside), elbow_outside=part(outside),
                outside_by_distance={f"{lo}-{hi}": part([r for r in outside if lo < r["distance"] <= hi])
                                     for lo, hi in DISTANCE_BINS})


def analyse(args, regressors=None) -> dict:
    from signtranslator.data_engineering.world_regressor import WorldRegressors
    from signtranslator.pose.multichannel import MultichannelMotion
    from validate_semlex_adapter import load_npz, two_view_frames
    if regressors is None:
        payload = Path(args.regressors).read_bytes()
        regressors = WorldRegressors.load(payload, expected_sha256=args.regressors_sha256)
    rules = [r for r in args.rules.split(",") if r]
    signers = set(args.signers.split(","))
    if signers & set(EXCLUDED_SIGNERS):
        raise SystemExit("How2Sign signer 9 is the generator's final-test signer and is never read")
    index = json.loads((Path(args.two_view_dir) / "index.json").read_text())
    full, crops, skipped = {}, [], []
    for directory in args.crop_dirs:
        for path in sorted(Path(directory).glob("*.npz")):
            signer = signer_of(path.name)
            if signer in EXCLUDED_SIGNERS:
                raise SystemExit(f"{path.name}: How2Sign signer 9 is never read")
            if signer not in signers:
                continue
            meta, clip, _ = load_npz(path)
            (full.__setitem__(meta["video"], clip) if meta["shape"] == "full" else crops.append((meta, clip)))
    rows = []
    for meta, clip in crops:
        key = meta["video"].replace("-rgb_front.mp4", "")
        entry, mf = index.get(key), full.get(meta["video"])
        if entry is None or mf is None or len(mf.times) != len(clip.times):
            skipped.append(meta["name"])
            continue
        truth = MultichannelMotion.load(Path(args.two_view_dir) / f"{key}.npz", expected_sha256=entry["sha256"])
        frames = two_view_frames(truth, mf.times)
        if frames is None:
            skipped.append(meta["name"])
            continue
        rows.extend(clip_rows(meta, clip, truth, frames, regressors, rules))
    report = dict(regressors_sha256=regressors.sha256, features=regressors.features, signers=sorted(signers),
                  rules=rules, crops=len(crops) - len(skipped), skipped=skipped, rows=len(rows),
                  videos=len({r["video"] for r in rows}), mediapipe=rule_summary(rows, "mediapipe"),
                  by_rule={rule: rule_summary(rows, f"rule:{rule}") for rule in rules})
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--crop-dirs", nargs="+", required=True, help="validate_semlex_adapter / elbow_outside_accuracy crops")
    p.add_argument("--two-view-dir", required=True)
    p.add_argument("--regressors", required=True)
    p.add_argument("--regressors-sha256", required=True)
    p.add_argument("--rules", default="learned,inside,margin:0.1,margin:0.25,margin:0.5")
    p.add_argument("--signers", default="1,2", help="How2Sign signers read (the regressors' validation signers)")
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    report = analyse(args)
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(dict(rows=report["rows"], videos=report["videos"], mediapipe=report["mediapipe"]["all"],
                          by_rule={r: v["all"] for r, v in report["by_rule"].items()}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
