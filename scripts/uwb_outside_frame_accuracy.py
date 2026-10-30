"""Visibility rules for the UWB lifter, measured on How2Sign two-view truth with arms cut by the frame (local, CPU).

The question (reviews of e186d62 and 6625c83, item 1): YouTube framings cut the arms, so a visibility rule decides how
many arm frames survive. Which rule keeps arms that are accurate, and how many does it keep compared with MediaPipe's
own visibility, weighted the way UWB's frames are actually cut?

**Data** (the ``elbow_outside_accuracy.py`` set): How2Sign val front clips with How2Sign two-view states (the truth),
cropped to Sem-Lex framing (``validate_semlex_adapter.py``: 4:3, 16:9, 1:1) and to tighter crops (nose at 0.30,
shoulders at 0.80 of the height). **Inputs**: legacy MediaPipe Holistic 0.10.14 (``legacy_holistic_arrays.py`` run on
the same crop boxes; ``--legacy-dir``, preferred: UWB's own kind of landmarks) where an array of the crop exists with
the crop's frame count, otherwise the crop's stored Tasks image landmarks; every row records which input it used.
Only ``--signers`` are read (default How2Sign 1 and 2: the regressors' validation signers, never trained on);
**signer 9 (the generator's final-test signer) is refused outright**.

**Per crop**, the lifter (``uwb_keypoints.convert_clip``, generator flags) runs once per rule and once with every
point visible (``all``: the ungated arm). Per two-view frame and side:

- the elbow's distance outside the crop (frame heights; 0 inside), from the input's own image points;
- MediaPipe's keep decision: Tasks shoulder, elbow and wrist visibility >= 0.5 (our extraction's rule on that crop);
- each rule's keep decision (the forearm is valid in that rule's state);
- the ungated lifter's errors against the two-view state: upper arm and forearm directions, wrist (palm) frame.

**Report** per rule and for MediaPipe, per side and per bone (upper arm, forearm), per elbow bin (inside, outside by
0-0.1, 0.1-0.3, > 0.3 frame heights): survival, and the kept frames' median and p90 error of the ungated lifter, each
with the upper bound of a 95% interval from resampling videos. **UWB-weighted survival**: the per-bin survivals
weighted by UWB's own elbow-distance distribution (``--uwb-weights``: ``uwb_lifter_gate.py``'s
``summary.uwb_elbow_distance``, or ``{side: {bin: share}}``), against MediaPipe's survival weighted the same way, over
the bins with at least ``--min-bin-frames`` frames; ``covered`` says how much of UWB's weight those bins hold. These
crops cut the elbow on 97% of frames while UWB's elbow is inside the frame on about 75%, so the inside bin is thin and
the weighted survival must be read with its coverage.
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
BONES = ("UpperArm", "Forearm")
QUANTITIES = ("UpperArm", "Forearm", "Hand")
BINS = (("inside", 0., 0.), ("0-0.1", 0., .1), ("0.1-0.3", .1, .3), ("0.3-inf", .3, np.inf))
DEFAULT_RULES = "learned,inside,margin:0.1,margin:0.25"
EXCLUDED_SIGNERS = ("9",)
LEGACY_INPUT = "legacy-holistic-0.10.14"
TASKS_INPUT = "tasks-image"


def signer_of(name: str) -> str | None:
    match = re.search(r"-(\d+)-rgb_front", name)
    return match.group(1) if match else None


def bin_of(distance: float) -> str:
    for name, lo, hi in BINS:
        if (distance == 0 if hi == 0 else lo < distance <= hi):
            return name
    return BINS[-1][0]


def uwb_clip_from_points(pose_xy, hands_xy, found_pose, found_hands, size, name: str):
    """A ``UwbClip`` from normalised image points (pixels = normalised x size; NaN where not detected); no face."""
    from signtranslator.data_engineering.uwb_keypoints import UwbClip
    width, height = int(size[0]), int(size[1])
    scale = np.array([width, height], dtype=np.float64)
    frames = len(pose_xy)
    pose = np.where(found_pose[:, None, None], np.asarray(pose_xy, dtype=np.float64)[..., :2] * scale, np.nan)
    hands = np.full((frames, 2, 21, 2), np.nan)
    for k in range(min(2, hands_xy.shape[1])):
        hands[:, k] = np.where(found_hands[:, k, None, None], np.asarray(hands_xy[:, k], dtype=np.float64)[..., :2] * scale,
                               np.nan)
    return UwbClip(clip_id=f"{name}.000000-{frames - 1:06d}", video_id=name, first=0, last=frames - 1, width=width,
                   height=height, pose=pose, hands=hands, face=np.full((frames, 478, 2), np.nan))


def input_points(meta, clip, legacy_dir: Path | None):
    """(pose (T, 33, 2), hands (T, 2, 21, 2), pose found, hands found, input name) of a crop: legacy Holistic when its
    array exists with the crop's frame count and size, else the crop's Tasks image landmarks."""
    if legacy_dir is not None:
        path = Path(legacy_dir) / f"{meta['name']}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                holistic, size = z["holistic"].astype(np.float64), z["size"].tolist()
            if len(holistic) == len(clip.times) and list(size) == list(meta["size"]):
                pose = holistic[:, 478:511, :2]
                hands = np.stack((holistic[:, 511:532, :2], holistic[:, 532:553, :2]), axis=1)
                found_pose = np.isfinite(pose).all((1, 2))
                found_hands = np.isfinite(hands).all((2, 3))
                return np.nan_to_num(pose), np.nan_to_num(hands), found_pose, found_hands, LEGACY_INPUT
    return (np.asarray(clip.pose_image)[..., :2], np.asarray(clip.hands_image)[..., :2],
            np.asarray(clip.pose_found, bool), np.asarray(clip.hands_found, bool), TASKS_INPUT)


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


def clip_rows(meta, clip, truth_state, frames, regressors, rules, legacy_dir=None) -> list[dict]:
    """One row per (two-view frame, side): input, distance outside, MediaPipe's and each rule's keep decision,
    ungated lifter errors (NaN where the lifter or the truth has no value)."""
    from signtranslator.data_engineering import uwb_keypoints as uk
    from validate_semlex_adapter import angles, direction_angles
    from elbow_outside_accuracy import truth_world
    size = meta["size"]
    pose_xy, hands_xy, found_pose, found_hands, source = input_points(meta, clip, legacy_dir)
    uwb = uwb_clip_from_points(pose_xy, hands_xy, found_pose, found_hands, size, meta["name"])
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
        distance = outside_distance(pose_xy[frames, elbow], aspect)
        vis = np.asarray(clip.pose_visibility)[frames][:, [shoulder, elbow, wrist]]
        mediapipe = (vis >= .5).all(1) & np.asarray(clip.pose_found)[frames]
        errors = {}
        for q in QUANTITIES:
            value, ok = ungated[side][q]
            true_value, true_ok = truth[side][q]
            fn = direction_angles if q != "Hand" else angles
            errors[q] = np.where(ok[frames] & true_ok, fn(value[frames], true_value), np.nan)
        for t in range(len(frames)):
            if not found_pose[frames[t]]:
                continue
            rows.append(dict(video=meta["video"], clip=meta["name"], side=side, input=source,
                             distance=float(distance[t]), bin=bin_of(float(distance[t])), mediapipe=bool(mediapipe[t]),
                             **{f"rule:{r}": bool(kept[r][side][frames[t]]) for r in rules},
                             **{q: float(errors[q][t]) for q in QUANTITIES}))
    return rows


def _bootstrap_upper(rows: list[dict], quantity: str, q: float, draws: int, seed: int = 0) -> float | None:
    """Upper bound of the 95% interval of the ``q``-th percentile of ``quantity``, resampling videos."""
    videos = sorted({r["video"] for r in rows})
    by = {v: np.array([r[quantity] for r in rows if r["video"] == v and np.isfinite(r[quantity])]) for v in videos}
    by = {v: a for v, a in by.items() if len(a)}
    if len(by) < 2 or draws <= 0:
        return None
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    values = [np.percentile(np.concatenate([by[keys[k]] for k in rng.integers(0, len(keys), len(keys))]), q)
              for _ in range(draws)]
    return round(float(np.percentile(values, 97.5)), 2)


def kept_stats(rows: list[dict], key: str, bone: str, draws: int) -> dict:
    sel = [r for r in rows if r[key] and np.isfinite(r[bone])]
    if not sel:
        return dict(n=0)
    v = np.array([r[bone] for r in sel])
    return dict(n=len(v), median=round(float(np.median(v)), 2), p90=round(float(np.percentile(v, 90)), 2),
                median_ci_upper=_bootstrap_upper(sel, bone, 50, draws),
                p90_ci_upper=_bootstrap_upper(sel, bone, 90, draws), videos=len({r["video"] for r in sel}))


def decision_summary(rows: list[dict], key: str, draws: int) -> dict:
    """Per side and bin: frames, survival, and the kept frames' error statistics per bone."""
    out = {}
    for side in SIDES:
        out[side] = {}
        for name, *_ in (("all",), *BINS):
            sel = [r for r in rows if r["side"] == side and (name == "all" or r["bin"] == name)]
            if not sel:
                out[side][name] = dict(frames=0)
                continue
            out[side][name] = dict(frames=len(sel), survival=round(float(np.mean([r[key] for r in sel])), 4),
                                   kept={bone: kept_stats(sel, key, bone, draws) for bone in BONES})
    return out


def uwb_weights(spec: str | None) -> dict | None:
    """{side: {bin: share}} from a gate report (``summary.uwb_elbow_distance``) or a plain mapping."""
    if not spec:
        return None
    data = json.loads(Path(spec).read_text())
    if "summary" in data:
        data = data["summary"]["uwb_elbow_distance"]
    out = {}
    for side, entry in data.items():
        if side not in SIDES or not isinstance(entry, dict):
            continue
        shares = entry.get("share", entry)
        total = sum(shares.get(name, 0.) for name, *_ in BINS)
        out[side] = {name: shares.get(name, 0.) / total for name, *_ in BINS} if total else None
    return out


def weighted_survival(summary: dict, weights: dict | None, min_frames: int = 1) -> dict | None:
    """Per side: survival per bin weighted by UWB's bin shares, over the bins this data supports (>= ``min_frames``
    frames), renormalised; ``covered`` is the UWB weight those bins hold (a low value means the truth set does not
    represent UWB's framing and the weighted figure is not evidence)."""
    if not weights:
        return None
    out = {}
    for side, w in weights.items():
        if not w or side not in summary:
            continue
        cells = summary[side]
        used = {name: share for name, share in w.items() if cells.get(name, {}).get("frames", 0) >= min_frames}
        total = sum(used.values())
        out[side] = dict(survival=round(sum(share * cells[name]["survival"] for name, share in used.items()) / total, 4)
                         if total else None, covered=round(total, 4), bins=sorted(used))
    return out


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
    legacy_dir = Path(args.legacy_dir) if getattr(args, "legacy_dir", None) else None
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
        rows.extend(clip_rows(meta, clip, truth, frames, regressors, rules, legacy_dir))
    inputs = {}
    for r in rows:
        inputs.setdefault(r["input"], set()).add(r["clip"])
    if getattr(args, "require_legacy", False) and set(inputs) - {LEGACY_INPUT}:
        raise SystemExit(f"--require-legacy: {len(inputs.get(TASKS_INPUT, ()))} crops have no legacy Holistic array")
    weights = uwb_weights(getattr(args, "uwb_weights", None))
    draws = getattr(args, "bootstrap", 1000)
    decisions = {"mediapipe": "mediapipe", **{rule: f"rule:{rule}" for rule in rules}}
    by_decision = {name: decision_summary(rows, key, draws) for name, key in decisions.items()}
    min_frames = getattr(args, "min_bin_frames", 200)
    weighted = {name: weighted_survival(s, weights, min_frames) for name, s in by_decision.items()}
    relative = None
    if weights and weighted.get("mediapipe"):
        base = {side: v["survival"] for side, v in weighted["mediapipe"].items()}
        relative = {name: {side: (round(v["survival"] / base[side], 4) if base.get(side) and v["survival"] is not None
                                  else None) for side, v in (w or {}).items()} for name, w in weighted.items()}
    return dict(regressors_sha256=regressors.sha256, features=regressors.features, signers=sorted(signers),
                rules=rules, inputs={k: len(v) for k, v in inputs.items()}, crops=len(crops) - len(skipped),
                skipped=skipped, rows=len(rows), videos=len({r["video"] for r in rows}), bootstrap_draws=draws,
                uwb_weights=weights, uwb_weighted_survival=weighted, uwb_weighted_survival_vs_mediapipe=relative,
                by_decision=by_decision)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--crop-dirs", nargs="+", required=True, help="validate_semlex_adapter / elbow_outside_accuracy crops")
    p.add_argument("--two-view-dir", required=True)
    p.add_argument("--legacy-dir", help="legacy Holistic 0.10.14 arrays of the crops (<crop name>.npz), preferred")
    p.add_argument("--require-legacy", action="store_true", help="refuse when any crop lacks a legacy array")
    p.add_argument("--regressors", required=True)
    p.add_argument("--regressors-sha256", required=True)
    p.add_argument("--rules", default=DEFAULT_RULES)
    p.add_argument("--signers", default="1,2", help="How2Sign signers read (the regressors' validation signers)")
    p.add_argument("--uwb-weights", help="gate report (summary.uwb_elbow_distance) or {side: {bin: share}}")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--min-bin-frames", type=int, default=200, help="bins with fewer frames do not enter the weighting")
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    report = analyse(args)
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("rows", "videos", "inputs", "uwb_weighted_survival",
                                             "uwb_weighted_survival_vs_mediapipe")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
