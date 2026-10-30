"""Visibility rules for the UWB lifter, measured on How2Sign two-view truth with arms cut by the frame (local, CPU),
and the declared selection of the rule.

The question (reviews of e186d62, 6625c83 and 4c67724): YouTube framings cut the arms, so a visibility rule decides
how many arm frames survive. Which rule keeps arms as accurate as the lifter is on arms in full view, and how many does
it keep, weighted the way UWB's frames are actually cut?

**Data** (the ``elbow_outside_accuracy.py`` set): How2Sign val front clips with How2Sign two-view states (the truth):
the full frames (``shape`` ``full``), crops to Sem-Lex framing (4:3, 16:9, 1:1) and tighter crops (nose at 0.30,
shoulders at 0.80). **Inputs**: legacy MediaPipe Holistic 0.10.14 (``legacy_holistic_arrays.py`` on the same boxes;
``--legacy-dir``, preferred: UWB's own kind of landmarks) where an array exists with the clip's frame count and size,
otherwise the clip's stored Tasks image landmarks; every row records its input. Only ``--signers`` are read (default
How2Sign 1 and 2: the regressors' validation signers); **signer 9 (the generator's final-test signer) is refused**.

**Per clip**, the lifter (``uwb_keypoints.convert_clip``, generator flags) runs once per rule and once with every point
visible (``all``: the ungated arm). Per two-view frame, side and bone (upper arm, forearm):

- the bone's distance outside the frame (frame heights; 0 inside) is the **larger of its two ends'** (forearm: elbow
  and wrist; upper arm: shoulder and elbow), since its swing joint needs both (``video_landmarks.BODY_JOINTS``);
- MediaPipe's keep decision (Tasks visibility >= 0.5 on the bone's chain: shoulder and elbow for the upper arm, plus
  the wrist for the forearm) and each rule's (the bone valid in that rule's state);
- the ungated lifter's direction error against the two-view state.

**Anchor**: the ungated lifter's error on full frames where the arm (shoulder, elbow, wrist) is entirely in view, per
side and bone: median and p90 with by-video 95% intervals. (The tight crops' inside bin is too thin to anchor.)

**Per decision** (MediaPipe and each rule; on the crops), per side, bone and bin (inside, 0-0.1, 0.1-0.3, > 0.3):
frames, survival, and the kept frames' median and p90 error with by-video 95% upper bounds, kept frames and videos.
A bin is **admitted** by a rule when the rule keeps at least ``--admit-share`` of its frames, and **unmeasured** when
fewer than 200 kept frames or 5 videos support it. **UWB-weighted survival** per side and bone weights the bins by UWB's
own bone-distance distribution (``--uwb-weights``: ``uwb_lifter_gate.py``'s ``summary.uwb_arm_distance``) over the bins
with >= ``--min-bin-frames`` frames, with the UWB weight those bins cover.

**Selection (the declared rule).** A rule is admissible iff, for every side, bone and admitted bin (unmeasured =
fail), the CI-upper kept median <= anchor median + 3 deg and the CI-upper kept p90 <= anchor p90 + 5 deg. The admissible
rule with the largest UWB-weighted forearm survival (mean of the sides) is chosen; rules within 2 points of the best
go to the more conservative (``CONSERVATIVE_ORDER``). ``learned`` is chosen only conditional on the YouTube gate's
validity (all-frame forearm agreement >= 0.85 per side, UWB-only valid <= 10%). If no rule is admissible: ``inside``.

**Gate v2** (plan "UWB lifter v2 and gate v2", 2026-10-30 11:12 UTC): rule ``head`` (a bundle with a validity head) is
accepted like any rule. ``--tight-summary`` adds ``tight``: the per-decision summary on the tight crops only, with an
``outside`` cell (every frame whose bone lies outside the frame: distance > 0) per side and bone, and each rule's
amendment-1 admissibility on those crops (the chest-up survival row of gate v2). Without it the report is unchanged.
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
BINS = (("inside", 0., 0.), ("0-0.1", 0., .1), ("0.1-0.3", .1, .3), ("0.3-inf", .3, np.inf))
DEFAULT_RULES = "learned,inside,margin:0.1,margin:0.25"
CONSERVATIVE_ORDER = ("inside", "margin:0.1", "margin:0.25", "margin:0.5", "head", "learned")
EXCLUDED_SIGNERS = ("9",)
LEGACY_INPUT = "legacy-holistic-0.10.14"
TASKS_INPUT = "tasks-image"
MIN_KEPT_FRAMES, MIN_KEPT_VIDEOS = 200, 5
MEDIAN_SLACK, P90_SLACK, TIE = 3., 5., .02
LEARNED_CONDITION = ("conditional on the YouTube gate's validity: all-frame forearm agreement >= 0.85 per side and "
                     "UWB-only valid <= 10%")


def signer_of(name: str) -> str | None:
    match = re.search(r"-(\d+)-rgb_front", name)
    return match.group(1) if match else None


def bin_of(distance: float) -> str:
    for name, lo, hi in BINS:
        if (distance == 0 if hi == 0 else lo < distance <= hi):
            return name
    return BINS[-1][0]


def kind_of(meta) -> str:
    return "full" if meta["shape"] == "full" else "tight" if meta["shape"].endswith("-tight") else "semlex"


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


def frame_size(meta, clip, legacy_dir: Path | None) -> list | None:
    """The clip's frame size (w, h): the crop's recorded size, else (full frames, which record none) the legacy
    array's decoded size, else the landmarks' recorded image size; None when unknown."""
    if meta.get("size"):
        return list(meta["size"])
    if legacy_dir is not None and (Path(legacy_dir) / f"{meta['name']}.npz").exists():
        with np.load(Path(legacy_dir) / f"{meta['name']}.npz", allow_pickle=False) as z:
            return [int(v) for v in z["size"]]
    return list(clip.image_size) if getattr(clip, "image_size", None) else None


def input_points(meta, clip, legacy_dir: Path | None):
    """(pose (T, 33, 2), hands (T, 2, 21, 2), pose found, hands found, input name): legacy Holistic when its array
    exists with the clip's frame count and size, else the clip's Tasks image landmarks."""
    if legacy_dir is not None:
        path = Path(legacy_dir) / f"{meta['name']}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                holistic, size = z["holistic"].astype(np.float64), z["size"].tolist()
            if len(holistic) == len(clip.times) and list(size) == list(meta["size"]):
                pose = holistic[:, 478:511, :2]
                hands = np.stack((holistic[:, 511:532, :2], holistic[:, 532:553, :2]), axis=1)
                return (np.nan_to_num(pose), np.nan_to_num(hands), np.isfinite(pose).all((1, 2)),
                        np.isfinite(hands).all((2, 3)), LEGACY_INPUT)
    return (np.asarray(clip.pose_image)[..., :2], np.asarray(clip.hands_image)[..., :2],
            np.asarray(clip.pose_found, bool), np.asarray(clip.hands_found, bool), TASKS_INPUT)


def outside_distance(xy: np.ndarray, aspect: float) -> np.ndarray:
    """Distance (frame heights) of normalised image points from the image rectangle; 0 inside."""
    dx = np.maximum(np.maximum(-xy[..., 0], xy[..., 0] - 1.), 0.) * aspect
    dy = np.maximum(np.maximum(-xy[..., 1], xy[..., 1] - 1.), 0.)
    return np.hypot(dx, dy)


def lifted_world(state):
    """Per side and bone: world directions and validity from a state's forward kinematics."""
    from validate_semlex_adapter import ARM_REST, world_frames
    w = world_frames(state)
    out = {}
    for side in SIDES:
        rest = np.asarray(ARM_REST[f"{side}UpperArm"])
        out[side] = {bone: (w[f"{side}{bone}"][0] @ rest, w[f"{side}{bone}"][1].astype(bool)) for bone in BONES}
    return out


def clip_rows(meta, clip, truth_state, frames, regressors, rules, legacy_dir=None) -> list[dict]:
    """One row per (two-view frame, side, bone): input, crop kind, the bone's distance outside and bin, the arm's
    full-view flag, MediaPipe's and each rule's keep decision, and the ungated lifter's error (NaN where none)."""
    from signtranslator.data_engineering import uwb_keypoints as uk
    from validate_semlex_adapter import direction_angles
    from elbow_outside_accuracy import truth_world
    size = meta["size"]
    pose_xy, hands_xy, found_pose, found_hands, source = input_points(meta, clip, legacy_dir)
    uwb = uwb_clip_from_points(pose_xy, hands_xy, found_pose, found_hands, size, meta["name"])
    fps = 1. / float(np.median(np.diff(clip.times)))
    common = dict(source_sha256="0" * 64, fps_source="measured", regressors=regressors)
    ungated = lifted_world(uk.convert_clip(uwb, fps, visibility="all", **common)[0])
    kept = {rule: lifted_world(uk.convert_clip(uwb, fps, visibility=rule, **common)[0]) for rule in rules}
    truth = truth_world(truth_state)
    aspect = size[0] / size[1]
    kind = kind_of(meta)
    vis = np.asarray(clip.pose_visibility)[frames] >= .5
    tasks_found = np.asarray(clip.pose_found)[frames]
    rows = []
    for side, (shoulder, elbow, wrist) in SIDES.items():
        dist = {k: outside_distance(pose_xy[frames, k], aspect) for k in (shoulder, elbow, wrist)}
        full_view = (dist[shoulder] == 0) & (dist[elbow] == 0) & (dist[wrist] == 0)
        for bone, ends, chain in (("UpperArm", (shoulder, elbow), (shoulder, elbow)),
                                  ("Forearm", (elbow, wrist), (shoulder, elbow, wrist))):
            distance = np.maximum(dist[ends[0]], dist[ends[1]])
            mediapipe = vis[:, list(chain)].all(1) & tasks_found
            value, ok = ungated[side][bone]
            true_value, true_ok = truth[side][bone]
            error = np.where(ok[frames] & true_ok, direction_angles(value[frames], true_value), np.nan)
            for t in range(len(frames)):
                if not found_pose[frames[t]]:
                    continue
                rows.append(dict(video=meta["video"], clip=meta["name"], kind=kind, input=source, side=side, bone=bone,
                                 distance=float(distance[t]), bin=bin_of(float(distance[t])),
                                 full_view=bool(full_view[t]), mediapipe=bool(mediapipe[t]),
                                 **{f"rule:{r}": bool(kept[r][side][bone][1][frames[t]]) for r in rules},
                                 error=float(error[t])))
    return rows


def _by_video(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        if np.isfinite(r["error"]):
            by.setdefault(r["video"], []).append(r["error"])
    return {k: np.asarray(v) for k, v in by.items()}


def _interval(by: dict, q: float, draws: int, seed: int = 0) -> list | None:
    """95% interval of the ``q``-th percentile of the pooled errors, resampling videos."""
    keys = sorted(by)
    if len(keys) < 2 or draws <= 0:
        return None
    rng = np.random.default_rng(seed)
    values = [np.percentile(np.concatenate([by[keys[k]] for k in rng.integers(0, len(keys), len(keys))]), q)
              for _ in range(draws)]
    return [round(float(np.percentile(values, 2.5)), 2), round(float(np.percentile(values, 97.5)), 2)]


def error_stats(rows: list[dict], draws: int) -> dict:
    by = _by_video(rows)
    if not by:
        return dict(n=0, videos=0)
    v = np.concatenate(list(by.values()))
    median_ci, p90_ci = _interval(by, 50, draws), _interval(by, 90, draws)
    return dict(n=int(len(v)), videos=len(by), median=round(float(np.median(v)), 2),
                p90=round(float(np.percentile(v, 90)), 2), median_ci95=median_ci, p90_ci95=p90_ci,
                median_ci_upper=median_ci and median_ci[1], p90_ci_upper=p90_ci and p90_ci[1])


def anchor(rows: list[dict], draws: int) -> dict:
    """The ungated lifter on full frames with the whole arm in view, per side and bone."""
    return {side: {bone: error_stats([r for r in rows if r["kind"] == "full" and r["full_view"] and r["side"] == side
                                      and r["bone"] == bone], draws) for bone in BONES} for side in SIDES}


def decision_summary(rows: list[dict], key: str, draws: int, admit_share: float, outside: bool = False) -> dict:
    """Per side, bone and bin (crops only): frames, survival, kept error statistics, admitted and unmeasured.
    ``outside``: also an ``outside`` cell (distance > 0, every outside bin together; never admitted)."""
    out = {}
    for side in SIDES:
        out[side] = {}
        for bone in BONES:
            out[side][bone] = {}
            for name, *_ in (("all",), *BINS) + ((("outside",),) if outside else ()):
                sel = [r for r in rows if r["kind"] != "full" and r["side"] == side and r["bone"] == bone
                       and (name == "all" or r["bin"] == name or (name == "outside" and r["distance"] > 0))]
                if not sel:
                    out[side][bone][name] = dict(frames=0, survival=None, admitted=False)
                    continue
                kept_rows = [r for r in sel if r[key]]
                stats = error_stats(kept_rows, draws)
                survival = len(kept_rows) / len(sel)
                out[side][bone][name] = dict(
                    frames=len(sel), videos=len({r["video"] for r in sel}), survival=round(survival, 4),
                    kept=stats, admitted=bool(name not in ("all", "outside") and survival >= admit_share),
                    unmeasured=bool(stats["n"] < MIN_KEPT_FRAMES or stats["videos"] < MIN_KEPT_VIDEOS))
    return out


def uwb_weights(spec: str | None) -> dict | None:
    """{side: {bone: {bin: share}}} from a gate report (``summary.uwb_arm_distance``) or the same mapping."""
    if not spec:
        return None
    data = json.loads(Path(spec).read_text())
    if "summary" in data:
        data = data["summary"]["uwb_arm_distance"]
    out = {}
    for side in SIDES:
        for bone in BONES:
            entry = (data.get(side) or {}).get(bone)
            if not isinstance(entry, dict):
                continue
            shares = entry.get("share", entry)
            total = sum(shares.get(name, 0.) for name, *_ in BINS)
            if total:
                out.setdefault(side, {})[bone] = {name: shares.get(name, 0.) / total for name, *_ in BINS}
    return out


def weighted_survival(summary: dict, weights: dict | None, min_frames: int) -> dict | None:
    """Per side and bone: survival per bin weighted by UWB's bin shares over the bins with >= ``min_frames`` frames
    (renormalised), and ``covered``, the UWB weight of those bins."""
    if not weights:
        return None
    out = {}
    for side, bones in weights.items():
        for bone, w in bones.items():
            cells = summary[side][bone]
            used = {name: share for name, share in w.items() if cells.get(name, {}).get("frames", 0) >= min_frames}
            total = sum(used.values())
            out.setdefault(side, {})[bone] = dict(
                survival=round(sum(share * cells[name]["survival"] for name, share in used.items()) / total, 4)
                if total else None, covered=round(total, 4), bins=sorted(used))
    return out


def rule_verdict(cells: dict, anchors: dict, weighted: dict | None) -> dict:
    """Admissibility of one rule (``cells``: its per side/bone/bin summary) against the anchor, and its UWB-weighted
    forearm survival (mean over sides; a side without a weighted value counts 0)."""
    failures = []
    for side in SIDES:
        for bone in BONES:
            a = anchors[side][bone]
            for name, *_ in BINS:
                cell = cells[side][bone][name]
                if not cell.get("admitted"):
                    continue
                if cell["unmeasured"] or not a.get("n"):
                    failures.append(f"{side}.{bone}.{name}: unmeasured ({cell.get('kept', {}).get('n', 0)} kept frames, "
                                    f"{cell.get('kept', {}).get('videos', 0)} videos)")
                    continue
                k = cell["kept"]
                if k["median_ci_upper"] is None or k["median_ci_upper"] > a["median"] + MEDIAN_SLACK:
                    failures.append(f"{side}.{bone}.{name}: median CI upper {k['median_ci_upper']} > "
                                    f"{round(a['median'] + MEDIAN_SLACK, 2)}")
                if k["p90_ci_upper"] is None or k["p90_ci_upper"] > a["p90"] + P90_SLACK:
                    failures.append(f"{side}.{bone}.{name}: p90 CI upper {k['p90_ci_upper']} > "
                                    f"{round(a['p90'] + P90_SLACK, 2)}")
    forearm = {side: ((weighted or {}).get(side, {}).get("Forearm") or {}) for side in SIDES}
    score = float(np.mean([f.get("survival") or 0. for f in forearm.values()]))
    return dict(admissible=not failures, failures=failures, uwb_weighted_forearm_survival=round(score, 4),
                coverage={side: f.get("covered") for side, f in forearm.items()})


def select_rule(by_decision: dict, anchors: dict, weighted: dict, rules: list[str]) -> dict:
    """The declared selection (module docstring). ``weighted``: rule -> its UWB-weighted survival table."""
    verdicts = {rule: rule_verdict(by_decision[rule], anchors, (weighted or {}).get(rule)) for rule in rules}
    admissible = [r for r in rules if verdicts[r]["admissible"]]
    if not admissible:
        return dict(chosen="inside", reason="no rule admissible: the declared fallback", verdicts=verdicts)
    best = max(verdicts[r]["uwb_weighted_forearm_survival"] for r in admissible)
    near = [r for r in admissible if verdicts[r]["uwb_weighted_forearm_survival"] >= best - TIE]
    order = {r: k for k, r in enumerate(CONSERVATIVE_ORDER)}
    chosen = min(near, key=lambda r: (order.get(r, len(order)), r))
    out = dict(chosen=chosen, reason=f"admissible; UWB-weighted forearm survival within {TIE} of the best ({best}); "
                                     f"the most conservative of {sorted(near)}", verdicts=verdicts)
    if chosen == "learned":
        out["condition"] = LEARNED_CONDITION
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
    full, clips, skipped = {}, [], []
    for directory in args.crop_dirs:
        for path in sorted(Path(directory).glob("*.npz")):
            signer = signer_of(path.name)
            if signer in EXCLUDED_SIGNERS:
                raise SystemExit(f"{path.name}: How2Sign signer 9 is never read")
            if signer not in signers:
                continue
            meta, clip, _ = load_npz(path)
            if meta["shape"] == "full":
                full[meta["video"]] = clip
            clips.append((meta, clip))
    rows = []
    legacy_dir = Path(args.legacy_dir) if getattr(args, "legacy_dir", None) else None
    for meta, clip in clips:
        key = meta["video"].replace("-rgb_front.mp4", "")
        entry, mf = index.get(key), full.get(meta["video"])
        size = frame_size(meta, clip, legacy_dir)
        if size is None:
            skipped.append(meta["name"])
            continue
        meta = dict(meta, size=size)
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
        raise SystemExit(f"--require-legacy: {len(inputs.get(TASKS_INPUT, ()))} clips have no legacy Holistic array")
    draws = getattr(args, "bootstrap", 1000)
    admit = getattr(args, "admit_share", .05)
    anchors = anchor(rows, draws)
    decisions = {"mediapipe": "mediapipe", **{rule: f"rule:{rule}" for rule in rules}}
    by_decision = {name: decision_summary(rows, key, draws, admit) for name, key in decisions.items()}
    weights = uwb_weights(getattr(args, "uwb_weights", None))
    min_frames = getattr(args, "min_bin_frames", 200)
    weighted = {name: weighted_survival(s, weights, min_frames) for name, s in by_decision.items()}
    selection = select_rule(by_decision, anchors, weighted, rules)
    extra = {}
    if getattr(args, "tight_summary", False):
        tight_rows = [r for r in rows if r["kind"] == "tight"]
        tight = {name: decision_summary(tight_rows, key, draws, admit, outside=True) for name, key in decisions.items()}
        extra["tight"] = dict(by_decision=tight, clips=len({r["clip"] for r in tight_rows}),
                              videos=len({r["video"] for r in tight_rows}),
                              verdicts={rule: rule_verdict(tight[rule], anchors, None) for rule in rules})
    return dict(**extra, regressors_sha256=regressors.sha256, features=regressors.features, signers=sorted(signers),
                rules=rules, inputs={k: len(v) for k, v in inputs.items()},
                clips=dict(full=sum(1 for m, _ in clips if kind_of(m) == "full"),
                           crops=sum(1 for m, _ in clips if kind_of(m) != "full")),
                skipped=skipped, rows=len(rows), videos=len({r["video"] for r in rows}), bootstrap_draws=draws,
                admit_share=admit, min_kept=dict(frames=MIN_KEPT_FRAMES, videos=MIN_KEPT_VIDEOS),
                anchor=anchors, by_decision=by_decision, uwb_weights=weights, uwb_weighted_survival=weighted,
                selection=selection)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--crop-dirs", nargs="+", required=True, help="validate_semlex_adapter / elbow_outside_accuracy clips")
    p.add_argument("--two-view-dir", required=True)
    p.add_argument("--legacy-dir", help="legacy Holistic 0.10.14 arrays (<clip name>.npz), preferred")
    p.add_argument("--require-legacy", action="store_true", help="refuse when any clip lacks a legacy array")
    p.add_argument("--regressors", required=True)
    p.add_argument("--regressors-sha256", required=True)
    p.add_argument("--rules", default=DEFAULT_RULES)
    p.add_argument("--signers", default="1,2", help="How2Sign signers read (the regressors' validation signers)")
    p.add_argument("--uwb-weights", help="gate report (summary.uwb_arm_distance) or {side: {bone: {bin: share}}}")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--min-bin-frames", type=int, default=200, help="bins with fewer frames do not enter the weighting")
    p.add_argument("--admit-share", type=float, default=.05, help="a rule admits a bin when it keeps this share of it")
    p.add_argument("--tight-summary", action="store_true", help="gate v2: add the tight-crop summary (key 'tight')")
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    report = analyse(args)
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("rows", "videos", "inputs", "clips", "anchor")}, indent=1))
    print(json.dumps({k: v for k, v in report["selection"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
