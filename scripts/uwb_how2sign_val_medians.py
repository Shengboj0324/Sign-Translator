"""How2Sign-validation medians of the UWB lifter bundle, per gate group, by the gate's own code path (local, CPU).

The gate row "Relative to How2Sign validation: each group's YouTube median <= 1.6x the same bundle's How2Sign-val
median" needs the bundle's validation error **per gate group** (``uwb_lifter_gate.GROUPS``: upper_arms, forearms,
wrists, fingers). The bundle's training report (``train_world_regressors.py``) does not carry it in that form:

- arms: per side, the angle between *local joint rotations* solved with the upright chest (``arm_errors``), not the
  angle between *world bone directions* (``uwb_lifter_gate.state_items``: forward kinematics, convention-free);
- fingers: per side, local finger *rotation* angles of our IK, not finger bone *directions* in the palm frame;
- wrists: per side, the palm frame of the world hand points, not the state's hand frame in the camera frame;
- per side only (a pooled median is not a function of two medians), on 30,000 sampled frames, no common-frame rule.

So the medians are computed here on the same validation data the bundle was validated on, through the gate's code:

- **Data**: the SHA-pinned v2 world pairs (``collect_world_pairs.py``: How2Sign front clips cropped to Sem-Lex framing,
  Tasks image and world landmarks) with their legacy Holistic 0.10.14 arrays (the bundle's inputs, as in training:
  frame count and size must match), **validation signers only** (``--val-signers``, default h2s-1,h2s-2). How2Sign
  signer 9 (component c00, the generator's final-test signer) is refused outright, by file name and by the signer field.
- **Lifted** (per clip): a ``UwbClip`` of the legacy points (``uwb_outside_frame_accuracy.uwb_clip_from_points``),
  converted by ``uwb_keypoints.convert_clip`` for exactly the gate's methods: ``lifter[<declared>]``, each candidate
  rule, and the pseudo-world with the inside rule (they define the gate's *common* frames).
- **Truth**: the clip's Tasks world landmarks solved by ``canonical_from_landmarks`` (the regressors' targets), with
  ``--truth-flags`` (default ``generator``: upright chest and root, as the lifted states; world bone directions do not
  depend on the chest convention, only which frames hold an arm does: the original solver needs both hips, which the
  chest-up crops mostly cut).
- **Errors**: ``uwb_lifter_gate.evaluate_pair`` (frame clock shared, so the lag search is pinned to 0: ``--max-lag 0``)
  with the gate's per-pair cap, then ``group_stats`` per method and scope; the 95% interval of each median resamples
  How2Sign source videos (signers 1 and 2 share them).

The report's ``summary.methods[<primary>].common[<group>].median`` is what ``uwb_gate_verdict.py`` divides by.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

EXCLUDED_SIGNERS = ("h2s-9",)
TRUTH_FLAGS = {"generator": dict(upright_chest=True, upright_root=True), "original": {}}
_WORKER: dict = {}


def signer_of_name(name: str) -> str | None:
    match = re.search(r"-(\d+)-rgb_front", name)
    return f"h2s-{match.group(1)}" if match else None


def source_video(member: str) -> str:
    """How2Sign source video of a clip (``.../<video>_<n>-<signer>-rgb_front.mp4`` -> ``<video>``)."""
    stem = re.sub(r"-\d+-rgb_front.*$", "", Path(member).name)
    return stem.rsplit("_", 1)[0]


def val_files(pairs_dir: Path, signers: set) -> list[Path]:
    if signers & set(EXCLUDED_SIGNERS):
        raise SystemExit("How2Sign signer 9 (component c00) is never read")
    out = []
    for path in sorted(Path(pairs_dir).glob("*.npz")):
        signer = signer_of_name(path.name)
        if signer in EXCLUDED_SIGNERS:
            raise SystemExit(f"{path.name}: How2Sign signer 9 is never read")
        if signer in signers:
            out.append(path)
    return out


def clip_inputs(pairs_path: Path, legacy_dir: Path):
    """(truth ClipLandmarks, lifted UwbClip, fps, meta) of one validation clip, or None when the legacy array does not
    match (frame count or size), as ``train_world_regressors.load_pairs`` skips it."""
    from signtranslator.data_engineering.video_landmarks import ClipLandmarks
    from uwb_outside_frame_accuracy import uwb_clip_from_points
    legacy_path = Path(legacy_dir) / pairs_path.name
    if not legacy_path.exists():
        return None
    with np.load(pairs_path, allow_pickle=False) as z:
        pairs = {k: z[k] for k in z.files}
    signer = str(pairs["signer"])
    if signer in EXCLUDED_SIGNERS:
        raise SystemExit(f"{pairs_path.name}: signer {signer} is never read")
    with np.load(legacy_path, allow_pickle=False) as z:
        holistic, size, times = z["holistic"].astype(np.float64), z["size"].tolist(), z["times"].astype(np.float64)
    width, height = (int(v) for v in pairs["size"])
    frames = len(pairs["pose_found"])
    if len(holistic) != frames or size != [width, height]:
        return None
    legacy_pose = holistic[:, 478:511, :2]
    legacy_hands = np.stack((holistic[:, 511:532, :2], holistic[:, 532:553, :2]), axis=1)
    pose_ok = np.isfinite(legacy_pose).all((1, 2))
    hands_ok = np.isfinite(legacy_hands).all((2, 3))
    name = pairs_path.stem
    lifted = uwb_clip_from_points(np.nan_to_num(legacy_pose), np.nan_to_num(legacy_hands), pose_ok, hands_ok,
                                  (width, height), name)
    rate = 1. / float(np.median(np.diff(times))) if len(times) > 1 and np.all(np.diff(times) > 0) else 30.
    found = pairs["pose_found"].astype(bool)
    truth = ClipLandmarks(
        times=np.arange(frames, dtype=np.float64) / rate, pose_world=pairs["pose_world"].astype(np.float64),
        pose_visibility=pairs["pose_vis"].astype(np.float64), pose_image=pairs["pose_img"][..., :2].astype(np.float64),
        pose_found=found, hands_world=pairs["hand_world"].astype(np.float64),
        hands_image=pairs["hand_img"][..., :2].astype(np.float64),
        hands_found=pairs["hand_found"].astype(bool) & found[:, None], blendshapes=np.zeros((frames, 0)),
        blendshape_names=(), face_matrix=np.tile(np.eye(4), (frames, 1, 1)), face_found=np.zeros(frames, bool),
        roi=True, image_size=(width, height))
    meta = dict(name=name, signer=signer, member=str(pairs["clip"]), video_id=source_video(str(pairs["clip"])),
                shape=str(pairs["shape"]), frames=frames, fps=round(rate, 4))
    return truth, lifted, rate, meta


def evaluate_clip(pairs_path: Path, legacy_dir: Path, regressors, rules: dict, *, cap: int, truth_flags: str) -> dict:
    """One row in the gate's per-pair format (``status``, ``video_id``, ``methods``...)."""
    import uwb_lifter_gate as gate
    from signtranslator.data_engineering import uwb_keypoints as uk
    from signtranslator.data_engineering.video_landmarks import canonical_from_landmarks
    loaded = clip_inputs(pairs_path, legacy_dir)
    if loaded is None:
        return dict(status="refused", name=pairs_path.stem, reason="legacy array missing or not matching")
    truth_landmarks, clip, fps, meta = loaded
    sha = hashlib.sha256(Path(pairs_path).read_bytes()).hexdigest()
    try:
        truth, _ = canonical_from_landmarks(truth_landmarks, sample_id=meta["name"], source_id="how2sign-val-tasks",
                                            source_sha256=sha, clock_id=f"{meta['name']}:frame-index",
                                            inference_method="tasks-world", **TRUTH_FLAGS[truth_flags])
        common = dict(source_sha256=sha, fps_source="measured", flags="generator")
        states = {m: uk.convert_clip(clip, fps, regressors=regressors, visibility=rule, **common)[0]
                  for m, rule in rules.items()}
        states["pseudo"] = uk.convert_clip(clip, fps, world="pseudo", visibility="inside", **common)[0]
        result = gate.evaluate_pair(truth, states, clip, fps, 0, cap=cap)
    except (uk.Refusal, ValueError) as error:
        return dict(status="refused", name=meta["name"], reason=f"{type(error).__name__}: {error}"[:200])
    return dict(meta, status="evaluated", **result)


def _init(settings: dict) -> None:
    import torch
    torch.set_num_threads(1)
    from signtranslator.data_engineering.world_regressor import WorldRegressors
    import uwb_lifter_gate as gate
    regressors = WorldRegressors.load(Path(settings["regressors"]).read_bytes(),
                                      expected_sha256=settings["regressors_sha256"])
    _WORKER.update(settings=settings, regressors=regressors,
                   rules=gate.method_rules(settings["visibility"], settings["candidates"]))


def _evaluate(path: str) -> dict:
    s = _WORKER["settings"]
    return evaluate_clip(Path(path), Path(s["legacy_dir"]), _WORKER["regressors"], _WORKER["rules"],
                         cap=s["cap"], truth_flags=s["truth_flags"])


def summarise(rows: list[dict], primary: str, resamples: int) -> dict:
    import uwb_lifter_gate as gate
    ok = [r for r in rows if r["status"] == "evaluated"]
    methods = sorted({m for r in ok for m in r["methods"]})
    out = dict(pairs=len(ok), videos=len({r["video_id"] for r in ok}), primary=primary, methods={})
    for m in methods:
        out["methods"][m] = {scope: gate.group_stats(ok, m, scope, resamples) for scope in ("common", "own")}
        for scope, groups in out["methods"][m].items():
            for g, s in groups.items():      # source videos holding the group (the verdict's "measured" rule)
                s["videos"] = len({r["video_id"] for r in ok if len(r["methods"][m][scope].get(g, ()))})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pairs-dir", required=True, help="collect_world_pairs.py output (the v2 world pairs)")
    p.add_argument("--legacy-dir", required=True, help="the matching legacy Holistic arrays")
    p.add_argument("--regressors", required=True)
    p.add_argument("--regressors-sha256", required=True)
    p.add_argument("--visibility", default="inside", help="the declared rule (primary method)")
    p.add_argument("--candidate-visibility", default="learned,inside,margin:0.1,margin:0.25",
                   help="the gate run's candidate rules (they set the common frames)")
    p.add_argument("--val-signers", default="h2s-1,h2s-2")
    p.add_argument("--truth-flags", choices=sorted(TRUTH_FLAGS), default="generator")
    p.add_argument("--max-values-per-pair", type=int, default=600, help="the gate's per-pair cap")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--workers", type=int, default=0, help="0: from sched_getaffinity and cgroup memory")
    p.add_argument("--out", required=True)
    p.add_argument("--store", default="r2")
    p.add_argument("--output-key", help="also store the report here (refused if it exists)")
    args = p.parse_args(argv)
    import uwb_lifter_gate as gate
    from uwb_downstream_check import worker_count
    signers = set(args.val_signers.split(","))
    files = val_files(Path(args.pairs_dir), signers)
    if not files:
        raise SystemExit("no validation pairs files")
    candidates = [c for c in args.candidate_visibility.split(",") if c]
    gate.method_rules(args.visibility, candidates)
    store = None
    if args.output_key:
        from train_text_to_motion import make_store
        store = make_store(args.store)
        if store.exists(args.output_key):
            raise SystemExit(f"{args.output_key} exists; a rerun writes elsewhere")
    settings = dict(regressors=args.regressors, regressors_sha256=args.regressors_sha256, visibility=args.visibility,
                    candidates=candidates, legacy_dir=args.legacy_dir, cap=args.max_values_per_pair,
                    truth_flags=args.truth_flags)
    workers = args.workers or worker_count()
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_init, initargs=(settings,)) as pool:
        rows = list(pool.map(_evaluate, [str(f) for f in files], chunksize=1))
    primary = f"lifter[{args.visibility}]"
    out = dict(row="Relative to How2Sign validation: the bundle's How2Sign-val median per gate group",
               regressors_sha256=args.regressors_sha256, val_signers=sorted(signers),
               excluded_signers=list(EXCLUDED_SIGNERS), truth=f"Tasks world landmarks, canonical_from_landmarks "
                                                               f"({args.truth_flags} flags)",
               inputs="legacy Holistic 0.10.14 image landmarks (the bundle's training inputs)",
               method="uwb_lifter_gate.evaluate_pair (max lag 0) + group_stats, scope common",
               candidate_visibility=candidates, max_values_per_pair=args.max_values_per_pair, bootstrap=args.bootstrap,
               files=dict(read=len(files), evaluated=sum(r["status"] == "evaluated" for r in rows),
                          refused=[dict(name=r.get("name"), reason=r.get("reason")) for r in rows
                                   if r["status"] != "evaluated"]),
               files_sha256={Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
               summary=summarise(rows, primary, args.bootstrap))
    text = json.dumps(out, indent=1, sort_keys=True, default=str)
    Path(args.out).write_text(text)
    if store is not None:
        store.put(args.output_key, text.encode())
    print(json.dumps({g: out["summary"]["methods"].get(primary, {}).get("common", {}).get(g) for g in gate.GROUPS},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
