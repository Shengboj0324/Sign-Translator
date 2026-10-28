"""Convert a sample of Sem-Lex pose files with the adapter and check what the states carry (local, CPU).

Reads ``<video_id>.npy`` files (``allow_pickle=False``) and the pinned metadata CSV, picks ``--clips`` of them at random
(fixed seed), converts each with :func:`semlex_pose.convert_array` (aspect estimated per clip), and reports:

- refusals with reasons; the aspect estimate per clip and its consistency per signer (a signer records on one device);
- usable parts per clip: fractions of frames with a valid forearm (either side), a valid wrist, a tracked hand
  (any finger joint), the face; clips with at least ``--usable`` of their frames;
- the hand depth gain that makes hand bones most rigid on Sem-Lex (``rigidity_gain``, the same estimator the
  validation applies to the emulated arrays), so the measured gain's transfer to legacy Holistic can be checked;
- recogniser v2 on the converted states of clips whose ASL-LEX label maps to one of its glosses
  (``label.upper().replace("_", "")``, e.g. ``about_1`` -> ``ABOUT1``), at 24 fps: top-1 and top-5 for the real states and
  for the same states blanked (duration only), with bootstrap 95% intervals over clips and over signers; diagnostics:
  the states mirrored (``pose.mirror``; v2 is not mirror-invariant) and with Pelvis and Chest masked (v2's ASL Citizen
  states had no stored chest on hip-less frames).
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

METADATA_SHA256 = "f4250b2877e738f028e4b9517922952a350ec8952e59d53ceabdd778938fc3c3"


def mask_joints(state, joints):
    from dataclasses import replace
    body = state.channels["body"]
    keep = np.array([label not in joints for label in body.labels])
    valid = body.valid & keep[None]
    return replace(state, channels=dict(state.channels, body=replace(
        body, valid=valid, values=np.where(valid[..., None], body.values, 0.),
        confidence=np.where(valid, body.confidence, 0.), observed=body.observed & valid)))


def bootstrap(hits: np.ndarray, groups: np.ndarray | None = None, draws: int = 2000, seed: int = 0):
    """95% percentile interval of the mean of ``hits``, resampling clips (or whole groups)."""
    rng = np.random.default_rng(seed)
    if groups is None:
        means = [hits[rng.integers(0, len(hits), len(hits))].mean() for _ in range(draws)]
    else:
        keys = np.unique(groups)
        index = {k: np.flatnonzero(groups == k) for k in keys}
        means = []
        for _ in range(draws):
            pick = np.concatenate([index[k] for k in rng.choice(keys, len(keys))])
            means.append(hits[pick].mean())
    return [round(float(v), 4) for v in np.percentile(means, [2.5, 97.5])]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--npy-dir", required=True)
    p.add_argument("--metadata", required=True, help="local copy of semlex_metadata.csv (pinned SHA-256)")
    p.add_argument("--models-dir", default=os.path.expanduser("~/.sign-translator-ops/models"))
    p.add_argument("--clips", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--usable", type=float, default=.3)
    p.add_argument("--recognizer-run", default="sign-recognizer-v2")
    p.add_argument("--no-recognizer", action="store_true")
    p.add_argument("--pose-z-scale", type=float, help="override the adapter's declared depth gain (sensitivity runs)")
    p.add_argument("--hand-z-scale", type=float)
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    from signtranslator.data_engineering.semlex_pose import BlendshapeModel, Refusal, convert_array, load_face_resources
    from validate_semlex_adapter import HAND_BONES, PALM, rigidity_gain
    payload = Path(args.metadata).read_bytes()
    if hashlib.sha256(payload).hexdigest() != METADATA_SHA256:
        raise SystemExit("metadata CSV is not the pinned acquisition")
    meta = {r["video_id"]: r for r in csv.DictReader(io.StringIO(payload.decode("utf-8")))}
    files = sorted(Path(args.npy_dir).glob("*.npy"))
    files = random.Random(args.seed).sample(files, min(args.clips, len(files)))
    resources = load_face_resources(Path(args.models_dir) / "face_landmarker.task")
    model = BlendshapeModel(resources)
    states, results, refused = {}, [], []
    gains = collections.defaultdict(list)
    grid = [.25, .5, .75, 1., 1.25, 1.5, 2., 2.5, 3.]
    for path in files:
        vid = path.stem
        row = meta.get(vid)
        data = path.read_bytes()
        if row is None:
            refused.append((vid, "not in the metadata"))
            continue
        try:
            array = np.load(io.BytesIO(data), allow_pickle=False)
            gains_kw = {k: v for k, v in (("pose_z_scale", args.pose_z_scale), ("hand_z_scale", args.hand_z_scale))
                        if v is not None}
            state, report = convert_array(array, float(row["duration"]), resources, video_id=vid,
                                          source_sha256=hashlib.sha256(data).hexdigest(), blendshape_model=model,
                                          **gains_kw)
        except Refusal as error:
            refused.append((vid, str(error)))
            continue
        states[vid] = state
        body = state.channels["body"]
        valid = dict(zip(body.labels, body.valid.T))
        frames = len(state.timestamps)
        hand = {s: state.channels[n].valid.any(1) for s, n in (("Left", "left_hand"), ("Right", "right_hand"))}
        fractions = dict(forearm=float((valid["LeftForearm"] | valid["RightForearm"]).mean()),
                         wrist=float((valid["LeftHand"] | valid["RightHand"]).mean()),
                         hand=float((hand["Left"] | hand["Right"]).mean()),
                         face=float(state.channels["face"].valid.any(1).mean()),
                         left_forearm=float(valid["LeftForearm"].mean()), right_forearm=float(valid["RightForearm"].mean()),
                         left_hand=float(hand["Left"].mean()), right_hand=float(hand["Right"].mean()))
        results.append(dict(video_id=vid, signer=row["signer_id"], label_type=row["label_type"], label=row["label"],
                            frames=frames, fps=report["fps"], aspect=report["aspect"],
                            fractions={k: round(v, 4) for k, v in fractions.items()},
                            hand_assignment=report["hand_assignment"]))
        a = report["aspect"]["aspect"]
        for block in (slice(511, 532), slice(532, 553)):
            pts = array[:, block].astype(np.float64)
            g = rigidity_gain(pts, np.isfinite(pts).all((1, 2)), HAND_BONES, PALM, a, grid)
            for k, v in g.items():
                gains[k].append(v)
    out = dict(clips=len(files), converted=len(results), refused=refused,
               refusal_reasons=collections.Counter(r.split(" is ")[0] if "aspect" in r else r for _, r in refused))
    usable = {}
    for part in ("forearm", "wrist", "hand", "face"):
        v = np.array([r["fractions"][part] for r in results])
        usable[part] = dict(clips_at_least=int((v >= args.usable).sum()), any_frame=int((v > 0).sum()),
                            median_fraction=round(float(np.median(v)), 4) if len(v) else None)
    usable["forearm_and_hand"] = int(sum(r["fractions"]["forearm"] >= args.usable and r["fractions"]["hand"] >= args.usable
                                         for r in results))
    out["usable"] = dict(threshold=args.usable, **usable)
    dominance = collections.Counter("left" if r["fractions"]["left_hand"] > r["fractions"]["right_hand"] else
                                    "right" if r["fractions"]["right_hand"] > r["fractions"]["left_hand"] else "tie"
                                    for r in results)
    out["more_tracked_hand"] = dict(dominance)
    per_signer = collections.defaultdict(collections.Counter)
    for r in results:
        per_signer[r["signer"]][r["aspect"].get("snapped_to")] += 1
    out["aspect"] = dict(shapes=collections.Counter(r["aspect"].get("snapped_to") for r in results),
                         raw_percentiles=np.percentile([r["aspect"]["raw"] for r in results], [1, 10, 50, 90, 99]).round(4).tolist(),
                         signers=len(per_signer),
                         signers_with_one_shape=sum(len(c) == 1 for c in per_signer.values()),
                         per_signer={k: dict(v) for k, v in sorted(per_signer.items())})
    out["hand_rigidity_cv_by_gain"] = {str(k): round(float(np.median(v)), 4) for k, v in gains.items()}
    out["per_clip"] = results
    if not args.no_recognizer:
        out["recognizer"] = recognise(args, states, results)
    Path(args.report).write_text(json.dumps(out, indent=1, sort_keys=True, default=dict))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_clip", "refused")}, indent=1, default=dict)[:6000])
    return 0


def recognise(args, states: dict, results: list[dict]) -> dict:
    import torch
    from evaluate_text_to_motion import blanked
    from signtranslator.pose.mirror import mirror_state
    from signtranslator.pose.resample import resample_motion
    from train_text_to_motion import make_store
    from validate_sign_recognizer import load_model, logits_of
    store = make_store("r2")
    model, glosses, best = load_model(store, args.recognizer_run, "cpu")
    index = {g: i for i, g in enumerate(glosses)}
    reliable = set(json.loads(store.get(f"training-runs/{args.recognizer_run}/validation.json"))["per_gloss"]["reliable_glosses"])
    kept = [r for r in results if r["label_type"] == "asllex" and r["label"].upper().replace("_", "") in index]
    items = [(r["video_id"], index[r["label"].upper().replace("_", "")], r["signer"],
              r["label"].upper().replace("_", "") in reliable) for r in kept]
    hand_ok = np.array([r["fractions"]["hand"] >= args.usable for r in kept])
    arm_ok = np.array([r["fractions"]["forearm"] >= args.usable for r in kept])
    if not items:
        return dict(clips=0)
    labels = torch.tensor([y for _, y, _, _ in items])
    signers = np.array([s for *_, s, _ in items])
    real = [resample_motion(states[v], 24.) for v, *_ in items]
    variants = {"real": real, "blank": blanked(real), "mirrored": [mirror_state(s, sample_id=s.sample_id) for s in real],
                "chest_masked": [mask_joints(s, ("Pelvis", "Chest")) for s in real]}
    out = dict(run=args.recognizer_run, checkpoint=best["checkpoint_key"], clips=len(items),
               glosses=len(set(labels.tolist())), signers=len(set(signers.tolist())),
               reliable_clips=int(sum(r for *_, r in items)), mapping='label.upper().replace("_", "")')
    for name, batch in variants.items():
        logits = logits_of(model, batch, torch.device("cpu"))
        ranks = (logits > logits.gather(1, labels[:, None])).sum(1).numpy()
        entry = {}
        for k in (1, 5):
            hits = (ranks < k).astype(np.float64)
            entry[f"top{k}"] = round(float(hits.mean()), 4)
            entry[f"top{k}_ci_clips"] = bootstrap(hits)
            entry[f"top{k}_ci_signers"] = bootstrap(hits, signers) if len(set(signers.tolist())) >= 5 else None
        rel = np.array([r for *_, r in items])
        if rel.any():
            entry["reliable_top5"] = round(float((ranks[rel] < 5).mean()), 4)
        for label, mask in (("hand_tracked", hand_ok), ("hand_untracked", ~hand_ok), ("arm_and_hand", hand_ok & arm_ok)):
            if mask.any():
                entry[f"top5_{label}"] = dict(clips=int(mask.sum()), top5=round(float((ranks[mask] < 5).mean()), 4),
                                              ci=bootstrap((ranks[mask] < 5).astype(np.float64)))
        out[name] = entry
    return out


if __name__ == "__main__":
    raise SystemExit(main())
