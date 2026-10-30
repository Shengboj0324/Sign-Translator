"""Tight chest-up crops of How2Sign **training** signers for UWB lifter v2's validity head (CPU).

Declared in docs/NEXT_PHASE_PLAN.md, "UWB lifter v2 and gate v2: declared" (2026-10-30, 11:12 UTC), amendment 2
(12:03 UTC): the validity head also learns from tight crops of How2Sign training signers (3, 4, 5, 8, 11), labelled by
our Tasks visibility >= 0.5. The crops train **only** the validity head, never the pose regressor. Signer 9 (the generator's final-test signer) is
refused; the validation signers 1 and 2 (the How2Sign-val medians, the chest-up rows and the truth anchor of gate v2)
are refused too, so nothing they hold reaches training.

Stages (the pod job ``runpod/fleet/uwb_lifter_v2.sh crops`` runs them in order):

1. ``fetch``: from the verified How2Sign train zip (``collect_world_pairs.H2S``, provenance re-read), a seeded sample
   of ``--clips-per-signer`` front clips per training signer is copied to ``--video-dir`` and listed (``--list``);
2. ``elbow_outside_accuracy.py extract`` crops each listed video tightly (nose at 0.30, shoulders at 0.80 of the crop;
   4:3 and 16:9) and stores our Tasks landmarks of the crop;
3. ``legacy-jobs``: one ``legacy_holistic_arrays.py`` job per crop (the crop's own box on its source video), so the
   inputs are legacy Holistic 0.10.14 as UWB's and the v2 pairs' are;
4. ``legacy_holistic_arrays.py`` (its own Python 3.11 venv) writes the legacy arrays.

``train_uwb_lifter_v2.load_crops`` reads the result for the validity head (frame count and size must match, as for the
v2 pairs). ``check-models`` compares the legacy Holistic version and model SHA-256s that ``legacy_holistic_arrays.py
--hashes`` recorded with those of the v2 legacy pairs (``LEGACY_MODELS``) and refuses any difference.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

TRAIN_SIGNERS = ("3", "4", "5", "8", "11")
# The legacy Holistic of the v2 legacy pairs: mediapipe 0.10.14, complexity 1, refined face; the model SHA-256s of the
# venv that produced them (~/.venvs/semlex-legacy-mp, legacy_holistic_arrays.model_hashes, read 2026-10-30).
LEGACY_MODELS = {
    "mediapipe": "0.10.14", "model_complexity": 1, "refine_face_landmarks": True,
    "models": {
        "pose_landmark/pose_landmark_full.tflite": "e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f",
        "pose_detection/pose_detection.tflite": "9ba9dd3d42efaaba86b4ff0122b06f29c4122e756b329d89dca1e297fd8f866c",
        "face_landmark/face_landmark_with_attention.tflite":
            "e06a804e0144f9929eda782122916b35d60c697c3c9344013ca2bbe76a6ce2b4",
        "hand_landmark/hand_landmark_full.tflite": "11c272b891e1a99ab034208e23937a8008388cf11ed2a9d776ed3d01d0ba00e3",
        "holistic_landmark/hand_recrop.tflite": "67d996ce96f9d36fe17d2693022c6da93168026ab2f028f9e2365398d8ac7d5d",
        "face_detection/face_detection_short_range.tflite":
            "bbff11cebd1eb27a1e004cae0b0e63ec8c551cbf34a4451148b4908b8db3eca8",
        "palm_detection/palm_detection_full.tflite": "1b14e9422c6ad006cde6581a46c8b90dd573c07ab7f3934b5589e7cea3f89a54"}}


def check_models(recorded: dict) -> None:
    """Refuses (SystemExit) a legacy Holistic run whose version, settings or model SHA-256s differ from the v2 pairs'."""
    differs = [k for k in LEGACY_MODELS if recorded.get(k) != LEGACY_MODELS[k]]
    if differs:
        models = recorded.get("models") or {}
        bad = sorted(k for k in set(models) | set(LEGACY_MODELS["models"]) if models.get(k) != LEGACY_MODELS["models"].get(k))
        raise SystemExit(f"legacy Holistic differs from the v2 legacy pairs': {differs} {bad}")
REFUSED_SIGNERS = ("9", "1", "2")          # 9: final test; 1, 2: How2Sign validation (gate v2's controls)


def signer_of(name: str) -> str | None:
    match = re.search(r"-(\d+)-rgb_front", name)
    return match.group(1) if match else None


def check_signers(signers) -> list[str]:
    signers = [s for s in signers if s]
    if set(signers) & set(REFUSED_SIGNERS):
        raise SystemExit(f"signers {sorted(set(signers) & set(REFUSED_SIGNERS))} are never read for training")
    if not set(signers) <= set(TRAIN_SIGNERS):
        raise SystemExit(f"only How2Sign training signers {TRAIN_SIGNERS} are cropped")
    return signers


def sample_members(names, signers, per_signer: int, seed: int) -> list[str]:
    """A seeded sample of ``per_signer`` front clips per signer (sorted); refused signers are never listed."""
    rng = random.Random(seed)
    by_signer: dict = {}
    for name in sorted(n for n in names if n.lower().endswith(".mp4")):
        signer = signer_of(name)
        if signer in signers:
            by_signer.setdefault(signer, []).append(name)
    out = []
    for signer in sorted(by_signer, key=int):
        out += rng.sample(by_signer[signer], min(per_signer, len(by_signer[signer])))
    return sorted(out)


def legacy_jobs(crop_dir: Path, video_dir: Path, legacy_dir: Path) -> list[dict]:
    """One legacy Holistic job per tight crop: the crop's box on its source video."""
    jobs = []
    for path in sorted(Path(crop_dir).glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
        if signer_of(meta["video"]) not in TRAIN_SIGNERS:
            raise SystemExit(f"{path.name}: not a training signer's crop")
        jobs.append(dict(video=str(Path(video_dir) / meta["video"]), box=[int(v) for v in meta["box"]],
                         out=str(Path(legacy_dir) / path.name)))
    return jobs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=("fetch", "legacy-jobs", "check-models"))
    p.add_argument("--hashes", help="check-models: the --hashes file legacy_holistic_arrays.py wrote")
    p.add_argument("--signers", default=",".join(TRAIN_SIGNERS))
    p.add_argument("--clips-per-signer", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video-dir")
    p.add_argument("--list", help="fetch: write the copied video paths here (one per line)")
    p.add_argument("--crop-dir", help="legacy-jobs: the tight crops")
    p.add_argument("--legacy-dir", help="legacy-jobs: where the legacy arrays go")
    p.add_argument("--jobs", help="legacy-jobs: the jobs JSON")
    args = p.parse_args(argv)
    signers = check_signers(args.signers.split(","))
    if args.stage == "check-models":
        check_models(json.loads(Path(args.hashes).read_text()))
        print("legacy Holistic models: as the v2 legacy pairs'")
        return 0
    if not args.video_dir:
        raise SystemExit("--video-dir is required")
    if args.stage == "legacy-jobs":
        jobs = legacy_jobs(Path(args.crop_dir), Path(args.video_dir), Path(args.legacy_dir))
        Path(args.jobs).write_text(json.dumps(jobs, indent=1))
        print(json.dumps(dict(jobs=len(jobs))))
        return 0
    from signtranslator.data.text_motion_dataset import R2Store
    from collect_world_pairs import H2S
    from extract_video_states import StoreRangeFile
    store = R2Store(os.environ)
    key = H2S.format(split="train")
    if not json.loads(store.get(key + ".provenance.json")).get("verified_by_r2_reread"):
        raise SystemExit(f"{key} is not a verified acquisition")
    archive = zipfile.ZipFile(io.BufferedReader(StoreRangeFile(store.s3, store.bucket, key), 1 << 20))
    members = sample_members(archive.namelist(), signers, args.clips_per_signer, args.seed)
    Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for member in members:
        target = Path(args.video_dir) / Path(member).name
        if not target.exists():
            target.write_bytes(archive.read(member))
        paths.append(str(target))
    if args.list:
        Path(args.list).write_text("\n".join(paths) + "\n")
    print(json.dumps(dict(videos=len(paths), signers=signers)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
