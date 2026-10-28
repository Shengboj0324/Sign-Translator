"""Jobs for ``legacy_holistic_arrays.py``: the exact crops of the collected world pairs, and full-frame ASL Citizen clips.

For each ``collect_world_pairs.py`` file it copies the clip's video from its verified How2Sign zip (ranged reads) and
recomputes the crop box with the same deterministic ``validate_semlex_adapter.framing`` call, refusing the clip if the
box's size differs from the stored one. How2Sign signer 9 (the generator's final-test signer) is skipped by file name
before anything is read. ASL Citizen validation videos (recognition use only) are listed uncropped.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

EXCLUDED_SIGNERS = ("9",)          # How2Sign signer 9: the generator's untouched final-test signer


def excluded(name: str) -> bool:
    match = re.search(r"-(\d+)-rgb_front", name)
    return bool(match) and match.group(1) in EXCLUDED_SIGNERS


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pairs", required=True)
    p.add_argument("--video-dir", required=True)
    p.add_argument("--legacy-dir", required=True)
    p.add_argument("--citizen-videos", nargs="*", default=[])
    p.add_argument("--citizen-out", help="output directory of the legacy ASL Citizen arrays")
    p.add_argument("--jobs", required=True)
    p.add_argument("--models-dir", default=os.path.expanduser("~/.sign-translator-ops/models"))
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.video_landmarks import verify_models
    from collect_world_pairs import SHAPES
    from extract_video_states import StoreRangeFile
    from validate_semlex_adapter import framing
    store = R2Store(os.environ)
    models = verify_models(Path(args.models_dir))
    archives, jobs, refused = {}, [], []
    Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.pairs).glob("*.npz")):
        if excluded(path.name):
            continue
        with np.load(path, allow_pickle=False) as z:
            member, shape, size = str(z["clip"]), str(z["shape"]), z["size"].tolist()
        if excluded(member) or not member.endswith(".mp4"):
            continue
        video = Path(args.video_dir) / Path(member).name
        if not video.exists():
            for key in ("acquisition/2026-10-17-v1/how2sign-public/train_rgb_front_clips.zip",
                        "acquisition/2026-10-17-v1/how2sign-public/val_rgb_front_clips.zip"):
                archive = archives.get(key)
                if archive is None:
                    archive = archives[key] = zipfile.ZipFile(io.BufferedReader(
                        StoreRangeFile(store.s3, store.bucket, key), 1 << 20))
                if member in archive.NameToInfo:
                    video.write_bytes(archive.read(member))
                    break
        box = framing(video, models, SHAPES[shape])
        if box is None or [box[2], box[3]] != size:
            refused.append(dict(pairs=path.name, box=box, size=size))
            continue
        jobs.append(dict(video=str(video), box=list(box), out=str(Path(args.legacy_dir) / path.name), pairs=str(path)))
    for video in args.citizen_videos:
        jobs.append(dict(video=video, box=None, out=str(Path(args.citizen_out) / f"{Path(video).stem}.npz")))
    Path(args.jobs).write_text(json.dumps(jobs, indent=1))
    print(json.dumps(dict(jobs=len(jobs), refused=refused[:20], refused_count=len(refused))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
