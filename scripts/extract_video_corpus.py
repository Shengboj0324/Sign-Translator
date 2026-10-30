"""Convert a stored sentence-video corpus (one sentence per video) to canonical motion states, as the YouTube-ASL
pipeline converts a caption cue (``acquire_youtube_asl.py``): MediaPipe Tasks landmarkers over every frame with
ROI crops (``video_landmarks.extract_clip``), the raw landmarks stored, and the state solved from exactly those stored
landmarks with the arm-recovery conversion (upright chest and root; ``ARM_RECOVERY_METHOD``).

Input: a prefix written by ``acquire_hf_dataset.py`` (``INDEX.json``, ``files/...``) and its metadata CSVs
(``--metadata`` relative paths; columns ``file_name`` plus the corpus's text fields, kept verbatim). Each video gets
``<out>/states/<split>/<stem>.npz``, ``<out>/landmarks/<split>/<stem>.npz`` and a report
``<out>/videos/<split>/<stem>.json`` (text fields, source key and SHA-256, frames, seconds, hand and face coverage,
available channels; or the refusal). A video with a report is skipped (resumable); ``--shard i --shards n`` splits the
work by a hash of the path. Videos are fetched to a temporary file and deleted after conversion.

    python scripts/extract_video_corpus.py --source acquisition/2026-10-30-v1/2m-flores-asl \
        --metadata data/dev/metadata.csv --metadata data/devtest/metadata.csv \
        --out acquisition/2026-10-30-v1/2m-flores-asl/extract-v1 --source-id 2m-flores-asl --models-dir /root/work/models
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def jobs_of(store, source: str, metadata: list[str]) -> list[dict]:
    index = json.loads(store.get(f"{source}/INDEX.json"))
    by_path = {f["path"]: f for f in index["files"]}
    jobs = []
    for rel in metadata:
        split = Path(rel).parent.name
        for row in csv.DictReader(io.StringIO(store.get(f"{source}/files/{rel}").decode("utf-8"))):
            path = (Path(rel).parent / row["file_name"]).as_posix()
            entry = by_path.get(path)
            if entry is None:
                raise SystemExit(f"{path} is listed in {rel} but not in {source}/INDEX.json")
            jobs.append(dict(split=split, stem=Path(row["file_name"]).stem, key=entry["key"], sha256=entry["sha256"],
                             fields={k: v for k, v in row.items() if k != "file_name"}))
    return jobs


def convert(job: dict, args) -> dict:
    import numpy as np  # noqa: F401  (the converter's arrays)
    from acquire_youtube_asl import ARM_RECOVERY_METHOD, face_stats, hand_coverage
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.video_landmarks import (
        canonical_from_landmarks, extract_clip, inference_method, load_raw_landmarks, save_raw_landmarks)
    store = R2Store(os.environ)
    base = f"{job['split']}/{job['stem']}"
    report_key = f"{args.out}/videos/{base}.json"
    if store.exists(report_key):
        return dict(base=base, status="kept")
    from signtranslator.data_engineering.video_landmarks import verify_models
    models = verify_models(Path(args.models_dir))           # each model checked against its pinned SHA-256
    report = dict(base=base, source_key=job["key"], source_sha256=job["sha256"], **job["fields"])
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / Path(job["key"]).name
        payload = store.get(job["key"])
        if job["sha256"] and hashlib.sha256(payload).hexdigest() != job["sha256"]:
            raise SystemExit(f"{job['key']}: SHA-256 differs from the index")
        video.write_bytes(payload)
        del payload
        try:
            clip = extract_clip(video, models, roi=True, count_people_every=args.count_people_every)
            raw = save_raw_landmarks(clip)
            state, _ = canonical_from_landmarks(
                load_raw_landmarks(raw), sample_id=base.replace("/", "-"), source_id=args.source_id,
                source_sha256=job["sha256"], clock_id=f"{args.source_id}:{base}:pts",
                inference_method=inference_method() + ARM_RECOVERY_METHOD, upright_chest=True, upright_root=True)
        except ValueError as error:
            report.update(status="refused", reason=str(error)[:300])
            store.put(report_key, json.dumps(report, indent=1).encode())
            return report
        path = Path(tmp) / "s.npz"
        digest = state.save(path)
        state_key, landmarks_key = f"{args.out}/states/{base}.npz", f"{args.out}/landmarks/{base}.npz"
        store.put(landmarks_key, raw)
        store.put(state_key, path.read_bytes())
    coverage, (face, yaw) = hand_coverage(state), face_stats(state)
    report.update(status="converted", state_key=state_key, state_sha256=digest, landmarks_key=landmarks_key,
                  landmarks_sha256=hashlib.sha256(raw).hexdigest(), frames=int(len(state.timestamps)),
                  seconds=float(state.timestamps[-1] - state.timestamps[0]), hand_coverage=round(coverage, 3),
                  face_coverage=round(face, 3), median_abs_yaw_deg=None if yaw is None else round(yaw, 1),
                  available_channels=[n for n, c in state.channels.items() if c.valid.any()])
    store.put(report_key, json.dumps(report, indent=1).encode())
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", required=True)
    p.add_argument("--metadata", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source-id", required=True)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--count-people-every", type=int, default=24)
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    jobs = jobs_of(store, args.source, args.metadata)
    jobs = [j for j in jobs if int(hashlib.sha256(j["key"].encode()).hexdigest(), 16) % args.shards == args.shard]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} videos (shard {args.shard}/{args.shards})", flush=True)
    counts = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(convert, j, args) for j in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            r = future.result()
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if n % 50 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} {counts}", flush=True)
    print(f"EXTRACTED {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
