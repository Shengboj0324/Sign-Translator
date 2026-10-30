"""Convert a stored sentence-video corpus (one sentence per video) to canonical motion states, as the YouTube-ASL
pipeline converts a caption cue (``acquire_youtube_asl.py``): MediaPipe Tasks landmarkers over every frame with
ROI crops (``video_landmarks.extract_clip``), the raw landmarks stored, and the state solved from exactly those stored
landmarks with the arm-recovery conversion (upright chest and root; ``ARM_RECOVERY_METHOD``).

Input: a prefix written by ``acquire_hf_dataset.py`` (``INDEX.json``, ``files/...``) and its metadata CSVs
(``--metadata`` relative paths; columns ``file_name`` plus the corpus's text fields, kept verbatim). Each video gets
``<out>/states/<split>/<stem>.npz``, ``<out>/landmarks/<split>/<stem>.npz`` and a report
``<out>/videos/<split>/<stem>.json`` (text fields, source key and SHA-256, frames, seconds, hand and face coverage,
available channels; or the refusal). A video with a report is skipped (resumable); ``--shard i --shards n`` splits the
work by a hash of the path. Videos are fetched to a temporary file and deleted after conversion. ``--transcode HxF``
(e.g. ``720x24``) first re-encodes each video with ffmpeg to that height and frame rate (How2Sign's 720p at 24 fps):
2M-Flores is 1080p at 60 fps, about five times the landmarking work for no gain at the generator's 24 fps; the
setting is recorded in each report and in the inference method.

**Zip mode** (``--zip-key``): the videos are members of one stored zip (ASL STEM Wiki), read by ranged GETs without
unpacking; ``--zip-metadata`` is the member holding the CSV, ``--file-column`` its file-name column and
``--member-prefix`` the members' directory. Each record's split is ``--group-column``'s value (e.g. the participant),
so a downstream manifest can hold out whole signers; reports go under ``<out>/videos/<group>/<stem>.json``.

    python scripts/extract_video_corpus.py --source acquisition/2026-10-30-v1/2m-flores-asl \
        --metadata data/dev/metadata.csv --metadata data/devtest/metadata.csv \
        --out acquisition/2026-10-30-v1/2m-flores-asl/extract-v1 --source-id 2m-flores-asl --models-dir /root/work/models
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
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


_ZIP = {}


def zip_of(store, key: str):
    """One ranged-read ZipFile per process (the central directory is read once)."""
    if key not in _ZIP:
        from build_uwb_keypoint_shards import open_zip
        _ZIP[key] = open_zip(store, key)
    return _ZIP[key]


def zip_jobs_of(store, args) -> list[dict]:
    z = zip_of(store, args.zip_key)
    names = {i.filename: i for i in z.infolist()}
    jobs = []
    for row in csv.DictReader(io.StringIO(z.read(args.zip_metadata).decode("utf-8"))):
        member = args.member_prefix + row[args.file_column]
        if member not in names:
            raise SystemExit(f"{member} is listed in {args.zip_metadata} but not in the zip")
        jobs.append(dict(split=row[args.group_column], stem=Path(row[args.file_column]).stem, key=member, sha256=None,
                         member_crc=names[member].CRC, fields={k: v for k, v in row.items() if k != args.file_column}))
    return jobs


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
        if args.zip_key:
            import time as _time, zlib
            for attempt in range(5):               # transient TLS/HTTP errors on ranged reads: reopen and retry
                try:
                    payload = zip_of(store, args.zip_key).read(job["key"])
                    break
                except Exception:
                    _ZIP.pop(args.zip_key, None)
                    if attempt == 4:
                        raise
                    _time.sleep(5 * (attempt + 1))
            if zlib.crc32(payload) & 0xFFFFFFFF != job["member_crc"]:
                raise SystemExit(f"{job['key']}: CRC differs from the zip directory")
            job = dict(job, sha256=hashlib.sha256(payload).hexdigest())
            report.update(source_key=f"{args.zip_key}!{job['key']}", source_sha256=job["sha256"])
        else:
            payload = store.get(job["key"])
        if job["sha256"] and hashlib.sha256(payload).hexdigest() != job["sha256"]:
            raise SystemExit(f"{job['key']}: SHA-256 differs from the index")
        video.write_bytes(payload)
        del payload
        if args.transcode:
            import subprocess
            height, fps = (int(x) for x in args.transcode.split("x"))
            small = Path(tmp) / "t.mp4"
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video), "-an", "-vf",
                            f"scale=-2:{height},fps={fps}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
                            str(small)], check=True)
            video.unlink()
            video = small
            report["transcode"] = dict(height=height, fps=fps, codec="libx264 veryfast crf 17")
        try:
            clip = extract_clip(video, models, roi=True, count_people_every=args.count_people_every)
            raw = save_raw_landmarks(clip)
            state, _ = canonical_from_landmarks(
                load_raw_landmarks(raw), sample_id=base.replace("/", "-"), source_id=args.source_id,
                source_sha256=job["sha256"], clock_id=f"{args.source_id}:{base}:pts",
                inference_method=inference_method() + ARM_RECOVERY_METHOD + (
                    f"; transcoded to {args.transcode} (height x fps) before landmarking" if args.transcode else ""),
                upright_chest=True, upright_root=True)
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


def auto_workers(per_worker_gb: float = 4.0) -> int:
    """Workers that fit this container: CPUs minus two, and its cgroup memory limit at ``per_worker_gb`` each (a pod
    may come up with fewer CPUs and less memory than asked; too many workers get OOM-killed and break the pool)."""
    cpus = max(1, (os.cpu_count() or 2) - 2)
    limit = None
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
            if raw.isdigit() and int(raw) < 1 << 50:
                limit = int(raw) / 2 ** 30
                break
        except OSError:
            continue
    if limit is None:
        limit = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    return max(1, min(cpus, int(limit // per_worker_gb)))


def run_resilient(fn, jobs, args, workers: int, progress, *, max_attempts: int = 2, executor=ProcessPoolExecutor):
    """Run ``fn(job, args)`` over ``jobs`` in a process pool, calling ``progress(n, result)`` as each ends.

    At most ``2 x workers`` jobs are in flight. If a worker dies (for example, OOM-killed), the pool breaks and every
    in-flight job raises BrokenProcessPool, so the culprit cannot be told from the rest: those jobs become suspects and
    run one at a time on a fresh pool. A job that breaks the pool while alone in flight is the culprit; it counts as
    failed after ``max_attempts`` such breaks. Innocent suspects simply finish. One crash used to fail every pending
    video (3,170 of 5,252 in one pass). Any other exception fails just that job. Failed jobs have no report, so a rerun
    retries them."""
    queue = list(reversed(range(len(jobs))))           # job indices: strikes are counted per index
    suspects, strikes, done = [], {}, 0
    while queue or suspects:
        pool = executor(workers)
        broken = False
        inflight = {}
        try:
            while (queue or suspects or inflight) and not broken:
                solo = bool(suspects)                       # suspects run alone, so a break names its culprit
                while not inflight and suspects:
                    k = suspects.pop()
                    inflight[pool.submit(fn, jobs[k], args)] = k
                while not solo and queue and len(inflight) < 2 * workers:
                    k = queue.pop()
                    inflight[pool.submit(fn, jobs[k], args)] = k
                if not inflight:
                    break
                finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in finished:
                    k = inflight.pop(future)
                    try:
                        r = future.result()
                    except BrokenProcessPool:
                        broken = True
                        if solo:                            # alone in flight: this job killed its worker
                            strikes[k] = strikes.get(k, 0) + 1
                            if strikes[k] < max_attempts:
                                suspects.append(k)
                                continue
                            r = dict(status="failed")
                            print(f"FAILED one video: killed its worker {strikes[k]} times", flush=True)
                        else:
                            suspects.append(k)
                            continue
                    except Exception as error:  # noqa: BLE001 - one video's transient failure must not end the run
                        r = dict(status="failed")
                        print(f"FAILED one video: {type(error).__name__}: {str(error)[:160]}", flush=True)
                    done += 1
                    progress(done, r)
            if broken:   # the other in-flight jobs are lost with the pool: they become suspects too
                suspects.extend(inflight.values())
                print(f"worker pool broke (a worker died, likely OOM): {len(suspects)} suspects rerun one at a time "
                      f"on a fresh pool", flush=True)
        finally:
            pool.shutdown(wait=not broken, cancel_futures=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", help="prefix written by acquire_hf_dataset.py (INDEX.json, files/...)")
    p.add_argument("--metadata", action="append", help="metadata CSVs under the source's files/")
    p.add_argument("--zip-key", help="zip mode: the stored zip holding the videos")
    p.add_argument("--zip-metadata", help="zip mode: the member holding the metadata CSV")
    p.add_argument("--file-column", default="file_name")
    p.add_argument("--member-prefix", default="")
    p.add_argument("--group-column", help="zip mode: the column used as each record's split (e.g. participant)")
    p.add_argument("--out", required=True)
    p.add_argument("--source-id", required=True)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--workers", default="8", help="a number, or 'auto' (auto_workers: CPUs and cgroup memory)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--count-people-every", type=int, default=24)
    p.add_argument("--transcode", help="HEIGHTxFPS, e.g. 720x24: ffmpeg re-encode before landmarking")
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    if args.zip_key:
        if not (args.zip_metadata and args.group_column):
            raise SystemExit("zip mode needs --zip-metadata and --group-column")
        jobs = zip_jobs_of(store, args)
    else:
        if not (args.source and args.metadata):
            raise SystemExit("--source and --metadata are required (or zip mode)")
        jobs = jobs_of(store, args.source, args.metadata)
    jobs = [j for j in jobs if int(hashlib.sha256(j["key"].encode()).hexdigest(), 16) % args.shards == args.shard]
    if args.limit:
        jobs = jobs[:args.limit]
    args.workers = auto_workers() if args.workers == "auto" else int(args.workers)
    print(f"{len(jobs)} videos (shard {args.shard}/{args.shards}), {args.workers} workers", flush=True)
    counts = {}

    def progress(n, r):
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if n % 50 == 0 or n == len(jobs):
            print(f"{n}/{len(jobs)} {counts}", flush=True)

    run_resilient(convert, jobs, args, args.workers, progress)
    print(f"EXTRACTED {counts}", flush=True)
    return 3 if counts.get("failed") else 0           # failed videos have no report: a rerun picks them up


if __name__ == "__main__":
    raise SystemExit(main())
