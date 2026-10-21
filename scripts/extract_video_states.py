"""Estimate co-observed canonical states for every clip in a video zip (Runpod job).

Reads a publisher zip of clips from the object store by ranged reads (nothing is
staged beyond one clip per worker), runs the MediaPipe landmarkers with ROI crops,
converts to canonical states with :mod:`video_landmarks`, and writes each state
write-once under ``--prefix``. The job is sharded (``--shard i --shards n``) and
resumable: a clip whose state already exists is skipped. Models are fetched once
and checked against their pinned hashes. A per-shard report lists every clip,
converted or refused, with its reason.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_WORKER = {}


class StoreRangeFile(io.RawIOBase):
    """Seekable read-only view of one store object (R2 ranged GETs)."""

    def __init__(self, s3, bucket, key):
        self.s3, self.bucket, self.key, self.position = s3, bucket, key, 0
        self.size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = {0: offset, 1: self.position + offset, 2: self.size + offset}[whence]
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size is None or size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        data = self._ranged(self.position, size)
        if len(data) != size:
            raise IOError("short ranged read")
        self.position += size
        return data

    def _ranged(self, start: int, size: int, attempts: int = 12) -> bytes:
        """One ranged GET, retried with jittered exponential backoff when R2 throttles.

        R2 answers many workers reading the same object with ServiceUnavailable
        ("reduce your rate of simultaneous reads"); that is transient, not a failure.
        """
        import random
        import time
        from botocore.exceptions import ClientError
        for attempt in range(attempts):
            try:
                return self.s3.get_object(Bucket=self.bucket, Key=self.key,
                                          Range=f"bytes={start}-{start + size - 1}")["Body"].read()
            except ClientError as error:
                code = error.response.get("Error", {}).get("Code", "")
                if code not in ("ServiceUnavailable", "SlowDown", "503", "InternalError") or attempt == attempts - 1:
                    raise
                time.sleep(min(30., .5 * 2 ** attempt) * (.5 + random.random()))
        raise AssertionError("unreachable")

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def _init(zip_key, models_dir):
    import torch
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.video_landmarks import probe_runtime, verify_models
    torch.set_num_threads(1)
    store = R2Store(os.environ)
    models = verify_models(Path(models_dir))
    probe_runtime(models)               # an environment failure stops the job; it is not a clip refusal
    _WORKER.update(store=store, models=models,
                   archive=zipfile.ZipFile(io.BufferedReader(StoreRangeFile(store.s3, store.bucket, zip_key), 1 << 20)))


def _process(job):
    member, prefix, source_id, zip_sha256, upright_chest = job
    from signtranslator.data_engineering.video_landmarks import canonical_from_landmarks, extract_clip, inference_method
    store, archive = _WORKER["store"], _WORKER["archive"]
    sample = Path(member).stem
    key = f"{prefix}/states/{sample}.npz"
    if store.exists(key):
        # Resumed shard: report the stored state's hash, so the report stays a complete,
        # verifiable index of the shard (the manifest builder skips entries without one).
        from signtranslator.pose.multichannel import MultichannelMotion
        payload = store.get(key)
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.npz"
            path.write_bytes(payload)
            state = MultichannelMotion.load(path, expected_sha256=digest)
        return dict(sample_id=sample, status="already_present", state_key=key, state_sha256=digest,
                    seconds=float(state.timestamps[-1] - state.timestamps[0]),
                    available_channels=[n for n, c in state.channels.items() if c.valid.any()])
    payload = archive.read(member)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "clip.mp4"
            video.write_bytes(payload)
            clip = extract_clip(video, _WORKER["models"], roi=True)
            state, report = canonical_from_landmarks(
                clip, sample_id=sample, source_id=source_id, source_sha256=hashlib.sha256(payload).hexdigest(),
                clock_id=f"{source_id}:{sample}:pts", upright_chest=upright_chest,
                inference_method=inference_method() + ("; chest up axis from the camera vertical (upright prior)"
                                                        if upright_chest else ""))
            path = Path(temporary) / "state.npz"
            digest = state.save(path)
            store.put(key, path.read_bytes())
            # Raw monocular landmarks, kept for the two-view quality gate's comparison.
            buffer = io.BytesIO()
            np.savez_compressed(buffer, times=clip.times, pose_world=clip.pose_world,
                                pose_visibility=clip.pose_visibility, pose_found=clip.pose_found)
            store.put(f"{prefix}/landmarks/{sample}.npz", buffer.getvalue())
    except (ValueError, OSError) as error:
        return dict(sample_id=sample, status="refused", reason=str(error)[:300])
    available = [n for n, c in state.channels.items() if c.valid.any()]
    return dict(sample_id=sample, status="converted", state_key=key, state_sha256=digest,
                clip_sha256=hashlib.sha256(payload).hexdigest(), seconds=float(state.timestamps[-1] - state.timestamps[0]),
                available_channels=available, **report)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-key", required=True)
    parser.add_argument("--zip-sha256", required=True, help="the verified hash recorded at acquisition")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--source-id", default="how2sign_video_estimate")
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--upright-chest", action="store_true",
                        help="framings cut above the hips: chest up axis from the camera vertical (solve_rotations)")
    parser.add_argument("--match", help="only clips whose file name matches this regular expression")
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        raise SystemExit("shard must be in [0, shards)")
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    provenance = json.loads(store.get(args.zip_key + ".provenance.json"))
    if provenance.get("sha256") != args.zip_sha256 or not provenance.get("verified_by_r2_reread"):
        raise SystemExit("the zip is not the verified acquisition named by --zip-sha256")
    archive = zipfile.ZipFile(io.BufferedReader(StoreRangeFile(store.s3, store.bucket, args.zip_key), 1 << 20))
    members = sorted(n for n in archive.namelist() if n.endswith(".mp4"))
    if args.match:
        members = [m for m in members if re.search(args.match, Path(m).name)]
    shard = [m for i, m in enumerate(members) if i % args.shards == args.shard][:args.limit]
    print(f"{len(members)} clips, shard {args.shard}/{args.shards}: {len(shard)}", flush=True)
    jobs = [(m, args.prefix.rstrip("/"), args.source_id, args.zip_sha256, args.upright_chest) for m in shard]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.zip_key, args.models_dir)) as pool:
        for i, result in enumerate(pool.map(_process, jobs, chunksize=4)):
            results.append(result)
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} {result['status']}", flush=True)
    report = dict(schema_version=1, zip_key=args.zip_key, zip_sha256=args.zip_sha256, shard=args.shard,
                  upright_chest=args.upright_chest,
                  shards=args.shards, clips=len(results),
                  status_counts={s: sum(r["status"] == s for r in results) for s in {r["status"] for r in results}},
                  results=results)
    store.put(f"{args.prefix.rstrip('/')}/reports/{Path(args.zip_key).stem}-shard-{args.shard:03d}-of-{args.shards:03d}.json",
              json.dumps(report, sort_keys=True).encode())
    print(json.dumps(report["status_counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
