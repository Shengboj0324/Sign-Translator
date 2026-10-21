"""Run Sapiens-1B (Goliath 308) on every clip of a How2Sign zip (Runpod GPU job).

Each clip is decoded on its own presentation timestamps; the signer's box comes from
the green screen per frame; frames are batched through the TorchScript model in
bf16 on CUDA; heatmaps are decoded with sub-pixel refinement and mapped back to the
image. Each frame is also run mirrored and the heatmaps averaged with left/right
channels swapped (flip test-time augmentation; better keypoints against CMU Panoptic
ground truth on two sequences, see docs/DATA_QUALITY_DECISION.md §6; ``--no-flip``
halves the compute). Keypoints (float32), confidences (float16) and times go write-once to the
object store as one NPZ per clip. Sharded and resumable like the MediaPipe job.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_video_states import StoreRangeFile  # noqa: E402
from signtranslator.data.text_motion_dataset import R2Store  # noqa: E402
from signtranslator.data_engineering.sapiens_keypoints import (  # noqa: E402
    crop_box,
)

MODEL_URL = ("https://huggingface.co/facebook/sapiens-pose-1b-torchscript/resolve/main/"
             "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2")


def load_model(path: Path, expected_sha256: str | None, device):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise SystemExit("Sapiens checkpoint differs from the pinned hash")
    model = torch.jit.load(str(path), map_location=device).eval()
    return model, digest


_POOLS: dict = {}


def _pool(workers: int):
    """One box-finding process pool per worker count, kept for the life of the job."""
    from concurrent.futures import ProcessPoolExecutor
    if workers not in _POOLS:
        _POOLS[workers] = ProcessPoolExecutor(workers)
    return _POOLS[workers]


@torch.no_grad()
def frames_keypoints(frames, model, device, batch: int, flip: bool = True, workers: int | None = None):
    """Keypoints of an iterable of RGB frames, run in batches as the frames stream in.

    A producer thread groups decoded frames into batches and finds each frame's
    green-screen box in a process pool (``workers``; the CPU's count by default);
    the crop, normalisation, model, flip average and heat-map decoding then run on
    the device for the whole batch (``sapiens_batched``), overlapping with the next
    batch's decoding. At most a few batches are held at a time, so whole raw
    recordings fit in memory. A frame without a green screen uses the whole frame.
    """
    import queue
    import threading
    from signtranslator.data_engineering.sapiens_batched import (
        decode_batch, flip_average_batch, green_screen_box_fast, preprocess_batch,
    )
    pool = _pool(workers or min(32, os.cpu_count() or 1))
    ready: queue.Queue = queue.Queue(maxsize=3)
    failure = []

    def produce():
        try:
            pending = []
            def emit():
                found = list(pool.map(green_screen_box_fast, pending, chunksize=max(1, len(pending) // 8)))
                boxes = [crop_box(b if b is not None else (0., 0., float(f.shape[1]), float(f.shape[0])))
                         for f, b in zip(pending, found)]
                ready.put((np.stack(pending), np.array(boxes, dtype=np.float64)))
                pending.clear()
            for rgb in frames:
                pending.append(np.ascontiguousarray(rgb))
                if len(pending) == batch:
                    emit()
            if pending:
                emit()
        except BaseException as error:          # surfaced in the consumer
            failure.append(error)
        finally:
            ready.put(None)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    points, confidence = [], []
    try:
        while (item := ready.get()) is not None:
            stack, boxes = item
            frames_t = torch.from_numpy(stack).to(device, non_blocking=True)
            boxes_t = torch.from_numpy(boxes).to(device)
            inputs = preprocess_batch(frames_t, boxes_t)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                heat = model(inputs).float()
                if flip:
                    heat = flip_average_batch(heat, model(torch.flip(inputs, dims=[3])).float())
            p, c = decode_batch(heat, boxes_t if device.type != "mps" else boxes_t.float())
            points.append(p.float().cpu().numpy())
            confidence.append(c.cpu().numpy())
    finally:
        thread.join()
    if failure:
        raise failure[0]
    if not points:
        return np.zeros((0, 308, 2), np.float32), np.zeros((0, 308), np.float16)
    return np.concatenate(points).astype(np.float32), np.concatenate(confidence).astype(np.float16)


def clip_keypoints(video: Path, model, device, batch: int, flip: bool = True, workers: int | None = None):
    import av
    times = []
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError("the file has no video stream")

        def frames():
            for frame in container.decode(container.streams.video[0]):
                times.append(float(frame.pts * frame.time_base))
                yield frame.to_ndarray(format="rgb24")

        points, confidence = frames_keypoints(frames(), model, device, batch, flip, workers)
    return np.array(times), points, confidence


CLAIM_SECONDS = 600


def claim(store, key: str) -> bool:
    """Claim a clip for this job unless another job claimed it less than ``CLAIM_SECONDS`` ago.

    Pods sharing a clip list would otherwise catch up with one another and then process
    the same clips in lockstep. A claim older than ``CLAIM_SECONDS`` (a crashed pod) is
    taken over. Two jobs claiming at once: the store's write-then-re-read check fails for
    the loser, which skips the clip (at worst both proceed, one duplicate clip).
    """
    import time
    now = time.time()
    if store.exists(key):
        try:
            if now - float(json.loads(store.get(key))["time"]) < CLAIM_SECONDS:
                return False
        except (ValueError, KeyError):
            pass
    try:
        store.put(key, json.dumps(dict(time=now, host=os.uname().nodename, pid=os.getpid())).encode())
    except IOError:                          # the re-read saw another job's claim: it won the race
        return False
    except Exception as error:               # R2 refuses concurrent writes to one object (503): another job is
        if "ServiceUnavailable" in str(error) or "concurrent request rate" in str(error):   # claiming it now
            return False
        raise
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-key", required=True)
    parser.add_argument("--zip-sha256", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="local path of the TorchScript file")
    parser.add_argument("--checkpoint-sha256", help="pin; recorded on first use if omitted")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--match", help="only clips whose file name matches this regular expression")
    parser.add_argument("--no-flip", dest="flip", action="store_false", help="disable flip test-time augmentation")
    parser.add_argument("--workers", type=int, help="green-screen box processes (default: min(32, CPUs))")
    parser.add_argument("--rotate", default="0/1", help="i/n: start at the i-th n-th of the clip list and wrap, so n "
                        "pods sharing one shard cover it from n places and finish by taking the others' remainders")
    args = parser.parse_args(argv)
    store = R2Store(os.environ)
    provenance = json.loads(store.get(args.zip_key + ".provenance.json"))
    if provenance.get("sha256") != args.zip_sha256 or not provenance.get("verified_by_r2_reread"):
        raise SystemExit("the zip is not the verified acquisition named by --zip-sha256")
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model, model_sha = load_model(args.checkpoint, args.checkpoint_sha256, device)
    archive = zipfile.ZipFile(io.BufferedReader(StoreRangeFile(store.s3, store.bucket, args.zip_key), 1 << 20))
    members = sorted(n for n in archive.namelist() if n.endswith(".mp4"))
    if args.match:
        members = [m for m in members if re.search(args.match, Path(m).name)]
    shard = [m for i, m in enumerate(members) if i % args.shards == args.shard][:args.limit]
    part, parts = (int(x) for x in args.rotate.split("/"))
    start = part * len(shard) // parts
    shard = shard[start:] + shard[:start]
    prefix = args.prefix.rstrip("/")
    done, refused = 0, []
    for index, member in enumerate(shard):
        key = f"{prefix}/keypoints/{Path(member).stem}.npz"
        if store.exists(key) or not claim(store, f"{prefix}/claims/{Path(member).stem}"):
            continue
        payload = archive.read(member)
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "clip.mp4"
            video.write_bytes(payload)
            try:
                times, points, confidence = clip_keypoints(video, model, device, args.batch, args.flip, args.workers)
                if len(times) == 0:
                    raise ValueError("the video stream has no frames")
            except (ValueError, OSError) as error:      # unreadable clip in the source: recorded, not fatal
                refused.append(dict(member=member, clip_sha256=hashlib.sha256(payload).hexdigest(),
                                    reason=str(error)[:300]))
                continue
            if store.exists(key):                       # another pod on the same rotation got here first
                continue
            buffer = io.BytesIO()
            np.savez_compressed(buffer, times=times, points=points, confidence=confidence,
                                clip_sha256=np.array(hashlib.sha256(payload).hexdigest()),
                                model_sha256=np.array(model_sha), flip_tta=np.array(args.flip))
            store.put(key, buffer.getvalue())
        done += 1
        if index % 50 == 0:
            print(f"{index}/{len(shard)} {member}", flush=True)
    rotation = "" if args.rotate == "0/1" else f"-rotate-{part:02d}-of-{parts:02d}"
    store.put(f"{prefix}/reports/{Path(args.zip_key).stem}-shard-{args.shard:03d}-of-{args.shards:03d}{rotation}.json", json.dumps(dict(
        zip_key=args.zip_key, zip_sha256=args.zip_sha256, model_sha256=model_sha, flip_tta=args.flip, clips=len(shard),
        written=done, refused=refused)).encode())
    print(json.dumps(dict(model_sha256=model_sha, written=done, refused=len(refused), clips=len(shard))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
