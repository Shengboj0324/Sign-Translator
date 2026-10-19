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
    crop_box, decode_heatmaps, flip_average, green_screen_box, preprocess,
)

MODEL_URL = ("https://huggingface.co/facebook/sapiens-pose-1b-torchscript/resolve/main/"
             "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2")


def load_model(path: Path, expected_sha256: str | None, device):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise SystemExit("Sapiens checkpoint differs from the pinned hash")
    model = torch.jit.load(str(path), map_location=device).eval()
    return model, digest


@torch.no_grad()
def clip_keypoints(video: Path, model, device, batch: int, flip: bool = True):
    import av
    times, frames, boxes = [], [], []
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError("the file has no video stream")
        for frame in container.decode(container.streams.video[0]):
            rgb = frame.to_ndarray(format="rgb24")
            box = green_screen_box(rgb)
            if box is None:
                box = (0., 0., float(rgb.shape[1]), float(rgb.shape[0]))
            times.append(float(frame.pts * frame.time_base))
            frames.append(rgb)
            boxes.append(crop_box(box))
    points = np.zeros((len(frames), 308, 2), np.float32)
    confidence = np.zeros((len(frames), 308), np.float16)
    for start in range(0, len(frames), batch):
        chunk = range(start, min(start + batch, len(frames)))
        inputs = torch.cat([preprocess(frames[i], boxes[i]) for i in chunk]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            heatmaps = model(inputs).float().cpu().numpy()
            mirrored = model(torch.flip(inputs, dims=[3])).float().cpu().numpy() if flip else None
        for k, i in enumerate(chunk):
            heat = flip_average(heatmaps[k], mirrored[k]) if flip else heatmaps[k]
            p, c = decode_heatmaps(heat, boxes[i])
            points[i], confidence[i] = p, c
    return np.array(times), points, confidence


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
    prefix = args.prefix.rstrip("/")
    done, refused = 0, []
    for index, member in enumerate(shard):
        key = f"{prefix}/keypoints/{Path(member).stem}.npz"
        if store.exists(key):
            continue
        payload = archive.read(member)
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "clip.mp4"
            video.write_bytes(payload)
            try:
                times, points, confidence = clip_keypoints(video, model, device, args.batch, args.flip)
                if len(times) == 0:
                    raise ValueError("the video stream has no frames")
            except (ValueError, OSError) as error:      # unreadable clip in the source: recorded, not fatal
                refused.append(dict(member=member, clip_sha256=hashlib.sha256(payload).hexdigest(),
                                    reason=str(error)[:300]))
                continue
            buffer = io.BytesIO()
            np.savez_compressed(buffer, times=times, points=points, confidence=confidence,
                                clip_sha256=np.array(hashlib.sha256(payload).hexdigest()),
                                model_sha256=np.array(model_sha), flip_tta=np.array(args.flip))
            store.put(key, buffer.getvalue())
        done += 1
        if index % 50 == 0:
            print(f"{index}/{len(shard)} {member}", flush=True)
    store.put(f"{prefix}/reports/{Path(args.zip_key).stem}-shard-{args.shard:03d}-of-{args.shards:03d}.json", json.dumps(dict(
        zip_key=args.zip_key, zip_sha256=args.zip_sha256, model_sha256=model_sha, flip_tta=args.flip, clips=len(shard),
        written=done, refused=refused)).encode())
    print(json.dumps(dict(model_sha256=model_sha, written=done, refused=len(refused), clips=len(shard))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
