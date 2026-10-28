"""Paired image/world landmarks from our Tasks trackers, for the learned world regressors (local CPU).

For How2Sign front clips (train and val zips, CC BY-NC 4.0; the test zip stays unread) and staged YouTube-ASL copies
that already exist in R2 (nothing is downloaded from YouTube), every frame is cropped in memory to Sem-Lex's webcam
framing (``validate_semlex_adapter.framing``; one shape per clip from 4:3, 16:9 and 1:1, by a seeded draw) and run
through ``extract_clip``'s trackers in ROI mode (pose heavy in video mode; one hand tracker per pose-wrist crop). Per
frame it records:

- pose: normalised image landmarks with z (33, 3), world landmarks (33, 3, metres, hip-centred), visibility (33);
- hands, per wrist crop (slot 0 = the pose's left wrist, 1 = right, as Holistic attaches hands): image landmarks with z
  mapped to the full frame (z scaled by crop width / frame width, as MediaPipe projects ROI landmarks), and world
  landmarks (21, 3, metres).

One ``.npz`` per clip with its signer id (How2Sign: the ``-<signer>-rgb_front`` suffix; YouTube: the video id).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

H2S = "acquisition/2026-10-17-v1/how2sign-public/{split}_rgb_front_clips.zip"
YOUTUBE_STAGED = "acquisition/2026-10-17-v1/youtube-asl-v1/staged/"
SHAPES = {"4:3": 4 / 3, "16:9": 16 / 9, "1:1": 1.}
_WORKER: dict = {}


def signer_of(name: str) -> str | None:
    match = re.search(r"-(\d+)-rgb_front\.mp4$", name)
    return f"h2s-{match.group(1)}" if match else None


def collect(video: Path, models: dict, aspect: float, *, max_frames: int = 0, hand_scale: float = .9) -> dict | None:
    """Paired arrays for one video cropped to Sem-Lex framing at ``aspect``; None when the framing does not fit."""
    import av
    import mediapipe as mp
    from signtranslator.data_engineering.video_landmarks import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST, square_crop
    from validate_semlex_adapter import framing
    box = framing(video, models, aspect)
    if box is None:
        return None
    x0, y0, w, h = box
    vision, base = mp.tasks.vision, mp.tasks.BaseOptions
    mode = vision.RunningMode.VIDEO
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=base(model_asset_path=str(models["pose_landmarker_heavy.task"])), running_mode=mode))
    hands = [vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=base(model_asset_path=str(models["hand_landmarker.task"])), running_mode=mode, num_hands=1))
        for _ in range(2)]
    blank = np.zeros((16, 16, 3), np.uint8)
    rows = []

    def image(a):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(a))
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        last_ms, last_t = -1, -float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * frame.time_base)
            if t <= last_t:
                continue
            last_t = t
            ms = max(int(round(t * 1000)), last_ms + 1)
            last_ms = ms
            rgb = np.ascontiguousarray(frame.to_ndarray(format="rgb24")[y0:y0 + h, x0:x0 + w])
            row = dict(pose_found=False, hand_found=np.zeros(2, bool), pose_img=np.zeros((33, 3)),
                       pose_world=np.zeros((33, 3)), pose_vis=np.zeros(33), hand_img=np.zeros((2, 21, 3)),
                       hand_world=np.zeros((2, 21, 3)))
            p = pose.detect_for_video(image(rgb), ms)
            crops = [None, None]
            if p.pose_landmarks:
                row["pose_found"] = True
                row["pose_img"] = np.array([[l.x, l.y, l.z] for l in p.pose_landmarks[0]])
                row["pose_vis"] = np.array([l.visibility for l in p.pose_landmarks[0]])
                row["pose_world"] = np.array([[l.x, l.y, l.z] for l in p.pose_world_landmarks[0]])
                px = row["pose_img"][:, :2] * [w, h]
                shoulder = float(np.linalg.norm(px[L_SHOULDER] - px[R_SHOULDER]))
                crops = [square_crop(rgb, px[k], hand_scale * shoulder) for k in (L_WRIST, R_WRIST)]
            for k, (tracker, item) in enumerate(zip(hands, crops)):
                crop, offset = item if item is not None else (None, None)
                r = tracker.detect_for_video(image(crop if crop is not None else blank), ms)
                if crop is None or not r.hand_landmarks:
                    continue
                cw, ch = crop.shape[1], crop.shape[0]
                lm = np.array([[l.x, l.y, l.z] for l in r.hand_landmarks[0]])
                row["hand_img"][k] = np.c_[(lm[:, 0] * cw + offset[0]) / w, (lm[:, 1] * ch + offset[1]) / h,
                                           lm[:, 2] * cw / w]
                row["hand_world"][k] = np.array([[l.x, l.y, l.z] for l in r.hand_world_landmarks[0]])
                row["hand_found"][k] = True
            rows.append(row)
            if max_frames and len(rows) >= max_frames:
                break
    for d in (pose, *hands):
        d.close()
    if not rows:
        return None
    out = {k: np.stack([r[k] for r in rows]).astype(np.float32 if np.asarray(rows[0][k]).dtype.kind == "f" else bool)
           for k in rows[0]}
    out["size"] = np.array([w, h])
    return out


def _init():
    from signtranslator.data.text_motion_dataset import R2Store
    _WORKER["store"] = R2Store(os.environ)
    _WORKER["archives"] = {}


def _job(job) -> dict:
    source, member, shape, signer, out_dir, models_dir, max_frames = job
    from signtranslator.data_engineering.video_landmarks import verify_models
    from extract_video_states import StoreRangeFile
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(member).stem)
    target = Path(out_dir) / f"{name}__{shape.replace(':', 'x')}.npz"
    if target.exists():
        return dict(name=target.stem, status="present")
    store = _WORKER["store"]
    if source.endswith(".zip"):
        archive = _WORKER["archives"].get(source)
        if archive is None:
            archive = zipfile.ZipFile(io.BufferedReader(StoreRangeFile(store.s3, store.bucket, source), 1 << 20))
            _WORKER["archives"][source] = archive
        payload = archive.read(member)
    else:
        payload = store.get(member)
    models = verify_models(Path(models_dir))
    with tempfile.TemporaryDirectory() as t:
        video = Path(t) / "clip.mp4"
        video.write_bytes(payload)
        try:
            arrays = collect(video, models, SHAPES[shape], max_frames=max_frames)
        except Exception as error:  # noqa: BLE001 - one unreadable clip must not stop the collection
            return dict(name=target.stem, status="error", reason=f"{type(error).__name__}: {error}"[:200])
    if arrays is None:
        return dict(name=target.stem, status="framing-does-not-fit")
    np.savez_compressed(target, signer=np.array(signer), clip=np.array(member), shape=np.array(shape), **arrays)
    return dict(name=target.stem, status="collected", frames=int(len(arrays["pose_found"])),
                pose=int(arrays["pose_found"].sum()), hands=int(arrays["hand_found"].sum()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--models-dir", default=os.path.expanduser("~/.sign-translator-ops/models"))
    p.add_argument("--clips-per-signer", type=int, default=150)
    p.add_argument("--youtube-max-frames", type=int, default=6000)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    from signtranslator.data.text_motion_dataset import R2Store
    from extract_video_states import StoreRangeFile
    store = R2Store(os.environ)
    rng = random.Random(args.seed)
    jobs = []
    for split in ("train", "val"):
        key = H2S.format(split=split)
        provenance = json.loads(store.get(key + ".provenance.json"))
        if not provenance.get("verified_by_r2_reread"):
            raise SystemExit(f"{key} is not a verified acquisition")
        archive = zipfile.ZipFile(io.BufferedReader(StoreRangeFile(store.s3, store.bucket, key), 1 << 20))
        by_signer: dict = {}
        for name in sorted(n for n in archive.namelist() if n.lower().endswith(".mp4")):
            signer = signer_of(name)
            if signer:
                by_signer.setdefault(signer, []).append(name)
        for signer, names in sorted(by_signer.items()):
            for name in rng.sample(names, min(len(names), args.clips_per_signer)):
                jobs.append((key, name, rng.choice(sorted(SHAPES)), signer, args.out_dir, args.models_dir, 0))
    for key in store.list(YOUTUBE_STAGED):
        if key.endswith((".mp4", ".webm", ".mkv")):
            jobs.append(("youtube", key, "4:3", f"yt-{Path(key).stem}", args.out_dir, args.models_dir,
                         args.youtube_max_frames))
    print(len(jobs), "clips;", len({j[3] for j in jobs}), "signers", flush=True)
    if args.dry_run:
        return 0
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as pool:
        for result in pool.map(_job, jobs, chunksize=4):
            print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
