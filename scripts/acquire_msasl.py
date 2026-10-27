"""MS-ASL (Microsoft, C-UDA; owner-approved 2026-10-27): isolated-sign clips from YouTube, as canonical motion states.

``MS-ASL.zip`` lists 25,513 labelled clips (1,000 classes, 222 signers), each a time window ``start_time``..``end_time``
in one of 7,213 YouTube videos. Two stages, as for YouTube-ASL (``acquire_youtube_asl.py``):

- ``--stage download`` runs on the owner's Mac (the approved home route). One video at a time goes through the same
  ``PoliteGate`` (``--download-interval`` seconds apart; on a bot check or sign-in request every worker waits
  ``--blocked-backoff``, and after ``--max-blocks`` refusals it stops). There are no cookies, sign-in, proxies or
  address changes. The whole video (at most ``--max-height`` px, no audio) is staged under ``<prefix>/staged/`` with the
  video's clips. Videos with the most clips go first.
- ``--stage extract`` runs on CPU pods. It claims a staged video, runs the MediaPipe landmarkers on each clip's window
  (``--pad`` seconds either side) and converts it with the upright-chest prior, since many videos cut off the hips. It
  writes ``<prefix-derived>/states/<video>-<k>.npz`` and a per-video report with every clip's label, gloss, signer,
  split and tracking counts, then deletes the staged video. Videos are never kept; only derived motion and labels
  are, privately, for computational use under C-UDA.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from acquire_youtube_asl import BLOCK_PATTERNS, Blocked, PoliteGate, claim, drop_staged, release  # noqa: E402

ZIP_KEY = "acquisition/2026-10-17-v1/ms-asl-v1/MS-ASL.zip"
LICENCE = "MS-ASL (Microsoft Research): Computational Use of Data Agreement (C-UDA 0.1); videos owned by their YouTube uploaders; private store, computational use only"


def video_id(url: str) -> str | None:
    import re
    m = re.search(r"(?:v=|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})", url)
    return m[1] if m else None


def clips_by_video(store) -> dict:
    """video id -> list of clip dicts (split, label, text, signer_id, start_time, end_time, box)."""
    archive = zipfile.ZipFile(io.BytesIO(store.get(ZIP_KEY)))
    out = collections.defaultdict(list)
    for split in ("train", "val", "test"):
        for k, c in enumerate(json.loads(archive.read(f"MS-ASL/MSASL_{split}.json"))):
            vid = video_id(c["url"])
            if vid:
                out[vid].append(dict(c, split=split, clip_index=f"{split}-{k}"))
    return out


def download_video(vid: str, directory: Path, max_height: int) -> dict:
    import yt_dlp
    opts = dict(quiet=True, no_warnings=True, noprogress=True, outtmpl=str(directory / "%(id)s.%(ext)s"),
                format=f"bv*[height<={max_height}][ext=mp4]/bv*[height<={max_height}]/b[height<={max_height}]",
                noplaylist=True, retries=3)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
    except Exception as error:  # noqa: BLE001 - yt-dlp raises DownloadError for every refusal
        message = str(error).lower()
        if "confirm your age" in message or "age-restricted" in message or "inappropriate for some users" in message:
            return dict(status="age_restricted", reason=str(error)[:300])
        if "private video" in message:
            return dict(status="private", reason=str(error)[:300])
        if any(p in message for p in BLOCK_PATTERNS):
            raise Blocked(str(error)[:300]) from None
        status = "removed" if any(w in message for w in ("unavailable", "not available", "removed", "terminated",
                                                           "does not exist")) else "error"
        return dict(status=status, reason=str(error)[:300])
    files = [p for p in directory.iterdir() if p.suffix in (".mp4", ".webm", ".mkv") and p.stem.startswith(vid)]
    meta = dict(channel_id=info.get("channel_id"), channel=info.get("channel"), upload_date=info.get("upload_date"),
                duration=info.get("duration"), license=info.get("license"), title=info.get("title"),
                height=info.get("height"), fps=info.get("fps"))
    return dict(status="downloaded", video=files[0], meta=meta) if files else dict(status="error", reason="no file")


def stage_one(vid: str, clips: list, args, store, gate) -> dict:
    if store.exists(f"{args.prefix}/staged/{vid}.json") or store.exists(f"{args.derived}/videos/{vid}.json"):
        return dict(video_id=vid, status="already_done")
    with tempfile.TemporaryDirectory() as tmp:
        with gate:
            got = download_video(vid, Path(tmp), args.max_height)
        if got["status"] != "downloaded":
            store.put(f"{args.derived}/videos/{vid}.json", json.dumps(dict(video_id=vid, clips=len(clips), **got)).encode())
            return dict(video_id=vid, status=got["status"])
        payload = got["video"].read_bytes()
        key = f"{args.prefix}/staged/{vid}{got['video'].suffix}"
        store.put(key, payload)
        store.put(f"{args.prefix}/staged/{vid}.json", json.dumps(dict(
            video_id=vid, video_key=key, video_sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
            meta=got["meta"], clips=clips, staged_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            sort_keys=True).encode())
    return dict(video_id=vid, status="staged", bytes=len(payload), clips=len(clips))


def extract_one(vid: str, args, store, models) -> dict:
    from signtranslator.data_engineering.video_landmarks import canonical_from_landmarks, extract_clip, inference_method
    staged = json.loads(store.get(f"{args.prefix}/staged/{vid}.json"))
    payload = store.get(staged["video_key"])
    if hashlib.sha256(payload).hexdigest() != staged["video_sha256"]:
        raise ValueError("staged video differs from its record")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(staged["video_key"]).name
        path.write_bytes(payload)
        for k, c in enumerate(staged["clips"]):
            start, stop = max(float(c["start_time"]) - args.pad, 0.), float(c["end_time"]) + args.pad
            row = dict(clip_index=c["clip_index"], split=c["split"], label=c["label"], text=c.get("clean_text") or c["text"],
                       signer_id=c.get("signer_id"), start_time=c["start_time"], end_time=c["end_time"])
            try:
                clip = extract_clip(path, models, start_seconds=start, stop_seconds=stop)
                if len(clip.times) < 4:
                    results.append(dict(row, status="too_short")); continue
                sample = f"{vid}-{k}"
                state, report = canonical_from_landmarks(
                    clip, sample_id=sample, source_id="ms-asl-v1", source_sha256=staged["video_sha256"],
                    clock_id=f"{vid}-video-seconds", inference_method=inference_method(), upright_chest=True)
                out = Path(tmp) / f"{sample}.npz"
                digest = state.save(out)
                key = f"{args.derived}/states/{sample}.npz"
                store.put(key, out.read_bytes())
                results.append(dict(row, status="converted", state_key=key, state_sha256=digest,
                                    seconds=float(state.timestamps[-1] - state.timestamps[0]),
                                    available_channels=[n for n, ch in state.channels.items() if ch.valid.any()],
                                    **{q: report[q] for q in ("frames", "pose_frames", "face_frames", "left_hand_frames",
                                                              "right_hand_frames", "arm_frames")}))
            except Exception as error:  # noqa: BLE001 - one bad window does not lose the video
                results.append(dict(row, status="error", error=f"{type(error).__name__}: {error}"[:200]))
    store.put(f"{args.derived}/videos/{vid}.json", json.dumps(dict(
        video_id=vid, status="extracted", video_sha256=staged["video_sha256"], meta=staged["meta"], license=LICENCE,
        clips=results), sort_keys=True).encode())
    drop_staged(vid, args, store, staged["video_key"])
    return dict(video_id=vid, status="extracted", converted=sum(r["status"] == "converted" for r in results),
                clips=len(results))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage", choices=("download", "extract"), required=True)
    p.add_argument("--prefix", default="acquisition/2026-10-17-v1/ms-asl-v1")
    p.add_argument("--derived", default="acquisition/2026-10-17-v1/derived/ms-asl-v1")
    p.add_argument("--max-height", type=int, default=720)
    p.add_argument("--pad", type=float, default=.2)
    p.add_argument("--polite-dir", help="download gate directory (download stage)")
    p.add_argument("--download-interval", type=float, default=60.)
    p.add_argument("--blocked-backoff", type=float, default=7200.)
    p.add_argument("--max-blocks", type=int, default=3)
    p.add_argument("--limit", type=int)
    p.add_argument("--models-dir", help="MediaPipe models (extract stage)")
    p.add_argument("--workers", type=int, default=1, help="extract: videos at once in this process (run several processes for more)")
    p.add_argument("--idle-exit", type=float, default=3600., help="extract: stop after this long with nothing staged")
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    if args.stage == "download":
        by_video = clips_by_video(store)
        order = sorted(by_video, key=lambda v: (-len(by_video[v]), v))[:args.limit]
        gate = PoliteGate(args.polite_dir, interval=args.download_interval, backoff=args.blocked_backoff)
        blocks = 0
        for vid in order:
            try:
                r = stage_one(vid, by_video[vid], args, store, gate)
            except Blocked as error:
                blocks += 1; gate.mark_blocked()
                print(json.dumps(dict(video_id=vid, status="BLOCKED", reason=str(error))), flush=True)
                if blocks >= args.max_blocks:
                    return 3
                continue
            print(json.dumps(r), flush=True)
        return 0
    from concurrent.futures import ThreadPoolExecutor
    from signtranslator.data_engineering.video_landmarks import verify_models
    models = verify_models(Path(args.models_dir))
    idle_since = time.time()
    while True:
        staged = sorted(Path(k).stem for k in store.list(f"{args.prefix}/staged/") if k.endswith(".json"))
        mine = [v for v in staged if claim(v, args, store)][:args.workers * 2]
        if not mine:
            if time.time() - idle_since > args.idle_exit:
                return 0
            time.sleep(60); continue
        idle_since = time.time()
        with ThreadPoolExecutor(args.workers) as pool:
            for r in pool.map(lambda v: _safe_extract(v, args, store, models), mine):
                print(json.dumps(r), flush=True)


def _safe_extract(vid, args, store, models):
    try:
        return extract_one(vid, args, store, models)
    except Exception as error:  # noqa: BLE001 - released for a later attempt
        release(vid, args, store)
        return dict(video_id=vid, status="error", error=f"{type(error).__name__}: {error}"[:300])


if __name__ == "__main__":
    raise SystemExit(main())
