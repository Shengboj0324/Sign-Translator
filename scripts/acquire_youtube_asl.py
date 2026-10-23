"""Acquire a YouTube-hosted ASL corpus as motion states + English captions (plan item 7; owner-approved for research).

For every video id of ``--ids`` (one per line; sharded with ``--shard i --shards n``):

1. ``yt-dlp`` fetches the video (best stream at most ``--max-height`` px, no audio) and its English captions
   (uploader captions, whose track is usually filed under the sign language's code ``ase``, or ``sgn-US`` / ``en*``;
   auto-generated ones are taken only with ``--allow-auto-captions`` and are marked),
   plus the metadata a release needs: channel, upload date, duration and YouTube's licence field;
2. the MediaPipe landmarkers run once over the whole video (``video_landmarks.extract_clip``, ROI crops);
3. each caption cue is cut out of those landmarks by its own clock (``--min-seconds``..``--max-seconds``, with
   ``--pad`` seconds either side) and converted to a canonical state, written under
   ``<prefix>/states/<id>-<cue>.npz``;
4. the video is deleted; ``<prefix>/videos/<id>.json`` records provenance, every cue (text, times, state key and
   hash, hand coverage) and refusals. A video with a report is skipped on a rerun (resumable).

**Polite and no circumvention.** With ``--polite-dir`` every download on a machine goes through one gate: one at a time,
``--download-interval`` seconds apart (20 s: at most 180 videos/hour per machine, under YouTube's ~300/hour guest
rate), while extraction runs in parallel on the downloaded files. When YouTube asks for sign-in or a bot check
("confirm you're not a bot", HTTP 429, "Sign in"), the refusal is logged as ``BLOCKED`` and every worker on that machine
waits ``--blocked-backoff`` (1 h) before the next request; after ``--max-blocks`` refusals the shard stops. No cookies,
sign-in, proxies or address changes are ever used. Private, removed or caption-less videos are
recorded as such (the corpus's availability rate is part of the pilot's result).

Research-only data: the prefix is private; videos are never stored or redistributed, only the derived motion and the
caption text with their provenance.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BLOCK_PATTERNS = ("confirm you", "not a bot", "sign in to", "http error 429", "too many requests")
# Uploaders of signed videos often file the English caption track under the sign language's own code: most
# YouTube-ASL tracks are "ase" (American Sign Language), some "sgn-US"; English codes are taken as well.
CAPTION_LANGS = ("ase", "sgn-US", "en", "en-US", "en-GB", "en-CA", "en-AU")


def is_caption_lang(code: str) -> bool:
    return code in CAPTION_LANGS or code.startswith("en")


class Blocked(RuntimeError):
    """YouTube refused the datacenter client: stop, never work around it."""


class PoliteGate:
    """One download at a time per machine, at least ``interval`` seconds apart (an flock'd stamp file shared by all
    workers), so a pod stays under YouTube's guest rate however many extraction workers it runs; and, once any worker
    is refused, every worker waits ``backoff`` seconds before the next request (``blocked-until`` stamp). This only
    respects the service's limits: no cookies, sign-in, proxies or address changes are used to get around a refusal."""

    def __init__(self, directory: str, *, interval: float, backoff: float):
        self.dir, self.interval, self.backoff = Path(directory), interval, backoff
        self.dir.mkdir(parents=True, exist_ok=True)

    def _stamp(self, name: str) -> float:
        try:
            return float((self.dir / name).read_text() or 0)
        except (OSError, ValueError):
            return 0.

    def wait_blocked(self) -> float:
        until = self._stamp("blocked-until")
        if until > time.time():
            time.sleep(until - time.time())
        return until

    def mark_blocked(self) -> None:
        (self.dir / "blocked-until").write_text(str(time.time() + self.backoff))

    def __enter__(self):
        import fcntl
        import threading
        self.wait_blocked()
        fh = open(self.dir / "download.lock", "a+")
        fcntl.flock(fh, fcntl.LOCK_EX)                     # downloads are serialised across workers (processes)
        self.wait_blocked()                                 # a refusal while this worker queued is respected too
        wait = self._stamp("last-download") + self.interval - time.time()
        if wait > 0:
            time.sleep(wait)
        self._held = getattr(self, "_held", {})
        self._held[threading.get_ident()] = fh
        return self

    def __exit__(self, *exc):
        import fcntl
        import threading
        fh = self._held.pop(threading.get_ident())
        (self.dir / "last-download").write_text(str(time.time()))
        fcntl.flock(fh, fcntl.LOCK_UN); fh.close()
        return False


def parse_vtt(text: str) -> list[dict]:
    """WebVTT cues -> [{start, end, text}] in seconds; tags and duplicate rolling lines removed."""
    def seconds(stamp):
        parts = stamp.replace(",", ".").split(":")
        parts = [float(p) for p in parts]
        while len(parts) < 3:
            parts.insert(0, 0.)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    cues, block = [], []
    for line in text.splitlines() + [""]:
        if line.strip():
            block.append(line.strip()); continue
        timing = next((i for i, l in enumerate(block) if "-->" in l), None)
        if timing is not None:
            a, b = block[timing].split("-->")[:2]
            words = " ".join(re.sub(r"<[^>]+>", "", l) for l in block[timing + 1:]).strip()
            words = re.sub(r"\s+", " ", words)
            if words:
                cues.append(dict(start=seconds(a.strip()), end=seconds(b.strip().split()[0]), text=words))
        block = []
    out = []
    for c in cues:                                          # auto captions repeat the previous line while rolling
        if out and c["text"] == out[-1]["text"]:
            out[-1]["end"] = max(out[-1]["end"], c["end"]); continue
        out.append(c)
    return out


def slice_clip(clip, start: float, stop: float):
    """The frames of ``clip`` (``video_landmarks.ClipLandmarks``) with times in [start, stop)."""
    keep = (clip.times >= start) & (clip.times < stop)
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return None
    a, b = int(idx[0]), int(idx[-1]) + 1
    changes = {}
    for f in fields(clip):
        v = getattr(clip, f.name)
        if isinstance(v, np.ndarray) and v.shape[:1] == clip.times.shape[:1]:
            changes[f.name] = v[a:b]
    return replace(clip, **changes)


def hand_coverage(state) -> float:
    return max(float(state.channels[h].valid.any(axis=1).mean()) for h in ("left_hand", "right_hand"))


OTHER_PERSON_SCALE = .5   # a second person counts when their shoulders are at least half the signer's width


def other_person_fraction(clip) -> float:
    """Fraction of frames in which a second person of at least ``OTHER_PERSON_SCALE`` of the signer's size is seen."""
    return float((clip.other_person >= OTHER_PERSON_SCALE).mean()) if len(clip.other_person) else 0.


def face_stats(state) -> tuple[float, float | None]:
    """(fraction of frames with the face tracked, median |head yaw| in degrees) - for the frontal-view filter."""
    head = state.channels["head_rotation"]
    ok = head.valid[:, 0].astype(bool)
    if not ok.any():
        return 0., None
    from signtranslator.pose.rotations import rotation_6d_to_matrix
    import torch
    R = rotation_6d_to_matrix(torch.as_tensor(head.values[ok, 0], dtype=torch.float64)).numpy()
    forward = R[:, :, 2]                                    # the face's +z axis in the camera frame
    yaw = np.degrees(np.arctan2(forward[:, 0], forward[:, 2]))
    return float(ok.mean()), float(np.median(np.abs(yaw)))


def download(video_id: str, directory: Path, *, max_height: int, allow_auto: bool) -> dict:
    """(video path, captions text, caption kind, metadata) or a status for an unavailable video."""
    import yt_dlp
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = dict(quiet=True, no_warnings=True, outtmpl=str(directory / "%(id)s.%(ext)s"),
                format=f"bv*[height<={max_height}][ext=mp4]/bv*[height<={max_height}]/b[height<={max_height}]",
                writesubtitles=True, writeautomaticsub=allow_auto, subtitleslangs=list(CAPTION_LANGS),
                subtitlesformat="vtt", noplaylist=True, retries=3)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as error:  # noqa: BLE001 - yt-dlp raises DownloadError for every kind of refusal
        message = str(error).lower()
        if any(p in message for p in BLOCK_PATTERNS):
            raise Blocked(str(error)[:300]) from None
        status = ("private" if "private" in message else "removed" if any(w in message for w in
                  ("unavailable", "removed", "terminated", "does not exist")) else "error")
        return dict(status=status, reason=str(error)[:300])
    manual = {k: v for k, v in (info.get("subtitles") or {}).items() if is_caption_lang(k)}
    kind = "manual" if manual else ("auto" if allow_auto and any(k.startswith("en") for k in
                                                                    (info.get("automatic_captions") or {})) else None)
    captions = next(iter(sorted(directory.glob(f"{video_id}*.vtt"))), None)
    videos = [p for p in directory.iterdir() if p.suffix in (".mp4", ".webm", ".mkv") and p.stem.startswith(video_id)]
    meta = dict(caption_languages=sorted(manual), channel_id=info.get("channel_id"), channel=info.get("channel"),
                upload_date=info.get("upload_date"),
                duration=info.get("duration"), license=info.get("license"), title=info.get("title"),
                height=info.get("height"), fps=info.get("fps"))
    if not videos:
        return dict(status="error", reason="no video file", meta=meta)
    if captions is None or kind is None:
        return dict(status="no_captions", meta=meta)
    return dict(status="downloaded", video=videos[0], captions=captions.read_text(errors="replace"), kind=kind,
                meta=meta)


def process(video_id: str, args, store, models, gate=None) -> dict:
    from signtranslator.data_engineering.video_landmarks import canonical_from_landmarks, extract_clip, inference_method
    with tempfile.TemporaryDirectory() as tmp:
        if gate is None:
            got = download(video_id, Path(tmp), max_height=args.max_height, allow_auto=args.allow_auto_captions)
        else:
            with gate:
                got = download(video_id, Path(tmp), max_height=args.max_height, allow_auto=args.allow_auto_captions)
        if got["status"] != "downloaded":
            return dict(video_id=video_id, **got)
        video_sha = hashlib.sha256(got["video"].read_bytes()).hexdigest()
        all_cues = parse_vtt(got["captions"])
        caption_seconds = round(sum(c["end"] - c["start"] for c in all_cues), 2)
        cues = [c for c in all_cues if args.min_seconds <= c["end"] - c["start"] <= args.max_seconds]
        if not cues:
            return dict(video_id=video_id, status="no_usable_cues", meta=got["meta"], caption_kind=got["kind"],
                        caption_seconds=caption_seconds)
        clip = extract_clip(got["video"], models, roi=True, count_people_every=args.count_people_every)
    method = inference_method()
    segments, refused = [], []
    for k, cue in enumerate(cues):
        part = slice_clip(clip, cue["start"] - args.pad, cue["end"] + args.pad)
        sample = f"{video_id}-{k:04d}"
        if part is None or len(part.times) < 8:
            refused.append(dict(cue=k, reason="too few frames")); continue
        try:
            state, report = canonical_from_landmarks(
                part, sample_id=sample, source_id=args.source_id, source_sha256=video_sha,
                clock_id=f"{args.source_id}:{sample}:pts", inference_method=method)
        except ValueError as error:
            refused.append(dict(cue=k, reason=str(error)[:200])); continue
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.npz"
            digest = state.save(path)
            key = f"{args.prefix}/states/{sample}.npz"
            store.put(key, path.read_bytes())
        segments.append(dict(cue=k, sample_id=sample, start=cue["start"], end=cue["end"], text=cue["text"],
                             state_key=key, state_sha256=digest, seconds=float(state.timestamps[-1] - state.timestamps[0]),
                             hand_coverage=round(hand_coverage(state), 3),
                             face_coverage=round(face_stats(state)[0], 3),
                             median_abs_yaw_deg=(round(face_stats(state)[1], 1) if face_stats(state)[1] is not None else None),
                             available_channels=[n for n, c in state.channels.items() if c.valid.any()],
                             **({} if part.other_person is None else
                                dict(other_person_fraction=round(other_person_fraction(part), 3)))))
    return dict(video_id=video_id, status="converted", meta=got["meta"], caption_kind=got["kind"], video_sha256=video_sha,
                caption_seconds=caption_seconds,
                cues=len(cues), segments=segments, refused=refused, inference_method=method)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--ids", required=True, help="store key (or local path with --ids-local) of a video-id list")
    parser.add_argument("--ids-local", action="store_true")
    parser.add_argument("--prefix", required=True, help="private store prefix of this corpus")
    parser.add_argument("--source-id", default="youtube_asl_video_estimate")
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="at most this many videos of the shard (pilot)")
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--count-people-every", type=int, default=12,
                        help="count people on every n-th frame for the single-signer filter (0: off)")
    parser.add_argument("--allow-auto-captions", action="store_true")
    parser.add_argument("--min-seconds", type=float, default=1.)
    parser.add_argument("--max-seconds", type=float, default=20.)
    parser.add_argument("--pad", type=float, default=.2)
    parser.add_argument("--pause", type=float, default=2., help="seconds between videos (polite rate)")
    parser.add_argument("--polite-dir", help="shared directory of the per-machine download gate (PoliteGate)")
    parser.add_argument("--download-interval", type=float, default=20., help="seconds between downloads per machine")
    parser.add_argument("--blocked-backoff", type=float, default=3600., help="wait after a refusal, seconds")
    parser.add_argument("--max-blocks", type=int, default=3, help="stop after this many refusals")
    parser.add_argument("--retry-status", action="append", default=[],
                        help="reprocess videos whose stored report has this status (e.g. no_captions after a fix)")
    args = parser.parse_args(argv)
    from train_text_to_motion import make_store
    from signtranslator.data_engineering.video_landmarks import probe_runtime, verify_models
    store = make_store(args.store)
    text = Path(args.ids).read_text() if args.ids_local else store.get(args.ids).decode()
    ids = [line.strip() for line in text.splitlines() if re.fullmatch(r"[A-Za-z0-9_-]{11}", line.strip())]
    mine = [v for i, v in enumerate(ids) if i % args.shards == args.shard][:args.limit]
    models = verify_models(Path(args.models_dir))
    probe_runtime(models)
    counts, blocks = {}, 0
    gate = PoliteGate(args.polite_dir, interval=args.download_interval, backoff=args.blocked_backoff) if args.polite_dir else None
    for n, video_id in enumerate(mine):
        report_key = f"{args.prefix}/videos/{video_id}.json"
        if store.exists(report_key):
            if not args.retry_status or json.loads(store.get(report_key)).get("status") not in args.retry_status:
                continue
        started = time.monotonic()
        try:
            report = process(video_id, args, store, models, gate)
        except Blocked as blocked:
            blocks += 1
            print(f"BLOCKED at video {n} ({video_id}) [{blocks}/{args.max_blocks}]: {blocked}", flush=True)
            if gate is None or blocks >= args.max_blocks:
                return 3
            gate.mark_blocked()                             # every worker on this machine waits the refusal out
            continue
        except Exception as error:  # noqa: BLE001 - one bad video must not stop the shard; it is retried next run
            print(json.dumps(dict(video_id=video_id, status="error", reason=f"{type(error).__name__}: {error}"[:300])),
                  flush=True)
            continue
        report["seconds_taken"] = round(time.monotonic() - started, 1)
        if report["status"] != "error":
            store.put(report_key, json.dumps(report, sort_keys=True).encode())
        counts[report["status"]] = counts.get(report["status"], 0) + 1
        print(json.dumps(dict(n=n + 1, of=len(mine), video_id=video_id, status=report["status"],
                              segments=len(report.get("segments", [])), seconds=report["seconds_taken"])), flush=True)
        time.sleep(args.pause)
    print(json.dumps(dict(shard=args.shard, done=counts)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
