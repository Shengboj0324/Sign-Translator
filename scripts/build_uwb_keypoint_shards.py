"""YouTube-ASL Clip Keypoint Dataset (UWB, CC BY 4.0) -> compact training shards for the 2D back-translator.

Plan item 7 fallback: the clips are already acquired in R2 by ``runpod/fleet/acquire_uwb_keypoints.sh``.

**Input.** Each zip member is one caption-aligned clip, ``<video_id>.<first frame>-<last frame>.json``, holding
MediaPipe Holistic 2D per frame. The translations JSON (train / dev) maps clip ids to English.

**Per clip.**
1. Only the pose, hand and arm points of the shared 2D layout (``data.keypoints2d``) are kept, normalised there.
   Clips are stored at their source rate with that rate recorded. The source frame rate is not published, so
   ``--source-fps`` (30) is an assumption; trainers resample from it.
2. A clip is dropped when it:
   - has no translation;
   - leaks an evaluation sentence (How2Sign / ASLLRP val/test; exact text or ≥ 0.8 5-gram Jaccard,
     ``build_youtube_asl_manifest.EvalIndex``);
   - is too short or too long;
   - has hands in too few frames, or shoulders in too few frames.
3. UWB "train" clips are partition ``train`` and UWB "dev" clips ``monitor``. YouTube data is never an evaluation
   set: How2Sign val/test remain the judges.

**Output.**
- Per zip: shards of ``--shard-size`` clips. Each is an npz holding ``points`` (frames, 48, 2) float16, ``valid``
  (frames, 48), ``offsets`` (clips + 1), and a JSON sidecar of clip id, video id, text, partition and frames.
- A per-zip ``part.json`` with SHA-256 of every shard and drop counts, so a re-run skips finished zips.
- ``index.json`` over all parts at the end.

Zips are read by ranged GETs in member order (sequential), so nothing is staged on disk. Workers run one zip each.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
from multiprocessing import Pool
import os
from pathlib import Path
import re
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.keypoints2d import from_holistic, normalise  # noqa: E402

LICENCE = ("CC BY 4.0 (YouTube-ASL Clip Keypoint Dataset, Zelezny, Hruz, Straka, Gueuwou; UWB / JSALT 2024; "
           "LINDAT 11234/1-5898); derived from YouTube videos owned by their creators: research use, private store, "
           "no redistribution")
_CLIP = re.compile(r"^(?P<video>.+)\.(?P<a>\d+)-(?P<b>\d+)$")
_WORD = re.compile(r"[A-Za-z0-9']+")


class RangeReader(io.RawIOBase):
    """A seekable read-only view of one R2 object through ranged GETs."""

    def __init__(self, store, key: str) -> None:
        self.store, self.key, self.pos = store, key, 0
        self.size = store.s3.head_object(Bucket=store.bucket, Key=key)["ContentLength"]

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        data = self.store._call(lambda: self.store.s3.get_object(
            Bucket=self.store.bucket, Key=self.key, Range=f"bytes={self.pos}-{end}"))["Body"].read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_zip(store, key: str) -> zipfile.ZipFile:
    if hasattr(store, "s3"):
        return zipfile.ZipFile(io.BufferedReader(RangeReader(store, key), buffer_size=16 << 20))
    return zipfile.ZipFile(store.root / key)


def translations(store, prefix: str) -> dict[str, tuple[str, str]]:
    """clip id -> (text, partition) from the UWB train and dev translation files."""
    out = {}
    for name, partition in (("YT.translations.train.json", "train"), ("YT.translations.dev.json", "monitor")):
        for video, entry in json.loads(store.get(f"{prefix}/{name}")).items():
            for clip in entry.get("clip_order", []):
                text = (entry.get(clip) or {}).get("translation")
                if text:
                    out[clip] = (" ".join(text.split()), partition)
    return out


def clip_quality(points, valid, *, args) -> str | None:
    frames = len(points)
    if frames < args.min_frames:
        return "too_short"
    if frames / args.source_fps > args.max_seconds:
        return "too_long"
    if (valid[:, 0] & valid[:, 1]).mean() < args.min_shoulders:
        return "shoulders_not_visible"
    if (valid[:, 6:].any(axis=1)).mean() < args.min_hands:
        return "hands_not_visible"
    return None


def _write_shard(store, key: str, clips: list) -> dict:
    points = np.concatenate([c["points"] for c in clips]).astype(np.float16)
    valid = np.concatenate([c["valid"] for c in clips])
    offsets = np.cumsum([0] + [len(c["points"]) for c in clips])
    buffer = io.BytesIO()
    np.savez_compressed(buffer, points=points, valid=valid, offsets=offsets)
    payload = buffer.getvalue()
    meta = [{k: c[k] for k in ("clip_id", "video_id", "text", "partition", "frames")} for c in clips]
    meta_payload = json.dumps(meta, sort_keys=True).encode()
    store.put(key, payload)
    store.put(key[:-4] + ".json", meta_payload)
    return dict(key=key, sha256=hashlib.sha256(payload).hexdigest(), meta_sha256=hashlib.sha256(meta_payload).hexdigest(),
                clips=len(clips), frames=int(offsets[-1]),
                partitions=dict(collections.Counter(c["partition"] for c in clips)))


def convert_zip(job) -> dict:
    args, zip_name, texts, eval_sentences = job
    from build_youtube_asl_manifest import EvalIndex
    from train_text_to_motion import make_store
    store = make_store(args.store)
    stem = zip_name.rsplit(".", 1)[0]
    part_key = f"{args.out_prefix}/{stem}/part.json"
    if store.exists(part_key):
        return json.loads(store.get(part_key))
    index = EvalIndex(eval_sentences) if eval_sentences else None
    why, shards, pending, seen = collections.Counter(), [], [], set()
    with open_zip(store, f"{args.prefix}/{zip_name}") as z:
        members = sorted((i for i in z.infolist() if i.filename.endswith(".json")), key=lambda i: i.header_offset)
        for info in members[:args.limit or None]:
            clip_id = Path(info.filename).name[:-5]
            m = _CLIP.match(clip_id)
            if m is None:
                why["bad_name"] += 1; continue
            if clip_id in seen:
                why["duplicate"] += 1; continue
            seen.add(clip_id)
            if clip_id not in texts:
                why["no_translation"] += 1; continue
            text, partition = texts[clip_id]
            if len(_WORD.findall(text)) < args.min_words:
                why["few_words"] += 1; continue
            if index is not None and index.leaks(text):
                why["evaluation_leak"] += 1; continue
            try:
                frames = json.loads(z.read(info))["keypoints"]
                points, valid = from_holistic(frames)
            except (ValueError, KeyError, TypeError) as error:
                why["unreadable"] += 1
                print(f"{clip_id}: {str(error)[:120]}", flush=True); continue
            reason = clip_quality(points, valid, args=args)
            if reason:
                why[reason] += 1; continue
            pending.append(dict(clip_id=clip_id, video_id=m["video"], text=text, partition=partition,
                                frames=len(points), points=normalise(points, valid, y_down=True), valid=valid))
            why["kept"] += 1
            if len(pending) == args.shard_size:
                shards.append(_write_shard(store, f"{args.out_prefix}/{stem}/shard-{len(shards):05d}.npz", pending))
                pending = []
        if pending:
            shards.append(_write_shard(store, f"{args.out_prefix}/{stem}/shard-{len(shards):05d}.npz", pending))
    part = dict(zip=zip_name, shards=shards, counts=dict(why), members=len(members))
    store.put(part_key, json.dumps(part, indent=1, sort_keys=True).encode())
    print(json.dumps(dict(zip=zip_name, **dict(why))), flush=True)
    return part


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--prefix", default="acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1")
    parser.add_argument("--out-prefix", default="acquisition/2026-10-17-v1/derived/youtube-asl-uwb-keypoints-v1/shards-v1")
    parser.add_argument("--zip", action="append", default=[], help="zip names (default: every raw_keypoints_*.zip)")
    parser.add_argument("--eval-manifest", action="append", default=[], help="KEY@SHA whose val/test texts must not train")
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--source-fps", type=float, default=30., help="assumed: the dataset does not publish it")
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=20.)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--min-hands", type=float, default=.5, help="share of frames with at least one hand")
    parser.add_argument("--min-shoulders", type=float, default=.8)
    parser.add_argument("--limit", type=int, default=0, help="clips per zip (testing)")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    zips = args.zip or sorted(Path(k).name for k in store.list(args.prefix + "/")
                              if re.search(r"raw_keypoints_\d+\.zip$", k))
    if not zips:
        raise SystemExit("no keypoint zips under the prefix")
    texts = translations(store, args.prefix)
    sentences = []
    for spec in args.eval_manifest:
        from signtranslator.data_engineering.corpus_manifest import load_manifest
        key, _, sha = spec.rpartition("@")
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "m.json"; p.write_bytes(store.get(key) + b"\n")
            m = load_manifest(p, expected_sha256=sha)
        sentences += [r.get("english") or r.get("text") for r in m["records"]
                      if r.get("partition") in ("val", "test", "heldout_official") and (r.get("english") or r.get("text"))]
    jobs = [(args, z, texts, sentences) for z in zips]
    with Pool(min(args.workers, len(jobs))) as pool:
        parts = pool.map(convert_zip, jobs, chunksize=1)
    counts = collections.Counter()
    for p in parts:
        counts.update(p["counts"])
    index = dict(corpus_id="youtube-asl-uwb-keypoints-v1", licence=LICENCE, source_fps_assumed=args.source_fps,
                 layout="signtranslator.data.keypoints2d (48 points, hands local)", parameters=vars(args) | {"zip": zips},
                 eval_sentences=len(sentences), counts=dict(counts), parts=parts)
    store.put(f"{args.out_prefix}/index.json", json.dumps(index, indent=1, sort_keys=True, default=str).encode())
    print(json.dumps(dict(counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
