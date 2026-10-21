"""Recover How2Sign take-5 sessions: lateral keypoints re-cut from the raw recordings (audit B80).

In take-5 sessions the released lateral clips are mis-cut and cannot be synchronised,
so ``triangulate_how2sign_sessions.py`` refuses them. The raw 30 fps recordings of
both cameras are intact (``acquisition/acquire_how2sign_raw.py`` puts the mirror's
shards in R2). Per take-5 session this job

1. places every released frontal clip in the raw frontal recording, frame by frame;
2. runs Sapiens-1B on the raw lateral recording (only the frames the search can touch);
3. finds the sub-frame camera offset by the epipolar scan over the session's clips,
   refusing the session if the minimum is not sharp and clearly below lag 0;
4. writes lateral keypoints on each frontal clip's frames, linearly interpolated at
   the fractional offset, in the layout of ``sapiens_keypoints_runpod.py``.

Outputs under ``--prefix`` (write-once; a session with a report is skipped):
  raw-keypoints/<video>-<take>-rgb_side.npz   Sapiens on the raw lateral recording
  keypoints/<clip>-rgb_side.npz               re-cut lateral keypoints (accepted sessions)
  sessions/<video>-<take>.json                provenance: hashes, placements, offset, curves
  reports/shard-XXX-of-NNN.json

Every input is pinned: zips and CSVs by their verified acquisition SHA-256, raw
recordings and frontal clips by their own SHA-256 (the frontal clip must be the one
the frontal keypoints were computed from), and the Sapiens checkpoint by hash.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store  # noqa: E402
from signtranslator.data_engineering import how2sign_recut as recut  # noqa: E402
from signtranslator.data_engineering.two_view_states import SYNC_SHARPNESS, ClipKeypoints  # noqa: E402

CLIP = re.compile(r"(?P<video>.+)_(?P<sentence>\d+)-(?P<take>\d+)-rgb_front")


def verified_sha(store, key: str, expected: str | None = None) -> str:
    """SHA-256 of an acquired object from its provenance, which must be verified by an R2 re-read."""
    provenance = json.loads(store.get(key + ".provenance.json"))
    if not provenance.get("verified_by_r2_reread") or (expected and provenance.get("sha256") != expected):
        raise SystemExit(f"{key} is not the verified acquisition" + (f" {expected}" if expected else ""))
    return provenance["sha256"]


def open_object(store, key: str):
    """Seekable binary reader of a stored object (ranged GETs on R2)."""
    if isinstance(store, DirectoryStore):
        return (store.root / key).open("rb")
    from extract_video_states import StoreRangeFile
    return io.BufferedReader(StoreRangeFile(store.s3, store.bucket, key), 1 << 20)


class Archives:
    """Members of several verified zips by file name, opened lazily."""

    def __init__(self, store, keys: dict):
        self.store, self.keys, self.members, self.open = store, keys, {}, {}
        for key in keys:
            with zipfile.ZipFile(open_object(store, key)) as archive:
                for name in archive.namelist():
                    if name.endswith(".mp4"):
                        self.members[PurePosixPath(name).name] = (key, name)

    def read(self, file_name: str) -> tuple[bytes, dict]:
        key, member = self.members[file_name]
        if key not in self.open:
            self.open[key] = zipfile.ZipFile(open_object(self.store, key))
        payload = self.open[key].read(member)
        return payload, dict(zip_key=key, zip_sha256=self.keys[key], member=member,
                             member_sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))


def decoded(payload: bytes, *, thumbnails: bool):
    """Frame times and (thumbnails ? grey thumbnails : frame count) of an MP4 held in memory."""
    import av
    times, frames = [], []
    with av.open(io.BytesIO(payload)) as container:
        for frame in container.decode(container.streams.video[0]):
            times.append(float(frame.pts * frame.time_base))
            if thumbnails:
                width, height = recut.THUMBNAIL
                frames.append(frame.reformat(width=width, height=height, format="gray",
                                             interpolation="AREA").to_ndarray().astype(np.float32))
    return np.array(times), (np.stack(frames) if frames else None)


def check_rate(times: np.ndarray, what: str) -> None:
    """Raw recordings must be constant 30 fps: frame index is time."""
    steps = np.diff(times)
    if len(times) < 2 or np.max(np.abs(steps * recut.RAW_FPS - 1)) > .25:
        raise ValueError(f"the raw {what} recording is not constant {recut.RAW_FPS:g} fps")


def sapiens_raw(payload: bytes, first: int, stop: int, model, device, batch: int, flip: bool):
    """Sapiens on frames [first, stop) of a recording held in memory; other frames confidence 0."""
    import av
    from sapiens_keypoints_runpod import frames_keypoints
    times = []
    with av.open(io.BytesIO(payload)) as container:
        def frames():
            for index, frame in enumerate(container.decode(container.streams.video[0])):
                times.append(float(frame.pts * frame.time_base))
                if first <= index < stop:
                    yield frame.to_ndarray(format="rgb24")
        points, confidence = frames_keypoints(frames(), model, device, batch, flip)
    all_points = np.zeros((len(times), 308, 2), np.float32)
    all_confidence = np.zeros((len(times), 308), np.float16)
    stop = first + len(points)
    all_points[first:stop], all_confidence[first:stop] = points, confidence
    return np.array(times), all_points, all_confidence, stop


def load_keypoints(store, key: str, name: str):
    payload = store.get(key)
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        clip = ClipKeypoints(name, data["times"], data["points"].astype(np.float64),
                             data["confidence"].astype(np.float64))
        clip_sha = str(data["clip_sha256"]) if "clip_sha256" in data else None
    return clip, clip_sha, hashlib.sha256(payload).hexdigest()


def npz(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


class Sapiens:
    """The pinned checkpoint, loaded on first use (sessions with cached raw keypoints need none)."""

    def __init__(self, path, sha256, batch, flip):
        self.path, self.sha256, self.batch, self.flip, self.model = path, sha256, batch, flip, None

    def __call__(self, payload, first, stop):
        if self.model is None:
            import torch
            from sapiens_keypoints_runpod import load_model
            self.device = torch.device("cuda" if torch.cuda.is_available() else
                                       "mps" if torch.backends.mps.is_available() else "cpu")
            self.model, self.sha256 = load_model(Path(self.path), self.sha256, self.device)
        return sapiens_raw(payload, first, stop, self.model, self.device, self.batch, self.flip)


def recover_session(store, args, session: str, clips: dict, rows: dict, fronts: Archives, raws: Archives,
                    sapiens: Sapiens) -> dict:
    """Recover one session; returns its provenance record (written by the caller)."""
    prefix = args.prefix.rstrip("/")
    record = dict(schema_version=1, session=session, accepted=False, reason=None,
                  thresholds=dict(placement_window=recut.PLACEMENT_WINDOW, placement_sharpness=recut.PLACEMENT_SHARPNESS,
                                  match_ratio=recut.MATCH_RATIO, max_offset_frames=args.max_offset,
                                  sync_sharpness=SYNC_SHARPNESS, far_lag=recut.FAR_LAG,
                                  max_sampson_px=recut.MAX_SAMPSON_PX, fine_step=recut.FINE_STEP,
                                  front_frames_per_unit=recut.FRONT_FRAMES_PER_UNIT))
    front_raw, front_info = raws.read(f"{session}-rgb_front.mp4")
    front_times, front_thumbs = decoded(front_raw, thumbnails=True)
    check_rate(front_times, "frontal")
    record["raw_front"] = dict(front_info, frames=len(front_times))
    keypoints, thumbs, nominals, info = {}, {}, {}, {}
    for name, key in sorted(clips.items()):
        clip, clip_sha, keypoints_sha = load_keypoints(store, key, name)
        payload, member = fronts.read(f"{name}.mp4")
        if clip_sha is not None and member["member_sha256"] != clip_sha:
            raise SystemExit(f"{name}: the released clip differs from the one the frontal keypoints came from")
        times, small = decoded(payload, thumbnails=True)
        if len(times) != len(clip.times):
            raise SystemExit(f"{name}: {len(times)} frames in the clip, {len(clip.times)} in its keypoints")
        keypoints[name], thumbs[name], nominals[name] = clip, small, recut.nominal_start(rows[name])
        info[name] = dict(clip_zip=member["zip_key"], clip_zip_sha256=member["zip_sha256"],
                          clip_sha256=member["member_sha256"], keypoints_key=key, keypoints_sha256=keypoints_sha,
                          csv_start=rows[name], frames=len(times))
    placements = recut.place_clips(front_thumbs, thumbs, nominals)
    for name, placement in placements.items():
        info[name].update(placement=placement)
    record["front_clips"] = info
    placed = {n: p["start"] for n, p in placements.items() if p["status"] in ("placed", "still")}
    if not placed:
        return dict(record, reason="no frontal clip could be placed in the raw frontal recording")
    side_raw, side_info = raws.read(f"{session}-rgb_side.mp4")
    record["raw_side"] = side_info
    cache = f"{prefix}/raw-keypoints/{session}-rgb_side.npz"
    first, stop = recut.needed_frames([(s, len(keypoints[n].times)) for n, s in placed.items()],
                                      len(front_times) + args.max_offset, max_offset=args.max_offset)
    cached, cache_sha = None, None
    if store.exists(cache):
        payload = store.get(cache)
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            if (str(data["member_sha256"]) == side_info["member_sha256"] and int(data["first"]) <= first
                    and int(data["stop"]) >= min(stop, len(data["times"]))):
                cached, cache_sha = {k: data[k] for k in data.files}, hashlib.sha256(payload).hexdigest()
    if cached is None:
        side_times, points, confidence, computed_stop = sapiens(side_raw, first, stop)
        cached = dict(times=side_times, points=points, confidence=confidence, first=np.array(first),
                      stop=np.array(computed_stop), member_sha256=np.array(side_info["member_sha256"]),
                      model_sha256=np.array(sapiens.sha256), flip_tta=np.array(sapiens.flip))
        cache_sha = store.put(cache, npz(**cached))
    check_rate(cached["times"], "lateral")
    record["raw_side"].update(frames=len(cached["times"]))
    record["sapiens"] = dict(raw_keypoints_key=cache, raw_keypoints_sha256=cache_sha,
                             model_sha256=str(cached["model_sha256"]), flip_tta=bool(cached["flip_tta"]),
                             frames=[int(cached["first"]), int(cached["stop"])])
    side_points, side_confidence = cached["points"].astype(np.float64), cached["confidence"].astype(np.float64)
    names = sorted(placed)
    offset = recut.estimate_offset([(placed[n], keypoints[n]) for n in names], side_points, side_confidence,
                                   max_offset=args.max_offset)
    for name, clip in zip(names, offset.pop("clips")):
        info[name].update(offset_sampson_px=clip["sampson_px"], offset_observations=clip["observations"])
    record["offset"] = dict(offset, offset_seconds=None if offset["offset_frames"] is None
                            else offset["offset_frames"] / recut.RAW_FPS)
    if not offset["passed"]:
        return dict(record, reason=offset["reason"])
    outputs = {}
    for name in names:
        side_name = name[:-len("-rgb_front")] + "-rgb_side"
        side = recut.recut(keypoints[name], placed[name], offset["offset_frames"], side_points, side_confidence, side_name)
        payload = npz(times=side.times, points=side.points.astype(np.float32),
                      confidence=side.confidence.astype(np.float16), recut_offset_frames=np.array(offset["offset_frames"]),
                      raw_start=np.array(placed[name]), raw_side_sha256=np.array(side_info["member_sha256"]),
                      raw_front_sha256=np.array(front_info["member_sha256"]),
                      front_clip_sha256=np.array(info[name]["clip_sha256"]),
                      model_sha256=np.array(str(cached["model_sha256"])), flip_tta=np.array(bool(cached["flip_tta"])))
        key = f"{prefix}/keypoints/{side_name}.npz"
        outputs[side_name] = dict(key=key, sha256=store.put(key, payload))
    return dict(record, accepted=True, outputs=outputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2", help="r2, or dir:<path> for a local mirror")
    parser.add_argument("--front-keypoints", required=True, help="prefix holding keypoints/<clip>.npz (frontal)")
    parser.add_argument("--front-zip", action="append", required=True, metavar="KEY=SHA256",
                        help="released frontal clip zip, pinned (repeat per split)")
    parser.add_argument("--sentences", action="append", required=True,
                        help="original how2sign_<split>.csv key (START in the units of the cut; repeat per split)")
    parser.add_argument("--raw-prefix", required=True, help="prefix of the acquired raw-recording zips (both cameras)")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--checkpoint", help="local Sapiens TorchScript file (needed unless raw keypoints are cached)")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--no-flip", dest="flip", action="store_false", help="disable flip test-time augmentation")
    parser.add_argument("--take", default="5", help="recording take to recover (the mis-cut one)")
    parser.add_argument("--max-offset", type=int, default=recut.MAX_OFFSET_FRAMES,
                        help="camera offsets searched, in raw frames either way")
    parser.add_argument("--match", help="only sessions whose name matches this regular expression")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    zips = dict(item.split("=", 1) for item in args.front_zip)
    fronts = Archives(store, {key: verified_sha(store, key, sha) for key, sha in zips.items()})
    rows, csvs = {}, {}
    for key in args.sentences:
        csvs[key] = verified_sha(store, key)
        for row in csv.DictReader(io.StringIO(store.get(key).decode()), delimiter="\t"):
            rows[row["SENTENCE_NAME"]] = float(row["START"])
    # A zip without its provenance record is still being acquired (the record is written after the
    # re-read verification): it is not used, and its sessions are reported as not acquired.
    listed = set(store.list(args.raw_prefix.rstrip("/") + "/"))
    raw_keys = [k for k in sorted(listed) if k.endswith(".zip") and k + ".provenance.json" in listed]
    raws = Archives(store, {key: verified_sha(store, key) for key in raw_keys})
    sessions = collections.defaultdict(dict)
    for key in store.list(args.front_keypoints.rstrip("/") + "/keypoints/"):
        match = CLIP.fullmatch(Path(key).stem)
        if match and match["take"] == args.take and Path(key).stem in rows:
            sessions[f"{match['video']}-{match['take']}"][Path(key).stem] = key
    names = sorted(n for n in sessions if not args.match or re.search(args.match, n))
    names = [n for i, n in enumerate(names) if i % args.shards == args.shard]
    sapiens = Sapiens(args.checkpoint, args.checkpoint_sha256, args.batch, args.flip)
    prefix = args.prefix.rstrip("/")
    summary = []
    for session in names:
        key = f"{prefix}/sessions/{session}.json"
        if store.exists(key):
            record = json.loads(store.get(key))
        elif not all(f"{session}-rgb_{view}.mp4" in raws.members for view in ("front", "side")):
            summary.append(dict(session=session, accepted=False, written=False,
                                reason="raw recordings not acquired (scripts/acquisition/acquire_how2sign_raw.py)"))
            continue
        else:
            if sapiens.path is None and not store.exists(f"{prefix}/raw-keypoints/{session}-rgb_side.npz"):
                raise SystemExit("--checkpoint is required to run Sapiens on the raw lateral recordings")
            record = recover_session(store, args, session, sessions[session], rows, fronts, raws, sapiens)
            record.update(sentences=csvs, front_keypoints=args.front_keypoints)
            store.put(key, json.dumps(record, sort_keys=True, indent=1, default=float).encode())
        summary.append(dict(session=session, accepted=record["accepted"], reason=record["reason"], written=True,
                            offset_frames=(record.get("offset") or {}).get("offset_frames"),
                            sampson_px=(record.get("offset") or {}).get("sampson_px"),
                            clips=len(record.get("outputs", {}))))
        print(json.dumps(summary[-1]), flush=True)
    report = dict(schema_version=1, shard=args.shard, shards=args.shards, take=args.take, sessions=summary,
                  accepted=sum(s["accepted"] for s in summary))
    store.put(f"{prefix}/reports/shard-{args.shard:03d}-of-{args.shards:03d}.json",
              json.dumps(report, sort_keys=True, default=float).encode())
    print(json.dumps(dict(sessions=len(summary), accepted=report["accepted"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
