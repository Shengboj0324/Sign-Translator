"""Training pairs for UWB lifter v2: UWB 2D keypoints aligned with our own Tasks states (CPU; reads only, writes local
shards).

Declared in docs/NEXT_PHASE_PLAN.md, "UWB lifter v2 and gate v2: declared" (2026-10-30, 11:12 UTC). Population: the
**train** and **lifter-val** videos of the SHA-pinned channel-disjoint split (``build_uwb_lifter_split.py``); a held-out
video id is refused outright (before anything is read, and again in every worker), and the split's parts must be
disjoint by video and channel. Every candidate pair of every such video is built with the gate's own code
(``uwb_lifter_gate``: ``estimate_fps``, ``pair_clips``, ``uwb_clips_of``, ``members_index``, ``state_items``,
``image_arm_directions``, ``choose_lag`` with the gate's max lag 6 and margin 0.5, ``frame_map``). A pair is kept only if

- the streams match: the estimated rate is within 0.2% of our download's (``meta.fps``; the gate's stream stratum);
- the alignment is ``arm-2d``;
- the chosen lag and the best (unmargined) lag are both in {-1, 0, 1};
- it has >= 400 bone-frames at the chosen lag (``choose_lag``'s ``frames``);
- the lag-0 score is <= 20 degrees.

**Per kept pair**, on the aligned frames (our index ``i``, UWB index ``j`` at the chosen lag) where UWB holds a pose:

- inputs: UWB pose (33, 2) and hands (2, 21, 2), normalised by the stored frame size (``x / W``, ``y / H``; what the
  xy-v2 features read, with the pair's aspect), with detection flags;
- targets (MediaPipe axes: ``video_landmarks.Y_UP`` applied to our y-up forward kinematics): the four arm bone
  directions (left upper arm, left forearm, right upper arm, right forearm) and the shoulder line (the chest's x axis,
  right -> left), each with its validity in our state; per side the 15 finger bone directions in the camera frame (the
  hand's world frame times the palm-frame direction) with validity (only for the hand fine-tuning comparison);
- validity labels for points 11..16 (shoulders, elbows, wrists) on frames where our chest is valid (both hips: our arms
  exist there; elsewhere every label is masked): elbow = our ``<Side>UpperArm`` valid; wrist = our ``<Side>Forearm``
  valid, labelled only where the upper arm is valid; shoulders are not labelled here (the How2Sign crops label them);
  and our forearm validity per side on every aligned frame (``forearm_valid``: the gate's forearm row reads the same
  flag where our chest is valid; the head's thresholds are set on it).

Shards (``shard-NNNNN.npz``, ``--shard-pairs`` pairs each, in sorted (video, clip) order, no pickles) concatenate the
frames with per-pair offsets; ``pairs-index.json`` records the split SHA-256, the settings, every refusal reason with
counts, and each shard's SHA-256, pairs and frames by part. Workers come from ``os.sched_getaffinity`` and the cgroup
memory limit (``uwb_downstream_check.worker_count``), never ``nproc``.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

SCHEMA = "uwb-lifter-pairs-v1"
ARM_TARGETS = ("LeftUpperArm", "LeftForearm", "RightUpperArm", "RightForearm", "ShoulderLine")
LABEL_POINTS = (11, 12, 13, 14, 15, 16)
STREAM_TOLERANCE = .002
FILTERS = dict(lags=(-1, 0, 1), min_bone_frames=400, max_lag0_score=20., max_lag=6, lag_margin=.5)
MP_AXES = np.array([1., -1., -1.])          # y-up world -> MediaPipe axes (video_landmarks.Y_UP is diagonal)
_WORKER: dict = {}


class HeldOutRefused(RuntimeError):
    """A held-out video reached the pairs builder."""


def stream_match(fps: float | None, meta_fps) -> bool:
    return bool(fps and meta_fps and abs(fps / meta_fps - 1) < STREAM_TOLERANCE)


def refuse_held_out(video_ids, held_out: set) -> None:
    bad = sorted(set(video_ids) & set(held_out))
    if bad:
        raise HeldOutRefused(f"{len(bad)} held-out videos refused, e.g. {bad[:3]}")


def pair_arrays(ours_state, clip, fps: float, meta_fps, filters: dict = FILTERS) -> dict:
    """The kept pair's arrays (``status`` "kept") or ``status`` "refused" with ``reason``."""
    import uwb_lifter_gate as gate
    from validate_semlex_adapter import world_frames
    from signtranslator.data_engineering.uwb_keypoints import _found
    from signtranslator.data_engineering.video_landmarks import finger_joints
    if not stream_match(fps, meta_fps):
        return dict(status="refused", reason="stream mismatch")
    times = ours_state.timestamps
    ours = gate.state_items(ours_state)
    found = _found(clip.pose)
    a = gate.choose_lag(times, clip.first, clip.frames, fps, ours, gate.image_arm_directions(clip.pose, found),
                        filters["max_lag"], margin=filters["lag_margin"])
    if a["method"] != "arm-2d":
        return dict(status="refused", reason="not arm-aligned")
    if a["lag"] not in filters["lags"] or a["best_lag"] not in filters["lags"]:
        return dict(status="refused", reason="lag outside {-1, 0, 1}", alignment=a)
    if a["frames"] < filters["min_bone_frames"]:
        return dict(status="refused", reason="fewer than 400 bone-frames", alignment=a)
    if a["lag0_score"] is None or a["lag0_score"] > filters["max_lag0_score"]:
        return dict(status="refused", reason="lag-0 score over 20 deg", alignment=a)
    i, j = gate.frame_map(times, clip.first, clip.frames, fps, a["lag"])
    keep = found[j]
    i, j = i[keep], j[keep]
    if not len(i):
        return dict(status="refused", reason="no aligned pose frame", alignment=a)
    scale = np.array([clip.width, clip.height], dtype=np.float64)
    hands_found = np.stack([_found(clip.hands[j, k]) for k in range(2)], axis=1)
    hands = np.where(hands_found[:, :, None, None], np.nan_to_num(clip.hands[j]) / scale, 0.)
    world = world_frames(ours_state)
    body = ours_state.channels["body"]
    dirs = np.zeros((len(i), len(ARM_TARGETS), 3))
    dirs_ok = np.zeros((len(i), len(ARM_TARGETS)), bool)
    for k, label in enumerate(ARM_TARGETS[:4]):
        d, ok = ours[label]
        dirs[:, k], dirs_ok[:, k] = d[i] * MP_AXES, ok[i]
    chest_w, chest_ok = world["Chest"]
    dirs[:, 4], dirs_ok[:, 4] = (chest_w[i] @ np.array([1., 0., 0.])) * MP_AXES, chest_ok[i]
    chest = body.valid[i, body.labels.index("Chest")].astype(bool)
    labels = np.zeros((len(i), 6), np.uint8)
    mask = np.zeros((len(i), 6), bool)
    forearm = np.zeros((len(i), 2), bool)
    for k, (side, (elbow, wrist)) in enumerate((("Left", (2, 4)), ("Right", (3, 5)))):
        upper = body.valid[i, body.labels.index(f"{side}UpperArm")].astype(bool)
        fore = body.valid[i, body.labels.index(f"{side}Forearm")].astype(bool)
        labels[:, elbow], mask[:, elbow] = upper, chest
        labels[:, wrist], mask[:, wrist] = fore, chest & upper
        forearm[:, k] = fore
    fingers = np.zeros((len(i), 2, 15, 3))
    fingers_ok = np.zeros((len(i), 2, 15), bool)
    for s, side in enumerate(("Left", "Right")):
        hand_w, hand_ok = world[f"{side}Hand"]
        for k, joint in enumerate(finger_joints(side)):
            d, ok = ours[joint.name]
            fingers[:, s, k] = np.einsum("nij,nj->ni", hand_w[i], d[i]) * MP_AXES
            fingers_ok[:, s, k] = hand_ok[i] & ok[i]
    return dict(status="kept", alignment=a, our_index=i.astype(np.int32), uwb_index=j.astype(np.int32),
                pose=(clip.pose[j] / scale).astype(np.float32), hands=hands.astype(np.float32),
                hands_found=hands_found, arm_dirs=dirs.astype(np.float16), arm_valid=dirs_ok, chest_valid=chest,
                labels=labels, label_mask=mask, forearm_valid=forearm, finger_dirs=fingers.astype(np.float16), finger_valid=fingers_ok,
                aspect=clip.aspect)


FRAME_KEYS = ("our_index", "uwb_index", "pose", "hands", "hands_found", "arm_dirs", "arm_valid", "chest_valid",
              "labels", "label_mask", "forearm_valid", "finger_dirs", "finger_valid")


def write_shard(path: Path, rows: list[dict]) -> dict:
    """One shard of kept pairs (each row: job fields + arrays); returns its index entry."""
    sizes = [len(r["pose"]) for r in rows]
    arrays = {k: np.concatenate([r[k] for r in rows]) for k in FRAME_KEYS}
    meta = dict(pair_video=np.array([r["video_id"] for r in rows]), pair_clip=np.array([r["clip_id"] for r in rows]),
                pair_part=np.array([r["part"] for r in rows]), pair_channel=np.array([r["channel"] for r in rows]),
                pair_sample=np.array([r["sample_id"] for r in rows]),
                pair_offset=np.concatenate(([0], np.cumsum(sizes))).astype(np.int64),
                pair_aspect=np.array([r["aspect"] for r in rows], dtype=np.float64),
                pair_fps=np.array([r["fps"] for r in rows], dtype=np.float64),
                pair_lag=np.array([r["alignment"]["lag"] for r in rows], dtype=np.int8),
                pair_best_lag=np.array([r["alignment"]["best_lag"] for r in rows], dtype=np.int8),
                pair_lag0_score=np.array([r["alignment"]["lag0_score"] for r in rows], dtype=np.float32),
                pair_bone_frames=np.array([r["alignment"]["frames"] for r in rows], dtype=np.int32))
    np.savez_compressed(path, schema=np.array(SCHEMA), **meta, **arrays)
    parts = collections.Counter(r["part"] for r in rows)
    frames = collections.Counter()
    for r, n in zip(rows, sizes):
        frames[r["part"]] += n
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), pairs=len(rows),
                frames=int(sum(sizes)), pairs_by_part=dict(parts), frames_by_part=dict(frames))


def load_shard(path: Path, expected_sha256: str | None = None) -> dict:
    """A shard's arrays (``allow_pickle=False``), its SHA-256 checked when given."""
    payload = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise SystemExit(f"{path}: SHA-256 differs from the pairs index")
    with np.load(path, allow_pickle=False) as z:
        if str(z["schema"]) != SCHEMA:
            raise SystemExit(f"{path}: not a {SCHEMA} shard")
        return {k: z[k] for k in z.files}


def _init(settings: dict) -> None:
    import torch
    torch.set_num_threads(1)
    from train_text_to_motion import make_store
    _WORKER.update(settings=settings, store=make_store(settings["store"]), archives={},
                   held_out=frozenset(settings["held_out"]))


def _build(job: dict) -> dict:
    import uwb_lifter_gate as gate
    from signtranslator.data_engineering import uwb_keypoints as uk
    from signtranslator.pose.multichannel import MultichannelMotion
    refuse_held_out([job["video_id"]], _WORKER["held_out"])
    s, store = _WORKER["settings"], _WORKER["store"]
    base = {k: job[k] for k in ("video_id", "clip_id", "sample_id", "part", "channel", "fps", "meta_fps")}
    try:
        archive = _WORKER["archives"].get(job["zip"])
        if archive is None:
            archive = _WORKER["archives"][job["zip"]] = gate.open_member_zip(store, f"{s['uwb_prefix']}/{job['zip']}")
        clip = uk.parse_clip(archive.read(job["clip_id"] + ".json"), job["clip_id"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.npz"
            path.write_bytes(store.get(job["state_key"]))
            ours = MultichannelMotion.load(path, expected_sha256=job["state_sha256"])
        result = pair_arrays(ours, clip, job["fps"], job["meta_fps"], s["filters"])
    except (uk.Refusal, ValueError, KeyError) as error:
        return dict(base, status="refused", reason=f"{type(error).__name__}"[:60])
    return dict(base, **result)


def candidate_jobs(split: dict, uwb: dict, reports: dict, index: dict, *, min_text_matches: int = 3):
    """(jobs, counts): every candidate pair of the split's train and lifter-val videos, sorted by (video, clip)."""
    import uwb_lifter_gate as gate
    part_of = {v: p for p in ("train", "lifter_val") for v in split[p]["videos"]}
    refuse_held_out(reports, set(split["held_out"]["videos"]))
    counts = collections.Counter()
    jobs = []
    for video, report in sorted(reports.items()):
        if video not in part_of:
            raise SystemExit(f"{video} is in no training part of the split")
        meta = report.get("meta") or {}
        channel = meta.get("channel_id")
        if channel not in split[part_of[video]]["channels"]:
            raise SystemExit(f"{video}: channel {channel} is not a {part_of[video]} channel of the split")
        clips = gate.uwb_clips_of(uwb[video])
        cues = report.get("segments", [])
        estimate = gate.estimate_fps(clips, cues, min_text_matches=min_text_matches)
        if estimate["fps"] is None:
            counts["video: no fps estimate"] += 1
            continue
        if not stream_match(estimate["fps"], meta.get("fps")):
            counts["video: stream mismatch"] += 1
            counts["pairs: stream mismatch"] += len(gate.pair_clips(clips, cues, estimate["fps"]))
            continue
        counts["videos with candidate pairs"] += 1
        for clip, cue, iou in gate.pair_clips(clips, cues, estimate["fps"]):
            if clip["clip_id"] not in index:
                counts["pairs: clip not in the zips"] += 1
                continue
            jobs.append(dict(video_id=video, clip_id=clip["clip_id"], sample_id=cue["sample_id"], part=part_of[video],
                             channel=channel, fps=estimate["fps"], meta_fps=meta.get("fps"), iou=iou,
                             zip=index[clip["clip_id"]], state_key=cue["state_key"], state_sha256=cue["state_sha256"]))
    return sorted(jobs, key=lambda j: (j["video_id"], j["clip_id"])), counts


def main(argv=None) -> int:
    import uwb_lifter_gate as gate
    from uwb_downstream_check import worker_count
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2")
    p.add_argument("--uwb-prefix", default=gate.UWB_PREFIX)
    p.add_argument("--youtube-prefix", default=gate.YOUTUBE_PREFIX)
    p.add_argument("--translations", default="YT.translations.all.json")
    p.add_argument("--split-file", required=True, help="build_uwb_lifter_split.py output: path, key or KEY@SHA256")
    p.add_argument("--members-cache", help="local JSON cache of the zips' member lists")
    p.add_argument("--videos", type=int, default=0, help="smoke: at most this many videos (seeded); 0: all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard-pairs", type=int, default=2000)
    p.add_argument("--workers", type=int, default=0, help="0: from sched_getaffinity and cgroup memory")
    p.add_argument("--out-dir", required=True, help="local directory for the shards and pairs-index.json (new)")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    split, split_sha = gate.load_split(store, args.split_file)
    held_out = set(split["held_out"]["videos"])
    allowed = set(split["train"]["videos"]) | set(split["lifter_val"]["videos"])
    uwb = json.loads(store.get(f"{args.uwb_prefix}/{args.translations}"))
    ids = sorted(v for v in allowed if v in uwb)
    if args.videos:
        ids = sorted(random.Random(args.seed).sample(ids, min(len(ids), args.videos)))
    refuse_held_out(ids, held_out)
    workers = args.workers or worker_count()
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        loaded = list(pool.map(gate._load_report, [(args.store, f"{args.youtube_prefix}/videos/{v}.json") for v in ids],
                               chunksize=16))
    reports = {v: r for v, r in zip(ids, loaded) if r is not None and r.get("status") == "converted"}
    index = gate.members_index(store, args.uwb_prefix, Path(args.members_cache) if args.members_cache else None)
    jobs, counts = candidate_jobs(split, uwb, reports, index)
    refuse_held_out([j["video_id"] for j in jobs], held_out)
    print(json.dumps(dict(videos=len(reports), candidate_pairs=len(jobs), **counts)), flush=True)
    settings = dict(store=args.store, uwb_prefix=args.uwb_prefix, held_out=sorted(held_out), filters=FILTERS)
    shards, buffer, refusals, kept = [], [], collections.Counter(), collections.Counter()
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_init, initargs=(settings,)) as pool:
        for n, result in enumerate(pool.map(_build, jobs, chunksize=4), 1):
            if result["status"] == "kept":
                buffer.append(result)
                kept[result["part"]] += 1
            else:
                refusals[result["reason"]] += 1
            if len(buffer) >= args.shard_pairs:
                shards.append(write_shard(out_dir / f"shard-{len(shards):05d}.npz", buffer))
                buffer = []
            if n % 1000 == 0 or n == len(jobs):
                print(f"pairs {n}/{len(jobs)} kept {sum(kept.values())}", flush=True)
    if buffer:
        shards.append(write_shard(out_dir / f"shard-{len(shards):05d}.npz", buffer))
    videos = collections.defaultdict(set)
    for job in jobs:
        videos[job["part"]].add(job["video_id"])
    index_out = dict(schema=SCHEMA, split_file=args.split_file, split_sha256=split_sha, filters=FILTERS,
                     stream_tolerance=STREAM_TOLERANCE, arm_targets=list(ARM_TARGETS), label_points=list(LABEL_POINTS),
                     axes="MediaPipe (video_landmarks.Y_UP applied to our y-up forward kinematics)",
                     videos_read=len(reports), candidate_pairs=len(jobs), video_counts=dict(counts),
                     kept_pairs=dict(kept), refused_pairs=dict(refusals), shards=shards,
                     candidate_videos={p: len(v) for p, v in videos.items()},
                     settings=dict(videos=args.videos, seed=args.seed, shard_pairs=args.shard_pairs, workers=workers))
    (out_dir / "pairs-index.json").write_text(json.dumps(index_out, indent=1, sort_keys=True))
    print(json.dumps({k: index_out[k] for k in ("candidate_pairs", "kept_pairs", "refused_pairs", "video_counts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
