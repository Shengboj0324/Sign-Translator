"""Channel-disjoint split of the UWB-paired YouTube-ASL videos for UWB lifter v2 (CPU, reads only).

Declared in docs/NEXT_PHASE_PLAN.md, "UWB lifter v2 and gate v2: declared" (2026-10-30, 11:12 UTC):

- **population:** every video we extracted (``<youtube prefix>/videos/<id>.json`` with status ``converted``) that the
  UWB keypoint dataset also holds (its translations file), less the 70 development videos (``--exclude-videos``,
  ``reports/uwb_gate_smoke_videos.txt``: dropped everywhere) and less videos without a ``channel_id`` (no channel can be
  held out: dropped everywhere, listed);
- **held-out:** channels whose ``sha256("uwb-lifter-v2:" + channel)``, read as an integer over 2^256, is below 0.20
  (an exact integer comparison), plus every monitor channel (``build_youtube_asl_manifest.monitor_channel`` at the
  manifests' monitor share, 0.01);
- **lifter-val:** of the remaining channels, those whose ``sha256("uwb-lifter-v2-val:" + channel)`` is below 0.10;
- **train:** the rest.

Output (JSON, sorted, deterministic: the same inputs give the same bytes; the SHA-256 is printed): schema
``uwb-lifter-split-v1``, the salts and shares, ``train`` / ``lifter_val`` / ``held_out`` with their ``videos`` and
``channels``, the dropped videos, and ``composition`` per part (channels, videos, UWB clips, our cues, candidate pairs
by the gate's own fps estimate and pairing, monitor channels, and the largest channel's share of videos, clips and
candidate pairs). ``--output-key`` also stores it (refused if the key exists).
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

SCHEMA = "uwb-lifter-split-v1"
HELD_OUT_SALT, VAL_SALT = "uwb-lifter-v2:", "uwb-lifter-v2-val:"
HELD_OUT_SHARE, VAL_SHARE, MONITOR_SHARE = "0.20", "0.10", .01
PARTS = ("train", "lifter_val", "held_out")


def below(salt: str, channel: str, share: str) -> bool:
    """sha256(salt + channel) / 2^256 < share, compared exactly (``share`` a decimal string)."""
    value = int(hashlib.sha256((salt + channel).encode()).hexdigest(), 16)
    return Fraction(value, 2 ** 256) < Fraction(share)


def part_of(channel: str, *, held_out_share: str = HELD_OUT_SHARE, val_share: str = VAL_SHARE,
            monitor_share: float = MONITOR_SHARE) -> str:
    from build_youtube_asl_manifest import monitor_channel
    if monitor_channel(channel, monitor_share) or below(HELD_OUT_SALT, channel, held_out_share):
        return "held_out"
    return "lifter_val" if below(VAL_SALT, channel, val_share) else "train"


def split_videos(videos: dict, excluded: set, **shares) -> dict:
    """``videos``: id -> {channel, uwb_clips, our_cues, candidate_pairs}. The split (without composition)."""
    from build_youtube_asl_manifest import monitor_channel
    parts = {p: dict(videos=[], channels=set()) for p in PARTS}
    dropped = dict(development=[], no_channel=[])
    monitor = set()
    for video, row in sorted(videos.items()):
        if video in excluded:
            dropped["development"].append(video)
            continue
        channel = row.get("channel")
        if not channel:
            dropped["no_channel"].append(video)
            continue
        part = part_of(channel, **shares)
        parts[part]["videos"].append(video)
        parts[part]["channels"].add(channel)
        if monitor_channel(channel, shares.get("monitor_share", MONITOR_SHARE)):
            monitor.add(channel)
    out = {p: dict(videos=sorted(v["videos"]), channels=sorted(v["channels"])) for p, v in parts.items()}
    out["dropped"] = {k: sorted(v) for k, v in dropped.items()}
    out["monitor_channels"] = sorted(monitor)
    return out


def composition(split: dict, videos: dict) -> dict:
    out = {}
    for part in PARTS:
        ids = split[part]["videos"]
        by_channel = collections.defaultdict(lambda: collections.Counter())
        for v in ids:
            row = videos[v]
            by_channel[row["channel"]].update(videos=1, uwb_clips=row.get("uwb_clips", 0),
                                              our_cues=row.get("our_cues", 0),
                                              candidate_pairs=row.get("candidate_pairs", 0))
        totals = collections.Counter()
        for c in by_channel.values():
            totals.update(c)
        largest = max(sorted(by_channel), key=lambda c: by_channel[c]["videos"]) if by_channel else None
        out[part] = dict(channels=len(by_channel), videos=len(ids), uwb_clips=totals["uwb_clips"],
                         our_cues=totals["our_cues"], candidate_pairs=totals["candidate_pairs"],
                         monitor_channels=len(set(split["monitor_channels"]) & set(by_channel)),
                         largest_channel=None if largest is None else dict(
                             channel=largest, **{k: by_channel[largest][k] for k in
                                                 ("videos", "uwb_clips", "candidate_pairs")},
                             share_of_videos=round(by_channel[largest]["videos"] / max(len(ids), 1), 4),
                             share_of_uwb_clips=round(by_channel[largest]["uwb_clips"] / max(totals["uwb_clips"], 1), 4),
                             share_of_candidate_pairs=round(by_channel[largest]["candidate_pairs"]
                                                            / max(totals["candidate_pairs"], 1), 4)))
    return out


def video_rows(uwb: dict, reports: dict, min_text_matches: int = 3) -> dict:
    """id -> {channel, uwb_clips, our_cues, candidate_pairs, fps} by the gate's own estimate and pairing."""
    import uwb_lifter_gate as gate
    out = {}
    for video, report in sorted(reports.items()):
        clips = gate.uwb_clips_of(uwb[video])
        cues = report.get("segments", [])
        estimate = gate.estimate_fps(clips, cues, min_text_matches=min_text_matches)
        pairs = gate.pair_clips(clips, cues, estimate["fps"]) if estimate["fps"] is not None else []
        out[video] = dict(channel=(report.get("meta") or {}).get("channel_id"), uwb_clips=len(clips),
                          our_cues=len(cues), candidate_pairs=len(pairs), fps=estimate["fps"],
                          meta_fps=(report.get("meta") or {}).get("fps"))
    return out


def main(argv=None) -> int:
    import uwb_lifter_gate as gate
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2")
    p.add_argument("--uwb-prefix", default=gate.UWB_PREFIX)
    p.add_argument("--youtube-prefix", default=gate.YOUTUBE_PREFIX)
    p.add_argument("--translations", default="YT.translations.all.json")
    p.add_argument("--exclude-videos", required=True, help="the development videos (reports/uwb_gate_smoke_videos.txt)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--output-key", help="also store the split here (refused if it exists)")
    args = p.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists; a rerun writes elsewhere")
    excluded = gate.read_video_list(args.exclude_videos)
    uwb = json.loads(store.get(f"{args.uwb_prefix}/{args.translations}"))
    keys = sorted(k for k in store.list(f"{args.youtube_prefix}/videos/") if k.endswith(".json"))
    ids = [Path(k).name[:-5] for k in keys if Path(k).name[:-5] in uwb]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        loaded = list(pool.map(gate._load_report, [(args.store, f"{args.youtube_prefix}/videos/{v}.json") for v in ids],
                               chunksize=16))
    reports = {v: r for v, r in zip(ids, loaded) if r is not None and r.get("status") == "converted"}
    videos = video_rows(uwb, reports)
    split = split_videos(videos, excluded)
    out = dict(schema=SCHEMA, declaration="docs/NEXT_PHASE_PLAN.md: UWB lifter v2 and gate v2: declared "
                                          "(2026-10-30, 11:12 UTC)",
               rule=dict(held_out=f"sha256('{HELD_OUT_SALT}' + channel) / 2^256 < {HELD_OUT_SHARE}, plus every "
                                  f"monitor channel (build_youtube_asl_manifest.monitor_channel, share {MONITOR_SHARE})",
                         lifter_val=f"of the rest, sha256('{VAL_SALT}' + channel) / 2^256 < {VAL_SHARE}",
                         train="the rest", dropped="the development videos and videos without a channel id"),
               salts=dict(held_out=HELD_OUT_SALT, lifter_val=VAL_SALT),
               shares=dict(held_out=HELD_OUT_SHARE, lifter_val=VAL_SHARE, monitor=MONITOR_SHARE),
               population=dict(uwb_and_ours_converted=len(reports), development_videos_listed=len(excluded)),
               **{k: split[k] for k in (*PARTS, "dropped", "monitor_channels")},
               composition=composition(split, videos))
    text = json.dumps(out, indent=1, sort_keys=True)
    Path(args.out).write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    if args.output_key:
        store.put(args.output_key, text.encode())
    print(json.dumps(out["composition"], indent=1))
    print(f"split sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
