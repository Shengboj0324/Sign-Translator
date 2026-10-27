"""Round 10: one-word clips cut to the sign and a margin, in a copy of a mixed text manifest.

The 2026-10-27 data audit found the lexical movement of a one-word clip in only about 0.3-0.7 of it, with 0.5-0.7 s of
rest on each side (RIT, ASLLVD) or untracked arm frames (WLASL), so one-word clips carry about 20 word-specific frames
out of about 50. For every record of the ``--corpus`` corpora, the dominant wrist's height above its shoulder is read
from the state as the dataset serves it (conformed, 24 fps; ``skeleton_positions``):
- the dominant side is the one whose tracked height varies most;
- active frames are tracked frames above rest + ``--threshold`` x (peak - rest), with rest and peak the 10th and 95th
  percentiles of the tracked heights;
- the window is [first active - margin, last active + margin], clipped to the clip, and recorded as
  ``continuous.window_seconds`` (seconds from the first frame; ``load_sample`` applies it), with ``seconds`` updated.
A record whose dominant arm is tracked on fewer than ``--min-tracked`` of its frames, or whose window is shorter than
``--min-seconds``, is dropped (counted). Every other record is copied unchanged. Written once.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def sign_window(heights: np.ndarray, tracked: np.ndarray, fps: float, threshold: float, margin: float):
    """(lo, hi) seconds from the first frame, or None when no frame is tracked."""
    if not tracked.any():
        return None
    h = heights[tracked]
    rest, peak = np.percentile(h, 10), np.percentile(h, 95)
    active = np.flatnonzero(tracked & (heights > rest + threshold * (peak - rest)))
    if not len(active):
        return None
    last = (len(heights) - 1) / fps
    return max(active[0] / fps - margin, 0.), min(active[-1] / fps + margin, last)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest-key", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--corpus", action="append", required=True)
    p.add_argument("--threshold", type=float, default=.1)
    p.add_argument("--margin", type=float, default=.25)
    p.add_argument("--min-tracked", type=float, default=.3)
    p.add_argument("--min-seconds", type=float, default=.5)
    p.add_argument("--corpus-id", required=True)
    p.add_argument("--output-key", required=True)
    p.add_argument("--limit", type=int, help="first N records per corpus only (a dry run; nothing is written)")
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("ignore")
    from signtranslator.data.text_motion_dataset import R2Store, TextMotionDataset
    from signtranslator.data_engineering.corpus_manifest import (
        ContentCache, build_corpus_manifest, canonical_json, load_manifest)
    from signtranslator.generation.canonical_skeleton import skeleton_positions
    from canonicalize_asl_citizen import side_tracks
    from train_motion_tokens import layout_template
    store = R2Store(os.environ)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.manifest_key) + b"\n")
        source = load_manifest(path, expected_sha256=args.manifest_sha256)
    cache = ContentCache(Path(tempfile.gettempdir()) / "sign-translator-cache", 16 * 2 ** 30)
    template = layout_template(source, "how2sign-two-view-text-v1", cache, store.get)
    served = {}                                         # sign -> (dataset, index), as training serves each record
    for part in ("train", "val", "test"):
        ds = TextMotionDataset(source, partition=part, cache=cache, fetch=store.get, max_seconds=1e9, fps=24.,
                               template=template)
        served.update({r["sign"]: (ds, k) for k, r in enumerate(ds.records)})
    why, windows, lengths = collections.Counter(), {}, collections.defaultdict(list)
    seen = collections.Counter()
    for record in source["records"]:
        corpus = record.get("corpus")
        if corpus not in args.corpus:
            continue
        if args.limit and seen[corpus] >= args.limit:
            continue
        seen[corpus] += 1
        if record["sign"] not in served:
            why[f"{corpus}:not_served"] += 1; windows[record["sign"]] = None; continue
        ds, k = served[record["sign"]]
        _, state = ds[k]
        joints, pos, _ = skeleton_positions(state)
        tracks = side_tracks(state)
        best = None
        for side in ("Right", "Left"):
            ok = np.asarray(tracks[side][0], dtype=bool)
            h = pos[:, joints.index(f"{side}Hand"), 1] - pos[:, joints.index(f"{side}Shoulder"), 1]
            spread = float(np.ptp(h[ok])) if ok.any() else -1.
            if best is None or spread > best[0]:
                best = (spread, side, ok, h)
        _, side, ok, h = best
        if ok.mean() < args.min_tracked:
            why[f"{corpus}:arm_untracked"] += 1; windows[record["sign"]] = None; continue
        w = sign_window(h, ok, 24., args.threshold, args.margin)
        if w is None or w[1] - w[0] < args.min_seconds:
            why[f"{corpus}:no_sign_window"] += 1; windows[record["sign"]] = None; continue
        windows[record["sign"]] = (round(w[0], 4), round(w[1], 4))
        lengths[corpus].append((w[1] - w[0], (len(h) - 1) / 24.))
        why[f"{corpus}:trimmed"] += 1
    summary = {c: dict(clips=len(v), median_window_seconds=round(float(np.median([a for a, _ in v])), 3),
                       median_clip_seconds=round(float(np.median([b for _, b in v])), 3)) for c, v in lengths.items()}
    print(json.dumps(dict(why=dict(sorted(why.items())), summary=summary)), flush=True)
    if args.limit:
        return 0
    records = []
    for record in source["records"]:
        if record.get("corpus") not in args.corpus:
            records.append({k: v for k, v in record.items() if k != "usable"}); continue
        w = windows.get(record["sign"])
        if w is None:
            continue
        cont = dict(record["continuous"], window_seconds=list(w), seconds=round(w[1] - w[0] + 1 / 24., 4))
        records.append(dict({k: v for k, v in record.items() if k != "usable"}, continuous=cont))
    split = dict(source["split"], counts={p: sum(r["partition"] == p for r in records) for p in ("train", "val", "test")})
    out = build_corpus_manifest(corpus_id=args.corpus_id, inputs={"source_manifest": args.manifest_sha256},
                                records=records, split=split)
    for key in ("license", "licence"):
        if key in source:
            out[key] = source[key]
    out["trim"] = dict(rule="one-word clips cut to the dominant wrist's active span (height above rest + threshold x "
                            "(peak - rest)) +- margin", corpora=sorted(args.corpus), threshold=args.threshold,
                       margin=args.margin, min_tracked=args.min_tracked, min_seconds=args.min_seconds,
                       counts=dict(sorted(why.items())), summary=summary)
    if store.exists(args.output_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.output_key, canonical_json(out))
    print(json.dumps(dict(manifest_key=args.output_key, manifest_sha256=digest, records=len(records))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
