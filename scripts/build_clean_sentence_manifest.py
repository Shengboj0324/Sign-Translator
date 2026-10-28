"""Fix 4: sentence clips cut to their signing and likely caption misalignments dropped, in a copy of a mixed manifest.

Sentence clips (How2Sign, YouTube-ASL, ASLLRP) start and end with rest or with the neighbouring caption's signing, and
some captions are misaligned with their clip. For every multi-word record (two or more words in ``text``) of the
``--corpus`` corpora in the ``--partitions`` partitions (default: train only, so the validation and test sets every
pilot reads stay identical), the state is read as training serves it (``load_sample`` + ``conform`` to the How2Sign
layout + 24 fps) and:

- per side, the wrist height above its shoulder (``skeleton_positions``, canonical skeleton units, so the same for
  every signer) on the frames whose forearm is tracked (``side_tracks``); rest = the side's 10th percentile, but never
  above ``--rest-ceiling`` (an arm hanging or a hand in the lap reads about -0.35 to -0.46; a clip signed throughout has
  no rest, and its lowest signing must not be taken for rest); a frame is active on a side when its height exceeds
  rest + ``--threshold`` x the larger of the two sides' ranges (95th percentile - rest), so a resting non-dominant
  hand's noise does not count, and at least ``--min-lift`` above rest, so a clip at rest throughout has no activity;
  activity is the union of both sides, with runs shorter than ``--min-run`` frames removed;
- the signing span is [first active frame, last active frame]; the window is that span +- ``--margin`` seconds, clipped
  to the clip, written as ``continuous.window_seconds`` (seconds from the served state's first frame, composed with any
  window the record already had) with ``seconds`` updated.

A record is dropped, counted per corpus and reason, when (in this order):
- ``arms_untracked``: neither forearm is tracked on more than ``--max-untracked`` of its frames (no span can be
  measured; with ``--keep-untracked`` such records are copied unchanged instead, counted ``arms_untracked_kept``. On
  2026-10-27 about 60% of a random 200 YouTube-ASL sentence clips had no tracked arm at all while their hands were
  tracked: whether to drop them is a data decision, not a misalignment finding);
- ``no_activity``: no active run, or a span shorter than ``--min-seconds``;
- ``low_coverage``: the span (without margin) covers less than ``--min-coverage`` of the clip - the caption's clip is
  mostly rest, or most of its signing is someone else's caption;
- ``rate_outlier``: its words per active second lie outside median +- ``--rate-k`` x MAD over the corpus's records that
  passed the first three checks (MAD scaled by 1.4826, a consistent estimate of the standard deviation, so k = 3 is
  about 3 sigma). Far too many words for the signing time, or far too few, is the mark of a caption cut to the wrong
  clip. ``--rate-scale log`` (default) takes the band on log(rate): rates are ratios and right-skewed, and on the linear
  scale the lower bound falls below zero (How2Sign: median 2.45, scaled MAD 1.0 words/s), so too-few-words captions
  would never be caught; ``linear`` is the plain band.
Every other record (single-word texts, other corpora, other partitions, records training never serves) is copied
unchanged. ``--limit N`` reads only the first N candidate records per corpus and prints the summary (nothing written);
otherwise the manifest is written once (refusing to overwrite).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

FPS = 24.
MAD_SCALE = 1.4826
WORD = re.compile(r"[A-Za-z0-9']+")


def word_count(text: str) -> int:
    return len(WORD.findall(text or ""))


def drop_short_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """``mask`` without its True runs shorter than ``min_run`` frames."""
    mask = np.asarray(mask, dtype=bool).copy()
    if min_run <= 1 or not mask.any():
        return mask
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start < min_run:
            mask[start:stop] = False
    return mask


def active_frames(heights: np.ndarray, tracked: np.ndarray, threshold: float = .2, min_run: int = 3,
                  rest_ceiling: float = -.3, min_lift: float = .05) -> np.ndarray:
    """(T,) union of both sides' activity; ``heights``, ``tracked``: (sides, T) wrist height above shoulder and arm
    tracking. The level is ``threshold`` x the largest side range, above each side's own rest (its 10th percentile,
    at most ``rest_ceiling``), and at least ``min_lift``."""
    heights, tracked = np.asarray(heights, dtype=float), np.asarray(tracked, dtype=bool)
    rests, ranges = [], []
    for h, ok in zip(heights, tracked):
        if ok.any():
            rest, peak = min(np.percentile(h[ok], 10), rest_ceiling), np.percentile(h[ok], 95)
            rests.append(rest); ranges.append(max(peak - rest, 0.))
        else:
            rests.append(None); ranges.append(0.)
    level = max(threshold * max(ranges, default=0.), min_lift)
    active = np.zeros(heights.shape[1], dtype=bool)
    for h, ok, rest in zip(heights, tracked, rests):
        if rest is not None:
            active |= ok & (h > rest + level)
    return drop_short_runs(active, min_run)


def signing_span(active: np.ndarray, fps: float, margin: float):
    """(span, window) in seconds from the first frame, or None: span = [first, last] active frame, window = span +-
    margin clipped to the clip."""
    idx = np.flatnonzero(active)
    if not len(idx):
        return None
    last = (len(active) - 1) / fps
    span = (idx[0] / fps, idx[-1] / fps)
    return span, (max(span[0] - margin, 0.), min(span[1] + margin, last))


def robust_band(values, k: float = 3.):
    """(median, scaled MAD, lo, hi) of ``values``; the band is median +- k x 1.4826 x MAD."""
    v = np.asarray(values, dtype=float)
    if not len(v):
        return None
    med = float(np.median(v))
    mad = MAD_SCALE * float(np.median(np.abs(v - med)))
    return med, mad, med - k * mad, med + k * mad


def judge(*, untracked: float, span, frames: int, words: int, fps: float, max_untracked: float, min_seconds: float,
          min_coverage: float):
    """(reason or None, measures) by the first three rules; the rate rule needs the corpus band (``rate_outliers``)."""
    if untracked > max_untracked:
        return "arms_untracked", dict(untracked=untracked)
    if span is None or span[0][1] - span[0][0] + 1 / fps < min_seconds:
        return "no_activity", dict(untracked=untracked)
    active_seconds = span[0][1] - span[0][0] + 1 / fps
    coverage = active_seconds / (frames / fps)
    measures = dict(untracked=untracked, coverage=coverage, active_seconds=active_seconds, rate=words / active_seconds)
    if coverage < min_coverage:
        return "low_coverage", measures
    return None, measures


def rate_outliers(rates: dict, k: float, scale: str = "log"):
    """({key: is outlier}, band in words per second) for one corpus's {key: words per active second}."""
    f, inv = (np.log, np.exp) if scale == "log" else (lambda x: x, lambda x: x)
    band = robust_band([f(r) for r in rates.values()], k)
    if band is None:
        return {}, None
    med, mad, lo, hi = band
    out = {key: not (lo <= f(r) <= hi) for key, r in rates.items()}
    return out, (float(inv(med)), mad, float(inv(lo)), float(inv(hi)))


def wrist_tracks(state):
    """(heights (2, T), tracked (2, T)): wrist height above the shoulder and forearm tracking, Left then Right."""
    from signtranslator.generation.canonical_skeleton import skeleton_positions
    from canonicalize_asl_citizen import side_tracks
    joints, pos, _ = skeleton_positions(state)
    tracks = side_tracks(state)
    heights = np.stack([pos[:, joints.index(f"{s}Hand"), 1] - pos[:, joints.index(f"{s}Shoulder"), 1]
                        for s in ("Left", "Right")])
    tracked = np.stack([np.asarray(tracks[s][0], dtype=bool) for s in ("Left", "Right")])
    return heights, tracked


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest-key", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--corpus", action="append", required=True)
    p.add_argument("--partitions", default="train", help="comma-separated; others are copied unchanged")
    p.add_argument("--threshold", type=float, default=.2)
    p.add_argument("--rest-ceiling", type=float, default=-.3, help="canonical skeleton units (wrist above shoulder)")
    p.add_argument("--min-lift", type=float, default=.05, help="canonical skeleton units")
    p.add_argument("--min-run", type=int, default=3, help="frames (24 fps)")
    p.add_argument("--margin", type=float, default=.3)
    p.add_argument("--max-untracked", type=float, default=.5)
    p.add_argument("--keep-untracked", action="store_true", help="copy arm-untracked records unchanged (not dropped)")
    p.add_argument("--min-seconds", type=float, default=.5)
    p.add_argument("--min-coverage", type=float, default=.4)
    p.add_argument("--rate-k", type=float, default=3.)
    p.add_argument("--rate-scale", choices=("log", "linear"), default="log")
    p.add_argument("--workers", type=int, default=8, help="parallel R2 reads")
    p.add_argument("--corpus-id")
    p.add_argument("--output-key")
    p.add_argument("--limit", type=int, help="first N candidate records per corpus only (a dry run; nothing is written)")
    args = p.parse_args(argv)
    if not args.limit and not (args.corpus_id and args.output_key):
        p.error("--corpus-id and --output-key are required unless --limit is given")
    import warnings
    warnings.simplefilter("ignore")
    from concurrent.futures import ThreadPoolExecutor
    from signtranslator.data.layout_adapter import conform
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.corpus_manifest import (
        ContentCache, build_corpus_manifest, canonical_json, load_manifest, load_sample, usable_channels)
    from evaluate_text_to_motion import at_rate
    from train_motion_tokens import layout_template
    store = R2Store(os.environ)
    if args.output_key and not args.limit and store.exists(args.output_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.manifest_key) + b"\n")
        source = load_manifest(path, expected_sha256=args.manifest_sha256)
    known = {r.get("corpus") for r in source["records"]}
    if set(args.corpus) - known:
        raise SystemExit(f"unknown corpora: {sorted(set(args.corpus) - known)}")
    partitions = set(args.partitions.split(","))
    cache = ContentCache(Path(tempfile.gettempdir()) / "sign-translator-cache", 16 * 2 ** 30)
    template = layout_template(source, "how2sign-two-view-text-v1", cache, store.get)

    why, seen = collections.Counter(), collections.Counter()
    candidates = []
    for record in source["records"]:
        corpus = record.get("corpus")
        if corpus not in args.corpus or record.get("partition") not in partitions:
            continue
        if word_count(record.get("text")) < 2:
            why[f"{corpus}:single_word_unchanged"] += 1; continue
        if not usable_channels(record, "continuous"):
            why[f"{corpus}:not_served_unchanged"] += 1; continue
        if args.limit and seen[corpus] >= args.limit:
            continue
        seen[corpus] += 1
        candidates.append(record)

    def measure(record):
        state = at_rate(conform(load_sample(record, "continuous", cache=cache, fetch=store.get), template), FPS)
        heights, tracked = wrist_tracks(state)
        frames = heights.shape[1]
        untracked = float((~tracked.any(0)).mean()) if frames else 1.
        active = active_frames(heights, tracked, args.threshold, args.min_run, args.rest_ceiling, args.min_lift)
        span = signing_span(active, FPS, args.margin)
        reason, m = judge(untracked=untracked, span=span, frames=frames, words=word_count(record["text"]), fps=FPS,
                          max_untracked=args.max_untracked, min_seconds=args.min_seconds,
                          min_coverage=args.min_coverage)
        return reason, m, None if span is None else span[1], (frames - 1) / FPS if frames else 0.
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(measure, candidates))

    decided, rates, measures = {}, collections.defaultdict(dict), collections.defaultdict(list)
    for record, (reason, m, window, clip) in zip(candidates, results):
        corpus = record["corpus"]
        if reason == "arms_untracked" and args.keep_untracked:
            why[f"{corpus}:arms_untracked_kept"] += 1; continue
        if reason:
            decided[record["sign"]] = None; why[f"{corpus}:{reason}"] += 1
        else:
            decided[record["sign"]] = window; rates[corpus][record["sign"]] = m["rate"]
        if "coverage" in m:
            measures[corpus].append((m["coverage"], m["rate"], clip, m["active_seconds"]))
    bands = {}
    for corpus, r in rates.items():
        outlier, band = rate_outliers(r, args.rate_k, args.rate_scale)
        bands[corpus] = dict(median=round(band[0], 4), scaled_mad=round(band[1], 4), lo=round(band[2], 4),
                             hi=round(band[3], 4), scale=args.rate_scale)
        for sign, out in outlier.items():
            if out:
                decided[sign] = None; why[f"{corpus}:rate_outlier"] += 1
            else:
                why[f"{corpus}:trimmed"] += 1
    summary = {}
    for corpus, v in measures.items():
        cov, rate, clip, act = (np.array(x) for x in zip(*v))
        summary[corpus] = dict(read=seen[corpus], median_coverage=round(float(np.median(cov)), 3),
                               median_words_per_active_second=round(float(np.median(rate)), 3),
                               median_clip_seconds=round(float(np.median(clip)), 3),
                               median_active_seconds=round(float(np.median(act)), 3), rate_band=bands.get(corpus))
    print(json.dumps(dict(why=dict(sorted(why.items())), summary=summary)), flush=True)
    if args.limit:
        return 0

    records = []
    for record in source["records"]:
        base = {k: v for k, v in record.items() if k != "usable"}
        if record["sign"] not in decided:
            records.append(base); continue
        w = decided[record["sign"]]
        if w is None:
            continue
        offset = (record["continuous"].get("window_seconds") or [0., 0.])[0]
        lo, hi = round(offset + w[0], 4), round(offset + w[1], 4)
        cont = dict(record["continuous"], window_seconds=[lo, hi], seconds=round(hi - lo + 1 / FPS, 4))
        records.append(dict(base, continuous=cont))
    split = dict(source["split"], counts={p: sum(r["partition"] == p for r in records) for p in ("train", "val", "test")})
    out = build_corpus_manifest(corpus_id=args.corpus_id, inputs={"source_manifest": args.manifest_sha256},
                                records=records, split=split)
    for key in ("license", "licence", "trim"):
        if key in source:
            out[key] = source[key]
    out["sentence_clean"] = dict(
        rule="multi-word clips cut to the union of both wrists' active span (height above own rest, at most "
             "rest_ceiling, + threshold x the larger side range, runs >= min_run frames) +- margin; dropped: arms "
             "untracked, no activity, low coverage, words per active second outside median +- rate_k x 1.4826 MAD per "
             "corpus (on rate_scale)",
        corpora=sorted(args.corpus), partitions=sorted(partitions), threshold=args.threshold,
        rest_ceiling=args.rest_ceiling, min_lift=args.min_lift, rate_scale=args.rate_scale, min_run=args.min_run,
        margin=args.margin, max_untracked=args.max_untracked, keep_untracked=args.keep_untracked,
        min_seconds=args.min_seconds, min_coverage=args.min_coverage, rate_k=args.rate_k, counts=dict(sorted(why.items())), summary=summary)
    if store.exists(args.output_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.output_key, canonical_json(out))
    print(json.dumps(dict(manifest_key=args.output_key, manifest_sha256=digest, records=len(records))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
