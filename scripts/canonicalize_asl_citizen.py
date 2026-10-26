"""ASL Citizen canonical states: right-dominant, trimmed to the active signing (round 5 data).

The webcam recordings of ASL Citizen hold the signing hand on either side of the stored state: about 60% of clips
have it in the ``Left`` channels (left-handed participants or mirrored recordings), while How2Sign and the generator
sign right-handed. The resting hand mostly leaves the frame below, so the stored "other" arm is rarely tracked.
Per clip, this script:

1. **Dominant side:** the side with more tracked frames (forearm plus hand), among the sides with a tracked forearm.
   With ``--dominance signer`` (round 5 amendment 2) it is the signer's side for all of their clips, summed over the
   signer's clips by forearm frames (``signer_sides``): the per-clip rule split most signers' clips between the sides. A clip with no tracked forearm on
   either side is kept unchanged and marked ``no_arm``.
2. **Mirror** a left-dominant clip with ``signtranslator.pose.mirror.mirror_state``, which is exact against mirrored
   landmarks (``tests/test_pose_mirror.py``).
3. **Trim** to the active window: the first to the last frame with the dominant forearm tracked, widened by
   ``MARGIN_SECONDS`` on each side and clipped to the recording.
4. **Generator use** needs the window to last at least ``MIN_SECONDS``, with the dominant forearm tracked on at least
   ``MIN_ARM`` and the dominant hand on at least ``MIN_HAND`` of its frames.

The rule's constants are fixed here, declared with round 5 (docs/NEXT_PHASE_PLAN.md) before any result that uses
them.

``--shard i --shards n`` writes the canonical states (write-once) and a per-shard report. ``--assemble`` joins the
reports into the canonical manifest, which keeps the source manifest's partitions, glosses, pseudonyms and quality
counts: the recogniser's ``--min-hand-fraction`` filter reads the larger side's hand frames over all frames, which
mirroring does not change. Research use only under the source's terms; everything stays private in R2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

SOURCE_KEY = "acquisition/2026-10-17-v1/derived/asl-citizen-v1/manifest-v1/manifest.json"
SOURCE_SHA = "708af3a6748742c0ce8d147a984ef1de57abcec94da47ff0ce82391b067661a6"
PREFIX = "acquisition/2026-10-17-v1/derived/asl-citizen-v1/canonical-v1"
MARGIN_SECONDS, MIN_SECONDS, MIN_ARM, MIN_HAND = .2, .4, .7, .6


def side_tracks(state):
    """Per side: (forearm tracked (T,), hand tracked (T,)); the forearm needs its upper arm too."""
    body = state.channels["body"]
    labels = list(body.labels)
    out = {}
    for side in ("Left", "Right"):
        arm = body.valid[:, labels.index(f"{side}Forearm")] & body.valid[:, labels.index(f"{side}UpperArm")]
        out[side] = (arm, state.channels[f"{side.lower()}_hand"].valid.all(axis=1))
    return out


def canonicalize(state, trim: bool = True, side: str | None = None):
    """(canonical state, report) for one clip, by the rule in the module docstring (``trim=False``: mirror only).

    ``side`` forces the dominant side (round 5 amendment 2: the signer's, from ``signer_sides``); without it the clip
    decides from its own tracking (the per-clip rule of canonical-v1)."""
    from signtranslator.pose.mirror import mirror_state
    from render_avatar import slice_state
    tracks = side_tracks(state)
    score = {s: int(arm.sum() + hand.sum()) for s, (arm, hand) in tracks.items()}
    no_arm = not (tracks["Left"][0].any() or tracks["Right"][0].any())
    if side is None:
        if no_arm:
            return state, dict(dominant=None, mirrored=False, no_arm=True, window=None, generator_ok=False,
                               frames=len(state.timestamps))
        armed = [s for s in ("Left", "Right") if tracks[s][0].any()]   # the window needs a tracked forearm
        dominant = armed[0] if len(armed) == 1 else ("Left" if score["Left"] > score["Right"] else "Right")
    else:
        dominant = side
    mirrored = dominant == "Left"
    if mirrored:
        state = mirror_state(state, sample_id=state.sample_id)
    arm, hand = side_tracks(state)["Right"]
    t = state.timestamps
    idx = np.flatnonzero(arm)
    if trim and len(idx):
        lo, hi = max(t[idx[0]] - MARGIN_SECONDS, t[0]), min(t[idx[-1]] + MARGIN_SECONDS, t[-1])
    else:                      # mirror only, or the dominant forearm is never tracked: the whole clip
        lo, hi = t[0], t[-1]
    keep = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    state = slice_state(state, float(lo - t[0]), float(hi - lo) + 1e-6)
    seconds = float(t[keep][-1] - t[keep][0]) + (float(np.median(np.diff(t))) if len(t) > 1 else 0.)
    arm_fraction, hand_fraction = float(arm[keep].mean()), float(hand[keep].mean())
    ok = len(idx) > 0 and seconds >= MIN_SECONDS and arm_fraction >= MIN_ARM and hand_fraction >= MIN_HAND
    return state, dict(dominant=dominant, mirrored=mirrored, no_arm=no_arm, window=[float(lo - t[0]), float(hi - t[0])],
                       seconds=round(seconds, 4), arm_fraction=round(arm_fraction, 4),
                       hand_fraction=round(hand_fraction, 4), generator_ok=bool(ok), frames=int(keep.sum()),
                       score=score, dominance="signer" if side else "clip")


def signer_sides(records) -> dict:
    """Round 5 amendment 2: performer -> dominant side, for all of that signer's clips. A signer's handedness and
    recording (mirrored webcam or not) do not change between clips, so the side is the one whose forearm is tracked on
    more frames, summed over the signer's clips (each record's extraction quality counts). Hand counts are left out:
    the resting hand at the bottom of a webcam frame is detected through most of a clip while its forearm is not, so
    hand frames favour the non-dominant side."""
    import collections
    total = collections.defaultdict(lambda: collections.Counter())
    for r in records:
        q = r.get("quality") or {}
        if not q:
            raise SystemExit(f"{r['sign']}: no quality counts; the per-signer rule needs them")
        arms = q.get("arm_frames") or {}
        for s in ("Left", "Right"):
            total[r["performer_sha256"]][s] += int(arms.get(f"{s}Forearm") or 0)
    return {p: ("Left" if c["Left"] > c["Right"] else "Right") for p, c in total.items()}


def _work(job):
    record, prefix, trim, side = job
    from signtranslator.data.text_motion_dataset import R2Store
    from evaluate_text_to_motion import load_state
    store = R2Store(os.environ)
    key = f"{prefix}/states/{record['sign']}.npz"
    try:
        state, report = canonicalize(load_state(store, record["continuous"]["state_key"],
                                                record["continuous"]["state_sha256"]), trim=trim, side=side)
        if store.exists(key):
            digest = hashlib.sha256(store.get(key)).hexdigest()
        else:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "s.npz"
                state.save(path)
                digest = store.put(key, path.read_bytes())
        return dict(sign=record["sign"], status="converted", state_key=key, state_sha256=digest,
                    source_state_sha256=record["continuous"]["state_sha256"], **report)
    except Exception as error:        # reported per clip; the shard resumes
        return dict(sign=record["sign"], status="error", error=f"{type(error).__name__}: {error}"[:300])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shard", type=int)
    p.add_argument("--shards", type=int)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--source-key", default=SOURCE_KEY)
    p.add_argument("--source-sha256", default=SOURCE_SHA)
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--no-trim", action="store_true", help="mirror only (clips already cut to the sign, e.g. RIT)")
    p.add_argument("--dominance", choices=("clip", "signer"), default="clip",
                   help="per-clip side (canonical-v1) or the signer's side for all their clips (round 5 amendment 2)")
    p.add_argument("--corpus-id", default="asl-citizen-canonical-v1")
    p.add_argument("--manifest-key")
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json, load_manifest
    store = R2Store(os.environ)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        path.write_bytes(store.get(args.source_key) + b"\n")
        source = load_manifest(path, expected_sha256=args.source_sha256)
    prefix, trim = args.prefix, not args.no_trim
    args.manifest_key = args.manifest_key or f"{prefix}/manifest.json"
    records = sorted(source["records"], key=lambda r: r["sign"])
    if not args.assemble:
        mine = [r for k, r in enumerate(records) if k % args.shards == args.shard][:args.limit]
        report_key = f"{prefix}/reports/shard-{args.shard:03d}-of-{args.shards:03d}.json"
        done = {}
        if store.exists(report_key):
            done = {r["sign"]: r for r in json.loads(store.get(report_key))["results"] if r["status"] == "converted"}
        todo = [r for r in mine if r["sign"] not in done]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            sides = signer_sides(records) if args.dominance == "signer" else {}
            for k, result in enumerate(pool.imap_unordered(_work, [(r, prefix, trim, sides.get(r["performer_sha256"])) for r in todo], chunksize=8)):
                done[result["sign"]] = result
                if (k + 1) % 2000 == 0:
                    print(f"{k + 1}/{len(todo)}", flush=True)
        results = [done[r["sign"]] for r in mine if r["sign"] in done]
        store.put(report_key, json.dumps(dict(source_manifest_sha256=args.source_sha256, trim=trim, results=results)).encode())
        errors = sum(r["status"] != "converted" for r in results)
        print(json.dumps(dict(shard=args.shard, clips=len(mine), converted=len(results) - errors, errors=errors)))
        return 2 if errors else 0
    reports = {}
    inputs = {"source_manifest": args.source_sha256}
    for key in sorted(store.list(f"{prefix}/reports/")):
        payload = store.get(key)
        inputs[f"report:{key.rsplit('/', 1)[-1]}"] = hashlib.sha256(payload).hexdigest()
        for result in json.loads(payload)["results"]:
            if result["status"] == "converted":
                reports[result["sign"]] = result
    out = []
    for record in records:
        result = reports.get(record["sign"])
        if result is None:
            continue
        canonical = {k: result[k] for k in ("dominant", "mirrored", "no_arm", "window", "seconds", "arm_fraction",
                                             "hand_fraction", "generator_ok", "frames", "dominance") if k in result}
        out.append(dict({k: v for k, v in record.items() if k not in ("continuous", "usable")},
                        canonical=canonical,
                        continuous=dict(status="converted", state_key=result["state_key"],
                                        state_sha256=result["state_sha256"],
                                        available_channels=record["continuous"]["available_channels"],
                                        seconds=result.get("seconds", record["continuous"]["seconds"]))))
    split = dict(source["split"])
    split["counts"] = {p: sum(r["partition"] == p for r in out) for p in ("train", "val", "test")}
    manifest = build_corpus_manifest(corpus_id=args.corpus_id, inputs=inputs, records=out, split=split)
    manifest["licence"] = source.get("licence") or source.get("license")
    manifest["canonical_rule"] = dict(margin_seconds=MARGIN_SECONDS, min_seconds=MIN_SECONDS, min_arm=MIN_ARM,
                                      min_hand=MIN_HAND, trim=trim, dominance=args.dominance, mirror="signtranslator.pose.mirror.mirror_state")
    manifest["canonical_counts"] = {p: dict(clips=sum(r["partition"] == p for r in out),
                                            mirrored=sum(r["partition"] == p and r["canonical"]["mirrored"] for r in out),
                                            no_arm=sum(r["partition"] == p and r["canonical"]["no_arm"] for r in out),
                                            generator_ok=sum(r["partition"] == p and r["canonical"]["generator_ok"]
                                                             for r in out)) for p in ("train", "val", "test")}
    manifest["missing"] = len(records) - len(out)
    if store.exists(args.manifest_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.manifest_key, canonical_json(manifest))
    print(json.dumps(dict(manifest_key=args.manifest_key, manifest_sha256=digest, records=len(out),
                          missing=manifest["missing"], counts=manifest["canonical_counts"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
