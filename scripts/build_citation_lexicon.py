"""Citation-form lexicon for the stitching baseline (``generation.citation_stitch``; plan items 2 and 3).

One recorded clip per lexicon key, chosen from the **training partitions only** (ASLLVD validation/test signers and ASL
Citizen validation/test signers stay unread, so no evaluation clip is ever stitched in):

- ASLLVD (DAI 2, studio, Sign Bank glosses - the planner's vocabulary): key = the Sign Bank gloss. The DAI sign clips
  carry 0.833 s of context either side of the annotated sign (clip length = span / 60 + 1.667 s over 8,730 clips), so
  the entry's span is the annotated sign plus ``--margin`` seconds.
- ASL Citizen (webcam): key ``citizen:<label>`` for every label (the stitcher's pointing / possessive signs), and the
  Sign Bank gloss for labels whose map is ``exact``, unless ASLLVD already has that gloss. The span is the stretch on
  which a hand is tracked (rest before and after removed) - computed when the clip is stitched (``span: hands``).

Among a key's clips the one chosen is right-hand dominant if any is (the planner's surface tier is the dominant hand),
then has the highest dominant-hand coverage, then the duration closest to the key's median. Choices use the extraction
reports' frame counts, so no state is read here. Research-only data: the lexicon goes to a private prefix.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import canonical_json, load_manifest  # noqa: E402

ASLLVD_CONTEXT = 50 / 60          # seconds of context either side of an ASLLVD sign clip


def manifest_of(store, key, sha):
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(key) + b"\n")
        return load_manifest(p, expected_sha256=sha)


def coverage(frames: dict) -> tuple[bool, float]:
    """(right-hand dominant, dominant-hand coverage) from a clip's frame counts."""
    n = max(frames["frames"], 1)
    right, left = frames["right_hand_frames"] / n, frames["left_hand_frames"] / n
    return right >= left, max(right, left)


def choose(candidates: list[dict]) -> dict:
    median = statistics.median(c["seconds"] for c in candidates)
    return max(candidates, key=lambda c: (c["right_dominant"], round(c["coverage"], 2), -abs(c["seconds"] - median),
                                          c["state_key"]))


def build(asllvd_records, asllvd_frames, citizen_records, mapping, *, margin: float) -> dict:
    by_key = collections.defaultdict(list)
    for r in asllvd_records:
        if r.get("partition") != "train":
            continue
        f = asllvd_frames.get(r["continuous"]["state_key"])
        if f is None:
            continue
        dominant, cov = coverage(f)
        span = (r["source"]["frames"][1] - r["source"]["frames"][0]) / 60.
        by_key[r["gloss"]["label"]].append(dict(
            source="asllvd", state_key=r["continuous"]["state_key"], state_sha256=r["continuous"]["state_sha256"],
            seconds=r["continuous"]["seconds"], right_dominant=dominant, coverage=cov,
            span=[max(ASLLVD_CONTEXT - margin, 0.), min(ASLLVD_CONTEXT + span + margin, r["continuous"]["seconds"])],
            performer_sha256=r.get("performer_sha256")))
    asllvd_keys = set(by_key)
    exact = collections.defaultdict(set)
    for label, m in mapping.items():
        if m["method"] == "exact":
            exact[label].update(m["candidates"])
    for r in citizen_records:
        if r.get("partition") != "train" or not r.get("quality"):
            continue
        dominant, cov = coverage(r["quality"])
        entry = dict(source="asl-citizen", state_key=r["continuous"]["state_key"],
                     state_sha256=r["continuous"]["state_sha256"], seconds=r["continuous"]["seconds"],
                     right_dominant=dominant, coverage=cov, span="hands", performer_sha256=r.get("performer_sha256"))
        label = r["gloss"]["label"]
        by_key[f"citizen:{label}"].append(entry)
        for signbank in exact.get(label, ()):
            if signbank not in asllvd_keys:
                by_key[signbank].append(entry)
    return {k: {f: v for f, v in choose(c).items() if f not in ("right_dominant",)} | dict(candidates=len(c))
            for k, c in sorted(by_key.items())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2", help="r2, or dir:<path>")
    parser.add_argument("--asllvd-manifest-key", default="acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/asllvd-sign-manifest-v1/manifest.json")
    parser.add_argument("--asllvd-manifest-sha256", required=True)
    parser.add_argument("--asllvd-reports-prefix", default="acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/asllvd-sign-videos/reports/")
    parser.add_argument("--citizen-manifest-key", default="acquisition/2026-10-17-v1/derived/asl-citizen-v1/manifest-v1/manifest.json")
    parser.add_argument("--citizen-manifest-sha256", required=True)
    parser.add_argument("--gloss-map-key", default="acquisition/2026-10-17-v1/derived/signbank-v1/asl-citizen-gloss-map.json")
    parser.add_argument("--margin", type=float, default=.05)
    parser.add_argument("--output-key", default="acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/citation-lexicon-v1/lexicon.json")
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    asllvd = manifest_of(store, args.asllvd_manifest_key, args.asllvd_manifest_sha256)
    citizen = manifest_of(store, args.citizen_manifest_key, args.citizen_manifest_sha256)
    frames = {}
    for key in store.list(args.asllvd_reports_prefix):
        for r in json.loads(store.get(key))["results"]:
            if r.get("status") == "converted":
                frames[r["state_key"]] = r
    map_payload = store.get(args.gloss_map_key)
    entries = build(asllvd["records"], frames, citizen["records"], json.loads(map_payload)["mapping"],
                    margin=args.margin)
    lexicon = dict(schema_version=1, entries=entries,
                   inputs=dict(asllvd_manifest=args.asllvd_manifest_sha256, citizen_manifest=args.citizen_manifest_sha256,
                               gloss_map=hashlib.sha256(map_payload).hexdigest()),
                   rule="training partitions only; right-hand dominant, then coverage, then median duration",
                   license="ASLLRP Sign Bank Data Terms of Use (research only) and ASL Citizen licence; private")
    if store.exists(args.output_key):
        raise SystemExit("refusing to overwrite an existing lexicon")
    digest = store.put(args.output_key, canonical_json(lexicon))
    sources = collections.Counter(e["source"] for e in entries.values())
    print(json.dumps(dict(output_key=args.output_key, sha256=digest, entries=len(entries), sources=dict(sources))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
