"""Round 6 generator filter: right-dominant signers only (no mirroring) and per-clip right-arm/hand tracking.

From a corpus manifest whose records carry extraction ``quality`` counts, keep a record when:
1. its signer's side, from ``--sides-key`` (``signer_dominance_motion.py`` output, or a constant with ``--all-right``),
   is ``Right``. Left-dominant signers are left out, not mirrored: reflecting a stored state is not re-estimation;
   see the round-5 v3c audit;
2. its right forearm is tracked on at least ``--min-arm`` of its frames and its right hand on at least ``--min-hand``
   (round 6: 0.5 and 0.5).

Writes the filtered manifest (``corpus_id`` given) with the rule and counts recorded. Written once.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def keep(record: dict, side: str, min_arm: float, min_hand: float) -> str | None:
    """None if kept, else the reason."""
    if side != "Right":
        return "left_dominant_signer"
    q = record.get("quality") or {}
    frames = max(q.get("frames") or 0, 1)
    if ((q.get("arm_frames") or {}).get("RightForearm") or 0) / frames < min_arm:
        return "right_arm_untracked"
    if (q.get("right_hand_frames") or 0) / frames < min_hand:
        return "right_hand_untracked"
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest-key", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--sides-key")
    p.add_argument("--all-right", action="store_true", help="every signer is right-dominant (measured beforehand)")
    p.add_argument("--min-arm", type=float, default=.5)
    p.add_argument("--min-hand", type=float, default=.5)
    p.add_argument("--corpus-id", required=True)
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json, load_manifest
    store = R2Store(os.environ)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        path.write_bytes(store.get(args.manifest_key) + b"\n")
        source = load_manifest(path, expected_sha256=args.manifest_sha256)
    sides = {} if args.all_right else {k: (v["side"] if isinstance(v, dict) else v)
                                        for k, v in json.loads(store.get(args.sides_key)).items()}
    kept, why = [], collections.Counter()
    for r in source["records"]:
        reason = keep(r, "Right" if args.all_right else sides.get(r["performer_sha256"], "unknown"), args.min_arm,
                      args.min_hand)
        if reason:
            why[f"{r['partition']}:{reason}"] += 1
        else:
            kept.append({k: v for k, v in r.items() if k != "usable"})
            why[f"{r['partition']}:kept"] += 1
    inputs = {"source_manifest": args.manifest_sha256}
    if args.sides_key:
        import hashlib
        inputs["signer_sides"] = hashlib.sha256(store.get(args.sides_key)).hexdigest()
    split = dict(source["split"], counts={p: sum(r["partition"] == p for r in kept) for p in ("train", "val", "test")})
    out = build_corpus_manifest(corpus_id=args.corpus_id, inputs=inputs, records=kept, split=split)
    out["license"] = source.get("license") or source.get("licence")
    out["filter"] = dict(rule="right-dominant signers only (no mirroring); right forearm >= min_arm and right hand >= "
                              "min_hand of frames", min_arm=args.min_arm, min_hand=args.min_hand,
                         sides=args.sides_key or "all right", counts=dict(sorted(why.items())))
    if store.exists(args.output_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.output_key, canonical_json(out))
    print(json.dumps(dict(manifest_key=args.output_key, manifest_sha256=digest, kept=len(kept), why=dict(sorted(why.items())))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
