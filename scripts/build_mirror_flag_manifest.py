"""Round 5 amendment 4: a manifest whose records keep their stored states and carry ``"mirror": true`` for every
clip of a left-dominant signer (``canonicalize_asl_citizen.signer_sides``: forearm frames summed per signer).

``corpus_manifest.load_sample`` mirrors flagged records on load, so a consumer sees each signer right-dominant and
untrimmed: exactly the source data, with only the per-signer reflection applied. Written once.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source-key", required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--corpus-id", required=True)
    p.add_argument("--manifest-key", required=True)
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json, load_manifest
    from canonicalize_asl_citizen import signer_sides
    store = R2Store(os.environ)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        path.write_bytes(store.get(args.source_key) + b"\n")
        source = load_manifest(path, expected_sha256=args.source_sha256)
    sides = signer_sides(source["records"])
    records = [dict({k: v for k, v in r.items() if k != "usable"}, mirror=sides[r["performer_sha256"]] == "Left")
               for r in source["records"]]
    manifest = build_corpus_manifest(corpus_id=args.corpus_id, inputs={"source_manifest": args.source_sha256},
                                     records=records, split=dict(source["split"]))
    for k in ("licence", "license", "quality"):
        if k in source:
            manifest[k] = source[k]
    manifest["mirror_rule"] = "per signer: mirror iff the left forearm is tracked on more frames over the signer's clips"
    manifest["mirror_counts"] = {p: dict(clips=sum(r["partition"] == p for r in records),
                                         mirrored=sum(r["partition"] == p and r["mirror"] for r in records))
                                 for p in ("train", "val", "test")}
    if store.exists(args.manifest_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.manifest_key, canonical_json(manifest))
    print(json.dumps(dict(manifest_key=args.manifest_key, manifest_sha256=digest, counts=manifest["mirror_counts"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
