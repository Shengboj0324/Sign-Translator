"""ASLLRP / RIT plan -> motion manifest: each utterance's motion state conditioned on its plan line (plan item 2).

Joins the gloss -> motion manifest (``build_asllrp_gloss_manifest.py``: records with extracted states, signer-disjoint
partitions) with the English -> gloss pairs (``build_asllrp_english_gloss.py``: surface signs, loci and grammatical
markers) by (source collection, utterance number). A record's ``text`` becomes the plan line
(``signtranslator.planning.plan_text.plan_to_text``: base glosses with locus slots and bracketed markers), the
condition the byte-encoded generators train on; the English translation and the gloss record are kept beside it.
Partitions are those of the gloss manifest (val Jonathan, test Ben). Research use only; written once.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json, load_manifest  # noqa: E402
from signtranslator.planning.asllrp_gloss import plan_target  # noqa: E402
from signtranslator.planning.plan_text import plan_to_text  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2", help="r2, or dir:<path>")
    parser.add_argument("--gloss-manifest-key", default="acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/gloss-manifest-v1/manifest.json")
    parser.add_argument("--gloss-manifest-sha256", required=True)
    parser.add_argument("--pairs-key", default="acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/english-gloss-v1/pairs.json")
    parser.add_argument("--pairs-sha256", required=True)
    parser.add_argument("--manifest-key", default="acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/plan-manifest-v1/manifest.json")
    parser.add_argument("--corpus-id", default="asllrp-dai2-plan-v1")
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.gloss_manifest_key) + b"\n")
        gloss = load_manifest(p, expected_sha256=args.gloss_manifest_sha256)
    pairs_payload = store.get(args.pairs_key)
    if hashlib.sha256(pairs_payload).hexdigest() != args.pairs_sha256:
        raise SystemExit(f"{args.pairs_key} does not match --pairs-sha256")
    pairs = {(p["source"], p["utterance"]): p for p in json.loads(pairs_payload)["pairs"]}
    records, skipped = [], collections.Counter()
    for r in gloss["records"]:
        pair = pairs.get((r["source"]["source"], r["source"]["utterance"]))
        if pair is None:
            skipped["no plan pair"] += 1; continue
        if pair["partition"] != r["partition"]:
            raise SystemExit(f"partition mismatch for {r['sign']}")
        t = plan_target(pair)
        line = plan_to_text(list(t.bases), [tuple(x) for x in t.loci], [list(m) for m in t.markers])
        records.append(dict(r, text=line, english=pair["english"],
                            plan=dict(glosses=list(t.bases), loci=[list(x) for x in t.loci],
                                      markers=[list(m) for m in t.markers])))
    counts = {p: sum(r["partition"] == p for r in records) for p in ("train", "val", "test")}
    manifest = build_corpus_manifest(
        corpus_id=args.corpus_id, records=records,
        inputs={"gloss_manifest": args.gloss_manifest_sha256, "pairs": args.pairs_sha256},
        split=dict(kind="continuous", source="signer-disjoint, fixed: val Jonathan, test Ben", counts=counts))
    manifest["skipped"] = dict(skipped)
    manifest["license"] = "ASLLRP Sign Bank Data Terms of Use: research use only, no redistribution"
    if store.exists(args.manifest_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.manifest_key, canonical_json(manifest))
    print(json.dumps(dict(manifest_key=args.manifest_key, manifest_sha256=digest, records=len(records),
                          partitions=counts, skipped=dict(skipped), example=records[0]["text"] if records else None)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
