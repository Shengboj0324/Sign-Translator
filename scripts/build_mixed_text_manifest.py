"""One text -> motion manifest from several (plan item 2: How2Sign English + ASLLRP plan lines for the plan generator).

Each ``--input KEY@SHA`` is loaded under its hash; its records are kept as they are (text, state, partition), tagged
with ``corpus`` (the input's ``corpus_id``) so every evaluation can report per corpus. Sign identifiers must not
collide. Partitions are each input's own signer-disjoint ones; nothing is re-split. The licence of the result is the
union of the inputs' licences - with ASLLRP in it, research use only, private store only.

The mixed prior keeps the English condition (How2Sign) while learning the plan condition (ASLLRP): plan lines are
recognisably not English (``grammar_motion.text.is_plan_line``), so one model serves both, and its How2Sign
validation scores show whether the plan data cost the English condition anything.
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
from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json, load_manifest  # noqa: E402


def parse_input(spec: str) -> tuple[str, str]:
    key, _, sha = spec.rpartition("@")
    if not key or len(sha) != 64:
        raise ValueError(f"--input {spec!r}: expected KEY@<64-hex SHA-256>")
    return key, sha


def mix(manifests: list[tuple[str, dict]], *, corpus_id: str) -> dict:
    records, inputs, licences, counts = [], {}, [], collections.Counter()
    seen = set()
    for sha, m in manifests:
        name = m.get("corpus_id") or sha[:12]
        if name in inputs:
            raise ValueError(f"two inputs share corpus_id {name}")
        inputs[name] = sha
        if m.get("license"):
            licences.append(f"{name}: {m['license']}")
        for r in m["records"]:
            if r["sign"] in seen:
                raise ValueError(f"sign {r['sign']} occurs in two inputs")
            seen.add(r["sign"])
            records.append(dict(r, corpus=name))
            counts[(name, r.get("partition"))] += 1
    split = dict(kind="continuous", source="each input's own signer-disjoint partitions, unchanged",
                 counts={f"{c}/{p}": n for (c, p), n in sorted(counts.items(), key=lambda t: (t[0][0], str(t[0][1])))})
    manifest = build_corpus_manifest(corpus_id=corpus_id, records=records, inputs=inputs, split=split)
    manifest["license"] = "; ".join(licences) if licences else None
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2", help="r2, or dir:<path>")
    parser.add_argument("--input", action="append", required=True, help="KEY@SHA of an input manifest (repeatable)")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--manifest-key", required=True)
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    manifests = []
    for spec in args.input:
        key, sha = parse_input(spec)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "m.json"; p.write_bytes(store.get(key) + b"\n")
            manifests.append((sha, load_manifest(p, expected_sha256=sha)))
    manifest = mix(manifests, corpus_id=args.corpus_id)
    if store.exists(args.manifest_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.manifest_key, canonical_json(manifest))
    print(json.dumps(dict(manifest_key=args.manifest_key, manifest_sha256=digest, records=len(manifest["records"]),
                          split=manifest["split"]["counts"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
