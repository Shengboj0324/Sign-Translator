"""One text -> motion manifest from several (plan item 2: How2Sign English + ASLLRP plan lines for the plan generator).

Each ``--input KEY@SHA`` is loaded under its hash; its records are kept as they are (text, state, partition), tagged
with ``corpus`` (the input's ``corpus_id``) so every evaluation can report per corpus. Sign identifiers must not
collide. Partitions are each input's own signer-disjoint ones; nothing is re-split. The licence of the result is the
union of the inputs' licences - with ASLLRP in it, research use only, private store only.

The mixed prior keeps the English condition (How2Sign) while learning the plan condition (ASLLRP): plan lines are
recognisably not English (``grammar_motion.text.is_plan_line``), so one model serves both, and its How2Sign
validation scores show whether the plan data cost the English condition anything.

**Extending a mix** (``--base KEY@SHA``, round 14: ``round14-mixed-v1`` = ``round10-mixed-v1`` + ``semlex-text-v1``): the
base mix's records are kept exactly as they are - their own ``corpus`` tags, texts, windows, partitions - and every one
is checked to be byte-identical (canonical JSON) in the result; only the ``--input`` corpora's records are added, each
tagged with its input's ``corpus_id``. Corpus names and sign identifiers must not collide with the base's. The result
keeps the base's split description (counts recomputed over the same partitions) and its other top-level fields (the
round-10 ``trim``), adds each input's signer-disjointness (``split.signer_disjoint_by_corpus``) and appends each input's
licence; ``extends`` records the base, its SHA-256 and what each input added. ``--check-val-prefix N`` checks that
the first N validation records (the trainer's fixed validation batches: 32 x batch size, in record order, under its
``--max-seconds`` filter) are unchanged from the base's, and refuses to write the mix otherwise. Corpus weights are training flags (``--corpus-weight NAME=W``), not manifest content.
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


def extend(base_sha: str, base: dict, additions: list[tuple[str, dict]], *, corpus_id: str) -> dict:
    """``base``'s records unchanged plus each addition's records (tagged with its ``corpus_id``); see the docstring."""
    if not base.get("corpus_id"):
        raise ValueError("the base manifest has no corpus_id")
    records = [{k: v for k, v in r.items() if k != "usable"} for r in base["records"]]   # usable is recomputed
    seen = {r["sign"] for r in records}
    corpora = {r.get("corpus") for r in records}
    inputs, licences, added, disjoint = {base["corpus_id"]: base_sha}, [], {}, {}
    if base.get("license"):
        licences.append(base["license"])
    for sha, m in additions:
        name = m.get("corpus_id")
        if not name or name in inputs or name in corpora:
            raise ValueError(f"input corpus_id {name!r} is missing or already in the mix")
        inputs[name] = sha
        for r in m["records"]:
            if r.get("corpus") not in (None, name):
                raise ValueError(f"{r['sign']}: tagged {r.get('corpus')!r} inside input {name}")
            if r["sign"] in seen:
                raise ValueError(f"sign {r['sign']} occurs in two inputs")
            seen.add(r["sign"])
            records.append(dict({k: v for k, v in r.items() if k != "usable"}, corpus=name))
        if m.get("license"):
            licences.append(m["license"] if m["license"].startswith(f"{name}:") else f"{name}: {m['license']}")
        disjoint[name] = (m.get("split") or {}).get("signer_disjoint")
        added[name] = dict(sha256=sha, records=len(m["records"]),
                           partitions=dict(sorted(collections.Counter(r.get("partition") for r in m["records"]).items())),
                           split=m.get("split"), provenance=m.get("provenance"), trim=m.get("trim"),
                           filters=m.get("filters"))
    split = dict(base.get("split") or {})
    parts = list((split.get("counts") or {}).keys()) or ["train", "val", "test"]
    split["counts"] = {p: sum(r.get("partition") == p for r in records) for p in parts}
    split["signer_disjoint_by_corpus"] = dict(sorted({**(split.get("signer_disjoint_by_corpus") or {}),
                                                      **disjoint}.items()))
    split["source"] = "; ".join([s for s in [split.get("source")] if s] + [
        f"added {n}: its own partitions, unchanged (signer_disjoint {str(d).lower()})" for n, d in disjoint.items()])
    manifest = build_corpus_manifest(corpus_id=corpus_id, records=records, inputs=inputs, split=split)
    for key, value in base.items():
        if key not in manifest and key not in ("license", "licence", "extends"):
            manifest[key] = value
    manifest["license"] = "; ".join(licences) if licences else None
    manifest["extends"] = dict(base=base["corpus_id"], base_sha256=base_sha, base_records=len(base["records"]),
                               added=added, rule="every base record unchanged (byte-identical canonical JSON); only "
                                                 "the added corpora's records are new")
    by = {r["sign"]: r for r in manifest["records"]}
    for r in base["records"]:
        if canonical_json(by[r["sign"]]) != canonical_json(r):
            raise ValueError(f"{r['sign']}: a base record changed")
    canonical_json(manifest)
    return manifest


def val_prefix_unchanged(base: dict, out: dict, n: int, max_seconds: float | None = 20.) -> bool:
    """Whether the first ``n`` validation records the trainer would read are the same in both manifests:
    ``TextMotionDataset``'s filter (text, usable continuous channels, ``continuous.seconds <= max_seconds``; the
    trainer's ``--max-seconds`` default is 20), in record order."""
    def first(m):
        return [r["sign"] for r in m["records"] if r.get("partition") == "val" and r.get("text")
                and (r.get("usable") or {}).get("continuous")
                and (max_seconds is None or (r.get("continuous") or {}).get("seconds", 0.) <= max_seconds)][:n]
    return first(base) == first(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2", help="r2, or dir:<path>")
    parser.add_argument("--input", action="append", required=True, help="KEY@SHA of an input manifest (repeatable)")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--base", help="KEY@SHA of a mix to extend: its records kept byte-identical, the inputs added")
    parser.add_argument("--check-val-prefix", type=int, action="append", default=[],
                        help="with --base: report whether the first N validation records are unchanged (repeatable)")
    parser.add_argument("--max-seconds", type=float, default=20., help="the trainer's --max-seconds, for that check")
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    if store.exists(args.manifest_key):              # fail before reading the inputs; checked again before the put
        raise SystemExit("refusing to overwrite an existing manifest")

    def fetch(spec):
        key, sha = parse_input(spec)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "m.json"; p.write_bytes(store.get(key) + b"\n")
            return sha, load_manifest(p, expected_sha256=sha)
    manifests = [fetch(spec) for spec in args.input]
    extra = {}
    if args.base:
        base_sha, base = fetch(args.base)
        manifest = extend(base_sha, base, manifests, corpus_id=args.corpus_id)
        extra = dict(base=base["corpus_id"], base_sha256=base_sha, base_records=len(base["records"]),
                     base_records_identical=True,
                     val_prefix_unchanged={str(n): val_prefix_unchanged(base, manifest, n, args.max_seconds)
                                           for n in args.check_val_prefix},
                     by_corpus={f"{c}/{p}": n for (c, p), n in sorted(collections.Counter(
                         (r.get("corpus"), r.get("partition")) for r in manifest["records"]).items(), key=str)})
        changed = [n for n, same in extra["val_prefix_unchanged"].items() if not same]
        if changed:                                  # before the write-once put: the fixed validation batches moved
            raise SystemExit(f"the first {changed} validation records differ from the base's: refusing to write")
    else:
        manifest = mix(manifests, corpus_id=args.corpus_id)
    if store.exists(args.manifest_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.manifest_key, canonical_json(manifest))
    print(json.dumps(dict(manifest_key=args.manifest_key, manifest_sha256=digest, records=len(manifest["records"]),
                          split=manifest["split"]["counts"], **extra)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
