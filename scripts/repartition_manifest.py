"""Re-partition a How2Sign text manifest by signer component for multi-signer validation (round 3).

The full-corpus round validated on one unseen signer (component c03, signer 3) and was tested once on another (c00,
signer 9). What was tuned on signer 3 did not transfer: R6 found head speed 1.83× and nod rate 19.8× on signer 9. So
the round-3 manifest:
- validates on **several** unseen signers (``--val-components``);
- keeps the test component untouched (``--test-components``);
- leaves ``heldout_official`` as it is;
- trains on the remaining components.

Component membership comes from the frozen fold plan (``sample_components``: sample id → component; ``components``
list each component's signers). Every record keeps its fields; only ``partition`` changes, and a ``signer_component``
field is added so evaluations can be reported per signer. A record with no component is refused.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def repartition(manifest: dict, plan: dict, *, val: set, test: set, keep=("heldout_official",)) -> list[dict]:
    component_of = plan["sample_components"]
    records = []
    for r in manifest["records"]:
        if r.get("partition") in keep:
            records.append(dict(r)); continue
        c = component_of.get(r["sign"])
        if c is None:
            raise ValueError(f"{r['sign']}: no component in the fold plan")
        partition = "test" if c in test else "val" if c in val else "train"
        records.append(dict(r, partition=partition, signer_component=c))
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--fold-plan", type=Path, required=True)
    parser.add_argument("--fold-plan-sha256", required=True)
    parser.add_argument("--val-components", required=True, help="comma-separated component ids")
    parser.add_argument("--test-components", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output-key", required=True)
    args = parser.parse_args(argv)
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json, load_manifest
    from train_text_to_motion import make_store
    store = make_store(args.store)
    plan = json.loads(args.fold_plan.read_bytes())
    if hashlib.sha256(canonical_json(plan)).hexdigest() != args.fold_plan_sha256:   # the project's convention:
        raise SystemExit("the fold plan differs from --fold-plan-sha256")            # SHA of the canonical JSON
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    val, test = set(args.val_components.split(",")), set(args.test_components.split(","))
    if val & test:
        raise SystemExit("a component cannot be both validation and test")
    records = repartition(manifest, plan, val=val, test=test)
    counts = dict(collections.Counter(r["partition"] for r in records))
    signers = {c["component_id"]: c["signers"] for c in plan["components"]}
    split = dict(kind="continuous", source="signer components of the frozen fold plan (multi-signer validation)",
                 val_components={c: signers.get(c) for c in sorted(val)},
                 test_components={c: signers.get(c) for c in sorted(test)},
                 fold_plan_sha256=args.fold_plan_sha256, parent_manifest_sha256=args.manifest_sha256, counts=counts)
    out = build_corpus_manifest(corpus_id=args.corpus_id, records=records,
                                inputs={args.manifest_key: args.manifest_sha256, "fold-plan": args.fold_plan_sha256},
                                split=split)
    out["license"] = manifest.get("license")
    if store.exists(args.output_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    body = canonical_json(out)
    store.put(args.output_key, body)
    print(json.dumps(dict(counts=counts, sha256=hashlib.sha256(body).hexdigest(),
                          per_component=dict(collections.Counter(r.get("signer_component") for r in records
                                                                 if r["partition"] == "val")))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
