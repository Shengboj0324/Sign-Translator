"""Build the sentence-aligned training manifest from extraction reports.

Joins every converted video-estimated state with its How2Sign realigned English
sentence and assigns its partition from the frozen signer-disjoint fold plan
(B11): a clip's partition is its component's role in the chosen fold, so no signer
appears in two partitions. Clips missing from the plan (official val/test splits)
are kept out of train and validation and labelled ``heldout_official`` with their
signer overlap unverified. The manifest is canonical and written once.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json  # noqa: E402
from signtranslator.data_engineering.fold_plan import FoldPlan  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--reports-prefix", required=True)
    parser.add_argument("--sentences", nargs="+", required=True, help="store keys of realigned CSVs")
    parser.add_argument("--fold-plan", type=Path, required=True)
    parser.add_argument("--fold-plan-sha256", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--corpus-id", default="how2sign-video-estimate-text-v1",
                        help="how2sign-two-view-text-v1 when built from the two-view reports")
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    plan = FoldPlan.load(args.fold_plan, expected_sha256=args.fold_plan_sha256)
    sentences, inputs = {}, {"fold_plan": args.fold_plan_sha256}
    for key in args.sentences:
        payload = store.get(key)
        inputs[f"sentences:{key.rsplit('/', 1)[-1]}"] = hashlib.sha256(payload).hexdigest()
        for row in csv.DictReader(io.StringIO(payload.decode("utf-8")), delimiter="\t"):
            sentences[row["SENTENCE_NAME"]] = row["SENTENCE"]
    records = []
    for key in store.list(args.reports_prefix):
        payload = store.get(key)
        inputs[f"report:{key.rsplit('/', 1)[-1]}"] = hashlib.sha256(payload).hexdigest()
        for result in json.loads(payload)["results"]:
            if result["status"] not in ("converted", "already_present") or "state_sha256" not in result:
                continue
            sample = result["sample_id"]
            try:
                partition = plan.partition(args.fold, sample)
            except KeyError:
                partition = "heldout_official"
            except ValueError:
                partition = "quarantined"
            records.append(dict(sign=sample, text=sentences.get(sample), partition=partition, performer_sha256=None,
                                continuous=dict(status="converted", state_key=result["state_key"],
                                                state_sha256=result["state_sha256"],
                                                available_channels=result["available_channels"],
                                                seconds=result["seconds"])))
    split = dict(kind="continuous", fold_plan_sha256=args.fold_plan_sha256, fold=args.fold,
                 counts={p: sum(r["partition"] == p for r in records) for p in {r["partition"] for r in records}},
                 note="partitions inherit the frozen signer-disjoint fold plan")
    manifest = build_corpus_manifest(corpus_id=args.corpus_id, inputs=inputs, records=records,
                                     split=split)
    payload = canonical_json(manifest)
    if store.exists(args.manifest_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.manifest_key, payload)
    print(json.dumps(dict(manifest_key=args.manifest_key, manifest_sha256=digest, records=len(records),
                          partitions=split["counts"], missing_text=sum(not r["text"] for r in records))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
