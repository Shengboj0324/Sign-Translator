"""WLASL v0.3 dictionary-site clips (C-UDA 1.0) as a manifest: every extracted state with its gloss, signer and split.

Joins ``wlasl-v0.3/instances.json`` (written by ``acquire_wlasl_sites.py --pack``: video id, zip member, gloss, signer
id, split, source URL) with the extraction reports under ``derived/wlasl-v0.3/reports/``. A report's ``sample_id`` is
the member's file stem, which is the WLASL ``video_id``.

- Labels are WLASL's glosses, which are already English words (status ``wlasl_label``).
- Partitions are WLASL's own (``train`` / ``val`` / ``test``). They split instances, not signers, so validation is not
  signer-disjoint. The generator trains on ``train`` only, and nothing here is used to grade a model.
- Performers are pseudonymised (SHA-256 of ``wlasl:<signer_id>``) and keep the extraction's quality counts.
- Private; computational use only under C-UDA. Written once.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREFIX = "acquisition/2026-10-17-v1/wlasl-v0.3"
DERIVED = "acquisition/2026-10-17-v1/derived/wlasl-v0.3"
LICENCE = ("WLASL v0.3 (Li et al., WACV 2020): C-UDA 1.0, computational use only; videos remain their sites' property; "
           "private store")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest-key", default=f"{DERIVED}/manifest-v1/manifest.json")
    p.add_argument("--corpus-id", default="wlasl-sites-v1")
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json
    store = R2Store(os.environ)
    payload = store.get(f"{PREFIX}/instances.json")
    inputs = {"instances": hashlib.sha256(payload).hexdigest()}
    instances = {i["video_id"]: i for i in json.loads(payload)["instances"]}
    records, skipped = [], collections.Counter()
    for key in sorted(store.list(f"{DERIVED}/reports/")):
        report = store.get(key)
        inputs[f"report:{Path(key).name}"] = hashlib.sha256(report).hexdigest()
        for r in json.loads(report)["results"]:
            if r["status"] not in ("converted", "already_present") or "state_sha256" not in r:
                skipped[r["status"]] += 1; continue
            i = instances.get(Path(r["sample_id"]).stem)
            if i is None:
                skipped["no instance"] += 1; continue
            records.append(dict(
                sign=f"wlasl-{i['video_id']}", text=i["gloss"].lower(), partition=i["split"],
                performer_sha256=hashlib.sha256(f"wlasl:{i['signer_id']}".encode()).hexdigest(),
                gloss=dict(status="wlasl_label", label=i["gloss"], variation_id=i.get("variation_id")),
                source=dict(host=i["host"], url=i["url"], zip_key=i["zip_key"], member=i["member"]),
                quality={k: r.get(k) for k in ("frames", "pose_frames", "face_frames", "left_hand_frames",
                                               "right_hand_frames", "arm_frames")},
                continuous=dict(status="converted", state_key=r["state_key"], state_sha256=r["state_sha256"],
                                available_channels=r["available_channels"], seconds=r["seconds"])))
    counts = {p: sum(r["partition"] == p for r in records) for p in ("train", "val", "test")}
    split = dict(kind="isolated", source="WLASL v0.3 splits (instance-level, not signer-disjoint)", counts=counts,
                 glosses={p: len({r["text"] for r in records if r["partition"] == p}) for p in counts})
    manifest = build_corpus_manifest(corpus_id=args.corpus_id, inputs=inputs, records=records, split=split)
    manifest["skipped"] = dict(skipped)
    manifest["license"] = LICENCE
    if store.exists(args.manifest_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.manifest_key, canonical_json(manifest))
    print(json.dumps(dict(manifest_key=args.manifest_key, manifest_sha256=digest, records=len(records), split=split,
                          skipped=dict(skipped))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
