"""RIT citation-form sign manifest: every extracted RIT sign clip with its Sign Bank gloss and participant (DAI 2).

Joins the extraction reports of the RIT sign-video zips (``runpod/fleet/extract_rit_citation.sh``) with the RIT
annotation CSV through the CSV's own ``Sign clip video filename``. Labels are the annotators' main-entry glosses
(status ``asllrp_annotation``); entry/variant label, sign type and handshapes are kept beside them.

Partitions are signer-disjoint and fixed here (not tuned): validation = P15, test = P16, training = the other
``P<n>`` participants. ``P018`` is read as ``P18``, since the participant numbering has no leading zeros elsewhere.
Clips whose source video names no participant (``RIT-sb1-*``, ``take*``; 398 tokens) are left out: their signer
could be the validation or test signer. Research use only (ASLLRP Sign Bank Data Terms of Use); the manifest stays
private. Written once.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json  # noqa: E402

VAL, TEST = "P15", "P16"
_PARTICIPANT = re.compile(r"^P0*(\d+)_")
HANDSHAPES = ("Dominant start handshape", "Non-dominant start handshape", "Dominant end handshape",
              "Non-dominant end handshape")


def participant(full_video: str) -> str | None:
    m = _PARTICIPANT.match(full_video or "")
    return f"P{int(m[1]):02d}" if m else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2", help="r2, or dir:<path>")
    parser.add_argument("--annotations-key",
                        default="acquisition/2026-10-17-v1/asllrp-dai2-v1/rit-citation/rit_signs_2024_06_27.csv")
    parser.add_argument("--reports-prefix",
                        default="acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/rit-citation-sign-videos/reports/")
    parser.add_argument("--manifest-key",
                        default="acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/rit-sign-manifest-v1/manifest.json")
    parser.add_argument("--corpus-id", default="rit-dai2-signs-v1")
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    payload = store.get(args.annotations_key)
    inputs = {"annotations": hashlib.sha256(payload).hexdigest()}
    rows, duplicates = {}, 0
    for row in csv.DictReader(io.StringIO(payload.decode("utf-8-sig", errors="replace"))):
        clip = Path(row.get("Sign clip video filename", "").strip()).stem
        if clip in rows:
            duplicates += 1          # one clip annotated twice: the first row is kept
            continue
        if clip:
            rows[clip] = row
    records, skipped = [], collections.Counter()
    skipped["duplicate annotation rows"] = duplicates
    for key in sorted(store.list(args.reports_prefix)):
        report = store.get(key)
        inputs[f"report:{Path(key).name}"] = hashlib.sha256(report).hexdigest()
        for result in json.loads(report)["results"]:
            if result["status"] not in ("converted", "already_present") or "state_sha256" not in result:
                skipped[result["status"]] += 1; continue
            row = rows.get(Path(result["sample_id"]).stem)
            if row is None:
                skipped["no annotation row"] += 1; continue
            signer = participant(row.get("full video filename", ""))
            if signer is None:
                skipped["no participant"] += 1; continue
            gloss = row["main entry gloss label"].strip()
            if not gloss:
                skipped["no gloss"] += 1; continue
            partition = "val" if signer == VAL else "test" if signer == TEST else "train"
            records.append(dict(
                sign=f"rit-{result['sample_id']}", text=gloss, partition=partition,
                performer_sha256=hashlib.sha256(f"rit-dai2:{signer}".encode()).hexdigest(),
                gloss=dict(status="asllrp_annotation", label=gloss, variant=row.get("entry/variant gloss label", "").strip(),
                           sign_type=row.get("sign type", "").strip(),
                           handshapes={h: row[h].strip() for h in HANDSHAPES if row.get(h, "").strip()}),
                source=dict(full_video=row["full video filename"], clip=row["Sign clip video filename"]),
                continuous=dict(status="converted", state_key=result["state_key"], state_sha256=result["state_sha256"],
                                available_channels=result["available_channels"], seconds=result["seconds"])))
    counts = {p: sum(r["partition"] == p for r in records) for p in ("train", "val", "test")}
    glosses = {p: len({r["gloss"]["label"] for r in records if r["partition"] == p}) for p in ("train", "val", "test")}
    split = dict(kind="isolated", source=f"signer-disjoint, fixed: val {VAL}, test {TEST}", counts=counts, glosses=glosses)
    manifest = build_corpus_manifest(corpus_id=args.corpus_id, inputs=inputs, records=records, split=split)
    manifest["skipped"] = dict(skipped)
    manifest["license"] = "ASLLRP Sign Bank Data Terms of Use: research use only, no redistribution"
    if store.exists(args.manifest_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.manifest_key, canonical_json(manifest))
    print(json.dumps(dict(manifest_key=args.manifest_key, manifest_sha256=digest, records=len(records),
                          partitions=counts, glosses=glosses, skipped=dict(skipped))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
