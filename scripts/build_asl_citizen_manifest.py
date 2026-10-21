"""Build the ASL Citizen isolated-sign manifest from extraction reports and the publisher's splits.

Every converted state is joined with its row in ``ASL_Citizen/splits/{train,val,test}.csv``
(read from the verified zip by ranged reads): the gloss label, its ASL-LEX code and the
participant. Partitions are the publisher's splits, which are signer-disjoint; this is
checked, and a participant found in two splits stops the build. Participants appear only
as SHA-256 pseudonyms. Gloss labels are the publisher's (status ``publisher_label``): they
name the sign each participant was prompted to perform, not a reviewed annotation.

The quality block counts what the isolated-sign uses need: clips with arm rotations (hand
location), with any hand, and their frame fractions, per split. Research use only under the
Microsoft Research Licence Terms; the manifest stays private. Written once.
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
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json  # noqa: E402

SPLITS = ("train", "val", "test")


def read_splits(archive: zipfile.ZipFile) -> tuple[dict, dict]:
    """sample id -> row fields, and the split CSVs' hashes."""
    rows, hashes = {}, {}
    for split in SPLITS:
        payload = archive.read(f"ASL_Citizen/splits/{split}.csv")
        hashes[f"split:{split}.csv"] = hashlib.sha256(payload).hexdigest()
        for row in csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))):
            sample = Path(row["Video file"].strip()).stem
            if sample in rows:
                raise ValueError(f"{sample} is listed twice in the splits")
            rows[sample] = dict(split=split, participant=row["Participant ID"].strip(), gloss=row["Gloss"].strip(),
                                asl_lex_code=(row.get("ASL-LEX Code") or "").strip() or None)
    by_participant = collections.defaultdict(set)
    for row in rows.values():
        by_participant[row["participant"]].add(row["split"])
    shared = sorted(p for p, s in by_participant.items() if len(s) > 1)
    if shared:
        raise ValueError(f"participants in more than one split: {len(shared)}")
    return rows, hashes


def quality(records: list[dict], min_fraction: float = .2) -> dict:
    """Per split: clips, glosses, and the share of clips whose arms or hands are seen."""
    out = {}
    for split in SPLITS:
        part = [r for r in records if r["partition"] == split]
        q = [r["quality"] for r in part]
        arms = [max(x["arm_frames"].values()) / x["frames"] for x in q if x["frames"]]
        hands = [max(x["left_hand_frames"], x["right_hand_frames"]) / x["frames"] for x in q if x["frames"]]
        out[split] = dict(clips=len(part), glosses=len({r["gloss"]["label"] for r in part}),
                          participants=len({r["performer_sha256"] for r in part}),
                          arm_clips=sum(a >= min_fraction for a in arms), hand_clips=sum(h >= min_fraction for h in hands),
                          median_arm_fraction=sorted(arms)[len(arms) // 2] if arms else None,
                          median_hand_fraction=sorted(hands)[len(hands) // 2] if hands else None)
    out["min_fraction"] = min_fraction
    return out


def open_zip(store, key: str) -> zipfile.ZipFile:
    """The zip by ranged reads from R2, or directly from a directory store."""
    if isinstance(store, DirectoryStore):
        return zipfile.ZipFile(store.root / key)
    from extract_video_states import StoreRangeFile
    return zipfile.ZipFile(io.BufferedReader(StoreRangeFile(store.s3, store.bucket, key), 1 << 20))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--zip-key", required=True)
    parser.add_argument("--zip-sha256", required=True)
    parser.add_argument("--reports-prefix", required=True)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--corpus-id", default="asl-citizen-video-estimate-v1")
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    provenance = json.loads(store.get(args.zip_key + ".provenance.json"))
    if provenance.get("sha256") != args.zip_sha256 or not provenance.get("verified_by_r2_reread"):
        raise SystemExit("the zip is not the verified acquisition named by --zip-sha256")
    rows, inputs = read_splits(open_zip(store, args.zip_key))
    inputs["zip"] = args.zip_sha256
    records, statuses, unlisted = [], collections.Counter(), 0
    for key in sorted(store.list(args.reports_prefix)):
        payload = store.get(key)
        report = json.loads(payload)
        if not report.get("upright_chest"):
            raise SystemExit(f"{key}: extracted without the upright-chest prior")
        inputs[f"report:{key.rsplit('/', 1)[-1]}"] = hashlib.sha256(payload).hexdigest()
        for result in report["results"]:
            statuses[result["status"]] += 1
            if result["status"] != "converted" or "state_sha256" not in result:
                continue
            row = rows.get(result["sample_id"])
            if row is None:
                unlisted += 1
                continue
            records.append(dict(
                sign=result["sample_id"], text=None, partition=row["split"],
                performer_sha256=hashlib.sha256(f"asl-citizen:{row['participant']}".encode()).hexdigest(),
                gloss=dict(status="publisher_label", label=row["gloss"], asl_lex_code=row["asl_lex_code"],
                           source="ASL_Citizen/splits"),
                quality={k: result[k] for k in ("frames", "pose_frames", "face_frames", "left_hand_frames",
                                                "right_hand_frames", "arm_frames")},
                continuous=dict(status="converted", state_key=result["state_key"], state_sha256=result["state_sha256"],
                                available_channels=result["available_channels"], seconds=result["seconds"])))
    if len({r["sign"] for r in records}) != len(records):
        raise SystemExit("a clip was converted in two shards")
    split = dict(kind="continuous", source="publisher splits (signer-disjoint, checked)",
                 counts={p: sum(r["partition"] == p for r in records) for p in SPLITS})
    manifest = build_corpus_manifest(corpus_id=args.corpus_id, inputs=inputs, records=records, split=split)
    manifest["quality"] = quality(records)
    manifest["extraction"] = dict(statuses=dict(statuses), listed=len(rows), unlisted_converted=unlisted,
                                  missing=len(rows) - len(records))
    manifest["licence"] = provenance.get("license")
    payload = canonical_json(manifest)
    if store.exists(args.manifest_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.manifest_key, payload)
    print(json.dumps(dict(manifest_key=args.manifest_key, manifest_sha256=digest, records=len(records),
                          extraction=manifest["extraction"], quality=manifest["quality"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
