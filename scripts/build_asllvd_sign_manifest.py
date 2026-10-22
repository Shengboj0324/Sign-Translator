"""ASLLVD citation-form sign manifest: every extracted sign clip with its Sign Bank gloss and signer (DAI 2).

Joins the extraction reports of the ASLLVD sign-video zips (``extract_video_states.py``) with the ASLLVD annotation
CSV. A clip file ``<Signer>-session-<session>-scene-<n>-<start>-<end>-camera<c>`` is the sign's own frame range in
the full video ``<session>_scene<n>-camera<c>.mov``, so it is matched to the row with that full video file and
those sign start/end frames. Labels are the annotators' main-entry glosses (status ``asllrp_annotation``), with the
entry/variant label, sign type and handshapes kept beside them.

Partitions are signer-disjoint and fixed here (not tuned): validation = Naomi, test = Tyler, training = Brady, Liz,
Dana, Lana. Clips without a matching annotation row are left out and counted. Research use only (ASLLRP Sign Bank
Data Terms of Use); the manifest stays private. Written once.
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

PARTITION = {"Naomi": "val", "Tyler": "test", "Brady": "train", "Liz": "train", "Dana": "train", "Lana": "train"}
_CLIP = re.compile(r"^(?P<signer>[A-Za-z]+)-session-(?P<session>.+?)-scene-(?P<scene>\d+)-(?P<start>\d+)-(?P<end>\d+)"
                   r"-camera(?P<camera>\d+)$")
_FULL = re.compile(r"^(?P<session>.+?)_scene(?P<scene>\d+)-camera(?P<camera>\d+)\.mov$")
START, END = "start frame of the sign (relative to full videos)", "end frame of the sign (relative to full videos)"


def clip_key(sample: str):
    """(signer, (session, scene, start, end, camera)) of a sign clip file stem, or None."""
    m = _CLIP.match(sample)
    return (m["signer"], (m["session"], int(m["scene"]), int(m["start"]), int(m["end"]), int(m["camera"]))) if m else None


def row_key(row: dict):
    m = _FULL.match(row.get("full video file", ""))
    if not m or not row.get(START, "").strip().isdigit() or not row.get(END, "").strip().isdigit():
        return None
    return (m["session"], int(m["scene"]), int(row[START]), int(row[END]), int(m["camera"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2", help="r2, or dir:<path>")
    parser.add_argument("--annotations-key", default="acquisition/2026-10-17-v1/asllrp-dai2-v1/asllvd-citation/asllvd_signs_2024_06_27.csv")
    parser.add_argument("--reports-prefix", default="acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/asllvd-sign-videos/reports/")
    parser.add_argument("--manifest-key", default="acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/asllvd-sign-manifest-v1/manifest.json")
    parser.add_argument("--corpus-id", default="asllvd-dai2-signs-v1")
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    payload = store.get(args.annotations_key)
    inputs = {"annotations": hashlib.sha256(payload).hexdigest()}
    rows = {}
    for row in csv.DictReader(io.StringIO(payload.decode("utf-8-sig", errors="replace"))):
        key = row_key(row)
        if key is not None:
            rows.setdefault(key, row)
    records, skipped = [], collections.Counter()
    for key in sorted(store.list(args.reports_prefix)):
        report = store.get(key)
        inputs[f"report:{Path(key).name}"] = hashlib.sha256(report).hexdigest()
        for result in json.loads(report)["results"]:
            if result["status"] not in ("converted", "already_present") or "state_sha256" not in result:
                skipped[result["status"]] += 1; continue
            parsed = clip_key(result["sample_id"])
            if parsed is None:
                skipped["unparsed clip name"] += 1; continue
            signer, k = parsed
            row = rows.get(k)
            if row is None:
                skipped["no annotation row"] += 1; continue
            if signer not in PARTITION:
                skipped["unknown signer"] += 1; continue
            gloss = row["main entry gloss label"].strip()
            if not gloss:
                skipped["no gloss"] += 1; continue
            records.append(dict(
                sign=result["sample_id"], text=gloss, partition=PARTITION[signer],
                performer_sha256=hashlib.sha256(f"asllvd-dai2:{signer}".encode()).hexdigest(),
                gloss=dict(status="asllrp_annotation", label=gloss, variant=row.get("entry/variant gloss label", "").strip(),
                           sign_type=row.get("sign type", "").strip(),
                           handshapes={h: row[h].strip() for h in ("Dominant start handshape", "Non-dominant start handshape",
                                                                     "Dominant end handshape", "Non-dominant end handshape")
                                       if row.get(h, "").strip()}),
                source=dict(full_video=row["full video file"], frames=[k[2], k[3]], camera=k[4]),
                continuous=dict(status="converted", state_key=result["state_key"], state_sha256=result["state_sha256"],
                                available_channels=result["available_channels"], seconds=result["seconds"])))
    counts = {p: sum(r["partition"] == p for r in records) for p in ("train", "val", "test")}
    glosses = {p: len({r["gloss"]["label"] for r in records if r["partition"] == p}) for p in ("train", "val", "test")}
    split = dict(kind="isolated", source="signer-disjoint, fixed: val Naomi, test Tyler", counts=counts, glosses=glosses)
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
