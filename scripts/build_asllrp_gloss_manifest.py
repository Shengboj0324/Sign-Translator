"""ASLLRP / RIT utterance manifest: ASL gloss sequences paired with their extracted motion states (DAI 2).

Joins ``derived/signbank-v1/utterances.json`` (each utterance's signs in time order) with the extraction reports of
the utterance videos (``extract_video_states.py`` on the DAI utterance-video zips). A record's ``text`` is its
gloss sequence (main-entry glosses joined by spaces), so the existing text-conditioned trainers and evaluators can
learn gloss -> motion; ``gloss_sequence`` keeps each sign's gloss, sign type and frames.

Partitions are signer-disjoint and fixed here (not tuned): validation = Jonathan, test = Ben (both ASLLRP), training
= Cory, Rachel and the RIT participants P02 and P03. Utterances whose signer cannot be read from the source collection
name are left out. Research use only (ASLLRP Sign Bank Data Terms of Use); the manifest stays private. Written once.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json  # noqa: E402

PARTITION = {"Jonathan": "val", "Ben": "test", "Cory": "train", "Rachel": "train", "RIT-P02": "train", "RIT-P03": "train"}


def signer(source: str) -> str | None:
    """The participant of a DAI source collection: Cory_2013-6-25_sc0106 -> Cory, 4-Ben-Deaf-Speech -> Ben,
    RIT_P02-s01 / RIT-P03_s01 -> RIT-P02 / RIT-P03."""
    m = re.match(r"RIT[-_](P\d+)", source)
    if m:
        return f"RIT-{m.group(1)}"
    m = re.match(r"\d+-([A-Za-z]+)-", source)
    if m:
        return m.group(1)
    m = re.match(r"([A-Za-z]+)_\d{4}-", source)
    return m.group(1) if m else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2", help="r2, or dir:<path>")
    parser.add_argument("--utterances-key", default="acquisition/2026-10-17-v1/derived/signbank-v1/utterances.json")
    parser.add_argument("--reports-prefixes", nargs="+", default=[
        "acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/asllrp-continuous-utterances/reports/",
        "acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/rit-sentences-utterances/reports/"])
    parser.add_argument("--manifest-key", default="acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/gloss-manifest-v1/manifest.json")
    parser.add_argument("--corpus-id", default="asllrp-dai2-gloss-v1")
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    payload = store.get(args.utterances_key)
    inputs = {"utterances": hashlib.sha256(payload).hexdigest()}
    utterances = {Path(u["utterance_video"]).stem: u for u in json.loads(payload)["items"] if u.get("utterance_video")}
    converted = {}
    for prefix in args.reports_prefixes:
        for key in sorted(store.list(prefix)):
            report_payload = store.get(key)
            inputs[f"report:{Path(key).parent.parent.name}/{Path(key).name}"] = hashlib.sha256(report_payload).hexdigest()
            for result in json.loads(report_payload)["results"]:
                if result["status"] in ("converted", "already_present") and "state_sha256" in result:
                    converted.setdefault(result["sample_id"], result)
    records, skipped = [], collections.Counter()
    for sample, u in sorted(utterances.items()):
        who = signer(u["source"])
        if who not in PARTITION:
            skipped["unknown signer"] += 1; continue
        result = converted.get(sample)
        if result is None:
            skipped["no state"] += 1; continue
        glosses = [s["gloss"] for s in u["signs"] if s["gloss"]]
        if not glosses:
            skipped["no glosses"] += 1; continue
        records.append(dict(
            sign=f"{u['collection']}:{sample}", text=" ".join(glosses), partition=PARTITION[who],
            performer_sha256=hashlib.sha256(f"asllrp-dai2:{who}".encode()).hexdigest(),
            gloss=dict(status="asllrp_annotation", label=" ".join(glosses)),
            gloss_sequence=[dict(gloss=s["gloss"], sign_type=s["sign_type"], frames=s["frames"]) for s in u["signs"]],
            source=dict(collection=u["collection"], source=u["source"], utterance=u["utterance"],
                        utterance_video=u["utterance_video"]),
            continuous=dict(status="converted", state_key=result["state_key"], state_sha256=result["state_sha256"],
                            available_channels=result["available_channels"], seconds=result["seconds"])))
    counts = {p: sum(r["partition"] == p for r in records) for p in ("train", "val", "test")}
    split = dict(kind="continuous", source="signer-disjoint, fixed: val Jonathan, test Ben", counts=counts)
    manifest = build_corpus_manifest(corpus_id=args.corpus_id, inputs=inputs, records=records, split=split)
    manifest["skipped"] = dict(skipped)
    manifest["license"] = "ASLLRP Sign Bank Data Terms of Use: research use only, no redistribution"
    if store.exists(args.manifest_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.manifest_key, canonical_json(manifest))
    print(json.dumps(dict(manifest_key=args.manifest_key, manifest_sha256=digest, records=len(records),
                          partitions=counts, skipped=dict(skipped))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
