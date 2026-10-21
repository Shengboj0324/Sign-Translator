"""ASLLRP Sign Bank lexicon, ASL Citizen gloss map and ASLLRP utterance gloss sequences from the DAI 2 annotations.

Inputs are the annotation CSVs acquired from the ASLLRP Data Access Interface (private R2 copy, research use only):
citation-form signs (ASLLVD, RIT) and signs segmented from continuous signing (ASLLRP corpora, RIT sentences).
Three outputs, each written once with the SHA-256 of every input:

1. ``lexicon.json``: per main-entry gloss, its entry/variant labels, token counts per collection, start/end
   handshape counts of both hands, sign types and class labels.
2. ``asl-citizen-gloss-map.json``: every ASL Citizen gloss mapped to Sign Bank main entries in tiers, each match
   tagged with its method: ``exact`` (main entry), ``variant`` (an entry/variant label), ``letters`` (same letters
   once parenthesised prefixes and punctuation are removed: THANKYOU = THANK-YOU), ``letters_no_number`` (the
   same after dropping a trailing variant number: CANDY1 -> CANDY). Only ``exact`` is a direct label match;
   every other tier is a candidate that needs review (the repository's rule for gloss candidates). The unmatched
   glosses are listed.
3. ``utterances.json``: for the continuous collections, each utterance's signs in time order with sign type,
   handshapes and frames: the gloss sequences the English -> gloss planner learns (English translations are in
   the DAI utterance XML, acquired separately).
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
from signtranslator.data_engineering.corpus_manifest import canonical_json  # noqa: E402

PREFIX = "acquisition/2026-10-17-v1/asllrp-dai2-v1/"
CITATION = {"asllvd-citation": "asllvd-citation/asllvd_signs_2024_06_27.csv",
            "rit-citation": "rit-citation/rit_signs_2024_06_27.csv"}
CONTINUOUS = {"asllrp-continuous": "asllrp-continuous/asllrp_sentence_signs_2025_06_28.csv",
              "rit-sentences": "rit-sentences/rit_sentence_signs_2025_11_01.csv"}
HANDS = ("Dominant start handshape", "Non-dominant start handshape", "Dominant end handshape",
         "Non-dominant end handshape")


def field(row: dict, name: str) -> str:
    """A column by name, case-insensitively (the collections capitalise their headers differently)."""
    for key, value in row.items():
        if key and key.strip().lower() == name.lower():
            return (value or "").strip()
    return ""


def letters(gloss: str) -> str:
    """Upper-case letters and digits only, after removing leading parenthesised prefixes like (1h) or (2h)alt."""
    gloss = re.sub(r"^(\([^)]*\)(alt\.)?)+", "", gloss.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^A-Z0-9]", "", gloss.upper())


def read_rows(store, key: str) -> tuple[list[dict], str]:
    payload = store.get(PREFIX + key)
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig", errors="replace")))), hashlib.sha256(payload).hexdigest()


def build_lexicon(tables: dict[str, list[dict]]) -> dict:
    lexicon = {}
    for collection, rows in tables.items():
        for row in rows:
            if field(row, "Hidden") == "T":
                continue
            main = field(row, "Main entry gloss label")
            if not main:
                continue
            entry = lexicon.setdefault(main, dict(variants=collections.Counter(), tokens=collections.Counter(),
                                                  handshapes={h: collections.Counter() for h in HANDS},
                                                  sign_types=collections.Counter(), class_labels=collections.Counter()))
            entry["variants"][field(row, "Entry/variant gloss label") or main] += 1
            entry["tokens"][collection] += 1
            for h in HANDS:
                if field(row, h):
                    entry["handshapes"][h][field(row, h)] += 1
            entry["sign_types"][field(row, "Sign type")] += 1
            entry["class_labels"][field(row, "Class label")] += 1
    return {gloss: dict(variants=dict(e["variants"]), tokens=dict(e["tokens"]),
                        handshapes={h: dict(c) for h, c in e["handshapes"].items() if c},
                        sign_types=dict(e["sign_types"]), class_labels=dict(e["class_labels"]))
            for gloss, e in sorted(lexicon.items())}


def map_glosses(citizen: dict[str, str | None], lexicon: dict) -> dict:
    """ASL Citizen gloss -> tiered Sign Bank candidates (see module docstring)."""
    by_variant = collections.defaultdict(set)
    by_letters = collections.defaultdict(set)
    for main, entry in lexicon.items():
        for variant in entry["variants"]:
            by_variant[variant.upper()].add(main)
        by_letters[letters(main)].add(main)
        for variant in entry["variants"]:
            by_letters[letters(variant)].add(main)
    mapping, unmatched = {}, []
    for gloss, code in sorted(citizen.items()):
        tiers = (("exact", {gloss} & set(lexicon)), ("variant", by_variant.get(gloss.upper(), set())),
                 ("letters", by_letters.get(letters(gloss), set())),
                 ("letters_no_number", by_letters.get(re.sub(r"\d+$", "", letters(gloss)), set())))
        for method, candidates in tiers:
            if candidates:
                mapping[gloss] = dict(method=method, candidates=sorted(candidates), asl_lex_code=code,
                                      requires_review=method != "exact")
                break
        else:
            unmatched.append(gloss)
    counts = collections.Counter(m["method"] for m in mapping.values())
    return dict(counts=dict(counts, unmatched=len(unmatched)), mapping=mapping, unmatched=unmatched)


def build_utterances(tables: dict[str, list[dict]]) -> list[dict]:
    groups = collections.defaultdict(list)
    for collection, rows in tables.items():
        for row in rows:
            if field(row, "Hidden") == "T":
                continue
            key = (collection, field(row, "Source collection"), field(row, "Utterance number"))
            groups[key].append(row)

    def number(text):
        return int(text) if text.strip().lstrip("-").isdigit() else None

    utterances = []
    for (collection, source, utterance), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], number(kv[0][2]) or 0)):
        rows.sort(key=lambda r: (number(field(r, "Start frame of the sign video")) or 0))
        first = rows[0]
        utterances.append(dict(
            collection=collection, source=source, utterance=number(utterance),
            utterance_video=field(first, "Utterance video filename"), master_video=field(first, "Master video filename"),
            frames=[number(field(first, "Start frame of the containing utterance")),
                    number(field(first, "End frame of the containing utterance"))],
            signs=[dict(gloss=field(r, "Main entry gloss label"), variant=field(r, "Entry/variant gloss label"),
                        occurrence=field(r, "Occurrence label"), sign_type=field(r, "Sign type"),
                        frames=[number(field(r, "Start frame of the sign video")), number(field(r, "End frame of the sign video"))],
                        handshapes={h: field(r, h) for h in HANDS if field(r, h)}, sign_video=field(r, "Sign video filename"))
                   for r in rows]))
    return utterances


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2", help="r2, or dir:<path>")
    parser.add_argument("--citizen-manifest-key", default="acquisition/2026-10-17-v1/derived/asl-citizen-v1/manifest-v1/manifest.json")
    parser.add_argument("--output-prefix", default="acquisition/2026-10-17-v1/derived/signbank-v1/")
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    inputs, citation, continuous = {}, {}, {}
    for target, keys in ((citation, CITATION), (continuous, CONTINUOUS)):
        for collection, key in keys.items():
            target[collection], inputs[key] = read_rows(store, key)
    manifest_payload = store.get(args.citizen_manifest_key)
    inputs[args.citizen_manifest_key] = hashlib.sha256(manifest_payload).hexdigest()
    citizen = {r["gloss"]["label"]: r["gloss"].get("asl_lex_code") for r in json.loads(manifest_payload)["records"]}
    lexicon = build_lexicon({**citation, **continuous})
    gloss_map = map_glosses(citizen, lexicon)
    utterances = build_utterances(continuous)
    prefix = args.output_prefix.rstrip("/") + "/"
    outputs = {"lexicon.json": dict(entries=len(lexicon), lexicon=lexicon),
               "asl-citizen-gloss-map.json": gloss_map,
               "utterances.json": dict(utterances=len(utterances), signs=sum(len(u["signs"]) for u in utterances),
                                       items=utterances)}
    for name, body in outputs.items():
        key = prefix + name
        if store.exists(key):
            raise SystemExit(f"refusing to overwrite {key}")
        store.put(key, canonical_json(dict(schema_version=1, inputs=dict(sorted(inputs.items())),
                                           license="ASLLRP Sign Bank Data Terms of Use: research use only, no redistribution",
                                           **body)))
    print(json.dumps(dict(lexicon_entries=len(lexicon), gloss_map=gloss_map["counts"], utterances=len(utterances),
                          utterance_signs=outputs["utterances.json"]["signs"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
