"""One table comparing generator evaluations and tokenizer diagnoses stored in R2.

Reads ``<derived>/evaluations/<run>.json`` (``evaluate_text_to_motion.py``) and
``<derived>/diagnostics/<run>.json`` (``diagnose_tokenizer.py``) for the named runs and
prints, per run and guidance scale, the per-channel accuracy and articulation ratios,
Fréchet distance, signing-space errors, grammar association and locus adherence; and,
per tokenizer, the full / decoder-floor / target-jitter error split. Writes the table
as JSON with ``--output``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_text_to_motion import make_store  # noqa: E402

CHANNELS = ("body", "left_hand", "right_hand", "head_rotation", "root_translation", "left_gaze")


def generator_rows(report: dict) -> list[dict]:
    rows = []
    for name, res in report["results"].items():
        row = dict(setting=name)
        for ch in CHANNELS:
            if ch in res:
                error = next(k for k in res[ch] if k.endswith(("_deg", "_cm")))
                row[f"{ch}_{error}"] = res[ch][error]
                row[f"{ch}_speed"] = res[ch]["speed_ratio"]
                row[f"{ch}_variation"] = res[ch]["variation_ratio"]
        row["frechet"] = res["frechet"]
        row.update({f"space_{k}": v for k, v in (res.get("signing_space") or {}).items()})
        for test in (res.get("grammar") or {}).get("tests", []):
            row[f"grammar_{test['feature']}"] = (test["generated_auc"], test["reference_auc"])
        adherence = (res.get("grammar") or {}).get("locus_adherence") or {}
        row["locus_within_20deg"] = adherence.get("within")
        rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--derived", default="acquisition/2026-10-17-v1/derived/subset-v1")
    parser.add_argument("--generators", nargs="*", default=[])
    parser.add_argument("--tokenizers", nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    store = make_store(args.store)
    table = dict(generators={}, tokenizers={})
    for run in args.generators:
        key = f"{args.derived}/evaluations/{run}.json"
        if store.exists(key):
            table["generators"][run] = generator_rows(json.loads(store.get(key)))
    for run in args.tokenizers:
        key = f"{args.derived}/diagnostics/{run}.json"
        if store.exists(key):
            d = json.loads(store.get(key))
            table["tokenizers"][run] = dict(overall=d["overall"], hands=d["hands"],
                                            worst=[(j, v["full"]) for j, v in d["worst_joints"][:5]])
    for run, rows in table["generators"].items():
        for row in rows:
            print(run, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
    for run, t in table["tokenizers"].items():
        print(run, json.dumps(t, default=lambda x: round(x, 2)))
    if args.output:
        args.output.write_text(json.dumps(table, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
