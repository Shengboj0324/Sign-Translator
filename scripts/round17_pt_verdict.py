"""Round 17 PT-v2 verdict (plan "Round 17 PT-v2: declared", 10:17 UTC, and its 10:20 amendment): every declared plan
tokenizer gate for one arm, from its best-validation checkpoint's validation terms, the stored fidelity and
recognisability reports and the d2 reference perplexity. Prints a table and writes JSON; exit 0 PASS, 4 FAIL, 2 not
evaluable.

    python scripts/round17_pt_verdict.py --arm a --out /tmp/v2a.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

O = "acquisition/2026-10-17-v1/youtube-asl-v1/round17"
PARTS = ("body", "face", "left_hand", "right_hand")
REFERENCE = {"a": "every_4th_step", "b": "every_step"}          # amendment 10:20: arm b against d2 every step
FLOOR, RATIO, CONTENT, RECOGNISER, RESEED = 32., .8, .8, .9, .01


def verdict(best: dict, reference: dict, fidelity: dict | None, recognisability: dict | None, arm: str,
            log: list | None = None) -> dict:
    rows = []
    v = best["validation"]
    ref = reference["reference_stage1_perplexity"]
    for p in PARTS:
        value, d2 = v.get(f"val/stage1_perplexity_{p}"), ref[f"{REFERENCE[arm]}_{p}"]
        bar = max(FLOOR, RATIO * d2)
        rows.append(dict(row=f"perplexity.{p}", value=value, bar=f">= {bar:.1f} (0.8 x d2 {REFERENCE[arm]} {d2}, floor 32)",
                         status=None if value is None else value >= bar))
    for p in PARTS:
        value = v.get(f"stability/reseed_fraction_per_1k_{p}")
        rows.append(dict(row=f"reseed_per_1k.{p}", value=value, bar="< 0.01 after step 10k",
                         status=None if value is None else value < RESEED))
    if log is not None:
        bad = [r["step"] for r in log if "loss" in r and not (r["loss"] == r["loss"] and abs(r["loss"]) < 1e6)]
        rows.append(dict(row="no_divergence", value=len(bad), bar="no non-finite or exploding loss", status=not bad))
    if fidelity is None:
        rows.append(dict(row="content.round_trip_bleu", value=None, bar=">= 0.8 x d2 round trip, above blank",
                         status=None))
    else:
        r2b = fidelity["r2b"]
        ratio = r2b["round_trip"] / r2b["tokenizer_round_trip"]
        rows.append(dict(row="content.round_trip_bleu", value=f"{r2b['round_trip']} ({ratio:.4f} x d2 {r2b['tokenizer_round_trip']})",
                         bar=">= 0.8 x d2", status=ratio >= CONTENT))
        rows.append(dict(row="content.above_blank", value=fidelity.get("r2b_above_blank"), bar="lower bound above blank",
                         status=bool(fidelity.get("r2b_above_blank")) if "r2b_above_blank" in fidelity else None))
    if recognisability is None:
        rows.append(dict(row="content.recogniser_ratio", value=None, bar=">= 0.9 x real", status=None))
    else:
        rows.append(dict(row="content.recogniser_ratio", value=recognisability["ratio"], bar=">= 0.9 x real top-5",
                         status=recognisability["ratio"] >= RECOGNISER))
    statuses = [r["status"] for r in rows]
    overall = "FAIL" if False in statuses else ("NOT-EVALUABLE" if None in statuses else "PASS")
    return dict(arm=arm, checkpoint=best["checkpoint_key"], step=best["step"], rows=rows, overall=overall,
                report_only={k: fidelity["ratios"][k] for k in fidelity["ratios"]} if fidelity else None)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arm", choices=("a", "b"), required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    run = f"plan-vq-v2{args.arm}"
    get = lambda k: json.loads(store.get(k)) if store.exists(k) else None   # noqa: E731
    best = get(f"training-runs/{run}/best.json")
    logs = sorted(o["Key"] for o in store.s3.list_objects_v2(Bucket=store.bucket, Prefix=f"training-runs/{run}/log/")
                  .get("Contents", []))
    out = verdict(best, get(f"{O}/reference-perplexity-d2.json"), get(f"{O}/plan-fidelity-{run}.json"),
                  get(f"{O}/plan-recognisability-{run}.json"), args.arm, get(logs[-1]) if logs else None)
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"{run} best step {out['step']}: {out['overall']}")
    for r in out["rows"]:
        print(f"  {'PASS' if r['status'] else 'FAIL' if r['status'] is False else 'N/E '}  {r['row']}: {r['value']}  [{r['bar']}]")
    return {"PASS": 0, "FAIL": 4}.get(out["overall"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
