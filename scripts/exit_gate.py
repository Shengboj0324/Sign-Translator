"""Apply the generator round's declared exit gate to stored evaluations and select a generator.

The gate (docs/CONTINUOUS_TRAINING.md, "Generator round exit gate", declared before the last
results) scores every candidate at every guidance scale on the same validation clips:

1. **Realism (primary):** the lowest Fréchet distance.
2. **Articulation:** speed and variation ratios of body, left hand and right hand within
   ``ARTICULATION`` ([0.7, 1.3]).
3. **Grammar markers:** generated/reference rate ratios of headshake, nod, brow raise and
   brow furrow within ``MARKERS`` ([0.5, 2.0]).
4. **Accuracy:** geodesic and wrist errors, reported and not thresholded.
5. **Ties:** within 10% Fréchet, prefer more gates passed (2–3), then the lower right-hand error.

Selection: among settings that pass gates 2 and 3, the lowest Fréchet. If none passes, the
setting that passes the most gate checks, with ties broken by the lowest Fréchet. The result
says which gates the selection fails, which become the next round's problems. R-precision
from ``score_generated_retrieval.py`` is reported beside it when given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_text_to_motion import make_store  # noqa: E402

ARTICULATION = (.7, 1.3)
MARKERS = (.5, 2.)
ARTICULATED = ("body", "left_hand", "right_hand")
MARKER_SCORES = ("headshake_rate", "nod_rate", "brow_raise_fraction", "brow_furrow_fraction")


def score_setting(run: str, setting: str, result: dict) -> dict:
    """Gate checks of one (run, guidance) evaluation result."""
    checks = {}
    for channel in ARTICULATED:
        if channel in result:
            for ratio in ("speed_ratio", "variation_ratio"):
                value = result[channel][ratio]
                checks[f"{channel}_{ratio}"] = (value, ARTICULATION[0] <= value <= ARTICULATION[1])
    distributions = (result.get("grammar") or {}).get("distributions") or {}
    for marker in MARKER_SCORES:
        ratio = (distributions.get(marker) or {}).get("mean_ratio")
        if ratio is not None:
            checks[marker] = (ratio, MARKERS[0] <= ratio <= MARKERS[1])
    accuracy = {f"{c}_error": next((v for k, v in result[c].items() if k.endswith(("_deg", "_cm"))), None)
                for c in ARTICULATED if c in result}
    accuracy.update({f"space_{k}": v for k, v in (result.get("signing_space") or {}).items()})
    passed = sum(ok for _, ok in checks.values())
    return dict(run=run, setting=setting, frechet=result["frechet"], checks=checks, passed=passed,
                total=len(checks), all_pass=passed == len(checks) and len(checks) > 0, accuracy=accuracy)


def select(rows: list[dict]) -> dict:
    """The declared selection rule over scored settings."""
    if not rows:
        raise ValueError("no evaluated settings")
    passing = [r for r in rows if r["all_pass"]]
    if passing:
        best = min(passing, key=lambda r: r["frechet"])
        near = [r for r in passing if r["frechet"] <= 1.1 * best["frechet"]]
        return min(near, key=lambda r: (r["accuracy"].get("right_hand_error") or float("inf"), r["frechet"]))
    best = max(rows, key=lambda r: (r["passed"], -r["frechet"]))
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--derived", default="acquisition/2026-10-17-v1/derived/subset-v1")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--retrieval-key", help="score_generated_retrieval.py output (reported, not gated)")
    parser.add_argument("--output-key")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    rows = []
    for run in args.runs:
        report = json.loads(store.get(f"{args.derived.rstrip('/')}/evaluations/{run}.json"))
        rows += [score_setting(run, setting, result) for setting, result in report["results"].items()]
    retrieval = json.loads(store.get(args.retrieval_key)) if args.retrieval_key else None
    choice = select(rows)
    failing = [k for k, (_, ok) in choice["checks"].items() if not ok]
    for r in sorted(rows, key=lambda r: r["frechet"]):
        bad = [k for k, (_, ok) in r["checks"].items() if not ok]
        print(f"{r['run']:<45} {r['setting']:<28} Fréchet {r['frechet']:8.1f}  gates {r['passed']}/{r['total']}"
              f"  failing {', '.join(bad) or '-'}")
    print(f"\nselected: {choice['run']} / {choice['setting']} (Fréchet {choice['frechet']:.1f}); "
          f"failing gates: {', '.join(failing) or 'none'}")
    report = dict(gate=dict(articulation=ARTICULATION, markers=MARKERS, ties="10% Fréchet"), rows=rows,
                  selected=dict(run=choice["run"], setting=choice["setting"], frechet=choice["frechet"],
                                failing=failing), retrieval=retrieval)
    if args.output_key:
        store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True, default=list).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
