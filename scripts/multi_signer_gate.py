"""The exit gate across several unseen validation signers (round 3).

The full-corpus round's gate selected on one validation signer and then failed the test signer (R6: head speed 1.83×,
nod rate 19.8×). Here every candidate (run, setting) is scored by ``exit_gate.score_setting`` on each validation
signer's own evaluation, stored as ``<derived>/evaluations/<run>-<group>.json``. Thresholds and channels are the
declared exit gate's, unchanged.
- A candidate passes only if it passes every check on **every** group.
- Its Fréchet distance and right-hand error are the means over groups.
- Selection follows ``exit_gate.select``: among passing candidates within 10% of the best mean Fréchet, the lowest
  right-hand error.
- If no candidate passes everywhere, the one with the most checks passed over all groups is chosen, ties broken by
  mean Fréchet, and the failing (group, check) pairs are reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exit_gate import ARTICULATED, score_setting  # noqa: E402


def combine(per_group: dict[str, dict]) -> dict:
    """One candidate's rows (group -> score_setting row) as a single gate row."""
    rows = list(per_group.values())
    failing = [f"{g}:{k}" for g, r in per_group.items() for k, (_, ok) in r["checks"].items() if not ok]
    rh = [r["accuracy"].get("right_hand_error") for r in rows]
    return dict(run=rows[0]["run"], setting=rows[0]["setting"], groups=sorted(per_group),
                frechet=sum(r["frechet"] for r in rows) / len(rows),
                passed=sum(r["passed"] for r in rows), total=sum(r["total"] for r in rows),
                all_pass=all(r["all_pass"] for r in rows), failing=failing,
                accuracy=dict(right_hand_error=(sum(rh) / len(rh)) if all(v is not None for v in rh) else None),
                per_group={g: dict(frechet=r["frechet"], passed=r["passed"], total=r["total"],
                                   checks=r["checks"]) for g, r in per_group.items()})


def select(rows: list[dict]) -> dict:
    passing = [r for r in rows if r["all_pass"]]
    if passing:
        best = min(passing, key=lambda r: r["frechet"])
        near = [r for r in passing if r["frechet"] <= 1.1 * best["frechet"]]
        return min(near, key=lambda r: (r["accuracy"].get("right_hand_error") or float("inf"), r["frechet"]))
    return max(rows, key=lambda r: (r["passed"], -r["frechet"]))


def candidates(store, derived: str, runs: list[str], groups: list[str]) -> list[dict]:
    rows = []
    for run in runs:
        results = {g: json.loads(store.get(f"{derived.rstrip('/')}/evaluations/{run}-{g}.json"))["results"]
                   for g in groups}
        settings = set.intersection(*(set(r) for r in results.values()))
        for setting in sorted(settings):
            rows.append(combine({g: score_setting(run, setting, results[g][setting], ARTICULATED) for g in groups}))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--derived", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--groups", required=True, help="comma-separated validation signer components")
    parser.add_argument("--output-key")
    args = parser.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    rows = candidates(store, args.derived, args.runs, args.groups.split(","))
    if not rows:
        raise SystemExit("no candidate was evaluated on every group")
    choice = select(rows)
    for r in sorted(rows, key=lambda r: r["frechet"]):
        print(f"{r['run']:<50} {r['setting']:<44} mean Fréchet {r['frechet']:8.1f}  checks {r['passed']}/{r['total']}"
              f"  failing {', '.join(r['failing'][:6]) or '-'}")
    print(f"\nselected: {choice['run']} / {choice['setting']} (mean Fréchet {choice['frechet']:.1f}); "
          f"failing: {', '.join(choice['failing']) or 'none'}")
    if args.output_key:
        store.put(args.output_key, json.dumps(dict(groups=args.groups.split(","), rows=rows, selected=dict(
            run=choice["run"], setting=choice["setting"], frechet=choice["frechet"], failing=choice["failing"])),
            indent=2, sort_keys=True, default=list).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
