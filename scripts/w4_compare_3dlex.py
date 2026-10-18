"""Masked-reconstruction baselines on converted 3D-LEX canonical states (W4, B44).

Reads the verified conversion report, fetches each canonical state from R2 by its
recorded hash, hides one seeded interior span per sign, and scores the simple
baselines on common support with the solved skeleton explicitly declared as the
reference (``allow_inferred=True``). Errors pool cell numerators and counts; they
are never averaged per sign first. The independence unit is the performer, so
statistical resolution is reported from the number of performers, not signs.

This establishes how well trivial baselines reconstruct short gaps in isolated
signs. It is the bar any learned representation must beat on the same support;
it is not a statement about continuous signing or comprehension.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.fold_plan import minimum_sign_test_p  # noqa: E402
from signtranslator.generation.representation_comparison import (  # noqa: E402
    _cell_errors, hidden_spans, hold_out_rest_baseline, interpolation_baseline,
    zero_order_hold_baseline,
)
from signtranslator.pose.multichannel import CHANNELS, MultichannelMotion  # noqa: E402
from w4_convert_3dlex import client  # noqa: E402

SCORED = ("body", "left_hand", "right_hand", "head_rotation", "root_translation")
METHODS = {
    "interpolation": lambda s, h: interpolation_baseline(s, h, allow_inferred=True),
    "zero_order_hold": lambda s, h: zero_order_hold_baseline(s, h, allow_inferred=True),
    "rest": hold_out_rest_baseline,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-report", type=Path, required=True)
    parser.add_argument("--span-seconds", type=float, default=.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.conversion_report.read_bytes())
    records = [r for r in report["records"] if "state_key" in r]
    s3, bucket = client()
    pooled = {name: {m: [0., 0] for m in METHODS} for name in SCORED}
    per_performer = collections.defaultdict(lambda: {name: {m: [0., 0] for m in METHODS} for name in SCORED})
    paired = collections.defaultdict(list)  # (channel) -> per-sign (interp - hold) mean error
    coverage = {name: {m: [0, 0] for m in METHODS} for name in SCORED}
    skipped = collections.Counter()
    with tempfile.TemporaryDirectory() as temporary:
        for record in records:
            payload = s3.get_object(Bucket=bucket, Key=record["state_key"])["Body"].read()
            if hashlib.sha256(payload).hexdigest() != record["state_sha256"]:
                raise SystemExit(f"state hash mismatch for {record['state_key']}")
            path = Path(temporary) / "state.npz"
            path.write_bytes(payload)
            state = MultichannelMotion.load(path, expected_sha256=record["state_sha256"])
            path.unlink()
            seed = int(hashlib.sha256(record["sign"].encode()).hexdigest()[:8], 16)
            try:
                hidden = hidden_spans(state, channels=SCORED, span_seconds=args.span_seconds,
                                      spans=1, seed=seed)
            except ValueError:
                skipped["no interior span fits"] += 1
                continue
            outputs = {label: method(state, hidden) for label, method in METHODS.items()}
            for name in SCORED:
                channel = state.channels[name]
                kind = CHANNELS[name][0]
                eligible = channel.valid & hidden[name]
                common = eligible.copy()
                for label in METHODS:
                    covered = outputs[label][name][1]
                    coverage[name][label][0] += int((covered & eligible).sum())
                    coverage[name][label][1] += int(eligible.sum())
                    common &= covered
                if not common.any():
                    continue
                errors = {}
                for label in METHODS:
                    values = outputs[label][name][0]
                    errors[label] = _cell_errors(kind, values[common],
                                                 channel.values[common].astype(np.float64))
                    for target in (pooled[name][label], per_performer[record["performer_sha256"]][name][label]):
                        target[0] += float(errors[label].sum())
                        target[1] += int(errors[label].size)
                paired[name].append(float(errors["interpolation"].mean() - errors["zero_order_hold"].mean()))

    def summarise(table):
        return {name: {label: (None if count == 0 else dict(mean_error=total / count, cells=count))
                       for label, (total, count) in methods.items()} for name, methods in table.items()}

    performers = sorted(p for p in per_performer if p is not None)
    performer_wins = {}
    for name in SCORED:
        wins = []
        for performer in performers:
            interp, hold = per_performer[performer][name]["interpolation"], per_performer[performer][name]["zero_order_hold"]
            if interp[1] and hold[1]:
                wins.append(interp[0] / interp[1] < hold[0] / hold[1])
        performer_wins[name] = dict(performers_compared=len(wins), interpolation_better=sum(wins))
    units = len(performers)
    result = dict(
        schema_version=1, conversion_report_sha256=hashlib.sha256(args.conversion_report.read_bytes()).hexdigest(),
        signs_scored=len(records) - sum(skipped.values()), skipped=dict(skipped),
        span_seconds=args.span_seconds, reference="valid_including_inferred (solved skeleton)",
        units={"body": "degrees", "left_hand": "degrees", "right_hand": "degrees",
               "head_rotation": "degrees", "root_translation": "metres"},
        pooled=summarise(pooled),
        coverage={name: {m: (c[0] / c[1] if c[1] else None) for m, c in methods.items()}
                  for name, methods in coverage.items()},
        per_sign_interpolation_minus_hold={name: dict(
            signs=len(values), mean=float(np.mean(values)) if values else None,
            fraction_interpolation_better=float(np.mean(np.array(values) < 0)) if values else None)
            for name, values in paired.items()},
        performer_level=performer_wins, independent_units=units,
        best_possible_two_sided_sign_test_p=str(minimum_sign_test_p(units)) if units else None,
        interpretation=("Signs are not independent units: they share performers. With "
                        f"{units} performers no two-sided paired sign test can go below "
                        f"p = {minimum_sign_test_p(units) if units else 'n/a'}; these are "
                        "descriptive baseline errors, not a confirmatory comparison."),
        phase_exit_approved=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("signs_scored", "pooled", "independent_units",
                                             "best_possible_two_sided_sign_test_p")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
