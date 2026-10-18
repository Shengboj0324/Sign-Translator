"""Calibrated error bounds and abstention for gap reconstruction on real 3D-LEX data.

Loads the frozen corpus manifest from R2 and serves samples through the manifest
dataset, so only admitted channels are scored (an unvalidated hand is never used).
For each sign, one seeded span (the W4 seed rule) is hidden and reconstructed by
geodesic interpolation, the method selected by the W4 comparison. Per channel, a
split-conformal bound is fitted on the training performer and evaluated twice:

* on a seeded held-out half of the same performer, where calibration and test
  units are plausibly exchangeable, so the guarantee applies;
* on the other performer, which tests whether the bound transfers; there the
  guarantee is an assumption under test, not a theorem.

Declared before the run: alpha 0.1, difficulty floor 0.1 degrees, tolerance 2
degrees for every rotation channel, spans of 0.25, 0.5 and 1.0 seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.manifest_dataset import ManifestMotionDataset  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest  # noqa: E402
from signtranslator.generation.conformal_abstention import (  # noqa: E402
    evaluate_abstention, fit_conformal_bound, span_error_and_difficulty,
)
from signtranslator.generation.representation_comparison import (  # noqa: E402
    hidden_spans, interpolation_baseline,
)
from w4_convert_3dlex import client  # noqa: E402

ALPHA = .1
FLOOR_DEGREES = .1
TOLERANCE_DEGREES = 2.
SPANS = (.25, .5, 1.)
CHANNELS = ("body", "left_hand", "right_hand", "head_rotation")


def seed(sign: str) -> int:
    return int(hashlib.sha256(sign.encode()).hexdigest()[:8], 16)


def unit_scores(dataset, span):
    """Per channel: list of (sign, error, difficulty) for every scorable unit."""
    out = {name: [] for name in CHANNELS}
    for index in range(len(dataset)):
        record, state = dataset.records[index], dataset[index]
        usable = tuple(n for n in CHANNELS if state.channels[n].valid.any())
        try:
            hidden = hidden_spans(state, channels=usable, span_seconds=span, spans=1, seed=seed(record["sign"]))
        except ValueError:
            continue  # the sign is shorter than an interior span of this length
        recon = interpolation_baseline(state, hidden, allow_inferred=True)
        for name in usable:
            values, covered = recon[name]
            unit = span_error_and_difficulty(state, name, hidden[name], values, covered, allow_inferred=True)
            if unit is not None:
                out[name].append((record["sign"], *unit))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    s3, bucket = client()
    fetch = lambda key: s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "manifest.json"
        path.write_bytes(fetch(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
        cache = ContentCache(Path(temporary) / "cache", 64 * 2 ** 20)
        train = ManifestMotionDataset(manifest, "skeleton", partition="train", cache=cache, fetch=fetch)
        test = ManifestMotionDataset(manifest, "skeleton", partition="test", cache=cache, fetch=fetch)
        results = []
        for span in SPANS:
            calibration_units, transfer_units = unit_scores(train, span), unit_scores(test, span)
            for name in CHANNELS:
                units = calibration_units[name]
                rng = np.random.default_rng(seed(f"{name}:{span}"))
                order = rng.permutation(len(units))
                half = len(units) // 2
                fit = [units[i] for i in order[:half]]
                held = [units[i] for i in order[half:]]
                full = fit_conformal_bound([u[1] for u in units], [u[2] for u in units],
                                           alpha=ALPHA, floor=FLOOR_DEGREES)
                halfbound = fit_conformal_bound([u[1] for u in fit], [u[2] for u in fit],
                                                alpha=ALPHA, floor=FLOOR_DEGREES)
                other = transfer_units[name]
                row = dict(span_seconds=span, channel=name, calibration=halfbound.to_dict(),
                           same_performer_held_out=evaluate_abstention(
                               [u[1] for u in held], [u[2] for u in held], halfbound,
                               tolerance=TOLERANCE_DEGREES),
                           transfer_calibration=full.to_dict(),
                           other_performer=evaluate_abstention(
                               [u[1] for u in other], [u[2] for u in other], full,
                               tolerance=TOLERANCE_DEGREES) if other else None)
                results.append(row)
                print(span, name, len(units), len(other),
                      round(row["same_performer_held_out"]["coverage"], 3),
                      None if not other else round(row["other_performer"]["coverage"], 3), flush=True)
    record = dict(schema_version=1, manifest_sha256=args.manifest_sha256, method="geodesic interpolation",
                  alpha=ALPHA, floor_degrees=FLOOR_DEGREES, tolerance_degrees=TOLERANCE_DEGREES,
                  unit="one hidden span of one sign; error is the span's largest geodesic error",
                  difficulty="largest geodesic angle between the visible rotations bracketing the span",
                  results=results, independent_performers=2,
                  interpretation=("same-performer coverage is covered by the split-conformal theorem; "
                                  "other-performer coverage tests transfer and carries no guarantee"),
                  phase_exit_approved=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
