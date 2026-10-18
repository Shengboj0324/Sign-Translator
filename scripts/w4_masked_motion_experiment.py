"""Performer-held-out masked-motion experiment on validated 3D-LEX states.

Train on the performer with more validated signs, select the epoch on a held-out
tenth of that performer's signs, and test once on the other performer. Every
method is scored on identical hidden cells with the solved skeleton declared as
the reference. Evaluation spans use the same seeded rule as the W4 baseline
table, so numbers are directly comparable to it.

One held-out performer is one independent unit: the result is descriptive, and a
positive difference here is evidence worth replicating, not a confirmed claim.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.generation.masked_motion import (  # noqa: E402
    ROTATION_CHANNELS, ResidualMaskedMotion, build_example, evaluate, train_epoch,
)
from signtranslator.generation.representation_comparison import hidden_spans  # noqa: E402
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402
from w4_convert_3dlex import client  # noqa: E402



def sign_seed(sign: str, variant: int | None = None) -> int:
    text = sign if variant is None else f"{sign}#train-variant-{variant}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--variants", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--span-seconds", type=float, default=.25,
                        help="hidden span length; .25 is the original run, longer gaps were pre-declared")
    args = parser.parse_args(argv)
    report = json.loads(args.conversion_report.read_bytes())
    records = [r for r in report["records"] if "state_key" in r]
    groups = collections.Counter(r["performer_sha256"] for r in records)
    (train_performer, _), (test_performer, _) = groups.most_common(2)
    s3, bucket = client()
    states = {}
    with tempfile.TemporaryDirectory() as temporary:
        for record in records:
            payload = s3.get_object(Bucket=bucket, Key=record["state_key"])["Body"].read()
            if hashlib.sha256(payload).hexdigest() != record["state_sha256"]:
                raise SystemExit(f"state hash mismatch {record['state_key']}")
            path = Path(temporary) / "s.npz"
            path.write_bytes(payload)
            states[record["sign"]] = (record["performer_sha256"],
                                      MultichannelMotion.load(path, expected_sha256=record["state_sha256"]))
            path.unlink()

    def example(sign, seed):
        state = states[sign][1]
        try:
            hidden = hidden_spans(state, channels=ROTATION_CHANNELS, span_seconds=args.span_seconds,
                                  spans=1, seed=seed)
        except ValueError:
            return None
        return build_example(state, hidden)

    train_signs = sorted(s for s, (p, _) in states.items() if p == train_performer)
    test_signs = sorted(s for s, (p, _) in states.items() if p == test_performer)
    validation_signs = [s for s in train_signs if sign_seed(s) % 10 == 0]
    fit_signs = [s for s in train_signs if s not in set(validation_signs)]
    training = [e for s in fit_signs for v in range(args.variants)
                if (e := example(s, sign_seed(s, v))) is not None]
    validation = [e for s in validation_signs if (e := example(s, sign_seed(s))) is not None]
    test = [e for s in test_signs if (e := example(s, sign_seed(s))) is not None]
    print(f"train examples {len(training)} from {len(fit_signs)} signs; validation {len(validation)}; "
          f"test {len(test)}", flush=True)

    torch.manual_seed(0)
    model = ResidualMaskedMotion(training[0].target.shape[1], hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    start = evaluate(model, validation)
    best = (start["model_mean_degrees"], 0, {k: v.detach().clone() for k, v in model.state_dict().items()})
    history.append(dict(epoch=0, validation=start))
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, training, optimizer, seed=epoch)
        result = evaluate(model, validation)
        history.append(dict(epoch=epoch, train_loss=loss, validation=result))
        print(f"epoch {epoch:3d} train {loss:.6f} val model {result['model_mean_degrees']:.4f} "
              f"interp {result['interpolation_mean_degrees']:.4f}", flush=True)
        if result["model_mean_degrees"] < best[0]:
            best = (result["model_mean_degrees"], epoch,
                    {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    # The test performer is evaluated once, after selection is frozen.
    test_result = evaluate(model, test)
    record = dict(
        schema_version=1,
        conversion_report_sha256=hashlib.sha256(args.conversion_report.read_bytes()).hexdigest(),
        design="train one performer, select on held-out tenth of its signs, test once on the other",
        train_performer_sha256=train_performer, test_performer_sha256=test_performer,
        fit_signs=len(fit_signs), validation_signs=len(validation_signs), test_signs=len(test_signs),
        train_examples=len(training), span_seconds=args.span_seconds, variants=args.variants,
        hidden_dim=args.hidden_dim, epochs=args.epochs, selected_epoch=best[1],
        parameters=sum(p.numel() for p in model.parameters()),
        validation_history=history, test=test_result,
        reference="solved skeleton, declared allow_inferred",
        independent_test_units=1,
        interpretation=("One held-out performer is one independent unit; this is descriptive. "
                        "The model starts exactly at interpolation, so any difference is learned."),
        phase_exit_approved=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(selected_epoch=best[1], test=test_result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
