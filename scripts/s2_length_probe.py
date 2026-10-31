"""Stage S2's probe P3, predicted length (bars declared 2026-10-31 18:10 UTC, docs/NEXT_PHASE_PLAN.md, "Road to 3.8: S2
design received"): what does a text-only length cost? Nothing trains: a straight line on How2Sign train's metadata.

- ``lengths``: ``composition_v2.fit_duration_model`` signer-balanced on c02 + c04 (train partition, metadata only);
  ``lengths.json`` holds each list clip's predicted seconds, which B0 also uses, and the pooled ratio. Not blind: the
  design saw 1.08× on D20's metadata before this bar was declared.
- ``read``: real, blank, and real resampled to its predicted length.
- ``verdict``: pooled predicted/real length in [0.95, 1.10]; paired ΔNLL (``predicted`` minus ``real``) lower bound above −0.03;
  BLEU beside it, since slower output loses BLEU and faster gains it without content (S0, finding 1).

    python scripts/s2_length_probe.py lengths --out-dir W
    python scripts/s2_length_probe.py read --out-dir W
    python scripts/s2_length_probe.py verdict --out-dir W --output-file W/verdict.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import s2_lib as L  # noqa: E402
from signtranslator.eval import s0_metrics as M  # noqa: E402

FORMS = ("real", "blank", "predicted")
RATIO_BAND = (.95, 1.10)
DNLL_BAR = -.03


def predicted_lengths(manifest, records) -> dict:
    """The fit, every list clip's predicted seconds, and the pooled and median ratios against ``continuous.seconds``."""
    from signtranslator.generation.composition_v2 import fit_duration_model, predicted_seconds
    model = fit_duration_model(manifest, L.H2S, "train", components=("c02", "c04"), signer_balanced=True)
    predicted = [predicted_seconds(model, r["text"]) for r in records]
    real = [float(r["continuous"]["seconds"]) for r in records]
    ratio = np.array(predicted) / np.array(real)
    return dict(model=model, signs=[r["sign"] for r in records], predicted_seconds=[round(x, 5) for x in predicted],
                real_seconds=real, pooled_ratio=round(float(sum(predicted) / sum(real)), 5), median_ratio=round(float(np.median(ratio)), 5),
                median_abs_error=round(float(np.median(np.abs(ratio - 1))), 5))


def resampled(state, seconds: float):
    """``state`` played to last ``seconds`` (resampled to 24 fps, ``s0_ceilings.time_warp``)."""
    import s0_ceilings as s0
    own = (len(state.timestamps)) / L.FPS
    return s0.time_warp(state, own / seconds)


def make_verdict(reads: dict, lengths: dict) -> dict:
    report = L.verdict_shell("p3", reads)
    if lengths["signs"] != reads["predicted"]["signs"]:
        raise SystemExit("lengths.json is not this read's list")
    panels = {name: L.form_panel(reads, name) for name in reads}
    paired = panels["predicted"]["paired"]["real"]
    ratio = lengths["pooled_ratio"]
    bars = [L.bar("pooled_length_ratio", ratio, list(RATIO_BAND), RATIO_BAND[0] <= ratio <= RATIO_BAND[1]),
            L.bar("delta_nll_minus_real_lower_bound", paired["delta_nll"]["ci95"][0], DNLL_BAR,
                  paired["delta_nll"]["ci95"][0] > DNLL_BAR, difference=paired["delta_nll"]["difference"])]
    report.update(forms=panels, bars=bars, passed=all(b["passed"] for b in bars), model=lengths["model"],
                  median_ratio=lengths["median_ratio"], median_abs_error=lengths["median_abs_error"],
                  bleu_minus_real=paired["bleu"], blind=False,
                  note_blind="The design saw this fit's pooled 1.08× on D20's metadata before the bar was declared.")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("lengths")
    f.add_argument("--store", default="r2")
    f.add_argument("--out-dir", required=True)
    f.add_argument("--smoke-train", type=int, default=0)
    r = sub.add_parser("read")
    L.add_read_arguments(r, ",".join(FORMS))
    v = sub.add_parser("verdict")
    v.add_argument("--store", default="r2")
    v.add_argument("--out-dir", required=True)
    v.add_argument("--output-key")
    v.add_argument("--output-file")
    args = p.parse_args(argv)
    from train_text_to_motion import make_store
    if args.command == "lengths":
        store = make_store(args.store)
        records, declared, described = L.open_read_list(store, args.smoke_train)
        report = dict(predicted_lengths(L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA), records), declared_read=declared, list=described)
        L.write_file_once(Path(args.out_dir) / "lengths.json", report)
        print(json.dumps({k: report[k] for k in ("model", "pooled_ratio", "median_ratio", "median_abs_error")}), flush=True)
        return 0
    if args.command == "read":
        lengths = json.loads((Path(args.out_dir) / "lengths.json").read_text())

        def build(name, context):
            import s0_ceilings as s0
            if name in ("real", "blank"):
                return s0.build_form(name, context["refs"], context["tokenizer"], None)
            if name != "predicted":
                raise SystemExit(f"unknown form {name}")
            if lengths["signs"] != [r["sign"] for r in context["records"]]:
                raise SystemExit("lengths.json is not this read's list")
            return [resampled(s, t) for s, t in zip(context["refs"], lengths["predicted_seconds"])]
        import torch
        torch.set_num_threads(1)
        return L.read_forms(args, probe="p3", build=build)
    store = make_store(args.store) if args.output_key else None
    L.refuse_existing(store, args.output_key, args.output_file)
    report = make_verdict(L.load_form_reads(args.out_dir, FORMS), json.loads((Path(args.out_dir) / "lengths.json").read_text()))
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(passed=report["passed"], bars=[(b["name"], b["value"], b["passed"]) for b in report["bars"]])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
