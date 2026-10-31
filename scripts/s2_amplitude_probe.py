"""Stage S2's probe P4, amplitude (declared 2026-10-31 18:10 UTC, docs/NEXT_PHASE_PLAN.md, "Road to 3.8: S2 design
received"): does the instrument read smaller motion more easily, on real signing, with no tokenizer involved?

**Forms:** ``real``, ``blank``, and ``amp0.8``, ``amp0.9``, ``amp1.1``, ``amp1.27``: the body and both hands' rotations scaled
about each clip's mean rotation (``fit_tokenizer_gains.apply_gains``, the operation the decoder gains are, so ×1.27 is the
hands' accepted gain). Timing and validity do not change.

**Not a stop. The rule:** if ``amp0.8`` minus ``real`` has a paired ΔNLL lower bound above 0, the preference is real:
``motion_bands_bind`` is true, and BLEU or likelihood gains on v1 alone do not count. (ΔNLL over blank is positive on any
real motion, so the declared "lower bound > 0" is read as the difference from real, on the same clips.)

    python scripts/s2_amplitude_probe.py read --out-dir W
    python scripts/s2_amplitude_probe.py verdict --out-dir W --output-file W/verdict.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import s2_lib as L  # noqa: E402

AMPLITUDES = (.8, .9, 1.1, 1.27)
FORMS = ("real", "blank") + tuple(f"amp{a}" for a in AMPLITUDES)
CHANNELS = ("body", "left_hand", "right_hand")


def scaled(state, factor: float):
    from fit_tokenizer_gains import apply_gains
    return apply_gains(state, {name: factor for name in CHANNELS})


def build(name: str, context: dict):
    import s0_ceilings as s0
    if name in ("real", "blank"):
        return s0.build_form(name, context["refs"], context["tokenizer"], None)
    if not name.startswith("amp") or float(name[3:]) not in AMPLITUDES:
        raise SystemExit(f"unknown form {name}")
    return [scaled(s, float(name[3:])) for s in context["refs"]]


def make_verdict(reads: dict) -> dict:
    report = L.verdict_shell("p4", reads)
    panels = {name: L.form_panel(reads, name) for name in reads}
    low = panels["amp0.8"]["paired"]["real"]["delta_nll"]
    binds = low["ci95"][0] > 0
    report.update(forms=panels, motion_bands_bind=bool(binds), stop=False,
                  bars=[L.bar("amp0.8_delta_nll_minus_real_lower_bound", low["ci95"][0], 0., binds, difference=low["difference"])],
                  against_real={name: panels[name]["paired"]["real"] for name in panels if name.startswith("amp")},
                  consequence="motion bands bind; v1-only gains do not count" if binds else "no such preference resolved")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("read")
    L.add_read_arguments(r, ",".join(FORMS))
    v = sub.add_parser("verdict")
    v.add_argument("--store", default="r2")
    v.add_argument("--out-dir", required=True)
    v.add_argument("--output-key")
    v.add_argument("--output-file")
    args = p.parse_args(argv)
    if args.command == "read":
        import torch
        torch.set_num_threads(1)
        return L.read_forms(args, probe="p4", build=build)
    from train_text_to_motion import make_store
    store = make_store(args.store) if args.output_key else None
    L.refuse_existing(store, args.output_key, args.output_file)
    report = make_verdict(L.load_form_reads(args.out_dir, FORMS))
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(motion_bands_bind=report["motion_bands_bind"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
