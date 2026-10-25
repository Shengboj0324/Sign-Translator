"""Per-channel excursion gains for a uniformly damped tokenizer (round 3c amendment 5, declared 2026-10-25 05:00 PDT).

On ``--limit`` *training* clips of How2Sign (every training signer), each passed through the tokenizer and back and measured on
the frames the clip observed, the gain of channel c is

    g_c = clip( variation_real_c / variation_round_trip_c, 1, 1.5 ),

the factor that ``postprocess.damp_rotation_channel`` (an excursion scale about the clip's mean rotation) applies after
decoding. Fitted on training clips only, then judged on validation by ``tokenizer_fidelity_check.py --gains-key``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

CHANNELS = ("body", "left_hand", "right_hand")
CAP = (1., 1.5)


def decoder_gains_key(tokenizer_key: str) -> str:
    """Where an *accepted* gains file lives (amendment 5): written only after the gained fidelity and recognisability
    checks pass; ``evaluate_text_to_motion.load_generator`` then applies it to every state decoded by that tokenizer."""
    run, _, name = tokenizer_key.rpartition("/checkpoints/")
    if not run or not name.endswith(".pt"):
        raise ValueError(f"not a tokenizer checkpoint key: {tokenizer_key}")
    return f"{run}/decoder-gains-{name[:-3]}.json"


def apply_gains(state, gains: dict):
    """``state`` with each channel's excursions about its clip mean scaled by its gain (1 leaves it unchanged)."""
    from signtranslator.generation.postprocess import damp_rotation_channel
    for channel, g in (gains or {}).items():
        if channel in state.channels and state.channels[channel].valid.any() and g != 1:
            state = damp_rotation_channel(state, channel, float(g))
    return state


def fit(variation_ratios: dict) -> dict:
    """Gains from round-trip/real variation ratios, capped to [1, 1.5]."""
    return {c: round(min(max(1. / max(r, 1e-9), CAP[0]), CAP[1]), 4) for c, r in variation_ratios.items()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest-key", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--tokenizer-key", required=True)
    p.add_argument("--groups", default="", help="signer components to fit on (default: every training signer; the "
                                                "How2Sign splits are signer-disjoint, so c01/c03/c05 have no training clips)")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    import tempfile
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from signtranslator.generation.motion_metrics import compare
    from evaluate_text_to_motion import at_rate, load_state
    from tokenizer_fidelity_check import observed_like
    from tokenizer_recognisability_check import round_trip
    import diagnose_tokenizer as dt
    store = R2Store(os.environ)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    groups = {g for g in args.groups.split(",") if g}
    records = [r for r in manifest["records"] if r.get("corpus") == "how2sign-two-view-text-v1"
               and r.get("partition") == "train" and (not groups or r.get("signer_component") in groups)]
    if not records:
        raise SystemExit("no How2Sign training clips match")
    records = random.Random(0).sample(records, min(args.limit, len(records)))
    real = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), 24.)
            for r in records]
    model, _ = dt.load_tokenizer(store, args.tokenizer_key)
    model.eval()
    trips = [observed_like(round_trip(model, model.config.motion, s, 24.), s) for s in real]
    res = compare(trips, real, CHANNELS)
    ratios = {c: float(res[c]["variation_ratio"]) for c in CHANNELS if c in res}
    gains = fit(ratios)
    report = dict(tokenizer_key=args.tokenizer_key, partition="train", clips=len(records), cap=list(CAP),
                  variation_ratios={c: round(v, 4) for c, v in ratios.items()}, gains=gains)
    store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True).encode())
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
