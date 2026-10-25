"""Tokenizer fidelity acceptance (round 3c, declared 2026-10-25 00:15 PDT before any round-3c tokenizer result).

Round 3b's prior failed R4 (head speed 1.45x on short sentences) and the gate (c03 right-hand variation 0.664). Real
How2Sign validation clips passed through round 3a's tokenizer and back already show both: on observed frames the
head turns 1.58x faster, the hands move at 0.71x speed and 0.67-0.72x variation, and the 2D back-translator (R2b)
reads them at 3.05 [2.27, 3.90] against blank 1.74 [1.04, 2.32]. A prior cannot be more faithful than its tokenizer.

**Rule.** On ``--limit`` validation clips of the How2Sign signer groups, each passed through the tokenizer (the
tokenizer recognisability check's ``round_trip``) and compared with the same clip on the frames the clip observed:
- body, left hand, right hand and head rotation: speed ratio and variation ratio in [0.85, 1.15];
- R2b: the 2D back-translator's BLEU on the round trips has a bootstrap 95% interval entirely above blank's.
Both must hold. The report gives every ratio and BLEU, and ``pass``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

BAND = (0.85, 1.15)
CHANNELS = ("body", "left_hand", "right_hand", "head_rotation")


def observed_like(trip, real):
    """``trip`` with each channel's validity set to the real clip's, so both are measured on the same frames."""
    def masked(name, c):
        if name not in real.channels:
            return c
        valid = real.channels[name].valid & c.valid
        return replace(c, valid=valid, observed=c.observed & valid, confidence=np.where(valid, c.confidence, 0.),
                       values=np.where(valid[..., None], c.values, 0.))          # unavailable payload is zero
    return replace(trip, channels={name: masked(name, c) for name, c in trip.channels.items()})


def judge(ratios: dict[str, float], bleu: dict, separated) -> dict:
    bad = {k: v for k, v in ratios.items() if not BAND[0] <= v <= BAND[1]}
    r2b_ok = separated(bleu["round_trip"]["ci95"], bleu["blank"]["ci95"])
    return dict(band=list(BAND), out_of_band=bad, r2b_above_blank=r2b_ok, **{"pass": not bad and r2b_ok})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest-key", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--tokenizer-key", required=True)
    p.add_argument("--translator-key", required=True, help="the accepted 2D keypoint back-translator (R2b)")
    p.add_argument("--translator-manifest-sha256", required=True)
    p.add_argument("--groups", default="c01,c03,c05")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gains-key", help="amendment 5: fit_tokenizer_gains.py output; its per-channel excursion gains "
                                         "are applied to every round trip after decoding")
    p.add_argument("--output-key")
    args = p.parse_args(argv)
    import tempfile
    import torch
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from signtranslator.generation.motion_metrics import compare
    from evaluate_text_to_motion import at_rate, blanked, load_state
    from strict_prior_review import bleu_ci, separated
    from score_back_translation import load_translator, translate
    import diagnose_tokenizer as dt
    from tokenizer_recognisability_check import round_trip
    store = R2Store(os.environ)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    groups = set(args.groups.split(","))
    records = [r for r in manifest["records"] if r.get("corpus") == "how2sign-two-view-text-v1"
               and r.get("partition") == "val" and r.get("signer_component") in groups and r.get("text")]
    records = random.Random(0).sample(records, min(args.limit, len(records)))
    real = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), 24.)
            for r in records]
    model, _ = dt.load_tokenizer(store, args.tokenizer_key)
    model.eval()                 # round_trip collates on the CPU, as the recognisability check's tokenizer does
    trips = [round_trip(model, model.config.motion, s, 24.) for s in real]
    gains = json.loads(store.get(args.gains_key))["gains"] if args.gains_key else None
    if gains:
        from fit_tokenizer_gains import apply_gains
        trips = [apply_gains(t, gains) for t in trips]
    res = compare([observed_like(t, r) for t, r in zip(trips, real)], real, CHANNELS)
    ratios = {f"{c}_{k}": round(float(res[c][k]), 3) for c in CHANNELS if c in res
              for k in ("speed_ratio", "variation_ratio")}
    translator, kind = load_translator(store, args.translator_key, args.translator_manifest_sha256)
    if kind != "keypoints":
        raise SystemExit("--translator-key must be the 2D keypoint back-translator")
    translator = translator.to(torch.device(args.device))
    texts = [r["text"] for r in records]
    bleu = {}
    for name, states in (("real", real), ("round_trip", trips), ("blank", blanked(real))):
        point, ci = bleu_ci(translate(translator, kind, states), texts)
        bleu[name] = dict(bleu=point, ci95=ci)
    verdict = judge(ratios, bleu, separated)
    report = dict(tokenizer_key=args.tokenizer_key, clips=len(records), groups=sorted(groups), ratios=ratios, r2b=bleu,
                  gains=gains,
                  **verdict)
    print(json.dumps(dict(ratios=ratios, r2b={k: v["bleu"] for k, v in bleu.items()}, out_of_band=verdict["out_of_band"],
                          r2b_above_blank=verdict["r2b_above_blank"], **{"pass": verdict["pass"]})), flush=True)
    if args.output_key:
        store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True).encode())
    return 0 if verdict["pass"] else 4          # as the domain and recognisability checks: 4 = a verdict, not an error


if __name__ == "__main__":
    raise SystemExit(main())
