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

**Round 3d (declared 2026-10-25 08:20 PDT, before any round-3d result).** With ``--speed-joints-key`` (a
``joint_reliability.py`` report on How2Sign *training* clips), each channel's speed ratio is measured over the joints
whose real frame velocity is signal (lag-1 velocity autocorrelation above the report's threshold); How2Sign's torso
and upper-arm joints are jitter-dominated, so a speed ratio over them rewards reproducing extraction noise. The
all-joint speed ratios stay in the report (``*_speed_ratio_all_joints``); variation ratios use every joint. Round 3d
also runs R2b on 1000 clips (``--limit``), since at 300 even real clips barely separate from blank.
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


def restrict_joints(state, joints: dict):
    """``state`` with each named channel's validity limited to its listed joints (labels)."""
    channels = dict(state.channels)
    for name, keep in joints.items():
        if name not in channels:
            continue
        c = channels[name]
        mask = np.array([label in keep for label in c.labels])
        valid = c.valid & mask[None, :]
        channels[name] = replace(c, valid=valid, observed=c.observed & valid,
                                 confidence=np.where(valid, c.confidence, 0.),
                                 values=np.where(valid[..., None], c.values, 0.))      # unavailable payload is zero
    return replace(state, channels=channels)


def judge(ratios: dict[str, float], bleu: dict, separated) -> dict:
    bad = {k: v for k, v in ratios.items() if not k.endswith("_all_joints") and not BAND[0] <= v <= BAND[1]}
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
    p.add_argument("--speed-joints-key", help="round 3d: joint_reliability.py report; speed ratios over its reliable "
                                                "joints (all-joint speeds are still reported)")
    p.add_argument("--plan-tokenizer-key", help="round 17: a plan tokenizer over --tokenizer-key; the round trips decode "
                   "the latents reconstructed from its plan codes (the plan's fidelity gate: hand speed)")
    p.add_argument("--plan-tokenizer-sha256", help="its SHA-256 (required with --plan-tokenizer-key)")
    p.add_argument("--output-key")
    args = p.parse_args(argv)
    if bool(args.plan_tokenizer_key) != bool(args.plan_tokenizer_sha256):
        raise SystemExit("--plan-tokenizer-key and --plan-tokenizer-sha256 go together")
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
    plan = None
    if args.plan_tokenizer_key:                      # round 17: the plan's round trip through the same decoder
        import hashlib
        from signtranslator.generation.plan_codes import load_plan_tokenizer
        tokenizer_sha = hashlib.sha256(store.get(args.tokenizer_key)).hexdigest()   # the plan must read these bytes
        try:
            plan, _, saved_plan = load_plan_tokenizer(store.get(args.plan_tokenizer_key), args.plan_tokenizer_sha256,
                                                      tokenizer_sha256=tokenizer_sha)
        except ValueError as error:
            raise SystemExit(f"--plan-tokenizer-key: {error}")
        if saved_plan["tokenizer_key"] != args.tokenizer_key:
            raise SystemExit(f"the plan tokenizer reads {saved_plan['tokenizer_key']}, not --tokenizer-key")
    gains = json.loads(store.get(args.gains_key))["gains"] if args.gains_key else None
    speed_joints = None
    if args.speed_joints_key:
        reliability = json.loads(store.get(args.speed_joints_key))
        speed_joints = {name: v["reliable"] for name, v in reliability["channels"].items()}

    def trips_and_ratios(through):
        trips = [round_trip(model, model.config.motion, s, 24., plan=through) for s in real]
        if gains:
            from fit_tokenizer_gains import apply_gains
            trips = [apply_gains(t, gains) for t in trips]
        res = compare([observed_like(t, r) for t, r in zip(trips, real)], real, CHANNELS)
        ratios = {f"{c}_{k}": round(float(res[c][k]), 3) for c in CHANNELS if c in res
                  for k in ("speed_ratio", "variation_ratio")}
        if speed_joints:
            masked_trips = [restrict_joints(observed_like(t, r), speed_joints) for t, r in zip(trips, real)]
            restricted = compare(masked_trips, [restrict_joints(r, speed_joints) for r in real], CHANNELS)
            for c in CHANNELS:
                if c in speed_joints and c in restricted:
                    ratios[f"{c}_speed_ratio_all_joints"] = ratios[f"{c}_speed_ratio"]
                    ratios[f"{c}_speed_ratio"] = round(float(restricted[c]["speed_ratio"]), 3)
        return trips, ratios
    trips, ratios = trips_and_ratios(plan)
    # round 17 (review fix 2, report-only): with a plan, the tokenizer's own round trips of the same clips beside it
    tokenizer_trips, tokenizer_ratios = trips_and_ratios(None) if plan is not None else (None, None)
    translator, kind = load_translator(store, args.translator_key, args.translator_manifest_sha256)
    if kind != "keypoints":
        raise SystemExit("--translator-key must be the 2D keypoint back-translator")
    translator = translator.to(torch.device(args.device))
    texts = [r["text"] for r in records]
    bleu, hypotheses = {}, {}
    reads = [("real", real), ("round_trip", trips), ("blank", blanked(real))]
    if tokenizer_trips is not None:
        reads.append(("tokenizer_round_trip", tokenizer_trips))
    for name, states in reads:
        hypotheses[name] = translate(translator, kind, states)
        point, ci = bleu_ci(hypotheses[name], texts)
        bleu[name] = dict(bleu=point, ci95=ci)
    plan_block = {}
    if plan is not None:                    # the plan's content against the tokenizer's, paired over the same clips
        from compare_pilot_lexicon import paired_bleu_difference
        diff, (low, high) = paired_bleu_difference(hypotheses["round_trip"], hypotheses["tokenizer_round_trip"], texts)
        plan_block = dict(plan_tokenizer_key=args.plan_tokenizer_key, plan_tokenizer_sha256=args.plan_tokenizer_sha256,
                          tokenizer_sha256=tokenizer_sha, tokenizer_ratios=tokenizer_ratios,
                          plan_minus_tokenizer_bleu=dict(difference=round(diff, 3), ci95=[round(low, 3), round(high, 3)]))
    verdict = judge(ratios, {k: v for k, v in bleu.items() if k != "tokenizer_round_trip"}, separated)
    report = dict(tokenizer_key=args.tokenizer_key, clips=len(records), groups=sorted(groups), ratios=ratios, r2b=bleu,
                  **plan_block,
                  gains=gains, speed_joints=speed_joints,
                  **verdict)
    print(json.dumps(dict(ratios=ratios, r2b={k: v["bleu"] for k, v in bleu.items()}, out_of_band=verdict["out_of_band"],
                          r2b_above_blank=verdict["r2b_above_blank"], **{"pass": verdict["pass"]})), flush=True)
    if args.output_key:
        store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True).encode())
    return 0 if verdict["pass"] else 4          # as the domain and recognisability checks: 4 = a verdict, not an error


if __name__ == "__main__":
    raise SystemExit(main())
