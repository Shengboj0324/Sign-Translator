"""Round 5 pilot decision: an arm's lexicon against the control's, paired by word, with a bootstrap interval.

Both reports come from ``pilot_content_check.py`` with the same instrument (``lexicon.recognizer_run``) and the same
word list; ``per_word`` counts each word's top-5 hits over its two samples. The difference is the arm's top-5 minus
the control's. Its 95% interval comes from resampling words with replacement (``--resamples``, fixed seed). The rule
(docs/NEXT_PHASE_PLAN.md, round 5 D) has three parts: the difference is at least ``--min-gain``, its interval
excludes 0, and the arm's meaning BLEU is not below the control's.

Round 7 amendment 2 (2026-10-27): when both reports keep their per-clip translations (``meaning.<name>.hypotheses``
and ``meaning.references``), the BLEU condition is paired non-inferiority instead: the 95% lower bound of
BLEU(arm) - BLEU(control), over the same resampled clips, must be above ``--bleu-margin`` (-0.5). Comparing the two
point estimates passed an equal-meaning arm only half of the time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def paired_difference(arm: dict, control: dict, resamples: int = 10000, seed: int = 0):
    """(difference, (low, high)) of top-5 rates over the shared words, paired by word."""
    words = sorted(set(arm) & set(control))
    if not words or set(arm) != set(control):
        raise ValueError("the two reports must score the same words")
    a = np.array([arm[w] for w in words], dtype=float) / 2
    c = np.array([control[w] for w in words], dtype=float) / 2
    d = a - c
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (resamples, len(d)))].mean(axis=1)
    return float(d.mean()), (float(np.quantile(boot, .025)), float(np.quantile(boot, .975)))


def paired_bleu_difference(arm_hyps, control_hyps, refs, resamples: int = 2000, seed: int = 0):
    """(difference, (low, high)) of corpus BLEU, arm minus control, resampling the same clips for both."""
    from signtranslator.generation.motion_to_text import corpus_bleu
    if not (len(arm_hyps) == len(control_hyps) == len(refs)):
        raise ValueError("the two reports must translate the same clips")
    point = corpus_bleu(arm_hyps, refs)["bleu"] - corpus_bleu(control_hyps, refs)["bleu"]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(resamples):
        idx = rng.integers(0, len(refs), len(refs))
        r = [refs[i] for i in idx]
        diffs.append(corpus_bleu([arm_hyps[i] for i in idx], r)["bleu"] - corpus_bleu([control_hyps[i] for i in idx], r)["bleu"])
    return float(point), (float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975)))


def verdict(arm_report: dict, control_report: dict, min_gain: float = .05, resamples: int = 10000,
            bleu_margin: float = -.5) -> dict:
    if arm_report["lexicon"]["recognizer_run"] != control_report["lexicon"]["recognizer_run"]:
        raise ValueError("the arm and the control were read by different instruments")
    diff, (lo, hi) = paired_difference(arm_report["lexicon"]["per_word"], control_report["lexicon"]["per_word"], resamples)
    am, cm = arm_report["meaning"], control_report["meaning"]
    paired = None
    if "hypotheses" in am["generated"] and "hypotheses" in cm["generated"]:
        if am["references"] != cm["references"]:
            raise ValueError("the two reports translated different references")
        bd, (blo, bhi) = paired_bleu_difference(am["generated"]["hypotheses"], cm["generated"]["hypotheses"],
                                                am["references"])
        paired = dict(difference=round(bd, 3), ci95=[round(blo, 3), round(bhi, 3)], margin=bleu_margin)
        bleu_ok = blo > bleu_margin
    else:
        bleu_ok = am["generated"]["bleu"] >= cm["generated"]["bleu"]
    return dict(bleu_paired=paired, difference=round(diff, 4), ci95=[round(lo, 4), round(hi, 4)], min_gain=min_gain,
                gain_ok=diff >= min_gain, ci_excludes_zero=lo > 0, bleu_ok=bool(bleu_ok),
                promising=bool(diff >= min_gain and lo > 0 and bleu_ok),
                arm=dict(lexicon=arm_report["lexicon"]["generated_top5"], bleu=arm_report["meaning"]["generated"]["bleu"]),
                control=dict(lexicon=control_report["lexicon"]["generated_top5"],
                             bleu=control_report["meaning"]["generated"]["bleu"]),
                instrument=arm_report["lexicon"]["recognizer_run"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arm-key", required=True)
    p.add_argument("--control-key", required=True)
    p.add_argument("--min-gain", type=float, default=.05)
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--bleu-margin", type=float, default=-.5)
    p.add_argument("--output-key")
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    out = verdict(json.loads(store.get(args.arm_key)), json.loads(store.get(args.control_key)), args.min_gain,
                  args.resamples, args.bleu_margin)
    out.update(arm_key=args.arm_key, control_key=args.control_key)
    print(json.dumps(out))
    if args.output_key:
        store.put(args.output_key, json.dumps(out, indent=2).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
