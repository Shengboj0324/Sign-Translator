"""Power of the rater pilot's word-format contrast (docs/HUMAN_EVALUATION_PROTOCOL.md section 3; docs/NEXT_PHASE_PLAN.md, "Rater pilot:
word-format arms"). Simulation only, no rater and no clip: composition minus composition_shuffled on forced choice, four systems in a
Latin square, raters spread evenly over the four groups, every sentence moved.

**The declared rule is what is published,** ``human_study._by_the_rule`` on ``human_study._cluster_row``: method D's two-sided 95% t(R-1)
interval excludes zero, and the sign-flip p is under .05, in the same direction, with enough raters. The sign-flip test alone is printed
beside it and is higher, because it ignores raters.

**What "a true 20-point drop" means here, two readings.** Ratings follow a logit model with sentence SD 1 and rater SD 0.5, the release-bar
simulations' (plan, 2026-11-01, 03:28 UTC).
- ``accuracy``: over all sentences and raters, composition is understood 60% of the time and shuffled 40%: a difference in the accuracy the
  study reads. The intercepts are solved so that the mean over sentence and rater effects hits those rates.
- ``logit``: a median sentence with a median rater is at 60% and 40%. Sentence and rater spread pull both rates towards 50%, so the accuracy
  difference the study reads is smaller, about 16 points. Same for 10 points.

    python scripts/rater_pilot_power.py --sims 2000 --seed 20261102 --output reports/rater_pilot_power.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from signtranslator.eval_framework.human_study import _by_the_rule, _cluster_row  # noqa: E402

SENTENCE_SD, RATER_SD = 1., .5
BASE = .60
GROUPS = 4
COMPOSITION, SHUFFLED = 0, 2           # two of the four systems: with every group filled, which two does not matter
FLIPS = 2000
CONFIGS = ((20, 4), (20, 8), (60, 4), (60, 8), (60, 12), (60, 16), (200, 8), (200, 12))
DROPS = (.20, .10, 0.)
_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(60)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


def expit(x):
    return 1 / (1 + np.exp(-x))


def marginal(intercept: float) -> float:
    """The accuracy over sentence and rater effects at this intercept."""
    return float((_WEIGHTS * expit(intercept + np.hypot(SENTENCE_SD, RATER_SD) * _NODES)).sum())


def intercept_for(accuracy: float) -> float:
    low, high = -8., 8.
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if marginal(mid) < accuracy else (low, mid)
    return (low + high) / 2


def intercepts(reading: str, drop: float) -> tuple[float, float]:
    logit = lambda p: float(np.log(p / (1 - p)))      # noqa: E731
    if reading == "accuracy":
        return intercept_for(BASE), intercept_for(BASE - drop)
    if reading == "logit":
        return logit(BASE), logit(BASE - drop)
    raise ValueError(f"reading {reading!r}")


def one_study(rng, U: int, R: int, a: float, b: float, flips) -> tuple[float, bool, bool]:
    """(the study's accuracy difference, sign-flip detects a drop, the rule reads below)."""
    sentence, rater = rng.normal(0, SENTENCE_SD, U), rng.normal(0, RATER_SD, R)
    groups = np.arange(R) % GROUPS
    utterance, who, system, correct = [], [], [], []
    for k in range(R):
        for arm, level in ((COMPOSITION, a), (SHUFFLED, b)):
            mine = np.where((np.arange(U) + groups[k]) % GROUPS == arm)[0]
            hit = rng.random(len(mine)) < expit(level + sentence[mine] + rater[k])
            utterance.append(mine); who.append(np.full(len(mine), k)); system.append(np.full(len(mine), arm)); correct.append(hit.astype(float))
    utterance, who, system, correct = map(np.concatenate, (utterance, who, system, correct))
    rows = dict(systems=[0, 1, 2, 3], raters=list(range(R)), utterances=U, utterance=utterance, rater=who, system=system, correct=correct,
                said=np.zeros(len(correct)))
    means = {arm: {} for arm in (COMPOSITION, SHUFFLED)}
    for arm in means:
        mask = system == arm
        totals, counts = np.bincount(utterance[mask], correct[mask], U), np.bincount(utterance[mask], minlength=U)
        means[arm] = np.divide(totals, counts, out=np.full(U, np.nan), where=counts > 0)
    both = np.isfinite(means[COMPOSITION]) & np.isfinite(means[SHUFFLED])
    d = means[SHUFFLED][both] - means[COMPOSITION][both]
    mean = float(d.mean())
    scale = np.abs(d).max()
    p = 1. if scale == 0 else (1 + int((np.abs((flips[:, :len(d)] * np.abs(d)).mean(1)) >= abs(mean) - 1e-12).sum())) / (len(flips) + 1)
    row = dict(comprehension_choice=dict(p_value=p, mean_difference=mean), comprehension_choice_method_d=_cluster_row(rows, SHUFFLED, COMPOSITION),
               rater_balance=dict(confounded_with_rater=False))
    return mean, p < .05 and mean < 0, _by_the_rule(row) == "below"


def simulate(U: int, R: int, drop: float, reading: str, *, sims: int, seed: int) -> dict:
    rng = np.random.default_rng([seed, U, R, int(round(drop * 100)), ["accuracy", "logit"].index(reading)])
    flips = rng.choice([-1., 1.], size=(FLIPS, U))
    a, b = intercepts(reading, drop)
    out = [one_study(rng, U, R, a, b, flips) for _ in range(sims)]
    means = np.array([x[0] for x in out])
    return dict(sentences=U, raters=R, reading=reading, true_drop_points=round(drop * 100), sims=sims, mean_accuracy_drop_points=round(float(-means.mean()) * 100, 1),
                half_width_points=round(float(1.96 * means.std()) * 100, 1), sign_flip_detects=round(float(np.mean([x[1] for x in out])), 3),
                rule_reads_below=round(float(np.mean([x[2] for x in out])), 3))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--configs", default=",".join(f"{u}x{r}" for u, r in CONFIGS), help="SENTENCESxRATERS, comma-separated")
    args = parser.parse_args(argv)
    configs = [tuple(int(x) for x in c.split("x")) for c in args.configs.split(",") if c]
    rows = []
    for U, R in configs:
        for reading in ("accuracy", "logit"):
            for drop in DROPS:
                rows.append(simulate(U, R, drop, reading, sims=args.sims, seed=args.seed))
                print(json.dumps(rows[-1]), flush=True)
    report = dict(kind="rater-pilot-power", seed=args.seed, sims=args.sims, base=BASE, sentence_sd=SENTENCE_SD, rater_sd=RATER_SD, flips=FLIPS, simulation_only=True, rows=rows)
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
