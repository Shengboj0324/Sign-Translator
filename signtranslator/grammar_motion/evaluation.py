"""Grammar metrics for generated motion against references on the same sentences.

For every clip the detectors of this package give per-clip scores (brow raise /
furrow level and coverage, headshake rate, role-shift coverage, pointing rate, loci,
final lengthening). Two comparisons:

* **Grammar association** — for each English feature F and detector score S
  (``TESTS``), the Mann–Whitney AUC of S between sentences with and without F, on the
  generated clips and on their references. A generator that has learned the marker
  reproduces the reference AUC; one that ignores grammar sits at 0.5.
Non-manual scores use one neutral baseline for both sides, pooled over the
references, so generated faces are measured on the real signers' scale.

* **Distribution match** — for each score, the 1-Wasserstein distance between its
  generated and reference distributions (in the score's own units), and the ratio
  of means (articulation: < 1 means damped non-manuals).
"""
from __future__ import annotations

import numpy as np

from .text import sentence_features
from .nmm import pooled_baselines
from .tracks import grammar_record

TESTS = (("wh_question", "brow_furrow_mean_z"), ("yes_no_question", "brow_raise_mean_z"),
         ("conditional", "brow_raise_mean_z"), ("negation", "headshake_rate"),
         ("reported_speech", "role_shift_fraction"), ("deixis", "pointing_rate"))
SCORES = ("brow_raise_mean_z", "brow_furrow_mean_z", "brow_raise_fraction", "brow_furrow_fraction",
          "headshake_rate", "nod_rate", "role_shift_fraction", "pointing_rate", "loci", "final_lengthening")


def clip_scores(record: dict) -> dict:
    nmm, head, rs, pt, pr = (record.get(k, {}) for k in ("nmm", "head", "role_shift", "pointing", "prosody"))
    seconds = max(record["frames"] / record["fps"], 1e-6)
    out = {}
    for name in ("brow_raise", "brow_furrow", "squint", "mm", "puff", "oo"):
        if name in nmm:
            out[f"{name}_mean_z"] = nmm[name]["mean_z"]
            out[f"{name}_fraction"] = nmm[name]["fraction"]
    if head:
        out["headshake_rate"] = len(head["headshake"]) / seconds
        out["headshake_any"] = float(bool(head["headshake"]))
        out["nod_rate"] = len(head["nod"]) / seconds
    if rs:
        out["role_shift_fraction"] = rs["fraction"]
    if pt:
        out["pointing_rate"] = len(pt["events"]) / seconds
        out["loci"] = float(len(pt["loci"]))
    if pr and pr.get("final_lengthening") is not None:
        out["final_lengthening"] = pr["final_lengthening"]
        out["final_blink"] = float(pr["final_blink"])
    return out


def auc(pos, neg) -> float:
    """Mann–Whitney AUC with ties counted half (nan when a side is empty)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None]).sum() + .5 * (pos[:, None] == neg[None]).sum()
    return float(greater / (len(pos) * len(neg)))


def wasserstein1(a, b) -> float:
    """1-Wasserstein distance between two empirical distributions on the line."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return float("nan")
    q = np.linspace(0, 1, max(len(a), len(b)) * 4 + 1)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def grammar_comparison(generated, references, texts) -> dict:
    """Association AUCs and distribution distances of grammar scores, generated vs reference."""
    faces = [s.channels["face"] for s in references if s.channels["face"].valid.any() and "browInnerUp" in s.channels["face"].labels]
    baselines = pooled_baselines(faces) if faces else None       # one neutral face for both sides: the references'
    gen = [clip_scores(grammar_record(s, baselines)) for s in generated]
    ref = [clip_scores(grammar_record(s, baselines)) for s in references]
    feats = [sentence_features(t) for t in texts]
    tests = []
    for feature, score in TESTS:
        row = dict(feature=feature, score=score, positives=sum(bool(f[feature]) for f in feats))
        for label, scores in (("generated", gen), ("reference", ref)):
            pos = [s[score] for s, f in zip(scores, feats) if f[feature] and score in s]
            neg = [s[score] for s, f in zip(scores, feats) if not f[feature] and score in s]
            row[f"{label}_auc"] = auc(pos, neg)
        tests.append(row)
    distributions = {}
    for score in SCORES:
        g = [s[score] for s in gen if score in s]; r = [s[score] for s in ref if score in s]
        if g and r:
            mean_r = float(np.mean(r))
            distributions[score] = dict(wasserstein1=wasserstein1(g, r), generated_mean=float(np.mean(g)),
                                        reference_mean=mean_r,
                                        mean_ratio=float(np.mean(g) / mean_r) if abs(mean_r) > 1e-9 else None)
    return dict(tests=tests, distributions=distributions, clips=len(generated))


__all__ = ["TESTS", "SCORES", "clip_scores", "auc", "wasserstein1", "grammar_comparison"]
