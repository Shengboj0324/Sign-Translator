"""Measure ASL grammar on real How2Sign motion and validate the detectors against the English.

For a hashed sample of sentence clips (whole fold-plan components, so signers are
identifiable), non-manual tracks are z-scored against each recording session's
pooled neutral face (all of the session's clips, so a sentence's own marker cannot
move its baseline), and each canonical state gets calibrated gaze (``gaze_model.json``) when it
lacks one and its grammar record (``grammar_motion.tracks``); its English sentence
gets the deterministic grammatical features (``grammar_motion.text``). Two products:

1. **Detector validation without grammar labels.** ASL marks wh-questions with a brow
   furrow, yes/no questions and conditionals with a brow raise and negation with a
   headshake (Dachkovsky et al. 2013; Neidle et al. 2000). If the detectors measure
   those markers, clips whose English has the feature must score higher than clips
   without it. For each (feature, detector score) pair this reports the AUC (the
   Mann–Whitney probability that a clip with the feature outranks one without) with a
   95% bootstrap interval, overall and per fold component (signer group). It is a weak
   test: an English feature does not guarantee the ASL marker, and How2Sign is
   instructional monologue, but an AUC reliably above 0.5 cannot come from a detector
   blind to the marker.
2. **Reference distributions the literature lacks** (reports/ASL grammar and spatial
   reference.md, stage 4): locus azimuth/elevation and loci per clip, role-shift
   prevalence, phrase-final lengthening and final blinks, marker prevalence.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.text_motion_dataset import R2Store  # noqa: E402
from signtranslator.data_engineering.gaze import GazeModel, gaze_channels  # noqa: E402
from signtranslator.grammar_motion.text import sentence_features  # noqa: E402
from signtranslator.grammar_motion.tracks import grammar_record  # noqa: E402
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402

CSV_KEY = "acquisition/2026-10-17-v1/how2sign-public/how2sign_realigned_train.csv"
PLAN = ROOT / "Sign Translator Stage Documentation/evidence/w1-fold-plan-2026-10-17/fold-plan.json"
GAZE = ROOT / "signtranslator/data_engineering/gaze_model.json"
TESTS = [  # (English feature, detector score, expected direction)
    ("wh_question", "brow_furrow_mean_z"), ("wh_question", "brow_furrow_fraction"),
    ("yes_no_question", "brow_raise_mean_z"), ("yes_no_question", "brow_raise_fraction"),
    ("conditional", "brow_raise_mean_z"), ("negation", "headshake_rate"), ("negation", "headshake_any"),
    ("reported_speech", "role_shift_fraction"), ("deixis", "pointing_rate"), ("third_person", "pointing_rate"),
]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann–Whitney AUC with ties counted half."""
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                       # average ranks of ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_auc(pos, neg, rng, draws=1000):
    if len(pos) < 5 or len(neg) < 5:
        return [float("nan"), float("nan")]
    stats = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(draws)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def clip_scores(record: dict) -> dict:
    nmm, head, rs, pt, pr = (record.get(k, {}) for k in ("nmm", "head", "role_shift", "pointing", "prosody"))
    seconds = record["frames"] / record["fps"]
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
        out["loci"] = len(pt["loci"])
    if pr:
        out["final_lengthening"] = pr["final_lengthening"]
        out["final_blink"] = float(pr["final_blink"])
    return out


def load(store, key):
    payload = store.get(key)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "s.npz"; p.write_bytes(payload)
        return MultichannelMotion.load(p, expected_sha256=hashlib.sha256(payload).hexdigest())


def session_of(sample: str) -> str:
    """'<VIDEOID>_<n>-<take>-rgb_front' -> '<VIDEOID>-<take>' (one signer, one recording); other names are their own session."""
    stem = sample.removesuffix("-rgb_front")
    if "-" not in stem or "_" not in stem.rsplit("-", 1)[0]:
        return sample
    head, take = stem.rsplit("-", 1)
    return head.rsplit("_", 1)[0] + "-" + take


def session_baselines(store, prefix, rows, workers):
    """Pass 1: pooled neutral-face baselines per recording session."""
    from signtranslator.grammar_motion.nmm import raw_tracks
    def raw(row):
        try:
            face = load(store, f"{prefix}/states/{row['SENTENCE_NAME']}.npz").channels["face"]
        except Exception:
            return row["SENTENCE_NAME"], None
        if not face.valid.any() or "browInnerUp" not in face.labels:
            return row["SENTENCE_NAME"], None
        tracks, valid = raw_tracks(face)
        return row["SENTENCE_NAME"], {k: v[valid] for k, v in tracks.items()}
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        pairs = list(pool.map(raw, rows))
    grouped = {}
    for sample, tracks in pairs:
        if tracks is not None:
            grouped.setdefault(session_of(sample), []).append(tracks)
    out = {}
    for session, clips in grouped.items():
        out[session] = {}
        for name in clips[0]:
            v = np.concatenate([c[name] for c in clips])
            med = float(np.median(v))
            out[session][name] = (med, max(1.4826 * float(np.median(np.abs(v - med))), 1e-3))
    return out


def process(store, prefix, row, gaze_model, gaze_sha, baselines=None):
    from dataclasses import replace
    key = f"{prefix}/states/{row['SENTENCE_NAME']}.npz"
    try:
        state = load(store, key)
    except Exception as error:                     # missing or refused clip: skipped, counted
        return dict(sample=row["SENTENCE_NAME"], error=type(error).__name__)
    if not state.channels["left_gaze"].valid.any() and state.channels["face"].valid.any():
        g = gaze_channels(state.channels["face"], state.channels["head_rotation"], gaze_model, source_id="gaze",
                          source_sha256=gaze_sha, inference_method="calibrated gaze")
        state = replace(state, channels={**state.channels, **g})
    record = grammar_record(state, (baselines or {}).get(session_of(row["SENTENCE_NAME"])))
    return dict(sample=row["SENTENCE_NAME"], text=row["SENTENCE"], features=sentence_features(row["SENTENCE"]),
                scores=clip_scores(record), loci=record.get("pointing", {}).get("loci", []),
                role_shifts=record.get("role_shift", {}).get("spans", []), fps=record["fps"])


def summarise(rows, components, rng):
    ok = [r for r in rows if "scores" in r]
    out = dict(clips=len(rows), measured=len(ok), tests=[])
    for feature, score in TESTS:
        values = [(bool(r["features"][feature]), r["scores"].get(score), components.get(r["sample"])) for r in ok]
        values = [(f, s, c) for f, s, c in values if s is not None and np.isfinite(s)]
        pos = np.array([s for f, s, _ in values if f]); neg = np.array([s for f, s, _ in values if not f])
        per = {}
        for comp in sorted({c for _, _, c in values if c}):
            p = np.array([s for f, s, c in values if f and c == comp]); n = np.array([s for f, s, c in values if not f and c == comp])
            if len(p) >= 5 and len(n) >= 5:
                per[comp] = dict(auc=auc(p, n), positives=len(p))
        out["tests"].append(dict(feature=feature, score=score, positives=len(pos), negatives=len(neg),
                                 auc=auc(pos, neg), ci95=bootstrap_auc(pos, neg, rng), per_component=per))
    loci = [l for r in ok for l in r["loci"]]
    az = np.array([l["azimuth_deg"] for l in loci]); el = np.array([l["elevation_deg"] for l in loci])
    q = lambda x: np.percentile(x, [5, 25, 50, 75, 95]).round(1).tolist() if len(x) else []
    fl = np.array([r["scores"]["final_lengthening"] for r in ok if r["scores"].get("final_lengthening")])
    out["distributions"] = dict(
        locus_azimuth_deg_p5_25_50_75_95=q(az), locus_elevation_deg_p5_25_50_75_95=q(el),
        loci_per_clip_mean=float(np.mean([r["scores"].get("loci", 0) for r in ok])),
        clips_with_pointing=float(np.mean([r["scores"].get("pointing_rate", 0) > 0 for r in ok])),
        role_shift_clip_rate=float(np.mean([r["scores"].get("role_shift_fraction", 0) > 0 for r in ok])),
        final_lengthening_p25_50_75=np.percentile(fl, [25, 50, 75]).round(2).tolist() if len(fl) else [],
        final_blink_rate=float(np.mean([r["scores"].get("final_blink", 0) for r in ok])),
        marker_clip_rate={m: float(np.mean([r["scores"].get(f"{m}_fraction", 0) > 0 for r in ok]))
                          for m in ("brow_raise", "brow_furrow", "squint", "mm", "puff", "oo")},
        headshake_clip_rate=float(np.mean([r["scores"].get("headshake_any", 0) for r in ok])),
        english_feature_rate={f: float(np.mean([bool(r["features"][f]) for r in ok])) for f in
                              ("wh_question", "yes_no_question", "other_question", "negation", "conditional",
                               "reported_speech")})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", default="acquisition/2026-10-17-v1/derived/how2sign-video-estimate-v1")
    parser.add_argument("--clips", type=int, default=3000)
    parser.add_argument("--enrich-questions", type=int, default=600,
                        help="also include up to this many clips from the sessions richest in questions/negation")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-key")
    args = parser.parse_args(argv)
    store = R2Store(os.environ)
    rows = list(csv.DictReader(io.StringIO(store.get(CSV_KEY).decode()), delimiter="\t"))
    components = json.loads(PLAN.read_text())["sample_components"]
    rows = [r for r in rows if r["SENTENCE_NAME"] in components]
    # Whole sessions in hashed order (so every measured clip's session baseline pools all of
    # its clips), then sessions with marked English (questions, negation, conditionals) to
    # give the association tests enough positives.
    by_session = {}
    for r in rows:
        by_session.setdefault(session_of(r["SENTENCE_NAME"]), []).append(r)
    order = sorted(by_session, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    marked = lambda n: sum(any(sentence_features(r["SENTENCE"])[k] for k in
                               ("wh_question", "yes_no_question", "negation", "conditional")) for r in by_session[n])
    chosen_sessions, count = [], 0
    for name in order:
        if count >= args.clips:
            break
        chosen_sessions.append(name); count += len(by_session[name])
    extra = 0
    for name in sorted((n for n in order if n not in set(chosen_sessions)), key=marked, reverse=True):
        if extra >= args.enrich_questions:
            break
        chosen_sessions.append(name); extra += len(by_session[name])
    chosen = [r for n in chosen_sessions for r in by_session[n]]
    gaze_bytes = GAZE.read_bytes()
    gaze_model, gaze_sha = GazeModel.from_json(gaze_bytes.decode()), hashlib.sha256(gaze_bytes).hexdigest()
    prefix = args.states.rstrip("/")
    baselines = session_baselines(store, prefix, chosen, args.workers)     # all clips of each chosen session
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda r: process(store, prefix, r, gaze_model, gaze_sha, baselines), chosen))
    summary = summarise(results, components, np.random.default_rng(0))
    summary.update(states=args.states, gaze_model_sha256=gaze_sha, sessions=len(chosen_sessions),
                   sample_rule=f"whole sessions in sha256 order up to {args.clips} clips, then sessions richest in "
                   f"question/negation/conditional sentences up to {args.enrich_questions} more clips")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    if args.output_key:
        store.put(args.output_key, json.dumps(dict(summary=summary, clips=results), default=float).encode())
    for t in summary["tests"]:
        print(f"{t['feature']:16s} {t['score']:22s} AUC {t['auc']:.3f} CI {t['ci95']}  n+={t['positives']}")
    print(json.dumps(summary["distributions"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
