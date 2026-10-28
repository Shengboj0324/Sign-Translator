"""Where does the generator lose sentence meaning? The sentence-level exposure diagnostic (2026-10-28).

The pilot's meaning half (``pilot_content_check.py``) finds no sentence content in generated motion: on 300 How2Sign
validation sentences the 2D back-translator reads generated 1.81 BLEU, real 4.14 and blank (durations only) 1.86. This
script reads the same sentences (the pilot's ``validation_sentences``: three validation signer groups, seed 0,
``--clips``) in forms that walk from the real clip to the free run, so the step where content disappears can be named.
The word diagnostic (``exposure_diagnostic.py``) does this per word with the recogniser; its code paths are reused
unchanged (lifted to module level there), so both diagnostics teacher-force, sample, decode and post-process alike.

**Forms** (every generated or decoded form is post-processed as generations are: the tokenizer's accepted decoder
gains, then head rotation and head translation smoothed over 0.1 s):
- ``real``: the reference clip as the pilot reads it (``load_state`` at 24 fps), and ``blank`` (``blanked``: every
  channel removed, only the duration left: what the translator's text prior gives);
- ``real_dataset``: the clip as the prior's dataset serves it (unadmitted channels masked, conformed to the How2Sign
  layout); every tokenizer form below starts from it;
- ``recon_full`` / ``recon_masked``: its tokenizer codes decoded in the generator's form (every channel valid), and the
  same with each channel's validity copied from the real clip (instrument parity: the back-translator reads per-point
  "observed" flags, and real How2Sign hands drop out);
- ``onestep_sample`` / ``onestep_argmax``: every step's codes drawn from (or the mode of) the prior given the REAL
  previous codes, the text and the clip's own loci, no guidance, temperature 1 (``teacher_forced``);
- ``free_run``: the pilot's generated form, sampled exactly as the pilot samples (``generate_rvq_motion_batch``, the
  pilot's part temperatures and guidance, batches of 32, seeds 1000 + i, the reference duration);
- ``prefixNN``: the real first NN% of the codes, then the prior continues to the reference length (``continue_from``,
  guidance and part temperatures as ``free_run``, the text's planned loci, seed 9600 + 10 i);
- ``prefixNN_blank``: the same state with every frame from the prefix boundary on removed (the prefix frames are
  identical to ``prefixNN``'s); ``holdNN``: the real codes to the boundary, then the last real step held.
  A prefix form shows continued content only where it reads above both.

**Scores.** Every form is back-translated (``score_back_translation.translate``, beam 4) and scored by corpus BLEU with a
bootstrap 95% interval over sentences (``strict_prior_review.bleu_ci``'s resampling, seed 0, 1000 draws; computed from
per-sentence n-gram counts, which gives bleu_ci's numbers exactly: tested). Controls per form:
- ``mismatched``: its translations against another clip's reference (one seeded derangement of the references, no clip
  keeps its own): the chance level of that form's output distribution;
- paired differences (same resampled sentences on both sides) against its mismatched score, against blank, and for
  prefix forms against their ``_blank`` and ``hold`` controls.
**Tails.** The whole-clip BLEU of a prefix form is inflated by its real prefix. Each form is therefore also scored on
its continued portion: frames from the boundary on, back-translated alone, against the reference's last words in the
same proportion (``tail_words``; an approximation: it assumes the English words are signed in order at a uniform rate,
which ASL word order and free translation break). Because the approximation is the same for every form, tails are read
only against the real clip's own tail, the blank tail and the tails' mismatched control, never as absolute BLEU.

**Motion statistics** of each form against real: in the back-translator's own 2D view (FK, front projection, shoulder
units; ``keypoint_statistics``): wrist speed, fraction of frames with tracked hands, wrist and handshape spread over
time, and the share of wrist power in frequency bands; in rotation space ``motion_metrics.compare`` (speed and
variation ratios per channel, Fréchet distance), ``high_frequency_rms`` of the hands and the word diagnostic's
``activity``. **Repetition** of every code sequence (real, one-step, free run, prefix tails; ``code_repetition``): how
often stage-1 codes recur at lags 1-48 steps (a categorical autocorrelation), distinct code rows and run lengths.

Writes one JSON report (``--output-key`` to the store, or ``--output-file``; both write-once) and prints a summary.
"""
from __future__ import annotations

import argparse
import collections
from dataclasses import replace
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

TRANSLATOR_RUN = "back-translator-kp2d-uwb-ft-v1"       # the pilot's (R2b) instrument
FRACTIONS = (.25, .5, .75)
LAGS = (1, 2, 4, 8, 12, 24, 48)                          # steps (= frames at 24 fps)
BANDS = ((0., 1.), (1., 3.), (3., 6.), (6., 12.5))       # Hz; 12 Hz is Nyquist at 24 fps
WRISTS = (4, 5)                                          # keypoints2d.POINTS: LeftWrist, RightWrist
HAND_ROOTS = (6, 27)                                     # the first point (wrist) of each hand's 21
TAIL_FORMS = ("real", "blank", "recon_full", "onestep_sample", "free_run", "prefix", "hold")

CAVEATS = [
    "Instrument: back-translator-kp2d-uwb-ft-v1 was pretrained on UWB YouTube-ASL MediaPipe keypoints and fine-tuned on "
    "How2Sign FK keypoints of REAL states (build_fk_keypoint_shards.py: load_state, 24 fps, from_state, normalise). It "
    "never saw tokenizer reconstructions or generated motion; its checkpoint was selected on How2Sign val BLEU, the "
    "partition these clips come from, so real is if anything optimistic.",
    "It reads arms and hands only (48 2D points, nominal bone lengths, no face): head, face and mouthing are not scored.",
    "It reads per-point observed flags. Decoded and generated forms have every hand valid on every frame, real How2Sign "
    "hands drop out: recon_masked vs recon_full measures that shift.",
    "BLEU is near its floor here (real 4.14 vs blank 1.86 on these 300 clips in the pilot); read paired differences, not "
    "point gaps. Blank is not zero: it is the translator's text prior given the duration.",
    "onestep forms: no guidance, temperature 1 and the REAL clip's loci (F12 is locus-conditioned), while free run and "
    "prefixes use guidance, part temperatures and loci planned from the text. They differ in sampler settings as well as "
    "in history.",
    "history_stages=1 priors read only the stage-1 codes of the history: a real prefix passes its coarse codes to the "
    "prior; its finer stages only shape the decoded prefix frames.",
    "Prefix boundary: the decoder is convolutional, so the last prefix frames also depend on the first generated codes; "
    "prefixNN_blank keeps exactly the prefix form's own prefix frames.",
    "Tails: tail_words assumes the English words are signed in order at a uniform rate (not true of ASL or of free "
    "translation); tails are short, so the per-clip normalisation and BLEU-4 are noisier. Read tails against real, blank "
    "and mismatched tails only.",
    "free_run reproduces the pilot's generated samples when checkpoint, clips and device arithmetic agree; prefix "
    "continuations are sampled one clip at a time (the word diagnostic's path) with other seeds: same distribution "
    "given the prefix, different draws.",
    "Clips longer than the prior's max_seconds were never seen in training (reported as long_clips).",
]


# ---- pure logic --------------------------------------------------------------------------------------------------

def derangement(n: int, seed: int = 0) -> list[int]:
    """A uniformly random permutation of range(n) with no fixed point (rejection sampling), n >= 2."""
    if n < 2:
        raise ValueError("a derangement needs at least two items")
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return [int(x) for x in p]


def tail_words(text: str, fraction: float) -> str:
    """The reference's words after its first ``fraction`` (at least its last word): the approximate English of the
    clip's last 1 - fraction of time, assuming the words are signed in order at a uniform rate."""
    words = text.split()
    if not words:
        return ""
    return " ".join(words[min(int(round(fraction * len(words))), len(words) - 1):])


def tail_start(fraction: float, frames: int) -> int:
    """First frame of the continued portion: the prefix boundary (``prefix_steps``), at most frames - 1."""
    from exposure_diagnostic import prefix_steps
    return min(prefix_steps(fraction, frames), frames - 1)


def frames_from(state, k: int):
    """Frames k.. of ``state`` as a new, validated state (timestamps kept)."""
    from signtranslator.pose.multichannel import MultichannelMotion
    channels = {n: replace(c, values=c.values[k:], valid=c.valid[k:], observed=c.observed[k:],
                           confidence=c.confidence[k:]) for n, c in state.channels.items()}
    return MultichannelMotion(np.ascontiguousarray(state.timestamps[k:]), state.clock_id, state.sample_id, channels)


def blank_from(state, k: int):
    """``state`` with every channel removed from frame k on; frames before k are untouched (a prefix, then blank)."""
    from signtranslator.pose.multichannel import MultichannelMotion
    channels = {}
    for n, c in state.channels.items():
        valid = c.valid.copy()
        valid[k:] = False
        channels[n] = replace(c, valid=valid, observed=c.observed & valid,
                              values=np.where(valid[..., None], c.values, 0.).astype(c.values.dtype),
                              confidence=np.where(valid, c.confidence, 0.).astype(c.confidence.dtype))
    return MultichannelMotion(state.timestamps, state.clock_id, state.sample_id, channels)


def bleu_stats(hypotheses, references, max_n: int = 4) -> np.ndarray:
    """(N, 2 max_n + 2) int64 per-sentence terms of ``motion_to_text.corpus_bleu``: clipped n-gram matches and n-gram
    totals for n = 1..max_n, hypothesis and reference length (same tokenisation, same counting)."""
    from signtranslator.generation.motion_to_text import words
    out = np.zeros((len(hypotheses), 2 * max_n + 2), np.int64)
    for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
        h, r = words(hyp), words(ref)
        for n in range(1, max_n + 1):
            hc = collections.Counter(tuple(h[j:j + n]) for j in range(len(h) - n + 1))
            rc = collections.Counter(tuple(r[j:j + n]) for j in range(len(r) - n + 1))
            out[i, n - 1] = sum(min(c, rc[g]) for g, c in hc.items())
            out[i, max_n + n - 1] = max(len(h) - n + 1, 0)
        out[i, 2 * max_n], out[i, 2 * max_n + 1] = len(h), len(r)
    return out


def bleu_from_stats(total: np.ndarray, max_n: int = 4) -> float:
    """``corpus_bleu``'s ``bleu`` from summed per-sentence terms (the same arithmetic, in the same order)."""
    matches, totals = [int(x) for x in total[:max_n]], [int(x) for x in total[max_n:2 * max_n]]
    hyp_len, ref_len = int(total[2 * max_n]), int(total[2 * max_n + 1])
    precisions = [m / t if t else 0. for m, t in zip(matches, totals)]
    if hyp_len == 0 or min(precisions) == 0:
        return 0.
    bp = 1. if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100 * bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def _draws(n_items: int, seed: int, n: int):
    """``strict_prior_review.bleu_ci``'s bootstrap index draws, in its order."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield rng.integers(0, n_items, n_items)


def bleu_ci(hypotheses, references, *, seed: int = 0, n: int = 1000):
    """(point, [2.5%, 97.5%]) exactly as ``strict_prior_review.bleu_ci``, from per-sentence counts (fast)."""
    stats = bleu_stats(hypotheses, references)
    point = bleu_from_stats(stats.sum(0))
    vals = [bleu_from_stats(stats[idx].sum(0)) for idx in _draws(len(stats), seed, n)]
    return round(float(point), 3), [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def paired_bleu_difference(hyps_a, refs_a, hyps_b, refs_b, *, seed: int = 0, n: int = 1000) -> dict:
    """BLEU(a) - BLEU(b) on the same sentences, with the bootstrap interval of the difference: each draw resamples the
    sentence indices once (``bleu_ci``'s draws, so each side's distribution is its own ``bleu_ci``'s) and scores both
    sides on them. ``p_not_above`` is the fraction of draws where a does not exceed b."""
    if not len(hyps_a) == len(refs_a) == len(hyps_b) == len(refs_b):
        raise ValueError("paired forms need the same sentences")
    a, b = bleu_stats(hyps_a, refs_a), bleu_stats(hyps_b, refs_b)
    point = bleu_from_stats(a.sum(0)) - bleu_from_stats(b.sum(0))
    diffs = np.array([bleu_from_stats(a[idx].sum(0)) - bleu_from_stats(b[idx].sum(0)) for idx in _draws(len(a), seed, n)])
    return dict(difference=round(float(point), 3),
                ci95=[round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)],
                p_not_above=round(float((diffs <= 0).mean()), 4))


def code_repetition(codes, parts: int, lags=LAGS) -> dict:
    """How much a (S, J) code sequence repeats itself, on the stage-1 codes of every part (columns :parts):
    ``lag_match[L]``: the fraction of (step, part) pairs equal L steps later (a categorical autocorrelation; None when
    S <= L); ``distinct_rows``: distinct stage-1 rows over S; ``mean_run``: mean run length of identical stage-1 rows;
    ``full_lag1``: the fraction of steps whose every code equals the previous step's."""
    c = np.asarray(codes)
    if c.ndim != 2 or len(c) == 0:
        raise ValueError("codes must be a nonempty (S, J) array")
    c1 = c[:, :parts]
    S = len(c1)
    lag_match = {str(L): (round(float((c1[L:] == c1[:-L]).mean()), 4) if S > L else None) for L in lags}
    changes = int((c1[1:] != c1[:-1]).any(1).sum()) if S > 1 else 0
    return dict(lag_match=lag_match, distinct_rows=round(len({tuple(r) for r in c1.tolist()}) / S, 4),
                mean_run=round(S / (1 + changes), 3),
                full_lag1=(round(float((c[1:] == c[:-1]).all(1).mean()), 4) if S > 1 else None), steps=S)


def keypoint_statistics(points: np.ndarray, valid: np.ndarray, fps: float) -> dict:
    """Statistics of a clip in the back-translator's own view: normalised 2D keypoints (``keypoints2d.normalise``,
    shoulder units, hands local) and their validity. None where a statistic has no valid frames.
    - ``wrist_speed``: mean wrist displacement per second over frame pairs valid in both (both wrists pooled);
    - ``hands_tracked``: mean over both hands of the fraction of frames whose hand points are valid;
    - ``wrist_spread``: temporal standard deviation of each wrist's position over its valid frames (mean of x, y and
      both wrists): where in signing space the hands travel;
    - ``handshape_spread``: the same for each hand's 20 wrist-relative points: how much the handshape changes;
    - ``wrist_band_share``: per band of ``BANDS``, the share of a wrist's mean-removed power (valid frames
      concatenated, as ``motion_metrics.high_frequency_rms``; wrists with at least 16 valid frames), both wrists averaged."""
    speeds, spreads, shapes, shares = [], [], [], []
    for w in WRISTS:
        ok = valid[1:, w] & valid[:-1, w]
        if ok.any():
            speeds.append(np.linalg.norm(points[1:, w] - points[:-1, w], axis=-1)[ok] * fps)
        seen = valid[:, w]
        if seen.sum() >= 2:
            spreads.append(float(points[seen, w].std(0).mean()))
        if seen.sum() >= 16:
            x = points[seen, w] - points[seen, w].mean(0)
            power = (np.abs(np.fft.rfft(x, axis=0)) ** 2).sum(-1)
            f = np.fft.rfftfreq(len(x), 1. / fps)
            power, f = power[1:], f[1:]                       # the mean is removed: no DC
            if power.sum() > 0:
                shares.append([float(power[(f >= lo) & (f < hi)].sum() / power.sum()) for lo, hi in BANDS])
    for a in HAND_ROOTS:
        seen = valid[:, a + 1]
        if seen.sum() >= 2:
            shapes.append(float(points[seen, a + 1:a + 21].std(0).mean()))
    return dict(wrist_speed=float(np.concatenate(speeds).mean()) if speeds else None,
                hands_tracked=float(np.mean([valid[:, a].mean() for a in HAND_ROOTS])),
                wrist_spread=float(np.mean(spreads)) if spreads else None,
                handshape_spread=float(np.mean(shapes)) if shapes else None,
                wrist_band_share=[float(v) for v in np.mean(shares, 0)] if shares else None)


def mean_statistics(rows: list[dict]) -> dict:
    """Mean over clips of every statistic (Nones skipped; lists averaged elementwise), with the clip count used."""
    out = {}
    for key in dict.fromkeys(k for r in rows for k in r):         # every key of any row, in first-seen order
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            out[key] = None
        elif isinstance(vals[0], list):
            out[key] = [round(float(v), 4) for v in np.mean(vals, 0)]
        elif isinstance(vals[0], dict):
            out[key] = mean_statistics(vals)
        else:
            out[key] = round(float(np.mean(vals)), 4)
        out[f"{key}_clips"] = len(vals)
    return out


def ratio_to(form: dict, real: dict) -> dict:
    """Each scalar mean statistic of ``form`` over the same statistic of ``real``."""
    return {k: (round(v / real[k], 4) if isinstance(v, (int, float)) and isinstance(real.get(k), (int, float))
                and real[k] and not k.endswith("_clips") else None)
            for k, v in form.items() if not k.endswith("_clips") and not isinstance(v, (list, dict))}


def score_forms(hypotheses: dict, references: list[str], permutation: list[int], *, boot: int, pairs: dict) -> dict:
    """Per form: BLEU with its interval, the mismatched control (references permuted by ``permutation``), the paired
    difference to it, to ``blank`` and to the forms named in ``pairs`` (form -> {label: other form})."""
    from strict_prior_review import separated
    mismatched = [references[j] for j in permutation]
    out = {}
    for name, hyps in hypotheses.items():
        point, ci = bleu_ci(hyps, references, n=boot)
        m_point, m_ci = bleu_ci(hyps, mismatched, n=boot)
        row = dict(bleu=point, ci95=ci, mismatched=dict(bleu=m_point, ci95=m_ci),
                   minus_mismatched=paired_bleu_difference(hyps, references, hyps, mismatched, n=boot))
        if name != "blank" and "blank" in hypotheses:
            row["minus_blank"] = paired_bleu_difference(hyps, references, hypotheses["blank"], references, n=boot)
        for label, other in pairs.get(name, {}).items():
            if other in hypotheses:
                row[f"minus_{label}"] = paired_bleu_difference(hyps, references, hypotheses[other], references, n=boot)
        out[name] = row
    if "blank" in out:
        for name, row in out.items():
            if name != "blank":
                row["above_blank"] = separated(row["ci95"], out["blank"]["ci95"])
    for name, row in out.items():
        row["hypotheses"] = list(hypotheses[name])
    return out


# ---- the run -----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-key", required=True)
    p.add_argument("--mix-key", required=True, help="the manifest the prior was trained on (its How2Sign validation "
                   "records are the pilot's clips)")
    p.add_argument("--mix-sha256", required=True)
    p.add_argument("--clips", type=int, default=300)
    p.add_argument("--part-temperature", default="body=1.1,left_hand=1.1,right_hand=1.1,face=1.0",
                   help="free run and prefix continuations (the pilot's default)")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS), help="prefix fractions")
    p.add_argument("--seed", type=int, default=0, help="one-step draws (as exposure_diagnostic) and the derangement")
    p.add_argument("--translator-run", default=TRANSLATOR_RUN)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
    p.add_argument("--cache-dir", default=str(Path(tempfile.gettempdir()) / "sign-translator-cache"))
    p.add_argument("--output-key", help="the store key of the JSON report (write-once)")
    p.add_argument("--output-file", help="a local path for the JSON report instead (write-once)")
    args = p.parse_args(argv)
    if not (args.output_key or args.output_file):
        p.error("give --output-key or --output-file")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from signtranslator.data.keypoints2d import from_state, normalise
    from signtranslator.data.text_motion_dataset import TextMotionDataset
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest
    from signtranslator.generation.motion_metrics import compare, high_frequency_rms
    from signtranslator.generation.motion_rvq import MotionRVQ, RVQPrior, RVQPriorConfig, generate_rvq_motion_batch
    from evaluate_text_to_motion import accepted_decoder_gains, at_rate, blanked, load_state
    from exposure_diagnostic import (activity, continue_from, held_codes, make_decode, make_post, masked_like,
                                     prefix_steps, teacher_forced)
    from pilot_content_check import MIX_SHA, validation_sentences
    from score_back_translation import load_translator, translate
    from tokenizer_fidelity_check import observed_like
    from train_motion_tokens import layout_template
    from train_text_to_motion import make_store
    clock = {"start": time.time()}
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    fractions = [float(f) for f in args.fractions.split(",")]
    if not all(0 < f < 1 for f in fractions):
        raise SystemExit("--fractions must lie in (0, 1)")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    ck = torch.load(io.BytesIO(store.get(args.checkpoint_key)), map_location="cpu", weights_only=False)
    if ck.get("kind") != "motion_rvq_prior" or ck.get("manifest_sha256") != args.mix_sha256:
        raise SystemExit(f"{args.checkpoint_key} is not an RVQ prior trained on {args.mix_sha256}")
    prior = RVQPrior(RVQPriorConfig.from_dict(ck["config"])); prior.load_state_dict(ck.get("ema") or ck["model"])
    tokenizer = MotionRVQ(prior.config.tokenizer); tokenizer.load_state_dict(ck["tokenizer_state"])
    device = torch.device(args.device)
    prior.to(device).eval(); tokenizer.to(device).eval()
    gains = accepted_decoder_gains(store, ck.get("tokenizer_key"))
    motion_config = prior.config.tokenizer.motion
    post = make_post(gains)
    decode = make_decode(tokenizer, motion_config, post)
    temperature = {k: float(v) for k, v in (x.split("=") for x in args.part_temperature.split(","))}
    P = len(tokenizer.parts)

    # the pilot's clips, references and durations
    val = validation_sentences(manifest, args.clips)
    texts = [r["text"] for r in val]
    refs = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), 24.) for r in val]
    seconds = [float(s.timestamps[-1] - s.timestamps[0]) + 1 / 24. for s in refs]
    if min(len(s.timestamps) for s in refs) < 4:
        raise SystemExit("a clip has fewer than 4 frames: no prefix form is defined")
    cache = ContentCache(Path(args.cache_dir), 8 * 2 ** 30)
    template = layout_template(manifest, "how2sign-two-view-text-v1", cache, store.get)
    data = TextMotionDataset(manifest, partition="val", cache=cache, fetch=store.get, max_seconds=None, fps=24.,
                             template=template)
    by_sign = {r["sign"]: k for k, r in enumerate(data.records)}
    missing = [r["sign"] for r in val if r["sign"] not in by_sign]
    if missing:
        raise SystemExit(f"{len(missing)} clips have no usable continuous state in the dataset: {missing[:5]}")
    print(f"{len(val)} clips, {sum(seconds):.0f} s of reference motion", flush=True)

    forms = collections.defaultdict(list)
    codes_of = collections.defaultdict(list)
    # free run: the pilot's generated form, the pilot's batches and seeds
    for i in range(0, len(texts), 32):
        states, codes = generate_rvq_motion_batch(prior, tokenizer, texts[i:i + 32], fps=24., durations=seconds[i:i + 32],
                                                  guidance=args.guidance, temperature=temperature, top_k=0,
                                                  seed=1000 + i, return_codes=True)
        forms["free_run"] += [post(s) for s in states]
        codes_of["free_run"] += codes
    clock["free_run"] = time.time()
    print(f"free run done ({clock['free_run'] - clock['start']:.0f} s)", flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    rows = []
    tags = {f: f"{int(round(100 * f))}" for f in fractions}
    for i, r in enumerate(val):
        text, real = data[by_sign[r["sign"]]]
        if text != texts[i]:
            raise SystemExit(f"{r['sign']}: the dataset serves another text")
        motion, codes, drawn, modes = teacher_forced(prior, tokenizer, motion_config, real, text, generator, device)
        frames = int(motion.frame_valid.shape[1])
        with torch.no_grad():
            recon = decode(codes, frames, text)
            forms["real_dataset"].append(real)
            forms["recon_full"].append(recon)
            forms["recon_masked"].append(masked_like(recon, real))
            forms["onestep_sample"].append(decode(drawn[None], frames, text))
            forms["onestep_argmax"].append(decode(modes[None], frames, text))
        codes_of["real"].append(codes[0].cpu())
        codes_of["onestep_sample"].append(drawn.cpu())
        codes_of["onestep_argmax"].append(modes.cpu())
        for f in fractions:
            k, tag = prefix_steps(f, frames), tags[f]
            state, sequence = continue_from(prior, tokenizer, decode, text, codes[:, :k], frames, 9600 + 10 * i,
                                            guidance=args.guidance, device=device, temperature=temperature,
                                            return_codes=True)
            forms[f"prefix{tag}"].append(state)
            forms[f"prefix{tag}_blank"].append(blank_from(state, k))
            forms[f"hold{tag}"].append(decode(held_codes(codes, k, frames), frames, text))
            codes_of[f"prefix{tag}_tail"].append(sequence[k:])
            codes_of[f"real_tail{tag}"].append(codes[0, k:].cpu())
        rows.append(dict(sign=r["sign"], text=text, frames=frames, frames_reference=len(refs[i].timestamps),
                         frames_free_run=len(forms["free_run"][i].timestamps),
                         onestep_code_agreement=round(float((drawn == codes[0]).float().mean()), 4),
                         argmax_code_agreement=round(float((modes == codes[0]).float().mean()), 4),
                         onestep_stage1_agreement=round(float((drawn[:, :P] == codes[0, :, :P]).float().mean()), 4)))
        if (i + 1) % 10 == 0 or i + 1 == len(val):
            print(f"clip {i + 1}/{len(val)} ({time.time() - clock['start']:.0f} s)", flush=True)
    clock["forms"] = time.time()
    forms["real"], forms["blank"] = refs, blanked(refs)

    # back-translation, as the pilot reads
    kp = json.loads(store.get(f"training-runs/{args.translator_run}/best.json"))["checkpoint_key"]
    translator, kind = load_translator(store, kp, MIX_SHA)                       # the translator's own manifest
    translator = translator.to(device)
    fps = float(getattr(translator.config, "fps", 24.))
    order = ["real", "blank", "real_dataset", "recon_full", "recon_masked", "onestep_sample", "onestep_argmax",
             "free_run"] + [f"{kind_}{tags[f]}{suffix}" for f in fractions
                            for kind_, suffix in (("prefix", ""), ("prefix", "_blank"), ("hold", ""))]
    forms = {name: forms[name] for name in order}
    hyps = {name: list(translate(translator, kind, states)) for name, states in forms.items()}
    tails, tail_refs = {}, {}
    for f in fractions:
        tag = tags[f]
        tail_refs[tag] = [tail_words(t, f) for t in texts]
        tails[tag] = {}
        for name in TAIL_FORMS:
            source = {"prefix": f"prefix{tag}", "hold": f"hold{tag}"}.get(name, name)
            states = [frames_from(s, tail_start(f, len(s.timestamps))) for s in forms[source]]
            tails[tag][name] = list(translate(translator, kind, states))
    clock["translation"] = time.time()
    print(f"back-translation done ({clock['translation'] - clock['forms']:.0f} s)", flush=True)

    permutation = derangement(len(texts), args.seed)
    pairs = {f"prefix{t}": {"prefix_blank": f"prefix{t}_blank", "hold": f"hold{t}"} for t in tags.values()}
    pairs.update({"recon_full": {"real_dataset": "real_dataset"}, "free_run": {"onestep_sample": "onestep_sample"}})
    whole = score_forms(hyps, texts, permutation, boot=args.bootstrap, pairs=pairs)
    tail_scores = {tag: score_forms(tails[tag], tail_refs[tag], permutation, boot=args.bootstrap,
                                    pairs={"prefix": {"free_run": "free_run", "hold": "hold", "real": "real"}})
                   for tag in tails}
    clock["scoring"] = time.time()

    # motion statistics
    def view(state):
        points, valid = from_state(at_rate(state, fps))
        return normalise(points, valid, y_down=False), valid
    keypoint = {name: mean_statistics([keypoint_statistics(*view(s), fps) for s in states])
                for name, states in forms.items()}
    rotation, rotation_masked, hf, acts = {}, {}, {}, {}
    for name, states in forms.items():
        acts[name] = mean_statistics([activity(s) for s in states])
        hf[name] = {ch: mean_statistics([{"rms": high_frequency_rms(s, ch)} for s in states])["rms"]
                    for ch in ("left_hand", "right_hand")}
        if name in ("real", "blank"):
            continue
        rotation[name] = {k: ({m: round(float(x), 4) for m, x in v.items()} if isinstance(v, dict) else round(float(v), 4))
                          for k, v in compare(states, refs).items()}
        masked = [observed_like(s, ref) if len(s.timestamps) == len(ref.timestamps) else s for s, ref in zip(states, refs)]
        rotation_masked[name] = {k: ({m: round(float(x), 4) for m, x in v.items()} if isinstance(v, dict)
                                     else round(float(v), 4)) for k, v in compare(masked, refs).items()}
    repetition = {name: mean_statistics([code_repetition(c, P) for c in seqs if len(c)]) for name, seqs in codes_of.items()}
    clock["statistics"] = time.time()

    long_clips = int(sum(s > motion_config.max_seconds for s in seconds)) if motion_config.max_seconds else 0
    report = dict(
        checkpoint_key=args.checkpoint_key, mix_key=args.mix_key, mix_sha256=args.mix_sha256,
        config_sha256=ck.get("config_sha256"), tokenizer_key=ck.get("tokenizer_key"), translator_key=kp,
        translator_run=args.translator_run, clips=len(texts), guidance=args.guidance, part_temperature=temperature,
        fractions=fractions, seed=args.seed, bootstrap=args.bootstrap, beam=4, decoder_gains=gains,
        history_stages=prior.config.history_stages, locus_conditioning=prior.config.locus_conditioning,
        max_seconds=motion_config.max_seconds, long_clips=long_clips,
        frame_mismatches=dict(reference_vs_dataset=sum(r["frames"] != r["frames_reference"] for r in rows),
                              reference_vs_free_run=sum(r["frames_free_run"] != r["frames_reference"] for r in rows)),
        code_agreement={k: round(float(np.mean([r[k] for r in rows])), 4)
                        for k in ("onestep_code_agreement", "argmax_code_agreement", "onestep_stage1_agreement")},
        references=texts, derangement=permutation, whole=whole,
        tails={tag: dict(references=tail_refs[tag], forms=tail_scores[tag]) for tag in tails},
        motion=dict(keypoint=keypoint, keypoint_ratio_to_real={n: ratio_to(v, keypoint["real"]) for n, v in keypoint.items()},
                    rotation_vs_real=rotation, rotation_vs_real_on_real_frames=rotation_masked,
                    hands_high_frequency_rms=hf, activity=acts),
        code_repetition=repetition,
        seconds={k: round(v - clock["start"], 1) for k, v in clock.items() if k != "start"},
        caveats=CAVEATS, per_clip=rows)
    summary(report)
    payload = json.dumps(report, indent=1, default=float).encode()        # keys in reading order
    if args.output_file:
        with open(args.output_file, "xb") as stream:                     # write-once
            stream.write(payload)
    if args.output_key:
        if store.exists(args.output_key):
            raise SystemExit(f"{args.output_key} appeared during the run: refusing to overwrite")
        store.put(args.output_key, payload)
    return 0


def summary(report: dict) -> None:
    """The printed summary: BLEU and controls per form, whole clip and tails, then the main motion statistics."""
    def line(name, row):
        mm, mb = row["minus_mismatched"], row.get("minus_blank")
        extra = "".join(f"  {k[6:]} {v['difference']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
                        for k, v in row.items() if k.startswith("minus_") and k not in ("minus_mismatched", "minus_blank"))
        return (f"  {name:22s} {row['bleu']:6.2f} [{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}]  mismatched "
                f"{row['mismatched']['bleu']:5.2f}  -mism {mm['difference']:+.2f} [{mm['ci95'][0]:+.2f}, {mm['ci95'][1]:+.2f}]"
                + (f"  -blank {mb['difference']:+.2f} [{mb['ci95'][0]:+.2f}, {mb['ci95'][1]:+.2f}]" if mb else "") + extra)
    print(f"whole clips ({report['clips']}):", flush=True)
    for name, row in report["whole"].items():
        print(line(name, row), flush=True)
    for tag, t in report["tails"].items():
        print(f"tails from {tag}%:", flush=True)
        for name, row in t["forms"].items():
            print(line(name, row), flush=True)
    keys = ("wrist_speed", "hands_tracked", "wrist_spread", "handshape_spread")
    print("keypoint view (ratio to real): " + ", ".join(keys), flush=True)
    for name, ratio in report["motion"]["keypoint_ratio_to_real"].items():
        print(f"  {name:22s} " + "  ".join(f"{ratio.get(k) if ratio.get(k) is not None else float('nan'):6.3f}"
                                           for k in keys), flush=True)
    print("stage-1 code recurrence at lags " + ",".join(str(L) for L in LAGS), flush=True)
    for name, rep in report["code_repetition"].items():
        lm = rep.get("lag_match") or {}
        print(f"  {name:22s} " + " ".join(f"{lm.get(str(L)) if lm.get(str(L)) is not None else float('nan'):.3f}"
                                           for L in LAGS) + f"  distinct {rep.get('distinct_rows')}", flush=True)
    print(json.dumps(dict(code_agreement=report["code_agreement"], frame_mismatches=report["frame_mismatches"],
                          long_clips=report["long_clips"], seconds=report["seconds"])), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
