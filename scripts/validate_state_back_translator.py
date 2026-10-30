"""Instrument v2 acceptance: the declared gates G1-G6 (docs/NEXT_PHASE_PLAN.md, "Instrument v2: design and declared
gates", declared 2026-10-30 09:10 UTC, before any code). Exit 0 if every gate holds, 4 if any fails (a verdict, as the
other acceptance scripts), 2 if a gate could not be evaluated (missing input).

**Clips:** the pilots' 1,000 How2Sign val clips (``pilot_content_check.validation_sentences(--mix-key, --clips)``,
read as the pilot reads them: ``load_state``, 24 fps). Every form is read by the instrument with the pilot's
``translate`` (beam 4, batches of 32).

**Controls** (``build_controls``; each at the reference clip's duration):
- ``blank``: every channel invalid (``evaluate_text_to_motion.blanked``);
- ``blank_median``: blank at one fixed duration, the median of the clips' frame counts (one input for every clip);
- ``static_valid``: the first real frame held for the clip's length (``round15_compose_eval.held``) with every channel
  the instrument reads valid (each element's first valid value; the neutral value where never valid): the
  generator's form with no motion;
- ``mismatched``: a derangement (``sentence_exposure_diagnostic.derangement``, ``--seed``): clip B's real motion
  time-stretched to clip A's frame count (``compositional.time_scale``), scored against A's sentence.
Report-only: ``shuffled_chunks`` (each clip's 0.5 s chunks in a random order) and, per form, the length ratio
(hypothesis words / reference words).

**Every gate is decided on unrounded values**; the stored report is rounded to 4 decimals afterwards. Clips longer
than 20 s (the training cut; selection reads them in full, greedily) are reported separately (``long_clips``).

**Bootstrap:** single forms ``strict_prior_review.bleu_ci``'s draws (1,000 resamples, seed 0); every paired difference
``compare_pilot_lexicon.paired_bleu_difference`` (2,000 resamples of clips, seed 0); the gap over the strongest
control (``paired_gap``) resamples the same clips the same way and takes the maximum control in each resample.

**Gates** (all must hold):
- **G1 strength:** real >= 6.20 and real's CI lower bound > 4.136.
- **G2 gap:** real - max(blank, static_valid, mismatched) >= 4.55 and its paired CI lower bound > 2.277. Reported: the
  gap over the W14b - R16C paired half-width, against v1's (from ``--reread-key``).
- **G3 no same-space bias:** every generated arm of ``--reread-key`` (W14b, R16, R16C, composition) reads below real
  (paired upper bound < 0), and the d2 tokenizer round trip of the clips (``build_recon_states``'s form) reads at
  least 0.8 x real.
- **G4 cross-domain:** 2M-Flores (``--flores-manifest``) real above its own blank and mismatched controls (paired
  lower bounds > 0; the unpaired interval separation is reported too).
- **G5 ranking** (amendment 1, 09:56 UTC): from ``--reread-key``, W14b, R16, R16C, composition, real and blank are
  read by both v1 and v2 on the same regenerated samples (``v1_rerun`` and ``v2``); every ordering v1 resolves on those
  samples (paired CI excluding 0) keeps its sign under v2 (the point difference); Kendall's tau of the six BLEUs
  reported. Exact reproduction of v1's stored pilot hypotheses is reported as a diagnostic, not required.
- **G6 per-signer:** each How2Sign val signer group (c01, c03, c05) reads real above blank (paired lower bound > 0).

The test partitions and How2Sign signer 9 are never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

V1_REAL, V1_GAP = 4.136, 2.277                       # v1 on the same 1,000 clips (declared)
G1_REAL, G2_GAP, G3_ROUND_TRIP = 6.20, 4.55, .8
GAP_CONTROLS = ("blank", "static_valid", "mismatched")
GROUPS = ("c01", "c03", "c05")
ARMS = ("W14b", "R16", "R16C", "composition")
RANKED = ARMS + ("real", "blank")
CHUNK_SECONDS = .5
FPS = 24.


# ---- controls -------------------------------------------------------------------------------------------------

def blank_states(states):
    from evaluate_text_to_motion import blanked
    return blanked(states)


def static_valid(state, channels):
    """``state``'s first frame held for its length (``round15_compose_eval.held``), every channel in ``channels``
    valid on every frame: each element takes its first valid value, or the neutral value (identity rotation, forward
    gaze, zero translation and coefficient, blink 0.5) where it is never valid. Other channels keep their first frame."""
    from dataclasses import replace
    from round15_compose_eval import held
    from signtranslator.generation.text_to_motion import _FORWARD, _IDENTITY_6D
    from signtranslator.pose.multichannel import CHANNELS
    neutral = {"rotation_6d_columns": np.asarray(_IDENTITY_6D), "unit_vector": np.asarray(_FORWARD)}
    out = {}
    for name, c in state.channels.items():
        if name not in channels:
            out[name] = c
            continue
        kind, width, _ = CHANNELS[name]
        values, valid = c.values.copy(), c.valid.copy()
        for j in range(values.shape[1]):
            hit = np.flatnonzero(c.valid[:, j])
            values[0, j] = (c.values[hit[0], j] if len(hit) else
                            neutral.get(kind, np.full(width, .5 if kind == "bounded_scalar" else 0.)))
        valid[0] = True
        out[name] = replace(c, values=np.where(valid[..., None], values, 0.).astype(c.values.dtype), valid=valid,
                            observed=valid.copy(), confidence=np.where(valid, 1., 0.).astype(c.confidence.dtype))
    return held(replace(state, channels=out))


def mismatched(states, seed: int):
    """(states, derangement): clip ``perm[i]``'s motion time-stretched to clip i's frame count."""
    from sentence_exposure_diagnostic import derangement
    from signtranslator.generation.compositional import time_scale
    perm = derangement(len(states), seed)
    return [time_scale(states[j], len(states[i].timestamps)) for i, j in enumerate(perm)], perm


def shuffled_chunks(state, seed: int, index: int, fps: float = FPS):
    """The clip's ``CHUNK_SECONDS`` chunks in a random order (``np.random.default_rng([seed, index])``)."""
    from signtranslator.generation.compositional import take_frames
    T = len(state.timestamps)
    size = max(1, int(round(CHUNK_SECONDS * fps)))
    chunks = [np.arange(a, min(a + size, T)) for a in range(0, T, size)]
    order = np.random.default_rng([seed, index]).permutation(len(chunks))
    return take_frames(state, np.concatenate([chunks[k] for k in order]))


def blank_at(state, frames: int):
    """A blank state of ``frames`` frames (``state``'s layouts)."""
    from signtranslator.generation.compositional import take_frames
    return blank_states([take_frames(state, np.zeros(frames, dtype=int))])[0]


def build_controls(refs, *, seed: int, channels, fps: float = FPS) -> tuple[dict, dict]:
    """({form: states}, info) for the declared controls and the report-only shuffled chunks, each at the reference
    clip's duration except ``blank_median`` (one state of the median frame count)."""
    median = int(np.median([len(s.timestamps) for s in refs]))
    mism, perm = mismatched(refs, seed)
    forms = dict(blank=blank_states(refs), blank_median=[blank_at(refs[0], median)],
                 static_valid=[static_valid(s, channels) for s in refs], mismatched=mism,
                 shuffled_chunks=[shuffled_chunks(s, seed, i, fps) for i, s in enumerate(refs)])
    return forms, dict(median_frames=median, derangement=perm, chunk_seconds=CHUNK_SECONDS, seed=seed)


# ---- statistics -----------------------------------------------------------------------------------------------

def paired_gap(real, controls: dict, refs, resamples: int = 2000, seed: int = 0):
    """(gap, (low, high)): BLEU(real) - max over ``controls`` of BLEU(control), every form resampled on the same
    clips (``compare_pilot_lexicon.paired_bleu_difference``'s draws)."""
    from signtranslator.generation.motion_to_text import corpus_bleu
    n = len(refs)
    if not all(len(h) == n for h in [real, *controls.values()]):
        raise ValueError("every form must translate the same clips")
    point = corpus_bleu(real, refs)["bleu"] - max(corpus_bleu(h, refs)["bleu"] for h in controls.values())
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        r = [refs[i] for i in idx]
        gaps.append(corpus_bleu([real[i] for i in idx], r)["bleu"]
                    - max(corpus_bleu([h[i] for i in idx], r)["bleu"] for h in controls.values()))
    return float(point), (float(np.quantile(gaps, .025)), float(np.quantile(gaps, .975)))


def paired(a, b, refs) -> dict:
    """``compare_pilot_lexicon.paired_bleu_difference``, unrounded (gates decide on these; the report is rounded)."""
    from compare_pilot_lexicon import paired_bleu_difference
    d, (lo, hi) = paired_bleu_difference(a, b, refs)
    return dict(difference=float(d), ci95=[float(lo), float(hi)])


def bleu_ci_raw(hyps, refs, *, seed: int = 0, n: int | None = None):
    """``strict_prior_review.bleu_ci`` (the same draws: ``default_rng(seed)``, ``BOOT`` resamples, percentiles), without
    its rounding to 3 decimals."""
    from signtranslator.generation.motion_to_text import corpus_bleu
    from strict_prior_review import BOOT
    rng = np.random.default_rng(seed)
    point = corpus_bleu(hyps, refs)["bleu"]
    vals = []
    for _ in range(BOOT if n is None else n):
        idx = rng.integers(0, len(hyps), len(hyps))
        vals.append(corpus_bleu([hyps[i] for i in idx], [refs[i] for i in idx])["bleu"])
    return float(point), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def form_row(hyps, refs) -> dict:
    from signtranslator.generation.motion_to_text import corpus_bleu
    point, ci = bleu_ci_raw(hyps, refs)
    b = corpus_bleu(hyps, refs)
    return dict(bleu=point, ci95=ci, length_ratio=b["hypothesis_length"] / max(b["reference_length"], 1),
                hypotheses=list(hyps))


def rounded(value, digits: int = 4):
    """The report's copy of ``value`` with every float rounded (gates were decided on the unrounded values)."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v, digits) for v in value]
    return value


def kendall_tau(a, b) -> float:
    """Kendall's tau-a of two score lists over the same items."""
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return float(s / (n * (n - 1) / 2)) if n > 1 else float("nan")


# ---- gates ----------------------------------------------------------------------------------------------------

def gate_g1(real: dict) -> dict:
    return dict(real=real["bleu"], ci95=list(real["ci95"]), need=dict(real=G1_REAL, ci_low_above=V1_REAL),
                **{"pass": bool(real["bleu"] >= G1_REAL and real["ci95"][0] > V1_REAL)})


def gate_g2(gap: float, ci, halfwidth=None) -> dict:
    out = dict(gap=float(gap), ci95=[float(ci[0]), float(ci[1])], controls=list(GAP_CONTROLS),
               need=dict(gap=G2_GAP, ci_low_above=V1_GAP), **{"pass": bool(gap >= G2_GAP and ci[0] > V1_GAP)})
    if halfwidth:
        out["resolution"] = halfwidth
    return out


def gate_g3(arms: dict | None, round_trip: dict, real: dict) -> dict:
    """``arms``: {arm: paired(arm, real)} (None: not evaluated)."""
    trip_ok = bool(round_trip["bleu"] >= G3_ROUND_TRIP * real["bleu"])
    out = dict(round_trip=dict(bleu=round_trip["bleu"], need=G3_ROUND_TRIP * real["bleu"], **{"pass": trip_ok}))
    if arms is None:
        return dict(out, arms=None, **{"pass": False if not trip_ok else None})
    below = {a: dict(d, below_real=bool(d["ci95"][1] < 0)) for a, d in arms.items()}
    missing = [a for a in ARMS if a not in arms]
    ok = trip_ok and not missing and all(v["below_real"] for v in below.values())
    failed = not trip_ok or any(not v["below_real"] for v in below.values())
    return dict(out, arms=below, missing_arms=missing, **{"pass": False if failed else (None if missing else bool(ok))})


def gate_g4(flores: dict | None) -> dict:
    if flores is None:
        return {"pass": None, "reason": "no --flores-manifest"}
    checks = {c: dict(flores[f"real_minus_{c}"], **{"pass": bool(flores[f"real_minus_{c}"]["ci95"][0] > 0)})
              for c in ("blank", "mismatched")}
    separated = {c: bool(flores["real"]["ci95"][0] > flores[c]["ci95"][1]) for c in ("blank", "mismatched")}
    return dict(checks=checks, unpaired_separation=separated, **{"pass": all(v["pass"] for v in checks.values())})


def orderings(v1: dict, v2: dict, refs) -> dict:
    """Every pair of forms: v1's paired difference and whether it is resolved, v2's, and whether v2 keeps v1's sign;
    Kendall's tau of the BLEUs. ``v1`` / ``v2``: {form: hypotheses}."""
    from signtranslator.generation.motion_to_text import corpus_bleu
    names = [f for f in RANKED if f in v1 and f in v2]
    pairs, kept = {}, True
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d1, d2 = paired(v1[a], v1[b], refs), paired(v2[a], v2[b], refs)
            resolved = d1["ci95"][0] > 0 or d1["ci95"][1] < 0
            same = bool(np.sign(d1["difference"]) == np.sign(d2["difference"]) and d2["difference"] != 0)
            if resolved and not same:
                kept = False
            pairs[f"{a}-{b}"] = dict(v1=d1, v2=d2, v1_resolved=bool(resolved), keeps_sign=same if resolved else None,
                                     v2_resolved=bool(d2["ci95"][0] > 0 or d2["ci95"][1] < 0))
    s1 = [corpus_bleu(v1[f], refs)["bleu"] for f in names]
    s2 = [corpus_bleu(v2[f], refs)["bleu"] for f in names]
    return dict(forms=names, bleu_v1=s1, bleu_v2=s2, kendall_tau=kendall_tau(s1, s2), pairs=pairs,
                all_resolved_kept=kept)


def gate_g5(reread: dict | None, refs) -> dict:
    if reread is None:
        return {"pass": None, "reason": "no --reread-key"}
    if reread["references"] != list(refs):
        raise SystemExit("--reread-key translated other clips")
    forms = {f: x for f, x in reread["forms"].items() if x.get("v1_rerun") is not None and x.get("v2") is not None}
    missing = [f for f in RANKED if f not in forms]
    reproduced = {f: x.get("reproduced") for f, x in reread["forms"].items()}     # diagnostic only (amendment 1)
    order = orderings({f: forms[f]["v1_rerun"] for f in forms}, {f: forms[f]["v2"] for f in forms}, refs)
    failed = not order["all_resolved_kept"]
    return dict(missing_forms=missing, reproduced_stored_v1_diagnostic=reproduced, orderings=order,
                read="v1 and v2 on the same regenerated samples (v1_rerun against v2)",
                **{"pass": False if failed else (None if missing else True)})


def gate_g6(groups: dict) -> dict:
    checks = {g: dict(groups[g], **{"pass": bool(groups[g]["real_minus_blank"]["ci95"][0] > 0)}) for g in GROUPS
              if g in groups}
    missing = [g for g in GROUPS if g not in groups]
    return dict(groups=checks, missing=missing,
                **{"pass": None if missing else all(v["pass"] for v in checks.values())})


def resolution(reread: dict | None, gap_v2: float, refs) -> dict | None:
    """G2's report: the gap over the W14b - R16C paired half-width, v2 against v1."""
    if reread is None or not all(reread["forms"].get(f, {}).get("v1_rerun") for f in ("W14b", "R16C", "real", "blank")):
        return None
    from signtranslator.generation.motion_to_text import corpus_bleu
    forms = reread["forms"]
    out = {}
    for name, key, gap in (("v1", "v1_rerun", None), ("v2", "v2", gap_v2)):
        d = paired(forms["W14b"][key], forms["R16C"][key], refs)
        half = (d["ci95"][1] - d["ci95"][0]) / 2
        if gap is None:
            gap = corpus_bleu(forms["real"][key], refs)["bleu"] - corpus_bleu(forms["blank"][key], refs)["bleu"]
        out[name] = dict(w14b_minus_r16c=d, halfwidth=half, gap=gap, gap_over_halfwidth=gap / half if half > 0 else None)
    out["note"] = ("both on the regenerated samples; v1's gap is real - blank (as its declared 2.277), v2's is G2's gap "
                   "over max(blank, static, mismatched)")
    return out


# ---- reads ----------------------------------------------------------------------------------------------------

def translate_forms(model, forms: dict, *, beam: int, batch: int = 32) -> dict:
    from score_back_translation import translate
    return {name: translate(model, "states", states, beam=beam, batch=batch) for name, states in forms.items()}


def round_trips(store, refs, texts, tokenizer_key, tokenizer_sha, template, device):
    """The d2 tokenizer round trip of each clip, ``build_recon_states``'s form (codes decoded as a generation,
    accepted gains, head smoothing)."""
    import hashlib
    from build_recon_states import decode_state, load_tokenizer
    from evaluate_text_to_motion import accepted_decoder_gains, at_rate
    from exposure_diagnostic import make_post
    from signtranslator.data.layout_adapter import conform
    payload = store.get(tokenizer_key)
    if hashlib.sha256(payload).hexdigest() != tokenizer_sha:
        raise SystemExit(f"{tokenizer_key}: SHA-256 differs from --tokenizer-sha256")
    tokenizer, _ = load_tokenizer(payload, device)
    post = make_post(accepted_decoder_gains(store, tokenizer_key))
    return [decode_state(at_rate(conform(s, template), FPS), tokenizer, tokenizer.config.motion, post, text=t,
                         device=device) for s, t in zip(refs, texts)]


def flores_read(store, model, spec: str, *, clips: int, seed: int, beam: int, cache) -> dict:
    """G4's reads on 2M-Flores (a ``--crossdomain`` manifest): real, blank and mismatched, paired."""
    from evaluate_text_to_motion import at_rate
    from signtranslator.data_engineering.corpus_manifest import load_sample
    from signtranslator.generation import bt_corpora as bt
    from train_state_to_text_t5 import load_store_manifest
    key, sha = spec.rsplit("@", 1)
    manifest = load_store_manifest(store, key, sha)
    records = []
    for r in manifest["records"]:
        name = bt.corpus_name(manifest, r)
        if bt.family(name) != "flores":
            raise SystemExit(f"--flores-manifest holds {name!r}, not 2M-Flores")
        r = dict(r, corpus=name, _role=bt.role_of(r, name))
        bt.check_loadable_bt(r, training=False)
        records.append(r)
    records.sort(key=lambda r: r["sign"])
    if clips:
        import random
        records = [records[i] for i in sorted(random.Random(seed).sample(range(len(records)), min(clips, len(records))))]
    states = [at_rate(load_sample(r, "continuous", cache=cache, fetch=store.get), FPS) for r in records]
    texts = [bt.text_of(r) for r in records]
    mism, _ = mismatched(states, seed)
    hyps = translate_forms(model, dict(real=states, blank=blank_states(states), mismatched=mism), beam=beam)
    out = {name: {k: v for k, v in form_row(h, texts).items() if k != "hypotheses"} for name, h in hyps.items()}
    for c in ("blank", "mismatched"):
        out[f"real_minus_{c}"] = paired(hyps["real"], hyps[c], texts)
    out.update(clips=len(texts), manifest_key=key, manifest_sha256=sha)
    return out


def main(argv=None) -> int:
    import pilot_content_check as pcc
    from build_recon_states import TOKENIZER_KEY, TOKENIZER_SHA256
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default="r2")
    p.add_argument("--translator-key", required=True)
    p.add_argument("--translator-sha256", required=True)
    p.add_argument("--mix-key", default=pcc.MIX)
    p.add_argument("--mix-sha256", default=pcc.MIX_SHA)
    p.add_argument("--clips", type=int, default=1000)
    p.add_argument("--reread-key", help="reread_arms.py report (G3's arms, G5, G2's resolution report)")
    p.add_argument("--flores-manifest", help="KEY@SHA of the 2M-Flores manifest (G4)")
    p.add_argument("--flores-clips", type=int, default=0, help="0: every clip")
    p.add_argument("--tokenizer-key", default=TOKENIZER_KEY)
    p.add_argument("--tokenizer-sha256", default=TOKENIZER_SHA256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="smoke runs: only the first LIMIT of the clips")
    p.add_argument("--device", default="auto")
    p.add_argument("--cache-bytes", type=int, default=16 * 2 ** 30)
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    import os
    import torch
    from evaluate_text_to_motion import at_rate, load_state
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from signtranslator.generation.state_to_text_t5 import layout_template, load_model
    from train_state_to_text_t5 import load_store_manifest
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: the acceptance is written once")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    try:
        model, saved = load_model(store.get(args.translator_key), expected_sha256=args.translator_sha256)
    except ValueError as error:
        raise SystemExit(f"--translator-key: {error}")
    model = model.to(device)
    manifest = load_store_manifest(store, args.mix_key, args.mix_sha256)
    val = pcc.validation_sentences(manifest, args.clips)
    if args.limit:
        val = val[:args.limit]
    del manifest
    refs = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), FPS) for r in val]
    texts = [r["text"] for r in val]
    groups = [r["signer_component"] for r in val]
    controls, info = build_controls(refs, seed=args.seed, channels=set(model.config.motion.trained_channels))
    template = layout_template(model.config)
    forms = dict(real=refs, **controls,
                 round_trip=round_trips(store, refs, texts, args.tokenizer_key, args.tokenizer_sha256, template,
                                        str(device)))
    hyps = translate_forms(model, {k: v for k, v in forms.items() if k != "blank_median"}, beam=args.beam)
    one = translate_forms(model, dict(b=controls["blank_median"]), beam=args.beam)["b"]   # one input for every clip
    hyps["blank_median"] = one * len(refs)
    rows = {name: form_row(h, texts) for name, h in hyps.items()}
    paired_rows = {f"real_minus_{c}": paired(hyps["real"], hyps[c], texts) for c in hyps if c != "real"}
    gap, gap_ci = paired_gap(hyps["real"], {c: hyps[c] for c in GAP_CONTROLS}, texts)
    reread = json.loads(store.get(args.reread_key)) if args.reread_key else None
    if reread is not None and args.limit:
        raise SystemExit("--limit cannot be combined with --reread-key (the re-read covers every clip)")
    group_rows = {}
    for g in GROUPS:
        idx = [i for i, x in enumerate(groups) if x == g]
        if len(idx) >= 2:
            t = [texts[i] for i in idx]
            group_rows[g] = dict(clips=len(idx), real=form_row([hyps["real"][i] for i in idx], t)["bleu"],
                                 blank=form_row([hyps["blank"][i] for i in idx], t)["bleu"],
                                 real_minus_blank=paired([hyps["real"][i] for i in idx], [hyps["blank"][i] for i in idx], t))
    seconds = [len(r.timestamps) / FPS for r in refs]
    long_clips = {}
    for part, idx in (("over_20s", [i for i, x in enumerate(seconds) if x > 20.]),
                      ("up_to_20s", [i for i, x in enumerate(seconds) if x <= 20.])):
        t = [texts[i] for i in idx]
        long_clips[part] = dict(clips=len(idx), **({f: form_row([hyps[f][i] for i in idx], t)["bleu"]
                                                     for f in ("real", "blank")} if idx else {}))
    arms = None
    if reread is not None:
        if reread["references"] != texts:
            raise SystemExit("--reread-key translated other clips")
        if reread.get("translator_sha256") != args.translator_sha256:
            raise SystemExit("--reread-key was read with another instrument checkpoint")
        arms = {a: paired(reread["forms"][a]["v2"], hyps["real"], texts) for a in ARMS if a in reread["forms"]}
    cache = ContentCache(Path(os.environ.get("SIGN_CACHE_DIR", tempfile.gettempdir())) / "sign-translator-cache",
                         args.cache_bytes)
    flores = (flores_read(store, model, args.flores_manifest, clips=args.flores_clips, seed=args.seed, beam=args.beam,
                          cache=cache) if args.flores_manifest else None)
    gates = dict(G1=gate_g1(rows["real"]), G2=gate_g2(gap, gap_ci, resolution(reread, gap, texts)),
                 G3=gate_g3(arms, rows["round_trip"], rows["real"]), G4=gate_g4(flores), G5=gate_g5(reread, texts),
                 G6=gate_g6(group_rows))
    verdicts = {k: v["pass"] for k, v in gates.items()}
    complete = all(v is not None for v in verdicts.values())
    report = dict(kind="instrument-v2-acceptance", translator_key=args.translator_key,
                  translator_sha256=args.translator_sha256, run=saved.get("run"), step=saved.get("step"),
                  licence=saved.get("licence"), mix_key=args.mix_key, mix_sha256=args.mix_sha256, clips=len(texts),
                  beam=args.beam, controls=info, references=texts, signer_components=groups, forms=rows,
                  paired=paired_rows, gates=gates, verdicts=verdicts, complete=complete, long_clips=long_clips,
                  **{"pass": bool(complete and all(verdicts.values()))}, flores=flores, limit=args.limit or None)
    store.put(args.output_key, json.dumps(rounded(report), indent=1, sort_keys=True, default=float).encode())
    print(json.dumps(dict(forms={k: (v["bleu"], v["ci95"], v["length_ratio"]) for k, v in rows.items()},
                          gap=(round(gap, 3), gap_ci), verdicts=verdicts, complete=complete,
                          **{"pass": report["pass"]})), flush=True)
    return 0 if report["pass"] else (4 if any(v is False for v in verdicts.values()) else 2)


if __name__ == "__main__":
    raise SystemExit(main())
