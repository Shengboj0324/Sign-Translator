"""Round 17 diagnostics: the gate measurements of the plan prior and the detail prior that need generation or
teacher forcing (docs/NEXT_PHASE_PLAN.md, "Round 17: declared gates" and "Round 17 gates, amendment 1"). Each command
writes one JSON report with its gate flags.

``plan-route`` (P17 gate (b), amendment 1): the plan-only route on the pilot's 1,000 How2Sign validation clips
(``pilot_content_check``'s fixed sample, each clip's own duration): the plan prior's plans -> the plan decoder -> the
frozen tokenizer's decoder (no detail prior), read by the accepted 2D back-translator, beside each clip's plan round
trip (its own plan decoded), real and blank. BLEU with bootstrap intervals; paired differences plan-route minus blank
(the gate: lower bound > 0) and minus plan round trip. The script's own blank must reproduce the pilots' blank
(``--expected-blank``, 1.859); the report's ``blank_check`` says whether it does, and the exit status is 4 when it does
not. **Seeds:** the plan prior samples each block of 32 clips starting at index i with seed ``--seed`` + i (default
0, 32, 64, …), at ``--guidance`` and ``--part-temperature``. The arms' pilots draw P17 with seed 1000 + i + 104729
(``plan_codes.PLAN_SEED_OFFSET``) and the detail with 1000 + i, so this route's plans are an independent draw.

``text-gain`` (P17 gate (a), amendment 1): a prior's teacher-forced loss on fixed validation batches with each row's
text replaced by another row's (the batch rolled by one) minus with its own text. Per clip, the gain is the summed
part-weighted cross entropy (the weights of the prior's own loss: ``part_valid_loss``) over all its codes; the gate
reads it in **nats per second** (each clip's summed gain divided by its duration, ``utterance_duration``), averaged over
clips with a bootstrap interval over clips. Also reported: the pooled nats per second (Σ gain / Σ seconds), nats per
code (Σ gain / Σ weight) and the stage-1 codes' share. Works for an rvq prior (W14b, for the 2× reference), a
plan-conditioned detail prior (true plans) and a plan prior. Locus slots stay with their clip (measured in its motion).

``plan-following`` (D17 gate, amendment 1): unordered pairs {A, B} of the pilot's clips (``--pairing consecutive``:
clips 2k and 2k + 1; ``duration``: neighbours in duration order whose durations differ by at most 10%, a second read).
For each ordered direction (A, B) the detail prior is given A's oracle plan (the plan tokenizer's codes of clip A) at
A's duration with **no text** (guidance 0: the empty-text logits alone); the result is back-translated and scored
against A's and B's sentences (smoothed sentence BLEU): a hit when closer to A (ties one half). Controls on the same
directions: **blank at A's duration** (clip A with every channel removed) and the real clip A. Duration alone already
retrieves A well above 0.5 (BLEU's brevity penalty favours the sentence whose length matches): the reviewer measured
blank 0.6605 [0.631, 0.688] and real 0.757 on the pilots. The gate: the paired difference (A-plan minus blank) has a
lower bound > 0, and the A-plan rate ≥ blank + 0.5 × (real − blank). Every interval resamples unordered pairs (a pair's
two directions stay together). Reported only: B's text with A's plan (at ``--guidance``).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

KP2D_RUN = "back-translator-kp2d-uwb-ft-v1"
KP2D_MANIFEST = "004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd"
PILOT_BLANK_BLEU = 1.859            # the pilots' blank on the 1,000 clips (round 16 result)


def sentence_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """Smoothed sentence BLEU in [0, 1] (Lin & Och 2004: add one to the n-gram counts for n ≥ 2), over
    ``motion_to_text.words`` tokens; 0 for an empty hypothesis."""
    from signtranslator.generation.motion_to_text import words
    h, r = words(hypothesis), words(reference)
    if not h:
        return 0.
    logs = []
    for n in range(1, max_n + 1):
        hc = collections.Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        rc = collections.Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        match, total = sum(min(c, rc[g]) for g, c in hc.items()), max(len(h) - n + 1, 0)
        if n > 1:
            match, total = match + 1, total + 1
        if match == 0 or total == 0:
            return 0.
        logs.append(math.log(match / total))
    bp = 1. if len(h) > len(r) else math.exp(1 - len(r) / len(h))
    return bp * math.exp(sum(logs) / max_n)


def hits(hypotheses, own, other) -> np.ndarray:
    """1 where the hypothesis is closer to its ``own`` sentence than to the ``other``, 0.5 on a tie, else 0."""
    return np.array([1. if a > b else .5 if a == b else 0. for a, b in
                     ((sentence_bleu(h, o), sentence_bleu(h, x)) for h, o, x in zip(hypotheses, own, other))])


def _boot(rows: np.ndarray, resamples: int, seed: int) -> np.ndarray:
    """Bootstrap means of per-pair values ``rows`` (n_pairs,), resampling pairs."""
    rng = np.random.default_rng(seed)
    return rows[rng.integers(0, len(rows), (resamples, len(rows)))].mean(1)


def pair_rate(pair_hits: np.ndarray, *, resamples: int = 2000, seed: int = 0) -> dict:
    """Retrieval rate of (n_pairs, 2) directed hits, with a 95% interval resampling unordered pairs."""
    rows = pair_hits.mean(1)
    boot = _boot(rows, resamples, seed)
    return dict(rate=round(float(rows.mean()), 4), ci95=[round(float(np.quantile(boot, .025)), 4),
                                                         round(float(np.quantile(boot, .975)), 4)],
                pairs=len(rows), directions=int(pair_hits.size))


def paired_rate_difference(a: np.ndarray, b: np.ndarray, *, resamples: int = 2000, seed: int = 0) -> dict:
    """Rate a − rate b over the same (n_pairs, 2) directions, with a 95% interval resampling unordered pairs."""
    rows = (a - b).mean(1)
    boot = _boot(rows, resamples, seed)
    return dict(difference=round(float(rows.mean()), 4), ci95=[round(float(np.quantile(boot, .025)), 4),
                                                               round(float(np.quantile(boot, .975)), 4)])


def make_pairs(seconds, pairing: str, tolerance: float = .1) -> list[tuple[int, int]]:
    """Unordered pairs of clip indices: ``consecutive`` (2k, 2k + 1), or ``duration``: walking the clips in duration
    order, a clip is paired with the next when their durations differ by at most ``tolerance`` (longer / shorter ≤
    1 + tolerance); otherwise it is skipped."""
    if pairing == "consecutive":
        return [(i, i + 1) for i in range(0, len(seconds) // 2 * 2, 2)]
    order = sorted(range(len(seconds)), key=lambda i: (seconds[i], i))
    pairs, k = [], 0
    while k + 1 < len(order):
        a, b = order[k], order[k + 1]
        if max(seconds[a], seconds[b]) <= (1 + tolerance) * min(seconds[a], seconds[b]):
            pairs.append((a, b)); k += 2
        else:
            k += 1
    return pairs


def mean_ci(values, *, resamples: int = 2000, seed: int = 0, unit: str = "clips") -> dict:
    v = np.asarray(values, dtype=float)
    boot = _boot(v, resamples, seed)
    return {"mean": round(float(v.mean()), 5), "ci95": [round(float(np.quantile(boot, .025)), 5),
                                                         round(float(np.quantile(boot, .975)), 5)], unit: len(v)}


def load_pinned(store, key: str, sha256: str) -> dict:
    payload = store.get(key)
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise SystemExit(f"{key}: SHA-256 differs from the pin")
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def load_prior(saved: dict, device):
    """(prior, tokenizer, plan tokenizer or None, kind) from a prior checkpoint (EMA weights), every model in eval."""
    from signtranslator.generation.motion_rvq import MotionRVQ, RVQPrior, RVQPriorConfig
    from signtranslator.generation.plan_codes import PLAN_PRIOR_KIND, PlanPriorConfig, PlanVQ, PlanVQConfig
    kind = saved.get("kind")
    if kind == PLAN_PRIOR_KIND:
        prior = RVQPrior(PlanPriorConfig.from_dict(saved["config"]))
        tokenizer = MotionRVQ(prior.config.tokenizer.tokenizer)
    elif kind == "motion_rvq_prior":
        prior = RVQPrior(RVQPriorConfig.from_dict(saved["config"]))
        tokenizer = MotionRVQ(prior.config.tokenizer)
    else:
        raise SystemExit(f"not an rvq or plan prior checkpoint (kind {kind!r})")
    prior.load_state_dict(saved.get("ema") or saved["model"])
    tokenizer.load_state_dict(saved["tokenizer_state"])
    plan = None
    if saved.get("plan_tokenizer_state") is not None:
        plan = PlanVQ(PlanVQConfig.from_dict(saved["plan_tokenizer_config"]))
        plan.load_state_dict(saved["plan_tokenizer_state"])
        plan.to(device).eval()
    return prior.to(device).eval(), tokenizer.to(device).eval(), plan, kind


@torch.no_grad()
def row_nats(prior, tokenizer, plan, kind, tokens, mask, motion, loci=None):
    """((B,) summed part-weighted cross entropy per clip over all codes, (B,) the same over the stage-1 codes, (B,) the
    summed weights), teacher-forced in evaluation mode: the terms of ``rvq_prior_loss(train=False)`` (or
    ``plan_prior_loss``) per clip, so Σ nats / Σ weights is its ``tokens`` (tested)."""
    from torch.nn import functional as F
    from signtranslator.generation.motion_rvq import part_code_weights
    from signtranslator.generation.plan_codes import PLAN_PRIOR_KIND, plan_part_weights, tokenizer_latents
    from signtranslator.generation.text_to_motion import batch_seconds
    config = prior.config
    latents, valid, _ = tokenizer_latents(tokenizer, motion)
    plan_kw = {}
    if kind == PLAN_PRIOR_KIND:
        codes, valid = plan.eval().encode_latents(latents, valid)
        weight = (plan_part_weights(tokenizer, plan.config, motion, valid) if config.part_valid_loss else None)
    else:
        _, codes, _ = tokenizer.quantize(latents)
        weight = part_code_weights(tokenizer, config, motion, valid) if config.part_valid_loss else None
        if config.plan_window:
            plan_kw = {"plan": plan.eval().encode_latents(latents, valid)}
    hidden, _, text_mask = prior.encode(tokens, mask, loci)
    previous = torch.cat((torch.full_like(codes[:, :1], prior.K), codes[:, :-1]), 1)
    seconds = batch_seconds(motion).float()
    step_seconds = seconds[:, ::config.tokenizer.step][:, :codes.shape[1]]
    step_seconds = F.pad(step_seconds, (0, codes.shape[1] - step_seconds.shape[1]))
    duration = (seconds * motion.frame_valid).amax(1)
    h = prior.temporal_states(hidden, text_mask, previous, step_seconds, duration, **plan_kw)
    target = codes[valid]
    logits = prior.depth_logits(h[valid], target[:, :-1])
    per_code = F.cross_entropy(logits.reshape(-1, prior.K), target.reshape(-1), reduction="none").reshape(target.shape)
    weight = torch.ones_like(per_code) if weight is None else weight.to(per_code.dtype)
    P = len(tokenizer.parts)
    row = torch.arange(len(codes), device=codes.device)[:, None].expand_as(valid)[valid]
    total = lambda x: torch.zeros(len(codes), dtype=x.dtype, device=x.device).index_add_(0, row, x)
    return (total((per_code * weight).sum(1)), total((per_code[:, :P] * weight[:, :P]).sum(1)), total(weight.sum(1)))


@torch.no_grad()
def text_gain(prior, tokenizer, plan, kind, batches, device) -> dict:
    """Per-clip (shuffled-text − true-text) summed teacher-forced losses, reported per second and per code."""
    from signtranslator.generation.text_to_motion import utterance_duration
    from train_text_to_motion import to_device
    gain, gain1, weights, seconds, true = [], [], [], [], []
    for batch in batches:
        tokens, mask, motion, *loci = to_device(batch, device)
        if len(tokens) < 2:
            continue
        loci = loci[0] if loci else None
        own, own1, w = row_nats(prior, tokenizer, plan, kind, tokens, mask, motion, loci)
        other, other1, _ = row_nats(prior, tokenizer, plan, kind, tokens.roll(1, 0), mask.roll(1, 0), motion, loci)
        gain.append((other - own).double().cpu()); gain1.append((other1 - own1).double().cpu())
        weights.append(w.double().cpu()); true.append(own.double().cpu())
        seconds.append(utterance_duration(motion).double().cpu())
    if not gain:
        raise SystemExit("no batch with two rows")
    gain, gain1, weights, seconds, true = (torch.cat(x).numpy() for x in (gain, gain1, weights, seconds, true))
    keep = weights > 0
    gain, gain1, weights, seconds, true = gain[keep], gain1[keep], weights[keep], seconds[keep], true[keep]
    return dict(text_gain_nats_per_second=mean_ci(gain / seconds),
                text_gain_nats_per_second_pooled=round(float(gain.sum() / seconds.sum()), 5),
                text_gain_stage1_nats_per_second=mean_ci(gain1 / seconds),
                text_gain_nats_per_code=round(float(gain.sum() / weights.sum()), 5),
                text_gain_stage1_nats_per_code=round(float(gain1.sum() / weights.sum()), 5),
                tokens_true_text=round(float(true.sum() / weights.sum()), 5), clips=int(len(gain)),
                seconds=round(float(seconds.sum()), 2))


def validation_batches(store, args, prior):
    """The fixed validation batches (training's own dataset and collation; first ``--batches`` of ``--batch-size``)."""
    from functools import partial
    from torch.utils.data import DataLoader
    from signtranslator.data.text_motion_dataset import TextMotionDataset, collate_text_motion
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest
    from train_motion_tokens import layout_template
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    cache = ContentCache(Path(os.environ.get("SIGN_CACHE_DIR", tempfile.gettempdir())) / "sign-translator-cache", 8 * 2 ** 30)
    template = layout_template(manifest, args.conform_to_corpus, cache, store.get) if args.conform_to_corpus else None
    val = TextMotionDataset(manifest, partition=args.partition, cache=cache, fetch=store.get, max_seconds=args.max_seconds,
                            fps=args.fps, template=template)
    loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=partial(collate_text_motion, loci=prior.config.locus_conditioning))
    return [b for _, b in zip(range(args.batches), loader)]


def pilot_clips(store, args):
    """(texts, reference states at 24 fps, seconds) of the pilot's fixed How2Sign validation sample."""
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from evaluate_text_to_motion import at_rate, load_state
    from pilot_content_check import validation_sentences
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    val = validation_sentences(manifest, args.clips)
    refs = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), 24.) for r in val]
    return [r["text"] for r in val], refs, [float(s.timestamps[-1] - s.timestamps[0]) + 1 / 24. for s in refs]


def back_translator(store, device):
    from score_back_translation import load_translator, translate
    kp = json.loads(store.get(f"training-runs/{KP2D_RUN}/best.json"))["checkpoint_key"]
    translator, kind = load_translator(store, kp, KP2D_MANIFEST)
    translator = translator.to(torch.device(device))
    return lambda states: list(translate(translator, kind, states))


def blank_check(measured: float, expected: float) -> dict:
    """Whether the script's blank BLEU (rounded to 3 decimals, as every report) reproduces the pilots' blank."""
    return dict(expected=expected, measured=measured, reproduced=bool(abs(round(measured, 3) - expected) < 5e-4))


def main(argv=None, *, translate_fn=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=("plan-route", "text-gain", "plan-following"))
    p.add_argument("--store", default="r2")
    p.add_argument("--checkpoint-key", required=True, help="text-gain: the prior; plan-route: the plan prior; "
                   "plan-following: the plan-conditioned detail prior")
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--plan-tokenizer-key", help="plan-route / plan-following: the plan tokenizer (pinned)")
    p.add_argument("--plan-tokenizer-sha256")
    # text-gain data (the training run's validation partition and conventions)
    p.add_argument("--manifest-key"); p.add_argument("--manifest-sha256")
    p.add_argument("--conform-to-corpus", default="how2sign-two-view-text-v1")
    p.add_argument("--fps", type=float, default=24.)
    p.add_argument("--max-seconds", type=float, default=20.)
    p.add_argument("--partition", default="val", help="text-gain: the manifest partition read")
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    # pilot clips
    p.add_argument("--mix-key"); p.add_argument("--mix-sha256")
    p.add_argument("--clips", type=int, default=1000)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--part-temperature", default="body=1.1,left_hand=1.1,right_hand=1.1,face=1.0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--expected-blank", type=float, default=PILOT_BLANK_BLEU,
                   help="plan-route: the pilots' blank BLEU on the same clips, which the script's blank must reproduce")
    p.add_argument("--pairing", choices=("consecutive", "duration"), default="consecutive",
                   help="plan-following: consecutive clips, or neighbours in duration within 10%% (a second read)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    saved = load_pinned(store, args.checkpoint_key, args.checkpoint_sha256)
    report = dict(command=args.command, checkpoint_key=args.checkpoint_key, checkpoint_sha256=args.checkpoint_sha256)
    status = 0
    if args.command == "text-gain":
        if not (args.manifest_key and args.manifest_sha256):
            raise SystemExit("text-gain needs --manifest-key and --manifest-sha256")
        prior, tokenizer, plan, kind = load_prior(saved, device)
        batches = validation_batches(store, args, prior)
        report.update(kind=kind, manifest_key=args.manifest_key, partition=args.partition, batches=len(batches),
                      batch_size=args.batch_size, **text_gain(prior, tokenizer, plan, kind, batches, device))
    else:
        if not (args.plan_tokenizer_key and args.plan_tokenizer_sha256 and args.mix_key and args.mix_sha256):
            raise SystemExit(f"{args.command} needs --plan-tokenizer-key/--plan-tokenizer-sha256 and --mix-key/--mix-sha256")
        from signtranslator.generation.plan_codes import (decode_plan_states, generate_plan_codes_batch,
                                                           load_plan_tokenizer, oracle_plans)
        from pilot_content_check import postprocess
        from strict_prior_review import bleu_ci
        from compare_pilot_lexicon import paired_bleu_difference
        if saved.get("plan_tokenizer_sha256") != args.plan_tokenizer_sha256:
            raise SystemExit("the checkpoint was trained with another plan tokenizer than the one pinned")
        plan, d2, plan_saved = load_plan_tokenizer(store.get(args.plan_tokenizer_key), args.plan_tokenizer_sha256,
                                                   tokenizer_sha256=saved["tokenizer_sha256"], device=device)
        from evaluate_text_to_motion import accepted_decoder_gains, blanked
        gains = accepted_decoder_gains(store, saved.get("tokenizer_key"))
        if gains:
            from fit_tokenizer_gains import apply_gains
        post = lambda s: postprocess(apply_gains(s, gains) if gains else s)
        temperature = {k: float(v) for k, v in (x.split("=") for x in args.part_temperature.split(","))}
        texts, refs, seconds = pilot_clips(store, args)
        translate_fn = translate_fn or back_translator(store, args.device)
        own_plans, adjusted = oracle_plans(plan, d2, refs, seconds, 24.)
        report.update(clips=len(texts), plan_tokenizer_key=args.plan_tokenizer_key, oracle_length_adjusted=adjusted,
                      guidance=args.guidance, part_temperature=temperature, decoder_gains=gains, seed=args.seed)
        if args.command == "plan-route":
            prior, _, _, kind = load_prior(saved, device)
            if kind != "motion_plan_prior":
                raise SystemExit("plan-route needs a plan prior checkpoint")
            sampled = []
            for i in range(0, len(texts), 32):        # seeds: --seed + i per block of 32 (see the module docstring)
                sampled += generate_plan_codes_batch(prior, texts[i:i + 32], fps=24., durations=seconds[i:i + 32],
                                                     guidance=args.guidance, temperature=temperature,
                                                     seed=args.seed + i)
            reads = {"plan_route": [post(s) for s in decode_plan_states(plan, d2, sampled, texts, fps=24., durations=seconds)],
                     "plan_round_trip": [post(s) for s in decode_plan_states(plan, d2, own_plans, texts, fps=24.,
                                                                             durations=seconds)],
                     "real": refs, "blank": blanked(refs)}
            hyps = {name: translate_fn(states) for name, states in reads.items()}
            for name in reads:
                point, ci = bleu_ci(hyps[name], texts)
                report[name] = dict(bleu=point, ci95=ci, hypotheses=hyps[name])
            for other in ("blank", "plan_round_trip"):
                d, (lo, hi) = paired_bleu_difference(hyps["plan_route"], hyps[other], texts)
                report[f"plan_route_minus_{other}"] = dict(difference=round(d, 3), ci95=[round(lo, 3), round(hi, 3)])
            report["references"] = texts
            report["blank_check"] = blank_check(report["blank"]["bleu"], args.expected_blank)
            report["gate"] = dict(plan_route_minus_blank_lower_above_0=report["plan_route_minus_blank"]["ci95"][0] > 0,
                                  blank_reproduced=report["blank_check"]["reproduced"])
            report["gate"]["passes"] = all(report["gate"].values())
            status = 0 if report["blank_check"]["reproduced"] else 4
        else:                                          # plan-following
            prior, _, _, kind = load_prior(saved, device)
            if kind != "motion_rvq_prior" or not prior.config.plan_window:
                raise SystemExit("plan-following needs a plan-conditioned detail prior")
            from signtranslator.generation.motion_rvq import generate_rvq_motion_batch
            pairs = make_pairs(seconds, args.pairing)
            # ordered directions: (A, B) for every pair, then (B, A); hits are reshaped to (pairs, 2)
            own = [a for a, _ in pairs] + [b for _, b in pairs]
            other = [b for _, b in pairs] + [a for a, _ in pairs]
            own_texts, other_texts = [texts[r] for r in own], [texts[r] for r in other]
            as_pairs = lambda h: np.stack([h[:len(pairs)], h[len(pairs):]], 1)

            def generate(cond_texts, guidance):
                states = []
                for i in range(0, len(own), 32):
                    rows = own[i:i + 32]
                    states += [post(s) for s in generate_rvq_motion_batch(
                        prior, d2, cond_texts[i:i + 32], fps=24., durations=[seconds[r] for r in rows],
                        guidance=guidance, temperature=temperature, seed=args.seed + i,
                        plan=[own_plans[r] for r in rows])]
                return states
            reads = {"a_plan_no_text": generate(own_texts, 0.),   # guidance 0: the empty-text logits alone
                     "blank": blanked([refs[r] for r in own]),     # A's duration, nothing else
                     "real": [refs[r] for r in own],
                     "a_plan_b_text": generate(other_texts, args.guidance)}
            pair_hits = {}
            for name, states in reads.items():
                hyps = translate_fn(states)
                pair_hits[name] = as_pairs(hits(hyps, own_texts, other_texts))
                report[name] = dict(pair_rate(pair_hits[name]), hypotheses=hyps)
            difference = paired_rate_difference(pair_hits["a_plan_no_text"], pair_hits["blank"])
            blank, real = report["blank"]["rate"], report["real"]["rate"]
            threshold = round(blank + .5 * (real - blank), 4)
            report["a_plan_minus_blank"] = difference
            report["gate"] = dict(difference_lower_above_0=difference["ci95"][0] > 0, threshold=threshold,
                                  rate_at_least_threshold=report["a_plan_no_text"]["rate"] >= threshold)
            report["gate"]["passes"] = bool(report["gate"]["difference_lower_above_0"]
                                            and report["gate"]["rate_at_least_threshold"])
            report["pairing"] = args.pairing
            report["pairs"] = [list(pr) for pr in pairs]
    store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True, default=float).encode())
    print(json.dumps({k: v for k, v in report.items() if k not in ("references", "pairs")
                      and not (isinstance(v, dict) and "hypotheses" in v)} |
                     {k: {f: x for f, x in v.items() if f != "hypotheses"} for k, v in report.items()
                      if isinstance(v, dict) and "hypotheses" in v}, default=float), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
