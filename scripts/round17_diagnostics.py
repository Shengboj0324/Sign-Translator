"""Round 17 diagnostics (review of c37036b, fixes 3 and 5): what the plan carries, whether the detail follows it, and
whether the priors read their text. Report-only; each writes one JSON report.

``plan-route`` (fix 5): the plan-only route on the pilot's How2Sign validation clips (``pilot_content_check``'s fixed
sample, each clip's own duration): the plan prior's plans -> the plan decoder -> the frozen tokenizer's decoder (no
detail prior), read by the accepted 2D back-translator, beside each clip's plan round trip (its own plan decoded),
real and blank. BLEU with bootstrap intervals; paired differences plan-route minus blank and minus plan round trip.

``text-gain`` (fix 5): a prior's teacher-forced loss on fixed validation batches with each row's text replaced by
another row's (the batch rolled by one) minus with its own text, in nats per code (all codes and stage-1 codes), with
a bootstrap interval over batches. Works for an rvq prior (W14b, for reference), a plan-conditioned detail prior (true
plans) and a plan prior. Locus slots stay with their clip (they are measured in its motion).

``plan-following`` (fix 3): pairs (A, B) of the pilot's clips (consecutive, disjoint). The detail prior is given A's
oracle plan (the plan tokenizer's codes of clip A) at A's duration, with (i) no text (guidance 0: the empty-text
logits alone) and (ii) B's text (at --guidance); each
result is back-translated and scored against A's and B's sentences (smoothed sentence BLEU). The retrieval rate is the
fraction of pairs whose translation is closer to A than to B (ties count one half; chance 0.5), with a bootstrap
interval. Real clips A give the instrument's own ceiling. A detail prior that follows its plan retrieves A in (i) and
holds A against B's text in (ii).
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


def retrieval(hypotheses, own, other, *, resamples: int = 2000, seed: int = 0) -> dict:
    """Rate at which each hypothesis is closer to its ``own`` sentence than to the ``other`` (ties one half), with a
    bootstrap 95% interval over the pairs."""
    hits = np.array([1. if a > b else .5 if a == b else 0. for a, b in
                     ((sentence_bleu(h, o), sentence_bleu(h, x)) for h, o, x in zip(hypotheses, own, other))])
    rng = np.random.default_rng(seed)
    boot = hits[rng.integers(0, len(hits), (resamples, len(hits)))].mean(1)
    return dict(rate=round(float(hits.mean()), 4), ci95=[round(float(np.quantile(boot, .025)), 4),
                                                         round(float(np.quantile(boot, .975)), 4)], pairs=len(hits))


def mean_ci(values, *, resamples: int = 2000, seed: int = 0) -> dict:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), (resamples, len(v)))].mean(1)
    return dict(mean=round(float(v.mean()), 5), ci95=[round(float(np.quantile(boot, .025)), 5),
                                                      round(float(np.quantile(boot, .975)), 5)], batches=len(v))


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
def text_gain(prior, tokenizer, plan, kind, batches, device) -> dict:
    """Per-batch (shuffled-text − true-text) teacher-forced losses (nats per code), summarised with intervals."""
    from signtranslator.generation.motion_rvq import rvq_prior_loss
    from signtranslator.generation.plan_codes import PLAN_PRIOR_KIND, plan_prior_loss
    from train_text_to_motion import to_device

    def loss(tokens, mask, motion, loci):
        if kind == PLAN_PRIOR_KIND:
            return plan_prior_loss(prior, tokenizer, plan, tokens, mask, motion, *loci, train=False)[1]
        kw = {"plan_tokenizer": plan} if prior.config.plan_window else {}
        return rvq_prior_loss(prior, tokenizer, tokens, mask, motion, *loci, train=False, **kw)[1]
    gains, gains1, true, true1 = [], [], [], []
    for batch in batches:
        tokens, mask, motion, *loci = to_device(batch, device)
        if len(tokens) < 2:
            continue
        own = loss(tokens, mask, motion, loci)
        other = loss(tokens.roll(1, 0), mask.roll(1, 0), motion, loci)
        gains.append(other["tokens"] - own["tokens"]); gains1.append(other["stage1_tokens"] - own["stage1_tokens"])
        true.append(own["tokens"]); true1.append(own["stage1_tokens"])
    if not gains:
        raise SystemExit("no validation batch with two rows")
    return dict(text_gain=mean_ci(gains), text_gain_stage1=mean_ci(gains1), tokens=mean_ci(true),
                stage1_tokens=mean_ci(true1))


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
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    saved = load_pinned(store, args.checkpoint_key, args.checkpoint_sha256)
    report = dict(command=args.command, checkpoint_key=args.checkpoint_key, checkpoint_sha256=args.checkpoint_sha256)
    if args.command == "text-gain":
        if not (args.manifest_key and args.manifest_sha256):
            raise SystemExit("text-gain needs --manifest-key and --manifest-sha256")
        prior, tokenizer, plan, kind = load_prior(saved, device)
        batches = validation_batches(store, args, prior)
        report.update(kind=kind, manifest_key=args.manifest_key, batches=len(batches), batch_size=args.batch_size,
                      **text_gain(prior, tokenizer, plan, kind, batches, device))
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
                      guidance=args.guidance, part_temperature=temperature, decoder_gains=gains)
        if args.command == "plan-route":
            prior, _, _, kind = load_prior(saved, device)
            if kind != "motion_plan_prior":
                raise SystemExit("plan-route needs a plan prior checkpoint")
            sampled = []
            for i in range(0, len(texts), 32):
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
        else:                                          # plan-following
            prior, _, _, kind = load_prior(saved, device)
            if kind != "motion_rvq_prior" or not prior.config.plan_window:
                raise SystemExit("plan-following needs a plan-conditioned detail prior")
            from signtranslator.generation.motion_rvq import generate_rvq_motion_batch
            n = len(texts) // 2 * 2
            a, b = list(range(0, n, 2)), list(range(1, n, 2))
            # both directions: (A, B) and (B, A), so every clip is once the plan's owner
            own = a + b; other = b + a
            # (i) no text: guidance 0 samples from the empty-text logits alone (the text never reaches them);
            # (ii) B's text at the declared guidance
            conditions = {"empty_text": ([texts[r] for r in own], 0.),
                          "other_text": ([texts[j] for j in other], args.guidance)}
            for name, (cond_texts, guidance) in conditions.items():
                states = []
                for i in range(0, len(own), 32):
                    rows = own[i:i + 32]
                    states += [post(s) for s in generate_rvq_motion_batch(
                        prior, d2, cond_texts[i:i + 32], fps=24., durations=[seconds[r] for r in rows],
                        guidance=guidance, temperature=temperature, seed=args.seed + i,
                        plan=[own_plans[r] for r in rows])]
                hyps = translate_fn(states)
                report[name] = dict(retrieval(hyps, [texts[r] for r in own], [texts[r] for r in other]),
                                    hypotheses=hyps)
            real_hyps = translate_fn([refs[r] for r in own])
            report["real"] = dict(retrieval(real_hyps, [texts[r] for r in own], [texts[r] for r in other]))
            report["pairs"] = [[r, o] for r, o in zip(own, other)]
    store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True, default=float).encode())
    print(json.dumps({k: v for k, v in report.items() if k not in ("references", "pairs")
                      and not (isinstance(v, dict) and "hypotheses" in v)} |
                     {k: {f: x for f, x in v.items() if f != "hypotheses"} for k, v in report.items()
                      if isinstance(v, dict) and "hypotheses" in v}, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
