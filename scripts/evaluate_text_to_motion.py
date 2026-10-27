"""Evaluate a text-to-motion checkpoint against a manifest partition.

For every record of the partition the model generates motion for the record's
English sentence on the reference clip's own duration and frame rate, so the
two can be compared frame by frame; the metrics of
``signtranslator.generation.motion_metrics`` (geodesic error, articulation ratios,
Fréchet distance) are reported per channel. Diffusion checkpoints are evaluated
at each requested guidance scale. Read-only except for the optional output key.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.generation.motion_metrics import compare  # noqa: E402
from signtranslator.generation.signing_space import location_errors  # noqa: E402
from signtranslator.grammar_motion.evaluation import grammar_comparison, locus_adherence  # noqa: E402
from signtranslator.generation.motion_rvq import MotionRVQ, RVQPrior, RVQPriorConfig, generate_rvq_motion  # noqa: E402
from signtranslator.generation.motion_tokens import (  # noqa: E402
    MotionVQ, MotionVQConfig, TokenPrior, TokenPriorConfig, generate_tokens_motion,
)
from signtranslator.generation.text_to_motion import TextToMotion, TextToMotionConfig, generate  # noqa: E402
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402
from train_text_to_motion import make_store  # noqa: E402


def load_state(store, key: str, sha256: str) -> MultichannelMotion:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "s.npz"
        path.write_bytes(store.get(key))
        return MultichannelMotion.load(path, expected_sha256=sha256)


def accepted_decoder_gains(store, tokenizer_key):
    """The accepted per-channel gains for ``tokenizer_key`` (``fit_tokenizer_gains.decoder_gains_key``), or None."""
    if not tokenizer_key:
        return None
    from fit_tokenizer_gains import decoder_gains_key
    key = decoder_gains_key(tokenizer_key)
    if not store.exists(key):
        return None
    report = json.loads(store.get(key))
    if report.get("tokenizer_key") != tokenizer_key:
        raise SystemExit(f"{key} was fitted for {report.get('tokenizer_key')}, not {tokenizer_key}")
    return report["gains"]


def load_generator(store, key: str, manifest_sha256: str, device: str = "cpu", *, temperature: float = 1., top_k: int = 0):
    """(label, trained channels, sample(text, fps, seconds, guidance, steps, seed)) for any checkpoint kind."""
    checkpoint = torch.load(io.BytesIO(store.get(key)), map_location="cpu", weights_only=False)
    if checkpoint["manifest_sha256"] != manifest_sha256:
        raise SystemExit(f"{key} was trained on a different manifest")
    weights = checkpoint.get("ema") or checkpoint["model"]          # EMA weights when the run kept them
    if checkpoint.get("kind") == "motion_rvq_prior":
        prior = RVQPrior(RVQPriorConfig.from_dict(checkpoint["config"]))
        prior.load_state_dict(weights)
        tokenizer = MotionRVQ(prior.config.tokenizer)
        tokenizer.load_state_dict(checkpoint["tokenizer_state"])
        prior.to(device).eval(), tokenizer.to(device).eval()

        def sample(text, fps, seconds, guidance, steps, seed, loci=None):
            return generate_rvq_motion(prior, tokenizer, [text], fps=fps,    # seconds None: the predicted duration
                                       durations=None if seconds is None else [seconds], guidance=guidance,
                                       temperature=temperature, top_k=top_k, seed=seed,  # loci None: planned from text
                                       loci=None if loci is None else [loci])[0]

        def sample_batch(texts, fps, seconds, guidance, seed):
            from signtranslator.generation.motion_rvq import generate_rvq_motion_batch
            return generate_rvq_motion_batch(prior, tokenizer, texts, fps=fps, durations=seconds, guidance=guidance,
                                             temperature=temperature, top_k=top_k, seed=seed)
        sample.batch = sample_batch
        sample.decoder_gains = None
        gains = accepted_decoder_gains(store, checkpoint.get("tokenizer_key"))
        if gains:                      # amendment 5: the accepted gains are part of this tokenizer's decoder
            from fit_tokenizer_gains import apply_gains
            plain, plain_batch = sample, sample_batch
            sample = lambda *a, **k: apply_gains(plain(*a, **k), gains)
            sample.batch = lambda *a, **k: [apply_gains(o, gains) for o in plain_batch(*a, **k)]
            sample.decoder_gains = gains
            print(f"decoder gains applied: {gains}", flush=True)
        return "motion_tokens_v2", prior.config.tokenizer.motion.trained_channels, sample
    if checkpoint.get("kind") in ("motion_tokenizer", "motion_rvq_tokenizer"):
        raise SystemExit(f"{key} is a tokenizer, not a generator; evaluate its prior")
    if checkpoint.get("kind") == "motion_token_prior":
        prior = TokenPrior(TokenPriorConfig.from_dict(checkpoint["config"]))
        prior.load_state_dict(weights)
        tokenizer = MotionVQ(prior.config.tokenizer)
        tokenizer.load_state_dict(checkpoint["tokenizer_state"])
        prior.to(device).eval(), tokenizer.to(device).eval()

        def sample(text, fps, seconds, guidance, steps, seed):
            return generate_tokens_motion(prior, tokenizer, [text], fps=fps, durations=[seconds], guidance=guidance,
                                          temperature=temperature, top_k=top_k,
                                          seed=seed)[0]
        return "motion_tokens", prior.config.tokenizer.motion.trained_channels, sample
    if checkpoint.get("kind") == "motion_tokenizer":
        raise SystemExit(f"{key} is a tokenizer, not a generator; evaluate its prior")
    model = TextToMotion(TextToMotionConfig.from_dict(checkpoint["config"]))
    model.load_state_dict(weights)
    model.to(torch.device(device)).eval()

    def sample(text, fps, seconds, guidance, steps, seed):
        return generate(model, [text], fps=fps, durations=[seconds], steps=steps, guidance=guidance, seed=seed)[0]
    return model.config.objective, model.config.trained_channels, sample


def load_retrieval(store, key: str, manifest_sha256: str):
    """The frozen text-motion retrieval model (EMA weights), refusing one trained on another manifest."""
    from signtranslator.generation.text_motion_retrieval import TextMotionRetrieval, TextMotionRetrievalConfig
    saved = torch.load(io.BytesIO(store.get(key)), map_location="cpu", weights_only=False)
    if saved.get("kind") != "text_motion_retrieval" or saved["manifest_sha256"] != manifest_sha256:
        raise SystemExit(f"{key} is not a retrieval model trained on this manifest")
    model = TextMotionRetrieval(TextMotionRetrievalConfig.from_dict(saved["config"]))
    model.load_state_dict(saved.get("ema") or saved["model"])
    if model.config.text_embedding_key:                # frozen sentence embeddings: the table named in the config
        from signtranslator.generation.text_motion_retrieval import load_text_embeddings
        model.attach_text_embeddings(*load_text_embeddings(store, model.config.text_embedding_key,
                                                           model.config.text_embedding_sha256))
    return model.eval()


def at_rate(state, fps: float):
    """``state`` resampled to ``fps`` when its median rate differs by more than 1% (``pose.resample``)."""
    rate = 1. / float(np.median(np.diff(state.timestamps))) if len(state.timestamps) > 1 else fps
    if abs(rate - fps) <= .01 * fps:
        return state
    from signtranslator.pose.resample import resample_motion
    return resample_motion(state, fps)


def blanked(states):
    """The same clips with every channel removed: only their durations remain (the evaluator's control)."""
    from dataclasses import replace
    return [replace(s, channels={n: replace(c, valid=np.zeros_like(c.valid), confidence=np.zeros_like(c.confidence),
                                            observed=np.zeros_like(c.observed), values=np.zeros_like(c.values))
                                 for n, c in s.channels.items()}) for s in states]


@torch.no_grad()
def retrieval_scores(model, states, texts, *, batch: int = 32) -> dict:
    """R-precision (groups of 32, fixed seed) of ``states`` against their ``texts``."""
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.text_motion_retrieval import r_precision
    from signtranslator.generation.text_to_motion import encode_texts
    zm, zt = [], []
    for start in range(0, len(states), batch):
        zm.append(model.encode_motion(collate_multichannel(states[start:start + batch])))
        tokens, mask = encode_texts(texts[start:start + batch])
        zt.append(model.encode_text(tokens, mask))
    zm, zt = torch.cat(zm), torch.cat(zt)
    return r_precision(zm, zt, group=min(32, len(zm)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--checkpoint-key")
    parser.add_argument("--lexicon-key", help="instead of a checkpoint: the citation-stitching baseline from this lexicon "
                                              "(build_citation_lexicon.py), on plan records")
    parser.add_argument("--lexicon-sha256")
    parser.add_argument("--partition", default="val")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--guidance", default="1,2", help="diffusion guidance scales to evaluate")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-key")
    parser.add_argument("--temperature", type=float, default=1., help="token generators: sampling temperature")
    parser.add_argument("--part-temperature", help="RVQ generators: per-part temperatures, e.g. body=1.2,left_hand=1.2,"
                                                    "right_hand=1.2,face=1 (overrides --temperature)")
    parser.add_argument("--top-k", type=int, default=0, help="token generators: sample from the k most likely codes (0: all)")
    parser.add_argument("--head-smoothing", type=float, default=0.,
                        help="seconds: centred SO(3) moving average of the generated head rotation (postprocess.py)")
    parser.add_argument("--head-damping", type=float, default=1., help="head excursion factor after smoothing "
                                                                         "(postprocess.damp_rotation_channel)")
    parser.add_argument("--translation-smoothing", type=float, default=0.,
                        help="seconds of centred smoothing on head_translation after the head-rotation post-processing "
                             "(R7, declared 2026-10-24: the round-3b head position shook at 2x real; 0 = off)")
    parser.add_argument("--retrieval-key", help="frozen text-motion retrieval checkpoint: adds R-precision of the "
                                                 "generated motion and of the real motion on the same clips")
    parser.add_argument("--retrieval-manifest-sha256",
                        help="the manifest the retrieval model was trained on, when it is not --manifest-sha256")
    parser.add_argument("--corpus", help="mixed manifests: evaluate only records of this corpus")
    parser.add_argument("--signer-component", help="evaluate only records of this fold-plan signer component "
                                                   "(repartition_manifest.py; multi-signer validation)")
    parser.add_argument("--batch", type=int, default=32, help="clips generated together (token generators; 1 = one "
                                                               "at a time as before)")
    parser.add_argument("--face-calibration-key", help="store key of a face calibration (postprocess.estimate_face_"
                                                       "calibration JSON) applied to every generated state")
    parser.add_argument("--condition", choices=("text", "english"), default="text",
                        help="what the generator is conditioned on: the record's text (English or a plan line), or "
                             "its English (the pre-plan baseline on plan records)")
    parser.add_argument("--checkpoint-manifest-sha256",
                        help="the manifest the generator was trained on, when it is not --manifest-sha256")
    parser.add_argument("--fps", type=float, help="resample references to this rate and generate at it (the rate the "
                                                  "generator was trained at, for mixed-rate corpora)")
    parser.add_argument("--save-generated", help="store prefix for the generated states (<prefix>/<guidance>/<sign>.npz)")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "manifest.json"
        path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    temperature = args.temperature
    if args.part_temperature:
        temperature = {k: float(v) for k, v in (item.split("=") for item in args.part_temperature.split(","))}
    if bool(args.checkpoint_key) == bool(args.lexicon_key):
        raise SystemExit("give exactly one of --checkpoint-key and --lexicon-key")
    if args.lexicon_key:
        from signtranslator.generation.citation_stitch import load_stitcher
        stitcher = load_stitcher(store, args.lexicon_key, args.lexicon_sha256 or "")
        label, trained, sample = ("citation_stitch", ("body", "left_hand", "right_hand", "head_rotation", "face", "blink"),
                                  stitcher.sample)
    else:
        label, trained, sample = load_generator(store, args.checkpoint_key,
                                                args.checkpoint_manifest_sha256 or args.manifest_sha256, args.device,
                                                temperature=temperature, top_k=args.top_k)
    records = sorted((r for r in manifest["records"] if r["partition"] == args.partition and r.get("text")
                      and (args.corpus is None or r.get("corpus") == args.corpus)
                      and (args.signer_component is None or r.get("signer_component") == args.signer_component)
                      and (not args.lexicon_key or r.get("plan"))),
                     key=lambda r: r["sign"])[:args.limit]
    if not records:
        raise SystemExit("no records to evaluate")
    references = [load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]) for r in records]
    if args.fps:
        references = [at_rate(r, args.fps) for r in references]
    channels = tuple(c for c in ("body", "left_hand", "right_hand", "head_rotation", "root_translation",
                                 "head_translation", "left_gaze", "right_gaze") if c in trained)
    scales = [float(g) for g in args.guidance.split(",")] if label not in ("regression", "citation_stitch") else [None]
    retrieval = (load_retrieval(store, args.retrieval_key, args.retrieval_manifest_sha256 or args.manifest_sha256)
                 if args.retrieval_key else None)
    # the generator is conditioned on the record's text (English, or a plan line); meaning is judged against English
    texts = [r.get("english") or r["text"] for r in records]
    reference_retrieval = (dict(real=retrieval_scores(retrieval, references, texts),
                                duration_only=retrieval_scores(retrieval, blanked(references), texts))
                           if retrieval else None)
    calibration = json.loads(store.get(args.face_calibration_key)) if args.face_calibration_key else None
    results = {}
    for scale in scales:
        generated = []
        batched = {}
        if getattr(sample, "batch", None) is not None and args.batch > 1:
            # one batched generation per frame rate, in chunks of --batch clips, seeded per chunk
            jobs = collections.defaultdict(list)
            for i, (record, reference) in enumerate(zip(records, references)):
                fps = 1. / float(np.median(np.diff(reference.timestamps)))
                seconds = float(reference.timestamps[-1] - reference.timestamps[0]) + 1. / fps
                condition = record["text"] if args.condition == "text" else (record.get("english") or record["text"])
                jobs[round(fps, 4)].append((i, condition, seconds))
            chunk = 0
            for fps, items in sorted(jobs.items()):
                for start in range(0, len(items), args.batch):
                    part = items[start:start + args.batch]
                    outs = sample.batch([c for _, c, _ in part], fps, [sec for _, _, sec in part],
                                        scale if scale is not None else 1., args.seed + chunk)
                    batched.update({i: o for (i, _, _), o in zip(part, outs)})
                    chunk += 1
        for index, (record, reference) in enumerate(zip(records, references)):
            fps = 1. / float(np.median(np.diff(reference.timestamps)))
            seconds = float(reference.timestamps[-1] - reference.timestamps[0]) + 1. / fps
            condition = record["text"] if args.condition == "text" else (record.get("english") or record["text"])
            out = (batched[index] if index in batched else
                   sample(condition, fps, seconds, scale if scale is not None else 1., args.steps, args.seed))
            if args.head_smoothing > 0 and "head_rotation" in out.channels:
                from signtranslator.generation.postprocess import smooth_rotation_channel
                out = smooth_rotation_channel(out, "head_rotation", args.head_smoothing)
            if args.head_damping != 1 and "head_rotation" in out.channels:
                from signtranslator.generation.postprocess import damp_rotation_channel
                out = damp_rotation_channel(out, "head_rotation", args.head_damping)
            if args.translation_smoothing > 0 and "head_translation" in out.channels:
                from signtranslator.generation.postprocess import smooth_vector_channel
                out = smooth_vector_channel(out, "head_translation", args.translation_smoothing)
            if calibration is not None:
                from signtranslator.generation.postprocess import calibrate_face
                out = calibrate_face(out, calibration)
            generated.append(out)
        name = label if scale is None else f"{label} guidance {scale:g}"
        results[name] = compare(generated, references, channels)
        if {"body", "left_hand", "right_hand"} <= set(channels) and all(
                "LeftForearm" in r.channels["body"].labels for r in references):
            results[name]["signing_space"] = location_errors(generated, references)
        results[name]["grammar"] = grammar_comparison(generated, references, [r["text"] for r in records])
        results[name]["grammar"]["locus_adherence"] = locus_adherence(generated, [r["text"] for r in records])
        if retrieval:
            results[name]["retrieval"] = dict(generated=retrieval_scores(retrieval, generated, texts),
                                              reference=reference_retrieval)
        if args.save_generated:
            tag = "default" if scale is None else f"guidance-{scale:g}"
            with tempfile.TemporaryDirectory() as temporary:
                for record, state in zip(records, generated):
                    path = Path(temporary) / f"{record['sign']}.npz"
                    state.save(path)
                    store.put(f"{args.save_generated.rstrip('/')}/{tag}/{record['sign']}.npz", path.read_bytes())
        print(json.dumps({name: results[name]}, default=float), flush=True)
    report = dict(schema_version=1, checkpoint_key=args.checkpoint_key, lexicon_key=args.lexicon_key, generator=label,
                  retrieval_key=args.retrieval_key, saved_generated=args.save_generated,
                  temperature=temperature, top_k=args.top_k, head_smoothing=args.head_smoothing, head_damping=args.head_damping,
                  translation_smoothing=args.translation_smoothing,
                  decoder_gains=getattr(sample, "decoder_gains", None) if args.checkpoint_key else None,
                  manifest_sha256=args.manifest_sha256, partition=args.partition, corpus=args.corpus, fps=args.fps,
                  signer_component=args.signer_component,
                  condition=args.condition,
                  clips=len(records),
                  steps=args.steps, results=results)
    if args.output_key:
        store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True, default=float).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
