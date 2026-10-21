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


def load_generator(store, key: str, manifest_sha256: str, device: str = "cpu"):
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

        def sample(text, fps, seconds, guidance, steps, seed):
            return generate_rvq_motion(prior, tokenizer, [text], fps=fps, durations=[seconds], guidance=guidance,
                                       seed=seed)[0]
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
    return model.eval()


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
    parser.add_argument("--checkpoint-key", required=True)
    parser.add_argument("--partition", default="val")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--guidance", default="1,2", help="diffusion guidance scales to evaluate")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-key")
    parser.add_argument("--retrieval-key", help="frozen text-motion retrieval checkpoint: adds R-precision of the "
                                                 "generated motion and of the real motion on the same clips")
    parser.add_argument("--save-generated", help="store prefix for the generated states (<prefix>/<guidance>/<sign>.npz)")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "manifest.json"
        path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    label, trained, sample = load_generator(store, args.checkpoint_key, args.manifest_sha256, args.device)
    records = sorted((r for r in manifest["records"] if r["partition"] == args.partition and r.get("text")),
                     key=lambda r: r["sign"])[:args.limit]
    references = [load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]) for r in records]
    channels = tuple(c for c in ("body", "left_hand", "right_hand", "head_rotation", "root_translation",
                                 "head_translation", "left_gaze", "right_gaze") if c in trained)
    scales = [float(g) for g in args.guidance.split(",")] if label != "regression" else [None]
    retrieval = load_retrieval(store, args.retrieval_key, args.manifest_sha256) if args.retrieval_key else None
    texts = [r["text"] for r in records]
    reference_retrieval = (dict(real=retrieval_scores(retrieval, references, texts),
                                duration_only=retrieval_scores(retrieval, blanked(references), texts))
                           if retrieval else None)
    results = {}
    for scale in scales:
        generated = []
        for record, reference in zip(records, references):
            fps = 1. / float(np.median(np.diff(reference.timestamps)))
            seconds = float(reference.timestamps[-1] - reference.timestamps[0]) + 1. / fps
            generated.append(sample(record["text"], fps, seconds, scale if scale is not None else 1., args.steps,
                                    args.seed))
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
    report = dict(schema_version=1, checkpoint_key=args.checkpoint_key, generator=label,
                  retrieval_key=args.retrieval_key, saved_generated=args.save_generated,
                  manifest_sha256=args.manifest_sha256, partition=args.partition, clips=len(records),
                  steps=args.steps, results=results)
    if args.output_key:
        store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True, default=float).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
