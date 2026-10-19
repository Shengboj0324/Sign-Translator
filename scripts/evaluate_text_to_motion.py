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
from signtranslator.generation.text_to_motion import TextToMotion, TextToMotionConfig, generate  # noqa: E402
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402
from train_text_to_motion import make_store  # noqa: E402


def load_state(store, key: str, sha256: str) -> MultichannelMotion:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "s.npz"
        path.write_bytes(store.get(key))
        return MultichannelMotion.load(path, expected_sha256=sha256)


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
    args = parser.parse_args(argv)
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "manifest.json"
        path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    checkpoint = torch.load(io.BytesIO(store.get(args.checkpoint_key)), map_location="cpu", weights_only=False)
    if checkpoint["manifest_sha256"] != args.manifest_sha256:
        raise SystemExit("the checkpoint was trained on a different manifest")
    config = TextToMotionConfig.from_dict(checkpoint["config"])
    model = TextToMotion(config)
    model.load_state_dict(checkpoint["model"])
    model.to(torch.device(args.device)).eval()
    records = sorted((r for r in manifest["records"] if r["partition"] == args.partition and r.get("text")),
                     key=lambda r: r["sign"])[:args.limit]
    references = [load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]) for r in records]
    channels = tuple(c for c in ("body", "left_hand", "right_hand", "head_rotation") if c in config.trained_channels)
    scales = [float(g) for g in args.guidance.split(",")] if config.objective == "diffusion" else [None]
    results = {}
    for scale in scales:
        generated = []
        for record, reference in zip(records, references):
            fps = 1. / float(np.median(np.diff(reference.timestamps)))
            seconds = float(reference.timestamps[-1] - reference.timestamps[0]) + 1. / fps
            generated += generate(model, [record["text"]], fps=fps, durations=[seconds], steps=args.steps,
                                  guidance=scale if scale is not None else 1., seed=args.seed)
        label = config.objective if scale is None else f"{config.objective} guidance {scale:g}"
        results[label] = compare(generated, references, channels)
        print(json.dumps({label: results[label]}, default=float), flush=True)
    report = dict(schema_version=1, checkpoint_key=args.checkpoint_key, config_sha256=config.sha256(),
                  manifest_sha256=args.manifest_sha256, partition=args.partition, clips=len(records),
                  steps=args.steps, results=results)
    if args.output_key:
        store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True, default=float).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
