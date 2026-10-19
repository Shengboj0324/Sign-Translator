"""Render real and generated motion side by side for held-out sentences.

For each chosen record of a manifest partition, three skeleton previews are drawn
next to each other — left: the reference (estimated from the video), then one
column per checkpoint (e.g. regression, diffusion) generated from the English
sentence on the reference's own clock — and encoded as one H.264 file per
sentence. Each column is the four-panel preview of ``skeleton_preview`` (front,
side, left hand, right hand). An inspection aid: generated motion is
``observed=False`` and the skeleton uses nominal bone lengths.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_text_to_motion import load_state  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.generation.canonical_skeleton import skeleton_positions  # noqa: E402
from signtranslator.generation.skeleton_preview import encode_preview, render_preview_frames  # noqa: E402
from signtranslator.generation.text_to_motion import TextToMotion, TextToMotionConfig, generate  # noqa: E402
from train_text_to_motion import make_store  # noqa: E402


def load_model(store, key: str, manifest_sha256: str) -> TextToMotion:
    checkpoint = torch.load(io.BytesIO(store.get(key)), map_location="cpu", weights_only=False)
    if checkpoint["manifest_sha256"] != manifest_sha256:
        raise SystemExit(f"{key} was trained on a different manifest")
    model = TextToMotion(TextToMotionConfig.from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--checkpoint-key", action="append", required=True, help="one per generated column")
    parser.add_argument("--partition", default="val")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--min-seconds", type=float, default=3.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "manifest.json"
        path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    models = [load_model(store, key, args.manifest_sha256) for key in args.checkpoint_key]
    records = sorted((r for r in manifest["records"] if r["partition"] == args.partition and r.get("text")
                      and r["continuous"]["seconds"] >= args.min_seconds), key=lambda r: r["sign"])[:args.count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for record in records:
        reference = load_state(store, record["continuous"]["state_key"], record["continuous"]["state_sha256"])
        fps = 1. / float(np.median(np.diff(reference.timestamps)))
        seconds = len(reference.timestamps) / fps
        columns = [reference] + [generate(m, [record["text"]], fps=fps, durations=[seconds], guidance=1.,
                                          seed=args.seed)[0] for m in models]
        rendered = []
        for state in columns:
            joints, positions, bones = skeleton_positions(state)
            rendered.append(render_preview_frames(positions, joints, bones, panel=240, bounds_quantile=.02))
        frames = min(len(r) for r in rendered)
        video = [np.concatenate([r[t] for r in rendered], axis=1) for t in range(frames)]
        out = args.output_dir / f"{record['sign']}.mp4"
        encode_preview(video, np.asarray(reference.timestamps[:frames]) - reference.timestamps[0], out)
        index.append(dict(file=out.name, text=record["text"], seconds=round(frames / fps, 2),
                          columns=["reference (video estimate)"] + [m.config.objective for m in models]))
        print(json.dumps(index[-1]), flush=True)
    (args.output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
