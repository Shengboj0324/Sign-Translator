"""Latency of the translation service on real sentences (plan item 6): per-stage p50 / p95 / max in milliseconds.

Loads the release in process (``serving.service``), warms it up on one sentence, then translates ``--count``
validation sentences of a manifest one after another and reports the distribution of every stage's wall time, the
real-time factor (generation time / signed duration) and the environment (device, threads, torch version). Writes the
report to ``--output`` (JSON).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    return dict(n=len(v), p50=round(float(np.percentile(v, 50)), 1), p95=round(float(np.percentile(v, 95)), 1),
                max=round(float(v.max()), 1), mean=round(float(v.mean()), 1))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--field", default="text", help="record field holding the English (text or english)")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from signtranslator.serving.release import Release
    from signtranslator.serving.service import TranslatorService
    from train_text_to_motion import make_store
    store = make_store(args.store)
    release = Release.from_dict(json.loads(args.release.read_text()))
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    sentences = sorted({(r.get(args.field) or "").strip() for r in manifest["records"]
                        if r.get("partition") == "val" and (r.get(args.field) or "").strip()})[:args.count + 1]
    service = TranslatorService.from_release(store, release, device=args.device)
    service.translate(sentences[0], render=args.render)                  # warm-up, not counted
    runs = [service.translate(s, render=args.render) for s in sentences[1:]]
    stages = sorted({k for r in runs for k in r.timings_ms})
    report = dict(release=release.name, version=release.version, release_sha256=release.sha256(),
                  device=args.device, torch=torch.__version__, threads=torch.get_num_threads(),
                  machine=platform.machine(), sentences=len(runs), render=args.render,
                  stages_ms={k: stats([r.timings_ms[k] for r in runs if k in r.timings_ms]) for k in stages},
                  real_time_factor=stats([r.timings_ms["generate"] / 1000 / r.seconds for r in runs]),
                  examples=[r.summary() for r in runs[:3]])
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("device", "sentences", "stages_ms", "real_time_factor")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
