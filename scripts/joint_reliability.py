"""Which joints' frame-to-frame velocity in the real reference is signal, and which is extraction jitter? (round 3d)

On ``--limit`` training clips of one corpus (fixed seed), per joint of each rotation channel:
- the lag-1 autocorrelation of frame velocity, mean over valid triples (about -0.5 for white positional noise, near 1
  for smooth motion);
- the share of the median angular speed that survives a 0.1 s smoothing (true motion survives, jitter does not).
Round 3c's diagnosis found How2Sign's torso joints jitter-dominated (autocorrelation below zero, speed halved by the
smoothing), so a speed ratio over them measures whether a tokenizer reproduces noise. The report lists, per channel,
the joints whose autocorrelation exceeds ``--threshold``; round 3d's fidelity check measures body speed over them.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def velocity_autocorrelation(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(J,) mean cosine between consecutive frame differences of (T, J, D) values, over triples valid throughout."""
    d = np.diff(np.asarray(values, float), axis=0)
    ok = valid[2:] & valid[1:-1] & valid[:-2]
    num = (d[1:] * d[:-1]).sum(-1)
    den = np.sqrt((d[1:] ** 2).sum(-1) * (d[:-1] ** 2).sum(-1)) + 1e-12
    return np.nanmean(np.where(ok, num / den, np.nan), axis=0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest-key", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--corpus", default="how2sign-two-view-text-v1")
    p.add_argument("--partition", default="train")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--threshold", type=float, default=.1)
    p.add_argument("--channels", default="body,left_hand,right_hand")
    p.add_argument("--output-key")
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from signtranslator.generation.motion_metrics import angular_speeds
    from signtranslator.generation.postprocess import smooth_rotation_channel
    from evaluate_text_to_motion import at_rate, load_state
    store = R2Store(os.environ)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    records = [r for r in manifest["records"] if r.get("corpus") == args.corpus and r.get("partition") == args.partition]
    records = random.Random(0).sample(records, min(args.limit, len(records)))
    states = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), 24.)
              for r in records]
    report = dict(corpus=args.corpus, partition=args.partition, clips=len(states), threshold=args.threshold, channels={})
    for name in args.channels.split(","):
        ac, raw, smooth = [], [], []
        for s in states:
            c = s.channels[name]
            if not c.valid.any():
                continue
            ac.append(velocity_autocorrelation(c.values, c.valid))
            raw.append(angular_speeds(s, name)); smooth.append(angular_speeds(smooth_rotation_channel(s, name, .1), name))
        labels = list(states[0].channels[name].labels)
        ac = np.nanmean(np.stack(ac), axis=0)
        raw, smooth = np.concatenate(raw), np.concatenate(smooth)
        joints = {}
        for j, label in enumerate(labels):
            m = np.isfinite(raw[:, j]) & np.isfinite(smooth[:, j])
            if m.sum() < 50 or not np.isfinite(ac[j]):
                continue
            joints[label] = dict(autocorrelation=round(float(ac[j]), 3),
                                 smoothed_speed_share=round(float(np.median(smooth[m, j]) / np.median(raw[m, j])), 3),
                                 median_speed_deg_s=round(float(np.median(raw[m, j])), 1), pairs=int(m.sum()))
        reliable = [k for k, v in joints.items() if v["autocorrelation"] > args.threshold]
        report["channels"][name] = dict(joints=joints, reliable=reliable)
        print(name, "reliable:", reliable, "| excluded:", [k for k in joints if k not in reliable], flush=True)
    if args.output_key:
        store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
