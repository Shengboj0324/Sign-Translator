"""Smoothness rule R7 (declared 2026-10-24, 17:55 PDT, before the round-3b verdict): generated motion must not shake.

The owner saw the round-3b avatar's head shake while signing. No gate or review check measured it: the articulation
ratios compare mean speeds. Measured on 30 validation sentences (sweep h1.3-f0.8, c03, guidance 2) against the same
sentences' real clips, the share of motion energy at or above 6 Hz was 6.2% vs 0.6% for head translation (the
tokenizer's round trip alone gives 2.9%) and 3.6% vs 0.8% for head rotation; body and hands matched real. In
absolute amplitude (``motion_metrics.high_frequency_rms``, the rule's measure) head translation was 2.0x real and
head rotation 1.05x. Head translation was never smoothed. A 0.2 s window on both head channels brings every channel
to at most 0.73x.

**Rule:** for each channel in ``CHANNELS``, the median over clips of the generated high-frequency amplitude, after the candidate's own
post-processing, must be at most ``RATIO`` (1.5) times the median of the real clips of the same sentences, in every
signer group. A failure blocks release like R1-R6.

Reads an evaluation's saved generated states (``--generated-prefix``, one guidance directory) and the real states of
the same clips through the manifest; applies the stated head smoothing / damping / translation smoothing; writes
every number and the verdict to ``--output-key``.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signtranslator.generation.motion_metrics import high_frequency_rms  # noqa: E402

CHANNELS = ("body", "left_hand", "right_hand", "head_rotation", "head_translation", "root_translation")
RATIO = 1.5


def load_state(blob: bytes):
    """A saved state, verified and rebuilt by ``MultichannelMotion.load`` (no pickle)."""
    import hashlib
    import tempfile
    from signtranslator.pose.multichannel import MultichannelMotion
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.npz"
        path.write_bytes(blob)
        return MultichannelMotion.load(path, expected_sha256=hashlib.sha256(blob).hexdigest())


def post_process(state, head_smoothing: float, head_damping: float, translation_smoothing: float):
    from signtranslator.generation.postprocess import (damp_rotation_channel, smooth_rotation_channel,
                                                       smooth_vector_channel)
    if head_smoothing > 0 and "head_rotation" in state.channels:
        state = smooth_rotation_channel(state, "head_rotation", head_smoothing)
    if head_damping != 1 and "head_rotation" in state.channels:
        state = damp_rotation_channel(state, "head_rotation", head_damping)
    if translation_smoothing > 0 and "head_translation" in state.channels:
        state = smooth_vector_channel(state, "head_translation", translation_smoothing)
    return state


def judge(generated: dict[str, list[float]], real: dict[str, list[float]], ratio: float = RATIO) -> dict:
    """Per channel: medians, their ratio and the pass; ``pass`` overall when every measured channel passes."""
    out, ok = {}, True
    for c in CHANNELS:
        g = [v for v in generated.get(c, []) if v is not None]
        r = [v for v in real.get(c, []) if v is not None]
        if not g or not r:
            continue
        mg, mr = float(np.median(g)), float(np.median(r))
        passed = mg <= ratio * mr
        ok &= passed
        out[c] = dict(generated=round(mg, 5), real=round(mr, 5), ratio=round(mg / max(mr, 1e-12), 3), clips=len(g),
                      **{"pass": passed})
    return dict(channels=out, limit_ratio=ratio, **{"pass": ok and bool(out)})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--generated-prefix", action="append", required=True,
                   help="saved generated states of one signer group and guidance; repeat per group")
    p.add_argument("--manifest-key", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--head-smoothing", type=float, default=0.)
    p.add_argument("--head-damping", type=float, default=1.)
    p.add_argument("--translation-smoothing", type=float, default=0.)
    p.add_argument("--limit", type=int, default=0, help="clips per group (0: all)")
    p.add_argument("--output-key")
    args = p.parse_args(argv)
    import hashlib
    import os
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    blob = store.get(args.manifest_key)
    if hashlib.sha256(blob).hexdigest() != args.manifest_sha256:
        raise SystemExit("manifest SHA-256 mismatch")
    by_sign = {r["sign"]: r for r in json.loads(blob)["records"] if r.get("sign")}
    report = dict(rule="R7 smoothness", channels=list(CHANNELS), ratio=RATIO, groups={},
                  post_processing=dict(head_smoothing=args.head_smoothing, head_damping=args.head_damping,
                                       translation_smoothing=args.translation_smoothing))
    overall = True
    for prefix in args.generated_prefix:
        keys = sorted(k for k in store.list(prefix.rstrip("/") + "/") if k.endswith(".npz"))
        if args.limit:
            keys = keys[:args.limit]
        if not keys:
            raise SystemExit(f"no generated states under {prefix}")
        gen, real = {c: [] for c in CHANNELS}, {c: [] for c in CHANNELS}
        for k in keys:
            record = by_sign.get(Path(k).stem)
            if record is None:
                raise SystemExit(f"{k}: clip not in the manifest")
            cont = record["continuous"]
            rblob = store.get(cont["state_key"])
            if hashlib.sha256(rblob).hexdigest() != cont["state_sha256"]:
                raise SystemExit(f"{cont['state_key']}: state SHA-256 mismatch")
            g = post_process(load_state(store.get(k)), args.head_smoothing, args.head_damping,
                             args.translation_smoothing)
            r = load_state(rblob)
            for c in CHANNELS:
                gen[c].append(high_frequency_rms(g, c))
                real[c].append(high_frequency_rms(r, c))
        verdict = judge(gen, real)
        overall &= verdict["pass"]
        report["groups"][prefix] = verdict
        print(json.dumps({prefix: {c: (v["generated"], v["real"], v["pass"]) for c, v in verdict["channels"].items()}}),
              flush=True)
    report["pass"] = overall
    print(json.dumps({"R7 pass": overall}))
    if args.output_key:
        store.put(args.output_key, json.dumps(report, indent=2, sort_keys=True).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
