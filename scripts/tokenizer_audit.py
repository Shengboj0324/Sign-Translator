"""Deep audit of a motion tokenizer (diagnostic, not a rule): how faithfully real clips survive the round trip, broken
down far beyond the acceptance checks, so a flaw the checks miss is found before a 12-hour prior is trained on it.

Per corpus and How2Sign signer group, on the frames each clip observed (``tokenizer_fidelity_check.observed_like``):
- speed and variation ratios (round trip / real) for body, both hands, head rotation;
- head translation and root translation speed ratio (metres/s);
- jitter: the >= 6 Hz amplitude ratio (``motion_metrics.high_frequency_rms``, R7's measure) per channel;
- face: per-blendshape variation ratio (median and worst 5), the brow blendshapes separately (R6 failed on brow raises);
- codebook usage: distinct codes used per part and stage over all clips, against the codebook size.
Prints one JSON line per group and writes the whole report to ``--output-key`` if given.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

ROT = ("body", "left_hand", "right_hand", "head_rotation")
TRANS = ("head_translation", "root_translation")


def vec_speed(state, name):
    c = state.channels[name]
    t = np.asarray(state.timestamps, float)
    if len(t) < 2:
        return np.nan
    v = np.asarray(c.values, float)
    ok = c.valid[1:] & c.valid[:-1]
    d = np.linalg.norm(np.diff(v, axis=0), axis=-1) / np.diff(t)[:, None]
    return float(np.nanmean(np.where(ok, d, np.nan))) if ok.any() else np.nan


def face_variation(state):
    c = state.channels["face"]
    v = np.where(c.valid[..., None], np.asarray(c.values, float), np.nan)[..., 0]     # (T, 52)
    return np.nanstd(v, axis=0), list(c.labels)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest-key", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--tokenizer-key", required=True)
    p.add_argument("--per-group", type=int, default=40)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-key")
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    import torch
    from signtranslator.data.text_motion_dataset import R2Store
    from signtranslator.data.layout_adapter import conform
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest, load_sample
    from signtranslator.generation.motion_metrics import compare, high_frequency_rms
    from signtranslator.generation.text_to_motion import diffusion_targets
    from evaluate_text_to_motion import at_rate
    from train_motion_tokens import layout_template
    from tokenizer_fidelity_check import observed_like
    from tokenizer_recognisability_check import round_trip
    import diagnose_tokenizer as dt
    store = R2Store(os.environ)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    cache = ContentCache(Path(tempfile.gettempdir()) / "st-audit-cache", 8 * 2 ** 30)
    template = layout_template(manifest, "how2sign-two-view-text-v1", cache, store.get)
    model, _ = dt.load_tokenizer(store, args.tokenizer_key)
    model.eval()                 # round_trip collates on the CPU, as the recognisability check's tokenizer does
    cfg = model.config.motion
    rng = random.Random(0)
    groups = {}
    for g in ("c01", "c03", "c05"):
        groups[f"how2sign-{g}"] = [r for r in manifest["records"] if r.get("corpus") == "how2sign-two-view-text-v1"
                                   and r.get("partition") == "val" and r.get("signer_component") == g]
    for corpus, part in (("youtube-asl-v1", "monitor"), ("asllvd-text-v1", "val")):
        groups[corpus] = [r for r in manifest["records"] if r.get("corpus") == corpus and r.get("partition") == part]
    report = dict(tokenizer_key=args.tokenizer_key, groups={})
    usage = collections.defaultdict(set)
    for name, records in groups.items():
        records = rng.sample(records, min(args.per_group, len(records)))
        real = [at_rate(conform(load_sample(r, "continuous", cache=cache, fetch=store.get), template), 24.) for r in records]
        trips = [round_trip(model, cfg, s, 24.) for s in real]
        masked = [observed_like(t, r) for t, r in zip(trips, real)]
        res = compare(masked, real, ROT)
        row = {f"{c}_{k}": round(float(res[c][k]), 3) for c in ROT if c in res for k in ("speed_ratio", "variation_ratio")}
        for c in TRANS:
            a = np.nanmean([vec_speed(m, c) for m in masked]); b = np.nanmean([vec_speed(r, c) for r in real])
            row[f"{c}_speed_ratio"] = round(float(a / b), 3) if b and np.isfinite(b) else None
        for c in ROT + TRANS:
            hm = [high_frequency_rms(m, c) for m in masked]; hr = [high_frequency_rms(r, c) for r in real]
            hm = [x for x in hm if x is not None]; hr = [x for x in hr if x is not None]
            row[f"{c}_jitter_ratio"] = round(float(np.median(hm) / max(np.median(hr), 1e-12)), 3) if hm and hr else None
        fr = np.nanmedian(np.stack([face_variation(m)[0] for m in masked]), 0)
        ff = np.nanmedian(np.stack([face_variation(r)[0] for r in real]), 0)
        labels = face_variation(real[0])[1]
        ratio = fr / np.maximum(ff, 1e-6)
        order = np.argsort(np.abs(np.log(np.maximum(ratio, 1e-6))))[::-1]
        row["face_variation_ratio_median"] = round(float(np.nanmedian(ratio)), 3)
        row["face_worst5"] = {labels[i]: dict(ratio=round(float(ratio[i]), 3), real_std=round(float(ff[i]), 4),
                                              round_trip_std=round(float(fr[i]), 4)) for i in order[:5]}
        row["brows"] = {labels[i]: round(float(ratio[i]), 3) for i in range(len(labels)) if "brow" in labels[i].lower()}
        with torch.no_grad():
            for s in real[:16]:
                unit = diffusion_targets(cfg, collate_multichannel([s]))
                _, codes, _ = model.quantize(model.encode(unit))      # (quantized, codes (..., J), inputs)
                if isinstance(codes, torch.Tensor):
                    flat = codes.reshape(-1, codes.shape[-1]) if codes.dim() > 1 else codes[:, None]
                    for j in range(flat.shape[-1]):
                        usage[j].update(flat[:, j].tolist())
        row["clips"] = len(records)
        report["groups"][name] = row
        print(json.dumps({name: row}), flush=True)
    book = next((getattr(model.config, k) for k in ("codebook", "codebook_size") if hasattr(model.config, k)), None)
    report["codebook"] = dict(size=book, used_per_code_position=sorted(len(v) for v in usage.values()))
    print(json.dumps({"codebook": report["codebook"]}), flush=True)
    if args.output_key:
        store.put(args.output_key, json.dumps(report, indent=1, sort_keys=True).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
