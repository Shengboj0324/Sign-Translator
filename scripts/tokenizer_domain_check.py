"""Does a frozen motion tokenizer still encode a new corpus? (plan item 7, before a prior trains on YouTube-ASL).

The full-corpus tokenizer was trained on How2Sign two-view states. A prior trained on another corpus can only be as
good as the codes it predicts, so before mixing a corpus in, this reconstructs up to ``--limit`` clips of each
``--corpus`` of a (mixed) manifest - conformed to ``--conform-to-corpus``'s layouts and resampled to ``--fps``, as
training sees them - and compares, per channel, the mean absolute reconstruction error in the tokenizer's unit
space with the reference corpus's. Pass rule (fixed here): every channel's error ratio new/reference is at most
``--max-ratio`` (1.5); otherwise the tokenizer must be retrained on the mix first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.layout_adapter import conform  # noqa: E402
from signtranslator.data.multichannel import collate_multichannel  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.generation.text_to_motion import diffusion_layout, diffusion_targets  # noqa: E402


@torch.no_grad()
def channel_errors(model, states, batch: int = 16) -> dict:
    import diagnose_tokenizer as dt
    cfg = model.config.motion
    layout = diffusion_layout(cfg)
    sums, counts, scale = {}, {}, {}
    for i in range(0, len(states), batch):
        b = collate_multichannel(states[i:i + batch])
        unit = diffusion_targets(cfg, b)
        rec = dt.reconstructions(model, unit)["full"]
        fv = b.frame_valid
        off = 0
        for entry in layout:
            name, width = entry[0], entry[3]
            valid = b.channels[name].valid.any(-1) & fv if name in b.channels else fv
            if valid.any():
                err = (rec[..., off:off + width] - unit[..., off:off + width]).abs().mean(-1)[valid]
                sums[name] = sums.get(name, 0.) + float(err.sum()); counts[name] = counts.get(name, 0) + int(valid.sum())
                x = unit[..., off:off + width]   # signal scale: mean absolute deviation from each clip's own mean
                m = (x * valid[..., None]).sum(1, keepdim=True) / valid.sum(1).clamp(min=1)[:, None, None]
                scale[name] = scale.get(name, 0.) + float((x - m).abs().mean(-1)[valid].sum())
            off += width
    return {k: sums[k] / counts[k] for k in sums}, {k: scale[k] / counts[k] for k in scale}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--tokenizer-key", required=True)
    parser.add_argument("--conform-to-corpus", required=True)
    parser.add_argument("--corpus", action="append", required=True, help="corpus to check (repeatable)")
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--max-ratio", type=float, default=1.5)
    parser.add_argument("--output-key", required=True)
    args = parser.parse_args(argv)
    import diagnose_tokenizer as dt
    from evaluate_text_to_motion import at_rate, load_state
    from train_text_to_motion import make_store
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    model, _ = dt.load_tokenizer(store, args.tokenizer_key)
    rng = np.random.default_rng(0)

    def sample_of(corpus, partitions):
        rows = [r for r in manifest["records"] if r.get("corpus") == corpus and r.get("partition") in partitions]
        return [rows[i] for i in rng.choice(len(rows), min(args.limit, len(rows)), replace=False)] if rows else []
    reference_rows = sample_of(args.conform_to_corpus, ("val",))
    if not reference_rows:
        raise SystemExit("no reference validation clips")
    template = load_state(store, reference_rows[0]["continuous"]["state_key"], reference_rows[0]["continuous"]["state_sha256"])

    def states_of(rows):
        out = []
        for r in rows:
            s = load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"])
            out.append(at_rate(conform(s, template), args.fps))
        return out
    reference, reference_scale = channel_errors(model, states_of(reference_rows))
    report = dict(reference_corpus=args.conform_to_corpus, reference=reference, reference_scale=reference_scale,
                  corpora={}, max_ratio=args.max_ratio)
    ok = True
    for corpus in args.corpus:
        rows = sample_of(corpus, ("val", "monitor", "train"))
        errors, scale = channel_errors(model, states_of(rows))
        ratios = {k: round(errors[k] / max(reference.get(k, 1e-9), 1e-9), 3) for k in errors if k in reference}
        passed = all(v <= args.max_ratio for v in ratios.values())
        ok &= passed
        # diagnostic only (the pass rule is the absolute ratio): error relative to each corpus's own signal scale, so
        # a channel that simply moves more in the new corpus is told apart from one the tokenizer encodes worse
        relative = {k: round((errors[k] / max(scale[k], 1e-9)) / max(reference[k] / max(reference_scale[k], 1e-9), 1e-9), 3)
                    for k in errors if k in reference and k in scale and k in reference_scale}
        scale_ratios = {k: round(scale[k] / max(reference_scale[k], 1e-9), 3) for k in scale if k in reference_scale}
        report["corpora"][corpus] = dict(clips=len(rows), errors=errors, ratios=ratios, scale=scale,
                                         scale_ratios=scale_ratios, relative_ratios=relative, **{"pass": passed})
    report["pass"] = ok
    store.put(args.output_key, json.dumps(report, indent=1, sort_keys=True).encode())
    print(json.dumps({c: dict(ratios=v["ratios"], scale_ratios=v["scale_ratios"], relative_ratios=v["relative_ratios"],
                              passed=v["pass"]) for c, v in report["corpora"].items()}))
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
