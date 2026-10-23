"""Instrument acceptance of the 2D-keypoint back-translator, before it may judge generated motion (plan item 7).

The rule was declared before any result (docs/NEXT_PHASE_PLAN.md, "Second blocking event and fallback"). On How2Sign
validation clips, read the same way generated motion will be (canonical state → FK → front projection):
- real corpus BLEU-4 must have a 95% bootstrap interval (1,000 resamples of sentences) lying entirely above the blank
  control's interval (the clips with every channel removed, durations kept);
- and entirely above the shuffled control's (the real translations scored against other clips' sentences).

Also reported, though they do not decide the rule:
- per-sentence agreement with a reference back-translator (``--reference-key``, e.g. v2 T5). This is the Spearman
  correlation of sentence BLEU on the same clips: do the two instruments find the same sentences legible?
- examples.

Exit code 0 if the rule passes, 4 if it fails. The test partition is not read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def sentence_bleu(h: str, r: str) -> float:
    from signtranslator.generation.motion_to_text import corpus_bleu
    return float(corpus_bleu([h], [r])["bleu"])


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) and np.std(rb) else float("nan")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--translator-key", required=True)
    parser.add_argument("--reference-key", help="another back-translator for per-sentence agreement")
    parser.add_argument("--reference-manifest-sha256")
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-key", required=True)
    args = parser.parse_args(argv)
    import torch
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    from evaluate_text_to_motion import at_rate, blanked, load_state
    from score_back_translation import load_translator, shuffled_control, translate
    from strict_prior_review import bleu_ci
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from train_text_to_motion import make_store
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    rows = [r for r in manifest["records"] if r.get("partition") == "val" and r.get("text")
            and (r.get("continuous") or {}).get("status") == "converted"]
    rows = sorted(random.Random(0).sample(rows, min(args.limit, len(rows))), key=lambda r: r["sign"])
    states = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), args.fps)
              for r in rows]
    texts = [r["text"] for r in rows]
    model, kind = load_translator(store, args.translator_key, args.manifest_sha256)
    if kind != "keypoints":
        raise SystemExit("--translator-key is not a 2D-keypoint back-translator")
    model = model.to(device)
    real = translate(model, kind, states, beam=args.beam)
    blank = translate(model, kind, blanked(states), beam=args.beam)
    report = dict(translator_key=args.translator_key, clips=len(rows), beam=args.beam, controls={})
    for name, hyps, refs in (("real", real, texts), ("blank", blank, texts),
                             ("shuffled", real, texts[len(texts) // 2:] + texts[:len(texts) // 2])):
        point, ci = bleu_ci(hyps, refs)
        report["controls"][name] = dict(bleu=point, ci95=ci)
    assert abs(report["controls"]["shuffled"]["bleu"] - round(shuffled_control(real, texts)["bleu"], 3)) < 1e-6
    c = report["controls"]
    report["pass"] = bool(c["real"]["ci95"][0] > c["blank"]["ci95"][1] and c["real"]["ci95"][0] > c["shuffled"]["ci95"][1])
    if args.reference_key:
        ref_model, ref_kind = load_translator(store, args.reference_key, args.reference_manifest_sha256 or args.manifest_sha256)
        ref_model = ref_model.to(device)
        other = translate(ref_model, ref_kind, states, beam=args.beam)
        a = [sentence_bleu(h, r) for h, r in zip(real, texts)]
        b = [sentence_bleu(h, r) for h, r in zip(other, texts)]
        report["agreement"] = dict(reference_key=args.reference_key, spearman_sentence_bleu=round(spearman(a, b), 3),
                                   reference_bleu=bleu_ci(other, texts)[0])
    report["examples"] = list(zip(real[:10], texts[:10]))
    store.put(args.output_key, json.dumps(report, indent=1, sort_keys=True).encode())
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}))
    return 0 if report["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
