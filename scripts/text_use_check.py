"""Does a prior use its text? Teacher-forced token loss with the real text vs with an empty text (round 6 audit).

For training records of each ``--corpus`` (up to ``--clips``, fixed seed), the frozen tokenizer gives the target codes.
The prior's teacher-forced cross entropy and token accuracy (``rvq_prior_loss``, ``train=False``) are computed twice:
once with each record's own text, once with every text replaced by the empty text. Loci come from the state, as in
training. If the two agree for one-word corpora, the prior has not learned to condition on the word: the failure is
in learning, not in sampling. How2Sign sentences give the reference gap. Writes one JSON report.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-key", required=True)
    p.add_argument("--mix-key", required=True)
    p.add_argument("--mix-sha256", required=True)
    p.add_argument("--corpus", action="append", required=True)
    p.add_argument("--clips", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from functools import partial
    from torch.utils.data import DataLoader, Subset
    from signtranslator.data.text_motion_dataset import R2Store, TextMotionDataset, collate_text_motion
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest
    from signtranslator.generation.motion_rvq import MotionRVQ, RVQPrior, RVQPriorConfig, rvq_prior_loss
    from signtranslator.generation.motion_tokens import _empty_like
    from signtranslator.generation.text_to_motion import PAD
    from train_motion_tokens import layout_template
    from train_text_to_motion import to_device
    store = R2Store(os.environ)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    ck = torch.load(io.BytesIO(store.get(args.checkpoint_key)), map_location="cpu", weights_only=False)
    prior = RVQPrior(RVQPriorConfig.from_dict(ck["config"])); prior.load_state_dict(ck.get("ema") or ck["model"])
    tokenizer = MotionRVQ(prior.config.tokenizer); tokenizer.load_state_dict(ck["tokenizer_state"])
    device = torch.device(args.device)
    prior.to(device).eval(); tokenizer.to(device).eval()
    cache = ContentCache(Path(tempfile.gettempdir()) / "sign-translator-cache", 8 * 2 ** 30)
    template = layout_template(manifest, "how2sign-two-view-text-v1", cache, store.get)
    data = TextMotionDataset(manifest, partition="train", cache=cache, fetch=store.get,
                             max_seconds=prior.config.tokenizer.motion.max_seconds, fps=24., template=template)
    report = dict(checkpoint_key=args.checkpoint_key, corpora={})
    for corpus in args.corpus:
        idx = [k for k, r in enumerate(data.records) if r.get("corpus") == corpus]
        idx = random.Random(0).sample(idx, min(args.clips, len(idx)))
        loader = DataLoader(Subset(data, idx), batch_size=args.batch, num_workers=4,
                            collate_fn=partial(collate_text_motion, loci=prior.config.locus_conditioning))
        sums = {"text": [0., 0., 0], "empty": [0., 0., 0]}
        with torch.no_grad():
            for batch in loader:
                tokens, mask, motion, texts, *loci = batch
                for name in ("text", "empty"):
                    tk = tokens if name == "text" else _empty_like(tokens)   # the empty text, as in training's dropout
                    mk = tk != PAD
                    tk, mk, mo = to_device((tk, mk, motion, None), device)[:3]
                    lo = [l.to(device) for l in loci]
                    _, terms = rvq_prior_loss(prior, tokenizer, tk, mk, mo, *lo, train=False)
                    sums[name][0] += terms["tokens"] * len(texts); sums[name][1] += terms["token_accuracy"] * len(texts)
                    sums[name][2] += len(texts)
        out = {n: dict(cross_entropy=round(v[0] / v[2], 4), token_accuracy=round(v[1] / v[2], 4), clips=v[2])
               for n, v in sums.items()}
        out["text_gain_nats"] = round(out["empty"]["cross_entropy"] - out["text"]["cross_entropy"], 4)
        report["corpora"][corpus] = out
        print(json.dumps({corpus: out}), flush=True)
    store.put(args.output_key, json.dumps(report, indent=2).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
