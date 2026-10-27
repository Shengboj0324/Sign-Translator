"""Teacher-forced word retrieval (round 7, amendment 1): does a prior's likelihood pick a clip's own word?

The per-code text gain (``text_use_check.py``) is bounded by H(word) / codes per clip, about 0.003 nats, so it cannot
tell a prior that ignores its text from one that uses it fully. This asks directly. For each ``--corpus``, ``--clips``
training clips with as many distinct words are drawn (fixed seed). Every clip's frozen-tokenizer codes are scored under
every one of those words (``rvq_sequence_log_likelihood``: summed teacher-forced log-probability, no dropout, no
corruption, the clip's own loci and duration) and under the empty text. Reported per corpus:
- top-1, top-5 and mean rank of the clip's own word (chance: 1/n, 5/n, (n+1)/2);
- the margin, own word minus the mean of the other words, and the gain over the empty text, in nats per clip;
- the same ranks for a shuffled control (each clip's "own" word replaced by a derangement of the words), which must read
  at chance.
The score matrices are kept in the JSON report.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def ranks(scores: np.ndarray, own: np.ndarray) -> np.ndarray:
    """1-based rank of column ``own[i]`` in row i (ties count against it)."""
    target = scores[np.arange(len(scores)), own]
    return 1 + (scores > target[:, None]).sum(1) + ((scores == target[:, None]).sum(1) - 1)


def summary(r: np.ndarray) -> dict:
    return dict(top1=round(float((r == 1).mean()), 4), top5=round(float((r <= 5).mean()), 4),
                mean_rank=round(float(r.mean()), 2))


def derangement(n: int, rng: random.Random) -> np.ndarray:
    while True:
        p = list(range(n)); rng.shuffle(p)
        if all(i != v for i, v in enumerate(p)):
            return np.array(p)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-key", required=True)
    p.add_argument("--mix-key", required=True)
    p.add_argument("--mix-sha256", required=True)
    p.add_argument("--corpus", action="append", required=True)
    p.add_argument("--clips", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zero-loci", action="store_true", help="score every row with zero locus slots, the condition "
                   "one-word generation uses (plan_loci of a single word is empty), instead of the clip's measured loci")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.data.text_motion_dataset import R2Store, TextMotionDataset
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest
    from signtranslator.generation.motion_rvq import MotionRVQ, RVQPrior, RVQPriorConfig, rvq_sequence_log_likelihood
    from signtranslator.generation.motion_tokens import _empty_like
    from signtranslator.generation.text_to_motion import PAD, encode_texts
    from signtranslator.grammar_motion.loci_conditioning import state_loci
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
    report = dict(checkpoint_key=args.checkpoint_key, mix_key=args.mix_key, clips=args.clips, seed=args.seed,
                  zero_loci=args.zero_loci, corpora={})
    for corpus in args.corpus:
        rng = random.Random(args.seed)
        by_word = {}
        for k, r in enumerate(data.records):
            if r.get("corpus") == corpus and r.get("text"):
                by_word.setdefault(r["text"], []).append(k)
        words = rng.sample(sorted(by_word), min(args.clips, len(by_word)))
        idx = [rng.choice(by_word[w]) for w in words]
        n = len(words)
        cand, cmask = encode_texts(words)
        empty = _empty_like(cand[:1])
        tokens = torch.cat((cand, empty), 0); mask = tokens != PAD
        scores, empties, seconds = np.zeros((n, n)), np.zeros(n), []
        for i, k in enumerate(idx):
            text, state = data[k]
            assert text == words[i]
            motion = collate_multichannel([state] * (n + 1))
            loci = None
            if prior.config.locus_conditioning:
                slots = np.zeros_like(state_loci(state)) if args.zero_loci else state_loci(state)
                loci = torch.as_tensor(np.stack([slots] * (n + 1)), dtype=torch.float32).to(device)
            tk, mk, mo = to_device((tokens, mask, motion, None), device)[:3]
            ll = rvq_sequence_log_likelihood(prior, tokenizer, tk, mk, mo, loci).double().cpu().numpy()
            scores[i], empties[i] = ll[:n], ll[n]
            seconds.append(float(state.timestamps[-1] - state.timestamps[0]))
        own = np.arange(n)
        r = ranks(scores, own)
        shuffled = ranks(scores, derangement(n, random.Random(args.seed + 1)))
        others = (scores.sum(1) - scores[own, own]) / max(n - 1, 1)
        out = dict(words=n, chance=dict(top1=round(1 / n, 4), top5=round(min(5, n) / n, 4), mean_rank=(n + 1) / 2),
                   own=summary(r), shuffled_control=summary(shuffled),
                   margin_nats_per_clip=dict(mean=round(float((scores[own, own] - others).mean()), 2),
                                             median=round(float(np.median(scores[own, own] - others)), 2)),
                   gain_over_empty_nats_per_clip=round(float((scores[own, own] - empties).mean()), 2),
                   mean_seconds=round(float(np.mean(seconds)), 3))
        print(json.dumps({corpus: out}), flush=True)
        report["corpora"][corpus] = dict(out, word_list=words, ranks=r.tolist(), scores=np.round(scores, 3).tolist(),
                                         empty_scores=np.round(empties, 3).tolist())
    store.put(args.output_key, json.dumps(report, indent=1).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
