"""Train the motion -> English back-translator on real pairs (frozen afterwards; scores generated motion by BLEU).

Same store, manifest and checkpoint layout as the generators and the retrieval evaluator
(``training-runs/<run>/checkpoints``, ``log``, ``best.json``). The vocabulary is built from the training
partition's sentences only and stored with the run. Validation (EMA weights) reports the label-smoothed loss,
token accuracy and greedy corpus BLEU-4 on a fixed set of up to ``--val-clips`` validation clips; the checkpoint
with the highest validation BLEU is recorded in ``best.json``, and ``--patience`` stops after that many
validations without a new best. The test partition is never read here.
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
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.text_motion_dataset import TextMotionDataset, collate_text_motion  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest  # noqa: E402
from signtranslator.generation.canonical_state import channel_bindings_from_state  # noqa: E402
from signtranslator.generation.motion_to_text import (  # noqa: E402
    MotionToText, MotionToTextConfig, WordVocabulary, corpus_bleu, encode_batch)
from signtranslator.generation.text_to_motion import TextToMotionConfig  # noqa: E402
from signtranslator.training.weight_ema import WeightEMA  # noqa: E402
from train_text_to_motion import lr_at, make_store, to_device  # noqa: E402

KIND = "motion_to_text"


@torch.no_grad()
def evaluate(model, vocabulary, batches, device, *, max_tokens) -> dict:
    """Loss, token accuracy and greedy BLEU-4 over ``batches`` in eval mode."""
    model.eval()
    losses, accuracies, hypotheses, references = [], [], [], []
    for batch in batches:
        motion, texts = to_device(batch, device)[2], batch[3]
        tokens = encode_batch(vocabulary, texts, max_tokens=max_tokens).to(device)
        loss, terms = model.loss(motion, tokens)
        losses.append(float(loss)); accuracies.append(float(terms["token_accuracy"]))
        hypotheses += [vocabulary.decode(ids) for ids in model.generate(motion, beam=1)]
        references += list(texts)
    model.train()
    bleu = corpus_bleu(hypotheses, references)
    return {"val/loss": float(np.mean(losses)), "val/token_accuracy": float(np.mean(accuracies)),
            "val/bleu": bleu["bleu"], "val/bleu_1": bleu["precisions"][0], "examples": list(zip(hypotheses[:5], references[:5]))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--label-smoothing", type=float, default=.1)
    parser.add_argument("--max-words", type=int, default=8000)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=20.)
    parser.add_argument("--trained-channels", default="body,left_hand,right_hand,head_rotation,face,blink,root_translation,head_translation,left_gaze,right_gaze")
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--val-clips", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-bytes", type=int, default=8 * 2 ** 30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "manifest.json"
        path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    cache = ContentCache(Path(os.environ.get("SIGN_CACHE_DIR", tempfile.gettempdir())) / "sign-translator-cache",
                         args.cache_bytes)
    train = TextMotionDataset(manifest, partition="train", cache=cache, fetch=store.get, max_seconds=args.max_seconds)
    val = TextMotionDataset(manifest, partition="val", cache=cache, fetch=store.get, max_seconds=args.max_seconds)
    if not len(train) or not len(val):
        raise SystemExit("the manifest needs training and validation records")
    prefix = f"training-runs/{args.run}"
    if store.exists(f"{prefix}/run.json"):
        vocabulary = WordVocabulary(json.loads(store.get(f"{prefix}/run.json"))["vocabulary"]["words"])
    else:
        vocabulary = WordVocabulary.build([r["text"] for r in train.records], max_words=args.max_words,
                                          min_count=args.min_count)
    motion = TextToMotionConfig(bindings=channel_bindings_from_state(train[0][1]),
                                trained_channels=tuple(args.trained_channels.split(",")), max_seconds=args.max_seconds)
    config = MotionToTextConfig(motion, len(vocabulary), d_model=args.d_model, heads=args.heads,
                                encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
                                feedforward=args.feedforward, dropout=args.dropout, downsample=args.downsample,
                                max_tokens=args.max_tokens, label_smoothing=args.label_smoothing)
    model = MotionToText(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step, resumed_ema = 0, None
    checkpoints = [k for k in store.list(f"{prefix}/checkpoints/") if k.endswith(".pt")]
    if checkpoints:
        state = torch.load(io.BytesIO(store.get(checkpoints[-1])), map_location=device, weights_only=False)
        if state["config_sha256"] != config.sha256() or state["manifest_sha256"] != args.manifest_sha256 \
                or state["vocabulary_sha256"] != vocabulary.sha256():
            raise SystemExit("the newest checkpoint was trained with a different config, manifest or vocabulary")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        resumed_ema, step = state.get("ema"), state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed {checkpoints[-1]} at step {step}", flush=True)
    else:
        store.put(f"{prefix}/run.json", json.dumps(dict(
            kind=KIND, config=config.to_dict(), config_sha256=config.sha256(), vocabulary=vocabulary.to_dict(),
            vocabulary_sha256=vocabulary.sha256(), manifest_key=args.manifest_key, manifest_sha256=args.manifest_sha256,
            args=vars(args), train_records=len(train), val_records=len(val)), indent=2, sort_keys=True, default=str).encode())
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate_text_motion, generator=torch.Generator().manual_seed(args.seed + step),
                        persistent_workers=args.workers > 0, drop_last=True)
    clips = sorted(random.Random(1234).sample(range(len(val)), min(args.val_clips, len(val))))
    val_batches = list(DataLoader(Subset(val, clips), batch_size=args.batch_size, shuffle=False, num_workers=0,
                                  collate_fn=collate_text_motion))
    ema = WeightEMA(model, args.ema_decay)
    if resumed_ema is not None:
        ema.shadow = {n: resumed_ema[n].to(device).clone() for n in ema.shadow}
        ema.updates = step
    best = json.loads(store.get(f"{prefix}/best.json")) if store.exists(f"{prefix}/best.json") else None
    stale, log, started = 0, [], time.monotonic()

    def validate() -> dict:
        live = ema.swap(model)
        metrics = evaluate(model, vocabulary, val_batches, device, max_tokens=args.max_tokens)
        ema.restore(model, live)
        return metrics

    def checkpoint(metrics=None):
        nonlocal best
        buffer = io.BytesIO()
        torch.save(dict(kind=KIND, model=model.state_dict(), ema=ema.state_dict(model), optimizer=optimizer.state_dict(),
                        step=step, validation=metrics, torch_rng=torch.get_rng_state(), config=config.to_dict(),
                        config_sha256=config.sha256(), vocabulary=vocabulary.to_dict(),
                        vocabulary_sha256=vocabulary.sha256(), manifest_sha256=args.manifest_sha256), buffer)
        key = f"{prefix}/checkpoints/step-{step:08d}.pt"
        store.put(key, buffer.getvalue())
        store.put(f"{prefix}/log/step-{step:08d}.json", json.dumps(log, sort_keys=True, default=str).encode())
        if metrics and (best is None or metrics["val/bleu"] > best["val_bleu"]):
            best = dict(checkpoint_key=key, step=step, val_bleu=metrics["val/bleu"], validation=metrics)
            store.put(f"{prefix}/best.json", json.dumps(best, indent=2, sort_keys=True, default=str).encode())

    model.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            motion_batch, texts = to_device(batch, device)[2], batch[3]
            tokens = encode_batch(vocabulary, texts, max_tokens=args.max_tokens).to(device)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, base=args.lr, warmup=args.warmup, total=args.steps)
            loss, terms = model.loss(motion_batch, tokens)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            ema.update(model)
            step += 1
            entry = dict(step=step, loss=float(loss.detach()), grad_norm=float(norm),
                         seconds=round(time.monotonic() - started, 1), **{f"train/{k}": float(v) for k, v in terms.items()})
            metrics = None
            if step % args.val_every == 0 or step == args.steps:
                metrics = validate()
                entry.update({k: v for k, v in metrics.items() if k != "examples"})
                improved = best is None or metrics["val/bleu"] > best["val_bleu"]
                stale = 0 if improved else stale + 1
                print(json.dumps(dict(step=step, examples=metrics["examples"])), flush=True)
            log.append(entry)
            if step % 100 == 0 or step == 1 or metrics:
                print(json.dumps({k: v for k, v in entry.items()}), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps or (metrics and stale == 0):
                checkpoint(metrics)
            if metrics and stale >= args.patience:
                print(f"early stop at step {step}: no new best validation BLEU in {stale} validations", flush=True)
                checkpoint(metrics)
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
