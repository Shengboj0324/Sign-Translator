"""Train the 2D-keypoint back-translator (``generation.keypoints_to_text_t5``; plan item 7 fallback).

Training layout and rules follow ``train_motion_to_text_t5.py``:
- runs live under ``training-runs/<run>/`` (``checkpoints``, ``log``, ``best.json``);
- the best checkpoint is chosen by validation BLEU-4 (beam 1, EMA weights, a fixed subset of up to ``--val-clips``);
- ``--patience`` validations without a new best stop the run, and the test partitions are never read.

Data comes from keypoint shard indexes (``data.keypoint_shards``):
- **Training:** ``--train-index`` (repeatable; partition ``train``), with augmentation.
- **Selection:** ``--val-index`` (partition ``val``). This is How2Sign through forward kinematics, the domain the
  instrument judges.
- **Report only:** ``--monitor-index`` (partition ``monitor``, e.g. UWB dev), logged and never used for selection.

The two stages:
1. **Pretraining:** UWB YouTube-ASL keypoints, validated zero-shot on How2Sign.
2. **Fine-tuning:** How2Sign train, from ``--init-key`` (same config, SHA-checked).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.keypoint_shards import KeypointShardDataset, collate_keypoints  # noqa: E402
from signtranslator.generation.keypoints_to_text_t5 import KIND, KeypointsToT5Config, build  # noqa: E402
from signtranslator.generation.motion_to_text import corpus_bleu  # noqa: E402
from signtranslator.training.weight_ema import WeightEMA  # noqa: E402
from train_text_to_motion import lr_at, make_store  # noqa: E402


def to_device(batch, device):
    (x, valid), texts = batch
    return (x.to(device, non_blocking=True), valid.to(device, non_blocking=True)), texts


@torch.no_grad()
def evaluate(model, batches, device, *, beam: int = 1, prefix: str = "val") -> dict:
    model.eval()
    losses, accuracies, hypotheses, references = [], [], [], []
    for batch in batches:
        inputs, texts = to_device(batch, device)
        loss, terms = model.loss(inputs, texts)
        losses.append(float(loss)); accuracies.append(float(terms["token_accuracy"]))
        hypotheses += model.generate_text(inputs, beam=beam)
        references += list(texts)
    model.train()
    bleu = corpus_bleu(hypotheses, references)
    return {f"{prefix}/loss": float(np.mean(losses)), f"{prefix}/token_accuracy": float(np.mean(accuracies)),
            f"{prefix}/bleu": bleu["bleu"], f"{prefix}/bleu_1": bleu["precisions"][0],
            f"{prefix}_examples": list(zip(hypotheses[:5], references[:5]))}


def index_digest(store, keys) -> str:
    return hashlib.sha256(b"".join(hashlib.sha256(store.get(k)).digest() for k in sorted(keys))).hexdigest()


def main(argv=None, *, build_model=build) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--train-index", action="append", required=True)
    parser.add_argument("--val-index", required=True)
    parser.add_argument("--monitor-index")
    parser.add_argument("--run", required=True)
    parser.add_argument("--t5-name", default="google/flan-t5-base")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-seconds", type=float, default=20.)
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--val-clips", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="clips per dataset (testing)")
    parser.add_argument("--init-key", help="fine-tune from this checkpoint (same config)")
    parser.add_argument("--init-sha256")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    train = KeypointShardDataset(store, args.train_index, partition="train", target_fps=args.fps,
                                 augment=not args.no_augment, max_seconds=args.max_seconds, limit=args.limit)
    val = KeypointShardDataset(store, [args.val_index], partition="val", target_fps=args.fps,
                               max_seconds=args.max_seconds, limit=args.limit)
    monitor = (KeypointShardDataset(store, [args.monitor_index], partition="monitor", target_fps=args.fps,
                                    max_seconds=args.max_seconds, limit=args.limit) if args.monitor_index else None)
    if len(train) < args.batch_size or not len(val):
        raise SystemExit("validation clips and at least one full training batch are needed")
    print(f"train {len(train)} clips, val {len(val)}, monitor {len(monitor) if monitor else 0}", flush=True)
    data_sha = index_digest(store, args.train_index + [args.val_index])
    config = KeypointsToT5Config(fps=args.fps, t5_name=args.t5_name, d_model=args.d_model, heads=args.heads,
                                 encoder_layers=args.encoder_layers, feedforward=args.feedforward, dropout=args.dropout,
                                 downsample=args.downsample, max_tokens=args.max_tokens)
    prefix = f"training-runs/{args.run}"
    checkpoints = [k for k in store.list(f"{prefix}/checkpoints/") if k.endswith(".pt")]
    model = build_model(config, pretrained=not checkpoints).to(device)
    optimizer = torch.optim.AdamW(model.param_groups(motion_lr=args.lr, decoder_lr=args.decoder_lr,
                                                     weight_decay=args.weight_decay))
    step, resumed_ema = 0, None
    if checkpoints:
        state = torch.load(io.BytesIO(store.get(checkpoints[-1])), map_location=device, weights_only=False)
        if state["config_sha256"] != config.sha256() or state["data_sha256"] != data_sha:
            raise SystemExit("the newest checkpoint was trained with a different config or data")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        resumed_ema, step = state.get("ema"), state["step"]
        print(f"resumed {checkpoints[-1]} at step {step}", flush=True)
    else:
        if args.init_key:
            payload = store.get(args.init_key)
            if hashlib.sha256(payload).hexdigest() != args.init_sha256:
                raise SystemExit("the --init-key checkpoint differs from --init-sha256")
            init = torch.load(io.BytesIO(payload), map_location=device, weights_only=False)
            if init.get("kind") != KIND or init["config_sha256"] != config.sha256():
                raise SystemExit("--init-key is not a 2D-keypoint back-translator with this config")
            model.load_state_dict(init.get("ema") or init["model"])
            print(f"initialised from {args.init_key} (step {init.get('step')})", flush=True)
        store.put(f"{prefix}/run.json", json.dumps(dict(
            kind=KIND, config=config.to_dict(), config_sha256=config.sha256(), train_index=args.train_index,
            val_index=args.val_index, monitor_index=args.monitor_index, data_sha256=data_sha, args=vars(args),
            train_clips=len(train), val_clips=len(val)), indent=2, sort_keys=True, default=str).encode())
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate_keypoints, generator=torch.Generator().manual_seed(args.seed + step),
                        persistent_workers=args.workers > 0, drop_last=True, pin_memory=device.type == "cuda")

    def fixed(dataset):
        clips = sorted(random.Random(1234).sample(range(len(dataset)), min(args.val_clips, len(dataset))))
        return list(DataLoader(Subset(dataset, clips), batch_size=args.batch_size, collate_fn=collate_keypoints))
    val_batches = fixed(val)
    monitor_batches = fixed(monitor) if monitor else None
    ema = WeightEMA(model, args.ema_decay)
    if resumed_ema is not None:
        ema.shadow = {n: resumed_ema[n].to(device).clone() for n in ema.shadow}
        ema.updates = step
    best = json.loads(store.get(f"{prefix}/best.json")) if store.exists(f"{prefix}/best.json") else None
    stale, log, started = 0, [], time.monotonic()

    def validate() -> dict:
        live = ema.swap(model)
        metrics = evaluate(model, val_batches, device)
        if monitor_batches:
            metrics.update(evaluate(model, monitor_batches, device, prefix="monitor"))
        ema.restore(model, live)
        return metrics

    def checkpoint(metrics=None):
        nonlocal best
        buffer = io.BytesIO()
        torch.save(dict(kind=KIND, model=model.state_dict(), ema=ema.state_dict(model), optimizer=optimizer.state_dict(),
                        step=step, validation=metrics, config=config.to_dict(), config_sha256=config.sha256(),
                        data_sha256=data_sha, train_index=args.train_index, val_index=args.val_index), buffer)
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
            inputs, texts = to_device(batch, device)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, base=group["base_lr"], warmup=args.warmup, total=args.steps)
            loss, terms = model.loss(inputs, texts)
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
                entry.update({k: v for k, v in metrics.items() if not k.endswith("examples")})
                improved = best is None or metrics["val/bleu"] > best["val_bleu"]
                stale = 0 if improved else stale + 1
                print(json.dumps(dict(step=step, examples=metrics["val_examples"])), flush=True)
            log.append(entry)
            if step % 100 == 0 or step == 1 or metrics:
                print(json.dumps(entry), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps or (metrics and stale == 0):
                checkpoint(metrics)
            if metrics and stale >= args.patience:
                print(f"early stop at step {step}: no new best validation BLEU in {stale} validations", flush=True)
                checkpoint(metrics)
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
