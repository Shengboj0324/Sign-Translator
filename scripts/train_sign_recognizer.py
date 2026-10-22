"""Train the isolated-sign recogniser (plan item 4a) on a gloss-labelled manifest (ASL Citizen).

Same store and checkpoint layout as the other trainers (``training-runs/<run>/checkpoints``, ``log``,
``best.json``). The label set is the training partition's glosses (sorted, stored with the run). Validation (EMA
weights) reports loss and top-1/top-5 accuracy on a fixed set of up to ``--val-clips`` validation clips; the
checkpoint with the highest validation top-1 is recorded in ``best.json``; ``--patience`` stops after that many
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
torch.multiprocessing.set_sharing_strategy("file_system")   # many small tensors per batch exhaust fd passing ("received 0 items of ancdata")
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest  # noqa: E402
from signtranslator.generation.canonical_state import channel_bindings_from_state  # noqa: E402
from signtranslator.generation.sign_recognizer import (GlossMotionDataset, SignRecognizer,  # noqa: E402
                                                       SignRecognizerConfig, collate_gloss_motion, top_k_accuracy)
from signtranslator.generation.text_to_motion import TextToMotionConfig  # noqa: E402
from signtranslator.training.weight_ema import WeightEMA  # noqa: E402
from train_text_to_motion import lr_at, make_store, to_device  # noqa: E402

KIND = "sign_recognizer"


@torch.no_grad()
def evaluate(model, batches, device) -> dict:
    model.eval()
    logits, labels, losses = [], [], []
    for labels_batch, motion in batches:
        motion = to_device((torch.zeros(0), torch.zeros(0), motion, None), device)[2]
        out = model(motion)
        losses.append(float(torch.nn.functional.cross_entropy(out, labels_batch.to(device))))
        logits.append(out.cpu()); labels.append(labels_batch)
    model.train()
    return {"val/loss": float(np.mean(losses)),
            **{f"val/{k}": v for k, v in top_k_accuracy(torch.cat(logits), torch.cat(labels)).items()}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=.1)
    parser.add_argument("--min-hand-fraction", type=float, default=.2,
                        help="leave out clips whose hands are seen in fewer frames (ASL Citizen quality counts)")
    parser.add_argument("--max-seconds", type=float, default=8.)
    parser.add_argument("--trained-channels", default="body,left_hand,right_hand,head_rotation,face")
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--val-clips", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=8)
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
    prefix = f"training-runs/{args.run}"
    if store.exists(f"{prefix}/run.json"):
        glosses = json.loads(store.get(f"{prefix}/run.json"))["labels"]
    else:
        glosses = sorted({r["gloss"]["label"] for r in manifest["records"]
                          if r.get("partition") == "train" and (r.get("gloss") or {}).get("label")})
    labels = {g: i for i, g in enumerate(glosses)}
    kw = dict(labels=labels, cache=cache, fetch=store.get, min_hand_fraction=args.min_hand_fraction)
    train = GlossMotionDataset(manifest, partition="train", **kw)
    val = GlossMotionDataset(manifest, partition="val", **kw)
    keep = lambda d: [i for i, r in enumerate(d.records) if r["continuous"].get("seconds", 0.) <= args.max_seconds]  # noqa: E731
    train_ids, val_ids = keep(train), keep(val)
    if not train_ids or not val_ids:
        raise SystemExit("the manifest needs training and validation records")
    motion = TextToMotionConfig(bindings=channel_bindings_from_state(train[train_ids[0]][1]),
                                trained_channels=tuple(args.trained_channels.split(",")), max_seconds=args.max_seconds)
    config = SignRecognizerConfig(motion, len(glosses), d_model=args.d_model, heads=args.heads, layers=args.layers,
                                  feedforward=args.feedforward, dropout=args.dropout, downsample=args.downsample,
                                  label_smoothing=args.label_smoothing)
    model = SignRecognizer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step, resumed_ema = 0, None
    checkpoints = [k for k in store.list(f"{prefix}/checkpoints/") if k.endswith(".pt")]
    if checkpoints:
        state = torch.load(io.BytesIO(store.get(checkpoints[-1])), map_location=device, weights_only=False)
        if state["config_sha256"] != config.sha256() or state["manifest_sha256"] != args.manifest_sha256:
            raise SystemExit("the newest checkpoint was trained with a different config or manifest")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        resumed_ema, step = state.get("ema"), state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed {checkpoints[-1]} at step {step}", flush=True)
    else:
        store.put(f"{prefix}/run.json", json.dumps(dict(
            kind=KIND, config=config.to_dict(), config_sha256=config.sha256(), labels=glosses,
            manifest_key=args.manifest_key, manifest_sha256=args.manifest_sha256, args=vars(args),
            train_records=len(train_ids), val_records=len(val_ids)), indent=2, sort_keys=True, default=str).encode())
    loader = DataLoader(Subset(train, train_ids), batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate_gloss_motion, generator=torch.Generator().manual_seed(args.seed + step),
                        persistent_workers=args.workers > 0, drop_last=len(train_ids) > args.batch_size)
    chosen = sorted(random.Random(1234).sample(val_ids, min(args.val_clips, len(val_ids))))
    val_batches = list(DataLoader(Subset(val, chosen), batch_size=args.batch_size, shuffle=False, num_workers=0,
                                  collate_fn=collate_gloss_motion))
    ema = WeightEMA(model, args.ema_decay)
    if resumed_ema is not None:
        ema.shadow = {n: resumed_ema[n].to(device).clone() for n in ema.shadow}
        ema.updates = step
    best = json.loads(store.get(f"{prefix}/best.json")) if store.exists(f"{prefix}/best.json") else None
    stale, log, started = 0, [], time.monotonic()

    def validate() -> dict:
        live = ema.swap(model)
        metrics = evaluate(model, val_batches, device)
        ema.restore(model, live)
        return metrics

    def checkpoint(metrics=None):
        nonlocal best
        buffer = io.BytesIO()
        torch.save(dict(kind=KIND, model=model.state_dict(), ema=ema.state_dict(model), optimizer=optimizer.state_dict(),
                        step=step, validation=metrics, torch_rng=torch.get_rng_state(), config=config.to_dict(),
                        config_sha256=config.sha256(), labels=glosses, manifest_sha256=args.manifest_sha256), buffer)
        key = f"{prefix}/checkpoints/step-{step:08d}.pt"
        store.put(key, buffer.getvalue())
        store.put(f"{prefix}/log/step-{step:08d}.json", json.dumps(log, sort_keys=True).encode())
        if metrics and (best is None or metrics["val/top1"] > best["val_top1"]):
            best = dict(checkpoint_key=key, step=step, val_top1=metrics["val/top1"], validation=metrics)
            store.put(f"{prefix}/best.json", json.dumps(best, indent=2, sort_keys=True).encode())

    model.train()
    while step < args.steps:
        for labels_batch, motion_batch in loader:
            if step >= args.steps:
                break
            motion_batch = to_device((torch.zeros(0), torch.zeros(0), motion_batch, None), device)[2]
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, base=args.lr, warmup=args.warmup, total=args.steps)
            loss, terms = model.loss(motion_batch, labels_batch.to(device))
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
                entry.update(metrics)
                improved = best is None or metrics["val/top1"] > best["val_top1"]
                stale = 0 if improved else stale + 1
            log.append(entry)
            if step % 100 == 0 or step == 1 or metrics:
                print(json.dumps(entry), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps or (metrics and stale == 0):
                checkpoint(metrics)
            if metrics and stale >= args.patience:
                print(f"early stop at step {step}: no new best validation top-1 in {stale} validations", flush=True)
                checkpoint(metrics)
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
