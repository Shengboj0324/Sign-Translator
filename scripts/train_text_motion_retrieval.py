"""Train the text–motion retrieval evaluator on real pairs (frozen afterwards; scores generators by R-precision).

Same store, manifest and checkpoint layout as the generators
(``training-runs/<run>/checkpoints``, ``log``, ``best.json``). Validation embeds the
whole validation partition with the EMA weights and reports R@1/2/3 in fixed
groups of 32 (``text_motion_retrieval.r_precision``, seed 0) plus the InfoNCE loss.
The checkpoint with the highest validation R@3 is recorded in ``best.json``, and
``--patience`` stops after that many validations without a new best. The test
partition is never read here.
"""
from __future__ import annotations

import argparse
from functools import partial
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.text_motion_dataset import TextMotionDataset, collate_text_motion  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest  # noqa: E402
from signtranslator.generation.canonical_state import channel_bindings_from_state  # noqa: E402
from signtranslator.generation.text_motion_retrieval import (  # noqa: E402
    TextMotionRetrieval, TextMotionRetrievalConfig, r_precision, retrieval_loss)
from signtranslator.generation.text_to_motion import TextToMotionConfig  # noqa: E402
from signtranslator.training.weight_ema import WeightEMA  # noqa: E402
from train_text_to_motion import lr_at, make_store, to_device  # noqa: E402

KIND = "text_motion_retrieval"


@torch.no_grad()
def embed_all(model, batches, device):
    """(motion embeddings, text embeddings, mean loss) over ``batches`` in eval mode."""
    model.eval()
    zm, zt, losses = [], [], []
    for batch in batches:
        tokens, mask, motion = to_device(batch, device)[:3]
        zm.append(model.encode_motion(motion).cpu())
        zt.append(model.encode_text(tokens, mask).cpu())
        losses.append(float(retrieval_loss(model, tokens, mask, motion)[0]))
    model.train()
    return torch.cat(zm), torch.cat(zt), float(np.mean(losses))


_AUGMENT_RNG: dict = {}


def worker_augment(state):
    """Speed 0.85-1.15x and face dropout 0.2, no trimming (a sentence's first and last signs carry meaning); one
    generator per loader process, seeded from torch."""
    from signtranslator.generation.sign_recognizer import augment_sign
    pid = os.getpid()
    if pid not in _AUGMENT_RNG:
        _AUGMENT_RNG[pid] = random.Random(torch.initial_seed())
    return augment_sign(state, _AUGMENT_RNG[pid], speed=(.85, 1.15), trim=0., face_dropout=.2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--motion-layers", type=int, default=4)
    parser.add_argument("--text-layers", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--embedding", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=20.)
    parser.add_argument("--trained-channels", default="body,left_hand,right_hand,head_rotation,face,blink,root_translation,head_translation,left_gaze,right_gaze")
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-bytes", type=int, default=8 * 2 ** 30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-embedding-key", help="frozen sentence-embedding table (embed_texts.py) for the text side")
    parser.add_argument("--text-embedding-sha256")
    parser.add_argument("--fps", type=float, help="resample states whose rate differs from this (mixed-rate corpora)")
    parser.add_argument("--augment", action="store_true",
                        help="training clips at a random speed (0.85-1.15x) and, one time in five, without the face "
                             "(needs --fps: the resampler turns the rescaled clock into frames)")
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
    if args.augment and not args.fps:
        raise SystemExit("--augment needs --fps")
    train = TextMotionDataset(manifest, partition="train", cache=cache, fetch=store.get, max_seconds=args.max_seconds,
                              fps=args.fps, augment=worker_augment if args.augment else None)
    val = TextMotionDataset(manifest, partition="val", cache=cache, fetch=store.get, max_seconds=args.max_seconds,
                            fps=args.fps)
    if not len(train) or not len(val):
        raise SystemExit("the manifest needs training and validation records")
    motion = TextToMotionConfig(bindings=channel_bindings_from_state(train[0][1]),
                                trained_channels=tuple(args.trained_channels.split(",")), max_seconds=args.max_seconds)
    table = None
    if args.text_embedding_key:
        from signtranslator.generation.text_motion_retrieval import load_text_embeddings
        table = load_text_embeddings(store, args.text_embedding_key, args.text_embedding_sha256)
    config = TextMotionRetrievalConfig(motion, d_model=args.d_model, heads=args.heads, motion_layers=args.motion_layers,
                                       text_layers=args.text_layers, feedforward=args.feedforward, dropout=args.dropout,
                                       embedding=args.embedding, text_embedding_key=args.text_embedding_key,
                                       text_embedding_sha256=args.text_embedding_sha256,
                                       **(dict(text_embedding_dim=int(table[1].shape[1])) if table else {}))
    model = TextMotionRetrieval(config).to(device)
    if table:
        model.attach_text_embeddings(*table)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    prefix = f"training-runs/{args.run}"
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
        store.put(f"{prefix}/run.json", json.dumps(dict(kind=KIND, config=config.to_dict(), config_sha256=config.sha256(),
                                                        manifest_key=args.manifest_key, manifest_sha256=args.manifest_sha256,
                                                        args=vars(args), train_records=len(train), val_records=len(val)),
                                                   indent=2, sort_keys=True).encode())
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate_text_motion, generator=torch.Generator().manual_seed(args.seed + step),
                        persistent_workers=args.workers > 0, drop_last=True)
    val_batches = list(DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                  collate_fn=collate_text_motion))
    ema = WeightEMA(model, args.ema_decay)
    if resumed_ema is not None:
        ema.shadow = {n: resumed_ema[n].to(device).clone() for n in ema.shadow}
        ema.updates = step
    best = json.loads(store.get(f"{prefix}/best.json")) if store.exists(f"{prefix}/best.json") else None
    stale, log, started = 0, [], time.monotonic()

    def validate() -> dict:
        live = ema.swap(model)
        zm, zt, loss = embed_all(model, val_batches, device)
        ema.restore(model, live)
        return {"val/loss": loss, **{f"val/{k}": v for k, v in r_precision(zm, zt, group=min(32, len(zm))).items()}}

    def checkpoint(metrics=None):
        nonlocal best
        buffer = io.BytesIO()
        torch.save(dict(kind=KIND, model=model.state_dict(), ema=ema.state_dict(model), optimizer=optimizer.state_dict(),
                        step=step, validation=metrics, torch_rng=torch.get_rng_state(), config=config.to_dict(),
                        config_sha256=config.sha256(), manifest_sha256=args.manifest_sha256), buffer)
        key = f"{prefix}/checkpoints/step-{step:08d}.pt"
        store.put(key, buffer.getvalue())
        store.put(f"{prefix}/log/step-{step:08d}.json", json.dumps(log, sort_keys=True).encode())
        if metrics and (best is None or metrics["val/r_at_3"] > best["val_r_at_3"]):
            best = dict(checkpoint_key=key, step=step, val_r_at_3=metrics["val/r_at_3"], validation=metrics)
            store.put(f"{prefix}/best.json", json.dumps(best, indent=2, sort_keys=True).encode())

    model.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            tokens, mask, motion_batch = to_device(batch, device)[:3]
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, base=args.lr, warmup=args.warmup, total=args.steps)
            loss, terms = retrieval_loss(model, tokens, mask, motion_batch)
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
                improved = best is None or metrics["val/r_at_3"] > best["val_r_at_3"]
                stale = 0 if improved else stale + 1
            log.append(entry)
            if step % 50 == 0 or step == 1 or metrics:
                print(json.dumps(entry), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps or (metrics and stale == 0):
                checkpoint(metrics)
            if metrics and args.patience and stale >= args.patience:
                checkpoint(metrics)
                print(json.dumps(dict(step=step, early_stop=True, best_step=best["step"], best_val_r_at_3=best["val_r_at_3"])),
                      flush=True)
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
