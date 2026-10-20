"""Train the English-to-canonical-motion model from a frozen manifest (Runpod-ready).

Data, checkpoints and logs live in an object store: R2 in production (credentials
from environment variables, never logged) or a local directory for tests. The run
is identified by ``--run``; checkpoints are written under
``training-runs/<run>/checkpoints/step-XXXXXXXX.pt`` and the newest one is resumed
automatically, including optimizer, scheduler, RNG and step. Evaluation on the
validation partition reports per-channel loss and geodesic error; the test
partition is never touched here.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signtranslator.data.text_motion_dataset import (  # noqa: E402
    DirectoryStore, R2Store, TextMotionDataset, collate_text_motion,
)
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest  # noqa: E402
from signtranslator.generation.canonical_state import channel_bindings_from_state  # noqa: E402
from signtranslator.generation.text_to_motion import TextToMotion, TextToMotionConfig, training_loss  # noqa: E402


def make_store(spec: str):
    if spec == "r2":
        return R2Store(os.environ)
    if spec.startswith("dir:"):
        return DirectoryStore(Path(spec[4:]))
    raise ValueError("store must be 'r2' or 'dir:/path'")


def lr_at(step: int, *, base: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1., (step - warmup) / max(1, total - warmup))
    return base * .5 * (1 + math.cos(math.pi * progress))


def to_device(batch, device):
    tokens, mask, motion, texts = batch
    if device.type == "cpu":
        return tokens, mask, motion
    from dataclasses import replace
    channels = {name: replace(ch, values=ch.values.to(device), valid=ch.valid.to(device),
                              observed=ch.observed.to(device), confidence=ch.confidence.to(device))
                for name, ch in motion.channels.items()}
    motion = replace(motion, timestamps=motion.timestamps.to(device), frame_valid=motion.frame_valid.to(device),
                     lengths=motion.lengths.to(device), channels=channels)
    return tokens.to(device), mask.to(device), motion


@torch.no_grad()
def evaluate(model, loader, device, limit):
    model.eval()
    totals, counts = {}, {}
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        tokens, mask, motion = to_device(batch, device)
        _, terms, diagnostics = training_loss(model, tokens, mask, motion)
        for key, value in {**terms, **{k: v for k, v in diagnostics.items() if k.endswith("geodesic_degrees_mean")}}.items():
            totals[key] = totals.get(key, 0.) + value
            counts[key] = counts.get(key, 0) + 1
    model.train()
    return {key: totals[key] / counts[key] for key in totals}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--clip", type=float, default=1.)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--text-layers", type=int, default=6)
    parser.add_argument("--motion-layers", type=int, default=8)
    parser.add_argument("--feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--max-seconds", type=float, default=20.)
    parser.add_argument("--objective", choices=("regression", "diffusion"), default="regression",
                        help="diffusion samples a plausible motion instead of predicting the conditional mean")
    parser.add_argument("--cfg-dropout", type=float, default=.1, help="diffusion: empty-text probability")
    parser.add_argument("--trained-channels", default="body,left_hand,right_hand,head_rotation,face,blink,root_translation,head_translation")
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-bytes", type=int, default=8 * 2 ** 30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    args = parser.parse_args(argv)

    if args.device == "mps":
        # Measured 2026-10-19 (torch on Apple GPU): rotation checks on real batches fail on
        # MPS for every valid left-hand joint and produce non-finite values for the right
        # hand, while CPU passes them all in float32 and float64. Train on CUDA or CPU.
        raise SystemExit("MPS is refused: it miscomputes rotation operations at batch scale")
    store = make_store(args.store)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "manifest.json"
        path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    cache_dir = Path(os.environ.get("SIGN_CACHE_DIR", tempfile.gettempdir())) / "sign-translator-cache"
    cache = ContentCache(cache_dir, args.cache_bytes)
    train = TextMotionDataset(manifest, partition="train", cache=cache, fetch=store.get, max_seconds=args.max_seconds)
    val = TextMotionDataset(manifest, partition="val", cache=cache, fetch=store.get, max_seconds=args.max_seconds)
    if not len(train):
        raise SystemExit("the manifest has no training records")
    first_text, first_state = train[0]
    config = TextToMotionConfig(
        bindings=channel_bindings_from_state(first_state), trained_channels=tuple(args.trained_channels.split(",")),
        d_model=args.d_model, heads=args.heads, text_layers=args.text_layers, motion_layers=args.motion_layers,
        feedforward=args.feedforward, dropout=args.dropout, max_seconds=args.max_seconds,
        objective=args.objective, cfg_dropout=args.cfg_dropout)
    model = TextToMotion(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    prefix = f"training-runs/{args.run}"
    step = 0
    checkpoints = [k for k in store.list(f"{prefix}/checkpoints/") if k.endswith(".pt")]
    if checkpoints:
        state = torch.load(io.BytesIO(store.get(checkpoints[-1])), map_location=device, weights_only=False)
        if state["config_sha256"] != config.sha256() or state["manifest_sha256"] != args.manifest_sha256:
            raise SystemExit("the newest checkpoint was trained with a different config or manifest")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed {checkpoints[-1]} at step {step}", flush=True)
    else:
        store.put(f"{prefix}/run.json", json.dumps(dict(
            config=config.to_dict(), config_sha256=config.sha256(), manifest_key=args.manifest_key,
            manifest_sha256=args.manifest_sha256, args=vars(args), train_records=len(train), val_records=len(val),
            parameters=sum(p.numel() for p in model.parameters())), indent=2, sort_keys=True).encode())
    generator = torch.Generator().manual_seed(args.seed + step)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate_text_motion, generator=generator, persistent_workers=args.workers > 0,
                        drop_last=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=min(args.workers, 2),
                            collate_fn=collate_text_motion) if len(val) else None
    log, started = [], time.monotonic()
    model.train()

    def checkpoint():
        buffer = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                        torch_rng=torch.get_rng_state(), config=config.to_dict(), config_sha256=config.sha256(),
                        manifest_sha256=args.manifest_sha256), buffer)
        store.put(f"{prefix}/checkpoints/step-{step:08d}.pt", buffer.getvalue())
        store.put(f"{prefix}/log/step-{step:08d}.json", json.dumps(log, sort_keys=True).encode())

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            tokens, mask, motion = to_device(batch, device)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, base=args.lr, warmup=args.warmup, total=args.steps)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp == "bf16" and device.type == "cuda"):
                loss, terms, _ = training_loss(model, tokens, mask, motion)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            step += 1
            entry = dict(step=step, loss=float(loss.detach()), grad_norm=float(norm), lr=optimizer.param_groups[0]["lr"],
                         seconds=round(time.monotonic() - started, 1), **{f"train/{k}": v for k, v in terms.items()})
            if val_loader is not None and step % args.eval_every == 0:
                entry.update({f"val/{k}": v for k, v in evaluate(model, val_loader, device, args.eval_batches).items()})
            log.append(entry)
            if step % 50 == 0 or step == 1:
                print(json.dumps(entry), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps:
                checkpoint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
