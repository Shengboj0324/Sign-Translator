"""Train the motion-token generator: stage 1 tokenizer (VQ-VAE), stage 2 text prior.

``--architecture vq`` is tokens v1 (one codebook, causal token prior); ``rvq`` is
tokens v2 (part-wise residual VQ with the skeleton motion loss, depth-autoregressive
prior; ``generation/motion_rvq.py``).

**Batch noise.** A batch's loss varies with which clips it drew, so a single step's
loss is not a learning curve. Every ``--val-every`` steps the model is scored on the
same ``--val-batches`` batches of the validation partition (fixed order, fixed noise
seed, no text dropout), with an exponential moving average of the weights
(``--ema-decay``; codebooks, already EMA-updated, are taken from the live model).
The EMA weights are what checkpoints export for generation (``ema`` key), and the
checkpoint with the lowest validation loss is recorded in ``best.json``.

Same data, store and resume conventions as ``train_text_to_motion.py``:
checkpoints under ``training-runs/<run>/checkpoints/step-XXXXXXXX.pt``; the newest is
resumed and must match the configuration and manifest. Stage 2 freezes the stage 1
tokenizer named by ``--tokenizer-key`` (hash-pinned) and stores its weights inside
every prior checkpoint, so one prior checkpoint fully defines the generator.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.text_motion_dataset import TextMotionDataset, collate_text_motion  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest  # noqa: E402
from signtranslator.generation.canonical_state import channel_bindings_from_state  # noqa: E402
from signtranslator.generation.motion_rvq import (  # noqa: E402
    MotionRVQ, MotionRVQConfig, RVQPrior, RVQPriorConfig, rvq_loss, rvq_prior_loss,
)
from signtranslator.generation.motion_tokens import (  # noqa: E402
    MotionVQ, MotionVQConfig, TokenPrior, TokenPriorConfig, prior_loss, tokenizer_loss,
)
from signtranslator.generation.text_to_motion import TextToMotionConfig  # noqa: E402
from train_text_to_motion import lr_at, make_store, to_device  # noqa: E402
from signtranslator.training.weight_ema import WeightEMA  # noqa: E402


def kinds(architecture: str) -> tuple[str, str]:
    return ("motion_tokenizer", "motion_token_prior") if architecture == "vq" else \
        ("motion_rvq_tokenizer", "motion_rvq_prior")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tokenizer", "prior"), required=True)
    parser.add_argument("--architecture", choices=("vq", "rvq"), default="vq", help="tokens v1 (vq) or v2 (rvq)")
    parser.add_argument("--stages", type=int, default=4, help="rvq: residual stages per part")
    parser.add_argument("--base-blocks", type=int, default=0, help="rvq: residual blocks at full frame rate")
    parser.add_argument("--depth-layers", type=int, default=2, help="rvq prior: depth transformer layers")
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--patience", type=int, default=0,
                        help="stop after this many validations without a new best validation loss (0: never)")
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--tokenizer-key", help="prior stage: stage 1 checkpoint")
    parser.add_argument("--tokenizer-sha256", help="prior stage: its SHA-256")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--clip", type=float, default=1.)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--code-dim", type=int, default=128)
    parser.add_argument("--codebook", type=int, default=512)
    parser.add_argument("--down-layers", type=int, default=2)
    parser.add_argument("--locus-conditioning", action="store_true",
                        help="prior: condition on referent-locus slots measured in each clip's pointing")
    parser.add_argument("--no-grammar-features", dest="grammar_features", action="store_false",
                        help="do not condition the prior on the English's grammatical features")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--text-layers", type=int, default=4)
    parser.add_argument("--token-layers", type=int, default=6)
    parser.add_argument("--feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--cfg-dropout", type=float, default=.1)
    parser.add_argument("--max-seconds", type=float, default=20.)
    parser.add_argument("--trained-channels", default="body,left_hand,right_hand,head_rotation,face,blink,root_translation,head_translation,left_gaze,right_gaze")
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-bytes", type=int, default=8 * 2 ** 30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.device == "mps":
        raise SystemExit("MPS is refused: it miscomputes rotation operations at batch scale")
    store = make_store(args.store)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "manifest.json"
        path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    cache = ContentCache(Path(os.environ.get("SIGN_CACHE_DIR", tempfile.gettempdir())) / "sign-translator-cache",
                         args.cache_bytes)
    train = TextMotionDataset(manifest, partition="train", cache=cache, fetch=store.get, max_seconds=args.max_seconds)
    if not len(train):
        raise SystemExit("the manifest has no training records")
    val = TextMotionDataset(manifest, partition="val", cache=cache, fetch=store.get, max_seconds=args.max_seconds)
    tokenizer_kind, prior_kind = kinds(args.architecture)
    if args.stage == "tokenizer":
        motion = TextToMotionConfig(bindings=channel_bindings_from_state(train[0][1]),
                                    trained_channels=tuple(args.trained_channels.split(",")), max_seconds=args.max_seconds)
        if args.architecture == "vq":
            config = MotionVQConfig(motion, hidden=args.hidden, code_dim=args.code_dim, codebook=args.codebook,
                                    down_layers=args.down_layers)
            model = MotionVQ(config).to(device)
        else:
            config = MotionRVQConfig(motion, hidden=args.hidden, code_dim=args.code_dim, codebook=args.codebook,
                                     stages=args.stages, down_layers=args.down_layers, base_blocks=args.base_blocks)
            model = MotionRVQ(config).to(device)
        tokenizer, tokenizer_state = None, None
    else:
        if not args.tokenizer_key or not args.tokenizer_sha256:
            raise SystemExit("the prior stage needs --tokenizer-key and --tokenizer-sha256")
        payload = store.get(args.tokenizer_key)
        if hashlib.sha256(payload).hexdigest() != args.tokenizer_sha256:
            raise SystemExit("the tokenizer checkpoint differs from --tokenizer-sha256")
        saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        if saved.get("kind") != tokenizer_kind or saved["manifest_sha256"] != args.manifest_sha256:
            raise SystemExit("--tokenizer-key is not a tokenizer trained on this manifest")
        tokenizer_state = saved.get("ema") or saved["model"]       # the EMA weights, when the tokenizer has them
        if args.architecture == "vq":
            tokenizer = MotionVQ(MotionVQConfig.from_dict(saved["config"]))
        else:
            tokenizer = MotionRVQ(MotionRVQConfig.from_dict(saved["config"]))
        tokenizer.load_state_dict(tokenizer_state)
        tokenizer.to(device).eval().requires_grad_(False)
        common = dict(d_model=args.d_model, heads=args.heads, text_layers=args.text_layers, feedforward=args.feedforward,
                      dropout=args.dropout, cfg_dropout=args.cfg_dropout, grammar_features=args.grammar_features,
                      locus_conditioning=args.locus_conditioning)
        if args.architecture == "vq":
            config = TokenPriorConfig(tokenizer.config, token_layers=args.token_layers, **common)
            model = TokenPrior(config).to(device)
        else:
            config = RVQPriorConfig(tokenizer.config, temporal_layers=args.token_layers, depth_layers=args.depth_layers,
                                    **common)
            model = RVQPrior(config).to(device)
    kind = tokenizer_kind if args.stage == "tokenizer" else prior_kind

    def loss_of(motion, tokens, mask, loci, train: bool):
        if args.stage == "tokenizer":
            return tokenizer_loss(model, motion) if args.architecture == "vq" else rvq_loss(model, motion)
        if args.architecture == "vq":
            return prior_loss(model, tokenizer, tokens, mask, motion, *loci)
        return rvq_prior_loss(model, tokenizer, tokens, mask, motion, *loci, train=train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    prefix = f"training-runs/{args.run}"
    step = 0
    checkpoints = [k for k in store.list(f"{prefix}/checkpoints/") if k.endswith(".pt")]
    if checkpoints:
        state = torch.load(io.BytesIO(store.get(checkpoints[-1])), map_location=device, weights_only=False)
        if (state.get("kind") != kind or state["config_sha256"] != config.sha256()
                or state["manifest_sha256"] != args.manifest_sha256):
            raise SystemExit("the newest checkpoint was trained with a different config or manifest")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        resumed_ema = state.get("ema")
        step = state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed {checkpoints[-1]} at step {step}", flush=True)
    else:
        store.put(f"{prefix}/run.json", json.dumps(dict(kind=kind, config=config.to_dict(), config_sha256=config.sha256(),
                                                        manifest_key=args.manifest_key,
                                                        manifest_sha256=args.manifest_sha256, args=vars(args),
                                                        train_records=len(train)), indent=2, sort_keys=True).encode())
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=partial(collate_text_motion, loci=args.locus_conditioning and args.stage == "prior"),
                        generator=torch.Generator().manual_seed(args.seed + step),
                        persistent_workers=args.workers > 0, drop_last=True)
    log, started = [], time.monotonic()
    ema = WeightEMA(model, args.ema_decay)
    if checkpoints and resumed_ema is not None:
        ema.shadow = {n: resumed_ema[n].to(device).clone() for n in ema.shadow}
        ema.updates = step
    val_batches = []
    if len(val):
        val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                collate_fn=partial(collate_text_motion, loci=args.locus_conditioning and args.stage == "prior"))
        for index, batch in enumerate(val_loader):
            if index >= args.val_batches:
                break
            val_batches.append(batch)
    best = json.loads(store.get(f"{prefix}/best.json")) if store.exists(f"{prefix}/best.json") else None

    def validate() -> dict:
        """Mean validation terms of the EMA weights on the fixed batches (fixed noise seed)."""
        live = ema.swap(model)
        model.eval()
        totals, count = {}, 0
        with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(1234)
            for batch in val_batches:
                tokens_v, mask_v, motion_v, *loci_v = to_device(batch, device)
                loss_v, terms_v = loss_of(motion_v, tokens_v, mask_v, loci_v, train=False)
                for key, value in dict(loss=float(loss_v), **terms_v).items():
                    totals[key] = totals.get(key, 0.) + float(value)
                count += 1
        ema.restore(model, live)
        model.train()
        return {f"val/{k}": v / max(count, 1) for k, v in totals.items()}

    pending = []                               # log entries whose values are still device tensors

    def flush():
        """Read the pending entries' values off the device (one wait for the GPU), in place; refuse a non-finite loss.

        Reading a value back synchronises with the GPU, so the training step leaves its
        metrics on the device and they are read here only when they are printed,
        validated or checkpointed. A non-finite loss is therefore reported up to 50 steps
        late (or when the poisoned weights make a later step fail), always before a save.
        """
        cells = [(entry, key) for entry in pending for key, value in entry.items() if isinstance(value, torch.Tensor)]
        if cells:
            values = torch.stack([entry[key].detach().to(torch.float64).reshape(()) for entry, key in cells]).tolist()
            for (entry, key), value in zip(cells, values):
                entry[key] = value
        for entry in pending:
            if not math.isfinite(entry["loss"]):
                raise FloatingPointError(f"non-finite loss at step {entry['step'] - 1}")
        pending.clear()

    def checkpoint(metrics: dict | None = None):
        nonlocal best
        flush()
        buffer = io.BytesIO()
        torch.save(dict(kind=kind, model=model.state_dict(), ema=ema.state_dict(model), optimizer=optimizer.state_dict(),
                        step=step, validation=metrics,
                        torch_rng=torch.get_rng_state(), config=config.to_dict(), config_sha256=config.sha256(),
                        manifest_sha256=args.manifest_sha256, tokenizer_state=tokenizer_state,
                        tokenizer_key=args.tokenizer_key, tokenizer_sha256=args.tokenizer_sha256), buffer)
        key = f"{prefix}/checkpoints/step-{step:08d}.pt"
        store.put(key, buffer.getvalue())
        store.put(f"{prefix}/log/step-{step:08d}.json", json.dumps(log, sort_keys=True).encode())
        if metrics and (best is None or metrics["val/loss"] < best["val_loss"]):
            best = dict(checkpoint_key=key, step=step, val_loss=metrics["val/loss"], validation=metrics)
            store.put(f"{prefix}/best.json", json.dumps(best, indent=2, sort_keys=True).encode())

    model.train()
    stale = 0                                  # validations since the last best

    def note(metrics):
        """Early stopping: count validations that do not improve on the best (``best`` set by checkpoint())."""
        nonlocal stale
        best_loss = best["val_loss"] if best else float("inf")
        stale = 0 if metrics["val/loss"] < best_loss else stale + 1
        return args.patience and stale >= args.patience

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            tokens, mask, motion, *loci = to_device(batch, device)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, base=args.lr, warmup=args.warmup, total=args.steps)
            try:
                loss, terms = loss_of(motion, tokens, mask, loci, train=True)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                ema.update(model)
            except Exception:
                flush()                        # an unread non-finite loss before this failure is its cause
                raise
            step += 1
            entry = dict(step=step, loss=loss.detach(), grad_norm=norm.detach(),
                         seconds=round(time.monotonic() - started, 1), **{f"train/{k}": v for k, v in terms.items()})
            log.append(entry)
            pending.append(entry)
            due_validation = val_batches and (step % args.val_every == 0 or step == args.steps)
            if due_validation or step % 50 == 0 or step == 1 or step % args.checkpoint_every == 0 or step == args.steps:
                flush()
            metrics, stop = None, False
            if due_validation:
                metrics = validate()
                entry.update(metrics)
                stop = note(metrics)
                print(json.dumps({k: v for k, v in entry.items() if k == "step" or k.startswith("val/")}), flush=True)
                if metrics is not None and (best is None or metrics["val/loss"] < best["val_loss"]) \
                        and step % args.checkpoint_every != 0:
                    checkpoint(metrics)                  # a new best between regular checkpoints is kept too
            if step % 50 == 0 or step == 1:
                print(json.dumps(entry), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps or stop:
                checkpoint(metrics if metrics is not None else (validate() if val_batches else None))
            if stop:
                print(json.dumps(dict(step=step, early_stop=True, best_step=best["step"], best_val_loss=best["val_loss"])),
                      flush=True)
                return 0
    flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
