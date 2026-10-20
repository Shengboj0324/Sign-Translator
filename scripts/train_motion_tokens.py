"""Train the motion-token generator: stage 1 tokenizer (VQ-VAE), stage 2 text prior.

Same data, store and resume conventions as ``train_text_to_motion.py``:
checkpoints under ``training-runs/<run>/checkpoints/step-XXXXXXXX.pt``; the newest is
resumed and must match the configuration and manifest. Stage 2 freezes the stage 1
tokenizer named by ``--tokenizer-key`` (hash-pinned) and stores its weights inside
every prior checkpoint, so one prior checkpoint fully defines the generator.
"""
from __future__ import annotations

import argparse
import hashlib
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.text_motion_dataset import TextMotionDataset, collate_text_motion  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest  # noqa: E402
from signtranslator.generation.canonical_state import channel_bindings_from_state  # noqa: E402
from signtranslator.generation.motion_tokens import (  # noqa: E402
    MotionVQ, MotionVQConfig, TokenPrior, TokenPriorConfig, prior_loss, tokenizer_loss,
)
from signtranslator.generation.text_to_motion import TextToMotionConfig  # noqa: E402
from train_text_to_motion import lr_at, make_store, to_device  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tokenizer", "prior"), required=True)
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
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--text-layers", type=int, default=4)
    parser.add_argument("--token-layers", type=int, default=6)
    parser.add_argument("--feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--cfg-dropout", type=float, default=.1)
    parser.add_argument("--max-seconds", type=float, default=20.)
    parser.add_argument("--trained-channels", default="body,left_hand,right_hand,head_rotation,face,blink,root_translation,head_translation")
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
    if args.stage == "tokenizer":
        motion = TextToMotionConfig(bindings=channel_bindings_from_state(train[0][1]),
                                    trained_channels=tuple(args.trained_channels.split(",")), max_seconds=args.max_seconds)
        config = MotionVQConfig(motion, hidden=args.hidden, code_dim=args.code_dim, codebook=args.codebook,
                                down_layers=args.down_layers)
        model, tokenizer, tokenizer_state = MotionVQ(config).to(device), None, None
    else:
        if not args.tokenizer_key or not args.tokenizer_sha256:
            raise SystemExit("the prior stage needs --tokenizer-key and --tokenizer-sha256")
        payload = store.get(args.tokenizer_key)
        if hashlib.sha256(payload).hexdigest() != args.tokenizer_sha256:
            raise SystemExit("the tokenizer checkpoint differs from --tokenizer-sha256")
        saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        if saved.get("kind") != "motion_tokenizer" or saved["manifest_sha256"] != args.manifest_sha256:
            raise SystemExit("--tokenizer-key is not a tokenizer trained on this manifest")
        tokenizer = MotionVQ(MotionVQConfig.from_dict(saved["config"]))
        tokenizer.load_state_dict(saved["model"])
        tokenizer.to(device).eval().requires_grad_(False)
        tokenizer_state = saved["model"]
        config = TokenPriorConfig(tokenizer.config, d_model=args.d_model, heads=args.heads, text_layers=args.text_layers,
                                  token_layers=args.token_layers, feedforward=args.feedforward, dropout=args.dropout,
                                  cfg_dropout=args.cfg_dropout)
        model = TokenPrior(config).to(device)
    kind = "motion_tokenizer" if args.stage == "tokenizer" else "motion_token_prior"
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
        step = state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed {checkpoints[-1]} at step {step}", flush=True)
    else:
        store.put(f"{prefix}/run.json", json.dumps(dict(kind=kind, config=config.to_dict(), config_sha256=config.sha256(),
                                                        manifest_key=args.manifest_key,
                                                        manifest_sha256=args.manifest_sha256, args=vars(args),
                                                        train_records=len(train)), indent=2, sort_keys=True).encode())
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate_text_motion, generator=torch.Generator().manual_seed(args.seed + step),
                        persistent_workers=args.workers > 0, drop_last=True)
    log, started = [], time.monotonic()

    def checkpoint():
        buffer = io.BytesIO()
        torch.save(dict(kind=kind, model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                        torch_rng=torch.get_rng_state(), config=config.to_dict(), config_sha256=config.sha256(),
                        manifest_sha256=args.manifest_sha256, tokenizer_state=tokenizer_state,
                        tokenizer_key=args.tokenizer_key, tokenizer_sha256=args.tokenizer_sha256), buffer)
        store.put(f"{prefix}/checkpoints/step-{step:08d}.pt", buffer.getvalue())
        store.put(f"{prefix}/log/step-{step:08d}.json", json.dumps(log, sort_keys=True).encode())

    model.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            tokens, mask, motion = to_device(batch, device)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, base=args.lr, warmup=args.warmup, total=args.steps)
            if args.stage == "tokenizer":
                loss, terms = tokenizer_loss(model, motion)
            else:
                loss, terms = prior_loss(model, tokenizer, tokens, mask, motion)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            step += 1
            entry = dict(step=step, loss=float(loss.detach()), grad_norm=float(norm),
                         seconds=round(time.monotonic() - started, 1), **{f"train/{k}": v for k, v in terms.items()})
            log.append(entry)
            if step % 50 == 0 or step == 1:
                print(json.dumps(entry), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps:
                checkpoint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
