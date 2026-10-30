"""Round 17: train the plan tokenizer (``generation/plan_codes.PlanVQ``) over a frozen RVQ tokenizer (d2).

The plan tokenizer reads the frozen tokenizer's continuous per-part latents, pools windows of ``--window`` steps into
plan codes (``--codebook`` entries, ``--plan-stages`` residual stages per part) and decodes them back to per-step
latents; the loss is the relative latent error plus the frozen decoder's motion loss on the decoded motion
(``plan_tokenizer_loss``). docs/NEXT_PHASE_PLAN.md, "Round 17 design".

Same data, store, resume and validation conventions as ``train_motion_tokens.py``: the manifest is SHA-pinned; the
frozen tokenizer is named by ``--tokenizer-key`` with its SHA-256 (and the manifest it was trained on); checkpoints
``training-runs/<run>/checkpoints/step-XXXXXXXX.pt`` (``plan_codes.plan_checkpoint_bytes``: kind
``plan_vq_tokenizer``, config and config SHA, EMA weights, the frozen tokenizer's key, SHA-256 and weights); the
newest is resumed and must match the configuration, manifest and tokenizer; every ``--val-every`` steps the EMA
weights are scored on fixed validation batches and the best is recorded in ``best.json``.

**Fidelity diagnostic** (``--fidelity-clips N``, at every checkpoint): N validation clips (How2Sign first) are passed
through the frozen tokenizer alone and through the plan (the decoder given the latents decoded from the plan codes),
and both are compared with the real clips on the frames the clips observe (``motion_metrics.compare``): speed and
variation ratios of body, hands and head (all joints). The gate's measure is ``tokenizer_fidelity_check.py
--plan-tokenizer-key`` (reliable joints, 1000 clips) and ``tokenizer_recognisability_check.py --plan-tokenizer-key``
(recogniser top-5 ≥ 0.9 × real).
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
import warnings

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.text_motion_dataset import TextMotionDataset, collate_text_motion  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest  # noqa: E402
from signtranslator.generation.motion_rvq import MotionRVQ, MotionRVQConfig  # noqa: E402
from signtranslator.generation.plan_codes import (  # noqa: E402
    PLAN_KIND, PlanVQ, PlanVQConfig, plan_checkpoint_bytes, plan_tokenizer_loss,
)
from signtranslator.training.weight_ema import WeightEMA  # noqa: E402
from train_motion_tokens import corpus_sample_weights, layout_template  # noqa: E402
from train_text_to_motion import lr_at, make_store, to_device  # noqa: E402

FIDELITY_CHANNELS = ("body", "left_hand", "right_hand", "head_rotation")


def fidelity(plan: PlanVQ, tokenizer: MotionRVQ, states, fps: float) -> dict:
    """{"plan"|"tokenizer": {channel_speed_ratio, channel_variation_ratio}} of round trips of ``states`` through the
    plan and through the frozen tokenizer alone, against the real states on the frames they observe."""
    from signtranslator.generation.motion_metrics import compare
    from tokenizer_fidelity_check import observed_like
    from tokenizer_recognisability_check import round_trip
    import copy                                    # round_trip collates on the CPU: the round trips run on CPU copies
    plan, tokenizer = copy.deepcopy(plan).cpu().eval(), copy.deepcopy(tokenizer).cpu().eval()
    out = {}
    for name, through in (("plan", plan), ("tokenizer", None)):
        trips = [round_trip(tokenizer, tokenizer.config.motion, s, fps, plan=through) for s in states]
        channels = [c for c in FIDELITY_CHANNELS if c in tokenizer.config.motion.trained_channels
                    and all(c in s.channels for s in states)]
        with warnings.catch_warnings():        # compare's Fréchet term (not reported) warns below two clips
            warnings.simplefilter("ignore", RuntimeWarning)
            res = compare([observed_like(t, r) for t, r in zip(trips, states)], states, channels)
        out[name] = {f"{c}_{k}": round(float(res[c][k]), 4) for c in channels for k in ("speed_ratio", "variation_ratio")}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--tokenizer-key", required=True, help="the frozen RVQ tokenizer checkpoint (d2)")
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--tokenizer-manifest-sha256", help="the manifest the tokenizer was trained on, when it is not "
                                                            "--manifest-sha256")
    parser.add_argument("--conform-to-corpus", help="mixed manifests: conform every state to this corpus's layouts")
    parser.add_argument("--fps", type=float, help="resample states whose rate differs from this")
    parser.add_argument("--max-seconds", type=float, default=20.)
    parser.add_argument("--corpus-weight", action="append", default=[], help="CORPUS=W (repeatable)")
    parser.add_argument("--window", type=int, default=4, help="tokenizer steps per plan step")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--code-dim", type=int, default=64)
    parser.add_argument("--codebook", type=int, default=512)
    parser.add_argument("--plan-stages", type=int, default=2, help="residual stages per part")
    parser.add_argument("--blocks", type=int, default=2, help="residual blocks at each rate")
    parser.add_argument("--pooling", choices=("mean", "learned"), default="mean")
    parser.add_argument("--commitment", type=float, default=.25)
    parser.add_argument("--quantizer-dropout", type=float, default=0.)
    parser.add_argument("--latent-weight", type=float, default=1.)
    parser.add_argument("--motion-weight", type=float, default=1.)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--clip", type=float, default=1.)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument("--fidelity-clips", type=int, default=0,
                        help="at every checkpoint, speed/variation ratios of this many validation round trips (0: off)")
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
    payload = store.get(args.tokenizer_key)
    if hashlib.sha256(payload).hexdigest() != args.tokenizer_sha256:
        raise SystemExit("the tokenizer checkpoint differs from --tokenizer-sha256")
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if saved.get("kind") != "motion_rvq_tokenizer" or saved["manifest_sha256"] != (args.tokenizer_manifest_sha256
                                                                                   or args.manifest_sha256):
        raise SystemExit("--tokenizer-key is not an rvq tokenizer trained on this manifest (or on "
                         "--tokenizer-manifest-sha256)")
    tokenizer_state = saved.get("ema") or saved["model"]
    tokenizer = MotionRVQ(MotionRVQConfig.from_dict(saved["config"]))
    tokenizer.load_state_dict(tokenizer_state)
    tokenizer.to(device).eval().requires_grad_(False)
    config = PlanVQConfig(tokenizer.config, window=args.window, hidden=args.hidden, code_dim=args.code_dim,
                          codebook=args.codebook, stages=args.plan_stages, blocks=args.blocks, pooling=args.pooling,
                          commitment=args.commitment, quantizer_dropout=args.quantizer_dropout,
                          latent_weight=args.latent_weight, motion_weight=args.motion_weight)
    model = PlanVQ(config).to(device)
    cache = ContentCache(Path(os.environ.get("SIGN_CACHE_DIR", tempfile.gettempdir())) / "sign-translator-cache",
                         args.cache_bytes)
    template = layout_template(manifest, args.conform_to_corpus, cache, store.get) if args.conform_to_corpus else None
    train = TextMotionDataset(manifest, partition="train", cache=cache, fetch=store.get, max_seconds=args.max_seconds,
                              fps=args.fps, template=template)
    if not len(train):
        raise SystemExit("the manifest has no training records")
    val = TextMotionDataset(manifest, partition="val", cache=cache, fetch=store.get, max_seconds=args.max_seconds,
                            fps=args.fps, template=template)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    prefix = f"training-runs/{args.run}"
    step = 0
    resumed_ema = None
    checkpoints = [k for k in store.list(f"{prefix}/checkpoints/") if k.endswith(".pt")]
    if checkpoints:
        state = torch.load(io.BytesIO(store.get(checkpoints[-1])), map_location=device, weights_only=False)
        if (state.get("kind") != PLAN_KIND or state["config_sha256"] != config.sha256()
                or state["manifest_sha256"] != args.manifest_sha256 or state["tokenizer_sha256"] != args.tokenizer_sha256):
            raise SystemExit("the newest checkpoint was trained with a different config, manifest or tokenizer")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        resumed_ema, step = state.get("ema"), state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed {checkpoints[-1]} at step {step}", flush=True)
    else:
        store.put(f"{prefix}/run.json", json.dumps(dict(kind=PLAN_KIND, config=config.to_dict(),
                                                        config_sha256=config.sha256(), manifest_key=args.manifest_key,
                                                        manifest_sha256=args.manifest_sha256, args=vars(args),
                                                        train_records=len(train)), indent=2, sort_keys=True).encode())
    sampler = None
    if args.corpus_weight:
        from torch.utils.data import WeightedRandomSampler
        weights = {k: float(v) for k, v in (w.split("=", 1) for w in args.corpus_weight)}
        sampler = WeightedRandomSampler(corpus_sample_weights(train.records, weights), num_samples=len(train),
                                        replacement=True, generator=torch.Generator().manual_seed(args.seed + step))
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                        num_workers=args.workers, collate_fn=partial(collate_text_motion, loci=False),
                        generator=torch.Generator().manual_seed(args.seed + step),
                        persistent_workers=args.workers > 0, drop_last=True)
    ema = WeightEMA(model, args.ema_decay)
    if resumed_ema is not None:
        ema.shadow = {n: resumed_ema[n].to(device).clone() for n in ema.shadow}
        ema.updates = step
    val_batches = []
    if len(val):
        val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                collate_fn=partial(collate_text_motion, loci=False))
        for index, batch in enumerate(val_loader):
            if index >= args.val_batches:
                break
            val_batches.append(batch)
    fidelity_states = []
    if args.fidelity_clips:
        if args.fidelity_clips < 2:
            raise SystemExit("--fidelity-clips needs at least 2 clips (the ratios are medians over clips)")
        # validation clips first (How2Sign first), training clips only if the validation partition is too small
        pool = [(dataset, i) for dataset in (val, train) for i in sorted(
            range(len(dataset)), key=lambda i: (dataset.records[i].get("corpus") != "how2sign-two-view-text-v1", i))]
        fidelity_states = [dataset[i][1] for dataset, i in pool[:args.fidelity_clips]]
        print(json.dumps(dict(fidelity_clips=len(fidelity_states), from_validation=min(len(val), len(fidelity_states)))),
              flush=True)
    best = json.loads(store.get(f"{prefix}/best.json")) if store.exists(f"{prefix}/best.json") else None

    def validate() -> dict:
        """Mean validation terms of the EMA weights on the fixed batches (fixed noise seed)."""
        live = ema.swap(model)
        model.eval()
        totals, count = {}, 0
        with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(1234)
            for batch in val_batches:
                _, _, motion_v = to_device(batch, device)[:3]
                loss_v, terms_v = plan_tokenizer_loss(model, tokenizer, motion_v)
                for key, value in dict(loss=float(loss_v), **{k: float(v) for k, v in terms_v.items()}).items():
                    totals[key] = totals.get(key, 0.) + value
                count += 1
            metrics = {f"val/{k}": v / max(count, 1) for k, v in totals.items()}
            if fidelity_states:
                for name, ratios in fidelity(model, tokenizer, fidelity_states, args.fps or 24.).items():
                    metrics.update({f"fidelity/{name}/{k}": v for k, v in ratios.items()})
        ema.restore(model, live)
        model.train()
        return metrics

    from signtranslator.training.async_upload import Uploader
    uploader = Uploader(store)
    log, pending, started = [], [], time.monotonic()

    def flush():
        cells = [(entry, key) for entry in pending for key, value in entry.items() if isinstance(value, torch.Tensor)]
        if cells:
            values = torch.stack([entry[key].detach().to(torch.float64).reshape(()) for entry, key in cells]).tolist()
            for (entry, key), value in zip(cells, values):
                entry[key] = value
        for entry in pending:
            if not math.isfinite(entry["loss"]):
                raise FloatingPointError(f"non-finite loss at step {entry['step'] - 1}")
        pending.clear()

    def checkpoint(metrics: dict | None):
        nonlocal best
        flush()
        payload = plan_checkpoint_bytes(model, tokenizer_state, tokenizer_key=args.tokenizer_key,
                                        tokenizer_sha256=args.tokenizer_sha256, manifest_sha256=args.manifest_sha256,
                                        extra=dict(ema=ema.state_dict(model), optimizer=optimizer.state_dict(), step=step,
                                                   validation=metrics, torch_rng=torch.get_rng_state(),
                                                   tokenizer_manifest_sha256=args.tokenizer_manifest_sha256
                                                   or args.manifest_sha256))
        key = f"{prefix}/checkpoints/step-{step:08d}.pt"
        items = [(key, payload), (f"{prefix}/log/step-{step:08d}.json", json.dumps(log, sort_keys=True).encode())]
        if metrics and (best is None or metrics["val/loss"] < best["val_loss"]):
            best = dict(checkpoint_key=key, step=step, val_loss=metrics["val/loss"], validation=metrics)
            items.append((f"{prefix}/best.json", json.dumps(best, indent=2, sort_keys=True).encode()))
        uploader.submit(items)

    model.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            _, _, motion = to_device(batch, device)[:3]
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, base=args.lr, warmup=args.warmup, total=args.steps)
            try:
                loss, terms = plan_tokenizer_loss(model, tokenizer, motion)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                ema.update(model)
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                flush()
                raise
            step += 1
            entry = dict(step=step, loss=loss.detach(), grad_norm=norm.detach(),
                         seconds=round(time.monotonic() - started, 1), **{f"train/{k}": v for k, v in terms.items()})
            log.append(entry)
            pending.append(entry)
            due_validation = val_batches and (step % args.val_every == 0 or step == args.steps)
            if due_validation or step % 50 == 0 or step == 1 or step % args.checkpoint_every == 0 or step == args.steps:
                flush()
            metrics = None
            if due_validation:
                metrics = validate()
                entry.update(metrics)
                print(json.dumps({k: v for k, v in entry.items() if k == "step" or "/" in k and not k.startswith("train/")}),
                      flush=True)
            if step % 50 == 0 or step == 1:
                print(json.dumps(entry), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps:
                checkpoint(metrics if metrics is not None else (validate() if val_batches else None))
    flush()
    uploader.drain()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
