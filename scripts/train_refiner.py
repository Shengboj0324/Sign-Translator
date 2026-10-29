"""Train the sentence refiner (``signtranslator.generation.refiner``) on the pairs of ``build_refiner_pairs.py``.
Built and tested only; it trains once declared.

**Warm start** (``--init-key``, SHA-256 pinned; F12 step 60000 by default): the prior's EMA weights for every kept
module, the source encoder from its text encoder (``SentenceRefiner.warm_start``). The pairs' tokenizer must be the
prior's (``tokenizer_digest`` of the pairs' configuration against the prior's ``tokenizer_state``), so the codes the
refiner reads and writes are the ones the prior learnt.

**Data:** every pair of ``--pairs-prefix``'s ``index.json`` (status converted or already present), each file fetched
and checked against the SHA-256 the index records, held in memory. ``split: train`` pairs train: the batch of step s
is fixed by s alone (``batch_items``: a seeded permutation per epoch, and per sentence one of its K composed inputs,
seeded by epoch and sentence), so a resumed run sees exactly the batches an uninterrupted one would.

**Checkpoint selection** on the held-out train signers only (``split: selection``, every one of their K inputs, fixed
order, ``--val-every``): teacher-forced loss of the EMA weights, without source or history dropout, the lowest written
to ``best.json``. No How2Sign val clip, BLEU or other evaluator is read.

**Length** is declared in epochs (``--epochs``, default 8, at most ``--max-epochs`` 10: each epoch shows every
training sentence once, with one of its K inputs): steps = ceil(epochs x training pairs / batch). ``--steps`` overrides
it for smoke runs only and must stay within ``--max-epochs``; run.json records both.

**Memorisation index:** the stage-1 code n-grams of every training target (``memorisation.CodeIndex``), written once
to ``target-index.bin`` (deterministic bytes, its SHA-256 in run.json and every checkpoint) for the evaluation's
report-only memorisation read.

**Store layout** (as ``train_motion_tokens.py``): ``training-runs/<run>/checkpoints/step-XXXXXXXX.pt`` (the newest is
resumed and must match the configuration and the pairs' index), ``log/step-XXXXXXXX.json``, ``best.json`` and
``run.json``. A checkpoint carries the tokenizer's weights, so it fully defines the refiner.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

PAIRS_PREFIX = "acquisition/2026-10-17-v1/youtube-asl-v1/round15/refiner-pairs-v1"
INIT_KEY = "training-runs/tokens-rvq-d0b-prior-r12-f12/checkpoints/step-00060000.pt"
INIT_SHA256 = "99b235d30db16491f5f2d5713f4dd99b570a535d990e878b24f04173874aa51e"      # F12 step 60000
DONE = ("converted", "already_present")


def load_pairs(store, index: dict, threads: int = 16) -> list[dict]:
    """Every finished pair of the index, fetched, hash-checked and unpacked: ``{sign, split, target, valid, part,
    sources}`` (numpy), in the index's order."""
    from signtranslator.generation.refiner import unpack_pair
    entries = [e for e in index["entries"] if e["status"] in DONE]

    def one(e):
        payload = store.get(e["pair_key"])
        if hashlib.sha256(payload).hexdigest() != e["pair_sha256"]:
            raise SystemExit(f"{e['pair_key']}: SHA-256 differs from the index")
        arrays, meta = unpack_pair(payload)
        if meta["sign"] != e["sign"] or meta["split"] != e["split"] or meta["config_sha256"] != index["config_sha256"]:
            raise SystemExit(f"{e['pair_key']}: another sentence, split or configuration than the index says")
        k = index["config"]["k"]
        sources = [arrays[f"source_codes_{j}"] for j in range(k)]
        if any(len(s) != len(arrays["target_codes"]) for s in sources):
            raise SystemExit(f"{e['pair_key']}: an input's length differs from its target's")
        return dict(sign=e["sign"], split=e["split"], target=arrays["target_codes"], valid=arrays["target_valid"],
                    part=arrays["target_part"], sources=sources)
    with ThreadPoolExecutor(max(1, threads)) as pool:
        return list(pool.map(one, entries))


def batch_items(step: int, train: int, batch: int, k: int, seed: int) -> list[tuple[int, int]]:
    """(pair, input) of each row of step ``step``'s batch: steps walk through seeded per-epoch permutations of the
    ``train`` pairs (``batch`` rows per step, an epoch's tail completed from the next), each pair's input drawn per
    (epoch, pair)."""
    out, position = [], step * batch
    while len(out) < batch:
        epoch, offset = divmod(position, train)
        order = np.random.default_rng([seed, epoch]).permutation(train)
        pick = np.random.default_rng([seed, epoch, 1]).integers(k, size=train)
        take = order[offset:offset + batch - len(out)]
        out += [(int(i), int(pick[i])) for i in take]
        position += len(take)
    return out


def collate(rows, K: int, device) -> tuple:
    """(source (B, L, J), source mask, target (B, S, J), target valid (B, S), target parts (B, S, P)) of
    (pair, input) rows; sources padded with the start code K, targets with code 0 (never scored)."""
    J = rows[0][0]["target"].shape[1]
    P = rows[0][0]["part"].shape[1]
    L = max(len(p["sources"][k]) for p, k in rows)
    S = max(len(p["target"]) for p, _ in rows)
    source = torch.full((len(rows), L, J), K, dtype=torch.long)
    smask = torch.zeros((len(rows), L), dtype=torch.bool)
    target = torch.zeros((len(rows), S, J), dtype=torch.long)
    valid = torch.zeros((len(rows), S), dtype=torch.bool)
    part = torch.zeros((len(rows), S, P), dtype=torch.bool)
    for r, (p, k) in enumerate(rows):
        s = torch.from_numpy(p["sources"][k].astype(np.int64))
        source[r, :len(s)], smask[r, :len(s)] = s, True
        t = len(p["target"])
        target[r, :t] = torch.from_numpy(p["target"].astype(np.int64))
        valid[r, :t] = torch.from_numpy(p["valid"].astype(bool))
        part[r, :t] = torch.from_numpy(p["part"].astype(bool))
    return tuple(x.to(device) for x in (source, smask, target, valid, part))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
    p.add_argument("--pairs-prefix", default=PAIRS_PREFIX)
    p.add_argument("--pairs-index-sha256", help="pin the pairs' index.json")
    p.add_argument("--init-key", default=INIT_KEY)
    p.add_argument("--init-sha256", default=INIT_SHA256)
    p.add_argument("--run", required=True)
    p.add_argument("--epochs", type=float, default=8., help="the declared length: steps = ceil(epochs x pairs / batch)")
    p.add_argument("--max-epochs", type=float, default=10.)
    p.add_argument("--steps", type=int, help="smoke runs: this many steps instead (within --max-epochs)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--clip", type=float, default=1.)
    p.add_argument("--ema-decay", type=float, default=.999)
    p.add_argument("--source-layers", type=int, help="default: the prior's text layers")
    p.add_argument("--source-dropout", type=float, default=.1)
    p.add_argument("--history-dropout", type=float, help="default: the prior's (F12: 0.5)")
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--val-items", type=int, default=0, help="selection items scored (0: all, every input)")
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--index-ngram", type=int, default=12, help="memorisation index: steps per code n-gram")
    p.add_argument("--index-max-df", type=float, default=.01, help="memorisation index: n-grams in more than this "
                   "share of the targets are left out")
    p.add_argument("--fetch-threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    if args.device == "mps":
        raise SystemExit("MPS is refused (as for the prior)")
    import warnings
    warnings.simplefilter("ignore")
    from signtranslator.generation.motion_rvq import RVQPriorConfig
    from signtranslator.generation.refiner import (KIND, RefinerConfig, SentenceRefiner, refiner_loss,
                                                   state_dict_digest)
    from signtranslator.training.weight_ema import WeightEMA
    from train_text_to_motion import lr_at, make_store
    store = make_store(args.store)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    torch.manual_seed(args.seed)
    index_key = f"{args.pairs_prefix.rstrip('/')}/index.json"
    index_payload = store.get(index_key)
    index_sha = hashlib.sha256(index_payload).hexdigest()
    if args.pairs_index_sha256 and index_sha != args.pairs_index_sha256:
        raise SystemExit(f"{index_key} differs from --pairs-index-sha256")
    index = json.loads(index_payload)
    payload = store.get(args.init_key)
    if hashlib.sha256(payload).hexdigest() != args.init_sha256:
        raise SystemExit("the --init-key checkpoint differs from --init-sha256")
    init = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    del payload
    if init.get("kind") != "motion_rvq_prior":
        raise SystemExit(f"--init-key is not an RVQ prior ({init.get('kind')})")
    tokenizer_state = init["tokenizer_state"]
    tokenizer_digest = state_dict_digest(tokenizer_state)
    if tokenizer_digest != index["config"]["tokenizer_digest"]:
        raise SystemExit("the pairs were tokenized with another tokenizer than the prior's")
    config = RefinerConfig.from_prior(RVQPriorConfig.from_dict(init["config"]), source_layers=args.source_layers,
                                      source_dropout=args.source_dropout, history_dropout=args.history_dropout)
    model = SentenceRefiner(config).to(device)
    pairs = load_pairs(store, index, args.fetch_threads)
    train = [x for x in pairs if x["split"] == "train"]
    selection = [x for x in pairs if x["split"] == "selection"]
    if not train or not selection:
        raise SystemExit(f"{len(train)} training and {len(selection)} selection pairs: both are needed")
    K = index["config"]["k"]
    if not 0 < args.epochs <= args.max_epochs:
        raise SystemExit(f"--epochs {args.epochs}: must be in (0, --max-epochs {args.max_epochs}]")
    steps = args.steps if args.steps is not None else math.ceil(args.epochs * len(train) / args.batch_size)
    epochs = steps * args.batch_size / len(train)
    if steps < 1 or epochs > args.max_epochs:
        raise SystemExit(f"{steps} steps are {epochs:.2f} epochs: more than --max-epochs {args.max_epochs}")
    val_rows = [(x, k) for x in selection for k in range(K)]
    if args.val_items:
        val_rows = val_rows[:args.val_items]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    prefix = f"training-runs/{args.run}"
    step, resumed_ema, log_prefix = 0, None, []
    # the memorisation index of the training targets (written once; identical bytes on every resume)
    from signtranslator.generation.memorisation import CodeIndex
    from signtranslator.generation.refiner import pack_pair
    P = len(config.prior.tokenizer.part_slices())
    arrays, meta = CodeIndex.build([x["target"] for x in train], parts=P, codebook=model.K, n=args.index_ngram,
                                   max_df=args.index_max_df).to_arrays()
    target_index = pack_pair(arrays, dict(meta, signs=[x["sign"] for x in train], pairs_index_sha256=index_sha))
    target_index_key = f"{prefix}/target-index.bin"
    target_index_sha = hashlib.sha256(target_index).hexdigest()
    if store.exists(target_index_key):
        if store.get(target_index_key) != target_index:
            raise SystemExit(f"{target_index_key} exists with other contents")
    else:
        store.put(target_index_key, target_index)
    checkpoints = sorted(k for k in store.list(f"{prefix}/checkpoints/") if k.endswith(".pt"))
    if checkpoints:
        state = torch.load(io.BytesIO(store.get(checkpoints[-1])), map_location=device, weights_only=False)
        if state.get("kind") != KIND or state["config_sha256"] != config.sha256() \
                or state["pairs_index_sha256"] != index_sha or state["init_sha256"] != args.init_sha256:
            raise SystemExit("the newest checkpoint was trained with another configuration, pairs or warm start")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        resumed_ema, step = state.get("ema"), state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda_rng"]])
        logs = sorted(k for k in store.list(f"{prefix}/log/") if k.endswith(".json"))
        log_prefix = json.loads(store.get(logs[-1])) if logs else []
        print(f"resumed {checkpoints[-1]} at step {step}", flush=True)
    else:
        loaded = model.warm_start({k: v.to(device) for k, v in (init.get("ema") or init["model"]).items()})
        print(f"initialised from {args.init_key} (step {init.get('step')}): {json.dumps(loaded)}", flush=True)
        store.put(f"{prefix}/run.json", json.dumps(dict(
            kind=KIND, config=config.to_dict(), config_sha256=config.sha256(), init_key=args.init_key,
            init_sha256=args.init_sha256, pairs_index_key=index_key, pairs_index_sha256=index_sha, args=vars(args),
            train_pairs=len(train), selection_pairs=len(selection), selection_items=len(val_rows),
            selection_components=index["config"]["selection_components"], warm_start=loaded, steps=steps,
            epochs=round(epochs, 4), declared_epochs=args.epochs if args.steps is None else None,
            target_index_key=target_index_key, target_index_sha256=target_index_sha),
            indent=2, sort_keys=True).encode())
    del init
    ema = WeightEMA(model, args.ema_decay)
    if resumed_ema is not None:
        ema.shadow = {n: resumed_ema[n].to(device).clone() for n in ema.shadow}
        ema.updates = step
    best = json.loads(store.get(f"{prefix}/best.json")) if store.exists(f"{prefix}/best.json") else None

    def validate() -> dict:
        """Mean teacher-forced terms of the EMA weights on the selection items (no dropout of any kind)."""
        live = ema.swap(model)
        model.eval()
        totals, count = {}, 0
        with torch.no_grad():
            for b in range(0, len(val_rows), args.batch_size):
                rows = val_rows[b:b + args.batch_size]
                loss, terms = refiner_loss(model, *collate(rows, model.K, device), train=False)
                for key, value in dict(loss=float(loss), **terms).items():
                    totals[key] = totals.get(key, 0.) + float(value) * len(rows)
                count += len(rows)
        ema.restore(model, live)
        model.train()
        return {f"selection/{k}": v / max(count, 1) for k, v in totals.items()}

    log, started = list(log_prefix), time.monotonic()

    def checkpoint(metrics):
        nonlocal best
        buffer = io.BytesIO()
        torch.save(dict(kind=KIND, model=model.state_dict(), ema=ema.state_dict(model),
                        optimizer=optimizer.state_dict(), step=step, validation=metrics,
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                        target_index_key=target_index_key, target_index_sha256=target_index_sha,
                        epoch=round(step * args.batch_size / len(train), 4), config=config.to_dict(),
                        config_sha256=config.sha256(),
                        tokenizer_state=tokenizer_state, tokenizer_digest=tokenizer_digest,
                        tokenizer_key=index["config"]["tokenizer_key"], pairs_index_key=index_key,
                        pairs_index_sha256=index_sha, init_key=args.init_key, init_sha256=args.init_sha256), buffer)
        key = f"{prefix}/checkpoints/step-{step:08d}.pt"
        store.put(key, buffer.getvalue())
        store.put(f"{prefix}/log/step-{step:08d}.json", json.dumps(log, sort_keys=True).encode())
        if metrics and (best is None or metrics["selection/loss"] < best["selection_loss"]):
            best = dict(checkpoint_key=key, step=step, selection_loss=metrics["selection/loss"], validation=metrics,
                        selection_components=index["config"]["selection_components"])
            store.put(f"{prefix}/best.json", json.dumps(best, indent=2, sort_keys=True).encode())

    model.train()
    while step < steps:
        rows = [(train[i], k) for i, k in batch_items(step, len(train), args.batch_size, K, args.seed)]
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, base=args.lr, warmup=args.warmup, total=steps)
        loss, terms = refiner_loss(model, *collate(rows, model.K, device), train=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.step()
        ema.update(model)
        step += 1
        entry = dict(step=step, loss=value, grad_norm=float(norm), seconds=round(time.monotonic() - started, 1),
                     **{f"train/{k}": v for k, v in terms.items()})
        log.append(entry)
        metrics = None
        if step % args.val_every == 0 or step == steps:
            metrics = validate()
            entry.update(metrics)
            print(json.dumps({k: v for k, v in entry.items() if k == "step" or k.startswith("selection/")}),
                  flush=True)
            if (best is None or metrics["selection/loss"] < best["selection_loss"]) \
                    and step % args.checkpoint_every != 0 and step != steps:
                checkpoint(metrics)            # a new best between regular checkpoints is kept too
        if step % 50 == 0 or step == 1:
            print(json.dumps(entry), flush=True)
        if step % args.checkpoint_every == 0 or step == steps:
            checkpoint(metrics if metrics is not None else validate())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
