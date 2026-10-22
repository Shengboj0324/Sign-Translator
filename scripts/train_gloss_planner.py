"""Train the English -> ASL gloss planner (locus slots + non-manual markers) on the ASLLRP / RIT pairs (plan item 2).

Input: ``build_asllrp_english_gloss.py``'s ``pairs.json`` (hash-checked). The English word vocabulary and the gloss
vocabulary are built from the training partition only and stored with the run. Same layout as the other trainers
(``training-runs/<run>/run.json``, ``checkpoints``, ``log``, ``best.json``); EMA weights are validated every
``--val-every`` steps on the whole validation partition: teacher-forced losses plus decoded plans scored by
``evaluate_plans`` (gloss WER and BLEU, marker F1, locus accuracy). ``best.json`` keeps the checkpoint with the highest
validation gloss BLEU; ``--patience`` stops after that many validations without a new best. The test partition is
never read here.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.generation.motion_to_text import WordVocabulary  # noqa: E402
from signtranslator.planning.asllrp_gloss import MARKERS, GlossVocabulary, encode_target, plan_target  # noqa: E402
from signtranslator.planning.english_gloss import (EnglishGlossConfig, EnglishGlossPlanner, encode_sources,  # noqa: E402
                                                   evaluate_plans, plan_from_generation)
from signtranslator.training.weight_ema import WeightEMA  # noqa: E402
from train_text_to_motion import lr_at, make_store  # noqa: E402

KIND = "english_gloss_planner"


def batches_of(rows, words: WordVocabulary, glosses: GlossVocabulary, cfg: EnglishGlossConfig):
    """Tensors (source, tokens, source loci, target loci, markers, unit-end mask) for ``rows`` (pair, target)."""
    enc = [encode_target(glosses, t, max_tokens=cfg.max_target) for _, t in rows]
    L = max(len(e[0]) for e in enc)

    def pad(k, fill):
        return torch.tensor([e[k] + [fill] * (L - len(e[k])) for e in enc])
    source = encode_sources(words, [p["english"] for p, _ in rows], max_tokens=cfg.max_source)
    return source, pad(0, 0), pad(1, 0), pad(2, 0), pad(3, [0] * len(MARKERS)), pad(4, False)


def reference(target) -> dict:
    return dict(glosses=list(target.bases), loci=[tuple(x) for x in target.loci], markers=[list(m) for m in target.markers])


@torch.no_grad()
def validate(model, rows, words, glosses, cfg, device, *, batch: int, beam: int) -> dict:
    model.eval()
    losses, refs, hyps = [], [], []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        tensors = [t.to(device) for t in batches_of(chunk, words, glosses, cfg)]
        loss, _ = model.loss(*tensors)
        losses.append(float(loss))
        hyps += [plan_from_generation(glosses, g) for g in model.generate(tensors[0], beam=beam)]
        refs += [reference(t) for _, t in chunk]
    model.train()
    scores = evaluate_plans(refs, hyps)
    scores["loss"] = float(np.mean(losses))
    scores["examples"] = [dict(english=p["english"], reference=" ".join(r["glosses"]), plan=" ".join(h["glosses"]))
                          for (p, _), r, h in list(zip(rows, refs, hyps))[:5]]
    return scores


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--pairs-key", default="acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/english-gloss-v1/pairs.json")
    parser.add_argument("--pairs-sha256", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=.1)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=.3)
    parser.add_argument("--label-smoothing", type=float, default=.1)
    parser.add_argument("--min-word-count", type=int, default=2)
    parser.add_argument("--min-gloss-count", type=int, default=2)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-beam", type=int, default=1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--ema-decay", type=float, default=.995)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    payload = store.get(args.pairs_key)
    if hashlib.sha256(payload).hexdigest() != args.pairs_sha256:
        raise SystemExit(f"{args.pairs_key} does not match --pairs-sha256")
    pairs = json.loads(payload)["pairs"]
    train = [(p, plan_target(p)) for p in pairs if p["partition"] == "train"]
    val = [(p, plan_target(p)) for p in pairs if p["partition"] == "val"]
    if not train or not val:
        raise SystemExit("the pairs need training and validation partitions")
    prefix = f"training-runs/{args.run}"
    if store.exists(f"{prefix}/run.json"):
        saved = json.loads(store.get(f"{prefix}/run.json"))
        words, glosses = WordVocabulary(saved["words"]), GlossVocabulary(saved["glosses"])
    else:
        words = WordVocabulary.build([p["english"] for p, _ in train], min_count=args.min_word_count)
        glosses = GlossVocabulary.build([t for _, t in train], min_count=args.min_gloss_count)
    cfg = EnglishGlossConfig(len(words), len(glosses), d_model=args.d_model, heads=args.heads,
                             encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
                             feedforward=args.feedforward, dropout=args.dropout, label_smoothing=args.label_smoothing)
    model = EnglishGlossPlanner(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step, resumed_ema = 0, None
    checkpoints = [k for k in store.list(f"{prefix}/checkpoints/") if k.endswith(".pt")]
    if checkpoints:
        state = torch.load(io.BytesIO(store.get(checkpoints[-1])), map_location=device, weights_only=False)
        if state["config_sha256"] != cfg.sha256() or state["pairs_sha256"] != args.pairs_sha256:
            raise SystemExit("the newest checkpoint was trained with a different config or pairs file")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        resumed_ema, step = state.get("ema"), state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed {checkpoints[-1]} at step {step}", flush=True)
    else:
        store.put(f"{prefix}/run.json", json.dumps(dict(
            kind=KIND, config=cfg.to_dict(), config_sha256=cfg.sha256(), words=words.to_dict()["words"],
            glosses=glosses.to_dict()["glosses"], pairs_key=args.pairs_key, pairs_sha256=args.pairs_sha256,
            args=vars(args), train_pairs=len(train), val_pairs=len(val)), indent=2, sort_keys=True, default=str).encode())
    ema = WeightEMA(model, args.ema_decay)
    if resumed_ema is not None:
        ema.shadow = {n: resumed_ema[n].to(device).clone() for n in ema.shadow}
        ema.updates = step
    best = json.loads(store.get(f"{prefix}/best.json")) if store.exists(f"{prefix}/best.json") else None
    stale, log, started = 0, [], time.monotonic()
    rng = random.Random(args.seed + step)

    def checkpoint(metrics=None):
        nonlocal best
        buffer = io.BytesIO()
        torch.save(dict(kind=KIND, model=model.state_dict(), ema=ema.state_dict(model), optimizer=optimizer.state_dict(),
                        step=step, validation=metrics, torch_rng=torch.get_rng_state(), config=cfg.to_dict(),
                        config_sha256=cfg.sha256(), words=words.to_dict()["words"], glosses=glosses.to_dict()["glosses"],
                        pairs_sha256=args.pairs_sha256), buffer)
        key = f"{prefix}/checkpoints/step-{step:08d}.pt"
        store.put(key, buffer.getvalue())
        store.put(f"{prefix}/log/step-{step:08d}.json", json.dumps(log, sort_keys=True, default=str).encode())
        if metrics and (best is None or metrics["gloss_bleu"] > best["val_gloss_bleu"]):
            best = dict(checkpoint_key=key, step=step, val_gloss_bleu=metrics["gloss_bleu"], validation=metrics)
            store.put(f"{prefix}/best.json", json.dumps(best, indent=2, sort_keys=True, default=str).encode())

    model.train()
    while step < args.steps:
        chunk = [train[i] for i in rng.sample(range(len(train)), min(args.batch_size, len(train)))]
        tensors = [t.to(device) for t in batches_of(chunk, words, glosses, cfg)]
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, base=args.lr, warmup=args.warmup, total=args.steps)
        loss, terms = model.loss(*tensors)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        ema.update(model)
        step += 1
        entry = dict(step=step, loss=float(loss.detach()), grad_norm=float(norm), seconds=round(time.monotonic() - started, 1),
                     **{k: float(v) for k, v in terms.items()})
        metrics = None
        if step % args.val_every == 0 or step == args.steps:
            live = ema.swap(model)
            metrics = validate(model, val, words, glosses, cfg, device, batch=64, beam=args.val_beam)
            ema.restore(model, live)
            entry.update({f"val/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
            improved = best is None or metrics["gloss_bleu"] > best["val_gloss_bleu"]
            stale = 0 if improved else stale + 1
            print(json.dumps(dict(step=step, val=metrics)), flush=True)
        log.append(entry)
        if step % 100 == 0 or step == 1:
            print(json.dumps(entry), flush=True)
        if metrics:
            checkpoint(metrics)
            if stale >= args.patience:
                print(f"early stop at step {step}: no new best validation gloss BLEU in {stale} validations", flush=True)
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
