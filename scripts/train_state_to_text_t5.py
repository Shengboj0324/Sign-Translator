"""Train instrument v2, the state-to-text back-translator (``generation.state_to_text_t5``; run
``back-translator-state-t5-v1``; docs/NEXT_PHASE_PLAN.md, "Instrument v2: design and declared gates", 2026-10-30
09:10 UTC). A fork of ``train_keypoints_to_text_t5.py``: the same run layout (``training-runs/<run>/`` with
``run.json``, ``checkpoints``, ``log``, ``best.json``), EMA weights, two learning rates, resume and patience, with these
declared differences:

- **Data** (``--corpus KEY@SHA[:WEIGHT[:CORPUS,...]]``, repeated; ``generation.bt_corpora``): How2Sign train c02,
  YouTube-ASL train and ASL STEM Wiki train (minus its held-out participants) train the model, sampled in proportion
  to their corpus weights. How2Sign val is never loaded (``check_loadable_bt``), nor c00, test or held-out partitions;
  2M-Flores is refused in training. Train sentences equal or near to an evaluation sentence are dropped.
- **Input**: states conformed to the template (``--template KEY@SHA:CORPUS``: the first training state of that corpus,
  F12's ``--conform-to-corpus`` layout), at 24 fps, through ``data.state_features``.
- **Augmentation** (training only): ``augment_sign`` (speed, trim, face dropout, ``--validity-dropout``) and, with
  ``--recon-prefix`` / ``--recon-index-sha256`` (a ``build_recon_states.py`` index, pinned), the clip's tokenizer (d2)
  round-trip copy under the same text with probability ``--recon-p`` (default 0.4). No control input is ever
  trained on.
- **Selection** on held-out *train* data, never How2Sign val: ``--select`` roles (default How2Sign c04 and the
  ``--monitor-clips`` YouTube monitor clips: 536, all of them, amendment 1). ``--select-on gap`` (default, declared):
  BLEU(real) − BLEU(blank) on the pooled selection set, so a stronger text prior is not rewarded; ``real``: real BLEU
  (not the declared criterion). Held-out STEM participants are read too, reported only. Selection decodes greedily
  (beam 1, for speed) but reads every clip at full length, as the acceptance does (which decodes with beam 4); both
  are recorded in ``run.json`` (``selection.decoding``). Training clips are cut to ``--max-seconds``.
- **2M-Flores must be given** (``--corpus`` of a cross-domain manifest): its records are never trained on, but its
  sentences are always removed from the training text; without it the run is refused.
- ``run.json`` records the effective sampling proportions by mass (corpus weight x training clips, normalised).
- **Checkpoints** hold tensors and JSON-like values only (``state_to_text_t5.load_checkpoint`` reads them with
  ``weights_only=True``) and carry the licence tag.
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
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.keypoint_shards import collate_keypoints  # noqa: E402
from signtranslator.data.state_text_dataset import StateTextDataset  # noqa: E402
from signtranslator.generation import bt_corpora as bt  # noqa: E402
from signtranslator.generation.motion_to_text import corpus_bleu  # noqa: E402
from signtranslator.generation.state_to_text_t5 import (KIND, LICENCE, TEMPLATE_FIELDS, StateToT5Config,  # noqa: E402
                                                       build, motion_config)
from signtranslator.training.weight_ema import WeightEMA  # noqa: E402
from train_text_to_motion import lr_at, make_store  # noqa: E402

TEMPLATE = ("acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json"
            "@ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a:how2sign-two-view-text-v1")
SELECT = f"{bt.SELECT_H2S},{bt.SELECT_YT}"


def to_device(batch, device):
    (x, valid), texts = batch
    return (x.to(device, non_blocking=True), valid.to(device, non_blocking=True)), texts


@torch.no_grad()
def translate_batches(model, batches, device, *, beam: int = 1) -> tuple[list[str], list[str], float]:
    """(hypotheses, references, mean teacher-forced loss)."""
    model.eval()
    hypotheses, references, losses = [], [], []
    for batch in batches:
        inputs, texts = to_device(batch, device)
        loss, _ = model.loss(inputs, texts)
        losses.append(float(loss.detach()))
        hypotheses += model.generate_text(inputs, beam=beam)
        references += list(texts)
    model.train()
    return hypotheses, references, float(np.mean(losses)) if losses else float("nan")


def selection_metrics(model, sets: dict, device) -> dict:
    """Per set (name -> (real batches, blank batches)): BLEU of real and blank and their difference; ``pooled``: the
    same over every selection set (``select:*``) together. Report-only sets (``holdout:*``) are scored alone."""
    out, pooled = {}, {"real": ([], []), "blank": ([], [])}
    for name, (real, blank) in sets.items():
        row = {}
        for form, batches in (("real", real), ("blank", blank)):
            hyps, refs, loss = translate_batches(model, batches, device)
            row[form] = round(corpus_bleu(hyps, refs)["bleu"], 4)
            row[f"{form}_loss"] = round(loss, 4)
            if form == "real":
                row["examples"] = [list(x) for x in zip(hyps[:3], refs[:3])]
            if name.startswith("select"):
                pooled[form][0].extend(hyps); pooled[form][1].extend(refs)
        row["gap"] = round(row["real"] - row["blank"], 4)
        out[name] = row
    if pooled["real"][0]:
        real, blank = corpus_bleu(*pooled["real"])["bleu"], corpus_bleu(*pooled["blank"])["bleu"]
        out["pooled"] = dict(real=round(real, 4), blank=round(blank, 4), gap=round(real - blank, 4),
                             clips=len(pooled["real"][0]))
    return out


def sampling_mass(records) -> dict:
    """{corpus: clips, weight, mass_share}: the effective sampling proportions (weight x clips, normalised)."""
    rows = {}
    for r in records:
        row = rows.setdefault(r["corpus"], dict(clips=0, weight=float(r["_weight"]), mass=0.))
        row["clips"] += 1
        row["mass"] += float(r["_weight"])
    total = sum(row["mass"] for row in rows.values()) or 1.
    return {c: dict(clips=row["clips"], weight=row["weight"], mass_share=round(row["mass"] / total, 4))
            for c, row in sorted(rows.items())}


def load_store_manifest(store, key: str, sha: str):
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get(key) + b"\n")
        return load_manifest(path, expected_sha256=sha)


def template_state(store, spec: str, cache):
    """(state, pin) of ``KEY@SHA:CORPUS``: the corpus's first training record in the manifest (its layouts)."""
    from signtranslator.data_engineering.corpus_manifest import load_sample
    from signtranslator.generation.recognizer_corpora import check_loadable
    key, rest = spec.split("@")
    sha, corpus = rest.split(":")
    manifest = load_store_manifest(store, key, sha)
    record = next((r for r in manifest["records"] if r.get("corpus") == corpus and r.get("partition") == "train"), None)
    if record is None:
        raise SystemExit(f"--template: no training record of corpus {corpus!r}")
    check_loadable(record)
    pin = dict(manifest_key=key, manifest_sha256=sha, corpus=corpus, sign=record["sign"],
               state_sha256=record["continuous"]["state_sha256"])
    return load_sample(record, "continuous", cache=cache, fetch=store.get), tuple((k, pin[k]) for k in TEMPLATE_FIELDS)


def assemble_data(store, args):
    """(records, summary, recon info): the corpora assembled (``bt_corpora.assemble``) and the round-trip copies
    attached."""
    from signtranslator.generation import recognizer_corpora as rc
    specs = [bt.parse_corpus_spec(c) for c in args.corpus]
    for i, a in enumerate(specs):                      # a manifest may be named twice, for disjoint corpora (weights)
        for b in specs[i + 1:]:
            if a.key == b.key and (not a.corpora or not b.corpora or set(a.corpora) & set(b.corpora)):
                raise SystemExit(f"{a.key} is named twice for the same corpora")
    loaded = {}
    for s in specs:
        if (s.key, s.sha256) not in loaded:
            loaded[(s.key, s.sha256)] = load_store_manifest(store, s.key, s.sha256)
    corpora = [(loaded[(s.key, s.sha256)], s) for s in specs]
    records, summary = bt.assemble(corpora, monitor_clips=args.monitor_clips)
    del corpora, loaded
    if not any(r["_role"] == bt.CROSSDOMAIN for r in records):
        raise SystemExit("give the 2M-Flores (cross-domain) manifest as a --corpus: its sentences must be removed from "
                         "the training text (it is never trained on)")
    if args.limit:                                     # tests and smoke runs: the first LIMIT records of each role
        by_role = {}
        for r in records:
            by_role.setdefault(r["_role"], []).append(r)
        records = [r for rs in by_role.values() for r in rs[:args.limit]]
    recon = None
    if args.recon_prefix:
        if not args.recon_index_sha256:
            raise SystemExit("--recon-prefix needs --recon-index-sha256 (the index is pinned)")
        payload = store.get(f"{args.recon_prefix.rstrip('/')}/index.json")
        if hashlib.sha256(payload).hexdigest() != args.recon_index_sha256:
            raise SystemExit("the round-trip index differs from --recon-index-sha256")
        index = json.loads(payload)
        train = [r for r in records if r["_role"] == bt.TRAIN]
        counts = rc.attach_recon(train, rc.recon_lookup(index))
        for r in records:                              # only training records may carry a copy
            if r["_role"] != bt.TRAIN:
                r.pop("_recon", None)
        coverage = counts.get("train:with", 0) / max(len(train), 1)
        recon = dict(prefix=args.recon_prefix.rstrip("/"), index_sha256=args.recon_index_sha256,
                     index_config=index.get("config"), coverage=dict(sorted(counts.items())),
                     train_coverage=round(coverage, 4), recon_p=args.recon_p)
        print(json.dumps(dict(recon={k: v for k, v in recon.items() if k != "index_config"})), flush=True)
        if coverage < args.recon_min_coverage:
            raise SystemExit(f"round-trip copies cover {coverage:.3f} of the training clips, below "
                             f"--recon-min-coverage {args.recon_min_coverage}")
    elif args.recon_p:
        raise SystemExit("--recon-p needs --recon-prefix (or pass --recon-p 0)")
    return records, summary, recon


def main(argv=None, *, build_model=build) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--corpus", action="append", required=True, help="KEY@SHA[:WEIGHT[:CORPUS,...]] (repeated)")
    parser.add_argument("--template", default=TEMPLATE, help="KEY@SHA:CORPUS: the layouts states are conformed to")
    parser.add_argument("--select", default=SELECT, help="comma-separated selection roles (bt_corpora)")
    parser.add_argument("--select-on", choices=("gap", "real"), default="gap",
                        help="gap (declared): pooled selection BLEU(real) - BLEU(blank); real: real BLEU")
    parser.add_argument("--monitor-clips", type=int, default=bt.MONITOR_CLIPS)
    parser.add_argument("--select-clips", type=int, default=0, help="at most this many clips per selection set (0: all)")
    parser.add_argument("--report-clips", type=int, default=500, help="held-out (report-only) clips read per validation")
    parser.add_argument("--run", required=True)
    parser.add_argument("--t5-name", default="google/flan-t5-base")
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-seconds", type=float, default=20.)
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--validity-dropout", type=float, default=.3)
    parser.add_argument("--face-dropout", type=float, default=.2)
    parser.add_argument("--recon-prefix")
    parser.add_argument("--recon-index-sha256")
    parser.add_argument("--recon-p", type=float, default=.4)
    parser.add_argument("--recon-min-coverage", type=float, default=.9)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-bytes", type=int, default=64 * 2 ** 30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="records per role (tests and smoke runs)")
    args = parser.parse_args(argv)
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    store = make_store(args.store)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    cache = ContentCache(Path(os.environ.get("SIGN_CACHE_DIR", tempfile.gettempdir())) / "sign-translator-cache",
                         args.cache_bytes)
    selection_roles = [s for s in args.select.split(",") if s]
    if not selection_roles or not all(s.startswith("select:") for s in selection_roles):
        raise SystemExit("--select names selection roles (select:...), never validation or training data")
    template, pin = template_state(store, args.template, cache)
    records, summary, recon = assemble_data(store, args)
    config = StateToT5Config(motion_config(template), template=pin, fps=args.fps, t5_name=args.t5_name,
                             d_model=args.d_model, heads=args.heads, encoder_layers=args.encoder_layers,
                             feedforward=args.feedforward, dropout=args.dropout, downsample=args.downsample,
                             max_tokens=args.max_tokens)
    common = dict(cache=cache, fetch=store.get)
    train = StateTextDataset(bt.records_of(records, bt.TRAIN), config, training=True, augment=not args.no_augment,
                             recon_p=args.recon_p if recon else 0., validity_dropout=args.validity_dropout,
                             face_dropout=args.face_dropout, max_seconds=args.max_seconds, **common)
    if len(train) < args.batch_size:
        raise SystemExit("at least one full training batch is needed")

    def fixed(role, cap, seed=1234):
        rs = [r for r in records if r["_role"] == role]       # assembled order: (corpus, sign)
        if cap:
            rs = [rs[i] for i in sorted(random.Random(seed).sample(range(len(rs)), min(cap, len(rs))))]
        if not rs:
            return None

        def batches(blank):
            ds = StateTextDataset(rs, config, training=False, blank=blank, max_seconds=None, **common)   # full length
            return list(DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_keypoints))
        return batches(False), batches(True)
    sets = {}
    for role in selection_roles:
        got = fixed(role, args.select_clips)
        if got is None:
            raise SystemExit(f"--select {role}: no record has this role")
        sets[role] = got
    for role in sorted({r["_role"] for r in records if r["_role"].startswith("holdout:")}):
        sets[role] = fixed(role, args.report_clips)
    print(json.dumps(dict(train=len(train), roles=summary["roles"],
                          sets={k: sum(len(b[1]) for b in v[0]) for k, v in sets.items()})), flush=True)
    data_sha = bt.data_digest(records)
    sampling = sampling_mass(train.records)
    print(json.dumps(dict(sampling=sampling)), flush=True)
    prefix = f"training-runs/{args.run}"
    checkpoints = [k for k in store.list(f"{prefix}/checkpoints/") if k.endswith(".pt")]
    model = build_model(config, pretrained=not checkpoints).to(device)
    optimizer = torch.optim.AdamW(model.param_groups(motion_lr=args.lr, decoder_lr=args.decoder_lr,
                                                     weight_decay=args.weight_decay))
    step, resumed_ema = 0, None
    if checkpoints:
        from signtranslator.generation.state_to_text_t5 import load_checkpoint
        state = load_checkpoint(store.get(checkpoints[-1]))
        if state["config_sha256"] != config.sha256() or state["data_sha256"] != data_sha:
            raise SystemExit("the newest checkpoint was trained with a different config or data")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        resumed_ema, step = state.get("ema"), state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed {checkpoints[-1]} at step {step}", flush=True)
    else:
        store.put(f"{prefix}/run.json", json.dumps(dict(
            kind=KIND, config=config.to_dict(), config_sha256=config.sha256(), data_sha256=data_sha, licence=LICENCE,
            data=summary, recon=recon, sampling=sampling,
            selection=dict(roles=selection_roles, criterion=args.select_on,
                           decoding=dict(selection=dict(beam=1, max_seconds=None), training_max_seconds=args.max_seconds,
                                         acceptance=dict(beam=4, max_seconds=None),
                                         note="selection decodes greedily for speed; both read full-length clips")),
            args=vars(args), train_clips=len(train)), indent=2, sort_keys=True, default=str).encode())
    sampler = WeightedRandomSampler(train.weights(), num_samples=len(train), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed + step))
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        collate_fn=collate_keypoints, persistent_workers=args.workers > 0, drop_last=True,
                        pin_memory=device.type == "cuda")
    ema = WeightEMA(model, args.ema_decay)
    if resumed_ema is not None:
        ema.shadow = {n: resumed_ema[n].to(device).clone() for n in ema.shadow}
        ema.updates = step
    best = json.loads(store.get(f"{prefix}/best.json")) if store.exists(f"{prefix}/best.json") else None
    stale, log, started = 0, [], time.monotonic()
    selector = "pooled/gap" if args.select_on == "gap" else "pooled/real"

    def criterion(metrics):
        return float(metrics["pooled"][selector.split("/")[1]])

    def validate() -> dict:
        live = ema.swap(model)
        metrics = selection_metrics(model, sets, device)
        ema.restore(model, live)
        return metrics

    def checkpoint(metrics=None):
        nonlocal best
        buffer = io.BytesIO()
        torch.save(dict(kind=KIND, model=model.state_dict(), ema=ema.state_dict(model), optimizer=optimizer.state_dict(),
                        step=step, validation=json.loads(json.dumps(metrics)) if metrics else None,
                        torch_rng=torch.get_rng_state(), config=config.to_dict(), config_sha256=config.sha256(),
                        data_sha256=data_sha, licence=LICENCE, run=args.run), buffer)
        key = f"{prefix}/checkpoints/step-{step:08d}.pt"
        items = [(key, buffer.getvalue()),
                 (f"{prefix}/log/step-{step:08d}.json", json.dumps(log, sort_keys=True, default=str).encode())]
        if metrics and (best is None or criterion(metrics) > best["selection_value"]):
            best = dict(checkpoint_key=key, checkpoint_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(), step=step,
                        selection_value=criterion(metrics), selected_on=selector, validation=metrics)
            items.append((f"{prefix}/best.json", json.dumps(best, indent=2, sort_keys=True, default=str).encode()))
        uploader.submit(items)

    from signtranslator.training.async_upload import Uploader
    uploader = Uploader(store)
    model.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            inputs, texts = to_device(batch, device)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, base=group["base_lr"], warmup=args.warmup, total=args.steps)
            loss, terms = model.loss(inputs, texts)
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            ema.update(model)
            step += 1
            entry = dict(step=step, loss=float(loss.detach()), grad_norm=float(norm.detach()),
                         seconds=round(time.monotonic() - started, 1),
                         **{f"train/{k}": float(v.detach()) for k, v in terms.items()})
            metrics = None
            if step % args.val_every == 0 or step == args.steps:
                metrics = validate()
                entry.update({f"{k}/{f}": v for k, row in metrics.items() for f, v in row.items() if f != "examples"})
                improved = best is None or criterion(metrics) > best["selection_value"]
                stale = 0 if improved else stale + 1
                print(json.dumps(dict(step=step, examples={k: row.get("examples") for k, row in metrics.items()})),
                      flush=True)
            log.append(entry)
            if step % 100 == 0 or step == 1 or metrics:
                print(json.dumps(entry), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps or (metrics and stale == 0):
                checkpoint(metrics)
            if metrics and stale >= args.patience:
                print(f"early stop at step {step}: no new best {selector} in {stale} validations", flush=True)
                checkpoint(metrics)
                uploader.drain()
                return 0
    uploader.drain()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)                   # as train_keypoints_to_text_t5.py: no interpreter teardown after the uploads
