"""Train the isolated-sign recogniser (plan item 4a) on a gloss-labelled manifest (ASL Citizen).

Same store and checkpoint layout as the other trainers (``training-runs/<run>/checkpoints``, ``log``,
``best.json``). The label set is the training partition's glosses (sorted, stored with the run). Validation (EMA
weights) reports loss and top-1/top-5 accuracy on a fixed set of up to ``--val-clips`` validation clips; the
checkpoint with the highest validation top-1 is recorded in ``best.json``; ``--patience`` stops after that many
validations without a new best. The test partition is never read here.

**Recogniser v3** (docs/NEXT_PHASE_PLAN.md, "Instrument finding and the recogniser v3 plan"), on when any of its
options is given; without them every behaviour, the run's ``run.json`` included, is recogniser v2's (the v2 command
line is unchanged and gives the same data, random streams and records):
- ``--corpus KEY@SHA:MAPPING[:WEIGHT]`` (repeated): further corpora mapped into the label space
  (``signtranslator.generation.recognizer_corpora``: ``label``, ``asllex``, ``signbank``, ``upper``), the primary
  ``--manifest-key`` always taking part with mapping ``label`` (weight ``--primary-weight``). Weights other than 1
  sample training clips in proportion. ``--extra-manifest-key`` (v2's single ASLLVD corpus) cannot be combined.
- ``--holdout-signers corpus:id,id`` / ``--exclude-signers corpus:id,id``: every clip of a held-out signer becomes
  ``holdout:<corpus>`` (reported, never selected on, never trained); excluded signers are dropped.
- ``--labels-from-run``: the label space of an earlier run (v3: v2's 2,731 glosses).
- ``--conform-template KEY@SHA:CORPUS``: every state is conformed to that corpus's first training state (the
  generator's layout: round10-mixed-v1, how2sign-two-view-text-v1), so corpora and decoded copies batch together.
- ``--recon-prefix`` (``build_recon_states.py``'s index, pinned by ``--recon-index-sha256``): a training clip with a
  tokenizer-decoded copy (made from exactly the record as loaded) is served as that copy with probability
  ``--recon-p``; coverage below ``--recon-min-coverage`` of the training clips stops the run.
- ``--validity-dropout``: the probability of random arm / torso invalidation in ``augment_sign`` (with ``--augment``).
- **Selection:** the mean over corpora of validation top-1 on the real clips and on their decoded copies (up to
  ``--val-clips`` per corpus and form; the decoded set is the real set's clips that have a copy).
- **Guard:** no test-partition record and no How2Sign signer-9 record reaches a dataset (``check_loadable``, at
  assembly and on every load; it raises).
"""
from __future__ import annotations

import argparse
import functools
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
torch.multiprocessing.set_sharing_strategy("file_system")   # many small tensors per batch exhaust fd passing ("received 0 items of ancdata")
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest  # noqa: E402
from signtranslator.generation.canonical_state import channel_bindings_from_state  # noqa: E402
from signtranslator.generation.sign_recognizer import (GlossMotionDataset, augment_sign, SignRecognizer,  # noqa: E402
                                                       SignRecognizerConfig, collate_gloss_motion, top_k_accuracy)
from signtranslator.generation.text_to_motion import TextToMotionConfig  # noqa: E402
from signtranslator.training.weight_ema import WeightEMA  # noqa: E402
from train_text_to_motion import lr_at, make_store, to_device  # noqa: E402

KIND = "sign_recognizer"
_AUGMENT_RNG = {}
# Recogniser v3 options: absent from a v2 run's run.json, so the v2 command writes exactly what it always wrote.
V3_DEFAULTS = dict(corpus=None, holdout_signers=None, exclude_signers=None, whole_clip=None, labels_from_run=None,
                   conform_template=None, recon_prefix=None, recon_index_sha256=None, recon_p=.5,
                   recon_min_coverage=.95, validity_dropout=0., primary_weight=1.)


def worker_augment(state):
    """``augment_sign`` with one generator per loader process, seeded from torch (workers get distinct seeds)."""
    pid = os.getpid()
    if pid not in _AUGMENT_RNG:
        _AUGMENT_RNG[pid] = random.Random(torch.initial_seed())
    return augment_sign(state, _AUGMENT_RNG[pid])


def worker_augment_v3(state, *, validity_dropout: float):
    """``worker_augment`` with the v3 validity dropout (the same per-process generator)."""
    pid = os.getpid()
    if pid not in _AUGMENT_RNG:
        _AUGMENT_RNG[pid] = random.Random(torch.initial_seed())
    return augment_sign(state, _AUGMENT_RNG[pid], validity_dropout=validity_dropout)


def with_extra_corpus(store, manifest, key, sha, gloss_map_key, labels):
    """``manifest`` plus the extra corpus's clips whose Sign Bank gloss maps *exactly* to one label: its training
    partition joins training, its validation partition becomes ``cross_val`` (reported, never used for selection),
    its test partition is left out unread."""
    extra = load_store_manifest(store, key, sha)
    to_label = {}
    for g, m in json.loads(store.get(gloss_map_key))["mapping"].items():
        if m["method"] == "exact" and g in labels:
            for c in m["candidates"]:
                to_label.setdefault(c, set()).add(g)
    added = []
    for r in extra["records"]:
        target = to_label.get(r["gloss"]["label"], set())
        if len(target) != 1 or r.get("partition") not in ("train", "val"):
            continue
        added.append(dict(r, partition="train" if r["partition"] == "train" else "cross_val",
                          gloss=dict(r["gloss"], label=next(iter(target)), source_label=r["gloss"]["label"])))
    print(json.dumps(dict(extra_corpus=key, train=sum(a["partition"] == "train" for a in added),
                          cross_val=sum(a["partition"] == "cross_val" for a in added))), flush=True)
    return dict(manifest, records=list(manifest["records"]) + added)


def load_store_manifest(store, key, sha):
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "manifest.json"
        path.write_bytes(store.get(key) + b"\n")
        return load_manifest(path, expected_sha256=sha)


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


def is_v3(args) -> bool:
    return any(getattr(args, k) != v for k, v in V3_DEFAULTS.items())


def split_spec(text: str, what: str) -> tuple[str, str, str]:
    """``KEY@SHA:CORPUS`` -> (key, sha, corpus)."""
    if text.count("@") != 1 or text.split("@")[1].count(":") != 1:
        raise SystemExit(f"{what} {text!r}: expected KEY@SHA:CORPUS")
    key, rest = text.split("@")
    sha, corpus = rest.split(":")
    return key, sha, corpus


def conform_template(store, spec: str, cache):
    """The first training state of the named corpus of a manifest (``train_motion_tokens.layout_template``), after
    the loading guard."""
    from signtranslator.data_engineering.corpus_manifest import load_sample
    from signtranslator.generation.recognizer_corpora import check_loadable
    key, sha, corpus = split_spec(spec, "--conform-template")
    manifest = load_store_manifest(store, key, sha)
    record = next((r for r in manifest["records"] if r.get("corpus") == corpus and r.get("partition") == "train"), None)
    if record is None:
        raise SystemExit(f"--conform-template: no training record of corpus {corpus!r}")
    check_loadable(record)
    return load_sample(record, "continuous", cache=cache, fetch=store.get), dict(key=key, sha256=sha, corpus=corpus,
                                                                                  sign=record.get("sign"))


def build_v3_data(store, args, cache, prefix):
    """(records, labels, summary, template, data digest) of the v3 corpora."""
    from signtranslator.generation import recognizer_corpora as rc
    if args.extra_manifest_key:
        raise SystemExit("--extra-manifest-key is recogniser v2's single extra corpus; with v3 options use --corpus")
    primary = load_store_manifest(store, args.manifest_key, args.manifest_sha256)
    specs = [rc.CorpusSpec(args.manifest_key, args.manifest_sha256, "label", float(args.primary_weight))]
    specs += [rc.parse_corpus_spec(c) for c in args.corpus or ()]
    if len({s.key for s in specs}) != len(specs):
        raise SystemExit("a corpus is named twice (the primary manifest always takes part)")
    if store.exists(f"{prefix}/run.json"):
        labels = json.loads(store.get(f"{prefix}/run.json"))["labels"]
    elif args.labels_from_run:
        labels = json.loads(store.get(f"training-runs/{args.labels_from_run}/run.json"))["labels"]
    else:
        labels = sorted({r["gloss"]["label"] for r in primary["records"]
                         if r.get("partition") == "train" and (r.get("gloss") or {}).get("label")})
    label_set = set(labels)
    gloss_map = json.loads(store.get(args.gloss_map_key)) if any(s.mapping == "signbank" for s in specs) else None
    codes = rc.code_labels(primary, label_set)
    signbank = rc.signbank_targets(gloss_map, label_set) if gloss_map else None
    corpora = []
    for spec in specs:
        manifest = primary if spec.key == args.manifest_key else load_store_manifest(store, spec.key, spec.sha256)
        corpora.append((manifest, spec, rc.label_mapper(spec.mapping, label_set, codes=codes, signbank=signbank)))
    try:
        records, summary = rc.assemble(corpora, label_set, holdout=rc.parse_signer_specs(args.holdout_signers),
                                       exclude=rc.parse_signer_specs(args.exclude_signers),
                                       whole_clip=args.whole_clip or ())
    except ValueError as error:
        raise SystemExit(str(error))
    recon = None
    if args.recon_prefix:
        if not args.recon_index_sha256:
            raise SystemExit("--recon-prefix needs --recon-index-sha256 (the index is pinned)")
        payload = store.get(f"{args.recon_prefix.rstrip('/')}/index.json")
        if hashlib.sha256(payload).hexdigest() != args.recon_index_sha256:
            raise SystemExit("the decoded-state index differs from --recon-index-sha256")
        index = json.loads(payload)
        counts = rc.attach_recon(records, rc.recon_lookup(index))
        train_with, train_without = counts.get("train:with", 0), counts.get("train:without", 0)
        coverage = train_with / max(train_with + train_without, 1)
        recon = dict(prefix=args.recon_prefix.rstrip("/"), index_sha256=args.recon_index_sha256,
                     index_config=index.get("config"), coverage=dict(sorted(counts.items())),
                     train_coverage=round(coverage, 4), recon_p=args.recon_p)
        print(json.dumps(dict(recon=recon)), flush=True)
        if coverage < args.recon_min_coverage:
            raise SystemExit(f"decoded copies cover {coverage:.3f} of the training clips, below "
                             f"--recon-min-coverage {args.recon_min_coverage}")
    elif args.recon_p != V3_DEFAULTS["recon_p"]:
        raise SystemExit("--recon-p needs --recon-prefix")
    template, template_info = (conform_template(store, args.conform_template, cache) if args.conform_template
                               else (None, None))
    summary = dict(summary, recon=recon, template=template_info)
    rows = sorted((r["corpus"], r["sign"], r["partition"], r["gloss"]["label"], rc.load_digest(r),
                   (r.get("_recon") or {}).get("state_sha256"), r["_weight"]) for r in records)
    digest = rc.data_digest(dict(labels=labels, rows=rows, template=template_info, recon_p=args.recon_p if recon else 0.,
                                 validity_dropout=args.validity_dropout))
    print(json.dumps(dict(data=dict(partitions=summary["partitions"], corpora={k: v["counts"] for k, v in
                                                                           summary["corpora"].items()}))), flush=True)
    return records, labels, summary, template, digest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument("--fps", type=float, default=0., help="resample every clip to this rate (0: as stored)")
    parser.add_argument("--augment", action="store_true", help="speed, trim and face-dropout augmentation (training only)")
    parser.add_argument("--extra-manifest-key", help="a second sign manifest (ASLLVD) whose glosses map exactly to labels")
    parser.add_argument("--extra-manifest-sha256")
    parser.add_argument("--gloss-map-key", default="acquisition/2026-10-17-v1/derived/signbank-v1/asl-citizen-gloss-map.json")
    v3 = parser.add_argument_group("recogniser v3 (any of these turns it on; see the module text)")
    v3.add_argument("--corpus", action="append", help="KEY@SHA:MAPPING[:WEIGHT], repeated; mapping one of "
                    "label, asllex, signbank, upper")
    v3.add_argument("--primary-weight", type=float, default=V3_DEFAULTS["primary_weight"],
                    help="sampling weight of the primary manifest's training clips")
    v3.add_argument("--holdout-signers", action="append", help="CORPUS:ID[,ID...], repeated: every clip of these "
                    "signers is held out (holdout:<corpus>)")
    v3.add_argument("--exclude-signers", action="append", help="CORPUS:ID[,ID...], repeated: these signers are dropped")
    v3.add_argument("--whole-clip", action="append", help="CORPUS, repeated: its records' trim windows are removed")
    v3.add_argument("--labels-from-run", help="take the label space from this run's run.json (v3: sign-recognizer-v2)")
    v3.add_argument("--conform-template", help="KEY@SHA:CORPUS: conform every state to that corpus's first training "
                    "state's layouts")
    v3.add_argument("--recon-prefix", help="build_recon_states.py prefix (its index.json) of decoded training copies")
    v3.add_argument("--recon-index-sha256", help="the SHA-256 of that index.json (required with --recon-prefix)")
    v3.add_argument("--recon-p", type=float, default=V3_DEFAULTS["recon_p"],
                    help="probability that a training clip with a decoded copy is served as the copy")
    v3.add_argument("--recon-min-coverage", type=float, default=V3_DEFAULTS["recon_min_coverage"])
    v3.add_argument("--validity-dropout", type=float, default=V3_DEFAULTS["validity_dropout"],
                    help="probability of random arm/torso invalidation per training clip (with --augment)")
    args = parser.parse_args(argv)
    v3_on = is_v3(args)
    if v3_on and args.validity_dropout > 0 and not args.augment:
        raise SystemExit("--validity-dropout is an augmentation: it needs --augment")
    store = make_store(args.store)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    cache = ContentCache(Path(os.environ.get("SIGN_CACHE_DIR", tempfile.gettempdir())) / "sign-translator-cache",
                         args.cache_bytes)
    prefix = f"training-runs/{args.run}"
    template, summary, data_sha256 = None, None, None
    if v3_on:
        records, glosses, summary, template, data_sha256 = build_v3_data(store, args, cache, prefix)
        manifest = dict(records=records)
    else:
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "manifest.json"
            path.write_bytes(store.get(args.manifest_key) + b"\n")
            manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
        if store.exists(f"{prefix}/run.json"):
            glosses = json.loads(store.get(f"{prefix}/run.json"))["labels"]
        else:
            glosses = sorted({r["gloss"]["label"] for r in manifest["records"]
                              if r.get("partition") == "train" and (r.get("gloss") or {}).get("label")})
    labels = {g: i for i, g in enumerate(glosses)}
    if args.extra_manifest_key:
        manifest = with_extra_corpus(store, manifest, args.extra_manifest_key, args.extra_manifest_sha256,
                                     args.gloss_map_key, labels)
    kw = dict(labels=labels, cache=cache, fetch=store.get, min_hand_fraction=args.min_hand_fraction, fps=args.fps or None)
    if v3_on:
        kw.update(template=template, hand_fraction_fallback=True)
        augment = (functools.partial(worker_augment_v3, validity_dropout=args.validity_dropout) if args.augment
                   else None)
        train = GlossMotionDataset(manifest, partition="train", augment=augment,
                                   recon_p=args.recon_p if args.recon_prefix else 0., **kw)
    else:
        train = GlossMotionDataset(manifest, partition="train", augment=worker_augment if args.augment else None, **kw)
    keep = lambda d: [i for i, r in enumerate(d.records) if r["continuous"].get("seconds", 0.) <= args.max_seconds]  # noqa: E731
    train_ids = keep(train)
    if v3_on:
        val_sets = v3_validation_sets(manifest, kw, keep, args)
        val_ids = [i for name, sets in val_sets.items() if name.startswith("val:") for i in sets["real"][1]]
    else:
        val = GlossMotionDataset(manifest, partition="val", **kw)
        cross = GlossMotionDataset(manifest, partition="cross_val", **kw)
        val_ids = keep(val)
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
        if v3_on and state.get("data_sha256") != data_sha256:
            raise SystemExit("the newest checkpoint was trained on different recogniser v3 data")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        resumed_ema, step = state.get("ema"), state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed {checkpoints[-1]} at step {step}", flush=True)
    else:
        if v3_on:
            store.put(f"{prefix}/run.json", json.dumps(dict(
                kind=KIND, config=config.to_dict(), config_sha256=config.sha256(), labels=glosses,
                manifest_key=args.manifest_key, manifest_sha256=args.manifest_sha256, args=vars(args),
                recognizer_v3=True, data=summary, data_sha256=data_sha256, train_records=len(train_ids),
                val_records={name: len(s["real"][1]) for name, s in val_sets.items()},
                decoded_val_records={name: len(s["decoded"][1]) if s["decoded"] else 0 for name, s in val_sets.items()},
                selection="mean over corpora of val top-1, real and decoded clips"),
                indent=2, sort_keys=True, default=str).encode())
        else:
            recorded = {k: v for k, v in vars(args).items() if k not in V3_DEFAULTS}   # v2's run.json, unchanged
            store.put(f"{prefix}/run.json", json.dumps(dict(
                kind=KIND, config=config.to_dict(), config_sha256=config.sha256(), labels=glosses,
                manifest_key=args.manifest_key, manifest_sha256=args.manifest_sha256, args=recorded,
                train_records=len(train_ids), val_records=len(val_ids)), indent=2, sort_keys=True, default=str).encode())
    weights = [train.records[i]["_weight"] for i in train_ids] if v3_on else None
    if weights is not None and any(w != 1. for w in weights):
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ids), replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed + step))
        loader = DataLoader(Subset(train, train_ids), batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.workers, collate_fn=collate_gloss_motion,
                            persistent_workers=args.workers > 0, drop_last=len(train_ids) > args.batch_size)
    else:
        loader = DataLoader(Subset(train, train_ids), batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            collate_fn=collate_gloss_motion, generator=torch.Generator().manual_seed(args.seed + step),
                            persistent_workers=args.workers > 0, drop_last=len(train_ids) > args.batch_size)
    if v3_on:
        score_key, best_field = "select/score", "select_score"
    else:
        score_key, best_field = "val/top1", "val_top1"
        chosen = sorted(random.Random(1234).sample(val_ids, min(args.val_clips, len(val_ids))))
        val_batches = list(DataLoader(Subset(val, chosen), batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.workers, collate_fn=collate_gloss_motion))
        cross_ids = [i for i, r in enumerate(cross.records) if r["continuous"].get("seconds", 0.) <= args.max_seconds]
        cross_batches = list(DataLoader(Subset(cross, cross_ids), batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.workers, collate_fn=collate_gloss_motion)) if cross_ids else []
    ema = WeightEMA(model, args.ema_decay)
    if resumed_ema is not None:
        ema.shadow = {n: resumed_ema[n].to(device).clone() for n in ema.shadow}
        ema.updates = step
    best = json.loads(store.get(f"{prefix}/best.json")) if store.exists(f"{prefix}/best.json") else None
    stale, log, started = 0, [], time.monotonic()

    def validate() -> dict:
        live = ema.swap(model)
        if v3_on:
            metrics = v3_metrics(model, val_sets, device, args)
        else:
            metrics = evaluate(model, val_batches, device)
            if cross_batches:                        # held-out signers of the extra corpus (never used for selection)
                metrics.update({k.replace("val/", "cross/"): v for k, v in evaluate(model, cross_batches, device).items()})
        ema.restore(model, live)
        return metrics

    def checkpoint(metrics=None):
        nonlocal best
        buffer = io.BytesIO()
        torch.save(dict(kind=KIND, model=model.state_dict(), ema=ema.state_dict(model), optimizer=optimizer.state_dict(),
                        step=step, validation=metrics, torch_rng=torch.get_rng_state(), config=config.to_dict(),
                        config_sha256=config.sha256(), labels=glosses, manifest_sha256=args.manifest_sha256,
                        **({"data_sha256": data_sha256} if v3_on else {})), buffer)
        key = f"{prefix}/checkpoints/step-{step:08d}.pt"
        store.put(key, buffer.getvalue())
        store.put(f"{prefix}/log/step-{step:08d}.json", json.dumps(log, sort_keys=True).encode())
        if metrics and (best is None or metrics[score_key] > best[best_field]):
            best = dict(checkpoint_key=key, step=step, validation=metrics, **{best_field: metrics[score_key]})
            if v3_on:
                best["selection"] = "mean over corpora of val top-1, real and decoded clips"
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
                improved = best is None or metrics[score_key] > best[best_field]
                stale = 0 if improved else stale + 1
            log.append(entry)
            if step % 100 == 0 or step == 1 or metrics:
                print(json.dumps(entry), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps or (metrics and stale == 0):
                checkpoint(metrics)
            if metrics and stale >= args.patience:
                print(f"early stop at step {step}: no new best validation {'selection score' if v3_on else 'top-1'} "
                      f"in {stale} validations", flush=True)
                checkpoint(metrics)
                return 0
    return 0


def v3_validation_sets(manifest, kw, keep, args) -> dict:
    """{set name: dict(real=(dataset, ids), decoded=(dataset, ids) or None)} for every ``val:<corpus>`` and
    ``holdout:<corpus>`` partition: up to ``--val-clips`` clips each (seed 1234), the decoded set being those clips'
    decoded copies (validation clips have copies when the index covers them)."""
    names = sorted({r["partition"] for r in manifest["records"]
                    if r["partition"].startswith(("val:", "holdout:"))})
    sets = {}
    for name in names:
        real = GlossMotionDataset(manifest, partition=name, **kw)
        ids = keep(real)
        if not ids:
            continue
        chosen = sorted(random.Random(1234).sample(ids, min(args.val_clips, len(ids))))
        decoded = None
        if args.recon_prefix:
            dec = GlossMotionDataset(manifest, partition=name, decoded=True, **kw)
            position = {r["sign"]: i for i, r in enumerate(dec.records)}
            dec_ids = [position[real.records[i]["sign"]] for i in chosen if real.records[i]["sign"] in position]
            decoded = (dec, dec_ids) if dec_ids else None
        sets[name] = dict(real=(real, chosen), decoded=decoded)
    return sets


def v3_metrics(model, val_sets, device, args) -> dict:
    """Per set and form loss / top-1 / top-5; ``select/score``: the mean of the ``val:`` sets' real and decoded top-1
    (held-out signers are reported only)."""
    metrics, scores = {}, []
    for name, sets in val_sets.items():
        for form in ("real", "decoded"):
            if sets[form] is None:
                continue
            dataset, ids = sets[form]
            batches = DataLoader(Subset(dataset, ids), batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, collate_fn=collate_gloss_motion)
            out = evaluate(model, batches, device)
            for k, v in out.items():
                metrics[f"{name}/{form}/{k.split('/', 1)[1]}"] = v
            if name.startswith("val:"):
                scores.append(out["val/top1"])
    metrics["select/score"] = float(np.mean(scores))
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
