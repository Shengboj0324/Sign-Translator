"""The symmetric fine-tune: v1's parent fine-tuned on How2Sign's monocular states with v1's recipe, then read on D20 through both paths. Report only: this instrument gates and selects nothing, and no generated form or composition is read. It asks whether v1's preference for two-view follows its training path.

**How this will be read (stated before any read; the commit holding this text precedes the job).** On the marks-free pair, nearest two-view and nearest monocular, through the new instrument. s_two is its two-view gap over its monocular gap, each over its own blank; v1's mirror is its monocular share, 17.6% on likelihood and 33.1% on BLEU. Each metric reads alone, and they must agree:
- **(a) fit_shown:** s_two under 0.5, and monocular minus two-view has a lower bound above zero: it prefers monocular about as v1 prefers two-view, so preference follows the training path.
- **(b) content_gains_support:** monocular minus two-view has an upper bound under zero: trained on monocular, it still prefers two-view.
- **(c) unresolved:** anything else, including a monocular gap with a lower bound at or under zero, or the metrics disagreeing.
**Own-path peak, stated with the same care:** the new instrument on monocular against v1 on two-view, each on its training path's nearest form, each likelihood over its own blank, paired by clip. Thresholds are the path read's, 0.5 BLEU and 0.052 nats:
- **peak_near:** the difference at or above minus the threshold, with a lower bound at or above minus the threshold.
- **peak_below:** the difference under minus the threshold, with an upper bound under zero: an own-path peak well below v1's suggests lost content.
- Otherwise **unresolved**, and when the metrics disagree. A mirrored preference with ``peak_near`` shows fit; a mirrored preference with ``peak_below`` shows fit and still suggests the path carries less.
**What none of these can show.** Whether lost content is lost to people; (b) and ``peak_below`` are also what a harder fit would give, since monocular values are noisier and the recipe's budget and patience are v1's. (a) says an instrument can be tuned to this path, not that the path keeps its signing. Neither says anything about composition, whose word clips come from other corpora through the path. The case is a reading, not a rule: no step branches on it, and it starts no training and no decision.

**The recipe.** v1's, read from its pinned ``run.json``: the parent's step 76,000 EMA weights (sha ``f1ff9446...``), AdamW, learning rates 1e-4 and 3e-5, warmup 500, 30,000 steps at most, validation every 1,000, patience 6, selection by best validation BLEU over its fixed sample, batch 64, seed 0, augmentation on. ``trained`` refuses a run record that differs from v1's anywhere but ``run``, the two indexes and what they determine.

**What changes, exactly.**
1. How2Sign training and validation clips are read through ``derived/how2sign-video-estimate-v1`` instead of two-view: ``at_rate``, ``from_state``, ``normalise``, the FK builder's own steps. (Not conformed, as v1's weren't: the path read found conforming leaves v1's input byte-identical on all 500.) Same sentences, same order, same filter on shoulders.
2. **C20 is refused by exact record, though v1 read it.** v1's lists predate C20: its train list holds 455 C20 clips and its fixed validation sample 167. Those are dropped by exact sign; the list is derived from the manifest by ``build_round20_sets`` (``c20.json`` is never fetched), held to a pinned fingerprint and to reproducing D20 exactly. So training has 19,182 clips against v1's 19,637 (97.7%) and **selection reads 345 clips against v1's 512: not the same set, a strict subset.** Selection is noisier, and BLEU on these 345 does not compare with v1's stored validation BLEU.
3. ``--val-clips`` stays 512, and the validation series holds exactly those 345, in v1's order, so the trainer's fixed sample takes them all.
Nothing else: no c00 (none in v1's lists; refused if any appears), no test or held-out series (refused unread), no other corpus.

**Coverage, counted 2026-11-02 from index objects:** every kept clip has a monocular state: 19,182 of 19,182 train, 345 of 345 validation, no row lost. Under 90% of v1's 19,637 training clips, ``coverage`` and ``build`` say so and the job stops with status 4, before any training. Clips per path and per signer are in ``coverage.json`` and the report.

**Disclosed.** v1's list is ``full-v1``'s, so training holds 234 of D20's clips and the selection sample 88; every row is also split by whether a clip is in training. Augmentation is unseeded in the trainer, as it was for v1, so no run reproduces another exactly. ``transformers`` 4.57.6, the stored reads' version, not v1's own install. A resumed run resets patience, so a checkpoint without ``DONE`` refuses unless ``--allow-resume``, and the report says so.

**Forms** (D20; one process): ``mono__real_two_view``, ``mono__real_monocular``, ``mono__nearest_two_view``, ``mono__nearest_monocular``, ``mono__blank``, the path read's builders, reading a clip's reference and 4 others (the first column is the one-candidate read, the rest give a raw margin, report only). v1's are its stored forms by hash, not read again. **Controls:** v1's real and blank against the pinned report (``control``): a miss voids the read and stops it. The new instrument's selection sample must reproduce its ``best.json`` (``val-control``: BLEU within 0.05, loss within 0.01): a miss withholds the label. **Label withheld** in a smoke, on any alert, under 475 shared clips, or off the plan's v1 figures. The smoke: 8 c02 train clips through build, a 10-step train and every form, with no D20 state.

    python scripts/symmetric_ft.py recipe
    python scripts/symmetric_ft.py coverage --out-dir W
    python scripts/symmetric_ft.py build --out-dir W --workers 12
    python scripts/symmetric_ft.py train
    python scripts/symmetric_ft.py trained
    python scripts/symmetric_ft.py val-control --out-dir W/forms
    python scripts/symmetric_ft.py preflight --out-dir W --final-key K --void-prefix V
    python scripts/symmetric_ft.py read --forms control --out-dir W/forms --instrument-cache W/i
    python scripts/symmetric_ft.py control --out-dir W/forms --void-prefix V
    python scripts/symmetric_ft.py read --forms rest --resume --out-dir W/forms --instrument-cache W/i
    python scripts/symmetric_ft.py merge --out-dir W/forms --output-file W/report.json
    python scripts/symmetric_ft.py check --report W/report.json

The job is ``runpod/fleet/symmetric_ft.sh``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import parent_lib as Q  # noqa: E402
import path_parts as PP  # noqa: E402
import path_read as R  # noqa: E402
import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v1 as T  # noqa: E402
from signtranslator.eval import provenance, s0_metrics as M  # noqa: E402

PROBE = "symmetric-ft"
JOB_FILE = "runpod/fleet/symmetric_ft.sh"
RUN = Q.MONO_RUN
PREFIX = Q.MONO_PREFIX
INDEX_PREFIX = f"{PREFIX}/keypoints-mono"
C20_COUNT = 1000
C20_FINGERPRINT = "4b10200714f124c374398f2582587a79fe60137108f55616d8762694b134f894"
V1_TRAIN_CLIPS, V1_VAL_SAMPLE = 19637, 512
COVERAGE_FLOOR = .9
EXIT_COVERAGE = 4
SHARD_SIZE = 2000
MONITOR_SEED = 1234                                       # the trainer's fixed sample
SAME_RECIPE = dict(batch_size=64, checkpoint_every=2000, decoder_lr=3e-5, lr=1e-4, patience=6, seed=0, select_on="val", steps=30000, val_clips=512, val_every=1000, warmup=500, weight_decay=.05, ema_decay=.999, no_augment=False, monitor_index=None)
MAY_DIFFER_ARGS = ("run", "train_index", "val_index")
MAY_DIFFER_TOP = ("train_index", "val_index", "data_sha256", "train_clips", "val_clips", "args", "config")
SMOKE_OVERRIDES = dict(steps=10, val_every=10, checkpoint_every=10, val_clips=2, batch_size=2, warmup=2, workers=0)
SMOKE_TRAIN_CLIPS, SMOKE_VAL_CLIPS = 6, 2
BASE_FORMS = ("real_two_view", "real_monocular", "nearest_two_view", "nearest_monocular", "blank")
TWO, MONO = "nearest_two_view", "nearest_monocular"
MONOCULAR_NEEDED = ("real_monocular", "nearest_monocular")
TOKENIZER_NEEDED = ("nearest_two_view", "nearest_monocular")
CONTROL = ("v1__real_two_view", "v1__blank")
CANDIDATES = Q.CANDIDATES
BOOT = Q.BOOT
SHARE_MIRROR = .5
THRESHOLD = R.THRESHOLD
VAL_LIMITS = dict(bleu=.05, loss=.01)
PLAN = dict(two_bleu=4.621, two_likelihood=.2337, mono_bleu=2.653, mono_likelihood=.041, real_two_bleu=4.493, blank_bleu=1.68)     # docs/NEXT_PHASE_PLAN.md, Path read result and Parts read
PLAN_TOLERANCE = dict(bleu=.05, likelihood=.002)
LABEL_FIT, LABEL_CONTENT, LABEL_NEAR, LABEL_BELOW = "fit_shown", "content_gains_support", "peak_near", "peak_below"
PRESTATED = dict(
    a="**(a) fit_shown:** s_two under 0.5, and monocular minus two-view has a lower bound above zero: it prefers monocular about as v1 prefers two-view, so preference follows the training path.",
    b="**(b) content_gains_support:** monocular minus two-view has an upper bound under zero: trained on monocular, it still prefers two-view.",
    c="**(c) unresolved:** anything else, including a monocular gap with a lower bound at or under zero, or the metrics disagreeing.",
    near="**peak_near:** the difference at or above minus the threshold, with a lower bound at or above minus the threshold.",
    below="**peak_below:** the difference under minus the threshold, with an upper bound under zero: an own-path peak well below v1's suggests lost content.",
    none="**What none of these can show.** Whether lost content is lost to people;")
NOT_DECIDED = "report only: the header's readings are stated before the read; no rule, bar or verdict here gates anything"
DISCLOSURES = dict(
    c20="C20 is refused by exact record though v1 read it: training has 19,182 of v1's 19,637 clips and selection 345 of its 512, a strict subset. Selection is noisier, and its BLEU does not compare with v1's stored validation BLEU.",
    training="v1's lists are full-v1's: training holds 234 of D20's clips, the selection sample 88. Both instruments read those 234 on their own paths having trained on them; by_training splits every row.",
    budget="The recipe's budget and patience are v1's. Monocular values may be harder to fit, so a low reading is also what underfitting would give.",
    augmentation="The trainer's augmentation is unseeded, as for v1: no run reproduces another exactly.",
    transformers="transformers 4.57.6, the stored reads' version, not v1's own install.",
    conform="Monocular states are not conformed for training, as v1's two-view weren't; the D20 reads conform them as the path read did, and it found v1's input byte-identical either way.")


@functools.lru_cache(maxsize=1)
def code_digest() -> str:
    try:
        return provenance.code_digest(extra_files=(JOB_FILE,))
    except OSError as error:
        raise SystemExit(f"no code digest ({error}): a read that cannot name its code is void")


# ---- the recipe ------------------------------------------------------------------------------------------------------------------------

def load_v1_run(store) -> dict:
    raw = store.get(Q.V1_RUN_JSON)
    P.check_sha(raw, Q.V1_RUN_JSON_SHA, Q.V1_RUN_JSON)
    return json.loads(raw)


def check_recipe(run: dict) -> None:
    """v1's run record is the recipe the header states, from the parent's own checkpoint on How2Sign FK alone."""
    args = run["args"]
    wrong = [k for k, v in SAME_RECIPE.items() if args.get(k) != v]
    if args.get("init_key") != PP.UNFITTED_KEY or args.get("init_sha256") != PP.UNFITTED_SHA:
        wrong.append("init")
    if args.get("train_index") != [Q.FK_INDEX] or args.get("val_index") != Q.FK_INDEX:
        wrong.append("indexes")
    if wrong:
        raise SystemExit(f"v1's run record is not the recipe the header states: {wrong}")


def trainer_argv(v1_args: dict, *, index: str, run: str, overrides=None) -> list[str]:
    """The trainer's arguments: v1's own, every one, with ``run`` and the two indexes changed. ``overrides``: a smoke's."""
    args = dict(v1_args, run=run, train_index=[index], val_index=index, **(overrides or {}))
    out = []
    for key, value in sorted(args.items()):
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            out.append(flag)
        elif isinstance(value, list):
            for item in value:
                out += [flag, str(item)]
        else:
            out += [flag, str(value)]
    return out


def recipe_differences(new: dict, v1: dict) -> dict:
    """What a new run record holds other than v1's, beyond what the header lets change: {field: [v1's, new]}."""
    out = {}
    for key in set(new) | set(v1):
        if key not in MAY_DIFFER_TOP and new.get(key) != v1.get(key):
            out[key] = [v1.get(key), new.get(key)]
    for key in set(new["args"]) | set(v1["args"]):
        if key not in MAY_DIFFER_ARGS and new["args"].get(key) != v1["args"].get(key):
            out[f"args.{key}"] = [v1["args"].get(key), new["args"].get(key)]
    if new.get("config") != v1.get("config"):
        out["config"] = [v1.get("config"), new.get("config")]
    return out


def run_recipe(args) -> int:
    from train_text_to_motion import make_store
    run = load_v1_run(make_store(args.store))
    check_recipe(run)
    print(json.dumps(dict(recipe=run["args"], config_sha256=run["config_sha256"], v1_train_clips=run["train_clips"], v1_val_clips=run["val_clips"])), flush=True)
    return 0


# ---- what may be read ------------------------------------------------------------------------------------------------------------------

def c20_signs(store, manifest) -> frozenset:
    """C20's exact records, derived from the manifest as the list was (``c20.json`` is not fetched), held to the pinned fingerprint and to reproducing D20's stored signs. Any drift refuses."""
    from build_round20_sets import build
    d20, c20, _ = build(manifest)
    signs = sorted(r["sign"] for r in c20)
    if len(signs) != C20_COUNT or hashlib.sha256("\n".join(signs).encode()).hexdigest() != C20_FINGERPRINT:
        raise SystemExit("the derived C20 differs from the pinned fingerprint: nothing can be refused by it")
    if {r["sign"] for r in d20} != {r["sign"] for r in L.load_read_list(store, manifest)}:
        raise SystemExit("the derivation does not reproduce D20's stored signs")
    return frozenset(signs)


def check_training_record(sign: str, record: dict | None) -> None:
    """A clip of v1's lists may be read only as How2Sign train or val, never c00, test or held-out, whatever else it says."""
    if record is None:
        raise SystemExit(f"{sign}: not in the manifest")
    if record.get("signer_component") == "c00":
        raise SystemExit(f"{sign}: How2Sign signer 9 (c00) is never read")
    if record.get("corpus") != L.H2S or record.get("partition") not in ("train", "val"):
        raise SystemExit(f"{sign}: {record.get('corpus')} {record.get('partition')} is not a How2Sign train or val clip")


def fixed_sample(rows: list[dict], clips: int = V1_VAL_SAMPLE) -> list[dict]:
    """The trainer's fixed validation sample of ``rows``: the clips v1 selected on."""
    picks = sorted(random.Random(MONITOR_SEED).sample(range(len(rows)), min(clips, len(rows))))
    return [rows[i] for i in picks]


def planned_lists(store, manifest) -> dict:
    """{train, val}: v1's training list and its fixed selection sample, in its order, with C20 dropped by exact record and every kept clip checked; and what was refused."""
    c20 = c20_signs(store, manifest)
    wanted = {"train": Q.v1_series(store, "train"), "val": fixed_sample(Q.v1_series(store, "val"))}
    names = {r["clip_id"] for rows in wanted.values() for r in rows}
    records = {r["sign"]: r for r in manifest["records"] if r["sign"] in names}
    out = dict(refused_c20={})
    for series, rows in wanted.items():
        for r in rows:
            check_training_record(r["clip_id"], records.get(r["clip_id"]))
        out[series] = [r for r in rows if r["clip_id"] not in c20]
        out["refused_c20"][series] = len(rows) - len(out[series])
    out["records"] = records
    out["v1_counts"] = {series: len(rows) for series, rows in wanted.items()}
    return out


def match_rows(rows: list[dict], reports: dict) -> tuple[list[dict], list[dict]]:
    """(the clips with a monocular state, as {clip_id, text, partition, state_key, state_sha256}; the others, listed). A row naming another object refuses."""
    found, lost = [], []
    for r in rows:
        sign, row = r["clip_id"], reports.get(r["clip_id"])
        if row is None:
            lost.append(R.lost_clip(sign, "no_monocular_row"))
        elif not row.get("state_key") or not row.get("state_sha256"):
            lost.append(R.lost_clip(sign, "no_state_in_row", row.get("status")))
        elif row["state_key"] != R.state_key_of(sign):
            raise SystemExit(f"{sign}: its row names {row['state_key']}, not this clip's monocular state: refused unread")
        else:
            found.append(dict(clip_id=sign, text=r["text"], partition=r["partition"], state_key=row["state_key"], state_sha256=row["state_sha256"]))
    return found, lost


def coverage_of(plan: dict, found: dict, lost: dict) -> dict:
    """Clips per path, per series and signer. The share is of v1's whole training list, so a C20 clip dropped counts against it."""
    share = len(found["train"]) / plan["v1_counts"]["train"]
    by_signer = {series: dict(sorted(Counter(f"{plan['records'][r['clip_id']]['partition']}:{plan['records'][r['clip_id']]['signer_component']}" for r in found[series]).items())) for series in found}
    return dict(kind=f"{PROBE}-coverage", report_only=True, code_digest=code_digest(), two_view_clips=plan["v1_counts"], monocular_clips={s: len(found[s]) for s in found}, refused_c20=plan["refused_c20"], lost={s: dict(count=len(lost[s]), by_reason=dict(Counter(x["reason"] for x in lost[s])), clips=lost[s]) for s in lost}, by_signer=by_signer, train_share_of_v1=round(share, 4), floor=COVERAGE_FLOOR, enough=share >= COVERAGE_FLOOR)


def plan_and_match(store) -> tuple:
    manifest = L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA)
    plan = planned_lists(store, manifest)
    reports = R.monocular_rows(store, [r["clip_id"] for s in ("train", "val") for r in plan[s]])
    found, lost = {}, {}
    for series in ("train", "val"):
        found[series], lost[series] = match_rows(plan[series], reports)
    return plan, found, lost


def run_coverage(args) -> int:
    """Index objects only. Exit 4, stopping that part, under 90% of v1's training clips."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    plan, found, lost = plan_and_match(store)
    report = coverage_of(plan, found, lost)
    path = Path(args.out_dir) / "coverage.json"
    if path.exists():
        path.unlink()
    P.write_report_once(path, report)
    print(json.dumps({k: report[k] for k in ("two_view_clips", "monocular_clips", "refused_c20", "by_signer", "train_share_of_v1", "enough")}), flush=True)
    if not report["enough"]:
        print(f"ALERT only {report['train_share_of_v1']:.1%} of v1's {plan['v1_counts']['train']} training clips have monocular states, under {COVERAGE_FLOOR:.0%}: this part stops", flush=True)
        return EXIT_COVERAGE
    return 0


# ---- shards ----------------------------------------------------------------------------------------------------------------------------

_WORKER: dict = {}


def _init(store_spec: str) -> None:
    from train_text_to_motion import make_store
    _WORKER.update(store=make_store(store_spec))


def convert(item: dict) -> dict:
    """One clip as ``build_fk_keypoint_shards.convert``: its state at 24 fps, projected and normalised. A missing object is reported; anything else stops the job, there as a raised error here."""
    from evaluate_text_to_motion import at_rate, load_state
    from signtranslator.data.keypoints2d import from_state, normalise
    store = _WORKER["store"]
    if item["state_key"] != R.state_key_of(item["clip_id"]):
        raise ValueError(f"{item['clip_id']}: refused {item['state_key']}")
    if not store.exists(item["state_key"]):
        return dict(sign=item["clip_id"], lost="state_missing")
    state = at_rate(load_state(store, item["state_key"], item["state_sha256"]), L.FPS)
    points, valid = from_state(state)
    if not (valid[:, 0] & valid[:, 1]).any():
        return dict(sign=item["clip_id"], lost="no_shoulders")
    return dict(sign=item["clip_id"], points=normalise(points, valid, y_down=False), valid=valid)


def write_series(store, pool, items: list[dict], series: str, prefix: str, shard_size: int = SHARD_SIZE) -> tuple[list, list]:
    from build_uwb_keypoint_shards import _write_shard
    shards, pending, lost = [], [], []
    for item, result in zip(items, pool.imap(convert, items, chunksize=16)):
        if "lost" in result:
            lost.append(R.lost_clip(result["sign"], result["lost"]))
            continue
        sign = item["clip_id"]
        pending.append(dict(clip_id=sign, video_id=sign.split("_")[0], text=item["text"], partition=series, frames=len(result["points"]), points=result["points"], valid=result["valid"]))
        if len(pending) == shard_size:
            shards.append(_write_shard(store, f"{prefix}/{series}/shard-{len(shards):05d}.npz", pending))
            pending = []
    if pending:
        shards.append(_write_shard(store, f"{prefix}/{series}/shard-{len(shards):05d}.npz", pending))
    return shards, lost


def build_index(store, spec: str, found: dict, prefix: str, workers: int, extra: dict) -> dict:
    with Pool(workers, initializer=_init, initargs=(spec,)) as pool:
        parts, lost = [], {}
        for series in ("train", "val"):
            shards, lost[series] = write_series(store, pool, found[series], series, prefix)
            parts.append(dict(partition=series, shards=shards))
    kept = {p["partition"]: sum(s["clips"] for s in p["shards"]) for p in parts}
    index = dict(corpus_id="how2sign-fk-keypoints-mono-v1", source_fps=L.FPS, layout="signtranslator.data.keypoints2d (48 points, hands local)", projection="canonical_skeleton FK, orthographic front; monocular states, derived/how2sign-video-estimate-v1", counts=kept, lost=lost, parts=parts, **extra)
    return index


def run_build(args) -> int:
    """The monocular shards, built once: the index goes last, so an index that stands is a finished build. Smoke: c02 train clips only, the last two named val for the trainer. Exit 4 under the floor."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    key = f"{args.prefix.rstrip('/')}/index.json"
    if store.exists(key):
        print(json.dumps(dict(index=key, stored=True, sha256=hashlib.sha256(store.get(key)).hexdigest())), flush=True)
        return 0
    if args.smoke:
        manifest = L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA)
        records = L.smoke_records(manifest, SMOKE_TRAIN_CLIPS + SMOKE_VAL_CLIPS)
        L.check_train_records(records, ("c02",))
        rows = [dict(clip_id=r["sign"], text=r["text"], partition="train") for r in records]
        found, lost = match_rows(rows, R.monocular_rows(store, [r["clip_id"] for r in rows]))
        for r in found:
            check_training_record(r["clip_id"], next(x for x in records if x["sign"] == r["clip_id"]))
        series = dict(train=found[:SMOKE_TRAIN_CLIPS], val=found[SMOKE_TRAIN_CLIPS:])
        extra = dict(smoke=True, list="smoke:c02-train", lost_at_match=lost)
        needed = None
    else:
        plan, series, lost = plan_and_match(store)
        extra = dict(smoke=False, two_view_clips=plan["v1_counts"], refused_c20=plan["refused_c20"], lost_at_match=lost)
        needed = plan["v1_counts"]["train"]
    index = build_index(store, args.store, series, args.prefix.rstrip("/"), args.workers, extra)
    if needed and index["counts"]["train"] / needed < COVERAGE_FLOOR:
        print(f"ALERT only {index['counts']['train']} of v1's {needed} training clips built, under {COVERAGE_FLOOR:.0%}: this part stops; no index is stored", flush=True)
        return EXIT_COVERAGE
    raw = json.dumps(index, indent=1, sort_keys=True).encode()
    store.put(key, raw)
    print(json.dumps(dict(index=key, stored=False, sha256=hashlib.sha256(raw).hexdigest(), counts=index["counts"], lost={k: len(v) for k, v in index["lost"].items()})), flush=True)
    return 0


def check_index(index: dict) -> None:
    """A training index holds v1's train and val series alone, and on a full build none of C20's clips is possible because the plan dropped them; a test or held-out series refuses."""
    names = [p["partition"] for p in index["parts"]]
    if names != ["train", "val"]:
        raise SystemExit(f"the index holds {names}, not train and val")


def run_train(args) -> int:
    """The trainer, with v1's arguments and only the header's changes. A finished run (DONE) is not re-entered; checkpoints without it refuse unless ``--allow-resume``, since a resumed trainer resets its patience."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    prefix = f"training-runs/{args.run}"
    if store.exists(f"{prefix}/DONE"):
        print(json.dumps(dict(train="already done", run=args.run)), flush=True)
        return 0
    if store.list(f"{prefix}/checkpoints/") and not args.allow_resume:
        raise SystemExit(f"{prefix} has checkpoints and no DONE: a resumed trainer resets its patience. Give --allow-resume to accept that; the report says so")
    if args.allow_resume:
        store.put(f"{prefix}/resumed-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.json", json.dumps(dict(checkpoints=store.list(f"{prefix}/checkpoints/"))).encode())
    index_key = f"{args.index_prefix.rstrip('/')}/index.json"
    if not store.exists(index_key):
        raise SystemExit(f"{index_key}: no index; build first")
    check_index(json.loads(store.get(index_key)))
    v1 = load_v1_run(store)
    check_recipe(v1)
    overrides = dict(SMOKE_OVERRIDES) if args.smoke else {}
    argv = trainer_argv(dict(v1["args"], store=args.store), index=index_key, run=args.run, overrides=overrides)
    done = subprocess.run([sys.executable, str(ROOT / "scripts/train_keypoints_to_text_t5.py"), *argv], cwd=ROOT)
    if done.returncode:
        raise SystemExit(f"the trainer stopped with {done.returncode}")
    store.put(f"{prefix}/DONE", b"done")
    print(json.dumps(dict(train="done", run=args.run)), flush=True)
    return 0


def run_trained(args) -> int:
    """The run record against v1's, the best checkpoint's hash, written once as ``trained.json``. A record that differs from v1's beyond the header's changes refuses."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    prefix = f"training-runs/{args.run}"
    record_key = f"{prefix}/trained.json"
    if store.exists(record_key):
        print(json.dumps(dict(trained=json.loads(store.get(record_key)), stored=True)), flush=True)
        return 0
    for needed in (f"{prefix}/DONE", f"{prefix}/best.json", f"{prefix}/run.json"):
        if not store.exists(needed):
            raise SystemExit(f"{needed} is missing: the run has not finished")
    new, v1 = json.loads(store.get(f"{prefix}/run.json")), load_v1_run(store)
    expected_index = f"{args.index_prefix.rstrip('/')}/index.json"
    if new["args"].get("train_index") != [expected_index] or new["args"].get("val_index") != expected_index:
        raise SystemExit(f"{prefix}/run.json trained on {new['args'].get('train_index')}, not {expected_index}")
    differs = recipe_differences(new, v1)
    if not args.smoke and differs:
        raise SystemExit(f"{prefix}/run.json differs from v1's recipe: {differs}")
    best = json.loads(store.get(f"{prefix}/best.json"))
    if not str(best.get("checkpoint_key", "")).startswith(f"{prefix}/checkpoints/"):
        raise SystemExit(f"{prefix}/best.json names {best.get('checkpoint_key')}")
    record = dict(kind=f"{PROBE}-trained", report_only=True, checkpoint_key=best["checkpoint_key"], sha256=hashlib.sha256(store.get(best["checkpoint_key"])).hexdigest(), step=best["step"], selected_on=best["selected_on"], run_sha256=hashlib.sha256(store.get(f"{prefix}/run.json")).hexdigest(), index_sha256=hashlib.sha256(store.get(expected_index)).hexdigest(), train_clips=new["train_clips"], val_clips=new["val_clips"], recipe_differences=differs, resumed=sorted(k for k in store.list(f"{prefix}/") if "/resumed-" in k), smoke=bool(args.smoke), code_digest=code_digest())
    store.put(record_key, json.dumps(record, indent=1, sort_keys=True).encode())
    print(json.dumps(dict(trained={k: record[k] for k in ("checkpoint_key", "sha256", "step", "train_clips", "val_clips")})), flush=True)
    return 0


def run_val_control(args) -> int:
    """The new instrument's selection sample against its best.json; a miss is recorded, and withholds the label at merge."""
    from torch.utils.data import DataLoader, Subset
    from signtranslator.data.keypoint_shards import KeypointShardDataset, collate_keypoints
    from train_keypoints_to_text_t5 import evaluate
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    prefix = f"training-runs/{args.run}"
    index_key = f"{args.index_prefix.rstrip('/')}/index.json"
    best = json.loads(store.get(f"{prefix}/best.json"))
    reader = Q.load_mono(store, args, prefix)
    dataset = KeypointShardDataset(store, [index_key], partition="val", target_fps=L.FPS, max_seconds=20.)
    clips = sorted(random.Random(MONITOR_SEED).sample(range(len(dataset)), min(args.val_clips, len(dataset))))
    metrics = evaluate(reader.model, list(DataLoader(Subset(dataset, clips), batch_size=args.batch_size, collate_fn=collate_keypoints)), reader.device, beam=1, prefix="val")
    stored = dict(bleu=best["validation"]["val/bleu"], loss=best["validation"]["val/loss"])
    misses = [why for why, bad in (("BLEU", abs(metrics["val/bleu"] - stored["bleu"]) > VAL_LIMITS["bleu"]), ("loss", abs(metrics["val/loss"] - stored["loss"]) > VAL_LIMITS["loss"])) if bad]
    result = dict(kind=f"{PROBE}-val-control", report_only=True, code_digest=code_digest(), clips=len(clips), bleu=round(metrics["val/bleu"], 4), loss=round(metrics["val/loss"], 4), stored=stored, misses=misses, reproduced=not misses, limits=VAL_LIMITS)
    P.write_report_once(Path(args.out_dir) / "val_control.json", result)
    print(json.dumps({k: result[k] for k in ("clips", "bleu", "loss", "stored", "misses")}), flush=True)
    if misses:
        print(f"ALERT the new instrument does not reproduce its selection read: {misses}; the label is withheld", flush=True)
    return 0


# ---- reads -----------------------------------------------------------------------------------------------------------------------------

def run_preflight(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    if store.exists(args.final_key):
        raise SystemExit(f"{args.final_key} exists: this read is finished (write-once)")
    P.check_void(store, args, code_digest(), Path(args.out_dir))
    print(json.dumps(dict(preflight="clear", code_digest=code_digest())), flush=True)
    return 0


def resolve_forms(spec: str) -> list[str]:
    if spec == "control":
        return list(CONTROL)
    if spec == "rest":
        return [f"mono__{f}" for f in BASE_FORMS]
    raise SystemExit(f"--forms {spec}: control or rest")


def build_form(form: str, refs, clips, tokenizer) -> tuple[list[int], list, list]:
    """(record indices, states, the states motion is masked by): the path read's forms."""
    from evaluate_text_to_motion import blanked
    every = list(range(len(refs)))
    kept = [c.index for c in clips]
    if form == "real_two_view":
        return every, list(refs), list(refs)
    if form == "blank":
        return every, blanked(refs), list(refs)
    if form == TWO:
        return every, [T.cap_form("v0_nearest_fill", s, tokenizer) for s in refs], list(refs)
    if form == MONO:
        return kept, [T.cap_form("v0_nearest_fill", c.conformed, tokenizer) for c in clips], [c.conformed for c in clips]
    if form == "real_monocular":
        return kept, [c.conformed for c in clips], [c.conformed for c in clips]
    raise SystemExit(f"unknown form {form}")


def run_read(args) -> int:
    import torch
    from train_text_to_motion import make_store
    P.check_pins()
    R.check_smoke(args.smoke_train)
    torch.set_num_threads(args.threads)
    names = resolve_forms(args.forms)
    out_dir = Path(args.out_dir)
    stored = [n for n in names if (out_dir / f"form-{n}.json").exists()]
    if stored and not args.resume:
        raise SystemExit(f"{out_dir / f'form-{stored[0]}.json'} exists (write-once)")
    names = [n for n in names if n not in stored]
    if not names:
        print(json.dumps(dict(read=0, stored=len(stored))), flush=True)
        return 0
    store = make_store(args.store)
    records, declared, described = R.open_list(store, args)
    digest, place = code_digest(), P.environment()
    if declared and args.forms == "rest":
        R.passed_control(out_dir, digest, place)
    texts, signs, groups = [r["text"] for r in records], [r["sign"] for r in records], [r["signer_component"] for r in records]
    refs = L.load_states(store, records)
    forms = [n.split("__")[1] for n in names]
    clips, lost = [], []
    if any(f in MONOCULAR_NEEDED for f in forms):
        clips, lost, template = R.open_monocular(store, records, refs, declared)
        path = out_dir / "monocular.json"
        if not path.exists():
            P.write_report_once(path, dict(kind=f"{PROBE}-monocular", report_only=True, declared_read=declared, code_digest=digest, clips_loaded=len(clips), lost=lost, conform=R.conform_tally(clips, template)))
    tokenizer = L.load_tokenizer(store) if any(f in TOKENIZER_NEEDED for f in forms) else None
    candidates = M.retrieval_candidates(len(texts), CANDIDATES, seed=0)
    readers: dict = {}
    for name, form in zip(names, forms):
        instrument = name.split("__")[0]
        if instrument not in readers:
            readers[instrument] = P.load_reader(store, args) if instrument == "v1" else Q.load_mono(store, args, f"training-runs/{args.run}")
        reader = readers[instrument]
        started = time.time()
        kept, states, masks = build_form(form, refs, clips, tokenizer)
        built = time.time()
        raw = reader.read(states, texts, candidates[kept], group=args.nll_group)
        validity = [R.part_validity(s) for s in states]
        report = dict(kind=f"{PROBE}-form", report_only=True, form=name, declared_read=declared, list=described, code_digest=digest, environment=place, instrument=reader.info, signs=[signs[i] for i in kept], groups=[groups[i] for i in kept], references=[texts[i] for i in kept], lost=lost if form in MONOCULAR_NEEDED else [], hypotheses=raw["hypotheses"], nll=np.round(raw["nll"], 5).tolist(),
                      valid_frame_share=[R.valid_frame_share(s) for s in states], part_validity={part: [v[part] for v in validity] for part in R.PARTS}, motion=L.motion_statistics(states, masks), build_seconds=round(built - started, 2), read_seconds=round(time.time() - built, 2), peak_rss_bytes=P.peak_rss_bytes(), gpu_peak_reserved_bytes=P.gpu_peak_reserved_bytes())
        P.write_report_once(out_dir / f"form-{name}.json", report)
        print(json.dumps({name: dict(clips=len(kept), read_seconds=report["read_seconds"], peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2))}), flush=True)
        del states, raw
    return 0


def run_control(args) -> int:
    """Exit 3, voiding the read, unless v1's real and blank reproduce the pinned report. A smoke checks the pin and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    out = Path(args.out_dir)
    store = make_store(args.store)
    raw = store.get(P.CANARY_REPORT)
    P.check_sha(raw, P.CANARY_REPORT_SHA, P.CANARY_REPORT)
    canary_report = json.loads(raw)
    read = {n: json.loads((out / f"form-{n}.json").read_bytes()) for n in CONTROL}
    if not read["v1__real_two_view"]["declared_read"]:
        print(json.dumps(dict(control="not applicable: a smoke reads train clips; the pin is checked and nothing compared")), flush=True)
        return 0
    digest, place = P.check_same_run(read, "the control's forms")
    try:
        canary = P.canary_read(dict(real=read["v1__real_two_view"], blank=read["v1__blank"]), canary_report)
    except SystemExit as error:
        canary = dict(reads=None, misses=[str(error)], reproduced=False)
    result = dict(canary, code_digest=digest, environment=place)
    path = out / "control.json"
    if not path.exists():
        P.write_report_once(path, result)
    print(json.dumps(result), flush=True)
    if result["reproduced"]:
        return 0
    key = f"{args.void_prefix.rstrip('/')}/{digest}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-void", code_digest=digest, environment=place, control=result), indent=1, sort_keys=True).encode())
    print(f"ALERT v1's real and blank do not reproduce the stored report: {result['misses']}. The read is void: no other form follows, no retry; a rerun needs new code and a recorded cause", flush=True)
    return 3


# ---- reading ---------------------------------------------------------------------------------------------------------------------------

def measures(reads: dict, blank: str, idx: np.ndarray) -> tuple[dict, dict]:
    """(point, draws) per metric and form, against this instrument's own blank."""
    base = np.asarray(reads[blank]["nll"])[:, 0]
    point, draws = dict(bleu={}, likelihood={}), dict(bleu={}, likelihood={})
    for form, read in reads.items():
        stats = M.bleu_statistics(read["hypotheses"], read["references"])
        dnll = M.delta_nll(base, np.asarray(read["nll"])[:, 0])
        point["bleu"][form], draws["bleu"][form] = float(M.bleu_from_statistics(stats.sum(0))), M.bleu_from_statistics(stats[idx].sum(1))
        point["likelihood"][form], draws["likelihood"][form] = float(dnll.mean()), dnll[idx].mean(1)
    return point, draws


def mirror_case(gap: dict, share: dict | None, difference: dict) -> str:
    """One metric, the header's rule. ``gap``: monocular over its blank; ``share``: two-view's gap over it; ``difference``: monocular minus two-view."""
    if gap["ci95"][0] <= 0 or share is None:
        return "unresolved"
    if difference["ci95"][1] < 0:
        return LABEL_CONTENT
    return LABEL_FIT if share["point"] < SHARE_MIRROR and difference["ci95"][0] > 0 else "unresolved"


def peak_case(difference: dict, threshold: float) -> str:
    """One metric, the header's rule on the own-path difference."""
    if difference["point"] < -threshold:
        return LABEL_BELOW if difference["ci95"][1] < 0 else "unresolved"
    return LABEL_NEAR if difference["ci95"][0] >= -threshold else "unresolved"


def preference_block(reads: dict, blank: str, idx, two: str = TWO, mono: str = MONO) -> dict:
    """The pair ``two`` and ``mono`` on one instrument, each metric alone, under one bootstrap."""
    point, draws = measures(reads, blank, idx)
    out = {}
    for metric in ("bleu", "likelihood"):
        p, d = point[metric], draws[metric]
        gap_mono, gap_two = dict(point=round(p[mono] - p[blank], 5), ci95=P._quantiles(d[mono] - d[blank])), dict(point=round(p[two] - p[blank], 5), ci95=P._quantiles(d[two] - d[blank]))
        share_two = PP.ratio(p, d, (two, blank), (mono, blank))
        share_mono = PP.ratio(p, d, (mono, blank), (two, blank))
        difference = PP.difference(p, d, mono, two)
        out[metric] = dict(two_view=round(p[two], 5), monocular=round(p[mono], 5), blank=round(p[blank], 5), gap_monocular=gap_mono, gap_two_view=gap_two, two_view_share_of_monocular_gap=share_two, monocular_share_of_two_view_gap=share_mono, monocular_minus_two_view=difference, case=mirror_case(gap_mono, share_two, difference))
    out["case"] = PP.joint([out[m]["case"] for m in ("bleu", "likelihood")])
    return out


def peak_block(mono: dict, mono_blank: dict, v1: dict, v1_blank: dict, idx) -> dict:
    """The own-path peak: the new instrument on monocular against v1 on two-view, paired by clip, each likelihood over its own blank."""
    refs = mono["references"]
    a_stats, b_stats = M.bleu_statistics(mono["hypotheses"], refs), M.bleu_statistics(v1["hypotheses"], refs)
    a, b = float(M.bleu_from_statistics(a_stats.sum(0))), float(M.bleu_from_statistics(b_stats.sum(0)))
    diff_bleu = dict(point=round(a - b, 5), ci95=P._quantiles(M.bleu_from_statistics(a_stats[idx].sum(1)) - M.bleu_from_statistics(b_stats[idx].sum(1))))
    la, lb = M.delta_nll(np.asarray(mono_blank["nll"])[:, 0], np.asarray(mono["nll"])[:, 0]), M.delta_nll(np.asarray(v1_blank["nll"])[:, 0], np.asarray(v1["nll"])[:, 0])
    diff_ll = dict(point=round(float((la - lb).mean()), 5), ci95=P._quantiles((la - lb)[idx].mean(1)))
    out = dict(bleu=dict(new_on_monocular=round(a, 5), v1_on_two_view=round(b, 5), difference=diff_bleu, threshold=THRESHOLD["bleu"], case=peak_case(diff_bleu, THRESHOLD["bleu"])), likelihood=dict(new_on_monocular=round(float(la.mean()), 5), v1_on_two_view=round(float(lb.mean()), 5), difference=diff_ll, threshold=THRESHOLD["likelihood"], case=peak_case(diff_ll, THRESHOLD["likelihood"])))
    out["case"] = PP.joint([out[m]["case"] for m in ("bleu", "likelihood")])
    return out


def restricted(read: dict, signs) -> dict:
    where = {s: i for i, s in enumerate(read["signs"])}
    pick = [where[s] for s in signs]
    return dict(read, signs=list(signs), references=[read["references"][i] for i in pick], hypotheses=[read["hypotheses"][i] for i in pick], nll=[read["nll"][i] for i in pick])


def plan_misses(v1: dict) -> list[str]:
    """v1's stored forms against the plan's figures, on all their clips."""
    out = []
    base = np.asarray(v1["blank"]["nll"])[:, 0]
    for form, bleu_key, ll_key in ((TWO, "two_bleu", "two_likelihood"), (MONO, "mono_bleu", "mono_likelihood")):
        here = M.bleu_ci_fast(v1[form]["hypotheses"], v1[form]["references"])[0]
        dnll = float(M.delta_nll(base, np.asarray(v1[form]["nll"])[:, 0]).mean())
        if abs(here - PLAN[bleu_key]) > PLAN_TOLERANCE["bleu"]:
            out.append(f"v1's stored {form} reads BLEU {here}, not the plan's {PLAN[bleu_key]}")
        if abs(dnll - PLAN[ll_key]) > PLAN_TOLERANCE["likelihood"]:
            out.append(f"v1's stored {form} reads likelihood {dnll:.4f}, not the plan's {PLAN[ll_key]}")
    return out


def margin_row(reads: dict, idx) -> dict:
    """Report only, the new instrument: raw margins, mismatched minus own, on both pairs; no still is read here, so duration stays in them."""
    margin = {f: Q.margins(r) for f, r in reads.items()}
    return {f: dict(point=round(float(v.mean()), 5), ci95=P._quantiles(v[idx].mean(1))) for f, v in margin.items()} | dict(monocular_minus_two_view=dict(point=round(float((margin[MONO] - margin[TWO]).mean()), 5), ci95=P._quantiles((margin[MONO] - margin[TWO])[idx].mean(1))))


def load_reads(out_dir: Path) -> dict:
    reads = {}
    for form in BASE_FORMS:
        path = out_dir / f"form-mono__{form}.json"
        if not path.exists():
            raise SystemExit(f"{path} is missing")
        read = json.loads(path.read_bytes())
        if read.get("form") != f"mono__{form}":
            raise SystemExit(f"{path} holds {read.get('form')}")
        reads[form] = read
    return reads


def run_merge(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    L.refuse_existing(store, args.output_key, args.output_file)
    out_dir = Path(args.out_dir)
    all_reads = load_reads(out_dir)
    digest, place = P.check_same_run(all_reads, "the forms")
    if digest != code_digest():
        raise SystemExit(f"the forms are of code {digest}, not this code {code_digest()}: a stale read")
    signatures = {(r["declared_read"], r["list"]["list"]) for r in all_reads.values()}
    if len(signatures) != 1:
        raise SystemExit("the forms read another list, or a smoke mixed with a declared read")
    first = all_reads["blank"]
    declared = first["declared_read"]
    prefix = f"training-runs/{args.run}"
    if not store.exists(f"{prefix}/trained.json"):
        raise SystemExit(f"{prefix}/trained.json is missing")
    trained = json.loads(store.get(f"{prefix}/trained.json"))
    if {r["instrument"]["sha256"] for r in all_reads.values()} != {trained["sha256"]}:
        raise SystemExit("a form was not read by the trained checkpoint")
    alerts = []
    if trained.get("recipe_differences") and declared:
        alerts.append(f"the run differs from v1's recipe: {sorted(trained['recipe_differences'])}")
    control = R.passed_control(out_dir, digest, place) if declared else None
    shared = R.shared_signs(dict(all_reads, real_two_view=all_reads["real_two_view"]))
    reads = {f: restricted(r, shared) for f, r in all_reads.items()}
    n = len(shared)
    if declared and n < R.SHARED_FLOOR:
        alerts.append(f"{n} shared clips, under {R.SHARED_FLOOR}")
    val_path = out_dir / "val_control.json"
    val = json.loads(val_path.read_bytes()) if val_path.exists() else None
    if val is None or val.get("code_digest") != digest:
        alerts.append("no selection control of this code")
    elif val["misses"]:
        alerts.append(f"the new instrument missed its selection read: {val['misses']}")
    idx = np.random.default_rng(0).integers(0, n, (BOOT, n))
    marks_free = preference_block(reads, "blank", idx)
    marked = preference_block(reads, "blank", idx, "real_two_view", "real_monocular")
    v1 = peak = v1_block = None
    training = Q.v1_training_signs(store)
    masks = Q.subset_masks(shared, training)
    by_training = {}
    if declared:
        stored = {f: Q.stored_form(store, Q.STORED_V1, f) for f in (TWO, MONO, "real_two_view", "blank")}
        alerts += plan_misses(stored)
        if any(s not in set(stored[TWO]["signs"]) for s in shared):
            raise SystemExit("a shared clip is not in v1's stored forms")
        v1 = {f: restricted(r, shared) for f, r in stored.items()}
        v1_block = preference_block({f: v1[f] for f in (TWO, MONO, "blank")}, "blank", idx)
        peak = peak_block(reads[MONO], reads["blank"], v1[TWO], v1["blank"], idx)
    for name, mask in masks.items():
        if mask.sum() < 2:
            by_training[name] = None
            continue
        signs = [s for s, m in zip(shared, mask) if m]
        here = {f: restricted(r, signs) for f, r in all_reads.items() if f in (TWO, MONO, "blank")}
        sub = np.random.default_rng(0).integers(0, len(signs), (BOOT, len(signs)))
        row = dict(clips=len(signs), new_instrument=preference_block(here, "blank", sub))
        if v1 is not None:
            there = {f: restricted(v1[f], signs) for f in (TWO, MONO, "blank")}
            row["v1"] = preference_block(there, "blank", sub)
            row["own_path_peak"] = peak_block(here[MONO], here["blank"], there[TWO], there["blank"], sub)
        by_training[name] = row
    held = ([] if declared else ["a smoke carries no reading"]) + alerts
    report = dict(kind=f"{PROBE}-report", report_only=True, note=NOT_DECIDED, declared_read=declared, smoke=None if declared else "SMOKE on c02 train clips and a 10-step instrument: these numbers are plumbing, not a read", list=first["list"], code_digest=digest, environment=place, prestated=PRESTATED, disclosures=DISCLOSURES, alerts=alerts, withheld=held, instrument=first["instrument"], trained=trained,
                  clips=dict(list=len(all_reads["real_two_view"]["signs"]), shared=n, floor=R.SHARED_FLOOR, outside_shared=R.lost_rows(all_reads, shared), in_v1_training=int(masks["in_v1_training"].sum()), outside_v1_training=int(masks["outside_v1_training"].sum())),
                  preference=dict(marks_free=marks_free, marked_secondary=marked, label=None if held else marks_free["case"]), own_path_peak=None if peak is None else dict(peak, label=None if held else peak["case"]), v1=v1_block, by_training=by_training, raw_margin=margin_row({f: reads[f] for f in (TWO, MONO, "real_two_view", "real_monocular", "blank")}, idx),
                  forms={f: dict(bleu=M.bleu_ci_fast(r["hypotheses"], r["references"])[0], likelihood_minus_blank=round(float(M.delta_nll(np.asarray(reads["blank"]["nll"])[:, 0], np.asarray(r["nll"])[:, 0]).mean()), 5), valid_frame_share=round(float(np.mean(r["valid_frame_share"])), 4)) for f, r in reads.items()},
                  checks=dict(control=control, val_control=val, blank_identical_to_real=round(float(np.mean([a == b for a, b in zip(reads["blank"]["hypotheses"], reads["real_two_view"]["hypotheses"])])), 4)), rerun=json.loads((out_dir / "rerun.json").read_bytes()) if (out_dir / "rerun.json").exists() else None, resources={f: {k: r[k] for k in ("peak_rss_bytes", "gpu_peak_reserved_bytes", "build_seconds", "read_seconds")} for f, r in all_reads.items()})
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(merged=args.output_key or args.output_file, report_only=True, declared_read=declared, shared=n, forms=report["forms"], preference={m: {k: marks_free[m][k] for k in ("two_view_share_of_monocular_gap", "monocular_minus_two_view", "case")} for m in ("bleu", "likelihood")}, label=report["preference"]["label"], own_path_peak=report["own_path_peak"], withheld=held, alerts=alerts)), flush=True)
    return 0


def run_check(args) -> int:
    """Exit 3 on broken plumbing: not report_only, blank reading as real, a declared read without its passed control, any alert, or a label beside one. It judges no form."""
    report = json.loads(Path(args.report).read_text())
    problems = list(report["alerts"])
    if report.get("report_only") is not True:
        problems.append("the report does not say report_only")
    if report["checks"]["blank_identical_to_real"] >= 1:
        problems.append("blank reads exactly as real")
    if report.get("declared_read") and not (report["checks"].get("control") or {}).get("reproduced"):
        problems.append("a declared read without its passed control")
    if report["withheld"] and (report["preference"].get("label") is not None or (report.get("own_path_peak") or {}).get("label") is not None):
        problems.append("a report holding a label beside an alert")
    print(json.dumps(dict(problems=problems)), flush=True)
    for line in problems:
        print(f"ALERT {line}", flush=True)
    return 3 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    def of_run(q):
        q.add_argument("--store", default="r2")
        q.add_argument("--run", default=RUN)
        q.add_argument("--index-prefix", default=INDEX_PREFIX)
    recipe = sub.add_parser("recipe", help="the job's first step: v1's recipe from its pinned run record")
    recipe.add_argument("--store", default="r2")
    coverage = sub.add_parser("coverage", help="index objects only: clips per path; exit 4 under 90%")
    coverage.add_argument("--store", default="r2")
    coverage.add_argument("--out-dir", required=True)
    build = sub.add_parser("build", help="the monocular shards, once")
    build.add_argument("--store", default="r2")
    build.add_argument("--prefix", default=INDEX_PREFIX)
    build.add_argument("--workers", type=int, default=12)
    build.add_argument("--smoke", action="store_true")
    train = sub.add_parser("train")
    of_run(train)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--allow-resume", action="store_true", help="a resumed trainer resets its patience; the report says so")
    trained = sub.add_parser("trained", help="the run against v1's recipe; the best checkpoint's hash, once")
    of_run(trained)
    trained.add_argument("--smoke", action="store_true")
    val = sub.add_parser("val-control", help="the selection sample against best.json")
    of_run(val)
    val.add_argument("--out-dir", required=True)
    val.add_argument("--val-clips", type=int, default=V1_VAL_SAMPLE)
    val.add_argument("--batch-size", type=int, default=64)
    val.add_argument("--instrument-cache")
    val.add_argument("--device", default="cpu")
    pre = sub.add_parser("preflight")
    pre.add_argument("--store", default="r2")
    pre.add_argument("--out-dir", required=True)
    pre.add_argument("--final-key", required=True)
    pre.add_argument("--void-prefix", required=True)
    pre.add_argument("--rerun-cause")
    read = sub.add_parser("read")
    read.add_argument("--store", default="r2")
    read.add_argument("--run", default=RUN)
    read.add_argument("--out-dir", required=True)
    read.add_argument("--forms", required=True, help="control or rest")
    read.add_argument("--resume", action="store_true")
    read.add_argument("--nll-group", type=int, default=4)
    read.add_argument("--instrument-cache")
    read.add_argument("--smoke-train", type=int, default=0, help=f"this many c02 train clips, at most {R.SMOKE_CLIPS}, not D20")
    read.add_argument("--smoke-longest", type=int, default=0)
    read.add_argument("--smoke-never-tracked", type=int, default=0)
    read.add_argument("--device", default="cpu")
    read.add_argument("--threads", type=int, default=1)
    control = sub.add_parser("control")
    control.add_argument("--store", default="r2")
    control.add_argument("--out-dir", required=True)
    control.add_argument("--void-prefix", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--store", default="r2")
    merge.add_argument("--run", default=RUN)
    merge.add_argument("--out-dir", required=True)
    merge.add_argument("--output-key")
    merge.add_argument("--output-file")
    check = sub.add_parser("check")
    check.add_argument("--report", required=True)
    upload = sub.add_parser("upload")
    upload.add_argument("--store", default="r2")
    upload.add_argument("--out-dir", required=True)
    upload.add_argument("--prefix", required=True)
    upload.add_argument("--partial", action="store_true")
    log = sub.add_parser("upload-log")
    log.add_argument("--store", default="r2")
    log.add_argument("--file", required=True)
    log.add_argument("--key", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return dict(recipe=run_recipe, coverage=run_coverage, build=run_build, train=run_train, trained=run_trained, preflight=run_preflight, read=run_read, control=run_control, merge=run_merge, check=run_check, upload=PP.run_upload, **{"upload-log": R.run_upload_log, "val-control": run_val_control})[args.command](args)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
