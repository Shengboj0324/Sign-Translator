"""What the three parent reads share (``sensitivity_ladder``, ``parent_comp``, ``symmetric_ft``): the pinned stored reads, v1's fine-tuning lists from their own index objects, the mono-tuned instrument's record, and the margin arithmetic. Nothing here reads a form.

**A fact the three reads disclose (counted 2026-11-02 from index objects, no state and no score read).** v1's fine-tuning partitions are ``full-v1``'s, not r10's. Its train list holds 17,836 c02 and 343 c04 train clips and also 1,131 c01 and 327 c05 clips that r10 calls val; its val list is 1,627 c03 clips. So D20 (c01, c03, c05) has 234 clips in v1's training and 265 in the val pool it was selected on, and C20 has 455 and 544. v1's read of real two-view D20 is partly a read of its own training clips, through the path it trained on. The parent never trained on How2Sign.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import path_parts as PP  # noqa: E402
import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
from signtranslator.eval import s0_metrics as M  # noqa: E402

PARTS_FORMS = f"{L.O}/round20/path-read-parts/forms"
STORED_PARENT = {"real_two_view": ("form-uwb_real_two_view.json", "8052a9efbf05d1ca911d7dde1c1dcc8eb91cb382f49f0f17b069b434b1a5a06e"),
                 "blank": ("form-uwb_blank.json", "73b0d0f1c4722e086386dc19df0f5bae78efe94cb108baef97a3245db98fac50"),
                 "nearest_two_view": ("form-uwb_nearest_two_view.json", "e1f41351306ba35ec8ff367e0c6c588da6b5af067f393d393ebd1c0ce36cdb41"),
                 "still_two_view": ("form-uwb_still_two_view.json", "afd85321306c42404f3c05fd8d9e2c2fd0c3ac73301131f3ea5a661eb4266290")}
STORED_V1 = {"nearest_two_view": ("form-nearest_two_view.json", "149208eeddeac240c2e6dd0c7d9983bef8f25c4e3ba94449b9b4112d33c74507"),
             "nearest_monocular": ("form-nearest_monocular.json", "15b87f8efb4d2bb52fb09f1512bb4504216e53b950fd8b9dc4469a85165254f7"),
             "real_two_view": ("form-real_two_view.json", "6e1bd9f8e4e64e69f2ed28e2ed3548f9dd456fadf5835c72f4b50de9bc7d2213"),
             "blank": ("form-blank.json", "3f60a099812f82378770fc8b3c5273fb263d4db0004ae342316bdb831289a496")}
FK_INDEX = PP.HOW2SIGN_FK_INDEX
FK_INDEX_SHA = "38111da3c83de7b52b737e2452aa7367ff7af4e008325b95c54488b580795e8a"
V1_RUN_JSON = f"training-runs/{L.TRANSLATOR_RUN}/run.json"
V1_RUN_JSON_SHA = "289e579df81c9f10fa89658fba2648d0df21b7983e8cbc6cd06bcb2fbaa82bcc"
MONO_RUN = "back-translator-kp2d-uwb-ft-mono-v1"
MONO_PREFIX = f"training-runs/{MONO_RUN}"
TRAINED_RECORD = f"{MONO_PREFIX}/trained.json"
SERIES = ("train", "val")
CANDIDATES = PP.CANDIDATES
BOOT = 2000
READ_LIMITS = dict(identical=P.CANARY["identical"], nll=P.CANARY["nll"], bleu=P.CANARY["bleu_tolerance"])


def stored_form(store, table: dict, name: str) -> dict:
    """A stored read of the parts job, its bytes pinned here; another name is refused before it is fetched."""
    if name not in table:
        raise SystemExit(f"{name}: no stored read is pinned under this name")
    file, sha = table[name]
    key = f"{PARTS_FORMS}/{file}"
    raw = store.get(key)
    P.check_sha(raw, sha, key)
    return json.loads(raw)


def same_read(read: dict, stored: dict) -> dict:
    """A form of this run against its stored read: the canary's limits, on the same clips in the same order, over the NLL columns both hold. Other clips or another order refuse."""
    if read["signs"] != stored["signs"] or read["references"] != stored["references"]:
        raise SystemExit(f"{read.get('form')}: the stored read holds other clips or another order")
    identical = float(np.mean([a == b for a, b in zip(read["hypotheses"], stored["hypotheses"])]))
    a, b = np.asarray(read["nll"]), np.asarray(stored["nll"])
    columns = min(a.shape[1], b.shape[1])
    gap = float(np.mean(np.abs(a[:, :columns] - b[:, :columns])))
    here, there = M.bleu_ci_fast(read["hypotheses"], read["references"])[0], M.bleu_ci_fast(stored["hypotheses"], stored["references"])[0]
    row = dict(bleu=here, stored_bleu=there, bleu_gap=round(abs(here - there), 4), identical=round(identical, 4), nll_mean_absolute_difference=round(gap, 6), nll_columns=columns)
    row["misses"] = [why for why, bad in (("BLEU", row["bleu_gap"] > READ_LIMITS["bleu"]), ("hypotheses", identical < READ_LIMITS["identical"]), ("NLL", gap > READ_LIMITS["nll"])) if bad]
    return row


def v1_series(store, partition: str) -> list[dict]:
    """v1's own list of ``partition``, in its order, from the clip lists of its pinned index. Only its train and val series can be named: a test or held-out series is refused before anything is fetched, and no shard of points is opened."""
    if partition not in SERIES:
        raise SystemExit(f"{partition}: only v1's {SERIES} are read; test and held-out series are refused unread")
    raw = store.get(FK_INDEX)
    P.check_sha(raw, FK_INDEX_SHA, FK_INDEX)
    rows = []
    for part in json.loads(raw)["parts"]:
        if part["partition"] != partition:
            continue
        for shard in part["shards"]:
            meta = store.get(shard["key"][:-4] + ".json")
            P.check_sha(meta, shard["meta_sha256"], shard["key"])
            rows += json.loads(meta)
    wrong = sorted({r["partition"] for r in rows} - {partition})
    if wrong:
        raise SystemExit(f"{FK_INDEX}: its {partition} series holds {wrong}")
    return rows


def v1_training_signs(store) -> frozenset:
    return frozenset(r["clip_id"] for r in v1_series(store, "train"))


def subset_masks(signs, training) -> dict:
    """{in_v1_training, outside_v1_training}: boolean masks over ``signs``, report only."""
    inside = np.array([s in training for s in signs])
    return dict(in_v1_training=inside, outside_v1_training=~inside)


def margins(read: dict) -> np.ndarray:
    return PP.margins(read)


def load_mono(store, args, prefix: str = MONO_PREFIX):
    """The mono-tuned instrument, its checkpoint named by its run's record and held to the hash there; the record must agree with its best.json. Before training there is none, and this refuses. ``prefix``: a smoke's run."""
    from score_back_translation import load_translator
    import torch
    record_key = f"{prefix}/trained.json"
    if not store.exists(record_key):
        raise SystemExit(f"{record_key}: no trained record; the mono-tuned instrument is not ready")
    record = json.loads(store.get(record_key))
    best = json.loads(store.get(f"{prefix}/best.json"))
    if record.get("checkpoint_key") != best.get("checkpoint_key") or not str(record["checkpoint_key"]).startswith(f"{prefix}/checkpoints/"):
        raise SystemExit(f"{record_key} names {record.get('checkpoint_key')}, its best.json {best.get('checkpoint_key')}")
    payload, digest = L.instrument_payload(store, record["checkpoint_key"], record["sha256"], getattr(args, "instrument_cache", None))
    model, kind = load_translator(L._OneFile(payload), record["checkpoint_key"], L.TRANSLATOR_MANIFEST_SHA)
    del payload
    if kind != "keypoints":
        raise SystemExit(f"{record['checkpoint_key']}: not a 2D keypoint reader but {kind!r}")
    device = torch.device(args.device)
    return L.Reader(model.to(device).eval(), kind, dict(run=prefix.removeprefix("training-runs/"), checkpoint_key=record["checkpoint_key"], sha256=digest, kind=kind), device)


def digest_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
