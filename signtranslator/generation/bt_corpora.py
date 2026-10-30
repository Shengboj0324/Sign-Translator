"""Instrument v2 data definition (docs/NEXT_PHASE_PLAN.md, "Instrument v2: design and declared gates", 2026-10-30
09:10 UTC): which records train the state-to-text back-translator, which select its checkpoint, which are only
reported, and which may never be read by its trainer.

Corpora are given as ``KEY@SHA[:WEIGHT[:CORPUS,...]]`` (a manifest; optionally its sampling weight and the corpora of
a mixed manifest to take: by default every corpus below that the manifest holds). Each record gets a role from its
corpus and partition (``role_of``):

| corpus | record | role |
|---|---|---|
| ``how2sign-two-view-text-v1`` | ``train``, component c02 | ``train`` |
| | ``train``, component c04 | ``select:how2sign-c04`` (selection, never trained on) |
| | ``val`` (c01, c03, c05) | ``val:how2sign`` (the gates' clips; never loaded by the trainer) |
| | ``test``, ``heldout_official``, component c00 (signer 9) or unknown | never read (dropped unread / refused) |
| ``youtube-asl-*`` | ``train`` | ``train`` |
| | ``monitor`` (channel-disjoint) | ``select:youtube-monitor`` (a fixed sample of ``monitor_clips``) |
| ``asl-stem-wiki*`` | ``train`` | ``train`` |
| | ``holdout`` (4 participants) | ``holdout:asl-stem-wiki`` (reported only) |
| ``2m-flores*`` | any | ``crossdomain:2m-flores`` (G4 only; refused by the trainer) |

Resolution of the selection set (recorded in reports/instrument_v2_dev.md): the declaration lists How2Sign c04 both as
a training component and as selection data; selection data must be disjoint from training data, so c04 is selection
only and How2Sign trains on c02.

Text hygiene of every ``train`` record (counted): a sentence equal or near (``build_refiner_pairs.NearDuplicates``:
Jaccard >= 0.8, or containment >= 0.8 of the smaller set with >= 4 tokens and >= 0.6 of the larger) to any
How2Sign val, test or held-out sentence, or to any 2M-Flores sentence, is dropped, and so is one whose normalised
text equals a selection or report-only (held-out) sentence. ``check_loadable_bt`` is the hard guard, applied when
the data are assembled and again on every load: no test-partition record, no How2Sign signer 9 or unknown component,
and in training no How2Sign val record, no How2Sign component but c02, no cross-domain and no held-out record.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
import hashlib
import json
import random
import re
from typing import Mapping

from .recognizer_corpora import FORBIDDEN_PARTITIONS, check_loadable, corpus_name, is_how2sign_test_signer

HOW2SIGN = "how2sign-two-view-text-v1"
HOW2SIGN_TRAIN_COMPONENTS = ("c02",)
HOW2SIGN_SELECT_COMPONENTS = ("c04",)
HOW2SIGN_VAL_COMPONENTS = ("c01", "c03", "c05")
TRAIN = "train"
SELECT_H2S, SELECT_YT = "select:how2sign-c04", "select:youtube-monitor"
VAL_H2S = "val:how2sign"
HOLDOUT_STEM = "holdout:asl-stem-wiki"
CROSSDOMAIN = "crossdomain:2m-flores"
MONITOR_CLIPS = 1000
_SHA = re.compile(r"^[0-9a-f]{64}$")


def family(corpus: str) -> str | None:
    """The corpus family the roles are defined for, or None (not a back-translator corpus)."""
    c = (corpus or "").lower()
    if c == HOW2SIGN:
        return "how2sign"
    if c.startswith("youtube-asl"):
        return "youtube"
    if c.startswith("asl-stem-wiki"):
        return "stem"
    if c.startswith("2m-flores"):
        return "flores"
    return None


@dataclass(frozen=True)
class CorpusSpec:
    key: str
    sha256: str
    weight: float = 1.
    corpora: tuple = ()

    def to_dict(self) -> dict:
        return dict(key=self.key, sha256=self.sha256, weight=self.weight, corpora=list(self.corpora))


def parse_corpus_spec(text: str) -> CorpusSpec:
    """``KEY@SHA[:WEIGHT[:CORPUS,...]]``."""
    if text.count("@") != 1:
        raise ValueError(f"--corpus {text!r}: expected KEY@SHA[:WEIGHT[:CORPUS,...]]")
    key, rest = text.split("@")
    parts = rest.split(":")
    if not key or not 1 <= len(parts) <= 3 or not _SHA.fullmatch(parts[0]):
        raise ValueError(f"--corpus {text!r}: expected KEY@SHA[:WEIGHT[:CORPUS,...]]")
    weight = float(parts[1]) if len(parts) >= 2 and parts[1] else 1.
    if not weight > 0 or weight != weight or weight == float("inf"):
        raise ValueError(f"--corpus {text!r}: the weight must be a positive finite number")
    corpora = tuple(c for c in parts[2].split(",") if c) if len(parts) == 3 else ()
    for c in corpora:
        if family(c) is None:
            raise ValueError(f"--corpus {text!r}: {c!r} is not a back-translator corpus")
    return CorpusSpec(key, parts[0], weight, corpora)


def role_of(record: Mapping, name: str) -> str | None:
    """The record's role (module table), None for a record that is not read at all (dropped unread), or raise for a
    record whose identity is unsafe (How2Sign signer 9 or unknown component outside the forbidden partitions)."""
    partition = record.get("partition")
    fam = family(name)
    if partition in FORBIDDEN_PARTITIONS or record.get("source_partition") in FORBIDDEN_PARTITIONS:
        return None
    if fam == "flores":
        return CROSSDOMAIN
    if fam == "how2sign":
        if is_how2sign_test_signer(record, name):
            raise RuntimeError(f"{record.get('sign')}: How2Sign signer 9 or an unknown component in a "
                               "back-translator corpus")
        component = record.get("signer_component")
        if partition == "val":
            return VAL_H2S if component in HOW2SIGN_VAL_COMPONENTS else None
        if partition == "train":
            if component in HOW2SIGN_TRAIN_COMPONENTS:
                return TRAIN
            if component in HOW2SIGN_SELECT_COMPONENTS:
                return SELECT_H2S
        return None
    if fam == "youtube":
        return {"train": TRAIN, "monitor": SELECT_YT}.get(partition)
    if fam == "stem":
        return {"train": TRAIN, "holdout": HOLDOUT_STEM}.get(partition)
    return None


def check_loadable_bt(record: Mapping, *, training: bool) -> None:
    """Hard guard (not an ``assert``): raise unless ``record`` may be read; with ``training``, unless it may be trained
    on. Every read: never a test or held-out-official record, never How2Sign signer 9 or an unknown How2Sign
    component (``recognizer_corpora.check_loadable``). Training: only role ``train``, never a How2Sign val record or a
    How2Sign component other than c02, never a cross-domain (2M-Flores) or held-out record."""
    check_loadable(record)
    name = record.get("corpus") or ""
    role = record.get("_role")
    if role != role_of(record, name):
        raise RuntimeError(f"{record.get('sign')}: role {role!r} is not its corpus and partition's "
                           f"({role_of(record, name)!r})")
    if not training:
        return
    if family(name) == "flores" or role == CROSSDOMAIN:
        raise RuntimeError(f"{record.get('sign')}: a cross-domain (2M-Flores) record may never be trained on")
    if family(name) == "how2sign" and (record.get("partition") != "train"
                                       or record.get("signer_component") not in HOW2SIGN_TRAIN_COMPONENTS):
        raise RuntimeError(f"{record.get('sign')}: How2Sign {record.get('partition')} "
                           f"{record.get('signer_component')} may never be trained on (only train c02)")
    if role != TRAIN:
        raise RuntimeError(f"{record.get('sign')}: role {role!r} may never be trained on")


def text_of(record: Mapping) -> str:
    return record.get("english") or record.get("text") or ""


def assemble(corpora, *, monitor_clips: int = MONITOR_CLIPS, monitor_seed: int = 0, near: float = .8,
             min_tokens: int = 4, larger: float = .6) -> tuple[list[dict], dict]:
    """(records, summary) for ``corpora``: a list of (manifest, CorpusSpec). Each kept record is a copy with its
    ``corpus``, ``_role`` and ``_weight`` (its corpus weight). Roles ``train``, ``select:*``, ``holdout:*``,
    ``crossdomain:*`` and ``val:how2sign`` are kept (the last two never by the trainer's loaders); YouTube monitor
    records are the fixed sample of ``monitor_clips`` (``random.Random(monitor_seed)`` over the monitor records sorted
    by sign; fewer available raises). Records without usable continuous channels or text are dropped and counted."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from build_refiner_pairs import NearDuplicates, normalised_text
    from ..data_engineering.corpus_manifest import usable_channels
    kept, counts, seen = [], collections.defaultdict(collections.Counter), set()
    eval_texts, reserved_texts = set(), set()
    for manifest, spec in corpora:
        wanted = set(spec.corpora)
        for r in manifest["records"]:
            name = corpus_name(manifest, r)
            c = counts[name]
            fam = family(name)
            if fam == "how2sign" and (r.get("partition") != "train" or r.get("source_partition", "train") != "train"):
                eval_texts.add(normalised_text(text_of(r)))  # val, test, held-out sentences: texts only, never states
            if fam == "flores":                             # (from any manifest given, taken or not)
                eval_texts.add(normalised_text(text_of(r)))
            if fam is None or (wanted and name not in wanted):
                c["drop:not_taken"] += 1
                continue
            role = role_of(r, name)
            if role is None:
                c[f"drop:{r.get('partition')}:{r.get('signer_component') or '-'}"] += 1
                continue
            if not text_of(r) or not usable_channels(r, "continuous"):
                c["drop:no_text_or_channels"] += 1
                continue
            if (name, r["sign"]) in seen:
                raise ValueError(f"{name} {r['sign']}: listed twice")
            seen.add((name, r["sign"]))
            out = dict(r, corpus=name, _role=role, _weight=float(spec.weight))
            check_loadable_bt(out, training=False)
            kept.append(out)
    eval_texts.discard("")
    # the fixed YouTube monitor sample (selection)
    monitor = sorted((r for r in kept if r["_role"] == SELECT_YT), key=lambda r: (r["corpus"], r["sign"]))
    if monitor:
        if len(monitor) < monitor_clips:
            raise ValueError(f"{len(monitor)} YouTube monitor clips, fewer than the declared {monitor_clips}")
        chosen = {(r["corpus"], r["sign"]) for r in random.Random(monitor_seed).sample(monitor, monitor_clips)}
        dropped = [r for r in monitor if (r["corpus"], r["sign"]) not in chosen]
        for r in dropped:
            counts[r["corpus"]]["drop:monitor_not_sampled"] += 1
        drop = {(r["corpus"], r["sign"]) for r in dropped}
        kept = [r for r in kept if (r["corpus"], r["sign"]) not in drop]
    reserved_texts = {normalised_text(text_of(r)) for r in kept if r["_role"] not in (TRAIN,)} - {""}
    duplicates = NearDuplicates(sorted(eval_texts), near, min_tokens, larger)
    records = []
    for r in kept:
        if r["_role"] == TRAIN:
            t = normalised_text(text_of(r))
            if t in eval_texts:
                counts[r["corpus"]]["drop:text_in_evaluation"] += 1
                continue
            close = duplicates.reason(text_of(r))
            if close:
                counts[r["corpus"]][f"drop:near_evaluation:{close}"] += 1
                continue
            if t in reserved_texts:
                counts[r["corpus"]]["drop:text_in_selection_or_holdout"] += 1
                continue
        records.append(r)
        counts[r["corpus"]][f"keep:{r['_role']}"] += 1
    records.sort(key=lambda r: (r["corpus"], r["sign"]))
    summary = dict(corpora=[spec.to_dict() for _, spec in corpora],
                   counts={k: dict(sorted(v.items())) for k, v in sorted(counts.items())},
                   roles=dict(sorted(collections.Counter(r["_role"] for r in records).items())),
                   monitor_clips=monitor_clips, monitor_seed=monitor_seed,
                   near_duplicates=dict(threshold=near, min_tokens=min_tokens, larger=larger),
                   evaluation_texts=len(eval_texts))
    return records, summary


def records_of(records, role_prefix: str) -> list[dict]:
    return [r for r in records if r["_role"] == role_prefix or r["_role"].startswith(role_prefix + ":")]


def data_digest(records) -> str:
    """SHA-256 naming exactly the assembled records: corpus, sign, role, weight, state and recon copy."""
    rows = [(r["corpus"], r["sign"], r["_role"], r["_weight"], r["continuous"]["state_sha256"],
             (r.get("_recon") or {}).get("state_sha256")) for r in records]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


__all__ = ["HOW2SIGN", "TRAIN", "SELECT_H2S", "SELECT_YT", "VAL_H2S", "HOLDOUT_STEM", "CROSSDOMAIN", "MONITOR_CLIPS",
           "CorpusSpec", "family", "parse_corpus_spec", "role_of", "check_loadable_bt", "text_of", "assemble",
           "records_of", "data_digest"]
