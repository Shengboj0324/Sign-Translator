"""Recogniser v3 data definition (docs/NEXT_PHASE_PLAN.md, "Instrument finding and the recogniser v3 plan").

One label space (the ASL Citizen glosses: the primary manifest's, or a run's stored labels) and several corpora, each
given as ``KEY@SHA:MAPPING[:WEIGHT]`` and each mapped into that label space exactly:

- ``label``: the record's ``gloss.label`` is itself a label (ASL Citizen);
- ``asllex``: the record's ``gloss.asl_lex_code`` names exactly one label among the primary manifest's records carrying
  that code (Sem-Lex; ASL Citizen alike). Codes ``""`` and ``"NA"`` map nothing;
- ``signbank``: the Sign Bank gloss maps to exactly one label through the ``method == "exact"`` entries of the Sign Bank
  <-> ASL Citizen gloss map (ASLLVD, RIT; recogniser v2's rule, ``train_sign_recognizer.with_extra_corpus``);
- ``upper``: the record's ``gloss.label`` upper-cased is a label, spelled exactly (WLASL).

Partitions: ``train`` stays ``train``; ``val`` becomes ``val:<corpus>``; every clip (train or val) of a held-out signer
(``--holdout-signers corpus:id,id``) becomes ``holdout:<corpus>``; excluded signers (``--exclude-signers``) are dropped.
The ``test`` and ``heldout_official`` partitions are dropped before anything else and never reach a dataset, and no
How2Sign signer-9 record (component ``c00``, the generator's final-test signer) may be loaded: ``check_loadable`` is the
hard guard, applied when the data are assembled and again on every load.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, Iterable, Mapping

MAPPINGS = ("label", "asllex", "signbank", "upper")
FORBIDDEN_PARTITIONS = ("test", "heldout_official")
HOW2SIGN_TEST_COMPONENTS = ("c00",)          # How2Sign signer 9's component: the generator's untouched final test
HOW2SIGN_TEST_SIGNERS = ("9", "h2s-9")
NO_CODE = ("", "NA", None)
_SHA = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CorpusSpec:
    key: str
    sha256: str
    mapping: str
    weight: float = 1.

    def to_dict(self) -> dict:
        return dict(key=self.key, sha256=self.sha256, mapping=self.mapping, weight=self.weight)


def parse_corpus_spec(text: str) -> CorpusSpec:
    """``KEY@SHA:MAPPING[:WEIGHT]``; the key may hold ``/`` but not ``@``."""
    if text.count("@") != 1:
        raise ValueError(f"--corpus {text!r}: expected KEY@SHA:MAPPING[:WEIGHT]")
    key, rest = text.split("@")
    parts = rest.split(":")
    if not key or len(parts) not in (2, 3):
        raise ValueError(f"--corpus {text!r}: expected KEY@SHA:MAPPING[:WEIGHT]")
    sha, mapping = parts[0], parts[1]
    if not _SHA.fullmatch(sha):
        raise ValueError(f"--corpus {text!r}: {sha!r} is not a SHA-256")
    if mapping not in MAPPINGS:
        raise ValueError(f"--corpus {text!r}: mapping must be one of {MAPPINGS}")
    weight = float(parts[2]) if len(parts) == 3 else 1.
    if not weight > 0 or weight != weight or weight == float("inf"):
        raise ValueError(f"--corpus {text!r}: the weight must be a positive finite number")
    return CorpusSpec(key, sha, mapping, weight)


def parse_signer_specs(items: Iterable[str] | None) -> dict[str, tuple[str, ...]]:
    """Repeated ``corpus:id,id`` -> {corpus: sorted distinct ids}; a corpus may be named more than once (union)."""
    out = collections.defaultdict(set)
    for item in items or ():
        if item.count(":") != 1:
            raise ValueError(f"signer spec {item!r}: expected CORPUS:ID[,ID...]")
        corpus, ids = item.split(":")
        values = [v.strip() for v in ids.split(",")]
        if not corpus or not values or any(not v for v in values):
            raise ValueError(f"signer spec {item!r}: expected CORPUS:ID[,ID...]")
        out[corpus].update(values)
    return {c: tuple(sorted(v)) for c, v in sorted(out.items())}


def corpus_name(manifest: Mapping, record: Mapping) -> str:
    return record.get("corpus") or manifest.get("corpus_id") or "unnamed"


def signer_of(record: Mapping) -> str | None:
    """The corpus's own signer id when the record carries one (Sem-Lex ``source.signer_id``), else the pseudonymous
    ``performer_sha256``."""
    source = record.get("source") or {}
    return str(source["signer_id"]) if source.get("signer_id") not in (None, "") else record.get("performer_sha256")


def signer_matches(record: Mapping, ids) -> str | None:
    """The id in ``ids`` that names this record's signer: its own signer id, its performer SHA-256, or a prefix of at
    least 12 hex digits of that SHA-256."""
    own = (record.get("source") or {}).get("signer_id")
    performer = record.get("performer_sha256") or ""
    for i in ids:
        if (own not in (None, "") and str(own) == i) or (performer and (i == performer or
                                                                        (len(i) >= 12 and performer.startswith(i)))):
            return i
    return None


def is_how2sign_test_signer(record: Mapping, name: str | None = None) -> bool:
    corpus = (name or record.get("corpus") or "").lower()
    if "how2sign" not in corpus and "h2s" not in corpus:
        return False
    source = record.get("source") or {}
    return (record.get("signer_component") in HOW2SIGN_TEST_COMPONENTS or record.get("signer_component") is None
            or str(source.get("signer_id")) in HOW2SIGN_TEST_SIGNERS)


def check_loadable(record: Mapping) -> None:
    """Hard guard: raise unless ``record`` may be read (never a test-partition record, never How2Sign signer 9, and no
    How2Sign record whose component is unknown). Not an ``assert``: it holds under ``python -O`` too."""
    original = record.get("source_partition", record.get("partition"))
    if original in FORBIDDEN_PARTITIONS or record.get("partition") in FORBIDDEN_PARTITIONS:
        raise RuntimeError(f"{record.get('sign')}: a {original} record may never be loaded")
    if is_how2sign_test_signer(record):
        raise RuntimeError(f"{record.get('sign')}: How2Sign signer 9 (or an unknown How2Sign component) may never be "
                           "loaded")


def code_labels(primary: Mapping, labels) -> dict[str, set]:
    """ASL-LEX code -> the labels whose primary-manifest records carry it."""
    out = collections.defaultdict(set)
    for r in primary["records"]:
        g = r.get("gloss") or {}
        if g.get("asl_lex_code") not in NO_CODE and g.get("label") in labels:
            out[g["asl_lex_code"]].add(g["label"])
    return out


def signbank_targets(gloss_map: Mapping, labels) -> dict[str, set]:
    """Sign Bank gloss -> the labels it maps to *exactly* (``method == "exact"``), as recogniser v2."""
    out = collections.defaultdict(set)
    for g, m in gloss_map["mapping"].items():
        if m["method"] == "exact" and g in labels:
            for c in m["candidates"]:
                out[c].add(g)
    return out


def label_mapper(mapping: str, labels, *, codes: Mapping | None = None,
                 signbank: Mapping | None = None) -> Callable[[Mapping], str | None]:
    """record -> its label (exactly one) or None."""
    labels = set(labels)

    def one(targets):
        targets = set(targets or ())
        return next(iter(targets)) if len(targets) == 1 else None

    if mapping == "label":
        return lambda r: (r.get("gloss") or {}).get("label") if (r.get("gloss") or {}).get("label") in labels else None
    if mapping == "upper":
        def upper(r):
            label = (r.get("gloss") or {}).get("label")
            return label.upper() if isinstance(label, str) and label.upper() in labels else None
        return upper
    if mapping == "asllex":
        if codes is None:
            raise ValueError("the asllex mapping needs the primary manifest's codes")
        return lambda r: (None if (r.get("gloss") or {}).get("asl_lex_code") in NO_CODE
                          else one(codes.get((r.get("gloss") or {}).get("asl_lex_code"))))
    if mapping == "signbank":
        if signbank is None:
            raise ValueError("the signbank mapping needs the gloss map")
        return lambda r: one(signbank.get((r.get("gloss") or {}).get("label")))
    raise ValueError(f"unknown mapping {mapping!r}")


def assemble(corpora, labels, *, holdout: Mapping | None = None, exclude: Mapping | None = None,
             whole_clip: Iterable[str] = ()) -> tuple[list[dict], dict]:
    """(records, summary) for ``corpora``: a list of (manifest, spec, mapper). Each kept record is a copy with its
    label, partition (``train`` / ``val:<c>`` / ``holdout:<c>``), ``source_partition``, ``corpus`` and ``_weight``
    (its corpus weight). ``whole_clip``: corpora whose ``continuous.window_seconds`` trim is removed (the clip as
    stored). Unknown corpora or signer ids in ``holdout`` / ``exclude`` are refused."""
    holdout, exclude, whole_clip = dict(holdout or {}), dict(exclude or {}), set(whole_clip)
    names = []
    for manifest, spec, _ in corpora:
        own = {corpus_name(manifest, r) for r in manifest["records"]} or {manifest.get("corpus_id") or "unnamed"}
        if len(own) != 1:
            raise ValueError(f"{spec.key}: one corpus per manifest (found {sorted(own)})")
        names.append(own.pop())
    every = set(names)
    for what, spec in (("holdout", holdout), ("exclude", exclude), ("whole-clip", {c: () for c in whole_clip})):
        unknown = set(spec) - every
        if unknown:
            raise ValueError(f"--{what}: unknown corpora {sorted(unknown)} (known: {sorted(every)})")
    both = {c: set(holdout.get(c, ())) & set(exclude.get(c, ())) for c in holdout}
    if any(both.values()):
        raise ValueError(f"signers both held out and excluded: {({c: sorted(v) for c, v in both.items() if v})}")
    if len(set(names)) != len(names):
        raise ValueError(f"two corpora share a name: {names}")
    records, used = [], collections.defaultdict(set)
    counts = collections.defaultdict(collections.Counter)
    for (manifest, spec, mapper), name in zip(corpora, names):
        for r in manifest["records"]:
            c = counts[name]
            partition = r.get("partition")
            if partition in FORBIDDEN_PARTITIONS:
                c[f"drop:{partition}_partition"] += 1          # never looked at further
                continue
            if partition not in ("train", "val"):
                c[f"drop:partition_{partition}"] += 1
                continue
            if is_how2sign_test_signer(r, name):
                raise RuntimeError(f"{r.get('sign')}: How2Sign signer 9 is in a recogniser corpus")
            hit = signer_matches(r, exclude.get(name, ()))
            if hit is not None:
                used[(name, "exclude")].add(hit)
                c["drop:excluded_signer"] += 1
                continue
            label = mapper(r)
            if label is None:
                c["drop:unmapped_label"] += 1
                continue
            held = signer_matches(r, holdout.get(name, ()))
            if held is not None:
                used[(name, "holdout")].add(held)
            new = "train" if partition == "train" and held is None else (f"holdout:{name}" if held is not None
                                                                         else f"val:{name}")
            continuous = dict(r["continuous"])
            if name in whole_clip:
                continuous.pop("window_seconds", None)
            out = dict(r, corpus=name, partition=new, source_partition=partition, continuous=continuous,
                       gloss=dict(r.get("gloss") or {}, label=label, source_label=(r.get("gloss") or {}).get("label")),
                       _weight=float(spec.weight))
            check_loadable(out)
            records.append(out)
            c[f"keep:{new}"] += 1
    for name, spec in (("holdout", holdout), ("exclude", exclude)):
        for corpus, ids in spec.items():
            missing = set(ids) - used[(corpus, name)]
            if missing:
                raise ValueError(f"--{name} {corpus}: no record of signer(s) {sorted(missing)}")
    summary = dict(corpora={name: dict(spec.to_dict(), counts=dict(sorted(counts[name].items())))
                            for name, (_, spec, _) in zip(names, corpora)},
                   holdout={c: list(v) for c, v in holdout.items()}, exclude={c: list(v) for c, v in exclude.items()},
                   whole_clip=sorted(whole_clip),
                   partitions=dict(sorted(collections.Counter(r["partition"] for r in records).items())))
    return records, summary


def load_digest(record: Mapping) -> str:
    """SHA-256 naming exactly what ``corpus_manifest.load_sample`` reads for this record: the state, the usable
    channels, the trim window and the mirror flag. A decoded copy is used only for the record it was made from."""
    from ..data_engineering.corpus_manifest import usable_channels
    part = record["continuous"]
    spec = dict(state_sha256=part["state_sha256"], usable=list(usable_channels(record, "continuous")),
                window_seconds=part.get("window_seconds"), mirror=record.get("mirror") is True)
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def recon_lookup(index: Mapping) -> dict[tuple[str, str], dict]:
    """(corpus, sign) -> index entry for every decoded state of a ``build_recon_states.py`` index."""
    out = {}
    for e in index["entries"]:
        if e.get("status") not in ("converted", "already_present"):
            continue
        k = (e["corpus"], e["sign"])
        if k in out and out[k]["recon_sha256"] != e["recon_sha256"]:
            raise ValueError(f"{k}: two decoded states in the index")
        out[k] = e
    return out


def attach_recon(records: list[dict], lookup: Mapping) -> collections.Counter:
    """Give each record whose decoded copy matches its load digest a ``_recon`` (key, SHA-256); counts per partition."""
    counts = collections.Counter()
    for r in records:
        e = lookup.get((r["corpus"], r["sign"]))
        ok = e is not None and e["load_digest"] == load_digest(r)
        if ok:
            r["_recon"] = dict(state_key=e["recon_key"], state_sha256=e["recon_sha256"])
        counts[f"{r['partition']}:{'with' if ok else 'without'}"] += 1
    return counts


def data_digest(payload: Mapping) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


__all__ = ["CorpusSpec", "MAPPINGS", "FORBIDDEN_PARTITIONS", "parse_corpus_spec", "parse_signer_specs", "corpus_name",
           "signer_of", "signer_matches", "is_how2sign_test_signer", "check_loadable", "code_labels",
           "signbank_targets", "label_mapper", "assemble", "load_digest", "recon_lookup", "attach_recon", "data_digest"]
