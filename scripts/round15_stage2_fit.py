"""Round 15 stage 2: fit the compositional generator's grammar and timing on ASLLRP train, and check them on ASLLRP val
(docs/NEXT_PHASE_PLAN.md, "Round 15 stage 2 groundwork", 2026-10-28 21:16 UTC). Development data only: ASLLRP train
fits, ASLLRP val checks, the ASLLRP test partition is never read, and no How2Sign sentence is read at all.

**Inputs** (read-only; every payload's SHA-256 is recorded in the model):
- the ASLLRP corpus: ``derived/signbank-v1/utterances.json`` and ``derived/asllrp-dai2-v1/english-gloss-v1/pairs.json``
  joined by ``asllrp_gloss_corpus.build_corpus`` (train and val only; test pairs are counted and dropped there);
- the generator's training mix (``--mix-key`` / ``--mix-sha256``): its isolated-sign vocabulary and typical durations
  (``compositional.training_vocabulary``), the Sign Bank glosses / ASL-LEX codes of its train records and the ASLLVD
  annotated sign spans (``source.frames`` at 60 fps). Only isolated-corpus train records are read;
- ``derived/signbank-v1/asl-citizen-gloss-map.json`` (main entry -> ASL-LEX code, exact tier only), the ASLLRP Sign
  Bank lexicon ``derived/signbank-v1/lexicon.json`` (main entries and their variant labels) and ASL-LEX 2.0
  ``asl-lex-2.0/signdata.csv`` (EntryID, LemmaID, Code, DominantTranslation, SignBankEnglishTranslations).

**Gold:** each utterance's gloss sequence mapped to vocabulary entries exactly as the groundwork's ``rule_check``
(``map_sign`` per gloss type, tiers with index), unmapped glosses left out.

**Lexicon** (``build_lexicon``, selection beyond exact text; no ASLLRP sentence is read):
- ``signbank_variant``: a variant label of a Sign Bank main entry whose sign is in the vocabulary (through the train
  records' Sign Bank glosses, else the main entry's own name) is that entry ("feet" -> foot);
- ``signbank_alternative``: the other names of a slash or alternative main entry (THUMBTACK/BUTTON);
- ``asllex_dominant`` / ``asllex_translation``: the ASL-LEX DominantTranslation and the Signbank English translations
  of an ASL-LEX sign whose entry is in the vocabulary (Sem-Lex codes, else its EntryID / LemmaID text);
- a word is never mapped onto another sign when it names a sign of its own (an ASL-LEX EntryID / LemmaID or a Sign
  Bank main entry): it is a different sign the vocabulary lacks. Words in the vocabulary, stop words, single letters
  and pronouns (the pointing rule's) are never keys. Conflicts: tier order, then votes, then the entry with more
  train clips, then the name.

**Fit** (``fit_stage2``): ``compositional_grammar.fit_structure`` (keep / map / insert / drop per token, pooled
pointing, reorder moves) and ``fit_timing`` on ASLLRP train only (``check_fit_records``).

**Check on ASLLRP val** (``evaluate``): bag precision / recall / F1, entry WER and Kendall tau of the shared entries for
the stage-1 rule (``compositional.select_words``, which reproduces the groundwork's ``rule_check``), the stage-2
selection alone, selection + token operations, and the full model (+ moves), each against stage 1 with a paired
bootstrap; the lexicon's per-tier precision; the threshold sensitivity (refits at other thresholds, reported only);
and the timing model's error on val sign durations against a constant (train median) baseline.

Writes the model (``--model-file``, write-once, local: it holds ASLLRP-derived statistics, research terms) and the
report (``--output-file``, write-once).
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import asllrp_gloss_corpus as agc  # noqa: E402
from signtranslator.generation import compositional_grammar as cg  # noqa: E402
from signtranslator.generation.compositional import (ISOLATED_CORPORA, STOP_WORDS, normalise, select_words,  # noqa: E402
                                                     training_vocabulary)

ACQ = "acquisition/2026-10-17-v1/"
KEYS = dict(utterances=ACQ + "derived/signbank-v1/utterances.json",
            pairs=ACQ + "derived/asllrp-dai2-v1/english-gloss-v1/pairs.json",
            citizen_map=ACQ + "derived/signbank-v1/asl-citizen-gloss-map.json",
            signbank_lexicon=ACQ + "derived/signbank-v1/lexicon.json",
            asllex=ACQ + "asl-lex-2.0/signdata.csv")
ASLLEX_SHA256 = "080ecc3de4b307a04dd9b2c2583c22bc623a5c731ab869d6e3905d4a3540fbd3"
DEFAULTS = dict(threshold=.5, min_count=5, learned=3, move_min_count=10, move_margin=.02, min_signs=3, floor_q=.25,
                ceiling_q=.9)
LEXICAL_CATEGORIES = ("lexical", "loan", "compound", "number")


# ---- vocabulary-side tables --------------------------------------------------------------------------------------

def asllex_text(name: str) -> str:
    """An ASL-LEX EntryID / LemmaID as English (``build_semlex_manifest.asllex_text``), normalised."""
    return " ".join(normalise(re.sub(r"_\d+$", "", (name or "").strip()).replace("_", " ")))


def record_counts(manifest, vocab, corpora=ISOLATED_CORPORA) -> collections.Counter:
    """Train clips per vocabulary entry (the conflict tie-break)."""
    out = collections.Counter()
    for r in manifest["records"]:
        if r.get("corpus") in corpora and r.get("partition") == "train" and r.get("text"):
            e = " ".join(normalise(r["text"]))
            if e in vocab:
                out[e] += 1
    return out


def annotated_seconds(manifest, vocab) -> dict:
    """{entry: median ASLLVD annotated sign seconds} of the entry's ASLLVD train clips (``source.frames`` at 60 fps,
    the SignStream sign span, as the groundwork's ``vocabulary_indices``)."""
    from signtranslator.data_engineering.corpus_manifest import usable_channels
    spans = collections.defaultdict(list)
    for r in manifest["records"]:
        if r.get("corpus") != "asllvd-text-v1" or r.get("partition") != "train" or not r.get("text"):
            continue
        if not usable_channels(r, "continuous"):
            continue
        e = " ".join(normalise(r["text"]))
        f = (r.get("source") or {}).get("frames")
        if e in vocab and f and len(f) == 2 and f[1] >= f[0]:
            spans[e].append((f[1] - f[0] + 1) / agc.ASLLVD_FPS)
    return {e: float(statistics.median(v)) for e, v in sorted(spans.items())}


def _own_signs(signbank_lexicon: dict, asllex_rows) -> set[str]:
    """English names that are signs of their own: Sign Bank main entries and ASL-LEX EntryIDs / LemmaIDs."""
    own = set()
    for main in (signbank_lexicon.get("lexicon") or {}):
        if agc.category(main) in LEXICAL_CATEGORIES:
            for alt in agc.name_candidates(main):
                if len(alt) == 1:
                    own.add(alt[0])
    for row in asllex_rows:
        for name in (row.get("EntryID"), row.get("LemmaID")):
            t = asllex_text(name)
            if t:
                own.add(t)
    return own


def _translations(field: str) -> list[str]:
    """Comma- or semicolon-separated English translations; gloss-like items (capitals) are skipped."""
    out = []
    for item in re.split(r"[,;]", field or ""):
        item = item.strip()
        if item and not re.search(r"[A-Z]", item):
            t = " ".join(normalise(item))
            if t:
                out.append(t)
    return out


def build_lexicon(vocab, signbank_lexicon: dict, signbank: dict, asllex_rows, semlex: dict, counts) -> dict:
    """{English: {"entry", "tier", "votes"}} (module docstring). ``signbank``: Sign Bank gloss -> Counter(entry) and
    ``semlex``: ASL-LEX code -> Counter(entry), from ``asllrp_gloss_corpus.vocabulary_indices``."""
    own = _own_signs(signbank_lexicon, asllex_rows)
    votes = collections.defaultdict(collections.Counter)          # (word, tier) -> Counter(entry)

    def add(word, entry, tier, owner_names=()):
        word = " ".join(cg.tokens(word))                          # keyed as the selection splits ("can not")
        if not word or len(word) <= 1 or word == entry or word in vocab or word in STOP_WORDS or word in cg.POINTING:
            return
        if word in own and word not in owner_names:
            return                                                # a sign of its own, not this one
        votes[(word, tier)][entry] += 1

    for main, info in sorted((signbank_lexicon.get("lexicon") or {}).items()):
        if agc.category(main) not in LEXICAL_CATEGORIES:
            continue
        names = [alt[0] for alt in agc.name_candidates(main) if len(alt) == 1]
        if main in signbank:
            entry = agc._top(signbank[main])
        else:
            entry = next((n for n in names if agc._matchable(n, vocab)), None)
        if entry is None:
            continue
        for label in sorted(info.get("variants") or {}):
            w = agc.english_entry(label)
            add(w, entry, "signbank_variant", owner_names=names)
        for n in names:
            add(n, entry, "signbank_alternative", owner_names=names)
    for row in asllex_rows:
        hits = semlex.get(row.get("Code") or "")
        entry = agc._top(hits) if hits else None
        if entry is None:
            entry = next((t for t in (asllex_text(row.get("EntryID")), asllex_text(row.get("LemmaID")))
                          if agc._matchable(t, vocab)), None)
        if entry is None:
            continue
        names = {asllex_text(row.get("EntryID")), asllex_text(row.get("LemmaID"))}
        for w in _translations(row.get("DominantTranslation")):
            add(w, entry, "asllex_dominant", owner_names=names)
        for w in _translations(row.get("SignBankEnglishTranslations")):
            add(w, entry, "asllex_translation", owner_names=names)
    rank = {t: i for i, t in enumerate(cg.LEXICON_TIERS)}
    best = {}
    for (word, tier), entries in votes.items():
        for entry, n in entries.items():
            key = (rank[tier], -n, -counts.get(entry, 0), entry)
            if word not in best or key < best[word][0]:
                best[word] = (key, dict(entry=entry, tier=tier, votes=n))
    return {w: v for w, (_, v) in sorted(best.items())}


# ---- the ASLLRP side ---------------------------------------------------------------------------------------------

def gloss_table(records, vocab, signbank, codes, semlex) -> dict:
    """{gloss: map_sign of its first occurrence}: the groundwork's table."""
    first = {}
    for r in records:
        for s in r["signs"]:
            first.setdefault(s["gloss"], s)
    return {g: agc.map_sign(first[g], vocab, signbank, codes, semlex) for g in sorted(first)}


def gold_entries(record, table) -> list[str]:
    return [e for s in record["signs"] if table[s["gloss"]]["tier"] in agc.WITH_INDEX
            for e in table[s["gloss"]]["entries"]]


def fit_rows(records, table) -> list[dict]:
    """The records the fit functions read: id, collection, partition, english, gold, seconds and the timed signs
    ((single entry or None, category, start, end))."""
    out = []
    for r in records:
        timed = []
        for s in r["signs"]:
            m = table[s["gloss"]]
            entry = m["entries"][0] if m["tier"] in agc.WITH_INDEX and len(m["entries"]) == 1 else None
            timed.append((entry, s["category"], s["start"], s["end"]))
        out.append(dict(id=r["id"], collection=r["collection"], partition=r["partition"], english=r["english"],
                        gold=gold_entries(r, table), seconds=r["seconds"], timed=timed))
    return out


def fit_stage2(train_rows, vocab, lexicon, annotated, parameters=None) -> dict:
    """The stage-2 model from ASLLRP train rows only (``check_fit_records`` stops anything else)."""
    p = dict(DEFAULTS, **(parameters or {}))
    cg.check_fit_records(train_rows)
    structure = cg.fit_structure(train_rows, vocab, lexicon, threshold=p["threshold"], min_count=p["min_count"],
                                 learned=p["learned"], move_min_count=p["move_min_count"],
                                 move_margin=p["move_margin"])
    timing = cg.fit_timing(train_rows, vocab, annotated, min_signs=p["min_signs"], floor_q=p["floor_q"],
                           ceiling_q=p["ceiling_q"])
    return dict(schema_version=1, parameters=p, lexicon=lexicon, structure=structure, timing=timing,
                train_utterances=len(train_rows))


def fit_model(store, mix_key: str, mix_sha256: str, corpora=ISOLATED_CORPORA, parameters=None) -> tuple[dict, dict]:
    """(model with its ``sha256``, the prepared tables): the inputs read and checked, the model fitted on ASLLRP
    train only. The evaluation run calls exactly this, so a model pinned by its hash is reproduced bit for bit."""
    parsed, shas = load_inputs(store, mix_key, mix_sha256)
    prep = prepare(parsed, corpora)
    model = fit_stage2(prep["train"], prep["vocab"], prep["lexicon"], prep["annotated"], parameters)
    model.update(inputs=dict(sorted(shas.items())), corpora=list(corpora), mix_key=mix_key)
    model["sha256"] = cg.model_sha256(model)
    return model, prep


def check_model(model: dict) -> None:
    """Stop unless ``model["sha256"]`` is the hash of the rest of the model."""
    body = {k: v for k, v in model.items() if k != "sha256"}
    if cg.model_sha256(body) != model.get("sha256"):
        raise SystemExit("the stage-2 model does not match its sha256")


def plan_sentence(text, model, vocab) -> tuple[list[str], list[dict], list[dict]]:
    """(entries, selection rows, structure trace) of one English sentence under a stage-2 model."""
    rows = cg.selection_rows(text, vocab, model["lexicon"])
    entries, trace = cg.apply_structure(rows, model["structure"], vocab)
    return entries, rows, trace


def load_inputs(store, mix_key: str, mix_sha256: str, keys=None) -> tuple[dict, dict]:
    """(payloads parsed, {key: sha256}) of every input; the mix is checked against ``mix_sha256`` and ASL-LEX
    against its verified acquisition hash."""
    keys = dict(KEYS, **(keys or {}))
    shas, data = {}, {}
    for name, key in list(keys.items()) + [("mix", mix_key)]:
        payload = store.get(key)
        shas[key] = hashlib.sha256(payload).hexdigest()
        data[name] = payload
    if shas[mix_key] != mix_sha256:
        raise SystemExit(f"{mix_key} does not match --mix-sha256")
    if shas[keys["asllex"]] != ASLLEX_SHA256:
        raise SystemExit("ASL-LEX signdata.csv is not the verified acquisition")
    parsed = dict(utterances=json.loads(data["utterances"])["items"], pairs=json.loads(data["pairs"])["pairs"],
                  citizen_map=json.loads(data["citizen_map"]), signbank_lexicon=json.loads(data["signbank_lexicon"]),
                  asllex=list(csv.DictReader(io.StringIO(data["asllex"].decode("utf-8", "replace")))),
                  manifest=json.loads(data["mix"]))
    return parsed, shas


def prepare(parsed, corpora=ISOLATED_CORPORA) -> dict:
    """Everything the fit and the checks need, from the parsed inputs: vocabulary, tables, lexicon, rows by
    partition (train and val; ``build_corpus`` never returns test)."""
    manifest = parsed["manifest"]
    vocab = training_vocabulary(manifest, corpora)
    signbank, _, codes, semlex = agc.vocabulary_indices(manifest, vocab, parsed["citizen_map"])
    records, join = agc.build_corpus(parsed["utterances"], parsed["pairs"])
    if any(r["partition"] not in agc.READ_PARTITIONS for r in records):
        raise AssertionError("build_corpus returned a partition it must not read")
    table = gloss_table(records, vocab, signbank, codes, semlex)
    rows = fit_rows(records, table)
    lexicon = build_lexicon(vocab, parsed["signbank_lexicon"], signbank, parsed["asllex"], semlex,
                            record_counts(manifest, vocab, corpora))
    return dict(vocab=vocab, table=table, records=records, join=join, lexicon=lexicon,
                annotated=annotated_seconds(manifest, vocab),
                train=[r for r in rows if r["partition"] == "train"], val=[r for r in rows if r["partition"] == "val"])


# ---- checks on ASLLRP val ----------------------------------------------------------------------------------------

def lexicon_precision(rows, vocab, lexicon) -> dict:
    """Per tier: lexicon-matched spans, and how often the utterance's gold contains the mapped entry."""
    out = collections.defaultdict(lambda: dict(spans=0, in_gold=0))
    for r in rows:
        gold = collections.Counter(r["gold"])
        for row in cg.selection_rows(r["english"], vocab, lexicon):
            if row["kind"] == "lexicon":
                out[row["tier"]]["spans"] += 1
                out[row["tier"]]["in_gold"] += int(gold[row["entry"]] > 0)
    return {t: dict(v, precision=round(v["in_gold"] / v["spans"], 4) if v["spans"] else None)
            for t, v in sorted(out.items())}


SUMMARY = ("bag_precision", "bag_recall", "bag_f1", "entry_wer", "tau_mean", "tau_below_1_share", "tau_np_mean",
           "tau_np_below_1_share", "words_per_utterance")


def timing_check(rows, model, vocab, const: float | None = None) -> dict:
    """The timing model's targets against ASLLRP sentence-sign durations of mapped signs (absolute error, seconds,
    and absolute log ratio), beside a constant baseline ``const`` (the train median of the same signs)."""
    t = model["timing"]
    err, err_c, logr, logr_c, sources = [], [], [], [], collections.Counter()
    for r in rows:
        for entry, cat, a, b in r["timed"]:
            if entry is None or a is None or b is None or b - a <= 0:
                continue
            target, src = cg.target_seconds(entry, t, vocab)
            d = b - a
            sources[src] += 1
            err.append(abs(target - d)); logr.append(abs(np.log(target / d)))
            if const:
                err_c.append(abs(const - d)); logr_c.append(abs(np.log(const / d)))
    return dict(signs=len(err), sources=dict(sources), mae_seconds=round(float(np.mean(err)), 4) if err else None,
                median_abs_log_ratio=round(float(np.median(logr)), 4) if logr else None,
                baseline_constant_seconds=round(const, 4) if const else None,
                baseline_mae_seconds=round(float(np.mean(err_c)), 4) if err_c else None,
                baseline_median_abs_log_ratio=round(float(np.median(logr_c)), 4) if logr_c else None)


def evaluate(rows, model, vocab, *, boot: int = 1000) -> dict:
    """The stage-1 rule and the stage-2 ablations on ``rows`` (module docstring)."""
    gold = [r["gold"] for r in rows]
    lex, st = model["lexicon"], model["structure"]
    forms = {
        "stage1": [select_words(r["english"], vocab) for r in rows],
        "stage2_selection": [cg.selected(cg.selection_rows(r["english"], vocab, lex)) for r in rows],
        "stage2_selection_no_lexicon": [cg.selected(cg.selection_rows(r["english"], vocab, {})) for r in rows],
        "stage2_tokens": [cg.apply_structure(cg.selection_rows(r["english"], vocab, lex), st, vocab, moves=False)[0]
                          for r in rows],
        "stage2_full": [cg.apply_structure(cg.selection_rows(r["english"], vocab, lex), st, vocab)[0] for r in rows],
    }
    out, per = {}, {}
    for name, pred in forms.items():
        out[name], per[name] = cg.sequence_metrics(pred, gold)
    for name in forms:
        if name != "stage1":
            out[name]["minus_stage1"] = cg.paired_bootstrap(per["stage1"], per[name], boot=boot)
    out["stage2_full"]["minus_stage2_tokens"] = cg.paired_bootstrap(per["stage2_tokens"], per["stage2_full"],
                                                                    boot=boot)
    ops = collections.Counter()
    for r in rows:
        _, trace = cg.apply_structure(cg.selection_rows(r["english"], vocab, lex), st, vocab)
        ops.update(t["op"] for t in trace)
        ops.update(f"moved_{t['moved']}" for t in trace if t.get("moved"))
    out["operations"] = dict(sorted(ops.items()))
    return out


def summary_tables(model) -> dict:
    st = model["structure"]
    acts = collections.Counter(a["op"] for a in st["tokens"].values())
    return dict(token_actions=dict(acts),
                dropped=sorted(t for t, a in st["tokens"].items() if a["op"] == "drop"),
                mapped={t: a["entry"] for t, a in st["tokens"].items() if a["op"] in ("map", "insert")},
                pointing=st["pointing"], moves=st["moves"],
                lexicon_tiers=dict(collections.Counter(v["tier"] for v in model["lexicon"].values())),
                lexicon_entries=len(model["lexicon"]))


# ---- main --------------------------------------------------------------------------------------------------------

def _write_once(path: str, payload) -> None:
    p = Path(path)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", default="r2", help="r2, or dir:<path> (read-only either way)")
    ap.add_argument("--mix-key", required=True, help="the prior's training mix (its vocabulary)")
    ap.add_argument("--mix-sha256", required=True)
    ap.add_argument("--corpora", default=",".join(ISOLATED_CORPORA))
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--sensitivity", default="0.3,0.7", help="other thresholds, refitted and reported only")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--model-file", required=True, help="the fitted model (write-once; local, ASLLRP terms)")
    ap.add_argument("--output-file", required=True, help="the report (write-once)")
    args = ap.parse_args(argv)
    for path in (args.model_file, args.output_file):
        if Path(path).exists():
            raise SystemExit(f"refusing to overwrite {path}")
    store = agc._store(args.store)
    corpora = tuple(c for c in args.corpora.split(",") if c)
    params = {k: getattr(args, k) for k in DEFAULTS}
    model, prep = fit_model(store, args.mix_key, args.mix_sha256, corpora, params)
    vocab = prep["vocab"]
    constant = float(np.median([b - a for r in prep["train"] for e, c, a, b in r["timed"]
                                if e is not None and a is not None and b is not None]))
    val = evaluate(prep["val"], model, vocab, boot=args.bootstrap)
    train = evaluate(prep["train"], model, vocab, boot=200)
    sensitivity = {}
    overrides = [("threshold", float(x)) for x in args.sensitivity.split(",") if x] + [("learned", 0)]
    for key, value in overrides:
        m = fit_stage2(prep["train"], vocab, prep["lexicon"], prep["annotated"], dict(params, **{key: value}))
        e = evaluate(prep["val"], m, vocab, boot=200)
        sensitivity[f"{key}={value}"] = {k: {f: e[k][f] for f in SUMMARY} for k in ("stage2_tokens", "stage2_full")}
        sensitivity[f"{key}={value}"]["stage2_full"]["minus_stage1"] = {
            f: e["stage2_full"]["minus_stage1"][f] for f in ("bag_precision", "bag_recall", "bag_f1")}
    reproduction = {part: {k: agc.rule_check([r for r in prep["records"] if r["partition"] == part], prep["table"],
                                             vocab)[k] for k in ("bag_precision", "bag_recall", "entry_wer")}
                    for part in ("train", "val")}
    report = dict(
        purpose="round 15 stage 2: grammar and timing fitted on ASLLRP train, checked on ASLLRP val (development "
                "data; no How2Sign sentence read; ASLLRP test never read)",
        model_sha256=model["sha256"], inputs=model["inputs"], mix_key=args.mix_key, corpora=list(corpora),
        join=prep["join"], vocabulary_entries=len(vocab), parameters=params,
        utterances=dict(train=len(prep["train"]), val=len(prep["val"])),
        reproduction_groundwork_rule_check=reproduction,
        val=val, train=train, sensitivity=sensitivity,
        lexicon_precision=dict(train=lexicon_precision(prep["train"], vocab, prep["lexicon"]),
                               val=lexicon_precision(prep["val"], vocab, prep["lexicon"])),
        timing={k: v for k, v in model["timing"].items() if k not in ("sentence_seconds", "sentence_signs",
                                                                      "annotated")},
        timing_entries=dict(asllrp=len(model["timing"]["sentence_seconds"]),
                            annotated=len(model["timing"]["annotated"])),
        timing_check=dict(train=timing_check(prep["train"], model, vocab, constant),
                          val=timing_check(prep["val"], model, vocab, constant)),
        structure=summary_tables(model), move_scores=model["structure"]["move_scores"])
    _write_once(args.model_file, model)
    _write_once(args.output_file, report)
    v = report["val"]
    print(json.dumps(dict(model_sha256=model["sha256"], reproduction=reproduction,
                          val={k: {f: v[k][f] for f in SUMMARY} for k in v if k != "operations"},
                          timing=report["timing"], timing_check=report["timing_check"]), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
