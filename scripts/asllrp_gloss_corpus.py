"""ASLLRP English-gloss corpus and its statistics: the data groundwork for round 15 stage 2 (docs/NEXT_PHASE_PLAN.md,
"Round 15 stage 2, planned", 18:23 UTC, and "Oracle composition result", 20:11 UTC). Measurement only: nothing is
trained and nothing is generated, so no number here is a stage-2 result.

**Inputs** (read-only; every payload's SHA-256 is recorded):
- ``derived/signbank-v1/utterances.json`` (``build_signbank_lexicon.py``): each DAI 2 continuous utterance's signs in
  time order, from the Sign Bank sentence-sign CSVs, with main-entry gloss, variant, occurrence label, sign type and the
  start/end frame of every sign and of the utterance (frames of the master video);
- ``derived/asllrp-dai2-v1/english-gloss-v1/pairs.json`` (``build_asllrp_english_gloss.py``): the English translation,
  the dominant-hand surface tier and the grammatical non-manual markers of each utterance, with its signer-disjoint
  partition (val Jonathan, test Ben, train the others);
- the generator's training mix ``round14-mixed-v1`` (``--mix-key`` / ``--mix-sha256``): the isolated-sign vocabulary and
  each word's typical isolated duration (``compositional.training_vocabulary``: the one-word train records of
  asllvd-text-v1, rit-text-r6, wlasl-text-r6 and semlex-text-v1, median trimmed ``continuous.seconds``), the records'
  Sign Bank glosses (ASLLVD, RIT) and ASL-LEX codes (Sem-Lex), and the How2Sign val English;
- ``derived/signbank-v1/asl-citizen-gloss-map.json``: Sign Bank main entry -> ASL-LEX code, through the ASL Citizen
  glosses whose map method is ``exact`` (the only direct label match);
- optionally the stored round-15 stage-1 report (``--stage1-report-key``): its fitted word durations.

**The test partition (Ben) is counted, never read**: no statistic, corpus record or mapping uses it.

**Corpus** (``--corpus-file``, local and write-once; ASLLRP research terms, never uploaded): one record per train/val
utterance with its English, signer, source, partition, the ordered signs (gloss, variant, occurrence, sign type,
category, start/end seconds from the utterance's first frame), the surface tier and the markers; and the gloss ->
vocabulary mapping table. Frame rates are the corpora's nominal rates (ASLLRP 30000/1001, RIT sentences 60000/1001;
the extracted ASLLRP utterance states measure 29.970 fps as (frames - 1) / seconds). A sign lasts
``(end - start + 1) / fps``; the gap after it is ``(next start - end - 1) / fps``.

**Alignment** (``align``, simple matching): each gloss's English forms (``english_forms``: the gloss name through the
citation-stitch lookup alternatives and the corpora's gloss-to-English rule, slash alternatives and compound parts;
pronoun forms for indices by the occurrence's person and number; the spelled word of a fingerspelling or name sign;
none for classifiers and gestures; plus a declared 13-entry table of frequent function signs, ``SYNONYMS``, e.g.
FUTURE "will", reported with and without it) are matched to the English tokens (``english_tokens``: ``normalise`` with
contractions split, ``don't`` -> do not), a token matching a form exactly or through the round-15 inflection fallback
(``voices`` -> ``voice``). Glosses are taken in signing order; each takes
one unused matching token, the one closest to its proportional position. Kendall's tau between the glosses' order and
their aligned tokens' order is computed per utterance with at least two aligned glosses (no ties: alignment is
one-to-one). An unaligned English token is a "dropped" word; an unaligned gloss has no English word.

**Mapping** (``map_sign``), tiers in order, the first hit taken:
1. ``signbank_id``: the main-entry gloss is the Sign Bank gloss of ASLLVD / RIT train records (their ``gloss.label``):
   the records' most common vocabulary entry;
2. ``asl_lex_code``: the main entry's ASL-LEX codes (exact ASL Citizen map) are Sem-Lex train records' codes;
3. ``name`` / ``name_compound``: gloss-name normalisation (handshape prefixes, hold underscores, variant numbers,
   repetition and ``wg`` suffixes, ``-indef``; slash alternatives; compounds as their parts, every part an entry);
4. ``index_rule``: an index (``IX``, ``POSS``, ``SELF``) by its occurrence's person and number to a pronoun entry;
5. fallbacks, reported apart: ``lexicalised_fs`` (``#IF`` -> "if"), ``fingerspelled_word`` (``fs-OF`` -> the sign
   "of": a sign in place of the spelling).
Classifiers, gestures and name signs have no tier. Entries are matched as round 15 does: present in the vocabulary
and never a single letter.

**Timing:** continuous sign durations against the same signs' isolated durations (the mapped entry's typical seconds,
what a composition would be asked; and, for ``signbank_id`` signs, the median of the same Sign Bank gloss's ASLLVD /
RIT train clips); gaps between signs; sentence-level compression (utterance span over the sum of its signs' isolated
typical seconds); and round 15's fitted word durations.

**Rule check:** round 15 stage 1's word selection (``select_words``: English order, stop words dropped) applied to
ASLLRP English, against the ASLLRP gloss sequence mapped to vocabulary entries: bag precision / recall, entry-sequence
WER and the order agreement of the shared entries.

**Generalisation:** the English vocabulary of ASLLRP (train; train + val) against How2Sign val English (all val records
of the mix, and the round-15 pilot's 300 sentences), by type and by token (``english_tokens``), with and without stop
words, and the English types ASLLRP aligns to a gloss; and repeated sentences (ASLLRP elicits the same English from
several signers: unique sentences, and val sentences whose English is in train).

Writes the report to ``--output-file`` (write-once) and prints a short summary.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from build_asllrp_gloss_manifest import signer as signer_of  # noqa: E402
from build_citizen_text_manifest import english as gloss_english  # noqa: E402
from signtranslator.generation.citation_stitch import lookup_names  # noqa: E402
from signtranslator.generation.compositional import (ISOLATED_CORPORA, STOP_WORDS, inflection_candidates,  # noqa: E402
                                                     normalise, select_words, training_vocabulary)

ACQ = "acquisition/2026-10-17-v1/"
FPS = {"asllrp-continuous": 30000 / 1001, "rit-sentences": 60000 / 1001}
ASLLVD_FPS = 60.                           # ASLLVD clips: 50 frames (0.833 s) of context either side of the sign
READ_PARTITIONS = ("train", "val")
H2S_CORPUS = "how2sign-two-view-text-v1"
SIGNBANK_CORPORA = ("asllvd-text-v1", "rit-text-r6")
SEMLEX_CORPUS = "semlex-text-v1"
STAGE1_FITTED_PLAN = (0.65, 0.74)          # the brief's figure for round 15's fitted word durations (median, seconds)
STRICT = ("signbank_id", "asl_lex_code", "name", "name_compound")
WITH_INDEX = STRICT + ("index_rule",)
WITH_FALLBACKS = WITH_INDEX + ("lexicalised_fs", "fingerspelled_word")
TIERS = WITH_FALLBACKS
CATEGORIES = ("lexical", "index", "fingerspelled", "lexicalised_fs", "name_sign", "classifier", "gesture", "number",
              "compound", "loan")

# English word classes for the dropped-word report (a token may be in one class only; first listed wins)
WORD_CLASSES = {
    "article": {"a", "an", "the"},
    "copula": {"am", "is", "are", "was", "were", "be", "been", "being", "'s", "'re", "'m"},
    "auxiliary": {"do", "does", "did", "have", "has", "had", "will", "would", "shall"},
    "preposition": {"to", "of", "in", "on", "at", "for", "with", "from", "by", "about", "into", "as", "over", "up",
                    "out", "off", "than"},
    "conjunction": {"and", "or", "but", "so", "that", "because", "if", "when"},
    "pronoun": {"i", "me", "my", "mine", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her", "it",
                "its", "they", "them", "their", "this", "these", "those", "there", "here", "myself", "himself",
                "herself", "itself", "themselves", "yourself"},
}

# Pronoun forms of an index, by (kind, person, plural); person None = unspecified (main entry IX without person)
_THIRD = ("he", "she", "it", "him", "her", "this", "that")
PRONOUNS = {
    ("IX", "1p", False): ("me", "i"), ("IX", "1p", True): ("we", "us"),
    ("IX", "2p", False): ("you",), ("IX", "2p", True): ("you",),
    ("IX", "3p", False): _THIRD, ("IX", "3p", True): ("they", "them", "those", "these"),
    ("IX", "loc", False): ("there", "here"), ("IX", "loc", True): ("there", "here"),
    ("IX", None, False): _THIRD + ("there", "here"), ("IX", None, True): ("they", "them", "those", "these"),
    ("POSS", "1p", False): ("my", "mine"), ("POSS", "1p", True): ("our", "ours"),
    ("POSS", "2p", False): ("your", "yours"), ("POSS", "2p", True): ("your", "yours"),
    ("POSS", "3p", False): ("his", "her", "its", "hers"), ("POSS", "3p", True): ("their", "theirs"),
    ("POSS", None, False): ("his", "her", "its", "hers"), ("POSS", None, True): ("their", "theirs"),
    ("SELF", "1p", False): ("myself",), ("SELF", "1p", True): ("ourselves",),
    ("SELF", "2p", False): ("yourself",), ("SELF", "2p", True): ("yourselves",),
    ("SELF", "3p", False): ("himself", "herself", "itself"), ("SELF", "3p", True): ("themselves",),
    ("SELF", None, False): ("himself", "herself", "itself"), ("SELF", None, True): ("themselves",),
}
_PREFIX = re.compile(r"^(?:\([^)]*\)(?:alt\.)?)+")

# Alignment only (never the mapping): English forms of frequent function signs that their gloss names do not spell.
# Declared before the full run was read; the plain alignment (without them) is reported beside it.
SYNONYMS = {
    "FUTURE": ("will", "later"), "FINISH": ("already", "finished", "done"), "NONE/NOTHING": ("no",),
    "MOTHER": ("mom",), "FATHER": ("dad",), "MOTHER+FATHER": ("parents",), "GO-OUT": ("leave", "left", "out"),
    "DEPART": ("leave", "left"), "TEACH+AGENT": ("teacher",), "WHY": ("because",), "HAVE": ("has", "had"),
    "GO": ("went", "gone"), "CANNOT": ("can",),
}
_CONTRACTIONS = (("n't", "not"), ("'s", "'s"), ("'m", "am"), ("'re", "are"), ("'ll", "will"), ("'ve", "have"),
                 ("'d", "would"))
_IRREGULAR_NOT = {"won't": ("will", "not"), "can't": ("can", "not"), "shan't": ("shall", "not")}


# ---- parsing -----------------------------------------------------------------------------------------------------

def category(gloss: str, sign_type: str = "") -> str:
    """The sign's category from its main-entry gloss (and the Sign Bank sign type where the gloss is silent)."""
    g = gloss or ""
    if g.startswith("ns-"):
        return "name_sign"
    if g.startswith("fs-"):
        return "fingerspelled"
    if g.startswith("#"):
        return "lexicalised_fs"
    if re.match(r"^[A-Z]*CL(?:[:\\]|$)", g) or sign_type == "Classifiers":
        return "classifier"
    if '"' in g or "\\" in g or sign_type == "Gestures":
        return "gesture"
    if re.match(r"^(?:IX|POSS|SELF)(?:-|$)", g) or g.startswith("part:"):
        return "index"
    return {"Number Signs": "number", "Compound Signs": "compound", "Loan Signs": "loan"}.get(sign_type, "lexical")


def index_key(gloss: str, occurrence: str = "") -> tuple | None:
    """(kind, person, plural) of an index sign from its occurrence label (``(2h)IX-3p-pl-arc`` -> ("IX", "3p", True)),
    the main-entry gloss when the occurrence names no index; None for a non-index."""
    for label in (occurrence, gloss):
        t = _PREFIX.sub("", (label or "").strip())
        t = re.sub(r"^THUMB-", "", t, flags=re.IGNORECASE)
        m = re.match(r"^(IX|POSS|SELF)(?:-|$|\+|:)", t, flags=re.IGNORECASE)
        if m:
            low = t.lower()
            person = re.search(r"(1p|2p|3p|loc)", low)
            return m.group(1).upper(), person.group(1) if person else None, "-pl" in low
    return None


def english_entry(name: str) -> str | None:
    """A gloss name's English entry, written as the isolated corpora write theirs (``normalise``d words), or None.
    The corpora's own gloss-to-English rule (``build_citizen_text_manifest.english``) after removing what it does not:
    the citation-stitch pointing names (``citizen:ME``), handshape prefixes, repetition / ``wg`` suffixes, hold
    underscores with a variant number (``BELOW_2``), ``-indef`` and ``-arc``."""
    n = name.removeprefix("citizen:")
    n = _PREFIX.sub("", n)
    n = re.sub(r"(?:\+|wg)+$", "", n)
    n = re.sub(r"_+\d*$", "", n)
    n = re.sub(r"-(?:indef|arc)$", "", n)
    e = gloss_english(n) if n else None
    e = " ".join(normalise(e)) if e else ""
    return e or None


def name_candidates(gloss: str) -> list[tuple[str, ...]]:
    """Alternatives of a lexical gloss as vocabulary entries, most faithful first; a compound is one alternative of
    several entries (``PAST__+NIGHT`` -> ("past", "night")). Fingerspelling, name signs, lexicalised fingerspelling,
    classifiers and gestures have none (their own tiers or no tier)."""
    if category(gloss) in ("fingerspelled", "name_sign", "classifier", "gesture", "lexicalised_fs"):
        return []
    out = []
    for names in lookup_names(gloss):
        if any(n.startswith("citizen:") for n in names):
            continue                           # the stitch's pointing stand-ins: indices have their own tier here
        entries = tuple(english_entry(n) for n in names)
        if entries and all(entries) and entries not in out:
            out.append(entries)
    return out


def spelled_word(gloss: str) -> str | None:
    """The word a fingerspelling, name sign or lexicalised fingerspelling spells (``ns-fs-JOHN`` -> "john")."""
    m = re.match(r"^(?:ns-)?(?:fs-|#)?(.+)$", gloss or "")
    if not m or category(gloss) not in ("fingerspelled", "name_sign", "lexicalised_fs"):
        return None
    words = normalise(m.group(1).replace("-", " "))
    return " ".join(words) or None


def english_forms(sign: dict, synonyms: bool = True) -> list[tuple[str, ...]]:
    """The English token sequences a sign may align to (module docstring); ``synonyms`` adds ``SYNONYMS``."""
    gloss, cat = sign["gloss"], sign.get("category") or category(sign["gloss"], sign.get("sign_type", ""))
    forms: list[tuple[str, ...]] = []
    if cat == "index":
        key = index_key(gloss, sign.get("occurrence", ""))
        forms += [(w,) for w in PRONOUNS.get(key, ())]
    if cat in ("fingerspelled", "name_sign", "lexicalised_fs"):
        w = spelled_word(gloss)
        if w:
            forms.append(tuple(w.split()))
    if cat == "lexicalised_fs":
        for alt in name_candidates(gloss.lstrip("#")):
            forms += [tuple(e.split()) for e in alt]
    for alt in name_candidates(gloss):
        forms += [tuple(e.split()) for e in alt]
    if synonyms:
        forms += [(w,) for w in SYNONYMS.get(gloss, ())]
    return list(dict.fromkeys(f for f in forms if f))


def english_tokens(text: str) -> list[str]:
    """``normalise``d English with contractions split for alignment (``don't`` -> do not, ``it's`` -> it 's,
    ``won't`` -> will not), so negation and copulas are visible to the matching and the dropped-word counts."""
    out = []
    for t in normalise(text):
        if t in _IRREGULAR_NOT:
            out += _IRREGULAR_NOT[t]
            continue
        for suffix, word in _CONTRACTIONS:
            if t.endswith(suffix) and len(t) > len(suffix):
                out += [t[:-len(suffix)], word]
                break
        else:
            out.append(t)
    return out


def token_matches(token: str, word: str) -> bool:
    return token == word or word in inflection_candidates(token)


def align_spans(tokens: list[str], forms_per_sign: list[list[tuple[str, ...]]]) -> list[tuple[int, int] | None]:
    """One English token span (start, length) per sign, or None; one-to-one (module docstring)."""
    used: set[int] = set()
    out: list[tuple[int, int] | None] = []
    n_g, n_e = len(forms_per_sign), len(tokens)
    for g, forms in enumerate(forms_per_sign):
        cands = []
        for form in forms:
            span = len(form)
            for p in range(n_e - span + 1):
                if any(q in used for q in range(p, p + span)):
                    continue
                if all(token_matches(tokens[p + i], form[i]) for i in range(span)):
                    cands.append((p, span))
        if not cands:
            out.append(None)
            continue
        expected = (g + .5) / n_g * n_e
        p, span = min(cands, key=lambda c: (abs(c[0] + c[1] / 2 - expected), c[0], -c[1]))
        used.update(range(p, p + span))
        out.append((p, span))
    return out


def align(tokens: list[str], forms_per_sign: list[list[tuple[str, ...]]]) -> list[int | None]:
    """One English token position (a multi-word form's first) per sign, or None (``align_spans``)."""
    return [None if x is None else x[0] for x in align_spans(tokens, forms_per_sign)]


def kendall_tau(positions) -> tuple[float, int, int] | None:
    """(tau, concordant, discordant) of a sequence of distinct positions against its own index order; None below 2."""
    xs = [p for p in positions if p is not None]
    n = len(xs)
    if n < 2:
        return None
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            if xs[j] > xs[i]:
                c += 1
            elif xs[j] < xs[i]:
                d += 1
    return (c - d) / (n * (n - 1) / 2), c, d


def build_corpus(utterances: list[dict], pairs: list[dict]) -> tuple[list[dict], dict]:
    """Train/val corpus records (module docstring) and the join's counts. Test pairs are counted, never read further."""
    by_key = {(u["source"], u["utterance"]): u for u in utterances}
    records, counts = [], collections.Counter()
    for p in pairs:
        counts[f"pairs_{p['partition']}"] += 1
        if p["partition"] not in READ_PARTITIONS:
            continue
        u = by_key.get((p["source"], p["utterance"]))
        if u is None:
            counts["no_utterance_record"] += 1
            continue
        fps = FPS[u["collection"]]
        u0, u1 = u["frames"]
        signs = []
        for s in u["signs"]:
            if not s.get("gloss"):
                counts["signs_without_gloss"] += 1
                continue
            a, b = (s.get("frames") or [None, None])[:2]
            timed = None not in (a, b, u0, u1) and u0 <= a <= b <= u1
            if not timed:
                counts["signs_untimed"] += 1
            signs.append(dict(gloss=s["gloss"], variant=s.get("variant", ""), occurrence=s.get("occurrence", ""),
                              sign_type=s.get("sign_type", ""), category=category(s["gloss"], s.get("sign_type", "")),
                              start=round((a - u0) / fps, 4) if timed else None,
                              end=round((b - u0 + 1) / fps, 4) if timed else None))
        if [s["gloss"] for s in signs] != list(p["glosses"]):
            counts["gloss_sequence_mismatch"] += 1
        if None in (u0, u1):
            counts["utterance_untimed"] += 1
        records.append(dict(id=p["id"], collection=u["collection"], source=u["source"], utterance=u["utterance"],
                            signer=signer_of(u["source"]), partition=p["partition"], english=p["english"],
                            fps=round(fps, 5), seconds=round((u1 - u0 + 1) / fps, 4) if None not in (u0, u1) else None,
                            signs=signs, surface=list(p.get("surface") or []), markers=list(p.get("markers") or [])))
    return records, dict(sorted(counts.items()))


# ---- vocabulary and mapping --------------------------------------------------------------------------------------

def _matchable(entry: str | None, vocab) -> bool:
    return bool(entry) and len(entry) > 1 and entry in vocab


def vocabulary_indices(manifest: dict, vocab: dict, citizen_map: dict):
    """(Sign Bank gloss -> Counter(entry), Sign Bank gloss -> {corpus: [seconds]}, main entry -> ASL-LEX codes,
    ASL-LEX code -> Counter(entry)) from the mix's isolated train records (the vocabulary's records). The seconds are
    the trimmed clip lengths, plus, under ``"asllvd-annotated-sign"``, the ASLLVD clips' annotated sign spans
    (``source.frames`` at 60 fps: the same SignStream convention as the continuous spans)."""
    from signtranslator.data_engineering.corpus_manifest import usable_channels
    signbank, signbank_seconds, semlex = (collections.defaultdict(collections.Counter),
                                          collections.defaultdict(lambda: collections.defaultdict(list)),
                                          collections.defaultdict(collections.Counter))
    for r in manifest["records"]:
        corpus = r.get("corpus")
        if corpus not in SIGNBANK_CORPORA + (SEMLEX_CORPUS,) or r.get("partition") != "train" or not r.get("text"):
            continue
        if not usable_channels(r, "continuous"):
            continue
        entry = " ".join(normalise(r["text"]))
        if not _matchable(entry, vocab):
            continue
        gloss = r.get("gloss") or {}
        if corpus in SIGNBANK_CORPORA and gloss.get("label"):
            signbank[gloss["label"]][entry] += 1
            s = (r.get("continuous") or {}).get("seconds")
            if s:
                signbank_seconds[gloss["label"]][corpus].append(float(s))
            frames = (r.get("source") or {}).get("frames")
            if corpus == "asllvd-text-v1" and frames and len(frames) == 2 and frames[1] >= frames[0]:
                signbank_seconds[gloss["label"]]["asllvd-annotated-sign"].append((frames[1] - frames[0] + 1) / ASLLVD_FPS)
        elif corpus == SEMLEX_CORPUS and gloss.get("asl_lex_code"):
            semlex[gloss["asl_lex_code"]][entry] += 1
    codes = collections.defaultdict(set)
    for m in (citizen_map.get("mapping") or {}).values():
        if m.get("method") == "exact" and m.get("asl_lex_code"):
            for main in m.get("candidates", ()):
                codes[main].add(m["asl_lex_code"])
    return dict(signbank), {k: dict(v) for k, v in signbank_seconds.items()}, dict(codes), dict(semlex)


def _top(counter: collections.Counter) -> str:
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def map_sign(sign: dict, vocab, signbank: dict, codes: dict, semlex: dict) -> dict:
    """{"tier", "entries"} of one sign (module docstring); tier None and entries () when unmapped."""
    gloss = sign["gloss"]
    cat = sign.get("category") or category(gloss, sign.get("sign_type", ""))
    if gloss in signbank:
        return dict(tier="signbank_id", entries=(_top(signbank[gloss]),))
    hits = collections.Counter()
    for code in sorted(codes.get(gloss, ())):
        hits.update(semlex.get(code, {}))
    if hits:
        return dict(tier="asl_lex_code", entries=(_top(hits),))
    for alt in name_candidates(gloss):
        if all(_matchable(e, vocab) for e in alt):
            return dict(tier="name" if len(alt) == 1 else "name_compound", entries=alt)
    if cat == "index":
        for w in PRONOUNS.get(index_key(gloss, sign.get("occurrence", "")), ()):
            if _matchable(w, vocab):
                return dict(tier="index_rule", entries=(w,))
    if cat == "lexicalised_fs":
        for alt in [(spelled_word(gloss),)] + name_candidates(gloss.lstrip("#")):
            if all(_matchable(e, vocab) for e in alt):
                return dict(tier="lexicalised_fs", entries=alt)
    if cat == "fingerspelled":
        w = spelled_word(gloss)
        if _matchable(w, vocab):
            return dict(tier="fingerspelled_word", entries=(w,))
    return dict(tier=None, entries=())


# ---- statistics --------------------------------------------------------------------------------------------------

def summary(values) -> dict | None:
    v = sorted(float(x) for x in values)
    if not v:
        return None

    def q(f):
        return round(statistics.quantiles(v, n=100, method="inclusive")[int(f * 100) - 1], 4) if len(v) > 1 else v[0]
    return dict(n=len(v), mean=round(statistics.fmean(v), 4), median=round(statistics.median(v), 4),
                p10=q(.10), p25=q(.25), p75=q(.75), p90=q(.90))


def _share(a, b):
    return round(a / b, 4) if b else None


def word_class(token: str) -> str:
    return next((k for k, words in WORD_CLASSES.items() if token in words), "other")


def tau_bin(t: float) -> str:
    return ("below 0" if t < 0 else "[0, 0.5)" if t < .5 else "[0.5, 0.8)" if t < .8 else "[0.8, 1)"
            if t < 1 - 1e-9 else "1 (English order)")


def alignment_statistics(records: list[dict], synonyms: bool = True) -> dict:
    n_glosses, n_words, taus, conc, disc, few = [], [], [], 0, 0, 0
    dropped, word_total = collections.Counter(), collections.Counter()
    gloss_un, gloss_tot = collections.Counter(), collections.Counter()
    cat_un, cat_tot = collections.Counter(), collections.Counter()
    aligned_signs = aligned_tokens = total_signs = total_tokens = 0
    for r in records:
        tokens = english_tokens(r["english"])
        spans = align_spans(tokens, [english_forms(s, synonyms) for s in r["signs"]])
        pos = [None if x is None else x[0] for x in spans]
        n_glosses.append(len(r["signs"])); n_words.append(len(tokens))
        covered = set()
        for s, x in zip(r["signs"], spans):
            gloss_tot[s["gloss"]] += 1; cat_tot[s["category"]] += 1
            if x is None:
                gloss_un[s["gloss"]] += 1; cat_un[s["category"]] += 1
            else:
                covered.update(range(x[0], x[0] + x[1]))
        total_signs += len(pos); aligned_signs += sum(p is not None for p in pos)
        total_tokens += len(tokens); aligned_tokens += len(covered)
        for i, t in enumerate(tokens):
            word_total[t] += 1
            if i not in covered:
                dropped[t] += 1
        k = kendall_tau(pos)
        if k is None:
            few += 1
        else:
            taus.append(k[0]); conc += k[1]; disc += k[2]
    classes = collections.defaultdict(lambda: [0, 0])
    for t, n in word_total.items():
        classes[word_class(t)][0] += n
        classes[word_class(t)][1] += dropped[t]
    total_dropped = sum(dropped.values())
    hist = collections.Counter(tau_bin(t) for t in taus)
    return dict(
        utterances=len(records), signs=total_signs, english_tokens=total_tokens,
        glosses_per_utterance=summary(n_glosses), english_words_per_utterance=summary(n_words),
        glosses_per_english_word=_share(total_signs, total_tokens),
        aligned_share_of_glosses=_share(aligned_signs, total_signs),
        aligned_share_of_english_tokens=_share(aligned_tokens, total_tokens),
        order=dict(
            utterances_with_two_aligned=len(taus), utterances_below_two_aligned=few,
            kendall_tau=summary(taus), share_tau_below_1=_share(sum(t < 1 - 1e-9 for t in taus), len(taus)),
            share_tau_at_most_0_5=_share(sum(t <= .5 for t in taus), len(taus)),
            share_tau_negative=_share(sum(t < 0 for t in taus), len(taus)),
            discordant_pair_share=_share(disc, conc + disc), tau_histogram=dict(sorted(hist.items()))),
        dropped_english=dict(
            share_of_tokens=_share(total_dropped, total_tokens),
            by_class={k: dict(tokens=v[0], dropped=v[1], dropped_rate=_share(v[1], v[0]),
                              share_of_all_dropped=_share(v[1], total_dropped))
                      for k, v in sorted(classes.items(), key=lambda kv: -kv[1][1])},
            stop_words_share_of_dropped=_share(sum(dropped[t] for t in STOP_WORDS), total_dropped),
            top=[dict(word=t, dropped=n, occurrences=word_total[t], rate=_share(n, word_total[t]))
                 for t, n in sorted(dropped.items(), key=lambda kv: (-kv[1], kv[0]))[:50]]),
        glosses_without_english=dict(
            share_of_signs=_share(sum(gloss_un.values()), total_signs),
            by_category={c: dict(signs=cat_tot[c], unaligned=cat_un[c], unaligned_rate=_share(cat_un[c], cat_tot[c]),
                                 share_of_all_unaligned=_share(cat_un[c], sum(gloss_un.values())))
                         for c in CATEGORIES if cat_tot[c]},
            top=[dict(gloss=g, unaligned=n, occurrences=gloss_tot[g], category=category(g))
                 for g, n in sorted(gloss_un.items(), key=lambda kv: (-kv[1], kv[0]))[:50]]))


def mapping_statistics(records, table: dict) -> dict:
    tok, typ = collections.Counter(), collections.Counter()
    unmapped, unmapped_cat, cats = collections.Counter(), collections.Counter(), {}
    for r in records:
        for s in r["signs"]:
            tok[table[s["gloss"]]["tier"]] += 1
            cats[s["gloss"]] = s["category"]
            if table[s["gloss"]]["tier"] is None:
                unmapped[s["gloss"]] += 1; unmapped_cat[s["category"]] += 1
    for g in cats:
        typ[table[g]["tier"]] += 1
    n_tok, n_typ = sum(tok.values()), sum(typ.values())

    def cover(tiers):
        return dict(by_token=_share(sum(tok[t] for t in tiers), n_tok), by_type=_share(sum(typ[t] for t in tiers), n_typ))
    return dict(
        tokens=n_tok, types=n_typ,
        tiers={str(t): dict(tokens=tok[t], types=typ[t], token_share=_share(tok[t], n_tok),
                            type_share=_share(typ[t], n_typ)) for t in TIERS + (None,)},
        coverage=dict(strict=cover(STRICT), with_index=cover(WITH_INDEX), with_fallbacks=cover(WITH_FALLBACKS)),
        unmapped_by_category={c: dict(tokens=n, share_of_unmapped=_share(n, sum(unmapped_cat.values())))
                              for c, n in unmapped_cat.most_common()},
        top_unmapped=[dict(gloss=g, tokens=n, category=cats[g]) for g, n in
                      sorted(unmapped.items(), key=lambda kv: (-kv[1], kv[0]))[:50]])


def name_tier_agreement(glosses, vocab, signbank) -> dict:
    """For glosses with a Sign Bank ID match, how often the name tier alone gives the same entry (it validates the
    gloss-name normalisation that the other glosses rely on)."""
    same = differ = none = 0
    examples = []
    for g in sorted(glosses):
        if g not in signbank:
            continue
        alt = next((a for a in name_candidates(g) if all(_matchable(e, vocab) for e in a)), None)
        if alt is None:
            none += 1
        elif alt == (_top(signbank[g]),):
            same += 1
        else:
            differ += 1
            if len(examples) < 20:
                examples.append(dict(gloss=g, signbank=_top(signbank[g]), name=" + ".join(alt)))
    n = same + differ + none
    return dict(glosses=n, same=same, differ=differ, name_unmapped=none, same_share=_share(same, n),
                differ_examples=examples)


ANNOTATED = "asllvd-annotated-sign"


def trimmed_over_annotated(signbank_seconds) -> dict | None:
    """ASLLVD: the trimmed clip the prior trained on over the annotated sign span, per gloss (medians)."""
    out = [statistics.median(v["asllvd-text-v1"]) / statistics.median(v[ANNOTATED])
           for v in signbank_seconds.values() if v.get("asllvd-text-v1") and v.get(ANNOTATED)]
    return summary(out)


def timing_statistics(records, table, vocab, signbank_seconds, stage1_timing=None) -> dict:
    durations, by_cat = [], collections.defaultdict(list)
    gaps, lead, trail, slots, rates = [], [], [], [], []
    ratio_entry, ratio_same, ratio_same_by = [], [], collections.defaultdict(list)
    ratio_slot, sentence_ratio, per_type = [], [], collections.defaultdict(list)
    annotated_iso, slot_annotated = [], []
    for r in records:
        timed = [s for s in r["signs"] if s["start"] is not None]
        timed.sort(key=lambda s: s["start"])
        for k, s in enumerate(timed):
            d = s["end"] - s["start"]
            durations.append(d); by_cat[s["category"]].append(d)
            gap = timed[k + 1]["start"] - s["end"] if k + 1 < len(timed) else None
            if gap is not None:
                gaps.append(gap); slots.append(d + gap)
            m = table[s["gloss"]]
            if m["tier"] in WITH_INDEX:
                iso = sum(vocab[e] for e in m["entries"])
                ratio_entry.append(d / iso); per_type[s["gloss"]].append(d / iso)
                if gap is not None:
                    ratio_slot.append((d + gap) / iso)
            if m["tier"] == "signbank_id" and s["gloss"] in signbank_seconds:
                clips = signbank_seconds[s["gloss"]]
                pooled = [x for c, v in clips.items() if c != ANNOTATED for x in v]
                if pooled:
                    ratio_same.append(d / statistics.median(pooled))
                for corpus, v in clips.items():
                    ratio_same_by[corpus].append(d / statistics.median(v))
                if ANNOTATED in clips:
                    annotated_iso.append(statistics.median(clips[ANNOTATED]))
                    if gap is not None:
                        slot_annotated.append((d + gap) / statistics.median(clips[ANNOTATED]))
        if timed and r["seconds"]:
            lead.append(timed[0]["start"]); trail.append(r["seconds"] - timed[-1]["end"])
            span = timed[-1]["end"] - timed[0]["start"]
            if span > 0:
                rates.append(len(timed) / span)
            if len(timed) == len(r["signs"]) and all(table[s["gloss"]]["tier"] in WITH_INDEX for s in timed) and span > 0:
                sentence_ratio.append(span / sum(vocab[e] for s in timed for e in table[s["gloss"]]["entries"]))
    gap_n = len(gaps)
    out = dict(
        sign_seconds=summary(durations),
        sign_seconds_by_category={c: summary(v) for c, v in sorted(by_cat.items())},
        gap_seconds=summary(gaps),
        gaps=dict(n=gap_n, overlapping_share=_share(sum(g < 0 for g in gaps), gap_n),
                  at_least_0_2_share=_share(sum(g >= .2 for g in gaps), gap_n),
                  at_least_0_5_share=_share(sum(g >= .5 for g in gaps), gap_n),
                  at_least_1_0_share=_share(sum(g >= 1. for g in gaps), gap_n)),
        slot_seconds_sign_plus_following_gap=summary(slots),
        leading_seconds=summary(lead), trailing_seconds=summary(trail),
        signs_per_second_over_signed_span=summary(rates),
        compression=dict(
            note="continuous sign seconds / isolated seconds of the same sign (below 1: shorter in sentences). The "
                 "isolated seconds are the trimmed clip lengths the prior trained on (dominant-wrist window with "
                 "0.25 s margins), the continuous spans the annotated sign without its transitions.",
            vs_vocabulary_entry_typical=summary(ratio_entry),
            vs_vocabulary_entry_typical_per_gloss_type_median=summary(statistics.median(v) for v in per_type.values()),
            vs_same_signbank_gloss_clips=summary(ratio_same),
            vs_same_signbank_gloss_clips_by_corpus={c: summary(v) for c, v in sorted(ratio_same_by.items())},
            same_convention_note=f"by_corpus[{ANNOTATED!r}]: continuous span / the same gloss's ASLLVD annotated "
                                 "sign span (both SignStream sign boundaries, no transitions, no margins): the "
                                 "like-for-like compression of the sign itself",
            asllvd_annotated_sign_seconds_of_these_tokens=summary(annotated_iso),
            slot_vs_asllvd_annotated_sign=summary(slot_annotated),
            slot_vs_vocabulary_entry_typical=summary(ratio_slot),
            sentence_span_vs_sum_of_isolated_typical=summary(sentence_ratio),
            asllvd_trimmed_clip_over_annotated_sign_per_gloss=trimmed_over_annotated(signbank_seconds)))
    sign_med = (out["sign_seconds"] or {}).get("median")
    slot_med = (out["slot_seconds_sign_plus_following_gap"] or {}).get("median")
    out["round15_comparison"] = dict(
        plan_fitted_word_seconds_median=list(STAGE1_FITTED_PLAN),
        stage1_report_fitted_word_seconds_median={k: v.get("fitted_word_seconds_median")
                                                  for k, v in (stage1_timing or {}).items()} or None,
        asllrp_sign_seconds_median=sign_med, asllrp_slot_seconds_median=slot_med,
        fitted_over_asllrp_sign={k: round(v.get("fitted_word_seconds_median") / sign_med, 3)
                                 for k, v in (stage1_timing or {}).items()
                                 if sign_med and v.get("fitted_word_seconds_median")} or None,
        fitted_over_asllrp_slot={k: round(v.get("fitted_word_seconds_median") / slot_med, 3)
                                 for k, v in (stage1_timing or {}).items()
                                 if slot_med and v.get("fitted_word_seconds_median")} or None)
    return out


def levenshtein(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def rule_check(records, table, vocab) -> dict:
    """Round 15 stage 1's English-order selection against the ASLLRP glosses mapped to entries (tiers with index)."""
    tp = n_rule = n_gold = edits = 0
    taus, few, empty_gold = [], 0, 0
    for r in records:
        rule = select_words(r["english"], vocab)
        gold = [e for s in r["signs"] if table[s["gloss"]]["tier"] in WITH_INDEX for e in table[s["gloss"]]["entries"]]
        if not gold:
            empty_gold += 1
            continue
        common = collections.Counter(rule) & collections.Counter(gold)
        tp += sum(common.values()); n_rule += len(rule); n_gold += len(gold)
        edits += levenshtein(rule, gold)
        left, positions = list(rule), []           # gold order of the shared entries, against their rule positions
        for e in gold:
            if e in left:
                i = left.index(e)
                positions.append(i); left[i] = None
        k = kendall_tau(positions)
        if k is None:
            few += 1
        else:
            taus.append(k[0])
    p, rcl = _share(tp, n_rule), _share(tp, n_gold)
    return dict(utterances=len(records) - empty_gold, utterances_without_mapped_gloss=empty_gold,
                rule_entries=n_rule, gold_entries=n_gold, bag_precision=p, bag_recall=rcl,
                bag_f1=round(2 * p * rcl / (p + rcl), 4) if p and rcl else None,
                entry_wer=_share(edits, n_gold),
                shared_entry_order_kendall_tau=summary(taus), shared_entry_order_tau_below_1=_share(
                    sum(t < 1 - 1e-9 for t in taus), len(taus)), utterances_below_two_shared=few,
                note="gold: the ASLLRP gloss sequence mapped to vocabulary entries (strict + index tiers), unmapped "
                     "glosses left out; rule: compositional.select_words on the English (round 15 stage 1).")


def vocabulary_overlap(reference_texts, target_texts, aligned_types=None) -> dict:
    ref = collections.Counter(t for x in reference_texts for t in english_tokens(x))
    tgt = collections.Counter(t for x in target_texts for t in english_tokens(x))

    def part(keep):
        r = {t for t in ref if keep(t)}
        types = [t for t in tgt if keep(t)]
        tokens = sum(n for t, n in tgt.items() if keep(t))
        out = dict(target_types=len(types), target_tokens=tokens,
                   type_overlap=_share(sum(t in r for t in types), len(types)),
                   token_overlap=_share(sum(n for t, n in tgt.items() if keep(t) and t in r), tokens))
        if aligned_types is not None:
            out["token_overlap_aligned_types"] = _share(
                sum(n for t, n in tgt.items() if keep(t) and t in aligned_types), tokens)
        return out
    missing = sorted(((t, n) for t, n in tgt.items() if t not in ref and t not in STOP_WORDS), key=lambda kv: (-kv[1], kv[0]))
    return dict(reference_types=len(ref), reference_tokens=sum(ref.values()),
                target_sentences=len(target_texts), all_words=part(lambda t: True),
                content_words=part(lambda t: t not in STOP_WORDS),
                target_words_per_sentence=round(sum(tgt.values()) / len(target_texts), 3) if target_texts else None,
                top_missing_content=[dict(word=t, count=n) for t, n in missing[:40]])


def aligned_english_types(records) -> set[str]:
    out = set()
    for r in records:
        tokens = english_tokens(r["english"])
        for x in align_spans(tokens, [english_forms(s) for s in r["signs"]]):
            if x is not None:
                out.update(tokens[x[0]:x[0] + x[1]])
    return out


def pilot_sentences(manifest, clips: int = 300) -> list[dict]:
    """``pilot_content_check.validation_sentences`` (round 15's 300 How2Sign val sentences), the same rule and seed."""
    val = [r for r in manifest["records"] if r.get("corpus") == H2S_CORPUS and r.get("partition") == "val"
           and r.get("signer_component") in {"c01", "c03", "c05"} and r.get("text")]
    return random.Random(0).sample(val, min(clips, len(val)))


def duplicate_statistics(subsets: dict, h2s_val) -> dict:
    """Repeated English: ASLLRP elicits the same sentences from several signers, so unique sentences (not utterances)
    bound what a planner can learn, and an exact val sentence in train inflates val scores."""
    key = lambda t: " ".join(normalise(t))  # noqa: E731
    train = collections.Counter(key(r["english"]) for r in subsets["train"])
    val = [key(r["english"]) for r in subsets["val"]]
    both = collections.Counter(key(r["english"]) for r in subsets["train+val"])
    return dict(train_utterances=sum(train.values()), train_unique_english=len(train),
                train_val_unique_english=len(both), val_unique_english=len(set(val)),
                val_utterances_with_english_in_train=sum(v in train for v in val),
                val_share_with_english_in_train=_share(sum(v in train for v in val), len(val)),
                how2sign_val_share_with_english_in_asllrp_train=_share(sum(key(t) in train for t in h2s_val),
                                                                       len(h2s_val)),
                most_repeated_train=[dict(english=t, utterances=n) for t, n in train.most_common(10)])


def marker_statistics(records) -> dict:
    tiers, utts = collections.Counter(), collections.Counter()
    for r in records:
        seen = {m["tier"] for m in r["markers"]}
        tiers.update(m["tier"] for m in r["markers"]); utts.update(seen)
    return {t: dict(spans=tiers[t], utterances=utts[t], utterance_share=_share(utts[t], len(records)))
            for t in sorted(tiers)}


# ---- main --------------------------------------------------------------------------------------------------------

def _store(spec):
    from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store
    return R2Store(os.environ) if spec == "r2" else DirectoryStore(Path(spec[4:]))


def _write_once(path: str, payload: dict) -> None:
    p = Path(path)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="r2", help="r2, or dir:<path> (a local mirror; read-only either way)")
    ap.add_argument("--utterances-key", default=ACQ + "derived/signbank-v1/utterances.json")
    ap.add_argument("--pairs-key", default=ACQ + "derived/asllrp-dai2-v1/english-gloss-v1/pairs.json")
    ap.add_argument("--citizen-map-key", default=ACQ + "derived/signbank-v1/asl-citizen-gloss-map.json")
    ap.add_argument("--mix-key", default=ACQ + "youtube-asl-v1/round14-mixed-v1/manifest.json")
    ap.add_argument("--mix-sha256", required=True)
    ap.add_argument("--corpora", default=",".join(ISOLATED_CORPORA), help="the isolated corpora of the vocabulary")
    ap.add_argument("--stage1-report-key", help="round 15 stage 1's stored report (its fitted durations)")
    ap.add_argument("--output-file", required=True, help="the JSON report (write-once)")
    ap.add_argument("--corpus-file", help="the local JSON corpus (write-once; ASLLRP terms: never upload or share)")
    args = ap.parse_args(argv)
    store = _store(args.store)
    inputs, payloads = {}, {}
    for name, key in (("utterances", args.utterances_key), ("pairs", args.pairs_key),
                      ("citizen_map", args.citizen_map_key), ("mix", args.mix_key)) + (
            (("stage1_report", args.stage1_report_key),) if args.stage1_report_key else ()):
        payloads[name] = store.get(key)
        inputs[key] = hashlib.sha256(payloads[name]).hexdigest()
    if inputs[args.mix_key] != args.mix_sha256:
        raise SystemExit(f"{args.mix_key} does not match --mix-sha256")
    utterances = json.loads(payloads["utterances"])["items"]
    pairs = json.loads(payloads["pairs"])["pairs"]
    manifest = json.loads(payloads.pop("mix"))
    citizen_map = json.loads(payloads["citizen_map"])
    stage1 = json.loads(payloads["stage1_report"]) if "stage1_report" in payloads else None

    records, join = build_corpus(utterances, pairs)
    corpora = tuple(c for c in args.corpora.split(",") if c)
    vocab = training_vocabulary(manifest, corpora)
    signbank, signbank_seconds, codes, semlex = vocabulary_indices(manifest, vocab, citizen_map)
    glosses = sorted({s["gloss"] for r in records for s in r["signs"]})
    first = {}
    for r in records:
        for s in r["signs"]:
            first.setdefault(s["gloss"], s)
    table = {g: map_sign(first[g], vocab, signbank, codes, semlex) for g in glosses}
    # an index gloss maps per occurrence (person), so the table holds its first occurrence's; the statistics use it
    subsets = {"train": [r for r in records if r["partition"] == "train"],
               "val": [r for r in records if r["partition"] == "val"]}
    subsets["train+val"] = subsets["train"] + subsets["val"]
    h2s_val = [r["text"] for r in manifest["records"] if r.get("corpus") == H2S_CORPUS and r.get("partition") == "val"
               and r.get("text")]
    pilot = [r["text"] for r in pilot_sentences(manifest)]
    aligned_train = aligned_english_types(subsets["train"])
    report = dict(
        purpose="round 15 stage 2 data groundwork: measurement only (no training, no generation, no stage-2 result)",
        inputs=dict(sorted(inputs.items())), corpora=list(corpora),
        license="ASLLRP Sign Bank / DAI Data Terms of Use: research use only, no redistribution",
        test_partition="counted, never read: " + str(join.get("pairs_test", 0)) + " pairs (Ben)",
        join=join, fps=FPS,
        utterances={k: dict(utterances=len(v), by_collection=dict(collections.Counter(r["collection"] for r in v)),
                            by_signer=dict(collections.Counter(r["signer"] for r in v))) for k, v in subsets.items()},
        sign_types=dict(collections.Counter(s["sign_type"] for r in subsets["train+val"] for s in r["signs"]).most_common()),
        categories=dict(collections.Counter(s["category"] for r in subsets["train+val"] for s in r["signs"]).most_common()),
        markers=marker_statistics(subsets["train+val"]),
        alignment={k: alignment_statistics(v) for k, v in subsets.items()},
        alignment_plain={k: {f: v[f] for f in ("aligned_share_of_glosses", "aligned_share_of_english_tokens", "order")}
                         for k, v in ((k, alignment_statistics(x, synonyms=False)) for k, x in subsets.items())},
        alignment_synonyms=SYNONYMS,
        duplicates=duplicate_statistics(subsets, h2s_val),
        vocabulary=dict(entries=len(vocab), single_letter_entries=sum(len(e) == 1 for e in vocab),
                        signbank_glosses_with_entries=len(signbank), asl_lex_codes_with_entries=len(semlex),
                        main_entries_with_asl_lex_code=len(codes)),
        mapping={k: mapping_statistics(v, table) for k, v in subsets.items()},
        name_tier_agreement=name_tier_agreement(glosses, vocab, signbank),
        timing=timing_statistics(subsets["train+val"], table, vocab, signbank_seconds,
                                 (stage1 or {}).get("timing")),
        rule_check={k: rule_check(v, table, vocab) for k, v in subsets.items()},
        generalisation=dict(
            asllrp_train_vs_how2sign_val=vocabulary_overlap([r["english"] for r in subsets["train"]], h2s_val,
                                                            aligned_train),
            asllrp_train_val_vs_how2sign_val=vocabulary_overlap([r["english"] for r in subsets["train+val"]], h2s_val),
            asllrp_train_vs_round15_pilot=vocabulary_overlap([r["english"] for r in subsets["train"]], pilot,
                                                             aligned_train),
            asllrp_val_vs_asllrp_train=vocabulary_overlap([r["english"] for r in subsets["train"]],
                                                          [r["english"] for r in subsets["val"]], aligned_train),
            note="reference: ASLLRP English; target: How2Sign val English (all val records of the mix, and round 15's "
                 "300 pilot sentences); ASLLRP val against train is the in-domain reference point. "
                 "token_overlap_aligned_types: target tokens whose type ASLLRP train aligns to a gloss at least once."),
        caveats=[
            "Alignment is lexical string matching: synonyms, irregular forms (went/GO) and paraphrase are not "
            "matched, so dropped-word and unaligned-gloss rates are upper bounds, and tau is measured on the "
            "matched subset only (order of matched content words).",
            "A repeated English word is aligned to the occurrence nearest the gloss's proportional position; this "
            "tie-break favours monotone alignments, so the order-difference rates are if anything low.",
            "Index signs (IX, POSS, SELF) align to pronoun forms by person; the main entry IX covers 2p/3p/loc, the "
            "occurrence label gives the person. Index glosses map per gloss type in the table (first occurrence).",
            "Durations: the annotated sign spans exclude transitions; the isolated trimmed clips keep rest and "
            "margins (about 2.8x their annotated sign), so ratios against the trimmed clips overstate compression. "
            "The like-for-like ratio is against ASLLVD's annotated sign spans (same SignStream convention); the slot "
            "ratio (sign plus following gap) is the comparison for fitted durations, which fill the whole sentence.",
            "Alignment uses english_tokens (contractions split) and a 13-entry synonym table for frequent function "
            "signs (alignment_synonyms); alignment_plain gives the same measures without the table.",
            "The test partition (Ben) is not read; train is 1,372 ASLLRP plus 280 RIT sentences (RIT's English is "
            "a short read-aloud set).",
        ])
    if args.corpus_file:
        _write_once(args.corpus_file, dict(
            schema_version=1, inputs=report["inputs"], license=report["license"], fps=FPS,
            note="local research copy; never upload or redistribute (ASLLRP terms); test partition excluded",
            records=records,
            gloss_map={g: dict(tier=m["tier"], entries=list(m["entries"]), category=first[g]["category"])
                       for g, m in table.items()}))
    _write_once(args.output_file, report)
    tv = report["alignment"]["train+val"]
    print(json.dumps(dict(
        utterances={k: v["utterances"] for k, v in report["utterances"].items()},
        glosses_per_utterance=tv["glosses_per_utterance"]["mean"],
        english_words_per_utterance=tv["english_words_per_utterance"]["mean"],
        tau_median=(tv["order"]["kendall_tau"] or {}).get("median"), order_differs=tv["order"]["share_tau_below_1"],
        coverage=report["mapping"]["train+val"]["coverage"],
        sign_seconds_median=report["timing"]["sign_seconds"]["median"],
        compression_vs_trimmed_clip=(report["timing"]["compression"]["vs_vocabulary_entry_typical"] or {}).get("median"),
        compression_vs_annotated_sign=(report["timing"]["compression"]["vs_same_signbank_gloss_clips_by_corpus"]
                                       .get(ANNOTATED) or {}).get("median"),
        rule_check_val=dict((k, report["rule_check"]["val"][k]) for k in ("bag_precision", "bag_recall", "entry_wer")),
        h2s_val_token_overlap=report["generalisation"]["asllrp_train_vs_how2sign_val"]["all_words"]["token_overlap"]),
        indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
