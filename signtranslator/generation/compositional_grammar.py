"""Round 15 stage 2: sentence grammar and timing for the compositional generator, fitted on ASLLRP train only
(docs/NEXT_PHASE_PLAN.md, "Round 15 stage 2 groundwork", 2026-10-28 21:16 UTC). Pure Python: no model, no states.

Everything fitted here reads ASLLRP *train* utterances (``check_fit_records``: any record that is not an ASLLRP train
utterance stops the fit). How2Sign sentences are never an input to a fit; they are only planned with a fitted model.

**Selection, stage 2** (``selection_rows``), on top of round 15's stage-1 rule (``compositional.selection_details``):
- tokens are ``tokens``: ``compositional.normalise`` with contractions split (``don't`` -> do not, ``it's`` -> it 's,
  ``mother's`` -> mother 's; the groundwork's ``english_tokens``), so negations and possessed nouns are visible;
- entries are matched greedily left to right, longest first, single letters never (as stage 1), against the vocabulary
  and, for spans that are not a vocabulary entry, the **lexicon** (``lexicon``: English word -> vocabulary entry, built
  from the Sign Bank and ASL-LEX tables, ``scripts/round15_stage2_fit.py``); a token that starts no entry tries the
  inflection fallback (stage 1's ``inflection_candidates``) against the vocabulary, then against the lexicon;
- stop words (stage 1's list) are dropped as in stage 1. Row kinds: exact, fallback, lexicon, stop, unmatched.

**Structure** (``apply_structure``), operations on the selected rows, fitted by ``fit_structure``:
- **keep / map / insert / drop per English token** (``model["tokens"]``): for a token seen at least ``min_count`` times in
  ASLLRP train, each candidate entry is scored by its hit rate: how often the utterance's gold contains it beyond what
  the other tokens' entries explain (``hits / n``). Candidates: the row's own entry; for a pronoun, the vocabulary's
  pointing sign (``POINTING``); and up to ``learned`` entries learned by co-occurrence with the token among the
  unexplained gold entries (``learned_links``: competitive linking, in at least ``min_count`` distinct sentences;
  ``children`` -> child, ``store`` -> sell; names and pronouns learn nothing). The best candidate is taken when its
  rate is at least ``threshold``, never one that loses a negation of the token or of its entry ("not happy" ->
  happy). Otherwise a closed-class token (``DROPPABLE``, or a
  pronoun) is dropped, and an open-class one keeps stage 1's behaviour: ASLLRP signs most such words another way (a
  classifier, fingerspelling, a synonym), which the vocabulary cannot reproduce, so dropping them would lose content.
  Rarer tokens keep stage 1's behaviour, except pronouns, which use their pointing class's pooled rate
  (``model["pointing"]``);
- **insert IX**: a pronoun (unmatched in stage 1, or matched to a one-clip pronoun entry) becomes the vocabulary's
  pointing sign of its kind (``ix``, ``poss``, ``self``: the Sign Bank main entries IX, POSS, SELF, which cover the
  non-first persons; IX-1p is a separate Sign Bank entry, so I / me map to ``me`` only when the vocabulary has it);
- **reorder** (``model["moves"]``): an entry of a declared class (``MOVE_CLASSES``: WH-words, negation, time and aspect)
  moves to the front or the end of the sentence when that raises the mean Kendall tau against the gold order on
  ASLLRP train by at least ``move_margin`` over at least ``move_min_count`` utterances; moves are stable (the moved
  entries keep their relative order). Topic-first is not detectable from English text and is not attempted.

**Timing** (``fit_timing``, ``target_seconds``): a sign's sentence duration is the median ASLLRP train sentence-sign
duration of its entry when the entry has at least ``min_signs`` timed train signs; otherwise ``ratio`` x its isolated
annotated length, ``ratio`` the median over train signs of (sentence span / the same entry's ASLLVD annotated sign
span). The annotated length is the median ASLLVD annotated span (``source.frames`` at 60 fps) of the entry's train
clips, else the entry's typical trimmed seconds / ``trimmed_over_annotated`` (the median, over ASLLVD entries, of
trimmed clip over annotated span). Every target is clipped to [floor, ceiling] (train lexical sign-duration
quantiles). The gap between signs is the median train gap; the leading share of padding the train ratio of leading
to leading + trailing time.

**Frame plan** (``plan_frames``): each core gets ``round(target x 24)`` frames (at least 1), each join ``round(gap x
24)``; the sentence is then fitted to the reference: ``pad`` (shorter sentences are padded with the first word's
approach and the last word's return, split by the leading share; longer ones are compressed uniformly to the
reference), ``scale`` (cores and gaps scaled uniformly to the reference in both directions) or ``none`` (ASLLRP pace,
whatever the length). ``allocate`` rounds a scaled plan to the exact total.

**Metrics** (``sequence_metrics``): bag precision / recall / F1 (micro, multiset intersection), entry WER and the
Kendall tau of the shared entries' order, as the groundwork's ``rule_check``; ``paired_bootstrap`` for differences.
"""
from __future__ import annotations

import collections
import re
import hashlib
import json
import statistics

import numpy as np

from .compositional import FPS, STOP_WORDS, inflection_candidates, normalise

ASLLRP_COLLECTIONS = ("asllrp-continuous", "rit-sentences")
LEXICON_TIERS = ("signbank_variant", "signbank_alternative", "asllex_dominant", "asllex_translation")
KEPT2 = ("exact", "fallback", "lexicon")
TOTAL_FITS = ("pad", "scale", "none")

# Pronoun -> the vocabulary's pointing sign of its kind (Sign Bank main entries IX, POSS, SELF cover the non-first
# persons; IX-1p / POSS-1p / SELF-1p are separate entries, so first-person forms point only to a first-person entry).
POINTING = {
    **{w: "ix" for w in ("you", "he", "she", "it", "him", "they", "them", "this", "that", "these", "those", "there",
                         "here")},
    **{w: "poss" for w in ("your", "yours", "his", "its", "their", "theirs", "hers")},
    **{w: "self" for w in ("yourself", "himself", "herself", "itself", "themselves", "yourselves")},
    "i": "me", "me": "me",
}
POINTING_ENTRIES = frozenset(POINTING.values())
# Closed-class English (the groundwork's word classes, with modals and a few determiners and adverbs): the only tokens
# a fit may drop. An open-class word ASLLRP does not sign with our entry is usually signed another way (a classifier,
# fingerspelling, a synonym), so it is kept or mapped, never dropped.
DROPPABLE = frozenset({
    "do", "does", "did", "have", "has", "had", "will", "would", "shall", "can", "could", "should", "must", "may",
    "might",                                                                                  # auxiliaries, modals
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "about", "into", "as", "over", "up", "out", "off", "than",
    "through", "around", "under", "after", "before",                                          # prepositions
    "and", "or", "but", "so", "that", "because", "if", "when",                                # conjunctions
    "some", "any", "all", "every", "each", "very", "just", "also", "too", "really", "then",   # determiners, adverbs
    "'s",
})
MOVE_CLASSES = {
    "wh": ("what", "where", "who", "why", "how", "when", "which"),
    "negation": ("not", "no", "none", "never", "nothing", "can't", "cannot", "don't", "won't"),
    "time": ("finish", "future", "will", "already", "later", "tomorrow", "yesterday", "today", "now", "before",
             "after", "past", "soon", "tonight", "recently", "ago", "then", "next"),
}
MOVES = ("stay", "front", "end")
_CONTRACTIONS = (("n't", "not"), ("'s", "'s"), ("'m", "am"), ("'re", "are"), ("'ll", "will"), ("'ve", "have"),
                 ("'d", "would"))
_IRREGULAR_NOT = {"won't": ("will", "not"), "can't": ("can", "not"), "shan't": ("shall", "not")}


# ---- tokens and stage-2 selection --------------------------------------------------------------------------------

def tokens(text) -> list[str]:
    """``normalise``d words with contractions split (the groundwork's ``english_tokens``, moved here unchanged)."""
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


def _matchable(entry: str, table) -> bool:
    return len(entry) > 1 and entry in table


def selection_rows(text, vocab, lexicon=None, stop_words=STOP_WORDS) -> list[dict]:
    """Every token span of ``tokens(text)`` with its fate, in order: ``{"tokens", "text", "entry", "kind", "tier"}``
    (module docstring). ``lexicon``: {English (normalised words joined by a space): {"entry", "tier"}}; its keys are
    used only where no vocabulary entry matches the same span."""
    lexicon = {k: v for k, v in (lexicon or {}).items() if _matchable(k, lexicon) and v["entry"] in vocab
               and len(v["entry"]) > 1}
    toks = tokens(text)
    longest = max((len(w.split(" ")) for w in list(vocab) + list(lexicon)), default=0)
    rows, p = [], 0
    while p < len(toks):
        for n in range(min(longest, len(toks) - p), 0, -1):
            span = " ".join(toks[p:p + n])
            if _matchable(span, vocab):
                kind, entry, tier = ("stop" if span in stop_words else "exact"), span, None
            elif span not in stop_words and _matchable(span, lexicon):
                kind, entry, tier = "lexicon", lexicon[span]["entry"], lexicon[span]["tier"]
                if entry in stop_words:
                    kind = "stop"
            else:
                continue
            rows.append(dict(tokens=toks[p:p + n], text=span, entry=entry, kind=kind, tier=tier))
            p += n
            break
        else:
            t = toks[p]
            row = dict(tokens=[t], text=t, entry=None, kind="unmatched", tier=None)
            if t in stop_words:
                row["kind"] = "stop"
            else:
                cands = inflection_candidates(t)
                hit = next((c for c in cands if _matchable(c, vocab)), None)
                if hit is not None:
                    row.update(entry=hit, kind="stop" if hit in stop_words else "fallback")
                else:
                    alias = next((c for c in cands if c not in stop_words and _matchable(c, lexicon)), None)
                    if alias is not None:
                        e = lexicon[alias]["entry"]
                        row.update(entry=e, kind="stop" if e in stop_words else "lexicon", tier=lexicon[alias]["tier"])
            rows.append(row)
            p += 1
    return rows


def selected(rows) -> list[str]:
    return [r["entry"] for r in rows if r["kind"] in KEPT2]


# ---- the fit guard -----------------------------------------------------------------------------------------------

def check_fit_records(records, partition: str = "train") -> None:
    """Stop unless every record is an ASLLRP utterance (``ASLLRP_COLLECTIONS``) of ``partition``: no How2Sign (or any
    other) sentence, no ASLLRP val or test utterance, ever reaches a fit."""
    for r in records:
        if r.get("collection") not in ASLLRP_COLLECTIONS:
            raise ValueError(f"fit record {r.get('id')!r} is not an ASLLRP utterance ({r.get('collection')!r})")
        if r.get("partition") != partition:
            raise ValueError(f"fit record {r.get('id')!r} is {r.get('partition')!r}, not {partition!r}")


# ---- structure ---------------------------------------------------------------------------------------------------

def droppable(token: str) -> bool:
    """A closed-class English token (``DROPPABLE``) or a pronoun (``POINTING``): the only tokens a fit may drop."""
    return token in DROPPABLE or token in POINTING


def _utterance_tokens(record, vocab, lexicon):
    """(gold Counter, {token: occurrences}, {token: default entry or None}, Counter of every row's default entry) of
    one utterance's non-stop token spans."""
    occ, default, base = collections.Counter(), {}, collections.Counter()
    for row in selection_rows(record["english"], vocab, lexicon):
        if row["kind"] == "stop":
            continue
        occ[row["text"]] += 1
        default[row["text"]] = row["entry"] if row["kind"] in KEPT2 else None
        if row["kind"] in KEPT2:
            base[row["entry"]] += 1
    return collections.Counter(record["gold"]), occ, default, base


def _unexplained(gold, base, default, n):
    """The gold entries the other tokens' defaults do not explain (the token's own default counted back in)."""
    others = base.copy()
    if default is not None:
        others[default] -= n
    return gold - others


_WORD = re.compile(r"[A-Za-z]+")


def proper_nouns(records) -> set[str]:
    """Lower-cased words capitalised in at least half of their (at least two) occurrences that do not start a
    sentence: names (``Diego``, ``Bob``), which learn no sign (ASLLRP signs them by fingerspelling or a name sign)."""
    cap, total = collections.Counter(), collections.Counter()
    for r in records:
        for sentence in re.split(r"[.!?\"]+", r["english"] or ""):
            for k, w in enumerate(_WORD.findall(sentence)):
                if k == 0 or w == "I":
                    continue
                total[w.lower()] += 1
                cap[w.lower()] += w[0].isupper()
    return {w for w, n in total.items() if n >= 2 and cap[w] / n >= .5}


def learned_links(records, vocab, lexicon, *, min_count: int) -> dict:
    """{token: Counter(entry)}: competitive linking (after Melamed 2000) of English tokens to unexplained gold entries.
    Pass 1 scores every (token, entry) pair by its Dice coefficient over utterances (entry unexplained for the
    token); pass 2 credits, in each utterance, every unexplained entry to the one token that explains it best (the
    highest score; ties by name), so ``teacher`` -> teach is not also ``listen`` -> teach; a token may take several
    entries (a compound). Links are counted once per distinct English sentence (ASLLRP elicits the same English from
    several signers); pairs linked in fewer than ``min_count`` distinct sentences are left out. Pointing entries and
    stop words are never learned, and pronouns learn nothing (they point, ``POINTING``)."""
    parsed = [(" ".join(normalise(r["english"])), _utterance_tokens(r, vocab, lexicon)) for r in records]
    names = proper_nouns(records)
    pair, n_tok, n_ent = collections.Counter(), collections.Counter(), collections.Counter()
    utterance_pairs = []
    for _, (gold, occ, default, base) in parsed:
        pairs = set()
        for t, n in occ.items():
            if t in POINTING or t in names:
                continue
            for e in _unexplained(gold, base, default[t], n):
                if _matchable(e, vocab) and e not in STOP_WORDS and e not in POINTING_ENTRIES and e != default[t]:
                    pairs.add((t, e))
        utterance_pairs.append(pairs)
        pair.update(pairs)
        n_tok.update(occ.keys())
        n_ent.update(set(gold))
    dice = {(t, e): 2 * c / (n_tok[t] + n_ent[e]) for (t, e), c in pair.items()}
    links = collections.defaultdict(set)
    for (sentence, _), pairs in zip(parsed, utterance_pairs):
        best = {}
        for t, e in sorted(pairs, key=lambda p: (-dice[p], p)):
            best.setdefault(e, t)
        for e, t in best.items():
            links[(t, e)].add(sentence)
    out = collections.defaultdict(collections.Counter)
    for (t, e), sentences in links.items():
        if len(sentences) >= min_count:
            out[t][e] = len(sentences)
    return {t: c for t, c in sorted(out.items())}


def _token_stats(records, vocab, lexicon, pointing, *, min_count: int, learned: int):
    """{token: {"n", "default", "hits": Counter(candidate), "learned": [entries]}} over ``records``.

    Candidates of a token: its default entry, its pointing sign (``pointing``) and up to ``learned`` entries learned
    by competitive linking (``learned_links``: ``children`` -> child, ``store`` -> sell). A candidate's hits in an
    utterance: min(the token's occurrences, the gold count of the candidate the other tokens' defaults leave
    unexplained)."""
    links = learned_links(records, vocab, lexicon, min_count=min_count)
    parsed = [_utterance_tokens(r, vocab, lexicon) for r in records]
    stats = {}
    for gold, occ, default, base in parsed:
        for t, n in occ.items():
            s = stats.get(t)
            if s is None:
                ranked = sorted((links.get(t) or {}).items(), key=lambda kv: (-kv[1], kv[0]))
                lrn = [e for e, _ in ranked][:learned]
                cands = ([default[t]] if default[t] is not None else []) + lrn
                p = pointing.get(t)
                if p and _matchable(p, vocab) and p not in cands:
                    cands.append(p)
                s = stats[t] = dict(n=0, default=default[t], learned=lrn,
                                    hits=collections.Counter({c: 0 for c in cands}))
            unexplained = _unexplained(gold, base, default[t], n)
            s["n"] += n
            for c in s["hits"]:
                s["hits"][c] += min(n, unexplained[c])
    return {t: s for t, s in stats.items() if s["hits"]}


NEGATIONS = frozenset({"not", "no", "never", "none", "nothing", "cannot", "can't", "don't", "won't", "neg"})


def drops_negation(default, candidate) -> bool:
    """True when mapping ``default`` to ``candidate`` would lose a negation ("not happy" -> happy): never allowed."""
    words = lambda e: set((e or "").split())        # noqa: E731
    return bool(words(default) & NEGATIONS) and not words(candidate) & NEGATIONS


def _decide(token, n, hits, default, threshold):
    """The token's operation: its best candidate when the hit rate reaches ``threshold`` (keep, map or insert; never
    a map that loses a negation); otherwise drop when the token is droppable, else stage 1's behaviour (keep its
    default, or none)."""
    hits = {c: h for c, h in hits.items() if c == default or not (drops_negation(default, c)
                                                                    or drops_negation(token, c))}
    best = max(hits, key=lambda c: (hits[c] / n, c == default, c)) if hits else None
    rate = hits[best] / n if hits else 0.
    if rate < threshold:
        if droppable(token):
            return dict(op="drop", entry=None, n=n, rate=round(rate, 4))
        return dict(op="keep" if default else "none", entry=default, n=n,
                    rate=round(hits.get(default, 0) / n, 4) if default else 0.)
    op = "keep" if best == default else ("insert" if default is None else "map")
    return dict(op=op, entry=best, n=n, rate=round(rate, 4))


def tau_of(predicted, gold):
    """Kendall tau of the shared entries: the gold order of the entries both sequences contain (duplicates matched
    left to right) against their predicted positions (the groundwork's ``rule_check``); None below two shared."""
    left, positions = list(predicted), []
    for e in gold:
        if e in left:
            i = left.index(e)
            positions.append(i)
            left[i] = None
    n = len(positions)
    if n < 2:
        return None
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            c += positions[j] > positions[i]
            d += positions[j] < positions[i]
    return (c - d) / (n * (n - 1) / 2)


def _strip(entries) -> list[str]:
    """The sequence without pointing entries (their duplicates blur the order of the content signs)."""
    return [e for e in entries if e not in POINTING_ENTRIES]


def apply_moves(entries, moves) -> list[str]:
    """Stable moves: entries with move "front" first, then the unmoved, then "end", each group in its order."""
    front = [e for e in entries if moves.get(e) == "front"]
    end = [e for e in entries if moves.get(e) == "end"]
    rest = [e for e in entries if moves.get(e) not in ("front", "end")]
    return front + rest + end


def apply_structure(rows, model, vocab, *, moves: bool = True) -> tuple[list[str], list[dict]]:
    """(entries, trace) of selected rows under a fitted structure model (module docstring). ``trace``: one row per
    non-stop token span with its operation (keep, map, insert, drop, none) and the entry it signs."""
    model = model or {}
    actions, pooled = model.get("tokens", {}), model.get("pointing", {})
    pointing = (model.get("parameters") or {}).get("pointing") or POINTING
    out, trace = [], []
    for row in rows:
        if row["kind"] == "stop":
            continue
        a = actions.get(row["text"])
        default = row["entry"] if row["kind"] in KEPT2 else None
        if a is None:
            p = pointing.get(row["text"])
            pa = pooled.get(p) if p else None
            if pa is not None and pa["op"] == "insert" and _matchable(p, vocab):
                a = dict(op="insert" if default is None else "map", entry=p)
            else:
                a = dict(op="keep" if default else "none", entry=default)
        entry = a.get("entry") if a["op"] in ("keep", "map", "insert") else None
        if entry is not None:
            out.append(entry)
        trace.append(dict(text=row["text"], kind=row["kind"], op=a["op"], entry=entry))
    if moves and model.get("moves"):
        moved = apply_moves(out, model["moves"])
        for t in trace:
            if t["entry"] is not None and model["moves"].get(t["entry"]) in ("front", "end"):
                t["moved"] = model["moves"][t["entry"]]
        out = moved
    return out, trace


def fit_structure(train, vocab, lexicon=None, *, threshold: float = .5, min_count: int = 5, learned: int = 3,
                  move_min_count: int = 10, move_margin: float = .02, pointing=None, move_classes=None) -> dict:
    """The structure model (module docstring) from ASLLRP train utterances (``check_fit_records``), each with
    ``english`` and ``gold`` (its gloss sequence as vocabulary entries)."""
    check_fit_records(train)
    pointing = POINTING if pointing is None else pointing
    move_classes = MOVE_CLASSES if move_classes is None else move_classes
    stats = _token_stats(train, vocab, lexicon, pointing, min_count=min_count, learned=learned)
    actions = {t: dict(_decide(t, s["n"], s["hits"], s["default"], threshold), learned=s["learned"])
               for t, s in sorted(stats.items()) if s["n"] >= min_count}
    pooled = {}
    for p in sorted({v for v in pointing.values() if _matchable(v, vocab)}):
        n = sum(s["n"] for t, s in stats.items() if pointing.get(t) == p)
        hits = sum(s["hits"][p] for t, s in stats.items() if pointing.get(t) == p)
        if n >= min_count:
            rate = hits / n
            pooled[p] = dict(op="insert" if rate >= threshold else "drop", n=n, rate=round(rate, 4))
    model = dict(tokens=actions, pointing=pooled, moves={})
    seqs = [(apply_structure(selection_rows(r["english"], vocab, lexicon), model, vocab, moves=False)[0], r["gold"])
            for r in train]
    movable = {e for words in move_classes.values() for e in words}
    counts = collections.Counter(e for s, _ in seqs for e in set(s) if e in movable)
    moves, report = {}, {}
    for e, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        scores = {}
        for m in MOVES:
            trial = dict(moves, **({e: m} if m != "stay" else {}))
            taus = [tau_of(_strip(apply_moves(s, trial)), _strip(g)) for s, g in seqs if e in s]
            taus = [t for t in taus if t is not None]
            scores[m] = (float(np.mean(taus)) if taus else None, len(taus))
        n = scores["stay"][1]
        best = max(MOVES, key=lambda m: (scores[m][0] if scores[m][0] is not None else -9, m == "stay"))
        gain = (scores[best][0] - scores["stay"][0]) if n else 0.
        report[e] = dict(n=n, **{f"tau_{m}": round(v[0], 4) if v[0] is not None else None for m, v in scores.items()})
        if best != "stay" and n >= move_min_count and gain >= move_margin:
            moves[e] = best
    model["moves"] = moves
    model["move_scores"] = report
    model["parameters"] = dict(threshold=threshold, min_count=min_count, learned=learned,
                               move_min_count=move_min_count, move_margin=move_margin,
                               droppable=sorted(DROPPABLE), pointing=dict(sorted(pointing.items())),
                               move_classes={k: list(v) for k, v in move_classes.items()})
    return model


# ---- timing ------------------------------------------------------------------------------------------------------

def _q(values, q):
    return float(np.quantile(np.asarray(values, dtype=float), q))


def fit_timing(train, typical, annotated, *, min_signs: int = 3, floor_q: float = .25, ceiling_q: float = .9) -> dict:
    """The timing model (module docstring) from ASLLRP train utterances (``check_fit_records``). Each record's
    ``timed``: [(entry or None, category, start, end)] in time order, the entry its sign's single vocabulary entry
    (None when unmapped or a compound); ``typical``: {entry: trimmed isolated seconds}; ``annotated``: {entry: median
    ASLLVD annotated sign seconds}. Returns plain numbers and tables (JSON)."""
    check_fit_records(train)
    by_entry, lexical, gaps, ratios, lead, trail = collections.defaultdict(list), [], [], [], [], []
    for r in train:
        timed = sorted((x for x in r["timed"] if x[2] is not None and x[3] is not None), key=lambda x: x[2])
        for k, (entry, cat, a, b) in enumerate(timed):
            d = b - a
            if cat == "lexical":
                lexical.append(d)
            if entry is not None:
                by_entry[entry].append(d)
                if annotated.get(entry):
                    ratios.append(d / annotated[entry])
            if k + 1 < len(timed):
                gaps.append(timed[k + 1][2] - b)
        if timed and r.get("seconds"):
            lead.append(timed[0][2])
            trail.append(r["seconds"] - timed[-1][3])
    if not ratios or not lexical:
        raise ValueError("no timed train signs with an annotated isolated length")
    both = [e for e in annotated if e in typical and annotated[e] > 0]
    trimmed = float(np.median([typical[e] / annotated[e] for e in both])) if both else None
    lead_m, trail_m = float(np.median(lead)), float(np.median(trail))
    return dict(
        ratio=round(float(np.median(ratios)), 4), ratio_signs=len(ratios),
        trimmed_over_annotated=round(trimmed, 4) if trimmed else None, trimmed_over_annotated_entries=len(both),
        floor=round(_q(lexical, floor_q), 4), ceiling=round(_q(lexical, ceiling_q), 4),
        gap=round(float(np.median(gaps)), 4), lead_share=round(lead_m / (lead_m + trail_m), 4),
        lead_seconds=round(lead_m, 4), trail_seconds=round(trail_m, 4), min_signs=min_signs,
        quantiles=dict(floor=floor_q, ceiling=ceiling_q),
        sentence_seconds={e: round(float(np.median(v)), 4) for e, v in sorted(by_entry.items()) if len(v) >= min_signs},
        sentence_signs={e: len(v) for e, v in sorted(by_entry.items()) if len(v) >= min_signs},
        annotated={e: round(float(v), 4) for e, v in sorted(annotated.items())})


def target_seconds(entry: str, timing: dict, typical) -> tuple[float, str]:
    """(seconds, source) of one entry's sign in a sentence: its ASLLRP sentence median ("asllrp"), else ratio x its
    ASLLVD annotated length ("annotated"), else ratio x typical / trimmed_over_annotated ("trimmed"); clipped."""
    if entry in timing["sentence_seconds"]:
        s, src = timing["sentence_seconds"][entry], "asllrp"
    elif entry in timing["annotated"]:
        s, src = timing["ratio"] * timing["annotated"][entry], "annotated"
    else:
        s, src = timing["ratio"] * float(typical[entry]) / timing["trimmed_over_annotated"], "trimmed"
    clipped = min(max(s, timing["floor"]), timing["ceiling"])
    return float(clipped), src + ("" if clipped == s else "_clipped")


def allocate(lengths, total: int, minimum) -> list[int]:
    """Non-negative integers proportional to ``lengths`` summing to ``total``, each at least its ``minimum``
    (largest-remainder rounding; when the minima exceed the total, the minima)."""
    lengths, minimum = [float(x) for x in lengths], [int(m) for m in minimum]
    if sum(minimum) >= total or not lengths:
        return list(minimum)
    s = sum(lengths)
    raw = [total * x / s if s > 0 else total / len(lengths) for x in lengths]
    out = [max(int(np.floor(r)), m) for r, m in zip(raw, minimum)]
    while sum(out) < total:
        k = max(range(len(out)), key=lambda i: (raw[i] - out[i], -i))
        out[k] += 1
    while sum(out) > total:
        k = max((i for i in range(len(out)) if out[i] > minimum[i]), key=lambda i: (out[i] - raw[i], i))
        out[k] -= 1
    return out


def plan_frames(targets, gap_seconds: float, reference_seconds: float, total_fit: str = "pad",
                lead_share: float = .4, fps: float = FPS) -> dict:
    """The frame plan of one sentence (module docstring): ``cores`` (frames per sign), ``gaps`` (frames per join),
    ``lead`` and ``trail`` (padding frames), ``fit`` (what was done: padded, compressed, scaled, none) and ``scale``
    (body frames after / before)."""
    if total_fit not in TOTAL_FITS:
        raise ValueError(f"total_fit must be one of {TOTAL_FITS}")
    if not targets:
        raise ValueError("no signs to plan")
    cores = [max(1, int(round(t * fps))) for t in targets]
    gaps = [max(0, int(round(gap_seconds * fps)))] * (len(cores) - 1)
    body = sum(cores) + sum(gaps)
    R = max(1, int(round(reference_seconds * fps)))
    lead = trail = 0
    fit = "none"
    if total_fit == "pad" and body <= R:
        pad = R - body
        lead = int(round(pad * lead_share))
        trail, fit = pad - lead, "padded"
    elif total_fit in ("pad", "scale") and body != R:
        scaled = allocate(cores + gaps, R, [1] * len(cores) + [0] * len(gaps))
        cores, gaps = scaled[:len(cores)], scaled[len(cores):]
        fit = "compressed" if R < body else "scaled"
    return dict(cores=cores, gaps=gaps, lead=lead, trail=trail, fit=fit, body_planned=body,
                scale=round((sum(cores) + sum(gaps)) / body, 4), frames=lead + sum(cores) + sum(gaps) + trail)


# ---- metrics -----------------------------------------------------------------------------------------------------

def levenshtein(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def utterance_counts(predicted, gold) -> dict:
    """One utterance's counts: true positives (multiset intersection), lengths, edits, tau, and tau without the
    pointing entries (``tau_np``: duplicates of ``ix`` match left to right and blur the order of content signs)."""
    common = collections.Counter(predicted) & collections.Counter(gold)
    return dict(tp=sum(common.values()), pred=len(predicted), gold=len(gold), edits=levenshtein(predicted, gold),
                tau=tau_of(predicted, gold), tau_np=tau_of(_strip(predicted), _strip(gold)))


def _mean(values):
    return round(float(np.mean(values)), 4) if values else None


def aggregate(rows) -> dict:
    tp, pred, gold, edits = (sum(r[k] for r in rows) for k in ("tp", "pred", "gold", "edits"))
    taus = [r["tau"] for r in rows if r["tau"] is not None]
    taus_np = [r["tau_np"] for r in rows if r["tau_np"] is not None]
    p = tp / pred if pred else None
    rc = tp / gold if gold else None
    return dict(utterances=len(rows), predicted_entries=pred, gold_entries=gold,
                bag_precision=round(p, 4) if p is not None else None, bag_recall=round(rc, 4) if rc is not None else None,
                bag_f1=round(2 * p * rc / (p + rc), 4) if p and rc else None,
                entry_wer=round(edits / gold, 4) if gold else None,
                tau_mean=_mean(taus), tau_median=round(float(np.median(taus)), 4) if taus else None,
                tau_below_1_share=_mean([t < 1 - 1e-9 for t in taus]), tau_utterances=len(taus),
                tau_np_mean=_mean(taus_np), tau_np_below_1_share=_mean([t < 1 - 1e-9 for t in taus_np]),
                tau_np_utterances=len(taus_np), words_per_utterance=round(pred / len(rows), 3) if rows else None)


def sequence_metrics(predicted, gold) -> tuple[dict, list[dict]]:
    """(aggregate, per-utterance rows) over utterances with a non-empty gold sequence (as ``rule_check``)."""
    rows = [utterance_counts(p, g) for p, g in zip(predicted, gold) if g]
    return aggregate(rows), rows


PAIRED_KEYS = ("bag_precision", "bag_recall", "bag_f1", "entry_wer", "tau_paired", "tau_np_paired")


def _paired_stats(a_rows, b_rows) -> dict:
    a, b = aggregate(a_rows), aggregate(b_rows)
    out = {k: (None if a[k] is None or b[k] is None else b[k] - a[k]) for k in PAIRED_KEYS[:4]}
    for key, field in (("tau_paired", "tau"), ("tau_np_paired", "tau_np")):
        d = [y[field] - x[field] for x, y in zip(a_rows, b_rows) if x[field] is not None and y[field] is not None]
        out[key] = float(np.mean(d)) if d else None
    return out


def paired_bootstrap(rows_a, rows_b, *, boot: int = 1000, seed: int = 0) -> dict:
    """b minus a for the bag measures and WER (aggregates) and for tau (``tau_paired``, ``tau_np_paired``: the mean
    per-utterance difference over utterances where both have a tau, so both are read on the same utterances), with a
    95% bootstrap interval over utterances (resampled jointly)."""
    if len(rows_a) != len(rows_b):
        raise ValueError("paired rows differ in length")
    base = _paired_stats(rows_a, rows_b)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in PAIRED_KEYS}
    n = len(rows_a)
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        d = _paired_stats([rows_a[i] for i in idx], [rows_b[i] for i in idx])
        for k in PAIRED_KEYS:
            if d[k] is not None:
                draws[k].append(d[k])
    n_tau = sum(x["tau"] is not None and y["tau"] is not None for x, y in zip(rows_a, rows_b))
    n_np = sum(x["tau_np"] is not None and y["tau_np"] is not None for x, y in zip(rows_a, rows_b))
    out = {k: dict(difference=round(base[k], 4) if base[k] is not None else None,
                   ci95=[round(float(np.percentile(draws[k], 2.5)), 4), round(float(np.percentile(draws[k], 97.5)), 4)]
                   if draws[k] else None) for k in PAIRED_KEYS}
    out["tau_paired"]["utterances"], out["tau_np_paired"]["utterances"] = n_tau, n_np
    return out


def model_sha256(model: dict) -> str:
    return hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def median(values):
    return float(statistics.median(values)) if values else None


__all__ = ["ASLLRP_COLLECTIONS", "LEXICON_TIERS", "KEPT2", "TOTAL_FITS", "POINTING", "POINTING_ENTRIES", "DROPPABLE",
           "NEGATIONS", "MOVE_CLASSES", "MOVES", "droppable", "drops_negation", "proper_nouns", "learned_links", "tokens",
           "selection_rows", "selected", "check_fit_records", "tau_of", "apply_moves", "apply_structure",
           "fit_structure", "fit_timing", "target_seconds", "allocate", "plan_frames", "levenshtein",
           "utterance_counts", "aggregate", "sequence_metrics", "paired_bootstrap", "model_sha256", "median"]
