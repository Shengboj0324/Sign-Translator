"""Blinded evaluation of English -> ASL output by Deaf ASL users: design, sheets, answers and scores (plan item 3b).

Protocol: docs/HUMAN_EVALUATION_PROTOCOL.md. The pieces here are pure functions so the design and the scoring are
tested without video or people.

**Design (Latin square).** ``U`` utterances are each produced by ``S`` systems (the recorded reference rendered the same
way, one or more generators, baselines). Reviewers are split into ``S`` groups; group ``g`` sees utterance ``u`` from
system ``(u + g) mod S``. Every reviewer therefore sees each utterance **once** (no priming by a second rendering of the
same sentence) and every system on ``~U/S`` utterances; across groups every (utterance, system) cell is rated, so the
comparison between systems is paired by utterance. Two or more reviewers per group give inter-rater agreement.

**Blinding.** Each item has a random id and a video named after it. The sheet shows only that id, the video and four
English options; which system made the video, which option is correct and the plan's grammatical markers live only in
the key, which reviewers never receive.

**Questions per item.**

- ``translation``: free English translation (kept for proposition scoring by the study team; reported only as
  indicative BLEU);
- ``choice``: which of four English sentences was signed (``a``-``d``); the distractors are other sentences of the same
  partition (not only the study's, so elimination does not help) with the closest word count, so chance is 25% and
  length is not a cue;
- ``sentence_type``: ``statement``, ``yes_no_question``, ``wh_question`` or ``rhetorical_question``;
- ``negated``, ``topic``: ``yes`` / ``no`` - is the sentence negated, is a topic set up front;
- ``naturalness``: 1 (not signing) .. 5 (fluent, natural);
- ``viewable``: ``no`` when the video cannot be judged (it is then excluded from every score but counted).

The grammatical truth comes from the plan's markers (ASLLRP annotation): ``y/n question`` -> yes/no question, else
``wh question`` -> wh-question, else ``rhq question`` -> rhetorical, else statement; ``negative`` -> negated;
``topic/focus`` -> topic.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import math
import random
import re
from typing import Iterable, Mapping, Sequence

from signtranslator.planning.asllrp_gloss import MARKERS

OPTIONS = ("a", "b", "c", "d")
SENTENCE_TYPES = ("statement", "yes_no_question", "wh_question", "rhetorical_question")
YES_NO = ("yes", "no")
NATURALNESS = (1, 2, 3, 4, 5)
SHEET_COLUMNS = ("item_id", "video", "option_a", "option_b", "option_c", "option_d",
                 "viewable", "translation", "choice", "sentence_type", "negated", "topic", "naturalness",
                 "reviewer_sha256", "notes")
_ANSWER_COLUMNS = ("viewable", "translation", "choice", "sentence_type", "negated", "topic", "naturalness",
                   "reviewer_sha256", "notes")
_SHA = re.compile(r"[0-9a-f]{64}")


def grammatical_truth(markers: Sequence[Sequence[int]]) -> dict:
    """Sentence type, negation and topic of one plan, from its per-sign marker vectors."""
    present = {MARKERS[k] for row in markers for k, v in enumerate(row) if v}
    if "y/n question" in present:
        kind = "yes_no_question"
    elif "wh question" in present:
        kind = "wh_question"
    elif "rhq question" in present:
        kind = "rhetorical_question"
    else:
        kind = "statement"
    return dict(sentence_type=kind, negated="yes" if "negative" in present else "no",
                topic="yes" if "topic/focus" in present else "no")


def latin_square(utterances: int, systems: int) -> list[list[int]]:
    """``assignment[g][u]`` = the system group ``g`` sees utterance ``u`` from."""
    if utterances < systems:
        raise ValueError("need at least as many utterances as systems for every group to see every system")
    return [[(u + g) % systems for u in range(utterances)] for g in range(systems)]


def distractors(english: Sequence[str], index: int, rng: random.Random, *, count: int = 3) -> list[int]:
    """Indices of ``count`` other sentences with the closest word counts (ties broken at random), never the same text."""
    target = english[index]
    length = len(target.split())
    pool = [j for j, e in enumerate(english) if j != index and e.strip().lower() != target.strip().lower()]
    if len(pool) < count:
        raise ValueError("not enough distinct sentences for distractors")
    rng.shuffle(pool)
    pool.sort(key=lambda j: abs(len(english[j].split()) - length))
    chosen, seen = [], {target.strip().lower()}
    for j in pool:
        key = english[j].strip().lower()
        if key not in seen:
            chosen.append(j); seen.add(key)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise ValueError("not enough distinct sentences for distractors")
    return chosen


def design_study(utterances: Sequence[Mapping], systems: Sequence[str], *, seed: int,
                 distractor_pool: Sequence[str] | None = None) -> dict:
    """The study key: every item (id, group, utterance, system, options, correct option, grammatical truth).

    ``utterances``: dicts with ``sign`` (the record id), ``english`` and ``markers`` (the plan's marker vectors).
    Every utterance gets one set of options shared by all its systems, so systems differ only in the video.
    Distractors come from ``distractor_pool`` (pass every sentence of the partition): drawn only from the study's own
    sentences, each sentence would answer exactly one item and a reviewer could answer by elimination.
    """
    if len(set(systems)) != len(systems) or not systems:
        raise ValueError("system labels must be distinct")
    rng = random.Random(seed)
    english = [u["english"] for u in utterances]
    pool = list(dict.fromkeys(list(distractor_pool or []) + english))     # the study's sentences are always in it
    options = []
    for i in range(len(utterances)):
        target = pool.index(english[i])
        offered = [english[i]] + [pool[j] for j in distractors(pool, target, rng)]
        rng.shuffle(offered)
        options.append(dict(texts=offered, correct=OPTIONS[offered.index(english[i])]))
    square = latin_square(len(utterances), len(systems))
    items, used = [], set()
    for g, row in enumerate(square):
        order = list(range(len(utterances)))
        rng.shuffle(order)                                          # presentation order per group
        for position, u in enumerate(order):
            while True:
                item_id = "%012x" % rng.getrandbits(48)
                if item_id not in used:
                    used.add(item_id); break
            items.append(dict(item_id=item_id, group=g, position=position, sign=utterances[u]["sign"],
                              utterance=u, system=systems[row[u]], english=english[u],
                              options=options[u]["texts"], correct=options[u]["correct"],
                              truth=grammatical_truth(utterances[u]["markers"])))
    return dict(seed=seed, systems=list(systems), groups=len(systems), utterances=len(utterances), items=items)


def write_sheet(key: Mapping, group: int) -> str:
    """The CSV one reviewer of ``group`` fills: presentation order, answer columns empty."""
    rows = sorted((i for i in key["items"] if i["group"] == group), key=lambda i: i["position"])
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=SHEET_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for i in rows:
        writer.writerow(dict(item_id=i["item_id"], video=f"{i['item_id']}.mp4",
                             **{f"option_{o}": t for o, t in zip(OPTIONS, i["options"])},
                             **{c: "" for c in _ANSWER_COLUMNS}))
    return out.getvalue()


def parse_sheet(text: str, key: Mapping) -> tuple[list[dict], list[dict]]:
    """(answers, problems) of one returned sheet. Offered columns must be unchanged; a row is answered when its
    ``choice`` or ``viewable`` cell is filled, and an answered row must be complete and in range. One reviewer per
    sheet."""
    items = {i["item_id"]: i for i in key["items"]}
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if tuple(reader.fieldnames or ()) != SHEET_COLUMNS:
        return [], [dict(row=0, issue=f"columns must be exactly {', '.join(SHEET_COLUMNS)}")]
    answers, problems = [], []
    for n, row in enumerate(reader, start=2):
        row = {k: (v or "").strip() for k, v in row.items()}
        item = items.get(row["item_id"])
        if item is None:
            problems.append(dict(row=n, issue="unknown item_id")); continue
        if row["video"] != f"{item['item_id']}.mp4" or [row[f"option_{o}"] for o in OPTIONS] != list(item["options"]):
            problems.append(dict(row=n, issue="offered columns were changed")); continue
        viewable = row["viewable"].lower() or "yes"
        if not row["choice"] and not row["viewable"]:
            continue                                                # unanswered
        issues = []
        if viewable not in YES_NO:
            issues.append("viewable must be yes or no")
        if not _SHA.fullmatch(row["reviewer_sha256"]):
            issues.append("reviewer_sha256 must be a 64-hex SHA-256 key, never a name")
        if viewable == "yes":
            if row["choice"].lower() not in OPTIONS:
                issues.append("choice must be a, b, c or d")
            if row["sentence_type"] not in SENTENCE_TYPES:
                issues.append(f"sentence_type must be one of {', '.join(SENTENCE_TYPES)}")
            for c in ("negated", "topic"):
                if row[c].lower() not in YES_NO:
                    issues.append(f"{c} must be yes or no")
            if row["naturalness"] not in {str(v) for v in NATURALNESS}:
                issues.append("naturalness must be 1-5")
        if issues:
            problems.extend(dict(row=n, item_id=item["item_id"], issue=i) for i in issues); continue
        answers.append(dict(item_id=item["item_id"], reviewer_sha256=row["reviewer_sha256"], viewable=viewable,
                            translation=row["translation"], choice=row["choice"].lower(),
                            sentence_type=row["sentence_type"], negated=row["negated"].lower(),
                            topic=row["topic"].lower(),
                            naturalness=int(row["naturalness"]) if viewable == "yes" else None, notes=row["notes"]))
    reviewers = {a["reviewer_sha256"] for a in answers}
    if len(reviewers) > 1:
        return [], [dict(row=0, issue="one sheet must carry one reviewer key")]
    return answers, problems


def merge_answers(sheets: Iterable[list[dict]], key: Mapping) -> tuple[list[dict], list[dict]]:
    """Answers of all sheets; a reviewer who answered sheets of two groups saw sentences twice and is excluded."""
    items = {i["item_id"]: i for i in key["items"]}
    answers = [a for sheet in sheets for a in sheet]
    groups = collections.defaultdict(set)
    for a in answers:
        groups[a["reviewer_sha256"]].add(items[a["item_id"]]["group"])
    crossed = {r for r, g in groups.items() if len(g) > 1}
    seen = collections.Counter((a["reviewer_sha256"], a["item_id"]) for a in answers)
    problems = [dict(reviewer_sha256=r, issue="answered sheets of more than one group (primed); excluded")
                for r in sorted(crossed)]
    problems += [dict(reviewer_sha256=r, item_id=i, issue="answered the item twice; excluded")
                 for (r, i), c in sorted(seen.items()) if c > 1]
    kept = [a for a in answers if a["reviewer_sha256"] not in crossed and seen[(a["reviewer_sha256"], a["item_id"])] == 1]
    return kept, problems


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval of a proportion."""
    if n == 0:
        return 0., 1.
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0., centre - half), min(1., centre + half)


def _proportion(hits: Sequence[bool]) -> dict:
    n, k = len(hits), int(sum(hits))
    lo, hi = wilson(k, n)
    return dict(n=n, rate=round(k / n, 4) if n else None, ci95=[round(lo, 4), round(hi, 4)])


def score_study(answers: Sequence[Mapping], key: Mapping, *, reference: str | None = None, seed: int = 0) -> dict:
    """Per-system scores, paired comparisons with ``reference`` by utterance, and inter-rater agreement."""
    from signtranslator.eval_framework.statistics import bootstrap_ci, paired_permutation_pvalue
    from signtranslator.eval_framework.text_metrics import corpus_bleu
    from signtranslator.grammar.signbleu import fleiss_kappa
    items = {i["item_id"]: i for i in key["items"]}
    by_system = collections.defaultdict(list)
    excluded = collections.Counter()
    for a in answers:
        item = items[a["item_id"]]
        if a["viewable"] != "yes":
            excluded[item["system"]] += 1; continue
        by_system[item["system"]].append((item, a))
    systems = {}
    per_utterance = collections.defaultdict(dict)                   # system -> utterance -> mean choice accuracy
    for system in key["systems"]:
        rows = by_system.get(system, [])
        choice = [a["choice"] == i["correct"] for i, a in rows]
        kind = [a["sentence_type"] == i["truth"]["sentence_type"] for i, a in rows]
        out = dict(answers=len(rows), not_viewable=excluded[system],
                   comprehension_choice=dict(**_proportion(choice), chance=.25),
                   sentence_type=_proportion(kind),
                   sentence_type_by_truth={t: _proportion([a["sentence_type"] == t for i, a in rows
                                                           if i["truth"]["sentence_type"] == t])
                                           for t in SENTENCE_TYPES},
                   negation_recall=_proportion([a["negated"] == "yes" for i, a in rows if i["truth"]["negated"] == "yes"]),
                   negation_false_alarm=_proportion([a["negated"] == "yes" for i, a in rows
                                                     if i["truth"]["negated"] == "no"]),
                   topic_recall=_proportion([a["topic"] == "yes" for i, a in rows if i["truth"]["topic"] == "yes"]),
                   topic_false_alarm=_proportion([a["topic"] == "yes" for i, a in rows if i["truth"]["topic"] == "no"]))
        if rows:
            nat = [a["naturalness"] for _, a in rows]
            lo, hi = bootstrap_ci(nat, seed=seed) if len(nat) > 1 else (nat[0], nat[0])
            out["naturalness"] = dict(n=len(nat), mean=round(sum(nat) / len(nat), 3), ci95=[round(lo, 3), round(hi, 3)])
            translated = [(a["translation"], i["english"]) for i, a in rows if a["translation"]]
            if translated:
                out["translation_bleu_indicative"] = round(float(corpus_bleu([h for h, _ in translated],
                                                                             [r for _, r in translated])[0]), 4)
            out["comprehension_choice"]["above_chance"] = out["comprehension_choice"]["ci95"][0] > .25
        hits = collections.defaultdict(list)
        for i, a in rows:
            hits[i["utterance"]].append(float(a["choice"] == i["correct"]))
        per_utterance[system] = {u: sum(v) / len(v) for u, v in hits.items()}
        systems[system] = out
    paired = {}
    if reference is not None and reference in per_utterance:
        ref = per_utterance[reference]
        for system in key["systems"]:
            if system == reference:
                continue
            common = sorted(set(ref) & set(per_utterance[system]))
            if len(common) < 2:
                continue
            a = [per_utterance[system][u] for u in common]
            b = [ref[u] for u in common]
            paired[system] = dict(utterances=len(common), mean_difference=round((sum(a) - sum(b)) / len(common), 4),
                                  p_value=round(paired_permutation_pvalue(a, b, seed=seed), 4))
    # agreement: items rated by the most common number (>= 2) of reviewers
    ratings = collections.defaultdict(list)
    for answers_of in by_system.values():
        for i, a in answers_of:
            ratings[i["item_id"]].append(a)
    agreement = {}
    counts = collections.Counter(len(v) for v in ratings.values() if len(v) >= 2)
    if counts:
        k = counts.most_common(1)[0][0]
        rated = [v for v in ratings.values() if len(v) == k]
        for name, cats, field in (("choice", OPTIONS, "choice"), ("sentence_type", SENTENCE_TYPES, "sentence_type"),
                                  ("naturalness", NATURALNESS, "naturalness")):
            table = [[sum(1 for a in v if a[field] == c) for c in cats] for v in rated]
            agreement[name] = dict(items=len(rated), raters=k, fleiss_kappa=round(fleiss_kappa(table, len(cats)), 4))
    return dict(systems=systems, paired_with_reference=paired, reference=reference, agreement=agreement,
                reviewers=len({a["reviewer_sha256"] for a in answers}), answers=len(answers))


def key_sha256(key: Mapping) -> str:
    import json
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


__all__ = ["OPTIONS", "SENTENCE_TYPES", "SHEET_COLUMNS", "grammatical_truth", "latin_square", "distractors",
           "design_study", "write_sheet", "parse_sheet", "merge_answers", "wilson", "score_study", "key_sha256"]
