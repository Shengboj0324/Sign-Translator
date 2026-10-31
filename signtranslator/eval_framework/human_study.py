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

**How2Sign studies (no ASLLRP markers; first Deaf-rater read, docs/NEXT_PHASE_PLAN.md "Rater read terms defined").** An
utterance without ``markers`` carries its ``truth`` directly (``negated`` from the checked negation truth; the sentence
type and topic have no truth, ``None``, and are left out of their scores). ``draw_study_sentences``, the negation sheet
and ``negation_truth`` implement the sentence draw and the negation truth; ``bar_read`` reads the declared release-bar
terms.
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


def _truth_of(utterance: Mapping) -> dict:
    """The plan markers' truth, or the utterance's own ``truth`` when it has no markers (a How2Sign study)."""
    if "markers" in utterance:
        return grammatical_truth(utterance["markers"])
    truth = utterance.get("truth") or {}
    return dict(sentence_type=truth.get("sentence_type"), negated=truth.get("negated"), topic=truth.get("topic"))


def latin_square(utterances: int, systems: int) -> list[list[int]]:
    """``assignment[g][u]`` = the system group ``g`` sees utterance ``u`` from."""
    if utterances < systems:
        raise ValueError("need at least as many utterances as systems for every group to see every system")
    return [[(u + g) % systems for u in range(utterances)] for g in range(systems)]


def _same_text(text: str) -> str:
    return text.strip().lower()


def distractors(english: Sequence[str], index: int, rng: random.Random, *, count: int = 3,
                text_key=_same_text) -> list[int]:
    """Indices of ``count`` other sentences with the closest word counts (ties broken at random), never the same text
    (``text_key``: what counts as the same text; default: case and outer spaces ignored)."""
    target = english[index]
    length = len(target.split())
    pool = [j for j, e in enumerate(english) if j != index and text_key(e) != text_key(target)]
    if len(pool) < count:
        raise ValueError("not enough distinct sentences for distractors")
    rng.shuffle(pool)
    pool.sort(key=lambda j: abs(len(english[j].split()) - length))
    chosen, seen = [], {text_key(target)}
    for j in pool:
        key = text_key(english[j])
        if key not in seen:
            chosen.append(j); seen.add(key)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise ValueError("not enough distinct sentences for distractors")
    return chosen


def design_study(utterances: Sequence[Mapping], systems: Sequence[str], *, seed: int,
                 distractor_pool: Sequence[str] | None = None, text_key=_same_text) -> dict:
    """The study key: every item (id, group, utterance, system, options, correct option, grammatical truth).

    ``utterances``: dicts with ``sign`` (the record id), ``english`` and ``markers`` (the plan's marker vectors) or,
    without markers, ``truth`` (``sentence_type``, ``negated``, ``topic``; missing: None, no truth for that question).
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
        offered = [english[i]] + [pool[j] for j in distractors(pool, target, rng, text_key=text_key)]
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
                              truth=_truth_of(utterances[u])))
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
        kind = [a["sentence_type"] == i["truth"]["sentence_type"] for i, a in rows
                if i["truth"]["sentence_type"] is not None]
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


# ---- sentence draw and negation truth (plan, "Rater read terms defined") --------------------------------------------

NEGATION_COLUMNS = ("sign", "english", "negation_words", "no_generated_clip_seen", "confirmed", "checker_sha256",
                    "notes")
# The English rule only makes candidates, so it leans towards recall: the Deaf checker decides. Semi-negatives (hardly,
# barely, rarely, seldom, few) and negative affixes (unhappy) are not candidates.
_NEGATION = re.compile(r"\b(?:\w*n['’]t|not|no|never|none|nobody|nothing|nowhere|neither|nor|cannot|without|aint|"
                       r"dont|doesnt|didnt|isnt|arent|wasnt|werent|cant|couldnt|wouldnt|shouldnt|wont|hasnt|havent|"
                       r"hadnt|mustnt|neednt)\b", re.IGNORECASE)


def negation_words(english: str) -> list[str]:
    """The negation words of an English sentence, lower-cased, distinct, in order (empty: no negation)."""
    return list(dict.fromkeys(m.lower().replace("’", "'") for m in _NEGATION.findall(english)))


def draw_study_sentences(eligible: Sequence[Mapping], *, seed: int, main: int = 200, pilot: int = 20) -> dict:
    """``dict(main=[sign], pilot=[sign])`` from ``eligible`` (rows with ``sign`` and ``negation_words``, in a fixed
    order): every negation candidate is kept among ``main`` and the rest of ``main``, then ``pilot``, are a random
    sample of the other sentences, so the pilot holds no candidate. Both lists keep ``eligible``'s order."""
    signs = [r["sign"] for r in eligible]
    if len(set(signs)) != len(signs):
        raise ValueError("the eligible sentences repeat a clip")
    kept = [r["sign"] for r in eligible if r["negation_words"]]
    others = [r["sign"] for r in eligible if not r["negation_words"]]
    if len(kept) > main:
        raise ValueError(f"{len(kept)} negation candidates do not fit in {main} sentences")
    extra = main - len(kept)
    if len(others) < extra + pilot:
        raise ValueError(f"{len(eligible)} eligible sentences cannot give {main} + {pilot}")
    drawn = random.Random(seed).sample(others, extra + pilot)
    chosen, trial = set(kept) | set(drawn[:extra]), set(drawn[extra:])
    return dict(main=[s for s in signs if s in chosen], pilot=[s for s in signs if s in trial])


def write_negation_sheet(candidates: Sequence[Mapping]) -> str:
    """The Deaf checker's sheet: one row per negation candidate (``sign`` names the real recording), answer cells empty.
    The checker is shown the real recording only, never a generated clip."""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=NEGATION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for c in candidates:
        writer.writerow(dict(sign=c["sign"], english=c["english"], negation_words=" ".join(c["negation_words"]),
                             no_generated_clip_seen="", confirmed="", checker_sha256="", notes=""))
    return out.getvalue()


def parse_negation_sheet(text: str, candidates: Sequence[Mapping]) -> tuple[dict, list[dict]]:
    """({sign: yes/no}, problems) of the returned sheet. Every candidate must be answered once, the offered columns
    unchanged, ``confirmed`` yes or no, ``no_generated_clip_seen`` yes and the checker a hashed key, never a name."""
    want = {c["sign"]: c for c in candidates}
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if tuple(reader.fieldnames or ()) != NEGATION_COLUMNS:
        return {}, [dict(row=0, issue=f"columns must be exactly {', '.join(NEGATION_COLUMNS)}")]
    truth, problems = {}, []
    for n, row in enumerate(reader, start=2):
        row = {k: (v or "").strip() for k, v in row.items()}
        c = want.get(row["sign"])
        if c is None:
            problems.append(dict(row=n, issue="not a negation candidate: its English has no negation")); continue
        if row["sign"] in truth:
            problems.append(dict(row=n, sign=row["sign"], issue="answered twice")); continue
        issues = []
        if row["english"] != c["english"] or row["negation_words"] != " ".join(c["negation_words"]):
            issues.append("offered columns were changed")
        if row["confirmed"].lower() not in YES_NO:
            issues.append("confirmed must be yes or no")
        if row["no_generated_clip_seen"].lower() != "yes":
            issues.append("the checker must not have been shown a generated clip")
        if not _SHA.fullmatch(row["checker_sha256"]):
            issues.append("checker_sha256 must be a 64-hex SHA-256 key, never a name")
        if issues:
            problems.extend(dict(row=n, sign=row["sign"], issue=i) for i in issues); continue
        truth[row["sign"]] = row["confirmed"].lower()
    missing = sorted(set(want) - set(truth) - {p.get("sign") for p in problems})
    problems.extend(dict(row=0, sign=s, issue="candidate not answered") for s in missing)
    return truth, problems


def negation_truth(sign: str, confirmed: Mapping[str, str]) -> str:
    """``yes`` only for a sentence the checker confirmed: its English has a negation and the real recording signs it.
    A candidate the checker rejected, like a sentence with no negation in its English, is ``no``."""
    return "yes" if confirmed.get(sign) == "yes" else "no"


# ---- the release-bar terms (docs/HUMAN_EVALUATION_PROTOCOL.md section 6, fixed before any sheet) ---------------------

BAR = dict(comprehension_lower=.70, within_points=.10, recall_lower=.70, false_alarm_upper=.20, unviewable_share=.05)
NOT_READ_HERE = ("wrist_speed_0.8_1.25", "predicted_timing", "unseen_signers")


def _bar_rows(answers: Sequence[Mapping], key: Mapping) -> dict:
    """The viewable answers as arrays, with the indices the bootstrap resamples."""
    import numpy as np
    items = {i["item_id"]: i for i in key["items"]}
    systems = list(key["systems"])
    raters = sorted({a["reviewer_sha256"] for a in answers})
    group_of = {a["reviewer_sha256"]: items[a["item_id"]]["group"] for a in answers}
    viewable = [a for a in answers if a["viewable"] == "yes"]
    negated = {i["utterance"]: i["truth"]["negated"] == "yes" for i in key["items"]}
    field = lambda f: np.array([f(items[a["item_id"]], a) for a in viewable])   # noqa: E731
    return dict(systems=systems, raters=raters, group_of=group_of, utterances=max(negated) + 1,
                negated=np.array([negated[u] for u in range(max(negated) + 1)]),
                utterance=field(lambda i, a: i["utterance"]).astype(int).reshape(-1),
                system=field(lambda i, a: systems.index(i["system"])).astype(int).reshape(-1),
                rater=field(lambda i, a: raters.index(a["reviewer_sha256"])).astype(int).reshape(-1),
                correct=field(lambda i, a: a["choice"] == i["correct"]).astype(float).reshape(-1),
                said=field(lambda i, a: a["negated"] == "yes").astype(float).reshape(-1))


def _bar_statistics(rows: Mapping, share: float, reference: int, sentence_weight, rater_weight) -> dict:
    """Every bootstrapped quantity under one weighting of sentences and raters (all ones: the point estimates). Rates
    pool the ratings; differences are per sentence, reference minus system, over the sentences both have, then
    averaged by sentence: paired. Weighted: the two strata (confirmed negated, the rest) mixed by ``share``."""
    import numpy as np
    S, U = len(rows["systems"]), rows["utterances"]
    weight = sentence_weight[rows["utterance"]] * rater_weight[rows["rater"]]
    cell = rows["utterance"] * S + rows["system"]
    W = np.bincount(cell, weight, U * S).reshape(U, S)
    C = np.bincount(cell, weight * rows["correct"], U * S).reshape(U, S)
    Y = np.bincount(cell, weight * rows["said"], U * S).reshape(U, S)
    neg = rows["negated"]

    def ratio(a, b):
        return float(a / b) if b > 0 else float("nan")

    def paired(values, mask, s):
        both = mask & (W[:, reference] > 0) & (W[:, s] > 0)
        d = values[both, reference] / W[both, reference] - values[both, s] / W[both, s]
        return ratio((sentence_weight[both] * d).sum(), sentence_weight[both].sum())
    out = {}
    for s in range(S):
        negated_rate = ratio(C[neg, s].sum(), W[neg, s].sum())
        other_rate = ratio(C[~neg, s].sum(), W[~neg, s].sum())
        row = dict(weighted=_mix(share, negated_rate, other_rate), unweighted=ratio(C[:, s].sum(), W[:, s].sum()),
                   recall=ratio(Y[neg, s].sum(), W[neg, s].sum()),
                   false_alarm=ratio(Y[~neg, s].sum(), W[~neg, s].sum()))
        if s != reference:
            row.update(minus_weighted=_mix(share, paired(C, neg, s), paired(C, ~neg, s)),
                       minus_unweighted=paired(C, np.ones(U, bool), s), minus_recall=paired(Y, neg, s))
        out[s] = row
    return out


def _mix(share: float, negated: float, other: float) -> float:
    """``share`` x the negated stratum + the rest x the other; a stratum with no weight needs no data."""
    if share == 0:
        return other
    if share == 1:
        return negated
    return share * negated + (1 - share) * other


def _bar_bootstrap(rows: Mapping, share: float, reference: int, *, boot: int, seed: int) -> dict:
    """Two-way cluster bootstrap (the pigeonhole bootstrap), as declared and with no further structure: per replicate,
    all sentences with replacement and, independently, all raters with replacement; a rating's weight is the product of
    its sentence's and its rater's counts. A replicate can lose a Latin-square group, and then a system's cells of that
    group: its differences read the sentences both systems still have. This bootstrap counts the rating noise about
    three times (Owen 2007), so it is wider than Wilson even with no clustering: conservative, never looser.
    {system: {quantity: array}}."""
    import numpy as np
    if boot < 1:
        raise ValueError("the bootstrap needs at least one replicate")
    rng = np.random.default_rng(seed)
    U, R = rows["utterances"], len(rows["raters"])
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for _ in range(boot):
        sentence_weight = np.bincount(rng.integers(0, U, U), minlength=U).astype(float)
        rater_weight = np.bincount(rng.integers(0, R, R), minlength=R).astype(float)
        for s, row in _bar_statistics(rows, share, reference, sentence_weight, rater_weight).items():
            for name, value in row.items():
                out[s][name].append(value)
    return {s: {name: np.array(v) for name, v in row.items()} for s, row in out.items()}


def _interval(values) -> list | None:
    """Two-sided 95% percentile interval, None when over 1% of the replicates leave the quantity undefined."""
    import numpy as np
    finite = values[np.isfinite(values)]
    if len(finite) < .99 * len(values) or not len(finite):
        return None
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def _condition(value, threshold: float, *, at_least: bool, void: bool = False) -> dict:
    """One bar condition on the unrounded value: ``pass`` or ``fail``; ``void`` (the negation read has no valid
    reference); ``not_read`` when the data give no value, which never passes."""
    result = ("void" if void else "not_read" if value is None else
              "pass" if (value >= threshold if at_least else value <= threshold) else "fail")
    return dict(value=value, threshold=threshold, rule="at least" if at_least else "at most", result=result)


def bar_read(answers: Sequence[Mapping], key: Mapping, *, reference: str, boot: int = 2000, seed: int = 0) -> dict:
    """Per system and per condition, the release-bar terms as declared: comprehension weighted to the negated share,
    unweighted beside it, Wilson and clustered lower bounds; upper bound of reference minus system; negation recall and
    false-alarm rules and the void rule; the unviewable rule. Intervals: two-way cluster bootstrap (``_bar_bootstrap``).

    Reading choices the declaration leaves open, stated in the output (``readings``): the Wilson bound of the weighted
    rate uses Kish's effective sample size; "95% bound" is the bound of the two-sided 95% interval; one rater alone has
    no rater variance, so the clustered read is ``not_read``. Wrist speed, timing and unseen signers are not read here:
    this scorer never says a system may be released."""
    import numpy as np
    if reference not in key["systems"]:
        raise ValueError(f"{reference} is not a system of this study")
    bar = key.get("bar") or {}
    share = bar.get("negated_share")
    if share is None or not 0 <= share <= 1:
        raise ValueError("the key declares no negated share (key['bar']['negated_share'])")
    if any(i["truth"]["negated"] not in YES_NO for i in key["items"]):
        raise ValueError("the key has a sentence with no negation truth")
    items = {i["item_id"]: i for i in key["items"]}
    rows = _bar_rows(answers, key)
    ref = rows["systems"].index(reference)
    U, R = rows["utterances"], len(rows["raters"])
    point = _bar_statistics(rows, share, ref, np.ones(U), np.ones(R))
    group_sizes = collections.Counter(rows["group_of"].values())
    clustered = R >= 2
    replicates = _bar_bootstrap(rows, share, ref, boot=boot, seed=seed) if clustered else None
    interval = (lambda s, name: _interval(replicates[s][name])) if clustered else (lambda s, name: None)   # noqa: E731
    rounded = lambda v: None if v is None or not math.isfinite(v) else round(v, 4)   # noqa: E731
    answered = collections.Counter(items[a["item_id"]]["system"] for a in answers)
    hidden = collections.Counter(items[a["item_id"]]["system"] for a in answers if a["viewable"] != "yes")
    report = {}
    for s, name in enumerate(rows["systems"]):
        mine = rows["system"] == s
        neg = rows["negated"][rows["utterance"]]
        n_negated, n_other = int((mine & neg).sum()), int((mine & ~neg).sum())
        recall_hits, alarms = int(rows["said"][mine & neg].sum()), int(rows["said"][mine & ~neg].sum())
        hits = int(rows["correct"][mine].sum())
        kish = _kish(share, n_negated, n_other)
        rate = point[s]["weighted"]
        wilson_weighted = wilson(rate * kish, kish) if kish and math.isfinite(rate) else None
        recall_ci, alarm_ci = wilson(recall_hits, n_negated), wilson(alarms, n_other)
        out = dict(
            ratings=int(mine.sum()), negated_ratings=n_negated, other_ratings=n_other,
            comprehension=dict(
                weighted=dict(rate=rounded(rate), n_effective=round(kish, 1), share=share,
                              wilson95=list(wilson_weighted) if wilson_weighted else None,
                              clustered95=interval(s, "weighted")),
                unweighted=dict(rate=rounded(point[s]["unweighted"]),
                                wilson95=list(wilson(hits, int(mine.sum()))),
                                clustered95=interval(s, "unweighted"))),
            negation=dict(recall=dict(n=n_negated, rate=rounded(point[s]["recall"]),
                                      wilson95=list(recall_ci), clustered95=interval(s, "recall")),
                          false_alarm=dict(n=n_other, rate=rounded(point[s]["false_alarm"]),
                                           wilson95=list(alarm_ci),
                                           clustered95=interval(s, "false_alarm"))),
            unviewable=dict(rows=answered[name], not_viewable=hidden[name],
                            share=hidden[name] / answered[name] if answered[name] else None))
        if s != ref:
            out["reference_minus_system"] = dict(
                weighted=dict(difference=rounded(point[s]["minus_weighted"]),
                              clustered95=interval(s, "minus_weighted")),
                unweighted=dict(difference=rounded(point[s]["minus_unweighted"]),
                                clustered95=interval(s, "minus_unweighted")),
                recall=dict(difference=rounded(point[s]["minus_recall"]), clustered95=interval(s, "minus_recall")))
        report[name] = out
    reference_recall = report[reference]["negation"]["recall"]
    void = reference_recall["n"] == 0 or reference_recall["wilson95"][0] < BAR["recall_lower"]
    for name, out in report.items():
        if name == reference:
            continue
        comp, gap, negation = out["comprehension"]["weighted"], out["reference_minus_system"], out["negation"]
        out["conditions"] = dict(
            comprehension_wilson_lower=_condition(comp["wilson95"][0] if comp["wilson95"] else None,
                                                  BAR["comprehension_lower"], at_least=True),
            comprehension_clustered_lower=_condition(comp["clustered95"][0] if comp["clustered95"] else None,
                                                     BAR["comprehension_lower"], at_least=True),
            within_10_points_upper=_condition(
                gap["weighted"]["clustered95"][1] if gap["weighted"]["clustered95"] else None,
                BAR["within_points"], at_least=False),
            negation_recall_wilson_lower=_condition(
                negation["recall"]["wilson95"][0] if negation["recall"]["n"] else None,
                BAR["recall_lower"], at_least=True, void=void),
            negation_recall_within_10_points_upper=_condition(
                gap["recall"]["clustered95"][1] if gap["recall"]["clustered95"] else None,
                BAR["within_points"], at_least=False, void=void),
            negation_false_alarm_wilson_upper=_condition(
                negation["false_alarm"]["wilson95"][1] if negation["false_alarm"]["n"] else None,
                BAR["false_alarm_upper"], at_least=False, void=void),
            unviewable_share=_condition(out["unviewable"]["share"], BAR["unviewable_share"], at_least=False))
        out["passes_rater_terms"] = all(c["result"] == "pass" for c in out["conditions"].values())
    said = collections.defaultdict(list)
    for a in answers:
        if a["viewable"] == "yes":
            said[a["reviewer_sha256"]].append((items[a["item_id"]]["truth"]["negated"], a["negated"] == "yes"))
    raters = {r[:12]: dict(group=rows["group_of"][r], ratings=len(v),
                           says_negated_on_negated=_share([y for t, y in v if t == "yes"]),
                           says_negated_on_others=_share([y for t, y in v if t != "yes"]),
                           always_says_negated=bool(v) and all(y for _, y in v))
               for r, v in sorted(said.items())}
    return dict(terms="docs/HUMAN_EVALUATION_PROTOCOL.md section 6; docs/NEXT_PHASE_PLAN.md 'Rater read terms defined'",
                thresholds=BAR, reference=reference, negation_read_void=void,
                reference_recall_wilson_lower=reference_recall["wilson95"][0], negated_share=share,
                share_population=bar.get("population"), bootstrap=dict(replicates=boot if clustered else 0, seed=seed),
                clustered_read=clustered, raters_per_group=dict(sorted(group_sizes.items())), systems=report,
                raters=raters,
                readings=dict(
                    wilson_of_weighted_rate="Wilson at the weighted rate with Kish's effective sample size",
                    bound_95="the bound of the two-sided 95% interval (percentile interval for the bootstrap)",
                    resampling="all sentences and all raters with replacement, no strata (pigeonhole bootstrap)",
                    lone_rater="one rater alone has no rater variance: the clustered read is not_read",
                    undefined_replicates="over 1% of replicates without a value: that bound is not_read",
                    false_alarm_denominator="every sentence not confirmed negated, rejected candidates included"),
                not_read_here=list(NOT_READ_HERE), may_release=False,
                may_release_reason="this scorer does not read wrist speed, timing or unseen signers")


def _kish(share: float, negated: int, other: int) -> float:
    """Kish's effective sample size of the rate mixing two strata by ``share``."""
    v = (share * share / negated if negated else 0. if share == 0 else math.inf) + \
        ((1 - share) ** 2 / other if other else 0. if share == 1 else math.inf)
    return 0. if v == 0 or math.isinf(v) else 1 / v


def _share(hits: Sequence[bool]) -> float | None:
    return round(sum(hits) / len(hits), 4) if hits else None




def key_sha256(key: Mapping) -> str:
    import json
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


__all__ = ["OPTIONS", "SENTENCE_TYPES", "SHEET_COLUMNS", "grammatical_truth", "latin_square", "distractors",
           "design_study", "write_sheet", "parse_sheet", "merge_answers", "wilson", "score_study", "key_sha256",
           "NEGATION_COLUMNS", "negation_words", "draw_study_sentences", "write_negation_sheet",
           "parse_negation_sheet", "negation_truth", "BAR", "bar_read"]
