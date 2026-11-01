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
terms. Such a study uses the two-stage sheet: everything but the forced choice is answered and returned before the options
are shown (``write_stage1_sheet``, ``write_stage2_sheet``, ``read_two_stage``), because the options hint at negation.
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
    return dict(sentence_type=truth.get("sentence_type"), negated=truth.get("negated"), topic=truth.get("topic"),
                **({"stratum": truth["stratum"]} if "stratum" in truth else {}))


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
                 distractor_pool: Sequence[str] | None = None, text_key=_same_text, new_item_id=None) -> dict:
    """The study key: every item (id, group, utterance, system, options, correct option, grammatical truth).

    ``utterances``: dicts with ``sign`` (the record id), ``english`` and ``markers`` (the plan's marker vectors) or,
    without markers, ``truth`` (``sentence_type``, ``negated``, ``topic``; missing: None, no truth for that question).
    Every utterance gets one set of options shared by all its systems, so systems differ only in the video.
    Distractors come from ``distractor_pool`` (pass every sentence of the partition): drawn only from the study's own
    sentences, each sentence would answer exactly one item and a reviewer could answer by elimination.
    ``new_item_id``: where item ids come from (default: the seeded generator, reproducible from the key's seed); a
    study that must not derive them from the design passes a secret source.
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
                item_id = new_item_id() if new_item_id else "%012x" % rng.getrandbits(48)
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
    rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    carried = {row["reviewer_sha256"] for row in rows if row["reviewer_sha256"]}
    sheet_key = next(iter(carried)) if len(carried) == 1 else None
    for n, row in enumerate(rows, start=2):
        item = items.get(row["item_id"])
        if item is None:
            problems.append(dict(row=n, issue="unknown item_id")); continue
        if row["viewable"].lower() == "no" and not row["reviewer_sha256"] and sheet_key:
            row["reviewer_sha256"] = sheet_key                      # the guide says to leave the rest of the row empty
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
_NEGATION = re.compile(r"\b(?:\w*n['’`´]t|not|no|noone|nope|nah|never|none|nobody|nothing|nowhere|neither|nor|cannot|"
                       r"without|aint|"
                       r"dont|doesnt|didnt|isnt|arent|wasnt|werent|cant|couldnt|wouldnt|shouldnt|wont|hasnt|havent|"
                       r"hadnt|mustnt|neednt)\b", re.IGNORECASE)


def negation_words(english: str) -> list[str]:
    """The negation words of an English sentence, lower-cased, distinct, in order (empty: no negation)."""
    return list(dict.fromkeys(re.sub("[’`´]", "'", m.lower()) for m in _NEGATION.findall(english)))


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


def negation_stratum(sign: str, confirmed: Mapping[str, str]) -> str:
    """Which of the three strata a sentence is in: ``confirmed``, ``rejected`` (a candidate the checker said no to) or
    ``non_candidate`` (no negation in its English). Candidates are all kept in the draw, non-candidates are sampled."""
    return dict(yes="confirmed", no="rejected").get(confirmed.get(sign), "non_candidate")


def strata_shares(confirmed: Mapping[str, str], eligible: int) -> dict:
    """D20's eligible sentences by stratum: the weights of comprehension."""
    yes = sum(v == "yes" for v in confirmed.values())
    no = sum(v == "no" for v in confirmed.values())
    return dict(confirmed=yes / eligible, rejected=no / eligible, non_candidate=(eligible - yes - no) / eligible)


# ---- the two-stage sheet (plan, 2026-11-01, 03:28 UTC): the options hint at negation, so they come after --------------

STAGE1_COLUMNS = ("item_id", "video", "viewable", "translation", "sentence_type", "negated", "topic", "naturalness",
                  "reviewer_sha256", "notes")
STAGE2_COLUMNS = ("item_id", "video", "option_a", "option_b", "option_c", "option_d", "choice", "reviewer_sha256",
                  "stage1_sha256")
_STAGE1_ANSWERS = ("viewable", "translation", "sentence_type", "negated", "topic", "naturalness", "notes")


def write_stage1_sheet(key: Mapping, group: int) -> str:
    """Stage 1, what one rater of ``group`` fills: presentation order, no options anywhere."""
    rows = sorted((i for i in key["items"] if i["group"] == group), key=lambda i: i["position"])
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=STAGE1_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for i in rows:
        writer.writerow(dict(item_id=i["item_id"], video=f"{i['item_id']}.mp4",
                             **{c: "" for c in STAGE1_COLUMNS[2:]}))
    return out.getvalue()


def _sheet_key(rows: Sequence[Mapping]) -> tuple[str | None, list[dict]]:
    """The one reviewer key of a sheet, from the rows that carry one. Several, or a name: refused."""
    keys = {r["reviewer_sha256"] for r in rows if r["reviewer_sha256"]}
    if len(keys) > 1:
        return None, [dict(row=0, issue="one sheet must carry one reviewer key")]
    return (next(iter(keys)) if keys else None), []


def parse_stage1(text: str, key: Mapping) -> tuple[list[dict], list[dict]]:
    """(answers, problems) of a returned stage 1 sheet. A row is answered when any answer cell is filled; ``viewable=no``
    needs no key on a sheet whose key is otherwise known. A problem row is dropped and carries the sheet's key and item."""
    items = {i["item_id"]: i for i in key["items"]}
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if tuple(reader.fieldnames or ()) != STAGE1_COLUMNS:
        return [], [dict(row=0, issue=f"columns must be exactly {', '.join(STAGE1_COLUMNS)}")]
    rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    reviewer, problems = _sheet_key(rows)
    if problems:
        return [], problems
    answers = []
    for n, row in enumerate(rows, start=2):
        item = items.get(row["item_id"])
        if item is None:
            problems.append(dict(row=n, issue="unknown item_id")); continue
        if not any(row[c] for c in _STAGE1_ANSWERS):
            continue                                                # unanswered
        viewable = row["viewable"].lower() or "yes"
        issues = []
        if row["video"] != f"{item['item_id']}.mp4":
            issues.append("offered columns were changed")
        if viewable not in YES_NO:
            issues.append("viewable must be yes or no")
        if reviewer is None or not _SHA.fullmatch(reviewer):
            issues.append("reviewer_sha256 must be a 64-hex SHA-256 key, never a name")
        if viewable == "yes":
            if row["sentence_type"] not in SENTENCE_TYPES:
                issues.append(f"sentence_type must be one of {', '.join(SENTENCE_TYPES)}")
            for c in ("negated", "topic"):
                if row[c].lower() not in YES_NO:
                    issues.append(f"{c} must be yes or no")
            if row["naturalness"] not in {str(v) for v in NATURALNESS}:
                issues.append("naturalness must be 1-5")
        if issues:
            problems.extend(dict(row=n, item_id=item["item_id"], issue=i, reviewer_sha256=reviewer) for i in issues)
            continue
        answers.append(dict(item_id=item["item_id"], reviewer_sha256=reviewer, viewable=viewable,
                            translation=row["translation"], choice="", sentence_type=row["sentence_type"],
                            negated=row["negated"].lower(), topic=row["topic"].lower(),
                            naturalness=int(row["naturalness"]) if viewable == "yes" else None, notes=row["notes"]))
    return answers, problems


def write_stage2_sheet(key: Mapping, answers: Sequence[Mapping], stage1_sha256: str) -> str:
    """Stage 2, released only after stage 1 came back: one row per item the rater could view, with the options, the
    rater's key and the returned stage 1's hash already filled."""
    items = {i["item_id"]: i for i in key["items"]}
    reviewers = {a["reviewer_sha256"] for a in answers}
    if len(reviewers) != 1:
        raise ValueError("stage 2 is released to one rater")
    reviewer = next(iter(reviewers))
    rows = sorted((items[a["item_id"]] for a in answers if a["viewable"] == "yes"), key=lambda i: i["position"])
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=STAGE2_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for i in rows:
        writer.writerow(dict(item_id=i["item_id"], video=f"{i['item_id']}.mp4",
                             **{f"option_{o}": t for o, t in zip(OPTIONS, i["options"])}, choice="",
                             reviewer_sha256=reviewer, stage1_sha256=stage1_sha256))
    return out.getvalue()


def parse_stage2(text: str, key: Mapping, stage1: Mapping[str, Mapping]) -> tuple[dict, list[dict]]:
    """({(reviewer, item): choice}, problems) of a returned stage 2 sheet. ``stage1``: {reviewer: {"sha256": the returned
    stage 1 sheet's hash, "answers": {item: its answer}}}. An answer without a matching stage 1 hash is refused, as
    is one for an item the rater did not view in stage 1. The rater's key must be the stage 1 key."""
    items = {i["item_id"]: i for i in key["items"]}
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if tuple(reader.fieldnames or ()) != STAGE2_COLUMNS:
        return {}, [dict(row=0, issue=f"columns must be exactly {', '.join(STAGE2_COLUMNS)}")]
    rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    reviewer, problems = _sheet_key(rows)
    if problems:
        return {}, problems
    if reviewer not in stage1:
        return {}, [dict(row=0, issue="stage 2 from a reviewer key with no stage 1 sheet")]
    choices = {}
    for n, row in enumerate(rows, start=2):
        item = items.get(row["item_id"])
        if item is None:
            problems.append(dict(row=n, issue="unknown item_id")); continue
        if not row["choice"]:
            continue                                                # unanswered
        issues = []
        if row["stage1_sha256"] != stage1[reviewer]["sha256"]:
            issues.append("stage 2 answer without a matching stage 1 hash")
        if row["video"] != f"{item['item_id']}.mp4" or [row[f"option_{o}"] for o in OPTIONS] != list(item["options"]):
            issues.append("offered columns were changed")
        if row["choice"].lower() not in OPTIONS:
            issues.append("choice must be a, b, c or d")
        first = stage1[reviewer]["answers"].get(item["item_id"])
        if first is None or first["viewable"] != "yes":
            issues.append("no viewable stage 1 answer for this item")
        if issues:
            problems.extend(dict(row=n, item_id=item["item_id"], issue=i, reviewer_sha256=reviewer) for i in issues)
            continue
        choices[(reviewer, item["item_id"])] = row["choice"].lower()
    return choices, problems


def read_two_stage(stage1_sheets: Sequence[bytes], stage2_sheets: Sequence[bytes], key: Mapping,
                   receipts: Mapping[str, str]) -> tuple[list[dict], list[dict]]:
    """(answers, problems) of the returned sheets. ``receipts``: {reviewer: the stage 1 hash the team recorded when it
    released stage 2}, so a stage 1 edited after the options were seen is refused even if its stage 2 hash was edited
    too. A viewable row without a stage 2 choice is not an answer: it counts as blank."""
    stage1, problems = {}, []
    for sheet in stage1_sheets:
        answers, issues = parse_stage1(sheet.decode("utf-8-sig"), key)
        problems += issues
        if not answers:
            continue
        reviewer = answers[0]["reviewer_sha256"]
        sha = hashlib.sha256(sheet).hexdigest()
        if reviewer in stage1:
            problems.append(dict(row=0, reviewer_sha256=reviewer, issue="two stage 1 sheets from one reviewer key"))
            continue
        if receipts.get(reviewer) != sha:
            problems.append(dict(row=0, reviewer_sha256=reviewer, issue="stage 1 sheet does not match the recorded receipt"))
            continue
        stage1[reviewer] = dict(sha256=sha, answers={a["item_id"]: a for a in answers})
    choices = {}
    for sheet in stage2_sheets:
        got, issues = parse_stage2(sheet.decode("utf-8-sig"), key, stage1)
        problems += issues
        twice = set(got) & set(choices)
        problems += [dict(row=0, reviewer_sha256=r, item_id=i, issue="answered the item twice in stage 2; excluded")
                     for r, i in sorted(twice)]
        choices.update({k: v for k, v in got.items() if k not in twice})
    out = []
    for reviewer, sheet in stage1.items():
        for item_id, a in sheet["answers"].items():
            if a["viewable"] != "yes":
                out.append(a)
            elif (reviewer, item_id) in choices:
                out.append(dict(a, choice=choices[(reviewer, item_id)]))
    return out, problems


# ---- the release-bar terms (docs/HUMAN_EVALUATION_PROTOCOL.md section 6; interval method declared 2026-11-01, 03:28 UTC) -

BAR = dict(comprehension_lower=.70, within_points=.10, recall_lower=.70, false_alarm_upper=.20, unviewable_share=.05,
           coverage=.95, comprehension_alpha=.04, gap_alpha=.05)
NOT_READ_HERE = ("wrist_speed_0.8_1.25", "predicted_timing", "unseen_signers")
STRATA = ("confirmed", "rejected", "non_candidate")
AGREEMENT_Z = -3.                # a rater is flagged at this robust z of agreement with the others
IDENTICAL_MIN_ROWS = 10          # fewer shared rows than this can match by chance
_ANSWER_VECTOR = ("choice", "negated", "sentence_type", "topic", "naturalness", "translation")


class BarRefused(ValueError):
    """The read cannot be made: nothing about any system may be concluded. ``details`` says why, for the report."""

    def __init__(self, reason: str, **details):
        super().__init__(reason)
        self.details = dict(refused=reason, **details)


def parse_roster(text: str) -> dict[str, int]:
    """The private roster, JSON {reviewer key: group}: who the raters are, without names."""
    import json
    roster = json.loads(text)
    if not isinstance(roster, dict) or any(not _SHA.fullmatch(str(k)) or type(v) is not int for k, v in roster.items()):
        raise ValueError("the roster must map 64-hex reviewer keys to group numbers")
    return roster


def _bar_check_people(answers: Sequence[Mapping], key: Mapping, roster: Mapping[str, int]) -> None:
    """The private roster is the list of raters: nobody else counts, each rater is in the group the roster says, and
    the Deaf checker, who has seen the real recordings, is not a rater."""
    if not roster:
        raise BarRefused("the roster is empty")
    if any(g not in range(key["groups"]) for g in roster.values()):
        raise BarRefused("the roster names a group the study does not have")
    items = {i["item_id"]: i for i in key["items"]}
    raters = {a["reviewer_sha256"] for a in answers}
    checkers = set(key.get("negation_checkers") or ())
    if checkers & (set(roster) | raters):
        raise BarRefused("the Deaf negation checker cannot be a rater")
    unknown = sorted(r[:12] for r in raters - set(roster))
    if unknown:
        raise BarRefused(f"{len(unknown)} sheets carry keys that are not on the roster", keys=unknown)
    wrong = sorted({a["reviewer_sha256"][:12] for a in answers
                    if items[a["item_id"]]["group"] != roster[a["reviewer_sha256"]]})
    if wrong:
        raise BarRefused(f"{len(wrong)} raters answered another group's sheet", keys=wrong)


def _bar_coverage(answers: Sequence[Mapping], key: Mapping, roster: Mapping[str, int],
                  problems: Sequence[Mapping]) -> dict:
    """Designed rows (every rater of the roster, every item of their group) against the rows returned, per system and
    per group; the rows lost per system: blank between a rater's first and last answered rows, or dropped by validation
    (reported, and counted against the 5% rule)."""
    items = {i["item_id"]: i for i in key["items"]}
    by_group = collections.defaultdict(list)
    for i in key["items"]:
        by_group[i["group"]].append(i)
    answered = collections.defaultdict(set)
    for a in answers:
        answered[a["reviewer_sha256"]].add(a["item_id"])
    designed_s, designed_g, got_s, got_g = (collections.Counter() for _ in range(4))
    blank = collections.Counter()
    for rater, g in roster.items():
        mine = answered[rater]
        positions = [items[i]["position"] for i in mine]
        for i in by_group[g]:
            designed_s[i["system"]] += 1
            designed_g[g] += 1
            if i["item_id"] in mine:
                got_s[i["system"]] += 1
                got_g[g] += 1
            elif positions and min(positions) < i["position"] < max(positions):
                blank[i["system"]] += 1
    dropped = collections.Counter()
    seen = set()
    for p in problems:
        pair = (p.get("reviewer_sha256"), p.get("item_id"))
        if pair[1] in items and pair not in seen:
            seen.add(pair)
            inside = pair[0] in roster and pair[1] not in answered[pair[0]] and answered[pair[0]] and (
                min(items[i]["position"] for i in answered[pair[0]]) < items[pair[1]]["position"]
                < max(items[i]["position"] for i in answered[pair[0]]))
            if not inside:                     # a dropped row inside the answered range is already a blank there
                dropped[items[pair[1]]["system"]] += 1
    return dict(systems={s: dict(designed=designed_s[s], returned=got_s[s], blank_inside=blank[s], dropped=dropped[s],
                                 share=got_s[s] / designed_s[s] if designed_s[s] else 0.) for s in key["systems"]},
                groups={g: dict(designed=designed_g[g], returned=got_g[g],
                                share=got_g[g] / designed_g[g] if designed_g[g] else 0.)
                        for g in range(key["groups"])},
                unattributed_problems=sum(1 for p in problems if p.get("item_id") not in items))


def _bar_require_coverage(coverage: Mapping) -> None:
    short = ({f"system {s}": v["share"] for s, v in coverage["systems"].items() if v["share"] < BAR["coverage"]}
             | {f"group {g}": v["share"] for g, v in coverage["groups"].items() if v["share"] < BAR["coverage"]})
    if short:
        raise BarRefused(f"designed-row coverage is under {BAR['coverage']:.0%}: {sorted(short)}", coverage=coverage,
                         short=short)


def _identical_raters(answers: Sequence[Mapping], items: Mapping) -> list[list[str]]:
    """Pairs of raters whose every shared sentence has the same answers (over ``_ANSWER_VECTOR``): copies, or one person
    under two keys. By sentence, so raters of different groups are compared too."""
    vectors = collections.defaultdict(dict)
    for a in answers:
        if a["viewable"] == "yes":
            vectors[a["reviewer_sha256"]][items[a["item_id"]]["utterance"]] = tuple(a[c] for c in _ANSWER_VECTOR)
    names, out = sorted(vectors), []
    for x, r in enumerate(names):
        for t in names[x + 1:]:
            shared = set(vectors[r]) & set(vectors[t])
            if len(shared) >= IDENTICAL_MIN_ROWS and all(vectors[r][u] == vectors[t][u] for u in shared):
                out.append([r[:12], t[:12]])
    return out


def _rater_agreement(answers: Sequence[Mapping], items: Mapping) -> dict:
    """Each rater's agreement with the others, blind to system: per sentence, the share of the other raters choosing
    the same option whatever system they saw; averaged per rater. Flagged: robust z (median, 1.4826 MAD, floored at the
    rater's own sampling error) at or below ``AGREEMENT_Z``."""
    import numpy as np
    choices = collections.defaultdict(collections.Counter)
    rows = [(a["reviewer_sha256"], items[a["item_id"]]["utterance"], a["choice"]) for a in answers if a["viewable"] == "yes"]
    for _, u, c in rows:
        choices[u][c] += 1
    per = collections.defaultdict(list)
    for r, u, c in rows:
        others = sum(choices[u].values()) - 1
        if others:
            per[r].append((choices[u][c] - 1) / others)
    means = {r: float(np.mean(v)) for r, v in per.items()}
    if len(means) < 3:
        return dict(agreement={r[:12]: m for r, m in means.items()}, flagged=[], flagged_keys=[])
    values = np.array(list(means.values()))
    centre = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - centre)))
    z = {r: (m - centre) / max(mad, math.sqrt(max(m * (1 - m), .01) / len(per[r]))) for r, m in means.items()}
    flagged = sorted(r for r, v in z.items() if v <= AGREEMENT_Z)
    return dict(agreement={r[:12]: round(m, 4) for r, m in means.items()}, z={r[:12]: round(v, 2) for r, v in z.items()},
                flagged=[r[:12] for r in flagged], flagged_keys=flagged)


def _bar_rows(answers: Sequence[Mapping], key: Mapping) -> dict:
    """The viewable answers as arrays, with the indices the variance and the bootstrap cluster on."""
    import numpy as np
    items = {i["item_id"]: i for i in key["items"]}
    systems = list(key["systems"])
    raters = sorted({a["reviewer_sha256"] for a in answers})
    stratum = {i["utterance"]: STRATA.index(i["truth"]["stratum"]) for i in key["items"]}
    U = max(stratum) + 1
    viewable = [a for a in answers if a["viewable"] == "yes"]
    field = lambda f: np.array([f(items[a["item_id"]], a) for a in viewable])   # noqa: E731
    return dict(systems=systems, raters=raters, utterances=U, stratum=np.array([stratum[u] for u in range(U)]),
                utterance=field(lambda i, a: i["utterance"]).astype(int).reshape(-1),
                system=field(lambda i, a: systems.index(i["system"])).astype(int).reshape(-1),
                rater=field(lambda i, a: raters.index(a["reviewer_sha256"])).astype(int).reshape(-1),
                correct=field(lambda i, a: a["choice"] == i["correct"]).astype(float).reshape(-1),
                said=field(lambda i, a: a["negated"] == "yes").astype(float).reshape(-1))


# Method D: two-way cluster variance of influence values, V = sum_u (sum psi)^2 U/(U-1) +
# max(sum_r (sum psi)^2 R/(R-1) - sum psi^2, 0).

def twoway_variance(psi, utterance, rater) -> float:
    """V of the ratings' influence values ``psi``, clustered by sentence and by rater."""
    import numpy as np
    u = np.unique(utterance, return_inverse=True)[1]
    r = np.unique(rater, return_inverse=True)[1]
    U, R = int(u.max()) + 1, int(r.max()) + 1
    by_u, by_r = np.bincount(u, psi, U), np.bincount(r, psi, R)
    sentence = float((by_u ** 2).sum()) * U / (U - 1) if U > 1 else float("nan")
    extra = float((by_r ** 2).sum()) * R / (R - 1) - float((psi ** 2).sum()) if R > 1 else float("nan")
    return sentence + max(extra, 0.) if R > 1 else float("nan")


def _stratified(x, strata, shares):
    """(rate, influence values, kept) of the rate mixing the strata by ``shares``: the rows of a stratum of weight 0
    are out. Rate nan when a stratum with weight has no row."""
    import numpy as np
    keep = np.zeros(len(x), bool)
    psi = np.zeros(len(x))
    rate = 0.
    for k, share in enumerate(shares):
        if share == 0:
            continue
        mine = strata == k
        if not mine.any():
            return float("nan"), psi, keep
        p = float(x[mine].mean())
        rate += share * p
        psi[mine] = share / mine.sum() * (x[mine] - p)
        keep |= mine
    return rate, psi, keep


def lower_bound(rate: float, variance: float, raters: int, ratings: int, alpha: float = BAR["comprehension_alpha"]):
    """One-sided ``alpha`` lower bound of a rate, on the logit scale with t(R - 1); None without a value."""
    if not (math.isfinite(rate) and math.isfinite(variance)) or raters < 2 or ratings < 1:
        return None
    from scipy.stats import t
    q = min(max(rate, .5 / ratings), 1 - .5 / ratings)
    logit = math.log(q / (1 - q)) - float(t.ppf(1 - alpha, raters - 1)) * math.sqrt(variance) / (q * (1 - q))
    return 1 / (1 + math.exp(-logit))


def upper_bound(value: float, variance: float, raters: int, alpha: float = BAR["gap_alpha"]):
    """One-sided ``alpha`` upper bound of a difference with t(R - 1); None without a value."""
    if not (math.isfinite(value) and math.isfinite(variance)) or raters < 2:
        return None
    from scipy.stats import t
    return value + float(t.ppf(1 - alpha, raters - 1)) * math.sqrt(variance)


def _rate_read(rows: Mapping, x, mask, shares) -> dict:
    """A rate over the rows in ``mask`` with its method-D variance and raters."""
    import numpy as np
    strata = rows["stratum"][rows["utterance"]][mask]
    rate, psi, keep = _stratified(x[mask], strata, shares)
    if not keep.any():
        return dict(rate=float("nan"), variance=float("nan"), raters=0, ratings=0)
    u, r = rows["utterance"][mask][keep], rows["rater"][mask][keep]
    return dict(rate=rate, variance=twoway_variance(psi[keep], u, r), raters=len(np.unique(r)), ratings=int(keep.sum()))


def _gap_read(rows: Mapping, x, s: int, reference: int, shares) -> dict:
    """Reference minus system over the sentences both have, mixed by ``shares``; the influence values of both systems'
    ratings share sentences and raters, so the pairing cancels in V."""
    import numpy as np
    has = [np.zeros(rows["utterances"], bool) for _ in range(2)]
    for j, which in enumerate((reference, s)):
        has[j][rows["utterance"][rows["system"] == which]] = True
    common = has[0] & has[1]
    mask = np.isin(rows["system"], [reference, s]) & common[rows["utterance"]]
    if not mask.any():
        return dict(difference=float("nan"), variance=float("nan"), raters=0)
    values, utterance, rater, system = x[mask], rows["utterance"][mask], rows["rater"][mask], rows["system"][mask]
    strata = rows["stratum"][utterance]
    psi = np.zeros(len(values))
    difference = 0.
    keep = np.zeros(len(values), bool)
    for sign, which in ((1., reference), (-1., s)):
        mine = system == which
        rate, part, kept = _stratified(values[mine], strata[mine], shares)
        if not math.isfinite(rate):
            return dict(difference=float("nan"), variance=float("nan"), raters=0)
        difference += sign * rate
        psi[mine] = sign * part
        keep[mine] = kept
    return dict(difference=difference, variance=twoway_variance(psi[keep], utterance[keep], rater[keep]),
                raters=len(np.unique(rater[keep])))


def _bar_statistics(rows: Mapping, shares, reference: int, sentence_weight, rater_weight) -> dict:
    """Every bootstrapped quantity under one weighting of sentences and raters (report-only, beside method D)."""
    import numpy as np
    S, U = len(rows["systems"]), rows["utterances"]
    weight = sentence_weight[rows["utterance"]] * rater_weight[rows["rater"]]
    cell = rows["utterance"] * S + rows["system"]
    W = np.bincount(cell, weight, U * S).reshape(U, S)
    C = np.bincount(cell, weight * rows["correct"], U * S).reshape(U, S)
    Y = np.bincount(cell, weight * rows["said"], U * S).reshape(U, S)
    confirmed = rows["stratum"] == 0

    def ratio(a, b):
        return float(a / b) if b > 0 else float("nan")

    def mix(rate_of):
        return sum(share * rate_of(rows["stratum"] == k) for k, share in enumerate(shares) if share > 0)

    def paired(values, mask, s):
        both = mask & (W[:, reference] > 0) & (W[:, s] > 0)
        d = values[both, reference] / W[both, reference] - values[both, s] / W[both, s]
        return ratio((sentence_weight[both] * d).sum(), sentence_weight[both].sum())
    out = {}
    for s in range(S):
        row = dict(weighted=mix(lambda m, s=s: ratio(C[m, s].sum(), W[m, s].sum())),
                   unweighted=ratio(C[:, s].sum(), W[:, s].sum()),
                   recall=ratio(Y[confirmed, s].sum(), W[confirmed, s].sum()),
                   false_alarm=ratio(Y[~confirmed, s].sum(), W[~confirmed, s].sum()))
        if s != reference:
            row.update(minus_weighted=mix(lambda m, s=s: paired(C, m, s)),
                       minus_unweighted=paired(C, np.ones(U, bool), s), minus_recall=paired(Y, confirmed, s))
        out[s] = row
    return out


def _bar_bootstrap(rows: Mapping, shares, reference: int, *, boot: int, seed: int) -> dict:
    """Two-way cluster bootstrap (pigeonhole), reported beside method D and gating nothing: per replicate, all sentences
    with replacement and, independently, all raters. It counts the rating noise about three times, which is why the
    gates moved to method D (plan, 03:28 UTC). {system: {quantity: array}}."""
    import numpy as np
    if boot < 1:
        raise ValueError("the bootstrap needs at least one replicate")
    rng = np.random.default_rng(seed)
    U, R = rows["utterances"], len(rows["raters"])
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for _ in range(boot):
        sentence_weight = np.bincount(rng.integers(0, U, U), minlength=U).astype(float)
        rater_weight = np.bincount(rng.integers(0, R, R), minlength=R).astype(float)
        for s, row in _bar_statistics(rows, shares, reference, sentence_weight, rater_weight).items():
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


def _kish(shares, counts) -> float:
    """Kish's effective sample size of a rate mixing strata by ``shares``, with ``counts`` ratings in each."""
    v = 0.
    for share, n in zip(shares, counts):
        if share > 0:
            if not n:
                return 0.
            v += share * share / n
    return 1 / v if v else 0.


def _finite(v):
    return v if v is not None and math.isfinite(v) else None


def _share(hits: Sequence[bool]) -> float | None:
    return round(sum(hits) / len(hits), 4) if hits else None


def _plain_gap(rows: Mapping, s: int, reference: int) -> float:
    """Reference minus system, unweighted, over the sentences both have."""
    import numpy as np
    return float(_bar_statistics(rows, [1., 0., 0.], reference, np.ones(rows["utterances"]),
                                 np.ones(len(rows["raters"])))[s]["minus_unweighted"])


def _bar_core(answers, key, roster, reference, shares, problems, *, boot: int, seed: int) -> dict:
    """One read over ``answers`` and ``roster`` (raises ``BarRefused`` under the coverage gates)."""
    import numpy as np
    coverage = _bar_coverage(answers, key, roster, problems)
    _bar_require_coverage(coverage)
    items = {i["item_id"]: i for i in key["items"]}
    rows = _bar_rows(answers, key)
    ref = rows["systems"].index(reference)
    R = len(rows["raters"])
    replicates = _bar_bootstrap(rows, shares, ref, boot=boot, seed=seed) if boot and R >= 2 else None
    boot95 = (lambda s, name: _interval(replicates[s][name])) if replicates else (lambda s, name: None)   # noqa: E731
    hidden = collections.Counter(items[a["item_id"]]["system"] for a in answers if a["viewable"] != "yes")
    recall_only = [1., 0., 0.]
    strata = rows["stratum"][rows["utterance"]]
    report = {}
    for s, name in enumerate(rows["systems"]):
        mine = rows["system"] == s
        counts = [int((mine & (strata == k)).sum()) for k in range(len(STRATA))]
        comp = _rate_read(rows, rows["correct"], mine, shares)
        kish = _kish(shares, counts)
        wilson_weighted = wilson(comp["rate"] * kish, kish) if kish and math.isfinite(comp["rate"]) else None
        hits = int(rows["correct"][mine].sum())
        n_recall, n_alarm = counts[0], counts[1] + counts[2]
        recall = _rate_read(rows, rows["said"], mine & (strata == 0), recall_only)
        said_yes = int(rows["said"][mine & (strata == 0)].sum())
        alarms = int(rows["said"][mine & (strata != 0)].sum())
        stats = coverage["systems"][name]
        lost = stats["blank_inside"] + stats["dropped"]
        out = dict(
            ratings=int(mine.sum()), confirmed_ratings=counts[0], rejected_ratings=counts[1],
            non_candidate_ratings=counts[2],
            comprehension=dict(
                weighted=dict(rate=_finite(comp["rate"]), n_effective=round(kish, 1), shares=list(shares),
                              wilson95=list(wilson_weighted) if wilson_weighted else None,
                              lower_4pct=lower_bound(comp["rate"], comp["variance"], comp["raters"], comp["ratings"]),
                              bootstrap95=boot95(s, "weighted")),
                unweighted=dict(rate=hits / int(mine.sum()) if mine.any() else None,
                                wilson95=list(wilson(hits, int(mine.sum()))), bootstrap95=boot95(s, "unweighted"))),
            negation=dict(
                recall=dict(n=n_recall, rate=_finite(recall["rate"]), wilson95=list(wilson(said_yes, n_recall)),
                            lower_4pct=lower_bound(recall["rate"], recall["variance"], recall["raters"], recall["ratings"]),
                            bootstrap95=boot95(s, "recall")),
                false_alarm=dict(n=n_alarm, rate=alarms / n_alarm if n_alarm else None,
                                 wilson95=list(wilson(alarms, n_alarm)), bootstrap95=boot95(s, "false_alarm"))),
            coverage=stats,
            unviewable=dict(rows=stats["designed"], not_viewable=hidden[name], blank_inside=stats["blank_inside"],
                            dropped=stats["dropped"], share=(hidden[name] + lost) / stats["designed"]))
        if s != ref:
            gap = _gap_read(rows, rows["correct"], s, ref, shares)
            recall_gap = _gap_read(rows, rows["said"], s, ref, recall_only)
            upper = upper_bound(gap["difference"], gap["variance"], gap["raters"])
            out["reference_minus_system"] = dict(
                weighted=dict(difference=_finite(gap["difference"]), upper_5pct=upper,
                              bootstrap95=boot95(s, "minus_weighted")),
                unweighted=dict(difference=_finite(_plain_gap(rows, s, ref)), bootstrap95=boot95(s, "minus_unweighted")),
                recall=dict(difference=_finite(recall_gap["difference"]),
                            upper_5pct=upper_bound(recall_gap["difference"], recall_gap["variance"], recall_gap["raters"]),
                            bootstrap95=boot95(s, "minus_recall")))
            # protocol section 2: a generator must not beat the reference without independent replication
            out["beats_reference"] = dict(
                point=bool(gap["difference"] < 0), beyond_chance=upper is not None and upper < 0,
                note="a generator must not beat the reference without independent replication")
        report[name] = out
    return dict(systems=report, coverage=coverage, raters=R)


def _bar_conditions(core: Mapping, reference: str, *, identical, sheet_problems: int) -> bool:
    """The per-condition results of every generator in ``core`` (in place); returns the negation void."""
    base = core["systems"][reference]
    recall = base["negation"]["recall"]
    void = (recall["n"] == 0 or recall["wilson95"][0] < BAR["recall_lower"]
            or recall["lower_4pct"] is None or recall["lower_4pct"] < BAR["recall_lower"])
    for name, out in core["systems"].items():
        if name == reference:
            continue
        comp, gap, negation = out["comprehension"]["weighted"], out["reference_minus_system"], out["negation"]
        out["conditions"] = dict(
            comprehension_wilson_lower=_condition(comp["wilson95"][0] if comp["wilson95"] else None,
                                                  BAR["comprehension_lower"], at_least=True),
            comprehension_cluster_lower=_condition(comp["lower_4pct"], BAR["comprehension_lower"], at_least=True),
            within_10_points_upper=_condition(gap["weighted"]["upper_5pct"], BAR["within_points"], at_least=False),
            negation_recall_wilson_lower=_condition(negation["recall"]["wilson95"][0] if negation["recall"]["n"] else None,
                                                    BAR["recall_lower"], at_least=True, void=void),
            negation_recall_cluster_lower=_condition(negation["recall"]["lower_4pct"], BAR["recall_lower"], at_least=True,
                                                     void=void),
            negation_recall_within_10_points_upper=_condition(gap["recall"]["upper_5pct"], BAR["within_points"],
                                                              at_least=False, void=void),
            negation_false_alarm_wilson_upper=_condition(
                negation["false_alarm"]["wilson95"][1] if negation["false_alarm"]["n"] else None,
                BAR["false_alarm_upper"], at_least=False, void=void),
            unviewable_share=_condition(out["unviewable"]["share"], BAR["unviewable_share"], at_least=False),
            reference_unviewable_share=_condition(base["unviewable"]["share"], BAR["unviewable_share"], at_least=False),
            no_identical_answer_vectors=_condition(len(identical), 0, at_least=False),
            no_unattributed_sheet_problems=_condition(sheet_problems, 0, at_least=False))
    return void


def bar_read(answers: Sequence[Mapping], key: Mapping, *, reference: str, roster: Mapping[str, int],
             problems: Sequence[Mapping] = (), boot: int = 2000, seed: int = 0) -> dict:
    """Per system and per condition, the release-bar terms as declared (protocol section 6; plan, 03:28 UTC). Raises
    ``BarRefused`` when the read cannot be made: another person's sheet, the checker as a rater, or designed-row
    coverage under 95% for a system or a group, the reference included.

    Comprehension is weighted to D20's three strata (confirmed, rejected candidates, non-candidates), unweighted beside
    it; Wilson (at Kish's effective size) and the one-sided 4% logit bound of method D must both reach 70%. Within 10
    points: method D's one-sided 5% upper bound. Negation: recall by Wilson and by method D, its gap, Wilson's false-alarm
    bound, void if the reference's recall misses either bound. Blanks inside a rater's answered range and dropped rows
    count against the 5% rule; the reference has its own gate. Raters who agree far less than the rest are flagged,
    blind to system, and every condition must hold with and without them. Wrist speed, timing and unseen signers are not
    read here: this scorer never says a system may be released. The bootstrap is reported beside method D."""
    if reference not in key["systems"]:
        raise ValueError(f"{reference} is not a system of this study")
    bar = key.get("bar") or {}
    shares = [(bar.get("strata") or {}).get(k) for k in STRATA]
    if any(v is None or not 0 <= v <= 1 for v in shares) or abs(sum(shares) - 1) > 1e-9:
        raise ValueError("the key declares no strata shares (key['bar']['strata'])")
    if any(i["truth"].get("stratum") not in STRATA or i["truth"]["negated"] not in YES_NO for i in key["items"]):
        raise ValueError("the key has a sentence with no negation truth")
    _bar_check_people(answers, key, roster)
    items = {i["item_id"]: i for i in key["items"]}
    core = _bar_core(answers, key, roster, reference, shares, problems, boot=boot, seed=seed)
    identical = _identical_raters(answers, items)
    sheet_problems = core["coverage"]["unattributed_problems"]
    void = _bar_conditions(core, reference, identical=identical, sheet_problems=sheet_problems)
    agreement = _rater_agreement(answers, items)
    flagged = agreement.pop("flagged_keys")
    without = None
    if flagged:
        try:
            kept = [a for a in answers if a["reviewer_sha256"] not in flagged]
            without = _bar_core(kept, key, {r: g for r, g in roster.items() if r not in flagged}, reference, shares,
                                [p for p in problems if p.get("reviewer_sha256") not in flagged], boot=0, seed=seed)
            without_void = _bar_conditions(without, reference, identical=identical, sheet_problems=sheet_problems)
            without["negation_read_void"] = without_void
        except BarRefused as e:
            without = dict(e.details)
    for name, out in core["systems"].items():
        if name == reference:
            continue
        holds = all(c["result"] == "pass" for c in out["conditions"].values())
        if flagged:
            other = (without.get("systems") or {}).get(name)
            out["without_flagged_raters"] = (
                dict(refused=without["refused"]) if other is None else
                dict(comprehension=other["comprehension"]["weighted"]["rate"],
                     conditions={k: v["result"] for k, v in other["conditions"].items()}))
            holds = holds and other is not None and all(c["result"] == "pass" for c in other["conditions"].values())
        out["passes_rater_terms"] = holds
    said = collections.defaultdict(list)
    for a in answers:
        if a["viewable"] == "yes":
            said[a["reviewer_sha256"]].append((items[a["item_id"]]["truth"]["negated"], a["negated"] == "yes"))
    raters = {r[:12]: dict(group=roster[r], ratings=len(v), says_negated_on_negated=_share([y for t, y in v if t == "yes"]),
                           says_negated_on_others=_share([y for t, y in v if t != "yes"]),
                           always_says_negated=bool(v) and all(y for _, y in v),
                           flagged_low_agreement=r in flagged)
               for r, v in sorted(said.items())}
    return dict(terms="docs/HUMAN_EVALUATION_PROTOCOL.md section 6; docs/NEXT_PHASE_PLAN.md 'Rater read terms defined' and "
                      "'interval method and sheet order changed'",
                thresholds=BAR, reference=reference, negation_read_void=void, shares=dict(zip(STRATA, shares)),
                share_population=bar.get("population"), bootstrap=dict(replicates=boot, seed=seed, gates_nothing=True),
                method="method D: two-way cluster variance of influence values; t(R-1); logit for rates",
                raters_read=core["raters"], rosters=len(roster), coverage=core["coverage"], systems=core["systems"],
                raters=raters, rater_quality=agreement, identical_answer_vectors=identical,
                dropped_rows=[dict(p) for p in problems], readings=dict(
                    wilson_of_weighted_rate="Wilson at the weighted rate with Kish's effective sample size",
                    false_alarm_denominator="every sentence not confirmed negated, rejected candidates included",
                    void="the reference's recall under 70% by Wilson or by method D",
                    beats_reference="flagged, not failed: a point estimate below 0, and whether the bound is below 0"),
                not_read_here=list(NOT_READ_HERE), may_release=False,
                may_release_reason="this scorer does not read wrist speed, timing or unseen signers")


def key_sha256(key: Mapping) -> str:
    import json
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


__all__ = ["OPTIONS", "SENTENCE_TYPES", "SHEET_COLUMNS", "grammatical_truth", "latin_square", "distractors",
           "design_study", "write_sheet", "parse_sheet", "merge_answers", "wilson", "score_study", "key_sha256",
           "NEGATION_COLUMNS", "negation_words", "draw_study_sentences", "write_negation_sheet",
           "parse_negation_sheet", "negation_truth", "negation_stratum", "strata_shares", "STAGE1_COLUMNS",
           "STAGE2_COLUMNS", "write_stage1_sheet", "parse_stage1", "write_stage2_sheet", "parse_stage2",
           "read_two_stage", "parse_roster", "BAR", "STRATA", "BarRefused", "twoway_variance", "lower_bound", "upper_bound", "bar_read"]
