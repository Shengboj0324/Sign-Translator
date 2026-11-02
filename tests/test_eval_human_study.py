"""Blinded human study: Latin-square design, sheets, answer validation and scoring."""
import csv
import hashlib
import io

import numpy as np
import pytest

from signtranslator.eval_framework.human_study import (OPTIONS, SHEET_COLUMNS, STAGE1_COLUMNS, STAGE2_COLUMNS, BarRefused,
                                                       bar_read, design_study, distractors, grammatical_truth,
                                                       latin_square, merge_answers, negation_stratum, parse_sheet,
                                                       parse_stage1, read_two_stage, score_study, strata_shares, wilson,
                                                       write_sheet, write_stage1_sheet, write_stage2_sheet)

Z = [0] * 7


def utterances(n=8):
    words = ["a", "b c", "d e f", "g h i j", "k l", "m n o", "p", "q r s t"]
    marks = [[[0, 0, 1, 0, 0, 0, 0]], [[0, 1, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 1, 0, 0]], [[1, 0, 0, 0, 0, 0, 0]]]
    return [dict(sign=f"s{i}", english=f"sentence {i} " + words[i % 8], markers=marks[i % 4] + [Z]) for i in range(n)]


SYSTEMS = ["reference", "generator", "baseline"]


def test_truth_follows_the_plan_markers():
    assert grammatical_truth([[0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]])["sentence_type"] == "yes_no_question"
    assert grammatical_truth([[0, 1, 0, 0, 0, 0, 0]])["sentence_type"] == "wh_question"
    t = grammatical_truth([[1, 0, 0, 0, 1, 0, 0]])
    assert t == dict(sentence_type="statement", negated="yes", topic="yes")


def test_latin_square_shows_each_utterance_once_per_group_and_every_cell_once():
    square = latin_square(7, 3)
    cells = {(u, s) for row in square for u, s in enumerate(row)}
    assert len(cells) == 21
    for row in square:
        assert len(row) == 7 and set(row) == {0, 1, 2}
    with pytest.raises(ValueError):
        latin_square(2, 3)


def test_distractors_are_distinct_other_sentences_close_in_length():
    import random
    english = ["one two", "three four", "five", "six seven eight nine ten", "eleven twelve", "one two"]
    chosen = distractors(english, 0, random.Random(0))
    assert 0 not in chosen and 5 not in chosen and len(set(english[j] for j in chosen)) == 3
    assert 3 not in chosen                                          # the long one is the worst length match


def test_design_is_blind_balanced_and_reproducible():
    key = design_study(utterances(), SYSTEMS, seed=7)
    assert key == design_study(utterances(), SYSTEMS, seed=7)
    assert len(key["items"]) == 24 and len({i["item_id"] for i in key["items"]}) == 24
    for g in range(3):
        mine = [i for i in key["items"] if i["group"] == g]
        assert sorted(i["utterance"] for i in mine) == list(range(8))
    for u in range(8):
        same = [i for i in key["items"] if i["utterance"] == u]
        assert {i["system"] for i in same} == set(SYSTEMS)
        assert len({tuple(i["options"]) for i in same}) == 1         # systems differ only in the video
        assert same[0]["options"][OPTIONS.index(same[0]["correct"])] == same[0]["english"]
    sheet = write_sheet(key, 0)
    assert sheet.splitlines()[0] == ",".join(SHEET_COLUMNS)
    for word in SYSTEMS + ["correct", "s0"]:
        assert word not in sheet                                    # nothing in the sheet identifies a system


def test_distractors_come_from_the_whole_partition():
    pool = [f"other sentence number {k}" for k in range(40)]
    key = design_study(utterances(), SYSTEMS, seed=2, distractor_pool=pool)
    offered = {t for i in key["items"] for t in i["options"]}
    assert offered & set(pool)
    for i in key["items"]:
        assert i["options"].count(i["english"]) == 1 and i["options"][OPTIONS.index(i["correct"])] == i["english"]


def fill(sheet, key, reviewer, *, right=True, natural=4):
    items = {i["item_id"]: i for i in key["items"]}
    rows = list(csv.DictReader(io.StringIO(sheet)))
    for r in rows:
        item = items[r["item_id"]]
        wrong = next(o for o in OPTIONS if o != item["correct"])
        r.update(viewable="yes", translation=item["english"], choice=item["correct"] if right else wrong,
                 sentence_type=item["truth"]["sentence_type"], negated=item["truth"]["negated"],
                 topic=item["truth"]["topic"], naturalness=str(natural), reviewer_sha256=reviewer)
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=SHEET_COLUMNS, lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    return out.getvalue()


def test_sheets_are_validated():
    key = design_study(utterances(), SYSTEMS, seed=1)
    sheet = write_sheet(key, 0)
    assert parse_sheet(sheet, key) == ([], [])                      # blank rows are unreviewed
    good = fill(sheet, key, "a" * 64)
    answers, problems = parse_sheet(good, key)
    assert len(answers) == 8 and problems == []
    named = fill(sheet, key, "Jane Doe")
    assert parse_sheet(named, key)[0] == [] and parse_sheet(named, key)[1]
    tampered = good.replace("sentence 3", "sentence 9", 1)
    assert any(p["issue"] == "offered columns were changed" for p in parse_sheet(tampered, key)[1])
    two = good.replace("a" * 64, "b" * 64, 1)
    assert parse_sheet(two, key) == ([], [dict(row=0, issue="one sheet must carry one reviewer key")])


def test_a_primed_reviewer_is_excluded():
    key = design_study(utterances(), SYSTEMS, seed=1)
    s0 = parse_sheet(fill(write_sheet(key, 0), key, "a" * 64), key)[0]
    s1 = parse_sheet(fill(write_sheet(key, 1), key, "a" * 64), key)[0]
    kept, problems = merge_answers([s0, s1], key)
    assert kept == [] and problems


def test_scores_separate_a_system_that_conveys_meaning_from_one_that_does_not():
    key = design_study(utterances(9), SYSTEMS, seed=3)
    sheets = []
    for g, reviewers in enumerate([("1" * 64, "2" * 64), ("3" * 64, "4" * 64), ("5" * 64, "6" * 64)]):
        for r in reviewers:
            answers = parse_sheet(fill(write_sheet(key, g), key, r), key)[0]
            items = {i["item_id"]: i for i in key["items"]}
            for a in answers:                                       # the baseline conveys nothing
                if items[a["item_id"]]["system"] == "baseline":
                    a["choice"] = next(o for o in OPTIONS if o != items[a["item_id"]]["correct"])
                    a["naturalness"] = 1
            sheets.append(answers)
    answers, problems = merge_answers(sheets, key)
    assert problems == []
    s = score_study(answers, key, reference="reference")
    assert s["systems"]["reference"]["comprehension_choice"]["rate"] == 1
    assert s["systems"]["reference"]["comprehension_choice"]["above_chance"]
    assert s["systems"]["baseline"]["comprehension_choice"]["rate"] == 0
    assert s["paired_with_reference"]["baseline"]["mean_difference"] == -1
    assert s["paired_with_reference"]["generator"]["mean_difference"] == 0
    assert s["agreement"]["choice"]["raters"] == 2 and s["agreement"]["choice"]["fleiss_kappa"] == 1
    assert s["systems"]["reference"]["sentence_type"]["rate"] == 1


def test_wilson_interval():
    lo, hi = wilson(5, 10)
    assert lo < .5 < hi and abs((lo + hi) / 2 - .5) < 1e-9
    assert wilson(0, 0) == (0., 1.)


# ---- How2Sign studies: no markers, the sentence draw, the negation truth ---------------------------------------------

def test_negation_words_are_candidates_from_the_english():
    from signtranslator.eval_framework.human_study import negation_words
    assert negation_words("I don't know, and I never will") == ["don't", "never"]
    assert negation_words("It isn’t NOT here") == ["isn't", "not"]
    assert negation_words("No one came without a coat") == ["no", "without"]
    assert negation_words("cannot, nothing, nobody, none, neither, nor, nowhere, cant") == [
        "cannot", "nothing", "nobody", "none", "neither", "nor", "nowhere", "cant"]
    assert negation_words("noone came, nope, nah, I don`t, I don´t") == ["noone", "nope", "nah", "don't"]
    for clean in ("Cut the onion now", "a note on nobility", "I know the snow", "Notice the nose",
                  "nonetheless, happy"):
        assert negation_words(clean) == [], clean


def test_the_draw_keeps_every_eligible_candidate_among_the_main_set():
    from signtranslator.eval_framework.human_study import draw_study_sentences
    rows = [dict(sign=f"s{k}", negation_words=["not"] if k % 10 == 0 else []) for k in range(100)]
    d = draw_study_sentences(rows, seed=4, main=30, pilot=5)
    assert d == draw_study_sentences(rows, seed=4, main=30, pilot=5)
    assert d != draw_study_sentences(rows, seed=5, main=30, pilot=5)
    candidates = {r["sign"] for r in rows if r["negation_words"]}
    assert candidates <= set(d["main"]) and not candidates & set(d["pilot"]) and not set(d["main"]) & set(d["pilot"])
    assert len(d["main"]) == 30 and len(d["pilot"]) == 5
    assert d["main"] == [r["sign"] for r in rows if r["sign"] in set(d["main"])]         # the eligible rows' order
    with pytest.raises(ValueError, match="do not fit"):
        draw_study_sentences(rows, seed=4, main=9, pilot=5)
    with pytest.raises(ValueError, match="cannot give"):
        draw_study_sentences(rows, seed=4, main=90, pilot=11)


def checker_sheet(candidates, answers, **change):
    from signtranslator.eval_framework.human_study import NEGATION_COLUMNS, write_negation_sheet
    rows = list(csv.DictReader(io.StringIO(write_negation_sheet(candidates))))
    for r in rows:
        r.update({**dict(only_real_recordings_seen="yes", watched_before_reading="yes", confirmed=answers.get(r["sign"], ""), checker_sha256="c" * 64),
                  **change})
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=NEGATION_COLUMNS, lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    return out.getvalue()


def test_only_sentences_the_checker_confirms_count_as_negated():
    from signtranslator.eval_framework.human_study import negation_truth, parse_negation_sheet
    candidates = [dict(sign="a", english="I do not", negation_words=["not"]),
                  dict(sign="b", english="never mind", negation_words=["never"])]
    truth, problems = parse_negation_sheet(checker_sheet(candidates, dict(a="yes", b="NO")), candidates)
    assert truth == dict(a="yes", b="no") and problems == []
    assert negation_truth("a", truth) == "yes"
    assert negation_truth("b", truth) == "no"          # the English has one; the recording does not sign it
    assert negation_truth("z", truth) == "no"          # no negation in its English: never negated
    assert parse_negation_sheet(checker_sheet(candidates, dict(a="yes")), candidates)[1]     # b is unanswered
    assert parse_negation_sheet(checker_sheet(candidates, dict(a="yes", b="maybe")), candidates)[1]
    named = checker_sheet(candidates, dict(a="yes", b="no"), checker_sha256="Jane Doe")
    assert any("never a name" in p["issue"] for p in parse_negation_sheet(named, candidates)[1])
    shown = checker_sheet(candidates, dict(a="yes", b="no"), only_real_recordings_seen="no")
    assert any("computer-made clip" in p["issue"] for p in parse_negation_sheet(shown, candidates)[1])
    assert parse_negation_sheet(checker_sheet(candidates, dict(a="yes", b="no"), only_real_recordings_seen=""), candidates)[1]     # blank is not yes
    from signtranslator.eval_framework.human_study import negation_read_first
    read_first = checker_sheet(candidates, dict(a="yes", b="no"), watched_before_reading="no")     # kept, and disclosed
    assert parse_negation_sheet(read_first, candidates) == (dict(a="yes", b="no"), []) and negation_read_first(read_first) == ["a", "b"]
    assert negation_read_first(checker_sheet(candidates, dict(a="yes", b="no"))) == []
    unanswered = checker_sheet(candidates, dict(a="yes", b="no"), watched_before_reading="")
    assert any("before reading" in p["issue"] for p in parse_negation_sheet(unanswered, candidates)[1])
    edited = checker_sheet(candidates, dict(a="yes", b="no")).replace("I do not", "I do", 1)
    assert any(p["issue"] == "offered columns were changed" for p in parse_negation_sheet(edited, candidates)[1])
    extra = checker_sheet(candidates, dict(a="yes", b="no")) + "z,cut it,,yes,yes,yes," + "c" * 64 + ",\n"
    assert any("not a negation candidate" in p["issue"] for p in parse_negation_sheet(extra, candidates)[1])
    assert parse_negation_sheet("sign,confirmed\n", candidates)[0] == {}


def test_a_study_without_markers_has_no_keyerror_and_no_truth_for_unmarked_questions():
    plain = [dict(sign=f"s{k}", english=f"plain sentence number {'x ' * k}{k}",
                  truth=dict(negated="yes" if k < 2 else "no")) for k in range(8)]
    key = design_study(plain, SYSTEMS, seed=3)
    for i in key["items"]:
        assert i["truth"]["sentence_type"] is None and i["truth"]["topic"] is None
        assert i["truth"]["negated"] == ("yes" if i["sign"] in ("s0", "s1") else "no")
    assert design_study([dict(sign="a", english="one two"), dict(sign="b", english="three four"),
                         dict(sign="c", english="five six"), dict(sign="d", english="seven eight")],
                        ["x"], seed=1)["items"][0]["truth"]["negated"] is None
    answers = []
    for g, reviewer in enumerate(("1" * 64, "2" * 64, "3" * 64)):
        answers += parse_sheet(fill_plain(write_sheet(key, g), key, reviewer), key)[0]
    merged, problems = merge_answers([answers], key)
    s = score_study(merged, key, reference="reference")
    row = s["systems"]["generator"]
    assert row["sentence_type"]["n"] == 0 and row["sentence_type"]["rate"] is None and row["topic_recall"]["n"] == 0
    assert row["negation_recall"]["n"] > 0 and row["comprehension_choice"]["rate"] == 1
    # with markers the plan is still the truth, as before
    assert design_study(utterances(), SYSTEMS, seed=7)["items"][0]["truth"]["sentence_type"] in SENTENCE_TYPE_NAMES


def fill_plain(sheet, key, reviewer):
    items = {i["item_id"]: i for i in key["items"]}
    rows = list(csv.DictReader(io.StringIO(sheet)))
    for r in rows:
        item = items[r["item_id"]]
        r.update(viewable="yes", translation=item["english"], choice=item["correct"], sentence_type="statement",
                 negated=item["truth"]["negated"], topic="no", naturalness="3", reviewer_sha256=reviewer)
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=SHEET_COLUMNS, lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    return out.getvalue()


SENTENCE_TYPE_NAMES = ("statement", "yes_no_question", "wh_question", "rhetorical_question")


def test_a_text_key_keeps_near_identical_sentences_out_of_the_options():
    import random
    english = ["Cut it now.", "cut it, now", "CUT  IT NOW", "other one here", "a third one", "the last one"]
    norm = lambda t: " ".join("".join(c if c.isalnum() else " " for c in t.lower()).split())   # noqa: E731
    chosen = distractors(english, 0, random.Random(0), text_key=norm)
    assert not {1, 2} & set(chosen)
    assert set(chosen) == {3, 4, 5}
    # the default draws as it always has (same random numbers, same comparison)
    assert distractors(english, 0, random.Random(0)) == distractors(english, 0, random.Random(0),
                                                                    text_key=lambda t: t.strip().lower())


# ---- the release-bar terms, on synthetic sheets -----------------------------------------------------------------------

def bar_key(sentences, confirmed, rejected=0, systems=("reference", "gen_a", "gen_b"), shares=None, seed=9):
    def stratum(k):
        return "confirmed" if k < confirmed else "rejected" if k < confirmed + rejected else "non_candidate"
    plain = [dict(sign=f"s{k:04d}", english=(f"plain sentence {k} " + "word " * (k % 7)).strip(),
                  truth=dict(negated="yes" if k < confirmed else "no", stratum=stratum(k))) for k in range(sentences)]
    key = design_study(plain, list(systems), seed=seed)
    a, b = shares or (.2, 0.)
    key["bar"] = dict(strata=dict(confirmed=a, rejected=b, non_candidate=1 - a - b))
    key["stages"] = 2
    return key


def rater_of(g, k):
    return f"{g}{k}".ljust(64, "a")


def roster_of(key, per_group):
    return {rater_of(g, k): g for g in range(key["groups"]) for k in range(per_group)}


def bar_sheets(key, per_group, *, choice, says=None, viewable=None, blank=None, seed=0, copy_of=None):
    """Filled sheets, through the real sheet round trip. ``choice(item, rater, rng) -> True when right``; ``says(item,
    rater, rng) -> the negated answer`` (default: the truth); ``viewable(item, rater) -> False`` for ``viewable=no`` (no
    key on the row); ``blank(item, rater) -> True`` for a row left empty. Translations differ by rater."""
    rng = np.random.default_rng(seed)
    sheets = []
    items = {i["item_id"]: i for i in key["items"]}
    for g in range(key["groups"]):
        for k in range(per_group):
            rater = rater_of(g, k)
            rows = list(csv.DictReader(io.StringIO(write_sheet(key, g))))
            for r in rows:
                item = items[r["item_id"]]
                if blank is not None and blank(item, rater):
                    continue
                if viewable is not None and not viewable(item, rater):
                    r.update(viewable="no")
                    continue
                right = choice(item, rater, rng)
                wrong = next(o for o in OPTIONS if o != item["correct"])
                said = says(item, rater, rng) if says else item["truth"]["negated"]
                r.update(viewable="yes", translation=f"{rater[:2]} {item['position']}",
                         choice=item["correct"] if right else wrong,
                         sentence_type="statement", negated=said, topic="no", naturalness="3", reviewer_sha256=rater)
            out = io.StringIO()
            w = csv.DictWriter(out, fieldnames=SHEET_COLUMNS, lineterminator="\n")
            w.writeheader(); w.writerows(rows)
            answers, problems = parse_sheet(out.getvalue(), key)
            assert not problems
            sheets.append(answers)
    merged, problems = merge_answers(sheets, key)
    assert not problems
    return merged


def read_of(key, per_group, *, boot=0, seed=0, **sheet):
    return bar_read(bar_sheets(key, per_group, seed=seed, **sheet), key, reference="reference",
                    roster=roster_of(key, per_group), boot=boot, seed=seed)


def rate(by_system):
    """A choice function: each system right on a fixed fraction of the sentences (the same rows for every rater)."""
    return lambda item, rater, rng: (item["utterance"] % 100) < by_system[item["system"]] * 100


GOOD = dict(reference=.9, gen_a=.9, gen_b=.9)


def results(read, system="gen_a"):
    return {k: v["result"] for k, v in read["systems"][system]["conditions"].items()}


def test_a_system_as_good_as_the_reference_passes_every_condition_with_no_release():
    key = bar_key(100, 30)
    read = read_of(key, 3, choice=rate(GOOD), boot=100)
    assert set(results(read).values()) == {"pass"} and read["systems"]["gen_a"]["passes_rater_terms"]
    assert "conditions" not in read["systems"]["reference"]
    assert read["may_release"] is False and "wrist_speed_0.8_1.25" in read["not_read_here"]
    assert read["negation_read_void"] is False and read["coverage"]["systems"]["gen_a"]["share"] == 1
    assert read["systems"]["gen_a"]["comprehension"]["weighted"]["bootstrap95"] is not None   # beside method D


def test_a_low_system_fails_comprehension_and_the_gap():
    key = bar_key(100, 30)
    read = read_of(key, 3, choice=rate(dict(reference=.95, gen_a=.6, gen_b=.95)))
    r = results(read)
    assert r["comprehension_wilson_lower"] == r["comprehension_cluster_lower"] == "fail"
    assert r["within_10_points_upper"] == "fail"
    assert r["negation_recall_wilson_lower"] == "pass" and not read["systems"]["gen_a"]["passes_rater_terms"]
    assert read["systems"]["gen_b"]["passes_rater_terms"]            # one failing system does not sink another


def test_comprehension_is_weighted_by_three_strata_with_the_unweighted_beside_it():
    key = bar_key(100, 30, rejected=10, shares=(.1, .05))
    # confirmed and rejected sentences never understood, the others always: unweighted 0.6; weighted 0.85. Merging the
    # rejected candidates into the sampled stratum would read 0.1*0 + 0.9*(60/70) = 0.771
    read = read_of(key, 2, choice=lambda i, r, g: i["truth"]["stratum"] == "non_candidate", boot=50)
    row = read["systems"]["gen_a"]
    c = row["comprehension"]
    assert c["unweighted"]["rate"] == .6 and abs(c["weighted"]["rate"] - .85) < 1e-12
    assert read["shares"] == dict(confirmed=.1, rejected=.05, non_candidate=.85)
    kish = 1 / (.01 / row["confirmed_ratings"] + .0025 / row["rejected_ratings"] + .85 ** 2 / row["non_candidate_ratings"])
    assert c["weighted"]["n_effective"] == round(kish, 1)
    assert c["weighted"]["wilson95"] == list(wilson(c["weighted"]["rate"] * kish, kish))
    assert c["unweighted"]["wilson95"] == list(wilson(int(.6 * row["ratings"]), row["ratings"]))
    assert results(read)["comprehension_wilson_lower"] == "pass"     # the unweighted 0.6 alone would have failed


def test_method_d_by_hand():
    from scipy.stats import t
    from signtranslator.eval_framework.human_study import lower_bound, twoway_variance, upper_bound
    psi = np.array([.1, -.2, .3, -.1, .2, -.3])
    u = np.array([0, 0, 1, 1, 2, 2])
    r = np.array([0, 1, 0, 1, 0, 1])
    # by sentence: -.1, .2, -.1; by rater: .6, -.6; sum psi^2 = .28
    by_u = (.01 + .04 + .01) * 3 / 2
    by_r = (.36 + .36) * 2 / 1
    assert abs(twoway_variance(psi, u, r) - (by_u + by_r - .28)) < 1e-12
    # the rater term is floored at zero: raters that differ less than the noise add nothing
    same = np.array([.1, -.1, .1, -.1, .1, -.1]) * np.array([1, 1, 1, 1, -1, -1])
    r2 = np.array([0, 0, 1, 1, 2, 2])
    u2 = np.array([0, 1, 0, 1, 0, 1])
    sentence = ((.1 + .1 - .1) ** 2 + (-.1 - .1 + .1) ** 2) * 2 / 1
    assert abs(twoway_variance(same, u2, r2) - sentence) < 1e-12
    assert np.isnan(twoway_variance(psi, u, np.zeros(6, int)))       # one rater: no read
    # the bounds: logit scale, one-sided 4%, t(R-1); difference, one-sided 5%, t(R-1)
    tq = float(t.ppf(.96, 8))
    want = 1 / (1 + np.exp(-(np.log(.8 / .2) - tq * .02 / (.8 * .2))))
    assert abs(lower_bound(.8, .0004, 9, 600) - want) < 1e-12
    assert abs(upper_bound(.05, .0004, 9) - (.05 + float(t.ppf(.95, 8)) * .02)) < 1e-12
    assert lower_bound(.8, .0004, 1, 600) is None and upper_bound(.05, float("nan"), 9) is None
    assert .99 < lower_bound(1., 0., 9, 600) < 1                     # clipped, never infinite


def planted(key, *, level, rater_sd, sentence_sd, seed):
    """A logit model: sentence and rater effects, one level, so ratings cluster. ``level``: the system's intercept."""
    rng = np.random.default_rng(seed)
    sentence = rng.normal(0, sentence_sd, key["utterances"])
    raters = {}

    def choice(item, rater, g):
        effect = raters.setdefault(rater, rng.normal(0, rater_sd))
        return g.random() < 1 / (1 + np.exp(-(level[item["system"]] + sentence[item["utterance"]] + effect)))
    return choice


def test_planted_clustering_moves_the_bound_past_wilson():
    """Strong sentence and rater effects: near the bar, Wilson passes where method D does not."""
    key = bar_key(200, 40)
    level = dict(reference=2.6, gen_a=1.8, gen_b=1.8)
    read = read_of(key, 3, choice=planted(key, level=level, rater_sd=1., sentence_sd=1.5, seed=PLANTED_SEED),
                   seed=PLANTED_SEED)
    w = read["systems"]["gen_a"]["comprehension"]["weighted"]
    assert w["wilson95"][0] >= .7 and w["lower_4pct"] < .7, w
    r = results(read)
    assert r["comprehension_wilson_lower"] == "pass" and r["comprehension_cluster_lower"] == "fail"
    assert not read["systems"]["gen_a"]["passes_rater_terms"]


PLANTED_SEED = 1


def calibrated(p, sd=np.hypot(1., .5)):
    x, w = np.polynomial.hermite_e.hermegauss(40)
    grid = np.linspace(-3, 6, 3001)
    rates = (w / w.sum() / (1 + np.exp(-(grid[:, None] + sd * x)))).sum(1)
    return float(grid[np.argmin(abs(rates - p))])


def cluster_rows(rng, level, raters=9, U=200, confirmed=70, rejected=10, sentence_sd=1., rater_sd=.5):
    """The arrays of the plan's simulation: 200 sentences, three groups, sentence SD 1.0, rater SD 0.5."""
    a = np.array(level)
    se, re = rng.normal(0, sentence_sd, U), rng.normal(0, rater_sd, raters)
    u, r = np.tile(np.arange(U), raters), np.repeat(np.arange(raters), U)
    s = (u + r % 3) % 3
    hit = (rng.random(len(u)) < 1 / (1 + np.exp(-(a[s] + se[u] + re[r])))).astype(float)
    strata = np.full(U, 2)
    pick = rng.permutation(U)
    strata[pick[:confirmed]] = 0
    strata[pick[confirmed:confirmed + rejected]] = 1
    return dict(systems=[0, 1, 2], raters=list(range(raters)), utterances=U, stratum=strata, utterance=u, system=s, rater=r,
                correct=hit, said=hit)


SIM_SHARES = [70 / 427, 10 / 427, 347 / 427]


def simulated(rows):
    from signtranslator.eval_framework.human_study import _gap_read, _kish, _rate_read, lower_bound, upper_bound
    mine = rows["system"] == 1
    strata = rows["stratum"][rows["utterance"]]
    comp = _rate_read(rows, rows["correct"], mine, SIM_SHARES)
    kish = _kish(SIM_SHARES, [int((mine & (strata == k)).sum()) for k in range(3)])
    gap = _gap_read(rows, rows["correct"], 1, 0, SIM_SHARES)
    return (wilson(comp["rate"] * kish, kish)[0] >= .7,
            lower_bound(comp["rate"], comp["variance"], comp["raters"], comp["ratings"]) >= .7,
            upper_bound(gap["difference"], gap["variance"], gap["raters"]) <= .1)


@pytest.mark.parametrize("raters", [6, 9, 15])
def test_method_d_holds_the_declared_false_pass_rates(raters):
    """The plan's simulated ranges for R >= 6 (.036-.045 at a true 70%, .024-.055 at a true 10-point gap), with room
    for 400 simulated studies; scratch runs at 1,500 are in the commit's report. Wilson alone, under this clustering,
    passes a true 70% far more often."""
    rng = np.random.default_rng(raters)
    level = [calibrated(.80), calibrated(.70), calibrated(.70)]
    sims = 400
    wilson_alone = both = within = 0
    for _ in range(sims):
        w, d, g = simulated(cluster_rows(rng, level, raters=raters))
        wilson_alone += w
        both += w and d
        within += g
    assert .015 <= both / sims <= .075, both / sims
    assert .015 <= within / sims <= .085, within / sims
    assert wilson_alone > both


def test_method_d_has_the_declared_power_at_nine_raters():
    rng = np.random.default_rng(99)
    sims = 300
    passes = {}
    for name, ref, gen in (("gap5", .85, .80), ("gap0", .80, .80), ("true75", .85, .75)):
        level = [calibrated(ref), calibrated(gen), calibrated(gen)]
        got = [simulated(cluster_rows(rng, level)) for _ in range(sims)]
        passes[name] = (np.mean([a and b for a, b, _ in got]), np.mean([g for _, _, g in got]))
    assert .75 <= passes["gap5"][0] <= .95 and .5 <= passes["gap5"][1] <= .8, passes        # true 80%, about 85%; gap 5, 65%
    assert passes["gap0"][1] >= .95, passes                                                  # gap 0, about 100%
    assert .22 <= passes["true75"][0] <= .45, passes                                         # true 75%, about 33%


def test_within_ten_points_is_the_upper_bound_not_the_point_estimate():
    key = bar_key(200, 40)
    # a true gap of 7 points with independent noise: the point estimate is within 10; its upper bound is not
    level = dict(reference=1.6, gen_a=1.1, gen_b=1.6)
    read = read_of(key, 3, choice=planted(key, level=level, rater_sd=.3, sentence_sd=.5, seed=8), seed=8)
    gap = read["systems"]["gen_a"]["reference_minus_system"]["weighted"]
    assert gap["difference"] <= .10 < gap["upper_5pct"], gap
    assert results(read)["within_10_points_upper"] == "fail"
    assert read["systems"]["gen_a"]["reference_minus_system"]["unweighted"]["difference"] is not None


def test_the_difference_is_paired_by_sentence():
    """Hard sentences shared by both systems are no gap: every system is right on the same sentences, so the difference
    and its variance are zero: sentence effects cancel."""
    key = bar_key(60, 12)
    read = read_of(key, 3, choice=rate(dict(reference=.8, gen_a=.8, gen_b=.8)), boot=50)
    gap = read["systems"]["gen_a"]["reference_minus_system"]
    assert abs(gap["weighted"]["upper_5pct"]) < 1e-12 and abs(gap["recall"]["upper_5pct"]) < 1e-12
    assert gap["weighted"]["bootstrap95"] == [0., 0.]


def test_the_gap_reads_only_sentences_both_systems_have():
    from signtranslator.eval_framework.human_study import _gap_read
    # sentence 2 was rated only by the system, and only wrongly: it is not in the gap
    rows = dict(systems=[0, 1], raters=[0, 1], utterances=3, stratum=np.array([2, 2, 2]),
                utterance=np.array([0, 0, 1, 1, 2]), system=np.array([0, 1, 0, 1, 1]), rater=np.array([0, 1, 1, 0, 0]),
                correct=np.array([1., 1., 0., 1., 0.]), said=np.zeros(5))
    gap = _gap_read(rows, rows["correct"], 1, 0, [0., 0., 1.])
    assert abs(gap["difference"] - (.5 - 1.)) < 1e-12 and gap["raters"] == 2


def says_rates(recall, alarm):
    return lambda item, rater, g: ("yes" if g.random() < (recall if item["truth"]["negated"] == "yes" else alarm)
                                   else "no")


def test_negation_recall_and_false_alarm_rules():
    key = bar_key(150, 50)
    by_system = dict(reference=(.95, .02), gen_a=(.5, .02), gen_b=(.95, .4))

    def says(item, rater, g):
        recall, alarm = by_system[item["system"]]
        return says_rates(recall, alarm)(item, rater, g)
    read = read_of(key, 3, choice=rate(GOOD), says=says)
    a, b = results(read, "gen_a"), results(read, "gen_b")
    assert a["negation_recall_wilson_lower"] == a["negation_recall_cluster_lower"] == "fail"
    assert a["negation_recall_within_10_points_upper"] == "fail" and a["negation_false_alarm_wilson_upper"] == "pass"
    assert b["negation_recall_wilson_lower"] == b["negation_recall_cluster_lower"] == "pass"
    assert b["negation_false_alarm_wilson_upper"] == "fail" and not read["systems"]["gen_b"]["passes_rater_terms"]


def test_recall_needs_both_wilson_and_the_clustered_bound():
    """Raters split between seeing all negation and little of it: the pooled rate clears Wilson, the bound does not."""
    key = bar_key(240, 120)
    good = {rater_of(g, k) for g in range(3) for k in (0, 1)}
    says = lambda i, r, g: says_rates(.98 if r in good or i["system"] == "reference" else .35, .02)(i, r, g)   # noqa: E731
    row = read_of(key, 3, choice=rate(GOOD), says=says)
    r = results(row)
    assert r["negation_recall_wilson_lower"] == "pass" and r["negation_recall_cluster_lower"] == "fail", r
    assert not row["systems"]["gen_a"]["passes_rater_terms"]


def test_an_always_says_negated_rater_cannot_game_recall():
    key = bar_key(150, 50)
    # every rater always says negated: recall 100%, false alarms 100%: the read does not pass
    read = read_of(key, 3, choice=rate(GOOD), says=lambda i, r, g: "yes")
    row = read["systems"]["gen_a"]
    assert row["negation"]["recall"]["rate"] == 1 and row["negation"]["false_alarm"]["rate"] == 1
    assert results(read)["negation_recall_wilson_lower"] == "pass"
    assert results(read)["negation_false_alarm_wilson_upper"] == "fail" and not row["passes_rater_terms"]
    assert all(v["always_says_negated"] for v in read["raters"].values())
    # one such rater among nine, the others honest but missing most negation: recall rises, false alarms expose it,
    # the rater is named, and nothing is silently excluded
    gamer = rater_of(0, 0)
    honest = says_rates(.55, .05)
    says = lambda i, r, g: "yes" if r == gamer else honest(i, r, g)    # noqa: E731
    mixed = read_of(key, 3, choice=rate(GOOD), says=says)
    clean = read_of(key, 3, choice=rate(GOOD), says=lambda i, r, g: honest(i, r, g))
    m, c = mixed["systems"]["gen_a"]["negation"], clean["systems"]["gen_a"]["negation"]
    assert m["recall"]["rate"] > c["recall"]["rate"] and m["false_alarm"]["rate"] > c["false_alarm"]["rate"] + .05
    assert mixed["raters"][gamer[:12]]["always_says_negated"]
    assert sum(v["always_says_negated"] for v in mixed["raters"].values()) == 1
    assert sum(v["ratings"] for v in mixed["raters"].values()) == sum(v["ratings"] for v in clean["raters"].values())


def test_the_read_is_void_when_the_reference_misses_negation():
    key = bar_key(100, 40)
    # raters cannot see negation even in real signing; the generators, oddly, score well: void, not passed
    by_system = dict(reference=.4, gen_a=.95, gen_b=.95)
    read = read_of(key, 3, choice=rate(GOOD), says=lambda i, r, g: says_rates(by_system[i["system"]], .02)(i, r, g))
    assert read["negation_read_void"] is True
    assert read["systems"]["reference"]["negation"]["recall"]["wilson95"][0] < .7
    r = results(read)
    assert r["negation_recall_wilson_lower"] == r["negation_recall_within_10_points_upper"] == "void"
    assert r["negation_recall_cluster_lower"] == r["negation_false_alarm_wilson_upper"] == "void"
    assert r["comprehension_wilson_lower"] == "pass"
    assert not read["systems"]["gen_a"]["passes_rater_terms"]


def test_the_read_is_void_when_the_reference_passes_wilson_but_not_the_bound():
    key = bar_key(240, 120)
    good = {rater_of(g, k) for g in range(3) for k in (0, 1)}
    says = lambda i, r, g: says_rates((.98 if r in good else .35) if i["system"] == "reference" else .98, .02)(i, r, g)   # noqa: E731
    read = read_of(key, 3, choice=rate(GOOD), says=says)
    recall = read["systems"]["reference"]["negation"]["recall"]
    assert recall["wilson95"][0] >= .7 > recall["lower_4pct"], recall
    assert read["negation_read_void"] is True and results(read)["negation_recall_wilson_lower"] == "void"


def test_no_confirmed_sentence_reads_void():
    key = bar_key(100, 0, shares=(0., 0.))
    read = read_of(key, 2, choice=rate(GOOD))
    assert read["negation_read_void"] is True and not read["systems"]["gen_a"]["passes_rater_terms"]
    assert read["systems"]["gen_a"]["comprehension"]["weighted"]["rate"] == .9   # only the other stratum has weight


# ---- missing rows fail closed --------------------------------------------------------------------------------------------

def test_a_system_cannot_pass_on_its_blank_wrong_rows():
    """The review's case: a true 50% system whose wrong rows are almost all left blank."""
    key = bar_key(120, 30)
    wrong_kept = lambda i, r: i["system"] == "gen_a" and (i["utterance"] * 7 + i["group"]) % 100 >= 50 \
        and (i["utterance"] * 13) % 100 >= 5                                                  # noqa: E731
    answers = bar_sheets(key, 3, choice=lambda i, r, g: i["system"] != "gen_a" or (i["utterance"] * 7 + i["group"]) % 100 < 50,
                         blank=wrong_kept)
    with pytest.raises(BarRefused, match="coverage") as e:
        bar_read(answers, key, reference="reference", roster=roster_of(key, 3), boot=0)
    assert "system gen_a" in e.value.details["short"]


def test_a_reference_rated_only_on_its_easiest_sentences_is_refused():
    key = bar_key(120, 30)
    answers = bar_sheets(key, 3, choice=rate(GOOD), blank=lambda i, r: i["system"] == "reference" and i["utterance"] > 65)
    with pytest.raises(BarRefused) as e:
        bar_read(answers, key, reference="reference", roster=roster_of(key, 3), boot=0)
    assert "system reference" in e.value.details["short"]


def test_a_missing_group_is_refused():
    key = bar_key(120, 30)
    items = {i["item_id"]: i for i in key["items"]}
    answers = [a for a in bar_sheets(key, 3, choice=rate(GOOD)) if items[a["item_id"]]["group"] != 2]
    with pytest.raises(BarRefused) as e:
        bar_read(answers, key, reference="reference", roster=roster_of(key, 3), boot=0)
    assert "group 2" in e.value.details["short"]
    # and a group with one rater short of its roster: 1 of 3 sheets lost
    answers = [a for a in bar_sheets(key, 3, choice=rate(GOOD)) if a["reviewer_sha256"] != rater_of(1, 2)]
    with pytest.raises(BarRefused) as e:
        bar_read(answers, key, reference="reference", roster=roster_of(key, 3), boot=0)
    assert "group 1" in e.value.details["short"]


def test_coverage_at_the_edge():
    key = bar_key(120, 30)
    # unreached rows at the end of each sheet: read, and not counted against the 5% rule
    read = read_of(key, 3, choice=rate(GOOD), blank=lambda i, r: i["position"] >= 118)
    assert read["coverage"]["groups"][0]["share"] == 118 / 120
    assert all(read["systems"][s]["unviewable"]["blank_inside"] == 0 for s in key["systems"])
    with pytest.raises(BarRefused):
        read_of(key, 3, choice=rate(GOOD), blank=lambda i, r: i["position"] >= 100)
    # the gate itself: 95% reads, under it does not, per system and per group alike
    from signtranslator.eval_framework.human_study import _bar_require_coverage
    fine = dict(systems=dict(a=dict(share=.95)), groups={0: dict(share=.95)})
    _bar_require_coverage(fine)
    for short in (dict(fine, systems=dict(a=dict(share=.9499))), dict(fine, groups={0: dict(share=.9499)})):
        with pytest.raises(BarRefused):
            _bar_require_coverage(short)


def test_blanks_inside_an_answered_range_count_against_the_five_percent_rule():
    key = bar_key(120, 30)
    gen_a = sorted((i["item_id"] for i in key["items"] if i["system"] == "gen_a" and 0 < i["position"] < 119))
    chosen = set(gen_a[:12])
    blanks = lambda item, rater: item["item_id"] in chosen and rater.startswith(str(item["group"]) + "0")   # noqa: E731
    at = read_of(key, 2, choice=rate(GOOD), blank=blanks)
    assert at["systems"]["gen_a"]["unviewable"] == dict(rows=240, not_viewable=0, blank_inside=12, dropped=0, share=.05)
    assert at["coverage"]["systems"]["gen_a"]["share"] == .95 and results(at)["unviewable_share"] == "pass"
    # one viewable=no more: returned, so coverage holds, but blanks and unviewable rows together are over 5%
    extra = gen_a[12]
    over = read_of(key, 2, choice=rate(GOOD), blank=blanks,
                   viewable=lambda i, r: not (i["item_id"] == extra and r.startswith(str(i["group"]) + "0")))
    assert over["systems"]["gen_a"]["unviewable"]["share"] == 13 / 240 and results(over)["unviewable_share"] == "fail"
    assert not over["systems"]["gen_a"]["passes_rater_terms"]


def test_the_unviewable_rule_is_over_five_percent_of_the_system_rows():
    key = bar_key(120, 30)
    gen_a = sorted(i["item_id"] for i in key["items"] if i["system"] == "gen_a")

    def hide(count):
        chosen = set(gen_a[:count])
        return lambda item, rater: not (item["item_id"] in chosen and rater.startswith(str(item["group"]) + "0"))
    # 2 raters per group: 240 gen_a rows; 12 hidden is exactly 5%, 13 is over. viewable=no carries no key
    at = read_of(key, 2, choice=rate(GOOD), viewable=hide(12))
    over = read_of(key, 2, choice=rate(GOOD), viewable=hide(13))
    assert at["systems"]["gen_a"]["unviewable"]["not_viewable"] == 12 and at["systems"]["gen_a"]["unviewable"]["share"] == .05
    assert results(at)["unviewable_share"] == "pass" and results(over)["unviewable_share"] == "fail"
    assert not over["systems"]["gen_a"]["passes_rater_terms"]
    assert results(over, "gen_b")["unviewable_share"] == "pass"
    assert at["systems"]["gen_a"]["ratings"] == 240 - 12                          # excluded from every score


def test_a_true_thirty_percent_unviewable_is_seen():
    """The review's case: the guide says to leave the rest of an unviewable row empty, the reviewer key included."""
    key = bar_key(120, 30)
    read = read_of(key, 3, choice=rate(GOOD), viewable=lambda i, r: not (i["system"] == "gen_a" and i["utterance"] % 10 < 3))
    assert read["systems"]["gen_a"]["unviewable"]["not_viewable"] == 108
    assert abs(read["systems"]["gen_a"]["unviewable"]["share"] - .3) < .02
    assert results(read)["unviewable_share"] == "fail"


def test_viewable_no_needs_no_key_only_on_a_sheet_with_a_known_key():
    key = bar_key(30, 5)
    rows = list(csv.DictReader(io.StringIO(write_sheet(key, 0))))
    rows[0].update(viewable="no")

    def text(with_key):
        for r in rows[1:]:
            r.update(viewable="yes", translation="x", choice="a", sentence_type="statement", negated="no", topic="no",
                     naturalness="3", reviewer_sha256=with_key)
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=SHEET_COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
        return out.getvalue()
    answers, problems = parse_sheet(text("a" * 64), key)
    assert problems == [] and answers[0]["reviewer_sha256"] == "a" * 64 and answers[0]["viewable"] == "no"
    only = "\n".join(text("").splitlines()[:2]) + "\n"
    answers, problems = parse_sheet(only, key)                       # no key anywhere on the sheet: refused
    assert answers == [] and any("never a name" in p["issue"] for p in problems)


def test_dropped_rows_are_reported_and_counted():
    key = bar_key(120, 30)
    answers = bar_sheets(key, 2, choice=rate(GOOD))
    roster = roster_of(key, 2)
    gen_a = [i for i in key["items"] if i["system"] == "gen_a" and i["group"] == 0][:6]
    r0, r1 = rater_of(0, 0), rater_of(0, 1)
    gone = {i["item_id"] for i in gen_a}
    kept = [a for a in answers if not (a["reviewer_sha256"] in (r0, r1) and a["item_id"] in gone)]
    problems = [dict(row=2, item_id=i, issue="naturalness must be 1-5", reviewer_sha256=r) for i in sorted(gone)
                for r in (r0, r1)]
    fine = bar_read(kept, key, reference="reference", roster=roster, problems=problems, boot=0)
    row = fine["systems"]["gen_a"]["unviewable"]
    assert row["blank_inside"] + row["dropped"] == 12 and len(fine["dropped_rows"]) == 12 and row["share"] == .05
    assert results(fine)["unviewable_share"] == "pass"
    # one more unviewable row: the dropped rows carry it over the 5% rule, and the pass goes
    target = next(a for a in kept if a["reviewer_sha256"] == rater_of(1, 0)
                  and next(i for i in key["items"] if i["item_id"] == a["item_id"])["system"] == "gen_a")
    worse = [dict(a, viewable="no") if a is target else a for a in kept]
    read = bar_read(worse, key, reference="reference", roster=roster, problems=problems, boot=0)
    assert results(read)["unviewable_share"] == "fail" and not read["systems"]["gen_a"]["passes_rater_terms"]
    # dropped rows outside a rater's answered range, or with no rater known, are counted as dropped
    items = {i["item_id"]: i for i in key["items"]}
    mine = sorted((i for i in key["items"] if i["group"] == 0), key=lambda i: i["position"])
    last = mine[-1]
    trimmed = [a for a in answers if not (a["reviewer_sha256"] == r0 and a["item_id"] == last["item_id"])]
    tail = bar_read(trimmed, key, reference="reference", roster=roster, boot=0, problems=[
        dict(row=9, item_id=last["item_id"], issue="naturalness must be 1-5", reviewer_sha256=r0)])
    assert tail["systems"][last["system"]]["unviewable"]["dropped"] == 1
    assert tail["systems"][last["system"]]["unviewable"]["blank_inside"] == 0
    assert tail["systems"][last["system"]]["unviewable"]["share"] == 1 / 240
    anon = bar_read(answers, key, reference="reference", roster=roster, boot=0, problems=[
        dict(row=9, item_id=mine[3]["item_id"], issue="offered columns were changed")])
    assert anon["systems"][items[mine[3]["item_id"]]["system"]]["unviewable"]["dropped"] == 1
    # a sheet-level problem that names no row still blocks the pass
    other = bar_read(answers, key, reference="reference", roster=roster, problems=[dict(row=0, issue="bad columns")], boot=0)
    assert results(other)["no_unattributed_sheet_problems"] == "fail"
    assert not other["systems"]["gen_a"]["passes_rater_terms"]


# ---- who the raters are --------------------------------------------------------------------------------------------------

def test_the_roster_decides_who_counts():
    from signtranslator.eval_framework.human_study import parse_roster
    key = bar_key(30, 5)
    answers = bar_sheets(key, 2, choice=rate(GOOD))
    roster = roster_of(key, 2)
    missing = dict(roster)
    del missing[rater_of(1, 1)]
    with pytest.raises(BarRefused, match="not on the roster"):
        bar_read(answers, key, reference="reference", roster=missing, boot=0)
    moved = dict(roster, **{rater_of(0, 0): 1})
    with pytest.raises(BarRefused, match="another group"):
        bar_read(answers, key, reference="reference", roster=moved, boot=0)
    with pytest.raises(BarRefused, match="group the study does not have"):
        bar_read(answers, key, reference="reference", roster=dict(roster, **{rater_of(0, 0): 7}), boot=0)
    with pytest.raises(BarRefused, match="empty"):
        bar_read(answers, key, reference="reference", roster={}, boot=0)
    assert parse_roster('{"%s": 1}' % ("a" * 64)) == {"a" * 64: 1}
    for bad in ('{"Jane": 1}', '{"%s": "1"}' % ("a" * 64), "[1]"):
        with pytest.raises(ValueError):
            parse_roster(bad)


def test_the_checker_cannot_be_a_rater():
    key = bar_key(30, 5)
    answers = bar_sheets(key, 2, choice=rate(GOOD))
    roster = roster_of(key, 2)
    with pytest.raises(BarRefused, match="checker"):
        bar_read(answers, dict(key, negation_checkers=[rater_of(1, 0)]), reference="reference", roster=roster, boot=0)
    assert bar_read(answers, dict(key, negation_checkers=["f" * 64]), reference="reference", roster=roster, boot=0)


def test_identical_answer_vectors_block_the_pass():
    key = bar_key(60, 12)
    answers = bar_sheets(key, 2, choice=rate(GOOD))
    a_rows = {a["item_id"]: a for a in answers if a["reviewer_sha256"] == rater_of(0, 0)}
    copied = [dict(a_rows[a["item_id"]], reviewer_sha256=rater_of(0, 1)) if a["reviewer_sha256"] == rater_of(0, 1) else a
              for a in answers]
    read = bar_read(copied, key, reference="reference", roster=roster_of(key, 2), boot=0)
    assert read["identical_answer_vectors"] == [[rater_of(0, 0)[:12], rater_of(0, 1)[:12]]]
    assert results(read)["no_identical_answer_vectors"] == "fail"
    assert not read["systems"]["gen_a"]["passes_rater_terms"]
    clean = bar_read(answers, key, reference="reference", roster=roster_of(key, 2), boot=0)
    assert clean["identical_answer_vectors"] == [] and clean["systems"]["gen_a"]["passes_rater_terms"]


def guesses(bad, right=.95, low=.25):
    return lambda i, r, g: g.random() < (low if r == bad else right)


def test_a_rater_far_below_the_rest_is_flagged_blind_to_system_and_the_pass_needs_both_reads():
    key = bar_key(120, 30)
    bad = rater_of(0, 0)
    # the wrong rows pick one fixed option, so honest raters agree; the bad rater also picks it, on 75% of its rows
    read = read_of(key, 3, choice=guesses(bad))
    assert read["rater_quality"]["flagged"] == [bad[:12]] and read["raters"][bad[:12]]["flagged_low_agreement"]
    assert read["systems"]["gen_a"]["without_flagged_raters"]["comprehension"] > (
        read["systems"]["gen_a"]["comprehension"]["weighted"]["rate"])
    # the flag does not read the system: the same rater is flagged whichever system its rows came from
    assert "system" not in read["rater_quality"]
    # no flag among honest raters
    honest = read_of(key, 3, choice=guesses("nobody"))
    assert honest["rater_quality"]["flagged"] == [] and "without_flagged_raters" not in honest["systems"]["gen_a"]
    # 18 hidden rows of 360 is 5%; without the flagged rater, 18 of 320 is over it: the pass needs both reads
    gen_a = sorted(i["item_id"] for i in key["items"] if i["system"] == "gen_a")
    others = [rater_of(g, k) for g in range(3) for k in range(3) if rater_of(g, k) != bad]
    by_group = {g: sorted(i["item_id"] for i in key["items"] if i["system"] == "gen_a" and i["group"] == g) for g in range(3)}
    hidden = set()
    for r in others:
        g = int(r[0])
        hidden |= {(r, x) for x in by_group[g][:2]}
    hidden |= {(rater_of(1, 0), by_group[1][2]), (rater_of(2, 0), by_group[2][2])}
    assert len(hidden) == 18 and len(gen_a) == 120
    both = read_of(key, 3, choice=guesses(bad, right=.98, low=.7), viewable=lambda i, r: (r, i["item_id"]) not in hidden)
    assert both["rater_quality"]["flagged"] == [bad[:12]]
    assert set(results(both).values()) == {"pass"}                   # the read with every rater passes all conditions
    assert both["systems"]["gen_a"]["without_flagged_raters"]["conditions"]["unviewable_share"] == "fail"
    assert not both["systems"]["gen_a"]["passes_rater_terms"]


def test_removing_a_flagged_rater_can_leave_no_read():
    key = bar_key(120, 30)
    bad = rater_of(0, 0)
    read = read_of(key, 1, choice=guesses(bad))
    assert read["rater_quality"]["flagged"] == [bad[:12]]
    assert "refused" in read["systems"]["gen_a"]["without_flagged_raters"]
    assert not read["systems"]["gen_a"]["passes_rater_terms"]
    # the same case directly: two raters per group, every rater of group 0 removed
    from signtranslator.eval_framework.human_study import _bar_core
    answers = bar_sheets(key, 2, choice=rate(GOOD))
    roster = roster_of(key, 2)
    kept = [a for a in answers if a["reviewer_sha256"] not in (rater_of(0, 0), rater_of(0, 1))]
    with pytest.raises(BarRefused):
        _bar_core(kept, key, {r: g for r, g in roster.items() if g != 0}, "reference", [.2, 0., .8], [], boot=0, seed=0)


def test_the_reference_has_its_own_gates_and_a_generator_that_beats_it_is_flagged():
    key = bar_key(120, 30)
    hidden = {i["item_id"] for i in key["items"] if i["system"] == "reference" and i["utterance"] % 10 == 0}
    read = read_of(key, 3, choice=rate(GOOD), viewable=lambda i, r: i["item_id"] not in hidden)
    assert read["systems"]["reference"]["unviewable"]["share"] > .05
    assert results(read)["reference_unviewable_share"] == "fail" and results(read)["unviewable_share"] == "pass"
    assert not read["systems"]["gen_a"]["passes_rater_terms"]
    level = dict(reference=.6, gen_a=2.4, gen_b=.6)
    beat = read_of(key, 3, choice=planted(key, level=level, rater_sd=.1, sentence_sd=.1, seed=2), seed=2)
    flag = beat["systems"]["gen_a"]["beats_reference"]
    assert flag["point"] is True and flag["beyond_chance"] is True and "replication" in flag["note"]
    assert beat["systems"]["gen_b"]["beats_reference"]["beyond_chance"] is False


def test_the_read_refuses_a_key_it_cannot_read():
    key = bar_key(30, 5)
    answers = bar_sheets(key, 2, choice=rate(GOOD))
    roster = roster_of(key, 2)
    with pytest.raises(ValueError, match="not a system"):
        bar_read(answers, key, reference="real", roster=roster)
    with pytest.raises(ValueError, match="strata shares"):
        bar_read(answers, dict(key, bar={}), reference="reference", roster=roster)
    with pytest.raises(ValueError, match="strata shares"):
        bar_read(answers, dict(key, bar=dict(strata=dict(confirmed=.5, rejected=.1, non_candidate=.1))),
                 reference="reference", roster=roster)
    undecided = dict(key, items=[dict(i, truth=dict(i["truth"], negated=None)) for i in key["items"]])
    with pytest.raises(ValueError, match="no negation truth"):
        bar_read(answers, undecided, reference="reference", roster=roster)
    nostratum = dict(key, items=[dict(i, truth=dict(negated=i["truth"]["negated"])) for i in key["items"]])
    with pytest.raises(ValueError, match="no negation truth"):
        bar_read(answers, nostratum, reference="reference", roster=roster)


def test_the_bootstrap_gates_nothing_and_is_fixed_by_its_seed():
    from signtranslator.eval_framework.human_study import _bar_bootstrap
    key = bar_key(60, 12)
    level = dict(reference=1.5, gen_a=1.2, gen_b=1.2)
    choice = planted(key, level=level, rater_sd=.5, sentence_sd=1., seed=4)
    answers = bar_sheets(key, 2, choice=choice, seed=4)
    roster = roster_of(key, 2)
    with_boot = bar_read(answers, key, reference="reference", roster=roster, boot=60)
    without = bar_read(answers, key, reference="reference", roster=roster, boot=0)
    assert results(with_boot) == results(without)
    assert with_boot["systems"]["gen_a"]["comprehension"]["weighted"]["bootstrap95"] is not None
    assert without["systems"]["gen_a"]["comprehension"]["weighted"]["bootstrap95"] is None
    rows = cluster_rows(np.random.default_rng(1), [1., 1., 1.], U=60, confirmed=10, rejected=0, raters=6)
    a, b = (_bar_bootstrap(rows, SIM_SHARES, 0, boot=30, seed=7) for _ in range(2))
    c = _bar_bootstrap(rows, SIM_SHARES, 0, boot=30, seed=8)
    np.testing.assert_array_equal(a[1]["weighted"], b[1]["weighted"])
    assert not np.array_equal(a[1]["weighted"], c[1]["weighted"])
    with pytest.raises(ValueError, match="at least one"):
        _bar_bootstrap(rows, SIM_SHARES, 0, boot=0, seed=7)


def test_a_gate_reads_the_unrounded_value():
    from signtranslator.eval_framework.human_study import _condition
    assert _condition(.69996, .7, at_least=True)["result"] == "fail"
    assert _condition(.7, .7, at_least=True)["result"] == "pass"
    assert _condition(.10004, .1, at_least=False)["result"] == "fail"
    assert _condition(None, .1, at_least=False)["result"] == "not_read"
    assert _condition(.9, .7, at_least=True, void=True)["result"] == "void"


# ---- the two-stage sheet ---------------------------------------------------------------------------------------------------

def stage1_text(key, group, reviewer, *, skip=(), unviewable=(), negated="no"):
    rows = list(csv.DictReader(io.StringIO(write_stage1_sheet(key, group))))
    for r in rows:
        if r["item_id"] in skip:
            continue
        if r["item_id"] in unviewable:
            r.update(viewable="no")
            continue
        r.update(viewable="yes", translation=f"{reviewer[:2]} says {r['item_id'][:4]}", sentence_type="statement",
                 negated=negated, topic="no", naturalness="3", reviewer_sha256=reviewer)
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=STAGE1_COLUMNS, lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    return out.getvalue().encode()


def stage2_text(key, stage1_bytes, *, choice="a", edit=None):
    answers = parse_stage1(stage1_bytes.decode(), key)[0]
    text = write_stage2_sheet(key, answers, hashlib.sha256(stage1_bytes).hexdigest())
    rows = list(csv.DictReader(io.StringIO(text)))
    for r in rows:
        r["choice"] = choice
        if edit:
            edit(r)
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=STAGE2_COLUMNS, lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    return out.getvalue().encode()


def test_stage_one_shows_no_options_and_stage_two_shows_them_with_the_hash():
    key = bar_key(12, 3)
    sheet = write_stage1_sheet(key, 0)
    assert sheet.splitlines()[0] == ",".join(STAGE1_COLUMNS)
    assert "option" not in sheet and "plain sentence" not in sheet
    for word in key["systems"] + ["correct", "choice", "s0"]:
        assert word not in sheet.replace("sentence_type", "")
    first = stage1_text(key, 0, "a" * 64)
    second = stage2_text(key, first).decode()
    assert second.splitlines()[0] == ",".join(STAGE2_COLUMNS)
    assert "plain sentence" in second and hashlib.sha256(first).hexdigest() in second and "a" * 64 in second


def test_two_stages_join_on_a_matching_hash():
    key = bar_key(12, 3)
    r0, r1 = "a" * 64, "b" * 64
    first = stage1_text(key, 0, r0)
    other = stage1_text(key, 1, r1)
    receipts = {r0: hashlib.sha256(first).hexdigest(), r1: hashlib.sha256(other).hexdigest()}
    answers, problems = read_two_stage([first, other], [stage2_text(key, first), stage2_text(key, other, choice="b")], key,
                                       receipts)
    assert problems == [] and len(answers) == 24
    assert {a["choice"] for a in answers if a["reviewer_sha256"] == r0} == {"a"}
    assert {a["choice"] for a in answers if a["reviewer_sha256"] == r1} == {"b"}
    assert all(a["translation"].startswith(a["reviewer_sha256"][:2]) for a in answers)


def test_a_stage_two_answer_without_a_matching_stage_one_hash_is_refused():
    key = bar_key(12, 3)
    r0 = "a" * 64
    first = stage1_text(key, 0, r0)
    receipts = {r0: hashlib.sha256(first).hexdigest()}
    good = stage2_text(key, first)
    # another hash on the rows
    bad = stage2_text(key, first, edit=lambda r: r.update(stage1_sha256="0" * 64))
    answers, problems = read_two_stage([first], [bad], key, receipts)
    assert answers == [] and any("matching stage 1 hash" in p["issue"] for p in problems)
    # stage 1 edited after the options were seen: its hash no longer matches stage 2 or the receipt
    edited = first.replace(b",no,no,3,", b",yes,no,3,", 1)
    assert edited != first
    answers, problems = read_two_stage([edited], [good], key, receipts)
    assert answers == [] and any("receipt" in p["issue"] for p in problems)
    # ... and rewriting stage 2's hash to match the edit does not help: the receipt was recorded at release
    forged = stage2_text(key, edited)
    answers, problems = read_two_stage([edited], [forged], key, receipts)
    assert answers == [] and any("receipt" in p["issue"] for p in problems)
    # no stage 1 at all from this key
    answers, problems = read_two_stage([], [good], key, {})
    assert answers == [] and any("no stage 1" in p["issue"] for p in problems)
    # keys must match across stages
    other_key = stage2_text(key, first, edit=lambda r: r.update(reviewer_sha256="b" * 64))
    answers, problems = read_two_stage([first], [other_key], key, receipts)
    assert answers == [] and any("no stage 1" in p["issue"] for p in problems)
    # a tampered option
    moved = stage2_text(key, first, edit=lambda r: r.update(option_a=r["option_b"], option_b=r["option_a"]))
    answers, problems = read_two_stage([first], [moved], key, receipts)
    assert answers == [] and any("offered columns" in p["issue"] for p in problems)


def test_stage_rows_are_viewable_answered_rows_and_blanks_stay_blank():
    key = bar_key(12, 3)
    items = sorted(i["item_id"] for i in key["items"] if i["group"] == 0)
    r0 = "a" * 64
    first = stage1_text(key, 0, r0, skip=items[:1], unviewable=items[1:2])
    answers1, problems = parse_stage1(first.decode(), key)
    assert problems == [] and len(answers1) == 11
    no = next(a for a in answers1 if a["item_id"] == items[1])
    assert no["viewable"] == "no" and no["reviewer_sha256"] == r0     # no key on the row: the sheet's
    second = stage2_text(key, first)
    assert items[0] not in second.decode() and items[1] not in second.decode()     # no options for them
    receipts = {r0: hashlib.sha256(first).hexdigest()}
    answers, problems = read_two_stage([first], [second], key, receipts)
    assert problems == [] and len(answers) == 11 and {a["viewable"] for a in answers} == {"yes", "no"}
    # a viewable row with no stage 2 choice is not an answer
    partial = stage2_text(key, first, edit=lambda r: r.update(choice="" if r["item_id"] == items[2] else "a"))
    answers, _ = read_two_stage([first], [partial], key, receipts)
    assert len(answers) == 10 and items[2] not in {a["item_id"] for a in answers}
    # a choice for a row the rater called unviewable, or left blank, is refused
    from signtranslator.eval_framework.human_study import parse_stage2
    by_id = {i["item_id"]: i for i in key["items"]}
    forged = list(csv.DictReader(io.StringIO(second.decode())))
    for target in items[:2]:
        forged.append(dict(forged[0], item_id=target, video=f"{target}.mp4", choice="a",
                           **{f"option_{o}": t for o, t in zip(OPTIONS, by_id[target]["options"])}))
    buffer = io.StringIO()
    w = csv.DictWriter(buffer, fieldnames=STAGE2_COLUMNS, lineterminator="\n")
    w.writeheader(); w.writerows(forged)
    got, problems = parse_stage2(buffer.getvalue(), key, {r0: dict(
        sha256=receipts[r0], answers={a["item_id"]: a for a in answers1})})
    assert [p["item_id"] for p in problems if "no viewable stage 1 answer" in p["issue"]] == items[:2]
    assert (r0, items[0]) not in got and (r0, items[1]) not in got
    # a second stage 1 from the same key
    answers, problems = read_two_stage([first, first], [second], key, receipts)
    assert any("two stage 1" in p["issue"] for p in problems)
    # one sheet, one key
    mixed = first.replace(r0.encode(), b"c" * 64, 1)
    assert parse_stage1(mixed.decode(), key) == ([], [dict(row=0, issue="one sheet must carry one reviewer key")])
    assert parse_stage1(b"a,b\n".decode(), key)[0] == []


def test_a_problem_row_carries_its_key_and_item():
    key = bar_key(12, 3)
    first = stage1_text(key, 0, "a" * 64).decode().replace(",3,", ",9,", 1)
    answers, problems = parse_stage1(first, key)
    assert len(answers) == 11 and problems[0]["reviewer_sha256"] == "a" * 64 and problems[0]["item_id"]


def test_a_study_read_through_both_stages():
    key = bar_key(30, 10)
    items = {i["item_id"]: i for i in key["items"]}
    first, second, receipts = [], [], {}
    for g in range(3):
        for k in range(2):
            r = rater_of(g, k)
            one = stage1_text(key, g, r)
            receipts[r] = hashlib.sha256(one).hexdigest()
            first.append(one)
            second.append(stage2_text(key, one, edit=lambda row: row.update(choice=items[row["item_id"]]["correct"])))
    answers, problems = read_two_stage(first, second, key, receipts)
    read = bar_read(answers, key, reference="reference", roster=roster_of(key, 2), problems=problems, boot=20)
    assert read["systems"]["gen_a"]["comprehension"]["unweighted"]["rate"] == 1
    assert read["systems"]["gen_a"]["negation"]["recall"]["rate"] == 0      # the sheets said no to everything
    # without stage 2 nothing can be read
    answers, problems = read_two_stage(first, [], key, receipts)
    with pytest.raises(BarRefused, match="coverage"):
        bar_read(answers, key, reference="reference", roster=roster_of(key, 2), problems=problems, boot=0)


def test_item_ids_can_come_from_a_secret_source():
    import secrets
    a = design_study(utterances(), SYSTEMS, seed=7, new_item_id=lambda: secrets.token_hex(16))
    b = design_study(utterances(), SYSTEMS, seed=7, new_item_id=lambda: secrets.token_hex(16))
    assert all(len(i["item_id"]) == 32 for i in a["items"])
    assert {i["item_id"] for i in a["items"]}.isdisjoint(i["item_id"] for i in b["items"])
    assert [i["options"] for i in a["items"]] == [i["options"] for i in b["items"]]


def test_strata_shares_and_stratum():
    assert negation_stratum("a", dict(a="yes", b="no")) == "confirmed"
    assert negation_stratum("b", dict(a="yes", b="no")) == "rejected"
    assert negation_stratum("c", dict(a="yes", b="no")) == "non_candidate"
    assert strata_shares(dict(a="yes", b="yes", c="no"), 10) == dict(confirmed=.2, rejected=.1, non_candidate=.7)
