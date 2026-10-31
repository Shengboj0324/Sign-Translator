"""Blinded human study: Latin-square design, sheets, answer validation and scoring."""
import csv
import io

import numpy as np
import pytest

from signtranslator.eval_framework.human_study import (OPTIONS, SHEET_COLUMNS, bar_read, design_study, distractors,
                                                       grammatical_truth, latin_square, merge_answers, parse_sheet,
                                                       score_study, wilson, write_sheet)

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
        r.update({**dict(no_generated_clip_seen="yes", confirmed=answers.get(r["sign"], ""), checker_sha256="c" * 64),
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
    shown = checker_sheet(candidates, dict(a="yes", b="no"), no_generated_clip_seen="no")
    assert any("generated clip" in p["issue"] for p in parse_negation_sheet(shown, candidates)[1])
    edited = checker_sheet(candidates, dict(a="yes", b="no")).replace("I do not", "I do", 1)
    assert any(p["issue"] == "offered columns were changed" for p in parse_negation_sheet(edited, candidates)[1])
    extra = checker_sheet(candidates, dict(a="yes", b="no")) + "z,cut it,,yes,yes," + "c" * 64 + ",\n"
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

def bar_key(sentences, negated, systems=("reference", "gen_a", "gen_b"), share=.2, seed=9):
    plain = [dict(sign=f"s{k:04d}", english=(f"plain sentence {k} " + "word " * (k % 7)).strip(),
                  truth=dict(negated="yes" if k < negated else "no")) for k in range(sentences)]
    key = design_study(plain, list(systems), seed=seed)
    key["bar"] = dict(negated_share=share)
    return key


def bar_sheets(key, per_group, *, choice, says=None, viewable=None, seed=0):
    """Filled sheets, through the real sheet round trip. ``choice(item, rater) -> True when right``; ``says(item, rater)
    -> the negated answer`` (default: the truth); ``viewable(item, rater) -> False for an unviewable row``."""
    rng = np.random.default_rng(seed)
    sheets = []
    items = {i["item_id"]: i for i in key["items"]}
    for g in range(key["groups"]):
        for k in range(per_group):
            rater = f"{g}{k}".ljust(64, "a")
            rows = list(csv.DictReader(io.StringIO(write_sheet(key, g))))
            for r in rows:
                item = items[r["item_id"]]
                if viewable is not None and not viewable(item, rater):
                    r.update(viewable="no", reviewer_sha256=rater)
                    continue
                right = choice(item, rater, rng)
                wrong = next(o for o in OPTIONS if o != item["correct"])
                said = says(item, rater, rng) if says else item["truth"]["negated"]
                r.update(viewable="yes", translation="x", choice=item["correct"] if right else wrong,
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


def rate(by_system):
    """A choice function: each system right on a fixed fraction of the sentences (the same rows for every rater)."""
    return lambda item, rater, rng: (item["utterance"] % 100) < by_system[item["system"]] * 100


def results(read, system="gen_a"):
    return {k: v["result"] for k, v in read["systems"][system]["conditions"].items()}


def test_a_system_as_good_as_the_reference_passes_every_condition_with_no_release():
    key = bar_key(100, 30)
    read = bar_read(bar_sheets(key, 3, choice=rate(dict(reference=.9, gen_a=.9, gen_b=.9))), key, reference="reference",
                    boot=200)
    assert set(results(read).values()) == {"pass"} and read["systems"]["gen_a"]["passes_rater_terms"]
    assert "conditions" not in read["systems"]["reference"]
    assert read["may_release"] is False and "wrist_speed_0.8_1.25" in read["not_read_here"]
    assert read["negation_read_void"] is False


def test_a_low_system_fails_comprehension_and_the_gap():
    key = bar_key(100, 30)
    read = bar_read(bar_sheets(key, 3, choice=rate(dict(reference=.95, gen_a=.6, gen_b=.95))), key,
                    reference="reference", boot=200)
    r = results(read)
    assert r["comprehension_wilson_lower"] == r["comprehension_clustered_lower"] == "fail"
    assert r["within_10_points_upper"] == "fail"
    assert r["negation_recall_wilson_lower"] == "pass" and not read["systems"]["gen_a"]["passes_rater_terms"]
    assert read["systems"]["gen_b"]["passes_rater_terms"]            # one failing system does not sink another


def test_weighted_comprehension_is_the_stratified_rate_and_the_unweighted_sits_beside_it():
    key = bar_key(100, 40, share=.1)
    # negated sentences never understood; the others always: unweighted 0.6, weighted to a share of 0.1: 0.9
    read = bar_read(bar_sheets(key, 2, choice=lambda i, r, g: i["truth"]["negated"] == "no"), key,
                    reference="reference", boot=100)
    c = read["systems"]["gen_a"]["comprehension"]
    assert c["unweighted"]["rate"] == .6 and c["weighted"]["rate"] == .9 and c["weighted"]["share"] == .1
    # the Kish size: 1 / (0.1^2 / n_negated + 0.9^2 / n_other)
    row = read["systems"]["gen_a"]
    kish = 1 / (.01 / row["negated_ratings"] + .81 / row["other_ratings"])
    assert c["weighted"]["n_effective"] == round(kish, 1)
    assert c["weighted"]["wilson95"] == list(wilson(.9 * kish, kish))
    assert c["unweighted"]["wilson95"] == list(wilson(int(.6 * row["ratings"]), row["ratings"]))
    # the gate reads the weighted rate: the unweighted 0.6 alone would have failed
    assert results(read)["comprehension_wilson_lower"] == "pass"


def planted(key, *, level, rater_sd, sentence_sd, seed):
    """A logit model: sentence and rater effects, one level, so ratings cluster. ``level``: the system's intercept."""
    rng = np.random.default_rng(seed)
    sentence = rng.normal(0, sentence_sd, key["utterances"])
    raters = {}

    def choice(item, rater, g):
        effect = raters.setdefault(rater, rng.normal(0, rater_sd))
        return g.random() < 1 / (1 + np.exp(-(level[item["system"]] + sentence[item["utterance"]] + effect)))
    return choice


def test_planted_clustering_widens_the_bound_past_wilson():
    """Strong sentence and rater effects: the bootstrap interval is far wider than Wilson's, and at a true rate near
    the bar Wilson passes where the clustered bound does not."""
    key = bar_key(200, 40)
    level = dict(reference=2.6, gen_a=1.8, gen_b=1.8)
    read = bar_read(bar_sheets(key, 3, choice=planted(key, level=level, rater_sd=1., sentence_sd=1.5, seed=1), seed=1),
                    key, reference="reference", boot=400)
    w = read["systems"]["gen_a"]["comprehension"]["weighted"]
    assert (w["clustered95"][1] - w["clustered95"][0]) > 1.5 * (w["wilson95"][1] - w["wilson95"][0])
    assert w["wilson95"][0] >= .7 and w["clustered95"][0] < .7, w
    r = results(read)
    assert r["comprehension_wilson_lower"] == "pass" and r["comprehension_clustered_lower"] == "fail"
    assert not read["systems"]["gen_a"]["passes_rater_terms"]


def cluster_rows(rng, level, U=120, per_group=3, negated=30, systems=3, sentence_sd=1.2, rater_sd=.7):
    """The bootstrap's arrays, directly (fast): the design of ``latin_square`` with planted effects."""
    R = per_group * systems
    a = np.array(level)
    se, re = rng.normal(0, sentence_sd, U), rng.normal(0, rater_sd, R)
    u, r = np.tile(np.arange(U), R), np.repeat(np.arange(R), U)
    s = (u + r % systems) % systems
    hit = (rng.random(len(u)) < 1 / (1 + np.exp(-(a[s] + se[u] + re[r])))).astype(float)
    mask = np.zeros(U, bool); mask[rng.permutation(U)[:negated]] = True
    return dict(systems=list(range(systems)), raters=[f"r{k}" for k in range(R)],
                group_of={f"r{k}": k % systems for k in range(R)}, utterances=U, negated=mask,
                utterance=u, system=s, rater=r, correct=hit, said=hit)


def test_the_clustered_bound_holds_its_level_where_wilson_does_not():
    """At a planted true rate equal to the bar, Wilson's lower bound clears it far too often; the bootstrap's seldom."""
    from signtranslator.eval_framework.human_study import _bar_bootstrap, _bar_statistics, _kish
    # the intercept giving a marginal rate of 0.70 under these effects (Monte Carlo, fixed seed)
    draw = np.random.default_rng(0)
    effects = draw.normal(0, np.hypot(1.2, .7), 400000)
    grid = np.linspace(0, 3, 301)
    level = grid[np.argmin([abs((1 / (1 + np.exp(-(g + effects)))).mean() - .7) for g in grid])]
    rng = np.random.default_rng(5)
    wilson_pass = clustered_pass = 0
    sims = 60
    for _ in range(sims):
        rows = cluster_rows(rng, [level] * 3)
        point = _bar_statistics(rows, .2, 0, np.ones(120), np.ones(9))
        mine = rows["system"] == 1
        neg = rows["negated"][rows["utterance"]]
        kish = _kish(.2, int((mine & neg).sum()), int((mine & ~neg).sum()))
        wilson_pass += wilson(point[1]["weighted"] * kish, kish)[0] >= .7
        boot = _bar_bootstrap(rows, .2, 0, boot=150, seed=int(rng.integers(1 << 30)))
        clustered_pass += np.percentile(boot[1]["weighted"], 2.5) >= .7
    assert wilson_pass >= 6                      # the failure the clustered bound exists for
    assert clustered_pass <= 3                   # nominal 2.5%: at most 3 of 60 by chance


def test_within_ten_points_is_the_upper_bound_not_the_point_estimate():
    key = bar_key(200, 40)
    # a true gap of 7 points with independent noise: the point estimate is within 10; its upper bound is not
    level = dict(reference=1.6, gen_a=1.1, gen_b=1.6)
    read = bar_read(bar_sheets(key, 3, choice=planted(key, level=level, rater_sd=.3, sentence_sd=.5, seed=8), seed=8),
                    key, reference="reference", boot=400)
    gap = read["systems"]["gen_a"]["reference_minus_system"]["weighted"]
    assert gap["difference"] <= .10 < gap["clustered95"][1], gap
    assert results(read)["within_10_points_upper"] == "fail"
    assert read["systems"]["gen_a"]["reference_minus_system"]["unweighted"]["clustered95"] is not None


def test_the_difference_is_paired_by_sentence():
    """Hard sentences shared by both systems are no gap: every system is right on the same sentences, so the paired
    difference is zero in every replicate, whatever the sentence draw."""
    key = bar_key(60, 12)
    read = bar_read(bar_sheets(key, 2, choice=rate(dict(reference=.8, gen_a=.8, gen_b=.8))), key, reference="reference",
                    boot=100)
    assert read["systems"]["gen_a"]["reference_minus_system"]["weighted"]["clustered95"] == [0., 0.]
    assert read["systems"]["gen_a"]["reference_minus_system"]["recall"]["clustered95"] == [0., 0.]


def says_rates(recall, alarm):
    return lambda item, rater, g: ("yes" if g.random() < (recall if item["truth"]["negated"] == "yes" else alarm)
                                   else "no")


def test_negation_recall_and_false_alarm_rules():
    key = bar_key(150, 50)
    right = rate(dict(reference=.9, gen_a=.9, gen_b=.9))
    by_system = dict(reference=(.95, .02), gen_a=(.5, .02), gen_b=(.95, .4))

    def says(item, rater, g):
        recall, alarm = by_system[item["system"]]
        return says_rates(recall, alarm)(item, rater, g)
    read = bar_read(bar_sheets(key, 3, choice=right, says=says), key, reference="reference", boot=200)
    a, b = results(read, "gen_a"), results(read, "gen_b")
    assert a["negation_recall_wilson_lower"] == "fail" and a["negation_recall_within_10_points_upper"] == "fail"
    assert a["negation_false_alarm_wilson_upper"] == "pass"
    assert b["negation_recall_wilson_lower"] == "pass" and b["negation_false_alarm_wilson_upper"] == "fail"
    assert not read["systems"]["gen_b"]["passes_rater_terms"]


def test_an_always_says_negated_rater_cannot_game_recall():
    key = bar_key(150, 50)
    right = rate(dict(reference=.9, gen_a=.9, gen_b=.9))
    # every rater always says negated: recall 100%, false alarms 100%: the read does not pass
    read = bar_read(bar_sheets(key, 3, choice=right, says=lambda i, r, g: "yes"), key, reference="reference", boot=100)
    row = read["systems"]["gen_a"]
    assert row["negation"]["recall"]["rate"] == 1 and row["negation"]["false_alarm"]["rate"] == 1
    assert results(read)["negation_recall_wilson_lower"] == "pass"
    assert results(read)["negation_false_alarm_wilson_upper"] == "fail" and not row["passes_rater_terms"]
    assert all(v["always_says_negated"] for v in read["raters"].values())
    # one such rater among nine, the others honest but missing most negation: recall rises, false alarms expose it,
    # the rater is named, and nothing is silently excluded
    gamer = "00".ljust(64, "a")
    honest = says_rates(.55, .05)
    says = lambda i, r, g: "yes" if r == gamer else honest(i, r, g)    # noqa: E731
    mixed = bar_read(bar_sheets(key, 3, choice=right, says=says), key, reference="reference", boot=100)
    clean = bar_read(bar_sheets(key, 3, choice=right, says=lambda i, r, g: honest(i, r, g)), key, reference="reference",
                     boot=100)
    m, c = mixed["systems"]["gen_a"]["negation"], clean["systems"]["gen_a"]["negation"]
    assert m["recall"]["rate"] > c["recall"]["rate"] and m["false_alarm"]["rate"] > c["false_alarm"]["rate"] + .05
    assert mixed["raters"][gamer[:12]]["always_says_negated"]
    assert sum(v["always_says_negated"] for v in mixed["raters"].values()) == 1
    assert sum(v["ratings"] for v in mixed["raters"].values()) == sum(v["ratings"] for v in clean["raters"].values())


def test_the_read_is_void_when_the_reference_misses_negation():
    key = bar_key(100, 40)
    # raters cannot see negation even in real signing; the generators, oddly, score well: void, not passed
    by_system = dict(reference=.4, gen_a=.95, gen_b=.95)
    read = bar_read(bar_sheets(key, 3, choice=rate(dict(reference=.9, gen_a=.9, gen_b=.9)),
                               says=lambda i, r, g: says_rates(by_system[i["system"]], .02)(i, r, g)), key,
                    reference="reference", boot=100)
    assert read["negation_read_void"] is True and read["reference_recall_wilson_lower"] < .7
    r = results(read)
    assert r["negation_recall_wilson_lower"] == r["negation_recall_within_10_points_upper"] == "void"
    assert r["negation_false_alarm_wilson_upper"] == "void"
    assert r["comprehension_wilson_lower"] == "pass"
    assert not read["systems"]["gen_a"]["passes_rater_terms"]


def test_no_confirmed_sentence_reads_void():
    key = bar_key(100, 0, share=0.)
    read = bar_read(bar_sheets(key, 2, choice=rate(dict(reference=.9, gen_a=.9, gen_b=.9))), key, reference="reference",
                    boot=50)
    assert read["negation_read_void"] is True and not read["systems"]["gen_a"]["passes_rater_terms"]
    assert read["systems"]["gen_a"]["comprehension"]["weighted"]["rate"] == .9   # share 0: the other stratum alone


def test_the_unviewable_rule_is_over_five_percent_of_the_system_rows():
    key = bar_key(120, 30)
    right = rate(dict(reference=.95, gen_a=.95, gen_b=.95))
    gen_a = sorted(i["item_id"] for i in key["items"] if i["system"] == "gen_a")      # 120 rows per rater group

    def hide(count):
        chosen = set(gen_a[:count])
        return lambda item, rater: not (item["item_id"] in chosen and rater.startswith(str(item["group"]) + "0"))
    # 2 raters per group: 240 gen_a rows; 12 hidden is exactly 5%, 13 is over
    at = bar_read(bar_sheets(key, 2, choice=right, viewable=hide(12)), key, reference="reference", boot=50)
    over = bar_read(bar_sheets(key, 2, choice=right, viewable=hide(13)), key, reference="reference", boot=50)
    assert at["systems"]["gen_a"]["unviewable"] == dict(rows=240, not_viewable=12, share=.05)
    assert results(at)["unviewable_share"] == "pass" and results(over)["unviewable_share"] == "fail"
    assert not over["systems"]["gen_a"]["passes_rater_terms"]
    assert results(over, "gen_b")["unviewable_share"] == "pass"
    assert at["systems"]["gen_a"]["ratings"] == 240 - 12                          # excluded from every score


def test_too_few_raters_leave_the_clustered_bounds_unread():
    key = bar_key(60, 20)
    right = rate(dict(reference=.95, gen_a=.95, gen_b=.95))
    # one rater per group: a replicate of three often keeps one group, with no paired sentence: the gap is not read
    read = bar_read(bar_sheets(key, 1, choice=right), key, reference="reference", boot=200)
    assert read["clustered_read"] is True and read["raters_per_group"] == {0: 1, 1: 1, 2: 1}
    r = results(read)
    assert r["comprehension_wilson_lower"] == "pass" and r["within_10_points_upper"] == "not_read"
    assert not read["systems"]["gen_a"]["passes_rater_terms"]
    # one rater alone: no rater variance at all
    alone = [a for a in bar_sheets(key, 1, choice=right) if a["reviewer_sha256"].startswith("00")]
    read = bar_read(alone, key, reference="reference", boot=50)
    assert read["clustered_read"] is False and read["bootstrap"]["replicates"] == 0
    assert results(read)["comprehension_clustered_lower"] == "not_read"
    assert bar_read([], key, reference="reference", boot=50)["systems"]["gen_a"]["passes_rater_terms"] is False


def test_the_read_refuses_a_key_it_cannot_read():
    key = bar_key(30, 5)
    answers = bar_sheets(key, 2, choice=rate(dict(reference=.9, gen_a=.9, gen_b=.9)))
    with pytest.raises(ValueError, match="not a system"):
        bar_read(answers, key, reference="real")
    with pytest.raises(ValueError, match="negated share"):
        bar_read(answers, dict(key, bar={}), reference="reference")
    undecided = dict(key, items=[dict(i, truth=dict(i["truth"], negated=None)) for i in key["items"]])
    with pytest.raises(ValueError, match="no negation truth"):
        bar_read(answers, undecided, reference="reference")


def test_the_bootstrap_is_fixed_by_its_seed_and_resamples_both_ways():
    from signtranslator.eval_framework.human_study import _bar_bootstrap
    rng = np.random.default_rng(1)
    rows = cluster_rows(rng, [1., 1., 1.], U=60, negated=10)
    a = _bar_bootstrap(rows, .2, 0, boot=30, seed=7)
    b = _bar_bootstrap(rows, .2, 0, boot=30, seed=7)
    c = _bar_bootstrap(rows, .2, 0, boot=30, seed=8)
    np.testing.assert_array_equal(a[1]["weighted"], b[1]["weighted"])
    assert not np.array_equal(a[1]["weighted"], c[1]["weighted"])
    assert np.isfinite(a[1]["weighted"]).all()
    assert _bar_bootstrap(rows, .2, 0, boot=30, seed=7)[1]["weighted"].std() > 0
    no_raters = dict(rows, rater=np.zeros_like(rows["rater"]), raters=["only"], group_of=dict(only=0))
    no_sentences = dict(rows, utterance=np.zeros_like(rows["utterance"]), utterances=1, negated=np.zeros(1, bool))
    a_sentences = _bar_bootstrap(no_raters, .2, 0, boot=30, seed=7)[1]["weighted"]
    a_raters = _bar_bootstrap(no_sentences, 0., 0, boot=30, seed=7)[1]["weighted"]
    assert a_sentences.std() > 0 and a_raters.std() > 0                  # each way moves the rate alone
    with pytest.raises(ValueError, match="at least one"):
        _bar_bootstrap(rows, .2, 0, boot=0, seed=7)


def test_a_gate_reads_the_unrounded_value():
    from signtranslator.eval_framework.human_study import _condition
    assert _condition(.69996, .7, at_least=True)["result"] == "fail"
    assert _condition(.7, .7, at_least=True)["result"] == "pass"
    assert _condition(.10004, .1, at_least=False)["result"] == "fail"
    assert _condition(None, .1, at_least=False)["result"] == "not_read"
    assert _condition(.9, .7, at_least=True, void=True)["result"] == "void"
