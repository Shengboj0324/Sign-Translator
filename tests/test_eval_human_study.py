"""Blinded human study: Latin-square design, sheets, answer validation and scoring."""
import csv
import io

import pytest

from signtranslator.eval_framework.human_study import (OPTIONS, SHEET_COLUMNS, design_study, distractors,
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
