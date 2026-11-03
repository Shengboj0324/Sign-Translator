"""The rater pilot's word-format arms: design balance, blinding, the paired contrasts, the seeded order, the export."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import build_human_eval_packet as bh  # noqa: E402
import rater_word_format as wf  # noqa: E402
import score_human_eval as sh  # noqa: E402
from signtranslator.eval_framework.human_study import (BAR, BarRefused, INSTRUMENT_SHUFFLE_SHARES, NEGATION_COLUMNS,  # noqa: E402
                                                       RULE_MIN_RATERS, SHARE_MIN_RATERS, STAGE1_COLUMNS, STAGE2_COLUMNS, _by_the_rule,
                                                       bar_read, design_study, score_study,
                                                       twoway_variance, word_format_pairing, word_format_read, write_stage1_sheet)
from signtranslator.eval_framework.statistics import paired_permutation_pvalue  # noqa: E402
from tests.test_round15_compose_eval import ce, run_args  # noqa: E402
from tests.test_round15_compositional import stubbed  # noqa: E402,F401
from tests.test_round15_stage2 import stage2_store  # noqa: E402,F401
from tests.test_round20_eval import put_set  # noqa: E402

SYSTEMS = ["composition_shuffled", "oracle", "composition", "reference"]
ROLES = dict(composition="composition", composition_shuffled="composition_shuffled", oracle="oracle", reference="reference")
A, B, C, D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def study(U=20, *, unmoved=(), partial=()):
    utterances = [dict(sign=f"s{u:02d}", english=f"cut the onion number {u} slowly",
                       truth=dict(negated="no", stratum="non_candidate")) for u in range(U)]
    pool = [f"other sentence of the same length {k}" for k in range(8)]
    key = design_study(utterances, SYSTEMS, seed=3, distractor_pool=pool)
    key.update(stages=2, report_only_systems=["composition_shuffled", "oracle"],
               word_format=dict(roles=ROLES, report_only=True, pilot_only=True, dropped_no_real_clip=[]),
               bar=dict(strata=dict(confirmed=0., rejected=0., non_candidate=1.)))
    for i in key["items"]:
        i["state"] = {}
        if i["system"] == "composition_shuffled":
            i["state"]["moved"] = i["utterance"] not in unmoved
        if i["system"] == "oracle":
            i["state"]["same_words"] = i["utterance"] not in partial
    return key


def answer(item, reviewer, *, right=True, natural=3, translation="", viewable="yes"):
    wrong = next(o for o in "abcd" if o != item["correct"])
    return dict(item_id=item["item_id"], reviewer_sha256=reviewer, viewable=viewable, translation=translation,
                choice=item["correct"] if right else wrong, sentence_type="statement", negated="no", topic="no",
                naturalness=natural, notes="")


def sheet(key, groups, **per_system):
    """Answers of one rater per group: right on every sentence unless ``per_system[system]`` says otherwise."""
    raters = dict(zip(groups, (A, B, C, D)))
    out = []
    for i in key["items"]:
        if i["group"] in raters:
            right = per_system.get(i["system"], True)
            out.append(answer(i, raters[i["group"]], right=right(i) if callable(right) else right))
    return out


def eight(key, **per_system):
    """Answers of 2 raters per group, all four groups."""
    keys = {g: (f"{g}a".ljust(64, "a"), f"{g}b".ljust(64, "b")) for g in range(4)}
    out = []
    for i in key["items"]:
        for reviewer in keys[i["group"]]:
            right = per_system.get(i["system"], True)
            out.append(answer(i, reviewer, right=right(i, reviewer) if callable(right) else right))
    return out


# ---- the design -------------------------------------------------------------------------------------------------------------

def test_the_arms_fit_the_latin_square_at_sixty_sentences_and_eight_raters():
    key = study(60)
    assert key["groups"] == 4
    cells = {(i["utterance"], i["system"]) for i in key["items"]}
    assert len(key["items"]) == len(cells) == 240                   # 240 clips to render: every (sentence, system) cell, once
    for g in range(4):
        mine = [i for i in key["items"] if i["group"] == g]
        assert len(mine) == 60                                      # a rater's load is the sentences, whatever the systems
        assert len({i["utterance"] for i in mine}) == 60            # no rater sees a sentence twice
        assert {s: sum(i["system"] == s for i in mine) for s in SYSTEMS} == {s: 15 for s in SYSTEMS}
        assert sorted(i["position"] for i in mine) == list(range(60))
    options = {}
    for i in key["items"]:
        assert options.setdefault(i["utterance"], i["options"]) == i["options"]   # systems differ only in the video


def test_eight_raters_pair_every_two_arms_on_every_sentence_and_give_kappa():
    key = study(60)
    answers = eight(key)
    assert len(answers) == 480
    pairing = word_format_pairing(SYSTEMS, ROLES)
    assert pairing["raters"] == 8 and pairing["groups"] == 4 and pairing["raters_per_group"] == 2 and pairing["clips_per_sentence"] == 4
    assert len(pairing["paired_by_sentence"]) == 6
    read = word_format_read(answers, key)
    assert read["raters"] == 8 and read["ratings"] == 480
    for row in read["contrasts"].values():
        assert row["all"]["comprehension_choice"]["utterances"] == 60 and "within_rater" not in row
        assert row["all"]["rater_balance"]["confounded_with_rater"] is False
        assert len(row["all"]["rater_balance"]["ratings"]) == 8
        assert all(v == [15, 15] for v in row["all"]["rater_balance"]["ratings"].values())
    plain = score_study(answers, key, reference="reference")
    assert plain["agreement"]["choice"]["raters"] == 2 and plain["agreement"]["choice"]["items"] == 240     # kappa is back
    assert plain["systems"]["composition"]["answers"] == 120


def test_the_defaults_are_the_owners_pilot():
    import inspect
    from signtranslator.eval_framework.human_study import draw_study_sentences
    assert inspect.signature(draw_study_sentences).parameters["pilot"].default == 60
    assert inspect.signature(word_format_pairing).parameters["raters_per_group"].default == 2
    protocol = (ROOT / "docs/HUMAN_EVALUATION_PROTOCOL.md").read_text()
    guide = (ROOT / "reviewer_package/OPERATOR_GUIDE.md").read_text()
    assert "60 sentences" in protocol and "8 raters" in protocol and "60 sentences" in guide and "8 raters" in guide
    assert "groups 0 and 2" not in protocol + guide                 # the opposite-groups workaround is gone


def test_only_the_four_systems_are_accepted():
    word_format_pairing(["composition_shuffled", "composition", "oracle", "reference"], ROLES)      # any order: all groups are filled
    with pytest.raises(ValueError, match="exactly the four"):
        word_format_pairing(SYSTEMS + ["arm-b"], dict(ROLES, other="arm-b"))
    with pytest.raises(ValueError, match="exactly the four"):
        word_format_pairing(SYSTEMS[:3], {k: ROLES[k] for k in ("composition", "composition_shuffled", "oracle")})


def test_the_published_power_comes_from_the_declared_rule():
    """The committed script at the plan's numbers, smaller: the rule is under sign-flip, and the two readings of a true drop differ."""
    import rater_pilot_power as power
    assert abs(power.marginal(power.intercept_for(.6)) - .6) < 1e-6
    accuracy = power.simulate(60, 8, .20, "accuracy", sims=400, seed=1)
    logit = power.simulate(60, 8, .20, "logit", sims=400, seed=1)
    assert abs(accuracy["mean_accuracy_drop_points"] - 20) < 2 and abs(logit["mean_accuracy_drop_points"] - 16) < 2
    assert .70 < accuracy["rule_reads_below"] < .88 and .45 < logit["rule_reads_below"] < .65
    assert accuracy["sign_flip_detects"] > accuracy["rule_reads_below"]
    assert power.simulate(60, 8, .20, "accuracy", sims=50, seed=1) == power.simulate(60, 8, .20, "accuracy", sims=50, seed=1)
    assert power.simulate(60, 4, .20, "accuracy", sims=30, seed=1)["rule_reads_below"] == 0     # under the rule's rater minimum
    stored = json.loads((ROOT / "reports/rater_pilot_power.json").read_text())
    row = next(r for r in stored["rows"] if (r["sentences"], r["raters"], r["reading"], r["true_drop_points"]) == (60, 8, "accuracy", 20))
    assert stored["seed"] == 20261102 and stored["sims"] == 4000 and .76 <= row["rule_reads_below"] <= .82


# ---- blinding ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["docs/HUMAN_EVALUATION_REVIEWER_GUIDE_TWO_STAGE.md", "docs/HUMAN_EVALUATION_REVIEWER_GUIDE.md",
                                  "reviewer_package/FOR_REVIEWERS.md", "reviewer_package/RATING_SHEET.md"])
def test_what_raters_read_while_rating_reveals_neither_the_arm_nor_the_question(name):
    """The consent form is not here: whatever the review board requires it to disclose must not fail a test."""
    text = (ROOT / name).read_text().lower()
    for word in ("shuffl", "scrambl", "permut", "word order", "word-order", "english order", "order of the words", "oracle",
                 "composition", "stitch", "word-for-word", "word by word", "real clip", "generated", "control"):
        assert word not in text, word


def test_the_board_and_the_debrief_are_told_and_the_debrief_stays_out_of_the_packets():
    consent = (ROOT / "reviewer_package/CONSENT_FORM.md").read_text()
    note = consent[consent.index("**Note for the owner"):consent.index("**Project:**")]
    assert "scrambled" in note and "remove before printing" in note and "debrief" in note.lower()
    debrief = (ROOT / "reviewer_package/DEBRIEF.md").read_text()
    assert "after all sheets" in debrief.lower() and "out of order" in debrief and "real" in debrief
    guide = (ROOT / "reviewer_package/OPERATOR_GUIDE.md").read_text()
    needs = guide[guide.index("**What you need to do:**"):guide.index("**What changed for raters:**")]
    assert "DEBRIEF.md" in needs and "review board" in needs
    assert "DEBRIEF.md" in (ROOT / "reviewer_package/README.md").read_text()
    assert "debrief" in (ROOT / "docs/HUMAN_EVALUATION_PROTOCOL.md").read_text().lower()


def test_sheets_name_no_arm_and_item_ids_carry_no_order():
    key = study(8)
    for g in range(4):
        text = write_stage1_sheet(key, g)
        for word in (*SYSTEMS, "shuffled", "moved", "same_words", "correct"):
            assert word not in text
    ids = sorted(i["item_id"] for i in key["items"])
    assert len(set(ids)) == len(ids)


# ---- the release bar is out of reach ------------------------------------------------------------------------------------------

def test_the_arms_cannot_reach_the_release_bar():
    key = study(20)
    answers = sheet(key, (0, 2))
    with pytest.raises(BarRefused, match="report-only pilot arms"):
        bar_read(answers, key, reference="reference", roster={A: 0, B: 2}, boot=0)
    without = {k: v for k, v in key.items() if k not in ("report_only_systems", "word_format")}
    with pytest.raises(BarRefused, match="coverage"):              # and two raters of four groups never could be read
        bar_read(answers, without, reference="reference", roster={A: 0, B: 2}, boot=0)


def bar_study(U=8):
    utterances = [dict(sign=f"s{u}", english=f"cut the onion number {u} slowly", truth=dict(negated="no", stratum="non_candidate")) for u in range(U)]
    key = design_study(utterances, ["reference", "gen"], seed=1, distractor_pool=[f"other sentence of the same length {k}" for k in range(8)])
    key["bar"] = dict(strata=dict(confirmed=0., rejected=0., non_candidate=1.))
    return key


def test_a_pilot_rater_on_the_main_keys_excluded_list_is_refused():
    key = bar_study()
    answers = [answer(i, A if i["group"] == 0 else B) for i in key["items"]]
    assert bar_read(answers, dict(key, excluded_raters=[C]), reference="reference", roster={A: 0, B: 1}, boot=0)["raters_read"] == 2
    with pytest.raises(BarRefused, match="excluded list"):
        bar_read(answers, dict(key, excluded_raters=[A]), reference="reference", roster={A: 0, B: 1}, boot=0)
    with pytest.raises(BarRefused, match="excluded list"):        # on the roster, with no sheet yet
        bar_read([a for a in answers if a["reviewer_sha256"] == B], dict(key, excluded_raters=[A]), reference="reference", roster={A: 0, B: 1}, boot=0)
    assert "excluded_raters" in (ROOT / "docs/HUMAN_EVALUATION_PROTOCOL.md").read_text()


def test_the_declared_bar_is_unchanged():
    assert BAR == dict(comprehension_lower=.70, within_points=.10, recall_lower=.70, false_alarm_upper=.20, unviewable_share=.05,
                       coverage=.95, comprehension_alpha=.04, gap_alpha=.05)
    guide = (ROOT / "docs/HUMAN_EVALUATION_PROTOCOL.md").read_text()
    bar = guide[guide.index("**Release bar (owner's rule"):guide.index("The instrument, realism, serving")]
    assert hashlib.sha256(bar.encode()).hexdigest() == "4aaa1ebbd19ae49f638388976c4aeaa6e96aff6f4cabfe3def6874efa304599b"   # byte for byte the declared text


# ---- the contrasts ------------------------------------------------------------------------------------------------------------

def per_sentence(key, answers, system):
    items = {i["item_id"]: i for i in key["items"]}
    out = {}
    for a in answers:
        i = items[a["item_id"]]
        if i["system"] == system:
            out.setdefault(i["utterance"], []).append(float(a["choice"] == i["correct"]))
    return {u: sum(v) / len(v) for u, v in out.items()}


def test_the_shuffle_contrast_is_paired_by_sentence_as_the_scorer_pairs():
    key = study(20)
    answers = sheet(key, (0, 2), composition_shuffled=lambda i: i["utterance"] % 3 == 0)
    assert "reads" in word_format_read(answers, key)["contrasts"]["reference_minus_oracle"]
    assert "not the word-by-word format alone" in word_format_read(answers, key)["contrasts"]["reference_minus_oracle"]["reads"]
    row = word_format_read(answers, key)["contrasts"]["shuffled_minus_composition"]["all"]["comprehension_choice"]
    mine, other = per_sentence(key, answers, "composition_shuffled"), per_sentence(key, answers, "composition")
    common = sorted(set(mine) & set(other))
    assert row["utterances"] == len(common) == 10
    assert row["mean_difference"] == pytest.approx(sum(mine[u] - other[u] for u in common) / 10, abs=1e-4)
    assert row["p_value"] == pytest.approx(paired_permutation_pvalue([mine[u] for u in common], [other[u] for u in common], seed=0), abs=1e-4)
    assert row["ci95_sentences_only"][0] <= row["mean_difference"] <= row["ci95_sentences_only"][1]
    plain = score_study(answers, key, reference="reference")
    assert plain["systems"]["composition"]["comprehension_choice"]["rate"] == 1.


def test_method_d_of_a_contrast_by_hand():
    key = study(20)
    answers = sheet(key, (0, 1, 2, 3), composition_shuffled=lambda i: i["utterance"] % 2 == 0)
    row = word_format_read(answers, key)["contrasts"]["shuffled_minus_composition"]["all"]["comprehension_choice_method_d"]
    items = {i["item_id"]: i for i in key["items"]}
    use = [a for a in answers if items[a["item_id"]]["system"] in ("composition", "composition_shuffled")]
    u = np.array([items[a["item_id"]]["utterance"] for a in use])
    r = np.array([a["reviewer_sha256"] == A for a in use]) * 1 + np.array([a["reviewer_sha256"] == B for a in use]) * 2 + np.array([a["reviewer_sha256"] == C for a in use]) * 3
    sign = np.array([1. if items[a["item_id"]]["system"] == "composition_shuffled" else -1. for a in use])
    x = np.array([float(a["choice"] == items[a["item_id"]]["correct"]) for a in use])
    psi = sign * (x - np.where(sign > 0, x[sign > 0].mean(), x[sign < 0].mean())) / 20
    from scipy.stats import t
    half = t.ppf(.975, 3) * np.sqrt(twoway_variance(psi, u, r))
    difference = x[sign > 0].mean() - x[sign < 0].mean()
    assert row["difference"] == pytest.approx(difference, abs=1e-4) and row["raters"] == 4
    assert row["cluster95"] == pytest.approx([difference - half, difference + half], abs=1e-3)


def test_moved_and_same_words_rows_read_only_those_sentences():
    key = study(20, unmoved=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), partial=(0, 1, 2, 3))
    answers = sheet(key, (0, 1, 2, 3))
    read = word_format_read(answers, key)
    assert read["sentences_moved"] == 10 and read["sentences_same_words"] == 16
    shuffled = read["contrasts"]["shuffled_minus_composition"]
    assert shuffled["all"]["comprehension_choice"]["utterances"] == 20 and shuffled["moved"]["comprehension_choice"]["utterances"] == 10
    assert read["contrasts"]["oracle_minus_composition"]["same_words"]["naturalness"]["utterances"] == 16
    assert "same_words" not in shuffled and "moved" not in read["contrasts"]["reference_minus_oracle"]
    for row in read["contrasts"].values():
        assert row["all"]["rater_balance"]["confounded_with_rater"] is False


def test_a_contrast_no_rater_saw_both_arms_of_is_flagged_as_confounded():
    key = study(20)
    answers = sheet(key, (0, 1))                                   # adjacent groups: one rater per arm on the shared sentences
    pair = word_format_read(answers, key)["contrasts"]["oracle_minus_composition"]["all"]
    assert pair["comprehension_choice"]["utterances"] == 5 and pair["rater_balance"]["confounded_with_rater"] is True
    assert pair["comprehension_choice_by_the_rule"] == "unresolved"
    assert word_format_read(answers, key)["contrasts"]["shuffled_minus_composition"]["all"]["comprehension_choice"]["utterances"] == 0
    opposite = word_format_read(sheet(key, (0, 2)), key)["contrasts"]["shuffled_minus_composition"]["all"]
    assert opposite["rater_balance"]["confounded_with_rater"] is False
    assert opposite["rater_balance"]["ratings"] == {A[:12]: [5, 5], B[:12]: [5, 5]}


def test_a_contrast_with_no_shared_sentence_says_so_and_reads_within_raters():
    key = study(20)
    answers = sheet(key, (0, 2), composition=lambda i: True, oracle=False)
    pair = word_format_read(answers, key)["contrasts"]["oracle_minus_composition"]
    assert pair["all"]["comprehension_choice"] == dict(utterances=0, mean_difference=None, p_value=None, ci95_sentences_only=None)
    assert pair["all"]["comprehension_choice_method_d"]["cluster95"] is None
    assert pair["within_rater"]["per_rater"] == {A[:12]: -1., B[:12]: -1.} and pair["within_rater"]["mean"] == -1.
    assert "different sentences" in pair["within_rater"]["note"] and "decides nothing" in pair["within_rater"]["note"]
    shared = word_format_read(answers, key)["contrasts"]["shuffled_minus_composition"]     # a shared sentence: no within-rater row
    assert "within_rater" not in shared and shared["all"]["comprehension_choice"]["utterances"] == 10


def test_method_d_gives_no_interval_on_zero_variance():
    key = study(20)
    row = word_format_read(sheet(key, (0, 1, 2, 3), composition_shuffled=False), key)["contrasts"]["shuffled_minus_composition"]["all"]
    assert row["comprehension_choice_method_d"]["difference"] == -1 and row["comprehension_choice_method_d"]["cluster95"] is None
    assert row["comprehension_choice_by_the_rule"] == "unresolved"


def test_two_raters_give_an_honest_interval():
    key = study(20)
    answers = sheet(key, (0, 2), composition_shuffled=lambda i: i["utterance"] % 8 in (0, 2))
    read = word_format_read(answers, key)["contrasts"]["shuffled_minus_composition"]["all"]
    row = read["comprehension_choice_method_d"]
    assert row["raters"] == 2 and row["cluster95"][1] - row["cluster95"][0] > .5 and read["comprehension_choice_by_the_rule"] == "unresolved"


def rule_row(*, p=.01, mean=-.3, cluster=(-.5, -.1), raters=8, confounded=False):
    return dict(comprehension_choice=dict(p_value=p, mean_difference=mean),
                comprehension_choice_method_d=dict(cluster95=None if cluster is None else list(cluster), raters=raters),
                rater_balance=dict(confounded_with_rater=confounded))


def test_the_rule_needs_method_d_the_permutation_the_raters_and_no_confound():
    assert _by_the_rule(rule_row()) == "below"
    assert _by_the_rule(rule_row(mean=.3, cluster=(.1, .5))) == "above"
    assert _by_the_rule(rule_row(raters=RULE_MIN_RATERS)) == "below"
    assert _by_the_rule(rule_row(raters=RULE_MIN_RATERS - 1)) == "unresolved"
    assert _by_the_rule(rule_row(cluster=(-.5, .1))) == "unresolved"            # method D holds zero, however small p is
    assert _by_the_rule(rule_row(cluster=None)) == "unresolved"
    assert _by_the_rule(rule_row(p=.2)) == "unresolved"                         # the permutation disagrees
    assert _by_the_rule(rule_row(confounded=True)) == "unresolved"
    assert _by_the_rule(rule_row(mean=.3)) == "unresolved"                      # the two point the opposite ways
    assert _by_the_rule(rule_row(mean=-.3, cluster=(.1, .5))) == "unresolved"
    empty = dict(rule_row(), comprehension_choice=dict(p_value=None, mean_difference=None))
    assert _by_the_rule(empty) == "unresolved"


def test_the_rule_reads_a_real_key_only_with_enough_raters():
    key = study(60)
    raters_of = lambda a: a["reviewer_sha256"]  # noqa: E731
    rule = lambda answers: word_format_read(answers, key)["contrasts"]["shuffled_minus_composition"]["all"]["comprehension_choice_by_the_rule"]  # noqa: E731
    rng = np.random.default_rng(5)
    chance = {(g, u): rng.random() for g in range(8) for u in range(60)}
    answers = eight(key, composition_shuffled=lambda i, r: chance[(i["group"], i["utterance"])] < .3,
                    composition=lambda i, r: chance[(i["group"], i["utterance"])] < .9)
    assert rule(answers) == "below"
    few = sorted({raters_of(a) for a in answers})[:4]
    assert rule([a for a in answers if raters_of(a) in few]) == "unresolved"


def test_a_rater_who_saw_both_arms_clears_the_confound():
    key = study(20)
    answers = eight(key)
    items = {i["item_id"]: i for i in key["items"]}
    lost = "0a".ljust(64, "a")                                      # one rater never saw composition; the others saw both
    one_sided = [a for a in answers if not (a["reviewer_sha256"] == lost and items[a["item_id"]]["system"] == "composition")]
    balance = word_format_read(one_sided, key)["contrasts"]["shuffled_minus_composition"]["all"]["rater_balance"]
    assert balance["ratings"][lost[:12]][1] == 0 and balance["ratings"][lost[:12]][0] > 0
    assert balance["confounded_with_rater"] is False                # any rater with both arms is enough


def test_translation_and_naturalness_are_contrasted_too():
    key = study(20)
    items = {i["item_id"]: i for i in key["items"]}
    answers = []
    for i in key["items"]:
        if i["group"] in (0, 2):
            shuffled = i["system"] == "composition_shuffled"
            text = "slowly onion cut banana" if shuffled else i["english"] if i["utterance"] % 2 else ""     # a blank translation counts as 0
            answers.append(answer(i, A if i["group"] == 0 else B, natural=2 if shuffled else 4, translation=text))
    assert {items[a["item_id"]]["system"] for a in answers} == set(SYSTEMS)
    row = word_format_read(answers, key)["contrasts"]["shuffled_minus_composition"]["all"]
    assert row["naturalness"]["mean_difference"] == -2.
    assert row["translation_content_words"]["utterances"] == 10
    expected = np.mean([(3 / 5) - (1. if items[a["item_id"]]["utterance"] % 2 else 0.) for a in answers
                        if items[a["item_id"]]["system"] == "composition"])
    assert row["translation_content_words"]["mean_difference"] == pytest.approx(expected, abs=1e-4)
    assert -1 <= row["translation_bleu_indicative"]["mean_difference"] <= 1


def test_the_share_is_read_against_the_instruments_only_with_raters_and_room():
    key = study(60)
    rng = np.random.default_rng(3)
    draws = {(i["group"], i["utterance"], k): rng.random() for i in key["items"] for k in "ab"}
    level = lambda comp, mixed: dict(  # noqa: E731
        composition=lambda i, r: draws[(i["group"], i["utterance"], r[1])] < comp,
        composition_shuffled=lambda i, r: draws[(i["group"], i["utterance"], r[1])] < mixed)
    def share_of(**levels):
        return word_format_read(eight(key, **level(**levels)), key)["share_of_lead_over_chance_removed_by_shuffling"]
    share = share_of(comp=.75, mixed=.55)
    assert share["instruments_blank_and_bleu"] == INSTRUMENT_SHUFFLE_SHARES and share["chance"] == .25 and share["raters"] == 8
    assert share["share"] is not None and share["cluster95"][0] < share["share"] < share["cluster95"][1]
    assert share["ci95_sentences_only"][0] < share["share"] < share["ci95_sentences_only"][1]
    # by hand: the delta method's variance with the two-way formula
    items = {i["item_id"]: i for i in key["items"]}
    answers = eight(key, **level(comp=.75, mixed=.55))
    use = [a for a in answers if items[a["item_id"]]["system"] in ("composition", "composition_shuffled")]
    x = np.array([float(a["choice"] == items[a["item_id"]]["correct"]) for a in use])
    is_c = np.array([items[a["item_id"]]["system"] == "composition" for a in use])
    c, m = x[is_c].mean(), x[~is_c].mean()
    assert share["share"] == pytest.approx((c - m) / (c - .25), abs=1e-3)
    lead = c - .25
    psi = np.where(is_c, (m - .25) / lead ** 2 * (x - c) / is_c.sum(), -(x - m) / (~is_c).sum() / lead)
    u = np.array([items[a["item_id"]]["utterance"] for a in use])
    names = sorted({a["reviewer_sha256"] for a in use})
    r = np.array([names.index(a["reviewer_sha256"]) for a in use])
    from scipy.stats import t
    half = t.ppf(.975, 7) * np.sqrt(twoway_variance(psi, u, r))
    assert share["cluster95"] == pytest.approx([share["share"] - half, share["share"] + half], abs=2e-3)
    # the verdict: below, above, around, each only on method D's interval
    assert share_of(comp=.8, mixed=.79)["against_instruments"] in ("below", "around, or too wide to tell")
    assert share_of(comp=.85, mixed=.3)["against_instruments"] == "above"
    # at the ceiling, or with fewer raters, or no lead: not read
    ceiling = share_of(comp=1., mixed=.85)
    assert ceiling["against_instruments"] == "not read" and ceiling["share"] is not None and ceiling["cluster95"] is not None
    four = [a for a in eight(key, **level(comp=.75, mixed=.55)) if a["reviewer_sha256"][1] == "a"]
    read = word_format_read(four, key)["share_of_lead_over_chance_removed_by_shuffling"]
    assert read["raters"] == 4 and read["against_instruments"] == "not read" and read["share"] is not None
    floor = word_format_read(eight(key, composition=False), key)["share_of_lead_over_chance_removed_by_shuffling"]
    assert floor["share"] is None and floor["against_instruments"] == "not read" and floor["ci95_sentences_only"] is None
    few_key = study(20, unmoved=range(20))
    assert word_format_read(sheet(few_key, (0, 2)), few_key)["share_of_lead_over_chance_removed_by_shuffling"]["utterances"] == 0


def test_no_share_is_printed_when_its_interval_over_sentences_is_undefined():
    key = study(8)
    # composition barely above chance: over 1% of the resamples have no lead, so no interval, so no share
    answers = sheet(key, (0, 1, 2, 3), composition=lambda i: i["utterance"] in (0, 1, 2), composition_shuffled=False)
    share = word_format_read(answers, key)["share_of_lead_over_chance_removed_by_shuffling"]
    assert share["share"] is None and share["ci95_sentences_only"] is None and share["against_instruments"] == "not read"
    assert share["utterances"] == 8


def test_the_read_says_it_is_report_only_and_counts_unviewable_rows_out():
    key = study(20)
    answers = sheet(key, (0, 2))
    answers[0]["viewable"] = "no"
    read = word_format_read(answers, key)
    assert read["report_only"] is True and read["counts_towards_release_bar"] is False and read["ratings"] == 39
    assert "small drop in choice" in read["reading"] and read["arms"]["reference"]["label"] == "reference"
    assert f"at least {SHARE_MIN_RATERS} raters" in read["reading"] and "ci95_sentences_only" in read["reading"]
    assert read["arms"]["composition"]["ci95"][1] == 1.
    assert sum(read["arms"][r]["n"] for r in read["arms"]) == 39
    nothing = word_format_read([], key)                            # no sheet back: nothing is read, and it says so
    assert nothing["ratings"] == 0 and nothing["contrasts"]["shuffled_minus_composition"]["all"]["comprehension_choice_by_the_rule"] == "unresolved"
    assert nothing["share_of_lead_over_chance_removed_by_shuffling"]["share"] is None


# ---- the seeded order ---------------------------------------------------------------------------------------------------------

def test_the_order_is_seeded_and_never_the_identity():
    assert wf.shuffled_order(6, 11, 0) == wf.shuffled_order(6, 11, 0)
    assert any(wf.shuffled_order(6, 11, s) != wf.shuffled_order(6, 11, 0) for s in range(1, 5))
    assert any(wf.shuffled_order(6, c, 0) != wf.shuffled_order(6, 11, 0) for c in range(5))
    for n in range(2, 9):
        for clip in range(30):
            order = wf.shuffled_order(n, clip, 0)
            assert sorted(order) == list(range(n)) and order != list(range(n))
    assert wf.shuffled_order(1, 3, 0) == [0] and wf.shuffled_order(0, 3, 0) == []
    assert wf.moved(["cut", "onion"], [1, 0]) and not wf.moved(["cut", "cut"], [1, 0]) and not wf.moved(["cut"], [0])


def test_keep_frames_moves_the_plan_as_the_instrument_control_does():
    """``render_form(keep_frames=True)`` is byte for byte ``parent_controls.render_order``: same cores, same gaps, same total."""
    import parent_controls as pc
    import round15_stage2 as r2
    from build_recon_states import state_bytes
    from signtranslator.generation.compositional import parts_from_pieces
    from tests.test_parent_controls import TIMING, VOCAB, core_frames, core_spans, rendered_pieces
    pieces = rendered_pieces()
    material = dict(pieces=pieces, parts=parts_from_pieces(pieces))
    words = ["alpha", "beta", "gamma", "delta"]
    job = dict(words=words, weights=r2.plan_words(words, dict(timing=TIMING), VOCAB)["weights"])
    model = dict(timing=TIMING)
    order = [2, 0, 3, 1]
    base, plan = r2.render_form(material, job, 3., model, timing="sign-fitted", sample_id="x")
    moved, moved_plan = r2.render_form(material, job, 3., model, timing="sign-fitted", order=order, keep_frames=True, sample_id="x")
    assert len(moved.timestamps) == len(base.timestamps) == plan["frames"]
    assert moved_plan == pc.moved_plan(plan, order) and moved_plan["gaps"] == plan["gaps"]
    assert state_bytes(moved)[1] == state_bytes(pc.render_order(pieces, plan, order, "x"))[1]
    base_spans, moved_spans = core_spans(plan), core_spans(moved_plan)
    for at, k in enumerate(order):                                  # every word's rendered core is frame for frame the composition's
        one, two = core_frames(base, *base_spans[k]), core_frames(moved, *moved_spans[at])
        for name in one:
            assert np.array_equal(one[name], two[name]), (name, k)
    assert state_bytes(moved)[1] != state_bytes(base)[1]
    realloc, other = r2.render_form(material, job, 3., model, timing="sign-fitted", order=order, sample_id="x")
    assert other["cores"] != moved_plan["cores"]                    # without keep_frames the frames are re-allocated
    assert len(realloc.timestamps) == len(base.timestamps)
    with pytest.raises(ValueError, match="keep_frames"):
        r2.render_form(material, job, 3., model, timing="sign-fitted", keep_frames=True)
    with pytest.raises(ValueError, match="keep_frames"):
        r2.render_form(material, job, 3., model, timing="stage1", order=order, keep_frames=True)
    # the other route through render_form, a bridge or a signing space, moves the same way
    jerk, jerk_plan = r2.render_form(material, job, 3., model, timing="sign-fitted", order=order, keep_frames=True, bridge="minimum-jerk", sample_id="x")
    assert len(jerk.timestamps) == plan["frames"] and jerk_plan["cores"] == moved_plan["cores"]


def continuous_args():
    from types import SimpleNamespace
    return SimpleNamespace(retime="continuous", continuous_ratio=.6, tempo=1., no_bridge_velocity=False, bridge_min_speed=.3, bridge_max_seconds=.6,
                           bridge="linear", lowpass_hz=None, arm_scale=None, hold_config=None, timing="sign-fitted", total_fit="pad")


def test_keep_frames_on_the_continuous_route_keeps_the_plan_of_the_composition():
    """Gaps and core frames come from the sentence order; each word then keeps its own. Moved joins keep gaps sized to other neighbours."""
    from build_recon_states import state_bytes
    from signtranslator.generation.compositional import parts_from_pieces
    from tests.test_parent_controls import FRACTIONS, TIMING, VOCAB, core_frames, laid_out, signer_state, TWO_VIEW
    import round15_stage2 as r2
    words = ["alpha", "beta", "gamma", "delta"]
    amplitude = (.2, 1.4, .3, 1.1)
    pieces = []
    for n, w in enumerate(words):
        clip = laid_out(signer_state(50 + n * 7, frames=30 + 6 * n, amplitude=amplitude[n]), TWO_VIEW)
        pieces.append(dict(state=clip, **r2.sign_core(w, clip, fractions=FRACTIONS, margin=0.)))
    material = dict(pieces=pieces, parts=parts_from_pieces(pieces))
    timing = dict(TIMING, gap=0., lead_share=.4, sentence_seconds=dict(alpha=.5, beta=.2, gamma=.9, delta=.35))      # every gap is the bound's own
    model = dict(timing=timing)
    job = dict(words=words)
    order = [2, 3, 0, 1]
    args = continuous_args()
    base, plan = ce.render_round20(args, material, job, 5., model, VOCAB, order=None, sample_id="x", space=None)
    moved, moved_plan = ce.render_round20(args, material, job, 5., model, VOCAB, order=order, sample_id="x", space=None, keep_frames=True)
    assert moved_plan["cores"] == [plan["cores"][k] for k in order]     # the plan is the sentence's, moved once
    assert moved_plan["gaps"] == plan["gaps"] and moved_plan["lead"] == plan["lead"] and moved_plan["trail"] == plan["trail"]
    assert moved_plan["sources"] == [plan["sources"][k] for k in order]
    assert len(moved.timestamps) == len(base.timestamps) == plan["frames"]
    assert state_bytes(moved)[1] != state_bytes(base)[1]
    start = plan["lead"]
    spans_base, at = {}, start
    for k, f in enumerate(plan["cores"]):
        spans_base[k] = (at, f)
        at += f + (plan["gaps"][k] if k < 3 else 0)
    at = start
    for place, k in enumerate(order):
        f = moved_plan["cores"][place]
        one, two = core_frames(base, *spans_base[k]), core_frames(moved, at, f)
        for name in one:
            assert np.array_equal(one[name], two[name]), (name, k)
        at += f + (moved_plan["gaps"][place] if place < 3 else 0)
    # without keep_frames the velocity plan is built for the new neighbours: another plan
    free, free_plan = ce.render_round20(args, material, job, 5., model, VOCAB, order=order, sample_id="x", space=None)
    assert free_plan["gaps"] != moved_plan["gaps"]
    # the log states it: the moved joins are measured against the bound, from the moved parts
    joins = wf.join_excess(parts_from_pieces(pieces, order), moved_plan, min_speed=.3)
    own = wf.join_excess(material["parts"], plan, min_speed=.3)
    assert [j["join"] for j in joins] == [0, 1, 2] and all(j["gap_frames"] == g for j, g in zip(joins, plan["gaps"]))
    assert all(j["over_bound"] <= 1 for j in own)                        # sized to the bound in sentence order
    assert max(j["over_bound"] for j in joins) > 1                       # a moved join past the bound, which is the stated limit


def test_keep_frames_off_changes_nothing_on_the_main_route():
    """The default call is the plan and render it always was, on the routes other suites read: their results cannot have moved."""
    import inspect
    import round15_stage2 as r2
    from build_recon_states import state_bytes
    from signtranslator.generation import compositional_grammar as cg
    from signtranslator.generation.compositional import parts_from_pieces, render
    from tests.test_parent_controls import TIMING, VOCAB, rendered_pieces
    assert inspect.signature(r2.render_form).parameters["keep_frames"].default is False
    assert inspect.signature(ce.render_round20).parameters["keep_frames"].default is False
    pieces = rendered_pieces()
    material = dict(pieces=pieces, parts=parts_from_pieces(pieces))
    words = ["alpha", "beta", "gamma", "delta"]
    weights = r2.plan_words(words, dict(timing=TIMING), VOCAB)["weights"]
    job = dict(words=words, weights=weights)
    model = dict(timing=TIMING)
    for order in (None, [2, 0, 3, 1]):
        idx = list(range(4)) if order is None else order
        plan = cg.fitted_plan([weights[k] for k in idx], cg.gap_ratio(TIMING), 3.)
        parts = material["parts"] if order is None else parts_from_pieces(pieces, idx)
        for bridge in ("linear", "minimum-jerk"):
            hand = render(parts, plan, sample_id="x", bridge_mode=bridge)
            state, got = r2.render_form(material, job, 3., model, timing="sign-fitted", order=order, sample_id="x", bridge=bridge)
            assert state_bytes(state)[1] == state_bytes(hand)[1] and got["cores"] == plan["cores"] and got["gaps"] == plan["gaps"]
    from types import SimpleNamespace
    args = SimpleNamespace(retime="reference", timing="sign-fitted", total_fit="pad", bridge="linear", lowpass_hz=None, arm_scale=None)
    via, _ = ce.render_round20(args, material, job, 3., model, VOCAB, order=None, sample_id="x", space=None)
    assert state_bytes(via)[1] == state_bytes(r2.render_form(material, job, 3., model, timing="sign-fitted", sample_id="x")[0])[1]


# ---- what the oracle may show -------------------------------------------------------------------------------------------------

def test_only_permitted_corpora_reach_raters_and_every_one_needs_the_owners_note():
    wf.check_oracle_corpora(("asllvd-text-v1",), "owner confirmed ASLLVD train clips, 2026-11-05")
    wf.check_oracle_corpora(("asllvd-text-v1", "rit-text-r6"), "owner confirmed ASLLVD and RIT, 2026-11-05")
    for corpus in ("asl-citizen-text-v1", "asl-citizen", "stem-wiki-text-v1", "asl-stem-wiki", "youtube-asl-v1", "semlex-text-v1", "how2sign-two-view-text-v1"):
        with pytest.raises(SystemExit, match="may not show"):
            wf.check_oracle_corpora(("asllvd-text-v1", corpus), "a note does not help")
    for corpus in ("asllvd-text-v1", "rit-text-r6", "wlasl-text-r6"):
        for note in (None, "  ", ""):
            with pytest.raises(SystemExit, match="confirmation"):
                wf.check_oracle_corpora((corpus,), note)
    with pytest.raises(SystemExit, match="needs a corpus"):
        wf.check_oracle_corpora((), "x")


@pytest.mark.parametrize("change,message", [(dict(signer_component="c00"), "c00"), (dict(partition="test"), "train"), (dict(partition="val"), "train"),
                                            (dict(corpus="rit-text-r6"), "outside"), (dict(corpus="asl-citizen-text-v1"), "outside")])
def test_a_record_is_refused_before_it_is_fetched(change, message):
    record = dict(sign="w1", corpus="asllvd-text-v1", partition="train", signer_component=None)
    assert wf.check_oracle_record(record, ("asllvd-text-v1",)) is record
    with pytest.raises(SystemExit, match=message):
        wf.check_oracle_record(dict(record, **change), ("asllvd-text-v1",))


def test_the_pools_hold_train_records_of_the_permitted_corpora_and_nothing_else(tmp_path, stage2_store):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    manifest = json.loads(store.get("mix.json"))
    word = next(r for r in manifest["records"] if r["corpus"] == "asllvd-text-v1" and r["partition"] == "train")
    planted = [dict(word, sign="leak-test", partition="test"), dict(word, sign="leak-val", partition="val"),
               dict(word, sign="leak-rit", corpus="rit-text-r6"), dict(word, sign="leak-citizen", corpus="asl-citizen-text-v1")]
    pools = wf.oracle_pools(dict(records=manifest["records"] + planted), ("asllvd-text-v1",), frozenset())
    held = {r["sign"] for v in pools.values() for r in v}
    assert word["sign"] in held and not any(s.startswith("leak") for s in held)
    entry = next(e for e, v in pools.items() if word in v)
    assert entry not in wf.oracle_pools(manifest, ("asllvd-text-v1",), frozenset({entry}))     # an added entry has none
    with pytest.raises(SystemExit, match="c00"):
        wf.oracle_pools(dict(records=[dict(word, signer_component="c00")]), ("asllvd-text-v1",), frozenset())
    how2sign = next(r for r in manifest["records"] if r["corpus"] == "how2sign-two-view-text-v1" and r["partition"] == "train")
    wf.check_template_record(dict(records=[how2sign]))
    with pytest.raises(SystemExit, match="c00"):
        wf.check_template_record(dict(records=[dict(how2sign, signer_component="c00")]))
    with pytest.raises(SystemExit, match="c00"):
        wf.check_template_record(dict(records=[]))


def test_the_form_name_keeps_the_timing_mark_last():
    assert wf.form_name("core_crossfade", wf.SHUFFLED) == "core_crossfade/composition_shuffled"
    assert wf.form_name("core_crossfade@predicted", wf.ORACLE) == "core_crossfade/oracle@predicted"
    row = dict(form=wf.form_name("core_crossfade@predicted", wf.ORACLE), frames=60, reference_frames=48, sign="s")
    assert bh.composition_timing(row, "oracle") == "predicted"
    row = dict(form=wf.form_name("core_crossfade", wf.ORACLE), frames=60, reference_frames=48, sign="s")
    with pytest.raises(SystemExit, match="duration would be a cue"):
        bh.composition_timing(row, "oracle")


# ---- the packet refuses what does not fit -----------------------------------------------------------------------------------

def built(frames=48, words=("cut", "onion"), shuffled_words=None, oracle_words=None, role_arm="arm", empty=False, form="core_crossfade@predicted"):
    rows = {s: dict(sign=s, frames=frames, reference_frames=frames, empty=False, state_sha256=f"sha-{s}", form=form,
                    words=[dict(word=w) for w in words]) for s in ("p1", "p2", "p3", "p4")}
    link = dict(arm=role_arm, form=form)
    shuffled = {s: dict(r, composition_state_sha256=r["state_sha256"], form=wf.form_name(form, wf.SHUFFLED),
                        words=[dict(word=w) for w in (shuffled_words or tuple(reversed(words)))]) for s, r in rows.items()}
    oracle = {s: dict(r, composition_state_sha256=r["state_sha256"], form=wf.form_name(form, wf.ORACLE), same_words=False,
                      words=[dict(word=w, corpus="asllvd-text-v1") for w in (oracle_words or words[:1])]) for s, r in rows.items()}
    if empty:
        oracle["p2"] = dict(oracle["p2"], empty=True, words=[])
    return {"gen": dict(arm="arm", rows=rows, form=form, word_format=None),
            "mixed": dict(arm="arm/composition_shuffled", rows=shuffled, form=wf.form_name(form, wf.SHUFFLED),
                          word_format=dict(role=wf.SHUFFLED, composition=link, form=form, seed=0)),
            "real": dict(arm="arm/oracle", rows=oracle, form=wf.form_name(form, wf.ORACLE),
                         word_format=dict(role=wf.ORACLE, composition=link, form=form, licence_note="owner's note on ASLLVD train clips"))}


PACKET = [dict(label="mixed", kind="composition"), dict(label="real", kind="composition"), dict(label="gen", kind="composition"),
          dict(label="truth", kind="reference")]
SIGNS = ["p1", "p2", "p3", "p4"]


def test_the_packet_takes_the_arms_in_the_pilot_only():
    made = bh.check_word_format(PACKET, built(), which="pilot", signs=SIGNS)
    assert made["roles"] == dict(composition="gen", composition_shuffled="mixed", oracle="real", reference="truth")
    assert made["signs"] == SIGNS and made["dropped_no_real_clip"] == [] and ["mixed", "gen"] in made["pairing"]["paired_by_sentence"]
    with pytest.raises(SystemExit, match="pilot only"):
        bh.check_word_format(PACKET, built(), which="main", signs=SIGNS)
    plain = {"gen": dict(arm="arm", rows={}, form="core_crossfade", word_format=None)}
    assert bh.check_word_format(PACKET[2:], plain, which="main", signs=SIGNS) is None     # the main read is untouched


def test_the_packet_refuses_arms_that_would_show():
    arms = built()
    arms["real"]["word_format"]["licence_note"] = None
    with pytest.raises(SystemExit, match="confirmation"):
        bh.check_word_format(PACKET, arms, which="pilot", signs=SIGNS)
    arms = built()
    only_one = {k: v for k, v in arms.items() if k != "real"}
    with pytest.raises(SystemExit, match="one composition"):
        bh.check_word_format(PACKET, only_one, which="pilot", signs=SIGNS)
    with pytest.raises(SystemExit, match="another composition"):
        bh.check_word_format(PACKET, built(role_arm="other"), which="pilot", signs=SIGNS)
    arms = built()
    arms["mixed"]["rows"]["p1"]["frames"] += 1
    with pytest.raises(SystemExit, match="frames"):
        bh.check_word_format(PACKET, arms, which="pilot", signs=SIGNS)
    arms = built()
    arms["real"]["rows"]["p1"]["composition_state_sha256"] = "other"
    with pytest.raises(SystemExit, match="another composition"):
        bh.check_word_format(PACKET, arms, which="pilot", signs=SIGNS)
    with pytest.raises(SystemExit, match="exactly the composition's words"):
        bh.check_word_format(PACKET, built(shuffled_words=("cut",)), which="pilot", signs=SIGNS)
    with pytest.raises(SystemExit, match="does not"):
        bh.check_word_format(PACKET, built(oracle_words=("salt",)), which="pilot", signs=SIGNS)
    arms = built()
    arms["real"]["rows"]["p3"]["words"][0]["corpus"] = "asl-citizen-text-v1"
    with pytest.raises(SystemExit, match="may not show"):
        bh.check_word_format(PACKET, arms, which="pilot", signs=SIGNS)


def test_a_sentence_the_oracle_cannot_sign_leaves_the_pilot_and_is_listed():
    made = bh.check_word_format(PACKET, built(empty=True), which="pilot", signs=SIGNS)
    assert made["dropped_no_real_clip"] == ["p2"] and made["signs"] == ["p1", "p3", "p4"]


# ---- the export and the whole pilot, on the synthetic store --------------------------------------------------------------------

NOTE = "owner recorded: ASLLVD train clips may be shown under the DAI terms (test value)"


def pilot_setup(tmp_path, store, sha, monkeypatch, *, seed="0", extra=()):
    """A D20-like set of five, a word cache, a draw of 1 + 4, and the arm's export command at predicted timing, the official arm's."""
    import torch
    import validate_sign_recognizer as vsr
    from scripts.round20_word_cache import main as cache_main
    from signtranslator.generation import composition_v2 as cv2
    from tests.test_round20_predicted import MODEL, PIN
    monkeypatch.setattr(cv2, "fit_duration_model", lambda manifest: MODEL)   # the fixture's mix is too small to fit one
    set_sha, _ = put_set(store, sha, order=(0, 1, 2, 3, 4))
    monkeypatch.setattr(bh, "D20_SHA256", set_sha)
    labels = ["CUT", "ONION", "SALT", "HEAT", "KNIFE", "HALF", "NOW"]
    monkeypatch.setattr(vsr, "load_model", lambda store, run, device: (None, labels, {"checkpoint_key": "r.pt"}))

    def logits(model, states, device, batch=64):
        out = []
        for s in states:
            rh = s.channels["right_hand"]
            level = float(np.abs(rh.values[rh.valid]).mean()) if rh.valid.any() else 0.
            out.append(torch.tensor([np.sin(level * (k + 3) * 50.) for k in range(len(labels))]))
        return torch.stack(out)
    monkeypatch.setattr(vsr, "logits_of", logits)
    root = f"dir:{tmp_path / 'store'}"
    assert cache_main(["run", "--store", root, "--prefix", "cache/w", "--mix-key", "mix.json", "--mix-sha256", sha, "--sentences-key",
                       "sets/d20.json", "--sentences-sha256", set_sha, "--clips", "5", "--checkpoint-key", "prior.pt", "--samples", "2",
                       "--choose", "1,2", "--device", "cpu"]) == 0
    assert cache_main(["merge", "--store", root, "--prefix", "cache/w"]) == 0
    cache_sha = hashlib.sha256(store.get("cache/w/index.json")).hexdigest()
    monkeypatch.setattr(bh, "negation_words", lambda t: [])
    draw_dir = tmp_path / "draw"
    assert bh.main(["how2sign-draw", "--store", root, "--mix-key", "mix.json", "--mix-sha256", sha, "--sentences-key", "sets/d20.json",
                    "--sentences-sha256", set_sha, "--main", "1", "--pilot", "4", "--min-seconds", ".5", "--seed", "11", "--name", "test-draw", "--output-dir", str(draw_dir)]) == 0
    draw_path = draw_dir / "draw.json"
    draw_sha = hashlib.sha256(draw_path.read_bytes()).hexdigest()
    args = [a for a in run_args(tmp_path, sha) if a not in ("--checkpoint-key", "prior.pt")]
    args[args.index("core_crossfade,oracle_stage2")] = "core_crossfade"
    args += ["--clips", "5", "--selection", "stage1", "--timing", "sign-fitted", "--sentences-key", "sets/d20.json", "--sentences-sha256", set_sha, "--word-cache-key", "cache/w/index.json",
             "--word-cache-sha256", cache_sha, "--word-cache-n", "2", "--export-prefix", "x/arm", "--export-arm", "arm", "--export-sentences-file", str(draw_path),
             "--export-sentences-sha256", draw_sha, "--export-only", "--export-word-format", "--word-format-seed", seed, "--word-format-licence-note", NOTE, "--output-file", str(tmp_path / "arm.json"),
             "--predicted-duration", f"--expect-duration-model={PIN}", "--export-forms", "core_crossfade@predicted", *extra]
    return args, dict(set_sha=set_sha, draw_path=draw_path, draw_sha=draw_sha, draw=json.loads(draw_path.read_text()), root=root)


def test_the_export_writes_both_arms_for_the_pilot_at_the_composition_frames(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    from evaluate_text_to_motion import load_state
    import build_recon_states as brs
    store, sha, s1, oracle, argv = stage2_store
    args, info = pilot_setup(tmp_path, store, sha, monkeypatch)
    fetched = []
    original_get = type(store).get

    def spying(self, key):
        fetched.append(key)
        return original_get(self, key)
    monkeypatch.setattr(type(store), "get", spying)
    decodes, real_decode = [], brs.decode_state

    def counted(state, tokenizer, motion_config, post, *, text, device="cpu"):
        decodes.append(text)
        return real_decode(state, tokenizer, motion_config, post, text=text, device=device)
    monkeypatch.setattr(brs, "decode_state", counted)
    assert ce.main(args) == 0
    plain = json.loads(store.get("x/arm/index.json"))
    assert "export_word_format" not in plain["settings"] and "word_format_seed" not in plain["settings"]
    assert len(plain["states"]) == 5                                   # the arm's own export covers the whole list
    comp = {r["sign"]: r for r in plain["states"]}
    shuffled = json.loads(store.get("x/arm/word-format/composition_shuffled/index.json"))
    real = json.loads(store.get("x/arm/word-format/oracle/index.json"))
    assert sorted(r["sign"] for r in shuffled["states"]) == sorted(info["draw"]["pilot"]) == sorted(r["sign"] for r in real["states"])
    assert shuffled["sentences"] == plain["sentences"] and shuffled["sentence_set"]["set"] == "D20"
    assert shuffled["word_format"]["role"] == "composition_shuffled" and shuffled["word_format"]["report_only"] is True
    assert real["word_format"]["pilot_only"] is True and real["word_format"]["oracle_corpora"] == ["asllvd-text-v1"]
    assert real["word_format"]["licence_note"] == NOTE and real["word_format"]["coverage"]["core_stretch_mean"] >= 1
    assert real["word_format"]["composition"]["sha256"] == hashlib.sha256(store.get("x/arm/index.json")).hexdigest()
    assert real["word_format"]["round_trip"]["checkpoint_key"] == "prior.pt"
    assert decodes and set(decodes) <= {w["word"] for r in plain["states"] for w in r["words"]}
    assert len(decodes) == len(set(decodes))                           # one real clip per entry, as a word cache holds one sign
    words_moved = 0
    for row in shuffled["states"]:
        base = comp[row["sign"]]
        assert row["frames"] == base["frames"] and row["composition_state_sha256"] == base["state_sha256"]
        state = load_state(store, row["state_key"], row["state_sha256"])
        assert len(state.timestamps) == base["frames"]                 # the shuffled form has the composition's frame count
        assert sorted(w["word"] for w in row["words"]) == sorted(w["word"] for w in base["words"])     # every word kept
        assert [base["words"][k]["word"] for k in row["order"]] == [w["word"] for w in row["words"]]
        assert row["order"] == wf.shuffled_order(len(base["words"]), row["clip"], 0)
        assert len(row["joins"]) == len(row["composition_joins"]) == len(base["words"]) - 1
        assert all(j["gap_frames"] >= 0 and j["bound_needs_frames"] >= 0 and (j["over_bound"] is None or j["over_bound"] >= 0) for j in row["joins"])
        words_moved += row["moved"]
        if row["moved"]:
            assert row["state_sha256"] != base["state_sha256"]
    assert words_moved == shuffled["word_format"]["sentences_moved"] > 0
    for row in real["states"]:
        base = comp[row["sign"]]
        assert row["empty"] is False and row["frames"] == base["frames"]
        state = load_state(store, row["state_key"], row["state_sha256"])
        assert len(state.timestamps) == base["frames"]
        assert all(w["corpus"] == "asllvd-text-v1" and w["sign"] for w in row["words"])
        assert row["same_words"] == (len(row["words"]) == len(base["words"]))
    # the states fetched are D20's own, the word cache's and the train records: the manifest is parsed, nothing else of it is read
    records = json.loads(original_get(store, "mix.json"))["records"]
    d20 = {r["sign"] for r in json.loads(original_get(store, "sets/d20.json"))["records"]}
    touched = [r for r in records if (r.get("continuous") or {}).get("state_key") in set(fetched)]
    assert {r["corpus"] for r in touched} >= {"asllvd-text-v1"}
    for r in touched:
        assert r.get("signer_component") != "c00" and (r["partition"] == "train" or r["sign"] in d20), r["sign"]
    assert sum(r["corpus"] == "asllvd-text-v1" and r["partition"] == "train" for r in touched) >= 1
    # written once, and the seed is what the order follows
    with pytest.raises(SystemExit, match="refusing to overwrite|exists"):
        ce.main(args)


def test_another_seed_gives_another_order_from_the_same_composition(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    args, info = pilot_setup(tmp_path, store, sha, monkeypatch, seed="5")
    assert ce.main(args) == 0
    row = next(r for r in json.loads(store.get("x/arm/word-format/composition_shuffled/index.json"))["states"] if r["moved"])
    assert row["order"] == wf.shuffled_order(len(row["order"]), row["clip"], 5)


def test_the_predicted_form_keeps_its_mark_and_its_frames(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    from evaluate_text_to_motion import load_state
    store, sha, s1, oracle, argv = stage2_store
    args, info = pilot_setup(tmp_path, store, sha, monkeypatch)
    assert ce.main(args) == 0
    comp = {r["sign"]: r for r in json.loads(store.get("x/arm/index.json"))["states"]}
    assert {r["form"] for r in comp.values()} == {"core_crossfade@predicted"}
    for role in (wf.SHUFFLED, wf.ORACLE):
        index = json.loads(store.get(f"x/arm/word-format/{role}/index.json"))
        assert index["word_format"]["form"] == "core_crossfade@predicted"
        for row in index["states"]:
            assert row["form"] == f"core_crossfade/{role}@predicted" and row["frames"] == comp[row["sign"]]["frames"]
            assert len(load_state(store, row["state_key"], row["state_sha256"]).timestamps) == row["frames"]
            assert bh.composition_timing(row, role) == "predicted"
    assert any(r["frames"] != r["reference_frames"] for r in comp.values())


def test_the_export_is_refused_when_it_would_step_outside_the_pilot(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    args, info = pilot_setup(tmp_path, store, sha, monkeypatch)
    out = list(args)
    out[out.index(str(tmp_path / "arm.json"))] = str(tmp_path / "other.json")
    no_only = [a for a in out if a != "--export-only"]
    with pytest.raises(SystemExit):
        ce.main(no_only)
    with pytest.raises(SystemExit):                                    # without the export it means nothing
        k = out.index("--export-prefix")
        ce.main(out[:k] + ["--export-word-format", "--output-file", str(tmp_path / "other.json")])
    with pytest.raises(SystemExit, match="may not show"):
        ce.main(out + ["--word-format-corpora", "asl-citizen-text-v1"])
    k = out.index("--word-format-licence-note")
    with pytest.raises(SystemExit, match="confirmation"):             # ASLLVD too, before anything is composed
        ce.main(out[:k] + out[k + 2:])
    bare = tmp_path / "no-pilot.json"
    listing = dict(name="no pilot", signs=info["draw"]["signs"])
    bare.write_text(json.dumps(listing))
    moved = list(out)
    moved[moved.index(str(info["draw_path"]))] = str(bare)
    moved[moved.index(info["draw_sha"])] = hashlib.sha256(bare.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="pilot"):
        ce.main(moved)
    stored = json.loads(store.get("sets/d20.json"))
    payload = json.dumps(dict(stored, set="C20")).encode()
    store.put("sets/c20.json", payload)
    c20 = list(out)
    c20[c20.index("sets/d20.json")] = "sets/c20.json"
    c20[c20.index(info["set_sha"])] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(SystemExit, match="refuses C20"):
        ce.main(c20)
    other = dict(json.loads(store.get("sets/d20.json")), set="not-d20")
    payload = json.dumps(other).encode()
    store.put("sets/not.json", payload)
    named = list(out)
    named[named.index("sets/d20.json")] = "sets/not.json"
    named[named.index(info["set_sha"])] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(SystemExit, match="D20 only"):
        ce.main(named)
    labelled = dict(json.loads(store.get("sets/d20.json")), rule="another D20")      # labelled D20, another hash
    payload = json.dumps(labelled).encode()
    store.put("sets/other.json", payload)
    pinned = list(out)
    pinned[pinned.index("sets/d20.json")] = "sets/other.json"
    pinned[pinned.index(info["set_sha"])] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(SystemExit, match="is not D20's"):
        ce.main(pinned)
    assert not store.exists("x/arm/index.json")                        # refused before anything is written


def test_the_pilots_records_must_be_validation_clips():
    export = dict(pilot=["a"], signs=frozenset(["a"]))
    good = dict(sign="a", corpus="how2sign-two-view-text-v1", partition="val", signer_component="c03")
    sentence_set = dict(set="D20", sha256=bh.D20_SHA256)
    wf.check_export_inputs(export, [good], sentence_set)
    for change in (dict(signer_component="c00"), dict(partition="test"), dict(partition="train"), dict(corpus="asllvd-text-v1")):
        with pytest.raises(SystemExit, match="not a How2Sign val clip"):
            wf.check_export_inputs(export, [dict(good, **change)], sentence_set)
    wf.check_export_inputs(export, [good, dict(good, sign="b", partition="test")], sentence_set)    # only the pilot's records are read here
    with pytest.raises(SystemExit, match="no pilot"):
        wf.check_export_inputs(dict(export, pilot=[]), [good], sentence_set)


def test_the_round_trip_follows_the_caches_gains(monkeypatch):
    import evaluate_text_to_motion as etm
    import exposure_diagnostic as ed
    seen = {}

    def fake_generator(store, key, sha, device, *, decoder_gains=True):
        seen.update(key=key, sha=sha, decoder_gains=decoder_gains)
        sample = type("S", (), dict(tokenizer="tok", decoder_gains="gains" if decoder_gains else None))
        return None, None, sample
    monkeypatch.setattr(etm, "load_generator", fake_generator)
    monkeypatch.setattr(ed, "make_post", lambda gains: ("post", gains))
    assert wf.round_trip(None, dict(checkpoint_key="k", no_decoder_gains=True), "s" * 64, "cpu") == ("tok", ("post", None))
    assert seen == dict(key="k", sha="s" * 64, decoder_gains=False)
    assert wf.round_trip(None, dict(checkpoint_key="k"), "s" * 64, "cpu") == ("tok", ("post", "gains"))
    assert seen["decoder_gains"] is True


def test_the_generator_and_template_checks_run_before_the_prefix_is_used(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    args, info = pilot_setup(tmp_path, store, sha, monkeypatch)
    monkeypatch.setattr(wf, "check_template_record", lambda manifest: (_ for _ in ()).throw(SystemExit("template refused")))
    with pytest.raises(SystemExit, match="template refused"):
        ce.main(args)
    assert not store.exists("x/arm/index.json")
    monkeypatch.setattr(wf, "round_trip", lambda *a: (_ for _ in ()).throw(SystemExit("generator refused")))
    with pytest.raises(SystemExit, match="generator refused"):
        ce.main(args)
    assert not store.exists("x/arm/index.json") and not (tmp_path / "store" / "x").exists()


@pytest.mark.parametrize("arm,message", [("shuffled", "shuffled has"), ("oracle", "the oracle has")])
def test_a_frame_mismatch_stops_the_export_and_writes_nothing(tmp_path, stage2_store, monkeypatch, arm, message):  # noqa: F811
    from signtranslator.generation.compositional import take_frames
    store, sha, s1, oracle, argv = stage2_store
    args, info = pilot_setup(tmp_path, store, sha, monkeypatch)
    real = ce.render_round20

    def short(*a, **k):
        state, plan = real(*a, **k)
        if f"/{arm}:" in k["sample_id"]:
            state = take_frames(state, np.arange(len(state.timestamps) - 1))
        return state, plan
    monkeypatch.setattr(ce, "render_round20", short)
    with pytest.raises(SystemExit, match=message):
        ce.main(args)
    assert not store.exists("x/arm/index.json") and not list((tmp_path / "store" / "x").rglob("*.npz"))


def test_dropped_words_are_recorded_and_the_kept_ones_stretch(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    args, info = pilot_setup(tmp_path, store, sha, monkeypatch)
    real = wf.oracle_pools
    cut = {}

    def fewer(manifest, corpora, added):
        pools = real(manifest, corpora, added)
        cut["entry"] = "onion" if "onion" in pools else sorted(pools)[0]
        return {e: v for e, v in pools.items() if e != cut["entry"]}
    monkeypatch.setattr(wf, "oracle_pools", fewer)
    assert ce.main(args) == 0
    index = json.loads(store.get("x/arm/word-format/oracle/index.json"))
    comp = {r["sign"]: r for r in json.loads(store.get("x/arm/index.json"))["states"]}
    affected = [r for r in index["states"] if any(w["word"] == cut["entry"] for w in comp[r["sign"]]["words"])]
    assert affected
    for row in index["states"]:
        wanted = [w["word"] for w in comp[row["sign"]]["words"] if w["word"] == cut["entry"]]
        assert [d["word"] for d in row["dropped"]] == wanted and all(d["reason"] == "no_permitted_clip" for d in row["dropped"])
        assert row["same_words"] == (not wanted)
        assert row["empty"] or len(row["words"]) == len(comp[row["sign"]]["words"]) - len(wanted)
        if not row["empty"]:
            assert row["frames"] == comp[row["sign"]]["frames"] and (row["core_stretch"] > 1 if wanted else row["core_stretch"] == 1)
    coverage = index["word_format"]["coverage"]
    assert coverage["dropped"] == dict(no_permitted_clip=sum(len(r["dropped"]) for r in index["states"]))
    assert coverage["words_selected"] - coverage["words_kept"] == coverage["dropped"]["no_permitted_clip"]
    assert coverage["same_words"] == sum(r["same_words"] for r in index["states"])


def test_a_sentence_left_with_no_real_clip_is_marked_empty_and_leaves_the_pilot(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    args, info = pilot_setup(tmp_path, store, sha, monkeypatch)
    monkeypatch.setattr(wf, "oracle_pools", lambda manifest, corpora, added: {})
    assert ce.main(args) == 0
    index = json.loads(store.get("x/arm/word-format/oracle/index.json"))
    assert all(r["empty"] and r["state_key"] is None and r["dropped"] and r["same_words"] is False for r in index["states"])
    assert index["word_format"]["coverage"]["empty"] == len(index["states"]) and index["word_format"]["coverage"]["words_kept"] == 0


def test_sentences_with_no_selected_word_leave_the_draw_and_are_disclosed(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    import scripts.round15_compose_eval as sce
    store, sha, s1, oracle, argv = stage2_store
    set_sha, _ = put_set(store, sha, order=(0, 1, 2, 3, 4))
    monkeypatch.setattr(bh, "D20_SHA256", set_sha)
    stored = json.loads(store.get("sets/d20.json"))
    wordless = stored["records"][2]["text"]
    real = ce.stage1_selection

    def selection(texts, vocab):
        details, words = real(texts, vocab)
        return details, [[] if t == wordless else w for t, w in zip(texts, words)]
    for module in (ce, sce):
        monkeypatch.setattr(module, "stage1_selection", selection)
    monkeypatch.setattr(sys.modules["round15_compose_eval"], "stage1_selection", selection)
    assert bh.main(["how2sign-draw", "--store", f"dir:{tmp_path / 'store'}", "--mix-key", "mix.json", "--mix-sha256", sha, "--sentences-key", "sets/d20.json",
                    "--sentences-sha256", set_sha, "--main", "1", "--pilot", "3", "--min-seconds", ".5", "--seed", "1", "--name", "test-draw", "--output-dir", str(tmp_path / "draw")]) == 0
    draw = json.loads((tmp_path / "draw" / "draw.json").read_text())
    left_out = stored["records"][2]["sign"]
    assert draw["excluded_no_selected_word"] == [left_out] and left_out not in draw["signs"] and draw["counts"]["no_selected_word"] == 1
    assert draw["counts"]["eligible"] == 4 and "selected_word" in draw["rule"]
    # the unit: checked before a state is loaded, so a wordless sentence costs no fetch
    loaded = []
    rows, counts = bh.eligible_sentences([dict(sign="a", text="x", continuous=dict(seconds=3.))], lambda r: loaded.append(r), min_seconds=1., min_hand_coverage=.7, words_of=dict(a=[]))
    assert rows == [] and counts["no_selected_word"] == 1 and not loaded


def test_the_whole_pilot_through_the_packet_two_stages_and_the_scorer(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    args, info = pilot_setup(tmp_path, store, sha, monkeypatch)
    assert ce.main(args) == 0
    pin = lambda key: f"{key}@{hashlib.sha256(store.get(key)).hexdigest()}"   # noqa: E731
    systems = [f"mixed=composition:{pin('x/arm/word-format/composition_shuffled/index.json')}",
               f"real=composition:{pin('x/arm/word-format/oracle/index.json')}", f"arm=composition:{pin('x/arm/index.json')}", "truth=reference"]
    sheet_path = tmp_path / "checked.csv"
    sheet_path.write_text(",".join(NEGATION_COLUMNS) + "\n")
    packet = ["how2sign-packet", "--store", info["root"], "--draw", str(info["draw_path"]), "--draw-sha256", info["draw_sha"], "--negation-sheet", str(sheet_path),
              "--negation-sheet-sha256", hashlib.sha256(sheet_path.read_bytes()).hexdigest(), "--panel", "96", "--seed", "5", "--renderer", "skeleton"]
    for s in systems:
        packet += ["--system", s]
    with pytest.raises(SystemExit, match="pilot only"):
        bh.main(packet + ["--set", "main", "--output-dir", str(tmp_path / "m")])
    out = tmp_path / "packet"
    assert bh.main(packet + ["--set", "pilot", "--output-dir", str(out)]) == 0
    key = json.loads((out / "key" / "key.json").read_text())
    assert key.pop("key_sha256")
    assert key["systems"] == ["mixed", "real", "arm", "truth"] and key["groups"] == 4 and key["report_only_systems"] == ["mixed", "real"]
    assert key["word_format"]["roles"]["composition_shuffled"] == "mixed" and key["word_format"]["report_only"] is True
    assert key["utterances"] == 4 and set(key["video_bytes_per_frame"]) == set(key["systems"])
    assert {i["sign"] for i in key["items"]} == set(info["draw"]["pilot"])
    by_cell = {(i["utterance"], i["system"]): i for i in key["items"]}
    moved_frames = 0
    for i in key["items"]:
        assert i["frames"] == i["seconds"] * 24 or abs(i["frames"] - i["seconds"] * 24) < 1
        if i["system"] != "truth":                                          # the reference plays at its recorded length, the others at the predicted one
            assert i["frames"] == by_cell[(i["utterance"], "arm")]["frames"]
        if i["system"] == "mixed":
            moved_frames += i["state"]["moved"]
            assert "order" in i["state"]
        if i["system"] == "real":
            assert "same_words" in i["state"] and "dropped" in i["state"]
    assert moved_frames > 0
    # blinding: nothing a rater receives names an arm, or the question
    for g in range(4):
        folder = out / "packet" / f"group-{g}"
        text = (folder / "sheet.csv").read_text()
        for word in ("mixed", "real", "composition", "shuffled", "oracle", "moved", "order", "truth"):
            assert word not in text
        assert not any(p.suffix == ".mp4" for p in folder.iterdir())                 # the videos sit in the zip alone once it is read back
        with zipfile.ZipFile(out / "packet" / f"group-{g}.zip") as z:
            assert sorted(n.split("/")[1] for n in z.namelist() if n.endswith(".mp4")) == sorted(f"{i['item_id']}.mp4" for i in key["items"] if i["group"] == g)
            assert {i.date_time for i in z.infolist()} == {(1980, 1, 1, 0, 0, 0)} and not any("key" in n for n in z.namelist())
            for name in z.namelist():
                if name.endswith((".csv", ".md")):
                    lowered = z.read(name).decode().lower()
                    assert not re.search(r"shuffl|oracle|word order|composition", lowered)
    assert not any("DEBRIEF" in p.name for p in (out / "packet").rglob("*"))
    # two raters per group, all four groups, through both stages
    items = {i["item_id"]: i for i in key["items"]}
    roster = {f"{g}{k}".ljust(64, "e"): g for g in range(4) for k in ("a", "b")}
    roster_path = tmp_path / "roster.json"
    roster_path.write_text(json.dumps(roster))
    receipts = tmp_path / "receipts.json"
    key_path = out / "key" / "key.json"
    firsts, seconds = [], []
    for reviewer, g in roster.items():
        filled = list(csv.DictReader(io.StringIO(write_stage1_sheet(key, g))))
        for r in filled:
            r.update(viewable="yes", translation=items[r["item_id"]]["english"], sentence_type="statement", negated="no", topic="no", naturalness="3", reviewer_sha256=reviewer)
        buffer = io.StringIO()
        w = csv.DictWriter(buffer, fieldnames=STAGE1_COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(filled)
        path = tmp_path / f"stage1-{reviewer[:2]}.csv"
        path.write_text(buffer.getvalue())
        assert bh.main(["how2sign-stage2", "--key", str(key_path), "--roster", str(roster_path), "--stage1-sheet", str(path), "--receipts", str(receipts), "--output-dir", str(tmp_path / "stage2")]) == 0
        firsts += ["--sheet", str(path)]
        back = list(csv.DictReader(io.StringIO((tmp_path / "stage2" / f"stage2-{reviewer[:12]}.csv").read_text(encoding="utf-8-sig"))))
        for r in back:
            item = items[r["item_id"]]
            r["choice"] = item["correct"] if item["system"] != "mixed" else next(o for o in "abcd" if o != item["correct"])
        returned = tmp_path / f"stage2-back-{reviewer[:2]}.csv"
        buffer = io.StringIO()
        w = csv.DictWriter(buffer, fieldnames=STAGE2_COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(back)
        returned.write_text(buffer.getvalue())
        seconds += ["--stage2-sheet", str(returned)]
    common = ["--key", str(key_path), *firsts, *seconds, "--reference", "truth", "--stage1-receipts", str(receipts), "--roster", str(roster_path)]
    assert sh.main(common + ["--output", str(tmp_path / "report.json")]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["report_only_systems"] == ["mixed", "real"] and report["word_format"]["counts_towards_release_bar"] is False
    read = report["word_format"]
    assert read["raters"] == 8 and read["arms"]["composition_shuffled"]["rate"] == 0 and read["arms"]["composition"]["rate"] == 1
    assert read["contrasts"]["shuffled_minus_composition"]["all"]["comprehension_choice"]["utterances"] == 4
    assert read["contrasts"]["shuffled_minus_composition"]["all"]["comprehension_choice"]["mean_difference"] == -1
    assert report["paired_with_reference"]["real"]["utterances"] == 4 and report["paired_with_reference"]["mixed"]["utterances"] == 4
    assert report["agreement"]["choice"]["raters"] == 2             # raters of one group rate the same videos
    # the bar read refuses this key, says why, and exits nonzero
    assert sh.main(common + ["--bar", "--bootstrap", "5", "--output", str(tmp_path / "bar.json")]) == 2
    assert "report-only pilot arms" in json.loads((tmp_path / "bar.json").read_text())["bar"]["refused"]
    # a pilot rater cannot take a main sheet: the key's excluded list, checked as the negation checker is
    guarded = json.loads(key_path.read_text())
    guarded.pop("key_sha256")
    guarded["excluded_raters"] = [next(iter(roster))]
    guarded["key_sha256"] = bh.key_sha256(guarded)
    guarded_path = tmp_path / "guarded.json"
    guarded_path.write_text(json.dumps(guarded))
    with pytest.raises(SystemExit, match="excluded list"):
        bh.main(["how2sign-stage2", "--key", str(guarded_path), "--roster", str(roster_path), "--stage1-sheet", str(tmp_path / f"stage1-{next(iter(roster))[:2]}.csv"),
                 "--receipts", str(tmp_path / "other-receipts.json"), "--output-dir", str(tmp_path / "stage2b")])
    assert json.loads(key_path.read_text())["excluded_raters"] == []
    pilot_roster = tmp_path / "pilot-roster.json"
    pilot_roster.write_text(json.dumps({A: 0}))
    again = bh.main(packet + ["--set", "pilot", "--excluded-raters-roster", str(pilot_roster), "--output-dir", str(tmp_path / "packet2")])
    assert again == 0 and json.loads((tmp_path / "packet2" / "key" / "key.json").read_text())["excluded_raters"] == [A]
