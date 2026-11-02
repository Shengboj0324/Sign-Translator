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
                                                       STAGE1_COLUMNS, STAGE2_COLUMNS, bar_read, design_study, score_study,
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


# ---- the design -------------------------------------------------------------------------------------------------------------

def test_the_arms_fit_the_latin_square_without_adding_load():
    key = study(20)
    assert key["groups"] == 4
    cells = {(i["utterance"], i["system"]) for i in key["items"]}
    assert len(key["items"]) == len(cells) == 80                    # every (sentence, system) cell, once
    for g in range(4):
        mine = [i for i in key["items"] if i["group"] == g]
        assert len(mine) == 20                                      # a rater's load is the sentences, whatever the systems
        assert len({i["utterance"] for i in mine}) == 20            # no rater sees a sentence twice
        assert {s: sum(i["system"] == s for i in mine) for s in SYSTEMS} == {s: 5 for s in SYSTEMS}
        assert sorted(i["position"] for i in mine) == list(range(20))
    options = {}
    for i in key["items"]:
        assert options.setdefault(i["utterance"], i["options"]) == i["options"]   # systems differ only in the video


def test_two_raters_in_opposite_groups_pair_shuffled_with_composition_in_both_directions():
    key = study(20)
    assert word_format_pairing(SYSTEMS, ROLES)["paired_by_sentence"] == [["composition_shuffled", "composition"], ["oracle", "reference"]]
    seen = {g: {i["utterance"]: i["system"] for i in key["items"] if i["group"] == g} for g in range(4)}
    for g, h in ((0, 2), (1, 3)):
        pairs = [(seen[g][u], seen[h][u]) for u in range(20)]
        assert sorted(set(pairs)) == sorted({(SYSTEMS[k], SYSTEMS[(k + 2) % 4]) for k in range(4)})
        assert all(pairs.count(p) == 5 for p in set(pairs))         # 5 sentences each way: 10 for each contrast
        assert not any({"composition", "oracle"} == set(p) for p in pairs)   # that contrast needs other raters


def test_only_four_systems_with_shuffled_opposite_composition_are_accepted():
    with pytest.raises(ValueError, match="two places"):
        word_format_pairing(["composition_shuffled", "composition", "oracle", "reference"], ROLES)
    with pytest.raises(ValueError, match="exactly the four"):
        word_format_pairing(SYSTEMS + ["arm-b"], dict(ROLES, other="arm-b"))
    with pytest.raises(ValueError, match="exactly the four"):
        word_format_pairing(SYSTEMS[:3], {k: ROLES[k] for k in ("composition", "composition_shuffled", "oracle")})


def test_a_pilot_of_two_raters_cannot_measure_a_large_drop():
    """The protocol's numbers: on the logit model of the release-bar simulations (sentence SD 1, rater SD .5), a true 20-point
    drop from 60%, 10 paired sentences. The interval stays near +/- 45 points, and most such pilots do not detect it."""
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    rng = np.random.default_rng(0)
    differences, hits = [], 0
    flips = rng.choice([-1., 1.], size=(400, 10))
    for _ in range(600):
        sentence = rng.normal(0, 1., 10)
        rater = rng.normal(0, .5, 2)
        one = rng.random(10) < 1 / (1 + np.exp(-(logit(.6) + sentence + rater[0])))
        two = rng.random(10) < 1 / (1 + np.exp(-(logit(.4) + sentence + rater[1])))
        d = one.astype(float) - two.astype(float)
        differences.append(d.mean())
        hits += np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-9) < .05
    assert 1.96 * np.std(differences) > .35
    assert hits / 600 < .2


# ---- blinding ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["docs/HUMAN_EVALUATION_REVIEWER_GUIDE_TWO_STAGE.md", "docs/HUMAN_EVALUATION_REVIEWER_GUIDE.md",
                                  "reviewer_package/FOR_REVIEWERS.md", "reviewer_package/RATING_SHEET.md",
                                  "reviewer_package/CONSENT_FORM.md"])
def test_what_raters_read_reveals_neither_the_arm_nor_the_question(name):
    text = (ROOT / name).read_text().lower()
    for word in ("shuffl", "scrambl", "permut", "word order", "word-order", "english order", "order of the words", "oracle",
                 "composition", "stitch", "word-for-word", "word by word", "real clip", "generated", "control"):
        assert word not in text, word


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
    row = word_format_read(answers, key)["contrasts"]["shuffled_minus_composition"]["all"]["comprehension_choice"]
    mine, other = per_sentence(key, answers, "composition_shuffled"), per_sentence(key, answers, "composition")
    common = sorted(set(mine) & set(other))
    assert row["utterances"] == len(common) == 10
    assert row["mean_difference"] == pytest.approx(sum(mine[u] - other[u] for u in common) / 10, abs=1e-4)
    assert row["p_value"] == pytest.approx(paired_permutation_pvalue([mine[u] for u in common], [other[u] for u in common], seed=0), abs=1e-4)
    assert row["ci95"][0] <= row["mean_difference"] <= row["ci95"][1]
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
    assert pair["all"]["comprehension_choice"] == dict(utterances=0, mean_difference=None, p_value=None, ci95=None)
    assert pair["all"]["comprehension_choice_method_d"]["cluster95"] is None
    assert pair["within_rater"]["per_rater"] == {A[:12]: -1., B[:12]: -1.} and pair["within_rater"]["mean"] == -1.
    assert "different sentences" in pair["within_rater"]["note"]


def test_two_raters_give_an_honest_interval():
    key = study(20)
    answers = sheet(key, (0, 2), composition_shuffled=lambda i: i["utterance"] % 8 in (0, 2))
    read = word_format_read(answers, key)["contrasts"]["shuffled_minus_composition"]["all"]
    row = read["comprehension_choice_method_d"]
    assert row["raters"] == 2 and row["cluster95"][1] - row["cluster95"][0] > .5 and read["comprehension_choice_by_the_rule"] == "unresolved"


def test_the_rule_needs_the_interval_and_the_permutation_in_one_direction():
    key = study(40)
    drop = word_format_read(sheet(key, (0, 1, 2, 3), composition_shuffled=lambda i: i["utterance"] % 10 == 0), key)
    assert drop["contrasts"]["shuffled_minus_composition"]["all"]["comprehension_choice_by_the_rule"] == "below"
    rise = word_format_read(sheet(key, (0, 1, 2, 3), composition=lambda i: i["utterance"] % 10 == 0), key)
    assert rise["contrasts"]["shuffled_minus_composition"]["all"]["comprehension_choice_by_the_rule"] == "above"
    same = word_format_read(sheet(key, (0, 1, 2, 3)), key)
    assert same["contrasts"]["shuffled_minus_composition"]["all"]["comprehension_choice_by_the_rule"] == "unresolved"


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


def test_the_share_is_set_beside_the_instruments_with_no_cut():
    key = study(40)
    good = word_format_read(sheet(key, (0, 1, 2, 3), composition_shuffled=lambda i: i["utterance"] % 4 != 0, reference=True), key)
    share = good["share_of_lead_over_chance_removed_by_shuffling"]
    assert share["instruments_blank_and_bleu"] == INSTRUMENT_SHUFFLE_SHARES and share["chance"] == .25
    assert share["share"] == pytest.approx(.25 / .75, abs=1e-3)
    assert share["ci95"][0] < share["share"] < share["ci95"][1]
    assert share["against_instruments"] == "around, or too wide to tell"
    none = word_format_read(sheet(key, (0, 1, 2, 3)), key)["share_of_lead_over_chance_removed_by_shuffling"]
    assert none["share"] == 0. and none["against_instruments"] == "below"
    wiped = word_format_read(sheet(key, (0, 1, 2, 3), composition_shuffled=False), key)["share_of_lead_over_chance_removed_by_shuffling"]
    assert wiped["share"] == pytest.approx(1 / .75, abs=1e-3) and wiped["against_instruments"] == "above"
    floor = word_format_read(sheet(key, (0, 1, 2, 3), composition=False), key)["share_of_lead_over_chance_removed_by_shuffling"]
    assert floor["share"] is None and floor["against_instruments"] == "not read"
    few = word_format_read(sheet(study(20, unmoved=range(20)), (0, 2)), study(20, unmoved=range(20)))
    assert few["share_of_lead_over_chance_removed_by_shuffling"]["utterances"] == 0


def test_the_read_says_it_is_report_only_and_counts_unviewable_rows_out():
    key = study(20)
    answers = sheet(key, (0, 2))
    answers[0]["viewable"] = "no"
    read = word_format_read(answers, key)
    assert read["report_only"] is True and read["counts_towards_release_bar"] is False and read["ratings"] == 39
    assert "small drop in choice" in read["reading"] and read["arms"]["reference"]["label"] == "reference"
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


# ---- what the oracle may show -------------------------------------------------------------------------------------------------

def test_only_permitted_corpora_reach_raters():
    wf.check_oracle_corpora(("asllvd-text-v1",), None)
    wf.check_oracle_corpora(("asllvd-text-v1", "rit-text-r6"), "owner confirmed RIT's terms, 2026-11-05")
    for corpus in ("asl-citizen-text-v1", "asl-citizen", "stem-wiki-text-v1", "asl-stem-wiki", "youtube-asl-v1", "semlex-text-v1", "how2sign-two-view-text-v1"):
        with pytest.raises(SystemExit, match="may not show"):
            wf.check_oracle_corpora(("asllvd-text-v1", corpus), "a note does not help")
    for corpus in ("rit-text-r6", "wlasl-text-r6"):
        with pytest.raises(SystemExit, match="confirmation"):
            wf.check_oracle_corpora((corpus,), "  ")
    with pytest.raises(SystemExit, match="needs a corpus"):
        wf.check_oracle_corpora((), None)


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

def built(frames=48, words=("cut", "onion"), shuffled_words=None, oracle_words=None, role_arm="arm", empty=False):
    rows = {s: dict(sign=s, frames=frames, reference_frames=frames, empty=False, state_sha256=f"sha-{s}", form="core_crossfade",
                    words=[dict(word=w) for w in words]) for s in ("p1", "p2", "p3", "p4")}
    link = dict(arm=role_arm, form="core_crossfade")
    shuffled = {s: dict(r, composition_state_sha256=r["state_sha256"], form="core_crossfade/composition_shuffled",
                        words=[dict(word=w) for w in (shuffled_words or tuple(reversed(words)))]) for s, r in rows.items()}
    oracle = {s: dict(r, composition_state_sha256=r["state_sha256"], form="core_crossfade/oracle", same_words=False,
                      words=[dict(word=w, corpus="asllvd-text-v1") for w in (oracle_words or words[:1])]) for s, r in rows.items()}
    if empty:
        oracle["p2"] = dict(oracle["p2"], empty=True, words=[])
    return {"gen": dict(arm="arm", rows=rows, form="core_crossfade", word_format=None),
            "mixed": dict(arm="arm/composition_shuffled", rows=shuffled, form="core_crossfade/composition_shuffled",
                          word_format=dict(role=wf.SHUFFLED, composition=link, form="core_crossfade", seed=0)),
            "real": dict(arm="arm/oracle", rows=oracle, form="core_crossfade/oracle",
                         word_format=dict(role=wf.ORACLE, composition=link, form="core_crossfade", licence_note=None))}


PACKET = [dict(label="mixed", kind="composition"), dict(label="real", kind="composition"), dict(label="gen", kind="composition"),
          dict(label="truth", kind="reference")]
SIGNS = ["p1", "p2", "p3", "p4"]


def test_the_packet_takes_the_arms_in_the_pilot_only():
    made = bh.check_word_format(PACKET, built(), which="pilot", signs=SIGNS)
    assert made["roles"] == dict(composition="gen", composition_shuffled="mixed", oracle="real", reference="truth")
    assert made["signs"] == SIGNS and made["dropped_no_real_clip"] == [] and made["pairing"]["paired_by_sentence"][0] == ["mixed", "gen"]
    with pytest.raises(SystemExit, match="pilot only"):
        bh.check_word_format(PACKET, built(), which="main", signs=SIGNS)
    plain = {"gen": dict(arm="arm", rows={}, form="core_crossfade", word_format=None)}
    assert bh.check_word_format(PACKET[2:], plain, which="main", signs=SIGNS) is None     # the main read is untouched


def test_the_packet_refuses_arms_that_would_show():
    with pytest.raises(SystemExit, match="two places"):
        bh.check_word_format([PACKET[0], PACKET[2], PACKET[1], PACKET[3]], built(), which="pilot", signs=SIGNS)
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
    arms["real"]["rows"]["p3"]["words"][0]["corpus"] = "wlasl-text-r6"
    with pytest.raises(SystemExit, match="confirmation"):
        bh.check_word_format(PACKET, arms, which="pilot", signs=SIGNS)
    arms = built()
    arms["real"]["rows"]["p3"]["words"][0]["corpus"] = "asl-citizen-text-v1"
    with pytest.raises(SystemExit, match="may not show"):
        bh.check_word_format(PACKET, arms, which="pilot", signs=SIGNS)


def test_a_sentence_the_oracle_cannot_sign_leaves_the_pilot_and_is_listed():
    made = bh.check_word_format(PACKET, built(empty=True), which="pilot", signs=SIGNS)
    assert made["dropped_no_real_clip"] == ["p2"] and made["signs"] == ["p1", "p3", "p4"]


# ---- the export and the whole pilot, on the synthetic store --------------------------------------------------------------------

def pilot_setup(tmp_path, store, sha, monkeypatch, *, seed="0", extra=()):
    """A D20-like set of five, a word cache, a draw of 1 + 4, and the arm's export command."""
    import torch
    import validate_sign_recognizer as vsr
    from scripts.round20_word_cache import main as cache_main
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
                    "--sentences-sha256", set_sha, "--main", "1", "--pilot", "4", "--min-seconds", ".5", "--seed", "11", "--output-dir", str(draw_dir)]) == 0
    draw_path = draw_dir / "draw.json"
    draw_sha = hashlib.sha256(draw_path.read_bytes()).hexdigest()
    args = [a for a in run_args(tmp_path, sha) if a not in ("--checkpoint-key", "prior.pt")]
    args[args.index("core_crossfade,oracle_stage2")] = "core_crossfade"
    args += ["--clips", "5", "--selection", "stage1", "--timing", "sign-fitted", "--sentences-key", "sets/d20.json", "--sentences-sha256", set_sha, "--word-cache-key", "cache/w/index.json",
             "--word-cache-sha256", cache_sha, "--word-cache-n", "2", "--export-prefix", "x/arm", "--export-arm", "arm", "--export-sentences-file", str(draw_path),
             "--export-sentences-sha256", draw_sha, "--export-only", "--export-word-format", "--word-format-seed", seed, "--output-file", str(tmp_path / "arm.json"), *extra]
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
    assert real["word_format"]["composition"]["sha256"] == hashlib.sha256(store.get("x/arm/index.json")).hexdigest()
    assert real["word_format"]["round_trip"]["checkpoint_key"] == "prior.pt"
    assert decodes and set(decodes) <= {w["word"] for r in plain["states"] for w in r["words"]}
    assert len(decodes) == len(set(decodes))                           # one real clip per entry, as a word cache holds one sign
    words_moved = 0
    for row in shuffled["states"]:
        base = comp[row["sign"]]
        assert row["frames"] == base["frames"] == row["reference_frames"] and row["composition_state_sha256"] == base["state_sha256"]
        state = load_state(store, row["state_key"], row["state_sha256"])
        assert len(state.timestamps) == base["frames"]                 # the shuffled form has the composition's frame count
        assert sorted(w["word"] for w in row["words"]) == sorted(w["word"] for w in base["words"])     # every word kept
        assert [base["words"][k]["word"] for k in row["order"]] == [w["word"] for w in row["words"]]
        assert row["order"] == wf.shuffled_order(len(base["words"]), row["clip"], 0)
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
    from tests.test_round20_predicted import PIN
    store, sha, s1, oracle, argv = stage2_store
    from signtranslator.generation import composition_v2 as cv2
    from tests.test_round20_predicted import MODEL
    monkeypatch.setattr(cv2, "fit_duration_model", lambda manifest: MODEL)   # the fixture's mix is too small to fit one
    args, info = pilot_setup(tmp_path, store, sha, monkeypatch, extra=("--predicted-duration", f"--expect-duration-model={PIN}", "--export-forms", "core_crossfade@predicted"))
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
    with pytest.raises(SystemExit, match="confirmation"):
        ce.main(out + ["--word-format-corpora", "asllvd-text-v1,wlasl-text-r6"])
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
    assert not store.exists("x/arm/index.json")                        # refused before anything is written


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
              "--negation-sheet-sha256", hashlib.sha256(sheet_path.read_bytes()).hexdigest(), "--panel", "96", "--seed", "5"]
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
        assert i["frames"] == by_cell[(i["utterance"], "arm")]["frames"]    # every arm of a sentence has one length
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
        names = [p.name for p in folder.iterdir()]
        assert sorted(n for n in names if n.endswith(".mp4")) == sorted(f"{i['item_id']}.mp4" for i in key["items"] if i["group"] == g)
        with zipfile.ZipFile(out / "packet" / f"group-{g}.zip") as z:
            assert {i.date_time for i in z.infolist()} == {(1980, 1, 1, 0, 0, 0)} and not any("key" in n for n in z.namelist())
            for name in z.namelist():
                if name.endswith((".csv", ".md")):
                    lowered = z.read(name).decode().lower()
                    assert not re.search(r"shuffl|oracle|word order|composition", lowered)
    # two raters, in opposite groups, through both stages
    items = {i["item_id"]: i for i in key["items"]}
    roster = {A: 0, B: 2}
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
        path = tmp_path / f"stage1-{g}.csv"
        path.write_text(buffer.getvalue())
        assert bh.main(["how2sign-stage2", "--key", str(key_path), "--roster", str(roster_path), "--stage1-sheet", str(path), "--receipts", str(receipts), "--output-dir", str(tmp_path / "stage2")]) == 0
        firsts += ["--sheet", str(path)]
        back = list(csv.DictReader(io.StringIO((tmp_path / "stage2" / f"stage2-{reviewer[:12]}.csv").read_text())))
        for r in back:
            item = items[r["item_id"]]
            r["choice"] = item["correct"] if item["system"] != "mixed" else next(o for o in "abcd" if o != item["correct"])
        returned = tmp_path / f"stage2-back-{g}.csv"
        buffer = io.StringIO()
        w = csv.DictWriter(buffer, fieldnames=STAGE2_COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(back)
        returned.write_text(buffer.getvalue())
        seconds += ["--stage2-sheet", str(returned)]
    common = ["--key", str(key_path), *firsts, *seconds, "--reference", "truth", "--stage1-receipts", str(receipts)]
    assert sh.main(common + ["--output", str(tmp_path / "report.json")]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["report_only_systems"] == ["mixed", "real"] and report["word_format"]["counts_towards_release_bar"] is False
    read = report["word_format"]
    assert read["raters"] == 2 and read["arms"]["composition_shuffled"]["rate"] == 0 and read["arms"]["composition"]["rate"] == 1
    assert read["contrasts"]["shuffled_minus_composition"]["all"]["comprehension_choice"]["utterances"] == 2
    assert read["contrasts"]["shuffled_minus_composition"]["all"]["comprehension_choice"]["mean_difference"] == -1
    assert report["paired_with_reference"]["real"]["utterances"] == 2 and "mixed" not in report["paired_with_reference"]   # opposite arms pair
    # the bar read refuses this key, says why, and exits nonzero
    assert sh.main(common + ["--bar", "--roster", str(roster_path), "--bootstrap", "5", "--output", str(tmp_path / "bar.json")]) == 2
    assert "report-only pilot arms" in json.loads((tmp_path / "bar.json").read_text())["bar"]["refused"]
