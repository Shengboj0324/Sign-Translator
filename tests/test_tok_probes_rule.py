"""The declared rule (docs/NEXT_PHASE_PLAN.md, "Tokenizer probes and seam ablation declared") on synthetic reports: every outcome and each boundary. No live report and no
store is read: the script takes a local file only."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tok_probes_rule as Rule  # noqa: E402
from tests.test_tok_probes import CLIPS, harness, merged, read, tiny_tokenizer  # noqa: E402

PLACE = dict(python="3.12", gpu="L40S")


def metric(p=.5, d_low=.1, gain=(.1, 1.), rest_low=.1):
    return dict(d=1., p=p, p_ci95=[0., 1.], identity_minus_round_trip=dict(ci95=[d_low, 1.]), no_quant_minus_round_trip=dict(ci95=list(gain)), identity_minus_no_quant=dict(ci95=[rest_low, 1.]))


AUTOENCODER = dict(p=.1, gain=(-.1, .5))
QUANTISER = dict(p=.9, gain=(.1, 1.))


def share_form(share=.9, low=.86, high=.97, likelihood=.9, loop_low=-.1, loop_point=.3, **seam):
    return dict(share_of_gap=dict(bleu=share, bleu_ci95=[low, high], likelihood=likelihood), loop_share=dict(mean=loop_point), paired=dict(real=dict(loop_share=dict(ci95=[loop_low, .2]))),
                comparators=dict(seam_cmin45=dict(bleu=dict(difference=-.2, ci95=[-.5, .1]))), seam=dict(dict(bridged_frame_share=.45, frames_per_bridge=8., filled_cuts=2), **seam))


def report(bleu=None, likelihood=None, masked=None, masked_likelihood=None, primary=None, cmax=.9, crnd=.9, **top):
    out = dict(report_only=True, declared_read=True, code_digest="d", environment=PLACE, sensitivity_only=["seam_min45"],
               checks=dict(identity_masked_identical_to_real=1., blank_identical_to_real=.1, canary=dict(reproduced=True, misses=[], code_digest="d", environment=PLACE)),
               b1=dict(unmasked=dict(bleu=metric(**(bleu or {})), likelihood_minus_blank=metric(**(likelihood or {}))), masked=dict(bleu=metric(**(masked or {})), likelihood_minus_blank=metric(**(masked_likelihood or {})))),
               forms=dict(seam_cval45=share_form(**(primary or {})), seam_cmax45=share_form(share=cmax), seam_crnd45=share_form(share=crnd), seam_cmin45=share_form(share=.95),
                          seam_csq196_min=share_form(share=.5), seam_min45=share_form(share=.97)))
    out.update(top)
    return out


# ---- void, all reads -----------------------------------------------------------------------------------------------------------

def test_a_clean_report_is_not_void_and_says_the_plan_decides():
    r = Rule.read_rule(report())
    assert r["void"] is False and r["reasons"] == [] and r["status"] == "declared in plan; the plan's text decides" and r["code_digest"] == "d"


@pytest.mark.parametrize("change,reason", [(dict(code_digest=None), "no code digest"), (dict(environment=None), "no environment"), (dict(report_only=False), "check nonzero"),
                                           (dict(code_digest="other"), "more than one digest"), (dict(environment=dict(PLACE, gpu="A5000")), "more than one digest"),
                                           (dict(checks=dict(identity_masked_identical_to_real=.9, blank_identical_to_real=.1, canary=dict(reproduced=True, misses=[], code_digest="d", environment=PLACE))), "check nonzero"),
                                           (dict(checks=dict(identity_masked_identical_to_real=1., blank_identical_to_real=1., canary=dict(reproduced=True, misses=[], code_digest="d", environment=PLACE))), "check nonzero"),
                                           (dict(checks=dict(identity_masked_identical_to_real=1., blank_identical_to_real=.1, canary=dict(reproduced=False, misses=["real NLL"], code_digest="d", environment=PLACE))), "canary not reproduced"),
                                           (dict(checks=dict(identity_masked_identical_to_real=1., blank_identical_to_real=.1, canary=None)), "canary not reproduced"), (dict(checks=None), "check nonzero")])
def test_each_void_condition_voids_both_reads(change, reason):
    r = Rule.read_rule(report(**change))
    assert r["void"] is True and any(reason in x for x in r["reasons"]) and r["b1"]["outcome"] == "void" and r["c0"]["outcome"] == "void" and "reported" not in r


@pytest.mark.parametrize("value,void", [(.39, True), (.40, False), (.45, False), (.47, False), (.48, True), (None, True)])
def test_bridged_frame_share_is_bounded(value, void):
    r = Rule.read_rule(report(primary=dict(bridged_frame_share=value)))
    assert r["void"] is void and (not void or "bridged_frame_share" in r["reasons"][0])


@pytest.mark.parametrize("value,void", [(6.9, True), (7., False), (8., False), (9., False), (9.1, True), (None, True)])
def test_frames_per_bridge_is_bounded(value, void):
    r = Rule.read_rule(report(primary=dict(frames_per_bridge=value)))
    assert r["void"] is void and (not void or "frames_per_bridge" in r["reasons"][0])


def test_a_smoke_is_refused():
    with pytest.raises(SystemExit, match="smoke"):
        Rule.read_rule(report(declared_read=False))


# ---- B1 ---------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw,expected", [(dict(p=.25, gain=(-.1, 0.)), "autoencoder"), (dict(p=-.2, gain=(-.5, .1)), "autoencoder"), (dict(p=.75, gain=(.01, 1.)), "quantiser"), (dict(p=1.2, gain=(.5, 2.)), "quantiser"),
                                         (dict(p=.5), "unresolved"), (dict(p=.26, gain=(-.1, .5)), "unresolved"), (dict(p=.74, gain=(.1, 1.)), "unresolved"),
                                         (dict(p=.1, rest_low=0., gain=(-.1, .5)), "unresolved"), (dict(p=.9, gain=(0., 1.)), "unresolved"), (dict(d_low=0.), "void"), (dict(d_low=-.1, p=.1), "void"), (dict(p=None), "void")])
def test_each_b1_outcome_on_corpus_bleu(kw, expected):
    assert Rule.read_rule(report(bleu=kw))["b1"]["outcome"] == expected


def test_no_quant_below_the_round_trip_is_unresolved_and_says_so():
    b1 = Rule.read_rule(report(bleu=dict(p=-.3, gain=(-.5, -.01))))["b1"]
    assert b1["outcome"] == "unresolved" and b1["primary"]["below_round_trip"] is True
    assert Rule.read_rule(report(bleu=dict(p=.1, gain=(-.5, 0.))))["b1"]["outcome"] == "autoencoder"          # upper bound exactly 0 still holds


@pytest.mark.parametrize("primary,opposite,agree", [(AUTOENCODER, QUANTISER, "autoencoder"), (QUANTISER, AUTOENCODER, "quantiser")])
@pytest.mark.parametrize("read", ["likelihood", "masked"])
def test_an_agreement_read_on_the_opposite_branch_makes_it_unresolved(primary, opposite, agree, read):
    kw = {"bleu": primary, read: opposite}
    b1 = Rule.read_rule(report(**kw))["b1"]
    assert b1["outcome"] == "unresolved" and b1["downgraded_by"] == [read] and b1["primary"]["branch"] == agree
    same = {"bleu": primary, "likelihood": primary, "masked": primary}
    assert Rule.read_rule(report(**same))["b1"]["outcome"] == agree


@pytest.mark.parametrize("branch", [AUTOENCODER, QUANTISER])
@pytest.mark.parametrize("read", ["likelihood", "masked"])
def test_agreement_reads_cannot_create_a_branch(branch, read):
    assert Rule.read_rule(report(bleu=dict(p=.5), **{read: branch}))["b1"]["outcome"] == "unresolved"
    assert Rule.read_rule(report(bleu=dict(d_low=0.), **{read: branch}))["b1"]["outcome"] == "void"


@pytest.mark.parametrize("read", ["likelihood", "masked"])
@pytest.mark.parametrize("neutral", [dict(p=.5), dict(d_low=0.)])
def test_an_unresolved_or_void_agreement_read_does_not_downgrade(read, neutral):
    b1 = Rule.read_rule(report(bleu=AUTOENCODER, **{read: neutral}))["b1"]
    assert b1["outcome"] == "autoencoder" and b1["downgraded_by"] == []


def test_the_masked_forms_likelihood_is_shown_and_changes_nothing():
    b1 = Rule.read_rule(report(bleu=AUTOENCODER, masked_likelihood=QUANTISER))["b1"]
    assert b1["outcome"] == "autoencoder" and b1["shown_only"]["masked_likelihood"]["branch"] == "quantiser"


def test_b1_is_void_without_its_forms():
    r = report()
    r["b1"]["unmasked"] = None
    assert Rule.read_rule(r)["b1"]["outcome"] == "void"


@pytest.mark.parametrize("kw,text", [(AUTOENCODER, "V3-D is eligible for section 9's 200-clip test, nothing more"), (QUANTISER, "no training by itself"), (dict(p=.5), "no tokenizer training on B1's evidence")])
def test_each_b1_outcome_carries_the_plans_action(kw, text):
    assert Rule.read_rule(report(bleu=kw))["b1"]["action"] == text


# ---- C0 --------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw,expected", [(dict(share=.7, low=.6, high=.84), "fall_shown"), (dict(share=.8, low=.7, high=.8499), "fall_shown"), (dict(share=.9, low=.85, high=.97), "seams_ruled_out"),
                                         (dict(share=.95, low=.9, high=.99), "seams_ruled_out"), (dict(share=.8, low=.7, high=.85), "unresolved"), (dict(share=.86, low=.8499, high=.93), "unresolved"),
                                         (dict(share=.84, low=.7, high=.9), "unresolved")])
def test_each_c0_outcome_on_its_interval(kw, expected):
    assert Rule.read_rule(report(primary=kw))["c0"]["outcome"] == expected


@pytest.mark.parametrize("over,condition", [(dict(cmax=.8499), "seam_cmax45"), (dict(crnd=.8499), "seam_crnd45"), (dict(primary=dict(likelihood=.8499)), "likelihood"), (dict(cmax=None), "seam_cmax45"), (dict(primary=dict(likelihood=None)), "likelihood")])
def test_seams_are_not_ruled_out_unless_every_condition_holds(over, condition):
    c0 = Rule.read_rule(report(**over))["c0"]
    assert c0["outcome"] == "unresolved" and c0["conditions"][condition] is False and sum(not v for v in c0["conditions"].values()) == 1


def test_the_ruled_out_bars_are_at_least_at_point():
    c0 = Rule.read_rule(report(cmax=.85, crnd=.85, primary=dict(likelihood=.85)))["c0"]
    assert c0["outcome"] == "seams_ruled_out"


def test_a_fall_shown_does_not_need_the_other_forms():
    assert Rule.read_rule(report(cmax=.1, crnd=.1, primary=dict(share=.7, low=.6, high=.8, likelihood=.1)))["c0"]["outcome"] == "fall_shown"


def test_a_share_with_no_interval_is_unresolved():
    r = report()
    r["forms"]["seam_cval45"]["share_of_gap"].update(bleu=None, bleu_ci95=None)
    assert Rule.read_rule(r)["c0"]["outcome"] == "unresolved"


@pytest.mark.parametrize("kw,text", [(dict(share=.7, low=.6, high=.8), "200-clip test, nothing else"), (dict(), "linear bridges between continuous real poses do not limit v1"), (dict(share=.8, low=.7, high=.9), "goes to the owner")])
def test_each_c0_outcome_carries_the_plans_action(kw, text):
    assert text in Rule.read_rule(report(primary=kw))["c0"]["action"]


# ---- reported, not deciding -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("low,point,shown", [(.01, .413, True), (.01, .9, True), (0., .9, False), (.01, .4129, False), (-.1, .9, False)])
def test_the_loop_effect_needs_the_lower_bound_and_the_41_3_point(low, point, shown):
    effect = Rule.read_rule(report(primary=dict(loop_low=low, loop_point=point)))["reported"]["loop_effect"]["seam_cval45"]
    assert effect["shown"] is shown


def test_the_loop_effect_decides_nothing():
    a = Rule.read_rule(report(primary=dict(loop_low=.1, loop_point=.9)))
    b = Rule.read_rule(report())
    assert a["c0"] == b["c0"] and a["b1"] == b["b1"]


def test_what_is_reported_beside_the_outcomes():
    r = Rule.read_rule(report())["reported"]
    assert r["healing_effect"]["cmin_share"] == .95 and r["healing_effect"]["cval_share"] == .9
    assert r["healing_effect"]["cmin_minus_cval_bleu"] == dict(difference=.2, ci95=[-.1, .5])       # stored as cval − cmin
    assert r["squeeze"] == dict(seam_csq196_min=.5) and r["filled_cuts"] == 2 and r["bridged_frame_share"] == .45 and r["frames_per_bridge"] == 8. and r["likelihood_share"] == .9
    assert r["word_count_forms"] == dict(seam_min45=.97) and set(r["loop_effect"]) == {"seam_cval45", "seam_cmin45", "seam_cmax45", "seam_crnd45", "seam_csq196_min"}


# ---- the script itself ----------------------------------------------------------------------------------------------------------------

def test_the_constants_are_the_plans():
    assert (Rule.P_AUTOENCODER, Rule.P_QUANTISER, Rule.SEAM_BAR, Rule.BRIDGED_SHARE, Rule.FRAMES_PER_BRIDGE, Rule.LOOP_POINT) == (.25, .75, .85, (.40, .47), (7., 9.), .413)
    plan = (ROOT / "docs/NEXT_PHASE_PLAN.md").read_text()
    section = plan[plan.index("### Tokenizer probes and seam ablation declared"):]
    for text in ("p ≤ 0.25", "p ≥ 0.75", "[0.40, 0.47]", "[7, 9]", "41.3%", "upper bound < 0.85", "lower bound ≥ 0.85"):
        assert text in section


def test_it_runs_as_a_script_on_a_local_file_and_opens_no_store(tmp_path, capsys):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report()))
    assert Rule.main(["--report", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["c0"]["outcome"] == "seams_ruled_out" and out["status"] == Rule.STATUS
    source = (ROOT / "scripts/tok_probes_rule.py").read_text()
    for word in ("make_store", "R2Store", "tok-probes/", "boto"):
        assert word not in source


def test_the_rule_reads_every_field_from_a_report_the_merge_really_writes(monkeypatch, tmp_path):
    """The shape, not the numbers: a tiny synthetic read through read and merge. A tiny clip's bridges miss the bounds, which is the void path; the outcomes then read the same report."""
    harness(monkeypatch, tiny_tokenizer(), n=CLIPS)
    forms = ["real", "blank", "round_trip", "identity", "no_quant", "round_trip_masked", "identity_masked", "no_quant_masked", "seam_cval45", "seam_cmax45", "seam_crnd45", "seam_cmin45"]
    assert read(tmp_path, forms) == 0
    merged_report = merged(tmp_path, forms)
    r = Rule.read_rule(merged_report)
    assert r["void"] is True and all("seam_cval45" in x for x in r["reasons"]) and len(r["reasons"]) >= 1
    merged_report["checks"]["canary"].update(code_digest=merged_report["code_digest"], environment=merged_report["environment"])
    seam = merged_report["forms"]["seam_cval45"]["seam"]
    seam.update(bridged_frame_share=.45, frames_per_bridge=8.)
    full = Rule.read_rule(merged_report)
    assert full["void"] is False and full["b1"]["outcome"] in ("autoencoder", "quantiser", "unresolved", "void") and full["c0"]["outcome"] in ("fall_shown", "seams_ruled_out", "unresolved")
    assert set(full["reported"]["loop_effect"]) == {"seam_cval45", "seam_cmin45", "seam_cmax45", "seam_crnd45"} and full["reported"]["healing_effect"]["cmin_minus_cval_bleu"] is not None
