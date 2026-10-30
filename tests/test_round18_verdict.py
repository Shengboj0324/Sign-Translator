"""Round 18 verdict: the declared three-row rule over compare_pilot_lexicon.py outputs, and the not-evaluable checks."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import round18_verdict as v  # noqa: E402


def pilot(**over):
    p = dict(checkpoint_key=v.ARM_CHECKPOINT, clips=1000, word_list=dict(sha256=v.WORD_LIST_SHA),
             lexicon=dict(recognizer_run=v.INSTRUMENT, generated_top5=.66),
             meaning=dict(generated=dict(bleu=2.1, ci95=[1.8, 2.4], hypotheses=["a"] * 3), blank=dict(bleu=1.859),
                          real=dict(bleu=4.136), above_blank=True))
    p.update(over)
    return p


def comparison(control, bleu_lo, lex_lo, **over):
    c = dict(arm_key=v.ARM_PILOT, control_key=control, instrument=v.INSTRUMENT,
             bleu_paired=dict(difference=bleu_lo + .3, ci95=[bleu_lo, bleu_lo + .6], margin=-.5),
             difference=lex_lo + .03, ci95=[lex_lo, lex_lo + .06], control=dict(lexicon=.62, bleu=1.658))
    c.update(over)
    return c


@pytest.mark.parametrize("r16c_bleu, w14b_bleu, r16c_lex, expected", [
    (.01, .02, -.04, "PASS"),
    (0., .02, -.04, "FAIL"),            # a lower bound of exactly 0 is not above 0
    (.01, -.01, .02, "FAIL"),
    (.01, .02, -.05, "FAIL"),           # the lexicon floor is strict
    (-.2, -.3, .0, "FAIL"),
])
def test_the_rule_needs_all_three_rows(r16c_bleu, w14b_bleu, r16c_lex, expected):
    out = v.verdict(pilot(), comparison(v.R16C_PILOT, r16c_bleu, r16c_lex), comparison(v.W14B_PILOT, w14b_bleu, -.9))
    assert out["verdict"] == expected
    assert [r["passes"] for r in out["rows"]] == [r16c_bleu > 0, w14b_bleu > 0, r16c_lex > -.05]
    assert out["reported"]["blank_bleu"] == 1.859 and out["reported"]["arm_minus_blank"] == round(2.1 - 1.859, 3)


def test_wrong_inputs_are_not_evaluable():
    good = (comparison(v.R16C_PILOT, .1, 0.), comparison(v.W14B_PILOT, .1, 0.))
    for bad_pilot, match in [(pilot(clips=300), "clips"), (pilot(word_list={}), "word list"),
                             (pilot(checkpoint_key="training-runs/x/checkpoints/step-00010000.pt"), "checkpoint"),
                             (pilot(lexicon=dict(recognizer_run="sign-recognizer-v2", generated_top5=.6)), "lexicon read"),
                             (pilot(meaning=dict(generated=dict(bleu=2.))), "hypotheses")]:
        with pytest.raises(v.NotEvaluable, match=match):
            v.verdict(bad_pilot, *good)
    for bad, match in [(comparison(v.W14B_PILOT, .1, 0.), "control"),
                       (comparison(v.R16C_PILOT, .1, 0., arm_key="other.json"), "arm"),
                       (comparison(v.R16C_PILOT, .1, 0., instrument="sign-recognizer-v2"), "instrument"),
                       (comparison(v.R16C_PILOT, .1, 0., bleu_paired=None), "paired BLEU")]:
        with pytest.raises(v.NotEvaluable, match=match):
            v.verdict(pilot(), bad, good[1])


def test_main_writes_the_verdict_and_exits_0_4_or_2(tmp_path, capsys):
    files = {}
    for name, value in [("pilot", pilot()), ("pass16", comparison(v.R16C_PILOT, .01, 0.)),
                        ("pass14", comparison(v.W14B_PILOT, .01, 0.)), ("fail14", comparison(v.W14B_PILOT, -.01, 0.)),
                        ("bad", comparison(v.R16C_PILOT, .01, 0., bleu_paired=None))]:
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(value))
    out = tmp_path / "verdict.json"
    run = lambda a, b: v.main(["--arm-pilot", str(files["pilot"]), "--vs-r16c", str(files[a]),  # noqa: E731
                               "--vs-w14b", str(files[b]), "--out", str(out)])
    assert run("pass16", "pass14") == 0 and json.loads(out.read_text())["verdict"] == "PASS"
    assert "R18-VERDICT PASS" in capsys.readouterr().out
    assert run("pass16", "fail14") == 4 and json.loads(out.read_text())["verdict"] == "FAIL"
    assert run("bad", "pass14") == 2 and json.loads(out.read_text())["verdict"] == "NOT EVALUABLE"
