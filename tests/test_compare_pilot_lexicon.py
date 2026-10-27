"""Round 5 pilot decision: paired lexicon difference and the three-part rule."""
import pytest

from scripts.compare_pilot_lexicon import paired_difference, verdict


def report(hits, bleu, run="sign-recognizer-v3"):
    return dict(lexicon=dict(per_word=hits, generated_top5=sum(hits.values()) / (2 * len(hits)), recognizer_run=run),
                meaning=dict(generated=dict(bleu=bleu)))


def test_a_clear_gain_on_many_words_is_promising():
    words = [f"w{k}" for k in range(150)]
    arm = report({w: (2 if k % 3 == 0 else 0) for k, w in enumerate(words)}, 1.7)
    control = report({w: 0 for w in words}, 1.4)
    v = verdict(arm, control)
    assert v["difference"] == pytest.approx(50 / 150, abs=1e-4) and v["ci_excludes_zero"] and v["promising"]


def test_a_gain_below_the_margin_or_lower_bleu_is_not_promising():
    words = [f"w{k}" for k in range(150)]
    arm = report({w: (1 if k < 9 else 0) for k, w in enumerate(words)}, 1.7)      # +0.03
    control = report({w: 0 for w in words}, 1.4)
    assert not verdict(arm, control)["promising"]
    strong = report({w: 2 for w in words}, 1.3)
    assert not verdict(strong, control)["promising"]                                # meaning fell


def test_reports_must_share_words_and_instrument():
    with pytest.raises(ValueError):
        paired_difference({"a": 1}, {"b": 1})
    with pytest.raises(ValueError):
        verdict(report({"a": 1}, 1.), report({"a": 1}, 1., run="sign-recognizer-v2"))


def test_paired_bleu_is_non_inferiority_when_both_reports_keep_their_translations():
    refs = [f"the cat sat on mat number {i}" for i in range(40)]
    good = list(refs)
    worse = [r.replace("cat sat", "dog ran") for r in refs]
    words = {f"w{i}": 0 for i in range(10)}

    def report(hyps, hits):
        return dict(lexicon=dict(recognizer_run="v2", per_word=hits, generated_top5=0.),
                    meaning=dict(generated=dict(bleu=0., hypotheses=hyps), references=refs))
    same = verdict(report(good, words), report(good, words))
    assert same["bleu_paired"]["difference"] == 0 and same["bleu_ok"]                 # equal meaning passes
    lower = verdict(report(worse, words), report(good, words))
    assert lower["bleu_paired"]["ci95"][0] < -.5 and not lower["bleu_ok"]             # clearly worse fails
