"""Round 5 pilot decision: paired lexicon difference and the three-part rule."""
import json

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


def split_report(hits, mirrored_hits, words, run="sign-recognizer-v2"):
    r = report(hits, 1., run=run)
    r["lexicon_mirrored"] = dict(per_word=mirrored_hits, recognizer_run=run)
    r["lexicon_split"] = dict(words=words)
    return r


def test_per_subset_differences_resample_within_each_subset_for_both_reads():
    from scripts.compare_pilot_lexicon import per_subset_differences
    covered, uncovered = [f"c{k}" for k in range(30)], [f"u{k}" for k in range(10)]
    words = dict(covered=covered, uncovered=uncovered)
    arm = split_report({**{w: 2 for w in covered}, **{w: k % 3 for k, w in enumerate(uncovered)}},
                       {w: 1 for w in covered + uncovered}, words)
    control = split_report({w: 0 for w in covered + uncovered}, {w: 1 for w in covered + uncovered}, words)
    out = per_subset_differences(arm, control, resamples=2000)
    assert set(out) == {"lexicon", "lexicon_mirrored"} and set(out["lexicon"]) == {"covered", "uncovered"}
    assert out["lexicon"]["covered"] == dict(words=30, difference=1., ci95=[1., 1.], arm=1., control=0.)
    u = out["lexicon"]["uncovered"]
    assert u["words"] == 10 and u["difference"] == pytest.approx(sum(k % 3 for k in range(10)) / 20, abs=1e-4)
    assert u["ci95"][0] < u["difference"] < u["ci95"][1]
    assert out["lexicon_mirrored"]["covered"]["difference"] == 0 == out["lexicon_mirrored"]["uncovered"]["difference"]
    one = dict(covered=covered + uncovered, uncovered=[])
    empty = per_subset_differences(split_report(arm["lexicon"]["per_word"], arm["lexicon_mirrored"]["per_word"], one),
                                   split_report(control["lexicon"]["per_word"], control["lexicon_mirrored"]["per_word"],
                                                one), resamples=50)
    assert empty["lexicon"]["uncovered"] == dict(words=0, difference=None, ci95=None, arm=None, control=None)
    # the default verdict is unchanged by the split blocks
    plain = verdict(report(arm["lexicon"]["per_word"], 1., run="sign-recognizer-v2"),
                    report(control["lexicon"]["per_word"], 1., run="sign-recognizer-v2"))
    assert verdict(arm, control) == plain
    # a mirrored read only one report has is skipped; differing splits or a missing split are refused
    del control["lexicon_mirrored"]
    assert set(per_subset_differences(arm, control, resamples=100)) == {"lexicon"}
    regrouped = split_report(control["lexicon"]["per_word"], {}, dict(covered=covered + uncovered[:5],
                                                                      uncovered=uncovered[5:]))
    with pytest.raises(ValueError, match="split the words differently"):
        per_subset_differences(arm, regrouped)
    with pytest.raises(ValueError, match="lexicon_split"):
        per_subset_differences(arm, report(control["lexicon"]["per_word"], 1., run="sign-recognizer-v2"))
    with pytest.raises(ValueError, match="different instruments"):
        per_subset_differences(arm, split_report(control["lexicon"]["per_word"], {}, words, run="sign-recognizer-v3"))


def test_the_cli_adds_per_subset_only_with_the_flag(tmp_path, monkeypatch):
    import scripts.compare_pilot_lexicon as cpl
    import signtranslator.data.text_motion_dataset as tmd
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path)
    words = dict(covered=["a", "b"], uncovered=["c"])
    for key, hits in (("arm.json", {"a": 2, "b": 1, "c": 0}), ("control.json", {"a": 0, "b": 1, "c": 0})):
        store.put(key, json.dumps(split_report(hits, hits, words)).encode())
    monkeypatch.setattr(tmd, "R2Store", lambda env: store)
    base = ["--arm-key", "arm.json", "--control-key", "control.json", "--resamples", "50"]
    cpl.main(base + ["--output-key", "plain.json"])
    cpl.main(base + ["--per-subset", "--output-key", "split.json"])
    plain, split = json.loads(store.get("plain.json")), json.loads(store.get("split.json"))
    assert "per_subset" not in plain and {k: v for k, v in split.items() if k != "per_subset"} == plain
    assert split["per_subset"]["lexicon"]["covered"]["difference"] == .5
