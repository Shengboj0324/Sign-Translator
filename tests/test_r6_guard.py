"""Test-set guard (scripts/r6_guard.py) and the strict review's split modes: R6 and every other test-reading step run
only if the exit gate, R1-R5 (R2b included) and R7 on the validation states all passed; otherwise nothing from the test
partition is read and "R6 not run" is recorded."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import r6_guard  # noqa: E402
import strict_prior_review as spr  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402

SEL = "P-sweep-h1.1-f1.0-hs0.1"
SETTING = "motion_tokens_v2 guidance 1"


def gate(all_pass=True, failing=None, run=SEL):
    failing = [] if failing is None else failing
    checks = {"body_speed_ratio": [1.0, True], "body_variation_ratio": [1.0, True]}
    rows = [dict(run=run, setting=SETTING, all_pass=all_pass, passed=2 if all_pass else 1, total=2, failing=failing,
                 per_group={"c01": dict(checks=checks), "pool": dict(checks=checks)}),
            dict(run="other", setting=SETTING, all_pass=True, passed=2, total=2, failing=[], per_group={})]
    return dict(rows=rows, selected=dict(run=run, setting=SETTING, frechet=1., failing=failing))


def review(**overrides):
    rules = {k: {"pass": True} for k in r6_guard.VALIDATION_RULES}
    rules["R6_test_confirmation"] = dict(not_run=spr.R6_PENDING, **{"pass": None})
    out = dict(selected=spr.selected_setting(gate()), rules=rules, validation_only=True, quick_dry_run=False,
               derived="d", groups=list(GROUPS), translation_smoothing=.1, heldout_save_generated=None, **{"pass": False})
    out.update(overrides)
    return out


GROUPS = ("c01", "c03", "c05")
# what smoothness_check.py writes for the selection's validation states and for R6's saved test states
R7 = {"pass": True, "groups": {f"d/generated/P-sweep-h1.1-f1.0-{g}/guidance-1": {} for g in GROUPS},
      "post_processing": dict(head_smoothing=.1, head_damping=1., translation_smoothing=.1)}
R7_TEST = {"pass": True, "groups": {"d/review-generated/test/guidance-1": {}},
           "post_processing": dict(head_smoothing=0., head_damping=1., translation_smoothing=0.)}


def test_everything_passed_allows_r6():
    d = r6_guard.decide(gate(), review(), R7)
    assert d["run_r6"] is True and d["reasons"] == [] and all(d["rules"].values())


@pytest.mark.parametrize("rule", r6_guard.VALIDATION_RULES)
def test_any_failed_validation_rule_blocks(rule):
    rv = review()
    rv["rules"][rule]["pass"] = False
    d = r6_guard.decide(gate(), rv, R7)
    assert d["run_r6"] is False and d["message"] == r6_guard.NOT_RUN
    assert d["rules"][rule] is False and any(r.startswith(rule) for r in d["reasons"])


@pytest.mark.parametrize("value", [None, 1, "true", "True", [True], {"x": True}, 1.0])
def test_only_a_literal_true_passes(value):
    rv = review()
    rv["rules"]["R2b_2d_back_translation_above_blank"]["pass"] = value
    assert not r6_guard.decide(gate(), rv, R7)["run_r6"]
    assert not r6_guard.decide(gate(), review(), {"pass": value})["run_r6"]


@pytest.mark.parametrize("rule", r6_guard.VALIDATION_RULES)
def test_a_missing_rule_blocks(rule):   # e.g. R2b silently skipped without --second-translator-key
    rv = review()
    del rv["rules"][rule]
    assert not r6_guard.decide(gate(), rv, R7)["run_r6"]


def test_the_gate_must_pass():
    assert not r6_guard.decide(gate(all_pass=False, failing=["c01:nod_rate"]), review(), R7)["run_r6"]
    assert not r6_guard.decide(gate(all_pass=True, failing=["c01:nod_rate"]), review(), R7)["run_r6"]
    assert not r6_guard.decide(gate(all_pass=False), review(), R7)["run_r6"]
    g = gate(); g["selected"].pop("failing")
    assert not r6_guard.decide(g, review(), R7)["run_r6"]
    g = gate(); g["rows"].append(dict(g["rows"][0]))       # an ambiguous selection
    assert not r6_guard.decide(g, review(), R7)["run_r6"]
    assert not r6_guard.decide(None, review(), R7)["run_r6"]
    assert not r6_guard.decide({"rows": []}, review(), R7)["run_r6"]


def test_the_review_must_be_a_real_validation_report_of_the_selection():
    assert not r6_guard.decide(gate(), review(quick_dry_run=True), R7)["run_r6"]
    assert not r6_guard.decide(gate(), review(validation_only=False), R7)["run_r6"]
    assert not r6_guard.decide(gate(), review(selected=dict(run="other", setting=SETTING)), R7)["run_r6"]
    assert not r6_guard.decide(gate(), review(selected=dict(run=SEL, setting="motion_tokens_v2 guidance 2")), R7)["run_r6"]
    rv = review(); del rv["quick_dry_run"]
    assert not r6_guard.decide(gate(), rv, R7)["run_r6"]
    assert not r6_guard.decide(gate(), None, R7)["run_r6"]


def test_r7_on_validation_must_pass():
    assert not r6_guard.decide(gate(), review(), {"pass": False})["run_r6"]
    assert not r6_guard.decide(gate(), review(), None)["run_r6"]


def test_a_stale_r7_report_blocks():
    """An R7 report of other states or other post-processing (e.g. left from an earlier run) never opens the guard."""
    for change in (dict(groups={f"d/generated/other-{g}/guidance-1": {} for g in GROUPS}),
                   dict(groups={f"d/generated/P-sweep-h1.1-f1.0-{g}/guidance-1.5": {} for g in GROUPS}),
                   dict(groups={"d/generated/P-sweep-h1.1-f1.0-c01/guidance-1": {}}),
                   dict(post_processing=dict(head_smoothing=.2, head_damping=1., translation_smoothing=.2)),
                   dict(post_processing=None), dict(groups=None)):
        d = r6_guard.decide(gate(), review(), {**R7, **change})
        assert d["run_r6"] is False and d["rules"]["R7_smoothness_validation"] is False, change
    for missing in ("derived", "groups", "translation_smoothing"):
        rv = review(); del rv[missing]
        assert not r6_guard.decide(gate(), rv, R7)["run_r6"], missing


def test_verdict_when_blocked_records_r6_not_run():
    rv = review()
    rv["rules"]["R5_sign_level_words"]["pass"] = False
    guard = r6_guard.decide(gate(), rv, R7)
    v = r6_guard.verdict(rv, guard, dict(validation=R7, test=None))
    assert v["pass"] is False and v["r6"] == r6_guard.NOT_RUN and v["rules"]["R6_test_confirmation"] == "not run"
    assert v["r7_test"] == "not run" and v["rules"]["R7_smoothness"] is False
    assert v["test_set_guard"]["message"] == r6_guard.NOT_RUN


def test_verdict_when_r6_ran():
    guard = r6_guard.decide(gate(), review(), R7)
    final = review(validation_only=False, heldout_save_generated="d/review-generated", **{"pass": True})
    final["rules"]["R6_test_confirmation"] = {"pass": True}
    v = r6_guard.verdict(final, guard, dict(validation=R7, test=R7_TEST))
    assert v["pass"] is True and "r6" not in v and v["rules"]["R7_smoothness"] is True
    final["rules"]["R6_test_confirmation"] = {"pass": False}
    final["pass"] = False
    assert r6_guard.verdict(final, guard, dict(validation=R7, test=R7_TEST))["pass"] is False
    final["rules"]["R6_test_confirmation"] = {"pass": True}
    final["pass"] = True
    assert r6_guard.verdict(final, guard, dict(validation=R7, test={**R7_TEST, "pass": False}))["pass"] is False
    assert r6_guard.verdict(final, guard, dict(validation=R7, test=None))["pass"] is False
    stale = {**R7_TEST, "groups": {"old/review-generated/test/guidance-1": {}}}
    assert r6_guard.verdict(final, guard, dict(validation=R7, test=stale))["pass"] is False
    assert r6_guard.verdict(review(validation_only=False, heldout_save_generated="d/review-generated",
                                   selected=dict(run="other", setting=SETTING), **{"pass": True}) | {"rules": final["rules"]},
                            guard, dict(validation=R7, test=R7_TEST))["pass"] is False


def test_cli_exit_codes_and_reports(tmp_path):
    store = DirectoryStore(tmp_path)
    store.put("e/gate.json", json.dumps(gate()).encode())
    store.put("e/r7.json", json.dumps(R7).encode())
    store.put("e/ok.json", json.dumps(review()).encode())
    bad = review(); bad["rules"]["R2b_2d_back_translation_above_blank"]["pass"] = False
    store.put("e/bad.json", json.dumps(bad).encode())
    common = ["--store", f"dir:{tmp_path}", "--gate-key", "e/gate.json", "--r7-validation-key", "e/r7.json"]
    assert r6_guard.main(["decide", *common, "--review-key", "e/ok.json", "--output-key", "e/g-ok.json"]) == 0
    assert r6_guard.main(["decide", *common, "--review-key", "e/bad.json", "--output-key", "e/g-bad.json"]) == r6_guard.BLOCKED
    assert json.loads(store.get("e/g-bad.json"))["message"] == r6_guard.NOT_RUN
    assert r6_guard.main(["decide", *common, "--review-key", "e/missing.json", "--output-key", "e/g-miss.json"]) == r6_guard.BLOCKED
    assert r6_guard.main(["verdict", *common, "--review-key", "e/bad.json", "--guard-key", "e/g-bad.json",
                          "--output-key", "e/verdict.json"]) == 0
    v = json.loads(store.get("e/verdict.json"))
    assert v["pass"] is False and v["r6"] == r6_guard.NOT_RUN


def test_guard_on_a_real_multi_signer_gate_report(tmp_path):
    """The guard reads multi_signer_gate.py's own output format."""
    import multi_signer_gate
    from exit_gate import MARKER_SCORES
    store = DirectoryStore(tmp_path)
    ch = lambda s: dict(speed_ratio=s, variation_ratio=1., geodesic_deg=20.)  # noqa: E731
    res = lambda head: dict(frechet=100., body=ch(1.), left_hand=ch(1.), right_hand=ch(1.), head_rotation=ch(head),  # noqa: E731
                            grammar=dict(distributions={m: dict(mean_ratio=1.) for m in MARKER_SCORES}))
    for run, head in ((SEL, 1.0), ("Q-sweep-h1.0-f1.0-hs0.1", 1.0)):
        for g in ("c01", "c03"):
            store.put(f"d/evaluations/{run}-{g}.json", json.dumps(dict(results={SETTING: res(head)})).encode())
    assert multi_signer_gate.main(["--store", f"dir:{tmp_path}", "--derived", "d", "--runs", SEL, "--groups", "c01,c03",
                                   "--output-key", "d/gate.json"]) == 0
    g = json.loads(store.get("d/gate.json"))
    assert r6_guard.decide(g, review(), R7)["run_r6"] is True
    store.put(f"d/evaluations/{SEL}-c03.json", json.dumps(dict(results={SETTING: res(1.8)})).encode())   # head too fast
    multi_signer_gate.main(["--store", f"dir:{tmp_path}", "--derived", "d", "--runs", SEL, "--groups", "c01,c03",
                            "--output-key", "d/gate.json"])
    d = r6_guard.decide(json.loads(store.get("d/gate.json")), review(), R7)
    assert d["run_r6"] is False and d["rules"]["exit_gate"] is False


# --- strict_prior_review.py modes --------------------------------------------------------------------------------

def review_store(tmp_path, monkeypatch, validation=None, r7=R7):
    store = DirectoryStore(tmp_path)
    store.put("m.json", b"{}")
    store.put("d/evaluations/gate.json", json.dumps(gate()).encode())
    if validation is not None:
        store.put("d/evaluations/val.json", json.dumps(validation).encode())
    if r7 is not None:
        store.put("d/evaluations/r7.json", json.dumps(r7).encode())
    else:
        (tmp_path / "d/evaluations/r7.json").unlink(missing_ok=True)
    monkeypatch.setattr(spr, "load_manifest", lambda path, expected_sha256: {"records": []})
    calls = []
    import evaluate_text_to_motion as evaluate

    def fake_evaluate(argv):            # stands in for every test-partition read
        calls.append(argv)
        key = argv[argv.index("--output-key") + 1]
        result = dict(frechet=1., body=dict(speed_ratio=1., variation_ratio=1., geodesic_deg=5.))
        store.put(key, json.dumps(dict(results={"motion_tokens_v2 guidance 1": result})).encode())
        return 0
    monkeypatch.setattr(evaluate, "main", fake_evaluate)
    return store, calls


BASE = ["--derived", "d", "--gate-key", "d/evaluations/gate.json", "--manifest-key", "m.json", "--manifest-sha256", "x",
        "--checkpoint-key", "ckpt.pt", "--retrieval-key", "r.pt", "--translator-key", "t.pt", "--device", "cpu",
        "--translation-smoothing", "0.1", "--groups", "c01,c03,c05"]


def cli_args(tmp_path):
    return ["--store", f"dir:{tmp_path}", *BASE]


def validation_report(tmp_path, monkeypatch, extra=(), **rule_overrides):
    """What --validation-only writes for SEL with cli_args (R1-R5 stubbed), with some rules' pass overridden."""
    store, _ = review_store(tmp_path, monkeypatch)
    monkeypatch.setattr(spr, "validation_rules", lambda *a: ({k: {"pass": True} for k in r6_guard.VALIDATION_RULES},
                                                             {"r5_word_source": {"words": ["A"]}}))
    assert spr.main([*cli_args(tmp_path), *extra, "--validation-only", "--output-key", "d/evaluations/val.json"]) == 0
    report = json.loads(store.get("d/evaluations/val.json"))
    for k, v in rule_overrides.items():
        report["rules"][k]["pass"] = v
    return report


def test_validation_only_reads_no_test_data(tmp_path, monkeypatch):
    store, calls = review_store(tmp_path, monkeypatch)
    monkeypatch.setattr(spr, "validation_rules", lambda *a: ({k: {"pass": True} for k in r6_guard.VALIDATION_RULES},
                                                             {"r5_word_source": {}}))
    assert spr.main([*cli_args(tmp_path), "--validation-only", "--output-key", "d/evaluations/val.json"]) == 0
    assert calls == []
    out = json.loads(store.get("d/evaluations/val.json"))
    assert out["validation_only"] is True and out["pass"] is False and out["quick_dry_run"] is False
    assert out["rules"]["R6_test_confirmation"] == {"not_run": spr.R6_PENDING, "pass": None}
    assert r6_guard.decide(gate(), out, R7)["run_r6"] is True       # the report the guard expects


def test_r6_only_refuses_before_reading_test_data_when_a_rule_failed(tmp_path, monkeypatch):
    store, calls = review_store(tmp_path, monkeypatch, validation_report(tmp_path, monkeypatch, R2b_2d_back_translation_above_blank=False))
    with pytest.raises(SystemExit, match="R6 not run: validation rules failed"):
        spr.main([*cli_args(tmp_path), "--r6-from-validation-key", "d/evaluations/val.json", "--r7-validation-key",
                  "d/evaluations/r7.json", "--output-key", "d/evaluations/final.json"])
    assert calls == [] and not store.exists("d/evaluations/final.json")


def test_r6_only_refuses_without_r7_validation(tmp_path, monkeypatch):
    store, calls = review_store(tmp_path, monkeypatch, validation_report(tmp_path, monkeypatch), r7=None)
    with pytest.raises(SystemExit, match="R6 not run"):
        spr.main([*cli_args(tmp_path), "--r6-from-validation-key", "d/evaluations/val.json", "--r7-validation-key",
                  "d/evaluations/r7.json", "--output-key", "d/evaluations/final.json"])
    assert calls == []
    with pytest.raises(SystemExit):       # argparse: the R7 report is required
        spr.main([*cli_args(tmp_path), "--r6-from-validation-key", "d/evaluations/val.json", "--output-key", "x.json"])


def test_r6_only_refuses_a_report_of_another_setting(tmp_path, monkeypatch):
    other = validation_report(tmp_path, monkeypatch)
    other["translation_smoothing"] = 0.
    store, calls = review_store(tmp_path, monkeypatch, other)
    with pytest.raises(SystemExit, match="another selection or setting"):
        spr.main([*cli_args(tmp_path), "--r6-from-validation-key", "d/evaluations/val.json", "--r7-validation-key",
                  "d/evaluations/r7.json", "--output-key", "d/evaluations/final.json"])
    assert calls == []


def test_r6_only_runs_and_merges_when_the_guard_allows(tmp_path, monkeypatch):
    store, calls = review_store(tmp_path, monkeypatch, validation_report(tmp_path, monkeypatch))
    assert spr.main([*cli_args(tmp_path), "--r6-from-validation-key", "d/evaluations/val.json", "--r7-validation-key",
                     "d/evaluations/r7.json", "--output-key", "d/evaluations/final.json"]) == 0
    assert [a[a.index("--partition") + 1] for a in calls] == ["test", "heldout_official"]
    assert all(a[a.index("--checkpoint-key") + 1] == "ckpt.pt" and a[a.index("--guidance") + 1] == "1" for a in calls)
    out = json.loads(store.get("d/evaluations/final.json"))
    assert out["rules"]["R6_test_confirmation"]["pass"] is True and out["rules"]["R6_test_confirmation"]["regressions"] == []
    assert set(out["rules"]) == set(r6_guard.VALIDATION_RULES) | {"R6_test_confirmation"}
    assert out["pass"] is True and out["validation_only"] is False and out["test_set_guard"]["run_r6"] is True
    assert out["r5_word_source"] == {"words": ["A"]} and out["validation_report_key"] == "d/evaluations/val.json"


def test_r6_only_never_runs_on_a_quick_report(tmp_path, monkeypatch):
    quick = validation_report(tmp_path, monkeypatch, extra=("--quick",))
    assert quick["quick_dry_run"] is True
    store, calls = review_store(tmp_path, monkeypatch, quick)
    with pytest.raises(SystemExit, match="R6 not run"):
        spr.main([*cli_args(tmp_path), "--quick", "--r6-from-validation-key", "d/evaluations/val.json", "--r7-validation-key",
                  "d/evaluations/r7.json", "--output-key", "d/evaluations/final.json"])
    assert calls == []


def test_the_modes_are_exclusive(tmp_path, monkeypatch):
    review_store(tmp_path, monkeypatch, validation_report(tmp_path, monkeypatch))
    with pytest.raises(SystemExit):
        spr.main([*cli_args(tmp_path), "--validation-only", "--r6-from-validation-key", "d/evaluations/val.json",
                  "--r7-validation-key", "d/evaluations/r7.json", "--output-key", "x.json"])


def test_default_mode_still_runs_r1_to_r6_in_one_process(tmp_path, monkeypatch):
    store, calls = review_store(tmp_path, monkeypatch)
    monkeypatch.setattr(spr, "validation_rules", lambda *a: ({k: {"pass": True} for k in r6_guard.VALIDATION_RULES}, {}))
    assert spr.main([*cli_args(tmp_path), "--output-key", "d/evaluations/full.json"]) == 0
    out = json.loads(store.get("d/evaluations/full.json"))
    assert len(calls) == 2 and out["pass"] is True and out["validation_only"] is False


# --- pinned R5 words ----------------------------------------------------------------------------------------------

def test_r5_words_in_the_pilots_form():
    records = [dict(corpus="how2sign-two-view-text-v1", partition="train", text="My friend likes the HOUSE, ok?"),
               dict(corpus="how2sign-two-view-text-v1", partition="val", text="cow"),          # not training
               dict(corpus="youtube-asl-v1", partition="train", text="horse"),               # not How2Sign
               dict(corpus="how2sign-two-view-text-v1", partition="train", text="thank-you, bus")]
    glosses = ["FRIEND", "HOUSE", "COW", "HORSE", "OK", "BUS", "THANK-YOU", "LIKES", "MY"]
    index = {g: i for i, g in enumerate(["BUS", "FRIEND", "HOUSE", "OK", "COW", "HORSE", "THANK-YOU", "MY"])}
    # LIKES: not known to the recogniser; THANK-YOU: not alphabetic; COW: only in val; HORSE: only in YouTube
    assert spr.r5_vocabulary_words(glosses, records, index) == ["BUS", "FRIEND", "HOUSE", "MY", "OK"]
