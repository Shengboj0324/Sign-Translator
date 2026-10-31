"""Round 20 phase a10, report-only: the adopted arm (W14b, best of 16, gains and no gains) at predicted sentence length. The
job on stubs (argument lists, refusals, the stop), then the real ``--predicted-duration`` path from a word cache on the
round-15 synthetic store: what each form's timing is, the main form reproducing a stored read, and the option off."""
import copy
import json

import numpy as np
import pytest

import scripts.round15_compose_eval as ce
from scripts.round20_compare import compare_forms, main as compare_main, same_codes_check
from signtranslator.generation import composition_v2 as cv2
from signtranslator.generation.compositional_grammar import fitted_plan
from tests.test_round15_stage2 import stage2_store, generator_store  # noqa: F401  (fixtures)
from tests.test_round15_compositional import stubbed  # noqa: F401
from tests.test_round20_eval import likelihood_stub, merge, put_gains, put_set  # noqa: F401
from tests.test_round20_phase_b import FORM, OUT_A, PHASE_A, d20_cache, d20_report, parsed_run, phase_a, scripts_run

PIN = "-0.083,0.3691,18289"
MODEL = dict(corpus=cv2.H2S, partition="train", records=18289, intercept=-.083, slope=.3691, median_abs_error=1.249)
PREDICTED = f"{FORM}{ce.PREDICTED}"
CACHES = dict(gains=f"{OUT_A}/cache-w14b-r10/index.json", nogains=f"{OUT_A}/cache-w14b-nogains-r10/index.json")
READS = dict(gains=f"{OUT_A}/a4-w14b-n16-r10", nogains=f"{OUT_A}/a9-nogains-n16-r10")
OPTIONS = f" --predicted-duration --expect-duration-model={PIN}"


def run_line(commands):
    return next(c for c in commands if "round15_compose_eval.py run" in c)


def stored_arm_line(tmp_path, arm):
    """The stored read's own list, from its own phase."""
    phase, cache = ("a4", CACHES["gains"]) if arm == "gains" else ("a9", CACHES["nogains"])
    return run_line(phase_a(tmp_path, phase, arm="n16", stored=[cache])[2])


# ---- the job's argument lists ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["gains", "nogains"])
def test_a10_is_the_stored_reads_list_plus_the_option(tmp_path, monkeypatch, arm):
    code, out, commands = phase_a(tmp_path, "a10", arm=arm)
    assert code == 0 and "R20-PHASE-A-DONE 0" in out
    run = run_line(commands)
    assert run.endswith(f"--output-key {OUT_A}/a10-predicted-{arm}-r10-shard0.json")
    assert f"--word-cache-key {CACHES[arm]} " in run and "--word-cache-n 16 " in run and "--clips 500" in run
    assert "--likelihood" in run and "--recognizer-run sign-recognizer-v2" in run and "--checkpoint-key" not in run
    assert "--retime" not in run and "--tempo" not in run and "--no-decoder-gains" not in run
    assert ("--lowpass-report-hz 6" in run) == (arm == "nogains")          # each arm as its stored read
    # against the stored read's own list, only the option, the name and the literal cache pin differ
    own = stored_arm_line(tmp_path, arm)
    sha = run.split("--word-cache-sha256 ")[1].split()[0]
    assert run.replace(OPTIONS, "").replace(f"--word-cache-sha256 {sha}", "--word-cache-sha256 stubsha").replace(
        f"a10-predicted-{arm}-r10", READS[arm].rsplit("/", 1)[1]) == own
    seen = parsed_run(tmp_path, monkeypatch, run, "a10")                   # the script's own refusals pass it
    assert seen["predicted_duration"] and seen["word_cache_n"] == 16 and seen["retime"] == "reference"
    assert seen["expect_duration_model"] == PIN and ce.parse_duration_pin(PIN) == (-.083, .3691, 18289)
    assert seen["sentences_key"].endswith("round20/d20.json") and seen["mix_key"].endswith("round10-mixed-v1/manifest.json")
    assert seen["vocabulary_mix_key"] is None


@pytest.mark.parametrize("arm", ["gains", "nogains"])
def test_a10_pins_checks_every_clip_then_merges_paired_with_the_stored_read(tmp_path, arm):
    _, _, commands = phase_a(tmp_path, "a10", arm=arm)
    assert [c for c in commands if c.startswith("pin ") and "acquisition" in c and "/phase-a/" in c] == [
        f"pin {CACHES[arm]}", f"pin {READS[arm]}.json", f"pin {READS[arm]}-shard0.json"]
    done = scripts_run(commands)
    assert ["round15_compose_eval.py run" in done[0], " same-codes " in done[1], " merge " in done[2]] == [True] * 3
    part = f"{OUT_A}/a10-predicted-{arm}-r10-shard0.json"
    assert (f"same-codes --a {part} --b {READS[arm]}-shard0.json --form core_crossfade --first 500 "
            f"--output-key {OUT_A}/a10-predicted-{arm}-r10-reproduces.json") in done[1]
    assert done[2].endswith(f"merge --shard-keys {part} --paired-with {READS[arm]}.json "
                            f"--output-key {OUT_A}/a10-predicted-{arm}-r10.json")


def test_a10s_pins_are_the_hashes_the_stored_shards_recorded():
    text = PHASE_A.read_text()
    for sha in ("d316d544b3a76099e6b5960a8c521ec57aa232f583379b09043b8ec17fc9756a",     # a4 n16's word_cache.sha256
                "507ee9cc8f436107863d203ec844e46114d55d7a053883f030d901b07c7f137b"):    # a9 n16's
        assert f"IDXSHA={sha}" in text
    assert f"--expect-duration-model={PIN}" in text


def test_a10_smoke_reads_the_stored_cache_on_five_clips_against_the_stored_shard(tmp_path):
    code, out, commands = phase_a(tmp_path, "a10", arm="gains", smoke=True, gpu=False)
    assert code == 0
    done = scripts_run(commands)
    assert f"--word-cache-key {CACHES['gains']} " in done[0] and "--word-cache-n 16 " in done[0] and "--clips 5 " in done[0]
    assert f"{OUT_A}/smoke/" in done[0].split("--output-key ")[1]
    assert f"--b {READS['gains']}-shard0.json --form core_crossfade --first 5 " in done[1]
    assert f"{OUT_A}/smoke/" in done[1].split("--a ")[1].split()[0]
    assert "--paired-with" not in done[2]                                  # a smoke's clips are not the stored report's


def test_a10_stops_with_an_alert_and_no_merge_when_the_main_form_differs(tmp_path):
    code, out, commands = phase_a(tmp_path, "a10", arm="nogains", STUB_SAME="0")
    assert code == 1 and "ALERT a10: the main form differs" in out and "R20-PHASE-A-DONE 1" in out
    assert any(" same-codes " in c for c in commands) and not any(" merge " in c for c in commands)
    # the shard stored, the check not passed: a rerun runs nothing new and stops again
    code, out, commands = phase_a(tmp_path, "a10", arm="nogains", STUB_SAME="0",
                                  stored=[f"{OUT_A}/a10-predicted-nogains-r10-shard0.json"])
    assert code == 1 and "ALERT" in out and not any("round15_compose_eval.py" in c for c in commands)


def test_a10_is_write_once(tmp_path):
    code, out, commands = phase_a(tmp_path, "a10", arm="gains", stored=[f"{OUT_A}/a10-predicted-gains-r10.json"])
    assert code == 0 and "stored:" in out and not scripts_run(commands)


@pytest.mark.parametrize("kwargs,message", [
    (dict(arm="gains", vocab="r14b-fraction"), "a10: VOCAB must be r10"),
    (dict(arm="nogains", vocab="r14b-activity"), "a10: VOCAB must be r10"),
    (dict(arm="gains", vocab="r14b-activity", smoke=True), "a10: VOCAB must be r10"),
    (dict(arm="gains", vocab="other"), "VOCAB must be r10, r14b-fraction or r14b-activity"),
    (dict(arm="n16"), "a10: ARM gains or nogains"),
    (dict(), "a10: ARM gains or nogains"),
    (dict(arm="gains", gpu=False), "a10: a full run needs a GPU"),
    (dict(arm="gains", PRIOR="r16"), "PRIOR belongs to the a4 and a9 phases"),
    (dict(arm="gains", PRIOR="w14b"), "PRIOR belongs to the a4 and a9 phases"),
    (dict(arm="nogains", NO_GAINS="1"), "belongs to the cache phase"),
])
def test_a10_refusals(tmp_path, kwargs, message):
    code, out, commands = phase_a(tmp_path, "a10", **kwargs)
    assert code == 1 and message in out and "ALERT" in out and "R20-PHASE-A-DONE 1" in out and not scripts_run(commands)


def test_the_other_phases_do_not_take_the_option():
    text = PHASE_A.read_text()
    lines = [k for k, line in enumerate(text.splitlines()) if "--predicted-duration" in line and not line.startswith("#")]
    assert len(lines) == 2      # a3's tempo-1.0 and a10
    assert "a1 a0 a7 a2 a3 cache a4 a5 a8 a9 a10" in text


# ---- the option's refusals --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("extra,message", [
    ([f"--expect-duration-model={PIN}"], "given without the option they qualify"),
    (["--predicted-duration", "--expect-duration-model", "1,2"], "INTERCEPT,SLOPE,RECORDS"),
    (["--predicted-duration", "--expect-duration-model", "a,b,c"], "INTERCEPT,SLOPE,RECORDS"),
])
def test_the_pin_refusals(tmp_path, stage2_store, capsys, extra, message):  # noqa: F811
    _, sha, *_ = stage2_store
    from tests.test_round15_compose_eval import run_args
    with pytest.raises(SystemExit):
        ce.main(run_args(tmp_path, sha, *extra, "--output-file", str(tmp_path / "x.json")))
    assert message in capsys.readouterr().err


def test_the_duration_pin():
    ce.check_duration_model(MODEL, PIN)
    ce.check_duration_model(dict(MODEL, slope=9.), None)                   # no pin: no check
    for broken in (dict(intercept=-.0831), dict(slope=.3692), dict(records=18288), dict(partition="val")):
        with pytest.raises(SystemExit, match="not the pinned train fit"):
            ce.check_duration_model(dict(MODEL, **broken), PIN)


def test_the_length_report_on_known_rows():
    def row(frames, predicted, hyp):
        plan = {FORM: dict(frames=frames, predicted=dict(frames=predicted, fit="fitted"))} if predicted else {}
        return dict(frames_reference=frames, plans=plan,
                    hypotheses={PREDICTED: hyp, FORM: "a b", "real": "a b", "blank": "a"})
    rows = [row(100, 150, "a b c d"), row(100, 50, "a b"), row(40, None, "a b"), row(50, 100, "a b c d")]
    out = ce.predicted_report(rows, [FORM], ["a b"] * 4, dict(round20=dict(retime="reference")))["predicted_timing"]
    assert out["report_only"] is True and out["timing"] == ce.PREDICTED_TIMING["reference"]
    f = out["forms"][PREDICTED]
    assert f["clips"] == 3 and f["blank_clips_at_reference_length"] == 1         # the word-less clip apart, not in any ratio
    assert f["seconds_over_reference"] == dict(pooled=1.2, mean_per_clip=pytest.approx(4 / 3, abs=1e-4),
                                               median_per_clip=1.5, shorter_than_reference=pytest.approx(.3333, abs=1e-4))
    assert f["fits"] == dict(fitted=3)
    assert f["hypothesis_length_ratio"] == {PREDICTED: 1.5, FORM: 1., "real": 1., "blank": .5}
    assert ce.predicted_report([dict(r, hypotheses={FORM: "a"}) for r in rows], [FORM], ["a"] * 4, {}) == {}
    with pytest.raises(SystemExit, match="no clip has a predicted plan"):
        ce.predicted_report([row(10, None, "a")], [FORM], ["a"], {})


# ---- the real path, from a word cache ------------------------------------------------------------------------------------

@pytest.fixture
def arms(tmp_path, stage2_store, likelihood_stub, monkeypatch):  # noqa: F811
    """The stored read and the a10 read of one cache (both real runs of the script on the synthetic store)."""
    import torch
    import validate_sign_recognizer as vsr
    store, sha, *_ = stage2_store
    put_gains(store)
    ssha, texts = put_set(store, sha)
    labels = ["CUT", "ONION", "SALT", "HEAT", "KNIFE", "HALF", "NOW", "STIR"]
    monkeypatch.setattr(vsr, "load_model", lambda store, run, device: (None, labels, {"checkpoint_key": "r.pt"}))

    def logits(model, states, device, batch=64):       # a motion-dependent read, so spotting tells forms apart
        rows = []
        for s in states:
            rh = s.channels["right_hand"]
            level = float(np.abs(rh.values[rh.valid]).mean()) if rh.valid.any() else 0.
            rows.append(torch.tensor([np.sin(level * (k + 3) * 50.) for k in range(len(labels))]))
        return torch.stack(rows) if rows else torch.zeros(0, len(labels))
    monkeypatch.setattr(vsr, "logits_of", logits)
    # the tiny store has one How2Sign train clip: the duration model is given (its fit is unit-tested)
    monkeypatch.setattr(cv2, "fit_duration_model", lambda manifest: dict(MODEL, intercept=.4, slope=.5, records=18))
    cache = d20_cache(tmp_path, store, sha, ssha, "cache/w14b")
    stored, stored_shards = d20_report(tmp_path, store, sha, ssha, "e/stored", *cache)
    a10, a10_shards = d20_report(tmp_path, store, sha, ssha, "e/a10", *cache, "--predicted-duration",
                                 "--expect-duration-model", ".4,.5,18")
    return dict(store=store, sha=sha, ssha=ssha, texts=texts, cache=cache, stored=stored, stored_shard=stored_shards[0],
                a10=a10, a10_shard=a10_shards[0])


def test_the_two_forms_timing_and_the_report(arms):
    m, shard = arms["a10"], arms["a10_shard"]
    assert m["settings"]["round20"]["predicted_duration"] and m["settings"]["round20"]["retime"] == "reference"
    assert m["round20"]["duration_model"]["slope"] == .5 and m["round20"]["duration_model"]["partition"] == "train"
    motion = 0
    for row, text in zip(shard["clips"], arms["texts"]):
        plan = row["plans"][FORM]
        assert plan["frames"] == row["frames_reference"]                   # the main form: the reference's true length
        seconds = cv2.predicted_seconds(m["round20"]["duration_model"], text)
        assert plan["predicted"]["frames"] == int(round(seconds * 24))     # @predicted: from the text alone
        assert plan["predicted"]["fit"] == "fitted"                        # the same plan rule, no padding
        assert row["motion"][PREDICTED]["seconds"] == pytest.approx(plan["predicted"]["frames"] / 24)
        assert row["motion"][FORM]["seconds_over_reference"] == 1.
        motion += row["motion"][PREDICTED]["seconds_over_reference"] != 1.
    assert motion == 4
    # no form of its own beside the main form: no controls at predicted length
    assert not [f for f in m["forms"] if ce.PREDICTED in f and f != PREDICTED]
    # report-only: no verdict, no rule
    assert PREDICTED not in m["verdicts"] and set(m["verdicts"]) == {FORM}
    p = m["predicted_timing"]
    assert p["report_only"] is True and "same words" in p["timing"]
    f = p["forms"][PREDICTED]
    true = np.array([r["frames_reference"] for r in shard["clips"]], float)
    pred = np.array([r["plans"][FORM]["predicted"]["frames"] for r in shard["clips"]], float)
    assert f["seconds_over_reference"]["pooled"] == round(pred.sum() / true.sum(), 4) and f["clips"] == 4
    assert f["blank_clips_at_reference_length"] == 0 and set(f["hypothesis_length_ratio"]) == {PREDICTED, FORM, "real", "blank"}
    # BLEU, likelihood, motion and spotting of the form, paired with the main form
    assert {"minus_blank", "minus_mismatched", "minus_reference_duration"} <= set(m["forms"][PREDICTED])
    assert {"minus_blank", "minus_mismatched", "minus_reference_duration"} <= set(m["likelihood"]["forms"][PREDICTED])
    assert {"shuffled", "mismatched_words"} <= {k[6:] for k in m["forms"][FORM] if k.startswith("minus_")}
    ratios = m["motion_ratio_to_real"][PREDICTED]
    assert ratios["wrist_speed"] > 0 and ratios["wrist_spread"] > 0 and ratios["seconds_over_reference"] > 0
    spotting = m["spotting"]["scores"][PREDICTED]
    assert {"recall", "order", "minus_mismatched_paired", "minus_reference_duration_paired"} <= set(spotting)
    assert "minus_reference_duration_paired" not in m["spotting"]["scores"][FORM]
    # the paired comparison reads it, spotting included: the same generated words on both sides
    c = compare_forms(m, PREDICTED, m, FORM, boot=20, shards_a=[shard], shards_b=[shard])
    assert c["spotting_guard"]["available"] and c["spotting_guard"]["word_source"] == ["generated"]
    assert "difference" in c["likelihood_minus_blank"]["difference"] and "difference" in c["bleu"]


def test_at_a_shorter_total_the_same_words_share_the_frames_by_the_same_rule():
    """What @predicted changes in the plan: only the total, against the reference's."""
    weights, ratio = [1.2, .8, 1.], .3
    full, short = fitted_plan(weights, ratio, 8.), fitted_plan(weights, ratio, 4.)
    assert full["frames"] == 192 and short["frames"] == 96
    assert all(abs(a / 2 - b) <= 1 for a, b in zip(full["cores"] + full["gaps"], short["cores"] + short["gaps"]))


def test_the_main_form_reproduces_the_stored_read_and_one_value_breaks_it(tmp_path, arms):
    stored, part = arms["stored_shard"], arms["a10_shard"]
    assert same_codes_check(part, stored, FORM, first=4)["same"]
    for a, b in zip(part["clips"], stored["clips"]):               # the whole main form, not only what the check reads
        for k in (FORM, f"{FORM}/shuffled", f"{FORM}/mismatched_words", "real", "blank"):
            assert a["hypotheses"][k] == b["hypotheses"][k] and a["likelihood"][k] == b["likelihood"][k]
        for k in (FORM, f"{FORM}/shuffled", f"{FORM}/mismatched_words", "real"):
            assert a["motion"][k] == b["motion"][k]
        assert a["spotting"][FORM] == b["spotting"][FORM]

    def broken(report, clip, key):                                 # one value, by the smallest amount
        out = copy.deepcopy(report)
        row = out["clips"][clip]["motion"][FORM]
        row[key] = np.nextafter(row[key], np.inf).item()
        return out
    for clip in range(4):
        with pytest.raises(SystemExit, match=rf"clip {clip} motion.core_crossfade.wrist_speed"):
            same_codes_check(part, broken(stored, clip, "wrist_speed"), FORM, first=4)
    with pytest.raises(SystemExit, match="motion.core_crossfade.seconds_over_reference"):
        same_codes_check(broken(part, 3, "seconds_over_reference"), stored, FORM, first=4)
    # the last clip counts: the check reads every clip, as the job asks (--first = the run's clips)
    assert same_codes_check(broken(part, 3, "wrist_speed"), stored, FORM, first=3)["same"]
    # @predicted moves freely: it has no stored read
    moved = copy.deepcopy(part)
    moved["clips"][0]["motion"][PREDICTED]["wrist_speed"] += 1.
    assert same_codes_check(moved, stored, FORM, first=4)["same"]
    # another cache's words never pass as the stored read
    other = copy.deepcopy(part)
    other["settings"]["round20"]["word_cache"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="settings.round20.word_cache"):
        same_codes_check(other, stored, FORM, first=4)
    # the command the job calls exits non-zero
    root = ["--store", f"dir:{tmp_path / 'store'}", "--first", "4"]
    assert compare_main(["same-codes", *root, "--a", "e/a10-shard0.json", "--b", "e/stored-shard0.json"]) == 0
    arms["store"].put("e/broken-shard0.json", json.dumps(broken(stored, 2, "wrist_spread")).encode())
    with pytest.raises(SystemExit, match="clip 2 motion.core_crossfade.wrist_spread") as e:
        compare_main(["same-codes", *root, "--a", "e/a10-shard0.json", "--b", "e/broken-shard0.json"])
    assert e.value.code not in (0, None)


def test_the_merge_reports_the_main_form_against_the_stored_read(tmp_path, arms):
    merge(tmp_path, "e/a10-shard0.json", "e/a10p.json", "--paired-with", "e/stored.json")
    m = json.loads(arms["store"].get("e/a10p.json"))
    assert m["comparators"]["paired_with"]["key"] == "e/stored.json"
    assert m["forms"][FORM]["minus_paired_core_crossfade"]["difference"] == 0
    assert m["likelihood"]["forms"][FORM]["minus_paired_core_crossfade"]["difference"] == 0


def test_a_wrong_fit_stops_the_run(tmp_path, arms):
    from tests.test_round15_compose_eval import run_args
    args = [a for a in run_args(tmp_path, arms["sha"]) if a not in ("--checkpoint-key", "prior.pt")]
    args[args.index("core_crossfade,oracle_stage2")] = FORM
    with pytest.raises(SystemExit, match="not the pinned train fit"):
        ce.main([*args, "--sentences-key", "sets/d20.json", "--sentences-sha256", arms["ssha"], *arms["cache"],
                 "--predicted-duration", f"--expect-duration-model={PIN}", "--output-key", "e/wrong.json"])
    assert not arms["store"].exists("e/wrong.json")


def test_the_option_off_leaves_the_stored_shape(arms):
    """No --predicted-duration: no form, no spotting row, no report key, and the settings keys the stored reads have."""
    shard, m = arms["stored_shard"], arms["stored"]
    text = json.dumps(shard)
    assert "@predicted" not in text and '"predicted"' not in text and "duration_model" not in text
    assert "predicted_timing" not in m and "@predicted" not in json.dumps(m)
    assert shard["settings"]["round20"]["predicted_duration"] is False
    assert list(shard["clips"][0]["spotting"]) == [FORM]
    assert "expect_duration_model" not in text
    # the whole shard equals the a10 run's once the option's own rows are removed
    def without(s):
        out = copy.deepcopy(s)
        out.pop("seconds")
        out["round20"].pop("duration_model", None)
        out["settings"]["round20"]["predicted_duration"] = False
        for r in out["clips"]:
            for block in ("hypotheses", "likelihood", "motion", "spotting"):
                r[block].pop(PREDICTED, None)
            r["plans"][FORM].pop("predicted", None)
        return out
    assert json.dumps(without(arms["a10_shard"])) == json.dumps(without(shard))


def test_the_default_round15_run_is_untouched(tmp_path, stage2_store):  # noqa: F811
    from tests.test_round15_compose_eval import run_args
    store, sha, *_ = stage2_store
    assert ce.main(run_args(tmp_path, sha, "--clips", "5", "--output-key", "e/d.json")) == 0
    shard = json.loads(store.get("e/d.json"))
    assert "round20" not in shard and "round20" not in shard["settings"]
    assert sorted(shard["clips"][0]["spotting"]) == ["core_crossfade", "oracle_stage2"]
    merge(tmp_path, "e/d.json", "e/dm.json")
    assert "predicted_timing" not in json.loads(store.get("e/dm.json"))
    assert "expect_duration_model" not in ce.ROUND20_DEFAULTS    # the pin alone never makes a run a round-20 run
