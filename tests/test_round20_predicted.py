"""Round 20 phase a10, report-only: the adopted arm (W14b, best of 16, gains and no gains) at predicted sentence length. The
job on stubs (argument lists, refusals, the stop), then the real ``--predicted-duration`` path from a word cache on the
round-15 synthetic store: what each form's timing is, the main form reproducing a stored read, and the option off."""
import copy
import json
import re
import subprocess

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
BLANK = ce.BLANK_PREDICTED
WORDLESS = 1                    # the fixture's clip with no selected word (D20 has six)
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
    assert (f"same-codes --whole-read --a {part} --b {READS[arm]}-shard0.json --form core_crossfade --first 500 "
            f"--output-key {OUT_A}/a10-predicted-{arm}-r10-reproduces.json") in done[1]
    assert done[2].endswith(f"merge --report-only --shard-keys {part} --paired-with {READS[arm]}.json "
                            f"--output-key {OUT_A}/a10-predicted-{arm}-r10.json")


@pytest.mark.parametrize("arm", ["gains", "nogains"])
def test_a10s_cache_pin_is_the_hash_the_run_is_given_and_a_failed_pin_stops_it(tmp_path, arm):
    pins = tmp_path / "pins.txt"
    _, _, commands = phase_a(tmp_path, "a10", arm=arm, STUB_PINS=str(pins))
    pinned = dict(line.split() for line in pins.read_text().splitlines())
    run = run_line(commands)
    assert pinned[CACHES[arm]] == run.split("--word-cache-sha256 ")[1].split()[0]    # one hash, pinned and passed
    assert re.fullmatch(r"[0-9a-f]{64}", pinned[CACHES[arm]])
    assert f"--word-cache-key {CACHES[arm]} " in run
    # each stored object's pin stops the job before anything runs
    for key in (CACHES[arm], f"{READS[arm]}.json", f"{READS[arm]}-shard0.json"):
        code, out, commands = phase_a(tmp_path, "a10", arm=arm, STUB_PIN_FAIL=key)
        assert code == 1 and f"ALERT {key} differs from its pin" in out and not scripts_run(commands)


def test_the_real_pin_function_compares_the_stores_hash():
    line = next(x for x in PHASE_A.read_text().splitlines() if x.startswith("pin() "))
    die = next(x for x in PHASE_A.read_text().splitlines() if x.startswith("die() "))
    for stored, code in (("aaa", 0), ("bbb", 1), ("", 1)):
        run = subprocess.run(["bash", "-c", f'sha() {{ echo "{stored}"; }}\n{die}\n{line}\npin key aaa\necho after'],
                             capture_output=True, text=True)
        assert run.returncode == code and ("after" in run.stdout) == (code == 0)


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
    code, out, commands = phase_a(tmp_path, "a10", arm="nogains", STUB_SAME="1")
    assert code == 1 and "ALERT a10: the read differs" in out and "R20-PHASE-A-DONE 1" in out
    assert any(" same-codes " in c for c in commands) and not any(" merge " in c for c in commands)
    # the shard stored, the check not passed: a rerun runs nothing new and stops again
    code, out, commands = phase_a(tmp_path, "a10", arm="nogains", STUB_SAME="1",
                                  stored=[f"{OUT_A}/a10-predicted-nogains-r10-shard0.json"])
    assert code == 1 and "ALERT" in out and not any("round15_compose_eval.py" in c for c in commands)


def test_a10_alerts_on_other_main_form_hypotheses_and_still_merges(tmp_path):
    code, out, commands = phase_a(tmp_path, "a10", arm="gains", STUB_SAME="3")
    assert code == 0 and "ALERT a10: main-form hypotheses differ" in out and "R20-PHASE-A-DONE 0" in out
    assert any(" merge " in c for c in commands)
    assert "ALERT" not in phase_a(tmp_path, "a10", arm="gains")[1]
    assert "transformers" in out and "R20 a10 device cuda" in out


@pytest.mark.parametrize("match,message,reached", [
    ("round15_compose_eval.py run", "ALERT run a10-predicted-gains-r10", ("run",)),
    ("round15_compose_eval.py merge", "ALERT merge a10-predicted-gains-r10", ("run", "same-codes", "merge")),
])
def test_a10_stops_on_a_failed_run_or_merge(tmp_path, match, message, reached):
    code, out, commands = phase_a(tmp_path, "a10", arm="gains", STUB_FAIL_MATCH=match)
    assert code == 1 and message in out and "R20-PHASE-A-DONE 1" in out and "R20-PHASE-A-DONE 0" not in out
    assert f"R20 a10-predicted-gains-r10 {OUT_A}" not in out
    assert [next(w for w in ("same-codes", "merge", "run") if f" {w} " in c) for c in scripts_run(commands)] == list(reached)


def test_a10_stops_rather_than_print_a_blank_hash(tmp_path):
    code, out, _ = phase_a(tmp_path, "a10", arm="gains", STUB_SHA_EMPTY="1")
    assert code == 1 and "ALERT a10: no hash of" in out and "R20-PHASE-A-DONE 0" not in out
    assert "R20 a10-predicted-gains-r10 " not in out
    code, out, _ = phase_a(tmp_path, "a10", arm="gains")
    assert code == 0 and f"R20 a10-predicted-gains-r10 {OUT_A}/a10-predicted-gains-r10.json stubsha" in out


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
                    hypotheses={PREDICTED: hyp, FORM: "a b", "real": "a b", "blank": "a", BLANK: "a b c"})
    rows = [row(100, 150, "a b c d"), row(100, 50, "a b"), row(40, None, "a b"), row(50, 100, "a b c d")]
    out = ce.predicted_report(rows, [FORM], ["a b"] * 4, dict(round20=dict(retime="reference")))["predicted_timing"]
    assert out["report_only"] is True and out["timing"] == ce.PREDICTED_TIMING["reference"]
    f = out["forms"][PREDICTED]
    assert f["clips_with_words"] == 3 and f["wordless_clips"] == 1               # the word-less clip apart, not in the ratio
    assert f["seconds_over_reference"] == dict(pooled=1.2, mean_per_clip=pytest.approx(4 / 3, abs=1e-4),
                                               median_per_clip=1.5, shorter_than_reference=pytest.approx(.3333, abs=1e-4),
                                               over_twice_reference=0.)
    assert f["fits"] == dict(fitted=3)
    assert f["hypothesis_length_ratio"] == {PREDICTED: 1.5, FORM: 1., "real": 1., "blank": .5, BLANK: 1.5}
    many = [row(100, 201, "a b"), row(100, 200, "a b")]
    assert ce.predicted_report(many, [FORM], ["a b"] * 2, {})["predicted_timing"]["forms"][PREDICTED][
        "seconds_over_reference"]["over_twice_reference"] == .5                  # strictly above twice
    assert ce.predicted_report([dict(r, hypotheses={FORM: "a"}) for r in rows], [FORM], ["a"] * 4, {}) == {}
    with pytest.raises(SystemExit, match="no clip has a predicted plan"):
        ce.predicted_report([row(10, None, "a")], [FORM], ["a"], {})


# ---- the real path, from a word cache ------------------------------------------------------------------------------------

@pytest.fixture
def arms(tmp_path, stage2_store, likelihood_stub, monkeypatch):  # noqa: F811
    """The stored read and the a10 read of one cache (real runs on the synthetic store), one clip without a selected word,
    with spies on the a10 run's rendering and spotting."""
    import torch
    import sentence_word_spotting as sws
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
    selection = ce.stage1_selection

    def without_words(texts, vocab):
        details, words = selection(texts, vocab)
        details[WORDLESS], words[WORDLESS] = [], []
        return details, words
    monkeypatch.setattr(ce, "stage1_selection", without_words)
    stored, stored_shards = d20_report(tmp_path, store, sha, ssha, "e/stored", *cache)
    renders, spots = {}, []
    real_render, real_spot = ce.render_round20, sws.window_predictions

    def render(args, material, job, reference_seconds, *rest, **kw):
        state, plan = real_render(args, material, job, reference_seconds, *rest, **kw)
        renders[kw["sample_id"]] = dict(material=material, job=job, seconds=reference_seconds, order=kw["order"], state=state)
        return state, plan

    def spot(model, state, device, **kw):
        spots.append(len(state.timestamps))
        return real_spot(model, state, device, **kw)
    monkeypatch.setattr(ce, "render_round20", render)
    monkeypatch.setattr(sws, "window_predictions", spot)
    a10, a10_shards = d20_report(tmp_path, store, sha, ssha, "e/a10", *cache, "--predicted-duration",
                                 "--expect-duration-model", ".4,.5,18")
    monkeypatch.setattr(sws, "window_predictions", real_spot)
    monkeypatch.setattr(ce, "render_round20", real_render)
    return dict(store=store, sha=sha, ssha=ssha, texts=texts, cache=cache, stored=stored, stored_shard=stored_shards[0],
                a10=a10, a10_shard=a10_shards[0], renders=renders, spots=spots, labels=labels)


def sample(name, i):
    return f"round15-eval-{name}:{i}"


def test_predicted_renders_the_main_forms_words_in_their_order_at_the_predicted_seconds(arms):
    r, model = arms["renders"], arms["a10"]["round20"]["duration_model"]
    for i, text in enumerate(arms["texts"]):
        if i == WORDLESS:
            assert sample(PREDICTED, i) not in r and sample(FORM, i) not in r
            continue
        main, predicted = r[sample(FORM, i)], r[sample(PREDICTED, i)]
        assert predicted["material"] is main["material"] and predicted["job"] is main["job"]
        partner = r.get(sample(f"{FORM}/mismatched_words", i))             # none when the partner is the word-less clip
        assert partner is None or predicted["material"] is not partner["material"]
        assert predicted["order"] is None and main["order"] is None
        assert r[sample(f"{FORM}/shuffled", i)]["order"] is not None        # the spy does tell orders apart
        assert predicted["seconds"] == cv2.predicted_seconds(model, text)
        assert main["seconds"] == pytest.approx(arms["a10_shard"]["clips"][i]["frames_reference"] / 24)   # the clip's own
        assert predicted["seconds"] != pytest.approx(main["seconds"])


def test_the_word_less_clip_is_the_blank_at_predicted_length_and_the_form_is_text_only(arms):
    m, shard = arms["a10"], arms["a10_shard"]
    model = m["round20"]["duration_model"]
    for i, (row, text) in enumerate(zip(shard["clips"], arms["texts"])):
        frames = max(1, int(round(cv2.predicted_seconds(model, text) * 24)))
        assert row["frames_blank_predicted"] == frames
        assert row["motion"][PREDICTED]["seconds"] == pytest.approx(frames / 24)
        if i != WORDLESS:
            assert row["plans"][FORM]["predicted"]["frames"] == frames      # the two forms are the same length
    row = shard["clips"][WORDLESS]
    assert not row["entries"] and FORM not in row["plans"] and row["frames_reference"] != row["frames_blank_predicted"]
    assert row["hypotheses"][PREDICTED] == row["hypotheses"][BLANK]
    assert row["likelihood"][PREDICTED] == row["likelihood"][BLANK]
    assert row["motion"][PREDICTED]["hands_tracked"] == 0.                  # no real motion in it
    assert row["motion"][FORM]["seconds"] == pytest.approx(row["frames_reference"] / 24)
    assert BLANK not in row["motion"]
    timing = m["predicted_timing"]["forms"][PREDICTED]
    assert timing["wordless_clips"] == 1 and timing["clips_with_words"] == 3


def test_the_two_forms_timing_and_the_report(arms):
    m, shard = arms["a10"], arms["a10_shard"]
    assert m["settings"]["round20"]["predicted_duration"] and m["settings"]["round20"]["retime"] == "reference"
    assert m["round20"]["duration_model"]["slope"] == .5 and m["round20"]["duration_model"]["partition"] == "train"
    timed = [(row, text) for k, (row, text) in enumerate(zip(shard["clips"], arms["texts"])) if k != WORDLESS]
    for row, text in timed:
        plan = row["plans"][FORM]
        assert plan["frames"] == row["frames_reference"]                   # the main form: the reference's true length
        seconds = cv2.predicted_seconds(m["round20"]["duration_model"], text)
        assert plan["predicted"]["frames"] == int(round(seconds * 24))     # @predicted: from the text alone
        assert plan["predicted"]["fit"] == "fitted"                        # the same plan rule, no padding
        assert row["motion"][FORM]["seconds_over_reference"] == 1.
    # beside the main form: only @predicted and its blank; no controls at predicted length
    assert sorted(f for f in m["forms"] if ce.PREDICTED in f) == sorted([PREDICTED, BLANK])
    assert m["predicted_timing"]["report_only"] is True and "same words" in m["predicted_timing"]["timing"]
    f = m["predicted_timing"]["forms"][PREDICTED]
    true = np.array([r["frames_reference"] for r, _ in timed], float)
    pred = np.array([r["plans"][FORM]["predicted"]["frames"] for r, _ in timed], float)
    assert f["seconds_over_reference"]["pooled"] == round(pred.sum() / true.sum(), 4)
    assert set(f["hypothesis_length_ratio"]) == {PREDICTED, FORM, "real", "blank", BLANK}
    # BLEU, likelihood, motion and spotting of the form, paired with the main form and both blanks
    pairs = {"minus_blank", "minus_blank_at_predicted_length", "minus_mismatched", "minus_reference_duration"}
    assert pairs <= set(m["forms"][PREDICTED]) and pairs <= set(m["likelihood"]["forms"][PREDICTED])
    assert "bleu" in m["forms"][BLANK] and "mean" in m["likelihood"]["forms"][BLANK]
    assert {"shuffled", "mismatched_words"} <= {k[6:] for k in m["forms"][FORM] if k.startswith("minus_")}
    ratios = m["motion_ratio_to_real"][PREDICTED]
    assert ratios["wrist_speed"] > 0 and ratios["wrist_spread"] > 0 and ratios["seconds_over_reference"] > 0
    assert BLANK not in m["motion"]
    spotting = m["spotting"]["scores"][PREDICTED]
    assert {"recall", "order", "minus_mismatched_paired", "minus_reference_duration_paired"} <= set(spotting)
    assert "minus_reference_duration_paired" not in m["spotting"]["scores"][FORM]


def test_spotting_of_predicted_reads_the_predicted_form(arms):
    import sentence_word_spotting as sws
    from evaluate_text_to_motion import at_rate
    shard, spots = arms["a10_shard"], arms["spots"]
    n = len(shard["clips"])
    assert len(spots) == 2 * n
    reference = [r["frames_reference"] for r in shard["clips"]]
    predicted = [r["frames_blank_predicted"] for r in shard["clips"]]
    assert reference != predicted and spots == reference + predicted         # main forms, then each @predicted's own length
    labels = arms["labels"]
    index = {g: k for k, g in enumerate(labels)}
    for i, row in enumerate(shard["clips"]):
        if i == WORDLESS:
            continue
        state = at_rate(arms["renders"][sample(PREDICTED, i)]["state"], ce.FPS)
        pred = sws.window_predictions(None, state, "cpu", windows=ce.SPOT["windows"], hop=ce.SPOT["hop"], k=ce.SPOT["top_k"])
        own = sws.score_form([pred], [sws.sentence_labels(arms["texts"][i], set(labels))], index)[0]
        assert row["spotting"][PREDICTED]["own"] == json.loads(json.dumps(own))


def test_the_pairings_have_the_right_forms_and_signs(tmp_path, arms):
    """Merge alone, on a doctored shard where every form differs: each paired number from its own two forms."""
    from sentence_exposure_diagnostic import paired_bleu_difference
    shard = copy.deepcopy(arms["a10_shard"])
    refs = shard["texts"] = [f"{w} a b c d e f g h" for w in ("one", "two", "three", "four")]
    made = {FORM: lambda t: t, f"{FORM}/shuffled": lambda t: " ".join(t.split()[::-1]),
            PREDICTED: lambda t: " ".join(t.split()[:7]), "blank": lambda t: "q r s t u",
            BLANK: lambda t: " ".join(t.split()[:5])}
    lls = {FORM: -3., f"{FORM}/shuffled": -5., PREDICTED: -2., "blank": -7., BLANK: -6.}
    for row, text in zip(shard["clips"], refs):
        for name, f in made.items():
            row["hypotheses"][name] = f(text)
            row["likelihood"][name][0] = lls[name]
        row["spotting"][PREDICTED]["own"]["recall"] = 1.
        row["spotting"][FORM]["own"]["recall"] = .25
    arms["store"].put("e/doctored.json", json.dumps(shard).encode())
    merge(tmp_path, "e/doctored.json", "e/doctoredm.json")
    m = json.loads(arms["store"].get("e/doctoredm.json"))
    row = m["forms"][PREDICTED]

    def diff(other):
        return paired_bleu_difference([made[PREDICTED](t) for t in refs], refs, [made[other](t) for t in refs], refs, n=20)["difference"]
    assert row["minus_reference_duration"]["difference"] == diff(FORM) != diff(f"{FORM}/shuffled")
    assert row["minus_blank"]["difference"] == diff("blank")
    assert row["minus_blank_at_predicted_length"]["difference"] == diff(BLANK) != diff("blank")
    ll = m["likelihood"]["forms"][PREDICTED]
    assert ll["minus_reference_duration"]["difference"] == pytest.approx(1.)   # -2 - (-3): the sign is @predicted minus main
    assert ll["minus_blank"]["difference"] == pytest.approx(5.)
    assert ll["minus_blank_at_predicted_length"]["difference"] == pytest.approx(4.)
    assert m["spotting"]["scores"][PREDICTED]["minus_reference_duration_paired"]["difference"] == pytest.approx(.75)


def test_the_merge_can_be_report_only(tmp_path, arms, capsys):
    capsys.readouterr()
    merge(tmp_path, "e/a10-shard0.json", "e/ro.json", "--report-only")
    printed = capsys.readouterr().out
    m = json.loads(arms["store"].get("e/ro.json"))
    assert m["report_only"] is True and m["verdicts"] == {} and "VERDICT" not in printed and "report-only" in printed
    assert "report_only" not in arms["a10"] and set(arms["a10"]["verdicts"]) == {FORM}    # absent without the flag
    with pytest.raises(SystemExit):
        merge(tmp_path, "e/a10-shard0.json", "e/ro2.json", "--report-only", "--rule", "bleu:shuffled")
    assert not arms["store"].exists("e/ro2.json")


def test_forms_prints_no_rule_for_predicted_or_a_report_only_report(tmp_path, arms):
    m, shard = arms["a10"], arms["a10_shard"]
    merge(tmp_path, "e/a10-shard0.json", "e/ro.json", "--report-only")
    ro = json.loads(arms["store"].get("e/ro.json"))
    for a, form_a, b, form_b in ((m, PREDICTED, m, FORM), (m, FORM, m, PREDICTED), (ro, FORM, arms["stored"], FORM)):
        c = compare_forms(a, form_a, b, form_b, boot=20)
        assert c["report_only"] is True and "report-only" in c["note"]
        assert not {"both_improve", "rule", "spotting_guard_lower_bound"} & set(c)
        assert "difference" in c["bleu"] and "difference" in c["likelihood_minus_blank"]["difference"]
    # each likelihood is read against the blank at its own length
    c = compare_forms(m, PREDICTED, m, FORM, boot=20, shards_a=[shard], shards_b=[shard])
    assert c["likelihood_minus_blank"]["blanks"] == [BLANK, "blank"]
    assert c["spotting_guard"]["available"] and c["spotting_guard"]["word_source"] == ["generated"]
    # the adoption comparison itself keeps its rule
    plain = compare_forms(arms["stored"], FORM, arms["stored"], FORM, boot=20)
    assert "report_only" not in plain and "rule" in plain and "both_improve" in plain
    assert plain["likelihood_minus_blank"]["blanks"] == ["blank", "blank"]


def test_at_a_shorter_total_the_same_words_share_the_frames_by_the_same_rule():
    """What @predicted changes in the plan: only the total, against the reference's."""
    weights, ratio = [1.2, .8, 1.], .3
    full, short = fitted_plan(weights, ratio, 8.), fitted_plan(weights, ratio, 4.)
    assert full["frames"] == 192 and short["frames"] == 96
    assert all(abs(a / 2 - b) <= 1 for a, b in zip(full["cores"] + full["gaps"], short["cores"] + short["gaps"]))


def broken(report, clip, form, key):                               # one value, by the smallest amount
    out = copy.deepcopy(report)
    row = out["clips"][clip]["motion"][form]
    row[key] = 0. if row[key] is None else np.nextafter(row[key], np.inf).item()     # a blank's wrist speed is None
    return out


def test_the_main_form_reproduces_the_stored_read_and_one_value_breaks_it(tmp_path, arms):
    stored, part = arms["stored_shard"], arms["a10_shard"]
    assert same_codes_check(part, stored, FORM, first=4)["same"]
    for a, b in zip(part["clips"], stored["clips"]):               # the whole main form, not only what the check reads
        for k in (FORM, f"{FORM}/shuffled", f"{FORM}/mismatched_words", "real", "blank"):
            assert a["hypotheses"][k] == b["hypotheses"][k] and a["likelihood"][k] == b["likelihood"][k]
        for k in (FORM, f"{FORM}/shuffled", f"{FORM}/mismatched_words", "real"):
            assert a["motion"][k] == b["motion"][k]
        assert a["spotting"][FORM] == b["spotting"][FORM]
    for clip in range(4):
        with pytest.raises(SystemExit, match=rf"clip {clip} motion.core_crossfade.wrist_speed"):
            same_codes_check(part, broken(stored, clip, FORM, "wrist_speed"), FORM, first=4)
    with pytest.raises(SystemExit, match="motion.core_crossfade.seconds_over_reference"):
        same_codes_check(broken(part, 3, FORM, "seconds_over_reference"), stored, FORM, first=4)
    # the last clip counts: the check reads every clip, as the job asks (--first = the run's clips)
    assert same_codes_check(broken(part, 3, FORM, "wrist_speed"), stored, FORM, first=3)["same"]
    # @predicted moves freely: it has no stored read
    moved = copy.deepcopy(part)
    moved["clips"][0]["motion"][PREDICTED]["wrist_speed"] += 1.
    assert same_codes_check(moved, stored, FORM, first=4, whole_read=True)["same"]
    # another cache's words never pass as the stored read
    other = copy.deepcopy(part)
    other["settings"]["round20"]["word_cache"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="settings.round20.word_cache"):
        same_codes_check(other, stored, FORM, first=4)
    # the command the job calls exits non-zero
    root = ["--store", f"dir:{tmp_path / 'store'}", "--first", "4", "--whole-read"]
    assert compare_main(["same-codes", *root, "--a", "e/a10-shard0.json", "--b", "e/stored-shard0.json"]) == 0
    arms["store"].put("e/broken-shard0.json", json.dumps(broken(stored, 2, FORM, "wrist_spread")).encode())
    with pytest.raises(SystemExit, match="clip 2 motion.core_crossfade.wrist_spread") as e:
        compare_main(["same-codes", *root, "--a", "e/a10-shard0.json", "--b", "e/broken-shard0.json"])
    assert e.value.code not in (0, None)


def test_the_whole_read_also_holds_the_controls_and_plans_and_counts_hypotheses(tmp_path, arms):
    stored, part = arms["stored_shard"], arms["a10_shard"]
    out = same_codes_check(part, stored, FORM, first=4, whole_read=True)
    assert out["controls_compared"] == sorted([f"{FORM}/shuffled", f"{FORM}/mismatched_words", "real"])
    assert out["hypotheses_identical"] == {n: 4 for n in stored["clips"][0]["hypotheses"]}
    assert out["main_hypotheses_differ"] == [] and out["plans_compared"] is True
    # a control's motion, by the smallest amount: only the whole read sees it
    for form in (f"{FORM}/shuffled", f"{FORM}/mismatched_words", "real"):
        changed = broken(stored, 2, form, "wrist_speed")
        with pytest.raises(SystemExit, match=rf"clip 2 motion.{form}"):
            same_codes_check(part, changed, FORM, first=4, whole_read=True)
        assert same_codes_check(part, changed, FORM, first=4)["same"]
    # a plan's value, and a form the run lacks
    planned = copy.deepcopy(stored)
    planned["clips"][0]["plans"][FORM]["round20"]["cores"][0] += 1
    with pytest.raises(SystemExit, match="clip 0 plans"):
        same_codes_check(part, planned, FORM, first=4, whole_read=True)
    lacking = copy.deepcopy(part)
    lacking["clips"][0]["motion"].pop("real")
    with pytest.raises(SystemExit, match="clip 0 motion.real"):
        same_codes_check(lacking, stored, FORM, first=4, whole_read=True)
    # hypotheses differ: counted, the clip named, exit 3 once the report is written; a control's difference is only counted
    moved = copy.deepcopy(part)
    moved["clips"][3]["hypotheses"][FORM] += " other"
    moved["clips"][0]["hypotheses"]["real"] += " other"
    out = same_codes_check(moved, stored, FORM, first=4, whole_read=True)
    assert out["same"] and out["main_hypotheses_differ"] == [3]
    assert out["hypotheses_identical"][FORM] == 3 and out["hypotheses_identical"]["real"] == 3
    arms["store"].put("e/moved-shard0.json", json.dumps(moved).encode())
    root = ["--store", f"dir:{tmp_path / 'store'}", "--first", "4", "--whole-read", "--a", "e/moved-shard0.json",
            "--b", "e/stored-shard0.json"]
    assert compare_main(["same-codes", *root, "--output-key", "e/moved-check.json"]) == 3
    assert json.loads(arms["store"].get("e/moved-check.json"))["main_hypotheses_differ"] == [3]
    # without the flag: unchanged, whatever the hypotheses
    assert compare_main(["same-codes", *root[:-5], "--a", "e/moved-shard0.json", "--b", "e/stored-shard0.json"]) == 0


def test_a_smoke_is_held_to_everything_but_the_partners_that_follow_its_length(tmp_path, arms):
    d20_report(tmp_path, arms["store"], arms["sha"], arms["ssha"], "e/smoke", *arms["cache"], "--clips", "2",
               "--predicted-duration")
    smoke = json.loads(arms["store"].get("e/smoke-shard0.json"))
    out = same_codes_check(smoke, arms["stored_shard"], FORM, first=2, whole_read=True)
    assert out["controls_compared"] == sorted([f"{FORM}/shuffled", "real"])
    assert f"{FORM}/mismatched_words" not in out["hypotheses_identical"]
    with pytest.raises(SystemExit, match=rf"clip 1 motion.{FORM}/shuffled|clip 1 motion.real"):
        same_codes_check(smoke, broken(arms["stored_shard"], 1, "real", "wrist_speed"), FORM, first=2, whole_read=True)


def test_the_merge_reports_the_main_form_against_the_stored_read(tmp_path, arms):
    merge(tmp_path, "e/a10-shard0.json", "e/a10p.json", "--paired-with", "e/stored.json")
    m = json.loads(arms["store"].get("e/a10p.json"))
    assert m["comparators"]["paired_with"]["key"] == "e/stored.json"
    assert m["forms"][FORM]["minus_paired_core_crossfade"]["difference"] == 0
    assert m["likelihood"]["forms"][FORM]["minus_paired_core_crossfade"]["difference"] == 0


def a10_args(tmp_path, arms, *extra):
    from tests.test_round15_compose_eval import run_args
    args = [a for a in run_args(tmp_path, arms["sha"]) if a not in ("--checkpoint-key", "prior.pt")]
    args[args.index("core_crossfade,oracle_stage2")] = FORM
    return [*args, "--sentences-key", "sets/d20.json", "--sentences-sha256", arms["ssha"], *extra]


def test_a_wrong_fit_stops_the_run(tmp_path, arms):
    with pytest.raises(SystemExit, match="not the pinned train fit"):
        ce.main(a10_args(tmp_path, arms, *arms["cache"], "--predicted-duration", f"--expect-duration-model={PIN}",
                         "--output-key", "e/wrong.json"))
    assert not arms["store"].exists("e/wrong.json")


def test_a_wrong_cache_hash_stops_the_run(tmp_path, arms):
    """What a10's literal pin is held to: the stored object's own bytes."""
    cache = list(arms["cache"])
    cache[cache.index("--word-cache-sha256") + 1] = "0" * 64
    with pytest.raises(SystemExit, match="does not match --word-cache-sha256"):
        ce.main(a10_args(tmp_path, arms, *cache, "--predicted-duration", "--output-key", "e/wrong.json"))
    assert not arms["store"].exists("e/wrong.json")


def test_the_option_off_leaves_the_stored_shape(arms):
    """No --predicted-duration: no form, no spotting row, no report key, and the settings keys the stored reads have."""
    shard, m = arms["stored_shard"], arms["stored"]
    text = json.dumps(shard)
    assert "@predicted" not in text and '"predicted"' not in text and "duration_model" not in text
    assert "predicted_timing" not in m and "@predicted" not in json.dumps(m) and "report_only" not in m
    assert shard["settings"]["round20"]["predicted_duration"] is False
    assert list(shard["clips"][0]["spotting"]) == [FORM]
    assert "expect_duration_model" not in text and "frames_blank_predicted" not in text

    # the whole shard equals the a10 run's once the option's own rows are removed
    def without(s):
        out = copy.deepcopy(s)
        out.pop("seconds")
        out["round20"].pop("duration_model", None)
        out["settings"]["round20"]["predicted_duration"] = False
        for r in out["clips"]:
            r.pop("frames_blank_predicted", None)
            for block in ("hypotheses", "likelihood", "motion", "spotting"):
                r[block].pop(PREDICTED, None)
                r[block].pop(BLANK, None)
            r["plans"].get(FORM, {}).pop("predicted", None)
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
