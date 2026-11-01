"""Post-S2 A5: fresh word draws, the read across them and its rule, and the job on stubs. Offline: synthetic reports and stores only; no test starts a real job and nothing is read
from R2."""
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest
import torch

import scripts.round15_compose_eval as ce
import scripts.round20_word_cache as cw
import comp_a5_declared as dc5
import comp_a5_read as a5
import comp_a_lib as lib
import comp_a_read as rd
import round20_lib
from signtranslator.generation.motion_to_text import corpus_bleu
from tests.test_comp_a import PROVENANCE, REFS, SIDES, arm_report, oracle_args, cache_for_words, d20, recogniser, write_list, token_rows, part, generated_args  # noqa: F401
from tests.test_round15_stage2 import stage2_store, generator_store  # noqa: F401
from tests.test_round15_compositional import stubbed  # noqa: F401
from tests.test_round20_eval import likelihood_stub  # noqa: F401
from tests.test_round20_phase_b import on_stubs, scripts_run

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "runpod/fleet/comp_a5.sh"
CHAIN = ROOT / "runpod/fleet/comp_a5_chain.sh"
PREDICTED, TRUE = a5.FORMS["predicted"], a5.FORMS["true_length"]
O = "acquisition/2026-10-17-v1/youtube-asl-v1"
OUT = f"{O}/round20/comp-a5"
DONE = re.compile(r"^COMP-A5-DONE (0|1) phase=\S+ draw=\S+ smoke=(0|1)$", re.M)


# ---- draws -------------------------------------------------------------------------------------------------------------------------

def test_a_draws_seeds_are_no_other_runs():
    from signtranslator.generation.rerank import rerank_seeds
    stored = set(rerank_seeds(round20_lib.entry_seed("you"), 16))
    seen = stored
    for k in a5.DRAWS:
        seeds = set(rerank_seeds(round20_lib.draw_seed("you", k), 16))
        assert len(seeds) == 16 and not seeds & seen
        seen = seen | seeds
    assert round20_lib.draw_seed("you", 1) == round20_lib.draw_seed("you", 1) != round20_lib.draw_seed("it", 1)
    with pytest.raises(ValueError, match="numbered from 1"):
        round20_lib.draw_seed("you", 0)


def test_the_cache_makes_a_draw_on_its_own_seeds(d20, recogniser):  # noqa: F811
    c = d20
    cache_for_words(c, "cache/stored")
    cache_for_words(c, "cache/draw1", "--draw", "1", "--comp-a-job-file", a5.JOB_FILE)
    cache_for_words(c, "cache/draw2", "--draw", "2", "--comp-a-job-file", a5.JOB_FILE)
    stored, one, two = (part(c, f"cache/{n}/index.json") for n in ("stored", "draw1", "draw2"))
    assert "draw" not in stored["settings"] and "provenance" not in stored["settings"]
    assert one["settings"]["draw"] == 1 and two["settings"]["draw"] == 2
    from signtranslator.eval.provenance import code_digest
    assert one["settings"]["provenance"]["code_digest"] == code_digest(extra_files=(a5.JOB_FILE,))
    by = {r["entry"]: r for r in one["entries"]}
    for row in stored["entries"]:
        assert not set(row["seeds"]) & set(by[row["entry"]]["seeds"])
        assert by[row["entry"]]["seeds"][0] == round20_lib.draw_seed(row["entry"], 1)
    assert {r["entry"] for r in stored["entries"]} == set(by)


@pytest.mark.parametrize("flags,message", [(["--draw", "0"], "numbered from 1"), (["--draw", "1", "--sentences-key", "sets/c20.json", "--sentences-sha256", "0" * 64], "D20 only")])
def test_the_cache_refuses_a_bad_draw(d20, capsys, flags, message):  # noqa: F811
    c = d20
    args = ["run", "--store", f"dir:{c['tmp'] / 'store'}", "--prefix", "cache/x", "--mix-key", "mix.json", "--mix-sha256", c["sha"],
            "--sentences-key", "sets/d20.json", "--sentences-sha256", c["ssha"], "--checkpoint-key", "prior.pt", "--device", "cpu", *flags]
    with pytest.raises(SystemExit):
        cw.main(args)
    assert message in capsys.readouterr().err


def test_a_draws_new_entries_must_be_that_draw(d20, monkeypatch, recogniser):  # noqa: F811
    c = d20
    path, sha = write_list(c, token_rows(c, monkeypatch))
    extra = ("--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top", "10")
    main_sha = cache_for_words(c, "cache/d1", "--draw", "1")
    other = cache_for_words(c, "cache/x2", "--draw", "2", *extra)
    same = cache_for_words(c, "cache/x1", "--draw", "1", *extra)
    words = ("--word-cache-key", "cache/d1/index.json", "--word-cache-sha256", main_sha, "--word-cache-n", "4", "--core-margin", "0", *extra)
    with pytest.raises(SystemExit, match="draw"):
        ce.main([*generated_args(c, *words, "--word-cache-extra-key", "cache/x2/index.json", "--word-cache-extra-sha256", other), "--output-key", "e/x.json"])
    assert ce.main([*generated_args(c, *words, "--word-cache-extra-key", "cache/x1/index.json", "--word-cache-extra-sha256", same, "--comp-a-job-file", a5.JOB_FILE),
                    "--output-key", "e/ok.json"]) == 0
    shard = part(c, "e/ok.json")
    assert shard["settings"]["round20"]["word_cache"]["draw"] == 1
    from signtranslator.eval.provenance import code_digest
    assert shard["settings"]["round20"]["provenance"]["code_digest"] == code_digest(extra_files=(a5.JOB_FILE,))


# ---- a synthetic set of draws -------------------------------------------------------------------------------------------------------

def report(k, hyps=None, *, top=False, ll_shift=0., env=None, digest=None, words=None):
    settings = dict(word_cache=dict(sha256=f"cache{k}", draw=k, key="k"), provenance=dict(PROVENANCE, **({"environment": env} if env else {}), **({"code_digest": digest} if digest else {})))
    if top:
        settings.update(word_cache_extra=dict(sha256=f"extra{k}"), vocabulary_extra=dict(top=10))
    r = arm_report(settings=settings, extra=hyps or {})
    r["likelihood"]["per_form"] = {f: [x + (ll_shift if f in (PREDICTED, TRUE) else 0.) for x in v] for f, v in r["likelihood"]["per_form"].items()}
    r["motion_ratio_to_real"][PREDICTED].update(wrist_speed=1.5, hands_tracked=1.)
    return r


def shard_of(r, recall=.3):
    return [dict(settings=r["settings"], clips=[dict(clip=i, sign=s, spotting={f: dict(own=dict(recall=recall)) for f in (PREDICTED, TRUE)}) for i, s in enumerate(r["signs"])])]


def draws(top_recall=.35, ll_shift=.1, words_top=None):
    """Five draws where top10 reads the references in full on half the sentences, margin0 on a quarter."""
    pairs, shards = {}, {}
    for k in a5.DRAWS:
        m_hyps = [r if i < 3 else "some long words in" for i, r in enumerate(REFS)]
        t_hyps = [r if i < 6 else "some long words in" for i, r in enumerate(REFS)]
        m = report(k, {PREDICTED: m_hyps, TRUE: m_hyps})
        t = report(k, {PREDICTED: t_hyps, TRUE: t_hyps}, top=True, ll_shift=ll_shift)
        pairs[k], shards[k] = (m, t), (shard_of(m), shard_of(t, top_recall))
    return pairs, shards


def read(pairs=None, shards=None, boot=60):
    if pairs is None:
        pairs, shards = draws()
    return a5.read(pairs, shards, a9_sides=SIDES, boot=boot)


def test_the_read_reports_every_metric_at_both_lengths():
    out = read()
    assert out["report_only"] is True and out["declared"] is False and "adopted" not in json.dumps(out)
    assert out["draws"] == [1, 2, 3, 4, 5] and set(out["forms"]) == {"predicted", "true_length"}
    for form in out["forms"].values():
        for metric in ("bleu", "likelihood_minus_blank", "spotting"):
            row = form[metric]
            assert set(row) == {"per_draw", "mean", "between_draw_sd", "positive_draws", "two_way_ci95", "t_ci95"}, metric
            assert len(row["per_draw"]) == 5
        assert len(form["margin0_bleu"]) == 5 and len(form["gap_share"]["top10"]) == 5
    assert out["forms"]["predicted"]["blank"] == "blank@predicted" and out["forms"]["true_length"]["blank"] == "blank"
    assert out["rule"]["form"] == "predicted" and "DRAFT" in out["rule"]["note"] and "stored read" in out["stored_read_excluded"]


def test_the_paired_numbers_are_top10_minus_margin0_per_draw():
    out = read()["forms"]["predicted"]
    m_hyps = [r if i < 3 else "some long words in" for i, r in enumerate(REFS)]
    t_hyps = [r if i < 6 else "some long words in" for i, r in enumerate(REFS)]
    diff = corpus_bleu(t_hyps, REFS)["bleu"] - corpus_bleu(m_hyps, REFS)["bleu"]
    assert out["bleu"]["per_draw"] == pytest.approx([diff] * 5, abs=1e-3)
    assert out["likelihood_minus_blank"]["per_draw"] == pytest.approx([.1] * 5, abs=1e-6)
    assert out["spotting"]["per_draw"] == pytest.approx([.05] * 5, abs=1e-6)
    assert out["bleu"]["positive_draws"] == 5 and out["bleu"]["between_draw_sd"] == pytest.approx(0., abs=1e-6)


def test_the_likelihood_is_read_against_each_forms_own_length_blank():
    pairs, shards = draws(ll_shift=0.)
    for k in a5.DRAWS:
        t = pairs[k][1]
        t["likelihood"]["per_form"]["blank@predicted"] = [x - .5 for x in t["likelihood"]["per_form"]["blank@predicted"]]
    out = read(pairs, shards)["forms"]
    assert out["predicted"]["likelihood_minus_blank"]["mean"] == pytest.approx(.5, abs=1e-6)
    assert out["true_length"]["likelihood_minus_blank"]["mean"] == pytest.approx(0., abs=1e-6)


def stats_like(seed, k=5, n=40):
    from signtranslator.eval.s0_metrics import bleu_statistics
    refs = [f"alpha beta gamma delta epsilon zeta {i}" for i in range(n)]
    rng = np.random.default_rng(seed)
    out = []
    for d in range(k):
        out.append(bleu_statistics([" ".join(r.split()[:int(rng.integers(2, 7))]) for r in refs], refs))
    return np.array(out)


def test_the_two_way_bootstrap_spans_sentences_and_draws():
    a, b = stats_like(1), stats_like(2)
    row = a5.bleu_row(a, b, boot=200)
    from signtranslator.eval.s0_metrics import bleu_from_statistics
    expect = bleu_from_statistics(a.sum(1)) - bleu_from_statistics(b.sum(1))
    assert row["per_draw"] == pytest.approx(list(np.round(expect, 4)), abs=1e-4)
    assert row["mean"] == pytest.approx(float(expect.mean()), abs=1e-4) and row["between_draw_sd"] == pytest.approx(float(expect.std(ddof=1)), abs=1e-4)
    lo, hi = row["two_way_ci95"]
    assert lo <= row["mean"] <= hi
    # every draw the same: only the sentences move it, so the interval stays wider than nothing while the t interval collapses
    same = a5.bleu_row(np.repeat(a[:1], 5, 0), np.repeat(b[:1], 5, 0), boot=200)
    assert same["between_draw_sd"] == 0. and same["t_ci95"][0] == same["t_ci95"][1] and same["two_way_ci95"][1] > same["two_way_ci95"][0]
    # one sentence set, draws far apart: only the draws move it; each resample uses every draw's same sentences, and an arm against itself is exactly nothing
    assert a5.bleu_row(a, a, boot=50)["two_way_ci95"] == [0., 0.]


def test_draw_noise_alone_widens_the_two_way_interval():
    values = np.tile(np.array([[-1.], [0.], [1.], [2.], [3.]]), (1, 30))
    row = a5.mean_row(values, boot=400)
    assert row["mean"] == 1. and row["between_draw_sd"] == pytest.approx(float(np.std([-1, 0, 1, 2, 3], ddof=1)), abs=1e-4)
    assert row["two_way_ci95"][0] < .5 and row["two_way_ci95"][1] > 1.5        # sentences here are all alike: this width is the draws'
    from scipy import stats
    half = stats.t.ppf(.975, 4) * np.std([-1, 0, 1, 2, 3], ddof=1) / np.sqrt(5)
    assert row["t_ci95"] == [round(1 - half, 4), round(1 + half, 4)]


def test_absent_sentences_stay_out_of_a_mean():
    values = np.array([[1., np.nan, 3.]] * 5)
    assert a5.mean_row(values, boot=20)["per_draw"] == [2.] * 5


# ---- the rule ---------------------------------------------------------------------------------------------------------------------------

def metric(mean=.3, two=.1, t=.1, positive=5):
    return dict(mean=mean, two_way_ci95=[two, 9.], t_ci95=[t, 9.], positive_draws=positive, per_draw=[], between_draw_sd=.1)


def predicted(**kw):
    out = dict(bleu=metric(), likelihood_minus_blank=metric(), spotting=metric(two=-.01), motion=dict(top10_wrist=1.5, margin0_wrist=1.3, top10_hands=1., margin0_hands=1.))
    out.update(kw)
    return out


def test_all_checks_confirm():
    out = a5.decide(predicted(), 1.84)
    assert out["outcome"] == "confirmed" and all(out["checks"].values())


@pytest.mark.parametrize("kw,check,outcome", [
    (dict(bleu=metric(two=0.)), "bleu_two_way_lower_bound_above_0", "direction_only_not_confirmed"),
    (dict(bleu=metric(t=-.1)), "bleu_t_lower_bound_above_0", "direction_only_not_confirmed"),
    (dict(likelihood_minus_blank=metric(two=-.1)), "likelihood_two_way_lower_bound_above_0", "direction_only_not_confirmed"),
    (dict(likelihood_minus_blank=metric(t=0.)), "likelihood_t_lower_bound_above_0", "direction_only_not_confirmed"),
    (dict(spotting=metric(two=-.02)), "spotting_two_way_lower_bound_above_minus_0_02", "direction_only_not_confirmed"),
    (dict(bleu=metric(positive=3)), "bleu_positive_in_4_of_5", "not_confirmed"),
    (dict(likelihood_minus_blank=metric(positive=3)), "likelihood_positive_in_4_of_5", "not_confirmed"),
    (dict(motion=dict(top10_wrist=1.85, top10_hands=1., margin0_hands=1.)), "wrist_no_further_than_a9", "direction_only_not_confirmed"),
    (dict(motion=dict(top10_wrist=1.5, top10_hands=.89, margin0_hands=1.)), "hands_at_least_0_9_of_margin0", "direction_only_not_confirmed"),
])
def test_each_check_can_fail_alone(kw, check, outcome):
    out = a5.decide(predicted(**kw), 1.84)
    assert [k for k, v in out["checks"].items() if not v] == [check] and out["outcome"] == outcome


@pytest.mark.parametrize("kw", [dict(bleu=metric(mean=-.1, two=-.3, t=-.3, positive=4)), dict(likelihood_minus_blank=metric(mean=0., two=-.3, t=-.3))])
def test_a_mean_not_above_zero_is_not_even_direction(kw):
    assert a5.decide(predicted(**kw), 1.84)["outcome"] == "not_confirmed"


def test_a_missing_number_fails():
    out = a5.decide(predicted(motion=dict()), 1.84)
    assert out["checks"]["wrist_no_further_than_a9"] is False and out["checks"]["hands_at_least_0_9_of_margin0"] is False
    assert a5.decide(predicted(), None)["checks"]["wrist_no_further_than_a9"] is False


def test_four_of_five_is_the_bar():
    assert a5.decide(predicted(bleu=metric(positive=4)), 1.84)["outcome"] == "confirmed"


def test_only_predicted_timing_decides():
    pairs, shards = draws()
    for k in a5.DRAWS:                                           # true length collapses, predicted stands
        m, t = pairs[k]
        t["forms"][TRUE]["hypotheses"] = ["words"] * 12
    out = read(pairs, shards)
    assert out["forms"]["true_length"]["bleu"]["mean"] < 0 < out["forms"]["predicted"]["bleu"]["mean"]
    assert out["rule"]["checks"] == a5.decide(out["forms"]["predicted"], SIDES["forms"][rd.A9_FORM]["pooled_wrist_ratio"])["checks"]


def test_the_rule_text_says_what_does_not_confirm():
    for phrase in ("exactly five", "two-way bootstrap", "t interval", "at least 4 of the 5", "direction only", "stored read of A3 counts in no mean", "true-length forms are reported and decide nothing",
                   "rank-2 or rank-3", "best-of-4 or best-of-8", "subset of entries", "incomplete", "no draw is replaced, added or dropped", "no second A5", "not fresh", "no M1 claim"):
        assert phrase in a5.RULE, phrase
    assert "+0.343" in a5.KNOWN_AT_WRITING and "no fresh draw" in a5.KNOWN_AT_WRITING.lower()


# ---- what counts as a draw --------------------------------------------------------------------------------------------------------------

def bend(pairs, k, index, fn):
    bent = copy.deepcopy(pairs)
    fn(bent[k][index])
    return bent


@pytest.mark.parametrize("keep", [(1, 2, 3, 4), (1, 2, 3, 4, 5, 6), (2, 3, 4, 5), ()])
def test_anything_but_the_five_draws_is_incomplete(keep):
    pairs, _ = draws()
    if 6 in keep:
        pairs[6] = pairs[1]
    with pytest.raises(SystemExit, match="incomplete"):
        a5.check_draws({k: pairs[k] for k in keep})


@pytest.mark.parametrize("change,message", [
    (lambda r: r.pop("report_only"), "not report-only"),
    (lambda r: r.update(clips=499), "not all of D20"),
    (lambda r: r["settings"]["round20"]["sentences"].update(key="round20/c20.json"), "D20 only"),
    (lambda r: r["settings"]["round20"].update(hold_left=dict(sha256="x")), "undeclared differences|is not what"),
    (lambda r: r["settings"].update(core_margin=.1), "undeclared differences"),
    (lambda r: r["settings"]["round20"]["provenance"].update(code_digest="e" * 64), "other code or another environment"),
    (lambda r: r["settings"]["round20"]["provenance"].update(environment=dict(PROVENANCE["environment"], gpu="other")), "other code or another environment"),
])
@pytest.mark.parametrize("index", [0, 1])
def test_a_draws_two_reports_are_checked(change, message, index):
    pairs, _ = draws()
    with pytest.raises(SystemExit, match=message):
        a5.check_draws(bend(pairs, 3, index, change))


@pytest.mark.parametrize("change,message", [
    (lambda r: r["settings"]["round20"]["word_cache"].pop("draw"), "stored read's words count in no mean"),
    (lambda r: r["settings"]["round20"]["word_cache"].update(draw=2), "is draw 2"),
])
def test_a_draw_is_that_draws_words_on_both_arms(change, message):
    pairs, _ = draws()
    bent = bend(bend(pairs, 3, 0, change), 3, 1, change)
    with pytest.raises(SystemExit, match=message):
        a5.check_draws(bent)
    with pytest.raises(SystemExit, match=f"undeclared differences .*word_cache|{message}"):
        a5.check_draws(bend(pairs, 3, 1, change))


def test_a_margin0_must_be_named_by_its_settings():
    pairs, _ = draws()
    bent = bend(pairs, 2, 0, lambda r: r["settings"]["round20"].pop("comp_a_read"))
    with pytest.raises(SystemExit, match="is not what"):
        a5.check_draws(bent)


@pytest.mark.parametrize("change,message", [
    (lambda r: r["settings"]["round20"]["vocabulary_extra"].update(top=30), "not the top-10 arm|is not what"),
    (lambda r: r["settings"]["round20"].pop("word_cache_extra"), "is not what|no new entries"),
    (lambda r: r.update(round20=dict(vocabulary_extra=dict(fingerspelled_single_clips=["as"]))), "not the top-10 arm"),
])
def test_a_top10_is_the_top10(change, message):
    pairs, _ = draws()
    with pytest.raises(SystemExit, match=message):
        a5.check_draws(bend(pairs, 2, 1, change))


def test_two_draws_cannot_share_words():
    pairs, _ = draws()
    for index in (0, 1):
        pairs[2][index]["settings"]["round20"]["word_cache"]["sha256"] = "cache1"
    with pytest.raises(SystemExit, match="share a word cache"):
        a5.check_draws(pairs)
    pairs, _ = draws()
    pairs[2][1]["settings"]["round20"]["word_cache_extra"]["sha256"] = "extra1"
    with pytest.raises(SystemExit, match="share a word cache"):
        a5.check_draws(pairs)


def test_draws_share_code_and_packages_and_may_differ_in_gpu():
    pairs, _ = draws()
    for index in (0, 1):
        pairs[4][index]["settings"]["round20"]["provenance"]["environment"] = dict(PROVENANCE["environment"], gpu="RTX 6000 Ada")
    a5.check_draws(pairs)                                         # a GPU may differ between draws: each draw's words and reads are its own
    for field, value in (("code_digest", "e" * 64),):
        bent = copy.deepcopy(pairs)
        for index in (0, 1):
            bent[4][index]["settings"]["round20"]["provenance"][field] = value
        with pytest.raises(SystemExit, match="other code or packages"):
            a5.check_draws(bent)
    bent = copy.deepcopy(pairs)
    for index in (0, 1):
        bent[4][index]["settings"]["round20"]["provenance"]["environment"] = dict(PROVENANCE["environment"], transformers="9")
    with pytest.raises(SystemExit, match="other code or packages"):
        a5.check_draws(bent)


def test_other_sentences_stop_the_read():
    pairs, _ = draws()
    for index in (0, 1):
        pairs[3][index]["signs"] = list(reversed(pairs[3][index]["signs"]))
    with pytest.raises(SystemExit, match="other sentences"):
        a5.check_draws(pairs)


def test_spotting_must_be_there():
    pairs, shards = draws()
    for row in shards[2][1][0]["clips"]:
        row["spotting"].pop(PREDICTED)
    with pytest.raises(SystemExit, match="no per-clip spotting"):
        read(pairs, shards)


# ---- its own declaration --------------------------------------------------------------------------------------------------------------

def test_a5_has_its_own_declaration_and_digest(tmp_path):
    from signtranslator.eval.provenance import code_digest
    assert not (ROOT / dc5.RECORD).exists()                       # nothing is declared
    root = tmp_path / "bundle"
    (root / "docs").mkdir(parents=True)
    (root / "runpod/fleet").mkdir(parents=True)
    with pytest.raises(SystemExit, match="nothing shows a declaration"):
        dc5.check("abc1234", root)
    (root / "docs/NEXT_PHASE_PLAN.md").write_text("Plan. " + a5.RULE)
    record = dc5.make_record("abc1234")
    assert record["code_digest"] == code_digest(extra_files=(a5.JOB_FILE,)) != code_digest(extra_files=(lib.JOB_FILE,)) and set(record["rules"]) == {"a5"}
    (root / dc5.RECORD).write_text(json.dumps(record))
    assert dc5.check("abc1234", root) == "record"
    with pytest.raises(SystemExit, match="names abc1234"):
        dc5.check("abc1235", root)
    (root / dc5.RECORD).write_text(json.dumps(dict(record, code_digest="0" * 64)))
    with pytest.raises(SystemExit, match="declared for code"):
        dc5.check("abc1234", root)
    (root / dc5.RECORD).write_text(json.dumps(dict(record, rules=dict(a5="0"))))
    with pytest.raises(SystemExit, match="does not hold this rule"):
        dc5.check("abc1234", root)
    (root / dc5.RECORD).write_text(json.dumps(record))
    (root / "docs/NEXT_PHASE_PLAN.md").write_text("Plan.")
    with pytest.raises(SystemExit, match="plan does not hold"):
        dc5.check("abc1234", root)
    assert not (ROOT / "runpod/fleet/comp_a_declared.json").read_text().count(record["code_digest"])   # A1 to A4's record is untouched and apart


def test_the_a1_to_a4_record_still_reads_its_own_job():
    import comp_a_declared as dc
    assert dc.make_record("abc1234")["code_digest"] != dc5.make_record("abc1234")["code_digest"] and set(dc.make_record("abc1234")["rules"]) == set(rd.DRAFT_RULES)


# ---- the job on stubs: no real job runs ---------------------------------------------------------------------------------------------------

M = {f"M{k}_SHA": f"m{k}sha" for k in a5.DRAWS}
T = {f"T{k}_SHA": f"t{k}sha" for k in a5.DRAWS}
EVERY = [f"{OUT}/draw{k}-{a}.json" for k in a5.DRAWS for a in ("margin0", "top10")]


def job(tmp_path, phase, *, draw="1", smoke=False, gpu=True, stored=(), declared="abc1234", **env):
    e = dict(COMP_PHASE=phase, STUB_EXISTS=" ".join(stored), STUB_GATE="1", STUB_FAIL_SHARD="", STUB_HANG="")
    e.update(env)
    if draw:
        e["DRAW"] = draw
    if declared:
        e["COMP_A_DECLARED"] = declared
    return on_stubs(tmp_path, JOB.read_text(), e, smoke=smoke, gpu=gpu)


def one_done(out, code):
    lines = DONE.findall(out)
    assert len(lines) == 1 and lines[0][0] == str(code), out
    assert out.strip().splitlines()[-1].startswith("COMP-A5-DONE")


def run_line(commands):
    return next(c for c in commands if "round15_compose_eval.py run" in c)


def parsed(tmp_path, monkeypatch, line, name):
    seen = {}
    monkeypatch.setattr(ce, "run", lambda args: seen.update(vars(args)) or {})
    assert ce.main(line.split()[1:] + ["--store", f"dir:{tmp_path / ('s-' + name)}"]) == 0
    return seen


def test_the_jobs_parse_and_start_no_pod():
    for path in (JOB, CHAIN):
        subprocess.run(["bash", "-n", str(path)], check=True)
        for needle in ("runpodctl", "podCreate", "run_job_pod", "create-pod"):
            assert needle not in path.read_text()


def test_the_jobs_pins_are_the_committed_files():
    text = JOB.read_text()
    for name, path in (("TOKENSSHA", "reports/comp_a3_tokens.json"), ("A2SHA", "reports/comp_a2_left_arm.json")):
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == re.search(rf"{name}=([0-9a-f]{{64}})", text).group(1)
    assert "c20" not in "\n".join(x for x in text.splitlines() if "SET:" not in x and not x.lstrip().startswith("#")).lower()


def test_the_cache_phase_is_the_recipe_at_this_draw(tmp_path):
    code, out, commands = job(tmp_path, "cache", draw="3")
    assert code == 0 and "COMP-A5-DONE 0 phase=cache draw=3 smoke=0" in out
    runs = [c for c in scripts_run(commands) if "round20_word_cache.py run" in c]
    assert len(runs) == 4
    assert sorted(re.search(r"--shards 4 --shard (\d)", line).group(1) for line in runs) == ["0", "1", "2", "3"]
    for line in runs:
        assert f"--prefix {OUT}/draw3-cache " in line and "--draw 3" in line and "--samples 16 --choose 16" in line and "--no-decoder-gains" in line
        assert "--recognizer-run sign-recognizer-v3d" in line and "--vocabulary-extra" not in line and "--clips 500" in line
    assert any(f"round20_word_cache.py merge --prefix {OUT}/draw3-cache --shards 4" in c for c in commands)
    assert "--comp-a-job-file runpod/fleet/comp_a5.sh" in runs[0]


def test_a_failed_shard_stops_the_draw(tmp_path):
    code, out, commands = job(tmp_path, "cache", STUB_FAIL_SHARD="1")
    assert code == 1 and "a cache shard failed" in out and not any(" merge " in c for c in commands)
    one_done(out, 1)


def test_the_extra_phase_is_the_seven_new_entries_at_this_draw(tmp_path):
    code, _, commands = job(tmp_path, "extra", draw="4")
    assert code == 0
    run = next(c for c in commands if "round20_word_cache.py run" in c)
    assert f"--prefix {OUT}/draw4-extra " in run and "--draw 4" in run and "--vocabulary-extra-top 10" in run and "--samples 16 --choose 16" in run and "--no-decoder-gains" in run


@pytest.mark.parametrize("arm", ["margin0", "top10"])
def test_the_arms_are_a3s_lists_on_this_draws_words(tmp_path, monkeypatch, arm):
    stored = [f"{OUT}/draw2-cache/index.json", f"{OUT}/draw2-extra/index.json"] + ([f"{OUT}/draw2-margin0.json"] if arm == "top10" else [])
    code, out, commands = job(tmp_path, arm, draw="2", stored=stored, MARGIN0_SHA="m")
    assert code == 0
    one_done(out, 0)
    seen = parsed(tmp_path, monkeypatch, run_line(commands), arm)
    assert seen["word_cache_key"] == f"{OUT}/draw2-cache/index.json" and seen["word_cache_n"] == 16 and seen["core_margin"] == 0. and seen["predicted_duration"]
    assert seen["sentences_key"] == lib.D20_KEY and seen["clips"] == 500 and seen["likelihood"] and seen["comp_a_job_file"] == "runpod/fleet/comp_a5.sh"
    assert seen["expect_duration_model"] == "-0.083,0.3691,18289" and seen["variants"] == "core_crossfade" and seen["recognizer_run"] == "sign-recognizer-v2"
    assert seen["checkpoint_key"] is None and seen["arm_scale"] is None and seen["hold_left_file"] is None
    if arm == "margin0":
        assert seen["comp_a_read"] and not seen["vocabulary_extra_file"]
    else:
        assert seen["vocabulary_extra_top"] == 10 and seen["word_cache_extra_key"] == f"{OUT}/draw2-extra/index.json" and f"pin {OUT}/draw2-margin0.json" in commands
    assert "--report-only" in next(c for c in commands if "round15_compose_eval.py merge" in c)


def test_top10_needs_its_pinned_comparator(tmp_path):
    base = [f"{OUT}/draw1-cache/index.json", f"{OUT}/draw1-extra/index.json"]
    code, out, _ = job(tmp_path, "top10", stored=base)
    assert code == 1 and "comes first" in out
    code, out, _ = job(tmp_path, "top10", stored=[*base, f"{OUT}/draw1-margin0.json"])
    assert code == 1 and "give MARGIN0_SHA" in out
    code, out, commands = job(tmp_path, "top10", stored=[*base, f"{OUT}/draw1-margin0.json"], MARGIN0_SHA="m", STUB_PIN_FAIL=f"{OUT}/draw1-margin0.json")
    assert code == 1 and "differs from its pin" in out and "round15_compose_eval.py run" not in "\n".join(commands)
    one_done(out, 1)


def test_the_summary_needs_all_five_pinned(tmp_path):
    code, out, commands = job(tmp_path, "summary", draw=None, stored=EVERY, **M, **T)
    assert code == 0
    run = next(c for c in commands if "comp_a5_read.py" in c)
    for k in a5.DRAWS:
        assert f"--draw {k}={OUT}/draw{k}-margin0.json,{OUT}/draw{k}-top10.json" in run
        assert f"pin {OUT}/draw{k}-top10.json" in commands
    assert "--a9-sides-file reports/comp_a2_left_arm.json" in run and "--declared-commit abc1234" in run
    code, out, commands = job(tmp_path, "summary", draw=None, stored=EVERY[:-1], **M, **T)
    assert code == 1 and "incomplete: draw 5" in out and "comp_a5_read.py" not in "\n".join(commands)
    env = dict(M, **T)
    env.pop("T3_SHA")
    code, out, _ = job(tmp_path, "summary", draw=None, stored=EVERY, **env)
    assert code == 1 and "give M3_SHA and T3_SHA" in out
    code, out, _ = job(tmp_path, "summary", draw=None, stored=EVERY, STUB_PIN_FAIL=EVERY[4], **M, **T)
    assert code == 1 and "differs from its pin" in out
    code, out, commands = job(tmp_path, "summary", draw=None, stored=[f"{OUT}/a5-summary.json"])
    assert code == 0 and "stored:" in out and "comp_a5_read.py" not in "\n".join(commands)


@pytest.mark.parametrize("kwargs,message", [
    (dict(phase="cache", draw="6"), "DRAW must be"), (dict(phase="cache", draw="0"), "DRAW must be"), (dict(phase="margin0", draw=None), "DRAW must be"),
    (dict(phase="summary", draw="1"), "no DRAW"), (dict(phase="summary", draw=None, smoke=True), "no smoke"), (dict(phase="other"), "unknown COMP_PHASE"),
    (dict(phase="cache", SET="c20"), "D20 only"), (dict(phase="cache", SET="val1000"), "D20 only"), (dict(phase="cache", gpu=False), "needs a GPU"),
    (dict(phase="cache", declared=""), "nothing is declared yet"), (dict(phase="margin0"), "is not stored"), (dict(phase="cache", SMOKE="2"), "SMOKE must be"),
])
def test_refusals(tmp_path, kwargs, message):
    phase = kwargs.pop("phase")
    code, out, commands = job(tmp_path, phase, **kwargs)
    assert code == 1 and message in out and "ALERT" in out and "round15_compose_eval.py" not in "\n".join(commands) and "round20_word_cache.py" not in "\n".join(commands)
    one_done(out, 1)


def test_the_declaration_is_a5s_own(tmp_path):
    _, _, commands = job(tmp_path, "cache")
    assert scripts_run(commands)[0] == "scripts/comp_a5_declared.py abc1234"
    code, out, commands = job(tmp_path, "cache", STUB_FAIL_MATCH="comp_a5_declared.py")
    assert code == 1 and "does not declare A5" in out and len(scripts_run(commands)) == 1
    one_done(out, 1)


def test_a_smoke_is_five_clips_four_samples_and_needs_no_declaration(tmp_path):
    code, out, commands = job(tmp_path, "cache", smoke=True, gpu=False, declared="", CACHE_PROCS="1", SMOKE_STAMP="s1")
    assert code == 0 and "smoke=1" in out
    run = next(c for c in commands if "round20_word_cache.py run" in c)
    assert "--clips 5 " in run and "--samples 4 --choose 4" in run and f"{OUT}/smoke/s1/draw1-cache" in run
    code, _, commands = job(tmp_path, "margin0", smoke=True, gpu=False, declared="", SMOKE_STAMP="s1", stored=[f"{OUT}/smoke/s1/draw1-cache/index.json"])
    assert code == 0 and "--clips 5 " in run_line(commands) and "--word-cache-n 4" in run_line(commands)


@pytest.mark.parametrize("match,message", [("round15_compose_eval.py run", "ALERT run draw1-margin0"), ("round15_compose_eval.py merge", "ALERT merge draw1-margin0")])
def test_a_failed_step_stops_with_done_one(tmp_path, match, message):
    code, out, _ = job(tmp_path, "margin0", stored=[f"{OUT}/draw1-cache/index.json"], STUB_FAIL_MATCH=match)
    assert code == 1 and message in out and "COMP-A5 draw1-margin0" not in out
    one_done(out, 1)


@pytest.mark.parametrize("phase,stored,line", [("margin0", [f"{OUT}/draw1-cache/index.json"], "no hash"), ("cache", [], "no hash"), ("extra", [], "no hash")])
def test_no_hash_is_no_success(tmp_path, phase, stored, line):
    code, out, _ = job(tmp_path, phase, stored=stored, STUB_SHA_EMPTY="1")
    assert code == 1 and line in out and "COMP-A5 " not in out
    one_done(out, 1)


def test_a_failed_install_and_an_unexpected_end_say_done_one(tmp_path):
    base = dict(COMP_PHASE="cache", DRAW="1", COMP_A_DECLARED="abc1234", STUB_EXISTS="", STUB_GATE="1", STUB_FAIL_SHARD="", STUB_HANG="")
    code, out, commands = on_stubs(tmp_path, JOB.read_text().replace("uv pip install -q", "false uv pip install -q", 1), dict(base), smoke=False, gpu=True)
    assert code == 1 and "uv pip install failed" in out and not scripts_run(commands)
    one_done(out, 1)
    code, out, _ = on_stubs(tmp_path, JOB.read_text().replace("case $PHASE in\n  cache)\n", "exit 7\ncase $PHASE in\n  cache)\n", 1), dict(base), smoke=False, gpu=True)
    assert code == 1 and "stopped unexpectedly, exit 7" in out
    one_done(out, 1)


def test_the_job_is_write_once(tmp_path):
    code, out, commands = job(tmp_path, "cache", stored=[f"{OUT}/draw1-cache/index.json"])
    assert code == 0 and "stored:" in out and not any("round20_word_cache.py" in c for c in commands)
    code, out, commands = job(tmp_path, "margin0", stored=[f"{OUT}/draw1-cache/index.json", f"{OUT}/draw1-margin0.json"])
    assert code == 0 and "stored:" in out and "round15_compose_eval.py" not in "\n".join(commands)


def test_the_chain_runs_a_draw_on_one_pod_in_order():
    text = CHAIN.read_text()
    full = text.split(': > "$LOG"')[1]
    assert full.index("COMP_PHASE=$P") < full.index("hash_of draw$D-margin0") < full.index("COMP_PHASE=top10")
    assert 'for P in cache extra margin0; do run COMP_PHASE=$P DRAW=$D COMP_A_DECLARED="$DECLARED"' in full and 'MARGIN0_SHA="$M"' in full
    assert "SMOKE=1" in text.split(': > "$LOG"')[0] and "COMP-A5-CHAIN-DONE 1" in text


def test_the_bleu_interval_resamples_the_draws():
    """Sentences all alike, so resampling them moves nothing: any width in this interval is the draws'."""
    from signtranslator.eval.s0_metrics import bleu_statistics
    refs = ["alpha beta gamma delta epsilon zeta eta theta"] * 20
    a = np.array([bleu_statistics([" ".join(r.split()[:n]) for r in refs], refs) for n in (4, 5, 6, 7, 8)])
    b = np.repeat(bleu_statistics([" ".join(r.split()[:4]) for r in refs], refs)[None], 5, 0)
    row = a5.bleu_row(a, b, boot=200)
    assert row["between_draw_sd"] > 0 and row["two_way_ci95"][1] - row["two_way_ci95"][0] > row["between_draw_sd"]
