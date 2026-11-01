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
import comp_a5_guard as guard
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
JOBS = tuple(a5.JOB_FILE.split(","))
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
    assert one["settings"]["provenance"]["code_digest"] == code_digest(extra_files=JOBS)
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
    assert shard["settings"]["round20"]["provenance"]["code_digest"] == code_digest(extra_files=JOBS)


# ---- a smoke's cache covers what top10's few sentences need ---------------------------------------------------------------------------------

def only_through_a_contraction(c, monkeypatch):
    """Every sentence reads ``didn't`` and none reads ``not``; r10's vocabulary has ``not``; the list maps n't to it. The base selection never asks for ``not``."""
    from signtranslator.generation import compositional as comp
    load = round20_lib.load_sentence_set
    monkeypatch.setattr(round20_lib, "load_sentence_set", lambda *a, **k: [dict(r, text=f"{r['text']} didn't") for r in load(*a, **k)])
    vocabulary = comp.training_vocabulary
    monkeypatch.setattr(comp, "training_vocabulary", lambda *a, **k: dict(vocabulary(*a, **k), **{"not": 1.}))
    path, sha = write_list(c, token_rows(c, monkeypatch), negation_tokens=["didn't"])
    cover = ("--smoke-cover-file", str(path), "--smoke-cover-sha256", sha, "--smoke-cover-top", "10", "--clips", "4")
    extra = ("--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top", "10")
    return cover, extra


def top10_args(c, cache_sha, extra_sha, extra, *more):
    return generated_args(c, "--word-cache-key", "cache/main/index.json", "--word-cache-sha256", cache_sha, "--word-cache-n", "4", "--core-margin", "0", *extra,
                          "--word-cache-extra-key", "cache/x/index.json", "--word-cache-extra-sha256", extra_sha, "--clips", "4", "--comp-a-job-file", a5.JOB_FILE, *more)


def test_a_sentence_needing_not_only_through_a_contraction(d20, monkeypatch, recogniser):  # noqa: F811
    """The failure of the first A5 smokes, then the fix: top10 stops on the plain cache and runs on the covered one."""
    c = d20
    cover, extra = only_through_a_contraction(c, monkeypatch)
    extra_sha = cache_for_words(c, "cache/x", "--draw", "3", "--clips", "4", *extra)
    assert "not" not in {r["entry"] for r in part(c, "cache/x/index.json")["entries"]}             # the extra phase only generates new entries
    plain = cache_for_words(c, "cache/plain", "--draw", "3", "--clips", "4")
    plain_rows = {r["entry"]: r for r in part(c, "cache/plain/index.json")["entries"]}
    assert "not" not in plain_rows
    with pytest.raises(SystemExit, match=r"lacks 1 entries the sentences need: \['not'\]"):
        ce.main([*top10_args(c, plain, extra_sha, extra)[:-4], "--word-cache-key", "cache/plain/index.json", "--word-cache-sha256", plain, "--clips", "4", "--output-key", "e/lacks.json"])
    main_sha = cache_for_words(c, "cache/main", "--draw", "3", *cover)
    index = part(c, "cache/main/index.json")
    rows = {r["entry"]: r for r in index["entries"]}
    assert set(rows) == set(plain_rows) | {"not"} and index["settings"]["smoke_cover"]["top"] == 10 and index["settings"]["entries_total"] == len(rows)
    assert rows["not"]["seeds"][0] == round20_lib.draw_seed("not", 3)
    for entry, row in plain_rows.items():                                                          # nothing else moves: same seeds, same words
        assert rows[entry]["seeds"] == row["seeds"] and rows[entry]["chosen"] == {k: dict(v, state_key=v["state_key"].replace("cache/plain", "cache/main")) for k, v in row["chosen"].items()}
    assert ce.main([*top10_args(c, main_sha, extra_sha, extra), "--output-key", "e/top10.json"]) == 0
    shard = part(c, "e/top10.json")
    assert all("not" in row["stage1_words"] for row in shard["clips"]) and shard["round20"]["vocabulary_extra"]["tokens"]["didn't"] == 4


def test_the_default_cache_has_no_cover(d20, recogniser):  # noqa: F811
    c = d20
    cache_for_words(c, "cache/d", "--draw", "1")
    assert "smoke_cover" not in part(c, "cache/d/index.json")["settings"]


@pytest.mark.parametrize("flags,message", [
    (["--clips", "6"], "at most 5 clips"), (["--clips", "500"], "at most 5 clips"), ([], "at most 5 clips"),
    (["--clips", "5", "--smoke-cover-top", "7"], "go together"), (["--clips", "5", "--drop-sha"], "go together"), (["--clips", "5", "--drop-top"], "go together"),
    (["--clips", "5", "--sentences-key", "sets/c20.json", "--sentences-sha256", "0" * 64], "D20 only"),
    (["--clips", "5", "--vocabulary-extra-file", "x", "--vocabulary-extra-sha256", "0" * 64, "--vocabulary-extra-top", "10"], "main cache"),
])
def test_the_cover_is_a_smokes_and_nothing_elses(d20, capsys, flags, message):  # noqa: F811
    c = d20
    args = ["run", "--store", f"dir:{c['tmp'] / 'store'}", "--prefix", "cache/x", "--mix-key", "mix.json", "--mix-sha256", c["sha"], "--sentences-key", "sets/d20.json", "--sentences-sha256", c["ssha"],
            "--checkpoint-key", "prior.pt", "--device", "cpu", "--smoke-cover-file", "x", "--smoke-cover-sha256", "0" * 64, "--smoke-cover-top", "10", *flags]
    if "--drop-sha" in args:
        at = args.index("--smoke-cover-sha256")
        args = [a for k, a in enumerate(args) if k not in (at, at + 1) and a != "--drop-sha"]
    if "--drop-top" in args:
        at = args.index("--smoke-cover-top")
        args = [a for k, a in enumerate(args) if k not in (at, at + 1) and a != "--drop-top"]
    with pytest.raises(SystemExit):
        cw.main(args)
    assert message in capsys.readouterr().err


def test_the_cover_refuses_the_wrong_list_and_an_entry_outside_r10(d20, monkeypatch, recogniser):  # noqa: F811
    c = d20
    cover, _ = only_through_a_contraction(c, monkeypatch)
    wrong = list(cover)
    wrong[wrong.index("--smoke-cover-sha256") + 1] = "0" * 64
    with pytest.raises(SystemExit, match="expected"):
        cache_for_words(c, "cache/w", "--draw", "1", *wrong)
    from signtranslator.generation import compositional as comp
    with_not = comp.training_vocabulary
    monkeypatch.setattr(comp, "training_vocabulary", lambda *a, **k: {w: s for w, s in with_not(*a, **k).items() if w != "not"})
    with pytest.raises(SystemExit, match="outside r10's vocabulary"):
        cache_for_words(c, "cache/o", "--draw", "1", *cover)


def test_a_full_composition_refuses_a_smokes_covered_cache(d20, monkeypatch, recogniser):  # noqa: F811
    c = d20
    cover, _ = only_through_a_contraction(c, monkeypatch)
    sha = cache_for_words(c, "cache/main", "--draw", "1", *cover)
    words = ("--word-cache-key", "cache/main/index.json", "--word-cache-sha256", sha, "--word-cache-n", "4", "--core-margin", "0")
    with pytest.raises(SystemExit, match="is a smoke's"):
        ce.main([*generated_args(c, *words, "--clips", "6"), "--output-key", "e/full.json"])


def test_not_is_the_only_r10_entry_the_committed_list_maps_to():
    """A full cache needs no cover: at top 10 the map reaches r10 only at ``not``, and ``select_entries`` never reads the draw, so a draw selects what any other selects."""
    import inspect
    arm = lib.arm_of(lib.load_token_list(ROOT / "reports/comp_a3_tokens.json", hashlib.sha256((ROOT / "reports/comp_a3_tokens.json").read_bytes()).hexdigest()), 10)
    assert set(arm["token_map"].values()) - set(arm["entries"]) == {"not"}
    assert "draw" not in inspect.getsource(cw.select_entries)


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
    """Eight draws where top10 reads the references in full on half the sentences, margin0 on a quarter."""
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
    assert out["draws"] == list(range(1, 9)) and set(out["forms"]) == {"predicted", "true_length"}
    for form in out["forms"].values():
        for metric in ("bleu", "likelihood_minus_blank", "spotting"):
            row = form[metric]
            assert set(row) == {"per_draw", "mean", "between_draw_sd", "positive_draws", "two_way_ci95", "t_ci95"}, metric
            assert len(row["per_draw"]) == 8
        assert len(form["margin0_bleu"]) == 8 and len(form["gap_share"]["top10"]) == 8
    assert out["forms"]["predicted"]["blank"] == "blank@predicted" and out["forms"]["true_length"]["blank"] == "blank"
    assert out["rule"]["form"] == "predicted" and "DRAFT" in out["rule"]["note"] and "stored read" in out["stored_read_excluded"]


def test_the_paired_numbers_are_top10_minus_margin0_per_draw():
    out = read()["forms"]["predicted"]
    m_hyps = [r if i < 3 else "some long words in" for i, r in enumerate(REFS)]
    t_hyps = [r if i < 6 else "some long words in" for i, r in enumerate(REFS)]
    diff = corpus_bleu(t_hyps, REFS)["bleu"] - corpus_bleu(m_hyps, REFS)["bleu"]
    assert out["bleu"]["per_draw"] == pytest.approx([diff] * 8, abs=1e-3)
    assert out["likelihood_minus_blank"]["per_draw"] == pytest.approx([.1] * 8, abs=1e-6)
    assert out["spotting"]["per_draw"] == pytest.approx([.05] * 8, abs=1e-6)
    assert out["bleu"]["positive_draws"] == 8 and out["bleu"]["between_draw_sd"] == pytest.approx(0., abs=1e-6)


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

def metric(mean=.3, two=.1, t=.1, positive=8):
    return dict(mean=mean, two_way_ci95=[two, 9.], t_ci95=[t, 9.], positive_draws=positive, per_draw=[], between_draw_sd=.1)


def predicted(**kw):
    out = dict(bleu=metric(), likelihood_minus_blank=metric(), spotting=metric(two=-.01), motion=dict(top10_wrist=1.5, margin0_wrist=1.3, top10_hands=1., margin0_hands=1.))
    out.update(kw)
    return out


def test_all_checks_confirm():
    out = a5.decide(predicted(), 1.84)
    assert out["outcome"] == "confirmed" and all(out["checks"].values())


@pytest.mark.parametrize("kw,check,outcome", [
    (dict(bleu=metric(mean=.149)), "bleu_mean_at_least_0_15", "direction_only_not_confirmed"),
    (dict(bleu=metric(t=-.1)), "bleu_t_lower_bound_above_0", "direction_only_not_confirmed"),
    (dict(likelihood_minus_blank=metric(two=-.1)), "likelihood_two_way_lower_bound_above_0", "direction_only_not_confirmed"),
    (dict(likelihood_minus_blank=metric(t=0.)), "likelihood_t_lower_bound_above_0", "direction_only_not_confirmed"),
    (dict(spotting=metric(two=-.02)), "spotting_two_way_lower_bound_above_minus_0_02", "direction_only_not_confirmed"),
    (dict(bleu=metric(positive=6)), "bleu_positive_in_7_of_8", "not_confirmed"),
    (dict(likelihood_minus_blank=metric(positive=6)), "likelihood_positive_in_7_of_8", "not_confirmed"),
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


def test_seven_of_eight_and_the_floor_are_the_bar():
    assert a5.decide(predicted(bleu=metric(positive=7)), 1.84)["outcome"] == "confirmed"
    assert a5.decide(predicted(likelihood_minus_blank=metric(positive=7)), 1.84)["outcome"] == "confirmed"
    assert a5.decide(predicted(bleu=metric(mean=.15)), 1.84)["outcome"] == "confirmed"


def test_bleus_two_way_interval_is_reported_and_does_not_decide():
    out = a5.decide(predicted(bleu=metric(two=-1.)), 1.84)
    assert out["outcome"] == "confirmed" and not any("bleu_two_way" in k for k in out["checks"])
    assert "two_way_ci95" in read()["forms"]["predicted"]["bleu"] and "does not decide" in read()["bleu_two_way_note"]
    assert a5.BLEU_FLOOR == .15 and a5.MIN_POSITIVE == 7 and a5.DRAWS == (1, 2, 3, 4, 5, 6, 7, 8)


def test_only_predicted_timing_decides():
    pairs, shards = draws()
    for k in a5.DRAWS:                                           # true length collapses, predicted stands
        m, t = pairs[k]
        t["forms"][TRUE]["hypotheses"] = ["words"] * 12
    out = read(pairs, shards)
    assert out["forms"]["true_length"]["bleu"]["mean"] < 0 < out["forms"]["predicted"]["bleu"]["mean"]
    assert out["rule"]["checks"] == a5.decide(out["forms"]["predicted"], SIDES["forms"][rd.A9_FORM]["pooled_wrist_ratio"])["checks"]


RULE_AS_GIVEN = ("Primary comparison: A3 generated top10 against margin0, paired within each of exactly eight pre-registered fresh word draws (draw 1 to 8: new seeds for every entry and the same recipe, best of 16 by v3d, no gains), on D20 at predicted timing, report only. "
                 "Confirmed on D20's sentences only if all hold: the mean paired BLEU's 95% lower bound above 0 on the t interval over the eight draws, and that mean at least +0.15 (about 5 points of D20's gap); the mean paired likelihood minus blank@predicted's 95% lower bound above 0 on both the two-way bootstrap over sentences and draws and the t interval; v2 spotting's two-way lower bound above -0.02; BLEU and likelihood each above 0 in at least 7 of the 8 draws; mean wrist no further from 1.0x real than a9@predicted's; mean hands at least 0.9x margin0's. "
                 "BLEU's two-way interval is reported beside its t interval and does not decide, because the sentence set is fixed and this read tests word draws, not sentences. Mean BLEU and likelihood above 0 with at least 7 of 8 draws positive on each, but a bound or the floor not met, is direction only, which is not confirmed. Anything else is not confirmed. A true effect can still fail this rule, so not confirmed is not evidence that top10 does not beat margin0; the failed check is reported. "
                 "What does not confirm: the stored read of A3 counts in no mean; true-length forms are reported and decide nothing; rank-2 or rank-3 samples, best-of-4 or best-of-8 choices and any subset of entries are not draws; fewer than eight finished draws is incomplete, not a smaller test; no draw is replaced, added or dropped after any result. "
                 "A job failure before both of a draw's arms are stored reruns both arms on the same seeds and nothing else; a draw with both arms stored is final. Each arm's hash goes in the ledger when its draw ends, and no per-draw number is read until the summary. No second A5 with more draws on D20 follows a not-confirmed. "
                 "Confirmed means only that top10 beats margin0 across word draws on D20's own sentences: D20 chose this arm, so its sentences are not fresh. It is no M1 claim, no adoption and no release bar.")


def test_the_rule_is_the_text_as_given():
    assert a5.RULE == RULE_AS_GIVEN
    assert "+0.343" in a5.KNOWN_AT_WRITING and "no fresh draw" in a5.KNOWN_AT_WRITING.lower()
    assert a5.KNOWN_AT_WRITING.endswith("Stored likelihood interval [+0.0015, +0.0235] and spotting [+0.0062, +0.0248] were known. The stored margin0 ran on an L40S and top10 on an RTX 6000 Ada, a mix this read refuses. The +0.15 floor and eight draws were set after the stored +0.343 was known.")


# ---- what counts as a draw --------------------------------------------------------------------------------------------------------------

def bend(pairs, k, index, fn):
    bent = copy.deepcopy(pairs)
    fn(bent[k][index])
    return bent


@pytest.mark.parametrize("keep", [(1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 6, 7, 8, 9), (2, 3, 4, 5, 6, 7, 8), (1, 2, 3, 4, 5), ()])
def test_anything_but_the_eight_draws_is_incomplete(keep):
    pairs, _ = draws()
    if 9 in keep:
        pairs[9] = pairs[1]
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
    root = tmp_path / "bundle"
    (root / "docs").mkdir(parents=True)
    (root / "runpod/fleet").mkdir(parents=True)
    with pytest.raises(SystemExit, match="nothing shows a declaration"):
        dc5.check("abc1234", root)
    (root / "docs/NEXT_PHASE_PLAN.md").write_text("Plan. " + a5.RULE)
    record = dc5.make_record("abc1234")
    assert record["code_digest"] == code_digest(extra_files=JOBS) != code_digest(extra_files=(lib.JOB_FILE,)) and set(record["rules"]) == {"a5"}
    assert JOBS == ("runpod/fleet/comp_a5.sh", "runpod/fleet/comp_a5_chain.sh") and code_digest(extra_files=JOBS) != code_digest(extra_files=JOBS[:1])       # the chain is in the digest
    (root / "runpod/fleet/comp_a_declared.json").write_text(json.dumps(record))         # A1 to A4's record declares no A5
    with pytest.raises(SystemExit, match="nothing shows a declaration"):
        dc5.check("abc1234", root)
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
EVERY = [f"{OUT}/draw{k}-attempt1-{a}.json" for k in a5.DRAWS for a in ("margin0", "top10")]


def caches(k=1, attempt=1):
    return [f"{OUT}/draw{k}-cache/attempt{attempt}/index.json", f"{OUT}/draw{k}-extra/attempt{attempt}/index.json"]


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
    assert re.search(r"uv pip install -q 'transformers==(\S+)' 'tokenizers==(\S+)' sentencepiece", text).groups() == (a5.PINNED["transformers"], a5.PINNED["tokenizers"])
    assert f"JOBFILE={a5.JOB_FILE}\n" in text


def test_the_cache_phase_is_the_recipe_at_this_draw(tmp_path):
    code, out, commands = job(tmp_path, "cache", draw="3")
    assert code == 0 and "COMP-A5-DONE 0 phase=cache draw=3 smoke=0" in out
    runs = [c for c in scripts_run(commands) if "round20_word_cache.py run" in c]
    assert len(runs) == 4
    assert sorted(re.search(r"--shards 4 --shard (\d)", line).group(1) for line in runs) == ["0", "1", "2", "3"]
    for line in runs:
        assert f"--prefix {OUT}/draw3-cache/attempt1 " in line and "--draw 3" in line and "--samples 16 --choose 16" in line and "--no-decoder-gains" in line
        assert "--recognizer-run sign-recognizer-v3d" in line and "--vocabulary-extra" not in line and "--clips 500" in line
    assert any(f"round20_word_cache.py merge --prefix {OUT}/draw3-cache/attempt1 --shards 4" in c for c in commands)
    assert f"--comp-a-job-file {a5.JOB_FILE} " in runs[0]


def test_only_a_smokes_cache_phase_is_covered(tmp_path):
    text = JOB.read_text()
    sha = re.search(r"TOKENSSHA=([0-9a-f]{64})", text).group(1)
    cover = f"--smoke-cover-file reports/comp_a3_tokens.json --smoke-cover-sha256 {sha} --smoke-cover-top 10"
    _, _, commands = job(tmp_path, "cache", smoke=True, gpu=False, declared="", CACHE_PROCS="1", SMOKE_STAMP="s1")
    assert cover in next(c for c in commands if "round20_word_cache.py run" in c)
    _, _, commands = job(tmp_path, "extra", smoke=True, gpu=False, declared="", SMOKE_STAMP="s1")
    assert "--smoke-cover" not in "\n".join(commands)
    _, _, commands = job(tmp_path, "cache", draw="3")
    runs = [c for c in commands if "round20_word_cache.py run" in c]
    assert len(runs) == 4 and "--smoke-cover" not in "\n".join(commands)


def test_a_failed_shard_stops_the_draw(tmp_path):
    code, out, commands = job(tmp_path, "cache", STUB_FAIL_SHARD="1")
    assert code == 1 and "a cache shard failed" in out and not any(" merge " in c for c in commands)
    one_done(out, 1)


def test_the_extra_phase_is_the_seven_new_entries_at_this_draw(tmp_path):
    code, _, commands = job(tmp_path, "extra", draw="4")
    assert code == 0
    run = next(c for c in commands if "round20_word_cache.py run" in c)
    assert f"--prefix {OUT}/draw4-extra/attempt1 " in run and "--draw 4" in run and "--vocabulary-extra-top 10" in run and "--samples 16 --choose 16" in run and "--no-decoder-gains" in run


@pytest.mark.parametrize("arm", ["margin0", "top10"])
def test_the_arms_are_a3s_lists_on_this_draws_words(tmp_path, monkeypatch, arm):
    stored = caches(2) + ([f"{OUT}/draw2-attempt1-margin0.json"] if arm == "top10" else [])
    code, out, commands = job(tmp_path, arm, draw="2", stored=stored, MARGIN0_SHA="m")
    assert code == 0
    one_done(out, 0)
    seen = parsed(tmp_path, monkeypatch, run_line(commands), arm)
    assert seen["word_cache_key"] == caches(2)[0] and seen["word_cache_n"] == 16 and seen["core_margin"] == 0. and seen["predicted_duration"]
    assert seen["sentences_key"] == lib.D20_KEY and seen["clips"] == 500 and seen["likelihood"] and seen["comp_a_job_file"] == a5.JOB_FILE
    assert seen["expect_duration_model"] == "-0.083,0.3691,18289" and seen["variants"] == "core_crossfade" and seen["recognizer_run"] == "sign-recognizer-v2"
    assert seen["checkpoint_key"] is None and seen["arm_scale"] is None and seen["hold_left_file"] is None
    if arm == "margin0":
        assert seen["comp_a_read"] and not seen["vocabulary_extra_file"]
    else:
        assert seen["vocabulary_extra_top"] == 10 and seen["word_cache_extra_key"] == caches(2)[1] and f"pin {OUT}/draw2-attempt1-margin0.json" in commands
    assert "--report-only" in next(c for c in commands if "round15_compose_eval.py merge" in c)


def test_top10_needs_its_pinned_comparator(tmp_path):
    base, m0 = caches(), f"{OUT}/draw1-attempt1-margin0.json"
    code, out, _ = job(tmp_path, "top10", stored=base)
    assert code == 1 and "comes first" in out
    code, out, _ = job(tmp_path, "top10", stored=[*base, m0])
    assert code == 1 and "give MARGIN0_SHA" in out
    code, out, commands = job(tmp_path, "top10", stored=[*base, m0], MARGIN0_SHA="m", STUB_PIN_FAIL=m0)
    assert code == 1 and "differs from its pin" in out and "round15_compose_eval.py run" not in "\n".join(commands)
    one_done(out, 1)


# ---- attempts ---------------------------------------------------------------------------------------------------------------------------

def test_an_attempt_names_its_arms_keys_and_prints_its_hash(tmp_path):
    code, out, commands = job(tmp_path, "margin0", stored=caches(), ATTEMPT="2")
    assert code == 0 and f"COMP-A5 draw1-attempt2-margin0 {OUT}/draw1-attempt2-margin0.json stubsha" in out
    assert f"--output-key {OUT}/draw1-attempt2-margin0-shard0.json" in run_line(commands)
    assert f"--output-key {OUT}/draw1-attempt2-margin0.json" in next(c for c in commands if "round15_compose_eval.py merge" in c)
    assert f"scripts/comp_a5_guard.py started {OUT}/draw1-attempt2-margin0.started" in commands
    assert commands.index(f"scripts/comp_a5_guard.py started {OUT}/draw1-attempt2-margin0.started") < commands.index(run_line(commands))


@pytest.mark.parametrize("attempt", ["0", "6", "x", ""])
def test_a_bad_attempt_is_refused(tmp_path, attempt):
    code, out, commands = job(tmp_path, "cache", ATTEMPT=attempt)
    if attempt == "":                                             # empty is the default, attempt 1
        assert code == 0 and f"--prefix {OUT}/draw1-cache/attempt1 " in "\n".join(commands)
        return
    assert code == 1 and "ATTEMPT must be" in out and "round20_word_cache.py" not in "\n".join(commands)
    one_done(out, 1)


@pytest.mark.parametrize("phase", ["cache", "extra", "margin0", "top10"])
@pytest.mark.parametrize("attempt,final", [("2", 1), ("3", 1), ("1", 2)])
def test_a_finished_pair_is_final(tmp_path, phase, attempt, final):
    stored = [*caches(), f"{OUT}/draw1-attempt{final}-margin0.json", f"{OUT}/draw1-attempt{final}-top10.json"]
    code, out, commands = job(tmp_path, phase, stored=stored, ATTEMPT=attempt, MARGIN0_SHA="m")
    assert code == 1 and f"draw 1 attempt {final} has both arms: a finished pair is final" in out
    assert "round15_compose_eval.py" not in "\n".join(commands) and "round20_word_cache.py" not in "\n".join(commands)
    one_done(out, 1)


def test_another_draws_finished_pair_stops_nothing(tmp_path):
    code, _, _ = job(tmp_path, "margin0", stored=[*caches(), f"{OUT}/draw2-attempt1-top10.json"], ATTEMPT="2")
    assert code == 0


def test_an_unfinished_attempt_reruns_both_arms_on_the_same_words(tmp_path):
    left = [*caches(), f"{OUT}/draw1-attempt1-margin0.json", f"{OUT}/draw1-attempt1-top10.started"]
    code, _, commands = job(tmp_path, "margin0", stored=left, ATTEMPT="2")
    assert code == 0 and f"--output-key {OUT}/draw1-attempt2-margin0-shard0.json" in run_line(commands) and f"--word-cache-key {caches()[0]} " in run_line(commands)
    code, _, commands = job(tmp_path, "top10", stored=[*left, f"{OUT}/draw1-attempt2-margin0.json"], ATTEMPT="2", MARGIN0_SHA="m")
    assert code == 0 and f"--output-key {OUT}/draw1-attempt2-top10-shard0.json" in run_line(commands) and f"--word-cache-extra-key {caches()[1]} " in run_line(commands)
    assert f"pin {OUT}/draw1-attempt2-margin0.json" in commands and f"scripts/comp_a5_guard.py same-pod {OUT}/draw1-attempt2-margin0.json" in commands


@pytest.mark.parametrize("arm", ["margin0", "top10"])
def test_an_arm_that_began_and_did_not_finish_never_reruns_under_its_attempt(tmp_path, arm):
    stored = [*caches(), f"{OUT}/draw1-attempt1-{arm}.started"] + ([f"{OUT}/draw1-attempt1-margin0.json"] if arm == "top10" else [])
    code, out, commands = job(tmp_path, arm, stored=stored, MARGIN0_SHA="m")
    assert code == 1 and "started and did not finish: rerun both arms under ATTEMPT=2" in out and "round15_compose_eval.py" not in "\n".join(commands)
    one_done(out, 1)


def test_a_failed_start_mark_stops_the_arm(tmp_path):
    code, out, commands = job(tmp_path, "margin0", stored=caches(), STUB_FAIL_MATCH="comp_a5_guard.py started")
    assert code == 1 and "cannot mark its start" in out and "round15_compose_eval.py" not in "\n".join(commands)


@pytest.mark.parametrize("phase,index", [("cache", 0), ("extra", 1)])
def test_a_complete_stored_cache_is_reused_by_a_later_attempt(tmp_path, phase, index):
    code, out, commands = job(tmp_path, phase, stored=[caches()[index]], ATTEMPT="3")
    assert code == 0 and f"stored: {caches()[index]}" in out and f"COMP-A5 draw1-{phase} {caches()[index]} stubsha" in out and "round20_word_cache.py" not in "\n".join(commands)


def test_a_later_attempt_finds_an_earlier_attempts_words(tmp_path):
    code, _, commands = job(tmp_path, "margin0", stored=[f"{OUT}/draw1-cache/attempt2/index.json"], ATTEMPT="3")
    assert code == 0 and f"--word-cache-key {OUT}/draw1-cache/attempt2/index.json " in run_line(commands)


def test_a_partial_cache_stays_in_its_attempts_folder(tmp_path):
    code, _, commands = job(tmp_path, "cache", stored=[f"{OUT}/draw1-cache/attempt2/shard-0-of-4.json"], ATTEMPT="2")
    runs = [c for c in commands if "round20_word_cache.py run" in c]
    assert code == 0 and len(runs) == 3 and all(f"--prefix {OUT}/draw1-cache/attempt2 " in r for r in runs)      # its own attempt resumes
    code, _, commands = job(tmp_path, "cache", stored=[f"{OUT}/draw1-cache/attempt1/shard-0-of-4.json"], ATTEMPT="2")
    assert code == 0 and len([c for c in commands if "round20_word_cache.py run" in c]) == 4                    # another attempt's part is not a complete cache


# ---- within a draw, one pod's code and environment --------------------------------------------------------------------------------------------

def test_top10_compares_margin0_with_this_pod_before_it_runs(tmp_path):
    stored = [*caches(), f"{OUT}/draw1-attempt1-margin0.json"]
    code, _, commands = job(tmp_path, "top10", stored=stored, MARGIN0_SHA="m")
    guard = f"scripts/comp_a5_guard.py same-pod {OUT}/draw1-attempt1-margin0.json"
    assert code == 0 and commands.index(guard) < commands.index(f"scripts/comp_a5_guard.py started {OUT}/draw1-attempt1-top10.started") < commands.index(run_line(commands))
    code, out, commands = job(tmp_path, "top10", stored=stored, MARGIN0_SHA="m", STUB_FAIL_MATCH="comp_a5_guard.py same-pod")
    assert code == 1 and "margin0 came from other code or another environment; rerun both arms under ATTEMPT=2" in out and "round15_compose_eval.py" not in "\n".join(commands)
    assert "scripts/comp_a5_guard.py started" not in "\n".join(commands)
    one_done(out, 1)


def test_a_stored_top10_needs_no_comparison_and_still_prints_its_hash(tmp_path):
    key = f"{OUT}/draw1-attempt1-top10.json"
    code, out, commands = job(tmp_path, "top10", stored=[*caches(), f"{OUT}/draw1-attempt1-margin0.json", key], MARGIN0_SHA="m", STUB_FAIL_MATCH="comp_a5_guard.py same-pod")
    assert code == 0 and f"COMP-A5 draw1-attempt1-top10 {key} stubsha" in out and "round15_compose_eval.py" not in "\n".join(commands)


def test_a_smoke_compares_too(tmp_path):
    stored = [f"{OUT}/smoke/s1/draw1-cache/attempt1/index.json", f"{OUT}/smoke/s1/draw1-extra/attempt1/index.json", f"{OUT}/smoke/s1/draw1-attempt1-margin0.json"]
    code, out, commands = job(tmp_path, "top10", smoke=True, gpu=False, declared="", SMOKE_STAMP="s1", stored=stored, STUB_FAIL_MATCH="comp_a5_guard.py same-pod")
    assert code == 1 and "other code or another environment" in out


def pod(monkeypatch, **env):
    from signtranslator.eval import provenance
    base = dict(a5.PINNED, gpu="NVIDIA L40S")
    monkeypatch.setattr(provenance, "environment", lambda: dict(base, **env))


def test_the_environment_guard_passes_the_pins_with_any_gpu(monkeypatch):
    pod(monkeypatch)
    guard.check_environment()
    pod(monkeypatch, gpu="RTX 6000 Ada")
    guard.check_environment()
    assert a5.PINNED == dict(python="3.12.16", torch="2.13.0+cu130", transformers="4.57.6", tokenizers="0.22.2")


@pytest.mark.parametrize("package", ["python", "torch", "transformers", "tokenizers"])
@pytest.mark.parametrize("value", ["9.9", None])
def test_the_environment_guard_stops_on_any_other_package(monkeypatch, package, value):
    pod(monkeypatch, **{package: value})
    with pytest.raises(SystemExit, match=f"not the pinned .*{package}"):
        guard.main(["env"])


def test_the_job_stops_at_start_on_another_environment(tmp_path):
    code, out, commands = job(tmp_path, "cache", STUB_FAIL_MATCH="comp_a5_guard.py env")
    assert code == 1 and "not the pinned ones" in out and scripts_run(commands) == ["scripts/comp_a5_guard.py env"]
    one_done(out, 1)
    code, out, _ = job(tmp_path, "cache", smoke=True, gpu=False, declared="", STUB_FAIL_MATCH="comp_a5_guard.py env")
    assert code == 1 and "not the pinned ones" in out


def stored_margin0(**kw):
    return dict(settings=dict(round20=dict(provenance=dict(code_digest="d", environment=dict(a5.PINNED, gpu="L40S"), **kw))))


def test_the_same_pod_guard(monkeypatch):
    monkeypatch.setattr(lib, "provenance_row", lambda job: dict(code_digest="d", environment=dict(a5.PINNED, gpu="L40S")) if job == a5.JOB_FILE else None)
    guard.check_same_pod(stored_margin0())
    other_gpu = stored_margin0()
    other_gpu["settings"]["round20"]["provenance"]["environment"]["gpu"] = "RTX 6000 Ada"          # the mix in A3's stored read
    other_code = stored_margin0()
    other_code["settings"]["round20"]["provenance"]["code_digest"] = "e"
    for bad in (other_gpu, other_code, dict(settings=dict(round20=dict())), dict(settings=None), {}):
        with pytest.raises(SystemExit, match="other code or another environment"):
            guard.check_same_pod(bad)


def test_the_guards_commands(tmp_path, monkeypatch):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    import train_text_to_motion
    store = DirectoryStore(tmp_path)
    monkeypatch.setattr(train_text_to_motion, "make_store", lambda name: store)
    monkeypatch.setattr(lib, "provenance_row", lambda job: dict(code_digest="d", environment=dict(a5.PINNED, gpu="L40S")))
    assert guard.main(["started", "x/a.started"]) == 0 and store.exists("x/a.started")
    with pytest.raises(SystemExit, match="exists"):
        guard.main(["started", "x/a.started"])
    store.put("m.json", json.dumps(stored_margin0()).encode())
    assert guard.main(["same-pod", "m.json"]) == 0
    store.put("n.json", json.dumps({}).encode())
    with pytest.raises(SystemExit, match="other code"):
        guard.main(["same-pod", "n.json"])
    for bad in ([], ["other"], ["env", "x"], ["started"], ["same-pod"]):
        with pytest.raises(SystemExit, match="usage"):
            guard.main(bad)


# ---- the summary ------------------------------------------------------------------------------------------------------------------------

def test_the_summary_needs_all_eight_pinned(tmp_path):
    code, out, commands = job(tmp_path, "summary", draw=None, stored=EVERY, **M, **T)
    assert code == 0
    run = next(c for c in commands if "comp_a5_read.py" in c)
    for k in a5.DRAWS:
        assert f"--draw {k}={OUT}/draw{k}-attempt1-margin0.json,{OUT}/draw{k}-attempt1-top10.json" in run
        assert f"pin {OUT}/draw{k}-attempt1-margin0.json" in commands and f"pin {OUT}/draw{k}-attempt1-top10.json" in commands
    assert "--a9-sides-file reports/comp_a2_left_arm.json" in run and "--declared-commit abc1234" in run
    code, out, commands = job(tmp_path, "summary", draw=None, stored=EVERY[:-1], **M, **T)
    assert code == 1 and "incomplete: draw 8" in out and "comp_a5_read.py" not in "\n".join(commands)
    env = dict(M, **T)
    env.pop("T3_SHA")
    code, out, _ = job(tmp_path, "summary", draw=None, stored=EVERY, **env)
    assert code == 1 and "give M3_SHA and T3_SHA" in out
    code, out, _ = job(tmp_path, "summary", draw=None, stored=EVERY, STUB_PIN_FAIL=EVERY[4], **M, **T)
    assert code == 1 and "differs from its pin" in out
    code, out, commands = job(tmp_path, "summary", draw=None, stored=[f"{OUT}/a5-summary.json"])
    assert code == 0 and "stored:" in out and "comp_a5_read.py" not in "\n".join(commands)


def test_the_summary_lists_every_attempt_and_takes_the_one_with_both_arms(tmp_path):
    stored = [x for x in EVERY if "draw3-" not in x] + [f"{OUT}/draw3-attempt1-margin0.json", f"{OUT}/draw3-attempt2-margin0.json", f"{OUT}/draw3-attempt2-top10.json", f"{OUT}/draw3-attempt1-top10.started"]
    code, out, commands = job(tmp_path, "summary", draw=None, stored=stored, **M, **T)
    assert code == 0 and "COMP-A5-ATTEMPTS draw=3 attempts=1:m;2:mt; used=2" in out and "COMP-A5-ATTEMPTS draw=4 attempts=1:mt; used=1" in out
    run = next(c for c in commands if "comp_a5_read.py" in c)
    assert f"--draw 3={OUT}/draw3-attempt2-margin0.json,{OUT}/draw3-attempt2-top10.json" in run and "--attempts 3=1:m;2:mt;" in run
    assert f"pin {OUT}/draw3-attempt2-top10.json" in commands and f"pin {OUT}/draw3-attempt1-margin0.json" not in commands


def test_the_summary_refuses_two_finished_attempts_and_a_draw_with_none(tmp_path):
    code, out, commands = job(tmp_path, "summary", draw=None, stored=[*EVERY, f"{OUT}/draw2-attempt4-margin0.json", f"{OUT}/draw2-attempt4-top10.json"], **M, **T)
    assert code == 1 and "draw 2 has two finished attempts, 1 and 4" in out and "comp_a5_read.py" not in "\n".join(commands)
    one_done(out, 1)
    stored = [x for x in EVERY if "draw5-" not in x] + [f"{OUT}/draw5-attempt1-margin0.json", f"{OUT}/draw5-attempt2-margin0.json"]
    code, out, commands = job(tmp_path, "summary", draw=None, stored=stored, **M, **T)
    assert code == 1 and "incomplete: draw 5 has no attempt with both arms (1:m;2:m;)" in out and "comp_a5_read.py" not in "\n".join(commands)


def test_the_read_keeps_the_attempts_and_refuses_to_overwrite(tmp_path, monkeypatch):
    import round20_compare
    from signtranslator.data.text_motion_dataset import DirectoryStore
    import train_text_to_motion
    store = DirectoryStore(tmp_path)
    pairs, shards = draws()
    argv = []
    for k, (m, t) in pairs.items():
        store.put(f"m{k}", json.dumps(m).encode())
        store.put(f"t{k}", json.dumps(t).encode())
        argv += ["--draw", f"{k}=m{k},t{k}"]
    monkeypatch.setattr(train_text_to_motion, "make_store", lambda name: store)
    monkeypatch.setattr(round20_compare, "report_shards", lambda s, key, report, _: shards[int(key[1:])][0 if key[0] == "m" else 1])
    sides = tmp_path / "sides.json"
    sides.write_text(json.dumps(SIDES))
    sha = hashlib.sha256(sides.read_bytes()).hexdigest()
    args = [*argv, "--attempts", "3=1:m;2:mt;", "--a9-sides-file", str(sides), "--a9-sides-sha256", sha, "--bootstrap", "30", "--output-key", "out.json"]
    assert a5.main(args) == 0
    out = json.loads(store.get("out.json"))
    assert out["attempts"] == {"3": "1:m;2:mt;"} and out["rule"]["outcome"] in ("confirmed", "direction_only_not_confirmed", "not_confirmed")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        a5.main(args)
    with pytest.raises(SystemExit, match="does not match"):
        a5.main([*argv, "--a9-sides-file", str(sides), "--a9-sides-sha256", "0" * 64, "--output-key", "o2.json"])


@pytest.mark.parametrize("kwargs,message", [
    (dict(phase="cache", draw="9"), "DRAW must be"), (dict(phase="cache", draw="0"), "DRAW must be"), (dict(phase="margin0", draw=None), "DRAW must be"),
    (dict(phase="summary", draw="1"), "no DRAW"), (dict(phase="summary", draw=None, smoke=True), "no smoke"), (dict(phase="other"), "unknown COMP_PHASE"),
    (dict(phase="cache", SET="c20"), "D20 only"), (dict(phase="cache", SET="val1000"), "D20 only"), (dict(phase="cache", gpu=False), "needs a GPU"),
    (dict(phase="cache", declared=""), "nothing is declared yet"), (dict(phase="margin0"), "is not stored"), (dict(phase="cache", SMOKE="2"), "SMOKE must be"),
])
def test_refusals(tmp_path, kwargs, message):
    phase = kwargs.pop("phase")
    code, out, commands = job(tmp_path, phase, **kwargs)
    assert code == 1 and message in out and "ALERT" in out and "round15_compose_eval.py" not in "\n".join(commands) and "round20_word_cache.py" not in "\n".join(commands)
    one_done(out, 1)


@pytest.mark.parametrize("draw", ["1", "8"])
def test_every_draw_of_the_rule_is_accepted(tmp_path, draw):
    code, _, _ = job(tmp_path, "cache", draw=draw)
    assert code == 0


def test_the_declaration_is_a5s_own(tmp_path):
    _, _, commands = job(tmp_path, "cache")
    assert scripts_run(commands)[:2] == ["scripts/comp_a5_guard.py env", "scripts/comp_a5_declared.py abc1234"]
    code, out, commands = job(tmp_path, "cache", STUB_FAIL_MATCH="comp_a5_declared.py")
    assert code == 1 and "does not declare A5" in out and len(scripts_run(commands)) == 2
    one_done(out, 1)


def test_a_smoke_is_five_clips_four_samples_and_needs_no_declaration(tmp_path):
    code, out, commands = job(tmp_path, "cache", smoke=True, gpu=False, declared="", CACHE_PROCS="1", SMOKE_STAMP="s1")
    assert code == 0 and "smoke=1" in out
    run = next(c for c in commands if "round20_word_cache.py run" in c)
    assert "--clips 5 " in run and "--samples 4 --choose 4" in run and f"{OUT}/smoke/s1/draw1-cache/attempt1 " in run
    code, _, commands = job(tmp_path, "margin0", smoke=True, gpu=False, declared="", SMOKE_STAMP="s1", stored=[f"{OUT}/smoke/s1/draw1-cache/attempt1/index.json"])
    assert code == 0 and "--clips 5 " in run_line(commands) and "--word-cache-n 4" in run_line(commands)


@pytest.mark.parametrize("match,message", [("round15_compose_eval.py run", "ALERT run draw1-attempt1-margin0"), ("round15_compose_eval.py merge", "ALERT merge draw1-attempt1-margin0")])
def test_a_failed_step_stops_with_done_one(tmp_path, match, message):
    code, out, _ = job(tmp_path, "margin0", stored=caches(), STUB_FAIL_MATCH=match)
    assert code == 1 and message in out and "COMP-A5 draw1-attempt1-margin0" not in out
    one_done(out, 1)


@pytest.mark.parametrize("phase,stored", [("margin0", caches()), ("margin0", [*caches(), f"{OUT}/draw1-attempt1-margin0.json"]), ("cache", []), ("extra", []), ("cache", caches())])
def test_no_hash_is_no_success(tmp_path, phase, stored):
    code, out, _ = job(tmp_path, phase, stored=stored, STUB_SHA_EMPTY="1")
    assert code == 1 and "no hash" in out and "COMP-A5 " not in out
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
    code, out, commands = job(tmp_path, "cache", stored=[caches()[0]])
    assert code == 0 and "stored:" in out and not any("round20_word_cache.py" in c for c in commands)
    code, out, commands = job(tmp_path, "margin0", stored=[caches()[0], f"{OUT}/draw1-attempt1-margin0.json"])
    assert code == 0 and "stored:" in out and "round15_compose_eval.py" not in "\n".join(commands)


@pytest.mark.parametrize("phase,stored", [("cache", []), ("extra", []), ("margin0", caches()), ("top10", [*caches(), f"{OUT}/draw1-attempt1-margin0.json"])])
def test_no_per_draw_number_reaches_the_log(tmp_path, phase, stored):
    """A merge prints the arm's BLEU; the job's output is the log the chain tees."""
    code, out, commands = job(tmp_path, phase, stored=stored, MARGIN0_SHA="m", STUB_PRINT="BLEU on 500 clips")
    assert code == 0 and any(" merge " in c for c in commands) and "BLEU" not in out


# ---- the chain, on a fake job ----------------------------------------------------------------------------------------------------------------

FAKE_JOB = '''#!/bin/bash
echo "$COMP_PHASE smoke=${SMOKE:-0} attempt=${ATTEMPT:-} margin0=${MARGIN0_SHA:-} declared=${COMP_A_DECLARED:-}" >> "$FAKE_CALLS"
[ "${FAKE_FAIL:-}" = "$COMP_PHASE" ] && [ -z "${SMOKE:-}" ] && exit 1
[ "${FAKE_FAIL_SMOKE:-}" = "$COMP_PHASE" ] && [ -n "${SMOKE:-}" ] && exit 1
if [ -z "${SMOKE:-}" ] && [ "${FAKE_NO_HASH:-}" = "$COMP_PHASE" ]; then echo "stored: nothing said"; exit 0; fi
[ "$COMP_PHASE" = margin0 ] && echo "COMP-A5 draw$DRAW-attempt$ATTEMPT-margin0 key/m ${SMOKE:+smoke}hash-m"
[ "$COMP_PHASE" = top10 ] && echo "COMP-A5 draw$DRAW-attempt$ATTEMPT-top10 key/t ${SMOKE:+smoke}hash-t"
exit 0
'''


def chain(tmp_path, *args, **env):
    """The real chain on a fake job: returns (exit code, stdout, stderr, the job's calls)."""
    root = tmp_path / "code"
    (root / "runpod/fleet").mkdir(parents=True, exist_ok=True)
    (root / "runpod/fleet/comp_a5.sh").write_text(FAKE_JOB)
    text = CHAIN.read_text()
    assert "cd /root/work/Sign-Translator" in text and "LOG=/root/comp_a5_chain.log" in text
    text = text.replace("cd /root/work/Sign-Translator", f"cd {root}").replace("LOG=/root/comp_a5_chain.log", f"LOG={tmp_path}/chain.log")
    script = tmp_path / "chain.sh"
    script.write_text(text)
    calls = tmp_path / "calls"
    calls.write_text("")
    run = subprocess.run(["bash", str(script), *args], capture_output=True, text=True, timeout=30, env=dict(PATH="/usr/bin:/bin", FAKE_CALLS=str(calls), **env))
    return run.returncode, run.stdout, run.stderr, calls.read_text().splitlines()


def last_line(out):
    return out.strip().splitlines()[-1]


def test_the_chain_runs_a_draw_in_order_and_prints_the_hashes(tmp_path):
    code, out, _, calls = chain(tmp_path, "3", "abc1234")
    assert code == 0 and last_line(out) == "COMP-A5-CHAIN-DONE 0"
    assert "COMP-A5-HASHES draw=3 attempt=1 M3_SHA=hash-m T3_SHA=hash-t" in out and "COMP-A5 draw3-attempt1-margin0 key/m hash-m" in out and "ALERT" not in out
    assert calls == [f"{p} smoke=1 attempt=1 margin0= declared=" for p in ("cache", "extra", "margin0", "top10")] + \
        [f"{p} smoke=0 attempt=1 margin0= declared=abc1234" for p in ("cache", "extra", "margin0")] + ["top10 smoke=0 attempt=1 margin0=hash-m declared=abc1234"]


def test_a_later_attempt_runs_under_its_number_and_smokes_apart(tmp_path):
    code, out, _, calls = chain(tmp_path, "2", "abc1234", "3")
    assert code == 0 and "COMP-A5-HASHES draw=2 attempt=3 M2_SHA=hash-m T2_SHA=hash-t" in out
    assert calls[3].startswith("top10 smoke=1 attempt=1") and calls[-1] == "top10 smoke=0 attempt=3 margin0=hash-m declared=abc1234"


@pytest.mark.parametrize("phase,name", [("margin0", "margin0"), ("top10", "top10")])
def test_a_missing_hash_alerts_on_stderr_and_still_says_done(tmp_path, phase, name):
    code, out, err, calls = chain(tmp_path, "1", "abc1234", FAKE_NO_HASH=phase)
    assert code == 1 and last_line(out) == "COMP-A5-CHAIN-DONE 1" and f"ALERT chain: no hash for draw1-attempt1-{name}" in err
    assert f"ALERT chain: no hash for {name} of draw 1 attempt 1" in out
    assert "COMP-A5-HASHES" not in out and out.count("COMP-A5-CHAIN-DONE") == 1
    if phase == "margin0":
        assert not any(c.startswith("top10 smoke=0") for c in calls)       # no comparator's hash, no top10, and the smoke's hash is never taken


@pytest.mark.parametrize("phase", ["cache", "extra", "margin0", "top10"])
def test_a_failed_phase_stops_the_chain_with_an_alert_and_done(tmp_path, phase):
    code, out, _, calls = chain(tmp_path, "1", "abc1234", FAKE_FAIL=phase)
    assert code == 1 and last_line(out) == "COMP-A5-CHAIN-DONE 1" and f"ALERT chain: COMP_PHASE={phase}" in out and out.count("COMP-A5-CHAIN-DONE") == 1
    assert "COMP-A5-HASHES" not in out and calls[-1].startswith(f"{phase} smoke=0")


def test_a_failed_smoke_stops_before_any_full_phase(tmp_path):
    code, out, _, calls = chain(tmp_path, "1", "abc1234", FAKE_FAIL_SMOKE="extra")
    assert code == 1 and last_line(out) == "COMP-A5-CHAIN-DONE 1" and calls == ["cache smoke=1 attempt=1 margin0= declared=", "extra smoke=1 attempt=1 margin0= declared="]


@pytest.mark.parametrize("args", [(), ("1",)])
def test_missing_arguments_still_say_done(tmp_path, args):
    code, out, err, calls = chain(tmp_path, *args)
    assert code != 0 and calls == [] and "ALERT chain stopped unexpectedly" in out and last_line(out) == "COMP-A5-CHAIN-DONE 1"


def test_no_code_on_the_pod_still_says_done(tmp_path):
    script = tmp_path / "chain.sh"
    script.write_text(CHAIN.read_text().replace("cd /root/work/Sign-Translator", f"cd {tmp_path}/absent"))
    run = subprocess.run(["bash", str(script), "1", "abc1234"], capture_output=True, text=True, timeout=30, env=dict(PATH="/usr/bin:/bin"))
    assert run.returncode == 1 and "ALERT chain: no code" in run.stdout and last_line(run.stdout) == "COMP-A5-CHAIN-DONE 1"


def test_the_chain_is_in_the_digest_and_the_real_job_prints_what_it_reads(tmp_path):
    """The chain's hash line pattern against the real job's own line, stored arm included."""
    from signtranslator.eval.provenance import code_digest
    assert code_digest(extra_files=JOBS) != code_digest(extra_files=JOBS[:1])
    _, out, _ = job(tmp_path, "margin0", stored=[*caches(), f"{OUT}/draw1-attempt1-margin0.json"])
    line = next(x for x in out.splitlines() if x.startswith("COMP-A5 draw1-attempt1-margin0 "))
    assert line.split()[3] == "stubsha" and "hash_of \"draw$D-attempt$ATTEMPT-margin0\"" in CHAIN.read_text()


def test_the_bleu_interval_resamples_the_draws():
    """Sentences all alike, so resampling them moves nothing: any width in this interval is the draws'."""
    from signtranslator.eval.s0_metrics import bleu_statistics
    refs = ["alpha beta gamma delta epsilon zeta eta theta"] * 20
    a = np.array([bleu_statistics([" ".join(r.split()[:n]) for r in refs], refs) for n in (4, 5, 6, 7, 8)])
    b = np.repeat(bleu_statistics([" ".join(r.split()[:4]) for r in refs], refs)[None], 5, 0)
    row = a5.bleu_row(a, b, boot=200)
    assert row["between_draw_sd"] > 0 and row["two_way_ci95"][1] - row["two_way_ci95"][0] > row["between_draw_sd"]
