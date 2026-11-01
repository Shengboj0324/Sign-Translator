"""V1 on synthetic data: the cap's route, neutral hand frames, the after-the-fact sensitivity, the job and the draft rule. No test starts a real job: the runner tests use fake executables and no credentials."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v1 as W  # noqa: E402
import tok_v1_rule as Rule  # noqa: E402
from tests.test_generation_signing_space import signer_state  # noqa: E402
from tests.test_tok_probes import Perfect, clip_records, forbidden_keys, gappy, plant_canary, tiny_tokenizer  # noqa: E402
from tests.test_tok_v0 import ValuesReader, gapped, never_tracked  # noqa: E402

N = 12
JOB = ROOT / "runpod/fleet/tok_v1.sh"
IDENTITY = np.array([1., 0., 0., 0., 1., 0.])


def with_hand(state, hand, values):
    c = state.channels[hand]
    return replace(state, channels={**state.channels, hand: replace(c, values=values.astype(c.values.dtype))})


def neutral_frames(state, frames, hand="left_hand"):
    """``state`` with these frames of a hand at exactly the rest pose."""
    values = state.channels[hand].values.copy()
    values[frames] = IDENTITY
    return with_hand(state, hand, values)


def rotated(angle_degrees, frames=8, joints=15):
    """6D values at this angle about one axis, every joint."""
    a = np.radians(angle_degrees)
    return np.broadcast_to(np.array([np.cos(a), np.sin(a), 0., -np.sin(a), np.cos(a), 0.]), (frames, joints, 6)).copy()


# ---- neutral hand frames ------------------------------------------------------------------------------------------------------

def test_hand_distance_is_the_mean_angle_to_rest_over_tracked_joints():
    state = signer_state(0, frames=8, amplitude=.4)
    c = state.channels["left_hand"]
    J = c.values.shape[1]
    d = W.hand_distance(replace(c, values=rotated(30., 8, J)))
    np.testing.assert_allclose(d, 30., atol=1e-6)
    valid = c.valid.copy()
    valid[:, 1:] = False
    values = rotated(30., 8, J)
    values[:, 0] = rotated(10., 8, J)[:, 0]
    d = W.hand_distance(replace(c, values=np.where(valid[..., None], values, 0.), valid=valid, observed=c.observed & valid, confidence=c.confidence * valid))
    np.testing.assert_allclose(d, 10., atol=1e-6)                                 # untracked joints do not count


def test_a_frame_with_no_tracked_joint_has_no_distance():
    state = gappy(signer_state(0, frames=20, amplitude=.4), 6, 9)
    d = W.hand_distance(state.channels["left_hand"])
    assert np.isnan(d[6:9]).all() and np.isfinite(d[:6]).all()
    assert np.isnan(W.hand_distance(never_tracked().channels["left_hand"])).all()


def test_neutral_shares_count_each_threshold_and_can_keep_to_tracked_frames():
    state = neutral_frames(signer_state(0, frames=20, amplitude=.8), [0, 1, 2, 3])
    s = W.neutral_shares(state)
    assert s["left_hand"]["5.0"] == .2 and s["right_hand"]["5.0"] == 0
    assert list(s["left_hand"]) == [str(t) for t in W.THRESHOLDS]
    gapped_state = gappy(state, 0, 2)
    assert W.neutral_shares(gapped_state, tracked_only_by=gapped_state)["left_hand"]["5.0"] == pytest.approx(2 / 18, abs=1e-4)
    assert W.neutral_shares(never_tracked())["left_hand"]["5.0"] is None


def test_every_threshold_is_stated_and_one_is_flagged():
    assert W.THRESHOLDS == (1., 2., 5., 10., 20.) and W.FLAG_DEGREES == 5. and W.BINS == 20 and W.WORD_CLIPS == 8 and W.CORPUS_CLIPS == 300


def test_thresholds_are_read_as_degrees():
    state = signer_state(0, frames=8, amplitude=.4)
    J = state.channels["left_hand"].values.shape[1]
    s = W.neutral_shares(with_hand(state, "left_hand", rotated(7., 8, J)))["left_hand"]
    assert s["5.0"] == 0 and s["10.0"] == 1


def test_replacing_neutral_frames_holds_the_nearest_other_frame_of_that_hand_only():
    state = neutral_frames(signer_state(0, frames=20, amplitude=.8), [4, 5, 6, 12])
    out, counts = W.replaced_neutral(state)
    c, r = out.channels["left_hand"].values, state.channels["left_hand"].values
    np.testing.assert_array_equal(c[4], r[3]) ; np.testing.assert_array_equal(c[6], r[7]); np.testing.assert_array_equal(c[12], r[11])
    np.testing.assert_array_equal(c[0], r[0])
    np.testing.assert_array_equal(out.channels["right_hand"].values, state.channels["right_hand"].values)
    np.testing.assert_array_equal(out.channels["body"].values, state.channels["body"].values)
    assert counts == dict(flagged_frames=4, hand_frames=40, hands_all_neutral=0)
    assert not (W.hand_distance(out.channels["left_hand"]) < 5.).any()


def test_a_hand_all_neutral_stays_and_is_counted_and_a_threshold_is_respected():
    state = neutral_frames(signer_state(0, frames=8, amplitude=.8), list(range(8)))
    out, counts = W.replaced_neutral(state)
    assert counts["hands_all_neutral"] == 1 and counts["flagged_frames"] == 0
    np.testing.assert_array_equal(out.channels["left_hand"].values, state.channels["left_hand"].values)
    J = state.channels["left_hand"].values.shape[1]
    near = with_hand(signer_state(0, frames=8, amplitude=.8), "left_hand", np.concatenate((rotated(3., 2, J), signer_state(0, frames=6, amplitude=.8).channels["left_hand"].values)))
    assert W.replaced_neutral(near, 5.)[1]["flagged_frames"] == 2 and W.replaced_neutral(near, 1.)[1]["flagged_frames"] == 0


def test_time_bins_and_the_untracked_profile():
    np.testing.assert_allclose(W.time_bins(np.arange(40.), 4), [4.5, 14.5, 24.5, 34.5])
    assert np.isnan(W.time_bins(np.array([np.nan] * 4), 2)).all()
    state = gappy(signer_state(0, frames=40, amplitude=.4), 0, 20)
    u = W.untracked_profile([state], "left_hand")
    assert len(u) == W.BINS and (u[:10] == 1).all() and (u[10:] == 0).all()
    assert np.isnan(W.untracked_profile([], "left_hand")).all()
    u = W.untracked_profile([state, signer_state(1, frames=40, amplitude=.4)], "left_hand")
    assert (u[:10] == .5).all()


def alignment_row(entry, neutral_where):
    u = np.zeros(W.BINS)
    u[:10] = 1.
    d = np.full(W.BINS, 50.)
    d[neutral_where] = 0.
    return dict(entry=entry, generated=dict(left_hand=d, right_hand=np.full(W.BINS, 50.)), untracked=dict(left_hand=u, right_hand=np.zeros(W.BINS)))


def test_alignment_finds_neutral_frames_where_training_was_untracked_and_not_otherwise():
    aligned = W.word_alignment([alignment_row(f"w{i}", np.arange(10)) for i in range(20)])["thresholds"]["5.0"]
    assert aligned["neutral_where_untracked"] == 1 and aligned["neutral_where_tracked"] == 0 and aligned["difference"] == 1 and aligned["ci95"][0] > 0
    assert aligned["bins_untracked"] == 200 and aligned["bins_tracked"] == 600
    other = W.word_alignment([alignment_row(f"w{i}", np.arange(10, 20)) for i in range(20)])["thresholds"]["5.0"]
    assert other["difference"] < 0 and other["ci95"][1] < 0
    assert W.word_alignment([]) == dict(entries=0)


def test_population_and_the_paired_excess():
    a = [dict(left_hand={str(t): .3 for t in W.THRESHOLDS}, right_hand={str(t): .1 for t in W.THRESHOLDS}) for _ in range(5)]
    b = [dict(left_hand={str(t): .1 for t in W.THRESHOLDS}, right_hand={str(t): None for t in W.THRESHOLDS}) for _ in range(5)]
    pop = W.population(a)
    assert pop["both"]["5.0"]["mean"] == pytest.approx(.2) and pop["left_hand"]["5.0"]["mean"] == pytest.approx(.3) and pop["clips"] == 5
    assert W.population(b)["right_hand"]["5.0"]["n"] == 0
    assert W.paired_population(a, b)["5.0"]["difference"] == pytest.approx(.1)


def record(sign, corpus="c", partition="train", component="x", text="dog"):
    return dict(sign=sign, corpus=corpus, partition=partition, signer_component=component, text=text, continuous=dict(state_key=sign))


def fake_load(monkeypatch, gaps):
    def load(r, field, cache=None, fetch=None):
        state = signer_state(0, frames=20, amplitude=.4)
        return gappy(state, 0, gaps.get(r["sign"], 0)) if gaps.get(r["sign"]) else state
    monkeypatch.setattr("signtranslator.data_engineering.corpus_manifest.load_sample", load)


def test_corpus_neutral_counts_untracked_hand_frames_per_corpus_and_never_c00_or_test(monkeypatch):
    fake_load(monkeypatch, {"a": 10})
    manifest = dict(records=[record("a", "one"), record("b", "one"), record("c", "two"), record("z", "one", component="c00"), record("t", "one", partition="test"), record("v", "three", partition="val")])
    out = W.corpus_neutral(manifest, None, None)
    assert set(out) == {"one", "two", "weighted_by_records"} and out["one"]["records"] == 2 and out["one"]["sampled"] == 2
    assert out["one"]["every_joint_untracked"]["mean"] == pytest.approx(.25) and out["two"]["every_joint_untracked"]["mean"] == 0
    assert out["weighted_by_records"]["every_joint_untracked"] == pytest.approx(.5 * 2 * .25 / 3 + 0, abs=.2)
    only = W.corpus_neutral(manifest, None, None, components=("x",), clips=1)
    assert only["one"]["sampled"] == 1


def test_a_words_training_clips_are_its_own_never_c00_or_test_and_capped(monkeypatch):
    fake_load(monkeypatch, {})
    rows = [record(f"s{i}", "asllvd-text-v1") for i in range(12)] + [record("z", "asllvd-text-v1", component="c00"), record("t", "asllvd-text-v1", partition="test"), record("o", "asllvd-text-v1", text="cat"), record("h", "how2sign-two-view-text-v1")]
    out = W.word_training_clips(dict(records=rows), None, None, ["dog", "fish"])
    assert list(out) == ["dog"] and len(out["dog"]) == W.WORD_CLIPS
    assert W.word_training_clips(dict(records=rows), None, None, ["dog"], clips=3)["dog"] is not None


def test_top_tokens_are_the_committed_lists_first_ranks_and_a_changed_list_refuses(monkeypatch):
    payload = json.loads((ROOT / W.TOKENS_FILE).read_text())
    top = {t["token"] for t in payload["tokens"] if t["rank"] <= 10}
    assert W.top_tokens(10) == top and len(top) == 10
    monkeypatch.setattr(W, "TOKENS_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        W.top_tokens(10)


class Files:
    def __init__(self, files):
        self.files = files

    def get(self, key):
        return self.files[key]

    def exists(self, key):
        return key in self.files


def test_a_cache_is_pinned_and_gives_best_of_16():
    payload = json.dumps(dict(entries=[dict(entry="a", chosen={"16": dict(state_key="k", state_sha256="s"), "4": dict(state_key="x", state_sha256="y")}), dict(entry="b", chosen={"16": dict(state_key="k2", state_sha256="s2")})])).encode()
    sha = hashlib.sha256(payload).hexdigest()
    rows = W.cache_entries(Files(dict(c=payload)), "c", sha)
    assert rows[0] == dict(entry="a", state_key="k", state_sha256="s") and len(rows) == 2
    assert [r["entry"] for r in W.cache_entries(Files(dict(c=payload)), "c", sha, {"b"})] == ["b"]
    with pytest.raises(SystemExit, match="SHA-256"):
        W.cache_entries(Files(dict(c=payload)), "c", "0" * 64)


def test_a_composed_export_must_hold_every_sign_and_be_an_export():
    index = json.dumps(dict(kind="round15-composed-states", states=[dict(sign="a", form=W.COMPOSED_FORM, state_key="k", state_sha256="s"), dict(sign="b", form="other", state_key="k", state_sha256="s")])).encode()
    with pytest.raises(SystemExit, match="not exported"):
        W.load_composed(Files(dict(i=index)), "i", ["a", "b"])
    with pytest.raises(SystemExit, match="not a composed"):
        W.load_composed(Files(dict(i=b"{}")), "i", ["a"])


# ---- the cap's route ---------------------------------------------------------------------------------------------------------------

def test_the_cap_forms_send_the_nearest_filled_clip_through_the_tokenizer():
    tok = tiny_tokenizer()
    state = gapped(1, start=6)
    filled = W.nearest_input(state, tok)
    assert filled.channels["left_hand"].valid.all()
    seen = {}
    decode = tok.decode_latents
    encode = tok.encode

    def spy_encode(unit):
        seen["unit"] = unit.clone()
        return encode(unit)
    def spy_decode(q, frames):
        seen["q"] = {k: v.clone() for k, v in q.items()}
        return decode(q, frames)
    tok.encode, tok.decode_latents = spy_encode, spy_decode
    rt = W.cap_form("rt_nearest", state, tok)
    assert not torch.allclose(seen["unit"][0, 6:9], torch.tensor(0.).expand_as(seen["unit"][0, 6:9]))
    latents = encode(seen["unit"])
    full = tok.quantize(latents)[0]
    assert all(torch.equal(seen["q"][p], full[p]) for p, _ in tok.parts)
    nq = W.cap_form("no_quant_nearest", state, tok)
    assert all(torch.equal(seen["q"][p], latents[p]) for p, _ in tok.parts)
    tok.encode, tok.decode_latents = encode, decode
    assert rt.channels["left_hand"].valid.all() and nq.channels["left_hand"].valid.all() and not np.allclose(rt.channels["left_hand"].values, nq.channels["left_hand"].values)
    other = P.decoded_form("round_trip", state, tok, None)
    assert not np.allclose(rt.channels["left_hand"].values, other.channels["left_hand"].values)      # the neutral-filled route differs


def test_a_decoder_that_returns_its_input_reads_the_nearest_filled_clip_exactly():
    tok = Perfect(tiny_tokenizer())
    state = gapped(1, start=6)
    filled, rt = W.cap_form("v0_nearest_fill", state, tok), W.cap_form("rt_nearest", state, tok)
    for name in ("body", "left_hand", "right_hand"):
        np.testing.assert_allclose(rt.channels[name].values, filled.channels[name].values, atol=1e-5)
    assert W.cap_form("real", state, tok) is state and not W.cap_form("blank", state, tok).channels["left_hand"].valid.any()
    with pytest.raises(SystemExit, match="unknown"):
        W.cap_form("x", state, tok)


def test_a_never_tracked_hand_builds_in_every_cap_form():
    tok = tiny_tokenizer()
    for name in W.CAP_FORMS:
        assert len(W.cap_form(name, never_tracked(), tok).timestamps) == 20


# ---- read, merge ----------------------------------------------------------------------------------------------------------------------

def harness(monkeypatch, tok=None, *, declared=True, n=N, reader=ValuesReader):
    records = clip_records(n, declared)
    refs = [gapped(i) for i in range(n)]
    tok = tok or tiny_tokenizer()
    monkeypatch.setattr(L, "open_read_list", lambda store, smoke: (records, declared and not smoke, dict(list="t", clips=n)))
    monkeypatch.setattr(L, "load_states", lambda store, recs: refs)
    monkeypatch.setattr(L, "load_tokenizer", lambda store: tok)
    monkeypatch.setattr(L, "make_reader", lambda store, **k: reader(refs))
    monkeypatch.setattr(W, "load_composed", lambda store, key, signs: [neutral_frames(signer_state(i, frames=20, amplitude=.8), [1, 2, 3]) for i in range(len(signs))])
    return records, refs, tok


def read(tmp_path, forms=W.FORMS, *extra):
    return W.main(["read", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w"), "--forms", ",".join(forms), *extra])


def plant_checks(d):
    plant_canary(d)
    path = d / "path_check.json"
    if (d / "canary.json").exists() and not path.exists():
        path.write_text(json.dumps(dict(json.loads((d / "canary.json").read_text()), reproduced=True)))


def merged(tmp_path, forms=W.FORMS):
    plant_checks(tmp_path / "w")
    out = tmp_path / f"report-{len(list(tmp_path.iterdir()))}.json"
    assert W.main(["merge", "--store", "dir:x", "--out-dir", str(tmp_path / "w"), "--forms", ",".join(forms), "--output-file", str(out)]) == 0
    return json.loads(out.read_text())


def test_a_perfect_codec_keeps_every_nearest_filled_read_and_the_panel_says_so(monkeypatch, tmp_path):
    harness(monkeypatch, Perfect(tiny_tokenizer()))
    assert read(tmp_path) == 0
    r = merged(tmp_path)
    k = r["reads"]["likelihood_minus_blank"]
    assert k["cap_share_of_nearest_fill"]["point"] == 1 and k["rt_nearest_minus_v0_nearest_fill"]["point"] == 0 and k["no_quant_nearest_minus_rt_nearest"]["point"] == 0
    assert set(r["forms"]["rt_nearest"]["comparators"]) == {"v0_nearest_fill", "round_trip"} and set(r["forms"]["no_quant_nearest"]["comparators"]) == {"v0_nearest_fill", "rt_nearest"}
    assert r["reads"]["primary"] == "likelihood_minus_blank" and r["reads"]["bleu"]["cap_share_of_nearest_fill"]["point"] == 1 and r["reads"]["bootstrap"] == dict(n=2000, seed=0)
    assert r["report_only"] is True and forbidden_keys(r) == [] and r["code_digest"] == W.code_digest()


def test_a_codec_that_breaks_a_hand_loses_the_cap(monkeypatch, tmp_path):
    from tests.test_tok_probes import neutral_columns
    harness(monkeypatch, Perfect(tiny_tokenizer(), damage=neutral_columns("left_hand")))
    assert read(tmp_path, ["real", "blank", "v0_nearest_fill", "rt_nearest"]) == 0
    k = merged(tmp_path, ["real", "blank", "v0_nearest_fill", "rt_nearest"])["reads"]["likelihood_minus_blank"]
    assert k["cap_share_of_nearest_fill"]["point"] == 0 and k["rt_nearest_minus_v0_nearest_fill"]["point"] < 0 and k["rt_nearest_minus_round_trip"] is None


def test_the_sensitivity_reads_the_composed_export_and_replaces_only_flagged_frames(monkeypatch, tmp_path):
    harness(monkeypatch)
    assert read(tmp_path, ["real", "blank", "a9p", "a9p_replaced"]) == 0
    r = merged(tmp_path, ["real", "blank", "a9p", "a9p_replaced"])
    assert r["sensitivity_counts"]["flagged_frames"] == 3 * N and r["sensitivity_counts"]["flag_degrees"] == 5. and "sensitivity" in r["disclosures"]
    assert set(r["forms"]["a9p_replaced"]["comparators"]) == {"a9p"} and r["reads"]["likelihood_minus_blank"]["a9p_replaced_minus_a9p"] is not None
    with pytest.raises(SystemExit, match="composed export"):
        W.build_form("a9p", [], None, None)


def test_a_smoke_stands_in_train_clips_for_the_composed_forms_and_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "load_composed", lambda *a: pytest.fail("a smoke reads no composed form"))
    records, refs, tok = harness(monkeypatch, declared=False)
    monkeypatch.setattr(W, "load_composed", lambda *a: pytest.fail("a smoke reads no composed form"))
    assert read(tmp_path, ["real", "blank", "a9p"], "--smoke-train", "12") == 0
    assert "SMOKE" in merged(tmp_path, ["real", "blank", "a9p"])["smoke"]


@pytest.mark.parametrize("name", ["D20_SHA", "TOKENIZER_SHA", "TRANSLATOR_SHA"])
def test_a_mutated_pin_refuses_before_any_read(monkeypatch, tmp_path, name):
    monkeypatch.setattr(L, name, "0" * 64)
    monkeypatch.setattr(L, "open_read_list", lambda *a: pytest.fail("read after a mutated pin"))
    with pytest.raises(SystemExit, match="pins moved"):
        read(tmp_path, ["real"])


@pytest.mark.parametrize("change", [dict(partition="test"), dict(signer_component="c00"), dict(partition="train")])
def test_a_declared_read_refuses_what_is_not_d20(monkeypatch, tmp_path, change):
    records, _, _ = harness(monkeypatch, n=3)
    records[1].update(change)
    with pytest.raises(SystemExit, match="not a How2Sign val clip"):
        read(tmp_path, ["real"])
    with pytest.raises(SystemExit, match="not a How2Sign val clip"):
        W.main(["neutral", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w")])


def test_outputs_are_write_once_and_resume_reads_only_what_is_missing(monkeypatch, tmp_path):
    harness(monkeypatch, n=3)
    assert read(tmp_path, ["real"]) == 0
    with pytest.raises(SystemExit, match="write-once"):
        read(tmp_path, ["real"])
    assert read(tmp_path, ["real", "blank"], "--resume") == 0
    with pytest.raises(SystemExit, match="unknown forms"):
        read(tmp_path, ["k512"])
    for flag in ("--smoke-longest", "--smoke-never-tracked"):
        with pytest.raises(SystemExit, match="belong to a smoke"):
            read(tmp_path, ["identity"], flag, "2")


@pytest.mark.parametrize("bad,name", [("missing", "canary.json"), ("failed", "canary.json"), ("code", "canary.json"), ("missing", "path_check.json"), ("failed", "path_check.json"), ("code", "path_check.json")])
def test_a_declared_merge_refuses_without_its_checks(monkeypatch, tmp_path, bad, name):
    harness(monkeypatch, n=4)
    assert read(tmp_path, ["real", "blank"]) == 0
    w = tmp_path / "w"
    plant_checks(w)
    r = json.loads((w / name).read_text())
    if bad == "missing":
        (w / name).unlink()
    else:
        (w / name).write_text(json.dumps(dict(r, reproduced=False) if bad == "failed" else dict(r, code_digest="z")))
    with pytest.raises(SystemExit, match="void"):
        W.main(["merge", "--store", "dir:x", "--out-dir", str(w), "--forms", "real,blank", "--output-file", str(tmp_path / "x.json")])


def test_a_merge_refuses_a_motion_measure_of_another_run(monkeypatch, tmp_path):
    harness(monkeypatch, n=4)
    assert read(tmp_path, ["real", "blank"]) == 0
    plant_checks(tmp_path / "w")
    r = json.loads((tmp_path / "w/form-real.json").read_text())
    (tmp_path / "w/neutral.json").write_text(json.dumps(dict(code_digest="z", environment=r["environment"], declared_read=True)))
    with pytest.raises(SystemExit, match="another run"):
        merged(tmp_path, ["real", "blank"])
    (tmp_path / "w/neutral.json").write_text(json.dumps(dict(code_digest=r["code_digest"], environment=r["environment"], declared_read=True, populations={})))
    assert merged(tmp_path, ["real", "blank"])["neutral"]["populations"] == {}


def test_the_motion_measure_runs_on_a_declared_read_and_reads_no_instrument(monkeypatch, tmp_path):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    records, refs, tok = harness(monkeypatch, n=6)
    monkeypatch.setattr(L, "make_reader", lambda *a, **k: pytest.fail("the motion measure reads no instrument"))
    monkeypatch.setattr(P, "load_reader", lambda *a, **k: pytest.fail("the motion measure reads no instrument"))
    store = DirectoryStore(tmp_path / "s")
    run = json.dumps(dict(manifest_key="d2/m", manifest_sha256="abc")).encode()
    store.put(f"{L.TOKENIZER.split('/checkpoints/')[0]}/run.json", run)
    seen = []
    monkeypatch.setattr(L, "load_manifest_at", lambda s, k, sha: seen.append((k, sha)) or dict(records=[]))
    entries = {W.A9_CACHE: [dict(entry="dog", state_key="a", state_sha256="s"), dict(entry="cat", state_key="b", state_sha256="s")], W.A3_CACHE: [dict(entry="you", state_key="c", state_sha256="s")]}
    monkeypatch.setattr(W, "cache_entries", lambda store, key, sha, only=None: entries[key])
    monkeypatch.setattr("evaluate_text_to_motion.load_state", lambda store, key, sha: neutral_frames(signer_state(1, frames=20, amplitude=.8), list(range(10))))
    monkeypatch.setattr(W, "word_training_clips", lambda manifest, cache, fetch, states, **k: {e: [gappy(signer_state(2, frames=20, amplitude=.4), 0, 10)] for e in states if e != "cat"})
    monkeypatch.setattr(W, "corpus_neutral", lambda manifest, cache, fetch, **k: dict(one=dict(records=1)))
    assert W.main(["neutral", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w")]) == 0
    r = json.loads((tmp_path / "w/neutral.json").read_text())
    assert set(r["populations"]) == {"real_tracked_frames", "real_with_neutral_fill", "real_with_nearest_fill", "a9_words", "a3_new_words", "a9_predicted", "a3_top10_predicted"}
    assert (L.TRAIN_MIX, L.TRAIN_MIX_SHA) in seen and ("d2/m", "abc") in seen and r["corpus_neutral"]["d2_manifest"]["key"] == "d2/m" and set(r["corpus_neutral"]["w14b"]) == {"one"}
    assert r["populations"]["a9_words"]["left_hand"]["5.0"]["mean"] == .5 and r["populations"]["real_with_neutral_fill"]["both"]["5.0"]["mean"] > r["populations"]["real_tracked_frames"]["both"]["5.0"]["mean"]
    assert r["populations"]["real_with_nearest_fill"]["both"]["5.0"]["mean"] < r["populations"]["real_with_neutral_fill"]["both"]["5.0"]["mean"]
    assert r["word_alignment"]["a9_words"]["entries"] == 1 and r["word_alignment"]["a9_words"]["entries_without_training_clips"] == 1 and r["word_alignment"]["a9_words"]["thresholds"]["5.0"]["difference"] == .5      # the right hand is neutral nowhere
    assert r["a9_predicted_minus_real_tracked"]["5.0"]["difference"] > 0 and r["a3_top10_predicted_minus_real_tracked"]["5.0"] is not None and r["flag_degrees"] == 5. and r["thresholds_degrees"] == list(W.THRESHOLDS)
    assert r["code_digest"] == W.code_digest() and r["peak_rss_bytes"] > 0
    with pytest.raises(SystemExit, match="write-once"):
        W.main(["neutral", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w")])


def test_a_smoke_of_the_motion_measure_reads_c02_only_and_no_cache_or_composed_form(monkeypatch, tmp_path):
    harness(monkeypatch, n=4, declared=False)
    for name in ("cache_entries", "load_composed", "word_training_clips"):
        monkeypatch.setattr(W, name, lambda *a, **k: pytest.fail(f"a smoke reads no {name}"))
    seen = {}
    monkeypatch.setattr(L, "load_manifest_at", lambda s, k, sha: dict(records=[]))
    monkeypatch.setattr(W, "corpus_neutral", lambda manifest, cache, fetch, **k: seen.update(k) or {})
    assert W.main(["neutral", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w"), "--smoke-train", "4"]) == 0
    assert seen == dict(clips=4, components=("c02",)) and json.loads((tmp_path / "w/neutral.json").read_text())["declared_read"] is False


def test_the_digest_names_this_job():
    from signtranslator.eval import provenance
    W.code_digest.cache_clear()
    assert W.code_digest() == provenance.code_digest(extra_files=("runpod/fleet/tok_v1.sh",)) and W.code_digest() != provenance.code_digest()


def stored_form(refs, nll=1.):
    return dict(signs=[f"s{i}" for i in range(100)], references=refs, hypotheses=refs, nll=[[nll]] * 100)


def path_setup(monkeypatch, tmp_path, *, hyp_change=0, nll=0., declared=True, order=False):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "s")
    refs = [f"w{i} a" for i in range(100)]
    pins = {}
    out = tmp_path / "f"
    out.mkdir()
    for form in ("identity", "round_trip", "v0_nearest_fill"):
        raw = json.dumps(stored_form(refs)).encode()
        store.put(f"stored/{form}", raw)
        pins[form] = (f"stored/{form}", hashlib.sha256(raw).hexdigest())
        read = stored_form(refs, 1. + (nll if form == "round_trip" else 0.))
        if form == "v0_nearest_fill":
            read["hypotheses"] = [r + " x" if i < hyp_change else r for i, r in enumerate(refs)]
        if order and form == "identity":
            read["signs"] = list(reversed(read["signs"]))
        (out / f"form-{form}.json").write_text(json.dumps(dict(read, form=form, declared_read=declared, code_digest="d", environment=dict(gpu="g"))))
    monkeypatch.setattr(W, "STORED_FORMS", pins)
    return store, out


def path_check(tmp_path, out):
    return W.main(["path-check", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(out), "--void-prefix", "v/void"])


def test_the_path_check_reads_every_anchor_passes_at_its_limits_and_voids_on_a_miss(monkeypatch, tmp_path):
    store, out = path_setup(monkeypatch, tmp_path, hyp_change=2, nll=.0009)
    assert path_check(tmp_path, out) == 0
    r = json.loads((out / "path_check.json").read_text())
    assert set(r["reads"]) == {"identity", "round_trip", "v0_nearest_fill"} and r["reproduced"] and not store.list("v/void")


@pytest.mark.parametrize("kw,miss", [(dict(hyp_change=3), "v0_nearest_fill hypotheses"), (dict(nll=.0012), "round_trip NLL"), (dict(nll=-.0012), "round_trip NLL")])
def test_a_path_miss_exits_3_and_stores_a_void(monkeypatch, tmp_path, kw, miss):
    store, out = path_setup(monkeypatch, tmp_path, **kw)
    assert path_check(tmp_path, out) == 3
    assert json.loads((out / "path_check.json").read_text())["misses"] == [miss] and store.exists("v/void/d.json")


def test_the_path_check_refuses_other_clips_a_changed_pin_and_a_smoke_checks_pins_only(monkeypatch, tmp_path, capsys):
    _, out = path_setup(monkeypatch, tmp_path, order=True)
    with pytest.raises(SystemExit, match="other clips"):
        path_check(tmp_path, out)
    other = tmp_path / "second"
    other.mkdir()
    _, out = path_setup(monkeypatch, other, declared=False, hyp_change=50)
    assert path_check(other, out) == 0 and "not applicable" in capsys.readouterr().out and not (out / "path_check.json").exists()
    monkeypatch.setitem(W.STORED_FORMS, "identity", ("stored/identity", "0" * 64))
    with pytest.raises(SystemExit, match="SHA-256"):
        path_check(other, out)


def test_the_stored_pins_are_the_probes_and_v0s():
    assert W.STORED_FORMS["identity"][1].startswith("24958feb") and W.STORED_FORMS["round_trip"][1].startswith("18e6b175") and W.STORED_FORMS["v0_nearest_fill"][1].startswith("e1bfd0ac")
    assert W.A10_SHA.startswith("1e912c9a") and W.A9_CACHE_SHA.startswith("507ee9cc") and W.A3_CACHE_SHA.startswith("2d614f41") and W.TOKENS_SHA.startswith("c1105410")


def composed_setup(monkeypatch, tmp_path, change=0, *, declared=True):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "s")
    refs = [f"w{i}" for i in range(100)]
    raw = json.dumps(dict(signs=[f"s{i}" for i in range(100)], forms={W.COMPOSED_FORM: dict(hypotheses=refs)})).encode()
    store.put(W.A10_REPORT, raw)
    monkeypatch.setattr(W, "A10_SHA", hashlib.sha256(raw).hexdigest())
    out = tmp_path / "f"
    out.mkdir()
    (out / "form-a9p.json").write_text(json.dumps(dict(form="a9p", declared_read=declared, signs=[f"s{i}" for i in range(100)], hypotheses=[r + " x" if i < change else r for i, r in enumerate(refs)], code_digest="d")))
    return out


@pytest.mark.parametrize("change,good", [(0, True), (2, True), (3, False)])
def test_the_composed_check_reports_and_does_not_stop(monkeypatch, tmp_path, change, good, capsys):
    out = composed_setup(monkeypatch, tmp_path, change)
    assert W.main(["composed-check", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(out)]) == 0
    assert json.loads((out / "composed_check.json").read_text())["reproduced"] is good and ("ALERT" in capsys.readouterr().out) is (not good)


def test_the_composed_check_is_a_pin_check_on_a_smoke(monkeypatch, tmp_path, capsys):
    out = composed_setup(monkeypatch, tmp_path, declared=False)
    assert W.main(["composed-check", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(out)]) == 0 and not (out / "composed_check.json").exists()
    monkeypatch.setattr(W, "A10_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        W.main(["composed-check", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(out)])


@pytest.mark.parametrize("kw,code", [(dict(report_only=True, checks=dict(blank_identical_to_real=.1)), 0), (dict(report_only=False, checks=dict(blank_identical_to_real=.1)), 3), (dict(report_only=True, checks=dict(blank_identical_to_real=1.)), 3),
                                     (dict(report_only=True, declared_read=True, neutral=dict(a=1), checks=dict(blank_identical_to_real=.1, canary=dict(reproduced=True), path_check=dict(reproduced=True))), 0),
                                     (dict(report_only=True, declared_read=True, checks=dict(blank_identical_to_real=.1, canary=dict(reproduced=True), path_check=dict(reproduced=True))), 3),
                                     (dict(report_only=True, declared_read=True, neutral=dict(a=1), checks=dict(blank_identical_to_real=.1, path_check=dict(reproduced=True))), 3),
                                     (dict(report_only=True, declared_read=True, neutral=dict(a=1), checks=dict(blank_identical_to_real=.1, canary=dict(reproduced=True))), 3)])
def test_check_judges_plumbing(tmp_path, kw, code):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(kw))
    assert W.main(["check", "--report", str(path)]) == code


def test_the_export_list_is_d20s_signs_in_order_or_its_first_few(monkeypatch, tmp_path, capsys):
    harness(monkeypatch, n=6)
    out = tmp_path / "x.json"
    assert W.main(["export-list", "--store", f"dir:{tmp_path / 's'}", "--out", str(out), "--first", "2"]) == 0
    sha = capsys.readouterr().out.strip()
    assert json.loads(out.read_text())["signs"] == ["s0", "s1"] and sha == hashlib.sha256(out.read_bytes()).hexdigest()


# ---- the job ---------------------------------------------------------------------------------------------------------------------------

def run_job(tmp_path, fail_when=None, code=7, stored=False, **env):
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    pattern = "*" + "*".join(f'"{x}"' for x in fail_when.split("*")) + "*" if fail_when else ""
    match = f'case "$*" in {pattern}) echo "planted failure" >&2; exit {code};; esac' if fail_when else ""
    (bin_dir / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\n{match}\ncase "$*" in *"tok_v1.py stored"*) exit {0 if stored else 1};; *"export-list"*) echo aaaa;; esac\nexit 0\n')
    (bin_dir / "uv").write_text("#!/bin/bash\nexit 0\n")
    for name in ("python", "uv"):
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    environment = dict(PATH=f"{bin_dir}:/usr/bin:/bin", HOME=str(tmp_path), VENV_BIN=str(bin_dir), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path), WORK=str(tmp_path / "work"), **env)
    done = subprocess.run(["bash", str(JOB)], env=environment, capture_output=True, text=True, timeout=60)
    return done, log.read_text() if log.exists() else ""


def done_line(result) -> int:
    lines = [x for x in result.stdout.splitlines() if x.startswith("TOK-V1-DONE ")]
    assert len(lines) == 1 and lines[0] == result.stdout.strip().splitlines()[-1]
    return int(lines[0].split()[1])


def test_the_runner_sees_no_credentials():
    assert not [k for k in os.environ if k.startswith(("R2_", "AWS_", "RUNPOD_"))]


def test_the_job_smokes_on_train_first_then_reads_the_anchors_first(tmp_path):
    result, calls = run_job(tmp_path)
    assert result.returncode == 0 and done_line(result) == 0
    own = [x for x in calls.splitlines() if x.startswith("scripts/tok_v1.py ")]
    parsed = [W.build_parser().parse_args(x.split()[1:]) for x in own]
    order = [a.command for a in parsed if a.command not in ("stored", "export-list")]
    assert order == ["read", "neutral", "canary", "path-check", "composed-check", "merge", "check", "upload", "preflight", "read", "canary", "path-check", "read", "neutral", "read", "composed-check", "merge", "upload", "check"]
    reads = [a for a in parsed if a.command == "read"]
    assert reads[0].smoke_train == 8 and reads[0].smoke_longest == 4 and reads[0].smoke_never_tracked == 2 and reads[0].forms == ",".join(W.FORMS)
    assert reads[1].forms == "real,blank,identity,round_trip,v0_nearest_fill" and reads[1].smoke_train == 0 and reads[2].forms == "rt_nearest,no_quant_nearest" and reads[3].forms == "a9p,a9p_replaced"
    assert all(a.device == "cuda" and a.nll_group == 4 and a.resume for a in reads[1:])
    exports = [x for x in calls.splitlines() if "round15_compose_eval.py" in x]
    assert len(exports) == 4 and all("--export-only" in x and "--likelihood" not in x and "core_crossfade@predicted" in x for x in exports)
    a9, a3 = exports[2], exports[3]
    assert "--core-margin 0.1" in a9 and "--clips 500" in a9 and "tok-probes-v1/composed/a9" in a9 and "word-cache-extra" not in a9
    assert "--core-margin 0 " in a3 and "--vocabulary-extra-top 10" in a3 and "--word-cache-extra-sha256 2d614f41" in a3 and "tok-probes-v1/composed/a3-top10" in a3
    assert all("--clips 5 " in x and "/smoke/" in x for x in exports[:2])


def test_a_stored_export_is_not_composed_again(tmp_path):
    _, calls = run_job(tmp_path, stored=True)
    assert "round15_compose_eval.py" not in calls


def test_smoke_only_stops_before_any_declared_read(tmp_path):
    result, calls = run_job(tmp_path, SMOKE="1")
    assert result.returncode == 0 and done_line(result) == 0 and "/d20/forms" not in calls and "preflight" not in calls and "tok-probes-v1/composed" not in calls


@pytest.mark.parametrize("fail_when,expected", [("tok_v1.py read*smoke-", 2), ("tok_v1.py neutral*smoke-", 2), ("tok_v1.py canary*smoke-", 2), ("tok_v1.py path-check*smoke-", 2), ("tok_v1.py composed-check*smoke-", 2), ("tok_v1.py merge*smoke-", 2),
                                                ("check --report*smoke-", 2), ("round15_compose_eval.py*smoke/", 2), ("upload --out-dir*smoke-", 2), ("preflight", 1), ("--forms real,blank", 1), ("tok_v1.py canary*d20", 1), ("tok_v1.py path-check*d20", 1),
                                                ("--forms rt_nearest", 1), ("tok-probes-v1/composed/a9 --export", 1), ("tok-probes-v1/composed/a3-top10 --export", 1), ("tok_v1.py neutral --out-dir*d20", 1), ("--forms a9p", 1), ("tok_v1.py composed-check*d20", 1), ("tok_v1.py merge*d20", 1),
                                                ("upload --out-dir*d20", 1), ("check --report*d20", 1)])
def test_any_failure_exits_nonzero_with_its_done_line_and_uploads_its_work(tmp_path, fail_when, expected):
    result, calls = run_job(tmp_path, fail_when=fail_when)
    assert result.returncode == expected and done_line(result) == expected, result.stdout + result.stderr
    assert "upload --partial" in calls.splitlines()[-1]


@pytest.mark.parametrize("fail", ["tok_v1.py canary*d20", "tok_v1.py path-check*d20"])
def test_a_void_stops_before_any_other_form_with_its_own_status(tmp_path, fail):
    result, calls = run_job(tmp_path, fail_when=fail, code=3)
    assert result.returncode == 3 and done_line(result) == 3 and "do not retry" in result.stdout
    assert "--forms rt_nearest" not in calls and "tok-probes-v1/composed/a9 " not in calls and len(calls.splitlines()) >= 2 and "upload --partial" in calls.splitlines()[-1]


def test_a_rerun_passes_its_cause(tmp_path):
    _, calls = run_job(tmp_path, RERUN_CAUSE="why now")
    assert "--rerun-cause why now" in next(x for x in calls.splitlines() if "preflight" in x)


def test_the_job_pipes_nothing_names_no_other_list_pins_transformers_and_is_valid_bash():
    text = JOB.read_text()
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0
    code = [line.split(" #")[0] for line in text.splitlines() if not line.startswith("#")]
    for line in code:
        assert not re.search(r"(?<!\|)\|(?!\|)", line), line
    for word in ("c20", "C20", "val1000", "eval1000", "tok_v1_rule"):
        assert word not in "\n".join(code)
    assert "transformers==4.57.6" in text and "set -uo pipefail" in text and "PIN_" not in text and "PIN_" not in (ROOT / "scripts/tok_v1.py").read_text()


def test_the_script_holds_no_decision_rule():
    source = (ROOT / "scripts/tok_v1.py").read_text()
    for word in ("outcome", "not_detected", "no_rise", ".015", ".75"):
        assert word not in source


# ---- the draft rule: one synthetic report per outcome ------------------------------------------------------------------------------------

def diff(point=.1, low=.05, high=.2):
    return dict(point=point, ci95=[low, high])


def excess(main=.1, low=.05, high=.2, near=.05, far=.2):
    def row(d, a, b):
        return dict(difference=d, ci95=[a, b])
    return {"1.0": None, "2.0": row(near, near - .01, near + .01), "5.0": row(main, low, high), "10.0": row(far, far - .01, far + .01), "20.0": None}


def rule_report(cap_share=None, gen=None, align=(.1, .2), sens=None, composed=True, **top):
    place = dict(gpu="g")
    out = dict(report_only=True, declared_read=True, code_digest="d", environment=place, reads=dict(likelihood_minus_blank=dict(cap_share_of_nearest_fill=cap_share or diff(.5, .3, .7), a9p_replaced_minus_a9p=sens or diff(.1, .05, .2))),
               neutral=dict(a9_predicted_minus_real_tracked=gen or excess(), word_alignment=dict(a9_words=dict(thresholds={"5.0": dict(ci95=list(align), difference=sum(align) / 2)}))),
               checks=dict(blank_identical_to_real=.1, canary=dict(reproduced=True, code_digest="d", environment=place), path_check=dict(reproduced=True, code_digest="d"), composed_check=dict(reproduced=composed)))
    out.update(top)
    return out


@pytest.mark.parametrize("value,expected", [(diff(.75, .5, 1.), "kept"), (diff(.95, .7, 1.1), "kept"), (diff(.75, .49, 1.), "partial"), (diff(.74, .6, 1.), "partial"), (diff(.25, 0., .5), "lost"), (diff(0., -.2, .2), "lost"), (diff(.25, 0., .51), "partial"), (diff(.26, 0., .4), "partial"), (dict(point=None, ci95=None), "partial")])
def test_each_cap_outcome(value, expected):
    assert Rule.read_rule(rule_report(cap_share=value))["cap"]["outcome"] == expected


@pytest.mark.parametrize("gen,expected", [(excess(.05, .01, .1), "shown"), (excess(.2, .1, .3), "shown"), (excess(.049, .01, .1), "unresolved"), (excess(.1, 0., .2), "unresolved"), (excess(.1, .05, .2, near=0.), "unresolved"), (excess(.1, .05, .2, far=0.), "unresolved"),
                                          (excess(0., -.01, .0199), "absent"), (excess(-.1, -.2, -.05), "absent"), (excess(0., -.01, .02), "unresolved")])
def test_each_generated_outcome(gen, expected):
    assert Rule.read_rule(rule_report(gen=gen))["generated"]["outcome"] == expected


def test_a_missing_excess_is_unresolved():
    gen = excess()
    gen["5.0"] = None
    assert Rule.read_rule(rule_report(gen=gen))["generated"]["outcome"] == "unresolved"


@pytest.mark.parametrize("align,expected", [((.01, .2), True), ((0., .2), False), ((-.1, .2), False)])
def test_alignment_needs_its_lower_bound_above_0(align, expected):
    assert Rule.read_rule(rule_report(align=align))["alignment"]["aligned"] is expected


@pytest.mark.parametrize("sens,expected", [(diff(.015, .001, .03), "rise"), (diff(.2, .1, .3), "rise"), (diff(.014, .001, .03), "unresolved"), (diff(.02, 0., .03), "unresolved"), (diff(0., -.01, .0149), "no_rise"), (diff(0., -.01, .015), "unresolved"), (dict(point=None, ci95=None), "unresolved")])
def test_each_sensitivity_outcome(sens, expected):
    assert Rule.read_rule(rule_report(sens=sens))["sensitivity"]["outcome"] == expected


def test_the_sensitivity_is_void_when_a9p_did_not_reproduce():
    r = Rule.read_rule(rule_report(composed=False))
    assert r["sensitivity"]["outcome"] == "void" and r["void"] is False and r["cap"]["outcome"] == "partial"


@pytest.mark.parametrize("kw,expected", [(dict(), "supported"), (dict(align=(0., .2)), "open"), (dict(sens=diff(0., -.01, .01)), "refuted_for_generated_signing"), (dict(sens=diff(.02, 0., .03)), "open"), (dict(gen=excess(0., -.01, .01)), "refuted_for_generated_signing"),
                                         (dict(gen=excess(.1, 0., .2), sens=diff(0., -.01, .01)), "open"), (dict(composed=False), "open")])
def test_the_hypothesis_is_supported_refuted_or_open(kw, expected):
    assert Rule.read_rule(rule_report(**kw))["hypothesis"] == expected


GOOD = dict(reproduced=True, code_digest="d", environment=dict(gpu="g"))


@pytest.mark.parametrize("change,reason", [(dict(checks=dict(blank_identical_to_real=.1, canary=dict(GOOD, reproduced=False), path_check=GOOD)), "canary"), (dict(checks=dict(blank_identical_to_real=.1, canary=GOOD, path_check=dict(GOOD, reproduced=False))), "path check"),
                                           (dict(report_only=False), "check nonzero"), (dict(neutral=None), "check nonzero"), (dict(code_digest=None), "no code digest"), (dict(environment=None), "no code digest"), (dict(code_digest="z"), "mixed"), (dict(environment=dict(gpu="z")), "mixed")])
def test_each_void_voids_all(change, reason):
    r = Rule.read_rule(rule_report(**change))
    assert r["void"] is True and any(reason in x for x in r["reasons"]) and {r["cap"]["outcome"], r["generated"]["outcome"], r["sensitivity"]["outcome"], r["hypothesis"]} == {"void"}


def test_the_rule_refuses_a_smoke_says_it_is_a_draft_and_opens_no_store(tmp_path, capsys):
    with pytest.raises(SystemExit, match="smoke"):
        Rule.read_rule(rule_report(declared_read=False))
    path = tmp_path / "r.json"
    path.write_text(json.dumps(rule_report()))
    assert Rule.main(["--report", str(path)]) == 0
    assert "draft" in json.loads(capsys.readouterr().out)["status"]
    source = (ROOT / "scripts/tok_v1_rule.py").read_text()
    for word in ("make_store", "R2Store", "boto"):
        assert word not in source
    assert (Rule.HIGH, Rule.LOW, Rule.BOUND, Rule.SHOWN_POINT, Rule.ABSENT_UPPER, Rule.RISE_NATS) == (.75, .25, .5, .05, .02, .015)
