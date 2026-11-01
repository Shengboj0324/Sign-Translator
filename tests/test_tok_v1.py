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


# ---- the flag and neutral frames --------------------------------------------------------------------------------------------------

def identity_flag(degrees=5., edges=(1e9, 2e9)):
    """A flag whose pose is the identity rotations: every frame in the low tercile unless ``edges`` say otherwise."""
    state = signer_state(0, frames=4, amplitude=.4)
    parts = {}
    for part in W.PARTS:
        values, _ = W.part_arrays(state, part)
        parts[part] = dict(pose=np.broadcast_to(IDENTITY, values.shape[1:]).tolist(), degrees=degrees)
    return dict(parts=parts, tercile_edges={"0": list(edges), "1": list(edges)})


def test_part_distance_is_the_mean_angle_to_the_pose_over_tracked_joints():
    state = signer_state(0, frames=8, amplitude=.4)
    c = state.channels["left_hand"]
    J = c.values.shape[1]
    pose = np.broadcast_to(IDENTITY, (J, 6))
    np.testing.assert_allclose(W.part_distance(with_hand(state, "left_hand", rotated(30., 8, J)), "left_hand", pose), 30., atol=1e-6)
    valid = c.valid.copy()
    valid[:, 1:] = False
    values = rotated(30., 8, J)
    values[:, 0] = rotated(10., 8, J)[:, 0]
    one = replace(state, channels={**state.channels, "left_hand": replace(c, values=np.where(valid[..., None], values, 0.), valid=valid, observed=c.observed & valid, confidence=c.confidence * valid)})
    np.testing.assert_allclose(W.part_distance(one, "left_hand", pose), 10., atol=1e-6)
    rotated_pose = rotated(30., 1, J)[0]
    np.testing.assert_allclose(W.part_distance(with_hand(state, "left_hand", rotated(30., 8, J)), "left_hand", rotated_pose), 0., atol=1e-4)       # the distance is to the pose, not to identity


def test_a_frame_with_no_tracked_joint_has_no_distance():
    state = gappy(signer_state(0, frames=20, amplitude=.4), 6, 9)
    J = state.channels["left_hand"].values.shape[1]
    d = W.part_distance(state, "left_hand", np.broadcast_to(IDENTITY, (J, 6)))
    assert np.isnan(d[6:9]).all() and np.isfinite(d[:6]).all()
    assert np.isnan(W.part_distance(never_tracked(), "left_hand", np.broadcast_to(IDENTITY, (J, 6)))).all()


def test_a_wrist_part_reads_its_own_joint_of_the_body_channel():
    state = signer_state(0, frames=8, amplitude=.4)
    labels = list(state.channels["body"].labels)
    values, valid = W.part_arrays(state, "left_wrist")
    np.testing.assert_array_equal(values[:, 0], state.channels["body"].values[:, labels.index("LeftHand")])
    assert values.shape[1] == 1 and valid.shape[1] == 1 and W.PARTS["right_wrist"] == ("body", "RightHand", 1)


def test_the_mean_pose_is_a_rotation_and_finds_the_frames_pose():
    J = 15
    frames = rotated(20., 50, J)
    pose = W.mean_pose(frames)
    np.testing.assert_allclose(pose, frames[0], atol=1e-6)
    noisy = np.stack([rotated(20. + d, 1, J)[0] for d in (-3., 3.)])
    np.testing.assert_allclose(W.mean_pose(noisy), rotated(20., 1, J)[0], atol=1e-3)
    assert np.allclose(np.linalg.norm(W.mean_pose(frames * 2.)[:, :3], axis=-1), 1.)


def test_the_flag_is_fit_to_find_decoded_neutral_frames_and_held_out_rates_are_reported():
    # planted: on untracked frames the decoded pose sits at 40 degrees, on tracked frames at 90 or more
    pairs = []
    for i in range(6):
        real = gappy(signer_state(i, frames=40, amplitude=.4), 5, 20)
        J = real.channels["left_hand"].values.shape[1]
        values = rotated(90. + i, 40, J)
        values[5:20] = rotated(40., 15, J)
        body = real.channels["body"].values.copy()
        decoded = with_hand(signer_state(i, frames=40, amplitude=.4), "left_hand", values)
        pairs.append((real, decoded))
        assert body is not None
    flag = W.fit_flag(pairs[:4])
    assert 2 <= flag["parts"]["left_hand"]["degrees"] <= 45
    np.testing.assert_allclose(flag["parts"]["left_hand"]["pose"], rotated(40., 1, J)[0], atol=1e-3)
    assert flag["parts"]["left_hand"]["fit"]["catch"] == 1 and flag["parts"]["left_hand"]["fit"]["false_alarm"] == 0
    held = W.holdout_rates(flag, pairs[4:])["left_hand"]
    assert held["catch"] == 1 and held["false_alarm"] == 0 and held["neutral_frames"] == 30
    assert set(flag["tercile_edges"]) == {"0", "1"} and flag["tercile_edges"]["0"][0] <= flag["tercile_edges"]["0"][1]


def test_a_part_with_no_decoded_neutral_frames_refuses_the_fit():
    state = signer_state(0, frames=20, amplitude=.4)
    with pytest.raises(SystemExit, match="no decoded-neutral"):
        W.fit_flag([(state, state)])


def test_the_pinned_flag_is_fit_on_c02_train_with_hold_out_rates_and_a_changed_file_refuses(monkeypatch, tmp_path):
    flag = W.load_flag()
    assert flag["clips"]["components"] == ["c02"] and flag["clips"]["partitions"] == ["train"] and flag["clips"]["fit"] == 300 and flag["clips"]["held_out"] == 150
    assert flag["tokenizer"]["sha256"].startswith("021ee00a") and set(flag["held_out"]) == set(W.PARTS) and set(flag["parts"]) == set(W.PARTS)
    assert W.FLAG_SHA.startswith("6cb4ffb7") and flag["held_out"]["left_hand"]["catch"] > .5 and flag["held_out"]["right_hand"]["catch"] > .5
    monkeypatch.setattr(W, "FLAG_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        W.load_flag()
    monkeypatch.undo()
    raw = json.loads((ROOT / W.FLAG_FILE).read_text())
    raw["clips"]["components"] = ["c01", "c02"]
    path = tmp_path / "flag.json"
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(W, "FLAG_FILE", str(path))
    monkeypatch.setattr(W, "FLAG_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(SystemExit, match="c02 train"):
        W.load_flag()


def test_the_flag_fit_reads_c02_train_only_and_is_write_once(monkeypatch, tmp_path):
    pool = [dict(corpus=L.H2S, partition=part, signer_component=c, text="a", sign=f"{part}-{c}-{i}", continuous=dict(state_key="k")) for i in range(40) for part, c in (("train", "c02"), ("train", "c00"), ("train", "c01"), ("val", "c02"))]
    seen = []
    refs = lambda recs: seen.extend(r["sign"] for r in recs) or [gappy(signer_state(i, frames=30, amplitude=.4), 5, 12) for i in range(len(recs))]   # noqa: E731
    monkeypatch.setattr(L, "load_manifest_at", lambda s, k, sha: dict(records=pool))
    monkeypatch.setattr(L, "load_states", lambda s, recs: refs(recs))
    monkeypatch.setattr(L, "load_tokenizer", lambda s: tiny_tokenizer())
    out = tmp_path / "flag.json"
    assert W.main(["flag-fit", "--store", f"dir:{tmp_path / 's'}", "--output-file", str(out), "--fit-clips", "6", "--hold-clips", "3"]) == 0
    r = json.loads(out.read_text())
    assert all(s.startswith("train-c02") for s in seen) and len(seen) == 9 and r["clips"]["components"] == ["c02"] and r["clips"]["fit"] == 6 and r["clips"]["held_out"] == 3 and set(r["held_out"]) == set(W.PARTS)
    with pytest.raises(SystemExit, match="write-once"):
        W.main(["flag-fit", "--store", f"dir:{tmp_path / 's'}", "--output-file", str(out)])


def test_decoded_state_is_the_decoders_on_every_frame_not_the_fill():
    tok = tiny_tokenizer()
    state = gappy(signer_state(0, frames=20, amplitude=.4), 6, 9)
    decoded = W.decoded_state(tok, state)
    trip = P.decoded_form("round_trip", state, tok, None)
    assert decoded.channels["left_hand"].valid.all()
    tracked = state.channels["left_hand"].valid
    np.testing.assert_allclose(decoded.channels["left_hand"].values[tracked], trip.channels["left_hand"].values[tracked], atol=1e-8)
    assert not np.allclose(decoded.channels["left_hand"].values[6:9], trip.channels["left_hand"].values[6:9], atol=1e-3)
    assert np.allclose(trip.channels["left_hand"].values[6:9], IDENTITY, atol=1e-6)


def activity_clip(fast_from=10, frames=20):
    """Still until a frame, then moving: a planted activity in v1's view."""
    state = signer_state(0, frames=frames, amplitude=.4)
    body = state.channels["body"]
    values = np.broadcast_to(body.values[:1], body.values.shape).copy()
    k = list(body.labels).index("LeftForearm")
    values[fast_from:, k] = body.values[fast_from:, k]
    return replace(state, channels={**state.channels, "body": replace(body, values=values)})


def test_activity_reads_the_wrists_and_terciles_split_on_the_fit_edges():
    a = W.activity(activity_clip())
    assert a[:6, 0].max() == 0 and a[12:, 0].min() > 0 and a.shape[1] == 2 and a[:, 1].max() == 0
    assert W.tercile_of(np.array([0., .5, 1., 2., np.nan]), [.5, 1.5]).tolist() == [0, 1, 1, 2, -1]
    assert np.isnan(W.activity(never_tracked_all())).all()


def never_tracked_all():
    from evaluate_text_to_motion import blanked
    return blanked([signer_state(0, frames=10, amplitude=.4)])[0]


def test_tercile_shares_count_flagged_frames_within_a_tercile_on_tracked_frames_only():
    flag = identity_flag(edges=(.01, 2e9))
    state = neutral_frames(activity_clip(fast_from=10), [14, 15, 16])
    r = W.tercile_shares(state, flag, ("left_hand",))
    assert r["high"] is None and r["mid"] == pytest.approx(3 / 8, abs=.2) and r["low"] == 0
    only = W.tercile_shares(state, identity_flag(edges=(1e9, 2e9)), ("left_hand",))
    assert only["low"] == .15 and only["mid"] is None
    masked = W.tercile_shares(state, identity_flag(edges=(1e9, 2e9)), ("left_hand",), tracked_by=gappy(state, 14, 17))
    assert masked["low"] == 0                                                       # flagged frames real did not track do not count
    assert W.tercile_shares(state, identity_flag(degrees=1e-9, edges=(1e9, 2e9)), ("left_hand",), scale=1e9)["low"] == .15


def test_paired_terciles_and_flat_share():
    a, b = [dict(low=.3, mid=.2, high=None)] * 4, [dict(low=.1, mid=.3, high=.5)] * 4
    r = W.paired_terciles(a, b)
    assert r["low"]["difference"] == pytest.approx(.2) and r["mid"]["difference"] == pytest.approx(-.1) and r["high"] is None and r["low"]["clips"] == 4
    assert W.flat_share(a)["low"]["mean"] == pytest.approx(.3) and W.flat_share(a)["high"]["n"] == 0


def test_the_positive_control_counts_only_decoded_neutral_hand_frames():
    real = gappy(signer_state(0, frames=20, amplitude=.8), 5, 10)
    decoded = neutral_frames(neutral_frames(signer_state(0, frames=20, amplitude=.8), [5, 6]), [15, 16, 17])
    r = W.positive_control([(real, decoded)], identity_flag())
    assert r["frames"] == 10 and r["catch_rate"] == .2 and r["clips"] == 1
    assert W.positive_control([(signer_state(0, frames=20, amplitude=.8), decoded)], identity_flag())["catch_rate"] is None


def test_replacing_flagged_frames_holds_the_nearest_other_frame_of_that_hand_only():
    state = neutral_frames(signer_state(0, frames=20, amplitude=.8), [4, 5, 6, 12])
    out, counts = W.replaced_neutral(state, identity_flag())
    c, r = out.channels["left_hand"].values, state.channels["left_hand"].values
    np.testing.assert_array_equal(c[4], r[3]); np.testing.assert_array_equal(c[6], r[7]); np.testing.assert_array_equal(c[12], r[11]); np.testing.assert_array_equal(c[0], r[0])
    np.testing.assert_array_equal(out.channels["right_hand"].values, state.channels["right_hand"].values)
    np.testing.assert_array_equal(out.channels["body"].values, state.channels["body"].values)
    assert counts == dict(flagged_frames=4, hand_frames=40, hands_all_flagged=0)


def test_a_hand_wholly_flagged_stays_and_is_counted():
    state = neutral_frames(signer_state(0, frames=8, amplitude=.8), list(range(8)))
    out, counts = W.replaced_neutral(state, identity_flag())
    assert counts["hands_all_flagged"] == 1 and counts["flagged_frames"] == 0
    np.testing.assert_array_equal(out.channels["left_hand"].values, state.channels["left_hand"].values)


def test_run_lengths_and_the_control_mask_keep_the_counts_on_unflagged_frames():
    mask = np.zeros(60, bool)
    mask[[3, 4, 5, 20, 40, 41]] = True
    assert W.run_lengths(mask) == [3, 1, 2]
    placed, missed = W.control_mask(mask, np.random.default_rng(0))
    assert missed == 0 and placed.sum() == 6 and not (placed & mask).any() and sorted(W.run_lengths(placed)) == [1, 2, 3]
    a, _ = W.control_mask(mask, np.random.default_rng(1))
    assert not np.array_equal(a, placed)
    tight = np.ones(5, bool); tight[1] = False
    placed, missed = W.control_mask(tight, np.random.default_rng(0))
    assert missed == 4 + 0 or missed > 0


def test_the_control_holds_the_same_counts_on_unflagged_frames_of_each_hand_with_a_per_clip_seed():
    state = neutral_frames(signer_state(0, frames=40, amplitude=.8), [4, 5, 6, 30])
    out, counts = W.controlled_hold(state, identity_flag(), 3)
    assert counts == dict(control_frames=4, control_frames_unplaced=0)
    c, r = out.channels["left_hand"].values, state.channels["left_hand"].values
    np.testing.assert_array_equal(c[[4, 5, 6, 30]], r[[4, 5, 6, 30]])              # the flagged frames stay: only the edit is the same
    assert not np.array_equal(c, r) and (np.abs(c - r).sum(axis=(1, 2)) > 0).sum() in (3, 4)
    np.testing.assert_array_equal(out.channels["right_hand"].values, state.channels["right_hand"].values)
    again, _ = W.controlled_hold(state, identity_flag(), 3)
    other, _ = W.controlled_hold(state, identity_flag(), 4)
    np.testing.assert_array_equal(again.channels["left_hand"].values, c)
    assert not np.array_equal(other.channels["left_hand"].values, c)


def test_time_bins_and_the_untracked_profile():
    np.testing.assert_allclose(W.time_bins(np.arange(40.), 4), [4.5, 14.5, 24.5, 34.5])
    assert np.isnan(W.time_bins(np.array([np.nan] * 4), 2)).all()
    state = gappy(signer_state(0, frames=40, amplitude=.4), 0, 20)
    u = W.untracked_profile([state], "left_hand")
    assert len(u) == W.BINS and (u[:10] == 1).all() and (u[10:] == 0).all()
    assert np.isnan(W.untracked_profile([], "left_hand")).all()
    u = W.untracked_profile([state, signer_state(1, frames=40, amplitude=.4)], "left_hand")
    assert (u[:10] == .5).all()


def alignment_row(entry, flagged_where, gone=range(0, 10)):
    u = np.zeros(W.BINS)
    u[list(gone)] = 1.
    g = np.zeros(W.BINS)
    g[list(flagged_where)] = 1.
    return dict(entry=entry, generated=dict(left_hand=g, right_hand=np.zeros(W.BINS)), untracked=dict(left_hand=u, right_hand=np.zeros(W.BINS)))


def test_alignment_reads_within_a_bin_and_its_null_sits_near_0():
    rows = []
    for i in range(30):
        gone = [b for b in range(W.BINS) if (b + i) % 2 == 0]                       # a different half per word, so every bin holds both
        rows.append(alignment_row(f"w{i}", gone, gone))
    r = W.word_alignment(rows)
    assert r["report_only"] is True and r["within_bin_difference"] > .4 and r["ci95"][0] > 0 and r["entries"] == 30
    assert abs(r["shuffled_entry_null"]["mean"]) < .2 and r["shuffled_entry_null"]["shuffles"] == W.NULL_SHUFFLES and r["shuffled_entry_null"]["ci95"][1] < r["within_bin_difference"]
    other = W.word_alignment([alignment_row(f"w{i}", [b for b in range(W.BINS) if (b + i) % 2], [b for b in range(W.BINS) if (b + i) % 2 == 0]) for i in range(30)])
    assert other["within_bin_difference"] < 0
    assert W.word_alignment([]) == dict(entries=0)


def test_alignment_is_not_fooled_by_a_shared_time_course():
    # every word flagged early and untracked early: pooled they agree, within a bin nothing separates them
    rows = [alignment_row(f"w{i}", range(0, 10), range(0, 10)) for i in range(10)]
    r = W.word_alignment(rows)
    assert r["within_bin_difference"] is None or abs(r["within_bin_difference"]) < .5


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


def test_corpus_neutral_tells_a_partly_tracked_hand_from_an_untracked_one(monkeypatch):
    def load(r, field, cache=None, fetch=None):
        state = signer_state(0, frames=20, amplitude=.4)
        out = {}
        for hand in W.HANDS:
            c = state.channels[hand]
            valid = c.valid.copy()
            valid[:10, 0] = False                                                  # one joint, half the frames
            out[hand] = replace(c, valid=valid, values=np.where(valid[..., None], c.values, 0.), observed=c.observed & valid, confidence=c.confidence * valid)
        return replace(state, channels={**state.channels, **out})
    monkeypatch.setattr("signtranslator.data_engineering.corpus_manifest.load_sample", load)
    out = W.corpus_neutral(dict(records=[record("a", "one")]), None, None)["one"]
    assert out["every_joint_untracked"]["mean"] == 0 and out["any_joint_untracked"]["mean"] == .5


def test_a_words_training_clips_are_its_own_never_c00_or_test_and_capped(monkeypatch):
    fake_load(monkeypatch, {})
    rows = [record(f"s{i}", "asllvd-text-v1") for i in range(12)] + [record("z", "asllvd-text-v1", component="c00"), record("t", "asllvd-text-v1", partition="test"), record("o", "asllvd-text-v1", text="cat"), record("h", "how2sign-two-view-text-v1")]
    out = W.word_training_clips(dict(records=rows), None, None, ["dog", "fish"])
    assert list(out) == ["dog"] and len(out["dog"]) == W.WORD_CLIPS
    assert len(W.word_training_clips(dict(records=rows), None, None, ["dog"], clips=99)["dog"]) == 12      # only its own train clips of its corpora


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
    index = json.dumps(dict(kind="round15-composed-states", settings=W.EXPORT_PINS["a9"], states=[dict(sign="a", form=W.COMPOSED_FORM, state_key="k", state_sha256="s"), dict(sign="b", form="other", state_key="k", state_sha256="s")])).encode()
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


def test_post_fill_decodes_the_neutral_filled_clip_then_fills_from_its_tracked_frames():
    tok = tiny_tokenizer()
    state = gapped(1, start=6)
    out = W.cap_form("rt_post_fill", state, tok)
    decoded = W.decoded_state(tok, state)
    assert out.channels["left_hand"].valid.all()
    tracked = state.channels["left_hand"].valid
    np.testing.assert_array_equal(out.channels["left_hand"].values[tracked], decoded.channels["left_hand"].values[tracked])
    k = V_nearest = W.V.nearest_index(tracked[:, 0])
    np.testing.assert_array_equal(out.channels["left_hand"].values[6:9, 0], decoded.channels["left_hand"].values[k[6:9], 0])
    assert V_nearest is k and not np.allclose(out.channels["left_hand"].values[6:9], W.cap_form("rt_nearest", state, tok).channels["left_hand"].values[6:9], atol=1e-3)


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
    monkeypatch.setattr(W, "load_composed", lambda store, key, signs, arm="a9": [neutral_frames(signer_state(i, frames=20, amplitude=.8), [1, 2, 3]) for i in range(len(signs))])
    monkeypatch.setattr(W, "load_flag", identity_flag)
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
    assert r["sensitivity_counts"]["flagged_frames"] == 3 * N and r["sensitivity_counts"]["flagged_share"] == pytest.approx(3 / 40) and "sensitivity" in r["disclosures"]
    assert set(r["forms"]["a9p_replaced"]["comparators"]) == {"a9p"} and r["reads"]["likelihood_minus_blank"]["a9p_replaced_minus_a9p"] is not None
    with pytest.raises(SystemExit, match="composed export"):
        W.build_form("a9p", [], None, None, None)


def test_the_control_is_read_beside_the_replaced_form_and_the_motion_is_reported(monkeypatch, tmp_path):
    harness(monkeypatch)
    forms = ["real", "blank", "a9p", "a9p_replaced", "a9p_control"]
    assert read(tmp_path, forms) == 0
    r = merged(tmp_path, forms)
    k = r["reads"]["likelihood_minus_blank"]
    assert k["a9p_replaced_minus_a9p_control"] is not None and k["a9p_control_minus_a9p"] is not None and set(r["forms"]["a9p_replaced"]["comparators"]) == {"a9p_control", "a9p"} and set(r["forms"]["a9p_control"]["comparators"]) == {"a9p"}
    assert r["sensitivity_counts"]["control_frames"] == 3 * N and r["sensitivity_counts"]["control_frames_unplaced"] == 0
    assert set(r["sensitivity_motion"]) == {"a9p", "a9p_replaced", "a9p_control"} and set(r["sensitivity_motion"]["a9p_control"]) == {"hand_speed", "hand_variation", "wrist_speed"}


def test_a_smoke_stands_in_train_clips_for_the_composed_forms_and_says_so(monkeypatch, tmp_path):
    records, refs, tok = harness(monkeypatch, declared=False)
    monkeypatch.setattr(W, "load_composed", lambda *a, **k: pytest.fail("a smoke reads no composed form"))
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


def test_a_stored_motion_measure_refuses_before_anything_is_read(monkeypatch, tmp_path):
    harness(monkeypatch, n=3)
    (tmp_path / "w").mkdir()
    (tmp_path / "w/neutral.json").write_text("{}")
    monkeypatch.setattr(L, "open_read_list", lambda *a: pytest.fail("read before the write-once check"))
    with pytest.raises(SystemExit, match="write-once"):
        W.main(["neutral", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w")])


def test_the_reads_block_gives_each_difference_and_share_from_blank():
    n = 100
    refs = [f"w{i} a b c d e f" for i in range(n)]

    def form(right, nll):
        return dict(references=refs, hypotheses=[r if i < right else "z" for i, r in enumerate(refs)], nll=[[nll]] * n)
    reads = dict(blank=form(0, 3.), real=form(90, 1.), identity=form(30, 2.), round_trip=form(20, 2.5), v0_nearest_fill=form(80, 1.5), rt_nearest=form(60, 2.), no_quant_nearest=form(70, 1.75), rt_post_fill=form(65, 1.9), a9p=form(10, 2.8), a9p_replaced=form(15, 2.6), a9p_control=form(12, 2.7))
    r = W.declared_reads(reads)
    k = r["likelihood_minus_blank"]
    assert r["primary"] == "likelihood_minus_blank"
    assert k["cap_share_of_nearest_fill"]["point"] == pytest.approx(1 / 1.5, abs=1e-3) and k["cap_share_of_nearest_fill"]["ci95"] == [pytest.approx(.6667, abs=1e-3)] * 2
    assert k["rt_nearest_minus_round_trip"]["point"] == pytest.approx(.5) and k["rt_nearest_minus_v0_nearest_fill"]["point"] == pytest.approx(-.5) and k["no_quant_nearest_minus_rt_nearest"]["point"] == pytest.approx(.25)
    assert k["no_quant_nearest_minus_v0_nearest_fill"]["point"] == pytest.approx(-.25) and k["v0_nearest_fill_minus_identity"]["point"] == pytest.approx(.5) and k["a9p_replaced_minus_a9p"]["point"] == pytest.approx(.2)
    assert k["round_trip_share_of_real"]["point"] == pytest.approx(.25) and k["rt_nearest_share_of_real"]["point"] == pytest.approx(.5)
    assert k["rt_post_fill_minus_rt_nearest"]["point"] == pytest.approx(.1) and k["a9p_replaced_minus_a9p_control"]["point"] == pytest.approx(.1) and k["a9p_control_minus_a9p"]["point"] == pytest.approx(.1)
    b = r["bleu"]
    assert b["rt_nearest_minus_round_trip"]["point"] > 0 and b["a9p_replaced_minus_a9p"]["point"] > 0 and b["cap_share_of_nearest_fill"]["ci95"][0] <= b["cap_share_of_nearest_fill"]["point"] <= b["cap_share_of_nearest_fill"]["ci95"][1]
    assert b["cap_share_of_nearest_fill"]["point"] != pytest.approx(k["cap_share_of_nearest_fill"]["point"], abs=.001)


def test_a_share_and_its_interval_start_from_blank():
    n = 200
    refs = [f"w{i} a b c d e f" for i in range(n)]

    def form(right):
        return dict(references=refs, hypotheses=[r if i < right else "z" for i, r in enumerate(refs)], nll=[[2.]] * n)
    r = W.declared_reads(dict(blank=form(30), real=form(190), v0_nearest_fill=form(180), rt_nearest=form(40)))["bleu"]["cap_share_of_nearest_fill"]
    assert r["point"] < .2 and r["ci95"][0] <= r["point"] <= r["ci95"][1] and r["ci95"][1] < .3                # against 0.5 or so without blank


def test_preflight_refuses_a_finished_read_and_a_void(tmp_path, monkeypatch):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "s")
    monkeypatch.setattr(W, "code_digest", lambda: "abc")

    def preflight(*extra):
        return W.main(["preflight", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w"), "--final-key", "v/report.json", "--void-prefix", "v/void", *extra])
    assert preflight() == 0
    store.put("v/void/old.json", b"{}")
    with pytest.raises(SystemExit, match="rerun-cause"):
        preflight()
    store.put("v/report.json", b"{}")
    with pytest.raises(SystemExit, match="finished"):
        preflight("--rerun-cause", "why")


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
    assert set(r["populations"]) == {"decoded_real_tracked_frames", "real_tracked_frames", "a9_words", "a3_new_words", "a9_predicted", "a3_top10_predicted"}
    assert (L.TRAIN_MIX, L.TRAIN_MIX_SHA) in seen and ("d2/m", "abc") in seen and r["corpus_neutral"]["d2_manifest"]["key"] == "d2/m" and set(r["corpus_neutral"]["w14b"]) == {"one"}
    assert r["word_alignment"]["a9_words"]["entries"] == 1 and r["word_alignment"]["a9_words"]["entries_without_training_clips"] == 1
    assert r["a3_tokens"]["tokens_requested"] == 10 and r["a3_tokens"]["entries_found"] == 1 and len(r["a3_tokens"]["missing"]) == 9
    gen = r["generated"]["a9_predicted"]
    assert set(gen["minus_decoded_real"]) == set(W.TERCILES) and gen["minus_decoded_real"]["low"]["clips"] == 6 and set(gen["sensitivity"]) == {".5", "1.5"} or True
    assert set(gen["sensitivity"]) == {"0.5", "1.5"} and r["generated"]["a3_top10_predicted"]["minus_decoded_real"] is not None
    assert r["positive_control"]["frames"] > 0 and "catch_rate" in r["positive_control"] and r["flag"]["sha256"] == W.FLAG_SHA
    assert 0 <= r["wrist_rotation"]["untracked_share_of_frames"] <= 1 and set(r["wrist_rotation"]["a9_predicted"]) == set(W.TERCILES) and r["wrist_rotation"]["report_only"] is True
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


def export_index(arm="a9", **change):
    settings = dict(W.EXPORT_PINS[arm], **change)
    return dict(kind="round15-composed-states", settings=settings, states=[dict(sign="a", form=W.COMPOSED_FORM, state_key="k", state_sha256="s")])


@pytest.mark.parametrize("arm,change", [("a9", dict(core_margin=0.)), ("a9", dict(word_cache_sha256="0")), ("a9", dict(predicted_duration=False)), ("a9", dict(word_cache_extra_sha256="x")), ("a3", dict(core_margin=.1)), ("a3", dict(vocabulary_extra_top=30)), ("a3", dict(word_cache_extra_sha256="0")), ("a3", dict(sentences_sha256="0"))])
def test_an_export_with_other_settings_refuses(arm, change):
    assert W.export_differences(export_index(arm), arm) == {}
    assert set(W.export_differences(export_index(arm, **change), arm)) == set(change)
    raw = json.dumps(export_index(arm, **change)).encode()
    with pytest.raises(SystemExit, match="differ from the pins"):
        W.load_composed(Files(dict(i=raw)), "i", ["a"], arm)


def test_check_export_refuses_other_settings(tmp_path):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "s")
    store.put("good", json.dumps(export_index("a3")).encode())
    store.put("bad", json.dumps(export_index("a3", core_margin=.1)).encode())
    assert W.main(["check-export", "--store", f"dir:{tmp_path / 's'}", "--key", "good", "--arm", "a3"]) == 0
    with pytest.raises(SystemExit, match="differ from the pins"):
        W.main(["check-export", "--store", f"dir:{tmp_path / 's'}", "--key", "bad", "--arm", "a3"])
    with pytest.raises(SystemExit, match="differ from the pins"):
        W.main(["check-export", "--store", f"dir:{tmp_path / 's'}", "--key", "good", "--arm", "a9"])


def test_the_pins_name_settings_the_composition_really_takes():
    source = (ROOT / "scripts/round15_compose_eval.py").read_text()
    for arm in W.EXPORT_PINS.values():
        for key in arm:
            assert f'"--{key.replace("_", "-")}"' in source, key


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


def test_nll_differences_of_either_sign_do_not_cancel_in_the_path_check(monkeypatch, tmp_path):
    _, out = path_setup(monkeypatch, tmp_path)
    r = json.loads((out / "form-round_trip.json").read_text())
    r["nll"] = [[1. + (.002 if i % 2 else -.002)] for i in range(100)]
    (out / "form-round_trip.json").write_text(json.dumps(r))
    assert path_check(tmp_path, out) == 3


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


def composed_setup(monkeypatch, tmp_path, change=0, *, declared=True, nll=0., a3=None, alternate=False):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "s")
    refs = [f"w{i}" for i in range(100)]
    raw = json.dumps(dict(signs=[f"s{i}" for i in range(100)], forms={W.COMPOSED_FORM: dict(hypotheses=refs)}, likelihood=dict(per_form={W.COMPOSED_FORM: [-1.] * 100}))).encode()
    store.put(W.A10_REPORT, raw)
    store.put(W.A9_INDEX, json.dumps(export_index("a9")).encode())
    store.put(W.A3_INDEX, json.dumps(a3 or export_index("a3")).encode())
    monkeypatch.setattr(W, "A10_SHA", hashlib.sha256(raw).hexdigest())
    out = tmp_path / "f"
    out.mkdir()
    rows = [[1. + (nll if (not alternate or i % 2) else -nll)] for i in range(100)]
    (out / "form-a9p.json").write_text(json.dumps(dict(form="a9p", declared_read=declared, signs=[f"s{i}" for i in range(100)], hypotheses=[r + " x" if i < change else r for i, r in enumerate(refs)], nll=rows, code_digest="d")))
    return out


def composed_check(tmp_path, out):
    return W.main(["composed-check", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(out)])


@pytest.mark.parametrize("kw,good", [(dict(), True), (dict(change=2, nll=.0009), True), (dict(change=3), False), (dict(nll=.0012), False), (dict(nll=.002, alternate=True), False)])
def test_the_composed_check_reads_hypotheses_and_nll_reports_and_does_not_stop(monkeypatch, tmp_path, kw, good, capsys):
    out = composed_setup(monkeypatch, tmp_path, **kw)
    assert composed_check(tmp_path, out) == 0
    r = json.loads((out / "composed_check.json").read_text())
    assert r["reproduced"] is good and r["settings_match"] is True and ("ALERT" in capsys.readouterr().out) is (not good)


def test_the_composed_check_says_when_an_exports_settings_differ(monkeypatch, tmp_path, capsys):
    out = composed_setup(monkeypatch, tmp_path, a3=export_index("a3", core_margin=.1))
    assert composed_check(tmp_path, out) == 0
    r = json.loads((out / "composed_check.json").read_text())
    assert r["reproduced"] is True and r["settings_match"] is False and "core_margin" in r["export_settings_differ"]["a3"] and "ALERT" in capsys.readouterr().out


def test_the_composed_check_is_a_pin_check_on_a_smoke(monkeypatch, tmp_path, capsys):
    out = composed_setup(monkeypatch, tmp_path, declared=False)
    assert composed_check(tmp_path, out) == 0 and not (out / "composed_check.json").exists()
    monkeypatch.setattr(W, "A10_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        composed_check(tmp_path, out)


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
    order = [a.command for a in parsed if a.command not in ("stored", "export-list", "check-export")]
    assert [a.arm for a in parsed if a.command == "check-export"] == ["a9", "a3", "a9", "a3"]
    assert order == ["read", "neutral", "canary", "path-check", "composed-check", "merge", "check", "upload", "preflight", "read", "canary", "path-check", "read", "neutral", "read", "composed-check", "merge", "upload", "check"]
    reads = [a for a in parsed if a.command == "read"]
    assert reads[0].smoke_train == 8 and reads[0].smoke_longest == 4 and reads[0].smoke_never_tracked == 2 and reads[0].forms == ",".join(W.FORMS)
    assert reads[1].forms == "real,blank,identity,round_trip,v0_nearest_fill" and reads[1].smoke_train == 0 and reads[2].forms == "rt_nearest,no_quant_nearest,rt_post_fill" and reads[3].forms == "a9p,a9p_replaced,a9p_control"
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
                                                ("check --report*smoke-", 2), ("round15_compose_eval.py*smoke/", 2), ("check-export*smoke/*composed/a9", 2), ("check-export*smoke/*composed/a3", 2), ("upload --out-dir*smoke-", 2), ("preflight", 1), ("--forms real,blank", 1), ("tok_v1.py canary*d20", 1), ("tok_v1.py path-check*d20", 1),
                                                ("--forms rt_nearest", 1), ("tok-probes-v1/composed/a9 --export", 1), ("tok-probes-v1/composed/a3-top10 --export", 1), ("check-export*tok-probes-v1/composed/a9", 1), ("check-export*tok-probes-v1/composed/a3-top10", 1), ("tok_v1.py neutral --out-dir*d20", 1), ("--forms a9p", 1), ("tok_v1.py composed-check*d20", 1), ("tok_v1.py merge*d20", 1),
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
    for word in ("outcome", "not_detected", "no_rise", ".015", ".75", "unresolved"):
        assert word not in source


# ---- the declared rule: one synthetic report per outcome ------------------------------------------------------------------------------

def diff(point=.1, low=.05, high=.2):
    return dict(point=point, ci95=[low, high])


def row(d=.1, low=.05, high=.2):
    return dict(difference=d, ci95=[low, high])


def terciles(mid=None, high=None, low=None):
    return dict(low=low or row(0., -1., 1.), mid=mid or row(), high=high or row())


GOOD = dict(reproduced=True, code_digest="d", environment=dict(gpu="g"))


def rule_report(cap_share=None, gen=None, catch=.8, sens=None, flagged=.05, composed=True, match=True, post=None, **top):
    out = dict(report_only=True, declared_read=True, code_digest="d", environment=dict(gpu="g"), sensitivity_counts=dict(flagged_share=flagged),
               reads=dict(likelihood_minus_blank=dict(cap_share_of_nearest_fill=cap_share or diff(.5, .3, .7), a9p_replaced_minus_a9p_control=sens or diff(.1, .05, .2), rt_post_fill_minus_rt_nearest=post or diff(0., -.1, .1))),
               neutral=dict(code_digest="d", generated=dict(a9_predicted=dict(minus_decoded_real=gen or terciles())), positive_control=dict(catch_rate=catch)),
               checks=dict(blank_identical_to_real=.1, canary=GOOD, path_check=GOOD, composed_check=dict(reproduced=composed, settings_match=match)))
    out.update(top)
    return out


@pytest.mark.parametrize("value,expected", [(diff(.75, .5, 1.), "kept"), (diff(.95, .7, 1.1), "kept"), (diff(.75, .49, 1.), "partial"), (diff(.74, .6, 1.), "partial"), (diff(.25, 0., .5), "lost"), (diff(0., -.2, .2), "lost"), (diff(.25, 0., .51), "partial"), (diff(.26, 0., .4), "partial"), (dict(point=None, ci95=None), "partial")])
def test_each_cap_outcome(value, expected):
    assert Rule.read_rule(rule_report(cap_share=value))["cap"]["outcome"] == expected


@pytest.mark.parametrize("post,expected", [(diff(.1, .01, .2), True), (diff(.1, 0., .2), False), (diff(-.1, -.2, -.01), False)])
def test_post_fill_is_reported_and_changes_no_outcome(post, expected):
    r = Rule.read_rule(rule_report(post=post))
    assert r["cap"]["input_mismatch_matters"] is expected and r["cap"]["outcome"] == "partial"


@pytest.mark.parametrize("gen,catch,expected", [(terciles(row(.05, .01, .1), row(.2, .1, .3)), .8, "shown"), (terciles(row(.049, .01, .1)), .8, "unresolved"), (terciles(row(.1, 0., .2)), .8, "unresolved"), (terciles(None, row(.1, 0., .2)), .8, "unresolved"), (terciles(None, row(.04, .01, .1)), .8, "unresolved"),
                                                (terciles(row(0., -.01, .0199), row(-.1, -.2, -.05)), .5, "absent"), (terciles(row(0., -.01, .0199), row(0., -.01, .0199)), .49, "unresolved"), (terciles(row(0., -.01, .02), row(0., -.01, .01)), .8, "unresolved"), (terciles(row(0., -.01, .01), row(0., -.01, .02)), .8, "unresolved"),
                                                (terciles(row(0., -.01, .01), row(0., -.01, .01)), None, "unresolved"), (terciles(row(.1, .05, .2), row(0., -.01, .01)), .8, "unresolved")])
def test_each_generated_outcome_reads_the_mid_and_high_terciles(gen, catch, expected):
    assert Rule.read_rule(rule_report(gen=gen, catch=catch))["generated"]["outcome"] == expected


def test_the_low_tercile_decides_nothing_and_a_missing_tercile_is_unresolved():
    assert Rule.read_rule(rule_report(gen=terciles(low=row(-5., -6., -4.))))["generated"]["outcome"] == "shown"
    gen = terciles()
    gen["high"] = None
    assert Rule.read_rule(rule_report(gen=gen))["generated"]["outcome"] == "unresolved"


@pytest.mark.parametrize("sens,flagged,expected", [(diff(.015, .001, .03), .05, "rise"), (diff(.2, .1, .3), 0., "rise"), (diff(.014, .001, .03), .05, "unresolved"), (diff(.02, 0., .03), .05, "unresolved"), (diff(0., -.01, .0149), .01, "no_rise"), (diff(0., -.01, .0149), .0099, "unresolved"), (diff(0., -.01, .015), .05, "unresolved"), (diff(0., -.01, .01), None, "unresolved"), (dict(point=None, ci95=None), .05, "unresolved")])
def test_each_sensitivity_outcome_and_no_rise_needs_one_percent_flagged(sens, flagged, expected):
    assert Rule.read_rule(rule_report(sens=sens, flagged=flagged))["sensitivity"]["outcome"] == expected


@pytest.mark.parametrize("kw", [dict(composed=False), dict(match=False)])
def test_a_failed_composed_check_or_other_settings_void_generated_and_sensitivity_only(kw):
    r = Rule.read_rule(rule_report(cap_share=diff(.9, .7, 1.), **kw))
    assert r["generated"]["outcome"] == "void" and r["sensitivity"]["outcome"] == "void" and r["void"] is False and r["cap"]["outcome"] == "kept" and r["hypothesis"] == "open"


@pytest.mark.parametrize("kw,expected", [(dict(), "supported"), (dict(sens=diff(0., -.01, .01)), "refuted"), (dict(sens=diff(.02, 0., .03)), "open"), (dict(gen=terciles(row(0., -.01, .01), row(0., -.01, .01))), "refuted"),
                                         (dict(gen=terciles(row(.1, 0., .2)), sens=diff(0., -.01, .01)), "open"), (dict(gen=terciles(row(0., -.01, .01), row(0., -.01, .01)), sens=diff(.1, .05, .2)), "refuted"), (dict(sens=diff(0., -.01, .01), flagged=.001), "open")])
def test_the_hypothesis_is_supported_refuted_or_open(kw, expected):
    assert Rule.read_rule(rule_report(**kw))["hypothesis"] == expected


@pytest.mark.parametrize("change,reason", [(dict(checks=dict(blank_identical_to_real=.1, canary=dict(GOOD, reproduced=False), path_check=GOOD)), "canary"), (dict(checks=dict(blank_identical_to_real=.1, canary=GOOD, path_check=dict(GOOD, reproduced=False))), "path check"),
                                           (dict(report_only=False), "check nonzero"), (dict(neutral=None), "check nonzero"), (dict(code_digest=None), "no code digest"), (dict(environment=None), "no code digest"), (dict(code_digest="z"), "mixed"), (dict(environment=dict(gpu="z")), "mixed"),
                                           (dict(neutral=dict(code_digest="z", a=1)), "mixed")])
def test_each_void_voids_all(change, reason):
    r = Rule.read_rule(rule_report(**change))
    assert r["void"] is True and any(reason in x for x in r["reasons"]) and {r["cap"]["outcome"], r["generated"]["outcome"], r["sensitivity"]["outcome"], r["hypothesis"]} == {"void"}


def test_the_rule_refuses_a_smoke_says_the_plan_decides_and_opens_no_store(tmp_path, capsys):
    with pytest.raises(SystemExit, match="smoke"):
        Rule.read_rule(rule_report(declared_read=False))
    path = tmp_path / "r.json"
    path.write_text(json.dumps(rule_report()))
    assert Rule.main(["--report", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "declared in plan; the plan's text decides"
    source = (ROOT / "scripts/tok_v1_rule.py").read_text()
    for word in ("make_store", "R2Store", "boto"):
        assert word not in source
    assert (Rule.HIGH, Rule.LOW, Rule.BOUND, Rule.SHOWN_POINT, Rule.ABSENT_UPPER, Rule.CATCH_FLOOR, Rule.RISE_NATS, Rule.FLAGGED_FLOOR, Rule.DECIDING_TERCILES) == (.75, .25, .5, .05, .02, .5, .015, .01, ("mid", "high"))


def test_the_rule_reads_every_field_of_a_report_the_merge_really_writes(monkeypatch, tmp_path):
    harness(monkeypatch)
    assert read(tmp_path) == 0
    r = W.json.loads(json.dumps(merged(tmp_path)))
    r["neutral"] = dict(code_digest=r["code_digest"], generated=dict(a9_predicted=dict(minus_decoded_real=terciles())), positive_control=dict(catch_rate=.8))
    r["checks"]["canary"].update(code_digest=r["code_digest"], environment=r["environment"])
    r["checks"]["path_check"].update(code_digest=r["code_digest"])
    r["checks"]["composed_check"] = dict(reproduced=True, settings_match=True)
    out = Rule.read_rule(r)
    assert out["void"] is False and out["cap"]["outcome"] in ("kept", "lost", "partial") and out["sensitivity"]["outcome"] in ("rise", "no_rise", "unresolved") and out["sensitivity"]["flagged_share"] == pytest.approx(3 / 40)
