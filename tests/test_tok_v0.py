"""V0 on synthetic data: do real's validity marks carry content? Planted readers that see only marks or only values must land where each form says they should. No test starts a real
job: the runner tests put fake executables first on PATH, with no credentials in the environment."""
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
import tok_v0 as V  # noqa: E402
from tests.test_generation_signing_space import signer_state  # noqa: E402
from tests.test_tok_probes import clip_records, forbidden_keys, gappy, plant_canary, tiny_tokenizer  # noqa: E402

N = 16
JOB = ROOT / "runpod/fleet/tok_v0.sh"


def gapped(seed, start=None, frames=20):
    """A clip with its own tracking gap, so every clip's marks differ."""
    start = 2 + (seed * 3) % 10 if start is None else start
    return gappy(signer_state(seed, frames=frames, amplitude=.4), start, start + 3 + seed % 3)


def marks_of(state):
    return np.concatenate([state.channels[n].valid for n in V.MASKED], axis=1)


class MarksReader:
    """Says the sentence when a clip's marks are its real marks, whatever the values: an instrument that reads tracking."""
    info = dict(run="marks", checkpoint_key="k", sha256="f" * 64, kind="fake")

    def __init__(self, refs) -> None:
        self.refs = refs

    def right(self, state, ref) -> bool:
        return np.array_equal(marks_of(state), marks_of(ref))

    def read(self, states, texts, candidates, group=2):
        rows = [self.right(s, r) for s, r in zip(states, self.refs)]
        return dict(hypotheses=[t if ok else "z y x w" for t, ok in zip(texts, rows)], nll=np.array([[1. if ok else 3.] * candidates.shape[1] for ok in rows]))


class ValuesReader(MarksReader):
    """Says the sentence when a clip's values are right wherever both mark them: an instrument that reads motion."""
    info = dict(MarksReader.info, run="values")

    def right(self, state, ref) -> bool:
        c, r = state.channels["left_hand"], ref.channels["left_hand"]
        both = c.valid & r.valid
        return bool(both.any()) and bool(np.allclose(c.values[both], r.values[both], atol=1e-4))


def harness(monkeypatch, reader=MarksReader, *, declared=True, n=N):
    records = clip_records(n, declared)
    refs = [gapped(i) for i in range(n)]
    tok = tiny_tokenizer()
    monkeypatch.setattr(L, "open_read_list", lambda store, smoke: (records, declared and not smoke, dict(list="t", clips=n)))
    monkeypatch.setattr(L, "load_states", lambda store, recs: refs)
    monkeypatch.setattr(L, "load_tokenizer", lambda store: tok)
    monkeypatch.setattr(L, "make_reader", lambda store, **k: reader(refs))
    return records, refs, tok


def read(tmp_path, forms=V.FORMS, *extra):
    return V.main(["read", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w"), "--forms", ",".join(forms), *extra])


def plant_checks(forms_dir):
    plant_canary(forms_dir)
    path = forms_dir / "path_check.json"
    if (forms_dir / "canary.json").exists() and not path.exists():
        path.write_text(json.dumps(dict(json.loads((forms_dir / "canary.json").read_text()), reproduced=True)))


def merged(tmp_path, forms=V.FORMS):
    plant_checks(tmp_path / "w")
    out = tmp_path / f"report-{len(list(tmp_path.iterdir()))}.json"
    assert V.main(["merge", "--store", "dir:x", "--out-dir", str(tmp_path / "w"), "--forms", ",".join(forms), "--output-file", str(out)]) == 0
    return json.loads(out.read_text())


# ---- marks and values -----------------------------------------------------------------------------------------------------

def test_nearest_index_finds_the_nearest_tracked_frame_the_earlier_on_a_tie():
    valid = np.array([False, False, True, False, False, False, True, False])
    assert V.nearest_index(valid).tolist() == [2, 2, 2, 2, 2, 6, 6, 6]
    assert V.nearest_index(np.zeros(3, bool)).tolist() == [0, 1, 2]
    assert V.nearest_index(np.ones(3, bool)).tolist() == [0, 1, 2]


def test_filled_keeps_tracked_values_and_holds_the_nearest_on_the_rest_and_neutral_for_a_joint_never_tracked():
    state = gapped(1, start=6)
    c = state.channels["left_hand"]
    neutral = V.neutral_pose(state, tiny_tokenizer())
    never = c.valid.copy()
    never[:, 0] = False
    state = replace(state, channels={**state.channels, "left_hand": replace(c, valid=never, values=np.where(never[..., None], c.values, 0.), observed=c.observed & never, confidence=c.confidence * never)})
    out = V.filled(state, neutral=neutral).channels["left_hand"]
    assert out.valid.all()
    np.testing.assert_array_equal(out.values[never], state.channels["left_hand"].values[never])
    np.testing.assert_array_equal(out.values[6, 1], state.channels["left_hand"].values[5, 1])
    np.testing.assert_array_equal(out.values[:, 0], neutral.channels["left_hand"].values[:, 0])


def test_real_values_under_real_marks_are_real_exactly_as_v1_receives_them():
    state = gapped(2)
    out = V.with_marks(V.filled(state, neutral=V.neutral_pose(state, tiny_tokenizer())), state)
    for name in state.channels:
        np.testing.assert_array_equal(out.channels[name].valid, state.channels[name].valid)
        np.testing.assert_array_equal(out.channels[name].values, state.channels[name].values)
    assert V.reader_input_differences(state, out) == dict(marks_added=0., marks_removed=0., arm_difference_on_both=0., hand_local_difference_on_both=0., origin_shift_in_shoulder_widths=0., unit_ratio=1.)


def test_with_marks_stores_zero_on_untracked_frames_and_refuses_another_length():
    state = gapped(3)
    out = V.with_marks(signer_state(9, frames=20, amplitude=.4), state)
    seen = state.channels["left_hand"].valid
    assert (out.channels["left_hand"].values[~seen] == 0).all() and (out.channels["left_hand"].values[seen] != 0).any()
    with pytest.raises(SystemExit, match="differ"):
        V.with_marks(signer_state(9, frames=30, amplitude=.4), state)


def test_stretched_marks_keep_the_partners_pattern_at_this_length():
    partner = gapped(4, start=10, frames=40)
    out = V.stretched_marks(partner, 20)
    assert len(out.timestamps) == 20
    assert (~out.channels["left_hand"].valid.all(1)).any()
    assert set(map(tuple, out.channels["left_hand"].valid.tolist())) <= set(map(tuple, partner.channels["left_hand"].valid.tolist()))
    index = np.rint(np.linspace(0, 39, 20)).astype(int)                            # by nearest frame across the partner's whole length
    np.testing.assert_array_equal(out.channels["left_hand"].valid, partner.channels["left_hand"].valid[index])


def test_shuffled_runs_keep_each_joints_rate_and_run_lengths_and_move_the_pattern():
    state = gappy(gapped(5, start=2, frames=48), 20, 28)
    a, b = marks_of(state), marks_of(V.shuffled_runs(state, np.random.default_rng(1)))
    np.testing.assert_array_equal(a.sum(0), b.sum(0))
    from collections import Counter
    assert Counter(map(bytes, a)) == Counter(map(bytes, b))                       # the same frames' patterns, exactly as often
    gaps = np.flatnonzero(~b.all(1))
    assert np.diff(gaps).max() > 1 and (np.diff(gaps) == 1).sum() >= 5            # runs move whole: no flicker frame by frame
    moved = [not np.array_equal(a, marks_of(V.shuffled_runs(state, np.random.default_rng(s)))) for s in range(8)]
    assert any(moved)


def test_runs_are_cut_on_every_read_channel():
    state = signer_state(5, frames=48, amplitude=.4)
    c = state.channels["right_hand"]
    valid = c.valid.copy()
    valid[20:28] = False                                                           # only the right hand: no other channel changes
    state = replace(state, channels={**state.channels, "right_hand": replace(c, valid=valid, values=np.where(valid[..., None], c.values, 0.), observed=c.observed & valid, confidence=c.confidence * valid)})
    a = marks_of(state)
    assert any(not np.array_equal(a, marks_of(V.shuffled_runs(state, np.random.default_rng(s)))) for s in range(8))


def test_seeds_depend_on_the_clip_and_the_form_only():
    a = V.v0_rng(3, "v0_random_marks").integers(2 ** 32)
    assert a == V.v0_rng(3, "v0_random_marks").integers(2 ** 32) and a != V.v0_rng(4, "v0_random_marks").integers(2 ** 32) and a != V.v0_rng(3, "v0_other_marks").integers(2 ** 32)


def pair_rows(rows):
    """(sign, signer, frames) rows into records and lengths."""
    return [dict(sign=a, signer_component=b, text="x") for a, b, _ in rows], [c for _, _, c in rows]


def test_partners_are_adjacent_by_length_within_a_signer():
    rows = [("a_1-1-rgb_front", "c1", 100), ("b_1-1-rgb_front", "c1", 300), ("c_1-1-rgb_front", "c1", 110), ("d_1-1-rgb_front", "c1", 290),
            ("e_1-1-rgb_front", "c3", 200), ("f_1-1-rgb_front", "c3", 205)]
    partner, info = V.pair_partners(*pair_rows(rows))
    assert partner == [2, 3, 0, 1, 5, 4]
    assert info["same_video"] == 0 and info["cross_signer"] == 0 and info["cycles"] == 0 and info["video_ids_exposed"] == 6 and info["clips"] == 6
    assert info["length_ratio"]["outside_25_percent"] == 0 and .9 < info["length_ratio"]["p10"] <= 1 <= info["length_ratio"]["p90"] < 1.12


def test_no_pair_comes_from_one_video():
    rows = [("a_1-1-rgb_front", "c1", 100), ("a_2-1-rgb_front", "c1", 101), ("b_1-1-rgb_front", "c1", 102), ("b_2-1-rgb_front", "c1", 103)]
    partner, info = V.pair_partners(*pair_rows(rows))
    assert all(rows[i][0][0] != rows[partner[i]][0][0] for i in range(4)) and sorted(partner) == [0, 1, 2, 3] and info["same_video"] == 0


def test_video_ids_with_underscores_and_ids_that_expose_none():
    assert V.video_of("1p2RAtk_CxY_3-11-rgb_front") == "1p2RAtk_CxY" and V.video_of("s0") is None
    partner, info = V.pair_partners(*pair_rows([("s0", "c", 1), ("s1", "c", 2)]))
    assert partner == [1, 0] and info["video_ids_exposed"] == 0


def test_an_odd_clip_joins_its_nearest_pair_as_a_cycle_and_never_itself():
    rows = [("a_1-1-rgb_front", "c1", 100), ("b_1-1-rgb_front", "c1", 110), ("c_1-1-rgb_front", "c1", 500)]
    partner, info = V.pair_partners(*pair_rows(rows))
    assert sorted(partner) == [0, 1, 2] and all(i != p for i, p in enumerate(partner)) and info["cycles"] == 1
    assert V.pair_partners(*pair_rows(rows))[0] == partner


def test_a_signer_alone_goes_across_and_is_counted():
    rows = [("a_1-1-rgb_front", "c1", 100), ("b_1-1-rgb_front", "c1", 110), ("c_1-1-rgb_front", "c2", 105)]
    partner, info = V.pair_partners(*pair_rows(rows))
    assert info["cross_signer"] == 1 and partner[2] in (0, 1) and all(i != p for i, p in enumerate(partner))
    with pytest.raises(ValueError):
        V.pair_partners(*pair_rows(rows[:1]))


def test_a_lone_video_is_counted_when_nothing_else_is_possible():
    rows = [("a_1-1-rgb_front", "c1", 100), ("a_2-1-rgb_front", "c1", 101), ("b_1-1-rgb_front", "c1", 102)]
    partner, info = V.pair_partners(*pair_rows(rows))
    assert all(i != p for i, p in enumerate(partner)) and info["cycles"] == 1 and info["same_video"] >= 1


def never_tracked(seed=3, hand="left_hand", frames=20):
    """A clip with one hand never tracked, as real clips have, carrying no inference method."""
    state = signer_state(seed, frames=frames, amplitude=.4)
    c = state.channels[hand]
    return replace(state, channels={**state.channels, hand: replace(c, valid=np.zeros_like(c.valid), observed=np.zeros_like(c.valid), confidence=np.zeros_like(c.confidence), values=np.zeros_like(c.values), inference_method=None)})


def test_a_clip_with_a_never_tracked_hand_builds_in_every_form_as_itself_and_as_a_partner():
    tok = tiny_tokenizer()
    bad, ok = never_tracked(), gapped(1)
    for name in V.FORMS:
        assert all(len(x.timestamps) == 20 for x in V.build_form(name, [bad, ok], tok, [1, 0]))
    out = V.filled(bad, neutral=V.neutral_pose(bad, tok)).channels["left_hand"]
    assert out.valid.all() and out.inference_method == "tok_v0 nearest tracked value"
    kept = V.filled(ok, neutral=V.neutral_pose(ok, tok)).channels["left_hand"]
    assert kept.inference_method == ok.channels["left_hand"].inference_method


def test_a_smoke_finds_never_tracked_clips_in_c02_train_or_says_it_did_not(monkeypatch):
    pool = [dict(corpus=L.H2S, partition="train", signer_component=c, text="a", sign=f"{c}-{i}", continuous=dict(state_key="k", seconds=1.)) for i, c in enumerate(["c02"] * 60 + ["c00", "c01"] * 5)]
    monkeypatch.setattr(L, "load_manifest_at", lambda store, k, s: dict(records=pool))
    monkeypatch.setattr(L, "load_states", lambda store, recs: [never_tracked() if r["sign"] in ("c02-5", "c02-40") else gapped(1) for r in recs])
    found = V.never_tracked_records(None, 2)
    assert sorted(r["sign"] for r in found) == ["c02-40", "c02-5"]
    with pytest.raises(SystemExit, match="never-tracked"):
        V.never_tracked_records(None, 3)


# ---- the forms ------------------------------------------------------------------------------------------------------------

def form(name, clip=0):
    tok = tiny_tokenizer()
    state, partner = gapped(clip), gapped(clip + 7)
    return V.v0_form(name, state, partner, clip, tok), state, partner, tok


def test_blank_with_real_marks_is_one_pose_under_real_marks():
    out, state, _, tok = form("v0_blank_real_marks")
    np.testing.assert_array_equal(marks_of(out), marks_of(state))
    c = out.channels["left_hand"]
    frames = c.values[c.valid.all(1)]
    assert len(frames) > 1 and np.allclose(frames, frames[0], atol=1e-6)
    neutral = V.neutral_pose(state, tok).channels["left_hand"].values[0]
    np.testing.assert_allclose(frames[0], neutral, atol=1e-6)


def test_its_control_is_the_same_pose_under_the_partners_marks():
    a, state, partner, _ = form("v0_blank_real_marks")
    b, *_ = form("v0_blank_other_marks")
    np.testing.assert_array_equal(marks_of(b), marks_of(V.stretched_marks(partner, 20)))
    assert not np.array_equal(marks_of(a), marks_of(b))
    seen = a.channels["left_hand"].valid & b.channels["left_hand"].valid
    np.testing.assert_allclose(a.channels["left_hand"].values[seen], b.channels["left_hand"].values[seen], atol=1e-6)


def test_other_motion_is_the_partners_values_time_scaled_under_this_clips_marks():
    from signtranslator.generation.compositional import time_scale
    out, state, partner, tok = form("v0_other_motion")
    np.testing.assert_array_equal(marks_of(out), marks_of(state))
    expected = time_scale(V.filled(partner, neutral=V.neutral_pose(partner, tok)), 20)
    seen = state.channels["right_hand"].valid
    np.testing.assert_allclose(out.channels["right_hand"].values[seen], expected.channels["right_hand"].values[seen], atol=1e-9)
    assert not np.allclose(out.channels["right_hand"].values[seen], state.channels["right_hand"].values[seen], atol=1e-3)


def test_other_marks_are_this_clips_values_under_the_partners_marks():
    out, state, partner, tok = form("v0_other_marks")
    np.testing.assert_array_equal(marks_of(out), marks_of(V.stretched_marks(partner, 20)))
    both = state.channels["left_hand"].valid & out.channels["left_hand"].valid
    np.testing.assert_array_equal(out.channels["left_hand"].values[both], state.channels["left_hand"].values[both])
    only = out.channels["left_hand"].valid & ~state.channels["left_hand"].valid
    assert only.any() and (out.channels["left_hand"].values[only] != 0).any()         # a frame this clip did not track holds its nearest tracked value


def test_random_marks_are_this_clips_values_at_its_own_tracked_rate():
    state = gappy(gapped(0, start=2, frames=48), 20, 28)
    out = V.v0_form("v0_random_marks", state, gapped(7, frames=48), 0, tiny_tokenizer())
    np.testing.assert_array_equal(marks_of(out).sum(0), marks_of(state).sum(0))
    both = state.channels["left_hand"].valid & out.channels["left_hand"].valid
    np.testing.assert_array_equal(out.channels["left_hand"].values[both], state.channels["left_hand"].values[both])


def test_nearest_fill_differs_from_identity_only_in_the_fill():
    out, state, _, tok = form("v0_nearest_fill")
    path = P.decoded_form("identity", state, tok, None)
    for name in V.MASKED:
        np.testing.assert_array_equal(out.channels[name].valid, path.channels[name].valid)
        seen = state.channels[name].valid
        np.testing.assert_array_equal(out.channels[name].values[seen], path.channels[name].values[seen])
        assert not np.allclose(out.channels[name].values[~seen], path.channels[name].values[~seen], atol=1e-3)
    assert out.channels["left_hand"].valid.all()
    np.testing.assert_array_equal(out.channels["left_hand"].values[6:9, 0], path.channels["left_hand"].values[np.array([V.nearest_index(state.channels["left_hand"].valid[:, 0])[f] for f in (6, 7, 8)]), 0])
    for name in set(state.channels) - set(V.MASKED):
        np.testing.assert_array_equal(out.channels[name].values, path.channels[name].values)


def test_every_form_keeps_the_clips_length_and_bad_names_refuse():
    refs = [gapped(i) for i in range(4)]
    tok = tiny_tokenizer()
    for name in V.FORMS:
        assert all(len(s.timestamps) == 20 for s in V.build_form(name, refs, tok, [1, 0, 3, 2]))
    with pytest.raises(SystemExit, match="unknown"):
        V.build_form("v0_x", refs, tok, [1, 0, 3, 2])
    with pytest.raises(SystemExit, match="unknown"):
        V.v0_form("v0_x", refs[0], refs[1], 0, tok)


# ---- what v1 receives --------------------------------------------------------------------------------------------------------

def test_reader_input_says_marks_added_when_identity_marks_an_untracked_stretch():
    state = gapped(1, start=6)
    d = V.reader_input_differences(state, P.decoded_form("identity", state, tiny_tokenizer(), None))
    assert d["marks_added"] > 0 and d["marks_removed"] == 0
    assert d["hand_local_difference_on_both"] > 0 and d["arm_difference_on_both"] >= 0         # normalise's medians take the newly marked frames too
    assert d["origin_shift_in_shoulder_widths"] is not None and d["unit_ratio"] > 0


def test_reader_input_counts_marks_removed_and_coordinates_apart():
    state = signer_state(1, frames=20, amplitude=.4)
    other = gappy(state, 3, 9)
    d = V.reader_input_differences(state, other)
    assert d["marks_removed"] > 0 and d["marks_added"] == 0
    moved = replace(state, channels={**state.channels, "left_hand": replace(state.channels["left_hand"], values=signer_state(5, frames=20, amplitude=.4).channels["left_hand"].values)})
    d = V.reader_input_differences(state, moved)
    assert d["hand_local_difference_on_both"] > 0 and d["arm_difference_on_both"] == 0 and d["unit_ratio"] == 1


def untracked_wrist_rotation(state, start=6, stop=9):
    body = state.channels["body"]
    k = list(body.labels).index("LeftHand")
    valid = body.valid.copy()
    valid[start:stop, k] = False
    return replace(state, channels={**state.channels, "body": replace(body, valid=valid, values=np.where(valid[..., None], body.values, 0.), observed=body.observed & valid, confidence=body.confidence * valid)})


def test_fold_share_counts_a_forms_own_marked_hands_on_the_wrist():
    state = signer_state(1, frames=20, amplitude=.4)
    assert V.fold_share(state) == 0
    folded = untracked_wrist_rotation(state)                                       # three frames of one hand, of 40 hand-frames
    assert V.fold_share(folded) == pytest.approx(3 / 40)
    assert V.fold_share(P.decoded_form("identity", folded, tiny_tokenizer(), None)) == 0
    hidden = gappy(folded, 6, 9)
    assert V.fold_share(hidden) == 0                                                # an unmarked hand is not a fold
    from evaluate_text_to_motion import blanked
    assert V.fold_share(blanked([state])[0]) == 0


def test_marks_as_real_is_about_marks_only():
    state = gapped(1)
    assert V.marks_as_real(state, state) and V.marks_as_real(state, V.with_marks(signer_state(4, frames=20, amplitude=.4), state))
    assert not V.marks_as_real(state, V.with_marks(V.filled(state), gapped(9)))


def test_reader_input_is_undefined_only_without_shoulders():
    from evaluate_text_to_motion import blanked
    state = signer_state(1, frames=20, amplitude=.4)
    d = V.reader_input_differences(state, blanked([state])[0])
    assert d["origin_shift_in_shoulder_widths"] is None and d["unit_ratio"] is None and d["marks_removed"] == 1 and d["arm_difference_on_both"] is None


# ---- planted instruments, through read and merge ---------------------------------------------------------------------------------

def test_an_instrument_that_reads_marks_lands_where_the_forms_say(monkeypatch, tmp_path):
    harness(monkeypatch, MarksReader)
    assert read(tmp_path) == 0
    r = merged(tmp_path)["forms"]
    top = r["real"]["bleu"]["bleu"]
    assert top > 0
    assert r["v0_blank_real_marks"]["bleu"]["bleu"] == top and r["v0_other_motion"]["bleu"]["bleu"] == top             # real's marks, whatever the values
    for name in ("v0_blank_other_marks", "v0_other_marks", "v0_random_marks", "v0_nearest_fill", "identity"):
        assert r[name]["bleu"]["bleu"] < top
    assert r["v0_blank_real_marks"]["comparators"]["v0_blank_other_marks"]["bleu"]["difference"] > 0
    assert r["v0_blank_real_marks"]["comparators"]["v0_blank_other_marks"]["likelihood_minus_blank"]["difference"] > 0
    assert r["v0_blank_real_marks"]["paired"]["blank"]["likelihood_minus_blank"]["difference"] > 0
    assert r["v0_other_motion"]["comparators"]["v0_other_motion_other_marks"]["likelihood_minus_blank"]["difference"] > 0
    assert r["v0_other_motion_other_marks"]["bleu"]["bleu"] < top
    assert r["v0_other_marks"]["likelihood_ratio_of_means"]["ratio"] == 0 and r["v0_blank_real_marks"]["likelihood_ratio_of_means"]["ratio"] == 1   # a form over real


def test_an_instrument_that_reads_values_lands_where_the_forms_say(monkeypatch, tmp_path):
    harness(monkeypatch, ValuesReader)
    assert read(tmp_path) == 0
    r = merged(tmp_path)["forms"]
    top = r["real"]["bleu"]["bleu"]
    assert top > 0
    assert r["v0_other_marks"]["bleu"]["bleu"] == top and r["v0_random_marks"]["bleu"]["bleu"] == top and r["v0_nearest_fill"]["bleu"]["bleu"] == top
    for name in ("v0_blank_real_marks", "v0_blank_other_marks", "v0_other_motion"):
        assert r[name]["bleu"]["bleu"] < top
    assert r["v0_blank_real_marks"]["comparators"]["v0_blank_other_marks"]["likelihood_minus_blank"]["difference"] == 0
    assert r["v0_other_motion"]["comparators"]["v0_other_motion_other_marks"]["likelihood_minus_blank"]["difference"] == 0
    assert r["v0_other_marks"]["comparators"]["identity"]["bleu"]["difference"] >= 0


def test_every_report_has_its_read_tracked_share_and_reader_input_and_decides_nothing(monkeypatch, tmp_path):
    harness(monkeypatch, MarksReader)
    assert read(tmp_path) == 0
    r = merged(tmp_path)
    assert r["report_only"] is True and "report only" in r["note"] and forbidden_keys(r) == [] and r["code_digest"] == V.code_digest()
    f = r["forms"]
    assert f["blank"]["tracked_share"]["hands"]["mean"] == 0 and 0 < f["real"]["tracked_share"]["hands"]["mean"] < 1
    assert f["identity"]["tracked_share"]["hands"]["mean"] == 1 and f["v0_nearest_fill"]["tracked_share"]["hands"]["mean"] == 1
    assert f["v0_blank_real_marks"]["tracked_share"]["hands"]["minus_real"]["difference"] == 0 and f["v0_other_motion"]["tracked_share"]["hands"]["minus_real"]["difference"] == 0
    assert f["identity"]["reader_input"]["marks_added"] > 0 and f["real"]["reader_input"]["marks_added"] == 0
    assert set(f["identity"]["comparators"]) == {"real"} and set(f["v0_blank_real_marks"]["comparators"]) == {"v0_blank_other_marks"} and set(f["v0_nearest_fill"]["comparators"]) == {"identity"}
    assert set(f["v0_other_marks"]["comparators"]) == {"v0_random_marks", "identity"} and set(f["v0_other_motion"]["comparators"]) == {"v0_other_motion_other_marks", "v0_blank_real_marks"}
    assert r["pairing"]["clips"] == N and all(i != p for i, p in enumerate(r["partner"])) and f["real"]["clips_with_real_marks"] == 1 and f["identity"]["clips_with_real_marks"] == 0
    assert f["v0_blank_real_marks"]["clips_with_real_marks"] == 1 and 0 <= f["v0_random_marks"]["clips_with_real_marks"] < 1 and f["real"]["fold_share"]["mean"] == 0
    assert r["fold_flag"] == dict(limit=.005, real=0., flagged=False, forms_over=[]) and f["real"]["likelihood_ratio_of_means"]["ratio"] == 1
    for key in ("normalise", "fill", "removing_marks", "folds"):
        assert key in r["disclosures"]


def test_the_fold_flag_says_when_real_folds_over_the_limit(monkeypatch, tmp_path):
    records, refs, tok = harness(monkeypatch, ValuesReader)
    refs[:] = [untracked_wrist_rotation(signer_state(i, frames=20, amplitude=.4)) for i in range(N)]
    assert read(tmp_path, ["real", "blank"]) == 0
    r = merged(tmp_path, ["real", "blank"])
    assert r["fold_flag"]["flagged"] is True and r["fold_flag"]["forms_over"] == ["real"] and r["fold_flag"]["real"] == pytest.approx(3 / 40)


def test_the_reads_block_gives_a_b_p2_p3_as_the_declaration_reads_them():
    n = 100
    refs = [f"w{i} a b c d e f" for i in range(n)]

    def form(name, right, nll):
        return dict(form=name, references=refs, hypotheses=[r if i < right else "z" for i, r in enumerate(refs)], nll=[[nll]] * n)
    reads = dict(blank=form("blank", 0, 3.), real=form("real", 80, 1.), identity=form("identity", 20, 2.), v0_blank_real_marks=form("a", 30, 2.8), v0_blank_other_marks=form("b", 20, 2.9),
                 v0_other_motion=form("c", 25, 2.5), v0_other_motion_other_marks=form("d", 25, 2.7), v0_other_marks=form("e", 50, 1.5), v0_nearest_fill=form("f", 30, 1.75))
    r = V.declared_reads(reads)
    assert r["primary"] == "likelihood_minus_blank" and r["bootstrap"] == dict(n=2000, seed=0)
    k = r["likelihood_minus_blank"]
    assert k["a"]["point"] == pytest.approx(.1) and k["b"]["point"] == pytest.approx(.2) and k["real_minus_identity"]["point"] == pytest.approx(1.)
    assert k["p2"]["point"] == pytest.approx(.5) and k["p3"]["point"] == pytest.approx(.25) and k["a"]["ci95"] == [pytest.approx(.1), pytest.approx(.1)]
    assert k["p2"]["ci95"] == [pytest.approx(.5), pytest.approx(.5)] and k["p3"]["ci95"] == [pytest.approx(.25), pytest.approx(.25)]      # constant per clip: no spread
    assert r["bleu"]["p2"]["ci95"][0] <= r["bleu"]["p2"]["point"] <= r["bleu"]["p2"]["ci95"][1] and r["bleu"]["a"]["point"] > 0 and r["bleu"]["real_minus_identity"]["point"] > 0
    del reads["v0_nearest_fill"]
    assert V.declared_reads(reads) is None


def test_a_form_read_alone_equals_it_read_with_others(monkeypatch, tmp_path):
    harness(monkeypatch, ValuesReader, n=6)
    assert read(tmp_path, ["v0_random_marks", "v0_other_marks"]) == 0
    other = tmp_path / "other"
    other.mkdir()
    assert read(other, ["v0_random_marks"]) == 0
    a, b = (json.loads((p / "w/form-v0_random_marks.json").read_text()) for p in (tmp_path, other))
    assert a["hypotheses"] == b["hypotheses"] and a["reader_input"] == b["reader_input"] and a["tracked"] == b["tracked"]


# ---- hygiene ------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["D20_SHA", "TOKENIZER_SHA", "TRANSLATOR_SHA", "EVAL_MIX_SHA"])
def test_a_mutated_pin_refuses_before_any_read(monkeypatch, tmp_path, name):
    monkeypatch.setattr(L, name, "0" * 64)
    monkeypatch.setattr(L, "open_read_list", lambda *a: pytest.fail("read after a mutated pin"))
    with pytest.raises(SystemExit, match="pins moved"):
        read(tmp_path, ["real"])


def test_no_command_has_an_option_for_another_list_or_instrument(tmp_path):
    for flag in ("--sentences-key", "--translator-run", "--sentence-set", "--tokenizer-key"):
        with pytest.raises(SystemExit):
            V.main(["read", "--out-dir", str(tmp_path), flag, "x"])


@pytest.mark.parametrize("change", [dict(partition="test"), dict(partition="heldout_official"), dict(signer_component="c00"), dict(partition="train"), dict(corpus="youtube-asl-v1")])
def test_a_declared_read_refuses_test_held_out_c00_and_train_records(monkeypatch, tmp_path, change):
    records, _, _ = harness(monkeypatch, n=3)
    records[1].update(change)
    with pytest.raises(SystemExit, match="not a How2Sign val clip"):
        read(tmp_path, ["real"])


@pytest.mark.parametrize("change", [dict(partition="val"), dict(signer_component="c00"), dict(signer_component="c01")])
def test_a_smoke_reads_c02_train_clips_and_says_so(monkeypatch, tmp_path, change):
    harness(monkeypatch, declared=False, n=3)
    assert read(tmp_path, ["real", "blank"], "--smoke-train", "3") == 0
    assert "SMOKE" in merged(tmp_path, ["real", "blank"])["smoke"]
    records, _, _ = harness(monkeypatch, declared=False, n=3)
    records[1].update(change)
    with pytest.raises(SystemExit, match="not a train clip"):
        read(tmp_path, ["identity"], "--smoke-train", "3")


def test_outputs_are_write_once_and_a_resume_reads_only_what_is_missing(monkeypatch, tmp_path):
    harness(monkeypatch, n=3)
    assert read(tmp_path, ["real"]) == 0
    with pytest.raises(SystemExit, match="write-once"):
        read(tmp_path, ["real"])
    stamp = (tmp_path / "w/form-real.json").read_bytes()
    assert read(tmp_path, ["real", "blank"], "--resume") == 0 and (tmp_path / "w/form-real.json").read_bytes() == stamp and (tmp_path / "w/form-blank.json").exists()
    with pytest.raises(SystemExit, match="unknown forms"):
        read(tmp_path, ["k512"])
    for flag in ("--smoke-longest", "--smoke-never-tracked"):
        with pytest.raises(SystemExit, match="belong to a smoke"):
            read(tmp_path, ["identity"], flag, "2")


@pytest.mark.parametrize("bad", ["missing", "failed", "other code", "path missing", "path failed", "path other code"])
def test_a_declared_merge_refuses_without_a_passed_canary_of_its_own_read(monkeypatch, tmp_path, bad):
    harness(monkeypatch, n=4)
    assert read(tmp_path, ["real", "blank"]) == 0
    w = tmp_path / "w"
    plant_checks(w)
    name = "path_check.json" if bad.startswith("path") else "canary.json"
    r = json.loads((w / name).read_text())
    if bad.endswith("missing"):
        (w / name).unlink()
    else:
        (w / name).write_text(json.dumps(dict(r, reproduced=False) if bad.endswith("failed") else dict(r, code_digest="z")))
    with pytest.raises(SystemExit, match="void"):
        V.main(["merge", "--store", "dir:x", "--out-dir", str(w), "--forms", "real,blank", "--output-file", str(tmp_path / "x.json")])


@pytest.mark.parametrize("change,message", [(dict(code_digest=None), "void"), (dict(code_digest="z"), "code digests"), (dict(environment=dict(gpu="z")), "environments"), (dict(partner=[1, 0]), "partners")])
def test_a_merge_refuses_a_mix(monkeypatch, tmp_path, change, message):
    harness(monkeypatch, n=4)
    assert read(tmp_path, ["real", "blank"]) == 0
    plant_checks(tmp_path / "w")
    path = tmp_path / "w/form-blank.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), **change)))
    with pytest.raises(SystemExit, match=message):
        merged(tmp_path, ["real", "blank"])


def test_the_digest_names_this_job_and_is_not_the_probes():
    V.code_digest.cache_clear()
    from signtranslator.eval import provenance
    assert V.code_digest() == provenance.code_digest(extra_files=("runpod/fleet/tok_v0.sh",)) and V.code_digest() != provenance.code_digest()
    assert V.JOB_FILE == "runpod/fleet/tok_v0.sh" and V.code_digest() != P.code_digest()


def test_preflight_refuses_a_finished_read_and_a_void_and_new_code_needs_a_cause(tmp_path, monkeypatch):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "s")
    monkeypatch.setattr(V, "code_digest", lambda: "abc")

    def preflight(*extra):
        return V.main(["preflight", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w"), "--final-key", "v/report.json", "--void-prefix", "v/void", *extra])
    assert preflight() == 0
    store.put("v/void/old.json", b"{}")
    with pytest.raises(SystemExit, match="rerun-cause"):
        preflight()
    assert preflight("--rerun-cause", "why") == 0
    assert json.loads((tmp_path / "w/forms/rerun.json").read_text())["cause"] == "why"
    store.put("v/void/abc.json", b"{}")
    with pytest.raises(SystemExit, match="void"):
        preflight("--rerun-cause", "same code")
    store.put("v/report.json", b"{}")
    with pytest.raises(SystemExit, match="finished"):
        preflight()


@pytest.mark.parametrize("kw,code", [(dict(report_only=True, checks=dict(blank_identical_to_real=.1, canary=None)), 0), (dict(report_only=False, checks=dict(blank_identical_to_real=.1, canary=None)), 3),
                                     (dict(report_only=True, checks=dict(blank_identical_to_real=1., canary=None)), 3), (dict(report_only=True, declared_read=True, checks=dict(blank_identical_to_real=.1, canary=None)), 3),
                                     (dict(report_only=True, declared_read=True, checks=dict(blank_identical_to_real=.1, canary=dict(reproduced=False))), 3),
                                     (dict(report_only=True, declared_read=True, checks=dict(blank_identical_to_real=.1, canary=dict(reproduced=True))), 3),
                                     (dict(report_only=True, declared_read=True, checks=dict(blank_identical_to_real=.1, canary=dict(reproduced=True), path_check=dict(reproduced=False))), 3),
                                     (dict(report_only=True, declared_read=True, checks=dict(blank_identical_to_real=.1, canary=dict(reproduced=True), path_check=dict(reproduced=True))), 0),
                                     (dict(report_only=True, declared_read=True, checks=dict(blank_identical_to_real=.1, canary=None, path_check=dict(reproduced=True))), 3),
                                     (dict(report_only=True, declared_read=True, checks=dict(blank_identical_to_real=.1, canary=dict(reproduced=False), path_check=dict(reproduced=True))), 3)])
def test_check_judges_plumbing(tmp_path, kw, code):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(kw))
    assert V.main(["check", "--report", str(path)]) == code


def test_the_shared_canary_runs_and_voids_under_this_prefix(monkeypatch, tmp_path):
    from tests.test_tok_probes import canary_setup
    store, out = canary_setup(monkeypatch, tmp_path, hyp_change=3)
    monkeypatch.setitem(P.CANARY, "bleu_tolerance", 5.)
    assert V.main(["canary", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(out), "--void-prefix", "tok-probes-v0/void"]) == 3
    assert store.exists("tok-probes-v0/void/d.json")


def path_setup(monkeypatch, tmp_path, *, declared=True, hyp_change=0, nll=0., order=False):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "s")
    refs, signs = [f"w{i} a" for i in range(100)], [f"s{i}" for i in range(100)]
    stored = dict(signs=signs, references=refs, hypotheses=refs, nll=[[1.]] * 100, environment=dict(gpu="stored"))
    raw = json.dumps(stored).encode()
    store.put(V.PROBES_IDENTITY, raw)
    monkeypatch.setattr(V, "PROBES_IDENTITY_SHA", hashlib.sha256(raw).hexdigest())
    out = tmp_path / "f"
    out.mkdir()
    hyps = [r + " x" if i < hyp_change else r for i, r in enumerate(refs)]
    (out / "form-identity.json").write_text(json.dumps(dict(form="identity", declared_read=declared, signs=list(reversed(signs)) if order else signs, references=refs, hypotheses=hyps, nll=[[1. + nll]] * 100, code_digest="d", environment=dict(gpu="g"))))
    return store, out


def path_check(tmp_path, out):
    return V.main(["path-check", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(out), "--void-prefix", "tok-probes-v0/void"])


def test_the_path_check_passes_at_its_limits_and_voids_on_a_miss(monkeypatch, tmp_path):
    store, out = path_setup(monkeypatch, tmp_path, hyp_change=2, nll=.0009)
    assert path_check(tmp_path, out) == 0
    r = json.loads((out / "path_check.json").read_text())
    assert r["reproduced"] and r["identical"] == .98 and r["stored_environment"] == dict(gpu="stored") and not store.list("tok-probes-v0/void")


@pytest.mark.parametrize("kw,miss", [(dict(hyp_change=3), "hypotheses"), (dict(nll=.0012), "NLL"), (dict(nll=-.0012), "NLL")])
def test_a_path_check_miss_exits_3_and_stores_a_void(monkeypatch, tmp_path, kw, miss):
    store, out = path_setup(monkeypatch, tmp_path, **kw)
    assert path_check(tmp_path, out) == 3
    assert json.loads((out / "path_check.json").read_text())["misses"] == [miss] and store.exists("tok-probes-v0/void/d.json")


def test_path_check_nll_differences_of_either_sign_do_not_cancel(monkeypatch, tmp_path):
    _, out = path_setup(monkeypatch, tmp_path)
    r = json.loads((out / "form-identity.json").read_text())
    r["nll"] = [[1. + (.002 if i % 2 else -.002)] for i in range(100)]
    (out / "form-identity.json").write_text(json.dumps(r))
    assert path_check(tmp_path, out) == 3


def test_the_path_check_refuses_other_clips_and_a_changed_stored_read_and_a_smoke_checks_the_pin(monkeypatch, tmp_path, capsys):
    _, out = path_setup(monkeypatch, tmp_path, order=True)
    with pytest.raises(SystemExit, match="other clips"):
        path_check(tmp_path, out)
    other = tmp_path / "second"
    other.mkdir()
    _, out = path_setup(monkeypatch, other, declared=False, hyp_change=50)
    assert path_check(other, out) == 0 and "not applicable" in capsys.readouterr().out and not (out / "path_check.json").exists()
    monkeypatch.setattr(V, "PROBES_IDENTITY_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        path_check(other, out)


def test_the_pinned_stored_read_is_the_probes():
    assert V.PROBES_IDENTITY.endswith("tok-probes/forms/form-identity.json") and V.PROBES_IDENTITY_SHA.startswith("24958feb")


# ---- the job ---------------------------------------------------------------------------------------------------------------------------

def run_job(tmp_path, fail_when=None, code=7, **env):
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    pattern = "*" + "*".join(f'"{x}"' for x in fail_when.split("*")) + "*" if fail_when else ""
    match = f'case "$*" in {pattern}) echo "planted failure" >&2; exit {code};; esac' if fail_when else ""
    (bin_dir / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\n{match}\nexit 0\n')
    (bin_dir / "uv").write_text("#!/bin/bash\nexit 0\n")
    for name in ("python", "uv"):
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    environment = dict(PATH=f"{bin_dir}:/usr/bin:/bin", HOME=str(tmp_path), VENV_BIN=str(bin_dir), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path), WORK=str(tmp_path / "work"), **env)
    done = subprocess.run(["bash", str(JOB)], env=environment, capture_output=True, text=True, timeout=60)
    return done, log.read_text() if log.exists() else ""


def done_line(result) -> int:
    lines = [x for x in result.stdout.splitlines() if x.startswith("TOK-V0-DONE ")]
    assert len(lines) == 1 and lines[0] == result.stdout.strip().splitlines()[-1]
    return int(lines[0].split()[1])


def test_the_runner_sees_no_credentials():
    assert not [k for k in os.environ if k.startswith(("R2_", "AWS_", "RUNPOD_"))]


def test_the_job_smokes_on_train_first_then_reads_with_its_declared_calls(tmp_path):
    result, calls = run_job(tmp_path)
    assert result.returncode == 0 and done_line(result) == 0
    parsed = [V.build_parser().parse_args(x.split()[1:]) for x in calls.splitlines()]
    assert [a.command for a in parsed] == ["read", "canary", "path-check", "merge", "check", "upload", "preflight", "read", "canary", "path-check", "read", "merge", "upload", "check"]
    assert parsed[0].smoke_train == 8 and parsed[0].smoke_longest == 4 and parsed[0].smoke_never_tracked == 2 and parsed[0].forms == ",".join(V.FORMS)
    assert parsed[7].forms == "real,blank,identity" and parsed[7].smoke_train == 0 and parsed[10].forms == ",".join(V.FORMS) and parsed[10].resume and parsed[10].device == "cuda" and parsed[10].nll_group == 4
    assert "tok-probes-v0" in parsed[6].final_key and "tok-probes-v0" in parsed[6].void_prefix and parsed[12].prefix.endswith("tok-probes-v0")


def test_smoke_only_stops_before_any_declared_read(tmp_path):
    result, calls = run_job(tmp_path, SMOKE="1")
    assert result.returncode == 0 and done_line(result) == 0 and "d20" not in calls and "preflight" not in calls


@pytest.mark.parametrize("fail_when,expected", [("tok_v0.py read*smoke-", 2), ("tok_v0.py canary*smoke-", 2), ("tok_v0.py merge*smoke-", 2), ("check --report*smoke-", 2), ("upload --out-dir*smoke-", 2), ("tok_v0.py path-check*smoke-", 2), ("preflight", 1), ("--forms real,blank,identity", 1),
                                                ("tok_v0.py canary*d20", 1), ("tok_v0.py path-check*d20", 1), ("read --resume*d20", 1), ("tok_v0.py merge*d20", 1), ("upload --out-dir*d20", 1), ("check --report*d20", 1)])
def test_any_failure_exits_nonzero_with_its_done_line_and_uploads_its_work(tmp_path, fail_when, expected):
    result, calls = run_job(tmp_path, fail_when=fail_when)
    assert result.returncode == expected and done_line(result) == expected, result.stdout + result.stderr
    assert "upload --partial" in calls.splitlines()[-1]


def test_a_failed_canary_stops_before_any_other_form_with_its_own_status(tmp_path):
    result, calls = run_job(tmp_path, fail_when="tok_v0.py canary*d20", code=3)
    assert result.returncode == 3 and done_line(result) == 3 and "do not retry" in result.stdout
    lines = calls.splitlines()
    at = next(i for i, x in enumerate(lines) if "tok_v0.py canary" in x and "d20" in x)
    assert len(lines[at + 1:]) == 1 and "upload --partial" in lines[at + 1]


def test_a_failed_path_check_stops_before_any_other_form_with_the_voids_status(tmp_path):
    result, calls = run_job(tmp_path, fail_when="tok_v0.py path-check*d20", code=3)
    assert result.returncode == 3 and done_line(result) == 3 and "do not retry" in result.stdout
    lines = calls.splitlines()
    at = next(i for i, x in enumerate(lines) if "tok_v0.py path-check" in x and "d20" in x)
    assert len(lines[at + 1:]) == 1 and "upload --partial" in lines[at + 1]


def test_a_rerun_passes_its_cause_and_a_failed_partial_hides_no_status(tmp_path):
    _, calls = run_job(tmp_path, RERUN_CAUSE="why now")
    assert "--rerun-cause why now" in next(x for x in calls.splitlines() if "preflight" in x)
    other = tmp_path / "second"
    other.mkdir()
    result, _ = run_job(other, fail_when="upload --partial")
    assert result.returncode == 0 and done_line(result) == 0


def test_the_job_pipes_nothing_names_no_other_list_pins_transformers_and_is_valid_bash():
    text = JOB.read_text()
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0
    code = [line.split(" #")[0] for line in text.splitlines() if not line.startswith("#")]
    for line in code:
        assert not re.search(r"(?<!\|)\|(?!\|)", line), line
    for word in ("c20", "C20", "val1000", "eval1000", "--output-key", "tok_probes.py", "tok-probes/"):
        assert word not in "\n".join(code)
    assert "transformers==4.57.6" in text and "set -uo pipefail" in text


def test_the_script_holds_no_decision_rule():
    source = (ROOT / "scripts/tok_v0.py").read_text()
    for word in ("outcome", "marks_carry", "unresolved", "not_detected", "Q1_NATS", ".015", ".75"):
        assert word not in source
    assert "tok_v0_rule" not in JOB.read_text()


# ---- the rule, apart from the probe: one synthetic report per outcome -------------------------------------------------------------

import tok_v0_rule as Rule  # noqa: E402


def diff(point=.1, low=.05, high=.2):
    return dict(point=point, ci95=[low, high])


def rule_report(a=None, b=None, gap=None, p2=None, p3=None, **top):
    reads = dict(a=a or diff(), b=b or diff(), real_minus_identity=gap or diff(1., .5, 1.5), p2=p2 or dict(point=.5, ci95=[.3, .7]), p3=p3 or dict(point=.5, ci95=[.3, .7]))
    place = dict(gpu="g")
    out = dict(report_only=True, declared_read=True, code_digest="d", environment=place, reads=dict(primary="likelihood_minus_blank", likelihood_minus_blank=reads, bleu=dict(a=diff(-9., -9., -9.))),
               checks=dict(blank_identical_to_real=.1, canary=dict(reproduced=True, code_digest="d", environment=place), path_check=dict(reproduced=True, code_digest="d")))
    out.update(top)
    return out


@pytest.mark.parametrize("a,b,expected", [(diff(.015, .001, .03), diff(0., -.01, .01), "shown"), (diff(0., -.01, .01), diff(.02, .01, .03), "shown"), (diff(.014, .001, .03), diff(0., -.01, .01), "unresolved"), (diff(.02, 0., .03), diff(0., -.01, .01), "unresolved"),
                                          (diff(0., -.01, .0149), diff(0., -.01, .0149), "not_detected"), (diff(0., -.01, .015), diff(0., -.01, .01), "unresolved"), (diff(0., -.01, .01), diff(0., -.01, .015), "unresolved"), (diff(-.1, -.2, -.05), diff(-.1, -.2, -.05), "not_detected")])
def test_each_q1_outcome(a, b, expected):
    assert Rule.read_rule(rule_report(a=a, b=b))["q1"]["outcome"] == expected


def p(point, low, high):
    return dict(point=point, ci95=[low, high])


@pytest.mark.parametrize("value,expected", [(p(.75, .5, 1.), "yes"), (p(.9, .6, 1.2), "yes"), (p(.75, .49, 1.), "partial"), (p(.74, .6, 1.), "partial"), (p(.25, 0., .5), "no"), (p(-.1, -.4, .2), "no"), (p(.25, 0., .51), "partial"), (p(.26, 0., .4), "partial"), (dict(point=None, ci95=None), "partial")])
def test_each_q2_outcome(value, expected):
    assert Rule.read_rule(rule_report(p2=value))["q2"]["outcome"] == expected


@pytest.mark.parametrize("value,expected", [(p(.75, .5, 1.), "fill"), (p(.9, .6, 1.2), "fill"), (p(.75, .49, 1.), "both"), (p(.74, .6, 1.), "both"), (p(.25, 0., .5), "mark"), (p(-.1, -.4, .2), "mark"), (p(.25, 0., .51), "both"), (p(.26, 0., .4), "both"), (dict(point=None, ci95=None), "both")])
def test_each_q3_outcome(value, expected):
    assert Rule.read_rule(rule_report(p3=value))["q3"]["outcome"] == expected


@pytest.mark.parametrize("gap", [diff(1., 0., 2.), diff(1., -.5, 2.), dict(point=None, ci95=None)])
def test_q2_and_q3_are_void_when_real_minus_identity_is_not_above_0_and_q1_still_reads(gap):
    r = Rule.read_rule(rule_report(gap=gap, a=diff(.1, .05, .2)))
    assert r["q2"]["outcome"] == "void" and r["q3"]["outcome"] == "void" and r["q1"]["outcome"] == "shown" and r["void"] is False


@pytest.mark.parametrize("change,reason", [(dict(checks=dict(blank_identical_to_real=.1, canary=dict(reproduced=False, code_digest="d", environment=dict(gpu="g")), path_check=dict(reproduced=True, code_digest="d"))), "canary"),
                                           (dict(checks=dict(blank_identical_to_real=.1, canary=dict(reproduced=True, code_digest="d", environment=dict(gpu="g")), path_check=dict(reproduced=False, code_digest="d"))), "path check"),
                                           (dict(report_only=False), "check nonzero"), (dict(checks=dict(blank_identical_to_real=1., canary=dict(reproduced=True, code_digest="d", environment=dict(gpu="g")), path_check=dict(reproduced=True, code_digest="d"))), "check nonzero"),
                                           (dict(code_digest=None), "no code digest"), (dict(environment=None), "no code digest"), (dict(code_digest="z"), "mixed"), (dict(environment=dict(gpu="z")), "mixed"), (dict(reads=None), "reads")])
def test_each_void_voids_all(change, reason):
    r = Rule.read_rule(rule_report(**change))
    assert r["void"] is True and any(reason in x for x in r["reasons"]) and {r["q1"]["outcome"], r["q2"]["outcome"], r["q3"]["outcome"]} == {"void"}


def test_the_rule_reads_likelihood_not_bleu_and_refuses_a_smoke_and_runs_as_a_script(tmp_path, capsys):
    assert Rule.read_rule(rule_report())["q1"]["outcome"] == "shown"                # the BLEU block in the report says otherwise
    with pytest.raises(SystemExit, match="smoke"):
        Rule.read_rule(rule_report(declared_read=False))
    path = tmp_path / "r.json"
    path.write_text(json.dumps(rule_report()))
    assert Rule.main(["--report", str(path)]) == 0
    assert "the plan's text decides" in json.loads(capsys.readouterr().out)["status"]
    source = (ROOT / "scripts/tok_v0_rule.py").read_text()
    for word in ("make_store", "R2Store", "boto"):
        assert word not in source


def test_the_rule_reads_every_field_of_a_report_the_merge_really_writes(monkeypatch, tmp_path):
    harness(monkeypatch, ValuesReader)
    assert read(tmp_path) == 0
    r = merged(tmp_path)
    r["checks"]["canary"].update(code_digest=r["code_digest"], environment=r["environment"])
    r["checks"]["path_check"].update(code_digest=r["code_digest"])
    out = Rule.read_rule(r)
    assert out["void"] is False and out["q1"]["outcome"] in ("shown", "not_detected", "unresolved")
    assert out["q2"]["outcome"] in ("yes", "no", "partial", "void") and out["q3"]["outcome"] in ("fill", "mark", "both", "void")


def test_the_constants_are_the_ones_given():
    assert (Rule.Q1_NATS, Rule.HIGH, Rule.LOW, Rule.BOUND) == (.015, .75, .25, .5)
