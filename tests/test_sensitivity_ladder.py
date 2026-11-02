"""The sensitivity ladder on synthetic data and the real builders' layouts: what each rung changes and what it holds, the controls that stop the job, the reading's arithmetic and labels, what is refused, and the job itself."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import parent_lib as Q  # noqa: E402
import path_parts as PP  # noqa: E402
import path_read as R  # noqa: E402
import s2_lib as L  # noqa: E402
import sensitivity_ladder as S  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v0 as V  # noqa: E402
import tok_v1 as T  # noqa: E402
from signtranslator.data.keypoints2d import from_state, normalise  # noqa: E402
from signtranslator.eval import s0_metrics as M  # noqa: E402
from tests.test_path_read import Harness, N, d20_records, tiny_tokenizer, two_view_state  # noqa: E402
from tests.test_tok_probes import SpyStore  # noqa: E402

JOB = ROOT / "runpod/fleet/sensitivity_ladder.sh"
# per form: (words said, own NLL, margin over the mismatched): the base reads high, each rung lower
LEVELS = dict(real_two_view=(14, 1.1, 1.), blank=(0, 3., 0.), nearest_two_view=(15, 1., 1.), still_two_view=(1, 2.9, .1), fingers_frozen=(10, 1.5, .5), fingers_neutral=(9, 1.7, .3), hands_still=(9, 1.6, .4), finger_noise_015=(13, 1.2, .9), finger_noise_030=(11, 1.4, .7), finger_noise_060=(8, 1.8, .4), hand_dropout_marked=(9, 1.7, .6), hand_dropout_filled=(10, 1.6, .7), wrist_turned=(9, 1.7, .55), wrist_moved=(12, 1.3, .8), arms_frozen=(11, 1.4, .75))


class LevelReader:
    """Reads a form at its planted level, told which form ``build_states`` just made; clip by clip a little different, so the bootstrap has something to resample."""

    def __init__(self, info, levels=None) -> None:
        self.info, self.levels, self.form = info, dict(LEVELS, **(levels or {})), None
        self.calls = []

    def read(self, states, texts, candidates, group=2):
        self.calls.append((self.form, group, candidates.copy()))
        k, nll, margin = self.levels[self.form]
        own = [texts[c] for c in candidates[:, 0]]
        wobble = np.arange(len(states)) * .001
        rows = np.array([[nll + w] + [nll + w + margin] * (candidates.shape[1] - 1) for w in wobble])
        return dict(hypotheses=[" ".join(t.split()[:k]) if k else "z y x w" for t in own], nll=rows)


V1_INFO = dict(run=L.TRANSLATOR_RUN, checkpoint_key="k", sha256=L.TRANSLATOR_SHA, kind="keypoints")
PARENT_INFO = dict(run=PP.UNFITTED_RUN, checkpoint_key=PP.UNFITTED_KEY, sha256=PP.UNFITTED_SHA, kind="keypoints")


class Ladder(Harness):
    def __init__(self, monkeypatch, tmp_path, *, v1=None, parent=None, mono=None, training=None, **kw):
        super().__init__(monkeypatch, tmp_path, **kw)
        self.v1, self.parent = LevelReader(V1_INFO, v1), LevelReader(PARENT_INFO, parent)
        self.mono = LevelReader(dict(run=Q.MONO_RUN, checkpoint_key="m", sha256="a" * 64, kind="keypoints"), mono)
        readers = dict(v1=self.v1, parent=self.parent, mono=self.mono)
        real_build = S.build_states

        def tracked(form, refs, tokenizer):
            for reader in readers.values():
                reader.form = form
            return real_build(form, refs, tokenizer)
        monkeypatch.setattr(S, "build_states", tracked)
        monkeypatch.setattr(S, "load_instrument", lambda name, store, args: readers[name])
        self.plant_training(training if training is not None else [r["sign"] for r in self.records[:4]])
        monkeypatch.setattr(L, "D20_CLIPS", self.n)
        monkeypatch.setattr(S, "code_digest", lambda: "c" * 64)
        monkeypatch.setattr(P, "environment", lambda: dict(gpu="test"))
        monkeypatch.setattr(S, "DROP_FLOOR", 0.)      # nine short clips do not average to the long-run rate
        monkeypatch.setattr(S, "DROP_CEILING", 1.)
        monkeypatch.setattr(S, "V1_TRAINING_IN_D20", 4)
        monkeypatch.setattr(S, "PARENT_FIGURES", dict(nearest_over_still=.9, still_raw=.1))

    def plant_training(self, signs):
        meta = json.dumps([dict(clip_id=s, partition="train", text="t", video_id="v", frames=1) for s in signs]).encode()
        self.store.put("fk/train/shard-00000.json", meta)
        index = json.dumps(dict(parts=[dict(partition="train", shards=[dict(key="fk/train/shard-00000.npz", meta_sha256=hashlib.sha256(meta).hexdigest())]), dict(partition="test", shards=[dict(key="fk/test/shard-00000.npz", meta_sha256="0" * 64)])])).encode()
        self.store.put(Q.FK_INDEX, index)
        self.monkeypatch.setattr(Q, "FK_INDEX_SHA", hashlib.sha256(index).hexdigest())

    def main(self, *argv):
        return S.main([argv[0], "--store", f"dir:{self.tmp_path / 's'}", *argv[1:]])

    def read_forms(self, forms, *, which="base", smoke=False, resume=False):
        args = ["read", "--set", which, "--out-dir", str(self.forms), "--forms", forms]
        if smoke:
            args += ["--smoke-train", str(min(self.n, R.SMOKE_CLIPS))]
        if resume:
            args.append("--resume")
        return self.main(*args)

    def plant_stored_reads(self, *, change=None):
        """The parts job's stored forms and the canary's report, holding what this run read; ``change``: {form name: {field: value}}."""
        change = change or {}
        read = {n: json.loads((self.forms / f"form-{n}.json").read_text()) for n in S.control_forms("base")}
        v1, parent = {}, {}
        for table, instrument, names in ((v1, "v1", (S.BASE,)), (parent, "parent", S.ANCHORS)):
            for name in names:
                form = S.form_name(instrument, name)
                stored = dict(read[form], **change.get(form, {}))
                raw = json.dumps(stored).encode()
                file = f"stored-{form}.json"
                self.store.put(f"{Q.PARTS_FORMS}/{file}", raw)
                table[name] = (file, hashlib.sha256(raw).hexdigest())
        self.monkeypatch.setattr(Q, "STORED_V1", v1)
        self.monkeypatch.setattr(Q, "STORED_PARENT", parent)
        self.plant_canary(read)

    def plant_stored_check(self, change=None):
        """Every stored form at its pin, each with the shape the check asks for; ``change``: {"parent.blank": {field: value}}."""
        records = self.records
        for instrument, names, columns in (("parent", S.ANCHORS, 5), ("v1", tuple(Q.STORED_V1), 1)):
            table = {}
            for name in names:
                form = dict(signs=[r["sign"] for r in records], nll=[[1.] * columns] * self.n, environment=dict(gpu="stored-gpu"))
                form.update((change or {}).get(f"{instrument}.{name}", {}))
                raw = json.dumps(form).encode()
                self.store.put(f"{Q.PARTS_FORMS}/check-{instrument}.{name}.json", raw)
                table[name] = (f"check-{instrument}.{name}.json", hashlib.sha256(raw).hexdigest())
            self.monkeypatch.setattr(Q, "STORED_PARENT" if instrument == "parent" else "STORED_V1", table)
        raw = b"{}"
        self.store.put(P.CANARY_REPORT, raw)
        self.monkeypatch.setattr(P, "CANARY_REPORT_SHA", hashlib.sha256(raw).hexdigest())

    def plant_canary(self, read):
        real, blank = read[S.form_name("v1", "real_two_view")], read[S.form_name("v1", "blank")]
        stored = dict(signs=real["signs"], references=real["references"], forms=dict(real=dict(hypotheses=real["hypotheses"], per_clip=dict(nll=[r[0] for r in real["nll"]])), blank=dict(hypotheses=blank["hypotheses"], per_clip=dict(nll=[r[0] for r in blank["nll"]]))))
        raw = json.dumps(stored).encode()
        self.store.put(P.CANARY_REPORT, raw)
        self.monkeypatch.setattr(P, "CANARY_REPORT_SHA", hashlib.sha256(raw).hexdigest())
        self.monkeypatch.setattr(P, "CANARY", dict(P.CANARY, real=M.bleu_ci_fast(real["hypotheses"], real["references"])[0], blank=M.bleu_ci_fast(blank["hypotheses"], blank["references"])[0]))

    def control(self, which="base"):
        return self.main("control", "--set", which, "--out-dir", str(self.forms), "--void-prefix", "lad/void")

    def to_control(self):
        assert self.read_forms("control") == 0
        self.plant_stored_reads()
        assert self.control() == 0

    def merge(self, name="report.json", which="base"):
        out = self.tmp_path / name
        assert self.main("merge", "--set", which, "--out-dir", str(self.forms), "--output-file", str(out)) == 0
        return json.loads(out.read_text())

    def pipeline(self, name="report.json"):
        self.to_control()
        assert self.read_forms("rest", resume=True) == 0
        return self.merge(name)


@pytest.fixture
def base_state():
    return T.cap_form("v0_nearest_fill", two_view_state(0, "a" * 64), tiny_tokenizer())


def view(state):
    points, valid = from_state(state)
    return normalise(points, valid, y_down=False), valid


# ---- what each rung changes and holds -----------------------------------------------------------------------------------------------

def test_rotation_matrices_are_the_identity_at_zero_a_quarter_turn_at_pi_over_2_and_orthonormal():
    assert np.allclose(S.rotation_matrices(np.zeros((2, 3))), np.eye(3))
    assert np.allclose(S.rotation_matrices(np.array([0., 0., np.pi / 2])) @ np.array([1., 0., 0.]), [0., 1., 0.])
    many = S.rotation_matrices(np.random.default_rng(0).normal(size=(20, 3)))
    assert np.allclose(many @ np.swapaxes(many, -1, -2), np.eye(3), atol=1e-9) and np.allclose(np.linalg.det(many), 1.)


def test_smooth_noise_is_unit_variance_correlated_over_its_box_and_shaped():
    noise = S.smooth_noise(np.random.default_rng(0), 4000, 3)
    assert noise.shape == (4000, 3, 3) and abs(noise.std() - 1.) < .05
    x = noise[:, 0, 0]
    assert np.corrcoef(x[:-1], x[1:])[0, 1] > .7 and abs(np.corrcoef(x[:-S.BOX], x[S.BOX:])[0, 1]) < .15


def test_frozen_fingers_stand_still_and_nothing_else_changes(base_state):
    state, row = S.rung("fingers_frozen", base_state, 0)
    assert "local_motion" not in row
    for hand in T.HANDS:
        values = state.channels[hand].values
        assert np.array_equal(values, np.repeat(base_state.channels[hand].values[[len(values) // 2]], len(values), 0))
    assert np.array_equal(state.channels["body"].values, base_state.channels["body"].values)
    assert row["distance"]["hands"] > .01 and row["distance"]["wrists"] == 0. and row["distance"]["elbows"] == 0.


@pytest.mark.parametrize("form,target", list(S.NOISE.items()))
def test_each_noise_rung_reaches_its_distance_and_leaves_the_arms(base_state, form, target):
    state, row = S.rung(form, base_state, 3)
    assert abs(row["distance"]["hands"] - target) <= .011 * target
    assert row["distance"]["wrists"] == 0. and row["distance"]["shoulders"] == 0. and np.array_equal(state.channels["body"].values, base_state.channels["body"].values)
    assert not np.array_equal(state.channels["left_hand"].values, base_state.channels["left_hand"].values) and not np.array_equal(state.channels["right_hand"].values, base_state.channels["right_hand"].values)


def test_noise_is_one_pattern_per_clip_scaled_by_rung_and_another_for_another_clip(base_state):
    a, b = S.finger_noise_of(base_state, 1), S.finger_noise_of(base_state, 1)
    assert np.array_equal(a["left_hand"], b["left_hand"])
    assert not np.array_equal(a["left_hand"], S.finger_noise_of(base_state, 2)["left_hand"])
    assert not np.array_equal(a["left_hand"], a["right_hand"])
    assert S.rung("finger_noise_060", base_state, 1)[1]["scale"] > S.rung("finger_noise_015", base_state, 1)[1]["scale"]
    assert np.allclose(S.finger_noised(base_state, a, 0.).channels["left_hand"].values, base_state.channels["left_hand"].values, atol=1e-5)


def test_a_clip_that_cannot_reach_its_target_keeps_its_nearest(base_state, monkeypatch):
    monkeypatch.setattr(S, "group_distances", lambda state, base: dict(hands=.05))
    state, scale, distance = S.fit_scale(lambda s: base_state, base_state, "hands", .6)
    assert distance == .05 and state is base_state and scale == .1


def test_the_scale_stops_at_its_ceiling_and_a_part_with_no_point_refuses(base_state, monkeypatch):
    seen = []
    monkeypatch.setattr(S, "group_distances", lambda state, base: dict(hands=.001))
    S.fit_scale(lambda s: seen.append(s) or base_state, base_state, "hands", .6)
    assert max(seen) == S.MAX_SCALE and len(seen) == S.FIT_STEPS
    monkeypatch.setattr(S, "group_distances", lambda state, base: dict(hands=None))
    with pytest.raises(SystemExit, match="no point"):
        S.fit_scale(lambda s: base_state, base_state, "hands", .6)


def test_a_turned_wrist_turns_the_hand_keeps_the_arm_points_and_the_fingers_channel(base_state):
    state, row = S.rung("wrist_turned", base_state, 0)
    assert abs(row["distance"]["hands"] - S.PATH_HANDS) <= .011 * S.PATH_HANDS and row["distance"]["wrists"] < 1e-5
    for hand in T.HANDS:
        assert np.array_equal(state.channels[hand].values, base_state.channels[hand].values)


def test_a_moved_wrist_moves_the_wrist_and_holds_every_hand_local_point(base_state):
    state, row = S.rung("wrist_moved", base_state, 0)
    assert abs(row["distance"]["wrists"] - S.PATH_WRISTS) <= .011 * S.PATH_WRISTS and row["distance"]["hands"] < 1e-3
    assert row["distance"]["shoulders"] == 0.
    (a, _), (b, _) = view(state), view(base_state)
    assert np.abs(a[:, S.PP.HAND_LOCAL_POINTS] - b[:, S.PP.HAND_LOCAL_POINTS]).max() < 1e-3


def test_frozen_arms_stand_still_and_hold_every_hand_local_point(base_state):
    state, row = S.rung("arms_frozen", base_state, 0)
    a, _ = view(state)
    b, _ = view(base_state)
    arms = list(range(6))
    assert np.abs(a[:, arms] - a[[0]][:, arms]).max() < 1e-5 and np.abs(b[:, arms] - b[[0]][:, arms]).max() > .01
    assert row["distance"]["hands"] < 1e-3 and np.abs(a[:, PP.HAND_LOCAL_POINTS] - b[:, PP.HAND_LOCAL_POINTS]).max() < 1e-3


def test_frozen_fingers_leave_hand_local_points_moving_and_hands_still_holds_them(base_state):
    assert S.local_motion(S.held_fingers(base_state)) > .01
    state, row = S.rung("hands_still", base_state, 0)
    assert S.local_motion(state) < 1e-3 and row["local_motion"] < 1e-3 and row["distance"]["wrists"] < 1e-5 and row["distance"]["elbows"] == 0. and row["distance"]["hands"] > .01
    middle = len(base_state.timestamps) // 2
    for hand in T.HANDS:
        assert np.array_equal(state.channels[hand].values, np.repeat(base_state.channels[hand].values[[middle]], len(base_state.timestamps), 0))
    (a, _), (b, _) = view(state), view(base_state)
    assert np.abs(a[:, list(range(6))] - b[:, list(range(6))]).max() < 1e-5


def test_neutral_fingers_hold_one_constant_neutral_hand_and_the_arms_move(base_state):
    tokenizer = tiny_tokenizer()
    state, row = S.rung("fingers_neutral", base_state, 0, tokenizer)
    neutral = V.neutral_pose(base_state, tokenizer)
    for hand in T.HANDS:
        values = state.channels[hand].values
        assert np.array_equal(values[0], values[-1]) and np.allclose(values[0], neutral.channels[hand].values[0]) and values.dtype == base_state.channels[hand].values.dtype
    assert np.array_equal(state.channels["body"].values, base_state.channels["body"].values) and row["distance"]["wrists"] == 0.


def test_a_neutral_hand_of_another_shape_refuses(base_state, monkeypatch):
    other = replace(base_state, channels={**base_state.channels, "left_hand": replace(base_state.channels["left_hand"], values=base_state.channels["left_hand"].values[:, :3], valid=base_state.channels["left_hand"].valid[:, :3], observed=base_state.channels["left_hand"].observed[:, :3], confidence=base_state.channels["left_hand"].confidence[:, :3], labels=base_state.channels["left_hand"].labels[:3])})
    monkeypatch.setattr(V, "neutral_pose", lambda state, tokenizer: other)
    with pytest.raises(SystemExit, match="the neutral pose"):
        S.neutral_fingers(base_state, None)


def test_dropout_runs_at_its_rate_and_mean_run_independently_per_hand():
    frames = S.dropout_frames(np.random.default_rng(0), 200000)
    assert abs(frames.mean() - S.DROP_RATE) < .01
    edges = np.flatnonzero(np.diff(frames.astype(int)))
    runs = np.diff(edges)
    starts_lost = bool(frames[edges[0] + 1])
    assert abs(runs[0 if starts_lost else 1::2].mean() - S.DROP_RUN) < .3
    never = S.dropout_frames(np.random.default_rng(1), 500)
    assert not np.array_equal(never, S.dropout_frames(np.random.default_rng(2), 500))


def test_a_marked_dropout_unmarks_whole_hands_zeroes_them_and_a_filled_one_marks_them_again(base_state):
    clip = next(c for c in range(50) if all(S.dropped_state(base_state, c)[1].values()))
    marked, row = S.rung("hand_dropout_marked", base_state, clip)
    filled, same = S.rung("hand_dropout_filled", base_state, clip)
    assert row["dropped"] == same["dropped"]
    lost_any = False
    for hand in T.HANDS:
        c, kept = marked.channels[hand], base_state.channels[hand]
        lost = ~c.valid.any(1)
        lost_any = lost_any or lost.any()
        assert (~c.valid).sum() == lost.sum() * c.valid.shape[1] and abs(lost.mean() - row["dropped"][hand]) < 1e-9
        assert (c.values[lost] == 0).all() and (~c.observed[lost]).all() and (c.confidence[lost] == 0).all()
        assert np.array_equal(c.values[~lost], kept.values[~lost])
        assert filled.channels[hand].valid.all()
    assert lost_any and np.array_equal(marked.channels["body"].values, base_state.channels["body"].values) and marked.channels["body"].valid.all()


def test_dropout_is_fixed_by_clip_and_differs_between_clips(base_state):
    a, b, c = S.dropped_state(base_state, 0)[1], S.dropped_state(base_state, 0)[1], S.dropped_state(base_state, 1)[1]
    assert a == b and a != c


def test_an_unknown_rung_refuses(base_state):
    with pytest.raises(SystemExit, match="unknown rung"):
        S.rung("nothing", base_state, 0)


def test_build_states_gives_each_anchor_its_own_states_and_masks():
    refs = [two_view_state(i, "a" * 64) for i in range(2)]
    tokenizer = tiny_tokenizer()
    states, masks, rows = S.build_states("real_two_view", refs, tokenizer)
    assert states == refs and masks == refs and rows is None
    states, masks, rows = S.build_states("blank", refs, tokenizer)
    assert not states[0].channels["body"].valid.any() and masks == refs and rows is None
    states, masks, rows = S.build_states(S.BASE, refs, tokenizer)
    assert states[0].channels["body"].valid.all() and masks == refs and rows is None
    still, masks, rows = S.build_states(S.STILL, refs, tokenizer)
    assert np.array_equal(still[0].channels["body"].values[0], still[0].channels["body"].values[-1]) and masks == refs and rows is None
    damaged, masks, rows = S.build_states("wrist_moved", refs, tokenizer)
    assert len(rows) == 2 and masks[0].channels["body"].valid.all() and "scale" in rows[0] and "distance" in rows[1]


# ---- names and refusals ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["v2__blank", "v1__c20", "blank", "parent__test", "mono__real_c00"])
def test_a_form_name_outside_the_ladder_refuses(name):
    with pytest.raises(SystemExit, match="unknown form"):
        S.split_name(name)


def test_the_sets_name_their_instruments_and_their_forms():
    assert S.SETS == dict(base=("v1", "parent"), mono=("mono",))
    assert len(S.all_forms("base")) == 2 * len(S.FORMS) and len(set(S.all_forms("base"))) == 2 * len(S.FORMS)
    assert S.control_forms("mono") == ("v1__real_two_view", "v1__blank")
    assert set(S.control_forms("base")) == {"v1__real_two_view", "v1__blank", "v1__nearest_two_view", "parent__real_two_view", "parent__blank", "parent__nearest_two_view", "parent__still_two_view"}
    assert not set(S.resolve_forms("rest", "base")) & set(S.control_forms("base"))
    assert set(S.resolve_forms("rest", "mono")) == {f"mono__{f}" for f in S.FORMS}


def test_a_forms_list_from_another_set_refuses():
    with pytest.raises(SystemExit, match="not forms of set"):
        S.resolve_forms("mono__blank", "base")
    assert S.resolve_forms("v1__blank", "base") == ["v1__blank"]


@pytest.mark.parametrize("change,match", [(dict(signer_component="c00"), "c00"), (dict(partition="test"), "not a How2Sign val"), (dict(partition="heldout_official"), "not a How2Sign val"), (dict(corpus="other"), "not a How2Sign record")])
def test_a_record_of_c00_a_test_partition_or_another_corpus_refuses_before_any_state(monkeypatch, tmp_path, change, match):
    h = Ladder(monkeypatch, tmp_path)
    h.records[0].update(change)
    loaded = []
    monkeypatch.setattr(L, "load_states", lambda store, recs: loaded.append(1))
    with pytest.raises(SystemExit, match=match):
        h.read_forms("control")
    assert not loaded


def test_c20_cannot_be_named_and_is_refused_unread():
    store = SpyStore({"acquisition/2026-10-17-v1/youtube-asl-v1/round20/c20.json": b"{}"})
    with pytest.raises(SystemExit, match="D20 only"):
        L.load_read_list(store, dict(records=[]), "acquisition/2026-10-17-v1/youtube-asl-v1/round20/c20.json")
    assert store.read == []
    for command in S.build_parser()._subparsers._group_actions[0].choices.values():
        flags = {o for a in command._actions for o in a.option_strings}
        assert not flags & {"--sentences-key", "--list", "--instrument", "--translator-run", "--checkpoint", "--tokenizer"}


def test_a_smoke_is_five_train_clips_at_most(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path, declared=False, n=2)
    with pytest.raises(SystemExit, match="at most 5"):
        h.main("read", "--out-dir", str(h.forms), "--smoke-train", "6")


@pytest.mark.parametrize("partition", ["test", "heldout_official", "monitor", "c20"])
def test_v1s_test_or_held_out_series_is_refused_before_anything_is_fetched(partition):
    store = SpyStore({})
    with pytest.raises(SystemExit, match="refused unread"):
        Q.v1_series(store, partition)
    assert store.read == []


def test_v1s_training_list_opens_only_its_train_series(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path, training=["s0", "s5"])
    assert Q.v1_training_signs(h.store) == {"s0", "s5"}
    (h.store.root / "fk/test").mkdir(parents=True)     # no test shard exists here: reading one would raise


def test_v1s_index_on_other_bytes_refuses_and_a_list_with_another_partition_refuses(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    monkeypatch.setattr(Q, "FK_INDEX_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        Q.v1_series(h.store, "train")
    meta = json.dumps([dict(clip_id="x", partition="test")]).encode()
    h.store.put("fk/train/shard-00000.json", meta)
    index = json.dumps(dict(parts=[dict(partition="train", shards=[dict(key="fk/train/shard-00000.npz", meta_sha256=hashlib.sha256(meta).hexdigest())])])).encode()
    h.store.put(Q.FK_INDEX, index)
    monkeypatch.setattr(Q, "FK_INDEX_SHA", hashlib.sha256(index).hexdigest())
    with pytest.raises(SystemExit, match="holds"):
        Q.v1_series(h.store, "train")
    h.store.put("fk/train/shard-00000.json", b"[]")
    with pytest.raises(SystemExit, match="SHA-256"):
        Q.v1_series(h.store, "train")


def test_a_stored_form_is_named_and_pinned(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="no stored read"):
        Q.stored_form(h.store, Q.STORED_V1, "c20_real")
    h.store.put(f"{Q.PARTS_FORMS}/x.json", b"{}")
    with pytest.raises(SystemExit, match="SHA-256"):
        Q.stored_form(h.store, dict(blank=("x.json", "0" * 64)), "blank")
    assert Q.stored_form(h.store, dict(blank=("x.json", hashlib.sha256(b"{}").hexdigest())), "blank") == {}


def test_the_stored_pins_are_the_parts_jobs_and_v1s_own():
    assert set(Q.STORED_PARENT) == set(S.ANCHORS) and S.BASE in Q.STORED_V1 and all(len(sha) == 64 for _, sha in {**Q.STORED_PARENT, **Q.STORED_V1}.values())
    assert Q.PARTS_FORMS.endswith("round20/path-read-parts/forms") and Q.FK_INDEX.endswith("how2sign-fk-keypoints-v1/index.json")
    assert S.INSTRUMENT_SHA == dict(v1=L.TRANSLATOR_SHA, parent=PP.UNFITTED_SHA) and Q.CANDIDATES == 5 and Q.BOOT == 2000


# ---- the mono-tuned instrument's record ---------------------------------------------------------------------------------------------

class Args:
    device = "cpu"
    instrument_cache = None


def test_no_mono_tuned_instrument_before_it_is_trained(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="no trained record"):
        Q.load_mono(h.store, Args())
    with pytest.raises(SystemExit, match="no trained record"):
        h.main("instruments", "--set", "mono")
    with pytest.raises(SystemExit, match="no trained record"):
        S.expected_sha_of("mono", h.store)


def plant_mono(store, *, key=f"{Q.MONO_PREFIX}/checkpoints/step-00001000.pt", best_key=None, payload=b"weights", sha=None):
    store.put(key, payload)
    store.put(Q.TRAINED_RECORD, json.dumps(dict(checkpoint_key=key, sha256=sha or hashlib.sha256(payload).hexdigest())).encode())
    store.put(f"{Q.MONO_PREFIX}/best.json", json.dumps(dict(checkpoint_key=best_key or key)).encode())


def test_the_mono_tuned_checkpoint_is_refused_on_any_other_bytes(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    plant_mono(h.store, sha="0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        Q.load_mono(h.store, Args())


def test_a_record_against_its_best_or_outside_its_run_refuses(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    plant_mono(h.store, best_key=f"{Q.MONO_PREFIX}/checkpoints/step-00002000.pt")
    with pytest.raises(SystemExit, match="its best.json"):
        Q.load_mono(h.store, Args())
    plant_mono(h.store, key=f"training-runs/{L.TRANSLATOR_RUN}/checkpoints/step-00009000.pt")
    with pytest.raises(SystemExit, match="its best.json"):
        Q.load_mono(h.store, Args())


def test_each_instrument_loads_by_its_own_name(monkeypatch):
    monkeypatch.setattr(P, "load_reader", lambda store, args: "v1")
    monkeypatch.setattr(PP, "load_unfitted", lambda store, args: "parent")
    monkeypatch.setattr(Q, "load_mono", lambda store, args: "mono")
    assert [S.load_instrument(n, None, None) for n in ("v1", "parent", "mono")] == ["v1", "parent", "mono"]


# ---- reads and controls --------------------------------------------------------------------------------------------------------------

def test_every_rung_needs_a_passed_control_before_it_loads_a_state(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    loaded = []
    monkeypatch.setattr(L, "load_states", lambda store, recs: loaded.append(1))
    with pytest.raises(SystemExit, match="no passed control"):
        h.read_forms("rest")
    assert not loaded


@pytest.mark.parametrize("change", [dict(reproduced=False), dict(code_digest="d" * 64), dict(environment=dict(gpu="other"))])
def test_a_failed_control_or_one_of_other_code_or_environment_refuses(monkeypatch, tmp_path, change):
    h = Ladder(monkeypatch, tmp_path)
    h.to_control()
    path = h.forms / "control.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), **change)))
    with pytest.raises(SystemExit, match="no passed control"):
        h.read_forms("rest", resume=True)


def test_a_control_reads_with_the_parts_jobs_candidates_and_group(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    assert h.read_forms("control") == 0
    assert all(c[1] == 4 for c in h.parent.calls + h.v1.calls)
    want = M.retrieval_candidates(N, 5, seed=0)
    for reader in (h.parent, h.v1):
        for _, _, candidates in reader.calls:
            assert np.array_equal(candidates, want)
    assert h.mono.calls == []


def test_each_form_is_read_by_its_instrument_each_rung_once(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", resume=True) == 0
    assert sorted(c[0] for c in h.v1.calls) == sorted(S.FORMS) and sorted(c[0] for c in h.parent.calls) == sorted(S.FORMS)
    form = json.loads((h.forms / "form-parent__fingers_frozen.json").read_text())
    assert form["instrument"]["sha256"] == PP.UNFITTED_SHA and form["instrument_name"] == "parent" and form["form"] == "fingers_frozen" and form["kind"] == "sensitivity-ladder-form" and form["report_only"] is True and form["declared_read"] is True
    assert len(form["rows"]) == N and form["signs"] == [r["sign"] for r in h.records] and form["tokenizer"]["sha256"] == L.TOKENIZER_SHA and np.asarray(form["nll"]).shape == (N, 5)
    assert json.loads((h.forms / "form-v1__blank.json").read_text())["tokenizer"] is None


def test_real_and_blank_load_no_tokenizer(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    monkeypatch.setattr(L, "load_tokenizer", lambda store: (_ for _ in ()).throw(AssertionError("loaded")))
    assert h.read_forms("v1__real_two_view,v1__blank") == 0


def test_a_read_is_write_once_and_a_resume_leaves_what_stands(monkeypatch, tmp_path, capsys):
    h = Ladder(monkeypatch, tmp_path)
    assert h.read_forms("v1__real_two_view,v1__blank") == 0
    with pytest.raises(SystemExit, match="write-once"):
        h.read_forms("v1__blank")
    before = (h.forms / "form-v1__blank.json").read_bytes()
    assert h.read_forms("v1__blank", resume=True) == 0
    assert (h.forms / "form-v1__blank.json").read_bytes() == before and '"read": 0' in capsys.readouterr().out


def test_the_control_passes_and_writes_what_it_compared(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    h.to_control()
    control = json.loads((h.forms / "control.json").read_text())
    assert control["reproduced"] is True and set(control["stored"]) == {"v1__nearest_two_view", "parent__real_two_view", "parent__blank", "parent__nearest_two_view", "parent__still_two_view"} and control["code_digest"] == "c" * 64


@pytest.mark.parametrize("form,field,value,miss", [("parent__nearest_two_view", "hypotheses", ["changed"] * N, "hypotheses"), ("parent__still_two_view", "nll", [[9.] * 5] * N, "NLL"), ("v1__nearest_two_view", "signs", list(reversed([f"s{i}" for i in range(N)])), "other clips"), ("parent__blank", "references", ["x y"] * N, "other clips")])
def test_a_stored_form_missed_voids_the_read(monkeypatch, tmp_path, capsys, form, field, value, miss):
    h = Ladder(monkeypatch, tmp_path)
    assert h.read_forms("control") == 0
    h.plant_stored_reads(change={form: {field: value}})
    assert h.control() == 3
    out = capsys.readouterr().out
    assert miss in out and "ALERT" in out and "no retry" in out
    assert h.store.exists("lad/void/" + "c" * 64 + ".json")
    with pytest.raises(SystemExit, match="no passed control"):
        h.read_forms("rest", resume=True)
    with pytest.raises(SystemExit, match="this code's read is void"):
        h.main("preflight", "--out-dir", str(tmp_path / "w2"), "--final-key", "lad/report.json", "--void-prefix", "lad/void")


def test_a_canary_miss_voids_the_read(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    assert h.read_forms("control") == 0
    h.plant_stored_reads()
    monkeypatch.setattr(P, "CANARY", dict(P.CANARY, real=P.CANARY["real"] + 1.))
    assert h.control() == 3


def test_a_canary_of_another_list_voids_rather_than_crashes(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    assert h.read_forms("control") == 0
    h.plant_stored_reads()
    stored = json.loads(h.store.get(P.CANARY_REPORT))
    stored["signs"] = list(reversed(stored["signs"]))
    raw = json.dumps(stored).encode()
    h.store.put(P.CANARY_REPORT, raw)
    monkeypatch.setattr(P, "CANARY_REPORT_SHA", hashlib.sha256(raw).hexdigest())
    assert h.control() == 3


def test_the_canary_report_on_other_bytes_refuses(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    assert h.read_forms("control") == 0
    h.plant_stored_reads()
    monkeypatch.setattr(P, "CANARY_REPORT_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        h.control()


def test_a_smoke_control_compares_nothing_and_says_so(monkeypatch, tmp_path, capsys):
    h = Ladder(monkeypatch, tmp_path, declared=False, n=5)
    assert h.read_forms("control", smoke=True) == 0
    h.plant_stored_reads()
    assert h.control() == 0 and "a smoke reads train clips" in capsys.readouterr().out and not (h.forms / "control.json").exists()


def test_the_control_of_the_mono_set_is_the_canary_alone(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    assert h.read_forms("control", which="mono") == 0
    read = {n: json.loads((h.forms / f"form-{n}.json").read_text()) for n in S.control_forms("mono")}
    h.plant_canary(read)
    assert h.control("mono") == 0
    assert json.loads((h.forms / "control.json").read_text())["stored"] == {}
    assert h.parent.calls == [] and h.mono.calls == []


def test_same_read_holds_the_canarys_limits():
    base = dict(form="f", signs=["a", "b"], references=["one two three four five", "six seven eight nine ten"], hypotheses=["one two three four five", "six seven eight"], nll=[[1., 2.], [1., 2.]])
    assert Q.same_read(base, base)["misses"] == []
    assert Q.same_read(base, dict(base, nll=[[1.01, 2.], [1., 2.]]))["misses"] == ["NLL"]
    assert Q.same_read(base, dict(base, nll=[[1.], [1.]]))["nll_columns"] == 1
    assert "hypotheses" in Q.same_read(base, dict(base, hypotheses=["x", "six seven eight"]))["misses"]
    assert "BLEU" in Q.same_read(base, dict(base, hypotheses=["x", "y"]))["misses"]


# ---- reading --------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("own,share,drop,case", [
    (dict(ci95=[.1, .2]), dict(ci95=[.75, 1.]), dict(ci95=[-.1, .1]), "holds"),
    (dict(ci95=[.1, .2]), dict(ci95=[.74, 1.]), dict(ci95=[.1, .2]), "unresolved"),
    (dict(ci95=[.1, .2]), dict(ci95=[.2, .75]), dict(ci95=[.01, .2]), "falls"),
    (dict(ci95=[.1, .2]), dict(ci95=[.2, .76]), dict(ci95=[.01, .2]), "unresolved"),
    (dict(ci95=[.1, .2]), dict(ci95=[.2, .7]), dict(ci95=[0., .2]), "unresolved"),
    (dict(ci95=[0., .2]), dict(ci95=[.9, 1.]), dict(ci95=[-.1, .1]), "unresolved"),
    (dict(ci95=[.1, .2]), None, dict(ci95=[.1, .2]), "unresolved")])
def test_the_frozen_case_of_one_instrument(own, share, drop, case):
    assert S.frozen_case(own, share, drop) == case


def linear_values(slope, noise=0.):
    rng = np.random.default_rng(0)
    base = rng.normal(1., noise or .01, 200)
    values = {S.BASE: base}
    for form, d in S.NOISE.items():
        values[form] = base - slope * d + rng.normal(0., noise or .01, 200) * (1 if noise else 0)
    return values


def test_the_detectable_size_is_the_half_width_over_the_slope():
    idx = np.random.default_rng(0).integers(0, 200, (Q.BOOT, 200))
    out = S.detectable(linear_values(1., noise=.5), idx, dict(S.NOISE))
    assert abs(out["slope"] - 1.) < .25
    expected = out["half_width"] / out["slope"]
    assert abs(out["detectable_palm_widths"] - expected) < 1e-3 and out["sees_path_distance"] == (out["detectable_palm_widths"] <= S.PATH_HANDS) and out["path_distance"] == .587 and out["mirror_control"] == 1.286
    assert out["levels"]["finger_noise_060"] == .6


def test_a_steep_instrument_sees_the_path_and_a_flat_one_does_not():
    idx = np.random.default_rng(0).integers(0, 200, (Q.BOOT, 200))
    steep = S.detectable(linear_values(2., noise=.05), idx, dict(S.NOISE))
    assert steep["noise_rungs_agree"] is True and steep["reads_fingers"] is True and "necessary, not sufficient" in steep["sees_path_distance_note"]
    assert steep["sees_path_distance"] is True and steep["first_rung_detected"] == "finger_noise_015" and steep["note"] is None
    flat = S.detectable(linear_values(0., noise=.5) | {f: linear_values(0., noise=.5)[S.BASE] + .2 for f in S.NOISE}, idx, dict(S.NOISE))
    assert flat["detectable_palm_widths"] is None and flat["sees_path_distance"] is None and flat["note"] == "not detectable in this range" and flat["first_rung_detected"] is None
    shallow = S.detectable(linear_values(.05, noise=.5), idx, dict(S.NOISE))
    assert shallow["sees_path_distance"] is False


def test_seeing_the_path_distance_is_not_reading_fingers_unless_the_rungs_agree():
    idx = np.random.default_rng(0).integers(0, 200, (Q.BOOT, 200))
    values = linear_values(2., noise=.05)
    values["finger_noise_030"], values["finger_noise_060"] = values["finger_noise_060"], values["finger_noise_030"]       # drops out of order
    out = S.detectable(values, idx, dict(S.NOISE))
    assert out["sees_path_distance"] is True and out["noise_rungs_agree"] is False and out["reads_fingers"] is False
    base = np.random.default_rng(1).normal(1., 1., 200)
    weak = {S.BASE: base, "finger_noise_015": base - .0001, "finger_noise_030": base - .0002, "finger_noise_060": base + np.random.default_rng(2).normal(-.0003, .5, 200) * 0 - .0003}
    flat = S.detectable(weak, idx, dict(S.NOISE))
    assert flat["noise_rungs_agree"] is True and flat["reads_fingers"] is flat["sees_path_distance"]
    noisy = {S.BASE: base, **{f: base - .0005 * i + np.random.default_rng(i).normal(0, .5, 200) for i, f in enumerate(S.NOISE)}}
    assert S.detectable(noisy, idx, dict(S.NOISE))["reads_fingers"] is False


@pytest.mark.parametrize("own,neutral,reading", [("falls", "falls", "reads_finger_motion"), ("falls", "holds", "reads_finger_motion"), ("falls", "unresolved", "reads_finger_motion"), ("holds", "falls", "static_handshape_not_motion"), ("holds", "holds", "no_finger_content"), ("holds", "unresolved", "unresolved"), ("unresolved", "falls", "unresolved"), ("unresolved", "holds", "unresolved")])
def test_the_finger_reading_is_the_headers(own, neutral, reading):
    assert S.finger_reading(own, neutral) == reading


def test_paired_share_reads_the_ratio_of_means_and_none_without_a_denominator():
    idx = np.random.default_rng(0).integers(0, 3, (50, 3))
    assert S.paired_share(np.array([1., 1., 1.]), np.array([2., 2., 2.]), idx) == dict(point=.5, ci95=[.5, .5])
    assert S.paired_share(np.ones(3), np.zeros(3), idx) is None


def test_the_ladder_has_its_planted_values_on_each_instruments_own_measures(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path, parent=dict(fingers_frozen=(14, 1.05, .95), fingers_neutral=(14, 1.05, .95), hands_still=(14, 1.05, .95)))
    report = h.pipeline()
    assert report["kind"] == "sensitivity-ladder-report" and report["report_only"] is True and report["alerts"] == [] and report["withheld"] == []
    parent, v1 = report["ladder"]["parent"], report["ladder"]["v1"]
    # margin over the still: the planted margin less the still's .1; likelihood: blank's 3 less the planted NLL
    assert abs(parent["forms"][S.BASE]["margin_over_still"]["point"] - .9) < 1e-6 and abs(v1["forms"][S.BASE]["likelihood_minus_blank"]["point"] - 2.) < 1e-6
    assert abs(parent["forms"]["fingers_frozen"]["margin_share"]["point"] - .85 / .9) < 1e-3 and abs(v1["forms"]["fingers_frozen"]["likelihood_share"]["point"] - 1.5 / 2.) < 1e-3
    assert abs(v1["forms"]["fingers_frozen"]["margin_drop"]["point"] - .5) < 1e-6 and abs(v1["forms"]["fingers_frozen"]["likelihood_drop"]["point"] - .5) < 1e-6
    assert parent["cases"]["margin"]["hands_still"] == "holds" and parent["reading_on_margin"] == dict(hands_still="holds", says_nothing_about_hands=True, finger_articulation="no_finger_content") and parent["label"] == parent["reading_on_margin"]
    # v1: its case on the margin beside its case on likelihood; only the margin can carry the bold sentence
    assert v1["cases"]["margin"]["hands_still"] == "falls" and v1["cases"]["likelihood"]["hands_still"] == "falls" and v1["reading_on_margin"]["says_nothing_about_hands"] is False and v1["reading_on_margin"]["finger_articulation"] == "reads_finger_motion"
    # the neutral rung's raw paired drop against the own pose, each measure
    assert abs(v1["neutral_against_frozen"]["margin"]["point"] - .2) < 1e-6 and abs(v1["neutral_against_frozen"]["likelihood"]["point"] - .2) < 1e-6
    assert report["parent_figures"]["nearest_over_still"] == pytest.approx(.9 * 1, abs=1e-5) and report["parent_figures"]["still_raw"] == pytest.approx(.1, abs=1e-5)
    assert report["clips"] == dict(list=N, in_v1_training=4, outside_v1_training=N - 4)
    assert parent["by_v1_training"]["in_v1_training"]["clips"] == 4 and parent["by_v1_training"]["outside_v1_training"]["clips"] == N - 4
    assert abs(parent["by_v1_training"]["in_v1_training"]["forms"][S.BASE]["margin_over_still"]["point"] - .9) < 1e-6
    assert report["instruments"]["parent"]["sha256"] == PP.UNFITTED_SHA and set(report["resources"]) == set(S.all_forms("base"))
    assert report["prestated"] == S.PRESTATED and report["set"] == "base"
    assert parent["forms"]["fingers_frozen"]["bleu"] < parent["forms"][S.BASE]["bleu"] and parent["forms"]["fingers_frozen"]["bleu_drop"]["point"] > 0


def test_each_rungs_distances_and_dropout_rates_are_reported(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    report = h.pipeline()
    distances = report["ladder"]["parent"]["distances"]
    assert set(distances) == set(S.RUNGS) and abs(distances["finger_noise_030"]["hands"] - .3) < .01 and abs(distances["wrist_moved"]["wrists"] - S.PATH_WRISTS) < .01
    assert set(report["ladder"]["v1"]["dropout_rates"]) == set(T.HANDS)
    assert report["ladder"]["parent"]["detectable"]["margin"]["levels"]["finger_noise_060"] == round(distances["finger_noise_060"]["hands"], 4)


def test_a_label_is_withheld_in_a_smoke(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path, declared=False, n=5)
    assert h.read_forms("control", smoke=True) == 0
    assert h.read_forms("rest", smoke=True, resume=True) == 0
    report = h.merge()
    assert report["ladder"]["parent"]["label"] is None and "a smoke carries no reading" in report["withheld"] and report["declared_read"] is False and "SMOKE" in report["smoke"]


@pytest.mark.parametrize("patch,word", [(dict(TARGET_TOLERANCE=-1.), "target"), (dict(HOLD_TOLERANCE=-1.), "moved"), (dict(DROP_FLOOR=.9, DROP_CEILING=1.), "realised rate")])
def test_an_alert_withholds_every_label_and_check_exits_3(monkeypatch, tmp_path, patch, word, capsys):
    h = Ladder(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", resume=True) == 0
    for k, v in patch.items():
        monkeypatch.setattr(S, k, v)
    report = h.merge()
    assert any(word in a for a in report["alerts"]) and report["ladder"]["parent"]["label"] is None and report["ladder"]["v1"]["label"] is None and report["ladder"]["parent"]["reading_on_margin"]["hands_still"] != ""
    assert S.main(["check", "--report", str(tmp_path / "report.json")]) == 3 and "ALERT" in capsys.readouterr().out


def test_a_single_clip_off_its_target_raises_an_alert_though_the_mean_holds(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", resume=True) == 0
    path = h.forms / "form-parent__finger_noise_060.json"
    form = json.loads(path.read_text())
    rows = form["rows"]
    rows[0]["distance"]["hands"] *= 1.2
    rows[1]["distance"]["hands"] *= .8
    path.write_text(json.dumps(form))
    report = h.merge()
    assert report["ladder"]["parent"]["label"] is None and any("2 clips miss the target 0.6" in a and "s0" in a for a in report["alerts"]) and not any("mean hands" in a for a in report["alerts"])


def test_hands_still_with_moving_points_raises_an_alert(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", resume=True) == 0
    path = h.forms / "form-v1__hands_still.json"
    form = json.loads(path.read_text())
    form["rows"][3]["local_motion"] = .5
    path.write_text(json.dumps(form))
    assert any("hands_still: hand-local points move on 1 clips, for example s3" in a for a in h.merge()["alerts"])


def test_the_count_in_v1s_training_is_checked_against_the_disclosed_number(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    assert S.V1_TRAINING_IN_D20 == 4        # the harness plants its own; the real pin is checked below
    h.to_control()
    assert h.read_forms("rest", resume=True) == 0
    monkeypatch.setattr(S, "V1_TRAINING_IN_D20", 234)
    assert any("4 of D20 are in v1's training list, not the disclosed 234" in a for a in h.merge()["alerts"])


def test_the_parent_figures_are_logged_and_each_is_held_within_its_tolerance(monkeypatch, tmp_path, capsys):
    h = Ladder(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", resume=True) == 0
    monkeypatch.setattr(S, "PARENT_FIGURES", dict(nearest_over_still=.9 + S.FIGURE_TOLERANCE * 2, still_raw=1.))
    report = h.merge()
    assert report["parent_figures"]["still_raw"] == pytest.approx(.1, abs=1e-5)
    assert any("nearest_over_still margin" in a for a in report["alerts"]) and any("still_raw margin" in a for a in report["alerts"]) and "parent_figures" in capsys.readouterr().out


@pytest.mark.parametrize("off,alerted", [(.0049, False), (.0051, True)])
def test_a_figure_is_held_to_its_tolerance_either_side(monkeypatch, off, alerted):
    monkeypatch.setattr(S, "PARENT_FIGURES", dict(still_raw=.06))
    assert bool(S.figure_alerts(dict(still_raw=.06 + off))) is alerted and bool(S.figure_alerts(dict(still_raw=.06 - off))) is alerted


def test_the_real_figures_and_count_are_pinned_as_recorded():
    assert S.PARENT_FIGURES == dict(nearest_over_still=.2372, still_raw=.0610) and S.FIGURE_TOLERANCE == .005 and S.V1_TRAINING_IN_D20 == 234


def test_the_stored_check_hashes_every_form_and_prints_each_environment(monkeypatch, tmp_path, capsys):
    h = Ladder(monkeypatch, tmp_path)
    h.plant_stored_check()
    assert h.main("stored-check") == 0
    out = capsys.readouterr().out
    assert '"environment_here": {"gpu": "test"}' in out and '"parent.still_two_view"' in out and '"v1.nearest_monocular"' in out and "stored-gpu" in out


@pytest.mark.parametrize("form,field,value,word", [("parent.blank", "nll", [[1.]] * N, "columns"), ("parent.real_two_view", "signs", ["a"], "signs"), ("v1.blank", "signs", [], "signs")])
def test_the_stored_check_refuses_a_form_of_the_wrong_shape(monkeypatch, tmp_path, form, field, value, word):
    h = Ladder(monkeypatch, tmp_path)
    h.plant_stored_check({form: {field: value}})
    with pytest.raises(SystemExit, match=word):
        h.main("stored-check")


def test_a_pin_failure_exits_1_and_voids_nothing(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    h.plant_stored_check()
    h.store.put(f"{Q.PARTS_FORMS}/check-parent.blank.json", b"{}")
    with pytest.raises(SystemExit) as stopped:
        h.main("stored-check")
    assert "SHA-256" in str(stopped.value.code)
    h2 = Ladder(monkeypatch, tmp_path / "b") if (tmp_path / "b").mkdir() is None else None
    assert h2.read_forms("control") == 0
    h2.plant_stored_reads()
    key, _ = Q.STORED_PARENT["blank"]
    h2.store.put(f"{Q.PARTS_FORMS}/{key}", b"{}")
    with pytest.raises(SystemExit, match="SHA-256"):
        h2.control()
    assert not h2.store.exists("lad/void/" + "c" * 64 + ".json")


def test_a_canary_pin_failure_stops_the_stored_check(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    h.plant_stored_check()
    monkeypatch.setattr(P, "CANARY_REPORT_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        h.main("stored-check")


def test_another_instruments_hash_raises_an_alert(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    h.parent.info = dict(PARENT_INFO, sha256="b" * 64)
    report = h.pipeline()
    assert any("not its pin" in a for a in report["alerts"]) and report["ladder"]["parent"]["label"] is None


def test_too_few_clips_raise_an_alert(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", resume=True) == 0
    monkeypatch.setattr(L, "D20_CLIPS", N + 1)
    assert any("not D20's" in a for a in h.merge()["alerts"])


def test_the_check_passes_a_clean_report_and_refuses_broken_plumbing(monkeypatch, tmp_path, capsys):
    h = Ladder(monkeypatch, tmp_path)
    h.pipeline()
    path = tmp_path / "report.json"
    assert S.main(["check", "--report", str(path)]) == 0
    clean = json.loads(path.read_text())
    for change, word in ((dict(report_only=False), "report_only"), (dict(checks=dict(clean["checks"], control=None)), "passed control"), (dict(checks=dict(clean["checks"], blank_identical_to_base=dict(v1=1.))), "blank reads"), (dict(withheld=["x"]), "label beside")):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(dict(clean, **change)))
        assert S.main(["check", "--report", str(bad)]) == 3
        assert word in capsys.readouterr().out


def test_a_missing_form_or_one_holding_another_refuses_the_merge(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", resume=True) == 0
    moved = h.forms / "form-parent__arms_frozen.json"
    raw = moved.read_bytes()
    moved.unlink()
    with pytest.raises(SystemExit, match="is missing"):
        h.merge()
    moved.write_bytes((h.forms / "form-v1__arms_frozen.json").read_bytes())
    with pytest.raises(SystemExit, match="holds v1 arms_frozen"):
        h.merge()
    moved.write_bytes(raw)
    assert h.merge("again.json")["alerts"] == []


def test_stale_forms_of_other_code_refuse_the_merge(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", resume=True) == 0
    monkeypatch.setattr(S, "code_digest", lambda: "e" * 64)
    with pytest.raises(SystemExit, match="a stale read"):
        h.merge()
    monkeypatch.setattr(S, "code_digest", lambda: "c" * 64)
    path = h.forms / "form-parent__arms_frozen.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), code_digest="e" * 64)))
    with pytest.raises(SystemExit, match="different code digests"):
        h.merge()


@pytest.mark.parametrize("change,match", [(dict(environment=dict(gpu="other")), "different environments"), (dict(declared_read=False), "smoke mixed"), (dict(signs=[f"z{i}" for i in range(N)]), "other clips")])
def test_forms_of_another_run_refuse_the_merge(monkeypatch, tmp_path, change, match):
    h = Ladder(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", resume=True) == 0
    path = h.forms / "form-parent__arms_frozen.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), **change)))
    with pytest.raises(SystemExit, match=match):
        h.merge()


def test_a_report_is_write_once_and_a_finished_read_stops_preflight(monkeypatch, tmp_path, capsys):
    h = Ladder(monkeypatch, tmp_path)
    h.pipeline()
    with pytest.raises(SystemExit, match="write-once"):
        h.merge()
    h.store.put("lad/report.json", b"{}")
    with pytest.raises(SystemExit, match="finished"):
        h.main("preflight", "--out-dir", str(tmp_path / "w3"), "--final-key", "lad/report.json", "--void-prefix", "lad/void")
    assert h.main("preflight", "--out-dir", str(tmp_path / "w3"), "--final-key", "lad/other.json", "--void-prefix", "lad/void") == 0 and "clear" in capsys.readouterr().out


def test_the_base_instruments_step_reads_records_and_fetches_no_checkpoint(monkeypatch, tmp_path, capsys):
    h = Ladder(monkeypatch, tmp_path)
    monkeypatch.setattr(PP, "instrument_block", lambda store: dict(unfitted="ok"))
    assert h.main("instruments") == 0 and '"unfitted": "ok"' in capsys.readouterr().out
    plant_mono(h.store)
    assert h.main("instruments", "--set", "mono") == 0 and Q.MONO_PREFIX in capsys.readouterr().out
    assert S.expected_sha_of("mono", h.store) == hashlib.sha256(b"weights").hexdigest() and S.expected_sha_of("parent", h.store) == PP.UNFITTED_SHA


def test_the_mono_set_merges_on_its_own_with_its_pin(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    plant_mono(h.store, payload=b"mono")
    h.mono.info = dict(h.mono.info, sha256=hashlib.sha256(b"mono").hexdigest())
    assert h.read_forms("control", which="mono") == 0
    read = {n: json.loads((h.forms / f"form-{n}.json").read_text()) for n in S.control_forms("mono")}
    h.plant_canary(read)
    assert h.control("mono") == 0 and h.read_forms("rest", which="mono") == 0
    report = h.merge(which="mono")
    assert list(report["ladder"]) == ["mono"] and report["alerts"] == [] and set(report["ladder"]["mono"]["cases"]) == {"margin", "likelihood"}


def test_the_header_holds_the_reading_before_any_read():
    head = S.__doc__
    for text in S.PRESTATED.values():
        assert text in head
    assert "0.587" in head and "1.286" in head and "0.231" in head and "234 of D20" in head
    for text in ("finger articulation", "one real handshape", "Only that rung may say it", "biased low", "presence offset", "about 0.18", "necessary, not sufficient"):
        assert text in head, text
    for text in ("one real handshape", "biased low", "presence offset", "necessary, not sufficient"):
        assert any(text in v for v in S.DISCLOSURES.values()), text
    assert "not checked against the source" in head and S.KEPT == .75 and S.PATH_HANDS == .587 and S.MIRROR_HANDS == 1.286 and S.PATH_WRISTS == .231
    assert S.NOISE == dict(finger_noise_015=.15, finger_noise_030=.30, finger_noise_060=.60) and S.DROP_RATE == .25 and S.DROP_RUN == 6. and S.BOX == 7


def test_no_code_path_names_a_generated_form_c20_or_a_test_report():
    text = (ROOT / "scripts/sensitivity_ladder.py").read_text().split('"""', 2)[2]
    for word in ("round15_compose_eval", "core_crossfade", "c20.json", "test_rgb_front", "val_rgb_front", "monocular_rows", "open_monocular"):
        assert word not in text, word


# ---- the job ---------------------------------------------------------------------------------------------------------------------------

def run_job(tmp_path, fail_when=None, code=7, tee_fails=False, **env):
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    pattern = "*" + "*".join(f'"{x}"' for x in fail_when.split("*")) + "*" if fail_when else ""
    match = f'case "$*" in {pattern}) echo "planted failure" >&2; exit {code};; esac' if fail_when else ""
    (bin_dir / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\n{match}\necho "ran $1 $2"\nexit 0\n')
    (bin_dir / "uv").write_text("#!/bin/bash\nexit 0\n")
    names = ["python", "uv"]
    if tee_fails:
        (bin_dir / "tee").write_text("#!/bin/bash\n/bin/cat > /dev/null\nexit 9\n")
        names.append("tee")
    for name in names:
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    environment = dict(PATH=f"{bin_dir}:/usr/bin:/bin", HOME=str(tmp_path), VENV_BIN=str(bin_dir), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path), WORK=str(tmp_path / "work"), **env)
    done = subprocess.run(["bash", str(JOB)], env=environment, capture_output=True, text=True, timeout=60)
    return done, log.read_text() if log.exists() else ""


def done_line(result) -> int:
    lines = [x for x in result.stdout.splitlines() if x.startswith("SENSITIVITY-LADDER-DONE ")]
    assert len(lines) == 1 and lines[0] == result.stdout.strip().splitlines()[-1]
    return int(lines[0].split()[1])


def test_the_runner_sees_no_credentials():
    assert not [k for k in os.environ if k.startswith(("R2_", "AWS_", "RUNPOD_"))]


def commands(calls: str) -> list[str]:
    own = calls.splitlines()
    assert all(x.startswith("scripts/sensitivity_ladder.py ") for x in own)
    return [S.build_parser().parse_args(x.split()[1:]).command for x in own]


def test_the_job_smokes_then_reads_the_control_first(tmp_path):
    result, calls = run_job(tmp_path)
    assert result.returncode == 0 and done_line(result) == 0
    assert commands(calls) == ["instruments", "stored-check", "read", "control", "read", "merge", "upload", "check", "preflight", "read", "control", "read", "merge", "upload", "check", "upload-log"]
    lines = calls.splitlines()
    assert "--forms control" in lines[9] and "--forms rest" in lines[11] and "--resume" in lines[11] and "--smoke-train 5" in lines[2] and "--smoke-train" not in lines[9]
    assert "round20/sensitivity-ladder/report.json" in lines[8] and "--set base" in lines[9] and "--nll-group 4" in lines[9]


def test_the_smoke_only_mode_stops_before_d20(tmp_path):
    result, calls = run_job(tmp_path, SMOKE="1")
    assert result.returncode == 0 and "preflight" not in calls and "SMOKE done" in result.stdout


def test_the_mono_set_writes_under_its_own_prefix(tmp_path):
    result, calls = run_job(tmp_path, SET="mono")
    assert result.returncode == 0 and "round20/sensitivity-ladder-mono/report.json" in calls and "--set mono" in calls and "--set base" not in calls


def test_another_set_refuses_before_anything_runs(tmp_path):
    result, calls = run_job(tmp_path, SET="c20")
    assert result.returncode == 1 and done_line(result) == 1 and calls == "" and "SET must be" in result.stdout


@pytest.mark.parametrize("fail,status,word", [("instruments", 1, "instruments are not"), ("stored-check", 1, "this is no void"), ("--forms rest*--smoke-train", 2, "smoke read failed"), ("--forms control*--resume", 1, "read of the control"), ("preflight", 1, "preflight refused"), ("--forms rest*--resume", 1, "read of the rungs"), ("upload --out-dir", 2, "smoke upload failed")])
def test_a_failing_step_stops_the_job_with_its_status(tmp_path, fail, status, word):
    result, calls = run_job(tmp_path, fail_when=fail)
    assert result.returncode == status and done_line(result) == status and word in result.stdout and "upload --partial" in calls


def test_a_void_control_stops_with_3_and_reads_no_rung(tmp_path):
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    (bin_dir / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\ncase "$*" in *control*work/d20/forms*--void*) exit 3;; esac\nexit 0\n')
    (bin_dir / "uv").write_text("#!/bin/bash\nexit 0\n")
    for name in ("python", "uv"):
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    done = subprocess.run(["bash", str(JOB)], env=dict(PATH=f"{bin_dir}:/usr/bin:/bin", HOME=str(tmp_path), VENV_BIN=str(bin_dir), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path), WORK=str(tmp_path / "work")), capture_output=True, text=True, timeout=60)
    assert done.returncode == 3 and done_line(done) == 3 and "do not retry" in done.stdout
    assert not [x for x in log.read_text().splitlines() if "--forms rest" in x and "work/d20" in x]


def test_a_log_that_cannot_be_written_fails_the_step(tmp_path):
    result, _ = run_job(tmp_path, tee_fails=True)
    assert result.returncode != 0 and "log could not be written" in result.stdout


def test_a_missing_repo_prints_its_done_line(tmp_path):
    done = subprocess.run(["bash", str(JOB)], env=dict(PATH="/usr/bin:/bin", HOME=str(tmp_path), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path / "gone")), capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and done.stdout.strip().endswith("SENSITIVITY-LADDER-DONE 1")


def test_the_job_passes_every_step_through_pipestatus_and_names_no_other_list():
    text = JOB.read_text()
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0
    code = [line.split(" #")[0] for line in text.splitlines() if not line.startswith("#")]
    assert [line for line in code if re.search(r"(?<!\|)\|(?!\|)", line) and "case" not in line] == ['  "$@" 2>&1 | tee -a "$LOG"'] and "${PIPESTATUS[@]}" in text and "set -uo pipefail" in text and "transformers==4.57.6" in text
    python_lines = [line.strip() for line in code if re.search(r"\bpython \$T\b", line)]
    assert python_lines and all(line.startswith("step python $T") or "upload --partial" in line or "upload-log" in line for line in python_lines)
    for word in ("c20", "C20", "round15_compose_eval", "composed", "--export", "create-pod", "runpodctl"):
        assert word not in "\n".join(code), word
    assert "L40S" in text


def test_the_boundaries_of_seeing_the_path_and_of_agreeing(monkeypatch):
    idx = np.random.default_rng(0).integers(0, 200, (Q.BOOT, 200))
    values = linear_values(2., noise=.05)
    size = S.detectable(values, idx, dict(S.NOISE))["detectable_palm_widths"]
    monkeypatch.setattr(S, "PATH_HANDS", size)
    assert S.detectable(values, idx, dict(S.NOISE))["sees_path_distance"] is True
    equal = dict(values, finger_noise_030=values["finger_noise_015"])
    assert S.detectable(equal, idx, dict(S.NOISE))["noise_rungs_agree"] is True


def test_exactly_at_a_tolerance_raises_no_alert(monkeypatch, tmp_path):
    h = Ladder(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", resume=True) == 0
    reads = S.load_reads(h.forms, "parent")
    reads["hands_still"]["rows"][0]["local_motion"] = S.HOLD_TOLERANCE
    for form, target in S.TARGETS.items():
        reads[form]["rows"][0]["distance"][S.TARGET_GROUP[form]] = target * (1 + S.TARGET_TOLERANCE / 2)
    monkeypatch.setattr(S, "TARGET_TOLERANCE", S.TARGET_TOLERANCE)
    alerts = S.plumbing_alerts(reads, S.distance_means(reads), S.dropout_rates(reads), "parent", PP.UNFITTED_SHA, True)
    assert alerts == []
    reads["hands_still"]["rows"][0]["local_motion"] = S.HOLD_TOLERANCE * 1.01
    assert len(S.plumbing_alerts(reads, S.distance_means(reads), S.dropout_rates(reads), "parent", PP.UNFITTED_SHA, True)) == 1
