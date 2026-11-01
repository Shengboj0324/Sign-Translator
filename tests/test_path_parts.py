"""The path read's follow-up on synthetic data and the real builders' layouts: which rotations each form takes, the isolated split in what v1 reads, the instruments' pins and training records, the control, both readings and their alerts, and the job. No test starts a real job: the runner tests use fake executables and no credentials, and the store is a directory or a stand-in."""
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
import path_parts as PP  # noqa: E402
import path_read as R  # noqa: E402
import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v0 as V  # noqa: E402
import tok_v1 as T  # noqa: E402
from signtranslator.data import layout_adapter  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402
from signtranslator.eval import provenance, s0_metrics as M  # noqa: E402
from tests.test_path_read import CUT, Harness, N, SHARD, built_pair, d20_records, monocular_state, plant_shard, template_state, tiny_tokenizer, train_records, two_view_state, with_empty_hand  # noqa: E402
from tests.test_tok_probes import SpyStore  # noqa: E402

JOB = ROOT / "runpod/fleet/path_parts.sh"
LEFT_LOCAL, RIGHT_LOCAL = range(7, 27), range(28, 48)
ARM_POINTS = (0, 1, 2, 3, 4, 5, 6, 27)


def plant_instruments(store, **change):
    """The three run records as R2 holds them, changed as asked: {"unfitted.args.x": value}."""
    unfitted = dict(args=dict(init_key=None, init_sha256=None, train_index=[PP.UWB_INDEX], select_on="monitor", monitor_index=PP.UWB_INDEX, val_index=PP.HOW2SIGN_FK_INDEX), config=dict(features=144, fps=24., d_model=384), train_clips=323978)
    best = dict(checkpoint_key=PP.UNFITTED_KEY, step=76000)
    v1 = dict(args=dict(init_key=PP.UNFITTED_KEY, init_sha256=PP.UNFITTED_SHA, train_index=[PP.HOW2SIGN_FK_INDEX], select_on="val"), config=dict(features=144, fps=24., d_model=384), train_clips=17000)
    objects = dict(unfitted=unfitted, best=best, v1=v1)
    for path, value in change.items():
        where = objects
        *parts, last = path.split(".")
        for part in parts:
            where = where[part]
        where[last] = value
    store.put(f"training-runs/{PP.UNFITTED_RUN}/run.json", json.dumps(unfitted).encode())
    store.put(f"training-runs/{PP.UNFITTED_RUN}/best.json", json.dumps(best).encode())
    store.put(f"training-runs/{L.TRANSLATOR_RUN}/run.json", json.dumps(v1).encode())


class PartsReader:
    """v1: reads whose rotations a state holds, joint group by joint group, under whose marks, so each form has a known level and a mislabelled swap reads as something else."""
    info = dict(run="planted-v1", checkpoint_key="k", sha256="f" * 64, kind="fake")
    LEVELS = dict(two_view=(15, 1.), monocular_values=(10, 1.6), mono_body=(11, 1.5), mono_hands=(13, 1.2), arms=(12, 1.4), hand_pose=(14, 1.1), nearest_two=(11, 1.5), nearest_mono=(7, 2.2), blank=(0, 3.), rt_two_view=(5, 2.75), other=(2, 2.95))

    def __init__(self, truth, levels=None) -> None:
        self.truth, self.levels = truth, dict(self.LEVELS, **(levels or {}))

    @staticmethod
    def source(channel, picks, two, mono) -> str:
        for name, other in (("two", two), ("mono", mono)):
            both = channel.valid[:, picks] & other.valid[:, picks]
            if both.any() and np.allclose(channel.values[:, picks][both], other.values[:, picks][both], atol=1e-3):
                return name
        return "other"

    def level(self, state, text) -> str:
        marks = np.concatenate([state.channels[n].valid for n in ("body", "left_hand", "right_hand")], axis=1)
        if not marks.any():
            return "blank"
        two, mono = self.truth[text]
        whole = bool(marks.all())
        if len(state.timestamps) != len(two.timestamps):
            return "nearest_mono" if whole else "other"
        if whole:
            return "nearest_two"
        cut = R.cut_frames(mono, np.arange(*CUT))
        body = state.channels["body"]
        labels = list(body.labels)
        arms = [labels.index(x) for x in ("Chest", "LeftUpperArm", "LeftForearm", "RightUpperArm", "RightForearm")]
        wrists = [labels.index(x) for x in ("LeftHand", "RightHand")]
        fingers = list(range(state.channels["left_hand"].values.shape[1]))
        key = (self.source(body, arms, two.channels["body"], cut.channels["body"]), self.source(body, wrists, two.channels["body"], cut.channels["body"]), self.source(state.channels["left_hand"], fingers, two.channels["left_hand"], cut.channels["left_hand"]))
        return {("two", "two", "two"): "two_view", ("mono", "mono", "mono"): "monocular_values", ("mono", "mono", "two"): "mono_body", ("two", "two", "mono"): "mono_hands", ("mono", "other", "two"): "arms", ("two", "other", "mono"): "hand_pose"}.get(key, "other")

    def read(self, states, texts, candidates, group=2):
        rows = [self.levels[self.level(s, t)] for s, t in zip(states, texts)]
        return dict(hypotheses=[" ".join(t.split()[:k]) if k else "z y x w" for t, (k, _) in zip(texts, rows)], nll=np.array([[v] for _, v in rows]))


class UnfittedReader:
    """The unfitted instrument: by default it holds a monocular gap of 0.5 of its own 1.0, and its own blank."""
    info = dict(run=PP.UNFITTED_RUN, checkpoint_key=PP.UNFITTED_KEY, sha256=PP.UNFITTED_SHA, kind="keypoints")

    def __init__(self, two=(6, 2.), mono=(5, 2.5), blank=(0, 3.)) -> None:
        self.two, self.mono, self.blank = two, mono, blank

    def read(self, states, texts, candidates, group=2):
        rows = []
        for s in states:
            rows.append(self.blank if not s.channels["left_hand"].valid.any() else (self.two if len(s.timestamps) == 20 else self.mono))
        return dict(hypotheses=[" ".join(t.split()[:k]) if k else "z y x w" for t, (k, _) in zip(texts, rows)], nll=np.array([[v] for _, v in rows]))


class Parts(Harness):
    def __init__(self, monkeypatch, tmp_path, *, unfitted=None, part_levels=None, floor=True, **kw):
        super().__init__(monkeypatch, tmp_path, reader=PartsReader({}, part_levels), **kw)
        self.reader.truth = {r["text"]: (ref, state) for r, ref, state in zip(self.records, self.refs, self.mono)}
        self.unfitted = unfitted or UnfittedReader()
        monkeypatch.setattr(PP, "load_unfitted", lambda store, args: self.unfitted)
        plant_instruments(self.store)
        if floor:
            monkeypatch.setattr(R, "SHARED_FLOOR", 1)

    def main(self, *argv):
        return PP.main([argv[0], "--store", f"dir:{self.tmp_path / 's'}", *argv[1:]])

    def plant_control(self, *, hyp_change=0, nll_shift=0., bleu_shift=0., order=False):
        read = {n: json.loads((self.forms / f"form-{n}.json").read_text()) for n in PP.ANCHORS}
        real = list(read["real_two_view"]["hypotheses"])
        for i in range(hyp_change):
            real[i] += " changed"
        stored = dict(signs=list(reversed(read["real_two_view"]["signs"])) if order else read["real_two_view"]["signs"], references=read["real_two_view"]["references"],
                      forms=dict(real=dict(hypotheses=real, per_clip=dict(nll=[r[0] + nll_shift for r in read["real_two_view"]["nll"]])), blank=dict(hypotheses=read["blank"]["hypotheses"], per_clip=dict(nll=[r[0] for r in read["blank"]["nll"]]))))
        raw = json.dumps(stored).encode()
        self.store.put(P.CANARY_REPORT, raw)
        self.monkeypatch.setattr(P, "CANARY_REPORT_SHA", hashlib.sha256(raw).hexdigest())
        self.monkeypatch.setattr(P, "CANARY", dict(P.CANARY, real=M.bleu_ci_fast(read["real_two_view"]["hypotheses"], read["real_two_view"]["references"])[0] + bleu_shift, blank=M.bleu_ci_fast(read["blank"]["hypotheses"], read["blank"]["references"])[0]))

    def to_control(self):
        assert self.read(*PP.ANCHORS) == 0
        self.plant_control()
        assert self.control() == 0

    def plant_figures(self):
        figures = {}
        blank = json.loads((self.forms / "form-blank.json").read_text())
        base = {s: row[0] for s, row in zip(blank["signs"], blank["nll"])}
        for form in PP.FIGURES:
            read = json.loads((self.forms / f"form-{form}.json").read_text())
            figures[form] = (M.bleu_ci_fast(read["hypotheses"], read["references"])[0], float(np.mean([base[s] - r[0] for s, r in zip(read["signs"], read["nll"])])))
        self.monkeypatch.setattr(PP, "FIGURES", figures)

    def pipeline(self, name="report.json", figures=True):
        self.to_control()
        assert self.read(*PP.PART_FORMS, *PP.NEAREST_FORMS, resume=True) == 0
        assert self.read(*PP.UNFITTED_FORMS, resume=True) == 0
        if figures:
            self.plant_figures()
        return self.merge(name)


# ---- pins and instruments ---------------------------------------------------------------------------------------------------------

def test_the_unfitted_instrument_is_pinned_and_is_not_v1():
    assert PP.UNFITTED_RUN == "back-translator-kp2d-uwb-v2" and PP.UNFITTED_KEY == "training-runs/back-translator-kp2d-uwb-v2/checkpoints/step-00076000.pt" and PP.UNFITTED_SHA == "f1ff94466a9e12c4d281c2d3dbc281c1049bc081f9123957c5e2dacde9e1821b" and PP.UNFITTED_SHA != L.TRANSLATOR_SHA


def test_the_instruments_block_states_what_each_trained_on(tmp_path):
    store = DirectoryStore(tmp_path / "s")
    plant_instruments(store)
    block = PP.instrument_block(store)
    assert block["unfitted"]["trained_on"] == [PP.UWB_INDEX] and block["unfitted"]["sha256"] == PP.UNFITTED_SHA and block["unfitted"]["step"] == 76000 and block["unfitted"]["selected_on"] == "monitor" and block["unfitted"]["validation_monitor_only"] == PP.HOW2SIGN_FK_INDEX and block["unfitted"]["train_clips"] == 323978 and "MediaPipe" in block["unfitted"]["landmarks"] and block["unfitted"]["started_from"] is None
    assert block["v1"]["trained_on"] == [PP.HOW2SIGN_FK_INDEX] and block["v1"]["started_from"] == PP.UNFITTED_KEY and block["v1"]["started_from_sha256"] == PP.UNFITTED_SHA and block["v1"]["sha256"] == L.TRANSLATOR_SHA and block["v1"]["selected_on"] == "val" and "two-view" in block["v1"]["landmarks"]
    assert block["pre_fine_tune_is_the_unfitted_checkpoint"] is True and block["same_input"] == dict(features=144, fps=24., configs_identical=True)
    plant_instruments(store, **{"unfitted.config.d_model": 512})
    assert PP.instrument_block(store)["same_input"]["configs_identical"] is False


@pytest.mark.parametrize("change,why", [({"best.checkpoint_key": "other.pt"}, "not its run's best"), ({"unfitted.args.init_key": "x"}, "other weights"), ({"unfitted.args.init_sha256": "x"}, "other weights"), ({"unfitted.args.train_index": [PP.UWB_INDEX, PP.HOW2SIGN_FK_INDEX]}, "more than UWB"), ({"unfitted.args.select_on": "val"}, "selected on How2Sign"), ({"unfitted.args.monitor_index": PP.HOW2SIGN_FK_INDEX}, "selected on How2Sign"),
                                        ({"v1.args.init_key": "x"}, "not fine-tuned from this"), ({"v1.args.init_sha256": "0" * 64}, "not fine-tuned from this"), ({"v1.args.train_index": [PP.UWB_INDEX]}, "did not fine-tune on How2Sign"), ({"unfitted.config.features": 96}, "different features"), ({"unfitted.config.fps": 25.}, "different features")])
def test_instruments_not_as_the_header_states_stop_the_job(tmp_path, change, why):
    store = DirectoryStore(tmp_path / "s")
    plant_instruments(store, **change)
    with pytest.raises(SystemExit, match=why):
        PP.instrument_block(store)


def test_the_instruments_step_fetches_no_checkpoint(monkeypatch, tmp_path, capsys):
    store = DirectoryStore(tmp_path / "s")
    plant_instruments(store)
    seen = []
    original = store.get
    store.get = lambda key: seen.append(key) or original(key)
    PP.instrument_block(store)
    assert seen and not any(k.endswith(".pt") for k in seen)
    assert PP.main(["instruments", "--store", f"dir:{tmp_path / 's'}"]) == 0
    assert json.loads(capsys.readouterr().out)["instruments"]["unfitted"]["sha256"] == PP.UNFITTED_SHA


def test_the_unfitted_checkpoint_is_refused_on_any_other_bytes():
    store = SpyStore({PP.UNFITTED_KEY: b"other bytes"})
    args = PP.build_parser().parse_args(["read", "--out-dir", "x"])
    with pytest.raises(SystemExit, match="SHA-256"):
        PP.load_unfitted(store, args)
    assert store.read == [PP.UNFITTED_KEY]


def test_the_unfitted_reader_carries_its_own_pin_and_refuses_another_kind(monkeypatch):
    import score_back_translation as S
    args = PP.build_parser().parse_args(["read", "--out-dir", "x"])
    seen = {}

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self
    monkeypatch.setattr(L, "instrument_payload", lambda store, key, sha, cache: seen.update(key=key, sha=sha) or (b"p", sha))
    monkeypatch.setattr(S, "load_translator", lambda one, key, manifest: seen.update(loaded=key, payload=one.get("x")) or (Model(), "keypoints"))
    reader = PP.load_unfitted(None, args)
    assert seen == dict(key=PP.UNFITTED_KEY, sha=PP.UNFITTED_SHA, loaded=PP.UNFITTED_KEY, payload=b"p") and reader.info == dict(run=PP.UNFITTED_RUN, checkpoint_key=PP.UNFITTED_KEY, sha256=PP.UNFITTED_SHA, kind="keypoints") and reader.kind == "keypoints"
    monkeypatch.setattr(S, "load_translator", lambda one, key, manifest: (Model(), "states"))
    with pytest.raises(SystemExit, match="not a 2D keypoint reader"):
        PP.load_unfitted(None, args)


def test_a_form_takes_its_own_instrument_loaded_once(monkeypatch):
    loads = []
    monkeypatch.setattr(PP, "load_unfitted", lambda store, args: loads.append("u") or "U")
    monkeypatch.setattr(P, "load_reader", lambda store, args: loads.append("v") or "V")
    cache = {}
    assert [PP.reader_of(n, None, None, cache) for n in ("real_two_view", "uwb_blank", "nearest_monocular", "uwb_nearest_monocular")] == ["V", "U", "V", "U"] and loads == ["v", "u"]


def test_no_command_has_an_option_for_another_list_instrument_or_tokenizer(capsys):
    for command in ("read", "merge", "control", "preflight", "instruments"):
        with pytest.raises(SystemExit):
            PP.build_parser().parse_args([command, "--help"])
        text = capsys.readouterr().out
        for flag in ("--sentences", "--translator", "--instrument-run", "--tokenizer", "--manifest", "--list", "--clips", "--template", "--states", "--checkpoint", "--reader"):
            assert flag not in text


def test_no_code_path_names_another_list_a_generated_form_or_a_test_report():
    source = (ROOT / "scripts/path_parts.py").read_text()
    code = re.sub(r"\b[0-9a-f]{64}\b", "", source[source.index('from __future__'):])
    for word in ("c20", "C20", "val1000", "eval1000", "composed", "compose_eval", "word_cache", "a9p", "test_rgb", "val_rgb", ".list(", "state-t5", "kp2d-h2s"):
        assert word not in code.split("DISCLOSURES")[0] and word not in code.split("def own_values")[1], word


# ---- which rotations each form takes ----------------------------------------------------------------------------------------------

def pair(i=0):
    raw = monocular_state(i)
    trimmed = R.cut_frames(layout_adapter.conform(raw, template_state()), np.arange(*CUT))
    return trimmed, two_view_state(i, "a" * 64)


def test_own_values_are_every_frame_tracked_and_a_never_tracked_joint_holds_the_neutral():
    tok = tiny_tokenizer()
    trimmed, _ = pair()
    out = PP.own_values(trimmed, tok)
    assert out.channels["left_hand"].valid.all()
    c = trimmed.channels["left_hand"]
    np.testing.assert_array_equal(out.channels["left_hand"].values[c.valid], c.values[c.valid])
    empty = with_empty_hand(trimmed)
    np.testing.assert_allclose(PP.own_values(empty, tok).channels["left_hand"].values, V.neutral_pose(empty, tok).channels["left_hand"].values)


def test_body_and_hands_take_each_channel_from_its_own_source_under_the_marks():
    tok = tiny_tokenizer()
    trimmed, ref = pair()
    mono, two = PP.own_values(trimmed, tok), PP.own_values(ref, tok)
    out = PP.body_and_hands(mono, two, ref)
    for name, source in (("body", mono), ("left_hand", two), ("right_hand", two)):
        c = out.channels[name]
        np.testing.assert_array_equal(c.valid, ref.channels[name].valid)
        np.testing.assert_array_equal(c.values[c.valid], source.channels[name].values[c.valid])


def v1_points(state):
    from signtranslator.data.keypoints2d import from_state, normalise
    points, valid = from_state(state)
    return normalise(points, valid, y_down=False), valid


def real_builder_forms():
    """Two clearly different clips through the real builders' layouts: one state and the other's rotations under the same frames."""
    tok = tiny_tokenizer()
    trimmed, ref = pair(1)
    return PP.own_values(trimmed, tok), PP.own_values(ref, tok), ref


def test_arms_only_gives_v1_the_monoculars_arm_points_and_two_views_hand_local_points_exactly():
    mono, two, ref = real_builder_forms()
    out, ok = v1_points(PP.arms_and_hand_pose(mono, two, ref))
    arm, _ = v1_points(V.with_marks(mono, ref))
    hand, _ = v1_points(V.with_marks(two, ref))
    both = ok.any(0)
    assert both[list(ARM_POINTS)].all() and both[list(LEFT_LOCAL)].all()
    np.testing.assert_allclose(out[:, list(ARM_POINTS)], arm[:, list(ARM_POINTS)], atol=1e-5)
    np.testing.assert_allclose(out[:, list(LEFT_LOCAL) + list(RIGHT_LOCAL)], hand[:, list(LEFT_LOCAL) + list(RIGHT_LOCAL)], atol=1e-4)
    assert np.abs(arm[:, list(LEFT_LOCAL)] - hand[:, list(LEFT_LOCAL)]).max() > .05 and np.abs(arm[:, list(ARM_POINTS)] - hand[:, list(ARM_POINTS)]).max() > .05       # the two sources really differ on both


def test_hand_pose_only_is_the_reverse():
    mono, two, ref = real_builder_forms()
    out, _ = v1_points(PP.arms_and_hand_pose(two, mono, ref))
    arm, _ = v1_points(V.with_marks(two, ref))
    hand, _ = v1_points(V.with_marks(mono, ref))
    np.testing.assert_allclose(out[:, list(ARM_POINTS)], arm[:, list(ARM_POINTS)], atol=1e-5)
    np.testing.assert_allclose(out[:, list(LEFT_LOCAL) + list(RIGHT_LOCAL)], hand[:, list(LEFT_LOCAL) + list(RIGHT_LOCAL)], atol=1e-4)


def test_a_plain_arm_swap_turns_the_hand_which_is_why_the_wrist_is_recomputed():
    """The header's reason: rotations are parent-relative. Taking the arm rotations alone moves two-view's hand-local points."""
    mono, two, ref = real_builder_forms()
    body = mono.channels["body"]
    keep = [k for k, label in enumerate(body.labels) if label in ("LeftHand", "RightHand")]
    values = body.values.copy()
    values[:, keep] = two.channels["body"].values[:, keep]
    plain = V.with_marks(replace(two, channels={**two.channels, "body": replace(body, values=values)}), ref)
    out, _ = v1_points(plain)
    hand, _ = v1_points(V.with_marks(two, ref))
    assert np.abs(out[:, list(LEFT_LOCAL)] - hand[:, list(LEFT_LOCAL)]).max() > .05


def test_the_held_hand_changes_only_the_wrist_rotations_of_the_arms_source():
    mono, two, _ = real_builder_forms()
    out = PP.arms_with_hand_pose(mono, two)
    labels = list(out.labels)
    wrists = [labels.index("LeftHand"), labels.index("RightHand")]
    others = [k for k in range(len(labels)) if k not in wrists]
    np.testing.assert_array_equal(out.values[:, others], mono.channels["body"].values[:, others])
    assert np.abs(out.values[:, wrists] - mono.channels["body"].values[:, wrists]).max() > .01 and out.values.dtype == mono.channels["body"].values.dtype


def test_the_held_hand_works_through_the_real_builders():
    mono, two = built_pair()
    done = layout_adapter.conform(mono, two)
    assert PP.forearm_world(done.channels["body"], "Left").shape == (40, 3, 3)
    out = PP.arms_and_hand_pose(done, two, two)
    a, _ = v1_points(out)
    b, _ = v1_points(two)
    np.testing.assert_allclose(a[:, list(LEFT_LOCAL)], b[:, list(LEFT_LOCAL)], atol=1e-4)


def test_two_body_channels_of_other_joints_or_frames_refuse():
    mono, two, _ = real_builder_forms()
    with pytest.raises(SystemExit, match="differ in joints or frames"):
        PP.arms_with_hand_pose(mono, R.cut_frames(two, np.arange(10)))


def test_each_form_is_built_from_the_right_states(monkeypatch):
    tok = tiny_tokenizer()
    clips = []
    refs = []
    for i in range(2):
        raw = monocular_state(i)
        done = layout_adapter.conform(raw, template_state())
        clips.append(R.Clip(i, f"s{i}", raw, done, R.cut_frames(done, np.arange(*CUT)), {}))
        refs.append(two_view_state(i, "a" * 64))
    refs.append(two_view_state(2, "a" * 64))
    kept, states, masks = PP.build_form("real_two_view", refs, clips, None)
    assert kept == [0, 1, 2] and states[1] is refs[1] and masks[1] is refs[1]
    for name in ("blank", "uwb_blank"):
        kept, states, masks = PP.build_form(name, refs, clips, None)
        assert kept == [0, 1, 2] and not states[0].channels["body"].valid.any() and masks[0] is refs[0] and len(states[0].timestamps) == 20
    calls = []
    original = T.cap_form
    monkeypatch.setattr(T, "cap_form", lambda name, state, tokenizer: calls.append((name, state)) or original(name, state, tokenizer))
    for name in ("nearest_two_view", "uwb_nearest_two_view"):
        kept, states, masks = PP.build_form(name, refs, clips, tok)
        assert kept == [0, 1, 2] and calls[-1] == ("v0_nearest_fill", refs[2]) and masks[1] is refs[1]
    for name in ("nearest_monocular", "uwb_nearest_monocular"):
        kept, states, masks = PP.build_form(name, refs, clips, tok)
        assert kept == [0, 1] and calls[-1][0] == "v0_nearest_fill" and calls[-1][1] is clips[1].conformed and masks[1] is clips[1].conformed and len(states[0].timestamps) == 24
    mono, two = PP.own_values(clips[1].trimmed, tok), PP.own_values(refs[1], tok)
    expected = dict(monocular_values_two_view_marks=V.with_marks(mono, refs[1]), mono_body_two_view_hands=PP.body_and_hands(mono, two, refs[1]), two_view_body_mono_hands=PP.body_and_hands(two, mono, refs[1]), mono_arms_only=PP.arms_and_hand_pose(mono, two, refs[1]), mono_hand_pose_only=PP.arms_and_hand_pose(two, mono, refs[1]))
    for name, state in expected.items():
        kept, states, masks = PP.build_form(name, refs, clips, tok)
        assert kept == [0, 1] and masks[1] is refs[1]
        for channel in ("body", "left_hand", "right_hand"):
            np.testing.assert_array_equal(states[1].channels[channel].values, state.channels[channel].values)
            np.testing.assert_array_equal(states[1].channels[channel].valid, refs[1].channels[channel].valid)
    with pytest.raises(SystemExit, match="unknown form"):
        PP.build_form("a_generated_form", refs, clips, tok)


# ---- the smoke's list ------------------------------------------------------------------------------------------------------------

def rate_rows():
    rows = [dict(sample_id="z60", frames=61, seconds=1.), dict(sample_id="a30", frames=31, seconds=1.), dict(sample_id="n24", frames=25, seconds=1.), dict(sample_id="n2398", frames=239, seconds=238 / 23.98), dict(sample_id="unframed", seconds=1.), dict(sample_id="nosec", frames=31), dict(sample_id="c00clip", frames=31, seconds=1.), dict(sample_id="valclip", frames=31, seconds=1.)]
    return rows


def rate_store(monkeypatch, tmp_path, pool=("z60", "a30", "n24", "n2398", "unframed", "nosec")):
    store = DirectoryStore(tmp_path / "s")
    plant_shard(monkeypatch, store, rate_rows())
    records = [dict(sign=s, corpus=L.H2S, partition="train", signer_component="c02", text="t x", continuous=dict(state_key="k")) for s in pool] + [dict(sign="c00clip", corpus=L.H2S, partition="test", signer_component="c00", text="t", continuous=dict(state_key="k")), dict(sign="valclip", corpus=L.H2S, partition="val", signer_component="c01", text="t", continuous=dict(state_key="k"))]
    monkeypatch.setattr(L, "load_manifest_at", lambda s, key, sha: dict(records=records))
    return store


def test_other_rate_clips_are_c02_train_in_sign_order_and_never_near_24(monkeypatch, tmp_path):
    store = rate_store(monkeypatch, tmp_path)
    assert [r["sign"] for r in PP.other_rate_records(store, 1)] == ["a30"] and [r["sign"] for r in PP.other_rate_records(store, 2)] == ["a30", "z60"]
    with pytest.raises(SystemExit, match="2 c02 train clips at another rate, of 3 wanted"):
        PP.other_rate_records(store, 3)


def test_other_rate_rows_are_held_to_their_pins(monkeypatch, tmp_path):
    store = rate_store(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "MONOCULAR_REPORTS", {SHARD: "0" * 64})
    with pytest.raises(SystemExit, match="SHA-256"):
        PP.other_rate_records(store, 1)
    test_key = f"{R.REPORTS}test_rgb_front_clips-shard-000-of-008.json"
    spy = SpyStore({test_key: b"never"})
    monkeypatch.setattr(R, "MONOCULAR_REPORTS", {test_key: "0" * 64})
    with pytest.raises(SystemExit, match="refused unread"):
        PP.other_rate_records(spy, 1)
    assert spy.read == []


def test_an_other_rate_clip_outside_c02_train_is_refused_though_the_pool_held_it(monkeypatch, tmp_path):
    store = rate_store(monkeypatch, tmp_path)
    monkeypatch.setattr(L, "train_records", lambda manifest, components, corpus=None: [dict(sign="a30", partition="test", signer_component="c00")])
    with pytest.raises(SystemExit, match="not a train clip"):
        PP.other_rate_records(store, 1)


def test_a_smoke_adds_its_other_rate_clip_once(monkeypatch):
    records = train_records(3)
    monkeypatch.setattr(L, "open_read_list", lambda store, smoke: (records, False, dict(list="smoke", clips=3)))
    monkeypatch.setattr(PP, "other_rate_records", lambda store, k: [dict(records[0], sign="rate")] + [records[1]])
    out, declared, described = PP.open_list(None, PP.build_parser().parse_args(["read", "--out-dir", "x", "--smoke-train", "3", "--smoke-other-rate", "2"]))
    assert [r["sign"] for r in out] == ["s0", "s1", "s2", "rate"] and described == dict(list="smoke", clips=4, other_rate=2) and declared is False
    assert len(PP.open_list(None, PP.build_parser().parse_args(["read", "--out-dir", "x", "--smoke-train", "3"]))[0]) == 3


def test_a_smoke_extra_is_a_smokes_and_a_c02_train_clip(monkeypatch):
    with pytest.raises(SystemExit, match="belongs to a smoke"):
        PP.open_list(None, PP.build_parser().parse_args(["read", "--out-dir", "x", "--smoke-other-rate", "1"]))
    monkeypatch.setattr(L, "open_read_list", lambda store, smoke: (train_records(2), False, dict(list="smoke", clips=2)))
    monkeypatch.setattr(PP, "other_rate_records", lambda store, k: [dict(train_records(1)[0], sign="bad", signer_component="c00")])
    with pytest.raises(SystemExit, match="c00"):
        PP.open_list(None, PP.build_parser().parse_args(["read", "--out-dir", "x", "--smoke-train", "2", "--smoke-other-rate", "1"]))


# ---- reads ------------------------------------------------------------------------------------------------------------------------

def test_a_smoke_is_five_train_clips_at_most(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, declared=False, n=2)
    with pytest.raises(SystemExit, match="at most 5"):
        h.main("read", "--out-dir", str(h.forms), "--smoke-train", "6")


def test_a_form_read_alongside_monocular_ones_keeps_its_own_lost_list(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, declared=False, n=4, missing=(1,))
    assert h.read(smoke=True) == 0
    assert json.loads((h.forms / "form-real_two_view.json").read_text())["lost"] == [] and json.loads((h.forms / "form-uwb_blank.json").read_text())["lost"] == [] and len(json.loads((h.forms / "form-uwb_nearest_monocular.json").read_text())["lost"]) == 1


def test_each_forms_motion_is_measured_on_its_own_paths_tracked_frames(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=3)
    h.to_control()
    assert h.read("nearest_monocular") == 0
    form = json.loads((h.forms / "form-nearest_monocular.json").read_text())
    conformed = [layout_adapter.conform(s, template_state()) for s in h.mono]
    states = [T.cap_form("v0_nearest_fill", c, h.tokenizer) for c in conformed]
    assert form["motion"] == json.loads(json.dumps(L.motion_statistics(states, conformed))) and form["motion"] != json.loads(json.dumps(L.motion_statistics(states, states)))
    assert form["hands"] == json.loads(json.dumps(P.hand_statistics(states, conformed))) and form["hands"] != json.loads(json.dumps(P.hand_statistics(states, states)))


def test_each_form_is_read_by_its_own_instrument(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=3)
    h.pipeline()
    for name in PP.V1_FORMS:
        assert json.loads((h.forms / f"form-{name}.json").read_text())["instrument"]["sha256"] == "f" * 64
    for name in PP.UNFITTED_FORMS:
        read = json.loads((h.forms / f"form-{name}.json").read_text())
        assert read["instrument"]["sha256"] == PP.UNFITTED_SHA and read["instrument"]["run"] == PP.UNFITTED_RUN


@pytest.mark.parametrize("form", PP.PART_FORMS + PP.NEAREST_FORMS + PP.UNFITTED_FORMS)
def test_every_form_but_the_controls_needs_a_passed_control_before_it_loads_a_state(monkeypatch, tmp_path, form):
    h = Parts(monkeypatch, tmp_path, n=2)
    monkeypatch.setattr(L, "load_states", lambda *a: pytest.fail("a state was read"))
    with pytest.raises(SystemExit, match="no passed control"):
        h.read(form)


@pytest.mark.parametrize("change", [dict(reproduced=False), dict(code_digest="other"), dict(environment=dict(gpu="other"))])
def test_a_read_refuses_a_failed_control_or_one_of_other_code_or_environment(monkeypatch, tmp_path, change):
    h = Parts(monkeypatch, tmp_path, n=3)
    h.to_control()
    path = h.forms / "control.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), **change)))
    with pytest.raises(SystemExit, match="no passed control"):
        h.read("nearest_monocular")


def test_the_controls_read_without_a_control_and_load_no_tokenizer_or_monocular_clip(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=2)
    monkeypatch.setattr(L, "load_tokenizer", lambda *a: pytest.fail("the tokenizer"))
    monkeypatch.setattr(R, "open_monocular", lambda *a: pytest.fail("a monocular lookup"))
    monkeypatch.setattr(PP, "load_unfitted", lambda *a: pytest.fail("the unfitted instrument"))
    assert h.read(*PP.ANCHORS) == 0


def test_the_unfitted_blank_needs_no_tokenizer_or_monocular_clip(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=2)
    h.to_control()
    monkeypatch.setattr(L, "load_tokenizer", lambda *a: pytest.fail("the tokenizer"))
    monkeypatch.setattr(R, "open_monocular", lambda *a: pytest.fail("a monocular lookup"))
    assert h.read("uwb_blank") == 0


def test_an_unknown_form_refuses_and_a_read_is_write_once(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=2)
    with pytest.raises(SystemExit, match="unknown forms"):
        h.read("a9p")
    assert h.read("blank") == 0
    with pytest.raises(SystemExit, match="write-once"):
        h.read("blank")
    assert h.read("blank", resume=True) == 0


def test_a_read_writes_what_it_read(monkeypatch, tmp_path, capsys):
    h = Parts(monkeypatch, tmp_path, n=3)
    h.to_control()
    assert h.read("mono_body_two_view_hands", "nearest_monocular") == 0
    form = json.loads((h.forms / "form-mono_body_two_view_hands.json").read_text())
    assert form["kind"] == "path-parts-form" and form["report_only"] is True and form["form"] == "mono_body_two_view_hands" and form["code_digest"] == PP.code_digest() and form["tokenizer"]["sha256"] == L.TOKENIZER_SHA and form["signs"] == ["s0", "s1", "s2"] and form["groups"] == ["c01", "c03", "c05"] and form["seconds_over_reference"] == [1.] * 3
    assert form["hypotheses"][0].split() == h.records[0]["text"].split()[:11] and form["nll"][0] == [1.5] and form["lost"] == [] and set(form["part_validity"]) == set(R.PARTS) and form["part_validity"]["left_hand"][1] == R.part_validity(h.refs[1])["left_hand"] and form["part_validity"]["shoulders"][1] == R.part_validity(h.refs[1])["shoulders"] and form["part_validity"]["shoulders"][1] != form["part_validity"]["left_hand"][1]
    assert json.loads((h.forms / "form-real_two_view.json").read_text())["tokenizer"] is None
    near = json.loads((h.forms / "form-nearest_monocular.json").read_text())
    assert near["seconds_over_reference"] == [1.2] * 3 and near["valid_frame_share"] == [1.] * 3
    plan = json.loads((h.forms / "monocular.json").read_text())
    assert plan["kind"] == "path-parts-monocular" and plan["clips_loaded"] == 3 and plan["code_digest"] == PP.code_digest() and plan["conform"]["kinds"]["body:equivalent"] == 3 and plan["pairing"][1]["sign"] == "s1" and plan["pairing"][1]["validity"] == R.part_validity(layout_adapter.conform(h.mono[1], template_state())) and plan["pairing"][1]["source_hash_matches"] is True
    assert any('"monocular_clips": 3' in line and '"24.0": 3' in line for line in capsys.readouterr().out.splitlines())


def test_the_group_reaches_the_reader_and_a_clip_lost_stays_out(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, missing=(1,))
    h.to_control()
    seen = []
    original = h.reader.read
    monkeypatch.setattr(h.reader, "read", lambda states, texts, candidates, group=2: seen.append((len(states), list(texts), candidates.shape, group)) or original(states, texts, candidates, group=group))
    assert h.main("read", "--out-dir", str(h.forms), "--forms", "nearest_monocular", "--nll-group", "4") == 0
    assert seen == [(N - 1, [r["text"] for r in h.records if r["sign"] != "s1"], (N - 1, 1), 4)]
    form = json.loads((h.forms / "form-nearest_monocular.json").read_text())
    assert [x["sign"] for x in form["lost"]] == ["s1"] and "s1" not in form["signs"] and form["groups"] == [r["signer_component"] for r in h.records if r["sign"] != "s1"]
    assert json.loads((h.forms / "form-real_two_view.json").read_text())["lost"] == []


def test_the_read_checks_its_records_before_it_loads_any_state(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=3)
    h.records[1]["signer_component"] = "c00"
    monkeypatch.setattr(L, "load_states", lambda *a: pytest.fail("a state was read"))
    with pytest.raises(SystemExit, match="c00"):
        h.read("real_two_view")


def test_a_smoke_reads_every_form_on_both_instruments_without_a_control(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, declared=False, n=3)
    assert h.read(smoke=True) == 0
    assert sorted(p.name for p in h.forms.glob("form-*.json")) == sorted(f"form-{n}.json" for n in PP.FORMS)


# ---- the control ------------------------------------------------------------------------------------------------------------------

def test_the_control_passes_and_writes_what_it_read(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    assert h.read(*PP.ANCHORS) == 0
    h.plant_control()
    assert h.control() == 0
    result = json.loads((h.forms / "control.json").read_text())
    assert result["reproduced"] is True and result["code_digest"] == PP.code_digest() and result["environment"] == P.environment() and result["reads"]["real"]["identical"] == 1. and not h.store.list("pr/void")


@pytest.mark.parametrize("change,miss", [(dict(hyp_change=1), "real hypotheses"), (dict(nll_shift=.0012), "real NLL"), (dict(bleu_shift=.06), "real BLEU")])
def test_a_control_miss_exits_3_voids_the_read_and_a_read_refuses_it(monkeypatch, tmp_path, change, miss):
    h = Parts(monkeypatch, tmp_path)
    assert h.read(*PP.ANCHORS) == 0
    h.plant_control(**change)
    assert h.control() == 3
    result = json.loads((h.forms / "control.json").read_text())
    assert not result["reproduced"] and any(miss in m for m in result["misses"])
    void = json.loads(h.store.get(f"pr/void/{PP.code_digest()}.json"))
    assert void["kind"] == "path-parts-void" and void["control"]["misses"] == result["misses"] and void["code_digest"] == PP.code_digest()
    with pytest.raises(SystemExit, match="no passed control"):
        h.read("uwb_blank")


def test_a_control_of_other_clips_or_a_changed_or_shapeless_report_refuses(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    assert h.read(*PP.ANCHORS) == 0
    h.plant_control(order=True)
    with pytest.raises(SystemExit, match="other clips or another order"):
        h.control()
    h.plant_control()
    monkeypatch.setattr(P, "CANARY_REPORT_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        h.control()
    raw = json.dumps(dict(signs=["a", "b"], references=["x", "y"], forms=dict(real=dict(hypotheses=[], per_clip=dict(nll=[1.])), blank=dict(hypotheses=[], per_clip=dict(nll=[1., 2.]))))).encode()
    h.store.put(P.CANARY_REPORT, raw)
    monkeypatch.setattr(P, "CANARY_REPORT_SHA", hashlib.sha256(raw).hexdigest())
    with pytest.raises(SystemExit, match="no per-clip NLL"):
        h.control()


def test_a_smoke_control_checks_the_pin_and_says_it_compared_nothing(monkeypatch, tmp_path, capsys):
    h = Parts(monkeypatch, tmp_path, declared=False, n=3)
    assert h.read(*PP.ANCHORS, smoke=True) == 0
    h.plant_control(hyp_change=3)
    assert h.control() == 0 and not (h.forms / "control.json").exists() and "not applicable" in capsys.readouterr().out
    monkeypatch.setattr(P, "CANARY_REPORT_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        h.control()


def test_the_control_refuses_forms_of_different_runs(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    assert h.read(*PP.ANCHORS) == 0
    h.plant_control()
    path = h.forms / "form-blank.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), code_digest="other")))
    with pytest.raises(SystemExit, match="different code"):
        h.control()


# ---- question 1 -------------------------------------------------------------------------------------------------------------------

def test_the_parts_block_has_its_planted_values(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    report = h.pipeline()
    block = report["which_part"]
    likelihood = block["likelihood"]
    assert likelihood["cells"]["real_two_view"] == 2. and likelihood["cells"]["monocular_values_two_view_marks"] == pytest.approx(1.4) and likelihood["lost_by_swapping_both"]["point"] == pytest.approx(.6)
    assert likelihood["lost"]["body"]["point"] == pytest.approx(.5) and likelihood["lost"]["hands"]["point"] == pytest.approx(.2) and likelihood["lost"]["arms"]["point"] == pytest.approx(.4) and likelihood["lost"]["hand_pose"]["point"] == pytest.approx(.1)
    assert likelihood["share"]["body"]["point"] == pytest.approx(.8333, abs=1e-3) and likelihood["share"]["hands"]["point"] == pytest.approx(.3333, abs=1e-3) and likelihood["share"]["arms"]["point"] == pytest.approx(.6667, abs=1e-3) and likelihood["share"]["hand_pose"]["point"] == pytest.approx(.1667, abs=1e-3)
    assert likelihood["case"] == "body" and likelihood["view_case_report_only"] == "arms" and block["bootstrap"] == dict(n=2000, seed=0) and block["clips"] == N


@pytest.mark.parametrize("total,first,second,case", [
    (.1, (.67, .34), (.34, .1), "both"), (.1, (.67, .34), (.33, .1), "body"), (.1, (.67, .34), (.32, .1), "body"), (.1, (.66, .34), (.2, .1), "unresolved"), (.1, (.67, .32), (.2, .1), "unresolved"),
    (.1, (.32, .1), (.67, .34), "hands"), (.1, (.2, .1), (.66, .4), "unresolved"), (.1, (.2, .1), (.7, .3), "unresolved"),
    (.1, (.4, .2), (.5, .3), "both"), (.1, (.34, .1), (.34, .1), "both"), (.1, (.33, .1), (.5, .3), "unresolved"), (.1, (.2, .1), (.2, .1), "unresolved"),
    (0., (.8, .5), (.1, 0.), "unresolved"), (-.1, (.8, .5), (.1, 0.), "unresolved")])
def test_the_part_case_of_one_metric(total, first, second, case):
    row_ = lambda v: dict(point=v[0], ci95=[v[1], v[0] + .2])           # noqa: E731
    assert PP.part_case(dict(point=.5, ci95=[total, 1.]), row_(first), row_(second), ("body", "hands")) == case


def test_no_share_is_unresolved():
    assert PP.part_case(dict(point=.5, ci95=[.1, 1.]), None, dict(point=.1, ci95=[0., .2]), ("a", "b")) == "unresolved" and PP.part_case(dict(point=.5, ci95=[.1, 1.]), dict(point=.8, ci95=[.5, 1.]), None, ("a", "b")) == "unresolved"


def test_the_metrics_disagreeing_reads_unresolved():
    assert PP.joint(["body", "body"]) == "body" and PP.joint(["body", "hands"]) == "unresolved" and PP.joint(["both", "unresolved"]) == "unresolved"
    assert (PP.FIT_ONE_THIRD, PP.FIT_TWO_THIRDS) == (1 / 3, 2 / 3)


def test_the_label_needs_both_metrics_to_agree(monkeypatch, tmp_path):
    """BLEU's cells are planted against likelihood's so the two metrics differ."""
    h = Parts(monkeypatch, tmp_path, part_levels=dict(mono_body=(15, 1.5), mono_hands=(10, 1.2)))
    report = h.pipeline()
    assert report["which_part"]["bleu"]["case"] != report["which_part"]["likelihood"]["case"] and report["which_part"]["case"] == "unresolved"


def test_the_view_split_needs_both_metrics_to_agree_too(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, part_levels=dict(arms=(15, 1.4), hand_pose=(10, 1.1)))
    block = h.pipeline()["which_part"]
    assert block["likelihood"]["view_case_report_only"] == "arms" and block["bleu"]["view_case_report_only"] == "hand_pose" and block["view_case_report_only"] == "unresolved"


def test_a_ratio_with_no_denominator_is_none():
    assert PP.ratio(dict(a=1., b=1., c=2.), dict(a=np.ones(3), b=np.ones(3), c=np.ones(3)), ("c", "a"), ("a", "b")) is None
    out = PP.ratio(dict(a=2., b=1., c=4.), dict(a=np.array([2., 4.]), b=np.array([1., 2.]), c=np.array([4., 6.])), ("c", "b"), ("a", "b"))
    assert out["point"] == 3. and out["ci95"] == P._quantiles(np.array([3., 2.]))


# ---- question 2 -------------------------------------------------------------------------------------------------------------------

def test_the_fit_block_uses_each_instruments_own_blank(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, unfitted=UnfittedReader(two=(6, 8.), mono=(5, 8.5), blank=(0, 9.)))
    report = h.pipeline()
    block = report["fit_or_content"]["likelihood"]
    assert block["unfitted"]["own_gap"]["point"] == 1. and block["unfitted"]["monocular_minus_two_view"]["point"] == -.5 and block["unfitted"]["monocular_share_of_own_gap"]["point"] == .5 and block["unfitted"]["cells"]["uwb_blank"] == 0.
    assert block["v1"]["own_gap"]["point"] == 1.5 and block["v1"]["monocular_minus_two_view"]["point"] == pytest.approx(-.7) and block["v1"]["monocular_share_of_own_gap"]["point"] == pytest.approx(.5333, abs=1e-3)
    assert block["share_difference"] == pytest.approx(.5 - .5333, abs=1e-3)
    assert report["fit_or_content"]["thresholds"] == dict(comparable=.25, kept=.75, readable_width=.5)
    assert report["forms"]["uwb_nearest_monocular"]["likelihood_minus_blank"]["mean"] == .5 and report["forms"]["uwb_nearest_two_view"]["likelihood_minus_blank"]["mean"] == 1. and report["forms"]["nearest_two_view"]["likelihood_minus_blank"]["mean"] == 1.5


def row(point, low, high):
    return dict(point=point, ci95=[low, high])


GAP, DROP = row(.2, .1, .3), row(-.1, -.15, -.05)


@pytest.mark.parametrize("gap,share_u,drop,share_v1,case", [
    (GAP, row(.3, .1, .5), DROP, row(.18, .1, .3), "content"), (GAP, row(.43, .3, .75), DROP, row(.18, .1, .3), "content"), (GAP, row(.44, .3, .75), DROP, row(.18, .1, .3), "unresolved"),
    (GAP, row(.5, .3, .76), DROP, row(.4, .3, .5), "unresolved"), (GAP, row(.3, .1, .5), row(-.1, -.2, 0.), row(.18, .1, .3), "unresolved"), (GAP, row(.3, .1, .5), row(-.1, -.2, .01), row(.18, .1, .3), "unresolved"),
    (GAP, row(.9, .75, 1.1), DROP, row(.18, .1, .3), "fit"), (GAP, row(1.2, 1., 1.4), row(.04, .01, .07), row(.18, .1, .3), "fit"), (GAP, row(.9, .74, 1.1), DROP, row(.18, .1, .3), "unresolved"),
    (row(.2, 0., .3), row(.3, .1, .5), DROP, row(.18, .1, .3), "unresolved"), (row(.2, -.1, .3), row(.9, .8, 1.), DROP, row(.18, .1, .3), "unresolved"), (GAP, row(.3, 0., .51), DROP, row(.18, .1, .3), "unresolved"), (GAP, row(.3, 0., .5), DROP, row(.18, .1, .3), "content"),
    (GAP, row(.1, 0., .3), DROP, row(.5, .4, .6), "unresolved"), (GAP, row(.25, .1, .4), DROP, row(.5, .4, .6), "content"), (GAP, None, DROP, row(.18, .1, .3), "unresolved"), (GAP, row(.3, .1, .5), DROP, None, "unresolved")])
def test_the_fit_case_at_its_thresholds(gap, share_u, drop, share_v1, case):
    assert PP.fit_case(gap, share_u, drop, share_v1) == case


def test_the_label_is_likelihoods_and_bleu_is_reported_beside_it(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    report = h.pipeline()
    block = report["fit_or_content"]
    assert block["case"] == block["likelihood"]["case"] and "case" in block["bleu"] and block["bleu"]["unfitted"]["own_gap"]["point"] > 0


def test_a_gap_that_survives_the_unfitted_instrument_reads_as_fit(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, unfitted=UnfittedReader(two=(6, 2.), mono=(6, 2.), blank=(0, 3.)))
    report = h.pipeline()
    assert report["fit_or_content"]["likelihood"]["unfitted"]["monocular_share_of_own_gap"]["point"] == 1. and report["fit_or_content"]["case"] == "fit" and report["fit_or_content"]["label"] == "fit"


def test_a_gap_lost_alike_reads_as_content(monkeypatch, tmp_path):
    """v1 keeps 0.533 of its gap; this instrument keeps 0.5 of its own."""
    h = Parts(monkeypatch, tmp_path)
    report = h.pipeline()
    block = report["fit_or_content"]["likelihood"]
    assert block["unfitted"]["monocular_minus_two_view"]["ci95"] == [-.5, -.5] and block["case"] == "content" and report["fit_or_content"]["label"] == "content"


def test_an_instrument_with_no_gap_of_its_own_is_unresolved(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, unfitted=UnfittedReader(two=(6, 3.), mono=(5, 3.), blank=(0, 3.)))
    report = h.pipeline()
    assert report["fit_or_content"]["likelihood"]["unfitted"]["monocular_share_of_own_gap"] is None and report["fit_or_content"]["case"] == "unresolved"


# ---- intervals as wide as D20's ---------------------------------------------------------------------------------------------------

def d20_like(n=500, seed=0, shift_mono=-.1, gap=.2, noise=.35):
    """500 sentences with rare short matches, NLL with clip spread: an instrument with a small real gap, as a floor reader may show."""
    rng = np.random.default_rng(seed)
    refs = [" ".join(f"r{rng.integers(3000)}" for _ in range(14)) for _ in range(n)]

    def hypotheses(p):
        out = []
        for ref in refs:
            words, hyp = ref.split(), [f"h{rng.integers(3000)}" for _ in range(13)]
            if rng.random() < p:
                k = int(rng.integers(0, 10))
                hyp[k:k + 4] = words[k:k + 4]
            out.append(" ".join(hyp))
        return out
    base = rng.normal(size=n) * noise
    blank = dict(references=refs, hypotheses=hypotheses(.1), nll=np.c_[5. + base + rng.normal(size=n) * .05].tolist())
    two = dict(references=refs, hypotheses=hypotheses(.4), nll=np.c_[5. - gap + base + rng.normal(size=n) * .05].tolist())
    mono = dict(references=refs, hypotheses=hypotheses(.2), nll=np.c_[5. - gap - shift_mono + base + rng.normal(size=n) * .05].tolist())
    return blank, two, mono


def fit_rows(blank, two, mono):
    point, draws = PP.share_rows(dict(uwb_blank=blank, two=two, mono=mono), "uwb_blank")
    p, d = point["likelihood"], draws["likelihood"]
    return PP.difference(p, d, "two", "uwb_blank"), PP.ratio(p, d, ("mono", "uwb_blank"), ("two", "uwb_blank")), PP.difference(p, d, "mono", "two")


def test_at_500_clips_a_real_gap_with_half_lost_reads_content_with_tight_shares():
    gap, share, drop = fit_rows(*d20_like())
    assert gap["ci95"][0] > 0 and share["ci95"][1] - share["ci95"][0] < .5 and drop["ci95"][1] < 0
    assert PP.fit_case(gap, share, drop, dict(point=.4, ci95=[.2, .6])) == "content"


def test_at_500_clips_a_gap_near_the_floor_is_too_small_to_read():
    """A gap this instrument barely shows: the share's interval is too wide to tell a kept gap from a lost one, whatever its point."""
    gap, share, drop = fit_rows(*d20_like(gap=.012, shift_mono=-.006, noise=.35))
    assert share["ci95"][1] - share["ci95"][0] > .5 and PP.fit_case(gap, share, drop, dict(point=.18, ci95=[.1, .3])) == "unresolved"


def test_at_500_clips_a_gap_kept_whole_reads_fit():
    gap, share, drop = fit_rows(*d20_like(shift_mono=0.))
    assert gap["ci95"][0] > 0 and share["ci95"][0] >= .75 and PP.fit_case(gap, share, drop, dict(point=.18, ci95=[.1, .3])) == "fit"


def test_at_500_clips_bleu_intervals_are_as_wide_as_d20s():
    blank, two, _ = d20_like()
    bleu, ci = M.bleu_ci_fast(two["hypotheses"], two["references"])
    assert 1. < bleu < 12. and .3 < ci[1] - ci[0] < 3.5


def test_a_reading_is_taken_jointly_over_clips():
    blank, two, mono = d20_like(n=60)
    point, draws = PP.share_rows(dict(uwb_blank=blank, two=two, mono=mono), "uwb_blank")
    idx = np.random.default_rng(0).integers(0, 60, (PP.BOOT, 60))
    base = np.asarray(blank["nll"])[:, 0]
    a = (base - np.asarray(two["nll"])[:, 0])[idx].mean(1)
    np.testing.assert_allclose(draws["likelihood"]["two"], a)
    s = M.bleu_statistics(two["hypotheses"], two["references"])
    np.testing.assert_allclose(draws["bleu"]["two"], M.bleu_from_statistics(s[idx].sum(1)))
    assert point["likelihood"]["uwb_blank"] == 0. and point["bleu"]["two"] == pytest.approx(float(M.bleu_from_statistics(s.sum(0))))


# ---- alerts and labels ------------------------------------------------------------------------------------------------------------

def test_figures_off_the_first_read_raise_an_alert_and_withhold_both_labels(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    report = h.pipeline(figures=False)
    assert PP.FIGURES["nearest_two_view"] == (4.621, .234) and any("nearest_two_view reads BLEU" in a for a in report["alerts"]) and any("nearest_monocular reads likelihood" in a for a in report["alerts"]) and any("monocular_values_two_view_marks reads BLEU" in a for a in report["alerts"])
    assert report["which_part"]["label"] is None and report["fit_or_content"]["label"] is None and report["withheld"] == report["alerts"] and report["which_part"]["case"] is not None


@pytest.mark.parametrize("bleu,likelihood,count", [(.049, .0019, 0), (.051, 0., 1), (0., .0021, 1), (.051, .0021, 2)])
def test_figures_are_held_to_their_tolerances(monkeypatch, bleu, likelihood, count):
    refs = [r["text"] for r in d20_records(10)]
    blank = dict(signs=[f"s{i}" for i in range(10)], nll=[[3.]] * 10, references=refs)
    form = dict(signs=blank["signs"], references=refs, hypotheses=[" ".join(t.split()[:6]) for t in refs], nll=[[2.]] * 10)
    here = M.bleu_ci_fast(form["hypotheses"], refs)[0]
    monkeypatch.setattr(PP, "FIGURES", dict(nearest_two_view=(here + bleu, 1. + likelihood)))
    assert len(PP.figure_misses(dict(blank=blank, nearest_two_view=form))) == count
    assert (PP.BLEU_TOLERANCE, PP.LIKELIHOOD_TOLERANCE) == (.05, .002)


def test_figures_are_the_first_reads():
    assert PP.FIGURES == dict(nearest_two_view=(4.621, .234), nearest_monocular=(2.653, .041), monocular_values_two_view_marks=(2.546, .097))


def test_an_unfitted_form_reading_as_its_blank_raises_an_alert(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, unfitted=UnfittedReader(two=(0, 3.), mono=(5, 2.5), blank=(0, 3.)))
    report = h.pipeline()
    assert "uwb_nearest_two_view reads exactly as uwb_blank" in report["alerts"] and "uwb_nearest_monocular reads exactly as uwb_blank" not in report["alerts"] and report["fit_or_content"]["label"] is None
    assert PP.main(["check", "--report", str(tmp_path / "report.json")]) == 3


def test_a_form_reads_as_blank_only_when_every_clip_does():
    def form(*hypotheses):
        return dict(signs=["a", "b"], hypotheses=list(hypotheses))
    assert PP.constant_reads(dict(uwb_blank=form("x", "y"), uwb_nearest_two_view=form("x", "z"), uwb_nearest_monocular=form("q", "y"))) == []
    assert PP.constant_reads(dict(uwb_blank=form("x", "y"), uwb_nearest_two_view=form("q", "z"), uwb_nearest_monocular=form("x", "y"))) == ["uwb_nearest_monocular reads exactly as uwb_blank"]
    assert PP.constant_reads(dict(uwb_blank=form("x", "y"), uwb_nearest_two_view=form("x", "y"), uwb_nearest_monocular=form("q", "z"))) == ["uwb_nearest_two_view reads exactly as uwb_blank"]


def test_an_emptied_monocular_hand_alerts_through_the_records_validity(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, mono=lambda i: with_empty_hand(monocular_state(i)))
    report = h.pipeline()
    assert any("monocular left_hand valid on 0.000" in a for a in report["alerts"]) and not any("two-view left_hand" in a for a in report["alerts"]) and report["which_part"]["label"] is None


def test_a_source_hash_that_differs_withholds_the_labels(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    h.refs[0] = two_view_state(0, "b" * 64)
    h.plant_two_view()
    h.reader.truth[h.records[0]["text"]] = (h.refs[0], h.mono[0])
    report = h.pipeline()
    assert any("do not name their monocular row's hash" in a for a in report["alerts"]) and report["fit_or_content"]["label"] is None and report["pairing"]["source_hash_matches"] == N - 1


def test_under_the_floor_no_label_is_read(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, floor=False)
    report = h.pipeline()
    assert any(f"{N} shared clips, under 475" in a for a in report["alerts"]) and report["which_part"]["label"] is None and report["fit_or_content"]["label"] is None


def test_a_clean_read_carries_both_labels_and_passes_check(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    report = h.pipeline()
    assert report["alerts"] == [] and report["withheld"] == [] and report["which_part"]["label"] == report["which_part"]["case"] and report["fit_or_content"]["label"] == report["fit_or_content"]["case"]
    assert PP.main(["check", "--report", str(tmp_path / "report.json")]) == 0


def test_a_smoke_carries_no_label_and_reports_its_rows(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, declared=False, n=4)
    assert h.read(smoke=True) == 0
    h.plant_control()
    assert h.control() == 0
    report = h.merge()
    assert report["declared_read"] is False and report["smoke"].startswith("SMOKE") and report["withheld"] == ["a smoke carries no reading"] and report["which_part"]["label"] is None and report["fit_or_content"]["label"] is None and report["alerts"] == [] and report["checks"]["control"] is None and report["which_part"]["likelihood"]["share"]["body"]["point"] > 0
    assert PP.main(["check", "--report", str(tmp_path / "report.json")]) == 0


def test_the_report_names_its_list_code_environment_instruments_and_rows(monkeypatch, tmp_path, capsys):
    h = Parts(monkeypatch, tmp_path, missing=(1,))
    report = h.pipeline()
    assert report["kind"] == "path-parts-report" and report["report_only"] is True and report["note"] == PP.NOT_DECIDED and report["code_digest"] == PP.code_digest() and report["environment"] == P.environment() and report["list"]["list"] == "t" and report["disclosures"] == json.loads(json.dumps(PP.DISCLOSURES)) and report["prestated"] == PP.PRESTATED and report["instruments"]["unfitted"]["sha256"] == PP.UNFITTED_SHA and report["rerun"] is None
    assert report["clips"] == dict(list=N, shared=N - 1, floor=1, outside_shared=dict(count=1, by_reason=dict(no_monocular_row=1), clips=report["clips"]["outside_shared"]["clips"])) and report["clips"]["outside_shared"]["clips"][0]["sign"] == "s1" and report["pairing"]["clips"] == N - 1 and report["conform"]["kinds"]["body:equivalent"] == N - 1 and report["checks"]["control"]["reproduced"] is True and report["checks"]["blank_identical_to_real_two_view"] == 0.
    row = report["forms"]["uwb_nearest_monocular"]
    assert row["instrument"] == PP.UNFITTED_RUN and row["clips"] == N - 1 and row["likelihood_minus_blank"]["mean"] == .5 and row["seconds_over_reference"] == 1.2 and row["valid_frame_share"]["mean"] == 1. and report["forms"]["nearest_monocular"]["likelihood_minus_blank"]["mean"] == pytest.approx(.8) and report["forms"]["real_two_view"]["part_validity"]["left_hand"] < 1
    assert set(report["resources"]) == set(PP.FORMS) and report["pairing"]["offsets"]["max"] == 0.
    signers = report["per_signer"]
    assert sorted(signers) == ["c01", "c03", "c05"] and sum(s["clips"] for s in signers.values()) == N - 1 and signers["c03"]["which_part"]["likelihood"]["body"]["point"] == pytest.approx(.8333, abs=1e-3) and signers["c03"]["fit_or_content"]["likelihood"]["own_gap"]["point"] == 1. and signers["c03"]["fit_or_content"]["v1_share"]["point"] == pytest.approx(.5333, abs=1e-3)
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["shared"] == N - 1 and printed["of"] == N and printed["lost"] == dict(no_monocular_row=1) and printed["fit_or_content"]["label"] == report["fit_or_content"]["label"] and printed["which_part"]["case"] == report["which_part"]["case"] and printed["forms"]["real_two_view"]["bleu"] == report["forms"]["real_two_view"]["bleu"] and printed["withheld"] == [] and printed["pairing"]["offsets"]["clips"] == N - 1


def test_a_merge_refuses_what_is_not_one_run(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=4)
    h.pipeline()
    path = h.forms / "form-uwb_blank.json"
    good = path.read_text()

    def expect(change, match):
        path.write_text(json.dumps(dict(json.loads(good), **change)))
        with pytest.raises(SystemExit, match=match):
            h.merge("again.json")
    expect(dict(code_digest="other"), "different code")
    expect(dict(environment=dict(gpu="other")), "different environments")
    expect(dict(form="other"), "holds form")
    expect(dict(declared_read=False), "smoke mixed")
    expect(dict(list=dict(list="other")), "another list")
    expect(dict(instrument=dict(sha256="f" * 64)), "its own instrument")
    expect(dict(instrument=dict(sha256="0" * 64)), "its own instrument")
    path.write_text(good)
    other = h.forms / "form-nearest_two_view.json"
    kept = other.read_text()
    other.write_text(json.dumps(dict(json.loads(kept), instrument=dict(sha256="0" * 64))))
    with pytest.raises(SystemExit, match="its own instrument"):
        h.merge("again.json")
    other.write_text(kept)
    stored = {n: (h.forms / f"form-{n}.json").read_text() for n in PP.UNFITTED_FORMS}
    for n, text in stored.items():
        (h.forms / f"form-{n}.json").write_text(json.dumps(dict(json.loads(text), instrument=dict(sha256="f" * 64))))
    with pytest.raises(SystemExit, match="its own instrument"):
        h.merge("again.json")
    for n, text in stored.items():
        (h.forms / f"form-{n}.json").write_text(text)
    path.unlink()
    with pytest.raises(SystemExit, match="form-uwb_blank.json is missing"):
        h.merge("again.json")
    path.write_text(good)
    assert h.merge("again.json")["clips"]["shared"] == 4


def test_a_merge_refuses_a_failed_control_and_another_runs_record(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=3)
    h.pipeline()
    path = h.forms / "control.json"
    good = json.loads(path.read_text())
    path.write_text(json.dumps(dict(good, reproduced=False)))
    with pytest.raises(SystemExit, match="no passed control"):
        h.merge("again.json")
    path.unlink()
    with pytest.raises(SystemExit, match="no passed control"):
        h.merge("again.json")
    path.write_text(json.dumps(good))
    plan = h.forms / "monocular.json"
    kept = json.loads(plan.read_text())
    for change in (dict(code_digest="other"), dict(declared_read=False)):
        plan.write_text(json.dumps(dict(kept, **change)))
        with pytest.raises(SystemExit, match="another run's record"):
            h.merge("again.json")
    plan.unlink()
    with pytest.raises(SystemExit, match="monocular.json is missing"):
        h.merge("again.json")


def test_a_merge_with_no_shared_clip_refuses(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=3)
    h.pipeline()
    for name, pick in (("nearest_monocular", [0]), ("uwb_nearest_monocular", [1])):
        path = h.forms / f"form-{name}.json"
        form = json.loads(path.read_text())
        form.update(signs=[form["signs"][i] for i in pick], references=[form["references"][i] for i in pick], lost=[])
        path.write_text(json.dumps(form))
    with pytest.raises(SystemExit, match="share no clip"):
        h.merge("again.json")


def test_a_merge_is_write_once_and_refuses_before_it_loads_anything(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=3)
    taken = tmp_path / "taken.json"
    taken.write_text("{}")
    monkeypatch.setattr(PP, "load_reads", lambda *a: pytest.fail("the forms were loaded"))
    with pytest.raises(SystemExit, match="write-once"):
        h.main("merge", "--out-dir", str(h.forms), "--output-file", str(taken))


def test_a_merge_with_instruments_not_as_stated_stops(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=3)
    h.to_control()
    assert h.read(*PP.PART_FORMS, *PP.NEAREST_FORMS, *PP.UNFITTED_FORMS, resume=True) == 0
    h.plant_figures()
    plant_instruments(h.store, **{"v1.args.init_sha256": "0" * 64})
    with pytest.raises(SystemExit, match="not fine-tuned from this"):
        h.merge()


def test_the_pairing_counts_only_shared_clips(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    h.pipeline()
    path = h.forms / "form-uwb_nearest_monocular.json"
    form = json.loads(path.read_text())
    keep = list(range(1, N))
    for key in ("signs", "groups", "references", "hypotheses", "nll", "seconds_over_reference", "valid_frame_share"):
        form[key] = [form[key][i] for i in keep]
    form["part_validity"] = {k: [v[i] for i in keep] for k, v in form["part_validity"].items()}
    form["motion"] = {k: [v[i] for i in keep] for k, v in form["motion"].items()}
    form["hands"] = {x: {k: [v[i] for i in keep] for k, v in row_.items()} for x, row_ in form["hands"].items()}
    form["tracked"] = {k: [v[i] for i in keep] for k, v in form["tracked"].items()}
    form["lost"] = [dict(sign="s0", reason="planted")]
    path.write_text(json.dumps(form))
    report = h.merge("again.json")
    assert report["clips"]["shared"] == N - 1 and report["pairing"]["clips"] == N - 1 and report["forms"]["real_two_view"]["clips"] == N - 1


@pytest.mark.parametrize("kw,code", [(dict(report_only=True, alerts=[], declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3), which_part=dict(label=None), fit_or_content=dict(label=None)), 0), (dict(report_only=True, alerts=[], declared_read=True, checks=dict(blank_identical_to_real_two_view=0., control=dict(reproduced=True)), clips=dict(shared=3), which_part=dict(label="body"), fit_or_content=dict(label="fit")), 0),
                                     (dict(report_only=False, alerts=[], declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3), which_part=dict(), fit_or_content=dict()), 3), (dict(report_only=True, alerts=[], declared_read=False, checks=dict(blank_identical_to_real_two_view=1.), clips=dict(shared=3), which_part=dict(), fit_or_content=dict()), 3),
                                     (dict(report_only=True, alerts=[], declared_read=True, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3), which_part=dict(), fit_or_content=dict()), 3), (dict(report_only=True, alerts=[], declared_read=True, checks=dict(blank_identical_to_real_two_view=0., control=dict(reproduced=False)), clips=dict(shared=3), which_part=dict(), fit_or_content=dict()), 3),
                                     (dict(report_only=True, alerts=[], declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=0), which_part=dict(), fit_or_content=dict()), 3), (dict(report_only=True, alerts=["x"], declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3), which_part=dict(), fit_or_content=dict()), 3)])
def test_check_judges_plumbing_only(tmp_path, kw, code):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(kw))
    assert PP.main(["check", "--report", str(path)]) == code


@pytest.mark.parametrize("which", ["which_part", "fit_or_content"])
def test_check_calls_out_a_label_beside_an_alert(tmp_path, capsys, which):
    body = dict(report_only=True, alerts=["x"], declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3), which_part=dict(label=None), fit_or_content=dict(label=None))
    body[which] = dict(label="y")
    path = tmp_path / "r.json"
    path.write_text(json.dumps(body))
    assert PP.main(["check", "--report", str(path)]) == 3
    assert "ALERT a report holding a label beside an alert" in capsys.readouterr().out


def test_preflight_refuses_a_finished_or_voided_read_and_a_rerun_needs_a_cause(monkeypatch, tmp_path, capsys):
    h = Parts(monkeypatch, tmp_path, n=2)
    args = ("preflight", "--out-dir", str(tmp_path / "d"), "--final-key", "pr/report.json", "--void-prefix", "pr/void")
    assert h.main(*args) == 0 and PP.code_digest() in capsys.readouterr().out
    h.store.put("pr/void/other.json", b"{}")
    with pytest.raises(SystemExit, match="rerun-cause"):
        h.main(*args)
    assert h.main(*args, "--rerun-cause", "why") == 0 and json.loads((tmp_path / "d/forms/rerun.json").read_text())["cause"] == "why"
    h.store.put(f"pr/void/{PP.code_digest()}.json", b"{}")
    with pytest.raises(SystemExit, match="void"):
        h.main(*args, "--rerun-cause", "why")
    h.store.put("pr/report.json", b"{}")
    with pytest.raises(SystemExit, match="finished"):
        h.main(*args, "--rerun-cause", "why")


def test_a_rerun_record_reaches_the_report(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=3)
    h.forms.mkdir(parents=True)
    (h.forms / "rerun.json").write_text(json.dumps(dict(cause="why")))
    assert h.pipeline()["rerun"] == dict(cause="why")


# ---- the header -------------------------------------------------------------------------------------------------------------------

def test_the_readings_are_stated_in_the_header_before_any_code():
    assert all(text in PP.__doc__ for text in PP.PRESTATED.values()) and set(PP.PRESTATED) == {"body", "both", "readable", "content", "fit", "neutral"}
    doc = PP.__doc__
    for text in ("report only", "no step branches on it", "no claim about composition", "likelihood is read first", "It is also v1's own pre-fine-tune weights", "323,978 UWB YouTube-ASL clips", "the same 144 features", "wrist' = inv(forearm') @ forearm @ wrist", "at most 0.25", "wider than 0.5", "below 475 shared clips", "Can its input see the difference", "kp2d-h2s-v1"):
        assert text.lower() in doc.lower(), text
    assert (PP.COMPARABLE, PP.KEPT, PP.READABLE_WIDTH) == (.25, .75, .5)


def test_the_constants_are_the_declared_ones():
    assert PP.PREFIX == "acquisition/2026-10-17-v1/youtube-asl-v1/round20/path-read-parts" and PP.BOOT == 2000 and PP.JOB_FILE == "runpod/fleet/path_parts.sh" and PP.ANCHORS == ("real_two_view", "blank")
    assert PP.FORMS == ("real_two_view", "blank", "monocular_values_two_view_marks", "mono_body_two_view_hands", "two_view_body_mono_hands", "mono_arms_only", "mono_hand_pose_only", "nearest_two_view", "nearest_monocular", "uwb_blank", "uwb_nearest_two_view", "uwb_nearest_monocular")
    assert set(PP.MONOCULAR_NEEDED) == {"monocular_values_two_view_marks", "mono_body_two_view_hands", "two_view_body_mono_hands", "mono_arms_only", "mono_hand_pose_only", "nearest_monocular", "uwb_nearest_monocular"} and set(PP.TOKENIZER_NEEDED) == set(PP.FORMS) - {"real_two_view", "blank", "uwb_blank"}


def test_the_code_digest_holds_the_job_file():
    assert PP.code_digest() == provenance.code_digest(extra_files=(PP.JOB_FILE,)) and PP.code_digest() != provenance.code_digest()


def test_a_digest_that_cannot_be_named_voids(monkeypatch):
    PP.code_digest.cache_clear()
    monkeypatch.setattr(provenance, "code_digest", lambda extra_files=(): (_ for _ in ()).throw(OSError("gone")))
    with pytest.raises(SystemExit, match="no code digest"):
        PP.code_digest()
    PP.code_digest.cache_clear()


# ---- the job ----------------------------------------------------------------------------------------------------------------------

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
    lines = [x for x in result.stdout.splitlines() if x.startswith("PATH-PARTS-DONE ")]
    assert len(lines) == 1 and lines[0] == result.stdout.strip().splitlines()[-1]
    return int(lines[0].split()[1])


def test_the_runner_sees_no_credentials():
    assert not [k for k in os.environ if k.startswith(("R2_", "AWS_", "RUNPOD_"))]


def test_the_job_checks_instruments_smokes_then_reads_the_control_first_and_the_unfitted_form_last(tmp_path):
    result, calls = run_job(tmp_path)
    assert result.returncode == 0 and done_line(result) == 0
    own = calls.splitlines()
    parsed = [PP.build_parser().parse_args(x.split()[1:]) for x in own]
    assert all(x.startswith("scripts/path_parts.py ") for x in own)
    assert [a.command for a in parsed] == ["instruments", "read", "control", "merge", "upload", "check", "preflight", "read", "control", "read", "read", "merge", "upload", "check", "upload-log"]
    reads = [a for a in parsed if a.command == "read"]
    assert (reads[0].smoke_train, reads[0].smoke_longest, reads[0].smoke_never_tracked, reads[0].smoke_other_rate) == (5, 4, 2, 1) and reads[0].forms == ",".join(PP.FORMS)
    assert reads[1].forms == ",".join(PP.ANCHORS) and reads[2].forms == ",".join(PP.PART_FORMS + PP.NEAREST_FORMS) and reads[3].forms == ",".join(PP.UNFITTED_FORMS) and all(a.resume and a.smoke_train == 0 and a.smoke_other_rate == 0 for a in reads[1:])
    assert all(a.device == "cuda" and a.nll_group == 4 and a.threads == 8 and a.instrument_cache.endswith("/instrument") for a in reads)
    uploads = [a for a in parsed if a.command == "upload"]
    assert uploads[0].prefix.startswith(f"{PP.PREFIX}/smoke/") and uploads[1].prefix == PP.PREFIX and uploads[1].out_dir.endswith("/d20")
    pre = next(a for a in parsed if a.command == "preflight")
    assert pre.final_key == f"{PP.PREFIX}/report.json" and pre.void_prefix == f"{PP.PREFIX}/void" and next(a for a in parsed if a.command == "control").void_prefix == f"{PP.PREFIX}/void"
    assert parsed[-1].key.startswith(f"{PP.PREFIX}/logs/") and parsed[-1].key.endswith(".log") and parsed[-1].file.endswith("/path_parts.log")


def test_smoke_only_stops_before_any_declared_read(tmp_path):
    result, calls = run_job(tmp_path, SMOKE="1")
    assert result.returncode == 0 and done_line(result) == 0 and "SMOKE done" in result.stdout and "preflight" not in calls and "/d20/forms" not in calls and "--smoke-other-rate 1" in calls and "SMOKE done" in (tmp_path / "work" / "path_parts.log").read_text() and "upload-log" in calls.splitlines()[-1]


@pytest.mark.parametrize("fail_when,expected", [("path_parts.py instruments", 1), ("path_parts.py read*smoke-other", 2), ("path_parts.py control*smoke-", 2), ("path_parts.py merge*smoke-", 2), ("path_parts.py check --report*smoke-", 2), ("path_parts.py upload --out-dir*smoke-", 2),
                                                ("path_parts.py preflight", 1), ("--forms real_two_view", 1), ("--forms monocular_values", 1), ("--forms uwb_blank", 1), ("path_parts.py merge*/work/d20", 1), ("path_parts.py upload --out-dir*/work/d20", 1), ("path_parts.py check --report*/work/d20", 1)])
def test_any_failure_exits_nonzero_with_its_done_line_through_the_pipe_uploads_its_work_and_its_log(tmp_path, fail_when, expected):
    result, calls = run_job(tmp_path, fail_when=fail_when)
    assert result.returncode == expected and done_line(result) == expected, result.stdout + result.stderr
    lines = calls.splitlines()
    assert "upload --partial" in lines[-2] and f"{PP.PREFIX}/partial/" in lines[-2] and "upload-log" in lines[-1] and "ALERT" in result.stdout and "planted failure" in result.stdout and "ALERT" in (tmp_path / "work" / "path_parts.log").read_text()


def test_instruments_that_fail_stop_before_the_smoke(tmp_path):
    _, calls = run_job(tmp_path, fail_when="path_parts.py instruments")
    assert "path_parts.py read" not in calls


def test_a_smoke_failure_reads_nothing_of_d20(tmp_path):
    _, calls = run_job(tmp_path, fail_when="path_parts.py read*smoke-other")
    assert "preflight" not in calls and "--forms" not in calls


def test_a_smoke_alert_stops_the_job_with_its_report_stored(tmp_path):
    result, calls = run_job(tmp_path, fail_when="path_parts.py check --report*smoke-", code=3)
    assert result.returncode == 2 and done_line(result) == 2 and "preflight" not in calls and calls.index("path_parts.py upload --out-dir") < calls.index("path_parts.py check --report") and "its report is stored" in result.stdout


def test_a_control_miss_stops_before_any_other_form_with_its_own_status(tmp_path):
    result, calls = run_job(tmp_path, fail_when="path_parts.py control*/work/d20", code=3)
    assert result.returncode == 3 and done_line(result) == 3 and "do not retry" in result.stdout and "--forms monocular_values" not in calls and "--forms uwb_blank" not in calls and "upload --partial" in calls.splitlines()[-2]


def test_a_control_that_did_not_run_is_a_failed_read_not_a_void(tmp_path):
    result, calls = run_job(tmp_path, fail_when="path_parts.py control*/work/d20", code=1)
    assert result.returncode == 1 and done_line(result) == 1 and "did not run" in result.stdout and "--forms monocular_values" not in calls


def test_a_log_that_cannot_be_written_is_not_hidden_by_the_pipe(tmp_path):
    result, calls = run_job(tmp_path, tee_fails=True)
    assert result.returncode == 1 and done_line(result) == 1 and "path_parts.py read" not in calls and "upload --partial" in calls.splitlines()[-2]


def test_a_log_upload_that_fails_keeps_the_jobs_status(tmp_path):
    result, _ = run_job(tmp_path, fail_when="path_parts.py upload-log", code=1)
    assert result.returncode == 0 and done_line(result) == 0 and "ALERT log upload failed" in result.stdout


def test_a_rerun_passes_its_cause(tmp_path):
    _, calls = run_job(tmp_path, RERUN_CAUSE="why now")
    assert "--rerun-cause why now" in next(x for x in calls.splitlines() if "preflight" in x)


def test_an_install_failure_stops_before_anything_reads(tmp_path):
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    (bin_dir / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\nexit 0\n')
    (bin_dir / "uv").write_text("#!/bin/bash\nexit 4\n")
    for name in ("python", "uv"):
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    done = subprocess.run(["bash", str(JOB)], env=dict(PATH=f"{bin_dir}:/usr/bin:/bin", HOME=str(tmp_path), VENV_BIN=str(bin_dir), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path), WORK=str(tmp_path / "work")), capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and done_line(done) == 1 and "transformers install failed" in done.stdout and "path_parts.py instruments" not in log.read_text() and "upload --partial" in log.read_text()


def test_a_missing_repo_prints_its_done_line(tmp_path):
    done = subprocess.run(["bash", str(JOB)], env=dict(PATH="/usr/bin:/bin", HOME=str(tmp_path), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path / "gone")), capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and done.stdout.strip().endswith("PATH-PARTS-DONE 1")


def test_the_job_passes_every_step_and_alert_through_pipestatus_names_no_other_list_and_is_valid_bash():
    text = JOB.read_text()
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0
    code = [line.split(" #")[0] for line in text.splitlines() if not line.startswith("#")]
    assert [line for line in code if re.search(r"(?<!\|)\|(?!\|)", line)] == ['  "$@" 2>&1 | tee -a "$LOG"'] and "${PIPESTATUS[@]}" in text and "set -uo pipefail" in text and "PATH-PARTS-DONE" in text and "transformers==4.57.6" in text
    python_lines = [line.strip() for line in code if re.search(r"\bpython \$T\b", line)]
    assert python_lines and all(line.startswith("step python $T") or "upload --partial" in line or "upload-log" in line for line in python_lines)
    for word in ("c20", "C20", "val1000", "eval1000", "round15_compose_eval", "composed", "--export", "create-pod", "runpodctl"):
        assert word not in "\n".join(code), word
    assert "D=$O/round20/path-read-parts" in text and "L40S" in text


def test_no_existing_file_changes_and_path_reads_own_constants_stand():
    assert R.PREFIX.endswith("round20/path-read") and R.JOB_FILE == "runpod/fleet/path_read.sh" and PP.PREFIX != R.PREFIX
