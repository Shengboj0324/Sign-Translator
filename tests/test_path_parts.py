"""The path read's follow-up on synthetic data and the real builders' layouts: which rotations each form takes, the isolated split in what v1 reads, the instruments' pins and training records, the control, both readings and their alerts, and the job. No test starts a real job: the runner tests use fake executables and no credentials, and the store is a directory or a stand-in."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
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
from tests.test_tok_probes import SpyStore, gappy  # noqa: E402

JOB = ROOT / "runpod/fleet/path_parts.sh"
LEFT_LOCAL, RIGHT_LOCAL = range(7, 27), range(28, 48)
ARM_POINTS = PP.ARM_POINTS


def plant_instruments(store, **change):
    """The three run records as R2 holds them, changed as asked: {"unfitted.args.x": value}."""
    unfitted = dict(args=dict(init_key=None, init_sha256=None, train_index=[PP.UWB_INDEX], select_on="monitor", monitor_index=PP.UWB_INDEX, val_index=PP.HOW2SIGN_FK_INDEX, t5_name="google/flan-t5-base"), config=dict(features=144, fps=24., d_model=384), train_clips=323978)
    best = dict(checkpoint_key=PP.UNFITTED_KEY, step=76000, validation={"val/bleu": 1.1016, "monitor/bleu": 6.0888, "monitor/loss": 2.8261, "other": 1})
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
    LEVELS = dict(two_view=(15, 1.), monocular_values=(10, 1.6), mono_body=(11, 1.5), mono_hands=(13, 1.2), arms=(12, 1.4), hand_pose=(14, 1.1), orientation=(14, 1.05), nearest_two=(11, 1.5), nearest_mono=(7, 2.2), blank=(0, 3.), rt_two_view=(5, 2.75), other=(2, 2.95))

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
        return {("two", "two", "two"): "two_view", ("mono", "mono", "mono"): "monocular_values", ("mono", "mono", "two"): "mono_body", ("two", "two", "mono"): "mono_hands", ("mono", "other", "two"): "arms", ("two", "other", "mono"): "hand_pose", ("two", "other", "two"): "orientation"}.get(key, "other")

    def read(self, states, texts, candidates, group=2):
        own = [texts[c] for c in candidates[:, 0]]
        rows = [self.levels[self.level(s, t)] for s, t in zip(states, own)]
        return dict(hypotheses=[" ".join(t.split()[:k]) if k else "z y x w" for t, (k, _) in zip(own, rows)], nll=np.array([[v] * candidates.shape[1] for _, v in rows]))


class UnfittedReader:
    """The unfitted instrument: words said, NLL of a clip's own reference, and its margin over the mismatched ones, by what a state holds. By default monocular keeps 0.5 of two-view's margin."""
    info = dict(run=PP.UNFITTED_RUN, checkpoint_key=PP.UNFITTED_KEY, sha256=PP.UNFITTED_SHA, kind="keypoints")

    def __init__(self, two=(6, 2., 1.), mono=(5, 2.5, .5), real=(6, 2.1, .9), blank=(0, 3., 0.)) -> None:
        self.two, self.mono, self.real, self.blank = two, mono, real, blank

    def row(self, state):
        valid = state.channels["left_hand"].valid
        if not valid.any():
            return self.blank
        if not valid.all():
            return self.real
        return self.two if len(state.timestamps) == 20 else self.mono

    def read(self, states, texts, candidates, group=2):
        own = [texts[c] for c in candidates[:, 0]]
        rows = [self.row(s) for s in states]
        nll = np.array([[v] + [v + m] * (candidates.shape[1] - 1) for _, v, m in rows])
        return dict(hypotheses=[" ".join(t.split()[:k]) if k else "z y x w" for t, (k, _, _) in zip(own, rows)], nll=nll)


class ContentBlind(UnfittedReader):
    """Sees that points are marked and nothing else: every marked input reads each sentence the same, offset from blank by what the flags alone do; a sentence's own NLL depends on the sentence, not the clip."""

    def __init__(self, offset) -> None:
        super().__init__()
        self.offset = offset

    def read(self, states, texts, candidates, group=2):
        prior = {t: 3. + (i % 7) * .1 for i, t in enumerate(texts)}
        nll = np.array([[prior[texts[c]] - (self.offset if s.channels["left_hand"].valid.any() else 0.) for c in row] for s, row in zip(states, candidates)])
        return dict(hypotheses=["a b c d e f"] * len(states), nll=nll)


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

    def plant_steps(self, monitor=None):
        """The positive control's and the marginals' records, as those steps leave them."""
        self.forms.mkdir(parents=True, exist_ok=True)
        (self.forms / "monitor_control.json").write_text(json.dumps(dict(dict(kind="path-parts-monitor-control", code_digest=PP.code_digest(), smoke=not self.declared, reproduced=True if self.declared else None, misses=[], bleu=6.09, stored=dict(bleu=6.0888, loss=2.8261)), **(monitor or {}))))
        (self.forms / "marginals.json").write_text(json.dumps(dict(kind="path-parts-marginals", code_digest=PP.code_digest(), clips=1)))

    def read_rest(self):
        assert self.read(*PP.PART_FORMS, *PP.NEAREST_FORMS, resume=True) == 0
        assert self.read(*PP.UNFITTED_FORMS, resume=True) == 0

    def pipeline(self, name="report.json", figures=True, monitor=None):
        self.to_control()
        self.read_rest()
        if figures:
            self.plant_figures()
        self.plant_steps(monitor)
        return self.merge(name)


# ---- pins and instruments ---------------------------------------------------------------------------------------------------------

def test_the_unfitted_instrument_is_pinned_and_is_not_v1():
    assert PP.UNFITTED_RUN == "back-translator-kp2d-uwb-v2" and PP.UNFITTED_KEY == "training-runs/back-translator-kp2d-uwb-v2/checkpoints/step-00076000.pt" and PP.UNFITTED_SHA == "f1ff94466a9e12c4d281c2d3dbc281c1049bc081f9123957c5e2dacde9e1821b" and PP.UNFITTED_SHA != L.TRANSLATOR_SHA


def test_the_instruments_block_states_what_each_trained_on(tmp_path):
    store = DirectoryStore(tmp_path / "s")
    plant_instruments(store)
    block = PP.instrument_block(store)
    assert block["unfitted"]["trained_on"] == [PP.UWB_INDEX] and block["unfitted"]["sha256"] == PP.UNFITTED_SHA and block["unfitted"]["step"] == 76000 and block["unfitted"]["selected_on"] == "monitor" and block["unfitted"]["validation_monitor_only"] == PP.HOW2SIGN_FK_INDEX and block["unfitted"]["train_clips"] == 323978 and "MediaPipe" in block["unfitted"]["landmarks"] and block["unfitted"]["started_from"] == dict(weights=None, pretrained_decoder="google/flan-t5-base") and "parent" in block["unfitted"]["relation"]
    assert block["unfitted"]["validation"] == {"val/bleu": 1.1016, "monitor/bleu": 6.0888, "monitor/loss": 2.8261} and "beam 1" in block["unfitted"]["validation_note"] and "512" in block["unfitted"]["validation_note"] and "20 s" in block["unfitted"]["validation_note"]
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


def test_orientation_only_holds_two_views_arm_points_and_moves_only_the_hands_turn():
    mono, two, ref = real_builder_forms()
    out, _ = v1_points(PP.orientation_only(two, mono, ref))
    base, _ = v1_points(V.with_marks(two, ref))
    pose, _ = v1_points(PP.arms_and_hand_pose(two, mono, ref))
    np.testing.assert_allclose(out[:, list(ARM_POINTS)], base[:, list(ARM_POINTS)], atol=1e-5)
    assert np.abs(out[:, list(LEFT_LOCAL)] - base[:, list(LEFT_LOCAL)]).max() > .05 and np.abs(out[:, list(LEFT_LOCAL)] - pose[:, list(LEFT_LOCAL)]).max() > .05
    body = PP.orientation_only(two, mono, ref).channels["left_hand"]
    np.testing.assert_array_equal(body.values[body.valid], two.channels["left_hand"].values[body.valid])


def test_the_point_check_reads_each_forms_sources_in_v1s_view():
    mono, two, ref = real_builder_forms()
    row = PP.isolation_row("mono_arms_only", PP.arms_and_hand_pose(mono, two, ref), mono, two, ref)
    assert row["arm"] < 1e-3 and row["hand"] < 1e-3
    row = PP.isolation_row("mono_hand_pose_only", PP.arms_and_hand_pose(two, mono, ref), mono, two, ref)
    assert row["arm"] < 1e-3 and row["hand"] < 1e-3
    row = PP.isolation_row("mono_hand_orientation_only", PP.orientation_only(two, mono, ref), mono, two, ref)
    assert row["arm"] < 1e-3 and row["hand"] is None
    assert PP.isolation_row("mono_body_two_view_hands", PP.body_and_hands(mono, two, ref), mono, two, ref)["hand"] is None and PP.isolation_row("mono_body_two_view_hands", PP.body_and_hands(mono, two, ref), mono, two, ref)["arm"] < 1e-3
    assert PP.isolation_row("two_view_body_mono_hands", PP.body_and_hands(two, mono, ref), mono, two, ref)["arm"] < 1e-3
    assert PP.isolation_row("monocular_values_two_view_marks", V.with_marks(mono, ref), mono, two, ref) is None
    bad = PP.isolation_row("mono_arms_only", PP.arms_and_hand_pose(two, mono, ref), mono, two, ref)
    assert bad["arm"] > .05 and bad["hand"] > .05
    plain = PP.isolation_row("mono_arms_only", V.with_marks(replace(two, channels={**two.channels, "body": mono.channels["body"]}), ref), mono, two, ref)
    assert plain["arm"] < 1e-3 and plain["hand"] > .05        # an arm swap with no held hand misses on the hand


def test_a_miss_in_the_point_check_is_named_and_withholds_question_1_alone(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    report = h.pipeline()
    assert report["alerts"] == []
    path = h.forms / "form-mono_arms_only.json"
    form = json.loads(path.read_text())
    form["isolation"][2] = dict(arm=0., hand=.0011)
    path.write_text(json.dumps(form))
    report = h.merge("again.json")
    assert report["alerts"] == [f"mono_arms_only: s2 is off its sources in v1's view: {dict(hand=.0011)}"] and report["which_part"]["label"] is None and report["fit_or_content"]["label"] is not None
    form["isolation"][2] = dict(arm=.001, hand=None)
    path.write_text(json.dumps(form))
    assert PP.isolation_misses(dict(mono_arms_only=form)) == [] and PP.ISOLATION_TOLERANCE == 1e-3


def test_nearest_filled_frames_are_counted_per_part_under_two_views_marks():
    trimmed, ref = pair()
    counts = PP.filled_counts(trimmed, ref)
    from signtranslator.data.keypoints2d import from_state
    own, marks = from_state(trimmed)[1], from_state(ref)[1]
    assert counts["left_hand"] == [int((marks & ~own)[:, 6:27].sum()), int(marks[:, 6:27].sum())] and counts["left_hand"][0] > 0 and counts["shoulders"] == [int((marks & ~own)[:, [0, 1]].sum()), int(marks[:, [0, 1]].sum())] and set(counts) == set(R.PARTS)
    assert PP.filled_counts(ref, ref)["left_hand"][0] == 0
    wide = gappy(ref, 8, 16)          # two-view marks fewer than the monocular clip, so the counts cannot agree by chance
    mine, theirs = from_state(trimmed)[1], from_state(wide)[1]
    assert PP.filled_counts(trimmed, wide)["left_hand"] == [int((theirs & ~mine)[:, 6:27].sum()), int(theirs[:, 6:27].sum())] and int((theirs & ~mine)[:, 6:27].sum()) != int((mine & ~theirs)[:, 6:27].sum()) and int(theirs[:, 6:27].sum()) != int(mine[:, 6:27].sum())
    rows = [dict(filled_under_two_view_marks={p: [1, 4] for p in R.PARTS}), dict(filled_under_two_view_marks={p: [1, 4] for p in R.PARTS})]
    assert PP.filled_shares(rows)["left_hand"] == dict(filled=2, marked=8, share=.25)
    assert PP.filled_shares([dict(filled_under_two_view_marks={p: [0, 0] for p in R.PARTS})])["left_hand"]["share"] is None


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
    for name in ("real_two_view", "uwb_real_two_view"):
        kept, states, masks, checks = PP.build_form(name, refs, clips, None)
        assert kept == [0, 1, 2] and states[1] is refs[1] and masks[1] is refs[1] and checks is None
    for name in ("blank", "uwb_blank"):
        kept, states, masks, checks = PP.build_form(name, refs, clips, None)
        assert kept == [0, 1, 2] and not states[0].channels["body"].valid.any() and masks[0] is refs[0] and len(states[0].timestamps) == 20
    calls = []
    original = T.cap_form
    monkeypatch.setattr(T, "cap_form", lambda name, state, tokenizer: calls.append((name, state)) or original(name, state, tokenizer))
    for name in ("nearest_two_view", "uwb_nearest_two_view"):
        kept, states, masks, checks = PP.build_form(name, refs, clips, tok)
        assert kept == [0, 1, 2] and calls[-1] == ("v0_nearest_fill", refs[2]) and masks[1] is refs[1]
    for name in ("nearest_monocular", "uwb_nearest_monocular"):
        kept, states, masks, checks = PP.build_form(name, refs, clips, tok)
        assert kept == [0, 1] and calls[-1][0] == "v0_nearest_fill" and calls[-1][1] is clips[1].conformed and masks[1] is clips[1].conformed and len(states[0].timestamps) == 24
    mono, two = PP.own_values(clips[1].trimmed, tok), PP.own_values(refs[1], tok)
    expected = dict(monocular_values_two_view_marks=V.with_marks(mono, refs[1]), mono_body_two_view_hands=PP.body_and_hands(mono, two, refs[1]), two_view_body_mono_hands=PP.body_and_hands(two, mono, refs[1]), mono_arms_only=PP.arms_and_hand_pose(mono, two, refs[1]), mono_hand_pose_only=PP.arms_and_hand_pose(two, mono, refs[1]), mono_hand_orientation_only=PP.orientation_only(two, mono, refs[1]))
    for name, state in expected.items():
        kept, states, masks, checks = PP.build_form(name, refs, clips, tok)
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
    assert len(form["isolation"]) == 3 and form["isolation"][0]["arm"] < 1e-3 and form["isolation"][0]["hand"] is None
    assert json.loads((h.forms / "form-real_two_view.json").read_text())["tokenizer"] is None
    near = json.loads((h.forms / "form-nearest_monocular.json").read_text())
    assert near["seconds_over_reference"] == [1.2] * 3 and near["valid_frame_share"] == [1.] * 3
    plan = json.loads((h.forms / "monocular.json").read_text())
    assert plan["kind"] == "path-parts-monocular" and plan["clips_loaded"] == 3 and plan["code_digest"] == PP.code_digest() and plan["conform"]["kinds"]["body:equivalent"] == 3 and plan["pairing"][1]["sign"] == "s1" and plan["pairing"][1]["validity"] == R.part_validity(layout_adapter.conform(h.mono[1], template_state())) and plan["pairing"][1]["source_hash_matches"] is True and plan["pairing"][1]["filled_under_two_view_marks"] == PP.filled_counts(R.cut_frames(layout_adapter.conform(h.mono[1], template_state()), np.arange(*CUT)), h.refs[1]) and plan["pairing"][1]["filled_under_two_view_marks"]["left_hand"][0] > 0
    assert any('"monocular_clips": 3' in line and '"24.0": 3' in line for line in capsys.readouterr().out.splitlines())


def test_the_group_reaches_the_reader_and_a_clip_lost_stays_out(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, missing=(1,))
    h.to_control()
    seen = []
    original = h.reader.read
    monkeypatch.setattr(h.reader, "read", lambda states, texts, candidates, group=2: seen.append((len(states), list(texts), candidates.copy(), group)) or original(states, texts, candidates, group=group))
    assert h.main("read", "--out-dir", str(h.forms), "--forms", "nearest_monocular", "--nll-group", "4") == 0
    count, texts, candidates, group = seen[0]
    assert count == N - 1 and texts == [r["text"] for r in h.records] and candidates.shape == (N - 1, 1) and group == 4 and candidates[:, 0].tolist() == [i for i in range(N) if i != 1]
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


def test_a_control_of_another_list_voids_the_read_and_writes_its_record(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    assert h.read(*PP.ANCHORS) == 0
    h.plant_control(order=True)
    assert h.control() == 3
    result = json.loads((h.forms / "control.json").read_text())
    assert result["reproduced"] is False and "other clips or another order" in result["misses"][0] and result["code_digest"] == PP.code_digest() and result["reads"] is None
    assert json.loads(h.store.get(f"pr/void/{PP.code_digest()}.json"))["control"]["misses"] == result["misses"]
    with pytest.raises(SystemExit, match="no passed control"):
        h.read("uwb_blank")


def test_a_changed_or_shapeless_stored_report_refuses(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    assert h.read(*PP.ANCHORS) == 0
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
    assert likelihood["lost"]["hand_orientation"]["point"] == pytest.approx(.05) and likelihood["share"]["hand_orientation"]["point"] == pytest.approx(.0833, abs=1e-3) and likelihood["together"]["channels"]["point"] == pytest.approx(.1667, abs=1e-3) and likelihood["together"]["view"]["point"] == pytest.approx(-.1667, abs=1e-3)
    assert likelihood["case"] == PP.LABEL_BODY and likelihood["view_case_report_only"] == "arms" and block["bootstrap"] == dict(n=2000, seed=0) and block["clips"] == N


def share(point, low, high):
    return dict(point=point, ci95=[low, high])


@pytest.mark.parametrize("total,first,second,case", [
    (.1, share(.67, .34, .9), share(.34, .1, .49), "both"), (.1, share(.67, .34, .9), share(.33, .1, .49), PP.LABEL_BODY), (.1, share(.67, .34, .9), share(.32, .1, .5), "unresolved"), (.1, share(.66, .34, .9), share(.2, .1, .4), "unresolved"), (.1, share(.67, .32, .9), share(.2, .1, .4), "unresolved"),
    (.1, share(.32, .1, .4), share(.67, .34, .9), PP.LABEL_FINGERS), (.1, share(.32, .1, .5), share(.67, .34, .9), "unresolved"), (.1, share(.2, .1, .4), share(.66, .4, .9), "unresolved"), (.1, share(.2, .1, .4), share(.7, .3, .9), "unresolved"), (.1, share(.34, .1, .49), share(.67, .34, .9), "both"),
    (.1, share(.4, .2, .6), share(.5, .3, .7), "both"), (.1, share(.34, .01, .6), share(.34, .01, .6), "both"), (.1, share(.34, 0., .6), share(.5, .3, .7), "unresolved"), (.1, share(.5, .3, .7), share(.34, 0., .6), "unresolved"), (.1, share(.33, .1, .5), share(.5, .3, .7), "unresolved"), (.1, share(.2, .1, .4), share(.2, .1, .4), "unresolved"),
    (0., share(.8, .5, .9), share(.1, 0., .2), "unresolved"), (-.1, share(.8, .5, .9), share(.1, 0., .2), "unresolved")])
def test_the_part_case_of_one_metric(total, first, second, case):
    assert PP.part_case(dict(point=.5, ci95=[total, 1.]), first, second, (PP.LABEL_BODY, PP.LABEL_FINGERS)) == case


def test_the_two_shares_less_one_are_reported_with_their_interval():
    point = dict(real_two_view=2., a=1.5, b=1.8, both=1.)
    draws = dict(real_two_view=np.array([2., 3.]), a=np.array([1.5, 2.]), b=np.array([1.8, 2.]), both=np.array([1., 1.]))
    out = PP.together(point, draws, "a", "b", "both")
    assert out["point"] == pytest.approx(-.3) and out["ci95"] == P._quantiles(np.array([-.3, 0.]))
    assert PP.together(dict(point, both=2.), draws, "a", "b", "both") is None


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

def test_the_fit_block_reads_the_margin_and_each_instruments_own_blank(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, unfitted=UnfittedReader(two=(6, 8., 1.), mono=(5, 8.5, .5), real=(6, 8.1, .9), blank=(0, 9., 0.)))
    report = h.pipeline()
    block = report["fit_or_content"]["likelihood"]
    unfitted = block["unfitted"]
    assert unfitted["margins"] == dict(uwb_blank=0., uwb_real_two_view=.9, uwb_nearest_two_view=1., uwb_nearest_monocular=.5) and unfitted["own_margin"]["point"] == 1. and unfitted["monocular_minus_two_view"]["point"] == -.5 and unfitted["monocular_share_of_own_margin"]["point"] == .5 and unfitted["real_two_view_margin"]["point"] == .9 and unfitted["blank_margin"]["point"] == 0.
    assert unfitted["over_blank"]["cells"]["uwb_nearest_two_view"] == 1. and unfitted["over_blank"]["share"]["point"] == .5 and "offset" in unfitted["over_blank"]["note"]
    assert block["v1"]["own_gap"]["point"] == 1.5 and block["v1"]["monocular_minus_two_view"]["point"] == pytest.approx(-.7) and block["v1"]["monocular_share_of_own_gap"]["point"] == pytest.approx(.5333, abs=1e-3)
    assert block["share_difference"]["point"] == pytest.approx(.5 - .5333, abs=1e-3) and block["share_difference"]["one_sided_upper_95"] == pytest.approx(.5 - .5333, abs=1e-3)
    assert report["fit_or_content"]["thresholds"] == dict(comparable=.25, kept=.75, readable_width=.5) and report["fit_or_content"]["candidates"] == 5 and PP.CANDIDATES == 5
    assert report["forms"]["uwb_nearest_monocular"]["likelihood_minus_blank"]["mean"] == .5 and report["forms"]["uwb_nearest_two_view"]["likelihood_minus_blank"]["mean"] == 1. and report["forms"]["nearest_two_view"]["likelihood_minus_blank"]["mean"] == 1.5


def test_only_uwb_forms_read_mismatched_references_and_v1_keeps_one(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    h.pipeline()
    assert np.asarray(json.loads((h.forms / "form-real_two_view.json").read_text())["nll"]).shape == (N, 1) and np.asarray(json.loads((h.forms / "form-nearest_monocular.json").read_text())["nll"]).shape == (N, 1)
    for name in PP.UNFITTED_FORMS:
        assert np.asarray(json.loads((h.forms / f"form-{name}.json").read_text())["nll"]).shape == (N, 5)


def test_a_lost_clip_moves_no_other_clips_candidates(monkeypatch, tmp_path):
    seen = {}

    class Spy(UnfittedReader):
        def read(self, states, texts, candidates, group=2):
            seen[len(states)] = candidates.copy()
            return super().read(states, texts, candidates, group=group)
    h = Parts(monkeypatch, tmp_path, missing=(1,), unfitted=Spy())
    h.to_control()
    assert h.read("uwb_nearest_two_view", "uwb_nearest_monocular") == 0
    np.testing.assert_array_equal(seen[N - 1], seen[N][[i for i in range(N) if i != 1]])
    assert (seen[N][:, 0] == np.arange(N)).all() and seen[N].shape == (N, 5)


def row(point, low, high):
    return dict(point=point, ci95=[low, high])


OWN, DROP = row(.2, .1, .3), row(-.1, -.15, -.05)


@pytest.mark.parametrize("own,share_u,drop,upper,case", [
    (OWN, row(.3, .1, .5), DROP, .1, PP.LABEL_LOSS), (OWN, row(.43, .3, .75), DROP, .25, PP.LABEL_LOSS), (OWN, row(.44, .3, .75), DROP, .2501, "unresolved"), (OWN, row(.1, 0., .3), DROP, -.5, PP.LABEL_LOSS),
    (OWN, row(.5, .3, .76), DROP, .1, "unresolved"), (OWN, row(.3, .1, .5), row(-.1, -.2, 0.), .1, "unresolved"), (OWN, row(.3, .1, .5), row(-.1, -.2, .01), .1, "unresolved"),
    (OWN, row(.9, .75, 1.1), row(-.01, -.1, 0.), .9, PP.LABEL_GAP), (OWN, row(1.2, 1., 1.4), row(.04, .01, .07), 1.1, PP.LABEL_GAP), (OWN, row(.9, .74, 1.1), row(-.01, -.1, .05), .9, "unresolved"),
    (OWN, row(.9, .8, 1.), DROP, .8, "unresolved"), (OWN, row(.9, .8, 1.), row(-.01, -.1, -.001), .8, "unresolved"),
    (row(.2, 0., .3), row(.3, .1, .5), DROP, .1, "unresolved"), (row(.2, -.1, .3), row(.9, .8, 1.), row(0., -.1, .1), .1, "unresolved"), (OWN, row(.3, 0., .51), DROP, .1, "unresolved"), (OWN, row(.3, 0., .5), DROP, .1, PP.LABEL_LOSS),
    (OWN, None, DROP, .1, "unresolved"), (OWN, row(.3, .1, .5), DROP, None, "unresolved")])
def test_the_fit_case_at_its_thresholds(own, share_u, drop, upper, case):
    assert PP.fit_case(own, share_u, drop, upper) == case


def test_the_labels_say_only_what_they_can():
    assert PP.LABEL_LOSS == "loss_not_explained_by_fine_tune" and PP.LABEL_GAP == "gap_not_reproduced" and (PP.LABEL_BODY, PP.LABEL_FINGERS) == ("body_channel_with_wrist", "fingers")


def test_the_label_is_likelihoods_and_bleu_is_reported_beside_it(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    report = h.pipeline()
    block = report["fit_or_content"]
    assert block["case"] == block["likelihood"]["case"] and "case" in block["bleu"] and block["bleu"]["unfitted"]["own_gap"]["point"] > 0 and block["bleu"]["unfitted"]["cells"]["uwb_real_two_view"] > 0 and block["bleu"]["share_difference"]["one_sided_upper_95"] is not None and block["bleu"]["unfitted"]["monocular_minus_two_view"]["point"] < 0


def test_a_content_blind_instrument_with_an_offset_cannot_read_as_keeping_the_gap(monkeypatch, tmp_path):
    """The flaw this rule replaced: s = (m + O) / (t + O) goes to 1 as O grows. Over blank it would read as kept; on the margin it has nothing to read."""
    h = Parts(monkeypatch, tmp_path, unfitted=ContentBlind(offset=2.))
    report = h.pipeline()
    block = report["fit_or_content"]
    unfitted = block["likelihood"]["unfitted"]
    assert unfitted["over_blank"]["cells"]["uwb_nearest_two_view"] == pytest.approx(2.) and unfitted["over_blank"]["share"]["point"] == 1.
    assert abs(unfitted["own_margin"]["point"]) < .6 and unfitted["monocular_minus_two_view"]["point"] == 0. and block["likelihood"]["case"] == "unresolved" and block["label"] != PP.LABEL_GAP


@pytest.mark.parametrize("offset", [0., .5, 5., 50.])
def test_at_500_clips_an_offset_never_reads_as_kept(offset):
    rng = np.random.default_rng(0)
    refs = [f"s{i}" for i in range(500)]
    candidates = M.retrieval_candidates(500, PP.CANDIDATES, seed=0)
    prior = rng.normal(size=500) * .4 + 3.

    def form(off):
        return dict(signs=refs, references=refs, hypotheses=["a b c d e f"] * 500, nll=np.array([[prior[c] - off for c in row] for row in candidates]).tolist())
    reads = {n: form(offset) for n in ("uwb_real_two_view", "uwb_nearest_two_view", "uwb_nearest_monocular")}
    reads["uwb_blank"] = form(0.)
    reads["blank"] = dict(signs=refs, references=refs, hypotheses=["z y x w"] * 500, nll=[[3.]] * 500)
    reads["nearest_two_view"], reads["nearest_monocular"] = dict(reads["blank"], hypotheses=["a b c d e f"] * 500, nll=[[2.]] * 500), dict(reads["blank"], hypotheses=["a b c d e f"] * 500, nll=[[2.8]] * 500)
    block = PP.fit_block(reads)["likelihood"]
    assert block["case"] != PP.LABEL_GAP and (block["unfitted"]["own_margin"]["ci95"][0] <= 0 or block["unfitted"]["monocular_share_of_own_margin"]["ci95"][1] - block["unfitted"]["monocular_share_of_own_margin"]["ci95"][0] > .5)


def test_a_significant_loss_cannot_read_as_gap_not_reproduced():
    assert PP.fit_case(OWN, row(.85, .76, .95), row(-.05, -.08, -.02), .9) == "unresolved"


def test_marginal_rows_give_a_paths_distance_from_uwb():
    uwb = {p: (np.array([0., 0.]), np.array([2., 2.])) for p in R.DISTANCE_POINTS}
    path = dict(uwb, left_hand=(np.array([2., 4.]), np.array([4., 2.])), right_hand=None)
    out = PP.marginal_row(path, uwb)
    assert out["left_hand"] == dict(mean_in_uwb_std=1.5, std_ratio=1.5) and out["right_hand"] is None and out["shoulders"] == dict(mean_in_uwb_std=0., std_ratio=1.)
    assert PP.marginal_row(uwb, dict(uwb, wrist=None, left_elbow=None))["left_elbow"] is None
    points = np.array([[[1., 2.]] * 48, [[3., 4.]] * 48], np.float64)
    valid = np.ones((2, 48), bool)
    valid[1, 2] = False
    rows = PP.part_points(points, valid)
    np.testing.assert_allclose(rows["left_wrist"][0], [2., 3.])
    np.testing.assert_allclose(rows["left_elbow"][0], [1., 2.])
    assert PP.part_points(points, np.zeros((2, 48), bool))["shoulders"] is None


def test_a_margin_kept_whole_with_noise_reads_gap_not_reproduced_and_no_loss_reads_otherwise(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, unfitted=UnfittedReader(mono=(6, 2., 1.)))
    report = h.pipeline()
    assert report["fit_or_content"]["likelihood"]["unfitted"]["monocular_share_of_own_margin"]["point"] == 1. and report["fit_or_content"]["case"] == PP.LABEL_GAP and report["fit_or_content"]["label"] == PP.LABEL_GAP


def test_a_margin_lost_alike_reads_loss_not_explained_by_fine_tune(monkeypatch, tmp_path):
    """v1 keeps 0.533 of its gap; this instrument keeps 0.5 of its margin."""
    h = Parts(monkeypatch, tmp_path)
    report = h.pipeline()
    block = report["fit_or_content"]["likelihood"]
    assert block["unfitted"]["monocular_minus_two_view"]["ci95"] == [-.5, -.5] and block["case"] == PP.LABEL_LOSS and report["fit_or_content"]["label"] == PP.LABEL_LOSS and report["fit_or_content"]["void"] is False


def test_an_instrument_with_no_margin_of_its_own_is_unresolved(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, unfitted=UnfittedReader(two=(6, 2., 0.), mono=(5, 2.5, 0.)))
    report = h.pipeline()
    assert report["fit_or_content"]["likelihood"]["unfitted"]["monocular_share_of_own_margin"] is None and report["fit_or_content"]["case"] == "unresolved"


def test_a_one_candidate_form_has_no_margin():
    with pytest.raises(SystemExit, match="no mismatched reference"):
        PP.margins(dict(form="x", nll=[[1.], [2.]]))
    np.testing.assert_allclose(PP.margins(dict(nll=[[1., 2., 4.], [3., 3., 3.]])), [2., 0.])


def test_the_paired_upper_bound_pairs_its_draws():
    a, b = np.linspace(.2, .8, 100), np.ones(100)
    assert PP.paired_upper(a, b, a, b) == 0.
    assert PP.paired_upper(a, b, a[::-1], b) == round(float(np.percentile(a - a[::-1], 95)), 4)
    assert PP.paired_upper(np.zeros(3), np.zeros(3), a[:3], b[:3]) is None


# ---- intervals as wide as D20's ---------------------------------------------------------------------------------------------------

def d20_like(n=500, seed=0, shift_mono=.1, margin=.2, noise=.35, k=5):
    """500 sentences with rare short matches, candidates' NLL with clip and sentence spread: an instrument with a small margin, as a floor reader may show."""
    rng = np.random.default_rng(seed)
    refs = [" ".join(f"r{rng.integers(3000)}" for _ in range(14)) for _ in range(n)]
    candidates = M.retrieval_candidates(n, k, seed=0)
    prior = rng.normal(size=n) * noise + 5.

    def hypotheses(p):
        out = []
        for ref in refs:
            words, hyp = ref.split(), [f"h{rng.integers(3000)}" for _ in range(13)]
            if rng.random() < p:
                j = int(rng.integers(0, 10))
                hyp[j:j + 4] = words[j:j + 4]
            out.append(" ".join(hyp))
        return out

    def nll(own_margin, offset):
        base = np.array([[prior[c] - offset for c in row] for row in candidates]) + rng.normal(size=(n, k)) * .15
        base[:, 0] -= own_margin
        return base.tolist()
    blank = dict(references=refs, hypotheses=hypotheses(.1), nll=nll(0., 0.))
    two = dict(references=refs, hypotheses=hypotheses(.4), nll=nll(margin, .3))
    mono = dict(references=refs, hypotheses=hypotheses(.2), nll=nll(margin - shift_mono, .3))
    return blank, two, mono


def fit_rows(blank, two, mono, upper=.1):
    idx = PP.draw_indices(len(two["references"]))
    m_two, m_mono = PP.margins(two), PP.margins(mono)
    point = dict(two=float(m_two.mean()), mono=float(m_mono.mean()), zero=0.)
    draws = dict(two=m_two[idx].mean(1), mono=m_mono[idx].mean(1), zero=np.zeros(PP.BOOT))
    return PP.difference(point, draws, "two", "zero"), PP.ratio(point, draws, ("mono", "zero"), ("two", "zero")), PP.difference(point, draws, "mono", "two")


def test_at_500_clips_a_margin_half_lost_reads_as_a_loss_with_tight_shares():
    own, share, drop = fit_rows(*d20_like())
    assert own["ci95"][0] > 0 and share["ci95"][1] - share["ci95"][0] < .5 and drop["ci95"][1] < 0
    assert PP.fit_case(own, share, drop, .1) == PP.LABEL_LOSS


def test_at_500_clips_a_margin_near_the_floor_is_too_small_to_read():
    """A margin this instrument barely shows: its interval reaches zero or the share's is too wide, whatever the point says."""
    own, share, drop = fit_rows(*d20_like(margin=.012, shift_mono=.006))
    assert own["ci95"][0] <= 0 or share["ci95"][1] - share["ci95"][0] > .5
    assert PP.fit_case(own, share, drop, .1) == "unresolved"


def test_at_500_clips_a_margin_kept_whole_reads_gap_not_reproduced():
    own, share, drop = fit_rows(*d20_like(margin=.3, shift_mono=0.))
    assert own["ci95"][0] > 0 and drop["ci95"][1] >= 0
    assert PP.fit_case(own, share, drop, .9) == (PP.LABEL_GAP if share["ci95"][0] >= .75 else "unresolved")


def test_at_500_clips_bleu_intervals_are_as_wide_as_d20s():
    _, two, _ = d20_like()
    bleu, ci = M.bleu_ci_fast(two["hypotheses"], two["references"])
    assert 1. < bleu < 12. and .3 < ci[1] - ci[0] < 3.5


def test_a_reading_is_taken_jointly_over_clips():
    blank, two, mono = d20_like(n=60)
    assert (PP.draw_indices(60) == np.random.default_rng(0).integers(0, 60, (PP.BOOT, 60))).all()
    point, draws = PP.share_rows(dict(uwb_blank=blank, two=two, mono=mono), "uwb_blank")
    idx = np.random.default_rng(0).integers(0, 60, (PP.BOOT, 60))
    base = np.asarray(blank["nll"])[:, 0]
    a = (base - np.asarray(two["nll"])[:, 0])[idx].mean(1)
    np.testing.assert_allclose(draws["likelihood"]["two"], a)
    s = M.bleu_statistics(two["hypotheses"], two["references"])
    np.testing.assert_allclose(draws["bleu"]["two"], M.bleu_from_statistics(s[idx].sum(1)))
    assert point["likelihood"]["uwb_blank"] == 0. and point["bleu"]["two"] == pytest.approx(float(M.bleu_from_statistics(s.sum(0))))


# ---- alerts and labels ------------------------------------------------------------------------------------------------------------

def test_figures_off_the_first_read_raise_alerts_and_each_withholds_its_own_question(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path)
    report = h.pipeline(figures=False)
    assert PP.FIGURES["nearest_two_view"] == (4.621, .234) and any("nearest_two_view reads BLEU" in a for a in report["alerts"]) and any("nearest_monocular reads likelihood" in a for a in report["alerts"]) and any("monocular_values_two_view_marks reads BLEU" in a for a in report["alerts"])
    assert report["which_part"]["label"] is None and report["fit_or_content"]["label"] is None and report["which_part"]["case"] is not None
    assert all("nearest" not in a for a in report["which_part"]["withheld"]) and all("monocular_values" not in a for a in report["fit_or_content"]["withheld"])


@pytest.mark.parametrize("form,question,other", [("monocular_values_two_view_marks", "which_part", "fit_or_content"), ("nearest_monocular", "fit_or_content", "which_part")])
def test_a_figure_off_the_first_read_leaves_the_other_question_standing(monkeypatch, tmp_path, form, question, other):
    h = Parts(monkeypatch, tmp_path)
    h.to_control()
    h.read_rest()
    h.plant_figures()
    monkeypatch.setattr(PP, "FIGURES", dict(PP.FIGURES, **{form: (0., 0.)}))
    h.plant_steps()
    report = h.merge()
    assert report[question]["label"] is None and report[question]["withheld"] and report[other]["label"] is not None and report[other]["withheld"] == []


@pytest.mark.parametrize("bleu,likelihood,count", [(.049, .0019, 0), (.051, 0., 1), (0., .0021, 1), (.051, .0021, 2)])
def test_figures_are_held_to_their_tolerances(monkeypatch, bleu, likelihood, count):
    refs = [r["text"] for r in d20_records(10)]
    blank = dict(signs=[f"s{i}" for i in range(10)], nll=[[3.]] * 10, references=refs)
    form = dict(signs=blank["signs"], references=refs, hypotheses=[" ".join(t.split()[:6]) for t in refs], nll=[[2.]] * 10)
    here = M.bleu_ci_fast(form["hypotheses"], refs)[0]
    monkeypatch.setattr(PP, "FIGURES", dict(nearest_two_view=(here + bleu, 1. + likelihood)))
    assert len(PP.figure_misses(dict(blank=blank, nearest_two_view=form), ("nearest_two_view",))) == count
    assert (PP.BLEU_TOLERANCE, PP.LIKELIHOOD_TOLERANCE) == (.05, .002) and PP.FIGURES_OF_PARTS == ("monocular_values_two_view_marks",)


def test_figures_are_the_first_reads():
    assert PP.FIGURES == dict(nearest_two_view=(4.621, .234), nearest_monocular=(2.653, .041), monocular_values_two_view_marks=(2.546, .097))


def test_an_unfitted_form_reading_as_its_blank_withholds_question_2_alone(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, unfitted=UnfittedReader(two=(0, 3., 0.), mono=(5, 2.5, .5), blank=(0, 3., 0.)))
    report = h.pipeline()
    assert "uwb_nearest_two_view reads exactly as uwb_blank" in report["alerts"] and "uwb_nearest_monocular reads exactly as uwb_blank" not in report["alerts"] and report["fit_or_content"]["label"] is None and report["which_part"]["label"] is not None
    assert PP.main(["check", "--report", str(tmp_path / "report.json")]) == 3


def test_a_form_reads_as_blank_only_when_every_clip_does():
    def form(*hypotheses):
        return dict(signs=["a", "b"], hypotheses=list(hypotheses))
    assert PP.constant_reads(dict(uwb_blank=form("x", "y"), uwb_real_two_view=form("q", "y"), uwb_nearest_two_view=form("x", "z"), uwb_nearest_monocular=form("q", "y"))) == []
    assert PP.constant_reads(dict(uwb_blank=form("x", "y"), uwb_real_two_view=form("q", "z"), uwb_nearest_two_view=form("q", "z"), uwb_nearest_monocular=form("x", "y"))) == ["uwb_nearest_monocular reads exactly as uwb_blank"]
    assert PP.constant_reads(dict(uwb_blank=form("x", "y"), uwb_real_two_view=form("q", "z"), uwb_nearest_two_view=form("x", "y"), uwb_nearest_monocular=form("q", "z"))) == ["uwb_nearest_two_view reads exactly as uwb_blank"]
    assert PP.constant_reads(dict(uwb_blank=form("x", "y"), uwb_real_two_view=form("x", "y"), uwb_nearest_two_view=form("q", "z"), uwb_nearest_monocular=form("q", "z"))) == ["uwb_real_two_view reads exactly as uwb_blank"]


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
    assert report["alerts"] == [] and report["which_part"]["withheld"] == [] and report["fit_or_content"]["withheld"] == [] and report["which_part"]["label"] == report["which_part"]["case"] and report["fit_or_content"]["label"] == report["fit_or_content"]["case"] and report["which_part"]["view_beside_label"] == report["which_part"]["view_case_report_only"]
    assert PP.main(["check", "--report", str(tmp_path / "report.json")]) == 0


def test_a_smoke_carries_no_label_and_reports_its_rows(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, declared=False, n=4)
    assert h.read(smoke=True) == 0
    h.plant_control()
    assert h.control() == 0
    h.plant_steps()
    report = h.merge()
    assert report["declared_read"] is False and report["smoke"].startswith("SMOKE") and report["which_part"]["withheld"] == ["a smoke carries no reading"] and report["fit_or_content"]["withheld"] == ["a smoke carries no reading"] and report["which_part"]["label"] is None and report["fit_or_content"]["label"] is None and report["alerts"] == [] and report["checks"]["control"] is None and report["which_part"]["likelihood"]["share"]["body"]["point"] > 0
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
    assert sorted(signers) == ["c01", "c03", "c05"] and sum(s["clips"] for s in signers.values()) == N - 1 and signers["c03"]["which_part"]["likelihood"]["body"]["point"] == pytest.approx(.8333, abs=1e-3) and signers["c03"]["fit_or_content"]["margin"]["point"] == 1. and signers["c03"]["fit_or_content"]["likelihood"]["point"] == .5 and signers["c03"]["fit_or_content"]["v1_share"]["point"] == pytest.approx(.5333, abs=1e-3)
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["shared"] == N - 1 and printed["of"] == N and printed["lost"] == dict(no_monocular_row=1) and printed["fit_or_content"]["label"] == report["fit_or_content"]["label"] and printed["which_part"]["case"] == report["which_part"]["case"] and printed["which_part"]["view_beside_label"] == report["which_part"]["view_case_report_only"] and printed["forms"]["real_two_view"]["bleu"] == report["forms"]["real_two_view"]["bleu"] and printed["which_part"]["withheld"] == [] and printed["pairing"]["offsets"]["clips"] == N - 1 and printed["fit_or_content"]["val_bleu_beside"] == 1.1016 and printed["monitor"]["reproduced"] is True and printed["which_part"]["together"]["likelihood"]["channels"] is not None and printed["pairing"]["filled_under_two_view_marks"]["left_hand"]["marked"] > 0


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


# ---- the positive control and the marginals ---------------------------------------------------------------------------------------

class FakeModel:
    pass


def monitor_setup(monkeypatch, tmp_path, *, bleu=6.0888, loss=2.8261, clips=700):
    import signtranslator.data.keypoint_shards as shards
    import torch
    import train_keypoints_to_text_t5 as trainer
    store = DirectoryStore(tmp_path / "s")
    plant_instruments(store)
    raw = b'{"parts": []}'
    store.put(PP.UWB_INDEX, raw)
    monkeypatch.setattr(PP, "UWB_INDEX_SHA", hashlib.sha256(raw).hexdigest())
    seen = {}

    class Dataset(list):
        pass

    def fake(s, keys, partition, target_fps, max_seconds, limit=0):
        seen.update(keys=keys, partition=partition, target_fps=target_fps, max_seconds=max_seconds, limit=limit)
        return Dataset((torch.zeros((3, 144)), str(i)) for i in range(limit or clips))
    monkeypatch.setattr(shards, "KeypointShardDataset", fake)

    def evaluate(model, batches, device, beam, prefix):
        seen.update(batches=[len(b[1]) for b in batches], beam=beam, prefix=prefix, model=model, clips=[int(t) for b in batches for t in b[1]])
        return {"monitor/bleu": bleu, "monitor/loss": loss}
    monkeypatch.setattr(trainer, "evaluate", evaluate)
    monkeypatch.setattr(PP, "load_unfitted", lambda store, args: type("Reader", (), dict(model=FakeModel, device="cpu"))())
    return store, seen


def monitor(tmp_path, *extra):
    return PP.main(["monitor-control", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w"), *extra])


def test_the_positive_control_rebuilds_the_trainers_sample_and_reproduces(monkeypatch, tmp_path, capsys):
    _, seen = monitor_setup(monkeypatch, tmp_path)
    assert monitor(tmp_path) == 0
    result = json.loads((tmp_path / "w/monitor_control.json").read_text())
    assert result["reproduced"] is True and result["misses"] == [] and result["smoke"] is False and result["sample"] == 512 and result["monitor_clips"] == 700 and result["code_digest"] == PP.code_digest() and result["stored"] == dict(bleu=6.0888, loss=2.8261) and result["limits"] == dict(bleu=.05, loss=.01)
    assert seen["keys"] == [PP.UWB_INDEX] and seen["partition"] == "monitor" and seen["target_fps"] == 24. and seen["max_seconds"] == 20. and seen["limit"] == 0 and seen["beam"] == 1 and seen["prefix"] == "monitor" and seen["model"] is FakeModel and sum(seen["batches"]) == 512 and seen["batches"][0] == 64
    assert seen["clips"] == sorted(random.Random(1234).sample(range(700), 512))
    assert PP.MONITOR == dict(seed=1234, clips=512, batch=64, seconds=20., beam=1) and "ALERT" not in capsys.readouterr().out


@pytest.mark.parametrize("bleu,loss,misses", [(6.0888 + .049, 2.8261 + .009, []), (6.0888 + .051, 2.8261, ["BLEU"]), (6.0888 - .051, 2.8261, ["BLEU"]), (6.0888, 2.8261 + .011, ["loss"]), (1.1, 9., ["BLEU", "loss"])])
def test_the_positive_control_holds_to_its_limits_and_records_a_miss_without_stopping(monkeypatch, tmp_path, capsys, bleu, loss, misses):
    monitor_setup(monkeypatch, tmp_path, bleu=bleu, loss=loss)
    assert monitor(tmp_path) == 0
    result = json.loads((tmp_path / "w/monitor_control.json").read_text())
    assert result["misses"] == misses and result["reproduced"] is (not misses) and ("Question 2 is void" in capsys.readouterr().out) is bool(misses)


def test_a_smoke_positive_control_takes_the_first_clips_and_compares_nothing(monkeypatch, tmp_path):
    _, seen = monitor_setup(monkeypatch, tmp_path, bleu=0., loss=0.)
    assert monitor(tmp_path, "--smoke-clips", "64") == 0
    result = json.loads((tmp_path / "w/monitor_control.json").read_text())
    assert result["smoke"] is True and result["reproduced"] is None and result["misses"] == [] and seen["limit"] == 64 and sum(seen["batches"]) == 64


def test_the_positive_control_holds_the_index_to_its_pin_and_is_write_once(monkeypatch, tmp_path):
    monitor_setup(monkeypatch, tmp_path)
    assert PP.UWB_INDEX_SHA != "c71421b91629db0dae23584910bcad95ae104ff93a75e18e859399432fcb8700"
    assert monitor(tmp_path) == 0
    with pytest.raises(SystemExit, match="write-once"):
        monitor(tmp_path)
    monkeypatch.setattr(PP, "UWB_INDEX_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        PP.main(["monitor-control", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "other")])


def test_a_monitor_miss_voids_question_2_and_leaves_question_1_standing(monkeypatch, tmp_path, capsys):
    h = Parts(monkeypatch, tmp_path)
    report = h.pipeline(monitor=dict(reproduced=False, misses=["BLEU"]))
    assert report["fit_or_content"]["void"] is True and report["fit_or_content"]["label"] is None and report["fit_or_content"]["withheld"] == ["the unfitted instrument missed its monitor read: BLEU"] and report["which_part"]["label"] == report["which_part"]["case"] and report["which_part"]["withheld"] == [] and report["monitor_control"]["reproduced"] is False
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["fit_or_content"]["void"] is True
    assert PP.main(["check", "--report", str(tmp_path / "report.json")]) == 3


@pytest.mark.parametrize("name", ["monitor_control.json", "marginals.json"])
def test_a_merge_needs_each_steps_record_of_this_code(monkeypatch, tmp_path, name):
    h = Parts(monkeypatch, tmp_path, n=3)
    h.pipeline()
    path = h.forms / name
    good = json.loads(path.read_text())
    path.write_text(json.dumps(dict(good, code_digest="other")))
    with pytest.raises(SystemExit, match="another run's record"):
        h.merge("again.json")
    path.unlink()
    with pytest.raises(SystemExit, match=f"{name} is missing"):
        h.merge("again.json")


def test_a_declared_merge_refuses_a_smokes_monitor_record_and_a_smoke_a_declared_ones(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=3)
    h.pipeline()
    path = h.forms / "monitor_control.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), smoke=True)))
    with pytest.raises(SystemExit, match="another run's record"):
        h.merge("again.json")
    good = dict(code_digest="d", smoke=True)
    other = tmp_path / "x.json"
    other.write_text(json.dumps(good))
    assert PP.record_of(other, "d", False) == good
    other.write_text(json.dumps(dict(good, smoke=False)))
    with pytest.raises(SystemExit, match="another run's record"):
        PP.record_of(other, "d", False)
    assert PP.record_of(other, "d", None)["smoke"] is False


def test_the_marginals_compare_each_path_with_uwbs_first_train_clips_on_c02_train(monkeypatch, tmp_path, capsys):
    import signtranslator.data.keypoint_shards as shards
    h = Parts(monkeypatch, tmp_path, declared=False, n=4)
    raw = b"{}"
    h.store.put(PP.UWB_INDEX, raw)
    monkeypatch.setattr(PP, "UWB_INDEX_SHA", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(L, "load_manifest_at", lambda store, key, sha: dict(records=h.records))
    monkeypatch.setattr(L, "train_records", lambda manifest, components, corpus=None: manifest["records"] if components == ("c02",) else pytest.fail("another signer"))
    seen = {}
    points = np.random.default_rng(0).normal(size=(30, 48, 2)).astype(np.float32)

    class Dataset:
        clips = [(points, np.ones((30, 48), bool), 30.)]

        def __len__(self):
            return 1

    def fake(store, keys, partition, target_fps, max_seconds, limit=0):
        seen.update(keys=keys, partition=partition, limit=limit)
        return Dataset()
    monkeypatch.setattr(shards, "KeypointShardDataset", fake)
    assert PP.main(["marginals", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(h.forms), "--clips", "3", "--uwb-clips", "100"]) == 0
    result = json.loads((h.forms / "marginals.json").read_text())
    assert seen == dict(keys=[PP.UWB_INDEX], partition="train", limit=100) and result["report_only"] is True and result["clips"] == 3 and result["code_digest"] == PP.code_digest() and result["uwb_clips"] == 1 and set(result["two_view"]) == set(R.DISTANCE_POINTS) and result["two_view"]["left_wrist"]["std_ratio"] > 0 and result["monocular"]["left_wrist"] != result["two_view"]["left_wrist"] and "not its bias" in result["note"]
    assert json.loads(capsys.readouterr().out)["marginals"]["clips"] == 3 and (PP.MARGINAL_CLIPS, PP.MARGINAL_UWB_CLIPS) == (100, 2000)
    monkeypatch.setattr(PP, "UWB_INDEX_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        PP.main(["marginals", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "other"), "--clips", "1"])


def test_the_marginals_refuse_a_clip_outside_c02_train(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, declared=False, n=2)
    raw = b"{}"
    h.store.put(PP.UWB_INDEX, raw)
    monkeypatch.setattr(PP, "UWB_INDEX_SHA", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(L, "load_manifest_at", lambda store, key, sha: dict(records=[dict(h.records[0], signer_component="c00")]))
    monkeypatch.setattr(L, "train_records", lambda manifest, components, corpus=None: manifest["records"])
    monkeypatch.setattr(L, "load_states", lambda *a: pytest.fail("a state was read"))
    with pytest.raises(SystemExit, match="c00"):
        PP.main(["marginals", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(h.forms), "--clips", "1"])


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
    with pytest.raises(SystemExit, match="not this code's report"):
        h.main("merge", "--out-dir", str(h.forms), "--output-file", str(taken))
    taken.write_text("not json")
    with pytest.raises(SystemExit, match="not this code's report"):
        h.main("merge", "--out-dir", str(h.forms), "--output-file", str(taken))
    with pytest.raises(SystemExit, match="exactly one"):
        h.main("merge", "--out-dir", str(h.forms))
    with pytest.raises(SystemExit, match="exactly one"):
        h.main("merge", "--out-dir", str(h.forms), "--output-file", "a", "--output-key", "b")


def test_a_merge_accepts_an_existing_report_of_this_code_and_leaves_it(monkeypatch, tmp_path, capsys):
    h = Parts(monkeypatch, tmp_path, n=3)
    h.pipeline()
    path = tmp_path / "report.json"
    before = path.read_bytes()
    monkeypatch.setattr(PP, "load_reads", lambda *a: pytest.fail("the forms were loaded"))
    assert h.main("merge", "--out-dir", str(h.forms), "--output-file", str(path)) == 0 and path.read_bytes() == before
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["already"] is True
    h.store.put("pr/report.json", before)
    assert h.main("merge", "--out-dir", str(h.forms), "--output-key", "pr/report.json") == 0 and h.store.get("pr/report.json") == before
    h.store.put("pr/other.json", json.dumps(dict(code_digest="other")).encode())
    with pytest.raises(SystemExit, match="not this code's report"):
        h.main("merge", "--out-dir", str(h.forms), "--output-key", "pr/other.json")


def test_upload_resumes_over_the_same_bytes_and_refuses_others_whole(monkeypatch, tmp_path, capsys):
    h = Parts(monkeypatch, tmp_path, n=2)
    work = tmp_path / "work"
    (work / "forms").mkdir(parents=True)
    (work / "forms" / "form-x.json").write_text("{}")
    (work / "report.json").write_text("{}")
    (work / "note.log").write_text("no")
    assert h.main("upload", "--out-dir", str(work), "--prefix", "pr") == 0 and h.store.get("pr/report.json") == b"{}" and h.store.exists("pr/forms/form-x.json") and not h.store.exists("pr/note.log")
    assert json.loads(capsys.readouterr().out)["uploaded"] == 2
    (work / "more.json").write_text("[]")
    written = []
    original = DirectoryStore.put
    monkeypatch.setattr(DirectoryStore, "put", lambda self, key, payload: written.append(key) or original(self, key, payload))
    assert h.main("upload", "--out-dir", str(work), "--prefix", "pr") == 0 and h.store.get("pr/more.json") == b"[]" and written == ["pr/more.json"]
    assert json.loads(capsys.readouterr().out) == dict(uploaded=1, kept=2, prefix="pr")
    (work / "another.json").write_text("[]")
    (work / "report.json").write_text('{"changed": 1}')
    with pytest.raises(SystemExit, match="holds other bytes"):
        h.main("upload", "--out-dir", str(work), "--prefix", "pr")
    assert not h.store.exists("pr/another.json") and h.store.get("pr/report.json") == b"{}"
    with pytest.raises(SystemExit, match="nothing to upload"):
        h.main("upload", "--out-dir", str(tmp_path / "empty"), "--prefix", "pr")
    assert h.main("upload", "--partial", "--out-dir", str(work), "--prefix", "pr") == 0 and h.store.get("pr/report.json") == b"{}"


def test_a_merge_with_instruments_not_as_stated_stops(monkeypatch, tmp_path):
    h = Parts(monkeypatch, tmp_path, n=3)
    h.to_control()
    h.read_rest()
    h.plant_figures()
    h.plant_steps()
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


@pytest.mark.parametrize("kw,code", [(dict(report_only=True, alerts=[], declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3), which_part=dict(label=None), fit_or_content=dict(label=None)), 0), (dict(report_only=True, alerts=[], declared_read=True, checks=dict(blank_identical_to_real_two_view=0., control=dict(reproduced=True)), clips=dict(shared=3), which_part=dict(label="body", withheld=[]), fit_or_content=dict(label="fit", withheld=[])), 0),
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
    body[which] = dict(label="y", withheld=["x"])
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
    assert all(text in PP.__doc__ for text in PP.PRESTATED.values()) and set(PP.PRESTATED) == {"body", "both", "readable", "loss", "gap", "parent", "bias"}
    doc = PP.__doc__
    assert "stronger" not in doc and "more likely content" not in doc.lower()
    for text in ("report only", "no step branches on it", "no claim about composition", "v1's pre-fine-tune weights", "323,978 UWB YouTube-ASL clips", "the same 144 features", "wrist' = inv(forearm') @ forearm @ wrist", "at most 0.25", "wider than 0.5", "below 475 shared clips", "Can its input see the difference", "kp2d-h2s-v1", "mediapipe image xy", "world points, ik, nominal bones and orthographic fk", "sapiens triangulated in the frontal camera's frame", "may be two-view", "not an independent reader", "s = (m + o) / (t + o)", "one-sided 95% upper bound", "beam 1 on up to 512 clips truncated at 20 s", "6.09", "voids question 2 only", "pretrained t5 decoder", "a + b − 1"):
        assert text.lower() in doc.lower(), text
    assert PP.ARM_POINTS == (0, 1, 2, 3, 4, 5, 6, 27) and PP.HAND_LOCAL_POINTS == tuple(range(7, 27)) + tuple(range(28, 48))
    assert (PP.COMPARABLE, PP.KEPT, PP.READABLE_WIDTH, PP.MINORITY_UPPER) == (.25, .75, .5, .5) and (PP.MONITOR_BLEU_TOLERANCE, PP.MONITOR_LOSS_TOLERANCE) == (.05, .01) and PP.UWB_INDEX_SHA.startswith("c71421b9")


def test_the_disclosures_name_the_blank_the_margin_and_the_parent():
    assert "durations" in PP.DISCLOSURES["blank"] and "seconds_over_reference" in PP.DISCLOSURES["blank"] and "parent" in PP.DISCLOSURES["unfitted"] and "unknown" in PP.DISCLOSURES["unfitted"] and "zero within noise" in PP.DISCLOSURES["margin"] and "filled_under_two_view_marks" in PP.DISCLOSURES["never_tracked"]


def test_the_constants_are_the_declared_ones():
    assert PP.PREFIX == "acquisition/2026-10-17-v1/youtube-asl-v1/round20/path-read-parts" and PP.BOOT == 2000 and PP.JOB_FILE == "runpod/fleet/path_parts.sh" and PP.ANCHORS == ("real_two_view", "blank")
    assert PP.FORMS == ("real_two_view", "blank", "monocular_values_two_view_marks", "mono_body_two_view_hands", "two_view_body_mono_hands", "mono_arms_only", "mono_hand_pose_only", "mono_hand_orientation_only", "nearest_two_view", "nearest_monocular", "uwb_blank", "uwb_real_two_view", "uwb_nearest_two_view", "uwb_nearest_monocular")
    assert set(PP.MONOCULAR_NEEDED) == set(PP.PART_FORMS) | {"nearest_monocular", "uwb_nearest_monocular"} and set(PP.TOKENIZER_NEEDED) == set(PP.FORMS) - {"real_two_view", "blank", "uwb_blank", "uwb_real_two_view"}


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
    assert [a.command for a in parsed] == ["instruments", "read", "control", "monitor-control", "marginals", "merge", "upload", "check", "preflight", "read", "control", "read", "monitor-control", "marginals", "read", "merge", "upload", "check", "upload-log"]
    monitors = [a for a in parsed if a.command == "monitor-control"]
    assert monitors[0].smoke_clips == 64 and monitors[1].smoke_clips == 0 and all(a.device == "cuda" and a.instrument_cache.endswith("/instrument") for a in monitors) and "smoke-" in monitors[0].out_dir and monitors[1].out_dir.endswith("/d20/forms")
    margins_ = [a for a in parsed if a.command == "marginals"]
    assert (margins_[0].clips, margins_[0].uwb_clips) == (3, 100) and (margins_[1].clips, margins_[1].uwb_clips) == (PP.MARGINAL_CLIPS, PP.MARGINAL_UWB_CLIPS)
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
                                                ("path_parts.py preflight", 1), ("--forms real_two_view", 1), ("--forms monocular_values", 1), ("--forms uwb_blank", 1), ("path_parts.py monitor-control*smoke-", 2), ("path_parts.py marginals*smoke-", 2), ("path_parts.py monitor-control*/work/d20", 1), ("path_parts.py marginals*/work/d20", 1), ("path_parts.py merge*/work/d20", 1), ("path_parts.py upload --out-dir*/work/d20", 1), ("path_parts.py check --report*/work/d20", 1)])
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


def test_a_step_already_done_on_this_pod_is_not_run_again(tmp_path):
    (tmp_path / "work" / "d20" / "forms").mkdir(parents=True)
    (tmp_path / "work" / "d20" / "forms" / "monitor_control.json").write_text("{}")
    (tmp_path / "work" / "d20" / "forms" / "marginals.json").write_text("{}")
    result, calls = run_job(tmp_path)
    assert result.returncode == 0 and calls.count("path_parts.py monitor-control") == 1 and calls.count("path_parts.py marginals") == 1


def test_a_monitor_miss_does_not_stop_the_job_before_the_unfitted_read(tmp_path):
    _, calls = run_job(tmp_path)
    assert calls.index("path_parts.py monitor-control --out-dir " + str(tmp_path / "work/d20/forms")) < calls.index("--forms uwb_blank")


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
    assert python_lines and all(re.match(r'(\[ -f "\$R/forms/\w+\.json" \] \|\| )?step python \$T', line) or "upload --partial" in line or "upload-log" in line for line in python_lines)
    for word in ("c20", "C20", "val1000", "eval1000", "round15_compose_eval", "composed", "--export", "create-pod", "runpodctl"):
        assert word not in "\n".join(code), word
    assert "D=$O/round20/path-read-parts" in text and "L40S" in text


def test_no_existing_file_changes_and_path_reads_own_constants_stand():
    assert R.PREFIX.endswith("round20/path-read") and R.JOB_FILE == "runpod/fleet/path_read.sh" and PP.PREFIX != R.PREFIX
