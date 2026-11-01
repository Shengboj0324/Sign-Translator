"""The path read on synthetic data and on the real builders' layouts: what the lookups refuse, how a monocular clip is conformed and paired, the control that stops the job, the shared-clip reads, the label and its alerts, and the job itself. No test starts a real job: the runner tests use fake executables and no credentials, and the store is a directory or a stand-in."""
import ast
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
import path_read as R  # noqa: E402
import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v0 as V  # noqa: E402
import tok_v1 as T  # noqa: E402
from signtranslator.data import layout_adapter  # noqa: E402
from signtranslator.data.keypoints2d import from_state  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402
from signtranslator.eval import provenance, s0_metrics as M  # noqa: E402
from signtranslator.generation.motion_rvq import MotionRVQ, MotionRVQConfig  # noqa: E402
from tests.test_generation_motion_rvq import motion_config  # noqa: E402
from tests.test_generation_signing_space import signer_state  # noqa: E402
from tests.test_tok_probes import SpyStore, gappy  # noqa: E402

N = 9
JOB = ROOT / "runpod/fleet/path_read.sh"
SHARD = f"{R.REPORTS}train_rgb_front_clips-shard-000-of-008.json"
WORDS = [f"w{i}" for i in range(15)]
# the real pair the adapter declares: nothing here adds to EQUIVALENT
TWO_VIEW = ("two-view-triangulated-camera1-y-up", "sapiens-two-view-swing-ik-y-up-v1")
MONOCULAR = ("camera-aligned-y-up-metres", "mediapipe-tasks-swing-ik-y-up-v1")
GROUP_NAMES = ("c01", "c03", "c05")
CUT = (2, 22)        # a two-view clip's frames of its 24-frame monocular clip


def d20_records(n=N):
    return [dict(sign=f"s{i}", corpus=L.H2S, partition="val", signer_component=GROUP_NAMES[i % 3], text=" ".join(f"{w}x{i}" for w in WORDS), continuous=dict(seconds=2.)) for i in range(n)]


def train_records(n=R.SMOKE_CLIPS):
    return [dict(r, partition="train", signer_component="c02") for r in d20_records(n)]


def laid_out(state, layout, placeholder=False):
    """``state`` with its tracked channels in this frame and convention; an untracked channel under a placeholder name, as the monocular path stores it."""
    channels = {}
    for name, c in state.channels.items():
        if c.valid.any():
            channels[name] = replace(c, coordinate_frame=layout[0], convention=layout[1])
        else:
            channels[name] = replace(c, labels=(f"{name}-unavailable",), coordinate_frame="not-estimated") if placeholder else c
    return replace(state, channels=channels)


def template_state():
    return laid_out(signer_state(99, frames=4, amplitude=.4), TWO_VIEW)


def tiny_tokenizer():
    import torch
    torch.manual_seed(0)
    return MotionRVQ(MotionRVQConfig(motion_config(template_state()), hidden=32, code_dim=8, codebook=16, stages=8, down_layers=0)).eval()


def monocular_state(i, frames=24, layout=MONOCULAR):
    """A clip as the monocular path holds it: its own values and gaps, a longer clock, its own layout, placeholder channels. Timestamps stand at the source clip's own, from 0."""
    return laid_out(gappy(signer_state(100 + i, frames=frames, amplitude=.4), 3, 6), layout, placeholder=True)


def two_view_state(i, source_sha, cut=CUT, noise=.05):
    """Its two-view clip: a slice of the same frames (``two_view_states.triangulated_state``), a little off in value, its own gaps, naming its monocular file."""
    base = R.cut_frames(signer_state(100 + i, frames=24, amplitude=.4), np.arange(*cut))
    rng = np.random.default_rng(1000 + i)
    channels = {n: replace(c, values=c.values + rng.normal(size=c.values.shape) * noise) if c.valid.any() else c for n, c in base.channels.items()}
    state = gappy(replace(base, channels=channels), 8 + i % 3, 11 + i % 3)
    return laid_out(replace(state, channels={n: replace(c, source_sha256=source_sha) if c.valid.any() else c for n, c in state.channels.items()}), TWO_VIEW)


def plant_state(store, tmp_path, sign, state):
    path = tmp_path / f"plant-{sign}-{len(list(tmp_path.glob('plant-*')))}.npz"
    digest = state.save(path)
    store.put(R.state_key_of(sign), path.read_bytes())
    return digest


def plant_shard(monkeypatch, store, rows, key=SHARD):
    raw = json.dumps(dict(results=rows)).encode()
    store.put(key, raw)
    monkeypatch.setattr(R, "MONOCULAR_REPORTS", {key: hashlib.sha256(raw).hexdigest()})


class ContentReader:
    """Reads what a state holds, not its length or marks alone: whose values it carries on the frames they were tracked, whose marks, and how long it is. So an emptied or swapped hand reads as something else, and each form has a known level."""
    info = dict(run="planted", checkpoint_key="k", sha256="f" * 64, kind="fake")
    LEVELS = dict(two_view=(15, 1.), rt_two_view=(5, 2.75), monocular=(8, 2.), nearest_monocular=(7, 2.2), round_trip=(4, 2.5), trimmed=(9, 1.9), monocular_values_two_view_marks=(10, 1.6), two_view_values_monocular_marks=(12, 1.3), blank=(0, 3.), other=(2, 2.95))

    def __init__(self, truth: dict, levels=None) -> None:
        self.truth, self.levels = truth, dict(self.LEVELS, **(levels or {}))

    @staticmethod
    def holds(state, other) -> bool:
        for name in ("body", "left_hand", "right_hand"):
            a, b = state.channels[name], other.channels[name]
            both = a.valid & b.valid
            if not both.any() or not np.allclose(a.values[both], b.values[both], atol=1e-3):
                return False
        return True

    @staticmethod
    def marks(state) -> np.ndarray:
        return np.concatenate([state.channels[n].valid for n in ("body", "left_hand", "right_hand")], axis=1)

    def level(self, state, text) -> str:
        if not self.marks(state).any():
            return "blank"
        two, mono = self.truth[text]
        whole = bool(self.marks(state).all())
        if len(state.timestamps) == len(two.timestamps):
            cut = R.cut_frames(mono, np.arange(*CUT))
            marks = self.marks(state)
            if self.holds(state, two) and np.array_equal(marks, self.marks(two)):
                return "two_view"
            if self.holds(state, cut) and np.array_equal(marks, self.marks(cut)):
                return "trimmed"
            if self.holds(state, cut) and np.array_equal(marks, self.marks(two)):
                return "monocular_values_two_view_marks"
            if self.holds(state, two) and np.array_equal(marks, self.marks(cut)):
                return "two_view_values_monocular_marks"
            return "rt_two_view" if whole else "other"
        if self.holds(state, mono):
            return "nearest_monocular" if whole else ("monocular" if np.array_equal(self.marks(state), self.marks(mono)) else "other")
        return "round_trip" if whole else "other"

    def read(self, states, texts, candidates, group=2):
        rows = [self.levels[self.level(s, t)] for s, t in zip(states, texts)]
        return dict(hypotheses=[" ".join(t.split()[:k]) if k else "z y x w" for t, (k, _) in zip(texts, rows)], nll=np.array([[v] for _, v in rows]))


class Harness:
    def __init__(self, monkeypatch, tmp_path, *, declared=True, n=N, reader=None, levels=None, missing=(), no_state=(), unreadable=(), absent=(), mono=None):
        self.monkeypatch, self.tmp_path, self.declared, self.n = monkeypatch, tmp_path, declared, n
        self.records = d20_records(n) if declared else train_records(n)
        self.store = DirectoryStore(tmp_path / "s")
        self.forms = tmp_path / "w"
        self.tokenizer = tiny_tokenizer()
        rows, self.refs, self.mono = [], [], []
        for i, r in enumerate(self.records):
            state = (mono or monocular_state)(i)
            self.mono.append(state)
            digest = plant_state(self.store, tmp_path, r["sign"], state)
            self.refs.append(two_view_state(i, digest))
            if i in unreadable:
                digest = "0" * 64
            if i in absent:
                (self.store.root / R.state_key_of(r["sign"])).unlink()
            if i not in missing:
                rows.append(dict(sample_id=r["sign"], state_key=None if i in no_state else R.state_key_of(r["sign"]), state_sha256=digest, status="converted"))
        plant_shard(monkeypatch, self.store, rows)
        self.plant_two_view()
        truth = {r["text"]: (ref, state) for r, ref, state in zip(self.records, self.refs, self.mono)}
        self.reader = reader or ContentReader(truth, levels)
        monkeypatch.setattr(L, "open_read_list", lambda store, smoke: (self.records, declared and not smoke, dict(list="t", sha256=L.D20_SHA, clips=n)))
        monkeypatch.setattr(L, "load_states", lambda store, recs: self.refs)
        monkeypatch.setattr(L, "load_tokenizer", lambda store: self.tokenizer)
        monkeypatch.setattr(L, "make_reader", lambda store, **k: self.reader)
        monkeypatch.setattr(R, "load_template", lambda store: template_state())
        self.plant_stored()

    def plant_two_view(self):
        """Each two-view clip as stored: its clock is read back from here."""
        for k, (r, ref) in enumerate(zip(self.records, self.refs)):
            path = self.tmp_path / f"two-view-{k}-{len(list(self.tmp_path.glob('two-view-*')))}.npz"
            digest = ref.save(path)
            key = f"two-view/{r['sign']}.npz"
            self.store.put(key, path.read_bytes())
            r["continuous"] = dict(seconds=2., state_key=key, state_sha256=digest)

    def main(self, *argv):
        return R.main([argv[0], "--store", f"dir:{self.tmp_path / 's'}", *argv[1:]])

    def read(self, *forms, smoke=False, resume=False):
        args = ["read", "--out-dir", str(self.forms)]
        if forms:
            args += ["--forms", ",".join(forms)]
        if smoke:
            args += ["--smoke-train", str(min(self.n, R.SMOKE_CLIPS))]
        if resume:
            args.append("--resume")
        return self.main(*args)

    def stored_form(self, kind, name, keep, nll, change):
        records = d20_records(self.n)
        texts = [r["text"] for r in records]
        zeros = [0.] * self.n
        form = dict(kind=kind, form=name, report_only=True, declared_read=True, code_digest=f"{R.STORED_RT_DIGEST}aaaa", list=dict(sha256=L.D20_SHA), instrument=dict(sha256=L.TRANSLATOR_SHA), tokenizer=dict(sha256=L.TOKENIZER_SHA), environment=dict(gpu="stored"),
                    signs=[r["sign"] for r in records], references=texts, hypotheses=[" ".join(t.split()[:keep]) for t in texts], nll=[[nll] for _ in texts], motion=dict(wrist_speed=zeros, wrist_spread=zeros), hands={h: dict(speed=zeros, variation=zeros) for h in P.HANDS}, tracked=dict(wrists=zeros, hands=zeros))
        form.update(change)
        return form

    def plant_stored(self, *, rt=None, nearest=None):
        """V1's stored rt_nearest, as this run reads two-view's, and V0's stored v0_nearest_fill: every clip its first 11 words."""
        levels = self.reader.levels if hasattr(self.reader, "levels") else ContentReader.LEVELS
        rt_form = self.stored_form("tok-v1-form", "rt_nearest", levels["rt_two_view"][0], levels["rt_two_view"][1], rt or {})
        nearest_form = self.stored_form("tok-v0-form", "v0_nearest_fill", 11, 1.5, nearest or {})
        raw = json.dumps(nearest_form).encode()
        self.store.put(R.STORED_RT, json.dumps(rt_form).encode())
        self.store.put(R.STORED_NEAREST, raw)
        bleu, ci = M.bleu_ci_fast(rt_form["hypotheses"], rt_form["references"])
        self.monkeypatch.setattr(R, "STORED_RT_BLEU", bleu)
        self.monkeypatch.setattr(R, "STORED_RT_CI", ci)
        self.monkeypatch.setattr(R, "STORED_RT_SHA256", hashlib.sha256(json.dumps(rt_form).encode()).hexdigest())
        self.monkeypatch.setattr(R, "STORED_NEAREST_SHA256", hashlib.sha256(raw).hexdigest())
        self.monkeypatch.setattr(R, "STORED_NEAREST_BLEU", M.bleu_ci_fast(nearest_form["hypotheses"], nearest_form["references"])[0])
        self.monkeypatch.setattr(L, "D20_CLIPS", self.n)
        return rt_form, nearest_form

    def plant_control(self, *, hyp_change=0, nll_shift=0., bleu_shift=0., order=False, rt_hyp_change=0, rt_nll_shift=0.):
        """The pinned stored report, holding what real_two_view and blank read, changed as asked; the stored rt_nearest likewise."""
        read = {n: json.loads((self.forms / f"form-{n}.json").read_text()) for n in R.ANCHORS}
        real = list(read["real_two_view"]["hypotheses"])
        for i in range(hyp_change):
            real[i] += " changed"
        stored = dict(signs=list(reversed(read["real_two_view"]["signs"])) if order else read["real_two_view"]["signs"], references=read["real_two_view"]["references"],
                      forms=dict(real=dict(hypotheses=real, per_clip=dict(nll=[r[0] + nll_shift for r in read["real_two_view"]["nll"]])), blank=dict(hypotheses=read["blank"]["hypotheses"], per_clip=dict(nll=[r[0] for r in read["blank"]["nll"]]))))
        raw = json.dumps(stored).encode()
        self.store.put(P.CANARY_REPORT, raw)
        self.monkeypatch.setattr(P, "CANARY_REPORT_SHA", hashlib.sha256(raw).hexdigest())
        self.monkeypatch.setattr(P, "CANARY", dict(P.CANARY, real=M.bleu_ci_fast(read["real_two_view"]["hypotheses"], read["real_two_view"]["references"])[0] + bleu_shift, blank=M.bleu_ci_fast(read["blank"]["hypotheses"], read["blank"]["references"])[0]))
        trip = read["rt_nearest_two_view"]
        hypotheses = list(trip["hypotheses"])
        for i in range(rt_hyp_change):
            hypotheses[i] += " changed"
        self.plant_stored(rt=dict(hypotheses=hypotheses, nll=[[r[0] + rt_nll_shift] for r in trip["nll"]]))

    def control(self):
        return self.main("control", "--out-dir", str(self.forms), "--void-prefix", "pr/void")

    def merge(self, name="report.json"):
        out = self.tmp_path / name
        code = self.main("merge", "--out-dir", str(self.forms), "--output-file", str(out))
        assert code == 0
        return json.loads(out.read_text())

    def to_control(self):
        assert self.read(*R.ANCHORS) == 0
        self.plant_control()
        assert self.control() == 0

    def pipeline(self, name="report.json"):
        self.to_control()
        assert self.read(*R.MONOCULAR_FORMS, resume=True) == 0
        return self.merge(name)


# ---- what a lookup refuses --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("change", [dict(partition="test"), dict(partition="heldout_official"), dict(partition="heldout_official", signer_component=None), dict(partition="train", signer_component="c02"), dict(signer_component="c02"), dict(corpus="youtube-asl-v1")])
def test_a_declared_lookup_refuses_a_record_that_is_not_a_d20_kind_of_clip(change):
    records = d20_records(3)
    records[1].update(change)
    with pytest.raises(SystemExit, match="s1"):
        R.check_lookup_records(records, True)
    with pytest.raises(SystemExit, match="s1"):
        R.match_monocular(records, {r["sign"]: dict(state_key=R.state_key_of(r["sign"]), state_sha256="a" * 64) for r in records}, True)


@pytest.mark.parametrize("change", [dict(partition="val"), dict(partition="test"), dict(signer_component="c04"), dict(corpus="asllrp-dai2-plan-v2")])
def test_a_smoke_lookup_takes_c02_train_clips_only(change):
    records = train_records(3)
    records[2].update(change)
    with pytest.raises(SystemExit, match="s2"):
        R.check_lookup_records(records, False)


@pytest.mark.parametrize("declared,base", [(True, dict(partition="val")), (False, dict(partition="train")), (True, dict(partition="test")), (False, dict(partition="test"))])
def test_c00_is_refused_whatever_else_the_record_says(declared, base):
    records = (d20_records(2) if declared else train_records(2))
    records[0].update(base, signer_component="c00")
    with pytest.raises(SystemExit, match="c00"):
        R.check_lookup_records(records, declared)


def test_lists_of_the_kind_each_mode_expects_pass():
    R.check_lookup_records(d20_records(), True)
    R.check_lookup_records(train_records(), False)


@pytest.mark.parametrize("name", ["test_rgb_front_clips-shard-000-of-001.json", "val_rgb_front_clips-shard-000-of-001.json", "train_test_rgb_front_clips-shard-000-of-008.json", "contest_train_rgb_front_clips-shard-000-of-008.json", "train_rgb_front_clips-shard-000-of-008.json.bak", "test_rgb_front_clips-shard-000-of-008.json", "val_rgb_front_clips-shard-000-of-008.json", "train_rgb_front_clips-shard-000-of-001.json"])
def test_a_report_that_is_not_a_train_shard_is_refused_unread_even_when_pinned(monkeypatch, name):
    key = f"{R.REPORTS}{name}"
    raw = json.dumps(dict(results=[dict(sample_id="s0", state_key=R.state_key_of("s0"), state_sha256="a" * 64)])).encode()
    store = SpyStore({key: raw})
    monkeypatch.setattr(R, "MONOCULAR_REPORTS", {key: hashlib.sha256(raw).hexdigest()})
    with pytest.raises(SystemExit, match="refused unread"):
        R.monocular_rows(store, {"s0"})
    assert store.read == []


@pytest.mark.parametrize("key", [f"{R.REPORTS}train_rgb_front_clips-shard-000-of-008.json", f"other/{SHARD}", f"{R.MONOCULAR}/train_rgb_front_clips-shard-000-of-008.json"])
def test_an_unpinned_report_key_is_refused(monkeypatch, key):
    monkeypatch.setattr(R, "MONOCULAR_REPORTS", {})
    with pytest.raises(SystemExit, match="refused unread"):
        R.check_report_key(key)


def test_a_report_named_as_a_train_shard_outside_the_monocular_reports_is_refused_though_pinned(monkeypatch):
    key = "other/reports/train_rgb_front_clips-shard-000-of-008.json"
    monkeypatch.setattr(R, "MONOCULAR_REPORTS", {key: "a" * 64})
    with pytest.raises(SystemExit, match="refused unread"):
        R.check_report_key(key)


def test_a_key_of_the_same_length_outside_the_reports_prefix_is_refused_though_pinned(monkeypatch):
    key = f"{'x' * len(R.REPORTS)}train_rgb_front_clips-shard-000-of-008.json"
    monkeypatch.setattr(R, "MONOCULAR_REPORTS", {key: "a" * 64})
    with pytest.raises(SystemExit, match="refused unread"):
        R.check_report_key(key)


def test_the_pins_name_eight_train_shards_and_nothing_else():
    assert len(R.MONOCULAR_REPORTS) == 8 and all(re.fullmatch(r"[0-9a-f]{64}", s) for s in R.MONOCULAR_REPORTS.values())
    for key in R.MONOCULAR_REPORTS:
        assert key.startswith("acquisition/2026-10-17-v1/derived/how2sign-video-estimate-v1/reports/train_rgb_front_clips-shard-") and "test" not in key and "val_" not in key
        R.check_report_key(key)
    assert sorted(R.MONOCULAR_REPORTS)[0].endswith("shard-000-of-008.json") and sorted(R.MONOCULAR_REPORTS)[-1].endswith("shard-007-of-008.json")
    assert R.MONOCULAR_REPORTS[sorted(R.MONOCULAR_REPORTS)[0]].startswith("c4b5c02d") and R.MONOCULAR_REPORTS[sorted(R.MONOCULAR_REPORTS)[-1]].startswith("26361e93")


def test_the_rows_read_exactly_the_pinned_shards_and_never_list_a_prefix(monkeypatch):
    keys = {f"{R.REPORTS}train_rgb_front_clips-shard-{i:03d}-of-008.json": json.dumps(dict(results=[dict(sample_id=f"s{i}", state_key=R.state_key_of(f"s{i}"), state_sha256="a" * 64, status="converted")])).encode() for i in range(2)}
    test_key = f"{R.REPORTS}test_rgb_front_clips-shard-000-of-001.json"
    store = SpyStore({**keys, test_key: b"never"})
    monkeypatch.setattr(R, "MONOCULAR_REPORTS", {k: hashlib.sha256(v).hexdigest() for k, v in keys.items()})
    rows = R.monocular_rows(store, {"s0", "s1"})
    assert set(rows) == {"s0", "s1"} and sorted(store.read) == sorted(keys) and test_key not in store.read and not hasattr(store, "list")
    assert rows["s1"]["shard"] == "train_rgb_front_clips-shard-001-of-008.json" and rows["s1"]["state_key"] == R.state_key_of("s1") and rows["s1"]["status"] == "converted"


def test_a_changed_report_refuses(monkeypatch):
    store = SpyStore({SHARD: b'{"results": []}'})
    monkeypatch.setattr(R, "MONOCULAR_REPORTS", {SHARD: "0" * 64})
    with pytest.raises(SystemExit, match="SHA-256"):
        R.monocular_rows(store, {"s0"})


def test_rows_keep_only_exact_signs_never_a_substring(monkeypatch):
    near = ["xs1", "s1x", "s1-rgb_front", "S1", " s1", "s11", "1", "s", "--s1"]
    raw = json.dumps(dict(results=[dict(sample_id=x, state_key=R.state_key_of(x), state_sha256="a" * 64) for x in near])).encode()
    monkeypatch.setattr(R, "MONOCULAR_REPORTS", {SHARD: hashlib.sha256(raw).hexdigest()})
    assert R.monocular_rows(SpyStore({SHARD: raw}), {"s1"}) == {}
    found, lost = R.match_monocular(d20_records(2), R.monocular_rows(SpyStore({SHARD: raw}), {"s0", "s1"}), True)
    assert found == [] and [(x["sign"], x["reason"]) for x in lost] == [("s0", "no_monocular_row"), ("s1", "no_monocular_row")]


def test_a_match_takes_a_row_only_under_the_records_exact_sign():
    rows = {x: dict(state_key=R.state_key_of("s1"), state_sha256="a" * 64) for x in ("xs1", "s1x", "s11", "S1")}
    found, lost = R.match_monocular(d20_records(2)[1:], rows, True)
    assert found == [] and lost == [dict(sign="s1", reason="no_monocular_row", detail=None)]


def test_a_rows_substring_key_for_a_longer_sign_is_not_this_clips_state():
    rows = dict(s1=dict(state_key=f"{R.STATES}s1-other.npz", state_sha256="a" * 64))
    with pytest.raises(SystemExit, match="refused unread"):
        R.match_monocular(d20_records(2)[1:], rows, True)


@pytest.mark.parametrize("key", [f"{R.MONOCULAR}/test-states/s0.npz", "acquisition/2026-10-17-v1/derived/full-v1/how2sign-two-view-v1/states/s0.npz", f"{R.STATES}s0.npz.bak", f"{R.STATES}s00.npz", f"x{R.STATES}s0.npz"])
def test_a_row_naming_any_other_object_never_reaches_the_store(key):
    with pytest.raises(SystemExit, match="refused unread"):
        R.match_monocular(d20_records(1), dict(s0=dict(state_key=key, state_sha256="a" * 64)), True)
    store = SpyStore({key: b"never"})
    with pytest.raises(SystemExit, match="refused"):
        R.load_clip(store, dict(index=0, sign="s0", state_key=key, state_sha256="a" * 64), template_state(), two_view_state(0, "a" * 64), np.arange(20) / 24.)
    assert store.read == []


def test_the_exact_row_matches_in_list_order_and_unmatched_clips_are_listed():
    rows = {f"s{i}": dict(state_key=R.state_key_of(f"s{i}"), state_sha256=f"{i}" * 64) for i in (0, 2, 3)}
    rows["s3"] = dict(state_key=None, state_sha256=None, status="failed")
    found, lost = R.match_monocular(d20_records(4), rows, True)
    assert [(f["index"], f["sign"], f["state_sha256"]) for f in found] == [(0, "s0", "0" * 64), (2, "s2", "2" * 64)] and found[0]["state_key"] == R.state_key_of("s0")
    assert lost == [dict(sign="s1", reason="no_monocular_row", detail=None), dict(sign="s3", reason="no_state_in_row", detail="failed")]


def test_a_sign_with_two_different_states_refuses_and_a_repeat_of_one_does_not(monkeypatch):
    row = dict(sample_id="s0", state_key=R.state_key_of("s0"), state_sha256="a" * 64)
    raw = json.dumps(dict(results=[row, row])).encode()
    monkeypatch.setattr(R, "MONOCULAR_REPORTS", {SHARD: hashlib.sha256(raw).hexdigest()})
    assert R.monocular_rows(SpyStore({SHARD: raw}), {"s0"})["s0"]["state_sha256"] == "a" * 64
    raw = json.dumps(dict(results=[row, dict(row, state_sha256="b" * 64)])).encode()
    monkeypatch.setattr(R, "MONOCULAR_REPORTS", {SHARD: hashlib.sha256(raw).hexdigest()})
    with pytest.raises(SystemExit, match="two different"):
        R.monocular_rows(SpyStore({SHARD: raw}), {"s0"})


@pytest.mark.parametrize("first,message", [(dict(sign="other", signer_component="c02"), "not --7E2"), (dict(sign=R.TEMPLATE_SIGN, signer_component="c00"), "not --7E2")])
def test_the_template_is_the_named_clip_of_a_train_signer(monkeypatch, first, message):
    records = [dict(first, corpus=L.H2S, partition="train", continuous=dict(state_key="k", state_sha256="a" * 64))]
    monkeypatch.setattr(L, "load_manifest_at", lambda store, key, sha: dict(records=records))
    store = SpyStore({})
    with pytest.raises(SystemExit, match=message):
        R.load_template(store)
    assert store.read == []


def test_the_template_skips_other_corpora_and_partitions_and_loads_by_hash(monkeypatch, tmp_path):
    store = DirectoryStore(tmp_path / "s")
    digest = plant_state(store, tmp_path, "t", signer_state(5, frames=4, amplitude=.4))
    records = [dict(sign="a", corpus="other", partition="train"), dict(sign="b", corpus=L.H2S, partition="val"), dict(sign=R.TEMPLATE_SIGN, corpus=L.H2S, partition="train", signer_component="c02", continuous=dict(state_key=R.state_key_of("t"), state_sha256=digest))]
    seen = []
    monkeypatch.setattr(L, "load_manifest_at", lambda s, key, sha: seen.append((key, sha)) or dict(records=records))
    assert len(R.load_template(store).timestamps) == 4 and seen == [(L.EVAL_MIX, L.EVAL_MIX_SHA)]
    records[2]["continuous"]["state_sha256"] = "0" * 64
    with pytest.raises(Exception, match="(?i)sha"):
        R.load_template(store)


def test_no_template_record_refuses(monkeypatch):
    monkeypatch.setattr(L, "load_manifest_at", lambda store, key, sha: dict(records=[dict(sign="b", corpus=L.H2S, partition="val")]))
    with pytest.raises(SystemExit, match="template"):
        R.load_template(SpyStore({}))


def code_without_docstrings() -> str:
    """The script's code: comments and docstrings can name what the code refuses."""
    source = (ROOT / "scripts/path_read.py").read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node) is not None:
            first = node.body[0]
            for i in range(first.lineno - 1, first.end_lineno):
                lines[i] = ""
    return re.sub(r"\b[0-9a-f]{64}\b", "", "\n".join(line.split("  #")[0] for line in lines))     # a hash can hold any short word


def test_no_code_path_names_another_list_a_generated_form_or_a_test_report():
    code = code_without_docstrings()
    for word in ("c20", "C20", "val1000", "eval1000", "composed", "compose", "word_cache", "a9p", "test_rgb", "val_rgb", ".list(", "uwb-ft", "state-t5", "v2"):
        assert word not in code, word
    assert "load_read_list" not in code and "L.open_read_list" in code


def test_no_command_has_an_option_for_another_list_instrument_or_tokenizer(capsys):
    for command in ("read", "lookups", "merge", "control", "preflight"):
        parser = R.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([command, "--help"])
        text = capsys.readouterr().out
        for flag in ("--sentences", "--translator", "--instrument-run", "--tokenizer", "--manifest", "--list", "--clips", "--report-key", "--template", "--states"):
            assert flag not in text


def test_s2_lib_still_refuses_c20_before_reading_it():
    key = f"{L.O}/round20/c20.json"
    store = SpyStore({key: b"{}", L.D20_KEY: b"{}"})
    with pytest.raises(SystemExit, match="D20 only"):
        L.load_read_list(store, dict(records=[]), key)
    assert store.read == []


@pytest.mark.parametrize("command", ["read", "lookups"])
def test_a_smoke_is_five_clips_at_most(monkeypatch, tmp_path, command):
    Harness(monkeypatch, tmp_path, declared=False)
    assert R.SMOKE_CLIPS == 5
    with pytest.raises(SystemExit, match="at most 5"):
        R.main([command, "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w"), "--smoke-train", "6"])


def test_lookups_read_no_state_and_list_what_is_lost(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, tmp_path, missing=(1,), no_state=(2,))
    monkeypatch.setattr(L, "load_states", lambda *a: pytest.fail("a state was read"))
    monkeypatch.setattr(R, "load_clip", lambda *a: pytest.fail("a state was read"))
    assert h.main("lookups", "--out-dir", str(h.forms)) == 0
    report = json.loads((h.forms / "lookups.json").read_text())
    assert report["clips"] == N and report["matched"] == N - 2 and [x["sign"] for x in report["lost"]] == ["s1", "s2"] and report["report_only"] is True and report["declared_read"] is True
    assert report["signers"] == dict(c01=3, c03=2, c05=2) and report["shards"] == {SHARD[len(R.REPORTS):]: N - 2} and report["reports_opened"] == [SHARD] and report["states"][0] == R.state_key_of("s0") and len(report["states"]) == N - 2
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["matched"] == N - 2 and printed["lost"] == 2
    before = (h.forms / "lookups.json").read_bytes()
    assert h.main("lookups", "--out-dir", str(h.forms)) == 0 and (h.forms / "lookups.json").read_bytes() == before


def test_lookups_are_idempotent_only_for_the_same_answer(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=(1,))
    assert h.main("lookups", "--out-dir", str(h.forms)) == 0
    path = h.forms / "lookups.json"
    good = json.loads(path.read_text())
    for change in (dict(matched=0), dict(clips=1), dict(lost=[]), dict(states=[]), dict(declared_read=False), dict(code_digest="other")):
        path.write_text(json.dumps(dict(good, **change)))
        with pytest.raises(SystemExit, match="another lookup"):
            h.main("lookups", "--out-dir", str(h.forms))


# ---- a monocular clip -------------------------------------------------------------------------------------------------------------

def item_of(store, tmp_path, state, sign="s0", index=0):
    return dict(index=index, sign=sign, state_key=R.state_key_of(sign), state_sha256=plant_state(store, tmp_path, sign, state))


def test_the_real_equivalence_conforms_a_clip_to_the_two_view_layout_with_its_values_and_marks_kept(tmp_path):
    store = DirectoryStore(tmp_path / "s")
    raw = monocular_state(0)
    item = item_of(store, tmp_path, raw)
    assert ((TWO_VIEW), {MONOCULAR}) == next(iter(layout_adapter.EQUIVALENT.items()))
    ref = two_view_state(0, item["state_sha256"])
    clip, why = R.load_clip(store, item, template_state(), ref, ref.timestamps)
    assert why is None and clip.index == 0 and clip.sign == "s0"
    assert R.layout_of(clip.conformed) == R.layout_of(ref) and R.layout_of(clip.raw) != R.layout_of(ref) and set(R.layout_of(clip.raw)) == {"body", "left_hand", "right_hand"}
    for name in ("body", "left_hand", "right_hand"):
        np.testing.assert_array_equal(clip.conformed.channels[name].values, raw.channels[name].values)
        np.testing.assert_array_equal(clip.conformed.channels[name].valid, raw.channels[name].valid)
    assert len(clip.conformed.timestamps) == 24 and len(clip.raw.timestamps) == 24


def test_a_clip_is_cut_to_its_two_view_frames_by_timestamp(tmp_path):
    store = DirectoryStore(tmp_path / "s")
    raw = monocular_state(0)
    item = item_of(store, tmp_path, raw)
    ref = two_view_state(0, item["state_sha256"])
    clip, _ = R.load_clip(store, item, template_state(), ref, ref.timestamps)
    assert len(clip.trimmed.timestamps) == 20 and clip.trimmed.timestamps[0] == pytest.approx(ref.timestamps[0]) and clip.trimmed.timestamps[-1] == pytest.approx(ref.timestamps[-1])
    np.testing.assert_array_equal(clip.trimmed.channels["body"].values, raw.channels["body"].values[2:22])
    np.testing.assert_array_equal(clip.trimmed.channels["left_hand"].valid, raw.channels["left_hand"].valid[2:22])
    assert clip.slice_of == dict(start_frame=2, frames_two_view=20, frames_monocular=24, stored_fps=24., max_offset_seconds=0., source_hash_matches=True)


def test_the_two_view_clips_source_hash_proves_the_pairing_on_the_stored_files(tmp_path):
    store = DirectoryStore(tmp_path / "s")
    item = item_of(store, tmp_path, monocular_state(0))
    ref = two_view_state(0, "b" * 64)
    clip, _ = R.load_clip(store, item, template_state(), ref, ref.timestamps)
    assert clip.slice_of["source_hash_matches"] is False


@pytest.mark.parametrize("times", [np.arange(20) / 24. + 100., np.arange(20) / 12.])
def test_a_two_view_clip_that_is_no_slice_of_its_monocular_clip_stops_the_job(tmp_path, times):
    store = DirectoryStore(tmp_path / "s")
    item = item_of(store, tmp_path, monocular_state(0))
    ref = replace(two_view_state(0, item["state_sha256"]), timestamps=times)
    with pytest.raises(SystemExit, match="not a slice"):
        R.load_clip(store, item, template_state(), ref, times)


def stopped_clip_clocks():
    """The clip that stopped the first run, as stored: 30 fps, the monocular clock from 0.03301, two-view from its second frame."""
    monocular = .03301 + np.arange(167) / 30.
    return monocular, monocular[1:]


def thirty_fps_pair(tmp_path, monocular, two_view):
    """The two states at those clocks, conformed and resampled the way a read takes them."""
    from evaluate_text_to_motion import at_rate
    store = DirectoryStore(tmp_path / "s")
    mono = replace(monocular_state(0, frames=len(monocular)), timestamps=monocular)
    item = item_of(store, tmp_path, mono)
    start = int(np.argmin(np.abs(monocular - two_view[0])))
    base = R.cut_frames(signer_state(100, frames=len(monocular), amplitude=.4), np.arange(start, start + len(two_view)))
    stored = laid_out(replace(base, timestamps=two_view, channels={n: replace(c, source_sha256=item["state_sha256"]) if c.valid.any() else c for n, c in base.channels.items()}), TWO_VIEW)
    return store, item, at_rate(stored, R.FPS)


def test_the_clip_that_stopped_the_first_run_passes_on_its_stored_clocks_and_trims_onto_two_views_grid(tmp_path):
    from evaluate_text_to_motion import at_rate
    monocular, two_view = stopped_clip_clocks()
    store, item, ref = thirty_fps_pair(tmp_path, monocular, two_view)
    full_grid = at_rate(replace(monocular_state(0, frames=len(monocular)), timestamps=monocular), R.FPS).timestamps
    assert R.trim_index(full_grid, ref.timestamps)[1] > .03          # what stopped it: the two grids differ in phase after resampling
    clip, why = R.load_clip(store, item, template_state(), ref, two_view)
    assert why is None and clip.slice_of["max_offset_seconds"] == 0. and clip.slice_of["start_frame"] == 1 and clip.slice_of["stored_fps"] == 30. and clip.slice_of["source_hash_matches"] is True
    np.testing.assert_array_equal(clip.trimmed.timestamps, ref.timestamps)
    assert R.layout_of(clip.trimmed) == R.layout_of(ref) and R.layout_of(clip.raw) != R.layout_of(ref)
    assert len(clip.trimmed.timestamps) == 133 and R.key_distance(clip.trimmed, ref)["left_wrist"] is not None
    assert clip.slice_of["frames_monocular"] == len(full_grid)


def test_an_offset_just_inside_a_quarter_of_a_stored_frame_passes_and_is_reported_and_just_above_it_stops(tmp_path):
    monocular, two_view = stopped_clip_clocks()
    quarter = 1 / 120.
    inside = two_view[:40] + quarter - 1e-4
    store, item, ref = thirty_fps_pair(tmp_path, monocular, inside)
    clip, why = R.load_clip(store, item, template_state(), ref, inside)
    assert why is None and clip.slice_of["max_offset_seconds"] == pytest.approx(quarter - 1e-4, abs=1e-6) and clip.slice_of["start_frame"] == 1
    np.testing.assert_array_equal(clip.trimmed.timestamps, ref.timestamps)
    with pytest.raises(SystemExit, match="not a slice of its monocular clip on the stored clocks: largest offset 0.0084"):
        R.load_clip(store, item, template_state(), ref, two_view[:40] + quarter + 1e-4)


@pytest.mark.parametrize("frames", [.26, .4, .5, .6, .74])
def test_a_clock_off_the_frame_grid_stops_though_its_nearest_frame_is_within_half_a_frame(frames):
    times = np.arange(60) / 30.
    index, offset = R.trim_index(times, times[5:15] + frames / 30.)
    assert index is None and offset == pytest.approx(min(frames, 1 - frames) / 30.)


def test_the_tolerance_is_this_clips_own_frame_not_24_fps():
    assert R.SLICE_TOLERANCE == .25
    fast, slow = np.arange(60) / 60., np.arange(60) / 24.
    assert R.trim_index(fast, fast[5:15] + .004)[0] is not None and R.trim_index(fast, fast[5:15] + .0043)[0] is None
    assert R.trim_index(slow, slow[5:15] + .0103)[0] is not None and R.trim_index(slow, slow[5:15] + .0105)[0] is None


def test_a_clock_one_stored_frame_past_the_end_stops(tmp_path):
    monocular, two_view = stopped_clip_clocks()
    index, offset = R.trim_index(monocular, np.r_[two_view, two_view[-1] + 1 / 30.])
    assert index is None and offset == pytest.approx(1 / 30.)


def test_a_cut_that_does_not_land_on_its_two_view_clips_length_stops(tmp_path):
    store = DirectoryStore(tmp_path / "s")
    item = item_of(store, tmp_path, monocular_state(0))
    ref = two_view_state(0, item["state_sha256"])
    with pytest.raises(SystemExit, match="holds 19 frames, its two-view clip 20"):
        R.load_clip(store, item, template_state(), ref, ref.timestamps[:19])


def test_the_stored_clock_is_read_from_the_clips_own_two_view_state(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    seen = []
    original = R.load_clip
    monkeypatch.setattr(R, "load_clip", lambda store, item, template, ref, times: seen.append((item["sign"], np.asarray(times).copy())) or original(store, item, template, ref, times))
    R.open_monocular(h.store, h.records, h.refs, True)
    for sign, times in seen:
        np.testing.assert_array_equal(times, h.refs[int(sign[1:])].timestamps)
    h.store.put(h.records[1]["continuous"]["state_key"], b"changed")
    with pytest.raises(Exception, match="(?i)sha"):
        R.open_monocular(h.store, h.records, h.refs, True)


def test_trim_index_finds_the_frames_and_their_offset_and_refuses_what_is_not_contained():
    times = np.arange(24) / 24.
    index, offset = R.trim_index(times, times[2:22])
    np.testing.assert_array_equal(index, np.arange(2, 22))
    assert offset == 0.
    near, offset = R.trim_index(times, times[2:22] + .01)
    np.testing.assert_array_equal(near, np.arange(2, 22))
    assert offset == pytest.approx(.01)
    assert R.trim_index(times, np.array([-1 / 24, 0., 1 / 24]))[0] is None and R.trim_index(times, np.array([22 / 24, 23 / 24, 24 / 24]))[0] is None
    assert R.trim_index(times, np.array([0., 2 / 24]))[0] is None
    index, offset = R.trim_index(times, np.array([100.]))
    assert index is None and offset == pytest.approx(100. - 23 / 24)
    near, _ = R.trim_index(times, np.array([0., 1 / 24]))
    np.testing.assert_array_equal(near, [0, 1])


def test_a_clip_at_another_rate_comes_to_24_fps_conformed_and_raw(tmp_path):
    monocular, two_view = stopped_clip_clocks()
    store, item, ref = thirty_fps_pair(tmp_path, monocular, two_view)
    clip, _ = R.load_clip(store, item, template_state(), ref, two_view)
    assert len(clip.conformed.timestamps) == len(clip.raw.timestamps) and abs(np.median(np.diff(clip.conformed.timestamps)) - 1 / 24) < 1e-6 and abs(np.median(np.diff(clip.raw.timestamps)) - 1 / 24) < 1e-6


def test_a_missing_object_alone_loses_a_clip(tmp_path):
    item = dict(index=0, sign="s0", state_key=R.state_key_of("s0"), state_sha256="0" * 64)
    clip, why = R.load_clip(DirectoryStore(tmp_path / "empty"), item, template_state(), two_view_state(0, "0" * 64), np.arange(20) / 24.)
    assert clip is None and why == dict(sign="s0", reason="state_missing", detail=R.state_key_of("s0"))


def test_a_hash_mismatch_stops_the_job_and_loses_nothing(tmp_path):
    store = DirectoryStore(tmp_path / "s")
    item = dict(item_of(store, tmp_path, monocular_state(0)), state_sha256="0" * 64)
    with pytest.raises(SystemExit, match="does not load"):
        R.load_clip(store, item, template_state(), two_view_state(0, "0" * 64), np.arange(20) / 24.)


@pytest.mark.parametrize("error", [MemoryError("planted"), ConnectionError("planted"), OSError("planted")])
def test_a_network_or_memory_error_stops_the_job_and_loses_nothing(monkeypatch, tmp_path, error):
    h = Harness(monkeypatch, tmp_path, n=3)

    class Broken(DirectoryStore):
        def get(self, key):
            if key == R.state_key_of("s1"):
                raise error
            return super().get(key)
    with pytest.raises(type(error), match="planted"):
        R.open_monocular(Broken(tmp_path / "s"), h.records, h.refs, True)


def test_an_exists_that_fails_is_not_a_missing_object(tmp_path):
    class Broken(DirectoryStore):
        def exists(self, key):
            raise ConnectionError("planted")
    item = dict(index=0, sign="s0", state_key=R.state_key_of("s0"), state_sha256="0" * 64)
    with pytest.raises(ConnectionError):
        R.load_clip(Broken(tmp_path / "s"), item, template_state(), two_view_state(0, "0" * 64), np.arange(20) / 24.)


def test_an_undeclared_layout_stops_the_job(tmp_path):
    store = DirectoryStore(tmp_path / "s")
    item = item_of(store, tmp_path, monocular_state(0, layout=("undeclared", "undeclared")))
    with pytest.raises(SystemExit, match="conform refused it: ValueError"):
        R.load_clip(store, item, template_state(), two_view_state(0, item["state_sha256"]), np.arange(2, 22) / 24.)


def test_a_conform_missing_a_channel_stops_the_job(monkeypatch, tmp_path):
    store = DirectoryStore(tmp_path / "s")
    item = item_of(store, tmp_path, monocular_state(0))

    def missing(state, template):
        raise KeyError("left_hand")
    monkeypatch.setattr(layout_adapter, "conform", missing)
    with pytest.raises(SystemExit, match="conform refused it: KeyError"):
        R.load_clip(store, item, template_state(), two_view_state(0, item["state_sha256"]), np.arange(2, 22) / 24.)


def test_a_clip_with_one_frame_stops_the_job_and_two_do_not(tmp_path):
    store = DirectoryStore(tmp_path / "s")
    item = item_of(store, tmp_path, monocular_state(0, frames=1))
    with pytest.raises(SystemExit, match="1 frames"):
        R.load_clip(store, item, template_state(), two_view_state(0, item["state_sha256"]), np.arange(2, 22) / 24.)
    item = item_of(store, tmp_path, monocular_state(0, frames=2))
    ref = replace(R.cut_frames(two_view_state(0, item["state_sha256"]), np.arange(2)), timestamps=np.arange(2) / 24.)
    assert R.load_clip(store, item, template_state(), ref, ref.timestamps)[0] is not None


def test_a_conformed_clip_whose_read_channels_are_not_the_two_view_clips_stops_the_job(tmp_path):
    store = DirectoryStore(tmp_path / "s")
    item = item_of(store, tmp_path, monocular_state(0))
    ref = two_view_state(0, item["state_sha256"])
    other = replace(ref, channels={**ref.channels, "body": replace(ref.channels["body"], convention="other")})
    with pytest.raises(SystemExit, match="layout"):
        R.load_clip(store, item, template_state(), other, other.timestamps)


def test_an_unavailable_channel_named_differently_in_the_two_view_clip_loses_nothing(tmp_path):
    """v1 never reads it, and a two-view clip can hold a placeholder where the template holds data."""
    store = DirectoryStore(tmp_path / "s")
    item = item_of(store, tmp_path, monocular_state(0))
    ref = two_view_state(0, item["state_sha256"])
    name = next(n for n, c in ref.channels.items() if not c.valid.any())
    other = replace(ref, channels={**ref.channels, name: replace(ref.channels[name], labels=("placeholder",), coordinate_frame="not-estimated")})
    clip, why = R.load_clip(store, item, template_state(), other, other.timestamps)
    assert why is None and clip is not None


def test_a_clip_is_checked_against_its_own_two_view_clip(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    ref = h.refs[1]
    h.refs[1] = replace(ref, channels={**ref.channels, "body": replace(ref.channels["body"], convention="other")})
    with pytest.raises(SystemExit, match="s1: conformed"):
        R.open_monocular(h.store, h.records, h.refs, True)


def test_no_usable_clip_refuses(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=range(N))
    with pytest.raises(SystemExit, match="no clip has a usable monocular state"):
        R.open_monocular(h.store, h.records, h.refs, True)


def test_open_monocular_keeps_list_order_and_reasons(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=(1,), absent=(4,), no_state=(6,))
    clips, lost, template = R.open_monocular(h.store, h.records, h.refs, True)
    assert [c.index for c in clips] == [0, 2, 3, 5, 7, 8] and {x["sign"]: x["reason"] for x in lost} == dict(s1="no_monocular_row", s4="state_missing", s6="no_state_in_row") and len(template.timestamps) == 4
    assert clips[2].slice_of["frames_two_view"] == 20 and clips[2].sign == "s3" and np.array_equal(clips[2].raw.channels["body"].values, h.mono[3].channels["body"].values)


# ---- through the real builders ----------------------------------------------------------------------------------------------------

def built_pair(frames=40, cut=(0, 40)):
    """The same 3D points through the real monocular builder and the real two-view builder (the review's scratch check): a real layout on each side."""
    from signtranslator.data_engineering.sapiens_keypoints import BODY_TO_POSE33, HAND21
    from signtranslator.data_engineering.two_view_states import rotation_inputs, triangulated_state
    from signtranslator.data_engineering.video_landmarks import ClipLandmarks, canonical_from_landmarks, solve_rotations
    rng = np.random.default_rng(0)
    t = np.arange(frames) / 24.
    pose = np.zeros((frames, 33, 3))
    base = {11: (.18, -.45, 0.), 12: (-.18, -.45, 0.), 13: (.3, -.25, -.05), 14: (-.3, -.25, -.05), 15: (.25, -.3, -.25), 16: (-.2, -.1, -.25), 23: (.1, 0., 0.), 24: (-.1, 0., 0.), 25: (.1, .4, 0.), 26: (-.1, .4, 0.), 27: (.1, .8, 0.), 28: (-.1, .8, 0.), 0: (0., -.65, -.05)}
    for k, v in base.items():
        pose[:, k] = np.asarray(v) + .02 * np.sin(2 * np.pi * (t * (1 + k / 10.)))[:, None] * rng.normal(size=3)
    shape = np.zeros((21, 3))
    for finger in range(5):
        for seg in range(4):
            shape[1 + 4 * finger + seg] = (.03 * (finger - 2), -.05 - .025 * seg, -.005 * seg)
    hands = np.zeros((frames, 2, 21, 3))
    hands_image = np.zeros((frames, 2, 21, 2))
    pose_image = np.zeros((frames, 33, 2))
    pose_image[:] = np.asarray([.5, .5])
    pose_image[:, 15] = (.65, .5)
    pose_image[:, 16] = (.35, .5)
    for h, wrist in enumerate((15, 16)):
        hands[:, h] = shape + .01 * np.sin(2 * np.pi * t * (2 + h))[:, None, None] * rng.normal(size=(21, 3)) + (.01 * h) * rng.normal(size=(21, 3))
        hands_image[:, h, :] = pose_image[:, [wrist]]
    clip = ClipLandmarks(times=t, pose_world=pose, pose_visibility=np.ones((frames, 33)), pose_image=pose_image, pose_found=np.ones(frames, bool), hands_world=hands, hands_image=hands_image, hands_found=np.ones((frames, 2), bool),
                         blendshapes=np.full((frames, 3), .2), blendshape_names=("eyeBlinkLeft", "eyeBlinkRight", "jawOpen"), face_matrix=np.broadcast_to(np.eye(4), (frames, 4, 4)).copy(), face_found=np.ones(frames, bool), roi=True)
    mono, _ = canonical_from_landmarks(clip, sample_id="x", source_id="m", source_sha256="a" * 64, clock_id="c", inference_method="mono")
    points = np.full((frames, 308, 3), np.nan)
    for goliath, k in BODY_TO_POSE33.items():
        points[:, goliath] = pose[:, k]
    for h, side in enumerate(("Left", "Right")):
        points[:, list(HAND21[side])] = pose[:, [15 + h]] + (hands[:, h] - hands[:, h, [0]])
    usable = np.isfinite(points).all(-1)
    rotations, hand_rotations = solve_rotations(*(lambda x: (x["world"], x["visible"], x["hands"]))(rotation_inputs(points[cut[0]:cut[1]], usable[cut[0]:cut[1]])))
    return mono, triangulated_state(mono, slice(*cut), rotations, hand_rotations, source_id="t", source_sha256="b" * 64, inference_method="tv")


def test_the_real_builders_layouts_conform_through_the_real_equivalence_and_v1s_input_does_not_move():
    mono, two = built_pair()
    assert mono.channels["body"].coordinate_frame == MONOCULAR[0] and mono.channels["body"].convention == MONOCULAR[1] and two.channels["body"].coordinate_frame == TWO_VIEW[0] and two.channels["body"].convention == TWO_VIEW[1]
    done = layout_adapter.conform(mono, two)
    assert R.layout_of(done) == R.layout_of(two)
    np.testing.assert_array_equal(R.reader_input(mono), R.reader_input(done))
    assert R.unconformed_plan([R.Clip(0, "x", mono, done, done, {})])["read"] is False
    distance = R.key_distance(done, two)
    assert all(v is not None and v < .05 for k, v in distance.items())


def test_the_real_builders_clip_is_a_slice_by_timestamp():
    mono, two = built_pair(cut=(5, 35))
    index, offset = R.trim_index(mono.timestamps, two.timestamps)
    np.testing.assert_array_equal(index, np.arange(5, 35))
    assert offset < 1e-9


def conformed_built():
    mono, two = built_pair()
    return layout_adapter.conform(mono, two), two


def test_a_mirrored_clip_sits_farther_than_the_clip_itself():
    from signtranslator.pose.mirror import mirror_state
    done, two = conformed_built()
    here, mirrored = R.key_distance(done, two), R.key_distance(mirror_state(done), two)
    for group in R.MIRRORED_GROUPS:
        assert R.group_distance(mirrored, group) > R.group_distance(here, group), group


def with_empty_hand(state, hand="left_hand"):
    c = state.channels[hand]
    return replace(state, channels={**state.channels, hand: replace(c, values=np.zeros_like(c.values), valid=np.zeros_like(c.valid), observed=np.zeros_like(c.observed), confidence=np.zeros_like(c.confidence), inference_method=None)})


def with_swapped_hands(state):
    """Each hand's rotations under the other hand's labels: a left and right swapped in the extraction."""
    left, right = state.channels["left_hand"], state.channels["right_hand"]
    return replace(state, channels={**state.channels, "left_hand": replace(left, values=right.values, valid=right.valid, observed=right.observed, confidence=right.confidence), "right_hand": replace(right, values=left.values, valid=left.valid, observed=left.observed, confidence=left.confidence)})


def pairing_of(clip_state, two, frames=None):
    from signtranslator.pose.mirror import mirror_state
    return [dict(sign="x", start_frame=0, frames_two_view=len(two.timestamps), frames_monocular=len(two.timestamps), max_offset_seconds=0., source_hash_matches=True, distance=R.key_distance(clip_state, two), mirrored=R.key_distance(mirror_state(clip_state), two))]


def validity_rows(state):
    return {part: [v] for part, v in R.part_validity(state).items()}


def alerts_of(state, two, shared=500, declared=True):
    rows = pairing_of(state, two)
    block = dict(clips=1, source_hash_matches=1, max_offset_seconds=0., group_distance={g: R.group_distance(rows[0]["distance"], g) for g in R.GROUPS}, group_mirrored={g: R.group_distance(rows[0]["mirrored"], g) for g in R.GROUPS})
    return R.plumbing_alerts(validity_rows(two), validity_rows(state), block, shared, declared), block


def test_each_part_reads_its_own_points(monkeypatch):
    import signtranslator.data.keypoints2d as keypoints
    valid = np.zeros((10, 48), bool)
    valid[:5, 0] = True                   # one shoulder on half the frames
    valid[:2, 2] = True                   # the left elbow, not the right
    valid[:3, 5] = True                   # the right wrist, not the left
    valid[:, 6:16] = True                 # 10 of the left hand's 21 points
    valid[:, 47] = True                   # 1 of the right hand's
    monkeypatch.setattr(keypoints, "from_state", lambda state: (np.zeros((10, 48, 2)), valid))
    assert R.part_validity(None) == dict(shoulders=.25, left_elbow=.2, right_elbow=0., left_wrist=0., right_wrist=.3, left_hand=round(10 / 21, 5), right_hand=round(1 / 21, 5))


def test_a_clean_pair_raises_no_alert():
    done, two = conformed_built()
    assert alerts_of(done, two)[0] == []


def test_an_emptied_hand_surfaces_through_validity_and_the_alert():
    done, two = conformed_built()
    empty = with_empty_hand(done)
    assert R.part_validity(empty)["left_hand"] == 0. and R.part_validity(empty)["right_hand"] == 1. and R.part_validity(empty)["left_wrist"] == 1.
    assert R.key_distance(empty, two)["left_hand"] is None
    alerts, _ = alerts_of(empty, two)
    assert any("monocular left_hand valid on 0.000" in a for a in alerts) and any("left_hand validity is 0.000 of two-view's" in a for a in alerts)


def test_a_swapped_hand_surfaces_through_distance():
    done, two = conformed_built()
    swapped = with_swapped_hands(done)
    assert R.part_validity(swapped) == R.part_validity(done)
    assert R.key_distance(swapped, two)["left_hand"] > 5 * R.key_distance(done, two)["left_hand"] and R.key_distance(swapped, two)["right_hand"] > 5 * R.key_distance(done, two)["right_hand"]


def test_a_clip_mirrored_in_extraction_raises_the_negative_controls_alert():
    from signtranslator.pose.mirror import mirror_state
    done, two = conformed_built()
    alerts, block = alerts_of(mirror_state(done), two)
    assert any("mirrored monocular clip sits as near" in a for a in alerts) and block["group_mirrored"]["wrists"] < block["group_distance"]["wrists"]


def test_a_chest_with_no_frames_empties_the_arms_for_v1_and_raises_an_alert():
    done, two = conformed_built()
    body = done.channels["body"]
    k = list(body.labels).index("Chest")
    valid = body.valid.copy()
    valid[:, k] = False
    state = replace(done, channels={**done.channels, "body": replace(body, valid=valid, values=np.where(valid[..., None], body.values, 0.), observed=body.observed & valid, confidence=body.confidence * valid)})
    shares = R.part_validity(state)
    assert all(shares[p] == 0. for p in ("shoulders", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"))
    alerts, _ = alerts_of(state, two)
    for part in ("shoulders", "left_wrist", "right_hand"):
        assert any(f"monocular {part} valid on 0.000" in a for a in alerts)
    assert any("shoulders: no frame both paths mark" in a for a in alerts)


def test_part_validity_reads_each_part_alone():
    state = gappy(signer_state(0, frames=20, amplitude=.4), 5, 10)
    points, valid = from_state(state)
    shares = R.part_validity(state)
    assert shares["shoulders"] == round(float(valid[:, [0, 1]].mean()), 5) and shares["left_elbow"] == round(float(valid[:, 2].mean()), 5) and shares["right_elbow"] == round(float(valid[:, 3].mean()), 5)
    assert shares["left_wrist"] == round(float(valid[:, 4].mean()), 5) and shares["right_wrist"] == round(float(valid[:, 5].mean()), 5) and shares["left_hand"] == round(float(valid[:, 6:27].mean()), 5) and shares["right_hand"] == round(float(valid[:, 27:48].mean()), 5)
    assert shares["left_hand"] < 1. and shares["shoulders"] == 1. and R.DISTANCE_POINTS["left_hand"] == tuple(range(7, 27)) and R.DISTANCE_POINTS["right_hand"] == tuple(range(28, 48)) and R.PARTS["left_hand"] == tuple(range(6, 27))


def test_key_distance_is_the_mean_over_points_both_mark():
    from signtranslator.data.keypoints2d import normalise
    a, b = gappy(signer_state(0, frames=20, amplitude=.4), 5, 8), gappy(signer_state(1, frames=20, amplitude=.4), 7, 10)
    pa, va = from_state(a)
    pb, vb = from_state(b)
    gap = np.linalg.norm(normalise(pa, va, y_down=False) - normalise(pb, vb, y_down=False), axis=-1)
    both = va & vb
    out = R.key_distance(a, b)
    assert out["left_wrist"] == round(float(gap[:, 4][both[:, 4]].mean()), 5) and out["right_hand"] == round(float(gap[:, 28:48][both[:, 28:48]].mean()), 5) and out["shoulders"] == round(float(gap[:, [0, 1]][both[:, [0, 1]]].mean()), 5)


@pytest.mark.parametrize("part,floor", [("shoulders", .5), ("left_elbow", .5), ("right_elbow", .5), ("left_wrist", .5), ("right_wrist", .5), ("left_hand", .3), ("right_hand", .3)])
def test_validity_floors_are_those_fixed_in_the_header_on_either_path(part, floor):
    ok = {p: [1.] for p in R.VALID_FLOOR}
    block = dict(clips=1, source_hash_matches=1, max_offset_seconds=0., group_distance=dict(shoulders=.1, elbows=.1, wrists=.1, hands=.1), group_mirrored=dict(shoulders=1., elbows=1., wrists=1., hands=1.))
    assert R.plumbing_alerts(ok, ok, block, 500, True) == []
    at = dict(ok, **{part: [floor]})
    below = dict(ok, **{part: [floor - .01]})
    assert R.plumbing_alerts(at, at, block, 500, True) == []
    assert any(f"two-view {part} valid" in a for a in R.plumbing_alerts(below, below, block, 500, True)) and any(f"monocular {part} valid" in a for a in R.plumbing_alerts(ok, below, block, 500, True))
    assert not any(f"two-view {part} valid" in a for a in R.plumbing_alerts(ok, below, block, 500, True))
    assert not any("validity is" in a for a in R.plumbing_alerts(dict(ok, **{part: [1.]}), dict(ok, **{part: [.5]}), block, 500, True))
    assert any(f"{part} validity is 0.490" in a for a in R.plumbing_alerts(dict(ok, **{part: [1.]}), dict(ok, **{part: [.49]}), block, 500, True))


@pytest.mark.parametrize("group,ceiling", [("shoulders", .5), ("elbows", 1.), ("wrists", 1.), ("hands", 3.)])
def test_distance_ceilings_are_those_fixed_in_the_header(group, ceiling):
    ok = {p: [1.] for p in R.VALID_FLOOR}
    block = dict(clips=1, source_hash_matches=1, max_offset_seconds=0., group_distance=dict(shoulders=.1, elbows=.1, wrists=.1, hands=.1), group_mirrored=dict(shoulders=9., elbows=9., wrists=9., hands=9.))
    assert R.plumbing_alerts(ok, ok, dict(block, group_distance=dict(block["group_distance"], **{group: ceiling})), 500, True) == []
    assert any(f"{group}: keypoint distance" in a for a in R.plumbing_alerts(ok, ok, dict(block, group_distance=dict(block["group_distance"], **{group: ceiling + .01})), 500, True))
    assert any(f"{group}: no frame both paths mark" in a for a in R.plumbing_alerts(ok, ok, dict(block, group_distance=dict(block["group_distance"], **{group: None})), 500, True))


def test_the_other_alerts_are_those_fixed_in_the_header():
    ok = {p: [1.] for p in R.VALID_FLOOR}
    block = dict(clips=4, source_hash_matches=4, max_offset_seconds=.001, group_distance=dict(shoulders=.1, elbows=.1, wrists=.1, hands=.1), group_mirrored=dict(shoulders=.05, elbows=1., wrists=1., hands=1.))
    assert R.plumbing_alerts(ok, ok, block, 475, True) == [] and R.plumbing_alerts(ok, ok, block, 3, False) == []       # shoulders are not in the mirrored check; a smoke has no floor
    assert any("474 shared clips, under 475" in a for a in R.plumbing_alerts(ok, ok, block, 474, True))
    assert any("1 of 4 two-view clips do not name" in a for a in R.plumbing_alerts(ok, ok, dict(block, source_hash_matches=3), 500, True))
    assert any("stored clocks differ" in a for a in R.plumbing_alerts(ok, ok, dict(block, max_offset_seconds=.0011), 500, True))
    for group in R.MIRRORED_GROUPS:
        assert any(f"{group}: a mirrored" in a for a in R.plumbing_alerts(ok, ok, dict(block, group_mirrored=dict(block["group_mirrored"], **{group: .1})), 500, True))
        assert R.plumbing_alerts(ok, ok, dict(block, group_mirrored=dict(block["group_mirrored"], **{group: None})), 500, True) == []
    assert set(R.MIRRORED_GROUPS) == {"elbows", "wrists", "hands"} and R.SHARED_FLOOR == 475 and R.VALID_RATIO_FLOOR == .5 and R.OFFSET_CEILING == .001


def test_conforming_is_compared_on_what_v1_receives():
    raw = monocular_state(0)
    done = layout_adapter.conform(raw, template_state())
    np.testing.assert_array_equal(R.reader_input(raw), R.reader_input(done))
    plan = R.unconformed_plan([R.Clip(0, "s0", raw, done, done, {})])
    assert plan["read"] is False and plan["clips_compared"] == 1 and plan["clips_differing"] == 0 and "byte-identical" in plan["reason"] and plan["differing"] == []


def test_a_conform_that_changes_what_v1_receives_means_the_form_is_read():
    raw = monocular_state(0)
    plan = R.unconformed_plan([R.Clip(0, "same", raw, raw, raw, {}), R.Clip(1, "moved", raw, gappy(raw, 8, 12), raw, {})])
    assert plan["read"] is True and plan["clips_differing"] == 1 and plan["differing"] == ["moved"] and "is read" in plan["reason"] and plan["clips_compared"] == 2


def test_reader_input_is_what_translate_builds():
    from signtranslator.data.keypoints2d import features, normalise
    state = gappy(signer_state(0, frames=20, amplitude=.4), 5, 8)
    points, valid = from_state(state)
    np.testing.assert_array_equal(R.reader_input(state), features(normalise(points, valid, y_down=False), valid))


def test_valid_frame_share_is_v1s_points_marked():
    assert R.valid_frame_share(signer_state(0, frames=8, amplitude=.4)) == 1.
    state = gappy(signer_state(0, frames=20, amplitude=.4), 5, 10)
    assert R.valid_frame_share(state) == round(float(from_state(state)[1].mean()), 5) and 0 < R.valid_frame_share(state) < 1


def test_the_conform_tally_says_how_each_channel_was_treated(monkeypatch):
    import tokenizer_domain_read as domain
    raw = monocular_state(0)
    tally = R.conform_tally([R.Clip(0, "s0", raw, raw, raw, {}), R.Clip(1, "s1", raw, raw, raw, {})], template_state())
    assert tally["kinds"]["body:equivalent"] == 2 and tally["kinds"]["left_hand:equivalent"] == 2 and tally["dropped"] == {} and sum(v for k, v in tally["kinds"].items() if k.endswith(":absent")) > 0
    monkeypatch.setattr(domain, "conform_audit", lambda state, template: dict(kinds=dict(body="same"), extra=["stray"]))
    assert R.conform_tally([R.Clip(0, "s0", raw, raw, raw, {}), R.Clip(1, "s1", raw, raw, raw, {})], template_state()) == dict(kinds={"body:same": 2}, dropped=dict(stray=2))


# ---- forms ------------------------------------------------------------------------------------------------------------------------

def clips_of(n=3):
    clips, refs = [], []
    for i in range(n):
        raw = monocular_state(i)
        done = layout_adapter.conform(raw, template_state())
        clips.append(R.Clip(i, f"s{i}", raw, done, R.cut_frames(done, np.arange(*CUT)), {}))
        refs.append(two_view_state(i, "a" * 64))
    return clips, refs


def test_each_form_is_built_from_the_right_states():
    from evaluate_text_to_motion import blanked
    clips, refs = clips_of(3)
    refs = refs + [two_view_state(3, "a" * 64)]
    kept, states, masks = R.build_form("real_two_view", refs, clips, None)
    assert kept == [0, 1, 2, 3] and states[2] is refs[2] and masks[2] is refs[2]
    kept, states, masks = R.build_form("blank", refs, clips, None)
    assert kept == [0, 1, 2, 3] and not states[0].channels["body"].valid.any() and masks[1] is refs[1] and len(states[0].timestamps) == len(refs[0].timestamps)
    np.testing.assert_array_equal(states[1].channels["body"].values, blanked(refs)[1].channels["body"].values)
    kept, states, masks = R.build_form("real_monocular", refs, clips, None)
    assert kept == [0, 1, 2] and states[1] is clips[1].conformed and masks[1] is clips[1].conformed
    kept, states, masks = R.build_form(R.UNCONFORMED, refs, clips, None)
    assert kept == [0, 1, 2] and states[1] is clips[1].raw and masks[1] is clips[1].raw
    kept, states, masks = R.build_form("real_monocular_trimmed", refs, clips, None)
    assert kept == [0, 1, 2] and states[1] is clips[1].trimmed and masks[1] is clips[1].trimmed
    with pytest.raises(SystemExit, match="unknown form"):
        R.build_form("a_generated_form", refs, clips, None)


@pytest.mark.parametrize("form,cap,source", [("round_trip_monocular", "rt_nearest", "conformed"), ("nearest_monocular", "v0_nearest_fill", "conformed"), ("rt_nearest_two_view", "rt_nearest", "ref")])
def test_each_cap_form_is_v1s_on_the_right_clip(monkeypatch, form, cap, source):
    tok = tiny_tokenizer()
    clips, refs = clips_of(2)
    seen = []
    original = T.cap_form
    monkeypatch.setattr(T, "cap_form", lambda name, state, tokenizer: seen.append((name, state, tokenizer)) or original(name, state, tokenizer))
    kept, states, masks = R.build_form(form, refs, clips, tok)
    want = (lambda i: clips[i].conformed) if source == "conformed" else (lambda i: refs[i])
    assert kept == [0, 1] and masks[1] is want(1) and [x[0] for x in seen] == [cap, cap] and seen[1][1] is want(1) and seen[0][2] is tok
    np.testing.assert_array_equal(states[0].channels["left_hand"].values, original(cap, want(0), tok).channels["left_hand"].values)
    assert all(states[0].channels[n].valid.all() for n in ("body", "left_hand", "right_hand"))


def test_nearest_monocular_keeps_the_clips_values_with_no_mark_and_no_decoder():
    tok = tiny_tokenizer()
    clips, refs = clips_of(1)
    _, states, _ = R.build_form("nearest_monocular", refs, clips, tok)
    c = clips[0].conformed.channels["left_hand"]
    assert states[0].channels["left_hand"].valid.all() and np.allclose(states[0].channels["left_hand"].values[c.valid], c.values[c.valid], atol=1e-3)


def test_the_2x2_transplants_values_and_marks_on_the_trimmed_frames():
    tok = tiny_tokenizer()
    clips, refs = clips_of(2)
    kept, states, masks = R.build_form("monocular_values_two_view_marks", refs, clips, tok)
    ref, trimmed = refs[1], clips[1].trimmed
    assert kept == [0, 1] and masks[1] is refs[1]
    for name in ("body", "left_hand", "right_hand"):
        c, t, r = states[1].channels[name], trimmed.channels[name], ref.channels[name]
        np.testing.assert_array_equal(c.valid, r.valid)
        both = r.valid & t.valid
        np.testing.assert_allclose(c.values[both], t.values[both])
    kept, states, masks = R.build_form("two_view_values_monocular_marks", refs, clips, tok)
    assert masks[1] is clips[1].trimmed
    for name in ("body", "left_hand", "right_hand"):
        c, t, r = states[1].channels[name], trimmed.channels[name], ref.channels[name]
        np.testing.assert_array_equal(c.valid, t.valid)
        both = r.valid & t.valid
        np.testing.assert_allclose(c.values[both], r.values[both])


def test_the_reverse_transplant_onto_a_joint_two_view_never_tracked_holds_the_neutral_not_zeros():
    tok = tiny_tokenizer()
    clips, refs = clips_of(1)
    refs[0] = with_empty_hand(refs[0])
    _, states, _ = R.build_form("two_view_values_monocular_marks", refs, clips, tok)
    c = states[0].channels["left_hand"]
    np.testing.assert_array_equal(c.valid, clips[0].trimmed.channels["left_hand"].valid)
    np.testing.assert_allclose(c.values[c.valid], V.neutral_pose(refs[0], tok).channels["left_hand"].values[c.valid])


def test_a_transplant_onto_a_joint_its_values_never_tracked_holds_the_neutral_not_zeros():
    tok = tiny_tokenizer()
    clips, refs = clips_of(1)
    trimmed = clips[0].trimmed
    empty = with_empty_hand(trimmed)
    clips[0] = R.Clip(0, "s0", clips[0].raw, clips[0].conformed, empty, {})
    _, states, _ = R.build_form("monocular_values_two_view_marks", refs, clips, tok)
    c = states[0].channels["left_hand"]
    np.testing.assert_array_equal(c.valid, refs[0].channels["left_hand"].valid)
    assert np.abs(c.values[c.valid]).sum(-1).min() > .5
    neutral = V.neutral_pose(empty, tok).channels["left_hand"].values
    np.testing.assert_allclose(c.values[c.valid], neutral[c.valid])


def test_an_unknown_form_refuses(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=2)
    with pytest.raises(SystemExit, match="unknown forms"):
        h.read("a9p")


def test_a_read_writes_what_it_read(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, tmp_path)
    h.to_control()
    real = json.loads((h.forms / "form-real_two_view.json").read_text())
    assert real["kind"] == "path-read-form" and real["report_only"] is True and real["form"] == "real_two_view" and real["declared_read"] is True and real["signs"] == [f"s{i}" for i in range(N)] and real["groups"][:3] == list(GROUP_NAMES)
    assert real["hypotheses"][0] == h.records[0]["text"] and real["references"] == [r["text"] for r in h.records] and real["nll"][0] == [1.] and real["code_digest"] == R.code_digest() and real["environment"] == P.environment() and real["instrument"]["sha256"] == "f" * 64 and real["tokenizer"] is None
    assert real["lost"] == [] and real["seconds_over_reference"] == [1.] * N and len(real["valid_frame_share"]) == N and real["valid_frame_share"][0] == R.valid_frame_share(h.refs[0]) and real["motion"]["wrist_speed"][0] is not None and real["tracked"]["wrists"][0] > 0 and real["hands"]["left_hand"]["speed"][0] is not None
    assert real["part_validity"]["left_hand"][1] == R.part_validity(h.refs[1])["left_hand"] and real["part_validity"]["left_hand"][0] < 1 and set(real["part_validity"]) == set(R.PARTS)
    assert not (h.forms / "monocular.json").exists() and json.loads((h.forms / "form-rt_nearest_two_view.json").read_text())["tokenizer"] == dict(key=L.TOKENIZER, sha256=L.TOKENIZER_SHA, gains=None)
    assert h.read("round_trip_monocular", "real_monocular") == 0
    mono = json.loads((h.forms / "form-real_monocular.json").read_text())
    trip = json.loads((h.forms / "form-round_trip_monocular.json").read_text())
    assert mono["seconds_over_reference"] == [1.2] * N and mono["hypotheses"][0].split() == h.records[0]["text"].split()[:8] and mono["tokenizer"] is None and trip["tokenizer"]["sha256"] == L.TOKENIZER_SHA
    assert all(v == 1. for v in trip["valid_frame_share"]) and all(v < 1. for v in mono["valid_frame_share"]) and all(v == 1. for v in trip["part_validity"]["left_hand"])
    assert any('"monocular_clips": 9' in line for line in capsys.readouterr().out.splitlines())


def test_a_read_passes_the_group_and_one_candidate_each_and_reads_its_own_clips_texts(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=(1,))
    h.to_control()
    seen = []
    original = h.reader.read
    monkeypatch.setattr(h.reader, "read", lambda states, texts, candidates, group=2: seen.append((len(states), list(texts), candidates.shape, group)) or original(states, texts, candidates, group=group))
    assert h.main("read", "--out-dir", str(h.forms), "--forms", "real_monocular", "--nll-group", "4") == 0
    assert seen[0][0] == N - 1 and seen[0][2] == (N - 1, 1) and seen[0][3] == 4 and seen[0][1] == [r["text"] for r in h.records if r["sign"] != "s1"]


def test_a_read_refuses_to_write_over_a_form_and_a_resume_leaves_it(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=2)
    assert h.read("real_two_view") == 0
    with pytest.raises(SystemExit, match="write-once"):
        h.read("real_two_view")
    before = (h.forms / "form-real_two_view.json").read_bytes()
    assert h.read("real_two_view", "blank", resume=True) == 0
    assert (h.forms / "form-real_two_view.json").read_bytes() == before and (h.forms / "form-blank.json").exists()


def test_the_read_checks_its_records_before_it_loads_any_state(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.records[1]["signer_component"] = "c00"
    monkeypatch.setattr(L, "load_states", lambda *a: pytest.fail("a state was read"))
    with pytest.raises(SystemExit, match="c00"):
        h.read("real_two_view")


def test_anchors_alone_need_no_monocular_lookup(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=2)
    monkeypatch.setattr(R, "open_monocular", lambda *a: pytest.fail("a monocular lookup"))
    assert h.read("real_two_view", "blank") == 0


def test_forms_without_the_tokenizer_do_not_load_it(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=2)
    monkeypatch.setattr(L, "load_tokenizer", lambda *a: pytest.fail("the tokenizer"))
    assert h.read("real_two_view", "blank") == 0
    assert set(R.TOKENIZER_FORMS) == {"rt_nearest_two_view", "nearest_monocular", "round_trip_monocular", "monocular_values_two_view_marks", "two_view_values_monocular_marks"}


@pytest.mark.parametrize("form", R.MONOCULAR_FORMS)
def test_a_declared_monocular_read_refuses_without_a_passed_control_before_it_loads_a_state(monkeypatch, tmp_path, form):
    h = Harness(monkeypatch, tmp_path, n=2)
    monkeypatch.setattr(L, "load_states", lambda *a: pytest.fail("a state was read"))
    with pytest.raises(SystemExit, match="no passed control"):
        h.read(form)


@pytest.mark.parametrize("change", [dict(reproduced=False), dict(code_digest="other"), dict(environment=dict(gpu="other"))])
def test_a_read_refuses_a_failed_control_or_one_of_other_code_or_environment(monkeypatch, tmp_path, change):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.to_control()
    path = h.forms / "control.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), **change)))
    with pytest.raises(SystemExit, match="no passed control"):
        h.read("real_monocular")


def test_a_smoke_monocular_read_needs_no_control(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, declared=False, n=3)
    assert h.read("real_monocular", smoke=True) == 0


def test_the_unconformed_form_is_skipped_with_its_reason_when_v1s_input_is_identical(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, tmp_path)
    h.to_control()
    assert h.read(*R.MONOCULAR_FORMS) == 0
    assert not (h.forms / f"form-{R.UNCONFORMED}.json").exists() and (h.forms / "form-real_monocular.json").exists()
    plan = json.loads((h.forms / "monocular.json").read_text())
    assert plan["unconformed"]["read"] is False and "byte-identical" in plan["unconformed"]["reason"] and plan["clips_loaded"] == N and plan["code_digest"] == R.code_digest() and plan["declared_read"] is True and plan["conform"]["kinds"]["body:equivalent"] == N
    assert "byte-identical" in capsys.readouterr().out
    assert h.read(R.UNCONFORMED, resume=True) == 0 and not (h.forms / f"form-{R.UNCONFORMED}.json").exists()


def test_the_monocular_record_holds_each_clips_pairing(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.to_control()
    assert h.read("real_monocular") == 0
    plan = json.loads((h.forms / "monocular.json").read_text())
    row = plan["pairing"][1]
    from signtranslator.pose.mirror import mirror_state
    trimmed = R.cut_frames(layout_adapter.conform(h.mono[1], template_state()), np.arange(*CUT))
    assert row["sign"] == "s1" and row["start_frame"] == 2 and row["frames_two_view"] == 20 and row["frames_monocular"] == 24 and row["source_hash_matches"] is True
    assert row["distance"] == R.key_distance(trimmed, h.refs[1]) and row["mirrored"] == R.key_distance(mirror_state(trimmed), h.refs[1])


def test_the_unconformed_form_is_read_when_conforming_changes_v1s_input(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.to_control()
    original = layout_adapter.conform
    monkeypatch.setattr(layout_adapter, "conform", lambda state, template: gappy(original(state, template), 8, 12))
    assert h.read(*R.MONOCULAR_FORMS) == 0
    plan = json.loads((h.forms / "monocular.json").read_text())
    assert plan["unconformed"]["read"] is True and json.loads((h.forms / f"form-{R.UNCONFORMED}.json").read_text())["signs"] == ["s0", "s1", "s2"]


def test_lost_clips_are_in_each_monocular_forms_report_and_no_anchors(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=(1,), absent=(4,))
    h.to_control()
    assert h.read("real_monocular") == 0
    anchor, mono = (json.loads((h.forms / f"form-{n}.json").read_text()) for n in ("real_two_view", "real_monocular"))
    assert anchor["lost"] == [] and len(anchor["signs"]) == N
    assert {x["sign"]: x["reason"] for x in mono["lost"]} == dict(s1="no_monocular_row", s4="state_missing") and mono["signs"] == [f"s{i}" for i in (0, 2, 3, 5, 6, 7, 8)]
    assert mono["references"] == [h.records[i]["text"] for i in (0, 2, 3, 5, 6, 7, 8)] and mono["groups"] == [h.records[i]["signer_component"] for i in (0, 2, 3, 5, 6, 7, 8)]


def test_a_resume_after_the_unconformed_form_was_settled_opens_nothing(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.to_control()
    assert h.read(*R.MONOCULAR_FORMS) == 0
    monkeypatch.setattr(R, "open_monocular", lambda *a: pytest.fail("the monocular clips were opened again"))
    assert h.read(*R.MONOCULAR_FORMS, resume=True) == 0


def test_each_forms_motion_is_measured_on_its_own_paths_tracked_frames(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.to_control()
    assert h.read("round_trip_monocular") == 0
    form = json.loads((h.forms / "form-round_trip_monocular.json").read_text())
    conformed = [layout_adapter.conform(s, template_state()) for s in h.mono]
    states = [T.cap_form("rt_nearest", c, h.tokenizer) for c in conformed]
    assert form["motion"] == json.loads(json.dumps(L.motion_statistics(states, conformed))) and form["motion"] != json.loads(json.dumps(L.motion_statistics(states, states)))
    assert form["hands"] == json.loads(json.dumps(P.hand_statistics(states, conformed))) and form["hands"] != json.loads(json.dumps(P.hand_statistics(states, states)))


def test_a_form_read_alongside_monocular_ones_keeps_its_own_lost_list(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, declared=False, n=4, missing=(1,))
    assert h.read(smoke=True) == 0
    assert json.loads((h.forms / "form-real_two_view.json").read_text())["lost"] == [] and len(json.loads((h.forms / "form-real_monocular.json").read_text())["lost"]) == 1


def test_the_tokenizer_failing_on_a_clip_stops_the_read(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.to_control()
    monkeypatch.setattr(T, "cap_form", lambda *a: (_ for _ in ()).throw(RuntimeError("planted")))
    with pytest.raises(RuntimeError, match="planted"):
        h.read("round_trip_monocular")


# ---- the smoke's list ------------------------------------------------------------------------------------------------------------

def test_a_smoke_takes_the_longest_and_never_tracked_clips_as_tok_v1s_does(monkeypatch, tmp_path):
    records = train_records(5)
    extra = [dict(records[0], sign="long")] + [dict(records[0], sign="s1")]
    never = [dict(records[0], sign="never")]
    monkeypatch.setattr(L, "open_read_list", lambda store, smoke: (records, False, dict(list="smoke", clips=len(records))))
    monkeypatch.setattr(P, "longest_train_records", lambda store, k: extra[:k])
    monkeypatch.setattr(V, "never_tracked_records", lambda store, k: never[:k])
    args = R.build_parser().parse_args(["read", "--out-dir", "x", "--smoke-train", "5", "--smoke-longest", "2", "--smoke-never-tracked", "1"])
    out, declared, described = R.open_list(None, args)
    assert [r["sign"] for r in out] == ["s0", "s1", "s2", "s3", "s4", "long", "never"] and declared is False and described == dict(list="smoke", clips=7, longest=2, never_tracked=1)
    args = R.build_parser().parse_args(["read", "--out-dir", "x", "--smoke-train", "5"])
    assert len(R.open_list(None, args)[0]) == 5


@pytest.mark.parametrize("flag", ["--smoke-longest", "--smoke-never-tracked"])
def test_smoke_extras_belong_to_a_smoke(flag):
    with pytest.raises(SystemExit, match="belong to a smoke"):
        R.open_list(None, R.build_parser().parse_args(["read", "--out-dir", "x", flag, "1"]))


def test_a_smoke_extra_that_is_no_c02_train_clip_is_refused(monkeypatch):
    monkeypatch.setattr(L, "open_read_list", lambda store, smoke: (train_records(2), False, dict(list="smoke", clips=2)))
    monkeypatch.setattr(P, "longest_train_records", lambda store, k: [dict(train_records(1)[0], sign="bad", signer_component="c00")])
    with pytest.raises(SystemExit, match="c00"):
        R.open_list(None, R.build_parser().parse_args(["read", "--out-dir", "x", "--smoke-train", "2", "--smoke-longest", "1"]))


# ---- stored reads -----------------------------------------------------------------------------------------------------------------

def plant_blank_report(h, blank_nll=3.):
    stored = dict(signs=[f"s{i}" for i in range(h.n)], references=[], forms=dict(real=dict(per_clip=dict(nll=[1.] * h.n)), blank=dict(per_clip=dict(nll=[blank_nll] * h.n))))
    raw = json.dumps(stored).encode()
    h.store.put(P.CANARY_REPORT, raw)
    h.monkeypatch.setattr(P, "CANARY_REPORT_SHA", hashlib.sha256(raw).hexdigest())


def test_stored_check_prints_both_hashes_and_figures(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, tmp_path, n=3)
    plant_blank_report(h)
    monkeypatch.setattr(R, "STORED_RT_LIKELIHOOD", .25)
    monkeypatch.setattr(R, "STORED_NEAREST_LIKELIHOOD", 1.5)
    assert h.main("stored-check") == 0
    rows = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["stored_check"]
    assert [r["form"] for r in rows] == ["rt_nearest", "v0_nearest_fill"] and rows[0]["sha256"] == hashlib.sha256(h.store.get(R.STORED_RT)).hexdigest() and rows[1]["sha256"] == hashlib.sha256(h.store.get(R.STORED_NEAREST)).hexdigest()
    assert rows[0]["likelihood_minus_blank"] == .25 and rows[1]["likelihood_minus_blank"] == 1.5 and rows[0]["clips"] == 3 and rows[0]["bleu"] == R.STORED_RT_BLEU and rows[0]["ci95"] == R.STORED_RT_CI


@pytest.mark.parametrize("rt,nearest,name", [(.25 + .0011, 1.5, "rt_nearest"), (.25, 1.5 + .0011, "v0_nearest_fill")])
def test_stored_check_stops_on_another_likelihood(monkeypatch, tmp_path, rt, nearest, name):
    h = Harness(monkeypatch, tmp_path, n=3)
    plant_blank_report(h)
    monkeypatch.setattr(R, "STORED_RT_LIKELIHOOD", rt)
    monkeypatch.setattr(R, "STORED_NEAREST_LIKELIHOOD", nearest)
    with pytest.raises(SystemExit, match=name):
        h.main("stored-check")


def test_stored_check_holds_the_blank_report_to_its_pin(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    plant_blank_report(h)
    monkeypatch.setattr(P, "CANARY_REPORT_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        h.main("stored-check")


def test_the_stored_rt_nearest_is_held_to_its_pinned_hash(monkeypatch, tmp_path):
    assert R.STORED_RT_SHA256 == "4577be5a56205038e97c6ff2b091015e9cf3a072f5f1ae40a7708c26819d2817"
    h = Harness(monkeypatch, tmp_path, n=3)
    good = hashlib.sha256(h.store.get(R.STORED_RT)).hexdigest()
    assert R.load_stored_rt_nearest(h.store)[1] == good
    monkeypatch.setattr(R, "STORED_RT_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        R.load_stored_rt_nearest(h.store)


def test_the_stored_nearest_is_held_to_its_hash_from_tok_v1s_pin(monkeypatch, tmp_path):
    assert (R.STORED_NEAREST, R.STORED_NEAREST_SHA256) == T.STORED_FORMS["v0_nearest_fill"] and R.STORED_NEAREST_SHA256.startswith("e1bfd0ac") and R.STORED_NEAREST.endswith("tok-probes-v0/forms/form-v0_nearest_fill.json")
    h = Harness(monkeypatch, tmp_path, n=3)
    assert R.load_stored_nearest(h.store)[1] == R.STORED_NEAREST_SHA256
    monkeypatch.setattr(R, "STORED_NEAREST_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        R.load_stored_nearest(h.store)


@pytest.mark.parametrize("change,word", [(dict(kind="other"), "kind"), (dict(form="a9p"), "form"), (dict(declared_read=False), "declared_read"), (dict(report_only=False), "report_only"), (dict(list=dict(sha256="0" * 64)), "list"), (dict(signs=["s0"]), "clips"), (dict(instrument=dict(sha256="0" * 64)), "instrument"), (dict(tokenizer=dict(sha256="0" * 64)), "tokenizer")])
@pytest.mark.parametrize("which", ["rt", "nearest"])
def test_a_stored_form_that_is_not_the_stored_read_refuses(monkeypatch, tmp_path, change, word, which):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.plant_stored(**{which: change})
    with pytest.raises(SystemExit, match=f"not the stored read: .*{word}"):
        (R.load_stored_rt_nearest if which == "rt" else R.load_stored_nearest)(h.store)


def test_a_stored_rt_nearest_of_other_code_refuses(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.plant_stored(rt=dict(code_digest="0000"))
    with pytest.raises(SystemExit, match="another code digest"):
        R.load_stored_rt_nearest(h.store)


def test_a_stored_form_reading_another_figure_refuses(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    monkeypatch.setattr(R, "STORED_RT_BLEU", 4.124)
    with pytest.raises(SystemExit, match="not the plan's"):
        R.load_stored_rt_nearest(h.store)
    h.plant_stored()
    monkeypatch.setattr(R, "STORED_RT_CI", [0., 0.])
    with pytest.raises(SystemExit, match="not the plan's"):
        R.load_stored_rt_nearest(h.store)
    h.plant_stored()
    monkeypatch.setattr(R, "STORED_NEAREST_BLEU", 4.621)
    with pytest.raises(SystemExit, match="not the plan's"):
        R.load_stored_nearest(h.store)


def test_the_pinned_stored_figures_are_the_plans():
    assert (R.STORED_RT_BLEU, R.STORED_RT_CI, R.STORED_RT_DIGEST, R.STORED_RT_LIKELIHOOD) == (4.124, [3.485, 4.764], "abcb46bd", .1883) and R.STORED_RT == "acquisition/2026-10-17-v1/youtube-asl-v1/tok-probes-v1/forms/form-rt_nearest.json"
    assert (R.STORED_NEAREST_BLEU, R.STORED_NEAREST_LIKELIHOOD) == (4.621, .2337) and R.THRESHOLD == dict(bleu=.5, likelihood=.052) and R.BAR == 3.8


# ---- the control ------------------------------------------------------------------------------------------------------------------

def read_anchors(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    assert h.read(*R.ANCHORS) == 0
    return h


def test_the_control_passes_and_writes_what_it_read(monkeypatch, tmp_path):
    h = read_anchors(monkeypatch, tmp_path)
    h.plant_control()
    assert h.control() == 0
    result = json.loads((h.forms / "control.json").read_text())
    assert result["reproduced"] is True and result["misses"] == [] and result["code_digest"] == R.code_digest() and result["environment"] == P.environment() and result["reads"]["real"]["identical"] == 1. and result["stored_sha256"] == P.CANARY_REPORT_SHA
    assert result["round_trip"]["identical"] == 1. and result["round_trip"]["nll_mean_absolute_difference"] == 0. and result["round_trip"]["misses"] == [] and not h.store.list("pr/void")


@pytest.mark.parametrize("change,miss", [(dict(hyp_change=1), "real hypotheses"), (dict(nll_shift=.0012), "real NLL"), (dict(bleu_shift=.06), "real BLEU"), (dict(rt_hyp_change=1), "rt_nearest hypotheses"), (dict(rt_nll_shift=.0012), "rt_nearest NLL")])
def test_a_control_miss_exits_3_voids_the_read_and_a_read_and_a_merge_refuse_it(monkeypatch, tmp_path, change, miss):
    h = read_anchors(monkeypatch, tmp_path)
    h.plant_control(**change)
    assert h.control() == 3
    result = json.loads((h.forms / "control.json").read_text())
    assert not result["reproduced"] and any(miss in m for m in result["misses"])
    void = json.loads(h.store.get(f"pr/void/{R.code_digest()}.json"))
    assert void["kind"] == "path-read-void" and void["control"]["misses"] == result["misses"] and void["code_digest"] == R.code_digest()
    with pytest.raises(SystemExit, match="no passed control"):
        h.read("real_monocular")


def test_a_merge_refuses_a_failed_control(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.pipeline()
    path = h.forms / "control.json"
    good = json.loads(path.read_text())
    for change in (dict(reproduced=False), dict(code_digest="other"), dict(environment=dict(gpu="other"))):
        path.write_text(json.dumps(dict(good, **change)))
        with pytest.raises(SystemExit, match="no passed control"):
            h.merge("again.json")
    path.unlink()
    with pytest.raises(SystemExit, match="no passed control"):
        h.merge("again.json")


def test_a_control_at_its_limits_passes(monkeypatch, tmp_path):
    h = read_anchors(monkeypatch, tmp_path)
    h.plant_control(nll_shift=.0009, bleu_shift=.04, rt_nll_shift=.0009)
    assert h.control() == 0


def test_a_control_of_other_clips_or_order_refuses(monkeypatch, tmp_path):
    h = read_anchors(monkeypatch, tmp_path)
    h.plant_control(order=True)
    with pytest.raises(SystemExit, match="other clips or another order"):
        h.control()


def test_a_round_trip_read_off_the_plans_bleu_misses_alone(monkeypatch):
    refs = [r["text"] for r in d20_records(10)]
    read = dict(signs=[f"s{i}" for i in range(10)], references=refs, hypotheses=[" ".join(t.split()[:6]) for t in refs], nll=[[1.]] * 10)
    bleu = M.bleu_ci_fast(read["hypotheses"], refs)[0]
    monkeypatch.setattr(R, "STORED_RT_BLEU", bleu + .04)
    assert R.round_trip_control(read, read)["misses"] == []
    monkeypatch.setattr(R, "STORED_RT_BLEU", bleu + .06)
    row = R.round_trip_control(read, read)
    assert row["misses"] == ["rt_nearest BLEU"] and row["identical"] == 1. and row["stored_bleu"] == bleu + .06 and row["bleu"] == bleu and row["bleu_gap"] == .06


def test_a_round_trip_control_of_other_clips_or_references_refuses():
    read = dict(signs=["a", "b"], references=["x y z w v", "p q r s t"], hypotheses=["x y z w v", "p q r s t"], nll=[[1.], [2.]])
    with pytest.raises(SystemExit, match="stored rt_nearest read other clips"):
        R.round_trip_control(read, dict(read, signs=["b", "a"]))
    with pytest.raises(SystemExit, match="stored rt_nearest read other clips"):
        R.round_trip_control(read, dict(read, references=["other", "p q r s t"]))


def test_a_changed_stored_report_refuses_and_one_without_per_clip_nll_refuses(monkeypatch, tmp_path):
    h = read_anchors(monkeypatch, tmp_path)
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
    h = Harness(monkeypatch, tmp_path, declared=False, n=5)
    assert h.read(*R.ANCHORS, smoke=True) == 0
    h.plant_control(hyp_change=5)
    assert h.control() == 0 and not (h.forms / "control.json").exists()
    assert "not applicable" in capsys.readouterr().out
    monkeypatch.setattr(P, "CANARY_REPORT_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        h.control()


def test_a_smoke_control_still_checks_the_stored_rt_nearest(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, declared=False, n=5)
    assert h.read(*R.ANCHORS, smoke=True) == 0
    h.plant_control()
    h.plant_stored(rt=dict(form="other"))
    with pytest.raises(SystemExit, match="not the stored read"):
        h.control()


def test_the_control_refuses_forms_of_different_runs(monkeypatch, tmp_path):
    h = read_anchors(monkeypatch, tmp_path)
    h.plant_control()
    path = h.forms / "form-rt_nearest_two_view.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), code_digest="other")))
    with pytest.raises(SystemExit, match="different code"):
        h.control()


# ---- merge ------------------------------------------------------------------------------------------------------------------------

def test_every_read_runs_on_the_shared_clips_and_lost_clips_are_listed(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=(1,), no_state=(2,), absent=(4,))
    report = h.pipeline()
    shared = [f"s{i}" for i in (0, 3, 5, 6, 7, 8)]
    assert report["clips"]["list"] == N and report["clips"]["shared"] == 6
    lost = report["clips"]["outside_shared"]
    assert lost["count"] == 3 and lost["by_reason"] == dict(no_monocular_row=1, no_state_in_row=1, state_missing=1) and {c["sign"] for c in lost["clips"]} == {"s1", "s2", "s4"} and lost["clips"][0]["forms"]["real_monocular"]["reason"] == "no_monocular_row"
    forms = {n: json.loads((h.forms / f"form-{n}.json").read_text()) for n in ("real_two_view", "blank", "real_monocular", "nearest_monocular", "round_trip_monocular", "real_monocular_trimmed", "rt_nearest_two_view")}
    assert len(forms["real_two_view"]["signs"]) == N and forms["real_monocular"]["signs"] == shared
    for name, form in forms.items():
        pick = [form["signs"].index(s) for s in shared]
        bleu, ci = M.bleu_ci_fast([form["hypotheses"][i] for i in pick], [form["references"][i] for i in pick])
        assert report["forms"][name]["bleu"] == dict(bleu=bleu, ci95=ci) and report["forms"][name]["clips"] == 6
    two = [forms["real_two_view"]["hypotheses"][forms["real_two_view"]["signs"].index(s)] for s in shared]
    mono = [forms["real_monocular"]["hypotheses"][forms["real_monocular"]["signs"].index(s)] for s in shared]
    refs = [forms["real_monocular"]["references"][forms["real_monocular"]["signs"].index(s)] for s in shared]
    assert report["forms"]["real_monocular"]["paired"]["real_two_view"]["bleu"] == M.paired_bleu_fast(M.bleu_statistics(mono, refs), M.bleu_statistics(two, refs))
    assert report["reading"]["marked_secondary"]["bleu"]["difference"] == report["forms"]["real_monocular"]["paired"]["real_two_view"]["bleu"]["difference"]
    assert report["forms"]["real_monocular"]["paired"]["real_two_view"]["likelihood_minus_blank"]["difference"] == pytest.approx(-1.)
    assert report["same_path"]["clips"] == 6 and report["stored"]["rt_nearest"]["shared"]["clips"] == 6 and report["stored"]["v0_nearest_fill"]["shared"]["clips"] == 6 and report["pairing"]["clips"] == 6


def test_the_marks_free_pair_and_ratio_have_their_planted_values(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    report = h.pipeline()
    likelihood = report["same_path"]["likelihood_minus_blank"]
    free = likelihood["marks_free"]
    # dnll: stored nearest 1.5, nearest_monocular 0.8, round trip 0.5, stored rt 0.25
    assert free["monocular_gap"]["point"] == .8 and free["two_view_gap"]["point"] == 1.5 and free["monocular_gap_over_two_view_gap"]["point"] == pytest.approx(.5333) and free["monocular_minus_two_view"]["point"] == -.7
    assert free["round_trip_monocular_share_of_monocular_gap"]["point"] == .625 and free["round_trip_monocular_share_of_two_view_gap"]["point"] == pytest.approx(.3333) and free["stored_rt_nearest_share_of_two_view_gap"]["point"] == pytest.approx(.1667)
    assert free["monocular_gap_over_two_view_gap"]["ci95"] == [.5333, .5333]
    marked = likelihood["marked"]
    assert marked["monocular_gap"]["point"] == 1. and marked["two_view_gap"]["point"] == 2. and marked["monocular_gap_over_two_view_gap"]["point"] == .5 and marked["monocular_minus_two_view"]["point"] == -1. and marked["trimmed_minus_two_view"]["point"] == -.9
    assert marked["round_trip_monocular_share_of_monocular_gap"]["point"] == .5 and marked["round_trip_monocular_share_of_two_view_gap"]["point"] == .25
    trip = likelihood["round_trip"]
    assert trip["monocular_minus_real_monocular"]["point"] == -.5 and trip["monocular_minus_two_view_this_run"]["point"] == .25 and trip["monocular_minus_stored"]["point"] == .25 and trip["this_run_minus_stored"]["point"] == 0.
    bleu = report["same_path"]["bleu"]
    b = {n: report["forms"][n]["bleu"]["bleu"] for n in report["forms"]}
    assert bleu["marked"]["monocular_gap_over_two_view_gap"]["point"] == pytest.approx((b["real_monocular"] - b["blank"]) / (b["real_two_view"] - b["blank"]), abs=2e-3) and report["same_path"]["bootstrap"] == dict(n=2000, seed=0)
    assert report["forms"]["real_monocular"]["share_of_gap"]["likelihood"] == .5 and report["forms"]["round_trip_monocular"]["share_of_gap"]["likelihood"] == .25 and report["forms"]["real_two_view"]["share_of_gap"]["likelihood"] == 1. and report["forms"]["blank"]["share_of_gap"]["bleu"] == 0.


def test_the_2x2_has_its_planted_cells_and_effects(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    cells = h.pipeline()["same_path"]["likelihood_minus_blank"]["values_and_marks"]
    assert cells["report_only"] is True and cells["two_view_values_two_view_marks"] == 2. and cells["monocular_values_monocular_marks"] == pytest.approx(1.1) and cells["monocular_values_two_view_marks"] == pytest.approx(1.4) and cells["two_view_values_monocular_marks"] == pytest.approx(1.7)
    assert cells["values_under_two_view_marks"]["point"] == -.6 and cells["values_under_monocular_marks"]["point"] == -.6 and cells["marks_on_two_view_values"]["point"] == -.3 and cells["marks_on_monocular_values"]["point"] == -.3


def test_shares_are_taken_from_blank_when_blank_reads_above_zero(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, levels=dict(blank=(4, 3.), round_trip=(6, 2.5)))
    report = h.pipeline()
    forms = {n: json.loads((h.forms / f"form-{n}.json").read_text()) for n in ("real_two_view", "blank", "real_monocular", "round_trip_monocular", "nearest_monocular")}
    refs = forms["real_two_view"]["references"]
    stats = {n: M.bleu_statistics(f["hypotheses"], refs) for n, f in forms.items()}
    point = {n: float(M.bleu_from_statistics(s.sum(0))) for n, s in stats.items()}
    assert point["blank"] > 0
    idx = np.random.default_rng(0).integers(0, len(refs), (R.BOOT, len(refs)))
    draws = {n: M.bleu_from_statistics(s[idx].sum(1)) for n, s in stats.items()}
    row = report["same_path"]["bleu"]["marks_free"]["round_trip_monocular_share_of_monocular_gap"]
    assert row["point"] == round((point["round_trip_monocular"] - point["blank"]) / (point["nearest_monocular"] - point["blank"]), 4)
    assert row["ci95"] == P._quantiles((draws["round_trip_monocular"] - draws["blank"]) / (draws["nearest_monocular"] - draws["blank"]))


def test_bleu_differences_and_shares_resample_the_forms_jointly_over_clips():
    refs = [r["text"] for r in d20_records(30)]

    def form(keep):
        return dict(references=refs, hypotheses=[" ".join(t.split()[:keep(i)]) for i, t in enumerate(refs)], nll=[[3. - keep(i) / 10.] for i in range(30)])
    reads = dict(real_two_view=form(lambda i: 5 + i % 11), real_monocular=form(lambda i: 5 + i % 7), nearest_monocular=form(lambda i: 5 + i % 6), round_trip_monocular=form(lambda i: 4 + i % 5), blank=form(lambda i: 4))
    out = R.same_path(reads, dict(stored_rt_nearest=form(lambda i: 4 + i % 3), stored_v0_nearest_fill=form(lambda i: 5 + i % 9)))
    idx = np.random.default_rng(0).integers(0, 30, (R.BOOT, 30))
    stats = {n: M.bleu_statistics(f["hypotheses"], refs) for n, f in reads.items()}
    draws = {n: M.bleu_from_statistics(x[idx].sum(1)) for n, x in stats.items()}
    assert out["bleu"]["marked"]["monocular_minus_two_view"]["ci95"] == P._quantiles(draws["real_monocular"] - draws["real_two_view"])
    assert out["bleu"]["marked"]["monocular_gap_over_two_view_gap"]["ci95"] == P._quantiles((draws["real_monocular"] - draws["blank"]) / (draws["real_two_view"] - draws["blank"]))
    assert out["clips"] == 30 and out["bleu"]["marks_free"]["stored_rt_nearest_share_of_two_view_gap"] is not None and out["bleu"]["values_and_marks"]["values_under_two_view_marks"] is None
    both = R.same_path(dict(reads, rt_nearest_two_view=form(lambda i: 5 + i % 4)), dict(stored_rt_nearest=form(lambda i: 4 + i % 3), stored_v0_nearest_fill=form(lambda i: 5 + i % 9)))
    assert both["likelihood_minus_blank"]["round_trip"]["monocular_minus_two_view_this_run"]["point"] != both["likelihood_minus_blank"]["round_trip"]["monocular_minus_stored"]["point"]
    assert both["likelihood_minus_blank"]["round_trip"]["this_run_minus_stored"]["point"] != both["likelihood_minus_blank"]["round_trip"]["monocular_minus_stored"]["point"]


def test_likelihood_differences_stay_paired_clip_by_clip(monkeypatch, tmp_path):
    """Levels that vary by clip: a joint resample of a constant difference has no spread at all, though each form's own mean does."""
    h = Harness(monkeypatch, tmp_path)
    original = h.reader.read

    def varying(states, texts, candidates, group=2):
        out = original(states, texts, candidates, group=group)
        moving = np.array([[.25 * int(t.split()[0].split("x")[1]) if h.reader.level(s, t) != "blank" else 0.] for s, t in zip(states, texts)])
        out["nll"] = out["nll"] + moving
        return out
    monkeypatch.setattr(h.reader, "read", varying)
    report = h.pipeline()
    marked = report["same_path"]["likelihood_minus_blank"]["marked"]["monocular_minus_two_view"]
    assert marked["ci95"] == [-1., -1.] and report["forms"]["real_monocular"]["likelihood_minus_blank"]["ci95"][0] < report["forms"]["real_monocular"]["likelihood_minus_blank"]["ci95"][1]
    assert report["forms"]["real_monocular"]["paired"]["real_two_view"]["likelihood_minus_blank"]["ci95"] == [-1., -1.]


def test_the_round_trip_is_reported_beside_3_8():
    refs = [r["text"] for r in d20_records(30)]
    read = dict(references=refs, hypotheses=[" ".join(t.split()[:4 + i % 4]) for i, t in enumerate(refs)])
    bleu, ci = M.bleu_ci_fast(read["hypotheses"], refs)
    out = R.beside_bar(read)
    assert out == dict(bar=3.8, bleu=bleu, ci95=ci, minus_bar=round(bleu - 3.8, 3), lower_bound_above_bar=ci[0] > 3.8, upper_bound_below_bar=ci[1] < 3.8) and out["lower_bound_above_bar"] is True and out["upper_bound_below_bar"] is False
    low = dict(references=refs, hypotheses=["z y x w"] * 30)
    assert R.beside_bar(low)["upper_bound_below_bar"] is True and R.beside_bar(low)["lower_bound_above_bar"] is False


def test_an_interval_across_the_bar_clears_neither_way(monkeypatch):
    monkeypatch.setattr(M, "bleu_ci_fast", lambda hypotheses, references: (4., [3.5, 4.5]))
    out = R.beside_bar(dict(hypotheses=[], references=[]))
    assert out["lower_bound_above_bar"] is False and out["upper_bound_below_bar"] is False and out["minus_bar"] == .2


def test_the_report_puts_the_round_trip_beside_3_8(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    report = h.pipeline()
    assert report["round_trip_beside_bar"]["bleu"] == report["forms"]["round_trip_monocular"]["bleu"]["bleu"] and report["round_trip_beside_bar"]["bar"] == 3.8


def test_per_signer_rows_partition_the_shared_clips(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=(0,))
    report = h.pipeline()
    rows = report["per_signer"]
    assert sorted(rows) == list(GROUP_NAMES) and [rows[g]["real_monocular"]["clips"] for g in GROUP_NAMES] == [2, 3, 3] and sum(rows[g]["real_two_view"]["clips"] for g in GROUP_NAMES) == N - 1
    form = json.loads((h.forms / "form-real_monocular.json").read_text())
    index = [i for i, g in enumerate(form["groups"]) if g == "c03"]
    assert rows["c03"]["real_monocular"]["bleu"] == dict(zip(("bleu", "ci95"), M.bleu_ci_fast([form["hypotheses"][i] for i in index], [form["references"][i] for i in index])))
    row = rows["c03"]["real_monocular"]
    assert row["likelihood_minus_blank"]["mean"] == 1. and row["share_of_gap"]["likelihood"] == .5 and row["paired"]["real_two_view"]["likelihood_minus_blank"]["difference"] == -1. and 0 < row["valid_frame_share"]["mean"] < 1
    assert set(row) == set(R.SIGNER_ROW) and "tracked_share" in row and row["seconds_over_reference"] == 1.2 and row["part_validity"]["left_hand"]["mean"] < 1


def test_shares_of_validity_are_read_per_form_on_the_shared_clips(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    forms = h.pipeline()["forms"]
    assert forms["blank"]["valid_frame_share"]["mean"] == 0. and forms["round_trip_monocular"]["valid_frame_share"]["mean"] == 1. and forms["nearest_monocular"]["valid_frame_share"]["mean"] == 1. and forms["real_monocular"]["valid_frame_share"]["mean"] == pytest.approx(np.mean([R.valid_frame_share(s) for s in h.mono]), abs=1e-4)
    assert forms["real_two_view"]["valid_frame_share"]["mean"] == pytest.approx(np.mean([R.valid_frame_share(s) for s in h.refs]), abs=1e-4)
    assert forms["real_monocular"]["part_validity"]["right_hand"]["mean"] == pytest.approx(np.mean([R.part_validity(s)["right_hand"] for s in h.mono]), abs=1e-4) and forms["real_two_view"]["part_validity"]["left_elbow"]["mean"] == pytest.approx(np.mean([R.part_validity(s)["left_elbow"] for s in h.refs]), abs=1e-4)
    assert forms["real_monocular"]["seconds_over_reference"] == 1.2 and forms["real_monocular_trimmed"]["seconds_over_reference"] == 1. and forms["real_two_view"]["seconds_over_reference"] == 1. and forms["real_monocular"]["identical_to_real_two_view"] == 0. and forms["real_two_view"]["identical_to_real_two_view"] == 1.
    assert "real_two_view" in forms["real_monocular"]["paired"] and "real" not in forms["real_monocular"]["paired"] and "blank" in forms["real_monocular"]["paired"]


def test_stored_forms_are_compared_on_the_shared_clips_beside_their_500(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=(1,))
    report = h.pipeline()
    row, nearest = report["stored"]["rt_nearest"], report["stored"]["v0_nearest_fill"]
    stored = json.loads(h.store.get(R.STORED_RT))
    shared = [i for i in range(N) if i != 1]
    assert row["all_500"]["clips"] == N and row["all_500"]["bleu"] == R.STORED_RT_BLEU and row["sha256"] == hashlib.sha256(h.store.get(R.STORED_RT)).hexdigest() and row["key"] == R.STORED_RT and nearest["sha256"] == R.STORED_NEAREST_SHA256 and nearest["key"] == R.STORED_NEAREST
    bleu, ci = M.bleu_ci_fast([stored["hypotheses"][i] for i in shared], [stored["references"][i] for i in shared])
    assert row["shared"]["bleu"] == bleu and row["shared"]["ci95"] == ci and row["shared"]["clips"] == N - 1 and row["shared"]["likelihood_minus_blank"]["mean"] == .25 and nearest["shared"]["likelihood_minus_blank"]["mean"] == 1.5
    assert row["environment"] == dict(gpu="stored") and row["same_environment"] is False and row["code_digest"].startswith(R.STORED_RT_DIGEST)


@pytest.mark.parametrize("which", ["rt", "nearest"])
def test_a_stored_form_with_another_reference_refuses(monkeypatch, tmp_path, which):
    h = Harness(monkeypatch, tmp_path)
    h.to_control()
    assert h.read(*R.MONOCULAR_FORMS, resume=True) == 0
    current = json.loads(h.store.get(R.STORED_RT if which == "rt" else R.STORED_NEAREST))
    references = list(current["references"])
    references[0] = "another sentence"
    if which == "rt":
        kept = json.loads(h.store.get(R.STORED_RT))
        h.plant_stored(rt=dict(kept, references=references))
        bleu, ci = M.bleu_ci_fast(kept["hypotheses"], references)
        monkeypatch.setattr(R, "STORED_RT_BLEU", bleu)
        monkeypatch.setattr(R, "STORED_RT_CI", ci)
    else:
        rt = json.loads(h.store.get(R.STORED_RT))
        h.plant_stored(rt=rt, nearest=dict(references=references))
        monkeypatch.setattr(R, "STORED_NEAREST_BLEU", M.bleu_ci_fast(current["hypotheses"], references)[0])
        bleu, ci = M.bleu_ci_fast(rt["hypotheses"], rt["references"])
        monkeypatch.setattr(R, "STORED_RT_BLEU", bleu)
        monkeypatch.setattr(R, "STORED_RT_CI", ci)
    with pytest.raises(SystemExit, match="another reference"):
        h.merge()


def test_shared_clips_refuse_a_form_holding_a_foreign_clip_or_another_reference():
    base = dict(signs=["a", "b"], references=["x", "y"])
    assert R.shared_signs(dict(real_two_view=base, other=dict(signs=["b"], references=["y"]))) == ["b"]
    with pytest.raises(SystemExit, match="not this list's clip"):
        R.shared_signs(dict(real_two_view=base, other=dict(signs=["c"], references=["z"])))
    with pytest.raises(SystemExit, match="not this list's clip"):
        R.shared_signs(dict(real_two_view=base, other=dict(signs=["a"], references=["changed"])))


def test_shared_clips_are_those_every_form_read_in_list_order():
    forms = dict(real_two_view=dict(signs=list("abcd"), references=list("abcd")), x=dict(signs=list("dba"), references=list("dba")), y=dict(signs=list("bcd"), references=list("bcd")))
    assert R.shared_signs(forms) == ["b", "d"]


def test_a_clip_gone_with_no_reason_refuses():
    forms = dict(real_two_view=dict(signs=["a", "b"], lost=[]), x=dict(signs=["a"], lost=[]))
    with pytest.raises(SystemExit, match="no reason"):
        R.lost_rows(forms, ["a"])
    forms["x"]["lost"] = [dict(sign="b", reason="r")]
    assert R.lost_rows(forms, ["a"]) == dict(count=1, by_reason=dict(r=1), clips=[dict(sign="b", forms=dict(x=dict(reason="r", detail=None)))])


def test_the_lost_count_takes_each_reason_once_per_clip():
    forms = dict(real_two_view=dict(signs=["a", "b"]), x=dict(signs=["a"], lost=[dict(sign="b", reason="r")]), y=dict(signs=["a"], lost=[dict(sign="b", reason="r")]), z=dict(signs=["a"], lost=[dict(sign="b", reason="q")]))
    assert R.lost_rows(forms, ["a"])["by_reason"] == dict(q=1, r=1)


def test_restrict_moves_every_per_clip_field_together():
    read = dict(signs=["a", "b", "c"], groups=["1", "2", "3"], references=["ra", "rb", "rc"], hypotheses=["ha", "hb", "hc"], nll=[[1.], [2.], [3.]], seconds_over_reference=[.1, .2, .3], valid_frame_share=[.4, .5, .6], form="f", part_validity=dict(left_hand=[.7, .8, .9]),
                motion=dict(wrist_speed=[1, 2, 3], wrist_spread=[4, 5, 6]), hands=dict(left_hand=dict(speed=[7, 8, 9], variation=[1, 1, 2])), tracked=dict(wrists=[0, 1, 2], hands=[3, 4, 5]))
    out = R.restrict(read, ["c", "a"])
    assert out["signs"] == ["c", "a"] and out["groups"] == ["3", "1"] and out["references"] == ["rc", "ra"] and out["hypotheses"] == ["hc", "ha"] and out["nll"] == [[3.], [1.]] and out["seconds_over_reference"] == [.3, .1] and out["valid_frame_share"] == [.6, .4] and out["part_validity"] == dict(left_hand=[.9, .7])
    assert out["motion"] == dict(wrist_speed=[3, 1], wrist_spread=[6, 4]) and out["hands"]["left_hand"] == dict(speed=[9, 7], variation=[2, 1]) and out["tracked"] == dict(wrists=[2, 0], hands=[5, 3]) and out["form"] == "f" and read["signs"] == ["a", "b", "c"]
    del read["groups"], read["part_validity"]
    out = R.restrict(read, ["a"])
    assert "groups" not in out and "part_validity" not in out


def test_the_report_names_its_list_code_environment_and_reading(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, tmp_path, missing=(1,))
    assert h.main("lookups", "--out-dir", str(h.forms)) == 0
    report = h.pipeline()
    assert report["kind"] == "path-read-report" and report["report_only"] is True and report["note"] == R.NOT_DECIDED and report["declared_read"] is True and report["smoke"] is None
    assert report["code_digest"] == R.code_digest() and report["environment"] == P.environment() and report["instrument"]["sha256"] == "f" * 64 and report["list"]["list"] == "t" and report["tokenizer"]["sha256"] == L.TOKENIZER_SHA
    assert report["disclosures"] == json.loads(json.dumps(R.DISCLOSURES)) and report["monocular"]["unconformed"]["read"] is False and report["monocular"]["clips_loaded"] == N - 1 and report["monocular"]["conform"]["kinds"]["body:equivalent"] == N - 1
    assert report["lookups"]["matched"] == N - 1 and report["checks"]["control"]["reproduced"] is True and report["checks"]["blank_identical_to_real_two_view"] == 0. and report["rerun"] is None and report["clips"]["floor"] == 475
    assert report["reading"]["prestated"] == R.PRESTATED and set(report["resources"]) == set(R.FORMS) - {R.UNCONFORMED} and report["resources"]["blank"]["read_seconds"] >= 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["shared"] == N - 1 and printed["of"] == N and printed["lost"] == dict(no_monocular_row=1) and printed["forms"]["real_monocular"]["gap_share"] == report["forms"]["real_monocular"]["share_of_gap"]["bleu"] and printed["forms"]["real_monocular"]["valid_frame_share"] == report["forms"]["real_monocular"]["valid_frame_share"]["mean"]
    assert printed["pairing"]["offsets"] == report["pairing"]["offsets"] and printed["pairing"]["source_hash_matches"] == N - 1 and printed["round_trip_beside_bar"]["bar"] == 3.8 and printed["validity"]["real_monocular"]["left_hand"] == pytest.approx(np.mean([R.part_validity(s)["left_hand"] for i, s in enumerate(h.mono) if i != 1])) and printed["stored"]["v0_nearest_fill"]["clips"] == N and printed["case"] == report["reading"]["case"] and printed["withheld"] == report["reading"]["withheld"]


def test_the_pairing_block_and_its_regressions_rest_on_the_shared_clips(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=(2,))
    block = h.pipeline()["pairing"]
    plan = json.loads((h.forms / "monocular.json").read_text())["pairing"]
    assert block["clips"] == N - 1 == len(plan) and block["source_hash_matches"] == N - 1 and block["max_offset_seconds"] == 0. and block["frames_ratio_median"] == 1.2
    assert block["offsets"] == dict(clips=N - 1, nonzero=0, median=0., p90=0., p99=0., max=0., stored_fps={"24.0": N - 1})
    assert block["distance"]["left_wrist"]["mean"] == pytest.approx(np.mean([r["distance"]["left_wrist"] for r in plan]), abs=1e-4) and block["mirrored"]["left_wrist"]["mean"] > block["distance"]["left_wrist"]["mean"]
    assert block["group_distance"]["wrists"] == round(float(np.mean([R.group_distance(r["distance"], "wrists") for r in plan])), 5) and block["group_mirrored"]["hands"] > block["group_distance"]["hands"]
    assert set(block["regression"]) == {"marked", "trimmed", "marks_free"} and set(block["regression"]["marks_free"]) == set(R.GROUPS)


def test_the_offset_distribution_summarises_every_clip():
    rows = [dict(max_offset_seconds=v, stored_fps=f) for v, f in ((0., 24.), (0., 24.), (0., 30.), (.01, 60.), (.02, 24.))]
    assert R.offset_distribution(rows) == dict(clips=5, nonzero=2, median=0., p90=round(float(np.percentile([0, 0, 0, .01, .02], 90)), 6), p99=round(float(np.percentile([0, 0, 0, .01, .02], 99)), 6), max=.02, stored_fps={"24.0": 3, "30.0": 1, "60.0": 1})


def test_the_regressions_take_each_pairs_own_difference_and_sign(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    rows = h.pipeline()["pairing"]["regression"]
    assert rows["marked"]["wrists"]["intercept"] == pytest.approx(-1., abs=1e-6) and rows["marks_free"]["wrists"]["intercept"] == pytest.approx(-.7, abs=1e-6) and rows["trimmed"]["wrists"]["intercept"] == pytest.approx(-.9, abs=1e-6)


def test_the_pairing_rests_on_the_shared_clips_alone(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    h.pipeline()
    path = h.forms / "form-real_monocular_trimmed.json"
    form = json.loads(path.read_text())
    keep = list(range(1, N))
    for key in ("signs", "groups", "references", "hypotheses", "nll", "seconds_over_reference", "valid_frame_share"):
        form[key] = [form[key][i] for i in keep]
    form["part_validity"] = {k: [v[i] for i in keep] for k, v in form["part_validity"].items()}
    form["motion"] = {k: [v[i] for i in keep] for k, v in form["motion"].items()}
    form["hands"] = {h_: {k: [v[i] for i in keep] for k, v in row.items()} for h_, row in form["hands"].items()}
    form["tracked"] = {k: [v[i] for i in keep] for k, v in form["tracked"].items()}
    form["lost"] = [dict(sign="s0", reason="planted")]
    path.write_text(json.dumps(form))
    report = h.merge("again.json")
    assert report["clips"]["shared"] == N - 1 and report["pairing"]["clips"] == N - 1 and report["clips"]["outside_shared"]["by_reason"] == dict(planted=1)


def test_a_timestamp_offset_reaches_the_pairing_and_the_alert(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.refs[1] = replace(h.refs[1], timestamps=h.refs[1].timestamps + .01)
    h.plant_two_view()
    report = h.pipeline()
    plan = json.loads((h.forms / "monocular.json").read_text())["pairing"]
    assert plan[1]["max_offset_seconds"] == .01 and plan[0]["max_offset_seconds"] == 0. and report["pairing"]["max_offset_seconds"] == .01 and any("stored clocks differ by up to 0.0100" in a for a in report["alerts"])


def test_an_emptied_monocular_hand_alerts_on_the_monocular_side_alone(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, mono=lambda i: with_empty_hand(monocular_state(i)))
    report = h.pipeline()
    assert any("monocular left_hand valid on 0.000" in a for a in report["alerts"]) and any("left_hand validity is 0.000" in a for a in report["alerts"]) and not any("two-view left_hand" in a for a in report["alerts"]) and report["reading"]["case"] is None
    assert report["forms"]["real_monocular"]["part_validity"]["left_hand"]["mean"] == 0. and report["forms"]["real_two_view"]["part_validity"]["left_hand"]["mean"] > .5


def test_the_regression_recovers_a_planted_slope_and_says_report_only():
    x = np.arange(40.)
    noise = np.random.default_rng(0).normal(size=40) * .1
    out = R.regression(x.tolist(), (.5 * x + 2 + noise).tolist())
    assert out["report_only"] is True and out["clips"] == 40 and out["slope"] == pytest.approx(.5, abs=.01) and out["intercept"] == pytest.approx(2, abs=.1) and out["r"] > .99 and out["slope_ci95"][0] < .5 < out["slope_ci95"][1]
    assert R.regression([1., 2.], [1., 2.]) is None and R.regression([1., 1., 1.], [1., 2., 3.]) is None and R.regression([1., None, 2., 3.], [2., 9., 4., 6.])["clips"] == 3
    assert R.regression([1., 2., 3.], [1., 1., 1.])["r"] is None


def test_a_read_unconformed_form_gets_its_own_panel(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    original = layout_adapter.conform
    monkeypatch.setattr(layout_adapter, "conform", lambda state, template: gappy(original(state, template), 8, 12))
    report = h.pipeline()
    assert report["monocular"]["unconformed"]["read"] is True and R.UNCONFORMED in report["forms"] and R.UNCONFORMED in report["per_signer"]["c01"]


def test_a_merge_refuses_what_is_not_one_run(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    h.to_control()
    with pytest.raises(SystemExit, match="monocular.json is missing"):
        h.merge()
    assert h.read(*R.MONOCULAR_FORMS, resume=True) == 0
    trip = h.forms / "form-round_trip_monocular.json"
    good = trip.read_text()

    def expect(change, match):
        trip.write_text(json.dumps(dict(json.loads(good), **change)))
        with pytest.raises(SystemExit, match=match):
            h.merge()
    expect(dict(code_digest="other"), "different code")
    expect(dict(environment=dict(gpu="other")), "different environments")
    expect(dict(form="other"), "holds form")
    expect(dict(declared_read=False), "smoke mixed")
    expect(dict(instrument=dict(sha256="0" * 64)), "another instrument")
    expect(dict(list=dict(list="other")), "another list")
    trip.unlink()
    with pytest.raises(SystemExit, match="form-round_trip_monocular.json is missing"):
        h.merge()
    trip.write_text(good)
    assert h.merge()["clips"]["shared"] == N


def test_a_merge_refuses_a_monocular_record_of_another_run_and_a_form_it_says_was_not_read(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    h.pipeline()
    path = h.forms / "monocular.json"
    good = json.loads(path.read_text())
    for change in (dict(code_digest="other"), dict(declared_read=False)):
        path.write_text(json.dumps(dict(good, **change)))
        with pytest.raises(SystemExit, match="another run's record"):
            h.merge("again.json")
    path.write_text(json.dumps(good))
    stray = h.forms / f"form-{R.UNCONFORMED}.json"
    stray.write_text((h.forms / "form-real_monocular.json").read_text())
    with pytest.raises(SystemExit, match="record says it was not"):
        h.merge("again.json")
    stray.unlink()
    path.write_text(json.dumps(dict(good, unconformed=dict(good["unconformed"], read=True))))
    with pytest.raises(SystemExit, match=f"form-{R.UNCONFORMED}.json is missing"):
        h.merge("again.json")


def test_a_merge_with_no_shared_clip_refuses(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.pipeline()
    for name, pick in (("real_monocular", [0]), ("round_trip_monocular", [1])):
        path = h.forms / f"form-{name}.json"
        form = json.loads(path.read_text())
        form.update(signs=[form["signs"][i] for i in pick], references=[form["references"][i] for i in pick], lost=[])
        path.write_text(json.dumps(form))
    with pytest.raises(SystemExit, match="share no clip"):
        h.merge("again.json")


def test_a_merge_is_write_once(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.pipeline()
    with pytest.raises(SystemExit, match="write-once"):
        h.main("merge", "--out-dir", str(h.forms), "--output-file", str(tmp_path / "report.json"))
    out = "pr/report.json"
    assert h.main("merge", "--out-dir", str(h.forms), "--output-key", out) == 0 and json.loads(h.store.get(out))["report_only"] is True
    with pytest.raises(SystemExit, match="write-once"):
        h.main("merge", "--out-dir", str(h.forms), "--output-key", out)
    with pytest.raises(SystemExit, match="exactly one"):
        h.main("merge", "--out-dir", str(h.forms))


def test_a_merge_refuses_a_taken_output_before_it_loads_anything(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    taken = tmp_path / "taken.json"
    taken.write_text("{}")
    monkeypatch.setattr(R, "load_reads", lambda *a: pytest.fail("the forms were loaded"))
    with pytest.raises(SystemExit, match="write-once"):
        h.main("merge", "--out-dir", str(h.forms), "--output-file", str(taken))


def test_a_smoke_runs_every_form_on_train_clips_and_says_so(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, tmp_path, declared=False, n=5, missing=(4,))
    assert h.read(smoke=True) == 0
    h.plant_control()
    assert h.control() == 0
    report = h.merge()
    assert report["declared_read"] is False and report["smoke"].startswith("SMOKE on c02 train clips") and report["clips"]["shared"] == 4 and report["checks"]["control"] is None and report["alerts"] == []
    assert report["reading"]["case"] is None and report["reading"]["withheld"] == ["a smoke carries no reading"] and report["reading"]["marks_free"] is None and report["reading"]["marked_secondary"]["bleu"]["difference"] < 0
    assert report["stored"]["rt_nearest"]["shared"] is None and "checked, not compared" in report["stored"]["rt_nearest"]["note"] and report["stored"]["rt_nearest"]["all_500"]["clips"] == 5 and report["stored"]["v0_nearest_fill"]["shared"] is None
    block = report["same_path"]["bleu"]
    assert block["marks_free"]["stored_rt_nearest_share_of_two_view_gap"] is None and block["round_trip"]["monocular_minus_stored"] is None and block["marked"]["monocular_gap"] is not None and "marks_free" not in report["pairing"]["regression"]
    assert list(report["per_signer"]) == ["c02"]
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["case"] is None
    path = tmp_path / "report.json"
    assert R.main(["check", "--report", str(path)]) == 0


def test_an_alert_on_a_smoke_makes_check_exit_3(monkeypatch, tmp_path, capsys):
    from signtranslator.pose.mirror import mirror_state
    h = Harness(monkeypatch, tmp_path, declared=False, n=5, mono=lambda i: mirror_state(layout_adapter.conform(monocular_state(i), template_state())))
    # a clip mirrored in extraction: the two-view clips stay as they are
    h.refs = [two_view_state(i, h.refs[i].channels["body"].source_sha256) for i in range(5)]
    assert h.read(smoke=True) == 0
    h.plant_control()
    assert h.control() == 0
    report = h.merge()
    assert any("mirrored monocular clip sits as near" in a for a in report["alerts"]) and report["reading"]["case"] is None
    assert R.main(["check", "--report", str(tmp_path / "report.json")]) == 3
    assert "ALERT" in capsys.readouterr().out


def test_a_source_hash_that_differs_withholds_the_label_and_check_alerts(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, tmp_path)
    h.refs[0] = two_view_state(0, "b" * 64)
    h.reader.truth[h.records[0]["text"]] = (h.refs[0], h.mono[0])
    report = h.pipeline()
    assert report["pairing"]["source_hash_matches"] == N - 1 and any("do not name their monocular row's hash" in a for a in report["alerts"])
    assert report["reading"]["case"] is None and report["reading"]["withheld"] == report["alerts"] and report["reading"]["marks_free"]["case"] == "b"
    assert R.main(["check", "--report", str(tmp_path / "report.json")]) == 3
    assert "ALERT 1 of 9 two-view clips" in capsys.readouterr().out


def test_a_shared_count_under_the_floor_withholds_the_label(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    report = h.pipeline()
    assert report["clips"]["shared"] == N and any(f"{N} shared clips, under 475" in a for a in report["alerts"]) and report["reading"]["case"] is None


def test_at_the_floor_a_clean_read_carries_its_label(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "SHARED_FLOOR", N)
    report = h.pipeline()
    assert report["alerts"] == [] and report["reading"]["withheld"] == [] and report["reading"]["case"] == "b" and report["reading"]["marks_free"]["bleu"]["case"] == "b" and report["reading"]["marks_free"]["likelihood_minus_blank"]["case"] == "b"
    assert R.main(["check", "--report", str(tmp_path / "report.json")]) == 0


def test_the_label_rests_on_the_marks_free_pair_not_the_marked_one(monkeypatch, tmp_path):
    """The marked pair reads lower; the marks-free pair matches stored: the label follows the second."""
    h = Harness(monkeypatch, tmp_path, levels=dict(nearest_monocular=(11, 1.5)))
    monkeypatch.setattr(R, "SHARED_FLOOR", N)
    report = h.pipeline()
    assert report["reading"]["marked_secondary"]["case"] == "b" and report["reading"]["marks_free"]["case"] == "a" and report["reading"]["case"] == "a"
    assert report["same_path"]["likelihood_minus_blank"]["marks_free"]["monocular_gap_over_two_view_gap"]["point"] == 1. and report["same_path"]["likelihood_minus_blank"]["marked"]["monocular_gap_over_two_view_gap"]["point"] == .5


# ---- the label on realistic intervals ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("difference,paired,a,b,threshold,case", [
    (0., [-.2, .2], [4., 5.], [4., 5.], .5, "a"), (-.5, [-.5, .1], [4., 5.], [3.5, 4.5], .5, "a"), (.5, [-.1, 1.], [4., 5.], [4.5, 5.5], .5, "a"),
    (-.5001, [-1., .1], [4., 5.], [3.5, 4.5], .5, "b"), (-2., [-3., -1.], [4., 5.], [2., 3.], .5, "b"), (-.6, [-1., 0.], [4., 5.], [3.9, 4.5], .5, "b"),
    (.5001, [-.1, 1.], [4., 5.], [4.5, 5.5], .5, "c"), (.2, [.05, .4], [4., 5.], [4.2, 5.2], .5, "c"), (.2, [-.05, .4], [4., 5.], [4.2, 5.2], .5, "a"), (.2, [0., .4], [4., 5.], [4.2, 5.2], .5, "a"),
    (-.2, [-.4, -.05], [4., 5.], [3.8, 4.8], .5, "a"), (-.2, [-.51, .1], [3.9, 5.1], [3.7, 4.9], .5, "unresolved"), (-.4, [-.6, -.1], [4.4, 4.5], [4., 4.1], .5, "unresolved"), (.4, [-.1, .6], [4., 4.1], [4.4, 4.5], .5, "unresolved"), (-.1, [-.2, 0.], [4., 4.5], [4.5, 5.], .5, "a"), (-.1, [.01, .2], [4., 5.], [4., 5.], .5, "a"),
    (-.04, [-.06, -.02], [.2, .25], [.17, .22], .052, "unresolved"), (-.04, [-.052, -.02], [.2, .25], [.17, .22], .052, "a"), (-.053, [-.07, -.03], [.2, .25], [.16, .21], .052, "b"), (.06, [.01, .1], [.2, .25], [.23, .28], .052, "c")])
def test_the_prestated_case_of_one_metric(difference, paired, a, b, threshold, case):
    assert R.prestated_case(difference, paired, a, b, threshold) == case


@pytest.mark.parametrize("bleu,likelihood,case", [("a", "a", "a"), ("b", "b", "b"), ("c", "c", "c"), ("unresolved", "unresolved", "unresolved"), ("a", "b", "unresolved"), ("b", "a", "unresolved"), ("c", "a", "unresolved"), ("a", "unresolved", "unresolved")])
def test_the_metrics_disagreeing_reads_unresolved(bleu, likelihood, case):
    assert R.joint_case(dict(bleu=bleu, likelihood=likelihood)) == case


def test_overlaps():
    assert R.overlaps([1., 2.], [2., 3.]) and R.overlaps([2., 3.], [1., 2.]) and not R.overlaps([1., 2.], [2.1, 3.]) and not R.overlaps([2.1, 3.], [1., 2.])


def thousand_like_d20(seed, bleu_words=(6, .28), n=500):
    """Hypotheses and references near D20's: rare short matches among 500 sentences, so BLEU sits near 4.5 with an interval about as wide as D20's."""
    rng = np.random.default_rng(seed)
    refs = [" ".join(f"r{rng.integers(3000)}" for _ in range(14)) for _ in range(n)]
    return rng, refs


def hypotheses(rng, refs, p):
    out = []
    for ref in refs:
        words = ref.split()
        hyp = [f"h{rng.integers(3000)}" for _ in range(13)]
        if rng.random() < p:
            k = int(rng.integers(0, 10))
            size = int(rng.integers(2, 5))
            hyp[k:k + size] = words[k:k + size]
        out.append(" ".join(hyp))
    return out


def realistic_pair(p_mono, p_two, seed=0, shift=0., noise=.35, n=500):
    rng, refs = thousand_like_d20(seed, n=n)
    mono, two = hypotheses(rng, refs, p_mono), hypotheses(rng, refs, p_two)
    base = rng.normal(size=n) * noise
    blank = dict(references=refs, hypotheses=["z"] * n, nll=np.c_[np.full(n, 5.)].tolist())
    return (dict(references=refs, hypotheses=mono, nll=np.c_[5. - .29 + shift + base + rng.normal(size=n) * .05].tolist()), dict(references=refs, hypotheses=two, nll=np.c_[5. - .29 + base + rng.normal(size=n) * .05].tolist()), blank)


def test_at_500_clips_bleu_intervals_are_as_wide_as_d20s():
    a, b, blank = realistic_pair(.5, .5)
    facts = R.pair_facts(a, b, blank)
    width = facts["bleu"]["two_view_ci95"][1] - facts["bleu"]["two_view_ci95"][0]
    assert 2 < facts["bleu"]["two_view"] < 12 and .6 < width / max(facts["bleu"]["two_view"] / 4.5, 1) < 3.5        # D20's stored round trip: 4.124 [3.485, 4.764], width 1.28
    likelihood = facts["likelihood_minus_blank"]
    assert .25 < likelihood["two_view"] < .33 and likelihood["two_view_ci95"][1] - likelihood["two_view_ci95"][0] < .08


def test_at_500_clips_a_small_difference_with_noisy_pairs_does_not_read_as_no_cap():
    """Independent hypotheses of equal level: the point sits near zero and the paired bound usually under -0.5, which is unresolved under the header, not (a)."""
    seen = set()
    for seed in range(6):
        a, b, blank = realistic_pair(.5, .5, seed=seed)
        facts = R.pair_facts(a, b, blank)
        row = facts["bleu"]
        assert row["case"] == R.prestated_case(row["difference"], row["ci95"], row["monocular_ci95"], row["two_view_ci95"], .5)
        seen.add(row["case"])
        if abs(row["difference"]) <= .5 and row["ci95"][0] < -.5:
            assert row["case"] == "unresolved"
    assert "unresolved" in seen


def test_at_500_clips_a_likelihood_drop_past_its_threshold_is_b_and_a_small_one_is_not():
    a, b, blank = realistic_pair(.5, .5, shift=.09)
    low = R.pair_facts(a, b, blank)["likelihood_minus_blank"]
    assert low["difference"] < -.052 and low["case"] == "b" and low["threshold"] == .052 and low["monocular_higher"] is False
    a, b, blank = realistic_pair(.5, .5, shift=.0)
    near = R.pair_facts(a, b, blank)["likelihood_minus_blank"]
    assert abs(near["difference"]) < .02 and near["case"] == "a" and near["intervals_overlap"] is True


def test_pair_facts_report_each_metric_and_their_joint_case():
    refs = [r["text"] for r in d20_records(30)]

    def form(keep, nll):
        return dict(references=refs, hypotheses=[" ".join(t.split()[:keep(i)]) for i, t in enumerate(refs)], nll=[[nll + i / 100.] for i in range(30)])
    facts = R.pair_facts(form(lambda i: 6, 2.), form(lambda i: 6 + i % 9, 1.), form(lambda i: 4, 3.))
    bleu, likelihood = facts["bleu"], facts["likelihood_minus_blank"]
    assert bleu["monocular"] == M.bleu_ci_fast(form(lambda i: 6, 2.)["hypotheses"], refs)[0] and bleu["two_view"] == M.bleu_ci_fast(form(lambda i: 6 + i % 9, 1.)["hypotheses"], refs)[0] and bleu["threshold"] == .5 and bleu["monocular_higher"] is False
    assert bleu["difference"] == M.paired_bleu_fast(M.bleu_statistics(form(lambda i: 6, 2.)["hypotheses"], refs), M.bleu_statistics(form(lambda i: 6 + i % 9, 1.)["hypotheses"], refs))["difference"]
    assert likelihood["monocular"] == pytest.approx(1.) and likelihood["two_view"] == pytest.approx(2.) and likelihood["difference"] == pytest.approx(-1.) and likelihood["ci95"] == [-1., -1.] and likelihood["case"] == "b"
    assert facts["case"] == R.joint_case(dict(bleu=bleu["case"], likelihood=likelihood["case"]))


@pytest.mark.parametrize("keep,nll,bleu_case,likelihood_case", [(6, 2., "a", "b"), (12, 1., "c", "a")])
def test_one_metric_alone_does_not_carry_the_label(keep, nll, bleu_case, likelihood_case):
    refs = [r["text"] for r in d20_records(30)]

    def form(k, value):
        return dict(references=refs, hypotheses=[" ".join(t.split()[:k]) for t in refs], nll=[[value + i / 100.] for i in range(30)])
    facts = R.pair_facts(form(keep, nll), form(6, 1.), form(4, 3.))
    assert facts["bleu"]["case"] == bleu_case and facts["likelihood_minus_blank"]["case"] == likelihood_case and facts["case"] == "unresolved"


def test_a_form_higher_on_both_metrics_reads_c():
    refs = [r["text"] for r in d20_records(30)]

    def form(keep, nll):
        return dict(references=refs, hypotheses=[" ".join(t.split()[:keep]) for t in refs], nll=[[nll + i / 100.] for i in range(30)])
    facts = R.pair_facts(form(12, 1.), form(6, 2.), form(4, 3.))
    assert facts["case"] == "c" and facts["bleu"]["monocular_higher"] is True and facts["likelihood_minus_blank"]["monocular_higher"] is True


# ---- the header -------------------------------------------------------------------------------------------------------------------

def test_the_reading_is_stated_in_the_header_before_any_code():
    assert all(text in R.__doc__ for text in R.PRESTATED.values()) and set(R.PRESTATED) == {"a", "b", "c", "unresolved", "settled", "zero_read"}
    doc = R.__doc__
    assert "report only" in doc.lower() and "(b) has no interval condition" in doc and "no step of the job branches on it" in doc and "No reading here makes a claim about composition" in doc
    assert "lost content, marks, length and v1's fit are not separated" in R.PRESTATED["b"] and "composition" not in R.PRESTATED["b"]
    assert "4.493 to 4.621" in doc and "+0.2920 to +0.2337" in doc and "0.052 nats" in doc and "17.8%" in doc and "paired lower bound" in R.PRESTATED["a"]
    assert "fine-tuned on How2Sign FK keypoints" in doc and "below 475 of D20's 500" in doc and "docs/RUNPOD.md" in doc and "hip-gated" in doc and "extracted as YouTube's states are" not in doc


@pytest.mark.parametrize("text", ["at least 0.5; each hand at least 0.3", "Monocular over two-view at least 0.5", "shoulders at most 0.5 and elbows and wrists at most 1.0 shoulder widths; hands at most 3.0 palm widths", "on elbows, wrists and hands", "at most 0.001 s"])
def test_the_alert_thresholds_are_fixed_in_the_header(text):
    assert text in R.__doc__


def test_the_threshold_is_the_stored_figures_arithmetic():
    gap = 4.493 - 1.68
    assert .5 / gap == pytest.approx(.178, abs=.001) and .5 / gap * .2920 == pytest.approx(R.THRESHOLD["likelihood"], abs=.0005)


def test_the_disclosures_name_the_instruments_domain_extraction_and_blank():
    assert "two-view" in R.DISCLOSURES["instrument"] and "pretrained" in R.DISCLOSURES["instrument"] and "durations" in R.DISCLOSURES["blank"] and "1.00-1.05" in R.DISCLOSURES["frames"] and "upright_chest" in R.DISCLOSURES["extraction"]


@pytest.mark.parametrize("kw,code", [(dict(report_only=True, alerts=[], declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3)), 0), (dict(report_only=True, alerts=[], declared_read=True, checks=dict(blank_identical_to_real_two_view=0., control=dict(reproduced=True)), clips=dict(shared=3)), 0),
                                     (dict(report_only=False, alerts=[], declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3)), 3), (dict(report_only=True, alerts=[], declared_read=False, checks=dict(blank_identical_to_real_two_view=1.), clips=dict(shared=3)), 3),
                                     (dict(report_only=True, alerts=[], declared_read=True, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3)), 3), (dict(report_only=True, alerts=[], declared_read=True, checks=dict(blank_identical_to_real_two_view=0., control=dict(reproduced=False)), clips=dict(shared=3)), 3),
                                     (dict(report_only=True, alerts=[], declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=0)), 3), (dict(report_only=True, alerts=["x"], declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3)), 3),
                                     (dict(report_only=False, alerts=[], reading=dict(case="a"), declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3)), 3)])
def test_check_judges_plumbing_only(tmp_path, kw, code):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(kw))
    assert R.main(["check", "--report", str(path)]) == code


def test_check_calls_out_a_label_beside_an_alert(tmp_path, capsys):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(dict(report_only=True, alerts=["x"], reading=dict(case="a"), declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3))))
    assert R.main(["check", "--report", str(path)]) == 3
    assert "ALERT a report holding a case label beside an alert" in capsys.readouterr().out


def test_preflight_refuses_a_finished_read_and_a_voided_one(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, tmp_path, n=2)
    args = ("preflight", "--out-dir", str(tmp_path / "d"), "--final-key", "pr/report.json", "--void-prefix", "pr/void")
    assert h.main(*args) == 0 and R.code_digest() in capsys.readouterr().out
    h.store.put(f"pr/void/{R.code_digest()}.json", b"{}")
    with pytest.raises(SystemExit, match="void"):
        h.main(*args)
    h.store.put("pr/report.json", b"{}")
    with pytest.raises(SystemExit, match="finished"):
        h.main(*args)


def test_preflight_after_another_codes_void_needs_a_cause_and_records_it(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=2)
    h.store.put("pr/void/other.json", b"{}")
    args = ("preflight", "--out-dir", str(tmp_path / "d"), "--final-key", "pr/report.json", "--void-prefix", "pr/void")
    with pytest.raises(SystemExit, match="rerun-cause"):
        h.main(*args)
    assert h.main(*args, "--rerun-cause", "why") == 0
    assert json.loads((tmp_path / "d/forms/rerun.json").read_text())["cause"] == "why"


def test_a_rerun_record_reaches_the_report(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.forms.mkdir(parents=True)
    (h.forms / "rerun.json").write_text(json.dumps(dict(cause="why")))
    assert h.pipeline()["rerun"] == dict(cause="why")


def test_upload_is_write_once_and_a_partial_keeps_what_is_stored(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=2)
    work = tmp_path / "work"
    (work / "forms").mkdir(parents=True)
    (work / "forms" / "form-x.json").write_text("{}")
    (work / "report.json").write_text("{}")
    prefix = R.PREFIX
    assert h.main("upload", "--out-dir", str(work), "--prefix", prefix) == 0 and h.store.exists(f"{prefix}/report.json") and h.store.exists(f"{prefix}/forms/form-x.json")
    with pytest.raises(SystemExit, match="write-once"):
        h.main("upload", "--out-dir", str(work), "--prefix", prefix)
    assert h.main("upload", "--partial", "--out-dir", str(work), "--prefix", prefix) == 0


def test_upload_log_is_write_once(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=2)
    log = tmp_path / "x.log"
    log.write_text("hello")
    assert h.main("upload-log", "--file", str(log), "--key", "pr/logs/a.log") == 0 and h.store.get("pr/logs/a.log") == b"hello"
    with pytest.raises(SystemExit, match="write-once"):
        h.main("upload-log", "--file", str(log), "--key", "pr/logs/a.log")


def test_the_code_digest_holds_the_job_file():
    assert R.JOB_FILE == "runpod/fleet/path_read.sh" and R.code_digest() == provenance.code_digest(extra_files=(R.JOB_FILE,)) and R.code_digest() != provenance.code_digest()


def test_a_digest_that_cannot_be_named_voids(monkeypatch):
    R.code_digest.cache_clear()
    monkeypatch.setattr(provenance, "code_digest", lambda extra_files=(): (_ for _ in ()).throw(OSError("gone")))
    with pytest.raises(SystemExit, match="no code digest"):
        R.code_digest()
    R.code_digest.cache_clear()


def test_the_constants_are_the_declared_ones():
    assert R.PREFIX == "acquisition/2026-10-17-v1/youtube-asl-v1/round20/path-read" and R.BOOT == 2000 and R.ANCHORS == ("real_two_view", "blank", "rt_nearest_two_view")
    assert R.MONOCULAR_FORMS == ("real_monocular", "real_monocular_unconformed", "nearest_monocular", "round_trip_monocular", "real_monocular_trimmed", "monocular_values_two_view_marks", "two_view_values_monocular_marks") and R.FORMS == R.ANCHORS + R.MONOCULAR_FORMS
    assert "a quarter of that clip's own stored frame interval" in R.__doc__ and "46 D20 clips including all 18 report rows" in R.__doc__ and "0.0333 s" in R.__doc__ and "at_rate" in R.__doc__ and "A shift of whole frames shows on no clock" in R.__doc__
    assert R.TEMPLATE_SIGN == "--7E2sU6zP4_10-5-rgb_front" and R.MONOCULAR == "acquisition/2026-10-17-v1/derived/how2sign-video-estimate-v1" and R.state_key_of("x") == f"{R.MONOCULAR}/states/x.npz" and R.FPS == 24.


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
    lines = [x for x in result.stdout.splitlines() if x.startswith("PATH-READ-DONE ")]
    assert len(lines) == 1 and lines[0] == result.stdout.strip().splitlines()[-1]
    return int(lines[0].split()[1])


def test_the_runner_sees_no_credentials():
    assert not [k for k in os.environ if k.startswith(("R2_", "AWS_", "RUNPOD_"))]


def test_the_job_checks_the_stored_forms_first_then_smokes_then_reads_the_control_first(tmp_path):
    result, calls = run_job(tmp_path)
    assert result.returncode == 0 and done_line(result) == 0
    own = [x for x in calls.splitlines() if x.startswith("scripts/path_read.py ")]
    assert len(own) == len(calls.splitlines())
    parsed = [R.build_parser().parse_args(x.split()[1:]) for x in own]
    assert [a.command for a in parsed] == ["stored-check", "lookups", "lookups", "read", "control", "merge", "upload", "check", "preflight", "lookups", "read", "control", "read", "merge", "upload", "check", "upload-log"]
    assert parsed[1].smoke_train == 0 and parsed[1].out_dir.endswith("/d20") and "smoke-" in parsed[1].out_dir and parsed[2].smoke_train == 5
    reads = [a for a in parsed if a.command == "read"]
    assert reads[0].smoke_train == 5 and reads[0].smoke_longest == 4 and reads[0].smoke_never_tracked == 2 and reads[0].forms == ",".join(R.FORMS)
    assert reads[1].forms == ",".join(R.ANCHORS) and reads[1].smoke_train == 0 and reads[2].forms == ",".join(R.MONOCULAR_FORMS) and reads[2].smoke_train == 0 and reads[1].resume and reads[2].resume and reads[1].smoke_longest == 0 and reads[2].smoke_never_tracked == 0
    assert all(a.device == "cuda" and a.nll_group == 4 and a.threads == 8 and a.instrument_cache.endswith("/instrument") for a in reads)
    uploads = [a for a in parsed if a.command == "upload"]
    assert uploads[0].prefix.startswith(f"{R.PREFIX}/smoke/") and "smoke-" in uploads[0].out_dir and uploads[1].prefix == R.PREFIX and uploads[1].out_dir.endswith("/d20")
    assert next(a for a in parsed if a.command == "preflight").final_key == f"{R.PREFIX}/report.json" and next(a for a in parsed if a.command == "preflight").void_prefix == f"{R.PREFIX}/void" and next(a for a in parsed if a.command == "control").void_prefix == f"{R.PREFIX}/void"
    log = parsed[-1]
    assert log.key.startswith(f"{R.PREFIX}/logs/") and log.key.endswith(".log") and log.file.endswith("/path_read.log")


def test_smoke_only_stops_before_any_declared_read(tmp_path):
    result, calls = run_job(tmp_path, SMOKE="1")
    assert result.returncode == 0 and done_line(result) == 0 and "SMOKE done" in result.stdout and "preflight" not in calls and "/d20/forms" not in calls and "path_read.py upload --out-dir" in calls
    assert "--smoke-train" in calls and "read --forms" not in calls and "SMOKE done" in (tmp_path / "work" / "path_read.log").read_text() and "upload-log" in calls.splitlines()[-1]


@pytest.mark.parametrize("fail_when,expected", [("path_read.py stored-check", 1), ("path_read.py lookups --out-dir*smoke-*/d20", 2), ("path_read.py lookups*smoke-train", 2), ("path_read.py read*smoke-train", 2), ("path_read.py control*smoke-", 2), ("path_read.py merge*smoke-", 2), ("path_read.py check --report*smoke-", 2), ("path_read.py upload --out-dir*smoke-", 2),
                                                ("path_read.py preflight", 1), ("path_read.py lookups --out-dir*/work/d20/forms", 1), ("--forms real_two_view", 1), ("--forms real_monocular", 1), ("path_read.py merge*/work/d20", 1), ("path_read.py upload --out-dir*/work/d20", 1), ("path_read.py check --report*/work/d20", 1)])
def test_any_failure_exits_nonzero_with_its_done_line_through_the_pipe_uploads_its_work_and_its_log(tmp_path, fail_when, expected):
    result, calls = run_job(tmp_path, fail_when=fail_when)
    assert result.returncode == expected and done_line(result) == expected, result.stdout + result.stderr
    lines = calls.splitlines()
    assert "upload --partial" in lines[-2] and f"{R.PREFIX}/partial/" in lines[-2] and "upload-log" in lines[-1] and "ALERT" in result.stdout and "planted failure" in result.stdout
    assert "ALERT" in (tmp_path / "work" / "path_read.log").read_text() and "planted failure" in (tmp_path / "work" / "path_read.log").read_text()


def test_a_stored_form_that_fails_stops_before_the_smoke(tmp_path):
    _, calls = run_job(tmp_path, fail_when="path_read.py stored-check")
    assert "path_read.py lookups" not in calls and "path_read.py read" not in calls


def test_a_smoke_failure_reads_nothing_of_d20(tmp_path):
    _, calls = run_job(tmp_path, fail_when="path_read.py read*smoke-train")
    assert "preflight" not in calls and "real_two_view,blank" not in calls


def test_a_smoke_alert_stops_the_job_with_its_report_stored_and_no_d20_read(tmp_path):
    result, calls = run_job(tmp_path, fail_when="path_read.py check --report*smoke-", code=3)
    assert result.returncode == 2 and done_line(result) == 2 and "preflight" not in calls and calls.index("path_read.py upload --out-dir") < calls.index("path_read.py check --report") and "its report is stored" in result.stdout


def test_a_control_miss_stops_before_any_monocular_form_with_its_own_status(tmp_path):
    result, calls = run_job(tmp_path, fail_when="path_read.py control*/work/d20", code=3)
    assert result.returncode == 3 and done_line(result) == 3 and "do not retry" in result.stdout
    assert "--forms real_monocular" not in calls and "merge" not in calls.split("preflight")[-1] and "upload --partial" in calls.splitlines()[-2]


def test_a_control_that_did_not_run_is_a_failed_read_not_a_void(tmp_path):
    result, calls = run_job(tmp_path, fail_when="path_read.py control*/work/d20", code=1)
    assert result.returncode == 1 and done_line(result) == 1 and "did not run" in result.stdout and "--forms real_monocular" not in calls


def test_a_control_failure_in_the_smoke_stops_the_job(tmp_path):
    result, calls = run_job(tmp_path, fail_when="path_read.py control*smoke-", code=3)
    assert result.returncode == 2 and done_line(result) == 2 and "preflight" not in calls


def test_a_log_that_cannot_be_written_is_not_hidden_by_the_pipe(tmp_path):
    result, calls = run_job(tmp_path, tee_fails=True)
    assert result.returncode == 1 and done_line(result) == 1 and "path_read.py lookups" not in calls and "upload --partial" in calls.splitlines()[-2]


def test_a_failure_with_tee_succeeding_keeps_python_status(tmp_path):
    result, _ = run_job(tmp_path, fail_when="path_read.py preflight", code=9)
    assert result.returncode == 1 and done_line(result) == 1


def test_the_log_holds_every_steps_output(tmp_path):
    run_job(tmp_path)
    text = (tmp_path / "work" / "path_read.log").read_text()
    assert text.count("ran scripts/path_read.py") == 16


def test_a_log_upload_that_fails_keeps_the_jobs_status(tmp_path):
    result, _ = run_job(tmp_path, fail_when="path_read.py upload-log", code=1)
    assert result.returncode == 0 and done_line(result) == 0 and "ALERT log upload failed" in result.stdout


def test_a_rerun_passes_its_cause(tmp_path):
    _, calls = run_job(tmp_path, RERUN_CAUSE="why now")
    assert "--rerun-cause why now" in next(x for x in calls.splitlines() if "preflight" in x)


def test_no_rerun_cause_passes_none(tmp_path):
    _, calls = run_job(tmp_path)
    assert "--rerun-cause" not in calls


def test_an_install_failure_stops_before_anything_reads(tmp_path):
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    (bin_dir / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\nexit 0\n')
    (bin_dir / "uv").write_text("#!/bin/bash\nexit 4\n")
    for name in ("python", "uv"):
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    done = subprocess.run(["bash", str(JOB)], env=dict(PATH=f"{bin_dir}:/usr/bin:/bin", HOME=str(tmp_path), VENV_BIN=str(bin_dir), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path), WORK=str(tmp_path / "work")), capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and done_line(done) == 1 and "transformers install failed" in done.stdout and "path_read.py read" not in log.read_text() and "upload --partial" in log.read_text()


def test_a_missing_repo_prints_its_done_line(tmp_path):
    done = subprocess.run(["bash", str(JOB)], env=dict(PATH="/usr/bin:/bin", HOME=str(tmp_path), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path / "gone")), capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and done.stdout.strip().endswith("PATH-READ-DONE 1")


def test_the_job_passes_every_step_and_alert_through_pipestatus_names_no_other_list_and_is_valid_bash():
    text = JOB.read_text()
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0
    code = [line.split(" #")[0] for line in text.splitlines() if not line.startswith("#")]
    pipes = [line for line in code if re.search(r"(?<!\|)\|(?!\|)", line)]
    assert pipes == ['  "$@" 2>&1 | tee -a "$LOG"'] and "${PIPESTATUS[@]}" in text and "set -uo pipefail" in text and "PATH-READ-DONE" in text and "transformers==4.57.6" in text
    python_lines = [line.strip() for line in code if re.search(r"\bpython \$T\b", line)]
    assert python_lines and all(line.startswith("step python $T") or "upload --partial" in line or "upload-log" in line for line in python_lines)
    assert [line.strip() for line in code if 'echo "ALERT' in line] == ['[ "${codes[1]}" = 0 ] || { echo "ALERT the log could not be written (tee ${codes[1]})"; return 1; }', 'alert() { step echo "ALERT $*" || true; }', 'python $T upload-log --file "$LOG" --key "$D/logs/$TS.log" || echo "ALERT log upload failed"']
    for word in ("c20", "C20", "val1000", "eval1000", "round15_compose_eval", "composed", "--export", "tok_v1_rule"):
        assert word not in "\n".join(code), word
    assert "D=$O/round20/path-read" in text and "O=acquisition/2026-10-17-v1/youtube-asl-v1" in text and "--smoke-train 5 --smoke-longest 4 --smoke-never-tracked 2" in text


def test_the_job_launches_nothing_and_no_new_file_touches_a5s():
    text = (ROOT / "scripts/path_read.py").read_text() + JOB.read_text()
    for word in ("create-pod", "runpodctl", "comp_a5", "COMP_A_DECLARED", "git "):
        assert word not in text
