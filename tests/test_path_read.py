"""The path read on synthetic data: what the lookups refuse, how a monocular clip is conformed, the control that stops the job, the shared-clip reads, and the job itself. No test starts a real job: the runner tests use fake executables and no credentials, and the store is a directory or a stand-in."""
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
import tok_v1 as T  # noqa: E402
from signtranslator.data import layout_adapter  # noqa: E402
from signtranslator.data.keypoints2d import from_state  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402
from signtranslator.eval import provenance, s0_metrics as M  # noqa: E402
from tests.test_generation_signing_space import signer_state  # noqa: E402
from tests.test_tok_probes import SpyStore, gappy, tiny_tokenizer  # noqa: E402
from tests.test_tok_v0 import gapped  # noqa: E402

N = 9
JOB = ROOT / "runpod/fleet/path_read.sh"
SHARD = f"{R.REPORTS}train_rgb_front_clips-shard-000-of-008.json"
WORDS = [f"w{i}" for i in range(15)]
MONO_FRAME, MONO_CONVENTION = "mono-frame", "mono-convention"
GROUPS = ("c01", "c03", "c05")


def d20_records(n=N):
    return [dict(sign=f"s{i}", corpus=L.H2S, partition="val", signer_component=GROUPS[i % 3], text=" ".join(f"{w}x{i}" for w in WORDS), continuous=dict(seconds=2.)) for i in range(n)]


def train_records(n=R.SMOKE_CLIPS):
    return [dict(r, partition="train", signer_component="c02") for r in d20_records(n)]


def monocular_state(i, frames=22, frame=MONO_FRAME, convention=MONO_CONVENTION):
    """A clip of the same sign as the monocular path holds it: other values, its own gaps, a longer clock, its own layout, and placeholder channels."""
    state = gappy(signer_state(100 + i, frames=frames, amplitude=.4), 3, 6)
    channels = {}
    for name, c in state.channels.items():
        channels[name] = replace(c, coordinate_frame=frame, convention=convention) if c.valid.any() else replace(c, labels=(f"{name}-unavailable",), coordinate_frame="not-estimated")
    return replace(state, channels=channels)


def plant_state(store, tmp_path, sign, state):
    path = tmp_path / f"plant-{sign}-{len(list(tmp_path.glob('plant-*')))}.npz"
    digest = state.save(path)
    store.put(R.state_key_of(sign), path.read_bytes())
    return digest


def plant_shard(monkeypatch, store, rows, key=SHARD):
    raw = json.dumps(dict(results=rows)).encode()
    store.put(key, raw)
    monkeypatch.setattr(R, "MONOCULAR_REPORTS", {key: hashlib.sha256(raw).hexdigest()})


class PlantedReader:
    """Reads each path at its planted level, so every share has a known answer: two-view says the whole sentence, monocular the first 8 words, a round trip the first 4, blank nothing."""
    info = dict(run="planted", checkpoint_key="k", sha256="f" * 64, kind="fake")
    LEVELS = dict(two_view=(15, 1.), monocular=(8, 2.), round_trip=(4, 2.5), blank=(0, 3.))

    def __init__(self, levels=None) -> None:
        self.levels = levels or self.LEVELS

    def level(self, state) -> str:
        hand = state.channels["left_hand"].valid
        if not hand.any():
            return "blank"
        return "round_trip" if hand.all() else ("two_view" if len(state.timestamps) == 20 else "monocular")

    def read(self, states, texts, candidates, group=2):
        rows = [self.levels[self.level(s)] for s in states]
        return dict(hypotheses=[" ".join(t.split()[:k]) if k else "z y x w" for t, (k, _) in zip(texts, rows)], nll=np.array([[v] for _, v in rows]))


class Harness:
    def __init__(self, monkeypatch, tmp_path, *, declared=True, n=N, reader=None, missing=(), no_state=(), unreadable=()):
        self.monkeypatch, self.tmp_path, self.declared, self.n = monkeypatch, tmp_path, declared, n
        self.records = d20_records(n) if declared else train_records(n)
        self.store = DirectoryStore(tmp_path / "s")
        self.forms = tmp_path / "w"
        self.refs = [gapped(i) for i in range(n)]
        self.tokenizer = tiny_tokenizer()
        rows = []
        for i, r in enumerate(self.records):
            digest = plant_state(self.store, tmp_path, r["sign"], monocular_state(i))
            if i in unreadable:
                digest = "0" * 64
            if i not in missing:
                rows.append(dict(sample_id=r["sign"], state_key=None if i in no_state else R.state_key_of(r["sign"]), state_sha256=digest, status="converted"))
        plant_shard(monkeypatch, self.store, rows)
        monkeypatch.setitem(layout_adapter.EQUIVALENT, ("f", "c"), {(MONO_FRAME, MONO_CONVENTION)})
        monkeypatch.setattr(L, "open_read_list", lambda store, smoke: (self.records, declared and not smoke, dict(list="t", sha256=L.D20_SHA, clips=n)))
        monkeypatch.setattr(L, "load_states", lambda store, recs: self.refs)
        monkeypatch.setattr(L, "load_tokenizer", lambda store: self.tokenizer)
        monkeypatch.setattr(L, "make_reader", lambda store, **k: reader or PlantedReader())
        monkeypatch.setattr(R, "load_template", lambda store: signer_state(99, frames=4, amplitude=.4))
        self.plant_stored()

    def main(self, *argv):
        return R.main([argv[0], "--store", f"dir:{self.tmp_path / 's'}", *argv[1:]])

    def read(self, *forms, smoke=False, resume=False):
        args = ["read", "--out-dir", str(self.forms)]
        if forms:
            args += ["--forms", ",".join(forms)]
        if smoke:
            args += ["--smoke-train", str(self.n)]
        if resume:
            args.append("--resume")
        return self.main(*args)

    def plant_stored(self, **change):
        """V1's stored rt_nearest on this list, every clip reading its first 5 words."""
        records = d20_records(self.n)
        texts = [r["text"] for r in records]
        hypotheses = [" ".join(t.split()[:5]) for t in texts]
        zeros = [0.] * self.n
        form = dict(kind="tok-v1-form", form="rt_nearest", report_only=True, declared_read=True, code_digest=f"{R.STORED_RT_DIGEST}aaaa", list=dict(sha256=L.D20_SHA), instrument=dict(sha256=L.TRANSLATOR_SHA), tokenizer=dict(sha256=L.TOKENIZER_SHA), environment=dict(gpu="stored"),
                    signs=[r["sign"] for r in records], references=texts, hypotheses=hypotheses, nll=[[2.75] for _ in texts], motion=dict(wrist_speed=zeros, wrist_spread=zeros), hands={h: dict(speed=zeros, variation=zeros) for h in P.HANDS}, tracked=dict(wrists=zeros, hands=zeros))
        form.update(change)
        self.store.put(R.STORED_RT, json.dumps(form).encode())
        bleu, ci = M.bleu_ci_fast(hypotheses, texts)
        self.monkeypatch.setattr(R, "STORED_RT_BLEU", bleu)
        self.monkeypatch.setattr(R, "STORED_RT_CI", ci)
        self.monkeypatch.setattr(L, "D20_CLIPS", self.n)
        return form

    def plant_control(self, *, hyp_change=0, nll_shift=0., bleu_shift=0., order=False):
        """The pinned stored report, holding what real_two_view and blank read, changed as asked."""
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

    def control(self):
        return self.main("control", "--out-dir", str(self.forms), "--void-prefix", "pr/void")

    def merge(self, name="report.json"):
        out = self.tmp_path / name
        code = self.main("merge", "--out-dir", str(self.forms), "--output-file", str(out))
        assert code == 0
        return json.loads(out.read_text())

    def pipeline(self):
        assert self.read(*R.ANCHORS) == 0
        self.plant_control()
        assert self.control() == 0
        assert self.read(*R.MONOCULAR_FORMS, resume=True) == 0
        return self.merge()


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
    found, lost = R.match_monocular(d20_records(2)[1:], rows, True)
    assert found == [] and lost[0]["reason"] == "state_key_not_this_clip" and lost[0]["detail"] == f"{R.STATES}s1-other.npz"


@pytest.mark.parametrize("key", [f"{R.MONOCULAR}/test-states/s0.npz", "acquisition/2026-10-17-v1/derived/full-v1/how2sign-two-view-v1/states/s0.npz", f"{R.STATES}s0.npz.bak", f"{R.STATES}s00.npz", f"x{R.STATES}s0.npz"])
def test_a_row_naming_any_other_object_never_reaches_the_store(key):
    found, lost = R.match_monocular(d20_records(1), dict(s0=dict(state_key=key, state_sha256="a" * 64)), True)
    assert not found and lost[0]["reason"] == "state_key_not_this_clip"
    store = SpyStore({key: b"never"})
    with pytest.raises(SystemExit, match="refused"):
        R.load_clip(store, dict(index=0, sign="s0", state_key=key, state_sha256="a" * 64), signer_state(99, frames=4), gapped(0))
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
    for word in ("c20", "C20", "val1000", "eval1000", "composed", "compose", "word_cache", "a9p", "a3", "test_rgb", "val_rgb", ".list(", "uwb-ft", "state-t5", "v2"):
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
    with pytest.raises(SystemExit, match="write-once"):
        h.main("lookups", "--out-dir", str(h.forms))


# ---- a monocular clip -------------------------------------------------------------------------------------------------------------

def test_a_clip_is_conformed_to_the_two_view_layout_with_its_values_and_marks_kept(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=1)
    raw = monocular_state(0)
    item = dict(index=0, sign="s0", state_key=R.state_key_of("s0"), state_sha256=plant_state(h.store, tmp_path, "s0", raw))
    clip, why = R.load_clip(h.store, item, signer_state(99, frames=4, amplitude=.4), h.refs[0])
    assert why is None and clip.index == 0 and clip.sign == "s0"
    assert R.layout_of(clip.conformed) == R.layout_of(h.refs[0]) and R.layout_of(clip.raw) != R.layout_of(h.refs[0]) and set(R.layout_of(clip.raw)) == {"body", "left_hand", "right_hand"}
    for name in ("body", "left_hand", "right_hand"):
        np.testing.assert_array_equal(clip.conformed.channels[name].values, raw.channels[name].values)
        np.testing.assert_array_equal(clip.conformed.channels[name].valid, raw.channels[name].valid)
    assert len(clip.conformed.timestamps) == 22 and len(clip.raw.timestamps) == 22


def test_a_clip_at_another_rate_comes_to_24_fps_conformed_and_raw(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=1)
    raw = replace(monocular_state(0, frames=30), timestamps=np.arange(30) / 30.)
    item = dict(index=0, sign="s0", state_key=R.state_key_of("s0"), state_sha256=plant_state(h.store, tmp_path, "s0", raw))
    clip, why = R.load_clip(h.store, item, signer_state(99, frames=4, amplitude=.4), h.refs[0])
    assert why is None and len(clip.conformed.timestamps) == len(clip.raw.timestamps) == 24 and abs(np.median(np.diff(clip.conformed.timestamps)) - 1 / 24) < 1e-6


def test_a_clip_whose_layout_has_no_declared_equivalence_is_lost_with_its_reason(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=2)
    plain = plant_state(h.store, tmp_path, "s1", monocular_state(1, frame="undeclared", convention="undeclared"))
    item = dict(index=1, sign="s1", state_key=R.state_key_of("s1"), state_sha256=plain)
    clip, why = R.load_clip(h.store, item, signer_state(99, frames=4, amplitude=.4), h.refs[1])
    assert clip is None and why["sign"] == "s1" and why["reason"] == "conform_refused" and "declared equivalence" in why["detail"]


@pytest.mark.parametrize("sha,reason", [("0" * 64, "state_unreadable")])
def test_a_clip_whose_state_fails_its_hash_or_is_missing_is_lost_with_its_reason(monkeypatch, tmp_path, sha, reason):
    h = Harness(monkeypatch, tmp_path, n=1)
    item = dict(index=0, sign="s0", state_key=R.state_key_of("s0"), state_sha256=sha)
    clip, why = R.load_clip(h.store, item, signer_state(99, frames=4, amplitude=.4), h.refs[0])
    assert clip is None and why["reason"] == reason and why["detail"]
    empty = DirectoryStore(tmp_path / "empty")
    clip, why = R.load_clip(empty, item, signer_state(99, frames=4, amplitude=.4), h.refs[0])
    assert clip is None and why["reason"] == reason and "FileNotFoundError" in why["detail"]


def test_a_clip_missing_a_channel_the_template_has_is_lost_with_its_reason(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=1)
    item = dict(index=0, sign="s0", state_key=R.state_key_of("s0"), state_sha256=plant_state(h.store, tmp_path, "s0", monocular_state(0)))
    template = signer_state(99, frames=4, amplitude=.4)

    def missing(state, template):
        raise KeyError("left_hand")
    monkeypatch.setattr(layout_adapter, "conform", missing)
    clip, why = R.load_clip(h.store, item, template, h.refs[0])
    assert clip is None and why["reason"] == "conform_refused" and "KeyError" in why["detail"]


def test_a_clip_with_one_frame_is_lost(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=1)
    item = dict(index=0, sign="s0", state_key=R.state_key_of("s0"), state_sha256=plant_state(h.store, tmp_path, "s0", monocular_state(0, frames=1)))
    clip, why = R.load_clip(h.store, item, signer_state(99, frames=4, amplitude=.4), h.refs[0])
    assert clip is None and why["reason"] == "too_few_frames"
    item = dict(item, state_sha256=plant_state(h.store, tmp_path, "s0", monocular_state(0, frames=2)))
    clip, why = R.load_clip(h.store, item, signer_state(99, frames=4, amplitude=.4), h.refs[0])
    assert clip is not None and why is None


def test_a_conformed_clip_whose_layout_is_not_the_two_view_clips_is_lost(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=1)
    item = dict(index=0, sign="s0", state_key=R.state_key_of("s0"), state_sha256=plant_state(h.store, tmp_path, "s0", monocular_state(0)))
    other = h.refs[0]
    other = replace(other, channels={**other.channels, "body": replace(other.channels["body"], convention="other")})
    clip, why = R.load_clip(h.store, item, signer_state(99, frames=4, amplitude=.4), other)
    assert clip is None and why["reason"] == "layout_differs_from_two_view"


def test_an_unavailable_channel_named_differently_in_the_two_view_clip_loses_nothing(monkeypatch, tmp_path):
    """v1 never reads it, and a two-view clip can hold a placeholder where the template holds data."""
    h = Harness(monkeypatch, tmp_path, n=1)
    item = dict(index=0, sign="s0", state_key=R.state_key_of("s0"), state_sha256=plant_state(h.store, tmp_path, "s0", monocular_state(0)))
    ref = h.refs[0]
    name = next(n for n, c in ref.channels.items() if not c.valid.any())
    other = replace(ref, channels={**ref.channels, name: replace(ref.channels[name], labels=("placeholder",), coordinate_frame="not-estimated")})
    clip, why = R.load_clip(h.store, item, signer_state(99, frames=4, amplitude=.4), other)
    assert why is None and clip is not None


def test_a_clip_is_checked_against_its_own_two_view_clip(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    ref = h.refs[1]
    h.refs[1] = replace(ref, channels={**ref.channels, "body": replace(ref.channels["body"], convention="other")})
    clips, lost, _ = R.open_monocular(h.store, h.records, h.refs, True)
    assert [c.sign for c in clips] == ["s0", "s2"] and [(x["sign"], x["reason"]) for x in lost] == [("s1", "layout_differs_from_two_view")]


def test_no_usable_clip_refuses(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=range(N))
    with pytest.raises(SystemExit, match="no clip has a usable monocular state"):
        R.open_monocular(h.store, h.records, h.refs, True)


def test_open_monocular_keeps_list_order_and_reasons(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=(1,), unreadable=(4,))
    clips, lost, template = R.open_monocular(h.store, h.records, h.refs, True)
    assert [c.index for c in clips] == [0, 2, 3, 5, 6, 7, 8] and {x["sign"]: x["reason"] for x in lost} == dict(s1="no_monocular_row", s4="state_unreadable") and len(template.timestamps) == 4


def test_v1s_input_is_byte_identical_under_conform_so_a_read_of_it_would_repeat():
    raw = monocular_state(0)
    with pytest.MonkeyPatch.context() as m:
        m.setitem(layout_adapter.EQUIVALENT, ("f", "c"), {(MONO_FRAME, MONO_CONVENTION)})
        done = layout_adapter.conform(raw, signer_state(99, frames=4, amplitude=.4))
    np.testing.assert_array_equal(R.reader_input(raw), R.reader_input(done))
    clips = [R.Clip(0, "s0", raw, done)]
    plan = R.unconformed_plan(clips)
    assert plan["read"] is False and plan["clips_compared"] == 1 and plan["clips_differing"] == 0 and "byte-identical" in plan["reason"] and plan["differing"] == []


def test_a_conform_that_changes_what_v1_receives_means_the_form_is_read():
    raw = monocular_state(0)
    plan = R.unconformed_plan([R.Clip(0, "same", raw, raw), R.Clip(1, "moved", raw, gappy(raw, 8, 12))])
    assert plan["read"] is True and plan["clips_differing"] == 1 and plan["differing"] == ["moved"] and "is read" in plan["reason"] and plan["clips_compared"] == 2


def test_reader_input_is_what_translate_builds():
    from signtranslator.data.keypoints2d import features, normalise
    state = gapped(1)
    points, valid = from_state(state)
    np.testing.assert_array_equal(R.reader_input(state), features(normalise(points, valid, y_down=False), valid))


def test_valid_frame_share_is_v1s_points_marked():
    assert R.valid_frame_share(signer_state(0, frames=8, amplitude=.4)) == 1.
    state = gappy(signer_state(0, frames=20, amplitude=.4), 5, 10)
    assert R.valid_frame_share(state) == round(float(from_state(state)[1].mean()), 5) and 0 < R.valid_frame_share(state) < 1


def test_the_conform_tally_says_how_each_channel_was_treated(monkeypatch):
    import tokenizer_domain_read as domain
    monkeypatch.setitem(layout_adapter.EQUIVALENT, ("f", "c"), {(MONO_FRAME, MONO_CONVENTION)})
    raw = monocular_state(0)
    template = signer_state(99, frames=4, amplitude=.4)
    tally = R.conform_tally([R.Clip(0, "s0", raw, raw), R.Clip(1, "s1", raw, raw)], template)
    assert tally["kinds"]["body:equivalent"] == 2 and tally["kinds"]["left_hand:equivalent"] == 2 and tally["dropped"] == {} and sum(v for k, v in tally["kinds"].items() if k.endswith(":absent")) > 0
    monkeypatch.setattr(domain, "conform_audit", lambda state, template: dict(kinds=dict(body="same"), extra=["stray"]))
    assert R.conform_tally([R.Clip(0, "s0", raw, raw), R.Clip(1, "s1", raw, raw)], template) == dict(kinds={"body:same": 2}, dropped=dict(stray=2))


# ---- forms ------------------------------------------------------------------------------------------------------------------------

def clips_of(n=3):
    with pytest.MonkeyPatch.context() as m:
        m.setitem(layout_adapter.EQUIVALENT, ("f", "c"), {(MONO_FRAME, MONO_CONVENTION)})
        template = signer_state(99, frames=4, amplitude=.4)
        return [R.Clip(i, f"s{i}", monocular_state(i), layout_adapter.conform(monocular_state(i), template)) for i in range(n)]


def test_each_form_is_built_from_the_right_states():
    from evaluate_text_to_motion import blanked
    refs, clips = [gapped(i) for i in range(4)], clips_of(3)
    kept, states, masks, lost = R.build_form("real_two_view", refs, clips, None)
    assert kept == [0, 1, 2, 3] and states[2] is refs[2] and masks[2] is refs[2] and lost == []
    kept, states, masks, lost = R.build_form("blank", refs, clips, None)
    assert kept == [0, 1, 2, 3] and not states[0].channels["body"].valid.any() and masks[1] is refs[1] and len(states[0].timestamps) == len(refs[0].timestamps)
    np.testing.assert_array_equal(states[1].channels["body"].values, blanked(refs)[1].channels["body"].values)
    kept, states, masks, lost = R.build_form("real_monocular", refs, clips, None)
    assert kept == [0, 1, 2] and states[1] is clips[1].conformed and masks[1] is clips[1].conformed and lost == []
    kept, states, masks, lost = R.build_form(R.UNCONFORMED, refs, clips, None)
    assert kept == [0, 1, 2] and states[1] is clips[1].raw and masks[1] is clips[1].raw and lost == []
    with pytest.raises(SystemExit, match="unknown form"):
        R.build_form("a_generated_form", refs, clips, None)


def test_round_trip_monocular_is_v1s_rt_nearest_of_the_conformed_clip(monkeypatch):
    tok, clips = tiny_tokenizer(), clips_of(2)
    seen = []
    original = T.cap_form
    monkeypatch.setattr(T, "cap_form", lambda name, state, tokenizer: seen.append((name, state, tokenizer)) or original(name, state, tokenizer))
    kept, states, masks, lost = R.build_form("round_trip_monocular", [gapped(i) for i in range(2)], clips, tok)
    assert kept == [0, 1] and lost == [] and masks[1] is clips[1].conformed
    assert [x[0] for x in seen] == ["rt_nearest", "rt_nearest"] and seen[1][1] is clips[1].conformed and seen[0][2] is tok
    np.testing.assert_array_equal(states[0].channels["left_hand"].values, original("rt_nearest", clips[0].conformed, tok).channels["left_hand"].values)
    assert all(states[0].channels[n].valid.all() for n in ("body", "left_hand", "right_hand"))


def test_a_clip_the_tokenizer_path_fails_on_is_lost_with_its_reason(monkeypatch):
    tok, clips = tiny_tokenizer(), clips_of(3)
    original = T.cap_form

    def fail_on_one(name, state, tokenizer):
        if state is clips[1].conformed:
            raise RuntimeError("planted")
        return original(name, state, tokenizer)
    monkeypatch.setattr(T, "cap_form", fail_on_one)
    kept, states, masks, lost = R.build_form("round_trip_monocular", [gapped(i) for i in range(3)], clips, tok)
    assert kept == [0, 2] and len(states) == 2 and masks == [clips[0].conformed, clips[2].conformed] and lost == [dict(sign="s1", reason="tokenizer_path_failed", detail="RuntimeError: planted")]
    monkeypatch.setattr(T, "cap_form", lambda *a: (_ for _ in ()).throw(ValueError("all")))
    with pytest.raises(SystemExit, match="no clip passed"):
        R.build_form("round_trip_monocular", [gapped(i) for i in range(3)], clips, tok)


def test_an_unknown_form_refuses(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=2)
    with pytest.raises(SystemExit, match="unknown forms"):
        h.read("a9p")


def test_a_read_writes_what_it_read(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, tmp_path, n=N)
    assert h.read(*R.ANCHORS) == 0
    real = json.loads((h.forms / "form-real_two_view.json").read_text())
    assert real["kind"] == "path-read-form" and real["report_only"] is True and real["form"] == "real_two_view" and real["declared_read"] is True and real["signs"] == [f"s{i}" for i in range(N)] and real["groups"][:3] == list(GROUPS)
    assert real["hypotheses"][0] == h.records[0]["text"] and real["references"] == [r["text"] for r in h.records] and real["nll"][0] == [1.] and real["code_digest"] == R.code_digest() and real["environment"] == P.environment() and real["instrument"]["sha256"] == "f" * 64 and real["tokenizer"] is None
    assert real["lost"] == [] and real["seconds_over_reference"] == [1.] * N and len(real["valid_frame_share"]) == N and real["valid_frame_share"][0] == R.valid_frame_share(h.refs[0]) and real["motion"]["wrist_speed"][0] is not None and real["tracked"]["wrists"][0] > 0 and real["hands"]["left_hand"]["speed"][0] is not None
    assert not (h.forms / "monocular.json").exists()
    assert h.read("round_trip_monocular", "real_monocular") == 0
    mono = json.loads((h.forms / "form-real_monocular.json").read_text())
    trip = json.loads((h.forms / "form-round_trip_monocular.json").read_text())
    assert mono["seconds_over_reference"] == [1.1] * N and mono["hypotheses"][0].split() == h.records[0]["text"].split()[:8] and mono["tokenizer"] is None and trip["tokenizer"] == dict(key=L.TOKENIZER, sha256=L.TOKENIZER_SHA, gains=None)
    assert all(v == 1. for v in trip["valid_frame_share"]) and all(v < 1. for v in mono["valid_frame_share"])
    assert any('"monocular_clips": 9' in line for line in capsys.readouterr().out.splitlines())


def test_a_read_passes_the_group_and_one_candidate_each_and_reads_its_own_clips_texts(monkeypatch, tmp_path):
    seen = []

    class Spy(PlantedReader):
        def read(self, states, texts, candidates, group=2):
            seen.append((len(states), list(texts), candidates.shape, group))
            return super().read(states, texts, candidates, group=group)
    h = Harness(monkeypatch, tmp_path, reader=Spy(), missing=(1,))
    assert h.main("read", "--out-dir", str(h.forms), "--forms", "real_two_view,real_monocular", "--nll-group", "4") == 0
    assert seen[0][0] == N and seen[0][2] == (N, 1) and seen[0][3] == 4
    assert seen[1][0] == N - 1 and seen[1][2] == (N - 1, 1) and seen[1][1] == [r["text"] for r in h.records if r["sign"] != "s1"]


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


def test_anchors_alone_need_no_monocular_lookup_or_tokenizer(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=2)
    monkeypatch.setattr(R, "open_monocular", lambda *a: pytest.fail("a monocular lookup"))
    monkeypatch.setattr(L, "load_tokenizer", lambda *a: pytest.fail("the tokenizer"))
    assert h.read(*R.ANCHORS) == 0


def test_the_unconformed_form_is_skipped_with_its_reason_when_v1s_input_is_identical(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, tmp_path)
    assert h.read(*R.MONOCULAR_FORMS) == 0
    assert not (h.forms / f"form-{R.UNCONFORMED}.json").exists() and (h.forms / "form-real_monocular.json").exists()
    plan = json.loads((h.forms / "monocular.json").read_text())
    assert plan["unconformed"]["read"] is False and "byte-identical" in plan["unconformed"]["reason"] and plan["clips_loaded"] == N and plan["code_digest"] == R.code_digest() and plan["declared_read"] is True and plan["conform"]["kinds"]["body:equivalent"] == N
    assert "byte-identical" in capsys.readouterr().out
    assert h.read(R.UNCONFORMED, resume=True) == 0 and not (h.forms / f"form-{R.UNCONFORMED}.json").exists()


def test_the_unconformed_form_is_read_when_conforming_changes_v1s_input(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    original = layout_adapter.conform

    monkeypatch.setattr(layout_adapter, "conform", lambda state, template: gappy(original(state, template), 8, 12))
    assert h.read(*R.MONOCULAR_FORMS) == 0
    plan = json.loads((h.forms / "monocular.json").read_text())
    assert plan["unconformed"]["read"] is True and json.loads((h.forms / f"form-{R.UNCONFORMED}.json").read_text())["signs"] == ["s0", "s1", "s2"]


def test_lost_clips_are_in_each_monocular_forms_report_and_no_anchors(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=(1,), unreadable=(4,))
    assert h.read("real_two_view", "real_monocular") == 0
    anchor, mono = (json.loads((h.forms / f"form-{n}.json").read_text()) for n in ("real_two_view", "real_monocular"))
    assert anchor["lost"] == [] and len(anchor["signs"]) == N
    assert {x["sign"]: x["reason"] for x in mono["lost"]} == dict(s1="no_monocular_row", s4="state_unreadable") and mono["signs"] == [f"s{i}" for i in (0, 2, 3, 5, 6, 7, 8)]
    assert mono["references"] == [h.records[i]["text"] for i in (0, 2, 3, 5, 6, 7, 8)] and mono["groups"] == [h.records[i]["signer_component"] for i in (0, 2, 3, 5, 6, 7, 8)]


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
    assert not h.store.list("pr/void")


@pytest.mark.parametrize("change,miss", [(dict(hyp_change=1), "real hypotheses"), (dict(nll_shift=.0012), "real NLL"), (dict(bleu_shift=.06), "real BLEU")])
def test_a_control_miss_exits_3_voids_the_read_and_a_merge_refuses_it(monkeypatch, tmp_path, change, miss):
    h = read_anchors(monkeypatch, tmp_path)
    h.plant_control(**change)
    assert h.control() == 3
    result = json.loads((h.forms / "control.json").read_text())
    assert not result["reproduced"] and any(miss in m for m in result["misses"])
    void = json.loads(h.store.get(f"pr/void/{R.code_digest()}.json"))
    assert void["kind"] == "path-read-void" and void["control"]["misses"] == result["misses"] and void["code_digest"] == R.code_digest()
    assert h.read(*R.MONOCULAR_FORMS, resume=True) == 0        # the job stops first; a merge still refuses this
    with pytest.raises(SystemExit, match="no passed control"):
        h.merge()


def test_a_control_at_its_limits_passes(monkeypatch, tmp_path):
    h = read_anchors(monkeypatch, tmp_path)
    h.plant_control(nll_shift=.0009, bleu_shift=.04)
    assert h.control() == 0


def test_a_control_of_other_clips_or_order_refuses(monkeypatch, tmp_path):
    h = read_anchors(monkeypatch, tmp_path)
    h.plant_control(order=True)
    with pytest.raises(SystemExit, match="other clips or another order"):
        h.control()


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


def test_the_control_refuses_real_and_blank_of_different_runs(monkeypatch, tmp_path):
    h = read_anchors(monkeypatch, tmp_path)
    h.plant_control()
    path = h.forms / "form-blank.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), code_digest="other")))
    with pytest.raises(SystemExit, match="different code"):
        h.control()


# ---- merge ------------------------------------------------------------------------------------------------------------------------

def test_every_read_runs_on_the_shared_clips_and_lost_clips_are_listed(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=(1,), no_state=(2,), unreadable=(4,))
    report = h.pipeline()
    shared = [f"s{i}" for i in (0, 3, 5, 6, 7, 8)]
    assert report["clips"]["list"] == N and report["clips"]["shared"] == 6
    lost = report["clips"]["outside_shared"]
    assert lost["count"] == 3 and lost["by_reason"] == dict(no_monocular_row=1, no_state_in_row=1, state_unreadable=1) and {c["sign"] for c in lost["clips"]} == {"s1", "s2", "s4"} and lost["clips"][0]["forms"]["real_monocular"]["reason"] == "no_monocular_row"
    forms = {n: json.loads((h.forms / f"form-{n}.json").read_text()) for n in ("real_two_view", "blank", "real_monocular", "round_trip_monocular")}
    assert len(forms["real_two_view"]["signs"]) == N and forms["real_monocular"]["signs"] == shared
    for name, form in forms.items():
        pick = [form["signs"].index(s) for s in shared]
        hypotheses, references = [form["hypotheses"][i] for i in pick], [form["references"][i] for i in pick]
        bleu, ci = M.bleu_ci_fast(hypotheses, references)
        assert report["forms"][name]["bleu"] == dict(bleu=bleu, ci95=ci) and report["forms"][name]["clips"] == 6
    two = [forms["real_two_view"]["hypotheses"][forms["real_two_view"]["signs"].index(s)] for s in shared]
    mono = [forms["real_monocular"]["hypotheses"][forms["real_monocular"]["signs"].index(s)] for s in shared]
    refs = [forms["real_monocular"]["references"][forms["real_monocular"]["signs"].index(s)] for s in shared]
    assert report["forms"]["real_monocular"]["paired"]["real_two_view"]["bleu"] == M.paired_bleu_fast(M.bleu_statistics(mono, refs), M.bleu_statistics(two, refs))
    assert report["reading"]["facts"]["difference"] == report["forms"]["real_monocular"]["paired"]["real_two_view"]["bleu"]
    assert report["forms"]["real_monocular"]["paired"]["real_two_view"]["likelihood_minus_blank"]["difference"] == pytest.approx(-1.)
    assert report["same_path"]["clips"] == 6 and report["stored_rt_nearest"]["shared"]["clips"] == 6


def test_a_clip_the_tokenizer_path_lost_leaves_every_comparison(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    original = T.cap_form
    calls = []

    def fail_on_second(name, state, tokenizer):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("planted")
        return original(name, state, tokenizer)
    monkeypatch.setattr(T, "cap_form", fail_on_second)
    report = h.pipeline()
    assert report["clips"]["shared"] == N - 1 and report["clips"]["outside_shared"]["by_reason"] == dict(tokenizer_path_failed=1) and report["forms"]["real_monocular"]["clips"] == N - 1 and report["clips"]["outside_shared"]["clips"][0]["sign"] == "s1"


def test_the_shares_and_same_path_figures_have_their_planted_values(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    report = h.pipeline()
    forms = report["forms"]
    assert forms["real_two_view"]["likelihood_minus_blank"]["mean"] == 2. and forms["real_monocular"]["likelihood_minus_blank"]["mean"] == 1. and forms["round_trip_monocular"]["likelihood_minus_blank"]["mean"] == .5 and forms["blank"]["likelihood_minus_blank"]["mean"] == 0.
    assert forms["real_monocular"]["share_of_gap"]["likelihood"] == .5 and forms["round_trip_monocular"]["share_of_gap"]["likelihood"] == .25 and forms["real_two_view"]["share_of_gap"]["likelihood"] == 1.
    assert forms["blank"]["share_of_gap"]["bleu"] == 0. and forms["real_two_view"]["share_of_gap"]["bleu"] == 1. and 0 < forms["round_trip_monocular"]["share_of_gap"]["bleu"] < forms["real_monocular"]["share_of_gap"]["bleu"] < 1
    likelihood = report["same_path"]["likelihood_minus_blank"]
    assert likelihood["real_two_view_gap"]["point"] == 2. and likelihood["real_monocular_gap"]["point"] == 1. and likelihood["monocular_gap_over_two_view_gap"]["point"] == .5 and likelihood["real_monocular_minus_real_two_view"]["point"] == -1.
    assert likelihood["round_trip_monocular_minus_real_monocular"]["point"] == -.5 and likelihood["round_trip_monocular_share_of_monocular_gap"]["point"] == .5 and likelihood["round_trip_monocular_share_of_two_view_gap"]["point"] == .25
    assert likelihood["stored_rt_nearest_share_of_two_view_gap"]["point"] == .125 and likelihood["round_trip_monocular_minus_stored_rt_nearest"]["point"] == .25
    assert likelihood["monocular_gap_over_two_view_gap"]["ci95"] == [.5, .5]
    bleu = report["same_path"]["bleu"]
    b = {n: forms[n]["bleu"]["bleu"] for n in forms}
    assert bleu["monocular_gap_over_two_view_gap"]["point"] == pytest.approx((b["real_monocular"] - b["blank"]) / (b["real_two_view"] - b["blank"]), abs=2e-3)
    assert bleu["round_trip_monocular_share_of_monocular_gap"]["point"] == pytest.approx((b["round_trip_monocular"] - b["blank"]) / (b["real_monocular"] - b["blank"]), abs=2e-3) and bleu["round_trip_monocular_share_of_monocular_gap"]["point"] != bleu["round_trip_monocular_share_of_two_view_gap"]["point"]
    assert bleu["real_monocular_gap"]["point"] == pytest.approx(b["real_monocular"] - b["blank"], abs=2e-3) and bleu["real_two_view_gap"]["point"] == pytest.approx(b["real_two_view"] - b["blank"], abs=2e-3)
    assert report["same_path"]["bootstrap"] == dict(n=2000, seed=0)


def test_likelihood_differences_stay_paired_clip_by_clip(monkeypatch, tmp_path):
    """Levels that vary by clip: a joint resample of a constant difference has no spread at all, though each form's own mean does."""
    class Varying(PlantedReader):
        def read(self, states, texts, candidates, group=2):
            out = super().read(states, texts, candidates, group=group)
            moving = np.array([[.25 * i if self.level(s) != "blank" else 0.] for i, s in enumerate(states)])
            out["nll"] = out["nll"] + moving
            return out
    h = Harness(monkeypatch, tmp_path, reader=Varying())
    report = h.pipeline()
    likelihood = report["same_path"]["likelihood_minus_blank"]
    assert likelihood["real_monocular_minus_real_two_view"]["ci95"] == [-1., -1.] and report["forms"]["real_monocular"]["likelihood_minus_blank"]["ci95"][0] < report["forms"]["real_monocular"]["likelihood_minus_blank"]["ci95"][1]
    assert report["forms"]["real_monocular"]["paired"]["real_two_view"]["likelihood_minus_blank"]["ci95"] == [-1., -1.]


def test_shares_are_taken_from_blank_when_blank_reads_above_zero(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, reader=PlantedReader(dict(PlantedReader.LEVELS, blank=(4, 3.), round_trip=(6, 2.5))))
    report = h.pipeline()
    forms = {n: json.loads((h.forms / f"form-{n}.json").read_text()) for n in ("real_two_view", "blank", "real_monocular", "round_trip_monocular")}
    refs = forms["real_two_view"]["references"]
    stats = {n: M.bleu_statistics(f["hypotheses"], refs) for n, f in forms.items()}
    point = {n: float(M.bleu_from_statistics(s.sum(0))) for n, s in stats.items()}
    assert point["blank"] > 0
    idx = np.random.default_rng(0).integers(0, len(refs), (R.BOOT, len(refs)))
    draws = {n: M.bleu_from_statistics(s[idx].sum(1)) for n, s in stats.items()}
    row = report["same_path"]["bleu"]["round_trip_monocular_share_of_monocular_gap"]
    assert row["point"] == round((point["round_trip_monocular"] - point["blank"]) / (point["real_monocular"] - point["blank"]), 4)
    assert row["ci95"] == P._quantiles((draws["round_trip_monocular"] - draws["blank"]) / (draws["real_monocular"] - draws["blank"]))


def test_bleu_differences_and_shares_resample_the_forms_jointly_over_clips():
    refs = [r["text"] for r in d20_records(30)]

    def form(keep):
        return dict(references=refs, hypotheses=[" ".join(t.split()[:keep(i)]) for i, t in enumerate(refs)], nll=[[3. - keep(i) / 10.] for i in range(30)])
    reads = dict(real_two_view=form(lambda i: 5 + i % 11), real_monocular=form(lambda i: 5 + i % 7), round_trip_monocular=form(lambda i: 4 + i % 5), blank=form(lambda i: 4))
    out = R.same_path(reads, form(lambda i: 4 + i % 3))
    idx = np.random.default_rng(0).integers(0, 30, (R.BOOT, 30))
    stats = {n: M.bleu_statistics(f["hypotheses"], refs) for n, f in reads.items()}
    draws = {n: M.bleu_from_statistics(x[idx].sum(1)) for n, x in stats.items()}
    assert out["bleu"]["real_monocular_minus_real_two_view"]["ci95"] == P._quantiles(draws["real_monocular"] - draws["real_two_view"])
    assert out["bleu"]["monocular_gap_over_two_view_gap"]["ci95"] == P._quantiles((draws["real_monocular"] - draws["blank"]) / (draws["real_two_view"] - draws["blank"]))
    assert out["clips"] == 30 and out["bleu"]["stored_rt_nearest_share_of_two_view_gap"] is not None


def test_each_forms_motion_is_measured_on_its_own_paths_tracked_frames(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    assert h.read("round_trip_monocular") == 0
    form = json.loads((h.forms / "form-round_trip_monocular.json").read_text())
    template = signer_state(99, frames=4, amplitude=.4)
    conformed = [layout_adapter.conform(monocular_state(i), template) for i in range(3)]
    states = [T.cap_form("rt_nearest", c, h.tokenizer) for c in conformed]
    assert form["motion"] == json.loads(json.dumps(L.motion_statistics(states, conformed))) and form["motion"] != json.loads(json.dumps(L.motion_statistics(states, states)))
    assert form["hands"] == json.loads(json.dumps(P.hand_statistics(states, conformed))) and form["hands"] != json.loads(json.dumps(P.hand_statistics(states, states)))


def test_a_resume_after_the_unconformed_form_was_settled_opens_nothing(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    assert h.read(*R.MONOCULAR_FORMS) == 0
    monkeypatch.setattr(R, "open_monocular", lambda *a: pytest.fail("the monocular clips were opened again"))
    assert h.read(*R.MONOCULAR_FORMS, resume=True) == 0


def test_a_merge_refuses_a_taken_output_before_it_loads_anything(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    taken = tmp_path / "taken.json"
    taken.write_text("{}")
    monkeypatch.setattr(R, "load_reads", lambda *a: pytest.fail("the forms were loaded"))
    with pytest.raises(SystemExit, match="write-once"):
        h.main("merge", "--out-dir", str(h.forms), "--output-file", str(taken))


def test_the_reading_reports_whether_the_intervals_overlap_and_labels_only_a_declared_read():
    def panels(two, mono, difference):
        return dict(real_two_view=dict(bleu=dict(bleu=4., ci95=two)), real_monocular=dict(bleu=dict(bleu=4., ci95=mono), paired=dict(real_two_view=dict(bleu=dict(difference=difference, ci95=[difference - .1, difference + .1]), likelihood_minus_blank=dict(difference=0., ci95=[0., 0.])))))
    apart = R.reading_block(panels([4.4, 4.5], [4., 4.1], -.4), True)
    assert apart["facts"]["intervals_overlap"] is False and apart["case"] is None and apart["facts"]["monocular_higher"] is False
    near = R.reading_block(panels([4., 5.], [4.1, 5.1], .1), True)
    assert near["facts"]["intervals_overlap"] is True and near["case"] == "a" and near["facts"]["monocular_higher"] is True
    assert R.reading_block(panels([4., 5.], [4.1, 5.1], .1), False)["case"] is None
    assert R.overlaps([1., 2.], [2., 3.]) and R.overlaps([2., 3.], [1., 2.]) and not R.overlaps([1., 2.], [2.1, 3.]) and not R.overlaps([2.1, 3.], [1., 2.])


def test_per_signer_rows_partition_the_shared_clips(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=(0,))
    report = h.pipeline()
    rows = report["per_signer"]
    assert sorted(rows) == list(GROUPS) and [rows[g]["real_monocular"]["clips"] for g in GROUPS] == [2, 3, 3] and sum(rows[g]["real_two_view"]["clips"] for g in GROUPS) == N - 1
    form = json.loads((h.forms / "form-real_monocular.json").read_text())
    index = [i for i, g in enumerate(form["groups"]) if g == "c03"]
    assert rows["c03"]["real_monocular"]["bleu"] == dict(zip(("bleu", "ci95"), M.bleu_ci_fast([form["hypotheses"][i] for i in index], [form["references"][i] for i in index])))
    row = rows["c03"]["real_monocular"]
    assert row["likelihood_minus_blank"]["mean"] == 1. and row["share_of_gap"]["likelihood"] == .5 and row["paired"]["real_two_view"]["likelihood_minus_blank"]["difference"] == -1. and 0 < row["valid_frame_share"]["mean"] < 1
    assert set(row) == set(R.SIGNER_ROW) and "tracked_share" in row and row["seconds_over_reference"] == 1.1


def test_valid_frame_shares_are_read_per_form_on_the_shared_clips(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    forms = h.pipeline()["forms"]
    assert forms["blank"]["valid_frame_share"]["mean"] == 0. and forms["round_trip_monocular"]["valid_frame_share"]["mean"] == 1. and forms["real_monocular"]["valid_frame_share"]["mean"] == pytest.approx(np.mean([R.valid_frame_share(monocular_state(i)) for i in range(N)]), abs=1e-4)
    assert forms["real_two_view"]["valid_frame_share"]["mean"] == pytest.approx(np.mean([R.valid_frame_share(gapped(i)) for i in range(N)]), abs=1e-4)
    assert forms["real_monocular"]["seconds_over_reference"] == 1.1 and forms["real_two_view"]["seconds_over_reference"] == 1. and forms["real_monocular"]["identical_to_real_two_view"] == 0. and forms["real_two_view"]["identical_to_real_two_view"] == 1.
    assert "real_two_view" in forms["real_monocular"]["paired"] and "real" not in forms["real_monocular"]["paired"] and "blank" in forms["real_monocular"]["paired"]


def test_the_stored_rt_nearest_is_compared_on_the_shared_clips_beside_its_500(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, missing=(1,))
    report = h.pipeline()
    row = report["stored_rt_nearest"]
    stored = json.loads(h.store.get(R.STORED_RT))
    shared = [i for i in range(N) if i != 1]
    assert row["all_500"]["clips"] == N and row["all_500"]["bleu"] == R.STORED_RT_BLEU and row["sha256"] == hashlib.sha256(h.store.get(R.STORED_RT)).hexdigest() and row["key"] == R.STORED_RT
    bleu, ci = M.bleu_ci_fast([stored["hypotheses"][i] for i in shared], [stored["references"][i] for i in shared])
    assert row["shared"]["bleu"] == bleu and row["shared"]["ci95"] == ci and row["shared"]["clips"] == N - 1 and row["shared"]["likelihood_minus_blank"]["mean"] == .25
    assert row["shared"]["round_trip_monocular_minus_stored"]["likelihood_minus_blank"]["difference"] == .25
    trip = json.loads((h.forms / "form-round_trip_monocular.json").read_text())
    assert row["shared"]["round_trip_monocular_minus_stored"]["bleu"] == M.paired_bleu_fast(M.bleu_statistics(trip["hypotheses"], trip["references"]), M.bleu_statistics([stored["hypotheses"][i] for i in shared], [stored["references"][i] for i in shared]))
    assert row["environment"] == dict(gpu="stored") and row["same_environment"] is False and row["code_digest"].startswith(R.STORED_RT_DIGEST)


@pytest.mark.parametrize("change,word", [(dict(kind="other"), "kind"), (dict(form="a9p"), "form"), (dict(declared_read=False), "declared_read"), (dict(report_only=False), "report_only"), (dict(code_digest="0000"), "code_digest"), (dict(list=dict(sha256="0" * 64)), "list"), (dict(signs=["s0"]), "clips"), (dict(instrument=dict(sha256="0" * 64)), "instrument"), (dict(tokenizer=dict(sha256="0" * 64)), "tokenizer")])
def test_a_stored_form_that_is_not_v1s_refuses(monkeypatch, tmp_path, change, word):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.plant_stored(**change)
    with pytest.raises(SystemExit, match=f"not V1's stored read: .*{word}"):
        R.load_stored_rt_nearest(h.store)


def test_a_stored_form_reading_another_figure_refuses(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    h.plant_stored()
    monkeypatch.setattr(R, "STORED_RT_BLEU", 4.124)
    with pytest.raises(SystemExit, match="not the plan's"):
        R.load_stored_rt_nearest(h.store)
    h.plant_stored()
    monkeypatch.setattr(R, "STORED_RT_CI", [0., 0.])
    with pytest.raises(SystemExit, match="not the plan's"):
        R.load_stored_rt_nearest(h.store)


def test_the_pinned_stored_figures_are_the_plans():
    assert (R.STORED_RT_BLEU, R.STORED_RT_CI, R.STORED_RT_DIGEST) == (4.124, [3.485, 4.764], "abcb46bd") and R.STORED_RT == "acquisition/2026-10-17-v1/youtube-asl-v1/tok-probes-v1/forms/form-rt_nearest.json"


def test_a_stored_form_with_another_reference_refuses(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=N)
    form = h.plant_stored()
    assert h.read(*R.ANCHORS) == 0
    h.plant_control()
    assert h.control() == 0
    assert h.read(*R.MONOCULAR_FORMS, resume=True) == 0
    references = list(form["references"])
    references[0] = "another sentence"
    h.plant_stored(references=references, hypotheses=form["hypotheses"])
    monkeypatch.setattr(R, "STORED_RT_BLEU", M.bleu_ci_fast(form["hypotheses"], references)[0])
    monkeypatch.setattr(R, "STORED_RT_CI", M.bleu_ci_fast(form["hypotheses"], references)[1])
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
    read = dict(signs=["a", "b", "c"], groups=["1", "2", "3"], references=["ra", "rb", "rc"], hypotheses=["ha", "hb", "hc"], nll=[[1.], [2.], [3.]], seconds_over_reference=[.1, .2, .3], valid_frame_share=[.4, .5, .6], form="f",
                motion=dict(wrist_speed=[1, 2, 3], wrist_spread=[4, 5, 6]), hands=dict(left_hand=dict(speed=[7, 8, 9], variation=[1, 1, 2])), tracked=dict(wrists=[0, 1, 2], hands=[3, 4, 5]))
    out = R.restrict(read, ["c", "a"])
    assert out["signs"] == ["c", "a"] and out["groups"] == ["3", "1"] and out["references"] == ["rc", "ra"] and out["hypotheses"] == ["hc", "ha"] and out["nll"] == [[3.], [1.]] and out["seconds_over_reference"] == [.3, .1] and out["valid_frame_share"] == [.6, .4]
    assert out["motion"] == dict(wrist_speed=[3, 1], wrist_spread=[6, 4]) and out["hands"]["left_hand"] == dict(speed=[9, 7], variation=[2, 1]) and out["tracked"] == dict(wrists=[2, 0], hands=[5, 3]) and out["form"] == "f" and read["signs"] == ["a", "b", "c"]
    del read["groups"]
    assert "groups" not in R.restrict(read, ["a"])


def test_the_report_names_its_list_code_environment_and_reading(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, tmp_path, missing=(1,))
    assert h.main("lookups", "--out-dir", str(h.forms)) == 0
    report = h.pipeline()
    assert report["kind"] == "path-read-report" and report["report_only"] is True and report["note"] == R.NOT_DECIDED and report["declared_read"] is True and report["smoke"] is None
    assert report["code_digest"] == R.code_digest() and report["environment"] == P.environment() and report["instrument"]["sha256"] == "f" * 64 and report["list"]["list"] == "t" and report["tokenizer"]["sha256"] == L.TOKENIZER_SHA
    assert report["disclosures"] == json.loads(json.dumps(R.DISCLOSURES)) and report["monocular"]["unconformed"]["read"] is False and report["monocular"]["clips_loaded"] == N - 1 and report["monocular"]["conform"]["kinds"]["body:equivalent"] == N - 1
    assert report["lookups"]["matched"] == N - 1 and report["checks"]["control"]["reproduced"] is True and report["checks"]["blank_identical_to_real_two_view"] == 0. and report["rerun"] is None
    assert report["reading"]["prestated"] == R.PRESTATED and report["reading"]["case"] == "b" and set(report["resources"]) == {"real_two_view", "blank", "real_monocular", "round_trip_monocular"} and report["resources"]["blank"]["read_seconds"] >= 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["case"] == "b" and printed["shared"] == N - 1 and printed["of"] == N and printed["lost"] == dict(no_monocular_row=1) and printed["forms"]["real_monocular"]["gap_share"] == report["forms"]["real_monocular"]["share_of_gap"]["bleu"] and printed["forms"]["real_monocular"]["valid_frame_share"] == report["forms"]["real_monocular"]["valid_frame_share"]["mean"]


def test_a_read_unconformed_form_gets_its_own_panel(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    original = layout_adapter.conform

    monkeypatch.setattr(layout_adapter, "conform", lambda state, template: gappy(original(state, template), 8, 12))
    report = h.pipeline()
    assert report["monocular"]["unconformed"]["read"] is True and R.UNCONFORMED in report["forms"] and report["forms"][R.UNCONFORMED]["paired"]["real_two_view"]["bleu"]["difference"] < 0 and R.UNCONFORMED in report["per_signer"]["c01"]


def test_a_merge_refuses_what_is_not_one_run(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    assert h.read(*R.ANCHORS) == 0
    h.plant_control()
    assert h.control() == 0
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


def test_a_merge_refuses_a_control_of_other_code_or_environment_or_none(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    assert h.read(*R.ANCHORS) == 0
    h.plant_control()
    assert h.control() == 0
    assert h.read(*R.MONOCULAR_FORMS, resume=True) == 0
    path = h.forms / "control.json"
    good = json.loads(path.read_text())
    for change in (dict(reproduced=False), dict(code_digest="other"), dict(environment=dict(gpu="other"))):
        path.write_text(json.dumps(dict(good, **change)))
        with pytest.raises(SystemExit, match="no passed control"):
            h.merge()
    path.unlink()
    with pytest.raises(SystemExit, match="no passed control"):
        h.merge()


def test_a_merge_refuses_a_monocular_record_of_another_run_and_a_form_it_says_was_not_read(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    assert h.read(*R.ANCHORS) == 0
    h.plant_control()
    assert h.control() == 0
    assert h.read(*R.MONOCULAR_FORMS, resume=True) == 0
    path = h.forms / "monocular.json"
    good = json.loads(path.read_text())
    for change in (dict(code_digest="other"), dict(declared_read=False)):
        path.write_text(json.dumps(dict(good, **change)))
        with pytest.raises(SystemExit, match="another run's record"):
            h.merge()
    path.write_text(json.dumps(good))
    stray = h.forms / f"form-{R.UNCONFORMED}.json"
    stray.write_text((h.forms / "form-real_monocular.json").read_text())
    with pytest.raises(SystemExit, match="record says it was not"):
        h.merge()
    stray.unlink()
    path.write_text(json.dumps(dict(good, unconformed=dict(good["unconformed"], read=True))))
    with pytest.raises(SystemExit, match=f"form-{R.UNCONFORMED}.json is missing"):
        h.merge()


def test_a_merge_with_no_shared_clip_refuses(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, n=3)
    assert h.read(*R.ANCHORS) == 0
    h.plant_control()
    assert h.control() == 0
    assert h.read(*R.MONOCULAR_FORMS, resume=True) == 0
    for name in ("real_monocular", "round_trip_monocular"):
        path = h.forms / f"form-{name}.json"
        form = json.loads(path.read_text())
        pick = [0] if name == "real_monocular" else [1]
        form.update(signs=[form["signs"][i] for i in pick], references=[form["references"][i] for i in pick], lost=[])
        path.write_text(json.dumps(form))
    with pytest.raises(SystemExit, match="share no clip"):
        h.merge()


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


def test_a_smoke_runs_every_form_on_train_clips_and_says_so(monkeypatch, tmp_path, capsys):
    h = Harness(monkeypatch, tmp_path, declared=False, n=5, missing=(4,))
    assert h.read(smoke=True) == 0
    h.plant_control()
    assert h.control() == 0
    report = h.merge()
    assert report["declared_read"] is False and report["smoke"].startswith("SMOKE on c02 train clips") and report["clips"]["shared"] == 4 and report["checks"]["control"] is None
    assert report["reading"]["case"] is None and "smoke" in report["reading"]["case_note"] and report["reading"]["facts"]["difference"]["difference"] < 0
    assert report["stored_rt_nearest"]["shared"] is None and "checked, not compared" in report["stored_rt_nearest"]["note"] and report["stored_rt_nearest"]["all_500"]["clips"] == 5
    assert report["same_path"]["bleu"]["stored_rt_nearest_share_of_two_view_gap"] is None and report["same_path"]["bleu"]["round_trip_monocular_minus_stored_rt_nearest"] is None and report["same_path"]["bleu"]["real_monocular_gap"] is not None
    assert list(report["per_signer"]) == ["c02"]
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["case"] is None


def test_a_smoke_still_checks_the_stored_form(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, declared=False, n=5)
    assert h.read(smoke=True) == 0
    h.plant_stored(form="other")
    with pytest.raises(SystemExit, match="not V1's stored read"):
        h.merge()


def test_two_views_identical_reads_as_a_clean_case_a(monkeypatch, tmp_path):
    levels = dict(PlantedReader.LEVELS, monocular=(15, 1.))
    h = Harness(monkeypatch, tmp_path, reader=PlantedReader(levels))
    report = h.pipeline()
    assert report["reading"]["case"] == "a" and report["reading"]["facts"]["intervals_overlap"] is True and report["reading"]["facts"]["monocular_higher"] is False and report["same_path"]["likelihood_minus_blank"]["monocular_gap_over_two_view_gap"]["point"] == 1.


def test_a_monocular_read_above_two_view_reads_as_case_c(monkeypatch, tmp_path):
    levels = dict(PlantedReader.LEVELS, two_view=(8, 2.), monocular=(15, 1.))
    h = Harness(monkeypatch, tmp_path, reader=PlantedReader(levels))
    report = h.pipeline()
    assert report["reading"]["case"] == "c" and report["reading"]["facts"]["monocular_higher"] is True


@pytest.mark.parametrize("difference,paired,two,mono,case", [
    (0., [-.2, .2], [4., 5.], [4., 5.], "a"), (-.5, [-1., .1], [4., 5.], [3.5, 4.5], "a"), (.5, [-.1, 1.], [4., 5.], [4.5, 5.5], "a"),
    (-.5001, [-1., .1], [4., 5.], [3.5, 4.5], "b"), (-2., [-3., -1.], [4., 5.], [2., 3.], "b"), (-.6, [-1., 0.], [4., 5.], [3.9, 4.5], "b"),
    (.5001, [-.1, 1.], [4., 5.], [4.5, 5.5], "c"), (.2, [.05, .4], [4., 5.], [4.2, 5.2], "c"), (.2, [-.05, .4], [4., 5.], [4.2, 5.2], "a"), (.2, [0., .4], [4., 5.], [4.2, 5.2], "a"),
    (-.2, [-.4, -.05], [4., 5.], [3.8, 4.8], "a"), (-.4, [-.6, -.1], [4.4, 4.5], [4., 4.1], None), (.4, [-.1, .6], [4., 4.1], [4.4, 4.5], None), (-.1, [-.2, 0.], [4., 4.5], [4.5, 5.], "a"), (-.1, [.01, .2], [4., 5.], [4., 5.], "a")])
def test_the_prestated_cases(difference, paired, two, mono, case):
    assert R.prestated_case(difference, paired, two, mono) == case


def test_the_reading_is_stated_in_the_header_before_any_code():
    assert all(text in R.__doc__ for text in R.PRESTATED.values()) and set(R.PRESTATED) == {"a", "b", "c", "settled"}
    assert R.LARGE == .5 and "report only" in R.__doc__.lower() and "(b) has no interval condition" in R.__doc__ and "no step of the job branches on it" in R.__doc__


def test_the_disclosures_name_the_instruments_domain_and_blank():
    assert "two-view" in R.DISCLOSURES["instrument"] and "pretrained" in R.DISCLOSURES["instrument"] and "durations" in R.DISCLOSURES["blank"] and "1.00-1.05" in R.DISCLOSURES["frames"]


@pytest.mark.parametrize("kw,code", [(dict(report_only=True, declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3)), 0), (dict(report_only=True, declared_read=True, checks=dict(blank_identical_to_real_two_view=0., control=dict(reproduced=True)), clips=dict(shared=3)), 0),
                                     (dict(report_only=False, declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3)), 3), (dict(report_only=True, declared_read=False, checks=dict(blank_identical_to_real_two_view=1.), clips=dict(shared=3)), 3),
                                     (dict(report_only=True, declared_read=True, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=3)), 3), (dict(report_only=True, declared_read=True, checks=dict(blank_identical_to_real_two_view=0., control=dict(reproduced=False)), clips=dict(shared=3)), 3),
                                     (dict(report_only=True, declared_read=False, checks=dict(blank_identical_to_real_two_view=0.), clips=dict(shared=0)), 3)])
def test_check_judges_plumbing_only(tmp_path, kw, code):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(kw))
    assert R.main(["check", "--report", str(path)]) == code


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
    (h.forms).mkdir(parents=True)
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


def test_the_code_digest_holds_the_job_file():
    assert R.JOB_FILE == "runpod/fleet/path_read.sh" and R.code_digest() == provenance.code_digest(extra_files=(R.JOB_FILE,)) and R.code_digest() != provenance.code_digest()


def test_a_digest_that_cannot_be_named_voids(monkeypatch):
    R.code_digest.cache_clear()
    monkeypatch.setattr(provenance, "code_digest", lambda extra_files=(): (_ for _ in ()).throw(OSError("gone")))
    with pytest.raises(SystemExit, match="no code digest"):
        R.code_digest()
    R.code_digest.cache_clear()


def test_the_constants_are_the_declared_ones():
    assert R.PREFIX == "acquisition/2026-10-17-v1/youtube-asl-v1/round20/path-read" and R.FORMS == ("real_two_view", "blank", "real_monocular", "real_monocular_unconformed", "round_trip_monocular") and R.BOOT == 2000
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


def test_the_job_smokes_five_train_clips_then_reads_the_control_first(tmp_path):
    result, calls = run_job(tmp_path)
    assert result.returncode == 0 and done_line(result) == 0
    own = [x for x in calls.splitlines() if x.startswith("scripts/path_read.py ")]
    assert len(own) == len(calls.splitlines())
    parsed = [R.build_parser().parse_args(x.split()[1:]) for x in own]
    assert [a.command for a in parsed] == ["lookups", "lookups", "read", "control", "merge", "check", "upload", "preflight", "lookups", "read", "control", "read", "merge", "upload", "check"]
    assert parsed[0].smoke_train == 0 and parsed[0].out_dir.endswith("/d20") and "smoke-" in parsed[0].out_dir and parsed[1].smoke_train == 5 and parsed[2].smoke_train == 5 and parsed[2].forms == ",".join(R.FORMS)
    reads = [a for a in parsed if a.command == "read"]
    assert reads[1].forms == "real_two_view,blank" and reads[1].smoke_train == 0 and reads[2].forms == "real_monocular,real_monocular_unconformed,round_trip_monocular" and reads[2].smoke_train == 0 and reads[1].resume and reads[2].resume
    assert all(a.device == "cuda" and a.nll_group == 4 and a.threads == 8 and a.instrument_cache.endswith("/instrument") for a in reads)
    uploads = [a for a in parsed if a.command == "upload"]
    assert uploads[0].prefix.startswith(f"{R.PREFIX}/smoke/") and "smoke-" in uploads[0].out_dir and uploads[1].prefix == R.PREFIX and uploads[1].out_dir.endswith("/d20")
    assert next(a for a in parsed if a.command == "preflight").final_key == f"{R.PREFIX}/report.json" and next(a for a in parsed if a.command == "preflight").void_prefix == f"{R.PREFIX}/void" and next(a for a in parsed if a.command == "control").void_prefix == f"{R.PREFIX}/void"


def test_smoke_only_stops_before_any_declared_read(tmp_path):
    result, calls = run_job(tmp_path, SMOKE="1")
    assert result.returncode == 0 and done_line(result) == 0 and "SMOKE done" in result.stdout and "preflight" not in calls and "/d20/forms" not in calls and "upload" in calls
    assert "--smoke-train" in calls and "read --forms" not in calls


@pytest.mark.parametrize("fail_when,expected", [("path_read.py lookups --out-dir*smoke-*/d20", 2), ("path_read.py lookups*smoke-train", 2), ("path_read.py read*smoke-train", 2), ("path_read.py control*smoke-", 2), ("path_read.py merge*smoke-", 2), ("path_read.py check --report*smoke-", 2), ("path_read.py upload --out-dir*smoke-", 2),
                                                ("path_read.py preflight", 1), ("path_read.py lookups --out-dir*/work/d20/forms", 1), ("--forms real_two_view", 1), ("--forms real_monocular", 1), ("path_read.py merge*/work/d20", 1), ("path_read.py upload --out-dir*/work/d20", 1), ("path_read.py check --report*/work/d20", 1)])
def test_any_failure_exits_nonzero_with_its_done_line_through_the_pipe_and_uploads_its_work(tmp_path, fail_when, expected):
    result, calls = run_job(tmp_path, fail_when=fail_when)
    assert result.returncode == expected and done_line(result) == expected, result.stdout + result.stderr
    assert "upload --partial" in calls.splitlines()[-1] and f"{R.PREFIX}/partial/" in calls.splitlines()[-1] and "ALERT" in result.stdout and "planted failure" in result.stdout


def test_a_smoke_failure_reads_nothing_of_d20(tmp_path):
    _, calls = run_job(tmp_path, fail_when="path_read.py read*smoke-train")
    assert "preflight" not in calls and "real_two_view,blank" not in calls


def test_a_control_miss_stops_before_any_monocular_form_with_its_own_status(tmp_path):
    result, calls = run_job(tmp_path, fail_when="path_read.py control*/work/d20", code=3)
    assert result.returncode == 3 and done_line(result) == 3 and "do not retry" in result.stdout
    assert "real_monocular" not in calls.split("control --out-dir")[-1] and "--forms real_monocular" not in calls and "merge" not in calls.split("preflight")[-1] and "upload --partial" in calls.splitlines()[-1]


def test_a_control_that_did_not_run_is_a_failed_read_not_a_void(tmp_path):
    result, calls = run_job(tmp_path, fail_when="path_read.py control*/work/d20", code=1)
    assert result.returncode == 1 and done_line(result) == 1 and "did not run" in result.stdout and "--forms real_monocular" not in calls


def test_a_control_failure_in_the_smoke_stops_the_job(tmp_path):
    result, calls = run_job(tmp_path, fail_when="path_read.py control*smoke-", code=3)
    assert result.returncode == 2 and done_line(result) == 2 and "preflight" not in calls


def test_a_log_that_cannot_be_written_is_not_hidden_by_the_pipe(tmp_path):
    result, calls = run_job(tmp_path, tee_fails=True)
    assert result.returncode == 2 and done_line(result) == 2 and "preflight" not in calls and "upload --partial" in calls.splitlines()[-1]


def test_a_failure_with_tee_succeeding_keeps_python_status(tmp_path):
    result, _ = run_job(tmp_path, fail_when="path_read.py preflight", code=9)
    assert result.returncode == 1 and done_line(result) == 1


def test_the_log_holds_every_steps_output(tmp_path):
    run_job(tmp_path)
    text = (tmp_path / "work" / "path_read.log").read_text()
    assert text.count("ran scripts/path_read.py") == 15


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


def test_the_job_passes_every_step_through_pipestatus_names_no_other_list_and_is_valid_bash():
    text = JOB.read_text()
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0
    code = [line.split(" #")[0] for line in text.splitlines() if not line.startswith("#")]
    pipes = [line for line in code if re.search(r"(?<!\|)\|(?!\|)", line)]
    assert pipes == ['  "$@" 2>&1 | tee -a "$LOG"'] and "${PIPESTATUS[@]}" in text and "set -uo pipefail" in text and "PATH-READ-DONE" in text and "transformers==4.57.6" in text
    python_lines = [line.strip() for line in code if re.search(r"\bpython \$T\b", line)]
    assert python_lines and all(line.startswith("step python $T") or "upload --partial" in line for line in python_lines)
    for word in ("c20", "C20", "val1000", "eval1000", "round15_compose_eval", "composed", "--export", "train_", "tok_v1_rule"):
        assert word not in "\n".join(code), word
    assert "D=$O/round20/path-read" in text and "O=acquisition/2026-10-17-v1/youtube-asl-v1" in text and "--smoke-train 5" in text


def test_the_job_launches_nothing_and_no_new_file_touches_a5s():
    text = (ROOT / "scripts/path_read.py").read_text() + JOB.read_text()
    for word in ("create-pod", "runpodctl", "comp_a5", "COMP_A_DECLARED", "git "):
        assert word not in text
