"""The symmetric fine-tune on synthetic data: the recipe taken whole from v1's record, what may be read and what is refused by exact record, the shards, a tiny real training, the reads, the labels, and the job."""
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
import parent_lib as Q  # noqa: E402
import path_parts as PP  # noqa: E402
import path_read as R  # noqa: E402
import s2_lib as L  # noqa: E402
import symmetric_ft as F  # noqa: E402
import tok_probes as P  # noqa: E402
import train_keypoints_to_text_t5 as trainer  # noqa: E402
from signtranslator.data.keypoint_shards import KeypointShardDataset  # noqa: E402
from signtranslator.eval import s0_metrics as M  # noqa: E402
from tests.test_path_read import Harness, N, monocular_state, plant_shard, plant_state  # noqa: E402
from tests.test_tok_probes import SpyStore  # noqa: E402
from tests.test_train_keypoints_to_text_t5_script import build_tiny  # noqa: E402

JOB = ROOT / "runpod/fleet/symmetric_ft.sh"
TINY = dict(d_model=32, heads=4, encoder_layers=1, feedforward=64, max_tokens=16, t5_name="tiny", workers=0, batch_size=2, steps=4, val_every=2, checkpoint_every=2, warmup=1, val_clips=512)
V1_ARGS = dict(batch_size=64, checkpoint_every=2000, d_model=384, decoder_lr=3e-05, device="auto", downsample=2, dropout=.1, ema_decay=.999, encoder_layers=4, feedforward=1536, fps=24., heads=6, init_key=PP.UNFITTED_KEY, init_sha256=PP.UNFITTED_SHA, limit=0, lr=1e-4, max_seconds=20., max_tokens=64, monitor_index=None, no_augment=False, patience=6, run=L.TRANSLATOR_RUN, seed=0, select_on="val", steps=30000, store="r2", t5_name="google/flan-t5-base", train_index=[Q.FK_INDEX], val_clips=512, val_every=1000, val_index=Q.FK_INDEX, warmup=500, weight_decay=.05, workers=10)
LEVELS = dict(real_two_view=(10, 1.5, .4), real_monocular=(12, 1.2, .5), nearest_two_view=(9, 1.6, .3), nearest_monocular=(13, 1., .6), blank=(0, 3., .02), still_two_view=(1, 2.9, .1), still_monocular=(1, 2.9, .05))


def v1_run(**args):
    return dict(kind="keypoints_to_text_t5", config=dict(d_model=384), config_sha256="a" * 64, train_index=[Q.FK_INDEX], val_index=Q.FK_INDEX, monitor_index=None, data_sha256="d" * 64, train_clips=19637, val_clips=1627, args=dict(V1_ARGS, **args))


class LevelReader:
    def __init__(self, sha="a" * 64, levels=None) -> None:
        self.info, self.levels, self.form, self.calls = dict(run=F.RUN, checkpoint_key="k", sha256=sha, kind="keypoints"), dict(LEVELS, **(levels or {})), None, []

    def read(self, states, texts, candidates, group=2):
        self.calls.append(self.form)
        k, nll, margin = self.levels[self.form]
        wobble = np.arange(len(states)) * .001
        return dict(hypotheses=[" ".join(texts[c[0]].split()[:k]) if k else "z y x w" for c in candidates], nll=np.array([[nll + w] + [nll + w + margin] * (candidates.shape[1] - 1) for w in wobble]))


class Sym(Harness):
    def __init__(self, monkeypatch, tmp_path, *, levels=None, **kw):
        super().__init__(monkeypatch, tmp_path, **kw)
        self.mono_reader = LevelReader(levels=levels)
        self.v1_reader = LevelReader(sha=L.TRANSLATOR_SHA)
        real = F.build_form

        def tracked(form, refs, clips, tokenizer):
            self.mono_reader.form = self.v1_reader.form = form
            return real(form, refs, clips, tokenizer)
        monkeypatch.setattr(F, "build_form", tracked)
        monkeypatch.setattr(Q, "load_mono", lambda store, args, prefix=Q.MONO_PREFIX: self.mono_reader)
        monkeypatch.setattr(P, "load_reader", lambda store, args: self.v1_reader)
        monkeypatch.setattr(F, "code_digest", lambda: "c" * 64)
        monkeypatch.setattr(P, "environment", lambda: dict(gpu="test"))
        monkeypatch.setattr(L, "D20_CLIPS", self.n)
        monkeypatch.setattr(R, "SHARED_FLOOR", 1)
        self.plant_v1_lists(["s0", "s1", "s2", "s3"], ["s4", "s5"])
        self.store.put(f"{Q.MONO_PREFIX}/trained.json", json.dumps(dict(checkpoint_key=f"{Q.MONO_PREFIX}/checkpoints/x.pt", sha256="a" * 64, step=9, train_clips=4, val_clips=2, recipe_differences={})).encode())

    def plant_v1_lists(self, train, val, extra=None):
        parts = []
        for partition, signs in (("train", train), ("val", val)):
            meta = json.dumps([dict(clip_id=s, video_id=s, partition=partition, text=f"text of {s}", frames=20) for s in signs]).encode()
            key = f"fk/{partition}/shard-00000"
            self.store.put(f"{key}.json", meta)
            parts.append(dict(partition=partition, shards=[dict(key=f"{key}.npz", sha256="0" * 64, meta_sha256=hashlib.sha256(meta).hexdigest(), clips=len(signs), partitions={partition: len(signs)})]))
        parts.append(dict(partition="test", shards=[dict(key="fk/test/shard-00000.npz", sha256="0" * 64, meta_sha256="0" * 64, clips=1, partitions=dict(test=1))]))
        raw = json.dumps(dict(parts=parts, source_fps=24.)).encode()
        self.store.put(Q.FK_INDEX, raw)
        self.monkeypatch.setattr(Q, "FK_INDEX_SHA", hashlib.sha256(raw).hexdigest())

    def plant_v1_run(self, **args):
        raw = json.dumps(v1_run(**args)).encode()
        self.store.put(Q.V1_RUN_JSON, raw)
        self.monkeypatch.setattr(Q, "V1_RUN_JSON_SHA", hashlib.sha256(raw).hexdigest())

    def plant_manifest(self, signs=None, c20=(), **per_sign):
        """A manifest of the given signs, all How2Sign c02 train, with changes by sign; and the C20 derivation."""
        signs = signs or ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
        records = []
        for s in signs:
            records.append(dict(dict(sign=s, corpus=L.H2S, partition="train", signer_component="c02", text=f"text of {s}", continuous=dict(state_key=f"two/{s}.npz", state_sha256="0" * 64, seconds=2.)), **per_sign.get(s, {})))
        self.monkeypatch.setattr(L, "load_manifest_at", lambda store, key, sha: dict(records=records))
        self.monkeypatch.setattr(F, "c20_signs", lambda store, manifest: frozenset(c20))
        return records

    def main(self, *argv):
        return F.main([argv[0], *(() if argv[0] == "check" else ("--store", f"dir:{self.tmp_path / 's'}")), *argv[1:]])

    def plant_canary(self):
        read = {n: json.loads((self.forms / f"form-{n}.json").read_text()) for n in F.CONTROL}
        real, blank = read["v1__real_two_view"], read["v1__blank"]
        stored = dict(signs=real["signs"], references=real["references"], forms=dict(real=dict(hypotheses=real["hypotheses"], per_clip=dict(nll=[r[0] for r in real["nll"]])), blank=dict(hypotheses=blank["hypotheses"], per_clip=dict(nll=[r[0] for r in blank["nll"]]))))
        raw = json.dumps(stored).encode()
        self.store.put(P.CANARY_REPORT, raw)
        self.monkeypatch.setattr(P, "CANARY_REPORT_SHA", hashlib.sha256(raw).hexdigest())
        self.monkeypatch.setattr(P, "CANARY", dict(P.CANARY, real=M.bleu_ci_fast(real["hypotheses"], real["references"])[0], blank=M.bleu_ci_fast(blank["hypotheses"], blank["references"])[0]))

    def plant_v1_stored(self, two=(15, 1.), mono=(7, 2.5), blank=(0, 3.)):
        """v1's stored forms: per clip, as v1's reads hold them; high on two-view."""
        records = self.records
        texts = [r["text"] for r in records]
        table = {}
        for form, (k, nll) in dict(nearest_two_view=two, nearest_monocular=mono, real_two_view=(14, 1.1), blank=blank).items():
            zeros = [0.] * len(texts)
            raw = json.dumps(dict(form=form, signs=[r["sign"] for r in records], groups=[r["signer_component"] for r in records], motion=dict(wrist_speed=zeros, wrist_spread=zeros), hands=dict(left_hand=dict(speed=zeros, variation=zeros)), tracked=dict(wrists=zeros), references=texts, hypotheses=[" ".join(t.split()[:k]) if k else "z y x w" for t in texts], nll=[[nll + i * .001] for i in range(len(texts))])).encode()
            self.store.put(f"{Q.PARTS_FORMS}/v-{form}.json", raw)
            table[form] = (f"v-{form}.json", hashlib.sha256(raw).hexdigest())
        self.monkeypatch.setattr(Q, "STORED_V1", table)
        read = lambda f: json.loads(self.store.get(f"{Q.PARTS_FORMS}/v-{f}.json"))      # noqa: E731
        self.monkeypatch.setattr(F, "PLAN", dict(F.PLAN, two_bleu=M.bleu_ci_fast(read("nearest_two_view")["hypotheses"], texts)[0], two_likelihood=3. - two[1], mono_bleu=M.bleu_ci_fast(read("nearest_monocular")["hypotheses"], texts)[0], mono_likelihood=3. - mono[1]))

    def plant_val_control(self, misses=()):
        self.forms.mkdir(parents=True, exist_ok=True)
        (self.forms.parent / "coverage.json").write_text(json.dumps(dict(code_digest="c" * 64, monocular_clips=dict(train=4, val=2))))
        (self.forms / "val_control.json").write_text(json.dumps(dict(code_digest="c" * 64, misses=list(misses), clips=2)))

    def read_forms(self, forms, *extra):
        return self.main("read", "--forms", forms, "--out-dir", str(self.forms), *extra)

    def to_control(self):
        assert self.read_forms("control") == 0
        self.plant_canary()
        assert self.main("control", "--out-dir", str(self.forms), "--void-prefix", "sf/void") == 0

    def merge(self, name="report.json"):
        out = self.tmp_path / name
        assert self.main("merge", "--out-dir", str(self.forms), "--output-file", str(out)) == 0
        return json.loads(out.read_text())

    def pipeline(self, **stored):
        self.to_control()
        assert self.read_forms("rest", "--resume") == 0
        self.plant_val_control()
        self.plant_v1_stored(**stored)
        return self.merge()


class InProcessPool:
    def __init__(self, workers, initializer, initargs):
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap(self, fn, items, chunksize=1):
        return map(fn, items)


def report_rows(signs, h, monkeypatch, missing=()):
    rows = []
    for i, s in enumerate(signs):
        if s in missing:
            continue
        state = monocular_state(i)
        digest = plant_state(h.store, h.tmp_path, s, state)
        rows.append(dict(sample_id=s, state_key=R.state_key_of(s), state_sha256=digest, status="converted"))
    plant_shard(monkeypatch, h.store, rows)
    return rows


@pytest.fixture
def built(monkeypatch, tmp_path):
    """A planted world with its monocular states, ready to build."""
    h = Sym(monkeypatch, tmp_path)
    h.plant_manifest()
    h.plant_v1_run()
    report_rows(["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"], h, monkeypatch)
    monkeypatch.setattr(F, "Pool", InProcessPool)
    from signtranslator.data import keypoint_shards
    monkeypatch.setattr(keypoint_shards, "_SHARDS", {})     # shards are kept per process by key: each test has its own
    return h


# ---- the recipe ------------------------------------------------------------------------------------------------------------------------

def test_the_recipes_pin_and_shape_are_v1s():
    assert F.RUN == "back-translator-kp2d-uwb-ft-mono-v1" and Q.V1_RUN_JSON_SHA.startswith("289e579d") and PP.UNFITTED_SHA.startswith("f1ff9446") and F.SAME_RECIPE["steps"] == 30000 and F.SAME_RECIPE["lr"] == 1e-4 and F.SAME_RECIPE["decoder_lr"] == 3e-5 and F.SAME_RECIPE["patience"] == 6
    F.check_recipe(v1_run())


@pytest.mark.parametrize("change,word", [(dict(lr=3e-4), "lr"), (dict(steps=100), "steps"), (dict(patience=12), "patience"), (dict(init_sha256="0" * 64), "init"), (dict(init_key="other"), "init"), (dict(train_index=["other"]), "indexes"), (dict(no_augment=True), "no_augment"), (dict(select_on="monitor"), "select_on")])
def test_a_run_record_other_than_the_headers_recipe_refuses(change, word):
    with pytest.raises(SystemExit, match=word):
        F.check_recipe(v1_run(**change))


def test_the_recipe_step_checks_v1s_stored_pins(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    h.plant_v1_run()
    h.plant_v1_stored()
    assert h.main("recipe") == 0
    monkeypatch.setattr(Q, "STORED_V1", dict(Q.STORED_V1, blank=(Q.STORED_V1["blank"][0], "0" * 64)))
    with pytest.raises(SystemExit, match="SHA-256"):
        h.main("recipe")


def test_v1s_run_record_on_other_bytes_refuses(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    h.plant_v1_run()
    h.store.put(Q.V1_RUN_JSON, b"{}")
    with pytest.raises(SystemExit, match="SHA-256"):
        F.load_v1_run(h.store)


def parse_trainer_args(argv):
    seen = {}

    class Stop(Exception):
        pass
    import argparse
    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        parsed = original(self, args, namespace)
        seen.update(vars(parsed))
        raise Stop()
    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(Stop):
            trainer.main(argv)
    finally:
        argparse.ArgumentParser.parse_args = original
    return seen


def test_the_trainer_takes_every_one_of_v1s_arguments_and_only_run_and_indexes_change():
    argv = F.trainer_argv(V1_ARGS, index="mono/index.json", run=F.RUN)
    parsed = parse_trainer_args(argv)
    expected = dict(V1_ARGS, run=F.RUN, train_index=["mono/index.json"], val_index="mono/index.json")
    assert parsed == expected


def test_a_smoke_changes_the_budget_and_nothing_else():
    argv = F.trainer_argv(V1_ARGS, index="i", run="r", overrides=F.SMOKE_OVERRIDES)
    parsed = parse_trainer_args(argv)
    assert parsed["steps"] == 10 and parsed["init_sha256"] == PP.UNFITTED_SHA and parsed["lr"] == 1e-4 and parsed["weight_decay"] == .05 and parsed["patience"] == 6


def test_recipe_differences_name_anything_beyond_the_headers_changes():
    v1 = v1_run()
    same = json.loads(json.dumps(v1))
    same["args"].update(run=F.RUN, train_index=["m"], val_index="m")
    same.update(train_index=["m"], val_index="m", data_sha256="e" * 64, train_clips=1, val_clips=1)
    assert F.recipe_differences(same, v1) == {}
    for field, value in (("lr", 1.), ("seed", 1), ("steps", 5), ("init_sha256", "0" * 64), ("workers", 2), ("device", "cpu")):
        changed = json.loads(json.dumps(same))
        changed["args"][field] = value
        assert f"args.{field}" in F.recipe_differences(changed, v1)
    for field, value in (("config_sha256", "0" * 64), ("config", dict(d_model=1)), ("monitor_index", "x")):
        assert field in F.recipe_differences(dict(same, **{field: value}), v1)


# ---- what may be read ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("change,word", [(dict(signer_component="c00"), "c00"), (dict(partition="test"), "not a How2Sign train or val"), (dict(partition="heldout_official"), "not a How2Sign train or val"), (dict(corpus="other"), "not a How2Sign train or val")])
def test_a_c00_test_held_out_or_other_record_refuses(change, word):
    record = dict(dict(corpus=L.H2S, partition="train", signer_component="c02"), **change)
    with pytest.raises(SystemExit, match=word):
        F.check_training_record("x", record)


def test_a_clip_outside_the_manifest_refuses_and_a_train_or_val_clip_passes():
    with pytest.raises(SystemExit, match="not in the manifest"):
        F.check_training_record("x", None)
    F.check_training_record("x", dict(corpus=L.H2S, partition="val", signer_component="c01"))


def planted_c20(monkeypatch, d20, c20, stored_d20=None):
    import build_round20_sets
    monkeypatch.setattr(build_round20_sets, "build", lambda manifest: ([dict(sign=s) for s in d20], [dict(sign=s) for s in c20], {}))
    monkeypatch.setattr(L, "load_read_list", lambda store, manifest: [dict(sign=s) for s in (stored_d20 if stored_d20 is not None else d20)])


def test_c20_is_derived_held_to_its_fingerprint_and_to_d20(monkeypatch):
    signs = [f"c{i}" for i in range(1000)]
    monkeypatch.setattr(F, "C20_FINGERPRINT", hashlib.sha256("\n".join(sorted(signs)).encode()).hexdigest())
    planted_c20(monkeypatch, ["d0"], signs)
    assert F.c20_signs(None, {}) == frozenset(signs)
    planted_c20(monkeypatch, ["d0"], signs[:-1] + ["other"])
    with pytest.raises(SystemExit, match="fingerprint"):
        F.c20_signs(None, {})
    planted_c20(monkeypatch, ["d0"], signs, stored_d20=["d1"])
    with pytest.raises(SystemExit, match="reproduce D20"):
        F.c20_signs(None, {})
    planted_c20(monkeypatch, ["d0"], signs[:-1])
    with pytest.raises(SystemExit, match="fingerprint"):
        F.c20_signs(None, {})


def test_c20s_list_is_never_fetched_and_its_fingerprint_is_pinned():
    text = (ROOT / "scripts/symmetric_ft.py").read_text().split('"""', 2)[2]
    assert 'c20.json"' not in text and F.C20_COUNT == 1000 and F.C20_FINGERPRINT.startswith("4b102007")


def test_the_fixed_sample_is_the_trainers():
    rows = [dict(clip_id=str(i)) for i in range(1627)]
    want = sorted(random.Random(1234).sample(range(1627), 512))
    assert [r["clip_id"] for r in F.fixed_sample(rows)] == [str(i) for i in want]
    assert F.fixed_sample(rows[:3]) == rows[:3]


def test_c20_clips_in_v1s_lists_are_dropped_by_exact_record_in_v1s_order(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    h.plant_v1_lists(["s0", "s1", "s2", "s3"], ["s4", "s5", "s6"])
    h.plant_manifest(c20=("s1", "s5"))
    plan = F.planned_lists(h.store, L.load_manifest_at(h.store, "", ""))
    assert [r["clip_id"] for r in plan["train"]] == ["s0", "s2", "s3"] and [r["clip_id"] for r in plan["val"]] == ["s4", "s6"]
    assert plan["refused_c20"] == dict(train=1, val=1) and plan["v1_counts"] == dict(train=4, val=3)


@pytest.mark.parametrize("sign,change,word", [("s1", dict(signer_component="c00"), "c00"), ("s4", dict(partition="test"), "not a How2Sign train or val"), ("s4", dict(partition="heldout_official"), "not a How2Sign train or val")])
def test_a_c00_or_test_clip_in_v1s_lists_refuses_the_plan(monkeypatch, tmp_path, sign, change, word):
    h = Sym(monkeypatch, tmp_path)
    h.plant_manifest(**{sign: change})
    with pytest.raises(SystemExit, match=word):
        F.planned_lists(h.store, L.load_manifest_at(h.store, "", ""))


def test_no_test_series_is_opened(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    h.plant_manifest()
    F.planned_lists(h.store, L.load_manifest_at(h.store, "", ""))
    assert not h.store.exists("fk/test/shard-00000.json")      # planted nowhere: opening it would have raised


def test_match_rows_lists_what_is_lost_and_refuses_another_object():
    rows = [dict(clip_id=x, text="t", partition="train") for x in ("a", "b", "c")]
    reports = dict(a=dict(state_key=R.state_key_of("a"), state_sha256="f" * 64), b=dict(state_key=None, status="failed"))
    found, lost = F.match_rows(rows, reports)
    assert [f["clip_id"] for f in found] == ["a"] and {x["sign"]: x["reason"] for x in lost} == dict(b="no_state_in_row", c="no_monocular_row")
    with pytest.raises(SystemExit, match="refused unread"):
        F.match_rows(rows[:1], dict(a=dict(state_key="other.npz", state_sha256="f" * 64)))


def test_coverage_counts_clips_per_path_and_says_enough_at_the_floor(built, capsys):
    assert built.main("coverage", "--out-dir", str(built.tmp_path / "w")) == 0
    report = json.loads((built.tmp_path / "w/coverage.json").read_text())
    assert report["two_view_clips"] == dict(train=4, val=2) and report["monocular_clips"] == dict(train=4, val=2) and report["train_share_of_v1"] == 1. and report["enough"] is True and report["by_signer"]["train"] == {"train:c02": 4}
    assert "ALERT" not in capsys.readouterr().out


def test_under_the_floor_coverage_says_so_and_stops_with_4(monkeypatch, tmp_path, capsys):
    h = Sym(monkeypatch, tmp_path)
    h.plant_v1_lists(["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"], ["s4"])
    h.plant_manifest(signs=[f"s{i}" for i in range(10)], c20=("s9",))
    report_rows([f"s{i}" for i in range(10)], h, monkeypatch, missing=("s8",))
    assert h.main("coverage", "--out-dir", str(tmp_path / "w")) == F.EXIT_COVERAGE
    out = capsys.readouterr().out
    assert "ALERT only 80.0%" in out
    report = json.loads((tmp_path / "w/coverage.json").read_text())
    assert report["monocular_clips"]["train"] == 8 and report["refused_c20"]["train"] == 1 and report["lost"]["train"]["by_reason"] == dict(no_monocular_row=1) and report["enough"] is False


def test_exactly_at_the_floor_is_enough(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    h.plant_v1_lists([f"s{i}" for i in range(10)], ["s0"])
    h.plant_manifest(signs=[f"s{i}" for i in range(10)], c20=("s9",))
    report_rows([f"s{i}" for i in range(10)], h, monkeypatch)
    assert h.main("coverage", "--out-dir", str(tmp_path / "w")) == 0


# ---- the shards ------------------------------------------------------------------------------------------------------------------------

def test_the_shards_hold_the_monocular_states_v1s_text_and_order_and_no_c20_clip(built, monkeypatch):
    train = ["s0", "s1", "s2", "s3"] + [f"t{i}" for i in range(16)]
    built.plant_v1_lists(train, ["s4", "s5"])
    built.plant_manifest(signs=train + ["s4", "s5"], c20=("s1",))
    report_rows(train + ["s4", "s5"], built, monkeypatch)
    assert built.main("build", "--workers", "1") == 0
    index = json.loads(built.store.get(f"{F.INDEX_PREFIX}/index.json"))
    assert [p["partition"] for p in index["parts"]] == ["train", "val"] and index["counts"] == dict(train=19, val=2) and index["refused_c20"] == dict(train=1, val=0) and index["source_fps"] == 24. and index["smoke"] is False
    dataset = KeypointShardDataset(built.store, [f"{F.INDEX_PREFIX}/index.json"], partition="train", target_fps=24.)
    assert dataset.ids == [s for s in train if s != "s1"] and dataset.texts[:2] == ["text of s0", "text of s2"]
    val = KeypointShardDataset(built.store, [f"{F.INDEX_PREFIX}/index.json"], partition="val", target_fps=24.)
    assert val.ids == ["s4", "s5"]
    assert "s1" not in json.dumps(index) and not KeypointShardDataset(built.store, [f"{F.INDEX_PREFIX}/index.json"], partition="test", target_fps=24.).ids


def test_a_clips_points_are_what_the_fk_builder_gives_its_monocular_state(built):
    from evaluate_text_to_motion import at_rate
    from signtranslator.data.keypoints2d import from_state, normalise
    assert built.main("build", "--workers", "1") == 0
    dataset = KeypointShardDataset(built.store, [f"{F.INDEX_PREFIX}/index.json"], partition="train", target_fps=24.)
    points, valid = from_state(at_rate(monocular_state(1), L.FPS))
    want = normalise(points, valid, y_down=False)
    got_points, got_valid, _ = dataset.clips[1]
    assert np.array_equal(got_valid, valid) and np.allclose(got_points, want, atol=2e-2)


def bad_state_world(built, monkeypatch, bad):
    signs = [f"t{i}" for i in range(20)] + ["s4", "s5"]
    built.plant_v1_lists(signs[:20], ["s4", "s5"])
    built.plant_manifest(signs=signs)
    rows = report_rows(signs, built, monkeypatch)
    for row in rows[:bad]:
        row["state_sha256"] = "0" * 64
    plant_shard(monkeypatch, built.store, rows)


def test_a_bad_state_is_skipped_and_counted_up_to_the_limit(built, monkeypatch):
    monkeypatch.setattr(F, "BAD_STATE_LIMIT", 2)
    bad_state_world(built, monkeypatch, 2)
    assert built.main("build", "--workers", "1") == 0
    index = json.loads(built.store.get(f"{F.INDEX_PREFIX}/index.json"))
    assert index["counts"]["train"] == 18 and [x["reason"] for x in index["lost"]["train"]] == ["bad_state", "bad_state"] and "hash" in index["lost"]["train"][0]["detail"]


def test_over_the_limit_of_bad_states_the_build_stops_and_no_index_stands(built, monkeypatch):
    monkeypatch.setattr(F, "BAD_STATE_LIMIT", 2)
    bad_state_world(built, monkeypatch, 3)
    with pytest.raises(SystemExit, match="over 2 bad states"):
        built.main("build", "--workers", "1")
    assert not built.store.exists(f"{F.INDEX_PREFIX}/index.json") and F.BAD_STATE_LIMIT == 2


def test_the_bad_state_limit_is_small_and_stated():
    assert 0 < F.BAD_STATE_LIMIT <= 20 and "BAD_STATE_LIMIT" in F.__doc__


def test_the_cut_at_20_seconds_is_counted_on_both_paths(built, monkeypatch):
    built.plant_v1_lists(["s0", "s1"], ["s4", "s5"])
    monkeypatch.setattr(F, "MAX_FRAMES", 22)     # monocular clips hold 24 frames, two-view's list says 20
    monkeypatch.setattr(F, "COVERAGE_FLOOR", 0.)
    assert built.main("build", "--workers", "1") == 0
    index = json.loads(built.store.get(f"{F.INDEX_PREFIX}/index.json"))
    assert index["cut_at_20s"] == dict(train=dict(two_view=0, monocular=2), val=dict(two_view=0, monocular=2))


def test_a_missing_object_is_listed_and_under_the_floor_no_index_is_stored(built, monkeypatch, capsys):
    (built.store.root / R.state_key_of("s0")).unlink()
    assert built.main("build", "--workers", "1") == F.EXIT_COVERAGE
    assert "under 90%" in capsys.readouterr().out and not built.store.exists(f"{F.INDEX_PREFIX}/index.json")


def test_a_clip_with_no_shoulders_is_listed_as_the_fk_builder_drops_it(built, monkeypatch):
    built.plant_v1_lists([f"s{i}" for i in range(4)] + [f"t{i}" for i in range(16)], ["s4"])
    signs = [f"s{i}" for i in range(5)] + [f"t{i}" for i in range(16)]
    built.plant_manifest(signs=signs)
    report_rows(signs, built, monkeypatch)
    import signtranslator.data.keypoints2d as K
    real = K.from_state
    calls = {"n": 0}

    def nothing_on_second(state):
        points, valid = real(state)
        calls["n"] += 1
        return (points, np.zeros_like(valid)) if calls["n"] == 2 else (points, valid)
    monkeypatch.setattr(K, "from_state", nothing_on_second)
    assert built.main("build", "--workers", "1") == 0
    index = json.loads(built.store.get(f"{F.INDEX_PREFIX}/index.json"))
    assert index["counts"]["train"] == 19 and [x["reason"] for x in index["lost"]["train"]] == ["no_shoulders"]


def test_convert_refuses_another_clips_state():
    F._WORKER["store"] = None
    with pytest.raises(ValueError, match="refused"):
        F.convert(dict(clip_id="a", state_key="states/b.npz", state_sha256="0" * 64))


def test_a_finished_build_is_left(built, capsys):
    assert built.main("build", "--workers", "1") == 0
    first = built.store.get(f"{F.INDEX_PREFIX}/index.json")
    assert built.main("build", "--workers", "1") == 0 and '"stored": true' in capsys.readouterr().out and built.store.get(f"{F.INDEX_PREFIX}/index.json") == first


def test_a_smokes_list_holds_the_longest_and_the_never_tracked_once_each(monkeypatch):
    import tok_v0 as V
    records = [dict(sign=f"s{i}", partition="train", signer_component="c02") for i in range(10)]
    monkeypatch.setattr(L, "smoke_records", lambda manifest, n: records[:n])
    monkeypatch.setattr(P, "longest_train_records", lambda store, k: [records[8], records[0]][:k])
    monkeypatch.setattr(V, "never_tracked_records", lambda store, k: [records[9]][:k])
    out = F.smoke_list(None, {})
    assert [r["sign"] for r in out] == ["s8", "s9"] + [f"s{i}" for i in range(8)]
    monkeypatch.setattr(V, "never_tracked_records", lambda store, k: [dict(records[9], signer_component="c00")])
    with pytest.raises(SystemExit, match="not a train clip"):
        F.smoke_list(None, {})


def test_a_smoke_build_takes_c02_train_clips_only(built, monkeypatch):
    import tok_v0 as V
    records = built.plant_manifest()
    monkeypatch.setattr(L, "smoke_records", lambda manifest, n: records[:n])
    monkeypatch.setattr(P, "longest_train_records", lambda store, k: [])
    monkeypatch.setattr(V, "never_tracked_records", lambda store, k: [])
    assert built.main("build", "--smoke", "--prefix", "smk/idx", "--workers", "1") == 0
    index = json.loads(built.store.get("smk/idx/index.json"))
    assert index["smoke"] is True and index["counts"] == dict(train=6, val=2)
    monkeypatch.setattr(L, "smoke_records", lambda manifest, n: [dict(records[0], partition="test")] + records[1:n])
    with pytest.raises(SystemExit, match="not a train clip"):
        built.main("build", "--smoke", "--prefix", "smk/idx2", "--workers", "1")


@pytest.mark.parametrize("parts", [["train"], ["train", "val", "test"], ["val", "train"]])
def test_an_index_of_other_series_refuses(parts):
    with pytest.raises(SystemExit, match="not train and val"):
        F.check_index(dict(parts=[dict(partition=p) for p in parts]))
    F.check_index(dict(parts=[dict(partition="train"), dict(partition="val")]))


# ---- training --------------------------------------------------------------------------------------------------------------------------

def plant_init(h):
    """The parent as a tiny checkpoint, its key and hash the pins'."""
    import io
    import torch
    from signtranslator.generation.keypoints_to_text_t5 import KIND, KeypointsToT5Config
    args = dict(V1_ARGS, **TINY)
    config = KeypointsToT5Config(fps=24., t5_name="tiny", d_model=32, heads=4, encoder_layers=1, feedforward=64, dropout=.1, downsample=2, max_tokens=16)
    model = build_tiny(config)
    buffer = io.BytesIO()
    torch.save(dict(kind=KIND, model=model.state_dict(), ema=None, config_sha256=config.sha256(), step=1), buffer)
    return args, buffer.getvalue()


def tiny_world(built, monkeypatch):
    args, payload = plant_init(built)
    built.store.put(PP.UNFITTED_KEY, payload)
    sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(PP, "UNFITTED_SHA", sha)
    built.plant_v1_run(**dict(args, init_sha256=sha))
    monkeypatch.setattr(F, "SAME_RECIPE", dict(seed=0, select_on="val"))
    calls = []

    def run_in_process(argv, cwd=None):
        calls.append(argv)
        return type("D", (), dict(returncode=trainer.main(argv[2:], build_model=build_tiny)))()
    monkeypatch.setattr(subprocess, "run", run_in_process)
    return calls


def S_form(h, form):
    clips, lost, _ = R.open_monocular(h.store, h.records, h.refs, True)
    return F.build_form.__wrapped__(form, h.refs, clips, h.tokenizer) if hasattr(F.build_form, "__wrapped__") else F.build_form(form, h.refs, clips, h.tokenizer)


def test_the_tiny_recipe_builds_trains_from_the_parent_and_passes_its_record(built, monkeypatch):
    calls = tiny_world(built, monkeypatch)
    assert built.main("build", "--workers", "1") == 0
    assert built.main("train") == 0
    run = json.loads(built.store.get(f"{Q.MONO_PREFIX}/run.json"))
    assert run["train_clips"] == 4 and run["val_clips"] == 2 and run["train_index"] == [f"{F.INDEX_PREFIX}/index.json"] and run["args"]["init_sha256"] == PP.UNFITTED_SHA and run["args"]["val_clips"] == 512
    assert built.store.exists(f"{Q.MONO_PREFIX}/DONE") and F.recipe_differences(run, json.loads(built.store.get(Q.V1_RUN_JSON))).keys() <= {"config", "config_sha256", "args.store"}
    assert len(calls) == 1
    # a second train is not re-entered
    assert built.main("train") == 0 and len(calls) == 1


def test_a_wrong_parent_hash_stops_the_trainer(built, monkeypatch):
    tiny_world(built, monkeypatch)
    assert built.main("build", "--workers", "1") == 0
    built.store.put(PP.UNFITTED_KEY, b"other bytes")
    with pytest.raises(SystemExit, match="differs from --init-sha256"):
        built.main("train")


def test_a_stale_index_does_not_train(built, monkeypatch):
    built.plant_v1_run()
    assert built.main("build", "--workers", "1") == 0
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    # the plan moved: a clip of the index is now refused
    built.plant_manifest(c20=("s2",))
    with pytest.raises(SystemExit, match="not this plan's, exactly and in order"):
        built.main("train")


def test_an_index_in_another_order_or_with_another_text_does_not_train(built, monkeypatch):
    built.plant_v1_run()
    assert built.main("build", "--workers", "1") == 0
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    built.plant_v1_lists(["s1", "s0", "s2", "s3"], ["s4", "s5"])
    with pytest.raises(SystemExit, match="not this plan's"):
        built.main("train")


def test_a_clip_list_on_other_bytes_or_a_smokes_index_does_not_train(built, monkeypatch):
    built.plant_v1_run()
    assert built.main("build", "--workers", "1") == 0
    index = json.loads(built.store.get(f"{F.INDEX_PREFIX}/index.json"))
    with pytest.raises(SystemExit, match="smoke's index"):
        F.check_lists(built.store, dict(index, smoke=True))
    built.store.put(f"{F.INDEX_PREFIX}/train/shard-00000.json", b"[]")
    with pytest.raises(SystemExit, match="SHA-256"):
        F.check_lists(built.store, index)


def test_a_clip_the_index_lists_as_lost_is_not_required(built, monkeypatch):
    built.plant_v1_run()
    signs = [f"t{i}" for i in range(20)] + ["s4", "s5"]
    built.plant_v1_lists(signs[:20], ["s4", "s5"])
    built.plant_manifest(signs=signs)
    rows = report_rows(signs, built, monkeypatch)
    rows[0]["state_sha256"] = "0" * 64
    plant_shard(monkeypatch, built.store, rows)
    assert built.main("build", "--workers", "1") == 0
    F.check_lists(built.store, json.loads(built.store.get(f"{F.INDEX_PREFIX}/index.json")))


def test_a_failing_trainer_leaves_no_done(built, monkeypatch):
    built.plant_v1_run()
    assert built.main("build", "--workers", "1") == 0
    monkeypatch.setattr(subprocess, "run", lambda argv, cwd=None: type("D", (), dict(returncode=9))())
    with pytest.raises(SystemExit, match="stopped with 9"):
        built.main("train")
    assert not built.store.exists(f"{Q.MONO_PREFIX}/DONE")


def test_training_needs_an_index_and_refuses_a_resume_unasked(built, monkeypatch):
    built.plant_v1_run()
    with pytest.raises(SystemExit, match="no index"):
        built.main("train")
    assert built.main("build", "--workers", "1") == 0
    built.store.put(f"{Q.MONO_PREFIX}/checkpoints/step-00001000.pt", b"x")
    monkeypatch.setattr(subprocess, "run", lambda argv, cwd=None: type("D", (), dict(returncode=0))())
    with pytest.raises(SystemExit, match="resets its patience"):
        built.main("train")
    assert built.main("train", "--allow-resume") == 0 and [k for k in built.store.list(f"{Q.MONO_PREFIX}/") if "/resumed-" in k]


def test_v1s_recipe_is_checked_before_the_trainer_runs(built, monkeypatch):
    built.plant_v1_run(lr=9.)
    assert built.main("build", "--workers", "1") == 0
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    with pytest.raises(SystemExit, match="lr"):
        built.main("train")


def trained_world(h, **change):
    h.plant_v1_run()
    v1 = json.loads(h.store.get(Q.V1_RUN_JSON))
    new = json.loads(json.dumps(v1))
    index = f"{F.INDEX_PREFIX}/index.json"
    new["args"].update(run=F.RUN, train_index=[index], val_index=index)
    new.update(train_index=[index], val_index=index, train_clips=4, val_clips=2)
    new["args"].update(change)
    key = f"{Q.MONO_PREFIX}/checkpoints/step-00002000.pt"
    h.store.put(key, b"weights")
    h.store.put(f"{Q.MONO_PREFIX}/run.json", json.dumps(new).encode())
    h.store.put(f"{Q.MONO_PREFIX}/best.json", json.dumps(dict(checkpoint_key=key, step=2000, selected_on="val/bleu")).encode())
    h.store.put(index, b"{}")
    h.store.put(f"{Q.MONO_PREFIX}/DONE", b"done")
    (h.store.root / Q.TRAINED_RECORD).unlink(missing_ok=True)
    return key


def test_the_trained_record_names_the_best_checkpoint_by_its_hash_and_is_written_once(monkeypatch, tmp_path, capsys):
    h = Sym(monkeypatch, tmp_path)
    key = trained_world(h)
    assert h.main("trained") == 0
    record = json.loads(h.store.get(Q.TRAINED_RECORD))
    assert record["checkpoint_key"] == key and record["sha256"] == hashlib.sha256(b"weights").hexdigest() and record["step"] == 2000 and record["recipe_differences"] == {} and record["smoke"] is False and record["train_clips"] == 4
    h.store.put(key, b"changed")
    assert h.main("trained") == 0 and json.loads(h.store.get(Q.TRAINED_RECORD))["sha256"] == record["sha256"] and '"stored": true' in capsys.readouterr().out


@pytest.mark.parametrize("change,word", [(dict(lr=9.), "args.lr"), (dict(init_sha256="0" * 64), "args.init_sha256"), (dict(seed=3), "args.seed")])
def test_a_run_off_v1s_recipe_refuses_its_record(monkeypatch, tmp_path, change, word):
    h = Sym(monkeypatch, tmp_path)
    trained_world(h, **change)
    with pytest.raises(SystemExit, match=word):
        h.main("trained")
    assert not h.store.exists(Q.TRAINED_RECORD)


def test_a_smoke_run_records_its_differences_rather_than_refusing(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    trained_world(h, steps=10)
    assert h.main("trained", "--smoke") == 0
    assert "args.steps" in json.loads(h.store.get(Q.TRAINED_RECORD))["recipe_differences"]


def test_a_run_on_another_index_or_checkpoint_outside_it_refuses(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    trained_world(h, train_index=[Q.FK_INDEX])
    with pytest.raises(SystemExit, match="trained on"):
        h.main("trained")
    trained_world(h)
    h.store.put(f"{Q.MONO_PREFIX}/best.json", json.dumps(dict(checkpoint_key=f"training-runs/{L.TRANSLATOR_RUN}/checkpoints/step-00009000.pt", step=1, selected_on="x")).encode())
    with pytest.raises(SystemExit, match="names"):
        h.main("trained")


def test_an_unfinished_run_has_no_record(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    trained_world(h)
    (h.store.root / f"{Q.MONO_PREFIX}/DONE").unlink()
    with pytest.raises(SystemExit, match="has not finished"):
        h.main("trained")


def test_the_selection_control_reads_the_trainers_sample_against_best(built, monkeypatch, capsys):
    tiny_world(built, monkeypatch)
    assert built.main("build", "--workers", "1") == 0
    assert built.main("train") == 0
    best = json.loads(built.store.get(f"{Q.MONO_PREFIX}/best.json"))
    import io
    import torch
    from signtranslator.generation.keypoints_to_text_t5 import KeypointsToT5Config
    saved = torch.load(io.BytesIO(built.store.get(best["checkpoint_key"])), map_location="cpu", weights_only=False)
    model = build_tiny(KeypointsToT5Config(fps=24., t5_name="tiny", d_model=32, heads=4, encoder_layers=1, feedforward=64, dropout=.1, downsample=2, max_tokens=16))
    model.load_state_dict(saved["ema"])
    monkeypatch.setattr(Q, "load_mono", lambda store, args, prefix: L.Reader(model.eval(), "keypoints", dict(sha256="a" * 64), torch.device("cpu")))
    out = built.tmp_path / "vc"
    assert built.main("val-control", "--out-dir", str(out), "--batch-size", "2") == 0
    record = json.loads((out / "val_control.json").read_text())
    assert record["clips"] == 2 and record["misses"] == [] and record["reproduced"] is True and abs(record["loss"] - best["validation"]["val/loss"]) <= .01
    best["validation"]["val/loss"] += 1.
    built.store.put(f"{Q.MONO_PREFIX}/best.json", json.dumps(best).encode())
    out2 = built.tmp_path / "vc2"
    assert built.main("val-control", "--out-dir", str(out2), "--batch-size", "2") == 0
    assert json.loads((out2 / "val_control.json").read_text())["misses"] == ["loss"] and "ALERT" in capsys.readouterr().out


# ---- reads and reading ---------------------------------------------------------------------------------------------------------------------

def test_rest_needs_a_passed_control_before_it_loads_a_state(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    loaded = []
    monkeypatch.setattr(L, "load_states", lambda store, recs: loaded.append(1))
    with pytest.raises(SystemExit, match="no passed control"):
        h.read_forms("rest")
    assert not loaded


@pytest.mark.parametrize("change,match", [(dict(signer_component="c00"), "c00"), (dict(partition="test"), "not a How2Sign val"), (dict(corpus="other"), "not a How2Sign record")])
def test_a_wrong_read_list_refuses_before_any_state(monkeypatch, tmp_path, change, match):
    h = Sym(monkeypatch, tmp_path)
    h.records[0].update(change)
    loaded = []
    monkeypatch.setattr(L, "load_states", lambda store, recs: loaded.append(1))
    with pytest.raises(SystemExit, match=match):
        h.read_forms("control")
    assert not loaded


def test_only_control_and_rest_can_be_named_and_no_option_names_another_list():
    assert F.resolve_forms("control") == ["v1__real_two_view", "v1__blank"] and F.resolve_forms("rest") == [f"mono__{f}" for f in F.BASE_FORMS]
    with pytest.raises(SystemExit, match="control or rest"):
        F.resolve_forms("mono__c20")
    for command in F.build_parser()._subparsers._group_actions[0].choices.values():
        flags = {o for a in command._actions for o in a.option_strings}
        assert not flags & {"--sentences-key", "--list", "--instrument", "--translator-run", "--checkpoint", "--tokenizer", "--partition"}


def test_each_form_is_read_by_its_instrument_and_a_smoke_stays_at_five(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", "--resume") == 0
    assert h.v1_reader.calls == ["real_two_view", "blank"] and h.mono_reader.calls == list(F.BASE_FORMS)
    form = json.loads((h.forms / "form-mono__nearest_monocular.json").read_text())
    assert form["form"] == "mono__nearest_monocular" and form["kind"] == "symmetric-ft-form" and np.asarray(form["nll"]).shape == (N, 5) and form["instrument"]["sha256"] == "a" * 64 and form["report_only"] is True
    with pytest.raises(SystemExit, match="at most 5"):
        h.main("read", "--forms", "control", "--out-dir", str(tmp_path / "x"), "--smoke-train", "6")
    with pytest.raises(SystemExit, match="write-once"):
        h.read_forms("rest")


def test_a_control_miss_voids_the_read(monkeypatch, tmp_path, capsys):
    h = Sym(monkeypatch, tmp_path)
    assert h.read_forms("control") == 0
    h.plant_canary()
    monkeypatch.setattr(P, "CANARY", dict(P.CANARY, real=P.CANARY["real"] + 1.))
    assert h.main("control", "--out-dir", str(h.forms), "--void-prefix", "sf/void") == 3
    assert "no retry" in capsys.readouterr().out and h.store.exists("sf/void/" + "c" * 64 + ".json")
    with pytest.raises(SystemExit, match="no passed control"):
        h.read_forms("rest", "--resume")


@pytest.mark.parametrize("gap,share,difference,case", [
    (dict(ci95=[.1, .2]), dict(point=.49), dict(ci95=[.01, .2]), F.LABEL_FIT),
    (dict(ci95=[.1, .2]), dict(point=.5), dict(ci95=[.01, .2]), "unresolved"),
    (dict(ci95=[.1, .2]), dict(point=.3), dict(ci95=[0., .2]), "unresolved"),
    (dict(ci95=[.1, .2]), dict(point=1.3), dict(ci95=[-.2, -.01]), F.LABEL_CONTENT),
    (dict(ci95=[.1, .2]), dict(point=1.1), dict(ci95=[-.2, 0.]), "unresolved"),
    (dict(ci95=[0., .2]), dict(point=.3), dict(ci95=[.1, .2]), "unresolved"),
    (dict(ci95=[.1, .2]), None, dict(ci95=[.1, .2]), "unresolved")])
def test_the_mirror_case_of_one_metric(gap, share, difference, case):
    assert F.mirror_case(gap, share, difference) == case


@pytest.mark.parametrize("difference,case", [
    (dict(point=-.2, ci95=[-.5, .1]), F.LABEL_NEAR), (dict(point=-.5, ci95=[-.5, 0.]), F.LABEL_NEAR), (dict(point=.3, ci95=[.1, .5]), F.LABEL_NEAR),
    (dict(point=-.51, ci95=[-.9, -.1]), F.LABEL_BELOW), (dict(point=-.51, ci95=[-.9, 0.]), "unresolved"), (dict(point=-.3, ci95=[-.6, 0.]), "unresolved")])
def test_the_peak_case_of_one_metric(difference, case):
    assert F.peak_case(difference, .5) == case


def test_a_mirrored_preference_reads_fit_and_a_peak_near_v1s(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path, levels=dict(nearest_two_view=(4, 2.5, .3), nearest_monocular=(15, 1., .6)))
    report = h.pipeline(two=(15, 1.), mono=(7, 2.5))
    assert report["alerts"] == [] and report["kind"] == "symmetric-ft-report" and report["report_only"] is True
    pref = report["preference"]
    assert pref["label"] == F.LABEL_FIT and pref["marks_free"]["likelihood"]["case"] == F.LABEL_FIT
    assert pref["marks_free"]["likelihood"]["two_view_share_of_monocular_gap"]["point"] == pytest.approx(.25, abs=1e-3)      # (3 - 2.5) / (3 - 1)
    assert report["own_path_peak"]["label"] == F.LABEL_NEAR and abs(report["own_path_peak"]["likelihood"]["difference"]["point"]) < 1e-6
    assert report["v1"]["likelihood"]["monocular_share_of_two_view_gap"]["point"] == pytest.approx(.25, abs=1e-3)
    assert report["clips"]["shared"] == N and report["clips"]["in_v1_training"] == 4 and report["by_training"]["in_v1_training"]["clips"] == 4 and report["by_training"]["outside_v1_training"]["own_path_peak"] is not None


def test_the_two_view_share_is_the_mirror_of_v1s(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path, levels=dict(nearest_two_view=(9, 2.8, .3), nearest_monocular=(13, 1., .6)))
    report = h.pipeline(two=(13, 1.), mono=(7, 2.5))
    row = report["preference"]["marks_free"]["likelihood"]
    assert row["two_view_share_of_monocular_gap"]["point"] == pytest.approx(.1, abs=1e-3) and row["monocular_share_of_two_view_gap"]["point"] == pytest.approx(10., abs=.01)


def test_still_preferring_two_view_reads_content_and_a_low_peak_reads_below(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path, levels=dict(nearest_two_view=(13, 1., .6), nearest_monocular=(7, 2.5, .3)))
    report = h.pipeline(two=(15, 1.), mono=(7, 2.5))
    assert report["preference"]["label"] == F.LABEL_CONTENT and report["own_path_peak"]["label"] == F.LABEL_BELOW


def test_margins_over_their_own_stills_stand_beside_the_shares_and_set_no_case(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    report = h.pipeline()
    row = report["margin"]["margin_over_still"]
    assert report["margin"]["report_only"] is True and "presence" in report["margin"]["note"] and "case" not in row
    assert row["two_view"]["point"] == pytest.approx(.3 - .1, abs=1e-5) and row["monocular"]["point"] == pytest.approx(.6 - .05, abs=1e-5) and row["monocular_minus_two_view"]["point"] == pytest.approx(.35, abs=1e-5) and row["two_view_share_of_monocular"]["point"] == pytest.approx(.2 / .55, abs=1e-3)
    assert report["margin"]["raw"]["still_monocular"]["point"] == pytest.approx(.05, abs=1e-5)


def test_each_still_holds_its_own_paths_middle_frame(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    kept, states, masks = S_form(h, "still_monocular")
    assert kept == list(range(N)) and np.array_equal(states[0].channels["left_hand"].values[0], states[0].channels["left_hand"].values[-1]) and len(states[0].timestamps) == 24
    kept, states, masks = S_form(h, "still_two_view")
    assert len(states[0].timestamps) == len(h.refs[0].timestamps) and states[0].channels["body"].valid.all()


def test_per_signer_rows_carry_no_label_and_groups_stay_aligned(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    report = h.pipeline(two=(15, 1.), mono=(7, 2.5))
    rows = report["per_signer"]
    assert set(rows) == {"c01", "c03", "c05"} and sum(r["clips"] for r in rows.values()) == N
    assert "case" not in json.dumps(rows) and rows["c01"]["v1"]["likelihood"]["monocular_share_of_two_view_gap"]["point"] == pytest.approx(.25, abs=.02)
    form = json.loads((h.forms / "form-mono__blank.json").read_text())
    picked = R.restrict(form, [form["signs"][7], form["signs"][2]])
    assert picked["groups"] == [form["groups"][7], form["groups"][2]] and picked["nll"][0] == form["nll"][7]


def test_a_signer_with_one_clip_has_a_count_alone(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", "--resume") == 0
    reads = F.load_reads(h.forms)
    assert F.signer_rows(reads, None, [reads["blank"]["signs"][0]]) == {reads["blank"]["groups"][0]: dict(clips=1)}


def test_the_selection_sample_is_split_out(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    report = h.pipeline()
    assert report["by_training"]["in_selection_sample"]["clips"] == 2 and report["by_training"]["in_v1_training"]["clips"] == 4 and report["by_training"]["outside_training_and_selection"]["clips"] == N - 6


def test_the_coverage_record_is_embedded_and_its_absence_raises_an_alert(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    report = h.pipeline()
    assert report["coverage"]["monocular_clips"] == dict(train=4, val=2) and report["alerts"] == []
    (h.forms.parent / "coverage.json").unlink()
    assert any("no coverage record" in a for a in h.merge("again.json")["alerts"])
    (h.forms.parent / "coverage.json").write_text(json.dumps(dict(code_digest="e" * 64)))
    assert any("no coverage record" in a for a in h.merge("third.json")["alerts"])


def test_metrics_that_disagree_read_unresolved(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path, levels=dict(nearest_two_view=(9, 1., .3), nearest_monocular=(13, 2.5, .6)))
    report = h.pipeline(two=(13, 1.), mono=(7, 2.5))
    assert report["preference"]["label"] == "unresolved"


def test_a_label_is_withheld_on_an_alert_and_check_exits_3(monkeypatch, tmp_path, capsys):
    h = Sym(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", "--resume") == 0
    h.plant_val_control(misses=("BLEU",))
    h.plant_v1_stored()
    report = h.merge()
    assert any("selection read" in a for a in report["alerts"]) and report["preference"]["label"] is None and report["own_path_peak"]["label"] is None
    assert h.main("check", "--report", str(tmp_path / "report.json")) == 3 and "ALERT" in capsys.readouterr().out


def test_no_selection_control_or_v1_off_the_plan_or_too_few_clips_raise_alerts(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", "--resume") == 0
    h.plant_v1_stored()
    monkeypatch.setattr(F, "PLAN", dict(F.PLAN, two_bleu=99., mono_likelihood=9.))
    monkeypatch.setattr(R, "SHARED_FLOOR", N + 1)
    alerts = h.merge()["alerts"]
    assert any("no selection control" in a for a in alerts) and any("BLEU" in a and "plan" in a for a in alerts) and any("likelihood" in a and "plan" in a for a in alerts) and any("shared clips" in a for a in alerts)


def test_a_recipe_difference_in_the_record_raises_an_alert(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    h.store.put(Q.TRAINED_RECORD, json.dumps(dict(checkpoint_key="k", sha256="a" * 64, recipe_differences={"args.lr": [1, 2]})).encode())
    assert any("v1's recipe" in a for a in h.pipeline()["alerts"])


def test_a_smoke_carries_no_label_and_no_stored_form(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path, declared=False, n=5)
    assert h.main("read", "--forms", "rest", "--out-dir", str(h.forms), "--smoke-train", "5") == 0
    h.plant_val_control()
    report = h.merge()
    assert report["preference"]["label"] is None and report["own_path_peak"] is None and report["v1"] is None and "a smoke carries no reading" in report["withheld"] and "SMOKE" in report["smoke"]


@pytest.mark.parametrize("field,value,match", [("code_digest", "e" * 64, "different code digests"), ("declared_read", False, "smoke mixed"), ("environment", dict(gpu="other"), "different environments"), ("form", "mono__blank", "holds")])
def test_forms_of_another_run_refuse_the_merge(monkeypatch, tmp_path, field, value, match):
    h = Sym(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", "--resume") == 0
    h.plant_val_control()
    h.plant_v1_stored()
    path = h.forms / "form-mono__real_monocular.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), **{field: value})))
    with pytest.raises(SystemExit, match=match):
        h.merge()


def test_stale_forms_a_missing_form_or_another_checkpoint_refuse_the_merge(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", "--resume") == 0
    h.plant_val_control()
    h.plant_v1_stored()
    monkeypatch.setattr(F, "code_digest", lambda: "e" * 64)
    with pytest.raises(SystemExit, match="a stale read"):
        h.merge()
    monkeypatch.setattr(F, "code_digest", lambda: "c" * 64)
    record = json.loads(h.store.get(Q.TRAINED_RECORD))
    h.store.put(Q.TRAINED_RECORD, json.dumps(dict(record, sha256="9" * 64)).encode())
    with pytest.raises(SystemExit, match="not read by the trained checkpoint"):
        h.merge()
    (h.store.root / Q.TRAINED_RECORD).unlink()
    with pytest.raises(SystemExit, match="trained.json is missing"):
        h.merge()
    (h.forms / "form-mono__blank.json").unlink()
    with pytest.raises(SystemExit, match="is missing"):
        h.merge()


def test_a_shared_clip_outside_v1s_stored_forms_refuses(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", "--resume") == 0
    h.plant_val_control()
    h.plant_v1_stored()
    table = dict(Q.STORED_V1)
    two = json.loads(h.store.get(f"{Q.PARTS_FORMS}/{table[F.TWO][0]}"))
    two["signs"] = [f"z{i}" for i in range(N)]
    raw = json.dumps(two).encode()
    h.store.put(f"{Q.PARTS_FORMS}/{table[F.TWO][0]}", raw)
    table[F.TWO] = (table[F.TWO][0], hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(Q, "STORED_V1", table)
    with pytest.raises(SystemExit, match="not in v1's stored forms"):
        h.merge()


def test_a_report_is_write_once_and_check_passes_a_clean_one(monkeypatch, tmp_path, capsys):
    h = Sym(monkeypatch, tmp_path)
    h.pipeline()
    path = tmp_path / "report.json"
    assert h.main("check", "--report", str(path)) == 0
    with pytest.raises(SystemExit, match="write-once"):
        h.merge()
    clean = json.loads(path.read_text())
    for change, word in ((dict(report_only=False), "report_only"), (dict(checks=dict(clean["checks"], control=None)), "passed control"), (dict(checks=dict(clean["checks"], blank_identical_to_real=1.)), "blank reads"), (dict(withheld=["x"]), "label beside")):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(dict(clean, **change)))
        assert h.main("check", "--report", str(bad)) == 3
        assert word in capsys.readouterr().out


def test_the_header_states_the_reading_the_thresholds_and_what_changes_before_any_read():
    head = F.__doc__
    for text in F.PRESTATED.values():
        assert text in head
    for text in ("Not independent", "may favour (a) and ``peak_near``", "signer split", "presence offset", "20 s cut", "0.5 BLEU and 0.052 nats", "19,182", "345", "strict subset", "exact record", "c20.json", "234 of D20", "unseeded", "What none of these can show", "--allow-resume"):
        assert text in head, text
    for text in ("Not independent", "signer split", "presence offset", "20 s cut"):
        assert any(text in v for v in F.DISCLOSURES.values()), text
    assert F.THRESHOLD == dict(bleu=.5, likelihood=.052) and F.SHARE_MIRROR == .5 and F.COVERAGE_FLOOR == .9 and F.EXIT_COVERAGE == 4


# ---- the job ---------------------------------------------------------------------------------------------------------------------------

def run_job(tmp_path, fail_when=None, code=7, **env):
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    pattern = "*" + "*".join(f'"{x}"' for x in fail_when.split("*")) + "*" if fail_when else ""
    match = f'case "$*" in {pattern}) echo "planted failure" >&2; exit {code};; esac' if fail_when else ""
    (bin_dir / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\n{match}\necho "ran $1 $2"\nexit 0\n')
    (bin_dir / "uv").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "timeout").write_text(f'#!/bin/bash\necho "timeout $*" >> {tmp_path / "timeouts.log"}\nshift 3\nexec "$@"\n')
    for name in ("python", "uv", "timeout"):
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    environment = dict(PATH=f"{bin_dir}:/usr/bin:/bin", HOME=str(tmp_path), VENV_BIN=str(bin_dir), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path), WORK=str(tmp_path / "work"), **env)
    done = subprocess.run(["bash", str(JOB)], env=environment, capture_output=True, text=True, timeout=60)
    return done, log.read_text() if log.exists() else ""


def done_line(result) -> int:
    lines = [x for x in result.stdout.splitlines() if x.startswith("SYM-FT-DONE ")]
    assert len(lines) == 1 and lines[0] == result.stdout.strip().splitlines()[-1]
    return int(lines[0].split()[1])


def commands(calls):
    return [F.build_parser().parse_args(x.split()[1:]).command for x in calls.splitlines()]


def test_the_runner_sees_no_credentials():
    assert not [k for k in os.environ if k.startswith(("R2_", "AWS_", "RUNPOD_"))]


def test_the_job_checks_coverage_smokes_trains_then_controls_before_reading(tmp_path):
    result, calls = run_job(tmp_path)
    assert result.returncode == 0 and done_line(result) == 0
    assert commands(calls) == ["recipe", "coverage", "build", "train", "trained", "val-control", "read", "control", "read", "merge", "upload", "check", "build", "train", "trained", "preflight", "val-control", "read", "control", "read", "merge", "upload", "check", "upload-log"]
    lines = calls.splitlines()
    assert "--smoke" in lines[2] and "--smoke" in lines[3] and "--smoke" not in lines[12] and "--smoke" not in lines[13] and "--smoke-train 5" in lines[6] and "--smoke-train" not in lines[17]
    assert "training-runs/back-translator-kp2d-uwb-ft-mono-v1/read/report.json" in lines[15] and "--forms control" in lines[17] and "--forms rest" in lines[19]


def test_the_trainer_runs_under_a_timeout_and_the_full_build_on_one_blas_thread(tmp_path):
    _, _ = run_job(tmp_path)
    log = (tmp_path / "timeouts.log").read_text()
    assert "-k 60 3600 python scripts/symmetric_ft.py train --smoke" in log and "-k 60 28800 python scripts/symmetric_ft.py train" in log
    text = JOB.read_text()
    assert 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 step python $T build --workers' in text and 'coverage --out-dir "$R"' in text


def test_the_smoke_only_mode_stops_before_the_shards(tmp_path):
    result, calls = run_job(tmp_path, SMOKE="1")
    assert result.returncode == 0 and "SMOKE done" in result.stdout and sum(x.startswith("scripts/symmetric_ft.py build") for x in calls.splitlines()) == 1


def test_too_few_clips_stop_with_4_before_any_training(tmp_path):
    result, calls = run_job(tmp_path, fail_when="coverage", code=4)
    assert result.returncode == 4 and done_line(result) == 4 and "this part stops" in result.stdout and not [x for x in calls.splitlines() if x.startswith("scripts/symmetric_ft.py train")]


def test_a_build_under_the_floor_stops_with_4(tmp_path):
    result, calls = run_job(tmp_path, fail_when="build --workers 12", code=4)
    assert result.returncode == 4 and done_line(result) == 4 and not [x for x in calls.splitlines() if x.startswith("scripts/symmetric_ft.py train") and "--smoke" not in x]


def test_allow_resume_reaches_the_trainer_only_when_given(tmp_path):
    _, calls = run_job(tmp_path, ALLOW_RESUME="1")
    assert [x for x in calls.splitlines() if x.startswith("scripts/symmetric_ft.py train") and "--allow-resume" in x and "--smoke" not in x]


def test_no_resume_unless_given(tmp_path):
    _, calls = run_job(tmp_path)
    assert "--allow-resume" not in calls


@pytest.mark.parametrize("fail,status,word", [("recipe", 1, "header's recipe"), ("train --smoke", 2, "smoke train failed"), ("trained --smoke", 2, "smoke trained"), ("trained --store", 0, ""), ("preflight", 1, "preflight refused"), ("--forms rest*--resume", 1, "new instrument failed")])
def test_a_failing_step_stops_the_job_with_its_status(tmp_path, fail, status, word):
    result, calls = run_job(tmp_path, fail_when=fail)
    assert result.returncode == status and done_line(result) == status and word in result.stdout
    assert ("upload --partial" in calls) is (status != 0)


def test_a_void_control_stops_with_3_and_reads_no_new_form(tmp_path):
    result, calls = run_job(tmp_path, fail_when="symmetric_ft.py control --out-dir*work/d20/forms", code=3)
    assert result.returncode == 3 and done_line(result) == 3 and "do not retry" in result.stdout and not [x for x in calls.splitlines() if "--forms rest" in x and "--resume" in x]


def test_a_missing_repo_prints_its_done_line(tmp_path):
    done = subprocess.run(["bash", str(JOB)], env=dict(PATH="/usr/bin:/bin", HOME=str(tmp_path), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path / "gone")), capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and done.stdout.strip().endswith("SYM-FT-DONE 1")


def test_the_job_passes_every_step_through_pipestatus_and_names_no_other_list():
    text = JOB.read_text()
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0
    code = [line.split(" #")[0] for line in text.splitlines() if not line.startswith("#")]
    assert [line for line in code if re.search(r"(?<!\|)\|(?!\|)", line)] == ['  "$@" 2>&1 | tee -a "$LOG"'] and "${PIPESTATUS[@]}" in text and "set -uo pipefail" in text and "transformers==4.57.6" in text
    for word in ("c20", "C20", "round15_compose_eval", "composed", "create-pod", "runpodctl"):
        assert word not in "\n".join(code), word
    assert "L40S" in text and "D=training-runs/back-translator-kp2d-uwb-ft-mono-v1" in text


def test_the_cut_counts_each_path_at_its_boundary(built, monkeypatch):
    built.plant_v1_lists(["s0", "s1"], ["s4", "s5"])
    monkeypatch.setattr(F, "MAX_FRAMES", 24)     # 24 frames sits exactly on the cut: not over it
    monkeypatch.setattr(F, "COVERAGE_FLOOR", 0.)
    assert built.main("build", "--workers", "1") == 0
    assert json.loads(built.store.get(f"{F.INDEX_PREFIX}/index.json"))["cut_at_20s"]["train"]["monocular"] == 0


def test_the_cut_counts_two_views_own_frames_from_its_list(built, monkeypatch):
    built.plant_v1_lists(["s0", "s1"], ["s4", "s5"])
    monkeypatch.setattr(F, "MAX_FRAMES", 19)     # v1's list says 20 frames each
    monkeypatch.setattr(F, "COVERAGE_FLOOR", 0.)
    assert built.main("build", "--workers", "1") == 0
    assert json.loads(built.store.get(f"{F.INDEX_PREFIX}/index.json"))["cut_at_20s"] == dict(train=dict(two_view=2, monocular=2), val=dict(two_view=2, monocular=2))


def test_a_resumed_run_says_so_in_its_record(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    trained_world(h)
    assert h.main("trained") == 0
    assert json.loads(h.store.get(Q.TRAINED_RECORD))["resumed"] == []
    trained_world(h)
    h.store.put(f"{Q.MONO_PREFIX}/resumed-20261102T000000.json", b"{}")
    assert h.main("trained") == 0
    assert json.loads(h.store.get(Q.TRAINED_RECORD))["resumed"] == [f"{Q.MONO_PREFIX}/resumed-20261102T000000.json"]


def test_a_signer_with_two_clips_has_its_rows(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_forms("rest", "--resume") == 0
    reads = F.load_reads(h.forms)
    groups = reads["blank"]["groups"]
    pair = [s for s, g in zip(reads["blank"]["signs"], groups) if g == groups[0]][:2]
    row = F.signer_rows(reads, None, pair)[groups[0]]
    assert row["clips"] == 2 and "new_instrument" in row


def test_the_margin_shares_interval_holds_its_point(monkeypatch, tmp_path):
    h = Sym(monkeypatch, tmp_path)
    share = h.pipeline()["margin"]["margin_over_still"]["two_view_share_of_monocular"]
    assert share["ci95"][0] <= share["point"] <= share["ci95"][1]
