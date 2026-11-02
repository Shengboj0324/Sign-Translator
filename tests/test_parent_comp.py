"""The parent composition read on synthetic data: the draws' pins, what the composed exports must rest on, the control, the margin arithmetic over draws and pods, what is refused, and the job."""
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
import parent_comp as C  # noqa: E402
import parent_lib as Q  # noqa: E402
import path_parts as PP  # noqa: E402
import path_read as R  # noqa: E402
import round15_compose_eval as R15  # noqa: E402
import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v1 as T  # noqa: E402
from evaluate_text_to_motion import blanked  # noqa: E402
from signtranslator.eval import s0_metrics as M  # noqa: E402
from tests.test_generation_signing_space import signer_state  # noqa: E402
from tests.test_path_read import Harness, N, laid_out, TWO_VIEW, tiny_tokenizer  # noqa: E402

JOB = ROOT / "runpod/fleet/parent_comp.sh"
PARENT_INFO = dict(run=PP.UNFITTED_RUN, checkpoint_key=PP.UNFITTED_KEY, sha256=PP.UNFITTED_SHA, kind="keypoints")
# per form: (words said, own NLL, margin over the mismatched)
ANCHOR_LEVELS = dict(real_two_view=(14, 1.1, .5), blank=(0, 3., .02), nearest_two_view=(15, 1., .6), still_two_view=(1, 2.9, .1))
EMPTY = 2      # this clip has no selected word, in every draw


def level_of(name):
    if name in ANCHOR_LEVELS:
        return ANCHOR_LEVELS[name]
    k, kind = C.draw_form(name)
    if kind == "blank":
        return (0, 3., .02)
    margin = dict(comp=.1 + .03 * k, still=.05, comp_true=.2 + .03 * k, still_true=.1)[kind]
    return (6 + k, 2., margin)


class FormReader:
    info = PARENT_INFO

    def __init__(self) -> None:
        self.form, self.calls = None, []

    def read(self, states, texts, candidates, group=2):
        self.calls.append((self.form, group, candidates.copy()))
        k, nll, margin = level_of(self.form)
        rows, said = [], []
        for s, row in zip(states, candidates):
            if self.form in ANCHOR_LEVELS or s.channels["body"].valid.any():
                rows.append([nll] + [nll + margin] * (candidates.shape[1] - 1))
                said.append(" ".join(texts[row[0]].split()[:k]) if k else "z y x w")
            else:
                rows.append([3.] + [3.02] * (candidates.shape[1] - 1))
                said.append("z y x w")
        return dict(hypotheses=said, nll=np.array(rows))


def settings_of(k, clips=N):
    pins = C.DRAW_INPUTS[k]
    return dict(mix_sha256=L.EVAL_MIX_SHA, sentences_sha256=L.D20_SHA, clips=clips, selection="stage1", timing="sign-fitted", expect_model_sha256=C.MODEL_SHA, variants="core_crossfade", word_cache_key=C.cache_key(k), word_cache_sha256=pins["cache"], word_cache_n=16, core_margin=0., predicted_duration=True, expect_duration_model=C.DURATION,
                vocabulary_extra_top=10, vocabulary_extra_sha256=T.TOKENS_SHA, word_cache_extra_key=C.extra_key(k), word_cache_extra_sha256=pins["extra"], export_only=True, export_forms=",".join(C.EXPORT_FORMS))


def top10_report(k, hypotheses=None, n=N, **change):
    pins = C.DRAW_INPUTS[k]
    refs = [f"w{i} x{i} y{i} z{i} a{i} b{i}" for i in range(n)]
    report = dict(signs=[f"s{i}" for i in range(n)], references=refs, settings=dict(core_margin=0.0, clips=L.D20_CLIPS, round20=dict(word_cache=dict(key=C.cache_key(k), sha256=pins["cache"], draw=k), word_cache_extra=dict(key=C.extra_key(k), sha256=pins["extra"]), vocabulary_extra=dict(top=10, sha256=T.TOKENS_SHA), predicted_duration=True, sentences=dict(sha256=L.D20_SHA))),
                  forms={C.PREDICTED: dict(hypotheses=hypotheses or list(refs)), "real": dict(hypotheses=list(refs)), "blank@predicted": dict(hypotheses=["z y x w"] * n)})
    report.update(change)
    return report


class Comp(Harness):
    def __init__(self, monkeypatch, tmp_path, *, draws=C.DRAWS, **kw):
        super().__init__(monkeypatch, tmp_path, **kw)
        self.reader_parent = FormReader()
        monkeypatch.setattr(PP, "load_unfitted", lambda store, args: self.reader_parent)
        real_read_one = C.read_one

        def tracked(name, *a, **k):
            self.reader_parent.form = name
            return real_read_one(name, *a, **k)
        monkeypatch.setattr(C, "read_one", tracked)
        monkeypatch.setattr(L, "D20_CLIPS", self.n)
        monkeypatch.setattr(C, "code_digest", lambda: "c" * 64)
        monkeypatch.setattr(P, "environment", lambda: dict(gpu="test"))
        self.plant_training([r["sign"] for r in self.records[:4]])
        self.pins = {}
        for k in draws:
            self.plant_export(k)
        self.plant_top10s()

    def plant_training(self, signs):
        meta = json.dumps([dict(clip_id=s, partition="train") for s in signs]).encode()
        self.store.put("fk/train/shard-00000.json", meta)
        index = json.dumps(dict(parts=[dict(partition="train", shards=[dict(key="fk/train/shard-00000.npz", meta_sha256=hashlib.sha256(meta).hexdigest())])])).encode()
        self.store.put(Q.FK_INDEX, index)
        self.monkeypatch.setattr(Q, "FK_INDEX_SHA", hashlib.sha256(index).hexdigest())

    def plant_export(self, k, prefix="comp", settings=None, skip=(), draw=None):
        rows = []
        for i, record in enumerate(self.records):
            if i in skip:
                continue
            for form, frames in ((C.PREDICTED, 24 + i), (C.TRUE_LENGTH, len(self.refs[i].timestamps))):
                empty = i == EMPTY
                state = laid_out(signer_state(200 + i, frames=frames, amplitude=.4), TWO_VIEW)
                if empty:
                    state = blanked([state])[0]
                path = self.tmp_path / f"c-{k}-{i}-{frames}-{form.replace('@', '-')}-{len(list(self.tmp_path.glob('c-*')))}.npz"
                sha = state.save(path)
                key = f"{prefix}/draw{k}/states/{i}-{form.replace('@', '-')}.npz"
                self.store.put(key, path.read_bytes())
                rows.append(dict(clip=i, sign=record["sign"], text=record["text"], form=form, empty=empty, state_key=key, state_sha256=sha, frames=frames, reference_sha256=record["continuous"]["state_sha256"], words=[] if empty else [dict(word="w")] * (1 + i % 3)))
        index = dict(kind="round15-composed-states", settings=settings or settings_of(k, self.n), word_cache=dict(draw=k if draw is None else draw), states=rows)
        self.store.put(C.export_index_key(prefix, k), json.dumps(index).encode())

    def plant_top10s(self, **change):
        inputs = {}
        for k in C.DRAWS:
            report = top10_report(k, n=self.n, **change)
            report["signs"], report["references"] = [r["sign"] for r in self.records], [r["text"] for r in self.records]
            report["forms"]["real"]["hypotheses"] = [r["text"] for r in self.records]
            report["forms"][C.PREDICTED]["hypotheses"] = [" ".join(r["text"].split()[:7]) for r in self.records]
            report["forms"]["blank@predicted"]["hypotheses"] = ["z y x w"] * self.n
            report["likelihood"] = dict(per_form={C.PREDICTED: [-2.] * self.n, "blank@predicted": [-3.] * (self.n - 1) + [None], "real": [-1.] * self.n})
            report["per_clip"] = [dict(sign=r["sign"], plans={} if i == EMPTY else dict(core_crossfade=dict(frames=len(self.refs[i].timestamps), predicted=dict(frames=24 + i)))) for i, r in enumerate(self.records)]
            raw = json.dumps(report).encode()
            self.store.put(C.top10_key(k), raw)
            inputs[k] = dict(C.DRAW_INPUTS[k], top10=hashlib.sha256(raw).hexdigest())
        self.monkeypatch.setattr(C, "DRAW_INPUTS", inputs)
        summary = json.dumps(dict(forms=dict(predicted=dict(gap_share=dict(top10=[.3 + .01 * k for k in range(8)]), top10_bleu=[2.2 + .05 * k for k in range(8)])))).encode()
        self.store.put(C.A5_SUMMARY, summary)
        self.monkeypatch.setattr(C, "A5_SUMMARY_SHA", hashlib.sha256(summary).hexdigest())

    def main(self, *argv):
        return C.main([argv[0], *(() if argv[0] == "check" else ("--store", f"dir:{self.tmp_path / 's'}")), *argv[1:]])

    def plant_stored_anchors(self, change=None):
        table = {}
        for name in C.ANCHORS:
            stored = dict(json.loads((self.forms / f"form-{name}.json").read_text()), **(change or {}).get(name, {}))
            raw = json.dumps(stored).encode()
            self.store.put(f"{Q.PARTS_FORMS}/s-{name}.json", raw)
            table[name] = (f"s-{name}.json", hashlib.sha256(raw).hexdigest())
        self.monkeypatch.setattr(Q, "STORED_PARENT", table)

    def anchors(self):
        return self.main("read", "--forms", "control", "--out-dir", str(self.forms))

    def control(self):
        return self.main("control", "--out-dir", str(self.forms), "--void-prefix", "pc/void")

    def to_control(self):
        assert self.anchors() == 0
        self.plant_stored_anchors()
        assert self.control() == 0

    def read_draws(self, draws="1,2,3,4,5,6,7,8", *extra):
        return self.main("read", "--draws", draws, "--composed", "comp", "--out-dir", str(self.forms), *extra)

    def fidelity(self, k):
        monkey = self.monkeypatch
        import score_back_translation
        monkey.setattr(score_back_translation, "translate", lambda model, kind, states: ["w x y z a b"] * len(states))
        monkey.setattr(P, "load_reader", lambda store, args: type("V", (), dict(model=None, kind="keypoints", info=dict(sha256=L.TRANSLATOR_SHA)))())
        return self.main("fidelity", "--draw", str(k), "--clips", str(self.n), "--composed", "comp", "--out-dir", str(self.forms))

    def plant_fidelity(self, k, identical=32, frames_off=0):
        (self.forms / f"fidelity-d{k}.json").write_text(json.dumps(dict(draw=k, clips=32, identical=identical, frames=dict(compared=8, off=frames_off, first_off=None), code_digest="c" * 64)))

    def upload_pod(self, pod="a", forms=None):
        return self.main("upload", "--out-dir", str(forms or self.forms), "--prefix", f"pods/{pod}/forms")

    def merge(self, pods="a", name="report.json"):
        out = self.tmp_path / name
        assert self.main("merge", "--pods", pods, "--prefix", "pods", "--output-file", str(out)) == 0
        return json.loads(out.read_text())

    def plant_plan(self):
        forms = {n: json.loads((self.forms / f"form-{n}.json").read_text()) for n in C.ANCHORS}
        self.monkeypatch.setattr(C, "PLAN", dict(nearest_margin=.5, nearest_bleu=M.bleu_ci_fast(forms["nearest_two_view"]["hypotheses"], forms["nearest_two_view"]["references"])[0], real_bleu=M.bleu_ci_fast(forms["real_two_view"]["hypotheses"], forms["real_two_view"]["references"])[0], blank_bleu=M.bleu_ci_fast(forms["blank"]["hypotheses"], forms["blank"]["references"])[0], still_bleu=M.bleu_ci_fast(forms["still_two_view"]["hypotheses"], forms["still_two_view"]["references"])[0], v1_share=.335, v1_bleu=2.375))

    def pipeline(self):
        self.to_control()
        assert self.read_draws() == 0
        for k in C.DRAWS:
            self.plant_fidelity(k)
        assert self.upload_pod() == 0
        self.plant_plan()
        return self.merge()


# ---- pins and inputs --------------------------------------------------------------------------------------------------------------------

def test_the_draws_are_the_eight_of_a5_each_pinned_and_none_shared():
    assert C.DRAWS == tuple(range(1, 9)) and set(C.DRAW_INPUTS) == set(C.DRAWS)
    for field in ("top10", "cache", "extra"):
        hashes = [C.DRAW_INPUTS[k][field] for k in C.DRAWS]
        assert len(set(hashes)) == 8 and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)
    assert C.top10_key(3).endswith("round20/comp-a5/draw3-attempt1-top10.json") and C.cache_key(3).endswith("comp-a5/draw3-cache/attempt1/index.json") and C.extra_key(3).endswith("comp-a5/draw3-extra/attempt1/index.json")
    assert C.A5_SUMMARY_SHA.startswith("4f3d1f38") and C.MODEL_SHA.startswith("42389ca4") and C.PREDICTED == "core_crossfade@predicted" and C.TRUE_LENGTH == "core_crossfade"


@pytest.mark.parametrize("k", [0, 9, -1])
def test_a_draw_outside_the_eight_refuses(k):
    with pytest.raises(SystemExit, match="pinned draws"):
        C.check_draw(k)
    with pytest.raises(SystemExit, match="pinned draws"):
        C.export_argv(k, 5, "p", "l", "s", "o")


def test_a_stored_report_rests_on_its_own_caches():
    C.check_top10(top10_report(2), 2)


@pytest.mark.parametrize("path,value,word", [(("word_cache", "sha256"), C.DRAW_INPUTS[1]["cache"], "cache hash"), (("word_cache", "key"), C.cache_key(1), "cache key"), (("word_cache", "draw"), 1, "cache draw"), (("word_cache_extra", "sha256"), C.DRAW_INPUTS[1]["extra"], "extra hash"), (("word_cache_extra", "key"), C.extra_key(1), "extra key"), (("vocabulary_extra", "top"), 30, "arm"), (("vocabulary_extra", "sha256"), "0" * 64, "arm"), (("sentences", "sha256"), "0" * 64, "sentences")])
def test_mismatched_draw_caches_or_another_arm_refuse(path, value, word):
    report = top10_report(2)
    report["settings"]["round20"][path[0]][path[1]] = value
    with pytest.raises(SystemExit, match=word):
        C.check_top10(report, 2)


@pytest.mark.parametrize("field,value,word", [("core_margin", .1, "margin"), ("clips", 5, "clips")])
def test_another_margin_or_part_of_d20_refuses(field, value, word):
    report = top10_report(2)
    report["settings"][field] = value
    with pytest.raises(SystemExit, match=word):
        C.check_top10(report, 2)


def test_a_report_of_true_timing_only_refuses():
    report = top10_report(2)
    report["settings"]["round20"]["predicted_duration"] = False
    with pytest.raises(SystemExit, match="predicted"):
        C.check_top10(report, 2)


def test_a_stale_stored_report_on_other_bytes_refuses(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=())
    h.store.put(C.top10_key(1), json.dumps(top10_report(1)).encode() + b" ")
    with pytest.raises(SystemExit, match="SHA-256"):
        C.load_top10(h.store, 1)


def test_the_inputs_step_prints_hashes_and_no_score(monkeypatch, tmp_path, capsys):
    h = Comp(monkeypatch, tmp_path, draws=())
    assert h.anchors() == 0
    h.plant_stored_anchors()
    capsys.readouterr()
    assert h.main("inputs") == 0
    out = capsys.readouterr().out
    assert "summary" in out and "bleu" not in out and "hypotheses" not in out and "gap_share" not in out
    h.store.put(C.A5_SUMMARY, b"{}")
    with pytest.raises(SystemExit, match="SHA-256"):
        h.main("inputs")


# ---- the export -----------------------------------------------------------------------------------------------------------------------

def parsed_export(monkeypatch, tmp_path, k=4, clips=500):
    seen = {}
    monkeypatch.setattr(R15, "run", lambda args: seen.update(vars(args)) or dict(done=True))
    listing = tmp_path / "list.json"
    listing.write_text("{}")
    argv = C.export_argv(k, clips, "pre/fix", str(listing), "0" * 64, str(tmp_path / "out.json"))
    assert R15.main([*argv, "--store", f"dir:{tmp_path}"]) == 0
    return seen


def test_the_export_arguments_are_accepted_and_are_the_top10_phase(monkeypatch, tmp_path):
    args = parsed_export(monkeypatch, tmp_path)
    pins = C.DRAW_INPUTS[4]
    assert args["word_cache_key"] == C.cache_key(4) and args["word_cache_sha256"] == pins["cache"] and args["word_cache_extra_key"] == C.extra_key(4) and args["word_cache_extra_sha256"] == pins["extra"] and args["word_cache_n"] == 16
    assert args["vocabulary_extra_top"] == 10 and args["vocabulary_extra_sha256"] == T.TOKENS_SHA and args["core_margin"] == 0. and args["predicted_duration"] is True and args["expect_duration_model"] == C.DURATION and args["expect_model_sha256"] == C.MODEL_SHA
    assert args["export_only"] is True and args["export_forms"] == "core_crossfade@predicted,core_crossfade" and args["likelihood"] is False and args["sentences_key"] == L.D20_KEY and args["sentences_sha256"] == L.D20_SHA and args["clips"] == 500 and args["variants"] == "core_crossfade" and args["selection"] == "stage1" and args["timing"] == "sign-fitted" and args["device"] == "cpu" and args["recognizer_run"] == "sign-recognizer-v2"


def test_the_exports_settings_match_what_the_arguments_store(monkeypatch, tmp_path):
    args = parsed_export(monkeypatch, tmp_path, k=4, clips=7)
    assert C.export_differences(dict(settings=args), 4, 7) == {}


@pytest.mark.parametrize("field,value", [("word_cache_sha256", C.DRAW_INPUTS[1]["cache"]), ("word_cache_key", C.cache_key(1)), ("word_cache_extra_sha256", C.DRAW_INPUTS[1]["extra"]), ("vocabulary_extra_top", 30), ("core_margin", .1), ("sentences_sha256", "0" * 64), ("predicted_duration", False), ("export_forms", C.PREDICTED)])
def test_an_export_of_mismatched_settings_refuses(field, value):
    index = dict(kind="round15-composed-states", settings=dict(settings_of(2), **{field: value}), word_cache=dict(draw=2))
    with pytest.raises(SystemExit, match=field):
        C.check_export(index, 2, N)


def test_an_export_of_another_kind_or_another_draws_cache_refuses():
    with pytest.raises(SystemExit, match="not a composed-state"):
        C.check_export(dict(kind="other"), 2, N)
    with pytest.raises(SystemExit, match="word cache is draw 3"):
        C.check_export(dict(kind="round15-composed-states", settings=settings_of(2), word_cache=dict(draw=3)), 2, N)
    C.check_export(dict(kind="round15-composed-states", settings=settings_of(2), word_cache=dict(draw=2)), 2, N)


def test_a_stored_export_is_left_and_a_mismatched_one_refuses_unrun(monkeypatch, tmp_path, capsys):
    h = Comp(monkeypatch, tmp_path, draws=(2,))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    assert h.main("export", "--draw", "2", "--clips", str(N), "--work", str(tmp_path / "e"), "--prefix", "comp") == 0 and '"stored": true' in capsys.readouterr().out
    h.plant_export(2, settings=dict(settings_of(2), word_cache_sha256="0" * 64))
    with pytest.raises(SystemExit, match="word_cache_sha256"):
        h.main("export", "--draw", "2", "--clips", str(N), "--work", str(tmp_path / "e"), "--prefix", "comp")


def test_an_export_that_stops_or_stores_other_settings_refuses(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=())
    records = h.records
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **k: calls.append(argv) or type("D", (), dict(returncode=7))())
    with pytest.raises(SystemExit, match="stopped with 7"):
        h.main("export", "--draw", "2", "--clips", "3", "--work", str(tmp_path / "e"), "--prefix", "comp")
    listed = json.loads(Path(calls[0][calls[0].index("--export-sentences-file") + 1]).read_text())
    assert listed["signs"] == [r["sign"] for r in records[:3]] and calls[0][calls[0].index("--export-sentences-sha256") + 1] == hashlib.sha256(Path(calls[0][calls[0].index("--export-sentences-file") + 1]).read_bytes()).hexdigest()
    assert calls[0][calls[0].index("--export-prefix") + 1] == "comp/draw2"


def test_a_c00_record_stops_the_export_before_it_runs(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=())
    h.records[0]["partition"] = "test"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    with pytest.raises(SystemExit, match="not a How2Sign val"):
        h.main("export", "--draw", "2", "--clips", "3", "--work", str(tmp_path / "e"), "--prefix", "comp")


def test_no_export_refuses(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=())
    with pytest.raises(SystemExit, match="no export of draw 3"):
        C.load_export(h.store, "comp", 3, h.records, True)


def test_a_clip_missing_from_an_export_refuses(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    h.plant_export(1, skip=(4,))
    with pytest.raises(SystemExit, match="missing from the export"):
        C.load_export(h.store, "comp", 1, h.records, True)
    with pytest.raises(SystemExit, match="missing from the export"):
        C.load_export(h.store, "comp", 1, h.records, False)


def test_a_row_of_another_clip_refuses(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    h.records[3]["text"] = "another sentence"
    with pytest.raises(SystemExit, match="another clip's"):
        C.load_export(h.store, "comp", 1, h.records, True)
    h = Comp(monkeypatch, tmp_path / "b", draws=(1,)) if (tmp_path / "b").mkdir() is None else None
    h.records[3]["continuous"]["state_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="another clip's"):
        C.load_export(h.store, "comp", 1, h.records, True)


def test_the_rows_come_back_in_list_order(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    order = list(reversed(h.records))
    rows = C.load_export(h.store, "comp", 1, order, True)
    assert [r["sign"] for r in rows[C.PREDICTED]] == [r["sign"] for r in order] and {r["form"] for r in rows[C.TRUE_LENGTH]} == {C.TRUE_LENGTH}
    smoke = C.load_export(h.store, "comp", 1, h.records, False)
    assert [r["clip"] for r in smoke[C.PREDICTED]] == list(range(N))


def test_a_compositions_still_holds_its_own_middle_frame_at_its_own_length_and_an_empty_clip_stays_blank():
    tokenizer = tiny_tokenizer()
    state = laid_out(signer_state(5, frames=30, amplitude=.4), TWO_VIEW)
    still = C.still_of(state, dict(empty=False), tokenizer)
    assert len(still.timestamps) == 30 and np.array_equal(still.channels["left_hand"].values[0], still.channels["left_hand"].values[-1]) and still.channels["body"].valid.all()
    empty = blanked([state])[0]
    assert C.still_of(empty, dict(empty=True), tokenizer) is empty


@pytest.mark.parametrize("name", ["d9_comp", "d1_c20", "real", "d1", "core_crossfade"])
def test_an_unknown_form_refuses(name):
    with pytest.raises(SystemExit, match="unknown form"):
        C.draw_form(name)


def test_a_draws_forms_are_its_four():
    assert C.forms_of_draw(3) == ["d3_comp", "d3_still", "d3_comp_true", "d3_still_true", "d3_blank"] and C.draw_form("d3_comp_true") == (3, "comp_true") and C.draw_form("d3_blank") == (3, "blank")
    assert C.KINDS == dict(comp=(C.PREDICTED, "as_composed"), still=(C.PREDICTED, "still"), comp_true=(C.TRUE_LENGTH, "as_composed"), still_true=(C.TRUE_LENGTH, "still"), blank=(C.PREDICTED, "blank"))


# ---- reads and controls ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("change,match", [(dict(signer_component="c00"), "c00"), (dict(partition="test"), "not a How2Sign val"), (dict(partition="train"), "not a How2Sign val"), (dict(corpus="other"), "not a How2Sign record")])
def test_a_wrong_partition_or_c00_refuses_before_any_state(monkeypatch, tmp_path, change, match):
    h = Comp(monkeypatch, tmp_path, draws=())
    h.records[0].update(change)
    loaded = []
    monkeypatch.setattr(L, "load_states", lambda store, recs: loaded.append(1))
    with pytest.raises(SystemExit, match=match):
        h.anchors()
    assert not loaded


def test_c20_cannot_be_named():
    for command in C.build_parser()._subparsers._group_actions[0].choices.values():
        flags = {o for a in command._actions for o in a.option_strings}
        assert not flags & {"--sentences-key", "--list", "--instrument", "--translator-run", "--checkpoint", "--tokenizer", "--word-cache-key"}
    text = (ROOT / "scripts/parent_comp.py").read_text().split('"""', 2)[2]
    assert "c20" not in text.lower() and "test_rgb_front" not in text


def test_a_smoke_is_five_clips_at_most_and_a_read_takes_one_kind(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=(), declared=False, n=5)
    with pytest.raises(SystemExit, match="at most 5"):
        h.main("read", "--forms", "control", "--out-dir", str(h.forms), "--smoke-train", "6")
    with pytest.raises(SystemExit, match="exactly one"):
        h.main("read", "--out-dir", str(h.forms))
    with pytest.raises(SystemExit, match="exactly one"):
        h.main("read", "--forms", "control", "--draws", "1", "--out-dir", str(h.forms))
    with pytest.raises(SystemExit, match="only control"):
        h.main("read", "--forms", "c20", "--out-dir", str(h.forms))
    with pytest.raises(SystemExit, match="pinned draws"):
        h.main("read", "--draws", "9", "--out-dir", str(h.forms))


def test_a_composition_needs_a_passed_control_before_it_loads_a_state(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    loaded = []
    monkeypatch.setattr(L, "load_states", lambda store, recs: loaded.append(1))
    with pytest.raises(SystemExit, match="no passed control"):
        h.read_draws("1")
    assert not loaded


@pytest.mark.parametrize("change", [dict(reproduced=False), dict(code_digest="d" * 64), dict(environment=dict(gpu="other"))])
def test_a_failed_or_stale_control_refuses_a_composition(monkeypatch, tmp_path, change):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    h.to_control()
    path = h.forms / "control.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), **change)))
    with pytest.raises(SystemExit, match="no passed control"):
        h.read_draws("1")


@pytest.mark.parametrize("name,field,value,word", [("nearest_two_view", "hypotheses", ["other"] * N, "hypotheses"), ("still_two_view", "nll", [[9.] * 5] * N, "NLL"), ("blank", "signs", [f"z{i}" for i in range(N)], "other clips")])
def test_an_anchor_missed_voids_the_read(monkeypatch, tmp_path, capsys, name, field, value, word):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    assert h.anchors() == 0
    h.plant_stored_anchors({name: {field: value}})
    assert h.control() == 3
    out = capsys.readouterr().out
    assert word in out and "no composition follows" in out and h.store.exists("pc/void/" + "c" * 64 + ".json")
    with pytest.raises(SystemExit, match="no passed control"):
        h.read_draws("1")
    with pytest.raises(SystemExit, match="this code's read is void"):
        h.main("preflight", "--out-dir", str(tmp_path / "w2"), "--final-key", "pc/report.json", "--void-prefix", "pc/void")


def test_a_stored_anchor_on_other_bytes_exits_1_with_no_control_and_no_void(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    assert h.anchors() == 0
    h.plant_stored_anchors()
    table = dict(Q.STORED_PARENT, still_two_view=(Q.STORED_PARENT["still_two_view"][0], "0" * 64))
    monkeypatch.setattr(Q, "STORED_PARENT", table)
    with pytest.raises(SystemExit, match="SHA-256"):
        h.control()
    assert not h.store.exists("pc/void/" + "c" * 64 + ".json") and not (h.forms / "control.json").exists()


def test_the_inputs_step_checks_the_stored_anchors_too(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=())
    assert h.anchors() == 0
    h.plant_stored_anchors()
    assert h.main("inputs") == 0
    monkeypatch.setattr(Q, "STORED_PARENT", dict(Q.STORED_PARENT, blank=(Q.STORED_PARENT["blank"][0], "0" * 64)))
    with pytest.raises(SystemExit, match="SHA-256"):
        h.main("inputs")


def test_a_smoke_control_compares_nothing_and_says_so(monkeypatch, tmp_path, capsys):
    h = Comp(monkeypatch, tmp_path, draws=(1,), declared=False, n=5)
    assert h.main("read", "--forms", "control", "--out-dir", str(h.forms), "--smoke-train", "5") == 0
    assert h.control() == 0 and "nothing compared" in capsys.readouterr().out and not (h.forms / "control.json").exists()


def test_every_form_reads_the_parts_jobs_candidates_and_each_draw_its_four_forms(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=(1, 2))
    h.to_control()
    assert h.read_draws("1,2") == 0
    calls = h.reader_parent.calls
    order = lambda k: [f"d{k}_comp", f"d{k}_still", f"d{k}_blank", f"d{k}_comp_true", f"d{k}_still_true"]      # noqa: E731
    assert [c[0] for c in calls] == list(C.ANCHORS) + order(1) + order(2)
    want = M.retrieval_candidates(N, 5, seed=0)
    assert all(np.array_equal(c[2], want) and c[1] == 4 for c in calls)
    form = json.loads((h.forms / "form-d1_still.json").read_text())
    assert form["word_counts"] == [0 if i == EMPTY else 1 + i % 3 for i in range(N)] and len(form["sentence_tokens"]) == N and "hands" in form and "tracked" in form
    assert form["seconds_over_reference"][1] == pytest.approx(form["frames"][1] / form["reference_frames"][1], abs=1e-4)
    blank = json.loads((h.forms / "form-d1_blank.json").read_text())
    assert blank["frames"] == form["frames"] and not any(blank["valid_frame_share"])
    assert form["kind"] == "parent-comp-form" and form["report_only"] is True and form["declared_read"] is True and form["instrument"]["sha256"] == PP.UNFITTED_SHA and form["empty"][EMPTY] is True and sum(form["empty"]) == 1 and form["frames"][0] == 24 and form["reference_frames"][0] == len(h.refs[0].timestamps)
    true = json.loads((h.forms / "form-d1_comp_true.json").read_text())
    assert true["frames"] == true["reference_frames"]


def test_a_read_is_write_once_and_a_resume_leaves_what_stands(monkeypatch, tmp_path, capsys):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    h.to_control()
    assert h.read_draws("1") == 0
    with pytest.raises(SystemExit, match="write-once"):
        h.read_draws("1")
    assert h.read_draws("1", "--resume") == 0 and '"read": 0' in capsys.readouterr().out


def test_fidelity_keeps_a_count_and_no_text_or_score(monkeypatch, tmp_path, capsys):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    with pytest.raises(SystemExit, match="this draw is not read"):
        h.fidelity(1)
    record = json.loads((h.forms / "fidelity-d1.json").read_text())
    assert record["clips"] == min(C.FIDELITY_CLIPS, N - 1) and record["identical"] == 0 and record["draw"] == 1 and record["kind"] == "parent-comp-fidelity"
    assert set(record) == {"kind", "report_only", "draw", "clips", "identical", "frames", "code_digest", "environment", "instrument", "note"}
    out = capsys.readouterr().out
    assert "bleu" not in out and "w x y" not in out


def test_fidelity_counts_what_matches_on_non_empty_clips(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    import score_back_translation
    kept = json.loads(h.store.get(C.top10_key(1)))["forms"][C.PREDICTED]["hypotheses"]
    seen = []
    monkeypatch.setattr(score_back_translation, "translate", lambda model, kind, states: seen.append(len(states)) or [kept[i] for i in range(N) if i != EMPTY][:len(states)])
    monkeypatch.setattr(P, "load_reader", lambda store, args: type("V", (), dict(model=None, kind="keypoints", info=dict(sha256=L.TRANSLATOR_SHA)))())
    assert h.main("fidelity", "--draw", "1", "--clips", str(N), "--composed", "comp", "--out-dir", str(h.forms)) == 0
    assert json.loads((h.forms / "fidelity-d1.json").read_text())["identical"] == min(C.FIDELITY_CLIPS, N - 1) == seen[0]


def test_fidelity_on_d20_exits_1_under_its_floor_and_a_rerun_checks_the_record_again(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    (h.forms).mkdir(parents=True, exist_ok=True)
    h.plant_fidelity(1, identical=23)
    with pytest.raises(SystemExit, match="23 of 32.*floor 24"):
        h.main("fidelity", "--draw", "1", "--clips", str(N), "--composed", "comp", "--out-dir", str(h.forms))
    h.plant_fidelity(1, identical=24)
    assert h.main("fidelity", "--draw", "1", "--clips", str(N), "--composed", "comp", "--out-dir", str(h.forms)) == 0
    h.plant_fidelity(1, identical=32, frames_off=1)
    with pytest.raises(SystemExit, match="1 frame counts off"):
        h.main("fidelity", "--draw", "1", "--clips", str(N), "--composed", "comp", "--out-dir", str(h.forms))


def test_a_smokes_fidelity_stops_nothing(monkeypatch, tmp_path):
    assert C.fidelity_passed(dict(identical=0, clips=5, frames=dict(off=3)), False) is True
    assert C.fidelity_floor(32) == 24 and C.fidelity_floor(8) == 6 and C.FIDELITY_CLIPS == 32


def index_for_frames(rows):
    return dict(states=rows)


def stored_for_frames(plan):
    return dict(per_clip=[dict(sign="a", plans=dict(core_crossfade=plan))])


def test_exported_frames_are_compared_on_both_forms_with_the_stored_plan():
    plan = dict(frames=30, predicted=dict(frames=24))
    good = [dict(sign="a", form=C.PREDICTED, frames=24, empty=False), dict(sign="a", form=C.TRUE_LENGTH, frames=30, empty=False)]
    assert C.frames_off(index_for_frames(good), stored_for_frames(plan)) == dict(compared=2, off=0, first_off=None)
    assert C.frames_off(index_for_frames([dict(good[0], frames=25)]), stored_for_frames(plan))["off"] == 1
    assert C.frames_off(index_for_frames([dict(good[1], frames=24)]), stored_for_frames(plan))["first_off"] == "a"
    assert C.frames_off(index_for_frames([dict(good[0], sign="z")]), stored_for_frames(plan))["off"] == 1
    assert C.frames_off(index_for_frames([dict(good[0], empty=True)]), stored_for_frames(plan))["off"] == 1
    assert C.frames_off(index_for_frames([dict(good[0])]), dict(per_clip=[dict(sign="a", plans={})]))["off"] == 1
    assert C.frames_off(index_for_frames([dict(good[0], empty=True)]), dict(per_clip=[dict(sign="a", plans={})])) == dict(compared=0, off=0, first_off=None)


def test_fidelity_compares_every_exported_clips_frames(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    import score_back_translation
    monkeypatch.setattr(score_back_translation, "translate", lambda model, kind, states: [json.loads(h.store.get(C.top10_key(1)))["forms"][C.PREDICTED]["hypotheses"][i] for i in range(N) if i != EMPTY][:len(states)])
    monkeypatch.setattr(P, "load_reader", lambda store, args: type("V", (), dict(model=None, kind="keypoints", info=dict(sha256=L.TRANSLATOR_SHA)))())
    assert h.main("fidelity", "--draw", "1", "--clips", str(N), "--composed", "comp", "--out-dir", str(h.forms)) == 0
    record = json.loads((h.forms / "fidelity-d1.json").read_text())
    assert record["frames"] == dict(compared=2 * (N - 1), off=0, first_off=None)


def test_fidelity_refuses_a_clip_outside_its_stored_report_and_mismatched_export(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    report = json.loads(h.store.get(C.top10_key(1)))
    report["signs"] = [f"z{i}" for i in range(N)]
    raw = json.dumps(report).encode()
    h.store.put(C.top10_key(1), raw)
    monkeypatch.setattr(C, "DRAW_INPUTS", {**C.DRAW_INPUTS, 1: dict(C.DRAW_INPUTS[1], top10=hashlib.sha256(raw).hexdigest())})
    with pytest.raises(SystemExit, match="not in its stored report"):
        h.main("fidelity", "--draw", "1", "--clips", str(N), "--composed", "comp", "--out-dir", str(h.forms))
    h.plant_export(1, draw=2)
    with pytest.raises(SystemExit, match="word cache is draw 2"):
        h.main("fidelity", "--draw", "1", "--clips", str(N), "--composed", "comp", "--out-dir", str(h.forms))


# ---- reading --------------------------------------------------------------------------------------------------------------------------

def test_over_draws_gives_the_mean_the_t_interval_on_draws_and_the_clip_interval():
    out = C.over_draws([1., 2., 3.], np.array([[1., 1.1], [2., 2.1], [3., 3.1]]))
    assert out["mean"] == 2. and out["per_draw"] == [1., 2., 3.] and out["between_draw_sd"] == 1. and out["draws"] == 3
    half = 4.302653 * 1. / np.sqrt(3)
    assert abs(out["t_ci95"][0] - (2. - half)) < 1e-3 and abs(out["t_ci95"][1] - (2. + half)) < 1e-3
    assert out["clips_ci95"][0] == pytest.approx(2.0025, abs=1e-3) and out["clips_ci95"][1] == pytest.approx(2.0975, abs=1e-3)
    one = C.over_draws([1.], np.array([[1., 1.]]))
    assert one["t_ci95"] is None and one["between_draw_sd"] is None


def test_the_pipeline_has_its_planted_values(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path)
    report = h.pipeline()
    assert report["kind"] == "parent-comp-report" and report["report_only"] is True and report["alerts"] == [] and report["declared_read"] is True
    # nearest and real over the still's .1
    assert abs(report["anchors"]["nearest_two_view"]["margin_over_still"]["point"] - .5) < 1e-6 and abs(report["anchors"]["real_two_view"]["margin_over_still"]["point"] - .4) < 1e-6
    # a non-empty clip reads comp .1 + .03k over still .05; the empty clip reads exactly zero and stays in the mean
    n_filled = N - 1
    draw3 = report["per_draw"]["3"]
    want = (.19 - .05) * n_filled / N
    assert abs(draw3["margin"]["point"] - want) < 1e-5 and draw3["empty"] == 1 and draw3["empty_true_length"] == 1 and draw3["pod"] == "a"
    assert abs(draw3["share_of_real"]["point"] - want / .4) < 1e-3 and abs(draw3["share_of_nearest"]["point"] - want / .5) < 1e-3
    means = [(.1 + .03 * k - .05) * n_filled / N for k in C.DRAWS]
    block = report["composition"]
    assert abs(block["margin"]["mean"] - np.mean(means)) < 1e-4 and block["margin"]["draws"] == 8 and block["margin"]["between_draw_sd"] > 0 and block["margin"]["t_ci95"][0] < block["margin"]["mean"] < block["margin"]["t_ci95"][1]
    assert abs(block["share_of_real"]["mean"] - np.mean(means) / .4) < 1e-3 and abs(block["share_of_nearest"]["mean"] - np.mean(means) / .5) < 1e-3
    assert block["bleu"]["per_draw"][0] < block["bleu"]["per_draw"][-1]
    # gap shares against this draw's blank: likelihood (2.97 + ... ) by the formula, BLEU by comp_a_read's
    a, r, b = -2., -1.1, -3.
    want_share = ((N - 1) * (a - b) + 0.) / N / (r - b)
    assert abs(draw3["likelihood_gap_share"]["point"] - want_share) < 1e-3 and draw3["bleu_gap_share"]["blank"] < draw3["bleu_gap_share"]["bleu"] < draw3["bleu_gap_share"]["real"]
    assert abs(block["likelihood_gap_share"]["mean"] - want_share) < 1e-3 and block["bleu_gap_share"]["draws"] == 8 and block["bleu_gap_share"]["per_draw"][2] == pytest.approx(draw3["bleu_gap_share"]["share"], abs=1e-3)
    assert report["clips"] == dict(list=N, in_v1_training=4, outside_v1_training=N - 4) and report["by_v1_training"]["in_v1_training"]["clips"] == 4
    assert set(report["fidelity"]) == {str(k) for k in C.DRAWS}


def test_the_length_rows_read_each_form_over_its_own_still(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path)
    length = h.pipeline()["composition"]["length"]
    filled = (N - 1) / N
    means_pred = np.array([(.1 + .03 * k - .05) * filled for k in C.DRAWS])
    means_true = np.array([(.2 + .03 * k - .1) * filled for k in C.DRAWS])
    assert abs(length["true_length_margin"]["mean"] - means_true.mean()) < 1e-4 and abs(length["true_minus_predicted"]["mean"] - (means_true - means_pred).mean()) < 1e-4
    # over the reference-length still's .1 rather than the composition's own .05: the empty clip's raw margin (.02) over .1 counts too
    over = np.array([((.1 + .03 * k - .1) * (N - 1) + (.02 - .1)) / N for k in C.DRAWS])
    assert abs(length["over_reference_still"]["mean"] - over.mean()) < 1e-4 and abs(length["own_still_minus_reference_still"]["mean"] - (means_pred - over).mean()) < 1e-4
    assert length["per_clip_margin_on_log_length_ratio"]["clips"] == N
    for row in (length["true_minus_predicted"], length["own_still_minus_reference_still"]):
        assert row["clips_ci95"][0] <= row["mean"] <= row["clips_ci95"][1]
    assert length["bleu_true_minus_predicted"]["clips_ci95"][0] <= length["bleu_true_minus_predicted"]["mean"] <= length["bleu_true_minus_predicted"]["clips_ci95"][1]


def test_the_length_ratio_is_the_compositions_frames_over_the_references(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path)
    h.pipeline()
    forms = json.loads((h.forms / "form-d1_comp.json").read_text())
    idx = np.zeros((2, N), int)
    built = C.draw_rows(1, {f"d1_{k}": json.loads((h.forms / f"form-d1_{k}.json").read_text()) for k in C.KINDS}, {a: json.loads((h.forms / f"form-{a}.json").read_text()) for a in C.ANCHORS}, idx)
    assert np.allclose(built["log_ratio"], np.log(np.asarray(forms["frames"], float) / np.asarray(forms["reference_frames"], float)))


def test_v1s_stored_share_stands_beside_it_and_on_clips_outside_its_training(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path)
    beside = h.pipeline()["v1_stored"]
    assert beside["gap_share"][0] == .3 and abs(beside["mean_gap_share"] - .335) < 1e-6 and abs(beside["mean_bleu"] - 2.375) < 1e-6 and beside["t_ci95_gap_share"][0] < .335 < beside["t_ci95_gap_share"][1]
    assert len(beside["gap_share_outside_v1_training"]) == 8 and all(0 < x < 1 for x in beside["gap_share_outside_v1_training"]) and beside["mean_gap_share_outside_v1_training"] > 0


def test_v1s_likelihood_share_is_the_formula_and_the_pairs_are_like_with_like(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path)
    report = h.pipeline()
    beside = report["v1_stored"]
    assert beside["likelihood_gap_share"] == [.5] * 8           # (-2 - -3) / (-1 - -3), the None left out
    pair = beside["parent_minus_v1"]
    block = report["composition"]
    assert pair["likelihood_gap_share"]["per_draw"][0] == pytest.approx(block["likelihood_gap_share"]["per_draw"][0] - .5, abs=1e-3)
    assert pair["bleu_gap_share"]["per_draw"][2] == pytest.approx(block["bleu_gap_share"]["per_draw"][2] - beside["gap_share"][2], abs=1e-3) and pair["bleu_gap_share"]["t_ci95"][0] <= pair["bleu_gap_share"]["mean"] <= pair["bleu_gap_share"]["t_ci95"][1]
    assert "not compared with v1's BLEU share" in pair["note"]


def test_v1_stored_of_other_clips_refuses():
    report = dict(signs=["b", "a"], references=[], forms={}, likelihood=dict(per_form={}))
    with pytest.raises(SystemExit, match="other clips or another order"):
        C.v1_beside(dict(forms=dict(predicted=dict(gap_share=dict(top10=[]), top10_bleu=[]))), {1: report}, dict(outside_v1_training=np.array([True, True])), ["a", "b"])


def test_per_signer_and_word_rows_are_report_only_and_add_up(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path)
    report = h.pipeline()
    signers = report["per_signer"]
    assert set(signers) == {"c01", "c03", "c05"} and sum(v["clips"] for v in signers.values()) == N and set(signers["c01"]) >= {"margin", "share_of_real", "bleu_gap_share", "likelihood_gap_share"}
    assert "c03" in C.DISCLOSURES["per_signer"] and report["disclosures"]["per_signer"] == C.DISCLOSURES["per_signer"]
    words = report["words"]
    assert words["report_only"] is True and words["non_empty"]["clips_per_draw"] == N - 1
    per_draw = sum(k["clips_per_draw"] for k in words["by_coverage"].values())
    assert per_draw == N - 1 and words["non_empty"]["margin"] == pytest.approx(np.mean([.1 + .03 * k - .05 for k in C.DRAWS]), abs=1e-4)


def test_coverage_bins_read_words_over_tokens():
    a = dict(margin=np.array([1., 2., 3., 0.]), real=np.ones(4), non_empty=np.array([True, True, True, False]), coverage=np.array([.2, .4, .9, 0.]))
    out = C.word_rows({1: a})
    assert out["by_coverage"]["0.0-0.25"]["margin"] == 1. and out["by_coverage"]["0.25-0.5"]["margin"] == 2. and out["by_coverage"]["0.5-inf"]["margin"] == 3. and out["non_empty"]["clips_per_draw"] == 3.
    assert C.word_rows({1: dict(a, non_empty=np.zeros(4, bool))})["non_empty"] == dict(clips_per_draw=0.)


def test_a_restricted_form_keeps_this_jobs_per_clip_fields_in_step(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=(1,))
    h.to_control()
    assert h.read_draws("1") == 0
    form = json.loads((h.forms / "form-d1_comp.json").read_text())
    out = C.restrict_form(form, ["s5", "s1"])
    assert out["signs"] == ["s5", "s1"] and out["frames"] == [form["frames"][5], form["frames"][1]] and out["word_counts"] == [form["word_counts"][5], form["word_counts"][1]] and out["groups"] == [form["groups"][5], form["groups"][1]] and out["empty"] == [False, False]


def test_a_still_or_blank_of_another_length_refuses_the_merge(monkeypatch, tmp_path):
    h = base_for_merge(monkeypatch, tmp_path)
    for name in ("d3_still", "d4_still_true", "d5_blank"):
        path = h.forms / f"form-{name}.json"
        raw = path.read_text()
        form = json.loads(raw)
        form["frames"] = [x + 1 for x in form["frames"]]
        path.write_text(json.dumps(form))
        pod = tmp_path / "s" / "pods"
        import shutil
        shutil.rmtree(pod, ignore_errors=True)
        assert h.upload_pod() == 0
        with pytest.raises(SystemExit, match="is not as long as"):
            h.merge()
        path.write_text(raw)


def test_a_blank_at_length_is_blank_and_as_long():
    state = laid_out(signer_state(5, frames=12, amplitude=.4), TWO_VIEW)
    out = C.blank_at_length(state, 30)
    assert len(out.timestamps) == 30 and not out.channels["body"].valid.any()


def test_v1s_outside_share_takes_only_clips_outside_its_training():
    refs = ["a b c d e f", "g h i j k l", "m n o p q r"]
    report = dict(signs=["a", "b", "c"], likelihood=dict(per_form={C.PREDICTED: [1., 1., 1.], "blank@predicted": [0., 0., 0.], "real": [2., 2., 2.]}), references=refs, forms={C.PREDICTED: dict(hypotheses=[refs[0], "z", "z"]), "real": dict(hypotheses=refs), "blank@predicted": dict(hypotheses=["y"] * 3)})
    summary = dict(forms=dict(predicted=dict(gap_share=dict(top10=[.1, .2]), top10_bleu=[1., 2.])))
    masks = dict(outside_v1_training=np.array([True, False, False]))
    assert C.v1_beside(summary, {1: report}, masks, ["a", "b", "c"])["gap_share_outside_v1_training"] == [1.]
    masks = dict(outside_v1_training=np.array([False, True, True]))
    assert C.v1_beside(summary, {1: report}, masks, ["a", "b", "c"])["gap_share_outside_v1_training"] == [0.]


def test_a_read_off_the_plan_raises_an_alert_and_check_exits_3(monkeypatch, tmp_path, capsys):
    h = Comp(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_draws() == 0
    for k in C.DRAWS:
        h.plant_fidelity(k)
    assert h.upload_pod() == 0
    h.plant_plan()
    monkeypatch.setattr(C, "PLAN", dict(C.PLAN, nearest_margin=.9, real_bleu=99., v1_share=.9, v1_bleu=9.))
    report = h.merge()
    assert len(report["alerts"]) == 4 and any("not the plan's .9" in a or "0.9" in a for a in report["alerts"])
    assert h.main("check", "--report", str(tmp_path / "report.json")) == 3 and "ALERT" in capsys.readouterr().out


@pytest.mark.parametrize("identical,alerted", [(23, True), (24, False)])
def test_fidelity_under_its_floor_raises_an_alert(monkeypatch, tmp_path, identical, alerted):
    h = Comp(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_draws() == 0
    for k in C.DRAWS:
        h.plant_fidelity(k, identical if k == 4 else 8)
    assert h.upload_pod() == 0
    h.plant_plan()
    assert any("draw 4" in a for a in h.merge()["alerts"]) is alerted


def test_no_fidelity_record_or_one_of_other_code_raises_an_alert(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_draws() == 0
    for k in C.DRAWS[:-1]:
        h.plant_fidelity(k)
    (h.forms / "fidelity-d8.json").write_text(json.dumps(dict(draw=8, clips=32, identical=32, frames=dict(compared=1, off=0, first_off=None), code_digest="d" * 64)))
    assert h.upload_pod() == 0
    h.plant_plan()
    assert any("draw 8: no fidelity" in a for a in h.merge()["alerts"])


def test_empty_counts_that_disagree_raise_an_alert(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_draws() == 0
    path = h.forms / "form-d5_comp_true.json"
    read = json.loads(path.read_text())
    read["empty"] = [False] * N
    path.write_text(json.dumps(read))
    for k in C.DRAWS:
        h.plant_fidelity(k)
    assert h.upload_pod() == 0
    h.plant_plan()
    assert any("draw 5: 1 empty" in a for a in h.merge()["alerts"])


def test_a_check_passes_a_clean_report_and_refuses_broken_plumbing(monkeypatch, tmp_path, capsys):
    h = Comp(monkeypatch, tmp_path)
    h.pipeline()
    path = tmp_path / "report.json"
    assert h.main("check", "--report", str(path)) == 0
    clean = json.loads(path.read_text())
    for change, word in ((dict(report_only=False), "report_only"), (dict(checks=dict(clean["checks"], controls=dict(a=dict(reproduced=False)))), "passed control"), (dict(checks=dict(clean["checks"], blank_identical_to_real=1.)), "blank reads")):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(dict(clean, **change)))
        assert h.main("check", "--report", str(bad)) == 3
        assert word in capsys.readouterr().out


def base_for_merge(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_draws() == 0
    for k in C.DRAWS:
        h.plant_fidelity(k)
    h.plant_plan()
    return h


def test_stale_forms_of_other_code_refuse_the_merge(monkeypatch, tmp_path):
    h = base_for_merge(monkeypatch, tmp_path)
    assert h.upload_pod() == 0
    monkeypatch.setattr(C, "code_digest", lambda: "e" * 64)
    with pytest.raises(SystemExit, match="a stale read"):
        h.merge()


@pytest.mark.parametrize("field,value,match", [("declared_read", False, "smoke mixed"), ("signs", [f"z{i}" for i in range(N)], "other clips"), ("instrument", dict(sha256="0" * 64), "not read by the parent"), ("form", "d1_still", "holds form")])
def test_forms_of_another_run_refuse_the_merge(monkeypatch, tmp_path, field, value, match):
    h = base_for_merge(monkeypatch, tmp_path)
    path = h.forms / "form-d1_comp.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), **{field: value})))
    assert h.upload_pod() == 0
    with pytest.raises(SystemExit, match=match):
        h.merge()


def test_a_missing_draw_or_a_draw_missing_a_form_refuses_the_merge(monkeypatch, tmp_path):
    h = base_for_merge(monkeypatch, tmp_path)
    (h.forms / "form-d8_comp.json").unlink()
    assert h.upload_pod() == 0
    with pytest.raises(SystemExit, match="draw 8 on pod a"):
        h.merge()
    h2 = Comp(monkeypatch, tmp_path / "x", draws=C.DRAWS) if (tmp_path / "x").mkdir() is None else None
    h2.to_control()
    assert h2.read_draws("1,2,3,4,5,6,7") == 0
    assert h2.upload_pod() == 0
    with pytest.raises(SystemExit, match="incomplete"):
        h2.merge()


def test_ready_waits_for_every_draw(monkeypatch, tmp_path, capsys):
    h = Comp(monkeypatch, tmp_path)
    h.to_control()
    assert h.read_draws("1,2,3,4") == 0
    assert h.upload_pod() == 0
    assert h.main("ready", "--pods", "a", "--prefix", "pods") == 5 and '"missing": [5, 6, 7, 8]' in capsys.readouterr().out
    assert h.read_draws("5,6,7,8", "--resume") == 0
    assert h.upload_pod() == 0
    assert h.main("ready", "--pods", "a", "--prefix", "pods") == 0


def split_pods(monkeypatch, tmp_path, second_control=None):
    """Draws 1-4 on pod a, 5-8 on pod b, each with its anchors."""
    h = Comp(monkeypatch, tmp_path)
    first, second = tmp_path / "pod-a", tmp_path / "pod-b"
    h.forms = first
    h.to_control()
    assert h.read_draws("1,2,3,4") == 0
    for k in (1, 2, 3, 4):
        h.plant_fidelity(k)
    h.forms = second
    monkeypatch.setattr(P, "environment", lambda: dict(gpu="other"))
    h.to_control()
    assert h.read_draws("5,6,7,8") == 0
    for k in (5, 6, 7, 8):
        h.plant_fidelity(k)
    if second_control:
        path = second / "control.json"
        path.write_text(json.dumps(dict(json.loads(path.read_text()), **second_control)))
    assert h.upload_pod("a", first) == 0 and h.upload_pod("b", second) == 0
    h.forms = first
    h.plant_plan()
    return h


def test_two_pods_merge_each_with_its_own_anchors(monkeypatch, tmp_path):
    h = split_pods(monkeypatch, tmp_path)
    report = h.merge("a,b")
    assert report["alerts"] == [] and report["per_draw"]["1"]["pod"] == "a" and report["per_draw"]["8"]["pod"] == "b" and set(report["environments"]) == {"a", "b"} and report["composition"]["margin"]["draws"] == 8


def test_a_pod_without_a_passed_control_refuses_the_merge(monkeypatch, tmp_path):
    h = split_pods(monkeypatch, tmp_path, second_control=dict(reproduced=False))
    with pytest.raises(SystemExit, match="pod b: no passed control"):
        h.merge("a,b")


def test_a_draw_read_on_two_pods_or_a_pod_without_anchors_refuses(monkeypatch, tmp_path):
    h = split_pods(monkeypatch, tmp_path)
    for kind in C.KINDS:
        h.store.put(f"pods/b/forms/form-d1_{kind}.json", h.store.get(f"pods/a/forms/form-d1_{kind}.json"))
    with pytest.raises(SystemExit, match="draw 1 was read on pods a and b"):
        h.merge("a,b")
    (h.store.root / "pods/a/forms/form-blank.json").unlink()
    with pytest.raises(SystemExit, match="draw 1 was read|its anchors are missing"):
        h.merge("a,b")


def test_no_form_refuses_the_merge(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=())
    with pytest.raises(SystemExit, match="no form"):
        h.merge()


def test_a_report_is_write_once(monkeypatch, tmp_path):
    h = base_for_merge(monkeypatch, tmp_path)
    assert h.upload_pod() == 0
    h.merge()
    with pytest.raises(SystemExit, match="write-once"):
        h.merge()
    path = tmp_path / "report.json"
    assert h.main("put-report", "--file", str(path), "--key", "pc/report.json") == 0
    with pytest.raises(SystemExit, match="write-once"):
        h.main("put-report", "--file", str(path), "--key", "pc/report.json")
    with pytest.raises(SystemExit, match="finished"):
        h.main("preflight", "--out-dir", str(tmp_path / "w3"), "--final-key", "pc/report.json", "--void-prefix", "pc/void")


def test_a_smoke_reads_composition_against_train_references_and_carries_no_stored_read(monkeypatch, tmp_path):
    h = Comp(monkeypatch, tmp_path, draws=(1,), declared=False, n=5)
    assert h.main("read", "--forms", "control", "--out-dir", str(h.forms), "--smoke-train", "5") == 0
    assert h.main("read", "--draws", "1", "--composed", "comp", "--out-dir", str(h.forms), "--smoke-train", "5") == 0
    assert h.upload_pod() == 0
    report = h.merge()
    assert report["declared_read"] is False and "SMOKE" in report["smoke"] and report["v1_stored"] is None and any("no fidelity" in a for a in report["alerts"])


def test_the_header_states_the_reading_before_any_read():
    head = C.__doc__
    for text in ("Like is compared with like", "about twofold", "also holds pose", "No label and no bar", "32.5%", "its own still pose at its own length", "A clip with no selected word is counted", "None of these says people understand", "not in the brief's forms", "234 of D20"):
        assert text in head, text
    assert C.FIDELITY_SHARE == .75 and C.PLAN["nearest_margin"] == .2372 and C.PLAN["v1_share"] == .325 and C.PLAN["v1_bleu"] == 2.402 and C.PLAN["real_bleu"] == 2.130 and C.PLAN["blank_bleu"] == 1.058 and C.PLAN["still_bleu"] == .895 and C.PLAN["nearest_bleu"] == 2.207


# ---- the job ---------------------------------------------------------------------------------------------------------------------------

def run_job(tmp_path, fail_when=None, code=7, ready=0, **env):
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    pattern = "*" + "*".join(f'"{x}"' for x in fail_when.split("*")) + "*" if fail_when else ""
    match = f'case "$*" in {pattern}) echo "planted failure" >&2; exit {code};; esac' if fail_when else ""
    (bin_dir / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\n{match}\ncase "$2" in ready) exit {ready};; esac\necho "ran $1 $2"\nexit 0\n')
    (bin_dir / "uv").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "timeout").write_text(f'#!/bin/bash\necho "timeout $*" >> {tmp_path / "timeouts.log"}\nshift 3\nexec "$@"\n')
    for name in ("python", "uv", "timeout"):
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    environment = dict(PATH=f"{bin_dir}:/usr/bin:/bin", HOME=str(tmp_path), VENV_BIN=str(bin_dir), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path), WORK=str(tmp_path / "work"), **env)
    done = subprocess.run(["bash", str(JOB)], env=environment, capture_output=True, text=True, timeout=60)
    return done, log.read_text() if log.exists() else ""


def done_line(result) -> int:
    lines = [x for x in result.stdout.splitlines() if x.startswith("PARENT-COMP-DONE ")]
    assert len(lines) == 1 and lines[0] == result.stdout.strip().splitlines()[-1]
    return int(lines[0].split()[1])


def commands(calls):
    return [C.build_parser().parse_args(x.split()[1:]).command for x in calls.splitlines()]


def test_the_runner_sees_no_credentials():
    assert not [k for k in os.environ if k.startswith(("R2_", "AWS_", "RUNPOD_"))]


def test_the_job_smokes_exports_controls_then_reads_every_draw(tmp_path):
    result, calls = run_job(tmp_path)
    assert result.returncode == 0 and done_line(result) == 0
    got = commands(calls)
    assert got[:11] == ["inputs", "export", "read", "control", "fidelity", "read", "upload", "merge", "put-report", "check", "preflight"]
    assert got.count("export") == 9 and got.index("control", 11) > max(i for i, c in enumerate(got) if c == "export")
    assert got[-5:] == ["ready", "merge", "put-report", "check", "upload-log"] and got.count("fidelity") == 9
    lines = calls.splitlines()
    assert "--clips 5" in lines[1] and "--smoke-train 5" in lines[2] and "smoke" in lines[1] and "round20/parent-comp/report.json" in lines[-3]
    assert all(line.startswith("scripts/parent_comp.py ") for line in lines)
    assert sum("--draws" in line and "--resume" in line for line in lines) == 8
    assert (tmp_path / "timeouts.log").read_text().count("-k 60 3600 python scripts/parent_comp.py export") == 9
    assert got.index("fidelity", 11) < next(i for i, x in enumerate(lines) if "--draws" in x and "--resume" in x)


def test_a_draw_under_its_fidelity_floor_is_not_read(tmp_path):
    result, calls = run_job(tmp_path, fail_when="fidelity --draw 4*work/d20/forms", code=1)
    assert result.returncode == 1 and done_line(result) == 1 and "draw 4 did not pass fidelity" in result.stdout and not [x for x in calls.splitlines() if "--draws 4" in x and "--resume" in x]


def test_two_pods_read_their_own_draws_and_wait(tmp_path):
    result, calls = run_job(tmp_path, POD="a", PODS="a,b", DRAWS="1 2 3 4", ready=5)
    assert result.returncode == 0 and "the merge waits" in result.stdout
    got = commands(calls)
    assert got.count("export") == 5 and got.count("fidelity") == 5 and "put-report" in got[:12] and got[-2:] == ["ready", "upload-log"]
    assert "pods/a/forms" in calls and "--pods a,b" in calls


@pytest.mark.parametrize("pod", ["", "A/b", "../x"])
def test_a_bad_pod_name_refuses(tmp_path, pod):
    result, calls = run_job(tmp_path, POD=pod)
    if pod == "":
        assert result.returncode == 0       # unset falls back to a
    else:
        assert result.returncode == 1 and calls == "" and done_line(result) == 1


def test_the_smoke_only_mode_stops_before_d20(tmp_path):
    result, calls = run_job(tmp_path, SMOKE="1")
    assert result.returncode == 0 and "preflight" not in calls and "SMOKE done" in result.stdout


@pytest.mark.parametrize("fail,status,word", [("inputs", 1, "pins do not hold"), ("--clips 5", 2, "smoke export failed"), ("preflight", 1, "preflight refused"), ("--draw 3 --work", 1, "an export failed"), ("--forms control*--resume", 1, "read of the anchors"), ("--draws 5*--resume", 1, "read of draw 5"), ("put-report*parent-comp/report.json", 1, "final report"), ("ready", 1, "ready did not run")])
def test_a_failing_step_stops_the_job_with_its_status(tmp_path, fail, status, word):
    result, calls = run_job(tmp_path, fail_when=fail)
    assert result.returncode == status and done_line(result) == status and word in result.stdout and "upload --partial" in calls


def test_a_void_control_stops_with_3_and_reads_no_draw(tmp_path):
    result, calls = run_job(tmp_path, fail_when="control --out-dir*work/d20/forms", code=3)
    assert result.returncode == 3 and done_line(result) == 3 and "do not retry" in result.stdout and not [x for x in calls.splitlines() if "--draws" in x and "--resume" in x]


def test_a_missing_repo_prints_its_done_line(tmp_path):
    done = subprocess.run(["bash", str(JOB)], env=dict(PATH="/usr/bin:/bin", HOME=str(tmp_path), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path / "gone")), capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and done.stdout.strip().endswith("PARENT-COMP-DONE 1")


def test_the_job_passes_every_step_through_pipestatus_and_names_no_other_list():
    text = JOB.read_text()
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0
    code = [line.split(" #")[0] for line in text.splitlines() if not line.startswith("#")]
    assert [line for line in code if re.search(r"(?<!\|)\|(?!\|)", line) and not line.startswith("case")] == ['  "$@" 2>&1 | tee -a "$LOG"'] and "${PIPESTATUS[@]}" in text and "set -uo pipefail" in text and "transformers==4.57.6" in text
    for word in ("c20", "C20", "create-pod", "runpodctl", "--likelihood"):
        assert word not in "\n".join(code), word
    assert "L40S" in text
