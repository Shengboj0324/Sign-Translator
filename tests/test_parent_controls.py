"""The parent controls on synthetic data: what a real word record must be, the seeded pick and its fallback, the permutations and what they keep, the composition rebuilt byte for byte, the controls, the arithmetic of the report, what is refused, and the job. Nothing here reads a real clip, a checkpoint or R2."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import parent_comp as C  # noqa: E402
import parent_controls as PC  # noqa: E402
import parent_lib as Q  # noqa: E402
import path_parts as PP  # noqa: E402
import path_read as R  # noqa: E402
import round15_stage2 as r2  # noqa: E402
import round15_stage2_fit as fit  # noqa: E402
import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v1 as T  # noqa: E402
from evaluate_text_to_motion import blanked, load_state  # noqa: E402
from signtranslator.generation import compositional_grammar as cg  # noqa: E402
from signtranslator.generation.composition_v2 import predicted_seconds  # noqa: E402
from signtranslator.generation.compositional import parts_from_pieces  # noqa: E402
from tests.test_generation_signing_space import signer_state  # noqa: E402
from tests.test_parent_comp import ANCHOR_LEVELS, Comp, PARENT_INFO, settings_of, top10_report  # noqa: E402
from tests.test_path_read import MONOCULAR, N, TWO_VIEW, laid_out, tiny_tokenizer  # noqa: E402

JOB = ROOT / "runpod/fleet/parent_controls.sh"
SENTENCES = (("alpha", "beta", "gamma"), ("you",), (), ("alpha", "you", "beta"), ("sem", "alpha"), ("fails", "gamma"), ("delta", "delta", "beta"), ("gamma", "alpha", "delta", "beta"), ("beta", "alpha"))
EMPTY, ONLY_ADDED = 2, 1
FILLED, ORACLE_FILLED = N - 1, N - 2
MARGIN = dict(oracle=.3, covered=.5, shuffled_s0=.55, shuffled_s1=.56, oracle_shuffled_s0=.25, oracle_shuffled_s1=.26, covered_shuffled_s0=.42, covered_shuffled_s1=.44)
STILL_MARGIN = .1
TIMING = dict(gap=.1668, default_seconds=.2336, floor=.2, ceiling=1.5, ratio=1., trimmed_over_annotated=2.9375, annotated=dict(alpha=.5), sentence_seconds={})
FRACTIONS = dict(entries=dict(alpha=(.2, .8)), all=(.1, .9), min_seconds={})
VOCAB = dict(alpha=.9, beta=1.1, gamma=.8, delta=1.3, you=.6, sem=1., fails=1.)
MONO_LEVELS = dict(real_monocular=(13, 1.2, .8), nearest_monocular=(12, 1.3, .75), still_monocular=(1, 2.9, .15))      # not two-view's still: an anchor taken from the wrong path shows
MIRROR_INFO = dict(PC.INSTRUMENTS["mirror"], kind="keypoints")
V1_INFO = dict(PC.INSTRUMENTS["v1"], kind="keypoints")


def level_of(name):
    """(words said, own NLL, margin over the mismatched) of a form's clips with signing."""
    if name in ANCHOR_LEVELS:
        return ANCHOR_LEVELS[name]
    if name in MONO_LEVELS:
        return MONO_LEVELS[name]
    match = PC.BASE_FORM.fullmatch(name)
    if match:
        k, kind = int(match[1]), match[2]
        return dict(comp_reread=(9 + k, 2., .6 + .001 * k), comp=(9 + k, 2., .6 + .001 * k), still=(1, 2.9, STILL_MARGIN), blank=(0, 3., .02))[kind]
    match = PC.FORM.fullmatch(name)
    assert match, name
    if match[3]:
        return (1, 2.9, STILL_MARGIN)
    return (5 if "oracle" in match[2] else 8, 2., MARGIN[match[2]])


class Reader:
    def __init__(self, info=PARENT_INFO):
        self.info, self.form, self.calls, self.inputs = info, None, [], {}

    def read(self, states, texts, candidates, group=2):
        self.calls.append((self.form, group, candidates.copy()))
        self.inputs[self.form] = [hashlib.sha256(R.reader_input(s).tobytes()).hexdigest() for s in states]
        k, nll, margin = level_of(self.form)
        rows, said = [], []
        for s, row in zip(states, candidates):
            if self.form in ANCHOR_LEVELS or self.form in MONO_LEVELS or s.channels["body"].valid.any():
                rows.append([nll] + [nll + margin] * (candidates.shape[1] - 1))
                said.append(" ".join(texts[row[0]].split()[:k]) if k else "z y x w")
            else:
                rows.append([3.] + [3.02] * (candidates.shape[1] - 1))
                said.append("z y x w")
        return dict(hypotheses=said, nll=np.array(rows))


def real_record(sign, corpus="rit-text-r6", **change):
    return dict(dict(sign=sign, corpus=corpus, partition="train", text=sign.split("-")[0], signer_component=None), **change)


POOLS = dict(alpha=[real_record("alpha-a", "asllvd-text-v1"), real_record("alpha-b")], beta=[real_record("beta-a", "wlasl-text-r6"), real_record("beta-b", "wlasl-text-r6")], gamma=[real_record("gamma-a")], delta=[real_record("delta-a")], fails=[real_record("fails-a"), real_record("fails-b")])


def real_state(record, entry):
    if record["sign"].startswith("fails"):
        raise ValueError(f"{record['sign']}: window [0.5, 0.6] contains no frames")
    seed = int.from_bytes(hashlib.sha256(record["sign"].encode()).digest()[:2], "big")
    return laid_out(signer_state(seed, frames=40, amplitude=.5), TWO_VIEW)


def context(**change):
    model = dict(timing=TIMING)
    intercept, slope, _ = (float(x) for x in C.DURATION.split(","))
    out = dict(model=model, plan_model=model, vocab=VOCAB, fractions=FRACTIONS, duration=dict(intercept=intercept, slope=slope), pools=POOLS, added=frozenset({"you"}))
    return SimpleNamespace(**dict(out, **change))


class Controls(Comp):
    """A stored export, stored parent forms and a word cache, all planted; the real plan, render, read and merge run on them."""

    def __init__(self, monkeypatch, tmp_path, *, compose_draws=(1, 2), **kw):
        super().__init__(monkeypatch, tmp_path, draws=(), **kw)
        self.reader_parent = Reader()
        self.reader_mirror = Reader(MIRROR_INFO)
        self.forms_mirror = tmp_path / "wm"
        self.reader_v1 = Reader(V1_INFO)
        self.forms_v1 = tmp_path / "wv1"
        monkeypatch.setattr(P, "load_reader", lambda store, args: self.reader_v1)
        monkeypatch.setattr(Q, "load_mono", lambda store, args: self.reader_mirror)
        current = C.read_one

        def tracked(name, *a, **k):
            k["reader"].form = name
            return current(name, *a, **k)
        monkeypatch.setattr(C, "read_one", tracked)
        self.clips = [R.Clip(i, r["sign"], *(laid_out(signer_state(300 + i, frames=48, amplitude=.4), MONOCULAR) for _ in range(3)), {}) for i, r in enumerate(self.records)]
        monkeypatch.setattr(R, "open_monocular", lambda store, records, refs, declared: (self.clips[:len(records)], [], None))
        monkeypatch.setattr(PC, "code_digest", lambda: "c" * 64)
        monkeypatch.setattr(PC, "STORED_DIGEST", "b" * 64)
        monkeypatch.setattr(PC, "STORED", {k: dict(index="0" * 64, comp="0" * 64, still="0" * 64, blank="0" * 64) for k in PC.DRAWS})
        self.ctx = context()
        monkeypatch.setattr(PC, "MAX_NOT_SERVED", .5)      # the planted word "fails" is above the real limit, which its own test holds
        monkeypatch.setattr(PC, "load_context", lambda store, model_file: self.ctx)
        monkeypatch.setattr(PC, "round_tripper", lambda store, device, work: real_state)
        for k in compose_draws:
            self.plant_stored_export(k)

    def plant_training(self, signs):
        """v1's lists: the first clips in training, the next two in its selection sample."""
        meta = {"train": json.dumps([dict(clip_id=s, partition="train") for s in signs]).encode(), "val": json.dumps([dict(clip_id=r["sign"], partition="val") for r in self.records[4:6]]).encode()}
        parts = []
        for partition, raw in meta.items():
            self.store.put(f"fk/{partition}/shard-00000.json", raw)
            parts.append(dict(partition=partition, shards=[dict(key=f"fk/{partition}/shard-00000.npz", meta_sha256=hashlib.sha256(raw).hexdigest())]))
        index = json.dumps(dict(parts=parts)).encode()
        self.store.put(Q.FK_INDEX, index)
        self.monkeypatch.setattr(Q, "FK_INDEX_SHA", hashlib.sha256(index).hexdigest())

    def main(self, *argv):
        return PC.main([argv[0], *(() if argv[0] in ("check", "digest") else ("--store", f"dir:{self.tmp_path / 's'}")), *argv[1:]])

    def word_clip(self, k, word):
        seed = int.from_bytes(hashlib.sha256(f"{k}|{word}".encode()).digest()[:2], "big")
        state = laid_out(signer_state(seed, frames=30, amplitude=.4), TWO_VIEW)
        path = self.tmp_path / f"word-{k}-{word}.npz"
        sha = state.save(path)
        key = f"cache/draw{k}/{word}.npz"
        self.store.put(key, path.read_bytes())
        return state, key, sha

    def plant_stored_export(self, k, ctx=None, pin=True, sentences=SENTENCES):
        """A stored export of draw k, made as the real run makes it."""
        ctx = ctx or self.ctx
        rows, clips = [], {w: self.word_clip(k, w) for w in VOCAB}
        for i, record in enumerate(self.records):
            words = list(sentences[i])
            seconds = predicted_seconds(ctx.duration, record["text"])
            frames = max(1, int(round(seconds * 24)))
            if words:
                pieces, provenance = [], []
                for w in words:
                    window = r2.sign_core(w, clips[w][0], fractions=FRACTIONS, margin=0.)
                    pieces.append(dict(state=clips[w][0], **window))
                    provenance.append(dict(word=w, lo=window["lo"], hi=window["hi"], reason=window["reason"], seed=7, sample=3, clip_key=clips[w][1], clip_sha256=clips[w][2]))
                job = dict(words=words, weights=r2.plan_words(words, ctx.plan_model, ctx.vocab)["weights"])
                state, _ = r2.render_form(dict(pieces=pieces, parts=parts_from_pieces(pieces)), job, seconds, ctx.model, timing="sign-fitted", sample_id=f"round15-eval-{C.PREDICTED}:{i}")
            else:
                provenance = []
                state = C.blank_at_length(self.refs[i], frames)
            path = self.tmp_path / f"export-{k}-{i}.npz"
            sha = state.save(path)
            for form in C.EXPORT_FORMS:
                key = f"{PC.STORED_COMPOSED}/draw{k}/states/{i}-{form.replace('@', '-')}.npz"
                self.store.put(key, path.read_bytes())
                rows.append(dict(clip=i, sign=record["sign"], text=record["text"], form=form, empty=not words, state_key=key, state_sha256=sha, frames=frames, reference_frames=len(self.refs[i].timestamps), reference_sha256=record["continuous"]["state_sha256"], words=provenance))
        raw = json.dumps(dict(kind="round15-composed-states", settings=settings_of(k, self.n), word_cache=dict(draw=k, checkpoint_key="prior/w14b.pt"), states=rows)).encode()
        self.store.put(C.export_index_key(PC.STORED_COMPOSED, k), raw)
        if pin:
            PC.STORED[k]["index"] = hashlib.sha256(raw).hexdigest()

    def compose(self, k=1, *extra, clips=None):
        return self.main("compose", "--draw", str(k), "--clips", str(self.n if clips is None else clips), "--work", str(self.tmp_path / "work"), "--prefix", "pc/composed", "--model-file", "unused", *extra)

    def index(self, k=1, prefix="pc/composed"):
        return json.loads(self.store.get(PC.index_key(prefix, k)))

    def plant_stored_forms(self, k, change=None):
        """Draw k's comp, still and blank as the stored read holds them, of the stored digest, by their pins."""
        rows = C.load_export(self.store, PC.STORED_COMPOSED, k, self.records, True)[C.PREDICTED]
        rows = [dict(r, state_key=None, empty=r["empty"]) if r["empty"] else r for r in rows]
        states = PC.read_states(self.store, rows, self.refs)
        out = self.tmp_path / f"stored-forms-{k}-{len(list(self.tmp_path.glob('stored-forms-*')))}"
        args = SimpleNamespace(nll_group=4)
        texts, signs, groups = [r["text"] for r in self.records], [r["sign"] for r in self.records], [r["signer_component"] for r in self.records]
        shared = dict(reader=self.reader_parent, texts=texts, signs=signs, groups=groups, candidates=C.M.retrieval_candidates(self.n, C.CANDIDATES, seed=0), refs=self.refs, args=args, declared=True, described=dict(list="t", sha256=L.D20_SHA, clips=self.n), digest=PC.STORED_DIGEST, place=dict(gpu="stored"), out_dir=out, started=0.)
        for kind, how in (("comp", "as_composed"), ("still", "still"), ("blank", "blank")):
            name = f"d{k}_{kind}"
            C.read_one(name, C.kind_states(how, states, rows, self.refs, self.tokenizer), list(self.refs), rows, **shared)
            read = json.loads((out / f"form-{name}.json").read_text())
            read.update((change or {}).get(kind, {}))
            raw = json.dumps(read).encode()
            self.store.put(PC.stored_form_key(k, kind), raw)
            PC.STORED[k][kind] = hashlib.sha256(raw).hexdigest()

    def comp_check(self, k=1):
        assert self.main("read", "--forms", "comp-check", "--draws", str(k), "--composed", "pc/composed", "--out-dir", str(self.forms)) == 0
        return self.main("comp-check", "--draw", str(k), "--out-dir", str(self.forms), "--void-prefix", "pc/void")

    def read_draws(self, draws="1,2", *extra):
        return self.main("read", "--draws", draws, "--composed", "pc/composed", "--out-dir", str(self.forms), *extra)

    def to_read(self, draws=(1, 2)):
        """Composed, the stored forms planted, the anchors and the composition check passed, each draw read."""
        for k in draws:
            assert self.compose(k) == 0
            self.plant_stored_forms(k)
        self.to_control()
        assert self.comp_check(draws[0]) == 0
        assert self.read_draws(",".join(str(k) for k in draws)) == 0

    def mirror(self, *argv):
        return self.main(argv[0], "--instrument", "mirror", *argv[1:])

    def mirror_anchors(self):
        return self.mirror("read", "--forms", "control", "--out-dir", str(self.forms_mirror))

    def mirror_control(self):
        return self.mirror("control", "--out-dir", str(self.forms_mirror), "--void-prefix", "pc/void-mirror")

    def plant_mirror_stored(self, change=None, offsets=None):
        """The mirror's stored forms and report by their pins, with this reader's values as the report's; ``offsets`` move a report value."""
        digest = "a" * 64
        self.monkeypatch.setattr(PC, "MIRROR_DIGEST", digest)
        hashes, values = {}, {}
        for name in PC.MIRROR_ANCHORS:
            read = json.loads((self.forms_mirror / f"form-{name}.json").read_text())
            raw = json.dumps(dict(dict(read, form=f"mono__{name}", code_digest=digest), **(change or {}).get(name, {}))).encode()
            self.store.put(f"{PC.MIRROR_FORMS}/form-mono__{name}.json", raw)
            hashes[name] = hashlib.sha256(raw).hexdigest()
            values[name] = C.M.bleu_ci_fast(read["hypotheses"], read["references"])[0] + (offsets or {}).get(name, 0.)
        self.monkeypatch.setattr(PC, "MIRROR_STORED", hashes)
        self.monkeypatch.setattr(PC, "MIRROR_VALUES", {name: values[name] for name in ("real_monocular", "nearest_monocular", "nearest_two_view", "blank")})
        report = json.dumps(dict(code_digest=digest, forms={name: dict(bleu=value) for name, value in values.items()})).encode()
        self.store.put(PC.MIRROR_REPORT, report)
        self.monkeypatch.setattr(PC, "MIRROR_REPORT_SHA", hashlib.sha256(report).hexdigest())

    def to_mirror_control(self):
        assert self.mirror_anchors() == 0
        self.plant_mirror_stored()
        assert self.mirror_control() == 0

    def read_mirror(self, draws="1,2", *extra):
        return self.mirror("read", "--draws", draws, "--composed", "pc/composed", "--out-dir", str(self.forms_mirror), *extra)

    def mirror_report(self, draws=(1, 2), name="report-mirror.json", parent_forms=None):
        for k in draws:
            if not self.store.exists(PC.index_key("pc/composed", k)):
                assert self.compose(k) == 0
        self.to_mirror_control()
        assert self.read_mirror(",".join(str(k) for k in draws)) == 0
        assert self.main("upload", "--out-dir", str(self.forms_mirror), "--prefix", "pc/forms-mirror") == 0
        out = self.tmp_path / name
        beside = ("--parent-forms", parent_forms) if parent_forms else ()
        assert self.mirror("merge", "--draws", ",".join(str(k) for k in draws), "--composed", "pc/composed", "--prefix", "pc/forms-mirror", "--output-file", str(out), *beside) == 0
        return json.loads(out.read_text())

    def v1(self, *argv):
        return self.main(argv[0], "--instrument", "v1", *argv[1:])

    def pin_composed(self, draws):
        """What v1 reads: this run's composed indexes, by their hashes, as the first read's."""
        self.monkeypatch.setattr(PC, "COMPOSED_DIGEST", "c" * 64)
        self.monkeypatch.setattr(PC, "COMPOSED_INDEX", {k: hashlib.sha256(self.store.get(PC.index_key("pc/composed", k))).hexdigest() for k in draws})
        self.monkeypatch.setattr(PC, "DRAWS", tuple(draws))
        self.monkeypatch.setattr(PC, "PREFIX", "pc")

    def plant_a5(self, draws, words=None, **change):
        """Each draw's stored A5 report and the summary, as the planted reader would read their compositions."""
        records = self.records
        bleus = list(PC.V1_A5_BLEU)
        inputs = dict(C.DRAW_INPUTS)
        for k in draws:
            said = [("z y x w" if i == EMPTY else " ".join(r["text"].split()[:(words or 9 + k)])) for i, r in enumerate(records)]
            report = dict(dict(top10_report(k, n=self.n), signs=[r["sign"] for r in records], references=[r["text"] for r in records]), **change)
            report["forms"][C.PREDICTED] = dict(hypotheses=said)
            report["likelihood"] = dict(per_form={C.PREDICTED: [-3. if i == EMPTY else -2. for i in range(self.n)]})
            raw = json.dumps(report).encode()
            self.store.put(C.top10_key(k), raw)
            inputs[k] = dict(inputs[k], top10=hashlib.sha256(raw).hexdigest())
            bleus[k - 1] = C.M.bleu_ci_fast(said, report["references"])[0]
        self.monkeypatch.setattr(C, "DRAW_INPUTS", inputs)
        self.monkeypatch.setattr(PC, "V1_A5_BLEU", tuple(bleus))
        summary = json.dumps(dict(forms=dict(predicted=dict(top10_bleu=bleus)))).encode()
        self.store.put(C.A5_SUMMARY, summary)
        self.monkeypatch.setattr(C, "A5_SUMMARY_SHA", hashlib.sha256(summary).hexdigest())

    def v1_anchors(self):
        return self.v1("read", "--forms", "control", "--out-dir", str(self.forms_v1))

    def v1_control(self):
        return self.v1("control", "--out-dir", str(self.forms_v1), "--void-prefix", "pc/void-v1")

    def plant_v1_stored(self, change=None, canary=None):
        """The pinned canary report and the parts job's forms, as this reader read them."""
        read = {n: json.loads((self.forms_v1 / f"form-{n}.json").read_text()) for n in PC.MIRROR_ANCHORS}
        table = {}
        for name in PC.V1_STORED_ANCHORS:
            raw = json.dumps(dict(read[name], **(change or {}).get(name, {}))).encode()
            self.store.put(f"{Q.PARTS_FORMS}/s-{name}.json", raw)
            table[name] = (f"s-{name}.json", hashlib.sha256(raw).hexdigest())
        self.monkeypatch.setattr(Q, "STORED_V1", table)
        forms = {new: dict(dict(hypotheses=read[old]["hypotheses"], per_clip=dict(nll=[x[0] for x in read[old]["nll"]])), **(canary or {}).get(new, {})) for new, old in (("real", "real_two_view"), ("blank", "blank"))}
        raw = json.dumps(dict(signs=read["real_two_view"]["signs"], references=read["real_two_view"]["references"], forms=forms)).encode()
        self.store.put(P.CANARY_REPORT, raw)
        self.monkeypatch.setattr(P, "CANARY_REPORT_SHA", hashlib.sha256(raw).hexdigest())
        self.monkeypatch.setattr(P, "CANARY", dict(P.CANARY, real=C.M.bleu_ci_fast(read["real_two_view"]["hypotheses"], read["real_two_view"]["references"])[0], blank=C.M.bleu_ci_fast(read["blank"]["hypotheses"], read["blank"]["references"])[0]))

    def v1_comp_check(self, k=1):
        assert self.v1("read", "--forms", "comp-check", "--draws", str(k), "--out-dir", str(self.forms_v1)) == 0
        return self.v1("comp-check", "--draw", str(k), "--out-dir", str(self.forms_v1), "--void-prefix", "pc/void-v1")

    def v1_read(self, draws=(1, 2)):
        for k in draws:
            if not self.store.exists(PC.index_key("pc/composed", k)):
                assert self.compose(k) == 0
        self.pin_composed(draws)
        self.plant_a5(draws)
        assert self.v1_anchors() == 0
        self.plant_v1_stored()
        assert self.v1_control() == 0
        assert self.v1_comp_check(draws[0]) == 0
        assert self.v1("read", "--draws", ",".join(str(k) for k in draws), "--composed", "pc/composed", "--out-dir", str(self.forms_v1)) == 0
        assert self.main("upload", "--out-dir", str(self.forms_v1), "--prefix", "pc/forms-v1") == 0

    def v1_report(self, draws=(1, 2), name="report-v1.json"):
        self.v1_read(draws)
        out = self.tmp_path / name
        assert self.v1("merge", "--draws", ",".join(str(k) for k in draws), "--composed", "pc/composed", "--prefix", "pc/forms-v1", "--output-file", str(out)) == 0
        return json.loads(out.read_text())

    def upload(self):
        return self.main("upload", "--out-dir", str(self.forms), "--prefix", "pc/forms")

    def merge(self, draws="1,2", name="report.json", composed="pc/composed", prefix="pc/forms"):
        out = self.tmp_path / name
        assert self.main("merge", "--draws", draws, "--composed", composed, "--prefix", prefix, "--output-file", str(out)) == 0
        return json.loads(out.read_text())

    def report(self, draws=(1, 2)):
        self.to_read(draws)
        assert self.upload() == 0
        return self.merge(",".join(str(k) for k in draws))


# ---- what the brief declares --------------------------------------------------------------------------------------------------------------

def test_the_header_states_the_reading_before_any_read():
    head = PC.__doc__
    for text in ("paired t interval over draws (``t_ci95``), on two metrics: margin over still, and BLEU", "excludes zero on BOTH metrics in the same direction", "is **unresolved**", "The clip bootstrap interval is reported beside and decides nothing", "Three mirror draws make unresolved likely there", "Generated against real clips is ``oracle`` minus ``covered``", "It is never oracle against composition", "The format is ``oracle`` against real signing", "On the mirror the like is ``nearest_monocular``", "with ``real_monocular`` beside it", "the middle case", "the format reads high AND generated clips read higher still", "the words the oracle lacks and the frames they free", "Sem-Lex-only words and words not served", "real word clips are MediaPipe monocular too", "selection: each generated clip was a recogniser's best of 16", "d2 decoding", "a different signer at every word", "applied to WLASL and RIT", "fewer words: the oracle lacks seven frequent ones, and each core is stretched", "A large drop", "the rebuilt joins or the different still", "A small drop", "order-invariant signals", "near-zero margin has nothing to lose", "Seeds are not independent replicates", "A two-word clip cannot vary", "None of this shows human comprehension", "No label and no bar", "no target moves", "a chance pick, not best of 16", "An uncovered word is dropped", "estimated, not measured", "14,400 s", "The mirror is report only and never moves M1. Generated motion reading high on it is flagged on the same terms as on the parent,", "How2Sign monocular trained it", "A parent void or a parent check alert stops the job before the mirror reads", "the composed state as it stands, the very bytes the parent read", "its like on the mirror is ``nearest_monocular``", "the mirror reads three draws (1, 2, 3), not eight", "**Resuming.**", "**On a new pod,**", "runs/<code digest>/", "may not be blank", "the NVIDIA L40S", "So the mirror's third draw can be left out", "must be a prefix of ``DRAWS``", "shared helpers", "Every step runs under a stated limit"):
        assert text in head, text
    assert "at or above" not in head and "at or below" not in head
    assert "separate two causes" not in head and "These forms separate" not in head
    assert PC.READING.startswith("A contrast is above or below only if its paired t interval") and "at or above" not in PC.READING and "None of it shows human comprehension" in PC.READING and "never moves M1" in PC.READING


def test_c20_cannot_be_named_and_no_other_instrument_or_selection_is_here():
    for command in PC.build_parser()._subparsers._group_actions[0].choices.values():
        flags = {o for a in command._actions for o in a.option_strings}
        assert not flags & {"--sentences-key", "--list", "--translator-run", "--checkpoint", "--tokenizer", "--word-cache-key"}
    code = re.sub(r"[0-9a-f]{64}", "", (ROOT / "scripts/parent_controls.py").read_text().split('"""', 2)[2]).lower()
    assert code.count("p.load_reader(") == 1 and "p.load_reader(store, args)" in code, "v1 is loaded in one place, to read"
    for word in ("c20", "test_rgb_front", "rerank=", "rerank(", "citizen", "stem", "backward", "load_recogn"):
        assert not re.search(rf"\b{re.escape(word)}", code), word


def test_the_pins_are_the_stored_reads():
    assert PC.STORED_REPORT_SHA.startswith("a41781a8") and PC.STORED_DIGEST.startswith("dc3d89b0") and set(PC.STORED) == set(PC.DRAWS) and PC.DRAWS == tuple(range(1, 9))
    hashes = [h for k in PC.DRAWS for h in PC.STORED[k].values()]
    assert len(set(hashes)) == 32 and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)
    assert PC.STORED[1]["index"].startswith("82ea1571") and PC.STORED[8]["blank"].startswith("6a16525e")
    assert PP.UNFITTED_SHA.startswith("f1ff9446") and PC.SEEDS == (0, 1) and PC.REQUIRED_GPU == "NVIDIA L40S"
    assert len(set(PC.MIRROR_STORED.values())) == 7 and set(PC.MIRROR_STORED) == set(PC.MIRROR_ANCHORS) and PC.MIRROR_REPORT_SHA.startswith("223dd372") and PC.MIRROR_DIGEST.startswith("de05576e") and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in PC.MIRROR_STORED.values())
    mirror = PC.INSTRUMENTS["mirror"]
    assert mirror["sha256"] == "1a996cfc9c1917181fef849c6a9b8dde0406c34e42fd48a75f1df63fb189ca59" and mirror["checkpoint_key"] == "training-runs/back-translator-kp2d-uwb-ft-mono-v1/checkpoints/step-00009000.pt"
    assert PC.MIRROR_VALUES == dict(real_monocular=4.715, nearest_monocular=4.783, nearest_two_view=3.702, blank=1.159) and PC.MIRROR_TOLERANCE == .05


# ---- what may be read ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("corpus", ["asllvd-text-v1", "rit-text-r6", "wlasl-text-r6"])
def test_a_train_record_of_the_three_corpora_passes(corpus):
    record = real_record("a", corpus)
    assert PC.check_word_record(record) is record


@pytest.mark.parametrize("change,word", [(dict(corpus="semlex-text-v1"), "outside"), (dict(corpus="how2sign-two-view-text-v1"), "outside"), (dict(corpus="asl-citizen-text-v1"), "outside"), (dict(corpus="asl-stem-wiki"), "outside"), (dict(corpus=None), "outside"), (dict(partition="val"), "not exactly train"), (dict(partition="test"), "not exactly train"), (dict(partition=None), "not exactly train"), (dict(partition="Train"), "not exactly train"), (dict(signer_component="c00"), "c00"), (dict(signer_component="c00", corpus="how2sign-two-view-text-v1", partition="test"), "c00")])
def test_any_other_record_is_refused(change, word):
    with pytest.raises(SystemExit, match=word):
        PC.check_word_record(real_record("a", **change))


def pool_manifest():
    return dict(records=[
        dict(real_record("in-1", "asllvd-text-v1", text="in"), continuous=dict(seconds=1., state_sha256="0" * 64)),
        dict(real_record("in-2", "wlasl-text-r6", text="in"), continuous=dict(seconds=1., state_sha256="0" * 64)),
        dict(real_record("val", "rit-text-r6", text="held", partition="val"), continuous=dict(seconds=1., state_sha256="0" * 64)),
        dict(real_record("sem", "semlex-text-v1", text="sem"), continuous=dict(seconds=1., state_sha256="0" * 64)),
        dict(real_record("h2s", "how2sign-two-view-text-v1", text="sentence", signer_component="c02"), continuous=dict(seconds=1., state_sha256="0" * 64)),
        dict(real_record("you-1", "wlasl-text-r6", text="you"), continuous=dict(seconds=1., state_sha256="0" * 64)),
        dict(real_record("no-seconds", "rit-text-r6", text="short"), continuous=dict())])


def test_the_pools_hold_only_train_records_of_the_three_corpora_and_no_added_entry(monkeypatch):
    import signtranslator.data_engineering.corpus_manifest as cm
    monkeypatch.setattr(cm, "usable_channels", lambda r, kind: True)
    pools = PC.real_pools(pool_manifest(), {"you"})
    assert set(pools) == {"in"} and [r["sign"] for r in pools["in"]] == ["in-1", "in-2"]
    with_added = PC.real_pools(pool_manifest(), ())
    assert set(with_added) == {"in", "you"}


def test_a_c00_record_in_a_pool_stops_the_job_before_any_clip(monkeypatch):
    import signtranslator.data_engineering.corpus_manifest as cm
    monkeypatch.setattr(cm, "usable_channels", lambda r, kind: True)
    manifest = pool_manifest()
    manifest["records"].append(dict(real_record("bad", "rit-text-r6", text="bad", signer_component="c00"), continuous=dict(seconds=1., state_sha256="0" * 64)))
    with pytest.raises(SystemExit, match="c00"):
        PC.real_pools(manifest, ())


def test_the_context_is_the_arms_and_the_mix_is_cut_to_train(monkeypatch, tmp_path):
    import comp_a_lib
    import signtranslator.data_engineering.corpus_manifest as cm
    monkeypatch.setattr(cm, "usable_channels", lambda r, kind: True)
    model = dict(mix_key=L.EVAL_MIX, corpora=["asllvd-text-v1", "rit-text-r6", "wlasl-text-r6", "semlex-text-v1"], timing=dict(TIMING, annotated=dict(alpha=.5, you=.9)))
    model["sha256"] = cg.model_sha256(model)
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model))
    monkeypatch.setattr(C, "MODEL_SHA", model["sha256"])
    manifest = pool_manifest()
    manifest["records"].append(dict(real_record("in-test", "wlasl-text-r6", text="only test", partition="test"), continuous=dict(seconds=9., state_sha256="0" * 64)))
    monkeypatch.setattr(L, "load_manifest_at", lambda store, key, sha: manifest)
    arm = comp_a_lib.arm_of(comp_a_lib.load_token_list(ROOT / T.TOKENS_FILE, T.TOKENS_SHA), 10)
    assert set(arm["entries"]) == {"you", "it", "your", "this", "so", "i", "or"}
    ctx = PC.load_context(None, path)
    assert ctx.added == frozenset(arm["entries"]) and set(ctx.pools) == {"in"} and "only test" not in ctx.vocab
    assert ctx.vocab["you"] == float(arm["entries"]["you"]["typical_seconds"]) and "you" not in ctx.plan_model["timing"]["annotated"] and "alpha" in ctx.plan_model["timing"]["annotated"]
    assert ctx.fractions["min_seconds"]["you"] == .2 and ctx.duration == dict(intercept=-.083, slope=.3691)
    other = dict(model, mix_key="other")
    other["sha256"] = cg.model_sha256({k: v for k, v in other.items() if k != "sha256"})
    path.write_text(json.dumps(other))
    monkeypatch.setattr(C, "MODEL_SHA", other["sha256"])
    with pytest.raises(SystemExit, match="another vocabulary"):
        PC.load_context(None, path)


def test_a_model_of_other_bytes_or_another_hash_refuses(tmp_path, monkeypatch):
    model = dict(timing=TIMING)
    model["sha256"] = cg.model_sha256(model)
    path = tmp_path / "m.json"
    path.write_text(json.dumps(model))
    with pytest.raises(SystemExit, match="not the pinned"):
        PC.load_model(path)
    path.write_text(json.dumps(dict(model, timing=dict(TIMING, gap=.5))))
    with pytest.raises(SystemExit, match="does not match"):
        PC.load_model(path)
    monkeypatch.setattr(C, "MODEL_SHA", model["sha256"])
    path.write_text(json.dumps(model))
    assert PC.load_model(path)["timing"]["gap"] == TIMING["gap"]


# ---- the pick and the fallback ------------------------------------------------------------------------------------------------------------

def test_the_pick_is_fixed_by_draw_and_entry_and_is_in_range():
    picks = {(k, w): PC.pick_index(k, w, 7) for k in PC.DRAWS for w in ("alpha", "beta", "gamma")}
    assert all(0 <= v < 7 for v in picks.values())
    assert picks == {(k, w): PC.pick_index(k, w, 7) for k in PC.DRAWS for w in ("alpha", "beta", "gamma")}
    assert len({PC.pick_index(k, "alpha", 1000) for k in PC.DRAWS}) > 1
    assert PC.pick_index(1, "alpha", 1000) != PC.pick_index(1, "beta", 1000)
    want = int.from_bytes(hashlib.sha256(b"parent-controls|oracle|draw3|alpha").digest()[:8], "big") % 1000
    assert PC.pick_index(3, "alpha", 1000) == want


def test_one_clip_per_entry_in_a_draw_each_loaded_once():
    calls = []
    real = PC.RealClips(POOLS, 1, lambda r, e: calls.append(r["sign"]) or real_state(r, e))
    first = real.of("alpha")
    assert real.of("alpha") is first and calls == [first[0]["sign"]]


def test_no_pool_is_the_fallback_and_a_clip_that_cannot_be_served_gives_way():
    real = PC.RealClips(POOLS, 1, real_state)
    assert real.of("sem") is None and real.failed == 0
    assert real.of("fails") is None and real.failed == 2
    pools = dict(word=[real_record("fails-x"), real_record("good")])
    start = PC.pick_index(1, "word", 2)
    real = PC.RealClips(pools, 1, real_state)
    record, _ = real.of("word")
    assert record["sign"] == "good" and real.failed == (1 if start == 0 else 0)


@pytest.mark.parametrize("error", [RuntimeError("CUDA out of memory"), ValueError("motion shard content hash mismatch"), ValueError("fetched bytes for k do not match the manifest hash"), ValueError("right_hand: layout has no declared equivalence"), ValueError("s: no usable continuous channels"), RuntimeError("collated frames differ from the state's")])
def test_only_an_empty_window_gives_way_and_anything_else_stops_the_draw(error):
    def load(record, entry):
        raise error
    real = PC.RealClips(dict(word=[real_record("a"), real_record("b")]), 1, load)
    with pytest.raises(type(error), match=str(error)[:12]):
        real.of("word")
    assert real.failed == 0


def test_a_torch_out_of_memory_stops_the_draw():
    import torch

    def load(record, entry):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        PC.RealClips(dict(word=[real_record("a")]), 1, load).of("word")


def test_an_empty_window_is_counted_and_gives_way():
    def load(record, entry):
        if record["sign"] == "a":
            raise ValueError("a: window [1.0, 2.0] contains no frames")
        return real_state(real_record("gamma-a"), entry)
    real = PC.RealClips(dict(word=[real_record("a"), real_record("b")]), 1, load)
    record, _ = real.of("word")
    assert record["sign"] == "b" and real.failed == (1 if PC.pick_index(1, "word", 2) == 0 else 0)


def test_too_many_entries_not_served_alerts_and_stops_with_no_index(monkeypatch, tmp_path, capsys):
    h = Controls(monkeypatch, tmp_path)
    monkeypatch.setattr(PC, "MAX_NOT_SERVED", .02)
    # the planted word "fails" is one entry of five with a pool, and two failed attempts: the entry counts
    with pytest.raises(SystemExit, match="1 of 5 entries with a pool not served"):
        h.compose()
    assert "ALERT draw 1: 1 of 5 entries" in capsys.readouterr().out and not h.store.exists(PC.index_key("pc/composed", 1))


def test_the_limit_counts_entries_finally_not_served_and_not_failed_attempts(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    monkeypatch.setattr(PC, "MAX_NOT_SERVED", .3)      # one entry of five is under it, two attempts of five would be over
    assert h.compose() == 0
    cover = h.index()["coverage"]
    assert cover["picks_not_served"] == 2 and cover["entries_not_served"] == 1 and cover["entries_with_a_pool"] == 5
    real = PC.RealClips(dict(a=[real_record("fails-a")], b=[real_record("gamma-a")]), 1, real_state)
    assert real.of("a") is None and real.of("b") and real.of("sem") is None
    assert (real.entries_with_a_pool(), real.entries_not_served()) == (2, 1), "an entry with no pool is neither"


def test_a_smoke_reports_entries_not_served_and_goes_on(monkeypatch, tmp_path, capsys):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    h.plant_stored_export(1, sentences=(("fails", "gamma"), *SENTENCES[1:]))
    monkeypatch.setattr(PC, "MAX_NOT_SERVED", .02)
    assert h.compose(1, clips=2) == 0
    out = capsys.readouterr().out
    assert "not served" in out and "reported, since a smoke's few entries" in out and "ALERT" not in out and h.store.exists(PC.index_key("pc/composed", 1))


def test_a_poisoned_pool_is_refused_before_it_is_served():
    calls = []
    real = PC.RealClips(dict(word=[real_record("x", "semlex-text-v1")]), 1, lambda r, e: calls.append(r))
    with pytest.raises(SystemExit, match="outside"):
        real.of("word")
    assert not calls


def test_the_round_trip_is_d2s_codes_decoded_as_a_generation_and_refuses_before_serving(monkeypatch, tmp_path):
    import build_recon_states
    import path_read
    import torch
    from signtranslator.data.text_motion_dataset import DirectoryStore
    tokenizer = tiny_tokenizer()
    store = DirectoryStore(tmp_path / "s")
    path = tmp_path / "t.pt"
    torch.save(dict(kind="motion_rvq_tokenizer", config=tokenizer.config.to_dict(), model=tokenizer.state_dict()), path)
    store.put(L.TOKENIZER, path.read_bytes())
    monkeypatch.setattr(L, "TOKENIZER_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(path_read, "load_template", lambda s: laid_out(signer_state(1, frames=4), TWO_VIEW))
    state = laid_out(signer_state(3, frames=24, amplitude=.4), TWO_VIEW)
    served = []
    monkeypatch.setattr(build_recon_states, "serve", lambda record, template, *, cache, fetch: served.append(record["sign"]) or state)
    load = PC.round_tripper(store, "cpu", tmp_path / "w")
    decoded = load(real_record("alpha-a", "asllvd-text-v1"), "alpha")
    assert served == ["alpha-a"] and len(decoded.timestamps) == 24 and decoded.channels["right_hand"].valid.all()
    assert not np.allclose(decoded.channels["right_hand"].values, state.channels["right_hand"].values)
    # A5's post: no gains, the head smoothed
    from exposure_diagnostic import make_post
    loaded, _ = build_recon_states.load_tokenizer(path.read_bytes(), "cpu")
    smoothed = build_recon_states.decode_state(state, loaded, loaded.config.motion, make_post(None), text="alpha")
    bare = build_recon_states.decode_state(state, loaded, loaded.config.motion, lambda x: x, text="alpha")
    assert np.array_equal(decoded.channels["head_rotation"].values, smoothed.channels["head_rotation"].values) and not np.array_equal(decoded.channels["head_rotation"].values, bare.channels["head_rotation"].values)
    with pytest.raises(SystemExit, match="outside"):
        load(real_record("b", "semlex-text-v1"), "b")
    assert served == ["alpha-a"]


def test_a_tokenizer_of_other_bytes_refuses(monkeypatch, tmp_path):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "s")
    store.put(L.TOKENIZER, b"other")
    with pytest.raises(SystemExit, match="SHA-256"):
        PC.round_tripper(store, "cpu", tmp_path / "w")


# ---- orders ---------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 5, 12])
@pytest.mark.parametrize("seed", PC.SEEDS)
def test_an_order_is_a_permutation_never_the_identity_and_fixed_by_its_key(n, seed):
    for clip in range(20):
        order = PC.word_order(n, clip, seed, 3)
        assert sorted(order) == list(range(n)) and order != list(range(n)) and order == PC.word_order(n, clip, seed, 3)


def test_an_order_varies_with_draw_seed_and_clip_and_a_short_clip_cannot():
    assert len({tuple(PC.word_order(8, 3, s, 1)) for s in PC.SEEDS}) == 2 and len({tuple(PC.word_order(8, c, 0, 1)) for c in range(5)}) == 5 and len({tuple(PC.word_order(8, 3, 0, k)) for k in PC.DRAWS}) == 8
    assert {tuple(PC.word_order(2, c, s, k)) for c in range(5) for s in PC.SEEDS for k in PC.DRAWS} == {(1, 0)}
    assert PC.word_order(1, 0, 0, 1) == [0] and PC.word_order(0, 0, 0, 1) == []
    with pytest.raises(SystemExit, match="seeds"):
        PC.word_order(3, 0, 2, 1)
    with pytest.raises(SystemExit, match="pinned draws"):
        PC.word_order(3, 0, 0, 9)


@pytest.mark.parametrize("order,kept,want", [([2, 0, 1], [0, 1, 2], [2, 0, 1]), ([2, 0, 1], [0, 2], [1, 0]), ([3, 1, 0, 2], [1, 3], [1, 0]), ([1, 0], [1], [0]), ([1, 0], [], [])])
def test_a_restricted_order_is_the_same_permutation_on_what_is_kept(order, kept, want):
    assert PC.restricted(order, kept) == want


def plan_of(words=("alpha", "beta", "gamma", "delta")):
    return cg.fitted_plan(r2.plan_words(list(words), dict(timing=TIMING), VOCAB)["weights"], cg.gap_ratio(TIMING), 3.)


def test_a_moved_plan_keeps_each_words_frames_the_gaps_in_place_and_the_total():
    plan = plan_of()
    order = [2, 0, 3, 1]
    moved = PC.moved_plan(plan, order)
    assert moved["cores"] == [plan["cores"][k] for k in order] and moved["gaps"] == plan["gaps"] and sum(moved["cores"]) + sum(moved["gaps"]) == plan["frames"] == moved["frames"]
    assert sorted(moved["cores"]) == sorted(plan["cores"])
    with pytest.raises(ValueError, match="not an order"):
        PC.moved_plan(plan, [0, 0, 1, 2])
    with pytest.raises(ValueError, match="not an order"):
        PC.moved_plan(plan, [0, 1, 2])


def rendered_pieces(words=("alpha", "beta", "gamma", "delta")):
    pieces = []
    for n, w in enumerate(words):
        clip = laid_out(signer_state(50 + n, frames=30 + 4 * n, amplitude=.4), TWO_VIEW)
        pieces.append(dict(state=clip, **r2.sign_core(w, clip, fractions=FRACTIONS, margin=0.)))
    return pieces


def core_spans(plan: dict) -> list[tuple[int, int]]:
    """(first frame, frames) of each word's core in a rendered plan with no lead."""
    if plan["lead"]:
        raise ValueError("a plan with a lead")
    spans, at = [], 0
    for k, frames in enumerate(plan["cores"]):
        spans.append((at, frames))
        at += frames + (plan["gaps"][k] if k < len(plan["gaps"]) else 0)
    return spans


def core_frames(state, start, frames):
    return {name: c.values[start:start + frames].copy() for name, c in state.channels.items()}


@pytest.mark.parametrize("seed", PC.SEEDS)
def test_a_shuffle_keeps_every_words_rendered_frames_and_the_length_and_a_seed_is_reproducible(seed):
    pieces = rendered_pieces()
    plan = plan_of()
    order = PC.word_order(4, 11, seed, 1)
    base = PC.render_order(pieces, plan, [0, 1, 2, 3], "x")
    shuffled = PC.render_order(pieces, plan, order, "x")
    assert len(shuffled.timestamps) == len(base.timestamps) == plan["frames"]
    base_spans, moved_spans = core_spans(plan), core_spans(PC.moved_plan(plan, order))
    for at, k in enumerate(order):
        a, b = core_frames(base, *base_spans[k]), core_frames(shuffled, *moved_spans[at])
        for name in a:
            assert np.array_equal(a[name], b[name]), (name, k)
    tmp = [PC.render_order(pieces, plan, order, "x") for _ in range(2)]
    assert np.array_equal(tmp[0].channels["body"].values, tmp[1].channels["body"].values)
    # the bytes of the unmoved order are render_form's: the shuffle starts from the composition itself
    job = dict(words=["alpha", "beta", "gamma", "delta"], weights=r2.plan_words(["alpha", "beta", "gamma", "delta"], dict(timing=TIMING), VOCAB)["weights"])
    same, _ = r2.render_form(dict(pieces=pieces, parts=parts_from_pieces(pieces)), job, 3., dict(timing=TIMING), timing="sign-fitted", sample_id="x")
    assert np.array_equal(same.channels["right_hand"].values, base.channels["right_hand"].values)
    # and the order does change the frames in place
    assert not np.array_equal(shuffled.channels["right_hand"].values, base.channels["right_hand"].values)


def test_a_truncated_plan_shuffles_with_its_empty_cores_and_the_same_length():
    pieces = rendered_pieces()
    plan = cg.fitted_plan([1., 1., 1., 1.], .7, 2 / 24)
    assert plan["fit"] == "truncated" and plan["cores"] == [1, 1, 0, 0]
    out = PC.render_order(pieces, plan, [3, 0, 1, 2], "x")
    assert len(out.timestamps) == 2


def test_a_core_span_needs_no_lead():
    assert core_spans(dict(lead=0, cores=[3, 2, 4], gaps=[1, 2])) == [(0, 3), (4, 2), (8, 4)]
    with pytest.raises(ValueError, match="lead"):
        core_spans(dict(lead=1, cores=[3], gaps=[]))


# ---- composing -----------------------------------------------------------------------------------------------------------------------------

def test_a_draw_is_rebuilt_byte_for_byte_and_its_coverage_counted(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    index = h.index()
    assert index["kind"] == "parent-controls-composed" and index["identity"] == dict(non_empty=FILLED, byte_identical=FILLED, within_tolerance=FILLED, clips=N) and index["seeds"] == [0, 1]
    cover = index["coverage"]
    assert cover["words_selected"] == 20 and cover["words_kept"] == 16 and cover["dropped"] == dict(added_entry=2, no_clip_in_the_three_corpora=1, not_served=1)
    assert cover["clips_with_words"] == 8 and cover["clips_fully_covered"] == 4 and cover["clips_left_empty"] == 1 and cover["picks_not_served"] == 2 and cover["share_kept"] == .8
    assert set(cover["corpora"]) <= set(PC.H.WORD_CORPORA) and sum(cover["corpora"].values()) == 16


def test_every_form_keeps_the_compositions_length_clip_by_clip(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    rows = h.index()["rows"]
    stored = C.load_export(h.store, PC.STORED_COMPOSED, 1, h.records, True)[C.PREDICTED]
    for kind in PC.KINDS:
        for row, want in zip(rows[kind], stored):
            assert row["frames"] == want["frames"]
            if not row["empty"]:
                assert len(load_state(h.store, row["state_key"], row["state_sha256"]).timestamps) == want["frames"]


def test_the_oracle_holds_real_clips_of_the_kept_words_and_covered_the_generated_ones(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    rows = h.index()["rows"]
    assert [w["word"] for w in rows["oracle"][3]["words"]] == ["alpha", "beta"] and [w["word"] for w in rows["covered"][3]["words"]] == ["alpha", "beta"]
    assert all(w["corpus"] in PC.H.WORD_CORPORA and "clip_sha256" not in w for w in rows["oracle"][3]["words"])
    assert all("clip_sha256" in w and "corpus" not in w for w in rows["covered"][3]["words"])
    assert rows["oracle"][3]["dropped"] == [dict(word="you", reason="added_entry")] and rows["oracle"][4]["dropped"] == [dict(word="sem", reason="no_clip_in_the_three_corpora")]
    assert rows["oracle"][0]["state_sha256"] != rows["covered"][0]["state_sha256"]
    # a word repeated in a sentence is one clip, as a word cache holds one
    repeated = [w["sign"] for w in rows["oracle"][6]["words"]]
    assert repeated[0] == repeated[1] == "delta-a"
    # a fully covered clip's covered form is the composition itself, frame for frame
    composed = load_state(h.store, rows["comp"][0]["state_key"], rows["comp"][0]["state_sha256"])
    covered = load_state(h.store, rows["covered"][0]["state_key"], rows["covered"][0]["state_sha256"])
    assert np.array_equal(composed.channels["right_hand"].values, covered.channels["right_hand"].values)


def test_the_oracle_cuts_an_annotated_clip_at_its_own_sign_and_any_other_as_a_generated_clip(monkeypatch, tmp_path):
    import round15_stage2_fit as stage_fit
    annotated = dict(POOLS, alpha=[real_record("alpha-a", "asllvd-text-v1")], beta=[real_record("beta-a", "wlasl-text-r6")])
    monkeypatch.setattr(stage_fit, "record_sign_fraction", lambda r: (.4, .6) if r["corpus"] == "asllvd-text-v1" else None)
    h = Controls(monkeypatch, tmp_path)
    h.ctx = context(pools=annotated)
    assert h.compose() == 0
    words = h.index()["rows"]["oracle"][0]["words"]
    assert words[0]["reason"] == "annotated" and words[1]["reason"] == "all" and words[0]["word"] == "alpha"
    # the cut is margin zero, the composition's own, on both kinds of clip
    from signtranslator.generation.compositional import fraction_window
    alpha, beta = (real_state(record, None) for record in (annotated["alpha"][0], annotated["beta"][0]))
    own = fraction_window(alpha, (.4, .6), 0., reason="annotated")
    assert (words[0]["lo"], words[0]["hi"]) == (own["lo"], own["hi"]) and (own["lo"], own["hi"]) != (lambda w: (w["lo"], w["hi"]))(fraction_window(alpha, (.4, .6), .1))
    core = r2.sign_core("beta", beta, fractions=FRACTIONS, margin=0.)
    assert (words[1]["lo"], words[1]["hi"]) == (core["lo"], core["hi"]) and (core["lo"], core["hi"]) != (lambda w: (w["lo"], w["hi"]))(r2.sign_core("beta", beta, fractions=FRACTIONS, margin=.1))


def test_covered_renders_the_kept_generated_pieces_and_not_the_composition(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    rows = h.index()["rows"]
    row, comp = rows["covered"][3], rows["comp"][3]
    assert [w["word"] for w in row["words"]] == ["alpha", "beta"] and [w["word"] for w in comp["words"]] == ["alpha", "you", "beta"]
    pieces = [dict(state=load_state(h.store, f"cache/draw1/{w}.npz", next(x["clip_sha256"] for x in comp["words"] if x["word"] == w)), lo=next(x["lo"] for x in comp["words"] if x["word"] == w), hi=next(x["hi"] for x in comp["words"] if x["word"] == w)) for w in ("alpha", "beta")]
    job = dict(words=["alpha", "beta"], weights=r2.plan_words(["alpha", "beta"], h.ctx.plan_model, h.ctx.vocab)["weights"])
    want, _ = r2.render_form(dict(pieces=pieces, parts=parts_from_pieces(pieces)), job, predicted_seconds(h.ctx.duration, row["text"]), h.ctx.model, timing="sign-fitted", sample_id="x")
    covered = load_state(h.store, row["state_key"], row["state_sha256"])
    composed = load_state(h.store, comp["state_key"], comp["state_sha256"])
    assert np.array_equal(covered.channels["right_hand"].values, want.channels["right_hand"].values) and not np.array_equal(covered.channels["right_hand"].values, composed.channels["right_hand"].values)


def test_the_picks_differ_between_draws_where_a_pool_allows(monkeypatch, tmp_path):
    big = dict(POOLS, alpha=[real_record(f"alpha-{i}") for i in range(40)], beta=[real_record(f"beta-{i}") for i in range(40)], gamma=[real_record(f"gamma-{i}") for i in range(40)], delta=[real_record(f"delta-{i}") for i in range(40)])
    h = Controls(monkeypatch, tmp_path)
    h.ctx = context(pools=big)
    assert h.compose(1) == 0 and h.compose(2) == 0
    a, b = h.index(1)["rows"]["oracle"][7]["words"], h.index(2)["rows"]["oracle"][7]["words"]
    assert [w["sign"] for w in a] != [w["sign"] for w in b] and [w["word"] for w in a] == [w["word"] for w in b]


def test_shuffled_and_oracle_shuffled_use_the_same_permutation_on_the_same_seeds(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    rows = h.index()["rows"]
    for s in PC.SEEDS:
        for i in (0, 3, 6, 7, 8):
            words = list(SENTENCES[i])
            order = PC.word_order(len(words), i, s, 1)
            assert rows[f"shuffled_s{s}"][i]["order"] == order and [w["word"] for w in rows[f"shuffled_s{s}"][i]["words"]] == [words[k] for k in order]
            kept = [p for p, w in enumerate(words) if w not in ("you", "sem", "fails")]
            assert rows[f"oracle_shuffled_s{s}"][i]["order"] == PC.restricted(order, kept) == rows[f"covered_shuffled_s{s}"][i]["order"]
            assert [w["word"] for w in rows[f"covered_shuffled_s{s}"][i]["words"]] == [words[p] for p in order if p in kept]
            assert [w["word"] for w in rows[f"oracle_shuffled_s{s}"][i]["words"]] == [words[p] for p in order if p in kept]


def test_the_shuffles_use_the_same_generated_clips_as_the_composition(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    rows = h.index()["rows"]
    for s in PC.SEEDS:
        for row, comp in zip(rows[f"shuffled_s{s}"], rows["comp"]):
            assert sorted(w["clip_sha256"] for w in row["words"]) == sorted(w["clip_sha256"] for w in comp["words"]) and row["empty"] == comp["empty"]


def test_a_word_with_no_clip_is_dropped_and_the_rest_share_the_length(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    rows = h.index()["rows"]
    row = rows["oracle"][4]
    assert [w["word"] for w in row["words"]] == ["alpha"] and not row["empty"] and row["composition_words"] == ["sem", "alpha"]
    state = load_state(h.store, row["state_key"], row["state_sha256"])
    assert len(state.timestamps) == row["frames"]
    assert state.channels["right_hand"].valid.all()


def test_a_sentence_left_with_no_word_is_its_blank_as_compositions_fallback(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    rows = h.index()["rows"]
    for kind in ("oracle", "covered", "oracle_shuffled_s0", "covered_shuffled_s1"):
        row = rows[kind][ONLY_ADDED]
        assert row["empty"] and row["state_key"] is None and row["words"] == [] and row["dropped"] == [dict(word="you", reason="added_entry")]
    assert not rows["shuffled_s0"][ONLY_ADDED]["empty"] and rows["comp"][EMPTY]["empty"] and rows["oracle"][EMPTY]["dropped"] == []
    states = PC.read_states(h.store, rows["oracle"], h.refs)
    blank = states[ONLY_ADDED]
    assert len(blank.timestamps) == rows["oracle"][ONLY_ADDED]["frames"] and not blank.channels["body"].valid.any()
    assert C.still_of(blank, rows["oracle"][ONLY_ADDED], h.tokenizer) is blank


def test_a_rebuilt_composition_off_its_stored_bytes_stops_the_draw_with_no_index(monkeypatch, tmp_path):
    # the clock: another duration model plans another length
    h = Controls(monkeypatch, tmp_path)
    h.ctx = context(duration=dict(intercept=h.ctx.duration["intercept"] + .5, slope=h.ctx.duration["slope"]))
    with pytest.raises(SystemExit, match="frames, not the composition's"):
        h.compose()
    assert not h.store.exists(PC.index_key("pc/composed", 1))


def test_float_noise_from_another_cpu_passes_and_is_counted_while_a_frame_off_does_not(monkeypatch, tmp_path):
    from dataclasses import replace
    h = Controls(monkeypatch, tmp_path)
    row = C.load_export(h.store, PC.STORED_COMPOSED, 1, h.records, True)[C.PREDICTED][0]
    state = load_state(h.store, row["state_key"], row["state_sha256"])
    def moved(by):
        hand = state.channels["right_hand"]
        return replace(state, channels=dict(state.channels, right_hand=replace(hand, values=hand.values + by)))
    assert PC.same_composition(moved(PC.IDENTITY_TOLERANCE / 2), row, h.store) == (False, True)
    assert PC.same_composition(moved(PC.IDENTITY_TOLERANCE * 2), row, h.store) == (False, False)
    assert PC.same_composition(blanked([state])[0], row, h.store) == (False, False)
    assert PC.same_composition(R.cut_frames(state, np.arange(len(state.timestamps) - 1)), row, h.store) == (False, False)


def test_another_plans_weights_stop_the_draw(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    other = dict(timing=dict(TIMING, annotated=dict(alpha=2.)))
    h.ctx = context(plan_model=other)
    with pytest.raises(SystemExit, match="is not its stored state"):
        h.compose()
    assert not h.store.exists(PC.index_key("pc/composed", 1))


def test_a_stored_export_on_other_bytes_refuses(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    PC.STORED[1]["index"] = "0" * 64
    with pytest.raises(SystemExit, match="SHA-256"):
        h.compose()


def test_a_compose_is_write_once_and_a_stored_index_of_other_code_refuses(monkeypatch, tmp_path, capsys):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    capsys.readouterr()
    assert h.compose() == 0 and '"stored": true' in capsys.readouterr().out
    monkeypatch.setattr(PC, "code_digest", lambda: "d" * 64)
    with pytest.raises(SystemExit, match="written once"):
        h.compose()
    monkeypatch.setattr(PC, "code_digest", lambda: "c" * 64)
    with pytest.raises(SystemExit, match="written once"):
        h.compose(1, clips=3)


def test_a_state_key_holding_other_bytes_refuses(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    key = h.index()["rows"]["oracle"][0]["state_key"]
    h.store.put(key, b"other")
    (h.store.root / PC.index_key("pc/composed", 1)).unlink()
    with pytest.raises(SystemExit, match="written once"):
        h.compose()


@pytest.mark.parametrize("clips", [0, 6, 7])
def test_only_d20_or_a_small_smoke_composes(monkeypatch, tmp_path, clips):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    with pytest.raises(SystemExit, match="--clips"):
        h.compose(1, clips=clips)


def test_a_draw_outside_the_eight_refuses(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    with pytest.raises(SystemExit, match="pinned draws"):
        h.compose(9)
    with pytest.raises(SystemExit, match="pinned draws"):
        h.read_draws("9")


# ---- reads and controls ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("change,match", [(dict(signer_component="c00"), "c00"), (dict(partition="test"), "not a How2Sign val"), (dict(partition="train"), "not a How2Sign val"), (dict(corpus="other"), "not a How2Sign record")])
def test_a_wrong_partition_or_c00_refuses_before_any_state(monkeypatch, tmp_path, change, match):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    h.records[0].update(change)
    loaded = []
    monkeypatch.setattr(L, "load_states", lambda store, recs: loaded.append(1))
    with pytest.raises(SystemExit, match=match):
        h.anchors()
    assert not loaded


def test_a_read_needs_the_control_and_the_composition_check_before_any_state(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    loaded = []
    real_load = L.load_states
    monkeypatch.setattr(L, "load_states", lambda store, recs: loaded.append(1) or real_load(store, recs))
    with pytest.raises(SystemExit, match="no passed control"):
        h.read_draws("1")
    assert not loaded
    h.to_control()
    loaded.clear()
    with pytest.raises(SystemExit, match="has not been reproduced"):
        h.read_draws("1")
    assert not loaded


@pytest.mark.parametrize("change,word", [(dict(hypotheses=["other"] * N), "hypotheses"), (dict(nll=[[9.] * 5] * N), "NLL"), (dict(signs=[f"z{i}" for i in range(N)]), "other clips")])
def test_a_composition_not_reproduced_voids_the_read(monkeypatch, tmp_path, capsys, change, word):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    h.plant_stored_forms(1, dict(comp=change))
    h.to_control()
    assert h.comp_check(1) == 3
    out = capsys.readouterr().out
    assert word in out and "no form follows" in out and h.store.exists("pc/void/" + "c" * 64 + ".json")
    with pytest.raises(SystemExit, match="has not been reproduced"):
        h.read_draws("1")
    with pytest.raises(SystemExit, match="this code's read is void"):
        h.main("preflight", "--out-dir", str(tmp_path / "w2"), "--final-key", "pc/report.json", "--void-prefix", "pc/void")


def test_the_composition_check_reads_the_stored_export_before_anything_is_composed(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    h.plant_stored_forms(1)
    h.to_control()
    assert not h.store.exists(PC.index_key("pc/composed", 1))
    assert h.main("read", "--forms", "comp-check", "--draws", "1", "--out-dir", str(h.forms)) == 0
    assert h.main("comp-check", "--draw", "1", "--out-dir", str(h.forms), "--void-prefix", "pc/void") == 0
    assert not h.store.exists(PC.index_key("pc/composed", 1)) and json.loads((h.forms / "comp-check.json").read_text())["reproduced"] is True


def test_a_stored_form_on_other_bytes_exits_1_with_no_void(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    h.plant_stored_forms(1)
    h.to_control()
    PC.STORED[1]["comp"] = "0" * 64
    assert h.main("read", "--forms", "comp-check", "--draws", "1", "--composed", "pc/composed", "--out-dir", str(h.forms)) == 0
    with pytest.raises(SystemExit, match="SHA-256"):
        h.main("comp-check", "--draw", "1", "--out-dir", str(h.forms), "--void-prefix", "pc/void")
    assert not h.store.exists("pc/void/" + "c" * 64 + ".json") and not (h.forms / "comp-check.json").exists()


@pytest.mark.parametrize("change,word", [(dict(form="d2_comp"), "stored parent read"), (dict(instrument=dict(sha256="0" * 64)), "stored parent read"), (dict(declared_read=False), "stored parent read"), (dict(code_digest="9" * 64), "stored parent read")])
def test_a_stored_form_of_another_read_refuses_though_its_hash_matches(monkeypatch, tmp_path, change, word):
    h = Controls(monkeypatch, tmp_path)
    h.plant_stored_forms(1, dict(comp=change))
    with pytest.raises(SystemExit, match=word):
        PC.load_stored_form(h.store, 1, "comp")


def test_no_other_stored_form_can_be_named(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    with pytest.raises(SystemExit, match="no stored form"):
        PC.load_stored_form(h.store, 1, "comp_true")


def test_a_read_takes_one_kind_and_only_its_own_forms(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    with pytest.raises(SystemExit, match="give --forms or --draws"):
        h.main("read", "--out-dir", str(h.forms))
    with pytest.raises(SystemExit, match="only control"):
        h.main("read", "--forms", "c20", "--out-dir", str(h.forms))
    with pytest.raises(SystemExit, match="no --draws"):
        h.main("read", "--forms", "control", "--draws", "1", "--out-dir", str(h.forms))
    with pytest.raises(SystemExit, match="one draw"):
        h.main("read", "--forms", "comp-check", "--draws", "1,2", "--out-dir", str(h.forms))
    with pytest.raises(SystemExit, match="at most 5"):
        h.main("read", "--forms", "control", "--out-dir", str(h.forms), "--smoke-train", "6")


def test_every_form_reads_the_parts_jobs_candidates_and_its_still_at_its_length(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    h.to_read((1,))
    calls = [c for c in h.reader_parent.calls if PC.FORM.fullmatch(c[0] or "")]
    assert [c[0] for c in calls] == [n for kind in PC.KINDS for n in (f"d1_{kind}", f"d1_{kind}_still")]
    want = C.M.retrieval_candidates(N, 5, seed=0)
    assert all(np.array_equal(c[2], want) and c[1] == 4 for c in calls)
    for kind in PC.KINDS:
        form = json.loads((h.forms / f"form-d1_{kind}.json").read_text())
        still = json.loads((h.forms / f"form-d1_{kind}_still.json").read_text())
        assert form["frames"] == still["frames"] and form["kind"] == "parent-controls-form" and form["report_only"] is True and form["instrument"]["sha256"] == PP.UNFITTED_SHA and form["declared_read"] is True and form["code_digest"] == "c" * 64
        assert form["empty"] == still["empty"] and form["word_counts"] == still["word_counts"]
    assert json.loads((h.forms / "form-d1_oracle.json").read_text())["word_counts"][3] == 2 and json.loads((h.forms / "form-d1_shuffled_s0.json").read_text())["word_counts"][3] == 3


def test_a_read_is_write_once_and_a_resume_leaves_what_stands(monkeypatch, tmp_path, capsys):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    h.to_read((1,))
    with pytest.raises(SystemExit, match="write-once"):
        h.read_draws("1")
    capsys.readouterr()
    assert h.read_draws("1", "--resume") == 0 and '"read": 0' in capsys.readouterr().out


def test_a_row_of_another_clip_or_a_short_index_refuses_the_read(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    assert h.compose() == 0
    h.records[3]["text"] = "another sentence"
    with pytest.raises(SystemExit, match="another clip's"):
        PC.load_composed(h.store, "pc/composed", 1, h.records, True)
    with pytest.raises(SystemExit, match="clips, not"):
        PC.load_composed(h.store, "pc/composed", 1, h.records[:3], True)
    with pytest.raises(SystemExit, match="not composed"):
        PC.load_composed(h.store, "pc/composed", 2, h.records, True)


# ---- the mutation checks: a wrong still or clock changes a result, or is refused --------------------------------------------------------------

def built(h, k, kind, still_kind=None):
    forms = {p.stem[5:]: json.loads(p.read_text()) for p in h.forms.glob("form-*.json")}
    idx = np.random.default_rng(0).integers(0, N, (200, N))
    anchors = {a: forms[a] for a in C.ANCHORS}
    base = {k: {x: PC.load_stored_form(h.store, k, x) for x in ("comp", "still", "blank")}}
    arrays, _, _ = PC.view_arrays({k: (forms[f"d{k}_{kind}"], forms[f"d{k}_{still_kind or kind}_still"])}, base, anchors, idx, kind)
    return arrays[k], forms, base, anchors, idx


def test_a_wrong_still_changes_the_margin(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    h.to_read((1,))
    right, forms, base, anchors, idx = built(h, 1, "oracle")
    want = (MARGIN["oracle"] - STILL_MARGIN) * ORACLE_FILLED / N
    assert abs(right["margin"].mean() - want) < 1e-4
    # the still of another form at the same length: the same margin arithmetic, another result for a planted other still
    other = json.loads(json.dumps(forms["d1_oracle_still"]))
    other["nll"] = [[n[0]] + [n[0] + .25] * 4 for n in other["nll"]]
    wrong, _, _ = PC.view_arrays({1: (forms["d1_oracle"], other)}, base, anchors, idx, "oracle")
    assert abs(wrong[1]["margin"].mean() - right["margin"].mean()) > .05


def test_a_still_of_another_length_or_a_form_off_the_compositions_clock_refuses(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    h.to_read((1,))
    _, forms, base, anchors, idx = built(h, 1, "oracle")
    short = dict(forms["d1_oracle_still"], frames=[x + 1 for x in forms["d1_oracle_still"]["frames"]])
    with pytest.raises(SystemExit, match="is not as long as"):
        PC.view_arrays({1: (forms["d1_oracle"], short)}, base, anchors, idx, "oracle")
    off = dict(forms["d1_oracle"], frames=[x + 1 for x in forms["d1_oracle"]["frames"]])
    with pytest.raises(SystemExit, match="is not as long as the composition"):
        PC.view_arrays({1: (off, dict(forms["d1_oracle_still"], frames=off["frames"]))}, base, anchors, idx, "oracle")


def test_a_wrong_clock_moves_the_built_form(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    row = C.load_export(h.store, PC.STORED_COMPOSED, 1, h.records, True)[C.PREDICTED][0]
    clips = {w["clip_sha256"]: load_state(h.store, w["clip_key"], w["clip_sha256"]) for w in row["words"]}
    real = PC.RealClips(POOLS, 1, real_state)
    good = PC.compose_clip(row, h.ctx, clips, real)
    assert PC.same_composition(good["comp"]["state"], row, h.store) == (True, True)
    with pytest.raises(SystemExit, match="frames, not the composition's"):
        PC.compose_clip(row, context(duration=dict(intercept=-.5, slope=.3691)), clips, real)
    # the same frames at another pace: a clock that still fits the length is caught by the bytes
    wrong = PC.compose_clip(dict(row, words=[dict(row["words"][0], lo=row["words"][0]["lo"] + 1), *row["words"][1:]]), h.ctx, clips, real)
    assert PC.same_composition(wrong["comp"]["state"], row, h.store) == (False, False)


# ---- the report ------------------------------------------------------------------------------------------------------------------------------

def test_the_report_has_its_planted_values(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    report = h.report()
    assert report["kind"] == "parent-controls-report" and report["report_only"] is True and report["declared_read"] is True and report["alerts"] == [] and report["draws"] == [1, 2] and "fewer than eight" in report["draws_note"]
    comp = {k: (.6 + .001 * k - STILL_MARGIN) * FILLED / N for k in (1, 2)}
    block = report["composition"]
    assert abs(block["margin"]["per_draw"][0] - comp[1]) < 1e-4 and abs(block["margin"]["per_draw"][1] - comp[2]) < 1e-4 and block["empty_clips"] == {"1": 1, "2": 1}
    oracle = (MARGIN["oracle"] - STILL_MARGIN) * ORACLE_FILLED / N
    assert abs(report["forms"]["oracle"]["margin"]["mean"] - oracle) < 1e-4 and report["forms"]["oracle"]["empty_clips"] == {"1": 2, "2": 2} and report["forms"]["oracle"]["margin"]["draws"] == 2
    real = .4
    assert abs(report["forms"]["oracle"]["share_of_real"]["mean"] - oracle / real) < 1e-3
    covered = (MARGIN["covered"] - STILL_MARGIN) * ORACLE_FILLED / N
    assert abs(report["forms"]["covered"]["margin"]["mean"] - covered) < 1e-4
    for s in PC.SEEDS:
        assert abs(report["forms"][f"shuffled_s{s}"]["margin"]["mean"] - (MARGIN[f"shuffled_s{s}"] - STILL_MARGIN) * FILLED / N) < 1e-4
    group = report["groups"]["shuffled"]
    assert group["seeds"] == [0, 1] and abs(group["margin"]["mean"] - (.555 - STILL_MARGIN) * FILLED / N) < 1e-4 and group["margin"]["draws"] == 2
    assert abs(report["groups"]["oracle_shuffled"]["margin"]["mean"] - (.255 - STILL_MARGIN) * ORACLE_FILLED / N) < 1e-4
    assert abs(report["groups"]["covered_shuffled"]["margin"]["mean"] - (.43 - STILL_MARGIN) * ORACLE_FILLED / N) < 1e-4
    assert abs(report["groups"]["covered_shuffled"]["share_of_nearest"]["mean"] - (.43 - STILL_MARGIN) * ORACLE_FILLED / N / .5) < 1e-3 and report["groups"]["shuffled"]["share_of_nearest"]["draws"] == 2
    assert report["stage"] == dict(planned_draws=[1, 2], draws=[1, 2], stop_reason="not_recorded", elapsed_seconds=None, budget_seconds=None) and report["rerun_cause"] is None and report["own_path_anchors"] == dict(real="real_two_view", nearest="nearest_two_view", still="still_two_view", blank="blank") and report["parent_on_mirror_draws"] is None
    assert report["forms"]["oracle"]["share_of_two_view_real"]["mean"] == pytest.approx(report["forms"]["oracle"]["share_of_real"]["mean"], abs=1e-3)
    assert report["forms"]["oracle"]["still_raw"]["mean"] > 0 and report["oracle_coverage"]["1"]["ci95"]["share_kept"][0] <= .8 <= report["oracle_coverage"]["1"]["ci95"]["share_kept"][1]
    assert report["oracle_coverage"]["1"]["words_kept"] == 16 and report["identity"]["1"]["byte_identical"] == FILLED and report["stored"]["digest"] == "b" * 64 and report["cost_estimate"]["measured"] is False
    assert report["disclosures"]["humans"].startswith("None of this shows human comprehension") and report["reading"] == PC.READING


def test_the_paired_differences_are_on_the_same_draws_and_clips(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    report = h.report()
    paired = report["paired"]
    comp = np.array([(.6 + .001 * k - STILL_MARGIN) * FILLED / N for k in (1, 2)])
    oracle = (MARGIN["oracle"] - STILL_MARGIN) * ORACLE_FILLED / N
    covered = (MARGIN["covered"] - STILL_MARGIN) * ORACLE_FILLED / N
    row = paired["oracle_minus_composition"]["margin"]
    assert row["per_draw"] == pytest.approx(list(oracle - comp), abs=1e-4) and row["draws"] == 2 and row["clips_ci95"][0] <= row["mean"] <= row["clips_ci95"][1]
    assert paired["oracle_minus_covered"]["margin"]["mean"] == pytest.approx(oracle - covered, abs=1e-4)
    shuffled = (.555 - STILL_MARGIN) * FILLED / N
    assert paired["shuffled_minus_composition"]["margin"]["mean"] == pytest.approx(shuffled - comp.mean(), abs=1e-4)
    oracle_shuffled = (.255 - STILL_MARGIN) * ORACLE_FILLED / N
    assert paired["oracle_shuffled_minus_oracle"]["margin"]["mean"] == pytest.approx(oracle_shuffled - oracle, abs=1e-4)
    assert paired["covered_minus_composition"]["margin"]["mean"] == pytest.approx(covered - comp.mean(), abs=1e-4)
    bleu = paired["oracle_minus_composition"]["bleu"]
    assert bleu["per_draw"][0] == pytest.approx(report["forms"]["oracle"]["bleu"]["per_draw"][0] - report["composition"]["bleu"]["per_draw"][0], abs=1e-3)
    for name in ("share_of_real", "bleu_gap_share", "likelihood_gap_share"):
        assert name in paired["oracle_minus_covered"]
    assert set(paired["oracle_minus_composition"]["by_composition_coverage"]) == {"0.0-0.25", "0.25-0.5", "0.5-inf"}


def test_the_moved_rows_take_only_clips_whose_word_sequence_changed(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    report = h.report()
    rows = h.index()["rows"]
    moved = report["paired"]["moved"]
    assert set(moved) == {"shuffled_minus_composition", "oracle_shuffled_minus_oracle", "covered_shuffled_minus_covered"}
    # every moved clip has signing on both sides, so the planted differences come through whole, where the all-clips row is diluted by empty clips
    assert moved["covered_shuffled_minus_covered"]["margin"]["mean"] == pytest.approx(.43 - .5, abs=1e-4) and moved["oracle_shuffled_minus_oracle"]["margin"]["mean"] == pytest.approx(.255 - .3, abs=1e-4)
    assert report["paired"]["covered_shuffled_minus_covered"]["margin"]["mean"] == pytest.approx((.43 - .5) * ORACLE_FILLED / N, abs=1e-4)
    # a clip of one word, or one whose order left its word sequence as it was, is not moved
    mask = PC.moved_mask(rows["oracle_shuffled_s0"])
    assert not mask[ONLY_ADDED] and not mask[EMPTY] and not mask[4] and not mask[5] and mask[0]
    assert PC.moved_mask([dict(words=[dict(word="a"), dict(word="a")], order=[1, 0]), dict(words=[dict(word="b"), dict(word="a")], order=[1, 0]), dict(words=[dict(word="a")], order=[0])]).tolist() == [False, True, False]
    assert moved["covered_shuffled_minus_covered"]["moved_clips_per_form"] == pytest.approx(np.mean([PC.moved_mask(h.index(k)["rows"][f"covered_shuffled_s{s}"]).sum() for k in (1, 2) for s in PC.SEEDS]), abs=.1)


def row_of(margin, bleu):
    return dict(margin=dict(t_ci95=margin), bleu=dict(t_ci95=bleu))


@pytest.mark.parametrize("margin,bleu,want", [([.1, .2], [.1, .2], "above"), ([-.2, -.1], [-.2, -.1], "below"), ([-.1, .2], [.1, .2], "unresolved"), ([.1, .2], [-.1, .2], "unresolved"), ([.1, .2], [-.2, -.1], "unresolved"), ([0., .2], [.1, .2], "unresolved"), ([-.2, 0.], [-.2, -.1], "unresolved"), (None, [.1, .2], "unresolved"), ([.1, .2], None, "unresolved"), ([float("nan"), .2], [.1, .2], "unresolved")])
def test_a_contrast_is_decided_on_both_t_intervals(margin, bleu, want):
    assert PC.contrast(row_of(margin, bleu)) == want


def test_the_clip_interval_decides_nothing():
    row = dict(margin=dict(t_ci95=[-.1, .2], clips_ci95=[.1, .2]), bleu=dict(t_ci95=[-.1, .2], clips_ci95=[.1, .2]))
    assert PC.contrast(row) == "unresolved"
    assert PC.contrast(dict(margin=dict(t_ci95=[.1, .2], clips_ci95=[-1., 1.]), bleu=dict(t_ci95=[.1, .2], clips_ci95=[-1., 1.]))) == "above"


@pytest.mark.parametrize("fmt,gen,cell", [("above", "above", "real clips read higher"), ("above", "below", "the middle case"), ("below", "above", "the words the oracle lacks and the frames they free"), ("below", "below", "generated clips read higher"), ("above", "unresolved", "unresolved"), ("unresolved", "below", "unresolved"), ("unresolved", "unresolved", "unresolved")])
def test_every_outcome_is_a_cell_or_unresolved(fmt, gen, cell):
    bounds = dict(above=[.1, .2], below=[-.2, -.1], unresolved=[-.1, .1])
    make = lambda x: row_of(bounds[x], bounds[x])      # noqa: E731
    paired = dict(form_minus_real=dict(oracle=make(fmt)), form_minus_nearest=dict(oracle=make("unresolved")), oracle_minus_covered=make(gen), covered_minus_composition=make("below"), moved=dict(shuffled_minus_composition=dict(moved_clips_per_form=0.)))
    out = PC.reading_contrasts(paired, "parent")
    assert cell in out["cell"] and out["format"] == fmt and out["generated_against_real_clips"] == gen and out["format_like"] == "real_two_view" and out["gates_nothing"] is True and out["shuffle_on_moved_clips"] == dict(shuffled_minus_composition="unresolved")
    assert out["unresolved"] == [n for n, v in (("format", fmt), ("generated_against_real_clips", gen)) if v == "unresolved"]
    swapped = dict(paired, form_minus_real=dict(oracle=make("unresolved")), form_minus_nearest=dict(oracle=make(fmt)))
    mirror = PC.reading_contrasts(swapped, "mirror")
    assert mirror["format"] == fmt and mirror["format_like"] == "nearest_monocular" and mirror["format_beside"] == dict(like="real_monocular", contrast="unresolved")


def test_the_report_states_its_contrasts_by_the_rule(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    report = h.report()
    # planted: each form's levels are the same on both draws, so every t interval sits at the difference
    reading = report["reading_contrasts"]
    assert reading["format"] == "below" and reading["generated_against_real_clips"] == "below" and "generated clips read higher" in reading["cell"] and reading["unresolved"] == []


def test_one_draw_leaves_every_contrast_unresolved(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    reading = h.report((1,))["reading_contrasts"]
    assert reading["format"] == reading["generated_against_real_clips"] == reading["cell"] == "unresolved" and set(reading["unresolved"]) == {"format", "generated_against_real_clips"}


def test_each_form_against_nearest_is_on_the_instruments_own_path(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    report = h.report()
    rows = report["paired"]["form_minus_nearest"]
    assert set(rows) == {"composition", "oracle", "covered", "shuffled", "oracle_shuffled", "covered_shuffled"} and set(rows["oracle"]) == {"margin", "bleu"}
    oracle = (MARGIN["oracle"] - STILL_MARGIN) * ORACLE_FILLED / N
    assert rows["oracle"]["margin"]["mean"] == pytest.approx(oracle - (.6 - .1), abs=1e-4)
    assert rows["oracle"]["bleu"]["per_draw"][0] == pytest.approx(report["forms"]["oracle"]["bleu"]["per_draw"][0] - report["anchors"]["nearest_two_view"]["bleu"], abs=1e-2)
    mirror = h.mirror_report()
    assert mirror["paired"]["form_minus_nearest"]["oracle"]["margin"]["mean"] == pytest.approx(oracle - (.75 - .15), abs=1e-4)
    assert mirror["paired"]["form_minus_nearest"]["oracle"]["bleu"]["per_draw"][0] == pytest.approx(mirror["forms"]["oracle"]["bleu"]["per_draw"][0] - mirror["anchors"]["nearest_monocular"]["bleu"], abs=1e-2)
    assert mirror["reading_contrasts"]["format_like"] == "nearest_monocular" and mirror["reading_contrasts"]["format_beside"]["like"] == "real_monocular"


def test_the_moved_rows_bootstrap_reads_the_mask_on_both_metrics():
    """Mine and its partner differ only on clips that did not move: every moved row, and every resample of it, is exactly zero."""
    stats = np.stack([M_row(i % 3) for i in range(8)])
    moved = np.array([True, True, True, True, False, False, False, False])
    same = dict(margin=np.array([.1, .2, .3, .4, .5, .6, .7, .8]), stats=stats)
    other_stats = stats.copy()
    other_stats[~moved] = M_row(4)
    other = dict(margin=np.where(moved, same["margin"], 9.), stats=other_stats)
    idx = np.random.default_rng(0).integers(0, 8, (200, 8))
    rows = PC.moved_rows([{1: same}, {1: same}], [{1: moved}, {1: moved}], [{1: other}], idx)
    assert rows["moved_clips_per_form"] == 4. and rows["margin"]["per_draw"] == [0.] and rows["bleu"]["per_draw"] == [0.]
    assert rows["margin"]["clips_ci95"] == [0., 0.] and rows["bleu"]["clips_ci95"] == [0., 0.]


def test_each_form_against_real_is_on_the_instruments_own_path(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    report = h.report()
    rows = report["paired"]["form_minus_real"]
    assert set(rows) == {"composition", "oracle", "covered", "shuffled", "oracle_shuffled", "covered_shuffled"}
    assert rows["oracle"]["margin"]["mean"] == pytest.approx((MARGIN["oracle"] - STILL_MARGIN) * ORACLE_FILLED / N - .4, abs=1e-4)
    assert rows["oracle"]["bleu"]["per_draw"][0] == pytest.approx(report["forms"]["oracle"]["bleu"]["per_draw"][0] - report["anchors"]["real_two_view"]["bleu"], abs=1e-2) and set(rows["oracle"]) == {"margin", "bleu"}
    mirror = h.mirror_report()["paired"]["form_minus_real"]["oracle"]["margin"]["mean"]
    assert mirror == pytest.approx((MARGIN["oracle"] - STILL_MARGIN) * ORACLE_FILLED / N - (.8 - .15), abs=1e-4)


def test_a_shuffled_stills_effect_is_reported(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    effect = h.report()["paired"]["still_effect"]
    assert set(effect) == {"shuffled_against_composition", "oracle_shuffled_against_oracle", "covered_shuffled_against_covered"}
    # planted: every still alike, so the raw difference is zero and the margin over the partner's still is the margin
    row = effect["covered_shuffled_against_covered"]
    assert row["still_raw_difference"]["mean"] == pytest.approx(0., abs=1e-6) and row["margin_over_partners_still"]["mean"] == pytest.approx((.43 - STILL_MARGIN) * ORACLE_FILLED / N, abs=1e-4)
    one = dict(margin=np.array([.5, .5]), still_raw=np.array([.3, .1]), stats=np.stack([M_row(0)] * 2), real=np.ones(2), real_stats=np.stack([M_row(1)] * 2), blank_stats=np.stack([M_row(0)] * 2), likelihood_top=np.ones(2), likelihood_bottom=np.ones(2))
    other = dict(one, still_raw=np.array([.1, .1]))
    moved = PC.still_effect([{1: one}], [{1: other}], np.array([[0, 1]]))
    assert moved["still_raw_difference"]["mean"] == pytest.approx(.1) and moved["margin_over_partners_still"]["mean"] == pytest.approx(.6)


def test_the_clip_subsets_carry_real_nearest_blank_and_the_gap_share(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    block = h.report()["forms"]["oracle"]["by_clip_subset"]
    assert block["in_v1_training"]["clips"] == 4 and block["outside_v1_training"]["clips"] == N - 4 and block["in_selection_sample"]["clips"] == 2 and block["outside_training_and_selection"]["clips"] == N - 6
    row = block["outside_training_and_selection"]
    assert {"margin", "share_of_real", "bleu", "real_bleu", "nearest_bleu", "blank_bleu", "bleu_gap_share"} <= set(row)
    assert row["bleu_gap_share"] == pytest.approx((row["bleu"] - row["blank_bleu"]) / (row["real_bleu"] - row["blank_bleu"]), abs=1e-2)


def test_oracle_rows_split_by_each_clips_main_pick_corpus(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    rows = h.report()["forms"]["oracle"]["by_pick_corpus"]
    assert set(rows) <= set(PC.H.WORD_CORPORA) and sum(v["clips_per_draw"] for v in rows.values()) == ORACLE_FILLED
    assert all(v["oracle_margin"] == pytest.approx(MARGIN["oracle"] - STILL_MARGIN, abs=1e-4) and v["covered_margin"] == pytest.approx(MARGIN["covered"] - STILL_MARGIN, abs=1e-4) for v in rows.values())


def test_a_composed_clip_with_a_gap_raises_an_alert(monkeypatch, tmp_path):
    assert PC.validity_alerts("x", {1: dict(valid_frame_share=[1., 0.], empty=[False, True])}) == []
    alert = PC.validity_alerts("x", {1: dict(valid_frame_share=[.9, 0.], empty=[False, True])})
    assert "draw 1 x: 1 clips" in alert[0] and "0.9" in alert[0]
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    h.to_read((1,))
    path = h.forms / "form-d1_oracle.json"
    read = json.loads(path.read_text())
    read["valid_frame_share"][0] = .5
    path.write_text(json.dumps(read))
    assert h.upload() == 0
    report = h.merge("1")
    assert any("oracle" in a and "valid_frame_share" in a for a in report["alerts"])
    assert h.main("check", "--report", str(tmp_path / "report.json")) == 3


def test_the_stage_is_recorded_in_the_report(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    h.to_read((1,))
    assert h.upload() == 0
    out = tmp_path / "staged.json"
    assert h.main("merge", "--draws", "1", "--composed", "pc/composed", "--prefix", "pc/forms", "--output-file", str(out), "--planned-draws", "1,2,3", "--stop-reason", "clock_rule", "--elapsed-seconds", "123") == 0
    assert json.loads(out.read_text())["stage"] == dict(planned_draws=[1, 2, 3], draws=[1], stop_reason="clock_rule", elapsed_seconds=123, budget_seconds=None)


def test_a_rerun_cause_is_copied_into_the_report_and_a_blank_one_refuses(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    h.to_read((1,))
    assert h.upload() == 0
    out = tmp_path / "cause.json"
    assert h.main("merge", "--draws", "1", "--composed", "pc/composed", "--prefix", "pc/forms", "--output-file", str(out), "--rerun-cause", "what was wrong", "--budget-seconds", "99") == 0
    report = json.loads(out.read_text())
    assert report["rerun_cause"] == "what was wrong" and report["stage"]["budget_seconds"] == 99
    for blank in ("", "   "):
        with pytest.raises(SystemExit, match="blank"):
            h.main("merge", "--draws", "1", "--composed", "pc/composed", "--prefix", "pc/forms", "--output-file", str(tmp_path / "x.json"), "--rerun-cause", blank)
        with pytest.raises(SystemExit, match="blank"):
            h.main("preflight", "--out-dir", str(tmp_path / "r"), "--final-key", "pc/none.json", "--void-prefix", "pc/void", "--rerun-cause", blank)


def test_a_finished_stage_is_restored_on_a_new_pod_only_if_of_this_code_environment_and_plan(monkeypatch, tmp_path, capsys):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    def plant(**change):
        report = dict(dict(kind="parent-controls-report", code_digest="c" * 64, environment=dict(gpu="test"), stage=dict(planned_draws=[1, 2], draws=[1, 2], stop_reason="clock_rule", elapsed_seconds=500, budget_seconds=900)), **change)
        h.store.put("pc/r.json", json.dumps(report).encode())
        return json.dumps(report).encode()
    new = tmp_path / "new-pod"
    state = lambda **k: h.main("stage-state", "--key", "pc/r.json", "--file", str(new / "r.json"), "--record", str(new / "stage.txt"), "--planned-draws", k.get("planned", "1,2"))      # noqa: E731
    raw = plant()
    capsys.readouterr()
    assert state() == 0 and '"state": "restored"' in capsys.readouterr().out
    assert (new / "r.json").read_bytes() == raw and (new / "stage.txt").read_text() == "1,2 clock_rule 500 900\n"
    assert state() == 0 and '"state": "done"' in capsys.readouterr().out
    for change, word, planned in ((dict(code_digest="d" * 64), "code", "1,2"), (dict(environment=dict(gpu="other")), "environment", "1,2"), (dict(kind="other"), "kind", "1,2"), ({}, "planned draws", "1,2,3")):
        (new / "r.json").unlink(missing_ok=True)
        (new / "stage.txt").unlink(missing_ok=True)
        plant(**change)
        with pytest.raises(SystemExit, match=word):
            state(planned=planned)
        assert not (new / "r.json").exists()


def test_a_paired_difference_of_a_form_with_itself_is_zero():
    idx = np.random.default_rng(0).integers(0, 4, (50, 4))
    arrays = {1: dict(margin=np.array([.1, .2, .3, .4]), still_raw=np.array([.1, .1, .2, .2]), real=np.array([1., 1., 1., 1.]), nearest=np.array([1., 1., 1., 1.]), stats=np.stack([M_row(i) for i in range(4)]), blank_stats=np.stack([M_row(0)] * 4), real_stats=np.stack([M_row(3)] * 4), likelihood_top=np.ones(4), likelihood_bottom=np.ones(4), coverage=np.array([.1, .3, .6, .7]), non_empty=np.ones(4, bool))}
    row = PC.paired_rows([arrays], [arrays], idx, arrays)
    assert row["margin"]["per_draw"] == [0.] and row["bleu"]["per_draw"] == [0.] and row["margin"]["clips_ci95"] == [0., 0.]
    assert row["by_composition_coverage"]["0.5-inf"]["clips_per_draw"] == 2. and row["by_composition_coverage"]["0.5-inf"]["margin"] == 0.
    assert PC.band_mask(arrays[1], .25, .5).tolist() == [False, True, False, False]


def M_row(i):
    """BLEU statistics of a short hypothesis against its reference, a different amount right per ``i``."""
    return C.M.bleu_statistics([" ".join(f"w{j}" for j in range(4 + i))], [" ".join(f"w{j}" for j in range(8))])[0]


def test_the_coverage_bands_read_a_forms_own_coverage_and_compositions(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    report = h.report()
    # 15 tokens a sentence: only the four-word clip passes a quarter, in the composition and in the oracle, which keeps all four
    composition = report["composition"]["by_composition_coverage"]
    assert composition["0.0-0.25"]["clips_per_draw"] == FILLED - 1 and composition["0.25-0.5"]["clips_per_draw"] == 1. and composition["0.5-inf"] == dict(clips_per_draw=0.)
    own = report["forms"]["oracle"]["by_own_coverage"]["0.0-0.25"]
    assert own["clips_per_draw"] == ORACLE_FILLED - 1 and own["margin"] == pytest.approx(MARGIN["oracle"] - STILL_MARGIN, abs=1e-4) and "bleu" in own
    # by composition's bands, the same clips: the clip the oracle left blank stays in, at zero
    fixed = report["forms"]["oracle"]["by_composition_coverage"]["0.0-0.25"]
    assert fixed["clips_per_draw"] == FILLED - 1 and fixed["margin"] == pytest.approx((MARGIN["oracle"] - STILL_MARGIN) * (ORACLE_FILLED - 1) / (FILLED - 1), abs=1e-4)


def test_per_signer_rows_are_report_only_and_carry_bleu(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    block = h.report()["forms"]["oracle"]
    assert set(block["per_signer"]) == {"c01", "c03", "c05"} and sum(v["clips"] for v in block["per_signer"].values()) == N and "report only" in block["per_signer_note"]
    assert all({"margin", "share_of_real", "bleu", "bleu_gap_share"} <= set(v) for v in block["per_signer"].values())


def test_the_hypothesis_rows_read_loops_and_content_words(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    report = h.report()
    row = report["forms"]["oracle"]["hypotheses"]
    assert set(row) == {"loop_share", "content_word_overlap", "words"} and row["loop_share"]["mean"] == 0. and 0 < row["content_word_overlap"]["mean"] < 1
    assert report["real_hypotheses"]["content_word_overlap"]["mean"] > row["content_word_overlap"]["mean"] and report["groups"]["shuffled"]["hypotheses"]["words"]["draws"] == 2


def test_content_overlap_is_the_share_of_the_references_content_words():
    from signtranslator.generation.compositional import STOP_WORDS
    stop = sorted(STOP_WORDS)[:2]
    assert PC.content_overlap("house river", f"{stop[0]} house river {stop[1]} mountain") == pytest.approx(2 / 3)
    assert PC.content_overlap(" ".join(stop), "house") == 0. and PC.content_overlap("house", " ".join(stop)) is None
    assert PC.content_overlap("House, RIVER!", "house river") == 1.


def test_the_nan_means_leave_out_clips_without_a_value_and_stay_quiet():
    values = np.array([1., np.nan, 3.])
    assert PC.nan_means(values, np.array([[0, 1, 2]]))[0] == 2.
    out = PC.nan_means(values, np.array([[1, 1, 1], [0, 0, 2]]))
    assert np.isnan(out[0]) and out[1] == pytest.approx(5 / 3)


def test_order_stats_see_a_repeated_word_and_a_kept_identity():
    def row(words, order):
        return dict(words=[dict(word=w) for w in words], order=order)
    rows = {k: [] for k in PC.KINDS}
    rows["shuffled_s0"] = [row(["a", "a"], [1, 0]), row(["b", "a"], [1, 0]), row(["a"], [0])]
    rows["oracle_shuffled_s0"] = [row(["a", "b"], [0, 1])]
    out = PC.order_stats(rows)
    assert out["shuffled_s0"] == dict(clips_with_two_words=2, unchanged_word_sequence=1, identity_order=0) and out["oracle_shuffled_s0"] == dict(clips_with_two_words=1, unchanged_word_sequence=1, identity_order=1)
    assert "oracle" not in out and PC.sorted_words(row(["b", "a"], [1, 0])) == ["a", "b"]


def test_a_stale_form_or_a_form_of_another_run_refuses_the_merge(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    h.to_read()
    assert h.upload() == 0
    monkeypatch.setattr(PC, "code_digest", lambda: "e" * 64)
    with pytest.raises(SystemExit, match="a stale read"):
        h.merge()


@pytest.mark.parametrize("field,value,match", [("declared_read", False, "smoke mixed"), ("signs", [f"z{i}" for i in range(N)], "other clips"), ("instrument", dict(sha256="0" * 64), "not read by the parent"), ("form", "d1_covered", "holds form")])
def test_forms_of_another_run_refuse_the_merge(monkeypatch, tmp_path, field, value, match):
    h = Controls(monkeypatch, tmp_path)
    h.to_read()
    path = h.forms / "form-d1_oracle.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), **{field: value})))
    assert h.upload() == 0
    with pytest.raises(SystemExit, match=match):
        h.merge()


def test_a_missing_form_control_or_check_refuses_the_merge(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    h.to_read()
    assert h.upload() == 0
    with pytest.raises(SystemExit, match="d3_oracle is missing"):
        h.merge("1,2,3")
    (h.store.root / "pc/forms/form-d2_oracle_shuffled_s1_still.json").unlink()
    with pytest.raises(SystemExit, match="d2_oracle_shuffled_s1_still is missing"):
        h.merge()
    (h.store.root / "pc/forms/comp-check.json").unlink()
    with pytest.raises(SystemExit, match="composition check"):
        h.merge("1")
    (h.store.root / "pc/forms/control.json").unlink()
    with pytest.raises(SystemExit, match="no passed control"):
        h.merge("1")


def test_a_failed_control_or_check_on_record_refuses_the_merge(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    h.to_read()
    path = h.forms / "comp-check.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), reproduced=False)))
    assert h.upload() == 0
    with pytest.raises(SystemExit, match="composition check"):
        h.merge()


def test_no_form_and_an_empty_draw_list_refuse_the_merge(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    with pytest.raises(SystemExit, match="no form"):
        h.merge()
    h = Controls(monkeypatch, tmp_path / "b") if (tmp_path / "b").mkdir() is None else None
    h.to_read()
    assert h.upload() == 0
    with pytest.raises(SystemExit, match="names no draw"):
        h.main("merge", "--draws", ",", "--composed", "pc/composed", "--prefix", "pc/forms", "--output-file", str(tmp_path / "x.json"))


def test_a_merge_and_a_put_report_accept_identical_bytes_and_refuse_others(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    h.report()
    path = tmp_path / "report.json"
    first = path.read_bytes()
    h.merge()
    assert path.read_bytes() == first
    assert h.main("put-report", "--file", str(path), "--key", "pc/report.json") == 0
    assert h.main("put-report", "--file", str(path), "--key", "pc/report.json") == 0 and h.store.get("pc/report.json") == first
    with pytest.raises(SystemExit, match="finished"):
        h.main("preflight", "--out-dir", str(tmp_path / "w3"), "--final-key", "pc/report.json", "--void-prefix", "pc/void")
    other = tmp_path / "other.json"
    other.write_bytes(first + b" ")
    with pytest.raises(SystemExit, match="holds other bytes"):
        h.main("put-report", "--file", str(other), "--key", "pc/report.json")
    path.write_bytes(first + b" ")
    with pytest.raises(SystemExit, match="holds other bytes"):
        h.merge()
    assert h.store.get("pc/report.json") == first


def test_a_stage_is_done_only_when_its_key_holds_its_local_bytes(monkeypatch, tmp_path, capsys):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    path = tmp_path / "r.json"
    state = lambda: h.main("stage-state", "--key", "pc/r.json", "--file", str(path))      # noqa: E731
    assert state() == 5
    path.write_bytes(b"{}")
    assert state() == 5
    h.store.put("pc/r.json", b"{}")
    assert state() == 0 and '"state": "done"' in capsys.readouterr().out
    path.write_bytes(b"[]")
    with pytest.raises(SystemExit, match="finished elsewhere"):
        state()


def test_a_rerun_of_new_code_writes_under_its_digest_while_the_void_stands(monkeypatch, tmp_path, capsys):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    assert h.compose() == 0
    h.plant_stored_forms(1, dict(comp=dict(hypotheses=["other"] * N)))
    h.to_control()
    assert h.comp_check(1) == 3
    out = capsys.readouterr().out
    assert "this code never reads again" in out and "RERUN_CAUSE" in out and "runs/<code digest>/" in out
    with pytest.raises(SystemExit, match="this code's read is void"):
        h.main("preflight", "--out-dir", str(tmp_path / "r0"), "--final-key", "pc/report.json", "--void-prefix", "pc/void")
    new = "d" * 64
    monkeypatch.setattr(PC, "code_digest", lambda: new)
    # the old prefix refuses new code, and that was the dead end
    with pytest.raises(SystemExit, match="written once"):
        h.compose(1)
    with pytest.raises(SystemExit, match="give --rerun-cause"):
        h.main("preflight", "--out-dir", str(tmp_path / "r1"), "--final-key", f"pc/runs/{new}/report.json", "--void-prefix", "pc/void")
    assert h.main("preflight", "--out-dir", str(tmp_path / "r2"), "--final-key", f"pc/runs/{new}/report.json", "--void-prefix", "pc/void", "--rerun-cause", "the cause") == 0
    assert json.loads((tmp_path / "r2" / "forms" / "rerun.json").read_text())["cause"] == "the cause"
    assert h.main("compose", "--draw", "1", "--clips", str(N), "--work", str(tmp_path / "work"), "--prefix", f"pc/runs/{new}/composed", "--model-file", "unused") == 0


def test_the_digest_command_prints_the_digest(monkeypatch, tmp_path, capsys):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    capsys.readouterr()
    assert h.main("digest") == 0 and capsys.readouterr().out.strip() == "c" * 64


def test_a_check_passes_a_clean_report_and_refuses_broken_plumbing(monkeypatch, tmp_path, capsys):
    h = Controls(monkeypatch, tmp_path)
    h.report()
    path = tmp_path / "report.json"
    assert h.main("check", "--report", str(path)) == 0
    clean = json.loads(path.read_text())
    for change, word in ((dict(report_only=False), "report_only"), (dict(checks=dict(clean["checks"], controls=dict(comp_check=dict(reproduced=False)))), "passed control"), (dict(checks=dict(clean["checks"], blank_identical_to_real=1.)), "blank reads"), (dict(alerts=["draw 1: its oracle holds no word"]), "holds no word")):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(dict(clean, **change)))
        assert h.main("check", "--report", str(bad)) == 3
        assert word in capsys.readouterr().out


def test_the_inputs_step_checks_every_pin_and_prints_no_score(monkeypatch, tmp_path, capsys):
    h = Controls(monkeypatch, tmp_path, compose_draws=PC.DRAWS)
    for k in PC.DRAWS:
        h.plant_stored_forms(k)
    assert h.anchors() == 0
    h.plant_stored_anchors()
    assert h.mirror_anchors() == 0
    h.plant_mirror_stored()
    h.store.put(PC.STORED_REPORT, b"{}")
    monkeypatch.setattr(PC, "STORED_REPORT_SHA", hashlib.sha256(b"{}").hexdigest())
    model = dict(timing=TIMING)
    model["sha256"] = cg.model_sha256(model)
    monkeypatch.setattr(C, "MODEL_SHA", model["sha256"])
    monkeypatch.setattr(fit, "fit_model", lambda *a, **k: (model, None))
    monkeypatch.setattr(fit.agc, "_store", lambda spec: None)
    checked = []
    monkeypatch.setattr(PC, "check_decoder", lambda store, key: checked.append(key) or dict(checkpoint_key=key))
    file = tmp_path / "w-m" / "model.json"
    capsys.readouterr()
    assert h.main("inputs", "--model-file", str(file)) == 0
    out = capsys.readouterr().out
    assert file.exists() and "bleu" not in out and "hypotheses" not in out and "margin" not in out
    assert checked == ["prior/w14b.pt"]
    # the mirror's stored forms must be D20's clips, in its order, and the stored compositions valid on every point
    h.plant_mirror_stored(dict(blank=dict(signs=[f"z{i}" for i in range(N)])))
    with pytest.raises(SystemExit, match="holds other clips than D20"):
        h.main("inputs", "--model-file", str(file))
    h.plant_mirror_stored(dict(still_monocular=dict(references=["other"] * N)))
    with pytest.raises(SystemExit, match="holds other clips than D20"):
        h.main("inputs", "--model-file", str(file))
    h.plant_mirror_stored()
    keep, raw = PC.STORED[4]["comp"], h.store.get(PC.stored_form_key(4, "comp"))
    gap = json.dumps(dict(json.loads(raw), valid_frame_share=[.5] * N)).encode()
    h.store.put(PC.stored_form_key(4, "comp"), gap)
    PC.STORED[4]["comp"] = hashlib.sha256(gap).hexdigest()
    capsys.readouterr()
    with pytest.raises(SystemExit, match="not valid on every point"):
        h.main("inputs", "--model-file", str(file))
    assert "ALERT" in capsys.readouterr().out
    h.store.put(PC.stored_form_key(4, "comp"), raw)
    PC.STORED[4]["comp"] = keep
    with pytest.raises(SystemExit, match="not NVIDIA L40S"):
        h.main("inputs", "--model-file", str(file), "--expect-gpu", "NVIDIA_L40S")
    monkeypatch.setattr(P, "environment", lambda: dict(gpu="NVIDIA L40S"))
    assert h.main("inputs", "--model-file", str(file), "--expect-gpu", "NVIDIA_L40S") == 0
    monkeypatch.setattr(PC, "MIRROR_REPORT_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        h.main("inputs", "--model-file", str(file))
    h.plant_mirror_stored(dict(blank=dict(code_digest="9" * 64)))
    with pytest.raises(SystemExit, match="stored mirror read"):
        h.main("inputs", "--model-file", str(file))
    h.plant_mirror_stored()
    PC.STORED[3]["blank"] = "0" * 64
    with pytest.raises(SystemExit, match="SHA-256"):
        h.main("inputs", "--model-file", str(file))


def smoke(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,), n=N)
    train = [dict(r, partition="train", signer_component="c02", sign=f"train-{i}") for i, r in enumerate(h.records[:5])]
    monkeypatch.setattr(L, "open_read_list", lambda store, n: (train, False, dict(list="smoke:c02-train", clips=5)) if n else (h.records, True, dict(list="t", sha256=L.D20_SHA, clips=N)))
    monkeypatch.setattr(L, "load_states", lambda store, recs: h.refs[:len(recs)])
    return h


def test_a_smoke_reads_d20_sentences_composed_against_train_references(monkeypatch, tmp_path):
    h = smoke(monkeypatch, tmp_path)
    assert h.compose(1, clips=5) == 0
    assert len(h.index()["rows"]["oracle"]) == 5
    assert h.main("read", "--forms", "control", "--out-dir", str(h.forms), "--smoke-train", "5") == 0
    assert h.main("control", "--out-dir", str(h.forms), "--void-prefix", "pc/void") == 0
    assert h.main("read", "--draws", "1", "--composed", "pc/composed", "--out-dir", str(h.forms), "--smoke-train", "5") == 0
    form = json.loads((h.forms / "form-d1_oracle.json").read_text())
    assert form["declared_read"] is False and all(s.startswith("train-") for s in form["signs"])
    assert (h.forms / "form-d1_comp.json").exists() and (h.forms / "form-d1_blank.json").exists() and not (h.forms / "control.json").exists()
    assert h.upload() == 0
    report = h.merge("1")
    assert report["declared_read"] is False and "SMOKE" in report["smoke"]


def test_a_stored_form_of_other_clips_refuses_the_merge(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    h.to_read()
    h.plant_stored_forms(2, dict(blank=dict(signs=[f"z{i}" for i in range(N)])))
    assert h.upload() == 0
    with pytest.raises(SystemExit, match="other clips or another order"):
        h.merge()


def test_the_cache_priors_tokenizer_must_be_this_d2(monkeypatch, tmp_path):
    import torch
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "s")
    tokenizer = tiny_tokenizer()
    weights = {k: v.clone() for k, v in tokenizer.state_dict().items()}

    def put(key, payload):
        path = tmp_path / "x.pt"
        torch.save(payload, path)
        store.put(key, path.read_bytes())
        return hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(L, "TOKENIZER_SHA", put(L.TOKENIZER, dict(ema=weights, model={})))
    put("p.pt", dict(tokenizer_key=L.TOKENIZER, tokenizer_state=weights))
    assert PC.check_decoder(store, "p.pt")["same_weights"] is True
    put("p.pt", dict(tokenizer_state=weights))
    assert PC.check_decoder(store, "p.pt")["tokenizer_key"] is None
    put("p.pt", dict(tokenizer_key="training-runs/other.pt", tokenizer_state=weights))
    with pytest.raises(SystemExit, match="decoded by another"):
        PC.check_decoder(store, "p.pt")
    changed = dict(weights)
    name = next(iter(changed))
    changed[name] = changed[name] + 1
    put("p.pt", dict(tokenizer_key=L.TOKENIZER, tokenizer_state=changed))
    with pytest.raises(SystemExit, match="decoded by another"):
        PC.check_decoder(store, "p.pt")
    put("p.pt", dict(tokenizer_key=L.TOKENIZER))
    with pytest.raises(SystemExit, match="decoded by another"):
        PC.check_decoder(store, "p.pt")


def test_the_gpu_check_takes_the_stored_forms_type(monkeypatch):
    monkeypatch.setattr(P, "environment", lambda: dict(gpu="NVIDIA RTX 6000 Ada Generation"))
    with pytest.raises(SystemExit, match="not NVIDIA L40S"):
        PC.check_gpu("NVIDIA_L40S")
    PC.check_gpu("NVIDIA_RTX_6000_Ada_Generation")
    PC.check_gpu(None)


def test_the_digest_names_this_job_and_composition_job():
    from signtranslator.eval import provenance
    assert PC.code_digest() == provenance.code_digest(extra_files=(PC.JOB_FILE, C.JOB_FILE)) and PC.PROBE == "parent-controls" and C.PROBE == "parent-comp"


def test_composition_forms_keep_their_own_kind():
    import inspect
    assert inspect.signature(C.read_one).parameters["probe"].default == "parent-comp"


# ---- the job -------------------------------------------------------------------------------------------------------------------------------

def run_job(tmp_path, fail_when=None, code=7, stage_zero=None, slow=False, hold=None, tee_fails=False, **env):
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    log.unlink(missing_ok=True)
    pattern = "*" + "*".join(f'"{x}"' for x in fail_when.split("*")) + "*" if fail_when else ""
    match = f'case "$*" in {pattern}) echo "planted failure" >&2; exit {code};; esac' if fail_when else ""
    zero = f'case "$*" in *"{stage_zero}"*) exit 0;; esac' if stage_zero else ""
    sleep = f'case "$*" in *"read --draws"*"--resume"*) echo $$ >> {tmp_path / "pids"}; exec sleep 60;; esac' if slow else ""
    if hold:
        sleep = f'case "$*" in *"{hold}"*"--resume"*) /bin/sleep 2;; esac'
    (bin_dir / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\n{match}\n{sleep}\ncase "$2" in digest) echo abc123def; exit 0;; stage-state) {zero}\n exit 5;; esac\necho "ran $1 $2"\nexit 0\n')
    (bin_dir / "uv").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "timeout").write_text(f'#!/bin/bash\necho "timeout $*" >> {tmp_path / "timeouts.log"}\nshift 3\nexec "$@"\n')
    names = ["python", "uv", "timeout"]
    if tee_fails:
        (bin_dir / "tee").write_text("#!/bin/bash\ncat > /dev/null\nexit 1\n")
        names.append("tee")
    for name in names:
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    environment = dict(dict(READ_POLL="0.05"), PATH=f"{bin_dir}:/usr/bin:/bin", HOME=str(tmp_path), VENV_BIN=str(bin_dir), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path), WORK=str(tmp_path / "work"), **env)
    if slow:
        return subprocess.Popen(["bash", str(JOB)], env=environment, stdout=subprocess.PIPE, text=True), log
    done = subprocess.run(["bash", str(JOB)], env=environment, capture_output=True, text=True, timeout=60)
    return done, log.read_text() if log.exists() else ""


def done_line(result) -> int:
    lines = [x for x in result.stdout.splitlines() if x.startswith("PARENT-CONTROLS-DONE ")]
    assert len(lines) == 1 and lines[0] == result.stdout.strip().splitlines()[-1]
    return int(lines[0].split()[1])


def commands(calls):
    return [PC.build_parser().parse_args(x.split()[1:]).command for x in calls.splitlines()]


SMOKE_STEPS = ["read", "control", "read", "upload", "merge", "put-report", "check"]


def draws_read(calls, instrument):
    return sorted(int(x.split("--draws ")[1].split()[0]) for x in calls.splitlines() if x.split()[1] == "read" and "--resume" in x and "--draws" in x and "comp-check" not in x and f"--instrument {instrument}" in x)


def test_the_job_smokes_composes_controls_then_reads_the_parent_then_the_mirror(tmp_path):
    result, calls = run_job(tmp_path)
    assert result.returncode == 0 and done_line(result) == 0
    got = commands(calls)
    assert got[:25] == ["digest", "inputs", "compose", *SMOKE_STEPS, *SMOKE_STEPS, "stage-state", "stage-state", "preflight", "preflight", "read", "control", "read", "comp-check"]
    assert got[25] == "compose", "the parent's controls come before the compose is paid for"
    assert got.count("compose") == 9 and got.count("comp-check") == 1 and got[-5:] == ["upload", "merge", "put-report", "check", "upload-log"]
    lines = calls.splitlines()
    assert all(line.startswith("scripts/parent_controls.py ") for line in lines)
    assert "--expect-gpu NVIDIA_L40S" in lines[1] and "--clips 5" in lines[2] and "--smoke-train 5" in lines[3] and "smoke" in lines[2]
    assert draws_read(calls, "parent") == list(PC.DRAWS) and draws_read(calls, "mirror") == [1, 2, 3]
    merges = [x for x in lines if x.split()[1] == "merge"]
    assert "--draws 1,2,3,4,5,6,7,8 --instrument parent --planned-draws 1,2,3,4,5,6,7,8 --stop-reason all_planned" in merges[-2] and "--draws 1,2,3 --instrument mirror --planned-draws 1,2,3 --stop-reason all_planned" in merges[-1] and "--prefix acquisition/2026-10-17-v1/youtube-asl-v1/round20/parent-controls/forms-mirror" in merges[-1] and "--parent-forms acquisition/2026-10-17-v1/youtube-asl-v1/round20/parent-controls/forms " in merges[-1]
    assert "--parent-forms" not in merges[-2] and "--budget-seconds 10800" in merges[-2] and "--budget-seconds 14400" in merges[-1]
    puts = [x for x in lines if x.split()[1] == "put-report"]
    assert "round20/parent-controls/report.json" in puts[-2] and "round20/parent-controls/report-mirror.json" in puts[-1]
    first_mirror = next(i for i, x in enumerate(lines) if "--instrument mirror" in x and "work/d20" in x)
    last_parent = max(i for i, x in enumerate(lines) if "--instrument parent" in x and "work/d20" in x)
    assert last_parent < first_mirror and next(i for i, x in enumerate(lines) if x.split()[1] == "comp-check") < next(i for i, x in enumerate(lines) if "--instrument parent" in x and "--resume" in x and "--draws" in x)
    timeouts = (tmp_path / "timeouts.log").read_text()
    assert timeouts.count("-k 60 5400 python scripts/parent_controls.py compose") == 9
    assert timeouts.count("-k 60 3600 python scripts/parent_controls.py read --forms control") == 4 and timeouts.count("-k 60 3600 python scripts/parent_controls.py read --forms comp-check") == 1 and timeouts.count("-k 60 3600 python scripts/parent_controls.py read --draws") == 13
    assert timeouts.count("-k 60 3600 python scripts/parent_controls.py merge") == 4 and "-k 60 1800 uv pip install" in timeouts
    assert timeouts.count("python scripts/parent_controls.py") == len(lines), "every python step of the job runs under a limit"


def test_the_smoke_only_mode_stops_before_d20(tmp_path):
    result, calls = run_job(tmp_path, SMOKE="1")
    assert result.returncode == 0 and "preflight" not in calls and "SMOKE done" in result.stdout and "--instrument mirror" in calls and "--instrument parent" in calls


def test_fewer_draws_named_are_merged_as_named(tmp_path):
    result, calls = run_job(tmp_path, DRAWS="2 4 6 8", MIRROR_DRAWS="2 4")
    assert result.returncode == 0
    merges = [x for x in calls.splitlines() if x.split()[1] == "merge"]
    assert "--draws 2,4,6,8 " in merges[-2] and "--draws 2,4 " in merges[-1] and sum(x.split()[1] == "compose" for x in calls.splitlines()) == 5


@pytest.mark.parametrize("mirror", ["1 5", "2 3", "2", "1 3", "1 2 3 4 5"])
def test_mirror_draws_not_a_prefix_of_the_draws_stop_the_job_before_anything_runs(tmp_path, mirror):
    result, calls = run_job(tmp_path, DRAWS="1 2 3 4", MIRROR_DRAWS=mirror)
    assert result.returncode == 1 and "is not a prefix of DRAWS" in result.stdout and [x.split()[1] for x in calls.splitlines()] == ["upload", "upload-log"]


def test_a_mirror_prefix_passes(tmp_path):
    result, calls = run_job(tmp_path, DRAWS="3 1 2", MIRROR_DRAWS="3 1")
    assert result.returncode == 0 and draws_read(calls, "mirror") == [1, 3]


@pytest.mark.parametrize("procs", ["1", "2"])
def test_the_clock_rule_counts_draws_started_for_each_instrument_and_the_mirrors_can_act(tmp_path, procs):
    result, calls = run_job(tmp_path, PARENT_BUDGET_SECONDS="-1", BUDGET_SECONDS="-1", READ_PROCS=procs)
    assert result.returncode == 0 and "the clock rule starts no more parent draws" in result.stdout and "the clock rule starts no more mirror draws" in result.stdout
    assert draws_read(calls, "parent") == [1, 2, 3, 4] and draws_read(calls, "mirror") == [1, 2]
    merges = [x for x in calls.splitlines() if x.split()[1] == "merge"]
    assert "--draws 1,2,3,4 --instrument parent --planned-draws 1,2,3,4,5,6,7,8 --stop-reason clock_rule --elapsed-seconds" in merges[-2] and "--budget-seconds -1" in merges[-2] and "--draws 1,2 --instrument mirror --planned-draws 1,2,3 --stop-reason clock_rule" in merges[-1]


def test_the_mirror_reads_only_draws_the_parent_read(tmp_path):
    result, calls = run_job(tmp_path, PARENT_BUDGET_SECONDS="-1", MIRROR_DRAWS="1 2 3 4 5 6")
    assert result.returncode == 0 and draws_read(calls, "parent") == [1, 2, 3, 4] and draws_read(calls, "mirror") == [1, 2, 3, 4]


def test_the_clock_and_elapsed_seconds_are_cumulative_across_a_resume(tmp_path):
    (tmp_path / "work/d20").mkdir(parents=True)
    (tmp_path / "work/d20/elapsed.txt").write_text("20000\n")
    result, calls = run_job(tmp_path)
    assert result.returncode == 0 and draws_read(calls, "parent") == [1, 2, 3, 4] and draws_read(calls, "mirror") == [1, 2]
    merge = next(x for x in calls.splitlines() if x.split()[1] == "merge" and "work/d20" in x)
    assert int(merge.split("--elapsed-seconds ")[1].split()[0]) >= 20000 and int((tmp_path / "work/d20/elapsed.txt").read_text()) >= 20000


def test_a_stage_record_carries_its_budget_to_a_resumed_merge(tmp_path):
    result, first = run_job(tmp_path, fail_when="put-report*d20/report.json", PARENT_BUDGET_SECONDS="9000")
    assert result.returncode == 1 and (tmp_path / "work/d20/stage-parent.txt").read_text().split()[-1] == "9000"
    result, second = run_job(tmp_path, PARENT_BUDGET_SECONDS="1")
    merge = next(x for x in second.splitlines() if x.split()[1] == "merge" and "--instrument parent" in x and "work/d20" in x)
    assert "--budget-seconds 9000" in merge


def test_a_changed_plan_refuses_on_resume(tmp_path):
    result, _ = run_job(tmp_path, fail_when="--draws 2 --instrument parent*--resume")
    assert result.returncode == 1
    for env in (dict(DRAWS="1 2 3 4 5 6"), dict(MIRROR_DRAWS="1 2")):
        result, calls = run_job(tmp_path, **env)
        assert result.returncode == 1 and "changed since this run began" in result.stdout and "inputs" not in calls
    result, calls = run_job(tmp_path)
    assert result.returncode == 0


@pytest.mark.parametrize("cause", ["", "   "])
def test_a_blank_rerun_cause_refuses(tmp_path, cause):
    result, calls = run_job(tmp_path, RERUN_CAUSE=cause)
    assert result.returncode == 1 and "RERUN_CAUSE is blank" in result.stdout and "inputs" not in calls


@pytest.mark.parametrize("fail,word,later", [("control --instrument parent --out-dir*work/d20/forms", "the control did not run", "comp-check"), ("comp-check --draw 1 --out-dir*work/d20/forms", "the composition check did not run", "--draw 2 --work"), ("control --instrument mirror --out-dir*work/d20/mirror/forms", "the mirror's control did not run", "--instrument mirror --composed")])
def test_a_control_that_exits_1_does_not_fall_through(tmp_path, fail, word, later):
    result, calls = run_job(tmp_path, fail_when=fail, code=1)
    assert result.returncode == 1 and done_line(result) == 1 and word in result.stdout
    assert not [x for x in calls.splitlines() if later in x and "work/d20" in x + "work/d20" and "smoke" not in x and "--clips 5" not in x and x.split()[1] != "control" and x.split()[1] != "upload"]


def test_a_tee_failure_stops_the_step(tmp_path):
    result, calls = run_job(tmp_path, tee_fails=True)
    assert result.returncode == 1 and "ALERT the log could not be written (tee 1)" in result.stdout
    assert "compose" not in calls and [x.split()[1] for x in calls.splitlines()] == ["digest", "inputs", "upload", "upload-log"]


@pytest.mark.parametrize("fail,status,word", [("inputs", 1, "pins do not hold"), ("--clips 5", 2, "smoke compose failed"), ("preflight", 1, "preflight refused"), ("--draw 3 --work", 1, "a compose failed"), ("--forms control --instrument parent --resume", 1, "read of the anchors"), ("--forms comp-check*--resume", 1, "read of the stored composition"), ("--draws 5 --instrument parent*--resume", 1, "read of parent draw 5"), ("put-report*d20/report.json", 1, "final report"), ("--forms control --instrument mirror --resume", 1, "mirror's anchors"), ("--draws 2 --instrument mirror*--resume", 1, "read of mirror draw 2"), ("put-report*d20/report-mirror.json", 1, "mirror's final report")])
def test_a_failing_step_stops_the_job_with_its_status(tmp_path, fail, status, word):
    result, calls = run_job(tmp_path, fail_when=fail)
    assert result.returncode == status and done_line(result) == status and word in result.stdout and "upload --partial" in calls


@pytest.mark.parametrize("procs", ["1", "2"])
@pytest.mark.parametrize("fail,word", [("--draws 8 --instrument parent*--resume", "read of parent draw 8"), ("--draws 3 --instrument mirror*--resume", "read of mirror draw 3")])
def test_the_last_read_to_finish_failing_fails_the_job(tmp_path, procs, fail, word):
    result, calls = run_job(tmp_path, fail_when=fail, READ_PROCS=procs)
    assert result.returncode == 1 and done_line(result) == 1 and word in result.stdout
    instrument = "parent" if "parent" in fail else "mirror"
    assert not [x for x in calls.splitlines() if x.split()[1] == "merge" and f"--instrument {instrument}" in x and "work/d20" in x]


def test_no_new_draw_starts_after_a_read_fails_whichever_fails_first(tmp_path):
    # the first read is still running when the second fails: no third starts
    result, calls = run_job(tmp_path, fail_when="--draws 2 --instrument parent*--resume", hold="--draws 1 --instrument parent", READ_PROCS="2")
    assert result.returncode == 1 and "read of parent draw 2 failed" in result.stdout and draws_read(calls, "parent") == [1, 2] and not draws_read(calls, "mirror")
    tmp = tmp_path / "one"
    tmp.mkdir()
    result, calls = run_job(tmp, fail_when="--draws 2 --instrument parent*--resume", READ_PROCS="1")
    assert result.returncode == 1 and draws_read(calls, "parent") == [1, 2]
    tmp = tmp_path / "first"
    tmp.mkdir()
    result, calls = run_job(tmp, fail_when="--draws 1 --instrument mirror*--resume", hold="--draws 2 --instrument mirror", READ_PROCS="2")
    assert result.returncode == 1 and "read of mirror draw 1 failed" in result.stdout and draws_read(calls, "mirror") == [1, 2]


@pytest.mark.parametrize("fail,word,mirror", [("control --instrument parent --out-dir*work/d20/forms", "the control failed", False), ("comp-check --draw 1 --out-dir*work/d20/forms", "not reproduced", False), ("control --instrument mirror --out-dir*work/d20/mirror/forms", "the mirror's control failed", True)])
def test_a_void_stops_with_3_and_reads_no_further_draw(tmp_path, fail, word, mirror):
    result, calls = run_job(tmp_path, fail_when=fail, code=3)
    assert result.returncode == 3 and done_line(result) == 3 and "do not retry this code" in result.stdout and "RERUN_CAUSE" in result.stdout and word in result.stdout
    assert not draws_read(calls, "mirror") and bool(draws_read(calls, "parent")) is mirror


def test_a_parent_check_alert_stops_the_mirror_and_a_rerun_stops_there_again_reading_nothing(tmp_path):
    fail = "check --report*d20/report.json"
    result, calls = run_job(tmp_path, fail_when=fail, code=3)
    assert result.returncode == 1 and "the mirror does not read" in result.stdout and draws_read(calls, "parent") == list(PC.DRAWS) and not draws_read(calls, "mirror") and "--instrument mirror --resume" not in calls
    result, calls = run_job(tmp_path, fail_when=fail, code=3, stage_zero="d20/report.json")
    assert result.returncode == 1 and "the mirror does not read" in result.stdout
    got = commands(calls)
    assert "merge" not in got and "put-report" not in got and "read" not in got and "compose" in got and "--clips 5" not in calls and "--final-key" in calls and "report-mirror.json" in calls
    assert not [x for x in calls.splitlines() if x.split()[1] == "preflight" and "--final-key acquisition/2026-10-17-v1/youtube-asl-v1/round20/parent-controls/report.json" in x]


def test_a_mirror_out_of_memory_resumes_on_the_mirror_alone(tmp_path):
    result, calls = run_job(tmp_path, fail_when="--draws 2 --instrument mirror*--resume")
    assert result.returncode == 1 and draws_read(calls, "parent") == list(PC.DRAWS) and (tmp_path / "work/d20/stage-parent.txt").exists() and not (tmp_path / "work/d20/stage-mirror.txt").exists()
    result, calls = run_job(tmp_path, stage_zero="d20/report.json")
    assert result.returncode == 0 and done_line(result) == 0 and not draws_read(calls, "parent") and draws_read(calls, "mirror") == [1, 2, 3]
    lines = calls.splitlines()
    assert not [x for x in lines if "--instrument parent" in x] and "comp-check" not in commands(calls) and "--clips 5" not in calls
    merges = [x for x in lines if x.split()[1] == "merge"]
    assert len(merges) == 1 and "--instrument mirror" in merges[0] and "--parent-forms" in merges[0]
    assert [x for x in lines if x.split()[1] == "check" and "d20/report.json" in x], "the parent's check runs again"


def test_a_failed_put_report_resumes_with_the_same_merge(tmp_path):
    result, first = run_job(tmp_path, fail_when="put-report*d20/report.json")
    assert result.returncode == 1 and not draws_read(first, "mirror")
    result, second = run_job(tmp_path)
    assert result.returncode == 0 and not draws_read(second, "parent") and draws_read(second, "mirror") == [1, 2, 3]
    merge = lambda calls: next(x for x in calls.splitlines() if x.split()[1] == "merge" and "--instrument parent" in x and "work/d20" in x)      # noqa: E731
    assert merge(first) == merge(second), "the same draws, stop reason and seconds, so the merge writes the same bytes"
    assert [x for x in second.splitlines() if x.split()[1] == "put-report" and "d20/report.json" in x]


def test_every_stage_done_runs_only_their_checks(tmp_path):
    (tmp_path / "work/d20").mkdir(parents=True)
    (tmp_path / "work/d20/stage-parent.txt").write_text("1,2,3,4 all_planned 100\n")
    (tmp_path / "work/d20/stage-mirror.txt").write_text("1,2 clock_rule 200\n")
    result, calls = run_job(tmp_path, stage_zero="stage-state")
    assert result.returncode == 0
    got = commands(calls)
    assert "merge" not in got[got.index("stage-state"):] and "compose" not in got[got.index("stage-state"):] and got[-3:] == ["check", "check", "upload-log"]


def test_a_stage_key_of_another_run_stops_the_job(tmp_path):
    result, calls = run_job(tmp_path, fail_when="stage-state*d20/report-mirror.json", code=1)
    assert result.returncode == 1 and "the mirror's final report is not this run's" in result.stdout and "preflight" not in calls


def test_a_rerun_cause_moves_its_work_and_outputs_under_the_digest_and_keeps_the_voids(tmp_path):
    result, calls = run_job(tmp_path, RERUN_CAUSE="the-cause")
    assert result.returncode == 0
    base = "round20/parent-controls"
    lines = calls.splitlines()
    assert [x for x in lines if x.split()[1] == "preflight" and f"--final-key acquisition/2026-10-17-v1/youtube-asl-v1/{base}/runs/abc123def/report.json --void-prefix acquisition/2026-10-17-v1/youtube-asl-v1/{base}/void --rerun-cause the-cause" in x]
    assert [x for x in lines if x.split()[1] == "preflight" and f"{base}/runs/abc123def/report-mirror.json --void-prefix acquisition/2026-10-17-v1/youtube-asl-v1/{base}/void-mirror --rerun-cause the-cause" in x]
    assert all(f"{base}/runs/abc123def/" in x for x in lines if x.split()[1] in ("merge", "put-report", "compose")) and "work/runs/abc123def/d20/forms" in calls
    assert all("--rerun-cause the-cause" in x for x in lines if x.split()[1] == "merge" and "smoke" not in x)
    assert (tmp_path / "work/runs/abc123def/d20/stage-parent.txt").exists()
    assert f"--key acquisition/2026-10-17-v1/youtube-asl-v1/{base}/logs/" in lines[-1]
    _, plain = run_job(tmp_path / "plain") if (tmp_path / "plain").mkdir() is None else (None, None)
    assert "runs/" not in plain


def test_the_mirrors_rerun_record_is_in_what_it_uploads(tmp_path):
    _, calls = run_job(tmp_path, RERUN_CAUSE="x")
    lines = calls.splitlines()
    preflight = next(x for x in lines if x.split()[1] == "preflight" and "report-mirror.json" in x)
    out = preflight.split("--out-dir ")[1].split()[0]
    assert [x for x in lines if x.split()[1] == "upload" and f"--out-dir {out}/forms " in x and "forms-mirror" in x], "P.check_void writes <out-dir>/forms/rerun.json"


def test_compose_and_read_logs_reach_the_jobs_log(tmp_path):
    result, _ = run_job(tmp_path, fail_when="--draw 3 --work")
    log = (tmp_path / "work/parent_controls.log").read_text()
    assert result.returncode == 1 and "planted failure" in log and "ran scripts/parent_controls.py compose" in log
    tmp = tmp_path / "b"
    tmp.mkdir()
    result, _ = run_job(tmp, fail_when="--draws 2 --instrument parent*--resume")
    assert result.returncode == 1 and "planted failure" in (tmp / "work/parent_controls.log").read_text()


def test_a_terminated_job_kills_its_reads_and_prints_its_done_line(tmp_path):
    import time
    job, log = run_job(tmp_path, slow=True, READ_PROCS="2")
    pids = tmp_path / "pids"
    try:
        for _ in range(600):
            if pids.exists() and len(pids.read_text().split()) >= 2:
                break
            time.sleep(.1)
        started = [int(x) for x in pids.read_text().split()]
        for pid in started:
            os.kill(pid, 0)
        job.terminate()
        out, _ = job.communicate(timeout=30)
    finally:
        job.kill()
        job.communicate()
    assert job.returncode == 143 and out.strip().endswith("PARENT-CONTROLS-DONE 143")
    for pid in started:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert "upload --partial" in log.read_text()


def test_a_missing_repo_prints_its_done_line(tmp_path):
    done = subprocess.run(["bash", str(JOB)], env=dict(PATH="/usr/bin:/bin", HOME=str(tmp_path), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path / "gone")), capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and done.stdout.strip().endswith("PARENT-CONTROLS-DONE 1")


def test_the_job_passes_every_step_through_pipestatus_and_names_no_other_list():
    text = JOB.read_text()
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0
    code = [line.split(" #")[0] for line in text.splitlines() if not line.startswith("#")]
    assert [line for line in code if re.search(r"(?<!\|)\|(?!\|)", line) and not line.strip().startswith("case")] == ['  "$@" 2>&1 | tee -a "$LOG"'] and "${PIPESTATUS[@]}" in text and "set -uo pipefail" in text and "transformers==4.57.6" in text
    for word in ("c20", "C20", "create-pod", "runpodctl", "--likelihood", "r2env.sh)"):
        assert word not in "\n".join(code), word
    assert "NVIDIA L40S" in text and "BUDGET_SECONDS" in text and "PARENT_BUDGET_SECONDS" in text and "MIRROR_DRAWS" in text and "it never moves M1" in text


# ---- the mirror ----------------------------------------------------------------------------------------------------------------------------

def test_an_instrument_outside_the_two_refuses(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    for command in (("read", "--forms", "control", "--out-dir", str(h.forms)), ("control", "--out-dir", str(h.forms), "--void-prefix", "v"), ("merge", "--composed", "c", "--prefix", "p", "--output-file", str(tmp_path / "x"))):
        with pytest.raises(SystemExit, match="--instrument v9"):
            h.main(command[0], "--instrument", "v9", *command[1:])
    with pytest.raises(SystemExit, match="one of"):
        PC.check_instrument("back-translator-kp2d-uwb-ft-v1")


def test_a_mirror_that_loads_another_checkpoint_refuses(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    assert PC.load_instrument("mirror", h.store, None) is h.reader_mirror and PC.load_instrument("parent", h.store, None) is h.reader_parent
    h.reader_mirror.info = dict(MIRROR_INFO, sha256="0" * 64)
    with pytest.raises(SystemExit, match="not the pinned"):
        PC.load_instrument("mirror", h.store, None)
    h.reader_mirror.info = dict(MIRROR_INFO, checkpoint_key=f"{Q.MONO_PREFIX}/checkpoints/step-00015000.pt")
    with pytest.raises(SystemExit, match="not the pinned"):
        PC.load_instrument("mirror", h.store, None)


def test_the_mirrors_anchors_are_both_paths_and_their_stills_with_its_own_reader(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    assert h.mirror_anchors() == 0
    assert [c[0] for c in h.reader_mirror.calls] == list(PC.MIRROR_ANCHORS) and not h.reader_parent.calls
    want = C.M.retrieval_candidates(N, 5, seed=0)
    assert all(np.array_equal(c[2], want) and c[1] == 4 for c in h.reader_mirror.calls)
    for name in PC.MIRROR_ANCHORS:
        form = json.loads((h.forms_mirror / f"form-{name}.json").read_text())
        assert form["instrument"]["sha256"] == PC.INSTRUMENTS["mirror"]["sha256"] and form["kind"] == "parent-controls-form" and form["signs"] == [r["sign"] for r in h.records]
    assert json.loads((h.forms_mirror / "form-real_monocular.json").read_text())["frames"] == [48] * N
    assert json.loads((h.forms_mirror / "form-real_two_view.json").read_text())["frames"] == [len(r.timestamps) for r in h.refs]


def test_a_lost_monocular_clip_refuses_a_full_read_before_any_anchor(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    monkeypatch.setattr(R, "open_monocular", lambda store, records, refs, declared: (h.clips[1:], [dict(sign="s0", reason="state_missing")], None))
    with pytest.raises(SystemExit, match="no monocular state"):
        h.mirror_anchors()
    assert not h.reader_mirror.calls
    monkeypatch.setattr(R, "open_monocular", lambda store, records, refs, declared: (list(reversed(h.clips)), [], None))
    with pytest.raises(SystemExit, match="out of order"):
        h.mirror_anchors()
    monkeypatch.setattr(R, "open_monocular", lambda store, records, refs, declared: (h.clips[:-1], [], None))
    with pytest.raises(SystemExit, match="no monocular state"):
        h.mirror_anchors()


def test_a_smoke_tolerates_a_lost_monocular_clip_through_the_mirrors_merge(monkeypatch, tmp_path, capsys):
    h = smoke(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "open_monocular", lambda store, records, refs, declared: ([c for c in h.clips[:len(records)] if c.index != 1], [dict(sign=records[1]["sign"], reason="state_missing")], None))
    assert h.compose(1, clips=5) == 0
    assert h.main("read", "--forms", "control", "--out-dir", str(h.forms), "--smoke-train", "5") == 0
    assert h.main("read", "--draws", "1", "--composed", "pc/composed", "--out-dir", str(h.forms), "--smoke-train", "5") == 0
    assert h.upload() == 0
    assert h.mirror("read", "--forms", "control", "--out-dir", str(h.forms_mirror), "--smoke-train", "5") == 0
    assert '"smoke_lost": 1' in capsys.readouterr().out
    assert h.mirror("read", "--draws", "1", "--composed", "pc/composed", "--out-dir", str(h.forms_mirror), "--smoke-train", "5") == 0
    form = json.loads((h.forms_mirror / "form-d1_oracle.json").read_text())
    assert form["signs"] == ["train-0", "train-2", "train-3", "train-4"] and len(form["hypotheses"]) == 4
    assert h.main("upload", "--out-dir", str(h.forms_mirror), "--prefix", "pc/forms-mirror") == 0
    out = tmp_path / "smoke-lost.json"
    assert h.mirror("merge", "--draws", "1", "--composed", "pc/composed", "--prefix", "pc/forms-mirror", "--parent-forms", "pc/forms", "--output-file", str(out)) == 0
    report = json.loads(out.read_text())
    assert report["declared_read"] is False and report["parent_on_mirror_draws"] is None and report["composition"]["margin"]["draws"] == 1


def test_a_smoke_with_under_two_monocular_clips_refuses(monkeypatch, tmp_path):
    h = smoke(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "open_monocular", lambda store, records, refs, declared: (h.clips[:1], [dict(sign="x", reason="state_missing")], None))
    with pytest.raises(SystemExit, match="two clips"):
        h.mirror("read", "--forms", "control", "--out-dir", str(h.forms_mirror), "--smoke-train", "5")


def test_the_mirror_reads_the_very_input_the_parent_reads(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    h.to_read((1,))
    h.to_mirror_control()
    assert h.read_mirror("1") == 0
    for kind in PC.KINDS:
        for name in (f"d1_{kind}", f"d1_{kind}_still"):
            assert h.reader_mirror.inputs[name] == h.reader_parent.inputs[name], name
    assert h.reader_mirror.inputs["d1_comp"] == h.reader_parent.inputs["d1_comp_reread"]


def test_a_composed_state_is_marks_free_and_a_layouts_names_never_reach_the_reader(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    assert h.compose() == 0
    rows = h.index()["rows"]
    for kind in ("comp", "oracle", "shuffled_s0"):
        row = rows[kind][0]
        state = load_state(h.store, row["state_key"], row["state_sha256"])
        assert R.valid_frame_share(state) == 1.
        assert np.array_equal(R.reader_input(laid_out(state, MONOCULAR)), R.reader_input(state))
    from signtranslator.data.layout_adapter import conform
    assert np.array_equal(R.reader_input(conform(laid_out(state, MONOCULAR), laid_out(state, TWO_VIEW))), R.reader_input(state))


def test_the_mirrors_own_composition_is_read_where_the_parent_takes_its_pins(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    assert h.compose() == 0
    h.to_mirror_control()
    assert h.read_mirror("1") == 0
    read = {c[0] for c in h.reader_mirror.calls}
    assert {"d1_comp", "d1_still", "d1_blank"} <= read and all(n in read for n in PC.forms_of_draw(1, True, "parent"))
    assert PC.forms_of_draw(1, True, "mirror") == PC.forms_of_draw(1, True, "parent") + ["d1_comp", "d1_still", "d1_blank"] and PC.forms_of_draw(1, False, "parent") == PC.forms_of_draw(1, True, "mirror")
    comp, blank = (json.loads((h.forms_mirror / f"form-d1_{x}.json").read_text()) for x in ("comp", "blank"))
    assert comp["frames"] == blank["frames"] and comp["instrument"]["sha256"] == PC.INSTRUMENTS["mirror"]["sha256"] and not any(blank["valid_frame_share"])


def test_a_mirror_read_needs_its_own_control_and_no_composition_check(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    assert h.compose() == 0
    with pytest.raises(SystemExit, match="no passed control"):
        h.read_mirror("1")
    with pytest.raises(SystemExit, match="the mirror has none"):
        h.mirror("read", "--forms", "comp-check", "--draws", "1", "--composed", "pc/composed", "--out-dir", str(h.forms_mirror))
    # the parent's passed control does not open the mirror's read
    h.to_control()
    with pytest.raises(SystemExit, match="no passed control"):
        h.read_mirror("1")
    h.to_mirror_control()
    assert h.read_mirror("1") == 0


@pytest.mark.parametrize("name", list(PC.MIRROR_ANCHORS))
def test_an_anchor_off_the_mirror_reports_value_voids_its_read(monkeypatch, tmp_path, capsys, name):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    assert h.compose() == 0
    assert h.mirror_anchors() == 0
    h.plant_mirror_stored(offsets={name: .051})
    assert h.mirror_control() == 3
    out = capsys.readouterr().out
    assert f"{name} reads BLEU" in out and "no form follows" in out and h.store.exists("pc/void-mirror/" + "c" * 64 + ".json")
    with pytest.raises(SystemExit, match="no passed control"):
        h.read_mirror("1")
    with pytest.raises(SystemExit, match="this code's read is void"):
        h.main("preflight", "--out-dir", str(tmp_path / "w2"), "--final-key", "pc/report-mirror.json", "--void-prefix", "pc/void-mirror")


def test_an_anchor_inside_the_tolerance_passes(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    assert h.mirror_anchors() == 0
    h.plant_mirror_stored(offsets={name: .049 for name in PC.MIRROR_ANCHORS})
    assert h.mirror_control() == 0


@pytest.mark.parametrize("change,word", [(dict(hypotheses=["other"] * N), "hypotheses"), (dict(nll=[[9.] * 5] * N), "NLL"), (dict(signs=[f"z{i}" for i in range(N)]), "other clips")])
@pytest.mark.parametrize("name", list(PC.MIRROR_ANCHORS))
def test_a_mirror_anchor_off_its_stored_form_voids_its_read(monkeypatch, tmp_path, capsys, change, word, name):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    assert h.mirror_anchors() == 0
    h.plant_mirror_stored({name: change})
    assert h.mirror_control() == 3
    assert word in capsys.readouterr().out


@pytest.mark.parametrize("change", [dict(code_digest="9" * 64), dict(instrument=dict(sha256="0" * 64)), dict(declared_read=False), dict(form="mono__blank")])
def test_a_stored_mirror_form_of_another_read_exits_1_with_no_void(monkeypatch, tmp_path, change):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    assert h.mirror_anchors() == 0
    h.plant_mirror_stored(dict(still_two_view=change))
    with pytest.raises(SystemExit, match="stored mirror read"):
        h.mirror_control()
    assert not h.store.exists("pc/void-mirror/" + "c" * 64 + ".json") and not (h.forms_mirror / "control.json").exists()


def test_a_mirror_report_on_other_bytes_exits_1_with_no_void(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    assert h.mirror_anchors() == 0
    h.plant_mirror_stored()
    h.store.put(PC.MIRROR_REPORT, b"other")
    with pytest.raises(SystemExit, match="SHA-256"):
        h.mirror_control()
    assert not h.store.exists("pc/void-mirror/" + "c" * 64 + ".json")
    # a stored form that is another file's bytes is refused by its hash too
    h.plant_mirror_stored()
    h.store.put(f"{PC.MIRROR_FORMS}/form-mono__blank.json", b"{}")
    with pytest.raises(SystemExit, match="SHA-256"):
        h.mirror_control()
    # the parsed report must hold the declared values and all seven forms
    h.plant_mirror_stored()
    monkeypatch.setattr(PC, "MIRROR_VALUES", dict(PC.MIRROR_VALUES, blank=PC.MIRROR_VALUES["blank"] + .001))
    with pytest.raises(SystemExit, match="does not hold the declared values"):
        PC.load_mirror_report(h.store)
    with pytest.raises(SystemExit, match="no stored form"):
        PC.load_mirror_form(h.store, "d1_comp")


def test_the_mirrors_shares_take_its_own_path_and_the_two_view_anchors_stand_beside(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    report = h.mirror_report()
    assert report["kind"] == "parent-controls-report" and report["instrument_name"] == "mirror" and report["instrument"]["sha256"] == PC.INSTRUMENTS["mirror"]["sha256"] and report["alerts"] == [] and report["draws"] == [1, 2] and report["report_only"] is True
    real = .8 - .15      # over the monocular still, not two-view's
    comp = (.6 + .0015 - STILL_MARGIN) * FILLED / N
    block = report["composition"]
    assert abs(block["margin"]["mean"] - comp) < 1e-4 and abs(block["share_of_real"]["mean"] - comp / real) < 1e-3 and abs(block["share_of_nearest"]["mean"] - comp / (.75 - .15)) < 1e-3
    assert abs(block["share_of_two_view_real"]["mean"] - comp / .4) < 1e-3
    assert report["own_path_anchors"] == dict(real="real_monocular", nearest="nearest_monocular", still="still_monocular", blank="blank")
    oracle = (MARGIN["oracle"] - STILL_MARGIN) * ORACLE_FILLED / N
    assert abs(report["forms"]["oracle"]["share_of_real"]["mean"] - oracle / real) < 1e-3
    anchors = report["anchors"]
    assert abs(anchors["real_monocular"]["margin_over_still"]["point"] - real) < 1e-6 and abs(anchors["nearest_monocular"]["margin_over_still"]["point"] - .6) < 1e-6 and abs(anchors["real_two_view"]["margin_over_still"]["point"] - .4) < 1e-6 and anchors["still_monocular"]["margin_over_still"]["point"] == 0.
    assert report["stored"] is None and set(report["checks"]["controls"]) == {"anchors"} and report["checks"]["controls"]["anchors"]["reproduced"] is True
    assert report["disclosures"]["mirror"].startswith("The mirror is report only and never moves M1") and "never moves M1" in report["reading"]
    frames = json.loads((h.forms_mirror / "form-d1_comp.json").read_text())["frames"]
    assert report["length_over_monocular"] == pytest.approx(np.mean(np.array(frames) / 48), abs=1e-3)
    assert h.main("check", "--report", str(tmp_path / "report-mirror.json")) == 0
    assert report["paired"]["oracle_minus_covered"]["margin"]["mean"] == pytest.approx(oracle - (MARGIN["covered"] - STILL_MARGIN) * ORACLE_FILLED / N, abs=1e-4)


def test_training_status_is_a_report_row_on_either_instrument(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    report = h.mirror_report((1,))
    row = report["forms"]["oracle"]["by_clip_subset"]
    assert row["in_v1_training"]["clips"] == 4 and row["outside_v1_training"]["clips"] == N - 4 and "bleu" in row["in_v1_training"]
    assert report["composition"]["by_clip_subset"]["in_v1_training"]["clips"] == 4
    # the mirror's nearest is its own path's
    forms = json.loads((h.forms_mirror / "form-nearest_monocular.json").read_text())
    assert row["outside_training_and_selection"]["nearest_bleu"] == pytest.approx(C.M.bleu_from_statistics(C.M.bleu_statistics(forms["hypotheses"], forms["references"])[row_mask()].sum(0)), abs=1e-3)


def row_mask():
    return np.array([i >= 6 for i in range(N)])


def test_the_mirrors_report_holds_the_parents_rows_on_its_own_draws(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    parent = h.report((1, 2))
    mirror = h.mirror_report((1,), parent_forms="pc/forms")
    beside = mirror["parent_on_mirror_draws"]
    assert set(beside) == {"composition", "oracle", "covered", "shuffled", "oracle_shuffled", "covered_shuffled"}
    row = beside["oracle"]
    assert row["parent"]["margin"]["draws"] == 1 and row["parent"]["margin"]["per_draw"][0] == pytest.approx(parent["forms"]["oracle"]["margin"]["per_draw"][0], abs=1e-4)
    # one draw of the parent's two, and the same reader on both, so the margin difference is exactly nothing
    assert row["parent_minus_mirror"]["margin"]["per_draw"] == pytest.approx([0.], abs=1e-6)
    # each share is over its own instrument's real: .4 on the parent, .65 on the mirror
    want = (MARGIN["oracle"] - STILL_MARGIN) * ORACLE_FILLED / N * (1 / .4 - 1 / .65)
    assert row["parent_minus_mirror"]["share_of_real"]["per_draw"][0] == pytest.approx(want, abs=1e-3)


def test_the_parents_rows_need_its_forms_of_this_code_and_every_draw(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1, 2))
    h.report((1,))
    with pytest.raises(SystemExit, match="draw 2: "):
        h.mirror_report((1, 2), parent_forms="pc/forms")


def test_a_form_of_the_other_instrument_refuses_the_merge(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    h.report()
    with pytest.raises(SystemExit, match="not read by the mirror"):
        h.mirror("merge", "--composed", "pc/composed", "--draws", "1,2", "--prefix", "pc/forms", "--output-file", str(tmp_path / "x.json"))
    h.mirror_report()
    with pytest.raises(SystemExit, match="not read by the parent"):
        h.main("merge", "--composed", "pc/composed", "--draws", "1,2", "--prefix", "pc/forms-mirror", "--output-file", str(tmp_path / "y.json"))


def test_a_mirror_form_missing_its_composition_refuses_the_merge(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    h.mirror_report((1,))
    (h.store.root / "pc/forms-mirror/form-d1_still.json").unlink()
    with pytest.raises(SystemExit, match="d1_still is missing"):
        h.mirror("merge", "--composed", "pc/composed", "--draws", "1", "--prefix", "pc/forms-mirror", "--output-file", str(tmp_path / "z.json"))


def test_a_wrong_still_or_clock_changes_or_refuses_the_mirrors_result_too(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    h.mirror_report((1,))
    forms = {p.stem[5:]: json.loads(p.read_text()) for p in h.forms_mirror.glob("form-*.json")}
    idx = np.random.default_rng(0).integers(0, N, (200, N))
    base = {1: {x: forms[f"d1_{x}"] for x in ("comp", "still", "blank")}}
    anchors = PC.own_path(forms, "mirror")
    right, _, _ = PC.view_arrays({1: (forms["d1_oracle"], forms["d1_oracle_still"])}, base, anchors, idx, "oracle")
    other = json.loads(json.dumps(forms["d1_oracle_still"]))
    other["nll"] = [[n[0]] + [n[0] + .25] * 4 for n in other["nll"]]
    wrong, _, _ = PC.view_arrays({1: (forms["d1_oracle"], other)}, base, anchors, idx, "oracle")
    assert abs(wrong[1]["margin"].mean() - right[1]["margin"].mean()) > .05
    # the wrong anchor is a wrong share
    two_view = {a: forms[a] for a in C.ANCHORS}
    other_share, _, _ = PC.view_arrays({1: (forms["d1_oracle"], forms["d1_oracle_still"])}, base, two_view, idx, "oracle")
    assert abs(other_share[1]["real"].mean() - right[1]["real"].mean()) > .1
    # the still alone taken from the wrong path moves real's margin
    wrong_still = PC.own_path(dict(forms, still_monocular=forms["still_two_view"]), "mirror")
    moved_still, _, _ = PC.view_arrays({1: (forms["d1_oracle"], forms["d1_oracle_still"])}, base, wrong_still, idx, "oracle")
    assert moved_still[1]["real"].mean() == pytest.approx(right[1]["real"].mean() + .05, abs=1e-4)
    off = dict(forms["d1_oracle"], frames=[x + 1 for x in forms["d1_oracle"]["frames"]])
    with pytest.raises(SystemExit, match="is not as long as the composition"):
        PC.view_arrays({1: (off, dict(forms["d1_oracle_still"], frames=off["frames"]))}, base, anchors, idx, "oracle")


def test_a_smoke_reads_through_the_mirror_and_compares_nothing(monkeypatch, tmp_path, capsys):
    h = smoke(monkeypatch, tmp_path)
    assert h.compose(1, clips=5) == 0
    assert h.mirror("read", "--forms", "control", "--out-dir", str(h.forms_mirror), "--smoke-train", "5") == 0
    capsys.readouterr()
    assert h.mirror("control", "--out-dir", str(h.forms_mirror), "--void-prefix", "pc/void-mirror") == 0 and "nothing compared" in capsys.readouterr().out
    assert h.mirror("read", "--draws", "1", "--composed", "pc/composed", "--out-dir", str(h.forms_mirror), "--smoke-train", "5") == 0
    assert h.main("upload", "--out-dir", str(h.forms_mirror), "--prefix", "pc/forms-mirror") == 0
    out = tmp_path / "smoke-mirror.json"
    assert h.mirror("merge", "--draws", "1", "--composed", "pc/composed", "--prefix", "pc/forms-mirror", "--output-file", str(out)) == 0
    report = json.loads(out.read_text())
    assert report["declared_read"] is False and "SMOKE" in report["smoke"] and report["instrument_name"] == "mirror"


# ---- v1, read only ------------------------------------------------------------------------------------------------------------------------------

def test_v1s_pins_and_declared_reading():
    head = PC.__doc__
    for text in ("how much of the official 2.40 is word order", "**Reads only: it is never used here for a loss, a rerank or a best-of-N,**", "real two-view (4.493) and nearest two-view (4.621), with real monocular beside them", "near the parent's and the mirror's, about 1 BLEU,** means much of the official 2.40 is word order", "**A small drop** means it mostly is not", "Neither is the better outcome", "``official_gap``", "**v1 trained on 234 of D20's clips and selected on 265 more,** so ``by_clip_subset`` rows matter", "No target moves", "None of it shows human comprehension", "they are not this recipe's margin", "no new compose", "NVIDIA L40S", "About 2.6 h and about $2.90", "All eight draws"):
        assert text in head, text
    pin = PC.INSTRUMENTS["v1"]
    assert pin["run"] == L.TRANSLATOR_RUN and pin["sha256"] == L.TRANSLATOR_SHA and pin["sha256"].startswith("2e3ded3f") and pin["checkpoint_key"].endswith("back-translator-kp2d-uwb-ft-v1/checkpoints/step-00009000.pt")
    assert PC.V1_A5_BLEU == (2.406, 2.469, 2.22, 2.449, 2.08, 2.35, 2.556, 2.687) and PC.COMPOSED_DIGEST.startswith("0936c98a") and set(PC.COMPOSED_INDEX) == set(PC.DRAWS) and len(set(PC.COMPOSED_INDEX.values())) == 8 and PC.COMPOSED_INDEX[1].startswith("02726a3d") and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in PC.COMPOSED_INDEX.values())
    assert "never for a loss, a rerank or a best-of-N" in PC.READING_V1 and "about 1 BLEU" in PC.READING_V1 and "never for a loss, a rerank or a best-of-N" in PC.DISCLOSURES["v1"] and PC.V1_COST_ESTIMATE["measured"] is False


@pytest.mark.parametrize("change", [dict(sha256="0" * 64), dict(checkpoint_key=f"training-runs/{L.TRANSLATOR_RUN}/checkpoints/step-00015000.pt")])
def test_a_v1_of_another_checkpoint_refuses(monkeypatch, tmp_path, change):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    assert PC.load_instrument("v1", h.store, None) is h.reader_v1
    h.reader_v1.info = dict(V1_INFO, **change)
    with pytest.raises(SystemExit, match="not the pinned"):
        PC.load_instrument("v1", h.store, None)


def test_v1s_anchors_pass_and_its_reads_are_v1s_alone(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    assert h.v1_anchors() == 0
    assert [c[0] for c in h.reader_v1.calls] == list(PC.MIRROR_ANCHORS) and not h.reader_parent.calls and not h.reader_mirror.calls
    h.plant_v1_stored()
    assert h.v1_control() == 0
    record = json.loads((h.forms_v1 / "control.json").read_text())
    assert record["reproduced"] is True and set(record["stored"]) == set(PC.V1_STORED_ANCHORS) and record["canary"]["reproduced"] is True


@pytest.mark.parametrize("canary,word", [(dict(real=dict(hypotheses=["other"] * N)), "canary real hypotheses"), (dict(blank=dict(per_clip=dict(nll=[9.] * N))), "canary blank NLL")])
def test_a_canary_miss_voids_v1s_read(monkeypatch, tmp_path, capsys, canary, word):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    assert h.compose() == 0
    h.pin_composed((1,))
    assert h.v1_anchors() == 0
    h.plant_v1_stored(canary=canary)
    assert h.v1_control() == 3
    out = capsys.readouterr().out
    assert word in out and "v1's read is void" in out and "RERUN_CAUSE" in out and h.store.exists("pc/void-v1/" + "c" * 64 + ".json")
    with pytest.raises(SystemExit, match="no passed control"):
        h.v1("read", "--draws", "1", "--composed", "pc/composed", "--out-dir", str(h.forms_v1))
    with pytest.raises(SystemExit, match="this code's read is void"):
        h.main("preflight", "--out-dir", str(tmp_path / "w2"), "--final-key", "pc/report-v1.json", "--void-prefix", "pc/void-v1")


@pytest.mark.parametrize("name", ["real_two_view", "blank", "nearest_two_view", "nearest_monocular"])
def test_an_anchor_off_the_parts_forms_voids_v1s_read(monkeypatch, tmp_path, capsys, name):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    assert h.v1_anchors() == 0
    h.plant_v1_stored({name: dict(hypotheses=["other"] * N)})
    assert h.v1_control() == 3
    assert f"{name} hypotheses" in capsys.readouterr().out


@pytest.mark.parametrize("what", ["canary", "parts"])
def test_a_v1_pin_on_other_bytes_exits_1_with_no_void(monkeypatch, tmp_path, what):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    assert h.v1_anchors() == 0
    h.plant_v1_stored()
    if what == "canary":
        h.store.put(P.CANARY_REPORT, b"{}")
    else:
        h.store.put(f"{Q.PARTS_FORMS}/s-blank.json", b"{}")
    with pytest.raises(SystemExit, match="SHA-256"):
        h.v1_control()
    assert not h.store.exists("pc/void-v1/" + "c" * 64 + ".json") and not (h.forms_v1 / "control.json").exists()


def test_a_v1_smoke_control_compares_nothing(monkeypatch, tmp_path, capsys):
    h = smoke(monkeypatch, tmp_path)
    assert h.v1("read", "--forms", "control", "--out-dir", str(h.forms_v1), "--smoke-train", "5") == 0
    capsys.readouterr()
    assert h.v1_control() == 0 and "nothing compared" in capsys.readouterr().out


def to_v1_comp_check(h, k=1):
    h.pin_composed((k,))
    assert h.v1_anchors() == 0
    h.plant_v1_stored()
    assert h.v1_control() == 0


def test_draw_ones_stored_composition_must_reproduce_before_a_draw_is_read(monkeypatch, tmp_path, capsys):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    assert h.compose() == 0
    to_v1_comp_check(h)
    with pytest.raises(SystemExit, match="has not been reproduced"):
        h.v1("read", "--draws", "1", "--composed", "pc/composed", "--out-dir", str(h.forms_v1))
    h.plant_a5((1,))
    assert h.v1_comp_check(1) == 0
    assert json.loads((h.forms_v1 / "comp-check.json").read_text())["stored"]["identical"] == 1.
    assert h.v1("read", "--draws", "1", "--composed", "pc/composed", "--out-dir", str(h.forms_v1)) == 0
    assert "d1_comp_reread" in {c[0] for c in h.reader_v1.calls} and not {c[0] for c in h.reader_parent.calls}


@pytest.mark.parametrize("change,word", [(dict(words=3), "hypotheses"), (dict(signs=[f"z{i}" for i in range(N)]), "other clips"), (dict(references=["other"] * N), "other clips")])
def test_a_composition_not_reproduced_voids_v1s_read(monkeypatch, tmp_path, capsys, change, word):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    assert h.compose() == 0
    to_v1_comp_check(h)
    h.plant_a5((1,), **change)
    assert h.v1_comp_check(1) == 3
    assert word in capsys.readouterr().out and h.store.exists("pc/void-v1/" + "c" * 64 + ".json")
    with pytest.raises(SystemExit, match="has not been reproduced"):
        h.v1("read", "--draws", "1", "--composed", "pc/composed", "--out-dir", str(h.forms_v1))


def test_a_bleu_off_the_summary_voids_and_own_nll_only_stands_beside(monkeypatch, tmp_path):
    read = dict(signs=["a", "b"], references=["w x y z a b", "c d e f g h"], hypotheses=["w x y z", "c d e f"], nll=[[2.], [2.]])
    top10 = dict(signs=["a", "b"], references=read["references"], forms={C.PREDICTED: dict(hypotheses=read["hypotheses"])}, likelihood=dict(per_form={C.PREDICTED: [-9., -9.]}))
    bleu = C.M.bleu_ci_fast(read["hypotheses"], read["references"])[0]
    row, misses = PC.a5_row(read, top10, bleu)
    assert misses == [] and row["own_nll_mean_absolute_difference"] == 7. and row["identical"] == 1.
    assert PC.a5_row(read, top10, bleu + .051)[1] == ["BLEU"]


def test_v1_reads_only_the_stored_composed_forms_by_their_full_hashes(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    assert h.compose() == 0
    h.pin_composed((1,))
    # new code reads the first read's index, where another instrument of this code would refuse it
    monkeypatch.setattr(PC, "code_digest", lambda: "d" * 64)
    assert PC.load_index(h.store, "pc/composed", 1, "v1")["draw"] == 1
    with pytest.raises(SystemExit, match="written once"):
        PC.load_index(h.store, "pc/composed", 1, "mirror")
    monkeypatch.setattr(PC, "COMPOSED_DIGEST", "e" * 64)
    with pytest.raises(SystemExit, match="not the stored composed forms"):
        PC.load_index(h.store, "pc/composed", 1, "v1")
    monkeypatch.setattr(PC, "COMPOSED_INDEX", {1: "0" * 64})
    with pytest.raises(SystemExit, match="SHA-256"):
        PC.load_index(h.store, "pc/composed", 1, "v1")


def test_a_composed_state_off_its_hash_in_the_index_stops_a_v1_read(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    assert h.compose() == 0
    to_v1_comp_check(h)
    h.plant_a5((1,))
    assert h.v1_comp_check(1) == 0
    row = h.index()["rows"]["oracle"][0]
    h.store.put(row["state_key"], b"other")
    with pytest.raises(ValueError, match="hash mismatch"):
        h.v1("read", "--draws", "1", "--composed", "pc/composed", "--out-dir", str(h.forms_v1))


def test_v1_reads_the_very_input_the_parent_reads(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=(1,))
    h.to_read((1,))
    h.v1_read((1,))
    for kind in PC.KINDS:
        for name in (f"d1_{kind}", f"d1_{kind}_still"):
            assert h.reader_v1.inputs[name] == h.reader_parent.inputs[name], name
    assert h.reader_v1.inputs["d1_comp"] == h.reader_parent.inputs["d1_comp_reread"] == h.reader_v1.inputs["d1_comp_reread"]


def test_v1s_report_anchors_on_two_view_with_monocular_beside(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    report = h.v1_report()
    assert report["kind"] == "parent-controls-report" and report["instrument_name"] == "v1" and report["instrument"]["sha256"] == L.TRANSLATOR_SHA and report["report_only"] is True and report["alerts"] == [] and report["draws"] == [1, 2] and report["reading"] == PC.READING_V1 and report["cost_estimate"] == PC.V1_COST_ESTIMATE
    assert report["own_path_anchors"] == dict(real="real_two_view", nearest="nearest_two_view", still="still_two_view", blank="blank")
    comp = (.6 + .0015 - STILL_MARGIN) * FILLED / N
    block = report["composition"]
    # real two-view over two-view's still, .4; real monocular's own .65 would be the wrong anchor
    assert abs(block["margin"]["mean"] - comp) < 1e-4 and abs(block["share_of_real"]["mean"] - comp / .4) < 1e-3 and abs(block["share_of_nearest"]["mean"] - comp / .5) < 1e-3
    anchors = report["anchors"]
    assert abs(anchors["real_two_view"]["margin_over_still"]["point"] - .4) < 1e-6 and abs(anchors["nearest_two_view"]["margin_over_still"]["point"] - .5) < 1e-6 and abs(anchors["real_monocular"]["margin_over_still"]["point"] - .65) < 1e-6 and anchors["still_monocular"]["margin_over_still"]["point"] == 0.
    oracle = (MARGIN["oracle"] - STILL_MARGIN) * ORACLE_FILLED / N
    assert report["paired"]["form_minus_real"]["oracle"]["margin"]["mean"] == pytest.approx(oracle - .4, abs=1e-4) and report["paired"]["form_minus_nearest"]["oracle"]["margin"]["mean"] == pytest.approx(oracle - .5, abs=1e-4)
    reading = report["reading_contrasts"]
    assert reading["format_like"] == "real_two_view" and reading["format_beside"]["like"] == "nearest_two_view" and reading["gates_nothing"] is True and reading["format"] == "below" and reading["generated_against_real_clips"] == "below" and set(reading["shuffle_on_moved_clips"]) == set(report["paired"]["moved"])
    assert report["stored"] is None and set(report["checks"]["controls"]) == {"anchors", "comp_check"} and report["disclosures"]["v1"] == PC.DISCLOSURES["v1"] and report["parent_on_mirror_draws"] is None
    assert report["forms"]["oracle"]["by_clip_subset"]["outside_training_and_selection"]["clips"] == N - 6 and report["composition"]["by_clip_subset"]["in_v1_training"]["clips"] == 4
    assert h.main("check", "--report", str(tmp_path / "report-v1.json")) == 0
    assert (tmp_path / "report-v1.json").read_bytes() and h.v1("merge", "--draws", "1,2", "--composed", "pc/composed", "--prefix", "pc/forms-v1", "--output-file", str(tmp_path / "report-v1.json")) == 0


def test_the_official_gap_row_holds_composition_and_its_shuffled_share(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    report = h.v1_report()
    gap = report["official_gap"]
    assert gap["composition"]["per_draw"] == pytest.approx(report["composition"]["bleu_gap_share"]["per_draw"], abs=1e-4) and gap["composition_shuffled"]["mean"] == pytest.approx(report["groups"]["shuffled"]["bleu_gap_share"]["mean"], abs=1e-4)
    assert gap["shuffled_minus_composition"]["per_draw"][0] == pytest.approx(gap["composition_shuffled"]["per_draw"][0] - gap["composition"]["per_draw"][0], abs=1e-3)
    comp, shuffled, blank = report["composition"]["bleu"]["per_draw"][0], report["groups"]["shuffled"]["bleu"]["per_draw"][0], report["composition"]["per_draw"]["1"]["bleu_gap_share"]["blank"]
    assert gap["share_of_composition_gap_removed_by_shuffling"]["per_draw"][0] == pytest.approx((comp - shuffled) / (comp - blank), abs=1e-3) and gap["share_of_composition_gap_removed_by_shuffling"]["draws"] == 2
    parent = h.report()
    assert parent["official_gap"] is None and parent["stored_a5"] is None


def test_every_draws_composition_is_set_beside_its_stored_a5_report(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    report = h.v1_report()
    assert set(report["stored_a5"]) == {"1", "2"} and report["stored_a5"]["2"]["identical"] == 1. and report["stored_a5"]["2"]["misses"] == []
    # a later draw off its stored report is an alert in the report, which the check stops on
    h.plant_a5((2,), words=4)
    out = tmp_path / "off.json"
    assert h.v1("merge", "--draws", "1,2", "--composed", "pc/composed", "--prefix", "pc/forms-v1", "--output-file", str(out)) == 0
    off = json.loads(out.read_text())
    assert any("draw 2: composition does not reproduce its stored A5 report" in a for a in off["alerts"])
    assert h.main("check", "--report", str(out)) == 3


def test_v1_refuses_fewer_than_all_its_draws(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    h.v1_read()
    with pytest.raises(SystemExit, match="v1 reads all eight draws"):
        h.v1("merge", "--draws", "1", "--composed", "pc/composed", "--prefix", "pc/forms-v1", "--output-file", str(tmp_path / "x.json"))


def test_forms_of_another_instrument_refuse_v1s_merge(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path)
    h.report()
    with pytest.raises(SystemExit, match="not read by the v1"):
        h.v1("merge", "--draws", "1,2", "--composed", "pc/composed", "--prefix", "pc/forms", "--output-file", str(tmp_path / "x.json"))


def test_v1s_inputs_hold_every_pin_before_any_spend(monkeypatch, tmp_path, capsys):
    h = Controls(monkeypatch, tmp_path)
    h.v1_read()
    capsys.readouterr()
    assert h.main("inputs", "--instrument", "v1") == 0
    out = capsys.readouterr().out
    assert "bleu" not in out and "hypotheses" not in out and "margin" not in out
    monkeypatch.setattr(P, "environment", lambda: dict(gpu="NVIDIA RTX 6000 Ada Generation"))
    with pytest.raises(SystemExit, match="not NVIDIA L40S"):
        h.main("inputs", "--instrument", "v1", "--expect-gpu", "NVIDIA_L40S")


@pytest.mark.parametrize("what,word", [("index", "SHA-256"), ("summary", "SHA-256"), ("bleu", "declared per-draw BLEU"), ("canary", "SHA-256"), ("parts", "SHA-256"), ("signs", "other clips than D20"), ("top10", "SHA-256")])
def test_a_v1_input_off_its_pin_stops_at_inputs(monkeypatch, tmp_path, what, word):
    h = Controls(monkeypatch, tmp_path)
    h.v1_read()
    if what == "index":
        monkeypatch.setattr(PC, "COMPOSED_INDEX", {**PC.COMPOSED_INDEX, 1: "0" * 64})
    elif what == "summary":
        h.store.put(C.A5_SUMMARY, b"{}")
    elif what == "bleu":
        monkeypatch.setattr(PC, "V1_A5_BLEU", (0.,) * 8)
    elif what == "canary":
        h.store.put(P.CANARY_REPORT, b"{}")
    elif what == "parts":
        h.store.put(f"{Q.PARTS_FORMS}/s-blank.json", b"{}")
    elif what == "top10":
        h.store.put(C.top10_key(2), b"{}")
    else:
        h.plant_v1_stored(dict(blank=dict(signs=[f"z{i}" for i in range(N)])))
    with pytest.raises(SystemExit, match=word):
        h.main("inputs", "--instrument", "v1")


def test_the_parent_and_mirrors_inputs_still_ask_for_their_model_file(monkeypatch, tmp_path):
    h = Controls(monkeypatch, tmp_path, compose_draws=())
    h.store.put(PC.STORED_REPORT, b"{}")
    monkeypatch.setattr(PC, "STORED_REPORT_SHA", hashlib.sha256(b"{}").hexdigest())
    with pytest.raises(SystemExit, match="--model-file"):
        h.main("inputs")


def test_a_v1_smoke_reads_the_stored_forms_against_train_references(monkeypatch, tmp_path):
    h = smoke(monkeypatch, tmp_path)
    assert h.compose(1) == 0
    h.pin_composed((1,))
    assert h.v1("read", "--forms", "control", "--out-dir", str(h.forms_v1), "--smoke-train", "5") == 0
    assert h.v1("read", "--draws", "1", "--composed", "pc/composed", "--out-dir", str(h.forms_v1), "--smoke-train", "5") == 0
    form = json.loads((h.forms_v1 / "form-d1_oracle.json").read_text())
    assert form["declared_read"] is False and all(x.startswith("train-") for x in form["signs"]) and form["instrument"]["sha256"] == L.TRANSLATOR_SHA
    assert h.main("upload", "--out-dir", str(h.forms_v1), "--prefix", "pc/forms-v1") == 0
    out = tmp_path / "smoke-v1.json"
    assert h.v1("merge", "--draws", "1", "--composed", "pc/composed", "--prefix", "pc/forms-v1", "--output-file", str(out)) == 0
    report = json.loads(out.read_text())
    assert report["declared_read"] is False and "SMOKE" in report["smoke"] and report["stored_a5"] is None and report["official_gap"]["composition"]["draws"] == 1


# ---- v1's path through the job
V1_SMOKE = ["read", "control", "read", "upload", "merge", "put-report", "check"]


def test_the_v1_job_reads_all_eight_stored_draws_with_no_compose(tmp_path):
    result, calls = run_job(tmp_path, INSTRUMENTS="v1")
    assert result.returncode == 0 and done_line(result) == 0
    got = commands(calls)
    assert got[:2] == ["digest", "inputs"] and got[2:9] == V1_SMOKE and got[9:15] == ["stage-state", "preflight", "read", "control", "read", "comp-check"]
    assert "compose" not in got and got[-5:] == ["upload", "merge", "put-report", "check", "upload-log"]
    lines = calls.splitlines()
    assert "--expect-gpu NVIDIA_L40S" in lines[1] and "--instrument v1" in lines[1] and "--model-file" not in lines[1]
    assert draws_read(calls, "v1") == list(PC.DRAWS) and not draws_read(calls, "parent") and not draws_read(calls, "mirror")
    base = "acquisition/2026-10-17-v1/youtube-asl-v1/round20/parent-controls"
    assert all(f"--composed {base}/composed " in x for x in lines if x.split()[1] == "merge" or "--draws" in x and "--forms" not in x)
    merge = [x for x in lines if x.split()[1] == "merge"][-1]
    assert f"--prefix {base}/forms-v1 --output-file" in merge and "--stop-reason all_planned" in merge and "--parent-forms" not in merge
    assert [x for x in lines if x.split()[1] == "put-report"][-1].endswith(f"--key {base}/report-v1.json")
    assert [x for x in lines if x.split()[1] == "preflight"][0].split("--void-prefix ")[1].startswith(f"{base}/void-v1")
    assert next(i for i, x in enumerate(lines) if x.split()[1] == "comp-check") < next(i for i, x in enumerate(lines) if "--resume" in x and "--draws" in x and "comp-check" not in x)
    timeouts = (tmp_path / "timeouts.log").read_text()
    assert timeouts.count("python scripts/parent_controls.py") == len(lines), "every python step of the job runs under a limit"


def test_the_clock_rule_never_drops_a_v1_draw(tmp_path):
    result, calls = run_job(tmp_path, INSTRUMENTS="v1", BUDGET_SECONDS="-1")
    assert result.returncode == 0 and draws_read(calls, "v1") == list(PC.DRAWS) and "clock rule" not in result.stdout


@pytest.mark.parametrize("env,word", [(dict(DRAWS="1 2 3"), "v1 reads all eight draws"), (dict(INSTRUMENTS="v2"), "INSTRUMENTS is parent,mirror or v1")])
def test_fewer_v1_draws_or_another_instrument_stop_the_job_first(tmp_path, env, word):
    result, calls = run_job(tmp_path, **dict(dict(INSTRUMENTS="v1"), **env))
    assert result.returncode == 1 and word in result.stdout and "inputs" not in calls


def test_the_v1_smoke_mode_stops_before_d20(tmp_path):
    result, calls = run_job(tmp_path, INSTRUMENTS="v1", SMOKE="1")
    assert result.returncode == 0 and "SMOKE done" in result.stdout and "preflight" not in calls and "--smoke-train 5" in calls and "smoke-v1/" in calls


@pytest.mark.parametrize("fail,word", [("control --instrument v1 --out-dir*work/d20-v1/v1/forms", "v1's control failed"), ("comp-check --instrument v1 --draw 1*work/d20-v1/v1/forms", "v1 did not reproduce the stored composition")])
def test_a_v1_void_stops_with_3_and_reads_no_draw(tmp_path, fail, word):
    result, calls = run_job(tmp_path, INSTRUMENTS="v1", fail_when=fail, code=3)
    assert result.returncode == 3 and done_line(result) == 3 and word in result.stdout and "do not retry this code" in result.stdout and "RERUN_CAUSE" in result.stdout and not draws_read(calls, "v1")


@pytest.mark.parametrize("fail,word", [("control --instrument v1 --out-dir*work/d20-v1/v1/forms", "v1's control did not run"), ("comp-check --instrument v1 --draw 1*work/d20-v1/v1/forms", "v1's composition check did not run")])
def test_a_v1_control_that_exits_1_does_not_fall_through(tmp_path, fail, word):
    result, calls = run_job(tmp_path, INSTRUMENTS="v1", fail_when=fail, code=1)
    assert result.returncode == 1 and word in result.stdout and not draws_read(calls, "v1")


@pytest.mark.parametrize("procs", ["1", "2"])
def test_the_last_v1_read_to_finish_failing_fails_the_job(tmp_path, procs):
    result, calls = run_job(tmp_path, INSTRUMENTS="v1", fail_when="--draws 8 --instrument v1*--resume", READ_PROCS=procs)
    assert result.returncode == 1 and "read of v1 draw 8" in result.stdout and not [x for x in calls.splitlines() if x.split()[1] == "merge" and "d20-v1/report-v1.json" in x]


def test_a_failed_v1_put_report_resumes_with_the_same_merge(tmp_path):
    result, first = run_job(tmp_path, INSTRUMENTS="v1", fail_when="put-report*d20-v1/report-v1.json")
    assert result.returncode == 1
    result, second = run_job(tmp_path, INSTRUMENTS="v1")
    assert result.returncode == 0 and not draws_read(second, "v1") and "--smoke-train" not in second
    merge = lambda calls: next(x for x in calls.splitlines() if x.split()[1] == "merge" and "d20-v1/report-v1.json" in x)      # noqa: E731
    assert merge(first) == merge(second)


def test_a_finished_v1_stage_runs_only_its_check_again(tmp_path):
    (tmp_path / "work/d20-v1").mkdir(parents=True)
    (tmp_path / "work/d20-v1/stage-v1.txt").write_text("1,2,3,4,5,6,7,8 all_planned 100 14400\n")
    result, calls = run_job(tmp_path, INSTRUMENTS="v1", stage_zero="report-v1.json")
    assert result.returncode == 0
    got = commands(calls)
    assert got[got.index("stage-state"):] == ["stage-state", "check", "upload-log"]


def test_a_v1_rerun_keeps_the_stored_composed_forms_and_writes_under_its_digest(tmp_path):
    result, calls = run_job(tmp_path, INSTRUMENTS="v1", RERUN_CAUSE="the-cause")
    assert result.returncode == 0
    base = "acquisition/2026-10-17-v1/youtube-asl-v1/round20/parent-controls"
    lines = calls.splitlines()
    merge = [x for x in lines if x.split()[1] == "merge"][-1]
    assert f"--composed {base}/composed " in merge and f"--prefix {base}/runs/abc123def/forms-v1" in merge and "--rerun-cause the-cause" in merge and "work/runs/abc123def/d20-v1/report-v1.json" in merge
    assert [x for x in lines if x.split()[1] == "preflight"][0].split("--void-prefix ")[1].startswith(f"{base}/void-v1 --rerun-cause the-cause")
