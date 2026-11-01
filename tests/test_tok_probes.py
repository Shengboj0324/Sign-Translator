"""The tokenizer probes and the seam ablation on synthetic data (reports/post_s2_plan.md, sections 4 and 5). Planted cases: a
decoder that returns real reads real, a blanked part drops only where that part is the decoded one, a mutated pin refuses, a fit
that is not c02 train refuses. No test starts a real job: the runner tests put fake executables first on PATH, with no credentials
in the environment, and the store is a directory or a stand-in."""
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
from signtranslator.generation.compositional import take_frames  # noqa: E402
from signtranslator.generation.motion_metrics import compare  # noqa: E402
from signtranslator.generation.motion_rvq import MotionRVQ, MotionRVQConfig  # noqa: E402
from tests.test_generation_motion_rvq import motion_config  # noqa: E402
from tests.test_generation_signing_space import signer_state  # noqa: E402

CLIPS = 24
WORDS = [f"w{i}" for i in range(15)]


def tiny_tokenizer(stages=8, codebook=16, seed=0):
    torch.manual_seed(seed)
    state = signer_state(0, frames=16, amplitude=.4)
    return MotionRVQ(MotionRVQConfig(motion_config(state), hidden=32, code_dim=8, codebook=codebook, stages=stages, down_layers=0)).eval()


def gappy(state, start=6, stop=10):
    """``state`` with both hands and the forearms untracked over some frames, as a real clip is."""
    channels = dict(state.channels)
    for name in ("left_hand", "right_hand", "body"):
        c = channels[name]
        valid = c.valid.copy()
        if name == "body":
            valid[start:stop, [i for i, label in enumerate(c.labels) if "Forearm" in label or label in ("LeftHand", "RightHand")]] = False
        else:
            valid[start:stop] = False
        channels[name] = replace(c, valid=valid, values=np.where(valid[..., None], c.values, 0.), observed=c.observed & valid,
                                 confidence=c.confidence * valid)
    return replace(state, channels=channels)


class Perfect:
    """A tokenizer whose decoder returns its input; ``damage`` changes the decoded unit vector, as a faulty decoder would."""

    def __init__(self, tokenizer, damage=None) -> None:
        self.tok, self.damage = tokenizer, damage
        self.config, self.parts = tokenizer.config, tokenizer.parts

    def token_valid(self, frame_valid):
        return frame_valid

    def encode(self, unit):
        return {p: unit[..., cols] for p, cols in self.parts}

    def quantize(self, latents, stages=None):
        return latents, None, None

    def decode_latents(self, quantised, frames):
        first = next(iter(quantised.values()))
        unit = torch.zeros(*first.shape[:2], self.config.width, dtype=first.dtype)
        for part, cols in self.parts:
            unit[..., cols] = quantised[part]
        return (self.damage(unit, self.config) if self.damage else unit)[:, :frames]


def neutral_columns(part: str):
    def damage(unit, config):
        unit = unit.clone()
        for name, cols in config.part_slices():
            if name == part:
                unit[..., cols] = torch.tensor([1., 0., 0., 0., 1., 0.], dtype=unit.dtype).repeat(len(cols) // 6)
        return unit
    return damage


class PartReader:
    """A reader that needs each part right to say its third of the sentence: it compares each channel with the real clip on real's
    tracked frames. Blank says nothing useful. It stands in for v1 so a planted fault must travel through forms, read and merge."""
    info = dict(run="part-reader", checkpoint_key="k", sha256="f" * 64, kind="fake")

    def __init__(self, refs) -> None:
        self.refs = refs

    def read(self, states, texts, candidates, group=2):
        hyps, nlls = [], []
        for s, ref, text in zip(states, self.refs, texts):
            w = text.split()
            third = len(w) // 3
            right = 0
            kept = []
            for k, name in enumerate(("body", "left_hand", "right_hand")):
                r, c = ref.channels[name], s.channels[name]
                ok = bool(c.valid.any()) and bool(np.allclose(c.values[r.valid], r.values[r.valid], atol=1e-4))
                right += ok
                kept += w[k * third:(k + 1) * third] if ok else ["x"] * third
            hyps.append(" ".join(kept))
            nlls.append([3. - right] * candidates.shape[1])
        return dict(hypotheses=hyps, nll=np.array(nlls))


def clip_records(n=CLIPS, declared=True):
    base = dict(corpus=L.H2S, partition="val", signer_component="c01") if declared else dict(corpus=L.H2S, partition="train", signer_component="c02")
    return [dict(base, sign=f"s{i}", text=" ".join(f"{w}x{i}" for w in WORDS), continuous=dict(seconds=2.)) for i in range(n)]


def harness(monkeypatch, tokenizer, *, declared=True, n=CLIPS, reader=None):
    records = clip_records(n, declared)
    refs = [gappy(signer_state(i, frames=20, amplitude=.4)) for i in range(n)]
    monkeypatch.setattr(L, "open_read_list", lambda store, smoke: (records, declared and not smoke, dict(list="t", clips=n)))
    monkeypatch.setattr(L, "load_states", lambda store, recs: refs)
    monkeypatch.setattr(L, "load_tokenizer", lambda store: tokenizer)
    monkeypatch.setattr(L, "make_reader", lambda store, **k: reader or PartReader(refs))
    return records, refs


def read(tmp_path, forms, *extra):
    return P.main(["read", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w"), "--forms", ",".join(forms), *extra])


def merged(tmp_path, forms):
    out = tmp_path / f"report-{len(list(tmp_path.iterdir()))}.json"
    assert P.main(["merge", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w"), "--forms", ",".join(forms), "--output-file", str(out)]) == 0
    return json.loads(out.read_text())


# ---- pins and the lists it refuses ---------------------------------------------------------------------------------

def test_the_pins_are_the_declared_ones():
    assert P.PINS["D20_SHA"].startswith("d311703b") and L.D20_SHA == P.PINS["D20_SHA"]
    assert P.PINS["TOKENIZER_SHA"].startswith("021ee00a") and P.PINS["TRANSLATOR_SHA"].startswith("2e3ded3f")
    P.check_pins()


@pytest.mark.parametrize("name", ["D20_SHA", "D20_KEY", "TOKENIZER_SHA", "TRANSLATOR_SHA", "TRANSLATOR_RUN", "EVAL_MIX_SHA"])
@pytest.mark.parametrize("command", ["read", "merge", "fit"])
def test_a_mutated_pin_refuses_every_command_before_any_read(monkeypatch, tmp_path, name, command):
    monkeypatch.setattr(L, name, "0" * 64)
    monkeypatch.setattr(L, "open_read_list", lambda *a: pytest.fail("read after a mutated pin"))
    extra = dict(read=["--forms", "real"], merge=["--output-file", str(tmp_path / "x.json")], fit=[])[command]
    with pytest.raises(SystemExit, match="pins moved"):
        P.main([command, "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w"), *extra])


class SpyStore:
    def __init__(self, files) -> None:
        self.files, self.read = files, []

    def get(self, key):
        self.read.append(key)
        return self.files[key]

    def exists(self, key):
        return key in self.files


def test_c20_and_any_other_list_are_refused_unread_and_changed_bytes_refuse():
    store = SpyStore({"acquisition/2026-10-17-v1/youtube-asl-v1/round20/c20.json": b"{}", L.D20_KEY: b"{}"})
    with pytest.raises(SystemExit, match="D20 only"):
        L.load_read_list(store, dict(records=[]), "acquisition/2026-10-17-v1/youtube-asl-v1/round20/c20.json")
    assert store.read == []
    with pytest.raises(SystemExit, match="SHA-256"):
        L.load_read_list(store, dict(records=[]))


def test_no_command_has_an_option_for_another_list_or_instrument(tmp_path):
    for flag in ("--sentences-key", "--translator-run", "--sentence-set", "--tokenizer-key"):
        with pytest.raises(SystemExit):
            P.main(["read", "--out-dir", str(tmp_path), flag, "x"])


@pytest.mark.parametrize("change", [dict(partition="test"), dict(partition="heldout_official"), dict(signer_component="c00"), dict(partition="train"),
                                    dict(corpus="youtube-asl-v1")])
def test_a_declared_read_refuses_test_held_out_c00_and_train_records(monkeypatch, tmp_path, change):
    records, _ = harness(monkeypatch, tiny_tokenizer(), n=3)
    records[1].update(change)
    with pytest.raises(SystemExit, match="not a How2Sign val clip"):
        read(tmp_path, ["real"])


@pytest.mark.parametrize("change", [dict(partition="val"), dict(signer_component="c00"), dict(signer_component="c01")])
def test_a_smoke_reads_c02_train_clips_and_nothing_else(monkeypatch, tmp_path, change):
    records, _ = harness(monkeypatch, tiny_tokenizer(), n=3, declared=False)
    records[1].update(change)
    with pytest.raises(SystemExit, match="not a train clip"):
        read(tmp_path, ["real"], "--smoke-train", "3")


def test_a_changed_v3d_checkpoint_or_best_or_gloss_map_refuses():
    best = json.dumps(dict(checkpoint_key="ck")).encode()
    with pytest.raises(SystemExit, match="best.json"):
        P.load_v3d(SpyStore({f"training-runs/{P.V3D_RUN}/best.json": best}), "cpu")
    good = {f"training-runs/{P.V3D_RUN}/best.json": best, "ck": b"changed"}
    original = P.V3D_BEST_SHA
    P.V3D_BEST_SHA = hashlib.sha256(best).hexdigest()
    try:
        with pytest.raises(SystemExit, match="ck"):
            P.load_v3d(SpyStore(good), "cpu")
    finally:
        P.V3D_BEST_SHA = original
    with pytest.raises(SystemExit, match="gloss"):
        P.word_records(SpyStore({P.GLOSS_MAP: b"changed"}), ["BOOK"])


def test_outputs_are_write_once_and_failures_exit_nonzero(monkeypatch, tmp_path):
    harness(monkeypatch, tiny_tokenizer(), n=3)
    assert read(tmp_path, ["real", "blank"]) == 0
    with pytest.raises(SystemExit, match="write-once"):
        read(tmp_path, ["real"])
    with pytest.raises(SystemExit, match="unknown forms"):
        read(tmp_path, ["k999"])
    monkeypatch.setattr(L, "open_read_list", lambda *a: pytest.fail("an existing form must refuse before any read"))
    with pytest.raises(SystemExit, match="write-once"):
        read(tmp_path, ["no_quant", "real"])
    out = tmp_path / "r.json"
    assert P.main(["merge", "--store", "dir:x", "--out-dir", str(tmp_path / "w"), "--forms", "real,blank", "--output-file", str(out)]) == 0
    with pytest.raises(SystemExit, match="write-once"):
        P.main(["merge", "--store", "dir:x", "--out-dir", str(tmp_path / "w"), "--forms", "real,blank", "--output-file", str(out)])
    with pytest.raises(SystemExit, match="missing"):
        P.main(["merge", "--store", "dir:x", "--out-dir", str(tmp_path / "w"), "--forms", "real,blank,no_quant", "--output-file", str(tmp_path / "s.json")])
    with pytest.raises(SystemExit, match="must include blank"):
        P.main(["merge", "--store", "dir:x", "--out-dir", str(tmp_path / "w"), "--forms", "real", "--output-file", str(tmp_path / "s.json")])


# ---- planted: a decoder that returns real ----------------------------------------------------------------------------

def values_equal(a, b, name, atol=1e-5):
    x, y = a.channels[name], b.channels[name]
    np.testing.assert_array_equal(x.valid, y.valid)
    np.testing.assert_allclose(x.values[x.valid], y.values[y.valid], atol=atol)


@pytest.mark.parametrize("form", ["identity_masked", "round_trip_masked", "no_quant_masked"])
def test_a_decoder_that_returns_real_reads_real_under_real_validity(form):
    tok = Perfect(tiny_tokenizer())
    state = gappy(signer_state(1, frames=20, amplitude=.4))
    out = P.decoded_form(form, state, tok, None)
    for name in ("body", "left_hand", "right_hand"):
        values_equal(out, state, name)


def test_a_perfect_decoder_reads_real_end_to_end(monkeypatch, tmp_path):
    harness(monkeypatch, Perfect(tiny_tokenizer()), n=CLIPS)
    forms = ["real", "blank", "identity_masked", "round_trip_masked", "no_quant_masked"]
    assert read(tmp_path, forms) == 0
    r = merged(tmp_path, forms)
    assert r["checks"]["identity_masked_identical_to_real"] == 1
    for f in forms[2:]:
        assert r["forms"][f]["identical_to_real"] == 1 and r["forms"][f]["bleu"]["bleu"] == r["forms"]["real"]["bleu"]["bleu"]
        assert r["forms"][f]["paired"]["real"]["bleu"]["difference"] == 0


def test_the_round_trip_path_differs_from_real_in_validity_and_identity_shows_it():
    """The neutral fill, marked valid: why ``identity`` exists. Even a perfect decoder's unmasked form is not real on a gappy clip."""
    tok = Perfect(tiny_tokenizer())
    state = gappy(signer_state(1, frames=20, amplitude=.4))
    out = P.decoded_form("identity", state, tok, None)
    assert out.channels["left_hand"].valid[6:10].all() and not state.channels["left_hand"].valid[6:10].any()
    tracked = state.channels["left_hand"].valid
    np.testing.assert_allclose(out.channels["left_hand"].values[tracked], state.channels["left_hand"].values[tracked], atol=1e-5)
    same = P.decoded_form("round_trip", state, tok, None)
    np.testing.assert_array_equal(out.channels["body"].values, same.channels["body"].values)


def test_identity_on_a_real_tokenizer_never_reaches_its_networks():
    tok = tiny_tokenizer()
    tok.encode = lambda u: pytest.fail("identity encoded")
    state = signer_state(1, frames=20, amplitude=.4)
    out = P.decoded_form("identity", state, tok, None)
    for name in ("body", "left_hand", "right_hand"):
        values_equal(out, state, name)


# ---- planted: a blanked part ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("part,channel", [("left_hand", "left"), ("right_hand", "right"), ("body", "body")])
def test_a_blanked_part_drops_where_it_is_decoded_and_only_there(monkeypatch, tmp_path, part, channel):
    harness(monkeypatch, Perfect(tiny_tokenizer(), damage=neutral_columns(part)), n=CLIPS)
    forms = ["real", "blank"] + [f"{way}_{p}" for way in ("rt_only", "real_only") for p in ("body", "left", "right")] + ["round_trip_masked"]
    assert read(tmp_path, forms) == 0
    r = merged(tmp_path, forms)["forms"]
    real = r["real"]["bleu"]["bleu"]
    assert real > 0
    for p in ("body", "left", "right"):
        rt, kept = r[f"rt_only_{p}"]["bleu"]["bleu"], r[f"real_only_{p}"]["bleu"]["bleu"]
        if p == channel:
            assert rt < real and r[f"rt_only_{p}"]["likelihood_minus_blank"]["mean"] < r["real"]["likelihood_minus_blank"]["mean"]
            assert r[f"real_only_{p}"]["identical_to_real"] == 1           # the fault is swapped away
        else:
            assert rt == real and r[f"rt_only_{p}"]["identical_to_real"] == 1
            assert kept < real
    assert r["round_trip_masked"]["bleu"]["bleu"] < real
    assert r[f"rt_only_{channel}"]["comparator"]["form"] == r[f"real_only_{channel}"]["comparator"]["form"] == "round_trip_masked"


def test_fingers_wrist_and_arms_swap_their_own_joints_only():
    state = signer_state(0, frames=12, amplitude=.4)
    trip = signer_state(1, frames=12, amplitude=.4)
    labels = state.channels["body"].labels
    wrist = [labels.index("LeftHand"), labels.index("RightHand")]
    arms = [i for i in range(len(labels)) if i not in wrist]
    out = P.overlay(state, trip, P.swap_selection("wrist", state))
    np.testing.assert_array_equal(out.channels["body"].values[:, wrist], trip.channels["body"].values[:, wrist])
    np.testing.assert_array_equal(out.channels["body"].values[:, arms], state.channels["body"].values[:, arms])
    np.testing.assert_array_equal(out.channels["left_hand"].values, state.channels["left_hand"].values)
    out = P.overlay(state, trip, P.swap_selection("arms", state))
    np.testing.assert_array_equal(out.channels["body"].values[:, wrist], state.channels["body"].values[:, wrist])
    out = P.overlay(state, trip, P.swap_selection("fingers", state))
    np.testing.assert_array_equal(out.channels["right_hand"].values, trip.channels["right_hand"].values)
    np.testing.assert_array_equal(out.channels["body"].values, state.channels["body"].values)


def test_a_layout_without_wrist_joints_is_refused_not_guessed():
    state = signer_state(0, frames=12, amplitude=.4)
    body = state.channels["body"]
    labels = tuple(f"J{i}" for i in range(len(body.labels)))
    other = replace(state, channels={**state.channels, "body": replace(body, labels=labels)})
    with pytest.raises(SystemExit, match="does not allow"):
        P.swap_selection("wrist", other)


def test_overlay_keeps_real_validity_and_touches_no_untracked_frame():
    state = gappy(signer_state(0, frames=20, amplitude=.4))
    trip = signer_state(1, frames=20, amplitude=.4)
    out = P.overlay(state, trip, P.swap_selection("left", state))
    c = out.channels["left_hand"]
    np.testing.assert_array_equal(c.valid, state.channels["left_hand"].valid)
    assert (c.values[6:10] == 0).all() and not np.allclose(c.values[0], state.channels["left_hand"].values[0])


# ---- B1, B4: the latents each form decodes ------------------------------------------------------------------------------

def decoder_input(tok, name, state, fit=None):
    """What the decoder receives for a form."""
    seen = {}
    decode = tok.decode_latents

    def spy(q, frames):
        seen.update({p: z.clone() for p, z in q.items()})
        return decode(q, frames)
    tok.decode_latents = spy
    try:
        P.decoded_form(name, state, tok, fit)
    finally:
        tok.decode_latents = decode
    return seen


def encoded(tok, state):
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.text_to_motion import diffusion_targets
    with torch.no_grad():
        return tok.encode(diffusion_targets(tok.config.motion, collate_multichannel([state])))


def test_no_quant_decodes_the_encoder_output_and_round_trip_does_not():
    tok = tiny_tokenizer()
    state = signer_state(2, frames=20, amplitude=.4)
    z = encoded(tok, state)
    a = decoder_input(tok, "no_quant", state)
    b = decoder_input(tok, "round_trip", state)
    for p, _ in tok.parts:
        assert torch.equal(a[p], z[p]) and not torch.equal(b[p], z[p])


def test_truncation_changes_only_the_named_parts_stages():
    tok = tiny_tokenizer()
    state = signer_state(2, frames=20, amplitude=.4)
    z = encoded(tok, state)
    all_stages = tok.quantize(z)[0]
    for q in P.STAGE_GRID:
        first = tok.quantize(z, q)[0]
        seen = decoder_input(tok, f"q_left{q}", state)
        assert torch.equal(seen["left_hand"], first["left_hand"])
        for p in ("body", "right_hand", "face"):
            assert torch.equal(seen[p], all_stages[p])
    seen = decoder_input(tok, "q_hands2", state)
    assert torch.equal(seen["right_hand"], tok.quantize(z, 2)[0]["right_hand"]) and torch.equal(seen["body"], all_stages["body"])
    seen = decoder_input(tok, "h8b4f2", state)
    assert torch.equal(seen["body"], tok.quantize(z, 4)[0]["body"]) and torch.equal(seen["face"], tok.quantize(z, 2)[0]["face"])
    assert torch.equal(seen["left_hand"], all_stages["left_hand"])


def test_every_stage_form_is_declared_and_bad_names_refuse():
    assert {"q_body1", "q_left4", "q_right2", "q_hands1", "q_face4", "hands12"} <= set(P.B4_FORMS)
    tok = tiny_tokenizer()
    for bad in ("q_nose2", "zzz"):
        with pytest.raises(SystemExit, match="unknown"):
            P.stage_map(bad, tok)


def zeros_fit(tok, name="ext", part_stages=4):
    return dict(arrays={f"{name}/{p}/{q}": np.zeros((4, tok.config.code_dim), np.float32) for p in P.HANDS for q in range(part_stages)}, report={})


def test_extended_hands_with_zero_codes_equal_the_round_trip_and_real_codes_get_closer():
    tok = tiny_tokenizer()
    state = signer_state(2, frames=20, amplitude=.4)
    z = encoded(tok, state)
    full = tok.quantize(z)[0]
    seen = decoder_input(tok, "hands12", state, zeros_fit(tok))
    for p, _ in tok.parts:
        assert torch.equal(seen[p], full[p])
    residual = (z["left_hand"] - full["left_hand"])[0]
    arrays = zeros_fit(tok)["arrays"]
    for q in range(4):
        book = P.kmeans(residual, 8, iterations=3, seed=q)
        arrays[f"ext/left_hand/{q}"] = book.numpy()
        residual = residual - book[P._nearest(residual, book)[0]]
    seen = decoder_input(tok, "hands12", state, dict(arrays=arrays, report={}))
    assert (z["left_hand"] - seen["left_hand"]).pow(2).sum() < (z["left_hand"] - full["left_hand"]).pow(2).sum()
    assert torch.equal(seen["body"], full["body"])


def test_post_hoc_stages_given_d2s_own_books_are_d2s_round_trip():
    tok = tiny_tokenizer()
    state = signer_state(2, frames=20, amplitude=.4)
    arrays = {}
    for k in P.KMEANS:
        for p, _ in tok.parts:
            for q in range(tok.config.stages):
                arrays[f"k{k}/{p}/{q}"] = getattr(tok.quantizers[p], "codes")[q].numpy()
    fit = dict(arrays=arrays, report={})
    seen, d2 = decoder_input(tok, "k512", state, fit), tok.quantize(encoded(tok, state))[0]
    for p, _ in tok.parts:
        torch.testing.assert_close(seen[p], d2[p], atol=1e-5, rtol=0)


# ---- the quantisers ----------------------------------------------------------------------------------------------------

def test_kmeans_recovers_planted_clusters_and_reseeds_dead_codes():
    g = torch.Generator().manual_seed(0)
    centres = torch.randn(8, 4, generator=g) * 10
    x = centres[torch.randint(0, 8, (2000,), generator=g)] + .01 * torch.randn(2000, 4, generator=g)
    found = P.kmeans(x, 32, iterations=10, seed=1)                                # more codes than clusters: no cluster left out
    assert P._nearest(x, found)[1].mean() < .01
    assert P.perplexity(P._nearest(x, found)[0], 32) > 6
    torch.testing.assert_close(P.kmeans(x, 32, iterations=10, seed=1), found)     # the same seed, the same books
    rows = torch.tensor([[0.], [0.], [1.], [1.], [10.]])                          # the code at 50 is nearest to no row
    stepped = P._lloyd_step(rows, torch.tensor([[0.], [1.], [50.]]))
    assert stepped[:, 0].tolist()[0] == 0 and stepped[2, 0] == 10                 # a dead code goes to the farthest row
    few = P.kmeans(x[:3], 8, iterations=2, seed=1)
    assert len(few) == 8 and P._nearest(x[:3], few)[1].max() == 0


def test_more_codes_and_more_stages_leave_less_error():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3000, 8, generator=g)
    errors = []
    for k in (4, 16, 64):
        book = P.kmeans(x, k, iterations=8, seed=0)
        errors.append(float((x - P.apply_kmeans_stages(x, [book])).pow(2).mean()))
    assert errors[0] > errors[1] > errors[2]
    books, residual = [], x
    for q in range(3):
        books.append(P.kmeans(residual, 16, iterations=8, seed=q))
        residual = residual - books[-1][P._nearest(residual, books[-1])[0]]
    torch.testing.assert_close(x - P.apply_kmeans_stages(x, books), residual, atol=1e-5, rtol=0)
    assert residual.pow(2).mean() < errors[1]


def test_fsq_is_about_ten_bits_within_half_a_cell_inside_its_bound_and_keeps_shrinking_per_stage():
    assert np.prod(P.FSQ_LEVELS) == 1200 and 10 < np.log2(1200) < 10.3
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4000, 4, generator=g) @ torch.linalg.qr(torch.randn(4, 4, generator=g))[0]
    stage = P.fsq_fit(x)
    out, index = P.fsq_stage(x, stage)
    z = (x - stage["mean"]) @ stage["basis"]
    inside = (z.abs() <= stage["bound"]).all(-1)
    cell = 2 * stage["bound"] / torch.tensor(P.FSQ_LEVELS, dtype=x.dtype)
    error = ((x - out) @ stage["basis"]).abs()
    assert (error[inside] <= cell / 2 + 1e-5).all()
    assert 0 <= int(index.min()) and int(index.max()) < 1200
    # a dimension a stage cannot hold goes to the next, on its own scale: the residual keeps falling
    y = torch.randn(4000, 8, generator=g)
    stages, residual, errors = [], y, []
    for _ in range(4):
        stages.append(P.fsq_fit(residual))
        residual = residual - P.fsq_stage(residual, stages[-1])[0]
        errors.append(float(residual.pow(2).sum(-1).mean()))
    assert errors[0] > errors[1] > errors[2] > errors[3]
    torch.testing.assert_close(y - P.apply_fsq_stages(y, stages), residual, atol=1e-4, rtol=0)


# ---- the fit: c02 train only ----------------------------------------------------------------------------------------------

def fit_harness(monkeypatch, tok, n=6):
    manifest = dict(records=[dict(corpus=L.H2S, partition=part, signer_component=c, text="a b", sign=f"{part}-{c}-{i}", continuous=dict(state_key="k"))
                             for i in range(n) for part, c in (("train", "c02"), ("train", "c00"), ("train", "c01"), ("val", "c02"), ("test", "c02"))])
    monkeypatch.setattr(L, "load_manifest_at", lambda store, k, s: manifest)
    monkeypatch.setattr(L, "load_states", lambda store, recs: [signer_state(i, frames=20, amplitude=.4) for i in range(len(recs))])
    monkeypatch.setattr(L, "load_tokenizer", lambda store: tok)
    monkeypatch.setattr(P, "KMEANS", (4, 8))
    return manifest


def fit(tmp_path, *extra, name="fit"):
    return P.main(["fit", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / name), "--iterations", "2", *extra])


def test_the_fit_reads_c02_train_only_and_says_so_and_is_write_once(monkeypatch, tmp_path):
    fit_harness(monkeypatch, tiny_tokenizer())
    assert fit(tmp_path) == 0
    report = json.loads((tmp_path / "fit" / "fit.json").read_text())
    assert report["components"] == ["c02"] and report["partitions"] == ["train"] and report["clips"] == 6 and report["declared_fit"]
    assert all(s.startswith("train-c02") for s in report["signs"]) and report["report_only"]
    assert set(report["statistics"]) == {"k4", "k8", "fsq10", "hands_extension"} and report["peak_rss_bytes"] > 0
    rows = report["statistics"]["k8"]["left_hand"]
    assert len(rows) == 8 and rows[0]["mse_after"] <= rows[0]["mse_before"]
    with pytest.raises(SystemExit, match="write-once"):
        fit(tmp_path)


def test_the_fit_never_trains_the_network(monkeypatch, tmp_path):
    tok = tiny_tokenizer()
    fit_harness(monkeypatch, tok)
    before = {k: v.clone() for k, v in tok.state_dict().items()}
    assert fit(tmp_path) == 0
    assert all(torch.equal(before[k], v) for k, v in tok.state_dict().items())


def test_the_fit_keeps_stored_latents_and_refuses_them_for_other_clips(monkeypatch, tmp_path):
    fit_harness(monkeypatch, tiny_tokenizer())
    assert fit(tmp_path) == 0
    (tmp_path / "fit" / "fit.json").unlink()
    (tmp_path / "fit" / "codebooks.npz").unlink()
    calls = []
    monkeypatch.setattr(P, "collect_latents", lambda *a, **k: calls.append(1))
    assert fit(tmp_path) == 0 and not calls
    (tmp_path / "fit" / "fit.json").unlink()
    (tmp_path / "fit" / "codebooks.npz").unlink()
    with pytest.raises(SystemExit, match="other clips"):
        fit(tmp_path, "--fit-clips", "2")


def edit_fit(path, **change):
    r = json.loads((path / "fit.json").read_text())
    r.update(change)
    (path / "fit.json").write_text(json.dumps(r))


@pytest.mark.parametrize("change,message", [(dict(components=["c00"]), "c02 train"), (dict(components=["c01", "c02"]), "c02 train"), (dict(partitions=["val"]), "c02 train"),
                                            (dict(corpora=["youtube-asl-v1"]), "c02 train"), (dict(tokenizer_sha256="0" * 64), "another tokenizer"),
                                            (dict(codebooks_sha256="0" * 64), "SHA-256"), (dict(declared_fit=False), "smoke fit")])
def test_a_read_refuses_a_fit_that_is_not_what_it_must_be(monkeypatch, tmp_path, change, message):
    fit_harness(monkeypatch, tiny_tokenizer())
    assert fit(tmp_path) == 0
    edit_fit(tmp_path / "fit", **change)
    with pytest.raises(SystemExit, match=message):
        P.load_fit(tmp_path / "fit", declared=True)


def test_a_read_refuses_a_fit_holding_its_clips_and_a_smoke_may_share_them(monkeypatch, tmp_path):
    fit_harness(monkeypatch, tiny_tokenizer())
    assert fit(tmp_path) == 0
    signs = json.loads((tmp_path / "fit" / "fit.json").read_text())["signs"]
    with pytest.raises(SystemExit, match="clips of the read list"):
        P.load_fit(tmp_path / "fit", declared=True, read_signs=signs[:1])
    assert P.load_fit(tmp_path / "fit", declared=False, read_signs=signs[:1])["report"]["clips"] == 6


def test_fit_forms_need_a_fit_and_a_smoke_fit_cannot_feed_a_declared_read(monkeypatch, tmp_path):
    tok = tiny_tokenizer()
    fit_harness(monkeypatch, tok)
    assert fit(tmp_path, "--smoke-train", "2") == 0
    records, _ = harness(monkeypatch, tok, n=3)
    with pytest.raises(SystemExit, match="--fit-dir"):
        read(tmp_path, ["hands12"])
    with pytest.raises(SystemExit, match="smoke fit"):
        read(tmp_path, ["hands12"], "--fit-dir", str(tmp_path / "fit"))


def test_every_decoded_form_builds_from_a_real_fit_and_keeps_the_clips_length(monkeypatch, tmp_path):
    tok = tiny_tokenizer()
    fit_harness(monkeypatch, tok)
    assert fit(tmp_path) == 0
    fit_data = P.load_fit(tmp_path / "fit", declared=True)
    state = gappy(signer_state(3, frames=20, amplitude=.4))
    for name in [f for f in P.DECODED_FORMS if not re.fullmatch(r"k\d+", f)] + ["k4", "k8"]:
        out = P.decoded_form(name, state, tok, fit_data)
        assert len(out.timestamps) == 20
        assert not np.isnan(out.channels["body"].values).any()


# ---- the seam forms ------------------------------------------------------------------------------------------------------

def seam_state(frames=120, seed=0):
    return signer_state(seed, frames=frames, amplitude=.3)


TEXT = "one two three four five"


def test_windows_at_cuts_hold_the_exact_share_are_disjoint_and_keep_real_frames_around_them():
    for frames, cuts, total in ((120, [30, 60, 90], 54), (100, [1, 3, 5, 50, 98], 40), (40, [20], 18), (31, [10, 20], 14)):
        w = P.windows_at_cuts(cuts, total, frames)
        assert sum(b - a + 1 for a, b in w) == total
        assert w[0][0] >= 1 and w[-1][1] <= frames - 2
        assert all(x[1] + 2 <= y[0] for x, y in zip(w, w[1:]))
    assert P.windows_at_cuts([], 10, 40) == []


def test_each_window_is_centred_on_its_cut_when_there_is_room():
    w = P.windows_at_cuts([30, 60, 90], 30, 120)
    for (a, b), c in zip(w, (30, 60, 90)):
        assert a <= c <= b and abs((a + b) / 2 - c) <= 1


def test_minima_are_the_low_points_far_enough_apart_and_filled_cuts_are_counted():
    a = np.ones(100)
    for t in (20, 21, 50, 80):
        a[t] = .1
    cuts, filled = P.minima_cuts(a, 3)
    assert filled == 0 and 20 <= cuts[0] <= 21 and cuts[1:] == [50, 80]
    assert all(y - x >= 100 // 8 for x, y in zip(cuts, cuts[1:]))
    cuts, filled = P.minima_cuts(np.linspace(0, 1, 100), 4)              # a ramp has no interior minimum
    assert filled == 4 and len(cuts) == 4


def test_cuts_land_on_planted_pauses_in_a_real_like_clip():
    frames = 96
    state = seam_state(frames)
    body = state.channels["body"]
    hold = body.values.copy()
    for a, b in ((30, 36), (62, 68)):
        hold[a:b] = hold[a]
    moved = replace(state, channels={**state.channels, "body": replace(body, values=hold)})
    activity = P.wrist_activity(moved)
    cuts, filled = P.minima_cuts(activity, 2)
    assert filled == 0 and 28 <= cuts[0] <= 38 and 60 <= cuts[1] <= 70


def test_a_bridge_replaces_only_the_window_and_is_linear_between_its_ends():
    state = seam_state()
    out = P.apply_windows(state, [(20, 29), (60, 69)])
    assert len(out.timestamps) == 120
    outside = np.ones(120, bool)
    outside[20:30] = outside[60:70] = False
    for name in ("body", "left_hand", "right_hand"):
        np.testing.assert_array_equal(out.channels[name].values[outside], state.channels[name].values[outside])
    # a rotation's 6D values, between the frames around the window, renormalised by the manifold blend: first column direction
    v = out.channels["left_hand"].values[20:30, 0, :3]
    a, b = state.channels["left_hand"].values[19, 0, :3], state.channels["left_hand"].values[30, 0, :3]
    for k, row in enumerate(v):
        w = (k + 1) / 11
        expected = (1 - w) * a + w * b
        np.testing.assert_allclose(row / np.linalg.norm(row), expected / np.linalg.norm(expected), atol=1e-6)
    assert not out.channels["left_hand"].observed[20:30].any()


def test_a_bridge_across_an_untracked_stretch_keeps_what_either_side_knows():
    state = gappy(seam_state(40), 12, 20)
    out = P.apply_windows(state, [(10, 21)])
    assert out.channels["left_hand"].valid[10:22].all() and len(out.timestamps) == 40


@pytest.mark.parametrize("share", P.SHARES)
def test_seam_forms_bridge_their_share_and_controls_use_the_same_window_lengths(share):
    state = seam_state()
    forms = {r: P.seam_form(f"seam_{r}{share}", state, TEXT, 3) for r in ("min", "rnd", "fix")}
    frames = {r: v[1]["bridged_frames"] for r, v in forms.items()}
    assert frames["min"] == frames["rnd"] == frames["fix"] == round(share / 100 * 120)
    for r, (form, info) in forms.items():
        assert len(form.timestamps) == 120 and info["windows"] == 4 and info["junctions"] == 4
        assert info["path_share"] is not None and 0 <= info["path_share"] <= 1
    moved = [np.flatnonzero(np.any(forms[r][0].channels["body"].values != state.channels["body"].values, axis=(1, 2))) for r in ("min", "rnd", "fix")]
    assert not np.array_equal(moved[0], moved[1]) and not np.array_equal(moved[1], moved[2]) and not np.array_equal(moved[0], moved[2])


def test_a_form_read_alone_equals_it_read_with_others_and_seeds_depend_only_on_clip_and_form():
    state = seam_state()
    a = P.seam_form("seam_rnd30", state, TEXT, 3)[0]
    P.seam_form("seam_rnd45", state, TEXT, 3)
    b = P.seam_form("seam_rnd30", state, TEXT, 3)[0]
    c = P.seam_form("seam_rnd30", state, TEXT, 4)[0]
    np.testing.assert_array_equal(a.channels["body"].values, b.channels["body"].values)
    assert not np.array_equal(a.channels["body"].values, c.channels["body"].values)
    assert P.seam_rng(3, "seam_rnd30").integers(2 ** 32) != P.seam_rng(3, "seam_rnd45").integers(2 ** 32)


def test_random_and_fixed_windows_fit_exactly_and_refuse_what_cannot_fit():
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = P.random_windows([5, 7, 3], 40, rng)
        assert sorted(b - a + 1 for a, b in w) == [3, 5, 7] and w[0][0] >= 1 and w[-1][1] <= 38
        assert all(x[1] + 2 <= y[0] for x, y in zip(w, w[1:]))
    w = P.fixed_windows([5, 5, 5], 40)
    assert [b - a + 1 for a, b in w] == [5, 5, 5] and w[0][0] >= 1 and w[-1][1] <= 38
    with pytest.raises(ValueError):
        P.random_windows([20, 20], 40, rng)
    with pytest.raises(ValueError):
        P.fixed_windows([20, 20], 40)


def test_the_squeeze_keeps_length_bridges_about_half_the_frames_and_time_scales_its_segments():
    state = seam_state()
    form, info = P.seam_form("seam_sq196_min", state, TEXT, 0)
    assert len(form.timestamps) == 120 and info["windows"] == 4
    assert abs(info["bridged_frames"] / 120 - (1 - 1 / 1.96)) < .03
    rnd, info2 = P.seam_form("seam_sq196_rnd", state, TEXT, 0)
    assert len(rnd.timestamps) == 120
    assert not np.array_equal(form.channels["body"].values, rnd.channels["body"].values)


def test_short_clips_untracked_clips_and_one_word_sentences_are_handled_and_reported():
    form, info = P.seam_form("seam_min45", seam_state(4), TEXT, 0)
    assert info["skipped"] and info["bridged_frames"] == 0
    form, info = P.seam_form("seam_min45", seam_state(9), " ".join(["w"] * 30), 0)
    assert info["capped"] and info["junctions"] == 4 and len(form.timestamps) == 9
    form, info = P.seam_form("seam_min30", seam_state(), "word", 0)
    assert info["junctions"] == 1 and info["windows"] == 1
    from evaluate_text_to_motion import blanked
    form, info = P.seam_form("seam_min30", blanked([seam_state()])[0], TEXT, 0)
    assert info["untracked"] and info["filled"] == 4 and len(form.timestamps) == 120


def test_a_seam_read_says_how2sign_has_no_boundaries_and_names_the_fallback(monkeypatch, tmp_path):
    harness(monkeypatch, tiny_tokenizer(), n=CLIPS)
    monkeypatch.setattr(L, "load_tokenizer", lambda store: pytest.fail("a seam form reads no tokenizer"))
    forms = ["real", "blank", "seam_min30", "seam_rnd30"]
    assert read(tmp_path, forms) == 0
    r = merged(tmp_path, forms)
    assert "none" in r["boundaries"]["sign_boundaries"] and "minima" in r["boundaries"]["used"]
    assert r["boundaries"]["records_with_boundaries"] == 0
    assert r["forms"]["seam_min30"]["comparator"]["form"] == "seam_rnd30"
    assert r["forms"]["seam_min30"]["seam"]["bridged_frame_share"] == pytest.approx(.3, abs=.03)
    assert r["forms"]["seam_min30"]["bleu"]["bleu"] < r["forms"]["real"]["bleu"]["bleu"]
    assert r["forms"]["seam_min30"]["loop_share"]["mean"] is not None and r["tokenizer"] is None


def test_boundary_records_would_be_counted_not_used_silently(monkeypatch, tmp_path):
    records, _ = harness(monkeypatch, tiny_tokenizer(), n=3)
    records[0]["gloss"] = dict(label="X")
    assert read(tmp_path, ["seam_min30"]) == 0
    r = json.loads((tmp_path / "w" / "form-seam_min30.json").read_text())
    assert r["boundaries"]["records_with_boundaries"] == 1


# ---- metrics ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [("a b c a b c a b c", True), ("a a a a a", True), ("a b c a b c", False), ("", False), ("a b", False),
                                           ("A B C. a b c, A b C", True), ("a b c d a b c d e a b x", False)])
def test_loop_share_is_a_three_gram_at_least_three_times(text, expected):
    assert P.has_loop(text) is expected


def test_hand_statistics_are_compares_own_ratio():
    refs = [signer_state(i, frames=24, amplitude=.4) for i in range(5)]
    forms = [signer_state(i + 10, frames=24, amplitude=.2) for i in range(5)]
    theirs = compare(forms, refs, ("left_hand", "right_hand"))
    a, b = P.hand_statistics(forms, refs), P.hand_statistics(refs, refs)
    idx = np.random.default_rng(0).integers(0, 5, (50, 5))
    for h in P.HANDS:
        v = P.median_ratio(a[h]["variation"], b[h]["variation"], idx)
        s = P.median_ratio(a[h]["speed"], b[h]["speed"], idx)
        assert v["ratio"] == pytest.approx(theirs[h]["variation_ratio"], abs=2e-3)
        assert s["ratio"] == pytest.approx(theirs[h]["speed_ratio"], abs=2e-3)
        assert v["ci95"][0] <= v["ratio"] <= v["ci95"][1]


def test_hand_statistics_mask_real_untracked_frames_and_blank_has_none():
    from evaluate_text_to_motion import blanked
    ref = gappy(signer_state(0, frames=24, amplitude=.4))
    filled = P.hand_statistics([signer_state(0, frames=24, amplitude=.4)], [ref])
    alone = P.hand_statistics([ref], [ref])
    assert filled == alone
    none = P.hand_statistics(blanked([ref]), [ref])
    assert none["left_hand"]["variation"] == [None] and P.median_ratio([None], [1.], np.zeros((5, 1), int)) is None


def test_the_length_ratio_and_its_interval_read_hypothesis_over_reference():
    from signtranslator.eval import s0_metrics as M
    refs = ["a b"] * 10
    long, short = M.bleu_statistics(["a b c d"] * 10, refs), M.bleu_statistics(["a"] * 10, refs)
    panel = P.length_panel(long, dict(real=short), np.random.default_rng(0).integers(0, 10, (20, 10)))
    assert panel["ratio"] == 2 and panel["ci95"] == [2, 2] and panel["paired"]["real"] == dict(difference=1.5, ci95=[1.5, 1.5])


def test_every_panel_has_every_number_paired_against_real_and_blank_and_blank_motion_says_why_not(monkeypatch, tmp_path):
    harness(monkeypatch, tiny_tokenizer(), n=CLIPS)
    forms = ["real", "blank", "round_trip", "no_quant", "q_left1"]
    assert read(tmp_path, forms) == 0
    r = merged(tmp_path, forms)
    for name in forms:
        f = r["forms"][name]
        assert f["bleu"]["ci95"] and f["likelihood_minus_blank"]["ci95"] and f["loop_share"]["ci95"]
        assert f["length_ratio"]["ci95"] and set(f["length_ratio"]["paired"]) == {"real", "blank"}
        assert set(f["paired"]) == {"real", "blank"}
        for pair in f["paired"].values():
            assert set(pair) == {"bleu", "likelihood_minus_blank", "loop_share"} and all(len(v["ci95"]) == 2 for v in pair.values())
    assert r["forms"]["real"]["share_of_gap"]["likelihood"] == 1
    for name in ("real", "round_trip", "no_quant"):                              # every interval brackets its own point
        f = r["forms"][name]
        assert f["length_ratio"]["ci95"][0] <= f["length_ratio"]["ratio"] <= f["length_ratio"]["ci95"][1]
        if f["share_of_gap"]["bleu"] is not None and f["share_of_gap"]["bleu_ci95"]:
            assert f["share_of_gap"]["bleu_ci95"][0] - 1e-6 <= f["share_of_gap"]["bleu"] <= f["share_of_gap"]["bleu_ci95"][1] + 1e-6
    assert r["forms"]["blank"]["share_of_gap"]["bleu"] == 0
    speed = r["forms"]["no_quant"]["wrist_speed"]
    assert speed["over_real"]["clips"] > 0 and speed["over_blank"] is None and "blank" in speed["blank_note"]
    assert r["forms"]["blank"]["wrist_speed"]["over_real"] is None
    assert r["forms"]["blank"]["hand_variation"]["left_hand"]["over_real"] is None
    assert r["forms"]["no_quant"]["hand_variation"]["left_hand"]["over_real"]["ratio"] > 0
    assert r["forms"]["no_quant"]["comparator"]["form"] == "round_trip" and r["forms"]["q_left1"]["comparator"]["form"] == "round_trip"


def forbidden_keys(value, found=None):
    found = [] if found is None else found
    if isinstance(value, dict):
        for k, v in value.items():
            if re.search(r"pass|verdict|^bars?$|adopt|winner|decision|surviv", str(k)):
                found.append(k)
            forbidden_keys(v, found)
    elif isinstance(value, list):
        for v in value:
            forbidden_keys(v, found)
    return found


def test_the_merged_report_says_report_only_and_decides_nothing(monkeypatch, tmp_path):
    harness(monkeypatch, tiny_tokenizer(), n=CLIPS)
    forms = ["real", "blank", "round_trip"]
    assert read(tmp_path, forms) == 0
    r = merged(tmp_path, forms)
    assert r["report_only"] is True and "report only" in r["note"] and forbidden_keys(r) == []
    assert r["declared_read"] and r["smoke"] is None and r["resources"]["real"]["peak_rss_bytes"] > 0
    assert json.loads((tmp_path / "w" / "form-real.json").read_text())["report_only"] is True


def test_a_smoke_says_so_and_a_form_read_cannot_be_merged_with_a_declared_one(monkeypatch, tmp_path):
    harness(monkeypatch, tiny_tokenizer(), n=4, declared=False)
    assert read(tmp_path, ["real", "blank"], "--smoke-train", "4") == 0
    r = merged(tmp_path, ["real", "blank"])
    assert r["declared_read"] is False and "SMOKE" in r["smoke"]
    harness(monkeypatch, tiny_tokenizer(), n=4)
    assert read(tmp_path, ["no_quant"]) == 0
    with pytest.raises(SystemExit, match="smoke mixed"):
        merged(tmp_path, ["real", "blank", "no_quant"])


def test_a_form_read_alone_equals_it_read_with_others(monkeypatch, tmp_path):
    harness(monkeypatch, tiny_tokenizer(), n=6)
    assert read(tmp_path, ["real", "seam_rnd15", "q_left2"]) == 0
    other = tmp_path / "other"
    other.mkdir()
    assert read(other, ["q_left2", "seam_rnd15"]) == 0
    for form in ("q_left2", "seam_rnd15"):
        a, b = (json.loads((p / "w" / f"form-{form}.json").read_text()) for p in (tmp_path, other))
        assert a["hypotheses"] == b["hypotheses"] and a["nll"] == b["nll"] and a["hands"] == b["hands"]


# ---- check -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("change,code", [(dict(identity_masked_identical_to_real=1., blank_identical_to_real=.1), 0),
                                         (dict(identity_masked_identical_to_real=.99, blank_identical_to_real=.1), 3),
                                         (dict(identity_masked_identical_to_real=1., blank_identical_to_real=1.), 3)])
def test_check_alerts_on_the_path_failing_its_planted_case(tmp_path, change, code):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(dict(report_only=True, checks=change)))
    assert P.main(["check", "--report", str(path)]) == code


def test_check_refuses_a_report_that_does_not_say_report_only(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(dict(checks=dict(identity_masked_identical_to_real=1., blank_identical_to_real=0.))))
    assert P.main(["check", "--report", str(path)]) == 3


# ---- B6 --------------------------------------------------------------------------------------------------------------------

def word_rows(corpus, n=3, **change):
    return [dict(corpus=corpus, sign=f"{corpus}-{i}", partition=f"gate:{corpus}", source_partition="val", performer_sha256="p", signer_component=None,
                 gloss=dict(label="BOOK"), continuous=dict(state_key="k", state_sha256="s", seconds=1.), **change) for i in range(n)]


def test_the_word_read_takes_gate_halves_or_train_for_a_smoke_and_refuses_test():
    rows = word_rows("rit") + [dict(word_rows("rit", 1)[0], sign="sel", partition="select:rit"), dict(word_rows("rit", 1)[0], sign="tr", partition="train", source_partition="train")]
    declared = P.word_partitions(rows, False)
    assert [r["sign"] for r in declared["rit"]] == ["rit-0", "rit-1", "rit-2"]
    assert [r["sign"] for r in P.word_partitions(rows, True)["rit"]] == ["tr"]
    with pytest.raises(SystemExit, match="test record"):
        P.word_partitions(word_rows("rit", 1, ) + [dict(word_rows("rit", 1)[0], source_partition="test")], False)
    with pytest.raises(RuntimeError):
        P.word_partitions([dict(word_rows("how2sign-two-view-text-v1", 1)[0], signer_component="c00")], False)


def test_word_records_keep_v3ds_own_assembly_and_drop_test():
    from signtranslator.generation import recognizer_corpora as rc
    gloss_map = json.dumps(dict(mapping=dict(BOOK=dict(method="exact", candidates=["BOOKX"]))))
    sha = hashlib.sha256(gloss_map.encode()).hexdigest()
    records = lambda corpus: [dict(corpus=corpus, sign=f"{corpus}{i}", partition=p, performer_sha256="p" * 64, gloss=dict(label="BOOKX"),  # noqa: E731
                                   continuous=dict(state_key=f"k{i}", state_sha256="s", seconds=1.)) for i, p in enumerate(["train", "val", "val", "val", "val", "test"])]
    keys = [k for k, _ in P.WORD_CORPORA]
    manifests = {keys[0]: dict(records=records("rit-dai2-signs-v2")), keys[1]: dict(records=records("asllvd-dai2-signs-v1"))}
    original = P.GLOSS_MAP_SHA
    P.GLOSS_MAP_SHA = sha
    saved = L.load_manifest_at
    L.load_manifest_at = lambda store, key, s: manifests[key]
    try:
        out = P.word_records(SpyStore({P.GLOSS_MAP: gloss_map.encode()}), ["BOOK"])
    finally:
        P.GLOSS_MAP_SHA, L.load_manifest_at = original, saved
    assert {r["source_partition"] for r in out} == {"train", "val"} and {r["gloss"]["label"] for r in out} == {"BOOK"}
    assert {r["partition"].split(":")[0] for r in out} <= {"train", "gate", "select"}
    assert not any(r["sign"].endswith("5") for r in out)
    assert rc.FORBIDDEN_PARTITIONS == ("test", "heldout_official")


def test_word_states_load_as_v3ds_gates_do(monkeypatch, tmp_path):
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from tests.test_recognizer_v3_data import rec, store_with_states
    import train_sign_recognizer
    root, shas = store_with_states(tmp_path, 3)
    rows = [dict(rec(f"s{i}", "BOOK", "gate:rit"), continuous=dict(rec(f"s{i}", "BOOK")["continuous"], state_sha256=shas[i], seconds=1.)) for i in range(3)]
    rows[2]["continuous"]["seconds"] = 9.
    template = signer_state(0, frames=12)
    monkeypatch.setattr(train_sign_recognizer, "conform_template", lambda store, spec, cache: (template, dict(spec=spec)))

    class Files:
        def get(self, key):
            return (root / key).read_bytes()
    states, y, signs = P.word_states(Files(), rows, ["BOOK", "CAT"], None, ContentCache(tmp_path / "c", 2 ** 26))
    assert signs == ["s0", "s1"] and y == [0, 0] and len(states) == 2       # over v3d's 8 s: out, as its gates
    assert P.word_states(Files(), rows, ["BOOK", "CAT"], 1, ContentCache(tmp_path / "c", 2 ** 26))[2] == ["s0"]


def test_words_read_each_form_report_top5_paired_and_records_who_chose_the_list(monkeypatch, tmp_path):
    import validate_sign_recognizer as vsr
    tok = tiny_tokenizer()
    rows = {"rit": word_rows("rit", 6), "asllvd": word_rows("asllvd", 6)}
    states = [gappy(signer_state(i, frames=20, amplitude=.4)) for i in range(6)]
    monkeypatch.setattr(P, "load_v3d", lambda store, device: (object(), ["BOOK", "CAT"]))
    monkeypatch.setattr(P, "word_records", lambda store, labels: rows["rit"] + rows["asllvd"])
    monkeypatch.setattr(P, "word_states", lambda store, rs, labels, limit, cache: (states, [0] * 6, [r["sign"] for r in rs]))
    monkeypatch.setattr(L, "load_tokenizer", lambda store: tok)

    def logits(model, forms, device):                                       # reads the label when the left hand moves at all
        moved = [float(np.abs(np.diff(f.channels["left_hand"].values, axis=0)).sum()) for f in forms]
        return torch.tensor([[1., 0.] if m > 0 else [0., 1.] for m in moved])
    monkeypatch.setattr(vsr, "logits_of", logits)
    w = tmp_path / "w"
    forms = "real,blank,round_trip,no_quant"
    assert P.main(["words", "--store", "dir:x", "--out-dir", str(w), "--forms", forms]) == 0
    with pytest.raises(SystemExit, match="write-once"):
        P.main(["words", "--store", "dir:x", "--out-dir", str(w), "--forms", "real"])
    with pytest.raises(SystemExit, match="not word forms"):
        P.main(["words", "--store", "dir:x", "--out-dir", str(w), "--forms", "seam_min30"])
    words = P.load_words(w, True)
    assert set(words) == {"real", "blank", "round_trip", "no_quant"} and "--forms" in words["real"]["forms_rule"]
    panel = P.word_panel(words, "no_quant", "rit")
    assert panel["clips"] == 6 and panel["top5"]["ci95"] and panel["top5_minus_real"]["difference"] == 0
    assert P.word_panel(words, "blank", "rit")["top1"]["mean"] == 0


def test_by_default_b6_reads_every_decoded_form_since_the_plan_declares_no_survival_rule():
    assert "no survival rule" in P.__doc__
    assert set(P.DECODED_FORMS) >= set(P.B1_FORMS) | set(P.B2_FORMS) | set(P.B4_FORMS) | set(P.B5_FORMS)
    assert not set(P.DECODED_FORMS) & set(P.SEAM_FORMS)


def test_words_refuse_mixing_a_smoke_or_other_clips_and_need_anchors(tmp_path):
    def write(name, declared=True, signs=("a",)):
        (tmp_path / f"word-{name}.json").write_text(json.dumps(dict(form=name, declared_read=declared, forms_rule="r", instrument={},
                                                                    corpora=dict(rit=dict(signs=list(signs), ranks=[0] * len(signs))))))
    write("real")
    with pytest.raises(SystemExit, match="blank"):
        P.load_words(tmp_path, True)
    write("blank", declared=False)
    with pytest.raises(SystemExit, match="smoke mixed"):
        P.load_words(tmp_path, True)
    write("blank", signs=("b",))
    with pytest.raises(SystemExit, match="other clips"):
        P.load_words(tmp_path, True)


# ---- the runner: fake executables, no credentials -------------------------------------------------------------------------

JOB = ROOT / "runpod/fleet/tok_probes.sh"


def fake_bin(tmp_path, fail_when: str | None):
    """A fake ``python`` and ``uv``: it logs its arguments, prints form groups when asked, and fails on a matching command."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    pattern = "*" + "*".join(f'"{piece}"' for piece in fail_when.split("*")) + "*" if fail_when else ""
    match = f'case "$*" in {pattern}) echo "planted failure" >&2; exit 7;; esac' if fail_when else ""
    (bin_dir / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\n{match}\ncase "$*" in *" groups "*) printf "a,b\\nc,d\\n";; esac\nexit 0\n')
    (bin_dir / "uv").write_text("#!/bin/bash\nexit 0\n")
    for name in ("python", "uv"):
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    return bin_dir, log


def run_job(tmp_path, fail_when=None, **env):
    bin_dir, log = fake_bin(tmp_path, fail_when)
    environment = dict(PATH=f"{bin_dir}:/usr/bin:/bin", HOME=str(tmp_path), VENV_BIN=str(bin_dir), R2ENV=str(tmp_path / "no-such-env"), ST_REPO=str(tmp_path),
                       WORK=str(tmp_path / "work"), **env)
    done = subprocess.run(["bash", str(JOB)], env=environment, capture_output=True, text=True, timeout=60)
    return done, log.read_text() if log.exists() else ""


def done_line(result) -> int:
    lines = [x for x in result.stdout.splitlines() if x.startswith("TOK-PROBES-DONE ")]
    assert len(lines) == 1 and lines[0] == result.stdout.strip().splitlines()[-1]
    return int(lines[0].split()[1])


def test_the_runner_sees_no_credentials():
    assert not [k for k in os.environ if k.startswith(("R2_", "AWS_", "RUNPOD_"))]


def test_the_job_smokes_on_train_first_then_reads_and_ends_with_a_parseable_done_line(tmp_path):
    result, calls = run_job(tmp_path)
    assert result.returncode == 0 and done_line(result) == 0
    lines = calls.splitlines()
    first_declared = next(i for i, x in enumerate(lines) if " fit " in x and "--smoke-train" not in x)
    assert all("--smoke-train" in x or x.split()[1] not in ("fit", "words") for x in lines[:first_declared] if len(x.split()) > 1 and x.split()[1] in ("fit", "words"))
    smoke_reads = [x for x in lines[:first_declared] if " read " in x]
    assert smoke_reads and all("--smoke-train 8" in x for x in smoke_reads)
    assert any(" check " in x for x in lines[:first_declared])
    declared_reads = [x for x in lines[first_declared:] if " read " in x]
    assert declared_reads and not any("--smoke-train" in x for x in declared_reads)
    assert not any("--output-key" in x for x in lines)


def test_every_call_the_job_makes_parses_with_the_real_parser(tmp_path):
    _, calls = run_job(tmp_path)
    lines = [x.split()[1:] for x in calls.splitlines() if x.startswith("scripts/tok_probes.py ")]
    assert {x[0] for x in lines} == {"fit", "read", "words", "merge", "check", "upload", "groups"}
    for line in lines:
        P.build_parser().parse_args(line)


def test_smoke_only_stops_before_any_declared_read(tmp_path):
    result, calls = run_job(tmp_path, SMOKE="1")
    assert result.returncode == 0 and done_line(result) == 0 and "SMOKE done" in result.stdout
    assert all("--smoke-train" in x for x in calls.splitlines() if " read " in x)
    assert "d20/forms" not in calls


@pytest.mark.parametrize("fail_when,expected", [("--smoke-train 16", 2), ("--smoke-train 8", 2), ("--smoke-train 4", 2), ("smoke/forms --output-file", 2),
                                                ("check --report*smoke/report.json", 2), ("d20/fit --device", 1), ("--forms a,b --out-dir*d20", 1), ("d20/forms --output-file", 1),
                                                ("check --report*d20/report.json", 1), ("upload*d20", 1)])
def test_any_failure_exits_nonzero_with_its_done_line_and_a_background_read_is_not_lost(tmp_path, fail_when, expected):
    result, _ = run_job(tmp_path, fail_when=fail_when)
    assert result.returncode == expected and done_line(result) == expected, result.stdout + result.stderr


def test_the_job_pipes_nothing_names_no_other_list_and_is_valid_bash():
    text = JOB.read_text()
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0
    code = [line.split(" #")[0] for line in text.splitlines() if not line.startswith("#")]
    for line in code:
        assert not re.search(r"(?<!\|)\|(?!\|)", line), line
    for word in ("c20", "C20", "val1000", "eval1000", "train_", "--output-key"):
        assert word not in "\n".join(code)
    assert "set -uo pipefail" in text and "done_ 0" in text


def test_form_groups_cover_every_form_once_and_never_more_than_the_processes():
    groups = P.form_groups(4, skip=("real", "blank"))
    assert len(groups) == 4
    names = [f for g in groups for f in g.split(",")]
    assert sorted(names) == sorted(set(P.FORMS) - {"real", "blank"})
    assert P.main(["groups", "--processes", "2"]) == 0


def test_upload_is_write_once_and_refused_whole(tmp_path):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    work = tmp_path / "w"
    (work / "fit").mkdir(parents=True)
    (work / "report.json").write_text("{}")
    (work / "fit" / "latents.npz").write_bytes(b"x")
    (work / "fit" / "codebooks.npz").write_bytes(b"y")
    (work / "read-0.log").write_text("log")
    store = f"dir:{tmp_path / 's'}"
    assert P.main(["upload", "--store", store, "--out-dir", str(work), "--prefix", "p"]) == 0
    s = DirectoryStore(tmp_path / "s")
    assert s.exists("p/report.json") and s.exists("p/fit/codebooks.npz") and not s.exists("p/fit/latents.npz") and not s.exists("p/read-0.log")
    with pytest.raises(SystemExit, match="write-once"):
        P.main(["upload", "--store", store, "--out-dir", str(work), "--prefix", "p"])
