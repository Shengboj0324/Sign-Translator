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
    monkeypatch.setattr(P, "composition_entries", lambda store, recs, declared: [5] * len(recs))
    return records, refs


def read(tmp_path, forms, *extra):
    return P.main(["read", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(tmp_path / "w"), "--forms", ",".join(forms), *extra])


def plant_canary(forms_dir: Path) -> None:
    """A passed canary of the code and environment that read these forms, as the job's step would leave it."""
    reads = sorted(forms_dir.glob("form-*.json"))
    if not reads or (forms_dir / "canary.json").exists():
        return
    r = json.loads(reads[0].read_text())
    if r["declared_read"]:
        (forms_dir / "canary.json").write_text(json.dumps(dict(reproduced=True, misses=[], code_digest=r["code_digest"], environment=r["environment"])))


def merged(tmp_path, forms):
    plant_canary(tmp_path / "w")
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
    plant_canary(tmp_path / "w")
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
    assert set(r[f"rt_only_{channel}"]["comparators"]) == set(r[f"real_only_{channel}"]["comparators"]) == {"round_trip_masked"}


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
    assert set(r["forms"]["seam_min30"]["comparators"]) == {"seam_rnd30"} and r["sensitivity_only"] == ["seam_min30", "seam_rnd30"]
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

def test_the_panel_reads_loops_on_whitespace_tokens_so_a_contraction_is_one_token(monkeypatch, tmp_path):
    class Says(PartReader):
        def read(self, states, texts, candidates, group=2):
            raw = super().read(states, texts, candidates, group)
            return dict(raw, hypotheses=[("you've you've you've" if s.channels["body"].valid.any() else "x y z x y z x y z") for s in states])
    refs = harness(monkeypatch, tiny_tokenizer(), n=CLIPS)[1]
    monkeypatch.setattr(L, "make_reader", lambda store, **k: Says(refs))
    assert read(tmp_path, ["real", "blank"]) == 0
    f = merged(tmp_path, ["real", "blank"])["forms"]
    assert f["real"]["loop_share"]["mean"] == 0 and f["blank"]["loop_share"]["mean"] == 1
    assert f["real"]["paired"]["blank"]["loop_share"]["difference"] == -1


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
    assert set(r["forms"]["no_quant"]["comparators"]) == {"round_trip"} and set(r["forms"]["q_left1"]["comparators"]) == {"round_trip"}
    for name in ("real", "round_trip"):
        tracked = r["forms"][name]["tracked_share"]["hands"]
        assert 0 < tracked["mean"] <= 1 and tracked["ci95"]
    assert r["forms"]["round_trip"]["tracked_share"]["hands"]["minus_real"]["difference"] > 0     # the path marks every frame valid
    assert r["forms"]["blank"]["tracked_share"]["hands"]["mean"] == 0


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
                                         (dict(identity_masked_identical_to_real=.98, blank_identical_to_real=.1), 3),
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
    monkeypatch.setattr(P, "word_states", lambda store, rs, labels, limit, cache, longest=False: (states, [0] * 6, [r["sign"] for r in rs]))
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


def fake_bin(tmp_path, fail_when: str | None, code: int = 7):
    """A fake ``python`` and ``uv``: it logs its arguments, prints form groups when asked, and fails on a matching command."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    pattern = "*" + "*".join(f'"{piece}"' for piece in fail_when.split("*")) + "*" if fail_when else ""
    match = f'case "$*" in {pattern}) echo "planted failure" >&2; exit {code};; esac' if fail_when else ""
    (bin_dir / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\n{match}\ncase "$*" in *" groups "*) printf "a,b\\nc,d\\n";; esac\nexit 0\n')
    (bin_dir / "uv").write_text("#!/bin/bash\nexit 0\n")
    for name in ("python", "uv"):
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    return bin_dir, log


def run_job(tmp_path, fail_when=None, code=7, **env):
    bin_dir, log = fake_bin(tmp_path, fail_when, code)
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
    assert {x[0] for x in lines} == {"fit", "read", "words", "merge", "check", "upload", "groups", "restore", "canary"}
    for line in lines:
        P.build_parser().parse_args(line)


def test_smoke_only_stops_before_any_declared_read(tmp_path):
    result, calls = run_job(tmp_path, SMOKE="1")
    assert result.returncode == 0 and done_line(result) == 0 and "SMOKE done" in result.stdout
    assert all("--smoke-train" in x for x in calls.splitlines() if " read " in x)
    assert "d20/forms" not in calls


@pytest.mark.parametrize("fail_when,expected", [("--smoke-train 16", 2), ("--smoke-train 8", 2), ("--smoke-train 4", 2), ("merge*smoke-", 2), ("check --report*smoke-", 2),
                                                ("upload --out-dir*smoke-", 2), ("restore", 1), ("d20/fit --device", 1), ("--forms real,blank*d20", 1), ("tok_probes.py canary*d20", 1),
                                                ("--forms a,b --out-dir*d20", 1), ("words*d20", 1), ("d20/forms --output-file", 1),
                                                ("check --report*d20/report.json", 1), ("upload --out-dir*d20", 1)])
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


# ---- review of 172be02: composition density, healing, the bracket ------------------------------------------------------------

def test_composition_density_takes_junctions_from_the_sentences_entries_and_word_count_stays_a_sensitivity():
    state, text = seam_state(), " ".join(["w"] * 30)
    assert P.seam_form("seam_cmin45", state, text, 0, 5)[1]["junctions"] == 4
    assert P.seam_form("seam_min45", state, text, 0, 5)[1]["junctions"] == 29
    assert P.seam_form("seam_cmin30", state, text, 0, 9)[1]["bridged_frames"] == round(.3 * 120)
    assert P.seam_form("seam_csq196_min", state, text, 0, 3)[1]["windows"] == 2
    assert set(P.SEAM_COMPOSITION_FORMS) == {"seam_cmin30", "seam_cmin45", "seam_crnd30", "seam_crnd45", "seam_csq196_min", "seam_csq196_rnd", "seam_cval45", "seam_cmax45"}
    assert not set(P.SEAM_WORD_FORMS) & set(P.SEAM_COMPOSITION_FORMS)


@pytest.mark.parametrize("entries", [0, 1])
def test_a_sentence_with_no_junction_stays_real_and_says_so(entries):
    state = seam_state()
    form, info = P.seam_form("seam_cmin45", state, TEXT, 0, entries)
    assert form is state and info["skipped"] and info["entries"] == entries and info["bridged_frames"] == 0


def test_a_composition_form_without_entries_refuses():
    with pytest.raises(SystemExit, match="entries"):
        P.seam_form("seam_cmin45", seam_state(), TEXT, 0)


def test_composition_entries_are_the_stage_1_selection_on_r10s_vocabulary_and_d20s_figures_guard_them(monkeypatch):
    manifest = dict(records=[dict(corpus=c, partition="train", text=w, sign=w, continuous=dict(state_key="k", seconds=1., channels=["body"]))
                             for c, w in (("asllvd-text-v1", "dog"), ("rit-text-r6", "cat"), ("how2sign-two-view-text-v1", "fish"))])
    monkeypatch.setattr(L, "load_manifest_at", lambda store, k, s: manifest)
    monkeypatch.setattr("signtranslator.generation.compositional.usable_channels", lambda r, n: True, raising=False)
    monkeypatch.setattr("signtranslator.data_engineering.corpus_manifest.usable_channels", lambda r, n: True)
    rows = [dict(text="the dog and the cat"), dict(text="fish fish"), dict(text="dog")]
    assert P.composition_entries(None, rows, False) == [2, 0, 1]             # How2Sign sentences are no vocabulary, stop words go
    with pytest.raises(SystemExit, match="recorded"):
        P.composition_entries(None, rows, True)
    P.check_d20_entries([0] * 6 + [7] * 260 + [11] * 83 + [12] * 151)          # mean 9.09, median 7, 6 empty
    with pytest.raises(SystemExit, match="recorded"):
        P.check_d20_entries([0] * 5 + [7] * 261 + [11] * 83 + [12] * 151)


def test_the_values_only_seam_keeps_reals_validity_and_the_others_heal_tracking():
    state = gappy(seam_state(40), 12, 20)
    entries = 2
    healed, _ = P.seam_form("seam_cmin45", state, TEXT, 0, entries)
    kept, info = P.seam_form("seam_cval45", state, TEXT, 0, entries)
    for name in state.channels:
        np.testing.assert_array_equal(kept.channels[name].valid, state.channels[name].valid)
    before, after, values = (P.tracked_statistics([x])["hands"][0] for x in (state, healed, kept))
    assert after > before == values
    assert info["windows"] == 1
    moved = np.any(kept.channels["right_hand"].values != state.channels["right_hand"].values, axis=(1, 2))
    assert moved.any() and not moved[12:20].any()                             # bridged where tracked, and untracked frames stay empty


def test_a_tracked_joint_with_no_bridge_keeps_its_real_value():
    state = seam_state(20)
    form = P.apply_windows(state, [(5, 10)])
    c = form.channels["left_hand"]
    form = replace(form, channels={**form.channels, "left_hand": replace(c, valid=np.zeros_like(c.valid), observed=np.zeros_like(c.valid), confidence=np.zeros_like(c.confidence), values=np.zeros_like(c.values))})
    out = P.values_only(state, form)
    np.testing.assert_array_equal(out.channels["left_hand"].values, state.channels["left_hand"].values)


def test_the_bracket_sits_on_activity_maxima_where_the_minima_form_sits_on_holds():
    state = seam_state(96)
    body = state.channels["body"]
    hold = body.values.copy()
    for a, b in ((20, 28), (60, 68)):
        hold[a:b] = hold[a]
    moved = replace(state, channels={**state.channels, "body": replace(body, values=hold)})
    low = P.seam_form("seam_cmin45", moved, TEXT, 0, 3)[1]["path_share"]
    high = P.seam_form("seam_cmax45", moved, TEXT, 0, 3)[1]["path_share"]
    assert not np.allclose(P.seam_form("seam_cmin45", moved, TEXT, 0, 3)[0].channels["body"].values, P.seam_form("seam_cmax45", moved, TEXT, 0, 3)[0].channels["body"].values)
    activity = P.wrist_activity(moved)
    cut_low, _ = P.minima_cuts(activity, 2)
    cut_high, _ = P.minima_cuts(-activity, 2)
    assert activity[cut_high].mean() > activity[cut_low].mean() and high > low


def test_activity_reads_the_wrists_not_the_elbows():
    state = seam_state(40)
    body = state.channels["body"]
    labels = body.labels
    fixed = body.values.copy()
    for i, label in enumerate(labels):
        if "Forearm" not in label:                                            # only the forearms move: elbows still, wrists moving
            fixed[:, i] = fixed[0, i]
    moved = replace(state, channels={**state.channels, "body": replace(body, values=fixed)})
    assert P.wrist_activity(moved).min() > 0
    still = replace(state, channels={**state.channels, "body": replace(body, values=np.broadcast_to(body.values[:1], body.values.shape).copy())})
    assert P.wrist_activity(still).max() == 0


def test_the_squeeze_time_scales_each_segment():
    from signtranslator.generation.compositional import time_scale
    state = seam_state(100)
    form, windows = P.squeeze_at(state, [50])
    assert windows == [(26, 73)] or len(windows) == 1
    head = time_scale(take_frames(state, np.arange(0, 50)), 26)
    np.testing.assert_allclose(form.channels["left_hand"].values[:26], head.channels["left_hand"].values, atol=1e-9)
    np.testing.assert_allclose(form.channels["left_hand"].values[25], state.channels["left_hand"].values[49], atol=1e-9)   # last frame kept


# ---- per-stage fits, principal directions, Lloyd, part maps ---------------------------------------------------------------

@pytest.mark.parametrize("fit_name,part", [("k8", "left_hand"), ("k4", "body"), ("fsq10", "body"), ("fsq10", "right_hand"), ("hands_extension", "left_hand")])
def test_each_fit_stage_lowers_the_error_and_starts_from_the_last_stages_residual(monkeypatch, tmp_path, fit_name, part):
    fit_harness(monkeypatch, tiny_tokenizer())
    assert fit(tmp_path) == 0
    rows = json.loads((tmp_path / "fit" / "fit.json").read_text())["statistics"][fit_name][part]
    assert len(rows) == (4 if fit_name == "hands_extension" else 8)
    for a, b in zip(rows, rows[1:]):
        assert b["mse_before"] == a["mse_after"]                              # the residual, not the latents again
    assert rows[-1]["mse_after"] < rows[0]["mse_before"]
    assert all(r["mse_after"] <= r["mse_before"] * 1.0001 for r in rows)


def test_the_hand_extension_fits_d2s_residual_not_the_latents(monkeypatch, tmp_path):
    tok = tiny_tokenizer()
    fit_harness(monkeypatch, tok)
    assert fit(tmp_path) == 0
    first = json.loads((tmp_path / "fit" / "fit.json").read_text())["statistics"]["hands_extension"]["left_hand"][0]
    with np.load(tmp_path / "fit" / "latents.npz") as stored:
        z = {p: torch.as_tensor(stored[p])[None] for p in stored.files}
    residual = (z["left_hand"] - tok.quantize(z)[0]["left_hand"])[0]
    assert first["mse_before"] == pytest.approx(float(residual.pow(2).sum(-1).mean()), abs=1e-4)


def test_fsq_reads_the_top_principal_directions_with_the_most_levels_on_the_largest():
    g = torch.Generator().manual_seed(0)
    scales = torch.tensor([1., 8., .1, 4., .2, 2., .3, 1.5])                  # variance order: dims 1, 3, 5, 7
    x = torch.randn(6000, 8, generator=g) * scales
    stage = P.fsq_fit(x)
    assert [int(stage["basis"][:, k].abs().argmax()) for k in range(4)] == [1, 3, 5, 7]
    assert list(P.FSQ_LEVELS) == [8, 6, 5, 5]
    out, _ = P.fsq_stage(x, stage)
    z = (out - stage["mean"]) @ stage["basis"]
    assert [len({round(v, 3) for v in z[:, k].tolist()}) for k in range(4)] == [8, 6, 5, 5]


def test_fsq_error_leaves_the_stage_smaller_than_the_stage_it_came_from():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(3000, 4, generator=g)
    stage = P.fsq_fit(x)
    assert (x - P.fsq_stage(x, stage)[0]).pow(2).mean() < .1 * x.pow(2).mean()


def test_kmeans_runs_every_lloyd_step():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 4, generator=g)
    generator = torch.Generator().manual_seed(3)
    start = x[torch.randperm(1000, generator=generator)[:8]].clone()
    expected = start
    for _ in range(3):
        expected = P._lloyd_step(x, expected)
    found = P.kmeans(x, 8, iterations=3, seed=3)
    torch.testing.assert_close(found, expected)
    assert P._nearest(x, found)[1].mean() < P._nearest(x, P.kmeans(x, 8, iterations=1, seed=3))[1].mean()


@pytest.mark.parametrize("name,expected", [("q_face2", dict(face=2)), ("q_body1", dict(body=1)), ("q_hands4", dict(left_hand=4, right_hand=4)), ("q_right1", dict(right_hand=1)),
                                           ("h8b4f2", dict(body=4, face=2)), ("h12b6f4", dict(left_hand=12, right_hand=12, body=6, face=4)), ("hands10", dict(left_hand=10, right_hand=10))])
def test_every_stage_form_maps_the_parts_it_names(name, expected):
    tok = tiny_tokenizer()
    assert P.stage_map(name, tok) == {**{p: 8 for p, _ in tok.parts}, **expected}


def test_h12b6f4_decodes_six_body_stages_four_face_stages_and_extended_hands():
    tok = tiny_tokenizer()
    state = signer_state(2, frames=20, amplitude=.4)
    z = encoded(tok, state)
    seen = decoder_input(tok, "h12b6f4", state, zeros_fit(tok))
    assert torch.equal(seen["body"], tok.quantize(z, 6)[0]["body"]) and torch.equal(seen["face"], tok.quantize(z, 4)[0]["face"])
    assert torch.equal(seen["left_hand"], tok.quantize(z)[0]["left_hand"])    # zero extension codes


# ---- paired differences and intervals ---------------------------------------------------------------------------------------

def test_every_paired_difference_is_the_form_minus_its_anchor(monkeypatch, tmp_path):
    refs = harness(monkeypatch, Perfect(tiny_tokenizer(), damage=neutral_columns("left_hand")), n=CLIPS)[1]

    class Short(PartReader):                                                  # a damaged form also says less and loops
        def read(self, states, texts, candidates, group=2):
            raw = PartReader.read(self, states, texts, candidates, group)
            looped = [h if "x" not in h.split() else "x y z x y z x y z" for h in raw["hypotheses"]]
            return dict(raw, hypotheses=looped)
    monkeypatch.setattr(L, "make_reader", lambda store, **k: Short(refs))
    forms = ["real", "blank", "rt_only_left", "round_trip_masked", "real_only_left"]
    assert read(tmp_path, forms) == 0
    every = merged(tmp_path, forms)["forms"]
    better = every["real_only_left"]["comparators"]["round_trip_masked"]         # the fault swapped away, against the form that has it
    assert better["bleu"]["difference"] > 0 and better["likelihood_minus_blank"]["difference"] > 0 and better["loop_share"]["difference"] == -1
    f = every["rt_only_left"]
    assert f["paired"]["real"]["bleu"]["difference"] < 0 and f["paired"]["real"]["likelihood_minus_blank"]["difference"] < 0
    assert f["paired"]["real"]["loop_share"]["difference"] == 1
    assert f["length_ratio"]["paired"]["real"]["difference"] < 0
    assert f["paired"]["blank"]["likelihood_minus_blank"]["difference"] > 0
    assert f["comparators"]["round_trip_masked"]["loop_share"]["difference"] == 0
    assert f["likelihood_minus_blank"]["ci95"][0] <= f["likelihood_minus_blank"]["mean"] <= f["likelihood_minus_blank"]["ci95"][1]
    share = f["share_of_gap"]
    assert share["likelihood_ci95"][0] - 1e-6 <= share["likelihood"] <= share["likelihood_ci95"][1] + 1e-6 and 0 < share["likelihood"] < 1
    speed = merged(tmp_path, forms)["forms"]["real"]["wrist_speed"]["over_real"]
    assert speed["difference"] == 0 and speed["ratio"] == 1


def crafted_read(name, n=8, *, speed=1., nll=1.):
    refs = [f"w{i} a b c d e" for i in range(n)]
    hands = {h: dict(speed=[speed] * n, variation=[speed] * n) for h in P.HANDS}
    return dict(form=name, references=refs, hypotheses=refs if speed == 1. else ["z"] * n, nll=[[nll]] * n, seconds_over_reference=[1.] * n,
                motion=dict(wrist_speed=[speed] * n, wrist_spread=[speed] * n), hands=hands, tracked=dict(wrists=[1.] * n, hands=[1.] * n))


def test_motion_pairs_read_the_form_over_real():
    reads = dict(real=crafted_read("real"), blank=crafted_read("blank", speed=.1, nll=3.), slow=crafted_read("slow", speed=.5, nll=2.))
    f = P.form_panel(reads, "slow")
    assert f["wrist_speed"]["over_real"]["ratio"] == .5 and f["wrist_speed"]["over_real"]["difference"] == -.5
    assert f["wrist_spread"]["over_real"]["ratio"] == .5 and f["hand_variation"]["left_hand"]["over_real"]["ratio"] == .5
    assert f["hand_speed"]["right_hand"]["over_real"]["ratio"] == .5


def test_word_pairs_read_the_form_minus_its_anchor():
    def word(ranks):
        return dict(corpora=dict(rit=dict(ranks=ranks, signs=list(range(len(ranks))))))
    words = dict(real=word([0] * 6), blank=word([100] * 6), form=word([0, 0, 0, 100, 100, 100]))
    panel = P.word_panel(words, "form", "rit")
    assert panel["top5"]["mean"] == .5 and panel["top5_minus_real"]["difference"] == -.5 and panel["top5_minus_blank"]["difference"] == .5
    assert panel["top1_minus_real"]["difference"] == -.5


def test_the_declared_comparators_are_named():
    assert P.comparators_of("fsq10") == ("k1024", "round_trip")
    for k in ("k256", "k1024", "k4096"):
        assert P.comparators_of(k) == ("k512", "round_trip")
    assert P.comparators_of("k512") == ("round_trip",)
    assert P.comparators_of("no_quant") == ("round_trip", "identity")
    assert P.comparators_of("identity") == ("round_trip",) and P.comparators_of("identity_masked") == ("round_trip_masked",)
    assert P.comparators_of("no_quant_masked") == ("round_trip_masked", "identity_masked")
    assert P.comparators_of("seam_cmin45") == ("seam_crnd45",) and P.comparators_of("seam_cval45") == ("seam_cmin45", "seam_crnd45")
    assert P.comparators_of("seam_csq196_min") == ("seam_csq196_rnd",) and P.comparators_of("seam_cmax45") == ("seam_crnd45",)
    assert P.comparators_of("seam_crnd45") == () and P.comparators_of("real") == ()


def test_a_comparator_block_is_read_in_a_merge(monkeypatch, tmp_path):
    harness(monkeypatch, tiny_tokenizer(), n=CLIPS)
    forms = ["real", "blank", "round_trip", "identity", "no_quant"]
    assert read(tmp_path, forms) == 0
    f = merged(tmp_path, forms)["forms"]["no_quant"]["comparators"]
    assert set(f) == {"round_trip", "identity"} and all(len(v["bleu"]["ci95"]) == 2 for v in f.values())


# ---- B6 filter and template, fit draw, declared parameters, stored fit -----------------------------------------------------

def test_word_states_pass_v3ds_filter_template_and_rate(monkeypatch, tmp_path):
    import train_sign_recognizer
    import signtranslator.generation.sign_recognizer as sr
    seen = {}
    template = object()

    class Spy:
        def __init__(self, manifest, **kw):
            seen.update(kw)
            self.records = manifest["records"]

        def __getitem__(self, i):
            return 0, f"state{i}"
    monkeypatch.setattr(sr, "GlossMotionDataset", Spy)
    monkeypatch.setattr(train_sign_recognizer, "conform_template", lambda store, spec, cache: seen.update(spec=spec) or (template, {}))
    rows = word_rows("rit", 3)
    rows[0]["continuous"] = dict(rows[0]["continuous"], seconds=2.)
    states, _, signs = P.word_states(type("Files", (), dict(get=lambda self, k: b""))(), rows, ["BOOK"], 1, None, longest=True)
    assert seen["min_hand_fraction"] == .2 and seen["fps"] == 24. and seen["template"] is template and seen["hand_fraction_fallback"] is True
    assert seen["partition"] == "gate:rit" and seen["spec"] == P.CONFORM_TEMPLATE and signs == ["rit-0"] and P.V3D_MIN_HAND == .2


def test_the_fit_draw_is_the_declared_seeded_sample_in_order(monkeypatch):
    import random
    manifest = fit_harness(monkeypatch, tiny_tokenizer(), n=50)
    pool = [r for r in manifest["records"] if r["partition"] == "train" and r["signer_component"] == "c02"]
    assert [r["sign"] for r in P.fit_records(None, 7)] == [r["sign"] for r in random.Random(P.FIT_SEED).sample(pool, 7)]


def test_the_declared_parameters_are_these():
    assert (P.FIT_CLIPS, P.FIT_STEPS, P.FIT_ITERATIONS, P.EXTENSION_STAGES, P.SQUEEZE) == (4000, 500_000, 15, 4, 1.96)
    assert P.KMEANS == (256, 512, 1024, 4096) and P.FSQ_LEVELS == (8, 6, 5, 5) and P.SHARES == (15, 30, 45) and P.STAGE_GRID == (1, 2, 4)
    assert P.D20_ENTRIES == dict(mean=9.09, median=7, empty=6) and P.CANARY == dict(real=4.493, blank=1.68, bleu_tolerance=.05, identical=.98, nll=.001) and P.CANARY_REPORT_SHA.startswith("30ddbe7d") and P.IDENTITY_TOLERANCE == .01


def test_the_jobs_declared_calls_take_the_declared_parameters(tmp_path):
    _, calls = run_job(tmp_path)
    parsed = [P.build_parser().parse_args(x.split()[1:]) for x in calls.splitlines() if x.startswith("scripts/tok_probes.py ")]
    fit_calls = [a for a in parsed if a.command == "fit" and not a.smoke_train]
    assert len(fit_calls) == 1 and (fit_calls[0].fit_clips, fit_calls[0].fit_steps, fit_calls[0].iterations) == (4000, 500_000, 15) and fit_calls[0].device == "cuda"
    reads = [a for a in parsed if a.command == "read" and not a.smoke_train]
    assert reads and all(a.candidates == 1 and a.nll_group == 4 and a.device == "cuda" and a.smoke_longest == 0 and a.fit_dir.endswith("d20/fit") for a in reads)
    smokes = [a for a in parsed if a.command == "read" and a.smoke_train]
    assert smokes and all(a.smoke_train == 8 and a.smoke_longest == 4 for a in smokes)
    words = [a for a in parsed if a.command == "words"]
    assert [a.smoke_train for a in words] == [4, 0] and words[1].forms is None and words[1].limit == 0 and words[1].resume
    groups = [a for a in parsed if a.command == "groups"]
    assert all(a.processes == 4 and a.skip == "real,blank" for a in groups)
    assert any(a.command == "merge" and a.forms is None and a.output_file.endswith("d20/report.json") for a in parsed)


def test_a_read_records_the_fit_it_read_and_a_merge_refuses_two(monkeypatch, tmp_path):
    tok = tiny_tokenizer()
    fit_harness(monkeypatch, tok)
    assert fit(tmp_path) == 0
    sha = json.loads((tmp_path / "fit" / "fit.json").read_text())["codebooks_sha256"]
    assert sha == hashlib.sha256((tmp_path / "fit" / "codebooks.npz").read_bytes()).hexdigest()
    harness(monkeypatch, tok, n=4)
    monkeypatch.setattr(P, "FIT_FORMS", P.FIT_FORMS)
    assert read(tmp_path, ["real", "blank", "hands12", "hands10"], "--fit-dir", str(tmp_path / "fit")) == 0
    path = tmp_path / "w" / "form-hands10.json"
    r = json.loads(path.read_text())
    assert r["fit"]["codebooks_sha256"] == sha and r["fit"]["components"] == ["c02"]
    assert merged(tmp_path, ["real", "blank", "hands12", "hands10"])["fit"]["codebooks_sha256"] == sha
    r["fit"]["codebooks_sha256"] = "0" * 64
    path.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="different fits"):
        merged(tmp_path, ["real", "blank", "hands12", "hands10"])


def test_stored_latents_with_changed_bytes_refuse(monkeypatch, tmp_path):
    fit_harness(monkeypatch, tiny_tokenizer())
    assert fit(tmp_path) == 0
    (tmp_path / "fit" / "fit.json").unlink()
    (tmp_path / "fit" / "codebooks.npz").unlink()
    with open(tmp_path / "fit" / "latents.npz", "ab") as f:
        f.write(b"changed")
    with pytest.raises(SystemExit, match="SHA-256"):
        fit(tmp_path)


# ---- canary, check, memory, smoke of the longest clips -------------------------------------------------------------------------

CANARY_REFS = [f"w{i} a b c d e f" for i in range(100)]


def canary_setup(monkeypatch, tmp_path, *, declared=True, hyp_change=0, nll_shift=0., order=False):
    """A pinned stored report and forms that read it back, changed as asked. real and blank's BLEU are pinned to what they read."""
    from signtranslator.data.text_motion_dataset import DirectoryStore
    from signtranslator.eval import s0_metrics as M
    signs = [f"s{i}" for i in range(100)]
    kept = dict(real=[r if i < 60 else "z" for i, r in enumerate(CANARY_REFS)], blank=[r if i < 30 else "z" for i, r in enumerate(CANARY_REFS)])
    nll = [1. + i * .01 for i in range(100)]
    stored = dict(signs=signs, references=CANARY_REFS, forms={n: dict(hypotheses=kept[n], per_clip=dict(nll=nll)) for n in kept})
    raw = json.dumps(stored).encode()
    store = DirectoryStore(tmp_path / "s")
    store.put(P.CANARY_REPORT, raw)
    monkeypatch.setattr(P, "CANARY_REPORT_SHA", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(P, "CANARY", dict(P.CANARY, real=M.bleu_ci_fast(kept["real"], CANARY_REFS)[0], blank=M.bleu_ci_fast(kept["blank"], CANARY_REFS)[0]))
    out = tmp_path / "f"
    out.mkdir()
    for name in kept:
        hyps = list(kept[name])
        for i in range(hyp_change):
            hyps[50 + i] += " changed"
        rows = list(reversed(signs)) if order and name == "real" else signs
        (out / f"form-{name}.json").write_text(json.dumps(dict(form=name, declared_read=declared, signs=rows, references=CANARY_REFS, hypotheses=hyps, code_digest="d", environment=dict(gpu="g"),
                                                               nll=[[n + nll_shift] for n in nll])))
    return store, out


def canary(tmp_path, out):
    return P.main(["canary", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(out), "--void-prefix", "tok/void"])


def test_the_canary_passes_at_its_limits_and_writes_what_it_read(monkeypatch, tmp_path):
    store, out = canary_setup(monkeypatch, tmp_path, hyp_change=2, nll_shift=.0009)    # 98% identical, under the NLL limit: BLEU is what moved
    monkeypatch.setitem(P.CANARY, "bleu_tolerance", 5.)
    assert canary(tmp_path, out) == 0
    r = json.loads((out / "canary.json").read_text())
    assert r["reproduced"] and r["misses"] == [] and r["reads"]["real"]["identical"] == .98 and r["code_digest"] == "d" and r["stored_sha256"] == P.CANARY_REPORT_SHA
    assert not store.list("tok/void")


@pytest.mark.parametrize("kw,miss", [(dict(hyp_change=3), "hypotheses"), (dict(nll_shift=.0012), "NLL")])
def test_a_canary_miss_exits_3_voids_the_read_and_a_merge_refuses_it(monkeypatch, tmp_path, kw, miss):
    store, out = canary_setup(monkeypatch, tmp_path, **kw)
    monkeypatch.setitem(P.CANARY, "bleu_tolerance", 5.)
    assert canary(tmp_path, out) == 3
    r = json.loads((out / "canary.json").read_text())
    assert not r["reproduced"] and any(miss in m for m in r["misses"])
    assert json.loads(store.get("tok/void/d.json"))["canary"]["misses"] == r["misses"]


def test_a_bleu_miss_alone_fails_the_canary(monkeypatch, tmp_path):
    _, out = canary_setup(monkeypatch, tmp_path)
    monkeypatch.setitem(P.CANARY, "real", P.CANARY["real"] + .06)
    assert canary(tmp_path, out) == 3


def test_the_canary_reads_the_blank_too(monkeypatch, tmp_path):
    _, out = canary_setup(monkeypatch, tmp_path)
    monkeypatch.setitem(P.CANARY, "blank", P.CANARY["blank"] - .06)
    assert canary(tmp_path, out) == 3


def test_the_canary_refuses_other_clips_and_a_stored_report_that_is_not_the_pinned_one(monkeypatch, tmp_path):
    _, out = canary_setup(monkeypatch, tmp_path, order=True)
    with pytest.raises(SystemExit, match="other clips"):
        canary(tmp_path, out)
    monkeypatch.setattr(P, "CANARY_REPORT_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        canary(tmp_path, out)


def test_a_smoke_checks_the_pin_only_and_says_so(monkeypatch, tmp_path, capsys):
    _, out = canary_setup(monkeypatch, tmp_path, declared=False, hyp_change=50)
    assert canary(tmp_path, out) == 0 and "not applicable" in capsys.readouterr().out and not (out / "canary.json").exists()
    monkeypatch.setattr(P, "CANARY_REPORT_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        canary(tmp_path, out)


def test_the_identity_tolerance_takes_one_in_500_and_check_reads_the_canary(tmp_path):
    path = tmp_path / "r.json"
    def check(**checks):
        path.write_text(json.dumps(dict(report_only=True, declared_read=True, checks=dict(dict(identity_masked_identical_to_real=1., blank_identical_to_real=.1, canary=dict(reproduced=True, misses=[])), **checks))))
        return P.main(["check", "--report", str(path)])
    assert check(identity_masked_identical_to_real=.998) == 0
    assert check(identity_masked_identical_to_real=.98) == 3
    assert check(canary=dict(reproduced=False, misses=["real NLL"])) == 3
    assert check(canary=None) == 3


@pytest.mark.parametrize("bad", ["missing", "failed", "other code", "other environment"])
def test_a_declared_merge_refuses_without_a_passed_canary_of_its_own_read(monkeypatch, tmp_path, bad):
    harness(monkeypatch, tiny_tokenizer(), n=4)
    assert read(tmp_path, ["real", "blank"]) == 0
    w = tmp_path / "w"
    plant_canary(w)
    r = json.loads((w / "canary.json").read_text())
    if bad == "missing":
        (w / "canary.json").unlink()
    elif bad == "failed":
        (w / "canary.json").write_text(json.dumps(dict(r, reproduced=False)))
    elif bad == "other code":
        (w / "canary.json").write_text(json.dumps(dict(r, code_digest="z")))
    else:
        (w / "canary.json").write_text(json.dumps(dict(r, environment=dict(gpu="z"))))
    with pytest.raises(SystemExit, match="void"):
        P.main(["merge", "--store", "dir:x", "--out-dir", str(w), "--forms", "real,blank", "--output-file", str(tmp_path / "x.json")])


def test_every_report_records_peak_memory_on_the_host_and_the_gpu(monkeypatch, tmp_path):
    harness(monkeypatch, tiny_tokenizer(), n=3)
    assert read(tmp_path, ["real"]) == 0
    r = json.loads((tmp_path / "w" / "form-real.json").read_text())
    assert r["peak_rss_bytes"] > 0 and "gpu_peak_reserved_bytes" in r
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 1234)
    assert P.gpu_peak_reserved_bytes() == 1234


def test_a_smoke_also_meets_the_longest_c02_train_clips(monkeypatch, tmp_path):
    records = [dict(corpus=L.H2S, partition="train", signer_component=c, text="a", sign=f"{c}-{s}", continuous=dict(state_key="k", seconds=s))
               for c, s in (("c02", 1.), ("c02", 90.), ("c02", 30.), ("c00", 900.), ("c01", 800.))]
    monkeypatch.setattr(L, "load_manifest_at", lambda store, k, s: dict(records=records))
    assert [r["sign"] for r in P.longest_train_records(None, 2)] == ["c02-90.0", "c02-30.0"]
    harness(monkeypatch, tiny_tokenizer(), n=2, declared=False)
    monkeypatch.setattr(L, "load_manifest_at", lambda store, k, s: dict(records=records))
    monkeypatch.setattr(L, "load_states", lambda store, recs: [gappy(signer_state(i, frames=20, amplitude=.4)) for i in range(len(recs))])
    monkeypatch.setattr(L, "make_reader", lambda store, **k: type("R", (), dict(info=PartReader.info, read=lambda self, st, te, ca, group=2: dict(hypotheses=["a"] * len(st), nll=np.ones((len(st), ca.shape[1])))))())
    assert read(tmp_path, ["real"], "--smoke-train", "2", "--smoke-longest", "2") == 0
    r = json.loads((tmp_path / "w" / "form-real.json").read_text())
    assert r["signs"][-2:] == ["c02-90.0", "c02-30.0"] and r["list"]["longest"] == 2
    with pytest.raises(SystemExit, match="belongs to a smoke"):
        read(tmp_path, ["blank"], "--smoke-longest", "2")


def test_the_low_memory_loader_gives_the_same_instrument(monkeypatch, tmp_path):
    from tests.test_s0s1_scripts import tiny_translator
    torch.manual_seed(0)
    model = tiny_translator().eval()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    cache = tmp_path / "cache"
    cache.mkdir()
    torch.save(dict(kind="keypoints_to_text_t5", config=model.config.to_dict(), ema=weights), cache / f"{L.TRANSLATOR_SHA}.pt")
    monkeypatch.setattr("signtranslator.generation.keypoints_to_text_t5.build", lambda config, pretrained: tiny_translator())
    monkeypatch.setattr(L, "make_reader", lambda *a, **k: pytest.fail("the cached copy goes through the low-memory path"))
    args = type("A", (), dict(instrument_cache=str(cache), device="cpu"))()
    reader = P.load_reader(SpyStore({f"training-runs/{L.TRANSLATOR_RUN}/best.json": json.dumps(dict(checkpoint_key="ck")).encode()}), args)
    assert reader.kind == "keypoints" and reader.info["sha256"] == L.TRANSLATOR_SHA and reader.info["checkpoint_key"] == "ck"
    loaded = reader.model.state_dict()
    assert set(loaded) == set(weights) and all(torch.equal(loaded[k], weights[k]) for k in weights)
    states = [signer_state(i, frames=16, amplitude=.4) for i in range(2)]
    from score_back_translation import translate
    assert translate(reader.model, "keypoints", states) == translate(model, "keypoints", states)
    torch.save(dict(kind="another", config={}, ema={}), cache / f"{L.TRANSLATOR_SHA}.pt")
    with pytest.raises(SystemExit, match="kind"):
        P.load_reader(SpyStore({}), args)
    (cache / f"{L.TRANSLATOR_SHA}.pt").unlink()
    monkeypatch.setattr(L, "make_reader", lambda store, **k: k)
    assert P.load_reader(SpyStore({}), args)["sha256"] == L.TRANSLATOR_SHA      # no copy yet: the original, hash-checked


# ---- lost work: exit trap, partial upload, restore --------------------------------------------------------------------------

def test_a_failed_job_uploads_its_work_to_partial_keeps_its_status_and_ends_with_done(tmp_path):
    result, calls = run_job(tmp_path, fail_when="d20/fit --device")
    assert result.returncode == 1 and done_line(result) == 1
    partial = [x for x in calls.splitlines() if " upload --partial " in x]
    assert partial and all("/partial/" in x for x in partial)
    assert " upload --partial " in calls.splitlines()[-1]                       # on exit, after the step that failed


def test_a_passing_job_uploads_no_partial_on_exit_and_a_failed_partial_upload_hides_no_status(tmp_path):
    result, calls = run_job(tmp_path, SMOKE="1")
    assert result.returncode == 0 and " upload --partial " not in calls.split("SMOKE")[0].splitlines()[-1]
    other = tmp_path / "second"
    other.mkdir()
    result, _ = run_job(other, fail_when="--partial")                          # the job itself passes; a partial sync's failure is only an alert
    assert result.returncode == 0 and "ALERT partial upload failed" in result.stdout and done_line(result) == 0


def test_a_terminated_job_still_uploads_and_says_143(tmp_path):
    bin_dir, log = fake_bin(tmp_path, None)
    (bin_dir / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\ncase "$*" in *d20/fit*) kill -TERM $PPID; sleep 1; exit 0;; esac\ncase "$*" in *" groups "*) printf "a,b\\n";; esac\nexit 0\n')
    env = dict(PATH=f"{bin_dir}:/usr/bin:/bin", HOME=str(tmp_path), VENV_BIN=str(bin_dir), R2ENV=str(tmp_path / "no-such-env"), ST_REPO=str(tmp_path), WORK=str(tmp_path / "work"))
    done = subprocess.run(["bash", str(JOB)], env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 143 and done_line(done) == 143 and " upload --partial " in log.read_text()


def test_a_retry_with_a_finished_stage_stored_skips_it(tmp_path):
    forms = tmp_path / "work" / "d20" / "forms"
    fits = tmp_path / "work" / "d20" / "fit"
    forms.mkdir(parents=True)
    fits.mkdir(parents=True)
    (forms / "form-real.json").write_text("{}")
    (forms / "form-blank.json").write_text("{}")
    (fits / "fit.json").write_text("{}")
    _, calls = run_job(tmp_path)
    lines = calls.splitlines()
    restore = next(i for i, x in enumerate(lines) if " restore " in x)
    after = lines[restore:]
    assert not any(" fit " in x for x in after) and not any("--forms real,blank" in x for x in after)
    assert any(" read --forms a,b" in x for x in after) and any("--stored-in" in x and "d20/forms" in x for x in after)


def test_groups_leave_out_stored_forms_and_words_resume(tmp_path, monkeypatch, capsys):
    (tmp_path / "form-no_quant.json").write_text("{}")
    groups = P.form_groups(3, ("real", "blank"), str(tmp_path))
    names = [f for g in groups for f in g.split(",")]
    assert "no_quant" not in names and "identity" in names
    import validate_sign_recognizer as vsr
    calls = []
    rows = word_rows("rit", 2)
    states = [signer_state(i, frames=16, amplitude=.4) for i in range(2)]
    monkeypatch.setattr(P, "load_v3d", lambda store, device: (object(), ["BOOK"]))
    monkeypatch.setattr(P, "word_records", lambda store, labels: rows)
    monkeypatch.setattr(P, "word_states", lambda store, rs, labels, limit, cache, longest=False: (states, [0, 0], ["rit-0", "rit-1"]))
    monkeypatch.setattr(L, "load_tokenizer", lambda store: tiny_tokenizer())
    monkeypatch.setattr(vsr, "logits_of", lambda model, forms, device: calls.append(1) or torch.ones(len(forms), 1))
    w = tmp_path / "w"
    assert P.main(["words", "--store", "dir:x", "--out-dir", str(w), "--forms", "real,blank"]) == 0
    calls.clear()
    assert P.main(["words", "--store", "dir:x", "--out-dir", str(w), "--forms", "real,blank,no_quant", "--resume"]) == 0
    assert len(calls) == 1 and (w / "word-no_quant.json").exists()


PLACE = dict(python="3.12", torch="2", transformers="4.5", tokenizers="0.2", gpu="L40S")
RUN = "tok/partial/20261101T000000Z"


def stored_report(digest="abc", place=PLACE, declared=True, **extra):
    return json.dumps(dict(code_digest=digest, environment=place, declared_read=declared, **extra)).encode()


def partial_store(tmp_path, digest="abc", *, declared=True, books=b"books", canary=True, place=PLACE, run=RUN):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "s")
    prefix = f"{run}/d20"
    store.put(f"{prefix}/forms/form-real.json", stored_report(digest, place, declared, form="real"))
    store.put(f"{prefix}/forms/form-blank.json", stored_report(digest, place, declared, form="blank"))
    store.put(f"{prefix}/forms/form-no_quant.json", stored_report(digest, place, declared, form="no_quant"))
    store.put(f"{prefix}/forms/form-identity.json", b'{"half a fi')
    store.put(f"{prefix}/forms/word-real.json", stored_report(digest, place, declared, form="real"))
    if canary is not None:
        store.put(f"{prefix}/forms/canary.json", stored_report(digest, place, declared, reproduced=canary))
    store.put(f"{prefix}/fit/fit.json", json.dumps(dict(code_digest=digest, environment=place, declared_fit=declared, codebooks_sha256=hashlib.sha256(b"books").hexdigest())).encode())
    store.put(f"{prefix}/fit/codebooks.npz", books)
    store.put(f"{run}/smoke-x/forms/form-seam_min15.json", stored_report(digest, place, False, form="seam_min15"))
    return store


def restore(tmp_path, out="out", *extra):
    return P.main(["restore", "--store", f"dir:{tmp_path / 's'}", "--prefix", "tok/partial", "--out-dir", str(tmp_path / out), "--final-key", "tok/report.json", "--void-prefix", "tok/void", *extra])


@pytest.fixture
def this_run(monkeypatch):
    monkeypatch.setattr(P, "code_digest", lambda: "abc")
    monkeypatch.setattr(P, "environment", lambda: PLACE)


def restored(tmp_path, out="out"):
    return sorted(str(p.relative_to(tmp_path / out)) for p in (tmp_path / out).rglob("*") if p.is_file()) if (tmp_path / out).exists() else []


def test_restore_brings_back_one_runs_d20_work_and_never_half_a_file_or_a_smoke(tmp_path, this_run):
    partial_store(tmp_path)
    assert restore(tmp_path) == 0
    assert restored(tmp_path) == ["fit/codebooks.npz", "fit/fit.json", "forms/form-blank.json", "forms/form-no_quant.json", "forms/form-real.json", "forms/word-real.json"]


def test_restore_reads_the_latest_run_only(tmp_path, this_run):
    store = partial_store(tmp_path, run="tok/partial/20261101T000000Z")
    store.put("tok/partial/20261102T000000Z/d20/forms/form-round_trip.json", stored_report(form="round_trip"))
    assert restore(tmp_path) == 0
    assert restored(tmp_path) == ["forms/form-round_trip.json"]


@pytest.mark.parametrize("kw", [dict(digest="other"), dict(declared=False), dict(place=dict(PLACE, transformers="4.6")), dict(place=dict(PLACE, tokenizers="0.3")),
                                dict(place=dict(PLACE, gpu="A5000")), dict(place=dict(PLACE, torch="3")), dict(place=dict(PLACE, python="3.11")), dict(canary=False)])
def test_restore_refuses_other_code_another_environment_a_smoke_and_a_failed_canary(tmp_path, this_run, capsys, kw):
    partial_store(tmp_path, **kw)
    assert restore(tmp_path) == 0
    assert restored(tmp_path) == [] and "ALERT" in capsys.readouterr().out


def test_real_and_blank_are_not_restored_from_a_run_with_no_canary(tmp_path, this_run):
    partial_store(tmp_path, canary=None)
    assert restore(tmp_path) == 0
    found = restored(tmp_path)
    assert "forms/form-real.json" not in found and "forms/form-blank.json" not in found and "forms/form-no_quant.json" in found


def test_restore_refuses_a_changed_codebook_and_a_finished_read(tmp_path, this_run):
    store = partial_store(tmp_path, books=b"changed")
    assert restore(tmp_path) == 0
    assert "fit/fit.json" not in restored(tmp_path) and "forms/form-real.json" in restored(tmp_path)
    store.put("tok/report.json", b"{}")
    with pytest.raises(SystemExit, match="finished"):
        restore(tmp_path, "again")


def test_restore_stops_on_local_forms_of_other_code_at_once(tmp_path, this_run):
    partial_store(tmp_path)
    forms = tmp_path / "out" / "forms"
    forms.mkdir(parents=True)
    (forms / "form-real.json").write_bytes(stored_report("other"))
    with pytest.raises(SystemExit, match="other code"):
        restore(tmp_path)


def test_a_voided_code_never_reads_again_and_new_code_needs_a_recorded_cause(tmp_path, this_run):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "s")
    store.put("tok/void/abc.json", b"{}")
    with pytest.raises(SystemExit, match="void"):
        restore(tmp_path, "out", "--rerun-cause", "same code")
    store2 = DirectoryStore(tmp_path / "t")
    (tmp_path / "s" / "tok" / "void" / "abc.json").unlink()
    store.put("tok/void/older.json", b"{}")
    with pytest.raises(SystemExit, match="rerun-cause"):
        restore(tmp_path)
    assert restore(tmp_path, "out", "--rerun-cause", "transformers moved; pinned") == 0
    assert json.loads((tmp_path / "out/forms/rerun.json").read_text()) == dict(cause="transformers moved; pinned", voided=["tok/void/older.json"], code_digest="abc")
    assert restore(tmp_path, "out", "--rerun-cause", "again") == 0 and not store2.exists("x")


def test_the_report_records_a_reruns_cause(monkeypatch, tmp_path):
    harness(monkeypatch, tiny_tokenizer(), n=4)
    assert read(tmp_path, ["real", "blank"]) == 0
    (tmp_path / "w" / "rerun.json").write_text(json.dumps(dict(cause="c")))
    assert merged(tmp_path, ["real", "blank"])["rerun"] == dict(cause="c")


def test_a_partial_upload_leaves_stored_keys_alone_and_a_merge_refuses_two_codes(tmp_path, monkeypatch):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    work = tmp_path / "w"
    work.mkdir()
    (work / "a.json").write_text("new")
    (work / "b.json").write_text("new")
    store = DirectoryStore(tmp_path / "s")
    store.put("p/a.json", b"old")
    args = ["upload", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(work), "--prefix", "p"]
    with pytest.raises(SystemExit, match="write-once"):
        P.main(args)
    assert not store.exists("p/b.json")
    assert P.main(args + ["--partial"]) == 0
    assert store.get("p/a.json") == b"old" and store.get("p/b.json") == b"new"
    other = tmp_path / "empty"
    other.mkdir()
    assert P.main(["upload", "--store", f"dir:{tmp_path / 's'}", "--out-dir", str(other), "--prefix", "p", "--partial"]) == 0
    harness(monkeypatch, tiny_tokenizer(), n=3)
    monkeypatch.setattr(P, "code_digest", lambda: "one")
    assert read(tmp_path, ["real"]) == 0
    monkeypatch.setattr(P, "code_digest", lambda: "two")
    assert read(tmp_path, ["blank"]) == 0
    with pytest.raises(SystemExit, match="different code digests"):
        merged(tmp_path, ["real", "blank"])


def test_a_write_leaves_no_half_file_under_the_reports_name(tmp_path, monkeypatch):
    path = tmp_path / "x" / "form-real.json"
    P.write_report_once(path, dict(a=1))
    assert json.loads(path.read_text()) == dict(a=1) and [p.name for p in path.parent.iterdir()] == ["form-real.json"]
    with pytest.raises(SystemExit, match="write-once"):
        P.write_report_once(path, dict(a=2))


def test_the_report_discloses_what_a_reader_needs(monkeypatch, tmp_path):
    harness(monkeypatch, tiny_tokenizer(), n=4)
    assert read(tmp_path, ["real", "blank"]) == 0
    d = merged(tmp_path, ["real", "blank"])["disclosures"]
    assert "20.0" in d["fsq"] and "V3-F" in d["fsq"] and "favours d2" in d["v3d"] and "119" in d["b6"] and "197" in d["b6"] and "rt_only_arms" in d["b2"]
    assert "99.5th percentile" in P.__doc__ and "0.5-99.5" not in P.__doc__


# ---- review of 366f213: the code digest, the environment, B1's block, the rule apart from the job ---------------------------

def digest_of(root: Path, monkeypatch) -> str:
    monkeypatch.setattr(P, "ROOT", root)
    P.code_digest.cache_clear()
    try:
        return P.code_digest()
    finally:
        P.code_digest.cache_clear()


def test_the_code_digest_names_scripts_and_the_package_and_catches_a_dirty_tree_without_git(tmp_path, monkeypatch):
    root = tmp_path / "code"
    (root / "scripts").mkdir(parents=True)
    (root / "signtranslator" / "eval").mkdir(parents=True)
    (root / "scripts" / "a.py").write_text("1")
    (root / "signtranslator" / "eval" / "b.py").write_text("2")
    first = digest_of(root, monkeypatch)
    assert len(first) == 64 and not (root / ".git").exists() and digest_of(root, monkeypatch) == first
    (root / "signtranslator" / "eval" / "b.py").write_text("3")
    second = digest_of(root, monkeypatch)
    assert second != first
    (root / "scripts" / "a.py").write_text("4")
    assert digest_of(root, monkeypatch) != second
    third = digest_of(root, monkeypatch)
    (root / "scripts" / "notes.txt").write_text("x")
    (root / "scripts" / "sub").mkdir()
    (root / "scripts" / "sub" / "c.py").write_text("x")
    assert digest_of(root, monkeypatch) == third                                 # scripts/*.py, not deeper, and code only
    (root / "signtranslator" / "eval" / "b.py").rename(root / "signtranslator" / "eval" / "c.py")
    assert digest_of(root, monkeypatch) != third                                 # a path is part of the code
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="void"):
        digest_of(empty, monkeypatch)


def test_every_report_names_its_code_and_environment(monkeypatch, tmp_path):
    harness(monkeypatch, tiny_tokenizer(), n=4)
    monkeypatch.setattr(P, "environment", lambda: PLACE)
    assert read(tmp_path, ["real", "blank"]) == 0
    r = json.loads((tmp_path / "w/form-real.json").read_text())
    assert r["code_digest"] == P.code_digest() and r["environment"] == PLACE
    report = merged(tmp_path, ["real", "blank"])
    assert report["code_digest"] == P.code_digest() and report["environment"] == PLACE


def test_the_environment_names_what_can_move_a_read(monkeypatch):
    place = P.environment()
    assert set(place) == {"python", "torch", "transformers", "tokenizers", "gpu"} and place["python"] and place["torch"] == torch.__version__
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "L40S")
    assert P.environment()["gpu"] == "L40S"


def rewrite(path, **change):
    path.write_text(json.dumps(dict(json.loads(path.read_text()), **change)))


@pytest.mark.parametrize("change,message", [(dict(code_digest=None), "void"), (dict(code_digest="z"), "code digests"), (dict(environment=dict(PLACE, transformers="9")), "environments")])
def test_a_merge_refuses_a_form_of_no_code_other_code_or_another_environment(monkeypatch, tmp_path, change, message):
    harness(monkeypatch, tiny_tokenizer(), n=4)
    assert read(tmp_path, ["real", "blank"]) == 0
    plant_canary(tmp_path / "w")
    rewrite(tmp_path / "w/form-blank.json", **change)
    with pytest.raises(SystemExit, match=message):
        merged(tmp_path, ["real", "blank"])


def word_read(tmp_path, monkeypatch, tok):
    import validate_sign_recognizer as vsr
    rows = word_rows("rit", 2)
    states = [signer_state(i, frames=16, amplitude=.4) for i in range(2)]
    monkeypatch.setattr(P, "load_v3d", lambda store, device: (object(), ["BOOK"]))
    monkeypatch.setattr(P, "word_records", lambda store, labels: rows)
    monkeypatch.setattr(P, "word_states", lambda store, rs, labels, limit, cache, longest=False: (states, [0, 0], ["rit-0", "rit-1"]))
    monkeypatch.setattr(vsr, "logits_of", lambda model, forms, device: torch.ones(len(forms), 1))
    return P.main(["words", "--store", "dir:x", "--out-dir", str(tmp_path / "w"), "--forms", "real,blank"])


@pytest.mark.parametrize("change,message", [(dict(code_digest="z"), "code digests"), (dict(environment=dict(PLACE)), "environments")])
def test_word_reports_must_be_one_run_with_the_forms(monkeypatch, tmp_path, change, message):
    tok = tiny_tokenizer()
    harness(monkeypatch, tok, n=4)
    assert read(tmp_path, ["real", "blank"]) == 0
    assert word_read(tmp_path, monkeypatch, tok) == 0
    assert merged(tmp_path, ["real", "blank"])["words"] is not None
    rewrite(tmp_path / "w/word-blank.json", **change)
    with pytest.raises(SystemExit, match=message):
        merged(tmp_path, ["real", "blank"])


def test_word_reports_of_other_code_than_the_forms_refuse(monkeypatch, tmp_path):
    tok = tiny_tokenizer()
    harness(monkeypatch, tok, n=4)
    assert read(tmp_path, ["real", "blank"]) == 0
    assert word_read(tmp_path, monkeypatch, tok) == 0
    for name in ("real", "blank"):
        rewrite(tmp_path / f"w/word-{name}.json", code_digest="z")
    with pytest.raises(SystemExit, match="other code"):
        merged(tmp_path, ["real", "blank"])


def test_word_reports_must_read_the_forms_fit(tmp_path):
    good = dict(codebooks_sha256="a")
    (tmp_path / "word-real.json").write_text(json.dumps(dict(form="real", declared_read=True, corpora={}, fit=dict(codebooks_sha256="other"), code_digest="d", environment={})))
    (tmp_path / "word-blank.json").write_text(json.dumps(dict(form="blank", declared_read=True, corpora={}, fit=None, code_digest="d", environment={})))
    with pytest.raises(SystemExit, match="another fit"):
        P.load_words(tmp_path, True, "d", {}, good)
    assert set(P.load_words(tmp_path, True, "d", {}, dict(codebooks_sha256="other"))) == {"real", "blank"}


@pytest.mark.parametrize("change", [dict(code_digest="z"), dict(environment=PLACE)])
def test_a_fit_of_other_code_or_another_environment_refuses(monkeypatch, tmp_path, change):
    fit_harness(monkeypatch, tiny_tokenizer())
    assert fit(tmp_path) == 0
    edit_fit(tmp_path / "fit", **change)
    with pytest.raises(SystemExit, match="other code or in another environment"):
        P.load_fit(tmp_path / "fit", declared=False)


def test_a_failed_canary_stops_the_job_before_any_other_form_with_its_own_status(tmp_path):
    result, calls = run_job(tmp_path, fail_when="tok_probes.py canary*d20", code=3)
    assert result.returncode == 3 and done_line(result) == 3 and "do not retry" in result.stdout
    lines = calls.splitlines()
    at = next(i for i, x in enumerate(lines) if " canary " in x and "d20" in x)
    assert all(" upload --partial " in x for x in lines[at + 1:]) and lines[at + 1:]        # only the failed run's work goes up: no form, no words, no merge
    assert not any("--forms a,b" in x and "d20" in x for x in lines)


def test_a_canary_that_could_not_run_is_an_ordinary_failure_and_a_rerun_passes_its_cause(tmp_path):
    result, _ = run_job(tmp_path, fail_when="tok_probes.py canary*d20")
    assert result.returncode == 1
    other = tmp_path / "second"
    other.mkdir()
    _, calls = run_job(other, RERUN_CAUSE="pinned transformers")
    restore_line = next(x for x in calls.splitlines() if " restore " in x)
    assert "--rerun-cause pinned transformers" in restore_line and "--void-prefix" in restore_line


def test_identity_is_paired_with_the_round_trip_in_a_merge(monkeypatch, tmp_path):
    harness(monkeypatch, tiny_tokenizer(), n=CLIPS)
    forms = ["real", "blank", "round_trip", "identity", "round_trip_masked", "identity_masked"]
    assert read(tmp_path, forms) == 0
    f = merged(tmp_path, forms)["forms"]
    assert set(f["identity"]["comparators"]) == {"round_trip"} and len(f["identity"]["comparators"]["round_trip"]["bleu"]["ci95"]) == 2
    assert set(f["identity_masked"]["comparators"]) == {"round_trip_masked"}


def b1_reads(no_quant=.5):
    """Three forms where a known share of clips is right; identity all, round trip none of the planted ones."""
    n = 100
    refs = [f"w{i} a b c d e f" for i in range(n)]
    def form(name, right, nll):
        return dict(form=name, references=refs, hypotheses=[r if i < right else "z" for i, r in enumerate(refs)], nll=[[nll]] * n)
    return {"blank": form("blank", 0, 3.), "round_trip": form("round_trip", 20, 2.), "identity": form("identity", 60, 1.), "no_quant": form("no_quant", 20 + int(40 * no_quant), 2. - no_quant)}


def test_b1s_block_gives_d_and_p_on_both_metrics_as_facts():
    from signtranslator.eval import s0_metrics as M
    reads = b1_reads(.5)
    block = P.b1_block(reads, "")
    refs = reads["no_quant"]["references"]
    bleu = {k: float(M.bleu_from_statistics(M.bleu_statistics(reads[k]["hypotheses"], refs).sum(0))) for k in ("round_trip", "identity", "no_quant")}
    assert block["bleu"]["d"] == pytest.approx(bleu["identity"] - bleu["round_trip"], abs=1e-3)
    assert block["bleu"]["p"] == pytest.approx((bleu["no_quant"] - bleu["round_trip"]) / (bleu["identity"] - bleu["round_trip"]), abs=1e-3)
    assert block["bleu"]["p_ci95"][0] <= block["bleu"]["p"] <= block["bleu"]["p_ci95"][1]
    assert block["bleu"]["no_quant_minus_round_trip"]["difference"] > 0 and block["bleu"]["identity_minus_no_quant"]["difference"] > 0 and block["bleu"]["identity_minus_round_trip"]["difference"] > 0
    likelihood = block["likelihood_minus_blank"]
    assert likelihood["d"] == 1 and likelihood["p"] == .5 and likelihood["no_quant_minus_round_trip"]["difference"] == .5 and likelihood["identity_minus_no_quant"]["difference"] == .5
    assert P.b1_block({k: v for k, v in reads.items() if k != "identity"}, "") is None


def test_a_report_exposes_every_field_the_declaration_reads(monkeypatch, tmp_path):
    harness(monkeypatch, tiny_tokenizer(), n=CLIPS)
    forms = ["real", "blank", "round_trip", "identity", "no_quant", "round_trip_masked", "identity_masked", "no_quant_masked", "seam_cval45", "seam_cmax45", "seam_crnd45"]
    assert read(tmp_path, forms) == 0
    r = merged(tmp_path, forms)
    for name in ("seam_cval45", "seam_cmax45", "seam_crnd45"):
        share = r["forms"][name]["share_of_gap"]
        assert {"bleu", "bleu_ci95", "likelihood", "likelihood_ci95"} <= set(share)
    seam = r["forms"]["seam_cval45"]
    assert {"bridged_frame_share", "frames_per_bridge", "filled_cuts", "path_share", "junctions_per_clip"} <= set(seam["seam"])
    assert len(seam["paired"]["real"]["loop_share"]["ci95"]) == 2
    for view in ("unmasked", "masked"):
        block = r["b1"][view]
        for metric in ("bleu", "likelihood_minus_blank"):
            assert {"d", "p", "p_ci95", "no_quant_minus_round_trip", "identity_minus_round_trip", "identity_minus_no_quant"} <= set(block[metric])
    assert r["b1"]["masked"]["forms"]["no_quant"] == "no_quant_masked"


def test_the_probe_script_holds_no_rule_and_the_job_never_calls_one():
    source = (ROOT / "scripts/tok_probes.py").read_text()
    for word in ("P_DECODER", "SEAM_KEEP", "fall_shown", "no_fall_shown", "unresolved"):
        assert word not in source
    assert not re.search(r"(?<![0-9.])0?\.(85|75|25)(?![0-9])", re.sub(r'"[^"]*"|"""[\s\S]*?"""', "", source))
    assert "tok_probes_rule" not in JOB.read_text()


# ---- the draft rule, apart from the probe: one synthetic report per outcome --------------------------------------------------

import tok_probes_rule as Rule  # noqa: E402


def rule_report(*, p=.5, d_low=.1, gain_low=.1, rest_low=.1, share=.9, high=.97, loop_low=-.1, cmax_high=.95):
    return dict(report_only=True, declared_read=True, code_digest="d", checks=dict(canary=dict(reproduced=True)),
                b1=dict(unmasked=dict(bleu=dict(p=p, identity_minus_round_trip=dict(ci95=[d_low, 1.]), no_quant_minus_round_trip=dict(ci95=[gain_low, 1.]), identity_minus_no_quant=dict(ci95=[rest_low, 1.])))),
                forms=dict(seam_cval45=dict(share_of_gap=dict(bleu=share, bleu_ci95=[share - .1, high], likelihood=.5), paired=dict(real=dict(loop_share=dict(difference=.1, ci95=[loop_low, .2]))),
                                            seam=dict(bridged_frame_share=.45, frames_per_bridge=7., filled_cuts=2)),
                           seam_cmax45=dict(share_of_gap=dict(bleu=.8, bleu_ci95=[.7, cmax_high]))))


@pytest.mark.parametrize("kw,expected", [(dict(d_low=0.), "void"), (dict(d_low=-.1, p=.1), "void"), (dict(p=None), "void"), (dict(p=.25), "decoder"), (dict(p=-.2), "decoder"),
                                         (dict(p=.75), "quantiser"), (dict(p=1.2), "quantiser"), (dict(p=.5), "unresolved"), (dict(p=.26), "unresolved"), (dict(p=.74), "unresolved"),
                                         (dict(p=.1, rest_low=0.), "unresolved"), (dict(p=.9, gain_low=0.), "unresolved")])
def test_each_b1_outcome(kw, expected):
    assert Rule.read_rule(rule_report(**kw))["b1"]["outcome"] == expected


def test_b1_is_void_without_its_forms():
    report = rule_report()
    report["b1"]["unmasked"] = None
    assert Rule.read_rule(report)["b1"]["outcome"] == "void"


@pytest.mark.parametrize("kw,expected", [(dict(share=.7, high=.84), "fall_shown"), (dict(share=.84, high=.8499), "fall_shown"), (dict(share=.85, high=.93), "no_fall_shown"), (dict(share=.95, high=.99), "no_fall_shown"),
                                         (dict(share=.8, high=.9), "unresolved"), (dict(share=.84, high=.85), "unresolved")])
def test_each_seam_outcome(kw, expected):
    assert Rule.read_rule(rule_report(**kw))["seam"]["outcome"] == expected


def test_a_keep_is_weak_when_the_bracket_falls_and_loops_are_reported_beside_it():
    keep = Rule.read_rule(rule_report(share=.9, high=.97, cmax_high=.84))["seam"]
    assert keep["outcome"] == "no_fall_shown" and keep["weak_keep"] is True
    assert Rule.read_rule(rule_report(share=.9, high=.97, cmax_high=.85))["seam"]["weak_keep"] is False
    fall = Rule.read_rule(rule_report(share=.7, high=.8, loop_low=.01))["seam"]
    assert fall["loops_rise"] is True and "weak_keep" not in fall and fall["bridged_frame_share"] == .45 and fall["frames_per_bridge"] == 7. and fall["filled_cuts"] == 2
    assert Rule.read_rule(rule_report(share=.7, high=.8, loop_low=0.))["seam"]["loops_rise"] is False


@pytest.mark.parametrize("change,message", [(dict(report_only=False), "report_only"), (dict(declared_read=False), "smoke"), (dict(checks=dict(canary=dict(reproduced=False))), "void"), (dict(checks=dict(canary=None)), "void")])
def test_the_rule_refuses_a_report_it_must_not_read(change, message):
    with pytest.raises(SystemExit, match=message):
        Rule.read_rule(dict(rule_report(), **change))


def test_the_rule_says_it_is_a_draft_and_runs_as_a_script(tmp_path, capsys):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(rule_report()))
    assert Rule.main(["--report", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "draft, not declared" and out["code_digest"] == "d"
