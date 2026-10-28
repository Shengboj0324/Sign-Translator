"""Sentence exposure diagnostic: its new logic (derangement, tails, prefix blanking, exact fast bootstrap, repetition and
keypoint statistics), the reuse of the word diagnostic's sampling path, and end-to-end runs of both scripts on a tiny
prior, a directory store and a stub back-translator / recogniser."""
import hashlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.exposure_diagnostic as ed
import scripts.sentence_exposure_diagnostic as sd
from signtranslator.data_engineering.corpus_manifest import KINDS, canonical_json, usable_channels
from signtranslator.data.text_motion_dataset import DirectoryStore
from signtranslator.generation.motion_rvq import (MotionRVQ, RVQPrior, RVQPriorConfig, generate_rvq_motion,
                                                  generate_rvq_motion_batch)
from signtranslator.generation.motion_to_text import corpus_bleu
from signtranslator.pose.multichannel import CHANNELS
from tests.test_generation_motion_rvq import small
from tests.test_generation_signing_space import signer_state

TEMPERATURE = {"body": 1.1, "left_hand": 1.1, "right_hand": 1.1, "face": 1.0}


# ---- pure logic --------------------------------------------------------------------------------------------------

def test_derangement_is_a_seeded_permutation_without_fixed_points():
    for n in (2, 3, 5, 17, 300):
        for seed in range(5):
            p = sd.derangement(n, seed)
            assert sorted(p) == list(range(n)) and all(p[i] != i for i in range(n))
    assert sd.derangement(300, 0) == sd.derangement(300, 0) != sd.derangement(300, 1)
    assert sd.derangement(2, 3) == [1, 0]
    with pytest.raises(ValueError):
        sd.derangement(1)


def test_prefix_boundary_and_tail_words():
    assert [ed.prefix_steps(f, 10) for f in (.25, .5, .75)] == [2, 5, 8]          # round half to even: 2.5 -> 2
    assert [ed.prefix_steps(f, 141) for f in (.25, .5, .75)] == [35, 70, 106]
    assert sd.tail_start(.75, 1) == 0 and sd.tail_start(.5, 141) == 70             # a tail keeps at least one frame
    text = "cut the onion in half"
    assert [sd.tail_words(text, f) for f in (.25, .5, .75)] == ["the onion in half", "onion in half", "half"]
    assert sd.tail_words("go", .75) == "go" and sd.tail_words("", .5) == ""         # at least the last word
    assert sd.tail_words("a b c d", .5) == "c d"


def test_frames_from_and_blank_from_cut_and_blank_exactly_the_tail():
    state = signer_state(0, frames=12)
    tail = sd.frames_from(state, 5)
    tail.validate()
    assert len(tail.timestamps) == 7 and tail.timestamps[0] == state.timestamps[5]
    np.testing.assert_array_equal(tail.channels["body"].values, state.channels["body"].values[5:])
    blank = sd.blank_from(state, 5)
    blank.validate()
    for name, c in blank.channels.items():
        ref = state.channels[name]
        np.testing.assert_array_equal(c.values[:5], ref.values[:5])
        np.testing.assert_array_equal(c.valid[:5], ref.valid[:5])
        assert not c.valid[5:].any() and not c.observed[5:].any()
        assert (c.values[5:] == 0).all() and (c.confidence[5:] == 0).all() and c.values.dtype == ref.values.dtype
    assert state.channels["body"].valid.all()                                      # the input is untouched
    assert sd.blank_from(state, 0).channels["body"].valid.sum() == 0


def _corpus(rng, n):
    vocab = ["i", "am", "going", "to", "show", "you", "how", "cut", "the", "onion", "now", "and"]
    sent = lambda: " ".join(rng.choice(vocab, rng.integers(0, 9)))
    return [sent() for _ in range(n)], [sent() or "the" for _ in range(n)]


def test_fast_bleu_equals_corpus_bleu_and_the_pilot_bootstrap_exactly():
    from scripts.strict_prior_review import bleu_ci
    rng = np.random.default_rng(0)
    for n in (1, 7, 40):
        hyps, refs = _corpus(rng, n)
        assert sd.bleu_from_stats(sd.bleu_stats(hyps, refs).sum(0)) == corpus_bleu(hyps, refs)["bleu"]
        assert sd.bleu_ci(hyps, refs, n=200) == bleu_ci(hyps, refs, n=200)
        assert sd.bleu_ci(hyps, refs, n=200, seed=3) == bleu_ci(hyps, refs, n=200, seed=3)
    same = ["i am going to show you"] * 10
    assert sd.bleu_ci(same, same, n=50) == bleu_ci(same, same, n=50) == (100., [100., 100.])
    assert sd.bleu_from_stats(sd.bleu_stats([""], ["the onion"]).sum(0)) == 0.


def test_paired_difference_is_zero_on_itself_and_its_marginal_is_the_pilot_interval():
    from scripts.strict_prior_review import bleu_ci
    rng = np.random.default_rng(1)
    hyps, refs = _corpus(rng, 60)
    zero = sd.paired_bleu_difference(hyps, refs, hyps, refs, n=100)
    assert zero == dict(difference=0., ci95=[0., 0.], p_not_above=1.)
    empty = [""] * len(hyps)                                     # BLEU 0 in every draw: the difference is a's own BLEU
    d = sd.paired_bleu_difference(hyps, refs, empty, refs, n=100)
    point, ci = bleu_ci(hyps, refs, n=100)
    assert d["difference"] == point and d["ci95"] == ci
    assert sd.paired_bleu_difference(empty, refs, hyps, refs, n=100)["difference"] == -point
    with pytest.raises(ValueError):
        sd.paired_bleu_difference(hyps, refs, hyps[:-1], refs[:-1])


def test_score_forms_reports_matched_mismatched_and_paired_controls():
    texts = ["cut the onion", "add the salt now", "show you how", "stir it well", "turn off the heat"]
    hyps = {"real": list(texts), "blank": ["the"] * 5, "gen": ["the onion"] * 5}
    perm = sd.derangement(5, 0)
    out = sd.score_forms(hyps, texts, perm, boot=50, pairs={"gen": {"real": "real"}})
    assert out["real"]["bleu"] == 100. and out["real"]["mismatched"]["bleu"] < 100.
    assert out["real"]["minus_mismatched"]["difference"] > 0 and out["real"]["above_blank"]
    assert "minus_real" in out["gen"] and "minus_blank" not in out["blank"] and out["gen"]["hypotheses"] == hyps["gen"]
    # the mismatched control is the same hypotheses against the permuted references
    assert out["gen"]["mismatched"]["bleu"] == sd.bleu_ci(hyps["gen"], [texts[j] for j in perm], n=50)[0]


def test_code_repetition_measures_stasis_and_loops():
    still = np.zeros((10, 6), int)
    r = sd.code_repetition(still, 2, lags=(1, 4, 12))
    assert r["lag_match"] == {"1": 1., "4": 1., "12": None} and r["distinct_rows"] == .1 and r["mean_run"] == 10.
    assert r["full_lag1"] == 1.
    loop = np.tile(np.array([[0, 1, 5, 5], [2, 3, 5, 5]]), (6, 1))               # period 2 in the stage-1 codes
    r = sd.code_repetition(loop, 2, lags=(1, 2))
    assert r["lag_match"] == {"1": 0., "2": 1.} and r["distinct_rows"] == round(2 / 12, 4) and r["mean_run"] == 1.
    fine_only = still.copy(); fine_only[::2, 3] = 7                                # only a finer stage changes
    r = sd.code_repetition(fine_only, 2, lags=(1,))
    assert r["lag_match"]["1"] == 1. and r["full_lag1"] == 0.
    with pytest.raises(ValueError):
        sd.code_repetition(np.zeros((0, 4), int), 2)


def test_keypoint_statistics_speed_tracking_spread_and_bands():
    fps, T = 24., 96
    points = np.zeros((T, 48, 2)); valid = np.ones((T, 48), bool)
    t = np.arange(T) / fps
    points[:, 4, 0] = .5 * t                                        # left wrist: 0.5 shoulder widths per second
    points[:, 5, 1] = .2 * np.sin(2 * np.pi * 2. * t)               # right wrist: a 2 Hz oscillation
    valid[T // 2:, 27:48] = False                                    # right hand untracked half the time
    s = sd.keypoint_statistics(points, valid, fps)
    assert s["hands_tracked"] == pytest.approx(.75)
    speeds = np.concatenate([np.full(T - 1, .5), np.abs(np.diff(points[:, 5, 1])) * fps])
    assert s["wrist_speed"] == pytest.approx(speeds.mean())
    assert s["handshape_spread"] == pytest.approx(0.)
    right_band = sd.keypoint_statistics(points[:, [0, 1, 2, 3, 5, 5] + list(range(6, 48))], valid, fps)
    assert right_band["wrist_band_share"][1] > .99                   # both wrists the 2 Hz one: all power in [1, 3) Hz
    none = sd.keypoint_statistics(points, np.zeros_like(valid), fps)
    assert none["wrist_speed"] is None and none["hands_tracked"] == 0. and none["wrist_band_share"] is None


def test_mean_statistics_skips_missing_values_and_unions_keys():
    rows = [dict(a=1., b=None, c=[1., 3.], d={"x": 2.}), dict(a=3., b=4., c=[3., 5.], d={"x": 4.}, e=1.), {}]
    m = sd.mean_statistics(rows)
    assert m["a"] == 2. and m["a_clips"] == 2 and m["b"] == 4. and m["b_clips"] == 1
    assert m["c"] == [2., 4.] and m["d"]["x"] == 3. and m["e"] == 1. and m["e_clips"] == 1
    assert sd.ratio_to(dict(a=2., a_clips=3, c=[1.]), dict(a=4.)) == {"a": .5}


# ---- reuse of the word diagnostic's paths ------------------------------------------------------------------------

def tiny_prior(seed=0, **prior_kw):
    torch.manual_seed(seed)
    state = signer_state(0, frames=16, amplitude=.4)
    tokenizer = MotionRVQ(small(state, down_layers=0)).eval()          # one token per frame, as the arms' tokenizers
    config = RVQPriorConfig(tokenizer.config, d_model=32, heads=4, text_layers=1, temporal_layers=1, depth_layers=1,
                            feedforward=64, dropout=0., **prior_kw)
    return tokenizer, RVQPrior(config).eval()


def test_continue_from_an_empty_prefix_is_generate_rvq_motion_with_the_same_seed():
    tokenizer, prior = tiny_prior(locus_conditioning=True, history_stages=1)
    decode = ed.make_decode(tokenizer, tokenizer.config.motion, lambda s: s)
    empty = torch.zeros((1, 0, prior.J), dtype=torch.long)
    for guidance, temperature in ((1., None), (1.5, TEMPERATURE), (2., 1.3)):
        ours, codes = ed.continue_from(prior, tokenizer, decode, "cut it", empty, 12, 5, guidance=guidance,
                                       device=torch.device("cpu"), temperature=temperature, return_codes=True)
        theirs = generate_rvq_motion(prior, tokenizer, ["cut it"], fps=24., durations=[.5], guidance=guidance,
                                     temperature=1. if temperature is None else temperature, seed=5)[0]
        assert codes.shape == (12, prior.J)
        for name in theirs.channels:
            np.testing.assert_array_equal(ours.channels[name].values, theirs.channels[name].values)
    # temperature None (the word diagnostic) is exactly temperature 1
    a = ed.continue_from(prior, tokenizer, decode, "go", empty, 6, 1, guidance=1.5, device=torch.device("cpu"))
    b = ed.continue_from(prior, tokenizer, decode, "go", empty, 6, 1, guidance=1.5, device=torch.device("cpu"),
                         temperature=1.)
    np.testing.assert_array_equal(a.channels["body"].values, b.channels["body"].values)
    with pytest.raises(ValueError, match="unknown parts"):
        ed.continue_from(prior, tokenizer, decode, "go", empty, 6, 1, guidance=1., device=torch.device("cpu"),
                         temperature={"tail": 2.})


def test_continue_from_keeps_the_real_prefix_and_samples_only_the_rest():
    tokenizer, prior = tiny_prior()
    decode = ed.make_decode(tokenizer, tokenizer.config.motion, lambda s: s)
    prefix = torch.randint(0, prior.K, (1, 5, prior.J))
    _, codes = ed.continue_from(prior, tokenizer, decode, "go", prefix, 12, 0, guidance=1.5,
                                device=torch.device("cpu"), return_codes=True)
    assert torch.equal(codes[:5], prefix[0]) and codes.shape == (12, prior.J)
    held = ed.held_codes(prefix, 3, 8)
    assert torch.equal(held[0, :3], prefix[0, :3]) and (held[0, 3:] == prefix[0, 2]).all()


def test_batched_free_run_returns_its_codes_without_changing_its_samples():
    tokenizer, prior = tiny_prior(locus_conditioning=True)
    texts, durations = ["ask him", "cut it now"], [.5, .3]
    plain = generate_rvq_motion_batch(prior, tokenizer, texts, fps=24., durations=durations, guidance=1.5,
                                      temperature=TEMPERATURE, seed=1000)
    states, codes = generate_rvq_motion_batch(prior, tokenizer, texts, fps=24., durations=durations, guidance=1.5,
                                              temperature=TEMPERATURE, seed=1000, return_codes=True)
    decode = ed.make_decode(tokenizer, tokenizer.config.motion, lambda s: s)
    assert [tuple(c.shape) for c in codes] == [(12, prior.J), (7, prior.J)]
    for a, b, c in zip(plain, states, codes):
        again = decode(c[None], len(b.timestamps), "x")
        for name in a.channels:
            np.testing.assert_array_equal(a.channels[name].values, b.channels[name].values)
            np.testing.assert_allclose(again.channels[name].values, b.channels[name].values, atol=1e-6)


def test_teacher_forced_one_step_reads_the_real_history_of_a_learnt_clip():
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.motion_rvq import rvq_prior_loss
    from signtranslator.generation.text_to_motion import encode_texts
    tokenizer, prior = tiny_prior()
    state = signer_state(3, frames=16, amplitude=.4)
    tokens, mask = encode_texts(["cut it"])
    optimizer = torch.optim.Adam(prior.parameters(), lr=3e-3)
    prior.train()
    for _ in range(150):
        loss, _ = rvq_prior_loss(prior, tokenizer, tokens, mask, collate_multichannel([state]), train=False)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    prior.eval()
    motion, codes, drawn, modes = ed.teacher_forced(prior, tokenizer, tokenizer.config.motion, state, "cut it",
                                                    torch.Generator().manual_seed(0), torch.device("cpu"))
    assert codes.shape == (1, 16, prior.J) and drawn.shape == modes.shape == (16, prior.J)
    assert float((modes == codes[0]).float().mean()) > .8                 # the modes reproduce the learnt codes


# ---- end to end ----------------------------------------------------------------------------------------------------

class StubTranslator:
    """Reads a state's motion as words: the wrists' mean height and travel, coarsely binned (a deterministic, motion-
    dependent stand-in for the back-translator), "blank" without valid hands."""
    config = type("C", (), {"fps": 24.})()

    def to(self, device):
        return self


def stub_translate(model, kind, states):
    from signtranslator.data.keypoints2d import from_state, normalise
    out = []
    for s in states:
        points, valid = from_state(s)
        p = normalise(points, valid, y_down=False)
        ok = valid[:, 4]
        if not ok.any():
            out.append("blank")
            continue
        out.append(" ".join(["cut", "the", "onion", "now", "salt", "it"][int(abs(v) * 7) % 6]
                            for v in p[ok, 4].mean(0).tolist() + p[ok, 5].std(0).tolist()))
    return out


def build_store(root: Path, *, clips=5, corpus="how2sign-two-view-text-v1", train_texts=("ask", "go")):
    store = DirectoryStore(root)
    texts = ["cut the onion in half", "add some salt now", "show you how to stir it", "turn the heat off",
             "this is the knife", "now the next step"]
    records = []

    def record(k, text, partition, corpus_id, frames):
        state = signer_state(10 + k, frames=frames, amplitude=.4)
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "s.npz"
            sha = state.save(path)
            payload = path.read_bytes()
        key = f"states/{k}.npz"
        store.put(key, payload)
        r = dict(sign=f"clip{k}", corpus=corpus_id, partition=partition, signer_component="c01", text=text,
                 continuous=dict(status="converted", state_key=key, state_sha256=sha, available_channels=list(CHANNELS),
                                 seconds=frames / 24.))
        r["usable"] = {kind: list(usable_channels(r, kind)) for kind in KINDS}
        return r
    records.append(record(0, "the start", "train", "how2sign-two-view-text-v1", 12))      # the layout template
    for k, text in enumerate(train_texts):
        records.append(record(1 + k, text, "train", corpus, 14))
    for k in range(clips):
        records.append(record(20 + k, texts[k % len(texts)], "val", "how2sign-two-view-text-v1", 18 + 2 * k))
    manifest = dict(records=records)
    payload = canonical_json(manifest)
    store.put("mix.json", payload)
    sha = hashlib.sha256(payload).hexdigest()
    tokenizer, prior = tiny_prior(locus_conditioning=True, history_stages=1)
    buffer = io.BytesIO()
    torch.save(dict(kind="motion_rvq_prior", config=prior.config.to_dict(), model=prior.state_dict(), ema=None,
                    tokenizer_state=tokenizer.state_dict(), manifest_sha256=sha, config_sha256=prior.config.sha256()),
               buffer)
    store.put("prior.pt", buffer.getvalue())
    store.put(f"training-runs/{sd.TRANSLATOR_RUN}/best.json", json.dumps(dict(checkpoint_key="translator.pt")).encode())
    return store, sha, texts


def test_free_run_is_the_pilot_generated_form(tmp_path):
    """The free-run form is the pilot's: load_generator's batch sampler, the pilot's post-processing."""
    from scripts.evaluate_text_to_motion import load_generator
    from scripts.pilot_content_check import postprocess
    store, sha, texts = build_store(tmp_path)
    _, _, sample = load_generator(store, "prior.pt", sha, "cpu", temperature=TEMPERATURE)
    pilot = [postprocess(o) for o in sample.batch(texts[:3], 24., [.5, .75, .6], 1.5, 1000)]
    ck = torch.load(io.BytesIO(store.get("prior.pt")), map_location="cpu", weights_only=False)
    prior = RVQPrior(RVQPriorConfig.from_dict(ck["config"])); prior.load_state_dict(ck["model"]); prior.eval()
    tokenizer = MotionRVQ(prior.config.tokenizer); tokenizer.load_state_dict(ck["tokenizer_state"]); tokenizer.eval()
    states, _ = generate_rvq_motion_batch(prior, tokenizer, texts[:3], fps=24., durations=[.5, .75, .6], guidance=1.5,
                                          temperature=TEMPERATURE, top_k=0, seed=1000, return_codes=True)
    post = ed.make_post(None)
    for a, b in zip(pilot, [post(s) for s in states]):
        for name in a.channels:
            np.testing.assert_array_equal(a.channels[name].values, b.channels[name].values)


@pytest.fixture
def stubbed(monkeypatch):
    import scripts.score_back_translation as sbt
    import score_back_translation
    for module in (sbt, score_back_translation):
        monkeypatch.setattr(module, "load_translator", lambda store, key, sha: (StubTranslator(), "keypoints"))
        monkeypatch.setattr(module, "translate", stub_translate)


def test_sentence_diagnostic_end_to_end_on_a_tiny_prior(tmp_path, stubbed, capsys):
    store, sha, texts = build_store(tmp_path / "store")
    out = tmp_path / "report.json"
    argv = ["--checkpoint-key", "prior.pt", "--mix-key", "mix.json", "--mix-sha256", sha, "--clips", "4",
            "--device", "cpu", "--store", f"dir:{tmp_path / 'store'}", "--cache-dir", str(tmp_path / "cache"),
            "--bootstrap", "40", "--output-key", "reports/sentence.json", "--output-file", str(out)]
    assert sd.main(argv) == 0
    report = json.loads(out.read_bytes())
    assert report == json.loads(store.get("reports/sentence.json"))
    expected = ["real", "blank", "real_dataset", "recon_full", "recon_masked", "onestep_sample", "onestep_argmax",
                "free_run"] + [f"{a}{t}{b}" for t in ("25", "50", "75") for a, b in (("prefix", ""), ("prefix", "_blank"),
                                                                                       ("hold", ""))]
    assert list(report["whole"]) == expected and report["clips"] == 4
    for row in report["whole"].values():
        assert len(row["hypotheses"]) == 4 and row["ci95"][0] <= row["bleu"] <= row["ci95"][1]
        assert {"mismatched", "minus_mismatched"} <= set(row)
    assert set(report["whole"]["prefix50"]) >= {"minus_prefix_blank", "minus_hold", "minus_blank"}
    assert all(report["whole"]["blank"]["hypotheses"][i] == "blank" for i in range(4))
    assert report["whole"]["real"]["hypotheses"] == report["whole"]["real_dataset"]["hypotheses"]
    perm = report["derangement"]
    assert sorted(perm) == list(range(4)) and all(perm[i] != i for i in range(4))
    assert set(report["tails"]) == {"25", "50", "75"}
    assert report["tails"]["75"]["references"] == [sd.tail_words(t, .75) for t in report["references"]]
    assert list(report["tails"]["50"]["forms"]) == list(sd.TAIL_FORMS)
    assert report["frame_mismatches"] == dict(reference_vs_dataset=0, reference_vs_free_run=0)
    assert report["history_stages"] == 1 and report["locus_conditioning"] is True
    kp = report["motion"]["keypoint"]
    assert kp["real"]["hands_tracked"] == 1. and kp["blank"]["hands_tracked"] == 0.
    assert kp["prefix50_blank"]["hands_tracked"] < kp["prefix50"]["hands_tracked"] == 1.
    assert report["motion"]["keypoint_ratio_to_real"]["real"]["wrist_speed"] == 1.
    assert "real" not in report["motion"]["rotation_vs_real"] and "free_run" in report["motion"]["rotation_vs_real"]
    reps = report["code_repetition"]
    assert {"real", "free_run", "onestep_sample", "onestep_argmax", "prefix25_tail", "real_tail25"} <= set(reps)
    assert reps["real"]["steps"] == np.mean([r["frames"] for r in report["per_clip"]])
    printed = capsys.readouterr().out
    assert "whole clips (4):" in printed and "tails from 50%:" in printed
    # write-once: neither output is overwritten
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        sd.main(argv)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        sd.main(argv[:-2] + ["--output-file", str(tmp_path / "other.json")])
    assert not (tmp_path / "other.json").exists()


def test_sentence_diagnostic_refuses_a_prior_of_another_manifest(tmp_path, stubbed):
    store, sha, _ = build_store(tmp_path / "store")
    ck = torch.load(io.BytesIO(store.get("prior.pt")), map_location="cpu", weights_only=False)
    ck["manifest_sha256"] = "0" * 64
    buffer = io.BytesIO(); torch.save(ck, buffer); store.put("other.pt", buffer.getvalue())
    with pytest.raises(SystemExit, match="not an RVQ prior trained on"):
        sd.main(["--checkpoint-key", "other.pt", "--mix-key", "mix.json", "--mix-sha256", sha, "--clips", "2",
                 "--device", "cpu", "--store", f"dir:{tmp_path / 'store'}", "--cache-dir", str(tmp_path / "cache"),
                 "--output-file", str(tmp_path / "r.json")])


def test_word_exposure_diagnostic_end_to_end_with_its_extra_forms(tmp_path, monkeypatch):
    """The word diagnostic still runs on the lifted code paths (stub recogniser: logits from the states' motion)."""
    import signtranslator.data.text_motion_dataset as tmd
    import validate_sign_recognizer as vsr
    store, sha, _ = build_store(tmp_path / "store", corpus="wlasl-text-r6", train_texts=("ask", "go"))
    monkeypatch.setattr(tmd, "R2Store", lambda env: store)
    monkeypatch.setattr(ed.tempfile, "gettempdir", lambda: str(tmp_path))
    seen = []

    def logits_of(model, states, device):
        seen.append(len(states))
        return torch.tensor([[float(s.channels["right_hand"].values[..., k % 6].mean()) for k in range(8)] for s in states])
    labels = ["ASK", "GO", "A", "B", "C", "D", "E", "F"]
    monkeypatch.setattr(vsr, "load_model", lambda store, run, device: (None, labels, None))
    monkeypatch.setattr(vsr, "logits_of", logits_of)
    assert ed.main(["--checkpoint-key", "prior.pt", "--mix-key", "mix.json", "--mix-sha256", sha, "--corpus",
                    "wlasl-text-r6", "--device", "cpu", "--extra", "--output-key", "reports/words.json"]) == 0
    report = json.loads(store.get("reports/words.json"))
    assert report["words"] == 2 and set(report["top5"]) == {
        "real", "recon_full", "recon_masked", "onestep_sample", "onestep_argmax", "free_run", "prefix25", "hold25",
        "prefix50", "hold50", "prefix75", "hold75", "coarse_sampled"}
    assert seen == [2] * 13 and len(report["per_word"]) == 2
