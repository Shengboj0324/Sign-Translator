"""S0 ceilings and S1 text KL scripts on synthetic data: stage-truncated round trips, face/warp/low-pass forms, the
teacher-forced reference NLL (planted match, ΔNLL sign), orderings, the stored-only end-to-end read (write-once,
mismatched clips skipped), regeneration seeds, and the stage-1 KL (zero when the text has no effect)."""
from dataclasses import replace
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import s0_ceilings as s0  # noqa: E402
import s1_text_kl as s1  # noqa: E402
from signtranslator.data.multichannel import collate_multichannel  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402
from signtranslator.eval import s0_metrics as M  # noqa: E402
from signtranslator.generation.motion_rvq import MotionRVQ, RVQPrior, RVQPriorConfig, rvq_sequence_log_likelihood  # noqa: E402
from signtranslator.generation.text_to_motion import PAD, encode_texts  # noqa: E402
from tests.test_generation_motion_rvq import small  # noqa: E402
from tests.test_generation_signing_space import signer_state  # noqa: E402


def with_face(state, seed=0):
    """``state`` with a valid, moving face channel and gaze."""
    rng = np.random.default_rng(seed)
    T = len(state.timestamps)
    channels = dict(state.channels)
    f = channels["face"]
    values = rng.uniform(0, 1, (T, 1, 1))
    channels["face"] = replace(f, values=values, valid=np.ones((T, 1), bool), observed=np.ones((T, 1), bool),
                               confidence=np.ones((T, 1)))
    g = channels["left_gaze"]
    v = rng.normal(size=(T, 1, 3)); v /= np.linalg.norm(v, axis=-1, keepdims=True)
    channels["left_gaze"] = replace(g, values=v, valid=np.ones((T, 1), bool), observed=np.ones((T, 1), bool),
                                    confidence=np.ones((T, 1)))
    return replace(state, channels=channels)


# ---- forms ---------------------------------------------------------------------------------------------------------

def test_stage_truncated_round_trip_with_all_stages_is_the_fidelity_round_trip():
    from tokenizer_recognisability_check import round_trip
    torch.manual_seed(0)
    state = signer_state(0, frames=16, amplitude=.4)
    model = MotionRVQ(small(state, down_layers=0)).eval()
    full = round_trip(model, model.config.motion, state, 24.)
    same = s0.truncated_round_trip(model, state, model.config.stages)
    none = s0.truncated_round_trip(model, state, None)
    one = s0.truncated_round_trip(model, state, 1)
    for name in ("body", "left_hand", "right_hand"):
        np.testing.assert_allclose(same.channels[name].values, full.channels[name].values, atol=1e-6)
        np.testing.assert_allclose(none.channels[name].values, full.channels[name].values, atol=1e-6)
    assert not np.allclose(one.channels["body"].values, full.channels["body"].values)


def test_face_forms_hold_or_remove_the_face_and_v1_cannot_see_either():
    from signtranslator.data.keypoints2d import features, from_state, normalise
    state = with_face(signer_state(1, frames=10, amplitude=.4))
    mean = s0.face_mean(state)
    face = mean.channels["face"]
    assert np.allclose(face.values, state.channels["face"].values.mean(0)) and face.valid.all()
    gaze = mean.channels["left_gaze"].values
    assert np.allclose(np.linalg.norm(gaze, axis=-1), 1.) and np.allclose(gaze, gaze[0])     # back on the sphere
    gone = s0.face_absent(state)
    assert not gone.channels["face"].valid.any() and (gone.channels["face"].values == 0).all()
    assert gone.channels["body"] is state.channels["body"]
    kp = lambda s: features(normalise(*from_state(s), y_down=False), from_state(s)[1])
    assert np.array_equal(kp(state), kp(mean)) and np.array_equal(kp(state), kp(gone))       # v1 is face-blind


def test_time_warp_changes_duration_by_the_inverse_speed():
    state = signer_state(2, frames=49, amplitude=.4)                     # 2 s at 24 fps
    slow, fast = s0.time_warp(state, .8), s0.time_warp(state, 1.25)
    assert slow.timestamps[-1] == pytest.approx(2.5) and fast.timestamps[-1] == pytest.approx(1.6, abs=1 / 24)
    assert np.allclose(np.diff(slow.timestamps), 1 / 24)
    np.testing.assert_allclose(s0.time_warp(state, 1.).channels["body"].values, state.channels["body"].values, atol=1e-9)


def test_low_pass_keeps_slow_motion_damps_fast_motion_and_keeps_validity():
    T = 96
    state = signer_state(3, frames=T, amplitude=0.)
    t = np.arange(T) / 24.
    channels = dict(state.channels)
    rt = channels["root_translation"]
    slow, fast = np.sin(2 * np.pi * 1. * t), np.sin(2 * np.pi * 10. * t)
    values = np.stack([slow, fast, np.zeros(T)], -1)[:, None]
    valid = np.ones((T, 1), bool); valid[40:42] = False                 # a gap: two runs filtered separately
    channels["root_translation"] = replace(rt, values=np.where(valid[..., None], values, 0.), valid=valid,
                                           observed=valid.copy(), confidence=valid.astype(float))
    out = s0.low_pass(replace(state, channels=channels), 6.)
    v = out.channels["root_translation"].values[:, 0]
    mid = slice(10, 30)
    assert np.abs(v[mid, 0] - slow[mid]).max() < .05                    # 1 Hz passes
    assert np.abs(v[mid, 1]).max() < .05                                 # 10 Hz is removed
    assert (out.channels["root_translation"].valid == valid).all() and (v[40:42] == 0).all()
    body = out.channels["body"].values
    assert np.allclose(np.linalg.norm(body[..., :3], axis=-1), 1.)      # rotations stay orthonormal


# ---- translator likelihoods ------------------------------------------------------------------------------------------

def tiny_translator():
    transformers = pytest.importorskip("transformers")
    from signtranslator.generation.keypoints_to_text_t5 import KeypointsToT5, KeypointsToT5Config
    from tests.test_generation_motion_to_text_t5 import CharTokenizer
    t5 = transformers.T5ForConditionalGeneration(transformers.T5Config(
        vocab_size=260, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=4,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    config = KeypointsToT5Config(t5_name="tiny", d_model=32, heads=4, encoder_layers=1, feedforward=64, dropout=0.,
                                 max_tokens=16)
    return KeypointsToT5(config, t5, CharTokenizer())


def test_candidate_nll_column_zero_is_the_teacher_forced_loss():
    torch.manual_seed(0)
    model = tiny_translator().eval()
    items = [torch.randn(n, 144) for n in (12, 20, 16)]
    refs = ["dog", "a cat", "sun"]
    cands = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    nll = s0.candidate_nll(model, items, refs, cands, device="cpu", group=2)
    for i in range(3):
        from signtranslator.data.keypoint_shards import collate_keypoints
        (x, valid), _ = collate_keypoints([(items[i], "")])
        with torch.no_grad():
            loss, _ = model.loss((x, valid), [refs[i]])
        assert nll[i, 0] == pytest.approx(float(loss), rel=1e-4)
    assert nll.shape == (3, 3)
    alone = s0.candidate_nll(model, items[:1], refs, cands[:1], device="cpu", group=1)     # batching changes nothing
    np.testing.assert_allclose(alone[0], nll[0], rtol=1e-4)


def test_a_trained_translator_retrieves_its_planted_references_and_gains_over_blank():
    torch.manual_seed(0)
    model = tiny_translator()
    texts = ["dog", "cat", "sun", "map"]
    items = [torch.randn(1, 144).repeat(n, 1) + torch.linspace(0, 1, n)[:, None] for n in (20, 24, 28, 22)]
    from signtranslator.data.keypoint_shards import collate_keypoints
    batch, _ = collate_keypoints([(f, "") for f in items])
    blank = [torch.zeros_like(f) for f in items]
    bbatch, _ = collate_keypoints([(f, "") for f in blank])
    groups = model.param_groups(motion_lr=3e-3, decoder_lr=3e-3, weight_decay=0.)
    opt = torch.optim.Adam([{k: v for k, v in g.items() if k != "base_lr"} for g in groups])
    for _ in range(300):
        loss, _ = model.loss(batch, texts)
        loss = loss + .25 * model.loss(bbatch, ["the"] * 4)[0]            # blank reads as a template sentence
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    cands = M.retrieval_candidates(4, 4, seed=0)
    nll = s0.candidate_nll(model, items, texts, cands, device="cpu")
    nll_blank = s0.candidate_nll(model, blank, texts, cands, device="cpu")
    assert (M.mid_ranks(nll) == 1).all()                                   # the planted match ranks first
    dnll = M.delta_nll(nll_blank[:, 0], nll[:, 0])
    assert (dnll > 0).all()                                                # real motion beats blank
    assert (M.delta_nll(nll_blank[:, 0], nll_blank[:, 0]) == 0).all()      # blank's own gain is zero
    lik = s0.likelihood_panel(nll, nll_blank, None)
    assert lik["retrieval"]["nll"]["top1"] == 1. and lik["delta_nll"]["mean"] > 0


# ---- orderings -----------------------------------------------------------------------------------------------------

def test_orderings_report_reproduced_and_tighter_metrics():
    rng = np.random.default_rng(0)
    n = 200
    refs = [" ".join(rng.choice(list("abcdefgh"), 6)) for _ in range(n)]
    def form(quality):
        hyps = [r if rng.uniform() < quality else " ".join(rng.choice(list("abcdefgh"), 6)) for r in refs]
        dn = quality * 2 + rng.normal(0, .1, n)
        rank = np.where(rng.uniform(size=n) < quality, 1., rng.integers(1, 101, n).astype(float))
        return dict(stats=M.bleu_statistics(hyps, refs), delta_nll=dn, rank=rank, rank_pmi=rank)
    table = dict(real=form(.9), blank=form(0.), mid=form(.5))
    out = s0.orderings(table)
    assert out["summary"]["delta_nll"]["bleu_resolved"] == 3
    assert out["summary"]["delta_nll"]["reproduced"] == 3 and out["summary"]["delta_nll"]["contradicted"] == 0
    assert out["summary"]["delta_nll"]["passes"]
    assert out["summary"]["delta_nll"]["kendall_tau_vs_bleu"] == 1.


# ---- stored reads end to end, write-once, regeneration seeds ---------------------------------------------------------

def stored_store(tmp_path):
    store = DirectoryStore(tmp_path / "store")
    refs = ["you like it a lot", "we go home now today", "the dog sat on the mat", "i see you over there"]
    compose = dict(kind="round15-compose-eval", references=refs, forms=dict(
        real=dict(hypotheses=list(refs)),
        blank=dict(hypotheses=["you", "you", "you", "you"]),
        core_crossfade=dict(hypotheses=["you like", "we go", "the cat sat", "i you"])))
    pilot = dict(meaning=dict(references=refs, generated=dict(hypotheses=["you like it a", "go home", "dog", "see"]),
                              real=dict(hypotheses=compose["forms"]["real"]["hypotheses"]),
                              blank=dict(hypotheses=["you"] * 4)), checkpoint_key="c.pt")
    other = dict(meaning=dict(references=["something else"] * 3, generated=dict(hypotheses=["x"] * 3)))
    store.put("a/compose.json", json.dumps(compose).encode())
    store.put("a/pilot.json", json.dumps(pilot).encode())
    store.put("a/other.json", json.dumps(other).encode())
    return store, refs


def test_stored_only_read_scores_hypotheses_and_is_write_once(tmp_path):
    store, refs = stored_store(tmp_path)
    argv = ["--store", f"dir:{tmp_path / 'store'}", "--forms", "none", "--stored", "a/compose.json",
            "--stored", "a/pilot.json", "--stored", "a/other.json", "--output-key", "out/s0.json"]
    assert s0.main(argv) == 0
    out = json.loads(store.get("out/s0.json"))
    assert out["settings"]["anchor"] == "a/compose.json"
    from signtranslator.generation.motion_to_text import corpus_bleu
    gen = out["stored"]["a/pilot.json"]["forms"]["generated"]
    assert len(out["stored"]["a/pilot.json"]["sha256"]) == 64
    hyps = ["you like it a", "go home", "dog", "see"]
    assert gen["bleu"]["bleu"] == round(corpus_bleu(hyps, refs)["bleu"], 3) > 0
    assert gen["chrf"]["chrf"] == round(M.corpus_chrf(hyps, refs), 3)
    assert out["stored"]["a/compose.json"]["forms"]["real"]["gap_fraction"]["bleu"] == 1.
    assert out["stored"]["a/compose.json"]["forms"]["blank"]["gap_fraction"]["bleu"] == 0.
    assert "vs_real" in gen["paired"] and gen["identical_to_real"] == 0.
    assert "skipped" in out["stored"]["a/other.json"]
    with pytest.raises(SystemExit):                                         # write-once
        s0.main(argv)
    with pytest.raises(SystemExit):                                         # real and blank are required
        s0.main(["--store", f"dir:{tmp_path / 'store'}", "--forms", "round_trip", "--output-file", str(tmp_path / "x")])


def test_regeneration_uses_the_pilots_blocks_seeds_and_settings(tmp_path, monkeypatch):
    import evaluate_text_to_motion as etm
    store = DirectoryStore(tmp_path / "store")
    store.put("training-runs/r/run.json", json.dumps(dict(manifest_sha256="m" * 64)).encode())
    store.put("p.json", json.dumps(dict(checkpoint_key="training-runs/r/checkpoints/step-1.pt", guidance=1.5,
                                        part_temperature=dict(body=1.1), meaning=dict(
                                            references=["t"] * 40, generated=dict(hypotheses=["h"] * 40)))).encode())
    calls = {}

    def loader(store_, key, sha, device, temperature=None, plan=None):
        calls["load"] = (key, sha, device, temperature)

        def sample(*a, **k):
            raise AssertionError("single samples are not used")

        def batch(texts, fps, seconds, guidance, seed):
            calls.setdefault("batches", []).append((len(texts), fps, guidance, seed))
            return [signer_state(0, frames=4) for _ in texts]
        sample.batch = batch
        return "x", None, sample
    monkeypatch.setattr(etm, "load_generator", loader)
    gens, report = s0.regenerate(store, "p.json", ["t"] * 40, [1.] * 40, "cpu")
    assert len(gens) == 40 and calls["load"] == ("training-runs/r/checkpoints/step-1.pt", "m" * 64, "cpu", dict(body=1.1))
    assert calls["batches"] == [(32, 24., 1.5, 1000), (8, 24., 1.5, 1032)]
    store.put("plan.json", json.dumps(dict(plan=dict(a=1), meaning=dict(generated={}))).encode())
    with pytest.raises(SystemExit):
        s0.regenerate(store, "plan.json", ["t"], [1.], "cpu")


# ---- S1 ------------------------------------------------------------------------------------------------------------

def tiny_prior(seed=0):
    torch.manual_seed(0)
    state = signer_state(0, frames=16, amplitude=.4)
    tokenizer = MotionRVQ(small(state, down_layers=0)).eval()
    torch.manual_seed(seed)
    config = RVQPriorConfig(tokenizer.config, d_model=32, heads=4, text_layers=1, temporal_layers=1, depth_layers=1,
                            feedforward=64, dropout=0., locus_conditioning=True)
    return tokenizer, RVQPrior(config).eval()


def test_stage_logits_sum_to_the_sequence_log_likelihood():
    tokenizer, prior = tiny_prior()
    states = [signer_state(s, frames=n, amplitude=.4) for s, n in ((1, 16), (2, 11))]
    tokens, _ = encode_texts(["dog", "a cat runs"])
    mask = tokens != PAD
    motion = collate_multichannel(states)
    logits, target, row, weights = s1.stage_logits(prior, tokenizer, tokens, mask, motion)
    lp = logits.log_softmax(-1).gather(-1, target[..., None])[..., 0].sum(1)
    per_row = torch.zeros(2, dtype=lp.dtype).index_add_(0, row, lp)
    torch.testing.assert_close(per_row, rvq_sequence_log_likelihood(prior, tokenizer, tokens, mask, motion), rtol=1e-5,
                               atol=1e-4)
    assert weights.shape == target.shape and logits.shape[1] == prior.J


def test_kl_is_zero_when_the_text_has_no_effect_and_positive_otherwise():
    tokenizer, prior = tiny_prior(seed=3)
    states = [signer_state(s, frames=12, amplitude=.4) for s in range(3)]
    texts = ["dog", "a cat runs", "sun"]
    live = s1.clip_reads(prior, tokenizer, texts, states, loci_mode="plan", device=torch.device("cpu"))
    assert live["kl"].shape[1] == len(tokenizer.parts) and (live["kl"] > 0).all()
    deaf = prior
    with torch.no_grad():                               # cut the temporal decoder's cross-attention to the text
        for layer in deaf.temporal.layers:
            layer.multihead_attn.out_proj.weight.zero_()
            layer.multihead_attn.out_proj.bias.zero_()
    dead = s1.clip_reads(deaf, tokenizer, texts, states, loci_mode="plan", device=torch.device("cpu"))
    assert np.abs(dead["kl"]).max() < 1e-10 and np.abs(dead["gain"]).max() < 1e-10
    seconds = [float(s.timestamps[-1] - s.timestamps[0]) + 1 / 24 for s in states]
    summary = s1.summarise(dead, [p for p, _ in tokenizer.parts], seconds)
    assert summary["kl"] == 0. and summary["per_clip_summary"]["summed_kl_nats"]["mean"] == 0.
    live_summary = s1.summarise(live, [p for p, _ in tokenizer.parts], seconds)
    assert live_summary["kl"] > 0 and set(live_summary["per_part"]) == {p for p, _ in tokenizer.parts}
    assert len(live_summary["per_clip"]["mean_kl"]) == 3


def test_checkpoint_specs_parse():
    assert s1.parse_checkpoint("w=a/b.pt@abc") == ("w", "a/b.pt", "abc")
    assert s1.parse_checkpoint("w=a/b.pt") == ("w", "a/b.pt", None)
    with pytest.raises(SystemExit):
        s1.parse_checkpoint("a/b.pt")
