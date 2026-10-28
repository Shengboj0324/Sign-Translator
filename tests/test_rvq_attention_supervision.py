"""Round 13 candidates: guided monotonic attention and word-level alignment supervision of the RVQ prior's
temporal cross-attention to the text."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from signtranslator.data.multichannel import collate_multichannel
from signtranslator.data.text_motion_dataset import collate_text_motion
from signtranslator.generation.motion_rvq import (
    AttentionRecordingDecoderLayer, RVQPrior, RVQPriorConfig, alignment_loss, alignment_targets,
    cross_attention_probabilities, frame_word_spans, guided_attention_loss, guided_attention_weights, rvq_prior_loss,
    word_byte_spans,
)
from signtranslator.generation.text_to_motion import encode_texts
from tests.test_generation_motion_rvq import prior_setup


def test_attention_options_are_opt_in_and_keep_earlier_config_shas():
    _, _, prior = prior_setup()
    data = prior.config.to_dict()
    for key in ("guided_attention", "guided_attention_width", "alignment_supervision"):
        assert key not in data
    # a width alone (weight 0) changes nothing either
    assert replace(prior.config, guided_attention_width=.3).sha256() == prior.config.sha256()
    guided = replace(prior.config, guided_attention=.5, guided_attention_width=.3)
    assert guided.to_dict()["guided_attention"] == .5 and guided.to_dict()["guided_attention_width"] == .3
    assert RVQPriorConfig.from_dict(guided.to_dict()) == guided and guided.sha256() != prior.config.sha256()
    aligned = replace(prior.config, alignment_supervision=1.)
    assert "guided_attention_width" not in aligned.to_dict()
    assert RVQPriorConfig.from_dict(aligned.to_dict()) == aligned and aligned.sha256() != prior.config.sha256()


def _decoders(dropout):
    torch.manual_seed(0)
    plain = nn.TransformerDecoder(nn.TransformerDecoderLayer(32, 4, 64, dropout, batch_first=True, norm_first=True), 2)
    recording = nn.TransformerDecoder(AttentionRecordingDecoderLayer(32, 4, 64, dropout, batch_first=True,
                                                                     norm_first=True), 2)
    recording.load_state_dict(plain.state_dict())
    return plain, recording


@pytest.mark.parametrize("train", [False, True])
def test_recording_layer_has_the_plain_layers_state_dict_and_bit_identical_outputs(train):
    plain, recording = _decoders(dropout=.1)
    assert list(plain.state_dict()) == list(recording.state_dict())
    plain.train(train); recording.train(train)
    x, memory = torch.randn(3, 7, 32), torch.randn(3, 5, 32)
    pad = torch.tensor([[False] * 5, [False] * 3 + [True] * 2, [False] * 4 + [True]])
    causal = torch.triu(torch.ones(7, 7, dtype=torch.bool), 1)
    outputs = []
    for model, record in ((plain, False), (recording, False), (recording, True)):
        for layer in model.layers:
            layer.record = record
        torch.manual_seed(5)
        outputs.append(model(x, memory, tgt_mask=causal, memory_key_padding_mask=pad))
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[0], outputs[2])
    weights = recording.layers[0].cross_attention.detach()
    assert weights.shape == (3, 7, 5)
    torch.testing.assert_close(weights.sum(-1), torch.ones(3, 7))
    assert float(weights[1, :, 3:].abs().max()) == 0.                           # padded bytes get no weight


def test_side_computed_weights_equal_the_attention_modules_own_weights():
    torch.manual_seed(0)
    attention = nn.MultiheadAttention(32, 4, dropout=.1, batch_first=True).eval()
    query, memory = torch.randn(2, 6, 32), torch.randn(2, 9, 32)
    pad = torch.zeros(2, 9, dtype=torch.bool); pad[1, 6:] = True
    _, reference = attention(query, memory, memory, key_padding_mask=pad, need_weights=True,
                             average_attn_weights=True)
    torch.testing.assert_close(cross_attention_probabilities(attention, query, memory, pad), reference)


def test_prior_checkpoints_load_unchanged_and_temporal_states_are_the_same_with_attention():
    _, _, prior = prior_setup()
    plain = nn.TransformerDecoder(nn.TransformerDecoderLayer(32, 4, 64, 0., batch_first=True, norm_first=True), 1)
    assert list(plain.state_dict()) == list(prior.temporal.state_dict())
    prior.temporal.load_state_dict(plain.state_dict())                          # strict: an old checkpoint's keys
    tokens, mask = encode_texts(["ask him", "cut it now"])
    hidden, _, mask = prior.encode(tokens, mask)
    previous = torch.randint(0, prior.K, (2, 4, prior.J))
    seconds, duration = torch.arange(4.)[None].expand(2, -1) * .2, torch.tensor([.8, .8])
    h = prior.temporal_states(hidden, mask, previous, seconds, duration)
    h2, attention = prior.temporal_states(hidden, mask, previous, seconds, duration, return_attention=True)
    assert torch.equal(h, h2) and attention.shape == (2, 1, 4, tokens.shape[1])
    assert all(not layer.record and layer.cross_attention is None for layer in prior.temporal.layers)


def test_guided_attention_is_small_on_the_diagonal_and_large_on_the_anti_diagonal():
    S, N = 20, 12
    step_valid, text_valid = torch.ones(1, S, dtype=torch.bool), torch.ones(1, N, dtype=torch.bool)
    t, n = torch.arange(S), torch.arange(N)
    diagonal = torch.zeros(1, S, N); diagonal[0, t, (t * N) // S] = 1
    anti = torch.zeros(1, S, N); anti[0, t, N - 1 - (t * N) // S] = 1
    low = float(guided_attention_loss(diagonal, step_valid, text_valid))
    high = float(guided_attention_loss(anti, step_valid, text_valid))
    assert low < .05 and high > .6 and high > 10 * low                          # expected penalty per step, in [0, 1]
    uniform = torch.full((1, S, N), 1. / N)
    assert low < float(guided_attention_loss(uniform, step_valid, text_valid)) < high
    W = guided_attention_weights(step_valid, text_valid)
    assert float(W[0, 0, 0]) == 0. and float(W[0, 0, -1]) > .99                  # BOS free at the start, EOS far
    # padding (steps and bytes) changes nothing: ranks and counts are per row
    padded = torch.zeros(1, S + 3, N + 4); padded[0, :S, :N] = anti[0]
    sv = torch.zeros(1, S + 3, dtype=torch.bool); sv[0, :S] = True
    tv = torch.zeros(1, N + 4, dtype=torch.bool); tv[0, :N] = True
    torch.testing.assert_close(guided_attention_loss(padded, sv, tv), torch.tensor(high))
    # (B, layers, S, L) averages over layers
    both = torch.stack((diagonal, anti), 1)
    torch.testing.assert_close(guided_attention_loss(both, step_valid, text_valid), torch.tensor((low + high) / 2))


def test_guided_attention_weight_zero_is_inert_and_the_term_is_purely_additive():
    states, tokenizer, prior = prior_setup()
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(["ask him", "cut it"])
    guided = RVQPrior(replace(prior.config, guided_attention=2.))
    guided.load_state_dict(prior.state_dict())
    zero = RVQPrior(replace(prior.config, guided_attention=0., guided_attention_width=.5))
    zero.load_state_dict(prior.state_dict())
    torch.manual_seed(4)
    plain, plain_terms = rvq_prior_loss(prior, tokenizer, tokens, mask, batch, train=True)
    torch.manual_seed(4)
    inert, inert_terms = rvq_prior_loss(zero, tokenizer, tokens, mask, batch, train=True)
    assert torch.equal(plain, inert) and "guided_attention" not in inert_terms
    torch.manual_seed(4)
    total, terms = rvq_prior_loss(guided, tokenizer, tokens, mask, batch, train=True)
    assert terms["guided_attention"] > 0 and terms["tokens"] == plain_terms["tokens"]
    torch.testing.assert_close(total, plain + 2. * terms["guided_attention"])
    torch.manual_seed(4)
    evaluation, eval_terms = rvq_prior_loss(guided, tokenizer, tokens, mask, batch, train=False)
    assert "guided_attention" not in eval_terms                                 # training only
    dropped = RVQPrior(replace(guided.config, cfg_dropout=1.))                  # every text dropped: no term
    dropped.load_state_dict(prior.state_dict())
    _, dropped_terms = rvq_prior_loss(dropped, tokenizer, tokens, mask, batch, train=True)
    assert "guided_attention" not in dropped_terms


def test_alignment_loss_prefers_the_current_words_bytes_and_skips_steps_without_spans():
    spans = torch.tensor([[[1, 4], [1, 4], [5, 8], [-1, -1]]])                  # (B=1, S=4, 2)
    target, has = alignment_targets(spans, 9)
    assert has.tolist() == [[True, True, True, False]]
    torch.testing.assert_close(target[0, 0, 1:4], torch.full((3,), 1 / 3))
    on_word = target.clone(); on_word[0, 3] = 1 / 9
    uniform = torch.full((1, 4, 9), 1 / 9)
    swapped = on_word[:, [2, 2, 0, 3]]
    good, flat, bad = (float(alignment_loss(a, spans)) for a in (on_word, uniform, swapped))
    assert good < flat < bad
    torch.testing.assert_close(torch.tensor(good), torch.tensor(np.log(3.), dtype=torch.float32), atol=1e-4, rtol=0)
    assert alignment_loss(uniform, torch.full_like(spans, -1)) is None
    assert alignment_loss(uniform, spans, step_valid=torch.zeros(1, 4, dtype=torch.bool)) is None


def test_word_byte_spans_index_the_encoded_memory_and_frame_spans_follow_the_signs():
    text = "[yn 1>GIVE>2 BOOK ]"
    tokens, _ = encode_texts([text])
    spans = word_byte_spans(text)
    assert [bytes(tokens[0, a:b].tolist()).decode() for a, b in spans] == ["[yn", "1>GIVE>2", "BOOK", "]"]
    times = np.arange(24) / 24.
    signs = [("i:GIVE:j", .25, .5), ("BOOK", .5 + 1 / 24, .75), ("NOT-IN-TEXT", .8, .9)]
    frames = frame_word_spans(text, signs, times + 3.)                          # relative to the first frame
    assert frames.shape == (24, 2)
    assert (frames[:6] == -1).all() and (frames[6:13] == spans[1]).all() and (frames[13:19] == spans[2]).all()
    assert (frames[19:] == -1).all()
    # a gloss line (the gloss manifest's text) matches the segmented signs one to one
    gloss = frame_word_spans("IX-1p LIKE fs-JOHN", [("IX-1p", 0., .1), ("LIKE", .2, .3), ("fs-JOHN", .4, .5)], times)
    assert {tuple(x) for x in gloss} == {(-1, -1), *map(tuple, word_byte_spans("IX-1p LIKE fs-JOHN"))}


def test_collate_threads_optional_word_spans_and_keeps_the_default_batch():
    states, _, _ = prior_setup()
    plain = collate_text_motion([("ask him", states[0]), ("cut it", states[1])])
    assert len(plain) == 4
    span = np.full((16, 2), -1); span[4:9] = (1, 4)
    batch = collate_text_motion([("ask him", states[0], span), ("cut it", states[1], None)], word_spans=True)
    assert len(batch) == 5 and batch[-1].shape == (2, batch[2].frame_valid.shape[1], 2)
    assert batch[-1][0, 4:9].tolist() == [[1, 4]] * 5 and bool((batch[-1][1] == -1).all())


def test_alignment_supervision_joins_training_with_finite_gradients_only_for_rows_with_spans():
    states, tokenizer, prior = prior_setup()
    batch = collate_multichannel(states)
    texts = ["ask him", "cut it"]
    tokens, mask = encode_texts(texts)
    model = RVQPrior(replace(prior.config, guided_attention=1., alignment_supervision=1., cfg_dropout=0.))
    model.load_state_dict(prior.state_dict())
    model.train()
    T = batch.frame_valid.shape[1]
    spans = torch.full((2, T, 2), -1, dtype=torch.long)
    (a0, b0), (a1, b1) = word_byte_spans(texts[0])
    spans[0, :T // 2] = torch.tensor([a0, b0]); spans[0, T // 2:] = torch.tensor([a1, b1])
    total, terms = rvq_prior_loss(model, tokenizer, tokens, mask, batch, train=True, word_spans=spans)
    assert {"guided_attention", "alignment"} <= set(terms)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    model.zero_grad()
    _, none_terms = rvq_prior_loss(model, tokenizer, tokens, mask, batch, train=True,
                                   word_spans=torch.full((2, T, 2), -1, dtype=torch.long))
    assert "alignment" not in none_terms and "guided_attention" in none_terms


def test_training_script_builds_the_options_and_threads_sign_spans(tmp_path, monkeypatch):
    import hashlib
    import json
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json
    from tests.test_train_motion_tokens_script import load_script
    from tests.test_train_text_to_motion_script import store_with_manifest
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    store = tmp_path / "store"
    store_with_manifest(store)
    manifest = json.loads((store / "manifest.json").read_text())
    records = manifest["records"]
    for r in records:                                     # each word signed in turn over the first half second
        words = r["text"].split()
        r["sign_spans"] = [dict(gloss=w, seconds=[i * .5 / len(words), (i + 1) * .5 / len(words)])
                           for i, w in enumerate(words)]
    payload = canonical_json(build_corpus_manifest(corpus_id="t", inputs={"x": "b" * 64}, records=records, split={}))
    (store / "manifest.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    script = load_script()
    base = ["--store", f"dir:{store}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
            "--batch-size", "2", "--warmup", "1", "--checkpoint-every", "2", "--val-every", "1", "--val-batches", "1",
            "--workers", "0", "--trained-channels", "body,head_rotation,face", "--architecture", "rvq"]
    assert script.main(base + ["--stage", "tokenizer", "--run", "tok", "--hidden", "16", "--code-dim", "8",
                               "--codebook", "16", "--stages", "2", "--down-layers", "1", "--steps", "2"]) == 0
    key = "training-runs/tok/checkpoints/step-00000002.pt"
    sha = hashlib.sha256((store / key).read_bytes()).hexdigest()
    assert script.main(base + ["--stage", "prior", "--run", "pri", "--tokenizer-key", key, "--tokenizer-sha256", sha,
                               "--d-model", "32", "--heads", "4", "--text-layers", "1", "--token-layers", "1",
                               "--depth-layers", "1", "--feedforward", "64", "--steps", "2", "--cfg-dropout", "0",
                               "--guided-attention", ".5", "--guided-attention-width", ".3",
                               "--alignment-supervision", "1"]) == 0
    run = json.loads((store / "training-runs/pri/run.json").read_text())
    assert run["config"]["guided_attention"] == .5 and run["config"]["guided_attention_width"] == .3
    assert run["config"]["alignment_supervision"] == 1.
    log = json.loads((store / "training-runs/pri/log/step-00000002.json").read_text())
    assert all("train/guided_attention" in e and "train/alignment" in e for e in log)



def test_guided_attention_is_the_same_per_step_for_short_and_long_texts_and_skips_one_word_texts():
    # per-step averaging: a perfect diagonal costs the same small amount whatever the text length
    for N in (8, 160):
        S = 40
        sv, tv = torch.ones(1, S, dtype=torch.bool), torch.ones(1, N, dtype=torch.bool)
        t = torch.arange(S)
        diag = torch.zeros(1, S, N); diag[0, t, (t * N) // S] = 1
        anti = torch.zeros(1, S, N); anti[0, t, N - 1 - (t * N) // S] = 1
        assert float(guided_attention_loss(diag, sv, tv)) < .05 and float(guided_attention_loss(anti, sv, tv)) > .6
    # one-word texts carry no guided term (no space byte among the valid bytes); multi-word texts do
    states, tokenizer, prior = prior_setup()
    batch = collate_multichannel(states)
    guided = RVQPrior(replace(prior.config, guided_attention=1.))
    guided.load_state_dict(prior.state_dict())
    tokens, mask = encode_texts(["ask", "cut"])
    torch.manual_seed(5)
    _, terms = rvq_prior_loss(guided, tokenizer, tokens, mask, batch, train=True)
    assert "guided_attention" not in terms
    tokens, mask = encode_texts(["ask him", "cut"])
    torch.manual_seed(5)
    _, terms = rvq_prior_loss(guided, tokenizer, tokens, mask, batch, train=True)
    assert terms["guided_attention"] > 0


def test_guided_attention_anchors_both_corners_and_detects_any_whitespace():
    for N, T in ((5, 2), (9, 60), (200, 60)):
        sv, tv = torch.ones(1, T, dtype=torch.bool), torch.ones(1, N, dtype=torch.bool)
        W = guided_attention_weights(sv, tv)
        assert float(W[0, 0, 0]) == 0. and float(W[0, -1, -1]) == 0.                # BOS first, EOS last, exactly
        assert int(W[0, -1].argmin()) == N - 1
    states, tokenizer, prior = prior_setup()
    batch = collate_multichannel(states)
    guided = RVQPrior(replace(prior.config, guided_attention=1.))
    guided.load_state_dict(prior.state_dict())
    tokens, mask = encode_texts(["ask\thim", "cut\nit"])
    torch.manual_seed(5)
    _, terms = rvq_prior_loss(guided, tokenizer, tokens, mask, batch, train=True)
    assert terms["guided_attention"] > 0
