"""Tokens v2: part-wise residual VQ and the depth-autoregressive prior."""
import numpy as np
import pytest
import torch

from signtranslator.data.multichannel import collate_multichannel
from signtranslator.generation.canonical_state import channel_bindings_from_state
from signtranslator.generation.motion_rvq import (
    MotionRVQ, MotionRVQConfig, RVQPrior, RVQPriorConfig, ResidualQuantizer, generate_rvq_motion, rvq_loss,
    rvq_prior_loss, rvq_sequence_log_likelihood,
)
from signtranslator.generation.text_to_motion import TextToMotionConfig, diffusion_targets, encode_texts
from tests.test_generation_signing_space import signer_state

TRAINED = ("body", "left_hand", "right_hand", "head_rotation")


def motion_config(state):
    return TextToMotionConfig(bindings=channel_bindings_from_state(state), trained_channels=TRAINED, d_model=32, heads=4,
                              text_layers=1, motion_layers=1, feedforward=64, dropout=0.)


def small(state, **kw):
    return MotionRVQConfig(motion_config(state), hidden=48, code_dim=16, codebook=32, stages=3, **kw)


def test_residual_quantizer_is_exact_on_sums_of_codes_and_improves_with_stages():
    torch.manual_seed(0)
    q = ResidualQuantizer(stages=3, codebook=8, dim=4, ema=.9)
    with torch.no_grad():
        q.codes.copy_(torch.stack([torch.eye(8, 4) * 4, torch.eye(8, 4), torch.eye(8, 4) * .25]))
    pick = torch.tensor([[0, 1, 2], [3, 3, 0]])
    z = torch.stack([q.codes[s][pick[:, s]] for s in range(3)]).sum(0)
    total, idx, _ = q(z)
    torch.testing.assert_close(total, z)
    assert idx.tolist() == pick.tolist()
    # learnt by EMA on Gaussian data, more stages never quantise worse on average
    q = ResidualQuantizer(stages=4, codebook=16, dim=6, ema=.8)
    data = torch.randn(4000, 6)
    for _ in range(30):
        total, idx, inputs = q(data)
        q.update(inputs, idx, torch.ones(len(data), dtype=torch.bool))
    errors = [float((q(data, stages=s)[0] - data).pow(2).sum(-1).mean().detach()) for s in range(1, 5)]
    assert all(b <= a + 1e-6 for a, b in zip(errors, errors[1:])) and errors[-1] < .5 * errors[0]


def test_parts_cover_every_unit_column_once_and_the_code_path_equals_the_forward_path():
    state = signer_state(0, frames=16)
    config = small(state)
    parts = config.part_slices()
    cols = sorted(c for _, cs in parts for c in cs)
    assert cols == list(range(config.width)) and [p for p, _ in parts] == ["body", "left_hand", "right_hand", "face"]
    assert config.codes_per_step == 4 * 3
    model = MotionRVQ(config).eval()
    batch = collate_multichannel([state])
    unit = diffusion_targets(config.motion, batch)
    out, latents, quantized, codes, valid = model(unit, batch.frame_valid)
    assert codes.shape == (1, 8, 12) and valid.all()
    torch.testing.assert_close(model.decode_codes(codes, unit.shape[1]), out)      # eval: straight-through == codes


def test_tokenizer_v2_learns_and_quantizer_dropout_uses_fewer_stages():
    torch.manual_seed(0)
    states = [signer_state(s, frames=24, amplitude=.4) for s in range(4)]
    batch = collate_multichannel(states)
    model = MotionRVQ(small(states[0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    history = []
    for step in range(120):
        loss, terms = rvq_loss(model, batch, stages=3)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        history.append(float(loss.detach()))
    assert history[-1] < .5 * history[0]
    assert {"skeleton_position", "skeleton_velocity", "skeleton_acceleration", "commitment"} <= set(terms)
    _, terms1 = rvq_loss(model, batch, stages=1)
    assert terms1["stages"] == 1.
    model.eval()
    coarse, _ = rvq_loss(model, batch, stages=1)
    fine, _ = rvq_loss(model, batch, stages=3)
    assert float(fine.detach()) < float(coarse.detach())                                            # more stages, better reconstruction


def prior_setup(seed=0):
    torch.manual_seed(seed)
    states = [signer_state(s, frames=16, amplitude=.4) for s in range(2)]
    tokenizer = MotionRVQ(small(states[0])).eval()
    config = RVQPriorConfig(tokenizer.config, d_model=32, heads=4, text_layers=1, temporal_layers=1, depth_layers=1,
                            feedforward=64, dropout=0.)
    return states, tokenizer, RVQPrior(config)


def test_depth_transformer_is_causal_over_the_codes_of_a_step():
    _, tokenizer, prior = prior_setup()
    prior.eval()
    h = torch.randn(3, 32)
    codes = torch.randint(0, 32, (3, prior.J - 1))
    full = prior.depth_logits(h, codes)
    changed = codes.clone(); changed[:, 5:] = (changed[:, 5:] + 1) % 32
    again = prior.depth_logits(h, changed)
    torch.testing.assert_close(full[:, :6], again[:, :6])                        # position j sees codes < j only
    assert not torch.allclose(full[:, 6:], again[:, 6:])
    torch.testing.assert_close(prior.depth_logits(h, codes[:, :4]), full[:, :5])  # prefixes agree


def test_prior_v2_learns_its_codes_and_generates_valid_states():
    states, tokenizer, prior = prior_setup()
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(["ask him", "cut it"])
    optimizer = torch.optim.Adam(prior.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss, terms = rvq_prior_loss(prior, tokenizer, tokens, mask, batch, train=False)
        first = first or float(loss.detach())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert float(loss.detach()) < .3 * first and terms["token_accuracy"] > .8
    out = generate_rvq_motion(prior, tokenizer, ["ask him"], fps=24., durations=[.5], guidance=2.)
    out[0].validate()
    assert out[0].channels["body"].valid.all() and len(out[0].timestamps) == 12


def test_base_blocks_give_a_full_rate_tokenizer_depth_and_keep_old_checkpoints_loadable():
    state = signer_state(0, frames=12)
    base = small(state)
    old = MotionRVQ(base)
    from dataclasses import replace
    assert MotionRVQ(MotionRVQConfig.from_dict(base.to_dict())).state_dict().keys() == old.state_dict().keys()
    deep = MotionRVQ(replace(base, down_layers=0, base_blocks=3))
    shallow = MotionRVQ(replace(base, down_layers=0))
    count = lambda m: sum(1 for x in m.decoder if type(x).__name__ == "_ResBlock")
    assert count(deep) == 3 and count(shallow) == 0
    # the receptive field grows: an impulse in frame 6 reaches frames further away
    torch.manual_seed(0)
    for m in (deep, shallow):
        for p in m.parameters():
            torch.nn.init.normal_(p, std=.2)
    x = torch.zeros(1, 12, base.width); x[:, 6] = 1.
    zero = torch.zeros_like(x)
    spread = lambda m: int(((m.encode(x)["body"] - m.encode(zero)["body"]).abs().sum(-1)[0] > 1e-6).sum())
    assert spread(deep) > spread(shallow)


def test_per_part_temperature_equals_the_scalar_when_uniform_and_changes_only_its_parts():
    import pytest
    states, tokenizer, prior = prior_setup()
    parts = [p for p, _ in tokenizer.parts]
    scalar = generate_rvq_motion(prior, tokenizer, ["ask him"], fps=24., durations=[.5], temperature=1.3, seed=3)[0]
    uniform = generate_rvq_motion(prior, tokenizer, ["ask him"], fps=24., durations=[.5],
                                  temperature={p: 1.3 for p in parts}, seed=3)[0]
    for name in ("body", "left_hand", "face"):
        np.testing.assert_array_equal(scalar.channels[name].values, uniform.channels[name].values)
    with pytest.raises(ValueError, match="unknown parts"):
        generate_rvq_motion(prior, tokenizer, ["ask him"], fps=24., durations=[.5], temperature={"tail": 2.})


def test_batched_generation_equals_per_text_generation_when_deterministic():
    from signtranslator.generation.motion_rvq import generate_rvq_motion_batch
    states, tokenizer, prior = prior_setup()
    texts, durations = ["ask him", "cut it now", "go"], [.5, .3, .42]
    for guidance in (1., 2.):
        one = [generate_rvq_motion(prior, tokenizer, [t], fps=24., durations=[d], guidance=guidance, top_k=1)[0]
               for t, d in zip(texts, durations)]
        many = generate_rvq_motion_batch(prior, tokenizer, texts, fps=24., durations=durations, guidance=guidance,
                                         top_k=1)
        for a, b in zip(one, many):
            b.validate()
            assert len(a.timestamps) == len(b.timestamps)
            for name in a.channels:
                torch.testing.assert_close(torch.as_tensor(a.channels[name].values),
                                           torch.as_tensor(b.channels[name].values), atol=1e-5, rtol=1e-5)


def test_velocity_matching_is_zero_when_exact_one_when_static_and_penalises_jitter():
    from signtranslator.generation.motion_rvq import velocity_matching_loss
    states = [signer_state(s, frames=24, amplitude=.4) for s in range(2)]
    batch = collate_multichannel(states)
    ch = batch.channels["right_hand"]
    exact = ch.values.double()
    assert float(velocity_matching_loss(exact, ch, batch.frame_valid)) < 1e-12
    static = exact[:, :1].expand_as(exact).clone()
    assert abs(float(velocity_matching_loss(static, ch, batch.frame_valid)) - 1.) < 1e-9       # damping to stillness
    jitter = exact + .05 * (-1.) ** torch.arange(exact.shape[1], dtype=exact.dtype)[None, :, None, None]
    assert float(velocity_matching_loss(jitter, ch, batch.frame_valid)) > .01                   # added shake


def test_velocity_weights_join_the_loss_only_when_set_and_leave_old_configs_unchanged():
    torch.manual_seed(0)
    states = [signer_state(s, frames=24, amplitude=.4) for s in range(2)]
    batch = collate_multichannel(states)
    plain = small(states[0])
    weighted = small(states[0], velocity_weights={"right_hand": 1., "body": 2.})
    assert "velocity_weights" not in plain.to_dict()                                  # earlier tokenizers' SHAs unchanged
    assert MotionRVQConfig.from_dict(plain.to_dict()).sha256() == plain.sha256()
    assert MotionRVQConfig.from_dict(weighted.to_dict()) == weighted and weighted.sha256() != plain.sha256()
    _, terms = rvq_loss(MotionRVQ(weighted), batch, stages=3)
    assert {"velocity_right_hand", "velocity_body"} <= set(terms)
    _, terms = rvq_loss(MotionRVQ(plain), batch, stages=3)
    assert not any(k.startswith("velocity_") for k in terms)


def test_stable_velocity_matching_uses_a_running_scale_and_is_not_blown_up_by_a_still_batch():
    from signtranslator.generation.motion_rvq import velocity_matching_loss
    states = [signer_state(s, frames=24, amplitude=.4) for s in range(2)]
    batch = collate_multichannel(states)
    ch = batch.channels["right_hand"]
    exact = ch.values.double().requires_grad_(True)
    scale = torch.zeros((), dtype=torch.float64)
    assert float(velocity_matching_loss(exact, ch, batch.frame_valid, scale=scale).detach()) < 1e-12 and float(scale) > 0
    moving = float(scale)
    still = [signer_state(s, frames=24, amplitude=1e-6) for s in range(2)]           # a near-still batch
    sb = collate_multichannel(still)
    sc = sb.channels["right_hand"]
    noisy = (sc.values.double() + 1e-3 * (-1.) ** torch.arange(24, dtype=torch.float64)[None, :, None, None])
    stable = float(velocity_matching_loss(noisy.requires_grad_(True), sc, sb.frame_valid, scale=scale).detach())
    unstable = float(velocity_matching_loss(noisy, sc, sb.frame_valid).detach())
    assert unstable > 1e3 * stable                     # the per-batch form explodes on a still batch; the running one does not
    assert abs(float(scale) - moving) / moving < .05   # one batch moves the running scale by at most (1 - momentum)


def test_velocity_scales_are_not_saved_so_s8v_checkpoints_load_under_the_new_code():
    states = [signer_state(s, frames=24, amplitude=.4) for s in range(2)]
    weighted = small(states[0], velocity_weights={"right_hand": 1.})
    model = MotionRVQ(weighted)
    assert not any(k.startswith("velocity_scale_") for k in model.state_dict())
    MotionRVQ(weighted).load_state_dict(model.state_dict())                 # strict load succeeds



def test_codebook_rate_scales_the_ema_update_zero_freezes_the_codebooks_and_the_flag_keeps_old_shas():
    states = [signer_state(s, frames=24, amplitude=.4) for s in range(2)]
    batch = collate_multichannel(states)
    base = small(states[0])
    assert "codebook_follows_lr" not in base.to_dict()                                  # earlier configs unchanged
    assert small(states[0], codebook_follows_lr=True).to_dict()["codebook_follows_lr"] is True

    def moved(rate, reseed=True):
        torch.manual_seed(0)
        model = MotionRVQ(base).train()
        model.codebook_rate, model.codebook_reseed = rate, reseed
        before = model.quantizers["body"].codes.clone()
        torch.manual_seed(1)
        rvq_loss(model, batch, stages=3)
        return float((model.quantizers["body"].codes - before).abs().max())
    full, half, frozen = moved(1.), moved(.5), moved(0.)
    assert frozen == 0. and 0. < half < full


class _FakeLM(torch.nn.Module):
    """A tiny deterministic stand-in for a Hugging Face encoder (last_hidden_state of a word-hash embedding)."""
    def __init__(self):
        super().__init__()
        self.config = type("C", (), {"hidden_size": 12})()
        self.table = torch.nn.Embedding(97, 12)

    def forward(self, input_ids, attention_mask):
        return type("O", (), {"last_hidden_state": self.table(input_ids % 97)})()


class _FakeTokenizer:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        rows = [[1] + [sum(map(ord, w)) % 90 + 3 for w in t.split()] + [2] for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


def test_a_prior_with_a_frozen_pretrained_text_model_trains_generates_and_keeps_it_out_of_its_state(monkeypatch):
    from dataclasses import replace
    import signtranslator.generation.motion_rvq as mr
    monkeypatch.setattr(mr, "load_text_model", lambda name, revision=None: (_FakeLM(), _FakeTokenizer()))
    states, tokenizer, byte_prior = prior_setup()
    config = replace(byte_prior.config, text_model="fake/encoder")
    assert "text_model" in config.to_dict() and "text_model" not in byte_prior.config.to_dict()   # old SHAs unchanged
    prior = RVQPrior(config)
    frozen = dict(prior._lm.named_parameters())
    assert not any(k.startswith("_lm") for k in prior.state_dict()) and all(not p.requires_grad for p in frozen.values())
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(["ask him", "cut it"])
    hidden, _, memory_mask = prior.encode(tokens, mask)
    assert hidden.shape[1] == memory_mask.shape[1] == 4            # the text model's tokens, not the bytes
    optimizer = torch.optim.Adam([p for p in prior.parameters() if p.requires_grad], lr=3e-3)
    first = None
    for _ in range(150):
        loss, terms = rvq_prior_loss(prior, tokenizer, tokens, mask, batch, train=False)
        first = first or float(loss.detach())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert float(loss.detach()) < .3 * first and terms["token_accuracy"] > .8
    assert all(torch.equal(p, frozen[k]) for k, p in prior._lm.named_parameters())   # the text model never trains
    out = generate_rvq_motion(prior, tokenizer, ["ask him"], fps=24., durations=[.5], guidance=2.)
    out[0].validate()
    from signtranslator.generation.motion_rvq import generate_rvq_motion_batch
    batch_out = generate_rvq_motion_batch(prior, tokenizer, ["ask him", "cut it"], fps=24., durations=[.5, .4], guidance=2.)
    assert len(batch_out) == 2 and all(o.channels["body"].valid.all() for o in batch_out)
    again = RVQPrior(config); again.load_state_dict(prior.state_dict())             # strict load without the LM


def test_input_corruption_is_opt_in_training_only_and_keeps_old_config_shas():
    from dataclasses import replace
    states, tokenizer, prior = prior_setup()
    assert "input_corruption" not in prior.config.to_dict()                     # earlier priors' SHAs unchanged
    corrupted = RVQPrior(replace(prior.config, input_corruption=.5))
    corrupted.load_state_dict(prior.state_dict())
    assert corrupted.config.to_dict()["input_corruption"] == .5
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(["ask him", "cut it"])
    prior.eval(); corrupted.eval()
    torch.manual_seed(1)
    plain, _ = rvq_prior_loss(prior, tokenizer, tokens, mask, batch, train=False)
    torch.manual_seed(1)
    same, _ = rvq_prior_loss(corrupted, tokenizer, tokens, mask, batch, train=False)
    torch.testing.assert_close(plain, same)                                     # evaluation never corrupts
    torch.manual_seed(2)
    a, _ = rvq_prior_loss(corrupted, tokenizer, tokens, mask, batch, train=True)
    torch.manual_seed(2)
    b, _ = rvq_prior_loss(prior, tokenizer, tokens, mask, batch, train=True)
    assert not torch.isclose(a, b)                                              # training sees a corrupted history


def test_sequence_log_likelihood_sums_the_evaluation_loss_per_row_and_depends_on_the_text():
    states, tokenizer, prior = prior_setup()
    prior.eval()
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(["ask him", "cut it"])
    loss, terms = rvq_prior_loss(prior, tokenizer, tokens, mask, batch, train=False)
    ll = rvq_sequence_log_likelihood(prior, tokenizer, tokens, mask, batch)
    valid = tokenizer.token_valid(batch.frame_valid)
    codes = int(valid.sum()) * tokenizer.config.codes_per_step
    assert ll.shape == (2,) and bool((ll < 0).all())
    torch.testing.assert_close(-ll.sum() / codes, torch.tensor(terms["tokens"], dtype=ll.dtype), rtol=1e-5, atol=1e-5)
    one = rvq_sequence_log_likelihood(prior, tokenizer, tokens[1:], mask[1:], collate_multichannel(states[1:]))
    torch.testing.assert_close(one, ll[1:], rtol=1e-4, atol=1e-4)            # rows are scored independently
    swapped, smask = encode_texts(["cut it", "ask him"])
    assert not torch.allclose(rvq_sequence_log_likelihood(prior, tokenizer, swapped, smask, batch), ll)


def test_guidance_encodes_the_empty_text_as_training_drops_it_zero_slots_through_the_locus_projection():
    from dataclasses import replace
    from signtranslator.generation.motion_rvq import generate_rvq_motion_batch
    from signtranslator.generation.motion_tokens import _empty_like
    from signtranslator.generation.text_to_motion import LOCUS_SLOT_WIDTH, PAD, training_loci
    states, tokenizer, prior = prior_setup()
    prior = RVQPrior(replace(prior.config, locus_conditioning=True, locus_dropout=0.)).eval()
    torch.nn.init.normal_(prior.loci.bias, std=1.)                             # a trained projection has a bias
    seen = []
    encode = prior.encode
    prior.encode = lambda tokens, mask, loci=None: seen.append(loci) or encode(tokens, mask, loci)
    generate_rvq_motion(prior, tokenizer, ["ask him"], fps=24., durations=[.25], guidance=2.)
    generate_rvq_motion_batch(prior, tokenizer, ["ask him", "cut it"], fps=24., durations=[.25, .25], guidance=2.)
    empties = [seen[1], seen[3]]                                                # each sampler: text, then empty text
    for loci in empties:
        assert loci is not None and loci.shape == (1, LOCUS_SLOT_WIDTH) and not bool(loci.any())
    # the same input as a training row whose text was dropped: slots zeroed, projection applied
    tokens, mask = encode_texts(["ask him"])
    dropped = training_loci(prior.config, torch.ones(1, LOCUS_SLOT_WIDTH), torch.zeros(1, dtype=torch.bool))
    empty = _empty_like(tokens[:, :2])
    with torch.no_grad():
        torch.testing.assert_close(encode(empty, empty != PAD, dropped)[0], encode(empty, empty != PAD, empties[0])[0])
        assert not torch.allclose(encode(empty, empty != PAD, None)[0], encode(empty, empty != PAD, empties[0])[0])


def test_scheduled_sampling_is_opt_in_training_only_feeds_model_samples_and_keeps_old_config_shas():
    from dataclasses import replace
    states, tokenizer, prior = prior_setup()
    assert "scheduled_sampling" not in prior.config.to_dict()                    # earlier priors' SHAs unchanged
    scheduled = RVQPrior(replace(prior.config, scheduled_sampling=1.))
    scheduled.load_state_dict(prior.state_dict())
    assert scheduled.config.to_dict()["scheduled_sampling"] == 1.
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(["ask him", "cut it"])
    prior.eval(); scheduled.eval()
    torch.manual_seed(1)
    plain, _ = rvq_prior_loss(prior, tokenizer, tokens, mask, batch, train=False)
    torch.manual_seed(1)
    same, _ = rvq_prior_loss(scheduled, tokenizer, tokens, mask, batch, train=False)
    torch.testing.assert_close(plain, same)                                     # evaluation never replaces history
    torch.manual_seed(2)
    a, _ = rvq_prior_loss(scheduled, tokenizer, tokens, mask, batch, train=True)
    torch.manual_seed(2)
    b, _ = rvq_prior_loss(prior, tokenizer, tokens, mask, batch, train=True)
    assert not torch.isclose(a, b)                                              # training sees its own samples
    a.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in scheduled.parameters())


def test_history_dropout_is_opt_in_training_only_and_hides_whole_spans_behind_the_start_code():
    from dataclasses import replace
    states, tokenizer, prior = prior_setup()
    assert "history_dropout" not in prior.config.to_dict() and "history_stages" not in prior.config.to_dict()
    dropped = RVQPrior(replace(prior.config, history_dropout=1.))
    dropped.load_state_dict(prior.state_dict())
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(["ask him", "cut it"])
    prior.eval(); dropped.eval()
    torch.manual_seed(1)
    plain, _ = rvq_prior_loss(prior, tokenizer, tokens, mask, batch, train=False)
    torch.manual_seed(1)
    same, _ = rvq_prior_loss(dropped, tokenizer, tokens, mask, batch, train=False)
    torch.testing.assert_close(plain, same)                                     # evaluation never hides history
    seen = []
    original = dropped.temporal_states
    dropped.temporal_states = lambda h, m, previous, *a: seen.append(previous.clone()) or original(h, m, previous, *a)
    torch.manual_seed(3)
    rvq_prior_loss(dropped, tokenizer, tokens, mask, batch, train=True)
    previous = seen[0]
    start = (previous == dropped.K).all(-1)                                     # (B, S): steps showing the start code
    for row in start:
        k = int(row.long().sum())
        assert k >= 1 and bool(row[:k].all()) and not bool(row[k:].any())       # one leading span, then real codes


def test_history_stages_limits_the_temporal_input_to_the_coarse_codes_in_training_and_generation():
    from dataclasses import replace
    states, tokenizer, prior = prior_setup()
    coarse = RVQPrior(replace(prior.config, history_stages=1))
    coarse.load_state_dict(prior.state_dict())
    P = coarse.J // coarse.config.tokenizer.stages
    hidden = torch.zeros(1, 3, coarse.config.d_model); mask = torch.ones(1, 3, dtype=torch.bool)
    codes = torch.randint(0, coarse.K, (1, 4, coarse.J))
    other = codes.clone(); other[..., P:] = torch.randint(0, coarse.K, other[..., P:].shape)   # fine stages differ
    seconds = torch.arange(4.)[None] / 24.; duration = torch.tensor([4 / 24.])
    coarse.eval()
    with torch.no_grad():
        torch.testing.assert_close(coarse.temporal_states(hidden, mask, codes, seconds, duration),
                                   coarse.temporal_states(hidden, mask, other, seconds, duration))
        assert not torch.allclose(prior.eval().temporal_states(hidden, mask, codes, seconds, duration),
                                  prior.temporal_states(hidden, mask, other, seconds, duration))
    out = generate_rvq_motion(coarse, tokenizer, ["ask him"], fps=24., durations=[.25], guidance=1.5)[0]
    assert len(out.timestamps) == 6
