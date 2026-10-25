"""Tokens v2: part-wise residual VQ and the depth-autoregressive prior."""
import numpy as np
import pytest
import torch

from signtranslator.data.multichannel import collate_multichannel
from signtranslator.generation.canonical_state import channel_bindings_from_state
from signtranslator.generation.motion_rvq import (
    MotionRVQ, MotionRVQConfig, RVQPrior, RVQPriorConfig, ResidualQuantizer, generate_rvq_motion, rvq_loss,
    rvq_prior_loss,
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
