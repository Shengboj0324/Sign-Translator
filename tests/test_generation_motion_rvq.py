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
    errors = [float((q(data, stages=s)[0] - data).pow(2).sum(-1).mean()) for s in range(1, 5)]
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
        history.append(float(loss))
    assert history[-1] < .5 * history[0]
    assert {"skeleton_position", "skeleton_velocity", "skeleton_acceleration", "commitment"} <= set(terms)
    _, terms1 = rvq_loss(model, batch, stages=1)
    assert terms1["stages"] == 1.
    model.eval()
    coarse, _ = rvq_loss(model, batch, stages=1)
    fine, _ = rvq_loss(model, batch, stages=3)
    assert float(fine) < float(coarse)                                            # more stages, better reconstruction


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
        first = first or float(loss)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert float(loss) < .3 * first and terms["token_accuracy"] > .8
    out = generate_rvq_motion(prior, tokenizer, ["ask him"], fps=24., durations=[.5], guidance=2.)
    out[0].validate()
    assert out[0].channels["body"].valid.all() and len(out[0].timestamps) == 12
