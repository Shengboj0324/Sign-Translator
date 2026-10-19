"""Motion tokens: quantizer maths, tokenizer reconstruction, prior learning and mode recovery."""
import numpy as np
import torch

from signtranslator.data.multichannel import collate_multichannel
from signtranslator.generation.canonical_state import channel_bindings_from_state
from signtranslator.generation.motion_tokens import (
    MotionVQ, MotionVQConfig, TokenPrior, TokenPriorConfig, generate_tokens_motion, prior_loss, tokenizer_loss,
)
from signtranslator.generation.text_to_motion import TextToMotionConfig, diffusion_targets, encode_texts
from tests.test_generation_diffusion import bimodal_state, body_angle
from tests.test_generation_text_to_motion import state


def motion_config(states, channels=("body", "head_rotation", "face")):
    return TextToMotionConfig(bindings=channel_bindings_from_state(states[0]), trained_channels=channels,
                              d_model=32, heads=4, text_layers=1, motion_layers=1, feedforward=64, dropout=0.)


def test_quantizer_picks_nearest_codes_and_passes_gradients_straight_through():
    states = [state(0, 20)]
    vq = MotionVQ(MotionVQConfig(motion_config(states), hidden=16, code_dim=4, codebook=8))
    with torch.no_grad():
        vq.codes.copy_(torch.eye(8, 4) * 3)
    latents = torch.tensor([[[2.9, 0, 0, 0], [0, 0, .1, 2.8]]], requires_grad=True)
    quantized, index, straight = vq.quantize(latents)
    assert index.tolist() == [[0, 3]]
    torch.testing.assert_close(straight.detach(), quantized)
    straight.sum().backward()
    torch.testing.assert_close(latents.grad, torch.ones_like(latents))


def test_frames_of_any_length_round_trip_in_shape():
    states = [state(0, 23)]
    vq = MotionVQ(MotionVQConfig(motion_config(states), hidden=16, code_dim=8, codebook=16))
    batch = collate_multichannel(states)
    unit = diffusion_targets(vq.config.motion, batch)
    out, latents, quantized, index, token_valid = vq(unit, batch.frame_valid)
    assert out.shape == unit.shape and index.shape == (1, 6) and token_valid.all()


def test_tokenizer_learns_to_reconstruct_and_uses_its_codebook():
    torch.manual_seed(0)
    states = [state(i, 24 + 8 * i) for i in range(4)]
    vq = MotionVQ(MotionVQConfig(motion_config(states), hidden=48, code_dim=16, codebook=32))
    batch = collate_multichannel(states)
    optimizer = torch.optim.Adam(vq.parameters(), lr=2e-3)
    vq.train()
    first, _ = tokenizer_loss(vq, batch)
    for _ in range(300):
        loss, terms = tokenizer_loss(vq, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss) < .3 * float(first) and terms["codes_used"] >= 4


def test_token_prior_samples_both_modes_and_states_are_valid():
    torch.manual_seed(0)
    states = [bimodal_state(1 if k % 2 else -1, k) for k in range(16)]
    batch = collate_multichannel(states)
    config = motion_config(states, channels=("body", "face"))
    vq = MotionVQ(MotionVQConfig(config, hidden=48, code_dim=16, codebook=16))
    optimizer = torch.optim.Adam(vq.parameters(), lr=2e-3)
    vq.train()
    for _ in range(300):
        loss, _ = tokenizer_loss(vq, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    prior = TokenPrior(TokenPriorConfig(vq.config, d_model=48, heads=4, text_layers=1, token_layers=2,
                                        feedforward=96, dropout=0.))
    tokens, mask = encode_texts(["sign"] * 16)
    optimizer = torch.optim.Adam(prior.parameters(), lr=2e-3)
    for _ in range(300):
        prior.train()
        loss, terms = prior_loss(prior, vq, tokens, mask, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    angles = np.array([body_angle(generate_tokens_motion(prior, vq, ["sign"], fps=24., seed=s)[0]) for s in range(12)])
    assert np.all(np.abs(np.abs(angles) - 45.) < 10.), angles                 # every sample on a mode
    assert (angles > 0).sum() >= 2 and (angles < 0).sum() >= 2, angles        # both modes sampled
    generated = generate_tokens_motion(prior, vq, ["sign"], fps=24., seed=1)[0]
    generated.validate()
    assert not generated.channels["body"].observed.any()
    again = generate_tokens_motion(prior, vq, ["sign"], fps=24., seed=1)[0]
    np.testing.assert_array_equal(generated.channels["body"].values, again.channels["body"].values)
