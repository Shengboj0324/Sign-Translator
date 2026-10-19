"""Diffusion objective: schedule, exact maps, regressor-equivalent start, mode recovery."""
import numpy as np
import torch

from signtranslator.data.multichannel import collate_multichannel
from signtranslator.generation.canonical_state import channel_bindings_from_state
from signtranslator.generation.text_to_motion import (
    TextToMotion, TextToMotionConfig, alpha_bar, diffusion_targets, encode_texts, generate, heads_to_unit,
    training_loss,
)
from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix
from tests.test_generation_text_to_motion import config, state


def bimodal_state(sign, k, frames=24):
    """Every frame rotates the body elements by sign·45° about z; the face is constant."""
    t = np.arange(frames) / 24.
    channels = {}
    for name, (kind, width, units) in CHANNELS.items():
        if name == "body":
            aa = np.zeros((frames, 3, 3))
            aa[..., 2] = sign * np.pi / 4
            values, valid = matrix_to_rotation_6d(axis_angle_to_matrix(torch.from_numpy(aa))).numpy(), np.ones((frames, 3), bool)
        elif name == "face":
            values, valid = np.full((frames, 2, 1), .5), np.ones((frames, 2), bool)
        else:
            values, valid = np.zeros((frames, 1, width)), np.zeros((frames, 1), bool)
        channels[name] = MotionChannel(np.where(valid[..., None], values, 0.), valid, np.zeros_like(valid),
                                       np.where(valid, 1., 0.), tuple(f"{name}{i}" for i in range(valid.shape[1])),
                                       "f", units, "src", "a" * 64, "c", inference_method="estimate" if valid.any() else None)
    return MultichannelMotion(t, "clock", f"s{k}", channels)


def body_angle(generated):
    m = rotation_6d_to_matrix(torch.from_numpy(generated.channels["body"].values))
    return float(np.degrees(np.arctan2(m[..., 1, 0].numpy(), m[..., 0, 0].numpy())).mean())


def test_cosine_schedule_runs_from_clean_to_noise():
    t = torch.linspace(0, 1, 101)
    ab = alpha_bar(t)
    assert abs(float(ab[0]) - 1.) < 1e-6 and float(ab[-1]) <= 1e-4 and bool((ab[1:] <= ab[:-1]).all())


def test_targets_map_exactly_into_the_unit_space_and_back():
    states = [state(0, 20), state(1, 30)]
    cfg = config(states, objective="diffusion")
    batch = collate_multichannel(states)
    x0 = diffusion_targets(cfg, batch)
    assert x0.shape == (2, 30, 3 * 6 + 1 * 6 + 3 * 1)
    assert float(x0[0, 20:].abs().max()) == 0.                       # padded frames are zero
    # The x0 estimate of perfect head outputs is the target itself.
    perfect = {name: batch.channels[name].values[:1, :20].float() for name in ("body", "head_rotation", "face")}
    np.testing.assert_allclose(heads_to_unit(cfg, perfect)[0].numpy(), x0[0, :20].numpy(), atol=1e-6)
    np.testing.assert_allclose(x0[0, :20, -3:].numpy(), (batch.channels["face"].values[0, :20, :, 0] / .5).numpy(),
                               atol=1e-6)


def test_untrained_diffusion_decoder_equals_the_regressor():
    states = [state(0, 20)]
    torch.manual_seed(0)
    regression = TextToMotion(config(states))
    torch.manual_seed(0)
    diffusion = TextToMotion(config(states, objective="diffusion"))
    shared = {k: v for k, v in diffusion.state_dict().items() if k in regression.state_dict()}
    regression.load_state_dict(shared)
    tokens, mask = encode_texts(["x"])
    seconds = torch.arange(20)[None].float() / 24
    frames = torch.ones(1, 20, dtype=torch.bool)
    hidden, _ = diffusion.encode(tokens, mask)
    noisy = torch.randn(1, 20, 27)
    a = diffusion.decode(hidden, mask, seconds, frames, torch.tensor([1.]), noisy=noisy, level=torch.tensor([.5]))
    b = regression.decode(*regression.encode(tokens, mask)[:1], mask, seconds, frames, torch.tensor([1.]))
    for name in a:
        torch.testing.assert_close(a[name], b[name])


def test_diffusion_recovers_both_modes_where_regression_averages_them():
    states = [bimodal_state(1 if k % 2 else -1, k) for k in range(16)]
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(["sign"] * 16)
    results = {}
    for objective in ("regression", "diffusion"):
        torch.manual_seed(0)
        cfg = TextToMotionConfig(bindings=channel_bindings_from_state(states[0]), trained_channels=("body", "face"),
                                 d_model=64, heads=4, text_layers=1, motion_layers=2, feedforward=128, dropout=0.,
                                 objective=objective)
        model = TextToMotion(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        for _ in range(400):
            model.train()
            loss, _, _ = training_loss(model, tokens, mask, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        results[objective] = [body_angle(generate(model, ["sign"], fps=24., seed=s, steps=30, guidance=1.)[0])
                              for s in range(12)]
    assert max(abs(a) for a in results["regression"]) < 5.                       # the mean of ±45°
    diffusion = np.array(results["diffusion"])
    assert np.all(np.abs(np.abs(diffusion) - 45.) < 10.)                          # every sample on a mode
    assert (diffusion > 0).sum() >= 2 and (diffusion < 0).sum() >= 2              # both modes are sampled


def test_diffusion_generation_is_valid_seeded_and_guided():
    states = [state(i, 24 + 12 * i) for i in range(3)]
    texts = ["the dog runs", "a cat sleeps on the mat", "birds fly south in winter now"]
    torch.manual_seed(0)
    model = TextToMotion(config(states, objective="diffusion"))
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(texts)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(60):
        loss, terms, _ = training_loss(model, tokens, mask, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert np.isfinite(float(loss.detach())) and "log_duration" in terms
    first = generate(model, texts[:2], fps=24., seed=3, steps=10)
    again = generate(model, texts[:2], fps=24., seed=3, steps=10)
    other = generate(model, texts[:2], fps=24., seed=4, steps=10)
    for g in first:
        g.validate()
        assert not g.channels["body"].observed.any() and g.channels["body"].valid.all()
        assert not g.channels["root_translation"].valid.any()
    np.testing.assert_array_equal(first[0].channels["body"].values, again[0].channels["body"].values)
    assert not np.array_equal(first[0].channels["body"].values, other[0].channels["body"].values)
    face = first[0].channels["face"].values
    assert np.isfinite(face).all()
