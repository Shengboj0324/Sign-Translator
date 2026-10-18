"""Text-to-motion: byte encoding, neutral start, real learning, valid generation."""
import numpy as np
import pytest
import torch

from signtranslator.data.multichannel import collate_multichannel
from signtranslator.generation.canonical_state import channel_bindings_from_state
from signtranslator.generation.text_to_motion import (
    TextToMotion, TextToMotionConfig, encode_texts, generate, training_loss, utterance_duration,
)
from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d


def state(seed, frames):
    rng = np.random.default_rng(seed)
    t = np.arange(frames) / 24.
    channels = {}
    for name, (kind, width, units) in CHANNELS.items():
        elements = 3 if name in ("body", "face") else 1
        if kind == "rotation_6d_columns":
            axis = rng.normal(size=(elements, 3))
            angle = (seed + 1) * .4 * np.sin(2 * t)[:, None]
            values = matrix_to_rotation_6d(axis_angle_to_matrix(torch.from_numpy(axis[None] * angle[..., None]))).numpy()
            valid = np.ones((frames, elements), bool)
        elif name == "face":
            values = np.clip(.5 + .3 * np.sin(t + seed)[:, None, None] * np.ones((1, elements, 1)), 0, 1)
            valid = np.ones((frames, elements), bool)
        else:
            values, valid = np.zeros((frames, 1, width)), np.zeros((frames, 1), bool)
            elements = 1
        channels[name] = MotionChannel(np.where(valid[..., None], values, 0.), valid, np.zeros_like(valid),
                                       np.where(valid, 1., 0.), tuple(f"{name}{i}" for i in range(valid.shape[1])),
                                       "f", units, "src", "a" * 64, "c", inference_method="estimate" if valid.any() else None)
    return MultichannelMotion(t, "clock", f"s{seed}", channels)


def config(states, **kw):
    bindings = channel_bindings_from_state(states[0])
    return TextToMotionConfig(bindings=bindings, trained_channels=("body", "head_rotation", "face"),
                              d_model=32, heads=4, text_layers=1, motion_layers=1, feedforward=64, dropout=0., **kw)


def test_byte_encoding_is_exact_and_rejects_empty_text():
    tokens, mask = encode_texts(["héllo", "a"])
    assert tokens[0, 1:-1].tolist()[:2] == [104, 195] and mask.sum(1).tolist() == [8, 3]
    with pytest.raises(ValueError):
        encode_texts(["  "])


def test_untrained_model_predicts_the_neutral_pose():
    states = [state(0, 20)]
    model = TextToMotion(config(states))
    tokens, mask = encode_texts(["x"])
    seconds = torch.arange(20)[None].float() / 24
    out, _ = model(tokens, mask, seconds, torch.ones(1, 20, dtype=torch.bool))
    torch.testing.assert_close(out["body"][0, 0, 0], torch.tensor([1., 0., 0., 0., 1., 0.]))
    assert out["face"].abs().max() == 0


def test_training_reduces_loss_and_learns_duration():
    torch.manual_seed(0)
    states = [state(i, 24 + 12 * i) for i in range(3)]
    texts = ["the dog runs", "a cat sleeps on the mat", "birds fly south in winter now"]
    model = TextToMotion(config(states))
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(texts)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    first, _, _ = training_loss(model, tokens, mask, batch)
    for _ in range(150):
        loss, terms, _ = training_loss(model, tokens, mask, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < .3 * float(first.detach())
    durations = utterance_duration(batch)
    np.testing.assert_allclose(durations.numpy(), [24 / 24, 36 / 24, 48 / 24], atol=1e-9)
    generated = generate(model, texts, fps=24.)
    for g, d in zip(generated, durations):
        g.validate()
        assert abs(len(g.timestamps) / 24. - float(d)) < .25
        assert not g.channels["body"].observed.any() and not g.channels["body"].confidence.any()
        assert not g.channels["left_gaze"].valid.any()  # untrained channels are not emitted
