"""Residual masked-motion model: exact baseline start, no leakage, real learning."""
from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from signtranslator.generation.masked_motion import (
    ROTATION_CHANNELS, ResidualMaskedMotion, build_example, evaluate, half_angle_loss,
    model_inputs, scored_cells, train_epoch,
)
from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d

FRAMES = 30


def state(seed=0, curvature=0.):
    rng = np.random.default_rng(seed)
    clock = np.cumsum(np.full(FRAMES, 1 / 24.))
    channels = {}
    for name, (kind, width, units) in CHANNELS.items():
        elements = 2 if name in ROTATION_CHANNELS and name != "head_rotation" else 1
        values = np.zeros((FRAMES, elements, width))
        if kind == "rotation_6d_columns":
            axis = rng.normal(size=(elements, 3))
            axis /= np.linalg.norm(axis, axis=1, keepdims=True)
            # Constant velocity plus a quadratic term that interpolation cannot follow.
            angle = 1.2 * clock[:, None] + curvature * (clock[:, None] - clock.mean()) ** 2
            aa = torch.from_numpy(axis[None] * angle[..., None])
            values = matrix_to_rotation_6d(axis_angle_to_matrix(aa)).numpy()
        elif kind == "unit_vector":
            values[:] = [1, 0, 0]
        observed = np.zeros((FRAMES, elements), dtype=bool)
        channels[name] = MotionChannel(values, np.ones((FRAMES, elements), dtype=bool), observed,
                                       np.ones((FRAMES, elements)),
                                       tuple(f"{name}-{i}" for i in range(elements)), "fixture", units,
                                       "fixture", "e" * 64, "fixture-v1", inference_method="fixture-solve")
    return MultichannelMotion(clock, "clock", f"s{seed}", channels)


def hidden_frames(frames=(10, 11, 12, 13)):
    mask = np.zeros(FRAMES, dtype=bool)
    mask[list(frames)] = True
    return {name: np.broadcast_to(mask[:, None], (FRAMES, 1 if name == "head_rotation" else 2)).copy()
            for name in ROTATION_CHANNELS}


def test_untrained_model_reproduces_the_interpolation_baseline_exactly():
    example = build_example(state(), hidden_frames())
    model = ResidualMaskedMotion(example.target.shape[1])
    prediction = model(*model_inputs(example))[0]
    torch.testing.assert_close(prediction, torch.from_numpy(example.interpolated).float(), rtol=0, atol=0)


def test_hidden_targets_never_influence_predictions():
    example = build_example(state(), hidden_frames())
    model = ResidualMaskedMotion(example.target.shape[1])
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * .05)
    before = model(*model_inputs(example))[0]
    poisoned = replace(example, target=np.where(example.hidden[..., None], 99., example.target))
    after = model(*model_inputs(poisoned))[0]
    assert torch.equal(before, after)


def test_half_angle_loss_matches_the_geodesic_identity():
    angles = torch.tensor([0., .3, 1., 2.5], dtype=torch.float64)
    target = matrix_to_rotation_6d(torch.eye(3, dtype=torch.float64).expand(4, 3, 3))
    prediction = matrix_to_rotation_6d(axis_angle_to_matrix(
        torch.tensor([[0., 0., 1.]], dtype=torch.float64) * angles[:, None]))
    loss = half_angle_loss(prediction, target, torch.ones(4, dtype=torch.bool))
    assert float(loss) == pytest.approx(float(((1 - torch.cos(angles)) / 2).mean()), abs=1e-12)


def test_training_learns_structure_interpolation_lacks():
    """With curved trajectories the residual model should beat interpolation."""
    examples = [build_example(state(seed, curvature=6.), hidden_frames()) for seed in range(6)]
    model = ResidualMaskedMotion(examples[0].target.shape[1], hidden_dim=32)
    before = evaluate(model, examples)
    assert before["model_mean_degrees"] == pytest.approx(before["interpolation_mean_degrees"], abs=1e-4)
    # Float32 rounding of an untrained model is a tie, not a win or a loss.
    assert before["signs_tied"] == 6 and before["signs_model_better"] == before["signs_model_worse"] == 0
    torch.manual_seed(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for epoch in range(60):
        train_epoch(model, examples, optimizer, seed=epoch)
    after = evaluate(model, examples)
    assert after["model_mean_degrees"] < .8 * after["interpolation_mean_degrees"]
    assert after["signs_model_better"] == 6
    assert all(np.isfinite(v) for v in (after["model_mean_degrees"],))


def test_scored_cells_exclude_unavailable_and_uncovered_cells():
    s = state()
    channel = s.channels["body"]
    valid = channel.valid.copy()
    valid[11, 0] = False
    s = replace(s, channels={**s.channels, "body": replace(
        channel, valid=valid, confidence=np.where(valid, 1., 0.),
        values=np.where(valid[..., None], channel.values, 0.))})
    example = build_example(s, hidden_frames())
    cells = scored_cells(example)
    assert not cells[11, 0] and cells[10, 0]
    edge = build_example(state(), hidden_frames((0, 1)))
    assert not scored_cells(edge)[:2].any()  # no left neighbour: uncovered, never scored
    assert math.isclose(1., 1.)
