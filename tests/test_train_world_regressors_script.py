"""World-regressor training script: IK error metrics are zero on identical points and the fit learns a mapping."""
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import train_world_regressors as job  # noqa: E402
from semlex_synthetic import body_y_up, hand_y_up, rotation  # noqa: E402

from signtranslator.data_engineering import world_regressor as wr  # noqa: E402
from signtranslator.data_engineering.video_landmarks import Y_UP  # noqa: E402


def test_metrics_are_zero_for_identical_points_and_see_a_turned_finger():
    body = body_y_up(6)
    hand = hand_y_up(body[:, 16], 3)
    fingers, palm = job.finger_errors(hand, hand, "Right")
    assert max(fingers) < 1e-4 and max(palm) < 1e-4
    bent = hand.copy()
    bent[:, 8] = bent[:, 7] + (bent[:, 8] - bent[:, 7]) @ rotation((1, 0, 0), 40.).T   # index tip bent 40 deg
    fingers, palm = job.finger_errors(hand, bent, "Right")
    assert 39 < max(fingers) < 41 and max(palm) < 1e-4
    world = body @ Y_UP
    errors = job.arm_errors(world, world, np.ones((6, 33), bool))
    assert all(max(v) < 1e-4 for v in errors.values() if v)


def test_fit_learns_a_linear_map_with_directions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 12)).astype(np.float32)
    m = rng.normal(size=(12, 9)).astype(np.float32)
    y = x @ m
    model, mean, std = job.fit(x, y, bones=((0, 1), (1, 2)), points=3, epochs=30, batch=256, lr=3e-3)
    with torch.no_grad():
        pred = model(torch.from_numpy((x - mean) / std)).numpy()
    assert np.mean((pred - y) ** 2) < .05 * np.mean(y ** 2)
    assert isinstance(model, wr.MLP)
