"""Amendment 5's excursion gains: inverse variation ratios capped to [1, 1.5]; a gain restores a damped channel."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load():
    spec = importlib.util.spec_from_file_location("gains", ROOT / "scripts/fit_tokenizer_gains.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gains_are_inverse_variation_ratios_capped_and_restore_a_damped_channel():
    from signtranslator.generation.postprocess import damp_rotation_channel
    from tests.test_generation_postprocess import with_head, yaw
    g = load()
    assert g.fit({"body": .8, "left_hand": .5, "right_hand": 1.2}) == {"body": 1.25, "left_hand": 1.5, "right_hand": 1.}
    t = np.arange(96) / 24.
    real = with_head(.3 * np.sin(2 * np.pi * .5 * t))
    damped = damp_rotation_channel(real, "head_rotation", .8)                       # a 20% damped reconstruction
    restored = g.apply_gains(damped, {"head_rotation": 1.25})
    assert abs(np.std(yaw(damped)) / np.std(yaw(real)) - .8) < .02
    assert abs(np.std(yaw(restored)) / np.std(yaw(real)) - 1.) < .02                # the gain undoes the damping
    assert g.apply_gains(real, {}) is real and g.apply_gains(real, {"head_rotation": 1}) is real
