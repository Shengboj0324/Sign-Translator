"""Round 3c grid measures: R4's length slices and R7's smoothness, computed as the review computes them."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load():
    spec = importlib.util.spec_from_file_location("grid", ROOT / "scripts/score_postprocessing_grid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_length_slices_split_at_sentence_length_terciles_and_smoothness_is_one_on_identical_clips():
    from tests.test_generation_text_to_motion import state
    grid = load()
    states = [state(i, 48 + 8 * i) for i in range(6)]
    records = [dict(text=" ".join(["w"] * n)) for n in (2, 3, 6, 7, 11, 12)]
    slices = grid.length_slices(states, states, records)
    assert set(slices) == {"short", "medium", "long"}
    assert all(abs(v - 1.) < 1e-9 for ratios in slices.values() for v in ratios.values())     # identical clips
    smooth = grid.smoothness(states, states)
    assert all(v is None or abs(v - 1.) < 1e-9 for v in smooth.values())
    result = grid.score(states, states, records, round3c=True, pooled=True)
    assert "length_slices" in result and "smoothness" in result
    assert "length_slices" not in grid.score(states, states, records, round3c=True, pooled=False)
    assert "smoothness" not in grid.score(states, states, records)                              # round 3b unchanged
