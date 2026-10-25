"""Tokenizer fidelity acceptance: every ratio in [0.85, 1.15] and the R2b round trip above blank, or a rejection."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("fidelity", ROOT / "scripts/tokenizer_fidelity_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def separated(a, b):
    return a[0] > b[1]


def test_judge_passes_only_with_every_ratio_in_band_and_r2b_above_blank():
    f = load()
    bleu_ok = dict(round_trip=dict(ci95=[2.5, 3.5]), blank=dict(ci95=[1.0, 2.0]))
    bleu_bad = dict(round_trip=dict(ci95=[1.9, 3.5]), blank=dict(ci95=[1.0, 2.0]))
    good = {"head_rotation_speed_ratio": 1.1, "right_hand_variation_ratio": 0.9}
    assert f.judge(good, bleu_ok, separated)["pass"]
    v = f.judge({**good, "head_rotation_speed_ratio": 1.58}, bleu_ok, separated)          # round 3a's tokenizer
    assert not v["pass"] and v["out_of_band"] == {"head_rotation_speed_ratio": 1.58}
    assert not f.judge(good, bleu_bad, separated)["pass"]
    assert f.judge({"x": 0.85, "y": 1.15}, bleu_ok, separated)["pass"]                     # the band is closed
