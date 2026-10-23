"""Strict prior review helpers: the same R-precision groups, interval separation, the gate selection parsed."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from strict_prior_review import bootstrap_mean, per_clip_hits, selected_setting, separated  # noqa: E402
from signtranslator.generation.text_motion_retrieval import r_precision  # noqa: E402


def test_per_clip_hits_average_to_r_precision():
    torch.manual_seed(0)
    zm = torch.nn.functional.normalize(torch.randn(96, 8), dim=-1)
    zt = torch.nn.functional.normalize(zm + .8 * torch.randn(96, 8), dim=-1)
    hits = per_clip_hits(zm, zt)
    assert len(hits) == 96 and abs(hits.mean() - r_precision(zm, zt, group=32)["r_at_3"]) < 1e-9


def test_intervals_and_separation():
    lo, hi = bootstrap_mean(np.r_[np.ones(60), np.zeros(40)])
    assert lo < .6 < hi
    assert separated([.3, .4], [.1, .2]) and not separated([.15, .4], [.1, .2])


def test_selected_setting_reads_core_and_sweep_runs():
    core = selected_setting(dict(selected=dict(run="p-pt-hs0.2", setting="motion_tokens_v2 guidance 2")))
    assert core["generated_run"] == "p-pt" and core["head_smoothing"] == .2 and core["tag"] == "guidance-2"
    assert core["part_temperature"].endswith("face=1.0")
    sweep = selected_setting(dict(selected=dict(run="p-sweep-h1.4-f0.8", setting="motion_tokens_v2 guidance 1.5")))
    assert sweep["generated_run"] == "p-sweep-h1.4-f0.8" and sweep["tag"] == "guidance-1.5"
    assert sweep["part_temperature"] == "body=1.4,left_hand=1.4,right_hand=1.4,face=0.8"


def test_selected_setting_reads_round3_variant_names():
    import strict_prior_review as spr
    sel = spr.selected_setting(dict(selected=dict(run="P-sweep-h1.2-f0.8-hs0.2-hd0.6-fc", setting="motion_tokens_v2 guidance 1.5")))
    assert (sel["head_smoothing"], sel["head_damping"], sel["face_calibrated"]) == (0.2, 0.6, True)
    assert sel["generated_run"] == "P-sweep-h1.2-f0.8" and sel["guidance"] == 1.5 and "face=0.8" in sel["part_temperature"]
    plain = spr.selected_setting(dict(selected=dict(run="P-sweep-h1.0-f1.0-hs0.1", setting="motion_tokens_v2 guidance 2")))
    assert (plain["head_smoothing"], plain["head_damping"], plain["face_calibrated"]) == (0.1, 1., False)
