"""Recognition-shift experiment helpers: phonology normalisation, channel swaps, scores and intervals."""
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import semlex_recognition_shift as job  # noqa: E402
from semlex_synthetic import FakeBlendshapes, fake_resources, semlex_array  # noqa: E402

from signtranslator.data_engineering.semlex_pose import convert_array  # noqa: E402


def test_phonology_values_compare_by_content_not_spelling():
    assert job.norm("Fully Open") == job.norm("FullyOpen") == "fullyopen"
    assert job.norm("1.0") == job.norm("1") == "1"
    assert job.norm("") == job.norm("NA") == ""
    assert job.norm("flat_o") != job.norm("b")


def test_swap_takes_only_the_named_channels():
    array, *_ = semlex_array((640, 640))
    a, _ = convert_array(array, 8000 / 30, fake_resources(), video_id="a", source_sha256="a" * 64,
                         blendshape_model=FakeBlendshapes())
    b, _ = convert_array(array, 8000 / 30, fake_resources(), video_id="b", source_sha256="a" * 64,
                         blendshape_model=FakeBlendshapes(), hand_z_scale=3.)
    mixed = job.swap(a, b, job.HANDS)
    assert mixed.channels["left_hand"] is b.channels["left_hand"] and mixed.channels["body"] is a.channels["body"]
    with pytest.raises(ValueError):
        job.swap(a, convert_array(array[:5], 5000 / 30, fake_resources(), video_id="c", source_sha256="a" * 64,
                                  blendshape_model=FakeBlendshapes(), aspect=1.)[0], job.HANDS)


def test_scores_and_paired_differences():
    ranks = np.array([0, 3, 7, 0, 10])
    reference = np.array([0, 0, 0, 9, 9])
    s = job.score(ranks, ["a", "a", "b", "b", "c"], reference)
    assert s["top1"] == .4 and s["top5"] == .6
    assert s["top5_minus_ours"] == pytest.approx(0.)          # 3 of 5 both times
    assert s["top1_minus_ours"] == pytest.approx(-.2)
    low, high = s["top5_ci_clips"]
    assert 0 <= low <= .6 <= high <= 1
    assert job.bootstrap(np.ones(10)) == [1., 1.]


def test_flag_sets_change_only_their_own_joints():
    from dataclasses import replace
    from signtranslator.data_engineering.semlex_pose import holistic_clip
    array, *_ = semlex_array((640, 470), frames=8)            # elbows near the bottom edge: some frames outside
    clip, _ = holistic_clip(array, 8000 / 30, fake_resources(), image_size=(640, 470),
                            blendshape_model=FakeBlendshapes(), pose_z_scale=1., hand_z_scale=1.)
    visibility = np.ones((8, 33))
    visibility[:, 23:] = 0.                                   # MediaPipe-style: hips unseen, elbows "visible"
    clip = replace(clip, pose_visibility=visibility)
    states = job.flag_states(clip, "c")
    assert set(states) == set(job.FLAG_SETS)
    root = job.body_changes(states["a_upright_chest"], states["b_plus_upright_root"])
    assert root["Pelvis"]["added"] == 8 and all(root[j] == dict(reference=root[j]["reference"], removed=0, added=0,
                                                                changed=0) for j in ("LeftUpperArm", "RightForearm"))
    gate = job.body_changes(states["a_upright_chest"], states["c_plus_elbow_in_image"])
    outside = ~((array[:, 478 + 13, :2] >= 0) & (array[:, 478 + 13, :2] <= 1)).all(-1)
    assert outside.any() and gate["LeftUpperArm"]["removed"] == int(outside.sum()) and gate["Pelvis"]["added"] == 0
    both = job.body_changes(states["a_upright_chest"], states["d_plus_both"])
    assert both["LeftUpperArm"] == gate["LeftUpperArm"] and both["Pelvis"] == root["Pelvis"]
    inside = job.inside_image_visibility(clip).pose_visibility
    assert (inside[:, 23:25] == 0).all() and (inside[:, 13] == (~outside)).all()
    vis = job.body_changes(states["a_upright_chest"], states["e_inside_image_visibility"])
    assert vis["LeftUpperArm"]["removed"] == int(outside.sum())
