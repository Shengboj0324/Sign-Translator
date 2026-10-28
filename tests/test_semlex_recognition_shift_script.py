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
