"""Legacy-Holistic helpers: signer 9 never enters, distribution statistics read the estimator's traits."""
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import compare_holistic_distributions as compare  # noqa: E402
import prepare_legacy_holistic_jobs as prepare  # noqa: E402
import train_world_regressors as train  # noqa: E402
from semlex_synthetic import fake_resources, semlex_array  # noqa: E402


def test_signer_nine_is_excluded_by_name():
    assert prepare.excluded("abc_3-9-rgb_front.mp4") and prepare.excluded("x_1-9-rgb_front__4x3.npz")
    assert not prepare.excluded("abc_3-19-rgb_front.mp4") and not prepare.excluded("abc_3-1-rgb_front.mp4")


def test_training_refuses_signer_nine(tmp_path):
    np.savez(tmp_path / "v_1-9-rgb_front__4x3.npz", signer=np.array("h2s-9"))
    with pytest.raises(SystemExit, match="signer 9"):
        train.load_pairs(tmp_path)
    (tmp_path / "v_1-9-rgb_front__4x3.npz").unlink()
    np.savez(tmp_path / "v_1-4-rgb_front__4x3.npz", signer=np.array("h2s-9"), size=np.array([640, 480]))
    with pytest.raises(SystemExit, match="excluded"):
        train.load_pairs(tmp_path)


def test_distribution_statistics_see_a_single_iris_depth_and_partial_blocks():
    resources = fake_resources()
    array, *_ = semlex_array((640, 480), resources=resources)
    flat = array.copy()
    flat[:, 468:473, 2] = flat[:, 468:469, 2]
    flat[:, 473:478, 2] = flat[:, 473:474, 2]
    s = compare.stats([flat], [4 / 3], resources)
    assert s["iris_z_spread_p10_p50_p90"][2] == 0. and s["partial_blocks"] == 0
    assert abs(s["aspect_raw_bias"][1]) < 1e-3
    broken = flat.copy()
    broken[0, 511:520] = np.nan
    assert compare.stats([broken], [4 / 3], resources)["partial_blocks"] == 1
