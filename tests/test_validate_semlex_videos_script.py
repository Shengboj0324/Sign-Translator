"""Sem-Lex video validation helpers: the mismatched control never scores a clip against its own gloss; paired
differences are per clip."""
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import validate_semlex_videos as job  # noqa: E402


def test_the_mismatched_control_is_a_derangement_of_the_labels():
    labels = [3, 3, 1, 2, 2, 2, 7, 9]
    other = job.derangement(labels, seed=0)
    assert sorted(other.tolist()) == sorted(labels) and not (other == np.array(labels)).any()
    assert np.array_equal(other, job.derangement(labels, seed=0))                # seeded
    with pytest.raises(RuntimeError):
        job.derangement([1, 1, 1, 2], seed=0)                                   # impossible: a majority label


def test_paired_differences_are_per_clip():
    a, b = np.array([0, 4, 5, 10]), np.array([5, 0, 0, 10])                    # ranks: 0 = top-1
    top5 = job.paired(a, b, 5)
    assert top5["difference"] == pytest.approx((2 - 2) / 4) and len(top5["ci"]) == 2
    assert job.paired(a, b, 1)["difference"] == pytest.approx((1 - 2) / 4)
