"""Recogniser validation helpers: in-context sign cutting keeps the clock and every channel's frames."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tests.test_pose_resample import state  # noqa: E402
from validate_sign_recognizer import cut  # noqa: E402


def test_cut_keeps_the_span_of_every_channel_and_its_clock():
    s = state(np.arange(30) / 30.)
    part = cut(s, 5, 12)
    assert len(part.timestamps) == 7 and part.timestamps[0] == s.timestamps[5]
    for name, c in part.channels.items():
        np.testing.assert_array_equal(c.values, s.channels[name].values[5:12])
        assert c.valid.shape[0] == c.confidence.shape[0] == c.observed.shape[0] == 7
