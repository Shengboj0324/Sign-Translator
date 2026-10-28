"""Extraction job: transient failures are retried per clip, and resumed clips keep their quality counts."""
from pathlib import Path
import sys
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extract_video_states as job  # noqa: E402


def test_a_transient_failure_is_retried_and_a_persistent_one_is_an_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def flaky(j):
        calls.append(j)
        if len(calls) < 3:
            raise ConnectionError("SSL: unexpected EOF")
        return dict(sample_id="a", status="converted")

    monkeypatch.setattr(job, "_process_once", flaky)
    assert job._process(("v/a.mp4",))["status"] == "converted" and len(calls) == 3
    monkeypatch.setattr(job, "_process_once", lambda j: (_ for _ in ()).throw(TimeoutError("read timed out")))
    result = job._process(("v/b-BOOK.mp4",))
    assert result == dict(sample_id="b-BOOK", status="error", reason="TimeoutError: read timed out")


def test_quality_of_a_stored_state():
    t = 6
    def channel(valid, labels):
        return types.SimpleNamespace(valid=np.asarray(valid, bool), labels=labels)
    body = np.zeros((t, 3), bool); body[1:4, 1] = True; body[:, 2] = True
    state = types.SimpleNamespace(timestamps=np.arange(t) / 30., channels=dict(
        body=channel(body, ("Chest", "LeftForearm", "RightForearm")),
        left_hand=channel(np.zeros((t, 2)), ("a", "b")),
        right_hand=channel(np.c_[np.r_[1, 1, 0, 0, 0, 0], np.r_[0, 1, 1, 0, 0, 0]], ("a", "b")),
        head_rotation=channel(np.ones((t, 1)), ("Head",))))
    assert job.state_quality(state) == dict(frames=6, face_frames=6, left_hand_frames=0, right_hand_frames=3,
                                            arm_frames={"LeftForearm": 3, "RightForearm": 6}, quality_from_state=True)


def test_upright_root_without_upright_chest_is_refused_before_any_work():
    import pytest
    with pytest.raises(SystemExit, match="needs --upright-chest"):
        job.main(["--zip-key", "z", "--zip-sha256", "0" * 64, "--prefix", "p", "--models-dir", "m", "--upright-root"])
