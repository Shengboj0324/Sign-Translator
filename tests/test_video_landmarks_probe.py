"""A broken MediaPipe runtime fails the extraction job at start, not as per-clip refusals."""
import sys
import types
from pathlib import Path

import pytest

from signtranslator.data_engineering.video_landmarks import probe_runtime

MODELS = {n: Path(n) for n in ("pose_landmarker_heavy.task", "hand_landmarker.task", "face_landmarker.task")}


def fake_mediapipe(error=None):
    created = []

    class Landmarker:
        @staticmethod
        def create_from_options(options):
            if error:
                raise error
            created.append(options)
            return types.SimpleNamespace(close=lambda: None)

    options = lambda **kw: kw  # noqa: E731
    vision = types.SimpleNamespace(PoseLandmarker=Landmarker, HandLandmarker=Landmarker, FaceLandmarker=Landmarker,
                                   PoseLandmarkerOptions=options, HandLandmarkerOptions=options,
                                   FaceLandmarkerOptions=options)
    return types.SimpleNamespace(tasks=types.SimpleNamespace(vision=vision, BaseOptions=lambda **kw: kw)), created


def test_probe_creates_each_landmarker(monkeypatch):
    mp, created = fake_mediapipe()
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    probe_runtime(MODELS)
    assert [c["base_options"]["model_asset_path"] for c in created] == list(MODELS)


def test_missing_system_library_is_a_runtime_error(monkeypatch):
    mp, _ = fake_mediapipe(OSError("libEGL.so.1: cannot open shared object file"))
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    with pytest.raises(RuntimeError, match="libEGL"):
        probe_runtime(MODELS)
