"""YouTube ASL acquisition: caption parsing, landmark slicing, and the no-circumvention stop."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acquire_youtube_asl as acq  # noqa: E402

VTT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.500
Hello, <c>my</c> name is Ana.

00:00:03.500 --> 00:00:04.000 align:start position:0%
Hello, my name is Ana.

00:01:02.250 --> 00:01:05.000
Today we talk about
sign language.
"""


def test_vtt_cues_are_parsed_and_rolling_repeats_merged():
    cues = acq.parse_vtt(VTT)
    assert cues == [dict(start=1., end=4., text="Hello, my name is Ana."),
                    dict(start=62.25, end=65., text="Today we talk about sign language.")]


def test_slice_keeps_every_per_frame_array_in_the_window():
    from signtranslator.data_engineering.video_landmarks import ClipLandmarks
    T = 30
    clip = ClipLandmarks(times=np.arange(T) / 10., pose_world=np.zeros((T, 33, 3)), pose_visibility=np.zeros((T, 33)),
                         pose_image=np.zeros((T, 33, 2)), pose_found=np.ones(T, bool), hands_world=np.zeros((T, 2, 21, 3)),
                         hands_image=np.zeros((T, 2, 21, 2)), hands_found=np.zeros((T, 2), bool),
                         blendshapes=np.zeros((T, 52)), blendshape_names=tuple(f"b{i}" for i in range(52)),
                         face_matrix=np.zeros((T, 4, 4)), face_found=np.ones(T, bool), roi=True)
    part = acq.slice_clip(clip, 1.0, 2.0)
    assert len(part.times) == 10 and part.times[0] == 1.0 and part.hands_world.shape[0] == 10
    assert part.blendshape_names == clip.blendshape_names and part.roi
    assert acq.slice_clip(clip, 9., 10.) is None


def test_a_bot_check_stops_instead_of_retrying(monkeypatch, tmp_path):
    class DL:
        def __init__(self, opts): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download):
            raise Exception("ERROR: Sign in to confirm you're not a bot")
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=DL))
    with pytest.raises(acq.Blocked):
        acq.download("abcdefghijk", tmp_path, max_height=720, allow_auto=False)

    class Gone(DL):
        def extract_info(self, url, download):
            raise Exception("ERROR: Video unavailable. This video has been removed by the uploader")
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=Gone))
    assert acq.download("abcdefghijk", tmp_path, max_height=720, allow_auto=False)["status"] == "removed"
