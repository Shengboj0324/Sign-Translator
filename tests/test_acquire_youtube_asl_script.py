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


def test_second_person_fraction_follows_the_slice():
    from dataclasses import replace
    from signtranslator.data_engineering.video_landmarks import ClipLandmarks
    T = 30
    other = np.zeros(T); other[15:20] = .8; other[20:] = .3     # a person half the signer's size or more in 5 frames
    clip = ClipLandmarks(times=np.arange(T) / 10., pose_world=np.zeros((T, 33, 3)), pose_visibility=np.zeros((T, 33)),
                         pose_image=np.zeros((T, 33, 2)), pose_found=np.ones(T, bool), hands_world=np.zeros((T, 2, 21, 3)),
                         hands_image=np.zeros((T, 2, 21, 2)), hands_found=np.zeros((T, 2), bool),
                         blendshapes=np.zeros((T, 52)), blendshape_names=tuple(f"b{i}" for i in range(52)),
                         face_matrix=np.zeros((T, 4, 4)), face_found=np.ones(T, bool), roi=True, other_person=other)
    assert acq.other_person_fraction(acq.slice_clip(clip, 0., 1.)) == 0.
    assert acq.other_person_fraction(acq.slice_clip(clip, 1., 2.)) == .5
    assert acq.other_person_fraction(acq.slice_clip(clip, 2., 3.)) == 0.   # a small background figure does not count
    assert replace(clip, other_person=None).other_person is None


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

    class AgeGate(DL):
        def extract_info(self, url, download):
            raise Exception("ERROR: [youtube] abcdefghijk: Sign in to confirm your age. This video may be inappropriate for some users.")
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=AgeGate))
    assert acq.download("abcdefghijk", tmp_path, max_height=720, allow_auto=False)["status"] == "age_restricted"

    class NotAvailable(DL):
        def extract_info(self, url, download):
            raise Exception("ERROR: [youtube] abcdefghijk: This video is not available")
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=NotAvailable))
    assert acq.download("abcdefghijk", tmp_path, max_height=720, allow_auto=False)["status"] == "removed"


def test_sign_language_caption_tracks_count_as_english_captions():
    assert acq.is_caption_lang("ase") and acq.is_caption_lang("sgn-US") and acq.is_caption_lang("en-CA")
    assert not acq.is_caption_lang("es") and "ase" in acq.CAPTION_LANGS


def test_the_polite_gate_serialises_downloads_and_spreads_them(tmp_path):
    import threading
    import time
    gate = acq.PoliteGate(str(tmp_path), interval=.2, backoff=.3)
    stamps = []

    def fetch():
        with gate:
            stamps.append(time.time())
    threads = [threading.Thread(target=fetch) for _ in range(3)]
    [t.start() for t in threads]; [t.join() for t in threads]
    stamps.sort()
    assert all(b - a >= .19 for a, b in zip(stamps, stamps[1:]))
    gate.mark_blocked()
    started = time.time(); gate.wait_blocked()
    assert time.time() - started >= .25


def test_home_staging_round_trip_and_deletion(monkeypatch, tmp_path):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "store")
    args = types.SimpleNamespace(prefix="yt", max_height=720, allow_auto_captions=False)

    def fake_download(video_id, directory, *, max_height, allow_auto):
        if video_id == "gonegonegon":
            return dict(status="removed", reason="gone")
        path = directory / f"{video_id}.mp4"; path.write_bytes(b"video-bytes-" + video_id.encode())
        return dict(status="downloaded", video=path, captions="WEBVTT\n", kind="manual", meta=dict(fps=30))
    monkeypatch.setattr(acq, "download", fake_download)
    assert acq.stage_download("gonegonegon", args, store)["status"] == "removed"
    assert acq.stage_download("abcdefghijk", args, store)["status"] == "staged"
    assert store.exists("yt/staged/abcdefghijk.json") and store.exists("yt/staged/abcdefghijk.mp4")
    got = acq.unstage("abcdefghijk", args, store, tmp_path)
    assert got["video"].read_bytes() == b"video-bytes-abcdefghijk" and got["kind"] == "manual"
    (store.root / "yt/staged/abcdefghijk.mp4").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        acq.unstage("abcdefghijk", args, store, tmp_path)
    acq.drop_staged("abcdefghijk", args, store, "yt/staged/abcdefghijk.mp4")
    assert store.list("yt/staged/") == []


def test_claims_are_exclusive_and_stale_ones_are_taken_over(tmp_path):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "store")
    args = types.SimpleNamespace(prefix="yt")
    assert acq.claim("abcdefghijk", args, store, now=1000.)
    assert not acq.claim("abcdefghijk", args, store, now=1000. + 60)          # held by another extractor
    assert acq.claim("abcdefghijk", args, store, now=1000. + acq.CLAIM_TTL + 1)   # its extractor vanished
    acq.release("abcdefghijk", args, store)
    assert acq.claim("abcdefghijk", args, store)
    acq.drop_staged("abcdefghijk", args, store, "yt/staged/abcdefghijk.mp4")
    assert store.list("yt/staged/") == []
