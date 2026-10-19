"""Clips without decodable video (How2Sign ships 81 empty 262-byte side clips) are refused, not fatal."""
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def streamless_mp4(path: Path) -> Path:
    ftyp = b"ftypisom" + struct.pack(">I", 512) + b"isomiso2"
    path.write_bytes(struct.pack(">I", 8 + len(ftyp)) + ftyp + struct.pack(">I", 8) + b"moov")
    return path


def test_extract_clip_refuses_a_file_without_video(tmp_path):
    from signtranslator.data_engineering.video_landmarks import extract_clip
    with pytest.raises(ValueError):
        extract_clip(streamless_mp4(tmp_path / "empty.mp4"), models={}, roi=True)


def test_sapiens_job_refuses_a_file_without_video(tmp_path):
    import torch
    from sapiens_keypoints_runpod import clip_keypoints
    with pytest.raises(ValueError):
        clip_keypoints(streamless_mp4(tmp_path / "empty.mp4"), model=None, device=torch.device("cpu"), batch=1)


def test_ranged_reads_retry_r2_throttling_and_raise_other_errors(monkeypatch):
    from botocore.exceptions import ClientError
    from extract_video_states import StoreRangeFile
    import time
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    calls = {"n": 0}

    class Body:
        def read(self):
            return b"abcd"

    class FakeS3:
        def head_object(self, **_):
            return {"ContentLength": 4}

        def get_object(self, **_):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ClientError({"Error": {"Code": "ServiceUnavailable"}}, "GetObject")
            return {"Body": Body()}

    stream = StoreRangeFile(FakeS3(), "bucket", "key")
    assert stream.read(4) == b"abcd" and calls["n"] == 3

    class Denied(FakeS3):
        def get_object(self, **_):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")

    with pytest.raises(ClientError):
        StoreRangeFile(Denied(), "bucket", "key").read(4)
