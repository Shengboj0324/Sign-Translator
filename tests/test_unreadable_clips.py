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
