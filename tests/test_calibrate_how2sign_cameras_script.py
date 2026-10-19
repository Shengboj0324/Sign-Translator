"""Shared-intrinsics calibration from stored per-clip keypoints, end to end."""
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_how2sign_cameras import main  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402
from tests.test_de_two_view_states import session_clips  # noqa: E402


def write(store, prefix, name, clip):
    buffer = io.BytesIO()
    np.savez(buffer, times=clip.times, points=clip.points.astype(np.float32),
             confidence=clip.confidence.astype(np.float16))
    store.put(f"{prefix}/keypoints/{name}.npz", buffer.getvalue())


def test_two_sessions_give_shared_intrinsics_near_the_truth(tmp_path):
    store = DirectoryStore(tmp_path)
    for video, seeds in (("vidA", (0, 1)), ("vidB", (2, 3))):
        for i, (_, front, side) in enumerate(session_clips(seeds=seeds)):
            write(store, "front", f"{video}_{i}-1-rgb_front", front)
            write(store, "side", f"{video}_{i}-1-rgb_side", side)
    assert main(["--store", f"dir:{tmp_path}", "--front-keypoints", "front", "--side-keypoints", "side",
                 "--sessions", "2", "--output-key", "cams/intrinsics.json"]) == 0
    record = json.loads(store.get("cams/intrinsics.json"))
    assert record["inputs"]["sessions_used"] == 2
    assert [s["lag_frames"] for s in record["sessions"]] == [2, 2]
    # These synthetic sessions are short (88 frames) with noisy 2-5 cm finger bones, so
    # each alone identifies the focal lengths to about ±10%; the pooled estimate must
    # land within that envelope. The statistical gain of pooling is tested in
    # test_de_two_view_geometry (four sessions with different camera poses).
    f1, f2 = record["intrinsics"]["focal1"], record["intrinsics"]["focal2"]
    assert abs(f1 / 1150 - 1) < .08 and abs(f2 / 1400 - 1) < .08
    assert record["intrinsics"]["distortion"] == [0.0, 0.0]          # off by default (DATA_QUALITY_DECISION §7)
    assert all(max(s["reprojection_px"]) < 1. for s in record["sessions"])
