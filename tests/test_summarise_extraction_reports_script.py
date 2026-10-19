"""Extraction report summary: status counts, hours and frame-weighted detection rates."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarise_extraction_reports import summarise  # noqa: E402


def test_detection_rates_are_frame_weighted_over_converted_clips():
    report = dict(zip_key="z", results=[
        dict(status="converted", seconds=10., frames=100, pose_frames=100, face_frames=90, left_hand_frames=50,
             right_hand_frames=80),
        dict(status="converted", seconds=2., frames=20, pose_frames=20, face_frames=20, left_hand_frames=20,
             right_hand_frames=0),
        dict(status="refused", reason="the file has no video stream"),
    ])
    out = summarise([report])["z"]
    assert out["clips"] == {"converted": 2, "refused": 1} and out["frames"] == 120
    assert out["detection_rate"] == {"pose": 1.0, "face": round(110 / 120, 4), "left_hand": round(70 / 120, 4),
                                     "right_hand": round(80 / 120, 4)}
    assert out["hours"] == round(12 / 3600, 2)
