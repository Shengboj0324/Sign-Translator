"""ASL Citizen pixel-side check: the active-hand image half per clip, the signer majority, the exclusion spec, the
signer-spread sample that never reads the test partition, and one clip measured through a stubbed hand landmarker
(MediaPipe is not needed)."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import asl_citizen_pixel_sides as job  # noqa: E402


def hand(x, y=.5, spread=.01):
    points = np.zeros((21, 2))
    points[:, 0], points[:, 1] = x + spread * np.arange(21) / 21, y
    return points


def test_the_more_moving_half_wins_and_too_little_evidence_is_undecided():
    frames = [[hand(.3 + .02 * t), hand(.7)] for t in range(10)]         # left half moves, right half still
    m = job.half_motion(frames)
    assert m["image_left"]["pairs"] == 9 and m["image_right"]["pairs"] == 9
    assert m["image_left"]["motion"] == pytest.approx(9 * .02) and m["image_right"]["motion"] == 0
    assert job.active_half(m) == "image_left"
    mirrored = [[hand(1 - .3 - .02 * t), hand(.3)] for t in range(10)]
    assert job.active_half(job.half_motion(mirrored)) == "image_right"
    sparse = [[hand(.3 + .1 * t)] if t % 2 == 0 else [] for t in range(10)]   # no consecutive detections
    assert job.half_motion(sparse)["image_left"]["pairs"] == 0 and job.active_half(job.half_motion(sparse)) is None
    still = [[hand(.3), hand(.7)] for _ in range(10)]
    assert job.active_half(job.half_motion(still)) is None              # a tie
    two_left = [[hand(.2 + .05 * t), hand(.4)] for t in range(6)]        # only the first hand of a half counts
    assert job.half_motion(two_left)["image_left"]["motion"] == pytest.approx(5 * .05)


def test_signer_majority_and_the_exclusion_spec():
    rows = [dict(signer="a" * 64, side="image_right"), dict(signer="a" * 64, side="image_right"),
            dict(signer="a" * 64, side="image_left"), dict(signer="b" * 64, side="image_left"),
            dict(signer="c" * 64, side="image_left"), dict(signer="c" * 64, side="image_right"),
            dict(signer="d" * 64, side=None)]
    sides = job.group_sides(rows)
    assert sides["a" * 64]["side"] == "image_right" and sides["b" * 64]["side"] == "image_left"
    assert sides["c" * 64]["side"] is None and sides["d" * 64] == dict(side=None, clips=1, image_left=0, image_right=0,
                                                                       undecided=1)
    assert sides["a" * 64]["clips"] == 3
    assert job.quantiles([v["clips"] for v in sides.values()]) == dict(min=1, median=1.5, max=3)
    assert job.quantiles([]) is None
    assert job.exclusion_spec(sides) == f"{job.CORPUS}:{'a' * 16}"
    assert job.exclusion_spec(job.group_sides(rows[3:5])) is None
    from signtranslator.generation.recognizer_corpora import parse_signer_specs, signer_matches
    spec = parse_signer_specs([job.exclusion_spec(sides)])
    assert signer_matches(dict(performer_sha256="a" * 64), spec[job.CORPUS]) == "a" * 16   # the recogniser reads it


def test_the_sample_is_spread_over_signers_and_never_reads_test():
    records = [dict(sign=f"{s}{i}", partition=p, performer_sha256=s * 64) for s in "abc" for i, p in
               enumerate(("train", "train", "val", "test", "test"))]
    picked = job.sample_records(dict(records=records), 6, seed=0)
    assert len(picked) == 6 and all(r["partition"] != "test" for r in picked)
    assert sorted(r["performer_sha256"][0] for r in picked) == ["a", "a", "b", "b", "c", "c"]
    assert job.sample_records(dict(records=records), 100, seed=0).__len__() == 9


def test_one_clip_is_measured_through_the_hand_landmarker(monkeypatch):
    import extract_video_states
    import signtranslator.data_engineering.video_landmarks as vl
    found = np.ones((8, 2), bool)
    found[:, 1] = False
    image = np.zeros((8, 2, 21, 2))
    for t in range(8):
        image[t, 0] = hand(.7 - .02 * t)                                # one hand, moving, on the image right
    clip = SimpleNamespace(hands_found=found, hands_image=image, times=np.arange(8) / 30.)
    monkeypatch.setitem(extract_video_states._WORKER, "archive", SimpleNamespace(read=lambda member: b"video-bytes"))
    monkeypatch.setitem(extract_video_states._WORKER, "models", {})
    monkeypatch.setattr(vl, "extract_clip", lambda video, models, roi, hand_scale: clip)
    row = job.measure(("s1", "ASL_Citizen/videos/s1.mp4", "a" * 64, "train"))
    assert row["status"] == "measured" and row["side"] == "image_right" and row["frames"] == 8
    assert row["image_right_pairs"] == 7 and row["image_left_pairs"] == 0

    def refuse(*a, **k):
        raise ValueError("no frames")
    monkeypatch.setattr(vl, "extract_clip", refuse)
    assert job.measure(("s2", "m.mp4", "a" * 64, "val"))["status"] == "refused"
