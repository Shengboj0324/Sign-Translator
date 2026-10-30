"""Outside-frame accuracy of the UWB lifter's visibility rules: survival against MediaPipe's, kept/dropped accuracy,
signer 9 refused."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import uwb_outside_frame_accuracy as job  # noqa: E402
from semlex_synthetic import T, semlex_array  # noqa: E402
from uwb_synthetic import TrueWorldXY  # noqa: E402

from signtranslator.data_engineering.video_landmarks import ClipLandmarks, Y_UP, canonical_from_landmarks  # noqa: E402

VIDEO = "abcDEF_1-1-rgb_front"


def write_crop(directory: Path, name: str, shape: str, size, fields: dict):
    meta = dict(name=name, video=f"{VIDEO}.mp4", shape=shape, size=list(size), names=[])
    np.savez(directory / f"{name}.npz", meta=np.array(json.dumps(meta)), **{f"ref_{k}": v for k, v in fields.items()})


def make_data(tmp_path: Path):
    size = (640, 480)
    array, body, hands = semlex_array(size)
    times = np.arange(T) / 30.
    visibility = np.ones((T, 33))
    visibility[:4, [13, 14]] = .1                       # MediaPipe would drop both forearms on the first 4 frames
    fields = dict(times=times, pose_world=body @ Y_UP, pose_visibility=visibility, pose_image=array[:, 478:511, :2],
                  pose_found=np.ones(T, bool), hands_world=np.zeros((T, 2, 21, 3)),
                  hands_image=np.stack((array[:, 511:532, :2], array[:, 532:553, :2]), 1),
                  hands_found=np.ones((T, 2), bool), blendshapes=np.zeros((T, 0)),
                  face_matrix=np.tile(np.eye(4), (T, 1, 1)), face_found=np.zeros(T, bool))
    crops = tmp_path / "crops"
    crops.mkdir()
    write_crop(crops, f"{VIDEO}__4x3", "4:3", size, fields)
    write_crop(crops, f"{VIDEO}__full", "full", size, fields)
    write_crop(crops, "other_1-3-rgb_front__4x3", "4:3", size, fields)        # signer 3: not read by default
    clip = ClipLandmarks(**fields, blendshape_names=(), roi=True)
    truth, _ = canonical_from_landmarks(clip, sample_id="tv", source_id="tv", source_sha256="0" * 64, clock_id="tv",
                                        inference_method="tv", points=dict(world=body, visible=np.ones((T, 33), bool),
                                        hands={s: (h - h[:, :1], np.ones(T, bool)) for s, h in hands.items()}))
    tv = tmp_path / "tv"
    tv.mkdir()
    digest = truth.save(tv / "abcDEF_1-1.npz")
    (tv / "index.json").write_text(json.dumps({"abcDEF_1-1": dict(sha256=digest)}))
    return crops, tv, TrueWorldXY(body, hands)


def args_for(crops, tv, **kw):
    return argparse.Namespace(crop_dirs=[str(crops)], two_view_dir=str(tv), regressors=None, regressors_sha256=None,
                              rules=kw.get("rules", "inside,margin:0.5,learned"), signers=kw.get("signers", "1,2"))


def test_rules_are_scored_against_two_view_truth_and_mediapipe_survival(tmp_path):
    crops, tv, stub = make_data(tmp_path)
    report = job.analyse(args_for(crops, tv), regressors=stub)
    assert report["videos"] == 1 and report["crops"] == 1 and report["rows"] == 2 * T and report["signers"] == ["1", "2"]
    mp = report["mediapipe"]["all"]
    assert mp["survival"] == .5 and mp["kept"] == T
    inside, margin = report["by_rule"]["inside"], report["by_rule"]["margin:0.5"]
    # the synthetic elbows lie below the 640x480 crop: inside-image keeps no forearm, the margin rule keeps them all
    assert report["by_rule"]["inside"]["elbow_outside"]["frames"] == 2 * T
    assert inside["all"]["survival"] == 0. and margin["all"]["survival"] == 1.
    assert margin["all"]["survival_vs_mediapipe"] == 2.
    for part in (margin["all"]["kept_errors"], inside["all"]["dropped_errors"]):
        assert part["Forearm"]["median"] < 1e-3 and part["UpperArm"]["median"] < 1e-3        # the stub is the truth
    assert report["by_rule"]["learned"]["all"]["survival"] == 1.                          # the stub calls all visible
    assert sum(v["frames"] for v in margin["outside_by_distance"].values()) == 2 * T


def test_other_signers_are_opt_in_and_signer_9_is_refused(tmp_path):
    crops, tv, stub = make_data(tmp_path)
    other = job.analyse(args_for(crops, tv, signers="3"), regressors=stub)   # read, but no full-frame twin: skipped
    assert other["crops"] == 0 and other["skipped"] == ["other_1-3-rgb_front__4x3"] and other["rows"] == 0
    with pytest.raises(SystemExit, match="signer 9"):
        job.analyse(args_for(crops, tv, signers="1,9"), regressors=stub)
    (crops / "x_1-9-rgb_front__4x3.npz").write_bytes(b"")
    with pytest.raises(SystemExit, match="signer 9"):
        job.analyse(args_for(crops, tv), regressors=stub)
