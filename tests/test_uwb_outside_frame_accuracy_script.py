"""Outside-frame accuracy of the UWB lifter's visibility rules: per side, bone and elbow bin; survival against
MediaPipe's, UWB-weighted; legacy Holistic inputs preferred; signer 9 refused."""
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
SIZE = (640, 480)


def write_crop(directory: Path, name: str, shape: str, fields: dict):
    meta = dict(name=name, video=f"{VIDEO}.mp4", shape=shape, size=list(SIZE), names=[], box=[0, 0, *SIZE])
    np.savez(directory / f"{name}.npz", meta=np.array(json.dumps(meta)), **{f"ref_{k}": v for k, v in fields.items()})


def make_data(tmp_path: Path):
    array, body, hands = semlex_array(SIZE)
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
    write_crop(crops, f"{VIDEO}__4x3", "4:3", fields)
    write_crop(crops, f"{VIDEO}__full", "full", fields)
    write_crop(crops, "other_1-3-rgb_front__4x3", "4:3", fields)        # signer 3: not read by default
    clip = ClipLandmarks(**fields, blendshape_names=(), roi=True)
    truth, _ = canonical_from_landmarks(clip, sample_id="tv", source_id="tv", source_sha256="0" * 64, clock_id="tv",
                                        inference_method="tv", points=dict(world=body, visible=np.ones((T, 33), bool),
                                        hands={s: (h - h[:, :1], np.ones(T, bool)) for s, h in hands.items()}))
    tv = tmp_path / "tv"
    tv.mkdir()
    digest = truth.save(tv / "abcDEF_1-1.npz")
    (tv / "index.json").write_text(json.dumps({"abcDEF_1-1": dict(sha256=digest)}))
    return crops, tv, TrueWorldXY(body, hands), array


def args_for(crops, tv, **kw):
    return argparse.Namespace(crop_dirs=[str(crops)], two_view_dir=str(tv), regressors=None, regressors_sha256=None,
                              rules=kw.get("rules", "inside,margin:0.5,learned"), signers=kw.get("signers", "1,2"),
                              legacy_dir=kw.get("legacy_dir"), require_legacy=kw.get("require_legacy", False),
                              uwb_weights=kw.get("uwb_weights"), bootstrap=20, min_bin_frames=kw.get("min_bin_frames", 1))


def test_rules_per_side_bone_and_bin_against_truth_and_mediapipe(tmp_path):
    crops, tv, stub, _ = make_data(tmp_path)
    report = job.analyse(args_for(crops, tv), regressors=stub)
    assert report["videos"] == 1 and report["crops"] == 1 and report["rows"] == 2 * T
    assert report["inputs"] == {job.TASKS_INPUT: 1}
    mp, inside, margin = (report["by_decision"][k] for k in ("mediapipe", "inside", "margin:0.5"))
    for side in ("Left", "Right"):
        assert mp[side]["all"]["survival"] == .5 and mp[side]["all"]["frames"] == T
        # the synthetic elbows lie below the 640x480 crop: inside-image keeps no forearm, the margin rule all
        assert inside[side]["all"]["survival"] == 0. and margin[side]["all"]["survival"] == 1.
        assert mp[side]["inside"]["frames"] == 0
        assert sum(margin[side][b]["frames"] for b, *_ in job.BINS) == T
        for bone in job.BONES:
            kept = margin[side]["all"]["kept"][bone]
            assert kept["n"] == T and kept["median"] < 1e-3 and kept["p90"] < 1e-3            # the stub is the truth
            assert kept["median_ci_upper"] is None                                          # one video: no interval
            assert inside[side]["all"]["kept"][bone] == dict(n=0)
    assert job.DEFAULT_RULES.split(",") == ["learned", "inside", "margin:0.1", "margin:0.25"]


def test_uwb_weighted_survival_uses_the_uwb_elbow_distribution(tmp_path):
    crops, tv, stub, array = make_data(tmp_path)
    gate_report = tmp_path / "gate.json"
    shares = {name: 0. for name, *_ in job.BINS}
    first = job.bin_of(float(job.outside_distance(array[0, 478 + 13, :2], 4 / 3)))
    shares[first] = 1.
    gate_report.write_text(json.dumps(dict(summary=dict(uwb_elbow_distance={
        "Left": dict(frames=10, share=shares), "Right": dict(frames=10, share=shares)}))))
    report = job.analyse(args_for(crops, tv, uwb_weights=str(gate_report)), regressors=stub)
    assert report["uwb_weights"]["Left"][first] == 1.
    mp_left = report["by_decision"]["mediapipe"]["Left"][first]["survival"]
    assert report["uwb_weighted_survival"]["mediapipe"]["Left"] == dict(survival=mp_left, covered=1., bins=[first])
    assert report["uwb_weighted_survival"]["margin:0.5"]["Left"]["survival"] == 1.
    assert report["uwb_weighted_survival_vs_mediapipe"]["margin:0.5"]["Left"] == round(1. / mp_left, 4)
    thin = job.analyse(args_for(crops, tv, uwb_weights=str(gate_report), min_bin_frames=1000), regressors=stub)
    assert thin["uwb_weighted_survival"]["mediapipe"]["Left"] == dict(survival=None, covered=0, bins=[])


def test_legacy_holistic_inputs_are_preferred_and_can_be_required(tmp_path):
    crops, tv, stub, array = make_data(tmp_path)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    holistic = array.copy()
    holistic[:, 511:532] = np.nan                                    # legacy found no left hand
    np.savez(legacy / f"{VIDEO}__4x3.npz", holistic=holistic.astype(np.float16), size=np.array(SIZE),
             times=np.arange(T) / 30., pose_visibility=np.zeros((T, 33), np.float16))
    report = job.analyse(args_for(crops, tv, legacy_dir=str(legacy)), regressors=stub)
    assert report["inputs"] == {job.LEGACY_INPUT: 1}
    with pytest.raises(SystemExit, match="require-legacy"):
        job.analyse(args_for(crops, tv, legacy_dir=str(tmp_path / "nothing"), require_legacy=True), regressors=stub)
    np.savez(legacy / f"{VIDEO}__4x3.npz", holistic=holistic[:-1].astype(np.float16), size=np.array(SIZE),
             times=np.arange(T - 1) / 30., pose_visibility=np.zeros((T - 1, 33), np.float16))
    assert job.analyse(args_for(crops, tv, legacy_dir=str(legacy)), regressors=stub)["inputs"] == {job.TASKS_INPUT: 1}


def test_other_signers_are_opt_in_and_signer_9_is_refused(tmp_path):
    crops, tv, stub, _ = make_data(tmp_path)
    other = job.analyse(args_for(crops, tv, signers="3"), regressors=stub)   # read, but no full-frame twin: skipped
    assert other["crops"] == 0 and other["skipped"] == ["other_1-3-rgb_front__4x3"] and other["rows"] == 0
    with pytest.raises(SystemExit, match="signer 9"):
        job.analyse(args_for(crops, tv, signers="1,9"), regressors=stub)
    (crops / "x_1-9-rgb_front__4x3.npz").write_bytes(b"")
    with pytest.raises(SystemExit, match="signer 9"):
        job.analyse(args_for(crops, tv), regressors=stub)
