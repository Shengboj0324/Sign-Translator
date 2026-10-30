"""Outside-frame accuracy of the UWB lifter's visibility rules: bones binned by their farther end, the full-view
anchor, per side x bone x bin statistics, UWB-weighted survival, the declared selection, legacy Holistic inputs
preferred, signer 9 refused."""
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


def write_clip(directory: Path, name: str, shape: str, fields: dict):
    meta = dict(name=name, video=f"{VIDEO}.mp4", shape=shape, size=list(SIZE), names=[], box=[0, 0, *SIZE])
    np.savez(directory / f"{name}.npz", meta=np.array(json.dumps(meta)), **{f"ref_{k}": v for k, v in fields.items()})


def make_data(tmp_path: Path):
    array, body, hands = semlex_array(SIZE)
    times = np.arange(T) / 30.
    visibility = np.ones((T, 33))
    visibility[:4, [13, 14]] = .1                       # MediaPipe would drop both arms on the first 4 frames
    image = array[..., :2]

    def fields(img):
        return dict(times=times, pose_world=body @ Y_UP, pose_visibility=visibility, pose_image=img[:, 478:511],
                    pose_found=np.ones(T, bool), hands_world=np.zeros((T, 2, 21, 3)),
                    hands_image=np.stack((img[:, 511:532], img[:, 532:553]), 1), hands_found=np.ones((T, 2), bool),
                    blendshapes=np.zeros((T, 0)), face_matrix=np.tile(np.eye(4), (T, 1, 1)), face_found=np.zeros(T, bool))
    clips = tmp_path / "clips"
    clips.mkdir()
    write_clip(clips, f"{VIDEO}__4x3", "4:3", fields(image))                       # elbows below this crop
    write_clip(clips, f"{VIDEO}__full", "full", fields(.5 + (image - .5) * .35))   # the whole arm in view
    write_clip(clips, "other_1-3-rgb_front__4x3", "4:3", fields(image))            # signer 3: not read by default
    clip = ClipLandmarks(**fields(image), blendshape_names=(), roi=True)
    truth, _ = canonical_from_landmarks(clip, sample_id="tv", source_id="tv", source_sha256="0" * 64, clock_id="tv",
                                        inference_method="tv", points=dict(world=body, visible=np.ones((T, 33), bool),
                                        hands={s: (h - h[:, :1], np.ones(T, bool)) for s, h in hands.items()}))
    tv = tmp_path / "tv"
    tv.mkdir()
    digest = truth.save(tv / "abcDEF_1-1.npz")
    (tv / "index.json").write_text(json.dumps({"abcDEF_1-1": dict(sha256=digest)}))
    return clips, tv, TrueWorldXY(body, hands), array


def args_for(clips, tv, **kw):
    return argparse.Namespace(crop_dirs=[str(clips)], two_view_dir=str(tv), regressors=None, regressors_sha256=None,
                              rules=kw.get("rules", "inside,margin:0.5,learned"), signers=kw.get("signers", "1,2"),
                              legacy_dir=kw.get("legacy_dir"), require_legacy=kw.get("require_legacy", False),
                              uwb_weights=kw.get("uwb_weights"), bootstrap=20, min_bin_frames=kw.get("min_bin_frames", 1),
                              admit_share=.05)


def test_bones_are_binned_by_their_farther_end_and_the_anchor_uses_full_view_frames(tmp_path):
    clips, tv, stub, array = make_data(tmp_path)
    report = job.analyse(args_for(clips, tv), regressors=stub)
    assert report["videos"] == 1 and report["clips"] == dict(full=1, crops=1) and report["inputs"] == {job.TASKS_INPUT: 2}
    assert report["rows"] == 2 * 2 * 2 * T                               # 2 clips x 2 sides x 2 bones x T frames
    for side in ("Left", "Right"):
        for bone in job.BONES:
            a = report["anchor"][side][bone]
            assert a["n"] == T and a["videos"] == 1 and a["median"] < 1e-3          # the stub is the truth
        mp, inside, margin = (report["by_decision"][k][side] for k in ("mediapipe", "inside", "margin:0.5"))
        assert mp["Forearm"]["all"]["survival"] == .5 and mp["Forearm"]["all"]["frames"] == T   # crops only
        assert inside["Forearm"]["all"]["survival"] == 0. and margin["Forearm"]["all"]["survival"] == 1.
        assert sum(margin["Forearm"][b]["frames"] for b, *_ in job.BINS) == T
        admitted = [b for b, *_ in job.BINS if margin["Forearm"][b]["admitted"]]
        assert admitted and all(margin["Forearm"][b]["unmeasured"] for b in admitted)      # 8 frames, 1 video
    # the forearm bin follows max(elbow, wrist); the upper arm's max(shoulder, elbow)
    xy = array[0, 478:511, :2]
    d = {k: float(job.outside_distance(xy[k], 4 / 3)) for k in (11, 13, 15)}
    rows_expected = {"Forearm": job.bin_of(max(d[13], d[15])), "UpperArm": job.bin_of(max(d[11], d[13]))}
    assert job.DEFAULT_RULES.split(",") == ["learned", "inside", "margin:0.1", "margin:0.25"]
    for bone, name in rows_expected.items():
        assert report["by_decision"]["mediapipe"]["Left"][bone][name]["frames"] >= 1
    # unmeasured admitted bins fail margin:0.5 and learned; inside admits no bin here (it keeps none)
    verdicts = report["selection"]["verdicts"]
    assert not verdicts["margin:0.5"]["admissible"] and not verdicts["learned"]["admissible"]
    assert verdicts["inside"]["admissible"] and report["selection"]["chosen"] == "inside"


def cells(stats, survival=.5, admitted=("0-0.1",)):
    """A rule's per side x bone x bin summary with the same kept statistics everywhere."""
    out = {}
    for side in job.SIDES:
        out[side] = {}
        for bone in job.BONES:
            out[side][bone] = {"all": dict(frames=1000, survival=survival, admitted=False)}
            for name, *_ in job.BINS:
                out[side][bone][name] = dict(frames=500, videos=8, survival=survival if name in admitted else 0.,
                                             admitted=name in admitted, unmeasured=stats.get("n", 0) < 200,
                                             kept=stats)
    return out


def test_the_declared_selection_rule():
    anchors = {side: {bone: dict(n=900, median=10., p90=20.) for bone in job.BONES} for side in job.SIDES}
    good = dict(n=400, videos=8, median=11., p90=22., median_ci_upper=12.9, p90_ci_upper=24.9)
    bad_median = dict(good, median_ci_upper=13.1)
    bad_p90 = dict(good, p90_ci_upper=25.1)
    thin = dict(good, n=150)

    def weighted(v):
        return {side: {"Forearm": dict(survival=v, covered=.9)} for side in job.SIDES}
    by = {"inside": cells(good), "margin:0.1": cells(good), "margin:0.25": cells(bad_median), "learned": cells(good)}
    w = {"inside": weighted(.60), "margin:0.1": weighted(.70), "margin:0.25": weighted(.95), "learned": weighted(.71)}
    rules = ["learned", "inside", "margin:0.1", "margin:0.25"]
    sel = job.select_rule(by, anchors, w, rules)
    assert sel["verdicts"]["margin:0.25"]["admissible"] is False and "median CI upper" in sel["verdicts"]["margin:0.25"]["failures"][0]
    assert sel["chosen"] == "margin:0.1"                      # learned (0.71) is within 2 points: the conservative wins
    assert "condition" not in sel
    w["learned"] = weighted(.80)
    sel = job.select_rule(by, anchors, w, rules)
    assert sel["chosen"] == "learned" and "0.85" in sel["condition"]
    by["learned"] = cells(bad_p90)
    assert job.select_rule(by, anchors, w, rules)["chosen"] == "margin:0.1"
    by = {r: cells(thin) for r in rules}                      # unmeasured admitted bins fail
    sel = job.select_rule(by, anchors, w, rules)
    assert sel["chosen"] == "inside" and "fallback" in sel["reason"]
    assert "unmeasured" in sel["verdicts"]["learned"]["failures"][0]
    by = {r: cells(thin, admitted=()) for r in rules}         # a rule admitting no bin is vacuously admissible
    assert job.select_rule(by, anchors, w, rules)["chosen"] == "margin:0.25"                     # largest survival


def test_uwb_weighted_survival_uses_the_uwb_bone_distribution(tmp_path):
    clips, tv, stub, array = make_data(tmp_path)
    xy = array[0, 478:511, :2]
    first = job.bin_of(max(float(job.outside_distance(xy[13], 4 / 3)), float(job.outside_distance(xy[15], 4 / 3))))
    shares = {name: 0. for name, *_ in job.BINS}
    shares[first] = 1.
    gate_report = tmp_path / "gate.json"
    gate_report.write_text(json.dumps(dict(summary=dict(uwb_arm_distance={
        side: {bone: dict(frames=10, share=shares) for bone in job.BONES} for side in ("Left", "Right")}))))
    report = job.analyse(args_for(clips, tv, uwb_weights=str(gate_report)), regressors=stub)
    mp = report["by_decision"]["mediapipe"]["Left"]["Forearm"][first]["survival"]
    assert report["uwb_weighted_survival"]["mediapipe"]["Left"]["Forearm"] == dict(survival=mp, covered=1., bins=[first])
    assert report["uwb_weighted_survival"]["margin:0.5"]["Left"]["Forearm"]["survival"] == 1.
    assert report["selection"]["verdicts"]["margin:0.5"]["uwb_weighted_forearm_survival"] == 1.
    thin = job.analyse(args_for(clips, tv, uwb_weights=str(gate_report), min_bin_frames=1000), regressors=stub)
    assert thin["uwb_weighted_survival"]["mediapipe"]["Left"]["Forearm"] == dict(survival=None, covered=0, bins=[])


def test_legacy_holistic_inputs_are_preferred_and_can_be_required(tmp_path):
    clips, tv, stub, array = make_data(tmp_path)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    holistic = array.copy()
    holistic[:, 511:532] = np.nan                                    # legacy found no left hand
    for name in (f"{VIDEO}__4x3", f"{VIDEO}__full"):
        np.savez(legacy / f"{name}.npz", holistic=holistic.astype(np.float16), size=np.array(SIZE),
                 times=np.arange(T) / 30., pose_visibility=np.zeros((T, 33), np.float16))
    report = job.analyse(args_for(clips, tv, legacy_dir=str(legacy)), regressors=stub)
    assert report["inputs"] == {job.LEGACY_INPUT: 2}
    with pytest.raises(SystemExit, match="require-legacy"):
        job.analyse(args_for(clips, tv, legacy_dir=str(tmp_path / "nothing"), require_legacy=True), regressors=stub)
    np.savez(legacy / f"{VIDEO}__4x3.npz", holistic=holistic[:-1].astype(np.float16), size=np.array(SIZE),
             times=np.arange(T - 1) / 30., pose_visibility=np.zeros((T - 1, 33), np.float16))
    assert job.analyse(args_for(clips, tv, legacy_dir=str(legacy)), regressors=stub)["inputs"] == {
        job.LEGACY_INPUT: 1, job.TASKS_INPUT: 1}


def test_other_signers_are_opt_in_and_signer_9_is_refused(tmp_path):
    clips, tv, stub, _ = make_data(tmp_path)
    other = job.analyse(args_for(clips, tv, signers="3"), regressors=stub)   # read, but no full-frame twin: skipped
    assert other["skipped"] == ["other_1-3-rgb_front__4x3"] and other["rows"] == 0
    with pytest.raises(SystemExit, match="signer 9"):
        job.analyse(args_for(clips, tv, signers="1,9"), regressors=stub)
    (clips / "x_1-9-rgb_front__4x3.npz").write_bytes(b"")
    with pytest.raises(SystemExit, match="signer 9"):
        job.analyse(args_for(clips, tv), regressors=stub)
