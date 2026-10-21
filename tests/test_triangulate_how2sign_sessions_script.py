"""Session script end to end: epipolar sync, calibration (alone or shared), gate, states."""
import hashlib
import io
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_how2sign_cameras import main as calibrate_main  # noqa: E402
from triangulate_how2sign_sessions import main  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402
from tests.test_calibrate_how2sign_cameras_script import write  # noqa: E402
from tests.test_de_two_view_states import session_clips  # noqa: E402
from tests.test_de_video_landmarks import clip as mp_clip, convert  # noqa: E402


def monocular_state(frames):
    base = mp_clip()
    tile = lambda a: np.repeat(a[:1], frames, axis=0)
    long_clip = replace(base, times=np.arange(frames) / 24., pose_world=tile(base.pose_world),
                        pose_visibility=tile(base.pose_visibility), pose_image=tile(base.pose_image),
                        pose_found=tile(base.pose_found), hands_world=tile(base.hands_world),
                        hands_image=tile(base.hands_image), hands_found=tile(base.hands_found),
                        blendshapes=tile(base.blendshapes), face_matrix=tile(base.face_matrix),
                        face_found=tile(base.face_found))
    state, _ = convert(long_clip)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "s.npz"
        state.save(path)
        return path.read_bytes()


def corpus(tmp_path):
    store = DirectoryStore(tmp_path)
    for video, seeds in (("vidA", (0, 1, 2)), ("vidB", (3, 4, 5))):
        for i, (_, front, side) in enumerate(session_clips(seeds=seeds)):
            write(store, "front", f"{video}_{i}-1-rgb_front", front)
            write(store, "side", f"{video}_{i}-1-rgb_side", side)
            store.put(f"mono/states/{video}_{i}-1-rgb_front.npz", monocular_state(len(front.times)))
    return store


def test_sessions_are_synchronised_calibrated_gated_and_converted(tmp_path):
    store = corpus(tmp_path)
    assert main(["--store", f"dir:{tmp_path}", "--front-keypoints", "front", "--side-keypoints", "side",
                 "--monocular", "mono", "--prefix", "out"]) == 0
    report = json.loads(store.get("out/reports/shard-000-of-001.json"))
    assert set(report["gates"]) == {"vidA-1", "vidB-1"}
    for gate in report["gates"].values():
        assert gate["passed"], gate["reasons"]
        assert gate["lag_frames"] == 2 and all(c["passed"] for c in gate["sync"])
        scale = gate["metric_scale"]
        assert scale["metres_per_unit"] > 0 and .05 < scale["relative_sd"] < .2 and len(scale["origin"]) == 3
    assert len(report["results"]) == 6
    for result in report["results"]:
        payload = store.get(result["state_key"])
        assert hashlib.sha256(payload).hexdigest() == result["state_sha256"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "s.npz"
            path.write_bytes(payload)
            state = MultichannelMotion.load(path, expected_sha256=result["state_sha256"])
        state.validate()
        assert state.channels["body"].coordinate_frame == "two-view-triangulated-camera1-y-up"
        root = state.channels["root_translation"]
        assert root.units == "m" and root.valid.mean() > .9 and "root_translation" in result["available_channels"]
        # the synthetic torso bobs ±1 cm vertically; metric scale is anthropometric, so within its prior
        assert .004 < float(np.std(root.values[root.valid[:, 0], 0, 1])) < .012


def test_shared_intrinsics_are_pinned_by_hash_and_used(tmp_path):
    store = corpus(tmp_path)
    assert calibrate_main(["--store", f"dir:{tmp_path}", "--front-keypoints", "front", "--side-keypoints", "side",
                           "--sessions", "2", "--output-key", "cams/intrinsics.json"]) == 0
    digest = hashlib.sha256(store.get("cams/intrinsics.json")).hexdigest()
    args = ["--store", f"dir:{tmp_path}", "--front-keypoints", "front", "--side-keypoints", "side",
            "--monocular", "mono", "--prefix", "shared", "--intrinsics-key", "cams/intrinsics.json"]
    try:
        main(args + ["--intrinsics-sha256", "0" * 64])
        raise AssertionError("a wrong intrinsics hash must be refused")
    except SystemExit as error:
        assert "differs" in str(error)
    assert main(args + ["--intrinsics-sha256", digest]) == 0
    report = json.loads(store.get("shared/reports/shard-000-of-001.json"))
    shared = json.loads(store.get("cams/intrinsics.json"))["intrinsics"]
    for gate in report["gates"].values():
        assert gate["shared_intrinsics"] == "cams/intrinsics.json"
        assert gate["focal"] == [shared["focal1"], shared["focal2"]]


def test_take5_sessions_use_recut_lateral_keypoints(tmp_path):
    store = DirectoryStore(tmp_path)
    for i, (_, front, side) in enumerate(session_clips(lag=0, seeds=(0, 1, 2))):
        write(store, "front", f"vidC_{i}-5-rgb_front", front)
        kept = len(side.times) * 3 // 4
        miscut = replace(side, times=side.times[:kept], points=side.points[-kept:], confidence=side.confidence[-kept:])
        write(store, "side", f"vidC_{i}-5-rgb_side", miscut)          # the release: ¾ of the frames, elsewhere
        write(store, "recut", f"vidC_{i}-5-rgb_side", side)           # re-cut from the raw recording
        store.put(f"mono/states/vidC_{i}-5-rgb_front.npz", monocular_state(len(front.times)))
    base = ["--store", f"dir:{tmp_path}", "--front-keypoints", "front", "--side-keypoints", "side", "--monocular", "mono"]
    assert main(base + ["--prefix", "released"]) == 0
    gate = json.loads(store.get("released/reports/shard-000-of-001.json"))["gates"]["vidC-5"]
    assert not gate["passed"] and "take-5" in gate["reasons"][0] and gate["side_source"] is None
    record = dict(accepted=True, offset=dict(offset_frames=9.35, sampson_px=4.2),
                  raw_side=dict(member_sha256="a" * 64), raw_front=dict(member_sha256="b" * 64))
    store.put("recut/sessions/vidC-5.json", json.dumps(record).encode())
    assert main(base + ["--prefix", "recovered", "--recut-side-keypoints", "recut"]) == 0
    report = json.loads(store.get("recovered/reports/shard-000-of-001.json"))
    gate = report["gates"]["vidC-5"]
    assert gate["passed"], gate["reasons"]
    assert gate["lag_frames"] == 0 and gate["time_scale"] == 1.
    source = gate["side_source"]
    assert source["offset_frames"] == 9.35 and source["session_key"] == "recut/sessions/vidC-5.json"
    assert source["session_sha256"] == hashlib.sha256(store.get("recut/sessions/vidC-5.json")).hexdigest()
    assert len(report["results"]) == 3
    # Re-cut keypoints whose session record is missing (an interrupted re-cut) refuse the session.
    (tmp_path / "recut/sessions/vidC-5.json").unlink()
    assert main(base + ["--prefix", "interrupted", "--recut-side-keypoints", "recut"]) == 0
    gate = json.loads(store.get("interrupted/reports/shard-000-of-001.json"))["gates"]["vidC-5"]
    assert not gate["passed"] and "session record" in gate["reasons"][0]


def test_a_clip_shorter_than_the_session_lag_is_dropped_not_fatal(monkeypatch):
    import triangulate_how2sign_sessions as tri
    clips = {f"s{i}": (f, sd) for i, (_, f, sd) in enumerate(session_clips(seeds=(0, 1, 2)))}
    monkeypatch.setattr(tri, "load_keypoints", lambda store, key, sentence: clips[sentence][0 if key == "f" else 1])
    real = tri.aligned
    def aligned(front, side, lag):
        if front is clips["s0"][0]:
            raise ValueError("views do not overlap after alignment")
        return real(front, side, lag)
    monkeypatch.setattr(tri, "aligned", aligned)
    out = tri.synchronised(None, {name: {"front": "f", "side": "s"} for name in clips})
    assert [p[0] for p in out["aligned"]] == ["s1", "s2"] and out["reason"] is None
    first = next(c for c in out["syncs"] if c.get("reason") == "views do not overlap after alignment")
    assert first["passed"] is False


def test_take_filter_splits_sessions_into_disjoint_report_sets(tmp_path):
    store = corpus(tmp_path)                                    # vidA-1 and vidB-1
    for i, (_, front, side) in enumerate(session_clips(lag=0, seeds=(0, 1, 2))):
        write(store, "front", f"vidC_{i}-5-rgb_front", front)
        write(store, "side", f"vidC_{i}-5-rgb_side", side)
        store.put(f"mono/states/vidC_{i}-5-rgb_front.npz", monocular_state(len(front.times)))
    base = ["--store", f"dir:{tmp_path}", "--front-keypoints", "front", "--side-keypoints", "side", "--monocular", "mono",
            "--prefix", "out"]
    assert main(base + ["--take-filter", "others"]) == 0
    assert main(base + ["--take-filter", "take5"]) == 0
    others = json.loads(store.get("out/reports/shard-000-of-001-others.json"))["gates"]
    take5 = json.loads(store.get("out/reports/shard-000-of-001-take5.json"))["gates"]
    assert set(others) == {"vidA-1", "vidB-1"} and set(take5) == {"vidC-5"}
    assert not store.exists("out/reports/shard-000-of-001.json")
