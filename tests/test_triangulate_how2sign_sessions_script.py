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
