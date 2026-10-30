"""z-free world regressors: the features read only x and y, bundles carry their feature version, v1 is unchanged."""
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
from semlex_synthetic import rotation, semlex_array  # noqa: E402
from test_de_world_regressor import hand_points, small_bundle  # noqa: E402

from signtranslator.data_engineering import world_regressor as wr  # noqa: E402


def xy_bundle(seed=0):
    torch.manual_seed(seed)
    hand = {s: (wr.MLP(40, 63, width=16, depth=1).eval(), np.zeros(40, np.float32), np.ones(40, np.float32))
            for s in ("Left", "Right")}
    pose = (wr.MLP(54, 75 + 33, width=16, depth=1).eval(), np.zeros(54, np.float32), np.ones(54, np.float32))
    return wr.WorldRegressors(hand=hand, pose=pose, features=wr.FEATURES_XY)


def test_xy_hand_features_ignore_z_and_keep_the_in_plane_invariances():
    image = hand_points()
    f0, rot0, scale0, ok = wr.hand_features_xy(image, 4 / 3)
    assert ok.all() and f0.shape == (len(image), 40)
    deep = image.copy()
    deep[..., 2] = np.random.default_rng(3).normal(scale=.5, size=deep.shape[:2])     # any depth at all
    f1, rot1, scale1, _ = wr.hand_features_xy(deep, 4 / 3)
    np.testing.assert_array_equal(f0, f1)
    np.testing.assert_array_equal(rot0, rot1)
    f2, *_ = wr.hand_features_xy(image[..., :2], 4 / 3)                              # (N, 21, 2) input
    np.testing.assert_array_equal(f0, f2)
    turned = image.copy()
    centre = image[:, :1, :2] * [1, .75]
    q = (image[..., :2] * [1, .75]) - centre
    turned[..., :2] = (q @ rotation((0, 0, 1), -52.)[:2, :2].T * .6 + centre + [-.2, .1]) / [1, .75]
    f3, *_ = wr.hand_features_xy(turned, 4 / 3)
    np.testing.assert_allclose(f0, f3, atol=1e-5)
    # the v1 features do read z (the reason for this variant)
    assert not np.allclose(wr.hand_features(image, 4 / 3)[0], wr.hand_features(deep, 4 / 3)[0])


def test_xy_hand_features_mark_degenerate_and_non_finite_hands_invalid():
    image = hand_points(n=4)
    image[1] = image[1, :1]                       # all points on the wrist: zero scale
    image[2, 7, 0] = np.nan
    image[3, 9, :2] = image[3, 0, :2]             # wrist -> middle MCP of zero length: no "up"
    features, _, _, ok = wr.hand_features_xy(image, 1.)
    assert ok.tolist() == [True, False, False, False]
    assert not features[1:].any()


def test_xy_pose_features_ignore_z_and_are_relative_to_the_shoulders():
    array, *_ = semlex_array((640, 480))
    pose = array[:, 478:511]
    f0, ok = wr.pose_features_xy(pose, 4 / 3)
    assert ok.all() and f0.shape == (len(pose), 54)
    flat = pose.copy()
    flat[..., 2] = 0.
    np.testing.assert_array_equal(f0, wr.pose_features_xy(flat, 4 / 3)[0])
    np.testing.assert_array_equal(f0, wr.pose_features_xy(pose[..., :2], 4 / 3)[0])
    shifted = wr.pose_features_xy(pose + [.05, .02, 0.], 4 / 3)[0]
    np.testing.assert_allclose(f0[:, :50], shifted[:, :50], atol=1e-5)
    assert not np.allclose(f0[:, 50:], shifted[:, 50:])
    # the same context block as v1 (shoulder midpoint, width, aspect)
    np.testing.assert_allclose(f0[:, 50:], wr.pose_features(pose, 4 / 3)[0][:, 75:], atol=1e-6)


def test_xy_bundle_round_trip_reads_xy_and_xyz_alike_and_records_its_version():
    bundle = xy_bundle()
    payload = bundle.save()
    loaded = wr.WorldRegressors.load(payload)
    assert loaded.features == wr.FEATURES_XY and loaded.sha256 == hashlib.sha256(payload).hexdigest()
    blob = torch.load(io.BytesIO(payload), weights_only=True)
    assert blob["version"] == wr.FEATURES_XY
    image = hand_points()
    a, ok = loaded.hands(image, 1., "Left")
    b, _ = loaded.hands(image[..., :2], 1., "Left")
    assert ok.all()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, bundle.hands(image, 1., "Left")[0])
    array, *_ = semlex_array((640, 480))
    w, vis, ok = loaded.pose_world(array[:, 478:511, :2], 4 / 3)
    assert ok.all() and not w[:, 25:].any() and ((vis >= 0) & (vis <= 1)).all()


def test_v1_bundles_keep_their_version_and_mismatched_widths_are_refused():
    v1 = small_bundle()
    assert v1.features == wr.FEATURES_VERSION
    payload = v1.save()
    assert torch.load(io.BytesIO(payload), weights_only=True)["version"] == wr.FEATURES_VERSION
    assert wr.WorldRegressors.load(payload).features == wr.FEATURES_VERSION
    # a v1-width model labelled xy (or the reverse) is refused at load
    for bundle, version in ((small_bundle(), wr.FEATURES_XY), (xy_bundle(), wr.FEATURES_VERSION)):
        blob = torch.load(io.BytesIO(bundle.save()), weights_only=True)
        blob["version"] = version
        buffer = io.BytesIO()
        torch.save(blob, buffer)
        with pytest.raises(ValueError, match="input widths"):
            wr.WorldRegressors.load(buffer.getvalue())
    blob = torch.load(io.BytesIO(xy_bundle().save()), weights_only=True)
    blob["version"] = "world-regressor-features-unknown"
    buffer = io.BytesIO()
    torch.save(blob, buffer)
    with pytest.raises(ValueError, match="unknown feature version"):
        wr.WorldRegressors.load(buffer.getvalue())


def test_the_pinned_bundle_constants_are_unchanged():
    assert wr.FEATURES_VERSION == "world-regressor-features-v1"
    assert wr.WORLD_REGRESSORS is wr.WORLD_REGRESSORS_V2
    assert wr.WORLD_REGRESSORS_V2["sha256"] == "0953cd2cb2b48506808dff2d7199061eac3218ad555ea35327ff34edac3acab1"
    assert wr.WORLD_REGRESSORS_V1["sha256"] == "2d17b7434a5360f850d5910c284ffee3ab040d9132933127e3a7ef6a1b3ce793"


def test_training_with_xy_features_writes_an_xy_bundle_and_a_zero_depth_baseline(tmp_path):
    import train_world_regressors as job
    from test_train_world_regressors_script import _write_pairs
    pairs, legacy = tmp_path / "pairs", tmp_path / "legacy"
    pairs.mkdir()
    legacy.mkdir()
    for k, signer in enumerate(("h2s-1", "h2s-2", "h2s-3", "h2s-5")):
        _write_pairs(pairs, f"v{k}_1-{signer[-1]}-rgb_front__1x1.npz", signer, k + 1, legacy)
    v2_like = small_bundle()
    (tmp_path / "v2.pt").write_bytes(v2_like.save())
    report_path, bundle_path = tmp_path / "report.json", tmp_path / "bundle.pt"
    assert job.main(["--pairs", str(pairs), "--legacy-dir", str(legacy), "--epochs", "1", "--features", "xy",
                     "--baseline-bundle", str(tmp_path / "v2.pt"), "--out", str(bundle_path),
                     "--report", str(report_path)]) == 0
    report = json.loads(report_path.read_text())
    bundle = wr.WorldRegressors.load(bundle_path.read_bytes(), expected_sha256=report["bundle_sha256"])
    assert bundle.features == wr.FEATURES_XY and report["settings"]["features"] == wr.FEATURES_XY
    assert report["pseudo_world_depth_gains"] == {"pose": 0., "hand": 0.}
    assert report["inputs"] == "legacy-holistic" and "baseline_bundle" in report["hand"]["Left"]
    assert bundle.hand["Left"][0].shape[0] == 40 and bundle.pose[0].shape[0] == 54
    # default v1 training is unchanged: no depth-gain entry, v1 features
    out = tmp_path / "v1.pt"
    assert job.main(["--pairs", str(pairs), "--legacy-dir", str(legacy), "--epochs", "1", "--out", str(out),
                     "--report", str(tmp_path / "v1.json")]) == 0
    v1_report = json.loads((tmp_path / "v1.json").read_text())
    assert "pseudo_world_depth_gains" not in v1_report and v1_report["settings"]["features"] == wr.FEATURES_VERSION
    assert wr.load_regressors(out).features == wr.FEATURES_VERSION
