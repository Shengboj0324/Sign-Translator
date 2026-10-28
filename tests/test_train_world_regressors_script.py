"""World-regressor training script: IK error metrics are zero on identical points and the fit learns a mapping."""
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import train_world_regressors as job  # noqa: E402
from semlex_synthetic import body_y_up, hand_y_up, rotation  # noqa: E402

from signtranslator.data_engineering import world_regressor as wr  # noqa: E402
from signtranslator.data_engineering.video_landmarks import Y_UP  # noqa: E402


def test_metrics_are_zero_for_identical_points_and_see_a_turned_finger():
    body = body_y_up(6)
    hand = hand_y_up(body[:, 16], 3)
    fingers, palm = job.finger_errors(hand, hand, "Right")
    assert max(fingers) < 1e-4 and max(palm) < 1e-4
    bent = hand.copy()
    bent[:, 8] = bent[:, 7] + (bent[:, 8] - bent[:, 7]) @ rotation((1, 0, 0), 40.).T   # index tip bent 40 deg
    fingers, palm = job.finger_errors(hand, bent, "Right")
    assert 39 < max(fingers) < 41 and max(palm) < 1e-4
    world = body @ Y_UP
    errors = job.arm_errors(world, world, np.ones((6, 33), bool))
    assert all(max(v) < 1e-4 for v in errors.values() if v)


def test_fit_learns_a_linear_map_with_directions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 12)).astype(np.float32)
    m = rng.normal(size=(12, 9)).astype(np.float32)
    y = x @ m
    model, mean, std = job.fit(x, y, bones=((0, 1), (1, 2)), points=3, epochs=30, batch=256, lr=3e-3)
    with torch.no_grad():
        pred = model(torch.from_numpy((x - mean) / std)).numpy()
    assert np.mean((pred - y) ** 2) < .05 * np.mean(y ** 2)
    assert isinstance(model, wr.MLP)


def _write_pairs(directory, name, signer, seed, legacy_dir=None):
    """A tiny pairs file (and its legacy-input twin) from the synthetic clip: Tasks image = projection, world = 3D."""
    from semlex_synthetic import semlex_array
    array, body, hands = semlex_array((640, 640), frames=12, seed=seed)
    hand_img = np.stack((array[:, 511:532], array[:, 532:553]), 1)
    hand_world = np.stack(((hands["Left"] - hands["Left"][:, :1]) @ Y_UP,
                           (hands["Right"] - hands["Right"][:, :1]) @ Y_UP), 1)
    np.savez(Path(directory) / name, signer=np.array(signer), clip=np.array("x.mp4"), shape=np.array("1:1"),
             size=np.array([640, 640]), pose_found=np.ones(12, bool), pose_img=array[:, 478:511],
             pose_world=body @ Y_UP, pose_vis=np.where(np.arange(33) < 23, .9, .1)[None].repeat(12, 0),
             hand_found=np.ones((12, 2), bool), hand_img=hand_img, hand_world=hand_world)
    if legacy_dir is not None:
        noisy = array + np.random.default_rng(seed).normal(scale=1e-3, size=array.shape)
        np.savez(Path(legacy_dir) / name, holistic=noisy.astype(np.float16), size=np.array([640, 640]),
                 times=np.arange(12) / 30., pose_visibility=np.zeros((12, 33), np.float16))


def test_main_trains_on_legacy_inputs_writes_the_bundle_first_and_scores_a_baseline(tmp_path):
    import hashlib
    import json
    import pytest
    pairs, legacy = tmp_path / "pairs", tmp_path / "legacy"
    pairs.mkdir()
    legacy.mkdir()
    for k, signer in enumerate(("h2s-1", "h2s-2", "h2s-3", "h2s-5")):
        _write_pairs(pairs, f"v{k}_1-{signer[-1]}-rgb_front__1x1.npz", signer, k + 1, legacy)
    baseline = wr.WorldRegressors(
        hand={s: (wr.MLP(60, 63, width=8, depth=1).eval(), np.zeros(60, np.float32), np.ones(60, np.float32))
              for s in ("Left", "Right")},
        pose=(wr.MLP(79, 108, width=8, depth=1).eval(), np.zeros(79, np.float32), np.ones(79, np.float32)))
    (tmp_path / "baseline.pt").write_bytes(baseline.save())
    report_path, bundle_path = tmp_path / "report.json", tmp_path / "bundle.pt"
    assert job.main(["--pairs", str(pairs), "--legacy-dir", str(legacy), "--epochs", "1", "--baseline-bundle",
                     str(tmp_path / "baseline.pt"), "--out", str(bundle_path), "--report", str(report_path)]) == 0
    report = json.loads(report_path.read_text())
    assert report["bundle_sha256"] == hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    assert report["inputs"] == "legacy-holistic" and report["val_signers"] == ["h2s-1", "h2s-2"]
    assert report["settings"]["epochs"] == 1 and report["settings"]["seed"] == 0
    assert {m["split"] for m in report["pairs_manifest"] if m["signer"] in ("h2s-1", "h2s-2")} == {"val"}
    assert all(m["legacy_sha256"] and m["pairs_sha256"] for m in report["pairs_manifest"])
    assert "baseline_bundle" in report["pose"] and "baseline_bundle" in report["hand"]["Left"]
    wr.WorldRegressors.load(bundle_path.read_bytes(), expected_sha256=report["bundle_sha256"])
    with pytest.raises(FileExistsError):                       # the bundle is written once
        job.main(["--pairs", str(pairs), "--legacy-dir", str(legacy), "--epochs", "1", "--out", str(bundle_path),
                  "--report", str(report_path)])
