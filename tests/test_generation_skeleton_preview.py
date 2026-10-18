"""Skeleton previews: body-frame invariance, drawing, encoding on the motion clock."""
import numpy as np
import pytest

from signtranslator.generation.skeleton_preview import (
    body_frame, encode_preview, mixamo_bones, preview_joints, render_preview_frames,
)

BONES = mixamo_bones()
JOINTS = preview_joints(BONES)


def skeleton(frames=6, yaw=0.):
    rng = np.random.default_rng(0)
    base = rng.normal(scale=.05, size=(len(JOINTS), 3))
    index = {n: i for i, n in enumerate(JOINTS)}
    base[index["LeftShoulder"]] = [.2, 1.4, 0.]
    base[index["RightShoulder"]] = [-.2, 1.4, 0.]
    base[index["Spine2"]] = [0., 1.3, 0.]
    for side, sign in (("Left", 1), ("Right", -1)):
        base[index[f"{side}Hand"]] = [.5 * sign, 1.1, .2]
        for k, j in enumerate(n for n in JOINTS if n.startswith(f"{side}Hand") and n != f"{side}Hand"):
            base[index[j]] = base[index[f"{side}Hand"]] + [.01 * sign * (k % 4 + 1), -.01 * (k // 4 + 1), .005 * k]
    motion = np.stack([base + [0., .01 * t, 0.] for t in range(frames)])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    return motion @ rotation.T + [3., 0., -2.]


def test_body_frame_removes_capture_yaw_and_translation():
    np.testing.assert_allclose(body_frame(skeleton(yaw=0.), JOINTS), body_frame(skeleton(yaw=1.1), JOINTS), atol=1e-12)


def test_frames_are_identical_under_capture_rotation_and_draw_bones():
    a = render_preview_frames(skeleton(yaw=0.), JOINTS, BONES, panel=128)
    b = render_preview_frames(skeleton(yaw=2.3), JOINTS, BONES, panel=128)
    assert len(a) == 6 and a[0].shape == (256, 256, 3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert (a[0] != 255).any(axis=-1).sum() > 200  # bones were drawn
    assert not np.array_equal(a[0], a[-1])         # motion is visible


def test_encoding_follows_the_motion_clock(tmp_path):
    frames = render_preview_frames(skeleton(frames=8), JOINTS, BONES, panel=64)
    clock = np.cumsum(np.full(8, 1 / 24.))
    info = encode_preview(frames, clock, tmp_path / "p.mp4")
    assert info["frames"] == 8 and (tmp_path / "p.mp4").stat().st_size > 0
    import av
    with av.open(str(tmp_path / "p.mp4")) as container:
        times = [float(f.pts * f.time_base) for f in container.decode(video=0)]
    np.testing.assert_allclose(times, clock - clock[0], atol=1e-4)
    with pytest.raises(ValueError):
        render_preview_frames(skeleton(), JOINTS, BONES, panel=63)
