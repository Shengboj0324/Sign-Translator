"""Elbow-outside accuracy script: distances, categories, the ungated solve and the clip-bootstrap difference."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import elbow_outside_accuracy as job  # noqa: E402
from semlex_synthetic import FakeBlendshapes, fake_resources, semlex_array  # noqa: E402

from signtranslator.data_engineering.semlex_pose import holistic_clip  # noqa: E402
from signtranslator.data_engineering.video_landmarks import canonical_from_landmarks  # noqa: E402


def test_outside_distance_is_in_frame_heights():
    xy = np.array([[.5, .5], [.5, 1.2], [1.1, .5], [-.1, 1.1]])
    np.testing.assert_allclose(job.outside_distance(xy, 4 / 3), [0., .2, .1 * 4 / 3, np.hypot(.1 * 4 / 3, .1)])


def test_categories_and_errors_against_a_truth_of_the_same_points():
    array, *_ = semlex_array((640, 470), frames=8)
    clip, _ = holistic_clip(array, 8000 / 30, fake_resources(), image_size=(640, 470), blendshape_model=FakeBlendshapes(),
                            pose_z_scale=1., hand_z_scale=1.)
    visibility = np.ones((8, 33))
    visibility[:, 23:] = 0.
    visibility[0, 13] = .2                                     # frame 0: the left elbow "not visible"
    clip = replace(clip, pose_visibility=visibility)
    truth_clip = replace(clip, pose_visibility=np.where(np.arange(33) < 23, 1., 0.)[None].repeat(8, 0))
    truth, _ = canonical_from_landmarks(truth_clip, sample_id="t", source_id="t", source_sha256="a" * 64, clock_id="t",
                                        inference_method="t", upright_chest=True, upright_root=True)
    meta = dict(video="v.mp4", name="v__4x3", size=[640, 470])
    rows = job.clip_rows(meta, clip, truth, np.arange(8))
    left = [r for r in rows if r["side"] == "Left"]
    outside = ~((array[:, 478 + 13, :2] >= 0) & (array[:, 478 + 13, :2] <= 1)).all(-1)
    expected = ["iii" if (t == 0 and not outside[t]) else "iv" if t == 0 else "ii" if outside[t] else "i" for t in range(8)]
    assert [r["category"] for r in left] == expected
    for r in rows:                                             # the ungated solve equals the truth's arms
        for q in ("UpperArm", "Forearm", "Hand"):
            assert not np.isfinite(r[q]) or r[q] < 1e-3, (r, q)
    assert left[0]["UpperArm_stored"] is False                 # the visibility rule drops frame 0's arm
    assert all(r["UpperArm_stored"] for r in left[1:])


def test_difference_of_medians_with_video_bootstrap():
    rows = [dict(video=f"v{k % 4}", clip="c", category=c, UpperArm=float(e))
            for k, (c, e) in enumerate([("i", 10), ("ii", 12), ("i", 11), ("ii", 13)] * 5)]
    d = job.difference_ci(rows, "UpperArm")
    assert d["difference"] == 2. and d["ci"][0] <= 2. <= d["ci"][1]
