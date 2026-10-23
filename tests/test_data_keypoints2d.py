"""Shared 2D keypoint layout: video Holistic landmarks and canonical motion land in the same normalised points."""
import numpy as np

from signtranslator.data.keypoints2d import (FEATURES, POINTS, features, from_holistic, from_state, normalise,
                                            resample)
from test_generation_canonical_skeleton import rest_state


def holistic_frame(shift=0.):
    pose = [[0., 0.]] * 33
    pose = [list(p) for p in pose]
    pose[11], pose[12] = [600. + shift, 300.], [400. + shift, 300.]   # signer's left shoulder is image right
    pose[13], pose[14], pose[15], pose[16] = [650., 400.], [350., 400.], [640., 500.], [360., 500.]
    left = [[640. + k, 500. - k] for k in range(21)]
    return dict(pose_landmarks=pose, left_hand_landmarks=left, right_hand_landmarks=None)


def test_holistic_points_follow_the_layout_and_mark_missing_parts():
    points, valid = from_holistic([holistic_frame(), dict(pose_landmarks=None, left_hand_landmarks=None)])
    assert points.shape == (2, len(POINTS), 2) and len(POINTS) == 48
    np.testing.assert_allclose(points[0, POINTS.index("LeftShoulder")], [600., 300.])
    np.testing.assert_allclose(points[0, POINTS.index("LeftHandWrist")], [640., 500.])
    np.testing.assert_allclose(points[0, POINTS.index("LeftHandPinky4")], [660., 480.])
    assert valid[0, :27].all() and not valid[0, 27:].any() and not valid[1].any()


def test_normalisation_puts_shoulders_at_unit_width_with_y_up():
    points, valid = from_holistic([holistic_frame(), holistic_frame()])
    n = normalise(points, valid, y_down=True)
    np.testing.assert_allclose(n[0, 0], [.5, 0.], atol=1e-6)
    np.testing.assert_allclose(n[0, 1], [-.5, 0.], atol=1e-6)
    assert n[0, POINTS.index("LeftWrist"), 1] < 0              # wrists below the shoulders
    assert (n[:, 27:] == 0).all()                               # missing right hand stays zero
    assert features(n, valid).shape == (2, FEATURES)


def test_canonical_rest_pose_projects_like_a_signer_facing_the_camera():
    points, valid = from_state(rest_state())
    assert valid.all()
    n = normalise(points, valid, y_down=False, hands_local=False)
    np.testing.assert_allclose(n[0, 0], [.5, 0.], atol=1e-6)    # left shoulder on +x, as in the image
    assert n[0, POINTS.index("LeftWrist"), 0] > n[0, POINTS.index("LeftElbow"), 0] > .5
    wrist, tip = POINTS.index("LeftHandWrist"), POINTS.index("LeftHandIndex4")
    np.testing.assert_allclose(n[0, wrist], n[0, POINTS.index("LeftWrist")], atol=1e-6)   # arm and hand wrist agree
    assert n[0, tip, 1] > n[0, wrist, 1]                        # rest fingers point up


def test_resampling_keeps_validity_and_endpoints():
    p = np.arange(31, dtype=np.float32)[:, None, None] * np.ones((1, 48, 2), np.float32)
    v = np.ones((31, 48), bool); v[10] = False
    q, ok = resample(p, v, fps=30., target_fps=24.)
    assert len(q) == 25 and q[0, 0, 0] == 0 and abs(q[-1, 0, 0] - 30.) < 1e-5
    assert not ok.all() and ok[0].all()


def test_local_hands_are_size_invariant():
    points, valid = from_state(rest_state())
    big = points.copy()
    for k in range(2):                                          # a hand twice the size around its wrist
        a = 6 + 21 * k
        big[:, a + 1:a + 21] = points[:, a:a + 1] + 2 * (points[:, a + 1:a + 21] - points[:, a:a + 1])
    n, m = normalise(points, valid, y_down=False), normalise(big, valid, y_down=False)
    np.testing.assert_allclose(n, m, atol=1e-5)
    np.testing.assert_allclose(n[0, POINTS.index("LeftHandWrist")], n[0, POINTS.index("LeftWrist")], atol=1e-6)
