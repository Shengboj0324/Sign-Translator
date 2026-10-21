"""Take-5 recovery: clip placement in the raw recording, sub-frame camera offset, re-cut."""
import numpy as np

from signtranslator.data_engineering.how2sign_recut import (
    estimate_offset, needed_frames, nominal_start, place_clips, recut, sample,
)
from signtranslator.data_engineering.two_view_geometry import project
from signtranslator.data_engineering.two_view_states import SYNC_JOINTS, ClipKeypoints
from tests.test_de_two_view_states import cameras


def scene(times, seed=0):
    """(T, 308, 3) camera-frame metres: every SYNC joint moves smoothly in continuous time (frames)."""
    rng = np.random.default_rng(seed)
    joints = list(SYNC_JOINTS)
    base = np.array([0., 0., 3.]) + rng.normal(scale=.15, size=(len(joints), 3))
    amplitude = rng.uniform(.03, .12, size=(len(joints), 3))
    omega = rng.uniform(.08, .25, size=(len(joints), 3))
    phase = rng.uniform(0, 2 * np.pi, size=(len(joints), 3))
    t = np.asarray(times, dtype=np.float64)[:, None, None]
    out = np.zeros((len(t), 308, 3))
    out[:, joints] = base + amplitude * np.sin(omega * t + phase)
    return out


def views(times, camera, seed=0, noise=.3):
    """Projected keypoints of the scene at ``times`` in one camera, with confidence 1 on SYNC joints."""
    points3d = scene(times, seed)
    joints = list(SYNC_JOINTS)
    points = np.zeros((len(points3d), 308, 2))
    points[:, joints] = project(camera, points3d[:, joints].reshape(-1, 3)).reshape(len(points3d), len(joints), 2)
    points[:, joints] += np.random.default_rng(seed + 50).normal(scale=noise, size=points[:, joints].shape)
    confidence = np.zeros((len(points3d), 308))
    confidence[:, joints] = 1.
    return points, confidence


def recordings(count=420, offset=9.25, seed=0, side_seed=None):
    """Raw front and raw side keypoints; raw side frame k shows front time k - offset."""
    p1, p2 = cameras()
    front = views(np.arange(count), p1, seed)
    side = views(np.arange(count) - offset, p2, seed if side_seed is None else side_seed)
    return front, side


def test_nominal_start_is_forty_times_the_csv_start_rounded_up():
    assert nominal_start(.28) == 12 and nominal_start(10.) == 400 and nominal_start(40.32) == 1613


def test_clips_are_placed_to_the_frame_and_still_or_foreign_clips_handled():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=(300, 2)), axis=0)

    def frame(i, still=None):
        y, x = np.mgrid[0:36, 0:64]
        w = walk[i if still is None else still]
        cy, cx = 8 + w[0] % 20, 10 + w[1] % 44
        return 100 + 80 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 20.)

    raw = np.stack([frame(i) for i in range(300)])
    noisy = lambda a: a + rng.normal(scale=3., size=a.shape)
    raw_still = raw.copy()
    raw_still[200:260] = frame(0, still=200)                   # the signer holds still here
    other = np.stack([frame(i) for i in range(299, -1, -1)])   # another recording
    clips = {"a": noisy(raw[40:100]), "b": noisy(raw[150:170]), "still": noisy(raw_still[210:240]),
             "foreign": noisy(other[40:100])}
    nominals = {"a": 38, "b": 153, "still": 211, "foreign": 40}
    placed = place_clips(raw_still, clips, nominals)
    assert placed["a"]["status"] == "placed" and placed["a"]["start"] == 40
    assert placed["b"]["status"] == "placed" and placed["b"]["start"] == 150
    assert placed["still"]["status"] == "still" and placed["still"]["start"] == 211
    assert placed["foreign"]["status"] == "refused" and placed["foreign"]["start"] is None


def test_sampling_interpolates_and_blanks_outside_the_recording():
    points = np.arange(10, dtype=float)[:, None, None] * np.ones((10, 308, 2))
    confidence = np.ones((10, 308))
    confidence[4, 0] = .2
    out, conf = sample(points, confidence, np.array([2.25, 3.5, -.5, 9., 9.5]))
    assert np.allclose(out[:, 0, 0], [2.25, 3.5, 0., 9., 0.])
    assert conf[1, 0] == .2 and conf[1, 1] == 1. and conf[2].max() == 0. and conf[3, 0] == 1. and conf[4].max() == 0.


def test_offset_is_recovered_to_a_fraction_of_a_frame_and_the_recut_matches_the_side_view():
    (fp, fc), (sp, sc) = recordings(offset=9.25)
    starts = (20, 140, 260)
    placed = [(s, ClipKeypoints("f", np.arange(90) / 30., fp[s:s + 90], fc[s:s + 90])) for s in starts]
    result = estimate_offset(placed, sp, sc, max_offset=20)
    assert result["passed"], result["reason"]
    assert result["integer_offset"] == 9 and abs(result["offset_frames"] - 9.25) < .1
    assert result["sharpness"] < .5 and result["zero_lag_ratio"] < .5 and result["sampson_px"] < 1.
    assert len(result["clips"]) == 3 and all(c["sampson_px"] < 1. for c in result["clips"])
    # The re-cut lateral keypoints show the frontal clip's moments: compare with the truth.
    _, p2 = cameras()
    start, front = placed[1]
    side = recut(front, start, result["offset_frames"], sp, sc, "s")
    truth, _ = views(start + np.arange(90), p2, noise=0.)
    joints = list(SYNC_JOINTS)
    assert np.array_equal(side.times, front.times)
    assert np.median(np.linalg.norm(side.points[:, joints] - truth[:, joints], axis=-1)) < 1.


def test_a_side_recording_of_another_session_is_refused():
    (fp, fc), (sp, sc) = recordings(side_seed=7)
    placed = [(s, ClipKeypoints("f", np.arange(90) / 30., fp[s:s + 90], fc[s:s + 90])) for s in (20, 140, 260)]
    result = estimate_offset(placed, sp, sc, max_offset=20)
    assert not result["passed"] and result["offset_frames"] is None and result["reason"]


def test_an_offset_beyond_the_search_range_is_refused():
    (fp, fc), (sp, sc) = recordings(offset=25.)
    placed = [(s, ClipKeypoints("f", np.arange(90) / 30., fp[s:s + 90], fc[s:s + 90])) for s in (40, 160, 280)]
    result = estimate_offset(placed, sp, sc, max_offset=12)
    assert not result["passed"]


def test_needed_frames_cover_the_search_and_stay_in_the_recording():
    assert needed_frames([(100, 50), (300, 20)], 1000, max_offset=10) == (88, 332)
    assert needed_frames([(5, 50)], 60, max_offset=10) == (0, 60)


def test_offset_scoring_in_processes_gives_the_same_result():
    (fp, fc), (sp, sc) = recordings(offset=9.25)
    placed = [(s, ClipKeypoints("f", np.arange(90) / 30., fp[s:s + 90], fc[s:s + 90])) for s in (20, 140, 260)]
    assert estimate_offset(placed, sp, sc, max_offset=12, workers=2) == estimate_offset(placed, sp, sc, max_offset=12)
