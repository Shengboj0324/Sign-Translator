"""Metric scale from anthropometric cues and the session-world translations."""
import numpy as np
import pytest

from signtranslator.data_engineering import body_space as bs
from signtranslator.data_engineering.video_landmarks import Y_UP

REF = {name: values[0] for name, values in bs.POPULATION.items()}


def signer(frames=200, factor=1., seed=0, ipd_factor=1.):
    """(T, 308, 3) Goliath points of a moving signer whose cue lengths are ``factor`` × the references (metres)."""
    rng = np.random.default_rng(seed)
    points = np.full((frames, 308, 3), np.nan)
    t = np.arange(frames)
    torso = np.c_[.05 * np.sin(t / 17), .02 * np.cos(t / 13), 2.5 + .03 * np.sin(t / 23)]
    def direction(k):
        v = np.c_[np.sin(t / (7 + k)), np.cos(t / (9 + k)), .5 + .2 * np.sin(t / 5)] + rng.normal(size=3)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    for sign, shoulder, elbow, wrist, middle in ((1, 5, 7, 62, bs.LEFT_MIDDLE), (-1, 6, 8, 41, bs.RIGHT_MIDDLE)):
        points[:, shoulder] = torso + [sign * .18, 0, 0]
        points[:, elbow] = points[:, shoulder] + factor * REF["upper_arm"] * direction(shoulder)
        points[:, wrist] = points[:, elbow] + factor * REF["forearm"] * direction(elbow)
        previous = wrist
        for k, share in zip(middle, (.45, .25, .17, .13)):
            points[:, k] = points[:, previous] + factor * share * REF["hand_length"] * direction(k)
            previous = k
    head = torso + [0, -.25, -.05]
    half = factor * ipd_factor * REF["interpupillary"] / 2
    points[:, 1], points[:, 2] = head + [half, 0, 0], head - [half, 0, 0]
    return points, torso, head


def test_scale_is_exact_for_a_population_reference_signer_in_any_units():
    points, _, _ = signer()
    for units in (1., .37, 4.2):                       # reconstruction units per metre
        estimate = bs.metric_scale(points * units)
        assert estimate.scale == pytest.approx(1 / units, rel=1e-9)
        assert estimate.chi2 == pytest.approx(0., abs=1e-12)
        assert set(estimate.cues) == set(bs.CUES)
        for name, (metres, samples) in estimate.cues.items():
            assert metres == pytest.approx(REF[name], rel=1e-9) and samples >= 200


def test_predicted_uncertainty_is_the_gls_variance_and_falls_with_more_cues():
    points, _, _ = signer()
    full = bs.metric_scale(points)
    names = list(bs.CUES)
    sd = np.array([bs.POPULATION[n][1] for n in names])
    extra = np.array([bs.POPULATION[n][2] ** 2 + bs.POPULATION[n][3] ** 2 for n in names])
    corr = np.array([[bs._correlation(a, b) for b in names] for a in names])
    covariance = np.outer(sd, sd) * corr + np.diag(extra)   # noiseless lengths: no sampling term
    assert full.relative_sd == pytest.approx(1 / np.sqrt(np.linalg.solve(covariance, np.ones(4)).sum()), rel=1e-9)
    eyes_only = points.copy()
    eyes_only[:, [5, 6, 7, 8, 62, 41]] = np.nan
    assert bs.metric_scale(eyes_only).relative_sd > full.relative_sd
    # The population term is a floor: a signer can only be measured, not known, to be average.
    assert full.relative_sd > .05


def test_a_proportionally_larger_signer_is_indistinguishable_and_a_disagreeing_cue_is_flagged():
    points, _, _ = signer(factor=1.08)
    assert bs.metric_scale(points).scale == pytest.approx(1 / 1.08, rel=1e-9)    # the inherent ambiguity
    wide_eyes, _, _ = signer(ipd_factor=1.3)
    estimate = bs.metric_scale(wide_eyes)
    assert estimate.chi2 > 3.                                                  # cues disagree
    assert abs(np.log(estimate.scale)) < np.log(1.3) / 4                      # the noisy eye cue is down-weighted


def test_rarely_observed_cues_are_skipped_and_no_cue_is_refused():
    points, _, _ = signer()
    points[bs.MIN_CUE_FRAMES // 2:, [1, 2]] = np.nan
    assert "interpupillary" not in bs.metric_scale(points).cues
    with pytest.raises(ValueError):
        bs.metric_scale(np.full((50, 308, 3), np.nan))


def test_translations_are_metric_y_up_and_relative_to_the_session_median():
    points, torso, head = signer()
    units = .5
    recon = points * units
    estimate = bs.metric_scale(recon)
    frame = bs.session_frame(recon, estimate)
    np.testing.assert_allclose(frame.origin, np.median(torso, axis=0) * units, atol=1e-12)
    recon[10, 6] = np.nan                                  # torso anchor lost in one frame
    root, root_ok, eye, eye_ok = bs.body_translations(recon, frame)
    assert not root_ok[10] and root_ok.sum() == len(root) - 1 and eye_ok.all()
    assert np.all(root[10] == 0)
    expected = (torso - np.median(torso, axis=0)) @ Y_UP.T
    np.testing.assert_allclose(root[root_ok], expected[root_ok], atol=1e-12)
    np.testing.assert_allclose(eye, (head - np.median(torso, axis=0)) @ Y_UP.T, atol=1e-12)
    assert np.all(eye[:, 1] > .2)                          # the eyes are above the shoulders in y-up


def test_two_view_state_carries_the_translations():
    from signtranslator.data_engineering.two_view_states import TRANSLATION_FRAME, triangulated_state
    import tests.test_de_two_view_states as tv
    from dataclasses import replace
    from tests.test_de_video_landmarks import clip as mp_clip, convert
    front, side, _ = tv.clips()
    times, fp, fc, sp, sc = tv.aligned(front, side, 2)
    calibration = tv.calibrate_session([(fp, fc, sp, sc)])
    points, usable = tv.triangulated_points(calibration, fp, fc, sp, sc)
    inputs = tv.rotation_inputs(points, usable)
    rotations, hands = tv.solve_rotations(inputs["world"], inputs["visible"], inputs["hands"])
    base = mp_clip()
    n = len(front.times)
    tile = lambda a: np.repeat(a[:1], n, axis=0)
    monocular, _ = convert(replace(base, times=np.arange(n) / 30., pose_world=tile(base.pose_world),
                                   pose_visibility=tile(base.pose_visibility), pose_image=tile(base.pose_image),
                                   pose_found=tile(base.pose_found), hands_world=tile(base.hands_world),
                                   hands_image=tile(base.hands_image), hands_found=tile(base.hands_found),
                                   blendshapes=tile(base.blendshapes), face_matrix=tile(base.face_matrix),
                                   face_found=tile(base.face_found)))
    count = len(times)
    root = np.c_[np.linspace(0, .1, count), np.zeros(count), np.zeros(count)]
    ok = np.ones(count, bool); ok[3] = False
    state = triangulated_state(monocular, slice(0, count), rotations, hands, source_id="tv", source_sha256="c" * 64,
                               inference_method="two-view", translations=dict(root_translation=(root, ok),
                                                                               head_translation=(root + [0, .3, 0], ok)))
    state.validate()
    channel = state.channels["root_translation"]
    assert channel.units == "m" and channel.coordinate_frame == TRANSLATION_FRAME and channel.labels == ("torso_anchor",)
    assert channel.valid[:, 0].sum() == count - 1 and np.all(channel.values[3] == 0) and not channel.observed.any()
    np.testing.assert_allclose(state.channels["head_translation"].values[ok, 0, 1], .3)
    with pytest.raises(ValueError):
        triangulated_state(monocular, slice(0, count), rotations, hands, source_id="tv", source_sha256="c" * 64,
                           inference_method="two-view", translations=dict(root_translation=(root[:-1], ok[:-1])))
