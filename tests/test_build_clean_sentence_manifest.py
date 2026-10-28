"""Fix 4: the sentence cleaner's pure rules (span finding, rate band, drop order) on synthetic wrist heights."""
import numpy as np
import pytest

from scripts.build_clean_sentence_manifest import (
    active_frames, drop_short_runs, judge, rate_outliers, robust_band, signing_span, word_count)

FPS = 24.


def clip(signing, frames=240, rest=-.42, up=-.1):
    """(2, T) heights: Left rests throughout (with noise), Right signs on ``signing`` (slice) and rests otherwise."""
    rng = np.random.default_rng(0)
    h = np.full((2, frames), rest) + rng.normal(0, .005, (2, frames))
    h[1, signing] = up + rng.normal(0, .03, len(range(*signing.indices(frames))))
    return h, np.ones((2, frames), bool)


def test_span_is_the_signing_and_margin_clipped_to_the_clip():
    h, ok = clip(slice(48, 168))
    span, window = signing_span(active_frames(h, ok), FPS, .3)
    assert span == pytest.approx((48 / FPS, 167 / FPS))
    assert window == pytest.approx((48 / FPS - .3, 167 / FPS + .3))
    h, ok = clip(slice(2, 239))
    _, window = signing_span(active_frames(h, ok), FPS, .3)
    assert window == pytest.approx((0., 239 / FPS))                    # clipped to the clip


def test_union_of_both_wrists_and_a_resting_hand_adds_nothing():
    h, ok = clip(slice(100, 160))
    h[0, 30:80] = -.12                                                  # the other hand signs earlier
    span, _ = signing_span(active_frames(h, ok), FPS, 0.)
    assert span == pytest.approx((30 / FPS, 159 / FPS))
    h, ok = clip(slice(100, 160))                                       # left only rests (noise): no activity from it
    assert not active_frames(h, ok)[:100].any()


def test_untracked_frames_and_short_spikes_are_not_activity():
    h, ok = clip(slice(100, 160))
    h[1, 10:12] = 0.                                                    # a 2-frame glitch
    ok[1, 200:220] = False; h[1, 200:220] = .5                          # untracked garbage
    a = active_frames(h, ok, min_run=3)
    assert not a[10:12].any() and not a[200:220].any() and a[100:160].mean() > .9


def test_a_clip_signed_throughout_keeps_its_lowest_signing():
    h, ok = clip(slice(0, 240), up=-.1)
    h[1, :24] = -.16; h[1, -24:] = -.16                                 # lower signing at the edges, no rest at all
    span, _ = signing_span(active_frames(h, ok), FPS, 0.)
    assert span == pytest.approx((0., 239 / FPS))                       # the rest ceiling keeps the edges


def test_no_tracking_or_flat_motion_is_no_span():
    h, ok = clip(slice(0, 0))
    assert signing_span(active_frames(h, ok), FPS, .3) is None
    assert signing_span(active_frames(h, np.zeros_like(ok)), FPS, .3) is None


def test_drop_short_runs():
    m = np.array([1, 1, 0, 1, 1, 1, 0, 1], bool)
    assert drop_short_runs(m, 3).tolist() == [0, 0, 0, 1, 1, 1, 0, 0]
    assert drop_short_runs(m, 1).tolist() == m.tolist()


def test_robust_band_and_rate_outliers():
    med, mad, lo, hi = robust_band([1., 2., 3., 4., 100.], k=3.)
    assert med == 3. and mad == pytest.approx(1.4826) and (lo, hi) == pytest.approx((3 - 3 * 1.4826, 3 + 3 * 1.4826))
    rates = {f"r{i}": r for i, r in enumerate([2., 2.2, 1.8, 2.1, 1.9, 2.0, 12., .2])}
    out, band = rate_outliers(rates, 3., "log")
    assert [k for k, v in out.items() if v] == ["r6", "r7"]             # too many and too few words both caught
    assert band[0] == pytest.approx(2., rel=.06)
    out, band = rate_outliers(rates, 3., "linear")
    assert out["r6"] and band[2] < 2.


def test_judge_order_and_coverage():
    kw = dict(fps=FPS, max_untracked=.5, min_seconds=.5, min_coverage=.4)
    assert judge(untracked=.6, span=((0., 5.), (0., 5.)), frames=240, words=8, **kw)[0] == "arms_untracked"
    assert judge(untracked=0., span=None, frames=240, words=8, **kw)[0] == "no_activity"
    assert judge(untracked=0., span=((1., 1.2), (.7, 1.5)), frames=240, words=8, **kw)[0] == "no_activity"
    reason, m = judge(untracked=0., span=((1., 3.), (.7, 3.3)), frames=240, words=8, **kw)
    assert reason == "low_coverage" and m["coverage"] == pytest.approx((2 + 1 / FPS) / 10.)
    reason, m = judge(untracked=0., span=((1., 8.), (.7, 8.3)), frames=240, words=14, **kw)
    assert reason is None and m["rate"] == pytest.approx(14 / (7 + 1 / FPS))


def test_word_count():
    assert word_count("Hello everyone.") == 2 and word_count("they're 3 cats") == 3 and word_count(None) == 0
