"""Where A1's loss sits, and the power table's arithmetic. Offline: synthetic rows only."""
import numpy as np
import pytest

import scripts.round15_compose_eval  # noqa: F401  (puts scripts on the path)
import comp_a6_hold_concentration as c


def clip(two):
    return dict(gloss=dict(handshapes={"Non-dominant start handshape": "B" if two else ""}))


def test_excluded_is_what_a1_held_and_hold_x_keeps():
    pools = dict(one=[clip(False)] * 4, quarter=[clip(True)] + [clip(False)] * 3, mostly=[clip(True)] * 3 + [clip(False)], none=[dict(gloss={})])
    assert c.excluded_entries(pools) == {"quarter"}


def test_clip_counts_split_the_held_uses():
    out = c.clip_counts([["quarter", "one", "other", "quarter"], ["other"]], dict(quarter=True, one=True, other=False), {"quarter"})
    assert out.tolist() == [[2., 1.], [0., 0.]]


def test_a_loss_only_in_excluded_sentences_is_all_there():
    delta = np.array([-1., -1., 0., 0.])
    counts = np.array([[2., 0.], [1., 1.], [0., 3.], [0., 1.]])
    out = c.concentration(delta, counts, boot=50)
    assert out["clips_with_excluded"] == 2 and out["ceiling"] == 1. and out["mean_with"] == -1. and out["mean_without"] == 0.
    assert out["scaled"] == pytest.approx((1. + .5) / 2., abs=1e-4) and out["share_of_clips_with_excluded"] == .5 and out["share_of_held_uses_excluded"] == pytest.approx(3 / 8, abs=1e-4)


def test_an_even_loss_sits_in_proportion_to_clips_and_absent_clips_stay_out():
    delta = np.array([-1., -1., -1., -1., np.nan])
    counts = np.array([[1., 0.], [0., 1.], [0., 2.], [2., 1.], [9., 9.]])
    out = c.concentration(delta, counts, boot=50)
    assert out["clips"] == 4 and out["ceiling"] == .5 and out["mean"] == -1.
    assert out["regression"] == pytest.approx(0., abs=1e-6)                        # no more loss per excluded use than per other


def test_the_regression_attributes_the_excluded_uses():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 4, (400, 2)).astype(float)
    delta = -.02 * counts[:, 0] - .001 * counts[:, 1]
    out = c.concentration(delta, counts, boot=50)
    assert out["slope_excluded"] == pytest.approx(-.02, abs=1e-6) and out["slope_other"] == pytest.approx(-.001, abs=1e-6)
    assert out["regression"] == pytest.approx(.02 * counts[:, 0].sum() / -delta.sum(), abs=1e-3)


def test_the_regression_through_the_origin_has_no_intercept():
    counts = np.array([[1., 0.], [0., 1.], [2., 1.], [1., 2.], [3., 0.], [0., 3.]])
    delta = -.03 * counts[:, 0] - .01 * counts[:, 1]
    out = c.concentration(delta - .5, counts, boot=20)
    assert out["slope_excluded"] == pytest.approx(-.03, abs=1e-6) and out["slope_excluded_through_origin"] != pytest.approx(-.03, abs=1e-3)      # the intercept moved to the slopes
    out = c.concentration(delta, counts, boot=20)
    assert out["slope_excluded_through_origin"] == pytest.approx(-.03, abs=1e-6) and out["slope_other_through_origin"] == pytest.approx(-.01, abs=1e-6)
    assert out["regression_through_origin"] == pytest.approx(.03 * counts[:, 0].sum() / -delta.sum(), abs=1e-3)


def test_the_frames_hold_x_stops_replacing(tmp_path):
    path = tmp_path / "m.json"
    path.write_text('{"forms": {"hold_a1": {"hold": {"frames": 100}}, "hold_x": {"hold": {"frames": 88}}}}')
    assert c.frames_share(path) == .12


SLOT = dict(sha256="h", bleu=.2, likelihood=.01, spotting=.02, t_lower=dict(bleu=.01, likelihood=.002, spotting=.003), paired_sd=dict(bleu=.1, likelihood=.004, spotting=.005))
REPORT = dict(metrics=dict(likelihood=dict(mean=-.0276, ceiling=.39, regression=.05, regression_through_origin=.1), spotting=dict(mean=-.0405, ceiling=.9, regression=.03, regression_through_origin=.1)))


def test_the_declaration_restates_the_shares_and_fills_the_power_table_per_rule():
    import copy
    out = c.restate(copy.deepcopy(REPORT), SLOT)
    assert out["metrics"]["likelihood"]["needed_share_at_current_floor"] == pytest.approx((.0276 - .006) / .0276, abs=1e-3) and out["metrics"]["spotting"]["needed_share_at_current_floor"] == pytest.approx((.0405 - .012) / .0405, abs=1e-3)
    assert out["metrics"]["likelihood"]["ceiling_reaches_needed"] is False and out["metrics"]["spotting"]["ceiling_reaches_needed"] is True
    table = out["power_table_at_declaration"]
    assert set(table) == {"hold", "length", "a5_summary_sha256"} and set(table["hold"]) == {"likelihood", "spotting"} and set(table["length"]) == {"likelihood", "spotting", "bleu"}
    assert table["hold"]["likelihood"]["floor"] == -.006 and table["hold"]["likelihood"]["sd"] == .004 and table["hold"]["likelihood"]["rows"][0]["true_change"] == -.0276 and len(table["length"]["bleu"]["rows"]) == 5
    passes = [r["passes"] for r in table["hold"]["likelihood"]["rows"]]
    assert passes == sorted(passes)


def test_no_room_has_no_share_and_no_row():
    import copy
    out = c.restate(copy.deepcopy(REPORT), dict(SLOT, spotting=0.))
    assert out["metrics"]["spotting"]["needed_share_at_current_floor"] is None and out["metrics"]["spotting"]["ceiling_reaches_needed"] is None
    assert out["power_table_at_declaration"]["hold"]["spotting"] is None and out["power_table_at_declaration"]["length"]["spotting"] is None


def test_the_share_a_floor_needs():
    assert c.needed_share(-.0276, -.0077) == pytest.approx(.7210, abs=1e-3) and c.needed_share(-.0405, -.0089) == pytest.approx(.7802, abs=1e-3)
    assert c.needed_share(-.005, -.0077) == 0. and c.needed_share(.01, -.0077) == 0.


def test_the_power_table_rises_with_a_smaller_loss_and_a_smaller_sd():
    rows = c.power_table(.01, -.0077, [-.02, -.0077, 0., .02])
    powers = [r["passes"] for r in rows]
    assert powers == sorted(powers) and powers[0] < .05 and powers[1] == pytest.approx(.025, abs=.002) and powers[3] > .9
    assert c.power_table(.002, -.0077, [0.])[0]["passes"] > c.power_table(.01, -.0077, [0.])[0]["passes"]
    from scipy import stats
    rng = np.random.default_rng(0)
    x = rng.normal(.0, .01, (20000, 8))
    lower = x.mean(1) - stats.t.ppf(.975, 7) * x.std(1, ddof=1) / np.sqrt(8)
    assert c.power_table(.01, -.0077, [0.])[0]["passes"] == pytest.approx((lower > -.0077).mean(), abs=.015)
