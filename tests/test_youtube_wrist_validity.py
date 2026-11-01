"""How much of YouTube's training data has no usable wrist. Offline: synthetic rows only."""
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.round15_compose_eval  # noqa: F401  (puts scripts on the path)
import youtube_wrist_validity as v

ROOT = Path(__file__).resolve().parents[1]


def test_a_side_needs_two_consecutive_valid_frames():
    assert v.usable(np.array([True, True, False])) is True
    assert v.usable(np.array([True, False, True, False])) is False and v.usable(np.array([False, False])) is False and v.usable(np.array([True])) is False


def test_a_corpus_share_is_records_times_weight():
    assert v.weight_share(dict(a=2, b=1, c=1), dict(b=4.), "a") == pytest.approx(2 / 7) and v.weight_share(dict(a=2, b=1), {}, "b") == pytest.approx(1 / 3)


def row(video, left, right, seconds=1., channel="c", chest=1.):
    return dict(video=video, channel=channel, seconds=seconds, chest=chest, body_columns=chest, right_hand=1., left_hand=1., left_usable=left > 0, right_usable=right > 0, left_share=left, right_share=right)


def test_the_shares_are_of_clips_and_seconds():
    rows = [row("a", 0., 0., seconds=3., chest=0.), row("a", 0., 1., seconds=1.), row("b", 1., 0., seconds=1.), row("c", .4, 1., seconds=1.), row("d", 1., 1., seconds=2.)]
    out = v.summarise(rows, boot=50)
    assert out["clips"] == 5 and out["videos"] == 4 and out["no_side_at_all"]["share"] == .2 and out["no_left"]["share"] == .4 and out["no_right"]["share"] == .4 and out["no_either_side"]["share"] == .6
    assert out["no_side_at_all"]["of_seconds"] == .375 and out["under_half_frames_on_either_side"]["share"] == .8 and out["chest_never"]["share"] == .2 and out["bimodal"]["at_zero"] == .2 and out["bimodal"]["at_least_90_both"] == .2
    assert out["body_columns_valid_on_clips_with_no_wrist"] == 0. and out["hands_on_clips_with_no_wrist"] == 1.


def test_the_interval_resamples_videos():
    same = [row("a", 0., 0.)] * 50 + [row("b", 1., 1.)] * 50
    out = v.summarise(same, boot=200)
    assert out["no_side_at_all"]["share"] == .5 and out["no_side_at_all"]["ci95_by_video"] == [0., 1.]            # two videos: the clips of one are not independent


def test_the_committed_report_states_the_finding_and_its_shares():
    d = json.loads((ROOT / "reports/youtube_wrist_validity.json").read_text())
    s = d["sample"]
    assert s["clips"] == 3000 and d["youtube_records"] == d["youtube_records_also_in_tokenizer_mix"] == 99918 and d["round13_youtube_train_records"] < d["youtube_records"]
    assert d["share_of_sampled_weight"]["no_side_at_all"]["generator"] == pytest.approx(d["generator"]["youtube_share_of_sampling"] * s["no_side_at_all"]["share"], abs=1e-3)
    assert d["share_of_sampled_weight"]["no_side_at_all"]["tokenizer"] == pytest.approx(d["tokenizer"]["youtube_share_of_sampling"] * s["no_side_at_all"]["share"], abs=1e-3)
    assert "not measured" in d["not_measured"].lower() or "Neither effect has been measured" in d["not_measured"]
