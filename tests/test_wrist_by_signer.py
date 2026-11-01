"""The wrist-by-signer report: who is read and how the numbers are made. Offline: synthetic records and rows only."""
import json

import numpy as np
import pytest

import scripts.round15_compose_eval  # noqa: F401  (puts scripts on the path)
import wrist_by_signer as w


def record(corpus, partition, sign, signer=None, **extra):
    return dict(corpus=corpus, partition=partition, sign=sign, signer_component=signer, continuous=dict(state_key=f"k/{sign}", state_sha256="s", seconds=4.), text="a b c d", **extra)


def manifest():
    rows = [record(w.H2S, "train", f"a{k}", "c02") for k in range(10)] + [record(w.H2S, "train", f"b{k}", "c04") for k in range(4)] + [record(w.H2S, "val", "v", "c01"), record(w.H2S, "val", "vc02", "c02"), record(w.H2S, "test", "t", "c00")]
    rows += [record(w.YOUTUBE, "train", f"y{k}", source=dict(channel_id=f"ch{k % 3}")) for k in range(30)] + [record(w.YOUTUBE, "monitor", f"ym{k}", source=dict(channel_id="ch9")) for k in range(6)] + [record(w.YOUTUBE, "train", "few", source=dict(channel_id="small"))]
    return dict(records=rows)


def test_how2sign_reads_one_train_signer_and_never_c00():
    rows = w.how2sign_train(manifest(), "c02", 5)
    assert len(rows) == 5 and {r["signer_component"] for r in rows} == {"c02"} and {r["partition"] for r in rows} == {"train"}
    assert rows == w.how2sign_train(manifest(), "c02", 5)            # a fixed draw
    assert len(w.how2sign_train(manifest(), "c04", 300)) == 4 and len(w.how2sign_train(manifest(), "c02", 300)) == 10
    with pytest.raises(SystemExit, match="never read"):
        w.how2sign_train(manifest(), "c00")
    with pytest.raises(SystemExit, match="never read"):
        w.refuse_c00([record(w.H2S, "test", "t", "c00")])


def test_youtube_is_train_by_channel_with_enough_clips():
    picked = w.youtube_by_channel(manifest(), channels=9, per=5)
    assert set(picked) == {"ch0", "ch1", "ch2"} and all(len(v) == 5 for v in picked.values())        # no monitor clip, no channel short of clips
    assert w.youtube_by_channel(manifest(), channels=2, per=5).keys() <= {"ch0", "ch1", "ch2"} and len(w.youtube_by_channel(manifest(), channels=2, per=5)) == 2


def utterance(source, video):
    return dict(collection="c", source=source, utterance_video=f"{video}.mp4")


def test_asllrp_is_train_only_and_never_ben():
    rows = [utterance("Cory_2013-6-25_sc0106", "a"), utterance("Rachel_2013-6-25_sc0106", "b"), utterance("4-Ben-Deaf-Speech", "c"), utterance("Jonathan_2013-6-25_sc0106", "d")]
    gloss = dict(records=[dict(sign=f"c:{v}", partition=p, continuous=dict(state_key="k")) for v, p in (("a", "train"), ("b", "train"), ("c", "test"), ("d", "val"))])
    assert set(w.asllrp_by_signer(gloss, rows)) == {"Cory", "Rachel"}
    gloss["records"][3]["partition"] = "train"
    assert set(w.asllrp_by_signer(gloss, rows)) == {"Cory", "Rachel"}             # Jonathan stays out by the fixed split
    gloss["records"][0]["partition"] = "val"
    assert set(w.asllrp_by_signer(gloss, rows)) == {"Rachel"}
    gloss["records"][2]["partition"] = "train"
    with pytest.raises(SystemExit, match="never read"):
        w.asllrp_by_signer(gloss, rows)


def row(speed, spread=1., fps=24., seconds=4.):
    one = dict(wrist_speed=speed, wrist_spread=spread, left_speed=speed * .8, right_speed=speed * 1.2, left_spread=spread, right_spread=spread)
    return dict(native_fps=fps, frames=int(seconds * fps), seconds=seconds, raw=dict(one), **{"6hz": dict(one, wrist_speed=speed * .7), "3hz": dict(one, wrist_speed=speed * .4)})


def test_a_group_is_read_against_the_reference_at_the_same_rate():
    out = w.group_row([row(2.), row(4.)], [row(1.), row(1.)])
    assert out["clips"] == 2 and out["native_fps_median"] == 24.
    assert out["raw"]["wrist_speed"] == 3. and out["raw"]["wrist_speed_over_d20"] == 3. and out["3hz"]["wrist_speed_over_d20"] == 3. and out["raw"]["left_speed_over_d20"] == 3.
    assert out["raw"]["wrist_speed_over_d20_ci95"][0] >= 2. and out["raw"]["wrist_speed_over_d20_ci95"][1] <= 4.


def test_a_missing_number_stays_out_of_a_mean():
    rows = [row(2.), row(4.)]
    rows[1]["raw"]["left_speed"] = None
    assert w.mean_key(rows, "raw", "left_speed") == 1.6 and w.mean_key([dict(raw=dict(left_speed=None))], "raw", "left_speed") is None


def test_a_corpus_row_gives_every_signer_the_same_weight():
    out = w.equal_weight(dict(many=[row(2.)] * 50, few=[row(4.)]), [row(1.)])
    assert out["groups"] == 2 and out["clips"] == 51 and out["raw"]["wrist_speed_over_d20"] == 3.


def test_the_same_clips_through_both_paths():
    a = [row(1., fps=24., seconds=4.), row(3., fps=24., seconds=8.)]
    m = [row(2., fps=50., seconds=4.), row(6., fps=50., seconds=8.)]
    out = w.paired_row(a, m)
    assert out["two_view_fps"] == 24. and out["monocular_fps"] == 50. and out["seconds_ratio_median"] == 1.
    assert out["raw"]["wrist_speed"]["monocular_over_two_view"] == 2. and out["raw"]["wrist_speed"]["median_per_clip"] == 2. and out["raw"]["wrist_speed"]["pairs"] == 2
    a[0]["raw"]["wrist_speed"] = None
    assert w.paired_row(a, m)["raw"]["wrist_speed"]["pairs"] == 1


def test_the_clock_check_is_seconds_per_token():
    assert w.tokens_rate([record(w.H2S, "train", "a"), record(w.H2S, "train", "b")]) == 1.
    assert w.tokens_rate([dict(text="", continuous=dict(seconds=1.))]) is None


def form(left, right, share=.4, steps=.2):
    regions = dict(core_annotated_one=dict(path_share=1 - share, step_share=1 - steps, speed=1.), bridge_one_to_one=dict(path_share=share / 2, step_share=steps / 2, speed=1.), bridge_other=dict(path_share=share / 2, step_share=steps / 2, speed=1.), speed=1.)
    return dict(ratio_to_real=dict(left=left, right=right), pooled_wrist_ratio=.4 * left + .6 * right, regions=dict(right=regions))


def test_the_carry_arithmetic():
    motion = dict(raw_real_weights=dict(left=.4, right=.6), forms=dict(base=form(2.228, 1.458), hold_x=form(1.791, 1.458), hold_x_length=form(2.03, 1.66), length_tokens=form(2.53, 1.66)))
    out = w.carry_arithmetic(motion)
    assert out["left_weight"] == .4 and out["check_top10"] == pytest.approx(out["stored_top10"], abs=1e-3)
    row = out["rows"]["hold_x"]
    assert row["pooled"] == pytest.approx(.4 * 1.791 + .6 * 1.458, abs=1e-3) and row["right_bridge_path_share"] == .4 and row["right_bridge_step_share"] == .2
    assert row["right_if_bridges_still"] == pytest.approx(1.458 * .6, abs=1e-3) and row["pooled_if_bridges_still"] == pytest.approx(.4 * 1.791 + .6 * 1.458 * .6, abs=1e-3)
    assert row["right_if_bridges_at_real_speed"] == pytest.approx(1.458 * .6 + .2, abs=1e-3)
    assert row["right_needed_for_1_44"] == pytest.approx((1.44 - .4 * 1.791) / .6, abs=1e-3)
    assert out["hold_x_left_cut"] == pytest.approx(1.791 / 2.228, abs=1e-3) and row["pooled_if_right_cut_as_hold_x_cut_left"] == pytest.approx(.4 * 1.791 + .6 * 1.458 * 1.791 / 2.228, abs=1e-3)


def test_the_review_numbers_are_kept_beside_the_measured_ones():
    assert w.REVIEW["raw"]["top10"] == 1.77 and w.REVIEW["low_3hz"]["c04"] == .8 and json.dumps(w.REVIEW)
    assert np.isclose(w.REVIEW["low_3hz"]["top10"], 2.19)


def test_youtube_on_the_monocular_path_is_over_d20s_monocular_row():
    pairs = dict(c01=dict(clips=100, monocular_only=dict(raw=dict(wrist_speed_over_d20=1.), **{"6hz": dict(wrist_speed_over_d20=1.), "3hz": dict(wrist_speed_over_d20=1.)})),
                 c03=dict(clips=300, monocular_only=dict(raw=dict(wrist_speed_over_d20=2.), **{"6hz": dict(wrist_speed_over_d20=1.), "3hz": dict(wrist_speed_over_d20=1.)})), c05=dict(clips=0))
    out = w.path_matched(pairs, dict(raw=dict(wrist_speed_over_d20=3.5), **{"6hz": dict(wrist_speed_over_d20=2.), "3hz": dict(wrist_speed_over_d20=1.)}))
    assert out["raw"]["d20_monocular_over_d20_two_view"] == 1.75 and out["raw"]["youtube_over_d20_monocular"] == 2. and out["6hz"]["youtube_over_d20_monocular"] == 2.
