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


def mono_row(value):
    one = {f"{k}_over_d20": value for k in w.KEYS}
    return dict(raw=dict(one), **{"6hz": dict(one), "3hz": dict(one)})


def test_composition_is_read_against_d20_through_the_monocular_path():
    pairs = dict(c01=dict(clips=100, monocular_only=mono_row(1.)), c03=dict(clips=300, monocular_only=mono_row(2.)), c05=dict(clips=0))
    composition = {n: mono_row(3.5) for n in ("base", "hold_x", "length_tokens", "hold_x_length")}
    out = w.composition_on_monocular(pairs, composition)
    assert out["d20_monocular_over_d20_two_view"]["raw"]["wrist_speed"] == 1.75 and out["base"]["raw"]["left_speed"] == 2. and out["hold_x"]["3hz"]["right_speed"] == 2.


def test_per_side_rows_are_monocular_over_two_view():
    row = {r: {k: dict(monocular_over_two_view=1.2) for k in ("left_speed", "right_speed", "left_spread", "right_spread")} for r in w.RATES}
    out = w.per_side_rows(dict(c01=dict(clips=3, **row), c05=dict(clips=0)))
    assert set(out) == {"c01"} and out["c01"]["3hz"]["left_speed"] == 1.2 and out["c01"]["raw"]["right_speed"] == 1.2 and out["c01"]["6hz"]["left_spread"] == 1.2 and out["c01"]["3hz"]["right_spread"] == 1.2


def test_the_word_path_answer_is_computed_and_states_what_is_not_measured():
    row = dict(one_word_weight_two_view_share=.0027, sampling_two_view_share=.088, sentence_token_weight_two_view_share=.2721, one_word_weight_by_corpus={w.H2S: 78., "rit-text-r6": 28766.}, tokenizer=dict(corpus_weight=[f"{w.H2S}=2"]))
    text = w.path_answer(row)
    for phrase in ("0.27%", "8.8%", "27.2%", "How2Sign's weight 78 of 28844", "decoder's share is not measured", "weights How2Sign x2", "No generated clip's amplitude was tested", "is an inference"):
        assert phrase in text, phrase
    assert "every corpus" not in text and "0.50%" in w.path_answer(dict(row, one_word_weight_two_view_share=.005))
    assert w.with_answer(row)["answer"] == text


def test_the_note_states_both_directions_and_its_limits():
    for phrase in ("monocular inflation fits the left-arm asymmetry at least as well as two-view shrinkage", "Per side it is not every signer", "1.00-1.05x", "not called jitter", "5 clips each", "0.27%", "8.8%", "27.2%", "That is an inference", "The plan records the span on main"):
        assert phrase in w.NOTE, phrase
    assert "jitter is filtered" not in w.NOTE


def channel_rows():
    rows = [record(w.YOUTUBE, "train", f"a{k}", source=dict(channel_id="big")) for k in range(40)] + [record(w.YOUTUBE, "train", f"b{k}", source=dict(channel_id="exact")) for k in range(30)]
    rows += [record(w.YOUTUBE, "train", f"c{k}", source=dict(channel_id="short")) for k in range(29)] + [record(w.YOUTUBE, "monitor", f"m{k}", source=dict(channel_id="monitor")) for k in range(40)]
    rows += [record(w.YOUTUBE, "train", f"n{k}", source={}) for k in range(40)]
    return dict(records=rows)


def test_a_channel_qualifies_by_train_clips_and_is_ordered_by_a_fixed_shuffle():
    qualifying = w.channels_with_clips(channel_rows())
    assert set(qualifying) == {"big", "exact"} and len(qualifying["big"]) == 40
    drawn = w.span_draw(qualifying)
    assert set(drawn) == {"big", "exact"} and len(drawn["big"]) == 40 and drawn == w.span_draw(qualifying) and drawn["big"] != qualifying["big"]
    assert len(w.span_draw(qualifying, channels=1)) == 1


NULL = dict(wrist_speed=None, wrist_spread=None, left_speed=None, right_speed=None, left_spread=None, right_spread=None)


def null_row():
    return dict(native_fps=24., frames=1, seconds=1., raw=dict(NULL), **{"6hz": dict(NULL), "3hz": dict(NULL)})


def test_a_group_of_nulls_cannot_count_toward_the_thirty():
    rows = [null_row() for _ in range(30)]
    row = w.group_row(rows, rows)
    assert row["clips"] == 30 and row["raw"]["numbered"]["wrist_speed"] == 0 and w.counted(row) is False and w.enough(rows) is False


def test_one_side_short_is_not_enough():
    rows = [row(1.) for _ in range(30)]
    assert w.enough(rows) is True and w.counted(w.group_row(rows, rows)) is True
    rows[0]["raw"]["left_speed"] = None
    assert w.enough(rows) is False and w.counted(w.group_row(rows, rows)) is False
    rows = [row(1.) for _ in range(30)]
    rows[0]["3hz"]["right_speed"] = None
    assert w.enough(rows) is False and w.counted(w.group_row(rows, rows)) is False


def test_reading_goes_on_until_thirty_numbered_or_the_records_run_out():
    rows, seen = w.read_numbered([null_row() for _ in range(5)] + [row(1.) for _ in range(40)], lambda r: r)
    assert seen == 35 and len(rows) == 30                                        # nulls add nothing, and reading stops at thirty
    rows, seen = w.read_numbered([null_row() for _ in range(10)] + [row(1.) for _ in range(5)], lambda r: r)
    assert seen == 15 and len(rows) == 5
    rows, seen = w.read_numbered([null_row() for _ in range(7)], lambda r: r)
    assert seen == 7 and rows == []


def test_going_on_looks_at_validity_never_at_speed():
    slow = w.read_numbered([row(.001) for _ in range(40)], lambda r: r)[1]
    fast = w.read_numbered([row(99.) for _ in range(40)], lambda r: r)[1]
    assert slow == fast == 30


def speed_row(left, right, pooled=None):
    rate = dict(wrist_speed=(left + right) / 2 if pooled is None else pooled, left_speed=left, right_speed=right)
    return dict(raw=dict(rate, numbered={k: 30 for k in rate}), **{"3hz": dict({k: v / 2 for k, v in rate.items()}, numbered={k: 30 for k in rate})})


def composed(value):
    return {r: {f"{k}_over_d20": value for k in w.SIDES.values()} for r in w.SPAN_RATES}


def span_input(value=1., n=20):
    groups = {f"g{k}": dict(corpus=w.YOUTUBE if k > 2 else "asllrp", named=k <= 2, clips_read=30, row=speed_row(.5 + k * .1, .5 + k * .1)) for k in range(n)}
    forms = {name: composed(value) for name in ("base", "hold_x", "length_tokens", "hold_x_length")}
    reference = {r: dict(wrist_speed=1., left_speed=1., right_speed=1.) for r in w.SPAN_RATES}
    return groups, dict(composition=forms, on_monocular={k: {r: dict(wrist_speed=.9, left_speed=.9, right_speed=.9) for r in w.SPAN_RATES} for k in forms}, reference=reference, d20=dict(c01=speed_row(1., 1.), c05=speed_row(9., 9.)), length_ratio=1.2, channels_qualifying=n, channels_read=n)


def test_the_span_is_p10_to_p90_across_counted_groups_per_rate_and_side():
    groups, kw = span_input()
    out = w.real_signer_span(groups, **kw)
    values = [.5 + k * .1 for k in range(20)]
    assert out["span"]["raw"]["left"] == [round(float(np.percentile(values, 10)), 5), round(float(np.percentile(values, 90)), 5)]
    assert out["span"]["3hz"]["right"] == [round(float(np.percentile(values, 10)) / 2, 5), round(float(np.percentile(values, 90)) / 2, 5)]
    assert out["units"] == "shoulder widths per second" and out["groups_counted"] == 20 and out["named_counted"] == 3 and out["channels_counted"] == 17 and out["refused"] == {} and out["channels"]["g5"]["clips_read"] == 30


def test_a_group_under_thirty_numbered_is_refused_named_and_in_no_percentile():
    groups, kw = span_input()
    groups["g19"]["row"]["raw"]["numbered"]["left_speed"] = 29
    groups["g19"]["clips_read"] = 400
    out = w.real_signer_span(groups, **kw)
    assert set(out["refused"]) == {"g19"} and out["refused"]["g19"]["clips_read"] == 400 and out["refused"]["g19"]["numbered"]["raw"]["left"] == 29 and "g19" not in out["channels"] and out["groups_counted"] == 19
    values = [.5 + k * .1 for k in range(19)]
    assert out["span"]["raw"]["left"][1] == round(float(np.percentile(values, 90)), 5)


def test_named_signers_stand_apart_with_their_passes_and_d20_is_placed():
    groups, kw = span_input()
    out = w.real_signer_span(groups, **kw)
    assert set(out["named_signers"]) == {"g0", "g1", "g2"} and "g0" not in out["channels"] and out["named_signers"]["g1"]["numbered"]["raw"]["left"] == 30
    low = out["span"]["raw"]["left"][0]
    assert out["named_signers"]["g0"]["speed"]["raw"]["left"] < low and out["named_signers"]["g0"]["passes_of_six"] < 6
    assert out["d20_position"]["c01"]["passes_of_six"] == w.passes_of(w.against_span(w.group_speeds(speed_row(1., 1.)), out["span"])) and out["d20_position"]["c05"]["passes_of_six"] == 0
    assert out["sensitivity"]["named_signers_only"]["raw"]["left"][1] < out["span"]["raw"]["left"][1]


def test_top10_is_placed_among_the_groups_and_scaled_roughly():
    groups, kw = span_input(value=1.)
    out = w.real_signer_span(groups, **kw)
    base = out["forms"]["base"]
    assert base["speed"]["raw"]["left"] == 1. and base["percentile_among_groups"]["raw"]["left"] == .25 and base["faster_than_named_signers"]["raw"]["left"] == 3 and base["old_reading"]["over_d20_monocular"]["raw"]["left"] == .9 and base["old_reading"]["over_d20_two_view"]["3hz"]["right"] == 1.
    rough = out["true_length_rough"]
    assert rough["speed"]["raw"]["pooled"] == 1.2 and "rough" in rough["note"] and rough["length_ratio"] == 1.2 and rough["inside"]["raw"]["pooled"] is True


def test_inside_is_at_or_between_the_ends_and_passes_are_counted():
    span = dict(raw=dict(pooled=[1., 2.], left=[1., 2.], right=[1., 2.]), **{"3hz": dict(pooled=[1., 2.], left=[1., 2.], right=[1., 2.])})
    form = {r: dict(pooled=1., left=2., right=1.5) for r in w.SPAN_RATES}
    assert w.passes_of(w.against_span(form, span)) == 6
    form["raw"]["left"] = 2.01
    form["3hz"]["pooled"] = .99
    assert w.passes_of(w.against_span(form, span)) == 4


def test_each_side_reads_its_own_number_and_a_form_is_made_absolute_by_the_reference():
    row_ = speed_row(2., 4., pooled=3.5)
    assert w.group_speeds(row_) == dict(raw=dict(pooled=3.5, left=2., right=4.), **{"3hz": dict(pooled=1.75, left=1., right=2.)})
    form = {r: dict(wrist_speed_over_d20=2., left_speed_over_d20=3., right_speed_over_d20=4.) for r in w.SPAN_RATES}
    reference = {r: dict(wrist_speed=.5, left_speed=.25, right_speed=.1) for r in w.SPAN_RATES}
    assert w.absolute_form(form, reference)["raw"] == dict(pooled=1., left=.75, right=.4)


def test_few_channels_are_said_on_counted_channels():
    groups, kw = span_input(n=22)
    assert w.real_signer_span(groups, **kw)["few_channels"] is True                # 19 channels counted, whatever qualified by clips
    groups, kw = span_input(n=23)
    assert w.real_signer_span(groups, **kw)["few_channels"] is False


def test_the_corpus_row_refuses_empty_groups_and_names_them():
    good = [row(2.) for _ in range(5)]
    out = w.equal_weight(dict(a=good, empty=[null_row() for _ in range(5)]), [row(1.)], minimum=5)
    assert out["groups"] == 1 and out["groups_read"] == 2 and out["refused"] == ["empty"] and out["raw"]["wrist_speed_over_d20"] == 2. and out["clips"] == 5
    assert w.equal_weight(dict(a=good[:4]), [row(1.)], minimum=5)["refused"] == ["a"]
    assert w.equal_weight(dict(a=good), [row(1.)])["groups"] == 1


def test_the_earlier_read_says_how_many_channels_had_no_number():
    out = w.hole_in_earlier_read(dict(a=[row(2.)], b=[row(4.)], empty=[null_row(), null_row()]), dict(raw=dict(wrist_speed=1.)))
    assert out["channels"] == 3 and out["channels_with_no_number"] == 1 and out["names"] == ["empty"] and out["range_on_the_rest_raw_pooled"][1] <= 4.


def test_the_committed_span_counts_its_clips_and_reads_no_test_partition():
    """This fails on the first span read, whose groups held no counts and 26 of whose channels held no number."""
    import pathlib
    report = json.loads((pathlib.Path(w.__file__).parents[1] / "reports/wrist_by_signer.json").read_text())
    span = report["real_signer_span"]
    for name, row_ in {**span["named_signers"], **span["channels"]}.items():
        assert all(v >= 30 for rate in row_["numbered"].values() for v in rate.values()), name
        assert all(v is not None for rate in row_["speed"].values() for v in rate.values()), name
    names = set(span["named_signers"]) | set(span["channels"]) | set(span["refused"])
    assert "how2sign_c00" not in names and "asllrp_Ben" not in names and "asllrp_Jonathan" not in names and span["channels_counted"] >= 20
    assert "hold_arms" not in span and "top10_inside_all_six" not in span and "groups" not in span and report["youtube"]["hole_check"]["channels"] == 60
    assert '"/test_" in key' in pathlib.Path(w.__file__).read_text()
