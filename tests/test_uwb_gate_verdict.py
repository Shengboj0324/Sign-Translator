"""UWB gate verdict: every declared row, the overall rule, and agreement with the gate's proposal_check."""
import copy
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import uwb_gate_verdict as gv  # noqa: E402
import uwb_lifter_gate as gate  # noqa: E402

PRIMARY = "lifter[inside]"
MEDIANS = dict(upper_arms=8., forearms=11., wrists=12., fingers=10.)


def stats(median, n=1000, p90=None, ci=.5):
    return dict(n=n, median=median, p75=median + 2, p90=median * 2 if p90 is None else p90, mean=median + 1,
                median_ci95_by_video=[median - ci, median + ci])


def stratum(pairs, **medians):
    return dict(pairs=pairs, common={g: stats(medians.get(g, MEDIANS[g])) for g in gv.GROUPS})


def synthetic_report(**changes):
    """A gate report whose primary method passes every row; ``changes`` edit the summary before proposal_check."""
    agreement = lambda a, u: dict(agreement=a, uwb_only_rate=u, frames=1000, kappa=.6)  # noqa: E731
    method = dict(
        common={g: stats(m) for g, m in MEDIANS.items()},
        own={"upper_arms": stats(9.), "forearms": stats(12.)}, rotated={"upper_arms": stats(8.), "forearms": stats(10.5)},
        hand_valid={"all.Left": agreement(.92, .04), "all.Right": agreement(.9, .05)},
        forearm_valid_where_our_chest_is={"Left": agreement(.9, .06), "Right": agreement(.87, .08)},
        forearm_survival={}, mirror_check=dict(clips=400, mirror_better=4, median_of_clip_median_arm_error=9.,
                                               median_of_clip_median_arm_error_if_mirrored=60.))
    summary = dict(pairs=1500, videos=300, primary=PRIMARY, methods={PRIMARY: method, "pseudo": copy.deepcopy(method)},
                   strata=dict(hips=dict(visible=stratum(900), not_visible=stratum(600)),
                               fps30=dict(yes=stratum(1100), no=stratum(400)),
                               stream_match=dict(yes=stratum(1200), no=stratum(300)),
                               fc_diff={"0": stratum(800), "-1": stratum(650), "other": stratum(50, forearms=40.)}),
                   alignment=dict(methods={"arm-2d": 1400, "none": 100}, within_1_share=.96, lags={}),
                   lag_drift_frames_per_minute={"a": .2, "b": .1}, inputs={"pose": dict(max_p99_abs_z=3.2)})
    for path, value in changes.items():
        target = summary
        keys = path.split("/")
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = value
    summary["proposal_check"] = gate.proposal_check(summary, PRIMARY)
    return dict(settings=dict(visibility="inside", pairs_per_video=5, bootstrap=2000, videos=0,
                              candidate_visibility="learned,inside,margin:0.1,margin:0.25"),
                regressors_sha256=gv.BUNDLE_SHA256, excluded_videos=[f"dev{k}" for k in range(70)], summary=summary)


def h2s_report(scale=1.25, n=1000, videos=90, sha=gv.BUNDLE_SHA256, primary=PRIMARY,
               candidates=("learned", "inside", "margin:0.1", "margin:0.25")):
    """How2Sign-val medians: the YouTube medians / ``scale``."""
    common = {g: dict(n=n, videos=videos, median=round(m / scale, 3), median_ci95_by_video=[1., 2.])
              for g, m in MEDIANS.items()}
    return dict(regressors_sha256=sha, visibility=primary[7:-1], candidate_visibility=list(candidates),
                summary=dict(primary=primary, methods={primary: dict(common=common)}))


def outside_report(kept=(.5, .5), mediapipe=(.55, .52), chosen="inside", outside_frames=1000, videos=20):
    """Per side, ``survival`` on every bin (all, and the three outside-frame bins of ``outside_frames`` each)."""
    def cells(v):
        bins = {b: dict(frames=outside_frames, survival=v, kept=dict(n=round(outside_frames * v), videos=videos))
                for b in gv.OUTSIDE_BINS}
        return dict(all=dict(frames=3 * outside_frames, survival=v), **bins)
    by = {name: {side: {"Forearm": cells(v[k])} for k, side in enumerate(("Left", "Right"))}
          for name, v in (("inside", kept), ("mediapipe", mediapipe))}
    return dict(regressors_sha256=gv.BUNDLE_SHA256, selection=dict(chosen=chosen), by_decision=by)


def boot(lifted=3.8, ours=4., blank=1.9, lifted_blank_low=.5, ours_blank_low=.8):
    return dict(reads=dict(lifted=dict(bleu=lifted), ours=dict(bleu=ours), blank=dict(bleu=blank)),
                differences=dict(lifted_minus_blank=dict(ci95=[lifted_blank_low, 9.]),
                                 ours_minus_blank=dict(ci95=[ours_blank_low, 9.]),
                                 lifted_minus_ours=dict(ci95=[-1., 1.])))


def downstream_report(pooled=None, dev=None, dev_pairs=300, dev_videos=80, refused=0, union=2300, sha="g" * 64,
                      bundle=gv.BUNDLE_SHA256, visibility="inside", bootstrap=2000, recorded="pass"):
    pooled, dev = pooled or boot(), dev or boot()
    return dict(gate_report_sha256=sha, regressors_sha256=bundle, visibility=visibility, bootstrap=bootstrap,
                **pooled, pairs=2000, dev=dict(dev, pairs=dev_pairs, videos_scored=dev_videos),
                sample=dict(sampled=2000, dev_sampled=dev_pairs, union=union, refused=refused),
                verdict=dict(status=recorded))


def verdict_of(report, **inputs):
    kw = dict(h2s=h2s_report(), downstream=downstream_report(), outside=outside_report(), gate_sha="g" * 64)
    kw.update(inputs)
    rows = gv.build_rows(report, **kw)
    check = gv.cross_check(rows, report["summary"])
    return rows, gv.overall(rows, check), check


def by_id(rows):
    return {r["id"]: r for r in rows}


# ----------------------------------------------------------------------------------------------------------------------

def test_the_declared_bars_are_the_gates_proposal():
    for key in ("median", "ci_slack", "p90", "rotation_gain", "stratum_pairs", "stratum_slack", "hand_agreement",
                "forearm_agreement", "uwb_only", "mirror_share", "lag_within_1", "drift_share", "input_z"):
        assert gv.BARS[key] == gate.PROPOSAL[key], key
    assert gv.BARS["relative"] == 1.6 and gv.BARS["chest_up_survival"] == .8 and gv.BARS["downstream_ratio"] == .9
    assert gv.GROUPS == gate.GROUPS and gv.GATING_STRATA == {"hips": ("visible", "not_visible"),
                                                               "fps30": ("yes", "no"), "fc_diff": ("0", "-1")}


def test_a_passing_report_passes_every_row_and_reproduces_proposal_check():
    report = synthetic_report()
    rows, overall, check = verdict_of(report)
    assert overall == "PASS", [r for r in rows if r["gating"] and r["status"] != "pass"]
    ids = by_id(rows)
    for g, bar in gv.BARS["median"].items():
        assert ids[f"median.{g}<={bar}"]["value"] == MEDIANS[g]
        assert ids[f"median_ci_upper.{g}<={bar + 1}"]["value"] == MEDIANS[g] + .5
        assert ids[f"relative.{g}<=1.6x"]["value"] == pytest.approx(1.25, abs=1e-3)
    assert ids["rotation_gain.forearms<=3.0"]["value"] == 1.5 and ids["uwb_only_rate.Right<=0.1"]["value"] == .08
    assert ids["chest_up_forearm_survival.Left>=0.8x_mediapipe"]["value"] == pytest.approx(.5 / .55, abs=1e-12)
    assert ids["downstream_back_translator"]["value"] == pytest.approx(.95)
    # the non-gating stratum fc_diff=other (50 pairs, forearms 40) is not judged
    assert ids["stratum.fc_diff=other"]["status"] == "not-evaluable" and not ids["stratum.fc_diff=other"]["gating"]
    # every boolean of proposal_check is recomputed and agrees
    proposal = report["summary"]["proposal_check"]
    booleans = [k for k, v in proposal.items() if isinstance(v, bool) or v is None]
    assert check["disagreements"] == [] and check["not_recomputed"] == []
    assert check["compared"] == len(booleans) == 22 and check["declared_differences"] == []
    assert all(r["status"] == "reported" for r in rows if r["check"] == "Diagnostics")


def test_the_verdict_reproduces_proposal_checks_booleans_when_rows_fail():
    report = synthetic_report(**{
        f"methods/{PRIMARY}/common/forearms": stats(19.),
        f"methods/{PRIMARY}/hand_valid/all.Left": dict(agreement=.8, uwb_only_rate=.15, frames=10, kappa=.1),
        f"methods/{PRIMARY}/mirror_check": dict(clips=100, mirror_better=9),
        "alignment/within_1_share": .7, "strata/hips/visible": stratum(900, wrists=19.)})
    rows, overall, check = verdict_of(report)
    proposal = report["summary"]["proposal_check"]
    assert proposal["median.forearms<=18.0"] is False and proposal["hand_agreement.Left>=0.88"] is False
    assert check["disagreements"] == [] and overall == "FAIL"
    ids = by_id(rows)
    for key in ("median.forearms<=18.0", "hand_agreement.Left>=0.88", "uwb_only_rate.Left<=0.1",
                "mirror_better_share<=0.03", "lag_within_1_share>=0.9", "strata_within_3deg_of_bar"):
        assert ids[key]["status"] == "fail", key
    assert ids["strata_within_3deg_of_bar"]["detail"]["failures"] == ["hips=visible:wrists=19.0"]


def test_a_tampered_proposal_check_is_flagged():
    report = synthetic_report()
    report["summary"]["proposal_check"]["p90.forearms<=40.0"] = False
    _, _, check = verdict_of(report)
    assert check["disagreements"] == [dict(key="p90.forearms<=40.0", proposal_check=False, recomputed=True)]


def test_stream_mismatch_and_drift_are_reported_not_gated_and_listed_as_declared_differences():
    report = synthetic_report(**{"strata/stream_match/no": stratum(300, forearms=25.),
                                 "lag_drift_frames_per_minute": {"a": 3., "b": 2., "c": .1}})
    rows, overall, check = verdict_of(report)
    assert overall == "PASS"
    assert report["summary"]["proposal_check"]["strata_within_3deg_of_bar"] is False
    assert check["disagreements"] == []                   # recomputed by the proposal's own reading
    keys = {d["key"] for d in check["declared_differences"]}
    assert keys == {"strata_within_3deg_of_bar", "drift_below_1_share>=0.95"}
    ids = by_id(rows)
    assert ids["stratum.stream_match=no"]["status"] == "fail" and not ids["stratum.stream_match=no"]["gating"]
    assert ids["strata_within_3deg_of_bar"]["status"] == "pass"


def test_missing_or_mismatched_inputs_leave_rows_not_evaluable_and_prevent_pass():
    report = synthetic_report()
    for inputs, row_id in ((dict(downstream=None), "downstream_back_translator"),
                           (dict(downstream=downstream_report(sha="x" * 64)), "downstream_back_translator"),
                           (dict(downstream=downstream_report(pooled=boot(ours_blank_low=-.1))),
                            "downstream_back_translator"),
                           (dict(h2s=None), "relative.fingers<=1.6x"),
                           (dict(h2s=h2s_report(sha="f" * 64)), "relative.wrists<=1.6x"),
                           (dict(h2s=h2s_report(n=150)), "relative.upper_arms<=1.6x"),
                           (dict(h2s=h2s_report(videos=4)), "relative.forearms<=1.6x"),
                           (dict(outside=None), "chest_up_forearm_survival.Left>=0.8x_mediapipe"),
                           (dict(outside=outside_report(chosen="learned")), "chest_up_forearm_survival.Right>=0.8x_mediapipe")):
        rows, overall, _ = verdict_of(report, **inputs)
        assert by_id(rows)[row_id]["status"] == "not-evaluable", (inputs, row_id)
        assert overall == "NOT-EVALUABLE", inputs
    rows, overall, _ = verdict_of(report, downstream=downstream_report(pooled=boot(lifted=2.8)))
    assert overall == "FAIL"                                # a failing row decides, whatever else is missing
    rows, overall, _ = verdict_of(report, downstream=None, h2s=h2s_report(scale=1.))
    assert overall == "NOT-EVALUABLE"


def test_relative_and_chest_up_rows_fail_on_their_bars():
    report = synthetic_report()
    rows, overall, _ = verdict_of(report, h2s=h2s_report(scale=1.7))
    assert overall == "FAIL" and by_id(rows)["relative.forearms<=1.6x"]["status"] == "fail"
    rows, overall, _ = verdict_of(report, h2s=h2s_report(scale=1.6))
    assert by_id(rows)["relative.forearms<=1.6x"]["status"] == "pass"                  # exactly 1.6x
    rows, overall, _ = verdict_of(report, outside=outside_report(kept=(0., .5), mediapipe=(.286, .52)))
    ids = by_id(rows)
    left = ids["chest_up_forearm_survival.Left>=0.8x_mediapipe"]
    assert left["status"] == "fail" and overall == "FAIL" and "fails by construction" in left["note"]
    assert ids["chest_up_forearm_survival.Right>=0.8x_mediapipe"]["status"] == "pass"
    rows, _, _ = verdict_of(report, outside=outside_report(mediapipe=(0., .5)))
    assert by_id(rows)["chest_up_forearm_survival.Left>=0.8x_mediapipe"]["status"] == "not-evaluable"


def test_a_chest_up_side_is_measured_only_on_200_mediapipe_frames_from_5_videos():
    """Review of d2591fa: the real Right side kept 0.0233 against MediaPipe's 0.0069, about 20 frames: 3.4x 'passed'."""
    report = synthetic_report()
    rows, overall, _ = verdict_of(report, outside=outside_report(kept=(.5, .0233), mediapipe=(.55, .0069)))
    right = by_id(rows)["chest_up_forearm_survival.Right>=0.8x_mediapipe"]
    assert right["status"] == "not-evaluable" and right["value"] is None and overall == "NOT-EVALUABLE"
    assert right["detail"]["mediapipe_outside_frames"] == 21 and "unmeasured" in right["detail"]["reason"]
    rows, _, _ = verdict_of(report, outside=outside_report(videos=4))
    assert by_id(rows)["chest_up_forearm_survival.Left>=0.8x_mediapipe"]["status"] == "not-evaluable"
    rows, _, _ = verdict_of(report, outside=outside_report(outside_frames=67))       # 3 x 34 = 102 kept: too few
    assert by_id(rows)["chest_up_forearm_survival.Left>=0.8x_mediapipe"]["status"] == "not-evaluable"
    rows, _, _ = verdict_of(report, outside=outside_report(outside_frames=134))      # 3 x 67 = 201 kept
    assert by_id(rows)["chest_up_forearm_survival.Left>=0.8x_mediapipe"]["status"] == "pass"


def test_setup_rows_check_the_declared_sample():
    report = synthetic_report()
    report["settings"]["pairs_per_video"] = 12
    rows, overall, _ = verdict_of(report)
    assert by_id(rows)["pairs_per_video"]["status"] == "fail" and overall == "FAIL"
    report = synthetic_report()
    report["settings"]["visibility"] = "learned"
    rows, overall, _ = verdict_of(report)
    assert by_id(rows)["visibility_is_selected_rule"]["status"] == "fail" and overall == "FAIL"


def test_main_writes_the_table_and_the_json(tmp_path):
    report = synthetic_report()
    paths = {}
    for name, data in (("gate", report), ("h2s", h2s_report()), ("outside", outside_report())):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(data))
    import hashlib
    sha = hashlib.sha256(paths["gate"].read_bytes()).hexdigest()
    paths["ds"] = tmp_path / "ds.json"
    paths["ds"].write_text(json.dumps(downstream_report(sha=sha)))
    out, md = tmp_path / "verdict.json", tmp_path / "verdict.md"
    assert gv.main(["--store", "none", "--gate-report", str(paths["gate"]), "--how2sign-val", str(paths["h2s"]),
                    "--downstream", str(paths["ds"]), "--outside-frame", str(paths["outside"]), "--out", str(out),
                    "--markdown", str(md)]) == 0
    written = json.loads(out.read_text())
    assert written["verdict"] == "PASS" and written["inputs"]["gate_report"][1] == sha
    assert md.read_text().startswith("**Overall: PASS**") and "`downstream_back_translator`" in md.read_text()
    assert gv.main(["--store", "none", "--gate-report", str(paths["gate"]), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == "NOT-EVALUABLE"


def test_the_relative_row_needs_the_gates_primary_and_candidate_rules():
    report = synthetic_report()
    h2s = h2s_report(primary="lifter[learned]")
    rows, overall, _ = verdict_of(report, h2s=h2s)
    row = by_id(rows)["relative.forearms<=1.6x"]
    assert row["status"] == "not-evaluable" and "primary" in row["detail"]["reason"] and overall == "NOT-EVALUABLE"
    rows, overall, _ = verdict_of(report, h2s=h2s_report(candidates=("inside",)))
    row = by_id(rows)["relative.forearms<=1.6x"]
    assert row["status"] == "not-evaluable" and "candidate" in row["detail"]["reason"] and overall == "NOT-EVALUABLE"
    rows, overall, _ = verdict_of(report, h2s=h2s_report(candidates=("margin:0.25", "margin:0.1", "inside", "learned")))
    assert overall == "PASS"                                                # the same rules in another order


def test_the_downstream_row_is_recomputed_never_trusted():
    report = synthetic_report()
    rows, overall, _ = verdict_of(report, downstream=downstream_report(pooled=boot(lifted=2.), recorded="pass"))
    row = by_id(rows)["downstream_back_translator"]
    assert row["status"] == "fail" and row["detail"]["recorded_differs"] and overall == "FAIL"
    rows, overall, _ = verdict_of(report, downstream=downstream_report(recorded="fail"))
    assert by_id(rows)["downstream_back_translator"]["status"] == "pass" and overall == "PASS"
    # the UWB dev partition must pass too, on >= 200 pairs from >= 50 videos
    rows, overall, _ = verdict_of(report, downstream=downstream_report(dev=boot(lifted=3.)))
    assert by_id(rows)["downstream_back_translator"]["status"] == "fail" and overall == "FAIL"
    for kw in (dict(dev_pairs=199), dict(dev_videos=49), dict(refused=231, union=2300)):
        rows, overall, _ = verdict_of(report, downstream=downstream_report(**kw))
        assert by_id(rows)["downstream_back_translator"]["status"] == "not-evaluable", kw
        assert overall == "NOT-EVALUABLE"
    rows, _, _ = verdict_of(report, downstream=downstream_report(refused=230, union=2300))   # exactly 10%
    assert by_id(rows)["downstream_back_translator"]["status"] == "pass"
    for kw in (dict(bundle="f" * 64), dict(visibility="learned"), dict(bootstrap=200), dict(sha="h" * 64)):
        rows, overall, _ = verdict_of(report, downstream=downstream_report(**kw))
        assert by_id(rows)["downstream_back_translator"]["status"] == "not-evaluable", kw


def test_values_are_compared_unrounded_where_the_inputs_carry_them():
    # 100 / 3329 = 0.030039 > 0.03 (rounded to 4 dp it would read 0.03)
    report = synthetic_report(**{f"methods/{PRIMARY}/mirror_check": dict(clips=3329, mirror_better=100)})
    rows, overall, check = verdict_of(report)
    assert by_id(rows)["mirror_better_share<=0.03"]["status"] == "fail" and overall == "FAIL"
    assert check["disagreements"] == []
    # agreement 0.879960 from the counts; the summary's 4-dp value is 0.88: proposal_check passes, the row fails
    cells = dict(both=21999, ours_only=1500, uwb_only=1501, neither=0)
    left = dict(gate.agreement_summary(cells))
    assert left["agreement"] == .88
    report = synthetic_report(**{f"methods/{PRIMARY}/hand_valid/all.Left": left})
    rows, overall, check = verdict_of(report)
    row = by_id(rows)["hand_agreement.Left>=0.88"]
    assert row["status"] == "fail" and row["value"] == pytest.approx(21999 / 25000)
    assert report["summary"]["proposal_check"]["hand_agreement.Left>=0.88"] is True
    assert check["disagreements"] == [] and overall == "FAIL"
    assert any(d["key"] == "hand_agreement.Left>=0.88" for d in check["declared_differences"])
    # the lag share from the report's pairs, not the summary's 4-dp value
    report = synthetic_report(**{"alignment/within_1_share": .9})
    report["pairs"] = [dict(status="evaluated", alignment=dict(method="arm-2d", lag=0 if k < 8999 else 3))
                       for k in range(10000)]
    rows, _, _ = verdict_of(report)
    assert by_id(rows)["lag_within_1_share>=0.9"]["status"] == "fail"
    assert by_id(rows)["median.forearms<=18.0"]["rounded_source"]           # medians exist only rounded


def test_any_disagreement_with_proposal_check_makes_the_verdict_not_evaluable(tmp_path):
    report = synthetic_report()
    report["summary"]["proposal_check"]["median.forearms<=18.0"] = False
    rows, overall, check = verdict_of(report)
    assert check["disagreements"] and overall == "NOT-EVALUABLE"
    report["summary"]["proposal_check"]["new_item"] = True
    _, _, check = verdict_of(report)
    assert check["not_recomputed"] == ["new_item"]
    assert "not recomputed: new_item" in gv.markdown(rows, "NOT-EVALUABLE", check)
