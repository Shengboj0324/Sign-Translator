"""Post-S2 A6: the narrower hold, the length fit, the read over word draws and its rules, and the job on stubs. Offline: synthetic reports and stores only; no test starts a real job and nothing is read from R2."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

import scripts.round15_compose_eval as ce
import comp_a5_read as a5
import comp_a6_declared as dc6
import comp_a6_guard as guard
import comp_a6_lengths as lengths
import comp_a6_read as a6
import comp_a6_rest_pose as rest_pose
import comp_a_lib as lib
import comp_a_read as rd
from signtranslator.generation import composition_v2 as cv2
from signtranslator.generation.composition_space import to_6d, wrist_of_rotations
from signtranslator.generation.motion_metrics import _matrices
from tests.test_comp_a import PROVENANCE, READY, REFS, arm_report, body_of, left_indices, oracle_args, cache_for_words, d20, recogniser, write_list, token_rows, part, generated_args, plan_of  # noqa: F401
from tests.test_generation_signing_space import signer_state
from tests.test_round15_stage2 import stage2_store, generator_store  # noqa: F401
from tests.test_round15_compositional import stubbed  # noqa: F401
from tests.test_round20_eval import likelihood_stub  # noqa: F401
from tests.test_round20_phase_b import on_stubs, scripts_run

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "runpod/fleet/comp_a6.sh"
CHAIN = ROOT / "runpod/fleet/comp_a6_chain.sh"
JOBS = tuple(a6.JOB_FILE.split(","))
FORM = a6.FORM
O = "acquisition/2026-10-17-v1/youtube-asl-v1"
OUT = f"{O}/round20/comp-a6"
A5OUT = f"{O}/round20/comp-a5"
DONE = re.compile(r"^COMP-A6-DONE (0|1) phase=\S+ words=\S+ smoke=(0|1)$", re.M)
REST = {j: [float(x) for x in to_6d(np.eye(3))] for j in cv2.HOLD_JOINTS}
L, S, B = a6.LIKELIHOOD_FLOOR or -.0077, a6.SPOTTING_FLOOR or -.0089, a6.BLEU_FLOOR or -.1509      # the declared floors: a metric with no room still has its checks tested


@pytest.fixture(autouse=True)
def room_and_code(monkeypatch):
    """Every metric has room, whatever the slot holds, and the reader's code is the synthetic reports': filling the slot changes no test but the ones about it."""
    monkeypatch.setattr(a6, "LIKELIHOOD_FLOOR", L)
    monkeypatch.setattr(a6, "SPOTTING_FLOOR", S)
    monkeypatch.setattr(a6, "BLEU_FLOOR", B)
    monkeypatch.setattr(a6, "NO_ROOM", {})
    monkeypatch.setattr(lib, "provenance_row", lambda job=None: dict(PROVENANCE))


# ---- the hold with a rest pose ---------------------------------------------------------------------------------------------------------------

def left_matrices(state):
    body = body_of(state)
    return {j: _matrices(body.values[:, k].astype(np.float64)) for j, k in zip(cv2.HOLD_JOINTS, left_indices(state))}


def peak_speed(state, a, b, fps=24.):
    m = left_matrices(state)
    p = wrist_of_rotations(m[cv2.HOLD_JOINTS[0]], m[cv2.HOLD_JOINTS[1]], "Left")
    return float(np.linalg.norm(p[a + 1:b + 1] - p[a:b], axis=-1).max() * fps)


def test_no_rest_is_the_plain_hold_and_its_info():
    state = signer_state(frames=40, amplitude=.2)
    plan = plan_of([10, 10, 10], [5, 5])
    plain, info = cv2.hold_left_arm(state, plan, [False, True, False], ready=READY, cap=99.)
    again, other = cv2.hold_left_arm(state, plan, [False, True, False], ready=READY, cap=99., rest=None)
    np.testing.assert_array_equal(body_of(plain).values, body_of(again).values)
    assert info == other and "rest_frames" not in info


def test_the_arm_goes_to_the_rest_pose_and_back_under_the_cap():
    state = signer_state(frames=60, amplitude=.2)
    plan = plan_of([10, 30, 10], [5, 5])
    out, info = cv2.hold_left_arm(state, plan, [False, True, False], ready=READY, cap=1.2, rest=REST)
    assert info["rest_ramps"] == 1 and info["rest_too_short"] == 0 and info["rest_frames"] == 40 and info["frames"] == 40 and info["peak"] <= 1.2
    m = left_matrices(out)
    for j in cv2.HOLD_JOINTS:
        np.testing.assert_allclose(m[j][29], np.eye(3), atol=2e-2)                       # the middle is at the pose
        np.testing.assert_allclose(m[j][30], np.eye(3), atol=2e-2)
    assert peak_speed(out, 9, 50) <= 1.2 + 1e-6
    body, new = body_of(state), body_of(out)
    keep = [k for k, label in enumerate(body.labels) if label not in cv2.HOLD_JOINTS]
    np.testing.assert_array_equal(body.values[:, keep], new.values[:, keep])            # nothing else moves
    np.testing.assert_array_equal(body.values[:10], new.values[:10])                    # the anchors stay
    np.testing.assert_array_equal(body.values[50:], new.values[50:])


def test_the_ramps_are_the_longest_the_stretch_allows_whatever_the_cap():
    state = signer_state(frames=80, amplitude=.2)
    plan = plan_of([10, 50, 10], [5, 5])
    free, _ = cv2.hold_left_arm(state, plan, [False, True, False], ready=READY, cap=99., rest=REST)
    tight, _ = cv2.hold_left_arm(state, plan, [False, True, False], ready=READY, cap=peak_speed(free, 9, 70) + 1e-3, rest=REST)
    np.testing.assert_array_equal(body_of(free).values, body_of(tight).values)
    m = left_matrices(free)[cv2.HOLD_JOINTS[0]]
    away = np.array([np.linalg.norm(m[i] - np.eye(3)) for i in range(10, 70)])
    assert (np.diff(away[:30]) <= 1e-9).all() and (np.diff(away[30:]) >= -1e-9).all()     # down all the way to the middle and back, with no hold at the pose
    assert (away[:29] > 1e-4).all()


def test_each_rest_stretch_reports_its_peak():
    state = signer_state(frames=60, amplitude=.2)
    _, info = cv2.hold_left_arm(state, plan_of([10, 30, 10], [5, 5]), [False, True, False], ready=READY, cap=9., rest=REST)
    assert len(info["rest_peaks"]) == 1 and info["rest_peaks"][0] == info["peak"]


def test_a_stretch_too_short_for_its_ramps_keeps_the_plain_path():
    state = signer_state(frames=26, amplitude=.2)
    plan = plan_of([10, 4, 10], [1, 1])
    out, info = cv2.hold_left_arm(state, plan, [False, True, False], ready=READY, cap=99., rest=REST)
    plain, _ = cv2.hold_left_arm(state, plan, [False, True, False], ready=READY, cap=99.)
    assert info["rest_too_short"] == 1 and info["rest_ramps"] == 0 and info["rest_frames"] == 0 and info["minimum_jerk"] == 1
    np.testing.assert_array_equal(body_of(out).values, body_of(plain).values)


def test_too_fast_for_any_ramp_keeps_the_plain_path_and_its_cap():
    state = signer_state(frames=60, amplitude=.2)
    plan = plan_of([10, 30, 10], [5, 5])
    _, info = cv2.hold_left_arm(state, plan, [False, True, False], ready=READY, cap=1e-4, rest=REST)
    assert info["rest_too_short"] == 1 and info["rest_ramps"] == 0 and info["constant_speed"] == 1


def test_a_sentence_edge_has_one_ramp_and_no_anchor_rests_throughout():
    state = signer_state(frames=30, amplitude=.2)
    out, info = cv2.hold_left_arm(state, plan_of([15, 15], [0]), [True, False], ready=READY, cap=99., rest=REST)
    assert info["rest_ramps"] == 1
    np.testing.assert_allclose(left_matrices(out)[cv2.HOLD_JOINTS[0]][0], np.eye(3), atol=5e-2)       # it starts at rest
    out, info = cv2.hold_left_arm(state, plan_of([15, 15], [0]), [False, True], ready=READY, cap=99., rest=REST)
    np.testing.assert_allclose(left_matrices(out)[cv2.HOLD_JOINTS[0]][29], np.eye(3), atol=5e-2)      # it ends at rest
    out, info = cv2.hold_left_arm(state, plan_of([15, 15], [0]), [True, True], ready=READY, cap=99., rest=REST)
    assert info["rest_whole"] == 1 and info["rest_frames"] == 30 and info["ready_pose"] == 0
    np.testing.assert_allclose(left_matrices(out)[cv2.HOLD_JOINTS[1]][7], np.eye(3), atol=1e-5)


def test_invalid_frames_stay_as_they_were():
    state = signer_state(frames=60, amplitude=.2)
    body = body_of(state)
    k = left_indices(state)[0]
    valid = body.valid.copy()
    valid[28:31, k] = False
    masked = replace(state, channels=dict(state.channels, body=replace(body, values=np.where(valid[..., None], body.values, 0.), valid=valid, observed=body.observed & valid, confidence=valid.astype(float))))
    out, _ = cv2.hold_left_arm(masked, plan_of([10, 30, 10], [5, 5]), [False, True, False], ready=READY, cap=9., rest=REST)
    np.testing.assert_array_equal(body_of(out).values[28:31, k], body_of(masked).values[28:31, k])
    assert body_of(out).valid[28:31, k].sum() == 0


# ---- which entries are held ------------------------------------------------------------------------------------------------------------------

def clip(two):
    return dict(gloss=dict(handshapes={"Non-dominant start handshape": "B" if two else ""}))


def test_an_entry_with_a_quarter_two_handed_records_is_an_anchor():
    pools = dict(one=[clip(False)] * 4, under=[clip(True)] + [clip(False)] * 4, exactly=[clip(True)] + [clip(False)] * 3, mostly=[clip(True)] * 3 + [clip(False)], none=[dict(gloss={})], new=[clip(False)])
    flags = lib.hold_flags(pools, two_handed_share=.25, new_entries={"new": 1})
    assert flags == dict(one=True, under=True, exactly=False, mostly=False)         # none and new have no label: anchors
    assert lib.entry_annotations(pools)["exactly"] is True                         # the majority read the plain hold used
    assert lib.hold_flags(pools, two_handed_share=.5, new_entries=())["exactly"] is True


def test_the_rest_pose_file_is_pinned_and_committed():
    path = ROOT / "reports/comp_a6_rest_pose.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == a6.REST_SHA
    pose = lib.load_rest_pose(path, a6.REST_SHA)
    assert set(pose) == set(cv2.HOLD_JOINTS)
    with pytest.raises(SystemExit, match="expected"):
        lib.load_rest_pose(path, "0" * 64)
    wrong = ROOT / "reports/comp_a_hold_stats.json"
    with pytest.raises(SystemExit, match="not the rest pose"):
        lib.load_rest_pose(wrong, hashlib.sha256(wrong.read_bytes()).hexdigest())
    payload = json.loads(path.read_text())
    assert "Ben" not in payload["signers"] and "Jonathan" not in payload["signers"] and set(payload["signers"]) == {"Cory", "Rachel", "RIT-P02", "RIT-P03"}


# ---- the rest pose's frames ------------------------------------------------------------------------------------------------------------------

def utterance(source, signs, video="v.mp4", frames=(100, 149)):
    return dict(collection="c", source=source, utterance_video=video, frames=list(frames), signs=signs)


ONE = dict(frames=[110, 119], handshapes={"Dominant start handshape": "G"})
TWO = dict(frames=[120, 129], handshapes={"Dominant start handshape": "G", "Non-dominant start handshape": "B"})
BARE = dict(frames=[130, 139])


def test_only_annotated_one_handed_signs_count():
    assert rest_pose.one_handed_spans(utterance("Cory_x", [ONE, TWO, BARE, dict(ONE, frames=[90, 95]), dict(ONE, frames=[None, None])])) == [(10, 19)]


def test_the_train_partition_only_and_never_ben():
    rows = [utterance("Cory_2013-6-25_sc0106", [ONE], "a.mp4"), utterance("Rachel_2013-6-25_sc0106", [ONE], "b.mp4"), utterance("4-Ben-Deaf-Speech", [ONE], "c.mp4"), utterance("Jonathan_2013-6-25_sc0106", [ONE], "d.mp4"), utterance("Cory_2013-6-25_sc0107", [TWO], "e.mp4")]
    records = {f"c:{v}": dict(sign=f"c:{v}", partition=p, continuous=dict(state_key="k")) for v, p in (("a", "train"), ("b", "train"), ("c", "test"), ("d", "val"), ("e", "train"))}
    out = rest_pose.train_utterances(rows, records)
    assert set(out) == {"Cory", "Rachel"} and [r["sign"] for _, r in out["Cory"]] == ["c:a"]         # no span in e
    records["c:d"]["partition"] = "train"                                                              # Jonathan stays out by the fixed split, whatever a record says
    assert set(rest_pose.train_utterances(rows, records)) == {"Cory", "Rachel"}
    records["c:a"]["partition"] = "val"
    assert set(rest_pose.train_utterances(rows, records)) == {"Rachel"}                               # the manifest's own split also counts
    records["c:c"]["partition"] = "train"
    with pytest.raises(SystemExit, match="never read"):
        rest_pose.train_utterances(rows, records)


def test_a_state_off_its_annotation_places_no_span():
    class S:
        timestamps = np.zeros(50)
    u = utterance("Cory", [ONE])
    assert rest_pose.span_frames(S, u).tolist() == list(range(10, 20))
    S.timestamps = np.zeros(60)
    assert len(rest_pose.span_frames(S, u)) == 0


def test_every_signer_weighs_the_same():
    a, b = np.repeat(np.eye(3)[None], 500, 0), np.repeat(_matrices(np.array([0., 1., 0., -1., 0., 0.]))[None], 10, 0)
    pose = rest_pose.mean_pose({"many": {j: [a] for j in rest_pose.JOINTS}, "few": {j: [b] for j in rest_pose.JOINTS}})
    from scipy.spatial.transform import Rotation
    angle = Rotation.from_matrix(_matrices(np.array(pose[rest_pose.JOINTS[0]])) ).magnitude()
    assert float(angle) == pytest.approx(np.pi / 4, abs=.01)                      # halfway, not pulled to the signer with 50 times the frames
    with pytest.raises(SystemExit, match="no ASLLRP"):
        rest_pose.mean_pose({})


# ---- the length fit --------------------------------------------------------------------------------------------------------------------------

def train_manifest():
    rows = []
    for k in range(200):
        n = 3 + k % 15
        rows.append(dict(corpus=lengths.H2S, partition="train", signer_component="c02", sign=f"vid{k % 9}_{k}-5-rgb_front", text=" ".join(["word"] * n), continuous=dict(seconds=.37 * n)))
    for k in range(20):
        n = 3 + k % 15
        rows.append(dict(corpus=lengths.H2S, partition="train", signer_component="c04", sign=f"other{k % 3}_{k}-5-rgb_front", text=" ".join(["word"] * n), continuous=dict(seconds=.29 * n)))
    rows.append(dict(corpus=lengths.H2S, partition="val", signer_component="c01", sign="x_1-5-rgb_front", text="val text here", continuous=dict(seconds=99.)))
    return dict(records=rows)


def test_a_fit_reads_train_and_refuses_c00():
    manifest = train_manifest()
    assert all(r["partition"] == "train" for r in lengths.train_records(manifest))
    assert not any(r["continuous"]["seconds"] == 99. for r in lengths.train_records(manifest))
    manifest["records"].append(dict(corpus=lengths.H2S, partition="train", signer_component="c00", sign="y_1-5-rgb_front", text="a b", continuous=dict(seconds=1.)))
    with pytest.raises(SystemExit, match="never read"):
        lengths.train_records(manifest)


def test_a_video_stays_on_one_side_of_a_fold():
    assert lengths.source_of(dict(sign="--7E_2sU6z_10-5-rgb_front")) == "--7E_2sU6z" == lengths.source_of(dict(sign="--7E_2sU6z_11-3-rgb_front"))
    folds = lengths.folds_of(["a", "a", "b", "a"])
    assert folds[0] == folds[1] == folds[3] and 0 <= folds.max() < lengths.FOLDS


def test_a_balanced_fit_weighs_each_signer_equally():
    x = np.arange(1., 201.)[:, None]
    group = np.array(["c02"] * 180 + ["c04"] * 20)
    y = np.where(group == "c02", .4, .2) * x[:, 0]
    unweighted = lengths.fit_line(x, y)
    balanced = lengths.fit_line(x, y, lengths.balance(group))
    assert balanced[1] < unweighted[1]
    fit = cv2.fit_duration_model(train_manifest(), lengths.H2S, "train", components=("c02", "c04"), signer_balanced=True)
    rows = lengths.train_records(train_manifest())
    tokens = np.array([len(r["text"].split()) for r in rows], float)[:, None]
    mine = lengths.fit_line(tokens, np.array([r["continuous"]["seconds"] for r in rows]), lengths.balance(np.array([r["signer_component"] for r in rows])))
    assert mine[1] == pytest.approx(fit["slope"], abs=1e-3) and mine[0] == pytest.approx(fit["intercept"], abs=1e-3)


def cv_row(error):
    return dict(cross_validated=dict(choosing_error=error))


@pytest.mark.parametrize("rows,chosen", [
    (dict(tokens_all=cv_row(.3), tokens_balanced=cv_row(.25), words_balanced=cv_row(.2)), "words_balanced"),
    (dict(tokens_all=cv_row(.3), tokens_balanced=cv_row(.247), words_balanced=cv_row(.249)), "tokens_balanced"),
    (dict(tokens_all=cv_row(.3), tokens_balanced=cv_row(.249), words_balanced=cv_row(.247)), "tokens_balanced"),      # within the tie: fewer parameters
    (dict(tokens_all=cv_row(.2), tokens_balanced=cv_row(.25), words_balanced=cv_row(.2)), "tokens_all"),
])
def test_the_choice_is_on_train_error_with_ties_to_fewer_parameters(rows, chosen):
    assert lengths.choose(rows) == chosen


def test_errors_are_per_clip_and_pooled():
    row = lengths.errors(np.array([2., 2.]), np.array([1., 4.]))
    assert row["pooled_ratio"] == .8 and row["median_ratio"] == 1.25 and row["median_abs_relative_error"] == .75 and row["clips"] == 2


def test_the_floor_holds_at_a_negative_intercept():
    assert lengths.apply_line([-5., .1], np.array([[1.]]))[0] == lengths.FLOOR


def test_the_committed_lengths_chose_on_train_and_the_composition_uses_that_fit():
    payload = json.loads((ROOT / "reports/comp_a6_lengths.json").read_text())
    assert payload["chosen"] == "tokens_balanced"
    m = payload["candidates"]["tokens_balanced"]["model"]
    assert (m["intercept"], m["slope"], m["records"]) == a6.BALANCED_FIT
    assert f"BALANCED={a6.BALANCED_FIT[0]},{a6.BALANCED_FIT[1]},{a6.BALANCED_FIT[2]}" in JOB.read_text()
    assert "D20" in payload["d20"]["note"] and "fit" in payload["d20"]["note"]


def test_the_plan_features_are_entry_seconds_and_unsigned_tokens():
    rows = [[dict(kind="exact", tokens=["a"], entry="a"), dict(kind="unmatched", tokens=["b", "c"], entry=None)]]
    assert lengths.plan_features(["a b c"], rows, [["a"]], dict(a=.7)).tolist() == [[.7, 2.]]


# ---- the options in the composition ----------------------------------------------------------------------------------------------------------

def run_top10(c, monkeypatch, *more, name="a"):
    path, sha = write_list(c, token_rows(c, monkeypatch))
    extra = ("--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top", "10")
    main = cache_for_words(c, f"cache/{name}", "--draw", "1")
    other = cache_for_words(c, f"cache/{name}x", "--draw", "1", *extra)
    hold_path = ROOT / "reports/comp_a_hold_stats.json"
    args = generated_args(c, "--word-cache-key", f"cache/{name}/index.json", "--word-cache-sha256", main, "--word-cache-n", "4", "--core-margin", "0", "--predicted-duration", *extra,
                          "--word-cache-extra-key", f"cache/{name}x/index.json", "--word-cache-extra-sha256", other, "--comp-a-job-file", a6.JOB_FILE, "--clips", "4",
                          *[x.replace("HOLD", str(hold_path)).replace("HSHA", a6.HOLD_SHA).replace("REST", str(ROOT / "reports/comp_a6_rest_pose.json")).replace("RSHA", a6.REST_SHA) for x in more], "--output-key", f"e/{name}.json")
    assert ce.main(args) == 0
    return part(c, f"e/{name}.json")


def spy_hold(monkeypatch):
    seen = []
    real = cv2.hold_left_arm

    def spy(state, plan, flags, **kw):
        seen.append((list(flags), kw))
        return real(state, plan, flags, **kw)
    monkeypatch.setattr(cv2, "hold_left_arm", spy)
    return seen


def test_exclusion_and_rest_reach_the_hold_and_their_settings(d20, monkeypatch, recogniser):  # noqa: F811
    c = d20
    seen = spy_hold(monkeypatch)
    shard = run_top10(c, monkeypatch, "--hold-left-file", "HOLD", "--hold-left-sha256", "HSHA", "--hold-left-exclude", "0.25", name="x")
    assert seen and not any("rest" in kw for _, kw in seen) and shard["settings"]["round20"]["hold_left"] == dict(sha256=a6.HOLD_SHA, exclude=.25)
    seen.clear()
    shard = run_top10(c, monkeypatch, "--hold-left-file", "HOLD", "--hold-left-sha256", "HSHA", "--hold-left-exclude", "0.25", "--hold-rest-file", "REST", "--hold-rest-sha256", "RSHA", name="xr")
    assert seen and all(kw["rest"] == lib.load_rest_pose(ROOT / "reports/comp_a6_rest_pose.json", a6.REST_SHA) for _, kw in seen)
    assert shard["settings"]["round20"]["hold_left"] == dict(sha256=a6.HOLD_SHA, exclude=.25, rest_sha256=a6.REST_SHA)


class Marks(dict):
    def get(self, word, default=None):
        return "MARK"


def test_the_exclusion_flags_are_what_the_hold_gets(d20, monkeypatch, recogniser):  # noqa: F811
    c = d20
    seen = spy_hold(monkeypatch)
    calls = []
    monkeypatch.setattr(lib, "hold_flags", lambda pools, *, two_handed_share, new_entries: (calls.append((two_handed_share, set(new_entries))), Marks())[1])
    run_top10(c, monkeypatch, "--hold-left-file", "HOLD", "--hold-left-sha256", "HSHA", "--hold-left-exclude", "0.25", name="f")
    assert seen and all(set(flags) == {"MARK"} for flags, _ in seen)
    assert calls and calls[0][0] == .25 and {"you", "half"} <= calls[0][1]


def test_new_entries_are_anchors_under_exclusion(d20, monkeypatch, recogniser):  # noqa: F811
    c = d20
    seen = spy_hold(monkeypatch)
    shard = run_top10(c, monkeypatch, "--hold-left-file", "HOLD", "--hold-left-sha256", "HSHA", "--hold-left-exclude", "0.25", name="n")
    tokens = shard["clips"][0]["stage1_words"]
    assert seen
    for row, (flags, _) in zip(shard["clips"], seen):
        for word, flag in zip(row["stage1_words"], flags):
            if word in ("you", "half", "it", "this"):
                assert flag is None, (word, flag, tokens)


def test_the_balanced_length_is_its_own_settings_and_its_blank_moves(d20, monkeypatch, recogniser):  # noqa: F811
    c = d20
    today = run_top10(c, monkeypatch, name="t")
    monkeypatch.setattr(ce, "check_duration_model", lambda fitted, pin: None)
    calls = []
    monkeypatch.setattr(cv2, "fit_duration_model", lambda *a, **k: (calls.append(k), dict(intercept=.1, slope=.2, records=3, partition="train", signer_balanced=k.get("signer_balanced", False)))[1])
    balanced = run_top10(c, monkeypatch, "--duration-balanced", "--expect-duration-model=.1,.2,3", name="b")
    assert calls and calls[0] == dict(components=("c02", "c04"), signer_balanced=True)
    assert balanced["settings"]["round20"]["duration_balanced"] is True and "duration_balanced" not in today["settings"]["round20"]
    assert balanced["round20"]["duration_model"]["signer_balanced"] is True
    assert [r["frames_blank_predicted"] for r in balanced["clips"]] != [r["frames_blank_predicted"] for r in today["clips"]]      # the blank has the arm's own length


@pytest.mark.parametrize("flags,message", [
    (["--hold-left-exclude", "0.25"], "qualify|without the option"),
    (["--hold-rest-file", "x", "--hold-rest-sha256", "y"], "qualify|without the option"),
    (["--hold-left-file", "x", "--hold-left-sha256", "y", "--hold-rest-file", "x", "--hold-rest-sha256", "y"], "only on the exclusion arm"),
    (["--hold-left-file", "x", "--hold-left-sha256", "y", "--hold-left-exclude", "0.25", "--hold-rest-file", "x"], "go together"),
    (["--hold-left-file", "x", "--hold-left-sha256", "y", "--hold-left-exclude", "0"], "share of records"),
    (["--hold-left-file", "x", "--hold-left-sha256", "y", "--hold-left-exclude", "1.5"], "share of records"),
    (["--duration-balanced"], "qualifies --predicted-duration"),
    (["--duration-balanced", "--predicted-duration"], "qualifies --predicted-duration"),
])
def test_the_options_refuse(d20, capsys, flags, message):  # noqa: F811
    c = d20
    args = generated_args(c, "--word-cache-key", "k", "--word-cache-sha256", "s", "--word-cache-n", "4", "--core-margin", "0", *flags, "--output-key", "e/r.json")
    with pytest.raises(SystemExit):
        ce.main(args)
    assert re.search(message, capsys.readouterr().err)


def test_the_balanced_length_refuses_another_set(tmp_path, capsys, stage2_store):  # noqa: F811
    from tests.test_round15_compose_eval import run_args
    _, sha, *_ = stage2_store
    with pytest.raises(SystemExit):
        ce.main([*run_args(tmp_path, sha), "--predicted-duration", "--duration-balanced", "--expect-duration-model=.1,.2,3", "--output-key", "e/o.json"])
    assert "D20" in capsys.readouterr().err


def test_a_pair_must_declare_the_length_setting():
    from round20_compare import check_pairable
    a, b = arm(a6.LENGTH, 1), arm(a6.BASE, 1)
    check_pairable(a, b, ("duration_balanced",))
    with pytest.raises(SystemExit, match="undeclared differences"):
        check_pairable(a, b, ())


# ---- a synthetic set of draws ----------------------------------------------------------------------------------------------------------------

def settings_for(name, k):
    s = dict(word_cache=dict(sha256=f"cache{k}", key="k", **({"draw": k} if k != a6.SCREEN else {})), word_cache_extra=dict(sha256=f"extra{k}", key="x"), vocabulary_extra=dict(top=10), provenance=dict(PROVENANCE))
    if k == a6.SCREEN:
        s["word_cache"]["sha256"], s["word_cache_extra"]["sha256"] = a6.A3_CACHE_SHA, a6.A3_EXTRA_SHA
    if name in (a6.HOLD, a6.BOTH):
        s["hold_left"] = dict(sha256=a6.HOLD_SHA, exclude=a6.EXCLUDE)
    if name == a6.HOLD_REST:
        s["hold_left"] = dict(sha256=a6.HOLD_SHA, exclude=a6.EXCLUDE, rest_sha256=a6.REST_SHA)
    if name in (a6.LENGTH, a6.BOTH):
        s["duration_balanced"] = True
    return s


WRISTS = {a6.BASE: 1.774, a6.HOLD: 1.595, a6.LENGTH: 2.019, a6.BOTH: 1.814, a6.HOLD_REST: 1.66}


def arm(name, k, *, ll=0., hyps=None, hands=1., ratio=None):
    report = arm_report(settings=settings_for(name, k), extra=hyps or {})
    report["settings"]["round20"].pop("comp_a_read", None)
    report["likelihood"]["per_form"] = {f: [x + (ll if f == FORM else 0.) for x in v] for f, v in report["likelihood"]["per_form"].items()}
    report["motion_ratio_to_real"][FORM].update(wrist_speed=WRISTS[name], hands_tracked=hands)
    report["comp_a_motion"][FORM].update(left_speed_over_real=WRISTS[name] + .4, right_speed_over_real=1.45, contact_share=.15)
    fitted = name in (a6.LENGTH, a6.BOTH)
    report["round20"]["duration_model"] = dict(intercept=a6.BALANCED_FIT[0], slope=a6.BALANCED_FIT[1], records=a6.BALANCED_FIT[2], signer_balanced=True) if fitted else dict(intercept=-.083, slope=.3691, records=18289)
    report["predicted_timing"] = dict(forms={FORM: dict(seconds_over_reference=dict(pooled=(1.078 if fitted else 1.198) if ratio is None else ratio, median_per_clip=1.147 if fitted else 1.287))})
    if name in (a6.BOTH, a6.HOLD, a6.HOLD_REST):
        report["hold_left"] = dict(cap_bound=2 if name == a6.BOTH else 0, peak=1.9)         # both forms of a run: not what KNOWN_AT_WRITING states
        report["hold_left_predicted"] = dict(cap_bound=1 if name == a6.BOTH else 0, peak=1.376 if name == a6.BOTH else 1.34, stretches=9)
    report["spotting"]["scores"][FORM]["minus_mismatched_paired"] = dict(difference=.3, ci95=[.2, .4], n=12)
    return report


def shard_of(report, recall=.3):
    return [dict(settings=report["settings"], clips=[dict(clip=i, sign=s, spotting={FORM: dict(own=dict(recall=recall))}) for i, s in enumerate(report["signs"])])]


THREE = [r if i < 3 else "some long words in" for i, r in enumerate(REFS)]
FIVE = [r if i < 5 else "some long words in" for i, r in enumerate(REFS)]
SIX = [r if i < 6 else "some long words in" for i, r in enumerate(REFS)]
HYPS = {a6.BASE: THREE, a6.HOLD: FIVE, a6.LENGTH: FIVE, a6.BOTH: SIX, a6.HOLD_REST: FIVE}
LIKELIHOOD = {a6.BASE: 0., a6.HOLD: .01, a6.LENGTH: .01, a6.BOTH: .02, a6.HOLD_REST: .01}
RECALL = {a6.BASE: .30, a6.HOLD: .31, a6.LENGTH: .31, a6.BOTH: .32, a6.HOLD_REST: .31}


def draws(keys=a6.DRAWS, *, by_arm=None):
    """In every draw hold-x and length each read two sentences more than base, and both together one more than length."""
    out, shards = {}, {}
    for k in keys:
        out[k], shards[k] = {}, {}
        for name in a6.arms_of(k):
            kw = dict(ll=LIKELIHOOD[name])
            kw.update((by_arm or {}).get(name, {}))
            out[k][name] = arm(name, k, hyps={FORM: HYPS[name]}, **kw)
            shards[k][name] = shard_of(out[k][name], RECALL[name])
    return out, shards


def tops(d):
    return {k: copy.deepcopy(arms[a6.BASE]) for k, arms in d.items() if k != a6.SCREEN}


def screen_summary(likelihood=.01, spotting=.01, **kw):
    """A stored screen's summary, as far as the gate reads it."""
    row = {a6.HOLD: dict(likelihood_minus_blank=dict(mean=likelihood), spotting=dict(mean=spotting))}
    return dict(dict(mode="screen", declared=True, declared_commit="abc1234", arms=row, floors=dict(a6.FLOORS, against=a6.BARS_AGAINST), provenance=lib.provenance_row(a6.JOB_FILE)), **kw)


def read(d=None, s=None, boot=60, **kw):
    if d is None:
        d, s = draws()
    kw.setdefault("a5_top10", tops(d))
    if a6.SCREEN not in d:
        kw.setdefault("screen_summary", screen_summary())
        kw.setdefault("screen_sha256", "h")
        kw.setdefault("declared_commit", "abc1234")
    return a6.read(d, s, boot=boot, **kw)


def test_the_read_reports_every_arm_over_the_draws():
    out = read()
    assert out["report_only"] is True and out["mode"] == "draws" and out["declared"] is True and out["draws"] == list(range(1, 9)) and "adopted" not in json.dumps(out)
    assert set(out["arms"]) == {a6.HOLD, a6.LENGTH, a6.BOTH} and out["reported_only"] == [a6.HOLD]
    for row in out["arms"].values():
        for metric in ("bleu", "likelihood_minus_blank", "spotting"):
            assert len(row[metric]["per_draw"]) == 8 and set(row[metric]) == {"per_draw", "mean", "between_draw_sd", "positive_draws", "two_way_ci95", "t_ci95"}
        assert row["bleu"]["mean"] > 0 and len(row["bleu_net_of_blank"]["per_draw"]) == 8 and len(row["comparator_bleu"]) == 8 and len(row["gap_share"]["arm"]) == 8
    assert out["arms"][a6.LENGTH]["motion"]["arm_wrist"] == 2.019 and out["arms"][a6.LENGTH]["motion"]["comparator_wrist"] == 1.774
    assert out["arms"][a6.BOTH]["motion"]["arm_wrist"] == 1.814 and out["arms"][a6.BOTH]["motion"]["comparator_wrist"] == 2.019 and out["arms"][a6.BOTH]["comparator"] == a6.LENGTH
    assert out["arms"][a6.HOLD]["comparator"] == a6.BASE and out["arms"][a6.LENGTH]["length_over_true"][0] == dict(pooled=1.078, median_per_clip=1.147)
    assert out["arms"][a6.BOTH]["cap_bound_per_draw"] == [1] * 8 and out["arms"][a6.BOTH]["hold_peak_per_draw"] == [1.376] * 8 and out["arms"][a6.HOLD]["cap_bound_per_draw"] == [0] * 8       # like KNOWN: the predicted form, one draw at a time
    assert out["arms"][a6.BOTH]["own_minus_mismatched"]["arm"][0]["difference"] == .3 and len(out["arms"][a6.LENGTH]["own_minus_mismatched"]["comparator"]) == 8
    assert out["floors"] == dict(a6.FLOORS, against=a6.BARS_AGAINST)


def test_the_hold_rule_reads_both_against_length_and_the_length_rule_length_against_base():
    row = read()["arms"]
    assert row[a6.BOTH]["likelihood_minus_blank"]["per_draw"] == pytest.approx([.01] * 8, abs=1e-6) and row[a6.BOTH]["spotting"]["per_draw"] == pytest.approx([.01] * 8, abs=1e-6)
    assert row[a6.LENGTH]["likelihood_minus_blank"]["per_draw"] == pytest.approx([.01] * 8, abs=1e-6)
    from signtranslator.generation.motion_to_text import corpus_bleu
    want = corpus_bleu(SIX, REFS)["bleu"] - corpus_bleu(FIVE, REFS)["bleu"]
    assert row[a6.BOTH]["bleu"]["per_draw"] == pytest.approx([want] * 8, abs=1e-3)
    want = corpus_bleu(FIVE, REFS)["bleu"] - corpus_bleu(THREE, REFS)["bleu"]
    assert row[a6.LENGTH]["bleu"]["per_draw"] == pytest.approx([want] * 8, abs=1e-3)


def test_both_against_base_and_the_difference_in_differences_are_reported_and_decide_nothing():
    out = read()
    both = out["both_on_base"]
    assert both["comparator"] == a6.BASE and both["likelihood_minus_blank"]["per_draw"] == pytest.approx([.02] * 8, abs=1e-6) and both["spotting"]["per_draw"] == pytest.approx([.02] * 8, abs=1e-6)
    diff = out["difference_in_differences"]
    assert diff["likelihood_minus_blank"]["per_draw"] == pytest.approx([0.] * 8, abs=1e-6) and set(diff) == {"bleu", "likelihood_minus_blank", "spotting"} and len(diff["bleu"]["t_ci95"]) == 2
    d, s = draws(by_arm={a6.BOTH: dict(ll=.03)})
    moved = read(d, s)
    assert moved["difference_in_differences"]["likelihood_minus_blank"]["mean"] == pytest.approx(.01, abs=1e-6)
    d, s = draws()
    for k in a6.DRAWS:                 # a large loss against base alone: neither rule reads that row
        d[k][a6.BOTH]["likelihood"]["per_form"][FORM] = [x - .5 for x in d[k][a6.BOTH]["likelihood"]["per_form"][FORM]]
        d[k][a6.LENGTH]["likelihood"]["per_form"][FORM] = [x - .5 for x in d[k][a6.LENGTH]["likelihood"]["per_form"][FORM]]
    assert read(d, s)["rule"]["hold"]["checks"]["likelihood_t_lower_bound_above_floor"] is True
    for row in (both, out["difference_in_differences"]):
        assert "checks" not in row and "outcome" not in row
    assert "single step" in a6.RULES["hold"] and "single step" in a6.RULES["length"]


def test_the_blank_is_each_reports_own_for_likelihood_and_net_bleu():
    d, s = draws(by_arm={a6.LENGTH: dict(ll=0.)})
    for k in a6.DRAWS:
        d[k][a6.LENGTH]["likelihood"]["per_form"][a6.BLANK] = [x - .5 for x in d[k][a6.LENGTH]["likelihood"]["per_form"][a6.BLANK]]
        d[k][a6.LENGTH]["forms"][a6.BLANK]["bleu"] = d[k][a6.BASE]["forms"][a6.BLANK]["bleu"] + .4
    row = read(d, s)["arms"][a6.LENGTH]
    assert row["likelihood_minus_blank"]["mean"] == pytest.approx(.5, abs=1e-6)
    assert row["bleu_net_of_blank"]["mean"] == pytest.approx(row["bleu"]["mean"] - .4, abs=2e-3) and row["bleu_net_of_blank"]["t_ci95"][0] == row["bleu_net_of_blank"]["t_ci95"][1]


def test_a_screen_decides_nothing_has_no_draw_variance_and_holds_hold_xr():
    d, s = draws((a6.SCREEN,))
    out = read(d, s)
    assert out["mode"] == "screen" and out["draws"] == ["a3"] and set(out["arms"]) == {a6.HOLD, a6.LENGTH, a6.BOTH, a6.HOLD_REST} and out["reported_only"] == [a6.HOLD, a6.HOLD_REST]
    assert out["rule"]["hold"]["outcome"] == out["rule"]["length"]["outcome"] == "screen_decides_nothing" and out["rule"]["hold"]["checks"] is None
    row = out["arms"][a6.LENGTH]
    assert row["bleu"]["t_ci95"] is None and row["spotting"]["between_draw_sd"] is None and row["bleu_net_of_blank"]["t_ci95"] is None and row["likelihood_minus_blank"]["two_way_ci95"][1] >= row["likelihood_minus_blank"]["two_way_ci95"][0]
    assert out["arms"][a6.HOLD_REST]["comparator"] == a6.BASE


def test_hold_xr_is_on_no_draw():
    d, s = draws()
    d[3][a6.HOLD_REST] = arm(a6.HOLD_REST, 3)
    with pytest.raises(SystemExit, match="its arms are"):
        a6.check_set(d)
    d, s = draws((a6.SCREEN,))
    del d[a6.SCREEN][a6.HOLD_REST]
    with pytest.raises(SystemExit, match="its arms are"):
        a6.check_set(d)


def test_a_screen_reads_a3s_stored_words_only():
    d, s = draws((a6.SCREEN,))
    for name in a6.SCREEN_ARMS:
        d[a6.SCREEN][name]["settings"]["round20"]["word_cache"]["sha256"] = "other"
    with pytest.raises(SystemExit, match="A3's stored words"):
        read(d, s)
    d, s = draws((a6.SCREEN,))
    for name in a6.SCREEN_ARMS:
        d[a6.SCREEN][name]["settings"]["round20"]["word_cache"]["draw"] = 1
    with pytest.raises(SystemExit, match="A3's stored words"):
        read(d, s)


def test_the_instrument_check_reports_every_draw_and_stops_below_the_rate():
    d, s = draws()
    a5s = tops(d)
    out = read(d, s, a5_top10=a5s)["a5_instrument_check"]
    assert out["draws"][1] == dict(form_identical=12, real_identical=12, blank_identical=12, clips=12, same_gpu=True, rate=1.)
    assert read(d, s, a5_top10=a5s)["rule"]["hold"]["outcome"] == "passed"
    for form in (FORM, "real", a6.BLANK):
        bent = tops(d)
        bent[2]["forms"][form]["hypotheses"][0] = "another hypothesis"
        with pytest.raises(SystemExit, match="below 0.98: the draw stops"):
            read(d, s, a5_top10=bent)
    bent = tops(d)
    bent[3]["settings"]["round20"]["provenance"] = dict(PROVENANCE, environment=dict(PROVENANCE["environment"], gpu="other"))
    bent[3]["forms"][FORM]["hypotheses"][0] = "another hypothesis"
    out = read(d, s, a5_top10=bent)["a5_instrument_check"]                          # another GPU may differ, and says so
    assert out["draws"][3]["same_gpu"] is False and out["draws"][3]["rate"] == .9167 and out["draws"][1]["same_gpu"] is True
    bent = tops(d)
    bent[4]["references"] = list(reversed(bent[4]["references"]))
    with pytest.raises(SystemExit, match="other sentences"):
        read(d, s, a5_top10=bent)


def test_the_draws_need_at_least_one_same_gpu_draw():
    d, s = draws()
    bent = tops(d)
    for k in a6.DRAWS:
        bent[k]["settings"]["round20"]["provenance"] = dict(PROVENANCE, environment=dict(PROVENANCE["environment"], gpu="other"))
    with pytest.raises(SystemExit, match="incomplete: no draw ran on A5's GPU"):
        read(d, s, a5_top10=bent)
    d, s = draws((a6.SCREEN,))
    top = copy.deepcopy(d[a6.SCREEN][a6.BASE])
    top["settings"]["round20"]["provenance"] = dict(PROVENANCE, environment=dict(PROVENANCE["environment"], gpu="other"))
    assert a6.read(d, s, boot=20, a5_top10={a6.SCREEN: top})["a5_instrument_check"]["draws"][a6.SCREEN]["same_gpu"] is False     # A3's stored top10 ran on two GPUs: the screen reports it


@pytest.mark.parametrize("cache", ["word_cache", "word_cache_extra"])
def test_the_instrument_check_refuses_other_words(cache):
    d, s = draws()
    a5s = tops(d)
    a5s[3]["settings"]["round20"][cache]["sha256"] = "other"
    with pytest.raises(SystemExit, match=f"other words \\({cache}\\)"):
        read(d, s, a5_top10=a5s)


@pytest.mark.parametrize("drop", [None, 1, 8])
def test_the_draws_need_all_eight_of_a5s_top10(drop):
    d, s = draws()
    a5s = tops(d)
    if drop is None:
        a5s = {}
    else:
        del a5s[drop]
    with pytest.raises(SystemExit, match="all eight of A5's top10"):
        a6.read(d, s, boot=20, a5_top10=a5s, screen_summary=screen_summary(), screen_sha256="h", declared_commit="abc1234")
    d, s = draws((a6.SCREEN,))
    assert a6.read(d, s, boot=20)["a5_instrument_check"] is None                    # a screen has none to compare


def test_a_declared_read_refuses_another_digest_and_says_to_redeclare(monkeypatch):
    d, s = draws()
    monkeypatch.setattr(lib, "provenance_row", lambda job=None: dict(PROVENANCE, code_digest="e" * 64))
    with pytest.raises(SystemExit, match="the screen's summary came from other code: re-declare under a new OUT"):
        read(d, s, screen_summary=dict(screen_summary(), provenance=dict(PROVENANCE)))
    with pytest.raises(SystemExit, match="re-declare under a new OUT named in the job, with what was stored disclosed"):
        read(d, s, screen_summary=dict(screen_summary()))
    d, s = draws((a6.SCREEN,))
    with pytest.raises(SystemExit, match="re-declare under a new OUT"):
        a6.read(d, s, boot=20, declared_commit="abc1234")
    assert "amendment" not in a6.read(d, s, boot=20)
    assert "--reader-amendment" not in Path(a6.__file__).read_text()


def test_a_stored_screen_from_older_code_stops_the_draws(monkeypatch):
    older = screen_summary(provenance=dict(PROVENANCE, code_digest="0" * 64))
    with pytest.raises(SystemExit, match="other code: re-declare under a new OUT named in the job"):
        a6.hold_gate(older, "abc1234")


# ---- the rules -----------------------------------------------------------------------------------------------------------------------------------

def metric(mean=.05, two=.01, t=.01, positive=8):
    return dict(mean=mean, two_way_ci95=[two, 9.], t_ci95=[t, 9.], positive_draws=positive, per_draw=[], between_draw_sd=.1)


def hold_row(**kw):
    out = dict(bleu=metric(t=B / 2), likelihood_minus_blank=metric(), spotting=metric(), motion=dict(arm_hands=1., comparator_hands=1.))
    out.update(kw)
    return out


def length_row(**kw):
    out = dict(hold_row(), bleu_net_of_blank=dict(t_ci95=[B / 2, 9.], mean=0., per_draw=[]), length_over_true=[dict(pooled=1.078)] * 8)
    out.update(kw)
    return out


ROOM = ("For each metric, if A5's mean gain is at or below 0, or its lower 95% t bound over the eight draws is at or below 0, that metric's outcome is no_room: no gain was established to protect. A rule with any no_room metric does not pass and prints no_room with the reason; the hold gate closes; the arms still report. ")
STEP = ("Each pass is a single step: the hold rule reads hold-x at the fitted length and the length rule reads the fitted length against today's, so passing both does not show that the two together lose little; both together against top10 at today's length and the difference in differences are reported and decide nothing. ")
MATCH = ("Each draw's top10 at today's length must match A5's top10 for that draw, and the screen's A3's stored top10: on the same GPU type and pinned thread count at least 98% of its hypotheses, for the form, for real and for blank@predicted. Below that, the draw stops and later attempts of that draw are refused until it is declared. The draws need at least one same-GPU draw, else the read is incomplete; the count and rate of every draw are printed. ")
AGAIN = "Any reader bug, or any code change after the screen or a draw, means re-declaring under a new OUT named in the job, with what was stored disclosed; no key is scoped by digest automatically. "
FRESH = "D20 chose top10 and A5's draws read its sentences, so D20's sentences are not fresh. "
FULL = "A6's full runs, the screen among them, run only if A5's summary reads confirmed or direction only."
RULE_HOLD = ("Primary arm: hold-x at the fitted length, against top10 at the fitted length, paired within each of A5's eight stored word draws, on D20, report only. hold-x keeps the left arm on entries with at least 25% two-handed records, on unannotated entries and on the list's new entries; the 25% cut was set after A1's D20 failure. hold-x at today's length is read beside it and decides nothing. Passed only if all hold: mean paired likelihood minus each form's own-length blank, its 95% lower bound " + a6.bar(a6.FLOORS["likelihood"]) + " on both the two-way bootstrap and the t interval over the eight draws; v2 spotting's 95% lower bound " + a6.bar(a6.FLOORS["spotting"]) + " on both; BLEU over draws, the t interval's lower bound " + a6.bar(a6.FLOORS["bleu"]) + "; mean hands at least 0.9x its comparator's. These are small-loss bars, set as shares of the mean paired gains of " + a6.BARS_AGAINST + ": 44% of BLEU's, 60% of likelihood's and of spotting's. " + ROOM + "A pass means any loss is small, possibly up to those shares, with less motion, and not that the arm reads better. Known before any read, and not evidence: pooled wrist 2.019x to 1.814x, contact unchanged by construction. The form stays far outside the 0.8-1.25x band, so a pass is no progress against the release bar. The gate: the screen on A3's stored words reads hold-x against top10 at today's length, report only, and hold-x and hold-x at the fitted length run on the eight draws only if the screen's point estimates of likelihood and spotting sit above the two floors above. Otherwise the draws run top10 and length only, the length rule still reads, and this rule reads not run, screen failed, which is printed. Known from A1's stored rows before this gate was set: the 34% of clips holding the excluded uses hold 39% of A1's likelihood loss and 40% of its spotting loss, and excluding those uses is estimated to remove 3-15% against the " + a6.needed_text() + " these floors need, so the screen may well stop the hold arms; that estimate is no read of hold-x. " + STEP + MATCH + AGAIN + FRESH + "Anything else is not passed, the failed check is reported, and not passed is not evidence of a loss. " + FULL)
RULE_LENGTH = ("Primary arm: length, the fit signer-balanced on How2Sign train's c02 + c04 (seconds = -0.3577 + 0.3445 x tokens), chosen by a train-only rule after its D20 ratio of about 1.08 was known, and never fitted on D20's lengths. Against top10 at today's length, paired within each of A5's eight stored word draws, on D20, report only. Passed only if all hold: likelihood and spotting as the hold rule; BLEU, the t interval's lower bound " + a6.bar(a6.FLOORS["bleu"]) + " on the lower of raw BLEU and BLEU net of the blank's own change; mean hands at least 0.9x top10's. " + ROOM + "Pooled predicted over true length is reported and decides nothing: known on the report's state frames, 1.071 against today's 1.198, median per clip 1.128 against 1.287. Today's length runs 1.20x true, which lowers wrist by construction; the fitted length removes that, so wrist rises from 1.774x to 2.019x. Wrist is reported, decides nothing, and this rise is neither a loss nor progress. A pass means a nearer length costs little content, not that it adds any. This rule reads whether or not the screen lets the hold arms run. " + STEP + MATCH + AGAIN + FRESH + FULL)


def test_the_rules_are_the_texts_as_given():
    assert a6.RULES == dict(hold=RULE_HOLD, length=RULE_LENGTH)


SLOT = dict(sha256="abc", outcome="direction_only_not_confirmed", bleu=.2, likelihood=.01, spotting=.02, t_lower=dict(bleu=.01, likelihood=.002, spotting=.003), paired_sd=dict(bleu=.1, likelihood=.004, spotting=.005))


def test_the_bars_are_the_stated_shares_of_the_gains():
    assert a6.floors() == dict(bleu=-.1509, likelihood=-.0077, spotting=-.0089)    # of A3's stored read: 44%, 60%, 60%
    assert a6.floors(SLOT) == dict(bleu=-.088, likelihood=-.006, spotting=-.012)   # restated against A5's mean


@pytest.mark.parametrize("change,metric,why", [(dict(likelihood=0.), "likelihood", "mean gain"), (dict(spotting=-.01), "spotting", "mean gain"), (dict(bleu=-.1), "bleu", "mean gain"),
                                               (dict(t_lower=dict(SLOT["t_lower"], likelihood=0.)), "likelihood", "lower 95% t bound"), (dict(t_lower=dict(SLOT["t_lower"], spotting=-.001)), "spotting", "lower 95% t bound"), (dict(t_lower=dict(SLOT["t_lower"], bleu=-.2)), "bleu", "lower 95% t bound")])
def test_no_gain_established_is_no_room(change, metric, why):
    slot = dict(SLOT, **change)
    room = a6.no_room(slot)
    assert set(room) == {metric} and why in room[metric] and "no gain was established to protect" in room[metric]
    assert a6.floors(slot)[metric] is None and all(v is not None for k, v in a6.floors(slot).items() if k != metric)
    assert a6.no_room(None) == {} and a6.no_room(SLOT) == {}
    assert a6.needed_share(-.0276, None) is None and a6.bar(None) == "above no bar (no_room)"


def test_the_slot_is_read_from_a5s_summary():
    row = lambda mean, sd, t: dict(mean=mean, between_draw_sd=sd, t_ci95=[t, 9.])       # noqa: E731
    summary = dict(rule=dict(outcome="confirmed"), forms=dict(predicted=dict(bleu=row(.3, .1, .05), likelihood_minus_blank=row(.012, .004, .002), spotting=row(.014, .005, .003))))
    assert a6.slot_from_summary(summary, "h") == dict(sha256="h", outcome="confirmed", bleu=.3, likelihood=.012, spotting=.014, t_lower=dict(bleu=.05, likelihood=.002, spotting=.003), paired_sd=dict(bleu=.1, likelihood=.004, spotting=.005))


def filled_module(slot, **others):
    """The module as a declaration leaves it: its source with the slots filled."""
    source = (ROOT / "scripts/comp_a6_read.py").read_text()
    for name, value in dict(others, A5_SUMMARY_AT_DECLARATION=slot).items():
        line = f"{name} = None"
        assert source.count(line) == 1
        source = source.replace(line, f"{name} = " + repr(value))
    space = dict(__name__="filled", __file__=str(ROOT / "scripts/comp_a6_read.py"))
    exec(compile(source, "filled", "exec"), space)
    return space


def test_the_path_read_and_the_prior_power_figures_have_slots():
    known = filled_module(SLOT, PATH_READ_AT_DECLARATION="path read says X", PRIOR_REVIEW_POWER="power says Y")["KNOWN_AT_WRITING"]
    assert "Path read (branch path-read): path read says X." in known and "Prior review's power figures: power says Y." in known


def test_a_filled_slot_moves_the_bars_the_shares_the_rule_texts_and_what_was_known():
    space = filled_module(SLOT)
    assert space["FLOORS"] == dict(bleu=-.088, likelihood=-.006, spotting=-.012) and (space["BLEU_FLOOR"], space["LIKELIHOOD_FLOOR"], space["SPOTTING_FLOOR"]) == (-.088, -.006, -.012) and space["NO_ROOM"] == {}
    needed = [round((.0276 - .006) / .0276, 4), round((.0405 - .012) / .0405, 4)]
    assert list(space["NEEDED"].values()) == needed and space["needed_text"]() == f"{needed[0]:.0%} and {needed[1]:.0%}"
    for phrase in ("above -0.0060 on both", "above -0.0120 on both", "above -0.0880", "the mean paired gains of A5's summary:", f"against the {needed[0]:.0%} and {needed[1]:.0%} these floors need"):
        assert phrase in space["RULES"]["hold"], phrase
    assert "above -0.0880" in space["RULES"]["length"] and "A3's stored read, until" not in space["RULES"]["hold"] and "72% and 78%" not in space["RULES"]["hold"]
    known = space["KNOWN_AT_WRITING"]
    for phrase in ("sha256 abc, reading direction_only_not_confirmed", "mean gains BLEU 0.2, likelihood 0.01, spotting 0.02", "paired SDs", "the power table is in", f"{needed[0]:.0%} and {needed[1]:.0%} the floors need"):
        assert phrase in known, phrase
    assert space["RULES"] != a6.RULES
    row = dict(hold_row(), bleu=metric(t=-.05), likelihood_minus_blank=metric(two=-.005, t=-.005), spotting=metric(two=-.011, t=-.011))
    assert space["decide_hold"](row)["outcome"] == "passed" and space["gate_of"](-.005, -.011)["passed"] is True and space["gate_of"](-.007, 0.)["passed"] is False


def test_a_slot_with_no_room_stops_the_rules_and_the_gate_and_prints_no_zero_bar():
    space = filled_module(dict(SLOT, spotting=0., t_lower=dict(SLOT["t_lower"], spotting=-.001)))
    assert "spotting" in space["NO_ROOM"] and space["FLOORS"]["spotting"] is None
    for text in space["RULES"].values():
        assert "-0.0000" not in text
    assert "lower bound above no bar (no_room) on both" in space["RULES"]["hold"]
    good = dict(hold_row(), bleu=metric(t=.1), likelihood_minus_blank=metric(two=.1, t=.1), spotting=metric(two=.1, t=.1))
    out = space["decide_hold"](good)
    assert out["outcome"] == "no_room" and "no gain was established to protect" in out["no_room"]["spotting"] and out["checks"]["spotting_t_lower_bound_above_floor"] is False
    assert space["decide_length"](dict(good, bleu_net_of_blank=dict(t_ci95=[.1, 9.])))["outcome"] == "no_room"
    gate = space["gate_of"](.1, .1)
    assert gate["passed"] is False and gate["no_room"] and gate["hold_arms"] == "not run, screen failed"
    known = space["KNOWN_AT_WRITING"]
    assert "no_room" in known and "no gain was established" in known


def test_a_closed_read_with_no_room_says_no_room_and_prints_why(monkeypatch):
    monkeypatch.setattr(a6, "NO_ROOM", dict(spotting="A5's mean gain is 0.0: no gain was established to protect"))
    d, s = closed_draws()
    out = read(d, s)
    assert out["rule"]["hold"]["outcome"] == "no_room" and out["rule"]["hold"]["no_room"]["spotting"] and out["rule"]["length"]["outcome"] == "no_room" and set(out["arms"]) == {a6.LENGTH}


def test_a_declaration_needs_the_a5_slot_and_the_power_table(tmp_path, monkeypatch):
    dc6.check_slot(ROOT, a6.A5_SUMMARY_AT_DECLARATION)                             # the repository as it stands: a record with an empty slot fails here
    (tmp_path / "runpod/fleet").mkdir(parents=True)
    dc6.check_slot(tmp_path, None)                                                 # no record, no slot needed
    (tmp_path / dc6.RECORD).write_text("{}")
    with pytest.raises(SystemExit, match="while A5_SUMMARY_AT_DECLARATION is empty"):
        dc6.check_slot(tmp_path, None)
    with pytest.raises(SystemExit, match="power_table_at_declaration is empty"):
        dc6.check_slot(tmp_path, dict(sha256="h"))                                 # the slot alone is not enough
    (tmp_path / "reports").mkdir()
    report = tmp_path / dc6.POWER_REPORT
    report.write_text(json.dumps(dict(power_table_at_declaration=None)))
    with pytest.raises(SystemExit, match="is empty"):
        dc6.check_slot(tmp_path, dict(sha256="h"))
    report.write_text(json.dumps(dict(power_table_at_declaration=dict(hold={}))))
    with pytest.raises(SystemExit, match="a row for each rule"):
        dc6.check_slot(tmp_path, dict(sha256="h"))
    report.write_text(json.dumps(dict(power_table_at_declaration=dict(hold={"likelihood": {}}, length={"likelihood": {}}))))
    dc6.check_slot(tmp_path, dict(sha256="h"))
    monkeypatch.setattr(a6, "A5_SUMMARY_AT_DECLARATION", None)
    with pytest.raises(SystemExit, match="is empty"):
        dc6.make_record("abc1234")
    monkeypatch.setattr(a6, "A5_SUMMARY_AT_DECLARATION", dict(sha256="h"))
    if not json.loads((ROOT / dc6.POWER_REPORT).read_text()).get("power_table_at_declaration"):
        with pytest.raises(SystemExit, match="power_table_at_declaration is empty"):
            dc6.make_record("abc1234")


def test_what_was_known_is_stated():
    known = a6.KNOWN_AT_WRITING
    for phrase in ("-0.0276", "-0.0405", "+0.343", "37c8d18", "fbe9b29", "11e310c", "1.376", "90% of stretches", "1.198x", "1.071x", "before the train-only rule was written", "08b9a150", "A5's summary at declaration", "two-view being the right reference", "39% of the likelihood loss", a6.needed_text(),
                   "The screen's gate on the hold arms was set after the concentration result", "one metadata row (seconds and frames) of How2Sign's test-split extraction report; no state was read; that split holds c00", "0.27%", "8.8%", "27.2%", "Path read (branch path-read)", "Prior review's power figures", "running them before declaration is recommended"):
        assert phrase in known, phrase
    assert "carry mostly that path" not in known
    assert "chosen on train alone" not in known and "chosen on train alone" not in a6.RULES["length"]


def test_every_check_passes_each_rule():
    assert a6.decide_hold(hold_row())["outcome"] == "passed" and a6.decide_length(length_row())["outcome"] == "passed"


CONTENT = [
    (dict(likelihood_minus_blank=metric(two=L)), "likelihood_two_way_lower_bound_above_floor"),
    (dict(likelihood_minus_blank=metric(t=L)), "likelihood_t_lower_bound_above_floor"),
    (dict(spotting=metric(two=S)), "spotting_two_way_lower_bound_above_floor"),
    (dict(spotting=metric(t=S)), "spotting_t_lower_bound_above_floor"),
    (dict(motion=dict(arm_hands=.89, comparator_hands=1.)), "hands_at_least_0_9_of_comparator"),
]


@pytest.mark.parametrize("kw,check", [*CONTENT, (dict(bleu=metric(t=B)), "bleu_t_lower_bound_above_floor")])
def test_each_hold_check_can_fail_alone(kw, check):
    out = a6.decide_hold(hold_row(**kw))
    assert [k for k, v in out["checks"].items() if not v] == [check] and out["outcome"] == "not_passed"


@pytest.mark.parametrize("kw,check", [*CONTENT, (dict(bleu_net_of_blank=dict(t_ci95=[B, 9.], mean=0., per_draw=[])), "bleu_lower_of_raw_and_net_t_lower_bound_above_floor"),
                                      (dict(bleu=metric(t=B)), "bleu_lower_of_raw_and_net_t_lower_bound_above_floor")])
def test_each_length_check_can_fail_alone(kw, check):
    out = a6.decide_length(length_row(**kw))
    assert [k for k, v in out["checks"].items() if not v] == [check] and out["outcome"] == "not_passed"


def test_the_bounds_are_where_the_rules_put_them():
    near = dict(likelihood_minus_blank=metric(two=L + 1e-4, t=L + 1e-4), spotting=metric(two=S + 1e-4, t=S + 1e-4), bleu=metric(t=B + 1e-4))
    assert a6.decide_hold(hold_row(**near))["outcome"] == "passed"
    assert a6.decide_length(length_row(**near, bleu_net_of_blank=dict(t_ci95=[B + 1e-4, 9.], mean=0., per_draw=[])))["outcome"] == "passed"
    assert a6.decide_hold(hold_row(motion=dict(arm_hands=.9, comparator_hands=1.)))["outcome"] == "passed"


def test_the_two_way_likelihood_bound_is_no_looser_than_the_t_bound():
    out = a6.decide_hold(hold_row(likelihood_minus_blank=metric(two=L - 1e-4, t=.01)))
    assert out["checks"]["likelihood_two_way_lower_bound_above_floor"] is False and out["outcome"] == "not_passed"


def test_length_decides_on_neither_its_ratio_nor_wrist_and_reads_the_lower_bleu():
    assert a6.decide_length(length_row(length_over_true=[dict(pooled=1.5)] * 8))["outcome"] == "passed"
    assert a6.decide_length(length_row(motion=dict(arm_hands=1., comparator_hands=1., arm_wrist=9., comparator_wrist=1.)))["outcome"] == "passed"
    assert a6.lower_of_raw_and_net(length_row(bleu=metric(t=-.2), bleu_net_of_blank=dict(t_ci95=[-.1, 9.]))) == -.2
    assert a6.lower_of_raw_and_net(length_row(bleu=metric(t=-.1), bleu_net_of_blank=dict(t_ci95=[-.2, 9.]))) == -.2
    assert a6.lower_of_raw_and_net(length_row(bleu=dict(metric(), t_ci95=None))) is None


def test_wrist_and_contact_decide_nothing_for_the_hold():
    assert a6.decide_hold(hold_row(motion=dict(arm_hands=1., comparator_hands=1., arm_wrist=9., comparator_wrist=1., arm_left=9., arm_contact=0., comparator_contact=1.)))["outcome"] == "passed"
    assert a6.decide_hold(hold_row(bleu=metric(t=.1, two=-9.)))["outcome"] == "passed"             # BLEU's two-way interval does not decide


def test_unknown_numbers_fail():
    assert not a6.at_least(None, 1., .9) and not a6.at_least(1., None, .9)
    assert a6.decide_hold(hold_row(likelihood_minus_blank=dict(metric(), t_ci95=None)))["checks"]["likelihood_t_lower_bound_above_floor"] is False
    assert a6.decide_length(length_row(bleu_net_of_blank=dict(t_ci95=None, mean=0., per_draw=[])))["checks"]["bleu_lower_of_raw_and_net_t_lower_bound_above_floor"] is False


@pytest.mark.parametrize("name,key", [(a6.BOTH, "hold"), (a6.LENGTH, "length")])
def test_each_rule_reads_its_own_arm_and_hold_x_decides_nothing(name, key):
    d, s = draws(by_arm={name: dict(ll=-.5)})
    if name == a6.LENGTH:        # both stays near its comparator
        for k in a6.DRAWS:
            d[k][a6.BOTH]["likelihood"]["per_form"][FORM] = list(d[k][a6.LENGTH]["likelihood"]["per_form"][FORM])
    out = read(d, s)
    other = dict(hold="length", length="hold")[key]
    assert out["rule"][key]["outcome"] == "not_passed" and out["rule"][key]["checks"]["likelihood_t_lower_bound_above_floor"] is False
    assert out["rule"][other]["checks"]["likelihood_t_lower_bound_above_floor"] is True
    d, s = draws(by_arm={a6.HOLD: dict(ll=-.5)})
    out = read(d, s)
    assert out["rule"]["hold"]["checks"]["likelihood_t_lower_bound_above_floor"] is True and out["rule"]["length"]["checks"]["likelihood_t_lower_bound_above_floor"] is True


# ---- the screen gates the hold arms -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("likelihood,spotting,passed", [
    (.01, .01, True), (0., 0., True), (L + 1e-4, S + 1e-4, True), (L, .01, False), (.01, S, False), (-.0276, -.0405, False), (None, .01, False), (.01, None, False),
])
def test_the_gate_is_both_point_estimates_above_the_hold_floors(likelihood, spotting, passed):
    out = a6.hold_gate(screen_summary(likelihood, spotting), "abc1234")
    assert out["passed"] is passed and out["hold_arms"] == ("run on the draws" if passed else "not run, screen failed") and out["likelihood_floor"] == L and out["spotting_floor"] == S


@pytest.mark.parametrize("bad,message", [(dict(mode="draws"), "reads a screen's summary"), (dict(mode="smoke"), "reads a screen's summary"), (dict(provenance=dict(code_digest="e" * 64)), "other code"), (dict(floors=dict(bleu=-1., likelihood=-1., spotting=-1.)), "are not this code's"),
                                         (dict(declared=False), "needs a declared screen"), (dict(declared_commit="other"), "needs a declared screen")])
def test_the_gate_reads_a_declared_screen_of_this_declaration_on_this_code_and_floors(bad, message):
    with pytest.raises(SystemExit, match=message):
        a6.hold_gate(screen_summary(**bad), "abc1234")


def test_the_gate_needs_a_declared_read():
    with pytest.raises(SystemExit, match="needs a declared screen"):
        a6.hold_gate(screen_summary(), None)
    d, s = draws()
    with pytest.raises(SystemExit, match="needs a declared screen"):
        read(d, s, declared_commit=None)
    with pytest.raises(SystemExit, match="its hash"):
        read(d, s, screen_sha256=None)
    assert read(d, s, screen_sha256="abc")["screen_sha256"] == "abc"


def test_a_real_screen_summary_sets_the_gate_from_hold_x_against_base():
    d, s = draws((a6.SCREEN,))
    out = a6.read(d, s, boot=20, declared_commit="abc1234")
    assert out["hold_gate"]["passed"] is True and out["hold_gate"]["likelihood"] == pytest.approx(.01, abs=1e-6) and a6.hold_gate(out, "abc1234")["passed"] is True
    d, s = draws((a6.SCREEN,), by_arm={a6.HOLD: dict(ll=-.5)})
    out = a6.read(d, s, boot=20, declared_commit="abc1234")
    assert out["hold_gate"]["passed"] is False and a6.hold_gate(out, "abc1234")["hold_arms"] == "not run, screen failed" and out["rule"]["hold"]["outcome"] == "screen_decides_nothing"
    d, s = draws((a6.SCREEN,), by_arm={a6.BOTH: dict(ll=-.5), a6.LENGTH: dict(ll=-.5)})     # only hold-x against base gates
    assert a6.read(d, s, boot=20)["hold_gate"]["passed"] is True
    d, s = draws((a6.SCREEN,))
    for name in a6.SCREEN_ARMS:
        d[a6.SCREEN][name]["clips"] = 5
    assert a6.read(d, s, boot=20, smoke=True)["hold_gate"] is None                         # a smoke sets no gate


def closed_draws(**kw):
    d, s = draws(**kw)
    for k in a6.DRAWS:
        for name in (a6.HOLD, a6.BOTH):
            del d[k][name], s[k][name]
    return d, s


def test_a_closed_gate_runs_base_and_length_and_says_so():
    d, s = closed_draws()
    out = read(d, s, screen_summary=screen_summary(-.5, .01))
    assert out["rule"]["hold"]["outcome"] == "not run, screen failed" and out["rule"]["hold"]["checks"] is None and out["rule"]["length"]["outcome"] == "passed" and out["rule"]["length"]["checks"]
    assert set(out["arms"]) == {a6.LENGTH} and out["both_on_base"] is None and out["difference_in_differences"] is None and out["reported_only"] == [] and out["hold_gate"]["passed"] is False


def test_a_closed_gate_refuses_hold_arms_and_an_open_gate_needs_them():
    d, s = draws()
    with pytest.raises(SystemExit, match="its arms are"):
        read(d, s, screen_summary=screen_summary(-.5, .01))                                 # a hold arm that should not have run
    d, s = closed_draws()
    with pytest.raises(SystemExit, match="its arms are"):
        read(d, s)                                                                          # the screen let them run
    d, s = draws()
    with pytest.raises(SystemExit, match="need the screen's summary"):
        a6.read(d, s, boot=20, a5_top10=tops(d), declared_commit="abc1234")
    assert read(d, s)["rule"]["hold"]["outcome"] == "passed" and read(d, s)["hold_gate"]["passed"] is True


def test_the_reader_prints_not_run_and_parses_the_closed_arms(tmp_path, monkeypatch, capsys):
    import round20_compare
    from signtranslator.data.text_motion_dataset import DirectoryStore
    import train_text_to_motion
    store = DirectoryStore(tmp_path)
    d, s = closed_draws()
    argv = []
    for k, arms in d.items():
        for name, report in arms.items():
            store.put(f"{k}|{name}", json.dumps(report).encode())
        argv += ["--draw", f"{k}=" + ",".join(f"{k}|{n}" for n in a6.GATE_CLOSED_ARMS), "--a5-top10", f"{k}={k}|{a6.BASE}"]
    raw = json.dumps(screen_summary(.01, -.5)).encode()
    store.put("screen.json", raw)
    monkeypatch.setattr(train_text_to_motion, "make_store", lambda name: store)
    monkeypatch.setattr(round20_compare, "report_shards", lambda st, key, report, _: s[int(key.split("|")[0])][key.split("|")[1]])
    args = [*argv, "--screen-key", "screen.json", "--declared-commit", "abc1234", "--bootstrap", "20"]
    with pytest.raises(SystemExit, match="does not match --screen-sha256"):
        a6.main([*args, "--screen-sha256", "0" * 64, "--output-key", "bad.json"])
    assert a6.main([*args, "--screen-sha256", hashlib.sha256(raw).hexdigest(), "--output-key", "out.json"]) == 0
    printed = capsys.readouterr().out
    assert "hold not run, screen failed" in printed and "COMP-A6-MATCH draw=1 identical=12/12 rate=1.0 same_gpu=True" in printed and "COMP-A6-MATCH draw=8 " in printed
    out = json.loads(store.get("out.json"))
    assert out["rule"]["hold"]["outcome"] == "not run, screen failed" and out["screen_sha256"] == hashlib.sha256(raw).hexdigest()


def test_the_guards_gate_exit_codes(tmp_path, monkeypatch, capsys):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    import train_text_to_motion
    store = DirectoryStore(tmp_path)
    monkeypatch.setattr(train_text_to_motion, "make_store", lambda name: store)
    store.put("open.json", json.dumps(screen_summary()).encode())
    store.put("closed.json", json.dumps(screen_summary(-.5, .01)).encode())
    assert guard.main(["hold-gate", "open.json", "abc1234"]) == 0 and "COMP-A6-HOLD-GATE open" in capsys.readouterr().out
    assert guard.main(["hold-gate", "closed.json", "abc1234"]) == 3 and "not run, screen failed" in capsys.readouterr().out
    store.put("bad.json", json.dumps(dict(mode="draws")).encode())
    with pytest.raises(SystemExit, match="reads a screen's summary"):
        guard.main(["hold-gate", "bad.json", "abc1234"])
    with pytest.raises(SystemExit, match="needs a declared screen"):
        guard.main(["hold-gate", "open.json", "other"])


# ---- what counts ------------------------------------------------------------------------------------------------------------------------------------

def bend(d, k, name, fn):
    bent = copy.deepcopy(d)
    fn(bent[k][name])
    return bent


@pytest.mark.parametrize("keep", [(1, 2, 3, 4, 5, 6, 7), (2, 3, 4, 5, 6, 7, 8), (1, 2, 3), ()])
def test_anything_but_all_eight_is_incomplete(keep):
    d, _ = draws()
    with pytest.raises(SystemExit, match="incomplete"):
        a6.check_set({k: d[k] for k in keep})


def test_nine_draws_or_a_screen_beside_draws_is_refused():
    d, _ = draws()
    bent = {**d, 9: d[1]}
    with pytest.raises(SystemExit, match="incomplete"):
        a6.check_set(bent)
    both = {**d, a6.SCREEN: draws((a6.SCREEN,))[0][a6.SCREEN]}
    with pytest.raises(SystemExit, match="incomplete"):
        a6.check_set(both)


@pytest.mark.parametrize("name", a6.ARMS)
def test_a_missing_arm_stops_the_draw(name):
    d, _ = draws()
    del d[2][name]
    with pytest.raises(SystemExit, match="its arms are"):
        a6.check_set(d)


@pytest.mark.parametrize("change,message", [
    (lambda r: r.pop("report_only"), "not report-only"),
    (lambda r: r.update(clips=499), "not all of D20"),
    (lambda r: r["settings"]["round20"]["sentences"].update(key="round20/c20.json"), "D20 only"),
    (lambda r: r["settings"]["round20"]["provenance"].update(code_digest="e" * 64), "other code or another environment"),
    (lambda r: r["settings"]["round20"]["provenance"].update(environment=dict(PROVENANCE["environment"], gpu="other")), "other code or another environment"),
    (lambda r: r["settings"]["round20"]["vocabulary_extra"].update(top=30), "is not what"),
    (lambda r: r["settings"]["round20"].pop("word_cache_extra"), "is not what"),
])
@pytest.mark.parametrize("name", a6.ARMS)
def test_a_draws_reports_are_checked(change, message, name):
    d, _ = draws()
    with pytest.raises(SystemExit, match=message):
        a6.check_set(bend(d, 3, name, change))


X = dict(sha256=a6.HOLD_SHA, exclude=.25)


@pytest.mark.parametrize("name,change", [
    (a6.BASE, lambda r: r["settings"]["round20"].update(hold_left=dict(X))),
    (a6.BASE, lambda r: r["settings"]["round20"].update(duration_balanced=True)),
    (a6.HOLD, lambda r: r["settings"]["round20"].update(hold_left=dict(sha256=a6.HOLD_SHA))),           # A1's plain hold is not the arm
    (a6.HOLD, lambda r: r["settings"]["round20"].update(hold_left=dict(X, exclude=.5))),
    (a6.HOLD, lambda r: r["settings"]["round20"].update(hold_left=dict(X, sha256="0" * 64))),
    (a6.HOLD, lambda r: r["settings"]["round20"]["hold_left"].update(rest_sha256=a6.REST_SHA)),
    (a6.HOLD, lambda r: r["settings"]["round20"].update(duration_balanced=True)),
    (a6.LENGTH, lambda r: r["settings"]["round20"].pop("duration_balanced")),
    (a6.LENGTH, lambda r: r["settings"]["round20"].update(hold_left=dict(X))),
    (a6.BOTH, lambda r: r["settings"]["round20"].pop("duration_balanced")),
    (a6.BOTH, lambda r: r["settings"]["round20"].pop("hold_left")),
    (a6.BOTH, lambda r: r["settings"]["round20"]["hold_left"].update(rest_sha256=a6.REST_SHA)),
])
def test_an_arm_is_what_its_name_says(name, change):
    d, _ = draws()
    with pytest.raises(SystemExit, match="is not what"):
        a6.check_set(bend(d, 2, name, change))


@pytest.mark.parametrize("change", [lambda r: r["settings"]["round20"]["hold_left"].pop("rest_sha256"), lambda r: r["settings"]["round20"]["hold_left"].update(rest_sha256="0" * 64), lambda r: r["settings"]["round20"].update(duration_balanced=True)])
def test_hold_xr_is_what_its_name_says(change):
    d, _ = draws((a6.SCREEN,))
    with pytest.raises(SystemExit, match="is not what"):
        a6.check_set(bend(d, a6.SCREEN, a6.HOLD_REST, change))


@pytest.mark.parametrize("name", [a6.LENGTH, a6.BOTH])
def test_a_fitted_arm_has_the_committed_fit(name):
    d, _ = draws()
    with pytest.raises(SystemExit, match="not the committed balanced one"):
        a6.check_set(bend(d, 2, name, lambda r: r["round20"]["duration_model"].update(slope=.3)))
    with pytest.raises(SystemExit, match="not the committed balanced one"):
        a6.check_set(bend(d, 2, name, lambda r: r["round20"]["duration_model"].update(signer_balanced=False)))


@pytest.mark.parametrize("name", [a6.BASE, a6.HOLD])
@pytest.mark.parametrize("change", [dict(signer_balanced=True), dict(slope=.3), dict(intercept=0.), dict(records=1)])
def test_an_arm_at_todays_length_has_todays_fit(name, change):
    d, _ = draws()
    with pytest.raises(SystemExit, match="not today's"):
        a6.check_set(bend(d, 2, name, lambda r: r["round20"]["duration_model"].update(change)))


@pytest.mark.parametrize("name", a6.ARMS[1:])
def test_an_undeclared_difference_stops_the_pair(name):
    d, _ = draws()
    with pytest.raises(SystemExit, match="undeclared differences"):
        a6.check_set(bend(d, 2, name, lambda r: r["settings"].update(core_margin=.1)))


def test_a_draw_is_that_draws_words_on_every_arm():
    d, _ = draws()
    bent = copy.deepcopy(d)
    for name in a6.ARMS:
        bent[3][name]["settings"]["round20"]["word_cache"]["draw"] = 2
    with pytest.raises(SystemExit, match="is draw 2"):
        a6.check_set(bent)
    bent = copy.deepcopy(d)
    for name in a6.ARMS:
        bent[3][name]["settings"]["round20"]["word_cache"].pop("draw")
    with pytest.raises(SystemExit, match="is draw None"):
        a6.check_set(bent)
    with pytest.raises(SystemExit, match="undeclared differences .*word_cache"):
        a6.check_set(bend(d, 3, a6.LENGTH, lambda r: r["settings"]["round20"]["word_cache"].update(sha256="other")))


def test_draws_share_code_and_packages_may_differ_in_gpu_and_never_share_words():
    d, _ = draws()
    gpu = copy.deepcopy(d)
    for name in a6.ARMS:
        gpu[4][name]["settings"]["round20"]["provenance"]["environment"] = dict(PROVENANCE["environment"], gpu="RTX 6000 Ada")
    a6.check_set(gpu)
    for field, value in (("code_digest", "e" * 64), ("environment", dict(PROVENANCE["environment"], transformers="9"))):
        bent = copy.deepcopy(d)
        for name in a6.ARMS:
            bent[4][name]["settings"]["round20"]["provenance"][field] = value
        with pytest.raises(SystemExit, match="other code or packages"):
            a6.check_set(bent)
    bent = copy.deepcopy(d)
    for name in a6.ARMS:
        bent[2][name]["settings"]["round20"]["word_cache"]["sha256"] = "cache1"
    with pytest.raises(SystemExit, match="share a word cache"):
        a6.check_set(bent)
    bent = copy.deepcopy(d)
    for name in a6.ARMS:
        bent[2][name]["settings"]["round20"]["word_cache_extra"]["sha256"] = "extra1"
    with pytest.raises(SystemExit, match="share a word cache"):
        a6.check_set(bent)
    bent = copy.deepcopy(d)
    for name in a6.ARMS:
        bent[2][name]["references"] = list(reversed(bent[2][name]["references"]))
    with pytest.raises(SystemExit, match="other sentences"):
        a6.check_set(bent)


def test_the_read_keeps_the_attempts_and_refuses_to_overwrite(tmp_path, monkeypatch):
    import round20_compare
    from signtranslator.data.text_motion_dataset import DirectoryStore
    import train_text_to_motion
    store = DirectoryStore(tmp_path)
    d, s = draws()
    argv = []
    for k, arms in d.items():
        for name, report in arms.items():
            store.put(f"{k}|{name}", json.dumps(report).encode())
        argv += ["--draw", f"{k}=" + ",".join(f"{k}|{n}" for n in a6.ARMS), "--a5-top10", f"{k}={k}|{a6.BASE}"]
    monkeypatch.setattr(train_text_to_motion, "make_store", lambda name: store)
    store.put("screen.json", json.dumps(screen_summary()).encode())
    argv += ["--screen-key", "screen.json"]
    monkeypatch.setattr(round20_compare, "report_shards", lambda st, key, report, _: s[int(key.split("|")[0])][key.split("|")[1]])
    raw = json.dumps(screen_summary()).encode()
    store.put("screen.json", raw)
    args = [*[x for x in argv if x not in ("--screen-key", "screen.json")], "--screen-key", "screen.json", "--screen-sha256", hashlib.sha256(raw).hexdigest(), "--declared-commit", "abc1234", "--attempts", "3=1:base+;2:base+hold-x+length+hold-x-length+;", "--bootstrap", "30", "--output-key", "out.json"]
    assert a6.main(args) == 0
    out = json.loads(store.get("out.json"))
    assert out["attempts"] == {"3": "1:base+;2:base+hold-x+length+hold-x-length+;"} and out["rule"]["hold"]["outcome"] in ("passed", "not_passed") and out["a5_instrument_check"]["draws"]["1"]["form_identical"] == 12
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        a6.main(args)
    for bad in (["--draw", "x=a,b,c,d"], ["--draw", "1=a,b,c"], ["--draw", "a3=a,b,c,d"], [*argv, "--a5-top10", "9=k"], [*argv, "--a5-top10", "x=k"]):
        with pytest.raises(SystemExit):
            a6.main([*bad, "--output-key", "o2.json"])


# ---- its own declaration --------------------------------------------------------------------------------------------------------------------------

def test_a6_has_its_own_declaration_and_digest(tmp_path, monkeypatch):
    from signtranslator.eval.provenance import code_digest
    root = tmp_path / "bundle"
    (root / "docs").mkdir(parents=True)
    (root / "runpod/fleet").mkdir(parents=True)
    monkeypatch.setattr(a6, "A5_SUMMARY_AT_DECLARATION", dict(sha256="h", outcome="confirmed"))
    monkeypatch.setattr(dc6, "check_power", lambda root: None)
    with pytest.raises(SystemExit, match="nothing shows a declaration"):
        dc6.check("abc1234", root)
    (root / "docs/NEXT_PHASE_PLAN.md").write_text("Plan. " + a6.RULES["hold"] + " " + a6.RULES["length"])
    record = dc6.make_record("abc1234")
    assert record["code_digest"] == code_digest(extra_files=JOBS) != code_digest(extra_files=JOBS[:1]) and set(record["rules"]) == {"hold", "length"}
    (root / "runpod/fleet/comp_a5_declared.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="nothing shows a declaration"):
        dc6.check("abc1234", root)
    (root / dc6.RECORD).write_text(json.dumps(record))
    assert dc6.check("abc1234", root) == "record"
    with pytest.raises(SystemExit, match="names abc1234"):
        dc6.check("abc1235", root)
    (root / dc6.RECORD).write_text(json.dumps(dict(record, rules=dict(record["rules"], length="0"))))
    with pytest.raises(SystemExit, match="does not hold this rule"):
        dc6.check("abc1234", root)
    (root / dc6.RECORD).write_text(json.dumps(record))
    (root / "docs/NEXT_PHASE_PLAN.md").write_text("Plan. " + a6.RULES["hold"])
    with pytest.raises(SystemExit, match="plan does not hold"):
        dc6.check("abc1234", root)


def test_a5s_files_are_untouched_by_a6():
    if subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "main"], capture_output=True).returncode:
        pytest.skip("no main branch in this checkout")
    run = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", "main", "--", "scripts/comp_a5_read.py", "scripts/comp_a5_guard.py", "scripts/comp_a5_declared.py", "runpod/fleet/comp_a5.sh", "runpod/fleet/comp_a5_chain.sh", "runpod/fleet/comp_a5_declared.json"],
                         capture_output=True, text=True)
    assert run.returncode == 0 and run.stdout == "", run.stderr


def test_the_job_says_the_launch_waits_and_names_a5s_record():
    for phrase in ("THE LAUNCH MUST WAIT", "08b9a150", "Don't merge it into A5's code"):
        assert phrase in JOB.read_text(), phrase


# ---- the guard -------------------------------------------------------------------------------------------------------------------------------------

def stored_base(**kw):
    return dict(settings=dict(round20=dict(provenance=dict(code_digest="d", environment=dict(a5.PINNED, gpu="L40S"), **kw))))


def test_the_same_pod_guard(monkeypatch):
    monkeypatch.setattr(lib, "provenance_row", lambda job: dict(code_digest="d", environment=dict(a5.PINNED, gpu="L40S")) if job == a6.JOB_FILE else None)
    guard.check_same_pod(stored_base())
    gpu = stored_base()
    gpu["settings"]["round20"]["provenance"]["environment"]["gpu"] = "RTX 6000 Ada"
    code = stored_base()
    code["settings"]["round20"]["provenance"]["code_digest"] = "e"
    for bad in (gpu, code, dict(settings=dict(round20=dict())), dict(settings=None), {}):
        with pytest.raises(SystemExit, match="other code or another environment"):
            guard.check_same_pod(bad)


def test_the_guards_commands(tmp_path, monkeypatch):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    from signtranslator.eval import provenance
    import train_text_to_motion
    store = DirectoryStore(tmp_path)
    monkeypatch.setattr(train_text_to_motion, "make_store", lambda name: store)
    monkeypatch.setattr(lib, "provenance_row", lambda job: dict(code_digest="d", environment=dict(a5.PINNED, gpu="L40S")))
    assert guard.main(["started", "x/a.started"]) == 0 and store.exists("x/a.started")
    with pytest.raises(SystemExit, match="exists"):
        guard.main(["started", "x/a.started"])
    store.put("m.json", json.dumps(stored_base()).encode())
    assert guard.main(["same-pod", "m.json"]) == 0
    monkeypatch.setattr(provenance, "environment", lambda: dict(a5.PINNED, gpu="any"))
    assert guard.main(["env"]) == 0
    monkeypatch.setattr(provenance, "environment", lambda: dict(a5.PINNED, torch="9"))
    with pytest.raises(SystemExit, match="not the pinned"):
        guard.main(["env"])
    for bad in ([], ["other"], ["env", "x"], ["started"], ["same-pod"], ["a5-done", "x"]):
        with pytest.raises(SystemExit, match="usage"):
            guard.main(bad)


def a5_summary(outcome):
    row = lambda mean: dict(mean=mean, between_draw_sd=.1, t_ci95=[mean / 2, 9.])       # noqa: E731
    return json.dumps(dict(rule=dict(outcome=outcome), forms=dict(predicted=dict(bleu=row(.3), likelihood_minus_blank=row(.012), spotting=row(.014))))).encode()


def slot_of(raw):
    return a6.slot_from_summary(json.loads(raw), hashlib.sha256(raw).hexdigest())


def test_a6s_full_runs_need_a5_confirmed_or_direction_only_and_the_slot_exactly(tmp_path, monkeypatch):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    import train_text_to_motion
    store = DirectoryStore(tmp_path)
    monkeypatch.setattr(train_text_to_motion, "make_store", lambda name: store)
    with pytest.raises(SystemExit, match="A6's draws wait"):
        guard.main(["a5-done"])
    for outcome in ("confirmed", "direction_only_not_confirmed"):
        raw = a5_summary(outcome)
        guard.check_a5_summary(raw, slot_of(raw))
    for outcome in ("not_confirmed", None):
        raw = a5_summary(outcome)
        with pytest.raises(SystemExit, match="A6's draws run only on"):
            guard.check_a5_summary(raw, slot_of(raw))
    raw = a5_summary("confirmed")
    store.put(guard.A5_SUMMARY, raw)
    with pytest.raises(SystemExit, match="the bars were not set against it"):                 # an empty slot: nothing runs
        guard.main(["a5-done"])
    monkeypatch.setattr(guard, "A5_SUMMARY_AT_DECLARATION", slot_of(raw))
    assert guard.main(["a5-done"]) == 0
    for change in (dict(sha256="other"), dict(bleu=.31), dict(paired_sd=dict(bleu=.2, likelihood=.1, spotting=.1)), dict(t_lower=dict(bleu=0., likelihood=.006, spotting=.007))):
        monkeypatch.setattr(guard, "A5_SUMMARY_AT_DECLARATION", dict(slot_of(raw), **change))
        with pytest.raises(SystemExit, match="the bars were not set against it"):             # any field the summary does not hold
            guard.main(["a5-done"])
    decided = a5.decide(dict(bleu=dict(t_ci95=[.1], mean=.3, positive_draws=8), likelihood_minus_blank=dict(two_way_ci95=[.1], t_ci95=[.1], mean=.1, positive_draws=8), spotting=dict(two_way_ci95=[.1]), motion=dict(top10_wrist=1.5, top10_hands=1., margin0_hands=1.)), 1.84)
    assert decided["outcome"] in a6.A5_GOES_ON                                     # the names A5's own rule writes


def matched(gpu_a="L40S", gpu_b="L40S", change=None):
    a, b = arm(a6.BASE, 3, hyps={FORM: SIX}), arm(a6.BASE, 3, hyps={FORM: SIX})
    a["settings"]["round20"]["provenance"] = dict(PROVENANCE, environment=dict(PROVENANCE["environment"], gpu=gpu_a))
    b["settings"]["round20"]["provenance"] = dict(PROVENANCE, environment=dict(PROVENANCE["environment"], gpu=gpu_b))
    if change:
        b["forms"][change]["hypotheses"][0] = "another hypothesis"
    return guard.match_counts(a, b)


def test_base_must_match_its_top10_at_98_percent_on_the_same_gpu():
    assert not guard.match_stops(matched()) and matched()["rate"] == 1. and matched()["threads"] == a6.PINNED_THREADS
    for form, name in ((FORM, "form"), ("real", "real"), (a6.BLANK, "blank_predicted")):
        counts = matched(change=form)
        assert guard.match_stops(counts) and counts[name] == 11 and counts["rate"] == .9167
        assert not guard.match_stops(matched(gpu_b="RTX", change=form))                       # another GPU may differ, and the counts say so
    assert not guard.match_stops(dict(same_gpu=True, rate=.98)) and guard.match_stops(dict(same_gpu=True, rate=.9799))
    assert matched(gpu_b="RTX")["same_gpu"] is False


def test_the_match_command_returns_4_on_a_stop_and_crashes_as_a_crash(tmp_path, monkeypatch, capsys):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    import train_text_to_motion
    store = DirectoryStore(tmp_path)
    monkeypatch.setattr(train_text_to_motion, "make_store", lambda name: store)
    a, b = arm(a6.BASE, 3), arm(a6.BASE, 3)
    store.put("a.json", json.dumps(a).encode())
    store.put("b.json", json.dumps(b).encode())
    assert guard.main(["a5-match", "a.json", "b.json"]) == 0
    assert "COMP-A6-MATCH" in capsys.readouterr().out
    b["forms"][FORM]["hypotheses"][1] = "another"
    store.put("c.json", json.dumps(b).encode())
    assert guard.main(["a5-match", "a.json", "c.json"]) == 4
    store.put("d.json", json.dumps(dict(forms={})).encode())
    with pytest.raises(KeyError):                                                             # uncaught: exit 1, which the job calls a crash
        guard.main(["a5-match", "a.json", "d.json"])


def test_a_top10_names_its_word_caches(tmp_path, monkeypatch, capsys):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    import train_text_to_motion
    store = DirectoryStore(tmp_path)
    monkeypatch.setattr(train_text_to_motion, "make_store", lambda name: store)
    store.put("t.json", json.dumps(arm(a6.BASE, 3)).encode())
    assert guard.main(["a5-caches", "t.json"]) == 0
    assert capsys.readouterr().out.split() == ["k", "cache3", "x", "extra3"]


# ---- a smoke through the reader's whole path ------------------------------------------------------------------------------------------------------

def test_five_smoke_reports_run_merge_and_read_and_print_no_numbers(d20, monkeypatch, recogniser, capsys):  # noqa: F811
    from tests.test_comp_a import token_rows, write_list
    import train_text_to_motion
    c = d20
    path, sha = write_list(c, token_rows(c, monkeypatch))
    extra = ("--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top", "10")
    main = cache_for_words(c, "cache/a3")
    other = cache_for_words(c, "cache/a3x", *extra)
    monkeypatch.setattr(a6, "A3_CACHE_SHA", main)
    monkeypatch.setattr(a6, "A3_EXTRA_SHA", other)
    monkeypatch.setattr(a6, "TODAY_FIT", (1., .1, 18))
    monkeypatch.setattr(a6, "BALANCED_FIT", (.5, .2, 18))
    monkeypatch.setattr(cv2, "fit_duration_model", lambda *a, **k: dict(corpus=cv2.H2S, partition="train", records=18, median_abs_error=1., **(dict(intercept=.5, slope=.2, signer_balanced=True) if k.get("signer_balanced") else dict(intercept=1., slope=.1))))
    hold = ("--hold-left-file", str(ROOT / "reports/comp_a_hold_stats.json"), "--hold-left-sha256", a6.HOLD_SHA, "--hold-left-exclude", "0.25")
    rest = ("--hold-rest-file", str(ROOT / "reports/comp_a6_rest_pose.json"), "--hold-rest-sha256", a6.REST_SHA)
    today, fitted = ("--expect-duration-model=1,.1,18",), ("--duration-balanced", "--expect-duration-model=.5,.2,18")
    keys = []
    for name, more in ((a6.BASE, today), (a6.HOLD, (*today, *hold)), (a6.LENGTH, fitted), (a6.BOTH, (*fitted, *hold)), (a6.HOLD_REST, (*today, *hold, *rest))):
        args = generated_args(c, "--word-cache-key", "cache/a3/index.json", "--word-cache-sha256", main, "--word-cache-n", "4", "--core-margin", "0", "--predicted-duration", *extra,
                              "--word-cache-extra-key", "cache/a3x/index.json", "--word-cache-extra-sha256", other, "--comp-a-job-file", a6.JOB_FILE, "--clips", "4", *more, "--output-key", f"s/{name}-shard0.json")
        assert ce.main(args) == 0
        assert ce.main(["merge", "--store", f"dir:{c['tmp'] / 'store'}", "--report-only", "--shard-keys", f"s/{name}-shard0.json", "--bootstrap", "20", "--output-key", f"s/{name}.json"]) == 0
        keys.append(f"s/{name}.json")
    args = generated_args(c, "--word-cache-key", "cache/a3/index.json", "--word-cache-sha256", main, "--word-cache-n", "4", "--core-margin", "0", "--predicted-duration", *extra,
                          "--word-cache-extra-key", "cache/a3x/index.json", "--word-cache-extra-sha256", other, "--comp-a-job-file", a6.JOB_FILE, "--clips", "4", *today, *hold[:4], "--output-key", "s/plain-shard0.json")
    assert ce.main(args) == 0
    assert ce.main(["merge", "--store", f"dir:{c['tmp'] / 'store'}", "--report-only", "--shard-keys", "s/plain-shard0.json", "--bootstrap", "20", "--output-key", "s/plain.json"]) == 0
    assert "hold_left" in part(c, "s/plain.json") and "hold_left_predicted" not in part(c, "s/plain.json")       # A1's plain hold keeps its report's shape
    capsys.readouterr()
    monkeypatch.setattr(train_text_to_motion, "make_store", lambda name: c["store"])
    assert a6.main(["--smoke", "--draw", "a3=" + ",".join(keys), "--bootstrap", "20", "--store", "x", "--output-key", "s/read.json"]) == 0
    printed = capsys.readouterr().out
    assert "smoke" in printed and not re.search(r"\d\.\d", printed)
    out = part(c, "s/read.json")
    assert out["mode"] == "smoke" and out["rule"]["hold"]["outcome"] == "smoke_decides_nothing" and set(out["arms"]) == {a6.HOLD, a6.LENGTH, a6.BOTH, a6.HOLD_REST}
    assert out["both_on_base"]["comparator"] == a6.BASE and "hold_left_predicted" in part(c, f"s/{a6.BOTH}.json") and "hold_left_predicted" not in part(c, f"s/{a6.BASE}.json")
    full = part(c, f"s/{a6.BOTH}.json")
    assert full["hold_left_predicted"]["stretches"] <= full["hold_left"]["stretches"]
    with pytest.raises(SystemExit):                                                 # the same reports are no full read
        a6.main(["--draw", "a3=" + ",".join(keys), "--bootstrap", "20", "--store", "x", "--output-key", "s/full.json"])


def test_a_smoke_read_refuses_draws_and_more_than_five_clips():
    d, s = draws()
    with pytest.raises(SystemExit, match="A3's words alone"):
        a6.read(d, s, boot=20, smoke=True, a5_top10=tops(d), screen_summary=screen_summary(), screen_sha256="h", declared_commit="abc1234")
    d, s = draws((a6.SCREEN,))
    for name in a6.SCREEN_ARMS:
        d[a6.SCREEN][name]["clips"] = 6
    with pytest.raises(SystemExit, match="at most 5 clips"):
        a6.read(d, s, boot=20, smoke=True)


# ---- the job on stubs: no real job runs ---------------------------------------------------------------------------------------------------------------

A3C = f"{O}/round20/phase-a/cache-w14b-nogains-r10/index.json"
A3X = f"{O}/round20/comp-a/cache-a3-top30-nogains/index.json"
A3TOP = f"{O}/round20/comp-a/a3-gen-top10.json"
SHAS = {f"{p}{d}_SHA": f"{p.lower()}{d}" for p in ("B", "HX", "L", "HXL") for d in a6.DRAWS}
A5T = {f"A5T{d}_SHA": "stubsha" for d in a6.DRAWS}
SCREEN_SHAS = {f"{p}a3_SHA": f"{p.lower()}a3" for p in ("B", "HX", "L", "HXL", "HXR")}
PINS = dict(BASE_SHA="b", LENGTH_SHA="l")


def key(words, name, attempt=1):
    return f"{OUT}/{words}-attempt{attempt}-{name}.json"


def caches(k=3, attempt=1):
    return [f"{A5OUT}/draw{k}-cache/attempt{attempt}/index.json", f"{A5OUT}/draw{k}-extra/attempt{attempt}/index.json"]


def a5_top10(k, attempt=1):
    return f"{A5OUT}/draw{k}-attempt{attempt}-top10.json"


CACHE_LINE = "ck csha xk xsha"
DRAW_ENV = dict(A5T_SHA="stubsha", SCREEN_SHA="s", STUB_PRINT=CACHE_LINE, STUB_PRINT_MATCH="comp_a6_guard.py a5-caches")
SCREEN_KEY = f"{OUT}/a6-summary-a3.json"
CLOSED = dict(STUB_EXIT_MATCH="comp_a6_guard.py hold-gate", STUB_EXIT_CODE="3")


def job(tmp_path, phase, *, words="a3", smoke=False, gpu=True, stored=(), declared="abc1234", text=None, top=True, **env):
    numeric = words not in (None, "a3", "draws", "0", "9")
    e = dict(COMP_PHASE=phase, STUB_EXISTS=" ".join([*stored, *([SCREEN_KEY] if numeric or words == "draws" else []), *([a5_top10(words)] if numeric and top else [])]), STUB_GATE="1", STUB_FAIL_SHARD="", STUB_HANG="")
    e.update(env)
    if words:
        e["WORDS"] = words
    if declared:
        e["COMP_A_DECLARED"] = declared
    return on_stubs(tmp_path, text or JOB.read_text(), e, smoke=smoke, gpu=gpu)


def one_done(out, code):
    lines = DONE.findall(out)
    assert len(lines) == 1 and lines[0][0] == str(code), out
    assert out.strip().splitlines()[-1].startswith("COMP-A6-DONE")


def run_line(commands):
    return next(c for c in commands if "round15_compose_eval.py run" in c)


def parsed(tmp_path, monkeypatch, line, name):
    seen = {}
    monkeypatch.setattr(ce, "run", lambda args: seen.update(vars(args)) or {})
    assert ce.main(line.split()[1:] + ["--store", f"dir:{tmp_path / ('s-' + name)}"]) == 0
    return seen


def test_the_jobs_parse_and_start_no_pod():
    for path in (JOB, CHAIN):
        subprocess.run(["bash", "-n", str(path)], check=True)
        for needle in ("runpodctl", "podCreate", "run_job_pod", "create-pod"):
            assert needle not in path.read_text()


def test_the_jobs_pins_are_the_committed_files():
    text = JOB.read_text()
    for name, path in (("TOKENSSHA", "reports/comp_a3_tokens.json"), ("HOLDSHA", "reports/comp_a_hold_stats.json"), ("RESTSHA", "reports/comp_a6_rest_pose.json")):
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == re.search(rf"{name}=([0-9a-f]{{64}})", text).group(1)
    assert re.search(r"HOLDSHA=(\S+)", text).group(1) == a6.HOLD_SHA and re.search(r"RESTSHA=(\S+)", text).group(1) == a6.REST_SHA
    assert re.search(r"A3CACHESHA=(\S+)", text).group(1) == a6.A3_CACHE_SHA and re.search(r"A3EXTRASHA=(\S+)", text).group(1) == a6.A3_EXTRA_SHA
    assert "c20" not in "\n".join(x for x in text.splitlines() if "SET:" not in x and not x.lstrip().startswith("#")).lower()
    assert re.search(r"uv pip install -q 'transformers==(\S+)' 'tokenizers==(\S+)' sentencepiece", text).groups() == (a5.PINNED["transformers"], a5.PINNED["tokenizers"])
    assert f"JOBFILE={a6.JOB_FILE}\n" in text


COMPARATOR_KEY = {a6.BASE: None, a6.HOLD: a6.BASE, a6.LENGTH: a6.BASE, a6.BOTH: a6.LENGTH, a6.HOLD_REST: a6.BASE}


@pytest.mark.parametrize("phase", a6.SCREEN_ARMS)
def test_each_arm_is_top10_on_the_stored_words(tmp_path, monkeypatch, phase):
    code, out, commands = job(tmp_path, phase, stored=[key("a3", COMPARATOR_KEY[phase])] if COMPARATOR_KEY[phase] else [], **PINS)
    assert code == 0
    one_done(out, 0)
    seen = parsed(tmp_path, monkeypatch, run_line(commands), phase)
    assert seen["word_cache_key"] == A3C and seen["word_cache_sha256"] == a6.A3_CACHE_SHA and seen["word_cache_extra_key"] == A3X and seen["word_cache_extra_sha256"] == a6.A3_EXTRA_SHA and seen["vocabulary_extra_top"] == 10
    assert seen["word_cache_n"] == 16 and seen["core_margin"] == 0. and seen["predicted_duration"] and seen["clips"] == 500 and seen["likelihood"] and seen["comp_a_job_file"] == a6.JOB_FILE and seen["variants"] == "core_crossfade"
    assert seen["sentences_key"] == lib.D20_KEY and seen["recognizer_run"] == "sign-recognizer-v2" and seen["arm_scale"] is None and seen["checkpoint_key"] is None
    fitted, held = phase in (a6.LENGTH, a6.BOTH), phase in (a6.HOLD, a6.BOTH, a6.HOLD_REST)
    assert seen["duration_balanced"] is fitted and seen["expect_duration_model"] == ("-0.3577,0.3445,18289" if fitted else "-0.083,0.3691,18289")
    assert seen["hold_left_file"] == ("reports/comp_a_hold_stats.json" if held else None) and seen["hold_left_exclude"] == (.25 if held else None)
    assert seen["hold_rest_file"] == ("reports/comp_a6_rest_pose.json" if phase == a6.HOLD_REST else None) and seen["hold_rest_sha256"] == (a6.REST_SHA if phase == a6.HOLD_REST else None)
    assert "--report-only" in next(c for c in commands if "round15_compose_eval.py merge" in c)
    assert f"--output-key {key('a3', phase)[:-5]}-shard0.json" in run_line(commands)
    assert f"COMP-A6 a3-attempt1-{phase} {key('a3', phase)} stubsha" in out
    assert not any("round20_word_cache" in c for c in commands)           # no word is generated


def hashed_job(by_key):
    """The job with A5's cache lookup reading these hashes, one per key."""
    table = " ".join(f"{k}={v}" for k, v in by_key.items())
    fake = 'fake_sha() { local p; for p in ' + table + '; do [ "${p%%=*}" = "$1" ] && { echo "${p#*=}"; return 0; }; done; echo none; }\n'
    text = JOB.read_text().replace('[ "$(sha $key)" = "$2" ]', '[ "$(fake_sha $key)" = "$2" ]')
    assert "fake_sha" in text
    return text.replace("set -uo pipefail\n", "set -uo pipefail\n" + fake, 1)


def test_a_draws_words_come_from_its_top10_found_by_hash(tmp_path, monkeypatch):
    stored = [a5_top10(3, 1), a5_top10(3, 2)]
    by = {a5_top10(3, 1): "old", a5_top10(3, 2): "new"}
    code, _, commands = job(tmp_path, "base", words="3", stored=stored, text=hashed_job(by), top=False, **dict(DRAW_ENV, A5T_SHA="new"))
    assert code == 0 and f"scripts/comp_a6_guard.py a5-caches {a5_top10(3, 2)}" in commands
    seen = parsed(tmp_path, monkeypatch, run_line(commands), "d3")
    assert (seen["word_cache_key"], seen["word_cache_sha256"], seen["word_cache_extra_key"], seen["word_cache_extra_sha256"]) == tuple(CACHE_LINE.split())
    assert "pin ck" in commands and "pin xk" in commands
    code, _, commands = job(tmp_path, "base", words="3", stored=stored, text=hashed_job(by), top=False, **dict(DRAW_ENV, A5T_SHA="old"))
    assert code == 0 and f"scripts/comp_a6_guard.py a5-caches {a5_top10(3, 1)}" in commands
    code, out, commands = job(tmp_path, "base", words="3", stored=stored, text=hashed_job(by), top=False, **dict(DRAW_ENV, A5T_SHA="other"))
    assert code == 1 and "no stored top10 of A5's has A5T_SHA" in out and "round15_compose_eval.py" not in "\n".join(commands)


@pytest.mark.parametrize("env,message", [(dict(DRAW_ENV, STUB_FAIL_MATCH="comp_a6_guard.py a5-caches"), "cannot read the word caches"), (dict(DRAW_ENV, STUB_PRINT="only three words"), "names no word caches"), ({k: v for k, v in DRAW_ENV.items() if k != "A5T_SHA"}, "give A5T_SHA")])
def test_a_draw_without_its_words_named_stops(tmp_path, env, message):
    code, out, commands = job(tmp_path, "base", words="3", **env)
    assert code == 1 and message in out and "round15_compose_eval.py" not in "\n".join(commands)
    one_done(out, 1)


def test_a_draw_needs_its_caches_pinned_and_a5s_summary(tmp_path):
    code, out, _ = job(tmp_path, "base", words="3", STUB_PIN_FAIL="ck", **DRAW_ENV)
    assert code == 1 and "differs from its pin" in out


@pytest.mark.parametrize("words,phase", [("a3", "base"), ("3", "base"), ("a3", "summary"), ("draws", "summary"), ("3", "gate")])
def test_every_full_phase_waits_for_a5(tmp_path, words, phase):
    code, out, commands = job(tmp_path, phase, words=words, STUB_FAIL_MATCH="comp_a6_guard.py a5-done", **DRAW_ENV)
    assert code == 1 and "A6's full runs wait for A5's summary" in out and "round15_compose_eval.py" not in "\n".join(commands) and "comp_a6_read.py" not in "\n".join(commands)
    one_done(out, 1)


def test_the_screens_base_is_matched_with_a3s_stored_top10(tmp_path):
    code, _, commands = job(tmp_path, "base")
    line = f"scripts/comp_a6_guard.py a5-match {key('a3', 'base')} {A3TOP}"
    assert code == 0 and line in commands and f"pin {A3TOP}" in commands and commands.index(run_line(commands)) < commands.index(line)
    code, _, commands = job(tmp_path, "base", smoke=True, gpu=False, declared="", SMOKE_STAMP="s1")
    assert code == 0 and "a5-match" not in "\n".join(commands)                                # a smoke's five clips are not A3's report
    assert re.search(r"A3TOPSHA=(\S+)", JOB.read_text()).group(1) == "cc47eeb474db85573d92197e65d03165e1c41e94d5f67cce1de2638a4f67ec12"


def test_a_stop_on_the_match_marks_the_draw_and_refuses_later_attempts(tmp_path):
    code, out, commands = job(tmp_path, "base", words="3", STUB_EXIT_MATCH="comp_a6_guard.py a5-match", STUB_EXIT_CODE="4", **DRAW_ENV)
    assert code == 1 and "below 98%: this draw stops until it is declared under a new OUT" in out and f"scripts/comp_a6_guard.py started {OUT}/3-match-stopped" in commands
    one_done(out, 1)
    code, out, commands = job(tmp_path, "base", words="3", stored=[f"{OUT}/3-match-stopped"], ATTEMPT="2", **DRAW_ENV)
    assert code == 1 and "stopped on the match with its top10: refused until it is declared" in out and "round15_compose_eval.py" not in "\n".join(commands)
    code, _, _ = job(tmp_path, "base", words="4", stored=[f"{OUT}/3-match-stopped"], **DRAW_ENV)
    assert code == 0                                                                           # another draw goes on


def test_a_crash_of_the_match_guard_says_crash_and_marks_nothing(tmp_path):
    code, out, commands = job(tmp_path, "base", words="3", STUB_EXIT_MATCH="comp_a6_guard.py a5-match", STUB_EXIT_CODE="1", **DRAW_ENV)
    assert code == 1 and "the match guard crashed, exit 1: this says nothing about the match" in out and "differs" not in out and "match-stopped" not in "\n".join(commands)
    one_done(out, 1)


def test_threads_are_pinned(tmp_path):
    code, out, _ = job(tmp_path, "base", THREADS="4")
    assert code == 1 and "pinned at 8" in out
    assert re.search(r"^T=(\d+)$", JOB.read_text(), re.M).group(1) == str(a6.PINNED_THREADS)
    text = JOB.read_text().replace('[ "$Q" -gt 0 ]; then C=$(( Q / P ))', '[ 1 = 1 ]; then C=4')
    code, out, _ = job(tmp_path, "base", text=text)
    assert code == 1 and "this pod has 4 CPUs" in out


def test_the_jobs_comments_say_what_it_does():
    text = JOB.read_text()
    assert "a screen that decides nothing" not in text and "free instrument" not in text and "reader-amendment" not in text and "CACHE_SHA" not in text and "re-declaring under a new OUT named here" in text


def test_a_draw_needs_the_pinned_screen(tmp_path):
    stored = [*caches(), a5_top10(3)]
    env = {k: v for k, v in DRAW_ENV.items() if k != "SCREEN_SHA"}
    code, out, commands = job(tmp_path, "base", words="3", stored=stored, **env)
    assert code == 1 and "give SCREEN_SHA" in out and "round15_compose_eval.py" not in "\n".join(commands)
    code, out, _ = job(tmp_path, "base", words="3", stored=stored, STUB_PIN_FAIL=SCREEN_KEY, **DRAW_ENV)
    assert code == 1 and "differs from its pin" in out
    code, out, commands = job(tmp_path, "base", words="3", stored=stored, STUB_EXIT_MATCH="comp_a6_guard.py hold-gate", STUB_EXIT_CODE="1", **DRAW_ENV)
    assert code == 1 and "the hold gate cannot read the screen" in out and "round15_compose_eval.py" not in "\n".join(commands)
    one_done(out, 1)
    code, _, commands = job(tmp_path, "base")
    assert code == 0 and "hold-gate" not in "\n".join(commands)                              # the screen sets the gate and needs none


def test_the_screens_absence_stops_a_draw(tmp_path):
    text = JOB.read_text().replace('exists $SCREENKEY ||', 'false ||')
    code, out, _ = job(tmp_path, "base", words="3", stored=[*caches(), a5_top10(3)], text=text, **DRAW_ENV)
    assert code == 1 and "the screen's summary is not stored" in out


@pytest.mark.parametrize("phase,comparator", [(a6.HOLD, a6.BASE), (a6.BOTH, a6.LENGTH)])
def test_a_closed_gate_refuses_the_hold_arms_in_the_job(tmp_path, phase, comparator):
    stored = [*caches(), key(3, comparator), a5_top10(3)]
    code, out, commands = job(tmp_path, phase, words="3", stored=stored, **PINS, **DRAW_ENV, **CLOSED)
    assert code == 1 and f"{phase}: not run, screen failed" in out and "round15_compose_eval.py" not in "\n".join(commands)
    one_done(out, 1)
    code, _, commands = job(tmp_path, phase, words="3", stored=stored, **PINS, **DRAW_ENV)
    assert code == 0 and "round15_compose_eval.py run" in "\n".join(commands)


@pytest.mark.parametrize("phase", [a6.BASE, a6.LENGTH])
def test_a_closed_gate_still_runs_base_and_length(tmp_path, phase):
    code, _, commands = job(tmp_path, phase, words="3", stored=[*caches(), key(3, "base"), a5_top10(3)] if phase == a6.LENGTH else [*caches(), a5_top10(3)], **PINS, **DRAW_ENV, **CLOSED)
    assert code == 0 and "round15_compose_eval.py run" in "\n".join(commands)


def test_the_gate_phase_says_which(tmp_path):
    code, out, commands = job(tmp_path, "gate", words="3", **DRAW_ENV)
    assert code == 0 and "COMP-A6-GATE open" in out and "not run" not in out and f"scripts/comp_a6_guard.py hold-gate {SCREEN_KEY} abc1234" in commands
    code, out, _ = job(tmp_path, "gate", words="3", **DRAW_ENV, **CLOSED)
    assert code == 0 and "COMP-A6-GATE closed" in out and "COMP-A6-HOLD not run, screen failed" in out
    code, out, _ = job(tmp_path, "gate")
    assert code == 1 and "gate: WORDS must be 1 to 8" in out


def test_a_closed_draw_is_final_at_length(tmp_path):
    stored = [*caches(), key(3, "base"), key(3, "length"), a5_top10(3)]
    code, out, _ = job(tmp_path, "base", words="3", stored=stored, ATTEMPT="2", **DRAW_ENV, **CLOSED)
    assert code == 1 and "a finished draw is final" in out
    code, _, _ = job(tmp_path, "base", words="3", stored=stored, ATTEMPT="2", **DRAW_ENV)
    assert code == 0                                                                          # with the gate open, length is not the last arm


CLOSED_EVERY = [key(d, n) for d in a6.DRAWS for n in a6.GATE_CLOSED_ARMS] + [a5_top10(d) for d in a6.DRAWS]
CLOSED_SHAS = {f"{p}{d}_SHA": f"{p.lower()}{d}" for p in ("B", "L") for d in a6.DRAWS}


def test_the_summary_on_a_closed_gate_needs_base_and_length_alone(tmp_path):
    code, out, commands = job(tmp_path, "summary", words="draws", stored=CLOSED_EVERY, SCREEN_SHA="s", **CLOSED_SHAS, **A5T, **CLOSED)
    assert code == 0 and "COMP-A6-HOLD not run, screen failed" in out
    run = next(c for c in commands if "comp_a6_read.py" in c)
    assert f"--screen-key {SCREEN_KEY} --screen-sha256 s" in run and "--draw 3=" + ",".join(key(3, n) for n in a6.GATE_CLOSED_ARMS) + " " in run and "hold-x" not in run
    code, _, commands = job(tmp_path, "summary", words="draws", stored=CLOSED_EVERY, SCREEN_SHA="s", **CLOSED_SHAS, **A5T)
    assert code == 1 and "comp_a6_read.py" not in "\n".join(commands)                         # open: the hold arms are missing
    code, out, _ = job(tmp_path, "summary", words="draws", stored=CLOSED_EVERY, **CLOSED_SHAS, **A5T, **CLOSED)
    assert code == 1 and "give SCREEN_SHA" in out


def test_a_store_error_is_not_absence(tmp_path):
    line = next(x for x in JOB.read_text().splitlines() if x.startswith("exists() "))
    stubs = tmp_path / "bin"
    stubs.mkdir()
    script = tmp_path / "s.sh"
    script.write_text('die() { echo "ALERT $*"; exit 9; }\n' + line + '\nif exists k; then echo stored; else echo absent; fi\n')
    for code, want, rc in ((0, "stored", 0), (3, "absent", 0), (1, "ALERT store error on k", 9), (2, "ALERT store error on k", 9)):
        (stubs / "python").write_text(f"#!/bin/bash\nexit {code}\n")
        (stubs / "python").chmod(0o755)
        run = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=dict(PATH=f"{stubs}:/usr/bin:/bin"))
        assert run.stdout.strip() == want and run.returncode == rc


@pytest.mark.parametrize("attempt,want", [("1", "ATTEMPT=2"), ("4", "ATTEMPT=5"), ("5", "attempt 5 was the last")])
def test_the_attempt_suggestion_is_capped(tmp_path, attempt, want):
    code, out, _ = job(tmp_path, a6.LENGTH, stored=[key("a3", "base", int(attempt)), f"{OUT}/a3-attempt{attempt}-length.started"], ATTEMPT=attempt, BASE_SHA="b")
    assert code == 1 and want in out and "ATTEMPT=6" not in out


def test_the_smoke_read_runs_the_five_reports_and_prints_no_number(tmp_path):
    code, out, commands = job(tmp_path, "smoke-read", smoke=True, gpu=False, declared="", SMOKE_STAMP="s1", STUB_PRINT="BLEU 3.1", STUB_PRINT_MATCH="comp_a6_read.py")
    run = next(c for c in commands if "comp_a6_read.py" in c)
    keys = [f"{OUT}/smoke/s1/a3-attempt1-{a}.json" for a in a6.SCREEN_ARMS]
    assert code == 0 and "--smoke" in run and "--draw a3=" + ",".join(keys) in run and "A6 smoke read passed" in out and "3.1" not in out and "BLEU" not in out
    code, out, _ = job(tmp_path, "smoke-read", declared="abc1234")
    assert code == 1 and "smoke-read: SMOKE=1" in out
    code, out, _ = job(tmp_path, "smoke-read", smoke=True, gpu=False, declared="", SMOKE_STAMP="s1", STUB_FAIL_MATCH="comp_a6_read.py")
    assert code == 1 and "ALERT smoke-read" in out
    one_done(out, 1)


def test_a_smoke_reads_a3s_words_only(tmp_path):
    code, out, commands = job(tmp_path, "base", words="3", smoke=True, gpu=False, declared="", stored=caches(), SMOKE_STAMP="s", **DRAW_ENV)
    assert code == 1 and "a smoke reads WORDS=a3 only" in out and "round15_compose_eval.py" not in "\n".join(commands)
    one_done(out, 1)
    code, out, commands = job(tmp_path, "base", smoke=True, gpu=False, declared="", SMOKE_STAMP="s1", STUB_FAIL_MATCH="comp_a6_guard.py a5-done")
    assert code == 0 and "smoke=1" in out and "--clips 5 " in run_line(commands) and f"--output-key {OUT}/smoke/s1/a3-attempt1-base-shard0.json" in run_line(commands)


def test_hold_xr_is_refused_on_a_draw(tmp_path):
    code, out, commands = job(tmp_path, a6.HOLD_REST, words="3", stored=[*caches(), key(3, "base")], BASE_SHA="b", **DRAW_ENV)
    assert code == 1 and "hold-xr: WORDS=a3 only" in out and "round15_compose_eval.py" not in "\n".join(commands)
    one_done(out, 1)


@pytest.mark.parametrize("phase,other,var", [(a6.HOLD, a6.BASE, "BASE_SHA"), (a6.LENGTH, a6.BASE, "BASE_SHA"), (a6.BOTH, a6.LENGTH, "LENGTH_SHA"), (a6.HOLD_REST, a6.BASE, "BASE_SHA")])
def test_an_arm_needs_its_own_pinned_comparator_from_this_pod(tmp_path, phase, other, var):
    code, out, _ = job(tmp_path, phase, **PINS)
    assert code == 1 and "comes first" in out
    code, out, _ = job(tmp_path, phase, stored=[key("a3", other)])
    assert code == 1 and f"give the hash of its comparator {other}" in out
    code, out, commands = job(tmp_path, phase, stored=[key("a3", other)], STUB_PIN_FAIL=key("a3", other), **PINS)
    assert code == 1 and "differs from its pin" in out and "round15_compose_eval.py run" not in "\n".join(commands)
    code, out, commands = job(tmp_path, phase, stored=[key("a3", other)], STUB_FAIL_MATCH="comp_a6_guard.py same-pod", **PINS)
    assert code == 1 and f"{other} came from other code or another environment; rerun every arm under ATTEMPT=2" in out and "comp_a6_guard.py started" not in "\n".join(commands)
    one_done(out, 1)
    code, _, commands = job(tmp_path, phase, stored=[key("a3", other)], **PINS)
    guard_line = f"scripts/comp_a6_guard.py same-pod {key('a3', other)}"
    assert code == 0 and commands.index(guard_line) < commands.index(f"scripts/comp_a6_guard.py started {key('a3', phase)[:-5]}.started") < commands.index(run_line(commands))


def test_a_stored_arm_needs_no_comparison_and_prints_its_hash(tmp_path):
    code, out, commands = job(tmp_path, a6.LENGTH, stored=[key("a3", "base"), key("a3", "length")], BASE_SHA="b", STUB_FAIL_MATCH="comp_a6_guard.py same-pod")
    assert code == 0 and f"COMP-A6 a3-attempt1-length {key('a3', 'length')} stubsha" in out and "round15_compose_eval.py" not in "\n".join(commands)
    code, out, _ = job(tmp_path, "base", stored=[key("a3", "base")])
    assert code == 0 and f"COMP-A6 a3-attempt1-base {key('a3', 'base')} stubsha" in out


@pytest.mark.parametrize("attempt", ["0", "6", "x"])
def test_a_bad_attempt_is_refused(tmp_path, attempt):
    code, out, commands = job(tmp_path, "base", ATTEMPT=attempt)
    assert code == 1 and "ATTEMPT must be" in out and "round15_compose_eval.py" not in "\n".join(commands)
    one_done(out, 1)


def full(words, attempt=1):
    return [key(words, n, attempt) for n in (a6.SCREEN_ARMS if words == "a3" else a6.ARMS)]


@pytest.mark.parametrize("phase", a6.ARMS)
@pytest.mark.parametrize("attempt,final", [("2", 1), ("1", 2)])
def test_a_finished_draw_is_final(tmp_path, phase, attempt, final):
    code, out, commands = job(tmp_path, phase, stored=full("a3", final), ATTEMPT=attempt, **PINS)
    assert code == 1 and f"a3 attempt {final} has all its arms: a finished draw is final" in out and "round15_compose_eval.py" not in "\n".join(commands)
    one_done(out, 1)


def test_a_draw_short_of_its_last_arm_is_not_final(tmp_path):
    stored = [key("a3", n) for n in (a6.BASE, a6.HOLD, a6.LENGTH)]
    code, _, commands = job(tmp_path, "base", stored=stored, ATTEMPT="2")
    assert code == 0 and f"--output-key {key('a3', 'base', 2)[:-5]}-shard0.json" in run_line(commands)


def test_an_unfinished_attempt_reruns_every_arm(tmp_path):
    left = [key("a3", "base"), key("a3", "hold-x"), f"{OUT}/a3-attempt1-length.started"]
    code, _, commands = job(tmp_path, "base", stored=left, ATTEMPT="2")
    assert code == 0 and f"--output-key {key('a3', 'base', 2)[:-5]}-shard0.json" in run_line(commands)
    code, out, commands = job(tmp_path, a6.LENGTH, stored=left, BASE_SHA="b")
    assert code == 1 and "started and did not finish: rerun every arm under ATTEMPT=2" in out and "round15_compose_eval.py" not in "\n".join(commands)
    one_done(out, 1)
    code, out, commands = job(tmp_path, "base", STUB_FAIL_MATCH="comp_a6_guard.py started")
    assert code == 1 and "cannot mark its start" in out and "round15_compose_eval.py" not in "\n".join(commands)


EVERY = [k for d in a6.DRAWS for k in full(d)] + [a5_top10(d) for d in a6.DRAWS]
OPEN = dict(SHAS, **A5T, SCREEN_SHA="s")


def test_the_summary_of_the_draws_needs_all_eight_pinned(tmp_path):
    code, out, commands = job(tmp_path, "summary", words="draws", stored=EVERY, **OPEN)
    assert code == 0
    one_done(out, 0)
    run = next(c for c in commands if "comp_a6_read.py" in c)
    for d in a6.DRAWS:
        assert f"--draw {d}=" + ",".join(full(d)) in run and f"--a5-top10 {d}={a5_top10(d)}" in run
        for k in full(d):
            assert f"pin {k}" in commands
    assert "--declared-commit abc1234" in run and f"--output-key {OUT}/a6-summary-draws.json" in run and "hold-xr" not in run
    code, out, commands = job(tmp_path, "summary", words="draws", stored=[k for k in EVERY if k != key(8, a6.BOTH)], **OPEN)
    assert code == 1 and "incomplete: 8 has no attempt with all its arms" in out and "comp_a6_read.py" not in "\n".join(commands)
    env = dict(OPEN)
    env.pop("HXL3_SHA")
    code, out, _ = job(tmp_path, "summary", words="draws", stored=EVERY, **env)
    assert code == 1 and "give HXL3_SHA" in out
    code, out, _ = job(tmp_path, "summary", words="draws", stored=EVERY, STUB_PIN_FAIL=EVERY[4], **OPEN)
    assert code == 1 and "differs from its pin" in out
    code, out, commands = job(tmp_path, "summary", words="draws", stored=[f"{OUT}/a6-summary-draws.json"])
    assert code == 0 and "stored:" in out and "comp_a6_read.py" not in "\n".join(commands)


def test_the_instrument_check_finds_a5s_top10_by_hash(tmp_path):
    env = dict(OPEN)
    env.pop("A5T4_SHA")
    code, out, _ = job(tmp_path, "summary", words="draws", stored=EVERY, **env)
    assert code == 1 and "give A5T4_SHA" in out
    code, out, _ = job(tmp_path, "summary", words="draws", stored=[k for k in EVERY if k != a5_top10(4)], **OPEN)
    assert code == 1 and "no stored top10 of A5's draw 4" in out
    stored = [k for k in EVERY if k != a5_top10(5)] + [a5_top10(5, 1), a5_top10(5, 3)]
    code, _, commands = job(tmp_path, "summary", words="draws", stored=stored, text=hashed_job({a5_top10(5, 1): "old", a5_top10(5, 3): "new", **{a5_top10(d): "stubsha" for d in a6.DRAWS if d != 5}}), **dict(OPEN, A5T5_SHA="new"))
    assert code == 0 and f"--a5-top10 5={a5_top10(5, 3)}" in next(c for c in commands if "comp_a6_read.py" in c)


def test_the_summary_of_the_screen_reads_a3_alone_with_hold_xr(tmp_path):
    code, _, commands = job(tmp_path, "summary", stored=full("a3"), **SCREEN_SHAS)
    run = next(c for c in commands if "comp_a6_read.py" in c)
    assert code == 0 and "--draw a3=" + ",".join(full("a3")) in run and f"--output-key {OUT}/a6-summary-a3.json" in run and f"--a5-top10 a3={A3TOP}" in run and "--screen-key" not in run
    code, out, _ = job(tmp_path, "summary", stored=[k for k in full("a3") if "hold-xr" not in k], **SCREEN_SHAS)
    assert code == 1 and "incomplete: a3" in out


def test_the_summary_lists_every_attempt_and_takes_the_one_with_all_its_arms(tmp_path):
    stored = [k for k in EVERY if "/3-attempt" not in k] + full(3, 2) + [key(3, "base", 1), key(3, "hold-x", 1)]
    code, out, commands = job(tmp_path, "summary", words="draws", stored=stored, **OPEN)
    assert code == 0 and "COMP-A6-ATTEMPTS words=3 attempts=1:base+hold-x+;2:base+hold-x+length+hold-x-length+; used=2" in out and "COMP-A6-ATTEMPTS words=4 attempts=1:base+hold-x+length+hold-x-length+; used=1" in out
    run = next(c for c in commands if "comp_a6_read.py" in c)
    assert "--draw 3=" + ",".join(full(3, 2)) in run and "--attempts 3=1:base+hold-x+;2:base+hold-x+length+hold-x-length+;" in run


def test_the_summary_refuses_two_finished_attempts(tmp_path):
    code, out, commands = job(tmp_path, "summary", words="draws", stored=[*EVERY, *full(2, 4)], **OPEN)
    assert code == 1 and "2 has two finished attempts, 1 and 4" in out and "comp_a6_read.py" not in "\n".join(commands)
    one_done(out, 1)


@pytest.mark.parametrize("kwargs,message", [
    (dict(phase="base", words="9"), "WORDS must be"), (dict(phase="base", words="0"), "WORDS must be"), (dict(phase="base", words=None), "WORDS must be"), (dict(phase="base", words="draws"), "WORDS must be"),
    (dict(phase="summary", words="3"), "summary: WORDS"), (dict(phase="summary", words=None), "summary: WORDS"), (dict(phase="summary", smoke=True), "no smoke"), (dict(phase="other"), "unknown COMP_PHASE"),
    (dict(phase="base", SET="c20"), "D20 only"), (dict(phase="base", SET="val1000"), "D20 only"), (dict(phase="base", gpu=False), "needs a GPU"),
    (dict(phase="base", declared=""), "nothing is declared yet"), (dict(phase="base", SMOKE="2"), "SMOKE must be"),
])
def test_refusals(tmp_path, kwargs, message):
    phase = kwargs.pop("phase")
    code, out, commands = job(tmp_path, phase, **kwargs)
    assert code == 1 and message in out and "ALERT" in out and "round15_compose_eval.py" not in "\n".join(commands)
    one_done(out, 1)


def test_the_environment_and_the_declaration_come_first(tmp_path):
    _, _, commands = job(tmp_path, "base")
    assert scripts_run(commands)[:2] == ["scripts/comp_a6_guard.py env", "scripts/comp_a6_declared.py abc1234"]
    code, out, commands = job(tmp_path, "base", STUB_FAIL_MATCH="comp_a6_declared.py")
    assert code == 1 and "does not declare A6" in out and len(scripts_run(commands)) == 2
    code, out, commands = job(tmp_path, "base", STUB_FAIL_MATCH="comp_a6_guard.py env")
    assert code == 1 and "not the pinned ones" in out and scripts_run(commands) == ["scripts/comp_a6_guard.py env"]


def test_a_smokes_arm_still_checks_its_comparator(tmp_path):
    code, _, commands = job(tmp_path, a6.LENGTH, smoke=True, gpu=False, declared="", SMOKE_STAMP="s1", stored=[f"{OUT}/smoke/s1/a3-attempt1-base.json"], STUB_FAIL_MATCH="comp_a6_guard.py same-pod")
    assert code == 1


@pytest.mark.parametrize("match,message", [("round15_compose_eval.py run", "ALERT run a3-attempt1-base"), ("round15_compose_eval.py merge", "ALERT merge a3-attempt1-base")])
def test_a_failed_step_stops_with_done_one(tmp_path, match, message):
    code, out, _ = job(tmp_path, "base", STUB_FAIL_MATCH=match)
    assert code == 1 and message in out and "COMP-A6 a3-attempt1-base" not in out
    one_done(out, 1)


def test_no_hash_is_no_success(tmp_path):
    code, out, _ = job(tmp_path, "base", STUB_SHA_EMPTY="1")
    assert code == 1 and "no hash" in out and "COMP-A6 " not in out
    one_done(out, 1)


def test_a_failed_install_and_an_unexpected_end_say_done_one(tmp_path):
    base = dict(COMP_PHASE="base", WORDS="a3", COMP_A_DECLARED="abc1234", STUB_EXISTS="", STUB_GATE="1", STUB_FAIL_SHARD="", STUB_HANG="")
    code, out, commands = on_stubs(tmp_path, JOB.read_text().replace("uv pip install -q", "false uv pip install -q", 1), dict(base), smoke=False, gpu=True)
    assert code == 1 and "uv pip install failed" in out and not scripts_run(commands)
    one_done(out, 1)
    code, out, _ = on_stubs(tmp_path, JOB.read_text().replace("case $PHASE in\n  base)\n", "exit 7\ncase $PHASE in\n  base)\n", 1), dict(base), smoke=False, gpu=True)
    assert code == 1 and "stopped unexpectedly, exit 7" in out
    one_done(out, 1)


@pytest.mark.parametrize("phase", ["base", "length"])
def test_no_per_draw_number_reaches_the_log(tmp_path, phase):
    code, out, commands = job(tmp_path, phase, stored=[key("a3", "base")] if phase != "base" else [], BASE_SHA="b", STUB_PRINT="BLEU on 500 clips")
    assert code == 0 and any(" merge " in c for c in commands) and "BLEU" not in out


# ---- the chain, on a fake job -------------------------------------------------------------------------------------------------------------------------

FAKE_JOB = '''#!/bin/bash
[ "$COMP_PHASE" = gate ] && echo "COMP-A6-GATE ${FAKE_GATE-open}" && [ -n "${FAKE_GATE-open}" ] && echo "$COMP_PHASE words=$WORDS" >> "$FAKE_CALLS" && exit 0
[ "$COMP_PHASE" = gate ] && exit 0
echo "$COMP_PHASE words=$WORDS smoke=${SMOKE:-0} attempt=${ATTEMPT:-} base=${BASE_SHA:-} length=${LENGTH_SHA:-} declared=${COMP_A_DECLARED:-}" >> "$FAKE_CALLS"
[ "${FAKE_FAIL:-}" = "$COMP_PHASE" ] && [ -z "${SMOKE:-}" ] && exit 1
[ "${FAKE_FAIL_SMOKE:-}" = "$COMP_PHASE" ] && [ -n "${SMOKE:-}" ] && exit 1
if [ -z "${SMOKE:-}" ] && [ "${FAKE_NO_HASH:-}" = "$COMP_PHASE" ]; then echo "stored: nothing said"; exit 0; fi
echo "COMP-A6 $WORDS-attempt$ATTEMPT-$COMP_PHASE key/$COMP_PHASE ${SMOKE:+smoke}hash-$COMP_PHASE"
exit 0
'''


def chain(tmp_path, *args, **env):
    root = tmp_path / "code"
    (root / "runpod/fleet").mkdir(parents=True, exist_ok=True)
    (root / "runpod/fleet/comp_a6.sh").write_text(FAKE_JOB)
    text = CHAIN.read_text()
    assert "cd /root/work/Sign-Translator" in text and "LOG=/root/comp_a6_chain.log" in text
    text = text.replace("cd /root/work/Sign-Translator", f"cd {root}").replace("LOG=/root/comp_a6_chain.log", f"LOG={tmp_path}/chain.log")
    script = tmp_path / "chain.sh"
    script.write_text(text)
    calls = tmp_path / "calls"
    calls.write_text("")
    run = subprocess.run(["bash", str(script), *args], capture_output=True, text=True, timeout=30, env=dict(PATH="/usr/bin:/bin", FAKE_CALLS=str(calls), **env))
    return run.returncode, run.stdout, run.stderr, calls.read_text().splitlines()


def last_line(out):
    return out.strip().splitlines()[-1]


SMOKES = [f"{p} words=a3 smoke=1 attempt=1 base= length= declared=" for p in (a6.BASE, a6.HOLD, a6.LENGTH, a6.BOTH, a6.HOLD_REST, "smoke-read")]


def test_a_draws_chain_smokes_on_a3_runs_in_order_and_prints_the_hashes(tmp_path):
    code, out, _, calls = chain(tmp_path, "3", "abc1234")
    assert code == 0 and last_line(out) == "COMP-A6-CHAIN-DONE 0" and "ALERT" not in out
    assert "COMP-A6-HASHES words=3 attempt=1 B3_SHA=hash-base L3_SHA=hash-length HX3_SHA=hash-hold-x HXL3_SHA=hash-hold-x-length gate=open\n" in out
    assert calls == [*SMOKES, "gate words=3", "base words=3 smoke=0 attempt=1 base= length= declared=abc1234", "hold-x words=3 smoke=0 attempt=1 base=hash-base length= declared=abc1234",
                     "length words=3 smoke=0 attempt=1 base=hash-base length= declared=abc1234", "hold-x-length words=3 smoke=0 attempt=1 base= length=hash-length declared=abc1234"]


def test_a_closed_gate_runs_base_and_length_and_prints_it(tmp_path):
    code, out, _, calls = chain(tmp_path, "3", "abc1234", FAKE_GATE="closed")
    assert code == 0 and last_line(out) == "COMP-A6-CHAIN-DONE 0" and "COMP-A6-HOLD not run, screen failed" in out
    assert calls == [*SMOKES, "gate words=3", "base words=3 smoke=0 attempt=1 base= length= declared=abc1234", "length words=3 smoke=0 attempt=1 base=hash-base length= declared=abc1234"]
    assert "COMP-A6-HASHES words=3 attempt=1 B3_SHA=hash-base L3_SHA=hash-length gate=closed" in out and "HX3" not in out and "HXL3" not in out


@pytest.mark.parametrize("gate", ["", "maybe"])
def test_no_readable_gate_stops_the_chain(tmp_path, gate):
    code, out, _, calls = chain(tmp_path, "3", "abc1234", FAKE_GATE=gate)
    assert code == 1 and "ALERT chain: no hold gate for 3" in out and last_line(out) == "COMP-A6-CHAIN-DONE 1" and not any("smoke=0" in c for c in calls)


def test_the_screens_chain_adds_hold_xr_and_a_later_attempt_keeps_its_number(tmp_path):
    code, out, _, calls = chain(tmp_path, "a3", "abc1234", "2")
    assert code == 0 and "COMP-A6-HASHES words=a3 attempt=2 Ba3_SHA=hash-base La3_SHA=hash-length HXa3_SHA=hash-hold-x HXLa3_SHA=hash-hold-x-length HXRa3_SHA=hash-hold-xr gate=open" in out and "gate words" not in "\n".join(calls)
    assert "hold-xr words=a3 smoke=0 attempt=2 base=hash-base length= declared=abc1234" in calls and calls[-1] == "hold-x-length words=a3 smoke=0 attempt=2 base= length=hash-length declared=abc1234"


@pytest.mark.parametrize("phase", a6.ARMS)
def test_a_missing_hash_alerts_on_stderr_and_still_says_done(tmp_path, phase):
    code, out, err, calls = chain(tmp_path, "1", "abc1234", FAKE_NO_HASH=phase)
    assert code == 1 and last_line(out) == "COMP-A6-CHAIN-DONE 1" and f"ALERT chain: no hash for 1-attempt1-{phase}" in err and f"ALERT chain: no hash for {phase} of 1 attempt 1" in out
    assert "COMP-A6-HASHES" not in out and out.count("COMP-A6-CHAIN-DONE") == 1
    if phase == "base":
        assert not any("smoke=0" in c and not c.startswith("base") for c in calls)      # no comparator's hash, no arm, and the smoke's hash is never taken


@pytest.mark.parametrize("phase", a6.ARMS)
def test_a_failed_phase_stops_the_chain(tmp_path, phase):
    code, out, _, calls = chain(tmp_path, "1", "abc1234", FAKE_FAIL=phase)
    assert code == 1 and last_line(out) == "COMP-A6-CHAIN-DONE 1" and f"ALERT chain: COMP_PHASE={phase}" in out and out.count("COMP-A6-CHAIN-DONE") == 1 and "COMP-A6-HASHES" not in out and calls[-1].startswith(f"{phase} words=1 smoke=0")


def test_the_screens_chain_never_takes_its_smokes_hash(tmp_path):
    """The screen's smoke and its full run share the words' name and attempt 1: only the cleared log tells them apart."""
    code, out, err, calls = chain(tmp_path, "a3", "abc1234", FAKE_NO_HASH=a6.BASE)
    assert code == 1 and "ALERT chain: no hash for a3-attempt1-base" in err and last_line(out) == "COMP-A6-CHAIN-DONE 1" and not any("smoke=0" in c and not c.startswith("base") for c in calls)


def test_a_failed_smoke_stops_before_any_full_phase(tmp_path):
    code, out, _, calls = chain(tmp_path, "1", "abc1234", FAKE_FAIL_SMOKE=a6.HOLD)
    assert code == 1 and last_line(out) == "COMP-A6-CHAIN-DONE 1" and calls == SMOKES[:2]


@pytest.mark.parametrize("args", [(), ("1",)])
def test_missing_arguments_still_say_done(tmp_path, args):
    code, out, _, calls = chain(tmp_path, *args)
    assert code != 0 and calls == [] and "ALERT chain stopped unexpectedly" in out and last_line(out) == "COMP-A6-CHAIN-DONE 1"


def test_no_code_on_the_pod_still_says_done(tmp_path):
    script = tmp_path / "chain.sh"
    script.write_text(CHAIN.read_text().replace("cd /root/work/Sign-Translator", f"cd {tmp_path}/absent"))
    run = subprocess.run(["bash", str(script), "1", "abc1234"], capture_output=True, text=True, timeout=30, env=dict(PATH="/usr/bin:/bin"))
    assert run.returncode == 1 and "ALERT chain: no code" in run.stdout and last_line(run.stdout) == "COMP-A6-CHAIN-DONE 1"
