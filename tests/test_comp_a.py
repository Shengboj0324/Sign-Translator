"""Post-S2 workstream A (reports/post_s2_plan.md section 3): the arm scale, the token list and its arms, real-clip choice, the left-arm
analysis's arithmetic, the shared reader, and the job on stubs. Offline: synthetic stores and stubs only; no test starts a real job
and nothing is read from R2."""
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest
import torch

import scripts.round15_compose_eval as ce
import scripts.round20_word_cache as cw
import comp_a2_left_arm as a2
import comp_a3_tokens as a3
import comp_a_lib as lib
import comp_a_read as rd
import comp_a_clip_choice as a4
import comp_a_declared as dc
import comp_a_hold_stats as hs
from signtranslator.generation import composition_v2 as cv2
from signtranslator.generation import compositional as comp
from signtranslator.generation.motion_metrics import _matrices
from signtranslator.generation.motion_to_text import corpus_bleu
from tests.test_generation_signing_space import signer_state
from tests.test_round15_compose_eval import run_args
from tests.test_round15_stage2 import stage2_store, generator_store  # noqa: F401  (fixtures)
from tests.test_round15_compositional import stubbed  # noqa: F401
from tests.test_round20_eval import likelihood_stub, merge, put_gains, put_gloss_inputs, put_set  # noqa: F401
from tests.test_round20_phase_b import on_stubs, scripts_run


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "runpod/fleet/comp_a.sh"
TOKENS = ROOT / "reports/comp_a3_tokens.json"
PREDICTED = "core_crossfade@predicted"
O = "acquisition/2026-10-17-v1/youtube-asl-v1"
OUT = f"{O}/round20/comp-a"
A = f"{O}/round20/phase-a"
DONE = re.compile(r"^COMP-A-DONE (0|1) phase=\S+ arm=\S+ smoke=(0|1)$", re.M)


# ---- the arm scale --------------------------------------------------------------------------------------------------------

ARM_JOINTS = {"Left": ("LeftUpperArm", "LeftForearm"), "Right": ("RightUpperArm", "RightForearm")}


def angle_to_mean(state, joint):
    """Each valid frame's rotation angle from the channel's chordal mean."""
    from signtranslator.generation.composition_space import chordal_mean
    body = state.channels["body"]
    k = list(body.labels).index(joint)
    ok = body.valid[:, k]
    R = _matrices(body.values[:, k].astype(float))[ok]
    M = chordal_mean(R)
    trace = np.einsum("tii->t", np.einsum("ba,tbc->tac", M, R))
    return np.arccos(np.clip((trace - 1) / 2, -1., 1.))


def test_a_scale_of_one_leaves_the_state_itself():
    state = signer_state(frames=40, amplitude=.2)
    assert cv2.arm_scale_state(state) is state
    assert cv2.arm_scale_state(state, left=1., right=1.) is state


def test_the_scale_halves_each_arm_angle_and_touches_only_that_side():
    state = signer_state(frames=60, amplitude=.2)
    out = cv2.arm_scale_state(state, left=.5)
    for joint in ARM_JOINTS["Left"]:
        a, b = angle_to_mean(state, joint), angle_to_mean(out, joint)
        assert b.mean() == pytest.approx(.5 * a.mean(), rel=.08)            # about the state's own mean; small amplitudes
        assert b.max() < a.max()
    body, new = state.channels["body"], out.channels["body"]
    labels = list(body.labels)
    for k, label in enumerate(labels):
        if label not in ARM_JOINTS["Left"]:
            np.testing.assert_array_equal(body.values[:, k], new.values[:, k])
    np.testing.assert_array_equal(body.valid, new.valid)
    np.testing.assert_array_equal(state.timestamps, out.timestamps)
    for name in ("left_hand", "right_hand", "face"):
        assert out.channels[name] is state.channels[name]
    both = cv2.arm_scale_state(state, left=.7, right=.7)
    for joint in ARM_JOINTS["Left"] + ARM_JOINTS["Right"]:
        assert angle_to_mean(both, joint).mean() == pytest.approx(.7 * angle_to_mean(state, joint).mean(), rel=.08)
    right = cv2.arm_scale_state(state, right=.5)
    np.testing.assert_array_equal(right.channels["body"].values[:, labels.index("LeftForearm")],
                                  body.values[:, labels.index("LeftForearm")])
    assert not np.array_equal(right.channels["body"].values[:, labels.index("RightForearm")],
                              body.values[:, labels.index("RightForearm")])


def test_the_scale_keeps_rotations_valid_and_leaves_invalid_frames():
    state = signer_state(frames=30, amplitude=.2)
    body = state.channels["body"]
    k = list(body.labels).index("LeftForearm")
    valid = body.valid.copy()
    valid[10:15, k] = False
    from dataclasses import replace
    values = np.where(valid[..., None], body.values, 0.)
    masked = replace(state, channels=dict(state.channels, body=replace(body, values=values, valid=valid, observed=body.observed & valid,
                                                                       confidence=valid.astype(float))))
    out = cv2.arm_scale_state(masked, left=.5).channels["body"]
    assert not out.values[10:15, k].any() and np.array_equal(out.valid, valid)
    R = _matrices(out.values[valid[:, k], k].astype(float))
    np.testing.assert_allclose(np.einsum("tab,tcb->tac", R, R), np.tile(np.eye(3), (len(R), 1, 1)), atol=1e-5)
    with pytest.raises(ValueError, match="positive"):
        cv2.arm_scale_state(state, left=0.)


def arms_only_state(frames=96):
    """Everything at rest but both arms, swinging slowly: what the metric reads is the arms alone."""
    from dataclasses import replace
    from signtranslator.generation.composition_space import axis_angle, to_6d
    state = signer_state(frames=frames, amplitude=0.)
    body = state.channels["body"]
    values = body.values.astype(float)
    angle = .5 * np.sin(np.arange(frames) * 2 * np.pi / 48)
    rotation = to_6d(axis_angle(np.array([0., 0., 1.]), angle))
    for side in ("Left", "Right"):
        values[:, list(body.labels).index(f"{side}UpperArm")] = rotation
    return replace(state, channels=dict(state.channels, body=replace(body, values=values.astype(body.values.dtype))))


def test_a_scale_moves_the_wrist_metric_down_on_that_side_only():
    def sides(s):
        return {k: float(v[0][v[1]].mean()) for k, v in a2.side_steps(*a2.keypoints(s)).items()}
    state = arms_only_state()
    base, out = sides(state), sides(cv2.arm_scale_state(state, left=.5))
    assert base["left"] > 0 and .4 * base["left"] < out["left"] < .6 * base["left"]
    assert out["right"] == pytest.approx(base["right"], rel=1e-3)
    both = sides(cv2.arm_scale_state(state, left=.7, right=.7))
    assert .6 * base["left"] < both["left"] < .8 * base["left"] and .6 * base["right"] < both["right"] < .8 * base["right"]


@pytest.mark.parametrize("text", ["left=0.5", "right=1", "left=0,right=1", "left=1.1,right=1", "left=1,right=1", "left=.5,left=.5",
                                  "up=.5,right=1", "left=,right=1"])
def test_the_scale_option_refuses(text):
    with pytest.raises(ValueError, match="--arm-scale"):
        lib.parse_arm_scale(text)


def test_the_scale_option_parses():
    assert lib.parse_arm_scale("left=0.5,right=1") == dict(left=.5, right=1.)
    assert lib.parse_arm_scale("right=0.7,left=0.7") == dict(left=.7, right=.7)


# ---- guards ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key,sha", [
    (f"{O}/round20/c20.json", "24ac16f220139365af2bc7a62e7555ab88d13e9adb9b0a5f078c8b4107f72956"),
    (f"{O}/round20/c20.json", lib.D20_SHA),
    (lib.D20_KEY, "24ac16f220139365af2bc7a62e7555ab88d13e9adb9b0a5f078c8b4107f72956"),
    (None, None),
])
def test_only_d20_is_read(key, sha):
    with pytest.raises(SystemExit, match="D20 only"):
        lib.require_d20(key, sha)
    lib.require_d20(lib.D20_KEY, lib.D20_SHA)


def test_a_list_that_says_it_is_not_d20_stops():
    lib.check_d20_payload(dict(set="D20"))
    with pytest.raises(SystemExit, match="not D20"):
        lib.check_d20_payload(dict(set="C20"))


@pytest.mark.parametrize("record,message", [
    (dict(sign="a", corpus="asl-citizen-text-v1", partition="train"), "not a vocabulary source"),
    (dict(sign="a", corpus="stem-wiki-text-v1", partition="train"), "not a vocabulary source"),
    (dict(sign="a", corpus="how2sign-two-view-text-v1", partition="train"), "not a vocabulary source"),
    (dict(sign="a", corpus="semlex-text-v2", partition="val"), "train only"),
    (dict(sign="a", corpus="asllvd-text-v1", partition="test"), "train only"),
    (dict(sign="a", corpus=lib.SEGMENT_CORPUS, partition="test"), "train only"),
    (dict(sign="a", corpus="rit-text-r6", partition="train", signer_component="c00"), "signer 9"),
])
def test_a_source_clip_is_checked(record, message):
    with pytest.raises(SystemExit, match=message):
        lib.check_source_record(record)


@pytest.mark.parametrize("corpus", lib.ALLOWED_CORPORA)
def test_the_allowed_sources_pass(corpus):
    lib.check_source_record(dict(sign="a", corpus=corpus, partition="train", signer_component=None))
    assert "citizen" not in corpus and "stem" not in corpus


# ---- the token list -------------------------------------------------------------------------------------------------------

def row(rank, token, status="clips", entry=None, kind="isolated", seconds=.5, sources=()):
    return dict(rank=rank, token=token, status=status, entry=entry if entry is not None else (token if status == "clips" else None),
                source_kind=kind if status == "clips" else None, typical_seconds=seconds, sources=list(sources),
                train_count=100 - rank, flags={})


def small_list(n=30):
    rows = [row(1, "you"), row(2, "so", "no_source"), row(3, "it's", "alias_list", "it"), row(4, "we're", "alias_r10", "we"),
            row(5, "don't", "no_source"), row(6, "i'm", "alias_list", "i")]
    rows += [row(k, f"w{k}", "no_source") for k in range(7, 11)]
    rows += [row(11, "it"), row(12, "i", kind="segment")]
    rows += [row(k, f"w{k}", "no_source") for k in range(13, n + 1)]
    return dict(kind="comp-a3-tokens", tokens=rows)


def test_an_arm_takes_its_ranks_and_nothing_slides_up():
    arm = lib.arm_of(small_list(), 10)
    assert arm["token_map"] == {"you": "you", "we're": "we"}            # it's and i'm wait for ranks 11, 12
    assert set(arm["entries"]) == {"you"}
    assert arm["dropped"]["so"] == "no_source" and "it" in arm["dropped"]["it's"] and "i" in arm["dropped"]["i'm"]
    assert "it" not in arm["token_map"]                                  # rank 11 does not fill rank 2's place
    wide = lib.arm_of(small_list(), 30)
    assert wide["token_map"] == {"you": "you", "it's": "it", "we're": "we", "i'm": "i", "it": "it", "i": "i"}
    assert set(wide["entries"]) == {"you", "it", "i"}


@pytest.mark.parametrize("top", [5, 20, 100, 0])
def test_only_the_two_lengths_are_arms(top):
    with pytest.raises(SystemExit, match="vocabulary-extra-top"):
        lib.arm_of(small_list(), top)


def test_a_list_with_a_missing_rank_stops():
    broken = small_list()
    broken["tokens"] = [r for r in broken["tokens"] if r["rank"] != 3]
    with pytest.raises(SystemExit, match="does not hold ranks"):
        lib.arm_of(broken, 10)


def test_the_token_map_changes_only_tokens_the_selection_left_unmatched():
    vocab = dict(cut=1., go=1., onion=1., i=1., we=1., you=1.)
    text = "you cut it going in we're the onion i"
    base = comp.selection_details(text, vocab)
    # "you" is in this vocabulary, so it stays exact; "i" is a letter by the selection's rule, so it was unmatched
    details, words = lib.apply_token_map([base], {"it": "it", "we're": "we", "i": "i", "you": "other", "going": "other",
                                                  "the": "other", "onion": "other"})
    assert words[0] == ["you", "cut", "it", "go", "we", "onion", "i"]
    new = details[0]
    assert [r["kind"] for r in base] == ["exact", "exact", "unmatched", "fallback", "unmatched", "unmatched", "stop", "exact", "unmatched"]
    for old, r in zip(base, new):
        if old["kind"] != "unmatched" or old["tokens"][0] not in ("it", "we're", "i"):
            assert r is old
        else:
            assert r["kind"] == "exact" and r["extra"] is True
    assert [r["entry"] for r in base if r["kind"] == "unmatched"] == [None, None, None, None]    # the base rows are unchanged


def test_the_token_list_is_checked_against_its_hash_and_its_sources(tmp_path):
    path = tmp_path / "t.json"
    payload = small_list()
    path.write_text(json.dumps(payload))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert lib.load_token_list(path, sha)["kind"] == "comp-a3-tokens"
    with pytest.raises(SystemExit, match="SHA-256"):
        lib.load_token_list(path, "0" * 64)
    payload["tokens"][0]["sources"] = [dict(sign="a", corpus="asl-citizen-text-v1", partition="train")]
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="not a vocabulary source"):
        lib.load_token_list(path, hashlib.sha256(path.read_bytes()).hexdigest())
    path.write_text(json.dumps(dict(kind="other", tokens=[])))
    with pytest.raises(SystemExit, match="not a token list"):
        lib.load_token_list(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_extra_pools_serve_listed_clips_only():
    arm = dict(entries=dict(a=dict(source_kind="isolated", sources=[dict(sign="r1")]),
                            b=dict(source_kind="segment", sources=[dict(sign="u1#0", record_sign="u1", start=0, end=3, utterance_frames=9)])))
    rec = dict(sign="r1", corpus="semlex-text-v2", partition="train", continuous=dict(state_sha256="s"))
    utt = dict(sign="u1", partition="train")
    iso, seg = lib.extra_pools(arm, dict(r1=rec), dict(u1=utt))
    assert iso == dict(a=[rec]) and seg["b"][0]["record"] is utt and seg["b"][0]["entry"] == "b"
    with pytest.raises(SystemExit, match="not in the source mix"):
        lib.extra_pools(arm, {}, dict(u1=utt))
    with pytest.raises(SystemExit, match="not a train utterance"):
        lib.extra_pools(arm, dict(r1=rec), dict(u1=dict(sign="u1", partition="test")))
    with pytest.raises(SystemExit, match="train only"):
        lib.extra_pools(arm, dict(r1=dict(rec, partition="val")), dict(u1=utt))


def fake_h2s(text, partition="train", component="c02", corpus="how2sign-two-view-text-v1"):
    return dict(corpus=corpus, partition=partition, signer_component=component, text=text, sign=text)


def test_ranks_come_from_train_texts_never_val_and_ties_are_alphabetical():
    manifest = dict(records=[fake_h2s("you and you the zebra"), fake_h2s("beta alpha"),
                             fake_h2s("valonly valonly valonly", partition="val", component="c01"),
                             fake_h2s("testonly testonly", partition="test", component="c01"),
                             fake_h2s("isolatedword", corpus="semlex-text-v2")])
    assert a3.ranked_tokens(manifest, dict(cut=1.)) == [("you", 2), ("alpha", 1), ("and", 1), ("beta", 1), ("zebra", 1)]
    with pytest.raises(SystemExit, match="signer 9"):
        a3.ranked_tokens(dict(records=[fake_h2s("hello", component="c00")]), dict(cut=1.))


def test_a_segment_needs_a_train_utterance_and_a_timed_sign():
    utterances = [dict(collection="c", source="Cory_2013-6-25_sc01", frames=[0, 19], utterance_video="u1.mov",
                       signs=[dict(gloss="IX-2p", occurrence="IX-2p", sign_type="", frames=[2, 5]),
                              dict(gloss="DIFFERENT", occurrence="DIFFERENT", sign_type="Lexical Signs", frames=[6, 8]),
                              dict(gloss="IX-3p", occurrence="IX-3p", sign_type="", frames=[9, 12]),
                              dict(gloss="IX-2p", occurrence="IX-2p", sign_type="", frames=[40, 45])]),
                  dict(collection="c", source="4-Ben-Deaf-Speech", frames=[0, 19], utterance_video="u2.mov",
                       signs=[dict(gloss="IX-2p", occurrence="IX-2p", sign_type="", frames=[2, 5])])]
    gloss = dict(records=[dict(sign="c:u1", partition="train"), dict(sign="c:u2", partition="test")])
    pools, shared = a3.token_segments(utterances, gloss, ["you", "different", "it", "this"])
    assert [s["sign"] for s in pools["you"]] == ["c:u1#0"]               # Ben (test) never; the untimed one dropped
    assert pools["you"][0]["start"] == 2 and pools["you"][0]["end"] == 5 and pools["you"][0]["utterance_frames"] == 20
    assert [s["gloss"] for s in pools["different"]] == ["DIFFERENT"]
    assert shared["it"] == ["this"] and shared["this"] == ["it"]


def test_bens_utterances_stay_out_even_if_a_manifest_called_them_train():
    utterances = [dict(collection="c", source="4-Ben-Deaf-Speech", frames=[0, 19], utterance_video="u2.mov",
                       signs=[dict(gloss="IX-2p", occurrence="IX-2p", sign_type="", frames=[2, 5])])]
    pools, _ = a3.token_segments(utterances, dict(records=[dict(sign="c:u2", partition="train")]), ["you"])
    assert pools == {}


def test_a_short_span_is_widened_only_when_its_row_asks():
    from round20_lib import segment_clip
    utterance = signer_state(frames=48, amplitude=.2)
    native = np.arange(48) / 24.
    one = dict(start=20, end=20, utterance_frames=48)
    plain = segment_clip(one, utterance, native, margin=0.)
    assert plain["hi"] - plain["lo"] + 1 == 1
    wide = segment_clip(dict(one, min_seconds=.2), utterance, native, margin=0.)
    assert 4 <= wide["hi"] - wide["lo"] + 1 <= 6 and len(wide["state"].timestamps) == wide["hi"] - wide["lo"] + 1
    long = dict(start=10, end=30, utterance_frames=48, min_seconds=.2)
    assert len(segment_clip(long, utterance, native, margin=0.)["state"].timestamps) == 21            # longer than the floor: as annotated
    assert len(segment_clip(dict(one, min_seconds=.2), utterance, native, margin=.5)["state"].timestamps) == 25    # never narrower than asked


def test_the_tracked_rule():
    assert a3.well_tracked(dict(arm=a3.MIN_ARM, hand=a3.MIN_HAND))
    assert not a3.well_tracked(dict(arm=a3.MIN_ARM - .01, hand=1.))
    assert not a3.well_tracked(dict(arm=1., hand=a3.MIN_HAND - .01))


def test_the_draw_is_fixed_and_bounded():
    pool = [dict(sign=f"s{k:03d}") for k in range(100)]
    a, b = a3.draw(pool, "you"), a3.draw(list(reversed(pool)), "you")
    assert len(a) == a3.MAX_CHECKED and [x["sign"] for x in a] == sorted(x["sign"] for x in a)
    assert a3.draw(pool[:3], "you") == pool[:3] and a != a3.draw(pool, "it")
    assert len(b) == a3.MAX_CHECKED


def committed():
    return json.loads(TOKENS.read_text())


def test_the_committed_list_is_the_jobs_pin_and_holds_every_rank():
    text = JOB.read_text()
    assert hashlib.sha256(TOKENS.read_bytes()).hexdigest() == re.search(r"TOKENSSHA=([0-9a-f]{64})", text).group(1)
    tokens = committed()["tokens"]
    assert [r["rank"] for r in tokens] == list(range(1, 31))
    counts = [r["train_count"] for r in tokens]
    assert counts == sorted(counts, reverse=True)
    assert committed()["inputs"]["rank_mix"]["sha256"] == a3.EVAL_MIX_SHA


def test_the_committed_lists_sources_are_all_allowed_and_well_tracked():
    payload = committed()
    for r in payload["tokens"]:
        for s in r["sources"]:
            lib.check_source_record(s)
            assert a3.well_tracked(s["tracked"]), s
            if r["source_kind"] == "segment":
                assert s["min_seconds"] == a3.MIN_SEGMENT_SECONDS and s["seconds"] >= .15, s    # never one still frame
        assert (r["status"] == "clips") == bool(r["sources"])
    lib.load_token_list(TOKENS, hashlib.sha256(TOKENS.read_bytes()).hexdigest())


def test_the_committed_lists_flags():
    by = {r["token"]: r for r in committed()["tokens"]}
    for token in ("you", "it", "your", "this", "i", "they", "our", "them", "their"):
        assert by[token]["flags"]["pointing"] is True, token
    assert by["i"]["flags"]["single_letter"] is True and by["i"]["source_kind"] == "segment"
    assert all(s["corpus"] == lib.SEGMENT_CORPUS for s in by["i"]["sources"])      # never ASLLVD's letter I
    for token in ("just", "into", "would", "actually"):
        assert by[token]["status"] == "no_source" and by[token]["flags"]["no_well_tracked_source"] is True, token
    for token in ("so", "or"):
        assert by[token]["status"] == "clips" and by[token]["flags"]["fingerspelled_source"] is True, token
        assert {s["gloss_kind"] for s in by[token]["sources"]} <= {"lexicalised_fs", "fingerspelled"}
    assert by["don't"]["status"] == "alias_r10" and by["don't"]["entry"] == "not" and by["don't"]["flags"]["negation"] is True
    assert by["it's"]["status"] == "alias_list" and by["it's"]["entry"] == "it"
    assert by["that's"]["status"] == "alias_r10" and by["that's"]["entry"] == "that"
    assert by["you"]["source_kind"] == "segment" and len(by["you"]["sources"]) == a3.MAX_KEPT     # no longer one Sem-Lex clip
    assert by["just"]["options_not_used"] == ["RECENT-PAST"] and by["actually"]["options_not_used"] == ["REALLY"]
    assert all(r["sources"] == [] for r in by.values() if r.get("options_not_used"))             # a partial meaning is only listed
    for r in by.values():
        assert r["source_rule"], r["token"]                                                       # stated per token, before any read


def test_the_committed_list_signs_every_negation_in_every_arm():
    payload = committed()
    assert "don't" in payload["negation_tokens"] and "can't" in payload["negation_tokens"]
    coverage = payload["d20_coverage_text_only"]
    assert coverage["r10"]["sentences_with_unsigned_negation"] == 43
    for top in lib.TOPS:
        arm = lib.arm_of(payload, top)
        assert all(arm["token_map"][t] == "not" for t in payload["negation_tokens"])
        assert arm["token_map"]["can't"] == "not" and "can" not in arm["token_map"].values()
        assert coverage[f"top{top}"]["sentences_with_unsigned_negation"] == 0


def test_contractions_keep_their_meaning():
    vocab = {"can": 1., "not": 1., "that": 1., "will": 1.}
    for token in ("don't", "can't", "won't", "shouldn't", "ain't"):
        assert a3.contraction_entry(token, vocab, set()) == ("not", "alias_r10"), token      # never the first piece: can't is not can
    assert a3.contraction_entry("that's", vocab, set()) == ("that", "alias_r10")
    assert a3.contraction_entry("it's", vocab, {"it"}) == ("it", "alias_list")
    assert a3.contraction_entry("it's", vocab, set()) == (None, "no_source")
    assert a3.contraction_entry("don't", {"can": 1.}, set()) == (None, "no_source")


def test_fingerspelled_glosses_serve_tokens_and_name_signs_never_do():
    utterance = dict(collection="c", source="Cory_2013-6-25_sc01", frames=[0, 39], utterance_video="u1.mov",
                     signs=[dict(gloss="#SO", occurrence="#SO", sign_type="", frames=[1, 4]), dict(gloss="fs-OR", occurrence="fs-OR", sign_type="", frames=[5, 8]),
                            dict(gloss="ns-fs-SO", occurrence="ns-fs-SO", sign_type="", frames=[9, 12])])
    pools, _ = a3.token_segments([utterance], dict(records=[dict(sign="c:u1", partition="train")]), ["so", "or"])
    assert [s["gloss"] for s in pools["so"]] == ["#SO"] and pools["so"][0]["gloss_kind"] == "lexicalised_fs"
    assert [s["gloss"] for s in pools["or"]] == ["fs-OR"] and pools["or"][0]["gloss_kind"] == "fingerspelled"
    assert pools["so"][0]["min_seconds"] == a3.MIN_SEGMENT_SECONDS


@pytest.mark.parametrize("isolated,segments,kind", [(1, 5, "segment"), (5, 1, "isolated"), (2, 2, "isolated"), (0, 3, "segment"), (3, 0, "isolated"), (0, 0, None)])
def test_the_kind_with_more_well_tracked_clips_serves_the_token(isolated, segments, kind):
    assert a3.pick_source(isolated, segments) == kind


def test_the_committed_arms():
    payload = committed()
    ten, thirty = lib.arm_of(payload, 10), lib.arm_of(payload, 30)
    assert set(ten["entries"]) <= set(thirty["entries"])
    assert set(ten["token_map"]) <= set(thirty["token_map"])
    assert "just" in ten["dropped"] and "so" in ten["token_map"] and "or" in ten["token_map"]
    assert "just" in thirty["dropped"] and "into" in thirty["dropped"]


# ---- the left-arm hold ------------------------------------------------------------------------------------------------------------

STATS = json.loads((ROOT / "reports/comp_a_hold_stats.json").read_text())["how2sign_train"]
READY, CAP = STATS["ready_pose"], STATS["left_speed_p99"]


def plan_of(cores, gaps, lead=0, trail=0):
    frames = lead + sum(cores) + sum(gaps) + trail
    return dict(cores=list(cores), gaps=list(gaps), lead=lead, trail=trail, frames=frames, fit="fitted", overlong=False)


def body_of(state):
    return state.channels["body"]


def left_indices(state):
    labels = list(body_of(state).labels)
    return [labels.index(j) for j in cv2.HOLD_JOINTS]


def test_a_stretch_is_every_frame_between_anchors_that_holds_a_one_handed_core():
    labels = cv2.frame_labels(plan_of([3, 2, 2, 3], [1, 1, 2]))
    assert labels.tolist() == [0, 0, 0, -1, 1, 1, -1, 2, 2, -1, -1, 3, 3, 3]
    assert cv2.hold_stretches(labels, [False, True, True, False]) == [(3, 10)]     # both bridges and both cores
    assert cv2.hold_stretches(labels, [False, False, False, False]) == []          # bridges between anchors stay
    assert cv2.hold_stretches(labels, [False, True, False, True]) == [(3, 6), (9, 13)]
    assert cv2.hold_stretches(labels, [True, True, True, True]) == [(0, 13)]
    assert cv2.hold_stretches(labels, [None, True, None, None]) == [(3, 6)]         # an unannotated word is an anchor, kept as it is
    assert cv2.hold_stretches(cv2.frame_labels(plan_of([2, 2], [1], lead=2, trail=1)), [True, False]) == [(0, 4)]    # the lead goes with its edge


def test_the_hold_leaves_everything_else_bit_exact_and_ends_on_the_anchors():
    state = signer_state(frames=14, amplitude=.2)
    plan = plan_of([3, 2, 2, 3], [1, 1, 2])
    out, info = cv2.hold_left_arm(state, plan, [False, True, True, False], ready=READY, cap=99.)
    body, new = body_of(state), body_of(out)
    up, fore = left_indices(state)
    for k in range(body.values.shape[1]):
        if k not in (up, fore):
            np.testing.assert_array_equal(body.values[:, k], new.values[:, k])            # the right arm and the torso, to the bit
    for k in (up, fore):
        np.testing.assert_array_equal(body.values[:3, k], new.values[:3, k])              # the anchors, to the bit
        np.testing.assert_array_equal(body.values[11:, k], new.values[11:, k])
        assert not np.array_equal(body.values[3:11, k], new.values[3:11, k])
    for name in state.channels:
        if name != "body":
            assert out.channels[name] is state.channels[name]                             # the hands' own values
    np.testing.assert_array_equal(body.valid, new.valid)
    np.testing.assert_array_equal(state.timestamps, out.timestamps)
    assert info["stretches"] == 1 and info["frames"] == 8 and info["minimum_jerk"] == 1 and info["cap_bound"] == 0


def angle_between(a, b):
    trace = np.einsum("...ii->...", np.swapaxes(a, -1, -2) @ b)
    return np.arccos(np.clip((trace - 1) / 2, -1., 1.))


def test_the_slerp_runs_minimum_jerk_from_the_last_pose_to_the_next():
    state = signer_state(frames=14, amplitude=.3)
    plan = plan_of([3, 2, 2, 3], [1, 1, 2])
    out, _ = cv2.hold_left_arm(state, plan, [False, True, True, False], ready=READY, cap=99.)
    k = left_indices(state)[0]
    before, after = _matrices(body_of(state).values[2, k].astype(float)), _matrices(body_of(state).values[11, k].astype(float))
    moved = _matrices(body_of(out).values[3:11, k].astype(float))
    total = angle_between(before, after)
    t = np.arange(1, 9) / 9
    expected = cv2.minimum_jerk(t) * total
    np.testing.assert_allclose(angle_between(before, moved), expected, atol=1e-5)
    np.testing.assert_allclose(angle_between(moved, after), total - expected, atol=1e-5)
    assert cv2.minimum_jerk(np.array([0., .5, 1.])).tolist() == [0., .5, 1.]


def test_a_sentence_edge_holds_its_neighbour_and_no_anchor_holds_the_ready_pose():
    state = signer_state(frames=9, amplitude=.2)
    up = left_indices(state)[0]
    out, info = cv2.hold_left_arm(state, plan_of([3, 3], [3]), [True, False], ready=READY, cap=99.)
    held = body_of(out).values[:6, up]
    np.testing.assert_allclose(_matrices(held.astype(float)), np.repeat(_matrices(body_of(state).values[6, up].astype(float))[None], 6, 0), atol=1e-5)
    assert info["held"] == 1
    out, info = cv2.hold_left_arm(state, plan_of([3, 3], [3]), [False, True], ready=READY, cap=99.)
    np.testing.assert_allclose(_matrices(body_of(out).values[3:, up].astype(float)),
                               np.repeat(_matrices(body_of(state).values[2, up].astype(float))[None], 6, 0), atol=1e-5)
    out, info = cv2.hold_left_arm(state, plan_of([3, 3], [3]), [True, True], ready=READY, cap=99.)
    np.testing.assert_allclose(_matrices(body_of(out).values[:, up].astype(float)),
                               np.repeat(_matrices(np.asarray(READY["LeftUpperArm"], dtype=float))[None], 9, 0), atol=1e-4)
    assert info["ready_pose"] == 1 and info["held"] == 0


def test_a_word_with_no_annotation_is_never_held():
    state = signer_state(frames=9, amplitude=.2)
    out, info = cv2.hold_left_arm(state, plan_of([3, 3], [3]), [None, None], ready=READY, cap=99.)
    assert out is state and info["stretches"] == 0


def test_the_speed_cap_flattens_the_profile_then_counts_what_the_plan_cannot_give():
    from signtranslator.generation.composition_space import axis_angle, to_6d
    from dataclasses import replace
    state = signer_state(frames=30, amplitude=0.)
    body = body_of(state)
    values = body.values.astype(float)
    k = left_indices(state)[0]
    values[:, k] = to_6d(axis_angle(np.array([0., 0., 1.]), np.array([0.] * 10 + [0.] * 10 + [1.2] * 10)))
    state = replace(state, channels=dict(state.channels, body=replace(body, values=values.astype(body.values.dtype))))
    plan = plan_of([10, 8, 10], [1, 1])
    free, info = cv2.hold_left_arm(state, plan, [False, True, False], ready=READY, cap=99.)
    assert info["minimum_jerk"] == 1 and info["constant_speed"] == 0 and info["cap_bound"] == 0
    # a cap between the two profiles' peaks: the constant-speed slerp meets it
    peak = info["peak"]
    middle, info = cv2.hold_left_arm(state, plan, [False, True, False], ready=READY, cap=peak * .75)
    assert info["constant_speed"] == 1 and info["minimum_jerk"] == 0 and info["cap_bound"] == 0 and info["peak"] <= peak * .75
    assert not np.array_equal(body_of(free).values, body_of(middle).values)
    # below even that: counted, never a jump and never a longer plan
    tight, info = cv2.hold_left_arm(state, plan, [False, True, False], ready=READY, cap=peak * .1)
    assert info["cap_bound"] == 1 and info["constant_speed"] == 1 and len(tight.timestamps) == 30


def swinging_state(frames, moved):
    from signtranslator.generation.composition_space import axis_angle, to_6d
    from dataclasses import replace
    state = signer_state(frames=frames, amplitude=0.)
    body = body_of(state)
    values = body.values.astype(float)
    k = left_indices(state)[0]
    values[:, k] = to_6d(axis_angle(np.array([0., 0., 1.]), np.array([0.] * moved + [1.2] * (frames - moved))))
    return replace(state, channels=dict(state.channels, body=replace(body, values=values.astype(body.values.dtype))))


def test_the_cap_includes_the_steps_into_and_out_of_a_stretch():
    """A stretch of one frame has no step inside it: all its motion is on its way in and out."""
    state = swinging_state(11, 6)
    out, info = cv2.hold_left_arm(state, plan_of([5, 1, 5], [0, 0]), [False, True, False], ready=READY, cap=.01)
    assert info["stretches"] == 1 and info["peak"] > 0 and info["cap_bound"] == 1 and len(out.timestamps) == 11


def test_a_stretch_where_one_joint_moves_is_capped_as_a_moving_one():
    """The forearm has no anchor pose either side, so it takes the ready pose; the upper arm still moves, and the cap still applies."""
    from dataclasses import replace
    state = swinging_state(30, 20)
    body = body_of(state)
    fore = left_indices(state)[1]
    valid = body.valid.copy()
    valid[:10, fore] = False
    valid[20:, fore] = False
    masked = replace(state, channels=dict(state.channels, body=replace(body, values=np.where(valid[..., None], body.values, 0.), valid=valid, observed=body.observed & valid,
                                                                       confidence=valid.astype(float))))
    _, info = cv2.hold_left_arm(masked, plan_of([10, 8, 10], [1, 1]), [False, True, False], ready=READY, cap=1e-6)
    assert info["held"] == 0 and info["ready_pose"] == 0 and info["constant_speed"] == 1 and info["cap_bound"] == 1


def test_wrist_of_rotations_is_wrist_from_shoulder():
    from signtranslator.generation.composition_space import wrist_from_shoulder, wrist_of_rotations
    state = signer_state(frames=6, amplitude=.3)
    body = body_of(state)
    labels = list(body.labels)
    for side in ("Left", "Right"):
        upper, fore = (_matrices(body.values[:, labels.index(f"{side}{j}")].astype(float)) for j in ("UpperArm", "Forearm"))
        np.testing.assert_allclose(wrist_of_rotations(upper, fore, side), wrist_from_shoulder(state, side)[0])


def test_the_arm_scale_docstring_no_longer_says_the_hands_stay():
    assert "Timing, hands," not in cv2.__doc__ and "the hands themselves do move" in cv2.__doc__


def test_the_hold_statistics_are_how2sign_train_and_committed():
    assert JOB.read_text().count("HOLDSHA=" + hashlib.sha256((ROOT / "reports/comp_a_hold_stats.json").read_bytes()).hexdigest()) == 1
    loaded = lib.load_hold_stats(ROOT / "reports/comp_a_hold_stats.json", hashlib.sha256((ROOT / "reports/comp_a_hold_stats.json").read_bytes()).hexdigest())
    assert loaded["cap"] == CAP and set(loaded["ready"]) == set(cv2.HOLD_JOINTS)
    with pytest.raises(SystemExit, match="SHA-256"):
        lib.load_hold_stats(ROOT / "reports/comp_a_hold_stats.json", "0" * 64)


def test_the_hold_statistics_refuse_c00_and_read_train_only():
    def record(k, partition="train", component="c02"):
        return dict(sign=f"s{k}", corpus=hs.H2S, partition=partition, signer_component=component, continuous=dict(state_key="k"))
    picked = hs.train_clips(dict(records=[record(1), record(2), record(3, "val", "c01"), record(4, "test", "c01")]), n=9)
    assert sorted(r["sign"] for r in picked) == ["s1", "s2"]
    with pytest.raises(SystemExit, match="signer 9"):
        hs.train_clips(dict(records=[record(1, component="c00")]))
    assert hs.summarise([signer_state(frames=30, amplitude=.1), signer_state(1, frames=30, amplitude=.1)])["clips"] == 2


def test_motion_rows_per_side_and_contact():
    points = np.zeros((6, 8, 2))
    valid = np.ones((6, 8), bool)
    points[:, 5, 0] = .3                                                  # right wrist fixed, left wrist moving away
    points[:, 4, 0] = [0., 0., 0., 1., 1., 1.]
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert lib.contact_counts(points, valid, labels, [False, False]) == [3, 6]
    assert lib.contact_counts(points, valid, labels, [True, False]) == [0, 3]
    assert lib.contact_counts(points, valid, labels, [True, True]) == [0, 0]
    assert lib.contact_counts(points, valid, labels, [None, False]) == [0, 3]
    side = lib.side_read(points, valid)
    assert side["left_speed"] == pytest.approx(24 / 5) and side["right_speed"] == 0.
    valid[:, 5] = False
    assert lib.contact_counts(points, valid, labels, [False, False]) == [0, 0]
    assert lib.side_read(points, valid)["right_speed"] is None
    rows = [dict(comp_a_motion={"f": dict(left_speed=2., right_speed=3., left_spread=4., right_spread=5., contact=[1, 2]),
                                "real": dict(left_speed=1., right_speed=1., left_spread=2., right_spread=2.)}),
            dict(comp_a_motion={"f": dict(left_speed=4., right_speed=None, left_spread=4., right_spread=5., contact=[3, 6]),
                                "real": dict(left_speed=1., right_speed=1., left_spread=2., right_spread=2.)}), dict()]
    out = lib.motion_report(rows)["f"]
    assert out["left_speed_over_real"] == 3. and out["right_speed_over_real"] == 3. and out["contact_share"] == .5 and out["clips"] == 2


# ---- reads on hypotheses ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,loop", [
    ("a b c a b c a b c", True),
    ("a b c a b c", False),
    ("a b c d e f g", False),
    ("a a a a a", True),                 # the 3-gram a a a, three times
    ("a a a a", False),
    ("", False),
    ("a b", False),
])
def test_a_loop_is_a_three_gram_repeated_three_times(text, loop):
    assert lib.is_loop(text) is loop


def test_word_echo_counts_content_words_of_the_reference_found():
    refs = ["cut the onion and salt it", "you it", "heat it", "go now"]
    rows = lib.echo_rows(["the cut onion cut", "nothing here", "", "go go"], [["cut", "onion", "salt", "the"], ["you", "it"], ["heat"], ["go", "stir"]], refs)
    assert rows == [(2, 3), (0, 0), (0, 1), (1, 1)]                       # "stir" is signed but not in its reference: not counted
    assert lib.echo_share(rows) == pytest.approx(3 / 5)
    assert lib.echo_share([(0, 0)]) is None
    assert lib.echo_rows(["CUT, now."], [["Cut"]], ["cut it."]) == [(1, 1)]           # BLEU's own tokens on both sides


def test_a_new_entrys_echo_counts_stop_words_too():
    refs = ["you cut it", "it is here", "cut"]
    hyps = ["you and more", "nothing", "cut"]
    words = [["you", "cut", "it"], ["it"], ["cut"]]
    assert lib.new_entry_echo(hyps, words, refs, ["you", "it"]) == pytest.approx(1 / 3)
    assert lib.new_entry_echo(hyps, words, refs, ["zebra"]) is None


REFS = [f"reference number {k} has some long words in it today" for k in range(12)]
PROVENANCE = dict(code_digest="d" * 64, environment=dict(python="3", torch="2", transformers="4", tokenizers="0", gpu="L40S"))
SIDES_OK = dict(left_speed_over_real=1.5, right_speed_over_real=1.1, left_spread_over_real=1.6, right_spread_over_real=1.6, contact_share=.3, contact_frames=9, clips=12)


def arm_report(*, clips=12, smoke=False, report_only=True, set_sha=None, key=None, words=None, control=False, extra=None, settings=None, form="core_crossfade"):
    """A merged report's shape, on hand-made hypotheses."""
    refs = REFS[:clips]
    predicted = f"{form}@predicted"
    hyps = {"real": refs, "blank": ["some long words in it"] * clips, "blank@predicted": [" ".join(r.split()[:6]) for r in refs],
            predicted: [r if k % 2 else "reference number has some long" for k, r in enumerate(refs)],
            form: [r if k % 3 else "words" for k, r in enumerate(refs)]}
    if control:
        hyps[lib.REAL_SCALED] = [r if k % 4 else "some long" for k, r in enumerate(refs)]
    hyps.update(extra or {})
    forms = {n: dict(bleu=round(corpus_bleu(h, refs)["bleu"], 3), ci95=[0., 9.], hypotheses=list(h)) for n, h in hyps.items()}
    ll = {n: [-3. + i * .01 - (.5 if n.startswith("blank") else 0.) for i in range(clips)] for n in hyps}
    round20 = dict(sentences=dict(key=key or lib.D20_KEY, sha256=set_sha or lib.D20_SHA), comp_a_read=True, provenance=dict(PROVENANCE))
    if control:
        round20["arm_scale"] = dict(left=.5, right=1.)
    round20.update(settings or {})
    out = dict(clips=500 if not smoke and clips == 12 else clips, references=refs, signs=[f"s{k}" for k in range(clips)], forms=forms,
               settings=dict(variants=[form], seed=0, translator_key="t.pt", recognizer_run="sign-recognizer-v2", core_margin=0., timing="sign-fitted", total_fit="pad", round20=round20),
               per_clip=[dict(stage1_words=list(words or ["number", "reference", "zebra"])) for _ in refs],
               likelihood=dict(per_form=ll), motion_ratio_to_real={n: dict(wrist_speed=1.2, wrist_spread=1.5, hands_tracked=1., seconds_over_reference=1.2) for n in hyps},
               spotting=dict(scores={predicted: dict(recall=dict(mean=.3), order=dict(mean=.9), minus_mismatched_paired=None),
                                     **({lib.REAL_SCALED: dict(recall=dict(mean=.2), minus_real_paired=dict(difference=-.1))} if control else {})}),
               comp_a_motion={predicted: dict(SIDES_OK)}, shards=[0], round20={})
    if report_only:
        out["report_only"] = True
    return out


PREDICTED = "core_crossfade@predicted"


@pytest.mark.parametrize("kwargs,message", [
    (dict(set_sha="0" * 64), "D20 only"),
    (dict(key="round20/c20.json"), "D20 only"),
    (dict(report_only=False), "not report-only"),
])
def test_the_reader_refuses(kwargs, message):
    with pytest.raises(SystemExit, match=message):
        rd.read_report("a1", arm_report(**kwargs), arm_name="margin0", boot=20)


def test_the_reader_refuses_a_partial_read_and_a_big_smoke():
    partial = arm_report()
    partial["clips"] = 499
    with pytest.raises(SystemExit, match="not all of D20"):
        rd.read_report("a1", partial, arm_name="margin0", boot=20)
    with pytest.raises(SystemExit, match="at most 5"):
        rd.read_report("a1", arm_report(), arm_name="margin0", boot=20, smoke=True)
    small = arm_report(clips=5, smoke=True)
    assert rd.read_report("a1", small, arm_name="margin0", boot=20, smoke=True)["smoke"] is True
    with pytest.raises(SystemExit, match="no comparator"):
        rd.read_report("a1", small, arm_name="margin0", boot=20, smoke=True, comparators=[("x", small, PREDICTED, None, ())])


def test_a_scaled_arm_without_its_control_is_not_read():
    with pytest.raises(SystemExit, match="no real control"):
        rd.read_report("a1", arm_report(settings=dict(arm_scale=dict(left=.5, right=1.))), arm_name="left0.5", boot=20)


def test_one_primary_arm_per_phase():
    assert rd.PRIMARY == {"a1": "hold", "a3-oracle": "top10", "a3-gen": "top10", "a4": "n16"}
    assert rd.role_of("a1", "hold") == "primary" and rd.role_of("a1", "margin0") == "comparator"
    assert rd.role_of("a1", "left0.5") == "sensitivity" and rd.role_of("a1", "both0.7") == "sensitivity"
    assert rd.role_of("a3-oracle", "top30") == "sensitivity" and rd.role_of("a3-oracle", "base") == "comparator"
    assert rd.role_of("a4", "sample0") == "comparator" and rd.role_of("a4", "draw2") == "sensitivity"
    assert set(rd.DRAFT_RULES) == set(rd.KNOWN_AT_WRITING) == set(rd.TESTS)


def test_each_form_reports_everything():
    out = rd.read_report("a1", arm_report(extra={"blank@predicted": ["some long words in it today"] * 12}), arm_name="margin0", boot=30)
    read = out["reads"][PREDICTED]
    for key in ("bleu", "ci95", "gap_share", "likelihood_minus_blank", "likelihood_gap_share", "spotting", "motion_to_real", "length_ratio",
                "loop_share", "word_echo", "echo_lift_over_blank_relative_to_real", "per_side"):
        assert read[key] is not None, key
    assert set(read["motion_to_real"]) == {"wrist_speed", "wrist_spread", "hands_tracked", "seconds_over_reference"}
    assert out["report_only"] is True and out["declared"] is False and "adopted" not in json.dumps(out)
    assert set(out["reads"]) == set(arm_report()["forms"])


def test_gap_share_is_against_the_blank_at_the_forms_own_length():
    report = arm_report()
    read = rd.form_read(report, PREDICTED, boot=30)
    bleu = {n: corpus_bleu(report["forms"][n]["hypotheses"], REFS)["bleu"] for n in report["forms"]}
    assert read["blank"] == "blank@predicted"
    assert read["gap_share"]["share"] == pytest.approx((bleu[PREDICTED] - bleu["blank@predicted"]) / (bleu["real"] - bleu["blank@predicted"]), abs=1e-3)
    assert read["gap_share_true_length_blank"]["share"] == pytest.approx((bleu[PREDICTED] - bleu["blank"]) / (bleu["real"] - bleu["blank"]), abs=1e-3)
    assert read["gap_share"]["share"] != read["gap_share_true_length_blank"]["share"]
    main = rd.form_read(report, "core_crossfade", boot=30)
    assert main["blank"] == "blank" and "gap_share_true_length_blank" not in main
    assert main["gap_share"]["share"] == pytest.approx((bleu["core_crossfade"] - bleu["blank"]) / (bleu["real"] - bleu["blank"]), abs=1e-3)
    lo, hi = read["gap_share"]["ci95"]
    assert lo <= read["gap_share"]["share"] <= hi
    assert read["likelihood_minus_blank"]["difference"] == pytest.approx(.5)


def test_echo_lift_is_against_the_blank_at_the_same_length():
    words = [["number", "reference", "today"]] * 12
    report = arm_report(words=words[0], extra={"blank": ["reference"] * 12, "blank@predicted": ["today number"] * 12, PREDICTED: ["reference number"] * 12})
    read = rd.form_read(report, PREDICTED, boot=20)
    # real echoes 3 of 3, the same-length blank 2 of 3, the true-length blank 1 of 3, the form 2 of 3
    assert read["echo_lift_blank"] == "blank@predicted" and read["word_echo"] == pytest.approx(2 / 3, abs=1e-4)
    assert read["echo_lift_over_blank_relative_to_real"] == pytest.approx(0.)
    assert rd.form_read(report, "core_crossfade", boot=20)["echo_lift_blank"] == "blank"


def test_loop_and_echo_join_the_read():
    extra = {PREDICTED: ["x y z x y z x y z"] * 6 + REFS[6:]}
    read = rd.form_read(arm_report(extra=extra), PREDICTED, boot=30)
    assert read["loop_share"]["difference"] == pytest.approx(.5)
    words = [["reference", "number", "zebra"]] * 12
    expect = lib.echo_share(lib.echo_rows(extra[PREDICTED], words, REFS))
    assert read["word_echo"] == pytest.approx(expect, abs=1e-4)
    assert read["echo_content_words"] == 24                                # "zebra" is in no reference


def test_a_comparison_pairs_and_counts_identical_reads():
    arm, other = arm_report(), arm_report(extra={PREDICTED: REFS})
    row = rd.comparison(arm, PREDICTED, other, PREDICTED, boot=30)
    assert row["report_only"] is True and "both_improve" not in row
    assert row["bleu"]["difference"] < 0
    assert row["likelihood_minus_blank"]["blanks"] == ["blank@predicted", "blank@predicted"]
    assert row["instrument_check"] == dict(real_identical=12, blank_identical=12, clips=12, moved=False)
    bleu = {n: corpus_bleu(arm["forms"][n]["hypotheses"], REFS)["bleu"] for n in arm["forms"]}
    assert row["share_points"]["difference"] == pytest.approx(row["bleu"]["difference"] / (bleu["real"] - bleu["blank@predicted"]), abs=1e-3)
    moved = copy.deepcopy(other)
    moved["forms"]["real"]["hypotheses"][3] = "other"
    assert rd.comparison(arm, PREDICTED, moved, PREDICTED, boot=30)["instrument_check"]["moved"] is True
    unpaired = copy.deepcopy(other)
    unpaired["references"] = list(reversed(unpaired["references"]))
    with pytest.raises(SystemExit, match="not paired"):
        rd.comparison(arm, PREDICTED, unpaired, PREDICTED, boot=30)


def test_the_share_bootstrap_is_paired():
    """The same form against itself differs by nothing on every resample: an unpaired bootstrap would not."""
    arm = arm_report()
    row = rd.comparison(arm, PREDICTED, arm, PREDICTED, boot=60)
    assert row["share_points"]["difference"] == 0. and row["share_points"]["ci95"] == [0., 0.]


def test_the_echo_difference_is_arm_minus_comparator_per_sentence():
    words = ["number", "reference", "today"]
    arm = arm_report(words=words, extra={PREDICTED: ["reference number today"] * 6 + ["reference"] * 6})
    other = arm_report(words=words, extra={PREDICTED: ["reference"] * 12})
    row = rd.comparison(arm, PREDICTED, other, PREDICTED, boot=30)
    assert row["echo_difference"]["difference"] == pytest.approx(1 / 3, abs=1e-3)
    assert row["loop_difference"]["difference"] == pytest.approx(0.)


@pytest.mark.parametrize("field,value", [("core_margin", .1), ("timing", "asllrp"), ("total_fit", "scale")])
def test_pairing_refuses_an_undeclared_top_level_difference(field, value):
    from scripts.round20_compare import check_pairable
    a, b = arm_report(), arm_report()
    b["settings"][field] = value
    with pytest.raises(SystemExit, match=f"undeclared differences .*{field}"):
        check_pairable(a, b)
    check_pairable(a, b, (field,))
    check_pairable(a, b, "any")


@pytest.mark.parametrize("field", ["arm_scale", "hold_left", "word_cache", "word_cache_extra", "vocabulary_extra", "oracle_choice", "retime", "tempo"])
def test_pairing_refuses_an_undeclared_scale_or_cache(field):
    from scripts.round20_compare import check_pairable
    a, b = arm_report(), arm_report(settings={field: dict(x=1)})
    with pytest.raises(SystemExit, match=f"undeclared differences .*{field}"):
        check_pairable(a, b)
    check_pairable(a, b, (field,))


def test_a_declared_difference_that_is_not_there_stops():
    from scripts.round20_compare import check_pairable
    with pytest.raises(SystemExit, match="declared but absent .*core_margin"):
        check_pairable(arm_report(), arm_report(), ("core_margin",))


def row_for_rule(bleu=.1, ll=.01, lower=-.01):
    return dict(bleu=dict(difference=bleu, ci95=[bleu - .1, bleu + .1]), likelihood_minus_blank=dict(difference=dict(difference=ll)),
                spotting_guard=dict(available=True, difference=dict(ci95=[lower, .1])))


def motion_report(contact=.3, left=1.4, right=1.1, **ratio):
    return dict(motion_ratio_to_real={PREDICTED: dict(dict(wrist_speed=1.2, wrist_spread=1.6, hands_tracked=1.), **ratio)},
                comp_a_motion={PREDICTED: dict(left_speed_over_real=left, right_speed_over_real=right, contact_share=contact)})


A9 = dict(wrist=1.55, spread=2.04, left=2.4, right=1.5)


def rule(row=None, arm=None, first=None, test="a1", a9=A9, **kw):
    return rd.draft_rule(test, row=row or row_for_rule(), arm=arm or motion_report(), form=PREDICTED, first=first or motion_report(), first_form=PREDICTED, a9=a9, **kw)


def test_the_a1_draft_rule_passes_at_the_bars():
    out = rule()
    assert out["all_true"] is True and out["declared"] is False and "DRAFT" in out["note"]
    assert "adopted" not in json.dumps(out) and "already" not in out["text"]
    assert "a10" in out["known_when_written"] and "arithmetic" in out["known_when_written"]


@pytest.mark.parametrize("kwargs,check", [
    (dict(row=row_for_rule(bleu=0.)), "paired_bleu_above_0"),
    (dict(row=row_for_rule(bleu=-.1)), "paired_bleu_above_0"),
    (dict(row=row_for_rule(ll=0.)), "paired_likelihood_above_0"),
    (dict(row=row_for_rule(lower=-.02)), "spotting_lower_bound_above_minus_0.02"),
    (dict(arm=motion_report(wrist_speed=1.56)), "wrist_pooled_no_further_than_a9"),
    (dict(arm=motion_report(left=2.41)), "wrist_left_no_further_than_a9"),
    (dict(arm=motion_report(right=1.51)), "wrist_right_no_further_than_a9"),
    (dict(arm=motion_report(wrist_speed=.79)), "wrist_pooled_at_least_0.8"),
    (dict(arm=motion_report(left=.79)), "wrist_left_at_least_0.8"),
    (dict(arm=motion_report(right=.79)), "wrist_right_at_least_0.8"),
    (dict(arm=motion_report(wrist_spread=2.05)), "spread_no_further_than_a9"),
    (dict(arm=motion_report(contact=.26)), "contact_at_least_0.9_of_margin0"),
])
def test_each_a1_check_can_fail_alone(kwargs, check):
    out = rule(**kwargs)
    assert [k for k, v in out["checks"].items() if not v] == [check] and out["all_true"] is False


def test_the_spread_bar_can_be_met_now_and_sits_either_side_of_one():
    assert rd.no_further(1.6, 2.04) and rd.no_further(1., 2.04) and not rd.no_further(2.05, 2.04)
    assert rd.no_further(.5, .5) and not rd.no_further(.49, .5) and rd.no_further(1.5, .5) and not rd.no_further(1.51, .5)
    assert not rd.no_further(None, 2.) and not rd.no_further(1., None)


def test_a_missing_number_fails_a_check():
    missing = dict(bleu=dict(difference=.1, ci95=[0, .2]), spotting_guard=dict(available=False))
    out = rule(row=missing)
    assert out["checks"]["paired_likelihood_above_0"] is False and out["checks"]["spotting_lower_bound_above_minus_0.02"] is False
    empty = rule(arm=dict(motion_ratio_to_real={}, comp_a_motion={}))
    assert not any(empty["checks"][k] for k in empty["checks"] if k.startswith(("wrist", "spread", "contact")))


def test_the_other_tests_draft_rules():
    gen = rule(test="a3-gen", arm=motion_report(wrist_speed=1.5, hands_tracked=.9), first=motion_report(hands_tracked=1.))
    assert gen["all_true"] is True and "wrist_pooled_at_least_0.8" not in gen["checks"] and "wrist_no_further_than_a9" in gen["checks"]
    assert rule(test="a3-gen", arm=motion_report(wrist_speed=1.56))["checks"]["wrist_no_further_than_a9"] is False
    assert rule(test="a3-gen", arm=motion_report(hands_tracked=.89))["checks"]["hands_at_least_0.9_of_comparator"] is False
    oracle = rule(test="a3-oracle", gap_share=.25, a1_share=.19, new_echo=.3, base_echo=.2)
    assert oracle["all_true"] is True and set(oracle["checks"]) == {"paired_bleu_above_0", "paired_likelihood_above_0", "wrist_no_further_than_a9",
                                                                    "hands_at_least_0.9_of_comparator", "new_entries_echo_above_the_base", "gap_share_above_a1"}
    assert rule(test="a3-oracle", gap_share=.19, a1_share=.19, new_echo=.3, base_echo=.2)["checks"]["gap_share_above_a1"] is False
    assert rule(test="a3-oracle", gap_share=None, a1_share=.19)["checks"]["gap_share_above_a1"] is False
    assert rule(test="a3-oracle", gap_share=.3, a1_share=.1, new_echo=.2, base_echo=.2)["checks"]["new_entries_echo_above_the_base"] is False
    assert rule(test="a3-oracle", gap_share=.3, a1_share=.1, new_echo=None, base_echo=None)["checks"]["new_entries_echo_above_the_base"] is False
    assert rule(test="a3-oracle", gap_share=.3, a1_share=.1, new_echo=.3, base_echo=None)["checks"]["new_entries_echo_above_the_base"] is False
    assert "0.2 s" in rd.DRAFT_RULES["a3-oracle"]
    assert "not informative" in rd.DRAFT_RULES["a4"] and "no ceiling" in rd.DRAFT_RULES["a4"]


def test_the_declared_commit_is_only_a_label():
    out = rd.draft_rule("a3-oracle", row=row_for_rule(), declared_commit="abc123", arm={}, form=PREDICTED, first={}, first_form=PREDICTED)
    assert out["declared"] is True and out["declared_commit"] == "abc123" and "adopted" not in json.dumps(out)


SIDES = dict(forms={rd.A9_FORM: dict(pooled_wrist_ratio=1.55, pooled_spread_ratio=2.04, ratio_to_real=dict(left=2.4, right=1.5))})


def test_only_the_primary_arm_has_a_rule_and_it_reads_its_first_comparator():
    margin0 = arm_report(settings=None)
    held = arm_report(settings=dict(hold_left=dict(sha256="x")), extra={PREDICTED: REFS})
    worse = arm_report(extra={PREDICTED: ["words"] * 12})
    better = arm_report(extra={PREDICTED: REFS})
    # the first comparator decides, whatever the later ones say
    for first, expected in ((worse, True), (better, False)):
        second = arm_report(extra={PREDICTED: (better if first is worse else worse)["forms"][PREDICTED]["hypotheses"]})
        second["motion_ratio_to_real"][PREDICTED].update(wrist_speed=1.55, wrist_spread=2.04)
        comparators = [("margin0", first, PREDICTED, None, ("hold_left",)), ("a9_predicted", second, PREDICTED, None, ("hold_left",))]
        out = rd.read_report("a1", held, arm_name="hold", boot=20, comparators=comparators, a9_sides=SIDES)
        assert out["draft_rule"]["comparator"] == "margin0" and out["draft_rule"]["checks"]["paired_bleu_above_0"] is expected
    assert out["role"] == "primary" and out["provenance"] == PROVENANCE
    side = rd.read_report("a1", arm_report(control=True), arm_name="left0.5", boot=20, comparators=[("margin0", margin0, PREDICTED, None, ("arm_scale",))])
    assert side["role"] == "sensitivity" and "draft_rule" not in side
    with pytest.raises(SystemExit, match="a9_predicted comparator"):
        rd.read_report("a1", held, arm_name="hold", boot=20, comparators=[("margin0", margin0, PREDICTED, None, ("hold_left",))], a9_sides=SIDES)
    with pytest.raises(SystemExit, match="A2 file"):
        rd.read_report("a1", held, arm_name="hold", boot=20, comparators=[("margin0", margin0, PREDICTED, None, ("hold_left",))])
    with pytest.raises(SystemExit, match="has no comparator"):
        rd.read_report("a1", held, arm_name="hold", boot=20)


@pytest.mark.parametrize("test,name,settings,fits", [
    ("a1", "margin0", dict(), True), ("a1", "margin0", dict(hold_left=dict(sha256="x")), False), ("a1", "hold", dict(hold_left=dict(sha256="x")), True),
    ("a1", "hold", dict(), False), ("a1", "left0.5", dict(arm_scale=dict(left=.5, right=1.)), True), ("a1", "left0.5", dict(arm_scale=dict(left=.7, right=.7)), False),
    ("a1", "both0.7", dict(arm_scale=dict(left=.7, right=.7)), True), ("a1", "top10", dict(), False),
    ("a3-oracle", "base", dict(), True), ("a3-oracle", "base", dict(vocabulary_extra=dict(top=10)), False), ("a3-oracle", "top10", dict(vocabulary_extra=dict(top=10)), True),
    ("a3-oracle", "top10", dict(vocabulary_extra=dict(top=30)), False), ("a3-oracle", "top30", dict(vocabulary_extra=dict(top=30)), True),
    ("a3-oracle", "top10", dict(vocabulary_extra=dict(top=10), word_cache_extra=dict(key="k")), False),
    ("a3-gen", "top10", dict(vocabulary_extra=dict(top=10), word_cache_extra=dict(key="k")), True), ("a3-gen", "top10", dict(vocabulary_extra=dict(top=10)), False),
    ("a4", "n16", dict(oracle_choice=dict(pick="16")), True), ("a4", "n16", dict(oracle_choice=dict(pick="sample0")), False), ("a4", "sample0", dict(oracle_choice=dict(pick="sample0")), True),
    ("a4", "draw2", dict(oracle_choice=dict(pick="draw2")), True), ("a4", "draw2", dict(oracle_choice=dict(pick="draw3")), False), ("a4", "n8", dict(oracle_choice=dict(pick="8")), False),
])
def test_an_arm_name_must_be_what_its_settings_say(test, name, settings, fits):
    round20 = dict(comp_a_read=True, **settings) if name in ("margin0", "base") and not settings else dict(settings)
    if fits:
        rd.check_arm_name(test, name, dict(round20=round20))
    else:
        with pytest.raises(SystemExit, match="is not what this report"):
            rd.check_arm_name(test, name, dict(round20=round20))


def test_a_job_written_comparator_must_be_this_code_and_environment():
    arm = arm_report(settings=dict(hold_left=dict(sha256="x")))
    for field, value in (("code_digest", "e" * 64), ("environment", dict(PROVENANCE["environment"], gpu="other")), ("environment", dict(PROVENANCE["environment"], transformers="5"))):
        other = arm_report(settings=dict(provenance=dict(PROVENANCE, **{field: value})))
        for label in ("margin0", "base", "sample0"):
            with pytest.raises(SystemExit, match="other code or another environment"):
                rd.check_provenance(arm, other, label)
        rd.check_provenance(arm, other, "a1")                 # stored a10 and a1 stay exempt, under instrument_check
        rd.check_provenance(arm, other, "a9_predicted")
    for bare in (arm_report(settings=dict(provenance=None)),):
        with pytest.raises(SystemExit, match="other code"):
            rd.check_provenance(arm, bare, "margin0")
    rd.check_provenance(arm, arm_report(), "margin0")
    other = arm_report(settings=dict(provenance=dict(PROVENANCE, code_digest="e" * 64)), extra={PREDICTED: REFS})
    comparators = [("margin0", other, PREDICTED, None, ("hold_left",))]
    with pytest.raises(SystemExit, match="other code"):
        rd.read_report("a1", arm, arm_name="hold", boot=20, comparators=comparators, a9_sides=SIDES)


def test_the_read_states_the_release_band_apart_from_the_rule():
    report = arm_report()
    report["motion_ratio_to_real"][PREDICTED]["wrist_speed"] = 1.29
    out = rd.read_report("a1", report, arm_name="margin0", boot=20)
    assert out["release_band"] == dict(pooled_wrist=1.29, within_0_8_to_1_25=False)
    assert rd.release_band(dict(motion_ratio_to_real={PREDICTED: dict(wrist_speed=1.25)}), PREDICTED)["within_0_8_to_1_25"] is True
    assert rd.release_band(dict(motion_ratio_to_real={PREDICTED: dict(wrist_speed=.79)}), PREDICTED)["within_0_8_to_1_25"] is False
    assert not any("0_8_to_1_25" in k or "1.25" in k for k in rule()["checks"])


def test_the_rule_text_is_the_reviewers():
    a1 = "wrist per side and pooled no further from 1.0x real than a9@predicted's and at least 0.8x real; spread no further from 1.0x real than a9@predicted's; two-handed contact at least 0.9x margin0's. These motion checks are a no-regression floor and not the release bar. The read states separately whether pooled wrist lies within 0.8-1.25x; hold's does not."
    known = "Known from stored motion before any read: hold's left 1.37x, right 1.24x, pooled 1.29x, spread 1.71x, left height -0.74, contact unchanged by construction. The motion checks therefore pass and are not evidence; the 1.25x bar was replaced knowing 1.29x. A pass means hold reads no worse than margin0 with less motion. It meets no release bar."
    oracle = "wrist no further from 1.0x real than a9@predicted's (known before any read, first 300 clips: base 1.32x, top10 1.54x, top30 1.64x); hands at least 0.9x the base's; the new entries' echo above the base's echo of the same entries on the same sentences; gap share against blank@predicted above a1's own, which favours predicted arms by about 3 points (a9@predicted 23.2 against 19.9). BLEU can rise with speed alone, so the echo check is what credits coverage. A failure does not close coverage: every new top-10 word comes from ASLLRP continuous segments, a route that read lower as a whole (a2 1.879 against a1 2.219), and source and coverage are not separated here. Fingerspelled single-clip tokens stay out of any primary arm or claim."
    gen = "wrist no further from 1.0x real than a9@predicted's; hands at least 0.9x that arm's."
    four = "Primary arm: n16. Read in the decomposition: n16's entry-and-sentence bootstrap lower bound against sample0 above 0, paired likelihood above 0, and n16's BLEU above each of sample0 and draw1 to draw4. Repetition is the mean of those five draws minus a1."
    assert a1 in rd.DRAFT_RULES["a1"] and known in rd.KNOWN_AT_WRITING["a1"] and oracle in rd.DRAFT_RULES["a3-oracle"] and gen in rd.DRAFT_RULES["a3-gen"]
    assert rd.DRAFT_RULES["a4"].startswith(four)
    assert not any("1.10x" in text for text in rd.DRAFT_RULES.values())


def oracle_arm(words=("reference", "number"), new=("reference",), name_settings=None, **kw):
    report = arm_report(form="oracle_stage2", words=list(words), settings=name_settings or dict(vocabulary_extra=dict(top=10)), **kw)
    report["round20"] = dict(vocabulary_extra=dict(new_entries=list(new), fingerspelled_single_clips=[]))
    report["motion_ratio_to_real"]["oracle_stage2@predicted"].update(wrist_speed=1.5, hands_tracked=1.)
    return report


def read_oracle(arm, base, a1=None):
    a1 = a1 or arm_report(form="oracle_stage2", settings=dict(comp_a_read=None), extra={"oracle_stage2": ["some long"] * 12})
    return rd.read_report("a3-oracle", arm, arm_name="top10", boot=20, a9_sides=SIDES, comparators=[
        ("base", base, "oracle_stage2@predicted", None, ("vocabulary_extra",)), ("a1", a1, "oracle_stage2", None, ("vocabulary_extra",))])


def test_the_new_entries_echo_is_read_on_the_same_entries_and_sentences():
    predicted = "oracle_stage2@predicted"
    signed = oracle_arm(extra={predicted: REFS})
    # the base never signed the entry: its word list has no such entry, but its echo is still read for the arm's entries on the arm's sentences
    silent = arm_report(form="oracle_stage2", words=["zebra"], extra={predicted: ["some long words in it today"] * 12})
    out = read_oracle(signed, silent)
    assert out["draft_rule"]["checks"]["new_entries_echo_above_the_base"] is True
    # the base already says the word: the coverage earned nothing
    echoing = arm_report(form="oracle_stage2", words=["zebra"], extra={predicted: REFS})
    assert read_oracle(signed, echoing)["draft_rule"]["checks"]["new_entries_echo_above_the_base"] is False
    # the arm's wrist is held to a9's, not to the base's
    assert out["draft_rule"]["checks"]["wrist_no_further_than_a9"] is True
    fast = oracle_arm(extra={predicted: REFS})
    fast["motion_ratio_to_real"][predicted]["wrist_speed"] = 1.6
    assert read_oracle(fast, silent)["draft_rule"]["checks"]["wrist_no_further_than_a9"] is False


def test_a_primary_arm_refuses_a_fingerspelled_single_clip_token():
    arm = oracle_arm()
    arm["round20"]["vocabulary_extra"]["fingerspelled_single_clips"] = ["as"]
    with pytest.raises(SystemExit, match="stay out of a primary arm"):
        read_oracle(arm, arm_report(form="oracle_stage2"))
    payload = committed()
    assert lib.fingerspelled_single_clips(lib.arm_of(payload, 10)) == []
    assert lib.fingerspelled_single_clips(lib.arm_of(payload, 30)) == ["as", "at", "way"]


def test_a9_per_side_must_be_the_reports_own():
    a9 = arm_report()
    a9["motion_ratio_to_real"][PREDICTED].update(wrist_speed=1.55, wrist_spread=2.04)
    sides = dict(forms={rd.A9_FORM: dict(pooled_wrist_ratio=1.555, ratio_to_real=dict(left=2.4, right=1.5))})
    assert rd.a9_reference(sides, a9, PREDICTED) == dict(wrist=1.55, spread=2.04, left=2.4, right=1.5)
    assert rd.a9_reference(dict(forms={rd.A9_FORM: dict(sides["forms"][rd.A9_FORM], pooled_spread_ratio=2.)})) == dict(wrist=1.555, spread=2., left=2.4, right=1.5)
    sides["forms"][rd.A9_FORM]["pooled_wrist_ratio"] = 1.6
    with pytest.raises(SystemExit, match="not its report's"):
        rd.a9_reference(sides, a9, PREDICTED)


def test_the_gate():
    read = dict(kind="comp-a-read", test="a3-oracle", role="primary", smoke=False, draft_rule=dict(all_true=True))
    assert rd.oracle_gate(read) is True
    for broken in (dict(read, smoke=True), dict(read, test="a1"), dict(read, kind="other"), dict(read, role="sensitivity"),
                   dict(read, draft_rule=dict(all_true=False)), {k: v for k, v in read.items() if k != "draft_rule"}):
        assert rd.oracle_gate(broken) is False


def test_the_control_read_is_in_points_of_reals_gap_with_likelihood_and_spotting():
    report = arm_report(control=True)
    report["motion_ratio_to_real"][lib.REAL_SCALED]["wrist_speed"] = .75
    report["motion_ratio_to_real"][PREDICTED]["wrist_speed"] = 1.2
    out = rd.control_read(report, PREDICTED, boot=30)
    bleu = {n: corpus_bleu(report["forms"][n]["hypotheses"], REFS)["bleu"] for n in ("real", lib.REAL_SCALED, "blank")}
    assert out["loss_share_of_gap"] == pytest.approx((bleu["real"] - bleu[lib.REAL_SCALED]) / (bleu["real"] - bleu["blank"]), abs=1e-3)
    assert out["scales"] == dict(left=.5, right=1.) and out["likelihood_paired"]["difference"] == pytest.approx(0.)
    assert out["spotting_paired"] == dict(difference=-.1) and out["composition_over_scaled_real"] == 1.6 and out["real_below_band"] is True
    assert rd.control_read(arm_report(), PREDICTED, boot=30) is None
    full = rd.read_report("a1", arm_report(control=True), arm_name="left0.5", boot=20, comparators=[("margin0", arm_report(), PREDICTED, None, ("arm_scale",))])
    assert full["control"]["scales"] == dict(left=.5, right=1.) and full["role"] == "sensitivity"


# ---- the clip-choice read's decomposition --------------------------------------------------------------------------------------------

def test_the_entry_bootstrap_moves_sentences_that_share_a_clip_together():
    """One entry serves the half that a does better, another the half that b does better. Resampling entries too swings the difference far more
    than resampling sentences alone, which treats them as twenty separate chances."""
    from signtranslator.eval.s0_metrics import bleu_statistics
    refs = [f"alpha beta gamma delta epsilon {k}" for k in range(40)]
    hyps_a = [r if k < 20 else "alpha" for k, r in enumerate(refs)]
    hyps_b = ["alpha" if k < 20 else r for k, r in enumerate(refs)]
    a, b = bleu_statistics(hyps_a, refs), bleu_statistics(hyps_b, refs)
    shared = rd.clustered_difference(a, b, [["one"] if k < 20 else ["two"] for k in range(40)], boot=300)
    apart = rd.clustered_difference(a, b, [[f"e{k}"] for k in range(40)], boot=300)
    assert shared["bootstrap"] == "sentences and entries" and shared["entries"] == 2 and apart["entries"] == 40
    assert shared["difference"] == apart["difference"] == pytest.approx(corpus_bleu(hyps_a, refs)["bleu"] - corpus_bleu(hyps_b, refs)["bleu"], abs=1e-3)
    assert shared["ci95"][1] - shared["ci95"][0] > 1.3 * (apart["ci95"][1] - apart["ci95"][0])
    same = rd.clustered_difference(a, a, [["one"]] * 40, boot=50)
    assert same["difference"] == 0. and same["ci95"] == [0., 0.]


def oracle_report(hyps, pick=None, a1=False):
    report = arm_report(form="oracle_stage2", extra={"oracle_stage2": hyps}, settings=dict(oracle_choice=dict(pick=pick)) if pick else None, report_only=not a1)
    return report


def test_the_decomposition_separates_choice_from_repetition_and_a_non_rise_is_not_informative():
    chosen = oracle_report(REFS, "16")
    sample0 = oracle_report(["reference number"] * 12, "sample0")
    a1 = oracle_report(["words"] * 12, a1=True)
    draws = [oracle_report(["reference number 1 has some"] * 12, f"draw{k}") for k in range(1, 5)]
    up, lower = copy.deepcopy(chosen), copy.deepcopy(chosen)                # BLEU up with likelihood up is a rise; BLEU up with likelihood down is not
    up["likelihood"]["per_form"]["oracle_stage2"] = [x + 1. for x in up["likelihood"]["per_form"]["oracle_stage2"]]
    lower["likelihood"]["per_form"]["oracle_stage2"] = [x - 1. for x in lower["likelihood"]["per_form"]["oracle_stage2"]]
    out = rd.decompose(up, sample0, draws, a1, boot=40)
    bleu = lambda hyps: corpus_bleu(hyps, REFS)["bleu"]                           # noqa: E731
    assert out["reading"].startswith("choice rises") and out["draft_rule"]["all_true"] is True and out["draft_rule"]["declared"] is False
    assert out["choice_chosen_minus_sample0"]["clustered_bleu"]["difference"] > 0
    five = [["reference number"] * 12] + [["reference number 1 has some"] * 12] * 4
    assert out["repetition_mean_of_five_draws_minus_a1"]["clustered_bleu"]["difference"] == pytest.approx(sum(bleu(h) for h in five) / 5 - bleu(["words"] * 12), abs=1e-3)
    assert len(out["draw_bleu"]) == 5 and out["report_only"] is True and out["provenance"]["code_digest"]
    assert rd.decompose(lower, sample0, draws, a1, boot=40)["reading"].startswith("not informative")
    flat = rd.decompose(oracle_report(["reference number"] * 12, "16"), sample0, draws, a1, boot=40)
    assert flat["reading"].startswith("not informative") and "no room" in flat["reading"]
    assert rd.decompose(up, sample0, draws, a1, boot=40, declared_commit="abc1234")["draft_rule"]["declared_commit"] == "abc1234"
    # above sample0 but not above one draw: not a rise, however good the choice looks
    best_draw = list(draws)
    best_draw[2] = oracle_report(REFS, "draw3")
    checks = rd.decompose(up, sample0, best_draw, a1, boot=40)
    assert checks["draft_rule"]["checks"]["n16_above_draw3"] is False and checks["reading"].startswith("not informative")
    with pytest.raises(SystemExit, match="not paired"):
        rd.decompose(chosen, sample0, draws, oracle_report(["words"] * 12, "16", a1=True) | dict(references=list(reversed(REFS))), boot=20)
    with pytest.raises(SystemExit, match="draw1 to draw4"):
        rd.decompose(chosen, sample0, draws[:3], a1, boot=20)
    with pytest.raises(SystemExit, match="is not what this report"):
        rd.decompose(chosen, sample0, [draws[1], draws[0], *draws[2:]], a1, boot=20)
    bent = oracle_report(["reference number"] * 12, "sample0")
    bent["settings"]["round20"]["provenance"] = dict(PROVENANCE, code_digest="e" * 64)
    with pytest.raises(SystemExit, match="other code"):
        rd.decompose(chosen, bent, draws, a1, boot=20)


def test_the_entry_bootstrap_takes_a_mean_of_forms():
    from signtranslator.eval.s0_metrics import bleu_statistics
    refs = [f"alpha beta gamma delta epsilon {k}" for k in range(10)]
    x, y, z = (bleu_statistics(h, refs) for h in (refs, ["alpha beta gamma delta"] * 10, ["alpha"] * 10))
    out = rd.clustered_difference([x, y], z, [["a"]] * 10, boot=20)
    assert out["difference"] == pytest.approx((corpus_bleu(refs, refs)["bleu"] + corpus_bleu(["alpha beta gamma delta"] * 10, refs)["bleu"]) / 2 - corpus_bleu(["alpha"] * 10, refs)["bleu"], abs=1e-3)


# ---- A2's arithmetic --------------------------------------------------------------------------------------------------------

def test_frame_labels_follow_the_plan():
    plan = dict(lead=2, cores=[3, 0, 2], gaps=[1, 0], trail=1)
    assert a2.frame_labels(plan).tolist() == [-2, -2, 0, 0, 0, -1, 2, 2, -2]


def test_steps_fall_into_regions():
    labels = np.array([-2, 0, 0, -1, -1, 1, 1, -2])
    assert a2.step_regions(labels, [True, False]) == ["lead_trail", "core_one_handed", "bridge", "bridge", "bridge", "core_two_handed", "lead_trail"]
    assert a2.step_regions(np.array([0, 0, 1, 1]), [True, True]) == ["core_one_handed", "bridge", "core_one_handed"]


def test_annotated_regions_use_the_joins_own_set():
    labels = np.array([-2, 0, 0, -1, 1, 1, -1, -1, 2, 2, -2])
    assert a2.annotated_step_regions(labels, [True, True, False]) == [
        "lead_trail", "core_annotated_one", "bridge_one_to_one", "bridge_one_to_one", "core_annotated_one", "bridge_one_to_two",
        "bridge_one_to_two", "bridge_one_to_two", "core_annotated_two", "lead_trail"]
    assert a2.annotated_step_regions(np.array([0, 0, -1, 1]), [None, False]) == ["core_unannotated", "bridge_other", "bridge_other"]
    assert a2.annotated_step_regions(np.array([0, 0, 1, 1]), [True, None]) == ["core_annotated_one", "bridge_one_to_unannotated", "core_unannotated"]


def test_the_annotated_labels_and_the_entry_majority():
    one, two = dict(gloss=dict(handshapes={"Dominant start handshape": "G"})), dict(gloss=dict(handshapes={"Non-dominant start handshape": "B"}))
    assert lib.annotated_label(one) is True and lib.annotated_label(two) is False and lib.annotated_label(dict()) is None
    pools = dict(a=[one, one, two], b=[two, one], c=[dict()], d=[one])
    assert lib.entry_annotations(pools) == dict(a=True, b=False, d=True)     # a tie is not a majority; no annotation, no label
    assert lib.bridge_kind(True, True) == "one_to_one" and lib.bridge_kind(True, False) == lib.bridge_kind(False, True) == "one_to_two"
    assert lib.bridge_kind(None, True) == "one_to_unannotated" and lib.bridge_kind(False, False) == lib.bridge_kind(None, False) == "other"


def test_one_handed_label():
    assert a2.is_one_handed(.34, 1.) is True and a2.is_one_handed(.35, 1.) is False
    assert lib.annotated_two_handed(dict(gloss=dict(handshapes={"Dominant start handshape": "G"}))) is False
    assert lib.annotated_two_handed(dict(gloss=dict(handshapes={"Non-dominant start handshape": "B-L"}))) is True
    assert lib.annotated_two_handed(dict(gloss=dict(label="x"))) is None and lib.annotated_two_handed(None) is None


def test_paths_by_region_and_what_if():
    speed = np.array([1., 3., 5., 7.])
    ok = np.array([True, True, True, False])
    regions = ["core_one_handed", "bridge", "bridge", "bridge"]
    paths = a2.path_by_region(dict(left=(speed, ok), right=(speed * 2, ok)), regions)
    assert paths["left"]["bridge"] == [8., 2] and paths["left"]["core_one_handed"] == [1., 1] and paths["left"]["lead_trail"] == [0., 0]
    table = a2.region_table(paths)
    assert table["left"]["speed"] == 3. and table["left"]["bridge"]["speed"] == 4. and table["left"]["bridge"]["path_share"] == pytest.approx(8 / 9, abs=1e-4)
    assert a2.what_if(paths, "left", (), 1.5) == 2.
    assert a2.what_if(paths, "left", ("bridge",), 1.) == pytest.approx(1 / 3, abs=1e-4)      # the steps stay
    total = a2.empty_paths()
    a2.add_paths(total, paths)
    a2.add_paths(total, paths)
    assert total["left"]["bridge"] == [16., 4]


def test_a_rebuilt_clip_must_match_the_stored_speed():
    assert a2.rebuild_check([1., None, 2.], [1., 5., 2.25]) == dict(clips=2, max_abs_difference=.25)


def word(corpus="asllvd-text-v1", one=True, left_speed=1., right_speed=2., left=.1, right=1., position=(0., -1.), annotated=False, name="cut"):
    return dict(word=name, corpus=corpus, left_path=left, right_path=right, one_handed=one, annotated_two_handed=annotated, frames_native=20,
                frames_planned=10, native=dict(left=dict(speed=left_speed, tracked=1., position=list(position)), right=dict(speed=right_speed, tracked=1., position=[.2, .2])))


def test_the_tables_on_known_words():
    words = [word(), word(corpus="rit-text-r6", position=(0., 1.)), word(one=False, annotated=True, left=2., right=1.), word(one=True, annotated=True)]
    assert a2.left_active(words)["asllvd-text-v1"] == dict(words=3, share=round(1 / 3, 4))
    check = a2.annotation_check(words)
    assert check["annotated"] == 4 and check["motion_label_agrees"] == .75
    assert a2.rest_pose_table(words)["centre_distance"]["asllvd-text-v1|rit-text-r6"] == pytest.approx(2.)
    entries = a2.per_entry(words, real_left=.5)
    assert entries["by_corpus"]["asllvd-text-v1"]["occurrences"] == 2 and entries["by_corpus"]["asllvd-text-v1"]["left_native_over_real"] == 2.
    assert entries["by_corpus"]["asllvd-text-v1"]["squeeze"]["mean"] == 2.
    joins = a2.join_table([dict(kind="one_to_one", frames=4, same_corpus=False, corpora=["a", "b"], left=dict(travel=1., speed=2.), right=dict(travel=.5, speed=1.)),
                           dict(kind="one_to_one", frames=6, same_corpus=True, corpora=["a", "a"], left=dict(travel=.1, speed=.2), right=dict(travel=None, speed=None))])
    assert joins["one_to_one"]["joins"] == 2 and joins["one_to_one|cross_corpus"]["left_travel"]["mean"] == 1. and joins["one_to_one|same_corpus"]["right_travel"] is None


def test_the_annotated_tables_on_known_words():
    words = [dict(word(annotated=False), annotated_one=True, performer="p1", left_path=.8, right_path=1.),
             dict(word(annotated=False), annotated_one=True, performer="p1", left_path=1.2, right_path=1.),
             dict(word(annotated=True, corpus="rit-text-r6"), annotated_one=False, performer="p2", left_path=2., right_path=2.)]
    paths = a2.empty_paths(a2.ANNOTATED_REGIONS)
    paths["left"]["core_annotated_one"] = [6., 2]
    paths["left"]["bridge_one_to_one"] = [4., 2]
    paths["right"]["core_annotated_two"] = [8., 4]
    joins = [dict(annotated_kind="one_to_one", same_performer=False, same_corpus=True, frames=4, left=dict(travel=1., speed=2.), right=dict(travel=.2, speed=.4))]
    out = a2.annotated_tables(words, joins, paths, dict(left=1., right=1.), dict(left=1., right=1.))
    row = out["by_label_and_corpus"]["one|asllvd-text-v1"]
    assert row["words"] == 2 and row["left_active"] == .5 and row["left_over_right_path"]["mean"] == 1.
    assert out["joins"]["one_to_one|cross_performer"]["left_travel"]["mean"] == 1.
    assert out["what_if"]["left"]["as_is"] == 2.5 and out["what_if"]["left"]["no_path_on_annotated_one_handed_cores"] == 1.
    assert out["what_if"]["left"]["no_path_in_one_to_one_bridges"] == 1.5 and out["what_if"]["left"]["no_path_in_any_bridge"] == 1.5


def test_a_committed_a2_file_uses_one_set_of_joins():
    payload = json.loads((ROOT / "reports/comp_a2_left_arm.json").read_text())
    point = payload["forms"]["a9_predicted_margin0"]
    assert {"one_to_one", "one_to_two", "one_to_unannotated", "other"} == set(point["joins"])
    assert "bridge_one_to_two" in point["regions"]["left"] and point["joins"]["one_to_two"]["joins"] > 0
    assert payload["forms"]["hold"]["ratio_to_real"]["right"] == point["ratio_to_real"]["right"]
    assert payload["forms"]["hold"]["ratio_to_real"]["left"] < point["ratio_to_real"]["left"]
    assert payload["forms"]["hold"]["contact_share"] == point["contact_share"]
    a9 = payload["forms"][rd.A9_FORM]
    assert a9["pooled_wrist_ratio"] and a9["ratio_to_real"]["left"] and a9["pooled_spread_ratio"]
    assert hashlib.sha256((ROOT / "reports/comp_a2_left_arm.json").read_bytes()).hexdigest() == re.search(r"A2SHA=([0-9a-f]{64})", JOB.read_text()).group(1)


def test_a2_is_d20_only():
    text = (ROOT / "scripts/comp_a2_left_arm.py").read_text()
    assert "require_d20(D20_KEY, D20_SHA)" in text and "check_read_records(val)" in text and "c20" not in text.lower()
    assert "load_translator" not in text and "load_generator" not in text          # no hypothesis, no generation


# ---- the clip-choice index --------------------------------------------------------------------------------------------------------

def test_the_candidates_and_draws_are_fixed_and_independent():
    pool = [dict(sign=f"s{k}") for k in range(40)]
    a = a4.candidates(pool, "cut", 16)
    assert len(a) == 16 and a == a4.candidates(pool, "cut", 16) and a != a4.candidates(pool, "onion", 16)
    assert len(a4.candidates(pool[:3], "cut", 16)) == 3
    draws = [a4.independent_draw(pool, "cut", k)["sign"] for k in range(1, 5)]
    assert draws == [a4.independent_draw(pool, "cut", k)["sign"] for k in range(1, 5)] and len(set(draws)) > 1


def choice_row(entry, label, signs, scores, ranks):
    best = {str(n): dict(index=int(np.argmax(scores[:n])) if scores else 0, sign=signs[int(np.argmax(scores[:n])) if scores else 0]) for n in (4, 16)}
    return dict(entry=entry, label=label, signs=signs, scores=scores, probabilities=list(np.exp(scores)) if scores else [], ranks=ranks, chosen=best)


def test_the_choice_summary_says_what_was_not_scored_and_what_saturates():
    rows = [choice_row("a", "A", ["x", "y"], [-1., -1.], [1, 1]), choice_row("b", "B", ["x", "y"], [-3., -.5], [9, 1]),
            choice_row("c", None, ["x"], [], []), choice_row("d", "D", ["x"], [-.1], [2])]
    s = a4.summarise(rows, [4, 16])
    assert s["unlabelled"] == 1 and s["labelled"] == 3 and s["single_clip_entries"] == 2
    assert s["saturation"] == dict(entries_with_several_clips=2, tied_at_the_top=1, top1_every_clip=1)
    assert s["sample0"]["top1"] == round(1 / 3, 4) and s["best_of_16"]["top1"] == round(2 / 3, 4)


def stored_choice(store, sha, ssha, rows, *, samples=16, corpora=None):
    payload = json.dumps(dict(kind="comp-a4-clip-choice", settings=dict(mix_sha256=sha, corpora=corpora or [], samples=samples, choose=[4, 16], picks=["4", "16", "sample0", "draw1"],
                                                                        recognizer_run="sign-recognizer-v3d", sentences=dict(sha256=ssha)),
                              entries=rows, summary=dict(x=1))).encode()
    store.put("c/choice.json", payload)
    return hashlib.sha256(payload).hexdigest()


class Args:
    def __init__(self, sha, csha, ssha, pick="16"):
        self.oracle_choice_key, self.oracle_choice_sha256, self.oracle_choice_pick, self.sentences_sha256, self.mix_sha256 = "c/choice.json", csha, pick, ssha, sha


def test_a_choice_is_checked_before_it_serves(tmp_path):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path)
    record = dict(sign="x", corpus="asllvd-text-v1", partition="train")
    pools = dict(cut=[record, dict(record, sign="y"), dict(record, sign="z")])
    rows = [dict(entry="cut", chosen={"16": dict(sign="y", index=1), "4": dict(sign="x", index=0), "sample0": dict(sign="x", index=0), "draw1": dict(sign="z")})]
    csha = stored_choice(store, "m", "s", rows)
    got = a4.load_choice(store, Args("m", csha, "s"), pools, [])
    assert got["records"]["cut"]["sign"] == "y" and got["info"]["pick"] == "16" and got["info"]["scorer"] == "sign-recognizer-v3d"
    assert a4.load_choice(store, Args("m", csha, "s", pick="4"), pools, [])["records"]["cut"]["sign"] == "x"
    assert a4.load_choice(store, Args("m", csha, "s", pick="sample0"), pools, [])["records"]["cut"]["sign"] == "x"
    assert a4.load_choice(store, Args("m", csha, "s", pick="draw1"), pools, [])["records"]["cut"]["sign"] == "z"
    with pytest.raises(SystemExit, match="does not match"):
        a4.load_choice(store, Args("m", "0" * 64, "s"), pools, [])
    with pytest.raises(SystemExit, match="not a choice on this sentence set"):
        a4.load_choice(store, Args("m", csha, "other"), pools, [])
    with pytest.raises(SystemExit, match="another vocabulary"):
        a4.load_choice(store, Args("other", csha, "s"), pools, [])
    with pytest.raises(SystemExit, match="another vocabulary"):
        a4.load_choice(store, Args("m", csha, "s"), pools, ["semlex"])
    with pytest.raises(SystemExit, match="no pick 8"):
        a4.load_choice(store, Args("m", csha, "s", pick="8"), pools, [])
    with pytest.raises(SystemExit, match="not in its pool"):
        a4.load_choice(store, Args("m", csha, "s"), dict(cut=[record]), [])
    with pytest.raises(SystemExit, match="not a vocabulary source"):
        a4.load_choice(store, Args("m", csha, "s"), dict(cut=[dict(record, sign="y", corpus="asl-citizen-text-v1")]), [])


def test_only_v3d_chooses_clips(capsys):
    with pytest.raises(SystemExit):
        a4.main(["--sentences-key", lib.D20_KEY, "--sentences-sha256", lib.D20_SHA, "--recognizer-run", "sign-recognizer-v2", "--output-file", "x.json"])
    assert "never rerank" in capsys.readouterr().err
    text = (ROOT / "scripts/comp_a_clip_choice.py").read_text()
    assert "load_translator" not in text and "ceiling" in text.split('"""')[1]


# ---- declared means declared ------------------------------------------------------------------------------------------------------

def git_repo(tmp_path, *contents):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    run = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a], check=True, capture_output=True, text=True)   # noqa: E731
    run("init", "-q")
    commits = []
    for k, text in enumerate(contents):
        (repo / dc.PLAN).write_text(text)
        (repo / f"f{k}").write_text("x")
        run("add", "-A")
        run("commit", "-qm", f"c{k}")
        commits.append(run("rev-parse", "HEAD").stdout.strip())
    return repo, commits, run


def put_record(root, commit, **changes):
    (root / "runpod/fleet").mkdir(parents=True, exist_ok=True)
    (root / dc.RECORD).write_text(json.dumps(dict(dc.make_record(commit), **changes)))


def test_declared_needs_an_ancestor_commit_holding_the_rule_and_this_code(tmp_path):
    rule = rd.DRAFT_RULES["a1"]
    words, lines, line = rule.split(), [], ""                                # the plan wraps at spaces
    for w in words:
        if len(line) + len(w) > 70:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    repo, (first, second), run = git_repo(tmp_path, "no rule here", "Plan.\n\n" + "\n".join(lines))
    with pytest.raises(SystemExit, match="nothing shows a declaration"):
        dc.check(second, "a1", repo)                                         # git is not enough: the record binds the code
    put_record(repo, second)
    assert dc.check(second, "a1", repo) == "git"
    with pytest.raises(SystemExit, match="names .*not"):
        dc.check(first, "a1", repo)
    put_record(repo, first)
    with pytest.raises(SystemExit, match="does not hold the rule of a1"):
        dc.check(first, "a1", repo)
    put_record(repo, second)
    with pytest.raises(SystemExit, match="does not hold the rule of a3-oracle"):
        dc.check(second, "a3-oracle", repo)
    put_record(repo, "deadbeef")
    with pytest.raises(SystemExit, match="not a commit of this repository"):
        dc.check("deadbeef", "a1", repo)
    with pytest.raises(SystemExit, match="is not a commit"):
        dc.check("HEAD", "a1", repo)
    with pytest.raises(SystemExit, match="is not a commit"):
        dc.check("", "a1", repo)
    put_record(repo, second)
    run("checkout", "-q", first)                                            # the declaring commit is not an ancestor of this code
    with pytest.raises(SystemExit, match="not an ancestor"):
        dc.check(second, "a1", repo)
    put_record(repo, second, code_digest="0" * 64)
    with pytest.raises(SystemExit, match="declared for code"):
        dc.check(second, "a1", repo)                                         # with git too
    assert dc.holds_rule("x\n" + "\n".join(lines) + "\ny", rule) and not dc.holds_rule(rule[:-2], rule)


def test_a_bundle_without_git_needs_the_record_the_text_and_this_code(tmp_path):
    root = tmp_path / "bundle"
    (root / "docs").mkdir(parents=True)
    with pytest.raises(SystemExit, match="nothing shows a declaration"):
        dc.check("abc1234", "a1", root)
    rule = rd.DRAFT_RULES["a1"]
    (root / dc.PLAN).write_text("Plan. " + rule)
    put_record(root, "abc1234")
    assert dc.check("abc1234", "a1", root) == "record"
    with pytest.raises(SystemExit, match="names abc1234"):
        dc.check("abc1235", "a1", root)
    with pytest.raises(SystemExit, match="bundle's plan does not hold the rule of a3-gen"):
        dc.check("abc1234", "a3-gen", root)
    put_record(root, "abc1234", rules=dict(a1="0"))
    with pytest.raises(SystemExit, match="does not hold this rule of a1"):
        dc.check("abc1234", "a1", root)
    put_record(root, "abc1234", code_digest="0" * 64)
    with pytest.raises(SystemExit, match="declared for code"):
        dc.check("abc1234", "a1", root)
    put_record(root, "abc1234")
    (root / dc.PLAN).write_text("Plan.")
    with pytest.raises(SystemExit, match="plan does not hold"):
        dc.check("abc1234", "a1", root)


def test_a_record_binds_the_repositorys_code():
    """The builder matches the code now. The stored record binds the code it was declared on, so later merges change the digest on purpose: a pod runs
    from the declared tree, and a read from any other refuses. Here: the record is whole, and its commit is an ancestor whose plan holds every rule."""
    from signtranslator.eval.provenance import code_digest
    digest = code_digest(extra_files=(lib.JOB_FILE,))
    assert dc.make_record("abc1234")["code_digest"] == digest and set(dc.make_record("abc1234")["rules"]) == set(rd.DRAFT_RULES)
    payload = json.loads((ROOT / dc.RECORD).read_text())
    assert re.fullmatch(r"[0-9a-f]{64}", payload["code_digest"]) and set(payload["rules"]) == set(rd.DRAFT_RULES)
    known = subprocess.run(["git", "cat-file", "-e", payload["commit"]], cwd=ROOT, capture_output=True).returncode == 0
    if known:     # a shallow CI checkout may not have it
        plan = subprocess.run(["git", "show", f"{payload['commit']}:{dc.PLAN}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        for text in rd.DRAFT_RULES.values():
            assert dc.holds_rule(plan, text)


def test_the_digest_covers_scripts_the_package_and_the_job(tmp_path, monkeypatch):
    from signtranslator.eval import provenance
    (tmp_path / "scripts").mkdir()
    (tmp_path / "signtranslator/eval").mkdir(parents=True)
    (tmp_path / "scripts/a.py").write_text("1")
    (tmp_path / "scripts/note.md").write_text("1")
    (tmp_path / "signtranslator/eval/b.py").write_text("2")
    (tmp_path / "job.sh").write_text("3")
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    base = provenance.code_digest(extra_files=("job.sh",))
    assert base == provenance.code_digest(extra_files=("job.sh",)) and base != provenance.code_digest()
    (tmp_path / "scripts/note.md").write_text("changed")
    assert provenance.code_digest(extra_files=("job.sh",)) == base          # only the code
    for path in ("scripts/a.py", "signtranslator/eval/b.py", "job.sh"):
        before = (tmp_path / path).read_text()
        (tmp_path / path).write_text(before + "x")
        assert provenance.code_digest(extra_files=("job.sh",)) != base, path
        (tmp_path / path).write_text(before)
    (tmp_path / "scripts/c.py").write_text("")
    assert provenance.code_digest(extra_files=("job.sh",)) != base          # a new file counts
    with pytest.raises(FileNotFoundError):
        provenance.code_digest(extra_files=("gone.sh",))


def test_the_environment_names_what_can_move_a_hypothesis():
    from signtranslator.eval import provenance
    env = provenance.environment()
    assert set(env) == {"python", "torch", "transformers", "tokenizers", "gpu"} and env["python"] and env["torch"]
    row = lib.provenance_row()
    assert row["environment"] == env and len(row["code_digest"]) == 64


def test_the_declared_rules_stay_in_the_plan():
    plan = (ROOT / dc.PLAN).read_text()
    for test, text in rd.DRAFT_RULES.items():
        assert dc.holds_rule(plan, text), test


# ---- the real path, on the round-15 synthetic store ----------------------------------------------------------------------------

@pytest.fixture
def d20(tmp_path, stage2_store, likelihood_stub, monkeypatch):  # noqa: F811
    """The synthetic store with its D20 set as the pinned one."""
    store, sha, *_ = stage2_store
    put_gains(store)
    ssha, texts = put_set(store, sha)
    monkeypatch.setattr(lib, "D20_KEY", "sets/d20.json")
    monkeypatch.setattr(lib, "D20_SHA", ssha)
    monkeypatch.setattr(cv2, "fit_duration_model", lambda manifest: dict(corpus=cv2.H2S, partition="train", records=18, intercept=.4,
                                                                         slope=.5, median_abs_error=1.))
    return dict(store=store, sha=sha, ssha=ssha, texts=texts, tmp=tmp_path)


def oracle_args(c, *extra):
    args = [a for a in run_args(c["tmp"], c["sha"]) if a not in ("--checkpoint-key", "prior.pt")]
    args[args.index("core_crossfade,oracle_stage2")] = "oracle_stage2"
    return [*args, "--sentences-key", "sets/d20.json", "--sentences-sha256", c["ssha"], "--likelihood", *extra]


def part(c, key):
    return json.loads(c["store"].get(key))


def test_scaled_forms_keep_their_plans_change_their_motion_and_read_the_control(d20):
    c = d20
    plain = oracle_args(c, "--core-margin", "0", "--predicted-duration")
    assert ce.main([*plain, "--comp-a-read", "--output-key", "e/plain-shard0.json"]) == 0
    assert ce.main([*plain, "--arm-scale", "left=0.5,right=1", "--output-key", "e/left-shard0.json"]) == 0
    a, b = part(c, "e/plain-shard0.json"), part(c, "e/left-shard0.json")
    assert b["settings"]["round20"]["arm_scale"] == dict(left=.5, right=1.) and "arm_scale" not in a["settings"]["round20"]
    from signtranslator.eval.provenance import code_digest
    assert a["settings"]["round20"]["provenance"] == b["settings"]["round20"]["provenance"]
    assert b["settings"]["round20"]["provenance"]["code_digest"] == code_digest(extra_files=(lib.JOB_FILE,)) and "torch" in b["settings"]["round20"]["provenance"]["environment"]
    moved = 0
    for x, y in zip(a["clips"], b["clips"]):
        assert x["plans"] == y["plans"]                                # the same words and frames
        assert x["motion"]["real"] == y["motion"]["real"]              # real itself is untouched
        assert x["hypotheses"]["blank"] == y["hypotheses"]["blank"] and x["likelihood"]["real"] == y["likelihood"]["real"]
        if x["entries"]:
            moved += x["motion"]["oracle_stage2@predicted"]["wrist_speed"] != y["motion"]["oracle_stage2@predicted"]["wrist_speed"]
            for form in ("oracle_stage2", "oracle_stage2/shuffled", "oracle_stage2/mismatched_words"):
                assert x["motion"][form]["wrist_speed"] != y["motion"][form]["wrist_speed"], form
            # left scaled: the right side of the composed form stays
            assert y["comp_a_motion"]["oracle_stage2"]["right_speed"] == pytest.approx(x["comp_a_motion"]["oracle_stage2"]["right_speed"], rel=1e-5)
            assert y["comp_a_motion"]["oracle_stage2"]["left_speed"] != pytest.approx(x["comp_a_motion"]["oracle_stage2"]["left_speed"], rel=1e-5)
        # the control scales the same side as the arm: the left of real moves, its right does not
        control = y["comp_a_motion"][lib.REAL_SCALED]
        assert control["right_speed"] == pytest.approx(y["comp_a_motion"]["real"]["right_speed"], rel=1e-5)
        assert control["left_speed"] != pytest.approx(y["comp_a_motion"]["real"]["left_speed"], rel=1e-5)
        assert y["motion"][lib.REAL_SCALED]["wrist_speed"] != y["motion"]["real"]["wrist_speed"]
        assert {"real", lib.REAL_SCALED} <= set(y["spotting"]) and lib.REAL_SCALED in y["likelihood"]
        assert lib.REAL_SCALED not in x["motion"] and "real" not in x["spotting"]
    assert moved == sum(bool(x["entries"]) for x in a["clips"])
    merge(c["tmp"], "e/left-shard0.json", "e/left.json", "--report-only")
    m = part(c, "e/left.json")
    assert m["report_only"] is True and m["verdicts"] == {}
    assert "minus_real" in m["forms"][lib.REAL_SCALED] and "minus_blank" in m["forms"][lib.REAL_SCALED]
    assert "minus_real" in m["likelihood"]["forms"][lib.REAL_SCALED]
    assert "minus_real_paired" in m["spotting"]["scores"][lib.REAL_SCALED]
    assert m["comp_a_motion"]["oracle_stage2"]["right_speed_over_real"] > 0
    assert rd.control_read(m, "oracle_stage2@predicted", boot=20)["scales"] == dict(left=.5, right=1.)


def cache_for_words(c, prefix, *extra):
    common = ["--store", f"dir:{c['tmp'] / 'store'}", "--prefix", prefix]
    assert cw.main(["run", *common, "--mix-key", "mix.json", "--mix-sha256", c["sha"], "--sentences-key", "sets/d20.json",
                    "--sentences-sha256", c["ssha"], "--checkpoint-key", "prior.pt", "--samples", "4", "--choose", "4",
                    "--device", "cpu", "--no-decoder-gains", *extra]) == 0
    assert cw.main(["merge", *common]) == 0
    return hashlib.sha256(c["store"].get(f"{prefix}/index.json")).hexdigest()


def generated_args(c, *extra):
    args = oracle_args(c)
    args[args.index("oracle_stage2")] = "core_crossfade"
    return [*args, *extra]


@pytest.fixture
def recogniser(monkeypatch):
    import validate_sign_recognizer as vsr
    labels = ["CUT", "ONION", "SALT", "HEAT", "KNIFE", "HALF", "NOW", "STIR", "I", "YOU"]
    monkeypatch.setattr(vsr, "load_model", lambda store, run, device: (None, labels, {"checkpoint_key": "r.pt"}))

    def logits(model, states, device, batch=64):
        out = []
        for s in states:
            rh = s.channels["right_hand"]
            level = float(np.abs(rh.values[rh.valid]).mean()) if rh.valid.any() else 0.
            out.append(torch.tensor([np.sin(level * (k + 3) * 50.) for k in range(len(labels))]))
        return torch.stack(out)
    monkeypatch.setattr(vsr, "logits_of", logits)
    return labels


def test_the_hold_arm_changes_only_the_left_arm_on_the_plan_it_was_given(d20, monkeypatch, recogniser):
    c = d20
    labels = {"cut": True, "onion": False, "salt": True, "stir": False, "heat": True, "knife": False, "half": None}
    import comp_a_lib
    monkeypatch.setattr(comp_a_lib, "entry_annotations", lambda pools: {k: v for k, v in labels.items() if v is not None})
    sha = cache_for_words(c, "cache/m")
    stats = ROOT / "reports/comp_a_hold_stats.json"
    words = ("--word-cache-key", "cache/m/index.json", "--word-cache-sha256", sha, "--word-cache-n", "4", "--core-margin", "0", "--predicted-duration")
    assert ce.main([*generated_args(c, *words, "--comp-a-read"), "--output-key", "e/m0-shard0.json"]) == 0
    assert ce.main([*generated_args(c, *words, "--hold-left-file", str(stats), "--hold-left-sha256", hashlib.sha256(stats.read_bytes()).hexdigest()),
                    "--output-key", "e/hold-shard0.json"]) == 0
    a, b = part(c, "e/m0-shard0.json"), part(c, "e/hold-shard0.json")
    assert b["settings"]["round20"]["hold_left"]["sha256"] and "hold_left" not in a["settings"]["round20"] and a["settings"]["round20"]["comp_a_read"] is True
    stretches = 0
    changed = 0
    for x, y in zip(a["clips"], b["clips"]):
        assert x["entries"] == y["entries"] and x["motion"]["real"] == y["motion"]["real"]
        for form in ("core_crossfade", PREDICTED):
            if form in x["comp_a_motion"]:
                assert y["comp_a_motion"][form]["right_speed"] == pytest.approx(x["comp_a_motion"][form]["right_speed"], rel=1e-6)
                assert y["comp_a_motion"][form]["right_spread"] == pytest.approx(x["comp_a_motion"][form]["right_spread"], rel=1e-6)
                changed += y["comp_a_motion"][form]["left_speed"] != x["comp_a_motion"][form]["left_speed"]
        if x["entries"]:
            plan_x, plan_y = x["plans"]["core_crossfade"], y["plans"]["core_crossfade"]
            assert {k: v for k, v in plan_y.items() if k not in ("hold", "predicted")} == {k: v for k, v in plan_x.items() if k != "predicted"}
            stretches += plan_y["hold"]["stretches"]
            assert plan_y["predicted"]["frames"] == plan_x["predicted"]["frames"] and "hold" in plan_y["predicted"]
    assert stretches > 0 and changed > 0
    merge(c["tmp"], "e/hold-shard0.json", "e/hold.json", "--report-only")
    m = part(c, "e/hold.json")
    assert m["hold_left"]["stretches"] > 0 and m["hold_left"]["cap"] == CAP and PREDICTED in m["comp_a_motion"]


def test_the_controls_of_the_hold_are_held_with_their_own_words(d20, monkeypatch, recogniser):
    """A shuffled form holds by its own order's labels; a mismatched form by its partner's words."""
    c = d20
    import comp_a_lib
    seen = []
    real_hold = cv2.hold_left_arm

    def spy(state, plan, flags, **kw):
        seen.append(list(flags))
        return real_hold(state, plan, flags, **kw)
    monkeypatch.setattr(cv2, "hold_left_arm", spy)
    monkeypatch.setattr(comp_a_lib, "entry_annotations", lambda pools: {"cut": True, "onion": False})
    sha = cache_for_words(c, "cache/m")
    stats = ROOT / "reports/comp_a_hold_stats.json"
    assert ce.main([*generated_args(c, "--word-cache-key", "cache/m/index.json", "--word-cache-sha256", sha, "--word-cache-n", "4", "--core-margin", "0",
                                    "--hold-left-file", str(stats), "--hold-left-sha256", hashlib.sha256(stats.read_bytes()).hexdigest()),
                    "--output-key", "e/h-shard0.json"]) == 0
    shard = part(c, "e/h-shard0.json")
    mapped = {"cut": True, "onion": False}
    row = next(r for r in shard["clips"] if "cut" in r["entries"] and "onion" in r["entries"] and r["shuffled_order"] != sorted(r["shuffled_order"]))
    assert [mapped.get(w) for w in row["entries"]] in seen
    assert [mapped.get(row["entries"][k]) for k in row["shuffled_order"]] in seen


def test_a_post_s2_shard_merges_only_as_report_only(d20):
    c = d20
    for flags in (["--arm-scale", "left=0.5,right=1"], ["--comp-a-read"]):
        key = f"e/s{len(flags)}"
        assert ce.main([*oracle_args(c, *flags), "--output-key", f"{key}-shard0.json"]) == 0
        with pytest.raises(SystemExit, match="report-only"):
            merge(c["tmp"], f"{key}-shard0.json", f"{key}.json")
        assert not c["store"].exists(f"{key}.json")


@pytest.mark.parametrize("flags", [["--arm-scale", "left=0.5,right=1"], ["--comp-a-read"],
                                   ["--oracle-choice-key", "c/x.json", "--oracle-choice-sha256", "0" * 64, "--oracle-choice-pick", "16"],
                                   ["--vocabulary-extra-file", str(TOKENS), "--vocabulary-extra-sha256", "0" * 64, "--vocabulary-extra-top", "10"]])
@pytest.mark.parametrize("sentences", [[], ["--sentences-key", "sets/c20.json", "--sentences-sha256", "0" * 64]])
def test_the_options_refuse_any_other_set_before_reading_it(d20, capsys, flags, sentences):
    c = d20
    args = list(oracle_args(c))
    k = args.index("--sentences-key")
    args = args[:k] + args[k + 4:]
    with pytest.raises(SystemExit):
        ce.main([*args, *sentences, *flags, "--output-key", "e/x.json"])
    assert "D20 only" in capsys.readouterr().err and not c["store"].exists("e/x.json")


def test_a_list_named_d20_by_key_but_not_by_its_own_field_stops(d20, monkeypatch):
    c = d20
    payload = json.loads(c["store"].get("sets/d20.json"))
    payload["set"] = "C20"
    raw = json.dumps(payload).encode()
    c["store"].put("sets/d20.json", raw)
    sha = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(lib, "D20_SHA", sha)
    args = oracle_args(c, "--arm-scale", "left=0.5,right=1")
    args[args.index(c["ssha"])] = sha
    with pytest.raises(SystemExit, match="not D20"):
        ce.main([*args, "--output-key", "e/x.json"])


@pytest.mark.parametrize("flags,message", [
    (["--arm-scale", "left=1,right=1"], "no scale"),
    (["--vocabulary-extra-sha256", "x"], "without the option"),
    (["--vocabulary-extra-top", "10"], "without the option"),
    (["--word-cache-extra-sha256", "x"], "without the option"),
    (["--oracle-choice-pick", "16"], "without the option"),
    (["--hold-left-sha256", "x"], "without the option"),
    (["--hold-left-file", "x.json"], "needs --hold-left-sha256"),
    (["--hold-left-file", "x.json", "--hold-left-sha256", "0"], "core_crossfade alone"),
    (["--vocabulary-extra-file", "x.json"], "needs --vocabulary-extra-sha256"),
    (["--vocabulary-extra-file", "x.json", "--vocabulary-extra-sha256", "0", "--vocabulary-extra-top", "20"], "needs --vocabulary-extra-sha256"),
    (["--vocabulary-extra-file", "x.json", "--vocabulary-extra-sha256", "0", "--vocabulary-extra-top", "10", "--oracle-source", "continuous"], "its own segments"),
    (["--vocabulary-extra-file", "x.json", "--vocabulary-extra-sha256", "0", "--vocabulary-extra-top", "10", "--retime", "continuous"], "reference retiming"),
    (["--word-cache-extra-key", "c.json"], "needs --word-cache-extra-sha256"),
    (["--oracle-choice-key", "c.json"], "needs --oracle-choice-sha256"),
    (["--oracle-choice-key", "c.json", "--oracle-choice-sha256", "0", "--oracle-choice-pick", "16", "--oracle-source", "continuous"], "alone, isolated"),
    (["--arm-scale", "left=.5,right=1", "--also-translator-run", "x"], "second translator"),
])
def test_the_options_refuse_what_does_not_apply(d20, capsys, flags, message):
    with pytest.raises(SystemExit):
        ce.main([*oracle_args(d20, *flags), "--output-key", "e/x.json"])
    assert message in capsys.readouterr().err


def test_generated_words_with_a_list_need_the_extra_cache(d20, capsys):
    c = d20
    with pytest.raises(SystemExit):
        ce.main([*generated_args(c), "--checkpoint-key", "prior.pt", "--vocabulary-extra-file", str(TOKENS), "--vocabulary-extra-sha256", "0",
                 "--vocabulary-extra-top", "10", "--output-key", "e/x.json"])
    assert "no fresh generation" in capsys.readouterr().err


def write_list(c, rows, **extra):
    path = c["tmp"] / "tokens.json"
    path.write_text(json.dumps(dict(kind="comp-a3-tokens", min_segment_seconds=.2, inputs=dict(source_mix=dict(key="mix.json", sha256=c["sha"])), tokens=rows, **extra)))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def full_ranks(*rows):
    rows = list(rows)
    have = {r["rank"] for r in rows}
    return rows + [row(k, f"unused{k}", "no_source") for k in range(1, 31) if k not in have]


def asllvd_clip(store, sha, text):
    from tests.test_round20_eval import manifest_of
    return next(r for r in manifest_of(store, sha)["records"] if r.get("text") == text and r.get("partition") == "train")


def token_rows(c, monkeypatch):
    """you: a segment of the fixture's utterances; half: an isolated clip of the mix; it: rank 11 (top 30 only); this: no source."""
    import round15_stage2_fit as fit
    import round20_lib
    gsha = put_gloss_inputs(c["store"])
    monkeypatch.setattr(round20_lib, "GLOSS_MANIFEST", ("gloss.json", gsha))
    items = json.loads(c["store"].get(fit.KEYS["utterances"]))["items"]
    k, u = next((k, u) for k, u in enumerate(items) if u["source"].startswith("Cory"))
    a, b = u["signs"][0]["frames"][:2]
    segment = dict(sign=f"{u['collection']}:u{k}#0", corpus=lib.SEGMENT_CORPUS, gloss="IX-2p", gloss_kind="index", occurrence="IX-2p", partition="train",
                   record_sign=f"{u['collection']}:u{k}", start=int(a - u["frames"][0]), end=int(b - u["frames"][0]), min_seconds=.2,
                   utterance_frames=int(u["frames"][1] - u["frames"][0] + 1), tracked=dict(arm=1., hand=1.), seconds=.3)
    clip = asllvd_clip(c["store"], c["sha"], "knife")
    source = dict(sign=clip["sign"], corpus=clip["corpus"], partition="train", signer_component=clip.get("signer_component"))
    return full_ranks(row(1, "you", kind="segment", seconds=.9, sources=[segment]), row(2, "half", seconds=.7, sources=[source]),
                      row(3, "this", "no_source"), row(11, "it", seconds=.6, sources=[source]))


def test_the_oracle_arm_adds_only_its_ranks_from_only_its_clips(d20, monkeypatch):
    c = d20
    path, sha = write_list(c, token_rows(c, monkeypatch))
    common = ("--core-margin", "0", "--predicted-duration")
    assert ce.main([*oracle_args(c, *common, "--comp-a-read"), "--output-key", "e/base-shard0.json"]) == 0
    assert ce.main([*oracle_args(c, *common, "--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top", "10"),
                    "--output-key", "e/top10-shard0.json"]) == 0
    base, top = part(c, "e/base-shard0.json"), part(c, "e/top10-shard0.json")
    assert top["settings"]["round20"]["vocabulary_extra"] == dict(sha256=sha, top=10) and "vocabulary_extra" not in base["settings"]["round20"]
    info = top["round20"]["vocabulary_extra"]
    assert info["new_entries"] == ["half", "you"] and "this" in info["unsigned_ranks"] and info["min_segment_seconds"] == .2       # the widening is disclosed
    assert info["tokens"].get("you", 0) >= 1 and info["tokens"].get("half", 0) >= 1 and "it" not in info["tokens"]   # rank 11 waits for the top 30
    for x, y in zip(base["clips"], top["clips"]):
        added = [w for w in y["entries"] if w in ("you", "half")]
        assert [w for w in y["entries"] if w not in ("you", "half")] == x["entries"]   # every r10 word, in order
        assert "this" not in y["entries"] and "it" not in y["entries"]
        if added:
            sources = y["plans"]["oracle_stage2"]["round20"]["sources"]
            assert dict(zip(y["entries"], sources)).get("you", "continuous") == "continuous"
            reasons = dict(zip(y["entries"], y["plans"]["oracle_stage2"]["round20"]["reasons"]))
            assert reasons.get("half", "all") == "all"                                  # a new isolated word: the plan weight's rule, not the activity rule
        assert x["motion"]["real"] == y["motion"]["real"]
    assert any("you" in r["entries"] for r in top["clips"]) and any("half" in r["entries"] for r in top["clips"])
    assert top["round20"]["coverage"]["words_total"] > base["round20"]["coverage"]["words_total"]
    merge(c["tmp"], "e/top10-shard0.json", "e/top10.json", "--report-only")
    assert part(c, "e/top10.json")["report_only"] is True


def test_a_negation_token_is_signed_as_not_whatever_its_rank():
    payload = dict(tokens=[row(k, f"w{k}", "no_source") for k in range(1, 31)], negation_tokens=["didn't", "can't"])
    for top in lib.TOPS:
        arm = lib.arm_of(payload, top)
        assert arm["token_map"] == {"didn't": "not", "can't": "not"} and not arm["entries"]
    details, words = lib.apply_token_map([comp.selection_details("you didn't cut it", dict(cut=1., can=1.))], arm["token_map"])
    assert words[0] == ["not", "cut"]


def test_a_new_entry_never_takes_a_letters_timing_or_fraction(d20, monkeypatch):
    """The fixture's "knife" stands for r10's "i": an annotated span and an ASLLVD median of its own that the list's word must not inherit."""
    c = d20
    import round15_stage2 as r2
    import round15_stage2_fit as fit
    source = token_rows(c, monkeypatch)[1]["sources"][0]
    path, sha = write_list(c, full_ranks(row(1, "knife", seconds=.77, sources=[source])))
    fractions_seen, plans_seen = [], []
    real_oracle, real_plan = r2.oracle_material, r2.plan_words
    monkeypatch.setattr(fit, "annotated_seconds", lambda manifest, vocab: {e: .3 for e in vocab})
    real_fractions = fit.sign_fractions

    def with_a_letter(manifest, vocab):                        # as ASLLVD gives r10's "i" a median of its own
        out = real_fractions(manifest, vocab)
        return dict(out, entries=dict(out["entries"], knife=(.1, .2)), all=(.25, .75))
    monkeypatch.setattr(fit, "sign_fractions", with_a_letter)

    def spy_oracle(job, clip, pools, **kw):
        fractions_seen.append(kw["fractions"])
        return real_oracle(job, clip, pools, **kw)

    def spy_plan(entries, model, vocab, **kw):
        plans_seen.append((model["timing"]["annotated"], vocab))
        return real_plan(entries, model, vocab, **kw)
    monkeypatch.setattr(r2, "oracle_material", spy_oracle)
    monkeypatch.setattr(r2, "plan_words", spy_plan)
    assert ce.main([*oracle_args(c, "--core-margin", "0", "--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top", "10"),
                    "--output-key", "e/k-shard0.json"]) == 0
    annotated, vocab = plans_seen[0]
    assert "knife" not in annotated and "cut" in annotated and vocab["knife"] == .77
    assert "knife" not in fractions_seen[0]["entries"] and fractions_seen[0]["min_seconds"] == {"knife": .2}
    assert fractions_seen[0]["all"] and "activity" not in fractions_seen[0]


def test_a_new_generated_word_gets_the_same_floor_as_its_real_piece():
    import scripts.round15_stage2 as r2
    state = signer_state(frames=48, amplitude=.2)
    fractions = dict(entries={}, all=(.5, .5), min_seconds={"you": .2})
    wide = r2.sign_core("you", state, fractions=fractions, margin=0.)
    assert wide["hi"] - wide["lo"] + 1 == round(.2 * 24) and wide["reason"] == "all"
    plain = r2.sign_core("cut", state, fractions=fractions, margin=0.)
    assert plain["hi"] - plain["lo"] + 1 == 1
    assert r2.sign_core("you", state, fractions=dict(entries={}, all=(.5, .5)), margin=0.)["hi"] == plain["hi"]        # no floor unless asked
    edge = r2.sign_core("you", signer_state(frames=3, amplitude=.2), fractions=fractions, margin=0.)
    assert edge["lo"] == 0 and edge["hi"] == 2                                                                           # a short clip stops at its ends


def test_a_letter_of_the_vocabulary_is_never_served_as_the_lists_word(d20, monkeypatch):
    c = d20
    rows = token_rows(c, monkeypatch)
    rows[0]["sources"][0]["utterance_frames"] = 9999
    path, sha = write_list(c, rows)
    with pytest.raises(KeyError, match="no training record"):
        ce.main([*oracle_args(c, "--core-margin", "0", "--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha,
                              "--vocabulary-extra-top", "10"), "--output-key", "e/x.json"])


def test_an_entry_the_vocabulary_also_holds_is_served_only_from_the_list(d20, monkeypatch):
    c = d20
    source = token_rows(c, monkeypatch)[0]["sources"][0]
    path, sha = write_list(c, full_ranks(row(1, "knife", kind="segment", seconds=.9, sources=[dict(source, utterance_frames=9999)])))
    with pytest.raises(KeyError, match="no training record"):
        ce.main([*oracle_args(c, "--core-margin", "0", "--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha,
                              "--vocabulary-extra-top", "10"), "--output-key", "e/x.json"])


def test_only_the_lists_new_entries_are_generated_and_the_composition_joins_both_caches(d20, monkeypatch, recogniser):
    c = d20
    path, sha = write_list(c, token_rows(c, monkeypatch))
    extra = ("--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top")
    main_sha = cache_for_words(c, "cache/main")
    extra_sha = cache_for_words(c, "cache/extra", *extra, "30")
    main_rows, extra_rows = part(c, "cache/main/index.json"), part(c, "cache/extra/index.json")
    assert sorted(r["entry"] for r in extra_rows["entries"]) == ["half", "it", "you"]
    assert extra_rows["settings"]["vocabulary_extra"] == dict(sha256=sha, top=30) and "vocabulary_extra" not in main_rows["settings"]
    assert extra_rows["settings"]["provenance"]["code_digest"] and "provenance" not in main_rows["settings"]
    assert not {r["entry"] for r in main_rows["entries"]} & {"half", "it", "you"}
    assert next(r for r in extra_rows["entries"] if r["entry"] == "you")["seconds"] == .9
    words = ("--word-cache-key", "cache/main/index.json", "--word-cache-sha256", main_sha, "--word-cache-n", "4", "--core-margin", "0",
             "--predicted-duration", "--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top", "30",
             "--word-cache-extra-key", "cache/extra/index.json", "--word-cache-extra-sha256", extra_sha)
    assert ce.main([*generated_args(c, *words), "--output-key", "e/gen-shard0.json"]) == 0
    shard = part(c, "e/gen-shard0.json")
    assert shard["settings"]["round20"]["word_cache_extra"]["sha256"] == extra_sha
    assert {"you", "half", "it"} <= {w for r in shard["clips"] for w in r["entries"]}
    reasons = {w: x for r in shard["clips"] for w, x in zip(r["entries"], r["plans"]["core_crossfade"]["round20"]["reasons"])}
    assert all(reasons[w] == "all" for w in ("you", "half", "it"))
    args = list(words)
    args[args.index(extra_sha)] = "0" * 64
    with pytest.raises(SystemExit, match="does not match --word-cache-extra-sha256"):
        ce.main([*generated_args(c, *args), "--output-key", "e/x.json"])
    with pytest.raises(SystemExit, match="holds a list's new entries"):
        args = list(words)
        args[args.index("cache/main/index.json")], args[args.index(main_sha)] = "cache/extra/index.json", extra_sha
        ce.main([*generated_args(c, *args), "--output-key", "e/x.json"])
    other_path, other_sha = write_list(c, token_rows(c, monkeypatch)[:-1] + [row(30, "unused30x", "no_source")])
    args = list(words)
    args[args.index(str(path))], args[args.index(sha)] = str(other_path), other_sha
    with pytest.raises(SystemExit, match="vocabulary_extra"):
        ce.main([*generated_args(c, *args), "--output-key", "e/x.json"])


@pytest.mark.parametrize("field,value,named", [("checkpoint_key", "another.pt", "checkpoint_key"), ("no_decoder_gains", False, "no_decoder_gains"),
                                               ("recognizer_run", "another", "recognizer_run"), ("guidance", 9., "guidance")])
def test_an_extra_cache_of_another_prior_or_decoder_never_joins(d20, monkeypatch, recogniser, field, value, named):
    c = d20
    path, sha = write_list(c, token_rows(c, monkeypatch))
    main_sha = cache_for_words(c, "cache/main")
    cache_for_words(c, "cache/extra", "--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top", "30")
    index = part(c, "cache/extra/index.json")
    index["settings"][field] = value
    raw = json.dumps(index).encode()
    c["store"].put("cache/bent/index.json", raw)
    with pytest.raises(SystemExit, match=named):
        ce.main([*generated_args(c, "--word-cache-key", "cache/main/index.json", "--word-cache-sha256", main_sha, "--word-cache-n", "4", "--core-margin", "0",
                                 "--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top", "30",
                                 "--word-cache-extra-key", "cache/bent/index.json", "--word-cache-extra-sha256", hashlib.sha256(raw).hexdigest()),
                 "--output-key", "e/x.json"])


def test_a_new_single_letter_word_is_never_scored_by_the_letters_label(d20, monkeypatch, recogniser):
    c = d20
    import round20_lib
    rows = token_rows(c, monkeypatch)
    import round15_compose_eval as top                # the word cache's own import of it
    selection = top.stage1_selection

    def with_i(texts, vocab):
        return selection([t + " i" for t in texts], vocab)
    monkeypatch.setattr(top, "stage1_selection", with_i)
    rows = full_ranks(rows[0], rows[1], rows[2], row(4, "i", kind="segment", seconds=.5, sources=rows[0]["sources"]))
    path, sha = write_list(c, rows)
    cache_for_words(c, "cache/letter", "--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top", "10")
    index = {r["entry"]: r for r in part(c, "cache/letter/index.json")["entries"]}
    assert index["i"]["label"] is None and index["i"]["scorer"] == "prior_pmi"
    assert index["you"]["label"] == "YOU" and index["you"]["scorer"] == "recogniser"
    assert round20_lib.entry_label("i", recogniser) == "I"            # what the plain rule would have used


@pytest.mark.parametrize("flags,message", [
    (["--vocabulary-extra-sha256", "x"], "go together"),
    (["--vocabulary-extra-file", "x", "--vocabulary-extra-sha256", "x", "--vocabulary-extra-top", "9"], "go together"),
])
def test_the_cache_refuses_a_partial_list(d20, capsys, flags, message):
    c = d20
    with pytest.raises(SystemExit):
        cw.main(["run", "--store", f"dir:{c['tmp'] / 'store'}", "--prefix", "cache/x", "--mix-key", "mix.json", "--mix-sha256", c["sha"],
                 "--sentences-key", "sets/d20.json", "--sentences-sha256", c["ssha"], "--checkpoint-key", "prior.pt", "--device", "cpu", *flags])
    assert message in capsys.readouterr().err


def test_the_cache_refuses_another_set_with_a_list(d20, capsys):
    c = d20
    with pytest.raises(SystemExit):
        cw.main(["run", "--store", f"dir:{c['tmp'] / 'store'}", "--prefix", "cache/x", "--mix-key", "mix.json", "--mix-sha256", c["sha"],
                 "--sentences-key", "sets/c20.json", "--sentences-sha256", "0" * 64, "--checkpoint-key", "prior.pt", "--device", "cpu",
                 "--vocabulary-extra-file", "x", "--vocabulary-extra-sha256", "x", "--vocabulary-extra-top", "10"])
    assert "D20 only" in capsys.readouterr().err


def choice_args(c, *extra):
    return ["--store", f"dir:{c['tmp'] / 'store'}", "--mix-key", "mix.json", "--mix-sha256", c["sha"], "--sentences-key", "sets/d20.json",
            "--sentences-sha256", c["ssha"], "--samples", "4", "--choose", "1,4", "--device", "cpu", "--cache-dir", str(c["tmp"] / "cache"), *extra]


def test_clips_are_chosen_by_the_recogniser_and_every_pick_serves_the_oracle(d20, recogniser, monkeypatch):
    c = d20
    import round15_oracle
    import round15_stage2 as r2
    plain_pools = round15_oracle.word_pools

    def two_clips(manifest, corpora, partition="train"):            # each entry also holds the next entry's clip
        pools = plain_pools(manifest, corpora, partition)
        names = sorted(pools)
        return {e: pools[e] + pools[names[(k + 1) % len(names)]] for k, e in enumerate(names)}
    monkeypatch.setattr(round15_oracle, "word_pools", two_clips)
    assert a4.main([*choice_args(c), "--output-key", "c/choice.json"]) == 0
    index = part(c, "c/choice.json")
    assert index["kind"] == "comp-a4-clip-choice" and index["report_only"] is True and index["settings"]["provenance"]["code_digest"]
    assert index["settings"]["sentences"]["sha256"] == c["ssha"] and index["settings"]["picks"] == ["1", "4", "sample0", "draw1", "draw2", "draw3", "draw4"]
    rows = {r["entry"]: r for r in index["entries"]}
    assert rows["cut"]["label"] == "CUT" and rows["cut"]["scorer"] == "recogniser"
    for r in rows.values():
        assert r["chosen"]["sample0"]["sign"] == r["signs"][0]
        if r["scores"]:
            assert r["scores"][r["chosen"]["4"]["index"]] == max(r["scores"])    # the best, by the unrounded score
            assert r["chosen"]["1"]["index"] == 0
        else:
            assert r["chosen"]["4"]["index"] == 0                                 # no score: the fixed draw
    assert any(r["chosen"]["4"]["index"] != 0 for r in rows.values())             # the recogniser's choice moved off sample 0 somewhere
    csha = hashlib.sha256(c["store"].get("c/choice.json")).hexdigest()
    assert ce.main([*oracle_args(c), "--comp-a-read", "--output-key", "e/a1-shard0.json"]) == 0
    real_material = r2.oracle_material
    for pick in ("4", "sample0", "draw2"):
        seen = []

        def spy(job, clip, pools, **kw):
            seen.append((job["words"], [r["sign"] for r in kw["records"]]))
            return real_material(job, clip, pools, **kw)
        monkeypatch.setattr(r2, "oracle_material", spy)
        assert ce.main([*oracle_args(c, "--oracle-choice-key", "c/choice.json", "--oracle-choice-sha256", csha, "--oracle-choice-pick", pick),
                        "--output-key", f"e/a4-{pick}-shard0.json"]) == 0
        monkeypatch.setattr(r2, "oracle_material", real_material)
        assert seen
        for words, signs in seen:
            assert signs == [rows[w]["chosen"][pick]["sign"] for w in words]
        shard = part(c, f"e/a4-{pick}-shard0.json")
        assert shard["settings"]["round20"]["oracle_choice"] == dict(key="c/choice.json", sha256=csha, pick=pick)
    trimmed = dict(index, entries=[r for r in index["entries"] if r["entry"] != "cut"])
    raw = json.dumps(trimmed).encode()
    c["store"].put("c/short.json", raw)
    with pytest.raises(KeyError, match="cut"):
        ce.main([*oracle_args(c, "--oracle-choice-key", "c/short.json", "--oracle-choice-sha256", hashlib.sha256(raw).hexdigest(),
                              "--oracle-choice-pick", "4"), "--output-key", "e/x.json"])


def test_the_choice_is_write_once_and_d20_only(d20, recogniser, capsys):
    c = d20
    assert a4.main([*choice_args(c), "--output-key", "c/choice.json"]) == 0
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        a4.main([*choice_args(c), "--output-key", "c/choice.json"])
    args = choice_args(c)
    args[args.index(c["ssha"])] = "0" * 64
    with pytest.raises(SystemExit):
        a4.main([*args, "--output-key", "c/y.json"])
    assert "D20 only" in capsys.readouterr().err and not c["store"].exists("c/y.json")


def test_a_choice_refuses_a_forbidden_clip_at_run_time(d20, recogniser, monkeypatch):
    c = d20
    monkeypatch.setattr(lib, "ALLOWED_CORPORA", ("rit-text-r6",))     # the fixture's clips are ASLLVD's
    with pytest.raises(SystemExit, match="not a vocabulary source"):
        a4.main([*choice_args(c), "--output-key", "c/z.json"])
    assert not c["store"].exists("c/z.json")


def test_the_reader_runs_on_a_merged_arm_and_stays_a_read(d20):
    c = d20
    assert ce.main([*oracle_args(c, "--core-margin", "0", "--predicted-duration", "--arm-scale", "left=0.7,right=0.7", "--clips", "4"),
                    "--output-key", "e/arm-shard0.json"]) == 0
    merge(c["tmp"], "e/arm-shard0.json", "e/arm.json", "--report-only")
    root = ["--store", f"dir:{c['tmp'] / 'store'}"]
    assert rd.main(["read", "--test", "a1", "--arm-name", "both0.7", "--arm", "e/arm.json", "--smoke", "--bootstrap", "20", *root, "--output-key", "e/arm-read.json"]) == 0
    read = part(c, "e/arm-read.json")
    assert read["report_only"] is True and read["declared"] is False and read["main_form"] == "oracle_stage2@predicted" and read["role"] == "sensitivity"
    assert read["control"]["scales"] == dict(left=.7, right=.7) and read["arm"]["sha256"] and "draft_rule" not in read
    assert read["reads"]["oracle_stage2@predicted"]["blank"] == "blank@predicted"
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        rd.main(["read", "--test", "a1", "--arm-name", "both0.7", "--arm", "e/arm.json", "--smoke", *root, "--output-key", "e/arm-read.json"])
    assert rd.main(["gate", "--read", "e/arm-read.json", *root]) == 1
    stretched = part(c, "e/arm.json")
    stretched["settings"]["round20"]["vocabulary_extra"] = dict(top=10)
    c["store"].put("e/top.json", json.dumps(stretched).encode())
    assert rd.main(["read", "--test", "a3-oracle", "--arm-name", "top10", "--arm", "e/top.json", "--smoke", "--bootstrap", "20", *root, "--output-key", "e/o.json"]) == 0
    assert rd.main(["gate", "--read", "e/o.json", *root]) == 1            # a smoke never opens the gate


def test_the_default_run_carries_no_post_s2_trace(d20):
    c = d20
    assert ce.main([*oracle_args(c), "--output-key", "e/d-shard0.json"]) == 0
    text = c["store"].get("e/d-shard0.json").decode()
    for needle in ("arm_scale", "armscale", "vocabulary_extra", "word_cache_extra", "oracle_choice", "hold_left", "comp_a_motion", "comp_a_read", "provenance"):
        assert needle not in text
    merge(c["tmp"], "e/d-shard0.json", "e/d.json")                         # no --report-only needed: the old path
    assert "verdicts" in part(c, "e/d.json")


def test_same_codes_refuses_a_scaled_or_held_run():
    from scripts.round20_compare import GENERATION_ROUND20
    assert {"arm_scale", "hold_left", "vocabulary_extra", "word_cache_extra", "oracle_choice"} <= set(GENERATION_ROUND20)


# ---- the job on stubs: no real job runs -----------------------------------------------------------------------------------------

M0 = f"{OUT}/a1-margin0.json"
BASE = f"{OUT}/a3-oracle-base.json"
S0 = f"{OUT}/a4-clipchoice-sample0.json"
INDEX = f"{OUT}/a4-clip-choice.json"
CACHE3 = f"{OUT}/cache-a3-top30-nogains/index.json"
OWN = dict(MARGIN0_SHA="m0sha", BASE_SHA="basesha", SAMPLE0_SHA="s0sha", TOP10_READ_SHA="readsha")
READ10 = f"{OUT}/a3-oracle-top10-read.json"


def job(tmp_path, phase, *, arm=None, smoke=False, gpu=True, stored=(), declared="abc1234", own=True, **env):
    e = dict(COMP_PHASE=phase, STUB_EXISTS=" ".join(stored), STUB_GATE="1", STUB_FAIL_SHARD="", STUB_HANG="", **(OWN if own else {}), **env)
    if declared:
        e["COMP_A_DECLARED"] = declared
    if arm:
        e["ARM"] = arm
    return on_stubs(tmp_path, JOB.read_text(), e, smoke=smoke, gpu=gpu)


def one_done(out, code):
    lines = DONE.findall(out)
    assert len(lines) == 1 and lines[0][0] == str(code), out
    assert out.strip().splitlines()[-1].startswith("COMP-A-DONE")


def run_line(commands):
    return next(c for c in commands if "round15_compose_eval.py run" in c)


def read_line(commands):
    return next(c for c in commands if "comp_a_read.py read" in c)


def parsed(tmp_path, monkeypatch, line, name):
    seen = {}
    monkeypatch.setattr(ce, "run", lambda args: seen.update(vars(args)) or {})
    assert ce.main(line.split()[1:] + ["--store", f"dir:{tmp_path / ('s-' + name)}"]) == 0
    return seen


def test_the_job_parses():
    subprocess.run(["bash", "-n", str(JOB)], check=True)


@pytest.mark.parametrize("arm,scale,hold", [("margin0", None, False), ("hold", None, True), ("left0.5", "left=0.5,right=1", False), ("both0.7", "left=0.7,right=0.7", False)])
def test_a1_is_a9s_list_at_margin_zero_with_the_arms_change(tmp_path, monkeypatch, arm, scale, hold):
    code, out, commands = job(tmp_path, "a1", arm=arm, stored=[] if arm == "margin0" else [M0])
    assert code == 0
    one_done(out, 0)
    assert f"COMP-A-DONE 0 phase=a1 arm={arm} smoke=0" in out
    seen = parsed(tmp_path, monkeypatch, run_line(commands), arm)
    assert seen["sentences_key"] == lib.D20_KEY and seen["sentences_sha256"] == lib.D20_SHA and seen["clips"] == 500
    assert seen["core_margin"] == 0. and seen["predicted_duration"] and seen["expect_duration_model"] == "-0.083,0.3691,18289"
    assert seen["variants"] == "core_crossfade" and seen["word_cache_n"] == 16 and seen["word_cache_key"].endswith("cache-w14b-nogains-r10/index.json")
    assert seen["arm_scale"] == scale and bool(seen["hold_left_file"]) == hold and seen["likelihood"] and seen["recognizer_run"] == "sign-recognizer-v2"
    assert seen["comp_a_read"] == (arm == "margin0")
    assert seen["checkpoint_key"] is None and seen["no_decoder_gains"] is False            # no generation: the gains are the cache's
    done = scripts_run(commands)
    names = [c.split()[1] for c in done if "comp_a_declared" not in c]
    assert names == ["run", "merge", "read"]
    assert "--report-only" in next(c for c in done if " merge " in c)
    read = read_line(commands)
    assert "--declared-commit abc1234" in read and "--test a1" in read and "--smoke" not in read and f"--arm-name {arm}" in read
    assert f"{OUT}/a1-{arm}-read.json" in read.split("--output-key ")[1]
    a9 = f"--comparator a9_predicted={A}/a10-predicted-nogains-r10.json:core_crossfade@predicted:core_margin"
    if arm == "margin0":
        assert f"{a9} " in read and "margin0=" not in read
    else:
        first, second = read.index(f"--comparator margin0={M0}:core_crossfade@predicted:"), read.index(a9)
        assert first < second                                                               # margin0 first, a9 second
        declared = read.split("margin0=")[1].split(":")[2].split()[0]
        assert declared == ("hold_left" if hold else "arm_scale")
        assert read.split(a9)[1].split()[0] == ("+hold_left" if hold else "+arm_scale")


def test_the_hold_arm_is_pinned_to_its_statistics_and_the_a2_table(tmp_path, monkeypatch):
    _, _, commands = job(tmp_path, "a1", arm="hold", stored=[M0])
    seen = parsed(tmp_path, monkeypatch, run_line(commands), "hold2")
    assert seen["hold_left_file"] == "reports/comp_a_hold_stats.json"
    assert seen["hold_left_sha256"] == hashlib.sha256((ROOT / "reports/comp_a_hold_stats.json").read_bytes()).hexdigest()
    read = read_line(commands)
    assert "--a9-sides-file reports/comp_a2_left_arm.json" in read
    assert hashlib.sha256((ROOT / "reports/comp_a2_left_arm.json").read_bytes()).hexdigest() in read


PINS = [f"{A}/a10-predicted-nogains-r10.json", f"{A}/a10-predicted-nogains-r10-shard0.json", f"{A}/a1-oracle-r10.json", f"{A}/a1-oracle-r10-shard0.json",
        f"{A}/a4-w14b-n16-r10.json", f"{A}/a4-w14b-n16-r10-shard0.json"]


@pytest.mark.parametrize("phase,arm,stored,own_key", [
    ("a1", "margin0", [], None), ("a1", "hold", [M0], M0), ("a3-oracle", "base", [], None), ("a3-oracle", "top10", [BASE], BASE),
    ("a3-gen", "top10", [CACHE3, M0], M0), ("a4", "sample0", [INDEX], None), ("a4", "n16", [INDEX, S0], S0), ("a4", "draw1", [INDEX, S0], S0)])
def test_every_comparator_is_pinned_before_its_run(tmp_path, phase, arm, stored, own_key):
    code, _, commands = job(tmp_path, phase, arm=arm, stored=stored)
    assert code == 0
    run = next(i for i, c in enumerate(commands) if "round15_compose_eval.py run" in c)
    pins = [c for c in commands[:run] if c.startswith("pin ")]
    for key in PINS + [f"{A}/cache-w14b-nogains-r10/index.json", f"{O}/round20/d20.json"]:
        assert f"pin {key}" in pins, (phase, key)
    if own_key:
        assert f"pin {own_key}" in pins
    # a stored comparator's bad bytes stop this phase, whichever it is
    for key in PINS[:1] + ([own_key] if own_key else []):
        code, out, commands = job(tmp_path, phase, arm=arm, stored=stored, STUB_PIN_FAIL=key)
        assert code == 1 and f"ALERT {key} differs from its pin" in out and "round15_compose_eval.py run" not in "\n".join(commands)
        one_done(out, 1)


@pytest.mark.parametrize("phase,arm,stored,name", [("a1", "hold", [M0], "MARGIN0_SHA"), ("a3-oracle", "top10", [BASE], "BASE_SHA"),
                                                   ("a3-gen", "top10", [CACHE3, M0], "MARGIN0_SHA"), ("a4", "n16", [INDEX, S0], "SAMPLE0_SHA")])
def test_a_comparator_this_job_wrote_needs_its_hash(tmp_path, phase, arm, stored, name):
    env = {k: v for k, v in OWN.items() if k != name}
    code, out, commands = job(tmp_path, phase, arm=arm, stored=stored, own=False, **env)
    assert code == 1 and "give the hash of its comparator" in out and "round15_compose_eval.py run" not in "\n".join(commands)
    one_done(out, 1)
    # and it must be stored at all
    stored_less = [s for s in stored if s in (CACHE3, INDEX)]
    code, out, commands = job(tmp_path, phase, arm=arm, stored=stored_less)
    assert code == 1 and "comes first" in out and "round15_compose_eval.py run" not in "\n".join(commands)


@pytest.mark.parametrize("key", [f"{O}/round20/d20.json", f"{A}/cache-w14b-nogains-r10/index.json"])
def test_a_failed_input_pin_stops_the_job(tmp_path, key):
    code, out, commands = job(tmp_path, "a1", arm="margin0", STUB_PIN_FAIL=key)
    assert code == 1 and f"ALERT {key} differs from its pin" in out
    assert not [c for c in scripts_run(commands) if "comp_a_declared" not in c]
    one_done(out, 1)


def test_nothing_full_runs_undeclared(tmp_path):
    code, out, commands = job(tmp_path, "a1", arm="margin0", declared="")
    assert code == 1 and "nothing is declared yet" in out and not scripts_run(commands)
    one_done(out, 1)
    code, out, commands = job(tmp_path, "a1", arm="margin0", STUB_FAIL_MATCH="comp_a_declared.py")
    assert code == 1 and "does not declare a1" in out and len(scripts_run(commands)) == 1
    one_done(out, 1)
    # a smoke needs no declaration, and says it is one
    code, out, commands = job(tmp_path, "a1", arm="margin0", declared="", smoke=True, gpu=False)
    assert code == 0 and "COMP-A-DONE 0 phase=a1 arm=margin0 smoke=1" in out
    done = [c for c in scripts_run(commands)]
    assert "--clips 5 " in done[0] and "--smoke" in read_line(commands) and "--comparator" not in read_line(commands) and "--declared-commit" not in read_line(commands)
    assert f"{OUT}/smoke/" in done[0].split("--output-key ")[1]


@pytest.mark.parametrize("phase,arm,test,stored", [("a1", "margin0", "a1", []), ("a3-oracle", "base", "a3-oracle", []), ("a3-gen", "top10", "a3-gen", [CACHE3, M0]),
                                                   ("a3-cache", None, "a3-gen", [READ10]), ("a4-index", None, "a4", []), ("a4", "sample0", "a4", [INDEX])])
def test_each_phase_checks_its_own_rule_is_declared(tmp_path, phase, arm, test, stored):
    _, _, commands = job(tmp_path, phase, arm=arm, stored=stored)
    assert f"scripts/comp_a_declared.py abc1234 {test}" == scripts_run(commands)[0]


def test_a_failed_install_stops_the_job(tmp_path):
    text = JOB.read_text().replace("uv pip install -q", "false uv pip install -q", 1)
    code, out, commands = on_stubs(tmp_path, text, dict(COMP_PHASE="a1", ARM="margin0", COMP_A_DECLARED="abc1234", STUB_EXISTS="", STUB_GATE="1", STUB_FAIL_SHARD="", STUB_HANG=""),
                                   smoke=False, gpu=True)
    assert code == 1 and "uv pip install failed" in out and not scripts_run(commands)
    one_done(out, 1)


@pytest.mark.parametrize("sentence_set", ["c20", "val1000", "test", "D20"])
def test_no_other_set_can_be_asked_for(tmp_path, sentence_set):
    code, out, commands = job(tmp_path, "a1", arm="margin0", SET=sentence_set)
    assert code == 1 and "D20 only" in out and not scripts_run(commands)
    one_done(out, 1)


def test_the_job_names_no_other_set():
    text = "\n".join(x for x in JOB.read_text().splitlines() if not x.lstrip().startswith("#") and "SET:" not in x)
    assert "c20" not in text.lower() and "val1000" not in text and "--stage1-report-key" not in text
    assert text.count("--sentences-key") == 3 and "$D20" in text


@pytest.mark.parametrize("kwargs,message", [
    (dict(phase="a1"), "a1: ARM margin0"),
    (dict(phase="a1", arm="left0.3"), "a1: ARM margin0"),
    (dict(phase="a3-oracle", arm="top20"), "a3-oracle: ARM base"),
    (dict(phase="a3-gen", arm="base"), "a3-gen: ARM top10"),
    (dict(phase="a4", arm="n8", stored=[INDEX]), "a4: ARM sample0"),
    (dict(phase="a4-summary", smoke=True), "no smoke"),
    (dict(phase="c20"), "unknown COMP_PHASE"),
    (dict(phase="none"), "unknown COMP_PHASE"),
    (dict(phase="a1", arm="margin0", gpu=False), "needs a GPU"),
    (dict(phase="a4", arm="n16"), "the choice"),
    (dict(phase="a3-gen", arm="top10"), "is not stored"),
    (dict(phase="a1", arm="margin0", SMOKE="2"), "SMOKE must be"),
    (dict(phase="a3-gen", arm="top10", smoke=True), "give CACHE_KEY"),
    (dict(phase="a4", arm="n16", smoke=True), "give CHOICE_KEY"),
])
def test_refusals(tmp_path, kwargs, message):
    phase = kwargs.pop("phase")
    code, out, commands = job(tmp_path, phase, **kwargs)
    assert code == 1 and message in out and "ALERT" in out and "round15_compose_eval.py" not in "\n".join(commands)
    one_done(out, 1)


@pytest.mark.parametrize("match,message", [
    ("round15_compose_eval.py run", "ALERT run a1-margin0"),
    ("round15_compose_eval.py merge", "ALERT merge a1-margin0"),
    ("comp_a_read.py read", "ALERT read a1-margin0"),
])
def test_a_failed_step_stops_with_done_one(tmp_path, match, message):
    code, out, commands = job(tmp_path, "a1", arm="margin0", STUB_FAIL_MATCH=match)
    assert code == 1 and message in out
    one_done(out, 1)
    assert "COMP-A a1-margin0" not in out


@pytest.mark.parametrize("phase,arm,stored,line", [("a1", "margin0", [], "COMP-A a1-margin0"), ("a3-cache", None, [READ10], "COMP-A cache-a3"), ("a4-index", None, [], "COMP-A a4-clip-choice"),
                                                   ("a4", "sample0", [INDEX], "COMP-A a4-clipchoice"), ("a3-gen", "top10", [CACHE3, M0], "COMP-A a3-gen")])
def test_no_hash_is_no_success(tmp_path, phase, arm, stored, line):
    code, out, _ = job(tmp_path, phase, arm=arm, stored=stored, STUB_SHA_EMPTY="1")
    assert code == 1 and "no hash of" in out and line not in out
    one_done(out, 1)
    code, out, _ = job(tmp_path, phase, arm=arm, stored=stored)
    assert code == 0 and line in out and "stubsha" in out
    if phase in ("a1", "a4", "a3-gen"):
        assert "COMP-A-READ " in out                                       # the gate's read has a hash to pin


def test_an_unexpected_end_still_says_done_one(tmp_path):
    text = JOB.read_text().replace('case $PHASE in\n  a1)\n', 'exit 7\ncase $PHASE in\n  a1)\n', 1)
    assert "exit 7" in text
    code, out, _ = on_stubs(tmp_path, text, dict(COMP_PHASE="a1", ARM="margin0", COMP_A_DECLARED="x", STUB_EXISTS="", STUB_GATE="1", STUB_FAIL_SHARD="", STUB_HANG=""), smoke=False, gpu=True)
    assert code == 1 and "ALERT comp_a stopped unexpectedly, exit 7" in out
    one_done(out, 1)


def test_the_job_is_write_once(tmp_path):
    read = f"{OUT}/a1-margin0-read.json"
    code, out, commands = job(tmp_path, "a1", arm="margin0", stored=[read])
    assert code == 0 and "stored:" in out and [c for c in scripts_run(commands) if "comp_a_declared" not in c] == []
    code, out, commands = job(tmp_path, "a1", arm="margin0", stored=[M0])
    assert code == 0 and [c.split()[1] for c in scripts_run(commands) if "comp_a_declared" not in c] == ["read"]
    code, out, commands = job(tmp_path, "a1", arm="margin0", stored=[f"{OUT}/a1-margin0-shard0.json"])
    assert code == 0 and [c.split()[1] for c in scripts_run(commands) if "comp_a_declared" not in c] == ["merge", "read"]


def test_the_oracle_arms_keep_their_order_and_their_comparators(tmp_path, monkeypatch):
    code, _, commands = job(tmp_path, "a3-oracle", arm="base")
    assert code == 0
    seen = parsed(tmp_path, monkeypatch, run_line(commands), "base")
    assert seen["variants"] == "oracle_stage2" and seen["core_margin"] == 0. and seen["predicted_duration"] and not seen["vocabulary_extra_file"]
    assert seen["semlex_oracle_tracked"] and seen["comp_a_read"]
    assert f"--comparator a1={A}/a1-oracle-r10.json:oracle_stage2:core_margin" in read_line(commands)
    code, _, commands = job(tmp_path, "a3-oracle", arm="top30", stored=[BASE])
    assert code == 0
    seen = parsed(tmp_path, monkeypatch, run_line(commands), "top30")
    assert seen["vocabulary_extra_file"] == "reports/comp_a3_tokens.json" and seen["vocabulary_extra_top"] == 30
    assert seen["vocabulary_extra_sha256"] == hashlib.sha256(TOKENS.read_bytes()).hexdigest()
    read = read_line(commands)
    assert "--a9-sides-file reports/comp_a2_left_arm.json" in read
    assert read.index(f"--comparator base={BASE}:oracle_stage2@predicted:vocabulary_extra") < read.index(f"--comparator a1={A}/a1-oracle-r10.json:oracle_stage2:core_margin+vocabulary_extra")


def test_generation_waits_for_the_primary_oracle_and_generates_only_the_new_entries(tmp_path):
    code, out, commands = job(tmp_path, "a3-cache", stored=[READ10], STUB_FAIL_MATCH="comp_a_read.py gate")
    assert code == 1 and "did not pass its gate; no generation" in out
    assert not any("round20_word_cache.py" in c for c in commands)
    one_done(out, 1)
    code, out, commands = job(tmp_path, "a3-cache", stored=[READ10])
    assert code == 0 and f"pin {READ10}" in commands
    code2, out2, commands2 = job(tmp_path, "a3-cache", stored=[READ10], STUB_PIN_FAIL=READ10)
    assert code2 == 1 and "differs from its pin" in out2 and not any("round20_word_cache.py" in c for c in commands2)
    code3, out3, _ = job(tmp_path, "a3-cache", stored=[READ10], own=False)
    assert code3 == 1 and "give the hash of its comparator" in out3
    code, out, commands = job(tmp_path, "a3-cache", stored=[READ10])
    done = [c for c in scripts_run(commands) if "comp_a_declared" not in c]
    assert f"gate --read {OUT}/a3-oracle-top10-read.json" in done[0]
    run = next(c for c in done if "round20_word_cache.py run" in c)
    assert "--vocabulary-extra-top 30" in run and "--no-decoder-gains" in run and "--samples 16 --choose 16" in run and "--recognizer-run sign-recognizer-v3d" in run
    assert any("round20_word_cache.py merge" in c for c in done)
    assert "pin training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt" in commands


def test_the_generated_arm_joins_both_caches_and_reads_against_margin_zero(tmp_path, monkeypatch):
    code, _, commands = job(tmp_path, "a3-gen", arm="top10", stored=[CACHE3, M0])
    assert code == 0
    seen = parsed(tmp_path, monkeypatch, run_line(commands), "gen")
    assert seen["word_cache_extra_key"] == CACHE3 and seen["vocabulary_extra_top"] == 10 and seen["word_cache_n"] == 16
    assert seen["core_margin"] == 0. and seen["predicted_duration"] and seen["checkpoint_key"] is None
    assert f"--comparator margin0={M0}:core_crossfade@predicted:vocabulary_extra+word_cache_extra" in read_line(commands)
    assert "--a9-sides-file reports/comp_a2_left_arm.json" in read_line(commands)


def test_the_clip_choice_arms_are_a1s_list_with_their_pick(tmp_path, monkeypatch):
    code, _, commands = job(tmp_path, "a4-index")
    assert code == 0
    run = next(c for c in scripts_run(commands) if "comp_a_clip_choice.py" in c)
    assert "--samples 16 --choose 4,8,16" in run and "--recognizer-run sign-recognizer-v3d" in run and run.endswith(f"--output-key {INDEX}")
    gains = f"--comparator a4_generated={A}/a4-w14b-n16-r10.json:core_crossfade:word_cache+oracle_choice"
    a1 = f"--comparator a1={A}/a1-oracle-r10.json:oracle_stage2:oracle_choice"
    for arm, pick, stored in (("sample0", "sample0", [INDEX]), ("n16", "16", [INDEX, S0]), ("draw3", "draw3", [INDEX, S0])):
        code, _, commands = job(tmp_path, "a4", arm=arm, stored=stored)
        assert code == 0
        seen = parsed(tmp_path, monkeypatch, run_line(commands), f"a4{arm}")
        assert seen["oracle_choice_key"] == INDEX and seen["oracle_choice_pick"] == pick and seen["oracle_choice_sha256"] == "stubsha"
        assert seen["core_margin"] == .1 and not seen["predicted_duration"] and seen["variants"] == "oracle_stage2"
        read = read_line(commands)
        assert a1 in read and gains in read and "--test a4" in read and f"--arm-name {arm}" in read
        if arm == "sample0":
            assert "sample0=" not in read
        else:
            assert read.index(f"--comparator sample0={S0}:oracle_stage2:oracle_choice") < read.index(a1) < read.index(gains)


def test_the_decomposition_phase_needs_every_arm(tmp_path):
    every = [S0] + [f"{OUT}/a4-clipchoice-draw{k}.json" for k in range(1, 5)] + [f"{OUT}/a4-clipchoice-n16.json"]
    code, out, commands = job(tmp_path, "a4-summary", stored=every)
    assert code == 0
    run = next(c for c in scripts_run(commands) if "comp_a_read.py decompose" in c)
    assert "--declared-commit abc1234" in run and f"--chosen {OUT}/a4-clipchoice-n16.json" in run and f"--sample0 {S0}" in run and run.count("--draw ") == 4 and f"--a1 {A}/a1-oracle-r10.json" in run
    code, out, _ = job(tmp_path, "a4-summary", stored=every[:-1])
    assert code == 1 and "n16 is not stored" in out
    code, out, _ = job(tmp_path, "a4-summary", stored=every[:2] + every[3:])
    assert code == 1 and "is not stored" in out


def test_the_job_starts_no_pod():
    text = JOB.read_text()
    for needle in ("runpodctl", "podCreate", "run_job_pod", "create-pod"):
        assert needle not in text


def test_the_harness_runs_no_real_interpreter_and_hands_no_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "real-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "real-key")
    monkeypatch.setenv("RUNPOD_API_KEY", "real-key")
    script = "#!/usr/bin/env bash\nexport PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1\nenv | grep -c -E 'SECRET|AWS_|RUNPOD' ; echo PATH=$PATH\ncommand -v python\nexit 0\n"
    code, out, _ = on_stubs(tmp_path, script, dict(STUB_EXISTS="", STUB_GATE="1", STUB_FAIL_SHARD="", STUB_HANG=""), smoke=False, gpu=True)
    assert code == 0 and out.splitlines()[0] == "0"
    assert "/root/venv" not in out and str(tmp_path / "stubs") in out.splitlines()[2]


def test_the_scales_mean_is_over_valid_frames_only():
    """Masked frames are zeros by the channel's own rule, and a mean that counted them would pull every frame elsewhere."""
    from dataclasses import replace
    state = signer_state(frames=20, amplitude=.2)
    body = body_of(state)
    k = list(body.labels).index("LeftForearm")
    valid = body.valid.copy()
    valid[3:8, k] = False
    masked = replace(state, channels=dict(state.channels, body=replace(body, values=np.where(valid[..., None], body.values, 0.), valid=valid, observed=body.observed & valid,
                                                                       confidence=valid.astype(float))))
    out = cv2.arm_scale_state(masked, left=.5)
    kept = np.flatnonzero(valid[:, k])
    alone = cv2.arm_scale_state(comp.take_frames(state, kept), left=.5)
    np.testing.assert_allclose(body_of(out).values[kept, k], body_of(alone).values[:, k], atol=1e-5)


def test_the_forms_command_takes_declared_differences(tmp_path):
    from scripts.round20_compare import main as compare_main
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "store")
    a, b = arm_report(), arm_report(settings=dict(hold_left=dict(sha256="x")))
    store.put("a.json", json.dumps(a).encode())
    store.put("b.json", json.dumps(b).encode())
    root = ["forms", "--store", f"dir:{tmp_path / 'store'}", "--bootstrap", "10", "--a", f"b.json:{PREDICTED}", "--b", f"a.json:{PREDICTED}"]
    with pytest.raises(SystemExit, match="undeclared differences .*hold_left"):
        compare_main(root)
    assert compare_main([*root, "--declared-difference", "hold_left"]) == 0
    with pytest.raises(SystemExit, match="declared but absent"):
        compare_main([*root, "--declared-difference", "hold_left,core_margin"])
    assert compare_main([*root, "--declared-difference", "any"]) == 0


def test_masked_frames_cannot_move_the_scales_mean():
    """Why a mean over every frame is the same arithmetic as a mean over the valid ones here: the channel's own rule makes a masked frame's values zero,
    and zero matrices add nothing to the chordal mean. Its mutant is equivalent, so this states the reason; valid frames are still what the code selects."""
    from signtranslator.generation.composition_space import chordal_mean
    R = _matrices(signer_state(frames=10, amplitude=.3).channels["body"].values[:, 3].astype(float))
    np.testing.assert_allclose(chordal_mean(R), chordal_mean(np.concatenate([R, np.zeros((5, 3, 3))])))
    assert (_matrices(np.zeros((1, 6))) == 0).all()


def test_the_docstrings_drop_the_asymmetry_as_evidence():
    for path in ("scripts/comp_a_hold_stats.py", "scripts/comp_a2_left_arm.py"):
        doc = (ROOT / path).read_text().split('"""')[1]
        assert "asymmetry" not in doc and "-1.46" in doc and "-1.08" in doc and "0.306" in doc and "not match" in doc, path


def test_a_read_refuses_a_name_its_settings_do_not_bear():
    with pytest.raises(SystemExit, match="is not what this report"):
        rd.read_report("a1", arm_report(), arm_name="hold", boot=20, comparators=[("margin0", arm_report(), PREDICTED, None, ())], a9_sides=SIDES)
