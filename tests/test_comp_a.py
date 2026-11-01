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
import comp_a_real_choice as a4
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
    for token in ("so", "just", "as", "into", "would", "actually"):
        assert by[token]["status"] == "no_source" and by[token]["flags"]["no_well_tracked_source"] is True, token
    assert by["don't"]["status"] == "no_source" and by["don't"]["flags"]["negation"] is True
    assert by["it's"]["status"] == "alias_list" and by["it's"]["entry"] == "it"
    assert by["that's"]["status"] == "alias_r10" and by["that's"]["entry"] == "that"


def test_the_committed_arms():
    payload = committed()
    ten, thirty = lib.arm_of(payload, 10), lib.arm_of(payload, 30)
    assert set(ten["entries"]) <= set(thirty["entries"])
    assert set(ten["token_map"]) <= set(thirty["token_map"])
    assert "so" in ten["dropped"] and "just" in ten["dropped"]


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


def arm_report(*, clips=12, smoke=False, report_only=True, set_sha=None, key=None, words=None, control=False, extra=None):
    """A merged report's shape, on hand-made hypotheses."""
    refs = REFS[:clips]
    hyps = {"real": refs, "blank": ["some long words in it"] * clips, "blank@predicted": [" ".join(r.split()[:6]) for r in refs],
            PREDICTED: [r if k % 2 else "reference number has some long" for k, r in enumerate(refs)],
            "core_crossfade": [r if k % 3 else "words" for k, r in enumerate(refs)]}
    if control:
        hyps[lib.REAL_SCALED] = [r if k % 4 else "some long" for k, r in enumerate(refs)]
    hyps.update(extra or {})
    forms = {n: dict(bleu=round(corpus_bleu(h, refs)["bleu"], 3), ci95=[0., 9.], hypotheses=list(h)) for n, h in hyps.items()}
    ll = {n: [-3. + i * .01 - (.5 if n.startswith("blank") else 0.) for i in range(clips)] for n in hyps}
    settings = dict(variants=["core_crossfade"], seed=0, translator_key="t.pt", recognizer_run="sign-recognizer-v2",
                    round20=dict(sentences=dict(key=key or lib.D20_KEY, sha256=set_sha or lib.D20_SHA),
                                 **({"arm_scale": dict(left=.5, right=1.)} if control else {})))
    out = dict(clips=500 if not smoke and clips == 12 else clips, references=refs, signs=[f"s{k}" for k in range(clips)], forms=forms,
               settings=settings, per_clip=[dict(stage1_words=list(words or ["number", "reference", "zebra"])) for _ in refs],
               likelihood=dict(per_form=ll), motion_ratio_to_real={n: dict(wrist_speed=1.2, wrist_spread=1.5, hands_tracked=1.,
                                                                          seconds_over_reference=1.2) for n in hyps},
               spotting=dict(scores={PREDICTED: dict(recall=dict(mean=.3), order=dict(mean=.9), minus_mismatched_paired=None),
                                     **({lib.REAL_SCALED: dict(recall=dict(mean=.2), minus_real_paired=dict(difference=-.1))}
                                        if control else {})}), shards=[0], round20={})
    if report_only:
        out["report_only"] = True
    return out


@pytest.mark.parametrize("kwargs,message", [
    (dict(set_sha="0" * 64), "D20 only"),
    (dict(key="round20/c20.json"), "D20 only"),
    (dict(report_only=False), "not report-only"),
])
def test_the_reader_refuses(kwargs, message):
    with pytest.raises(SystemExit, match=message):
        rd.read_report("a1", arm_report(**kwargs), boot=20)


def test_the_reader_refuses_a_partial_read_and_a_big_smoke():
    partial = arm_report()
    partial["clips"] = 499
    with pytest.raises(SystemExit, match="not all of D20"):
        rd.read_report("a1", partial, boot=20)
    big = arm_report()
    with pytest.raises(SystemExit, match="at most 5"):
        rd.read_report("a1", big, boot=20, smoke=True)
    small = arm_report(clips=5, smoke=True)
    assert rd.read_report("a1", small, boot=20, smoke=True)["smoke"] is True
    with pytest.raises(SystemExit, match="no comparator"):
        rd.read_report("a1", small, boot=20, smoke=True, comparators=[("x", small, PREDICTED, None)])


def test_each_form_reports_everything():
    out = rd.read_report("a1", arm_report(), boot=30)
    read = out["reads"][PREDICTED]
    for key in ("bleu", "ci95", "gap_share", "likelihood_minus_blank", "likelihood_gap_share", "spotting", "motion_to_real", "length_ratio",
                "loop_share", "word_echo", "echo_lift_over_blank_relative_to_real"):
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
    # likelihood against the same blank: each form here sits .5 above its blank
    assert read["likelihood_minus_blank"]["difference"] == pytest.approx(.5)


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
    assert "loop_difference" in row and "echo_difference" in row
    moved = copy.deepcopy(other)
    moved["forms"]["real"]["hypotheses"][3] = "other"
    assert rd.comparison(arm, PREDICTED, moved, PREDICTED, boot=30)["instrument_check"]["moved"] is True
    unpaired = copy.deepcopy(other)
    unpaired["references"] = list(reversed(unpaired["references"]))
    with pytest.raises(SystemExit, match="not paired"):
        rd.comparison(arm, PREDICTED, unpaired, PREDICTED, boot=30)


def row_for_rule(bleu=.1, ll=.01, lower=-.01):
    return dict(bleu=dict(difference=bleu, ci95=[bleu - .1, bleu + .1]),
                likelihood_minus_blank=dict(difference=dict(difference=ll)),
                spotting_guard=dict(available=True, difference=dict(ci95=[lower, .1])))


def motion_report(**ratio):
    return dict(motion_ratio_to_real={PREDICTED: dict(dict(wrist_speed=1.2, wrist_spread=1.6, hands_tracked=1.), **ratio)})


CONTROL = dict(loss_share_of_gap=.05)


def rule(row=None, arm=None, other=None, control=CONTROL, test="a1"):
    return rd.draft_rule(test, comparison_row=row or row_for_rule(), arm=arm or motion_report(), form=PREDICTED,
                         other=other or motion_report(wrist_spread=1.9), other_form=PREDICTED, control=control)


def test_the_a1_draft_rule_passes_at_the_bars():
    out = rule()
    assert out["all_true"] is True and out["declared"] is False and "DRAFT" in out["note"]
    assert "adopted" not in json.dumps(out)
    assert out["checks"]["spread_at_least_0.8_of_comparator"] is True


@pytest.mark.parametrize("kwargs,check", [
    (dict(row=row_for_rule(bleu=0.)), "paired_bleu_above_0"),
    (dict(row=row_for_rule(bleu=-.1)), "paired_bleu_above_0"),
    (dict(row=row_for_rule(ll=0.)), "paired_likelihood_above_0"),
    (dict(row=row_for_rule(lower=-.02)), "spotting_lower_bound_above_minus_0.02"),
    (dict(arm=motion_report(wrist_speed=1.2501)), "wrist_at_most_1.25"),
    (dict(arm=motion_report(wrist_spread=1.51)), "spread_at_least_0.8_of_comparator"),
    (dict(control=dict(loss_share_of_gap=.1001)), "real_control_loses_at_most_10_points"),
])
def test_each_a1_check_can_fail_alone(kwargs, check):
    out = rule(**kwargs)
    assert out["checks"][check] is False and out["all_true"] is False
    assert [k for k, v in out["checks"].items() if not v] == [check]


def test_a_missing_number_fails_a_check():
    missing = dict(bleu=dict(difference=.1, ci95=[0, .2]), spotting_guard=dict(available=False))
    out = rule(row=missing)
    assert out["checks"]["paired_likelihood_above_0"] is False
    assert out["checks"]["spotting_lower_bound_above_minus_0.02"] is False


def test_the_exact_wrist_and_control_bars_pass():
    out = rule(arm=motion_report(wrist_speed=1.25), control=dict(loss_share_of_gap=.1))
    assert out["checks"]["wrist_at_most_1.25"] and out["checks"]["real_control_loses_at_most_10_points"]


def test_the_margin_zero_arm_has_no_control_check():
    assert "real_control_loses_at_most_10_points" not in rule(control=None)["checks"]


def test_the_other_tests_draft_rules():
    gen = rule(test="a3-gen", arm=motion_report(wrist_speed=1.5, hands_tracked=.9), other=motion_report(wrist_speed=1.4, hands_tracked=1.))
    assert gen["all_true"] is True and "wrist_at_most_1.25" not in gen["checks"]
    assert rule(test="a3-gen", arm=motion_report(wrist_speed=1.55), other=motion_report(wrist_speed=1.4))["checks"]["wrist_at_most_1.10_of_comparator"] is False
    assert rule(test="a3-gen", arm=motion_report(hands_tracked=.89))["checks"]["hands_at_least_0.9_of_comparator"] is False
    oracle = rule(test="a3-oracle")
    assert set(oracle["checks"]) == {"paired_bleu_above_0", "paired_likelihood_above_0"}
    four = rule(test="a4", row=row_for_rule(bleu=.05))
    assert four["checks"]["paired_bleu_lower_bound_above_0"] is False
    assert rule(test="a4", row=row_for_rule(bleu=.2))["checks"]["paired_bleu_lower_bound_above_0"] is True


def test_the_declared_commit_is_only_a_label():
    out = rd.draft_rule("a3-oracle", comparison_row=row_for_rule(), declared_commit="abc123")
    assert out["declared"] is True and out["declared_commit"] == "abc123" and "adopted" not in json.dumps(out)


def test_the_gate():
    read = dict(kind="comp-a-read", test="a3-oracle", smoke=False, draft_rule=dict(checks=dict(paired_bleu_above_0=True, paired_likelihood_above_0=True)))
    assert rd.oracle_gate(read) is True
    for broken in (dict(read, smoke=True), dict(read, test="a1"), dict(read, kind="other"),
                   dict(read, draft_rule=dict(checks=dict(paired_bleu_above_0=False, paired_likelihood_above_0=True))),
                   dict(read, draft_rule=dict(checks=dict(paired_bleu_above_0=True, paired_likelihood_above_0=False))),
                   {k: v for k, v in read.items() if k != "draft_rule"}):
        assert rd.oracle_gate(broken) is False


def test_the_control_read_is_in_points_of_reals_gap():
    report = arm_report(control=True)
    out = rd.control_read(report, boot=30)
    bleu = {n: corpus_bleu(report["forms"][n]["hypotheses"], REFS)["bleu"] for n in ("real", lib.REAL_SCALED, "blank")}
    assert out["loss_share_of_gap"] == pytest.approx((bleu["real"] - bleu[lib.REAL_SCALED]) / (bleu["real"] - bleu["blank"]), abs=1e-3)
    assert out["scales"] == dict(left=.5, right=1.) and out["likelihood_paired"]["difference"] == pytest.approx(0.)
    assert rd.control_read(arm_report(), boot=30) is None
    full = rd.read_report("a1", arm_report(control=True), boot=20, comparators=[("a9", arm_report(), PREDICTED, None)])
    assert full["control"]["loss_share_of_gap"] == out["loss_share_of_gap"]
    assert "real_control_loses_at_most_10_points" in full["draft_rule"]["checks"]


# ---- A2's arithmetic --------------------------------------------------------------------------------------------------------

def test_frame_labels_follow_the_plan():
    plan = dict(lead=2, cores=[3, 0, 2], gaps=[1, 0], trail=1)
    assert a2.frame_labels(plan).tolist() == [-2, -2, 0, 0, 0, -1, 2, 2, -2]


def test_steps_fall_into_regions():
    labels = np.array([-2, 0, 0, -1, -1, 1, 1, -2])
    assert a2.step_regions(labels, [True, False]) == ["lead_trail", "core_one_handed", "bridge", "bridge", "bridge", "core_two_handed",
                                                      "lead_trail"]
    # two cores with no gap between them: the join step is a bridge step, not a core step
    assert a2.step_regions(np.array([0, 0, 1, 1]), [True, True]) == ["core_one_handed", "bridge", "core_one_handed"]


def test_annotated_regions_take_the_cores_either_side_of_a_bridge():
    labels = np.array([-2, 0, 0, -1, 1, 1, -1, -1, 2, 2, -2])
    flags = [True, True, False]
    assert a2.annotated_step_regions(labels, flags) == [
        "lead_trail", "core_annotated_one", "bridge_one_to_one", "bridge_one_to_one", "core_annotated_one", "bridge_touching_one",
        "bridge_touching_one", "bridge_touching_one", "core_annotated_two", "lead_trail"]
    assert a2.annotated_step_regions(np.array([0, 0, -1, 1]), [None, False]) == ["core_unannotated", "bridge_other", "bridge_other"]
    assert a2.annotated_step_regions(np.array([0, 0, 1, 1]), [True, None]) == ["core_annotated_one", "bridge_touching_one", "core_unannotated"]


def test_the_annotated_labels_and_the_entry_majority():
    one, two = dict(gloss=dict(handshapes={"Dominant start handshape": "G"})), dict(gloss=dict(handshapes={"Non-dominant start handshape": "B"}))
    assert a2.annotated_label(one) is True and a2.annotated_label(two) is False and a2.annotated_label(dict()) is None
    pools = dict(a=[one, one, two], b=[two, one], c=[dict()], d=[one])
    assert a2.entry_annotations(pools) == dict(a=True, b=False, d=True)     # a tie is not a majority; no annotation, no label
    assert a2.annotated_join(True, True) == "one_to_one" and a2.annotated_join(True, False) == "touching_one"
    assert a2.annotated_join(False, False) == "two_to_two" and a2.annotated_join(None, True) == "unannotated"


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
    assert out["what_if"]["left"]["as_is"] == 2.5                         # 10 over 4 steps
    assert out["what_if"]["left"]["no_path_on_annotated_one_handed_cores"] == 1.
    assert out["what_if"]["left"]["no_path_in_one_to_one_bridges"] == 1.5
    assert out["what_if"]["left"]["no_path_in_any_bridge"] == 1.5


def test_one_handed_and_join_kinds():
    assert a2.is_one_handed(.34, 1.) is True and a2.is_one_handed(.35, 1.) is False
    assert a2.join_kind(True, False) == "one_to_two"
    assert a2.annotated_two_handed(dict(gloss=dict(handshapes={"Dominant start handshape": "G"}))) is False
    assert a2.annotated_two_handed(dict(gloss=dict(handshapes={"Non-dominant start handshape": "B-L"}))) is True
    assert a2.annotated_two_handed(dict(gloss=dict(label="x"))) is None and a2.annotated_two_handed(None) is None


def test_paths_by_region_and_what_if():
    speed = np.array([1., 3., 5., 7.])
    ok = np.array([True, True, True, False])
    regions = ["core_one_handed", "bridge", "bridge", "bridge"]
    paths = a2.path_by_region(dict(left=(speed, ok), right=(speed * 2, ok)), regions)
    assert paths["left"]["bridge"] == [8., 2] and paths["left"]["core_one_handed"] == [1., 1]
    assert paths["left"]["lead_trail"] == [0., 0]                          # the invalid step never counts
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
                frames_planned=10, native=dict(left=dict(speed=left_speed, tracked=1., position=list(position)),
                                               right=dict(speed=right_speed, tracked=1., position=[.2, .2])))


def test_the_tables_on_known_words():
    words = [word(), word(corpus="rit-text-r6", position=(0., 1.)), word(one=False, annotated=True, left=2., right=1.), word(one=True, annotated=True)]
    active = a2.left_active(words)
    assert active["asllvd-text-v1"] == dict(words=3, share=round(1 / 3, 4))
    check = a2.annotation_check(words)
    assert check["annotated"] == 4 and check["motion_label_agrees"] == .75
    rest = a2.rest_pose_table(words)
    assert rest["centre_distance"]["asllvd-text-v1|rit-text-r6"] == pytest.approx(2.)
    entries = a2.per_entry(words, real_left=.5)
    assert entries["by_corpus"]["asllvd-text-v1"]["occurrences"] == 2                 # one-handed only
    assert entries["by_corpus"]["asllvd-text-v1"]["left_native_over_real"] == 2.
    assert entries["by_corpus"]["asllvd-text-v1"]["squeeze"]["mean"] == 2.
    joins = a2.join_table([dict(kind="one_to_one", frames=4, same_corpus=False, corpora=["a", "b"], left=dict(travel=1., speed=2.), right=dict(travel=.5, speed=1.)),
                           dict(kind="one_to_one", frames=6, same_corpus=True, corpora=["a", "a"], left=dict(travel=.1, speed=.2), right=dict(travel=None, speed=None))])
    assert joins["one_to_one"]["joins"] == 2 and joins["one_to_one|cross_corpus"]["left_travel"]["mean"] == 1.
    assert joins["one_to_one|same_corpus"]["right_travel"] is None


def test_a2_is_d20_only():
    text = (ROOT / "scripts/comp_a2_left_arm.py").read_text()
    assert "require_d20(D20_KEY, D20_SHA)" in text and "check_read_records(val)" in text and "c20" not in text.lower()
    assert "load_translator" not in text and "load_generator" not in text          # no hypothesis, no generation


# ---- A4's choice ------------------------------------------------------------------------------------------------------------

def test_the_candidates_are_a_fixed_draw():
    pool = [dict(sign=f"s{k}") for k in range(40)]
    a = a4.candidates(pool, "cut", 16)
    assert len(a) == 16 and a == a4.candidates(pool, "cut", 16) and a != a4.candidates(pool, "onion", 16)
    assert len(a4.candidates(pool[:3], "cut", 16)) == 3


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
    payload = json.dumps(dict(kind="comp-a4-real-choice", settings=dict(mix_sha256=sha, corpora=corpora or [], samples=samples, choose=[4, 16],
                                                                        recognizer_run="sign-recognizer-v3d", sentences=dict(sha256=ssha)),
                              entries=rows, summary=dict(x=1))).encode()
    store.put("c/choice.json", payload)
    return hashlib.sha256(payload).hexdigest()


class Args:
    def __init__(self, sha, csha, ssha, n=16):
        self.oracle_choice_key, self.oracle_choice_sha256, self.oracle_choice_n, self.sentences_sha256, self.mix_sha256 = "c/choice.json", csha, n, ssha, sha


def test_a_choice_is_checked_before_it_serves(tmp_path):
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path)
    record = dict(sign="x", corpus="asllvd-text-v1", partition="train")
    pools = dict(cut=[record, dict(record, sign="y")])
    rows = [dict(entry="cut", chosen={"16": dict(sign="y", index=1), "4": dict(sign="x", index=0)})]
    csha = stored_choice(store, "m", "s", rows)
    got = a4.load_choice(store, Args("m", csha, "s"), pools, [])
    assert got["records"]["cut"]["sign"] == "y" and got["info"]["n"] == 16 and got["info"]["scorer"] == "sign-recognizer-v3d"
    assert a4.load_choice(store, Args("m", csha, "s", n=4), pools, [])["records"]["cut"]["sign"] == "x"
    with pytest.raises(SystemExit, match="does not match"):
        a4.load_choice(store, Args("m", "0" * 64, "s"), pools, [])
    with pytest.raises(SystemExit, match="not a choice on this sentence set"):
        a4.load_choice(store, Args("m", csha, "other"), pools, [])
    with pytest.raises(SystemExit, match="another vocabulary"):
        a4.load_choice(store, Args("other", csha, "s"), pools, [])
    with pytest.raises(SystemExit, match="another vocabulary"):
        a4.load_choice(store, Args("m", csha, "s"), pools, ["semlex"])
    with pytest.raises(SystemExit, match="best-of-8"):
        a4.load_choice(store, Args("m", csha, "s", n=8), pools, [])
    with pytest.raises(SystemExit, match="not in its pool"):
        a4.load_choice(store, Args("m", csha, "s"), dict(cut=[record]), [])
    bad = dict(cut=[dict(record, sign="y", corpus="asl-citizen-text-v1")])
    with pytest.raises(SystemExit, match="not a vocabulary source"):
        a4.load_choice(store, Args("m", csha, "s"), bad, [])


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
    assert ce.main([*plain, "--output-key", "e/plain-shard0.json"]) == 0
    assert ce.main([*plain, "--arm-scale", "left=0.5,right=1", "--output-key", "e/left-shard0.json"]) == 0
    a, b = part(c, "e/plain-shard0.json"), part(c, "e/left-shard0.json")
    assert b["settings"]["round20"]["arm_scale"] == dict(left=.5, right=1.) and "arm_scale" not in a["settings"]["round20"]
    moved = 0
    for x, y in zip(a["clips"], b["clips"]):
        assert x["plans"] == y["plans"]                                # the same words and frames
        assert x["motion"]["real"] == y["motion"]["real"]              # real itself is untouched
        assert x["hypotheses"]["blank"] == y["hypotheses"]["blank"]
        assert x["likelihood"]["real"] == y["likelihood"]["real"]
        if x["entries"]:
            moved += x["motion"]["oracle_stage2@predicted"]["wrist_speed"] != y["motion"]["oracle_stage2@predicted"]["wrist_speed"]
            for form in ("oracle_stage2", "oracle_stage2/shuffled", "oracle_stage2/mismatched_words"):
                assert x["motion"][form]["wrist_speed"] != y["motion"][form]["wrist_speed"], form
        # the control: the real clip with the same scales, moved, read with spotting beside real's
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
    assert m["motion_ratio_to_real"][lib.REAL_SCALED]["wrist_speed"] != 1.
    assert rd.control_read(m, boot=20)["scales"] == dict(left=.5, right=1.)


def test_a_post_s2_shard_merges_only_as_report_only(d20):
    c = d20
    assert ce.main([*oracle_args(c, "--arm-scale", "left=0.5,right=1"), "--output-key", "e/s-shard0.json"]) == 0
    with pytest.raises(SystemExit, match="report-only"):
        merge(c["tmp"], "e/s-shard0.json", "e/s.json")
    assert not c["store"].exists("e/s.json")


@pytest.mark.parametrize("flags", [["--arm-scale", "left=0.5,right=1"],
                                   ["--oracle-choice-key", "c/x.json", "--oracle-choice-sha256", "0" * 64, "--oracle-choice-n", "16"],
                                   ["--vocabulary-extra-file", str(TOKENS), "--vocabulary-extra-sha256", "0" * 64, "--vocabulary-extra-top", "10"]])
@pytest.mark.parametrize("sentences", [[], ["--sentences-key", "sets/c20.json", "--sentences-sha256", "0" * 64]])
def test_the_options_refuse_any_other_set_before_reading_it(d20, capsys, flags, sentences):
    c = d20
    args = [a for a in oracle_args(c)]
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
    (["--oracle-choice-n", "16"], "without the option"),
    (["--vocabulary-extra-file", "x.json"], "needs --vocabulary-extra-sha256"),
    (["--vocabulary-extra-file", "x.json", "--vocabulary-extra-sha256", "0", "--vocabulary-extra-top", "20"], "needs --vocabulary-extra-sha256"),
    (["--vocabulary-extra-file", "x.json", "--vocabulary-extra-sha256", "0", "--vocabulary-extra-top", "10", "--oracle-source", "continuous"], "its own segments"),
    (["--vocabulary-extra-file", "x.json", "--vocabulary-extra-sha256", "0", "--vocabulary-extra-top", "10", "--retime", "continuous"], "reference retiming"),
    (["--word-cache-extra-key", "c.json"], "needs --word-cache-extra-sha256"),
    (["--oracle-choice-key", "c.json"], "needs --oracle-choice-sha256"),
    (["--oracle-choice-key", "c.json", "--oracle-choice-sha256", "0", "--oracle-choice-n", "16", "--oracle-source", "continuous"], "alone, isolated"),
    (["--arm-scale", "left=.5,right=1", "--also-translator-run", "x"], "second translator"),
])
def test_the_options_refuse_what_does_not_apply(d20, capsys, flags, message):
    with pytest.raises(SystemExit):
        ce.main([*oracle_args(d20, *flags), "--output-key", "e/x.json"])
    assert message in capsys.readouterr().err


def test_generated_words_with_a_list_need_the_extra_cache(tmp_path, d20, capsys):
    c = d20
    args = [a for a in oracle_args(c)]
    args[args.index("oracle_stage2")] = "core_crossfade"
    with pytest.raises(SystemExit):
        ce.main([*args, "--checkpoint-key", "prior.pt", "--vocabulary-extra-file", str(TOKENS), "--vocabulary-extra-sha256", "0",
                 "--vocabulary-extra-top", "10", "--output-key", "e/x.json"])
    assert "no fresh generation" in capsys.readouterr().err


def write_list(c, rows):
    path = c["tmp"] / "tokens.json"
    path.write_text(json.dumps(dict(kind="comp-a3-tokens", inputs=dict(source_mix=dict(key="mix.json", sha256=c["sha"])), tokens=rows)))
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
    segment = dict(sign=f"{u['collection']}:u{k}#0", corpus=lib.SEGMENT_CORPUS, gloss="IX-2p", occurrence="IX-2p", partition="train",
                   record_sign=f"{u['collection']}:u{k}", start=int(a - u["frames"][0]), end=int(b - u["frames"][0]),
                   utterance_frames=int(u["frames"][1] - u["frames"][0] + 1), tracked=dict(arm=1., hand=1.), seconds=.3)
    clip = asllvd_clip(c["store"], c["sha"], "knife")
    source = dict(sign=clip["sign"], corpus=clip["corpus"], partition="train", signer_component=clip.get("signer_component"))
    return full_ranks(row(1, "you", kind="segment", seconds=.9, sources=[segment]), row(2, "half", seconds=.7, sources=[source]),
                      row(3, "this", "no_source"), row(11, "it", seconds=.6, sources=[source]))


def test_the_oracle_arm_adds_only_its_ranks_from_only_its_clips(d20, monkeypatch):
    c = d20
    path, sha = write_list(c, token_rows(c, monkeypatch))
    common = ("--core-margin", "0", "--predicted-duration")
    assert ce.main([*oracle_args(c, *common), "--output-key", "e/base-shard0.json"]) == 0
    assert ce.main([*oracle_args(c, *common, "--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top", "10"),
                    "--output-key", "e/top10-shard0.json"]) == 0
    base, top = part(c, "e/base-shard0.json"), part(c, "e/top10-shard0.json")
    assert top["settings"]["round20"]["vocabulary_extra"] == dict(sha256=sha, top=10) and "vocabulary_extra" not in base["settings"]["round20"]
    info = top["round20"]["vocabulary_extra"]
    assert info["new_entries"] == ["half", "you"] and "this" in info["unsigned_ranks"]
    assert info["tokens"].get("you", 0) >= 1 and info["tokens"].get("half", 0) >= 1 and "it" not in info["tokens"]   # rank 11 waits for the top 30
    for x, y in zip(base["clips"], top["clips"]):
        added = [w for w in y["entries"] if w in ("you", "half")]
        assert [w for w in y["entries"] if w not in ("you", "half")] == x["entries"]   # every r10 word, in order
        assert "this" not in y["entries"] and "it" not in y["entries"]
        if added:
            sources = y["plans"]["oracle_stage2"]["round20"]["sources"]
            assert dict(zip(y["entries"], sources)).get("you", "continuous") == "continuous"
        assert x["motion"]["real"] == y["motion"]["real"]
    assert any("you" in r["entries"] for r in top["clips"]) and any("half" in r["entries"] for r in top["clips"])
    assert top["round20"]["coverage"]["words_total"] > base["round20"]["coverage"]["words_total"]
    merge(c["tmp"], "e/top10-shard0.json", "e/top10.json", "--report-only")
    assert part(c, "e/top10.json")["report_only"] is True


def test_a_letter_of_the_vocabulary_is_never_served_as_the_lists_word(d20, monkeypatch):
    """An entry the list adds is served only from the list: with its segment unplaceable the run stops, it does not fall back."""
    c = d20
    rows = token_rows(c, monkeypatch)
    rows[0]["sources"][0]["utterance_frames"] = 9999
    path, sha = write_list(c, rows)
    with pytest.raises(KeyError, match="no training record"):
        ce.main([*oracle_args(c, "--core-margin", "0", "--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha,
                              "--vocabulary-extra-top", "10"), "--output-key", "e/x.json"])


def test_an_entry_the_vocabulary_also_holds_is_served_only_from_the_list(d20, monkeypatch):
    """The fixture's "knife" has a clip in the mix, as r10 has ASLLVD's letter "i". The list's own unplaceable segment stops the run:
    the mix's clip of the same spelling is never the fallback."""
    c = d20
    source = token_rows(c, monkeypatch)[0]["sources"][0]
    path, sha = write_list(c, full_ranks(row(1, "knife", kind="segment", seconds=.9, sources=[dict(source, utterance_frames=9999)])))
    with pytest.raises(KeyError, match="no training record"):
        ce.main([*oracle_args(c, "--core-margin", "0", "--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha,
                              "--vocabulary-extra-top", "10"), "--output-key", "e/x.json"])


def generated_args(c, *extra):
    args = oracle_args(c)
    args[args.index("oracle_stage2")] = "core_crossfade"
    return [*args, *extra]


def build_cache(c, prefix, *extra):
    common = ["--store", f"dir:{c['tmp'] / 'store'}", "--prefix", prefix]
    assert cw.main(["run", *common, "--mix-key", "mix.json", "--mix-sha256", c["sha"], "--sentences-key", "sets/d20.json",
                    "--sentences-sha256", c["ssha"], "--checkpoint-key", "prior.pt", "--samples", "4", "--choose", "4",
                    "--device", "cpu", "--no-decoder-gains", *extra]) == 0
    assert cw.main(["merge", *common]) == 0
    return hashlib.sha256(c["store"].get(f"{prefix}/index.json")).hexdigest()


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


def test_only_the_lists_new_entries_are_generated_and_the_composition_joins_both_caches(d20, monkeypatch, recogniser):
    c = d20
    path, sha = write_list(c, token_rows(c, monkeypatch))
    extra = ("--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top")
    main_sha = build_cache(c, "cache/main")
    extra_sha = build_cache(c, "cache/extra", *extra, "30")
    main_rows, extra_rows = part(c, "cache/main/index.json"), part(c, "cache/extra/index.json")
    assert sorted(r["entry"] for r in extra_rows["entries"]) == ["half", "it", "you"]
    assert extra_rows["settings"]["vocabulary_extra"] == dict(sha256=sha, top=30) and "vocabulary_extra" not in main_rows["settings"]
    assert not {r["entry"] for r in main_rows["entries"]} & {"half", "it", "you"}
    assert next(r for r in extra_rows["entries"] if r["entry"] == "you")["seconds"] == .9
    words = ("--word-cache-key", "cache/main/index.json", "--word-cache-sha256", main_sha, "--word-cache-n", "4", "--core-margin", "0",
             "--predicted-duration", "--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top", "30",
             "--word-cache-extra-key", "cache/extra/index.json", "--word-cache-extra-sha256", extra_sha)
    assert ce.main([*generated_args(c, *words), "--output-key", "e/gen-shard0.json"]) == 0
    shard = part(c, "e/gen-shard0.json")
    assert shard["settings"]["round20"]["word_cache_extra"]["sha256"] == extra_sha
    entries = [w for r in shard["clips"] for w in r["entries"]]
    assert {"you", "half", "it"} <= set(entries)
    reasons = {w: x for r in shard["clips"] for w, x in zip(r["entries"], r["plans"]["core_crossfade"]["round20"]["reasons"])}
    assert all(reasons[w].startswith("activity") for w in ("you", "half", "it"))     # the clip's own activity, no pooled fraction
    assert reasons["cut"] in ("entry", "all")
    # wrong hash, or a cache built on another list or decoder, never joins
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


def test_a_new_single_letter_word_is_never_scored_by_the_letters_label(d20, monkeypatch, recogniser):
    c = d20
    import round20_lib
    rows = token_rows(c, monkeypatch)
    # the fixture's texts hold no "i": the rule on the entry itself, through the same code path
    import round15_compose_eval as top                # the word cache's own import of it
    selection = top.stage1_selection

    def with_i(texts, vocab):
        return selection([t + " i" for t in texts], vocab)
    monkeypatch.setattr(top, "stage1_selection", with_i)
    rows = full_ranks(rows[0], rows[1], rows[2], row(4, "i", kind="segment", seconds=.5, sources=rows[0]["sources"]))
    path, sha = write_list(c, rows)
    build_cache(c, "cache/letter", "--vocabulary-extra-file", str(path), "--vocabulary-extra-sha256", sha, "--vocabulary-extra-top", "10")
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


def test_real_clips_are_chosen_by_the_recogniser_and_serve_the_oracle(d20, recogniser, monkeypatch):
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
    assert index["kind"] == "comp-a4-real-choice" and index["report_only"] is True
    assert index["settings"]["sentences"]["sha256"] == c["ssha"]
    rows = {r["entry"]: r for r in index["entries"]}
    assert "cut" in rows and rows["cut"]["label"] == "CUT" and rows["cut"]["scorer"] == "recogniser"
    assert "thank you" not in rows or rows["thank you"]["scorer"] == "none"
    for r in rows.values():
        if r["scores"]:
            assert r["scores"][r["chosen"]["4"]["index"]] == max(r["scores"])    # the best, by the unrounded score
            assert r["chosen"]["1"]["index"] == 0
        else:
            assert r["chosen"]["4"]["index"] == 0                                 # no score: the fixed draw
    # the oracle now serves exactly those clips, on every word of every form it draws
    csha = hashlib.sha256(c["store"].get("c/choice.json")).hexdigest()
    assert ce.main([*oracle_args(c, "--oracle-choice-key", "c/choice.json", "--oracle-choice-sha256", csha, "--oracle-choice-n", "4"),
                    "--output-key", "e/a4-shard0.json"]) == 0
    assert ce.main([*oracle_args(c), "--output-key", "e/a1-shard0.json"]) == 0
    shard, plain = part(c, "e/a4-shard0.json"), part(c, "e/a1-shard0.json")
    assert shard["settings"]["round20"]["oracle_choice"] == dict(key="c/choice.json", sha256=csha, n=4)
    assert shard["round20"]["oracle_choice"]["scorer"] == "sign-recognizer-v3d"
    for x, y in zip(shard["clips"], plain["clips"]):
        assert x["entries"] == y["entries"] and x["motion"]["real"] == y["motion"]["real"]
    real_material, seen = r2.oracle_material, []

    def spy(job, clip, pools, **kw):
        seen.append((job["words"], [r["sign"] for r in kw["records"]]))
        return real_material(job, clip, pools, **kw)
    monkeypatch.setattr(r2, "oracle_material", spy)
    assert ce.main([*oracle_args(c, "--oracle-choice-key", "c/choice.json", "--oracle-choice-sha256", csha, "--oracle-choice-n", "4"),
                    "--output-key", "e/a4b-shard0.json"]) == 0
    monkeypatch.setattr(r2, "oracle_material", real_material)
    assert len(seen) >= 4 and {w for words, _ in seen for w in words}
    for words, signs in seen:
        assert signs == [rows[w]["chosen"]["4"]["sign"] for w in words]
    assert sum(len(r["signs"]) > 1 for r in rows.values()) == len(rows)
    assert any(r["chosen"]["4"]["index"] != 0 for r in rows.values())     # the recogniser's choice moved off sample 0 somewhere
    # a word without a choice stops the run
    trimmed = dict(index, entries=[r for r in index["entries"] if r["entry"] != "cut"])
    raw = json.dumps(trimmed).encode()
    c["store"].put("c/short.json", raw)
    with pytest.raises(KeyError, match="cut"):
        ce.main([*oracle_args(c, "--oracle-choice-key", "c/short.json", "--oracle-choice-sha256", hashlib.sha256(raw).hexdigest(),
                              "--oracle-choice-n", "4"), "--output-key", "e/x.json"])


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
    monkeypatch.setattr(a4, "check_source_record", lib.check_source_record)
    with pytest.raises(SystemExit, match="not a vocabulary source"):
        a4.main([*choice_args(c), "--output-key", "c/z.json"])
    assert not c["store"].exists("c/z.json")


def test_the_reader_runs_on_a_merged_arm_and_stays_a_read(d20):
    c = d20
    assert ce.main([*oracle_args(c, "--core-margin", "0", "--predicted-duration", "--arm-scale", "left=0.7,right=0.7", "--clips", "4"),
                    "--output-key", "e/arm-shard0.json"]) == 0
    merge(c["tmp"], "e/arm-shard0.json", "e/arm.json", "--report-only")
    root = ["--store", f"dir:{c['tmp'] / 'store'}"]
    assert rd.main(["read", "--test", "a1", "--arm", "e/arm.json", "--smoke", "--bootstrap", "20", *root, "--output-key", "e/arm-read.json"]) == 0
    read = part(c, "e/arm-read.json")
    assert read["report_only"] is True and read["declared"] is False and read["main_form"] == "oracle_stage2@predicted"
    assert read["control"]["scales"] == dict(left=.7, right=.7) and read["arm"]["sha256"]
    assert read["reads"]["oracle_stage2@predicted"]["blank"] == "blank@predicted"
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        rd.main(["read", "--test", "a1", "--arm", "e/arm.json", "--smoke", *root, "--output-key", "e/arm-read.json"])
    assert rd.main(["gate", "--read", "e/arm-read.json", *root]) == 1
    assert rd.main(["read", "--test", "a3-oracle", "--arm", "e/arm.json", "--smoke", "--bootstrap", "20", *root, "--output-key", "e/o.json"]) == 0
    assert rd.main(["gate", "--read", "e/o.json", *root]) == 1            # a smoke never opens the gate


def test_the_default_run_carries_no_post_s2_trace(d20):
    c = d20
    assert ce.main([*oracle_args(c), "--output-key", "e/d-shard0.json"]) == 0
    text = c["store"].get("e/d-shard0.json").decode()
    for needle in ("arm_scale", "armscale", "vocabulary_extra", "word_cache_extra", "oracle_choice"):
        assert needle not in text
    assert not ce.comp_a_on(type("A", (), dict(arm_scale=None, vocabulary_extra_file=None, word_cache_extra_key=None, oracle_choice_key=None))())
    merge(c["tmp"], "e/d-shard0.json", "e/d.json")                         # no --report-only needed: the old path
    assert "verdicts" in part(c, "e/d.json")


def test_same_codes_refuses_a_scaled_run():
    from scripts.round20_compare import GENERATION_ROUND20
    assert {"arm_scale", "vocabulary_extra", "word_cache_extra", "oracle_choice"} <= set(GENERATION_ROUND20)


# ---- the job on stubs: no real job runs -----------------------------------------------------------------------------------------

def job(tmp_path, phase, *, arm=None, smoke=False, gpu=True, stored=(), declared="abc123", **env):
    e = dict(COMP_PHASE=phase, STUB_EXISTS=" ".join(stored), STUB_GATE="1", STUB_FAIL_SHARD="", STUB_HANG="", **env)
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


def parsed(tmp_path, monkeypatch, line, name):
    seen = {}
    monkeypatch.setattr(ce, "run", lambda args: seen.update(vars(args)) or {})
    assert ce.main(line.split()[1:] + ["--store", f"dir:{tmp_path / ('s-' + name)}"]) == 0
    return seen


def test_the_job_parses():
    subprocess.run(["bash", "-n", str(JOB)], check=True)


@pytest.mark.parametrize("arm,scale", [("margin0", None), ("left0.5", "left=0.5,right=1"), ("both0.7", "left=0.7,right=0.7")])
def test_a1_is_a9s_list_at_margin_zero_with_the_arms_scale(tmp_path, monkeypatch, arm, scale):
    code, out, commands = job(tmp_path, "a1", arm=arm)
    assert code == 0
    one_done(out, 0)
    assert f"COMP-A-DONE 0 phase=a1 arm={arm} smoke=0" in out
    run = run_line(commands)
    seen = parsed(tmp_path, monkeypatch, run, arm)
    assert seen["sentences_key"] == lib.D20_KEY and seen["sentences_sha256"] == lib.D20_SHA and seen["clips"] == 500
    assert seen["core_margin"] == 0. and seen["predicted_duration"] and seen["expect_duration_model"] == "-0.083,0.3691,18289"
    assert seen["variants"] == "core_crossfade" and seen["word_cache_n"] == 16 and seen["word_cache_key"].endswith("cache-w14b-nogains-r10/index.json")
    assert seen["arm_scale"] == scale and seen["likelihood"] and seen["recognizer_run"] == "sign-recognizer-v2"
    assert seen["checkpoint_key"] is None and seen["no_decoder_gains"] is False            # no generation: the gains are the cache's
    done = scripts_run(commands)
    assert ["round15_compose_eval.py run" in done[0], " merge " in done[1], "comp_a_read.py read" in done[2]] == [True] * 3
    assert "--report-only" in done[1]
    assert f"--comparator a9_predicted={A}/a10-predicted-nogains-r10.json:core_crossfade@predicted" in done[2]
    assert "--declared-commit abc123" in done[2] and "--test a1" in done[2] and "--smoke" not in done[2]
    assert f"{OUT}/a1-{arm}-read.json" in done[2].split("--output-key ")[1]


def test_every_stored_comparator_is_pinned_before_the_run(tmp_path):
    _, _, commands = job(tmp_path, "a1", arm="margin0")
    run = next(i for i, c in enumerate(commands) if "round15_compose_eval.py run" in c)
    pins = [c for c in commands[:run] if c.startswith("pin ")]
    for key in (f"{A}/a10-predicted-nogains-r10.json", f"{A}/a10-predicted-nogains-r10-shard0.json", f"{A}/a1-oracle-r10.json",
                f"{A}/a4-w14b-n16-r10-shard0.json", f"{A}/cache-w14b-nogains-r10/index.json", f"{O}/round20/d20.json"):
        assert f"pin {key}" in pins


@pytest.mark.parametrize("key", [f"{A}/a10-predicted-nogains-r10.json", f"{O}/round20/d20.json", f"{A}/cache-w14b-nogains-r10/index.json"])
def test_a_failed_pin_stops_the_job(tmp_path, key):
    code, out, commands = job(tmp_path, "a1", arm="margin0", STUB_PIN_FAIL=key)
    assert code == 1 and f"ALERT {key} differs from its pin" in out and not scripts_run(commands)
    one_done(out, 1)


def test_nothing_full_runs_undeclared(tmp_path):
    code, out, commands = job(tmp_path, "a1", arm="margin0", declared="")
    assert code == 1 and "nothing is declared yet" in out and not scripts_run(commands)
    one_done(out, 1)
    # a smoke needs no declaration, and says it is one
    code, out, commands = job(tmp_path, "a1", arm="margin0", declared="", smoke=True, gpu=False)
    assert code == 0 and "COMP-A-DONE 0 phase=a1 arm=margin0 smoke=1" in out
    done = scripts_run(commands)
    assert "--clips 5 " in done[0] and "--smoke" in done[2] and "--comparator" not in done[2] and "--declared-commit" not in done[2]
    assert f"{OUT}/smoke/" in done[0].split("--output-key ")[1]


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
    (dict(phase="c20"), "unknown COMP_PHASE"),
    (dict(phase="none"), "unknown COMP_PHASE"),
    (dict(phase="a1", arm="margin0", gpu=False), "needs a GPU"),
    (dict(phase="a4"), "the choice"),
    (dict(phase="a3-gen", arm="top10"), "is not stored"),
    (dict(phase="a3-oracle", arm="top10"), "comes first"),
    (dict(phase="a1", arm="margin0", SMOKE="2"), "SMOKE must be"),
    (dict(phase="a3-gen", arm="top10", smoke=True), "give CACHE_KEY"),
    (dict(phase="a4", smoke=True), "give CHOICE_KEY"),
])
def test_refusals(tmp_path, kwargs, message):
    phase = kwargs.pop("phase")
    code, out, commands = job(tmp_path, phase, **kwargs)
    assert code == 1 and message in out and "ALERT" in out and not scripts_run(commands)
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


def test_no_hash_is_no_success(tmp_path):
    code, out, _ = job(tmp_path, "a1", arm="margin0", STUB_SHA_EMPTY="1")
    assert code == 1 and "no hash of" in out
    one_done(out, 1)


def test_an_unexpected_end_still_says_done_one(tmp_path):
    text = JOB.read_text().replace('case $PHASE in\n  a1)', 'exit 7\ncase $PHASE in\n  a1)', 1)
    code, out, _ = on_stubs(tmp_path, text, dict(COMP_PHASE="a1", ARM="margin0", COMP_A_DECLARED="x", STUB_EXISTS="", STUB_GATE="1",
                                                 STUB_FAIL_SHARD="", STUB_HANG=""), smoke=False, gpu=True)
    assert code == 1 and "ALERT comp_a stopped unexpectedly, exit 7" in out
    one_done(out, 1)


def test_the_job_is_write_once(tmp_path):
    read = f"{OUT}/a1-margin0-read.json"
    code, out, commands = job(tmp_path, "a1", arm="margin0", stored=[read])
    assert code == 0 and "stored:" in out and not scripts_run(commands)
    # a stored report is not recomputed: only the read runs
    code, out, commands = job(tmp_path, "a1", arm="margin0", stored=[f"{OUT}/a1-margin0.json"])
    assert code == 0 and [c.split()[1] for c in scripts_run(commands)] == ["read"]
    # a stored shard is merged, never rerun
    code, out, commands = job(tmp_path, "a1", arm="margin0", stored=[f"{OUT}/a1-margin0-shard0.json"])
    assert code == 0 and [c.split()[1] for c in scripts_run(commands)] == ["merge", "read"]


def test_the_oracle_arms_keep_their_order_and_their_comparators(tmp_path, monkeypatch):
    code, _, commands = job(tmp_path, "a3-oracle", arm="base")
    assert code == 0
    seen = parsed(tmp_path, monkeypatch, run_line(commands), "base")
    assert seen["variants"] == "oracle_stage2" and seen["core_margin"] == 0. and seen["predicted_duration"] and not seen["vocabulary_extra_file"]
    assert seen["semlex_oracle_tracked"]
    code, _, commands = job(tmp_path, "a3-oracle", arm="top30", stored=[f"{OUT}/a3-oracle-base.json"])
    assert code == 0
    seen = parsed(tmp_path, monkeypatch, run_line(commands), "top30")
    assert seen["vocabulary_extra_file"] == "reports/comp_a3_tokens.json" and seen["vocabulary_extra_top"] == 30
    assert seen["vocabulary_extra_sha256"] == hashlib.sha256(TOKENS.read_bytes()).hexdigest()
    read = scripts_run(commands)[-1]
    assert read.index(f"--comparator base={OUT}/a3-oracle-base.json:oracle_stage2@predicted") < read.index(f"--comparator a1={A}/a1-oracle-r10.json:oracle_stage2")


def test_generation_waits_for_the_oracle_and_generates_only_the_new_entries(tmp_path):
    code, out, commands = job(tmp_path, "a3-cache", STUB_FAIL_MATCH="comp_a_read.py gate")
    assert code == 1 and "did not pass its gate; no generation" in out
    assert not any("round20_word_cache.py" in c for c in commands)
    one_done(out, 1)
    code, out, commands = job(tmp_path, "a3-cache")
    assert code == 0
    done = scripts_run(commands)
    assert f"gate --read {OUT}/a3-oracle-top30-read.json" in done[0]
    run = next(c for c in done if "round20_word_cache.py run" in c)
    assert "--vocabulary-extra-top 30" in run and "--no-decoder-gains" in run and "--samples 16 --choose 16" in run and "--recognizer-run sign-recognizer-v3d" in run
    assert any("round20_word_cache.py merge" in c for c in done)
    assert f"pin training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt" in commands


def test_the_generated_arm_joins_both_caches_and_reads_against_margin_zero(tmp_path, monkeypatch):
    stored = [f"{OUT}/cache-a3-top30-nogains/index.json", f"{OUT}/a1-margin0.json"]
    code, _, commands = job(tmp_path, "a3-gen", arm="top10", stored=stored)
    assert code == 0
    seen = parsed(tmp_path, monkeypatch, run_line(commands), "gen")
    assert seen["word_cache_extra_key"] == stored[0] and seen["vocabulary_extra_top"] == 10 and seen["word_cache_n"] == 16
    assert seen["core_margin"] == 0. and seen["predicted_duration"] and seen["checkpoint_key"] is None
    assert f"--comparator margin0={OUT}/a1-margin0.json:core_crossfade@predicted" in scripts_run(commands)[-1]


def test_a4_is_a1s_list_with_the_choice(tmp_path, monkeypatch):
    code, _, commands = job(tmp_path, "a4-choice")
    assert code == 0
    run = scripts_run(commands)[0]
    assert "comp_a_real_choice.py" in run and "--samples 16 --choose 4,8,16" in run and "--recognizer-run sign-recognizer-v3d" in run
    code, _, commands = job(tmp_path, "a4", stored=[f"{OUT}/a4-real-choice.json"])
    assert code == 0
    seen = parsed(tmp_path, monkeypatch, run_line(commands), "a4")
    assert seen["oracle_choice_key"] == f"{OUT}/a4-real-choice.json" and seen["oracle_choice_n"] == 16 and seen["oracle_choice_sha256"] == "stubsha"
    assert seen["core_margin"] == .1 and not seen["predicted_duration"] and seen["variants"] == "oracle_stage2"
    read = scripts_run(commands)[-1]
    assert f"--comparator a1={A}/a1-oracle-r10.json:oracle_stage2" in read and "--test a4" in read


def test_the_job_starts_no_pod():
    text = JOB.read_text()
    for needle in ("runpodctl", "podCreate", "run_job_pod", "create-pod"):
        assert needle not in text
