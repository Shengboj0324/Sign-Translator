"""Round 20 tooling, unit level: sentence sets (exclusions, determinism, disjointness), the Sem-Lex activity core,
continuous-pace durations, velocity-bounded bridges and the continuous plan, the Butterworth low-pass, the predicted
duration model, ASLLRP continuous segments, the word cache's helpers, paired statistics and the reference
log-likelihood under a (tiny) back-translator."""
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from signtranslator.generation import composition_v2 as cv2
from signtranslator.generation import compositional as comp
from signtranslator.generation.composition_space import wrist_from_shoulder
from signtranslator.pose.multichannel import CHANNELS
import scripts.round20_lib as lib
from tests.test_generation_signing_space import signer_state


def h2s(k, text, *, component="c01", partition="val"):
    return dict(sign=f"v{k}", corpus=lib.H2S, partition=partition, signer_component=component, text=text,
                continuous=dict(status="converted", state_key=f"s/{k}.npz", state_sha256=f"{k:064x}"))


# ---- sets -----------------------------------------------------------------------------------------------------------

def test_the_pool_is_validation_sentences_candidates():
    from scripts.pilot_content_check import validation_sentences
    records = [h2s(k, f"text {k}") for k in range(40)] + [h2s(99, "x", component="c00"), h2s(98, "y", partition="train"),
                                                          dict(h2s(97, ""), text="")]
    manifest = dict(records=records)
    assert lib.validation_pool(manifest) == records[:40]
    assert validation_sentences(manifest, 40) == random.Random(0).sample(lib.validation_pool(manifest), 40)


def test_draw_sets_excludes_clips_and_texts_and_is_deterministic():
    pool = [h2s(k, f"Sentence number {k % 60}!") for k in range(200)]       # texts repeat every 60 clips
    excluded_signs = {f"v{k}" for k in range(10)}
    excluded_texts = {lib.text_key("sentence NUMBER 5")}
    d, c, counts = lib.draw_sets(pool, excluded_signs, excluded_texts, dev=40, confirm=60)
    keys = lambda rs: {lib.text_key(r["text"]) for r in rs}                  # noqa: E731
    assert not ({r["sign"] for r in d + c} & excluded_signs)
    assert lib.text_key("Sentence number 5") not in keys(d) | keys(c)
    assert not ({r["sign"] for r in d} & {r["sign"] for r in c}) and not (keys(d) & keys(c))
    remainder = [r for r in pool if r["sign"] not in excluded_signs and lib.text_key(r["text"]) not in excluded_texts]
    rng = random.Random(20)
    assert d == rng.sample(remainder, 40)                                    # D: Random(20).sample of the remainder
    assert counts["remainder"] == len(remainder) == 200 - 10 - 3 and counts["dev"] == 40 and counts["confirm"] == 60
    d2, c2, _ = lib.draw_sets(pool, excluded_signs, excluded_texts, dev=40, confirm=60)
    assert d2 == d and c2 == c
    with pytest.raises(ValueError, match="confirmation"):
        lib.draw_sets(pool, excluded_signs, excluded_texts, dev=40, confirm=500)
    with pytest.raises(ValueError, match="unique"):
        lib.draw_sets(pool + pool[:1], set(), set(), dev=1, confirm=1)


def test_a_stored_set_loads_only_on_its_own_manifest():
    records = [h2s(k, f"text {k}") for k in range(6)]
    manifest = dict(records=records)
    payload = lib.set_payload("D20", records[3:0:-1], manifest_key="m", manifest_sha256="a" * 64, rule="r", counts={})
    assert [r["sign"] for r in lib.load_sentence_set(payload, manifest, "m", "a" * 64)] == ["v3", "v2", "v1"]
    with pytest.raises(SystemExit, match="another manifest"):
        lib.load_sentence_set(payload, manifest, "m", "b" * 64)
    bad = dict(payload, records=[dict(payload["records"][0], text="other")])
    with pytest.raises(SystemExit, match="not this manifest"):
        lib.load_sentence_set(bad, manifest, "m", "a" * 64)


def test_build_sets_excludes_the_thousand_and_the_three_hundred(tmp_path):
    import hashlib
    import json
    from scripts.build_round20_sets import main
    from scripts.pilot_content_check import validation_sentences
    from signtranslator.data.text_motion_dataset import DirectoryStore
    from signtranslator.data_engineering.corpus_manifest import canonical_json
    from signtranslator.data_engineering.corpus_manifest import KINDS, usable_channels
    records = [h2s(k, f"line {k % 3000}") for k in range(3096)]           # some texts repeat
    for r in records:
        r["usable"] = {kind: list(usable_channels(r, kind)) for kind in KINDS}
    store = DirectoryStore(tmp_path / "s")
    payload = canonical_json(dict(records=records))
    store.put("mix.json", payload)
    sha = hashlib.sha256(payload).hexdigest()
    argv = ["--store", f"dir:{tmp_path / 's'}", "--mix-key", "mix.json", "--mix-sha256", sha, "--prefix", "r20"]
    assert main(argv) == 0
    d, c = (json.loads(store.get(f"r20/{n}.json")) for n in ("d20", "c20"))
    manifest = dict(records=records)
    thousand = validation_sentences(manifest, 1000)
    banned = {r["sign"] for r in thousand} | {r["sign"] for r in validation_sentences(manifest, 300)}
    texts = {lib.text_key(r["text"]) for r in thousand}
    assert d["sentences"] == 500 and c["sentences"] == 1000
    for s in (d, c):
        assert not ({r["sign"] for r in s["records"]} & banned)
        assert not ({lib.text_key(r["text"]) for r in s["records"]} & texts)
    assert not ({r["sign"] for r in d["records"]} & {r["sign"] for r in c["records"]})
    assert lib.load_sentence_set(d, manifest, "mix.json", sha)
    with pytest.raises(SystemExit, match="written once"):
        main(argv)


# ---- vocabulary -------------------------------------------------------------------------------------------------------

def test_semlex_only_entries_and_coverage():
    pools = {"go": [dict(corpus="semlex-text-v2")], "cut": [dict(corpus="semlex-text-v1"), dict(corpus="asllvd-text-v1")],
             "stir": [dict(corpus="semlex-text-v1")], "x": []}
    assert lib.semlex_only_entries(pools) == ["go", "stir"]
    vocab = {"cut": 1., "onion": 1., "thank you": 1.}
    details = [comp.selection_details(t, vocab) for t in ("Cut the onions now", "thank you, zebra")]
    cov = lib.coverage(details)
    assert cov["words_total"] == 3 and cov["entries_per_sentence"]["mean"] == 1.5
    assert dict(cov["top_unmatched"]) == {"now": 1, "zebra": 1} and dict(cov["top_stop"]) == {"the": 1}
    assert sum(cov["token_share"].values()) == pytest.approx(1., abs=1e-3)


def test_the_activity_core_replaces_the_pooled_fraction_only_for_listed_entries():
    import scripts.round15_stage2 as r2
    state = signer_state(3, frames=40, amplitude=.5)
    base = dict(entries={"onion": (.3, .6)}, all=(.275, .65), clips=3, records={})
    plain = r2.sign_core("cut", state, fractions=base, margin=.1)
    assert plain["reason"] == "all"
    act = dict(base, activity=frozenset({"cut", "onion"}))
    w = r2.sign_core("cut", state, fractions=act, margin=.1)
    expect = comp.core_window(state, dict(comp.CORE_RULE, margin=0.))
    assert (w["lo"], w["hi"]) == (expect["lo"], expect["hi"]) and w["reason"].startswith("activity")
    assert r2.sign_core("onion", state, fractions=act, margin=.1) == r2.sign_core("onion", state, fractions=base,
                                                                                  margin=.1)   # an ASLLVD median wins
    assert r2.sign_core("salt", state, fractions=act, margin=.1)["reason"] == "all"


# ---- durations, bridges, plans ------------------------------------------------------------------------------------------

def test_continuous_seconds():
    timing = dict(sentence_seconds={"cut": .25}, annotated={"onion": .6}, trimmed_over_annotated=2.,
                  default_seconds=.23, floor=.17)
    assert cv2.continuous_seconds("cut", timing, {"cut": 1.}) == (.25, "asllrp")
    assert cv2.continuous_seconds("onion", timing, {"onion": 1.}) == (pytest.approx(.385 * .6), "isolated_annotated")
    assert cv2.continuous_seconds("go", timing, {"go": 1.2}, ratio=.5) == (pytest.approx(.3), "isolated_trimmed")
    plain = dict(timing, trimmed_over_annotated=None)                          # no ratio: the sentence-sign default
    assert cv2.continuous_seconds("go", plain, {"go": 1.2}) == (.23, "default")   # not shrunk twice


def test_bridge_frames_bound_the_bridge_speed():
    a, b = signer_state(1, frames=10, amplitude=.4), signer_state(2, frames=10, amplitude=.4)
    frames, info = cv2.bridge_frames(a, b, 2, min_speed=.3)
    expect = 0
    for side in ("Left", "Right"):
        pa, _ = wrist_from_shoulder(a, side)
        pb, _ = wrist_from_shoulder(b, side)
        d = np.linalg.norm(pa[-1] - pb[0])
        va = np.linalg.norm(np.diff(pa[-4:], axis=0), axis=-1).mean() * 24
        vb = np.linalg.norm(np.diff(pb[:4], axis=0), axis=-1).mean() * 24
        bound = max((va + vb) / 2, .3)
        n = max(int(np.ceil(d * 24 / bound - 1 - 1e-9)), 0)
        assert info["sides"][side]["frames"] == n
        assert d / (n + 1) * 24 <= bound + 1e-9                               # a linear bridge of n frames is slow enough
        expect = max(expect, n)
    assert frames == max(2, expect) and info["needed"] == expect
    assert cv2.bridge_frames(a, b, 2, min_speed=.3, max_frames=3)[0] == min(max(2, expect), 3)
    assert cv2.bridge_frames(a, a, 5, min_speed=.3, max_frames=1)[0] == 5     # never below the base gap
    still = signer_state(4, frames=6, amplitude=0.)
    slow, info = cv2.bridge_frames(still, b, 0, min_speed=10.)
    assert info["sides"]["Right"]["bound"] == 10.                              # the floor on a still boundary


def test_the_continuous_plan_fills_the_reference_and_renders():
    states = [signer_state(k, frames=14, amplitude=.4) for k in range(3)]
    pieces = [dict(state=s, lo=3, hi=10) for s in states]
    parts = comp.parts_from_pieces(pieces)
    plan = cv2.continuous_plan(parts["cores"], [.25, .3, .2], 4., gap_seconds=.1667, tempo=1.25, lead_share=.44)
    assert plan["cores"] == [round(.25 * 1.25 * 24), round(.3 * 1.25 * 24), round(.2 * 1.25 * 24)]
    assert plan["base_gap"] == round(.1667 * 1.25 * 24) and all(g >= plan["base_gap"] for g in plan["gaps"])
    assert plan["fit"] == "padded" and plan["frames"] == 96 and plan["lead"] == round((96 - plan["body_planned"]) * .44)
    assert len(comp.render(parts, plan).timestamps) == 96
    short = cv2.continuous_plan(parts["cores"], [.25, .3, .2], .5, gap_seconds=.1667)
    assert short["fit"] == "compressed" and short["frames"] == 12 and min(short["cores"]) >= 1
    assert len(comp.render(parts, short).timestamps) == 12
    free = cv2.continuous_plan(parts["cores"], [.25, .3, .2], .5, gap_seconds=.1667, total_fit="none", velocity=False)
    assert free["fit"] == "predicted" and free["frames"] == free["body_planned"] and free["gaps"] == [4, 4]
    one = cv2.continuous_plan(parts["cores"][:1], [.2], .1, gap_seconds=.1)
    assert one["frames"] == 2 and one["fit"] == "compressed"
    with pytest.raises(ValueError):
        cv2.continuous_plan(parts["cores"], [.2], 1., gap_seconds=.1)


# ---- low-pass -------------------------------------------------------------------------------------------------------

def sine_state(frames=96):
    s = signer_state(5, frames=frames, amplitude=.3)
    t = np.arange(frames) / 24.
    c = s.channels["root_translation"]
    values = np.stack([np.sin(2 * np.pi * 1. * t), np.sin(2 * np.pi * 10. * t), np.zeros(frames)], -1)[:, None]
    valid = np.ones((frames, 1), bool)
    valid[40:46] = False
    values = np.where(valid[..., None], values, 0.)
    return replace(s, channels=dict(s.channels, root_translation=replace(c, values=values, valid=valid, inference_method="estimate",
                                                                           confidence=valid.astype(float))))


def test_the_lowpass_keeps_slow_motion_removes_fast_and_respects_masks():
    s = sine_state()
    out = cv2.lowpass_state(s, 6.)
    v, w = s.channels["root_translation"].values[:, 0], out.channels["root_translation"].values[:, 0]
    keep = np.r_[10:30, 56:86]                                                 # away from the runs' ends
    assert np.abs(w[keep, 0] - v[keep, 0]).max() < .05                        # 1 Hz kept (zero phase: no lag)
    assert np.abs(w[keep, 1]).max() < .1 * np.abs(v[keep, 1]).max()           # 10 Hz removed
    assert np.all(w[40:46] == 0.)                                              # invalid frames untouched
    body = out.channels["body"].values
    a, b = body[..., :3], body[..., 3:]
    assert np.allclose(np.linalg.norm(a, axis=-1), 1.) and np.allclose(np.linalg.norm(b, axis=-1), 1.)
    assert np.allclose((a * b).sum(-1), 0., atol=1e-9)
    for name in s.channels:
        assert np.array_equal(out.channels[name].valid, s.channels[name].valid)
    assert np.array_equal(out.timestamps, s.timestamps)
    for kind in ("unit_vector", "bounded_scalar"):
        assert any(k == kind for k, _, _ in CHANNELS.values())
    with pytest.raises(ValueError):
        cv2.lowpass_state(s, 12.)
    with pytest.raises(ValueError, match="24 fps"):                         # a 30 fps clock is refused
        cv2.lowpass_state(replace(s, timestamps=np.arange(len(s.timestamps)) / 30.), 6.)
    from scipy.signal import sosfreqz, butter                                  # the documented response
    _, h = sosfreqz(butter(4, 6., fs=24., output="sos"), worN=[6.], fs=24.)
    assert abs(h[0]) ** 2 == pytest.approx(.5, abs=1e-6)                      # -6 dB at the cutoff after filtfilt


def test_lowpass_short_runs_are_untouched():
    vals = np.arange(10, dtype=float).reshape(10, 1, 1)
    valid = np.zeros((10, 1), bool)
    valid[3] = True
    assert np.array_equal(cv2.lowpass_values(vals, valid, 6.), vals)


# ---- predicted duration ----------------------------------------------------------------------------------------------

def test_the_duration_model_is_least_squares_on_tokens():
    records = [dict(corpus=cv2.H2S, partition="train", text=" ".join(["w"] * n), continuous=dict(seconds=1. + .4 * n))
               for n in range(1, 12)] + [dict(corpus=cv2.H2S, partition="val", text="a b", continuous=dict(seconds=99.))]
    model = cv2.fit_duration_model(dict(records=records))
    assert model["records"] == 11 and model["slope"] == pytest.approx(.4) and model["intercept"] == pytest.approx(1.)
    assert cv2.predicted_seconds(model, "one two three") == pytest.approx(2.2)
    assert cv2.predicted_seconds(model, "") == 1.


# ---- continuous segments -------------------------------------------------------------------------------------------------

def gloss_inputs():
    utterances = [dict(collection="asllrp-continuous", source="Cory_2013-6-25_sc01", utterance=1, frames=[100, 139],
                       utterance_video="u1.mov",
                       signs=[dict(gloss="CUT", occurrence="CUT", sign_type="Lexical Signs", frames=[105, 112]),
                              dict(gloss="ONION", occurrence="ONION", sign_type="Lexical Signs", frames=[118, 125]),
                              dict(gloss="ZEBRA", occurrence="ZEBRA", sign_type="Lexical Signs", frames=[126, 130]),
                              dict(gloss="CUT", occurrence="CUT", sign_type="Lexical Signs", frames=[90, 95])]),
                  dict(collection="asllrp-continuous", source="4-Ben-Deaf-Speech", utterance=2, frames=[0, 39],
                       utterance_video="u2.mov",
                       signs=[dict(gloss="CUT", occurrence="CUT", sign_type="Lexical Signs", frames=[5, 9])])]
    gloss = dict(records=[dict(sign="asllrp-continuous:u1", partition="train"),
                          dict(sign="asllrp-continuous:u2", partition="test")])
    return utterances, gloss


def test_asllrp_segments_map_timed_signs_of_train_and_val_only():
    utterances, gloss = gloss_inputs()
    pools, counts = lib.asllrp_segments(utterances, gloss, {"cut": 1., "onion": 1.}, {}, {}, {})
    assert sorted(pools) == ["cut", "onion"] and len(pools["cut"]) == 1          # Ben (test) never; untimed CUT out
    seg = pools["cut"][0]
    assert (seg["start"], seg["end"], seg["utterance_frames"], seg["tier"]) == (5, 12, 40, "name")
    assert counts["unmapped"] == 1 and counts["sign_untimed"] == 1 and counts["segments"] == 2


def test_segment_clip_cuts_the_span_with_context():
    utterances, gloss = gloss_inputs()
    pools, _ = lib.asllrp_segments(utterances, gloss, {"cut": 1., "onion": 1.}, {}, {}, {})
    utt = signer_state(7, frames=40, amplitude=.4)
    native = np.arange(40) / 24.
    piece = lib.segment_clip(pools["onion"][0], utt, native, margin=1 / 24., context=4 / 24.)
    assert len(piece["state"].timestamps) == (25 + 5) - (18 - 5) + 1           # frames 13..30
    assert (piece["lo"], piece["hi"]) == (4, 13) and piece["reason"] == "continuous"
    np.testing.assert_array_equal(piece["state"].channels["body"].values, utt.channels["body"].values[13:31])
    core = lib.segment_clip(pools["onion"][0], utt, native, margin=1 / 24.)    # default: the piece is its core
    assert (core["lo"], core["hi"]) == (0, 9) and core["whole"]
    np.testing.assert_array_equal(core["state"].channels["body"].values, utt.channels["body"].values[17:27])
    lead = comp.lead_in(core["state"], core["lo"], 3)                           # a padded lead-in holds the core start
    np.testing.assert_array_equal(lead.channels["body"].values, utt.channels["body"].values[[17, 17, 17]])
    assert lib.segment_clip(pools["onion"][0], utt, native[:30]) is None       # the frame count disagrees (25%)
    long = signer_state(7, frames=41, amplitude=.4)                            # 2.5% long: mapped proportionally
    p2 = lib.segment_clip(pools["onion"][0], long, np.arange(41) / 24., margin=1 / 24., context=4 / 24.)
    assert p2["span_scale"] == 1.025                                           # sign frames 18 -> 18, 25 -> 26
    np.testing.assert_array_equal(p2["state"].channels["body"].values, long.channels["body"].values[13:32])


# ---- word cache -----------------------------------------------------------------------------------------------------

def test_cache_helpers():
    from scripts.round20_word_cache import choices
    assert lib.entry_seed("cut") == lib.entry_seed("cut") != lib.entry_seed("onion")
    assert lib.CACHE_SEED_BASE <= lib.entry_seed("cut") < lib.CACHE_SEED_BASE + 10 ** 7
    labels = ["CUT", "THANKYOU", "A-LOT"]
    assert lib.entry_label("cut", labels) == "CUT" and lib.entry_label("thank you", labels) == "THANKYOU"
    assert lib.entry_label("a lot", labels) == "A-LOT" and lib.entry_label("go", labels) is None
    assert choices([.1, .5, .5, .9], [1, 2, 3, 4]) == {"1": 0, "2": 1, "3": 1, "4": 3}


def test_cached_material_is_compose_cores_material():
    a, b = signer_state(1, frames=20, amplitude=.4), signer_state(2, frames=22, amplitude=.4)
    job = dict(mode="core_crossfade", words=["cut", "onion", "cut"])
    out = lib.cached_material(job, {"cut": a, "onion": b}, rule=comp.CORE_RULE)
    assert [s["word"] for s in out["segments"]] == job["words"] and all(s["cached"] for s in out["segments"])
    ref = comp.parts_from_pieces([dict(state=s, **comp.core_window(s, comp.CORE_RULE)) for s in (a, b, a)])
    for x, y in zip(out["parts"]["cores"], ref["cores"]):
        np.testing.assert_array_equal(x.channels["body"].values, y.channels["body"].values)


# ---- paired statistics -----------------------------------------------------------------------------------------------

def test_paired_mean():
    r = lib.paired_mean([1., 2., None, 4.], [0., 1., 5., float("nan")], boot=200)
    assert r["difference"] == 1. and r["n"] == 2 and r["ci95"] == [1., 1.]
    assert lib.paired_mean([None], boot=10) is None
    one = lib.paired_mean([1., 3.], boot=500)
    assert one["difference"] == 2. and 1. <= one["ci95"][0] <= one["ci95"][1] <= 3.


def test_likelihood_summary_pairs_forms_and_blank():
    rows = [dict(likelihood={"blank": [-3., -3.5, 4], "v": [-2., -3.2, 4], "v/shuffled": [-2.5, -3., 4]})
            for _ in range(3)]
    out = lib.likelihood_summary(rows, ["blank", "v", "v/shuffled"], {"v": {"shuffled": "v/shuffled"}}, boot=50,
                                 other=dict(per_form={"v": [-2.2, -2.2, -2.2]}))
    f = out["forms"]["v"]
    assert f["mean"]["difference"] == -2. and f["minus_blank"]["difference"] == 1.
    assert f["minus_mismatched"]["difference"] == pytest.approx(1.2) and f["minus_shuffled"]["difference"] == .5
    assert f["minus_paired_v"]["difference"] == pytest.approx(.2)
    assert "minus_blank" not in out["forms"]["blank"]


# ---- the reference log-likelihood ---------------------------------------------------------------------------------------

def test_reference_log_likelihood_is_the_teacher_forced_loss():
    pytest.importorskip("transformers")
    import scripts.score_back_translation as sbt
    from tests.test_generation_canonical_skeleton import rest_state
    from tests.test_generation_keypoints_to_text_t5 import tiny
    torch.manual_seed(0)
    model = tiny().eval()
    states = [signer_state(k, frames=20 + 3 * k, amplitude=.3) for k in range(3)] + [rest_state(9)]
    texts = ["cut the onion", "salt", "stir it now", "go"]
    one = sbt.reference_log_likelihood(model, "keypoints", states, texts, batch=1)
    many = sbt.reference_log_likelihood(model, "keypoints", states, texts, batch=3)
    for x, y in zip(one, many):
        assert x["tokens"] == y["tokens"] and x["log_likelihood"] == pytest.approx(y["log_likelihood"], abs=1e-3)
    assert [x["tokens"] for x in one] == [len(t) + 1 for t in texts]            # bytes + end of sequence
    for state, text, x in zip(states, texts, one):
        inputs = next(sbt._encoder_batches(model, "keypoints", [state], 1))
        loss, _ = model.loss(inputs, [text])
        assert x["per_token"] == pytest.approx(-loss.detach().item(), abs=1e-4)
        assert x["log_likelihood"] < 0
    with pytest.raises(ValueError, match="T5"):
        next(sbt._encoder_batches(model, object(), states, 1))


def test_tracked_pools_keep_well_tracked_clips_or_the_best_one():
    def arm_state(seed, share):
        st = signer_state(seed, frames=20, amplitude=.4)
        body = st.channels["body"]
        valid = body.valid.copy()
        labels = list(body.labels)
        off = int(round(20 * (1 - share)))
        valid[:off, labels.index("RightForearm")] = False
        values = np.where(valid[..., None], body.values, 0.)
        return replace(st, channels=dict(st.channels, body=replace(body, valid=valid, values=values,
                                                                   confidence=valid.astype(float))))
    shares = {"a": .1, "b": .5, "c": .9, "d": .2, "e": .05}
    states = {k: arm_state(i, sh) for i, (k, sh) in enumerate(shares.items())}
    pools = {"go": [dict(sign="a"), dict(sign="b"), dict(sign="c")], "stir": [dict(sign="d"), dict(sign="e")],
             "cut": [dict(sign="a")]}
    loaded = {}
    out, counts = lib.tracked_pools(pools, ["go", "stir"], load=lambda r: states[r["sign"]], loaded=loaded, floor=.3)
    assert [r["sign"] for r in out["go"]] == ["b", "c"] and [r["sign"] for r in out["stir"]] == ["d"]
    assert out["cut"] is pools["cut"] and counts["entries_best_tracked_fallback"] == 1 and counts["records_kept"] == 3
    assert lib.right_tracked_share(states["c"]) == pytest.approx(.9) and set(loaded) == {"a", "b", "c", "d", "e"}
