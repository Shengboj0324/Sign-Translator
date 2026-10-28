"""Compositional baseline: word selection, seeds and orders, the crossfade join and time scaling (continuity, manifolds,
validity), the composed forms and controls, the reuse of the pilot's sampling paths, and an end-to-end run on a tiny
prior, a directory store and stub translator / recogniser."""
from dataclasses import replace
import io
import json

import numpy as np
import pytest
import torch

import scripts.compositional_baseline as cb
from signtranslator.pose.multichannel import CHANNELS, MultichannelMotion
from signtranslator.pose.rotations import geodesic_distance, rotation_6d_to_matrix
from tests.test_generation_signing_space import signer_state
from tests.test_sentence_exposure_diagnostic import TEMPERATURE, StubTranslator, build_store, stub_translate


def full_state(seed=0, frames=12, amplitude=.6, offset=0.):
    """signer_state with the translation, gaze and bounded channels valid too (a ramp, a turning gaze, a level)."""
    s = signer_state(seed, frames=frames, amplitude=amplitude)
    t = np.arange(frames, dtype=np.float64)
    channels = dict(s.channels)
    for name, (kind, width, _) in CHANNELS.items():
        c = s.channels[name]
        if c.valid.any():
            continue
        if kind == "translation":
            v = np.stack([t + offset, 2 * t, np.full(frames, offset)], -1)[:, None]
        elif kind == "unit_vector":
            ang = .1 * t + offset
            v = np.stack([np.cos(ang), np.sin(ang), np.zeros(frames)], -1)[:, None]
        elif kind == "rotation_6d_columns":
            ang = .05 * t + offset
            v = np.stack([np.cos(ang), np.sin(ang), 0 * t, -np.sin(ang), np.cos(ang), 0 * t], -1)[:, None]
        elif kind == "bounded_scalar":
            v = np.full((frames, 1, 1), .25 + .5 * (offset > 0))
        else:
            v = (t[:, None, None] + offset) * np.ones((frames, 1, width))
        valid = np.ones((frames, 1), bool)
        channels[name] = replace(c, values=v, valid=valid, confidence=np.full((frames, 1), .5),
                                 inference_method="estimate")
    return replace(s, channels=channels)


def steps(values):
    return np.abs(np.diff(values, axis=0)).max()


# ---- words, seeds, orders ----------------------------------------------------------------------------------------

def test_tokens_keep_every_occurrence_contractions_and_digits():
    assert cb.tokens("The dog's red book, the DOG; I can’t add 3 cups.") == [
        "the", "dog's", "red", "book", "the", "dog", "i", "can't", "add", "3", "cups"]
    assert cb.tokens("") == [] and cb.tokens(None) == []


def test_word_selection_variants():
    words = cb.tokens("I am going to show you how the onion is cut of course")
    assert cb.select_words(words, "all") == list(range(len(words)))
    content = [words[j] for j in cb.select_words(words, "content")]
    assert content == ["i", "going", "show", "you", "how", "onion", "cut", "course"]
    labels = {"SHOW", "ONION", "CUT", "GO"}
    assert [words[j] for j in cb.select_words(words, "recognizer", labels)] == ["show", "onion", "cut"]
    assert {"a", "an", "the", "is", "are", "was", "be", "do", "have", "to", "of"} <= cb.STOP_WORDS
    assert not {"can", "will", "should", "not", "and", "i", "you"} & cb.STOP_WORDS        # modals and content kept
    with pytest.raises(ValueError):
        cb.select_words(words, "recognizer")
    with pytest.raises(ValueError):
        cb.select_words(words, "nouns")


def test_word_seeds_are_unique_per_clip_and_index():
    seeds = {cb.word_seed(i, j) for i in range(300) for j in range(60)}
    assert len(seeds) == 300 * 60 and min(seeds) == cb.WORD_SEED
    assert cb.word_seed(2, 5) == cb.WORD_SEED + 2 * cb.MAX_TOKENS + 5
    with pytest.raises(ValueError):
        cb.word_seed(0, cb.MAX_TOKENS)


def test_shuffle_orders_are_seeded_non_identity_permutations():
    for n in range(2, 9):
        for clip in range(20):
            p = cb.shuffle_order(n, clip, 0)
            assert sorted(p) == list(range(n)) and p != list(range(n))
            assert p == cb.shuffle_order(n, clip, 0)
    assert cb.shuffle_order(2, 0, 0) == [1, 0]
    assert cb.shuffle_order(1, 3, 0) == [0] and cb.shuffle_order(0, 3, 0) == []
    many = {tuple(cb.shuffle_order(8, c, v)) for c in range(10) for v in range(3)}
    assert len(many) == 30                                              # clips and variants draw independently


# ---- the join ----------------------------------------------------------------------------------------------------

def test_crossfade_weights_and_overlaps():
    assert cb.crossfade_frames(.15) == 4 and cb.crossfade_frames(0.) == 0
    with pytest.raises(ValueError):
        cb.crossfade_frames(-.1)
    w = cb.crossfade_weights(4)
    np.testing.assert_allclose(w, [.2, .4, .6, .8])
    assert (w > 0).all() and (w < 1).all() and np.allclose(w + w[::-1], 1.)
    assert cb.join_overlaps([20, 30, 25], 4) == [4, 4]
    assert cb.join_overlaps([5, 3, 1, 40], 4) == [1, 0, 0]              # at most half of either neighbour


def test_concatenate_copies_pure_frames_and_crossfades_the_joins():
    a, b, c = full_state(0, 12), full_state(1, 10, offset=100.), full_state(2, 9, offset=-50.)
    out = cb.concatenate([a, b, c], 4)
    out.validate()
    assert len(out.timestamps) == 12 + 10 + 9 - 8
    np.testing.assert_allclose(np.diff(out.timestamps), 1 / 24.)
    for name in ("body", "root_translation", "left_gaze"):
        o, ca, cbb, cc = (s.channels[name].values for s in (out, a, b, c))
        np.testing.assert_array_equal(o[:8], ca[:8])                     # A before its overlap, exactly
        np.testing.assert_array_equal(o[12:14], cbb[4:6])                # B between its two overlaps
        np.testing.assert_array_equal(o[18:], cc[4:])                    # C after its overlap
    # translations: the linear crossfade, exactly
    w = cb.crossfade_weights(4)[:, None, None]
    ra, rb = a.channels["root_translation"].values, b.channels["root_translation"].values
    np.testing.assert_allclose(out.channels["root_translation"].values[8:12], (1 - w) * ra[8:] + w * rb[:4])
    # confidence: the weighted mean; observed false (no clip observed); the inference method kept
    ch = out.channels["root_translation"]
    np.testing.assert_allclose(ch.confidence, .5)
    assert not ch.observed.any()
    assert ch.inference_method == "estimate"
    assert out.channels["blink"].values.min() >= 0 and out.channels["blink"].values.max() <= 1


def rotated_state(frames, angle):
    """Every rotation channel a constant rotation by ``angle`` about z (the other channels as ``full_state``)."""
    s = full_state(0, frames)
    six = np.array([np.cos(angle), np.sin(angle), 0., -np.sin(angle), np.cos(angle), 0.])
    channels = dict(s.channels)
    for name, (kind, _, _) in CHANNELS.items():
        c = s.channels[name]
        if kind == "rotation_6d_columns" and c.valid.any():
            channels[name] = replace(c, values=np.where(c.valid[..., None], six, 0.))
    return replace(s, channels=channels)


def test_crossfade_is_continuous_and_keeps_rotations_on_so3():
    a = full_state(0, 16, amplitude=.3)
    b = full_state(1, 16, amplitude=.3, offset=10.)
    hard = cb.concatenate([a, b], 0)
    soft = cb.concatenate([a, b], 4)
    # translations: the jump at the join is spread over n + 1 steps
    for name in ("root_translation", "face"):
        jump = steps(hard.channels[name].values[14:18])
        assert steps(soft.channels[name].values[10:17]) <= max(jump / 5 + 1e-9, steps(a.channels[name].values))
    # rotations after blending: unit, orthogonal 6D columns with determinant 1 (SO(3)); gaze stays unit
    for state in (soft, cb.concatenate([a, b], 3)):
        v = state.channels["body"].values
        np.testing.assert_allclose(np.linalg.norm(v[..., :3], axis=-1), 1., atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(v[..., 3:], axis=-1), 1., atol=1e-12)
        np.testing.assert_allclose((v[..., :3] * v[..., 3:]).sum(-1), 0., atol=1e-12)
        det = torch.linalg.det(rotation_6d_to_matrix(torch.from_numpy(v))).numpy()
        np.testing.assert_allclose(det, 1., atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(state.channels["left_gaze"].values, axis=-1), 1., atol=1e-12)
    # two still clips 120 degrees apart about z: the hard cut turns 120 degrees in one frame; the crossfade turns
    # monotonically along the geodesic, every step well below the jump, the steps summing to it
    angle = 2 * np.pi / 3
    still = cb.concatenate([rotated_state(10, 0.), rotated_state(10, angle)], 4)
    m = rotation_6d_to_matrix(torch.from_numpy(still.channels["body"].values[:, 0]))
    g = geodesic_distance(m[1:], m[:-1]).numpy()
    assert len(g) == 15 and g[:5].max() < 1e-9 and g[-5:].max() < 1e-9   # the pure frames stay still
    join = g[5:10]                                                       # out of A, through the blend, into B
    assert (join > 0).all() and join.max() < angle / 2
    assert join.sum() == pytest.approx(angle, abs=1e-9)
    hard_still = cb.concatenate([rotated_state(10, 0.), rotated_state(10, angle)], 0)
    hm = rotation_6d_to_matrix(torch.from_numpy(hard_still.channels["body"].values[:, 0]))
    assert geodesic_distance(hm[10], hm[9]).item() == pytest.approx(angle)


def test_degenerate_blends_fall_back_to_the_heavier_endpoint():
    a = np.array([[[1., 0, 0, 0, 1, 0]]] * 3)
    b = np.array([[[-1., 0, 0, 0, -1, 0]]] * 3)                           # 180 degrees about z: antipodal 6D
    out = cb.blend_values("rotation_6d_columns", a, b, np.array([.25, .5, .75]))
    np.testing.assert_allclose(out[0], a[0])
    np.testing.assert_allclose(out[1], a[1])
    np.testing.assert_allclose(out[2], b[2])
    g = cb.blend_values("unit_vector", a[..., :3], b[..., :3], np.array([.25, .5, .75]))
    np.testing.assert_allclose(g[:, 0], [[1, 0, 0], [1, 0, 0], [-1, 0, 0]])
    s = cb.blend_values("bounded_scalar", np.ones((2, 1, 1)), np.zeros((2, 1, 1)), np.array([.25, .75]))
    np.testing.assert_allclose(s[:, 0, 0], [.75, .25])


def test_crossfade_validity_one_side_takes_that_side_neither_stays_invalid():
    a, b = full_state(0, 10), full_state(1, 10, offset=5.)
    va = a.channels["root_translation"]
    mask = np.ones((10, 1), bool); mask[-4:-2] = False                   # A invalid on the first two overlap frames
    a = replace(a, channels=dict(a.channels, root_translation=replace(
        va, valid=mask, values=np.where(mask[..., None], va.values, 0.), confidence=np.where(mask, .5, 0.))))
    vb = b.channels["root_translation"]
    mb = np.ones((10, 1), bool); mb[1] = False                           # B invalid on its second overlap frame
    b = replace(b, channels=dict(b.channels, root_translation=replace(
        vb, valid=mb, values=np.where(mb[..., None], vb.values, 0.), confidence=np.where(mb, .5, 0.))))
    out = cb.concatenate([a, b], 4)
    ch = out.channels["root_translation"]
    blend = slice(6, 10)
    assert ch.valid[blend, 0].tolist() == [True, False, True, True]
    np.testing.assert_array_equal(ch.values[6], vb.values[0])           # only B valid: B exactly
    assert (ch.values[7] == 0).all() and ch.confidence[7, 0] == 0       # neither: canonical absent payload
    w = cb.crossfade_weights(4)
    np.testing.assert_allclose(ch.values[8], (1 - w[2]) * va.values[8] + w[2] * vb.values[2])
    out.validate()


def test_concatenate_without_overlap_is_plain_concatenation_and_checks_metadata():
    a, b = full_state(0, 7), full_state(1, 5)
    out = cb.concatenate([a, b], 0)
    for name in CHANNELS:
        np.testing.assert_array_equal(out.channels[name].values,
                                      np.concatenate([a.channels[name].values, b.channels[name].values]))
    one = cb.concatenate([a], 4)
    np.testing.assert_array_equal(one.channels["body"].values, a.channels["body"].values)
    other = replace(b, channels=dict(b.channels, body=replace(b.channels["body"], convention="other")))
    with pytest.raises(ValueError, match="convention"):
        cb.concatenate([a, other], 2)
    with pytest.raises(ValueError):
        cb.concatenate([], 2)
    assert cb.concatenate([a, b], 4).channels["body"].source_sha256 != a.channels["body"].source_sha256


# ---- time scaling ------------------------------------------------------------------------------------------------

def test_time_scale_hits_the_frame_count_and_interpolates_linearly():
    for T in (1, 2, 5, 31, 97, 240):
        for N in (1, 2, 7, 30, 96, 241, 500):
            s = full_state(T, T)
            out = cb.time_scale(s, N)
            out.validate()
            assert len(out.timestamps) == N
            np.testing.assert_allclose(np.diff(out.timestamps), 1 / 24.) if N > 1 else None
            np.testing.assert_array_equal(out.channels["root_translation"].values[0],
                                          s.channels["root_translation"].values[0])
            if T > 1 and N > 1:
                expected = np.arange(N) * (T - 1) / (N - 1)               # source frame index of each new frame
                np.testing.assert_allclose(out.channels["root_translation"].values[:, 0, 0], expected, atol=1e-9)
                np.testing.assert_allclose(out.channels["root_translation"].values[-1],
                                           s.channels["root_translation"].values[-1], atol=1e-9)
    s = full_state(0, 10)
    assert cb.time_scale(s, 10) is s
    with pytest.raises(ValueError):
        cb.time_scale(s, 0)


# ---- forms ---------------------------------------------------------------------------------------------------------

def test_composed_forms_and_their_controls():
    from scripts.evaluate_text_to_motion import blanked
    refs = [full_state(50 + i, 20 + 3 * i) for i in range(4)]
    blanks = blanked(refs)
    selected = [[0, 1, 2], [], [1], [0, 2]]
    clips = {(i, j): full_state(10 * i + j, 8 + j, offset=float(10 * i + j)) for i, idx in enumerate(selected)
             for j in idx}
    perm = [1, 2, 3, 0]
    forms, rows = cb.compose_forms(clips, selected, refs, blanks, perm, 2, 0)
    assert set(forms) == {"composed", "composed_scaled", "composed_shuffled", "composed_shuffled_scaled",
                          "composed_mismatched", "composed_mismatched_scaled"}
    assert all(len(v) == 4 for v in forms.values())
    for k, ref in enumerate(refs):                                       # scaled forms: the reference's frame count
        for name in ("composed_scaled", "composed_shuffled_scaled", "composed_mismatched_scaled"):
            assert len(forms[name][k].timestamps) == len(ref.timestamps)
    expected = cb.concatenate([clips[(0, 0)], clips[(0, 1)], clips[(0, 2)]], 2)
    np.testing.assert_array_equal(forms["composed"][0].channels["body"].values, expected.channels["body"].values)
    order = rows[0]["shuffled_order"]
    shuffled = cb.concatenate([clips[(0, j)] for j in [selected[0][k] for k in order]], 2)
    np.testing.assert_array_equal(forms["composed_shuffled"][0].channels["body"].values,
                                  shuffled.channels["body"].values)
    # empty sentence: blank everywhere; the sentence whose other is empty: its own blank
    for name in forms:
        assert forms[name][1] is blanks[1] or name.startswith("composed_mismatched")
    assert forms["composed_mismatched"][0] is blanks[0] and forms["composed_mismatched_scaled"][0] is blanks[0]
    assert forms["composed_mismatched"][3] is forms["composed"][0]       # the other sentence's composed clip
    assert forms["composed_mismatched"][1] is forms["composed"][2]
    assert forms["composed_shuffled"][2] is not forms["composed"][2]     # one word: same order, its own state
    np.testing.assert_array_equal(forms["composed_shuffled"][2].channels["body"].values,
                                  forms["composed"][2].channels["body"].values)
    assert [r["empty"] for r in rows] == [False, True, False, False] and rows[3]["mismatched_from"] == 0


def test_fitted_durations_share_the_reference_by_typical_duration():
    plan = cb.fitted_seconds(6., [1., 2., 3.], 4)
    budget = 6. + 2 * 4 / 24.                                            # the two joins' overlap frames added back
    np.testing.assert_allclose(plan, [budget / 6, budget / 3, budget / 2])
    assert sum(plan) - 2 * 4 / 24. == pytest.approx(6.)                  # joined: the reference duration
    floored = cb.fitted_seconds(1., [1.] * 10, 0)                         # 0.1 s each would be below the floor
    assert floored == [cb.FITTED_FLOOR] * 10
    mixed = cb.fitted_seconds(2., [.1, 1.9], 0)
    assert mixed[0] == cb.FITTED_FLOOR and mixed[1] == pytest.approx(1.9)
    assert cb.fitted_seconds(3., [2.], 4) == [3.]                        # one word: no join, the whole reference
    assert cb.fitted_seconds(3., [], 4) == []
    with pytest.raises(ValueError):
        cb.fitted_seconds(3., [1., 0.], 4)


def test_typical_seconds_and_fitted_seeds():
    words = [["cut", "the", "onion"], ["the", "salt"]]
    clips = {(0, 0): full_state(0, 24), (0, 1): full_state(1, 12), (1, 0): full_state(2, 36), (1, 1): full_state(3, 6)}
    t = cb.typical_seconds(words, clips)
    assert t == {"cut": 1., "the": pytest.approx(1.), "salt": .25}       # "the": mean of 0.5 s and 1.5 s
    seeds = {cb.word_seed(i, j) for i in range(300) for j in range(60)}
    fitted = {cb.fitted_seed(i, j, v) for i in range(300) for j in range(60) for v in range(3)}
    assert len(fitted) == 3 * 300 * 60 and not seeds & fitted
    assert cb.fitted_seed(2, 5, 1) == cb.word_seed(2, 5) + 2 * cb.FITTED_SEED_BLOCK


def test_form_pairs_compare_every_timing_and_control():
    pairs = cb.form_pairs(["content"])
    names = [f"content/{b}{t}" for b in ("composed", "composed_shuffled", "composed_mismatched") for t in cb.TIMINGS]
    assert sorted(pairs) == sorted(names)
    assert pairs["content/composed_fitted"] == {
        "end_to_end": "end_to_end", "shuffled": "content/composed_shuffled_fitted",
        "mismatched_words": "content/composed_mismatched_fitted", "unscaled": "content/composed",
        "scaled": "content/composed_scaled"}
    assert pairs["content/composed_shuffled_scaled"] == {
        "end_to_end": "end_to_end", "fitted": "content/composed_shuffled_fitted",
        "unscaled": "content/composed_shuffled"}
    for name, p in pairs.items():                                         # every pair names a form that exists
        assert set(p.values()) <= set(names) | {"end_to_end"} and name not in p.values()


def test_fitted_forms_compose_the_fitted_clips():
    from scripts.evaluate_text_to_motion import blanked
    refs = [full_state(50 + i, 20 + 3 * i) for i in range(3)]
    blanks = blanked(refs)
    selected = [[0, 1], [], [0, 2, 3]]
    clips = {(i, j): full_state(10 * i + j, 30) for i, idx in enumerate(selected) for j in idx}
    fitted = {(i, j): full_state(100 + 10 * i + j, 7 + j) for i, idx in enumerate(selected) for j in idx}
    forms, rows = cb.compose_forms(clips, selected, refs, blanks, [2, 0, 1], 2, 1, fitted)
    assert list(forms)[:3] == ["composed_fitted", "composed_shuffled_fitted", "composed_mismatched_fitted"]
    assert len(forms) == 9 and all(len(v) == 3 for v in forms.values())
    expected = cb.concatenate([fitted[(2, 0)], fitted[(2, 2)], fitted[(2, 3)]], 2)
    np.testing.assert_array_equal(forms["composed_fitted"][2].channels["body"].values, expected.channels["body"].values)
    order = rows[2]["shuffled_order"]
    shuffled = cb.concatenate([fitted[(2, [0, 2, 3][k])] for k in order], 2)
    np.testing.assert_array_equal(forms["composed_shuffled_fitted"][2].channels["body"].values,
                                  shuffled.channels["body"].values)
    # mismatched: the other sentence's fitted composition at this clip's frame count; an empty other: own blank
    assert len(forms["composed_mismatched_fitted"][0].timestamps) == len(refs[0].timestamps)
    m = forms["composed_mismatched_fitted"][0]                            # clip 0's other is clip 2
    np.testing.assert_array_equal(m.channels["root_translation"].values[0],
                                  expected.channels["root_translation"].values[0])
    assert forms["composed_mismatched_fitted"][2] is blanks[2]           # clip 2's other is the empty clip 1
    assert forms["composed_fitted"][1] is blanks[1] and forms["composed_shuffled_fitted"][1] is blanks[1]
    assert rows[0]["frames_fitted"] == 7 + 8 - 2 and rows[0]["frames_mismatched_fitted_source"] == len(expected.timestamps)
    assert rows[2]["frames_mismatched_fitted_source"] is None
    # without fitted clips: the six unscaled / scaled forms only
    plain, _ = cb.compose_forms(clips, selected, refs, blanks, [2, 0, 1], 2, 1)
    assert list(plain) == list(forms)[3:]


def test_reproduction_reads_pilot_sentence_and_own_reports():
    refs, hyps = ["a b", "c d", "e f"], ["x", "y", "z"]
    pilot = dict(meaning=dict(generated=dict(bleu=1.8, hypotheses=["x", "y", "w"]), references=refs))
    r = cb.reproduction(pilot, refs, hyps)
    assert r == dict(source="meaning.generated", comparable=True, identical=2, clips=3, identical_fraction=.6667,
                     earlier_bleu=1.8)
    assert cb.reproduction(dict(whole=dict(free_run=dict(hypotheses=hyps)), references=refs), refs, hyps)[
        "identical_fraction"] == 1.
    assert cb.reproduction(dict(forms=dict(end_to_end=dict(hypotheses=hyps)), references=refs[:2]), refs, hyps)[
        "comparable"] is False
    assert cb.reproduction({}, refs, hyps)["comparable"] is False


def test_hypothesis_length_ratio():
    assert cb.hypothesis_length_ratio(["a b c", "d"], ["a b", "c d"]) == 1.
    assert cb.hypothesis_length_ratio(["", ""], ["a b", "c d"]) == 0.
    assert cb.hypothesis_length_ratio(["a"], [""]) is None


# ---- the pilot's sampling paths ------------------------------------------------------------------------------------

def test_end_to_end_and_word_clips_are_the_pilot_paths(tmp_path):
    from scripts.evaluate_text_to_motion import load_generator
    from scripts.pilot_content_check import postprocess
    from signtranslator.pose.resample import resample_motion
    store, sha, texts = build_store(tmp_path)
    _, _, sample = load_generator(store, "prior.pt", sha, "cpu", temperature=TEMPERATURE)
    texts = texts * 6                                                   # 36 texts: two pilot batches
    seconds = [.4 + .05 * (k % 5) for k in range(len(texts))]
    ours = cb.generate_end_to_end(sample, texts, seconds, 1.5)
    pilot = []
    for i in range(0, len(texts), 32):                                  # pilot_content_check.main, verbatim
        pilot += [postprocess(o) for o in sample.batch(texts[i:i + 32], 24., seconds[i:i + 32], 1.5, 1000 + i)]
    assert len(ours) == len(pilot) == 36
    for a, b in zip(ours, pilot):
        for name in a.channels:
            np.testing.assert_array_equal(a.channels[name].values, b.channels[name].values)
    word = cb.generate_word(sample, "onion", 1.5, 123)
    lexicon = postprocess(resample_motion(sample("onion", 24., None, 1.5, 50, 123), 24.))   # the pilot's lexicon
    for name in word.channels:
        np.testing.assert_array_equal(word.channels[name].values, lexicon.channels[name].values)
    fitted = cb.generate_word(sample, "onion", 1.5, 123, .45)            # conditioned on 0.45 s
    conditioned = postprocess(resample_motion(sample("onion", 24., .45, 1.5, 50, 123), 24.))
    assert len(fitted.timestamps) == round(.45 * 24)
    for name in fitted.channels:
        np.testing.assert_array_equal(fitted.channels[name].values, conditioned.channels[name].values)
    other = cb.generate_word(sample, "onion", 1.5, 124)
    assert any(not np.array_equal(word.channels[n].values, other.channels[n].values) for n in ("body", "right_hand"))
    # word workers: the same clips, in order, from spawned processes
    jobs = [("onion", 1.5, 123), ("cut", 1.5, 7), ("salt", 1.5, 100_005), ("onion", 1.5, 124, .3), ("go", 1.5, 3, .7)]
    seen = []
    here = cb.sample_words(sample, jobs, progress=seen.append, every=2)
    assert seen == [2, 4, 5]
    there = cb.sample_words(None, jobs, workers=2, worker_args=(f"dir:{tmp_path}", "prior.pt", sha, "cpu", TEMPERATURE, 1))
    assert len(here) == len(there) == 5
    for a, b in zip(here, there):
        assert a.timestamps.tolist() == b.timestamps.tolist()
        for name in a.channels:
            np.testing.assert_array_equal(a.channels[name].values, b.channels[name].values)
    for name in word.channels:
        np.testing.assert_array_equal(here[0].channels[name].values, word.channels[name].values)


# ---- end to end ----------------------------------------------------------------------------------------------------

@pytest.fixture
def stubbed(monkeypatch):
    import scripts.score_back_translation as sbt
    import score_back_translation
    import validate_sign_recognizer as vsr
    for module in (sbt, score_back_translation):
        monkeypatch.setattr(module, "load_translator", lambda store, key, sha: (StubTranslator(), "keypoints"))
        monkeypatch.setattr(module, "translate", stub_translate)
    labels = ["CUT", "ONION", "SALT", "STIR", "KNIFE", "HEAT"]
    monkeypatch.setattr(vsr, "load_model", lambda store, run, device: (None, labels, {"checkpoint_key": "r.pt"}))


def test_compositional_baseline_end_to_end_on_a_tiny_prior(tmp_path, stubbed, capsys):
    store, sha, texts = build_store(tmp_path / "store")
    argv = ["--checkpoint-key", "prior.pt", "--mix-key", "mix.json", "--mix-sha256", sha, "--clips", "4",
            "--device", "cpu", "--store", f"dir:{tmp_path / 'store'}", "--bootstrap", "40",
            "--output-key", "reports/compositional.json"]
    assert cb.main(argv) == 0
    report = json.loads(store.get("reports/compositional.json"))
    assert report["clips"] == 4 and report["variants"] == list(cb.VARIANTS)
    timings = ("composed_fitted", "composed_shuffled_fitted", "composed_mismatched_fitted", "composed",
               "composed_scaled", "composed_shuffled", "composed_shuffled_scaled", "composed_mismatched",
               "composed_mismatched_scaled")
    expected = ["real", "blank", "end_to_end"] + [f"{v}/{f}" for v in cb.VARIANTS for f in timings]
    assert list(report["forms"]) == expected
    for v in cb.VARIANTS:
        assert {"minus_shuffled", "minus_mismatched_words", "minus_unscaled", "minus_scaled", "minus_end_to_end",
                "minus_blank"} <= set(report["forms"][f"{v}/composed_fitted"])
        assert report["selection"][v]["fitted_over_reference_median"] is not None
    for name, row in report["forms"].items():
        assert len(row["hypotheses"]) == 4 and row["ci95"][0] <= row["bleu"] <= row["ci95"][1]
        assert {"mismatched", "minus_mismatched", "hypothesis_over_reference_length"} <= set(row)
        if "/" in name:
            assert "minus_end_to_end" in row and "minus_blank" in row
    assert {"minus_shuffled", "minus_mismatched_words", "minus_scaled"} <= set(report["forms"]["all/composed"])
    assert "minus_unscaled" in report["forms"]["content/composed_shuffled_scaled"]
    perm, content = report["reference_derangement"], report["content_derangement"]
    assert all(p != i for i, p in enumerate(perm)) and all(p != i for i, p in enumerate(content))
    # every word of every clip sampled once (the all variant), shared by the other variants
    assert report["word_clips"]["count"] == sum(len(cb.tokens(t)) for t in report["references"])
    sel = report["selection"]
    assert sel["all"]["words_total"] >= sel["content"]["words_total"] >= 0
    rows = report["per_clip"]
    for row in rows:
        assert [row["tokens"][j] for j in row["content"]["words"]] == [w for w in row["tokens"] if w not in cb.STOP_WORDS]
        assert all(row["tokens"][j].upper() in {"CUT", "ONION", "SALT", "STIR", "KNIFE", "HEAT"}
                   for j in row["recognizer"]["words"])
        assert row["frames_end_to_end"] == row["frames_reference"]
        assert row["all"]["frames_composed"] == sum(row["word_frames"]) - sum(cb.join_overlaps(row["word_frames"], 4))
        # fitted: each word at its planned duration, the plan summing to the reference plus the overlaps
        for v in cb.VARIANTS:
            plan = row[v]["fitted_seconds"]
            assert len(plan) == len(row[v]["words"])
            if plan and min(plan) > cb.FITTED_FLOOR:
                assert sum(plan) - (len(plan) - 1) * 4 / 24. == pytest.approx(row["frames_reference"] / 24., abs=1e-3)
        frames = [round(s * 24) for s in row["all"]["fitted_seconds"]]
        assert row["all"]["frames_fitted"] == sum(frames) - sum(cb.join_overlaps(frames, 4))
        assert abs(row["all"]["frames_fitted"] - row["frames_reference"]) <= len(frames) + 1 or min(
            row["all"]["fitted_seconds"]) == cb.FITTED_FLOOR
    assert sel["recognizer"]["empty_sentences"] == sum(not r["recognizer"]["words"] for r in rows)
    m = report["motion"]
    assert m["real"]["seconds_over_reference"] == 1. and m["end_to_end"]["seconds_over_reference"] == 1.
    assert m["all/composed_scaled"]["seconds_over_reference"] == 1. and m["blank"]["hands_tracked"] == 0.
    assert m["all/composed"]["hands_tracked"] == 1.
    printed = capsys.readouterr().out
    assert "BLEU on 4 clips" in printed and "all/composed_mismatched_scaled" in printed
    assert printed.index("PRIMARY: the fitted composed form") < printed.index("BLEU on 4 clips")
    assert m["all/composed_fitted"]["hands_tracked"] == 1.
    assert report["fitted"]["floor_seconds"] == cb.FITTED_FLOOR and "the" in report["fitted"]["typical_seconds"]
    # write-once
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        cb.main(argv)
    # a variant subset samples only its words; the end-to-end form reproduces the first run's
    assert cb.main(argv[:-1] + ["reports/content.json", "--variants", "content",
                                "--reproduce-key", "reports/compositional.json"]) == 0
    sub = json.loads(store.get("reports/content.json"))
    assert list(sub["forms"])[3:] == [f"content/{f}" for f in timings]
    assert sub["forms"]["content/composed_fitted"]["hypotheses"] == report["forms"]["content/composed_fitted"][
        "hypotheses"]
    assert sub["word_clips"]["count"] == report["selection"]["content"]["words_total"]
    assert sub["reproduction"]["identical_fraction"] == 1.
    assert sub["forms"]["content/composed"]["hypotheses"] == report["forms"]["content/composed"]["hypotheses"]
    with pytest.raises(SystemExit):
        cb.main(argv[:-1] + ["reports/x.json", "--variants", "nouns"])
    with pytest.raises(SystemExit, match="does not exist"):                # checked before any sampling
        cb.main(argv[:-1] + ["reports/y.json", "--reproduce-key", "reports/missing.json"])
    assert not store.exists("reports/y.json")
