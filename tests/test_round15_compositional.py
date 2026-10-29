"""Round 15 compositional generator: word selection (multi-word, punctuation, stop words, out of vocabulary) and the
training vocabulary, fitted durations, the chained clock and text conditioning, the reused trim rule, the crossfade
path, the job plan, the declared verdict, and a fast end-to-end run on a tiny prior with stub instruments."""
import json

import numpy as np
import pytest
import torch

import scripts.compositional_baseline as cb
import scripts.round15_compositional as r15
import signtranslator.generation.compositional as comp
from signtranslator.data_engineering.corpus_manifest import window_state
from tests.test_compositional_baseline import full_state
from tests.test_generation_signing_space import signer_state
from tests.test_sentence_exposure_diagnostic import (TEMPERATURE, StubTranslator, build_store, stub_translate,
                                                     tiny_prior)

CPU = torch.device("cpu")
IDENTITY = lambda s: s                                                  # noqa: E731


# ---- word selection ----------------------------------------------------------------------------------------------

VOCAB = {"thank you": 1., "thank": 1., "a lot": 1., "a": 1., "next to": 1., "next": 1., "to": 1., "e mail": 1.,
         "dog": 1., "cut": 1., "the": 1., "onion": 1., "is": 1., "i": 1., "go out": 1., "go": 1., "out": 1.}


def test_normalise_writes_words_as_the_isolated_corpora_do():
    assert comp.normalise("Thank you! E-mail me, the DOG's bowl; it’s 3 o'clock.") == [
        "thank", "you", "e", "mail", "me", "the", "dog's", "bowl", "it's", "3", "o'clock"]
    assert comp.normalise("itâ\u0080\u0099s") == ["it's"]                 # mis-decoded right quote
    assert comp.normalise("different--there's") == ["different", "there's"]
    assert comp.normalise(None) == [] and comp.normalise("") == []


def test_select_words_multiword_longest_first_and_stop_words():
    # "thank you" before "thank"; "a lot" and "next to" are one sign each and keep their stop words
    assert comp.select_words("Thank you, a lot!", VOCAB) == ["thank you", "a lot"]
    assert comp.select_words("Next to the dog.", VOCAB) == ["next to", "dog"]
    assert comp.select_words("Go out", VOCAB) == ["go out"]
    assert comp.select_words("go, out", VOCAB) == ["go out"]                 # punctuation is a word break
    # stop words dropped where they are entries of their own; English order kept; every occurrence
    assert comp.select_words("The onion is cut, the dog is cut.", VOCAB) == ["onion", "cut", "dog", "cut"]
    assert comp.select_words("to a dog", VOCAB) == ["dog"]
    # punctuation and hyphens as the corpora write them
    assert comp.select_words("E-mail the dog", VOCAB) == ["e mail", "dog"]
    assert comp.select_words("THANK YOU.", VOCAB) == ["thank you"]


def test_select_words_drops_out_of_vocabulary_words():
    # "I" is the letter I (never matched), "onions" falls back to "onion", contractions and possessives stay OOV
    assert comp.select_words("I cut onions, it's the dog's.", VOCAB) == ["cut", "onion"]
    assert comp.select_words("Nothing here matches", VOCAB) == []
    assert comp.select_words("", VOCAB) == [] and comp.select_words("dog", {}) == []
    # a sentence ending inside a multi-word entry falls back to the shorter entry
    assert comp.select_words("thank", VOCAB) == ["thank"] and comp.select_words("next", VOCAB) == ["next"]
    # a custom stop-word list
    assert comp.select_words("the dog", VOCAB, stop_words=frozenset({"dog"})) == ["the"]   # it replaces the default


def test_single_letter_entries_are_never_matched():
    vocab = {"i": 1., "b": 1., "go": 1., "a lot": 1., "x ray": 1.}
    assert comp.select_words("I go, b", vocab) == ["go"]                     # letters never, stop word or not
    assert comp.select_words("a lot of x-ray", vocab) == ["a lot", "x ray"]  # multi-word entries with letters stay
    assert comp.select_words("as bs", vocab) == []                           # the fallback never reaches a letter
    rows = comp.selection_details("I go", vocab)
    assert [(r["tokens"], r["kind"]) for r in rows] == [(["i"], "unmatched"), (["go"], "exact")]


def test_inflection_candidates_are_the_declared_forms_in_order():
    assert comp.inflection_candidates("cats") == ["cat"]
    assert comp.inflection_candidates("boxes") == ["boxe", "box"]
    assert comp.inflection_candidates("babies") == ["babie", "babi", "baby"]  # -s, -es, -ies -> -y
    assert comp.inflection_candidates("making") == ["mak", "make"]            # dropped e restored
    assert comp.inflection_candidates("running") == ["runn", "run", "runne"]   # doubled consonant undoubled
    assert comp.inflection_candidates("seeing") == ["see", "seee"]           # a doubled vowel is not undoubled
    assert comp.inflection_candidates("stopped") == ["stopp", "stop", "stoppe"]
    assert comp.inflection_candidates("used") == ["us", "use"]               # -ed after e: the token minus its d
    assert comp.inflection_candidates("goes") == ["goe", "go"]
    assert comp.inflection_candidates("ing") == [] and comp.inflection_candidates("red") == []
    assert comp.inflection_candidates("dog") == [] and comp.inflection_candidates("it's") == []
    assert comp.inflection_candidates("3s") == [] and comp.inflection_candidates("dogs'") == []


def test_inflection_fallback_takes_the_first_vocabulary_hit():
    assert comp.select_words("cookies", {"cookie": 1., "cooky": 1.}) == ["cookie"]   # -s before -ies
    assert comp.select_words("cookies", {"cooky": 1.}) == ["cooky"]
    assert comp.select_words("Running dogs stopped, making babies", {"run": 1., "dog": 1., "stop": 1., "make": 1.,
                                                                     "baby": 1.}) == ["run", "dog", "stop", "make",
                                                                                      "baby"]
    assert comp.select_words("used", {"us": 1., "use": 1.}) == ["us"]       # the declared order, by spelling only
    # exact entries first: an exact single word and a multi-word entry beat the fallback
    assert comp.select_words("sing", {"sing": 1., "s": 1.}) == ["sing"]
    assert comp.select_words("thank yous", {"thank you": 1., "thank": 1., "you": 1.}) == ["thank", "you"]
    # no inflected multi-word matching: "going out" is "go" (fallback) then "out"
    assert comp.select_words("going out", {"go": 1., "go out": 1., "out": 1.}) == ["go", "out"]
    # a fallback onto a stop word is dropped as a stop word; stop-word tokens are never inflected
    rows = comp.selection_details("doing has", {"do": 1., "ha": 1.})
    assert [(r["tokens"], r["entry"], r["kind"]) for r in rows] == [(["doing"], "do", "stop"), (["has"], None, "stop")]
    # contractions and digits stay unmatched; no other mapping (irregular forms)
    assert comp.select_words("it's 3 cats went", {"it": 1., "three": 1., "cat": 1., "go": 1.}) == ["cat"]
    rows = comp.selection_details("it's went", {"it": 1., "go": 1.})
    assert [r["kind"] for r in rows] == ["unmatched", "unmatched"]


def test_selection_statistics_shares():
    vocab = {"cut": 1., "onion": 1., "thank you": 1., "go": 1.}
    details = [comp.selection_details(t, vocab) for t in ("Cut the onions, thank you!", "The dogs", "going")]
    stats = comp.selection_statistics(details)
    # tokens: cut(exact) the(stop) onions(fallback) thank you(exact x2) | the(stop) dogs(unmatched) | going(fallback)
    assert stats["sentences"] == 3 and stats["empty_sentences"] == 1 and stats["words_total"] == 4
    assert stats["words_per_sentence_mean"] == pytest.approx(4 / 3, abs=1e-3) and stats["words_per_sentence_median"] == 1.
    assert stats["fallback_words"] == 2 and stats["fallback_share"] == .5
    assert stats["tokens"] == 8
    assert stats["token_share"] == {"exact": .375, "fallback": .25, "stop": .25, "unmatched": .125}
    assert comp.selection_statistics([])["words_per_sentence_mean"] is None


def test_stop_words_are_the_declared_list():
    assert comp.STOP_WORDS == cb.STOP_WORDS
    assert {"a", "an", "the", "is", "are", "was", "be", "do", "did", "have", "has", "to", "of"} <= comp.STOP_WORDS
    assert not {"can", "will", "should", "not", "and", "i", "you", "with"} & comp.STOP_WORDS


def _record(corpus, partition, text, seconds, usable=True):
    return dict(corpus=corpus, partition=partition, text=text, sign=f"{corpus}-{text}-{seconds}",
                continuous=dict(status="converted", state_key="k", state_sha256="0" * 64, seconds=seconds,
                                available_channels=["body", "left_hand", "right_hand"] if usable else []))


def test_training_vocabulary_is_the_train_one_word_texts_with_median_seconds():
    manifest = dict(records=[
        _record("asllvd-text-v1", "train", "thank you", 1.), _record("rit-text-r6", "train", "thank you", 2.),
        _record("wlasl-text-r6", "train", "thank you", 4.), _record("asllvd-text-v1", "train", "dog", 1.5),
        _record("asllvd-text-v1", "val", "cat", 1.), _record("asllvd-text-v1", "test", "bird", 1.),   # not train
        _record("how2sign-two-view-text-v1", "train", "the dog runs", 5.),                     # not isolated
        _record("asllrp-dai2-plan-v2", "train", "IX-1p WANT", 3.),
        _record("rit-text-r6", "train", "fish", 1., usable=False),                             # not served
        _record("semlex-text-v1", "train", "Hello", 1.2)])
    vocab = comp.training_vocabulary(manifest)
    assert vocab == {"dog": 1.5, "hello": 1.2, "thank you": 2.}
    assert comp.training_vocabulary(manifest, ("asllvd-text-v1",)) == {"dog": 1.5, "thank you": 1.}


def test_semlex_text_v2_is_accepted_and_older_manifests_keep_round_15s_corpora():
    """Round 14b's semlex-text-v2 counts as isolated; a manifest without it gets round 15's four corpora, so every
    report and stage-2 model that records the list is unchanged."""
    assert "semlex-text-v2" in comp.ISOLATED_CORPORA and comp.ISOLATED_CORPORA[:4] == comp.ROUND15_CORPORA
    assert comp.ROUND15_CORPORA == ("asllvd-text-v1", "rit-text-r6", "wlasl-text-r6", "semlex-text-v1")
    old = dict(records=[_record("asllvd-text-v1", "train", "dog", 1.5), _record("semlex-text-v1", "train", "cat", 1.)])
    assert comp.isolated_corpora(old) == comp.ROUND15_CORPORA
    assert comp.training_vocabulary(old) == comp.training_vocabulary(old, comp.ROUND15_CORPORA) == {"cat": 1., "dog": 1.5}
    new = dict(records=[_record("asllvd-text-v1", "train", "dog", 1.5), _record("semlex-text-v2", "train", "ice", .8),
                        _record("semlex-text-v2", "val", "fall", .9)])
    assert comp.isolated_corpora(new) == comp.ROUND15_CORPORA + ("semlex-text-v2",)
    assert comp.training_vocabulary(new) == {"dog": 1.5, "ice": .8}
    assert comp.training_vocabulary(new, comp.isolated_corpora(new)) == {"dog": 1.5, "ice": .8}
    assert comp.training_vocabulary(new, comp.ROUND15_CORPORA) == {"dog": 1.5}


# ---- fitted durations --------------------------------------------------------------------------------------------

def test_fitted_durations_share_the_total_by_typical_duration_with_a_floor():
    typical = {"a": 1., "b": 3., "c": .1}
    d = comp.fitted_durations(["a", "b"], typical, 2.)
    assert d == pytest.approx([.5, 1.5]) and sum(d) == pytest.approx(2.)
    d = comp.fitted_durations(["a", "b", "a"], typical, 5.)
    assert d == pytest.approx([1., 3., 1.])                               # repeats share alike
    d = comp.fitted_durations(["c", "b"], typical, 1.55)
    assert d[0] == comp.FLOOR and d[1] == pytest.approx(1.5)                # the floor; the sum exceeds the total
    assert sum(d) > 1.55
    assert comp.fitted_durations([], typical, 3.) == []
    with pytest.raises(ValueError):
        comp.fitted_durations(["a"], {"a": 0.}, 1.)
    with pytest.raises(KeyError):
        comp.fitted_durations(["zzz"], typical, 1.)


def test_plan_durations_add_the_crossfade_overlaps_only():
    typical = {"a": 1., "b": 1.}
    assert sum(r15.plan_durations("crossfade", ["a", "b"], typical, 2., 4)) == pytest.approx(2. + 4 / 24)
    assert sum(r15.plan_durations("chained", ["a", "b"], typical, 2., 4)) == pytest.approx(2.)
    assert sum(r15.plan_durations("chained_trim", ["a"], typical, 2., 4)) == pytest.approx(2.)


# ---- chained: clock, duration and text ---------------------------------------------------------------------------

def spy(monkeypatch, prior):
    """Record which text each memory came from and every temporal call's (text, clock, duration, history length)."""
    texts, calls = {}, []
    encode, temporal = prior.encode, prior.temporal_states

    def recording_encode(tokens, mask, loci=None):
        out = encode(tokens, mask, loci)
        texts[id(out[0])] = bytes(int(b) for b in tokens[0].tolist() if 0 < int(b) < 256).decode()
        return out

    def recording_temporal(hidden, mask, previous, seconds, duration, **kw):
        calls.append(dict(text=texts.get(id(hidden), "?"), clock=seconds[0].tolist(), duration=float(duration[0]),
                          history=previous.shape[1]))
        return temporal(hidden, mask, previous, seconds, duration, **kw)
    monkeypatch.setattr(prior, "encode", recording_encode)
    monkeypatch.setattr(prior, "temporal_states", recording_temporal)
    return texts, calls


def test_chained_clock_is_continuous_and_each_segment_reads_its_own_word(monkeypatch):
    tokenizer, prior = tiny_prior(locus_conditioning=True)
    texts, calls = spy(monkeypatch, prior)
    words, durations = ["cut", "onion now", "go"], [.25, .3, .2]                 # 6, 7, 5 frames
    state, info = comp.compose(prior, tokenizer, words, durations, "chained", [3, 4, 5], guidance=1.5,
                               temperature=TEMPERATURE, post=IDENTITY, device=CPU)
    frames = [6, 7, 5]
    ends = np.cumsum(frames)
    assert len(state.timestamps) == sum(frames) and info["frames"] == sum(frames)
    conditional = [c for c in calls if c["text"] != "?" and c["text"] in words]
    guided = [c for c in calls if c not in conditional]
    assert len(conditional) == len(guided) == sum(frames)                   # one guided pair per step
    for step, c in enumerate(conditional):
        k = int(np.searchsorted(ends, step, side="right"))                  # the segment this step belongs to
        assert c["text"] == words[k]                                        # word k's text only, never the sentence
        assert c["clock"] == pytest.approx(list(np.arange(step + 1) / 24.))  # uniform from 0: continuous over words
        assert c["duration"] == pytest.approx(ends[k] / 24.)                 # the cumulative end of word k
        assert c["history"] == step + 1                                     # start row + every earlier code
    for g, c in zip(guided, conditional):                                   # the empty text, same clock and duration
        assert g["clock"] == c["clock"] and g["duration"] == c["duration"]
    assert [s["clock_start"] for s in info["segments"]] == pytest.approx([0., 6 / 24., 13 / 24.])
    assert [s["duration_input"] for s in info["segments"]] == pytest.approx(list(ends / 24.))
    assert not any(" " in t and t != "onion now" for t in texts.values())    # no sentence was ever encoded


def test_chained_continues_its_own_codes_and_one_word_is_the_isolated_word():
    tokenizer, prior = tiny_prior(locus_conditioning=True, history_stages=1)
    _, two = comp.compose(prior, tokenizer, ["cut", "go"], [.25, .2], "chained", [3, 4], guidance=1.5,
                          temperature=TEMPERATURE, post=IDENTITY, device=CPU)
    _, one = comp.compose(prior, tokenizer, ["cut"], [.25], "chained", [3], guidance=1.5, temperature=TEMPERATURE,
                          post=IDENTITY, device=CPU)
    assert torch.equal(two["codes"][:6], one["codes"])                      # the chain only appends
    state, _ = comp.compose(prior, tokenizer, ["cut"], [.25], "chained", [3], guidance=1.5, temperature=TEMPERATURE,
                            post=IDENTITY, device=CPU)
    word = comp.generate_word(prior, tokenizer, "cut", .25, 3, guidance=1.5, temperature=TEMPERATURE, post=IDENTITY)
    for name in word.channels:                                              # V2's first word is V1's first word
        np.testing.assert_allclose(state.channels[name].values, word.channels[name].values, atol=1e-6)
    with pytest.raises(ValueError):
        comp.compose(prior, tokenizer, ["cut"], [.2, .3], "chained", [1], guidance=1., temperature=None,
                     post=IDENTITY)
    with pytest.raises(ValueError):
        comp.compose(prior, tokenizer, ["cut"], [.2], "glued", [1], guidance=1., temperature=None, post=IDENTITY)


# ---- the trim rule ------------------------------------------------------------------------------------------------

def test_the_trim_rule_is_the_manifest_builders_own():
    import scripts.build_semlex_manifest as bsm
    import scripts.build_trimmed_word_manifest as btw
    import scripts.canonicalize_asl_citizen as cac
    import scripts.exposure_diagnostic as ed
    assert btw.sign_window is comp.sign_window and btw.trim_window is comp.trim_window
    assert cac.side_tracks is comp.side_tracks
    assert ed.continue_from is comp.continue_from and ed.make_decode is comp.make_decode
    assert cb.concatenate is comp.concatenate and cb.take_frames is comp.take_frames
    state = signer_state(4, frames=40, amplitude=.6)
    assert bsm.trim_window(state, .1, .25, .3, .5) == comp.trim_window(state, .1, .25, .3, .5)


def test_sign_window_rule_and_window_frames_match_the_dataset():
    fps = 24.
    heights = np.r_[np.zeros(10), np.linspace(0, 1, 5), np.ones(10), np.linspace(1, 0, 5), np.zeros(10)]
    tracked = np.ones(len(heights), bool)
    lo, hi = comp.sign_window(heights, tracked, fps, .1, .25)
    active = np.flatnonzero(heights > .1)
    assert lo == pytest.approx(active[0] / fps - .25) and hi == pytest.approx(active[-1] / fps + .25)
    assert comp.sign_window(heights, np.zeros_like(tracked), fps, .1, .25) is None
    state = full_state(frames=40)
    for window in ((.4583, 1.25), (0., 1.625), (.125, .125), (1.4, 5.)):
        kept = window_state(state, *window)
        idx = comp.window_frames(40, *window)
        np.testing.assert_array_equal(state.timestamps[idx], kept.timestamps)


def test_chained_trim_cuts_each_segment_before_the_next_word_continues(monkeypatch):
    tokenizer, prior = tiny_prior()
    windows = iter([((2 / 24., 5 / 24.), "trimmed"), (None, "no_sign_window"), ((0., 3 / 24.), "trimmed")])
    seen = []
    monkeypatch.setattr(comp, "trim_window", lambda segment, **kw: (seen.append((len(segment.timestamps), kw)),
                                                                     next(windows))[1])
    prefixes = []
    original = comp.continue_from

    def recording(prior_, tokenizer_, decode, text, prefix, count, seed, **kw):
        prefixes.append((text, prefix.shape[1], count))
        return original(prior_, tokenizer_, decode, text, prefix, count, seed, **kw)
    monkeypatch.setattr(comp, "continue_from", recording)
    trim = dict(threshold=.2, margin=.1, min_tracked=.3, min_seconds=.5)
    state, info = comp.compose(prior, tokenizer, ["cut", "go", "now"], [.4, .3, .25], "chained_trim", [1, 2, 3],
                               guidance=1.5, temperature=TEMPERATURE, post=IDENTITY, device=CPU, trim=trim)
    # segment frames 10, 7, 6: the rule saw each segment alone, with the given parameters
    assert [n for n, _ in seen] == [10, 7, 6] and all(kw == trim for _, kw in seen)
    kept = [4, 7, 4]                                                         # frames 2..5, all 7, frames 0..3
    assert [s["kept"] for s in info["segments"]] == kept
    assert [s["trim"] for s in info["segments"]] == ["trimmed", "no_sign_window", "trimmed"]
    # the next word continues from the trimmed codes
    assert prefixes == [("cut", 0, 10), ("go", 4, 11), ("now", 11, 17)]
    assert info["frames"] == len(state.timestamps) == sum(kept)
    # the kept codes are the untrimmed chain's: segment 1's frames 2..5 (same seed, so the same first segment)
    _, plain = comp.compose(prior, tokenizer, ["cut"], [.4], "chained", [1], guidance=1.5, temperature=TEMPERATURE,
                            post=IDENTITY, device=CPU)
    assert torch.equal(info["codes"][:4], plain["codes"][2:6])


def test_chained_trim_with_the_real_rule_runs_and_reports_reasons():
    tokenizer, prior = tiny_prior()
    state, info = comp.compose(prior, tokenizer, ["cut", "go"], [.7, .6], "chained_trim", [1, 2], guidance=1.5,
                               temperature=TEMPERATURE, post=IDENTITY, device=CPU)
    assert all(s["trim"] in ("trimmed", "no_sign_window", "arm_untracked") for s in info["segments"])
    assert len(state.timestamps) == sum(s["kept"] for s in info["segments"]) <= 17 + 14


# ---- crossfade -----------------------------------------------------------------------------------------------------

def test_crossfade_is_the_baselines_words_and_join_and_reorders_exactly(tmp_path):
    from scripts.evaluate_text_to_motion import load_generator
    store, sha, _ = build_store(tmp_path)
    _, _, sample = load_generator(store, "prior.pt", sha, "cpu", temperature=TEMPERATURE)
    job = dict(mode="crossfade", words=["cut", "onion", "go"], durations=[.3, .45, .35], seeds=[11, 12, 13])
    settings = dict(guidance=1.5, temperature=TEMPERATURE, crossfade=4, trim=comp.TRIM)
    state, info = r15.compose_job(sample, job, **settings)
    # the baseline's fitted word clips (the pilot's lexicon path), joined by its crossfade
    clips = [cb.generate_word(sample, w, 1.5, s, d) for w, s, d in zip(job["words"], job["seeds"], job["durations"])]
    joined = cb.concatenate(clips, 4)
    assert len(state.timestamps) == len(joined.timestamps)
    for name in state.channels:
        np.testing.assert_allclose(state.channels[name].values, joined.channels[name].values, atol=1e-6)
    # reordering the clips is composing the reordered words, durations and seeds
    order = [2, 0, 1]
    again, _ = r15.compose_job(sample, dict(mode="crossfade", words=[job["words"][k] for k in order],
                                            durations=[job["durations"][k] for k in order],
                                            seeds=[job["seeds"][k] for k in order]), **settings)
    reordered = comp.concatenate([info["clips"][k] for k in order], 4)
    for name in again.channels:
        np.testing.assert_array_equal(again.channels[name].values, reordered.channels[name].values)
    # workers: the same states, in order
    jobs = [job, dict(job, mode="chained"), dict(job, mode="chained_trim", words=["go", "cut"], durations=[.3, .3],
                                                  seeds=[1, 2])]
    here = r15.run_jobs(sample, jobs, settings)
    there = r15.run_jobs(None, jobs, settings, workers=2,
                         worker_args=(f"dir:{tmp_path}", "prior.pt", sha, "cpu", TEMPERATURE, 1))
    for (a, _), (b, _) in zip(here, there):
        for name in a.channels:
            np.testing.assert_array_equal(a.channels[name].values, b.channels[name].values)


# ---- plan and verdict ----------------------------------------------------------------------------------------------

def test_plan_jobs_seeds_controls_and_durations():
    typical = {"a": 1., "b": 2., "c": 1.}
    selected = [["a", "b"], ["c"], []]
    seconds = [3., 2., 1.]
    perm = [1, 2, 0]
    orders = [[1, 0], [0], []]
    jobs = r15.plan_jobs(["crossfade", "chained"], selected, typical, seconds, perm, orders, 4)
    m = jobs[("chained", "", 0)]
    assert m["words"] == ["a", "b"] and m["durations"] == pytest.approx([1., 2.]) and m["seeds"] == [100_000, 100_001]
    s = jobs[("chained", "/shuffled", 0)]
    assert s["words"] == ["b", "a"] and s["durations"] == pytest.approx([2., 1.]) and s["seeds"] == [100_001, 100_000]
    assert jobs[("crossfade", "/shuffled", 0)] is None                       # crossfade reorders its clips
    mm = jobs[("chained", "/mismatched_words", 0)]
    assert mm["words"] == ["c"] and mm["durations"] == pytest.approx([3.])  # fitted to THIS clip's reference
    assert mm["seeds"] == [100_000 + r15.MISMATCHED_BLOCK]
    assert jobs[("chained", "", 2)] is None and jobs[("chained", "/mismatched_words", 1)] is None   # no words
    assert jobs[("chained", "/mismatched_words", 2)]["words"] == ["a", "b"]
    assert sum(jobs[("crossfade", "", 0)]["durations"]) == pytest.approx(3. + 4 / 24)
    assert jobs[("chained", "", 1)]["seeds"] == [101_000]
    seeds = {x for i in range(300) for x in r15.word_seeds(i, 60) + r15.word_seeds(i, 60, mismatched=True)}
    assert len(seeds) == 2 * 300 * 60
    with pytest.raises(ValueError):
        r15.word_seeds(0, r15.MAX_WORDS + 1)
    pairs = r15.form_pairs(["crossfade", "chained"])
    assert pairs["chained"] == {"end_to_end": "end_to_end", "shuffled": "chained/shuffled",
                                "mismatched_words": "chained/mismatched_words", "variant_crossfade": "crossfade"}
    assert pairs["crossfade/mismatched_words"] == {"end_to_end": "end_to_end"}


def _row(e2e_lower, mism_lower):
    return dict(minus_end_to_end=dict(difference=e2e_lower + .2, ci95=[e2e_lower, e2e_lower + .4]),
                minus_mismatched_words=dict(difference=mism_lower + .2, ci95=[mism_lower, mism_lower + .4]))


def test_verdict_applies_the_declared_rule():
    above, below = dict(mean=.08, ci95=[.06, .10], n=290), dict(mean=.04, ci95=[.03, .05], n=290)
    overlap = dict(mean=.055, ci95=[.045, .065], n=290)
    assert r15.verdict(_row(.05, .1), above, below)["verdict"] == "PROMISING"
    v = r15.verdict(_row(0., .1), above, below)                               # lower bound exactly 0 fails
    assert v["verdict"] == "NOT PROMISING" and not v["bleu_vs_end_to_end_lower_above_0"]
    v = r15.verdict(_row(.05, -.01), above, below)
    assert v["verdict"] == "NOT PROMISING" and not v["bleu_vs_mismatched_words_lower_above_0"]
    v = r15.verdict(_row(.05, .1), above, overlap)                            # intervals overlap: not above
    assert v["verdict"] == "NOT PROMISING" and not v["spotting_recall_above_mismatched"]
    assert v["bleu_vs_end_to_end_lower_above_0"] and v["bleu_vs_mismatched_words_lower_above_0"]
    v = r15.verdict(_row(.05, .1), dict(mean=None, ci95=None, n=0), below)    # no spotting words
    assert v["verdict"] == "NOT PROMISING"
    assert r15.paired_recall([dict(recall=.5), dict(recall=None), dict(recall=.2)],
                             [dict(recall=.25), dict(recall=.1), dict(recall=.2)])["n"] == 2


def test_trim_parameters_come_from_the_manifest():
    assert r15.trim_parameters({}) == comp.TRIM
    block = dict(threshold=.2, margin=.3, min_tracked=.4, min_seconds=.6, rule="x", corpora=["a"])
    assert r15.trim_parameters(dict(trim=block)) == dict(threshold=.2, margin=.3, min_tracked=.4, min_seconds=.6)


# ---- end to end ----------------------------------------------------------------------------------------------------

@pytest.fixture
def stubbed(monkeypatch):
    import scripts.score_back_translation as sbt
    import score_back_translation
    import sentence_word_spotting as sws
    import validate_sign_recognizer as vsr
    for module in (sbt, score_back_translation):
        monkeypatch.setattr(module, "load_translator", lambda store, key, sha: (StubTranslator(), "keypoints"))
        monkeypatch.setattr(module, "translate", stub_translate)
    labels = ["CUT", "ONION", "SALT", "STIR", "KNIFE", "HEAT", "HALF", "NOW"]
    monkeypatch.setattr(vsr, "load_model", lambda store, run, device: (None, labels, {"checkpoint_key": "r.pt"}))

    def windows(model, state, device, *, windows, hop, k):                  # motion-dependent fake recogniser
        bounds = sws.window_bounds(len(state.timestamps), 24., windows, hop)
        rh = state.channels["right_hand"]
        level = int(abs(float(rh.values[rh.valid].mean())) * 1000) if rh.valid.any() else 0
        ids = np.array([[(level + j + b) % len(labels) for j in range(k)] for b in range(len(bounds))])
        return ids, np.full(ids.shape, 1. / k), np.array([c for _, _, c in bounds])
    monkeypatch.setattr(sws, "window_predictions", windows)


def test_round15_end_to_end_on_a_tiny_prior(tmp_path, stubbed, capsys):
    train = ("cut", "onion", "salt", "stir", "heat", "knife", "thank you")
    store, sha, texts = build_store(tmp_path / "store", corpus="asllvd-text-v1", train_texts=train)
    argv = ["--checkpoint-key", "prior.pt", "--mix-key", "mix.json", "--mix-sha256", sha, "--clips", "5",
            "--device", "cpu", "--store", f"dir:{tmp_path / 'store'}", "--bootstrap", "40",
            "--output-key", "reports/r15.json"]
    assert r15.main(argv) == 0
    report = json.loads(store.get("reports/r15.json"))
    variants = list(comp.MODES)
    assert report["clips"] == 5 and report["variants"] == variants
    expected = ["real", "blank", "end_to_end"] + [f"{v}{c}" for v in variants for c in r15.CONTROLS]
    assert list(report["forms"]) == expected
    for v in variants:
        row = report["forms"][v]
        assert {"minus_end_to_end", "minus_shuffled", "minus_mismatched_words", "minus_blank", "minus_mismatched"} \
            <= set(row)
        d = report["verdicts"][v]
        assert d["verdict"] in ("PROMISING", "NOT PROMISING")
        assert d["verdict"] == ("PROMISING" if all(d[k] for k in ("bleu_vs_end_to_end_lower_above_0",
                                                                   "bleu_vs_mismatched_words_lower_above_0",
                                                                   "spotting_recall_above_mismatched"))
                                else "NOT PROMISING")
        assert {v, f"{v}_mismatched"} <= set(report["spotting"]["scores"])
    for name, row in report["forms"].items():
        assert len(row["hypotheses"]) == 5 and row["ci95"][0] <= row["bleu"] <= row["ci95"][1]
    # word selection from the training vocabulary only (never the recogniser's labels): HALF and NOW are labels
    vocab = set(train)
    for row in report["per_clip"]:
        assert row["words"] == comp.select_words(row["text"], vocab)
        assert "half" not in row["words"] and "now" not in row["words"]
        assert row["frames_end_to_end"] == row["frames_reference"]
        if row["words"]:
            plan = row["chained"]["durations"]
            assert sum(plan) == pytest.approx(row["frames_reference"] / 24., abs=1e-3) or min(plan) == comp.FLOOR
            assert row["chained"]["frames"] == sum(s["frames"] for s in row["chained"]["segments"])
            assert row["chained_trim"]["frames"] == sum(s["kept"] for s in row["chained_trim"]["segments"])
            assert [s["word"] for s in row["chained"]["segments"]] == row["words"]
    assert report["selection"]["empty_sentences"] == sum(not r["words"] for r in report["per_clip"])
    sel = report["selection"]
    assert set(sel["per_variant"]) == set(variants)                         # amendment 1: per variant
    for v in variants:
        pv = sel["per_variant"][v]
        assert pv["words_total"] == sum(len(r["words"]) for r in report["per_clip"])
        assert pv["fallback_words"] == sum(len(r["fallback"]) for r in report["per_clip"])
        assert set(pv["token_share"]) == {"exact", "fallback", "stop", "unmatched"}
        assert sum(pv["token_share"].values()) == pytest.approx(1., abs=1e-3)
    assert sel["token_share"]["stop"] > 0 and sel["token_share"]["unmatched"] > 0
    assert report["typical_seconds"]["cut"] == pytest.approx(14 / 24., abs=1e-4)
    assert "cumulative end of word k" in report["chain_design"]
    m = report["motion"]
    assert m["real"]["seconds_over_reference"] == 1. and m["end_to_end"]["seconds_over_reference"] == 1.
    assert m["chained"]["hands_tracked"] == 1.
    printed = capsys.readouterr().out
    assert "VERDICT crossfade:" in printed and "VERDICT chained_trim:" in printed
    with pytest.raises(SystemExit, match="refusing to overwrite"):           # write-once
        r15.main(argv)
    # a variant subset reproduces the same compositions (the seeds do not depend on the other variants)
    assert r15.main(argv[:-1] + ["reports/sub.json", "--variants", "chained"]) == 0
    sub = json.loads(store.get("reports/sub.json"))
    assert list(sub["forms"])[3:] == ["chained", "chained/shuffled", "chained/mismatched_words"]
    for name in ("chained", "chained/shuffled", "chained/mismatched_words", "end_to_end"):
        assert sub["forms"][name]["hypotheses"] == report["forms"][name]["hypotheses"]
    with pytest.raises(SystemExit):
        r15.main(argv[:-1] + ["reports/x.json", "--variants", "glued"])
    out = tmp_path / "local.json"
    assert r15.main(argv[:-2] + ["--output-file", str(out), "--variants", "crossfade"]) == 0
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        r15.main(argv[:-2] + ["--output-file", str(out)])
