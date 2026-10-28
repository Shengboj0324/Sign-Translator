"""Round 15 oracle composition: the real-clip choice (seeded per clip and word, train partition of the isolated corpora
only), the resampling to fitted durations, the controls, the pairing check against the stored report, the declared
readings, and an end-to-end run on a directory store with stub instruments."""
from dataclasses import replace
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

import scripts.compositional_baseline as cb
import scripts.round15_compositional as r15
import scripts.round15_oracle as ro
import scripts.sentence_exposure_diagnostic as sd
import signtranslator.generation.compositional as comp
from signtranslator.data_engineering.corpus_manifest import ContentCache, KINDS, usable_channels, window_state
from signtranslator.data.text_motion_dataset import DirectoryStore
from signtranslator.pose.multichannel import CHANNELS
from tests.test_compositional_baseline import full_state
from tests.test_generation_signing_space import signer_state
from tests.test_round15_compositional import stubbed  # noqa: F401  (the stub instruments fixture)
from tests.test_sentence_exposure_diagnostic import build_store


def _record(corpus, partition, text, sign, seconds=1., usable=True):
    r = dict(corpus=corpus, partition=partition, text=text, sign=sign,
             continuous=dict(status="converted", seconds=seconds, state_key=f"k/{sign}", state_sha256=f"{sign:0>64}",
                             available_channels=["body", "left_hand", "right_hand"] if usable else []))
    return r


def _manifest():
    recs = []
    for k in range(6):
        recs.append(_record("asllvd-text-v1", "train", "Dog", f"a{k}"))
    recs += [_record("rit-text-r6", "train", "dog", "r0"), _record("wlasl-text-r6", "train", "cat", "w0"),
             _record("asllvd-text-v1", "val", "dog", "v0"), _record("asllvd-text-v1", "test", "dog", "t0"),
             _record("rit-text-r6", "train", "dog", "u0", usable=False),
             _record("how2sign-two-view-text-v1", "train", "dog", "h0"),
             _record("asllvd-text-v1", "train", "thank you", "m0")]
    return dict(records=recs)


# ---- clip choice -------------------------------------------------------------------------------------------------

def test_word_pools_are_the_training_vocabularys_train_records_only():
    manifest = _manifest()
    pools = ro.word_pools(manifest, comp.ISOLATED_CORPORA)
    assert set(pools) == set(comp.training_vocabulary(manifest, comp.ISOLATED_CORPORA)) == {"dog", "cat", "thank you"}
    assert [r["sign"] for r in pools["dog"]] == ["a0", "a1", "a2", "a3", "a4", "a5", "r0"]   # sorted by corpus, sign
    for pool in pools.values():                        # never val/test, another corpus or an unusable record
        assert all(r["partition"] == "train" and r["corpus"] in comp.ISOLATED_CORPORA for r in pool)
        assert not {"v0", "t0", "u0", "h0"} & {r["sign"] for r in pool}
    shuffled = dict(records=random.Random(3).sample(manifest["records"], len(manifest["records"])))
    assert ro.word_pools(shuffled, comp.ISOLATED_CORPORA) == pools            # the record order does not matter


def test_choose_record_is_fixed_per_clip_and_word_and_block():
    pools = ro.word_pools(_manifest(), comp.ISOLATED_CORPORA)
    picks = [ro.choose_record(pools, i, "dog")["sign"] for i in range(200)]
    assert picks == [ro.choose_record(pools, i, "dog")["sign"] for i in range(200)]          # seeded
    assert set(picks) == {r["sign"] for r in pools["dog"]}                                     # the whole pool
    counts = np.bincount([[r["sign"] for r in pools["dog"]].index(s) for s in picks])
    assert counts.min() > 10                                                                   # roughly uniform
    assert ro.clip_seed(3, "dog") == ro.clip_seed(3, "dog")
    assert len({ro.clip_seed(3, "dog"), ro.clip_seed(4, "dog"), ro.clip_seed(3, "cat"),
                ro.clip_seed(3, "dog", "mismatched_words")}) == 4                              # clip, word, block
    mism = [ro.choose_record(pools, i, "dog", "mismatched_words")["sign"] for i in range(200)]
    assert mism != picks
    assert ro.choose_record(pools, 7, "cat")["sign"] == "w0"
    with pytest.raises(KeyError):
        ro.choose_record(pools, 0, "bird")
    with pytest.raises(ValueError):
        ro.clip_seed(0, "dog", "shuffled")


def test_plan_oracle_has_v1_durations_and_its_controls():
    pools = ro.word_pools(_manifest(), comp.ISOLATED_CORPORA)
    vocab = comp.training_vocabulary(_manifest(), comp.ISOLATED_CORPORA)
    selected = [["dog", "cat", "dog"], [], ["thank you"]]
    seconds, perm = [2., 1., .9], [2, 0, 1]
    plans = ro.plan_oracle(selected, pools, vocab, seconds, perm, 4)
    p = plans[("matched", 0)]
    assert p["words"] == selected[0]
    assert p["fitted"] == r15.plan_durations("crossfade", selected[0], vocab, 2., 4)      # V1's durations
    assert p["frames"] == [comp.word_frames(d) for d in p["fitted"]]
    assert p["records"][0] is p["records"][2]                                  # a repeated word: the same clip
    assert [r["sign"] for r in p["records"]] == [ro.choose_record(pools, 0, w)["sign"] for w in selected[0]]
    assert plans[("matched", 1)] is None                                       # no word: the blank clip
    m = plans[("mismatched_words", 1)]                                         # sentence 0's words, clip 1's timing
    assert m["words"] == selected[0]
    assert m["fitted"] == r15.plan_durations("crossfade", selected[0], vocab, 1., 4)
    assert [r["sign"] for r in m["records"]] == [ro.choose_record(pools, 1, w, "mismatched_words")["sign"]
                                                 for w in selected[0]]
    assert plans[("mismatched_words", 2)] is None                              # sentence 1 has no word


# ---- resampling and composition ----------------------------------------------------------------------------------

def test_fitted_clips_are_time_resampled_to_their_frames_and_unscaled_kept():
    clips = [full_state(seed=k, frames=f, offset=k) for k, f in enumerate((40, 9, 25))]
    frames = [12, 18, 7]
    fitted = ro.compose_clips(clips, frames, 4, fitted=True)
    parts = [cb.time_scale(c, f) for c, f in zip(clips, frames)]
    assert len(fitted.timestamps) == sum(frames) - sum(comp.join_overlaps(frames, 4))
    expected = comp.concatenate(parts, 4)
    for name in fitted.channels:
        np.testing.assert_array_equal(fitted.channels[name].values, expected.channels[name].values)
    for c, part, f in zip(clips, parts, frames):                               # first and last real frames kept
        assert len(part.timestamps) == f
        v, w = c.channels["right_hand"].values, part.channels["right_hand"].values
        np.testing.assert_allclose(w[0], v[0])
        np.testing.assert_allclose(w[-1], v[-1])
    # a linear signal stays linear on the new clock (compressed 40 -> 12 frames)
    rt = parts[0].channels["root_translation"].values[:, 0, 0]
    np.testing.assert_allclose(rt, np.linspace(0., 39., 12), atol=1e-9)
    unscaled = ro.compose_clips(clips, frames, 4, fitted=False)
    lengths = [40, 9, 25]
    assert len(unscaled.timestamps) == sum(lengths) - sum(comp.join_overlaps(lengths, 4))
    np.testing.assert_array_equal(unscaled.channels["body"].values,
                                  comp.concatenate(clips, 4).channels["body"].values)


def test_shuffled_is_the_same_clips_reordered():
    clips = [full_state(seed=k, frames=f, offset=k) for k, f in enumerate((20, 14, 30))]
    frames, order = [10, 8, 12], [2, 0, 1]
    for fitted in (True, False):
        out = ro.compose_clips(clips, frames, 4, fitted=fitted, order=order)
        same = ro.compose_clips([clips[k] for k in order], [frames[k] for k in order], 4, fitted=fitted)
        for name in out.channels:
            np.testing.assert_array_equal(out.channels[name].values, same.channels[name].values)


def test_word_clip_keeps_the_sign_window_conforms_and_resamples(tmp_path):
    store = DirectoryStore(tmp_path / "s")
    state = signer_state(5, frames=60, amplitude=.4)
    state = replace(state, timestamps=np.arange(60) / 30.)                     # a 30 fps corpus
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "s.npz"
        sha = state.save(path)
        store.put("st.npz", path.read_bytes())
    r = dict(sign="w", corpus="rit-text-r6", partition="train", text="dog",
             continuous=dict(status="converted", state_key="st.npz", state_sha256=sha,
                             available_channels=list(CHANNELS), seconds=1.0, window_seconds=[.5, 1.5]))
    r["usable"] = {kind: list(usable_channels(r, kind)) for kind in KINDS}
    template = signer_state(0, frames=3)
    clip = ro.word_clip(r, cache=ContentCache(tmp_path / "c", 2 ** 20), fetch=store.get, template=template)
    windowed = window_state(state, .5, 1.5)                                    # 31 frames at 30 fps, 1.0 s
    assert len(windowed.timestamps) == 31
    assert len(clip.timestamps) == 25                                          # 1.0 s at 24 fps
    assert np.allclose(np.diff(clip.timestamps), 1 / 24.)
    np.testing.assert_allclose(clip.channels["body"].values[0], windowed.channels["body"].values[0])
    assert {n: c.labels for n, c in clip.channels.items()} == {n: c.labels for n, c in template.channels.items()}


# ---- readings and pairing ------------------------------------------------------------------------------------------

def _diff(lo, hi=None):
    return dict(difference=(lo + (hi if hi is not None else lo + 1)) / 2, ci95=[lo, hi if hi is not None else lo + 1])


def _row(e2e, gen):
    return dict(minus_end_to_end=_diff(e2e), minus_generated_crossfade=_diff(gen), minus_mismatched_words=_diff(.1),
                minus_shuffled=_diff(.1), minus_mismatched=_diff(.1))


def test_reading_applies_the_pinned_rule():
    assert ro.reading(_row(0., 1.))["reading"] == "i"                           # lower bound <= 0: (i)
    assert ro.reading(_row(-.4, 1.))["text"] == ro.READING_I
    assert ro.reading(_row(.2, .1))["reading"] == "ii"                          # lower bound above 0: (ii)
    r = ro.reading(_row(.2, -.1))                                               # the generated crossfade: secondary
    assert r["reading"] == "ii" and r["secondary"]["above_generated_crossfade"] is False
    assert ro.reading(_row(.2, .3))["secondary"]["above_generated_crossfade"] is True
    for row in (_row(0., 1.), _row(.2, -.1)):                                   # an earlier row reads the same
        assert ro.reading_of(dict(row, reading="neither"))["reading"] == ro.reading(row)["reading"]


def test_conclusion_reports_the_unscaled_oracle_beside_the_fitted_one():
    i, ii = ro.reading(_row(-.1, 0.)), ro.reading(_row(.1, 0.))
    assert ro.conclusion(ii, i)["text"] == ro.READING_II
    assert ro.conclusion(ii, ii)["text"] == ro.READING_II
    assert ro.conclusion(i, i)["text"] == ro.READING_I
    assert ro.conclusion(i, ii)["text"] == ro.NATURAL_PACE                      # never (i) alone
    c = ro.conclusion(i, ii)
    assert (c["fitted"], c["unscaled"]) == ("i", "ii") and "different signer" in c["limits"]


def test_v1_plan_check_counts_mismatches_against_the_stored_rows():
    rows = [dict(frames_reference=30, frames=dict(oracle=26),
                 matched=dict(fitted=[.5, .75], clips=[dict(frames_fitted=12), dict(frames_fitted=18)])),
            dict(frames_reference=20, frames=dict(oracle=20), matched=None)]
    stored = [dict(frames_reference=30, crossfade=dict(empty=False, durations=[.5, .75], frames=26,
                                                       segments=[dict(word="a", frames=12), dict(word="b", frames=18)])),
              dict(frames_reference=20, crossfade=dict(empty=True, frames=20))]
    check = ro.v1_plan_check(rows, stored)
    assert check["clips"] == 2 and set(check["mismatches"].values()) == {0} and not check["examples"]
    stored[0]["crossfade"]["segments"][1]["frames"] = 17
    stored[0]["crossfade"]["durations"] = [.5, .7501]
    stored[1]["frames_reference"] = 21
    check = ro.v1_plan_check(rows, stored)
    assert check["mismatches"] == dict(frames_reference=1, empty=0, durations=1, segment_frames=1, total_frames=0)
    assert check["examples"] == [dict(clip=0, failed=["durations", "segment_frames"]),
                                 dict(clip=1, failed=["frames_reference"])]
    stored[1]["crossfade"]["empty"] = False                                     # V1 had words, the oracle none
    assert ro.v1_plan_check(rows, stored)["mismatches"]["empty"] == 1


def test_form_pairs_compare_each_oracle_to_both_references_and_its_controls():
    pairs = ro.form_pairs()
    assert pairs["oracle"] == {"end_to_end": "end_to_end", "generated_crossfade": "generated_crossfade",
                               "shuffled": "oracle/shuffled", "mismatched_words": "oracle/mismatched_words",
                               "timing_oracle_unscaled": "oracle_unscaled"}
    assert pairs["oracle_unscaled/mismatched_words"] == {"end_to_end": "end_to_end"}
    assert pairs["generated_crossfade"] == {"end_to_end": "end_to_end"}


# ---- end to end ----------------------------------------------------------------------------------------------------

TRAIN = ("cut", "onion", "salt", "stir", "heat", "knife", "thank you")


def source_report(store, sha, *, clips=5, tamper=None):
    """A stored round-15 report of the store's clips, built with the round-15 functions (its hypotheses stubbed)."""
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from scripts.pilot_content_check import validation_sentences
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get("mix.json") + b"\n")
        manifest = load_manifest(path, expected_sha256=sha)
    vocab = comp.training_vocabulary(manifest, comp.ISOLATED_CORPORA)
    val = validation_sentences(manifest, clips)
    texts = [r["text"] for r in val]
    selected = [comp.select_words(t, vocab) for t in texts]
    perm, words_perm = sd.derangement(len(texts), 0), sd.derangement(len(texts), 1)
    spot = {f"{f}{s}": dict(recall=dict(mean=.1, ci95=[.05, .15], n=5), order=dict(mean=None, ci95=None, n=0))
            for f in ("real", "end_to_end", "crossfade") for s in ("", "_mismatched")}
    report = dict(
        mix_key="mix.json", mix_sha256=sha, corpora=list(comp.ISOLATED_CORPORA), clips=len(texts),
        seeds=dict(reference_derangement=0, words_derangement=1), crossfade=dict(seconds=.15, frames=4),
        references=texts, reference_derangement=perm, words_derangement=words_perm,
        typical_seconds={w: round(vocab[w], 4) for ws in selected for w in ws},
        per_clip=[dict(sign=r["sign"], words=w, shuffled_order=cb.shuffle_order(len(w), i, 0),
                       mismatched_from=words_perm[i]) for i, (r, w) in enumerate(zip(val, selected))],
        translator_run=sd.TRANSLATOR_RUN, translator_key="translator.pt", recognizer_run="sign-recognizer-v2",
        forms={f: dict(bleu=1., hypotheses=[f"{f} cut the onion {i}" for i in range(len(texts))])
               for f in ("real", "blank", "end_to_end", "crossfade")},
        spotting=dict(windows=[.6, .9, 1.2], hop=.1, top_k=5, scores=spot),
        motion_ratio_to_real=dict(end_to_end=dict(wrist_speed=1.4), crossfade=dict(wrist_speed=3.)))
    if tamper:
        tamper(report)
    store.put("source.json", json.dumps(report).encode())
    return report


def test_oracle_end_to_end_on_a_directory_store(tmp_path, stubbed, capsys):  # noqa: F811
    store, sha, texts = build_store(tmp_path / "store", corpus="asllvd-text-v1", train_texts=TRAIN)
    source = source_report(store, sha)
    argv = ["--source-report-key", "source.json", "--store", f"dir:{tmp_path / 'store'}", "--bootstrap", "40",
            "--cache-dir", str(tmp_path / "cache"), "--threads", "1", "--output-key", "reports/oracle.json"]
    assert ro.main(argv) == 0
    report = json.loads(store.get("reports/oracle.json"))
    expected = ["real", "blank", "end_to_end", "generated_crossfade"] + [f"{t}{c}" for t in ro.TIMINGS
                                                                         for c in ro.CONTROLS]
    assert list(report["forms"]) == expected
    for name in ("end_to_end", "generated_crossfade"):                 # the stored hypotheses, scored here
        stored = source["forms"][ro.STORED[name]]["hypotheses"]
        assert report["forms"][name]["hypotheses"] == stored
    assert report["reproduction"]["real"]["clips"] == 5
    for t in ro.TIMINGS:
        row = report["forms"][t]
        assert {"minus_end_to_end", "minus_generated_crossfade", "minus_shuffled", "minus_mismatched_words",
                "minus_mismatched", "minus_blank"} <= set(row)
        assert {t, f"{t}_mismatched"} <= set(report["spotting"]["scores"])
        assert report[{"oracle": "reading", "oracle_unscaled": "reading_unscaled"}[t]]["reading"] in (
            "i", "ii", "neither")
    assert "stored_end_to_end" in report["spotting"]["scores"] and "stored_generated_crossfade" in \
        report["motion_ratio_to_real"]
    # the clips: train records of the isolated corpus only, the seeded draw, real lengths and fitted frames
    assert report["clip_choice"]["corpus_share"] == {"asllvd-text-v1": 1.}
    for i, row in enumerate(report["per_clip"]):
        assert row["words"] == source["per_clip"][i]["words"]
        m = row["matched"]
        if not m:
            assert row["frames"]["oracle"] == row["frames_reference"]          # the blank clip
            continue
        assert [c["word"] for c in m["clips"]] == row["words"]
        assert all(c["frames_real"] == 14 for c in m["clips"])                # build_store's 14-frame train clips
        assert [c["frames_fitted"] for c in m["clips"]] == [comp.word_frames(d) for d in m["fitted"]]
        frames = [c["frames_fitted"] for c in m["clips"]]
        assert row["frames"]["oracle"] == sum(frames) - sum(comp.join_overlaps(frames, 4))
        assert row["frames"]["oracle_unscaled"] == 14 * len(frames) - sum(comp.join_overlaps([14] * len(frames), 4))
        mm = row["mismatched_words"]
        if mm:
            assert mm["words"] == source["per_clip"][source["per_clip"][i]["mismatched_from"]]["words"]
    printed = capsys.readouterr().out
    assert "READINGS (pinned rule" in printed and "CONCLUSION:" in printed and "limits:" in printed
    assert report["conclusion"]["text"] in (ro.READING_I, ro.READING_II, ro.NATURAL_PACE)
    assert report["clip_choice"]["untrimmed_fetched"] == report["clip_choice"]["distinct_fetched"]   # no windows
    assert report["clip_choice"]["untrimmed_pool_records"] == len(TRAIN)
    assert set(report["v1_plan_check"]["mismatches"]) >= {"frames_reference", "empty"}
    with pytest.raises(SystemExit, match="refusing to overwrite"):           # write-once
        ro.main(argv)
    # deterministic: a second run reproduces every hypothesis
    assert ro.main(argv[:-1] + ["reports/again.json"]) == 0
    again = json.loads(store.get("reports/again.json"))
    for name in report["forms"]:
        assert again["forms"][name]["hypotheses"] == report["forms"][name]["hypotheses"]
    out = tmp_path / "local.json"
    assert ro.main(argv[:-2] + ["--output-file", str(out)]) == 0
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        ro.main(argv[:-2] + ["--output-file", str(out)])


@pytest.mark.parametrize("field,change", [
    ("references", lambda r: r["references"].reverse()),
    ("words", lambda r: r["per_clip"][1]["words"].append("salt")),
    ("derangement", lambda r: r.update(words_derangement=r["words_derangement"][::-1])),
    ("overlap", lambda r: r["crossfade"].update(frames=3)),
    ("typical", lambda r: r["typical_seconds"].update({k: v + 1 for k, v in r["typical_seconds"].items()})),
    ("sign", lambda r: r["per_clip"][2].update(sign="another-clip")),
    ("shuffle", lambda r: r["per_clip"][next(i for i, c in enumerate(r["per_clip"]) if len(c["words"]) > 1)]
     ["shuffled_order"].reverse()),
    ("mismatched_from", lambda r: r["per_clip"][0].update(mismatched_from=0)),
    ("reference_derangement", lambda r: r.update(reference_derangement=r["reference_derangement"][1:]
                                                 + r["reference_derangement"][:1])),
    ("clips", lambda r: r.update(clips=r["clips"] + 1)),
])
def test_the_run_refuses_inputs_other_than_the_stored_reports(tmp_path, stubbed, field, change):  # noqa: F811
    store, sha, _ = build_store(tmp_path / "store", corpus="asllvd-text-v1", train_texts=TRAIN)
    source_report(store, sha, tamper=change)
    with pytest.raises(SystemExit, match="not the stored round-15 inputs"):
        ro.main(["--source-report-key", "source.json", "--store", f"dir:{tmp_path / 'store'}", "--bootstrap", "20",
                 "--cache-dir", str(tmp_path / "cache"), "--output-file", str(tmp_path / "x.json")])
