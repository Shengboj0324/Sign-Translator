"""Teacher-forced word retrieval (round 14b pilot add-on): ranks, candidates, clip draws and bootstrap; the part-valid
score against the repository's likelihood; an end-to-end run on tiny priors that reads only val clips of the named
corpora, pairs the two checkpoints per clip, and writes once."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import teacher_forced_word_retrieval as tf  # noqa: E402
from signtranslator.data.multichannel import collate_multichannel  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import KINDS, canonical_json, usable_channels  # noqa: E402
from signtranslator.generation.motion_rvq import MotionRVQ, RVQPrior, RVQPriorConfig, rvq_sequence_log_likelihood  # noqa: E402
from signtranslator.generation.text_to_motion import PAD, encode_texts  # noqa: E402
from signtranslator.pose.multichannel import CHANNELS  # noqa: E402
from tests.test_generation_motion_rvq import small  # noqa: E402
from tests.test_generation_signing_space import signer_state  # noqa: E402


# ---- pure parts ----------------------------------------------------------------------------------------------------

def test_rank_counts_ties_against_the_own_word():
    assert tf.rank_own([0., -1., -2.]) == 1 and tf.rank_own([-1., 0., -1.]) == 3 and tf.rank_own([-1., -1.]) == 2
    assert tf.rank_own([-3., -1., -2., -5.]) == 3


def test_candidates_follow_the_diagnosis_order():
    words = ["CAT", "DOG", "ICE", "FALL", "GO"]
    order = tf.candidate_order(words, "r14diag")
    assert order == sorted(words, key=lambda w: hashlib.sha256(f"r14diag-{w}".encode()).hexdigest())
    assert tf.candidate_order(list(reversed(words)), "r14diag") == order
    c = tf.candidates("ICE", order, 3)
    assert c[0] == "ICE" and c[1:] == [w for w in order if w != "ICE"][:2] and len(set(c)) == 3
    with pytest.raises(ValueError):
        tf.candidates("ZEBRA", order, 3)


def test_select_draws_per_word_as_the_diagnosis():
    records = [dict(corpus="a", text=t) for t in ("dog", "Dog", "cat", "dog", "ice", "thank you")] + \
        [dict(corpus="b", text="dog")]
    assert tf.select(records, ("a",), {"DOG", "THANKYOU", "ICE"}, 0, 14) == [(0, "DOG"), (1, "DOG"), (3, "DOG"),
                                                                           (4, "ICE"), (5, "THANKYOU")]
    one = tf.select(records, ("a",), {"DOG"}, 1, 14)
    assert one == [(sorted(random.Random("DOG-14").sample([0, 1, 3], 1))[0], "DOG")]
    assert tf.select(records, ("a", "b"), {"DOG"}, 5, 14) == [(k, "DOG") for k in (0, 1, 3, 6)]


def test_bootstrap_by_clips_and_by_word_clusters():
    assert tf.bootstrap([], n=10, seed=0, name="x") is None
    b = tf.bootstrap([1., 1., 1.], n=50, seed=0, name="x")
    assert b == dict(mean=1., ci95=[1., 1.], n=3)
    x, groups = [1.] * 9 + [0.], ["a"] * 9 + ["b"]
    clips, words = (tf.bootstrap(x, n=2000, seed=0, name="x"), tf.bootstrap(x, groups, n=2000, seed=0, name="x"))
    assert clips["mean"] == words["mean"] == .9 and words["groups"] == 2
    assert words["ci95"][0] == 0. and words["ci95"][1] == 1.            # two clusters: all-b draws and all-a draws
    assert clips["ci95"][0] > .6
    assert tf.bootstrap(x, groups, n=200, seed=0, name="x") == tf.bootstrap(x, groups, n=200, seed=0, name="x")
    y = np.random.default_rng(3).normal(size=40).tolist()
    assert tf.bootstrap(y, n=500, seed=0, name="x")["ci95"] != tf.bootstrap(y, n=500, seed=0, name="y")["ci95"]


def test_read_split_refuses_overlap_and_repeats():
    assert tf.read_split(json.dumps(dict(covered=["A"], uncovered=["B"])).encode()) == (["A"], ["B"])
    for bad in (dict(covered=["A"], uncovered=["A"]), dict(covered=["A", "A"], uncovered=["B"]),
                dict(covered=[], uncovered=["B"]), dict(uncovered=["B"])):
        with pytest.raises(SystemExit):
            tf.read_split(json.dumps(bad).encode())


# ---- the score ----------------------------------------------------------------------------------------------------

def tiny(seed, tokenizer_seed=0):
    torch.manual_seed(tokenizer_seed)
    state = signer_state(0, frames=16, amplitude=.4)
    tokenizer = MotionRVQ(small(state, down_layers=0)).eval()
    torch.manual_seed(seed)
    config = RVQPriorConfig(tokenizer.config, d_model=32, heads=4, text_layers=1, temporal_layers=1, depth_layers=1,
                            feedforward=64, dropout=0., locus_conditioning=True)
    return tokenizer, RVQPrior(config).eval()


def _batch(state, texts):
    from signtranslator.generation.motion_tokens import _empty_like
    cand, _ = encode_texts(texts)
    tokens = torch.cat((cand, _empty_like(cand[:1])), 0)
    return tokens, tokens != PAD, collate_multichannel([state] * len(tokens))


def _with(state, **valid):
    """``state`` with the named channels' validity set (True: every frame, values set to the identity rotation)."""
    channels = dict(state.channels)
    for name, on in valid.items():
        ch = channels[name]
        values = np.tile([1., 0., 0., 0., 1., 0.], ch.values.shape[:-1] + (1,)) if on else np.zeros_like(ch.values)
        mask = np.ones_like(ch.valid) if on else np.zeros_like(ch.valid)
        channels[name] = replace(ch, values=values, valid=mask, observed=mask.copy(), confidence=mask.astype(float))
    return replace(state, channels=channels)


def test_part_valid_score_weights_exactly_the_observed_parts():
    tokenizer, prior = tiny(1)
    P, (parts) = len(tokenizer.parts), [p for p, _ in tokenizer.parts]
    assert parts == ["body", "left_hand", "right_hand", "face"]                 # the face part reads head rotation
    full = _with(signer_state(3, frames=14, amplitude=.4), head_rotation=True)
    tokens, mask, motion = _batch(full, ["dog", "cat"])
    total, weighted, n_valid, steps, J = tf.part_valid_scores(prior, tokenizer, tokens, mask, motion, None)
    ref = rvq_sequence_log_likelihood(prior, tokenizer, tokens, mask, motion, None).double().numpy()
    np.testing.assert_allclose(total, ref, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(weighted, total, rtol=1e-6, atol=1e-4)          # every part observed: weight 1
    assert n_valid == steps * J and steps == 14
    # hands unobserved: only the body and face codes (j mod P in {0, 3}) count; the sum keeps fewer negative terms
    blind = _with(full, left_hand=False, right_hand=False)
    tokens, mask, motion = _batch(blind, ["dog", "cat"])
    total, weighted, n_valid, steps, J = tf.part_valid_scores(prior, tokenizer, tokens, mask, motion, None)
    assert n_valid == steps * J * 2 / P and np.all(weighted > total)
    np.testing.assert_allclose(total, rvq_sequence_log_likelihood(prior, tokenizer, tokens, mask, motion,
                                                                  None).double().numpy(), rtol=1e-6, atol=1e-4)


# ---- end to end ---------------------------------------------------------------------------------------------------

SPLIT = dict(covered=["CAT", "DOG", "GO"], uncovered=["ICE", "FALL", "MUMMY"])


def build_store(root: Path):
    store = DirectoryStore(root)
    records, k = [], [0]

    def record(text, partition, corpus, frames=14, stored=True, **extra):
        k[0] += 1
        state = signer_state(10 + k[0], frames=frames, amplitude=.4)
        key = f"states/{k[0]}.npz"
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "s.npz"
            sha = state.save(path)
            if stored:                                  # a record that must never be read has no stored state
                store.put(key, path.read_bytes())
        r = dict(sign=f"clip{k[0]}", corpus=corpus, partition=partition, text=text, signer_component="c01",
                 continuous=dict(status="converted", state_key=key, state_sha256=sha, available_channels=list(CHANNELS),
                                 seconds=frames / 24.), **extra)
        r["usable"] = {kind: list(usable_channels(r, kind)) for kind in KINDS}
        records.append(r)
    record("the start", "train", "how2sign-two-view-text-v1", 12)                 # the layout template
    record("a sentence", "val", "how2sign-two-view-text-v1", stored=False)          # never read
    record("a sentence", "heldout_official", "how2sign-two-view-text-v1", stored=False)
    for text, sid in (("ice", "8"), ("ice", "9"), ("fall", "8")):
        record(text, "val", "semlex-text-v2", source=dict(signer_id=sid))
    record("ice", "train", "semlex-text-v2", stored=False, source=dict(signer_id="8"))   # train: counted, not read
    record("ice", "test", "semlex-text-v2", stored=False, source=dict(signer_id="8"))
    record("ice", "val", "semlex-text-v1", stored=False, source=dict(signer_id="8"))     # another Sem-Lex corpus
    record("dog", "val", "asllvd-text-v1")
    record("dog", "val", "rit-text-r6")
    record("cat", "val", "wlasl-text-r6")
    record("cat", "test", "asllvd-text-v1", stored=False)
    record("go", "train", "asllvd-text-v1", stored=False)
    payload = canonical_json(dict(records=records))
    store.put("mix.json", payload)
    split = json.dumps(SPLIT).encode()
    store.put("split.json", split)
    for name, seed, tseed in (("a.pt", 1, 0), ("b.pt", 2, 0), ("c.pt", 3, 5)):     # c: another tokenizer
        tokenizer, prior = tiny(seed, tseed)
        buffer = io.BytesIO()
        torch.save(dict(kind="motion_rvq_prior", config=prior.config.to_dict(), model=prior.state_dict(), ema=None,
                        tokenizer_state=tokenizer.state_dict(), step=seed), buffer)
        store.put(name, buffer.getvalue())
    return store, hashlib.sha256(payload).hexdigest(), hashlib.sha256(split).hexdigest()


def argv(root, mix_sha, split_sha, *extra, checkpoint="a.pt", comparison="b.pt"):
    return ["--store", f"dir:{root}", "--checkpoint-key", checkpoint, "--compare-checkpoint-key", comparison,
            "--mix-key", "mix.json", "--mix-sha256", mix_sha, "--word-split-key", "split.json",
            "--word-split-sha256", split_sha, "--candidates", "4", "--covered-per-word", "2", "--bootstrap", "50",
            "--device", "cpu", "--cache-dir", str(root / "cache"), *extra]


def test_end_to_end_reads_only_val_clips_and_pairs_the_checkpoints(tmp_path):
    root = tmp_path / "store"
    store, mix_sha, split_sha = build_store(root)
    assert tf.main(argv(root, mix_sha, split_sha, "--output-key", "out/report.json",
                        "--output-file", str(tmp_path / "r.json"))) == 0
    report = json.loads((root / "out/report.json").read_text())
    assert (tmp_path / "r.json").read_bytes() == (root / "out/report.json").read_bytes()
    assert report["measure"].startswith("**Measure.**") and report["same_tokenizer"] is True
    assert report["checkpoints"]["checkpoint"]["key"] == "a.pt" and report["checkpoints"]["comparison"]["step"] == 2
    assert report["inputs"]["word_split_sha256"] == split_sha and report["inputs"]["template"]["partition"] == "train"
    unc, cov = report["sets"]["uncovered"], report["sets"]["covered"]
    # uncovered: the three semlex-text-v2 val clips (not v1, train or test); ICE has a v2 train record, FALL none
    assert unc["clips"] == 3 and [r["word"] for r in unc["rows"]] == ["FALL", "ICE", "ICE"]
    assert {r["corpus"] for r in unc["rows"]} == {"semlex-text-v2"} and unc["signers"] == 2
    assert unc["newly_covered"]["clips"] == 2 and unc["still_uncovered"]["clips"] == 1
    assert unc["words_without_val_clips"] == ["MUMMY"]
    # covered: two DOG val clips (ASLLVD, RIT) and the CAT val clip; not the CAT test clip or GO's train clip
    assert cov["clips"] == 3 and sorted(r["word"] for r in cov["rows"]) == ["CAT", "DOG", "DOG"]
    assert cov["chance"] == dict(top1=.25, top5=1., mean_rank=2.5)
    for s in (unc, cov):
        for r in s["rows"]:
            assert r["candidates"][0] == r["word"] and len(r["candidates"]) == 4
            for role in ("checkpoint", "comparison"):
                assert 1 <= r[role]["rank"] <= 4 and len(r[role]["part_valid_scores"]) == 4
                assert r[role]["rank"] == tf.rank_own(r[role]["part_valid_scores"])
        assert all(s["rows"][i]["checkpoint_checked_vs_rvq_sequence_log_likelihood"] for i in range(2))
        d = s["paired_checkpoint_minus_comparison"]["mean_rank"]["clips"]["mean"]
        assert d == pytest.approx(np.mean([r["checkpoint"]["rank"] - r["comparison"]["rank"] for r in s["rows"]]),
                                  abs=1e-4)
        top5 = s["checkpoint"]["top5"]["mean"]
        assert top5 == 1.                                                        # 4 candidates: always in the top 5
    # written once: the key and the file
    with pytest.raises(SystemExit, match="exists: refusing"):
        tf.main(argv(root, mix_sha, split_sha, "--output-key", "out/report.json"))
    with pytest.raises(SystemExit, match="exists: refusing"):
        tf.main(argv(root, mix_sha, split_sha, "--output-file", str(tmp_path / "r.json")))


def test_a_checkpoint_against_itself_pairs_to_zero(tmp_path):
    root = tmp_path / "store"
    _, mix_sha, split_sha = build_store(root)
    assert tf.main(argv(root, mix_sha, split_sha, "--output-file", str(tmp_path / "self.json"),
                        "--limit", "2", comparison="a.pt")) == 0
    report = json.loads((tmp_path / "self.json").read_text())
    assert report["same_tokenizer"] is True and report["parameters"]["limit"] == 2
    for s in report["sets"].values():
        assert s["clips"] == 2
        for stat in s["paired_checkpoint_minus_comparison"].values():
            assert stat["word_cluster"]["mean"] == 0 and stat["word_cluster"]["ci95"] == [0, 0]
        for r in s["rows"]:
            assert r["checkpoint"] == r["comparison"]


def test_refusals(tmp_path):
    root = tmp_path / "store"
    _, mix_sha, split_sha = build_store(root)
    out = ["--output-file", str(tmp_path / "x.json")]
    with pytest.raises(SystemExit, match="split.json: SHA-256"):
        tf.main(argv(root, mix_sha, "0" * 64, *out))
    with pytest.raises(SystemExit, match="a.pt: SHA-256"):
        tf.main(argv(root, mix_sha, split_sha, *out, "--checkpoint-sha256", "0" * 64))
    with pytest.raises(SystemExit, match="the mix lacks"):
        tf.main(argv(root, mix_sha, split_sha, *out, "--semlex-corpus", "semlex-text-v3"))
    with pytest.raises(SystemExit, match="tokenizers differ"):                  # codes not comparable: refused
        tf.main(argv(root, mix_sha, split_sha, *out, comparison="c.pt"))
    with pytest.raises(SystemExit, match="the split has 6 words"):
        tf.main(argv(root, mix_sha, split_sha, *out, "--candidates", "7"))
    with pytest.raises(SystemExit):
        tf.main(argv(root, mix_sha, split_sha))                                  # no output named
    assert not (tmp_path / "x.json").exists()
