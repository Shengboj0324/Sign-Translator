"""Adversarial tests for SacreBLEU + BERTScore (Doc-12, stage 12e)."""

import math

import pytest
import torch

from signtranslator.eval_framework.text_metrics import (
    tokenize, corpus_bleu, BLEUSignature, idf_weights, bert_score,
)


def test_identical_corpus_bleu_is_one():
    hyps = ["the cat sat on the mat", "a dog ran fast"]
    score, sig = corpus_bleu(hyps, list(hyps))
    assert score == pytest.approx(1.0, abs=1e-9)
    assert str(sig).startswith("BLEU|tok:basic_lc_v1|smooth:exp|maxn:4")


def test_bleu_brevity_penalty_for_short_hypothesis():
    # a hypothesis much shorter than the reference is penalised (< 1) even if all
    # its n-grams match.
    hyp = ["the cat"]
    ref = ["the cat sat on the mat today"]
    score, _ = corpus_bleu(hyp, ref, max_ngram=2)
    c, r = 2, 7
    assert score < 1.0
    # BP = exp(1 - r/c); precisions are 1 (both bigrams match) -> score == BP.
    assert score == pytest.approx(math.exp(1 - r / c), abs=1e-9)


def test_bleu_zero_when_no_overlap_and_no_smooth():
    score, _ = corpus_bleu(["xxx yyy zzz"], ["aaa bbb ccc"], max_ngram=2,
                           smooth="none")
    assert score == 0.0


def test_bleu_smoothing_avoids_hard_zero_on_partial_match():
    # unigrams overlap, higher orders do not -> exp smoothing keeps it > 0.
    score, _ = corpus_bleu(["the cat runs"], ["the dog runs slowly"], max_ngram=4,
                           smooth="exp")
    assert 0.0 < score < 1.0


def test_signature_is_deterministic():
    _, sig1 = corpus_bleu(["a b"], ["a b"])
    _, sig2 = corpus_bleu(["c d"], ["c d"])
    assert str(sig1) == str(sig2)                       # settings-only signature


def test_bertscore_identical_embeddings_is_one():
    # exact-identity claim -> float64 (float32 self-cosine rounds to ~1.00005).
    emb = torch.randn(5, 16, dtype=torch.float64)
    s = bert_score(emb, emb.clone())
    assert s["precision"] == pytest.approx(1.0, abs=1e-12)
    assert s["recall"] == pytest.approx(1.0, abs=1e-12)
    assert s["f1"] == pytest.approx(1.0, abs=1e-12)


def test_bertscore_is_token_order_invariant():
    ref = torch.randn(4, 8)
    cand = torch.randn(6, 8)
    base = bert_score(ref, cand)
    perm_ref = ref[torch.randperm(4)]
    perm_cand = cand[torch.randperm(6)]
    permuted = bert_score(perm_ref, perm_cand)
    assert permuted["f1"] == pytest.approx(base["f1"], abs=1e-6)


def test_bertscore_f1_between_zero_and_one():
    torch.manual_seed(0)
    s = bert_score(torch.randn(3, 8), torch.randn(7, 8))
    assert 0.0 <= s["f1"] <= 1.0


def test_idf_weights_downweight_common_tokens():
    docs = [["the", "cat"], ["the", "dog"], ["the", "bird"]]
    idf = idf_weights(docs)
    assert idf["the"] < idf["cat"]                      # 'the' appears everywhere


def test_bertscore_idf_changes_score():
    ref = torch.randn(4, 8)
    cand = torch.randn(4, 8)
    plain = bert_score(ref, cand)
    weighted = bert_score(ref, cand,
                          idf_ref=torch.tensor([5.0, 1.0, 1.0, 1.0]),
                          idf_cand=torch.tensor([5.0, 1.0, 1.0, 1.0]))
    assert plain["f1"] != weighted["f1"]
