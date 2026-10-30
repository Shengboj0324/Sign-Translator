"""S0/S1 metric helpers on synthetic data: ΔNLL signs, retrieval with a planted match, ρ_faith on identical and random
pairs, KL zero without a text effect, and the fast BLEU/chrF bootstraps against the repository's own scorers."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from signtranslator.eval import s0_metrics as m  # noqa: E402


# ---- ΔNLL ----------------------------------------------------------------------------------------------------------

def test_delta_nll_is_positive_when_motion_makes_the_reference_more_likely():
    d = m.delta_nll([2., 2., 2.], [1., 2., 3.])
    assert d.tolist() == [1., 0., -1.]                  # helps, no effect, hurts
    with pytest.raises(ValueError):
        m.delta_nll([1., 2.], [1.])


def test_token_nll_ignores_padding_and_equals_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 7)
    labels = torch.tensor([[1, 2, 3, -100], [4, 5, -100, -100]])
    total, count = m.token_nll(logits, labels)
    assert count.tolist() == [3., 2.]
    want = [torch.nn.functional.cross_entropy(logits[0, :3], labels[0, :3], reduction="sum"),
            torch.nn.functional.cross_entropy(logits[1, :2], labels[1, :2], reduction="sum")]
    torch.testing.assert_close(total, torch.stack(want).double())


def test_gap_fraction_and_ratio_of_means():
    assert m.gap_fraction(3., 1., 5.) == .5 and m.gap_fraction(1., 1., 5.) == 0. and m.gap_fraction(2., 1., 1.) is None
    r = m.ratio_of_means([1., 1., 1., 1.], [2., 2., 2., 2.])
    assert r["ratio"] == .5 and r["ci95"] == [.5, .5]
    assert m.ratio_of_means([1.], [0.])["ratio"] is None


def test_bootstrap_and_paired_difference():
    b = m.bootstrap_mean([1., 1., 1.])
    assert b == dict(mean=1., ci95=[1., 1.], n=3)
    d = m.paired_mean_difference([2., 3., 4.], [1., 2., 3.])
    assert d == dict(difference=1., ci95=[1., 1.])
    assert m.bootstrap_mean([])["mean"] is None


# ---- retrieval -----------------------------------------------------------------------------------------------------

def test_candidates_hold_the_clip_first_and_distinct_others():
    c = m.retrieval_candidates(50, 10, seed=3)
    assert c.shape == (50, 10) and (c[:, 0] == np.arange(50)).all()
    for i, row in enumerate(c):
        assert len(set(row.tolist())) == 10 and i not in row[1:]
    assert (m.retrieval_candidates(50, 10, seed=3) == c).all()
    assert m.retrieval_candidates(5, 100).shape == (5, 5)                  # k capped at n
    # a row's draw does not depend on the other rows
    assert (m.retrieval_candidates(60, 10, seed=3)[7, 0] == 7)


def test_planted_match_ranks_first_and_ties_take_the_mid_rank():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(30, 100)) + 5.
    scores[:, 0] = 0.                                                       # the own reference: lowest NLL
    r = m.mid_ranks(scores)
    assert (r == 1).all()
    s = m.retrieval_summary(r, 100)
    assert s["top1"] == 1. and s["mrr"] == 1. and s["chance"]["top1"] == .01
    flat = m.mid_ranks(np.zeros((4, 100)))                                  # no information: the chance mean
    assert (flat == 50.5).all()
    assert m.mid_ranks(np.array([[1., 0., 1., 2.]])).tolist() == [2.5]      # one better, one tied
    assert m.mid_ranks(np.array([[1., 0., 2.]]), lower_is_better=False).tolist() == [2.]


def test_random_scores_retrieve_at_chance():
    rng = np.random.default_rng(1)
    r = m.mid_ranks(rng.normal(size=(4000, 100)))
    s = m.retrieval_summary(r, 100)
    assert abs(s["top1"] - .01) < .006 and abs(s["mean_rank"] - 50.5) < 1.5


# ---- chrF and BLEU -------------------------------------------------------------------------------------------------

def test_chrf_matches_hand_computed_values():
    assert m.corpus_chrf(["abc"], ["abc"]) == 100.
    assert m.corpus_chrf(["xyz"], ["abc"]) == 0.
    # "ab" vs "abc": orders 1-2 have both sides; precision 1, recall 2/3 and 1/2
    f = lambda p, r: 5 * p * r / (4 * p + r)
    want = 100 * (f(1, 2 / 3) + f(1, 1 / 2)) / 2
    assert m.corpus_chrf(["ab"], ["a b c"]) == pytest.approx(want)
    stats = m.sentence_statistics(["ab", "abc"], ["abc", "abc"])
    assert np.allclose(m.chrf_from_statistics(np.stack([stats.sum(0)] * 3)), m.chrf_from_statistics(stats.sum(0)))


def test_fast_bleu_equals_the_repository_scorers():
    from signtranslator.generation.motion_to_text import corpus_bleu
    from strict_prior_review import bleu_ci
    from compare_pilot_lexicon import paired_bleu_difference
    rng = np.random.default_rng(0)
    vocab = "the a dog cat runs sat on mat house you i we".split()
    refs = [" ".join(rng.choice(vocab, rng.integers(3, 9))) for _ in range(40)]
    hyps = [" ".join(rng.choice(vocab, rng.integers(2, 9))) for _ in range(40)]
    other = [" ".join(rng.choice(vocab, rng.integers(2, 9))) for _ in range(40)]
    stats = m.bleu_statistics(hyps, refs)
    assert m.bleu_from_statistics(stats.sum(0)) == pytest.approx(corpus_bleu(hyps, refs)["bleu"])
    assert m.bleu_ci_fast(hyps, refs, n=200) == bleu_ci(hyps, refs, n=200)
    slow = paired_bleu_difference(hyps, other, refs, resamples=300)
    fast = m.paired_bleu_fast(stats, m.bleu_statistics(other, refs), n=300)
    assert fast["difference"] == pytest.approx(slow[0], abs=1e-4)
    assert fast["ci95"] == pytest.approx(list(slow[1]), abs=1e-4)
    c = corpus_bleu(hyps, refs)
    assert m.length_ratio(stats) == round(c["hypothesis_length"] / c["reference_length"], 4)


# ---- ρ_faith -------------------------------------------------------------------------------------------------------

def test_batched_dtw_equals_the_plain_recursion():
    rng = np.random.default_rng(0)
    a = [rng.normal(size=(n, 3)) for n in (5, 9, 1, 7)]
    b = [rng.normal(size=(n, 3)) for n in (8, 9, 4, 2)]
    fast = m.dtw_distances(a, b, chunk=3)
    assert fast == pytest.approx([m.dtw_reference(x, y) for x, y in zip(a, b)])
    assert m.dtw_distances([a[0]], [a[0]])[0] == pytest.approx(0.)
    # a time-stretched copy aligns at zero cost
    x = rng.normal(size=(6, 3))
    assert m.dtw_distances([x], [np.repeat(x, 2, 0)])[0] == pytest.approx(0.)


def test_rho_faith_is_zero_for_identical_pairs_and_near_one_for_random_ones():
    rng = np.random.default_rng(0)
    refs = [rng.normal(size=(rng.integers(8, 14), 4)) for _ in range(40)]
    same = m.rho_faith(refs, refs, K=5)
    assert same["mean"] == 0. and same["K"] == 5
    unrelated = [rng.normal(size=(rng.integers(8, 14), 4)) for _ in range(40)]
    rand = m.rho_faith(unrelated, refs, K=5)
    assert .9 < rand["mean"] < 1.1
    near = m.rho_faith([r + .1 * rng.normal(size=r.shape) for r in refs], refs, K=5)
    assert near["mean"] < .5 and near["ci95"][1] < rand["ci95"][0]
    mean_variant = m.rho_faith(refs, refs, K=5, distance="mean")
    assert mean_variant["mean"] == 0.


def test_random_references_never_pick_the_clip():
    r = m.random_references(12, 4, seed=1)
    assert r.shape == (12, 4) and all(i not in row for i, row in enumerate(r))
    assert m.random_references(3, 10).shape == (3, 2)


def test_standardiser_uses_every_frame():
    mean, std = m.standardiser([np.array([[0., 1.], [2., 1.]]), np.array([[4., 1.]])])
    assert mean.tolist() == [2., 1.] and std[0] == pytest.approx(np.std([0., 2., 4.])) and std[1] == 1e-6


# ---- KL ------------------------------------------------------------------------------------------------------------

def test_kl_is_zero_for_equal_logits_and_matches_the_definition():
    torch.manual_seed(0)
    a = torch.randn(5, 3, 11)
    assert torch.allclose(m.categorical_kl(a, a), torch.zeros(5, 3, dtype=torch.float64))
    assert torch.allclose(m.categorical_kl(a, a + 7.), torch.zeros(5, 3, dtype=torch.float64))   # shift-invariant
    b = torch.randn(5, 3, 11)
    p, q = a.double().softmax(-1), b.double().softmax(-1)
    torch.testing.assert_close(m.categorical_kl(a, b), (p * (p / q).log()).sum(-1))
    assert (m.categorical_kl(a, b) > 0).all()
    target = torch.randint(0, 11, (5, 3))
    gain = m.true_code_gain(a, b, target)
    torch.testing.assert_close(gain, p.log().gather(-1, target[..., None])[..., 0] - q.log().gather(-1, target[..., None])[..., 0])
