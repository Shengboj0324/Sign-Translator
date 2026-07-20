"""Adversarial tests for deduplication (Doc-10, stage 10d)."""

import numpy as np

from signtranslator.data_engineering.dedup import (
    average_hash, difference_hash, hamming_distance, jaccard_similarity,
    normalized_edit_distance, cluster_duplicates, near_threshold_pairs,
)

rng = np.random.default_rng(0)


def test_identical_images_hash_equal():
    img = rng.random((64, 48))
    assert average_hash(img) == average_hash(img.copy())
    assert difference_hash(img) == difference_hash(img.copy())
    assert hamming_distance(average_hash(img), average_hash(img.copy())) == 0


def test_small_perturbation_small_hamming():
    img = rng.random((32, 32))
    noisy = img + rng.normal(0, 1e-3, img.shape)
    assert hamming_distance(average_hash(img), average_hash(noisy)) <= 2


def test_distinct_images_large_hamming():
    a = rng.random((32, 32))
    b = rng.random((32, 32))                             # independent content
    assert hamming_distance(average_hash(a), average_hash(b)) > 10


def test_hash_is_64_bits():
    img = rng.random((40, 40))
    assert 0 <= average_hash(img) < (1 << 64)
    assert 0 <= difference_hash(img) < (1 << 64)


def test_jaccard_bounds_and_identity():
    a = "the cat sat on the mat".split()
    assert jaccard_similarity(a, a) == 1.0
    assert jaccard_similarity(a, ["xyz"]) == 0.0
    j = jaccard_similarity(a, "the dog sat".split())
    assert 0.0 < j < 1.0


def test_jaccard_bigrams_stricter_than_unigrams():
    a = "a b c d".split()
    b = "d c b a".split()                                # same tokens, new order
    assert jaccard_similarity(a, b, n=1) == 1.0
    assert jaccard_similarity(a, b, n=2) < 1.0           # bigrams differ


def test_edit_distance_bounds():
    assert normalized_edit_distance([], []) == 0.0
    a = "one two three".split()
    assert normalized_edit_distance(a, a) == 0.0
    assert normalized_edit_distance(a, "four five six".split()) == 1.0
    d = normalized_edit_distance(a, "one two four".split())
    assert abs(d - 1 / 3) < 1e-12


def test_cluster_groups_near_duplicates():
    base = rng.random((32, 32))
    imgs = [base, base + rng.normal(0, 1e-3, base.shape),   # near-dup pair
            np.tile(np.linspace(0, 1, 32), (32, 1))]        # distinct
    hs = [average_hash(x) for x in imgs]
    groups = cluster_duplicates(hs, threshold=2)
    assert [0, 1] in groups and [2] in groups


def test_near_threshold_pairs_flagged_for_review():
    # two hashes exactly 4 bits apart => flagged when tau=4.
    h0 = 0
    h1 = 0b1111                                          # distance 4 from h0
    pairs = near_threshold_pairs([h0, h1], tau=4, delta=1)
    assert pairs == [(0, 1, 4)]
    assert near_threshold_pairs([h0, h1], tau=10, delta=1) == []
