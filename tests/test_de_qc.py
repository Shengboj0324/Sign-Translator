"""Adversarial tests for per-tier agreement + QC sampling (Doc-10, stage 10e)."""

from signtranslator.data_engineering.qc import (
    per_tier_kappa, pooled_kappa, weakest_tier, stratify, stratified_qc_sample,
)


def test_per_tier_kappa_computed_independently():
    ratings = {
        "gloss":      ([0, 1, 0, 1, 0], [0, 1, 0, 1, 0]),       # perfect
        "nonmanual":  ([0, 1, 0, 1, 0], [1, 0, 1, 0, 1]),       # anti-agreement
    }
    k = per_tier_kappa(ratings)
    assert abs(k["gloss"] - 1.0) < 1e-9
    assert k["nonmanual"] < 0.0


def test_corpus_average_hides_weak_tier():
    # gloss perfect (kappa=1), discourse poor: the per-tier report exposes it;
    # a single pooled number sits in between and masks the weak tier.
    ratings = {
        "gloss":     ([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]),
        "discourse": ([0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]),
    }
    per = per_tier_kappa(ratings)
    tier, val = weakest_tier(per)
    assert tier == "discourse" and val < 0.0
    pooled = pooled_kappa(ratings)
    assert pooled > val                      # pooling hides the weak tier


def test_stratify_partitions_all_items():
    items = [("A", 1), ("B", 2), ("A", 3), ("C", 4)]
    strata = stratify(items, key=lambda x: x[0])
    assert strata["A"] == [0, 2] and strata["B"] == [1] and strata["C"] == [3]
    assert sum(len(v) for v in strata.values()) == len(items)


def test_stratified_sample_covers_every_stratum():
    # 4 signers, unbalanced counts; every signer must be represented.
    items = ([("s1",)] * 10 + [("s2",)] * 3 + [("s3",)] * 1 + [("s4",)] * 6)
    picks = stratified_qc_sample(items, key=lambda x: x[0], k_per_stratum=2, seed=1)
    covered = {items[i][0] for i in picks}
    assert covered == {"s1", "s2", "s3", "s4"}
    # rare stratum s3 (1 item) still contributes exactly 1.
    assert sum(1 for i in picks if items[i][0] == "s3") == 1
    assert sum(1 for i in picks if items[i][0] == "s1") == 2


def test_stratified_sample_deterministic():
    items = [("s1",)] * 5 + [("s2",)] * 5
    a = stratified_qc_sample(items, key=lambda x: x[0], k_per_stratum=2, seed=7)
    b = stratified_qc_sample(items, key=lambda x: x[0], k_per_stratum=2, seed=7)
    assert a == b
