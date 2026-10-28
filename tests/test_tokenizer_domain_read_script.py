"""Round 14 pre-check helpers: the signer-spread sample, per-clip pairing, intervals, the declared verdict and the
conform audit."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import tokenizer_domain_read as job  # noqa: E402


def test_the_sample_is_seeded_order_free_and_spread_over_signers():
    items = [(f"a{i}", "A") for i in range(50)] + [(f"b{i}", "B") for i in range(3)] + [("c0", "C")]
    sample = job.signer_spread_sample(items, 7, seed=0)
    assert sample == job.signer_spread_sample(list(reversed(items)), 7, seed=0)      # input order does not matter
    assert len(sample) == 7 and len(set(sample)) == 7
    signers = [s for _, s in sample]
    assert signers.count("C") == 1 and signers.count("B") == 3 and signers.count("A") == 3   # round-robin, then A
    assert set(signers[:3]) == {"A", "B", "C"}                                       # every signer before repeats
    assert job.signer_spread_sample(items, 1000, seed=0).__len__() == len(items)      # never more than there is
    assert job.signer_spread_sample(items, 7, seed=1) != sample                       # the seed matters


def test_paired_differences_are_per_clip_and_refuse_misaligned_forms():
    real = np.array([0, 3, 7, 2, 9])          # ranks (0 = top-1): top-5 hits 1 1 0 1 0
    recon = np.array([6, 3, 1, 8, 9])         # top-5 hits 0 1 1 0 0
    d = job.paired_difference(recon, real, 5)
    assert d["difference"] == pytest.approx((2 - 3) / 5)
    assert (d["wins"], d["losses"], d["clips"]) == (1, 2, 5)
    lo, hi = d["ci_clips"]
    assert lo <= d["difference"] <= hi
    assert job.paired_difference(recon, real, 1)["difference"] == pytest.approx((0 - 1) / 5)
    with pytest.raises(ValueError):
        job.paired_difference(recon[:4], real, 5)
    with pytest.raises(ValueError):
        job.paired_difference(np.array([]), np.array([]), 5)


def test_pairing_differs_from_an_unpaired_comparison():
    # Same marginal hit rates, opposite pairing: the paired interval is what the bar reads.
    a = np.array([0] * 50 + [9] * 50)
    same = job.paired_difference(a, a.copy(), 5)
    assert same["difference"] == 0 and same["ci_clips"] == [0.0, 0.0]
    flipped = job.paired_difference(a, a[::-1].copy(), 5)
    assert flipped["difference"] == 0 and flipped["ci_clips"][0] < -0.1 < 0.1 < flipped["ci_clips"][1]


def test_intervals_over_signers_resample_whole_signers():
    hits = np.array([1.] * 10 + [0.] * 10 + [1.] * 10)
    groups = ["x"] * 10 + ["y"] * 10 + ["z"] * 10
    lo, hi = job.bootstrap_mean(hits, groups)
    clip_lo, clip_hi = job.bootstrap_mean(hits)
    assert 0 <= lo < clip_lo < clip_hi < hi <= 1                # 3 clusters: far wider than 30 independent clips
    assert job.bootstrap_mean(hits, groups) == [lo, hi]                               # seeded
    d = job.paired_difference(np.zeros(30), np.where(hits > 0, 0, 9), 5, groups)
    assert "ci_signers" in d and d["difference"] == pytest.approx(10 / 30, abs=1e-4)
    assert "ci_signers" not in job.paired_difference(np.zeros(4), np.zeros(4), 5, ["x", "x", "y", "y"])


def test_the_verdict_is_the_declared_strict_bound():
    assert job.BAR == -0.15
    assert job.verdict(-0.149) == "pass" and job.verdict(0.02) == "pass"
    assert job.verdict(-0.15) == "fail" and job.verdict(-0.3) == "fail"


def test_form_scores_report_both_intervals():
    ranks = np.array([0, 1, 4, 5, 20, 0])
    s = job.form_scores(ranks, ["a", "a", "b", "b", "c", "c"])
    assert s["top1"] == pytest.approx(2 / 6, abs=1e-4) and s["top5"] == pytest.approx(4 / 6, abs=1e-4)
    assert len(s["top5_ci_clips"]) == 2 and len(s["top5_ci_signers"]) == 2
    assert job.form_scores(ranks, ["a"] * 6)["top5_ci_signers"] is None


def channel(frame="f", convention="c", labels=("x",), valid=True, width=6):
    return SimpleNamespace(labels=tuple(labels), coordinate_frame=frame, units="unitless", convention=convention,
                           valid=np.full((4, len(labels)), valid), values=np.zeros((4, len(labels), width)))


def test_the_conform_audit_follows_the_layout_adapters_rules():
    (ref_frame, ref_conv), sources = next(iter(job_equivalent().items()))
    src_frame, src_conv = next(iter(sources))
    ref = channel(ref_frame, ref_conv)
    assert job.conform_kind(channel(ref_frame, ref_conv), ref) == "same"
    assert job.conform_kind(channel(src_frame, src_conv), ref) == "equivalent"
    assert job.conform_kind(channel("other", "other", valid=False), ref) == "absent"
    assert job.conform_kind(channel("other", "other"), ref) == "refused"
    assert job.conform_kind(channel(src_frame, src_conv, labels=("y",)), ref) == "refused"
    template = SimpleNamespace(channels=dict(body=ref, face=ref))
    state = SimpleNamespace(channels=dict(body=channel(src_frame, src_conv), extra=ref))
    audit = job.conform_audit(state, template)
    assert audit == dict(kinds=dict(body="equivalent", face="missing"), extra=["extra"])


def job_equivalent():
    from signtranslator.data.layout_adapter import EQUIVALENT
    return EQUIVALENT


def test_native_errors_count_only_cells_the_real_clip_observes():
    identity = np.array([1., 0., 0., 0., 1., 0.])            # columns x, y of the identity (6D, column layout)
    quarter = np.array([0., 1., 0., -1., 0., 0.])            # 90 degrees about z
    real_values = np.tile(identity, (3, 2, 1))
    recon_values = real_values.copy()
    recon_values[:, 0] = quarter                              # joint 0 wrong by 90 degrees on every frame
    recon_values[2, 1] = quarter                              # joint 1 wrong on frame 2, which real does not observe
    real_valid = np.array([[True, True], [True, True], [True, False]])
    real = SimpleNamespace(channels=dict(body=SimpleNamespace(values=real_values, valid=real_valid, labels=("P", "C")),
                                         face=SimpleNamespace(values=np.zeros((3, 4)), valid=np.ones((3, 4), bool))))
    recon = SimpleNamespace(channels=dict(body=SimpleNamespace(values=recon_values, valid=np.ones((3, 2), bool)),
                                          face=SimpleNamespace(values=np.full((3, 4), .5), valid=np.ones((3, 4), bool))))
    sums, counts = {}, {}
    job.native_errors(recon, real, sums, counts)
    assert counts == {"body": 5, "face": 12, "body/P": 3, "body/C": 2}
    assert sums["body/P"] / 3 == pytest.approx(90) and sums["body/C"] == pytest.approx(0, abs=1e-6)
    assert sums["body"] / counts["body"] == pytest.approx(3 * 90 / 5)
    assert sums["face"] / counts["face"] == pytest.approx(.5)
