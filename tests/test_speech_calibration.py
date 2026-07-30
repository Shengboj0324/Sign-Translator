"""Verification of calibration metrics, temperature scaling, and Brier loss.

Two mathematical properties are proved rather than assumed:
  * the Brier score is a **strictly proper** scoring rule (minimised uniquely at
    the true distribution), and
  * Murphy's decomposition ``BS = REL - RES + UNC`` holds **exactly** when
    predictions are grouped by unique value.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from signtranslator.speech.calibration import (
    brier_score, binary_brier_score, brier_decomposition, BrierDecomposition,
    reliability_diagram, expected_calibration_error, maximum_calibration_error,
    negative_log_likelihood, TemperatureScaler, BrierLoss, CalibrationBin,
)


# ---------------------------------------------------------------------------
# Brier score: propriety
# ---------------------------------------------------------------------------
def test_brier_is_zero_for_perfect_prediction():
    probs = torch.eye(4)
    labels = torch.arange(4)
    assert abs(brier_score(probs, labels)) < 1e-12


def test_brier_is_two_for_confidently_wrong_prediction():
    """Maximum multiclass Brier: all mass on a wrong class -> 1 + 1 = 2."""
    probs = torch.tensor([[0.0, 1.0]])
    labels = torch.tensor([0])
    assert abs(brier_score(probs, labels) - 2.0) < 1e-12


def test_brier_is_a_strictly_proper_scoring_rule():
    """E_{y~q}[BS(p,y)] must be minimised UNIQUELY at p = q.

    This is the property that makes Brier a valid calibration objective:
    optimising it forces the model to report its true uncertainty. Accuracy has
    no such property -- it is invariant to any monotone distortion of p.
    """
    torch.manual_seed(0)
    q = F.softmax(torch.randn(5, dtype=torch.float64), dim=0)

    def expected_brier(p):
        # E_{y~q} sum_k (p_k - [k==y])^2  =  sum_k p_k^2 - 2 sum_c q_c p_c + 1
        return float(p.pow(2).sum() - 2 * (q * p).sum() + 1)

    at_truth = expected_brier(q)
    for _ in range(200):
        noise = torch.randn(5, dtype=torch.float64) * 0.3
        p = F.softmax(q.log() + noise, dim=0)
        if torch.allclose(p, q, atol=1e-9):
            continue
        assert expected_brier(p) > at_truth - 1e-12, "p=q was not the minimiser"


def test_brier_validates_shapes():
    with pytest.raises(ValueError):
        brier_score(torch.rand(4), torch.arange(4))
    with pytest.raises(ValueError):
        brier_score(torch.rand(4, 3), torch.arange(5))
    with pytest.raises(ValueError):
        binary_brier_score([0.5, 0.5], [True])
    with pytest.raises(ValueError):
        binary_brier_score([], [])


# ---------------------------------------------------------------------------
# Murphy decomposition
# ---------------------------------------------------------------------------
def test_murphy_decomposition_is_exact():
    """BS = REL - RES + UNC, to float precision, with unique-value grouping."""
    torch.manual_seed(1)
    conf = torch.tensor([0.1, 0.1, 0.4, 0.4, 0.4, 0.9, 0.9, 0.9, 0.9, 0.6],
                        dtype=torch.float64)
    correct = torch.tensor([0, 0, 1, 0, 0, 1, 1, 1, 0, 1]).bool()
    d = brier_decomposition(conf, correct)
    assert abs(d.reconstructed - d.brier) < 1e-12


def test_decomposition_on_perfectly_calibrated_predictions():
    """Confidence equal to the observed frequency in each group => REL = 0."""
    # group at 0.5: half correct; group at 1.0: all correct
    conf = torch.tensor([0.5] * 4 + [1.0] * 4, dtype=torch.float64)
    correct = torch.tensor([1, 1, 0, 0] + [1, 1, 1, 1]).bool()
    d = brier_decomposition(conf, correct)
    assert d.reliability < 1e-12
    assert abs(d.reconstructed - d.brier) < 1e-12


def test_uninformative_but_calibrated_predictor_has_zero_resolution():
    """Always predicting the base rate is calibrated AND useless.

    REL = 0 but RES = 0 too. Reporting only ECE would call this a good model,
    which is exactly why the decomposition is worth having.
    """
    correct = torch.tensor([1, 0, 1, 0, 1, 0]).bool()
    base = float(correct.double().mean())
    conf = torch.full((6,), base, dtype=torch.float64)
    d = brier_decomposition(conf, correct)
    assert d.reliability < 1e-12
    assert d.resolution < 1e-12
    assert abs(d.brier - d.uncertainty) < 1e-12


def test_uncertainty_depends_only_on_base_rate():
    correct = torch.tensor([1, 1, 1, 0]).bool()
    d = brier_decomposition(torch.tensor([0.3, 0.6, 0.9, 0.2]), correct)
    assert abs(d.uncertainty - 0.75 * 0.25) < 1e-12


# ---------------------------------------------------------------------------
# ECE / reliability
# ---------------------------------------------------------------------------
def test_ece_is_zero_when_confidence_equals_accuracy():
    """Perfectly calibrated by construction: 70% confident, 70% correct."""
    conf = torch.full((100,), 0.7, dtype=torch.float64)
    correct = torch.tensor([1] * 70 + [0] * 30).bool()
    assert expected_calibration_error(conf, correct, n_bins=10) < 1e-12


def test_ece_detects_maximal_overconfidence():
    """Always 100% confident, 50% correct -> ECE = 0.5 exactly."""
    conf = torch.ones(100, dtype=torch.float64)
    correct = torch.tensor([1] * 50 + [0] * 50).bool()
    assert abs(expected_calibration_error(conf, correct, n_bins=10) - 0.5) < 1e-12


def test_ece_detects_underconfidence_too():
    """Calibration error is symmetric: being too humble is also miscalibrated."""
    conf = torch.full((50,), 0.2, dtype=torch.float64)
    correct = torch.ones(50).bool()                 # always right
    assert abs(expected_calibration_error(conf, correct, n_bins=10) - 0.8) < 1e-12


def test_mce_reports_the_worst_bin_not_the_average():
    # One tiny, badly-miscalibrated bin among many good ones.
    conf = torch.cat([torch.full((99,), 0.5, dtype=torch.float64),
                      torch.tensor([0.95], dtype=torch.float64)])
    correct = torch.tensor([1] * 50 + [0] * 49 + [0]).bool()
    ece = expected_calibration_error(conf, correct, n_bins=10)
    mce = maximum_calibration_error(conf, correct, n_bins=10)
    assert mce > ece                                 # worst case exceeds average
    assert mce >= 0.9


def test_reliability_bins_partition_the_samples():
    torch.manual_seed(2)
    conf = torch.rand(200, dtype=torch.float64)
    correct = torch.rand(200) < conf
    bins = reliability_diagram(conf, correct, n_bins=10)
    assert sum(b.count for b in bins) == 200         # every sample counted once
    for b in bins:
        assert b.lower <= b.mean_confidence <= b.upper + 1e-12
        assert 0.0 <= b.accuracy <= 1.0


def test_quantile_binning_balances_bin_masses():
    """With confidences piled near 1, uniform bins are lopsided; quantile aren't."""
    conf = torch.cat([torch.full((90,), 0.99, dtype=torch.float64),
                      torch.linspace(0.0, 0.5, 10, dtype=torch.float64)])
    correct = torch.ones(100).bool()
    q_bins = reliability_diagram(conf, correct, n_bins=5, strategy="quantile")
    u_bins = reliability_diagram(conf, correct, n_bins=5, strategy="uniform")
    spread = lambda bs: max(b.count for b in bs) - min(b.count for b in bs)
    assert spread(q_bins) <= spread(u_bins)


def test_calibration_validates_arguments():
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [True, False])
    with pytest.raises(ValueError):
        expected_calibration_error([], [])
    with pytest.raises(ValueError):
        reliability_diagram([0.5], [True], n_bins=0)
    with pytest.raises(ValueError):
        reliability_diagram([0.5], [True], strategy="log")


def test_nll_matches_manual_computation():
    log_probs = F.log_softmax(torch.randn(6, 4, dtype=torch.float64), dim=-1)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])
    manual = -sum(float(log_probs[i, labels[i]]) for i in range(6)) / 6
    assert abs(negative_log_likelihood(log_probs, labels) - manual) < 1e-12


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def test_temperature_one_is_the_identity():
    scaler = TemperatureScaler(1.0)
    logits = torch.randn(8, 5, dtype=torch.float64)
    assert torch.allclose(scaler(logits), F.log_softmax(logits, dim=-1), atol=1e-12)


def test_temperature_preserves_argmax_and_accuracy():
    """T > 0 is strictly monotone, so ranking -- hence accuracy -- cannot change.

    This is what makes post-hoc calibration safe to apply to a trained model.
    """
    torch.manual_seed(3)
    logits = torch.randn(200, 7, dtype=torch.float64)
    base = logits.argmax(dim=-1)
    for t in (0.1, 0.5, 1.0, 2.0, 10.0, 100.0):
        scaled = TemperatureScaler(t)(logits)
        assert torch.equal(scaled.argmax(dim=-1), base), f"argmax changed at T={t}"


def test_high_temperature_approaches_uniform():
    logits = torch.randn(4, 5, dtype=torch.float64)
    probs = TemperatureScaler(1e4)(logits).exp()
    assert torch.allclose(probs, torch.full_like(probs, 0.2), atol=1e-3)


def test_low_temperature_approaches_one_hot():
    logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)
    probs = TemperatureScaler(0.01)(logits).exp()
    assert probs[0, 2] > 0.999


def test_fitting_recovers_a_known_overconfidence_factor():
    """Construct logits overconfident by exactly 3x; the fit must find T ~ 3.

    Labels are sampled from the TRUE distribution while the model reports
    3x-sharpened logits, so the correct remedy is division by 3.
    """
    torch.manual_seed(4)
    true_logits = torch.randn(4000, 5, dtype=torch.float64)
    labels = torch.distributions.Categorical(
        logits=true_logits).sample()                 # labels ~ true distribution
    overconfident = true_logits * 3.0                # model is 3x too sharp

    scaler = TemperatureScaler(1.0)
    fitted = scaler.fit(overconfident, labels, max_iter=400, lr=0.05)
    assert abs(fitted - 3.0) < 0.35, f"recovered T={fitted}, expected ~3"


def test_fitting_reduces_ece_of_an_overconfident_model():
    torch.manual_seed(5)
    true_logits = torch.randn(3000, 6, dtype=torch.float64)
    labels = torch.distributions.Categorical(logits=true_logits).sample()
    overconfident = true_logits * 4.0

    def ece_of(log_probs):
        conf, pred = log_probs.exp().max(dim=-1)
        return expected_calibration_error(conf, pred == labels, n_bins=15)

    before = ece_of(F.log_softmax(overconfident, dim=-1))
    scaler = TemperatureScaler(1.0)
    scaler.fit(overconfident, labels, max_iter=400)
    after = ece_of(scaler(overconfident))
    assert after < before * 0.5, f"ECE {before:.4f} -> {after:.4f}"


def test_fitting_reduces_nll():
    torch.manual_seed(6)
    logits = torch.randn(1500, 4, dtype=torch.float64) * 3.0
    labels = torch.distributions.Categorical(
        logits=logits / 3.0).sample()
    before = negative_log_likelihood(F.log_softmax(logits, dim=-1), labels)
    scaler = TemperatureScaler(1.0)
    scaler.fit(logits, labels, max_iter=300)
    after = negative_log_likelihood(scaler(logits), labels)
    assert after <= before + 1e-9


def test_temperature_stays_positive_by_construction():
    scaler = TemperatureScaler(1.0)
    torch.manual_seed(7)
    scaler.fit(torch.randn(200, 3, dtype=torch.float64) * 0.01,
               torch.randint(0, 3, (200,)), max_iter=100)
    assert scaler.temperature > 0.0


def test_scaler_validates_arguments():
    with pytest.raises(ValueError):
        TemperatureScaler(0.0)
    with pytest.raises(ValueError):
        TemperatureScaler().fit(torch.randn(4), torch.arange(4))
    with pytest.raises(ValueError):
        TemperatureScaler().fit(torch.randn(4, 3), torch.arange(5))


# ---------------------------------------------------------------------------
# Brier loss
# ---------------------------------------------------------------------------
def test_brier_loss_matches_the_metric():
    torch.manual_seed(8)
    logits = torch.randn(10, 5)
    labels = torch.randint(0, 5, (10,))
    loss = BrierLoss(from_logits=True)(logits, labels)
    metric = brier_score(F.softmax(logits, dim=-1), labels)
    assert abs(loss.detach().item() - metric) < 1e-6


def test_brier_loss_is_differentiable_and_pushes_toward_the_label():
    logits = torch.zeros(1, 3, requires_grad=True)
    labels = torch.tensor([1])
    BrierLoss()(logits, labels).backward()
    # Gradient must decrease the loss by raising the true-class logit.
    assert logits.grad[0, 1] < 0
    assert logits.grad[0, 0] > 0 and logits.grad[0, 2] > 0


def test_brier_loss_accepts_probabilities_directly():
    # float64: the assertion is an exact algebraic identity, so the input's
    # precision must not be what the tolerance measures.
    probs = torch.tensor([[0.2, 0.8]], dtype=torch.float64)
    labels = torch.tensor([1])
    loss = BrierLoss(from_logits=False)(probs, labels)
    assert abs(loss.detach().item() - (0.2 ** 2 + 0.2 ** 2)) < 1e-12


def test_brier_loss_validates_rank():
    with pytest.raises(ValueError):
        BrierLoss()(torch.randn(5), torch.tensor([1]))


def test_optimising_brier_loss_calibrates_a_biased_predictor():
    """Fit a single logit vector to a known distribution; it must converge to it."""
    torch.manual_seed(9)
    target = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    labels = torch.distributions.Categorical(target).sample((4000,))
    logits = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.05)
    loss_fn = BrierLoss()
    for _ in range(600):
        opt.zero_grad()
        loss_fn(logits.expand(labels.numel(), 3), labels).backward()
        opt.step()
    learned = F.softmax(logits.detach(), dim=-1).flatten()
    assert torch.allclose(learned, target, atol=0.05), learned
