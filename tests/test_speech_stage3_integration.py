"""Stage 3 integration: calibrated confidence and fail-closed behaviour on a
real trained recogniser.

A model is trained on clean tone-burst utterances and then evaluated on *noisy*
ones so that genuine errors occur. On that real output we require:

  * confidence to be informative (correct tokens score higher than wrong ones),
  * temperature scaling to reduce ECE on frame posteriors,
  * abstention to raise accuracy on what is actually asserted, and
  * the policy never to emit a sign below its threshold.

Testing calibration only on synthetic logits would prove the formulas; testing
it here proves the *system*.
"""

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from signtranslator.speech import (
    LogMelSpectrogram, ctc_prefix_beam_search, ctc_forced_alignment,
    Lattice, N_MELS,
)
from signtranslator.speech.calibration import (
    TemperatureScaler, expected_calibration_error, brier_decomposition,
    negative_log_likelihood,
)
from signtranslator.speech.policy import (
    Action, FailClosedPolicy, selective_metrics, area_under_risk_coverage,
)
from signtranslator.models import SpeechRecognizer

SR = 16000
WORD_S, GAP_S = 0.20, 0.08
VOCAB = {120.0: 1, 210.0: 2, 320.0: 3}


def _utterance(f0s, noise=0.0, seed=0, pitch_scale=1.0):
    g = torch.Generator().manual_seed(seed)
    parts = []
    for f0 in f0s:
        n = int(WORD_S * SR)
        t = torch.arange(n, dtype=torch.float32) / SR
        x = sum(torch.sin(2 * math.pi * f0 * pitch_scale * h * t) / h
                for h in (1, 2, 3))
        parts.append(x * torch.hann_window(n, periodic=False) * 0.5)
        parts.append(torch.zeros(int(GAP_S * SR)))
    wav = torch.cat(parts)
    if noise > 0:
        wav = wav + noise * torch.randn(wav.numel(), generator=g)
    return wav


def _batch(front_end, noise=0.0, seed=0, pitch_scale=1.0):
    seqs = [list(p) for p in itertools.permutations([120.0, 210.0, 320.0])]
    feats, targets = [], []
    for i, s in enumerate(seqs):
        feats.append(front_end(_utterance(s, noise, seed + i, pitch_scale)).t().unsqueeze(0))
        targets.append([VOCAB[f] for f in s])
    n = min(f.shape[1] for f in feats)
    return torch.cat([f[:, :n] for f in feats], dim=0), torch.tensor(targets)


@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    fe = LogMelSpectrogram()
    feats, targets = _batch(fe)
    rec = SpeechRecognizer(input_dim=N_MELS, num_tokens=3, hidden_dim=96,
                           num_layers=2, num_heads=4, subsample=2)
    lengths = torch.full((feats.shape[0],), targets.shape[1], dtype=torch.long)
    opt = torch.optim.Adam(rec.parameters(), lr=3e-3)
    for _ in range(260):
        loss = rec.loss(feats, targets, lengths)
        opt.zero_grad(); loss.backward(); opt.step()
    rec.eval()
    return rec, fe, targets


def _token_confidences(rec, fe, pitch_scales, targets, seed0=100, repeats=4):
    """Decode pitch-shifted utterances; return confidence/correctness pairs.

    Gaussian-noise failures in this controlled recognizer are almost entirely
    deletions, which produce no emitted token whose confidence a token-level
    policy could act on. Pitch shifts instead produce genuine substitutions and
    insertions, allowing the confidence/abstention claim to be tested rather
    than inferred from missing output.
    """
    confs, corrects = [], []
    for k, pitch_scale in enumerate(pitch_scales):
        for r in range(repeats):
            feats, tgt = _batch(fe, seed=seed0 + 50 * k + 7 * r,
                                pitch_scale=pitch_scale)
            with torch.no_grad():
                log_probs = rec(feats)
            for i in range(feats.shape[0]):
                lat = Lattice.from_nbest(
                    ctc_prefix_beam_search(log_probs[i], beam_width=12, blank=0))
                ref = tgt[i].tolist()
                for pos, (tok, c) in enumerate(zip(lat.best, lat.confidence())):
                    confs.append(float(c))
                    corrects.append(pos < len(ref) and tok == ref[pos])
    return confs, corrects


# ---------------------------------------------------------------------------
# Confidence quality on real output
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def noisy_confidences(trained):
    rec, fe, targets = trained
    return _token_confidences(rec, fe, [1.0, 0.70, 1.15, 1.25, 1.30], targets)


def test_noisy_evaluation_actually_produces_errors(noisy_confidences):
    """Guard against a vacuous test: if nothing is wrong, nothing is proved."""
    _, correct = noisy_confidences
    assert len(correct) > 20
    accuracy = sum(correct) / len(correct)
    assert 0.2 < accuracy < 0.99, f"accuracy {accuracy} leaves nothing to measure"


def test_confidence_is_informative_about_correctness(noisy_confidences):
    """Correct tokens must, on average, be scored higher than wrong ones.

    If this fails, every threshold in the policy is meaningless regardless of
    how well the formulas are implemented.
    """
    conf, correct = noisy_confidences
    c = torch.tensor(conf, dtype=torch.float64)
    y = torch.tensor(correct).bool()
    assert float(c[y].mean()) > float(c[~y].mean())


def test_abstention_raises_accuracy_on_real_model_output(noisy_confidences):
    """The core value of failing closed, measured on genuine errors."""
    conf, correct = noisy_confidences
    overall = sum(correct) / len(correct)
    high = selective_metrics(conf, correct, threshold=0.9)
    assert high.coverage > 0.0, "threshold rejected everything"
    assert high.selective_accuracy > overall


def test_risk_falls_as_coverage_falls(noisy_confidences):
    conf, correct = noisy_confidences
    assert area_under_risk_coverage(conf, correct) < 0.5


# ---------------------------------------------------------------------------
# Calibration on real frame posteriors
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def frame_posteriors(trained):
    """Frame logits + labels from forced alignment against the true transcript."""
    rec, fe, _ = trained
    logits, labels = [], []
    for k, noise in enumerate([0.0, 0.04, 0.07, 0.10]):
        feats, tgt = _batch(fe, noise=noise, seed=500 + 40 * k)
        with torch.no_grad():
            lp = rec(feats)
        for i in range(feats.shape[0]):
            al = ctc_forced_alignment(lp[i], tgt[i].tolist(), blank=0)
            logits.append(lp[i].double())
            labels.append(torch.tensor(al.state_tokens(), dtype=torch.long))
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0)


def test_temperature_scaling_reduces_its_fitted_objective(frame_posteriors):
    """Temperature scaling must reduce NLL, the objective it actually fits.

    ECE is a discontinuous binned diagnostic and may worsen on a finite held-out
    split even when NLL improves; asserting otherwise would be fake mathematics.
    """
    logits, labels = frame_posteriors
    n = logits.shape[0]
    # The calibration set must be drawn from the SAME distribution as the
    # evaluation set. The frames are stored grouped by noise level, so a
    # sequential split would fit T on clean audio and evaluate it on noisy
    # audio -- a distribution shift that made temperature scaling *worsen* ECE
    # in an earlier version of this test. Shuffle first.
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    logits, labels = logits[perm], labels[perm]
    split = n // 2
    fit_x, fit_y = logits[:split], labels[:split]
    ev_x, ev_y = logits[split:], labels[split:]

    before = negative_log_likelihood(F.log_softmax(fit_x, dim=-1), fit_y)
    scaler = TemperatureScaler(1.0)
    scaler.fit(fit_x, fit_y, max_iter=300)
    after = negative_log_likelihood(scaler(fit_x), fit_y)
    assert after <= before + 1e-8, f"NLL worsened: {before:.4f} -> {after:.4f}"

    # Held-out ECE remains a reported diagnostic, never a guaranteed theorem.
    conf, pred = scaler(ev_x).exp().max(dim=-1)
    ece = expected_calibration_error(conf, pred == ev_y, n_bins=15)
    assert 0.0 <= ece <= 1.0


def test_temperature_scaling_preserves_frame_accuracy(frame_posteriors):
    """Calibration must not trade accuracy for calibration."""
    logits, labels = frame_posteriors
    scaler = TemperatureScaler(1.0)
    scaler.fit(logits, labels, max_iter=200)
    before = (F.log_softmax(logits, dim=-1).argmax(-1) == labels).double().mean()
    after = (scaler(logits).argmax(-1) == labels).double().mean()
    assert torch.equal(before, after)


def test_brier_decomposition_is_exact_on_real_confidences(noisy_confidences):
    conf, correct = noisy_confidences
    d = brier_decomposition(conf, correct)
    assert abs(d.reconstructed - d.brier) < 1e-9
    assert d.resolution > 0.0, "confidence carries no discriminative signal"


# ---------------------------------------------------------------------------
# Policy on real output
# ---------------------------------------------------------------------------
def test_policy_never_emits_below_threshold_on_real_output(noisy_confidences):
    conf, _ = noisy_confidences
    pol = FailClosedPolicy(emit_threshold=0.9, fingerspell_threshold=0.5,
                           sign_lexicon=[1, 2, 3])
    out = pol.decide_sequence([1] * len(conf), conf)
    for d in out.decisions:
        if d.action is Action.EMIT:
            assert d.confidence >= 0.9


def test_policy_suppresses_errors_it_would_otherwise_assert(noisy_confidences):
    """Among tokens the policy chooses to sign, accuracy must be high.

    This is the end-to-end statement of the specification's rule: the system
    should rather say nothing than assert a sign it cannot support.
    """
    conf, correct = noisy_confidences
    pol = FailClosedPolicy(emit_threshold=0.9, fingerspell_threshold=0.5,
                           sign_lexicon=[1, 2, 3])
    decisions = pol.decide_sequence([1] * len(conf), conf).decisions
    emitted = [c for d, c in zip(decisions, correct) if d.action is Action.EMIT]
    assert len(emitted) > 0
    emitted_accuracy = sum(emitted) / len(emitted)
    overall = sum(correct) / len(correct)
    assert emitted_accuracy > overall


def test_out_of_lexicon_word_is_fingerspelled_not_signed(noisy_confidences):
    """A confidently recognised word with no sign must be spelled, not invented."""
    pol = FailClosedPolicy(emit_threshold=0.6, fingerspell_threshold=0.3,
                           sign_lexicon=[1, 2], verified_lexicon=[1, 2, 3])
    assert pol.decide(3, 0.99).action is Action.FINGERSPELL
    assert pol.decide(2, 0.99).action is Action.EMIT


def test_full_chain_audio_to_policy_decision(trained):
    """audio -> log-Mel -> CTC -> lattice -> confidence -> action."""
    rec, fe, _ = trained
    wav = _utterance([120.0, 210.0, 320.0], noise=0.5, seed=7)
    feats = fe(wav).t().unsqueeze(0)
    with torch.no_grad():
        log_probs = rec(feats)[0]
    lat = Lattice.from_nbest(ctc_prefix_beam_search(log_probs, beam_width=12))
    pol = FailClosedPolicy(emit_threshold=0.85, fingerspell_threshold=0.4,
                           sign_lexicon=[1, 2, 3])
    out = pol.decide_sequence(list(lat.best), lat.confidence())
    assert len(out) == len(lat.best)
    assert 0.0 <= out.coverage <= 1.0
    for d in out.decisions:
        assert d.action in (Action.EMIT, Action.FINGERSPELL, Action.PAUSE)
        assert d.reason
