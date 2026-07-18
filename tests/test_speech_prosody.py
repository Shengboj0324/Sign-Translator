"""Verification of the prosody pathway against signals of known ground truth.

F0 estimation is checked on synthesised signals whose fundamental is known
exactly, including harmonically rich waveforms that induce octave errors in
plain autocorrelation -- the specific failure YIN's cumulative mean
normalisation exists to prevent.
"""

import math

import pytest
import torch

from signtranslator.speech.prosody import (
    frame_signal, rms_energy, difference_function, cumulative_mean_normalized,
    estimate_f0_yin, detect_pauses, ProsodyExtractor,
)
from signtranslator.speech.features import SAMPLE_RATE

SR = SAMPLE_RATE


def _sine(f0, seconds=0.5, amp=1.0, sr=SR):
    t = torch.arange(int(seconds * sr), dtype=torch.float32) / sr
    return amp * torch.sin(2 * math.pi * f0 * t)


def _harmonic(f0, n_harmonics=6, seconds=0.5, sr=SR):
    """Harmonic stack with a *weak* fundamental -- the classic octave trap."""
    t = torch.arange(int(seconds * sr), dtype=torch.float32) / sr
    x = torch.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        amp = 0.3 if h == 1 else 1.0 / h        # deliberately weak fundamental
        x = x + amp * torch.sin(2 * math.pi * f0 * h * t)
    return x / x.abs().max()


# ---------------------------------------------------------------------------
# Framing / energy
# ---------------------------------------------------------------------------
def test_frame_signal_indices_match_convention():
    x = torch.arange(1000, dtype=torch.float32)
    frames = frame_signal(x, frame_length=400, hop_length=160)
    assert frames.shape == (4, 400)              # (1000-400)//160 + 1 = 4
    for m in range(frames.shape[0]):
        assert torch.equal(frames[m], x[m * 160: m * 160 + 400])


def test_frame_signal_too_short_returns_empty():
    assert frame_signal(torch.zeros(10), 400, 160).shape == (0, 400)


def test_rms_of_sine_is_amplitude_over_sqrt2():
    """Closed form: RMS of A*sin = A/sqrt(2)."""
    for amp in (0.25, 1.0, 3.0):
        e = rms_energy(_sine(200.0, amp=amp), frame_length=1600, hop_length=1600)
        expected = amp / math.sqrt(2.0)
        assert torch.allclose(e, torch.full_like(e, expected), rtol=1e-3)


def test_rms_of_silence_is_zero():
    assert float(rms_energy(torch.zeros(4000)).max()) == 0.0


# ---------------------------------------------------------------------------
# YIN internals
# ---------------------------------------------------------------------------
def test_difference_function_is_zero_at_lag_zero():
    """d(0) = sum (x[n]-x[n])^2 = 0 identically."""
    x = _sine(150.0, seconds=0.1)
    frames = frame_signal(x, 512 + 256, 256)
    d = difference_function(frames, window=512, tau_max=256)
    assert torch.allclose(d[:, 0], torch.zeros_like(d[:, 0]), atol=1e-6)


def test_difference_function_dips_at_true_period():
    """For a periodic signal, d(tau) must be near zero at tau = period."""
    f0 = 200.0
    period = int(round(SR / f0))                  # 80 samples
    x = _sine(f0, seconds=0.2)
    frames = frame_signal(x, 512 + 256, 256)
    d = difference_function(frames, window=512, tau_max=256)
    row = d[0]
    # The dip at the true period should be far below the mean level.
    assert row[period] < 0.05 * row.mean()


def test_cmnd_first_value_is_one_and_shape_preserved():
    d = torch.rand(3, 50) + 0.1
    dp = cumulative_mean_normalized(d)
    assert dp.shape == d.shape
    assert torch.allclose(dp[:, 0], torch.ones(3), atol=1e-9)


def test_cmnd_matches_manual_definition():
    """d'(tau) = d(tau) / [(1/tau) * sum_{j=1..tau} d(j)]."""
    d = torch.tensor([[0.0, 2.0, 1.0, 4.0, 3.0]], dtype=torch.float64)
    dp = cumulative_mean_normalized(d)
    for tau in range(1, 5):
        running_mean = sum(float(d[0, j]) for j in range(1, tau + 1)) / tau
        assert abs(float(dp[0, tau]) - float(d[0, tau]) / running_mean) < 1e-12


def test_cmnd_of_degenerate_frame_is_maximally_aperiodic():
    """REGRESSION: a constant/silent frame gives d(tau)=0 for all tau.

    The 0/0 ratio must resolve to 1 (no periodicity evidence), never to 0 --
    which would sit below every voicing threshold and report silence as a
    confidently pitched frame.
    """
    d = torch.zeros(1, 40)
    dp = cumulative_mean_normalized(d)
    assert torch.allclose(dp, torch.ones_like(dp))


# ---------------------------------------------------------------------------
# F0 estimation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("f0", [100.0, 150.0, 220.0, 330.0])
def test_f0_of_pure_sine_is_accurate(f0):
    res = estimate_f0_yin(_sine(f0), hop_length=320)
    voiced = res.f0[res.voiced]
    assert voiced.numel() > 0, f"no voiced frames at {f0} Hz"
    median = float(voiced.median())
    assert abs(median - f0) / f0 < 0.02, f"estimated {median} for {f0}"


def test_f0_avoids_octave_error_on_harmonic_stack():
    """The decisive YIN property: a weak fundamental must not be halved.

    Plain autocorrelation tends to lock onto tau = 2T here (reporting f0/2);
    the cumulative mean normalisation is what prevents it.
    """
    f0 = 200.0
    res = estimate_f0_yin(_harmonic(f0), hop_length=320)
    voiced = res.f0[res.voiced]
    assert voiced.numel() > 0
    median = float(voiced.median())
    assert abs(median - f0) / f0 < 0.05, f"got {median}, octave error?"
    assert abs(median - f0 / 2) / (f0 / 2) > 0.1     # explicitly not the octave


def test_silence_is_unvoiced():
    """REGRESSION: digital silence must never be reported as pitched.

    Before the 0/0 fix this returned voiced=True at f_max for every frame.
    """
    res = estimate_f0_yin(torch.zeros(SR // 2), hop_length=320)
    assert not bool(res.voiced.any())
    assert float(res.f0.abs().max()) == 0.0
    assert float(res.aperiodicity.min()) >= 1.0 - 1e-6   # maximally aperiodic


def test_near_silence_is_unvoiced_via_energy_gate():
    """A signal far below the noise floor carries no usable pitch evidence."""
    res = estimate_f0_yin(_sine(200.0, seconds=0.3) * 1e-9, hop_length=320)
    assert not bool(res.voiced.any())


def test_dc_offset_signal_is_unvoiced():
    """A constant non-zero signal is degenerate too (d(tau)=0 for all tau)."""
    res = estimate_f0_yin(torch.full((SR // 2,), 0.5), hop_length=320)
    assert not bool(res.voiced.any())


def test_white_noise_is_mostly_unvoiced():
    """Aperiodic input must not be reported as pitched."""
    torch.manual_seed(0)
    res = estimate_f0_yin(torch.randn(SR // 2) * 0.1, hop_length=320)
    assert float(res.voiced.float().mean()) < 0.25


def test_voiced_frames_have_low_aperiodicity():
    res = estimate_f0_yin(_sine(180.0), hop_length=320)
    assert bool(res.voiced.any())
    assert float(res.aperiodicity[res.voiced].max()) < 0.1   # below threshold


def test_f0_search_range_is_respected():
    res = estimate_f0_yin(_sine(200.0), hop_length=320, f_min=150.0, f_max=300.0)
    voiced = res.f0[res.voiced]
    assert voiced.numel() > 0
    assert float(voiced.min()) >= 150.0 * 0.95
    assert float(voiced.max()) <= 300.0 * 1.05


def test_invalid_f0_range_rejected():
    with pytest.raises(ValueError):
        estimate_f0_yin(_sine(200.0), f_min=400.0, f_max=100.0)
    with pytest.raises(ValueError):
        estimate_f0_yin(_sine(200.0), f_min=0.0, f_max=100.0)


def test_difference_function_rejects_short_frames():
    with pytest.raises(ValueError):
        difference_function(torch.randn(2, 100), window=80, tau_max=50)


# ---------------------------------------------------------------------------
# Pauses
# ---------------------------------------------------------------------------
def test_detect_pauses_finds_injected_silence():
    """Speech, 200 ms of silence, speech -> exactly one pause in the middle."""
    sr = SR
    speech = _sine(200.0, seconds=0.4)
    silence = torch.zeros(int(0.2 * sr))
    x = torch.cat([speech, silence, speech])
    energy = rms_energy(x, frame_length=400, hop_length=160)
    pauses = detect_pauses(energy, threshold_db=-35.0, min_frames=5)
    assert len(pauses) == 1
    start, end = pauses[0]
    # 0.4 s at 100 frames/s -> pause begins near frame 40 and lasts ~20 frames.
    assert 35 <= start <= 45
    assert (end - start) >= 15


def test_no_pause_in_continuous_speech():
    energy = rms_energy(_sine(200.0, seconds=1.0), 400, 160)
    assert detect_pauses(energy) == []


def test_short_gaps_are_not_pauses():
    """A 20 ms stop gap is not a discourse pause."""
    x = torch.cat([_sine(200.0, 0.3), torch.zeros(int(0.02 * SR)), _sine(200.0, 0.3)])
    energy = rms_energy(x, 400, 160)
    assert detect_pauses(energy, min_frames=10) == []


def test_pause_detection_is_gain_invariant():
    x = torch.cat([_sine(200.0, 0.3), torch.zeros(int(0.2 * SR)), _sine(200.0, 0.3)])
    loud = detect_pauses(rms_energy(x * 10.0, 400, 160))
    quiet = detect_pauses(rms_energy(x * 0.01, 400, 160))
    assert loud == quiet


def test_all_silence_is_one_pause():
    assert detect_pauses(torch.zeros(50)) == [(0, 50)]


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
def test_extractor_shape_and_finiteness():
    ex = ProsodyExtractor(hop_length=320)
    feats = ex(_sine(200.0, seconds=0.5))
    assert feats.dim() == 2 and feats.shape[1] == ProsodyExtractor.N_FEATURES
    assert torch.isfinite(feats).all()


def test_extractor_logf0_matches_voiced_frames():
    ex = ProsodyExtractor(hop_length=320)
    feats = ex(_sine(220.0, seconds=0.5))
    voiced = feats[:, 1] > 0.5
    assert bool(voiced.any())
    recovered = torch.exp(feats[voiced, 0])
    assert abs(float(recovered.median()) - 220.0) / 220.0 < 0.02
    # Unvoiced frames carry log-F0 = 0 as the explicit "no pitch" sentinel.
    if bool((~voiced).any()):
        assert torch.allclose(feats[~voiced, 0], torch.zeros(int((~voiced).sum())))


def test_extractor_resample_to_target_grid():
    ex = ProsodyExtractor(hop_length=320)
    feats = ex(_sine(200.0, seconds=0.5))
    out = ex.resample_to(feats, 77)
    assert out.shape == (77, ProsodyExtractor.N_FEATURES)
    assert torch.allclose(ex.resample_to(feats, feats.shape[0]), feats)


def test_extractor_handles_too_short_audio():
    ex = ProsodyExtractor()
    assert ex(torch.zeros(50)).shape == (0, ProsodyExtractor.N_FEATURES)
