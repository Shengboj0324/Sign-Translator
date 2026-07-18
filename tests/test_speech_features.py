"""Mathematical verification of the log-Mel front-end.

These are falsification tests for the claims in docs/SPEECH_FOUNDATION.md §3.
Where a closed-form property exists (invertibility, partition of unity,
Parseval) it is asserted exactly rather than approximated.
"""

import math

import pytest
import torch

from signtranslator.speech.features import (
    hz_to_mel, mel_to_hz, mel_filterbank, triangular_response, stft_power,
    num_frames, LogMelSpectrogram, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS,
)


# ---------------------------------------------------------------------------
# Mel scales
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale", ["htk", "slaney"])
def test_mel_scale_is_invertible(scale):
    f = torch.linspace(0.0, 8000.0, 1000, dtype=torch.float64)
    back = mel_to_hz(hz_to_mel(f, scale), scale)
    assert torch.allclose(back, f, atol=1e-6), f"{scale} not invertible"


@pytest.mark.parametrize("scale", ["htk", "slaney"])
def test_mel_scale_strictly_increasing(scale):
    f = torch.linspace(0.0, 8000.0, 2000, dtype=torch.float64)
    m = hz_to_mel(f, scale)
    assert torch.all(m[1:] > m[:-1]), f"{scale} not strictly increasing"


def test_htk_matches_closed_form():
    """Spot-check against the published formula 2595*log10(1+f/700)."""
    for f in (0.0, 100.0, 1000.0, 4000.0):
        expected = 2595.0 * math.log10(1.0 + f / 700.0)
        got = float(hz_to_mel(torch.tensor(f, dtype=torch.float64), "htk"))
        assert abs(got - expected) < 1e-9


def test_slaney_is_linear_below_1khz_and_continuous_at_break():
    """Slaney is linear (slope 3/200 mel/Hz) below 1 kHz, and joins smoothly."""
    f = torch.tensor([100.0, 200.0, 400.0], dtype=torch.float64)
    m = hz_to_mel(f, "slaney")
    assert torch.allclose(m, f * 3.0 / 200.0, atol=1e-9)
    eps = 1e-6
    left = hz_to_mel(torch.tensor(1000.0 - eps, dtype=torch.float64), "slaney")
    right = hz_to_mel(torch.tensor(1000.0 + eps, dtype=torch.float64), "slaney")
    assert abs(float(left) - float(right)) < 1e-4        # continuous at break


def test_unknown_scale_rejected():
    with pytest.raises(ValueError):
        hz_to_mel(torch.tensor(1.0), "bark")
    with pytest.raises(ValueError):
        mel_to_hz(torch.tensor(1.0), "bark")


# ---------------------------------------------------------------------------
# Filterbank
# ---------------------------------------------------------------------------
def test_filterbank_shape_and_nonnegativity():
    fb = mel_filterbank(n_mels=80, n_fft=400, sample_rate=16000)
    assert fb.shape == (80, 201)
    assert torch.all(fb >= 0)
    assert torch.isfinite(fb).all()


def test_each_filter_is_unimodal_with_interior_peak():
    """A triangular filter must rise then fall exactly once."""
    fb = mel_filterbank(n_mels=40, n_fft=400, sample_rate=16000).double()
    for j in range(fb.shape[0]):
        row = fb[j]
        support = (row > 0).nonzero().flatten()
        assert support.numel() >= 2, f"filter {j} degenerate"
        seg = row[support[0]:support[-1] + 1]
        peak = int(torch.argmax(seg))
        # non-decreasing up to the peak, non-increasing after
        assert torch.all(seg[1:peak + 1] - seg[:peak] >= -1e-12)
        assert torch.all(seg[peak + 1:] - seg[peak:-1] <= 1e-12)


def test_partition_of_unity_on_analytic_triangles():
    """Adjacent unit-peak triangles sum to exactly 1 between their centres.

    Asserted on the *continuous* response so the property is exact; the
    discretised bank only samples this.
    """
    left, centre, right = 300.0, 400.0, 520.0
    f = torch.linspace(centre, right, 257, dtype=torch.float64)[1:-1]
    h_j = triangular_response(f, left, centre, right)          # falling side
    h_next = triangular_response(f, centre, right, 700.0)      # rising side
    total = h_j + h_next
    assert torch.allclose(total, torch.ones_like(total), atol=1e-12)


def test_slaney_norm_gives_unit_area_not_unit_peak():
    """Slaney normalisation trades the peak-1 property for area-1."""
    kw = dict(n_mels=40, n_fft=512, sample_rate=16000, scale="slaney")
    plain = mel_filterbank(norm=None, **kw).double()
    slaney = mel_filterbank(norm="slaney", **kw).double()
    # Unnormalised triangles have unit PEAK, but the bank samples them at FFT
    # bin frequencies and a filter centre generally falls between bins -- so the
    # sampled maximum approaches 1 from below and never exceeds it. (The
    # continuous response attains exactly 1; see the analytic check below.)
    assert float(plain.max()) <= 1.0 + 1e-12
    assert float(plain.max()) > 0.99
    assert abs(float(triangular_response(torch.tensor(400.0, dtype=torch.float64),
                                         300.0, 400.0, 520.0)) - 1.0) < 1e-12
    # Slaney-normalised ones do not, but integrate to ~1 in Hz.
    df = (16000 / 2) / (512 // 2)         # Hz per FFT bin
    areas = slaney.sum(dim=1) * df
    assert torch.allclose(areas, torch.ones_like(areas), atol=0.05)
    assert float(slaney.max()) < 1.0


def test_filterbank_centres_are_monotone_in_frequency():
    """Filter j's peak bin must increase with j (mel spacing is monotone)."""
    fb = mel_filterbank(n_mels=40, n_fft=512, sample_rate=16000)
    peaks = fb.argmax(dim=1)
    assert torch.all(peaks[1:] >= peaks[:-1])


def test_filterbank_rejects_invalid_ranges():
    with pytest.raises(ValueError):
        mel_filterbank(f_min=1000.0, f_max=500.0)
    with pytest.raises(ValueError):
        mel_filterbank(f_max=99999.0)          # beyond Nyquist
    with pytest.raises(ValueError):
        mel_filterbank(n_mels=0)


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------
def test_parseval_energy_conservation():
    """Per frame: sum |w*x|^2 == (1/N) sum over ALL N bins of |X|^2.

    The one-sided spectrum must be folded (DC and Nyquist once, others twice)
    before comparison -- the classic energy-accounting bug.
    """
    torch.manual_seed(0)
    n_fft, hop = 256, 256                     # no overlap: frames are disjoint
    x = torch.randn(n_fft * 4, dtype=torch.float64)
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
    power = stft_power(x, n_fft=n_fft, hop_length=hop, center=False, window=window)

    weights = torch.full((n_fft // 2 + 1,), 2.0, dtype=torch.float64)
    weights[0] = 1.0            # DC counted once
    weights[-1] = 1.0           # Nyquist counted once
    spectral_energy = (weights.unsqueeze(1) * power).sum(dim=0) / n_fft

    for m in range(power.shape[1]):
        frame = x[m * hop: m * hop + n_fft] * window
        assert abs(float(frame.pow(2).sum()) - float(spectral_energy[m])) < 1e-8


def test_sinusoid_peaks_at_predicted_fft_bin():
    """A sinusoid at exactly bin-centre f_k = k*sr/N must peak at bin k."""
    sr, n_fft = 16000, 512
    k = 40
    f = k * sr / n_fft                       # 1250 Hz -> exact bin centre
    t = torch.arange(n_fft * 3, dtype=torch.float64) / sr
    x = torch.sin(2 * math.pi * f * t)
    power = stft_power(x, n_fft=n_fft, hop_length=n_fft, center=False,
                       window=torch.hann_window(n_fft, periodic=True,
                                                dtype=torch.float64))
    assert int(torch.argmax(power[:, 0])) == k


def test_num_frames_matches_actual_stft_output():
    for n_samples in (400, 401, 1000, 1600, 16000):
        expected = num_frames(n_samples, N_FFT, HOP_LENGTH)
        if expected == 0:
            continue
        x = torch.randn(n_samples)
        got = stft_power(x, N_FFT, HOP_LENGTH, center=False).shape[-1]
        assert got == expected, f"{n_samples}: {got} != {expected}"


def test_stft_rejects_bad_rank():
    with pytest.raises(ValueError):
        stft_power(torch.randn(2, 3, 4))


# ---------------------------------------------------------------------------
# Log-Mel module
# ---------------------------------------------------------------------------
def test_logmel_shape_and_frame_rate():
    fe = LogMelSpectrogram()
    assert abs(fe.frame_rate - 100.0) < 1e-9        # 16 kHz / 160 = 100 Hz
    x = torch.randn(SAMPLE_RATE)                    # 1 second
    out = fe(x)
    assert out.shape[0] == N_MELS
    assert out.shape[1] == fe.num_frames(SAMPLE_RATE)
    assert torch.isfinite(out).all()


def test_logmel_batched_matches_single():
    fe = LogMelSpectrogram()
    torch.manual_seed(0)
    xs = torch.randn(3, 4000)
    batched = fe(xs)
    for i in range(3):
        assert torch.allclose(batched[i], fe(xs[i]), atol=1e-5)


def test_dynamic_range_floor_is_enforced():
    """No value may sit more than `dynamic_range_db` below the per-item peak."""
    fe = LogMelSpectrogram(normalize=False, dynamic_range_db=8.0)
    x = torch.randn(4000)
    out = fe(x)
    assert float(out.max() - out.min()) <= 8.0 + 1e-6


def test_silence_is_finite_not_nan():
    """Digital silence must not produce log(0) = -inf."""
    fe = LogMelSpectrogram()
    out = fe(torch.zeros(4000))
    assert torch.isfinite(out).all()


def test_tone_energy_lands_in_correct_mel_channel():
    """A 1 kHz tone must peak in the mel channel whose triangle contains 1 kHz.

    The expected channel is derived analytically from the filterbank, so this
    cross-checks the STFT and the filterbank against each other.
    """
    sr = SAMPLE_RATE
    fe = LogMelSpectrogram(normalize=False, dynamic_range_db=None)
    f0 = 1000.0
    t = torch.arange(sr, dtype=torch.float32) / sr
    x = torch.sin(2 * math.pi * f0 * t)
    mel = fe.mel_energies(x).mean(dim=1)             # average over frames

    fb = fe.filterbank
    bin_idx = int(round(f0 * N_FFT / sr))            # FFT bin nearest 1 kHz
    expected_channel = int(torch.argmax(fb[:, bin_idx]))
    assert int(torch.argmax(mel)) == expected_channel


def test_louder_input_gives_greater_mel_energy():
    """Monotonicity in amplitude: mel energy is quadratic in signal gain."""
    fe = LogMelSpectrogram()
    torch.manual_seed(0)
    x = torch.randn(4000)
    quiet = fe.mel_energies(0.1 * x).sum()
    loud = fe.mel_energies(1.0 * x).sum()
    # Power scales with amplitude^2 => factor ~100 for a 10x gain.
    assert torch.allclose(loud / quiet, torch.tensor(100.0), rtol=1e-3)


def test_front_end_is_differentiable():
    """Gradients must flow to the waveform (joint fine-tuning must be possible)."""
    fe = LogMelSpectrogram()
    x = torch.randn(4000, requires_grad=True)
    fe(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_filterbank_and_window_are_buffers_not_parameters():
    fe = LogMelSpectrogram()
    params = {n for n, _ in fe.named_parameters()}
    buffers = {n for n, _ in fe.named_buffers()}
    assert params == set()
    assert {"filterbank", "window"} <= buffers
