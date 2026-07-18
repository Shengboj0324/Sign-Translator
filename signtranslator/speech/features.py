"""Waveform -> log-Mel front-end.

Implements the acoustic feature stage of ``01_speech_foundation_layer.md``
following the Whisper convention (16 kHz, 25 ms window, 10 ms hop, 80 mel
channels), with both HTK and Slaney mel scales.

The mathematics is specified in ``docs/SPEECH_FOUNDATION.md`` §3. Every claim
made there has a falsifying test in ``tests/test_speech_features.py``:

  * mel scale invertibility and strict monotonicity
  * triangular filterbank **partition of unity** (unnormalised) and unit-area
    (Slaney-normalised)
  * Parseval energy conservation of the STFT
  * a pure sinusoid landing in the analytically-predicted mel channel

Everything is ``torch`` and differentiable end-to-end, so the front-end can be
fine-tuned jointly with the encoder rather than frozen as a preprocessing step.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn

MelScale = Literal["htk", "slaney"]

# Whisper front-end constants (see docs/SPEECH_FOUNDATION.md §3.4).
SAMPLE_RATE = 16000
N_FFT = 400          # 25 ms
HOP_LENGTH = 160     # 10 ms  -> 100 Hz frame rate
N_MELS = 80

# Slaney scale break point.
_F_SP = 200.0 / 3.0                     # 66.667 Hz per mel below 1 kHz
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP      # = 15.0
_LOGSTEP = math.log(6.4) / 27.0


# ---------------------------------------------------------------------------
# Mel scales
# ---------------------------------------------------------------------------
def hz_to_mel(freq: torch.Tensor, scale: MelScale = "htk") -> torch.Tensor:
    """Frequency (Hz) -> mel. Strictly increasing and invertible."""
    if scale == "htk":
        return 2595.0 * torch.log10(1.0 + freq / 700.0)
    if scale == "slaney":
        mel = freq / _F_SP
        log_region = freq >= _MIN_LOG_HZ
        # torch.where evaluates both branches, so guard the log argument to keep
        # gradients and values finite on the unused branch.
        safe = torch.clamp(freq, min=_MIN_LOG_HZ)
        mel_log = _MIN_LOG_MEL + torch.log(safe / _MIN_LOG_HZ) / _LOGSTEP
        return torch.where(log_region, mel_log, mel)
    raise ValueError(f"unknown mel scale: {scale}")


def mel_to_hz(mel: torch.Tensor, scale: MelScale = "htk") -> torch.Tensor:
    """Mel -> frequency (Hz). Exact inverse of :func:`hz_to_mel`."""
    if scale == "htk":
        return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)
    if scale == "slaney":
        freq = _F_SP * mel
        log_region = mel >= _MIN_LOG_MEL
        safe = torch.clamp(mel, min=_MIN_LOG_MEL)
        freq_log = _MIN_LOG_HZ * torch.exp(_LOGSTEP * (safe - _MIN_LOG_MEL))
        return torch.where(log_region, freq_log, freq)
    raise ValueError(f"unknown mel scale: {scale}")


# ---------------------------------------------------------------------------
# Triangular filterbank
# ---------------------------------------------------------------------------
def triangular_response(freq: torch.Tensor, left: float, centre: float,
                        right: float) -> torch.Tensor:
    """Analytic response of one triangular filter (peak 1 at ``centre``).

    Exposed separately from the discretised bank so the partition-of-unity
    property can be tested on the *continuous* filter rather than on sampled
    FFT bins, where discretisation would only make it approximate.
    """
    rising = (freq - left) / max(centre - left, 1e-12)
    falling = (right - freq) / max(right - centre, 1e-12)
    return torch.clamp(torch.minimum(rising, falling), min=0.0)


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT,
                   sample_rate: int = SAMPLE_RATE, f_min: float = 0.0,
                   f_max: Optional[float] = None, scale: MelScale = "htk",
                   norm: Optional[Literal["slaney"]] = None,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build the ``(n_mels, n_fft//2 + 1)`` triangular mel filterbank.

    Args:
        norm: ``None`` gives unit-**peak** triangles, which satisfy partition of
            unity. ``"slaney"`` gives unit-**area** triangles (each scaled by
            ``2/(f_{j+1} - f_{j-1})``), which do not. The two are not
            interchangeable and the tests assert the correct property for each.
    """
    if n_mels < 1:
        raise ValueError("n_mels must be >= 1")
    if f_max is None:
        f_max = sample_rate / 2.0
    if not 0.0 <= f_min < f_max <= sample_rate / 2.0:
        raise ValueError("require 0 <= f_min < f_max <= Nyquist")

    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=torch.float64)

    # n_mels + 2 points equally spaced in the mel domain.
    m_min = hz_to_mel(torch.tensor(f_min, dtype=torch.float64), scale)
    m_max = hz_to_mel(torch.tensor(f_max, dtype=torch.float64), scale)
    m_pts = torch.linspace(float(m_min), float(m_max), n_mels + 2, dtype=torch.float64)
    f_pts = mel_to_hz(m_pts, scale)

    # Slope matrix: ramps[j, k] = f_pts[j] - fft_freqs[k]
    ramps = f_pts.unsqueeze(1) - fft_freqs.unsqueeze(0)
    fb = torch.zeros(n_mels, n_freqs, dtype=torch.float64)
    for j in range(n_mels):
        lower = -ramps[j] / max(float(f_pts[j + 1] - f_pts[j]), 1e-12)
        upper = ramps[j + 2] / max(float(f_pts[j + 2] - f_pts[j + 1]), 1e-12)
        fb[j] = torch.clamp(torch.minimum(lower, upper), min=0.0)

    if norm == "slaney":
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(1)

    return fb.to(dtype)


# ---------------------------------------------------------------------------
# STFT / spectrogram
# ---------------------------------------------------------------------------
def stft_power(waveform: torch.Tensor, n_fft: int = N_FFT,
               hop_length: int = HOP_LENGTH, center: bool = False,
               window: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Power spectrogram ``|STFT|^2`` of shape ``(..., n_freqs, n_frames)``.

    ``center=False`` (the default here) means frame ``m`` covers samples
    ``[m*hop, m*hop + n_fft)`` with no padding, so frame indices map to audio
    time by the simple algebra used in the streaming latency model. Whisper
    itself uses ``center=True``; that is available but shifts the time origin
    by ``n_fft/2``.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
        squeeze = True
    elif waveform.dim() == 2:
        squeeze = False
    else:
        raise ValueError("waveform must be (T,) or (N, T)")

    if window is None:
        window = torch.hann_window(n_fft, periodic=True, dtype=waveform.dtype,
                                   device=waveform.device)
    spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                      win_length=n_fft, window=window, center=center,
                      pad_mode="reflect", normalized=False, onesided=True,
                      return_complex=True)
    power = spec.real.pow(2) + spec.imag.pow(2)
    return power.squeeze(0) if squeeze else power


def num_frames(num_samples: int, n_fft: int = N_FFT,
               hop_length: int = HOP_LENGTH) -> int:
    """Frame count for non-centred STFT: ``floor((T - n_fft)/hop) + 1``."""
    if num_samples < n_fft:
        return 0
    return (num_samples - n_fft) // hop_length + 1


# ---------------------------------------------------------------------------
# Log-Mel module
# ---------------------------------------------------------------------------
class LogMelSpectrogram(nn.Module):
    """Differentiable waveform -> log-Mel features (Whisper convention).

    Output shape ``(..., n_mels, n_frames)``. The filterbank and window are
    registered buffers, so ``.to(device)`` moves them and they are excluded from
    the optimiser.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH, n_mels: int = N_MELS,
                 f_min: float = 0.0, f_max: Optional[float] = None,
                 scale: MelScale = "htk",
                 norm: Optional[Literal["slaney"]] = None,
                 center: bool = False, dynamic_range_db: float = 8.0,
                 normalize: bool = True) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.dynamic_range_db = dynamic_range_db
        self.normalize = normalize

        fb = mel_filterbank(n_mels, n_fft, sample_rate, f_min, f_max, scale, norm)
        self.register_buffer("filterbank", fb)
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))

    @property
    def frame_rate(self) -> float:
        """Feature frames per second."""
        return self.sample_rate / self.hop_length

    def num_frames(self, num_samples: int) -> int:
        if self.center:
            return num_samples // self.hop_length + 1
        return num_frames(num_samples, self.n_fft, self.hop_length)

    def mel_energies(self, waveform: torch.Tensor) -> torch.Tensor:
        """Linear mel energies (pre-log), shape ``(..., n_mels, n_frames)``."""
        power = stft_power(waveform, self.n_fft, self.hop_length,
                           center=self.center, window=self.window)
        # (n_mels, n_freqs) @ (..., n_freqs, n_frames)
        return torch.matmul(self.filterbank.to(power.dtype), power)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_energies(waveform)
        log_spec = torch.log10(torch.clamp(mel, min=1e-10))
        if self.dynamic_range_db is not None:
            # Floor at (peak - 80 dB); amax over the last two dims, per item.
            peak = torch.amax(log_spec, dim=(-2, -1), keepdim=True)
            log_spec = torch.maximum(log_spec, peak - self.dynamic_range_db)
        if self.normalize:
            log_spec = (log_spec + 4.0) / 4.0
        return log_spec
