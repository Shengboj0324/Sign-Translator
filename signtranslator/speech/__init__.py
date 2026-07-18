"""Speech foundation layer (see docs/SPEECH_FOUNDATION.md).

Stage 1: waveform -> log-Mel features, prosody, streaming, and the
resampler + gated projection into the planner width.
"""

from .features import (
    LogMelSpectrogram, mel_filterbank, hz_to_mel, mel_to_hz, stft_power,
    triangular_response, num_frames,
    SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS,
)

__all__ = [
    "LogMelSpectrogram", "mel_filterbank", "hz_to_mel", "mel_to_hz",
    "stft_power", "triangular_response", "num_frames",
    "SAMPLE_RATE", "N_FFT", "HOP_LENGTH", "N_MELS",
]
