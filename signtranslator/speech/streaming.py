"""Streaming feature extraction and the emission-latency model.

The source specification requires streaming with *explicitly reported* chunk
size and right context, and median plus p95 emission latency. Two things must
hold, and both are proved rather than assumed (docs/SPEECH_FOUNDATION.md §4):

1. **Equivalence.** Feeding audio incrementally must reproduce offline
   extraction exactly. This requires carrying ``n_fft - hop`` samples of overlap
   across chunk boundaries; dropping it silently corrupts every frame that
   straddles a boundary, which is invisible in a shape test.

2. **Latency.** The algorithmic latency is derived in closed form and then
   measured by simulation; the test asserts the two agree. Measuring alone would
   be a tautology (it would only confirm the simulator matches itself).

Latency is expressed in *audio time*, not wall-clock, so it is deterministic and
reproducible: it measures the algorithm's inherent delay, independent of how
fast the host happens to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .features import LogMelSpectrogram, N_FFT, HOP_LENGTH, SAMPLE_RATE


class StreamingFeatureExtractor:
    """Incremental log-Mel extraction with correct cross-chunk overlap.

    Call :meth:`push` with successive audio chunks; it returns only the frames
    that became complete, and retains the ``n_fft - hop`` sample tail needed by
    the next frame that straddles the boundary.
    """

    def __init__(self, front_end: Optional[LogMelSpectrogram] = None) -> None:
        # Default to a causal front-end: a streaming component should be
        # streamable out of the box rather than raising on its own default.
        self.front_end = front_end or LogMelSpectrogram(floor_mode="none")
        if self.front_end.center:
            raise ValueError(
                "streaming requires center=False; centred STFT pads the signal "
                "and cannot be produced causally")
        if not self.front_end.is_causal:
            raise ValueError(
                "streaming requires a causal front-end, but floor_mode='global' "
                "floors against the maximum over the WHOLE utterance, which a "
                "prefix cannot know. Chunked features would silently disagree "
                "with offline ones. Use floor_mode='fixed' or 'none'.")
        self.n_fft = self.front_end.n_fft
        self.hop = self.front_end.hop_length
        self.reset()

    def reset(self) -> None:
        self._buffer = torch.zeros(0)
        self.frames_emitted = 0
        self.samples_consumed = 0     # global index of buffer[0]

    @property
    def buffered_samples(self) -> int:
        return int(self._buffer.numel())

    def push(self, chunk: torch.Tensor) -> torch.Tensor:
        """Feed audio; return newly completed frames ``(n_mels, k)`` (k may be 0)."""
        if chunk.dim() != 1:
            raise ValueError("streaming push expects a 1-D chunk")
        if self._buffer.numel() == 0:
            self._buffer = chunk.clone()
        else:
            self._buffer = torch.cat([self._buffer, chunk])

        available = self._buffer.numel()
        if available < self.n_fft:
            return self._empty()

        k = (available - self.n_fft) // self.hop + 1
        needed = self.n_fft + (k - 1) * self.hop
        feats = self.front_end(self._buffer[:needed])

        # Advance by exactly k hops, keeping the overlap tail for the next frame.
        self._buffer = self._buffer[k * self.hop:]
        self.samples_consumed += k * self.hop
        self.frames_emitted += k
        return feats

    def _empty(self) -> torch.Tensor:
        return torch.zeros(self.front_end.n_mels, 0)

    def flush(self) -> torch.Tensor:
        """Emit nothing further; retained samples cannot form a complete frame."""
        return self._empty()


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LatencyModel:
    """Closed-form emission latency for a chunked, right-context pipeline.

    Args:
        chunk_frames: number of feature frames emitted per processing step.
        right_context: frames of lookahead the *encoder* needs before it can
            emit outputs for a frame. The feature extractor itself needs none;
            this models the downstream acoustic model.
    """

    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    sample_rate: int = SAMPLE_RATE
    chunk_frames: int = 1
    right_context: int = 0

    def __post_init__(self) -> None:
        if self.chunk_frames < 1:
            raise ValueError("chunk_frames must be >= 1")
        if self.right_context < 0:
            raise ValueError("right_context must be >= 0")

    @property
    def frame_period_s(self) -> float:
        return self.hop_length / self.sample_rate

    @property
    def algorithmic_latency_s(self) -> float:
        """Latency with no chunk buffering: ``(R*hop + n_fft/2)/sr``.

        A frame is centred at ``(t*hop + n_fft/2)/sr`` but needs audio through
        ``(t+R)*hop + n_fft`` before it can be emitted.
        """
        return (self.right_context * self.hop_length
                + self.n_fft / 2.0) / self.sample_rate

    def excess_samples(self) -> List[int]:
        """The exact set of buffering waits, in samples, over frame positions.

        Frame ``t`` needs audio through ``required(t) = (t+R)*hop + n_fft`` and
        is released at the next chunk boundary, so the wait is
        ``excess(t) = (-required(t)) mod chunk_samples``.

        **This does not simplify to multiples of the hop.** Writing
        ``rho = n_fft mod hop``, the waits are ``{C*hop - rho, ..., hop - rho}``
        when ``rho != 0``, and ``{0, hop, ..., (C-1)*hop}`` only when the window
        is an exact multiple of the hop. Whisper's 400/160 gives ``rho = 80``,
        so the naive "at most (C-1) frame periods" bound is wrong by up to a
        full frame -- which the measured-vs-analytic test detects.
        """
        chunk_samples = self.chunk_frames * self.hop_length
        return sorted(
            (-((t + self.right_context) * self.hop_length + self.n_fft)) % chunk_samples
            for t in range(self.chunk_frames)
        )

    @property
    def min_latency_s(self) -> float:
        return self.algorithmic_latency_s + min(self.excess_samples()) / self.sample_rate

    @property
    def max_latency_s(self) -> float:
        return self.algorithmic_latency_s + max(self.excess_samples()) / self.sample_rate

    @property
    def mean_latency_s(self) -> float:
        ex = self.excess_samples()
        return self.algorithmic_latency_s + (sum(ex) / len(ex)) / self.sample_rate

    def describe(self) -> str:
        return (f"chunk={self.chunk_frames} frames "
                f"({self.chunk_frames * self.frame_period_s * 1000:.0f} ms), "
                f"right_context={self.right_context} frames, "
                f"latency min/mean/max = "
                f"{self.min_latency_s * 1000:.1f}/"
                f"{self.mean_latency_s * 1000:.1f}/"
                f"{self.max_latency_s * 1000:.1f} ms")


def percentile(values: List[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0,100]); no numpy dependency."""
    if not values:
        raise ValueError("percentile of empty sequence")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


@dataclass
class LatencyMeasurement:
    latencies_s: List[float]

    @property
    def median_s(self) -> float:
        return percentile(self.latencies_s, 50.0)

    @property
    def p95_s(self) -> float:
        return percentile(self.latencies_s, 95.0)

    @property
    def max_s(self) -> float:
        return max(self.latencies_s)

    def report(self) -> str:
        return (f"median {self.median_s * 1000:.1f} ms | "
                f"p95 {self.p95_s * 1000:.1f} ms | "
                f"max {self.max_s * 1000:.1f} ms "
                f"(n={len(self.latencies_s)})")


def measure_emission_latency(num_samples: int, model: LatencyModel
                             ) -> LatencyMeasurement:
    """Simulate chunked streaming and record per-frame emission latency.

    Audio arrives in chunks of ``chunk_frames * hop`` samples. After the chunk
    ending at global sample ``S``, every frame ``t`` whose required audio
    ``(t+R)*hop + n_fft <= S`` is emitted at audio time ``S/sr``. Latency is
    measured against the frame's centre, ``(t*hop + n_fft/2)/sr``.
    """
    hop, n_fft, sr = model.hop_length, model.n_fft, model.sample_rate
    chunk_samples = model.chunk_frames * hop
    latencies: List[float] = []
    emitted = 0
    total_frames = (num_samples - n_fft) // hop + 1 if num_samples >= n_fft else 0

    consumed = 0
    while consumed < num_samples:
        consumed = min(consumed + chunk_samples, num_samples)
        emit_time = consumed / sr
        while emitted < total_frames:
            t = emitted
            required = (t + model.right_context) * hop + n_fft
            if required > consumed:
                break
            centre = (t * hop + n_fft / 2.0) / sr
            latencies.append(emit_time - centre)
            emitted += 1
    return LatencyMeasurement(latencies)
