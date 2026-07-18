"""Verification of streaming extraction and the latency model.

Two properties are proved:
  * streaming output equals offline output exactly (the cross-chunk overlap
    must be carried correctly), and
  * measured emission latency matches the independently-derived closed form.
"""

import math

import pytest
import torch

from signtranslator.speech.features import (
    LogMelSpectrogram, N_FFT, HOP_LENGTH, SAMPLE_RATE, num_frames,
)
from signtranslator.speech.streaming import (
    StreamingFeatureExtractor, LatencyModel, LatencyMeasurement,
    measure_emission_latency, percentile,
)

SR = SAMPLE_RATE


def _audio(seconds=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(int(seconds * SR), generator=g)


def _causal_front_end(**kw):
    """Streaming-safe front-end: no utterance-global dynamic-range floor."""
    kw.setdefault("floor_mode", "none")
    return LogMelSpectrogram(**kw)


# ---------------------------------------------------------------------------
# Streaming == offline
# ---------------------------------------------------------------------------
def test_streaming_matches_offline_uniform_chunks():
    """The core invariant: incremental extraction reproduces batch extraction."""
    fe = _causal_front_end()
    x = _audio(1.0)
    offline = fe(x)

    stream = StreamingFeatureExtractor(fe)
    pieces = []
    chunk = 1600                                   # 100 ms
    for i in range(0, x.numel(), chunk):
        out = stream.push(x[i:i + chunk])
        if out.shape[1]:
            pieces.append(out)
    streamed = torch.cat(pieces, dim=1)

    assert streamed.shape == offline.shape
    assert torch.allclose(streamed, offline, atol=1e-5)


def test_streaming_matches_offline_ragged_chunks():
    """Chunk boundaries must not need to align with hops."""
    fe = _causal_front_end()
    x = _audio(0.8, seed=1)
    offline = fe(x)

    stream = StreamingFeatureExtractor(fe)
    g = torch.Generator().manual_seed(7)
    pieces, i = [], 0
    while i < x.numel():
        n = int(torch.randint(37, 999, (1,), generator=g))   # deliberately odd
        out = stream.push(x[i:i + n])
        if out.shape[1]:
            pieces.append(out)
        i += n
    streamed = torch.cat(pieces, dim=1)
    assert streamed.shape == offline.shape
    # float32 tolerance: frame alignment is exact (proved in float64 below);
    # the residual is FFT accumulation order varying with the length of the
    # buffer handed to torch.stft, and is ~1e-4 on log-mel values of order 1.
    assert torch.allclose(streamed, offline, atol=2e-3)


def _tonal_with_silence(seconds=0.5, f0=200.0):
    """Tone bursts separated by true silence -- unlike random noise, this drives
    many mel channels to the clamp floor."""
    parts = []
    for _ in range(3):
        n = int(0.12 * SR)
        t = torch.arange(n, dtype=torch.float64) / SR
        parts.append(torch.sin(2 * math.pi * f0 * t) * 0.5)
        parts.append(torch.zeros(int(0.05 * SR), dtype=torch.float64))
    return torch.cat(parts)


def test_streaming_equals_offline_exactly_on_tonal_audio():
    """REGRESSION: silence in the signal exposes log-amplified float32 error.

    Every earlier streaming test used random noise, in which no mel channel is
    ever near zero. Real speech has silence, which drives channels to the 1e-10
    clamp floor where log10 amplifies rounding without bound (a 1.2e-10 vs
    1.4e-10 difference moves the log by ~0.014). In float64 the streaming logic
    is exact, proving the amplification -- not the alignment -- is responsible.
    """
    fe = _causal_front_end().double()
    x = _tonal_with_silence()
    offline = fe(x)
    stream = StreamingFeatureExtractor(fe)
    pieces = [stream.push(x[i:i + 1600]) for i in range(0, x.numel(), 1600)]
    streamed = torch.cat([p for p in pieces if p.shape[1]], dim=1)
    assert streamed.shape == offline.shape
    assert torch.allclose(streamed, offline, atol=1e-9)


def test_float32_streaming_on_tonal_audio_agrees_where_there_is_signal():
    """In float32 the agreement holds wherever a channel carries energy."""
    fe = _causal_front_end()
    x = _tonal_with_silence().float()
    offline = fe(x)
    stream = StreamingFeatureExtractor(fe)
    pieces = [stream.push(x[i:i + 1600]) for i in range(0, x.numel(), 1600)]
    streamed = torch.cat([p for p in pieces if p.shape[1]], dim=1)
    live = fe.mel_energies(x) > 1e-8
    assert bool(live.any())
    assert torch.allclose(streamed[live], offline[live], atol=2e-3)


def test_streaming_equals_offline_exactly_in_float64():
    """The alignment logic is exact; only float32 rounding separates the two.

    Repeating the ragged-chunk experiment in double precision drops the error
    by ~9 orders of magnitude, which proves the residual in the float32 tests
    is numerical rather than a frame-offset bug.
    """
    fe = _causal_front_end().double()
    x = _audio(0.8, seed=1).double()
    offline = fe(x)

    stream = StreamingFeatureExtractor(fe)
    g = torch.Generator().manual_seed(7)
    pieces, i = [], 0
    while i < x.numel():
        n = int(torch.randint(37, 999, (1,), generator=g))
        out = stream.push(x[i:i + n])
        if out.shape[1]:
            pieces.append(out)
        i += n
    streamed = torch.cat(pieces, dim=1)
    assert streamed.shape == offline.shape
    assert torch.allclose(streamed, offline, atol=1e-9)


def test_streaming_sample_by_sample_matches_offline():
    """The pathological case: one sample at a time."""
    fe = _causal_front_end(n_fft=64, hop_length=16, n_mels=8)
    x = _audio(0.02, seed=2)
    offline = fe(x)
    stream = StreamingFeatureExtractor(fe)
    pieces = [stream.push(x[i:i + 1]) for i in range(x.numel())]
    streamed = torch.cat([p for p in pieces if p.shape[1]], dim=1)
    assert streamed.shape == offline.shape
    assert torch.allclose(streamed, offline, atol=2e-3)   # float32; see float64 test


def test_frame_count_matches_analytic_formula():
    fe = _causal_front_end()
    x = _audio(0.5, seed=3)
    stream = StreamingFeatureExtractor(fe)
    total = 0
    for i in range(0, x.numel(), 800):
        total += stream.push(x[i:i + 800]).shape[1]
    assert total == num_frames(x.numel(), N_FFT, HOP_LENGTH)
    assert stream.frames_emitted == total


def test_no_output_before_first_full_window():
    """Nothing can be emitted until n_fft samples have arrived."""
    stream = StreamingFeatureExtractor()
    assert stream.push(torch.randn(N_FFT - 1)).shape[1] == 0
    assert stream.push(torch.randn(1)).shape[1] == 1     # exactly one frame now


def test_overlap_tail_is_retained():
    """After emitting k frames the buffer must keep n_fft - hop samples."""
    stream = StreamingFeatureExtractor()
    stream.push(torch.randn(N_FFT))
    assert stream.buffered_samples == N_FFT - HOP_LENGTH


def test_reset_clears_state():
    stream = StreamingFeatureExtractor()
    stream.push(torch.randn(4000))
    stream.reset()
    assert stream.frames_emitted == 0 and stream.buffered_samples == 0


def test_streaming_rejects_centered_front_end():
    """A centred STFT pads the signal and cannot be produced causally."""
    with pytest.raises(ValueError):
        StreamingFeatureExtractor(LogMelSpectrogram(center=True, floor_mode="none"))


def test_streaming_rejects_non_causal_global_floor():
    """REGRESSION: the Whisper global dynamic-range floor is non-causal.

    It floors against the maximum over the whole utterance, so chunked features
    cannot equal offline ones. The extractor must refuse rather than silently
    emit features that disagree with training-time ones.
    """
    with pytest.raises(ValueError, match="causal"):
        StreamingFeatureExtractor(LogMelSpectrogram(floor_mode="global"))


def test_causality_flag_reflects_floor_mode():
    assert not LogMelSpectrogram(floor_mode="global").is_causal
    assert LogMelSpectrogram(floor_mode="fixed").is_causal
    assert LogMelSpectrogram(floor_mode="none").is_causal
    # Disabling the dynamic range makes even "global" causal (nothing to floor).
    assert LogMelSpectrogram(floor_mode="global", dynamic_range_db=None).is_causal


def test_fixed_floor_is_applied_and_streaming_safe():
    fe = LogMelSpectrogram(normalize=False, floor_mode="fixed",
                           fixed_floor_log10=-3.0)
    out = fe(torch.zeros(4000))                     # silence -> clamped at floor
    assert float(out.min()) >= -3.0 - 1e-6
    StreamingFeatureExtractor(fe)                   # must not raise


def test_push_rejects_bad_rank():
    with pytest.raises(ValueError):
        StreamingFeatureExtractor().push(torch.randn(2, 100))


# ---------------------------------------------------------------------------
# Latency model
# ---------------------------------------------------------------------------
def test_algorithmic_latency_closed_form():
    """l = (R*hop + n_fft/2)/sr, independent of chunk size."""
    m = LatencyModel(right_context=4, chunk_frames=1)
    expected = (4 * HOP_LENGTH + N_FFT / 2) / SR
    assert abs(m.algorithmic_latency_s - expected) < 1e-12
    # 4 frames of lookahead = 40 ms, plus half a 25 ms window = 52.5 ms
    assert abs(m.algorithmic_latency_s - 0.0525) < 1e-9


def test_excess_set_when_window_is_a_multiple_of_hop():
    """rho = n_fft mod hop == 0: waits are exactly {0, hop, ..., (C-1)*hop}."""
    m = LatencyModel(n_fft=320, hop_length=160, chunk_frames=8, right_context=0)
    assert m.n_fft % m.hop_length == 0
    assert m.excess_samples() == [j * 160 for j in range(8)]
    period = 160 / SR
    assert abs(m.min_latency_s - m.algorithmic_latency_s) < 1e-12
    assert abs(m.max_latency_s - (m.algorithmic_latency_s + 7 * period)) < 1e-12


def test_excess_set_when_window_is_not_a_multiple_of_hop():
    """Whisper's 400/160 gives rho = 80, so waits are {hop-rho, ..., C*hop-rho}.

    The naive "(C-1) frame periods" bound is wrong here by a full frame; this
    is the discrepancy the measured-vs-analytic cross-check exposed.
    """
    m = LatencyModel(n_fft=400, hop_length=160, chunk_frames=8, right_context=0)
    rho = m.n_fft % m.hop_length
    assert rho == 80
    assert m.excess_samples() == sorted(j * 160 - rho for j in range(1, 9))
    assert min(m.excess_samples()) == 80          # hop - rho
    assert max(m.excess_samples()) == 1200        # C*hop - rho


def test_chunking_delay_is_bounded_by_one_chunk():
    """Whatever the alignment, the extra wait is under one full chunk."""
    for chunk in (1, 4, 8, 16):
        m = LatencyModel(chunk_frames=chunk)
        span = max(m.excess_samples()) - min(m.excess_samples())
        assert span < chunk * HOP_LENGTH
        assert m.min_latency_s <= m.mean_latency_s <= m.max_latency_s


def test_zero_lookahead_still_has_window_latency():
    """Even with no lookahead, half the analysis window is unavoidable."""
    m = LatencyModel(right_context=0, chunk_frames=1)
    assert abs(m.algorithmic_latency_s - (N_FFT / 2) / SR) < 1e-12


def test_latency_model_validates_arguments():
    with pytest.raises(ValueError):
        LatencyModel(chunk_frames=0)
    with pytest.raises(ValueError):
        LatencyModel(right_context=-1)


@pytest.mark.parametrize("chunk,right", [(1, 0), (4, 0), (8, 4), (16, 2)])
def test_measured_latency_matches_closed_form(chunk, right):
    """Cross-validation: simulation vs. independently-derived algebra."""
    m = LatencyModel(chunk_frames=chunk, right_context=right)
    meas = measure_emission_latency(SR, m)              # 1 s of audio
    assert len(meas.latencies_s) > 0
    tol = 1e-9
    assert min(meas.latencies_s) >= m.min_latency_s - tol
    assert meas.max_s <= m.max_latency_s + tol
    # Mean over chunk positions should sit near the analytic mean.
    mean = sum(meas.latencies_s) / len(meas.latencies_s)
    assert abs(mean - m.mean_latency_s) < 1.5 * (HOP_LENGTH / SR)


def test_median_and_p95_are_reported_and_ordered():
    m = LatencyModel(chunk_frames=8, right_context=2)
    meas = measure_emission_latency(SR, m)
    assert meas.median_s <= meas.p95_s <= meas.max_s
    assert "median" in meas.report() and "p95" in meas.report()


def test_larger_right_context_strictly_increases_latency():
    base = measure_emission_latency(SR, LatencyModel(chunk_frames=4, right_context=0))
    more = measure_emission_latency(SR, LatencyModel(chunk_frames=4, right_context=8))
    assert more.median_s > base.median_s


def test_describe_reports_chunk_and_right_context():
    """The spec requires these to be stated explicitly, not implied."""
    text = LatencyModel(chunk_frames=8, right_context=4).describe()
    assert "chunk=8" in text and "right_context=4" in text and "ms" in text


def test_percentile_matches_manual_values():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(percentile(xs, 0) - 1.0) < 1e-12
    assert abs(percentile(xs, 100) - 5.0) < 1e-12
    assert abs(percentile(xs, 50) - 3.0) < 1e-12
    assert abs(percentile(xs, 25) - 2.0) < 1e-12       # linear interpolation


def test_percentile_rejects_empty():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_end_to_end_streaming_with_latency_budget():
    """Integration: stream 1 s of audio and confirm the reported budget holds."""
    fe = _causal_front_end()
    model = LatencyModel(chunk_frames=8, right_context=4)
    stream = StreamingFeatureExtractor(fe)
    x = _audio(1.0, seed=5)
    chunk_samples = model.chunk_frames * HOP_LENGTH
    frames = 0
    for i in range(0, x.numel(), chunk_samples):
        frames += stream.push(x[i:i + chunk_samples]).shape[1]
    assert frames == num_frames(x.numel(), N_FFT, HOP_LENGTH)
    meas = measure_emission_latency(x.numel(), model)
    assert meas.p95_s <= model.max_latency_s + 1e-9
    assert model.max_latency_s < 0.15          # under 150 ms for this config
