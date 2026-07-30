"""Stage 2 integration on real audio: decode, timestamp, and revise.

A recogniser is trained on synthesised tone-burst utterances whose word
boundaries are known by construction, then:

  * prefix beam search must recover the spoken tokens,
  * forced alignment must place each token's timestamps INSIDE the interval
    where that tone actually sounded, and
  * the streaming decoder must commit monotonically while consuming the audio
    in chunks.

The timestamp check is the meaningful one: it validates the alignment against
physical ground truth rather than against the model's own opinion.
"""

import itertools
import math

import pytest
import torch

from signtranslator.speech import (
    LogMelSpectrogram, ctc_prefix_beam_search, ctc_forced_alignment,
    token_timings, FrameTimeMapper, StreamingDecoder, Lattice,
    ctc_greedy_path, collapse,
    SAMPLE_RATE, HOP_LENGTH, N_MELS, N_FFT,
)
from signtranslator.models import SpeechRecognizer

SR = SAMPLE_RATE
WORD_S, GAP_S = 0.20, 0.08


def _utterance(f0s, word_s=WORD_S, gap_s=GAP_S, sr=SR):
    """Tone bursts separated by silence; returns (waveform, word intervals)."""
    parts, spans, cursor = [], [], 0.0
    for f0 in f0s:
        n = int(word_s * sr)
        t = torch.arange(n, dtype=torch.float32) / sr
        x = sum(torch.sin(2 * math.pi * f0 * h * t) / h for h in (1, 2, 3))
        parts.append(x * torch.hann_window(n, periodic=False) * 0.5)
        spans.append((cursor, cursor + word_s))
        cursor += word_s
        parts.append(torch.zeros(int(gap_s * sr)))
        cursor += gap_s
    return torch.cat(parts), spans


# Three distinguishable "words" -> token ids 1, 2, 3
_VOCAB = {120.0: 1, 210.0: 2, 320.0: 3}


def _dataset(front_end):
    # All 6 orderings: with only a subset the model has too little signal to
    # escape CTC's all-blank local optimum within a short training budget.
    seqs = [list(p) for p in itertools.permutations([120.0, 210.0, 320.0])]
    feats, targets, spans = [], [], []
    for s in seqs:
        wav, sp = _utterance(s)
        feats.append(front_end(wav).t().unsqueeze(0))
        targets.append([_VOCAB[f] for f in s])
        spans.append(sp)
    n = min(f.shape[1] for f in feats)
    feats = torch.cat([f[:, :n] for f in feats], dim=0)
    return feats, torch.tensor(targets), spans


@pytest.fixture(scope="module")
def trained():
    """Train a small CTC recogniser on real log-Mel features."""
    torch.manual_seed(0)
    fe = LogMelSpectrogram()
    feats, targets, spans = _dataset(fe)
    rec = SpeechRecognizer(input_dim=N_MELS, num_tokens=3, hidden_dim=96,
                           num_layers=2, num_heads=4, subsample=2)
    lengths = torch.full((feats.shape[0],), targets.shape[1], dtype=torch.long)
    opt = torch.optim.Adam(rec.parameters(), lr=3e-3)
    first = None
    for _ in range(260):
        loss = rec.loss(feats, targets, lengths)
        first = first if first is not None else loss.detach().item()
        opt.zero_grad(); loss.backward(); opt.step()
    rec.eval()
    return rec, fe, feats, targets, spans, first, loss.detach().item()


def test_recognizer_actually_decodes_not_merely_loses_less(trained):
    """Acceptance is DECODING accuracy, not a loss ratio.

    CTC has an all-blank local optimum: a model can cut its loss ~10x while
    emitting blank on 100% of frames and decoding the empty string. An earlier
    version of this test asserted `loss < 0.2 * first` and passed on exactly
    such a useless model. Loss reduction is not evidence of a working decoder.
    """
    rec, _, feats, targets, _, _, _ = trained
    with torch.no_grad():
        log_probs = rec(feats)
    for i in range(feats.shape[0]):
        path = ctc_greedy_path(log_probs[i])
        blank_fraction = sum(1 for c in path if c == 0) / len(path)
        assert blank_fraction < 1.0, "model collapsed to all-blank"
        assert collapse(path, blank=0) == tuple(targets[i].tolist())


def test_beam_search_recovers_the_spoken_tokens(trained):
    rec, _, feats, targets, _, _, _ = trained
    with torch.no_grad():
        log_probs = rec(feats)
    for i in range(feats.shape[0]):
        nbest = ctc_prefix_beam_search(log_probs[i], beam_width=12, blank=0)
        assert nbest.best.tokens == tuple(targets[i].tolist()), (
            f"utterance {i}: {nbest.best.tokens} != {targets[i].tolist()}")


def test_nbest_is_ranked_and_top_hypothesis_dominates(trained):
    rec, _, feats, _, _, _, _ = trained
    with torch.no_grad():
        log_probs = rec(feats)
    nbest = ctc_prefix_beam_search(log_probs[0], beam_width=12)
    post = nbest.posteriors
    assert post == sorted(post, reverse=True)
    assert post[0] > 0.5           # confident after convergence


def test_forced_alignment_timestamps_land_on_the_actual_words(trained):
    """The decisive check: alignment vs. physical ground truth.

    Each token's predicted interval must overlap the interval in which that tone
    genuinely sounded. This validates the frame->time mapping (hop, window,
    subsampling) end to end -- an off-by-one in any of them shows up here.
    """
    rec, _, feats, targets, spans, _, _ = trained
    mapper = FrameTimeMapper(hop_length=HOP_LENGTH, sample_rate=SR,
                             n_fft=N_FFT, subsample=rec.subsample)
    with torch.no_grad():
        log_probs = rec(feats)

    for i in range(feats.shape[0]):
        al = ctc_forced_alignment(log_probs[i], targets[i].tolist(), blank=0)
        timings = token_timings(al, mapper=mapper, log_probs=log_probs[i])
        assert len(timings) == targets.shape[1]
        for timing, (true_start, true_end) in zip(timings, spans[i]):
            # Overlap of predicted [start,end] with the true word interval.
            overlap = min(timing.end_s, true_end) - max(timing.start_s, true_start)
            assert overlap > 0, (
                f"utt {i} token {timing.token}: predicted "
                f"[{timing.start_s:.3f},{timing.end_s:.3f}] misses true "
                f"[{true_start:.3f},{true_end:.3f}]")


def test_timestamps_are_ordered_and_within_the_audio(trained):
    rec, _, feats, targets, _, _, _ = trained
    mapper = FrameTimeMapper(hop_length=HOP_LENGTH, sample_rate=SR,
                             n_fft=N_FFT, subsample=rec.subsample)
    with torch.no_grad():
        log_probs = rec(feats)
    timings = token_timings(
        ctc_forced_alignment(log_probs[0], targets[0].tolist()),
        mapper=mapper)
    duration = mapper.duration_s(log_probs.shape[1])
    for a, b in zip(timings, timings[1:]):
        assert a.end_s <= b.end_s and a.start_frame < b.start_frame
    assert timings[0].start_s >= 0.0
    assert timings[-1].end_s <= duration + 1e-6


def test_streaming_decoder_commits_monotonically_on_real_audio(trained):
    rec, _, feats, targets, _, _, _ = trained
    with torch.no_grad():
        log_probs = rec(feats)[0]

    dec = StreamingDecoder(beam_width=10, agreement_k=2, stability=2)
    lengths = []
    step = 8
    for i in range(0, log_probs.shape[0], step):
        out = dec.update(log_probs[i:i + step])
        lengths.append(len(out.committed))
        assert out.full == out.committed + out.uncommitted
    assert lengths == sorted(lengths)
    final = dec.finalize()
    assert final.committed == tuple(targets[0].tolist())
    assert final.uncommitted == ()


def test_confident_stream_has_low_revision_rate(trained):
    """A converged model on clean audio should rarely rewrite itself."""
    rec, _, feats, _, _, _, _ = trained
    with torch.no_grad():
        log_probs = rec(feats)[0]
    dec = StreamingDecoder(beam_width=10, agreement_k=2, stability=2)
    for i in range(0, log_probs.shape[0], 8):
        dec.update(log_probs[i:i + 8])
    assert dec.stats.revision_rate < 0.25, dec.stats.report()


def test_lattice_exposes_low_confidence_regions(trained):
    """Uncertainty must be visible to the planner (input to fail-closed)."""
    rec, _, feats, _, _, _, _ = trained
    with torch.no_grad():
        log_probs = rec(feats)[0]
    lat = Lattice.from_nbest(ctc_prefix_beam_search(log_probs, beam_width=12))
    conf = lat.confidence()
    assert len(conf) == len(lat.best)
    assert all(0.0 <= c <= 1.0 + 1e-9 for c in conf)
    # On converged, clean audio there should be no ambiguous position.
    assert lat.ambiguous_positions(threshold=0.5) == []
