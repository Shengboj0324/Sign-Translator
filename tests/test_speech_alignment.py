"""Verification of CTC forced alignment and word timestamps.

The Viterbi score is proved against exhaustive enumeration: for small inputs
every alignment is generated, those collapsing to the target are kept, and the
maximum of that set must equal the reported score.
"""

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from signtranslator.speech.alignment import (
    FrameTimeMapper, TokenTiming, extended_targets, minimum_frames_required,
    ctc_forced_alignment, token_timings, align_and_time, Alignment,
)
from signtranslator.speech.decoding import collapse


def _random_log_probs(T, C, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.log_softmax(torch.randn(T, C, generator=g, dtype=torch.float64), dim=-1)


def _brute_force_best_alignment(lp, targets, blank=0):
    """Max score over all paths that collapse to `targets` (exponential)."""
    T, C = lp.shape
    best = -float("inf")
    tgt = tuple(targets)
    for path in itertools.product(range(C), repeat=T):
        if collapse(path, blank) != tgt:
            continue
        best = max(best, float(sum(lp[t, c] for t, c in enumerate(path))))
    return best


# ---------------------------------------------------------------------------
# Extended sequence & length bounds
# ---------------------------------------------------------------------------
def test_extended_sequence_interleaves_blanks():
    assert extended_targets([1, 2, 3], blank=0) == [0, 1, 0, 2, 0, 3, 0]
    assert extended_targets([], blank=0) == [0]


def test_minimum_frames_accounts_for_repeats():
    assert minimum_frames_required([1, 2, 3]) == 3
    assert minimum_frames_required([1, 1]) == 3          # blank must separate
    assert minimum_frames_required([1, 1, 1]) == 5
    assert minimum_frames_required([]) == 0


def test_alignment_rejects_impossible_lengths():
    lp = _random_log_probs(2, 4)
    with pytest.raises(ValueError, match="at least"):
        ctc_forced_alignment(lp, [1, 1])                 # needs 3 frames


def test_alignment_rejects_blank_in_targets():
    with pytest.raises(ValueError):
        ctc_forced_alignment(_random_log_probs(6, 4), [1, 0, 2])


def test_alignment_rejects_out_of_range_tokens():
    with pytest.raises(ValueError):
        ctc_forced_alignment(_random_log_probs(6, 4), [1, 9])


def test_alignment_rejects_bad_rank():
    with pytest.raises(ValueError):
        ctc_forced_alignment(torch.randn(2, 3, 4), [1])


# ---------------------------------------------------------------------------
# THE PROOF: Viterbi score == best collapsing path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("targets,T,C,seed", [
    ([1, 2], 5, 3, 0),
    ([1, 1], 5, 3, 1),          # the repeat case, needing a separating blank
    ([2, 1, 2], 6, 3, 2),
    ([1], 4, 3, 3),
])
def test_viterbi_score_matches_exhaustive_maximum(targets, T, C, seed):
    lp = _random_log_probs(T, C, seed)
    al = ctc_forced_alignment(lp, targets, blank=0)
    ref = _brute_force_best_alignment(lp.numpy(), targets, blank=0)
    assert abs(al.score - ref) < 1e-9, f"{al.score} vs {ref}"


def test_recovered_path_collapses_to_the_target():
    """The alignment must actually spell the target under the collapse rule."""
    for targets, seed in ([1, 2, 1], 4), ([3, 3], 5), ([1, 2, 3], 6):
        lp = _random_log_probs(8, 4, seed)
        al = ctc_forced_alignment(lp, targets)
        assert collapse(al.state_tokens(), blank=0) == tuple(targets)


def test_path_score_equals_sum_of_chosen_frame_logprobs():
    """Internal consistency: the reported score is the path's own probability."""
    lp = _random_log_probs(9, 4, seed=7)
    al = ctc_forced_alignment(lp, [1, 2, 3])
    manual = sum(float(lp[t, al.extended[s]]) for t, s in enumerate(al.path))
    assert abs(al.score - manual) < 1e-9


# ---------------------------------------------------------------------------
# Path structure
# ---------------------------------------------------------------------------
def test_path_is_monotonic_and_advances_by_at_most_two():
    lp = _random_log_probs(12, 5, seed=8)
    al = ctc_forced_alignment(lp, [1, 2, 3, 4])
    for a, b in zip(al.path, al.path[1:]):
        assert 0 <= b - a <= 2, f"illegal transition {a}->{b}"


def test_path_starts_and_ends_in_valid_states():
    lp = _random_log_probs(10, 4, seed=9)
    al = ctc_forced_alignment(lp, [1, 2])
    S = len(al.extended)
    assert al.path[0] in (0, 1)
    assert al.path[-1] in (S - 1, S - 2)


def test_two_state_skip_never_crosses_identical_labels():
    """A repeat's separating blank may not be skipped."""
    lp = _random_log_probs(9, 4, seed=10)
    al = ctc_forced_alignment(lp, [2, 2])
    for a, b in zip(al.path, al.path[1:]):
        if b - a == 2:
            assert al.extended[b] != al.extended[b - 2]


def test_alignment_recovers_hand_constructed_segments():
    """Constructed so the true boundaries are unambiguous."""
    # frames: [1,1,1, blank, 2,2, blank, 3,3,3]
    truth = [1, 1, 1, 0, 2, 2, 0, 3, 3, 3]
    p = torch.full((len(truth), 4), 0.001, dtype=torch.float64)
    for t, c in enumerate(truth):
        p[t, c] = 0.997
    al = ctc_forced_alignment(p.log(), [1, 2, 3])
    timings = token_timings(al)
    assert (timings[0].start_frame, timings[0].end_frame) == (0, 2)
    assert (timings[1].start_frame, timings[1].end_frame) == (4, 5)
    assert (timings[2].start_frame, timings[2].end_frame) == (7, 9)


# ---------------------------------------------------------------------------
# Frame <-> time
# ---------------------------------------------------------------------------
def test_frame_time_mapper_closed_form():
    m = FrameTimeMapper(hop_length=160, sample_rate=16000, n_fft=400, subsample=1)
    assert abs(m.start_s(0) - 0.0) < 1e-12
    assert abs(m.end_s(0) - 400 / 16000) < 1e-12          # 25 ms window
    assert abs(m.start_s(10) - 0.1) < 1e-12               # 10 frames x 10 ms
    assert abs(m.center_s(0) - 0.5 * (0.0 + 0.025)) < 1e-12


def test_frame_time_mapper_accounts_for_subsampling():
    """A stride-2 encoder frame covers twice the audio."""
    m = FrameTimeMapper(hop_length=160, sample_rate=16000, n_fft=400, subsample=2)
    assert abs(m.start_s(1) - (2 * 160) / 16000) < 1e-12
    assert abs(m.end_s(0) - (1 * 160 + 400) / 16000) < 1e-12
    assert m.end_s(0) - m.start_s(0) > FrameTimeMapper().end_s(0)


def test_mapper_validates_arguments():
    with pytest.raises(ValueError):
        FrameTimeMapper(subsample=0)
    with pytest.raises(ValueError):
        FrameTimeMapper(hop_length=0)


def test_duration_of_empty_sequence_is_zero():
    assert FrameTimeMapper().duration_s(0) == 0.0


# ---------------------------------------------------------------------------
# Timings
# ---------------------------------------------------------------------------
def test_timings_are_ordered_non_overlapping_and_bounded():
    lp = _random_log_probs(20, 5, seed=11)
    mapper = FrameTimeMapper()
    al, timings = align_and_time(lp, [1, 2, 3, 4], mapper=mapper)
    assert len(timings) == 4
    for a, b in zip(timings, timings[1:]):
        assert a.end_frame < b.start_frame      # strictly ordered, disjoint
        assert a.start_s <= a.end_s
    assert timings[-1].end_s <= mapper.duration_s(lp.shape[0]) + 1e-9
    assert timings[0].start_s >= 0.0


def test_timing_durations_are_positive():
    lp = _random_log_probs(15, 4, seed=12)
    _, timings = align_and_time(lp, [1, 2, 3])
    assert all(t.duration_s > 0 for t in timings)


def test_timing_scores_are_log_posteriors_of_their_token():
    lp = _random_log_probs(12, 4, seed=13)
    al = ctc_forced_alignment(lp, [1, 2])
    timings = token_timings(al, log_probs=lp)
    for t in timings:
        frames = range(t.start_frame, t.end_frame + 1)
        expected = sum(float(lp[f, t.token]) for f in frames) / len(list(frames))
        assert abs(t.score - expected) < 1e-9
        assert t.score <= 0.0


def test_timings_reflect_real_audio_rate():
    """A 100 Hz frame rate must yield timestamps in a plausible second range."""
    lp = _random_log_probs(100, 5, seed=14)          # 100 frames = 1 s
    mapper = FrameTimeMapper()
    _, timings = align_and_time(lp, [1, 2, 3], mapper=mapper)
    assert 0.0 <= timings[0].start_s < 1.05
    assert timings[-1].end_s <= 1.05
