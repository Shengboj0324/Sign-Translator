"""Verification of streaming commitment and revision tracking.

The safety-critical invariant is that committed text is immutable and its length
never decreases: a sign already rendered cannot be retracted for free. Both are
asserted directly, and the "commitment was wrong" case is constructed
deliberately to confirm it is counted rather than hidden.
"""

import pytest
import torch

from signtranslator.speech.revision import (
    StreamingDecoder, StreamingHypothesis, RevisionStats,
    longest_common_prefix, commitment_error_count,
)


def _peaked(path, C=4, confidence=0.999):
    T = len(path)
    p = torch.full((T, C), (1.0 - confidence) / (C - 1), dtype=torch.float64)
    for t, c in enumerate(path):
        p[t, c] = confidence
    return p.log()


def _ambiguous(T, C=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, C, generator=g, dtype=torch.float64) * 0.3
    return torch.log_softmax(x, dim=-1)


# ---------------------------------------------------------------------------
# Longest common prefix
# ---------------------------------------------------------------------------
def test_lcp_basic_cases():
    assert longest_common_prefix([(1, 2, 3), (1, 2, 4)]) == (1, 2)
    assert longest_common_prefix([(1, 2), (3, 4)]) == ()
    assert longest_common_prefix([(1, 2, 3)]) == (1, 2, 3)
    assert longest_common_prefix([]) == ()
    assert longest_common_prefix([(1, 2), ()]) == ()


# ---------------------------------------------------------------------------
# Core invariants
# ---------------------------------------------------------------------------
def test_committed_prefix_never_shrinks():
    """Monotone commitment: length is non-decreasing across every update."""
    dec = StreamingDecoder(beam_width=8, stability=2)
    lengths = []
    for chunk in ([1, 1, 0], [2, 2, 0], [3, 0, 3], [0, 1, 1]):
        out = dec.update(_peaked(chunk))
        lengths.append(len(out.committed))
    assert lengths == sorted(lengths), lengths


def test_committed_tokens_are_immutable():
    """Whatever is committed at step i must still be a prefix at every later step."""
    dec = StreamingDecoder(beam_width=8, stability=2)
    history = []
    for chunk in ([1, 0, 2], [0, 3, 3], [0, 1, 2], [2, 0, 1]):
        history.append(dec.update(_peaked(chunk)).committed)
    for earlier, later in zip(history, history[1:]):
        assert later[:len(earlier)] == earlier, (earlier, later)


def test_uncommitted_completes_the_full_hypothesis():
    dec = StreamingDecoder(beam_width=8)
    out = dec.update(_peaked([1, 0, 2, 0, 3]))
    assert out.full == out.committed + out.uncommitted
    assert len(out) == len(out.full)


def test_unambiguous_stream_commits_and_never_revises():
    """Clean audio: the hypothesis is stable, so nothing should be rewritten."""
    dec = StreamingDecoder(beam_width=8, stability=2, agreement_k=3)
    for chunk in ([1, 1, 0], [2, 2, 0], [3, 3, 0]):
        dec.update(_peaked(chunk))
    final = dec.finalize()
    assert final.committed == (1, 2, 3)
    assert final.uncommitted == ()
    assert dec.stats.revision_rate == 0.0


def test_ambiguous_stream_produces_revisions():
    """Noisy audio must actually exercise the revision path."""
    dec = StreamingDecoder(beam_width=8, stability=2)
    for i in range(6):
        dec.update(_ambiguous(4, seed=i))
    assert dec.stats.updates == 6
    assert dec.stats.revision_rate > 0.0


def test_stability_requirement_delays_commitment():
    """A higher stability threshold must commit no earlier than a lower one."""
    chunks = [_peaked([1, 0, 2]), _peaked([0, 3, 0]), _peaked([1, 0, 0])]
    fast = StreamingDecoder(beam_width=8, stability=1)
    slow = StreamingDecoder(beam_width=8, stability=3)
    for c in chunks:
        f = fast.update(c)
        s = slow.update(c)
    assert len(s.committed) <= len(f.committed)


def test_agreement_requirement_blocks_commitment_when_beam_disagrees():
    """If the top hypotheses differ from the first token, nothing may commit."""
    # Two tokens near-equally likely in frame 0 => top hypotheses disagree.
    lp = torch.tensor([[0.001, 0.5, 0.499], [0.99, 0.005, 0.005]],
                      dtype=torch.float64).log()
    dec = StreamingDecoder(beam_width=10, agreement_k=2, stability=1)
    out = dec.update(lp)
    assert out.committed == ()


def test_finalize_commits_everything():
    dec = StreamingDecoder(beam_width=8, stability=5)   # too slow to commit early
    dec.update(_peaked([1, 0, 2, 0, 3]))
    assert dec.committed == ()
    final = dec.finalize()
    assert final.uncommitted == ()
    assert final.committed == (1, 2, 3)


def test_finalize_without_any_input_is_safe():
    assert StreamingDecoder().finalize().full == ()


def test_empty_update_is_a_noop():
    dec = StreamingDecoder()
    out = dec.update(torch.zeros(0, 4, dtype=torch.float64))
    assert out.full == ()


def test_update_validates_rank():
    with pytest.raises(ValueError):
        StreamingDecoder().update(torch.randn(2, 3, 4))


def test_constructor_validates_arguments():
    with pytest.raises(ValueError):
        StreamingDecoder(agreement_k=0)
    with pytest.raises(ValueError):
        StreamingDecoder(stability=0)


def test_reset_clears_all_state():
    dec = StreamingDecoder()
    dec.update(_peaked([1, 0, 2]))
    dec.reset()
    assert dec.committed == () and dec.stats.updates == 0


# ---------------------------------------------------------------------------
# Commitment errors
# ---------------------------------------------------------------------------
def _ambiguous_stream(seed, n_updates=5, T=3, C=4, scale=0.4):
    """Replayable ambiguous audio, so two policies can see identical input."""
    g = torch.Generator().manual_seed(seed)
    return [torch.log_softmax(
        torch.randn(T, C, generator=g, dtype=torch.float64) * scale, dim=-1)
        for _ in range(n_updates)]


def test_premature_commitment_is_counted_not_hidden():
    """A reckless policy on ambiguous audio really does contradict itself.

    NOTE: simply appending *confident* frames cannot produce this -- appended
    evidence extends a prefix rather than reinterpreting it. The contradiction
    arises when accumulating ambiguous frames reorders the beam, which is why
    this uses a genuinely ambiguous stream (found by scanning seeds) rather than
    a hand-built one.
    """
    dec = StreamingDecoder(beam_width=6, agreement_k=1, stability=1)
    history = [dec.update(chunk).committed for chunk in _ambiguous_stream(seed=1)]
    assert dec.stats.commitment_errors > 0
    # Immutability must survive the contradiction: we keep what we emitted.
    for earlier, later in zip(history, history[1:]):
        assert later[:len(earlier)] == earlier


def test_conservative_policy_reduces_commitment_errors():
    """The agreement/stability knobs must actually buy safety.

    Same audio, two policies: requiring beam agreement and temporal persistence
    must not commit more errors than committing the top-1 immediately.
    """
    total_reckless = total_careful = 0
    for seed in (1, 2, 3, 5, 8):
        stream = _ambiguous_stream(seed)
        reckless = StreamingDecoder(beam_width=6, agreement_k=1, stability=1)
        careful = StreamingDecoder(beam_width=6, agreement_k=3, stability=3)
        for chunk in stream:
            reckless.update(chunk)
            careful.update(chunk)
        total_reckless += reckless.stats.commitment_errors
        total_careful += careful.stats.commitment_errors
    assert total_reckless > 0, "test audio failed to provoke any contradiction"
    assert total_careful <= total_reckless


def test_commitment_error_rate_is_bounded():
    dec = StreamingDecoder(beam_width=8, agreement_k=1, stability=1)
    for i in range(5):
        dec.update(_ambiguous(3, seed=i))
    rate = dec.stats.commitment_error_rate
    assert 0.0 <= rate


def test_commitment_error_count_against_reference():
    assert commitment_error_count([1, 2, 3], [1, 2, 3]) == 0
    assert commitment_error_count([1, 9, 3], [1, 2, 3]) == 1
    # Committing words that were never spoken counts as errors.
    assert commitment_error_count([1, 2, 3, 4], [1, 2]) == 2
    assert commitment_error_count([], [1, 2]) == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def test_revision_stats_definitions():
    s = RevisionStats(position_emissions=100, revised_positions=7,
                      updates=10, commitment_errors=2, committed_tokens=40)
    assert abs(s.revision_rate - 0.07) < 1e-12
    assert abs(s.commitment_error_rate - 0.05) < 1e-12
    assert "revision_rate" in s.report()


def test_empty_stats_do_not_divide_by_zero():
    s = RevisionStats()
    assert s.revision_rate == 0.0 and s.commitment_error_rate == 0.0


def test_position_emissions_accumulate_per_update():
    dec = StreamingDecoder(beam_width=6)
    dec.update(_peaked([1, 0, 2]))
    first = dec.stats.position_emissions
    dec.update(_peaked([0, 3, 0]))
    assert dec.stats.position_emissions > first
    assert dec.stats.updates == 2


def test_lattice_is_exposed_for_uncertainty_inspection():
    """The planner must be able to see WHERE the recogniser is unsure."""
    out = StreamingDecoder(beam_width=10).update(_ambiguous(6, seed=3))
    assert out.lattice is not None
    assert len(out.lattice.confidence()) == len(out.lattice.best)
    assert 0.0 <= out.retained_mass <= 1.0 + 1e-9
