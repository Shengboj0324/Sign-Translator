"""Verification of the evaluation metrics and report.

Edit-distance counts are checked against hand-worked alignments, because an
S/D/I breakdown derived from a wrong backtrace can still produce the right total
distance -- and the breakdown is precisely what makes the metric diagnostic.
"""

import pytest

from signtranslator.speech.evaluation import (
    EditOps, edit_ops, word_error_rate, character_error_rate, timestamp_error,
    TimestampError, Condition, STANDARD_CONDITIONS, ConditionProfile,
    characterise_condition, ArmResult, EvaluationReport,
)

SPELLING = {1: "cat", 2: "dog", 3: "bird"}


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------
def test_identical_sequences_have_no_errors():
    ops = edit_ops([1, 2, 3], [1, 2, 3])
    assert ops.distance == 0 and ops.error_rate == 0.0
    assert (ops.substitutions, ops.deletions, ops.insertions) == (0, 0, 0)


def test_pure_substitution_is_counted_as_such():
    ops = edit_ops([1, 9, 3], [1, 2, 3])
    assert (ops.substitutions, ops.deletions, ops.insertions) == (1, 0, 0)
    assert abs(ops.error_rate - 1 / 3) < 1e-12


def test_pure_deletion_is_counted_as_such():
    """Hypothesis omits a reference token."""
    ops = edit_ops([1, 3], [1, 2, 3])
    assert (ops.substitutions, ops.deletions, ops.insertions) == (0, 1, 0)
    assert abs(ops.error_rate - 1 / 3) < 1e-12


def test_pure_insertion_is_counted_as_such():
    """Hypothesis contains a token that was never said."""
    ops = edit_ops([1, 2, 9, 3], [1, 2, 3])
    assert (ops.substitutions, ops.deletions, ops.insertions) == (0, 0, 1)


def test_deletions_and_substitutions_are_distinguishable():
    """The reason the breakdown exists: identical WER, opposite failure modes.

    Stage 3 showed this recogniser degrades by deleting, not substituting; a
    scalar WER cannot tell those apart, and they need different fixes.
    """
    deleting = edit_ops([1, 2], [1, 2, 3])
    substituting = edit_ops([1, 2, 9], [1, 2, 3])
    assert deleting.error_rate == substituting.error_rate
    assert deleting.deletions == 1 and deleting.substitutions == 0
    assert substituting.substitutions == 1 and substituting.deletions == 0


def test_empty_hypothesis_is_all_deletions():
    ops = edit_ops([], [1, 2, 3])
    assert ops.deletions == 3 and ops.error_rate == 1.0


def test_empty_reference_is_all_insertions_and_rate_is_defined():
    ops = edit_ops([1, 2], [])
    assert ops.insertions == 2 and ops.reference_length == 0
    assert ops.error_rate == 0.0            # no reference: rate is undefined -> 0


def test_error_rate_can_exceed_one_with_many_insertions():
    ops = edit_ops([1, 2, 3, 4, 5, 6], [1])
    assert ops.error_rate > 1.0


def test_distance_matches_known_levenshtein_values():
    assert edit_ops("kitten", "sitting").distance == 3
    assert edit_ops("flaw", "lawn").distance == 2
    assert edit_ops("", "").distance == 0


def test_ops_are_additive():
    a, b = edit_ops([1, 9], [1, 2]), edit_ops([3], [3, 4])
    total = a + b
    assert total.substitutions == 1 and total.deletions == 1
    assert total.reference_length == 4


# ---------------------------------------------------------------------------
# Corpus-level rates
# ---------------------------------------------------------------------------
def test_corpus_wer_pools_rather_than_averages():
    """Pooling differs from the mean of per-utterance rates, which over-weights
    short utterances. The standard definition pools."""
    hyps = [[1], [1, 2, 3, 4]]
    refs = [[9], [1, 2, 3, 4]]                 # 1 error out of 5 reference words
    pooled = word_error_rate(hyps, refs)
    assert abs(pooled.error_rate - 0.2) < 1e-12
    naive_mean = (1.0 + 0.0) / 2
    assert abs(pooled.error_rate - naive_mean) > 0.2   # genuinely different


def test_wer_validates_alignment_of_inputs():
    with pytest.raises(ValueError):
        word_error_rate([[1]], [[1], [2]])


def test_cer_uses_the_spelling_and_differs_from_wer():
    """One wrong word of three is WER=1/3 but a smaller character error."""
    hyps, refs = [[1, 2, 3]], [[1, 2, 1]]
    wer = word_error_rate(hyps, refs)
    cer = character_error_rate(hyps, refs, SPELLING)
    assert abs(wer.error_rate - 1 / 3) < 1e-12
    assert cer.error_rate != wer.error_rate
    assert cer.reference_length == len("cat dog cat")


def test_cer_is_zero_for_identical_sequences():
    assert character_error_rate([[1, 2]], [[1, 2]], SPELLING).distance == 0


def test_cer_rejects_unspellable_tokens():
    with pytest.raises(KeyError):
        character_error_rate([[7]], [[1]], SPELLING)


# ---------------------------------------------------------------------------
# Timestamp error
# ---------------------------------------------------------------------------
def test_timestamp_error_is_zero_for_exact_predictions():
    spans = [(0.0, 0.2), (0.3, 0.5)]
    err = timestamp_error(spans, spans)
    assert err.mean_start_error_s == 0.0 and err.count == 2


def test_timestamp_error_matches_manual_computation():
    pred = [(0.05, 0.25), (0.35, 0.55)]
    ref = [(0.00, 0.20), (0.30, 0.50)]
    err = timestamp_error(pred, ref)
    assert abs(err.mean_start_error_s - 0.05) < 1e-12
    assert abs(err.mean_end_error_s - 0.05) < 1e-12
    assert abs(err.max_start_error_s - 0.05) < 1e-12


def test_timestamp_error_scores_only_matched_pairs():
    """A timestamp for a token that was never spoken has no reference.

    Scoring it would either flatter or arbitrarily punish the metric; those
    errors belong to WER instead.
    """
    err = timestamp_error([(0.0, 0.1), (0.2, 0.3), (0.4, 0.5)], [(0.0, 0.1)])
    assert err.count == 1 and err.mean_start_error_s == 0.0


def test_timestamp_error_on_empty_input():
    err = timestamp_error([], [])
    assert err.count == 0 and err.mean_start_error_s == 0.0


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
def test_standard_conditions_cover_the_required_set():
    """The document names exactly these five."""
    names = {c.name for c in STANDARD_CONDITIONS}
    assert names == {"clean", "noisy", "accented", "code_switched", "long_form"}


def test_each_condition_perturbs_something():
    baseline = Condition("clean")
    for cond in STANDARD_CONDITIONS:
        if cond.name == "clean":
            continue
        differs = (cond.noise != baseline.noise
                   or cond.pitch_scale != baseline.pitch_scale
                   or cond.vocabulary != baseline.vocabulary
                   or cond.words != baseline.words)
        assert differs, f"{cond.name} is identical to clean"
    assert "noise" in Condition("noisy", noise=0.1).describe()


def test_condition_profile_flags_a_ceiling_as_degenerate():
    """Perfect accuracy distinguishes nothing between systems."""
    p = characterise_condition("clean", [[1, 2, 3]], [[1, 2, 3]])
    assert p.accuracy == 1.0
    assert not p.is_informative


def test_condition_profile_flags_a_collapse_as_degenerate():
    """The Stage 3 failure: the model emits almost nothing, so nothing is measured."""
    p = characterise_condition("noisy", [[], [], [1]], [[1, 2, 3]] * 3)
    assert p.tokens_decoded == 1 and p.tokens_expected == 9
    assert not p.is_informative


def test_condition_profile_accepts_the_informative_band():
    hyps = [[1, 2, 9], [1, 9, 3], [1, 2, 3]]
    refs = [[1, 2, 3]] * 3
    p = characterise_condition("noisy", hyps, refs)
    assert 0.05 <= p.accuracy <= 0.999
    assert p.is_informative
    assert "informative" in p.summary()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _result(arm, cond, hyps, refs):
    return ArmResult(arm=arm, condition=cond, wer=word_error_rate(hyps, refs))


def test_report_groups_by_arm_and_condition():
    rep = EvaluationReport()
    rep.add(_result("fused", "clean", [[1, 2]], [[1, 2]]))
    rep.add(_result("acoustic_only", "clean", [[1, 9]], [[1, 2]]))
    rep.add(_result("fused", "noisy", [[1, 9]], [[1, 2]]))
    assert len(rep.by_arm("fused")) == 2
    assert len(rep.by_condition("clean")) == 2


def test_report_surfaces_degenerate_conditions_prominently():
    """A conclusion drawn on a degenerate condition is unsafe; say so loudly."""
    rep = EvaluationReport()
    rep.profiles.append(characterise_condition("noisy", [[]], [[1, 2, 3]]))
    rep.add(_result("fused", "noisy", [[]], [[1, 2, 3]]))
    assert rep.degenerate_conditions() == ["noisy"]
    assert "WARNING" in rep.summary()


def test_report_states_streaming_configuration_and_latency():
    """The document requires chunk size and right context be explicit."""
    rep = EvaluationReport(streaming_config="chunk=8 frames, right_context=4",
                           latency_median_s=0.075, latency_p95_s=0.12)
    text = rep.summary()
    assert "chunk=8" in text and "right_context=4" in text
    assert "median" in text and "p95" in text


def test_arm_result_summary_includes_supplied_metrics():
    r = ArmResult(arm="fused", condition="clean",
                  wer=word_error_rate([[1]], [[1]]),
                  ece=0.02, revision_rate=0.01, downstream_accuracy=0.9)
    s = r.summary()
    assert "ECE" in s and "rev" in s and "downstream" in s
