"""Adversarial tests for hard negatives + shortcut falsification (Doc-11, 11d)."""

import math

import pytest
import torch

from signtranslator.grammar.grammar_tests import (
    ControllableASLBuilder, GrammarFeatures,
)
from signtranslator.pretraining.hard_negatives import (
    HARD_NEGATIVE_DIMENSIONS, hard_negative, is_minimal_linguistic_contrast,
    is_licensed_contrast, contrast_changed_fields,
    signer_shortcut_embedding, length_shortcut_embedding, content_embedding,
)
from signtranslator.pretraining.contrast import info_nce_against_negatives

BUILDER = ControllableASLBuilder()
BASE = GrammarFeatures(predicate=10, subject=1, object=2)


@pytest.mark.parametrize("dim", HARD_NEGATIVE_DIMENSIONS)
def test_each_dimension_is_a_genuine_single_feature_contrast(dim):
    # every dimension flips one feature and produces a genuinely different SIR.
    assert is_minimal_linguistic_contrast(BUILDER, BASE, dim)


@pytest.mark.parametrize("dim", ["negated", "question", "aspect", "number",
                                 "role_shift"])
def test_licensed_dimensions_change_only_licensed_fields(dim):
    # these five flips confine their SIR change to the feature's licensed fields.
    assert is_licensed_contrast(BUILDER, BASE, dim)


def test_entity_swap_also_reorders_naming_signs():
    # naming a DIFFERENT referent legitimately reorders manual units -> 'order'
    # changes too, so entity is a genuine contrast but not licensed-subset-clean.
    changed = contrast_changed_fields(BUILDER, BASE, "entity")
    assert "order" in changed
    assert not is_licensed_contrast(BUILDER, BASE, "entity")


def test_hard_negative_changes_exactly_one_feature():
    neg = hard_negative(BASE, "negated")
    assert neg.negated != BASE.negated
    # every other field is untouched.
    for f in ("predicate", "subject", "object", "question", "aspect",
              "plural_subject", "role_shift", "topicalized", "conditional"):
        assert getattr(neg, f) == getattr(BASE, f)


def test_unknown_dimension_rejected():
    with pytest.raises(ValueError):
        hard_negative(BASE, "handedness_typo")


# ---- the document's core claim: shortcut solves random, fails hard ----------
def test_signer_shortcut_solves_random_but_fails_hard():
    tau = 0.05
    num_signers = 4
    # anchor: signer 0. positive: same sample (signer 0).
    anchor = signer_shortcut_embedding([0], num_signers)
    positive = signer_shortcut_embedding([0], num_signers)
    # RANDOM negatives: other samples, DIFFERENT signers.
    rand_neg = signer_shortcut_embedding([1, 2, 3], num_signers)
    # HARD negatives: minimal pair -> SAME signer 0, different linguistic content.
    hard_neg = signer_shortcut_embedding([0, 0], num_signers)

    loss_random = info_nce_against_negatives(anchor, positive, rand_neg, tau)
    loss_hard = info_nce_against_negatives(anchor, positive, hard_neg, tau)

    assert float(loss_random) < 1e-3                       # shortcut "wins"
    # hard negatives share the signer -> shortcut cannot separate -> ~log(1+M).
    assert float(loss_hard) == pytest.approx(math.log(3), abs=1e-2)
    assert float(loss_hard) > 100 * float(loss_random)


def test_length_shortcut_also_fails_on_length_matched_hard_negatives():
    tau = 0.05
    anchor = length_shortcut_embedding([30])
    positive = length_shortcut_embedding([30])
    rand_neg = length_shortcut_embedding([10, 50, 70])      # different length bins
    hard_neg = length_shortcut_embedding([30, 30])          # length-matched bin
    loss_random = info_nce_against_negatives(anchor, positive, rand_neg, tau)
    loss_hard = info_nce_against_negatives(anchor, positive, hard_neg, tau)
    assert float(loss_random) < 1e-3                        # length separates random
    assert float(loss_hard) > float(loss_random)            # fails length-matched


def test_genuine_content_succeeds_on_hard_negatives():
    # a representation encoding the linguistic label distinguishes the minimal pair.
    tau = 0.05
    num_labels = 6
    anchor = content_embedding([0], num_labels)
    positive = content_embedding([0], num_labels)
    hard_neg = content_embedding([1, 2], num_labels)        # different content
    loss_hard = info_nce_against_negatives(anchor, positive, hard_neg, tau)
    assert float(loss_hard) < 1e-3                          # content wins on hard
