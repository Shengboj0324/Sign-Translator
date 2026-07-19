"""Verification of phonological notation, variation preservation, and provenance.

The linguistically-critical property: dialect/register variants are kept
distinct and never collapsed to a "correct" form; and machine-generated glosses
carry provenance and are down-weighted relative to signer-validated ones.
"""

import pytest

from signtranslator.grammar.notation import (
    Handshape, Location, Movement, Orientation, SignPhonology,
    Register, SignVariant, VariationSet, GlossSource, GlossLabel,
)


def _phon(hs=Handshape.FLAT, loc=Location.CHEST, mv=Movement.STRAIGHT,
          ori=Orientation.PALM_OUT, two=False):
    return SignPhonology(hs, loc, mv, ori, two_handed=two)


# ---------------------------------------------------------------------------
# Phonology + SiGML
# ---------------------------------------------------------------------------
def test_phonology_validates_clean_sign():
    assert _phon().validate() == []


def test_sigml_round_trips():
    for two in (False, True):
        p = _phon(two=two)
        assert SignPhonology.from_sigml(p.to_sigml()) == p


def test_sigml_serialization_is_deterministic():
    p = _phon()
    assert p.to_sigml() == p.to_sigml()
    assert "hamshape" in p.to_sigml() and "hamori" in p.to_sigml()


def test_from_sigml_rejects_malformed():
    with pytest.raises(ValueError):
        SignPhonology.from_sigml("<hns></hns>")           # missing parameters


def test_two_handed_flag_survives_serialization():
    assert SignPhonology.from_sigml(_phon(two=True).to_sigml()).two_handed
    assert not SignPhonology.from_sigml(_phon(two=False).to_sigml()).two_handed


# ---------------------------------------------------------------------------
# Variation is preserved, never normalised
# ---------------------------------------------------------------------------
def test_dialect_variants_are_kept_distinct():
    """Two dialect realisations of one concept must both survive."""
    vs = VariationSet(concept=7)
    vs.add(SignVariant(_phon(loc=Location.CHIN), dialect="east"))
    vs.add(SignVariant(_phon(loc=Location.FOREHEAD), dialect="west"))
    assert len(vs) == 2
    assert vs.dialects() == ["east", "west"]
    # crucially, there is NO method that collapses them to one "correct" form
    assert not hasattr(vs, "canonicalize")
    assert len(vs.by_dialect("east")) == 1
    assert len(vs.by_dialect("west")) == 1


def test_register_variants_are_selectable_not_merged():
    vs = VariationSet(concept=3)
    vs.add(SignVariant(_phon(), register=Register.FORMAL))
    vs.add(SignVariant(_phon(mv=Movement.TAP), register=Register.INFORMAL))
    assert len(vs.by_register(Register.FORMAL)) == 1
    assert len(vs.by_register(Register.INFORMAL)) == 1
    assert len(vs) == 2                                   # nothing was normalised


def test_variation_sets_of_the_same_concept_are_recognised():
    a = VariationSet(concept=5)
    b = VariationSet(concept=5)
    c = VariationSet(concept=6)
    assert a.is_variant_of_same_concept(b)
    assert not a.is_variant_of_same_concept(c)


# ---------------------------------------------------------------------------
# Noisy-gloss provenance
# ---------------------------------------------------------------------------
def test_gloss_source_classification():
    gold = GlossLabel(1, GlossSource.SIGNER_VALIDATED)
    auto = GlossLabel(1, GlossSource.AUTO, confidence=0.9)
    assert gold.is_gold and not gold.is_noisy
    assert auto.is_noisy and not auto.is_gold


def test_noisy_labels_are_down_weighted_relative_to_gold():
    """At equal nominal confidence, an auto gloss must weigh less than gold."""
    gold = GlossLabel(1, GlossSource.SIGNER_VALIDATED, confidence=0.9)
    elan = GlossLabel(1, GlossSource.ELAN_TIER, confidence=0.9)
    auto = GlossLabel(1, GlossSource.AUTO, confidence=0.9)
    assert gold.training_weight() == 1.0
    assert elan.training_weight() == 0.9
    assert auto.training_weight() == pytest.approx(0.45)
    assert auto.training_weight() < elan.training_weight() < gold.training_weight()


def test_gloss_confidence_is_validated():
    with pytest.raises(ValueError):
        GlossLabel(1, GlossSource.AUTO, confidence=1.5)
    with pytest.raises(ValueError):
        GlossLabel(1, GlossSource.AUTO, confidence=-0.1)


def test_provenance_lets_a_consumer_prefer_validated_labels():
    """A downstream weigher must be able to distinguish sources -- the whole
    point of carrying provenance rather than a bare gloss id."""
    labels = [GlossLabel(1, GlossSource.AUTO, 0.95),
              GlossLabel(2, GlossSource.SIGNER_VALIDATED, 0.8)]
    best = max(labels, key=lambda l: l.training_weight())
    assert best.gloss == 2                                # validated wins despite lower conf
