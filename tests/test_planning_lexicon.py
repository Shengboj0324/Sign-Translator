"""Verification of the versioned lexicon, retrieval, and hallucination reporting."""

import pytest
import torch

from signtranslator.planning.lexicon import (
    LexEntry, SignLexicon, RetrievalResult, GroundingReport, ground_plan,
)
from signtranslator.planning.schema import SignPlan, SemanticFrame, NonmanualSpan


def _lexicon(n=6, dim=8, seed=0, version="1.0.0"):
    g = torch.Generator().manual_seed(seed)
    entries = [LexEntry(lexeme=i, gloss=f"SIGN_{i}",
                        embedding=tuple(torch.randn(dim, generator=g).tolist()))
               for i in range(n)]
    return SignLexicon(entries, version=version), entries


# ---------------------------------------------------------------------------
# Versioning / content hash
# ---------------------------------------------------------------------------
def test_content_hash_is_deterministic():
    a, entries = _lexicon()
    b = SignLexicon(entries, version="1.0.0")
    assert a.content_hash == b.content_hash
    assert a.fingerprint == b.fingerprint


def test_content_hash_changes_when_an_entry_changes():
    _, entries = _lexicon()
    edited = list(entries)
    edited[0] = LexEntry(lexeme=0, gloss="RENAMED", embedding=entries[0].embedding)
    assert SignLexicon(entries).content_hash != SignLexicon(edited).content_hash


def test_content_hash_changes_when_an_embedding_changes():
    _, entries = _lexicon()
    edited = list(entries)
    emb = list(entries[0].embedding); emb[0] += 1.0
    edited[0] = LexEntry(lexeme=0, gloss=entries[0].gloss, embedding=tuple(emb))
    assert SignLexicon(entries).content_hash != SignLexicon(edited).content_hash


def test_hash_is_order_independent():
    """Reordering the same entries must not change the content hash."""
    _, entries = _lexicon()
    a = SignLexicon(entries)
    b = SignLexicon(list(reversed(entries)))
    assert a.content_hash == b.content_hash


def test_version_is_part_of_the_fingerprint_but_not_the_hash():
    _, entries = _lexicon()
    a = SignLexicon(entries, version="1.0.0")
    b = SignLexicon(entries, version="2.0.0")
    assert a.content_hash == b.content_hash      # content identical
    assert a.fingerprint != b.fingerprint         # but the pinned version differs


def test_lexicon_rejects_malformed_construction():
    with pytest.raises(ValueError):
        SignLexicon([])                                   # empty
    with pytest.raises(ValueError):
        SignLexicon([LexEntry(0, "A", (1.0, 2.0)),
                     LexEntry(1, "B", (1.0,))])          # ragged dims
    with pytest.raises(ValueError):
        SignLexicon([LexEntry(0, "A", (1.0,)),
                     LexEntry(0, "B", (2.0,))])          # duplicate lexeme


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------
def test_membership_and_gloss_lookup():
    lex, _ = _lexicon(n=4)
    assert lex.contains(2) and not lex.contains(99)
    assert lex.gloss_of(2) == "SIGN_2"
    assert lex.gloss_of(99) is None
    assert len(lex) == 4


# ---------------------------------------------------------------------------
# Retrieval D(x)
# ---------------------------------------------------------------------------
def test_retrieval_returns_the_exact_entry_for_its_own_embedding():
    lex, entries = _lexicon(n=6)
    for e in entries:
        top = lex.retrieve(e.as_tensor(), top_k=1)[0]
        assert top.lexeme == e.lexeme                     # self is nearest
        assert abs(top.score - 1.0) < 1e-5                # cosine with itself


def test_retrieval_is_ranked_by_similarity():
    lex, _ = _lexicon(n=6)
    q = torch.randn(8)
    results = lex.retrieve(q, top_k=6)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in scores)


def test_retrieval_top_k_is_bounded_by_lexicon_size():
    lex, _ = _lexicon(n=3)
    assert len(lex.retrieve(torch.randn(8), top_k=10)) == 3


def test_retrieval_results_carry_lexicon_provenance():
    lex, _ = _lexicon(version="3.1.4")
    r = lex.retrieve(torch.randn(8), top_k=1)[0]
    assert r.lexicon_version == "3.1.4"
    assert r.lexicon_hash == lex.content_hash


def test_retrieval_validates_query():
    lex, _ = _lexicon(dim=8)
    with pytest.raises(ValueError):
        lex.retrieve(torch.randn(4))                      # wrong dim
    with pytest.raises(ValueError):
        lex.retrieve(torch.randn(2, 8))                   # not 1-D
    with pytest.raises(ValueError):
        lex.retrieve(torch.randn(8), top_k=0)


def test_provenance_stamp_pins_the_lexicon():
    lex, _ = _lexicon(version="2.2.2")
    stamp = lex.provenance_stamp()
    assert stamp["lexicon_version"] == "2.2.2"
    assert stamp["lexicon_hash"] == lex.content_hash


# ---------------------------------------------------------------------------
# Hallucination / invalid-reference reporting
# ---------------------------------------------------------------------------
def _plan(units, referents, loci, args=(), fingerspelling=()):
    return SignPlan(frame=SemanticFrame(predicate=0, args=list(args)),
                    referents=list(referents), loci=dict(loci),
                    manual_units=list(units), fingerspelling=list(fingerspelling),
                    tam=0, conf_bucket=0)


def test_grounded_plan_reports_no_problems():
    lex, _ = _lexicon(n=6)
    plan = _plan(units=[0, 1, 2], referents=[0, 1], loci={0: 0, 1: 1})
    report = ground_plan(plan, lex, num_loci=7)
    assert report.is_grounded
    assert report.hallucination_rate == 0.0


def test_out_of_lexicon_unit_is_flagged_as_hallucinated():
    lex, _ = _lexicon(n=3)                                 # lexemes 0,1,2
    plan = _plan(units=[0, 1, 99], referents=[0], loci={0: 0})   # 99 absent
    report = ground_plan(plan, lex, num_loci=7)
    assert report.hallucinated_units == [2]                # index of unit 99
    assert not report.is_grounded


def test_fingerspelled_out_of_lexicon_unit_is_not_hallucinated():
    """Fingerspelling is the licensed escape hatch for unknown words."""
    lex, _ = _lexicon(n=3)
    plan = _plan(units=[0, 99], referents=[0], loci={0: 0}, fingerspelling=[1])
    report = ground_plan(plan, lex, num_loci=7)
    assert report.hallucinated_units == []


def test_referent_without_a_locus_is_an_invalid_spatial_reference():
    lex, _ = _lexicon()
    plan = _plan(units=[0], referents=[0, 1], loci={0: 0})   # ref 1 unplaced
    report = ground_plan(plan, lex, num_loci=7)
    assert 1 in report.invalid_spatial_refs


def test_referent_at_out_of_range_locus_is_invalid():
    lex, _ = _lexicon()
    plan = _plan(units=[0], referents=[0], loci={0: 99})     # locus 99 > num_loci
    report = ground_plan(plan, lex, num_loci=7)
    assert 0 in report.invalid_spatial_refs


def test_argument_referent_must_be_placeable():
    """An arg referent with no locus cannot be pointed at in signing space."""
    lex, _ = _lexicon()
    plan = _plan(units=[0], referents=[0], loci={0: 0}, args=[(0, 3)])
    plan.referents.append(3)                                 # declared but...
    report = ground_plan(plan, lex, num_loci=7)              # ...no locus
    assert 3 in report.invalid_spatial_refs


def test_grounding_rates_are_computed():
    lex, _ = _lexicon(n=2)
    plan = _plan(units=[0, 88, 99], referents=[0], loci={0: 0})
    report = ground_plan(plan, lex, num_loci=7)
    assert abs(report.hallucination_rate - 2 / 3) < 1e-9
    assert "hallucinated" in report.summary()
