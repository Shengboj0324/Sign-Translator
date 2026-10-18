"""Registered acquired sources: honest capability and rights records."""
from signtranslator.data_engineering.source_portfolio import (
    CURRENT_PRE_PHASE_2_DECISION, CURRENT_SOURCE_CANDIDATES, IntendedUse, RightsStatus,
)

SOURCES = {s.source_id: s for s in CURRENT_SOURCE_CANDIDATES}


def failures(requirement_id, source_id):
    decision = next(d for d in CURRENT_PRE_PHASE_2_DECISION.bundle_decisions
                    if d.requirement_id == requirement_id)
    return dict(decision.source_failures)[source_id]


def test_three_d_lex_is_isolated_and_cannot_satisfy_continuous_coobservation():
    lex = SOURCES["three_d_lex_asl"]
    assert lex.rights_for(IntendedUse.RESEARCH_TRAINING) is RightsStatus.PERMITTED
    assert lex.rights_for(IntendedUse.COMMERCIAL_TRAINING) is RightsStatus.UNRESOLVED
    reasons = failures("continuous_coobserved_3d_research", "three_d_lex_asl")
    assert set(reasons) == {f"missing_capability:{c}" for c in (
        "continuous_asl", "text_alignment", "face_expression", "eye_gaze", "eyelid_state")}


def test_signbank_is_noncommercial_lexicon_only():
    bank = SOURCES["asl_signbank_ecv"]
    assert bank.rights_for(IntendedUse.COMMERCIAL_TRAINING) is RightsStatus.PROHIBITED
    assert {c.capability for c in bank.capabilities} == {"lexical_gloss_inventory"}
    assert "missing_capability:authentic_gloss" in failures("continuous_linguistic_reference", "asl_signbank_ecv")


def test_gate_stays_closed_with_new_sources():
    assert not CURRENT_PRE_PHASE_2_DECISION.approved


def test_estimated_and_licensed_sources_are_recorded_honestly():
    estimate = SOURCES["how2sign_video_estimate"]
    reasons = failures("continuous_coobserved_3d_research", "how2sign_video_estimate")
    for capability in ("body_3d", "left_hand_3d", "right_hand_3d", "face_expression", "head_pose", "eye_gaze"):
        assert f"missing_capability:{capability}" in reasons  # estimates never count as capture
    assert estimate.rights_for(IntendedUse.COMMERCIAL_TRAINING) is RightsStatus.PROHIBITED
    assert SOURCES["sapa_asl_lex_phonology"].rights_for(IntendedUse.COMMERCIAL_TRAINING) is RightsStatus.PROHIBITED
    assert SOURCES["cokely_afs_parallel_corpus"].rights_for(IntendedUse.RESEARCH_TRAINING) is RightsStatus.PERMITTED
