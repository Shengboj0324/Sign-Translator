"""Versioned research/commercial Phase-2 eligibility without changing legacy gates.

This checks supplied claims and evidence bytes, not legal meaning or reviewer
qualifications. An eligible portfolio is still not a validated model or phase exit.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path

from .schema import DataAuthorization, ConsentState, validate_authorization
from .source_portfolio import (
    SourceCandidate, RequirementBundle, IntendedUse, EvidenceLevel,
    PRE_PHASE_2_REQUIREMENTS, assess_source_portfolio,
)

POLICY_VERSION = 'phase2-action-scope-v1'


class Phase2Scope(str, Enum):
    RESEARCH = 'research'
    COMMERCIAL = 'commercial'


@dataclass(frozen=True)
class AuthorizationEvidence:
    authorization: DataAuthorization
    consent: ConsentState
    local_evidence: Path
    source_id: str


def requirements(scope: Phase2Scope) -> tuple[RequirementBundle, ...]:
    if not isinstance(scope, Phase2Scope):
        raise ValueError('an explicit typed research or commercial scope is required')
    training = (IntendedUse.RESEARCH_TRAINING if scope is Phase2Scope.RESEARCH
                else IntendedUse.COMMERCIAL_TRAINING)
    rendering = (IntendedUse.RESEARCH_TRAINING if scope is Phase2Scope.RESEARCH
                 else IntendedUse.COMMERCIAL_DEPLOYMENT)
    return (
        RequirementBundle('coobserved_motion', PRE_PHASE_2_REQUIREMENTS[0].required_capabilities,
                          EvidenceLevel.LOCAL_VERIFIED, training),
        RequirementBundle('linguistic_reference', frozenset({
            'continuous_asl', 'governed_linguistic_reference', 'manual_timing',
            'handshape_annotation', 'nonmanual_annotation', 'head_movement_annotation',
            'eye_gaze_annotation', 'qualified_asl_annotation'}),
            EvidenceLevel.QUALIFIED_HUMAN, training),
        RequirementBundle('render_rig', frozenset({'body_rig','hand_rig','face_rig','eye_rig'}),
                          EvidenceLevel.LOCAL_VERIFIED, rendering),
        RequirementBundle('review_governance', frozenset({
            'qualified_asl_review','deaf_co_design','accessible_consent'}),
            EvidenceLevel.QUALIFIED_HUMAN, rendering),
    )


def assess_phase2_scope(sources: tuple[SourceCandidate, ...], scope: Phase2Scope,
                        authorizations: dict[str, AuthorizationEvidence]) -> dict:
    """Require every bundle's capabilities AND its action-specific evidence.

    No capability stitching within a bundle. Different admissible sources can
    supply different bundles. Research approval never grants commercial use,
    redistribution, public deployment or identity use.
    """
    bundles = requirements(scope)
    baseline = assess_source_portfolio(sources, bundles)
    source_ids = {s.source_id for s in sources}
    if set(authorizations) - source_ids:
        raise ValueError('authorization references an unknown source')
    decisions = []
    for bundle, decision in zip(bundles, baseline.bundle_decisions):
        actions = ('create_derivatives', 'model_training') if bundle.requirement_id in (
            'coobserved_motion', 'linguistic_reference') else ('create_derivatives',)
        if scope is Phase2Scope.COMMERCIAL:
            actions += ('commercial_use',)
        qualified = []
        failures = {source: list(reasons) for source, reasons in decision.source_failures}
        evidence_hashes = {}
        for source in decision.satisfying_sources:
            supplied = authorizations.get(source)
            if not isinstance(supplied, AuthorizationEvidence):
                failures[source].append('action_authorization_evidence_missing')
                continue
            if supplied.source_id != source:
                failures[source].append('authorization_subject_source_mismatch')
                continue
            violations = validate_authorization(
                supplied.authorization, consent=supplied.consent,
                intended_use=scope.value, requested_actions=actions)
            if violations:
                failures[source].extend(violations)
                continue
            path = supplied.local_evidence
            if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
                violations.append('authorization_evidence_not_regular_local_file')
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != supplied.authorization.evidence_sha256.lower():
                    violations.append('authorization_evidence_hash_mismatch')
                else:
                    evidence_hashes[source] = digest
            failures[source].extend(violations)
            if not violations:
                qualified.append(source)
        decisions.append({'requirement':bundle.requirement_id, 'eligible_sources':qualified,
                          'requested_actions':list(actions), 'failures':failures,
                          'evidence_sha256':evidence_hashes})
    return {'policy_version':POLICY_VERSION, 'scope':scope.value,
            'portfolio_eligible':all(d['eligible_sources'] for d in decisions),
            'phase_exit_approved':False, 'decisions':decisions,
            'cross_source_capability_stitching_allowed':False,
            'limitations':['checks recorded claims and file integrity, not legal interpretation',
                           'real state, synchronization, rig round-trip and qualified review remain required',
                           'redistribution, public deployment and identity use require separate action evidence']}
