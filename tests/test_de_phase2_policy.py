"""Action scope separation; fixtures confer no real source authorization."""
from dataclasses import replace
import hashlib
import pytest
from signtranslator.data_engineering.phase2_policy import (
    requirements,Phase2Scope,AuthorizationEvidence,assess_phase2_scope,
)
from signtranslator.data_engineering.source_portfolio import (
    SourceCandidate,AccessStatus,RightsStatus,EvidenceLevel,evidence,
    CURRENT_SOURCE_CANDIDATES,CURRENT_PRE_PHASE_2_DECISION,
)
from signtranslator.data_engineering.schema import (
    DataAuthorization,AuthorizationBasis,ConsentState,PersonalityRightsStatus,
)


def fixture(tmp_path):
    path=tmp_path/'permission.txt';path.write_text('SYNTHETIC TEST EVIDENCE ONLY')
    capabilities=set().union(*(r.required_capabilities for r in requirements(Phase2Scope.RESEARCH)))
    source=SourceCandidate('fixture','test','https://example.test/source',AccessStatus.LOCAL_VERIFIED,
        evidence(*capabilities,level=EvidenceLevel.QUALIFIED_HUMAN),RightsStatus.PERMITTED,
        RightsStatus.PERMITTED,RightsStatus.PERMITTED,'fictional contract')
    auth=DataAuthorization(AuthorizationBasis.DIRECT_PARTICIPANT_CONSENT,'test-license',
         'https://example.test/license','test licensor',path.as_uri(),
         hashlib.sha256(path.read_bytes()).hexdigest(),('research',),
         ('create_derivatives','model_training'),PersonalityRightsStatus.VERIFIED)
    return source,AuthorizationEvidence(auth,ConsentState.GRANTED,path,source.source_id)


def test_research_eligibility_cannot_authorize_commercial_scope(tmp_path):
    source,auth=fixture(tmp_path)
    research=assess_phase2_scope((source,),Phase2Scope.RESEARCH,{'fixture':auth})
    assert research['portfolio_eligible'] and not research['phase_exit_approved']
    commercial=assess_phase2_scope((source,),Phase2Scope.COMMERCIAL,{'fixture':auth})
    assert not commercial['portfolio_eligible']
    both=replace(auth,authorization=replace(auth.authorization,permitted_uses=('research','commercial'),
        permitted_actions=('create_derivatives','model_training','commercial_use')))
    assert assess_phase2_scope((source,),Phase2Scope.COMMERCIAL,{'fixture':both})['portfolio_eligible']


def test_file_hash_tampering_and_missing_authorizations_fail(tmp_path):
    source,auth=fixture(tmp_path)
    assert not assess_phase2_scope((source,),Phase2Scope.RESEARCH,{})['portfolio_eligible']
    auth.local_evidence.write_text('changed bytes')
    result=assess_phase2_scope((source,),Phase2Scope.RESEARCH,{'fixture':auth})
    assert not result['portfolio_eligible']
    assert 'authorization_evidence_hash_mismatch' in result['decisions'][0]['failures']['fixture']


def test_aggregate_rights_flags_cannot_replace_individual_actions(tmp_path):
    source,auth=fixture(tmp_path)
    limited=replace(auth,authorization=replace(auth.authorization,permitted_actions=('create_derivatives',)))
    result=assess_phase2_scope((source,),Phase2Scope.RESEARCH,{'fixture':limited})
    assert not result['portfolio_eligible']
    assert 'action_not_permitted:model_training' in result['decisions'][0]['failures']['fixture']


def test_unknown_scope_and_unknown_source_evidence_rejected(tmp_path):
    source,auth=fixture(tmp_path)
    with pytest.raises(ValueError):assess_phase2_scope((source,),'research',{'fixture':auth})
    with pytest.raises(ValueError):assess_phase2_scope((source,),Phase2Scope.RESEARCH,{'other':auth})


def test_current_real_portfolio_and_legacy_gate_stay_closed():
    assert not CURRENT_PRE_PHASE_2_DECISION.approved
    for scope in Phase2Scope:
        result=assess_phase2_scope(CURRENT_SOURCE_CANDIDATES,scope,{})
        assert not result['portfolio_eligible'] and not result['phase_exit_approved']


def test_authorization_cannot_be_attached_to_a_different_source(tmp_path):
    source,auth=fixture(tmp_path)
    result=assess_phase2_scope((source,),Phase2Scope.RESEARCH,
                              {'fixture':replace(auth,source_id='different')})
    assert not result['portfolio_eligible']
    assert 'authorization_subject_source_mismatch' in result['decisions'][0]['failures']['fixture']
