"""Audit/metadata joins must refuse ambiguity before producing dispositions."""
import pytest

from scripts.w1_qc_dispositions import reconcile


def row(status='valid', sample_id='sample'):
    return dict(sample_id=sample_id, video_id='video', filename_code='5', status=status,
                error=None, raw_sha256='a' * 64, rendered_sha256='b' * 64,
                openpose_sha256='c' * 64)


def mapping(status='valid', sample_id='sample'):
    return dict(sample_id=sample_id, video_id='video', signer_id='5', audit_status=status)


@pytest.mark.parametrize('status,disposition', [
    ('valid', 'pending_acceptance'), ('quality_warning', 'pending_qualified_qc'),
    ('missing_source', 'technical_quarantine'), ('structural_failure', 'technical_quarantine')])
def test_historical_status_cannot_approve_training(status, disposition):
    result = reconcile([row(status)], [mapping(status)])[0]
    assert result['disposition'] == disposition
    assert result['qualified_acceptance'] is False and result['training_eligible'] is False


def test_unjoinable_artifact_remains_outside_metadata_population():
    result = reconcile([row(), row('unjoinable_artifact', 'artifact')], [mapping()])
    assert result[0]['sample_id'] == 'artifact'
    assert result[0]['signer_id'] == ''
    assert result[0]['disposition'] == 'technical_quarantine'


@pytest.mark.parametrize('defect', ['duplicate_audit', 'duplicate_mapping', 'source',
                                    'signer', 'status', 'missing_audit', 'missing_mapping',
                                    'unknown_status', 'mapped_orphan'])
def test_join_defects_refuse_dispositions(defect):
    rows, mappings = [row()], [mapping()]
    if defect == 'duplicate_audit': rows.append(row())
    elif defect == 'duplicate_mapping': mappings.append(mapping())
    elif defect == 'source': mappings[0]['video_id'] = 'different'
    elif defect == 'signer': mappings[0]['signer_id'] = '8'
    elif defect == 'status': mappings[0]['audit_status'] = 'quality_warning'
    elif defect == 'missing_audit': rows = []
    elif defect == 'missing_mapping': mappings = []
    elif defect == 'unknown_status': rows[0]['status'] = 'approved'
    elif defect == 'mapped_orphan': rows[0]['status'] = 'unjoinable_artifact'
    with pytest.raises(ValueError):
        reconcile(rows, mappings)
