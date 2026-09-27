"""Create a read-only-source QC ledger; no row is approved for training."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signtranslator.data_engineering.how2sign import read_how2sign_metadata
from signtranslator.data_engineering.how2sign_audit import How2SignAuditConfig, _compute_row


DISPOSITIONS = {
    'missing_source': 'technical_quarantine',
    'structural_failure': 'technical_quarantine',
    'unjoinable_artifact': 'technical_quarantine',
    'quality_warning': 'pending_qualified_qc',
    'valid': 'pending_acceptance',
}


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest(), 'size': path.stat().st_size}


def reconcile(rows, mappings):
    """Require an exact one-to-one join, apart from explicitly unjoinable artifacts."""
    by_id = {}
    for mapping in mappings:
        key = mapping['sample_id']
        if not key or key in by_id:
            raise ValueError('duplicate or empty mapping sample ID')
        by_id[key] = mapping
    seen = set()
    output = []
    for row in rows:
        key, status = row['sample_id'], row['status']
        if not key or key in seen or status not in DISPOSITIONS:
            raise ValueError('duplicate/empty audit ID or unknown audit status')
        seen.add(key)
        mapping = by_id.get(key)
        if status == 'unjoinable_artifact':
            if mapping is not None:
                raise ValueError('unjoinable artifact unexpectedly has a metadata mapping')
        elif (mapping is None or mapping['video_id'] != row['video_id']
              or mapping['audit_status'] != status
              or mapping['signer_id'] != row['filename_code']):
            raise ValueError(f'audit/mapping mismatch for {key}')
        output.append({
            'sample_id': key, 'source_id': row['video_id'] or '',
            'signer_id': mapping['signer_id'] if mapping else '',
            'historical_status': status, 'disposition': DISPOSITIONS[status],
            'historical_error': row['error'] or '',
            **{k: row[k] or '' for k in ('raw_sha256', 'rendered_sha256', 'openpose_sha256')},
            'qualified_acceptance': False, 'training_eligible': False,
        })
    if set(by_id) - seen:
        raise ValueError('mapping contains samples absent from the audit')
    return sorted(output, key=lambda row: row['sample_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--signer-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    if args.output.resolve().is_relative_to(source):
        raise ValueError('ledger output must be outside the preserved source tree')
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.audit_root / 'audit_manifest.json'
    db_path = args.audit_root / 'audit.sqlite3'
    cert_path = args.signer_root / 'certificate.json'
    mapping_path = args.signer_root / 'how2sign_train_signers.csv'
    metadata_path = source / 'how2sign_realigned_train.csv'
    inputs = [manifest_path, db_path, cert_path, mapping_path, metadata_path]
    before = [identity(path) for path in inputs]
    manifest = json.loads(manifest_path.read_text())
    certificate = json.loads(cert_path.read_text())
    if (manifest.get('audit_complete') is not True or certificate.get('certified') is not True
            or before[1]['sha256'] != manifest['audit_database']['sha256']
            or before[1]['sha256'] != certificate['source']['audit_database_sha256']
            or before[0]['sha256'] != certificate['source']['audit_manifest_sha256']
            or before[3]['sha256'] != certificate['mapping']['sha256']
            or before[4]['sha256'] != certificate['source']['metadata_sha256']
            or before[4]['sha256'] != manifest['identity']['metadata_sha256']):
        raise ValueError('historical certificate/manifest/input hashes disagree')
    with mapping_path.open(newline='') as stream:
        mappings = list(csv.DictReader(stream))
    with sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA query_only=ON')
        rows = [dict(row) for row in connection.execute(
            'SELECT sample_id,video_id,filename_code,status,error,raw_sha256,'
            'rendered_sha256,openpose_sha256 FROM clips ORDER BY sample_id')]
    if len(mappings) != certificate['mapping']['rows']:
        raise ValueError('mapping row count differs from certificate')
    ledger = reconcile(rows, mappings)
    status_counts = dict(Counter(row['status'] for row in rows))
    if status_counts != manifest['status_counts']:
        raise ValueError('database counts differ from manifest')
    metadata = {row.sentence_name: row for row in read_how2sign_metadata(metadata_path)}
    if set(metadata) != {row['sample_id'] for row in mappings}:
        raise ValueError('metadata membership differs from mapping')
    refresh = []
    for row in rows:
        if row['status'] not in ('missing_source', 'structural_failure'):
            continue
        sample = metadata[row['sample_id']]
        result = _compute_row(str(source), sample, How2SignAuditConfig())
        refresh.append({'sample_id': row['sample_id'], 'previous_status': row['status'],
                        **{key: result.get(key) for key in (
                            'status', 'error', 'raw_sha256', 'rendered_sha256', 'openpose_sha256')},
                        'status_unchanged': result['status'] == row['status'],
                        'qualified_acceptance': False})
    if before != [identity(path) for path in inputs]:
        raise ValueError('source evidence changed while building ledger')
    summary = {'schema_version': 1, 'input_files': before, 'historical_status_counts': status_counts,
               'disposition_counts': dict(Counter(row['disposition'] for row in ledger)),
               'ledger_rows': len(ledger), 'metadata_rows': len(mappings),
               'refreshed_rows': len(refresh),
               'refreshed_status_changes': sum(not row['status_unchanged'] for row in refresh),
               'source_files_modified': False, 'physical_quarantine_performed': False,
               'qualified_acceptance': False, 'training_eligible_rows': 0,
               'w1_complete': False, 'w2_complete': False,
               'limitations': ['Warnings and historically valid rows were not re-decoded or human-approved.',
                               'Unjoinable artifact status is historical, not a new semantic classification.',
                               'Logical quarantine only; no source artifact was moved or deleted.',
                               'No final split, QC threshold, rights approval or 3D eligibility is inferred.']}
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'dispositions.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    for filename, value in [('technical-refresh.json', refresh), ('summary.json', summary)]:
        with (args.output / filename).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({key: summary[key] for key in (
        'ledger_rows', 'disposition_counts', 'refreshed_rows', 'refreshed_status_changes')}))


if __name__ == '__main__':
    main()
