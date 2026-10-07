"""Finish an existing upload, perform real restore checks, then gated cleanup."""
import argparse
import gzip
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
import r2_data as r2


def finish(wait_pid):
    if wait_pid:
        while True:
            process = subprocess.run(['ps', '-p', str(wait_pid), '-o', 'command='],
                                     capture_output=True, text=True, check=False)
            if process.returncode != 0 or 'scripts/r2_data.py upload' not in process.stdout:
                break
            print(json.dumps({'state': 'waiting_for_upload', 'pid': wait_pid}), flush=True)
            time.sleep(30)
    complete = json.loads((r2.STATE / 'complete.json').read_text())
    if not complete['complete'] or r2.file_hash(r2.STATE / 'catalog.jsonl') != complete['catalog']['sha256']:
        raise ValueError('complete, unchanged catalog required')
    s3, bucket = r2.client()
    if bucket != complete['bucket']:
        raise ValueError('bucket changed')
    r2.verify_head(s3, bucket, complete['catalog'])
    selections = {}
    file_count = 0
    for record in r2.records():
        r2.verify_head(s3, bucket, record)
        manifest = r2.STATE / record['manifest']
        if r2.file_hash(manifest) != record['manifest_sha256']:
            raise ValueError('member index changed')
        with gzip.open(manifest, 'rt') as stream:
            members = json.load(stream)
        if len(members) != record['files']:
            raise ValueError('member count differs from catalog')
        file_count += len(members)
        for row in members:
            parts = r2.safe_key(row['key']).parts
            group = '/'.join(parts[:2])
            # At least one sample for every top-level data/evidence group.
            if group not in selections or row['size'] < selections[group]['size']:
                selections[group] = row
    if file_count != complete['files']:
        raise ValueError('catalog file count mismatch')
    checked = []
    with tempfile.TemporaryDirectory(prefix='restore-validation-', dir=r2.STATE) as temporary:
        for group, row in selections.items():
            destination = Path(temporary) / r2.safe_key(row['key']).parts[0]
            r2.restore(row['key'], destination)
            restored = destination.joinpath(*r2.safe_key(row['key']).parts[1:])
            if r2.file_hash(restored) != row['sha256']:
                raise ValueError('restore integration hash mismatch')
            checked.append({'key': row['key'], 'sha256': row['sha256'], 'bytes': row['size']})
    test = dict(passed=True, catalog_sha256=complete['catalog']['sha256'], samples=checked,
                scope='Actual R2 download, archive verification, path restoration and file SHA-256 for every inventory group')
    (r2.STATE / 'restore-tested.json').write_text(json.dumps(test, indent=2))
    r2.upload_verified(s3, bucket, r2.STATE / 'restore-tested.json', f'{r2.REMOTE}/restore-tested.json')
    # Preserve the independently usable recovery scripts and instructions remotely.
    for relative in ['scripts/r2_data.py', 'scripts/r2_finish.py', 'docs/R2_STORAGE.md']:
        r2.upload_verified(s3, bucket, r2.REPO / relative, f'{r2.REMOTE}/recovery/{relative}')
    print(json.dumps({'state': 'restore_tests_passed', 'groups': len(checked)}), flush=True)
    r2.cleanup('/Users/jiangshengbo/Volumes')
    remaining = 0
    for directory, folders, files in os.walk('/Users/jiangshengbo/Volumes'):
        remaining += sum(name != '.DS_Store' for name in files)
        remaining += sum((Path(directory) / name).is_symlink() for name in folders)
    report = dict(state='inventoried_data_cleanup_completed' if remaining == 0 else 'cleanup_finished_with_remaining_files',
                  remaining_non_finder_files=remaining, catalog_sha256=complete['catalog']['sha256'],
                  source_files_verified=file_count, retained='Project evidence, indexes, credentials and code retained; .DS_Store excluded')
    (r2.STATE / 'finished.json').write_text(json.dumps(report, indent=2))
    r2.upload_verified(s3, bucket, r2.STATE / 'finished.json', f'{r2.REMOTE}/finished.json')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    finish(args.wait_pid)
