"""Verified R2 archive migration. Local deletion is a separate, gated command.

Requires boto3 in an isolated environment. Credentials stay in .env.r2.local.
Archives preserve paths; existing local loaders consume explicitly restored files.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / '.r2-migration'
REMOTE = 'migration/2026-10-07-v1'
BLOCK = 1024 * 1024


def digest_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(BLOCK), b''):
        digest.update(chunk)
    return digest.hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return digest_stream(stream)


def client():
    import boto3
    from botocore.config import Config
    env = dict(line.split('=', 1) for line in (REPO / '.env.r2.local').read_text().splitlines() if line)
    return boto3.client('s3', endpoint_url=env['R2_ENDPOINT_URL'], region_name='auto',
                       aws_access_key_id=env['AWS_ACCESS_KEY_ID'],
                       aws_secret_access_key=env['AWS_SECRET_ACCESS_KEY'],
                       config=Config(retries={'mode': 'standard', 'max_attempts': 8},
                                     connect_timeout=15, read_timeout=120)), env['R2_BUCKET']


def safe_key(key):
    p = PurePosixPath(key)
    if p.is_absolute() or not p.parts or any(x in ('', '.', '..') for x in key.split('/')):
        raise ValueError('unsafe archive path')
    if p.parts[0] not in ('data', 'project-evidence'):
        raise ValueError('unknown data namespace')
    return p


def identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def check_source(row):
    path = Path(row['path'])
    if path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('source is no longer a regular non-symlink path')
    if identity(path.stat()) != tuple(row[k] for k in ('device', 'inode', 'size', 'mtime_ns')):
        raise ValueError('source changed since inventory: ' + str(path))
    safe_key(row['key'])
    return path


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
    def read(self, size):
        chunk = self.stream.read(size)
        self.digest.update(chunk)
        return chunk


def pack(rows, archive, manifest):
    members = []
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for row in rows:
            path = check_source(row)
            with path.open('rb') as source:
                if identity(os.fstat(source.fileno())) != identity(path.stat()):
                    raise ValueError('source changed at open')
                info = tarfile.TarInfo(row['key'])
                info.size, info.mode, info.mtime = row['size'], 0o600, row['mtime_ns'] / 1e9
                reader = HashReader(source)
                tar.addfile(info, reader)
                if source.read(1):
                    raise ValueError('source grew during archive creation')
            check_source(row)
            digest = reader.digest.hexdigest()
            if row.get('sha256') is not None and digest != row['sha256']:
                raise ValueError('source content changed since reconciliation: ' + str(path))
            members.append({**row, 'sha256': digest})
        payload = json.dumps(members, separators=(',', ':')).encode()
        info = tarfile.TarInfo('_r2_members.json')
        info.size, info.mode = len(payload), 0o600
        tar.addfile(info, io.BytesIO(payload))
    with gzip.open(manifest, 'wb', compresslevel=1) as output:
        output.write(payload)
    return members


def upload_verified(s3, bucket, path, key):
    digest = file_hash(path)
    s3.upload_file(str(path), bucket, key, ExtraArgs={'Metadata': {'sha256': digest}})
    response = s3.get_object(Bucket=bucket, Key=key)
    try:
        if digest_stream(response['Body']) != digest:
            raise ValueError('remote round-trip SHA-256 mismatch')
    finally:
        response['Body'].close()
    return {'key': key, 'sha256': digest, 'bytes': Path(path).stat().st_size,
            'etag': response['ETag']}


def verify_head(s3, bucket, record):
    head = s3.head_object(Bucket=bucket, Key=record['key'])
    if (head['ETag'] != record['etag'] or head['ContentLength'] != record['bytes']
            or head.get('Metadata', {}).get('sha256') != record['sha256']):
        raise ValueError('remote archive identity changed')


def records():
    path = STATE / 'catalog.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def bootstrap(catalog_sha256):
    """Recover the catalog and member indexes using an independently kept hash."""
    if len(catalog_sha256) != 64 or any(c not in '0123456789abcdef' for c in catalog_sha256):
        raise ValueError('exact lowercase catalog SHA-256 required')
    STATE.mkdir(parents=True, exist_ok=True)
    s3, bucket = client()
    with tempfile.TemporaryDirectory(dir=STATE) as temporary:
        def recover(key, destination, expected):
            target = STATE / destination
            if target.exists():
                if file_hash(target) != expected:
                    raise ValueError('refusing to overwrite conflicting local index')
                return
            staged = Path(temporary) / 'download'
            s3.download_file(bucket, key, str(staged))
            if file_hash(staged) != expected:
                raise ValueError('downloaded index hash mismatch')
            os.link(staged, target)
            staged.unlink()
        recover(f'{REMOTE}/catalog.jsonl', 'catalog.jsonl', catalog_sha256)
        for record in records():
            name = record['manifest']
            if Path(name).name != name or not name.startswith('part-') or not name.endswith('.json.gz'):
                raise ValueError('invalid manifest file name')
            recover(f'{REMOTE}/manifests/{name}', name, record['manifest_sha256'])
    print(json.dumps({'catalog_sha256': catalog_sha256, 'indexes_recovered': len(records())}), flush=True)


def batches(path):
    batch, size = [], 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if batch and (len(batch) >= 10000 or size + row['size'] > 256 * BLOCK):
                yield batch
                batch, size = [], 0
            batch.append(row)
            size += row['size']
        if batch:
            yield batch


def upload():
    inventory = STATE / 'inventory.jsonl'
    summary = json.loads((STATE / 'inventory-summary.json').read_text())
    if not summary['complete']:
        raise ValueError('inventory has unresolved errors or special files')
    inventory_hash = file_hash(inventory)
    binding = STATE / 'inventory.sha256'
    if binding.exists() and binding.read_text() != inventory_hash:
        raise ValueError('inventory changed after migration started')
    binding.write_text(inventory_hash)
    s3, bucket = client()
    completed = records()
    total = 0
    index = -1
    for index, rows in enumerate(batches(inventory)):
        total += len(rows)
        if index < len(completed):
            record = completed[index]
            if record['first'] != rows[0]['key'] or record['last'] != rows[-1]['key'] or record['files'] != len(rows):
                raise ValueError('catalog no longer matches inventory batch')
            verify_head(s3, bucket, record)
            if file_hash(STATE / record['manifest']) != record['manifest_sha256']:
                raise ValueError('local member manifest changed')
            if (index + 1) % 50 == 0:
                print(json.dumps({'state': 'rechecking_verified_archives',
                                  'checked': index + 1, 'total': len(completed)}), flush=True)
            continue
        archive = STATE / f'part-{index:06d}.tar.gz'
        manifest = STATE / f'part-{index:06d}.json.gz'
        print(json.dumps({'state': 'packing', 'part': index, 'files': len(rows)}), flush=True)
        pack(rows, archive, manifest)
        print(json.dumps({'state': 'uploading_and_verifying', 'part': index,
                          'archive_bytes': archive.stat().st_size}), flush=True)
        record = upload_verified(s3, bucket, archive, f'{REMOTE}/archives/{archive.name}')
        record.update(files=len(rows), first=rows[0]['key'], last=rows[-1]['key'],
                      source_bytes=sum(row['size'] for row in rows),
                      manifest=manifest.name, manifest_sha256=file_hash(manifest))
        upload_verified(s3, bucket, manifest, f'{REMOTE}/manifests/{manifest.name}')
        with (STATE / 'catalog.jsonl').open('a') as journal:
            journal.write(json.dumps(record) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        archive.unlink()  # Staging only; source data is never removed by upload.
        print(json.dumps({'verified_part': index, 'verified_files': total,
                          'source_bytes': record['source_bytes'], 'archive_bytes': record['bytes']}), flush=True)
    if len(records()) != index + 1:
        raise ValueError('catalog contains unexpected trailing archives')
    if total != sum(group['files'] for group in summary['groups'].values()):
        raise ValueError('inventory count does not match its completed summary')
    catalog = upload_verified(s3, bucket, STATE / 'catalog.jsonl', f'{REMOTE}/catalog.jsonl')
    summary_record = upload_verified(s3, bucket, STATE / 'inventory-summary.json', f'{REMOTE}/inventory-summary.json')
    complete = dict(complete=True, files=total, inventory_sha256=inventory_hash,
                    catalog=catalog, summary=summary_record, bucket=bucket, prefix=REMOTE)
    (STATE / 'complete.json').write_text(json.dumps(complete, indent=2))
    upload_verified(s3, bucket, STATE / 'complete.json', f'{REMOTE}/complete.json')
    print(json.dumps(complete), flush=True)


def restore(prefix, destination):
    safe_key(prefix.rstrip('/'))
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    s3, bucket = client()
    restored = 0
    for record in records():
        manifest = STATE / record['manifest']
        if file_hash(manifest) != record['manifest_sha256']:
            raise ValueError('member manifest changed')
        with gzip.open(manifest, 'rt') as stream:
            members = json.load(stream)
        selected = {row['key']: row for row in members
                    if row['key'] == prefix.rstrip('/') or row['key'].startswith(prefix.rstrip('/') + '/')}
        if not selected:
            continue
        verify_head(s3, bucket, record)
        with tempfile.TemporaryDirectory(dir=STATE) as temporary:
            archive = Path(temporary) / 'restore.tar.gz'
            s3.download_file(bucket, record['key'], str(archive))
            if file_hash(archive) != record['sha256']:
                raise ValueError('downloaded archive hash mismatch')
            found = set()
            with tarfile.open(archive, 'r|gz') as tar:
                for member in tar:
                    if member.name not in selected:
                        continue
                    if member.name in found or not member.isfile():
                        raise ValueError('duplicate or nonregular archive member')
                    found.add(member.name)
                    row = selected[member.name]
                    if member.size != row['size']:
                        raise ValueError('archive member size mismatch')
                    target = destination.joinpath(*safe_key(member.name).parts[1:])
                    if target.resolve() != target:
                        raise ValueError('restore target contains a symlink')
                    if target.exists():
                        if file_hash(target) != row['sha256']:
                            raise ValueError('refusing to overwrite differing local data')
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_name(target.name + '.r2-partial')
                    created = False
                    try:
                        output = partial.open('xb')
                        created = True
                        os.chmod(partial, 0o600)
                        with output, tar.extractfile(member) as source:
                            for block in iter(lambda: source.read(BLOCK), b''):
                                output.write(block)
                        if file_hash(partial) != row['sha256']:
                            raise ValueError('restored member hash mismatch')
                        os.link(partial, target)  # Atomic no-clobber publication.
                        os.utime(target, ns=(row['mtime_ns'], row['mtime_ns']))
                    finally:
                        if created:
                            partial.unlink(missing_ok=True)
                    restored += 1
            if found != set(selected):
                raise ValueError('archive omitted selected files')
    print(json.dumps({'restored_files': restored, 'destination': str(destination), 'prefix': prefix}), flush=True)


def cleanup(root):
    root = Path(root).resolve()
    if root != Path('/Users/jiangshengbo/Volumes'):
        raise ValueError('cleanup is restricted to the inventoried project data root')
    complete = json.loads((STATE / 'complete.json').read_text())
    if not complete['complete'] or file_hash(STATE / 'catalog.jsonl') != complete['catalog']['sha256']:
        raise ValueError('complete verified migration required')
    if not (STATE / 'restore-tested.json').exists():
        raise ValueError('successful integration restore test required before cleanup')
    restore_test = json.loads((STATE / 'restore-tested.json').read_text())
    if restore_test.get('catalog_sha256') != complete['catalog']['sha256'] or restore_test.get('passed') is not True:
        raise ValueError('restore test must bind the completed catalog')
    s3, bucket = client()
    if bucket != complete['bucket']:
        raise ValueError('cleanup bucket differs from completed migration')
    verify_head(s3, bucket, complete['catalog'])
    deleted = 0
    for record in records():
        verify_head(s3, bucket, record)
        if file_hash(STATE / record['manifest']) != record['manifest_sha256']:
            raise ValueError('member manifest changed')
        with gzip.open(STATE / record['manifest'], 'rt') as stream:
            members = json.load(stream)
        for row in members:
            path = Path(row['path'])
            if not row['key'].startswith('data/') or not path.is_relative_to(root):
                continue  # Keep project evidence in the checkout.
            if not path.exists():
                continue  # Allows cleanup to resume after interruption.
            check_source(row)
            if file_hash(path) != row['sha256']:
                raise ValueError('local data changed; refusing deletion: ' + str(path))
            check_source(row)
            path.unlink()
            deleted += 1
        print(json.dumps({'deleted_files': deleted, 'verified_archive': record['key']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('upload')
    bootstrap_parser = commands.add_parser('bootstrap')
    bootstrap_parser.add_argument('--catalog-sha256', required=True)
    restore_parser = commands.add_parser('restore')
    restore_parser.add_argument('--prefix', required=True)
    restore_parser.add_argument('--destination', required=True)
    cleanup_parser = commands.add_parser('cleanup')
    cleanup_parser.add_argument('--root', required=True)
    args = parser.parse_args()
    if args.command == 'upload':
        upload()
    elif args.command == 'bootstrap':
        bootstrap(args.catalog_sha256)
    elif args.command == 'restore':
        restore(args.prefix, args.destination)
    else:
        cleanup(args.root)
