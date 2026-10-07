"""Migration packing and restore safety, without real account credentials."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import pytest

spec = importlib.util.spec_from_file_location('r2_data', Path(__file__).parents[1] / 'scripts/r2_data.py')
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)


def row(path, key):
    s = path.stat()
    return dict(path=str(path), key=key, size=s.st_size, device=s.st_dev,
                inode=s.st_ino, mtime_ns=s.st_mtime_ns)


def test_pack_roundtrip_preserves_paths_bytes_and_hashes(tmp_path):
    paths = [tmp_path / 'one', tmp_path / 'two']
    for path, payload in zip(paths, [b'', b'abcd' * 100000]):
        path.write_bytes(payload)
    rows = [row(path, 'data/folder/' + path.name) for path in paths]
    archive, manifest = tmp_path / 'data.tar.gz', tmp_path / 'members.json.gz'
    members = r2.pack(rows, archive, manifest)
    with gzip.open(manifest, 'rt') as stream:
        assert json.load(stream) == members
    with tarfile.open(archive) as tar:
        for path, member in zip(paths, members):
            assert tar.extractfile(member['key']).read() == path.read_bytes()
            assert member['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert json.load(tar.extractfile('_r2_members.json')) == members


def test_pack_refuses_inventory_mutation(tmp_path):
    path = tmp_path / 'source'
    path.write_bytes(b'old')
    record = row(path, 'data/source')
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        r2.pack([record], tmp_path / 'archive', tmp_path / 'manifest')


@pytest.mark.parametrize('key', ['/data/x', 'data/../x', 'data//x', 'data/./x', 'secret/x'])
def test_paths_fail_closed(key):
    with pytest.raises(ValueError):
        r2.safe_key(key)


def test_cleanup_refuses_arbitrary_root(tmp_path):
    with pytest.raises(ValueError, match='restricted'):
        r2.cleanup(tmp_path)


def test_restore_verifies_and_refuses_overwrite(tmp_path, monkeypatch):
    source = tmp_path / 'original'
    source.write_bytes(b'actual test payload')
    state = tmp_path / 'state'
    state.mkdir()
    archive, manifest = state / 'archive', state / 'part-000000.json.gz'
    r2.pack([row(source, 'data/nested/source')], archive, manifest)
    record = dict(key='archive-key', sha256=r2.file_hash(archive), bytes=archive.stat().st_size,
                  etag='fixture', manifest=manifest.name, manifest_sha256=r2.file_hash(manifest))
    (state / 'catalog.jsonl').write_text(json.dumps(record) + '\n')
    class S3:
        def head_object(self, **kwargs):
            return dict(ETag='fixture', ContentLength=record['bytes'], Metadata={'sha256':record['sha256']})
        def download_file(self, bucket, key, destination):
            Path(destination).write_bytes(archive.read_bytes())
    monkeypatch.setattr(r2, 'STATE', state)
    monkeypatch.setattr(r2, 'client', lambda: (S3(), 'fixture-bucket'))
    destination = tmp_path / 'restored'
    r2.restore('data/nested', destination)
    target = destination / 'nested/source'
    assert target.read_bytes() == source.read_bytes()
    r2.restore('data/nested', destination)  # Matching files are reused.
    target.write_bytes(b'different')
    with pytest.raises(ValueError, match='overwrite'):
        r2.restore('data/nested', destination)
    assert target.read_bytes() == b'different'


def test_restore_rejects_corrupt_download(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.write_bytes(b'payload')
    manifest = tmp_path / 'part-000000.json.gz'
    archive = tmp_path / 'archive'
    r2.pack([row(source, 'data/source')], archive, manifest)
    record = dict(key='key', sha256=r2.file_hash(archive), bytes=archive.stat().st_size,
                  etag='etag', manifest=manifest.name, manifest_sha256=r2.file_hash(manifest))
    (tmp_path / 'catalog.jsonl').write_text(json.dumps(record)+'\n')
    class S3:
        def head_object(self, **kwargs):
            return dict(ETag='etag', ContentLength=record['bytes'], Metadata={'sha256':record['sha256']})
        def download_file(self, bucket, key, destination):
            Path(destination).write_bytes(b'corrupted')
    monkeypatch.setattr(r2,'STATE',tmp_path)
    monkeypatch.setattr(r2,'client',lambda:(S3(),'fixture'))
    with pytest.raises(ValueError,match='archive hash'):
        r2.restore('data/source',tmp_path/'out')
    assert not (tmp_path/'out/source').exists()
