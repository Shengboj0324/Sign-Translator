"""Incremental canonical JSON is byte-identical and atomic sidecar failures are contained."""
from itertools import chain
import json
import random

import pytest

from signtranslator.reproducibility import canonical_json_bytes, canonical_json_equal, iter_canonical_json_chunks
from signtranslator.training.trainer import _atomic_write_chunks
from test_epoch_commit import make
from test_supported_trainer import mixed


def values():
    rng = random.Random(193)
    result = [None, True, False, -0., 5e-324, 1e308, 2**2000,
              'é样本🎵\n\t\\\"', [], {}, ('tuple', [1, 2.])]
    for _ in range(30):
        result.append({'z': [rng.randrange(-100000, 100000) for _ in range(12)],
                       'a': {'unicode': '🦉' * rng.randrange(20), 'float': rng.random()}})
    return result


@pytest.mark.parametrize('size', [1, 3, 17, 65536])
def test_chunks_exactly_match_existing_encoder(size):
    for value in values():
        chunks = list(iter_canonical_json_chunks(value, size))
        assert b''.join(chunks) == canonical_json_bytes(value)
        assert all(len(chunk) == size for chunk in chunks[:-1])
        assert 0 < len(chunks[-1]) <= size


@pytest.mark.parametrize('size', [0, -1, True, 1.5])
def test_invalid_chunk_size_refused(size):
    with pytest.raises(ValueError):
        list(iter_canonical_json_chunks({}, size))


def test_exact_comparison_preserves_types_signed_zero_and_canonical_equivalence():
    assert canonical_json_equal({'b': (1, 2), 'a': '文'}, {'a': '文', 'b': [1, 2]})
    for left, right in [(1, True), (1, 1.), (-0., 0.), ([1], [1, 2]), ('é', 'e')]:
        assert not canonical_json_equal(left, right)
    assert canonical_json_equal(values(), json.loads(canonical_json_bytes(values())))


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), float('-inf'), {1, 2}])
def test_invalid_tail_is_not_skipped_after_early_mismatch(invalid):
    with pytest.raises((ValueError, TypeError)):
        canonical_json_equal(['a' * 100000, invalid], ['b' * 100000, 0])
    with pytest.raises((ValueError, TypeError)):
        canonical_json_equal(['a' * 100000, 0], ['b' * 100000, invalid])


def test_atomic_writer_preserves_previous_file_and_removes_failed_temp(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_bytes(b'previous')
    def fail():
        yield b'partial'
        raise RuntimeError('encode failed')
    with pytest.raises(RuntimeError, match='encode failed'):
        _atomic_write_chunks(path, fail())
    assert path.read_bytes() == b'previous'
    assert list(tmp_path.iterdir()) == [path]
    value = {'unicode': '样本', 'items': [-0., True, None]}
    _atomic_write_chunks(path, chain(iter_canonical_json_chunks(value, 2), (b'\n',)))
    assert path.read_bytes() == canonical_json_bytes(value) + b'\n'


def test_saved_manifest_retains_canonical_bytes_and_resume_identity(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary); trainer.fit(max_epochs=1)
    report = trainer.exposure_report().payload
    path = trainer.save(tmp_path / 'streamed.pt')
    raw = path.with_suffix('.pt.json').read_bytes()
    assert raw == canonical_json_bytes(json.loads(raw)) + b'\n'
    receiver = make(corpus, vocabulary); receiver.load(path)
    assert receiver.exposure_report().payload == report
