"""Source-language transport tests with explicitly fictional reviewer evidence."""
from dataclasses import replace
import hashlib

import pytest
import torch

from signtranslator.data.governed_motion import MAX_TRANSCRIPT_BYTES, load_governed_motion_pair
from signtranslator.data.governed_text import PLAINTEXT_ENCODING, encode_plaintext_transcripts
from test_governed_motion import inputs
from test_governed_corpus import population, admit
from signtranslator.data.governed_corpus import collate_governed_motion, move_governed_batch
from test_phase3_governance import _sample, _annotation


def annotated(payload, index=0):
    original = _annotation(index, _sample(index))
    digest = hashlib.sha256(payload).hexdigest()
    return replace(original,
                   source=replace(original.source, transcript_sha256=digest),
                   review=replace(original.review, reviewed_transcript_sha256=digest))


def encode(payloads, annotations=None, **kwargs):
    if annotations is None:
        annotations = [annotated(payload, i) for i, payload in enumerate(payloads)]
    return encode_plaintext_transcripts(payloads, annotations,
                                       declared_encoding=PLAINTEXT_ENCODING,
                                       max_bytes=1024, **kwargs)


def test_lossless_unicode_and_padding_without_target_vocabulary():
    payloads = ['Écho\n你\tA'.encode(), 'e\u0301'.encode(), b'X']
    batch = encode(payloads)
    assert batch.lengths.tolist() == [len(p) for p in payloads]
    assert batch.token_ids.dtype == batch.lengths.dtype == torch.int64
    assert batch.valid.dtype == torch.bool
    for row, payload in enumerate(payloads):
        values = batch.token_ids[row, batch.valid[row]].tolist()
        assert bytes(value - 1 for value in values) == payload
        assert not batch.token_ids[row, ~batch.valid[row]].any()
    assert batch.transcript_sha256 == tuple(hashlib.sha256(p).hexdigest() for p in payloads)


@pytest.mark.parametrize('payload', [b'', b' \r\n\t', b'\xff', b'\xc0\xaf', b'a' * 1025])
def test_invalid_empty_or_oversized_text_fails_without_truncation(payload):
    with pytest.raises((ValueError, UnicodeDecodeError)):
        encode([payload])


def test_annotation_reordering_or_replacement_cannot_relabel_source():
    payloads = [b'left', b'right']
    annotations = [annotated(p, i) for i, p in enumerate(payloads)]
    with pytest.raises(ValueError, match='transcript identity'):
        encode(payloads, list(reversed(annotations)))
    with pytest.raises(ValueError, match='transcript identity'):
        encode([b'changed'], annotations[:1])
    with pytest.raises(ValueError, match='equally sized'):
        encode(payloads, annotations[:1])


@pytest.mark.parametrize('encoding', [None, 'utf-8', 'gloss', 'json', True])
def test_format_is_not_guessed(encoding):
    with pytest.raises(ValueError, match='declaration'):
        encode_plaintext_transcripts([b'text'], [annotated(b'text')],
                                     declared_encoding=encoding, max_bytes=10)


@pytest.mark.parametrize('limit', [0, -1, True, 3.5, 65537])
def test_model_capacity_limit_is_explicit(limit):
    with pytest.raises(ValueError, match='max_bytes'):
        encode_plaintext_transcripts([b'text'], [annotated(b'text')],
                                     declared_encoding=PLAINTEXT_ENCODING, max_bytes=limit)


def test_loaded_payload_is_exact_verified_snapshot(tmp_path):
    options, _ = inputs(tmp_path)
    pair = load_governed_motion_pair(**options)
    original = options['transcript_path'].read_bytes()
    assert pair.transcript_payload == original
    options['transcript_path'].write_bytes(b'changed after read')
    assert pair.transcript_payload == original
    with pytest.raises(PermissionError):
        load_governed_motion_pair(**options)


def test_transcript_size_and_symlink_fail_before_admission(tmp_path):
    options, _ = inputs(tmp_path)
    path = options['transcript_path']
    path.write_bytes(b'x' * (MAX_TRANSCRIPT_BYTES + 1))
    with pytest.raises(ValueError, match='size limit'):
        load_governed_motion_pair(**options)
    path.unlink()
    target = tmp_path / 'target'
    target.write_bytes(b'transcript-0')
    path.symlink_to(target)
    with pytest.raises(ValueError, match='regular local file'):
        load_governed_motion_pair(**options)


def test_corpus_and_device_transport_keep_bound_source_bytes(tmp_path):
    records, options = population(tmp_path, [{}, {}])
    corpus = admit(records, options)
    dataset = corpus.split('train')
    batch = collate_governed_motion([dataset[0], dataset[1]])
    assert batch.transcript_payloads == (b'transcript-0', b'transcript-1')
    encoded = encode(batch.transcript_payloads, batch.annotations)
    assert encoded.annotation_sha256 == batch.sir_targets.annotation_sha256
    assert move_governed_batch(batch, 'cpu').transcript_payloads is batch.transcript_payloads
    records[1].transcript_path.write_bytes(b'changed')
    with pytest.raises(PermissionError):
        collate_governed_motion([dataset[0], dataset[1]])
