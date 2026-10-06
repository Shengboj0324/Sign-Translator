"""Corpus admission with fictional evidence; no real source or review approvals."""
from dataclasses import replace
import hashlib
import json

import pytest
import torch
from torch.utils.data import DataLoader

from signtranslator.data.governed_corpus import (
    GovernedItem, GovernedMotionCorpus, GovernedMotionRecord, collate_governed_motion,
)
from signtranslator.pose.multichannel import MultichannelMotion
from test_governed_motion import inputs, bind_alignment


def population(tmp_path, configurations):
    records, options = [], []
    for index, configuration in enumerate(configurations):
        directory = tmp_path / str(index)
        directory.mkdir()
        entry, _ = inputs(directory, index=index, **configuration)
        records.append(GovernedMotionRecord(**{
            key: value for key, value in entry.items() if key not in ('sources', 'authorizations')}))
        options.append(entry)
    return records, options


def admit(records, options):
    return GovernedMotionCorpus(records, sources=options[0]['sources'],
                                authorizations=options[0]['authorizations'])


def test_whole_population_admission_and_dataloader_preserve_sir(tmp_path):
    records, options = population(tmp_path, [{'split': 'train'}, {'split': 'val'}, {'split': 'test'}])
    corpus = admit(records, options)
    manifest = json.loads(corpus.manifest_bytes)
    assert manifest['phase_exit_approved'] is False
    assert [r['split'] for r in manifest['records']] == ['train', 'val', 'test']
    assert len(corpus.content_sha256) == 64
    batch = next(iter(DataLoader(corpus.split('train'), batch_size=1,
                                collate_fn=collate_governed_motion)))
    assert batch.split == 'train' and batch.corpus_sha256 == corpus.content_sha256
    assert batch.annotations[0].sir_payload == records[0].annotation.sir_payload
    assert batch.event_ids == ((0, 1, 2),)
    assert batch.event_frame_membership[0].tolist() == [[True]*3, [True]*3, [False]*3]
    assert batch.motion.timestamps.dtype == batch.annotation_times.dtype == torch.float64
    assert batch.sir_targets.event_ids[0].tolist() == list(batch.event_ids[0])
    assert batch.sir_targets.annotation_sha256 == (records[0].annotation.content_sha256(),)
    assert batch.sir_targets.event_valid[0].all()
    # Caller mutation does not silently change the admitted split snapshot.
    records[0].sample.split = 'test'
    assert corpus.split('train')[0].split == 'train'


@pytest.mark.parametrize('configurations,reason', [
    ([{'split': 'train', 'signer': 'same'}, {'split': 'test', 'signer': 'same'}], 'signer_split'),
    ([{'split': 'train', 'source_id': 'same'}, {'split': 'val', 'source_id': 'same'}], 'source_split'),
    ([{}, {'lexicon_name': 'different'}], 'mixed_sir_lexicons'),
    ([{'signer': 'a', 'source_id': 'x'}, {'signer': 'a', 'source_id': 'y'},
      {'split': 'test', 'signer': 'b', 'source_id': 'y'}], 'source_split'),
])
def test_cross_corpus_governance_rejected_before_split_views(tmp_path, configurations, reason):
    records, options = population(tmp_path, configurations)
    with pytest.raises(PermissionError, match=reason):
        admit(records, options)


def test_duplicate_sample_cannot_be_hidden_by_dictionary_conversion(tmp_path):
    records, options = population(tmp_path, [{}])
    with pytest.raises(ValueError, match='duplicate sample'):
        admit(records * 2, options)


def test_same_native_bytes_cannot_cross_splits_under_new_recording_names(tmp_path):
    records, options = population(tmp_path, [{}, {'split': 'test'}])
    first_bytes = next(iter(records[0].source_files.values())).read_bytes()
    native = next(iter(records[1].source_files.values()))
    native.write_bytes(first_bytes)
    motion = MultichannelMotion.load(records[1].motion_path, expected_sha256=records[1].motion_sha256)
    motion = replace(motion, channels={name: replace(ch, source_sha256=hashlib.sha256(first_bytes).hexdigest())
                                      for name, ch in motion.channels.items()})
    path = records[1].motion_path.with_name('duplicate-native.npz')
    digest = motion.save(path)
    options[1].update(motion_path=path, motion_sha256=digest)
    alignment = json.loads(records[1].alignment_path.read_bytes())
    alignment['motion_sha256'] = digest
    bind_alignment(options[1], alignment)
    records[1] = replace(records[1], motion_path=path, motion_sha256=digest,
                         alignment_sha256=options[1]['alignment_sha256'])
    with pytest.raises(PermissionError, match='content crosses splits'):
        admit(records, options)


@pytest.mark.parametrize('field', ['video_path', 'motion_path', 'annotation_authorization_path', 'alignment_path'])
def test_changed_file_after_admission_is_rejected_at_collation(tmp_path, field):
    records, options = population(tmp_path, [{}])
    corpus = admit(records, options)
    item = corpus.split('train')[0]
    getattr(records[0], field).write_bytes(b'changed since admission')
    with pytest.raises((ValueError, PermissionError)):
        collate_governed_motion([item])


def test_batch_rejects_mixed_splits_corpora_and_forged_split_reference(tmp_path):
    records, options = population(tmp_path, [{}, {'split': 'test'}])
    corpus = admit(records, options)
    train, test = corpus.split('train')[0], corpus.split('test')[0]
    with pytest.raises(ValueError, match='mix'):
        collate_governed_motion([train, test])
    other = admit(records, options)
    with pytest.raises(ValueError, match='mix'):
        collate_governed_motion([train, other.split('train')[0]])
    with pytest.raises(ValueError, match='split'):
        collate_governed_motion([GovernedItem(corpus, 1, 'train')])
    with pytest.raises(ValueError, match='index'):
        collate_governed_motion([GovernedItem(corpus, -1, 'train')])
    with pytest.raises(ValueError, match='no admitted'):
        corpus.split('val')
    with pytest.raises(ValueError, match='split'):
        corpus.split('validation')


def test_repeated_batch_reads_do_not_reuse_mutated_targets(tmp_path):
    records, options = population(tmp_path, [{}])
    corpus = admit(records, options)
    item = corpus.split('train')[0]
    first = collate_governed_motion([item])
    first.motion.channels['face'].values.fill_(42)
    first.event_frame_membership[0].fill_(False)
    second = collate_governed_motion([item])
    assert not second.motion.channels['face'].values.any()
    assert second.event_frame_membership[0][0].all()


def test_corpus_identity_binds_split_assignment(tmp_path):
    records, options = population(tmp_path, [{}])
    before = admit(records, options).content_sha256
    records[0].sample.split = 'test'
    after = admit(records, options).content_sha256
    assert before != after


def replace_motion(records, options, index, motion):
    path = records[index].motion_path.with_name('adapted.npz')
    digest = motion.save(path)
    alignment = json.loads(records[index].alignment_path.read_bytes())
    alignment['motion_sha256'] = digest
    bind_alignment(options[index], alignment)
    records[index] = replace(records[index], motion_path=path, motion_sha256=digest,
                             alignment_sha256=options[index]['alignment_sha256'])


def test_variable_length_graph_membership_and_clock_padding(tmp_path):
    records, options = population(tmp_path, [{}, {}])
    original = MultichannelMotion.load(records[1].motion_path, expected_sha256=records[1].motion_sha256)
    short = replace(original, timestamps=original.timestamps[:2].copy(), channels={
        name: replace(channel, **{field: getattr(channel, field)[:2].copy()
                                   for field in ('values', 'valid', 'observed', 'confidence')})
        for name, channel in original.channels.items()})
    replace_motion(records, options, 1, short)
    corpus = admit(records, options)
    batch = next(iter(DataLoader(corpus.split('train'), batch_size=2,
                                collate_fn=collate_governed_motion)))
    assert batch.motion.lengths.tolist() == [3, 2]
    assert batch.annotation_times[1].tolist() == [0., .1, 0.]
    assert batch.event_frame_membership[1].tolist() == [[True, True, False],
                                                      [True, True, False], [False]*3]
    assert not batch.motion.frame_valid[1, 2]
    assert not batch.motion.channels['body'].supervision_weights()[1, 2].any()


def test_incompatible_layout_is_rejected_across_separate_splits(tmp_path):
    records, options = population(tmp_path, [{}, {'split': 'test'}])
    motion = MultichannelMotion.load(records[1].motion_path, expected_sha256=records[1].motion_sha256)
    changed = replace(motion, channels={**motion.channels,
                                       'face': replace(motion.channels['face'], convention='different')})
    replace_motion(records, options, 1, changed)
    with pytest.raises(ValueError, match='layouts'):
        admit(records, options)


def test_absent_population_does_not_create_an_admitted_corpus():
    with pytest.raises(ValueError, match='nonempty'):
        GovernedMotionCorpus([], sources=(), authorizations={})
