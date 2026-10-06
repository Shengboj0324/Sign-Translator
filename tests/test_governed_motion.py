"""Fictional source/review evidence tests, never empirical ASL acceptance."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from signtranslator.data.governed_motion import load_governed_motion_pair
from signtranslator.grammar.sir import EventKind, EdgeType, SIREvent, SIREdge, SIRGraph, sir_sha256
from signtranslator.planning.supervision import GovernedSIRAnnotation
from test_de_phase2_policy import fixture as policy_fixture
from test_phase3_governance import _sample, _annotation
from test_pose_multichannel import fixture as motion_fixture


def inputs(tmp_path, *, index=0, split='train', signer=None, source_id=None,
           lexicon_name='sir-lexicon-v1'):
    source, authorization = policy_fixture(tmp_path)
    authorization = replace(authorization, authorization=replace(
        authorization.authorization,
        permitted_actions=('download', 'create_derivatives', 'model_training')))
    sample = replace(_sample(index, split=split, signer=signer), authorization=authorization.authorization,
                     license=authorization.authorization.license_identifier)
    if source_id is not None:
        sample = replace(sample, source_id=source_id)
    original = _annotation(index, sample, lexicon_name=lexicon_name)
    graph = SIRGraph([
        SIREvent(0, EventKind.MANUAL, 10, 0., 1.),
        SIREvent(1, EventKind.NONMANUAL, 11, 0., 1.),
        SIREvent(2, EventKind.MANUAL, 12, .15, .16),
    ], [SIREdge(1, 0, EdgeType.SCOPE)])
    annotation = GovernedSIRAnnotation.create(
        annotation_id=original.annotation_id, origin=original.origin,
        source=original.source, convention=original.convention, lexicon=original.lexicon,
        lexicon_convention_sha256=original.lexicon_convention_sha256, graph=graph,
        review=replace(original.review, reviewed_sir_sha256=sir_sha256(graph)),
        created_at=original.created_at)
    video = tmp_path / 'video'; video.write_bytes(f'video-{index}'.encode())
    transcript = tmp_path / 'transcript'; transcript.write_bytes(f'transcript-{index}'.encode())
    native = tmp_path / 'native'; native.write_bytes(f'fictional native motion {index}'.encode())
    digest = hashlib.sha256(native.read_bytes()).hexdigest()
    motion = motion_fixture()
    motion = replace(motion, sample_id=sample.sample_id, channels={
        name: replace(channel, source_id=source.source_id, source_sha256=digest)
        for name, channel in motion.channels.items()})
    motion_path = tmp_path / 'motion.npz'; motion_hash = motion.save(motion_path)
    alignment = dict(schema_version=1, sample_id=sample.sample_id,
                     source_recording_id=sample.source_id, motion_sha256=motion_hash,
                     annotation_sha256=annotation.content_sha256(),
                     video_sha256=annotation.source.video_sha256, clock_id=motion.clock_id,
                     scale=1., offset_seconds=0., interval_start_seconds=0., interval_end_seconds=1.)
    alignment_path = tmp_path / 'alignment.json'
    options = dict(motion_path=motion_path, motion_sha256=motion_hash,
                   annotation=annotation, sample=sample, video_path=video,
                   annotation_authorization_path=authorization.local_evidence,
                   transcript_path=transcript, alignment_path=alignment_path,
                   sources=(source,), authorizations={source.source_id: authorization},
                   source_files={source.source_id: native})
    bind_alignment(options, alignment)
    return options, alignment


def bind_alignment(options, alignment):
    payload = json.dumps(alignment).encode()
    options['alignment_path'].write_bytes(payload)
    options['alignment_sha256'] = hashlib.sha256(payload).hexdigest()


def test_direct_pair_preserves_temporal_graph_and_short_unsampled_events(tmp_path):
    options, _ = inputs(tmp_path)
    pair = load_governed_motion_pair(**options)
    assert pair.annotation.sir_payload == options['annotation'].sir_payload
    assert pair.event_ids == (0, 1, 2)
    assert pair.event_frame_membership.tolist() == [[True]*3, [True]*3, [False]*3]
    assert pair.annotation.graph().edges == [SIREdge(1, 0, EdgeType.SCOPE)]
    np.testing.assert_array_equal(pair.annotation_times, [0., .1, .3])


def test_half_open_end_and_affine_clock_are_exact(tmp_path):
    options, alignment = inputs(tmp_path)
    alignment.update(scale=5., offset_seconds=-.5, interval_start_seconds=-.5,
                     interval_end_seconds=1.1)
    bind_alignment(options, alignment)
    pair = load_governed_motion_pair(**options)
    np.testing.assert_array_equal(pair.annotation_times, [-.5, 0., 1.])
    assert pair.event_frame_membership[0].tolist() == [False, True, False]


@pytest.mark.parametrize('field,value', [
    ('sample_id', 'wrong'), ('source_recording_id', 'wrong'),
    ('clock_id', 'wrong'), ('motion_sha256', 'f'*64), ('annotation_sha256', 'f'*64),
    ('video_sha256', 'f'*64), ('schema_version', True), ('scale', 0.),
    ('scale', -1.), ('scale', True), ('offset_seconds', float('nan')),
    ('offset_seconds', 1e300), ('interval_end_seconds', .3),
    ('interval_end_seconds', .5), ('interval_start_seconds', .1),
])
def test_identity_and_temporal_adversaries_rejected_even_with_rehashed_manifest(tmp_path, field, value):
    options, alignment = inputs(tmp_path)
    alignment[field] = value
    bind_alignment(options, alignment)
    with pytest.raises(ValueError):
        load_governed_motion_pair(**options)


@pytest.mark.parametrize('path', ['video_path', 'transcript_path', 'alignment_path', 'motion_path',
                                'annotation_authorization_path'])
def test_changed_bound_files_fail(tmp_path, path):
    options, _ = inputs(tmp_path)
    options[path].write_bytes(b'changed')
    with pytest.raises((ValueError, PermissionError)):
        load_governed_motion_pair(**options)


def test_missing_motion_authorization_is_not_replaced_by_annotation_review(tmp_path):
    options, _ = inputs(tmp_path)
    options['authorizations'] = {}
    with pytest.raises(PermissionError, match='portfolio'):
        load_governed_motion_pair(**options)


def test_duplicate_alignment_keys_fail_even_with_correct_file_digest(tmp_path):
    options, _ = inputs(tmp_path)
    payload = options['alignment_path'].read_bytes().replace(b'"scale": 1.0', b'"scale": 1.0, "scale": 2.0')
    options['alignment_path'].write_bytes(payload)
    options['alignment_sha256'] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match='duplicate'):
        load_governed_motion_pair(**options)
