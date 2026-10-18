"""Canonical-state generation mathematics and support contracts.

Every fixture is explicitly fictional synthetic evidence. These tests establish
encoded behaviour, never ASL adequacy, calibration or phase acceptance.
"""
from dataclasses import replace
import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from signtranslator.config import TrainerConfig
from signtranslator.data.governed_corpus import collate_governed_motion
from signtranslator.data.governed_text import PLAINTEXT_ENCODING
from signtranslator.generation import (
    CANONICAL_BRANCHES, CanonicalChannelBinding, CanonicalStateSIRDecoder,
    channel_bindings_from_state, channel_loss_per_example,
)
from signtranslator.planning.loci import LocusAlphabet
from signtranslator.pose.multichannel import CHANNELS, MultichannelMotion
from signtranslator.pose.rotations import (
    axis_angle_to_matrix, geodesic_distance, matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from signtranslator.training import checkpoint_paths
from signtranslator.training.trainer import Trainer
from test_loci import PAYLOAD
from test_pose_multichannel import fixture as motion_fixture
from test_text_sir_labels import setup


WEIGHTS = {name: 1. for name in CANONICAL_BRANCHES}


def corpus_fixture(tmp_path, configurations=None):
    vocabulary, corpus = setup(
        tmp_path,
        configurations=[{}, {}, {'split': 'val'}] if configurations is None else configurations,
        convention_payload=PAYLOAD, loci=(0, None, 1))
    alphabet = LocusAlphabet(vocabulary.convention, PAYLOAD)
    return vocabulary, alphabet, corpus


def build_model(vocabulary, alphabet, **kwargs):
    bindings = channel_bindings_from_state(motion_fixture())
    defaults = dict(hidden_dim=8, event_dim=8, attention_heads=2, frame_frequencies=2,
                    max_frames=16, max_events=4, velocity_weight=1.)
    return CanonicalStateSIRDecoder(vocabulary, locus_alphabet=alphabet,
                                    bindings=bindings, **{**defaults, **kwargs})


def one_batch(corpus, split='train', batch_size=2):
    loader = DataLoader(corpus.split(split), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_governed_motion)
    return next(iter(loader))


# ---------------------------------------------------------------------------
# channel mathematics
# ---------------------------------------------------------------------------
def rotation_channel(target_6d, *, confidence=1.):
    """Build a single-example rotation batch of explicit 6D target values."""
    frames = target_6d.shape[0]
    reference = motion_fixture()
    channels = {}
    for name, (kind, width, units) in CHANNELS.items():
        values = np.zeros((frames, 1, width), dtype=np.float64)
        if kind == 'rotation_6d_columns':
            values[:] = [1, 0, 0, 0, 1, 0]
        elif kind == 'unit_vector':
            values[:] = [1, 0, 0]
        if name == 'body':
            values = target_6d.reshape(frames, 1, 6).to(torch.float64).numpy().copy()
        channels[name] = replace(
            reference.channels[name], values=values,
            valid=np.ones((frames, 1), dtype=bool),
            observed=np.ones((frames, 1), dtype=bool),
            confidence=np.full((frames, 1), confidence, dtype=np.float32))
    state = MultichannelMotion(np.arange(frames, dtype=np.float64) * .1,
                               reference.clock_id, reference.sample_id, channels)
    from signtranslator.data.multichannel import collate_multichannel
    batch = collate_multichannel([state])
    return batch.channels['body'], batch.frame_valid, batch.timestamps


def test_rotation_branch_equals_half_angle_identity_and_is_zero_only_at_equality():
    torch.manual_seed(3)
    axes = torch.randn(5, 3, dtype=torch.float64)
    axes = axes / torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
    angles = torch.tensor([0., .25, 1., 2., math.pi - 1e-6], dtype=torch.float64)
    target = matrix_to_rotation_6d(torch.eye(3, dtype=torch.float64).expand(5, 3, 3))
    prediction = matrix_to_rotation_6d(axis_angle_to_matrix(axes * angles[:, None]))
    channel, frame_valid, timestamps = rotation_channel(target)
    from signtranslator.generation.canonical_state import CanonicalStateConfig
    config = CanonicalStateConfig(
        hidden_dim=4, event_dim=4, attention_heads=2, frame_frequencies=2,
        translation_scale_metres=.05, coefficient_scale=.5, velocity_weight=0.,
        max_frames=16, max_events=4, lexicon_sha256='0' * 64,
        convention_sha256='1' * 64, locus_count=1,
        bindings=channel_bindings_from_state(motion_fixture()))
    values, support, diagnostics = channel_loss_per_example(
        'body', prediction.reshape(1, 5, 1, 6), channel, frame_valid, config=config,
        timestamps=timestamps)
    expected = (1. - torch.cos(angles)) / 2.
    assert bool(support.all())
    assert float(values[0]) == pytest.approx(float(expected.mean()), rel=0, abs=1e-12)
    # The reported geodesic diagnostic is the true angle, not the optimised loss.
    assert diagnostics['geodesic_degrees_max'] == pytest.approx(
        float(angles.max() * 180. / math.pi), abs=1e-6)
    # Zero exactly at equality, and strictly increasing in the geodesic angle.
    identical, _, _ = channel_loss_per_example(
        'body', target.reshape(1, 5, 1, 6), channel, frame_valid, config=config,
        timestamps=timestamps)
    assert float(identical[0]) == 0.
    assert all(float(expected[index]) < float(expected[index + 1])
               for index in range(len(angles) - 1))


def test_rotation_branch_matches_chordal_and_geodesic_definitions():
    torch.manual_seed(11)
    first = axis_angle_to_matrix(torch.randn(7, 3, dtype=torch.float64))
    second = axis_angle_to_matrix(torch.randn(7, 3, dtype=torch.float64))
    chordal = (first - second).square().sum(dim=(-1, -2)) / 8.
    angle = geodesic_distance(first, second)
    torch.testing.assert_close(chordal, (1. - torch.cos(angle)) / 2., rtol=0., atol=1e-12)
    torch.testing.assert_close(chordal, torch.sin(angle / 2.).square(), rtol=0., atol=1e-12)


def test_bounded_branch_is_bernoulli_divergence_with_exact_zero_minimum():
    from signtranslator.generation.canonical_state import _bounded_cells
    target = torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)
    logits = torch.logit(target.clamp(1e-12, 1 - 1e-12))
    at_target = _bounded_cells(logits, target)
    torch.testing.assert_close(at_target, torch.zeros_like(at_target), rtol=0., atol=1e-9)
    worse = _bounded_cells(logits + 1., target)
    assert bool((worse > 0).all())
    # Exact divergence identity at an interior point.
    t, p = .25, .5
    expected = t * math.log(t / p) + (1 - t) * math.log((1 - t) / (1 - p))
    assert float(_bounded_cells(torch.tensor([0.], dtype=torch.float64),
                                torch.tensor([t], dtype=torch.float64))) == pytest.approx(
        expected, abs=1e-12)


def test_direction_branch_ignores_magnitude_and_spans_zero_to_one():
    from signtranslator.generation.canonical_state import _direction_cells
    target = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
    for scale in (1e-3, 1., 1e3):
        aligned = _direction_cells(target * scale, target)
        assert float(aligned) == pytest.approx(0., abs=1e-12)
    opposite = _direction_cells(-target, target)
    assert float(opposite) == pytest.approx(1., abs=1e-12)
    orthogonal = _direction_cells(torch.tensor([[1., 0., 0.]], dtype=torch.float64), target)
    assert float(orthogonal) == pytest.approx(.5, abs=1e-12)
    with pytest.raises(ValueError, match="numerically unresolved"):
        _direction_cells(torch.zeros((1, 3), dtype=torch.float64), target)


def test_translation_velocity_term_uses_actual_elapsed_seconds(tmp_path):
    vocabulary, alphabet, corpus = corpus_fixture(tmp_path)
    batch = one_batch(corpus, batch_size=1)
    model = build_model(vocabulary, alphabet)
    prediction = torch.zeros_like(batch.motion.channels['root_translation'].values)
    prediction[:, 1:] = 1.  # a constant displacement after the first frame
    fast, _, _ = channel_loss_per_example(
        'root_translation', prediction, batch.motion.channels['root_translation'],
        batch.motion.frame_valid, config=model.model_cfg, timestamps=batch.motion.timestamps)
    slow, _, _ = channel_loss_per_example(
        'root_translation', prediction, batch.motion.channels['root_translation'],
        batch.motion.frame_valid, config=model.model_cfg,
        timestamps=batch.motion.timestamps * 10.)
    # The same displacement over a ten-times longer clock is a smaller velocity
    # error, so a frame-index surrogate cannot reproduce this ordering.
    assert float(fast[0]) > float(slow[0])
    without = replace(model.model_cfg, velocity_weight=0.)
    positions, _, _ = channel_loss_per_example(
        'root_translation', prediction, batch.motion.channels['root_translation'],
        batch.motion.frame_valid, config=without, timestamps=batch.motion.timestamps)
    assert float(positions[0]) < float(slow[0])


# ---------------------------------------------------------------------------
# support semantics
# ---------------------------------------------------------------------------
def test_unavailable_channel_reports_zero_support_not_a_measured_zero(tmp_path):
    vocabulary, alphabet, corpus = corpus_fixture(tmp_path)
    batch = one_batch(corpus, batch_size=2)
    model = build_model(vocabulary, alphabet)
    channel = batch.motion.channels['face']
    hidden = replace(channel, valid=torch.zeros_like(channel.valid),
                     confidence=torch.zeros_like(channel.confidence),
                     values=torch.zeros_like(channel.values))
    motion = replace(batch.motion, channels={**batch.motion.channels, 'face': hidden})
    objective = model.training_step(replace(batch, motion=motion), weights=WEIGHTS)
    assert objective.terms['face'].supported_examples == 0
    assert float(objective.terms['face'].example_loss_sum.detach()) == 0.
    with pytest.raises(KeyError, match='unavailable'):
        objective['face']
    assert objective.terms['body'].supported_examples == 2
    assert bool(torch.isfinite(objective.total()))


def test_zero_confidence_observation_is_unsupported_and_masks_change_the_loss(tmp_path):
    vocabulary, alphabet, corpus = corpus_fixture(tmp_path)
    batch = one_batch(corpus, batch_size=1)
    model = build_model(vocabulary, alphabet)
    torch.manual_seed(5)
    for parameter in model.parameters():
        with torch.no_grad():
            parameter.add_(torch.randn_like(parameter) * .1)
    baseline = float(model.training_step(batch, weights=WEIGHTS)['body'].detach())
    channel = batch.motion.channels['body']
    zeroed = replace(channel, confidence=torch.zeros_like(channel.confidence))
    motion = replace(batch.motion, channels={**batch.motion.channels, 'body': zeroed})
    objective = model.training_step(replace(batch, motion=motion), weights=WEIGHTS)
    assert objective.terms['body'].supported_examples == 0
    half = replace(channel, confidence=channel.confidence * .5)
    scaled = model.training_step(replace(batch, motion=replace(
        batch.motion, channels={**batch.motion.channels, 'body': half})), weights=WEIGHTS)
    # A uniform reliability rescaling leaves the weighted mean unchanged, while
    # removing support removes the branch entirely.
    assert float(scaled['body'].detach()) == pytest.approx(baseline, rel=1e-9)


def test_temporal_padding_cannot_change_a_branch_loss(tmp_path):
    """Appending invalid frames leaves the weighted example mean unchanged."""
    vocabulary, alphabet, corpus = corpus_fixture(tmp_path)
    batch = one_batch(corpus, batch_size=1)
    model = build_model(vocabulary, alphabet)
    channel = batch.motion.channels['body']
    frames = channel.values.shape[1]
    prediction = torch.zeros_like(channel.values)
    prediction[..., 0] = 1.
    prediction[..., 4] = 1.
    baseline, support, _ = channel_loss_per_example(
        'body', prediction, channel, batch.motion.frame_valid, config=model.model_cfg,
        timestamps=batch.motion.timestamps)

    def pad(tensor, value):
        extra = torch.full((tensor.shape[0], 2, *tensor.shape[2:]), value,
                           dtype=tensor.dtype)
        return torch.cat((tensor, extra), dim=1)

    padded_channel = replace(
        channel, values=pad(channel.values, 0.), valid=pad(channel.valid, False),
        observed=pad(channel.observed, False), confidence=pad(channel.confidence, 0.))
    padded = channel_loss_per_example(
        'body', pad(prediction, 0.), padded_channel,
        pad(batch.motion.frame_valid, False), config=model.model_cfg,
        timestamps=pad(batch.motion.timestamps, 0.))
    assert float(padded[0][0]) == pytest.approx(float(baseline[0]), rel=0., abs=1e-12)
    assert bool(padded[1].all()) and bool(support.all())
    assert padded_channel.values.shape[1] == frames + 2


# ---------------------------------------------------------------------------
# conditioning contracts
# ---------------------------------------------------------------------------
def test_conditioning_never_reads_channel_values(tmp_path):
    vocabulary, alphabet, corpus = corpus_fixture(tmp_path)
    batch = one_batch(corpus, batch_size=2)
    model = build_model(vocabulary, alphabet)
    torch.manual_seed(2)
    for parameter in model.parameters():
        with torch.no_grad():
            parameter.add_(torch.randn_like(parameter) * .1)
    before = model.predict(batch)
    channels = dict(batch.motion.channels)
    translated = channels['root_translation']
    channels['root_translation'] = replace(
        translated, values=translated.values + 7.)
    after = model.predict(replace(batch, motion=replace(batch.motion, channels=channels)))
    for name in CANONICAL_BRANCHES:
        assert torch.equal(before[name], after[name])


def test_frame_outside_every_event_still_has_a_defined_attention(tmp_path):
    vocabulary, alphabet, corpus = corpus_fixture(tmp_path)
    batch = one_batch(corpus, batch_size=1)
    empty = tuple(torch.zeros_like(mask) for mask in batch.event_frame_membership)
    model = build_model(vocabulary, alphabet)
    predictions = model.predict(replace(batch, event_frame_membership=empty))
    for name in CANONICAL_BRANCHES:
        assert bool(torch.isfinite(predictions[name]).all())


def test_reordered_membership_or_changed_layout_is_rejected(tmp_path):
    vocabulary, alphabet, corpus = corpus_fixture(tmp_path)
    batch = one_batch(corpus, batch_size=1)
    model = build_model(vocabulary, alphabet)
    flipped = (batch.event_frame_membership[0].flip(0),)
    reordered = replace(batch, event_ids=((2, 1, 0),), event_frame_membership=flipped)
    with pytest.raises(ValueError, match='membership order'):
        model.predict(reordered)
    layout = batch.motion.channels['body']
    renamed = replace(layout, layout=replace(layout.layout, coordinate_frame='other-frame'))
    with pytest.raises(ValueError, match='channel layout'):
        model.predict(replace(batch, motion=replace(
            batch.motion, channels={**batch.motion.channels, 'body': renamed})))


@pytest.mark.parametrize('weights,reason', [
    ({name: 1. for name in CANONICAL_BRANCHES if name != 'blink'}, 'missing'),
    ({**WEIGHTS, 'blink': 0.}, 'nonpositive'),
    ({**WEIGHTS, 'blink': float('nan')}, 'nonfinite'),
])
def test_branch_weight_contract(tmp_path, weights, reason):
    vocabulary, alphabet, corpus = corpus_fixture(tmp_path)
    batch = one_batch(corpus, batch_size=1)
    model = build_model(vocabulary, alphabet)
    with pytest.raises(ValueError, match='canonical channels'):
        model.training_step(batch, weights=weights)
    assert reason


def test_capacity_and_binding_declarations_are_enforced(tmp_path):
    vocabulary, alphabet, corpus = corpus_fixture(tmp_path)
    batch = one_batch(corpus, batch_size=1)
    with pytest.raises(ValueError, match='frame capacity'):
        build_model(vocabulary, alphabet, max_frames=2).predict(batch)
    with pytest.raises(ValueError, match='event capacity'):
        build_model(vocabulary, alphabet, max_events=1).predict(batch)
    with pytest.raises(ValueError, match='exactly one channel element'):
        CanonicalChannelBinding('left_gaze', ('a', 'b'), 'frame', 'unitless', 'convention')
    with pytest.raises(ValueError, match='units must be'):
        CanonicalChannelBinding('root_translation', ('a',), 'frame', 'unitless', 'convention')


def test_degenerate_rotation_prediction_fails_explicitly(tmp_path):
    vocabulary, alphabet, corpus = corpus_fixture(tmp_path)
    batch = one_batch(corpus, batch_size=1)
    model = build_model(vocabulary, alphabet)
    collinear = torch.zeros_like(batch.motion.channels['body'].values)
    collinear[..., 0] = 1.
    collinear[..., 3] = 1.  # second axis parallel to the first
    with pytest.raises(ValueError, match='collinear or numerically unresolved'):
        channel_loss_per_example('body', collinear, batch.motion.channels['body'],
                                 batch.motion.frame_valid, config=model.model_cfg,
                                 timestamps=batch.motion.timestamps)


# ---------------------------------------------------------------------------
# governed trainer integration
# ---------------------------------------------------------------------------
def trainer_for(values, **config):
    vocabulary, alphabet, corpus = values
    model = build_model(vocabulary, alphabet)
    train = DataLoader(corpus.split('train'), batch_size=2, shuffle=False, num_workers=0,
                       drop_last=False, collate_fn=collate_governed_motion)
    validation = DataLoader(corpus.split('val'), batch_size=1, shuffle=False, num_workers=0,
                            drop_last=False, collate_fn=collate_governed_motion)
    settings = TrainerConfig(epochs=2, batch_size=2, lr=1e-2, seed=17, loss_weights=WEIGHTS,
                             selection_metric='body', **config)
    return model, Trainer(model, settings, train, validation), corpus


def test_real_trainer_optimizes_every_supported_branch(tmp_path):
    torch.manual_seed(29)
    model, trainer, _ = trainer_for(corpus_fixture(tmp_path))
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    history = trainer.fit()
    assert trainer.global_step == 2
    for name in CANONICAL_BRANCHES:
        assert all(math.isfinite(value) for value in history[f'train_{name}'])
        assert history[f'train_support_{name}'] == [2, 2]
    assert any(not torch.equal(before[name], tensor)
               for name, tensor in model.state_dict().items())
    report = trainer.exposure_report().to_dict()
    assert report['recorded_optimizer_steps'] == 2
    assert report['committed_epoch_boundary'] is True
    assert report['phase_exit_approved'] is False
    # Dense motion declares support membership without cell-level targets.
    assert all(record['target_cells'][name] is None
               for record in trainer.optimizer_exposure for name in CANONICAL_BRANCHES)


def test_exact_checkpoint_continuation_matches_uninterrupted_training(tmp_path):
    values = corpus_fixture(tmp_path)
    prefix = str(tmp_path / 'run.pt')
    torch.manual_seed(31)
    _, full, _ = trainer_for(values)
    full.fit()
    torch.manual_seed(31)
    model, partial, _ = trainer_for(values, ckpt_path=prefix)
    partial.fit(max_epochs=1)
    torch.manual_seed(31)
    resumed_model, resumed, _ = trainer_for(values, ckpt_path=prefix)
    resumed.load(checkpoint_paths(prefix)['last'], mode='resume')
    resumed.fit()
    assert resumed.history == full.history
    assert all(torch.equal(value, full.model.state_dict()[name])
               for name, value in resumed_model.state_dict().items())
    del model


# ---------------------------------------------------------------------------
# generated state provenance
# ---------------------------------------------------------------------------
def test_generated_state_is_structurally_unusable_as_supervision(tmp_path):
    vocabulary, alphabet, corpus = corpus_fixture(tmp_path)
    batch = one_batch(corpus, batch_size=2)
    model = build_model(vocabulary, alphabet)
    torch.manual_seed(37)
    for parameter in model.parameters():
        with torch.no_grad():
            parameter.add_(torch.randn_like(parameter) * .2)
    states = model.generate_canonical_state(batch, inference_method='conditional-regression')
    assert len(states) == 2
    for row, state in enumerate(states):
        state.validate()
        assert state.sample_id == f'generated:{batch.motion.sample_ids[row]}'
        for name, channel in state.channels.items():
            assert not channel.observed.any()
            assert channel.valid.all()
            assert not channel.confidence.any()
            assert not channel.supervision_weights().any()
            assert not channel.supervision_weights(allow_inferred=True).any()
            assert 'conditional-regression' in channel.inference_method
            assert channel.source_sha256 == model.state_sha256()
            kind = CHANNELS[name][0]
            values = torch.from_numpy(channel.values)
            if kind == 'rotation_6d_columns':
                from signtranslator.pose.rotations import is_rotation_matrix
                assert bool(is_rotation_matrix(rotation_6d_to_matrix(values)).all())
            elif kind == 'unit_vector':
                norms = torch.linalg.vector_norm(values, dim=-1)
                torch.testing.assert_close(norms, torch.ones_like(norms), rtol=0., atol=1e-12)
            elif kind == 'bounded_scalar':
                assert bool(((values >= 0) & (values <= 1)).all())


def test_generated_state_round_trips_through_the_canonical_archive(tmp_path):
    vocabulary, alphabet, corpus = corpus_fixture(tmp_path)
    batch = one_batch(corpus, batch_size=1)
    model = build_model(vocabulary, alphabet)
    state = model.generate_canonical_state(batch, inference_method='diagnostic')[0]
    path = tmp_path / 'generated.npz'
    digest = state.save(path)
    reloaded = MultichannelMotion.load(path, expected_sha256=digest)
    assert reloaded.clock_id == state.clock_id
    for name in CANONICAL_BRANCHES:
        np.testing.assert_array_equal(reloaded.channels[name].values,
                                      state.channels[name].values)
        assert reloaded.channels[name].inference_method == state.channels[name].inference_method
    with pytest.raises(ValueError, match='explicit inference method'):
        model.generate_canonical_state(batch, inference_method=' ')
