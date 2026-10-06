"""Typed optimizer plumbing with fictional evidence, not a trained ASL model."""
from dataclasses import fields, is_dataclass, replace

import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler

from signtranslator.config import TrainerConfig
from signtranslator.data.governed_corpus import (
    GovernedMotionDataset, collate_governed_motion, move_governed_batch,
)
from signtranslator.pose.multichannel import MultichannelMotion
from signtranslator.training import Trainer
from test_governed_corpus import population, admit, replace_motion


class MechanicalProbe(torch.nn.Module):
    """Scalar regression solely to exercise the real trainer's typed interface."""
    governed_batch_schema_version = 1

    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(4., dtype=torch.float64))

    def training_step(self, batch, weights):
        assert batch.sir_targets.annotation_sha256 == tuple(
            a.content_sha256() for a in batch.annotations)
        assert batch.motion.timestamps.dtype == torch.float64
        target = batch.motion.channels['root_translation'].values[:, 0, 0, 0]
        error = (self.offset - target).square().mean()
        return {'total': error, 'mechanical_error': error}

    def validation_metrics(self, *args):
        raise AssertionError('legacy hook must not bypass typed validation')


def setup(tmp_path):
    records, options = population(tmp_path, [{}, {'split': 'val'},
                                            {'split': 'val'}, {'split': 'val'}])
    for index, record in enumerate(records):
        motion = MultichannelMotion.load(record.motion_path, expected_sha256=record.motion_sha256)
        motion.channels['root_translation'].values[:, 0, 0] = index
        replace_motion(records, options, index, motion)
    return admit(records, options), records, options


def loader(corpus, split, **kwargs):
    return DataLoader(corpus.split(split), batch_size=2,
                      collate_fn=collate_governed_motion, **kwargs)


def tensors(value):
    if torch.is_tensor(value):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, tuple):
        for item in value:
            yield from tensors(item)


def test_transport_preserves_all_tensor_fields_and_bindings(tmp_path):
    corpus, _, _ = setup(tmp_path)
    batch = next(iter(loader(corpus, 'val')))
    moved = move_governed_batch(batch, 'cpu')
    expected, actual = list(tensors(batch)), list(tensors(moved))
    assert len(actual) == len(expected) and len(actual) > 50
    for left, right in zip(expected, actual):
        assert left.dtype == right.dtype and torch.equal(left, right)
    assert moved.annotations is batch.annotations
    assert moved.corpus_sha256 == batch.corpus_sha256
    assert moved.sir_targets.lexicon == batch.sir_targets.lexicon
    assert all(t.device.type == 'meta' for t in tensors(move_governed_batch(batch, 'meta')))


def test_optimizer_and_exact_sample_weighted_validation(tmp_path):
    corpus, _, _ = setup(tmp_path)
    trainer = Trainer(MechanicalProbe(), TrainerConfig(epochs=3, lr=.1, selection_metric='mechanical_error'),
                      loader(corpus, 'train'), loader(corpus, 'val'))
    # Losses are 9, 4, 1: the 2+1 batch mean average would incorrectly be 3.75.
    before = torch.get_rng_state().clone()
    assert trainer.validate()['total'] == pytest.approx(14 / 3)
    assert torch.equal(before, torch.get_rng_state())
    assert trainer.model.training
    initial = trainer.model.offset.item()
    trainer.fit()
    assert trainer.global_step == 3 and trainer.model.offset.item() < initial
    assert trainer.history['train_total'][-1] < trainer.history['train_total'][0]


def test_exact_resume_and_manifest_corpus_binding(tmp_path):
    corpus, _, _ = setup(tmp_path)
    cfg = TrainerConfig(epochs=3, lr=.1, seed=9, selection_metric='mechanical_error')
    def build():
        return Trainer(MechanicalProbe(), cfg, loader(corpus, 'train'), loader(corpus, 'val'))
    full = build()
    full.fit()
    partial = build()
    partial.fit(max_epochs=1)
    path = partial.save(tmp_path / 'typed.pt')
    resumed = build()
    resumed.load(path)
    resumed.fit()
    assert resumed.history == full.history
    assert torch.equal(resumed.model.offset, full.model.offset)
    assert resumed.data_contract['train']['governed']['corpus_sha256'] == corpus.content_sha256
    assert resumed.data_contract['validation']['governed']['record_indices'] == [1, 2, 3]


def test_corpus_mismatch_resume_rejected_before_weights_change(tmp_path):
    corpus, records, options = setup(tmp_path)
    cfg = TrainerConfig(epochs=2, lr=.1)
    trained = Trainer(MechanicalProbe(), cfg, loader(corpus, 'train'))
    trained.fit(max_epochs=1)
    path = trained.save(tmp_path / 'typed.pt')
    # Same size and schema, different admitted ordering and corpus identity.
    other = admit(list(reversed(records)), options)
    candidate = Trainer(MechanicalProbe(), cfg, loader(other, 'train'))
    before = candidate.model.offset.detach().clone()
    with pytest.raises(ValueError, match='data-loader contract'):
        candidate.load(path)
    assert torch.equal(before, candidate.model.offset)
    assert candidate.global_step == 0 and not candidate.opt.state


@pytest.mark.parametrize('failure', ['opt_in', 'train_split', 'val_split', 'corpus',
                                    'collator', 'drop_last', 'mixed'])
def test_bad_training_routes_fail_before_optimization(tmp_path, failure):
    corpus, records, options = setup(tmp_path)
    model = MechanicalProbe()
    train, val = loader(corpus, 'train'), loader(corpus, 'val')
    if failure == 'opt_in':
        model.governed_batch_schema_version = True
    elif failure == 'train_split':
        train = loader(corpus, 'val')
    elif failure == 'val_split':
        val = loader(corpus, 'train')
    elif failure == 'corpus':
        val = loader(admit(list(reversed(records)), options), 'val')
    elif failure == 'collator':
        train.collate_fn = lambda items: {'fake': items}
    elif failure == 'drop_last':
        val = loader(corpus, 'val', drop_last=True)
    else:
        val = DataLoader([torch.zeros(1)])
    with pytest.raises(ValueError):
        Trainer(model, TrainerConfig(epochs=1), train, val)


def test_changed_evidence_prevents_optimizer_step(tmp_path):
    corpus, records, _ = setup(tmp_path)
    trainer = Trainer(MechanicalProbe(), TrainerConfig(epochs=1), loader(corpus, 'train'))
    records[0].video_path.write_bytes(b'changed evidence')
    with pytest.raises((ValueError, PermissionError)):
        trainer.train_epoch()
    assert trainer.global_step == 0 and not trainer.opt.state
    assert trainer.model.offset.item() == 4.


def test_changed_collator_and_forged_batch_are_rejected(tmp_path):
    corpus, _, _ = setup(tmp_path)
    train = loader(corpus, 'train')
    trainer = Trainer(MechanicalProbe(), TrainerConfig(epochs=1), train)
    batch = next(iter(train))
    with pytest.raises(ValueError, match='corpus/split'):
        trainer._prepare_batch(replace(batch, split='test'), train, 'train')
    with pytest.raises(TypeError, match='typed'):
        trainer._prepare_batch({}, train, 'train')
    train.collate_fn = lambda items: {}
    with pytest.raises(ValueError, match='collator'):
        trainer.train_epoch()
    assert trainer.global_step == 0


@pytest.mark.parametrize('indices,split', [((0, 0), 'train'), ((1,), 'train'),
                                         ((True,), 'val'), ((-1,), 'train'), ((), 'train')])
def test_dataset_view_cannot_forge_partition(tmp_path, indices, split):
    corpus, _, _ = setup(tmp_path)
    with pytest.raises(ValueError):
        GovernedMotionDataset(corpus, indices, split)


@pytest.mark.parametrize('metric', ['', 'total', None, True, 'error with spaces', 'λ'])
def test_selection_requires_an_explicit_named_branch(metric):
    with pytest.raises(ValueError, match='branch loss'):
        TrainerConfig(selection_metric=metric)


@pytest.mark.parametrize('sampling', ['replacement', 'generator', 'shuffled_validation'])
def test_undeclared_sampling_design_is_rejected(tmp_path, sampling):
    corpus, _, _ = setup(tmp_path)
    train, val = loader(corpus, 'train'), loader(corpus, 'val')
    if sampling == 'replacement':
        dataset = corpus.split('train')
        train = DataLoader(dataset, batch_size=2, collate_fn=collate_governed_motion,
                           sampler=RandomSampler(dataset, replacement=True))
    elif sampling == 'generator':
        train = loader(corpus, 'train', generator=torch.Generator())
    else:
        val = loader(corpus, 'val', shuffle=True)
    with pytest.raises(ValueError, match='sampling'):
        Trainer(MechanicalProbe(), TrainerConfig(epochs=1), train, val)
