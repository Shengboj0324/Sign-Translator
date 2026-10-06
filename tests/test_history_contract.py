"""Committed histories agree with exposure, schedule and selection bookkeeping."""
import hashlib
import json

import pytest
import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from signtranslator.data.governed_corpus import collate_governed_motion
from signtranslator.training import Trainer
from test_epoch_commit import make
from test_supported_trainer import mixed, config, model, loader


def corrupt_history(state, mutation):
    history = state['history']
    if mutation == 'missing_total':
        history['train_total'].clear()
    elif mutation == 'epoch_tag':
        history['train_total_epoch'][0] = 2
    elif mutation == 'float_tag':
        history['train_total_epoch'][0] = 1.
    elif mutation == 'support':
        history['train_support_sir_relations'][0] = 2
    elif mutation == 'float_support':
        history['train_support_total'][0] = 3.
    elif mutation == 'validation_population':
        history['val_support_total'][0] = 2
    elif mutation == 'rate':
        history['lr'][0] = .9
    elif mutation == 'negative':
        history['train_total'][0] = -1.
    elif mutation == 'extra':
        history['unrecorded'] = [1.]
    else:
        state['best_val'] += 1.


@pytest.mark.parametrize('mutation', ['missing_total', 'epoch_tag', 'float_tag', 'support',
    'float_support', 'validation_population', 'rate', 'negative', 'extra', 'best'])
def test_history_corruption_refused_before_resume_mutation(tmp_path, mutation):
    vocabulary, corpus = mixed(tmp_path)
    source = make(corpus, vocabulary)
    source.fit(max_epochs=1)
    path = source.save(tmp_path / 'source.pt')
    data = torch.load(path, weights_only=False)
    corrupt_history(data['training_state'], mutation)
    torch.save(data, path)
    sidecar = path.with_suffix('.pt.json')
    manifest = json.loads(sidecar.read_text())
    manifest['training_state'] = data['training_state']
    manifest['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['checkpoint_size'] = path.stat().st_size
    sidecar.write_text(json.dumps(manifest))
    receiver = make(corpus, vocabulary)
    before = {name: value.clone() for name, value in receiver.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='support-aware history'):
        receiver.load(path)
    assert all(torch.equal(before[name], value) for name, value in receiver.model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state()) and not receiver.opt.state
    assert receiver.global_step == 0 and receiver._epoch_committed


def test_missing_live_history_cannot_be_saved(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    trainer.fit(max_epochs=1)
    trainer.history['train_total'].clear()
    destination = tmp_path / 'not-created' / 'invalid.pt'
    with pytest.raises(ValueError, match='support-aware history'):
        trainer.save(destination)
    assert not destination.parent.exists()


@pytest.mark.parametrize('validation', [False, True])
@pytest.mark.parametrize('drop_last', [False, True])
def test_partial_batches_and_sparse_validation_round_trip(tmp_path, validation, drop_last):
    vocabulary, corpus = mixed(tmp_path)
    cfg = config(epochs=3)
    cfg.val_every = 2
    def build():
        view = corpus.split('train')
        train = DataLoader(view, batch_sampler=BatchSampler(SequentialSampler(view), 2, drop_last),
                           collate_fn=collate_governed_motion)
        return Trainer(model(vocabulary), cfg, train, loader(corpus, 'val') if validation else None)
    trainer = build()
    trainer.save(tmp_path / 'zero.pt')
    for epoch in range(3):
        trainer.fit(max_epochs=1)
        path = trainer.save(tmp_path / f'epoch-{epoch}.pt')
        previous = trainer.history
        trainer = build()
        trainer.load(path)
        assert trainer.history == previous
    assert trainer.history.get('val_total_epoch', []) == ([2] if validation else [])
    assert trainer.history['train_support_total'] == [2 if drop_last else 3] * 3


def test_explicit_validation_batch_sampler_cannot_drop_population(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    view = corpus.split('val')
    validation = DataLoader(view, batch_sampler=BatchSampler(SequentialSampler(view), 2, True),
                            collate_fn=collate_governed_motion)
    assert not validation.drop_last and validation.batch_sampler.drop_last
    with pytest.raises(ValueError, match='validation must not drop'):
        Trainer(model(vocabulary), config(), loader(corpus, 'train'), validation)


def test_zero_support_metrics_remain_absent_through_checkpoint(tmp_path):
    from test_text_sir_labels import setup

    vocabulary, corpus = setup(tmp_path, single_event_indices=(0, 1))
    def build():
        return Trainer(model(vocabulary), config(), loader(corpus, 'train'), loader(corpus, 'val'))
    source = build()
    source.fit(max_epochs=1)
    assert 'train_sir_relations' not in source.history and 'val_sir_relations' not in source.history
    path = source.save(tmp_path / 'unsupported.pt')
    receiver = build()
    receiver.load(path)
    receiver.fit()
    assert receiver.history['train_support_sir_relations'] == [0, 0]
    assert 'train_sir_relations' not in receiver.history


def test_duplicate_sample_across_batches_cannot_claim_full_epoch(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    source = make(corpus, vocabulary)
    source.fit(max_epochs=1)
    records = source.optimizer_exposure
    records[1]['sample_ids'][0] = records[0]['sample_ids'][0]
    records[1]['annotation_sha256'][0] = records[0]['annotation_sha256'][0]
    source._optimizer_exposure = [json.dumps(record) for record in records]
    with pytest.raises(ValueError, match='epoch sample membership'):
        source.save(tmp_path / 'duplicate.pt')
    assert not (tmp_path / 'duplicate.pt').exists()
