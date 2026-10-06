from copy import deepcopy
import hashlib
import json

import pytest
import torch

from test_best_checkpoint import build
from test_governed_trainer import setup


@pytest.mark.parametrize('field,value', [
    ('completed_epochs', True), ('completed_epochs', 1.5), ('completed_epochs', '1'),
    ('completed_epochs', -1), ('completed_epochs', 4),
    ('global_step', False), ('global_step', 1.5), ('global_step', -1),
    ('best_val', True), ('history', []), ('history', {'loss': [True]}),
    ('history', {'loss': ['1.']}), ('history', {'loss': 1.}), ('history', {'': []}),
])
def test_invalid_progress_rejected_before_model_optimizer_scheduler_or_rng_mutation(tmp_path, field, value):
    corpus, _, _ = setup(tmp_path)
    trained = build(corpus)
    trained.fit(max_epochs=1)
    path = trained.save(tmp_path / 'progress.pt')
    payload = torch.load(path, weights_only=False)
    payload['training_state'][field] = value
    torch.save(payload, path)
    sidecar = path.with_suffix('.pt.json')
    manifest = json.loads(sidecar.read_text())
    manifest['training_state'] = payload['training_state']
    manifest['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['checkpoint_size'] = path.stat().st_size
    sidecar.write_text(json.dumps(manifest, allow_nan=False))
    fresh = build(corpus)
    weights = {k: v.clone() for k, v in fresh.model.state_dict().items()}
    optimizer = deepcopy(fresh.opt.state_dict())
    scheduler = deepcopy(fresh.sched.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='checkpoint'):
        fresh.load(path)
    assert all(torch.equal(v, weights[k]) for k, v in fresh.model.state_dict().items())
    assert fresh.opt.state_dict() == optimizer
    assert fresh.sched.state_dict() == scheduler
    assert torch.equal(torch.get_rng_state(), rng)
    assert fresh.completed_epochs == fresh.global_step == 0
    assert fresh.history == {} and fresh.best_model_state is None


def test_resume_preserves_integer_support_history(tmp_path):
    from test_supported_trainer import mixed, loader, model, config
    from signtranslator.training import Trainer
    vocabulary, corpus = mixed(tmp_path)
    def make():
        return Trainer(model(vocabulary), config(), loader(corpus, 'train'), loader(corpus, 'val'))
    trainer = make()
    trainer.fit(max_epochs=1)
    path = trainer.save(tmp_path / 'support.pt')
    restored = make()
    restored.load(path)
    assert restored.history == trainer.history
    for name, values in restored.history.items():
        if '_support_' in name or name.endswith('_epoch'):
            assert all(type(value) is int for value in values)
