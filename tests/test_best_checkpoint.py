import hashlib
import json

import pytest
import torch

from signtranslator.config import TrainerConfig
from signtranslator.training import Trainer
from test_governed_trainer import MechanicalProbe, setup, loader


def build(corpus):
    return Trainer(MechanicalProbe(), TrainerConfig(epochs=3, selection_metric='mechanical_error'),
                   loader(corpus, 'train'), loader(corpus, 'val'))


def rewrite(path, change):
    data = torch.load(path, weights_only=False)
    change(data)
    torch.save(data, path)
    manifest_path = path.with_suffix(path.suffix + '.json')
    manifest = json.loads(manifest_path.read_text())
    manifest['schema_version'] = data['schema_version']
    manifest['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['checkpoint_size'] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))


def test_resume_preserves_earlier_best_when_future_validation_worsens(tmp_path):
    corpus, _, _ = setup(tmp_path)
    trainer = build(corpus)
    trainer.validate = lambda: {'mechanical_error': 1.}
    trainer.fit(max_epochs=1)
    best = {k: v.clone() for k, v in trainer.best_model_state.items()}
    trainer.validate = lambda: {'mechanical_error': 2.}
    trainer.fit(max_epochs=1)
    assert any(not torch.equal(v, best[k]) for k, v in trainer.model.state_dict().items())
    path = trainer.save(tmp_path / 'last.pt')
    resumed = build(corpus)
    resumed.load(path)
    resumed.validate = lambda: {'mechanical_error': 3.}
    resumed.fit()
    assert resumed.best_val == 1.
    assert all(torch.equal(v, best[k]) for k, v in resumed.best_model_state.items())
    assert all(v.device.type == 'cpu' for v in resumed.best_model_state.values())
    for parameter in resumed.model.parameters():
        parameter.data.add_(100.)
    assert all(torch.equal(v, best[k]) for k, v in resumed.best_model_state.items())


def test_prevalidation_checkpoint_and_old_schema_warm_start(tmp_path):
    corpus, _, _ = setup(tmp_path)
    trainer = build(corpus)
    path = trainer.save(tmp_path / 'initial.pt')
    restored = build(corpus)
    restored.load(path)
    assert restored.best_model_state is None
    def downgrade(data):
        data['schema_version'] = 2
        del data['best_model_state']
    rewrite(path, downgrade)
    with pytest.raises(ValueError, match='schema 2 lacks'):
        restored.load(path)
    restored.load(path, mode='weights')


def test_invalid_best_state_rejected_before_current_weights_change(tmp_path):
    corpus, _, _ = setup(tmp_path)
    trainer = build(corpus)
    trainer.fit(max_epochs=1)
    path = trainer.save(tmp_path / 'invalid.pt')
    rewrite(path, lambda data: data.update(best_model_state=None))
    fresh = build(corpus)
    before = {k: v.clone() for k, v in fresh.model.state_dict().items()}
    with pytest.raises(ValueError, match='best metric'):
        fresh.load(path)
    assert all(torch.equal(v, before[k]) for k, v in fresh.model.state_dict().items())
