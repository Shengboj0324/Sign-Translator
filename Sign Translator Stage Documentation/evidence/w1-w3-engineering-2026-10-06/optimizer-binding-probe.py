"""Run from the repository root; fictional evidence only."""
import json
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_epoch_commit import make
from test_supported_trainer import mixed


with tempfile.TemporaryDirectory() as directory:
    vocabulary, corpus = mixed(Path(directory))
    trainer = make(corpus, vocabulary)
    original = trainer.model.byte_embedding.weight
    replacement = torch.nn.Parameter(original.detach().clone())
    trainer.model.byte_embedding.weight = replacement
    before = replacement.detach().clone()
    try:
        trainer.fit(max_epochs=1)
        outcome = {'error': None}
    except ValueError as error:
        outcome = {'error': type(error).__name__, 'message': str(error)}
    outcome.update(
        optimizer_steps=trainer.global_step,
        exposure_records=len(trainer.optimizer_exposure),
        replacement_has_gradient=replacement.grad is not None,
        replacement_unchanged=torch.equal(before, replacement),
        optimizer_still_owns_original=any(
            parameter is original for group in trainer.opt.param_groups for parameter in group['params']),
        phase_exit_approved=False,
    )
    print(json.dumps(outcome, indent=2))
