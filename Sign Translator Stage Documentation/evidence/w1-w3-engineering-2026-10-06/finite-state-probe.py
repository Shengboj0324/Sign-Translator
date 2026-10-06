"""Run from repository root; actual AdamW overflow on a fictional scalar task."""
import json
from pathlib import Path
import sys
import tempfile
import torch

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_governed_trainer import MechanicalProbe, setup, loader
from signtranslator.training import Trainer
from signtranslator.config import TrainerConfig

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    corpus, _, _ = setup(root)
    model = MechanicalProbe().float()
    with torch.no_grad():
        model.offset.fill_(100.)
    trainer = Trainer(model, TrainerConfig(epochs=1, lr=1e37, weight_decay=1.), loader(corpus, 'train'))
    try:
        trainer.fit()
        refusal = None
    except FloatingPointError as failure:
        refusal = str(failure)
    print(json.dumps(dict(parameter_finite=bool(torch.isfinite(model.offset)),
                          returned_optimizer_steps=trainer.global_step,
                          committed=trainer._epoch_committed, completed_epochs=trainer.completed_epochs,
                          refusal=refusal, phase_exit_approved=False), indent=2))
