"""Run from the repository root; fictional corpus, no empirical acceptance."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_epoch_commit import make
from test_supported_trainer import mixed

with tempfile.TemporaryDirectory() as directory:
    vocabulary, corpus = mixed(Path(directory))
    trainer = make(corpus, vocabulary)
    original_scale = trainer.model.model_cfg.timing_scale_seconds
    trainer.model.model_cfg = replace(trainer.model.model_cfg, timing_scale_seconds=1.)
    try:
        trainer.fit(max_epochs=1)
        error = None
    except ValueError as failure:
        error = str(failure)
    print(json.dumps(dict(original_scale=original_scale,
                          changed_scale=trainer.model.model_cfg.timing_scale_seconds,
                          refusal=error, optimizer_steps=trainer.global_step,
                          exposure_records=len(trainer.optimizer_exposure),
                          phase_exit_approved=False), indent=2))
