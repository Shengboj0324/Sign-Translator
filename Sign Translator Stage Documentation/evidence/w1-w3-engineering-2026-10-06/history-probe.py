"""Run from repository root; incomplete committed-history probe on fictional data."""
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
    trainer.fit(max_epochs=1)
    trainer.history['train_total'].clear()
    path = Path(directory) / 'incomplete.pt'
    try:
        trainer.save(path)
        refusal = None
    except ValueError as failure:
        refusal = str(failure)
    print(json.dumps(dict(completed_epochs=trainer.completed_epochs,
                          train_total_rows=len(trainer.history['train_total']),
                          refusal=refusal, checkpoint_saved=path.exists(),
                          phase_exit_approved=False), indent=2))
