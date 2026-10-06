"""Run from repository root; fictional-corpus optimizer settings probe."""
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
    trainer.opt.param_groups[0]['weight_decay'] = .9
    errors = {}
    path = Path(directory) / 'changed.pt'
    for name, operation in [('train', lambda: trainer.fit(max_epochs=1)),
                            ('save', lambda: trainer.save(path))]:
        try:
            operation()
            errors[name] = None
        except ValueError as failure:
            errors[name] = str(failure)
    print(json.dumps(dict(recorded_weight_decay=trainer.cfg.weight_decay,
                          live_weight_decay=trainer.opt.param_groups[0]['weight_decay'],
                          errors=errors, optimizer_steps=trainer.global_step,
                          checkpoint_saved=path.exists(), phase_exit_approved=False), indent=2))
