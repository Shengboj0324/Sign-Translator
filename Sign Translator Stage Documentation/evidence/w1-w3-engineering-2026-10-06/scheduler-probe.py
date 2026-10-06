"""Run from repository root; consistent-but-wrong learning-rate probe."""
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
    expected_rate = trainer.opt.param_groups[0]['lr']
    trainer.opt.param_groups[0]['lr'] = .9
    trainer.sched._last_lr = [.9]
    path = Path(directory) / 'incorrect.pt'
    try:
        trainer.save(path)
        refusal = None
    except ValueError as failure:
        refusal = str(failure)
    print(json.dumps(dict(step=trainer.global_step, configured_rate=expected_rate,
                          optimizer_rate=trainer.opt.param_groups[0]['lr'],
                          scheduler_rate=trainer.sched.get_last_lr()[0],
                          refusal=refusal, checkpoint_saved=path.exists(),
                          phase_exit_approved=False), indent=2))
