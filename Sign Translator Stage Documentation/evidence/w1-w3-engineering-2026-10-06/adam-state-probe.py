"""Replay finite-negative-moment save refusal on a fictional governed corpus."""
import json
import sys
import tempfile
from pathlib import Path

sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / 'tests')]
from test_supported_trainer import mixed
from test_epoch_commit import make

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    vocabulary, corpus = mixed(root)
    trainer = make(corpus, vocabulary)
    trainer.fit(max_epochs=1)
    next(iter(trainer.opt.state.values()))['exp_avg_sq'].fill_(-1)
    try:
        trainer.save(root / 'invalid.pt')
    except ValueError as error:
        print(json.dumps({'fixture': 'fictional', 'refused': True,
                          'error': str(error), 'checkpoint_written': (root / 'invalid.pt').exists()}))
    else:
        raise AssertionError('invalid AdamW moment accepted')
