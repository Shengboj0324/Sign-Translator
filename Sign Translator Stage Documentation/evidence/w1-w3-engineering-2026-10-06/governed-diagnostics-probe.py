"""Fictional demonstration of model/view-bound runner diagnostics."""
import json
from pathlib import Path
import sys
import tempfile

sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / 'tests')]
from test_governed_run import fixture, invoke
from signtranslator.governed_run import diagnose_governed_planner

with tempfile.TemporaryDirectory() as directory:
    run = invoke(fixture(Path(directory)))
    current = diagnose_governed_planner(run, view='train', model_state='current',
                                       sample_indices=(2, 0), permutation=(1, 0), seed=9, max_samples=2)
    best = diagnose_governed_planner(run, view='validation', model_state='best_validation',
                                    permutation=(0,), seed=9, max_samples=1)
    print(json.dumps(dict(fixture='fictional admitted records', current_subset=current.to_dict(),
                         best_validation=best.to_dict(), phase_exit_approved=False), indent=2))
