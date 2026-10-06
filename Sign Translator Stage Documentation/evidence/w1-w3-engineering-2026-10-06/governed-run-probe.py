"""Replay the public governed runner on explicit fictional test evidence."""
import json
from pathlib import Path
import sys
import tempfile

sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / 'tests')]
from test_governed_run import fixture, invoke

with tempfile.TemporaryDirectory() as directory:
    result = invoke(fixture(Path(directory)))
    trainer = result.trainer
    report = trainer.exposure_report().to_dict()
    audit = result.exposure_audit.to_dict()
    print(json.dumps(dict(fixture='fictional admitted test records',
                         completed_epochs=trainer.completed_epochs, global_step=trainer.global_step,
                         batch_populations=[len(row['sample_ids']) for row in trainer.optimizer_exposure],
                         training_samples=result.training_support.to_dict()['sample_count'],
                         validation_samples=result.validation_support.to_dict()['sample_count'],
                         target_cells=report['target_cell_exposure'],
                         declaration_audit=audit['target_cell_audit'],
                         phase_exit_approved=result.phase_exit_approved), indent=2))
