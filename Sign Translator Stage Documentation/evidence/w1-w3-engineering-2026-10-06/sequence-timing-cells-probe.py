"""Fictional current five-head-compatible ledger semantics, not ASL evidence."""
import json
import sys
import tempfile
from pathlib import Path
sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / 'tests')]
from test_supported_trainer import mixed
from test_epoch_commit import make
from signtranslator.planning.exposure_audit import audit_exposure_declarations

with tempfile.TemporaryDirectory() as directory:
    vocabulary, corpus = mixed(Path(directory))
    trainer = make(corpus, vocabulary)
    trainer.fit(max_epochs=1)
    records = trainer.optimizer_exposure
    summary = trainer.exposure_report().to_dict()['target_cell_exposure']
    audit = audit_exposure_declarations(trainer, vocabulary).to_dict()['target_cell_audit']
    timing_example = records[0]['target_cells']['event_timing']['examples'][0]
    original = [list(cell) for cell in timing_example]
    timing_example[0][-1] += .25
    trainer._optimizer_exposure = [json.dumps(record) for record in records]
    changed = audit_exposure_declarations(trainer, vocabulary).to_dict()['target_cell_audit']
    print(json.dumps(dict(fixture='fictional', summary=summary, consistent_audit=audit,
                         original_timing_cells=original, altered_clock_target_audit=changed,
                         phase_exit_approved=False), indent=2))
