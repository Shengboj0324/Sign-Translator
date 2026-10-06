"""Fictional index allocation and full-audit parity against retained implementation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import tracemalloc

root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(root), str(root / 'tests')]
from signtranslator.planning.exposure_audit import _index_declarations, _cell_signature, audit_exposure_declarations
from test_exposure_audit import fixture
from test_epoch_commit import make

name = 'signtranslator.planning._retained_audit_before_interning'
spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('exposure-audit-before-interning.py'))
legacy = importlib.util.module_from_spec(spec)
sys.modules[name] = legacy
spec.loader.exec_module(legacy)


def measure(fn):
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def old_index(encoded):
    declarations = {'fictional': []}
    for record in [json.loads(x) for x in encoded]:
        for row, sample in enumerate(record['sample_ids']):
            declarations[sample].append((record, row))
    return declarations


def profile(unique):
    encoded = []
    for step in range(100):
        row = [[j, j % 2] for j in range(200)]
        if unique:
            row[0][0] = 1000 + step
            row.sort()
        encoded.append(json.dumps(dict(step=step + 1, sample_ids=['fictional'], target_cells={
            'labels': dict(axes=['event'], class_count=2, examples=[row])})))
    old, old_peak = measure(lambda: old_index(encoded))
    new, new_peak = measure(lambda: _index_declarations(encoded, ['fictional'], ['labels']))
    expected = [(record['step'], (_cell_signature(record['target_cells']['labels'], row),))
                for record, row in old['fictional']]
    assert expected == new['fictional']
    return dict(unique=unique, records=100, cells_per_record=200,
                old_peak_python_bytes=old_peak, new_peak_python_bytes=new_peak, equal_index_content=True)


def parity():
    with tempfile.TemporaryDirectory() as directory:
        vocabulary, corpus, alphabet = fixture(Path(directory))
        trainer = make(corpus, vocabulary); trainer.fit()
        original = trainer.optimizer_exposure
        results = []
        for mode in ('current', 'historical', 'contradictory_clock'):
            records = json.loads(json.dumps(original))
            if mode == 'historical':
                for record in records:
                    record.pop('target_cells')
                    record.pop('support_membership')
            elif mode == 'contradictory_clock':
                records[0]['target_cells']['event_timing']['examples'][0][0][-1] += 0.125
            trainer._optimizer_exposure = [json.dumps(r) for r in records]
            old = legacy.audit_exposure_declarations(trainer, vocabulary, alphabet)
            new = audit_exposure_declarations(trainer, vocabulary, alphabet)
            assert old.payload == new.payload
            results.append(dict(mode=mode, identical_payload=True))
        return results


if __name__ == '__main__':
    result = dict(scope='Synthetic index-only traced Python allocation; not full-audit peak, native memory or RSS',
                  profiles=[profile(False), profile(True)], audit_parity=parity())
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
