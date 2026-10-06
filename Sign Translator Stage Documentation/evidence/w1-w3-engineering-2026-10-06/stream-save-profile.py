"""Scoped pooled export allocation; no filesystem I/O or whole-save memory claim."""
import json
from pathlib import Path
import sys
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from signtranslator.reproducibility import canonical_json_bytes
from signtranslator.training.exposure import exposure_records
from signtranslator.training.exposure_codec import pack_exposure
from signtranslator.training.exposure_ledger import ExposureLedger


def measure(fn):
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def run(unique):
    encoded = []
    for step in range(100):
        row = [[j, j % 2] for j in range(200)]
        if unique:
            row[0][0] = 1000 + step
            row.sort()
        encoded.append(json.dumps(dict(step=step + 1, sample_ids=['fictional'], target_cells={
            'labels': dict(axes=['event'], class_count=2, examples=[row])})))
    ledger = ExposureLedger(encoded)
    old, old_peak = measure(lambda: pack_exposure(exposure_records(ledger)))
    new, new_peak = measure(ledger.storage_envelope)
    assert canonical_json_bytes(old) == canonical_json_bytes(new)
    return dict(unique=unique, records=100, cells_per_record=200,
                identical_canonical_envelope=True, old_peak_python_bytes=old_peak,
                new_peak_python_bytes=new_peak)


if __name__ == '__main__':
    result = dict(scope='Synthetic export-only traced Python allocation; excludes resident input, validation, I/O, native memory and RSS',
                  cases=[run(False), run(True)])
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
