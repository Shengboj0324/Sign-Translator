"""Scoped decode/validate/rebuild peak; excludes checkpoint deserialization and I/O."""
import json
from pathlib import Path
import sys
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from signtranslator.reproducibility import canonical_json_bytes
from signtranslator.training.exposure import validate_exposure, iter_validated_exposure
from signtranslator.training.exposure_codec import pack_exposure, unpack_exposure, iter_unpack_exposure
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
    records = []
    for step in range(100):
        row = [[j, j % 2] for j in range(200)]
        if unique:
            row[0][0] = 1000 + step
            row.sort()
        records.append(dict(step=step + 1, sample_ids=['fictional'], annotation_sha256=['a' * 64],
                            support={'labels': 1}, weights={'labels': 1.}, target_cells={
            'labels': dict(axes=['event'], class_count=2, examples=[row])}))
    envelope = pack_exposure(records)
    options = dict(supported=True, global_step=100, weights={'labels': 1.}, identities={'fictional': 'a' * 64})
    def old_path():
        logical = unpack_exposure(envelope)
        encoded = validate_exposure(logical, **options)
        return ExposureLedger(encoded)
    def new_path():
        return ExposureLedger(iter_validated_exposure(iter_unpack_exposure(envelope), **options))
    old, old_peak = measure(old_path)
    new, new_peak = measure(new_path)
    assert canonical_json_bytes(old.storage_envelope()) == canonical_json_bytes(new.storage_envelope())
    return dict(unique=unique, records=100, cells_per_record=200, identical_resident_content=True,
                old_peak_python_bytes=old_peak, new_peak_python_bytes=new_peak)


if __name__ == '__main__':
    result = dict(scope='Synthetic exposure decode/validate/rebuild only; excludes checkpoint/manifest parsing, history validation, model state, I/O, native allocation and RSS',
                  cases=[run(False), run(True)])
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
