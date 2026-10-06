"""Replay old/new summary equality and scoped peak Python allocation."""
import importlib.util
import json
from pathlib import Path
import sys
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from signtranslator.reproducibility import canonical_json_bytes
from signtranslator.training.exposure import summarize_exposure

path = Path(__file__).with_name('exposure-before-streaming.py')
name = 'signtranslator.training._retained_exposure_before_streaming'
spec = importlib.util.spec_from_file_location(name, path)
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


def run(count):
    encoded = [canonical_json_bytes(dict(step=i + 1, sample_ids=['fictional'],
        annotation_sha256=['a' * 64], support={'labels': 1}, weights={'labels': 1.},
        target_cells={'labels': dict(axes=['event'], class_count=2,
                                    examples=[[[j, j % 2] for j in range(200)]])})).decode()
        for i in range(count)]
    options = dict(global_step=count, completed_epochs=1, committed=True, weights={'labels': 1.},
                   identities={'fictional': 'a' * 64}, data_contract={}, implementation_identity={}, model_contract={})
    old, old_peak = measure(lambda: legacy.summarize_exposure([json.loads(x) for x in encoded], **options))
    new, new_peak = measure(lambda: summarize_exposure((json.loads(x) for x in encoded), **options))
    assert old.payload == new.payload
    return dict(records=count, cells_per_record=200, identical_payload=True,
                old_peak_python_bytes=old_peak, new_peak_python_bytes=new_peak)


if __name__ == '__main__':
    result = dict(scope='Synthetic report-only Python allocation; excludes resident input, native memory and RSS',
                  cases=[run(n) for n in (1, 20, 100)])
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
