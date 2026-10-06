"""Synthetic retained Python object size; excludes transient expansion and process RSS."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from signtranslator.reproducibility import canonical_json_bytes
from signtranslator.training.exposure_ledger import ExposureLedger


def retained_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(retained_bytes(k, seen) + retained_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        total += sum(retained_bytes(v, seen) for v in value)
    elif isinstance(value, ExposureLedger):
        total += retained_bytes(value._records, seen) + retained_bytes(value._definitions, seen)
    return total


def run(unique):
    records = []
    for step in range(100):
        row = [[i, i % 2] for i in range(200)]
        if unique:
            row[0][0] = 1000 + step
            row.sort()
        records.append(canonical_json_bytes(dict(step=step + 1, target_cells=dict(
            labels=dict(axes=['event'], class_count=2, examples=[row])))).decode('utf-8'))
    ledger = ExposureLedger(records)
    assert list(ledger) == records
    return dict(unique=unique, records=100, exact_roundtrip=True,
                expanded_retained_python_bytes=retained_bytes(records),
                pooled_retained_python_bytes=retained_bytes(ledger),
                pooled_encoded_payload_bytes=ledger.encoded_bytes)


if __name__ == '__main__':
    result = dict(scope='Fictional resident history only; no transient peak, RSS, throughput or corpus claim',
                  cases=[run(False), run(True)])
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
