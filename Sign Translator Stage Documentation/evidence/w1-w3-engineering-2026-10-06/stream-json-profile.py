"""Canonical encoding/comparison allocation only; no filesystem or whole-checkpoint claim."""
import hashlib
import json
from pathlib import Path
import sys
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from signtranslator.reproducibility import canonical_json_bytes, canonical_json_equal, iter_canonical_json_chunks


def measure(fn):
    tracemalloc.start()
    try:
        value = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, peak


def digest_stream(value):
    digest, size = hashlib.sha256(), 0
    for chunk in iter_canonical_json_chunks(value):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def run(name, value):
    other = json.loads(canonical_json_bytes(value))
    def old_digest():
        raw = canonical_json_bytes(value)
        return hashlib.sha256(raw).hexdigest(), len(raw)
    old, old_peak = measure(old_digest)
    new, new_peak = measure(lambda: digest_stream(value))
    assert old == new
    a, equality_old_peak = measure(lambda: canonical_json_bytes(value) == canonical_json_bytes(other))
    b, equality_new_peak = measure(lambda: canonical_json_equal(value, other))
    assert a and b
    return dict(name=name, bytes=old[1], identical_sha256_and_length=True,
                encoding_old_peak=old_peak, encoding_new_peak=new_peak,
                comparison_old_peak=equality_old_peak, comparison_new_peak=equality_new_peak)


if __name__ == '__main__':
    cases = [('many_small_cells', {'records': [dict(step=i, cells=[[j, j % 2] for j in range(200)]) for i in range(1000)]}),
             ('large_unicode_scalar', {'text': '样' * 250000})]
    result = dict(scope='Synthetic encoding/comparison peak Python allocation; inputs, I/O, native memory and RSS excluded; large tokens remain materialized',
                  cases=[run(name, value) for name, value in cases])
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
