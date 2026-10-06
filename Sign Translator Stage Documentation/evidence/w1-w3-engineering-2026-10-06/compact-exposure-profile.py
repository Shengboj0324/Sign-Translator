"""Bounded synthetic storage controls; not an admitted corpus or throughput test."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from signtranslator.reproducibility import canonical_json_bytes
from signtranslator.training.exposure_codec import pack_exposure, unpack_exposure


def binary_size(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.tell()


def run(steps, unique):
    records = []
    for step in range(steps):
        cells = dict(axes=['event'], class_count=2,
                     examples=[[[i, (i + (step if unique else 0)) % 2] for i in range(200)]])
        if unique:
            cells['examples'][0][0][0] = 1000 + step
            cells['examples'][0].sort()
        records.append(dict(step=step + 1, sample_ids=['fictional'], target_cells=dict(labels=cells)))
    packed = pack_exposure(records)
    expanded_bytes = canonical_json_bytes(records)
    compact_bytes = canonical_json_bytes(packed)
    assert canonical_json_bytes(unpack_exposure(packed)) == expanded_bytes
    return dict(steps=steps, unique=unique, definitions=len(packed['target_cells']),
                expanded_json_bytes=len(expanded_bytes), compact_json_bytes=len(compact_bytes),
                expanded_torch_bytes=binary_size(deepcopy(records)), compact_torch_bytes=binary_size(packed),
                exact_roundtrip=True)


if __name__ == '__main__':
    result = dict(scope='Fictional bounded ledger storage only; no training, RSS or throughput claim',
                  cases=[run(n, unique) for n in (1, 20, 100) for unique in (False, True)])
    path = Path(__file__).with_suffix('.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
