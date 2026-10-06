"""CPU capture/serialization controls; not full-training or accelerator throughput."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

import torch

here = Path(__file__).resolve().parent
repo = here.parents[2]
sys.path.insert(0, str(repo))
from signtranslator.training import target_cells as current
from signtranslator.reproducibility import canonical_json_bytes
name = 'signtranslator.training._before_vector_cells'
spec = importlib.util.spec_from_file_location(name, here / 'target-cells-before-vector.py')
baseline = importlib.util.module_from_spec(spec)
sys.modules[name] = baseline
spec.loader.exec_module(baseline)
torch.set_num_threads(1)
rows = []
for batch in (1, 4):
    for events in (16, 64, 128):
        for domain in ('binary', 'integer', 'continuous'):
            for density in ('dense', 'sparse'):
                shape = (batch, events, 2) if domain == 'continuous' else (batch, events, events, 5)
                index = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
                known = torch.ones(shape, dtype=torch.bool) if density == 'dense' else index.remainder(7) == 0
                if domain != 'continuous':
                    known &= (~torch.eye(events, dtype=torch.bool))[None, :, :, None]
                values = (index.double() / 7 - 4 if domain == 'continuous' else
                          index.remainder(2).bool() if domain == 'binary' else index.remainder(3))
                kwargs = (dict(axes=('event', 'endpoint'), unit='seconds') if domain == 'continuous' else
                          dict(axes=('source', 'target', 'type'), class_count=2 if domain == 'binary' else 3))
                function = 'selected_continuous_target_cells' if domain == 'continuous' else 'selected_target_cells'
                functions = [getattr(baseline, function), getattr(current, function)]
                times = [[], []]
                expected = functions[0](known, values, **kwargs).to_dict()
                for repeat in range(3):
                    for which in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        start = time.perf_counter()
                        result = functions[which](known, values, **kwargs)
                        times[which].append(time.perf_counter() - start)
                        assert result.to_dict() == expected
                start = time.perf_counter()
                payload = canonical_json_bytes(result.to_dict())
                serialization_seconds = time.perf_counter() - start
                rows.append(dict(batch=batch, events=events, domain=domain, density=density,
                                 selected_cells=int(known.sum()), old_seconds=times[0], new_seconds=times[1],
                                 median_speed_ratio=statistics.median(times[0]) / statistics.median(times[1]),
                                 canonical_payload_bytes=len(payload), serialization_seconds=serialization_seconds,
                                 payload_sha256=hashlib.sha256(payload).hexdigest()))
                if batch == 4 and events == 128 and domain == 'binary' and density == 'dense':
                    peaks = []
                    for fn in functions:
                        gc.collect(); tracemalloc.start()
                        captured = fn(known, values, **kwargs)
                        peaks.append(tracemalloc.get_traced_memory()[1])
                        tracemalloc.stop(); del captured
                    rows[-1]['python_capture_peak_bytes_old_new'] = peaks
        print(batch, events, 'done', flush=True)
result = dict(schema_version=1, scope='synthetic CPU target-cell extraction', cases=rows,
              torch_threads=1, repeats=3, phase_exit_approved=False,
              limitations=['Timings cover extraction/validation only; JSON serialization is reported separately.',
                           'Descriptive local controls, not latency percentiles or full-model throughput.',
                           'tracemalloc excludes tensor/native allocations and is not process RSS.',
                           'Exact record format is unchanged; storage still grows with declared cells per step.'])
(here / 'vector-cells-profile.json').write_text(json.dumps(result, indent=2) + '\n')
print('saved', len(rows), 'cases')
