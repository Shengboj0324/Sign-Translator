"""Compare table construction allocations and exact query results at the cap."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import tracemalloc

here = Path(__file__).resolve().parent
repo = here.parents[2]
sys.path.insert(0, str(repo))
from signtranslator.planning.partition_bounds import placed_pair_penalty_tables

spec = importlib.util.spec_from_file_location('_old_penalty_tables', here / 'partition-bounds-before-persistent.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def measure(builder, n, pair, place):
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    result = builder(n, pair, place)
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, dict(retained_bytes=current, peak_bytes=peak, seconds=elapsed)


records = []
for pattern in ('uniform_negative', 'distinct_negative', 'mixed'):
    n = 128
    pair = {(i, j): (-1 if pattern == 'uniform_negative' else
                      -(i*n+j+1) if pattern == 'distinct_negative' else
                      (0 if (i+j) % 7 == 0 else -(i*n+j+1)))
            for i in range(n) for j in range(i+1, n)}
    place = (True,) * n
    baseline, before = measure(old.placed_pair_penalty_tables, n, pair, place)
    current, after = measure(placed_pair_penalty_tables, n, pair, place)
    free, costs = baseline
    checked = 0
    for prefix in range(n+1):
        maximum = free[prefix] + len(costs[prefix]) - 1
        for forced in sorted({0, min(1, maximum), maximum//2, maximum}):
            assert current.penalty(prefix, forced) == costs[prefix][max(0, forced-free[prefix])]
            checked += 1
    assert after['peak_bytes'] < before['peak_bytes']
    records.append(dict(events=n, pattern=pattern, checked_queries=checked,
                        old_prefix_sum_entries=sum(map(len, costs)),
                        persistent_nodes=len(current.nodes), before=before, after=after))
report = dict(records=records, scope='Python tracemalloc construction allocations, input pair dictionary excluded.',
              implementation_sha256=hashlib.sha256((repo/'signtranslator/planning/partition_bounds.py').read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              phase_exit_approved=False,
              limitations=['Single constructions on local runtime; not RSS, end-to-end decoder memory or latency percentiles.',
                           'Queries now traverse a logarithmic tree rather than constant-time array lookup.'])
(here/'persistent-capacity-profile.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
