"""Compare retained identical fictional workloads; timings are descriptive only."""
import hashlib
import json
from collections import Counter
from pathlib import Path
p = Path(__file__).resolve().parent
old = json.loads((p / 'joint-workload-bound-order-baseline.json').read_text())['records']
new = json.loads((p / 'joint-workload-bound-order-profile.json').read_text())['records']
assert len(old) == len(new) == 72
resolved = []
for a, b in zip(old, new):
    for key in ('events', 'pattern', 'placement', 'budget', 'input_sha256'):
        assert a[key] == b[key]
    if a['status'] != 'search_exhausted':
        assert b['status'] != 'search_exhausted'
        for key in ('status', 'best_gain', 'runner_up_gain'):
            assert a[key] == b[key]
    elif b['status'] != 'search_exhausted':
        resolved.append({key: b[key] for key in ('events', 'pattern', 'placement', 'budget', 'status')})
report = dict(schema_version=1, scope='72 fixed synthetic configurations, two calls each',
    old_work=sum(r['work'] for r in old), new_work=sum(r['work'] for r in new),
    old_seconds=sum(sum(r['seconds']) for r in old), new_seconds=sum(sum(r['seconds']) for r in new),
    additional_completed=resolved, previously_completed_regressions=0,
    statuses={str(budget): {name: dict(Counter(r['status'] for r in rows if r['budget'] == budget))
                          for name, rows in [('old', old), ('new', new)]} for budget in (1000, 10000)},
    implementation_sha256=hashlib.sha256((p.parents[2] / 'signtranslator/planning/joint_spatial.py').read_bytes()).hexdigest(),
    phase_exit_approved=False,
    limitations=['Descriptive sequential timings, not controlled latency benchmarks.',
                 'Precomputed bounds cost CPU outside the counted node/probe budget.',
                 'Ordering can worsen other inputs; no universal work or time improvement.',
                 'Synthetic optimality and search completion do not establish ASL quality.'])
(p / 'bound-order-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
