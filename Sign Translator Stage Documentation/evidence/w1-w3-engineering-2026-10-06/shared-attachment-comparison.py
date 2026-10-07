"""Bounded fictional joint-search workload matrix; no real-score latency claim.

Run with the repository .venv. Test builders supply typed fictional candidates.
The generated score systems are explicit controls, not model outputs or ASL data.
"""
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys
import time

here = Path(__file__).resolve().parent
repo = here.parents[2]
sys.path[:0] = [str(repo), str(repo / 'tests')]
from test_joint_spatial import candidate
from signtranslator.planning.joint_spatial import decode_joint_spatial
import importlib.util
name = 'signtranslator.planning._before_shared_attachment'
spec = importlib.util.spec_from_file_location(name, here / 'joint-before-shared-attachment.py')
baseline = importlib.util.module_from_spec(spec)
sys.modules[name] = baseline
spec.loader.exec_module(baseline)

records = []
for n in (8, 16, 32, 64):
    for pattern in ('positive', 'negative', 'mixed'):
        rng = random.Random(20261005 + n)
        refs = [[0.] * n for _ in range(n)]
        for i in range(n):
            for j in range(i):
                value = 1 if pattern == 'positive' else -1 if pattern == 'negative' else rng.randint(-5, 5)
                refs[i][j] = refs[j][i] = float(value)
        loci = ([[4., 0., 0., 0.] for _ in range(n)] if pattern == 'positive' else
                [[float(rng.randint(-4, 4)) for _ in range(4)] for _ in range(n)])
        c, alphabet = candidate(loci, refs)
        for placement in ('all', 'alternating', 'none'):
            place = tuple(placement == 'all' or (placement == 'alternating' and i % 2 == 0) for i in range(n))
            inputs = dict(references=refs, loci=loci, place=place)
            digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
            previous = None
            for budget in (1000, 10000):
                kwargs = dict(place=place, referent_weight=Fraction(2, 3), locus_weight=Fraction(3, 2),
                              max_events=128, max_work=budget)
                times = []
                baseline_times = []
                result = None
                for repeat in range(2):
                    functions = [('old', baseline.decode_joint_spatial), ('new', decode_joint_spatial)]
                    if repeat:
                        functions.reverse()
                    outputs = {}
                    for tag, function in functions:
                        started = time.perf_counter()
                        outputs[tag] = function(c, alphabet, **kwargs)
                        elapsed = time.perf_counter() - started
                        (times if tag == 'new' else baseline_times).append(elapsed)
                    assert asdict(outputs['old']) == asdict(outputs['new'])
                    observed = outputs['new']
                    if result is not None:
                        assert asdict(observed) == asdict(result)
                    result = observed
                assert result.work <= budget
                if result.status == 'search_exhausted':
                    assert result.referents is result.loci is result.best_gain is result.runner_up_gain is None
                elif result.status == 'unique_optimum_candidate':
                    gain = Fraction(2, 3) * sum((Fraction(refs[i][j]) for i in range(n) for j in range(i + 1, n)
                                                if result.referents[i] == result.referents[j]), Fraction(0))
                    gain += Fraction(3, 2) * sum((Fraction(loci[i][result.loci[i]]) for i in range(n) if place[i]), Fraction(0))
                    assert result.best_gain == gain
                    assert result.runner_up_gain is None or result.best_gain > result.runner_up_gain
                elif result.status == 'ambiguous':
                    assert result.best_gain == result.runner_up_gain
                    assert result.referents is result.loci is None
                else:
                    raise AssertionError(result.status)
                if previous is not None and previous.status != 'search_exhausted':
                    assert asdict(previous) == asdict(result)
                previous = result
                records.append(dict(events=n, pattern=pattern, placement=placement, budget=budget,
                                    input_sha256=digest, status=result.status, work=result.work,
                                    evaluated_partitions=result.evaluated_partitions, seconds=times, baseline_seconds=baseline_times,
                                    best_gain=None if result.best_gain is None else str(result.best_gain),
                                    runner_up_gain=None if result.runner_up_gain is None else str(result.runner_up_gain)))
        print(n, pattern, 'complete', flush=True)
report = dict(schema_version=1, scope='fictional-joint-search-workload-matrix', records=records,
              weights=dict(referent='2/3', locus='3/2'), alphabet_size=4,
              implementation_sha256=hashlib.sha256((repo / 'signtranslator/planning/joint_spatial.py').read_bytes()).hexdigest(),
              partition_bounds_sha256=hashlib.sha256((repo / 'signtranslator/planning/partition_bounds.py').read_bytes()).hexdigest(),
              assignment_bounds_sha256=hashlib.sha256((repo / 'signtranslator/planning/assignment_bounds.py').read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              phase_exit_approved=False,
              complete_result_equality=True,
              baseline_sha256=hashlib.sha256((here / 'joint-before-shared-attachment.py').read_bytes()).hexdigest(),
              limitations=['Fixed synthetic systems, not independent draws from a real model distribution.',
                           'Two runs per configuration are descriptive timings, not latency percentiles.',
                           'Work budgets bound node/probe counts, not elapsed time; target hardware remains absent.',
                           'Score reconstruction and budget monotonicity are consistency checks, not independent exhaustive optimality proofs.'])
(here / 'shared-attachment-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print('saved', len(records), 'configurations', flush=True)
