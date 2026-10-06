"""Independent two-locus enumeration on heterogeneous negative synthetic scores."""
from fractions import Fraction
from itertools import product
import importlib.util
import json
from pathlib import Path
import random
import sys

here = Path(__file__).resolve().parent
repo = here.parents[2]
sys.path[:0] = [str(repo), str(repo / 'tests')]
from test_joint_spatial import candidate
from signtranslator.planning.joint_spatial import decode_joint_spatial
name = 'signtranslator.planning._before_forced_attachment'
spec = importlib.util.spec_from_file_location(name, here / 'joint-before-forced-attachment.py')
baseline = importlib.util.module_from_spec(spec)
sys.modules[name] = baseline
spec.loader.exec_module(baseline)
rows = []
for seed in range(20):
    rng = random.Random(20261006 + seed)
    n = 8
    refs = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            refs[i][j] = refs[j][i] = -rng.randint(1, 20)
    loci = [[rng.randint(-5, 5), rng.randint(-5, 5)] for _ in range(n)]
    c, alphabet = candidate(loci, refs)
    kwargs = dict(place=(True,) * n, referent_weight=Fraction(2, 3), locus_weight=Fraction(3, 2),
                  max_events=128, max_work=1000000)
    old = baseline.decode_joint_spatial(c, alphabet, **kwargs)
    new = decode_joint_spatial(c, alphabet, **kwargs)
    gains = sorted((Fraction(2, 3) * sum(refs[i][j] for i in range(n) for j in range(i + 1, n)
                                       if assignment[i] == assignment[j])
                    + Fraction(3, 2) * sum(loci[i][assignment[i]] for i in range(n))
                    for assignment in product(range(2), repeat=n)), reverse=True)
    assert old.best_gain == new.best_gain == gains[0]
    assert old.runner_up_gain == new.runner_up_gain == gains[1]
    assert old.status == new.status
    assert new.work <= old.work
    if new.status == 'unique_optimum_candidate':
        assert old.referents == new.referents and old.loci == new.loci
    rows.append(dict(seed=seed, references=refs, loci=loci, old_work=old.work, new_work=new.work,
                     status=new.status, best_gain=str(gains[0]), runner_up_gain=str(gains[1]),
                     exhaustive_assignments=2**n))
result = dict(schema_version=1, fixture='synthetic heterogeneous negative scores', cases=rows,
              improved_cases=sum(row['new_work'] < row['old_work'] for row in rows),
              old_total_work=sum(row['old_work'] for row in rows),
              new_total_work=sum(row['new_work'] for row in rows), phase_exit_approved=False)
(here / 'forced-attachment-oracle.json').write_text(json.dumps(result, indent=2) + '\n')
print({key: value for key, value in result.items() if key != 'cases'})
