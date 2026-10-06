"""Independent exhaustive locus-vector oracle for all-placed mixed-sign cases."""
from dataclasses import asdict
from fractions import Fraction
import importlib.util
from itertools import product
import json
from pathlib import Path
import random
import sys

here = Path(__file__).resolve().parent
repo = here.parents[2]
sys.path[:0] = [str(repo), str(repo / 'tests')]
from test_joint_spatial import candidate
from signtranslator.planning.joint_spatial import decode_joint_spatial

name = 'signtranslator.planning._before_mixed_capacity'
spec = importlib.util.spec_from_file_location(name, here / 'joint-before-mixed-capacity.py')
before = importlib.util.module_from_spec(spec)
sys.modules[name] = before
spec.loader.exec_module(before)
records = []
for n in (8, 10, 12):
    rng = random.Random(61006 + n)
    refs = [[0.] * n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            refs[i][j] = refs[j][i] = -float(rng.randint(1, 15))
    refs[0][1] = refs[1][0] = 3.
    refs[1][2] = refs[2][1] = 0.
    loci = [[float(rng.randint(-6, 6)) for _ in range(2)] for _ in range(n)]
    c, alphabet = candidate(loci, refs)
    kwargs = dict(place=(True,) * n, referent_weight=Fraction(2, 3),
                  locus_weight=Fraction(3, 2), max_events=128, max_work=1000000)
    old = before.decode_joint_spatial(c, alphabet, **kwargs)
    new = decode_joint_spatial(c, alphabet, **kwargs)
    # Injective loci make reference equality equivalent to locus equality when
    # every event is placed. Enumerate all 2**n complete assignments directly.
    gains = []
    for assigned in product(range(2), repeat=n):
        ref = sum(Fraction(refs[i][j]) for i in range(n) for j in range(i + 1, n)
                  if assigned[i] == assigned[j])
        spatial = sum(Fraction(loci[i][assigned[i]]) for i in range(n))
        gains.append(Fraction(2, 3) * ref + Fraction(3, 2) * spatial)
    expected = sorted(gains, reverse=True)[:2]
    assert [new.best_gain, new.runner_up_gain] == expected
    assert old.best_gain == new.best_gain and old.runner_up_gain == new.runner_up_gain
    assert new.work <= old.work
    assert new.status == old.status
    if expected[0] == expected[1]:
        assert new.referents is None and new.loci is None
    records.append(dict(events=n, assignments=len(gains), best=str(expected[0]), runner_up=str(expected[1]),
                        old_work=old.work, new_work=new.work, status=new.status,
                        references=refs, loci=loci))
(here / 'mixed-capacity-oracle.json').write_text(json.dumps(dict(records=records,
    scope='Fictional all-placed mixed-sign controls; exact exhaustive locus-vector oracle.',
    phase_exit_approved=False), indent=2) + '\n')
print(json.dumps([{k: v for k, v in r.items() if k not in ('references', 'loci')} for r in records], indent=2))
