"""Independent all-placed eight-event oracle: enumerate all 4**8 locus vectors."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import json
import random

here = Path(__file__).resolve().parent
rng = random.Random(20261005 + 8)
refs = [[0. if i == j else -1. for j in range(8)] for i in range(8)]
loci = [[float(rng.randint(-4, 4)) for _ in range(4)] for _ in range(8)]
inputs = dict(references=refs, loci=loci, place=(True,) * 8)
digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
best = second = None
for vector in product(range(4), repeat=8):
    # With all events placed, equal locus means equal referent under injectivity.
    # Common denominator 6 converts 2/3 and 3/2 to exact integer weights 4 and 9.
    score = -4 * sum(vector[i] == vector[j] for i in range(8) for j in range(i + 1, 8))
    score += 9 * sum(int(loci[i][vector[i]]) for i in range(8))
    if best is None or score > best:
        second, best = best, score
    elif second is None or score > second:
        second = score
observed = next(r for r in json.loads((here/'joint-workload-column-profile.json').read_text())['records']
                if r['events']==8 and r['pattern']=='negative' and r['placement']=='all' and r['budget']==10000)
assert observed['input_sha256'] == digest
assert observed['best_gain'] == str(Fraction(best, 6))
assert observed['runner_up_gain'] == str(Fraction(second, 6))
assert observed['status'] == 'ambiguous' and best == second
result = dict(configurations_enumerated=4**8, input_sha256=digest,
              best_gain=str(Fraction(best,6)), runner_up_gain=str(Fraction(second,6)),
              status='ambiguous', observed_work=observed['work'], phase_exit_approved=False)
(here/'column-bound-oracle.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
