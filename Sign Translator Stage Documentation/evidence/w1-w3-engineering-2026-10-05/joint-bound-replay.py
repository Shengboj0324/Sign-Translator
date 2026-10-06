"""Replay recorded fictional score probes against retained baseline and current code.

Run from the repository's .venv; test fixture builders supply fictional typed
candidates, never a reviewed data source. Timings are intentionally not asserted.
"""
from dataclasses import asdict
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
repo = here.parents[2]
sys.path[:0] = [str(repo), str(repo / 'tests')]
from test_joint_spatial import candidate
from signtranslator.planning.joint_spatial import decode_joint_spatial

name = 'signtranslator.planning._retained_joint_baseline'
spec = importlib.util.spec_from_file_location(name, here / 'joint-bound-baseline.py')
baseline = importlib.util.module_from_spec(spec)
sys.modules[name] = baseline
spec.loader.exec_module(baseline)
for row in json.loads((here / 'joint-bound-profile.json').read_text())['records']:
    c, a = candidate(row['locus_logits'], row['reference_logits'])
    kwargs = dict(place=tuple(row['place']), referent_weight=Fraction(row['referent_weight']),
                  locus_weight=Fraction(row['locus_weight']), max_events=128, max_work=row['budget'])
    before = baseline.decode_joint_spatial(c, a, **kwargs)
    after = decode_joint_spatial(c, a, **kwargs)
    assert asdict(before) == asdict(after)
    assert after.status == row['status'] and after.work == row['work']
    print(row['events'], after.status, after.work)
