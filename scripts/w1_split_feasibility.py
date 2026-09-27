"""Audit local grouping feasibility without accepting QC or freezing a split."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signtranslator.data_engineering.splitting import group_samples, grouped_split, certify_no_group_leakage


@dataclass(frozen=True)
class GroupingIdentity:
    sample_id: str
    signer_id_hash: str  # published pseudonymous code; no personal identity inferred
    source_id: str


def file_identity(path):
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'size':len(data), 'sha256':hashlib.sha256(data).hexdigest()}


def audit(rows):
    samples = [GroupingIdentity(r['sample_id'], r['signer_id'], r['video_id']) for r in rows]
    groups = group_samples(samples)
    keys = sorted(groups)
    sizes = [len(groups[k]) for k in keys]
    # Exhaustive finite search is appropriate for this six-component population.
    # Integer objectives avoid float-dependent ties: max deviation, then L1.
    if len(keys) > 12:
        raise ValueError('exact audit supports at most twelve components; use a separately validated solver')
    n = len(samples)
    targets = (70*n, 15*n, 15*n)
    best = None
    for assignment in itertools.product(range(3), repeat=len(keys)):
        counts = tuple(sum(size for size, split in zip(sizes, assignment) if split == k)
                       for k in range(3))
        if min(counts) == 0:
            continue
        errors = tuple(abs(100*c-t) for c,t in zip(counts, targets))
        candidate = (max(errors), sum(errors), assignment, counts)
        if best is None or candidate < best:
            best = candidate
    greedy = grouped_split(samples, seed=0)
    assert certify_no_group_leakage(samples, greedy).certified
    return {'rows':n, 'component_count':len(keys),
            'components':[{'rows':len(groups[k]), 'signer_codes':list(k[0]),
                           'sources':len(k[1]),
                           'membership_sha256':hashlib.sha256(json.dumps(k).encode()).hexdigest()}
                          for k in keys],
            'requested_ratios':[.7,.15,.15],
            'greedy_counts':{s:sum(v==s for v in greedy.values()) for s in ('train','val','test')},
            'nonempty_three_way_split_feasible':best is not None,
            'exact_minimax_counts':list(best[3]) if best else None,
            'minimum_max_absolute_fraction_error':best[0]/(100*n) if best else None,
            'candidate_component_assignment':list(best[2]) if best else None,
            'assignment_order':['train','val','test'],
            'candidate_only':True, 'qc_accepted':False, 'final_split_frozen':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    certificate=args.evidence_root/'certificate.json'
    mapping=args.evidence_root/'how2sign_train_signers.csv'
    before=[file_identity(p) for p in (certificate,mapping)]
    cert=json.loads(certificate.read_text())
    if cert.get('certified') is not True or before[1]['sha256']!=cert['mapping']['sha256']:
        raise ValueError('signer certificate is absent or mapping hash does not match')
    with mapping.open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=cert['mapping']['rows']:
        raise ValueError('mapping count differs from certificate')
    statuses={s:sum(r['audit_status']==s for r in rows) for s in sorted({r['audit_status'] for r in rows})}
    if set(statuses)-{'missing_source','quality_warning','structural_failure','valid'}:
        raise ValueError('unrecognized audit dispositions')
    eligible=[r for r in rows if r['audit_status'] in ('valid','quality_warning')]
    result={'schema_version':1, 'input_files':before, 'status_counts':statuses,
            'all_recorded_rows':audit(rows), 'structurally_eligible_candidates':audit(eligible),
            'limitations':['quality warnings are not accepted QC',
                          'historical audit statuses were not refreshed by decoding media',
                          'published signer evidence is reused locally, not independently reauthenticated',
                          '2D source feasibility does not establish canonical multichannel 3D eligibility'],
            'implementation_files':[file_identity(Path(__file__)),
                 file_identity(Path(__file__).resolve().parents[1]/'signtranslator/data_engineering/splitting.py')]}
    assert before==[file_identity(p) for p in (certificate,mapping)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k]['exact_minimax_counts'] for k in
                      ('all_recorded_rows','structurally_eligible_candidates')}))


if __name__=='__main__':
    main()
