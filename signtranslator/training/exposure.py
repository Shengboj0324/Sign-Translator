"""Checkpointed batch-level objective exposure, not a competence certificate."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from ..reproducibility import canonical_json_bytes
from .target_cells import parse_target_cells


def validate_exposure(records, *, supported, global_step, weights, identities):
    """Validate and copy records before a resumed model can be mutated."""
    if not isinstance(records, list):
        raise ValueError('optimizer exposure must be a list')
    if len(records) != (global_step if supported else 0):
        raise ValueError('optimizer exposure does not cover the recorded steps')
    fields = {'step', 'sample_ids', 'annotation_sha256', 'support', 'weights'}
    cell_contracts = {}
    for step, record in enumerate(records, 1):
        if (not isinstance(record, dict) or not fields <= set(record)
                or set(record) - fields - {'support_membership', 'target_cells'}):
            raise ValueError('invalid optimizer exposure fields')
        ids, hashes = record['sample_ids'], record['annotation_sha256']
        if (type(record['step']) is not int or record['step'] != step
                or not isinstance(ids, list) or not ids
                or any(not isinstance(x, str) or not x for x in ids)
                or len(set(ids)) != len(ids)
                or not isinstance(hashes, list) or len(hashes) != len(ids)
                or any(not isinstance(x, str) or re.fullmatch('[0-9a-f]{64}', x) is None
                       for x in hashes)):
            raise ValueError('invalid optimizer exposure identities or step')
        if any(identities.get(sample) != digest for sample, digest in zip(ids, hashes)):
            raise ValueError('optimizer exposure identities do not match admitted view')
        support = record['support']
        if (not isinstance(support, dict) or support.keys() != weights.keys()
                or any(type(n) is not int or not 0 <= n <= len(ids) for n in support.values())
                or not any(support.values()) or not isinstance(record['weights'], dict)
                or any(type(w) not in (int, float) for w in record['weights'].values())
                or record['weights'] != weights):
            raise ValueError('invalid optimizer exposure support or weights')
        if 'support_membership' in record:
            membership = record['support_membership']
            if not isinstance(membership, dict) or membership.keys() != weights.keys():
                raise ValueError('invalid optimizer exposure membership branches')
            for name, mask in membership.items():
                if mask is not None and (
                        not isinstance(mask, list) or len(mask) != len(ids)
                        or any(type(flag) is not bool for flag in mask)
                        or sum(mask) != support[name]):
                    raise ValueError('invalid optimizer exposure membership mask')
        if 'target_cells' in record:
            declarations = record['target_cells']
            if not isinstance(declarations, dict) or declarations.keys() != weights.keys():
                raise ValueError('invalid optimizer exposure target-cell branches')
            for name, declaration in declarations.items():
                if declaration is None:
                    continue
                cells = parse_target_cells(declaration)
                contract = {key: value for key, value in declaration.items() if key != 'examples'}
                if name in cell_contracts and cell_contracts[name] != contract:
                    raise ValueError('target-cell codebook changed within exposure history')
                cell_contracts[name] = contract
                mask = tuple(bool(row) for row in cells.examples)
                recorded_mask = record.get('support_membership', {}).get(name)
                if (len(mask) != len(ids) or sum(mask) != support[name]
                        or (recorded_mask is not None and tuple(recorded_mask) != mask)):
                    raise ValueError('optimizer target cells disagree with example support')
    # Store immutable strings; callers receive independent decoded copies.
    return [canonical_json_bytes(record).decode('utf-8') for record in records]


def exposure_records(encoded):
    return [json.loads(record) for record in encoded]


@dataclass(frozen=True)
class OptimizerExposureReport:
    """Immutable historical ledger summary; returned dictionaries are copies."""
    payload: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.payload).hexdigest()

    def to_dict(self):
        return json.loads(self.payload)


def summarize_exposure(records, *, global_step, completed_epochs, committed,
                       weights, identities, data_contract, implementation_identity, model_contract):
    if (type(global_step) is not int or global_step < 0
            or type(completed_epochs) is not int or completed_epochs < 0
            or type(committed) is not bool):
        raise ValueError('exposure summary requires exact nonnegative counters and boolean status')
    encoded = validate_exposure(records, supported=True, global_step=global_step,
                                weights=weights, identities=identities)
    # Work from the validated copy rather than caller-owned dictionaries.
    records = exposure_records(encoded)
    presentations = dict.fromkeys(identities, 0)
    branches = {name: {'supported_example_presentations': 0,
                       'steps_with_support': 0, 'steps_without_support': 0}
                for name in weights}
    sample_branches = {sample: {name: dict(supported=0, unsupported=0, unattributed=0)
                                for name in weights} for sample in identities}
    unattributed_support = dict.fromkeys(weights, 0)
    cell_summaries = {name: dict(axes=None, class_count=None, unit=None, value_kind=None, class_presentations={},
                                 target_presentations=0, recorded_example_presentations=0,
                                 unrecorded_example_presentations=0) for name in weights}
    for record in records:
        for sample in record['sample_ids']:
            presentations[sample] += 1
        for name, support in record['support'].items():
            summary = cell_summaries[name]
            declaration = record.get('target_cells', {}).get(name)
            if declaration is None:
                summary['unrecorded_example_presentations'] += len(record['sample_ids'])
            else:
                axes, classes = declaration['axes'], declaration.get('class_count')
                unit = declaration.get('unit')
                kind = 'categorical' if classes is not None else 'continuous'
                if summary['axes'] is None:
                    summary.update(axes=axes, class_count=classes, unit=unit, value_kind=kind)
                elif (summary['axes'] != axes or summary['class_count'] != classes
                      or summary['unit'] != unit or summary['value_kind'] != kind):
                    raise ValueError('target-cell codebook changed within exposure history')
                summary['recorded_example_presentations'] += len(declaration['examples'])
                for cells in declaration['examples']:
                    for cell in cells:
                        if classes is not None:
                            key = str(cell[-1])
                            summary['class_presentations'][key] = summary['class_presentations'].get(key, 0) + 1
                        summary['target_presentations'] += 1
            branches[name]['supported_example_presentations'] += support
            branches[name]['steps_with_support'] += int(support > 0)
            branches[name]['steps_without_support'] += int(support == 0)
            mask = record.get('support_membership', {}).get(name)
            if mask is None:
                unattributed_support[name] += support
            for index, sample in enumerate(record['sample_ids']):
                status = 'unattributed' if mask is None else ('supported' if mask[index] else 'unsupported')
                sample_branches[sample][name][status] += 1
    result = {
        'schema_version': 2,
        'evidence_kind': 'historical_returned_optimizer_calls',
        'committed_epoch_boundary': committed,
        'completed_epochs': completed_epochs,
        'recorded_optimizer_steps': global_step,
        'data_contract': data_contract,
        'implementation_identity': implementation_identity,
        'model_contract': model_contract,
        'ledger_sha256': hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
        'weights': dict(weights),
        'sample_presentations': sum(presentations.values()),
        'distinct_presented_samples': sum(n > 0 for n in presentations.values()),
        'admitted_view_samples': len(identities),
        'samples': [{'sample_id': sample, 'annotation_sha256': identities[sample],
                     'presentations': count, 'branch_presentations': sample_branches[sample]}
                    for sample, count in presentations.items()],
        'unattributed_supported_example_presentations': unattributed_support,
        'branches': branches,
        'target_cell_exposure': {'schema_version': 2, 'branches': cell_summaries},
        'phase_exit_approved': False,
        'limitations': [
            'Example membership is model-declared; absent historical masks remain unattributed.',
            'Target cells are loss-declared unweighted presentations, not gradient magnitude or independent units.',
            'Absent historical target cells remain unrecorded; no target cells are inferred from available annotations.',
            'Repeated presentations are not independent statistical units.',
            'Returned calls do not prove nonzero gradients or learned competence.',
            'Failed calls and arbitrary partial mutations are not certified.',
            'Historical bindings do not revalidate current source permission or bytes.',
            'The ledger hash identifies records, not model tensors or a checkpoint file.',
        ],
    }
    return OptimizerExposureReport(canonical_json_bytes(result))
