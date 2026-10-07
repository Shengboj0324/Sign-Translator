"""Type-specific historical relation declarations, never a competence gate."""
import hashlib
import json
import re

from ..training.exposure import iter_validated_exposure
from .tensors import EDGE_TYPES


def summarize_relation_exposure(records, *, global_step, weights, identities,
                                max_events, expected_ledger_sha256):
    """Project an exact selected ledger onto directed relation type/polarity.

    Validate every record and bind the projection to the same canonical ledger
    hash as the selected exposure summary. Repetitions count as presentations;
    distinct sample IDs are not assumed to be independent signers or recordings.
    Missing declarations and explicit empty declarations remain distinct.
    """
    if (type(global_step) is not int or global_step < 0
            or type(max_events) is not int or max_events <= 0
            or not isinstance(weights, dict) or 'sir_relations' not in weights
            or type(expected_ledger_sha256) is not str
            or re.fullmatch('[0-9a-f]{64}', expected_ledger_sha256) is None):
        raise ValueError('bounded relation domain and selected ledger identity required')
    counts = [[0, 0] for _ in EDGE_TYPES]
    samples = [[set(), set()] for _ in EDGE_TYPES]
    recorded = empty = unrecorded = 0
    digest = hashlib.sha256(b'[')
    validated = iter_validated_exposure(records, supported=True, global_step=global_step,
                                        weights=weights, identities=identities)
    for index, payload in enumerate(validated):
        if index:
            digest.update(b',')
        digest.update(payload.encode('utf-8'))
        row = json.loads(payload)
        declaration = row.get('target_cells', {}).get('sir_relations')
        if declaration is None:
            unrecorded += len(row['sample_ids'])
            continue
        if (declaration['axes'] != ['source_event', 'target_event', 'relation_type']
                or declaration.get('class_count') != 2):
            raise ValueError('canonical binary directed relation target contract required')
        recorded += len(row['sample_ids'])
        for sample, cells in zip(row['sample_ids'], declaration['examples'], strict=True):
            empty += int(not cells)
            for source, target, relation, polarity in cells:
                if source == target or max(source, target) >= max_events or relation >= len(EDGE_TYPES):
                    raise ValueError('relation target cell outside canonical nonself domain')
                counts[relation][polarity] += 1
                samples[relation][polarity].add(sample)
    digest.update(b']')
    if digest.hexdigest() != expected_ledger_sha256:
        raise ValueError('relation projection does not match selected exposure ledger')
    return dict(schema_version=1, evidence_kind='selected_historical_relation_target_declarations',
        ledger_sha256=digest.hexdigest(), recorded_optimizer_steps=global_step,
        recorded_example_presentations=recorded, empty_recorded_example_presentations=empty,
        unrecorded_example_presentations=unrecorded,
        relation_types=[dict(relation_type=kind.value,
            negative_cell_presentations=counts[index][0], positive_cell_presentations=counts[index][1],
            distinct_negative_samples=len(samples[index][0]), distinct_positive_samples=len(samples[index][1]),
            recorded_polarities=[polarity for polarity, count in zip(('negative', 'positive'), counts[index]) if count])
            for index, kind in enumerate(EDGE_TYPES)],
        phase_exit_approved=False,
        limitations=['Missing target declarations are not reconstructed from annotations.',
                    'No recorded cells does not imply a trained negative class.',
                    'Distinct sample IDs are not necessarily independent statistical units.',
                    'Presentations do not certify nonzero gradients, learned competence or calibration.',
                    'Unknown-cell totals cannot be reconstructed from the selected-cell ledger.'])
