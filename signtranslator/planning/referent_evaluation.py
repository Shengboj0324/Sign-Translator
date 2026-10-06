"""Known unordered referent-pair diagnostics, not partition accuracy."""
from dataclasses import dataclass
import hashlib
import json

import torch

from .referents import referent_targets
from .relation_evaluation import _scores, _summary
from .sequence_evaluation import evaluate_label_sequences
from .text_sequence import LabelSequenceCandidate
from .text_referents import ReferentSequenceCandidate
from .text_relations import RelationalSequenceCandidate
from .text_timing import TemporalSequenceCandidate


@dataclass(frozen=True)
class ReferentEvaluationReport:
    payload: bytes

    def to_dict(self):
        return json.loads(self.payload)

    @property
    def sha256(self):
        return hashlib.sha256(self.payload).hexdigest()


def evaluate_referent_sequences(candidates, annotations, vocabulary, *,
                                annotation_sha256, max_cells, max_pair_cells):
    """Score known equality once per unordered pair after exact label agreement.

    Missing IDs are unknown. Positive/negative summaries mean equal/unequal IDs,
    not the presence/absence of a directed COREF edge. No partition is inferred.
    """
    if (not isinstance(candidates, tuple) or not candidates
            or type(max_pair_cells) is not int or not 1 <= max_pair_cells <= 10_000_000):
        raise ValueError('candidate tuple and explicit pair-cell budget required')
    lexical, snapshots, cells = [], [], 0
    for candidate in candidates:
        if (not isinstance(candidate, ReferentSequenceCandidate)
                or not isinstance(candidate.relational, RelationalSequenceCandidate)
                or not isinstance(candidate.relational.temporal, TemporalSequenceCandidate)):
            raise ValueError('typed referent and temporal candidates required')
        temporal = candidate.relational.temporal
        lexical.append(LabelSequenceCandidate(tuple(e.label for e in temporal.events), temporal.status,
                                              temporal.diagnostic_prefix, temporal.vocabulary_sha256))
        if temporal.status != 'terminated':
            if candidate.equality_logits is not None or candidate.pair_valid is not None:
                raise ValueError('failed generation must not contain referent scores')
            snapshots.append(None)
            continue
        count = len(temporal.events)
        cells += count * count
        if cells > max_pair_cells:
            raise ValueError('referent scores exceed declared pair-cell budget')
        scores, valid = candidate.equality_logits, candidate.pair_valid
        if (not isinstance(scores, torch.Tensor) or scores.dtype not in (torch.float32, torch.float64)
                or scores.shape != (count, count)
                or not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
                or valid.shape != (count, count)):
            raise ValueError('invalid referent score or mask contract')
        snapshot = scores.detach().to(device='cpu', dtype=torch.float64).clone()
        mask = valid.detach().cpu().clone()
        if (not bool(torch.isfinite(snapshot).all()) or not torch.equal(snapshot, snapshot.T)
                or not torch.equal(mask, ~torch.eye(count, dtype=torch.bool))):
            raise ValueError('finite symmetric scores and complete nonself mask required')
        snapshots.append(snapshot.tolist())
    labels = evaluate_label_sequences(tuple(lexical), annotations, vocabulary,
                                      annotation_sha256=annotation_sha256, max_cells=max_cells).to_dict()
    targets = referent_targets(annotations, vocabulary)
    positives, negatives, unknowns, supported, eligible, rows = [], [], 0, 0, 0, []
    for row, (label_row, snapshot) in enumerate(zip(labels['rows'], snapshots)):
        summary = None
        if label_row['exact_match']:
            eligible += 1
            count = len(snapshot)
            known = targets.known[row, :count, :count].tolist()
            equal = targets.equal[row, :count, :count].tolist()
            pos, neg, unknown = [], [], 0
            for i in range(count):
                for j in range(i + 1, count):
                    if not known[i][j]:
                        unknown += 1
                    else:
                        values = _scores(snapshot[i][j], equal[i][j])
                        (pos if equal[i][j] else neg).append(values)
            summary = _summary(pos, neg, unknown)
            supported += bool(pos or neg)
            positives.extend(pos)
            negatives.extend(neg)
            unknowns += unknown
        rows.append(dict(annotation_sha256=label_row['annotation_sha256'],
                         exact_label_sequence=label_row['exact_match'], referent_equality=summary))
    data = dict(schema_version=1, scope='exact-label-known-referent-equality-diagnostic', rows=rows,
                vocabulary_sha256=vocabulary.lexicon.sha256, convention_sha256=vocabulary.convention.sha256,
                label_coverage=dict(numerator=eligible, denominator=len(candidates)),
                supported_example_coverage=dict(numerator=supported, denominator=len(candidates)),
                conditional_pair_weighted=_summary(positives, negatives, unknowns),
                phase_exit_approved=False,
                limitations=['Positive/negative referent pairs mean known equal/unequal IDs, not COREF edges.',
                             'Each unordered pair is counted once; missing IDs stay unknown.',
                             'Conditional pair scores do not certify a transitive partition, calibration, independence or ASL accuracy.'])
    return ReferentEvaluationReport(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode())
