"""Known-support relation diagnostics, conditional on exact generated labels."""
from dataclasses import dataclass
import hashlib
import json
import math

import torch

from .score_metrics import nonnegative_mean
from .relations import relation_targets
from .sequence_evaluation import evaluate_label_sequences
from .text_sequence import LabelSequenceCandidate
from .text_relations import RelationalSequenceCandidate
from .text_timing import TemporalSequenceCandidate
from .tensors import EDGE_TYPES


@dataclass(frozen=True)
class RelationEvaluationReport:
    payload: bytes

    def to_dict(self):
        return json.loads(self.payload)

    @property
    def sha256(self):
        return hashlib.sha256(self.payload).hexdigest()


def _scores(logit, positive):
    # Stable NLL and sigmoid error: avoid huge-term and near-one cancellation.
    tail = math.exp(-abs(logit))
    nll = max(-logit if positive else logit, 0.) + math.log1p(tail)
    correct_side = (logit >= 0) == positive
    error = tail / (1 + tail) if correct_side else 1 / (1 + tail)
    return nll, error ** 2


def _summary(positive, negative, unknown):
    def mean(values, index):
        return nonnegative_mean([pair[index] for pair in values])
    known = positive + negative
    return dict(positive=len(positive), negative=len(negative), unknown=unknown,
                known=len(known), nll=mean(known, 0), brier=mean(known, 1),
                positive_nll=mean(positive, 0), negative_nll=mean(negative, 0),
                positive_brier=mean(positive, 1), negative_brier=mean(negative, 1))


def evaluate_relation_sequences(candidates, annotations, vocabulary, *,
                                annotation_sha256, max_cells, max_relation_cells):
    """Evaluate observed relation support after free label generation.

    No thresholds are selected. Scores are conditional on exact serialized label
    agreement and observed target support, not full-graph accuracy or calibration.
    Counts refer to directed nonself cells, not independent statistical units.
    """
    if (not isinstance(candidates, tuple) or not candidates
            or type(max_relation_cells) is not int or not 1 <= max_relation_cells <= 10_000_000):
        raise ValueError('candidate tuple and explicit relation-cell budget required')
    lexical, snapshots, cells = [], [], 0
    for candidate in candidates:
        if not isinstance(candidate, RelationalSequenceCandidate) or candidate.relation_types != EDGE_TYPES:
            raise ValueError('typed relational candidate and exact relation codebook required')
        temporal = candidate.temporal
        if not isinstance(temporal, TemporalSequenceCandidate):
            raise ValueError("typed temporal candidate required")
        lexical.append(LabelSequenceCandidate(tuple(e.label for e in temporal.events), temporal.status,
                                              temporal.diagnostic_prefix, temporal.vocabulary_sha256))
        if temporal.status != 'terminated':
            if candidate.relation_logits is not None or candidate.relation_valid is not None:
                raise ValueError('failed generation must not contain relation scores')
            snapshots.append(None)
            continue
        events = len(temporal.events)
        cells += events * events * len(EDGE_TYPES)
        if cells > max_relation_cells:
            raise ValueError('relation scores exceed declared cell budget')
        scores, valid = candidate.relation_logits, candidate.relation_valid
        if (not isinstance(scores, torch.Tensor) or scores.dtype not in (torch.float32, torch.float64)
                or scores.shape != (events, events, len(EDGE_TYPES))
                or not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
                or valid.shape != (events, events)):
            raise ValueError('invalid relation score or mask contract')
        snapshot = scores.detach().to(device='cpu', dtype=torch.float64).clone()
        mask = valid.detach().cpu().clone()
        if not bool(torch.isfinite(snapshot).all()) or not torch.equal(mask, ~torch.eye(events, dtype=torch.bool)):
            raise ValueError('finite scores and complete directed nonself mask required')
        snapshots.append(snapshot.tolist())
    labels = evaluate_label_sequences(tuple(lexical), annotations, vocabulary,
                                      annotation_sha256=annotation_sha256, max_cells=max_cells).to_dict()
    targets = relation_targets(annotations, vocabulary)
    positives = [[] for _ in EDGE_TYPES]
    negatives = [[] for _ in EDGE_TYPES]
    unknowns = [0 for _ in EDGE_TYPES]
    rows, eligible = [], 0
    for row, (label_row, snapshot) in enumerate(zip(labels['rows'], snapshots)):
        summaries = None
        if label_row['exact_match']:
            eligible += 1
            count = len(snapshot)
            known = targets.known[row, :count, :count].tolist()
            positive = targets.positive[row, :count, :count].tolist()
            summaries = {}
            for k, relation in enumerate(EDGE_TYPES):
                pos, neg, unknown = [], [], 0
                for i in range(count):
                    for j in range(count):
                        if i == j:
                            continue
                        if not known[i][j][k]:
                            unknown += 1
                        else:
                            values = _scores(snapshot[i][j][k], positive[i][j][k])
                            (pos if positive[i][j][k] else neg).append(values)
                summaries[relation.value] = _summary(pos, neg, unknown)
                positives[k].extend(pos)
                negatives[k].extend(neg)
                unknowns[k] += unknown
        rows.append(dict(annotation_sha256=label_row['annotation_sha256'],
                         exact_label_sequence=label_row['exact_match'], relations=summaries))
    data = dict(schema_version=1, scope='exact-label-known-support-relation-diagnostic', rows=rows,
                vocabulary_sha256=vocabulary.lexicon.sha256, convention_sha256=vocabulary.convention.sha256,
                coverage=dict(numerator=eligible, denominator=len(candidates)),
                conditional_cell_weighted={relation.value: _summary(positives[k], negatives[k], unknowns[k])
                                           for k, relation in enumerate(EDGE_TYPES)},
                phase_exit_approved=False,
                limitations=['Scores condition on exact serialized labels and known relation targets.',
                             'Unknown cells are excluded, never counted as negative; self edges are outside the domain.',
                             'No decision threshold, calibrated graph, statistical independence or ASL accuracy is certified.'])
    return RelationEvaluationReport(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode())
