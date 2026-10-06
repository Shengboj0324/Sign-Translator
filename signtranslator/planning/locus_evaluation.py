"""Known categorical locus diagnostics, not inferred physical placements."""
from dataclasses import dataclass
import hashlib
import json
import math

import torch

from .loci import LocusAlphabet, locus_targets
from .score_metrics import nonnegative_mean
from .sequence_evaluation import evaluate_label_sequences
from .text_sequence import LabelSequenceCandidate
from .text_loci import LocusSequenceCandidate
from .text_referents import ReferentSequenceCandidate
from .text_relations import RelationalSequenceCandidate
from .text_timing import TemporalSequenceCandidate


@dataclass(frozen=True)
class LocusEvaluationReport:
    payload: bytes

    def to_dict(self):
        return json.loads(self.payload)

    @property
    def sha256(self):
        return hashlib.sha256(self.payload).hexdigest()


def _categorical_scores(logits, target):
    anchor = max(range(len(logits)), key=logits.__getitem__)
    maximum = logits[anchor]
    weights = [math.exp(value - maximum) for value in logits]
    tail = math.fsum(value for index, value in enumerate(weights) if index != anchor)
    nll = (maximum - logits[target]) + math.log1p(tail)
    if not math.isfinite(nll):
        raise ValueError('categorical NLL exceeds representable range')
    other = [value / (1 + tail) for index, value in enumerate(weights) if index != target]
    error = math.fsum(other)  # avoids 1 - rounded-near-one target probability
    brier = 0. if error == 0 else error * (error * (1 + math.fsum((p / error) ** 2 for p in other)))
    return nll, brier


def _summarize(values):
    return dict(known=len(values), nll=nonnegative_mean([v[0] for v in values]),
                brier=nonnegative_mean([v[1] for v in values]))


def evaluate_locus_sequences(candidates, annotations, vocabulary, alphabet, *,
                             annotation_sha256, max_cells, max_locus_cells):
    """Score known loci conditional on exact serialized generated labels.

    Multiclass Brier is the sum over classes, with range [0, 2]; it is not
    divided by alphabet size. Missing annotation loci remain unknown.
    """
    if (not isinstance(candidates, tuple) or not candidates or not isinstance(alphabet, LocusAlphabet)
            or type(max_locus_cells) is not int or not 1 <= max_locus_cells <= 10_000_000):
        raise ValueError('candidate tuple, governed alphabet and explicit locus-cell budget required')
    lexical, snapshots, cells = [], [], 0
    for candidate in candidates:
        if (not isinstance(candidate, LocusSequenceCandidate)
                or not isinstance(candidate.referential, ReferentSequenceCandidate)
                or not isinstance(candidate.referential.relational, RelationalSequenceCandidate)
                or not isinstance(candidate.referential.relational.temporal, TemporalSequenceCandidate)):
            raise ValueError('typed locus and temporal candidates required')
        if (candidate.convention_sha256 != alphabet.convention.sha256
                or candidate.locus_identities != alphabet.identities):
            raise ValueError('candidate locus alphabet binding mismatch')
        temporal = candidate.referential.relational.temporal
        lexical.append(LabelSequenceCandidate(tuple(e.label for e in temporal.events), temporal.status,
                                              temporal.diagnostic_prefix, temporal.vocabulary_sha256))
        if temporal.status != 'terminated':
            if candidate.locus_logits is not None:
                raise ValueError('failed generation must not contain locus scores')
            snapshots.append(None)
            continue
        count, classes = len(temporal.events), len(alphabet.identities)
        cells += count * classes
        if cells > max_locus_cells:
            raise ValueError('locus scores exceed declared cell budget')
        scores = candidate.locus_logits
        if (not isinstance(scores, torch.Tensor) or scores.dtype not in (torch.float32, torch.float64)
                or scores.shape != (count, classes)):
            raise ValueError('invalid locus score contract')
        snapshot = scores.detach().to(device='cpu', dtype=torch.float64).clone()
        if not bool(torch.isfinite(snapshot).all()):
            raise ValueError('finite locus scores required')
        snapshots.append(snapshot.tolist())
    labels = evaluate_label_sequences(tuple(lexical), annotations, vocabulary,
                                      annotation_sha256=annotation_sha256, max_cells=max_cells).to_dict()
    targets = locus_targets(annotations, vocabulary, alphabet)
    rows, all_values, by_class, unknown, eligible, supported = [], [], [[] for _ in alphabet.identities], 0, 0, 0
    for row, (label_row, snapshot) in enumerate(zip(labels['rows'], snapshots)):
        summary = None
        if label_row['exact_match']:
            eligible += 1
            values, missing = [], 0
            for event, logits in enumerate(snapshot):
                if not bool(targets.known[row, event]):
                    missing += 1
                    continue
                target = int(targets.classes[row, event])
                score = _categorical_scores(logits, target)
                values.append(score)
                by_class[target].append(score)
            summary = dict(**_summarize(values), unknown=missing)
            supported += bool(values)
            all_values.extend(values)
            unknown += missing
        rows.append(dict(annotation_sha256=label_row['annotation_sha256'],
                         exact_label_sequence=label_row['exact_match'], loci=summary))
    data = dict(schema_version=1, scope='exact-label-known-locus-diagnostic', rows=rows,
                vocabulary_sha256=vocabulary.lexicon.sha256, convention_sha256=alphabet.convention.sha256,
                locus_identities=list(alphabet.identities),
                label_coverage=dict(numerator=eligible, denominator=len(candidates)),
                supported_example_coverage=dict(numerator=supported, denominator=len(candidates)),
                conditional_event_weighted=dict(**_summarize(all_values), unknown=unknown),
                per_target_class=[dict(identity=identity, **_summarize(values))
                                  for identity, values in zip(alphabet.identities, by_class)],
                phase_exit_approved=False,
                limitations=['Scores condition on exact serialized labels and known loci; missing loci are unknown.',
                             'Multiclass Brier sums squared class errors without dividing by alphabet size.',
                             'No placement/absence decision, spatial consistency, calibration or ASL accuracy is certified.'])
    return LocusEvaluationReport(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                            allow_nan=False).encode())
