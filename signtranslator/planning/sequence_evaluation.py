"""Reference-bound serialization diagnostics, not linguistic ASL accuracy.

Unit-cost Levenshtein distance treats event-kind/lexical identity as one symbol.
Canonical event-ID serialization is not temporal order. Failed generation stays
unavailable for edit distance and remains in the overall exact-match denominator.
"""
from dataclasses import dataclass
import hashlib
import json

from .label_vocabulary import encode_governed_labels, SIRLabelEntry
from .text_sequence import LabelSequenceCandidate


@dataclass(frozen=True)
class SequenceEvaluationReport:
    payload: bytes

    def to_dict(self):
        return json.loads(self.payload)

    @property
    def sha256(self):
        return hashlib.sha256(self.payload).hexdigest()


def _edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def evaluate_label_sequences(candidates, annotations, vocabulary, *,
                             annotation_sha256: tuple[str, ...], max_cells: int):
    """Score explicitly row-bound candidates; preserve exact count denominators.

    Caller supplies prediction-to-reference identity, not inferred positional
    matching. This checks that binding, but cannot certify how predictions were
    obtained or that a held-out protocol was followed. No confidence intervals
    are inferred from examples that may share signer/recording dependencies.
    """
    if type(max_cells) is not int or not 1 <= max_cells <= 10_000_000:
        raise ValueError('explicit integer edit-cell budget between 1 and 10000000 required')
    if (not isinstance(candidates, tuple) or not isinstance(annotations, tuple)
            or not candidates or len(candidates) != len(annotations)):
        raise ValueError('nonempty equally sized candidate and annotation tuples required')
    targets = encode_governed_labels(annotations, vocabulary)
    hashes = tuple(a.content_sha256() for a in annotations)
    if not isinstance(annotation_sha256, tuple) or annotation_sha256 != hashes:
        raise ValueError('prediction/reference row binding mismatch')
    if len(set(hashes)) != len(hashes):
        raise ValueError('duplicate reviewed annotation in evaluation')
    lookup = {entry: index for index, entry in enumerate(vocabulary.entries)}
    prepared, cells = [], 0
    for candidate, target in zip(candidates, targets.tolist()):
        if not isinstance(candidate, LabelSequenceCandidate):
            raise ValueError('typed label candidate required')
        if candidate.vocabulary_sha256 != vocabulary.lexicon.sha256:
            raise ValueError('candidate vocabulary mismatch')
        if candidate.status not in ('terminated', 'empty_prediction', 'capacity_exceeded'):
            raise ValueError('unknown generation status')
        for entries in (candidate.labels, candidate.diagnostic_prefix):
            if not isinstance(entries, tuple):
                raise ValueError('immutable label tuples required')
            for entry in entries:
                if (not isinstance(entry, SIRLabelEntry) or type(entry.label_id) is not int
                        or entry not in lookup):
                    raise ValueError('candidate label outside governed vocabulary')
        terminated = candidate.status == 'terminated'
        if ((terminated and (not candidate.labels or candidate.diagnostic_prefix != candidate.labels))
                or (not terminated and candidate.labels)
                or (candidate.status == 'empty_prediction' and candidate.diagnostic_prefix)
                or (candidate.status == 'capacity_exceeded' and not candidate.diagnostic_prefix)):
            raise ValueError('candidate status and usable/diagnostic labels disagree')
        reference = tuple(i for i in target if i >= 0)
        prediction = tuple(lookup[e] for e in candidate.labels)
        if terminated:
            cells += len(reference) * len(prediction)
        if cells > max_cells:
            raise ValueError('edit-distance work exceeds declared budget')
        prepared.append((candidate, reference, prediction))
    rows = []
    edits = reference_count = terminated_count = exact_count = 0
    for sha, (candidate, reference, prediction) in zip(hashes, prepared):
        distance = None
        if candidate.status == 'terminated':
            distance = _edit_distance(reference, prediction)
            edits += distance
            reference_count += len(reference)
            terminated_count += 1
            exact_count += distance == 0
        rows.append(dict(annotation_sha256=sha, status=candidate.status,
                         reference_events=len(reference),
                         predicted_events=len(prediction) if distance is not None else None,
                         edit_distance=distance, exact_match=distance == 0))
    data = dict(schema_version=1, scope='canonical-serialization-label-diagnostic', rows=rows,
                vocabulary_sha256=vocabulary.lexicon.sha256,
                convention_sha256=vocabulary.convention.sha256,
                termination=dict(numerator=terminated_count, denominator=len(rows)),
                overall_exact_match=dict(numerator=exact_count, denominator=len(rows)),
                conditional_edit_rate=(None if not terminated_count else
                                       dict(numerator=edits, denominator=reference_count)),
                edit_cells=cells, phase_exit_approved=False,
                limitations=['Serialization agreement is not temporal, graph or linguistic accuracy.',
                             'Conditional edit rate excludes failed generations and can exceed one.',
                             'Caller-supplied row binding does not certify source-only or held-out generation.'])
    return SequenceEvaluationReport(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode())
