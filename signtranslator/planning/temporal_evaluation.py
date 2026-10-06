"""Conditional timing diagnostics in canonical event serialization order.

No event matching is inferred across different label sequences. Exact serialized
labels permit positional diagnostics only, not a claim of semantic correspondence.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math

from .sequence_evaluation import evaluate_label_sequences
from .text_sequence import LabelSequenceCandidate
from .text_timing import TemporalSequenceCandidate, TimedLabel
from .tensors import tensorize_sir_annotations


@dataclass(frozen=True)
class TemporalEvaluationReport:
    payload: bytes

    def to_dict(self):
        return json.loads(self.payload)

    @property
    def sha256(self):
        return hashlib.sha256(self.payload).hexdigest()


def _ratio(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def evaluate_temporal_sequences(candidates, annotations, vocabulary, *,
                                annotation_sha256, origins_seconds, max_cells):
    """Report exact rational MAE/IoU conditional on exact serialized labels.

    Origins must match caller-declared annotation-clock origins. Rational
    arithmetic operates on the supplied binary floating-point coordinates; it
    avoids endpoint-subtraction overflow but does not add physical precision.
    """
    if (not isinstance(candidates, tuple) or not candidates
            or not isinstance(origins_seconds, tuple) or len(origins_seconds) != len(candidates)):
        raise ValueError('nonempty candidate tuple and one explicit origin per row required')
    lexical = []
    for candidate, origin in zip(candidates, origins_seconds):
        if not isinstance(candidate, TemporalSequenceCandidate):
            raise ValueError('typed temporal candidate required')
        if (type(origin) not in (int, float) or not math.isfinite(origin)
                or type(candidate.origin_seconds) not in (int, float)
                or not math.isfinite(candidate.origin_seconds) or candidate.origin_seconds != origin):
            raise ValueError('candidate and declared clock origins differ')
        if not isinstance(candidate.events, tuple):
            raise ValueError('immutable temporal event tuple required')
        for event in candidate.events:
            if (not isinstance(event, TimedLabel)
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           for v in (event.start_seconds, event.end_seconds))
                    or event.start_seconds < origin or event.end_seconds <= event.start_seconds):
                raise ValueError('candidate intervals must be finite, positive and at or after origin')
        lexical.append(LabelSequenceCandidate(tuple(e.label for e in candidate.events), candidate.status,
                                              candidate.diagnostic_prefix, candidate.vocabulary_sha256))
    labels = evaluate_label_sequences(tuple(lexical), annotations, vocabulary,
                                      annotation_sha256=annotation_sha256, max_cells=max_cells).to_dict()
    targets = tensorize_sir_annotations(annotations, expected_lexicon=vocabulary.lexicon,
                                       expected_convention=vocabulary.convention)
    rows = []
    errors, overlaps, event_count, supported = Fraction(0), Fraction(0), 0, 0
    for index, (candidate, label_row) in enumerate(zip(candidates, labels['rows'])):
        diagnostics = None
        if label_row['exact_match']:
            row_error, row_overlap = Fraction(0), Fraction(0)
            for event, interval in zip(candidate.events, targets.intervals[index].tolist()):
                start, end = map(Fraction, (event.start_seconds, event.end_seconds))
                ref_start, ref_end = map(Fraction, interval)
                row_error += abs(start - ref_start) + abs(end - ref_end)
                intersection = max(Fraction(0), min(end, ref_end) - max(start, ref_start))
                union = (end - start) + (ref_end - ref_start) - intersection
                row_overlap += intersection / union
            count = len(candidate.events)
            diagnostics = dict(events=count, endpoint_mae_seconds=_ratio(row_error / (2 * count)),
                               mean_interval_iou=_ratio(row_overlap / count))
            errors += row_error
            overlaps += row_overlap
            event_count += count
            supported += 1
        rows.append(dict(annotation_sha256=label_row['annotation_sha256'], status=candidate.status,
                         exact_label_sequence=label_row['exact_match'], timing=diagnostics))
    aggregate = None if not event_count else dict(
        events=event_count, endpoint_mae_seconds=_ratio(errors / (2 * event_count)),
        mean_interval_iou=_ratio(overlaps / event_count))
    data = dict(schema_version=1, scope='exact-label-conditional-timing-diagnostic', rows=rows,
                vocabulary_sha256=vocabulary.lexicon.sha256,
                convention_sha256=vocabulary.convention.sha256,
                coverage=dict(numerator=supported, denominator=len(candidates)),
                conditional_event_weighted=aggregate, phase_exit_approved=False,
                limitations=['Timing is conditional on exact label serialization; mismatches and failures are unavailable.',
                             'Positional agreement does not establish semantic correspondence, physical calibration or ASL accuracy.',
                             'Rational values are exact for supplied coordinates, not evidence of physical precision.'])
    return TemporalEvaluationReport(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode())
