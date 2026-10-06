"""Byte-revalidated inventory of available supervision in one admitted view.

This reports targets, not optimizer exposure, learned competence, independent
statistical units or calibration. Counts are bound to the corpus, ordered view,
annotation identities and explicit vocabularies. Missing values remain unknown.
"""
from dataclasses import dataclass
import hashlib
import json

from ..data.governed_corpus import GovernedMotionDataset, collate_governed_motion
from .label_vocabulary import GovernedLabelVocabulary, encode_governed_labels
from .loci import LocusAlphabet, locus_targets
from .referents import referent_targets
from .relations import relation_targets
from .tensors import EDGE_TYPES


@dataclass(frozen=True)
class SupervisionSupportReport:
    payload: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def to_dict(self) -> dict:
        return json.loads(self.payload)


def _binary_counts():
    return dict(positive=0, negative=0, unknown=0, examples_positive=0,
                examples_negative=0, examples_known=0)


def _add_binary(counts, positive: int, negative: int, domain: int):
    counts['positive'] += positive
    counts['negative'] += negative
    counts['unknown'] += domain - positive - negative
    counts['examples_positive'] += int(positive > 0)
    counts['examples_negative'] += int(negative > 0)
    counts['examples_known'] += int(positive + negative > 0)


def audit_supervision_support(dataset: GovernedMotionDataset,
                              vocabulary: GovernedLabelVocabulary,
                              alphabet: LocusAlphabet) -> SupervisionSupportReport:
    """Audit every example of the exact view, re-reading admitted source bytes.

    A subset remains explicitly bound to its own ordered record indices. Train,
    validation and test views are not combined. An immutable report is a snapshot;
    subsequent source/consent changes require readmission and a fresh audit.
    """
    if not isinstance(dataset, GovernedMotionDataset) or len(dataset) == 0:
        raise ValueError('nonempty admitted governed dataset view required')
    if not isinstance(vocabulary, GovernedLabelVocabulary) or not isinstance(alphabet, LocusAlphabet):
        raise ValueError('typed lexical vocabulary and locus alphabet required')
    if vocabulary.convention != alphabet.convention:
        raise ValueError('lexical and locus conventions must match')
    contract = dataset.training_contract
    # Reconstruct to revalidate mutable view indices, uniqueness and partition.
    first = dataset[0]
    view = GovernedMotionDataset(first.corpus, tuple(contract['record_indices']), contract['split'])
    if view.training_contract != contract:
        raise ValueError('dataset view no longer matches its admission identity')
    labels = [dict(kind=e.kind.value, label_id=e.label_id, identity=e.identity,
                   target_events=0, examples=0) for e in vocabulary.entries]
    loci = [dict(locus_id=i, identity=name, target_events=0, examples=0)
            for i, name in enumerate(alphabet.identities)]
    relations = {kind.value: _binary_counts() for kind in EDGE_TYPES}
    references = _binary_counts()
    samples = []
    events = locus_known = locus_examples = relation_examples = 0
    for i in range(len(view)):
        batch = collate_governed_motion([view[i]])
        if batch.corpus_sha256 != contract['corpus_sha256'] or batch.split != contract['split']:
            raise ValueError('admission changed during support audit')
        annotations = batch.annotations
        annotation = annotations[0]
        count = len(annotation.graph().events)
        events += count
        samples.append(dict(sample_id=annotation.source.sample_id,
                            annotation_sha256=annotation.content_sha256(), event_count=count))
        target_labels = encode_governed_labels(annotations, vocabulary)[0]
        for j, row in enumerate(labels):
            occurrences = int((target_labels == j).sum())
            row['target_events'] += occurrences
            row['examples'] += int(occurrences > 0)
        rel = relation_targets(annotations, vocabulary)
        relation_examples += int(bool(rel.known.any()))
        for j, kind in enumerate(EDGE_TYPES):
            positive, known = int(rel.positive[..., j].sum()), int(rel.known[..., j].sum())
            _add_binary(relations[kind.value], positive, known - positive, count * (count - 1))
        ref = referent_targets(annotations, vocabulary)
        positive, known = int(ref.equal.sum()), int(ref.known.sum())
        _add_binary(references, positive, known - positive, count * (count - 1) // 2)
        target_loci = locus_targets(annotations, vocabulary, alphabet)
        known = int(target_loci.known.sum())
        locus_known += known
        locus_examples += int(known > 0)
        for j, row in enumerate(loci):
            occurrences = int((target_loci.classes == j).sum())
            row['target_events'] += occurrences
            row['examples'] += int(occurrences > 0)
    if dataset.training_contract != contract or view.training_contract != contract:
        raise ValueError('dataset view changed during support audit')
    for counts in (*relations.values(), references):
        positive, negative = counts['positive'] > 0, counts['negative'] > 0
        counts['observed_target_polarities'] = ('both' if positive and negative else
                                               'positive_only' if positive else
                                               'negative_only' if negative else 'none')
    result = dict(schema_version=1, scope='available-supervision-not-training-exposure',
                  data_contract=contract, lexicon_sha256=vocabulary.lexicon.sha256,
                  convention_sha256=vocabulary.convention.sha256, samples=samples,
                  sample_count=len(samples), event_count=events,
                  sequence=dict(examples=len(samples), stop_targets=len(samples), labels=labels),
                  timing=dict(examples=len(samples), event_intervals=events, endpoint_targets=2 * events),
                  relations=dict(examples_known=relation_examples, types=relations),
                  referent_equality=references,
                  locus_assignment=dict(examples_known=locus_examples, known_events=locus_known,
                                        unknown_events=events-locus_known, classes=loci),
                  limitations=['Counts are examples/events/cells, not independent statistical units.',
                               'Target availability does not prove optimization exposure or learning.',
                               'No model, calibrated decision policy or phase acceptance is certified.'],
                  phase_exit_approved=False)
    return SupervisionSupportReport(json.dumps(result, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode('utf-8'))
