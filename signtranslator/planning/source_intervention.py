"""Paired free-generation source interventions on an admitted dataset view.

Alternative source bytes remain bound to their own reviewed records. They are
never relabeled as an anchor's reviewed transcript. No target event labels, counts,
edges, references or endpoints enter generation. Clock origins stay at anchor
values to isolate the changed source input. Sensitivity is not semantic accuracy.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import torch

from ..data.governed_corpus import GovernedMotionDataset, collate_governed_motion
from ..data.governed_text import encode_plaintext_transcripts
from ..reproducibility import isolated_deterministic_rng as _intervention_rng
from ..inference_context import preserving_eval_mode
from .text_loci import LocusTextSIRModel
from .text_sequence import LabelSequenceCandidate
from .sequence_evaluation import evaluate_label_sequences
from .temporal_evaluation import evaluate_temporal_sequences
from .relation_evaluation import evaluate_relation_sequences, validated_relation_thresholds
from .referent_evaluation import evaluate_referent_sequences
from .locus_evaluation import evaluate_locus_sequences


@dataclass(frozen=True)
class SourceInterventionReport:
    payload: bytes

    def to_dict(self):
        return json.loads(self.payload)

    @property
    def sha256(self):
        return hashlib.sha256(self.payload).hexdigest()


def _state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(len(raw).to_bytes(8, 'big')); digest.update(raw)
    return digest.hexdigest()


def _candidate_dict(candidate):
    relational = candidate.referential.relational
    temporal = relational.temporal
    def label(entry):
        return dict(kind=entry.kind.value, label_id=entry.label_id, identity=entry.identity)
    def values(tensor):
        return None if tensor is None else tensor.detach().cpu().tolist()
    return dict(status=temporal.status, origin_seconds=temporal.origin_seconds,
                vocabulary_sha256=temporal.vocabulary_sha256,
                events=[dict(label=label(e.label), start_seconds=e.start_seconds, end_seconds=e.end_seconds)
                        for e in temporal.events],
                diagnostic_prefix=[label(e) for e in temporal.diagnostic_prefix],
                relation_types=[e.value for e in relational.relation_types],
                relation_logits=values(relational.relation_logits), relation_valid=values(relational.relation_valid),
                referent_logits=values(candidate.referential.equality_logits),
                referent_valid=values(candidate.referential.pair_valid), locus_logits=values(candidate.locus_logits),
                convention_sha256=candidate.convention_sha256, locus_identities=list(candidate.locus_identities))


@torch.no_grad()
def compare_source_intervention(model: LocusTextSIRModel, dataset: GovernedMotionDataset, *,
                                 permutation: tuple[int, ...], seed: int,
                                 max_samples: int, relation_thresholds=None) -> SourceInterventionReport:
    """Compare original and explicitly permuted sources using identical RNG seeds.

    Each pass restores external RNG state, and the original model mode is restored
    even on failure. Parameters/buffers must remain unchanged. Raw candidates are
    serialized immutably; no acceptance or reference-aligned accuracy is inferred.
    """
    if not isinstance(model, LocusTextSIRModel) or not isinstance(dataset, GovernedMotionDataset):
        raise ValueError('locus text model and admitted dataset view required')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('explicit uint32 intervention seed required')
    if type(max_samples) is not int or not 1 <= max_samples <= 64:
        raise ValueError('explicit max_samples between 1 and 64 required')
    count = len(dataset)
    if not 1 <= count <= max_samples:
        raise ValueError('view must be nonempty and fit the declared sample budget')
    if (not isinstance(permutation, tuple) or len(permutation) != count
            or any(type(i) is not int for i in permutation) or set(permutation) != set(range(count))):
        raise ValueError('explicit complete permutation of view rows required')
    relation_thresholds = validated_relation_thresholds(relation_thresholds)
    contract = dataset.training_contract
    first = dataset[0]
    view = GovernedMotionDataset(first.corpus, tuple(contract['record_indices']), contract['split'])
    if view.training_contract != contract:
        raise ValueError('view admission identity changed')
    batch = collate_governed_motion([view[i] for i in range(len(view))])
    if (batch.sir_targets.lexicon != model.vocabulary.lexicon
            or batch.sir_targets.convention != model.vocabulary.convention):
        raise ValueError('model and admitted annotation vocabularies differ')
    text = encode_plaintext_transcripts(batch.transcript_payloads, batch.annotations,
                                       declared_encoding=model.model_cfg.declared_encoding,
                                       max_bytes=model.model_cfg.max_bytes)
    device = model.byte_embedding.weight.device
    origins = torch.tensor([extent[0] for extent in batch.annotation_extents], dtype=torch.float64, device=device)
    ids, lengths = text.token_ids.to(device), text.lengths
    index = torch.tensor(permutation, dtype=torch.int64)
    state_before = _state_sha256(model)
    with preserving_eval_mode(model):
        with _intervention_rng(seed):
            original_candidates = model.generate_loci(ids, lengths, origins_seconds=origins.clone())
            original = tuple(_candidate_dict(c) for c in original_candidates)
            lexical = tuple(LabelSequenceCandidate(
                tuple(e.label for e in c.referential.relational.temporal.events),
                c.referential.relational.temporal.status,
                c.referential.relational.temporal.diagnostic_prefix,
                c.referential.relational.temporal.vocabulary_sha256) for c in original_candidates)
            relation_evaluation = evaluate_relation_sequences(
                tuple(c.referential.relational for c in original_candidates), tuple(batch.annotations),
                model.vocabulary, annotation_sha256=text.annotation_sha256,
                max_cells=10_000_000, max_relation_cells=10_000_000,
                relation_thresholds=relation_thresholds).to_dict()
            referent_evaluation = evaluate_referent_sequences(
                tuple(c.referential for c in original_candidates), tuple(batch.annotations),
                model.vocabulary, annotation_sha256=text.annotation_sha256,
                max_cells=10_000_000, max_pair_cells=10_000_000).to_dict()
            locus_evaluation = evaluate_locus_sequences(
                tuple(original_candidates), tuple(batch.annotations), model.vocabulary, model.locus_alphabet,
                annotation_sha256=text.annotation_sha256, max_cells=10_000_000,
                max_locus_cells=10_000_000).to_dict()
        with _intervention_rng(seed):
            intervened = tuple(_candidate_dict(c) for c in model.generate_loci(
                ids.index_select(0, index.to(device)), lengths.index_select(0, index),
                origins_seconds=origins.clone()))
        if _state_sha256(model) != state_before:
            raise RuntimeError('model parameters or buffers changed during intervention')
    if dataset.training_contract != contract or view.training_contract != contract:
        raise ValueError('dataset view changed during intervention')
    if len(original) != count or len(intervened) != count:
        raise ValueError('generation output count differs from admitted view')
    rows = []
    for row, source in enumerate(permutation):
        baseline, changed = original[row], intervened[row]
        both_terminated = baseline['status'] == changed['status'] == 'terminated'
        rows.append(dict(anchor_sample_id=batch.annotations[row].source.sample_id,
                         anchor_annotation_sha256=text.annotation_sha256[row],
                         source_sample_id=batch.annotations[source].source.sample_id,
                         source_annotation_sha256=text.annotation_sha256[source],
                         original_transcript_sha256=text.transcript_sha256[row],
                         intervened_transcript_sha256=text.transcript_sha256[source],
                         source_bytes_changed=text.transcript_sha256[row] != text.transcript_sha256[source],
                         original=baseline, intervened=changed, raw_candidate_changed=baseline != changed,
                         both_terminated=both_terminated,
                         same_label_sequence=(None if not both_terminated else
                            [e['label'] for e in baseline['events']] == [e['label'] for e in changed['events']])))
    reference_evaluation = evaluate_label_sequences(
        lexical, tuple(batch.annotations), model.vocabulary,
        annotation_sha256=text.annotation_sha256, max_cells=10_000_000).to_dict()
    temporal_evaluation = evaluate_temporal_sequences(
        tuple(c.referential.relational.temporal for c in original_candidates),
        tuple(batch.annotations), model.vocabulary, annotation_sha256=text.annotation_sha256,
        origins_seconds=tuple(extent[0] for extent in batch.annotation_extents), max_cells=10_000_000).to_dict()
    result = dict(schema_version=1, original_locus_evaluation=locus_evaluation,
                  original_referent_evaluation=referent_evaluation,
                  original_relation_evaluation=relation_evaluation,
                  original_temporal_evaluation=temporal_evaluation,
                  original_reference_evaluation=reference_evaluation, scope='source-permutation-diagnostic', data_contract=contract,
                  model_state_sha256=state_before, model_config=asdict(model.model_cfg),
                  seed=seed, permutation=list(permutation), clock_policy='hold-anchor-origin-fixed', rows=rows,
                  limitations=['Raw output sensitivity is not semantic accuracy or causal ASL competence.',
                               'Permuted source bytes retain their own reviewed identities; no new annotation is created.',
                               'Failed generations have no label-accuracy comparison; no graph or phase acceptance is certified.'],
                  phase_exit_approved=False)
    return SourceInterventionReport(json.dumps(result, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode('utf-8'))
