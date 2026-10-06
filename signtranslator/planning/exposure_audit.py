"""Fresh annotation checks of historical declared support, not gradient evidence."""
from dataclasses import dataclass
import hashlib
import json

from ..data.governed_corpus import GovernedMotionDataset, collate_governed_motion
from ..reproducibility import canonical_json_bytes
from .label_vocabulary import encode_governed_labels
from .relations import relation_targets
from .referents import referent_targets
from .loci import locus_targets
from .tensors import EDGE_TYPES


@dataclass(frozen=True)
class ExposureDeclarationAudit:
    payload: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.payload).hexdigest()

    def to_dict(self):
        return json.loads(self.payload)


def _target_inventory(annotations, vocabulary, alphabet, branches):
    """Unweighted available target counts for one freshly admitted example."""
    labels = encode_governed_labels(annotations, vocabulary)[0]
    count = len(annotations[0].graph().events)
    inventory = {
        'sir_sequence': {'event_labels': count, 'stop_targets': 1,
                         'labels': [dict(class_index=i, kind=entry.kind.value,
                                         label_id=entry.label_id, identity=entry.identity,
                                         target_events=int((labels == i).sum()))
                                    for i, entry in enumerate(vocabulary.entries)]},
        'event_timing': {'event_intervals': count, 'endpoint_targets': 2 * count},
    }
    expected = {'sir_sequence': True, 'event_timing': True}
    relations = relation_targets(annotations, vocabulary)
    relation_counts = {}
    for i, kind in enumerate(EDGE_TYPES):
        positive = int(relations.positive[..., i].sum())
        known = int(relations.known[..., i].sum())
        relation_counts[kind.value] = dict(positive=positive, negative=known - positive,
                                          unknown=count * (count - 1) - known)
    inventory['sir_relations'] = relation_counts
    expected['sir_relations'] = bool(relations.known.any())
    if 'referent_equality' in branches:
        references = referent_targets(annotations, vocabulary)
        positive, known = int(references.equal.sum()), int(references.known.sum())
        inventory['referent_equality'] = dict(positive=positive, negative=known - positive,
                                             unknown=count * (count - 1) // 2 - known)
        expected['referent_equality'] = known > 0
    if 'locus_assignment' in branches:
        loci = locus_targets(annotations, vocabulary, alphabet)
        known = int(loci.known.sum())
        inventory['locus_assignment'] = dict(
            known=known, unknown=count - known,
            classes=[dict(locus_id=i, identity=identity, target_events=int((loci.classes == i).sum()))
                     for i, identity in enumerate(alphabet.identities)])
        expected['locus_assignment'] = known > 0
    return {name: inventory[name] for name in branches}, expected


def audit_exposure_declarations(trainer, vocabulary, alphabet=None):
    """Compare canonical planner declarations with freshly revalidated targets.

    Unknown historical memberships are retained as unverified. This does not
    reconstruct missing masks or prove which gradients an optimizer applied.
    """
    from ..training.trainer import Trainer, _model_contract
    from .text_relations import RelationalTextSIRModel
    from .text_referents import ReferentTextSIRModel
    from .text_loci import LocusTextSIRModel

    if (not isinstance(trainer, Trainer) or type(trainer.model) not in
            (RelationalTextSIRModel, ReferentTextSIRModel, LocusTextSIRModel)):
        raise ValueError('canonical support-aware planner trainer required')
    current_contract = _model_contract(trainer.model)
    for field in ('governed_batch_schema_version', 'governed_objective_schema_version'):
        current_contract[field] = getattr(trainer.model, field, None)
    if current_contract != trainer.model_contract:
        raise ValueError('model contract changed since exposure was recorded')
    if vocabulary != trainer.model.vocabulary:
        raise ValueError('audit vocabulary must match the recorded planner')
    if type(trainer.model) is LocusTextSIRModel and alphabet != trainer.model.locus_alphabet:
        raise ValueError('audit alphabet must match the recorded planner')
    report = trainer.exposure_report()
    data = report.to_dict()
    dataset = trainer.train_loader.dataset
    contract = dataset.training_contract
    if data['data_contract']['governed'] != contract:
        raise ValueError('exposure and audited dataset views differ')
    first = dataset[0]
    branches = tuple(data['weights'])
    allowed = {'sir_sequence', 'event_timing', 'sir_relations',
               'referent_equality', 'locus_assignment'}
    if not set(branches) <= allowed:
        raise ValueError('unsupported exposure branch semantics')
    totals = {name: dict(verified_supported=0, verified_unsupported=0,
                        unverified_membership=0, contradictory=0) for name in branches}
    samples, contradictions = [], []
    for index, exposure in zip(contract['record_indices'], data['samples'], strict=True):
        view = GovernedMotionDataset(first.corpus, (index,), contract['split'])
        batch = collate_governed_motion([view[0]])
        if batch.corpus_sha256 != contract['corpus_sha256'] or batch.split != contract['split']:
            raise ValueError('admission changed during declaration audit')
        annotation = batch.annotations[0]
        identity = dict(sample_id=annotation.source.sample_id,
                        annotation_sha256=annotation.content_sha256())
        if any(identity[key] != exposure[key] for key in ('sample_id', 'annotation_sha256')):
            raise ValueError('exposure annotation identity does not match fresh audit')
        inventory, expected = _target_inventory(batch.annotations, vocabulary, alphabet, branches)
        for name in branches:
            counts = exposure['branch_presentations'][name]
            valid = 'supported' if expected[name] else 'unsupported'
            invalid = 'unsupported' if expected[name] else 'supported'
            totals[name]['verified_' + valid] += counts[valid]
            totals[name]['unverified_membership'] += counts['unattributed']
            totals[name]['contradictory'] += counts[invalid]
            if counts[invalid]:
                contradictions.append(dict(sample_id=exposure['sample_id'], branch=name,
                                           expected_supported=expected[name],
                                           contradictory_presentations=counts[invalid]))
        samples.append(dict(sample_id=identity['sample_id'],
                            annotation_sha256=identity['annotation_sha256'],
                            available_targets=inventory,
                            target_inventory_sha256=hashlib.sha256(canonical_json_bytes(inventory)).hexdigest(),
                            expected_support={name: expected[name] for name in branches}))
    if dataset.training_contract != contract or trainer.exposure_report().payload != report.payload:
        raise ValueError('training view or exposure changed during audit')
    status = ('inconsistent_declared_support' if contradictions else
              'no_recorded_exposure' if data['recorded_optimizer_steps'] == 0 else
              'partially_attributed' if any(t['unverified_membership'] for t in totals.values()) else
              'consistent_declared_support')
    result = dict(schema_version=2, status=status, exposure_report_sha256=report.sha256,
                  data_contract=contract, lexicon_sha256=vocabulary.lexicon.sha256,
                  convention_sha256=vocabulary.convention.sha256,
                  committed_epoch_boundary=data['committed_epoch_boundary'],
                  branch_presentations=totals, samples=samples, contradictions=contradictions,
                  phase_exit_approved=False,
                  limitations=['Checks declared example support against current admitted target definitions.',
                               'Does not prove historical gradient application, target-cell exposure or competence.',
                               'Available target counts are unweighted per-example counts, not optimizer exposure.',
                               'Unknown historical membership is not reconstructed.',
                               'Byte revalidation is not new human authorization or empirical acceptance.'])
    return ExposureDeclarationAudit(canonical_json_bytes(result))
