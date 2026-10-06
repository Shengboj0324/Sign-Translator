"""Explicit admitted-corpus W3 training; no synthetic fallback or phase approval."""
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

from torch.utils.data import DataLoader

from .config import TrainerConfig
from .data.governed_corpus import GovernedMotionCorpus, collate_governed_motion
from .data.governed_text import encode_plaintext_transcripts
from .planning.exposure_audit import ExposureDeclarationAudit, audit_exposure_declarations
from .planning.label_vocabulary import GovernedLabelVocabulary
from .planning.loci import LocusAlphabet
from .planning.support_audit import SupervisionSupportReport, audit_supervision_support
from .planning.text_loci import LocusTextConfig, LocusTextSIRModel
from .reproducibility import isolated_deterministic_rng
from .training.trainer import Trainer


@dataclass(frozen=True)
class GovernedPlannerRun:
    trainer: Trainer
    training_support: SupervisionSupportReport
    validation_support: SupervisionSupportReport | None
    exposure_audit: ExposureDeclarationAudit

    @property
    def phase_exit_approved(self):
        return False


def run_governed_planner(corpus: GovernedMotionCorpus, vocabulary: GovernedLabelVocabulary,
                         alphabet: LocusAlphabet, *, model_config: LocusTextConfig,
                         trainer_config: TrainerConfig, validation: bool, shuffle: bool,
                         resume_from: str | Path | None = None,
                         max_epochs: int | None = None) -> GovernedPlannerRun:
    """Preflight, initialize, fit/resume and audit the canonical five-head model.

    Initialization and execution use cfg.seed in an isolated RNG context. Resume
    restores the checkpoint RNG after construction. Loaders use zero workers and
    never drop samples. The test partition is not opened. Returned Trainer state
    is mutable; the reports are dated snapshots, not ongoing certifications.
    Checkpoint writes follow TrainerConfig.ckpt_path and are not rolled back if
    a later operation fails. Exact continuation is verified on CPU only.
    """
    if (not isinstance(corpus, GovernedMotionCorpus)
            or not isinstance(vocabulary, GovernedLabelVocabulary)
            or not isinstance(alphabet, LocusAlphabet)
            or type(model_config) is not LocusTextConfig
            or type(trainer_config) is not TrainerConfig):
        raise ValueError('admitted corpus and typed planner/trainer configurations required')
    if type(validation) is not bool or type(shuffle) is not bool:
        raise ValueError('explicit boolean validation and shuffle choices required')
    if max_epochs is not None and (type(max_epochs) is not int or max_epochs <= 0):
        raise ValueError('max_epochs must be a positive exact integer')
    cfg = deepcopy(trainer_config)
    cfg.__post_init__()
    model_config = replace(model_config)  # rerun validation against altered fields
    branches = {'sir_sequence', 'event_timing', 'sir_relations', 'referent_equality', 'locus_assignment'}
    if (set(cfg.loss_weights) != branches or any(weight <= 0 for weight in cfg.loss_weights.values())
            or cfg.selection_metric not in branches):
        raise ValueError('five explicit positive branch weights and a branch selection metric required')
    if (model_config.lexicon_sha256 != vocabulary.lexicon.sha256
            or model_config.convention_sha256 != vocabulary.convention.sha256
            or alphabet.convention != vocabulary.convention
            or model_config.locus_count != len(alphabet.identities)):
        raise ValueError('model configuration does not match vocabulary and locus alphabet')
    train = corpus.split('train')
    val = corpus.split('val') if validation else None
    reports = []
    for view in (train, val):
        if view is None:
            reports.append(None)
            continue
        report = audit_supervision_support(view, vocabulary, alphabet)
        if any(sample['event_count'] > model_config.max_events for sample in report.to_dict()['samples']):
            raise ValueError('admitted sequence exceeds configured event capacity')
        # Bound each transcript without allocating a whole-corpus padded batch.
        for index in range(len(view)):
            batch = collate_governed_motion([view[index]])
            encode_plaintext_transcripts(batch.transcript_payloads, batch.annotations,
                                         declared_encoding=model_config.declared_encoding,
                                         max_bytes=model_config.max_bytes)
        reports.append(report)
    if val is not None:
        support = reports[1].to_dict()
        availability = dict(sir_sequence=support['sample_count'], event_timing=support['sample_count'],
                            sir_relations=support['relations']['examples_known'],
                            referent_equality=support['referent_equality']['examples_known'],
                            locus_assignment=support['locus_assignment']['examples_known'])
        if not availability[cfg.selection_metric]:
            raise ValueError('validation selection metric has no admitted target support')
    context = dict(governed_runner_schema_version=1, train_support_sha256=reports[0].sha256,
                   validation_support_sha256=reports[1].sha256 if reports[1] is not None else None)
    with isolated_deterministic_rng(cfg.seed):
        model = LocusTextSIRModel(
            vocabulary, locus_alphabet=alphabet, declared_encoding=model_config.declared_encoding,
            max_bytes=model_config.max_bytes, max_events=model_config.max_events,
            embedding_dim=model_config.embedding_dim, hidden_dim=model_config.hidden_dim,
            timing_scale_seconds=model_config.timing_scale_seconds)
        if model.model_cfg != model_config:
            raise ValueError('constructed model differs from requested configuration')
        train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=shuffle,
                                  num_workers=0, drop_last=False, collate_fn=collate_governed_motion)
        val_loader = (DataLoader(val, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=0, drop_last=False, collate_fn=collate_governed_motion)
                      if val is not None else None)
        trainer = Trainer(model, cfg, train_loader, val_loader, artifact_context=context)
        if resume_from is not None:
            trainer.load(resume_from, mode='resume')
        trainer.fit(max_epochs=max_epochs)
        audit = audit_exposure_declarations(trainer, vocabulary, alphabet)
        data = audit.to_dict()
        if data['contradictions'] or data['target_cell_audit']['contradictions']:
            raise ValueError('completed run has contradictory exposure declarations')
    return GovernedPlannerRun(trainer, reports[0], reports[1], audit)
