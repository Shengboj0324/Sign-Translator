"""Executable scaffolded text model tests; all governance evidence is fictional."""
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from signtranslator.config import TrainerConfig
from signtranslator.data.governed_text import PLAINTEXT_ENCODING
from signtranslator.planning.label_vocabulary import GovernedLabelVocabulary
from signtranslator.planning.supervision import GovernedSIRAnnotation
from signtranslator.grammar.sir import SIRGraph, sir_sha256
from signtranslator.planning.text_labels import ScaffoldedTextSIRLabelModel
from signtranslator.training import Trainer
from test_governed_corpus import population, admit
from test_governed_motion import bind_alignment
from test_governed_trainer import loader


def setup(tmp_path, *, configurations=None, single_event_indices=(), referents=None, loci=None, convention_payload=None):
    records, options = population(tmp_path, [{}, {'split': 'val'}]
                                  if configurations is None else configurations)
    annotation = records[0].annotation
    convention = annotation.convention
    if convention_payload is not None:
        convention = replace(convention, sha256=hashlib.sha256(convention_payload).hexdigest())
    rows = [dict(kind='manual', label_id=10, identity='fictional-a'),
            dict(kind='nonmanual', label_id=11, identity='fictional-marker'),
            dict(kind='manual', label_id=12, identity='fictional-b')]
    payload = json.dumps(dict(schema_version=1, artifact_id=annotation.lexicon.artifact_id,
                              version=annotation.lexicon.version,
                              convention_sha256=convention.sha256, entries=rows)).encode()
    lexicon = replace(annotation.lexicon, sha256=hashlib.sha256(payload).hexdigest())
    vocabulary = GovernedLabelVocabulary(lexicon, convention, payload)
    for index, record in enumerate(records):
        annotation = replace(record.annotation, lexicon=lexicon, convention=convention,
                             lexicon_convention_sha256=convention.sha256,
                             review=replace(record.annotation.review,
                                            reviewed_lexicon_sha256=lexicon.sha256,
                                            reviewed_convention_sha256=convention.sha256))
        if index in single_event_indices:
            graph = SIRGraph([annotation.graph().events[0]], [])
            annotation = GovernedSIRAnnotation.create(
                annotation_id=annotation.annotation_id, origin=annotation.origin, source=annotation.source,
                convention=annotation.convention, lexicon=annotation.lexicon,
                lexicon_convention_sha256=annotation.lexicon_convention_sha256, graph=graph,
                review=replace(annotation.review, reviewed_sir_sha256=sir_sha256(graph)),
                created_at=annotation.created_at)
        if referents is not None or loci is not None:
            graph = annotation.graph()
            if referents is not None:
                assert len(referents) == len(graph.events)
                for event, referent in zip(graph.events, referents):
                    event.referent = referent
            if loci is not None:
                assert len(loci) == len(graph.events)
                for event, locus in zip(graph.events, loci):
                    event.locus = locus
            annotation = GovernedSIRAnnotation.create(
                annotation_id=annotation.annotation_id, origin=annotation.origin, source=annotation.source,
                convention=annotation.convention, lexicon=annotation.lexicon,
                lexicon_convention_sha256=annotation.lexicon_convention_sha256, graph=graph,
                review=replace(annotation.review, reviewed_sir_sha256=sir_sha256(graph)),
                created_at=annotation.created_at)
        alignment = json.loads(record.alignment_path.read_bytes())
        alignment['annotation_sha256'] = annotation.content_sha256()
        bind_alignment(options[index], alignment)
        records[index] = replace(record, annotation=annotation,
                                  alignment_sha256=options[index]['alignment_sha256'])
    return vocabulary, admit(records, options)


def model(vocabulary):
    return ScaffoldedTextSIRLabelModel(vocabulary, declared_encoding=PLAINTEXT_ENCODING,
                                      embedding_dim=8, hidden_dim=12, max_events=4, max_bytes=64)


def test_real_trainer_optimizes_text_head_and_binds_model_vocabulary(tmp_path):
    vocabulary, corpus = setup(tmp_path)
    torch.manual_seed(7)
    net = model(vocabulary)
    cfg = TrainerConfig(epochs=6, lr=.05, selection_metric='sir_labels',
                        loss_weights={'sir_labels': 1.})
    trainer = Trainer(net, cfg, loader(corpus, 'train'), loader(corpus, 'val'))
    initial = trainer.validate()['sir_labels']
    trainer.fit()
    assert trainer.validate()['sir_labels'] < initial * .6
    assert trainer.global_step == 6
    assert trainer.model_contract['configs']['model_cfg']['values']['lexicon_sha256'] == vocabulary.lexicon.sha256
    path = trainer.save(tmp_path / 'labels.pt')
    restored = Trainer(model(vocabulary), cfg, loader(corpus, 'train'), loader(corpus, 'val'))
    restored.load(path)
    assert restored.validate() == trainer.validate()


def test_forward_depends_on_source_and_is_invariant_to_padding(tmp_path):
    vocabulary, _ = setup(tmp_path)
    torch.manual_seed(11)
    net = model(vocabulary)
    a = net(torch.tensor([[66, 67]]), torch.tensor([2]), torch.tensor([2]))
    b = net(torch.tensor([[67, 66]]), torch.tensor([2]), torch.tensor([2]))
    assert not torch.equal(a, b)
    padded = net(torch.tensor([[66, 67, 0, 0]]), torch.tensor([2]), torch.tensor([2]))
    torch.testing.assert_close(a, padded, rtol=0, atol=0)
    mixed = net(torch.tensor([[66, 67, 0], [68, 69, 70]]), torch.tensor([2, 3]), torch.tensor([1, 3]))
    assert torch.equal(mixed[0, 1:], torch.zeros_like(mixed[0, 1:]))
    mixed.sum().backward()
    assert not net.byte_embedding.weight.grad[0].any()


def test_cached_target_and_motion_tensors_cannot_leak_into_predictions(tmp_path):
    vocabulary, corpus = setup(tmp_path)
    net = model(vocabulary)
    batch = next(iter(loader(corpus, 'train')))
    first = net.training_step(batch, weights={'sir_labels': 1.})['sir_labels']
    batch.sir_targets.label_ids.fill_(99999)
    batch.sir_targets.event_kinds.fill_(99999)
    batch.motion.channels['root_translation'].values.fill_(float('nan'))
    second = net.training_step(batch, weights={'sir_labels': 1.})['sir_labels']
    assert torch.equal(first, second)
    with pytest.raises(ValueError, match='transcript identity'):
        net.training_step(replace(batch, transcript_payloads=(b'changed',)), weights={'sir_labels': 1.})


@pytest.mark.parametrize('ids,lengths,counts', [
    ([[1, 0]], [2], [1]), ([[1, 2]], [1], [1]), ([[257]], [1], [1]),
    ([[1]], [0], [1]), ([[1]], [2], [1]), ([[1]], [1], [0]), ([[1]], [1], [5]),
])
def test_malformed_source_or_event_scaffold_is_refused(tmp_path, ids, lengths, counts):
    vocabulary, _ = setup(tmp_path)
    with pytest.raises(ValueError):
        model(vocabulary)(torch.tensor(ids), torch.tensor(lengths), torch.tensor(counts))


@pytest.mark.parametrize('weights', [{}, {'total': 1.}, {'sir_labels': 0.},
                                   {'sir_labels': float('nan')}, {'sir_labels': True}])
def test_objective_is_explicit(tmp_path, weights):
    vocabulary, corpus = setup(tmp_path)
    with pytest.raises(ValueError):
        model(vocabulary).training_step(next(iter(loader(corpus, 'train'))), weights=weights)


def test_reordered_classes_cannot_reuse_checkpoint_or_mutate_binding(tmp_path):
    vocabulary, corpus = setup(tmp_path)
    net = model(vocabulary)
    trainer = Trainer(net, TrainerConfig(epochs=1), loader(corpus, 'train'))
    path = trainer.save(tmp_path / 'bound.pt')
    payload = json.loads(vocabulary.payload)
    payload['entries'].reverse()
    raw = json.dumps(payload).encode()
    other = GovernedLabelVocabulary(
        replace(vocabulary.lexicon, sha256=hashlib.sha256(raw).hexdigest()),
        vocabulary.convention, raw)
    candidate = Trainer(model(other), TrainerConfig(epochs=1), loader(corpus, 'train'))
    with pytest.raises(ValueError, match='model contract'):
        candidate.load(path, mode='weights')
    net.vocabulary = other
    with pytest.raises(ValueError, match='class binding'):
        net.training_step(next(iter(loader(corpus, 'train'))), weights={'sir_labels': 1.})
