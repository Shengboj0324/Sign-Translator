"""Verification of the composite training objective and the freeze-first schedule.

The central risk in a multi-term loss is that a term is present in the sum but
inert -- mis-wired, mis-shaped, or gradient-blocked -- while the total still
falls because the other terms carry it. Every term is therefore checked in
isolation: its own gradient path, and the effect of zeroing its weight.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from signtranslator.speech.objective import (
    BoundaryHead, boundary_loss, balanced_pos_weight,
    boundary_targets_from_alignment, ObjectiveWeights, SpeechTrainingObjective,
    speech_sign_retrieval,
)
from signtranslator.speech.alignment import ctc_forced_alignment
from signtranslator.speech.schedule import (
    FreezeFirstSchedule, FreezeFirstConfig, Phase,
)
from signtranslator.speech.lora import inject_lora
from signtranslator.models import SpeechRecognizer
from signtranslator.models.alignment import ContrastiveAligner


def _log_probs(T, C, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.log_softmax(torch.randn(T, C, generator=g, dtype=torch.float64), dim=-1)


# ---------------------------------------------------------------------------
# Boundary targets
# ---------------------------------------------------------------------------
def test_boundary_targets_mark_the_first_frame_of_each_token():
    """Constructed alignment: [1,1,1,blank,2,2,blank,3] -> starts at 0, 4, 7."""
    truth = [1, 1, 1, 0, 2, 2, 0, 3]
    p = torch.full((len(truth), 4), 0.001, dtype=torch.float64)
    for t, c in enumerate(truth):
        p[t, c] = 0.997
    al = ctc_forced_alignment(p.log(), [1, 2, 3], blank=0)
    tgt = boundary_targets_from_alignment(al, len(truth))
    assert tgt.tolist() == [1, 0, 0, 0, 1, 0, 0, 1]
    assert int(tgt.sum()) == 3               # exactly one per token


def test_boundary_targets_are_sparse():
    lp = _log_probs(40, 5, seed=1)
    al = ctc_forced_alignment(lp, [1, 2, 3])
    tgt = boundary_targets_from_alignment(al, 40)
    assert int(tgt.sum()) == 3
    assert float(tgt.mean()) < 0.1           # the imbalance the loss must handle


# ---------------------------------------------------------------------------
# Class balancing
# ---------------------------------------------------------------------------
def test_pos_weight_equals_neg_over_pos():
    tgt = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    assert abs(float(balanced_pos_weight(tgt)) - 4.0) < 1e-12


def test_pos_weight_is_one_when_a_class_is_absent():
    assert float(balanced_pos_weight(torch.zeros(5))) == 1.0
    assert float(balanced_pos_weight(torch.ones(5))) == 1.0


def test_pos_weight_is_capped():
    tgt = torch.zeros(100_000); tgt[0] = 1.0
    assert float(balanced_pos_weight(tgt, cap=500.0)) == 500.0


def test_balancing_denies_the_all_negative_shortcut():
    """THE reason the term is balanced.

    With 3 boundaries in 60 frames, always predicting "no boundary" achieves a
    tiny unweighted BCE -- it looks converged while detecting nothing. Balancing
    must make that degenerate solution expensive.
    """
    targets = torch.zeros(60); targets[[5, 20, 41]] = 1.0
    always_negative = torch.full((60,), -6.0)      # sigmoid ~ 0.0025
    unbalanced = float(boundary_loss(always_negative, targets, balanced=False))
    balanced = float(boundary_loss(always_negative, targets, balanced=True))
    assert unbalanced < 0.35                        # deceptively small
    assert balanced > 3 * unbalanced                # properly penalised


def test_boundary_loss_is_minimised_by_correct_predictions():
    targets = torch.zeros(30); targets[[3, 12, 25]] = 1.0
    good = torch.where(targets > 0, torch.tensor(8.0), torch.tensor(-8.0))
    bad = -good
    assert float(boundary_loss(good, targets)) < float(boundary_loss(bad, targets))
    assert float(boundary_loss(good, targets)) < 1e-2


def test_boundary_loss_validates_shapes():
    with pytest.raises(ValueError):
        boundary_loss(torch.zeros(5), torch.zeros(6))
    with pytest.raises(ValueError):
        boundary_loss(torch.zeros(0), torch.zeros(0))


# ---------------------------------------------------------------------------
# Boundary head
# ---------------------------------------------------------------------------
def test_boundary_head_shapes_and_gradients():
    head = BoundaryHead(hidden_dim=16)
    hidden = torch.randn(3, 20, 16)
    logits = head(hidden)
    assert logits.shape == (3, 20)
    logits.sum().backward()
    assert head.net[0].weight.grad.abs().sum() > 0


def test_boundary_head_rejects_bad_rank():
    with pytest.raises(ValueError):
        BoundaryHead(8)(torch.randn(4, 8))


def test_boundary_head_can_learn_a_pattern():
    """Overfit a fixed boundary map: the head must be expressive enough."""
    torch.manual_seed(0)
    head = BoundaryHead(hidden_dim=12)
    hidden = torch.randn(1, 40, 12)
    targets = torch.zeros(1, 40); targets[0, [4, 17, 31]] = 1.0
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    first = None
    for _ in range(300):
        loss = boundary_loss(head(hidden), targets)
        first = first if first is not None else loss.detach().item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.detach().item() < first * 0.1
    assert torch.equal(head.predict_boundaries(hidden)[0].nonzero().flatten(),
                       torch.tensor([4, 17, 31]))


# ---------------------------------------------------------------------------
# Combined objective
# ---------------------------------------------------------------------------
def _objective(weights=None, with_aligner=True, hidden=48, n_tokens=4):
    rec = SpeechRecognizer(input_dim=20, num_tokens=n_tokens, hidden_dim=hidden,
                           num_layers=2, num_heads=2, subsample=2)
    head = BoundaryHead(hidden_dim=hidden)
    aligner = (ContrastiveAligner(motion_dim=hidden, language_dim=hidden,
                                  latent_dim=16) if with_aligner else None)
    obj = SpeechTrainingObjective(rec, weights=weights, boundary_head=head,
                                  aligner=aligner)
    return obj, rec, head, aligner


def _batch(n=4, T=60, F_=20, n_tokens=4, L=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(n, T, F_, generator=g)
    targets = torch.randint(1, n_tokens + 1, (n, L), generator=g)
    lengths = torch.full((n,), L, dtype=torch.long)
    return feats, targets, lengths


def test_feature_lengths_exclude_padding_from_ctc():
    """Passing real per-sample feature lengths must change the CTC loss (padding
    frames are no longer counted as audio) and stay finite. Converting raw ->
    encoded length is handled inside the objective (the recognizer subsamples)."""
    obj, rec, _, _ = _objective()
    feats, targets, lengths = _batch(n=4, T=60, L=3, seed=1)
    flen = torch.tensor([20, 60, 60, 60], dtype=torch.long)     # sample 0 padded
    out_default = obj(feats, targets, lengths, sign_embeddings=torch.randn(4, 48))
    out_len = obj(feats, targets, lengths, sign_embeddings=torch.randn(4, 48),
                  feature_lengths=flen)
    assert torch.isfinite(out_len.total)
    assert abs(out_default.terms["asr"].item() - out_len.terms["asr"].item()) > 1e-6
    # the CTC input length used is the ENCODED (subsampled) length, not the raw 20
    assert int(rec.output_lengths(flen)[0]) < 20


def test_all_four_terms_are_present_and_finite():
    obj, _, _, _ = _objective()
    feats, targets, lengths = _batch()
    out = obj(feats, targets, lengths, sign_embeddings=torch.randn(4, 48))
    for name in ("asr", "boundary", "brier", "contrastive", "total"):
        assert name in out.terms, f"missing term {name}"
        assert torch.isfinite(out.terms[name]), name
    assert out.total.ndim == 0


def test_total_equals_the_weighted_sum():
    w = ObjectiveWeights(contrastive=0.3, boundary=0.7, brier=0.2)
    obj, _, _, _ = _objective(weights=w)
    feats, targets, lengths = _batch()
    out = obj(feats, targets, lengths, sign_embeddings=torch.randn(4, 48))
    expected = (out.terms["asr"] + 0.3 * out.terms["contrastive"]
                + 0.7 * out.terms["boundary"] + 0.2 * out.terms["brier"])
    assert torch.allclose(out.total, expected, atol=1e-6)


def test_each_term_reaches_its_own_parameters():
    """Guard against a term that is summed in but gradient-inert."""
    obj, rec, head, aligner = _objective()
    feats, targets, lengths = _batch()
    out = obj(feats, targets, lengths, sign_embeddings=torch.randn(4, 48))
    out.total.backward()
    assert head.net[0].weight.grad.abs().sum() > 0, "boundary head got no gradient"
    assert aligner.motion_head.net[0].weight.grad.abs().sum() > 0, "aligner inert"
    assert rec.classifier.weight.grad.abs().sum() > 0, "recogniser inert"


def test_zeroing_a_weight_removes_that_term():
    obj, _, head, _ = _objective(
        weights=ObjectiveWeights(contrastive=0.0, boundary=0.0, brier=0.0))
    feats, targets, lengths = _batch()
    out = obj(feats, targets, lengths, sign_embeddings=torch.randn(4, 48))
    assert "boundary" not in out.terms and "brier" not in out.terms
    assert "contrastive" not in out.terms
    assert torch.allclose(out.total, out.terms["asr"], atol=1e-9)
    out.total.backward()
    assert head.net[0].weight.grad is None      # truly excluded, not just scaled


def test_contrastive_term_is_skipped_without_sign_embeddings():
    obj, _, _, _ = _objective()
    feats, targets, lengths = _batch()
    out = obj(feats, targets, lengths, sign_embeddings=None)
    assert "contrastive" not in out.terms
    assert "asr" in out.terms


def test_objective_handles_transcripts_too_long_for_the_frames():
    """A sample that cannot be aligned must not corrupt the batch loss."""
    obj, _, _, _ = _objective()
    # 8 frames after subsample=2 from T=16; ask for 12 tokens -> infeasible.
    feats = torch.randn(2, 16, 20)
    targets = torch.randint(1, 5, (2, 12))
    lengths = torch.full((2,), 12, dtype=torch.long)
    out = obj(feats, targets, lengths)
    assert torch.isfinite(out.total)


def test_objective_weights_reject_negatives():
    with pytest.raises(ValueError):
        ObjectiveWeights(contrastive=-0.1)
    with pytest.raises(ValueError):
        ObjectiveWeights(boundary=-1.0)


def test_optimising_the_objective_reduces_every_term():
    """End-to-end: the composite loss must not improve one term at another's cost."""
    torch.manual_seed(0)
    obj, _, _, _ = _objective()
    feats, targets, lengths = _batch(seed=3)
    signs = torch.randn(4, 48)
    opt = torch.optim.Adam(obj.parameters(), lr=3e-3)
    first = obj(feats, targets, lengths, signs).detached()
    for _ in range(120):
        out = obj(feats, targets, lengths, signs)
        opt.zero_grad(); out.total.backward(); opt.step()
    last = obj(feats, targets, lengths, signs).detached()
    for name in ("asr", "boundary", "brier", "total"):
        assert last[name] < first[name], f"{name}: {first[name]} -> {last[name]}"


# ---------------------------------------------------------------------------
# Retrieval validation
# ---------------------------------------------------------------------------
def test_retrieval_is_perfect_for_identical_embeddings():
    emb = torch.randn(16, 8)
    r = speech_sign_retrieval(emb, emb.clone(), ks=(1, 5))
    assert r[1] == 1.0 and r[5] == 1.0


def test_retrieval_is_chance_for_unrelated_embeddings():
    torch.manual_seed(1)
    r = speech_sign_retrieval(torch.randn(50, 16), torch.randn(50, 16), ks=(1,))
    assert r[1] < 0.2                     # chance is 1/50


def test_retrieval_validates_pairing():
    with pytest.raises(ValueError):
        speech_sign_retrieval(torch.randn(4, 8), torch.randn(5, 8))


def test_contrastive_training_improves_retrieval():
    """The spec demands retrieval evidence, not just a falling InfoNCE loss."""
    torch.manual_seed(2)
    n, d = 24, 16
    sign = torch.randn(n, d)
    speech = sign + 0.8 * torch.randn(n, d)          # correlated but noisy
    aligner = ContrastiveAligner(motion_dim=d, language_dim=d, latent_dim=8)
    before = speech_sign_retrieval(aligner.motion_head(speech),
                                   aligner.language_head(sign), ks=(1,))[1]
    opt = torch.optim.Adam(aligner.parameters(), lr=0.02)
    for _ in range(300):
        loss = aligner(speech, sign)["loss"]
        opt.zero_grad(); loss.backward(); opt.step()
    after = speech_sign_retrieval(aligner.motion_head(speech),
                                  aligner.language_head(sign), ks=(1,))[1]
    assert after > before


# ---------------------------------------------------------------------------
# Freeze-first schedule
# ---------------------------------------------------------------------------
def _schedule_model():
    rec = SpeechRecognizer(input_dim=20, num_tokens=4, hidden_dim=32,
                           num_layers=3, num_heads=2, subsample=2)
    # Feed-forward layers: safe to wrap. (Attention's out_proj is reached
    # attribute-wise by PyTorch and is skipped by inject_lora.)
    inject_lora(rec, target_suffixes=("linear1", "linear2"), r=4)
    head = BoundaryHead(hidden_dim=32)
    blocks = list(rec.encoder.layers)
    return rec, head, blocks


def test_phase_one_trains_only_adapters_and_new_heads():
    rec, head, blocks = _schedule_model()
    sched = FreezeFirstSchedule(rec, blocks, FreezeFirstConfig(adapt_steps=5),
                                extra_trainable=[head])
    assert sched.phase == Phase.ADAPT
    for name, p in rec.named_parameters():
        if p.requires_grad:
            assert "lora_" in name, f"unexpectedly trainable in phase 1: {name}"


def test_phase_two_releases_only_the_upper_blocks():
    rec, head, blocks = _schedule_model()
    cfg = FreezeFirstConfig(adapt_steps=2, unfreeze_blocks=1)
    sched = FreezeFirstSchedule(rec, blocks, cfg, extra_trainable=[head])
    for _ in range(2):
        sched.step(rec(torch.randn(2, 20, 20)).sum())
    assert sched.phase == Phase.REFINE
    # linear1 is LoRA-wrapped, so the pretrained weight lives on `.base`.
    bottom = blocks[0].linear1.base.weight
    top = blocks[-1].linear1.base.weight
    assert not bottom.requires_grad, "a lower block was released"
    assert top.requires_grad, "the top block stayed frozen"


def test_frozen_parameters_provably_do_not_change():
    """Not just requires_grad=False -- the values must be identical after steps."""
    rec, head, blocks = _schedule_model()
    sched = FreezeFirstSchedule(rec, blocks, FreezeFirstConfig(adapt_steps=100),
                                extra_trainable=[head])
    frozen_ref = blocks[0].linear1.base.weight.detach().clone()
    for _ in range(5):
        sched.step(rec(torch.randn(2, 20, 20)).pow(2).mean())
    assert torch.equal(blocks[0].linear1.base.weight, frozen_ref)


def test_encoder_learning_rate_is_reduced_in_phase_two():
    rec, head, blocks = _schedule_model()
    cfg = FreezeFirstConfig(adapt_steps=1, base_lr=1e-3, encoder_lr_scale=0.1,
                            unfreeze_blocks=2)
    sched = FreezeFirstSchedule(rec, blocks, cfg, extra_trainable=[head])
    sched.step(rec(torch.randn(2, 20, 20)).sum())
    lrs = sched.summary()["learning_rates"]
    assert len(lrs) == 2
    assert abs(min(lrs) - 1e-4) < 1e-12       # encoder group
    assert abs(max(lrs) - 1e-3) < 1e-12       # adapter/head group


def test_trainable_count_grows_at_the_transition():
    rec, head, blocks = _schedule_model()
    sched = FreezeFirstSchedule(rec, blocks, FreezeFirstConfig(adapt_steps=1),
                                extra_trainable=[head])
    before = sched.summary()["trainable"]
    sched.step(rec(torch.randn(2, 20, 20)).sum())
    assert sched.summary()["trainable"] > before


def test_schedule_config_validation():
    with pytest.raises(ValueError):
        FreezeFirstConfig(adapt_steps=-1)
    with pytest.raises(ValueError):
        FreezeFirstConfig(base_lr=0.0)
    with pytest.raises(ValueError):
        FreezeFirstConfig(encoder_lr_scale=2.0)   # encoder must not out-run adapters
    with pytest.raises(ValueError):
        FreezeFirstConfig(unfreeze_blocks=-1)


def test_zero_adapt_steps_starts_directly_in_refine():
    rec, head, blocks = _schedule_model()
    sched = FreezeFirstSchedule(rec, blocks, FreezeFirstConfig(adapt_steps=0),
                                extra_trainable=[head])
    sched.step(rec(torch.randn(2, 20, 20)).sum())
    assert sched.phase == Phase.REFINE
