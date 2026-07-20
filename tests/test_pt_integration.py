"""Doc-11 stage 11h: end-to-end pretraining integration + cycle stress."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from signtranslator.pretraining import (
    span_mask, is_certified_hard, worst_masked_floor,
    MaskedMotionModel, masked_token_nll,
    info_nce_against_negatives, signer_shortcut_embedding, content_embedding,
    hard_negative, is_minimal_linguistic_contrast,
    linear_probe_accuracy, chance_accuracy, signer_leakage_accuracy, is_leaky,
    loss_usefulness_dissociation,
    CURRICULUM, is_monotone_unlock, FrozenBaseline,
)
from signtranslator.grammar.grammar_tests import ControllableASLBuilder, GrammarFeatures


def test_full_pretraining_pipeline():
    torch.manual_seed(0)
    K, T, P = 16, 8, 2

    # 1) build a span mask over the (frame, part) grid and certify it is hard.
    m2d = span_mask(T, P, span_len=4, num_spans=1, seed=0)
    assert is_certified_hard(m2d, threshold=1.0)          # span interior > 1.0
    mask = torch.tensor(m2d.reshape(-1))

    # 2) masked motion modeling: a masked-only NLL that has a gradient.
    positions = torch.tensor([[t, p] for t in range(T) for p in range(P)])
    tokens = torch.randint(0, K, (T * P,))
    model = MaskedMotionModel(K, dim=16, max_frames=T, num_parts=P,
                              heads=2, enc_layers=1, dec_layers=1)
    logits = model(tokens, positions, mask)
    loss = masked_token_nll(logits, tokens, mask)
    loss.backward()
    assert model.head.weight.grad.abs().sum() > 0

    # 3) hard negatives from the oracle are genuine single-feature contrasts.
    builder = ControllableASLBuilder()
    base = GrammarFeatures(predicate=10, subject=1, object=2)
    assert is_minimal_linguistic_contrast(builder, base, "negated")
    assert hard_negative(base, "negated").negated != base.negated

    # 4) the signer shortcut fails on hard (signer-matched) negatives.
    anchor = signer_shortcut_embedding([0], 4)
    l_rand = info_nce_against_negatives(anchor, anchor.clone(),
                                        signer_shortcut_embedding([1, 2, 3], 4), 0.05)
    l_hard = info_nce_against_negatives(anchor, anchor.clone(),
                                        signer_shortcut_embedding([0, 0], 4), 0.05)
    assert float(l_rand) < float(l_hard)

    # 5) evidence: probe recovers content, leakage flagged, loss != usefulness.
    C, n = 4, 160
    y = torch.randint(0, C, (n,))
    good = F.one_hot(y, C).float() + 0.05 * torch.randn(n, C)
    assert linear_probe_accuracy(good[:80], y[:80], good[80:], y[80:], C) > 0.9
    leaky = F.one_hot(y, C).float()
    assert is_leaky(signer_leakage_accuracy(leaky, y.tolist(), C),
                    chance_accuracy(y.tolist()))
    diss = loss_usefulness_dissociation(seed=2)
    assert abs(diss["recon_loss_A"] - diss["recon_loss_B"]) < 1e-9
    assert diss["probe_acc_A"] - diss["probe_acc_B"] > 0.5

    # 6) curriculum monotone + frozen baseline retains stage-1 weights.
    assert is_monotone_unlock()
    fb = FrozenBaseline()
    fb.snapshot("stage1", model)
    assert fb.matches("stage1", model)


def test_cycle_stress_determinism():
    # the whole battery is deterministic given seeds (repeatable evidence).
    a = loss_usefulness_dissociation(seed=7)
    b = loss_usefulness_dissociation(seed=7)
    assert a == b

    m1 = span_mask(30, 2, span_len=6, num_spans=2, seed=3)
    m2 = span_mask(30, 2, span_len=6, num_spans=2, seed=3)
    assert np.array_equal(m1, m2)
    assert worst_masked_floor(m1) == worst_masked_floor(m2)


def test_curriculum_stage_count():
    assert len(CURRICULUM) == 5
