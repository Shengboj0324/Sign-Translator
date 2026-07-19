"""Verification of the semantic planner model.

Includes the standard autoregressive checks (causal masking, NLL correctness)
and the decisive integration test: overfit a controllable evidence->plan mapping
and confirm *constrained* generation reproduces the exact target plan.
"""

import pytest
import torch
import torch.nn.functional as F

from signtranslator.planning.schema import (
    PlanVocabulary, SignPlan, SemanticFrame, NonmanualSpan,
    serialize_plan, deserialize_plan, validate_plan,
)
from signtranslator.planning.planner import SemanticPlanner, pad_plan_batch

# A small vocab keeps the model and the plans compact.
VOCAB = PlanVocabulary(num_predicates=3, num_roles=2, num_referents=3, num_tam=3,
                       num_loci=4, num_lexemes=6, num_nonmanual=2, max_units=4,
                       num_conf_buckets=4)


def _plan(pred, refs, loci, units, tam=0, conf=1):
    return SignPlan(frame=SemanticFrame(predicate=pred),
                    referents=list(refs), loci=dict(loci),
                    manual_units=list(units), tam=tam, conf_bucket=conf)


def _acoustic(n, t=6, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, t, d, generator=g)


# ---------------------------------------------------------------------------
# Construction / shapes
# ---------------------------------------------------------------------------
def test_planner_requires_an_evidence_source():
    with pytest.raises(ValueError):
        SemanticPlanner(VOCAB, acoustic_dim=None, src_vocab=None)


def test_forward_produces_logits_over_plan_vocab():
    planner = SemanticPlanner(VOCAB, acoustic_dim=16, d_model=32,
                              num_encoder_layers=1, num_decoder_layers=1)
    plan = serialize_plan(_plan(0, [0], {0: 0}, [1, 2]), VOCAB)
    tokens = pad_plan_batch([plan], planner.pad_id)
    logits = planner(tokens, acoustic=_acoustic(1))
    assert logits.shape == (1, tokens.shape[1], VOCAB.size)
    assert torch.isfinite(logits).all()


def test_output_head_never_predicts_sos_or_pad():
    """The head is sized to the plan vocab only; SOS/PAD are input-only ids."""
    planner = SemanticPlanner(VOCAB, acoustic_dim=16, d_model=32)
    assert planner.out.out_features == VOCAB.size
    assert planner.sos_id == VOCAB.size and planner.pad_id == VOCAB.size + 1


# ---------------------------------------------------------------------------
# Teacher forcing / causal masking
# ---------------------------------------------------------------------------
def test_decoder_is_causal_no_future_leakage():
    """Changing a future target token must not change earlier-position logits."""
    torch.manual_seed(0)
    planner = SemanticPlanner(VOCAB, acoustic_dim=16, d_model=32,
                              num_decoder_layers=2).eval()
    acoustic = _acoustic(1)
    plan = serialize_plan(_plan(1, [0, 1], {0: 0, 1: 1}, [1, 2, 3]), VOCAB)
    tokens = pad_plan_batch([plan], planner.pad_id)
    base = planner(tokens, acoustic=acoustic)

    altered = tokens.clone()
    altered[0, -2] = (altered[0, -2] + 1) % VOCAB.size     # change a late token
    alt = planner(altered, acoustic=acoustic)
    # positions strictly before the change must be identical
    assert torch.allclose(base[:, :-2], alt[:, :-2], atol=1e-5)


def test_plan_nll_matches_manual_cross_entropy():
    planner = SemanticPlanner(VOCAB, acoustic_dim=16, d_model=32).eval()
    acoustic = _acoustic(2)
    plans = [serialize_plan(_plan(0, [0], {0: 0}, [1]), VOCAB),
             serialize_plan(_plan(1, [1], {1: 2}, [2, 3, 4]), VOCAB)]
    tokens = pad_plan_batch(plans, planner.pad_id)
    logits = planner(tokens, acoustic=acoustic)
    manual = F.cross_entropy(logits.reshape(-1, VOCAB.size), tokens.reshape(-1),
                             ignore_index=planner.pad_id)
    got = planner.plan_nll(tokens, acoustic=acoustic)
    assert torch.allclose(got, manual, atol=1e-6)


def test_padding_is_ignored_by_the_loss():
    """Two batches differing only in padding length give the same NLL."""
    torch.manual_seed(0)
    planner = SemanticPlanner(VOCAB, acoustic_dim=16, d_model=32).eval()
    acoustic = _acoustic(1)
    plan = serialize_plan(_plan(0, [0], {0: 0}, [1]), VOCAB)
    tight = pad_plan_batch([plan], planner.pad_id)
    padded = torch.cat([tight, torch.full((1, 3), planner.pad_id)], dim=1)
    a = planner.plan_nll(tight, acoustic=acoustic)
    b = planner.plan_nll(padded, acoustic=acoustic)
    assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------------
# Evidence actually conditions the output
# ---------------------------------------------------------------------------
def test_acoustic_evidence_changes_the_logits():
    """If the acoustic prefix were ignored, conditioning would be a lie."""
    torch.manual_seed(0)
    planner = SemanticPlanner(VOCAB, acoustic_dim=16, d_model=32).eval()
    plan = serialize_plan(_plan(0, [0], {0: 0}, [1, 2]), VOCAB)
    tokens = pad_plan_batch([plan], planner.pad_id)
    a = planner(tokens, acoustic=_acoustic(1, seed=1))
    b = planner(tokens, acoustic=_acoustic(1, seed=2))
    assert not torch.allclose(a, b, atol=1e-4)


def test_source_token_pathway_also_conditions():
    planner = SemanticPlanner(VOCAB, acoustic_dim=None, src_vocab=10, d_model=32).eval()
    plan = serialize_plan(_plan(0, [0], {0: 0}, [1]), VOCAB)
    tokens = pad_plan_batch([plan], planner.pad_id)
    a = planner(tokens, src_tokens=torch.tensor([[1, 2, 3]]))
    b = planner(tokens, src_tokens=torch.tensor([[4, 5, 6]]))
    assert not torch.allclose(a, b, atol=1e-4)


# ---------------------------------------------------------------------------
# Constrained generation
# ---------------------------------------------------------------------------
def test_generation_always_yields_a_wellformed_plan():
    """Even an untrained model must emit a parseable plan (the automaton forces it)."""
    torch.manual_seed(0)
    planner = SemanticPlanner(VOCAB, acoustic_dim=16, d_model=32)
    for seed in range(5):
        tokens = planner.generate(acoustic=_acoustic(1, seed=seed))
        assert planner.automaton.accepts(tokens)
        deserialize_plan(tokens, VOCAB)              # never raises


def test_generate_requires_batch_size_one():
    planner = SemanticPlanner(VOCAB, acoustic_dim=16, d_model=32)
    with pytest.raises(ValueError):
        planner.generate(acoustic=_acoustic(2))


# ---------------------------------------------------------------------------
# THE integration test: overfit a controllable mapping
# ---------------------------------------------------------------------------
def test_planner_overfits_a_controllable_evidence_to_plan_mapping():
    """A fixed evidence code must map to a fixed plan, recovered by generation.

    Evidence is a discrete source token; each code has one target plan. After
    training the plan NLL, constrained generation must reproduce the exact target
    serialization for each code -- the end-to-end statement that the model learns
    evidence -> plan, not merely that a loss fell.
    """
    torch.manual_seed(0)
    codes = [
        (0, _plan(0, [0], {0: 0}, [1, 2])),
        (1, _plan(1, [1], {1: 1}, [3])),
        (2, _plan(2, [0, 1], {0: 0, 1: 2}, [4, 5, 1])),
    ]
    planner = SemanticPlanner(VOCAB, acoustic_dim=None, src_vocab=4, d_model=64,
                              num_encoder_layers=2, num_decoder_layers=2)
    src = torch.tensor([[c] for c, _ in codes])
    plans = [serialize_plan(p, VOCAB) for _, p in codes]
    tokens = pad_plan_batch(plans, planner.pad_id)

    opt = torch.optim.Adam(planner.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        loss = planner.plan_nll(tokens, src_tokens=src)
        first = first if first is not None else float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < first * 0.05

    for code, target_plan in codes:
        generated = planner.generate(src_tokens=torch.tensor([[code]]))
        assert generated == serialize_plan(target_plan, VOCAB), (
            f"code {code}: generated plan != target")
        assert validate_plan(deserialize_plan(generated, VOCAB), VOCAB) == []


def test_pad_plan_batch_shapes_and_padding():
    out = pad_plan_batch([[1, 2, 3], [4, 5]], pad_id=99)
    assert out.shape == (2, 3)
    assert out[1].tolist() == [4, 5, 99]
    with pytest.raises(ValueError):
        pad_plan_batch([], pad_id=0)
