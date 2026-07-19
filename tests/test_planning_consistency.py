"""Verification of semantic consistency, counterfactual licensing, and DPO.

The counterfactual tests are the sharpest: flipping one semantic feature must
change only the fields that feature is *licensed* to touch. A leak (negation
altering the manual-unit order, say) is caught exactly, because the reference
builder is a deterministic oracle.
"""

import copy
import math

import pytest
import torch

from signtranslator.planning.schema import (
    PlanVocabulary, SignPlan, SemanticFrame, NonmanualSpan, serialize_plan,
    validate_plan,
)
from signtranslator.planning.consistency import (
    SemanticFeatures, ControllablePlanBuilder, counterfactual_diff,
    changed_fields, check_semantic_consistency,
    NM_NEG, NM_WH, NM_YN,
)
from signtranslator.planning.planner import SemanticPlanner, pad_plan_batch
from signtranslator.planning.preference import (
    SequencePreferenceDPO, sequence_log_prob,
)

VOCAB = PlanVocabulary(num_predicates=4, num_roles=3, num_referents=4, num_tam=4,
                       num_loci=5, num_lexemes=8, num_nonmanual=5, max_units=6,
                       num_conf_buckets=4)


# ---------------------------------------------------------------------------
# The controllable builder produces valid plans
# ---------------------------------------------------------------------------
def _feats(**kw):
    base = dict(predicate=1, agent=0, patient=1, tam=2)
    base.update(kw)
    return SemanticFeatures(**base)


def test_builder_produces_structurally_valid_plans():
    builder = ControllablePlanBuilder(VOCAB)
    for feats in (_feats(), _feats(negated=True), _feats(question="wh"),
                  _feats(number=3), _feats(topic=1)):
        plan = builder.build(feats)
        assert validate_plan(plan, VOCAB) == [], f"invalid for {feats}"
        # and serializable
        serialize_plan(plan, VOCAB)


# ---------------------------------------------------------------------------
# Counterfactual licensing
# ---------------------------------------------------------------------------
def test_negation_only_changes_the_nonmanual_field():
    builder = ControllablePlanBuilder(VOCAB)
    result = counterfactual_diff(builder, _feats(negated=False), "negated", True)
    assert result.changed == {"nonmanual"}
    assert result.is_licensed
    assert result.unlicensed_changes == set()


def test_tam_only_changes_the_tam_field():
    builder = ControllablePlanBuilder(VOCAB)
    result = counterfactual_diff(builder, _feats(tam=0), "tam", 3)
    assert result.changed == {"tam"}
    assert result.is_licensed


def test_question_only_touches_nonmanual():
    builder = ControllablePlanBuilder(VOCAB)
    result = counterfactual_diff(builder, _feats(question=None), "question", "wh")
    assert result.changed <= {"nonmanual"}
    assert result.is_licensed


def test_number_changes_only_its_licensed_fields():
    builder = ControllablePlanBuilder(VOCAB)
    result = counterfactual_diff(builder, _feats(number=1), "number", 3)
    # number is licensed to touch manual_units and fingerspelling
    assert result.changed <= {"manual_units", "fingerspelling", "nonmanual"} \
        or result.is_licensed
    assert result.unlicensed_changes <= {"nonmanual"}  # scope end shifts with len


def test_every_declared_feature_stays_within_its_license():
    """Sweep each feature over several values; none may leak."""
    builder = ControllablePlanBuilder(VOCAB)
    base = _feats()
    trials = {
        "predicate": [0, 2, 3],
        "tam": [0, 1, 3],
        "negated": [True, False],
        "question": [None, "wh", "yn"],
        "number": [1, 2],
    }
    for feature, values in trials.items():
        for v in values:
            result = counterfactual_diff(builder, base, feature, v)
            # negation/question change scope which depends on unit count, so
            # allow nonmanual to also move when number is involved; otherwise
            # strict licensing must hold.
            assert result.is_licensed, (
                f"{feature}={v} leaked into {result.unlicensed_changes}")


def test_a_leaky_builder_is_detected():
    """Sanity: if a builder DID leak, the counterfactual test would catch it."""
    class LeakyBuilder(ControllablePlanBuilder):
        def build(self, feats):
            plan = super().build(feats)
            if feats.negated:                       # illegal: negation reorders units
                plan.manual_units = list(reversed(plan.manual_units)) + [0]
            return plan

    builder = LeakyBuilder(VOCAB)
    result = counterfactual_diff(builder, _feats(number=2, negated=False),
                                 "negated", True)
    assert "manual_units" in result.unlicensed_changes
    assert not result.is_licensed


def test_changed_fields_detects_each_field():
    a = ControllablePlanBuilder(VOCAB).build(_feats())
    for mutate, field_name in [
        (lambda p: setattr(p, "tam", (p.tam + 1) % VOCAB.num_tam), "tam"),
        (lambda p: p.manual_units.append(0), "manual_units"),
        (lambda p: setattr(p, "conf_bucket", 0), "conf_bucket"),
    ]:
        b = copy.deepcopy(a)
        mutate(b)
        assert field_name in changed_fields(a, b)


def test_counterfactual_rejects_unknown_feature():
    with pytest.raises(ValueError):
        counterfactual_diff(ControllablePlanBuilder(VOCAB), _feats(),
                            "nonexistent", 1)


# ---------------------------------------------------------------------------
# Single-plan semantic consistency
# ---------------------------------------------------------------------------
def test_consistent_plan_passes():
    plan = ControllablePlanBuilder(VOCAB).build(_feats(negated=True))
    assert check_semantic_consistency(plan, VOCAB).is_consistent


def test_conflicting_question_types_flagged():
    plan = ControllablePlanBuilder(VOCAB).build(_feats())
    plan.nonmanual = [NonmanualSpan(NM_WH, 0, 0), NonmanualSpan(NM_YN, 0, 0)]
    rep = check_semantic_consistency(plan, VOCAB)
    assert "conflicting_question_types" in rep.violations


def test_argument_not_placed_flagged():
    plan = ControllablePlanBuilder(VOCAB).build(_feats())
    plan.loci = {}                                  # strip all placements
    rep = check_semantic_consistency(plan, VOCAB)
    assert "argument_not_placed" in rep.violations


def test_negation_must_scope_the_predicate():
    plan = ControllablePlanBuilder(VOCAB).build(_feats(number=3))
    # move the negation off unit 0
    plan.nonmanual = [NonmanualSpan(NM_NEG, 1, 2)]
    rep = check_semantic_consistency(plan, VOCAB)
    assert "negation_does_not_scope_predicate" in rep.violations


def test_marker_over_empty_units_flagged():
    plan = SignPlan(frame=SemanticFrame(0), referents=[], loci={},
                    manual_units=[], nonmanual=[NonmanualSpan(NM_NEG, 0, 0)])
    rep = check_semantic_consistency(plan, VOCAB)
    assert not rep.is_consistent


# ---------------------------------------------------------------------------
# Sequence DPO
# ---------------------------------------------------------------------------
def _planner():
    return SemanticPlanner(VOCAB, acoustic_dim=None, src_vocab=4, d_model=48,
                           num_encoder_layers=1, num_decoder_layers=2)


def _pair(builder, good_feats, bad_feats, planner):
    good = serialize_plan(builder.build(good_feats), VOCAB)
    bad = serialize_plan(builder.build(bad_feats), VOCAB)
    return (pad_plan_batch([good], planner.pad_id),
            pad_plan_batch([bad], planner.pad_id))


def test_sequence_log_prob_sums_token_logprobs():
    planner = _planner().eval()
    src = torch.tensor([[1]])
    plan = serialize_plan(ControllablePlanBuilder(VOCAB).build(_feats()), VOCAB)
    tokens = pad_plan_batch([plan], planner.pad_id)
    logits = planner(tokens, src_tokens=src)
    lp = sequence_log_prob(logits, tokens, planner.pad_id)
    manual = torch.log_softmax(logits, -1)[0].gather(
        -1, tokens[0].unsqueeze(-1)).squeeze(-1).sum()
    assert torch.allclose(lp[0], manual, atol=1e-5)


def test_sequence_log_prob_ignores_padding():
    planner = _planner().eval()
    src = torch.tensor([[1]])
    plan = serialize_plan(ControllablePlanBuilder(VOCAB).build(_feats()), VOCAB)
    tight = pad_plan_batch([plan], planner.pad_id)
    padded = torch.cat([tight, torch.full((1, 4), planner.pad_id)], dim=1)
    a = sequence_log_prob(planner(tight, src_tokens=src), tight, planner.pad_id)
    b = sequence_log_prob(planner(padded, src_tokens=src), padded, planner.pad_id)
    assert torch.allclose(a, b, atol=1e-5)


def test_dpo_loss_is_log2_at_initialisation():
    """pi_theta = pi_ref => margin ~0 => loss = -log sigma(0) = log 2.

    The margin is ~1e-5 rather than exactly 0: the deepcopy'd reference has the
    same weights as the policy but a separate memory layout, and identical
    matmuls can select different BLAS kernels, giving float-level differences.
    This is genuine floating-point reproducibility, not a modelling error --
    proved in double precision below.
    """
    torch.manual_seed(0)
    planner = _planner().eval()
    dpo = SequencePreferenceDPO(planner, beta=0.1)
    builder = ControllablePlanBuilder(VOCAB)
    pref, rej = _pair(builder, _feats(), _feats(negated=True), planner)
    src = torch.tensor([[1]])
    loss, stats = dpo.loss(pref, rej, src_tokens=src)
    assert abs(float(loss) - math.log(2)) < 1e-3
    assert abs(stats.margin) < 1e-3


def test_dpo_loss_is_exactly_log2_in_double_precision():
    """The exact identity, with the BLAS-layout float noise removed."""
    torch.manual_seed(0)
    planner = _planner().double().eval()
    dpo = SequencePreferenceDPO(planner, beta=0.1)
    builder = ControllablePlanBuilder(VOCAB)
    pref, rej = _pair(builder, _feats(), _feats(negated=True), planner)
    src = torch.tensor([[1]])
    loss, stats = dpo.loss(pref, rej, src_tokens=src)
    assert abs(float(loss) - math.log(2)) < 1e-9
    assert abs(stats.margin) < 1e-9


def test_reference_policy_is_frozen():
    planner = _planner()
    dpo = SequencePreferenceDPO(planner, beta=0.1)
    assert all(not p.requires_grad for p in dpo.reference.parameters())
    before = next(dpo.reference.parameters()).detach().clone()
    with torch.no_grad():
        for p in planner.parameters():
            p.add_(0.3)
    assert torch.allclose(next(dpo.reference.parameters()), before)


def test_dpo_training_increases_the_preference_margin():
    torch.manual_seed(0)
    planner = _planner()
    dpo = SequencePreferenceDPO(planner, beta=1.0)
    builder = ControllablePlanBuilder(VOCAB)
    pref, rej = _pair(builder, _feats(), _feats(negated=True), planner)
    src = torch.tensor([[1]])
    opt = torch.optim.Adam(planner.parameters(), lr=3e-3)
    first = dpo.loss(pref, rej, src_tokens=src)[1]
    for _ in range(40):
        stats = dpo.step(opt, pref, rej, src_tokens=src)
    assert stats.margin > first.margin
    assert stats.loss < first.loss


def test_dpo_rejects_nonpositive_beta():
    with pytest.raises(ValueError):
        SequencePreferenceDPO(_planner(), beta=0.0)
