"""Cycle-level integration and stress test for the semantic reasoning layer.

The full chain is exercised end to end:

    evidence (transcript-like tokens) -> SemanticPlanner (constrained) ->
    typed plan -> validate -> ground against a versioned lexicon ->
    manual_units as gloss -> the existing motion generator

plus the ablations the specification asks for (retrieval-off, LLM-off, with
hallucinated-entry and invalid-spatial-reference reporting) and the usual
adversarial / determinism / dtype stress.
"""

import pytest
import torch

from signtranslator.planning import (
    PlanVocabulary, SignPlan, SemanticFrame, serialize_plan, deserialize_plan,
    validate_plan, SemanticPlanner, pad_plan_batch, SchemaAutomaton,
    ConstrainedDecoder, SignLexicon, LexEntry, ground_plan,
)
from signtranslator.planning.consistency import (
    ControllablePlanBuilder, SemanticFeatures, check_semantic_consistency,
)
from signtranslator.models import BidirectionalSignTranslator
from signtranslator import ModelConfig, DiffusionConfig

VOCAB = PlanVocabulary(num_predicates=4, num_roles=2, num_referents=3, num_tam=3,
                       num_loci=4, num_lexemes=6, num_nonmanual=3, max_units=5,
                       num_conf_buckets=4)


def _lexicon(lexemes, dim=8, seed=0, version="1.0.0"):
    g = torch.Generator().manual_seed(seed)
    entries = [LexEntry(lexeme=l, gloss=f"S{l}",
                        embedding=tuple(torch.randn(dim, generator=g).tolist()))
               for l in lexemes]
    return SignLexicon(entries, version=version)


def _codes():
    """Fixed evidence-code -> plan mapping, all units drawn from lexemes {0,1,2}."""
    return [
        (0, SignPlan(frame=SemanticFrame(0, [(0, 0)]), referents=[0], loci={0: 0},
                     manual_units=[0, 1], tam=0, conf_bucket=3)),
        (1, SignPlan(frame=SemanticFrame(1, [(0, 1)]), referents=[1], loci={1: 1},
                     manual_units=[2, 0], tam=1, conf_bucket=3)),
        (2, SignPlan(frame=SemanticFrame(2, []), referents=[0, 1],
                     loci={0: 0, 1: 2}, manual_units=[1, 2, 0], tam=2, conf_bucket=3)),
    ]


@pytest.fixture(scope="module")
def trained_planner():
    torch.manual_seed(0)
    planner = SemanticPlanner(VOCAB, acoustic_dim=None, src_vocab=4, d_model=64,
                              num_encoder_layers=2, num_decoder_layers=2)
    codes = _codes()
    src = torch.tensor([[c] for c, _ in codes])
    tokens = pad_plan_batch([serialize_plan(p, VOCAB) for _, p in codes],
                            planner.pad_id)
    opt = torch.optim.Adam(planner.parameters(), lr=3e-3)
    for _ in range(300):
        loss = planner.plan_nll(tokens, src_tokens=src)
        opt.zero_grad(); loss.backward(); opt.step()
    planner.eval()
    return planner, codes


# ---------------------------------------------------------------------------
# Full chain
# ---------------------------------------------------------------------------
def test_evidence_to_plan_is_exact_after_training(trained_planner):
    planner, codes = trained_planner
    for code, target in codes:
        tokens = planner.generate(src_tokens=torch.tensor([[code]]))
        plan = deserialize_plan(tokens, VOCAB)
        assert tokens == serialize_plan(target, VOCAB)
        assert validate_plan(plan, VOCAB) == []


def test_plan_grounds_against_the_lexicon(trained_planner):
    planner, codes = trained_planner
    lexicon = _lexicon([0, 1, 2])                 # covers every unit used
    for code, _ in codes:
        plan = deserialize_plan(planner.generate(src_tokens=torch.tensor([[code]])),
                                VOCAB)
        report = ground_plan(plan, lexicon, num_loci=VOCAB.num_loci)
        assert report.is_grounded, report.summary()


def test_plan_manual_units_drive_the_motion_generator(trained_planner):
    """The cross-layer link: a plan's manual units become gloss for the generator."""
    planner, codes = trained_planner
    mcfg = ModelConfig(num_joints=27, num_frames=16, stgcn_channels=(16, 32),
                       text_embed_dim=32, text_layers=2, text_heads=2, latent_dim=16,
                       speech_input_dim=40)
    dcfg = DiffusionConfig(num_timesteps=20, denoiser_dim=32, denoiser_layers=2,
                           denoiser_heads=2)
    motion = BidirectionalSignTranslator(mcfg, dcfg, src_vocab=64,
                                         gloss_vocab=64, num_glosses=16)
    plan = deserialize_plan(planner.generate(src_tokens=torch.tensor([[2]])), VOCAB)
    # manual units (lexeme ids) act as gloss tokens for the motion generator.
    gloss = torch.tensor([[u + 1 for u in plan.manual_units]])   # +1: reserve 0=PAD
    out = motion.generate_from_gloss(gloss, num_frames=12, ddim_steps=3)
    assert out.shape == (1, 3, 12, 27)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Ablations required by the spec
# ---------------------------------------------------------------------------
def test_retrieval_off_reports_hallucinations(trained_planner):
    """A lexicon MISSING an emitted unit must flag it as hallucinated."""
    planner, codes = trained_planner
    plan = deserialize_plan(planner.generate(src_tokens=torch.tensor([[2]])), VOCAB)
    assert plan.manual_units                         # non-empty
    partial_lexicon = _lexicon([plan.manual_units[0]])   # only the first unit
    report = ground_plan(plan, partial_lexicon, num_loci=VOCAB.num_loci)
    # any unit beyond the first (and not fingerspelled) is a hallucination
    if len(set(plan.manual_units)) > 1:
        assert not report.is_grounded
        assert report.hallucination_rate > 0.0


def test_llm_off_baseline_hallucinates_more_than_the_trained_planner():
    """LLM-off ablation: random constrained plans vs the trained planner.

    The trained planner (fit on in-lexicon data) should emit far fewer
    out-of-lexicon units than an untrained model constrained-decoding at random.
    """
    torch.manual_seed(0)
    lexicon = _lexicon([0, 1, 2])                    # only lexemes 0,1,2 are real
    # trained planner on in-lexicon data
    codes = _codes()
    planner = SemanticPlanner(VOCAB, acoustic_dim=None, src_vocab=4, d_model=64,
                              num_decoder_layers=2)
    src = torch.tensor([[c] for c, _ in codes])
    tokens = pad_plan_batch([serialize_plan(p, VOCAB) for _, p in codes], planner.pad_id)
    opt = torch.optim.Adam(planner.parameters(), lr=3e-3)
    for _ in range(300):
        loss = planner.plan_nll(tokens, src_tokens=src)
        opt.zero_grad(); loss.backward(); opt.step()
    planner.eval()

    def halluc_rate(plans):
        tot = flagged = 0
        for plan in plans:
            r = ground_plan(plan, lexicon, num_loci=VOCAB.num_loci)
            tot += r.num_units; flagged += len(r.hallucinated_units)
        return flagged / max(tot, 1)

    trained_plans = [deserialize_plan(planner.generate(src_tokens=torch.tensor([[c]])),
                                      VOCAB) for c, _ in codes]
    # LLM-off: an untrained decoder produces arbitrary (but well-formed) plans
    random_planner = SemanticPlanner(VOCAB, acoustic_dim=None, src_vocab=4,
                                     d_model=32, num_decoder_layers=2)
    g = torch.Generator().manual_seed(7)
    random_plans = [deserialize_plan(
        random_planner.generate(src_tokens=torch.tensor([[c % 4]]),
                                sample=True, generator=g), VOCAB)
        for c in range(12)]

    assert halluc_rate(trained_plans) <= halluc_rate(random_plans)


# ---------------------------------------------------------------------------
# Stress: constrained decoding always valid, determinism, dtype
# ---------------------------------------------------------------------------
def test_constrained_generation_is_always_wellformed_across_seeds():
    """Even untrained, over many seeds, every generated plan parses."""
    for seed in range(20):
        torch.manual_seed(seed)
        planner = SemanticPlanner(VOCAB, acoustic_dim=None, src_vocab=4, d_model=24,
                                  num_decoder_layers=1)
        tokens = planner.generate(src_tokens=torch.tensor([[seed % 4]]))
        assert planner.automaton.accepts(tokens)
        deserialize_plan(tokens, VOCAB)              # never raises


def test_generation_is_deterministic_in_greedy_mode(trained_planner):
    planner, _ = trained_planner
    a = planner.generate(src_tokens=torch.tensor([[1]]))
    b = planner.generate(src_tokens=torch.tensor([[1]]))
    assert a == b


def test_planner_works_in_double_precision():
    torch.manual_seed(0)
    planner = SemanticPlanner(VOCAB, acoustic_dim=None, src_vocab=4, d_model=32,
                              num_decoder_layers=1).double()
    tokens = planner.generate(src_tokens=torch.tensor([[0]]))
    assert planner.automaton.accepts(tokens)


def test_serialization_roundtrip_stress():
    """Random valid plans survive serialize -> deserialize -> serialize."""
    import random
    from tests.test_planning_schema import _random_valid_plan
    rng = random.Random(0)
    for _ in range(300):
        plan = _random_valid_plan(rng, VOCAB)
        tokens = serialize_plan(plan, VOCAB)
        assert SchemaAutomaton(VOCAB).accepts(tokens)
        assert serialize_plan(deserialize_plan(tokens, VOCAB), VOCAB) == tokens


def test_controllable_builder_plus_planner_consistency():
    """A plan the controllable builder makes must pass both validator and
    the semantic-consistency checks -- the two are complementary."""
    builder = ControllablePlanBuilder(VOCAB)
    for feats in (SemanticFeatures(predicate=0, agent=0, patient=1),
                  SemanticFeatures(predicate=1, agent=0, negated=True),
                  SemanticFeatures(predicate=2, agent=1, question="wh")):
        plan = builder.build(feats)
        assert validate_plan(plan, VOCAB) == []
        assert check_semantic_consistency(plan, VOCAB).is_consistent
