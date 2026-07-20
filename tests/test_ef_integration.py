"""Doc-12 stage 12h: end-to-end evaluation-framework integration + cycle stress."""

import torch

from signtranslator.eval_framework import (
    Direction, EvaluationChain, Layer, metric, to_contract,
    paired_permutation_pvalue, aggregate_seeds, significant_and_meaningful,
    PreRegistration, EvaluationFirewall,
    corpus_bleu, bert_score,
    comprehension_f1, SystemScores, preference_comprehension_dissociate,
    has_required_baselines, REQUIRED_BASELINES, ModelCard, MODEL_CARD_SECTIONS,
)


def test_full_evaluation_framework_pipeline():
    # 1) caveat-bound metrics across the stack -> falsifiable contracts.
    speech = to_contract(metric(Layer.SPEECH, "wer", 0.08), 0.15, Direction.LE)
    plan = to_contract(metric(Layer.PLAN, "graph_f1", 0.86), 0.8, Direction.GE)
    manual = to_contract(metric(Layer.MANUAL, "geo_err", 0.05), 0.1, Direction.LE)
    nonman = to_contract(metric(Layer.NONMANUAL, "f1", 0.83), 0.8, Direction.GE)
    human = to_contract(metric(Layer.HUMAN, "adequacy", 0.9), 0.8, Direction.GE)
    chain = EvaluationChain([speech, plan, manual, nonman, human])
    assert chain.adequate                                # all layers pass

    # 2) a single failing layer breaks adequacy even with a perfect speech score.
    broken = EvaluationChain([speech, to_contract(
        metric(Layer.NONMANUAL, "f1", 0.4), 0.8, Direction.GE)])
    assert not broken.adequate

    # 3) statistics: the PAIRED TEST runs over the many test-set items (large n),
    #    while the >=3 seeds provide the confidence interval. (A permutation test on
    #    only 3 seed-means cannot reach p<=0.05 -- its floor is 2/2^3 = 0.25.)
    per_item_ours = [0.90 + 0.01 * (i % 3) for i in range(24)]
    per_item_base = [x - 0.08 for x in per_item_ours]        # consistently better
    p = paired_permutation_pvalue(per_item_ours, per_item_base)
    effect = sum(per_item_ours) / 24 - sum(per_item_base) / 24
    assert significant_and_meaningful(effect, min_effect=0.02, pvalue=p)
    # seeds give the CI, not the significance test.
    summary = aggregate_seeds([0.90, 0.91, 0.89])
    assert summary.n_seeds == 3 and summary.ci_low <= summary.mean <= summary.ci_high

    # 4) pre-registration firewall: only registered primaries, no test tuning.
    prereg = PreRegistration.create(["comprehension_f1"],
                                    {"comprehension_f1": 0.05})
    fw = EvaluationFirewall(prereg)
    fw.select_hyperparameters("val")
    assert fw.endpoint_confirmed("comprehension_f1", effect=0.08, pvalue=0.01)

    # 5) reproducible text metrics + comprehension.
    score, sig = corpus_bleu(["the cat sat"], ["the cat sat"])
    assert score == 1.0 and "tok:" in str(sig)
    emb = torch.randn(4, 8, dtype=torch.float64)
    assert bert_score(emb, emb.clone())["f1"] > 0.999
    intended = {"agent=DOG", "action=RUN"}
    assert comprehension_f1(set(intended), intended) == 1.0

    # 6) preference != comprehension; baselines + model card required.
    assert preference_comprehension_dissociate(
        SystemScores("A", 0.8, 0.6), SystemScores("B", 0.4, 0.9))
    assert has_required_baselines({b: 0.5 for b in REQUIRED_BASELINES})
    card = ModelCard(model_size="1M", train_compute="1h", inference_compute="1",
                     latency="10ms", failure_modes=("occlusion",),
                     **{s: "x" for s in MODEL_CARD_SECTIONS})
    assert card.is_complete()


def test_cycle_stress_determinism():
    a = [0.9, 0.85, 0.88, 0.92]
    b = [0.8, 0.79, 0.83, 0.81]
    assert paired_permutation_pvalue(a, b) == paired_permutation_pvalue(a, b)
    s1 = corpus_bleu(["a b c"], ["a b d"])[0]
    s2 = corpus_bleu(["a b c"], ["a b d"])[0]
    assert s1 == s2
