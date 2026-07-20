# 12 — Evaluation Framework — Design and Mathematics

This document fixes **all** mathematics of the evaluation framework before any code,
in the discipline of docs 01–11. It implements `12_evaluation_framework.md`: a chain
of **falsifiable contracts** across seven metric layers, each with a mandatory
caveat; a rigorous statistical protocol (pre-registration, held-out signers, paired
tests, CIs, minimum meaningful effect); reproducible text metrics (SacreBLEU,
BERTScore); blinded comprehension scoring; and a model card. The governing principle
is that **BLEU / WER / FID / MPJPE alone cannot establish that an avatar conveys the
intended sign-language message** — a single metric is necessary, never sufficient.

**Reuse.** This is a *framework* layer that aggregates the audited leaf metrics; it
does not reimplement them:

* `pose/rotations.geodesic_distance` — the document's `d_geo(R,R̂)=arccos((tr(RᵀR̂)−1)/2)`.
* `pose/metrics` (MPJPE, PA-MPJPE, fingertip-weighted), `hand_graph/metrics`
  (fingertip error in hand scale, `contact_prf`, `collision_rate`).
* `speech/evaluation` (WER/CER with S/D/I, timestamp error, `percentile`),
  `speech/calibration` (ECE, Brier, reliability).
* `grammar/signbleu` (`sign_bleu`, `cohens_kappa`, `fleiss_kappa`).
* `facial_nmm/evaluation` (multilabel F1, scope-boundary error, gaze agreement).
* `avatar_render/evaluation` (motion-to-photon p95, flicker, penetration, silhouette
  IoU, PSNR/SSIM), `diffusion_gen/evaluation` (jerk, generation-time p95).
* `pretraining/evaluation` (linear probes, cross-signer retrieval, dissociation),
  `data_engineering/splitting.grouped_split` (signer/source held-out),
  `data_engineering/datasheet` (the datasheet analogue for the model card).

Primary sources studied:

* Kim et al., *SignBLEU* (arXiv:2406.06648) — multi-channel symbolic metric; valuable
  **only** when multi-channel annotations exist, never a substitute for signer
  evaluation (already in Doc-03).
* Duarte et al., *How2Sign* (arXiv:2008.08143) — a signer study protocol.
* Post, *A Call for Clarity in Reporting BLEU / SacreBLEU* (W18-6319) — a canonical
  tokenisation + a version **signature** so scores are reproducible/comparable.
* Zhang et al., *BERTScore* (arXiv:1904.09675) — greedy token-embedding matching,
  precision/recall/F1, optional IDF weighting.
* Mitchell et al., *Model Cards* (arXiv:1810.03993) — model documentation sections.

## 0. Honest scope (read first)

No real signing systems, human raters, or learned FID embeddings exist here; the
framework's **logic and mathematics** are implemented and validated on controllable
synthetic inputs with known ground truth. Every property independent of a trained
model or human panel — the contract-chain conjunction, the caveat binding, the
statistics (permutation, bootstrap, sign test), the SacreBLEU/BERTScore algebra, the
comprehension F1, the pre-registration firewall, the model-card completeness — is
proved exactly. Human-panel scores are *instrumented* (schemas + reliability), not
performed.

## 1. Falsifiable contract chain (innovation)

A **contract** is `(name, layer, value, threshold, direction, caveat)`: it *passes*
iff the metric meets its threshold in the required direction (`ge` or `le`). A
**chain** over the metric stack is *adequate* iff **every** contract passes:

    adequate(chain) = ⋀_c passed(c).

**Single-metric insufficiency (proved).** A chain with one passing layer and one
failing layer is **not** adequate — so a high BLEU (or WER, FID, MPJPE) cannot on its
own certify the system. This is the document's principle turned into a theorem: the
conjunction is monotone (adding a failing contract can only lower adequacy) and no
single contract implies the conjunction.

## 2. Caveat-bound metric stack (innovation)

Each metric result carries its **mandatory caveat** from the document's table; a
result constructed without a caveat **raises** (a structural guard, extending Doc-08's
appearance/signing separation). The seven layers and their required caveats:

| Layer | example metrics | required caveat |
|---|---|---|
| speech | WER/CER, timestamp MAE, calibration, revision rate | transcript accuracy ≠ sign adequacy |
| plan | slot/graph F1, negation/question/scope acc, referent consistency | gloss agreement is annotation-dependent |
| manual | rotation geodesic, hand-scale fingertip error, velocity/jerk, contact F1 | one valid production may differ from reference |
| non-manual | multilabel F1, interval IoU/boundary error, gaze agreement | landmark accuracy ≠ grammatical accuracy |
| distribution | retrieval precision, diversity, FID-like | embedding choice can bias results |
| rendering | FPS, p95 motion-to-photon, flicker, collision, LPIPS | appearance quality ≠ comprehension |
| human | adequacy, grammaticality, naturalness, intelligibility, preference | use fluent target-language signers |

So a metric can never be reported stripped of its caveat — "transcript accuracy = sign
adequacy" is made unstatable.

## 3. Statistical rigor

* **Paired differences** `d_i = a_i − b_i`, mean `d̄`, sample sd `s` (ddof=1);
  **paired t-statistic** `t = d̄ / (s/√n)` (descriptive).
* **Paired permutation test** — under the symmetric null, each `d_i` may flip sign;
  the two-sided p-value is the fraction of the `2^n` sign-flips whose `|mean|` ≥
  `|d̄|` (exact enumeration for small `n`, sampled otherwise). Self-contained (no
  special functions), proved on a hand-computable case.
* **Sign test** — exact two-sided binomial tail on the count of positive differences
  (`p=0.5`), via exact `comb`.
* **Percentile bootstrap CI** — resample indices with replacement `B` times, take the
  `[α/2, 1−α/2]` percentiles of the statistic. A constant sample yields a degenerate CI.
* **Multi-seed aggregation** — mean ± bootstrap CI across ≥3 seeds.
* **Minimum meaningful effect** — a result counts only if the observed effect ≥ the
  pre-registered minimum AND the test is significant; either alone is insufficient.

## 4. Pre-registration lock + test-set firewall (innovation)

`PreRegistration(primary_endpoints, min_effects)` is **hash-locked** before any test
access. The `TestFirewall`:

* refuses hyperparameter selection on the test split (`select_on='test'` raises;
  `'val'` is allowed) — "no hyperparameter selection on test";
* refuses to report a **non-registered** endpoint as primary;
* uses the Doc-10 `grouped_split` so the test set is **signer/source-held-out**.

The protocol is enforced in code, not merely documented.

## 5. Reproducible text metrics

* **SacreBLEU** — corpus BLEU `= BP · exp(Σ_{n=1}^N (1/N) log p_n)`, clipped n-gram
  precision `p_n`, brevity penalty `BP = min(1, exp(1 − r/c))` (`c` = hyp length, `r`
  = ref length), exponential-decay smoothing for zero counts. A **signature** string
  records tokeniser + smoothing + max-n + #refs so a score is reproducible/comparable.
  Proved: identical hyp==ref → 1.0; a short hyp is brevity-penalised; the signature is
  deterministic.
* **BERTScore** — with unit-norm reference embeddings `x_i` and candidate `x̂_j`,
  `R = (1/|x|)Σ_i max_j x_i·x̂_j`, `P = (1/|x̂|)Σ_j max_i x_i·x̂_j`, `F = 2PR/(P+R)`;
  optional IDF weights replace the means with weighted means. Proved: identical
  embeddings → `P=R=F=1`; token-order permutation invariance; IDF re-weights emphasis.

Both are **automatic** metrics; the framework attaches the caveat that they do not
substitute for signer evaluation.

## 6. Blinded comprehension + preference dissociation (innovation)

For a generated answer `g` and an intended proposition set `P`, blinded raters recover
a proposition set `R(g)`; the adequacy endpoint is

    precision = |R∩P|/|R|,  recall = |R∩P|/|P|,  F1 = 2·prec·rec/(prec+rec).

This scores **recovered meaning**, not preference. **Dissociation (proved):** a
system can be *preferred* (visual appeal) yet score *lower* comprehension — so
preference cannot stand in for adequacy (echoing Doc-11's loss≠usefulness).

## 7. Baselines, stratification, model card

* **Baselines** — retrieval/stitching, deterministic seq2seq, and a human-recorded
  **upper reference**; a system is credible only relative to these, and must not
  exceed the human upper reference within CI without independent replication.
* **Stratification** — slice every endpoint by sentence length, signer, lexical
  frequency, non-manual construction, occlusion, accent/noise, and consented
  demographic slices; report per-slice, never only the aggregate (reuse Doc-10
  stratification discipline).
* **Inter-rater reliability** — Doc-03 `cohens_kappa`/`fleiss_kappa` on human ratings.
* **Model card** (Mitchell et al.) — Model Details, Intended Use, Factors, Metrics,
  Evaluation Data, Training Data, Quantitative Analyses, Ethical Considerations,
  Caveats & Recommendations, plus model size / compute / latency / **failure modes**.
  Completeness is validated (analogue of the Doc-10 datasheet).

## 8. Integration + innovations

The chain aggregates the audited leaf metrics; the split reuses Doc-10; reliability
reuses Doc-03; the model card mirrors the Doc-10 datasheet. Innovations: the
falsifiable-contract chain with single-metric-insufficiency, caveat-bound metrics, the
pre-registration + test-set firewall, and the preference/comprehension dissociation.

## 9. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| 12.0 | research + design/math spec (this doc) | done |
| 12a | falsifiable contract + chain | done (8 tests) |
| 12b | caveat-bound metric stack | done (6 tests) |
| 12c | statistical rigor | done (10 tests) |
| 12d | pre-registration lock + test firewall | done (6 tests) |
| 12e | reproducible text metrics (SacreBLEU + BERTScore) | done (10 tests) |
| 12f | blinded comprehension + preference dissociation | done (7 tests) |
| 12g | baselines + stratification + model card | done (7 tests) |
| 12h | integration + cycle stress + full regression | done (2 tests) |

Evaluation framework: 56 tests, green on two consecutive runs under
`-W error::UserWarning`; whole project green (all 128 test files).

## 11. Findings (post-implementation)

**Adequacy is a conjunction, and single metrics are provably insufficient.** A chain
is adequate iff every contract passes; a chain with an excellent speech score (0.99)
but a failing non-manual layer (0.40 < 0.80) is not adequate, and adding a failing
contract can only lower adequacy (monotone conjunction). The document's principle —
BLEU/WER/FID/MPJPE alone cannot certify a signed message — is a theorem here.

**Caveats are structurally bound to metrics.** Each `MetricResult` must carry exactly
its layer's required caveat; a speech metric labelled with the rendering caveat, or no
caveat, raises. So "transcript accuracy = sign adequacy" is unstatable, and the caveat
flows into every derived contract.

**The statistics are exact and self-contained — and exposed a real power limit.** The
paired sign-flip permutation p-value and the exact binomial sign test both equal the
hand-computed 0.25 for `d=[1,2,3]`. A discovered consequence: with only three seeds a
permutation/sign test's minimum two-sided p-value is `2/2³ = 0.25`, so **three seeds
can never reach p ≤ 0.05** by these tests. The correct design (now reflected in the
integration test) runs the paired significance test over the many held-out test items
(large n) while the ≥3 seeds supply the bootstrap **confidence interval** — seeds give
CIs, the test set gives significance. A result counts only if it is *both* significant
*and* meets the pre-registered minimum effect; either alone is insufficient.

**The protocol is enforced, not just described.** Primary endpoints + minimum effects
are hash-locked (order-independent); the firewall raises on hyperparameter selection
on the test split and on reporting a non-registered endpoint as primary; the test set
is signer/source-held-out via the Doc-10 grouped split with a leakage certificate.
(The `TestFirewall` class was renamed `EvaluationFirewall` because pytest collects any
imported `Test*`-named class.)

**Text metrics are reproducible and correct.** Corpus BLEU is `BP·exp(Σ (1/N) log pₙ)`
with clipped precisions and exponential smoothing: identical corpus → 1.0, a short
hypothesis is exactly `exp(1−r/c)`-penalised, no-overlap with `smooth='none'` → 0, and
partial overlap with `smooth='exp'` stays positive; a signature string records
tokeniser/smoothing/order/#refs for cross-lab comparability (the tokeniser is honestly
named `basic_lc_v1`, not a false `13a` claim). BERTScore greedy matching gives
`P=R=F=1` for identical embeddings (in float64 — float32 self-cosine rounds to
~1.00005), is token-order invariant, and IDF re-weights emphasis.

**Preference is dissociated from comprehension.** Recovered-proposition F1 scores
meaning, not appeal; a system can be preferred yet convey less meaning (opposite
orderings), so preference cannot substitute for adequacy — the human study separates
comprehensibility from visual preference, forbids text priming, and reports κ
reliability.

**Honest scope holds.** No real systems, human raters, or learned FID embeddings; the
framework's logic and mathematics are proved on controllable synthetic inputs, and
human-panel scores are instrumented (schemas + reliability), not performed.
Innovations: the falsifiable-contract chain with single-metric-insufficiency,
caveat-bound metrics, the pre-registration + test-set firewall, and the
preference/comprehension dissociation.
