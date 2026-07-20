# 11 — Self-Supervised and Weakly Supervised Pretraining — Design and Mathematics

This document fixes **all** mathematics of the pretraining layer before any code, in
the discipline of docs 01–10. It implements `11_self_supervised_pretraining.md`:
masked motion modeling, cross-modal contrast with **linguistically-grounded hard
negatives**, temporal/part consistency, and an evidence battery whose central rule
is *a lower pretraining loss alone is not evidence of linguistic usefulness*.

**Reuse.** This layer is heavily integrative:

* `models/alignment.py` — `info_nce_loss` (symmetric InfoNCE, exactly the document's
  `L_NCE`), `ContrastiveAligner` (learnable clamped log-temperature, CLIP), and
  `ProjectionHead`.
* `pose/leakage.py` — `LinearProbe` (closed-form ridge) and
  `normalised_recovery_error` for the linguistic probes AND the signer/background
  leakage tests.
* `motion_transformer/quantizer.py` — `VectorQuantizer` supplies discrete latent
  **token targets** `z_i` for masked modeling (wav2vec-2 / MAE style).
* `pose/rotations.py` — `rotation_6d_to_matrix`, `geodesic_distance` for masked
  **rotation** reconstruction.
* `grammar/grammar_tests.py` — `GrammarFeatures`, `ControllableASLBuilder`,
  `minimal_pair`, `LICENSED`: the oracle that flips exactly one licensed linguistic
  feature — the hard-negative generator.
* `data_engineering/splitting.py` — `grouped_split` for leakage-free **cross-signer**
  retrieval.

Primary sources studied:

* He et al., *Masked Autoencoders* (arXiv:2111.06377) — asymmetric encoder/decoder,
  high mask ratio, loss on **masked patches only**; the encoder sees only visible
  tokens.
* Tong et al., *VideoMAE* (arXiv:2203.12602) — **tube masking** at an extreme ratio
  (90–95%) because temporal redundancy makes adjacent frames leak; random/frame
  masking is easier. The motion analogue of "tube" is a **part** mask (mask one hand
  or the face across all frames).
* Radford et al., *CLIP* (arXiv:2103.00020) — symmetric image/text InfoNCE with a
  learnable temperature (clamped).
* Baevski et al., *wav2vec 2.0* (arXiv:2006.11477) — contrastive task over
  **quantized** latents at masked spans + a **diversity** loss maximising codebook
  entropy.
* Wong et al., *Sign2GPT* (arXiv:2405.04164) — weakly-supervised pseudo-gloss
  pretraining; labels are pseudo, so probes not loss decide usefulness.

## 0. Honest scope (read first)

No pretrained weights or real signing corpora are used — the datasets are licensed
and gated (Doc-10). We implement the **mathematics and mechanisms** of pretraining
and validate them on controllable synthetic motion/token/embedding streams with
known ground truth. Every property independent of a *trained* network — the mask
difficulty certificate, the masked-only NLL, the InfoNCE minimiser, the
hard-negative shortcut falsification, the augmentation guard, the probe/leakage
harness, the loss-vs-usefulness dissociation — is proved exactly. Claims that would
require real training (that pretraining *raises* probe accuracy on real signing) are
implemented as **harnesses** and stated as such, never asserted as results.

## 1. Masking strategies + difficulty certificate (innovation)

Let a clip be `T` frames over `P` part-streams (body, left hand, right hand, face).
A mask `M` selects (frame, part) tokens to hide. Three strategies:

* **Span mask** — contiguous runs of length `L` per part (defeats interpolation).
* **Part mask** — an entire part stream across all frames (the "tube" analogue;
  forces cross-part inference, e.g. infer the face from the hands).
* **Semantic-boundary mask** — tokens straddling SIR event boundaries (Doc-03), the
  linguistically informative transitions.

**Interpolation-defeating certificate (innovation).** The document's warning —
"random point masking is too easy because adjacent frames interpolate" — is made
*checkable*. For a masked index `t` with nearest visible neighbours at distance
`a` (left) and `b` (right), the linear-interpolation predictor
`x̂_t = (b·x_{t−a} + a·x_{t+b})/(a+b)` has, for a `C²` signal, error

    |x_t − x̂_t| = ½ |x''(ξ)| · a·b + o(a·b).

A single random masked point has `a=b=1` so the error floor is `≈ ½|x''|`; the
interior of a span of length `L` has `a·b ≈ (L/2)²`, so the floor grows like `L²`.
`mask_interpolation_error_floor(M)` returns this per-token floor; a mask is
**certified hard** iff its worst-token floor exceeds a threshold — turning
VideoMAE's empirical "use a high ratio" into a property we can verify. Proved on a
known-curvature synthetic signal (quadratic: interpolation error is exactly
`½·x''·a·b`).

## 2. Masked motion modeling objective

Following MAE, the encoder sees **only visible tokens**; a lightweight decoder
receives the encoded visibles plus learned mask placeholders (with frame/part
positional embeddings) and predicts the hidden tokens. Two prediction targets:

* **Latent tokens** — the frozen Doc-06 `VectorQuantizer` maps each token to a
  codebook index `z_i ∈ {1..K}`; the decoder emits logits and

      L_mask = (1/|M|) Σ_{i∈M} −log softmax(logits_i)[z_i]      (cross-entropy),

  the document's `Σ_{i∈M} −log p(z_i | z_{∖M}, c)`, averaged over masked tokens.
* **Rotations** — reconstruct 6D rotations at masked frames under the Doc-04
  geodesic loss `Σ_{i∈M} d_geo(R̂_i, R_i)`.

**Masked-only is not a detail (proved).** The loss is computed over `M` only. If
visible positions were included, an identity/copy decoder would score 0 there and
dilute the signal; restricting to `M` forces genuine prediction. We prove a
copy-through decoder attains 0 visible-loss but chance masked-loss, so masked-only
scoring separates prediction from copying. **Diversity term (wav2vec-2):** an
optional `L_div = −H(mean codebook usage)` discourages codebook collapse; proved
maximised (most negative) at uniform usage.

## 3. Cross-modal contrast

The document's `L_NCE` is **exactly** the symmetric InfoNCE already in
`models/alignment.py`. For unit-norm motion `u_i` and text/speech `v_i`,

    L_NCE = −(1/2B) Σ_i [ log softmax_j(u_i·v_j/τ)[i] + log softmax_j(v_i·u_j/τ)[i] ].

Reused, not reimplemented. Proved: perfectly aligned unit embeddings drive `L_NCE→0`
as `τ→0`; the minimiser matches `i↔i`; the temperature clamp caps the logit scale
(CLIP). Retrieval `recall@k` is the evaluation, not the loss value.

## 4. Linguistically-grounded hard negatives + shortcut falsification (innovation)

The document requires negatives "differing in negation, question type, entity,
handshape, direction, or number", warning that random negatives "permit shortcut
learning from signer, background, or sentence length". We realise this with the
Doc-03 oracle: `hard_negative(base, dimension)` flips exactly one `GrammarFeatures`
field (`negated`, `question`, `object`/entity, `aspect`, `plural_subject`,
`role_shift`) and, via `LICENSED`, changes **only** the SIR fields that feature
licenses — so a hard negative is a *minimal linguistic contrast*, not a random clip.

**Shortcut falsification (proved).** Construct an embedding that encodes **only a
shortcut** (signer id, or clip length). Then:

* against **random** negatives (which differ in the shortcut), the shortcut
  separates positive from negatives → InfoNCE ≈ 0 (the shortcut "solves" the task);
* against **hard** negatives (a minimal pair sharing signer and length), the
  shortcut is identical for positive and negative → InfoNCE is large (the shortcut
  fails).

So a low random-negative loss is **not** evidence of linguistic content, and hard
negatives are what force it — the document's claim, made a theorem on controllable
embeddings.

## 5. Temporal and part consistency + augmentation guard (innovation)

* **Order / boundary** — shuffle clip segments and predict the correct permutation
  (pairwise precedence accuracy), and predict segment boundaries (reuse the Doc-09
  boundary target machinery). Proved: the correct order is uniquely recoverable from
  a monotone timestamp.
* **Multi-view alignment** — RGB, 2D pose, 3D pose, and cropped-hand views of the
  **same** clip are positives; different clips are negatives (InfoNCE across views).
* **Handedness/direction-preserving augmentation guard (innovation).** Appearance
  augmentations (scale, translate, temporal jitter, additive noise) preserve
  linguistic content and are allowed. A **horizontal flip** swaps handedness and
  reverses spatial loci / agreement-verb direction — it changes meaning. `augment`
  **raises** on a flip unless a `direction_relabel` callback is supplied that
  transforms handedness, loci, and directional labels; a structural refusal
  (extends Doc-08's separation discipline), proved by construction.

## 6. Evidence battery — loss is not usefulness (innovation)

Per the document, evidence is probes and retrieval, never loss:

* **Linear probes** — freeze features, fit `LinearProbe` for handshape / non-manual
  markers, report accuracy vs a chance baseline.
* **Low-resource scaling curve** — probe accuracy vs number of labelled examples;
  the harness reports the curve (a *linearly decodable* feature set yields a higher
  curve than a scrambled one — proved on synthetic features).
* **Cross-signer retrieval** — split by signer with the Doc-10 `grouped_split`
  (no signer spans train/test), retrieve across signers, report recall@k.
* **Leakage test** — fit a signer/background probe on frozen features; **high**
  signer-accuracy is leakage (bad). We report it and flag it; low is the goal.
* **Loss-vs-usefulness dissociation (innovation).** Two feature sets are constructed
  with the **same** reconstruction/contrastive loss but **different** linguistic
  probe accuracy, so a test exhibits that equal loss does not imply equal usefulness
  — operationalising the document's central caveat.

## 7. Curriculum + frozen baselines

The five-stage curriculum — (1) unimodal masked motion, (2) RGB↔pose and 2D↔3D
alignment, (3) motion↔text/speech contrast with curated negatives, (4) supervised
sign-plan/production, (5) limited end-to-end tuning — is an explicit ordered
schedule with a `FrozenBaseline` registry: each stage snapshots the previous stage's
weights so a regression is always detectable (retain frozen baselines). Proved: the
schedule is monotone in unlocked capacity and a frozen snapshot is bit-identical.

## 8. Integration + innovations

Masked targets come from the Doc-06 VQ; rotations from Doc-04; contrast from
`models/alignment.py`; hard negatives from the Doc-03 oracle; probes/leakage from
Doc-04; the cross-signer split from Doc-10. Innovations: the interpolation-defeating
mask certificate, linguistically-grounded hard negatives with a shortcut
falsification, the handedness/direction-preserving augmentation guard, and the
loss-vs-linguistic-usefulness dissociation harness.

## 9. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| 11.0 | research + design/math spec (this doc) | done |
| 11a | masking strategies + difficulty certificate | done (10 tests) |
| 11b | masked motion modeling objective | done (9 tests) |
| 11c | cross-modal symmetric InfoNCE alignment | done (8 tests) |
| 11d | linguistic hard negatives + shortcut falsification | done (17 tests) |
| 11e | temporal/part consistency + augmentation guard | done (8 tests) |
| 11f | evidence battery + loss-vs-usefulness dissociation | done (7 tests) |
| 11g | curriculum orchestration + frozen baselines | done (7 tests) |
| 11h | integration + cycle stress + full regression | done (3 tests) |

Pretraining layer: 69 tests, green on two consecutive runs under
`-W error::UserWarning`; whole project green (all 120 test files).

## 11. Findings (post-implementation)

**The mask difficulty certificate is exact — and exposed a subtlety.** For a
quadratic signal the linear-interpolation error at a masked point between visible
neighbours at distances `a`, `b` is exactly `½·x''·a·b` (verified to `1e-12`), so a
single random point (`a=b=1`) floors low while a span interior floors like `L²`.
The subtlety: the *worst*-token floor is not the right lens for "random is easy",
because random masking can hit an edge token with no neighbour on one side
(genuinely uninterpolatable → `+inf`). The document's claim is about the *typical*
token, so the strategy comparison uses the **median** floor; the worst-case `+inf`
for a fully-masked part is a separate, legitimate fact (it forces cross-part
inference).

**MAE asymmetry is verified, not assumed.** The encoder ingests only visible tokens,
so the model output is invariant to the *identity* of masked input tokens (proved to
`1e-6`) yet depends on visible tokens. A copy-through decoder scores 0 on visible
positions but chance (`log K`) on masked positions — so masked-only scoring provably
separates prediction from copying. Latent-token targets come from the frozen Doc-06
VQ; rotations from the Doc-04 geodesic; the wav2vec-2 diversity term is minimised at
uniform codebook usage (`−log K`).

**The hard-negative claim is a theorem here, and the oracle corrected two premises.**
(1) A signer (or length-binned) shortcut drives random-negative InfoNCE to `~0` but
fails on signer/length-matched hard negatives (`log 3`), while a content
representation succeeds on both — exactly the document's warning, made provable on
controllable embeddings. (2) An **entity** swap is a genuine single-feature contrast
but additionally reorders naming signs (`order` changes), so it is *not* a subset of
the pre-declared licensed fields; the mining guarantee is "genuine single-feature
contrast (SIRs differ)", with `is_licensed_contrast` the stronger property that holds
for the other five dimensions. (3) A raw scalar length embedding is destroyed by the
InfoNCE L2 normalisation, so the length shortcut is honestly implemented as a binned
one-hot that survives it.

**Multi-view alignment is a shared-space property.** Two arbitrarily-rotated raw
views do NOT retrieve to each other (that is what the learned projection heads are
for); the honest test is that paired views in a shared space align (recall@1 = 1)
while a shuffled pairing does not.

**The augmentation guard is structural.** Appearance augmentations (scale, translate,
noise) preserve content and never require relabelling; a horizontal flip **raises**
unless a `LinguisticDirection` is supplied, and flipping mirrors the x-coordinate
while swapping handedness, negating loci, and reversing agreement — a consistent
involution (`flip∘flip = id`). Meaning-changing geometry cannot be applied silently.

**Loss is dissociated from usefulness by construction.** Two feature sets share a
block that reconstructs the input identically (equal reconstruction loss to `1e-9`)
but differ in a second block — label-decodable vs noise — giving a `>0.5` probe-
accuracy gap. Equal pretraining loss provably does not imply equal linguistic
usefulness; probes and cross-signer retrieval decide, and the signer leakage probe
flags representations that beat chance on signer identity.

**Honest scope holds.** No pretrained weights or real signing corpora are used
(licensed/gated, Doc-10); every property is proved on controllable synthetic
motion/token/embedding streams. Claims requiring real training (that pretraining
*raises* probe accuracy on real signing) are implemented as harnesses and stated as
such. Innovations delivered: the interpolation-defeating mask certificate,
linguistically-grounded hard negatives with a shortcut falsification, the
handedness/direction-preserving augmentation guard, and the loss-vs-usefulness
dissociation harness.
