# 09 — Facial and Non-Manual Modeling — Design and Mathematics

This document fixes **all** mathematics of the facial / non-manual layer before any
code, in the discipline of docs 01–08. It implements
`09_facial_expression_modeling.md`: non-manual grammatical channels (brows, eye
aperture / gaze, head / torso, cheeks, mouth) as scoped intervals, a conditioned
multi-label interval decoder, a rich loss suite, and articulation to FLAME / SMPL-X
expression coefficients — treating non-manuals as **concurrent grammatical
channels, not cosmetic emotion**.

**Reuse.** Doc-03 already has the multi-label non-manual machinery
(`multilabel_scope_bce`, `MarkerSpan`, `spans_from_activations`,
`scope_containment_loss`, `NonmanualScopeHead`), the SIR non-manual events + SCOPE
edges, the Allen `contains_loss`, and `sign_bleu` / `cohens_kappa` / `fleiss_kappa`
/ the `ControllableASLBuilder` minimal-pair oracle. Doc-04 has the SMPL-X jaw / eye
/ expression parts; Doc-08 the linear `apply_blendshapes`. This layer **builds on**
those and adds the facial-specific content.

Primary sources studied:

* Li, Bolkart, Black, Li, Romero, *FLAME* (flame.is.tue.mpg.de) — a parametric head
  model (identity `β`, expression `ψ`, pose `θ` = global/neck/jaw/eyes), with
  shape / expression / pose blendshapes and LBS (same structure as SMPL, and the
  face region of SMPL-X, Doc-04).
* Pavlakos et al., *SMPL-X* (arXiv:1904.05866) — the expressive body model whose
  face is FLAME (Doc-04).
* Kartynnik et al., *MediaPipe Face Mesh* (arXiv:1907.06724) — a 468-vertex
  monocular face-mesh tracker. Landmarks are **observations, not linguistic
  labels** (per the document).
* Kim et al., *SignBLEU* (arXiv:2406.06648) — the multi-channel metric (Doc-03).
* Yin et al., *Including Signed Languages in NLP* (ACL 2021) — the manifesto that
  non-manual features are grammatical and must be modelled as language.

## 0. Honest scope (read first)

The FLAME model tensors are licensed (like SMPL-X) and not downloaded; we implement
the **articulation mapping** (marker → expression coefficients / jaw / eye
rotations) and reuse the Doc-04 SMPL-X pipeline for the face mesh. There is no real
facial or MediaPipe-tracked data; we validate on controllable synthetic non-manual
streams with known ground truth. Every property independent of a *trained* model or
a licensed asset (the interval algebra, the Bernoulli decoder, every loss's
minimiser, the scope-nesting consistency, the disentanglement bound, the
intensity-monotone articulation, all metrics) is proved exactly. Human-signer
comprehension tests are instrumented, not performed.

## 1. Non-manual channels and scoped intervals

Concurrent channels `k`:

    BROW, EYE_APERTURE, GAZE, HEAD, TORSO, CHEEK, MOUTH.

A non-manual event is a scoped interval

    n_k = (type, value, t_s, t_e, confidence),

with `type` a (channel, grammatical-marker) pair, `value ∈ [0,1]` intensity,
`t_s < t_e` the scope, `confidence ∈ [0,1]`. Grammatical markers include
`YN_Q` (brow raise), `WH_Q` (brow furrow), `NEG` (head shake), `TOPIC` (brow raise
+ head tilt), `COND` (held brow raise + head tilt). `validate_event` checks
`t_s<t_e`, `value/confidence ∈ [0,1]`, valid type.

## 2. Concurrent-channel scope algebra (innovation)

Multiple channels are active at once, and their scopes **nest**: a WH-question scope
may contain a topic scope. Nesting must be consistent — a contained scope's
interval lies within its container's (Allen "during", reuse Doc-03
`classify_relation` / `contains_loss`). We form the **scope-nesting forest** of a
set of events and prove it is well-formed: overlapping scopes are either disjoint or
properly nested (no partial crossing), so the concurrent channels compose cleanly.

## 3. Multi-channel interval decoder

Conditioned on a hidden state `h_t` (from the semantic plan and the manual motion),
each channel predicts an **independent Bernoulli** per frame (concurrent, so NOT a
softmax):

    p_{t,k} = σ(w_kᵀ h_t),
    L_NMM   = −Σ_{t,k} [ y_{t,k} log p_{t,k} + (1−y_{t,k}) log(1−p_{t,k}) ]  (masked).

`spans_from_activations` (Doc-03) decodes contiguous activations into scoped
intervals. Proved: the decoder depends on the conditioning; the per-channel outputs
are independent (changing one channel's target does not change another's gradient);
gradient flows.

## 4. Loss suite

    L = λ_bce L_NMM + λ_b L_boundary + λ_s L_scope + λ_v L_smooth + λ_d L_disent.

* **Boundary** `L_boundary` — supervise marker onset/offset from the target's time
  difference (a marker's boundary frames get a boundary target), a masked BCE on a
  boundary head; zero iff onsets/offsets match.
* **Scope consistency** `L_scope` — the non-manual scope must **contain** the manual
  event it marks (Doc-03 `contains_loss`); zero iff contained.
* **Temporal smoothness** `L_smooth = Σ_t ‖p_{t+1}−p_t‖₁` — penalises flicker; zero
  iff constant over time.
* **Disentanglement** `L_disent` — §5.

## 5. Class imbalance, focal loss, and uncertainty

* **Focal loss** `FL = −(1−p_t)^γ log(p_t)` for the target class (and symmetric for
  the negative), which down-weights easy examples (proved: for a well-classified
  example the focal weight `(1−p)^γ → 0`).
* **Class-balanced** weights `(1−β)/(1−β^{n_c})` (Cui et al.) for rare markers,
  larger for smaller `n_c`.
* **Innovation — annotation-agreement-weighted heteroscedastic uncertainty.** The
  model predicts a per-marker log-variance `s_k = log σ²_k`; the NLL
  `½ e^{−s} (y−p)² + ½ s` is minimised by `σ² = (y−p)²`, and we *tie* the target
  spread to the Doc-03 inter-annotator `κ`: low-agreement markers get a wider
  predictive interval (proved: predicted σ² increases as κ decreases).

## 6. Linguistic ⟂ affect / identity disentanglement (innovation)

The linguistic-marker representation `z_ling` must be independent of affect `a` and
identity `id` (a raised brow is a YN-question marker, not "surprise"). We enforce
and **prove** separation with the Doc-04 leakage-probe method: a probe cannot
recover affect/identity from `z_ling` (normalised error ≈ 1), while the same probe
recovers it when affect is folded in (≈ 0) — so the guard has power and the
disentanglement is certified, not asserted.

## 7. Marker → FLAME articulation

A scoped marker is articulated into face parameters:

    ψ = A · (marker one-hot · value),   jaw = R(head_marker · value),   eye = R(gaze),

with `A` a learnable marker→expression matrix, the jaw/eye rotations built by the
Doc-04 6D→SO(3) map, and rig blendshapes by the Doc-08 linear `apply_blendshapes`.
**Innovation — intensity monotonicity:** the articulation is monotone in the marker
`value` (a stronger brow raise yields a larger expression coefficient), proved, so
grammatical intensity is preserved and never collapsed to a cosmetic constant.

## 8. Evaluation

* **Minimal-pair comprehension** — identical manual motion, changed non-manual
  markers must yield a different meaning; a classifier reading only the non-manual
  channel distinguishes the pair (reuse the Doc-03 minimal-pair oracle).
* **Scope boundary error** — `|t_s − t̂_s| + |t_e − t̂_e|` for YN / WH / topic /
  conditional scopes.
* **Gaze/locus agreement** — gaze direction aligns with the referenced spatial locus
  (reuse Doc-03 loci).
* **Head-manual synchronisation** — head-movement onset aligns with the manual
  event onset (temporal offset).
* **Channel ablation** — remove the face / gaze / torso channel and measure the
  comprehension drop, per the document.

## 9. Integration + innovations

Non-manual events land in the Doc-03 SIR (NONMANUAL nodes + SCOPE edges); the
articulated `ψ`/jaw/eye feed the Doc-04 face parts and the Doc-08 blendshapes; the
decoder conditions on the Doc-02 plan. Innovations: concurrent-channel scope
algebra, certified linguistic/affect disentanglement, annotation-agreement-weighted
uncertainty, and intensity-monotone articulation.

## 10. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| 9.0 | research + design/math spec (this doc) | done |
| 9a | non-manual channels + scoped intervals + scope algebra | done (7 tests) |
| 9b | multi-channel interval decoder | done (5 tests) |
| 9c | loss suite (BCE / boundary / scope / smoothness) | done (6 tests) |
| 9d | focal loss + class balance + uncertainty | done (5 tests) |
| 9e | linguistic / affect disentanglement | done (4 tests) |
| 9f | marker → FLAME articulation | done (6 tests) |
| 9g | evaluation batteries | done (6 tests) |
| 9h | integration + cycle stress + full regression | done (5 tests) |

Facial/NMM layer: 44 tests, green on two consecutive runs; whole project (1220
tests) green.

## 11. Findings (post-implementation)

**Non-manuals are modelled as concurrent grammatical channels, not emotion.** Each
event is a scoped interval `(channel, marker, value, t_s, t_e, confidence)`; the
scope algebra (built on the Doc-03 Allen relations) proves concurrent scopes are
either disjoint or properly nested — a partially-crossing pair is rejected — and
builds the nesting forest. The decoder emits **independent per-channel Bernoullis**
(two channels can both be ~1 at one frame; a softmax could not), so co-occurring
markers are representable.

**The loss suite reuses the audited Doc-03 pieces and adds the missing ones.** BCE
and scope containment come from Doc-03; boundary targets mark run onsets/offsets on
active frames; temporal smoothness is zero on a constant stream. Focal loss
provably suppresses easy examples (`(1−p)^γ → 0`), class-balanced weights are larger
for rarer markers, and the heteroscedastic NLL is minimised at `σ² = (y−p)²` — with
the innovation that low inter-annotator `κ` maps to a wider predictive interval.

**Disentanglement is certified, not asserted.** A gradient-reversal adversary
enforces affect-invariance, and the Doc-04 leakage probe certifies it: affect is
not recoverable from `z_ling` (normalised error > 0.85) yet is recovered when folded
in (< 0.05), so the guard has power.

**Articulation is intensity-monotone.** The marker→FLAME expression map is linear
and monotone in the marker `value` (a stronger brow raise yields a larger
coefficient, to machine precision); the jaw rotation's angle scales with the value
(value 0 → identity / jaw closed) and stays in SO(3). Grammatical intensity is
preserved, never collapsed to a cosmetic constant.

**The scope/structure division was respected, not misused.** A discovered subtlety:
`validate_sir` checks structural rules (a SCOPE edge's source must be non-manual,
target manual) but NOT interval containment — that is enforced by the differentiable
`scope_containment_loss`. The integration test asserts exactly this division (a
non-containing scope is structurally valid but has a positive scope loss), rather
than expecting the structural validator to do the loss's job.

**Honest scope holds.** FLAME tensors are licensed and not downloaded; the mesh
comes from the Doc-04 SMPL-X pipeline and the articulation matrix is a stand-in for
the anatomical/learned map. No real facial or MediaPipe-tracked data; all proofs are
on controllable synthetic non-manual streams. Innovations: concurrent-channel scope
algebra, certified linguistic/affect disentanglement, annotation-agreement-weighted
uncertainty, and intensity-monotone articulation.
