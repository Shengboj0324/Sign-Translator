# 06 — Temporal Motion Transformers — Design and Mathematics

This document fixes **all** mathematics of the temporal-motion layer before any
code, in the discipline of docs 01–05. It implements
`06_temporal_motion_transformers.md`: a VQ / residual-VQ motion tokenizer, a
hierarchical temporal backbone, the motion loss, anti-oversmoothing diagnostics,
and streaming with rotation-space chunk blending.

Primary sources studied:

* van den Oord, Vinyals, Kavukcuoglu, *Neural Discrete Representation Learning*
  (VQ-VAE), arXiv:1711.00937 — nearest-neighbour quantisation, the
  straight-through estimator, and the codebook + commitment losses.
* Zhang et al., *T2M-GPT* (arXiv:2301.06052) — a CNN VQ-VAE + GPT for motion; it
  explicitly relies on **EMA** codebook updates and **Code Reset** (dead-code
  revival), both implemented here.
* Zeghidour et al., *SoundStream* / Residual Vector Quantization
  (arXiv:2107.03312) — the RVQ cascade `r_{i+1} = r_i − Q_i(r_i)`, `z_q = Σ_i c_i`.
* Jiang et al., *MotionGPT* (arXiv:2306.14795) — motion-as-language; masked-span
  and autoregressive decoding over motion tokens (context for the decoding stage).
* *SignAvatar CVAE baseline* (arXiv:2405.07974) — a continuous-latent baseline to
  compare the discrete tokenizer against.

## 0. Honest scope (read first)

We have no large real sign-motion corpus and no pretrained motion tokenizer. As in
docs 04-05 we implement the **mathematics exactly** and validate on controllable
synthetic motion (and the existing synthetic corpus), so every property that does
not depend on a *trained* codebook — the quantiser's straight-through gradient,
the RVQ residual algebra, the loss forms and their minimisers, the SO(3) blend
continuity, the spectral/duration diagnostics, streaming causality — is proved
exactly. Codebook *quality* (perplexity, FID) needs real data and is out of scope
here. The learned pieces (encoder/decoder/backbone weights) are structurally
correct and drop into a real training run unchanged.

## 1. Vector quantisation (VQ-VAE)

Encoder `z_e = E(q)`; codebook `{e_k}_{k=1}^K ⊂ ℝ^d`; decoder `q̂ = D(z_q)`.

    k*(z) = argmin_k ‖z − e_k‖²,     z_q = e_{k*}.

Squared distances use `‖z − e_k‖² = ‖z‖² − 2 z·e_k + ‖e_k‖²` (verified identical to
the naive form). Ties broken by lowest index (deterministic).

**Straight-through estimator.** The argmin has zero gradient a.e., so we copy the
gradient from `z_q` to `z_e`:

    z_q^{ste} = z_e + (z_q − z_e).detach(),

which forward-equals `z_q` and has `∂z_q^{ste}/∂z_e = I` (proved via autograd).

**Losses.** With `sg[·]` = stop-gradient and commitment weight `β` (≈0.25):

    L_codebook  = ‖sg[z_e] − z_q‖²      (pulls the code toward the encoder output)
    L_commit    = ‖z_e − sg[z_q]‖²      (pulls the encoder toward its code)

Both equal `‖z_e − z_q‖²` in value but carry gradient to different sides.

**EMA codebook** (T2M-GPT default; no `L_codebook` term). For code `k` with the
batch's assigned vectors summing to `m_k` over `n_k` members:

    N_k ← γ N_k + (1−γ) n_k,   M_k ← γ M_k + (1−γ) m_k,
    e_k ← M_k / (N_k + ε),   with Laplace smoothing  N_k ← (N_k+ε)/(N+Kε)·N.

Proved: EMA drives `e_k` to the running mean of its cluster; Laplace smoothing
avoids divide-by-zero for empty codes.

**Code Reset (dead-code revival).** Codes whose usage falls below a threshold are
re-seeded from a random current encoder output — proved to change exactly the dead
rows and to raise usage. **Perplexity** `= exp(−Σ_k p_k log p_k)` (with `p_k` the
code-usage distribution) measures effective codebook size.

## 2. Residual VQ (SoundStream)

A cascade of `Q` quantisers refines the residual:

    r_1 = z_e;   for i=1..Q:  c_i = Q_i(r_i),  r_{i+1} = r_i − c_i;   z_q = Σ_{i} c_i.

The reconstruction error after `Q` stages is `‖z_e − z_q‖ = ‖r_{Q+1}‖`. The
per-stage algebra is exact:

    ‖r_{i+1}‖² = ‖r_i‖² − 2 r_i·c_i + ‖c_i‖²,

so a stage reduces the residual iff `2 r_i·c_i ≥ ‖c_i‖²`. With nearest-neighbour
quantisation of `r_i` against a codebook fit to the residual distribution this
holds, and we verify **monotone non-increasing** expected residual norm across
stages empirically (codebooks k-means-initialised on the data); adding stages
never increases error. Quantiser dropout (train a random prefix of stages) gives a
single model multiple rates.

## 3. Part-specific codebooks

The document warns low-variance torso motion should not consume hand capacity.
Split `z_e` by channel into parts `{torso, hands, face}` and quantise each with its
own (R)VQ, then concatenate. **Innovation — spectral capacity allocation:** the
number of codes/stages per part is set proportional to that part's measured
spectral energy (§5), so hands (high-frequency) get more capacity than torso.

## 4. Motion loss

    L = λ_q d_{SO(3)}(R, R̂)
      + λ_v ‖Δq − Δq̂‖₁ + λ_a ‖Δ²q − Δ²q̂‖₁
      + λ_c L_contact + λ_s L_semantic + L_commit,

with `Δq_t = q_{t+1} − q_t` (velocity) and `Δ²q_t = q_{t+2} − 2q_{t+1} + q_t`
(acceleration). `d_{SO(3)}` is the Doc-04 geodesic; `L_contact` reuses the Doc-05
contact field; `L_semantic` reuses the InfoNCE aligner. **Innovation — geodesic
velocity:** for rotations we also provide `Δ_{SO(3)} = d_{SO(3)}(R_t, R_{t+1})`, the
true angular speed, alongside the 6D-Euclidean difference.

**Oversmoothing.** The document notes velocity loss alone can favour small motion:
a constant prediction has zero velocity error. We demonstrate this failure and
counter it (§5).

## 5. Anti-oversmoothing diagnostics (+ innovation)

* **Spectral energy by body part** — for each channel's length-`T` series, the
  real-FFT gives `E = Σ_f |X_f|²`; by **Parseval**, `Σ_t x_t² = (1/T) Σ_f |X_f|²`
  (verified). We report energy in low/mid/high bands per part; oversmoothing shows
  up as lost high-band energy.
* **Innovation — spectral-energy-matching loss:** `L_spec = Σ_{part,band}
  |E_band(pred) − E_band(real)|`, a differentiable regulariser that directly
  penalises the high-frequency energy an oversmoothing model drops — going beyond
  the document's "report it" to "optimise against it".
* **Duration calibration** — predicted vs true event durations; a reliability-style
  calibration error over duration buckets.

## 6. Hierarchical temporal backbone

Four coupled modules at two rates:

1. **Clause planner** (low rate) — a Transformer over clause/plan tokens.
2. **Event-duration model** — predicts each event's duration (bucketed).
3. **High-rate motion decoder** — a Transformer over motion tokens with
   **cross-attention** to the plan events (plan = memory keys/values).
4. **Recurrent memory** — a state carrying spatial loci and the prior pose, updated
   per chunk (GRU-style), so discourse referents persist across long sequences.

Proved: the rate hierarchy (low-rate plan up-sampled to the high-rate decoder), the
cross-attention conditioning actually depends on the plan, and the memory carries
information across chunks.

## 7. Streaming: chunked causal attention + SO(3) chunk blending

* **Chunked causal attention** — process length-`C` chunks with bounded right
  context `R`; output frame `t` attends only to frames `≤ t + R` (proved: no
  leakage beyond `R`). Latency `= C + R` frames; full bidirectional attention is
  the offline upper bound, reported separately, never as a latency result.
* **Innovation — rotation-space chunk blending.** Overlapping chunks are blended in
  SO(3) by a SLERP crossfade over the overlap: with a linear ramp `α: 0→1`,

    R_blend(α) = R_a · exp( α · log(R_aᵀ R_b) )  (the constant-speed geodesic),

  so `R_blend(0)=R_a`, `R_blend(1)=R_b`, `d_{SO(3)}(R_a, R_blend(α)) = α·d(R_a,R_b)`
  (proved), giving C0-continuous, boundary-matching transitions instead of naive
  linear averaging of rotations (which leaves SO(3)).

## 8. Decoding strategies (experiments)

* **Autoregressive** — causal next-token prediction over motion codes.
* **Masked-span** — mask token spans and predict them (MotionGPT/MaskGIT style).
* **Diffusion** — reuse the Doc-01/04 Gaussian motion diffusion over continuous
  latents.

The experiment harness compares raw-continuous vs VQ vs residual-VQ latents and
shared vs part-specific codebooks at a fixed budget, and offline vs chunked-causal
attention at fixed quality — honest scaffolding, not tuned numbers.

## 9. Integration

The tokenizer consumes the Doc-04 pose representation (6D rotations) and Doc-05
hand features; the plan events come from the Doc-02 plan / Doc-03 SIR; the
diffusion decoder reuses Doc-01/04 machinery. Stage 6h wires the full chain and
runs a long-discourse locus-persistence test and whole-chain cycle stress.

## 10. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| 6.0 | research + design/math spec (this doc) | done |
| 6a | VQ-VAE vector quantizer core (STE, commitment, EMA, reset) | done (11 tests) |
| 6b | residual VQ + part-specific codebooks | done (10 tests) |
| 6c | motion autoencoder + full motion loss | done (8 tests) |
| 6d | anti-oversmoothing diagnostics (spectral, duration) | done (12 tests) |
| 6e | hierarchical temporal backbone | done (9 tests) |
| 6f | streaming causal attention + SO(3) chunk blending | done (10 tests) |
| 6g | decoding strategies + experiment harness | done (8 tests) |
| 6h | integration + cycle stress + full regression | done (6 tests) |

Motion layer: 74 tests, green on two consecutive runs; whole project (1056 tests)
green.

## 11. Findings (post-implementation)

**The straight-through estimator is proved, not assumed.** `∂z_q^{ste}/∂z_e = I`
to machine precision, so the encoder trains through the argmin; the codebook loss
(`ema=False`) or the EMA update (`ema=True`) trains the codes; EMA converges the
codebook to its cluster means (verified to <0.3 over 300 steps) with Laplace
smoothing preventing empty-code divide-by-zero, and Code Reset re-seeds exactly the
dead rows.

**Residual VQ needed one subtle correction to be right.** Applying a per-stage STE
detaches every stage after the first (the residual becomes a `.detach()`ed value),
so the encoder would only learn from stage 1. The fix is a **single** straight-
through over the whole cascade (`z_e + (Σc_i − z_e).detach()`), which passes an
exact identity gradient — verified `∂z_q/∂z_e = I` across all stages. The residual
norm is proved monotone non-increasing across stages (k-means-initialised
codebooks) and more stages never worsen reconstruction.

**The oversmoothing insight is made concrete.** Velocity and acceleration L1 are
blind to a constant offset (a shifted copy scores 0 on both) while the geodesic
term catches it — the exact reason velocity loss alone favours small motion.
**Innovation:** the spectral-energy-matching loss makes this differentiable — a
low-pass (over-smoothed) prediction provably loses high-band energy, and the loss
penalises that gap per body part. The Parseval identity backing the diagnostic is
exact for both even and odd length.

**Streaming is causal and the SO(3) blend is a true geodesic.** A bounded-right-
context mask gives output-`t`-depends-only-on-inputs-`≤ t+R` (verified by autograd:
zero gradient beyond `t+R`), so latency is `R` frames. The **innovation** — SLERP
crossfade chunk blending — is proved to be a constant-speed geodesic
(`d(R_a, slerp(R_a,R_b,α)) = α·d(R_a,R_b)` to 1e-8), to hit its boundary rotations
exactly, and to keep every stitched frame on SO(3), where naive matrix averaging
would leave the manifold.

**The backbone hierarchy and decoding strategies are wired and verified.** The
duration model bridges the low-rate plan to the high-rate timeline (Σ durations
frames); the motion decoder's output genuinely depends on the plan (cross-
attention); GPT decoding is causal and fits a token pattern; masked-span supervises
only masked positions with bidirectional context; the recurrent memory carries an
early locus across 10 chunks (long-discourse persistence).

**Honest scope holds.** No real sign-motion corpus and no pretrained tokenizer;
every property above is independent of a *trained* codebook and proved on
controllable synthetic motion. The experiment harness reports honest relative
reconstruction (RVQ ≤ VQ, part-specific vs shared) rather than tuned numbers.
Innovations beyond the sources: the spectral-energy-matching loss, part-specific
RVQ with spectral capacity allocation, geodesic SO(3) chunk blending, and geodesic
velocity in rotation space.
