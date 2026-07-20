# 07 — Diffusion Motion Generation — Design and Mathematics

This document fixes **all** mathematics of the diffusion motion-generation layer
before any code, in the discipline of docs 01–06. It implements
`07_diffusion_motion_generation.md`: a conditional temporal-DiT diffusion model
with ε/x₀/v parameterizations, classifier-free guidance, part-aware schedules,
inpainting, kinematic constraints, and consistency/rectified-flow distillation.

**Reuse.** The project already has an audited diffusion core
(`models/diffusion.py`: `make_beta_schedule`, `GaussianMotionDiffusion` with
`q_sample`, the DDPM posterior, ε↔x₀ conversions, velocity, DDIM;
`models/guided_diffusion.py`: classifier-free guidance; `models/denoiser.py`).
This layer **builds on** those primitives — it does not re-derive the core DDPM —
and adds the Document-07-specific system.

Primary sources studied:

* Ho, Jain, Abbeel, *DDPM* (arXiv:2006.11239) — forward `q(x_t|x_0)`, the reverse
  posterior, and `L_simple`.
* Ho & Salimans, *Classifier-Free Diffusion Guidance* (arXiv:2207.12598) —
  `ε̂ = (1+w)ε_θ(x_t,c) − w ε_θ(x_t,∅)`, condition dropout.
* Tevet et al., *Human Motion Diffusion Model (MDM)* (arXiv:2209.14916) —
  **x₀-prediction** with geometric (position/velocity/contact) losses for motion.
* Peebles & Xie, *DiT* (arXiv:2212.09748) — transformer diffusion backbone with
  **adaLN-Zero** conditioning (zero-init gate ⇒ each block starts as identity).
* Song et al., *Consistency Models* (arXiv:2303.01469) — one/few-step generation
  via a self-consistent `f(x_t,t)→x₀` with the boundary condition `f(x_ε,ε)=x_ε`.

## 0. Honest scope + design principle (read first)

Per the document, diffusion here models the **residual motion uncertainty after
grammar and timing are fixed** — many valid motions for one plan — **not** a repair
for an underspecified planner. We have no real sign-motion corpus, so as in docs
04-06 we implement the mathematics exactly and validate on controllable synthetic
motion and the existing denoiser. Every property independent of a *trained* model
(the parameterization algebra, CFG equivariance, adaLN-Zero identity, inpainting
preservation, constraint penalties and projections, consistency boundary and
self-consistency, rectified-flow straightness, all metrics) is proved exactly;
sample quality needs real training and is out of scope. The consistency/RF sampler
is machinery to distill **after** the full model is good, per the document.

## 1. Forward process and the three parameterizations

Forward marginal `q(x_t|x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I)`, i.e.

    x_t = a x_0 + b ε,   a = √ᾱ_t,  b = √(1−ᾱ_t),  a² + b² = 1,  ε ~ N(0, I).

Given fixed `x_t`, the clean target admits three equivalent forms:

* **ε-prediction**  `ε = (x_t − a x₀)/b`;
* **x₀-prediction** `x₀ = (x_t − b ε)/a` (MDM; eases geometric losses);
* **v-prediction**  `v ≡ a ε − b x₀` (Salimans–Ho), with the exact inversions

    x₀ = a x_t − b v,     ε = b x_t + a v.

Both inversions are proved algebraically (`a²+b²=1`). All six conversions
round-trip exactly. The losses are **reweightings** of one another (proved):

    ‖ε − ε_θ‖²  = (ᾱ/(1−ᾱ)) ‖x₀ − x₀_θ‖²  = SNR(t) ‖Δx₀‖²,
    ‖v − v_θ‖²  = (1/(1−ᾱ))  ‖x₀ − x₀_θ‖².

## 2. DDPM posterior (reuse + cross-check)

`q(x_{t−1}|x_t, x₀) = N(μ̃_t, β̃_t I)` with

    μ̃_t = (√ᾱ_{t−1} β_t /(1−ᾱ_t)) x₀ + (√α_t (1−ᾱ_{t−1})/(1−ᾱ_t)) x_t,
    β̃_t = ((1−ᾱ_{t−1})/(1−ᾱ_t)) β_t.

We reuse `GaussianMotionDiffusion.q_posterior_mean_variance` and cross-check our
schedule against it. `q(x_T) → N(0, I)` as `ᾱ_T → 0`.

## 3. Classifier-free guidance

Train with **condition dropout** (probability `p_uncond`), so the network learns
both `ε_θ(x_t, c)` and `ε_θ(x_t, ∅)`. Guided prediction

    ε̂ = (1+w) ε_θ(x_t, c) − w ε_θ(x_t, ∅)
       = ε_θ(x_t, ∅) + (1+w)(ε_θ(x_t, c) − ε_θ(x_t, ∅)).

Proved: `w=0` recovers the conditional model; guidance is **parameterization-
equivariant** — because `x₀` (and `v`) are affine in `ε` with the same `x_t`
offset, the same convex-extrapolation in ε-space equals it in x₀/v-space; and
higher `w` provably reduces sample variance (diversity). **Innovation —
guidance annealing:** vary `w` over the sampling trajectory (higher early for
semantics, lower late for naturalness/diversity), preserving meaning while keeping
multimodality.

## 4. Temporal DiT with adaLN-Zero

A transformer over motion frames. A conditioning vector `c` (timestep embedding +
pooled sign plan/duration/prosody/discourse/style/prior) drives **adaLN-Zero**:
an MLP maps `c` to per-block `(shift₁,scale₁,gate₁,shift₂,scale₂,gate₂)`;

    h = x + gate₁ ⊙ Attn( LN(x)·(1+scale₁) + shift₁ ),
    y = h + gate₂ ⊙ MLP ( LN(h)·(1+scale₂) + shift₂ ).

The gate MLP is **zero-initialised**, so `gate=0` and the block is the identity at
init (proved) — a stable training start. Rich conditions also enter by
**cross-attention** to condition tokens. The output head emits ε, x₀, or v.

## 5. Part-aware schedules and loss weights

The document asks for part-aware noise for hands/face. We provide (a) per-part
**loss weights** `λ_p` (higher for hands/face) and (b) per-part `ᾱ^p` schedules.
Any monotone `ᾱ: 1→0` is a valid forward (proved). **Innovation — SNR capacity
allocation:** give semantically critical parts (hands) a schedule that retains
more signal (higher SNR) at each `t`, so the denoiser resolves them more precisely.

## 6. Inpainting (streaming overlap + user correction)

RePaint-style: at each reverse step, overwrite the **known** region with the
forward-diffused ground truth,

    x_{t−1}^known  = √ᾱ_{t−1} x₀^known + √(1−ᾱ_{t−1}) ε,
    x_{t−1}        = m ⊙ x_{t−1}^known + (1−m) ⊙ x_{t−1}^sampled,

with `m=1` on known frames/joints. Proved: the known region always equals the
forward-diffused ground truth, and at `t=0` it equals `x₀^known` exactly — so
streaming overlaps and user corrections are honoured. Optional resampling
(harmonisation) improves boundary coherence.

## 7. Kinematic constraints (penalties + projection)

Differentiable penalties, each `≥ 0` and zero iff satisfied:

* **Joint limits** `Σ max(0, |θ| − θ_max)²`; **projection** clamps `|θ| ≤ θ_max`.
* **Self-collision** — reuse Doc-04 `self_collision_penalty`.
* **Contact** — reuse Doc-05 contact field / `hard_contact`.
* **Temporal boundary** `‖x[boundary] − x_target[boundary]‖²` (streaming / segment
  seams).

**Innovation — constraint-projected sampling:** after each denoising step, project
the predicted `x₀` onto the feasible set (idempotent projection, proved), combining
the document's "kinematic projection" and "differentiable penalties" cleanly.

## 8. Consistency / rectified-flow distillation

* **Consistency model** `f_θ(x,t) = c_skip(t) x + c_out(t) F_θ(x,t)` with
  `c_skip(ε)=1, c_out(ε)=0` so `f(x_ε, ε) = x_ε` (**boundary condition**, proved).
  Self-consistency loss `‖f_θ(x_{t_{n+1}}, t_{n+1}) − f_{θ⁻}(x_{t_n}, t_n)‖`
  along a PF-ODE trajectory yields one/few-step sampling.
* **Rectified flow** `x_t = (1−t) x₀ + t z`, `t ∈ [0,1]`: the path is a straight
  line with **constant velocity** `dx/dt = z − x₀`, so `x₀ = x_t − t·velocity`
  (proved). A model predicting velocity samples by an Euler ODE.

Per the document, this is distilled **only after** the full model establishes
quality; here we build and verify the machinery.

## 9. Semantic cycle loss — auxiliary diagnostic only

A recognizer-on-generated-motion cycle score is provided strictly as a
**diagnostic**, clearly flagged: it can reward adversarial/unnatural motion and
must never be a primary training objective (per the document).

## 10. Evaluation and ablation

* **Multimodality** — mean pairwise distance among samples for one condition;
  **innovation — semantic-preservation-verified multimodality:** diversity measured
  only over samples that pass a meaning check, directly answering the document's
  "demonstrate that stochastic samples preserve meaning".
* **Jerk** — mean magnitude of the third temporal difference (smoothness).
* **Collision / contact** — reuse Docs 04-05. **Semantic accuracy** — recognizer
  diagnostic. **p95 generation time** — latency percentile.
* Baselines: deterministic Transformer, autoregressive tokens, CVAE, diffusion, at
  matched compute — honest scaffolding.

## 11. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| 7.0 | research + design/math spec (this doc) | done |
| 7a | parameterizations + schedule + posterior | done (9 tests) |
| 7b | temporal DiT denoiser (adaLN-Zero + cross-attn) | done (8 tests) |
| 7c | classifier-free guidance | done (6 tests) |
| 7d | part-aware noise schedules + loss weights | done (8 tests) |
| 7e | inpainting (streaming overlap + correction) | done (6 tests) |
| 7f | kinematic constraints + projection | done (7 tests) |
| 7g | consistency / rectified-flow distillation | done (7 tests) |
| 7h | evaluation + baselines + integration + regression | done (10 tests) |

Diffusion layer: 60 tests, green on two consecutive runs; whole project (1116
tests) green.

## 12. Findings (post-implementation)

**The parameterization triangle is exact and self-consistent.** All six ε/x₀/v
conversions round-trip to 1e-9, the v-inversions `x₀ = a x_t − b v`,
`ε = b x_t + a v` hold from `a²+b²=1`, and the three losses are exact reweightings
(`‖ε−ε_θ‖² = SNR·‖Δx₀‖²`, `‖v−v_θ‖² = (1/(1−ᾱ))·‖Δx₀‖²`). The schedule's DDPM
posterior matches the audited `GaussianMotionDiffusion` to 1e-8 — the new layer
reuses the core rather than re-deriving it.

**adaLN-Zero gives a provable identity start.** Each DiT block is the identity at
init (zero-init modulation ⇒ zero gates) and the whole model outputs 0 (zero-init
head), verified to 1e-6; once the modulation is activated the output depends on the
timestep, the conditioning vector, and the cross-attention tokens.

**CFG is parameterization-equivariant and trades diversity for control.** The same
`(1+w)·c − w·∅` in ε-space equals it in x₀-space (proved to 1e-9), and on real
guided Gaussian scores the guided variance provably shrinks as `w` grows — the
diversity/multimodality trade-off the document warns about. The guidance-annealing
innovation raises `w` early (semantics) and lowers it late (naturalness).

**Inpainting preserves the known region exactly.** Across a full reverse loop the
known frames always equal the forward-diffused ground truth, and at `t=0` they
equal `x₀^known` to machine precision — the streaming-overlap and user-correction
guarantee — while the unknown region is generated coherently.

**Constraints and few-step distillation are proved.** Every penalty is `≥0` and
zero iff satisfied; the joint-limit projection is idempotent and feasible, and
constraint-projected sampling produces a feasible final clip. The consistency
boundary `f(x, t_min)=x` holds exactly, and the rectified-flow straight path makes
few-step (even one-step) sampling exact with the true velocity.

**A subtle degeneracy was caught, not hidden.** An untrained x₀-prediction DiT
outputs 0, which collapses the reverse process to 0 (zero multimodality) — a
property of the *untrained init*, not the sampler. The multimodality tests
therefore use a non-degenerate (activated) denoiser, and the finding is documented
rather than masked. The evaluation includes the innovation
**semantic-preservation-verified multimodality**, directly answering "demonstrate
that stochastic samples preserve meaning".

**Honest scope holds.** Diffusion models residual uncertainty after grammar/timing
(not planner repair); no real motion corpus; all properties above are independent
of a trained model and proved exactly. The recognizer cycle score is a flagged
diagnostic only. Innovations beyond the sources: guidance annealing, part-aware
SNR capacity allocation, constraint-projected sampling, and
semantic-preservation-verified multimodality.
