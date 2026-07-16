# Mathematical specification

This document states the exact mathematics implemented in the code, so the
implementation can be audited against it. Every identity below is exercised by a
test in `tests/`.

## 1. Skeleton graph convolution

The skeleton is an undirected tree of `V = 27` joints. Let `N(i)` be the graph
neighbours of joint `i`, and let `hop(i)` be the unweighted shortest-path
distance from `i` to the centre joint (the neck).

**Spatial-configuration partitioning** (`K = 3`) assigns each ordered pair
`(i, j)` with `j ∈ N(i)` to a subset:

- subset 0 (root): `j = i` (self-loop),
- subset 1 (centripetal): `hop(j) < hop(i)`,
- subset 2 (centrifugal): `hop(j) > hop(i)`.

For a tree rooted (for distance) at the centre, adjacent joints always differ in
hop distance by exactly 1, so the assignment is unambiguous.

The centripetal and centrifugal partition matrices are **directed** (e.g. an
edge feeds a node from its parent but not vice-versa), so symmetric
normalisation is not applicable. Each subset adjacency `A_k` is instead **row
(random-walk) normalised**

```
Â_k = D_k^{-1} A_k,    D_k = diag(Σ_j A_k[i, j]).
```

Each row of `Â_k` is sub-stochastic (row sums ≤ 1). By the Gershgorin circle
theorem every eigenvalue then satisfies `|λ| ≤ 1`, so stacked graph convolutions
cannot amplify activations (verified in
`test_row_normalisation_is_substochastic_and_bounds_spectrum`).

**Graph convolution.** With input node features `X ∈ R^{C_in × V}` (per frame),

```
GraphConv(X) = Σ_{k=1}^{K} Â_k Xᵀ W_kᵀ  →  R^{V × C_out},
```

implemented as a single `1×1` convolution producing `K · C_out` channels
followed by an `einsum` contraction `nkctv, kvw → nctw` against `Â`.

## 2. Contrastive alignment (symmetric InfoNCE)

Let `f^m_i` and `f^l_i` be motion and language features of the `i`-th of `N`
paired examples. Projection heads `g_m, g_l` map them to the shared manifold and
L2-normalise:

```
z^m_i = g_m(f^m_i) / ||g_m(f^m_i)||,    z^l_i = g_l(f^l_i) / ||g_l(f^l_i)||.
```

With similarity logits `S_{ij} = ⟨z^m_i, z^l_j⟩ / τ` and correct-match targets
`i`, the loss is the symmetric cross-entropy

```
L = ½ [ CE(row-softmax(S), I) + CE(col-softmax(S), I) ].
```

The temperature is parameterised as a learnable log-scale `s = log(1/τ)`,
clamped at `s ≤ ln 100` (CLIP convention). Properties tested:
`L` is symmetric under swapping the modalities, equals a manual cross-entropy
computation, and is strictly smaller for correctly-paired than for shuffled
batches.

## 3. Gaussian diffusion for motion

Let `x_0` be a motion clip. With a variance schedule `β_1, …, β_T`,
`α_t = 1 − β_t`, and `ᾱ_t = Π_{s≤t} α_s`:

**Forward process** (closed form):

```
q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1 − ᾱ_t) I),
x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε,   ε ~ N(0, I).
```

Verified statistically (`test_q_sample_matches_analytic_mean_and_variance`): the
empirical mean and variance of `x_t` over 40 000 samples match `√ᾱ_t x_0` and
`1 − ᾱ_t`.

**Noise → signal inversion:**

```
x̂_0 = (x_t − √(1 − ᾱ_t) · ε) / √ᾱ_t = √(1/ᾱ_t) x_t − √(1/ᾱ_t − 1) ε.
```

Verified to invert `q_sample` to `1e-4` for multiple timesteps
(`test_predict_start_inverts_q_sample_exactly`). Note this inversion is
ill-conditioned as `ᾱ_t → 0` (large timesteps under the cosine schedule): the
factor `√(1/ᾱ_t)` amplifies float32 round-off, so the exact-inversion test uses
the well-conditioned linear schedule.

**Reverse posterior** used for the sampler:

```
q(x_{t-1} | x_t, x_0) = N(x_{t-1}; μ̃_t, β̃_t I),
β̃_t = (1 − ᾱ_{t-1}) / (1 − ᾱ_t) · β_t,
μ̃_t = (√ᾱ_{t-1} β_t)/(1 − ᾱ_t) · x_0 + (√α_t (1 − ᾱ_{t-1}))/(1 − ᾱ_t) · x_t.
```

The stored coefficients `posterior_variance`, `posterior_mean_coef1/2` are tested
against these formulas elementwise.

**Training objective** (simplified DDPM): a network `ε_θ(x_t, t, c)` predicts the
noise, trained with

```
L_diff = E_{t, x_0, ε} || ε − ε_θ(√ᾱ_t x_0 + √(1 − ᾱ_t) ε, t, c) ||².
```

**Schedules.** Linear `β_t ∈ [β_start, β_end]`, or the cosine schedule of Nichol
& Dhariwal with `ᾱ_t = cos²((t/T + s)/(1 + s) · π/2) / cos²(...)|_{t=0}`,
`β_t = clip(1 − ᾱ_t/ᾱ_{t-1}, 0, 0.999)`. Both produce `β ∈ (0, 1)` and a strictly
decreasing `ᾱ_t` (tested).

**Sampling.** Ancestral DDPM (`sample`) applies the reverse posterior for
`t = T-1 … 0`, adding noise for `t > 0`. DDIM (`ddim_sample`) uses a strided
schedule with the deterministic (`η = 0`) or stochastic update

```
x_{t_next} = √ᾱ_{t_next} · x̂_0 + √(1 − ᾱ_{t_next} − σ²) · ε_θ + σ · ε,
σ = η √((1 − ᾱ_{t_next})/(1 − ᾱ_t) · (1 − ᾱ_t/ᾱ_{t_next})).
```

## 4. Joint objective

```
L = w_c · L_InfoNCE + w_d · L_diff.
```

The diffusion generator is conditioned on the language latent `z^l` so that the
same vector which contrastive learning aligns with motion also drives motion
synthesis. `test_training_reduces_loss_on_synthetic_data` confirms the combined
objective decreases under optimisation — the end-to-end correctness check.
