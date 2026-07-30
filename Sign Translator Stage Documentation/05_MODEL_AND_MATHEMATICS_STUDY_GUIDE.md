# 05 — Model and Mathematics Study Guide

## 1. How to study the mathematics

For every formula, the group should answer four questions:

1. What variables does it operate on?
2. What assumptions make the result valid?
3. Where is it used in the active training graph?
4. What experiment could falsify its claimed usefulness?

An identity can be mathematically correct without improving translation. Unit
tests establish algebraic behavior; real-data experiments establish relevance.

## 2. Motion representation

### Cartesian joints

The active model predicts:

```text
x in R^(N x C x T x V)
```

where `C = 3` coordinates, `T` frames, and `V = 27` joints.

Advantages:

- simple tensor contract;
- direct position error;
- easy compatibility with ST-GCN.

Limitations:

- unconstrained bone lengths;
- coordinate-frame sensitivity;
- no guaranteed valid rotations;
- weak hand articulation;
- no facial expression state;
- difficult direct retargeting to an avatar.

### Rotation-based motion

The specialized motion stack uses six-dimensional rotation representations. Two
vectors are orthonormalized to form the first two columns of a rotation matrix;
the third follows from their cross product.

For rotations `R` and `R_hat`, the geodesic error is:

```text
d(R, R_hat) = arccos((trace(R^T R_hat) - 1) / 2)
```

This measures angular distance on `SO(3)`, unlike elementwise matrix MSE.

### Study question

Should the generator predict:

- Cartesian joints;
- local rotations plus root translation;
- body-model parameters;
- latent motion tokens decoded into rotations?

The decision must be made before real dataset export because it determines
fitting, normalization, losses, rendering, and evaluation.

## 3. Spatiotemporal graph recognition

The active recognizer uses a skeleton graph and ST-GCN. For adjacency partitions
`A_k` and learned transforms `W_k`, a graph layer is conceptually:

```text
Y = sum_k A_k X W_k
```

Temporal convolution then models local motion across frames.

Important assumptions:

- the joint topology matches the input layout;
- missing joints are not silently interpreted as valid zeros;
- frame lengths are masked correctly;
- global normalization does not erase linguistically important spatial loci;
- pooling over joints does not discard detailed handshape.

Required real-data ablations:

- fixed versus adaptive adjacency;
- Cartesian versus rotation input;
- confidence-aware versus confidence-blind input;
- body-only versus body-plus-dense-hands-plus-face;
- signer-independent versus signer-overlapping splits.

## 4. CTC recognition

Connectionist Temporal Classification sums over frame-level alignments that
collapse to a target label sequence:

```text
P(y | x) = sum_(pi in B^-1(y)) product_t P(pi_t | x)
L_CTC = -log P(y | x)
```

CTC is useful when frame-level sign boundaries are unknown. It assumes a
monotonic alignment and does not directly model arbitrary reordering.

### Critical feasibility rule

The input length must be sufficient not only for the target labels but also for
separating adjacent repeated labels with blanks. After acoustic subsampling, the
usable speech length is shorter than the raw feature length.

`zero_infinity=True` prevents NaNs, but it must not be used as a substitute for
validating examples: impossible samples can otherwise contribute zero loss.

## 5. Semantic planning and grammar

The active `GlossPlanner` is a teacher-forced Transformer trained to learn a
synthetic vocabulary substitution. It does not implement the richer typed plan
or SIR.

The specialized planner and grammar packages model:

- typed events;
- constrained serialization;
- lexicon-grounded decoding;
- interval relations;
- non-manual scope;
- spatial reference.

For intervals `a = [a_s, a_e]` and `b = [b_s, b_e]`, a relation such as
precedence can be encouraged with a hinge penalty:

```text
L_precede = max(0, a_e - b_s + margin)
```

The distinction to preserve is:

- structural validation asks whether the graph is well-formed;
- differentiable losses ask whether predicted timings satisfy desired relations;
- human evaluation asks whether the structure communicates the intended meaning.

## 6. Contrastive alignment

Motion and language features are projected and normalized:

```text
z_m = g_m(f_m) / ||g_m(f_m)||
z_l = g_l(f_l) / ||g_l(f_l)||
```

For batch similarity:

```text
S_ij = z_m_i dot z_l_j / tau
```

the active system applies symmetric cross-entropy over rows and columns.

### Main real-data problem

Diagonal InfoNCE assumes each row has exactly one positive and every other batch
item is negative. In sign language, two clips can express the same proposition
or equally valid variants. Treating those as negatives can teach the model to
encode signer, camera, or duration rather than semantics.

Required improvements:

- multi-positive or supervised contrastive targets;
- false-negative masking;
- signer/source adversarial probes;
- hard negatives differing in one licensed semantic feature;
- retrieval evaluation by meaning, not file identity.

## 7. Diffusion motion generation

The forward process adds Gaussian noise:

```text
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon
```

The model predicts noise, clean motion, or velocity-like parameterization,
depending on the implementation. The active configuration predicts `x_0` and
adds a velocity loss:

```text
L = MSE(x_hat_0, x_0)
    + lambda_v MSE(Delta x_hat_0, Delta x_0)
```

Classifier-free guidance combines conditional and unconditional predictions:

```text
x_guided = x_uncond + w(x_cond - x_uncond)
```

### Missing active constraints

The real training objective should evaluate:

- geodesic rotation error;
- joint-position error after forward kinematics;
- bone-length consistency;
- velocity, acceleration, and jerk;
- joint limits;
- self-collision;
- required contact;
- foot or body stability where relevant;
- handshape and orientation;
- non-manual timing;
- boundary continuity for streaming.

These losses should not all be added with arbitrary weights. Measure gradient
magnitudes, conflicts, and ablations.

## 8. Motion tokenization and duration

Residual vector quantization represents motion using several codebooks:

```text
r_0 = z
c_i = nearest_code_i(r_i)
r_(i+1) = r_i - c_i
z_q = sum_i c_i
```

Potential benefits:

- shorter discrete sequences;
- reusable motion primitives;
- efficient autoregressive or masked pretraining;
- separation of temporal planning from low-level reconstruction.

Risks:

- codebook collapse;
- insufficient hand or facial fidelity;
- quantization artifacts at sign boundaries;
- tokens encoding signer identity;
- duration model learning corpus bias rather than grammar.

The tokenizer must first demonstrate reconstruction quality on real held-out
motion before it is used as the target for language-to-motion training.

## 9. Body, hand, and facial modeling

The toy body model verifies forward-kinematic and skinning identities but does
not include licensed production tensors. A real pipeline needs:

- a versioned body, hand, and face basis;
- a documented anatomical joint mapping;
- identity/shape treatment;
- a fit-quality score and uncertainty;
- retargeting tests on actual avatar rigs;
- face and hand visibility handling.

Non-manual channels are multi-label, not mutually exclusive. Brow raise, mouth
activity, gaze, and head movement may overlap, so independent sigmoid channels
are more appropriate than a single softmax.

## 10. Evaluation mathematics

No single metric can establish sign-language quality.

### Required layers

1. **Input:** ASR WER/CER, calibration, coverage.
2. **Semantic plan:** proposition accuracy and hallucination rate.
3. **Grammar:** sign-order and non-manual-scope correctness.
4. **Motion:** rotations, joints, contact, smoothness, and diversity.
5. **Recognition:** independent motion-to-sign recovery.
6. **Comprehension:** human proposition precision/recall.
7. **Runtime:** latency, revision, failure, and abstention behavior.

### Statistical discipline

- pre-declare primary endpoints;
- report effect sizes and confidence intervals;
- use signer/source-held-out items as the statistical units;
- do not treat random seeds as independent language samples;
- lock the final test set before model selection;
- correct for multiple exploratory comparisons;
- report failure slices, not only averages.

## 11. Mathematical work still needed

The largest remaining mathematical questions are empirical, not algebraic:

- Which representation preserves linguistic information most efficiently?
- Does explicit SIR conditioning outperform direct gloss conditioning?
- Do hard negatives remove signer and duration shortcuts?
- Which non-manual channels materially improve comprehension?
- Does cycle consistency correlate with human understanding?
- What uncertainty measure predicts semantic failure?
- How should conflicting multi-task gradients be balanced?

Each question needs a falsifiable experiment, a baseline, and a pre-declared
decision rule.

