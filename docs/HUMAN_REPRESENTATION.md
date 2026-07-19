# 04 — 3D Human Representation (SMPL-X) — Design and Mathematics

This document fixes **all** mathematics of the 3D human-representation layer before
any code is written, in the discipline used for docs 01–03. It implements
`04_3d_human_representation_smplx.md`. We model **SMPL-X**-style expressive bodies:
a differentiable mesh `M(q_t, β)` built from blend shapes, a joint regressor, pose
correctives, and linear blend skinning (LBS), with rotations carried in a
**continuous** 6D representation.

Primary sources studied:

* Zhou, Barnes, Lu, Yang, Li, *On the Continuity of Rotation Representations in
  Neural Networks*, arXiv:1812.07035 — 6D continuous rotations. Key theorem: every
  representation of SO(3) in ℝⁿ with n ≤ 4 is **discontinuous** (so Euler and
  quaternions are hard for networks); continuous representations exist in 5D/6D.
* Pavlakos et al., *Expressive Body Capture: 3D Hands, Face, and Body from a Single
  Image* (SMPL-X / SMPLify-X), arXiv:1904.05866 — the model `M(β,θ,ψ)`, the
  SMPLify-X fitting objective (2D re-projection under a robust error, a learned
  pose prior, a fast interpenetration penalty), and the honest caveat that
  monocular fits are **pseudo** ground truth.
* Cai et al., *SMPLer-X* (arXiv:2309.17448) — expressive whole-body pose foundation
  model (context for scale and the whole-body joint layout).
* Yu et al., *SignAvatars* (arXiv:2310.20436) and *SignAvatar / ASL3DWord*
  (arXiv:2405.07974) — holistic 3D sign datasets in SMPL-X (the eventual real
  data; licensed, not fetched here).

## 0. Honest scope (read first)

The **actual SMPL-X model tensors** (template mesh, shape/expression/pose
blendshape bases, joint regressor, skinning weights, kinematic tree) are licensed
data behind registration at `smpl-x.is.tue.mpg.de`; the source document itself
warns that *"research access is not automatic production permission."* We therefore
do **not** download them. Instead we implement the SMPL-X **pipeline exactly** —
every equation below — parameterised by model tensors, and validate on a small
**controllable toy body model** whose template/bases/weights/tree we set and whose
ground truth we therefore know exactly. Every property that does not depend on the
specific *learned* bases (rest-pose identity, rigid equivariance, blendshape
linearity, LBS partition-of-unity, differentiability, all rotation algebra, all
metrics, the fitting-objective behaviour) is proved exactly; the realistic mesh
drops in later as data with no code change. The pose prior is a proper
Mahalanobis/GMM prior (VPoser, a VAE, is the production choice and is noted as
such). No real capture data: fitting is validated on synthetic projections with
known ground truth. Pseudo-GT is never treated as exact (per the document).

## 1. Rotation representations

SMPL-X poses are per-joint rotations. We **regress** rotations in the 6D
representation and convert to SO(3); we never regress Euler angles.

### 1.1 Axis-angle ↔ SO(3) (Rodrigues)

SMPL-X's native pose encoding is axis-angle `θ ∈ ℝ³`, angle `φ = ‖θ‖`, axis
`k = θ/φ`. With `K = [k]_×` (skew-symmetric),

    R(θ) = I + sin φ · K + (1 − cos φ) · K²   =   exp(φ K).

Numerically, `sin φ / φ → 1` and `(1 − cos φ)/φ² → 1/2` as `φ → 0`; we use the
Taylor forms below a threshold so `R(0) = I` with a finite gradient. Inverse
(log map): `φ = arccos((tr R − 1)/2)`, axis from the skew part `(R − Rᵀ)/2`.

### 1.2 Quaternion → SO(3)

Unit `q = (w, x, y, z)`:

    R = [[1−2(y²+z²),  2(xy−wz),   2(xz+wy)],
         [2(xy+wz),    1−2(x²+z²), 2(yz−wx)],
         [2(xz−wy),    2(yz+wx),   1−2(x²+y²)]].

Double cover: `q` and `−q` give the same `R` (a source of discontinuity for
learning).

### 1.3 6D → SO(3) (Zhou et al., Gram–Schmidt)

The 6D vector is two 3-vectors `(a, b)` (the first two columns of a rotation).
Orthonormalise:

    r₁ = a / ‖a‖,
    r₂ = (b − (r₁ᵀ b) r₁) / ‖b − (r₁ᵀ b) r₁‖,
    r₃ = r₁ × r₂,
    R  = [r₁ | r₂ | r₃]   (columns).

The right inverse SO(3) → ℝ⁶ drops the third column (take `R[:, :2]`). Feeding
`R[:, :2]` back through Gram–Schmidt returns `R` exactly (the first two columns are
already orthonormal, so GS is the identity on them).

### 1.4 Continuity — the whole point

Define the representation map `f: SO(3) → ℝ⁶`, `f(R) = R[:, :2]` (continuous), and
the reconstruction `g: ℝ⁶ → SO(3)` (Gram–Schmidt, continuous on the dense set
where `a, b` are independent). `g ∘ f = id`, so 6D is a **continuous** section.
By Zhou et al. no continuous section exists into ℝ⁴ or fewer, so Euler and
quaternion encodings must jump somewhere. **We prove this empirically**: sweep a
frame rotating continuously through an angle that crosses an Euler gimbal / a
quaternion sign flip, and show `max‖Δ(6D)‖` stays `O(Δangle)` while the Euler /
quaternion encodings exhibit an `O(1)` jump.

### 1.5 Geodesic distance on SO(3)

    d(R₁, R₂) = arccos( (tr(R₁ᵀ R₂) − 1) / 2 ) ∈ [0, π].

This is the rotation angle of `R₁ᵀR₂`. It is a bi-invariant metric: `d ≥ 0`,
`d = 0 ⇔ R₁ = R₂`, symmetric, and satisfies the triangle inequality. Used for
rotation-error evaluation and as a proper loss on SO(3).

## 2. Representation contract (the per-frame state)

    q_t = (γ_t, θ^b_t, θ^{lh}_t, θ^{rh}_t, θ^{jaw}_t, θ^{eye}_t, ψ_t),   β = const.

* `γ_t ∈ ℝ³` root translation.
* `θ^b` body-joint rotations (incl. global orient), `θ^{lh}, θ^{rh}` 15 hand joints
  each, `θ^{jaw}` 1, `θ^{eye}` 2 — all stored as **6D** rotations.
* `ψ_t ∈ ℝ^{n_e}` expression coefficients.
* `β ∈ ℝ^{n_β}` identity shape, **sequence-constant**.

The contract enforces the design constraint *"keep identity shape separate from
linguistic motion to prevent signer leakage"*: `β` lives outside the per-frame
motion `q_t`, and the API refuses to fold it in. Serialization is a deterministic
flat-vector round trip; part boundaries are explicit so hands/face can be given
higher fidelity than the torso.

## 3. The SMPL-X forward model `M(q, β)`

Additive template in rest pose:

    T_P(β, θ, ψ) = T̄ + B_S(β) + B_E(ψ) + B_P(θ),

* `T̄ ∈ ℝ^{N_v×3}` template mesh,
* `B_S(β) = Σ_n β_n S_n` shape blendshapes (**linear** in β),
* `B_E(ψ) = Σ_n ψ_n E_n` expression blendshapes (**linear** in ψ),
* `B_P(θ) = Σ_j (R_j(θ) − R_j(θ*)) P_j` pose correctives, `θ*` rest ⇒ term 0 at
  rest; linear in the 3×3 rotation-matrix features.

Joints regress from the **shaped** template:

    J(β) = 𝒥 · (T̄ + B_S(β)),     𝒥 ∈ ℝ^{N_J×N_v}.

Kinematic tree: with parent `p(j)` and per-joint rotation `R_j`,

    G_j = G_{p(j)} · [[R_j, J_j − J_{p(j)}], [0, 1]]   (root: offset = J_root).

Remove the rest transform so the rest pose maps to identity:

    G'_j = G_j · [[I, −J_j], [0, 1]].

Linear blend skinning of every vertex `i` (blend weights `w_{ij} ≥ 0`,
`Σ_j w_{ij} = 1`):

    v_i = Σ_j w_{ij} · ( G'_j · [T_{P,i}; 1] )_{1:3}.

Global orientation is the root rotation inside `G`; root translation `γ` is added
last. Output posed joints are the translation parts of `G_j` (plus `γ`).

Provable properties (on the toy model, exactly):

1. **Rest identity.** `β=0, ψ=0, θ=0, γ=0 ⇒ v = T̄`, `J = 𝒥 T̄` (all `G'_j = I`).
2. **Blendshape linearity at rest.** `M(β, ψ, θ=0) = T̄ + B_S(β) + B_E(ψ)`, exactly
   affine in `(β, ψ)`; verified by finite differences matching the bases.
3. **Global rigid equivariance.** Composing global orient with `R_g` and setting
   `γ ↦ R_g γ + t` maps every output point `x ↦ R_g x + t`.
4. **Partition of unity.** `Σ_j w_{ij} = 1` ⇒ applying the *same* rigid transform
   to all joints applies it once to each vertex (no scale drift).
5. **Differentiability.** Gradients flow to `θ` (6D), `β`, `ψ`, `γ`.

## 4. Camera projection and robust re-projection

Pinhole intrinsics `K = [[f_x,0,c_x],[0,f_y,c_y],[0,0,1]]`. Perspective projection

    Π(X) = (f_x · X/Z + c_x,  f_y · Y/Z + c_y),   Z > 0.

Weak-perspective (scale `s`, offset `(t_x,t_y)`): `x = s·(X, Y) + (t_x, t_y)`; a
valid approximation when depth variation ≪ distance. Re-projection term with
detection confidences `c_i` and a robust error `ρ`:

    L_2D = Σ_i c_i · ρ( ‖Π(J_i(q,β)) − k_i‖ ).

Geman–McClure robustifier `ρ(r) = r² / (r² + σ²) ∈ [0, 1)`: quadratic near 0,
**bounded** as `r → ∞`, with a **redescending** influence `ρ'(r) = 2σ²r/(r²+σ²)²`
that → 0 for large `r`, so a few grossly wrong detections cannot dominate the fit.
Verified: a known camera projects known 3D to the correct 2D (round trip); `ρ`
saturates and its derivative redescends.

## 5. Fitting objective

    L_fit = λ_2D · L_2D + λ_d · L_depth + λ_p · L_prior + λ_v · ‖Δq‖₁ + λ_c · L_collision.

* **Pose prior** `L_prior`: Mahalanobis `(θ − μ)ᵀ Σ⁻¹ (θ − μ)` (or GMM negative
  log-likelihood `−log Σ_g π_g 𝒩(θ; μ_g, Σ_g)`). `≥ 0`, minimised at the mean.
  (Production SMPLify-X uses **VPoser**, a VAE prior; noted, not claimed.)
* **Temporal smoothness** `‖Δq‖₁ = Σ_t ‖q_{t+1} − q_t‖₁`: `= 0` iff the sequence is
  constant; L1 favours piecewise-constant (sparse-velocity) motion.
* **Collision** `L_collision`: sphere proxies of radius `r_j` at joints; for
  non-adjacent pairs `Σ max(0, (r_i+r_j) − ‖c_i−c_j‖)²`. Zero iff no proxy overlap,
  positive and once-differentiable under penetration (a squared hinge).
* **Depth** `L_depth`: optional generic residual `‖Z(q,β) − z_obs‖` where depth is
  available; off by default (no depth sensor data here).

Verified: gradient descent on `L_fit` from a perturbed initialisation recovers a
known pose from enough synthetic views (identifiability), and each term behaves as
specified in isolation.

## 6. Evaluation — never MPJPE alone

Per the document, *"small fingertip errors can change meaning"*, so we evaluate
rotations and surface landmarks, not just body-joint position.

* **MPJPE** `= mean_i ‖Ĵ_i − J_i‖` after root (pelvis) alignment.
* **PA-MPJPE**: Procrustes/Kabsch align `(s, R, t)` then MPJPE. Kabsch is optimal:
  `R = U diag(1,…,1, det(UVᵀ)) Vᵀ` from the SVD `UΣVᵀ` of the cross-covariance,
  the reflection guard keeps `det R = +1`; proved to minimise `‖sRA+t − B‖²`.
* **Geodesic rotation error** `= mean_j d(R̂_j, R_j)` (§1.5), catching wrong joint
  *orientation* even when position looks fine.
* **V2V / PVE** `= mean_i ‖v̂_i − v_i‖` over the mesh (surface, not just joints).
* **Fingertip-weighted error**: weighted MPJPE with fingertips up-weighted; we show
  a fixed fingertip displacement raises this far more than a torso-weighted error —
  the quantified reason MPJPE alone is inadequate for signing.

## 7. Identity ⟂ motion (anti-leakage)

The contract makes rotations/expression carry motion and `β` carry identity. We
prove separation, not merely assert it:

* For identities `β₁ ≠ β₂` under the **same** motion `q`, the pose parameters are
  *literally identical* (θ is shared), so any function of motion-only features is
  invariant to identity.
* `β` is **not recoverable** from `θ` (independent by construction): a probe
  regressing `β` from motion-only features cannot beat predicting the mean.

This operationalises *"keep identity shape separate … to prevent signer leakage."*

## 8. Integration

The toy/real body model outputs joints; the existing 27-joint sign skeleton
(`skeleton/graph.py`) consumes joints via `data/adapters.py`, which already lists
*"an SMPL-X joint regressor"* as a source. Stage 4h wires
`body-model joints → 27-joint skeleton` and round-trips the representation through
the motion pipeline, then runs whole-chain cycle stress and consecutive full
regressions.

## 9. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| 4.0 | research + design/math spec (this doc) | done |
| 4a | rotation representations + conversions (6D/quat/axis-angle ↔ SO(3), geodesic) | done (17 tests) |
| 4b | representation contract / state vector (q_t, β separation, serialization) | done (9 tests) |
| 4c | LBS + kinematic-tree forward body model (toy SMPL-X) | done (9 tests) |
| 4d | camera projection + robust re-projection | done (13 tests) |
| 4e | priors + smoothness + collision → full fitting objective | done (12 tests) |
| 4f | evaluation metrics beyond MPJPE | done (11 tests) |
| 4g | identity/motion separation + signer-leakage tests | done (5 tests) |
| 4h | integration + cycle stress + full regression | done (7 tests) |

Pose layer: 83 tests, green on two consecutive runs; whole project (901 tests) green.

## 10. Findings (post-implementation)

**The 6D-continuity claim is demonstrated, not just cited.** Sweeping a frame
through `angle = π` about a fixed axis, the 6D encoding's largest adjacent change
stays `O(step)` while the canonical (`w ≥ 0`) quaternion jumps by `≈ 2` — an
`O(1)` discontinuity — at the sign flip. This is Zhou et al.'s theorem made
concrete and is the reason we regress rotations in 6D.

**Global rigid equivariance is exact (1e-16), and it forced a specific design
choice.** Prepending a global rotation to the root orientation rotates every
output vertex and joint about the pelvis to machine precision. This holds only
because (a) skinning weights are a partition of unity and (b) the **pose
correctives exclude the root joint** — if global orientation fed the pose
blendshapes, changing where the body faces would spuriously deform it. The test
suite pins the precise internal fact (the pose feature is invariant to the root
rotation) rather than a fragile downstream consequence.

**Two float-artifact traps were distinguished from real error, per standing
discipline.** (1) A `cdist`-based "pairwise distances preserved" check failed at
`2e-8` purely because `sqrt` amplifies float error near zero distances; the real
equivariance holds at `1e-16`, so the test now asserts the exact internal
invariant. (2) `geodesic_distance` of *bit-identical* rotations returns `≈ 3e-8`,
not 0, because `arccos` near 1 has `sqrt`-amplified sensitivity; identity-
invariance is proved by `torch.equal`, and geodesic-zero checks use a `1e-6`
tolerance. In both cases the exact-math proof replaced the fragile one.

**Monocular fitting is underdetermined — shown deterministically, not hand-waved.**
Two 3D joint sets related by sliding each joint along its camera ray project to
*identical* 2D under one camera (re-projection cannot distinguish them) but differ
substantially in 3D; a second camera makes them distinguishable. A multi-view
gradient-descent fit then recovers 3D pose to `MPJPE < 1e-2`. This matches the
source document's warning and is the honest reason pseudo-GT from monocular fits
is never treated as exact.

**Identity/motion separation is enforced in the type system and probed for
leakage.** `beta` lives outside per-frame motion; the motion feature vector
excludes it; a linear probe cannot recover `beta` from motion features
(normalised error `> 0.85`) yet recovers it near-perfectly (`< 0.05`) when `beta`
is folded in — so the guard has power. Per-joint *world orientations* are
identity-invariant by construction (a pure function of rotations and the tree).

**Honest scope holds.** Every property proved here is independent of the licensed
SMPL-X bases; the realistic model tensors drop into `BodyModelTensors` unchanged.
The pose prior is a Mahalanobis/GMM prior (VPoser is the production choice, noted,
not claimed); no real capture data is used.
