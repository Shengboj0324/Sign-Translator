# 08 — Avatar Rendering Pipeline — Design and Mathematics

This document fixes **all** mathematics of the avatar-rendering layer before any
code, in the discipline of docs 01–07. It implements
`08_avatar_rendering_pipeline.md`: a parameter-stream interface with exact
contracts, three rendering tracks (rigged mesh with LBS / dual-quaternion skinning,
3D Gaussian Splatting, NeRF volume rendering), SO(3) frame pacing, a
linguistically-aware LOD, and — the document's core principle — a strict
**separation of rendering from linguistic evaluation**.

**Reuse.** The Doc-04 body model already implements LBS
(`forward_kinematics`, `G_rel`, `v = Σ_k w_k G'_k [v;1]`), Doc-06 the SO(3)
`slerp`, Doc-07 `joint_limit_penalty`/`project_joint_limits`, the pose layer the
quaternion↔matrix maps, the hand-graph `collision_rate`, and the speech layer
`percentile`. This layer builds on those primitives and does not re-derive them.

Primary sources studied:

* Kerbl, Kopanas, Leimkühler, Drettakis, *3D Gaussian Splatting* (arXiv:2308.04079)
  — anisotropic 3D Gaussians `(μ, Σ, α, c)`, `Σ = R S Sᵀ Rᵀ`, EWA projection
  `Σ' = J W Σ Wᵀ Jᵀ`, and front-to-back alpha compositing.
* Hu et al., *GaussianAvatar* (arXiv:2312.02134) and Yuan et al., *GAvatar*
  (arXiv:2312.11461) — animatable Gaussian humans (context; finger/topology/OOD
  risks the document flags).
* Mildenhall et al., *NeRF* (arXiv:2003.08934) — the volume-rendering integral
  `C = ∫ T(t) σ(t) c(t) dt`, `T(t)=exp(−∫σ)`, and its discretisation.
* Pavlakos et al., *SMPL-X* (arXiv:1904.05866) — the source rig (Doc-04).

## 0. Honest scope + separation of concerns (read first)

There is **no GPU engine, no Unity/Unreal/Blender, and no real image data** here.
We implement the **mathematics** of each rendering track exactly and validate on
small synthetic scenes; every property independent of a real asset (the contract
checks, LBS/DQS algebra, the splatting covariance/compositing, the volume-rendering
integral, SO(3) pacing, the LOD guarantee, the metric formulas) is proved exactly.
Photorealism, learned appearance, and engine integration are out of scope and drop
in later behind the same interface.

The document's central rule is enforced **in code**: appearance metrics
(PSNR/SSIM/LPIPS) are structurally separated from signing-correctness metrics so
that appearance can never be reported as, or mistaken for, correct signing.

## 1. Parameter-stream interface and contracts

The interface is a timestamped stream of frames, each carrying body/hand/face
parameters, plus a per-stream camera and avatar-metadata contract:

* **coordinate convention** (axis up, forward), **handedness** (right/left),
  **scale** (metres), **skeleton id**, **blendshape basis id**.
* **timestamps** strictly increasing, in seconds.

`validate_stream` checks all of these; a mismatch (e.g. a left-handed frame under a
right-handed contract, non-monotone timestamps, wrong skeleton) is rejected, never
silently coerced. **Deterministic replay:** rendering a stream is a pure function
of the stream (no wall-clock, seeded), so the same input yields byte-identical
output — verified.

## 2. Rigged mesh track (recommended baseline)

Linear blend skinning (the document's equation, = Doc-04):

    v_i' = Σ_k w_{ik} G_k(q) G_k(0)⁻¹ v_i.

Retargeting maps SMPL-X joints to a production skeleton by a joint correspondence
and a rest-pose alignment rotation `R_align ∈ SO(3)`. **Innovation —
handedness-certified retargeting:** `R_align` is constrained to `det R_align = +1`
(a proper rotation, never a reflection), so **mirroring errors are impossible by
construction** and left maps to left, right to right (proved via the determinant
certificate and an explicit left/right-preservation check). Wrist/finger joints
take priority in the correspondence. Joint limits are corrected with the Doc-07
projection. Facial blendshapes are linear: `f = f̄ + Σ_j ψ_j B_j` (verified affine
in the expression coefficients).

## 3. Dual-quaternion skinning

A dual quaternion `q̂ = q_r + ε q_d` (`ε²=0`) encodes a rigid transform: `q_r` a unit
quaternion (rotation), `q_d = ½ (0, t) ⊗ q_r` (translation `t`). Unit dual
quaternion: `‖q_r‖=1` and `⟨q_r, q_d⟩=0`. Recovery: `R` from `q_r`,
`t = 2 (q_d ⊗ q_r*)_{vec}`.

**Dual-quaternion linear blending (DLB):** `b̂ = Σ_k w_k q̂_k` with antipodality
(flip `q̂_k` if `⟨q_{r,k}, q_{r,0}⟩ < 0`), then normalise `b̂ / ‖b_r‖`, and apply the
resulting rigid transform to the vertex. Proved: transform ↔ DQ round-trips; DLB
with a single weight reproduces that transform; blending two identical transforms
is the identity; and on a large twist DLB **preserves volume** where LBS collapses
it (the candy-wrapper artefact the document names).

## 4. 3D Gaussian Splatting

Anisotropic 3D Gaussian `(μ, Σ, α, c)`, `G(x) = exp(−½ (x−μ)ᵀ Σ⁻¹ (x−μ))`, with

    Σ = R S Sᵀ Rᵀ,  R = rotation(quaternion),  S = diag(scales),

which is **positive semi-definite by construction** (`Σ = M Mᵀ`, `M = R S`). Project
to camera (view `W`), and form the screen-space covariance by the EWA/affine
approximation

    Σ' = J W Σ Wᵀ Jᵀ  (take the 2×2 image block),

where `J` is the Jacobian of the perspective map `(x,y,z) ↦ (f_x x/z, f_y y/z)`,

    J = [[f_x/z, 0, −f_x x/z²], [0, f_y/z, −f_y y/z²]].

Render by sorting Gaussians by depth and **front-to-back alpha compositing** (the
"over" operator): per pixel `p`, `α_i = α_i^opacity · exp(−½ (p−μ_i')ᵀ Σ'⁻¹ (p−μ_i'))`,

    C = Σ_i c_i α_i Π_{j<i}(1−α_j),   T_i = Π_{j<i}(1−α_j).

Proved: `Σ` PSD; the covariance transform; the 2D Gaussian normalises to
`2π√|Σ'|`; alpha compositing is associative in the "over" sense and a fully-opaque
near Gaussian occludes the ones behind it; depth ordering matters and is correct.

## 5. NeRF volume rendering

Discretised volume rendering along a ray with samples `t_i`, `δ_i = t_{i+1}−t_i`:

    α_i = 1 − exp(−σ_i δ_i),   T_i = exp(−Σ_{j<i} σ_j δ_j) = Π_{j<i}(1−α_j),
    C   = Σ_i T_i α_i c_i,     w_i = T_i α_i.

Proved: the transmittance recursion `T_{i+1} = T_i (1−α_i)`; the weights satisfy
`Σ_i w_i = 1 − T_{N}` (accumulated opacity, ≤ 1); the opaque limit
(`σ→∞ ⇒ α→1 ⇒ the first sample dominates`) and the transparent limit
(`σ=0 ⇒ w=0`); and that this is exactly the §4 alpha compositing with
`α_i = 1−exp(−σ_i δ_i)`.

## 6. Frame pacing and SO(3) interpolation

Resample a timestamped parameter stream to a target frame rate: translations by
linear interpolation, rotations by the Doc-06 **SLERP** (constant-speed geodesic,
exact endpoints). Dropped/duplicated frames are accounted explicitly. Lip/non-manual
channels are aligned to the same timeline. Replay is deterministic. Proved: the
resampled rotation at a keyframe equals that keyframe; the interpolated rotation is
the constant-speed geodesic; the output timeline is monotone at the target FPS.

## 7. Linguistically-aware level of detail

**Innovation:** LOD allocates a vertex/primitive budget by **linguistic
importance**, and is proved to **never drop fingers or facial cues** at any LOD
level — only torso/background detail is decimated. A per-part importance tier
(`FINGERS, FACE > HANDS_PALM > ARMS > TORSO > BACKGROUND`) gates decimation, and a
guarantee check confirms finger/face vertices survive every level.

## 8. Evaluation (appearance ≠ signing)

* **Motion-to-photon latency** (percentile, reuse speech), **dropped frames**
  (pacing), **temporal flicker** (frame-to-frame change), **mesh/hand penetration**
  (reuse Doc-04 collision), **silhouette IoU error**, **novel-view PSNR/SSIM**.
* **Structural separation:** appearance metrics live in an `AppearanceReport`
  typed object that is *incapable* of expressing a signing verdict, and a checker
  refuses to combine appearance and signing scores — the document's rule made
  unbreakable in code. `PSNR/SSIM assess appearance, not correct signing` is
  enforced, not just documented.

## 9. Integration + innovations

The rigged track consumes Doc-04 body output / Doc-07 diffusion samples through the
Doc-08 stream interface; SO(3) pacing reuses Doc-06; collision reuses Doc-04/05.
Innovations: handedness-certified retargeting, DQS volume preservation,
linguistically-aware LOD with a hard finger/face guarantee, and structurally
enforced appearance/signing separation.

## 10. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| 8.0 | research + design/math spec (this doc) | done |
| 8a | parameter-stream interface + contracts | done (8 tests) |
| 8b | rigged-mesh retargeting + blendshapes | done (8 tests) |
| 8c | dual-quaternion skinning | done (8 tests) |
| 8d | 3D Gaussian Splatting rasterizer | done (9 tests) |
| 8e | NeRF volume rendering | done (7 tests) |
| 8f | frame pacing + SO(3) interpolation | done (7 tests) |
| 8g | linguistically-aware LOD + consent | done (5 tests) |
| 8h | evaluation + separation + integration + regression | done (8 tests) |

Rendering layer: 69 tests, green on two consecutive runs; whole project (1176
tests) green.

## 11. Findings (post-implementation)

**Handedness is certified, so mirroring errors are impossible.** The retarget
alignment goes through the Doc-04 Kabsch reflection guard, which forces
`det R = +1`; a reflection cannot be produced, and a mirror-imaged target is left
with a large residual rather than silently mirrored. The contract carries its own
handedness certificate (the sign of the basis determinant).

**Dual-quaternion skinning preserves geometry where LBS collapses it.** On a 160°
twist blend, linear matrix averaging maps a unit vector to length `< 0.5` (the
candy-wrapper collapse), while DLB produces a proper `~80°` rotation that preserves
length exactly — the transform↔DQ round-trip, single-weight reproduction, and
antipodality handling are all proved.

**The splatting and volume-rendering math is exact.** `Σ = R S Sᵀ Rᵀ` is PSD by
construction, the projection Jacobian matches finite differences, the 2D Gaussian
integrates to `2π√|Σ'|`, and NeRF volume rendering reduces *exactly* to the §4
alpha "over" operator with `α = 1 − exp(−σδ)`, with weights summing to the
accumulated opacity `1 − T_final`. Depth ordering and occlusion are correct.

**Frame pacing is a true SO(3) geodesic.** Resampling at a keyframe returns that
keyframe, the midpoint is the constant-speed geodesic (equal geodesic distance to
both ends), the expression channel rides the same timeline (lip/non-manual sync),
and replay is deterministic.

**The document's core rule is enforced in code, not just documented.** Fingers and
facial cues are provably kept at *every* LOD level; and appearance metrics
(PSNR/SSIM/silhouette) live in an `AppearanceReport` that structurally cannot
express a signing verdict — `signing_quality_from_appearance` *raises* — so
appearance can never be reported as, or mistaken for, correct signing.

**Honest scope holds.** No GPU engine, no Unity/Unreal/Blender, no real image data;
every property above is the *mathematics* of rendering, proved on synthetic scenes,
and drops into a real renderer behind the same stream interface. Innovations:
handedness-certified retargeting, DQS volume preservation, linguistically-aware LOD
with a hard finger/face guarantee, and structurally-enforced appearance/signing
separation.
