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
| 8i | real-state retarget onto the MakeHuman rig + Blender render (§12) | done (25 tests) |

Rendering layer: 69 tests, green on two consecutive runs at that stage; the
whole-project total recorded then was 1,176 tests. These are stage snapshots, not the current verdict; rerun the suite (see the README reproducibility section).

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

## 12. Stage 8i: real-state retarget

Real and generated canonical states (`MultichannelMotion`, conventions
`mediapipe-tasks-swing-ik-y-up-v1` and `sapiens-two-view-swing-ik-y-up-v1`) are now
rendered on the audited MakeHuman default rig instead of a stick skeleton:
`signtranslator/avatar_render/retarget.py` (binding + retarget),
`signtranslator/avatar_render/retarget_check.py` (measurements) and
`scripts/render_avatar.py` (CLI). `w3_render_stream.py --synthetic-demo` is unchanged.

**Binding (computed, not tuned).** `build_retarget_binding(rig)`:

* *Axes.* Canonical: x = signer's left (image right), y up, z toward the camera.
  The rig's rest geometry is checked to have `.L` shoulders at +x, the neck above
  the hips and the eyes in front of the head joint, so the axis map is the identity
  (no reflection is possible).
* *Reference.* The project's own landmark IK (`video_landmarks.solve_rotations`) is
  run on the rig's rest skeleton (shoulders, elbows, wrists, hips, knees, ankles and
  the 21 hand points read from rest bone heads). That gives each canonical joint's
  world frame `W_ref` for the rig's rest pose; a state equal to it poses the rig at
  rest exactly (tested).
* *Retarget.* `Δ_j = W_j W_refᵀ`; each rig bone driven by joint `j` gets
  `G_b = Δ_j Rest_b`. Because the IK reproduces bone directions exactly, the rig's FK
  reproduces every canonical bone direction whatever the lengths.
* *Mapping.* Pelvis → `root`; chest → `spine02`, `spine01` (pelvis→chest blended over
  `spine05`–`spine03`); upper arm → `upperarm01/02`; forearm → `lowerarm01/02`; hand
  (full frame) → `wrist`; Thumb/Index/Middle/Ring/Pinky 1–3 → `finger1..5-1..3` on both
  sides (all 30 finger joints); `head_rotation` (camera frame) → `head`, with
  chest→head blended over `neck01`–`neck03`. Clavicles, shoulders, metacarpals and
  breasts keep their rest relation to the parent. Legs are held at rest by default
  (`--drive-legs` poses them): a waist-up render does not need them and generated
  states carry unconstrained leg rotations.
* *Twist* (not observed by swing IK; changes no direction): the upper arm is twisted
  about the shoulder→elbow segment so the elbow bends about the rig's rest hinge
  (weighted by the bend angle, zero for a straight arm), the forearm keeps its rest
  relation to it, and `lowerarm02` takes half the forearm-to-hand twist so pronation
  does not collapse at the wrist.
* *Missing values.* Gaps of at most 0.5 s are slerped on the state's clock; longer
  gaps (and clip ends) ease to the rig's rest relation over 0.25 s; never-valid joints
  keep it. Every filled frame is counted in `verification.json`, and fills are never
  scored.

**Face.** MediaPipe blendshapes drive the audited MakeHuman face pose units
(`face-poseunits.bvh`, bone rotations) by log-linear blending, after subtracting each
coefficient's clip median (`--face-baseline none` uses raw scores; MediaPipe scores
on signing video carry large constant offsets, e.g. `browDown*` ≈ 0.6–0.9 and
`eyeSquint*` ≈ 0.6 on neutral faces). The `blink` channel replaces `eyeBlink*`. Left/right
names are taken as the signer's own side (MakeHuman `Left*` units move the rig's +x
side; checked on the mesh).

| rendered (state coefficient → MakeHuman unit) | |
|---|---|
| brows | `browDown*` → `*BrowDown`; `browInnerUp` → both `*InnerBrowUp`; `browOuterUp*` → `*OuterBrowUp` |
| eyelids | `eyeBlink*`/`blink` → `*UpperLidClosed`; `eyeWide*` → `*UpperLidOpen`; `eyeSquint*` → `*LowerLidUp` |
| eyes | valid `left_gaze`/`right_gaze` aim the eye bones; else `eyeLookUp/Down/In/Out*` → `*EyeUp/Down/turnLeft/turnRight` |
| cheeks, nose | `cheekSquint*` → `*CheekUp`; `cheekPuff` → `CheeksPump`; `noseSneer*` → `NoseWrinkler` |
| jaw | `jawOpen` → `JawDrop`; `jawForward` → `ChinForward`; `jawLeft/Right` → `ChinLeft/Right` |
| lips | `mouthSmile*` → `Mouth*PullUp`; `mouthFrown*` → `Mouth*PullDown`; `mouthStretch*` → `Mouth*PullSide`; `mouthLeft/Right` → `MouthMoveLeft/Right`; `mouthPucker` → `LipsKiss`; `mouthFunnel` → ½ `UpperLipForward` + ½ `lowerLipForward`; `mouthRollLower/Upper` → `lower/UpperLipBackward`; `mouthLowerDown*` → `lowerLipDown`; `mouthUpperUp*` → `UpperLipUp`; `mouthShrugLower` → `lowerLipUp` |
| **not represented** | `_neutral` (not an expression), `mouthClose` (no unit separates lips from jaw), `mouthPressLeft/Right`, `mouthDimpleLeft/Right`, `mouthShrugUpper`; no tongue coefficient exists in the state, so the tongue units are unused |

The body has no eyebrow, hair or lip geometry; brows are painted on the skin carried
by the brow bones (so raise/furrow read at review resolution), eyes, teeth and tongue
are the audited attachments coloured from their pinned textures (the transparent
cornea shell is dropped), and the body wears a plain dark top so the hands contrast.

**CLI.** `render_avatar.py --state S.npz [--sha256 H] | --key K --sha256 H [--store r2|dir:…]
--out-dir NEW --allow-inferred [--resolution 384] [--start-seconds/--max-seconds]` writes
`bundle.npz` (posed meshes, provenance), `frames/`, `avatar.mp4` (H.264 4:2:0, PTS from the
state's clock; `--lossless` for 4:4:4 CRF 0) and `verification.json`. Rendering is Blender
Workbench in `--background --factory-startup` with a fixed orthographic front camera.

**Measurements** (`verification.json`; reference = `canonical_skeleton.skeleton_positions`
on the stored state, rig side = bone frames returned by `MakeHumanRig.pose`; only frames
where the joint and all ancestors are valid as stored; head scored against the stored
`head_rotation`, since the canonical skeleton has no head orientation). Three ASLLRP DAI 2
continuous utterances (plan manifest v2 records 0–2) and one generated state
(`t2m-diffusion-m1-subset-v1/guidance-1/-4Sl9mPOlfA_0-3-rgb_front`), each rendered from its
R2 key with its SHA-256 checked; the records are in
`Sign Translator Stage Documentation/evidence/avatar-retarget-2026-10-22/`. Rendering the
same state twice gave a byte-identical video:

| clip | frames | upper arm n / median / p95 / max (°) | forearm | fingers (non-thumb) | thumb | head |
|---|---|---|---|---|---|---|
| 10006709 | 281 | 558 / 0 / 1.2e-6 / 1.2e-6 | 558 / 0 / 8.5e-7 / 1.2e-6 | 2052 / 0 / 1.2e-6 / 1.5e-6 | 513 / 0 / 1.2e-6 / 1.2e-6 | 267 / 0 / 0 / 1.2e-6 |
| 10036884 | 105 | 210 / 0 / 1.2e-6 / 1.7e-6 | 210 / 0 / 8.5e-7 / 1.2e-6 | 1428 / 0 / 8.5e-7 / 1.5e-6 | 357 / 0 / 1.2e-6 / 1.2e-6 | 105 / 0 / 0 / 3.4e-6 |
| 10075867 | 230 | 460 / 0 / 1.2e-6 / 1.2e-6 | 460 / 0 / 8.5e-7 / 1.2e-6 | 2580 / 0 / 1.2e-6 / 1.5e-6 | 645 / 0 / 1.2e-6 / 1.5e-6 | 230 / 0 / 0 / 2.4e-6 |
| generated:0 | 206 | 412 / 0 / 1.2e-6 / 1.5e-6 | 412 / 0 / 8.5e-7 / 1.2e-6 | 4944 / 0 / 1.2e-6 / 1.7e-6 | 1236 / 0 / 1.2e-6 / 1.5e-6 | 206 / 0 / 0 / 2.4e-6 |

The residuals are float64 `arccos` resolution: directions are reproduced exactly by
construction, so these numbers verify the implementation (the rig's own FK, the
local-rotation conversion, the twist choices and fills), not a fit. Before the twist was
taken about the whole shoulder→elbow segment it was taken about `upperarm01`'s own axis,
and because `upperarm01`/`upperarm02` are not collinear the upper-arm error was a real
0.2–2.1° median (p95 ≤ 3.2°); the measurement caught it. The torso is deliberately
distributed over five spine bones; its hips→neck direction differs from the canonical
chest-up direction by a median 1.4–2.1° (informational, not an acceptance group).

**Left/right.** (1) Structural: every `.L` bone is at x ≥ 0 and `.R` at x ≤ 0 at rest. (2)
Per frame: the wrist's side of the chest (camera x) agrees between skeleton and rig in
100% of scored frames on all four clips (frames within 5 cm of the midline excluded).
(3) Rendered: a synthetic state built by the project's own adapter from the rig's rest
landmarks with the left arm swung straight up is rendered through the same pipeline;
all foreground above the head (126 px at 128×128) lies on the viewer's right. Tests
also check that the raised right arm is the exact mirror image and that a left blink
closes the eye on the viewer's right.

**Limitations and findings.**

* Only directions are copied. The rig keeps the default body's proportions, so
  contacts (hand to chin, hand to hand) are not preserved, and monocular depth errors
  can put a hand inside the torso (ASLLRP 10036884 frame 62: the canonical left hand is
  3 cm behind the chest plane at the midline). There is no collision handling.
* Hand coverage in the MediaPipe ASLLRP states is low: 10006709 has the right hand
  valid in 76 of 281 frames (longest gap 6.0 s), 10075867 in 40 of 230. Those frames
  are filled (slerp or ease to rest), counted, and excluded from every statistic.
* `head_rotation` is a camera-frame MediaPipe face-geometry rotation; on ASLLRP it is a
  median 14–20° from frontal (the head appears tilted back), which is rendered as is.
* `root_translation`, `head_translation` and `contact` are not rendered (the avatar's
  root is fixed in front of a fixed camera).
* Found and fixed here: `canonical_skeleton.skeleton_positions` placed the *right* hand's
  knuckle bases mirrored (`sign * x` with `sign = −1`, while the hand frame's x runs
  pinky→index MCP on both hands). Independent check against MediaPipe *pose* landmarks
  (BlazePose index/pinky knuckles 19/17 and 20/18, stored per clip next to the states), expressed
  in each clip's Hand joint frame over six ASLLRP clips: the index knuckle lies on +x in
  88–99% of right-hand frames and 65–97% of left-hand frames, whereas the skeleton put the
  right index base on −x in 100% of frames. Bone directions were never affected; knuckle
  bases and therefore fingertip positions of the right hand were (used by
  `grammar_motion/frames.py`, `render_text_to_motion_samples.py`, `build_human_eval_packet.py`).
* Workbench inspection render; not a qualified avatar appearance, and no claim of
  signing fidelity. `accepted_provenance` stays false.


## Avatar v2 plan (pre-designed 2026-10-26 02:40 UTC; starts once a generator passes its exit gate)

**Goal:** a production-quality avatar that keeps every verified property of the current pipeline (retarget errors, left/right checks, timestamps, determinism) and fixes two things the owner saw: the flat Workbench look and the shaking.

**Principle:** accuracy is a hard constraint and aesthetics are optimised inside it. Every change below is measured, and nothing makes it into a release render unless the measures hold.

### 1. Find where the shaking comes from before fixing it

Three test sets go through the same pipeline:
- 5 generated clips;
- 5 *real* How2Sign clips (the control: if real motion shakes too, the cause is the pipeline, not the model);
- 5 real clips smoothed by 0.1 s.

A *shake score* (≥ 6 Hz RMS) is measured at each stage:
- (a) the canonical state (R7's `high_frequency_rms`);
- (b) the retargeted bone rotations;
- (c) the skinned vertex positions of the hands, head and chest;
- (d) screen-space 2D positions of the same points.

The first stage where generated clips exceed the control is where the shaking starts.

Likely causes, each checked:
- quaternion hemisphere flips between frames (q and −q), which show as one-frame spins;
- per-frame retarget or IK solutions jumping between equivalent poses;
- jitter in root and head translation (How2Sign's torso joints are jitter-dominated, round-3c diagnosis);
- the generator's own ≥ 6 Hz content (R7 bounds it at 1.5× real, which still shows on a rigid mesh).

### 2. Remove the shaking, accuracy-preserving

**Fixes, in order:**
1. **Continuity:** align each frame's quaternion to the previous one's hemisphere before any interpolation (`pacing.py` SLERP).
2. **Adaptive smoothing on SO(3):** a per-joint one-euro filter on the rotation manifold, with log map about the previous frame. Its cutoff rises with angular speed, so slow, still periods lose jitter while fast signing movements and holds keep their timing. Parameters are per bone group (torso strongest, fingers lightest), fitted on real clips to minimise shake score subject to the accuracy guard.
3. **Stable base:** the pelvis and lower body locked to a seated or standing pose, and root translation low-passed at about 1 Hz. Torso rotation is filtered more strongly than the arms, since How2Sign's torso motion is mostly noise.
4. **Render at 60 fps:** SLERP-resample from the generator's 24 fps. This adds no information, only temporal smoothness.

**Accuracy guard** (automatic, per clip):
- the filtered motion's per-bone angular error against unfiltered is p95 ≤ 3° (fingers ≤ 4°);
- fingertip and wrist positions stay within 1.5 cm at p95;
- the sign recogniser (R5's) and the 2D back-translator (R2b's), run on the filtered states, must not lose accuracy;
- any joint that breaks the guard gets a lighter filter.

### 3. Appearance

- **Engine:** EEVEE (fast) for previews and Cycles for release, with AgX colour, soft key, fill and rim lights, ambient occlusion, a neutral plain background and fixed seeds for determinism.
- **Character:** MakeHuman/MPFB2 assets under CC0 or CC-BY only (licence checked, recorded in the render's INDEX.json):
  - realistic skin and hair materials;
  - a solid dark, fitted top, so the hands contrast with the torso and nothing occludes the signing space;
  - no loose sleeves;
  - skin-tone hands with nails.
- **Hands:** high-resolution hand mesh with dual-quaternion skinning (`dqs.py`, no candy-wrapper wrists), plus corrective shape keys at knuckles and wrist.
- **Face:** the 52 ARKit blendshapes as shape keys on a face with brow, eyelid and mouth topology good enough to read non-manual markers. Brow raise and furrow are grammatical, so they must be clearly visible. Eye gaze and blinks are driven.
- **Camera:** a waist-up frontal view of the full signing space, plus an optional slight 3/4 view, orthographic or a long lens to avoid distortion.

### 4. Slower option

Playback at 0.8× and 1.0× via `pacing.py` time-stretch, so expression channels stay synchronised; the owner asked for slower motion.

### 5. Verification and storage

Each render's `verification.json` records:
- the existing retarget checks;
- the shake score at each stage, against the real-clip control;
- the accuracy guard;
- the recogniser and back-translator scores on the rendered states;
- the determinism re-render.

Every render goes to R2 under `renders/<round>-<date>/` with `INDEX.json`.

### 6. Rounds

| Round | Content | Pass |
|---|---|---|
| A | diagnosis | stage reports |
| B | continuity, filters and stable base | shake score ≤ the real-clip control |
| C | EEVEE appearance and character assets | 10-clip preview set for the owner |
| D | Cycles release renders on held-out test sentences | the human reviewers' packet |

Each round is committed with its numbers.
