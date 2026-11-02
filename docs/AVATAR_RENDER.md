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
| 8j | render fixes before raters: filter, collision correction, frame, lights, CC0 appearance (§13) | done |

Rendering layer: 69 tests, green on two consecutive runs at that stage; the
whole-project total recorded then was 1,176 tests. These are stage snapshots, not the current verdict; rerun the suite (see RESEARCH_CONTRACTS.md's reproducibility section).

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

Eyes, teeth and tongue are the audited attachments coloured from their pinned textures (the
transparent cornea shell is dropped). Brows, lashes, hair, the top and the skin texture are in §13.

**CLI.** `render_avatar.py --state S.npz [--sha256 H] | --key K --sha256 H [--store r2|dir:…]
--out-dir NEW --allow-inferred [--resolution 384] [--start-seconds/--max-seconds]` writes
`bundle.npz` (posed meshes, provenance), `frames/`, `avatar.mp4` (H.264 4:2:0, PTS from the
state's clock; `--lossless` for 4:4:4 CRF 0) and `verification.json`. Rendering is Blender in
`--background --factory-startup` with a fixed orthographic front camera (§13 for the current engine, frame and filter).

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
* No qualified avatar appearance, and no claim of signing fidelity. `accepted_provenance` stays false.


## 13. Render fixes before Deaf raters (2026-11-02)

Raters judge signing only through this avatar. Every number below is from `scripts/avatar_fixes.py` on 36 D20 sentences (12 each from c01, c03, c05, a fixed draw), the real two-view state and draw 1's composition; `reports/avatar_render_fixes.json` holds them. Medians over clips. None of this says whether anything is correct signing: only a Deaf signer can.

**The order of operations.** State → fill gaps → filter → retarget → move covered hands in front of the body → skin → render. The packet builder gives the renderer the state as it is, so nothing is held first. No step reads a clip's name, clock, system or `observed` flag. `render_settings_sha256` in `verification.json` hashes the render, filter and correction settings, the encoder and its libraries, the Blender version, every source file and every asset. Each item in the key holds that hash and the code commit, and a packet with mixed values stops before its key is written. Tests read the bundle Blender draws from.

### Shake

Measure: RMS distance from a trajectory's own 6 Hz low-pass, in millimetres and degrees, and the share of velocity power at 6 Hz and up. `motion_quality.py`. "Old" is the path before this work: the packet held gaps, then no filter. "Filled" interpolates the gaps with no filter, so it shows what filling did alone. "Final" adds the filter and the correction. Generated states have no gaps, so old and filled are the same there.

| Median | Real: old → filled → final | Generated: old → final |
|---|---|---|
| Head position, mm | 5.29 → 5.26 → 0.72 | 3.08 → 0.53 |
| Chest rotation, degrees | 0.75 → 0.67 → 0.10 | 0.36 → 0.06 |
| Left wrist, mm | 5.19 → 4.65 → 1.09 | 7.60 → 3.38 |
| Right wrist, mm | 7.23 → 6.63 → 1.91 | 5.72 → 2.22 |
| Left fingertips, mm | 5.72 → 5.01 → 1.39 | 13.3 → 7.23 |
| Right fingertips, mm | 7.90 → 7.43 → 2.60 | 11.6 → 6.46 |
| Left hand rotation, degrees | 1.89 → 1.46 → 0.64 | 5.12 → 2.73 |
| Right hand rotation, degrees | 3.17 → 2.88 → 1.27 | 5.53 → 3.09 |
| Share at 6 Hz and up: head | 0.68 → 0.68 → 0.12 | 0.23 → 0.02 |
| Share at 6 Hz and up: right wrist | 0.49 → 0.44 → 0.08 | 0.16 → 0.03 |

On the rendered pixels themselves (head and hair centroid, six clips at 768 px, old renderer against the filter and frame): real 2.8–4.5 px → 0.3–0.5 px, generated 0.6–2.3 px → 0.1–0.6 px. A hand passing through the top of the picture inflates this read on some clips.

**The prior hypothesis, tested.** How2Sign's real states do carry estimator jitter: the torso and head's motion is two-thirds above 6 Hz. The model doesn't simply copy it. Generated torsos carry half of real's. Generated hands carry about twice real's: hand frame 5.1–5.5° against 1.5–2.9° (real, gaps filled), fingertips 12–13 mm against 5–7 mm. Real clips played at 2× read 2.9–5.3° and 8–12 mm, about the same as generated. So the excess in the hands is consistent with real-like jitter squeezed in time by composition's fitted timing, which the motion diagnosis measured at about 2×. This doesn't prove the cause. On torso and head the owner's complaint is mostly the estimator's and is now gone from both; on hands the generated clips still shake more than real, after the filter too.

**The filter** (`motion_filter.FILTER_V1`, one table, no argument changes it). Savitzky-Golay, zero-phase. Torso (pelvis, chest): window 9, order 2 (-3 dB at 2.9 Hz). Everything else, including fingers, face, blink and gaze: window 5, order 2 (-3 dB at 5.7 Hz; gain 0.97 at 3 Hz, 0.91 at 4 Hz, 0.66 at 6 Hz, 0.23 at 8 Hz, 0.19 at 10 Hz, and 0.37 at 12 Hz, where its sidelobe lets some frame-to-frame alternation through). Arms and hand frames are filtered as world rotations, because a local arm rotation holds the chest's noise inverted, and filtering chest and arm differently in local frames would push that noise into the wrist.

**Why these.** `reports/avatar_fixes_sweep.json` holds the candidates. One-Euro both ways lost more articulation for less shake on arms: at min cutoff 3 Hz, beta 3, the right wrist stayed at 5.8 mm with its p95 speed at 0.95, where Savitzky-Golay window 5 reached 5.4 mm at 0.99. On fingers, the same One-Euro took the 3 Hz band to 0.96 and the 6 Hz band to 0.93; window 5 kept 1.00 and 0.98. On a synthetic handshape change at 7 changes a second, window 5 keeps 0.94 of the plateau distance and window 9 order 3 keeps 0.51, which is what "a filter at the wrong frequency changes the sign" looks like (a test holds this). Fingers therefore get the lightest filter, and much of the finger wobble between handshapes stays: the quietest tenth of frames only goes from 0.77 to 0.75 rad/s on real. Video-path hands don't rest, and this filter does not make them.

**Articulation kept** (after the filter / before it, same clip, gaps filled on both sides; median, with the 10th percentile in brackets). This is 3D motion on the retarget before the hand correction. The correction moves wrists in depth, so a 3D wrist speed after it isn't what the front view shows.

| | real | generated |
|---|---|---|
| Finger speed, raw (the part above 6 Hz is mostly what the filter is for) | 0.95 (0.94) | 0.83 (0.79) |
| Finger speed, 6 Hz band | 0.98 (0.97) | 0.97 (0.95) |
| Finger speed, 3 Hz band | 1.00 | 1.00 |
| Finger peak speed (p95), 6 Hz band | 0.98 (0.94) | 0.99 (0.91) |
| Finger bend range | 1.00 (0.99) | 1.00 (0.98) |
| Dominant wrist peak (p95), 6 Hz band | 0.95 (0.87) | 0.99 (0.95) |
| Dominant wrist maximum, 6 Hz band | 0.93 (0.81) | 0.97 (0.87) |
| Hold count | 3 → 3 | no hold detected at 0.15 m/s |
| Hold share of frames | 0.22 → 0.35 | 0.00 → 0.00 |
| Wrist jitter inside holds, mm | 3.05 → 0.88 | 1.05 → 0.77 |
| Blink peak; head speed in movements | 0.99; 0.98 | 0.98; 0.99 |

Holds lengthen because jitter was breaking them, not because motion was removed. Real's dominant wrist loses 5 to 7%; on a state already low-passed at 6 Hz it loses 4 to 5%, which is the filter's own bite on motion below 6 Hz, and the rest was noise. Some clips lose more (10th percentile 0.81 to 0.87). Lag is 0 frames on every clip.

**Gaps.** Real arms fail more than expected: over all 500 D20 clips, c01's arm chain is invalid on a median 37% of frames; 60 of its 177 clips are over half invalid and 12 over 80%, with 12 gaps longer than 2 s. c03 and c05 are near 0. The hip-based chest frame drops, so arms, chest and pelvis all go invalid while the fingers stay valid. The packet used to hold the last value and then step, which a rater would see as a frozen arm and a jump, only on real clips. Gaps are now interpolated geodesically, whatever their length. That removes the step. It can't restore the arm: a clip with 98% of frames invalid shows almost no arm motion, and interpolation reads as slow drift. **The reference is a weaker upper bound on c01 than the eligibility rule suggests,** since that rule reads hand coverage only. The key holds each sentence's share, and the scorer reports the reference with and without clips over 50% and over 20%, report only (protocol section 6).

### Hands covered by the body

Two reads. **Covered:** a hand vertex with drawn body in front of it in the front view, whether inside the body or behind it. The drawn body is the torso and head, the top without its sleeves, and the hair, as skinned, in 1 cm cells per frame (`collision.front_maps`). **Inside:** a hand or forearm vertex inside the rest body's signed distance grids. Posed torso and head surface vertices read 1.9 mm median, 6.2 mm p95 and 30 mm at worst in those grids, so that's the second read's own error. After the correction an exact ray test against the drawn triangles, on 102,528 sampled hand vertices, found none behind them, at a clearance of 3 mm, so the clearance stays and the top is cleared.

| Before → after, share of frames | real | generated |
|---|---|---|
| Hand covered: any vertex | 41.9% → 0% | 22.1% → 0% |
| Hand covered: half a hand or more | 3.3% → 0% | 6.7% → 0% |
| Hand covered: all of it | 1.0% → 0% | 2.8% → 0% |
| Covered hand's deepest vertex behind the surface | 177 mm → 0 | 299 mm → 0 |
| Hand inside the body: any depth | 31.3% → 0% | 17.6% → 0% |
| Hand inside the body: over 10 mm | 23.0% → 0% | 14.5% → 0% |
| Hand inside the head: any depth | 2.2% → 0% | 0.1% → 0% |
| **Forearm inside the torso: any depth (not corrected)** | 59.4% → 42.9% | 78.8% → 77.5% |
| Forearm inside the torso: over 10 mm | 48.5% → 29.2% | 70.5% → 69.3% |

**Correction** (`collision.py`). Location is linguistic, so the hand moves only along the camera axis: the front view draws the wrist in exactly the same place (largest shift over all clips and sides: 2.3e-8 cm), the hand ends 3 mm in front of what covered it, and the hand's world rotation and finger locals are unchanged, so handshape and palm orientation don't move. A hand behind the torso and one inside it are treated alike, so nothing jumps between them. An arm too straight to reach keeps the hand's image place and stays partly covered; none did on these clips, and each clip's record counts it.

**The forearm decision.** An earlier version moved the wrist and swung the elbow to clear the forearm too, and that changed the arm raters see. These are the settings compared, on the same 72 clips, and on sheets in R2 (`cmp-*`). Elbow and forearm columns read each clip and side's 95th percentile, against the pose before correction, at 768 px; the first figure is the median over clips and sides, then its 90th percentile.

| Real | Elbow shift, px | Frames over 10 px | Forearm image direction, ° | Forearm image length kept | Wrist depth, clip max | Forearm in torso |
|---|---|---|---|---|---|---|
| Forearm drives, swivel to 70° | 39, 105 | 62% | 5, 49 | 0.73, 0.46 | 13.9 cm | 1.4% |
| Forearm drives, swivel to 20° | 39, 105 | 62% | 5, 49 | 0.73, 0.46 | 13.9 cm | 3.3% |
| Forearm drives, no swivel | 41, 105 | 60% | 4, 49 | 0.72, 0.46 | 13.9 cm | 22.2% |
| Hand only, elbow stays in the arm's plane | 12, 24 | 18% | 2, 5 | 0.94, 0.83 | 5.0 cm | 48.7% |
| **Hand only, elbow nearest where it was drawn (shipped)** | 7, 14 | 6% | 1, 3 | 0.97, 0.93 | 5.1 cm | 42.9% |

| Generated | Elbow shift, px | Frames over 10 px | Forearm image direction, ° | Forearm image length kept | Wrist depth, clip max | Forearm in torso |
|---|---|---|---|---|---|---|
| Forearm drives, swivel to 70° | 58, 76 | 60% | 22, 35 | 0.86, 0.74 | 13.2 cm | 0.9% |
| Forearm drives, swivel to 20° | 49, 69 | 60% | 19, 33 | 0.86, 0.74 | 13.2 cm | 15.6% |
| Forearm drives, no swivel | 39, 70 | 31% | 8, 23 | 0.82, 0.69 | 13.2 cm | 76.7% |
| Hand only, elbow stays in the arm's plane | 9, 34 | 14% | 1, 8 | 0.95, 0.85 | 6.8 cm | 77.7% |
| **Hand only, elbow nearest where it was drawn (shipped)** | 5, 23 | 9% | 1, 7 | 0.98, 0.92 | 6.9 cm | 77.5% |

**The trade, plainly.** Clearing the forearm moved the elbow across the picture by about 40–60 px on a typical clip, up to 165, and shortened or turned the forearm in the image, for a sleeve that goes into a top of the same dark colour. On the sheets, forearms entering the top read as a dark arm against a dark body, with a faint line of shade; the swung elbows read as a different arm. Signs that use the forearm or elbow as a location would be exposed to that swing, and nothing would tell the rater. So the shipped version keeps the state's arm: **forearms stay inside the torso in 43% of real frames and 78% of generated ones.** Elbows still move, because a fixed-length arm can't change its wrist's depth without moving something: over 30 px in 0.1% of real frames and 0.8% of generated, 54 and 96 px at worst. On the worst clips the forearm's direction in the image changes 19° on real and 35° on generated, its image length falls to 0.73 and 0.43, and a generated forearm's 3D direction changes up to 78°.

Acceptance in `verification.json` now reads the arm as drawn, after this correction. **Real clips can fail it where generated ones don't:** of 36 clips at the 5° median limit, 4 real clips fail on the upper arm (18° at worst) and 1 on the forearm (7°), and no generated clip fails. A real signer's hands sit on the body for long stretches, and the rig's torso is thicker than the estimate, so those arms are moved on most frames. `avatar_acceptance` reports this and doesn't refuse it.

For the record, the version merged before this work left part of a hand behind the drawn body in 6.2% of real frames and 4.0% of generated ones, 32 and 68 mm deep, with no half-covered hand: it read only the rest body, not the top, and moved a hand only once it was inside. Its elbows moved a median 23 px on real and 52 px on generated.

Not handled: hand against hand, a hand against the other arm or its own upper arm, and a hand behind its own sleeve.

### Readability

| | before | after |
|---|---|---|
| Frame | fitted to each clip: span 7.9 on real, 8.8–9.25 on generated (source units), so generated hands were drawn about 12% smaller | one frame for every clip, span 9.0, hair to hip |
| Engine, 768 px | Workbench, no shadows | EEVEE, 32 samples; key, fill and rim lights; soft shadows |
| Background | grey, skin against it at luminance contrast 1.14 | darker blue-grey, skin against it 2.6, the top against it 2.0 |
| Skin against the top | 3.3 | 5.2 |
| Determinism | re-render identical | two separate runs give a byte-identical video; a fresh process's first frame is identical |

**Hands leaving the frame.** Real: partly out 0.0%. Generated: partly out 16.6%, fully out 0.2%, nearly all a hand hanging below the hip; 1.1% are above the hip. Span 8.4 would draw hands 7% larger but lose a generated hand entirely in 10% of frames; span 10.2 would keep it but draw every hand 12% smaller.

**Resolution**, one 104-frame generated clip:

| Resolution | Finger width | Video | Whole render |
|---|---|---|---|
| 384 | 6.8 px | 168 KB | 25 s |
| 512 | 9.1 px | 253 KB | 25 s |
| 768 | 13.7 px | 460 KB | 26 s |
| 1024 | 18.2 px | 710 KB | 34 s |

768 stays. By eye, at 384 finger edges are blocks of a few pixels, at 768 they read cleanly, and 1024 adds little. Render time is mostly not pixels. It was 7.9 s for the same clip before. Uploading each render with its frames takes longer than rendering it: 65 s for a 236-frame clip, measured. So a pilot of 240 clips at a mean 170 frames takes about 2.5 hours of rendering and about 3 of upload on the Mac, about 6 hours in all, and about 40 GB in R2.

**Playback speed isn't changed.** The owner asked for slower motion. The release bar is read at predicted timing, and a slower video would change that condition, so this is reported, not done. Generated wrists move 1.45× real's on the right and 3.5× on the left at p95 in these clips, and that speed is still visible after the filter. A slow-motion control in the rater's player, used equally on every clip, is the place to ask about it; it isn't built.

### Appearance

All from MakeHuman's system asset pack, listed CC0 on the pack's page, each text asset stating its CC0 release in its own header; `source-intake/makehuman-system-assets/intake.json` holds the pack's URL and hash and each file's hash, and `load_wardrobe` checks them. Hair `short01` (leaves the forehead and brows clear; `short03` covers an eye, refused), brows `eyebrow002` in one mid-brown colour (a bolder pair, so raise and furrow read at 768), lashes `eyelashes01`, top `male_casualsuit02` in one plain dark colour with its printed texture unused, skin `young_lightskinned_male_diffuse`. No SMPL-X, no MANO. The eyes, teeth and tongue stay the audited ones. It is a plain figure; whether it reads as a person a signer is comfortable watching is for raters to say. The texture's near-black brows read as a stern line, so the shape stays and the colour became a mid brown: on six expressive frames, a lowered brow still shows plainly and the face looks less severe, though a state's real lowering still looks like lowering. A thinner cut (alpha threshold 0.5) lost the brows' ends and was refused.

### Blinding

The video has no sample id, clock id or system name in it (a test renders one motion under two names and two clock origins and gets byte-identical files). The packet copies only the video. Frame, lights, engine and encoder settings don't vary.

**File size: padded.** Generated clips encode larger because more moves: on six clips, about 4.4–5.1 KB per frame against 3.7–4.4 for real. A file's size would mark the arm for anyone sent the files, and the owner's flow may send them. So the packet appends an MP4 `free` box to each video, up to the next multiple of 8 MiB. It's a standard box players skip: the stream and every frame are unchanged, and it decoded in ffmpeg and macOS QuickLook. 240 clips take about 2 GB. A file's overall bit rate then follows its length alone, and length is the cue the protocol already records. A tool that reads the stream's own bit rate would still see the difference; raters aren't given tools, and playing the videos yourself stays the safest. The key keeps the encoded size as `video_bytes`, so `video_bytes_per_frame` still shows the difference to the team, and the render in R2 is unpadded.


## Avatar v2 plan (pre-designed 2026-10-26 02:30 UTC; starts once a generator passes its exit gate)

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
