# Data-quality decision for continuous training material (2026-10-18)

**Question (owner):** does the replacement for request-only continuous 3D data sacrifice
model quality, accuracy or translation correlation? If so, research better options
before implementation and training begin.

**Answer: yes, the first replacement loses quality, and training is blocked until a
better pipeline is implemented and measured.**

## 1. Evidence that the MediaPipe-only replacement loses quality

* A 2026 controlled study of pose estimators for sign language translation
  (arXiv 2604.24609; Phoenix and Signsuisse) ranks MediaPipe Holistic **6th of 8** on
  downstream translation: BLEU 10.33 against 11.68 (SDPose) and 11.53 (Sapiens), i.e.
  about 13% lower; MediaPipe drops one hand in 20–24% of signing frames and handles
  15 occlusion cases correctly in 73% against Sapiens' 100%.
* Monocular depth is ambiguous: every 3D value is a learned guess along the camera
  ray, worst for fingers and self-occluded hands; our pilot detected the left/right
  hand in only 70%/58% of frames.
* Current monocular hand models (WiLoR, HaMeR, GeoHand) reach 5–6 mm PA-MPJPE on
  FreiHAND, far ahead of MediaPipe's hand landmarks, but need the MANO model licence
  (a registration only the owner can do).

## 2. Better options found

| Option | Gain | Cost / condition |
|---|---|---|
| **Two-view triangulation of How2Sign** (frontal + lateral HD cameras, both 1280×720 H.264 at 24 fps in the same studio, measured on the acquired clips) | Removes monocular depth ambiguity: 3D from geometry, not a guess | Calibration is not released → self-calibrate per recording session from the signer's own keypoints; synchronisation not documented → measure it |
| **Sapiens 2D keypoints** (Meta, 308 points incl. dense hands and face; CC BY-NC 4.0) | Best occlusion robustness and near-best translation BLEU in the 2026 study | GPU (Runpod); non-commercial, like How2Sign |
| SDPose (MIT) | Best BLEU in the study | GPU, slow |
| WiLoR / HaMeR hands | Strong monocular fingers | MANO licence (owner registration) |
| SignAvatars SMPL-X fits | Large existing annotation | Owner request form |

## 3. Decision

Primary pipeline: **Sapiens 2D on both How2Sign views → per-session two-view
self-calibration → triangulated 3D → exact swing IK** (rotations are scale-invariant,
so the unidentifiable metric scale does not affect them). MediaPipe stays only for
the 52 face blendshapes (frontal ROI; 100% detection in the pilot), the one output
no alternative provides. Monocular estimates remain a labelled fallback for frames
the side view cannot see.

## 4. Measured against ground truth (CMU Panoptic, 2026-10-18)

Before How2Sign arrives, the pipeline was run on CMU Panoptic sequence
`171204_pose1_sample` (non-commercial research data), whose 3D body and hands are
reconstructed by the dome's many synchronised cameras. Two HD views shaped like
How2Sign's (36° and 79° from the subject's facing) were processed exactly as
How2Sign will be (`scripts/validate_two_view_on_panoptic.py`;
`evidence/panoptic-validation-2026-10-18/report.json`; 101 frames):

| Metric (lower is better) | MediaPipe monocular | Sapiens two-view, self-calibrated | Sapiens two-view, true calibration |
|---|---:|---:|---:|
| Arm bone direction error, median | 13.7° | **2.95°** | 2.81° |
| Finger bone direction error, median (L/R) | 14.9° / 19.3° | **8.6° / 6.2°** | 8.6° / 6.1° |
| Arm PA-MPJPE | 5.34 cm | **1.37 cm** | 1.30 cm |
| Bone-length CV | 0.102 | **0.023** | 0.026 |
| Frames with a usable hand (L/R) | 99% / 100% | 98% / 100% | 98% / 100% |

The views were synchronised at the recovered lag 0 (r = 0.93); self-calibration converged
at 0.58/0.63 px median reprojection, passing the gate. One focal length was 13% high, yet
the triangulated directions match the true-calibration result within 0.15°: the
rotations the model trains on are insensitive to that error.

**What this establishes.** The two-view pipeline is 4–5× more accurate than the monocular
fallback on arms and 2–3× on fingers, and is within 3° of the multi-camera reference on
arm directions. It is **not** equal to optical motion capture: fingers still carry 6–9°
of median direction error, gaze is not recovered, and metric scale is unidentified. It
is the most accurate continuous 3D the project can obtain without a request.

## 5. Gate before any training run

Training starts only when all of the following hold on real How2Sign front/side pairs
(acquired from the verified mirror, see CONTINUOUS_TRAINING.md):

1. **Synchronisation:** the per-clip view offset is measured by motion
   cross-correlation, is consistent within a session, and pairs failing the declared
   correlation threshold are excluded.
2. **Calibration:** median reprojection error after bundle adjustment ≤ 2 px per view
   (declared before the run), per session.
3. **Quality:** triangulated 3D beats the monocular estimate on bone-length constancy
   (coefficient of variation over time) and left/right limb symmetry, on the same
   clips.

These criteria were fixed before the Panoptic run and were not changed after it.
If a criterion fails, the result is reported and the next option in §2 is evaluated
before training — no training on a pipeline whose quality is unmeasured.

## 6. Error minimisation, measured (CMU Panoptic, 2026-10-19)

Goal: drive arm-direction, finger and joint-position error towards zero. Every change
was measured with `scripts/panoptic_accuracy_experiments.py` against the dome's
multi-camera reconstruction, tuned on `171026_pose3` (cameras 00_13/00_01, 150
frames) and checked on the held-out `170915_office1` (00_22/00_23, 150 frames) and on
the §4 sequence `171204_pose1_sample` (00_21/00_07, 97 frames). Video and ground
truth were verified to share frame numbering (offset 0 minimises the true-calibration
error on every sequence). Evidence: `evidence/panoptic-accuracy-2026-10-19/`.

**Adopted pipeline** (flip test-time augmentation → self-calibration on rigid-bone
joints with one division-model radial coefficient per camera → triangulation → rigid,
temporally smooth two-view refit of each hand with a hinge prior, λ = 20, μ = 20):

| Sequence | Method | Arm direction, median | Fingers L/R, median | Arm PA-MPJPE |
|---|---|---:|---:|---:|
| 171026_pose3 (tuning) | MediaPipe monocular | 15.3° | — | 5.87 cm |
| | §4 pipeline (all keypoints in calibration) | 14.6° (calibration failed) | 25.9° / 20.1° | 18.1 cm |
| | **adopted** | **3.23°** | **6.89° / 9.60°** | **2.21 cm** |
| | adopted keypoints, dome calibration | 3.53° | 5.65° / 8.23° | 1.03 cm |
| 170915_office1 (held out) | MediaPipe monocular | 10.6° | — | 5.60 cm |
| | **adopted** | **4.32°** | **7.04° / 11.77°** | **3.26 cm** |
| | adopted keypoints, dome calibration | 3.92° | 7.29° / 11.61° | 1.64 cm |
| 171204_pose1_sample (§4) | MediaPipe monocular | 14.0° | — | 5.29 cm |
| | §4 result | 2.95° | 8.6° / 6.2° | 1.37 cm |
| | **adopted** | **2.91°** | **5.56° / 4.29°** | **1.22 cm** |

What each step contributed, and what was rejected:

1. **Calibrate on rigid-bone joints only (plus the nose).** Under the dome's own
   calibration, Sapiens face-contour and other non-bone keypoints have a 6.2 px median
   two-view error (63% above 3 px) against 1.2 px for arm and hand joints, and they
   outnumber them four to one. Including them drove the self-calibration to a wrong
   solution on `171026_pose3` (focal 1 168 420 px; arm error 14.6°). Restricted to
   bone joints it converges (0.6/0.9 px median reprojection) on all three sequences.
2. **Flip test-time augmentation** (mirror the crop, swap left/right heatmap channels,
   average). With the dome calibration it improves fingers on all three sequences and
   arms on two, the third being equal within 0.04° (held-out: 4.37° → 3.92° arm,
   9.43°/17.31° → 8.00°/15.48° fingers).
3. **Hand refit** (`skeleton_fit.py`, `two_view_states.refine_hands`): each hand is a
   wrist-rooted chain of fixed (session-median) bone lengths whose directions minimise
   confidence-weighted reprojection in both views plus a second-difference penalty;
   the fitted fingers are re-attached to the triangulated wrist. Fingers improve on
   every sequence and calibration (by 0.9–2.4° median). Fitting the arms as well was
   rejected: it cost 0.3–1.5° of arm accuracy.
   **Hinge prior.** Interphalangeal joints are 1-DOF hinges, so a finger's proximal,
   middle and distal phalanges are coplanar; the fit adds μ·det[u₁, u₂, u₃] per finger
   and frame (zero exactly when coplanar). Fingers improved monotonically up to
   μ = 20 on all three sequences (held-out right hand 12.19° → 11.59°), plateau to
   μ = 100 on the tuning sequence and degrade at 300; μ = 20 is used.
4. **Radial distortion in the bundle adjustment.** Panoptic HD lenses have k1 ≈ −0.26;
   modelled (residuals in observed pixels, |λ| ≤ 1, focal bounds) arm direction and
   PA-MPJPE were equal or better on all three sequences with flip keypoints
   (PA-MPJPE 2.39 → 2.21, 3.33 → 3.26, 1.31 → 1.22 cm), fingers within ±0.2°. The coefficient itself is weakly identified; only its
   effect on the triangulation is relied on.
5. **Confidence-weighted triangulation — rejected.** Mixed; with the dome calibration
   it cost 0.6° arm and 0.8 cm PA-MPJPE on the tuning sequence.
6. **Temporal smoothing of the triangulated arms — not adopted.** A Gaussian of
   σ = 1 frame gained 0.1–0.3° arm direction under self-calibration on all three
   sequences but was mixed under the dome calibration (−0.18 to +0.10°); at Panoptic's 30 fps — and more so at How2Sign's 24 fps — it
   begins to attenuate the fastest arm movements, which the motion model must learn,
   so the gain does not justify the bandwidth loss.
7. **High-resolution hand crops — rejected.** The whole-body model is less confident
   on hand-only crops (0.93 → 0.1–0.76) and its crop positions made fingers worse
   (7.09° → 8.38° with the dome calibration).

**Session-scale calibration.** The rows above calibrate on the 150 evaluated frames
only. Production calibrates each How2Sign session on up to 3 000 frames from all its
clips. Measured the same way on the held-out sequence (calibration on the 150 frames
plus 300 frames spread over the whole recording, evaluation on the same 150 frames):

| 170915_office1 | Arm direction | Fingers L/R | Arm PA-MPJPE |
|---|---:|---:|---:|
| adopted, 150-frame calibration | 4.32° | 7.04° / 11.77° | 3.26 cm |
| **adopted, session-scale calibration (450 frames)** | **3.55°** | **7.10° / 11.59°** | **1.85 cm** |
| adopted keypoints, dome calibration | 3.92° | 7.29° / 11.61° | 1.64 cm |

With session-scale data the self-calibration is within 0.2 cm and 0.4° of the dome's own
calibration (focal lengths 1666/1516 px vs 1514/1572 px). Its radial coefficient came
out with the opposite sign to the lens's (−0.17/−0.14 vs ≈ +0.25): it is not
identified and acts only as a small regulariser.

**Where the remaining error lives, and the floor.** With the dome calibration the
adopted keypoints reach 3.5–3.9° on arms and 1.0–1.6 cm PA-MPJPE, and session-scale
self-calibration comes within 0.2 cm of that, so the remaining error is the 2D
keypoints themselves. Panoptic's reference itself is not exact: its arm
bone lengths vary by 3.7% (CV) over time and its hand reconstruction reprojects at
2.1 px, so errors below ≈2–3° cannot be resolved against it. "Infinitely close to 0"
is not measurable with any available reference; a mocap-grade check
(InterHand2.6M-class hand data, or an optical-mocap body set) is the next step if the
owner can obtain one.

## 7. First real How2Sign session (pilot, 2026-10-19)

The two-view pipeline was run end to end on one validation session
(`a4uz0W33REs-3`, 11 sentences, 832 frames per view, 1280×720 H.264 at 24 fps) from
the verified zips: MediaPipe states for the frontal view, Sapiens-1B with flip TTA on
both views, then synchronisation, calibration, triangulation and the §5 gate. Outputs
are in R2 under `acquisition/2026-10-17-v1/derived/pilot-val-a4uz0W33REs-3/`. Findings,
in the order they were found:

1. **Velocity-correlation synchronisation failed.** Only 1 of 11 clips reached the
   declared r ≥ 0.8, so the gate passed on a single clip and a calibration from it.
   Yet 7 of the 11 clips agreed on a lag of 8–9 frames.
2. **Synchronisation is now epipolar and per session.** For each candidate lag the
   clips' correspondences are pooled, a fundamental matrix is fitted robustly, and the
   median Sampson distance is scored; misaligned frames violate the epipolar
   constraint, so the true lag minimises it. On the pilot the minimum is at 8 frames,
   with a sharpness ratio of 0.54 (the declared requirement is < 0.8 against every lag
   ≥ 2 frames away), and all 11 clips pass. Within the longest clip the best sub-frame
   lag stays at 8 ± 0.25 frames throughout, so there is no clock drift and integer
   lags suffice.
3. **Keypoint consistency differs from Panoptic.** With the epipolar geometry fitted on
   body joints, the median Sampson distances are:
   - nose 0.75 px;
   - left shoulder 1.8 px, elbows 2.2–2.9 px;
   - left wrist 2.0 px, right wrist 5.0 px;
   - **hand keypoints 4.8 px** (Panoptic: 1.2 px under its own calibration).

   The right shoulder is never confidently seen by the lateral camera, and the leg
   keypoints fall outside the frame (y up to 765 on a 720-px image). Consequences:
   - keypoints outside the image are never observations;
   - the calibration's robust start is fitted on nose, shoulders, elbows and wrists;
   - timing uses body plus hands, because hands carry the fast motion. Body alone gives
     a flat curve (ratio 0.88); body plus hands gives a sharp one (0.54).

   Left/right label swaps and frame-rate jitter were tested and excluded: swapping
   labels raises the error to 18–57 px, and the error does not grow with joint speed.
4. **One short session does not identify the focal lengths.** Across reasonable
   settings the single-session self-calibration of the pilot lands anywhere from
   548 to 2 889 px, and the radial coefficients run to their bounds. Two consequences:
   - **Lens distortion is off by default.** On undistorted synthetic sessions the joint
     fit invents λ ≈ −0.27 and focal lengths 7% low. It helped on Panoptic only because
     those lenses are known to be strongly distorted.
   - **Focal lengths are now shared across sessions of the same studio cameras**
     (`calibrate_how2sign_cameras.py`), with a pose per session. On four short, noisy
     synthetic sessions the focal error falls from up to 12% alone to 0.1%/1.7% jointly.
5. **The gate refuses the pilot session, as it should.** Calibrated alone, its
   triangulated arm bone-length CV is 0.039–0.044 against the monocular 0.037, and its
   reprojection reaches 2.37 px. No threshold was changed to admit it.

6. **A stricter triangulation confidence threshold was measured on Panoptic and
   rejected.** Calibration was held at the declared 0.3 and only the threshold for
   triangulation was raised.
   - On `170915_office1`, raising it to 0.7 appears to cut arm error from 3.92° to
     1.18°. But at that threshold the left shoulder and right elbow are never
     triangulated (coverage 0%), so the error is measured on the easy bones that remain.
   - Where coverage stays near full (`171026_pose3`, `171204_pose1_sample`), the gain
     is ≤ 0.1° or none.
   - A missing joint is a missing rotation downstream, so the declared 0.3 stands.

7. **Three validation sessions with shared intrinsics.** Two more sessions were
   processed (`_2FBDaOPYig-3`, 12 clips; `eY32ru3Nstc-8`, 20 clips; Sapiens without flip
   TTA, diagnostic only). Epipolar sync finds a sharp lag in each (8, 2 and −2 frames)
   and admits every clip.

   For each session, the focal lengths were fitted jointly on the *other two* (a
   held-out test, as production uses intrinsics from other sessions), and only the
   session's pose was fitted. Arm bone-length CV, triangulated against monocular:

   | Session | Calibrated alone | Held-out shared intrinsics | Monocular | Gate (held-out) |
   |---|---:|---:|---:|---|
   | `a4uz0W33REs-3` | 0.0439 (f 807/2470) | 0.0415 (f 1021/1174) | 0.0366 | fails |
   | `_2FBDaOPYig-3` | 0.0455 (f 964/1103) | 0.0395 (f 1103/1368) | 0.0411 | passes |
   | `eY32ru3Nstc-8` | 0.0365 (f 1012/1339) | 0.0366 (f 1019/1392) | 0.0414 | passes |

   - Held-out focal lengths agree within ±4% (frontal) and ±8% (lateral), consistent
     with fixed studio cameras.
   - Shared intrinsics improve CV on two sessions and leave the third unchanged; the
     joint fit over all three gives 1051/1279 px.
   - Two of three sessions pass the declared gate.
   - Caution on the quality criterion: Panoptic's own multi-camera reference has an arm
     CV of 0.037, and MediaPipe's world landmarks carry a learned body prior. So
     "triangulated CV below MediaPipe's" is a strict test, not a measure of accuracy.
     It was kept unchanged.

   The full run calibrates intrinsics over 40 sessions (`calibrate_how2sign_cameras.py`)
   and reports every session's gate; training uses only passing sessions.
