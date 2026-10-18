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
| **Two-view triangulation of How2Sign** (frontal + lateral HD cameras, both 1280×720 at 30 fps in the same studio) | Removes monocular depth ambiguity: 3D from geometry, not a guess | Calibration is not released → self-calibrate per recording session from the signer's own keypoints; synchronisation not documented → measure it |
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

## 4. Gate before any training run

Training starts only when all of the following hold on real How2Sign front/side pairs
(they arrive when Google Drive's quota clears):

1. **Synchronisation:** the per-clip view offset is measured by motion
   cross-correlation, is consistent within a session, and pairs failing the declared
   correlation threshold are excluded.
2. **Calibration:** median reprojection error after bundle adjustment ≤ 2 px per view
   (declared before the run), per session.
3. **Quality:** triangulated 3D beats the monocular estimate on bone-length constancy
   (coefficient of variation over time) and left/right limb symmetry, on the same
   clips.

If a criterion fails, the result is reported and the next option in §2 is evaluated
before training — no training on a pipeline whose quality is unmeasured.
