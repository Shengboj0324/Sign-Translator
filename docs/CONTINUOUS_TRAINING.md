# Continuous co-observed data and English-to-sign training

This document covers the replacement for request-only continuous 3D sources, the
video-to-canonical adapter, the English-to-motion model, and how training runs on
Runpod. Operator commands are in [RUNPOD.md](RUNPOD.md).

## 0. Honest scope (read first)

* The continuous states here are **monocular estimates** from video, not motion
  capture: every value is `observed=False` with the exact model hashes as its
  inference method. They are co-observed in the strict sense that body, hands, head
  and face come from the same frames on one clock.
* How2Sign is **CC BY-NC 4.0**: research use only.
* The English-to-motion model learns from interpreted, translated sentences (How2
  instructional videos). It produces canonical motion, not reviewed ASL; quality
  claims need the reviewed glosses and the independent study, which need people.

## 1. Why a replacement, and what was researched

No continuous ASL 3D corpus could be acquired without a request (research pass,
2026-10-18):

| Source | Continuous ASL | 3D | Acquirable now |
|---|---|---|---|
| RIT/CUNY ASL Motion-Capture Corpus (98 stories, BVH/FBX, gloss keyframes) | yes | yes | by request (no reply); `bvh_motion.py` is ready |
| SignAvatars (SMPL-X fits incl. 34K How2Sign clips) | yes | yes (fitted) | request form |
| How2Sign Panoptic 3D subset | yes | yes | "under construction" on the official page |
| ASLLRP / NCSLGR | yes | no | account |
| SeRV (YouTube-ASL reconstructions) | yes | yes (fitted) | samples only |
| **How2Sign green-screen clips, all splits, front and side** | **yes** | **no (video)** | **public, direct** |
| OpenHands poses (WLASL, MS-ASL, ASLLVD), Zenodo CC BY 4.0 | no (isolated) | 2D | public, direct |

The replacement is therefore to **estimate** co-observed 3D from the public
continuous video the project may already use — the same route SignAvatars took —
with Apache-2.0 estimators, and to keep the request-only sources as upgrades.

Acquisition status (R2 `acquisition/2026-10-17-v1/`): How2Sign original and
realigned English translations for all splits are in R2. The large How2Sign zips
returned Google Drive's "quota exceeded" page; a watcher retries every two hours
and starts the verified transfer when Drive serves the files. Chunked range
requests were not used to get around the quota. OpenHands ASL pose archives are
acquired with the publisher's MD5 checked.

## 2. Video to canonical state (`data_engineering/video_landmarks.py`)

* **Estimators.** MediaPipe Tasks 0.10.21 with pinned model hashes: pose landmarker
  (heavy, 33 world landmarks in metres), hand landmarker (21 world landmarks per
  hand), face landmarker (52 blendshapes and the facial transformation matrix).
* **Regions of interest.** Face and hand landmarkers run on square crops centred on
  the pose's nose and wrists, sized by shoulder width, each in its own video-mode
  tracker, so they stay in range on full-body shots. On the Cokely HD video this
  raised face detection from 0% to 100% of frames.
* **Axes.** World landmarks are image-aligned (y down); they map to y-up by
  `(x, y, z) → (x, −y, −z)`, a proper rotation.
* **Rotations by swing IK** (`data_engineering/landmark_kinematics.py`). Root frames
  (pelvis from hips and spine, chest from shoulders and spine, each hand from
  index/pinky knuckles and wrist→middle knuckle) are full orthonormal frames. Every
  other joint takes the minimal rotation from its declared rest direction to the
  observed bone direction in its parent's frame,
  `R = I + [v]× + [v]×²/(1 + c)` with `v = r × u`, `c = r·u`. Forward kinematics then
  reproduces each observed bone direction exactly (`W_child r = b`, tested to
  1e-10, and invariant to a global rotation of all points). Twist about a bone is
  unobservable from two points and is zero by declaration; an antiparallel or
  degenerate configuration is invalid, never guessed.
* **Hands.** MediaPipe's handedness label assumes a mirrored image and is not used;
  each detection is assigned to the anatomical wrist by image distance.
* **Head and face.** Head rotation is the polar (nearest-rotation) factor of the
  face matrix; head translation is withheld in ROI mode because it refers to the
  crop's camera. The 52 blendshapes fill `face`; the two `eyeBlink` scores fill
  `blink`.
* **Declared unavailable.** Root translation (world landmarks are hip-centred),
  gaze (no calibrated mapping from landmarks to direction is established; the
  `eyeLook*` blendshapes remain in `face`), contact.
* **Validity.** Pose landmarks below visibility 0.5 invalidate their bones; frames
  without a detection are invalid for that channel.

Pilot on 10 s of the Cokely HD video (299 frames): pose and face in every frame,
left hand in 70% and right hand in 58%, body cells valid in 61% (legs below the
visibility threshold), 11.5 frames/s on this Mac's CPU. Full How2Sign extraction
(about 5 million frames) is therefore a Runpod CPU job (`scripts/extract_video_states.py`,
sharded and resumable).

## 3. Sentence-aligned manifest (`scripts/build_how2sign_text_manifest.py`)

Each converted clip is joined with its realigned English sentence and given its
partition from the **frozen signer-disjoint fold plan** (B11; fold 0: 29,012 train,
1,861 validation, 292 test rows). Clips outside the plan (official validation and
test splits) are labelled `heldout_official` and never used for training or model
selection. The manifest kind `continuous` admits every channel the estimator
produced; per-frame masks inside each state carry the rest.

## 4. English to motion (`generation/text_to_motion.py`)

* **Text.** NFC-normalised UTF-8 bytes with BOS/EOS — no downloaded tokenizer.
* **Model.** A transformer text encoder, a duration head (log seconds), and a
  non-autoregressive transformer decoder whose queries are sinusoidal features of
  the **true clock in seconds** and of the phase `t / duration`, cross-attending to
  the text. One head per canonical channel, initialised to the neutral value, so an
  untrained model emits the identity pose (tested).
* **Objective.** The canonical per-channel losses (`sin²(θ/2)` for rotations,
  Huber for coefficients, Bernoulli KL for bounded states) over supported cells
  only, with inferred targets declared, plus a Huber loss on log duration. The
  duration target is last − first frame time plus one median frame interval.
* **Generation.** Duration is predicted, frames are placed on a declared clock, each
  output is projected onto its manifold (Gram–Schmidt for rotations, sigmoid for
  bounded states), and the state is emitted `observed=False` with zero confidence.

Tests confirm exact byte encoding, the neutral start, that training reduces the
loss below 30% of its initial value on a toy set and learns durations, and that
generated states validate.

## 5. Training entrypoint (`scripts/train_text_to_motion.py`)

Data, checkpoints and logs live in an object store (R2, or a directory in tests).
AdamW with linear warmup and cosine decay, gradient clipping, bf16 autocast on CUDA,
validation every N steps (per-channel loss and geodesic error), checkpoints with
optimizer and RNG state under `training-runs/<run>/checkpoints/`, and automatic
resume that refuses a checkpoint from a different configuration or manifest. A test
trains, checkpoints, and resumes from the saved step.
