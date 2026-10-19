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
returned Google Drive's "quota exceeded" page, and the institutional copy (CORA,
doi:10.34810/DATA33) is behind an anti-bot challenge; neither was circumvented.
The clip zips are instead streamed into R2 (memory only, no local staging) from
the public Hugging Face mirror `aipieces/How2Sign` at pinned commit `1231830f`,
accepted only as a byte-identical copy of the publisher's files: its six
translation CSVs are SHA-256-identical to the Drive copies, every zip has exactly
the byte count Drive reported, the stream must hash to the mirror's declared
SHA-256 and the stored object is re-read and re-hashed, then
`scripts/verify_how2sign_zips.py` checks every member's CRC-32, reconciles clip
names with the official CSV, and byte-compares a sample of frontal clips with an
independently uploaded mirror (`WayenVan/how2sign-front-clips`). Each zip gets a
member manifest (name, size, CRC-32, SHA-256) in R2. OpenHands ASL pose archives
are acquired with the publisher's MD5 checked.

Verification results (2026-10-19), every member CRC-checked from R2:

| Zip | Clips | CSV sentences without a clip | Frontal clips byte-compared with the independent mirror |
|---|---:|---:|---|
| `train_rgb_front_clips.zip` | 31,047 | 118 | 621 / 621 identical |
| `train_rgb_side_clips.zip` | 31,165 | 0 | — (no independent side mirror) |
| `val_rgb_front_clips.zip` | 1,739 | 2 | 35 / 35 identical |
| `val_rgb_side_clips.zip` | 1,741 | 0 | — |
| `test_rgb_front_clips.zip` | 2,343 | 14 | 47 / 47 identical |
| `test_rgb_side_clips.zip` | 2,357 | 0 | — |

The frontal gaps are in the publisher's release itself: the independent mirror has
exactly the same clip counts. The lateral zips instead hold 81 empty 262-byte MP4
placeholders with no stream (69 train, 4 val, 8 test). 79 of them stand where the
frontal clip is also missing; 2 val sentences have a frontal clip but an empty lateral
one. Both extraction jobs record such clips as refused instead of failing. Sentences
with both views usable: 35,127 (train 31,047, val 1,737, test 2,343). Every readable
clip is 1280×720 H.264 at 24 fps.

## 2. Primary pipeline: two-view triangulation (decision in [DATA_QUALITY_DECISION.md](DATA_QUALITY_DECISION.md))

The monocular MediaPipe path below loses quality (6th of 8 estimators on downstream
sign-language translation in a 2026 study; monocular depth is a guess). The primary
pipeline replaces its body and hands:

* **2D:** Sapiens-1B (Goliath 308 keypoints, CC BY-NC 4.0) on both How2Sign views
  (`sapiens_keypoints.py`, `scripts/sapiens_keypoints_runpod.py`); the signer's box is
  the largest non-green region of the green-screen frame; heatmap peaks are refined
  to sub-pixel by per-axis parabola fits.
* **Synchronisation:** each frontal/lateral clip pair's lag is the maximiser of the
  normalised correlation of wrist/elbow vertical velocity (both cameras share the
  vertical); pairs below r = 0.8 are dropped, and a session's lags must agree.
* **Self-calibration** (`two_view_geometry.py`): RANSAC fundamental matrix, essential
  decomposition with the cheirality check, then bundle adjustment over both focal
  lengths and the relative pose with reprojection **and rigid-bone-length** residuals.
  The bone term is essential, not decorative: the two cameras fixate the signer, the
  degenerate case for focal recovery from epipolar geometry. In tests, without it the
  focal lengths come out 40–47% wrong while reprojection still looks perfect
  (0.3–0.4 px); with it they are within 1%.
* **Triangulation and IK:** linear triangulation in camera-1 coordinates, then the same
  swing IK; rotations are scale-invariant, so the unidentifiable metric scale does not
  affect them. MediaPipe still supplies the 52 face blendshapes, head rotation and
  blink on the same frontal frames.
* **Gate** (`two_view_states.py`, thresholds fixed before real data): bundle adjustment
  converged, median reprojection ≤ 2 px in each view, at least one synchronised pair
  with consistent lags, and triangulated bone-length CV below the monocular CV on the
  same clips. Only passing sessions produce states (`scripts/triangulate_how2sign_sessions.py`).
  End to end on a synthetic signer with fixating cameras: lag recovered exactly, focal
  lengths within 3%, forearm directions within 1.5° (median).

To verify on the real files: the lateral clip names are assumed to end in
`-rgb_side`, mirroring the frontal `-rgb_front`.

## 3. Monocular fallback (`data_engineering/video_landmarks.py`)

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

## 4. Sentence-aligned manifest (`scripts/build_how2sign_text_manifest.py`)

Each converted clip is joined with its realigned English sentence and given its
partition from the **frozen signer-disjoint fold plan** (B11; fold 0: 29,012 train,
1,861 validation, 292 test rows). Clips outside the plan (official validation and
test splits) are labelled `heldout_official` and never used for training or model
selection. The manifest kind `continuous` admits every channel the estimator
produced; per-frame masks inside each state carry the rest.

## 5. English to motion (`generation/text_to_motion.py`)

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

## 6. Training entrypoint (`scripts/train_text_to_motion.py`)

Data, checkpoints and logs live in an object store (R2, or a directory in tests).
AdamW with linear warmup and cosine decay, gradient clipping, bf16 autocast on CUDA,
validation every N steps (per-channel loss and geodesic error), checkpoints with
optimizer and RNG state under `training-runs/<run>/checkpoints/`, and automatic
resume that refuses a checkpoint from a different configuration or manifest. A test
trains, checkpoints, and resumes from the saved step.


## Generative decoders: diffusion and motion tokens (2026-10-19)

A decoder trained to minimise per-frame error predicts the conditional *mean*
motion. Where several signings of a sentence are plausible, the mean is none of
them: it is a damped, under-articulated pose. Two generative decoders were added
and measured against the regressor.

- **Diffusion** (`--objective diffusion`, `generation/text_to_motion.py`).
  - The trained channels, as a unit-scale vector, are noised on a cosine schedule.
  - The decoder, given the noisy motion and the noise level, predicts the clean
    motion through the same heads and per-channel losses (x₀ parameterisation).
  - Text is dropped 10% of the time for classifier-free guidance.
  - Sampling is seeded DDIM, with every estimate projected onto SO(3) and [0, 1].
- **Motion tokens** (`generation/motion_tokens.py`, `scripts/train_motion_tokens.py`),
  in the style of T2M-GPT.
  - Tokenizer: a 1D-conv VQ-VAE turns motion into discrete codes (time ÷4, EMA
    codebook, straight-through gradients), trained with the same per-channel losses.
  - Prior: a byte-level text encoder with a duration head conditions a causal
    transformer that predicts the codes, with seeded sampling and classifier-free
    guidance on the logits.

On a controlled bimodal task (one text, body rotated ±45°) the regressor outputs
0° — neither signing — while both generative decoders sample both modes (tested).

**Measured on real signing.**
- **Data:** proxy monocular states of train shard 0, split by the fold plan into 3,610
  train clips and 238 signer-disjoint val clips.
- **Models:** d = 128, 3+3 layers, 3,000 CPU steps each; tokenizer 128 hidden, 256
  codes.
- **Evaluation:** `scripts/evaluate_text_to_motion.py`, frame-aligned to each reference
  clip. Ratios are generated over real; 1 is realistic.

| | Regression | Diffusion (g = 1) | Motion tokens (g = 2) |
|---|---:|---:|---:|
| Fréchet distance (standardised clip features) | 2503 | 1999 | **1815** |
| Body angular-speed ratio | 0.024 | 0.229 | **0.774** |
| Body temporal-variation ratio | 0.081 | 0.377 | **0.675** |
| Hand angular-speed ratio (L/R) | 0.002 / 0.002 | 0.079 / 0.098 | **0.172 / 0.170** |
| Hand temporal-variation ratio (L/R) | 0.012 / 0.011 | 0.207 / 0.201 | **0.215** / 0.175 |
| Per-frame geodesic error, body | 18.2° | 19.3° | **17.0°** |
| Per-frame geodesic error, hands (L/R) | **24.0° / 23.8°** | 26.2° / 28.4° | 26.2° / 24.8° |

- **Motion tokens are best overall.** They are closest to real articulation and
  distribution, and best per frame on the body.
- **Diffusion** is second, ahead of regression on everything except per-frame error,
  which rewards the frozen mean.
- **Guidance:** it helps the token prior (Fréchet 2008 → 1815 from g = 1 to 2) and
  does not help diffusion.
- **The token generator's limit is its tokenizer.** Its own reconstructions of the
  validation clips have 16° body and 20° hand error and keep 16% of real hand speed,
  with 248 of 256 codes in use. A larger tokenizer trained longer is the next lever:
  reference tokenizers train about 100× longer.
- **Fréchet caveat:** generated clips are valid in every frame, while real clips lack
  hands in some frames, so part of each Fréchet value reflects missing hands rather
  than motion.

**Recommendation for the production run** (two-view states, full corpus, Runpod):
train the motion-token generator as the primary model, with a larger tokenizer
trained for longer, and diffusion as the second candidate; select on the
signer-disjoint validation metrics above. Rendered samples are in
`evidence/generated-motion-2026-10-19/`, each video showing reference | regression |
diffusion | motion tokens.
