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
with both views usable: 35,127 (train 31,047, val 1,737, test 2,343). MediaPipe monocular states exist for all 35,129 frontal clips (none refused).
Frame detection rates: pose 100%, face 99.9%, left hand 89–93%, right hand 92–95%;
the summary is in `evidence/how2sign-pilot-2026-10-19/monocular-extraction-summary.json`. Every readable
clip is 1280×720 H.264 at 24 fps.

**Release defect, take-5 sessions (audit B80).** In sessions whose name ends in `-5`,
about a third of How2Sign, every lateral clip holds ¾ of the frontal clip's frames and
shows a different moment of the recording. No time mapping aligns them: the epipolar
error stays at 15–16 px for every candidate scale, offset and linear placement,
against ~4 px for good sessions. The session script refuses them with that reason.
They are recovered from the raw 30 fps recordings (`runpod/pipeline.sh raw` and
`recut`, `docs/RUNPOD.md` §3c). The released frontal clips are exact runs of raw
frontal frames, so lateral keypoints are re-cut at the session's sub-frame camera
offset, found by the epipolar scan. On `a4uz0W33REs-5` the offset is 9.39 frames with
4.6 px epipolar error, and the session then passes the two-view gate. Until the stage
has run over the corpus, take-5 sentences keep monocular states.

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

### Body position in space (`data_engineering/body_space.py`, 2026-10-19)

Two-view geometry has no metric scale, so the signer's body sets it. Per session,
four anthropometric cues — interpupillary distance, hand length (wrist to middle
fingertip, summed along the fitted finger), forearm and upper arm — each give a
log-scale estimate. They are combined by generalised least squares under a
covariance with four parts:

* population variation and correlation from **ANSUR II** (6,068 US Army adults,
  equal-sex mixture; the public CSVs are archived in R2 under
  `acquisition/2026-10-17-v1/anthropometry-ansur2/` and pinned by SHA-256 in
  `scripts/derive_anthropometric_priors.py`, which `--check`s the embedded constants);
* a landmark-definition term;
* a reconstruction term;
* the median's sampling error.

A χ² of the residuals flags sessions whose cues disagree. The upper-arm reference is
de Leva's joint-centre length (278.4 mm), because ANSUR measures from the acromion.

`root_translation` (shoulder-joint midpoint) and `head_translation` (cyclopean eye,
the gaze origin) are written in metres in the session-world frame: camera-1 axes made
y-up, with the origin at the session's median torso anchor. The scale, its predicted
SD, χ² and per-cue lengths go in each session's gate report. Text-to-motion and the
token generators now train these channels by default, and evaluation reports
`position_cm`.

Validation against CMU Panoptic dome ground truth (3 sequences, self-calibrated
two-view):

| | office1 | pose3 | pose1 |
|---|---|---|---|
| scale error | +6.0% | +4.9% | −3.5% |
| torso displacement RMS | 1.5 cm (motion 5.5 cm) | 0.6 cm (motion 2.0 cm) | 0.9 cm (motion 2.8 cm) |
| head displacement RMS | 1.0 cm (motion 8.9 cm) | 0.7 cm (motion 2.2 cm) | 0.3 cm (motion 2.3 cm) |

The predicted SD is 6.7%. The reconstruction terms were set from these same
sequences, so this is a consistency check, not held-out accuracy. The floor is the
population term: a proportionally larger signer cannot be told apart from a nearer
one. Relative motion within a session is exact up to that one factor.

**On real How2Sign sessions** (the four smoke-test sessions, two passing the gate):

* **Scale:** two different signers on the same rig give 2.317 and 2.322 m per
  reconstruction unit. The unit is the camera baseline, so the two must agree; they
  do to within 0.2%, which checks the scale independently.
* **Cue lengths:** interpupillary 64–65 mm, forearm 25 cm, upper arm 29–31 cm, hand
  18 cm. The cues agree (χ² 0.58 and 2.14 on 3 degrees of freedom).
* **Positions:** the eyes sit a median 22.5 cm above the shoulder-joint midpoint, and
  the torso moves within 6 cm per clip.
* **Validity:** root and head translations are valid in 88% of frames and gaze in 89%.

### Eye gaze (`data_engineering/gaze.py`, `scripts/fit_gaze_model.py`, 2026-10-19)

Every How2Sign monocular state already carries MediaPipe's eye blendshapes
(`eyeLook{In,Out,Up,Down}`, blink, squint, wide) and the face crop's head rotation,
so no re-extraction is needed. They are calibrated to gaze on **MPIIFaceGaze**
(15 participants, 37,663 images with 3D gaze targets; CC BY-NC-SA 4.0, archived in R2
under `acquisition/2026-10-17-v1/gaze-mpiifacegaze/`). Each image is rescaled to
How2Sign's interpupillary distance (44.6 px) and cut to the same face crop that
`extract_clip` gives MediaPipe. Ground truth is expressed in that crop camera's
axes, since its optical axis is the ray through the crop centre.

Per eye, yaw and pitch are linear in the eye's seven blendshapes. The eye-in-head
direction is rotated by the head and the two eyes are averaged into one cyclopean
direction. Both `left_gaze` and `right_gaze` carry that direction: vergence moves
the iris by well under a pixel at this resolution, so a per-eye fit would only
reproduce the calibration screen's 0.55 m distance. A frame's gaze is withheld when
the eye is closing (blink ≥ 0.5), and confidence falls with blink. The data decides
which MediaPipe suffix is the anatomical left eye (r = +0.84 against horizontal gaze).

Leave-one-participant-out angular error at How2Sign resolution:

| model | mean | median | 90th pct |
|---|---|---|---|
| head direction only | 16.9° | 16.2° | 27.2° |
| head + constant eye offset | 11.7° | 10.9° | 20.5° |
| **calibrated blendshapes (used)** | **7.5°** | **7.0°** | **12.8°** |
| with a head-pose term | 6.6° | 6.2° | 11.2° |

The head-pose variant is not used: 73% of How2Sign head poses (yaw median −12°)
lie outside MPIIFaceGaze's calibrated range (yaw −11° to 14°), so its head term
would rest on unsupported covariates.

On 80 How2Sign clips both eyes are valid in 91% of frames, gaze lies a median 17°
from the camera axis, and median eye speed is 36°/s (99th percentile 250°/s,
saccade range). The model (`gaze_model.json`, pinned by SHA-256 in each report)
is applied in `scripts/triangulate_how2sign_sessions.py`. The generators train both
gaze channels by default and evaluation reports `angle_deg`.

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
- **Halving the tokenizer's downsampling (÷2 instead of ÷4, same budget)** improves its
  reconstruction: hand speed ratio 0.16 → 0.20–0.21, body 0.74 → 1.03, Fréchet 2506 →
  2351. Per-frame error stays at 16° body and 20° hands.
- **That error floor is partly the proxy data.** Monocular MediaPipe hand landmarks
  jitter, which inflates the reference hands' measured speed. The two-view states'
  hand refit (temporal smoothness, hinge prior) gives cleaner targets. The production
  tokenizer should use `--down-layers 1`.

**Recommendation for the production run** (two-view states, full corpus, Runpod):
train the motion-token generator as the primary model, with a larger tokenizer
trained for longer, and diffusion as the second candidate; select on the
signer-disjoint validation metrics above. Rendered samples are in
`evidence/generated-motion-2026-10-19/`, each video showing reference | regression |
diffusion | motion tokens.

## Tokens v2: part-wise residual VQ, skeleton motion loss, RQ prior (2026-10-20)

**Why.** On the How2Sign subset (3,968 training clips) the v1 tokenizer's
window-averaged loss sat flat at about 0.70–0.76 from step 800 on. At that point the
body rotation term (sin²(θ/2) ≈ 0.045) meant about 24° per joint, and the hand-location
term meant about 10 cm. The commitment term stayed near 1. A single codebook of 1,024
codes per two frames cannot cover the product space of body, 30 finger joints, face
and gaze.

| Problem | Fix | Where |
|---|---|---|
| A step's loss depends on which clips the batch drew (about ±0.1) | Validation every `--val-every` steps on fixed batches (fixed order, fixed noise seed, no text dropout). It uses an EMA of the weights (θ̄ ← 0.999 θ̄ + 0.001 θ), which checkpoints export as `ema`. The lowest validation loss is recorded in `best.json`, and evaluation loads EMA weights. | `scripts/train_motion_tokens.py` |
| Quantisation error from one codebook | Parts (body + translations, left hand, right hand, face + head/gaze/blink), each with its own encoder and **residual** quantiser: Q EMA codebooks on successive residuals, r_q = r_{q−1} − e_q, ẑ = Σ e_q. Quantiser dropout (p = 0.5, q ~ U{1..Q}). One shared decoder. | `generation/motion_rvq.py` |
| Rotation errors compound along the arm | Huber position, velocity and acceleration of all 44 arm and hand points by forward kinematics (scales 0.1 m, 0.5 m/s, 10 m/s²; weights 1, 0.5, 0.25) | `signing_space.skeleton_motion_loss` |
| Several codes per step to predict | Exact factorisation p(c_t) = Π_j p(c_t^j \| c_t^{<j}, h_t). A temporal transformer gives h_t from the past steps; a causal depth transformer over the J = P·Q codes of a step has one head per code position (RQ-Transformer). Classifier-free guidance applies at every depth step. | `RVQPrior`, `generate_rvq_motion` |
| Not knowing which part of the error is reachable | Per joint and per hand point, the error splits three ways: **target jitter** (the reference's angle to the geodesic midpoint of its neighbours), **decoder floor** (unquantised latents) and **quantisation gap** (full minus floor) | `scripts/diagnose_tokenizer.py` |

Tests check:
- RVQ is exact on sums of codes, and error does not increase with more stages;
- the part columns cover the unit vector exactly once;
- the code path equals the straight-through forward path in evaluation;
- the depth transformer is causal, and its prefixes agree;
- both models overfit a small batch;
- the EMA algebra is correct;
- train, validate, best.json, resume and evaluate work end to end.

Both versions run on the same manifest and are compared on validation:
- v1: `runpod/fleet/post.sh` chain on pod 1;
- v2: `runpod/fleet/train_v2.sh`, with tokenizer 15k steps and priors 8k steps,
  each selected by `best.json`.

**Tokenizer results** (`scripts/diagnose_tokenizer.py`, 120 validation clips; the
target jitter is a property of the data and is the same for both):

| Tokenizer | Joints full / floor | 44 FK points full / floor | Fingertips full | Wrists (body/LeftHand, body/RightHand) |
|---|---|---|---|---|
| v1 (one VQ codebook, step 15k) | 19.1° / 17.9° | 10.9 / 8.7 cm | 11.7 cm | 36–41° |
| v2 (RVQ + skeleton loss, best step 8k) | 9.6° / 9.1° | 2.2 / 1.8 cm | 2.5 cm | 22° |
| target jitter | 2.0° | 0.9 cm | 1.0 cm | 2.8–5.2° |

v2 halves the joint error and cuts point and fingertip error about fivefold. Its
quantisation gap (full − floor) is 0.5°, so the remaining error sits in the decoder
floor. v2's validation loss was lowest at steps 7.5k–8k and then rose (overfitting).
That motivated v2b (`runpod/fleet/train_v2b.sh`):
- no temporal downsampling (`--down-layers 0`);
- `--base-blocks` residual blocks at full frame rate (without them the encoder sees
  only 5 frames);
- early stopping with `--patience 4` validations.

**Tokenizer step speed.** A GPU profile of the v2b step (L40S, batch 64, synthetic
clips of 225–450 frames, hidden 512, 4 base blocks) put 55% of GPU time in float64
batched 3×3 matrix products. They came from rotation validation (`R^T R`, `det`) and
the geodesic diagnostic. cuBLAS pads each 3×3 block to a full GEMM tile, which costs
about 10× an elementwise product-sum in float64. The second cost was the arm and hand
FK, run one point and one joint at a time (5.6k of 9.2k forward ops). Changes, with
the same arithmetic:
- `pose/rotations.py`: 3×3 products and determinants written elementwise; each
  function reads its validity checks in one host sync, raising the same error in the
  same order;
- `signing_space.skeleton_points` composes both sides and all fingers at once,
  elementwise, and `skeleton_motion_loss` runs the prediction and the reference as
  one batch with masked means;
- `MotionRVQ.quantize`/`update_codebooks` handle all parts per stage at once.
  Codes and codebooks are bit-identical to the per-part quantiser on CPU and CUDA;
- `rvq_loss` returns its terms as device tensors. The training loop reads them only
  when it prints, validates or checkpoints (a non-finite loss is still refused before
  any save). `WeightEMA` uses `torch._foreach_*`.

| Synthetic batch of 64 | before | after |
|---|---|---|
| 225–450 frames | 397 ms/step | 180 ms/step |
| 300–600 frames | 518 ms/step | 248 ms/step |
| host syncs / dispatched ops per step (CPU count, small model) | ~830 / 15.5k | ~90 / 5.8k |

Peak memory rose 9.2 → 9.6 GiB. `torch.compile` of the encoders and decoder gave no
further gain (182 ms), so it is not used. Over several steps the new and old code
drift apart by float rounding about as fast as the old code drifts from itself
(cuDNN), and `tests/test_generation_motion_rvq_vectorised.py` pins the equivalence.
