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

### v2b tokenizer and code learnability (2026-10-21)

v2b (`--down-layers 0 --base-blocks 4`, best step 7,500, early-stopped at 9,500) has the
best reconstruction so far:

| | v2b | v2 |
|---|---|---|
| Joints | 8.3° | 9.6° |
| Points | 1.7 cm | 2.2 cm |
| Fingertips | 1.95 cm | 2.5 cm |
| Wrists | 19° | 22° |

Reconstruction does not decide generation, however. The diagnosis now also measures
**learnability** (`code_statistics`): per code position the repeat rate, the unigram
entropy and the plug-in bigram conditional entropy, and the rate in nats per second that
a prior has to model. On 24 real validation clips:

| Tokenizer | Code steps/s × codes per step | Mean repeat rate | Bigram rate |
|---|---|---|---|
| v1 | 15 × 1 | 0.79 | 8 nats/s (about 22 codes in effective use) |
| v2 | 15 × 16 | 0.08 | 422 nats/s |
| v2b | 30 × 16 | 0.14 | 1,070 nats/s |

**What the numbers say.**
- v1's codes carry too little information, which matches its under-articulated output.
- v2 and v2b are codecs: near-uniform codes that change almost every step.
- The v2b prior learns slowly accordingly. Validation token loss went 6.11 → 5.89 → 5.60
  at steps 500 / 1,000 / 1,500, against a uniform loss of 6.24.

**The v3 candidates** (`runpod/fleet/train_v3.sh`, one pod each) run at 7.5 code steps/s
(`--down-layers 2`), the rate class of published RVQ text-to-motion:
- v3a: 4 stages × 512 codes;
- v3b: 2 × 512;
- v3c: 4 × 256.

Each chain runs tokenizer → diagnosis with learnability → plain prior → evaluation. The
model is selected on generation quality, with reconstruction and learnability as
explanations.

**Relaunch notes.**
- The first v2b prior chain failed on both priors at batch 128. At batch 64 the prior uses
  32 GB of an L40S's 46 GB. Its logs were on the container disk, which is wiped at stop, so
  every stage now uploads its log to R2 (`training-runs/<run>/stage-<name>.log`).
- A 128-core pod built validation batches at under one clip per second with unbounded
  OpenMP threads. The chains export `OMP_NUM_THREADS=8`.

### Generator round exit gate (declared 2026-10-21 05:50 UTC, before the diffusion and locus results)

Candidates:
- v1 tokens;
- v2b tokens: plain and locus-conditioned;
- v3a, v3b, v3c tokens;
- diffusion d512 and d384.

All are scored by `scripts/evaluate_text_to_motion.py` on the same 200 validation clips at guidance 1, 2 and 3. Selection uses validation only. The test partition (92 clips) is scored once, for the selected model only, and reported as the final number.

1. **Realism (primary):** the lowest Fréchet distance, at any guidance.
2. **Articulation:** speed and variation ratios for body, left hand and right hand within [0.7, 1.3]. Outside that range, the motion is frozen or jittery.
3. **Grammar markers:** generated / reference rate ratios for headshake, nod, brow raise and brow furrow within [0.5, 2.0].
4. **Accuracy against the reference:** body and hand geodesic error and wrist error are reported, not thresholded. A single reference cannot score valid alternative signings.
5. **Ties:** within 10% Fréchet, prefer the model that passes more of 2–3, then the lower hand error.

The round closes when:
- every candidate is evaluated;
- the selected model's test-partition evaluation is stored;
- the comparison table is in these docs and the audit;
- the best checkpoints are mirrored to the network volume;
- every training pod is stopped.

Failing gates 2–3 does not block the round. It is recorded as the problem for the next round, which is the plan in docs/NEXT_PHASE_PLAN.md.

### Meaning-level evaluator: the duration confound (2026-10-21)

**Version 1 measured duration, not meaning.** The first text–motion retrieval evaluator (`retrieval-ret-subset-v1`) reached validation R@3 0.349, against chance 0.094. The control exposed it:

| Clips scored (200 validation clips, R@3) | Score |
|---|---|
| Real motion | 0.365 |
| Same clips, every channel removed (duration only) | 0.370 |
| Real motion of other clips | 0.104 |

English length and clip duration correlate at 0.91, and the encoder saw whole clips with an absolute time code, so it learned to rank by length. The generators are given the reference durations, so they would have been credited for nothing. Every R-precision number from version 1 is void, including v3a's 0.365–0.385. The generated states those evaluations stored remain valid.

**Version 2 is duration-blind** (`text_motion_retrieval.py`):
- 2 s windows at a random offset in training, with within-window positions;
- scoring by the mean over windows at a 1 s stride.

Every evaluation now reports a duration-only control beside the real motion. A meaning score counts only when the control is at chance and the real motion is clearly above it.

`scripts/score_generated_retrieval.py` re-scores stored generated states without regenerating them.

### Generator round results so far (2026-10-21 07:40 UTC; 200 validation clips)

| Generator | Best Fréchet (guidance) | Body speed / variation | Right-hand speed / variation | Headshake ratio | Notes |
|---|---|---|---|---|---|
| v1 tokens | 934 (3) | 0.84 / 0.92 | 0.34 / 0.21 | 2.44 | under-articulated hands |
| v3a tokens (7.5 steps/s) | 662 (3) | 0.85 / 1.13 | 0.64 / 0.48 | 0.63 | prior never below chance on validation |
| v3b tokens | 733 (3) | 0.69 / 0.89 | 0.59 / 0.42 | — | |
| v3c tokens | 656 (3) | 0.77 / 0.90 | 0.62 / 0.45 | — | |
| v2b tokens (30 steps/s) | 251 (3) | 0.97 / 0.96 | 0.85 / 0.66 | 2.28 (g3), 1.48 (g1) | |
| **v2b tokens + loci** | **194 (2)** | 0.97 / — | 0.86 / 0.65 | 2.13 (g2), 1.37 (g1) | best so far; 17–19% of planned loci pointed at within 20° |
| diffusion d384 | 938 (1) | **0.18** / 0.45 | 0.23 / 0.32 | 0.03 | near-static |

**Diffusion d384 is near-static.** Its denoising validation was good (0.719), but its samples move at 0.15–0.18 of real speed:
- not a sampler setting: η 0 or 1, 50 or 200 steps and guidance 1–3 change nothing;
- the per-frame channel losses train no dynamics;
- d512 had the same objective and was stopped at step 18k.

The fix is the tokenizers' skeleton position/velocity/acceleration loss (`--motion-weight`). Runs `t2m-diffusion-m1` and `-m3` (weights 1 and 3) are training.

**Gates.** No candidate passes all of them yet:
- v2b + loci passes realism (best Fréchet) and articulation, apart from right-hand variation 0.65 just under 0.7;
- it fails the headshake gate above guidance 1.

R-precision is pending the duration-blind evaluator (`retrieval-hold`).

### Sampling-temperature sweep for v2b + loci (2026-10-21, guidance 1, 200 validation clips)

| Temperature (body, left hand, right hand, face) | Fréchet | Right-hand variation | Headshake ratio | Gates passed |
|---|---|---|---|---|
| 1.0 everywhere | 234 | 0.66 | 1.37 | 9/10 |
| 1.1 everywhere | 223 | 0.66 | 2.13 | 7/10 |
| 1.2 everywhere | 199 | **0.72** | 3.00 | 8/10 |
| **1.2, 1.2, 1.2, face 1.0** (`--part-temperature`) | **214** | 0.68 | **1.88** | **9/10** |
| 1.1, 1.25, 1.25, face 0.9 | 224 | 0.67 | 2.39 | 8/10 |
| 1.1, 1.2, 1.35, face 1.0 | 240 | 0.64 | 2.58 | 8/10 |
| 1.2, 1.2, 1.4, face 0.95 | 222 | 0.68 | 2.22 | 8/10 |

**What the sweep shows.**
- Raising only the right-hand temperature does not raise right-hand variation: it stays at 0.64–0.72 for any setting.
- The headshake ratio moves non-monotonically between 1.9 and 2.6.
- At this sample size, differences of a few hundredths are noise, so the sweep stops. Tuning further would fit the validation set.

**Selection rule result.** No setting passes every check. The rule picks the most checks passed, then the lowest Fréchet: body and hands at 1.2, face at 1.0, guidance 1. Its one failing check, right-hand variation 0.68 against the 0.7 gate, is carried to the next round (more data).

**Meaning.** On R-precision with the validated evaluator, v2b settings score 0.15–0.19, against real motion 0.177 and chance 0.094 (192 clips). Uniform temperature 1.2 scores 0.161. v1 and v3 score 0.08–0.13.

**Statistical resolution of the gate** (bootstrap over the 200 clips, 60 resamples, selected setting):
- right-hand variation 0.684, 95% CI 0.654–0.712;
- headshake ratio 1.88, 95% CI 1.40–2.77.

Both CIs contain their gate boundary. On 200 validation clips the gate cannot separate this setting from passing or failing either check; the larger evaluation set of the full-How2Sign round can.

### Diffusion with the skeleton motion loss: worse, not better (2026-10-21)

| Run | Motion weight | Best Fréchet | Body speed | Right-hand speed |
|---|---|---|---|---|
| d384 | 0 | 938 | 0.18 | 0.23 |
| m1 | 1 | 1,265 | 0.08 | 0.07 |
| m3 | 3 | 1,513 | 0.09 | 0.08 |

Adding position, velocity and acceleration terms to the x₀ estimate made the samples *more* static. At high noise the x₀ estimate is uncertain, and a velocity penalty on an uncertain estimate is smallest at zero velocity. The continuous diffusion generator in its current form (x₀-parameterised, per-frame losses, DDIM) does not produce signing motion at this data scale, and it is not a candidate.

What would test it properly, for a later round with more data:
- ε- or v-prediction with an MSE objective (a proper score for DDIM sampling);
- geometric losses applied only at low noise levels, as MDM does in practice;
- or latent diffusion on the v2b tokenizer's continuous latents.

### Generator round: closed (2026-10-21 09:55 UTC)

**Selected generator:**
- model: v2b tokens + locus conditioning (`training-runs/tokens-rvq-d0b-prior-loci-subset-v1-r2/checkpoints/step-00005000.pt`), which embeds the v2b tokenizer state;
- sampling: guidance 1, temperature 1.2 for body and hands and 1.0 for the face (`--part-temperature body=1.2,left_hand=1.2,right_hand=1.2,face=1.0`).

**Mirrored** to the network volume at `/workspace/sign-translator/checkpoints/`, with a SHA-256 manifest:
- the selected model;
- the plain v2b prior;
- the v2b tokenizer;
- the validated retrieval evaluator (`retrieval-hold`, step 1,500).

**Test partition (92 clips, scored once):**

| Measure | Result |
|---|---|
| Fréchet | 255 (validation 214) |
| Body speed / variation | 1.23 / **1.38** |
| Left hand speed / variation | 1.22 / 1.12 |
| Right hand speed / variation | 1.25 / 0.97 |
| Errors | body 19.4°, hands 24–26°, wrist 10.0 cm, fingertip 15.3 cm |
| R@3 | generated 0.172, real 0.141 (2 groups of 32; duration-only control at chance, 0.094) |

The marker ratios do not transfer. Headshake is 5.7× and nod 20× the reference, because the test split has 6 sessions whose signers keep their heads still: reference rates are 0.017 and 0.003, against 0.048 and 0.046 on validation. The generated rates are the same on both splits.

**Problems carried to the next round:**
1. **Head rotation over-active.** Speed ratio 1.87 on validation and 3.29 on test. The declared gate never checked the head channel, so the next gate does. Most likely cause: face-part code noise at 30 code steps/s. Fix candidates are output smoothing of head rotation, tuned on validation, or a lower code rate for the face part only.
2. **Right-hand variation** on validation: 0.68, 95% CI 0.654–0.712, borderline.
3. **Small evaluation splits.** 200 validation and 92 test clips from 6 sessions cannot resolve the gates (the CIs straddle them). The full How2Sign round gives larger, more varied splits.
4. **Meaning.** R-precision is near the real-motion ceiling, but that ceiling is itself only about 1.9× chance with a 4k-pair evaluator. More data is needed before meaning claims.
5. **Diffusion is not viable as implemented** (see above).

All pods stopped at 09:48 UTC.

### Head-rotation smoothing (post-processing, validation only; 2026-10-21)

`generation/postprocess.smooth_rotation_channel` applies a centred moving average of the head rotation, projected back onto SO(3). `scripts/score_saved_generated.py` re-scores stored generated states under it, with no regeneration. At smoothing 0 it reproduces the stored evaluation exactly: Fréchet 213.565, head speed 1.867.

Selected subset generator, 200 validation clips, with the head now in the gate:

| Smoothing | Head speed ratio | Fréchet | Gate checks |
|---|---|---|---|
| 0 | 1.87 | 213.6 | 10/12 (head speed, right-hand variation) |
| **0.2 s** | **1.11** | **211.1** | **11/12** (right-hand variation) |
| 0.33 s | 0.81 | 209.6 | 9/12 (headshake, nod under-produced) |

0.2 s is chosen on validation. The subset round's test split has been used once already, so this is not a test-split claim. The full-How2Sign chain (`post_full_train.sh`) sweeps 0.1 / 0.2 / 0.3 s on its saved validation states before its single test evaluation.
