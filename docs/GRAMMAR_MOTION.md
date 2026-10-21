# Grammar on motion: spatial reference and non-manual markers (2026-10-20)

`signtranslator/grammar_motion/` measures ASL grammar in canonical 3D motion. It turns
those measurements into generator conditioning, losses and metrics. The design follows
[the research report](../reports/ASL%20grammar%20and%20spatial%20reference.md):

- grammar becomes explicit, measurable state: loci, vectors, angles and spans;
- the literature's numbers are used as reference bands;
- the numbers the literature lacks are measured on How2Sign itself.

The symbolic layers stay as they were: `planning` for typed plans, `grammar` for the
SIR, and `facial_nmm` for intervals. This package grounds them in motion.

## What is measured

| Phenomenon | Module | Method |
|---|---|---|
| Signing space | `frames` | Chest-centred frame: p_B = C(t)ᵀ(p − s(t)), lengths in nominal shoulder widths, azimuth and elevation. Invariant to whole-body rotation (tested). |
| Pointing and loci | `pointing` | An index point needs four things: an extended index (min segment cosine ≥ 0.85, reach ≥ 0.9), the other fingers flexed (reach < 0.7), a still wrist for ≥ 3 frames, and the wrist forward of the chest. Pointing rays meet at a least-squares locus x* = (ΣA_i)⁻¹ΣA_i p_i, with A_i = I − d_i d_iᵀ. Loci are clustered by DP-vMF-means (Straub et al. 2015), so their number K is inferred. Events steeper than 50° are index-up handshapes, not points. |
| Non-manual markers | `nmm` | Blendshape tracks: brow raise (AU1+2), brow furrow (AU4, with the raise removed), squint (AU6+10), MM, PUFF, OO and jaw open. Each is z-scored against a robust neutral face pooled per recording session, smoothed at 3.5 Hz and segmented by hysteresis (on at 2.5 z, off at 1.5 z). Span edges are then refined to half-level crossings, and spans shorter than 0.15 s are dropped. |
| Headshake and nod | `nmm` | At least 3 reversals of head yaw (or pitch), each ≥ 6° and 0.1–0.6 s apart. This is Neidle et al.'s detector recast for rotations. |
| Role shift | `role_shift` | Features: torso yaw, head-minus-torso yaw, lateral torso position and the gaze-to-addressee angle. Exact PELT (Killick et al. 2012, Gaussian mean cost, BIC-like penalty; exact against brute force in tests) finds the change points. A segment counts as a shift when it departs from the narrator baseline by ≥ 2.5 robust σ and ≥ 5°, for ≥ 0.6 s. |
| Gaze to loci | `gaze_locus` | Per-frame gaze target (a locus, the addressee or "other") within 12°. The agreement window is [onset − 160 ms, onset + duration/2], from Thompson et al. 2006. Their rates (0.738 / 0.825 / 0.722 / 0.682 / 0.111) are the reference bands. |
| Agreement verbs | `agreement` | Direction error e_dir, endpoint error e_end (an angle, since hands rarely reach a locus) and facing cosine f. The loss is L_agr = λ_dir(1 − c_dir) + λ_end·e_end² + λ_face(1 − f), with λ_dir > λ_end (Lu & Huenerfauth 2012). A regular or backwards verb is classified from the sign of ⟨Δh, n_palm⟩. |
| Prosody | `prosody` | Movement units are split at minima of wrist speed, inside the active range (frames with a wrist in signing space). Blinks are robust-z eyelid closures. A boundary needs two cues (Tang et al. 2010) from four: lengthening, blink, brow change and head change. |
| English grammar | `text` | wh / yes-no / other question, negation, conditional, reported speech, third-person load and deixis. All are deterministic functions of the sentence. |

## How it reaches the generator

- **Conditioning.** `TextToMotion` and `TokenPrior` take the 8 English grammar features
  through a zero-initialised projection added to every text token. A model therefore
  starts exactly as its text-only counterpart (tested). The features are computed from
  the byte tokens, so they are identical at training, at generation and for the empty
  guidance text. Both training scripts turn this on by default.
- **Referent loci** (`loci_conditioning`, opt-in with `--locus-conditioning`). Each
  training clip's own pointing clusters fill 4 slots (unit direction and presence), in
  the order the referents are established. The slots enter through a zero-initialised
  projection and are hidden 30% of the time, and always when the text is dropped for
  guidance. At generation, `plan_loci` assigns slots from the English using the
  contrastive policy: one slot per third-person referent, or one deictic slot. The
  first referent goes at the measured How2Sign median (25.5° toward the signer's right,
  +22.6°), the second is mirrored, and later ones go further out. This makes spatial
  reference controllable. Passing explicit `loci` places the referents.
- **Losses.** These already train every two-view channel:
  - the signing-space location loss (`generation/signing_space.py`);
  - the gaze channels;
  - torso and head translation;
  - the face blendshapes, which carry the markers.

  `agreement_loss` and `gaze_locus_loss` are ready for data that has verb spans and loci.
  How2Sign has none, so they are not applied yet.
- **Evaluation.** `scripts/evaluate_text_to_motion.py` now reports `grammar`, using one
  neutral face pooled over the references for both sides. It gives the association AUC
  of each (English feature, detector) pair for generated and reference clips, and the
  per-score 1-Wasserstein distance and mean ratio.

## Validation on real How2Sign (no grammar labels exist)

`scripts/measure_grammar_on_how2sign.py` runs the detectors on the monocular states of
whole recording sessions, adding calibrated gaze, and compares them with the English.
The results and the reference distributions are in
`Sign Translator Stage Documentation/evidence/grammar-motion-2026-10-20/` and in R2
under `derived/grammar-motion-v1/`.

The test is weak in one direction only:
- an English feature does not guarantee the ASL marker;
- How2Sign is instructional monologue;
- but an AUC reliably above 0.5 cannot come from a detector blind to the marker.

The sample is 4,196 clips from 276 whole sessions (monocular states).

| English feature | Detector score | AUC (95% bootstrap CI) | Positives | Reading |
|---|---|---|---|---|
| wh-question | brow furrow, mean z | **0.731 (0.632–0.826)** | 32 | wh-furrow detected |
| wh-question | brow furrow, coverage | 0.613 (0.518–0.705) | 32 | |
| negation | headshake rate | **0.617 (0.599–0.636)** | 714 | headshake detected; per signer group c02 0.62, c03 0.71 |
| conditional (if/unless) | brow raise, mean z | **0.593 (0.567–0.619)** | 490 | conditional raise detected |
| yes/no question | brow raise, mean z | 0.590 (0.500–0.674) | 54 | borderline |
| deixis | pointing rate | 0.530 (0.521–0.540) | 1,947 | weak but reliable |
| third-person pronouns | pointing rate | 0.521 (0.505–0.539) | 473 | weak |
| reported speech | role-shift coverage | 0.470 (0.429–0.517) | 40 | **not validated**: no association |

**Reference distributions.** The literature gives none of these for ASL:

- **Pointing (superseded; see the two-view section below):** 11% of clips, 0.13 loci
  per clip, median azimuth 25° toward the signer's right and elevation +23°. Two
  faults make this an upper bound on noise rather than a distribution of points.
  Monocular wrist travel picks the left hand as dominant in 84% of clips. Two-view
  picks it in 10%, and both label the arms anatomically, so the measurement ran mostly
  on the non-dominant hand. The detector also counted index handshapes aimed across
  the body.
- **Markers:** role-shift segments occur in 16% of clips and headshakes in 16%.
- **Gaze during pointing (779 points): not found.** The median angle between gaze and
  the pointing direction is 53°, against 52° for the same clip's gaze at least 0.5 s
  away. Gaze is closer than that control in 46% of events, and within 20° in 0.9% (control
  1.9%). Either How2Sign signers keep looking at the addressee (camera) while pointing,
  or a frame offset remains between the monocular body and the gaze. The two-view
  re-measurement (post chain) separates the two.
- **Indicating paths: none found.** No movement unit runs from one detected locus to
  another within 20° in clips with two or more loci. The criterion (start and end
  directions on loci) is stricter than verb paths, which start near the body.
- **Prosody:** the final movement unit runs at 0.88× the median unit (interquartile range
  0.60–1.29), so the speed-minimum proxy does not show the ≈2× sentence-final lengthening
  Tang et al. report. The proxy is not a sign segmentation. Blinks fall in the final window
  of 8.6% of clips.

### Two-view re-measurement (2026-10-21)

The same script ran on the two-view states (`--states …/how2sign-two-view-v1`): 1,387
clips with triangulated body and hands, 289 sessions. The results are in
`evidence/grammar-motion-2026-10-20/how2sign-two-view-grammar-measurements-v2.json`
and R2 `grammar-motion-v1/two-view-measurements-v2.json`. The grammar records use
schema 2, where pointing requires the index to aim away from the signer
(`pointing.OUTWARD_COS`).

**Why schema 2.** Before it, 84% of right-hand "points" aimed across the body from a
hand held low in front of the chest. Their cosine with the outward direction had a
median of 0.06, so these were index handshapes of lexical signs. The outward criterion
keeps 23% of them, with a median outward cosine of 0.49.

| English feature | Detector score | AUC (95% CI) | Positives |
|---|---|---|---|
| wh-question | brow furrow, mean z | **0.716 (0.546–0.867)** | 11 |
| yes/no question | brow raise, mean z | **0.791 (0.593–0.960)** | 9 |
| conditional | brow raise, mean z | **0.600 (0.558–0.642)** | 159 |
| negation | headshake rate | **0.606 (0.574–0.639)** | 233 |
| reported speech | role-shift coverage | 0.474 (0.401–0.609) | 8 |
| deixis | outward pointing rate | 0.509 (0.498–0.521) | 661 |
| third-person pronouns | outward pointing rate | 0.507 (0.491–0.527) | 179 |

**What replicates.** All four non-manual markers replicate on the second data set,
including the yes/no raise that was borderline on monocular.

**What does not.**
- **Pointing:** outward pointing does not track English deixis or pronouns. The
  earlier 0.53 came from index handshapes that co-occur with deictic words.
- **Gaze during points:** gaze does not follow points (67 events, median 54° vs 55°
  control; closer than control in 49%). Throughout, gaze stays 14–25° from the camera
  axis, i.e. on the addressee. The frames agree (gaze and chest share the
  camera-1 y-up axes; checked per clip), so this is how How2Sign signers behave, not
  an offset.

**Distributions** (the training distribution of `state_loci`):
- 4.3% of clips hold an outward point, with 0.045 loci per clip.
- Locus azimuth: median 22° toward the signer's left, contralateral for the 90%
  right-dominant signers (5–95%: −68° to +69°).
- Locus elevation: median +13° (5–95%: −32° to +47°).
- Role shift appears in 18% of clips, headshake in 19%.
- Final blinks occur in 13% of clips.

**Consequences.**
- `loci_conditioning.PLANNED_SLOTS` now uses these medians, ±22.4° at +13.3°.
- The locus-conditioned prior is conditioned on rare, real points. With so few, it is
  expected to differ little from the plain prior, and its locus adherence is reported
  rather than relied on.
- Two-view and monocular arms differ by 20–35° in forearm and index direction,
  without a mirror (a mirrored match is 45–160° off). The two-view set passed the
  reprojection gate and is the training set.

## Limits

- **No grammar ground truth for How2Sign.** Detector precision and recall are unmeasured.
  The validation above is an association test. ASLLRP's frame-level non-manual spans
  (research-only licence) or the CUNY mocap corpus (by request) would give a direct test.
  The research report gives the access routes.
- **Loci are signer-normalised** (nominal bone lengths), not metric. Locus-to-gaze angles
  use the cyclopean gaze, which has a 7.5° calibration error.
- **Movement units are not signs.** Without glosses, phrase-final lengthening is measured
  on speed-minimum units and is a proxy only.
- **The thresholds are stated a priori,** not fitted. This covers pointing reach and
  curl, role-shift σ, degrees and duration, the hysteresis z values and the 50° index-up
  cut. The literature gives no ASL values for torso angle in role shift or for brow lead
  times.
