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

- **Pointing:** 11% of clips contain a referential point, with 0.13 loci per clip. Locus
  azimuth has its median at 25° toward the signer's right (5–95%: 59° right to 41° left),
  so ipsilateral space dominates. Elevation has its median at +23° (5–95%: −16° to +46°).
- **Markers:** role-shift segments occur in 16% of clips and headshakes in 16%.
- **Prosody:** the final movement unit runs at 0.88× the median unit (interquartile range
  0.60–1.29), so the speed-minimum proxy does not show the ≈2× sentence-final lengthening
  Tang et al. report. The proxy is not a sign segmentation. Blinks fall in the final window
  of 8.6% of clips.

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
