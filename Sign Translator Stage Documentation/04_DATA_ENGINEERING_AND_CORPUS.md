# 04 — Data Engineering and Corpus Plan

## 1. Why data is the main bottleneck

The active synthetic corpus deliberately makes every modality consistent through
a vocabulary cipher and deterministic motion signatures. This is useful for
testing tensor contracts, but it removes the central difficulties of real sign
language:

- one meaning may have multiple acceptable realizations;
- timing and segmentation are ambiguous;
- signers vary in style, speed, handedness, and morphology;
- non-manual signals overlap manual signs;
- transcripts and glosses are not equivalent;
- camera geometry and pose tracking introduce uncertainty;
- available corpora differ in language, dialect, task, and license.

A larger synthetic dataset will not solve these problems.

## 2. Decisions required before acquisition

The group must record the following in a dataset charter:

1. **Target sign language and dialect.**
2. **Translation direction:** text-to-sign, speech-to-sign, sign-to-text, or a
   staged subset.
3. **Task:** isolated signs, continuous sentences, interpreted speech, or
   conversational signing.
4. **Output:** gloss, 3D joints, rotations, body-model parameters, or rendered
   avatar.
5. **Allowed use:** research, redistribution, model derivatives, commercial
   use, and identity rendering.
6. **Primary population and environment.**
7. **Definition of an unacceptable semantic or safety error.**

Datasets from different sign languages or annotation conventions must not be
combined merely because their files have similar shapes.

## 3. Canonical sample

A production sample should include at least:

```yaml
sample_id: stable unique ID
source_id: original recording or session ID
signer_id_hash: privacy-preserving stable group ID
language: target sign language
dialect: declared or unknown
split_group: signer plus source-session grouping
license: exact source license
consent_state: granted, withdrawn, or not_directly_verified
authorization:
  basis: direct_participant_consent or published_dataset_license
  license_url: canonical HTTPS license reference
  licensor: named rights-granting party
  evidence_uri: immutable local license or consent snapshot
  evidence_sha256: SHA-256 of that snapshot
  permitted_uses: exact allowed use strings
  permitted_actions: download, create_derivatives, model_training, commercial_use,
    redistribution, and/or identity_use
  personality_rights: verified or not_verified
  attribution_notice: required attribution text
  limitations: rights not granted or not independently verified
intended_use: one exact value from authorization.permitted_uses
video:
  uri: immutable source reference
  sha256: content hash
  fps: measured frame rate
  timestamps: per-frame times
audio:
  uri: optional immutable source reference
  sha256: optional content hash
  sample_rate: measured sample rate
transcript:
  text: source-language transcript
  timestamps: word or segment timing with uncertainty
sign_annotations:
  gloss: token sequence with time intervals
  nonmanual: channel-specific intervals
  spatial_loci: referent and locus annotations
  discourse: role shift or classifier annotations
motion:
  representation: joints, rotations, or body-model version
  values: versioned array reference
  confidence: per-frame/per-joint confidence
  validity_mask: valid observations
camera:
  intrinsics: optional calibrated matrix
  extrinsics: optional calibrated transform
provenance:
  pipeline_version: immutable version
  operations: ordered transformation records
split: train, validation, or test
```

The existing `data_engineering.Sample` is a useful governance foundation, but
an exporter to the active training format is still required.

## 4. Acquisition and governance

### Fail-closed acquisition

No download should occur until the following are recorded:

- valid license and named authorization basis;
- direct-consent status, without treating a publisher's license as direct consent;
- intended use;
- redistribution terms;
- derivative-model terms;
- face/identity restrictions;
- retention and withdrawal process.

Unknown permission is not permission.

For How2Sign under CC BY-NC 4.0, use
`published_dataset_license` with `not_directly_verified`; do not write
`granted` merely because the corpus is downloadable. Record the saved How2Sign
download/license page as the evidence file and hash its bytes. A conservative
non-commercial research configuration may permit `download`,
`create_derivatives`, and `model_training`, while omitting `commercial_use`,
`redistribution`, and `identity_use`. Set personality rights to `not_verified`
and record that CC BY-NC does not itself establish privacy, publicity, or other
personality-right permissions. This is a machine-checkable project policy, not
a legal conclusion.

### Provenance

Hash the original media and every derived artifact. Record:

- code revision;
- command/configuration;
- model versions used for pose fitting;
- environment and dependency lock;
- input hashes;
- output hashes;
- human correction events.

Reprocessing must create a new version rather than silently modifying an old
artifact.

## 5. Motion extraction and quality control

### Recommended staged representation

For an initial pilot:

1. retain original video as the source of truth;
2. extract dense 2D body, hand, and face landmarks with confidence;
3. fit or triangulate 3D motion where data permits;
4. map to a canonical rotation/body representation;
5. retain both observations and fitted parameters;
6. never discard confidence or validity masks.

### Required quality checks

- timestamp monotonicity and audio/video synchronization;
- left/right-hand consistency;
- scale and coordinate convention;
- bone-length stability;
- rotation validity;
- reprojection residual;
- missing-joint and interpolation rate;
- hand visibility and motion blur;
- face/non-manual visibility;
- implausible acceleration and joint limits;
- duplicate and near-duplicate recordings.

Automatic checks must produce review queues, not silently “clean” uncertain
samples into apparently reliable motion.

## 6. Annotation design

### Linguistic tiers

At minimum, distinguish:

- source transcript;
- gloss or lexical label;
- fingerspelling;
- manual sign intervals;
- brow, eye, gaze, mouth, head, and torso channels;
- negation and question scope;
- spatial referents and coreference;
- uncertain or disputed annotations.

### Agreement

Inter-annotator agreement should be calculated per tier and per phenomenon.
A high gloss agreement score cannot hide low agreement on mouth gestures or
spatial reference.

### Human review

Qualified signers should participate in:

- annotation-guideline design;
- ambiguous-case adjudication;
- error-taxonomy definition;
- output comprehension evaluation;
- release-risk review.

## 7. Splitting and leakage prevention

Split by groups before windowing or augmentation:

```text
group = signer identity + source recording/session
```

Recommended split roles:

- **training:** model fitting and augmentation;
- **validation:** checkpoint selection and calibration;
- **test:** locked final evaluation;
- **challenge sets:** rare phenomena, low visibility, dialectal variation,
  fingerspelling, spatial reference, and long-form discourse.

The current active readiness check detects only byte-identical pose overlap. The
grouped-split certificate in `data_engineering/splitting.py` should become the
authoritative splitter used by the actual corpus exporter.

## 8. Variable-length batching

The real loader must provide:

- motion lengths and frame masks;
- speech lengths after feature extraction;
- source-token masks;
- gloss and plan masks;
- non-manual channel masks;
- confidence weights;
- exact CTC-feasible lengths.

Padding must never be interpreted as concept zero or a valid acoustic frame.

## 9. Pilot dataset exit criteria

Do not call the corpus training-ready until:

- every sample validates against the canonical schema;
- authorization basis, license evidence, action scope, and consent status are
  machine-checkable;
- train/validation/test groups are leakage-free;
- every tensor can be traced to source media;
- variable lengths and confidence masks survive batching;
- normalization uses training data only;
- a small random sample is visually reviewed after every preprocessing stage;
- annotation agreement and uncertainty are reported;
- the active training loader consumes the exported batch without synthetic
  regeneration.
