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

## 10. Pre-Phase-2 evidence gate — 2026-09-10

Phase 2 is not authorized merely because several datasets collectively advertise the
needed fields. For requirement bundle (B), the gate is:

\[
\operatorname{pass}(B)=\exists d:\left(\bigwedge_{c\in B} I(d,c)\right)
\land A(d)\land R(d,u),
\]

where one source (d) must co-observe every required capability (c), (A(d)) means
the acquired bytes and schema were locally verified, and (R(d,u)) means the recorded
authorization permits intended use (u). Body motion from one corpus, gaze from another,
and English alignment from a third do not become a co-observed training record.

This rule is executable in `signtranslator/data_engineering/source_portfolio.py`. The
current registry fails all five required bundles:

| Bundle | Current result | Minimum evidence still required |
|---|---:|---|
| Continuous co-observed 3D research data | Fail | Locally verified continuous ASL with body, both hands, face, head, gaze, eyelid state, signer group, and text alignment in the same records |
| Continuous linguistic reference | Fail | Locally verified authentic gloss/manual timing, handshape, non-manual, head, and gaze tiers with qualified-ASL provenance |
| Commercial training authorization | Fail | Written permission covering the exact continuous multimodal corpus, stable signer grouping, and model-training use |
| Commercial render rig | Fail | Locally qualified body/hand/face/eye rig whose exact assets and runtime permit deployment |
| Qualified-ASL governance | Fail | Confirmed Deaf/ASL co-design and review relationship plus an accessible consent process |

The gate requires `LOCAL_VERIFIED`, not a publisher description, for data and rig
capabilities. It requires `QUALIFIED_HUMAN` for linguistic governance. `unresolved` and
`permission_required` both fail. These states are evidence records, not legal opinions.

### Resolved limitation: How2Sign pseudonymous signer grouping

The How2Sign CVPR 2021 supplemental states that signer information is provided per video
and gives Green Screen training counts by signer. The official counts sum to 31,047:

```text
1:892, 2:422, 3:1859, 4:398, 5:12102,
8:14596, 9:292, 10:0, 11:486
```

The immutable local audit contains exactly those non-missing per-code counts. Its 118
missing records reconcile as `1:1, 3:2, 5:44, 8:64, 11:7`. Therefore the numeric suffix
in the released Green Screen names is now certified as the official pseudonymous signer
ID for this release. It is not a person's identity and cannot be used to infer sensitive
attributes.

The addendum is stored outside the read-only source and historical v1 audit:

```text
/Users/jiangshengbo/Volumes/how2sign_audit/signer-evidence-v2/certificate.json
/Users/jiangshengbo/Volumes/how2sign_audit/signer-evidence-v2/how2sign_train_signers.csv
```

The mapping contains 31,165 rows and SHA-256
`82337aef0d4324cd121fd3e0be7d7b7e03838892466fa1b85d9e8618ccdea9cd`.
The certificate binds the original audit database, metadata, audit manifest, and pinned
official supplemental PDF, plus the exact certificate-generator source bytes and dependency
lock. The generator ran from a dirty worktree, so both the Git revision and source-content
hash are recorded rather than presenting the revision alone as the implementation state.
The earlier v1 addendum is superseded but retained as historical evidence. A final split
disjoint by both signer and `VIDEO_ID` has not been created; signer certification solves
the grouping-key ambiguity, not the split gate.

For the 31,044 clips in `valid` or `quality_warning`, usable counts by signer are:

```text
1:892, 2:422, 3:1859, 4:398, 5:12101,
8:14594, 9:292, 11:486
```

Signer IDs 8 and 5 alone contribute 47.01% and 38.98% of the usable corpus. Exhaustive
enumeration of all `3^8 - 3*2^8 + 3 = 5,796` nonempty assignments of eight signers to
train/validation/test shows that a 70/15/15 split cannot be closely attained. Under the
declared lexicographic objective—first minimize maximum absolute proportion error, then
sum of squared proportion errors, then validation/test count imbalance—the best counts are
26,695 / 2,198 / 2,151, or 85.99% / 7.08% / 6.93%. The unavoidable maximum deviation is
15.9909 percentage points. The balanced validation/test representative assigns signers
5 and 8 to training; 1, 2, 4, and 11 to validation; and 3 and 9 to test.

That assignment is a feasibility witness, not an approved final split. Its test set has
only two signer IDs, its validation/test coverage is small, and a single dominant-signer
choice can materially change results. The scientifically defensible interim design is
signer-held-out repeated evaluation with per-signer results and source grouping inside each
fold. A fixed industrial evaluation split requires additional authorized signers and a
new, preregistered split analysis.

Official evidence:

- [How2Sign download, license, and limitations](https://how2sign.github.io/)
- [How2Sign CVPR 2021 supplemental](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Duarte_How2Sign_A_Large-Scale_CVPR_2021_supplemental.pdf)

## 11. Source acquisition portfolio

| Source | What it can legitimately contribute | Access and rights status | Why it cannot independently pass Phase 2 |
|---|---|---|---|
| [How2Sign](https://how2sign.github.io/) | Local continuous frontal RGB, 137-node 2D OpenPose, English alignment, certified pseudonymous signer grouping | Locally verified; CC BY-NC 4.0 research use; commercial use prohibited by current terms | Released local observations are 2D; English is not gloss; no qualified review; no commercial path |
| [SignAvatars](https://github.com/ZhengdiYu/SignAvatars) | Requested SMPL-X body, articulated hands, jaw, expression, camera translation for continuous subsets | Request form; non-commercial research terms; commercial permission must be requested separately | Not acquired; source does not document explicit eye gaze; How2Sign text remains English translation, not gloss |
| [ASLLRP / ASLLVD](https://www.bu.edu/asllrp/av/dai-asllvd.html) | Authentic gloss conventions, sign timing, start/end handshapes, native-signer tokens, four synchronized views; continuous corpora add non-manual tiers | Account/access process; research and education; commercial use requires explicit permission under [DAI terms](https://www.bu.edu/asllrp/dai-terms.html) | Linguistic reference rather than continuous metric 3D motion; not acquired or commercially authorized |
| [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) | Community-sourced isolated-sign vocabulary and recognition challenge data | Downloadable; Microsoft asks prospective commercial users to contact `ASL_Citizen@microsoft.com` | Isolated recognition data cannot replace continuous language-to-motion co-observation |
| [NVIDIA ASL dataset](https://www.nvidia.com/en-us/gated-resources/trustworthy-ai-american-sign-language/dataset/) | Raw videos, images, hand landmarks, body poses, and facial meshes after approval | Personal application requires acceptance of dataset and privacy terms; commercial scope unresolved | Gated and isolated-sign oriented; no locally verified schema or continuous linguistic annotations |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/modellicense.html) | Research body/hand/face parameterization and evaluation bridge | Standard license is non-commercial; commercial incorporation/training requires a separate license | A representation/rig is not observations or linguistic truth; standard terms do not permit the target deployment |
| [MakeHuman core assets](https://static.makehumancommunity.org/about/license.html) | Candidate deployment rig assets published as CC0 | Publicly downloadable and potentially commercial, subject to exact asset/dependency inventory | Not locally qualified for ASL finger, face, eyelid, gaze, retargeting, or deterministic rendering fidelity |
| [Gallaudet Motion Light Lab](https://gallaudet.edu/visual-language-visual-learning/ml2/) | Potential Deaf-centered co-design, qualified review, and 3D motion-capture expertise | Partnership inquiry required | No agreement, review protocol, capture protocol, or accessible-consent artifact exists yet |

The preferred acquisition order is: (1) SignAvatars as a research-only 3D bridge;
(2) ASLLRP continuous data as linguistic and multiview reference; (3) a qualified Deaf/ASL
partnership; (4) an independently governed synchronized capture corpus with explicit
research and commercial model-derivative terms; and (5) a commercially authorized rig.
ASL Citizen and NVIDIA are auxiliary isolated-sign evidence, not substitutes for steps
1–4.

## 12. Secure source-specific data treatment

Every acquired source enters quarantine before the canonical schema:

1. save the exact license/permission and download manifest; hash them before extraction;
2. reject symlinks, absolute paths, `..` traversal, duplicate archive members, decompression
   bombs, unexpected file types, and case-folding name collisions;
3. mount raw media read-only and write versioned derived artifacts elsewhere;
4. inventory and hash every member before decoding; record tool and code identities;
5. validate timestamps, units, coordinate frames, handedness, camera assumptions, masks,
   signer/session keys, and annotation vocabulary against the source specification;
6. retain unavailable channels as unavailable masks—never zeros presented as observations;
7. run source-specific pilot review before full conversion; and
8. keep research-only and deployment-authorized lineages physically and logically separate.

SignAvatars publishes annotations as Python pickle files. Pickle may execute code while
loading. Treat every pickle as untrusted: conversion must run in a disposable, network-off
sandbox with read-only input, CPU/memory/time limits, no credentials, pinned dependencies,
and a strict allowlist of primitive array containers. The converter must reject arbitrary
classes and emit non-executable arrays plus a schema-and-hash manifest. Never unpickle the
download directly in the training process.

For NVIDIA JSON, a `z` field must not be called metric depth until the downloaded schema
and capture specification establish its coordinate system, scale, and provenance. For
ASLLRP/SignStream tiers, preserve the source gloss labels and tier conventions verbatim;
normalization must be a parallel, reversible layer. For all sources, a visually plausible
fit remains an estimate with its own confidence and cannot overwrite the observations.

## 13. Correspondence and access actions

### Sent and traceable

- **How2Sign gloss request:** [issue #28](https://github.com/how2sign/how2sign.github.io/issues/28),
  opened from the project's authenticated GitHub account and still open as checked on
  2026-09-10. It requests original EAF/lossless annotations, train/validation/test mapping,
  tier conventions, timing, checksums/version, and license coverage.
- Existing upstream requests for [3D keypoints](https://github.com/how2sign/how2sign.github.io/issues/14),
  [RGB-D](https://github.com/how2sign/how2sign.github.io/issues/15), and
  [camera intrinsics](https://github.com/how2sign/how2sign.github.io/issues/26) remain open.
  Duplicate issues should not be created.
- **SignAvatars schema request:** [issue #20](https://github.com/ZhengdiYu/SignAvatars/issues/20),
  opened on 2026-09-10. It requests release/checksum identity, exact How2Sign mappings,
  coordinate frames/units/camera conventions, validity masks, gaze/eyelid availability,
  and the supervision basis of facial parameters. It keeps license and commercial-use
  questions private, as directed by the project README.

### Prepared; not falsely marked sent

The current environment was denied computer-control access to Mail. The following are the
exact inquiries to send when an authenticated email client is available. They request
information and permission; they do not accept terms or claim institutional authority.

**To:** `shaolihuang@tencent.com`; cc `ZhengdiYu@hotmail.com`, `z.yu23@imperial.ac.uk`

**Subject:** SignAvatars access, schema, and future commercial-permission inquiry

**Body:** I am conducting a student-led English-to-ASL 3D translation research project.
I would like to request the SignAvatars annotations for non-commercial research and ask
separately whether a path exists for future commercial training/deployment permission.
Before applying, could you confirm dataset/version checksums, the exact How2Sign ID mapping,
coordinate and camera conventions, whether eye-gaze or eyelid state is represented, invalid
frame/hand masks, signer grouping fields, and whether trained model derivatives have terms
distinct from redistribution? I will keep research-only data isolated from any future
commercial lineage and will not treat English translations as gloss.

**To:** `carol@bu.edu`

**Subject:** ASLLRP continuous corpus access and annotation-tier inquiry

**Body:** I am conducting a student-led research project on reliable English-to-ASL 3D
generation. I am seeking ASLLRP data as a linguistic and multiview validation reference,
not as inferred metric 3D ground truth. Could you advise which available continuous corpus
includes manual gloss timing, handshape, brow/eye/gaze, mouth, head movement, and camera or
synchronization metadata; the current account/access procedure; and whether permission can
be discussed for future commercial model evaluation or derivatives? I will preserve source
tier conventions, citations, access controls, and non-redistribution requirements.

**To:** `motionlightlab@gallaudet.edu`

**Subject:** Deaf-centered review and 3D sign-avatar research partnership inquiry

**Body:** I am building a student-led research program toward a reliable English-to-ASL 3D
translation system. I am looking for qualified Deaf/ASL co-design and review—not a nominal
endorsement—and guidance on motion-capture protocol, non-manual/gaze annotation, error
taxonomy, accessible consent, and blinded comprehension evaluation. Would Motion Light Lab
be open to a preliminary conversation about appropriate partnership structure, scope,
compensation, participant protection, and what evidence would be required before any
deployment claim?

Additional commercial-rights inquiries should go to `ASL_Citizen@microsoft.com` for ASL
Citizen and the SMPL-X licensing contact identified on the official model-license page.
The NVIDIA form must be completed by the user because it transmits identity/organization
data and requires personal acceptance of dataset and privacy terms.

Any new human capture must undergo the applicable institutional human-subjects review.
[HHS informed-consent guidance](https://www.hhs.gov/ohrp/regulations-and-policy/guidance/faq/informed-consent/index.html)
requires prospective, legally effective consent unless an authorized exception applies,
and emphasizes understandable communication, voluntariness, questions, and withdrawal.
For ASL participants, accessibility cannot be reduced to an English-only signature form.
