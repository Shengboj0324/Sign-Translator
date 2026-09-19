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
| [RIT/CUNY ASL Motion-Capture Corpus](https://latlab.ist.rit.edu/downloads.html) | Unscripted continuous multi-sentence ASL with dominant-hand gloss timing, some non-dominant-hand gloss, English translation, spatial referents, BVH/FBX motion, and front/side/face video | Access by inquiry to Matt Huenerfauth; the download page does not publish a dataset license, commercial grant, or model-derivative terms | Closest known legacy co-observed linguistic-motion source, but only 98 stories from 3 signers; rights, capture schema, calibration, facial channels, consent scope, and present availability must be verified |
| [ASL STEM Dialogue Motion Capture (LREC 2026)](https://aclanthology.org/2026.lrec-1.669/) | Natural instructor-student ASL dialogue plus isolated vocabulary, synchronized Vicon 3D body/hand/finger trajectories, RGB video, sentence translation, and SLAASh-aligned STEM sign timing | Paper says data may be made available to researchers under proper agreements; no public dataset license or commercial grant is stated | Only 2 fluent Deaf signers and 8.52 minutes of dialogue; no co-observed facial-expression or gaze capture is reported; annotation authors explicitly report limitations |
| [3D-LEX v1.0](https://github.com/OlineRanum/SAPA) | True high-resolution Vicon body pose, StretchSense hand data, ARKit facial blendshapes, raw markers/sensors, and retargeted FBX for 1,000 ASL lexical items | Dataset is CC BY 4.0; project repository says to contact the authors for access | Best identified permissively licensed 3D lexical source, but isolated and one example per sign; 5 total signers and only 1 primarily ASL signer make it unfit as standalone ASL supervision |
| [ASLLRP / ASLLVD](https://www.bu.edu/asllrp/av/dai-asllvd.html) | Authentic gloss conventions, sign timing, start/end handshapes, native-signer tokens, four synchronized views; continuous corpora add non-manual tiers | Account/access process; research and education; commercial use requires explicit permission under [DAI terms](https://www.bu.edu/asllrp/dai-terms.html) | Linguistic reference rather than continuous metric 3D motion; not acquired or commercially authorized |
| [Dennis Cokely Parallel Corpus](https://encompass.eku.edu/cokely_videos/) | Downloadable MP4/EAF examples with project-specific ID-gloss annotations and English–ASL idea-unit alignment | One *I Have a Dream* EAF/HD/SD item is locally hash-bound; publisher labels the collection CC BY-NC-SA 4.0 | Six formal-speech translations are a valuable annotation/reference pilot, not broad-domain supervision, canonical SIR, 3D motion, or commercial authorization |
| [CARD / SLAASh](https://sites.google.com/gallaudet.edu/card/data) | Current ELAN templates and annotation-convention lineage, including ASL Signbank linkage | Public documentation/templates; exact rights must be preserved per downloaded artifact | Annotation infrastructure and conventions, not How2Sign glosses or a co-observed training corpus |
| [ASL Signbank](https://aslsignbank.haskins.yale.edu/) | Controlled ID-gloss and lexical/phonological reference for future human annotation | Registration and manual approval; database is actively curated | Lexical reference does not provide sentence-level How2Sign timing, SIR supervision, or motion |
| [ASL-LEX 2.0](https://asl-lex.org/download.html) | Downloadable lexical properties for 2,723 signs under the published database license | Database listed as CC BY-NC 4.0; its reference videos are expressly excluded from that license and may not be saved or reused without permission | English labels and lexical properties are not sentence annotations; video restrictions and noncommercial terms block deployment reuse |
| [ASL STEM Wiki](https://www.microsoft.com/en-us/research/project/asl-stem-wiki/) | 315.84 hours/64,266 sentence-aligned videos from 37 certified interpreters, consented under IRB review; 5 shared control articles support signer-variation analysis | Public download; license permits only non-commercial, non-revenue-generating research and forbids redistribution; Microsoft invites separate commercial-use inquiries | High-value continuous English-ASL research corpus, but not natural ASL-first discourse, only a subset has fingerspelling timing, no full gloss/non-manual tiers, and no 3D ground truth |
| [YouTube-ASL](https://github.com/google-research/google-research/blob/master/youtube_asl/README.md) | 11,093 video IDs, 984 hours, and 610,193 English-caption segments; native Deaf annotators removed low-quality or poorly aligned videos | Release provides YouTube identifiers rather than owned media; repository code licensing does not grant rights in the underlying videos | Useful broad-domain research index, but no authentic gloss, 3D, stable media availability, participant consent record, or dependable commercial media authorization |
| [OpenASL](https://github.com/chevalierNoir/OpenASL) | Reproducible ASL→English research baseline and pointers to continuous online media | Repository states CC BY-NC-ND 4.0; third-party source-media rights and link stability require per-item verification | Wrong task direction and no authentic gloss/SIR, canonical 3D, or commercial path; cannot substitute for governed Phase-3 evidence |
| [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) | Community-sourced isolated-sign vocabulary and recognition challenge data | Downloadable; Microsoft asks prospective commercial users to contact `ASL_Citizen@microsoft.com` | Isolated recognition data cannot replace continuous language-to-motion co-observation |
| [PopSign v2.1](https://signdata.cc.gatech.edu/view/datasets/popsign_v2_1/index.html) | 200,686 mobile videos of 562 isolated signs from 47 signers; strong viewpoint/device variation for auxiliary visual robustness | Dataset page states CC BY 4.0; 1.1 TB full size, so subset and manifest access should be negotiated before download | Permissively licensed and large, but prompt labels are isolated vocabulary labels, not continuous gloss/SIR, non-manual grammar, or 3D motion |
| [NVIDIA ASL 1000](https://registry.opendata.aws/asl_1000/) | Controlled-access high-fidelity videos with 2D hand, pose, and face landmarks; official description says automated labels were human-corrected | Purpose-limited, revocable NVIDIA data license allows derivatives only to advance technology access for the Deaf community; redistribution is prohibited and termination requires deletion | Sequence/label schema and exact scope are not locally verified; commercial product and trained-model distribution rights are not explicit enough for a deployment lineage without written clarification |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/modellicense.html) | Research body/hand/face parameterization and evaluation bridge | Standard license is non-commercial; commercial incorporation/training requires a separate license | A representation/rig is not observations or linguistic truth; standard terms do not permit the target deployment |
| [MakeHuman core assets](https://static.makehumancommunity.org/about/license.html) | Candidate deployment rig assets published as CC0 | Publicly downloadable and potentially commercial, subject to exact asset/dependency inventory | Not locally qualified for ASL finger, face, eyelid, gaze, retargeting, or deterministic rendering fidelity |
| [Gallaudet Motion Light Lab](https://gallaudet.edu/visual-language-visual-learning/ml2/) | Potential Deaf-centered co-design, qualified review, and 3D motion-capture expertise | Partnership inquiry required | No agreement, review protocol, capture protocol, or accessible-consent artifact exists yet |

### Acquisition verdict and priority order — 2026-09-19

No identified source is sufficient by itself for the industrial target. A sufficient source
would have to co-observe continuous ASL, authentic linguistic timing, body, both hands,
face, head, eyes/gaze and eyelids; preserve signer/session grouping; cover enough signers
and domains for held-out evaluation; and authorize commercial training, model derivatives,
deployment, retention, and the intended identity treatment. Combining unpaired fields from
different datasets cannot manufacture that co-observation.

The evidence-backed acquisition order is:

1. **Request the RIT/CUNY corpus and the 2026 ASL STEM dialogue corpus immediately.**
   They are the two strongest co-observed continuous ASL-plus-motion candidates. Require
   sample files, complete schemas, timing/calibration specifications, consent and license
   documents, signer/session keys, and written model-training/derivative terms before use.
2. **Acquire 3D-LEX as the first permissively licensed metric-3D lexical reference.** Its
   CC BY 4.0 status, raw Vicon/hand-sensor data, facial blendshapes, and FBX exports make it
   valuable for converters, retargeting, rotation validation, and hand/face unit tests. Its
   signer and language-authenticity limitations prohibit treating it as continuous ASL truth.
3. **Pursue ASLLRP continuous data and ASL Signbank access.** These remain the strongest
   linguistic-convention and multiview references for authentic source-native gloss and
   non-manual annotation, but require separate commercial permission.
4. **Acquire ASL STEM Wiki for research-scale continuous diversity if storage permits.**
   It provides consented, professional, sentence-aligned signing at far greater scale than
   How2Sign, but its current license and interpretese limitations keep it outside the future
   commercial lineage. Download only after recording size, checksum, version, and storage.
5. **Use PopSign v2.1 as the leading commercial-compatible auxiliary visual corpus.** Its
   CC BY 4.0 license and signer/device variation are useful for isolated-hand/body robustness,
   never as sentence grammar, coarticulation, non-manual, or motion-generation supervision.
6. **Evaluate NVIDIA ASL 1000 and ASL Citizen only after rights clarification.** Both are
   potentially useful auxiliary resources, but the current terms do not yet establish a safe
   industrial model-distribution path. YouTube-ASL and OpenASL are research indexes/baselines,
   not an industrial media lineage.
7. **Continue How2Sign, Cokely, and SignAvatars only inside their current research lineage.**
   Preserve source-native labels and never convert English text into gloss or SIR.
8. **Commission the missing production corpus with a qualified Deaf/ASL partner.** This is
   not optional unless a newly acquired source passes every gate above. The capture contract
   must include synchronized full-body/finger/face/eye data, direct consent, withdrawal and
   retention policy, qualified annotation/adjudication, diverse signers/domains, stable groups,
   and explicit commercial model-derivative and deployment rights.

This ordering makes 3D-LEX the best immediately identified reusable 3D engineering source,
RIT/CUNY and the 2026 STEM dialogue the best continuous motion inquiries, ASLLRP the best
linguistic reference, ASL STEM Wiki the best consented continuous scale source, and PopSign
the best permissively licensed large isolated auxiliary source. None is an industrial corpus.

The implemented Phase 3B software treatment for an accepted EAF source is documented in
`11_PHASE_3B_GOVERNED_MAPPING_AND_ADJUDICATION.md`. It binds source annotations to a
label-empty qualified-human workflow; it does not reinterpret any source ID-gloss as project
SIR or training truth.

The subsequent evidence-integration, lexical-motion, abstention, and artifact-blocker
contracts are documented in `12_PHASE_3C_EVIDENCE_INTEGRATION.md`. That pre-artifact
software boundary does not convert any current source into training-authorized SIR or
canonical 3D motion.

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

For Cokely or ASLLRP EAF/XML, parse with DTD resolution, external entities, XInclude, and
network access disabled; enforce byte, nesting-depth, tier-count, annotation-count, and
text-length limits. Preserve `TIME_SLOT_ID`, millisecond values, tier names, participant/
annotator metadata, controlled-vocabulary references, parent/ref-annotation chains, media
descriptors, and source order before any derived normalization. Bind each EAF to the exact
media hash rather than trusting a relative filename, reject dangling or cyclic references,
and report overlapping, negative, reversed, or media-out-of-range intervals. A source ID-
gloss remains an annotation under that source's convention; it must not be silently mapped
to the project's SIR lexicon or treated as a How2Sign label.

The implemented Phase-3A reader, immutable binding contract, qualified-human mapping
protocol, and real-artifact stop rules are specified in
`10_PHASE_3A_LINGUISTIC_REFERENCE_INGESTION.md`. As of 2026-09-13, one authentic Cokely
EAF/HD-video/SD-video item and its publisher page are present under the read-only local
data volume. They pass complete-inventory hashing, EAF 3.0 structural validation, exact
descriptor binding, independent video inspection, deterministic ingestion, and warning-
strict tests. This is a one-item source-compatibility result only: corpus-wide compatibility,
qualified-ASL mapping, linguistic validity, commercial authorization, and How2Sign label
transfer remain unpassed.

## 13. Correspondence and access actions

### Sent and traceable

- **How2Sign gloss request:** [issue #28](https://github.com/how2sign/how2sign.github.io/issues/28),
  opened from the project's authenticated GitHub account and still open as checked on
  2026-09-12, with zero replies/comments. It requests original EAF/lossless annotations,
  train/validation/test mapping, tier conventions, timing, checksums/version, and license
  coverage.
- Existing upstream requests for [3D keypoints](https://github.com/how2sign/how2sign.github.io/issues/14),
  [RGB-D](https://github.com/how2sign/how2sign.github.io/issues/15), and
  [camera intrinsics](https://github.com/how2sign/how2sign.github.io/issues/26) remain open.
  Duplicate issues should not be created.
- **SignAvatars schema request:** [issue #20](https://github.com/ZhengdiYu/SignAvatars/issues/20),
  opened on 2026-09-10 and still open with zero replies/comments as checked on
  2026-09-12. It requests release/checksum identity, exact How2Sign mappings,
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

**To:** `matt.huenerfauth@rit.edu`

**Subject:** RIT/CUNY ASL Motion-Capture Corpus access and rights inquiry

**Body:** I am conducting a student-led research project toward reliable English-to-ASL 3D
generation. Your ASL Motion-Capture Corpus appears uniquely relevant because it co-observes
continuous discourse, gloss timing, English translation, spatial referents, motion capture,
and multiple video views. Could I request access to the current release and a sample package?
Before use, I need to document the exact license; permitted model training and derivative
models; commercial/deployment path, if any; participant-consent and retention scope; signer
and session keys; skeleton, units, coordinate frames, calibration and synchronization; face,
eye and finger channels; annotation conventions; checksums; and any known defects. I will not
infer permissions or absent channels from the publication description.

**To:** `o.ranum@surrey.ac.uk`

**Subject:** 3D-LEX v1.0 access, release identity, and schema inquiry

**Body:** I am conducting a student-led research project on reliable English-to-ASL 3D
generation and would like to acquire the CC BY 4.0 3D-LEX v1.0 release. Could you confirm the
current access path, release/version and checksums, file inventory, signer/session mappings,
timebases, coordinate systems and units, calibration records, validity/missing-data encoding,
body-hand-face synchronization, FBX skeleton and blendshape definitions, and whether the
published CC BY 4.0 grant applies to every distributed data component? I will treat 3D-LEX as
an isolated lexical/engineering reference rather than continuous or signer-diverse ASL truth.

**To:** `Lorna.quandt@gallaudet.edu`

**Subject:** Access inquiry for 2026 STEM ASL dialogue motion-capture data

**Body:** I am conducting a student-led research project toward reliable English-to-ASL 3D
generation. I am interested in the motion-capture dataset described in *How Pragmatics Shape
Articulation* (LREC 2026), which states that access may be possible under proper agreements.
Could you advise the appropriate request process and whether sample 3D/RGB/ELAN artifacts,
schema and calibration documentation, signer/session keys, checksums, and license/consent
terms are available? I also need explicit clarification of research and commercial model-
derivative rights, retention/withdrawal obligations, and which body, finger, face, eye and
non-manual channels are genuinely recorded versus absent.

**To:** `motionlightlab@gallaudet.edu`

**Subject:** Deaf-centered review and 3D sign-avatar research partnership inquiry

**Body:** I am building a student-led research program toward a reliable English-to-ASL 3D
translation system. I am looking for qualified Deaf/ASL co-design and review—not a nominal
endorsement—and guidance on motion-capture protocol, non-manual/gaze annotation, error
taxonomy, accessible consent, and blinded comprehension evaluation. Would Motion Light Lab
be open to a preliminary conversation about appropriate partnership structure, scope,
compensation, participant protection, and what evidence would be required before any
deployment claim?

Additional commercial-rights inquiries should go to `ASL_Citizen@microsoft.com` for both
ASL Citizen and ASL STEM Wiki, and to the SMPL-X licensing contact identified on the official
model-license page. NVIDIA access requires personal acceptance of dataset and privacy terms.
Before acceptance, send `trustworthyaiprojects@nvidia.com` a written request confirming
whether commercial products and distribution of trained weights are permitted, how deletion
on termination applies to learned parameters, and whether any use would constitute prohibited
biometric processing. Do not treat the public code repository's Apache license as the license
for ASL 1000 media.

Any new human capture must undergo the applicable institutional human-subjects review.
[HHS informed-consent guidance](https://www.hhs.gov/ohrp/regulations-and-policy/guidance/faq/informed-consent/index.html)
requires prospective, legally effective consent unless an authorized exception applies,
and emphasizes understandable communication, voluntariness, questions, and withdrawal.
For ASL participants, accessibility cannot be reduced to an English-only signature form.
