# 10 — Dataset and Data Engineering — Design and Mathematics

## Active real-data bridge

`signtranslator.data_engineering.exporter` is the canonical governed-record to
training-shard boundary. It accepts timestamped holistic landmark tracks only after
authorization basis and evidence, consent status, provenance, media hash, language,
extractor version, coordinate system,
source tokens, gloss tokens, confidence, and validity masks are present. It then performs
grouped splitting before batching, fits normalization on valid training observations
only, verifies exact CTC feasibility (including adjacent repeated labels and acoustic
subsampling), writes v2 shards plus SHA-256 hashes, and emits a human-review queue.

Source video is decoded through pinned PyAV using each frame's container presentation
timestamp (PTS); nominal FPS is never substituted for a missing clock. The executable
`assess_stage_b_corpus` exit gate verifies that exported landmark timestamps occur on the
decoded source clock, source and authorization-evidence bytes match their recorded
SHA-256 values, action-scoped governed usage fields and provenance roots are present, and
batch sample IDs trace back to manifest records.

The active `SignDataset` and `collate_corpus` retain variable motion/speech lengths,
timestamps, frame masks, confidence, and validity. Missing observations are padded with
zero only while their masks remain false and confidence remains zero; padding is never
declared observed data.

The bridge does not perform raw-video landmark estimation. `decode_landmark_npz` reads a
strict output interchange from a separately licensed, versioned extractor, and
`assemble_holistic_track` combines body, dense-hand, and face tracks only when their
frame clocks match exactly. A raw-video extractor and a licensed real mini-corpus remain
external Stage B gate requirements; their absence must not be represented as a passed
real-data stage.

The splitter uses connected components of the signer/source bipartite graph. Grouping only
the pair `(signer, source)` is insufficient because the same signer can occur in multiple
recordings and a multi-signer source can connect several identities. Component assignment
guarantees independently that neither signer nor source crosses a split.

Weighted normalization uses a two-pass population-variance calculation. This avoids the
catastrophic cancellation in `E[x²] - E[x]²` when a coordinate frame has a large offset.
Weighted DLT rejects fewer than two positive-confidence views, invalid confidences,
rank-deficient camera geometry, points at infinity, and failed cheirality rather than
returning an unstable 3D point.

`assess_stage_b_corpus` approves progression only when three external, versioned artifacts
are colocated with the corpus: `dataset_charter.json`, `annotation_agreement.json`, and
`review_attestation.json`. The review attestation must bind the exact manifest hash and
record qualified target-language signer review of source video, extracted landmarks, and
exported shards. `review.html` alone remains only a queue, never proof of review.

This document fixes **all** mathematics and contracts of the dataset / data-
engineering layer before any code, in the discipline of docs 01–09. It implements
`10_dataset_and_data_engineering.md`: a canonical sample schema, an evidence-backed
authorization-gated pipeline, quality mathematics (weighted robust reprojection, multi-view
triangulation, deduplication, per-tier agreement), leakage-free splitting, and
governance. This is a **data-infrastructure** layer, not a neural model.

**Reuse.** Doc-04 already has the robust reprojection primitives
(`PerspectiveCamera.project`, `geman_mcclure`, `reprojection_loss`) — exactly the
document's `e = c·ρ(‖Π(J)−k‖)`. Doc-03 has `cohens_kappa` / `fleiss_kappa` for
inter-annotator agreement. `data/quality.py` has MAD robust stats and cleaning;
`data/readiness.py` has split-leakage detection. This layer builds on those.

Primary sources studied:

* Duarte et al., *How2Sign* (arXiv:2008.08143) — 80+ h continuous multimodal ASL,
  speech/text/depth, a Panoptic-studio 3D subset (interpreted, instructional).
* Li et al., *WLASL* (arXiv:1910.11006) and Vaezi Joze & Koller, *MS-ASL*
  (arXiv:1812.01053) — isolated lexical recognition / visual pretraining.
* Yu et al., *SignAvatars* (arXiv:2310.20436) and *SignAvatar / ASL3DWord*
  (arXiv:2405.07974) — fitted holistic 3D motion (auto-reconstructed, model bias).
* Gebru et al., *Datasheets for Datasets* (arXiv:1803.09010) — the datasheet
  sections (motivation, composition, collection, preprocessing, uses, distribution,
  maintenance).

## 0. Honest scope (read first)

The local development environment now contains 31,047 How2Sign frontal-view
sentence clips, matching OpenPose JSON directories, rendered review videos, and a
31,165-row realignment table. `data_engineering/how2sign.py` inventories those
artifacts, joins only exact sentence names, preserves container presentation
timestamps, and emits explicit 2D BODY_25 + hands + face tracks. It never promotes
2D observations to fake 3D, clips invalid confidences, invents missing clips, or
uses English translation tokens as ASL gloss.

This is not yet a training-ready How2Sign corpus. The local download lacks 118
metadata-listed clips, contains one unjoinable raw-video artifact, and does not
contain independently identified gloss annotations or a local immutable license
evidence file. The exporter therefore remains fail-closed. The schema, quality
mathematics, deduplication, splitting, governance, and datasheet contracts continue
to be validated on controllable synthetic records, while the real-media adapter is
tested separately against the downloaded source structure.

## 1. Canonical sample schema

A `Sample` is a typed record with the document's fields: `sample_id`, `source_id`,
`signer_id_hash`, `target_language`, `dialect`, `license`/`consent`, explicit
`authorization`, video/audio
URIs, `calibration`, `transcript_lattice`, `semantic_plan`, `annotation_tiers`,
`2d`/`3d confidence`, `smplx_version`, `frame/time transforms`, `provenance`,
`split`. `validate_sample` rejects a record missing a license, explicit authorization, a
signer-id hash, a split, or provenance — the governance-critical fields are never
optional. A `DatasetMap` registry records each source corpus's best use and
material limitation (How2Sign, WLASL, MS-ASL, ASLLVD, PHOENIX14T, SignAvatars,
ASL3DWord).

## 2. Evidence-backed authorization gate + provenance (innovation)

The pipeline **gates before download**: `gate_download(authorization, consent,
intended_use, requested_actions)` returns false unless immutable evidence supports
the exact use and every requested action. Direct participant consent requires
`GRANTED`; a published secondary-dataset license requires
`NOT_DIRECTLY_VERIFIED`, attribution, and explicit limitations when personality
rights are unverified. A license name alone grants nothing in code.
Originals are content-hashed (SHA-256) and the hash is immutable. **Innovation —
Merkle-style provenance chain:** each preprocessing step's output is hashed together
with the previous step's hash, `h_i = H(h_{i-1} ‖ step_i ‖ output_i)`, so the final
root certifies the *exact* sequence; any tampering or drift changes the root
(proved), giving a reproduction certificate.

## 3. Triangulation + weighted reprojection (quality math)

**Multi-view DLT triangulation.** For a 3D point `X` (homogeneous), camera
projection matrices `P_i` (3×4) and 2D observations `(u_i, v_i)`, each view gives

    u_i (p_i^3ᵀ X) − (p_i^1ᵀ X) = 0,    v_i (p_i^3ᵀ X) − (p_i^2ᵀ X) = 0,

(`p_i^j` = row `j` of `P_i`). Stacking gives `A X = 0`; `X` is the right singular
vector of `A` with the smallest singular value (SVD), then de-homogenised.
Confidence-weighting scales each view's rows by `√c_i`. Proved: DLT recovers a
known 3D point exactly from ≥2 noise-free views.

**Weighted robust reprojection residual** (the document's equation, = Doc-04):

    e_{t,j} = c_{t,j} · ρ(‖Π(J_{t,j}) − k_{t,j}‖₂),   ρ = Geman-McClure,

with the confidence `c` retained downstream (zero confidence ⇒ zero contribution).
**Innovation — confidence-propagated triangulation:** a scalar 3D confidence that
increases with the observations' confidences and decreases with the reprojection
residual (proved monotone), so uncertainty is carried, not discarded.

## 4. Deduplication

* **Perceptual hashing** — average hash (`aHash`, 8×8 mean threshold) and difference
  hash (`dHash`, adjacent-pixel comparison) give 64-bit codes; the **Hamming
  distance** between codes is the perceptual distance (0 for identical, small for
  near-duplicates).
* **Transcript similarity** — Jaccard over token n-grams and normalised edit
  distance.
* **Near-threshold clustering** — pairs whose distance is within `[τ−δ, τ+δ]` are
  flagged for **manual inspection** rather than auto-decided (per the document).

Proved: identical inputs → distance 0; a small perturbation → small distance; the
similarity metrics satisfy their bounds; duplicates and near-threshold clusters are
detected.

## 5. Per-tier agreement + QC stratified sampling

Inter-annotator agreement (`cohens_kappa` / `fleiss_kappa`, Doc-03) is estimated
**per annotation tier** (gloss, non-manual, discourse), **not** as one corpus-wide
number — a corpus average hides a tier where annotators disagree (demonstrated). QC
sampling is **stratified** by signer / skin-tone / lighting / motion-complexity so
every stratum is represented (coverage proved).

## 6. Leakage-certified grouped split (innovation)

Samples are partitioned by connected components of the signer/source bipartite graph,
not by pair keys. **Certificate:** neither an individual signer nor an individual source
spans two splits, including transitive connections. Windows and augmentations **inherit**
their sample's component, so windowing/augmentation after the split cannot leak.

## 7. Governance

* **Consent state machine** `GRANTED → WITHDRAWN`; `apply_withdrawal` removes every
  record of a withdrawn signer (proved: none remain). `apply_retention` removes
  records past their retention date.
* **Policy gates** for derivative-model, face/identity, commercial-use, and
  redistribution — each checked before the corresponding action.
* **Innovation — sensitive-trait non-inference guard.** The `Sample` record has no
  sensitive-trait field, and `infer_sensitive_trait` *raises* — so a sensitive-trait
  prediction cannot be computed from the record; "do not infer sensitive traits" is
  made unbreakable in code, not merely documented.

## 8. Datasheets + preprocessing manifest

A `Datasheet` (Gebru et al. sections: motivation, composition, collection,
preprocessing, uses, distribution, maintenance) with completeness validation, and an
exact **preprocessing manifest** carrying the §2 Merkle provenance root. Deaf
annotators are credited in the maintenance / error-taxonomy sections (per the
document's governance requirement).

## 9. Integration + innovations

The schema carries the Doc-04 SMPL-X version and Doc-02 semantic plan; the quality
math reuses Doc-04 reprojection; agreement reuses Doc-03 kappa; the split extends
`data/readiness.py`. Innovations: the Merkle provenance chain, the leakage-certified
grouped split with window/augmentation inheritance, confidence-propagated
triangulation, and the sensitive-trait non-inference guard.

## 10. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| 10.0 | research + design/math spec (this doc) | done |
| 10a | canonical sample schema + dataset map | done (12 tests) |
| 10b | action-scoped authorization gate + provenance hashing | done |
| 10c | triangulation + weighted reprojection quality | done (8 tests) |
| 10d | deduplication (perceptual hash + transcript) | done (9 tests) |
| 10e | per-tier agreement + QC stratified sampling | done (5 tests) |
| 10f | leakage-certified grouped split | done (6 tests) |
| 10g | governance + sensitive-trait non-inference | done (8 tests) |
| 10h | datasheets + manifest + integration + regression | done (5 tests) |

Data-engineering layer: 63 tests, green on two consecutive runs under
`-W error::UserWarning`; whole project green (all 112 test files).

## 11. Findings (post-implementation)

**The schema makes governance-critical fields non-optional.** `validate_sample`
rejects a record missing a license, consent, signer-id hash, intended use,
provenance, or a valid split; confidence fields are range-checked to `[0,1]`. The
record deliberately has **no sensitive-trait field** — the non-inference guard
begins structurally, at the type, not merely at a policy note.

**The gate precedes acquisition and provenance is a hash chain.** `gate_download`
permits acquisition only when evidence supports the exact use and requested actions.
Direct consent and a published dataset license are separate authorization bases; the
latter cannot be mislabeled as consent granted to this project. The Merkle-style chain
`h_i = H(h_{i-1} ‖ step ‖ output)` advances a
root per preprocessing step; tampering with any step's output, or reordering steps,
changes the root (both proved), so the manifest is a reproduction certificate.

**Triangulation is exact and reuses the audited Doc-04 reprojection.** The DLT
stacks `u(p3·X)−(p1·X)=0`, `v(p3·X)−(p2·X)=0` per view (confidence-weighted by `√c`)
and solves `AX=0` by SVD; it recovers a known 3D point to `1e-8` from ≥2 noise-free
views. The document's `e = c·ρ(‖Π(J)−k‖)` is the Doc-04 `reprojection_loss`
(Geman-McClure) — so `e < c` for any outlier and zero confidence contributes zero.
The confidence-propagated 3D confidence is provably monotone (↑ in each `c_i`, ↓ in
the residual).

**Deduplication distinguishes real matches from ambiguous ones.** aHash/dHash give
64-bit codes; identical frames have Hamming distance 0, a `1e-3` perturbation stays
within 2 bits, independent content exceeds 10 bits. Transcript Jaccard (bigrams
strictly catch reordering unigrams miss) and normalised edit distance are bounded in
`[0,1]`. Near-threshold pairs are **flagged for manual inspection**, not
auto-decided — a subtlety the document is explicit about.

**Agreement is per-tier by design.** A perfect gloss tier and an anti-correlated
discourse tier are reported separately; the single pooled kappa sits between them
and hides the weak tier (demonstrated). QC sampling is stratified so every populated
stratum — including a signer with a single clip — is represented.

**The split certifies leakage-freedom, not just detects it.** Connected-component
partitioning guarantees no signer or recording spans two splits;
`certify_no_group_leakage` reports offending signers and sources independently when a
sample is hand-moved. Windows and augmentations inherit their sample's split. This is
stronger than the post-hoc byte-identical detector in `data/readiness.py`, which it
complements.

**Governance is enforced in code.** Consent is `GRANTED → WITHDRAWN` (terminal;
re-granting raises); withdrawal removes every record of a signer; retention drops
expired records. Policy gates default closed. `infer_sensitive_trait` **always
raises** — there is no code path returning a sensitive-trait prediction, so the
"do-not-infer" policy cannot be silently bypassed.

**Honest scope holds.** No real corpora are downloaded — they are licensed and the
pipeline gates before download — so every property is proved on controllable
synthetic records/observations; real corpora drop in behind the same schema.
Innovations delivered: the Merkle provenance chain, the leakage-certified grouped
split with window/augmentation inheritance, confidence-propagated triangulation, and
the sensitive-trait non-inference structural guard.
