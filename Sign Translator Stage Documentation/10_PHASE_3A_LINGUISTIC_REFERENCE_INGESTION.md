# 10 — Phase 3A Linguistic-Reference Ingestion

## 1. Verdict and boundary

Phase 3A has separate software, source-compatibility, and linguistic acceptance states as
of 2026-09-13:

| State | Verdict | Evidence |
|---|---:|---|
| Secure ingestion software | Implemented and software-verified | 27 focused adversarial tests and 1,640 repository tests pass under warnings-as-errors |
| Real Cokely source-compatibility pilot | Passed for one item | The authentic *I Have a Dream* EAF, HD video, SD alternate, and publisher page are locally hash-bound and fully inventoried |
| ASLLRP source-compatibility pilot | Not started | No authorized ASLLRP artifact is locally present |
| Project SIR mapping and linguistic validity | Not started by design | No qualified-human mapping or independent ASL validation has occurred |

This round accepts publisher-native linguistic reference material without converting it
into project truth. It does not train a model, create a gloss, map a source ID-gloss to
the Sign Intermediate Representation (SIR), label How2Sign, or authorize commercial use.
Software acceptance cannot substitute for a real-artifact pilot or qualified-ASL review.

## 2. Primary-source basis

- ELAN publishes the normative [EAF 3.0 XML Schema](https://www.mpi.nl/tools/elan/EAFv3.0.xsd).
  The implementation accepts only declared EAF version/format 3.0 and rejects older or
  unrecognized structure rather than claiming unverified cross-version compatibility.
- The [Dennis Cokely American Sign Language–English Interpreter Corpus](https://encompass.eku.edu/cokely_videos/)
  describes six EAF/MP4 collections, project-specific ID-gloss conventions, and a
  CC BY-NC-SA 4.0 license.
- A publisher [item record](https://encompass.eku.edu/cokely_videos/1/) lists separate EAF,
  HD-video, and SD-video downloads and describes the EAF as primarily containing ASL
  ID-gloss annotations. Direct automated file retrieval returned HTTP 403, but ordinary
  interaction with the publisher's public download page succeeded on 2026-09-13. The
  delivered files and the public item page were preserved without modifying source bytes.

The Cokely corpus is suitable for a bounded annotation-ingestion and review pilot. Its
six interpreted formal speeches cannot establish broad-domain English-to-ASL coverage,
canonical SIR, metric 3D motion, How2Sign labels, or commercially deployable rights.

## 3. Implemented ingestion contract

The canonical entry point is:

```text
python -m signtranslator.data_engineering.eaf EAF_PATH \
  --source-root READ_ONLY_SOURCE_ROOT \
  --license-evidence LICENSE_EVIDENCE_PATH \
  --auxiliary-evidence OPTIONAL_ADDITIONAL_SOURCE_ARTIFACT \
  --binding-config BINDING_CONFIG_JSON \
  --output NEW_INSPECTION_DIRECTORY
```

The symbolic path names above are operator-supplied paths, not defaults. The command
refuses an existing output directory. The binding JSON has exactly this shape:

```json
{"media_paths":{"EXACT_MEDIA_URL_FROM_THE_EAF":"LOCAL_MEDIA_PATH"}}
```

Every key must exactly equal one `MEDIA_DESCRIPTOR.MEDIA_URL`; every descriptor must have
one and only one binding. The local path may be relative to the declared source root or an
absolute path inside it. Empty paths, extra bindings, missing bindings, symlinks, path
escapes, non-regular files, duplicate JSON keys, non-finite JSON, oversized configuration,
unaccounted source-root files, and mutation during hashing or decoding fail closed. Optional
auxiliary evidence is repeatable and independently hash-bound; it cannot duplicate the EAF,
license evidence, or bound primary media.

The source root is read-only by contract. Inspection output must be outside that root.
The tool emits only compact metadata:

- `source-native-manifest.json`: canonical JSON containing the lossless semantic EAF tree,
  file identities, hashes, exact descriptor bindings, auxiliary-evidence identities, pinned
  EAF 3.0 schema identity, and implementation identity;
- `annotation-inspection.csv`: spreadsheet-injection-neutralized review rows;
- `annotation-inspection.html`: escaped text and URL-encoded local media links;
- `artifact-index.json`: byte sizes and SHA-256 digests for the three preceding files.

No media is copied and no normalized linguistic label is generated.

## 4. Source-native schema and invariants

The manifest preserves source element order, attribute order, tier order, annotation order,
text after XML entity decoding, time-slot identifiers and millisecond values, media
descriptors, participants, annotators, languages, controlled vocabularies, tier parents,
reference annotations, previous-annotation chains, lexicon references, external references,
and reference-link sets. The exact original EAF bytes remain authoritative and are bound by
SHA-256 plus file identity metadata.

Every manifest is marked:

```text
source_label_semantics = publisher_defined_source_annotation
project_mapping_status = unmapped_not_project_sir
training_target_authorized = false
linguistically_validated_by_project = false
```

The loader rejects any manifest that changes those claims. It also reconstructs and
revalidates the EAF graph, resource limits, interval structure, and descriptor-to-media
binding. A canonical serialization alone is insufficient to pass.

The parser rejects DTDs, entity declarations, XInclude/namespaces, processing instructions,
CDATA, comments, unknown elements or attributes, malformed XML, invalid Unicode controls,
non-NFC text, duplicate identifiers, dangling references, cyclic graphs, invalid controlled-
vocabulary references, overlapping known intervals within one tier, and known endpoints
outside the media range. Valid EAF 3.0 time slots without `TIME_VALUE` are preserved as
explicitly untimed; the implementation does not interpolate or invent their values. XML,
tree depth, element, attribute, text, tier, annotation, source-file, media, and auxiliary-file
counts are bounded before semantic use.

`TIME_ORIGIN` is treated as the signed integer specified by ELAN. Media timing comes from
decoded presentation timestamps and declared stream/container duration, not a guessed
nominal frame rate. File identity is checked after each independent decode operation.

## 5. Qualified-human mapping protocol

Source annotations and project SIR must remain separate immutable records. Mapping is a
future governed human decision, not a parser feature.

### 5.1 Prerequisites

Before mapping begins, freeze and hash:

1. the source-native manifest and exact media;
2. the project ASL annotation convention and SIR schema version;
3. the allowed lexical reference version, including explicit `UNKNOWN` and abstention;
4. annotator and reviewer role definitions, qualification evidence, independence evidence,
   accessible consent, compensation, retention, and permitted-use records;
5. the sampling plan, adjudication rule, agreement metrics, tolerances, and stop rules.

An annotator must be qualified to assess ASL linguistic form and meaning. A distinct
reviewer must independently inspect the video, source annotation, and proposed SIR. A
credential string or unaudited model output is not qualified review.

### 5.2 Per-record workflow

1. Display the hash-bound video and publisher-native tiers without pre-populating SIR.
2. The primary annotator creates a project-human SIR record from the observed signing and
   declared English context, recording unavailable channels and uncertainty explicitly.
3. The independent reviewer produces a blind decision before seeing adjudication notes:
   accept, reject, or revise, with phenomenon-level reasons.
4. Disagreements are adjudicated by a declared qualified third role or returned for
   reannotation. The system must not choose a value by majority when the linguistic issue
   is unresolved.
5. The accepted envelope binds the exact source manifest, video hash, transcript hash,
   convention hash, lexicon hash, annotator attestation, reviewer attestation, and final
   SIR content hash.
6. The publisher-native value remains unchanged. Any relationship to a project lexical
   item is stored as a separate reviewed mapping with provenance.

### 5.3 Agreement and timing diagnostics

For two independently marked intervals (A=[a_0,a_1]) and (B=[b_0,b_1]), report temporal
intersection-over-union only when both intervals are nonempty:

\[
\operatorname{tIoU}(A,B)=
\frac{\max(0,\min(a_1,b_1)-\max(a_0,b_0))}
{\max(a_1,b_1)-\min(a_0,b_0)}.
\]

Also report signed onset and offset differences, exact categorical agreement, per-field
confusion matrices, and chance-corrected agreement only where its assumptions and category
prevalence are reported. Missing and abstained judgments are separate outcomes, never
converted to agreement. Thresholds must be preregistered from a qualified-ASL pilot; this
document does not invent a passing tIoU, kappa, or timing tolerance.

Aggregate agreement cannot hide failures in handshape, palm orientation, movement,
location, non-manual scope, reference, classifier structure, gaze, head movement, or timing.
Each declared channel receives its own support count and failure count.

## 6. Real-artifact pilot procedure

The procedure below was executed for the first authorized EAF/video item and remains the
required procedure for every additional item:

1. preserve the publisher filenames and download receipt or page snapshot outside the
   repository in a new read-only source directory;
2. record the item URL, access date, citation, displayed license, downloaded byte size, and
   SHA-256 without interpreting the license as consent or commercial permission;
3. inspect the EAF `MEDIA_URL` values and create an exact binding JSON outside the source;
4. run ingestion into a new output directory and retain the printed manifest hash;
5. reload the manifest with `load_eaf_manifest`, independently recompute source hashes, and
   compare file counts and durations with the publisher item record;
6. inspect every tier in the small pilot, then sample the HTML links against the exact media;
7. document unsupported valid EAF constructs as parser compatibility failures; do not edit
   the source EAF to make it pass;
8. rerun the warning-strict focused and complete suites after recording the immutable pilot.

The first pilot fails if any source artifact is missing, mutable, ambiguously bound,
structurally unsupported, outside its declared authorization, or not visually aligned.
A failed record remains failed; it is not repaired by synthetic timing, inferred gloss,
English substitution, or a different video.

### 6.1 Executed Cokely pilot evidence

The read-only source directory is
`/Users/jiangshengbo/Volumes/cokely_reference/v1/source/i_have_a_dream`. Its complete
four-file inventory is:

| Role | File | Bytes | SHA-256 |
|---|---|---:|---|
| Source EAF | `2_I_Have_a_Dream_CokelyAFSParallelCorpus_v1-0.eaf` | 253,475 | `3c981ba4df1a2e590339a93dbf7aa2fb8163a87538d1e86be58f80e6a294285c` |
| Bound HD media | `2_I_Have_a_Dream_720_CokelyAFSParallelCorpus_v1_0.mp4` | 650,953,008 | `8eb18b6b2f01a5a179100b0acd84a639b5f2a0ee95fb0195d477b946217c6bb3` |
| Auxiliary SD media | `2_I_Have_a_Dream_480_CokelyAFSParallelCorpus_v1-0.mp4` | 271,251,189 | `9d3832ff7b4bac573b3e228f198bad1468c8b80206f7a7fe0c9b7d6bacc9a926` |
| Publisher item page | `publisher-item-page.html` | 40,580 | `2d232e16af21e8dc319f6908bdf23b29ba1fc7433136fafe0e3a6fbedc824e30` |

The EAF contains 798 time slots, of which six have no supplied `TIME_VALUE`; seven tiers;
and 751 annotations. The tier counts are 180 `ASL-TT`, 524 `ASL-individual-cp`, 12
`ASL-right-hand`, 34 `ASL-left-hand`, one `English-ST`, and zero each in `Comment` and
`Feedback`. Of the 751 annotations, 742 are fully aligned, six have one unknown endpoint,
and three have two unknown endpoints. These nine records remain explicitly unfully timed.

The bound HD stream decodes to 12,904 frames at 1,280 by 720 pixels, with first PTS 0 ms,
last PTS 430,530.1 ms, and declared stream end 430,563.4667 ms. Independent `ffprobe`
inspection agrees on the H.264 dimensions, duration, and frame count. The unmodified EAF
also validates against the pinned official EAF 3.0 XSD, whose SHA-256 is
`59e76f90d5840813314b1e635480d044287b811ebd63a40e25cb4e73c6ad72bb`.

The canonical schema-version-2 manifest is stored at
`/Users/jiangshengbo/Volumes/cokely_reference/v1/audit/i_have_a_dream_v2/source-native-manifest.json`
with SHA-256 `cd6f100c16befb8e3d0ba5c829c8c0f20fa074b58946099cf4e08b0dd9c3327a`.
An independent repeat ingestion produced byte-identical artifacts. The manifest explicitly
sets `training_target_authorized=false`, `linguistically_validated_by_project=false`, and
`project_mapping_status=unmapped_not_project_sir`.

## 7. Acceptance matrix and stop rules

| Gate | Current status | Required evidence |
|---|---:|---|
| Parser security and structural validation | Software pass | 27 focused adversarial tests; 1,640-test warning-strict suite |
| Canonical source-native round trip | Software pass | Byte-stable canonical manifest reload and deterministic repeat test |
| Immutable license/EAF/media binding | Real-pilot pass for one item | Complete four-file source inventory and canonical hash index |
| Cokely source-specific compatibility | Bounded pass for one item | Authentic EAF, bound HD MP4, auxiliary SD MP4, and publisher page |
| Cokely corpus compatibility | Unproved | Each additional item must independently pass the same procedure |
| ASLLRP source-specific compatibility | Not started | Authorized authentic EAF/media artifacts |
| Source ID-gloss to project SIR mapping | Not started by design | Frozen convention plus qualified independent human review |
| Linguistic validity | Unproved | Blinded qualified-ASL reference evaluation |
| How2Sign label transfer | Prohibited | Direct record-level evidence and separately approved protocol |
| Commercial use | Prohibited by current source terms | Written rights for the exact data and intended derivatives |

Stop the affected path immediately on representation mismatch, unexplained tier semantics,
unresolved signer/source leakage, hash drift, annotation/media misalignment, noncommercial-
rights conflict, absent qualified review, or a test failure. Phase 3A is complete only at
the secure-ingestion and one-item Cokely source-compatibility boundary. It does not establish
corpus-wide compatibility, project-label validity, ASL correctness, or readiness to train.
