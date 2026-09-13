# 11 — Phase 3B Governed Mapping and Adjudication

## 1. Verdict and evidence boundary

Phase 3B implements the software control plane for qualified-human mapping from an exact
publisher-native EAF annotation and its bound video to the project's Structured
Intermediate Representation (SIR). It does not perform that linguistic mapping itself.

The implementation has three deliberately separate states:

| State | Current verdict | Meaning |
|---|---:|---|
| Source-to-review binding software | Implemented and adversarially tested | Exact EAF annotation, source value, interval, schema, manifest, and media hashes can be bound to an immutable case |
| Review/adjudication workflow software | Implemented and adversarially tested | Primary, blind reviewer, adjudicator, terminal decision, and hash-linked history are represented and revalidated |
| Qualified-ASL empirical review | Not started | No project-human SIR, independent linguistic judgment, adjudication result, or human agreement estimate has been collected |

No current Cokely record is authorized here as a training target. No current record is
authorized for commercial use. A software-complete workflow is not a certificate of ASL
correctness. The fixed claims in every case, queue, and batch report remain:

```text
source_training_target_authorized = false
commercial_use_authorized = false
project_linguistic_validation_complete = false
```

The canonical implementation is in:

- `signtranslator/planning/adjudication.py` — source bindings, human submissions,
  agreement mathematics, workflow state, adjudication, canonical reload, and batch audit;
- `signtranslator/data_engineering/phase3b_queue.py` — governance-file verification,
  exact sampling-plan enforcement, label-empty draft queues, and bundle verification;
- `tests/test_phase3b_adjudication.py` — targeted mathematical, security, state-machine,
  source-integration, and queue tests.

## 2. Preconditions: seven real governance artifacts

A review queue cannot be created until one directory contains exactly these seven distinct,
non-symlink regular files:

1. ASL annotation convention;
2. project SIR lexicon;
3. primary annotation protocol;
4. blind independent-review protocol;
5. third-person adjudication protocol;
6. exact machine-readable sampling plan;
7. agreement-metrics preregistration.

Each file receives a typed `GovernedArtifact` identity consisting of its role, restricted
artifact identifier, version, and full SHA-256. Extra files, missing files, duplicate paths,
duplicate roles, path traversal, symlinks, and mutation during or after hashing fail closed.
The source directory, governance directory, and output directory must be disjoint.

The software verifies file existence and content identity. It cannot determine whether an
ASL convention is linguistically adequate or whether a credential document is truthful.
Those are human-governance findings and must be independently verified before collection.

## 3. Exact sampling-plan contract

The sampling-plan file is canonical JSON rather than an informal statement. Its exact
schema is:

```json
{
  "kind": "phase3b_exact_source_sampling_plan",
  "schema_version": 1,
  "selections": [
    {
      "media_descriptor_order": 0,
      "source_annotation_id": "EXACT_EAF_ANNOTATION_ID",
      "tier_id": "EXACT_EAF_TIER_ID"
    }
  ],
  "source_manifest_sha256": "64-lowercase-hex-characters"
}
```

Selections must be nonempty, unique, bounded, and canonically sorted by tier, source
annotation identifier, and media-descriptor order. Queue arguments must equal the frozen
plan exactly. The plan must bind the exact source-native manifest. The system does not
choose records, optimize a sample, infer a tier, or replace a missing annotation.

## 4. Source binding and time semantics

`load_eaf_source_catalog` first reloads the complete Phase 3A manifest, revalidates the EAF
graph, inventories the current source root, rejects unexpected artifacts and symlinks, and
rehashes every source role. SHA-256 and byte size are the portable content gate. Device,
inode, and modification-time drift is reported separately because a mounted volume may
change storage identity without changing content.

Each `EAFAnnotationBinding` commits to:

- EAF manifest, EAF file, EAF 3.0 schema, and selected video SHA-256;
- media-descriptor order;
- exact tier, source annotation ID, annotation kind, and source order;
- exact publisher value and its SHA-256;
- original begin/end time-slot references;
- known begin/end millisecond values, or explicit `null` values.

Phase 3B SIR timing is not unitless. Every human submission and adjudication declares:

```text
sir_time_unit = milliseconds
sir_time_origin = eaf_time_order
```

For a source interval \([s_0,s_1)\), every submitted event interval
\([t_i^s,t_i^e)\) must satisfy

\[
s_0 \le t_i^s < t_i^e \le s_1.
\]

If either source endpoint is unavailable, the record cannot contain a timed SIR. The honest
path is abstention, normally with `source_misalignment` or
`insufficient_visual_evidence`, followed by a separately governed realignment procedure.
No timestamp interpolation, frame-index conversion, or guessed duration is permitted.

## 5. Immutable review state machine

The allowed workflow is:

```text
draft
  -> primary_complete
  -> blind_reviewed
  -> adjudicated       (required for every non-identical outcome except joint abstention)
  -> accepted | rejected | abstained
```

The draft event hashes the complete frozen case specification: source binding, convention,
lexicon, convention-to-lexicon binding, all three role protocols, sampling plan,
preregistration, and case ID. Substituting any governance artifact therefore invalidates
the event history.

Every later event contains a monotonic sequence number, case ID, typed state, actor,
artifact hash, timestamp with timezone, and the previous event's SHA-256. Reloading a case
reconstructs the expected event sequence and rejects omissions, reordering, replay across
cases, altered actors, altered timestamps, stale hashes, and broken predecessor links.
The blind-review event hashes both the exact reviewer submission and the canonical event
correspondence, because changing correspondence changes metric support and edge mapping.

### 5.1 Primary and blind reviewer

Each submission binds the exact case, source, source interval, timebase, protocol, canonical
SIR bytes, SIR hash, author pseudonym, qualification-evidence hash,
independence-evidence hash, attestation hash, and submission time. Primary and reviewer
identifiers and pseudonyms must be distinct. Both roles attest to qualified ASL competence,
direct video review, and independent authorship. These attestations are necessary evidence,
not proof that the human claims are true.

A SIR submission must contain at least one manual event and no abstention reason. An
abstention contains no SIR and must use one or more declared reason codes. English text,
publisher ID-gloss strings, `gloss_tokens`, and unknown JSON fields are rejected rather
than silently copied into SIR.

### 5.2 Third-person adjudication

Byte-identical independent SIRs may be accepted without adjudication. Two abstentions may
terminate as abstained. Every other outcome requires a qualified third adjudicator whose
pseudonym differs from both submitters and whose record binds both exact submission hashes.

Adjudication may accept the primary SIR, accept the reviewer SIR, produce a distinct revised
SIR, reject the record, or abstain. The accepted-primary and accepted-reviewer paths must
carry the corresponding exact SIR hash. A revised result must differ from both. Rejected or
abstained adjudication cannot carry a SIR. Reusing a submission identifier as an
adjudication identifier fails closed.

## 6. Agreement mathematics

No automatic graph matching is performed. Non-identical SIRs require a reviewer-declared
one-to-one event correspondence in canonical primary-event-ID order. Identical canonical
SIRs use their complete identity
correspondence; an explicit partial correspondence is rejected because it could suppress
disagreement support. Abstained submissions cannot declare event pairs.

For paired positive-duration intervals \(A=[a_0,a_1)\) and \(B=[b_0,b_1)\):

\[
\operatorname{tIoU}(A,B)=
\frac{\max(0,\min(a_1,b_1)-\max(a_0,b_0))}
{\max(a_1,b_1)-\min(a_0,b_0)}.
\]

The implementation also reports signed reviewer-minus-primary boundary differences:

\[
\Delta_{on}=b_0-a_0, \qquad \Delta_{off}=b_1-a_1.
\]

Tests establish symmetry, the \([0,1]\) range, translation invariance, and positive-scale
invariance over 2,000 deterministic random interval pairs, in addition to exact overlap,
containment, equality, and disjoint cases. NaN, infinity, zero duration, and reversed
intervals are rejected.

For each paired categorical field \(k\in\{kind,label,referent,locus\}\):

\[
A_k=\sum_i \mathbf 1[x_{ik}=y_{ik}],\qquad
r_k=\begin{cases}A_k/n_k,&n_k>0\\\text{unavailable},&n_k=0.\end{cases}
\]

Exact numerator, support, rate, and kind/label confusion counts are retained. Event-pair
coverage is reported separately for primary and reviewer inventories:

\[
c_P=n_{pair}/|V_P|,\qquad c_R=n_{pair}/|V_R|,
\]

and is unavailable when its denominator is zero. This prevents a high field agreement rate
on a small matched subset from hiding many unpaired events.

Only edges whose endpoints are paired are comparable. After mapping reviewer endpoints to
primary identifiers, the diagnostic is

\[
J_E=|E_P\cap E_R|/|E_P\cup E_R|.
\]

`J_E` is unavailable—not one—when the union is empty. Primary/reviewer total edge counts and
the comparable intersection and union are reported alongside it.

## 7. Batch accounting

`audit_phase3b_batch` requires an exact expected-case index. It rejects duplicate case IDs
and duplicate source annotations, detects missing/unexpected/nonterminal cases, and reports
mixed conventions, lexicons, protocols, sampling plans, or preregistrations. Results are
independent of input iteration order. The statistical source unit is identified by EAF-file
content hash, tier, and annotation ID; changing a manifest or camera view cannot make the
same linguistic annotation count as independent evidence. Queue construction rejects such
cross-view duplicates before publication.

Aggregate field counts, confusion matrices, temporal medians, and edge Jaccard are pooled
from their underlying supported events and edges. The implementation does not average
per-case rates or medians, which would give small and large cases equal statistical weight.
Unreviewed and abstained cases are counted but do not fabricate metric support.

The canonical batch report carries exact expected case IDs and case-content hashes. It can
be reloaded and independently recomputed from the case snapshots. A
`software_workflow_complete=true` finding means only that all expected cases are present,
terminal, consistently governed, and structurally valid. The batch report contains no
selected threshold, no “best” metric, and no training or validation approval.

## 8. Label-empty review queue

`write_phase3b_review_queue` emits one atomic four-file bundle:

- `phase3b-draft-cases.jsonl` — canonical draft cases with `primary=null` and
  `reviewer=null`;
- `phase3b-review-queue.html` — escaped publisher values and URL-encoded local media links;
- `queue-manifest.json` — source, governance, selection, case, implementation, and fixed
  non-approval identities;
- `artifact-index.json` — size and SHA-256 for every payload file.

The source and governance directories remain read-only. Media is linked, not copied. The
writer verifies the complete temporary bundle against current source and governance bytes
before atomically renaming it into place. Existing or symlinked output is refused. The writer
returns a receipt containing the absolute queue path, case count, and queue-manifest
SHA-256. That digest must be stored in a separate trusted record; internal hashes alone
cannot detect a wholesale self-consistent rewrite.

`verify_phase3b_review_queue` requires the exact four-file inventory, canonical JSON,
complete hash-index coverage, current source-manifest equality, current governance hashes,
canonical sampling order, label-empty draft state, case-index equality, and exact
reconstruction of every draft and HTML byte from the selected source evidence. Verification
requires the trusted receipt digest. HTML and URL escaping are tested with an injected
`<script>` source value and a media filename containing a space; changing HTML and
recomputing the internal artifact index still fails semantic reconstruction.
Manifest, index, sampling-plan, case, aggregate JSONL, and HTML byte limits are enforced
before bulk reads. Payload hashes stream with pre/post identity checks, and case JSONL is
loaded one bounded record at a time rather than as one corpus-sized allocation.

## 9. Threats and fail-closed controls

| Threat | Control |
|---|---|
| English or source label copied into project output | No conversion function; draft SIR fields are empty; unknown `gloss_tokens` rejected |
| Source or media substitution | Full SHA-256/size binding plus current complete-inventory revalidation |
| Governance substitution | Seven exact content hashes committed by the draft event and queue manifest |
| Sampling-plan drift | Canonical plan binds source-manifest hash and exact queue selections |
| Reviewer copies or self-reviews | Distinct roles/pseudonyms/IDs plus hashed independence and attestation evidence; real independence still requires governance audit |
| Replay across cases | Case ID in submissions, adjudication, and every hash-linked event |
| Partial matching hides errors | Canonically ordered one-to-one correspondence; correspondence bound into review history; explicit correspondence forbidden for identical graphs; pair coverage reported |
| Same annotation counted through multiple views or manifests | Statistical identity uses EAF-file hash, tier, and annotation ID; queue and batch reject duplicates |
| Untimed or out-of-range SIR | Fixed EAF-millisecond timebase and source-interval containment |
| Undefined statistic presented as success | Zero support produces `null`; no fake perfect score |
| Aggregate hides field failure | Per-field numerators/support/confusions plus unpaired-event and edge counts |
| HTML/URI injection | Unicode controls rejected where identifiers are accepted; text escaped; local URIs encoded |
| JSON ambiguity | Duplicate keys, non-finite values, unknown fields, noncanonical bytes, and size excess rejected |
| Oversized-file denial of service | Actual sizes checked before hashing; bounded reads and line-wise JSONL parsing |
| Output overwrite or partial publication | Existing output refused; temporary bundle verified before atomic rename |
| Whole-bundle or reindexed HTML rewrite | Out-of-band manifest receipt plus deterministic case and HTML reconstruction |

## 10. Executed verification and remaining empirical work

The focused Phase 3B test file exercises source inventory and mutation, storage
identity drift, unknown source timing, exact and property-style temporal mathematics,
categorical/edge agreement, zero-denominator behavior, abstention, all adjudication paths,
self-review and replay, chronological and hash-chain tampering, canonical JSON attacks,
governance substitution, batch reconciliation, label-empty queue creation, HTML/URI
injection, artifact mutation, governance mutation, and sampling-plan mismatch.

The final software verification on 2026-09-13 produced:

- **35/35** tests passing in the targeted Phase 3B adversarial file;
- **90/90** tests passing across Phase 3B, Phase 3 governance, and EAF ingestion;
- **1,675/1,675** repository tests passing with Python warnings treated as errors;
- successful bytecode compilation of the package and the targeted test file;
- a successful wheel build containing both Phase 3B production modules; and
- a fresh real Cokely catalog reload reproducing manifest SHA-256
  `cd6f100c16befb8e3d0ba5c829c8c0f20fa074b58946099cf4e08b0dd9c3327a`, all 751
  annotations, representative fully timed binding SHA-256
  `b65a58183b991d0c4383ad0a7fd17d3b23842442c2371cea11e6169077f78b1e`, and the
  expected partial/untimed records. All four source files retain exact size and SHA-256 but
  report storage-identity drift after remount; this is disclosed rather than promoted to
  physical-file identity equality.

These results accept the Phase 3B software control plane against its declared contract. They
do not accept the absent human evidence, the resulting linguistic mappings, training use,
commercial use, or industrial deployment.

The empirical work remains blocked until real, independently verified artifacts exist:

1. qualified-ASL convention and project SIR lexicon;
2. role, qualification, independence, consent, compensation, retention, and permitted-use
   records for primary annotators, blind reviewers, and adjudicators;
3. a qualified sampling decision and preregistered diagnostic/stop rules;
4. primary and blind project-human SIR judgments from the bound video;
5. third-person adjudication for every non-identical result;
6. independent qualified-ASL analysis of the resulting cases;
7. separate written authorization before any accepted mapping is used for training or a
   commercial product.

No current test can substitute for these observations. Phase 3B may be software-ready while
remaining empirically unapproved; Phase 2, model training, and industrial deployment remain
closed until their separate data, representation, rights, and human-validation gates pass.
