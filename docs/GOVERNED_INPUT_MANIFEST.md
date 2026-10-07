# Governed planner input manifest, schema 1

This is a local handoff format for **already converted canonical motion and reviewed
SIR artifacts**. It calls the existing full-population admission checks. It does not
extract motion from video, create annotations, infer permission, verify reviewer
qualifications, or approve W1/W2/W3. All nine project manual handoffs remain absent
until the user supplies or identifies their evidence.

## Validate without training

```sh
.venv/bin/python -m signtranslator.data.governed_manifest inputs.json --sha256 EXPECTED_MANIFEST_SHA256
```

Supply the expected lowercase SHA-256 from the handoff record. The loader hashes the
same bounded byte read it parses. Matching a hash establishes byte identity, not
trusted authorship. The command prints JSON containing input/corpus/artifact hashes,
sample and split counts, and `phase_exit_approved: false`. Failure exits nonzero;
success performs admission only. It does not train or write files.

Admission examines the **entire submitted population**, including declared test
records, before exposing split views. This is different from training/diagnostic
execution, which leaves the admitted test view unopened. It establishes isolation
inside the submitted population, not proof that the submission is complete.

## File and JSON rules

- The manifest is a regular local file, at most 16 MiB, with schema version exactly
  integer `1` and scope exactly `research`. Boolean versions are rejected.
- Duplicate JSON object keys and nonfinite numbers (including overflowed exponents)
  are rejected. Objects below require their exact named fields; omitted fields do
  not silently acquire defaults.
- Every file reference is a normalized relative POSIX path beneath the manifest's
  directory. Empty, `.` or `..` segments, absolute paths, backslashes and symlinks
  within referenced paths are rejected. The working directory does not affect
  resolution. The manifest's parent directory is the explicit bundle root.
- Keep bundle files stable while admission/loading runs. These ordinary filesystem
  checks do not provide an atomic snapshot against concurrent path replacement.
- Files already checked by the canonical loaders retain their original hash,
  alignment, authorization, source provenance, and size checks. Lexicon/convention
  payload reads are bounded at 4 MiB plus a one-byte overflow check.

## Top-level structure

| Field | Required value |
|---|---|
| `schema_version` | Integer `1` |
| `scope` | String `research` |
| `sources` | List of source objects below; no duplicate source IDs |
| `authorizations` | Object keyed by source ID, with explicit evidence objects below |
| `records` | Nonempty list of complete motion record objects below |
| `lexicon` | `{ "artifact": GOVERNED_ARTIFACT, "path": RELATIVE_PATH }` |
| `convention` | `{ "artifact": GOVERNED_ARTIFACT, "path": RELATIVE_PATH }` |

`GOVERNED_ARTIFACT` has exactly `kind`, `artifact_id`, `version`, `sha256`, using the
existing `GovernedArtifact.to_dict()` representation. Lexicon kind is `sir_lexicon`;
convention kind is `asl_convention`. The exact payloads must match these identities
and every annotation's bindings. Lexicon entry order defines output classes; the
convention contains the existing explicit `locus_schema_version` and `loci` fields.
No alphabet or vocabulary is learned from test labels.

## Source and authorization objects

A source has exactly `source_id`, `title`, `official_url`, `access`, `capabilities`,
`research_rights`, `commercial_training_rights`, `commercial_deployment_rights`,
`limitation`. It uses the existing `SourceCandidate` validity rules.

`access` uses an `AccessStatus` member **name**: `LOCAL_VERIFIED`, `PUBLIC_DOWNLOAD`,
`ACCOUNT_REQUIRED`, `REQUEST_REQUIRED`, or `UNRELEASED`. Rights fields use
`PROHIBITED`, `UNRESOLVED`, `PERMISSION_REQUIRED`, or `PERMITTED`. Each capability is
`{ "capability": NAME, "level": LEVEL }`, where level is `PUBLISHER_CLAIM`,
`LOCAL_VERIFIED`, or `QUALIFIED_HUMAN`. These are supplied claims, never values the
loader fills in. Existing policy checks decide which claimed levels are sufficient.

Each authorization-map value has exactly:

| Field | Meaning |
|---|---|
| `source_id` | Must equal the map key and identify a declared source |
| `authorization` | Existing `DataAuthorization.to_manifest()` representation |
| `consent` | `GRANTED`, `WITHDRAWN`, or `NOT_DIRECTLY_VERIFIED` member name |
| `local_evidence` | Relative path to the evidence bytes bound by the authorization |

`DataAuthorization` retains its existing enum **values**, not the member-name
encoding above: `basis`, `license_identifier`, `license_url`, `licensor`,
`evidence_uri`, `evidence_sha256`, `permitted_uses`, `permitted_actions`,
`personality_rights`, `attribution_notice`, `limitations`. Do not rewrite rights,
actions, identities or evidence URIs to make a bundle pass. Missing grants and
withdrawn consent are rejected by admission; research admission grants no other use.

## Motion records and canonical samples

Each record has exactly `motion_path`, `motion_sha256`, `annotation`, `sample`,
`video_path`, `transcript_path`, `annotation_authorization_path`, `alignment_path`,
`alignment_sha256`, `source_files`.

The five `*_path` fields are relative file references. `source_files` maps source IDs
to relative native-source files. `motion_sha256` and `alignment_sha256` bind the
canonical motion archive and alignment document. `annotation` is the complete
existing `GovernedSIRAnnotation.to_manifest()` representation, including independent
review and byte-bound source evidence. Its binding checks are rerun, not bypassed.

The sample has **all** canonical `Sample` fields:

| Fields | JSON representation |
|---|---|
| `sample_id`, `source_id`, `signer_id_hash`, `target_language`, `license`, `intended_use`, `smplx_version`, `provenance`, `split` | Explicit nonempty strings without surrounding whitespace |
| `consent` | Consent member name as above |
| `authorization` | Complete `DataAuthorization.to_manifest()` object |
| `dialect`, `video_uri`, `audio_uri` | String or null |
| `calibration`, `frame_transform`, `time_transform` | Object or null |
| `transcript_lattice` | List or null |
| `semantic_plan` | Preserved JSON value; no semantic interpretation is inferred |
| `annotation_tiers` | Object mapping names to lists |
| `confidence_2d`, `confidence_3d`, `retention_date` | Finite number or null; booleans rejected; existing confidence range validation applies |
| `weak_gloss_candidates` | List of existing `WeakGlossCandidateRecord.to_dict()` objects, or empty list; never promoted to authentic annotation |

Existing corpus admission then checks consent/actions, file bytes, review bindings,
canonical channel layouts, clock alignment, sample/recording/signer isolation and
cross-split content identity. It does not establish real ASL quality, physical
calibration, statistical power, or the truth of supplied attestations.

## Connect to the existing governed runner

```python
from pathlib import Path
from signtranslator.data.governed_manifest import load_governed_planner_inputs
from signtranslator.governed_run import run_governed_planner

inputs = load_governed_planner_inputs(
    Path("inputs.json"), expected_sha256=expected_manifest_sha256,
)
result = run_governed_planner(
    inputs.corpus, inputs.vocabulary, inputs.alphabet,
    model_config=model_config, trainer_config=trainer_config,
    validation=True, shuffle=True,
)
```

`expected_manifest_sha256`, `model_config` and `trainer_config` must be explicitly
supplied; the model vocabulary/convention hashes must match the admitted inputs.
The [README](../README.md) documents the existing typed model/trainer configuration.
The [explicit training CLI](GOVERNED_TRAINING_RUN.md) accepts a complete run
configuration using this handoff. Native-source conversion adapters, real handoff
evidence and empirical phase acceptance are still outstanding. Loader success does not update the manual
intervention register or create an approval.
