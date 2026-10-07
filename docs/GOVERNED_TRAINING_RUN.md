# Explicit governed training runs, schema 1

The command consumes a byte-bound run configuration and the
[canonical-artifact input handoff](GOVERNED_INPUT_MANIFEST.md). It runs the existing
five-head planner and preserves its admission, support, identity and continuation
checks. It does not use the legacy synthetic entry point or fabricate missing data.
All nine manual handoffs remain absent until supplied; successful execution is not
empirical W1/W2/W3 acceptance.

```sh
.venv/bin/python -m signtranslator.governed_train run.json --sha256 EXPECTED_RUN_SHA256 --max-epochs 1
```

The JSON configuration must be a nonsymlink regular file of at most one MiB. Its
expected SHA-256 must match the exact bytes. Duplicate keys, nonfinite numbers,
unknown/missing fields and boolean schema versions are rejected. No mathematical
configuration fields silently acquire defaults.

## Complete run document

| Field | Required representation |
|---|---|
| `schema_version` | Integer `1` |
| `input_manifest` | Normalized relative POSIX file path, resolved against the run configuration directory |
| `input_manifest_sha256` | Expected exact-byte lowercase SHA-256 for the input manifest |
| `model_config` | Complete `dataclasses.asdict(LocusTextConfig(...))` object |
| `trainer_config` | Complete `TrainerConfig(...).to_dict()` envelope and all its values |
| `validation` | Explicit boolean |
| `shuffle` | Explicit boolean |

The model object contains exactly: `embedding_dim`, `hidden_dim`, `max_bytes`,
`max_events`, `lexicon_sha256`, `convention_sha256`, `declared_encoding`,
`timing_scale_seconds`, `relation_supervision`, `objective_normalization`,
`referent_supervision`, `locus_count`, `locus_supervision`. Existing typed model
validation and exact vocabulary/convention binding apply.

The trainer envelope contains exactly `config_type: "TrainerConfig"`,
`schema_version: 1`, and `values`. Its values contain exactly: `epochs`, `batch_size`,
`lr`, `weight_decay`, `warmup_frac`, `min_lr_frac`, `grad_clip`, `loss_weights`,
`selection_metric`, `val_every`, `seed`, `device`, `ckpt_path`.

The five positive loss weights must explicitly name `sir_sequence`, `event_timing`,
`sir_relations`, `referent_equality`, `locus_assignment`. The selection metric must
name a supported branch. The seed is an exact integer in `[0, 2**32)`, and device is
an explicit string. CPU exact continuation is verified; accelerator continuation
remains unqualified. The full intended epoch horizon belongs in `epochs`; do not
change it merely to request a short initial run.

## Outputs and continuation

`ckpt_path` is required and must be a normalized relative POSIX prefix such as
`outputs/run.pt`. It resolves against the run configuration directory, not the
shell working directory. The trainer derives `outputs/run.best.pt`,
`outputs/run.last.pt` and their JSON sidecars. Best output is written only when a
validation selection occurs and improves. Returned `checkpoint_paths` identifies
configured destinations; it is not a claim that every destination was written in
this invocation.

Existing destination files are refused unless `--allow-checkpoint-replacement` is
specified. Even with that flag, outputs cannot overlap the configuration file or
any declared manifest input file, and output paths may not traverse symlinks.
Existing nondirectory parents and nonregular destinations are rejected. These
checks are preflight checks, not protection against concurrent filesystem changes.

To continue a prior committed run in the same destinations:

```sh
.venv/bin/python -m signtranslator.governed_train run.json --sha256 EXPECTED_RUN_SHA256 --resume-from outputs/run.last.pt --allow-checkpoint-replacement
```

The resume path is relative to the run configuration directory. To preserve prior
outputs, use a new checkpoint prefix and the new configuration's exact hash instead
of allowing replacement. Existing trainer rules permit changing storage location,
while optimization settings, data, model, runtime and implementation identities
must still match. `--max-epochs N` limits this invocation to N additional epochs,
up to the original total horizon. It does not redefine that horizon or permit
mid-epoch continuation.

On success stdout is one canonical JSON summary containing configuration/input/
corpus identities, current model-state hash, completed epoch/step counts, configured
checkpoint paths, history, exposure summary, fresh declaration audit and
`phase_exit_approved: false`. Redirect stdout explicitly if a saved summary is wanted.
Only checkpoint outputs are written automatically. A later failure does not roll
back already-written checkpoints; checkpoint and sidecar replacement is per file.

## Programmatic construction

Given explicitly selected typed configurations and an existing admitted-input
manifest, construct the document without omitting fields:

```python
from dataclasses import asdict
from pathlib import Path
from signtranslator.reproducibility import canonical_json_bytes, sha256_file
from signtranslator.governed_train import run_from_configuration

# model_config, trainer_config and input_manifest_sha256 are caller-supplied.
trainer_config.ckpt_path = "outputs/run.pt"
run_document = {
    "schema_version": 1,
    "input_manifest": "inputs.json",
    "input_manifest_sha256": input_manifest_sha256,
    "model_config": asdict(model_config),
    "trainer_config": trainer_config.to_dict(),
    "validation": True,
    "shuffle": True,
}
path = Path("run.json")
path.write_bytes(canonical_json_bytes(run_document))
run, summary_bytes = run_from_configuration(
    path, expected_sha256=sha256_file(path), max_epochs=1,
)
```

The example demonstrates serialization of chosen settings, not recommended
hyperparameters or permission to use a dataset. Native-source conversion, authentic
review/authorization handoffs, real-data pilots and empirical phase exits remain
separate work.


## Load a checkpoint for development diagnostics

`run_governed_planner(resume_from=...)` continues training. To inspect a partial
or completed checkpoint at exactly its saved boundary, use the load-only API:

```python
from signtranslator.governed_run import load_governed_planner, diagnose_governed_planner

saved = load_governed_planner(
    inputs.corpus, inputs.vocabulary, inputs.alphabet,
    model_config=model_config, trainer_config=trainer_config,
    validation=True, shuffle=True, checkpoint_path="outputs/run.last.pt",
)
diagnostic = diagnose_governed_planner(
    saved, view="train", model_state="current",
    sample_indices=(0, 1), permutation=(1, 0), seed=9, max_samples=2,
)
```

The example requires at least two admitted training samples and the original
validation/shuffle choices. Supply the original full training horizon and settings;
do not change `epochs` to the saved cursor. `best_validation` explicitly selects
the checkpoint's retained validation-selected weights when available.

Loading shares training's preflight, exact checkpoint/sidecar verification,
configuration/runtime/implementation/data bindings and fresh target-declaration
audit. It restores optimizer, scheduler, history, exposure and selected weights
without calling `fit` or `save`; no checkpoint or other file is written. Failed
source, configuration or checkpoint validation returns no run. It opens only the
configured train/validation views of an already admitted corpus. Constructing that
corpus through the input-manifest loader still validates its entire declared
population, including test records.

Caller RNG state is restored after loading. For exact continued training, invoke
`run_governed_planner(resume_from=...)` or the training CLI: calling `fit` directly
on the returned mutable trainer outside the isolated resume context is not a
verified exact-continuation path. The checkpoint must meet the existing strict
resume compatibility requirements; this API does not relax older implementation
or runtime mismatches or provide a weights-only migration path. Diagnostics remain
development evidence, not calibration, independent test evidence or ASL acceptance.


## Run checkpoint diagnostics from the command line

Use the same byte-bound run configuration and an explicitly identified checkpoint:

```sh
.venv/bin/python -m signtranslator.governed_diagnose run.json \
  --sha256 EXPECTED_RUN_CONFIGURATION_SHA256 \
  --checkpoint outputs/run.last.pt \
  --checkpoint-sha256 EXPECTED_CHECKPOINT_SHA256 \
  --view train --model-state current \
  --sample-indices 2 0 --permutation 1 0 --seed 9 --max-samples 2
```

Replace both digest placeholders with the lowercase SHA-256 values for the
intended files. This example requires three admitted training samples. All flags
are required; no checkpoint discovery, default sampling or test-view option is
provided. Checkpoint references resolve against the configuration directory,
independently of the shell's current directory. Absolute, traversal and symlink
references are refused by the existing file resolver.

`--sample-indices` selects an ordered, unique, zero-based subset of the chosen
train/validation view. `--permutation` maps that ordered subset to source rows;
`1 0` swaps the two selected source inputs while keeping anchor references fixed.
`--seed` is the diagnostic uint32 seed, separate from the recorded training seed.
`--max-samples` must be 1–64 and must cover the subset. The selected view must exist
in the original run configuration. `current` means the selected checkpoint's
current model tensors; `best_validation` means its retained validation-selected
weights, not an automatically chosen file or a calibrated model.

The command reads the shared strict run configuration, re-admits the complete
input manifest, restores the checkpoint through `load_governed_planner`, and runs
explicit development diagnostics. It never fits or saves and ignores configured
checkpoint output destinations. It emits one canonical JSON object on stdout;
Handled request, admission and identity errors exit with status 2 and emit no
diagnostic JSON; unexpected runtime errors may propagate. The Python
`diagnose_from_configuration` entry point returns those canonical bytes.

As with existing resume, checkpoint loading requires a trusted origin because
optimizer/RNG restoration deserializes Python objects; a hash identifies bytes
but does not establish that trust.

The schema-1 output binds configuration, input-manifest, corpus and checkpoint
hashes, plus the full diagnostic report/hash and fresh declaration audit/hash.
The nested report retains the actual evaluated tensor hash, selected view/subset,
training cursor, exposure identity and paired five-head results. Both the outer
and nested evidence keep `phase_exit_approved` false. Before/after checkpoint
hashes detect ordinary changes during execution, but they do not constitute an
atomic snapshot against concurrent file replacement; input files must remain
stable. Whole artifact parsing, checkpoint restoration, model copies and report
serialization still consume memory. No source permission, calibration, human
review, independent test performance or ASL competence is established by running
this command.


## Selected-model exposure boundaries

The nested governed diagnostic report now uses **schema 3**; the CLI wrapper
remains schema 1. Existing `completed_epochs`, `global_step` and
`training_exposure_sha256` describe the loaded run's current committed boundary.
They must not be interpreted as the training boundary of an earlier retained
best-validation model.

`selected_training_boundary` records the selected epoch, its optimizer-step count,
and derivation. For `current`, it uses the committed cursor. For `best_validation`,
it takes the first recorded minimum of the configured validation metric and uses
that metric's actual epoch index. This preserves the trainer's strict-improvement
tie rule and non-unit validation cadence. The entire support-aware history and
exposure sequence are validated before deriving this boundary.

`selected_boundary_exposure` and `selected_boundary_exposure_sha256` summarize
only optimizer declarations through that boundary. Later calls are excluded even
though they remain part of the loaded run's full exposure identity. At the current
boundary, the two hashes are equal. Selection history and cursor are checked again
after diagnostics; inconsistent or changing history is refused.

These are historical declarations, not proof that the evaluated tensor bytes
received specific gradients. Mutable in-memory weights, unsupported/shared
parameters and historical missing declarations retain their existing limitations.
The actual evaluated tensor hash remains separately recorded. The extra history
validation and selected-prefix summary add linear traversal cost; no new memory
or performance guarantee is made. Older schema-1 reports do not contain a
selected-boundary attribution. Schema 2 introduced that boundary; schema 3 retains
it and adds the relation-type projection below. Older reports must not be silently
interpreted as a later schema.


## Relation-type training declarations

Schema 3 adds `selected_relation_exposure`, a schema-1 projection of the same
selected-boundary ledger. It revalidates the records and canonical ledger hash;
per-type totals cannot silently come from a different epoch or ledger. In the
fixed relation codebook order, each row reports positive and negative cell
presentations, distinct sample IDs for each polarity, and the polarities actually
recorded. Empty lists mean no recorded polarity, not a trained negative class.

Repeated cells across optimizer calls count repeatedly as presentations; repeated
sample IDs count once per type/polarity. Distinct samples are not assumed to be
independent signers or recordings. Missing target declarations are counted as
`unrecorded_example_presentations`; explicitly recorded empty cell lists contribute
to `empty_recorded_example_presentations`. Missing historical targets are never
reconstructed from current annotations. Unknown-cell totals cannot be inferred
from this selected-cell ledger.

Only the canonical directed, nonself, binary relation contract is accepted, with
relation indices in the fixed codebook and event indices inside the model's
configured capacity. This projection does not use the graph decoder's separate
128-event search cap. It adds one selected-history traversal and per-type sample
sets. It does not change graph decisions, supply calibration thresholds, certify
nonzero gradients, or turn recorded polarity coverage into empirical acceptance.

## Evaluate explicit relation thresholds

The Python `evaluate_relation_sequences` API accepts an optional
`relation_thresholds=DiagnosticRelationThresholds(...)`. Supply one finite
negative and positive logit threshold per canonical relation type, with each
negative threshold strictly below its positive threshold. Scores below the lower
threshold are negative, scores above the upper threshold are positive, and both
boundary equalities abstain. Float32 scores are embedded exactly in float64 before
comparison; thresholds are not rounded down to the score dtype.

When supplied, the relation evaluation report uses schema 2 and adds
`threshold_diagnostic`: per-example and pooled per-type true/false positive and
negative counts, abstentions separated by target polarity, and positive/negative/
undecided decisions on unknown targets. Unknown decisions are descriptive only
and do not enter the confusion matrix or scored rates. Failed or nonmatching
label sequences have unavailable decision rows, not fabricated negative outcomes.

`known_decision_coverage` is decided known cells divided by all known cells.
`conditional_error` is incorrect decided known cells divided by decided known
cells. Both retain exact integer numerator/denominator pairs. A zero denominator
means unavailable, not zero error. Counts remain conditional on exact generated
label sequences and observed target support; they are not independent statistical
units, full-graph accuracy, calibration or ASL acceptance.

The function evaluates the caller's thresholds; it does not select or optimize
them. Omitting them preserves the existing schema-1 score report. This evaluation is enabled only when explicit thresholds are supplied to the
governed diagnostic API/CLI or source-intervention runner, as described below.
Graph decoding and its explicit uncalibrated status are unchanged.


## Supply thresholds to saved-run diagnostics

`diagnose_governed_planner` and `compare_source_intervention` accept optional
`relation_thresholds=DiagnosticRelationThresholds(...)`. The thresholds are
snapshotted and validated before model generation. They apply to the original
source candidates' reviewed-reference relation evaluation. Permuted sources are
not relabeled or used as newly annotated examples.

For the diagnostic CLI, supply both `--relation-thresholds-manifest thresholds.json`
and `--relation-thresholds-sha256 EXPECTED_THRESHOLD_MANIFEST_SHA256` alongside the
existing required options. The path is relative to the run-configuration directory;
no threshold is inferred when these flags are omitted. The file must be at most
four KiB and contain exactly these schema-1 fields:

```json
{
  "schema_version": 1,
  "relation_types": ["precedence", "overlap", "scope", "coref", "locus"],
  "negative_below": [-1.0, -1.0, -1.0, -1.0, -1.0],
  "positive_above": [1.0, 1.0, 1.0, 1.0, 1.0]
}
```

The example shows syntax only; these numbers are not selected or recommended
thresholds. Preserve the exact relation-type order. Bounds must be finite JSON
floating-point numbers, with lower strictly below upper for each type. Duplicate
keys, extra fields, unknown codebooks, integer/bool bounds, nonfinite values,
invalid paths and hash mismatches are refused. Configuration/checkpoint/source
requirements and the complete input-population admission remain unchanged.

With a threshold manifest, the outer CLI output uses schema 2 and includes
`relation_thresholds_manifest_sha256`; without it, the outer schema stays 1.
The nested governed diagnostic remains schema 3, and its intervention report's
`original_relation_evaluation` uses schema 2 when thresholds are supplied. That
relation report includes the actual values and all conditional counts. The
command still performs no training or file writes. This is reproducible evaluation
of supplied values, not a calibration certificate or threshold-selection workflow.
