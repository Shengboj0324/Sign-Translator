# 03 — Strict Readiness Audit

## 1. Executive verdict

| Question | Verdict |
|---|---|
| Does the package import and compile? | Yes |
| Are many mathematical primitives tested? | Yes |
| Is the synthetic pipeline executable? | Yes |
| Is the full thirteen-layer architecture integrated? | No |
| Is it ready for a small real-data pilot? | Not yet |
| Is it ready for expensive full training? | No |
| Is it ready for user deployment? | No |

The present codebase is between **verified components** and an **integrated
synthetic research core**. It has not reached real-data pilot training,
scientific validation, or deployment.

## 2. Audit evidence

### Source inventory

- 334 non-directory files
- 311 Python files
- approximately 39,000 Python lines
- 136 test files
- no `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.npz`, or `.npy` artifacts in the
  distributed archive

### Static validation

`python3 -m compileall -q .` completed successfully.

This proves that the files parse under the audited interpreter. It does not
prove that imports, dependencies, training, or inference are correct.

### Test results

Standard run:

```text
1454 passed
10 failed
1 warning
```

Warning-strict run:

```text
11 failed
```

The ten substantive failures occur in:

- `tests/test_speech_stage3_integration.py`;
- `tests/test_speech_stage5_harness.py`.

The synthetic noisy condition remains at 100% accuracy. As a result:

- there are no errors against which confidence can be discriminative;
- abstention cannot improve selective accuracy;
- the fail-closed policy cannot prove that it suppresses errors;
- Brier resolution is zero;
- temperature scaling worsens ECE in the tested split;
- noisy and long-form conditions are classified as degenerate.

These failures do not invalidate all speech mathematics. They invalidate the
claim that the current robustness harness demonstrates useful behavior.

### Synthetic training smoke test

The audited smoke run:

- generated 256 training and 64 validation samples;
- passed the repository’s internal readiness gate;
- constructed a 1,610,109-parameter model;
- completed one training epoch;
- wrote a roughly 20 MB checkpoint;
- reloaded that checkpoint;
- ran acoustic-feature-to-motion inference;
- returned finite tensors of the expected shape.

The one-epoch metrics failed, as expected for an untrained model. Their values
should not be interpreted as estimates of eventual accuracy. The useful result
is that the synthetic path executes mechanically.

## 3. Critical findings

### R1 — Corpus overwrite risk

`signtranslator/run.py` defaults to `regenerate=True`. The documented CLI does
not expose a `--no-regenerate` option. Supplying a real corpus directory to the
default command can replace its standard files with generated synthetic data.

**Required correction:** default to non-destructive behavior. Synthetic
generation should require an explicit flag and should refuse to write into a
non-empty corpus directory without confirmation.

### R2 — Missing modality can pass evaluation

`analysis/report.py` initializes `speech_wer = 0.0`. If a corpus has no speech
features, the speech branch is not evaluated but receives a passing value.

**Required correction:** represent missing evaluation as unavailable and fail
the corresponding gate when that modality is required.

### R3 — “Best” checkpoint is overwritten

The trainer saves a best-validation checkpoint during `fit()`. After fitting,
`run.py` saves the final model to the same path, overwriting the best model.

**Required correction:** maintain distinct `best`, `last`, and optional
milestone checkpoints.

### R4 — Checkpoint is not self-describing

The checkpoint includes:

- model state;
- optimizer state;
- global step;
- best validation loss.

It omits:

- architecture and diffusion configuration;
- scheduler and epoch;
- RNG states;
- vocabulary/tokenizer;
- pose normalization statistics;
- model-format version;
- dataset and preprocessing identifiers;
- source revision;
- evaluation thresholds.

Without those fields, a checkpoint cannot independently reproduce training or
standalone inference.

### R5 — Resume is not a true resume

`resume=True` loads weights with `load_optimizer=False`. Scheduler state is never
saved. The CLI does not expose the resume option. This is a warm restart, not a
continuation of the same optimization trajectory.

### R6 — Validation is stochastic

Diffusion validation samples random timesteps and noise. The composite
validation loss used for checkpoint selection therefore changes with random
draws. Branch losses also have different scales, so their weighted sum is not a
stable scientific selection criterion.

**Required correction:** use fixed validation seeds/noise/timesteps, report
per-branch confidence intervals, and choose a pre-declared primary checkpoint
criterion.

### R7 — CTC feasibility is under-specified

The readiness gate checks only `frames >= target_length`. CTC may require extra
frames when adjacent labels repeat. Speech subsampling further reduces the
usable input length. `zero_infinity=True` can convert impossible alignments into
zero loss, silently removing bad examples from training.

**Required correction:** compute exact per-sample minimum CTC length after
subsampling and reject impossible samples before batching.

### R8 — Documentation and code have drifted

Examples include:

- the architecture document describes cycle consistency as diagnostic, while
  the current code gates it;
- the data document reports an older number of test files and a green suite;
- the requirements comment references a missing
  `requirements-foundation.txt`;
- the package declares an Apache license but ships no license file.

Documentation should be versioned and checked in CI against executable
configuration.

## 4. Readiness gates

| Gate | Current state | Evidence needed to pass |
|---|---|---|
| Build and import | Pass | Clean installation in a pinned environment |
| Unit mathematics | Partial | Zero failures under warning-strict tests |
| Synthetic integration | Partial/pass | Reproducible full run and artifact reload |
| Real corpus ingestion | Fail | Versioned real batch through active loader |
| Full architecture integration | Fail | One traced semantics-to-render path |
| Real pilot training | Fail | Tiny-subset overfit and held-out experiment |
| Scientific validation | Fail | Independent baselines and signer-held-out tests |
| Human comprehension | Fail | Blinded qualified-signer evaluation |
| Runtime deployment | Fail | Loadable service/app with measured latency |
| Operational safety | Fail | Integrated abstention, monitoring, and rollback |

## 5. Strict release language

Until the failed gates are resolved, releases should say:

> Research scaffold with synthetic training and independently tested
> mathematical components.

They should not say:

> Production-ready translator, complete speech-to-avatar system, validated
> sign-language generator, or accessibility replacement.

