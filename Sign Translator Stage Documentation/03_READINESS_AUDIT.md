# 03 — Strict Readiness Audit

## W0 Days 2–5 repair status — 2026-09-25

The current repair record is in `06_IMPLEMENTATION_ROADMAP.md`, preceding the frozen Day-1
baseline. B35/B36 (statistical domains/tails), B18 (active masked motion objectives), B17
(declared topology and checkpoint binding), B23/B31 (nonempty/finite training) and B33
(checkpointable defaults) are implemented. **Final verification: 1,816 passed, zero failures/errors/skips; warnings treated as errors.**
The installed-wheel custom-topology smoke completed four optimizer steps and finite,
bit-identical seeded checkpoint reload. See
[the final verification record](evidence/w0-days2-5-2026-09-25-final/verification.json).
The original findings below remain historical evidence of the pre-repair state.

The scope of B18 closure is the active training objective; length-aware analysis and complete
feature masking remain B19/Day 6/W2–W3. B20 physical-time derivatives, B24–B28 evaluation and
selection, and B37–B39 empirical statistical design are not closed by these repairs.
**Empirical readiness remains unapproved.** B67 remains open: no missing Cokely source file was
restored or replaced, and no historical source manifest was modified.

## W0 Day-1 refresh — 2026-09-25

The Day-1 task rechecked checkout `89e2238d371ae8e0af826282520f4c14416bd9c5`, which adds
only the previous audit/roadmap documents over the package/test baseline named below.
Fresh results and exact file identities are in
[evidence/w0-day1-2026-09-25/baseline.json](evidence/w0-day1-2026-09-25/baseline.json).
The [Day-1 execution record](06_IMPLEMENTATION_ROADMAP.md#w0-day-1-execution-record--2026-09-25)
contains the frozen scope, owners, dependencies, and source/reviewer requirements.

**New current blocker B67 — missing Cokely primary HD media (P0 for reuse of that source).**
The historical source-native manifest and binding require
`2_I_Have_a_Dream_720_CokelyAFSParallelCorpus_v1_0.mp4`, expected SHA-256
`8eb18b6b2f01a5a179100b0acd84a639b5f2a0ee95fb0195d477b946217c6bb3`, 650,953,008 bytes.
That file is absent under the recorded local source root
`/Users/jiangshengbo/Volumes/cokely_reference/v1/source/i_have_a_dream/`.
The SD alternate, EAF and publisher-page snapshot remain present. No reason for the absence
is inferred. The original one-item ingestion result is historical, not a currently
reproducible source-bundle guarantee. Source integrity fails even if other hashes match.

Closure: restore the exact authorized, hash-matching HD file, or create a separately
versioned and independently validated binding with explicit timing/source treatment.
Do not silently substitute SD, fabricate the primary hash, modify the historical manifest,
or use that broken binding for review/training. Ownership and due gate are D23 in the
roadmap. This raises the blocker register from the prior B01–B66 to **B01–B67**.

**Day-1 verification status: complete capture.** All 1,726 tests passed with zero
failures/errors/skips; compile/dependency checks passed. Eight synthetic optimizer steps
produced finite output and bit-identical seeded last-checkpoint reload. **Source integrity
FAILS:** eleven recorded files match and the Cokely HD primary is missing. No model or
statistical implementation was repaired; the five reproduced defects remain open.

## Implementation audit snapshot — 2026-09-25

This snapshot supersedes older historical verdicts below; the Day-1 refresh above takes
precedence for current source availability. Scope: checkout
`839acf3bc8d67fef10b0e777a181a086f83a66d8`, active runtime, specialized component
interfaces, tests, and saved evidence under `/Users/jiangshengbo/Volumes`.
This is a comprehensive implementation/readiness audit, not a proof that every line
is defect-free. “Absent” means no accepted artifact was found in these inspected
locations; private agreements and artifacts elsewhere were not inspected.

### Current implementation status

| Layer | Status and actual boundary |
|---|---|
| Phase 1 / Stage A | Substantial execution-integrity implementation: opt-in corpus generation, exact CTC feasibility, schema-v2 checkpoints, best/last separation, epoch-boundary CPU resume, isolated validation RNG, archive CI. Remaining integration/statistical defects are below. |
| Active executable | `run.py` → `models/pipeline.py` → compact speech CTC, synthetic token/gloss planner, Cartesian diffusion, ST-GCN recognition, contrastive alignment. This is an executable research core. |
| Real-data engineering | Decoder/exporter, governed records, masks, timestamps, grouped splitting, source audit and 2D experiment exist. An accepted real multichannel training-to-render path does not. |
| Phase 2 | No accepted source-grounded canonical multichannel state/export/round-trip. Existing pose mathematics is not that completed state. |
| Phase 3A | Source-native EAF ingestion exists; Cokely one-item compatibility evidence exists. It does not authorize linguistic mappings. |
| Phase 3B | Governed queues, review/adjudication and acceptance software exist; an accepted independently reviewed SIR corpus was not found. |
| Phase 3C | Lexical registry, unknown/ambiguous/unauthorized abstention and evidence conjunction exist. Research and industrial empirical exits remain unapproved. |
| Phases 4–5 | Autoencoder/RVQ, transformer, DiT, constraints and facial modules exist independently. No accepted real multichannel representation or trained conditioned generator. |
| Phase 6 | Evaluation/statistical/human-protocol primitives exist, with newly reproduced statistical defects below. No completed independent signer-comprehension study. |
| Phase 7 | Streaming/latency/quantization/runtime primitives exist. No verified production input-to-avatar service, hardware benchmark or release bundle. |
| Reverse direction | Active sign recognition outputs gloss IDs. Full real-video ASL→English translation is a separate unfinished integration goal. |

### Evidence inspected and reproduced

- Fresh warning-strict suite: **1,726 passed**, process exit 0 (`.venv/bin/python -m pytest -q -W error`; quiet output records 1,726 passing test dots).
- `.venv/bin/python -m pip check`: no broken requirements.
- Current runtime observed: Python 3.12.14, Torch 2.13.0; historical experiments used a different Python patch release.
- Fresh bounded one-epoch synthetic/checkpoint smoke: **passed**, 8 optimizer steps; training/saving took 13.82 seconds in this small CPU run (not a real-data throughput estimate).
- A 137-joint `build_model()` reproduces `ValueError: skeleton graph is not connected` because the default edges still describe 27 joints.
- With identical model, batch and random seed, replacing frame/validity/confidence masks with all-zero masks leaves generation loss exactly unchanged: `3.1042861938476562` both times. The active loss ignores observation support.
- A batch size of 10,000 on the default synthetic training corpus yields zero training batches (`drop_last=True`); trainer construction does not reject this.
- General evaluation `sign_test_pvalue([1.]*60, [0.]*60)` returns `0.0`; its exact two-sided value is `2^-59 = 1.734723475976807e-18`. This is cancellation in the general helper, not the separately stabilized Phase-3 dependence helper.
- `paired_permutation_pvalue([NaN, 1], [0, 0])` returns `0.0` instead of rejecting invalid scores. Such a result must never be accepted as evidence of significance.
- An empty-input Phase-3C readiness call returns seven missing-evidence blockers. This verifies default fail-closed behavior; it is not itself a filesystem discovery mechanism.
- The saved signer CSV hash matches its certificate; the audit-manifest hash matches the signer certificate. The entire raw video corpus was not re-decoded or re-hashed in this audit.

The saved full-data audit accounts for **31,165 metadata rows**: 31,047 complete joins,
118 missing sources, 3 structural failures, 28,621 quality warnings, 2,423 valid rows,
and 1 additional orphan artifact. Quality warnings are not automatically unusable data.
There are 1,702 review-queue records and no selected quality threshold in that manifest.
Its old “signer identity absent” note is superseded by the verified pseudonymous-signer
certificate; the final signer-and-source-disjoint split is still explicitly absent.

The saved real 2D experiment has **76/10/10 train/validation/test clips**, source-disjoint
but not signer-disjoint. Test point error is 0.063139 for the model versus 0.004936 for
interpolation; span error is 0.071484 versus 0.010596. Supports differ (21,940 vs 20,989
point targets; 20,351 vs 19,591 span targets), so these are reported metrics, not a valid
paired effect estimate. Tube interpolation has zero support and is unavailable, not zero
error. Tiny-subset fitting succeeded, but useful held-out improvement was not established.

### Numbered blocker register

Priority: **P0** blocks trustworthy next-stage work; **P1** blocks scientific/vertical
integration acceptance; **P2** blocks product or scale. “Confirmed” describes inspected
code or reproduced behavior. “Evidence gap” describes missing accepted project evidence.
Work-package references refer to the updated roadmap. Estimates are allocated there to
avoid counting shared fixes repeatedly.

| ID | Priority / type | Finding, consequence, and required closure | Package |
|---|---|---|---|
| B01 | P0 / evidence | No accepted co-observed continuous multichannel 3D source. Frontal 2D does not identify depth, palm twist, occluded fingers or gaze. Obtain and inspect an authorized source; preserve unavailable/inferred channels explicitly. | W1–W2 |
| B02 | P0 / evidence | No accepted qualified ASL annotation/review relationship in inspected artifacts. Assign a qualified creator, independent reviewer and disagreement adjudicator with real availability and evidence. | W1/W3 |
| B03 | P0 / evidence | No training-authorized governed text/video→SIR corpus. Populate and independently accept Phase-3B records; English transcripts and native EAF values cannot simply be renamed SIR. | W3 |
| B04 | P0 / evidence | No human-reviewed lexical motion entries. Bind meaning, articulation, form, schema and exact motion bytes; retain unknown/ambiguous refusal. | W3–W4 |
| B05 | P0 / integration | Canonical Phase-2 state is not implemented and accepted against a real source. Freeze units, frames, time, joints, expressions, gaze, confidence and availability only after sample inspection. | W2 |
| B06 | P0 / evidence | Research authorization for the final combined supervision/motion bundle is missing. Existing How2Sign research evidence is not absent, but it is not blanket authorization for every new source, annotation, derivative and asset. | W1/W3 |
| B07 | P2 / evidence | Commercial training/deployment permissions are not established. Maintain a separate commercial lineage and signed/action-specific evidence; this is a commercial release blocker, not a reason to prohibit otherwise authorized research. | W1/W9 |
| B08 | P0 / evidence | No qualified, authorized body/hand/face rig and model asset bundle for the target slice. Verify individual assets, expression channels, export permission, articulation and signer review. | W1–W2 |
| B09 | P0 / planning | Old pre-Phase-2 language requires all five research/commercial bundles, whereas Phase-3C separates research and industry. Resolve this as an explicit roadmap/gate-policy change, never a silent relaxation of executable checks. | W0–W1 |
| B10 | P0 / planning | No frozen domain, phrase/phenomenon coverage, audience, target hardware or primary success endpoint. Freeze a narrow pilot; otherwise annotation, architecture and acceptance keep moving. | W0 |
| B11 | P0 / data | Final signer-and-source-disjoint split is absent. Use connected signer/source components, freeze before windowing, certify both separations and report actual counts. | W1 |
| B12 | P1 / data | Signer distribution is highly uneven: 12,102 and 14,596 available utterances belong to two codes, about 86% of the local 31,047 clips. Measure feasible grouped partitions, macro scores and subgroup uncertainty; clip count is not independent signer diversity. | W1/W6 |
| B13 | P1 / data | 118 missing sources, 3 structural failures and an orphan remain. Quarantine and disposition explicitly; do not fabricate missing files or silently repair truth. | W1 |
| B14 | P1 / data | 28,621 quality warnings, 1,702 queued reviews and no frozen QC threshold. Review stratified samples, quantify hand/face/occlusion failure, select thresholds on development data and record accepted exclusions. | W1 |
| B15 | P1 / data | Decoder clock support exists, but final source synchronization/calibration is unvalidated. Test clip origin, frame rate, dropped frames, EAF milliseconds, motion units and face/audio offsets with real samples. | W1–W2 |
| B16 | P0 / integration | Existing exporter explicitly requires source and gloss tokens. The gloss-independent roadmap has no direct governed SIR-to-training shard adapter. Implement a distinct typed path without fabricating gloss. | W3 |
| B17 | P0 / confirmed | `build_model` sizes joints from corpus but uses default 27-joint edges. A 137-joint model fails construction. Bind topology and joint ordering to the manifest and checkpoint. | W0/W2 |
| B18 | P0 / confirmed | Active generation ignores validity, confidence and frame masks. Invalid/padded targets enter MSE and velocity loss; all-invalid support still returns a loss. Carry support through losses and fail unavailable required objectives. | W0/W2 |
| B19 | P1 / integration | CTC lengths exist, but feature convolutions, pooling and decoding are not fully observation/padding aware. Test invariant output under added padding, padded batching, occlusion and variable length; CTC length alone is insufficient. | W2–W3 |
| B20 | P1 / confirmed | Velocity loss uses adjacent frame differences rather than source time intervals. Preserve timestamps; use time-aware derivatives and masks requiring both valid endpoints. | W2 |
| B21 | P1 / confirmed | Exporter builds vocabularies from all records after splitting. Declare a pre-existing closed lexicon or fit only on training data with explicit unseen-token handling; otherwise held-out vocabulary informs preprocessing. | W1/W3 |
| B22 | P1 / integration | Shards and datasets are materialized in memory, with corpus-wide temporal padding at export. Measure peak RAM/I/O and adopt bounded shards/bucketing as needed before full-scale runs. This is a scale risk, not a measured OOM. | W3/W7 |
| B23 | P0 / confirmed | `drop_last=True` can create a zero-step epoch. Reject empty loaders, adapt tiny-overfit batch size and require positive optimizer-step evidence. | W0 |
| B24 | P1 / confirmed | Validation averages batch means equally, including unequal final batches and unequal modality support. Accumulate numerators/denominators under the intended sample/frame estimand. | W0/W6 |
| B25 | P1 / confirmed | Best checkpoint is chosen by weighted total loss; analysis uses final in-memory model without explicitly selecting best. Freeze primary selection metric and name the exact checkpoint being reported. | W0/W5 |
| B26 | P1 / confirmed | Canonical validation RNG is isolated, but `analysis/report.py` independently draws generation noise and does not use that isolated protocol. Bind analysis seeds/replicates and restore caller state. | W0/W6 |
| B27 | P1 / confirmed | Analysis concatenates per-batch padded gloss tensors; different batch widths can fail. Recognition evaluation also omits input lengths. Implement ragged-safe, length-aware evaluation. | W0/W3 |
| B28 | P1 / confirmed | Planner metric decodes only eight tokens and counts reference-position matches without penalizing extra insertions. Add sequence/edit and semantic-field metrics with declared maximum-length/refusal behavior. | W0/W6 |
| B29 | P1 / confirmed | Diagonal-only InfoNCE treats semantically equivalent off-diagonal pairs as negatives. Define governed multi-positive grouping and false-negative controls before applying to repeated real meanings. | W4–W5 |
| B30 | P1 / integration | Joint weighted training starts all branches together; richer curricula are not active. Establish branch competence, gradient scale/flow diagnostics and controlled loss-weight ablations. | W4–W5 |
| B31 | P0 / confirmed | Trainer clips gradients but has no explicit non-finite-loss/gradient abort before optimizer mutation. Reject NaN/Inf, record offending batch provenance and preserve last good checkpoint. | W0 |
| B32 | P1 / confirmed | Exact continuation is scoped to epoch-boundary joint CPU training with zero loader workers. Fine-tune/polish checkpointing is rejected; worker, accelerator, distributed or mid-epoch recovery is not proven. Implement only the paths actually needed and label their guarantee. | W0/W7 |
| B33 | P1 / usability | CLI defaults to 30+175+16 epochs; `--ckpt` conflicts with nonzero secondary stages. API/CLI learning-rate defaults differ. Make bounded, checkpointable runs explicit and test documented commands. | W0 |
| B34 | P1 / integration | Runtime defaults to CPU and exposes no run-level device parameter. EMA, AMP and gradient accumulation are not integrated into the canonical trainer. Benchmark first; then add required hardware paths with parity/resume tests. | W4/W7 |
| B35 | P0 / confirmed | General sign-test helper has upper-tail cancellation; 60 all-positive pairs produce zero instead of 2^-59. Use stable tail/log-probability computation and numerical oracle cases. | W0 |
| B36 | P0 / confirmed | General permutation helper accepts NaN and can return p=0. Validate all score/config domains, finite values, minimum support and resample counts across general statistical helpers. | W0 |
| B37 | P1 / statistical | Video-dependence sign tests count examples; repeated sources/signers are not aggregated into independent units. Predeclare cluster-level tests or justify independence; otherwise significance can be overstated. | W1/W6 |
| B38 | P1 / statistical | General bootstrap is IID and seed aggregation accepts one seed. Add appropriate signer/source/reviewer hierarchy and distinguish training-seed variation from population uncertainty. | W6 |
| B39 | P1 / evidence | No frozen real-data primary endpoint, minimum useful effect, power/precision budget, exclusions, multiplicity family and calibration/test separation. Preregister before inspecting final outcomes. | W1/W6 |
| B40 | P1 / evidence | No trained direct paired learner and held-out four-intervention score artifact. Evaluator existence is not a passed experiment. Train only with accepted inputs and run all interventions. | W3–W5 |
| B41 | P1 / evidence | Existing 2D pretraining does not establish improvement over interpolation. Recompute on common support with coverage and clustered uncertainty; redesign only if a useful hypothesis survives. | W4 |
| B42 | P1 / integration | Active Cartesian motion lacks full finger rotations, facial grammar, head/gaze and contact state. A tensor-size increase alone cannot solve these semantics. | W2/W5 |
| B43 | P1 / integration | Hand-graph, SIR, duration, spatial reference and non-manual modules are disconnected from active generation. Add typed adapters and intervention tests at each boundary. | W3/W5 |
| B44 | P1 / evidence | No qualified representation comparison among retrieval/interpolation, continuous AE and RVQ; no accepted codebook-use or reconstruction/minimal-pair evidence. Keep simple baselines eligible to win. | W4 |
| B45 | P1 / integration | Rich DiT/constraints/part-aware generation is separate from active diffusion. Choose one generator, test parameterization/schedule/guidance compatibility, inpainting masks and constraint residuals. | W5 |
| B46 | P1 / integration | Biomechanical/temporal corrections are not shown to preserve meaning. Test handshape, palm orientation, contact, continuity and non-manual scope before/after projection and chunk stitching. | W5–W6 |
| B47 | P1 / integration | No complete inverse-normalization→body model→retarget→frame-render path. Persist units, handedness, rest pose, scale, face map, camera and timestamps; provide one reload-and-render command. | W2/W5 |
| B48 | P1 / evidence | Renderer functions include mathematical primitives/callbacks, not a qualified final avatar. Validate visible fingers, face/gaze, self-occlusion, mirroring and semantic fidelity with qualified signers. | W5–W6 |
| B49 | P1 / evidence | Cycle consistency uses a related recognizer and can reward shared shortcuts. Add independent baselines and blinded semantic comprehension; cycle WER remains diagnostic. | W6 |
| B50 | P1 / evidence | No frozen multi-seed real held-out generation result, counterfactuals or ablations. Report per-channel failures, coverage and confidence intervals, not only total loss. | W5–W6 |
| B51 | P1 / evidence | Human comprehension, grammaticality and error severity remain unevaluated. Recruit independent reviewers, randomize/blind presentation, retain disagreements and preregister analysis. | W6 |
| B52 | P1 / integration | Confidence/calibration/refusal primitives are not connected to real semantic failures. Fit on held-out calibration data, measure risk-versus-coverage, test out-of-domain inputs and empty evidence. | W6–W8 |
| B53 | P2 / integration | Raw audio backend/preprocessing is not active; `translate_audio_to_sign` receives acoustic features. Integrate waveform/sample-rate/channel contracts, real backend assets, tokenizer and timestamps. | W7 |
| B54 | P2 / evidence | Synthetic noise/pitch conditions do not validate real accents, noise, code switching or long speech. Collect eligible real stress cases and characterize actual failure/abstention. | W7 |
| B55 | P2 / integration | Streaming revision/commit and backpressure contracts are independent primitives. Connect cancellation, committed-prefix immutability, stale-result dropping and temporal seams to the renderer. | W7 |
| B56 | P2 / evidence | No measured first-output/end-to-end p50/p95/p99 latency, throughput, queue behavior, peak memory or target-hardware capacity. Benchmark the actual composed path before optimization promises. | W7–W8 |
| B57 | P2 / integration | Quantization/static-execution algebra is not optimized production execution. Require numerical AND per-channel/linguistic parity; reject speedups that harm signing. | W8 |
| B58 | P2 / integration | No self-contained service/app bundle with authorization, retention, logging, model/data identity, failure reporting, rollback and load-tested lifecycle. Build and rehearse a restricted release. | W8–W9 |
| B59 | P1 / trust | Hashes and typed attestations bind contents but do not authenticate qualifications, permission or observed truth by themselves. Verify issuers/documents and bind accepted evidence to the actual training run. | W1/W3/W9 |
| B60 | P1 / integration | `run.py` checks corpus structural readiness, not the complete source-portfolio/Phase-3/human release gates. Connect action-specific policy at training/export/release boundaries; do not treat its `passed` flag as project readiness. | W3/W9 |
| B61 | P1 / maintainability | Parallel implementations in `models/`, specialized packages, legacy trainers and evaluation can drift. Declare owners/canonical paths; retain explicitly quarantined research paths with scope labels. | W0 |
| B62 | P1 / evidence | CI is defined for macOS; current tests do not prove clean install today on every target, accelerator determinism or production data portability. Run the selected release environment and archive/wheel checks. | W0/W9 |
| B63 | P1 / documentation | Historical audit still lists fixed overwrite, missing-modality, checkpoint, CTC and license issues. Preserve history but make current verdict authoritative; remove these from the new outstanding-work queue. | W0 |
| B64 | P2 / scope | Full ASL→English needs real-video perception, multichannel temporal understanding, an English decoder and independent evaluation. Current sign→gloss recognition does not complete the bidirectional goal. | W10 |
| B65 | P1 / resourcing | External acquisition/review turnaround, annotation throughput, compute throughput and budget are unmeasured. Assign owners and benchmark them; engineering days cannot guarantee external evidence arrival. | W0–W1 |
| B66 | P1 / planning | Completing gate software has been advancing phase labels while empirical dependencies remain unresolved. Track software-complete and evidence-accepted separately; no downstream empirical phase passes on test count alone. | All |

### Minimal reproduction commands for newly found defects

Run from the repository with the existing `.venv`. These are diagnostic probes of the
current code, not assertions that the returned behavior is acceptable:

```python
from dataclasses import replace
from signtranslator.run import build_model
from signtranslator.data.corpus import CorpusSpec
from signtranslator.eval_framework.statistics import (
    sign_test_pvalue, paired_permutation_pvalue,
)

print(sign_test_pvalue([1.] * 60, [0.] * 60), 2. ** -59)
print(paired_permutation_pvalue([float("nan"), 1.], [0., 0.]))
spec = CorpusSpec.build(num_concepts=12, seq_len=4, num_joints=27,
                       in_channels=3, num_frames=32)
try:
    build_model(replace(spec, num_joints=137), diff_timesteps=10)
except ValueError as error:
    print(type(error).__name__, str(error))
```

For the mask test, take a batch from a newly generated temporary synthetic corpus,
put the model in evaluation mode, and compute `training_step(batch)["generation"]`
with seed 123. Repeat with the same seed after replacing `validity_mask`, `confidence`
and `frame_mask` with all-zero tensors. The identical losses above demonstrate that
these fields do not affect the active objective. Never use an existing real-data
folder as a synthetic probe target. No source code was changed during this audit.

### Findings that must not be charged again as unresolved baseline work

Corpus generation is now explicit and guarded; missing speech reports unavailable/failure;
best and last checkpoints are distinct; checkpoint metadata and CPU resume are substantially
implemented; canonical validation has isolated RNG; exact repeat-aware CTC feasibility and
`zero_infinity=False` are active; license and foundation requirements files exist. The
pseudonymous signer grouping key is certified. These are real accomplishments, with the
remaining limitations called out above.

### External-source check

On 2026-09-25 the [How2Sign publisher page](https://how2sign.github.io/) still lists
CC BY-NC 4.0 and research-only availability. The [SMPL-X license page](https://smpl-x.is.tue.mpg.de/modellicense.html)
still distinguishes noncommercial scientific use from separately licensed commercial use.
These publisher statements do not resolve project-specific permissions or authorize new
asset downloads, outreach, purchases or deployment.

---

## Historical audit baseline (retained for traceability; not current status)


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

### 2026-09-10 remediation status for R3–R7

These findings describe the earlier audit baseline. The current Phase-1 implementation
changes their status as follows:

- **R3 resolved for the canonical joint trainer:** `best` and `last` are distinct;
  explicit milestone saves are supported; the final state no longer overwrites best.
- **R4 substantially resolved:** schema-v2 checkpoints and hash-verified JSON sidecars
  bind model/diffusion/trainer configuration, corpus manifest and shard hashes,
  vocabulary and normalization through that manifest, data-loader contract, numerical
  runtime, source-byte identity, optimizer, scheduler, epoch/step, history, and RNG state.
  Future architecture migrations remain fail-closed rather than implicit.
- **R5 resolved for epoch-boundary joint CPU training with `num_workers=0`:** resume
  restores the same optimization trajectory and is tested bit-for-bit. Worker-local RNG
  and the legacy generator-only fine-tune/polish stages are not yet state-complete;
  checkpointing those paths is explicitly rejected.
- **R6 partially resolved:** validation now uses an isolated fixed RNG stream, produces
  repeatable values, and cannot advance training RNG or leave the model in evaluation
  mode. Per-branch confidence intervals and the preregistered primary selection criterion
  still require real experimental design and remain open.
- **R7 resolved in the active sign, speech, corpus, exporter, and speech-objective paths:**
  the exact adjacent-repeat minimum is enforced after subsampling, blank/out-of-range
  targets are rejected, and active CTC losses use `zero_infinity=False`.

This remediation establishes execution integrity only. It does not change the real-data,
linguistic, human-evaluation, or deployment gates below.

### 2026-09-10 pre-Phase-2 data verdict

One prior limitation is resolved: the How2Sign filename suffix is now certified as the
official pseudonymous signer ID by exact reconciliation of the immutable local audit with
the CVPR supplemental. All 31,047 available clips match published per-signer counts, and
all 118 missing rows reconcile by signer. This is a grouping key, not personal identity.

The new machine-enforced source-portfolio gate still fails every Phase-2 evidence bundle.
There is no locally verified single source with co-observed continuous 3D body, hands,
face, head, and gaze; no locally verified continuous qualified-ASL linguistic reference;
no commercial training authorization; no locally qualified commercial render rig; and no
confirmed qualified-ASL governance relationship. Capabilities from separate corpora are
not composable as if they had been co-observed. Phase 2 therefore remains closed. See
`04_DATA_ENGINEERING_AND_CORPUS.md` §§10–13 for evidence and acquisition treatment.

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
