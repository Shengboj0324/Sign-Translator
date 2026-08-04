# 06 — Implementation Roadmap

## 1. Roadmap principle

The next phase should optimize for a **complete vertical slice**, not additional
package count. Every milestone must produce an artifact that can be inspected,
reloaded, and falsified.

## 2. Stage A — Stabilize the repository

### Objectives

- obtain a clean, reproducible baseline;
- eliminate destructive and ambiguous behavior;
- designate canonical implementations.

### Required tasks

1. Pin Python, PyTorch, NumPy, and test dependencies in a lock file.
2. Fix all speech integration and warning-strict test failures.
3. Add CI for clean installation, compilation, tests, and a short smoke run.
4. Add the actual license file and correct packaging metadata.
5. Add the missing foundation requirements file or remove the reference.
6. Change corpus regeneration to explicit opt-in.
7. Add non-empty-directory and overwrite guards.
8. Select canonical planner, diffusion, speech, and evaluation packages.
9. Document deprecated duplicate paths.
10. Add typed configuration serialization and schema versions.

### Exit gate

- zero test failures under the pinned warning-strict environment;
- a fresh clone installs and reproduces the smoke result;
- no default command overwrites user data.

## 3. Stage B — Build the real-data bridge

### Objectives

- connect governed records to active training batches;
- preserve uncertainty and variable lengths.

### Required tasks

1. Implement media decoders with timestamp preservation.
2. Integrate body, dense hand, and face extraction.
3. Add optional multi-view triangulation or body-model fitting.
4. Implement an exporter from `data_engineering.Sample` to versioned shards.
5. Carry confidence and validity masks through every transform.
6. Implement signer/source grouped splitting before windowing.
7. Add variable-length collators and input-length propagation.
8. Validate exact CTC feasibility.
9. Persist vocabulary, label definitions, normalization, and coordinate metadata.
10. Produce HTML/video or equivalent human-review reports for sampled records.

### Exit gate

- one licensed, versioned real mini-corpus passes all schema and visual checks;
- every secondary-dataset record binds to a byte-verified local license-evidence
  snapshot and an exact action scope; no license label substitutes for consent;
- the active loader consumes it without synthetic generation;
- every tensor traces back to immutable source media.

### Current gloss-free preparation result (2026-08-03)

The canonical v1 full-corpus audit and the quarantined 2D masked-reconstruction
experiment are complete and reproducible. The audit accounts for all 31,165 metadata
rows plus one orphan artifact without fabricating the 118 missing clips or repairing
the three structural failures. Review queues and `VIDEO_ID` source constraints are
available, but they are not review attestations or signer identities. The 2D experiment
is intentionally disconnected from the exporter, active runtime, 6D motion tokenizer,
and Stage C; its held-out point/span model did not beat temporal interpolation.

Detailed evidence is recorded in `docs/DATA_ENGINEERING.md`. The pseudo-gloss
candidate-lattice implementation and its activation requirements are consolidated in
`09_PSEUDO_GLOSS_MODEL_RESEARCH.md`. The software path is implemented, but corpus-wide
generation remains gated on a versioned ASL lexicon/convention, independent qualified
human references, calibrated pretrained weights, frozen preregistration, and signer
mapping when signer-generalization is claimed.

**Stage B remains unapproved.** Authentic gloss, authoritative signer mapping, and
qualified signer review are absent. English transcripts, pseudo-glosses, filename
codes, review-queue generation, and source-disjoint partitions cannot substitute for
those gates. Stage C therefore remains blocked.

## 4. Stage C — Establish the minimal vertical model

### Recommended first scope

Start with:

```text
validated text or gloss
    -> minimal typed plan
    -> body + dense-hand motion
    -> deterministic rigged avatar
```

Defer raw speech and photorealistic rendering.

### Required tasks

1. Decide the canonical motion representation.
2. Wire minimal semantic/SIR fields into the generator.
3. Replace 27-joint Cartesian output if it cannot express the target phenomena.
4. Integrate variable duration and frame masks.
5. Connect biomechanical constraints to the active loss/sampler.
6. Add a real body/hand model under valid licensing.
7. Build a deterministic retarget-and-render path.
8. Add inverse normalization to inference.
9. Persist readable gloss/plan labels.
10. Add one command that loads a complete artifact and renders a sample.

### Exit gate

- the model overfits a deliberately tiny real subset;
- generated motion can be rendered and visually inspected;
- output quality changes appropriately when conditioning changes;
- shuffled conditioning performs materially worse.

## 5. Stage D — Make training scientifically credible

### Required tasks

- staged training rather than all-branch optimization from initialization;
- pretrained speech/language encoders where justified;
- deterministic validation;
- EMA for the generator;
- mixed precision and gradient accumulation;
- complete checkpoint/resume;
- per-loss and per-module gradient monitoring;
- automatic detection of zero, NaN, or exploding gradients;
- multi-positive contrastive alignment;
- loss-weight and curriculum ablations;
- experiment tracking with immutable configs.

### Recommended curriculum

1. fit and validate the motion representation;
2. train or validate the motion tokenizer;
3. train motion recognition;
4. train semantic planning;
5. train conditional motion generation;
6. add cross-modal alignment;
7. add non-manual and spatial conditioning;
8. jointly polish only after branch competence is established.

### Exit gate

- at least one signer-held-out real-data experiment;
- reproducible result across independent reruns;
- meaningful baselines and ablations;
- no branch passes solely because a modality is missing.

## 6. Stage E — Add speech and streaming

### Required tasks

1. Integrate raw waveform preprocessing with the active model.
2. Add a real ASR or speech representation backend.
3. preserve word/token timestamps and uncertainty;
4. connect calibration and fail-closed policy to actual inference;
5. implement revisable versus committed sign output;
6. test noise, accent, code-switching, long-form speech, and interruptions;
7. measure selective risk versus coverage on real errors.

### Exit gate

- raw audio reaches motion without manually supplied feature tensors;
- confidence predicts semantic failure;
- abstention or clarification reduces harmful assertions;
- streaming revision obeys the committed-prefix contract.

## 7. Stage F — Human evaluation

### Required tasks

- co-design the evaluation with qualified signers;
- pre-register primary endpoints;
- use blinded randomized presentation;
- compare against meaningful baselines;
- measure comprehension, grammaticality, naturalness, and acceptability;
- analyze errors by linguistic phenomenon and signer group;
- record uncertainty and disagreement.

### Exit gate

- the system demonstrates a practically meaningful improvement;
- failures are characterized well enough to define a safe use case;
- the target community considers the presentation and limitations acceptable.

## 8. Stage G — Deployment engineering

### Required tasks

- self-contained model bundle and loader;
- authenticated service or on-device runtime;
- actual optimized kernels and numerical equivalence tests;
- renderer and media output;
- latency and memory profiling on target hardware;
- monitoring, trace IDs, and failure reporting;
- model/data version telemetry;
- rollback and canary strategy;
- privacy, retention, and incident procedures;
- accessibility and user-experience testing.

### Exit gate

- an independently reproducible release candidate meets all quality, safety,
  latency, and operational gates on target hardware.

## 9. Priority matrix

| Priority | Work |
|---|---|
| P0 | Prevent corpus overwrite; fix test failures; connect real corpus exporter |
| P0 | Select canonical architecture and motion representation |
| P0 | Complete checkpoint/config/vocabulary/normalization bundle |
| P1 | Integrate SIR, biomechanical constraints, dense hands, and face |
| P1 | Deterministic training, baselines, ablations, and signer-held-out testing |
| P1 | Independent and human comprehension evaluation |
| P2 | Raw speech, calibration, and streaming revision |
| P2 | Optimized renderer and deployment runtime |
| P3 | Photorealism, large-scale pretraining, and advanced preference optimization |

## 10. What not to do next

- Do not add a fourteenth isolated subsystem.
- Do not run expensive full-corpus training before tiny-real-subset overfitting.
- Do not interpret synthetic cycle consistency as human comprehension.
- Do not merge incompatible sign-language datasets without linguistic mapping.
- Do not optimize rendering appearance before motion is understandable.
- Do not claim production readiness because latency or quantization formulas pass
  unit tests.
