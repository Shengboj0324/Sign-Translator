# 06 — Implementation Roadmap

## 1. Roadmap principle

The next phase should optimize for a **complete vertical slice**, not additional
package count. Every milestone must produce an artifact that can be inspected,
reloaded, and falsified.

## 1.1 Research-to-deployment program without a required pseudo-gloss model

The deployed product surface does not require gloss: it requires a governed input,
an inspectable linguistic plan, comprehensible multi-channel signing motion, a renderer,
and calibrated refusal. Pseudo-gloss remains one optional weak-supervision experiment;
it is not placed on the critical path. Removing it does not justify a promise of equal
performance. Capability is preserved architecturally, while performance equivalence must
be established by the same held-out and qualified-signer gates as every other route.

The replacement program is divided into the following consecutive phases.

### Phase 1 — reproducible mathematical execution

Implement before any new learned representation:

1. content-addressed implementation identity that remains valid outside Git;
2. distinct `best` and `last` checkpoints with atomic writes and verified sidecars;
3. model/config/corpus/normalization/vocabulary/code bindings in each checkpoint;
4. exact optimizer, scheduler, epoch, history, and Python/NumPy/Torch RNG restoration;
5. deterministic, RNG-isolated validation;
6. exact CTC feasibility, including mandatory blank frames between adjacent repeats;
7. explicit failure instead of `zero_infinity=True` suppression;
8. clean-source-archive CI, checkpoint reload, and deterministic-output comparison.

Exit evidence is a bit-identical interrupted-versus-uninterrupted CPU training test,
repeatable validation that does not advance the training RNG, adversarial checkpoint
tamper/configuration tests, archive execution without `.git`, and the complete
warning-strict suite. This phase is gloss-independent.

**Phase-1 status (2026-09-10): complete within the boundary above.** The source
checkout and a separate no-Git archive each pass all 1,575 tests under warnings-as-errors.
The targeted changed modules pass static type analysis; compilation, YAML parsing, and
diff validation pass. An installed wheel records content-only provenance without inventing
a Git revision, and its trained checkpoint reloads to bit-identical seeded inference.
The resume test proves bit-identical four-epoch CPU results after a two-epoch interruption,
including parameters, optimizer moments, scheduler, history, and step count. These results
certify the Phase-1 software contracts—not ASL validity, real-data model performance, or
deployment readiness.

### Pre-Phase-2 data-evidence gate — 2026-09-10

The official pseudonymous How2Sign signer code is now certified against the immutable
31,165-row audit and the CVPR supplemental counts: all 31,047 available clips reconcile
exactly, and the 118 missing rows reconcile by signer. This resolves the grouping-key
ambiguity, but no final signer-and-`VIDEO_ID`-disjoint split has been created.

The executable source-portfolio gate requires each modality bundle to be co-observed in
one locally verified, authorized source. It currently rejects all five pre-Phase-2 bundles:
continuous 3D research data, continuous linguistic reference, commercial training rights,
a qualified commercial render rig, and qualified-ASL governance. Publisher claims and
capabilities scattered across datasets cannot pass. The exact evidence, acquisition routes,
secure source treatments, and correspondence register are in
`04_DATA_ENGINEERING_AND_CORPUS.md` §§10–13.

**Phase 2 has not begun.** Its schema can be designed only after these empirical source
contracts are known; otherwise the project would risk freezing invented coordinate,
confidence, gaze, or availability semantics into the architecture.

### Phase 2 — canonical observable and generative sign state

Freeze one typed, time-indexed multi-channel representation before training:

\[
Y_t=(R_t^{body},R_t^{left\ hand},R_t^{right\ hand},R_t^{head},
     r_t,f_t,g_t^{left},g_t^{right},b_t,c_t),
\]

where articulated rotations use a continuous 6D parameterization during regression and
SO(3) geodesic error during evaluation; `r` is root translation; `f` is a declared facial
coefficient/action-unit vector; each gaze vector lies on the unit sphere; `b` carries
blink/eyelid state; and `c` carries declared contact or classifier channels. Handshape is
represented by articulated finger rotations/contact, palm orientation by wrist/palm
frames, movement by nonuniform-time derivatives, posture by the body kinematic tree,
and head movement by its own SE(3) channel. These quantities must not be collapsed into
one undifferentiated Cartesian loss.

Every channel carries `observed`, `valid`, `confidence`, `source`, coordinate-frame, and
timestamp fields. A loss is evaluated only on its declared support:

\[
\mathcal L_k=\frac{\sum_{t,j}m^{(k)}_{t,j}w^{(k)}_{t,j}
\ell_k(\hat Y^{(k)}_{t,j},Y^{(k)}_{t,j})}
{\sum_{t,j}m^{(k)}_{t,j}w^{(k)}_{t,j}},
\]

and is unavailable—not zero—when the denominator is zero. Confidence may be used as a
weight only under a declared interpretation; it is not called calibrated probability
without calibration evidence. Phase 2 ends only after source→state→inverse-transform→
render round-trips and left/right, gaze, palm, face, head, and temporal tests pass.

The current frontal 2D OpenPose release cannot uniquely determine this state. Monocular
depth, self-occluded fingers, palm twist, subtle facial action, and gaze are non-identifiable
in many frames. Those channels require licensed multi-view/depth evidence, a validated
fitting model, project-human correction, or an explicit unavailable mask—never synthesis
presented as observation.

### Phase 3 — direct governed language-to-SIR supervision

Replace the pseudo-gloss dependency with three non-substituting evidence routes:

1. qualified project-human text/video→SIR annotations under a frozen ASL convention;
2. direct text/video contrastive and temporal learning on authorized paired media, with
   transcript-only and blank/shuffled-video falsification tests;
3. a versioned lexical motion library for forms that have human-approved meaning and
   articulation, with `UNKNOWN` and abstention for uncovered forms.

Authentic gloss may later enter as an auxiliary observation, but no phase depends on a
full-corpus pseudo-gloss model. English words, uppercase lemmas, retrieval IDs, latent
visual tokens, and SIR fields remain distinct types. Phase 3 ends only when plan fields
are reference-backed, source-disjoint, intervention-sensitive, and independently reviewed.

### Phase 4 — multi-channel motion representation learning

Train retrieval/interpolation, continuous autoencoder, and residual-token baselines on the
Phase-2 state. Evaluate hands, palms, body, face, gaze, head, contact, velocity, acceleration,
and temporal scope separately. A learned representation must beat declared simple baselines
on held-out source groups and preserve meaning-critical minimal pairs before a generator is
authorized.

### Phase 5 — constrained text/SIR-to-motion generation

Train body, hand, face, gaze, and head branches in a staged curriculum, then integrate them
through typed SIR conditioning, cross-channel constraints, uncertainty, and abstention.
Require conditioning swaps, field interventions, ablations, multi-seed stability, exact
reload, and deterministic rigged rendering. A low aggregate pose loss is not a pass.

### Phase 6 — linguistic and human validation

Use preregistered, blinded evaluation with qualified ASL signers. Primary endpoints are
semantic proposition recovery and phenomenon-level accuracy; motion and rendering metrics
remain diagnostics. Failures in handshape, orientation, movement, posture, facial grammar,
gaze, head movement, reference, or timing are reported independently rather than averaged
away.

### Phase 7 — restricted product and industrialization

Integrate raw text/audio, rendering, calibrated refusal, traceable model/data identity,
privacy controls, observability, rollback, canaries, and target-hardware load tests. Release
first as a restricted-domain research demonstrator. Commercial deployment additionally
requires data/model/rig rights that do not depend on How2Sign's noncommercial permission.
Medical, legal, emergency, and interpreter-replacement claims remain prohibited until
separately validated and authorized.

### Current implementability boundary

| Can be implemented now | Requires external evidence or assets |
|---|---|
| Phase-1 reproducibility, CTC, checkpoint, and archive controls | Linguistically valid text/video→ASL plans |
| Typed multi-channel schemas and observability rules | Qualified-ASL annotation and blinded review |
| 2D source QC and source-disjoint experimentation; certified pseudonymous signer key | Final signer-and-source-disjoint split and certificate |
| Direct paired text/video representation baselines | Identifiable 3D hands, palms, face, gaze, and head state |
| Render/fitting interfaces and synthetic geometric proofs | Licensed production body/face/rig assets |
| Security, provenance, abstention, and deployment contracts | Commercial data and derivative-model rights |

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

### Current gloss-free preparation result (updated 2026-09-10)

The canonical v1 full-corpus audit and the quarantined 2D masked-reconstruction
experiment are complete and reproducible. The audit accounts for all 31,165 metadata
rows plus one orphan artifact without fabricating the 118 missing clips or repairing
the three structural failures. Review queues and `VIDEO_ID` source constraints are
available, but they are not review attestations. The official pseudonymous signer code
is now certified by exact reconciliation with the How2Sign supplemental; it is not a
person's identity. The 2D experiment
is intentionally disconnected from the exporter, active runtime, 6D motion tokenizer,
and Stage C; its held-out point/span model did not beat temporal interpolation.

Detailed evidence is recorded in `docs/DATA_ENGINEERING.md`. The pseudo-gloss
candidate-lattice implementation and its activation requirements are consolidated in
`09_PSEUDO_GLOSS_MODEL_RESEARCH.md`. The software path is implemented, but corpus-wide
generation remains gated on a versioned ASL lexicon/convention, independent qualified
human references, calibrated pretrained weights, frozen preregistration, and partitions
disjoint by both signer and source when generalization is claimed.

**Stage B remains unapproved.** Authentic gloss, a final signer-and-source-disjoint split,
qualified signer review, co-observed production 3D channels, and commercial authorization
are absent. English transcripts, pseudo-glosses, review-queue generation, 2D fits, and
cross-corpus modality composition cannot substitute for those gates. Stage C therefore
remains blocked.

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
