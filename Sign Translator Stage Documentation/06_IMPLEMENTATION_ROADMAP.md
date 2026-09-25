# 06 — Implementation Roadmap

## Execution-plan revision — 2026-09-25

This is the current proposed execution order and workload baseline. It supersedes old
priority lists that still treat repaired Phase-1 issues as unfinished. It does not claim
that an empirical phase passed, replace external authorizations, or change executable gate
behavior. The corresponding current audit contains B01–B66 in `03_READINESS_AUDIT.md`.

### Deliverable and scheduling assumptions

1. First deliverable: a narrow-domain, text→governed SIR→multichannel motion→deterministic
   avatar **research demonstrator**, with independently evaluated comprehension and explicit
   refusal outside supported scope. Include the non-manual channels needed for the chosen
   phenomena; do not defer grammar-critical face/gaze merely to make a manual-only demo.
2. Second deliverable: raw-audio input, streaming and a restricted operational release.
3. Third deliverable, if retaining the repository's full bidirectional goal: real-video
   ASL→English with its own uncertainty, comprehension and release evidence.
4. One dedicated engineer, **6 productive hours/day, 5 weekdays/week**. Code, debugging,
   validation, documentation and coordination all consume these hours. Overnight compute
   does not create additional engineering capacity. No unconfirmed GPU capacity is assumed.
5. A qualified ASL annotator and a distinct qualified reviewer work alongside the engineer;
   an adjudicator is available for disagreements. Their time is additional, not hidden in
   engineering estimates. If the engineer must also do coordination/annotation beyond the
   allowance, the schedule extends; engineering assistance cannot replace qualified review.
6. Conditional calendar starts **Monday 2026-09-28**. Dates count weekdays only and exclude
   no holidays, school commitments, leave or external delays. Add these explicitly before
   treating the calendar as a personal availability commitment.
7. Research source sample, exact permitted actions and usable rig evidence arrive by the
   end of Day 20; accepted pilot labels arrive by Day 50. These are scheduling dependencies,
   not predictions or assertions that access has been granted.
8. All phase lengths below already include debugging/rework days. External wait or scope
   expansion beyond the stated allowance is additional. A failed empirical experiment can
   require redesign; a timetable cannot guarantee a scientifically successful outcome.

### Resolve the phase logic before adding more modules

Keep the existing Phase 1–7 names, but track three columns for every phase: software
implemented, integrated on admissible inputs, and empirically accepted. They must never
collapse into one completion percentage.

- **Phase 1 refresh (W0):** retain achieved checkpoint/CTC/reproducibility work and repair
  newly found support, topology, analysis and statistical defects.
- **Phase 2 (W1–W2):** source contract → canonical multichannel state → validated round-trip.
- **Phase 3 (W3):** actual annotation/adjudication → governed SIR and lexical motion → direct
  training inputs and later intervention evidence. Existing 3A/B/C verifiers are reused.
- **Phase 4 (W4):** real motion representation and simple-baseline comparisons.
- **Phase 5 (W5):** conditioned generation integrated with deterministic rendering.
- **Phase 6 (protocol in W1, execution W6):** preregistration, independent statistical and
  qualified-human evaluation. Design it before training, not after observing results.
- **Phase 7 (W7–W9):** raw audio, streaming, hardware profiling and restricted release.
- **Reverse-direction extension (W10):** separately gated real-video ASL→English.

The previous pre-Phase-2 gate combines research and commercial prerequisites. Proposed
correction: define separate action-scoped research and commercial gates, consistent with
Phase-3C's existing separation. Research still requires source authorization, observed-state
contracts, qualified review and an authorized research rig. Commercial authorization stays
mandatory for commercial training/distribution/deployment. W0–W1 must explicitly reconcile
this policy in documentation and executable tests; until then, existing gates remain closed.
Do not silently bypass the portfolio gate or invent approval artifacts.

Authorized existing-2D QC, numerical repairs, split feasibility, statistical design and
source-independent interface experiments can proceed while access is unresolved. Freezing
source-dependent production semantics, populating approved libraries and claiming successful
multichannel training cannot. Pseudo-gloss is optional and off the critical path; adding a
pseudo-label generator does not supply missing independent references.

### Work packages, effort, dates and exit artifacts

“Build / debug” are engineer-days; debug includes regression, numerical investigation,
integration repair and explicit reruns. Specialist time and external waiting are separate.

| Package / legacy phase | Days | Conditional dates | Build / debug days | Hours | Required exit |
|---|---:|---|---:|---:|---|
| W0: baseline repair / Phase 1 refresh | 1–10 | Sep 28–Oct 9, 2026 | 7 / 3 | 60 | Reproduced defects fixed; warning-strict suite; honest empty-support/loader failures; canonical architecture and metric registry |
| W1: source/QC/split/protocol / pre-Phase 2 | 11–20 | Oct 12–23 | 8 / 2 | 60 | Accepted source/rig action contracts, grouped split, QC dispositions, pilot annotation protocol, preregistered estimands |
| W2: canonical state and round-trip / Phase 2 | 21–35 | Oct 26–Nov 13 | 11 / 4 | 90 | Typed real-state shards, units/frames/timestamps/masks, valid rotations/gaze, source→state→render round-trip |
| W3: supervised bridge / Phase 3 | 36–50 | Nov 16–Dec 4 | 11 / 4 | 90 | Qualified accepted pilot SIR, lexical motion, direct SIR loader, evidence bindings, no fabricated gloss |
| W4: representation learning / Phase 4 | 51–65 | Dec 7–25 | 11 / 4 | 90 | Tiny-real-subset fit, retrieval/interpolation/AE/RVQ comparison, held-out per-channel/minimal-pair report |
| W5: conditioned generation / Phase 5 | 66–85 | Dec 28–Jan 22, 2027 | 14 / 6 | 120 | Reloadable SIR→motion→avatar run, conditioning interventions, constraint/temporal tests and seed experiments |
| W6: independent validation / Phase 6 | 86–110 | Jan 25–Feb 26 | 18 / 7 | 150 | Blinded comprehension results, clustered uncertainty, baselines, coverage/refusal, failure taxonomy and research-demo go/no-go |
| W7: speech and streaming / Phase 7 | 111–130 | Mar 1–26 | 15 / 5 | 120 | Actual waveform→avatar, calibrated real errors, committed-prefix and interruption tests, target-hardware profile |
| W8: runtime/performance / Phase 7 | 131–140 | Mar 29–Apr 9 | 7 / 3 | 60 | Bounded service lifecycle, memory/load/queue profile, justified optimizations and parity evidence |
| W9: release hardening / Phase 7 | 141–150 | Apr 12–23 | 6 / 4 | 60 | Independent bundle reload, archive/wheel checks, monitoring/rollback rehearsal, domain and action-specific release gate |
| W10: reverse direction / extension | 151–175 | Apr 26–May 28 | 18 / 7 | 150 | Real-video ASL→English baseline, English decoding, scoped non-manual evidence, independent evaluation and separate acceptance |

Totals: **110 days / 660 engineer-hours** to the evaluated text research-demonstrator
candidate; **150 days / 900 hours** to the audio-enabled restricted-release candidate;
**175 days / 1,050 hours** including the bidirectional extension. The 150-day plan includes
42 debug days (252 hours); the 175-day plan includes 49 (294 hours). These are candidates
for acceptance, not promised successful or commercially authorized releases.

Planning ranges, not statistical confidence intervals: 120–210 engineer-days for the
forward/audio restricted product; 145–245 for the bidirectional scope. The shorter case
requires readily usable data/rigs, available reviewers and few redesigns. The longer case
allows source mismatch, representation/generator rework and extra human-evaluation rounds.
A new capture program is a separate project and can extend these ranges substantially.

### First ten days: daily execution plan

Each row budgets six productive hours. Source/reviewer preparation begins immediately;
any external messages or purchases still require the user's actual instruction.

| Day | Date | Work allocation | Deliverable |
|---|---|---|---|
| 1 | Sep 28 | 2h reproducible baseline; 2h scope/owners; 2h source and review requirements | Frozen audit, narrow scope, accountable owners and dependency ledger |
| 2 | Sep 29 | 4h statistical-domain/tail repair; 2h oracle/adversarial cases | NaN/Inf rejection and stable small-p tests |
| 3 | Sep 30 | 4h masked objectives and zero-support behavior; 2h gradient/support tests | Invalid targets cannot train motion objectives |
| 4 | Oct 1 | 4h topology/joint-order contracts; 2h graph/reload tests | 27-joint and alternate-topology behavior explicit |
| 5 | Oct 2 | 3h nonempty loader/nonfinite abort; 1h checkpointable CLI; 2h regression | No silent zero-step training or invalid optimizer update |
| 6 | Oct 5 | 4h length-aware/ragged analysis; 2h insertion/long-sequence cases | Evaluation accepts real variable-duration batches correctly |
| 7 | Oct 6 | 3h weighted estimands/selection; 1h analysis RNG; 2h unequal-batch tests | Reproducible named-checkpoint evaluation with correct support |
| 8 | Oct 7 | 2h canonical package mapping; 2h split/vocabulary design; 2h integration checks | Architecture/evaluation ownership and leakage rules |
| 9 | Oct 8 | 4h integration debugging; 2h complete strict regression | First repaired baseline candidate |
| 10 | Oct 9 | 3h archive/reload/smoke; 2h remaining repair; 1h go/no-go report | W0 acceptance or explicit extension with remaining defects |

This allocation is a target. If masked attention/pooling changes exceed W0, complete the
minimal loss safety fix and carry full encoder masking into W2–W3; record the remaining
integration gate rather than claiming all variable-length behavior solved.

### Subsequent five-day execution blocks

| Days | Concrete work and debugging focus |
|---|---|
| 11–15 | Inspect actual source/rig samples; reconcile research-vs-commercial gate; confirm reviewer workflow; disposition corruption; audit signer/source component balance. |
| 16–20 | Freeze QC on development data; create split and train-only preprocessing; define primary outcomes/independent units; approve source and schema requirements. Stop source-dependent work if evidence is missing. |
| 21–25 | Implement real canonical state, units/frames, availability/inferred provenance, rotation/gaze domains; validate numerical edge cases. |
| 26–30 | Build source adapters, synchronization, confidence/validity support, timestamp-aware derivatives and inverse transforms. |
| 31–35 | Round-trip real samples to authorized rig; inspect left/right, finger rotation, palm, face/gaze and timing; repair mismatches and freeze state v1. |
| 36–40 | Calibrate annotation on a small independently reviewed pilot; refine convention before bulk labeling; implement direct SIR shard/loader adapter. |
| 41–45 | Populate reviewed lexical segments and SIR records; measure disagreement/adjudication; bind rights/split/schema hashes to training manifests. |
| 46–50 | Integrate ragged masked batches and evidence checks; audit final pilot population and unseen forms; regression and Phase-3 input acceptance. Paired-learning empirical exit remains pending. |
| 51–55 | Profile data pipeline; fit simple retrieval/interpolation and tiny-subset continuous representation; debug observability and gradient flow. |
| 56–60 | Compare continuous AE and optional RVQ; inspect dead codes, duration error and channel losses; common-support baseline evaluation. |
| 61–65 | Repeat representation runs, test meaning-critical minimal pairs; freeze winning representation and abandon unnecessary complexity. |
| 66–70 | Integrate one generator with typed SIR and duration conditioning; overfit a tiny accepted subset; verify condition swaps affect output appropriately. |
| 71–75 | Add dense hands, required non-manuals, gaze/head/contact and projection; debug cross-channel timing and gradient imbalance. |
| 76–80 | Add inverse transforms and deterministic render command; run blank/shuffled/order-corrupted/text-only and field intervention experiments. |
| 81–85 | Multi-seed generation, ablations, reload parity, seams/collision/minimal-pair repair; freeze evaluation candidate and manifests. |
| 86–90 | Complete blinded study setup and stimulus QA; pilot comprehension tasks and reviewer instructions without opening final test outcomes. |
| 91–95 | Execute independent reviews and baselines; collect proposition/phenomenon scores, failure severity and disagreement. |
| 96–100 | Cluster-aware analysis, calibrated refusal and subgroup diagnostics; debug study data/metric joins and report uncertainty. |
| 101–105 | One budgeted remediation/retraining round on development data; add regression cases for observed failure classes. |
| 106–110 | Evaluate remediated candidate on a fresh reserved evaluation set; publish research-demo acceptance or a concrete failed-gate report. Reusing the previous test to tune is not a clean confirmatory result. |
| 111–115 | Integrate waveform preprocessing and a versioned real ASR/backend; sample-rate/timestamp/tokenizer contract checks. |
| 116–120 | Gather/evaluate eligible real acoustic stress cases; calibrate semantic risk and refusal on a calibration split. |
| 121–125 | Connect revisable output, commit boundary, interruptions, cancellation, stale-result disposal and queue policy to rendering. |
| 126–130 | Profile end-to-end latency/memory under load; debug streaming seams and actual device failures; accept speech/streaming integration. |
| 131–135 | Service lifecycle, request bounds, privacy/retention, trace identities and failure recovery; realistic load replay. |
| 136–140 | Optimize measured bottlenecks only; numerical and signer-relevant parity checks; rehearse overload behavior. |
| 141–145 | Package model/preprocessors/state/rig/rights and evaluation manifests; clean-machine/archive/wheel reproduction; target OS/device checks. |
| 146–150 | Final regression, rollback rehearsal, domain/coverage documentation and separate research/commercial authorization review. No automatic deployment. |
| 151–155 | Reverse-direction real-video perception and multichannel input contract; sentence/source grouping and English evaluation design. |
| 156–160 | Train sign-state→English baseline; preserve non-manual temporal evidence and uncertainty. |
| 161–165 | Held-out comparison, ablations and real-video degradation tests; diagnostic gloss remains optional. |
| 166–170 | Independent English semantic recovery evaluation, calibrated refusal and failure-driven repair. |
| 171–175 | Fresh reserved evaluation, bundle/reload parity, integration regression and bidirectional acceptance decision. |

### Mathematical implementation acceptance

| Contract | Implementation requirement | Falsification/exit evidence |
|---|---|---|
| Masked losses | `L_k = sum(m*w*ell)/sum(m*w)` over declared observed support. If denominator is zero, return unavailable or reject a required objective; never a successful zero loss. Specify whether normalization is per sample or per valid observation. | Invalid targets cannot affect loss/gradient; all-invalid and partially observed batches tested; confidence is not assumed calibrated. |
| Time | Use `(x[t+1]-x[t])/(time[t+1]-time[t])`; both endpoints must be supported. Define acceleration on nonuniform intervals and reject nonpositive deltas. | Resampling-invariance/timestamp-jitter tests; no derivative crosses a missing interval without declared inference. |
| Geometry | Continuous 6D regression mapped to SO(3); evaluate geodesic rotation error with stable near-zero/near-pi handling. Require orthogonality and determinant +1, unit gaze, declared local/global frames and physical units. | Degenerate axes, reflections, unit conversion, handedness and round-trip tests on synthetic edge cases AND real source samples. |
| Kinematics | Preserve articulated finger/palm/head/face/contact structure. Joint limits and collision corrections are constraints whose linguistic effects must be checked. | Forward kinematics/retarget parity; handshape/orientation/contact minimal pairs survive projection and rendering. |
| CTC | Retain `T_after_subsample >= U + adjacent_repeats`; blank/out-of-range labels fail. | Variable-length and repeated-token cases stay green across new backend adapters. |
| Contrastive learning | Typed multi-positive relation and permitted negatives; no test-derived lexicon; measure repeated-meaning false negatives. | Duplicate-meaning batches, signer/source controls and text-only/video interventions. |
| Representation | Retrieval/interpolation → continuous AE → optional RVQ; quantify per-part distortion, temporal/spectral error, code utilization and duration. | Common-support/coverage comparisons and held-out minimal pairs, not training loss alone. |
| Generation | One chosen diffusion/flow parameterization with consistent schedules/targets/guidance; condition, padding and inpainting masks propagate end-to-end. | Tiny overfit, condition swaps, null/shuffle tests, per-channel constraints, seed/reload tests. No advanced sampler adopted solely because its module exists. |
| Numerical optimization | Abort nonfinite loss/gradients before stepping; monitor branch gradients; introduce EMA/AMP/accumulation only with validated step/resume semantics. | Deliberate NaN/Inf injection, no-step loader, freeze/unfreeze and interrupted-run tests. |

### Statistical implementation acceptance

- Define population, experimental unit, primary endpoint, direction, minimum useful effect,
  confidence level, exclusion rules and comparison family before held-out evaluation.
- Split source and signer groups before windows; fit normalization/vocabulary/calibration
  only on their assigned partitions. A fixed external lexicon is allowed if declared in advance.
- Repair all general statistical input validation and small-tail numerics first. Maintain
  exact/log-probability reference cases; `NaN` never becomes a passing p-value.
- Repeated clips from one source/signer do not automatically create independent sign-test
  trials. Aggregate at the preregistered independent unit or use a defensible hierarchical
  resampling/model. Preserve pairing in interventions. For the existing four-comparison
  Bonferroni family at alpha .05, each threshold is .0125; practical effect remains required.
- Compute uncertainty at the correct source/signer/reviewer levels; separately report
  training-seed variability. Plan at least three training seeds initially, but do not call
  three seeds sufficient evidence for population generalization.
- Compare reconstruction methods on common eligible targets and publish coverage. Zero
  baseline support is unavailable. Report missingness and excluded-source sensitivity.
- Measure inter-reviewer agreement plus confusion, field-level support, temporal tolerance
  and adjudication rate; agreement alone cannot establish linguistic correctness.
- Human semantic proposition recovery and phenomenon-level correctness are primary;
  WER/SignBLEU, coordinate error, cycle score and appearance metrics are complementary.
- Select rejection thresholds on calibration data; report selective risk and coverage
  together on held-out data, including OOD/empty evidence and subgroup uncertainty.
- Estimate sample size from a reviewed pilot, meaningful effect and cluster correlation.
  For orientation only, an IID binary proportion near .5 needs about 384 independent
  observations for an approximate 95% interval with ±.05 margin; clustered observations
  need additional design analysis. This is not a prescription for 384 clips or reviewers.
- Keep a reserved final evaluation set for the budgeted remediation round; do not tune
  repeatedly on the same held-out result while still calling it confirmatory.

### Logical and integration acceptance

Use distinct states: implemented, artifact-blocked, empirically failed, accepted-for-research,
and authorized-for-commercial-action. Missing evidence is not “pass with caveat.”

The existing Phase-3 research gate remains the conjunction of accepted supervision,
video-dependence evidence, reviewed lexical motion, canonical state, independent qualified
validation and research rights. Full release additionally needs representation/generation,
human validation, reliability and runtime evidence. Commercial release adds action-specific
commercial permissions. Wire accepted identities into actual training and inference, not
just a disconnected reporting function. Preserve unknown/ambiguous/unauthorized refusal;
reject schema mismatch, stale evidence and unauthorized action. A digest is content identity,
not proof of issuer authority. Streaming cannot revise the committed prefix; queue bounds
must not be advertised as latency bounds without measured service/scheduling assumptions.

### External work and workload budget

| External lane | Target | Effort placeholder / measurement rule | Failure response |
|---|---|---|---|
| Source access and rig | Exact sample/schema/action evidence by D20 | Engineer preparation is budgeted in W0/W1; response/negotiation time is unknown | Continue independent repairs/QC; delay W2 rather than inventing state evidence |
| Qualified ASL team | Available during W1, pilot convention before W3 bulk work | Two distinct reviewers plus adjudicator; confirm actual availability | Annotation and human exit dates slide |
| Pilot annotation | Accepted pilot by D50 | Initial budgeting example: 300 short items × (12 min primary + 8 min review + .25×10 min adjudication) = 112.5 specialist hours | Calibrate these rates on 10 items, then resize; 300 is a workload example, not an adequate statistical sample claim |
| Specialist total | Annotation, calibration, lexical articulation checks, study design/reviews and adjudication | Reserve roughly 200–350 specialist hours across forward phases; estimate is provisional until pilot timing | At 16 combined specialist hours/week this alone occupies 12.5–22 weeks, overlapping engineering |
| New motion capture if required | Separate scoped acquisition project | Recruitment, consent, calibration, recording, synchronization and review need a new work breakdown | Rebaseline; do not bury capture in a 10-day ingestion estimate |
| Compute | Profile accepted mini-corpus in W4 | `training_hours = epochs * ceil(N/B) * measured_step_seconds / 3600`, then add validation, preprocessing, runs/seeds and restart overhead | If scheduled experiment set exceeds allocated wall time, reduce scope or add explicitly authorized capacity |
| Commercial evidence | Before any commercial action | Unbounded external dependency; research access is not commercial authorization | Restrict to authorized research; no promised commercial release date |

Reforecast every five workdays using actual completed artifacts, defect counts, measured
training time and annotation throughput. At D10, D20, D35, D50, D65, D85, D110, D130,
D150 and D175 issue a gate decision. For an unmet dependency, recompute each successor's
start as `max(predecessor completion, required artifact arrival, engineer availability)`.
Do not simply add all parallel waiting periods or pretend waiting consumes no calendar.

For personal scheduling: 900 productive hours at 15 hours/week is approximately 60 weeks
for the forward/audio candidate; 1,050 hours at that rate is 70 weeks for bidirectional,
before external delays. The six-hour-day calendar therefore requires dedicated capacity.

### Deliberately deferred work

Photorealistic NeRF/Gaussian appearance, large foundation pretraining, production pseudo-gloss,
preference optimization and distributed training are not prerequisites for the first scoped
research demonstrator. Reconsider only after a measured bottleneck and accepted evidence
justify them. They are not secretly included in the 150/175-day totals. Open-domain fluency,
interpreter replacement and unrestricted commercial deployment have no credible completion
date from the present evidence.

---

## Earlier roadmap and design rationale (historical; current sequencing above)


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

**Phase-3 software-boundary status (2026-09-12): implemented; empirical exit not
approved.** The repository now provides three fail-closed mechanisms that can accept
future evidence without manufacturing it:

1. a canonical, hash-bound human SIR annotation envelope restricted to
   `official_human` and `project_human`, with a frozen ASL convention and lexicon,
   exact video/transcript/provenance/authorization bindings, distinct qualified
   annotator and reviewer attestations bound to the exact reviewed content plus hashed
   qualification/independence evidence, and signer/source split-leakage checks;
2. strict, versioned SIR parsing and content hashing that reject unknown fields,
   malformed identifiers, non-finite or contradictory intervals, invalid edges,
   duplicate graph elements, and noncanonical payload bytes; and
3. a held-out paired-video dependence evaluator requiring aligned performance to
   exceed blank-video, shuffled-video, order-corrupted-video, and text-only
   interventions under a preregistered positive effect threshold, exact paired sign
   tests, and four-comparison familywise correction.

These mechanisms do not generate annotations, train a language model, produce motion,
or establish ASL correctness. The following blockers are deliberately recorded as
**unsolved**: governed qualified text/video→SIR supervision; a passed direct paired-
learning falsification; a human-approved lexical motion library; the canonical Phase-2
multi-channel state; independent qualified-ASL validation; and commercial training and
deployment rights. No downstream Phase-3/4/5 implementation may treat the software
boundary as evidence that any of those blockers has been resolved.

**Phase-3A linguistic-reference ingestion status (2026-09-13): secure software and one-item
Cokely source compatibility verified; linguistic approval not started.** The fail-closed EAF
reader preserves publisher-native tiers, untimed slots, and reference graphs; binds the exact
EAF, publisher page, primary HD media, and auxiliary SD media; and emits compact canonical
inspection artifacts without creating SIR or training labels. Its 27 focused adversarial
tests and the complete 1,640-test repository suite pass under warnings-as-errors. The
authentic *I Have a Dream* pilot contains 751 source annotations, nine of which have at least
one publisher-unspecified endpoint that remains uninterpolated. Its security contract,
hashes, and qualified-human mapping protocol are in
`10_PHASE_3A_LINGUISTIC_REFERENCE_INGESTION.md`. This bounded result does not establish
corpus-wide compatibility, project SIR validity, ASL correctness, commercial rights, or
readiness for training.

**Phase-3B governed-mapping software status (2026-09-13): implemented; qualified-human
execution not started.** Exact EAF source annotations can now enter a label-empty,
hash-indexed review queue governed by seven real artifacts and a machine-verifiable sampling
plan. Primary and blind-review SIR submissions use an explicit EAF-millisecond timebase;
disagreement requires third-person adjudication; per-field, temporal, event-coverage, and
edge diagnostics retain exact support; and a deterministic batch audit cannot authorize
training, commercial use, or linguistic validity. The complete contract and remaining human
evidence gates are in `11_PHASE_3B_GOVERNED_MAPPING_AND_ADJUDICATION.md`.

**Phase-3C pre-artifact software status (2026-09-14): implemented; empirical Phase-3 exit
not approved.** The project now has a distinct canonical lexical-motion registry, strict
external-evidence identities, fail-closed unknown/ambiguous/unauthorized motion lookup, and
a conjunctive readiness assessment over governed supervision, paired-video intervention
evidence, the canonical Phase-2 state, independent qualified-ASL validation, and separate
research/commercial rights. The runtime report also labels every implementation deferred
because its real source contract is absent. See
`12_PHASE_3C_EVIDENCE_INTEGRATION.md`. This does not authorize Phase 4.

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
| Governed supervision envelopes and paired-video dependence evaluation | Actual qualified human SIR annotations and a trained model passing the intervention gate |
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
