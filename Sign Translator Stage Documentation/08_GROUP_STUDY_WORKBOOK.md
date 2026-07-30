# 08 — Group Study Workbook

## 1. How to use this workbook

Each session should have:

- one facilitator;
- one code navigator;
- one skeptic responsible for challenging claims;
- one recorder;
- one concrete artifact or decision at the end.

Do not end a session with “we understand it better.” End with a diagram,
experiment specification, interface contract, risk register, or assigned change.

## 2. Eight-session study plan

### Session 1 — Capability versus aspiration

Read:

- `01_PROJECT_OVERVIEW.md`
- `03_READINESS_AUDIT.md`

Questions:

1. Which claims describe implemented code?
2. Which describe the active pipeline?
3. Which require real-data evidence?
4. What is the narrowest defensible project description?

Deliverable:

- a one-page capability statement with approved and prohibited wording.

### Session 2 — Trace the executable path

Read:

- `02_CURRENT_ARCHITECTURE.md`

Inspect:

- `signtranslator/run.py`;
- `signtranslator/models/pipeline.py`;
- `signtranslator/training/trainer.py`;
- `signtranslator/analysis/report.py`.

Exercise:

Trace one batch from disk through every tensor transformation. Record shapes,
masks, label conventions, losses, and output units.

Deliverable:

- a reviewed active-path diagram and interface table.

### Session 3 — Choose the target problem

Read:

- sections 1–3 of `04_DATA_ENGINEERING_AND_CORPUS.md`.

Questions:

1. Which sign language and dialect are in scope?
2. Is the first research target text-to-sign or speech-to-sign?
3. What is the exact output representation?
4. Which user setting is explicitly out of scope?

Deliverable:

- a dataset and task charter.

### Session 4 — Data and governance

Read:

- the remainder of `04_DATA_ENGINEERING_AND_CORPUS.md`.

Exercise:

Take three hypothetical samples and determine:

- whether acquisition is permitted;
- their split groups;
- required annotations;
- confidence and missingness representation;
- provenance records.

Deliverable:

- canonical sample schema version 1 and a data risk register.

### Session 5 — Recognition and alignment

Read:

- sections 2–6 of `05_MODEL_AND_MATHEMATICS_STUDY_GUIDE.md`.

Exercises:

1. Construct a CTC target with repeated adjacent labels and calculate its minimum
   feasible input length.
2. Construct a batch with two semantically equivalent glosses and show why
   diagonal InfoNCE creates a false negative.
3. List potential signer, camera, and duration shortcuts.

Deliverable:

- revised CTC and contrastive-learning requirements.

### Session 6 — Generation and rendering

Read:

- sections 7–9 of `05_MODEL_AND_MATHEMATICS_STUDY_GUIDE.md`.

Questions:

1. Can 27 Cartesian joints express the target language phenomena?
2. Which constraints belong in training, sampling, or post-processing?
3. What must the avatar preserve for linguistic correctness?
4. Which rendering metrics are irrelevant to comprehension?

Deliverable:

- canonical motion representation decision record.

### Session 7 — Experiments and evaluation

Read:

- `07_TRAINING_EVALUATION_DEPLOYMENT.md`.

Exercise:

Design one experiment comparing direct gloss conditioning with SIR conditioning.
Specify:

- hypothesis;
- train/validation/test groups;
- baseline;
- primary endpoint;
- minimum meaningful effect;
- confounds;
- stopping and decision rules.

Deliverable:

- pre-registration draft.

### Session 8 — Roadmap commitment

Read:

- `06_IMPLEMENTATION_ROADMAP.md`.

Exercise:

For every Stage A–C task, assign:

- owner;
- dependency;
- artifact;
- verification command;
- reviewer;
- exit criterion.

Deliverable:

- an implementation board containing only work needed for the first vertical
  slice.

## 3. Code-reading checklist

For each module:

- What are its input and output units?
- Does it handle batch, time, and padding masks?
- Is uncertainty represented?
- Is the implementation used by `run.py`?
- Does its test use real, synthetic, random, or hand-constructed data?
- What property does the test prove?
- What real-world claim does it not prove?
- Is the same concept implemented elsewhere?
- What happens when the modality is missing?
- Can the module fail closed?

## 4. Claim-evidence exercise

Complete this table during study:

| Claim | Code evidence | Test evidence | Real-data evidence | Human evidence | Status |
|---|---|---|---|---|---|
| Speech confidence predicts error |  |  |  |  |  |
| Planner produces valid grammar |  |  |  |  |  |
| Generated motion preserves meaning |  |  |  |  |  |
| Hands are readable |  |  |  |  |  |
| Non-manual scope is correct |  |  |  |  |  |
| Avatar output is understandable |  |  |  |  |  |
| Streaming is safe under revision |  |  |  |  |  |

Use only:

- **Established**
- **Partially established**
- **Not established**
- **Falsified**

## 5. Architecture decision record template

```markdown
# ADR: [Decision title]

## Status
Proposed / Accepted / Rejected / Superseded

## Context
What problem requires a decision?

## Options
1. Option A
2. Option B
3. Option C

## Decision criteria
- linguistic adequacy
- mathematical validity
- data availability
- training stability
- runtime cost
- licensing

## Decision
What was selected?

## Evidence
Which experiments, code paths, or external constraints support it?

## Consequences
What becomes easier, harder, or impossible?

## Falsification condition
What result would cause the team to revisit this decision?
```

## 6. Experiment review template

```markdown
# Experiment: [Name]

## Hypothesis

## Dataset version and split certificate

## Model and checkpoint format version

## Baseline

## Primary endpoint

## Secondary diagnostics

## Minimum meaningful effect

## Seeds and statistical units

## Known confounds

## Result

## Failure slices

## Decision

## Reproduction command and artifact hashes
```

## 7. Weekly review questions

1. What new capability became integrated this week?
2. What claim was weakened or falsified?
3. Did any test become less informative while remaining green?
4. Did any data transformation lose uncertainty or provenance?
5. Are train and evaluation paths using the same representation contract?
6. Is the best checkpoint independently loadable?
7. Did the team add infrastructure or merely another isolated primitive?
8. What is the single highest-risk assumption for next week?

## 8. Definition of group understanding

The group understands the system when every member can:

- trace a sample through the active pipeline;
- explain the difference between gloss, sign plan, SIR, motion, and rendering;
- state why synthetic success is insufficient;
- identify the canonical implementation of each active component;
- explain the chosen motion representation and losses;
- describe how leakage is prevented;
- interpret every primary evaluation metric;
- state the precise conditions blocking deployment.

