# 12 — Phase 3C Evidence Integration and Lexical-Motion Governance

## 1. Verdict and scope

Phase 3C completes the **pre-artifact software boundary** for integrating the three
Phase-3 evidence routes. It does not claim that Phase 3 has passed empirically.

| Boundary | Current status | Exact meaning |
|---|---:|---|
| Phase 3C software possible without new artifacts | Complete | External-evidence identities, canonical lexical-motion manifests, strict reload, lookup abstention, and combined readiness assessment are implemented |
| Governed text/video→SIR corpus | Blocked | No accepted, independently validated, training-authorized project-human SIR corpus exists |
| Direct paired-learning result | Blocked | No eligible trained model or held-out intervention-score artifact exists |
| Human-approved lexical-motion library | Blocked | No canonical Phase-2 motion segment with independent meaning and articulation reviews exists |
| Phase-3 research exit | Not approved | Supervision, intervention, Phase-2 state, validation, and research-rights evidence is absent |
| Industrial path | Not approved | Commercial rights, production motion/rig evidence, and product validation are absent |

The canonical implementation is `signtranslator/planning/phase3c.py`. Its focused
regression stack is `tests/test_phase3c_evidence.py`. It is not connected to `run.py`, model
training, generation, or rendering because doing so before the source contracts exist would
introduce guessed semantics.

## 2. Non-substitution rules

Phase 3C keeps the following types distinct:

- planner retrieval embeddings are not motion;
- English text and uppercase lemmas are not gloss or SIR;
- publisher-native EAF values are not project-human SIR until Phase 3B review completes;
- frontal 137-node OpenPose tracks are 2D observations, not the canonical 3D state;
- generated, fitted, or interpolated values are not observed motion;
- a software test certificate is not qualified-ASL evidence or a rights grant.

The lexical-motion registry accepts only `canonical_phase2_multichannel_v1`. This name is a
gate, not proof that Phase 2 exists. Every entry must also bind the exact Phase-2 schema,
motion manifest, motion payload, ASL convention, SIR lexicon, source recording,
pseudonymous signer, authorization evidence, and independent human review. The review
repeats and binds the exact lexeme, convention, lexicon, Phase-2 schema, motion manifest,
and motion payload identities, preventing replay onto another entry.

## 3. Lexical-motion contract

Each entry has one non-negative SIR lexeme identifier and one form identifier. Multiple
reviewed variants may exist for one lexeme, but the library never silently picks among them.
Lookup returns exactly one of:

```text
resolved
unknown_lexeme
ambiguous_form
action_not_authorized
```

Only `resolved` carries a selected entry. The other three states abstain. There is no
spelling fallback, identity motion, nearest English word, first-candidate selection, or
generated substitute.

The library is non-empty and canonically ordered. Entry IDs, `(lexeme_id, form_id)` pairs,
and motion payload digests are unique. All entries share the exact convention, lexicon, and
Phase-2 schema. Canonical JSON loading rejects unknown fields, duplicate keys, non-finite
numbers, invalid UTF-8, oversized input, and any byte-level reformatting.

## 4. Human and external evidence

Lexical meaning and articulation are separate review artifacts. A qualified creator and a
distinct qualified reviewer are identified pseudonymously. The reviewer must attest to
directly viewing the motion. Qualification, independence, protocol, meaning review, and
articulation review each have independent SHA-256 identities.

Phase 3C recognizes four non-substitutable external roles:

1. canonical Phase-2 state;
2. independent qualified-ASL validation;
3. research-training rights;
4. commercial-training and deployment rights.

One artifact cannot satisfy multiple roles. Issuer and verifier must be distinct, and
verification must occur after issuance. These checks establish content binding and internal
chronology only. They do not prove a credential, linguistic judgment, or legal
interpretation; those remain independently reviewable external findings.

## 5. Readiness logic

The research exit is conjunctive:

\[
R = S \land V \land L \land P \land Q \land A_r,
\]

where (S) is an approved governed supervision batch, (V) is a passed four-intervention
paired-video dependence certificate, (L) is a non-empty reviewed lexical-motion library
with research authorization, (P) is matching Phase-2 state evidence, (Q) is independent
qualified-ASL validation, and (A_r) is research-training authorization. Missing evidence
is false; it is never imputed.

Industrial readiness additionally requires commercial evidence and commercial permission
for every registered entry:

\[
I = R \land A_c \land \bigwedge_{e\in L}\operatorname{commercial}(e).
\]

This is not full deployment readiness. Phases 4–7 retain separate motion-learning,
generation, human-validation, reliability, security, latency, rollback, and domain gates.

## 6. Explicit artifact-blocked implementation register

The runtime `work_items` field labels what cannot honestly be completed yet:

| Component | Software contract present | State | Required external input |
|---|---:|---|---|
| Governed corpus population | Yes | External-artifact blocked | Accepted Phase-3B SIR, qualified validation, research rights |
| Final direct paired model training | No | External-artifact blocked | Governed corpus, frozen Phase-2 state, signer/source-disjoint split |
| Direct paired falsification execution | Yes | Empirical-execution blocked | Trained model, held-out scores, preregistered threshold |
| Lexical-motion library population | Yes | External-artifact blocked | Canonical motion, meaning/articulation reviews, motion rights |
| Canonical Phase-2 motion export | No | Predecessor-phase blocked | Authorized multichannel 3D source and validated Phase-2 round-trip |
| Industrial training and deployment | No | External-artifact blocked | Commercial data/motion/rig rights and qualified product validation |

“Software contract present” means the verifier or registry path is implemented. It does not
mean the external activity happened. “No” records code that cannot be specified correctly
until its source contract is known; no placeholder class, dummy tensor, or mock approval
substitutes for it.

## 7. Regression and acceptance stack

Focused tests cover wrong representations; self-review; unqualified or unseen-motion review;
canonical reload and stable hashing; empty, reordered, duplicated, cross-schema, and reused
motion entries; explicit unknown, ambiguous, unauthorized, and resolved lookup; duplicate
JSON keys, non-finite values, type confusion, and byte bounds; external-role reuse and
chronology; failed supervision and interventions; Phase-2 schema mismatch; separate research
and industrial gates; and prohibition of gloss or English-label fields.

Final verification on 2026-09-14 produced:

- **28/28** focused Phase 3C adversarial tests passing under warnings-as-errors;
- **228/228** integrated Phase 3A/3B/3C, governance, grammar/planning, and paired-video
  pretraining tests passing under warnings-as-errors;
- **1,703/1,703** repository tests passing under warnings-as-errors;
- successful bytecode compilation of the package and focused test;
- a clean wheel build containing the Phase 3C module; and
- a successful isolated-wheel import and blocker-registry smoke test.

These results accept the pre-artifact software boundary against its declared contracts.
They do not accept any absent external artifact, linguistic mapping, motion observation,
trained model, empirical score, permission, or deployment claim.

## 8. Stop rule

The current honest output is `preartifact_software_boundary_complete = true`,
`research_exit_approved = false`, and `industrial_path_approved = false`.

Do not start Phase 4 from the first flag. Phase 3 may proceed empirically only after supplied
artifacts survive canonical reload, source/signer leakage checks, intervention testing, and
independent human review. A missing or conflicting artifact remains a blocker rather than a
request to synthesize, infer, or silently downgrade it.
