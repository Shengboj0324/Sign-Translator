# LLM Semantic Reasoning Layer — design, mathematics, and staged plan

Implements `02_llm_semantic_reasoning_layer.md`. As in the speech layer, the
mathematics is fixed *before* the code, so every component has a stated property
a test can falsify. Anything asserted here without a corresponding test is a
defect.

---

## 1. Role and interface

The LLM is a **constrained semantic planner**, not a factual oracle and not the
motion generator. Its input `x` is the evidence bundle:

| Evidence | Source |
|---|---|
| transcript hypotheses / lattice | speech layer (Stage 02 of `01_*`) |
| acoustic states | speech-layer `SpeechProjector` output |
| discourse memory | running referent state |
| target-language metadata | which sign language, register |
| lexicon / retrieval result | versioned sign lexicon `D(x)` |

Its output `s` is a **typed sign plan** with explicit uncertainty — never
free-form gloss text as the sole interface. The plan then supplies the gloss /
conditioning the existing motion generator consumes.

## 2. Typed plan schema

Following the document's suggested schema, `SignPlan` carries:

```
utterance_semantics : predicate + [(role, referent)]     (semantic frame)
discourse_referents : {referent_id}                       (declared entities)
time / topic / focus: tense-aspect marker, topic ref, focus ref
manual_units        : ordered [lexical_id]                (the signs)
spatial_loci        : referent -> discrete locus          (LEFT/RIGHT/CENTER/...)
classifier_constructions : optional [classifier predicate]
nonmanual_scope     : [(marker, start_unit, end_unit)]    (NEG/WH/Y-N/TOPIC over spans)
fingerspelling      : subset of manual_units              (out-of-lexicon items)
confidence          : float in [0, 1]
provenance          : where each field came from          (hyp id, lexicon version)
```

### 2.1 Structural validity (the "structural validation" tests)

A plan is **valid** iff *all* of the following hold. Each is one test.

1. Every referent used in `utterance_semantics` args is declared in
   `discourse_referents`. *(referent consistency)*
2. `topic` and `focus`, when set, are declared referents.
3. Every declared referent has a `spatial_loci` entry, and loci are drawn from
   the fixed locus alphabet. *(no invalid spatial references)*
4. Distinct referents occupy distinct loci (a locus identifies a referent in
   signing space; collisions are ill-formed).
5. Each `nonmanual_scope` span `(m, i, j)` satisfies
   `0 <= i <= j < len(manual_units)`. *(scope bounds)*
6. `fingerspelling ⊆ indices(manual_units)`.
7. `0 <= confidence <= 1`.
8. Every manual unit is either a lexicon entry or fingerspelled. *(the
   hallucination rule — checked against `D` in Stage 2c)*

`validate(plan)` returns the list of violated rules; a plan is valid iff the
list is empty.

## 3. Serialization and the schema automaton

### 3.1 Regular serialization

The plan is linearized to a **token sequence** over a finite vocabulary with a
*fixed slot order* and typed value alphabets:

```
BOP  PRED <p>  ARGS (<role> <ref>)*  REFS <ref>*  TAM <t>
LOCI (<ref> <locus>)*  UNITS <lex>*  NMS (<marker> <i> <j>)*
FS <i>*  CONF <bucket>  EOP
```

Because the slot order is fixed and each slot's value alphabet is finite and
independent of history depth, the set of *valid next tokens* depends only on a
finite **slot state** — so the serialization language is **regular** and is
recognised by a DFA (§3.2). (Cross-slot consistency constraints such as "this
ref was declared" are *not* encoded in the DFA; they are enforced by the
validator of §2. The DFA guarantees a well-*formed skeleton*; the validator
guarantees a well-*formed plan*. Conflating the two would be a category error.)

### 3.2 Schema automaton

A DFA `A = (Q, Σ, δ, q0, F)` over the serialization vocabulary `Σ`. State `q`
encodes which slot is being filled and the position within it. `allowed(q) =
{v ∈ Σ : δ(q, v) is defined}`.

**Properties, all proved:**

* **Soundness.** Every string accepted by `A` deserializes without error to a
  structurally *well-formed skeleton* (correct field types and arities). Proved
  by construction and checked by round-tripping every accepted string in an
  exhaustive small-grammar enumeration.
* **Liveness (no dead ends).** For every reachable non-accepting state `q`,
  `allowed(q) ≠ ∅`. Hence constrained decoding can always take a legal step and
  never gets stuck before `EOP`. Proved by reachability enumeration.
* **Determinism.** `δ` is a partial function, so a given string drives `A`
  through at most one path; acceptance is unambiguous.

### 3.3 Constrained decoding (the provable core)

At decode step `t` the model emits logits `z ∈ R^{|Σ|}`; the automaton supplies
`A_t = allowed(q_t)`. The **masked distribution** is

```
p'(v) = p(v) / Z      for v ∈ A_t,     0 otherwise,
Z = Σ_{u ∈ A_t} p(u),   p = softmax(z).
```

This is exactly the conditional `p(v | v ∈ A_t)`. Three facts, all tested:

1. `p'` is a probability distribution (non-negative, sums to 1) whenever
   `A_t ≠ ∅` — which liveness guarantees.
2. `p'` preserves the **relative** ordering and ratios of allowed tokens:
   `p'(u)/p'(v) = p(u)/p(v)` for `u, v ∈ A_t`. So constraining never reorders
   preferences among legal tokens; it only removes illegal mass.
3. Greedy or sampled decoding under `p'` **stays in `A_t`**, so — by induction
   over steps and soundness — the produced string is always accepted by `A`.
   This is the safety property: *constrained decoding can only emit well-formed
   plans.* Verified exhaustively on a small grammar and on random logits.

In log space the mask is `z'_v = z_v` for `v ∈ A_t`, `-∞` otherwise, followed by
`log_softmax`; the `-∞` entries yield exactly 0 probability with no NaNs, which
is tested (a naive `exp(-inf)*something` can produce NaN).

## 4. Training objective

### 4.1 Plan NLL

```
L_plan = - Σ_i log p_θ(s_i | s_{<i}, x)
```

standard teacher-forced autoregressive cross-entropy over plan tokens, with the
padding positions masked out. Tested by exact agreement with a manual
computation and by overfitting a controllable evidence→plan mapping.

### 4.2 Semantic consistency loss

An auxiliary term penalising plans whose *decoded fields* violate cheap
structural predicates (e.g. a referent used but not declared). Implemented as a
differentiable surrogate on field-head logits, not on the discrete plan, so it
has a gradient. Its job is to make violations rare, not to replace the
hard validator.

### 4.3 Sequence DPO (mechanism only)

Per the document, appropriate **only after** pairwise judgments from qualified
signers. With sequence log-prob `log π(s|x) = Σ_i log π(s_i|s_{<i},x)`:

```
L_DPO = - log σ( β [ (log π_θ(s⁺|x) − log π_ref(s⁺|x))
                   − (log π_θ(s⁻|x) − log π_ref(s⁻|x)) ] )
```

At initialisation `π_θ = π_ref`, so the bracket is 0 and `L_DPO = log 2` — the
same closed-form check proved for Diffusion-DPO in the motion phase. **This
optimises observed preferences; it does not prove linguistic correctness**, and
the code says so.

## 5. Critical architecture decision — factorized training

FLa-LLM (arXiv:2403.12556) finds that *directly* introducing an LLM into
gloss-free SLT lets the LLM **dominate the learning curve** while the visual
representation stays weak; the fix is to factorize training — pre-train the
representation with a lightweight head, then freeze it and connect the LLM. Their
task is sign→text; the document asks us to treat the warning as a **testable
hypothesis in the inverse (speech/text→sign) direction**.

**Operationalisation.** After training, probe how much task-relevant information
the *acoustic/evidence representation* retains, with a frozen-representation
linear probe. The hypothesis predicts:

```
probe_accuracy(joint)  <  probe_accuracy(factorized)
```

i.e. joint training under-develops the representation because the planner LLM
shortcuts around it. This is measured, not assumed — and if the inverse
direction does *not* reproduce the effect, that is a real and reportable
negative result, exactly as the document frames it.

Factorized schedule:

* **Stage A (representation init).** Train the evidence encoder + a lightweight
  planner head. No LLM adapters.
* **Stage B (LLM fine-tune).** Freeze the encoder; train only the LLM adapters
  (LoRA) and cross-attention/prefix connectors.

## 6. Verification (from the document)

* Exact-match and structural validation of typed plans.
* Semantic role, negation, question type, number, tense/aspect, referent
  consistency, non-manual scope tests.
* **Counterfactual tests**: change one semantic feature and verify only licensed
  plan fields change.
* Retrieval-off and LLM-off ablations; report hallucinated lexical entries and
  invalid spatial references.

## 7. Honest scope

No real T5/Sign2GPT/LLM weights (multi-GB, unfetchable here). The planner is our
own compact encoder-decoder; the specification's "compact open-weight decoder
with parameter-efficient tuning" is honoured *architecturally* (encoder-decoder
+ LoRA + prefix/adapters + freeze-first), not by loading those weights. DPO is
implemented but not runnable without real signer preferences. All evidence is
synthetic; every threshold must be re-characterised on real data.

## 8. Findings from this round

Each was found by a property test rather than a smoke test.

**8.1 Constrained decoding could loop forever (real design flaw, fixed).**
The base DFA's variable-length slots are unbounded, so greedy decoding on an
untrained model -- which may always prefer a repeatable token (another manual
unit) -- never reaches EOP and hangs. Fixed by a **bounded runtime**: each
variable-length slot is capped by its natural cardinality (`num_referents`,
`max_units`, ...), making the language finite so decoding provably terminates.
The caps only remove the repeat option, so soundness and liveness transfer, and
the advance marker always survives the cap. Proved:
`test_bounded_decoding_terminates_even_with_adversarial_constant_logits`.

**8.2 Skeleton != plan (kept distinct, tested).** The automaton guarantees a
well-formed *skeleton* (grammar), not a consistent *plan* (cross-slot rules like
"an argument's referent is declared"). A constrained decode always deserializes,
but may still fail `validate_plan`. This is by design -- conflating the two would
be a category error -- and the tests assert exactly that boundary.

**8.3 The LLM-dominance hypothesis does not cleanly reproduce here (documented
negative/inconclusive result).** The document frames FLa-LLM's warning as a
testable hypothesis in the inverse direction and says a negative result is
reportable. Across several synthetic constructions the effect did not reproduce:
sometimes both regimes ceiling (probe 1.0 vs 1.0), sometimes joint *exceeds*
factorized. The deeper reason is methodological:

* a **linear representation probe** over a linear pooling of embeddings recovers
  linear features regardless of training, so it cannot cleanly separate a
  trained encoder from a random one at this scale -- a *random* encoder probes
  well **above chance** (`test_linear_probe_over_random_features_is_informative`);
* the attention pooling `query` is trained only through the content head, so
  probing it would compare a trained pool against a random one -- a confound
  removed by probing the regime-shared `memory` mean instead.

So Stage 2f ships the factorized **schedule** (proved: Stage B leaves the
encoder bit-identical; joint training moves it) and the dominance experiment as
a **reporting tool**, and asserts only the machinery -- never a predetermined
direction. Reproducing the dominance effect would require real pretrained
encoders and real data, not a synthetic probe. This is the same discipline as
the speech layer's Brier and ceiling findings: measure honestly, report the
negative result, do not tune until it looks positive.

**8.4 Float-layout reproducibility (test tolerance, not code).** DPO's margin is
`~1e-5` rather than exactly 0 at init because the `deepcopy`'d reference and the
policy have identical weights but separate memory layouts, and identical matmuls
can select different BLAS kernels. Proved exactly (`< 1e-9`) in double precision;
the float32 test uses a realistic tolerance and documents why.

## 9. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| 2.0 | design + math spec | done |
| 2a | typed schema + serialization + validator | done |
| 2b | schema automaton + constrained decoding (proved) | done |
| 2c | versioned lexicon retrieval + hallucination detection | done |
| 2d | planner model + acoustic prefix + plan NLL | done |
| 2e | semantic consistency + counterfactual + sequence DPO | done |
| 2f | factorized training + dominance hypothesis test | done |
| 2g | integration + cycle stress + full regression | done |
