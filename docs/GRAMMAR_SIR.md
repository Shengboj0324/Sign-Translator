# Sign-Language Grammar and Planning (SIR) — design, mathematics, staged plan

Implements `03_sign_language_grammar_translation.md`. As with the speech and
planner layers, the mathematics is fixed *before* the code, so every component
has a stated property a test can falsify. Anything asserted here without a
corresponding test is a defect.

We model **one named sign language: American Sign Language (ASL)**, as the
document requires a single named language and refuses a language-agnostic
reordering account.

---

## 1. Why this layer exists

English→ASL is **not word reordering**. ASL is *multi-channel* and *simultaneous*:
the hands carry manual signs while the face and body carry non-manual markers
(negation, question, topic, conditional, role shift) that scope *over intervals
of time*, co-occurring with the manual stream. A flat gloss string cannot express
this. So the layer's central object is a **Structured Intermediate Representation
(SIR)**: a temporal graph whose gloss is merely one observed *projection*.

This sits between the doc-02 typed plan (semantic content) and the motion
generator: plan → **SIR** (grammaticalised, timed, multi-channel) → gloss
projection / motion conditioning.

## 2. The SIR temporal graph

`G = (V, E)`.

* **Nodes** `V` are *events*, each either **manual** (a lexical sign, a
  classifier/depicting construction, or a fingerspelled item) or **non-manual**
  (a facial/body marker). Every event carries a half-open time interval
  `[t^s, t^e)` with `t^s < t^e`.
* **Edges** `E` are typed:
  `PRECEDENCE` (i before j), `OVERLAP` (co-temporal), `SCOPE` (a non-manual
  contains a manual span — Allen *during*), `COREF` (two events refer to the same
  discourse referent), `LOCUS` (an event is placed at a spatial locus in signing
  space).

### 2.1 Structural validity (tested rule-by-rule)

1. every interval satisfies `t^s < t^e`;
2. `PRECEDENCE` edges are acyclic (a DAG) — time cannot loop;
3. edges reference existing nodes;
4. a `SCOPE` edge's source is non-manual, its target manual;
5. `COREF` events share a declared referent;
6. `LOCUS` targets a locus in the fixed locus alphabet, and distinct referents
   occupy distinct loci (from the plan layer);
7. every manual event is a lexicon entry or fingerspelled (the hallucination
   rule, inherited).

### 2.2 Gloss as a projection

A gloss sequence is the **manual events in a linear order consistent with the
PRECEDENCE edges** — a topological sort of the manual sub-DAG. Because several
topological orders can exist, the projection is *one* observed gloss, not the
graph; the graph carries strictly more (timing, overlap, non-manual scope).
Tested: the projection is a valid topological order and every gloss token is a
manual event.

## 3. Temporal mathematics — Allen's interval algebra

The 13 mutually exclusive, jointly exhaustive relations between two intervals
`X=[x^s,x^e)`, `Y=[y^s,y^e)` (Allen 1983):

```
before(X,Y):     x^e < y^s          after: symmetric
meets(X,Y):      x^e = y^s          met-by
overlaps(X,Y):   x^s < y^s < x^e < y^e   overlapped-by
starts(X,Y):     x^s = y^s, x^e < y^e    started-by
during(X,Y):     y^s < x^s, x^e < y^e    contains
finishes(X,Y):   y^s < x^s, x^e = y^e    finished-by
equals(X,Y):     x^s = y^s, x^e = y^e
```

`SCOPE` is Allen **during/contains**: the non-manual interval *contains* the
manual span. `PRECEDENCE` is **before**. `OVERLAP` is the disjunction
{overlaps, starts, during, finishes, equals, and their inverses} — i.e. the
intervals intersect.

### 3.1 Differentiable temporal constraints

Each relation becomes a non-negative **hinge loss** that is exactly zero when the
relation holds (with a margin `ε > 0` for strict inequalities) and grows
linearly with the violation, so it is sub-differentiable with a correct descent
direction. The document gives precedence:

```
L_prec(i,j) = max(0, t_i^e - t_j^s + ε)
```

`= 0  ⇔  t_i^e ≤ t_j^s - ε  ⇔  i strictly before j`. Extended:

```
L_meets(i,j)     = |t_i^e - t_j^s|
L_contains(i,j)  = max(0, t_i^s - t_j^s + ε) + max(0, t_j^e - t_i^e + ε)
                   (i contains j: i^s < j^s and j^e < i^e)
L_overlap(i,j)   = max(0, t_i^s - t_j^e + ε) + max(0, t_j^s - t_i^e + ε)
                   (intervals intersect: not-before AND not-after)
L_equals(i,j)    = |t_i^s - t_j^s| + |t_i^e - t_j^e|
L_valid(i)       = max(0, t_i^s - t_i^e + ε)   (start strictly before end)
```

**Proved for each** (tests): (a) `L ≥ 0`; (b) `L = 0` iff the (margin-)relation
holds; (c) it is finite and has a gradient wherever the hinge is active; (d) a
few steps of gradient descent on the interval endpoints drive `L → 0` and the
Allen relation becomes satisfied — the operational statement that the constraint
is trainable. The `L_prec = 0 ⇔ before` equivalence is the document's own claim
and is checked directly.

The **total temporal loss** over an SIR is the sum of the per-edge losses for the
relation each edge type encodes, plus per-node validity. Minimising it yields a
temporally consistent SIR.

## 4. Relation-biased graph attention

The document's decoder attention, for node states `h_i` and a relation `r(i,j)`
between nodes:

```
α_ij = softmax_j( (W_Q h_i)·(W_K h_j) / √d + b_{r(i,j)} )
```

`b_r` is a learned scalar per relation type (including a "no-relation" default).
It biases attention *toward or away from* graph-adjacent nodes of a given
relation, letting precedence/scope/coref structure shape the computation.

**Proved (tests):** (a) with `b ≡ 0` it reduces **exactly** to standard scaled
dot-product attention; (b) `α` rows are a probability distribution; (c) raising
`b_r` for the relation on edge `(i,j)` strictly increases `α_ij` relative to
competitors (monotonic effect), so the bias does what it claims; (d) a padding /
adjacency mask sets `α_ij = 0` with no NaN (log-space masking).

## 5. Non-manual scope — multilabel interval prediction

Non-manual scope is **not punctuation appended after generation**; it is a set of
labelled time intervals co-temporal with the manual stream. For `M` markers and
`T` manual positions, the head predicts:

* a **multilabel** activation `σ(z_{m,t}) ∈ [0,1]^{M×T}` — several markers may be
  active at the same `t` (simultaneity), so this is `M` independent Bernoullis
  per position, trained with masked BCE, **not** a softmax over markers;
* the loss couples to §3: a predicted marker span must *contain* (Allen during)
  the manual events it scopes, via `L_contains`.

Tested: BCE matches a manual computation; co-occurring markers are representable
(a softmax head could not); the scope→manual containment loss is zero exactly
when the predicted span covers the manual span.

## 6. Notation and variation (HamNoSys / SignWriting)

A structured phonological notation records **handshape, location, movement,
orientation** for a sign, with validity checks (values in their inventories),
and a SiGML-like serialization. Critically, per the position paper and the
document: **dialect and register tags are carried, and variation is never
silently normalised into "incorrect" signing.** Automatically-generated glosses
are marked as *noisy* with a provenance record (source, confidence), so a
downstream consumer can weigh them. Tested: two dialect variants of the same
concept are both valid and are *not* collapsed; a noisy gloss keeps its
provenance.

## 7. Evaluation

### 7.1 SignBLEU (multi-channel)

Standard BLEU scores a single stream. Sign is multi-channel, so SignBLEU
(Kim et al. 2024) forms **temporal n-grams that blend within and across
channels** and scores modified n-gram precision with a brevity penalty. Our
implementation:

* linearise each channel's events into per-channel token streams keyed by time;
* form (a) *within-channel* n-grams and (b) *blended* grams that pair
  co-temporal tokens across channels;
* modified precision with clipping (a gram is credited at most its reference
  count), geometric mean over `n=1..N`, and brevity penalty
  `BP = min(1, exp(1 − r/c))`.

**Proved (tests):** identical multi-channel hypothesis/reference ⇒ score 1;
dropping a whole channel lowers the score (unlike single-stream BLEU, which
cannot see it); clipping caps repeated grams; BP penalises short output.

### 7.2 Inter-rater agreement

Human grammaticality judgments need agreement, not just a mean. **Cohen's κ**
(two raters) and **Fleiss' κ** (many raters):

```
κ = (p_o − p_e) / (1 − p_e)
```

`= 1` at perfect agreement, `≈ 0` at chance, and can be **negative** below
chance. All three regimes are asserted against hand-computed values.

## 8. Required grammatical tests (from the document)

Minimal pairs where exactly one grammatical feature changes and only the
*licensed* SIR fields change (extending the doc-02 counterfactual method):
negation, yes/no vs WH question, topicalization, conditional, aspect, plural
reference, role shift. Plus **spatial-locus persistence** across multi-sentence
discourse (a referent keeps its locus), and **OOV/name coverage via
fingerspelling**.

Human grammaticality/meaning evaluation with inter-rater agreement is specified;
we provide the *instruments* (rating schema + κ), not signer judgments, which
only qualified signers can supply.

## 9. Honest scope

The downloaded How2Sign frontal-view media has not been used to validate this
grammar layer: the local artifacts contain English translations but no independently
identified ASL gloss tier, and English tokens must not be relabelled as gloss. The
SIR, grammar rules and notation therefore remain linguistically motivated but
synthetic, and every threshold must be re-characterised on real, signer-validated
annotations. Human evaluation is instrumented, not performed. The model is our own
graph-aware Transformer, not a pretrained AMR parser.

## 10. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| 3.0 | design + math spec | done |
| 3a | SIR temporal graph + validation + gloss projection | done (20 tests) |
| 3b | Allen interval algebra + differentiable temporal losses | done (36 tests) |
| 3c | relation-biased graph attention + SIR decoder | done (12 tests) |
| 3d | non-manual multilabel interval prediction | done (13 tests) |
| 3e | HamNoSys notation + dialect/register/provenance | done (12 tests) |
| 3f | SignBLEU + inter-rater kappa | done (18 tests) |
| 3g | minimal-pairs battery + locus persistence + OOV | done (15 tests) |
| 3h | integration + cycle stress + regression | done (11 tests) |

Grammar layer: 137 tests, green on two consecutive runs; whole project green.

## 11. Findings (post-implementation)

**Minimal-pairs decomposition forced a correction, not a tuning.** The first
`_sir_fields` diff conflated (a) manual precedence structure with non-manual
*scope* edges, and (b) event *sequencing* with lexical *label* identity. Flipping
negation therefore appeared to "leak" into the `edges` field, and plural
inflection leaked into `order`. Both were artefacts of the diff, not the builder:
scope edges are a consequence of a marker (already captured by the `nonmanual`
field), and sequencing is about the referent-slot order, not which lexeme fills a
slot. Restricting `edges` to PRECEDENCE only and keying `order` on referent slots
made every one of the eight feature flips change *exactly* its licensed fields —
and a companion test guards against the vacuous pass (each flip must change ≥1
field). Plural is modelled as morphological inflection (a plural-marked lexeme),
not an appended sign, so it is genuinely "manual-label only".

**The plan→SIR bridge is faithful by construction and measured, not asserted.**
The manual gloss projected out of the SIR equals the plan's manual-unit order for
all 200 randomised stress plans; every non-manual span survives as a scoped
event; fingerspelled units become FINGERSPELL, never hallucinated signs. The
bridge encodes only what the plan specifies — it does **not** invent a
unit→referent map (loci/coref flow through only when a caller supplies that map).

**A quantified, explained residual, not zero.** On a valid plan the temporal loss
is not 0 but exactly `4·ε`: contiguous signs *meet* (each of 2 precedence edges
pays the strict-precedence hinge margin ε), and the two boundary-aligned units of
a full-width scope span each pay one containment margin ε. Validity loss is
exactly 0. This is the honest reading — touching intervals cost the margin the
hinge demands — rather than loosening a tolerance until it reads zero.

**Smoothed SignBLEU vs. exact zero.** Additive smoothing makes a no-overlap score
a tiny positive number; rather than weaken the test, the metric now follows the
standard BLEU convention (no unigram overlap ⇒ nothing matched ⇒ exactly 0),
keeping smoothing only to soften the higher-order "one missing n-gram zeroes all"
harshness.
