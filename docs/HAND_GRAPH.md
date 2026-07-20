# 05 — Hand-Motion Graph Reasoning — Design and Mathematics

This document fixes **all** mathematics of the hand-motion graph-reasoning layer
before any code is written, in the discipline of docs 01–04. It implements
`05_hand_motion_graph_reasoning.md`. The layer models signing hands at high
resolution as a **heterogeneous temporal graph** and reasons over it with
relational message passing plus Graphormer-style structural attention.

Primary sources studied:

* Yan, Xiong, Lin, *Spatial Temporal Graph Convolutional Networks* (ST-GCN),
  arXiv:1801.07455 — partitioned spatial graph convolution `Σ_k A_k X W_k`
  (already in `models/stgcn.py`; the kinematic-only baseline here).
* Ying et al., *Do Transformers Really Perform Bad for Graph Representation?*
  (Graphormer), arXiv:2106.05234 — three structural encodings: **centrality**
  (degree) added to node features, **spatial** (shortest-path-distance) added as a
  scalar attention bias `b_{φ(i,j)}`, and **edge** encoding along the SPD path.
  The paper proves popular GNNs are special cases; we use it as *inspiration for
  structural encodings, not a drop-in guarantee for skeletons* (per the document).
* Zhang et al., *MediaPipe Hands*, arXiv:2006.10214 — the **21-landmark** per-hand
  topology (0 wrist; 1–4 thumb; 5–8 index; 9–12 middle; 13–16 ring; 17–20 pinky).
* Romero, Tzionas, Black, *Embodied Hands* (MANO), arXiv:1705.04714 — a parametric
  hand (SMPL for hands): 778 vertices, 16 joints (wrist + 15 articulated),
  axis-angle pose + shape, linear blend skinning. This is exactly the per-hand
  component of the Doc-04 SMPL-X layout (15 hand joints each); we reuse that.
* Yu et al., *SignAvatars* (arXiv:2310.20436) — holistic 3D sign data (the eventual
  real hand annotations; licensed, not fetched here).

## 0. Honest scope (read first)

We do **not** have licensed MANO tensors or real MediaPipe-annotated sign data.
As in Doc 04, we implement the **mathematics exactly** and validate on a
**controllable synthetic hand graph** whose structure and ground truth we set, so
every property that does not depend on learned weights (graph structure, message-
passing normalisation and equivariance, translation/rotation invariance, the
contact field's monotonicity, Graphormer's reduction to vanilla attention, the
temporal receptive field, masked aggregation, all metrics and invariance tests) is
proved exactly. Realistic hand data and MANO drop in later with no code change.
The MANO/MediaPipe **conventions** (21 landmarks, kinematic tree) are used
structurally; no licensed weights are downloaded. Baselines (adaptive GCN, vanilla
Transformer, kinematic-only) are scaffolded for comparison, not trained on real
data here.

## 1. The heterogeneous temporal graph

Each node `i` carries a type `τ(i) ∈ {BODY, LEFT_HAND, RIGHT_HAND, FACE, LOCUS}`, a
feature `h_i ∈ ℝ^d`, a 3D position `x_i`, and a velocity `ẋ_i`. Per frame the node
inventory is the union of a body subset, the 21 left-hand landmarks, the 21
right-hand landmarks, a face subset, and locus (spatial-reference) points. A clip
of `T` frames replicates the nodes and adds temporal edges.

Edge types `r ∈ {BONE, TEMPORAL, SYMMETRY, CONTACT, DISTANCE, SEMANTIC}`:

* **BONE** — kinematic-tree edges within a hand/body (fixed, from MediaPipe/MANO).
* **TEMPORAL** — the same node in adjacent frames (identity across time).
* **SYMMETRY** — left-hand landmark `k` ↔ right-hand landmark `k` (mirror pairs).
* **CONTACT** — formed when two nodes are near (soft, §4).
* **DISTANCE** — spatial proximity (k-nearest in 3D), capturing hand-to-face etc.
* **SEMANTIC** — learned long-range relations (sparse global attention, §5).

Edges are directed; every relation stores `(source, target)` and an optional edge
feature `e_ij`. `validate_hand_graph` checks: node ids contiguous, edge endpoints
in range, BONE within one part, SYMMETRY strictly cross-hand, TEMPORAL strictly
between consecutive frames.

## 2. Relational message passing

The document's update (relational GAT with per-relation transforms):

    h_i' = W_0 h_i + Σ_r Σ_{j∈N_r(i)} α^r_ij W_r h_j,
    α^r_ij = softmax_j( g_r(h_i, h_j, e_ij) ),

with `g_r(h_i,h_j,e) = LeakyReLU( a_rᵀ [W_r h_i ‖ W_r h_j ‖ E e] )` (GAT scoring).
The softmax is over the neighbours `j ∈ N_r(i)` of node `i` under relation `r`
(per-node, per-relation normalisation).

**Basis decomposition** (R-GCN, Schlichtkrull et al.) to bound parameters:

    W_r = Σ_{b=1}^{B} a_{rb} V_b,     V_b ∈ ℝ^{d×d} shared,  a_{rb} ∈ ℝ.

This replaces `R·d²` parameters with `B·d² + R·B`. At `B = R` with one-hot
coefficients it recovers independent `W_r` exactly (full expressivity).

Proved (on a controllable graph):

1. **Softmax normalisation** — `Σ_{j∈N_r(i)} α^r_ij = 1` for every `(i, r)` with a
   non-empty neighbourhood.
2. **Neighbour-permutation equivariance** — permuting the neighbour order within
   `N_r(i)` leaves `h_i'` unchanged (sum + softmax are symmetric over `j`).
3. **Reduction** — with `g_r ≡ const`, `α^r_ij = 1/|N_r(i)|` recovers mean-pooling
   R-GCN; with a single relation and `W_0=W_r`, it recovers a GAT layer.
4. **Isolated node** — no neighbours ⇒ `h_i' = W_0 h_i` (empty sum, no NaN).
5. **Basis expressivity** — `B=R` one-hot reproduces the independent-`W_r` layer to
   floating precision.
6. **Gradient flow** — to `W_0`, bases `V_b`, coefficients `a_{rb}`, and attention
   parameters `a_r, E`.

## 3. Wrist-relative geometry (invariance)

Represent each hand landmark relative to its wrist:

    x̃_i = x_i − x_{wrist(τ(i))}   (translation-invariant),
    x̂_i = R_{wrist}ᵀ (x_i − x_{wrist})   (translation- AND rotation-invariant),

and relative velocity `ṽ_i = ẋ_i − ẋ_{wrist}`. Proved exactly:

* **Translation invariance** — `x̃` is unchanged by `x ↦ x + t` (the wrist shifts
  by the same `t`).
* **Rotation+translation invariance** — `x̂` is unchanged by `x ↦ R x + t` for any
  global `R ∈ SO(3)`, because `R_{wrist} ↦ R R_{wrist}` so
  `(R R_{wrist})ᵀ (R x + t − (R x_w + t)) = R_{wrist}ᵀ (x − x_w)`.

This is what makes the hand representation invariant to camera/global pose, as the
document requires.

## 4. Contact soft-distance field

    p^contact_ij = σ( wᵀ [h_i, h_j, d_ij, s_ij] ),   d_ij = ‖x_i − x_j‖,
                                                     s_ij = ‖ẋ_i − ẋ_j‖.

Using the scalar distance `d_ij` and **relative speed** `s_ij = ‖ẋ_i − ẋ_j‖` makes
the predictor symmetric (`p_ij = p_ji`) when `[h_i,h_j]` is symmetrised. To
guarantee the physically-correct monotonicity (closer ⇒ more contact) we
parameterise the distance weight as `−softplus(θ_d) ≤ 0`, so:

* `p ∈ (0, 1)`, differentiable everywhere;
* `∂p/∂d_ij ≤ 0` — strictly non-increasing in distance (monotone), proved by the
  sign of the distance weight;
* a hard contact label is `1[d_ij < ρ]` for threshold `ρ`; the field is its smooth
  surrogate.

## 5. Graphormer structural encodings

Attention logit between nodes `i, j`:

    A_ij = (h_i W_Q)(h_j W_K)ᵀ / √d  +  b_{φ(i,j)}  +  c_ij,

* `b_{φ(i,j)}` — **spatial** encoding: a learnable scalar indexed by the shortest-
  path distance `φ(i,j)` in the graph (BFS; unreachable pairs get a dedicated
  index). Distant nodes are biased down/up as learned.
* `c_ij` — **edge** encoding: mean of learnable edge-type embeddings along the SPD
  path (here, the direct edge type when adjacent, else 0).
* **Centrality** encoding: `h_i ← h_i + z_{deg(i)}` (learnable per-degree vector),
  injected before attention.

Proved: with `b ≡ 0`, `c ≡ 0`, and centrality disabled, `A` reduces **exactly** to
standard scaled dot-product attention (so the module is a strict superset). SPD is
finite and well-defined on the connected temporal graph; softmax uses max-
subtraction for numerical stability. (Graphormer proves mean/sum/GAT aggregation
are special cases; we cite, not re-derive.)

## 6. Multi-scale temporal pyramid

Parallel dilated 1-D temporal convolutions at dilations `{1, 2, 4, …}` capture
rapid finger articulation (small dilation), transitions (medium), and holds
(large). For `L` branches with kernel `k` and dilation `d_l`, each branch's
receptive field is `1 + (k−1) d_l`; the pyramid's is the max over branches. Proved
by construction; a causal variant left-pads so frame `t` sees only `≤ t`.

## 7. Confidence-aware masking

Each landmark carries a confidence `c_i ∈ [0,1]` (0 = fully occluded). Masked
relational aggregation reweights neighbours by confidence and renormalises:

    α̃^r_ij = c_j α^r_ij / Σ_{j'∈N_r(i)} c_{j'} α^r_{ij'},

so an occluded neighbour (`c_j = 0`) contributes **nothing** and cannot leak. If
every neighbour is occluded the sum is 0 and the node falls back to `W_0 h_i`
(no NaN). The masked temporal conv normalises by the sum of visible weights in
each window likewise. Proved: occluded → zero contribution; renormalisation over
the visible set; graceful all-occluded fallback.

## 8. Auxiliary heads and losses

When annotations or reliable pseudo-labels exist:

* **Handshape** — softmax classification over a handshape inventory (per hand).
* **Palm orientation** — 6D rotation regression with the geodesic loss of Doc 04.
* **Selected fingers** — multilabel (extended/closed per finger), independent-
  Bernoulli BCE (the Doc-03 multilabel discipline).
* **Contact** — BCE on the contact field vs `1[d < ρ]`.
* **Symmetry** — for two-handed symmetric signs, penalise deviation between the
  left hand and the mirror of the right hand.

Every loss is **annotation-masked**: samples without a label are excluded (weight
0), never assigned a fabricated target (the Doc-03/Doc-04 provenance discipline).

## 9. Evaluation — hand-specific

* **Fingertip error in hand scale** — mean fingertip position error divided by hand
  size (wrist→middle-MCP distance); scale-invariant.
* **Joint geodesic error** — Doc-04 `geodesic_distance` over hand-joint rotations.
* **Handshape accuracy**, **contact F1** (precision/recall/F1 on contacts),
  **collision rate** (Doc-04 self-collision proxy), **left/right consistency**
  (symmetry error), **mirror/handedness invariance** (mirroring input maps a sign
  to its mirror — an equivariance test).

## 10. Integration

Hand landmarks come from `data/adapters.py` (MediaPipe 21→skeleton) and the Doc-04
SMPL-X hands (15 joints each); the hand graph consumes the high-resolution 21-node
hands. Its pooled per-hand embedding maps back to the 27-joint skeleton pipeline
and can annotate the SIR (Doc 03). Stage 5h wires this end to end, adds the three
baselines for the required comparison, and runs whole-chain cycle stress.

## 11. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| 5.0 | research + design/math spec (this doc) | done |
| 5a | heterogeneous temporal graph structure | done (12 tests) |
| 5b | relational message passing (R-GCN + attention, basis decomp) | done (11 tests) |
| 5c | wrist-relative geometry + contact field | done (10 tests) |
| 5d | Graphormer structural encodings | done (8 tests) |
| 5e | multi-scale temporal pyramid + confidence masking | done (9 tests) |
| 5f | auxiliary heads + losses | done (10 tests) |
| 5g | hand-specific evaluation metrics | done (9 tests) |
| 5h | integration + cycle stress + full regression | done (8 tests) |

Hand-graph layer: 77 tests, green on two consecutive runs; whole project (982
tests) green.

## 12. Findings (post-implementation)

**Relational message passing was proved, not assumed.** The R-GCN + GAT update is
verified to normalise (`Σ_j α^r_ij = 1` per destination/relation), to be
neighbour-permutation equivariant, to reduce to uniform mean-pooling at zero
attention logits, and to recover independent per-relation weights from the basis
decomposition at `B=R` with one-hot coefficients. Isolated nodes fall back to the
self term `W_0 h_i` with no NaN, and confidence masking makes an occluded
neighbour bit-identical to a removed edge (all-occluded ⇒ self term only).

**The wrist frame reuses the pose layer's Gram-Schmidt map, giving exact rotation
invariance.** Building `R_wrist` from two hand-spanning vectors via
`rotation_6d_to_matrix` makes it a proper SO(3) frame that is *equivariant* to
global rotation (`GS(Ra, Rb) = R·GS(a,b)`), so `x̂ = R_wristᵀ(x − x_wrist)` is
invariant to any global `Rx + t` — verified to 1e-9. This is the same continuous-
rotation machinery audited in Doc 04, now load-bearing here.

**The contact field is monotone by construction, not by hope.** Parameterising the
distance weight as `−softplus(θ_d) ≤ 0` guarantees `∂p/∂d ≤ 0` (closer ⇒ more
contact), and symmetrising on `h_i + h_j` with the scalar distance/relative-speed
makes `p_ij = p_ji`. Both are proved (monotonicity across a distance sweep and via
the autograd gradient sign; symmetry to 1e-12).

**Graphormer is a strict superset of vanilla attention.** With the spatial (SPD)
and edge biases zeroed — which is their initialisation — the module equals
standard multi-head scaled dot-product attention to 1e-12. The three required
baselines (kinematic-only, vanilla Transformer, full reasoner) are obtained by
flags on one class and are shown to be genuinely different computations; the
vanilla-Transformer baseline is proved to ignore graph structure entirely.

**End-to-end translation invariance is a theorem here, verified to 1e-10.** Because
every node feature is built from wrist-relative coordinates, translating all input
positions leaves the whole model's output unchanged — the camera/global-pose
invariance the document demands, proved rather than asserted.

**Honest scope holds.** No licensed MANO tensors or real MediaPipe-annotated sign
data; every property above is independent of learned weights and proved on a
controllable synthetic hand graph. The masked evaluation metrics
(fingertip-in-hand-scale, contact F1, collision rate, L/R consistency, mirror
involution) and annotation-masked auxiliary losses follow the Doc-03/Doc-04
discipline: unlabelled samples are excluded, never fabricated.
