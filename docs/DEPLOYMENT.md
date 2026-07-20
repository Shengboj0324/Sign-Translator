# 13 — Real-Time Deployment and Optimization — Design and Mathematics

This document fixes **all** mathematics of the deployment/optimization layer before any
code, in the discipline of docs 01–12. It implements
`13_real_time_deployment_optimization.md`: the streaming latency contract, the latency
budget algebra, the optimization order (caching → chunked attention → ONNX → TensorRT
FP16/INT8 → distillation → CUDA Graphs), runtime controls (backpressure, confidence
gate, fallback, telemetry, privacy), and a replay validation harness. The governing
rule is that **every optimization must be numerically validated against eager
execution and quality compared before/after** — a faster path that changes the output
beyond tolerance, or regresses quality, is rejected.

**Reuse.** This layer builds on the audited streaming/latency/policy primitives:

* `speech/revision.py` — `StreamingDecoder`, `longest_common_prefix`,
  `commitment_error_count`, `RevisionStats`: the committed-prefix contract.
* `speech/streaming.py` — `LatencyModel`, `percentile`, `LatencyMeasurement`
  (p50/p95): the latency-budget base.
* `speech/policy.py` — `FailClosedPolicy` (`PAUSE<FINGERSPELL<EMIT`): the confidence
  gate / clarification.
* `motion_transformer/streaming.py` — bounded right-context mask, SLERP crossfade,
  overlap-add (chunked motion).
* `diffusion_gen/consistency.py` — few-step distillation (only after a quality baseline).
* `avatar_render` pacing/telemetry (motion-to-photon p95, dropped-frame, flicker).
* `eval_framework` — the Doc-12 contract chain for the quality-non-regression gate.

Primary sources studied:

* *ONNX Runtime performance* (onnxruntime.ai) — export supported subgraphs and
  **validate numerically against eager** execution.
* *NVIDIA TensorRT* (docs.nvidia.com) — symmetric INT8 with a per-tensor/per-channel
  **scale from max-abs**, calibrated (entropy/MSE/percentile/min-max); FP16/BF16 first.
* Dao, *FlashAttention-2* (arXiv:2307.08691) — exact attention via **online-softmax**
  tiling; numerically equivalent to standard attention.
* *CUDA Graphs* (developer.nvidia.com) — capture/replay for **stable shapes** only.
* Song et al., *Consistency Models* (arXiv:2303.01469) — few-step generation (Doc-07).

## 0. Honest scope (read first)

The sandbox is CPU-only: there is no GPU, TensorRT engine, ONNX Runtime, or CUDA Graph.
We implement the **mathematics and control logic** of deployment — the streaming
contract, the latency/queueing algebra, the quantization error bounds, the
numerical-equivalence gate, online-softmax exactness, the runtime controls, and the
replay harness — and validate them on controllable synthetic inputs with known ground
truth. Everything independent of specific hardware (the commit monotonicity, the
bottleneck/queueing theorems, the quantization bounds, online-softmax equality, the
gates and guards) is proved exactly. TensorRT/ONNX/CUDA-Graph *execution* is out of
scope; the code exposes the interfaces and the numerical validation those toolchains
require, so real engines drop in behind the same gate.

## 1. Streaming contract + display-commit monotonicity (innovation)

The system holds a committed prefix `C_t`, a revisable suffix `U_t`, and avatar state
`q_t`; it may revise only `U_t`. **Display-commit monotonicity** is the checkable
invariant: the committed prefix is **append-only** — `C_t` is a prefix of `C_{t+1}`
for all `t`, so an already-displayed sign is never silently changed. A revision that
rewrites a committed token is a **commitment error** (reuse Doc-01
`commitment_error_count`). `certify_commit_monotone(history)` returns True iff every
consecutive pair extends, and pinpoints the first violation otherwise.

## 2. Latency budget algebra

For `K` mostly-pipelined stages with service times `s_1..s_K`:

* **steady-state throughput** `= 1 / max_i s_i` items/sec — bounded by the **slowest
  stage** (the bottleneck);
* **first-output latency** is the dependency-critical path
  `L_first ≈ L_buffer + L_ASR + L_plan + L_motion + L_render` (the sum, since the first
  item traverses every stage serially).

Report **median, p95, p99** at batch size 1 on named hardware (reuse `percentile`).
**Credibility guard (innovation):** a latency *claim* (e.g. "<200 ms") is not credible
unless it discloses chunk size, lookahead, whether sentence reordering is allowed, and
the quality loss; `latency_claim_is_credible(claim)` returns False if any is missing —
encoding the document's explicit skepticism.

## 3. Backpressure + bounded-queue theorem (innovation)

A stage is a queue of capacity `B` served at rate `μ` with arrival rate `λ`.

* **Without backpressure**, if `λ > μ` the backlog grows as `(λ−μ)·t` and end-to-end
  latency is **unbounded**.
* **With backpressure** (block/slow the source when the queue is full), the backlog is
  `≤ B` at all times, so queueing latency is `≤ B/μ` — **bounded**. The document's rule
  "slow/pause signing rather than accumulate unbounded delay" is a theorem:
  `simulate_queue(..., backpressure=True)` keeps occupancy `≤ B`; with it off and
  `λ>μ`, occupancy diverges. Backpressure trades dropped/slowed input for bounded
  latency — proved.

## 4. Quantization math + error bounds

* **FP16 rounding** — `|x − fp16(x)| / |x| ≤ 2^{−11}` (10 explicit mantissa bits +
  implicit), proved by round-trip.
* **Symmetric INT8** — scale `s = max|x| / 127`, `q = clip(round(x/s), −127, 127)`,
  `x̂ = q·s`. For `|x| ≤ max|x|`, the rounding error `|x − x̂| ≤ s/2` (proved); outside
  the range there is additional **clipping** error, surfaced explicitly.
* **Affine (asymmetric) UINT8** — `s = (max−min)/255`, zero-point
  `z = round(−min/s)`, `q = clip(round(x/s)+z, 0, 255)`, `x̂ = (q−z)·s`; in-range error
  `≤ s/2`.
* **Calibration** — min/max or a **percentile** clip (trade a smaller scale, hence
  smaller rounding error, for some clipping); `range_coverage(x, qmin, qmax)` certifies
  the fraction of data inside the representable range (no silent large clipping).

## 5. Numerically-certified optimization gate (innovation)

Every transform (FP16, INT8, a chunked-attention kernel, a distilled model) must pass
**both** gates or be rejected:

* **numerical equivalence** — `max_abs_error` and `max_rel_error` between the optimized
  and the eager output must be within tolerance (the ONNX-vs-eager validation);
* **quality non-regression** — a Doc-12 falsifiable contract on the downstream quality
  metric (compare quality before/after every transform).

**Online-softmax exactness** is the worked chunked-attention example: computing softmax
blockwise with a running max `m` and rescaled running sum `ℓ` (rescale the accumulator
by `exp(m_old − m_new)` per block) equals the full softmax **exactly** (to float
rounding) — so a FlashAttention-style kernel passes the equivalence gate by
construction (proved in float64).

## 6. Runtime controls

* **Confidence gate + clarification** — reuse the Doc-03 `FailClosedPolicy`; names,
  numbers, and low-confidence ASR route to clarification/fingerspelling rather than a
  confident wrong sign.
* **Deterministic fallback** — a verified phrase-retrieval / fingerspelling path with a
  simple rig renderer, always available and side-effect-free.
* **Telemetry** — thermal, memory, dropped-frame, and desynchronisation gauges each
  with a threshold; `TelemetrySnapshot.healthy()` is the conjunction.
* **Privacy** — an on-device audio ring buffer with **immediate deletion**: `clear()`
  zeroises the buffer, and a privacy-mode buffer refuses to retain past its window.

## 7. Shape-stability guard + optimization order (innovation)

* **Static-buffer / CUDA-Graph guard** — a captured executor is valid **only for stable
  shapes**; replaying it with a different input shape **raises** (the document's
  caveat), structurally.
* **Optimization order** — distillation/quantization of the planner and few-step
  diffusion are permitted **only after** a quality baseline is established;
  `OptimizationPlan` enforces the ordered steps and refuses to distill before the
  baseline gate has passed.

## 8. Replay validation harness

A `ReplayHarness` takes timestamped audio and **expected semantic checkpoints**; it
replays deterministically and asserts each checkpoint is met at (or before) its time,
exercising chunk boundaries, interruptions, corrections, and cold starts. Quality is
compared before/after every transform through the §5 gate.

## 9. Integration + innovations

The contract reuses Doc-01 revision; latency reuses `speech/streaming`; the gate reuses
the Doc-12 contract chain; the confidence gate reuses Doc-03. Innovations: the
display-commit monotonicity certificate, the backpressure-bounded-latency theorem, the
numerically-certified optimization gate with online-softmax exactness, and the
shape-stability guard.

## 10. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| 13.0 | research + design/math spec (this doc) | done |
| 13a | streaming contract + display-commit monotonicity | done (7 tests) |
| 13b | latency budget algebra | done (8 tests) |
| 13c | backpressure + bounded-queue theorem | done (7 tests) |
| 13d | quantization math + error bounds | done (8 tests) |
| 13e | numerically-certified optimization gate | done (8 tests) |
| 13f | runtime controls | done (8 tests) |
| 13g | shape-stability guard + optimization order | done (7 tests) |
| 13h | replay harness + integration + full regression | done (3 tests) |

Deployment layer: 56 tests, green on two consecutive runs under
`-W error::UserWarning`; whole project green (all 136 test files).

## 11. Findings (post-implementation)

**The streaming contract is checkable, not aspirational.** The committed prefix is
append-only: `certify_commit_monotone` returns the first step at which a committed
(displayed) token is rewritten, and `StreamingContract.commit` raises rather than
silently changing a displayed sign. Committed-token errors reuse the Doc-01 accounting.

**The latency algebra separates throughput from first-output latency.** Steady-state
throughput is `1/max_i s_i` (the bottleneck stage), while first-output latency is the
*sum* of stage times `L_buffer+L_ASR+L_plan+L_motion+L_render` — the first item is
serial, so first-output latency strictly exceeds the per-item throughput period. The
credibility guard encodes the document's skepticism: a "<200 ms" claim missing chunk
size, lookahead, reordering, or quality loss is rejected, and a report needs named
hardware at batch size 1.

**Backpressure is a bounded-latency theorem.** With `λ>μ` and no backpressure the
backlog grows as `(λ−μ)·t` (unbounded, verified to grow with the step count); with
backpressure the backlog never exceeds capacity `B`, so queueing latency is `≤ B/μ` —
the document's "slow/pause rather than accumulate unbounded delay", proved.

**Quantization error is bounded exactly.** FP16 rounding has relative error `≤ 2⁻¹¹`
(verified on a normal-range sweep); symmetric INT8 (scale `max|x|/127`) and affine
UINT8 have in-range reconstruction error `≤ scale/2`; percentile calibration provably
shrinks the scale by clipping outliers, and `range_coverage` certifies how much data
the representable range actually covers (no silent clipping).

**Optimizations must pass two gates — and online softmax passes by construction.**
Every transform is accepted only if it is numerically equivalent to eager (max abs/rel
error within tolerance) *and* does not regress quality (a Doc-12 falsifiable contract).
The chunked-attention example — blockwise online softmax with a running max and a
rescaled running sum — equals full softmax and full attention **exactly** (to 1e-12 in
float64), so a FlashAttention-style kernel is certified, while a divergent or
quality-regressing transform is rejected.

**Runtime controls fail safe.** Names and numbers below a strict confidence route to
CLARIFY rather than a confident wrong sign (over the Doc-03 fail-closed policy); the
fallback is deterministic; telemetry health is the conjunction of thermal/memory/
dropped-frame/desync gauges; and the privacy ring buffer never retains beyond its
window and zeroises on `clear()` (immediate deletion). The static-shape executor
refuses a new input shape (CUDA-Graph caveat), and the optimization-order gate refuses
to distill/quantize before a quality baseline or to build INT8 before FP16.

**Honest scope holds.** The sandbox is CPU-only — no GPU, TensorRT engine, ONNX
Runtime, or CUDA Graph — so this is the deployment mathematics and control logic,
proved on controllable synthetic inputs; real engines drop in behind the same
numerical-equivalence gate. Innovations: the display-commit monotonicity certificate,
the backpressure-bounded-latency theorem, the numerically-certified optimization gate
with online-softmax exactness, and the shape-stability guard.
