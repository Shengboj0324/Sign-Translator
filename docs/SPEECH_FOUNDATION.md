# Speech Foundation Layer — design, mathematics, and staged plan

Implements `01_speech_foundation_layer.md`. This document fixes the mathematics
*before* the code, so that every component has a stated property a test can
falsify. Anything asserted here without a corresponding test is a defect.

---

## 1. Interface contract

The layer maps waveform chunks `a_{1:t}` to:

| Output | Symbol | Consumer |
|---|---|---|
| time-indexed acoustic states | `H^a ∈ R^{T_a × d_a}` | planner (acoustic pathway) |
| N-best / lattice hypotheses | `Y_t` | planner (lexical pathway) |
| word timestamps | `(w_i, t_start, t_end)` | sign timing / alignment |
| calibrated confidence | `c ∈ [0,1]` | fail-closed policy |
| prosody | `(F0, energy, pauses)` | discourse/affect conditioning |

Critically, the planner must be able to **revise uncommitted speech**. A design
that emits only a final string is non-compliant; hypotheses carry a committed
prefix and a revisable suffix.

## 2. Encoder and projection

```
H^a = E_φ(X) ∈ R^{T_a × d_a},     p(y|X) = Π_i p(y_i | y_<i, H^a)
H̃   = Resample(H^a)
G    = σ(W_g H̃)
H^p  = G ⊙ W_1 H̃ + (1 − G) ⊙ W_2 H̃
```

**Provable properties of the gated projection** (all tested):

* *Convexity collapse.* If `W_1 = W_2 = W` then for every gate value
  `H^p = G⊙WH̃ + (1−G)⊙WH̃ = WH̃`. The output is then **independent of `W_g`**.
  This is a strong structural test: it must hold exactly, for random gates.
* *Elementwise convex bound.* Since `G ∈ (0,1)` componentwise, every output
  element lies strictly between the corresponding elements of `W_1H̃` and
  `W_2H̃`: `min(a,b) ≤ H^p ≤ max(a,b)`.
* *Gradient reachability.* `W_1`, `W_2`, `W_g` all receive non-zero gradient.

The gate is a *learned per-dimension interpolation* between two projections —
not an attention mechanism, and not a residual. Reading it as anything else
leads to wrong tests.

## 3. Front-end mathematics

### 3.1 STFT

Frame `m`, bin `k`, window `w`, hop `H`, FFT size `N`:

```
X[k,m] = Σ_{n=0}^{N-1} w[n] · x[n + mH] · exp(−2πi kn/N)
```

Power spectrogram `P = |X|²`. **Parseval's theorem** gives an exact check on
each frame:

```
Σ_n |w[n]x[n+mH]|²  =  (1/N) Σ_{k=0}^{N-1} |X[k,m]|²
```

Note the sum on the right runs over the **full** `N` bins. A one-sided
(`rfft`) spectrum must be folded back — bins `0` and `N/2` counted once,
all others twice — before comparing. Getting this wrong is the classic
energy-accounting bug, so it is tested explicitly.

### 3.2 Mel scales

HTK (used by Whisper's `htk` path and by Kaldi):

```
mel(f) = 2595 · log10(1 + f/700),      f(mel) = 700 · (10^(mel/2595) − 1)
```

Slaney: linear at `f < 1000 Hz` with slope `3/200` mel/Hz, logarithmic above,
joined continuously at 1000 Hz with `log(6.4)/27` decades per mel step.

Both must satisfy `f(mel(x)) = x` to float precision, and be strictly
increasing. Tested.

### 3.3 Triangular filterbank

Given `M+2` points equally spaced **in mel** between `f_min` and `f_max`,
converted to Hz as `f_0 … f_{M+1}`, filter `j ∈ [1,M]` is

```
          ⎧ (f − f_{j−1}) / (f_j − f_{j−1})      f_{j−1} ≤ f ≤ f_j
H_j(f) =  ⎨ (f_{j+1} − f) / (f_{j+1} − f_j)      f_j ≤ f ≤ f_{j+1}
          ⎩ 0                                    otherwise
```

**Partition of unity.** For unnormalised (peak-1) triangles, at any frequency
`f` strictly between two adjacent centres `f_j < f < f_{j+1}`:

```
H_j(f) + H_{j+1}(f) = (f_{j+1}−f)/(f_{j+1}−f_j) + (f−f_j)/(f_{j+1}−f_j) = 1
```

so the filterbank sums to exactly 1 across the interior of its band. This is a
strong, exactly-checkable invariant and is tested on the analytic filter
response (not merely on the discretised bins, where sampling makes it
approximate).

*Slaney normalisation* instead scales filter `j` by `2/(f_{j+1} − f_{j−1})`,
making each filter unit-**area** rather than unit-peak; partition of unity then
does **not** hold, and the test asserts the area property instead. Conflating
the two normalisations is a common silent error.

### 3.4 Log-Mel (Whisper convention)

```
S = M · P                         (mel energies, M = filterbank)
L = log10(max(S, 1e-10))
L = max(L, max(L) − 8.0)          (dynamic-range floor, 80 dB)
L = (L + 4.0) / 4.0               (approx. zero-mean/unit-scale)
```

with `sr = 16000`, `n_fft = 400` (25 ms), `hop = 160` (10 ms), `n_mels = 80`,
Hann window — giving a 100 Hz frame rate (Whisper's encoder then halves it to
50 Hz via a stride-2 convolution).

### 3.5 Prosody — YIN F0

Difference function and cumulative mean normalisation (de Cheveigné & Kawahara):

```
d(τ)  = Σ_n (x[n] − x[n+τ])²
d'(0) = 1,    d'(τ) = d(τ) / [ (1/τ) Σ_{j=1}^{τ} d(j) ]
```

Pick the first `τ` with `d'(τ) < threshold` (absolute threshold, default 0.1),
else `argmin d'`. Refine with parabolic interpolation on `(τ−1, τ, τ+1)`:

```
τ* = τ + (d'(τ−1) − d'(τ+1)) / (2·(d'(τ−1) − 2d'(τ) + d'(τ+1)))
```

Then `F0 = sr / τ*`. The cumulative-mean step is what suppresses the
**octave error** (the τ=2T subharmonic dip), which plain autocorrelation
suffers; verified by testing a signal with strong harmonics.

## 4. Streaming and latency algebra

Let chunk size be `C` frames and right context `R` frames.

**Correctness invariant.** Streaming extraction must reproduce offline
extraction **exactly** (bit-for-bit up to float associativity), which requires
carrying `n_fft − hop` samples of overlap across chunk boundaries. This is
tested as an equality, not a tolerance-free-for-all.

**Emission latency.** Frame `t` is centred at audio time

```
c(t) = (t·H + N/2) / sr
```

and cannot be emitted until audio through sample `(t+R)·H + N` has arrived, at
wall time `a(t) = ((t+R)·H + N)/sr`. Algorithmic latency is therefore

```
ℓ(t) = a(t) − c(t) = (R·H + N/2) / sr        (constant in t)
```

Chunked processing adds buffering: a frame at position `i` within its chunk
waits for the remaining `C−1−i` frames, contributing `(C−1−i)·H/sr`. Hence

```
ℓ_max = (R·H + N/2 + (C−1)·H)/sr,    ℓ_median ≈ (R·H + N/2 + (C−1)H/2)/sr
```

Both the analytic values and the empirically measured median/p95 are reported,
and the test asserts they agree — a real cross-validation rather than a
self-consistent tautology.

## 5. Training objective (stages 3–4)

```
L = L_ASR + λ_c·L_contrast + λ_b·L_boundary + λ_cal·L_Brier
```

* `L_ASR` — CTC (streaming-compatible alignment).
* `L_contrast` — **symmetric InfoNCE** over pooled speech vs. target-sign
  representations. Per the source document this is *not* evidence of semantic
  equivalence; it must be validated by retrieval **and** downstream ablation.
* `L_boundary` — supervision on word/segment boundaries for timestamps.
* `L_Brier` — multiclass Brier score `(1/N)ΣΣ(p_ic − y_ic)²`, a **strictly
  proper** scoring rule, so its minimiser is the true conditional distribution;
  this is why it is the right calibration objective rather than accuracy.

Training protocol: **freeze-first**. Adapter only → converge → unfreeze upper
encoder blocks with LoRA / low LR.

### LoRA

`W' = W_0 + (α/r)·BA` with `B ∈ R^{d×r}` zero-initialised and
`A ∈ R^{r×k}` random, so `ΔW = 0` at initialisation and the adapted model is
**exactly** the base model on step 0. `B A` is mergeable into `W_0`, giving no
added inference latency. Both properties are tested.

## 6. Acceptance criteria (from the source document)

* Ablate transcript-only / acoustic-only / fused across clean, noisy, accented,
  code-switched, long-form conditions.
* Report WER/CER, timestamp error, expected calibration error, revision rate,
  downstream sign-plan degradation.
* Report chunk size and right context explicitly; median and p95 emission
  latency.
* **Fail closed**: on low confidence, pause or fingerspell a verified item —
  never hallucinate a sign.

## 7. Findings from Stage 1 (defects caught, and how)

Recorded because each was found by a *property* test rather than a smoke test,
and each would have been invisible to shape/finiteness checking.

**7.1 Silence reported as confident 400 Hz pitch (real bug, fixed).**
For digital silence every `d(τ) = 0`, so the CMND ratio is `0/0`. Clamping the
denominator produced `d' = 0`, which is *below* every voicing threshold — so the
estimator emitted a confident pitch at `f_max` for silent frames. The correct
resolution is that `0/0` means *no evidence of periodicity*, i.e. `d' = 1`.
A second safeguard (an RMS gate) now also forces quiet frames unvoiced, so the
prosody path fails closed on absent evidence, as the specification demands.

**7.2 Latency algebra wrong whenever `n_fft` is not a multiple of `hop`
(real bug, fixed).** The buffering wait was assumed to be at most `(C−1)` frame
periods. That holds only when `n_fft mod hop = 0`. Whisper's `400/160` gives
`ρ = 80`, and the true wait set is `{C·hop−ρ, …, hop−ρ}` — up to a **full extra
frame**. Measured min/max (0.0575 / 0.1275 s) disagreed with the predicted
0.0525 / 0.1225 s, which is precisely why the analytic value is cross-checked
against simulation rather than trusted.

**7.3 Whisper's dynamic-range floor is non-causal (real constraint, surfaced).**
`max(log S, max(log S) − 8 dB)` depends on the maximum over the *whole*
utterance, so a prefix cannot compute it and chunked features cannot equal
offline features. Rather than silently emitting mismatched features, the
front-end now exposes `floor_mode ∈ {global, fixed, none}` and an `is_causal`
flag, and `StreamingFeatureExtractor` **refuses** a non-causal front-end. This
is a genuine tension between the Whisper convention and the streaming
requirement, and it is better surfaced than hidden.

**7.4 Two numerical-analysis traps in the tests (tests wrong, not code).**
*(a)* Streaming vs. offline features differ by ~6e-4 in float32; repeating in
float64 drops the error to 2e-13, proving frame alignment is exact and the
residual is FFT accumulation order varying with buffer length.
*(b)* LoRA `merge()` shifts outputs by ~1e-3 relative in float32 because it
reassociates the matmul (rank-`r` bottleneck first vs. full matrix); in float64
the deviation is ~1e-16. Relatedly, computing `matrix_rank` after upcasting a
float32 product to float64 reports *full* rank, because upcasting preserves the
~1e-8 float32 noise while tightening the tolerance. Rank must be evaluated in
the tensor's own dtype. In both cases the honest resolution is to prove the
mathematical identity in double precision and *bound* the float32 deviation,
never to assert bitwise equality.

## 8. Stage 2 — decoding, timestamps, revision

### 8.1 Why a lattice, not a string

CTC scores a *label sequence* as a sum over every alignment collapsing to it,

```
p(l | x) = Σ_{π ∈ B⁻¹(l)} Π_t y_t(π_t)
```

Greedy decoding reports one alignment, so its score is the probability of that
**path**, not of the **label sequence**. The two disagree whenever a label's
mass is spread over several alignments — a case the tests construct explicitly
(`test_beam_can_beat_greedy_by_summing_alignments`).

Prefix beam search tracks, per prefix, the mass ending in a blank (`p_b`)
separately from that ending in a non-blank (`p_nb`). The split is what makes the
repeat rule expressible: an extension by `c == l[-1]` may draw only on `p_b`,
because two identical labels collapse into one unless a blank separates them.

**Proved, not asserted.** `ctc_exact_posteriors` enumerates all `C^T` paths;
the tests require the beam (with pruning disabled) to reproduce those
probabilities to 1e-9 and the posteriors to sum to exactly 1.

### 8.2 Forced alignment

Timestamps come from Viterbi alignment over `l' = [blank, l₁, blank, …, blank]`:

```
α(t,s) = max_{p ∈ pred(s)} α(t-1,p) + log y_t(l'[s])
```

with `pred(s) = {s, s-1}` plus `s-2` only when `l'[s] ≠ blank` and
`l'[s] ≠ l'[s-2]` — the same repeat rule, forbidding a jump over the blank that
separates a doubled label. Minimum feasible length is `L + #repeats`.
The Viterbi score is verified against the exhaustive maximum over collapsing
paths.

`FrameTimeMapper` converts frames to seconds through hop, window and encoder
subsampling, so a timestamp refers to audio actually observed. This is validated
against *physical* ground truth: tone bursts are synthesised at known times and
each token's predicted interval must overlap the interval in which its tone
genuinely sounded.

### 8.3 Commitment and revision

Commitment requires two independent forms of evidence — **beam agreement**
(prefix common to the top-`k`) and **temporal stability** (that agreement
persists for `stability` updates). Either alone commits too eagerly.

Committed text is **immutable and monotone**: a rendered sign cannot be
retracted for free. Later evidence can therefore contradict a commitment; that
event increments `commitment_errors` rather than being silently absorbed, since
it is the only signal that the policy is too aggressive. Tests confirm the
conservative policy commits no more errors than the reckless one on identical
audio.

`revision_rate` is defined per *position emission* (each update emits
`len(full)` positions; a position counts as revised when its token differs from
the previous update's), which makes it comparable across streams of different
length and update cadence.

### 8.4 Stage 2 findings

**8.4.1 Loss reduction is not evidence of a working decoder (weak test, fixed).**
The first integration test asserted `loss < 0.2 × first` and *passed* on a model
emitting blank at 100% of frames and decoding the empty string: CTC's all-blank
local optimum cut the loss ~10× while learning nothing usable. Acceptance is now
decoding accuracy against the known transcript. This is the single most
important lesson of the stage — a metric that moves is not a system that works.

**8.4.2 Appending confident audio cannot contradict a committed prefix.**
An attempt to provoke a commitment error by appending frames favouring a
different token failed, because appended evidence *extends* a CTC prefix rather
than reinterpreting fixed earlier frames. Real contradictions arise when
accumulating **ambiguous** frames reorder the beam; the test now uses such a
stream. The mechanism was correct; the scenario was not.

**8.4.3 Float32 log-softmax breaks exact probability identities.**
Assertions like "posteriors sum to 1" cannot hold to 1e-9 when the *input*
already sums to 1 ± 1e-7. Decoding tests use float64 inputs so a tight tolerance
tests the algorithm rather than the input's precision.

## 9. Stage 3 — calibration and failing closed

### 9.1 Why calibration is a safety requirement here

A sign is a fluent, confident assertion. A viewer cannot tell a guessed sign
from a correct one, so emitting a plausible sign for an uncertain word is
*worse* than emitting nothing: it destroys the viewer's ability to detect the
error. Silence and fingerspelling are both legible as "unsure"; a wrong sign is
not. The fail-closed rule follows — and it is only as good as the confidence it
thresholds, which is why calibration and policy are one stage.

### 9.2 Brier score and why not accuracy

The Brier score is **strictly proper**:

```
E_{y~q} BS(p, y) = Σ_k p_k² − 2 Σ_c q_c p_c + 1
```

is uniquely minimised at `p = q`. Optimising it therefore pushes the model to
report its true uncertainty. Accuracy is *not* proper — it is invariant to any
monotone distortion of the probabilities, so it cannot detect miscalibration at
all. Propriety is verified numerically in the tests.

**Murphy's decomposition** (exact when grouping by unique prediction value):

```
BS = REL − RES + UNC
```

This separates *being wrong* (reliability) from *being uninformative*
(resolution). A model that always predicts the base rate is perfectly calibrated
and completely useless — REL = 0, RES = 0. Reporting ECE alone would score it
well, which is precisely the trap the decomposition avoids. The identity is
asserted to 1e-12, and to 1e-9 on real model output.

### 9.3 Temperature scaling

`p = softmax(z / T)`, one parameter fitted on held-out data by minimising NLL.
Because `T > 0` is strictly monotone it **cannot change the argmax**: accuracy
is preserved exactly and only confidences move. That is what makes it safe to
apply post-hoc. Tested by recovering a *known* 3× overconfidence factor
(`T ≈ 3`) and by reducing ECE on both synthetic and real posteriors.

### 9.4 The policy

Two axes, not one: calibrated **confidence**, and **lexicon coverage**.

```
c ≥ emit_threshold        and token has a sign   → EMIT
c ≥ fingerspell_threshold and token is verified  → FINGERSPELL
otherwise                                        → PAUSE
```

Fingerspelling an *unverified* token is forbidden: it would merely move the
hallucination from the sign channel to the spelling channel. A confidently
recognised word with no sign (names, technical terms) is fingerspelled rather
than approximated — standard practice, and the honest option. Actions are
ordered `PAUSE < FINGERSPELL < EMIT` and the policy is proved monotone in
confidence over an exhaustive grid, with `emit_threshold ≥
fingerspell_threshold` enforced so the safer action never demands more evidence.

Selective-prediction metrics (coverage, selective accuracy, AURC) report the
quantity the system is really optimising: accuracy *on what it chose to assert*,
traded against how often it asserts anything. Coverage is provably non-increasing
in the threshold; the accuracy gain is demonstrated on real noisy output.

### 9.5 Stage 3 findings

**9.5.1 The policy crashed on valid input (real bug, fixed).** Lattice
posteriors are sums of floats and legitimately land an ulp outside `[0,1]`
(observed: `1.0000000000000002`). Strict range validation raised `ValueError` on
genuine model output. Fixed by tolerating rounding (±1e-6) and clamping, while
still rejecting real violations such as 1.001 — tolerate noise, catch bugs.

**9.5.2 A calibration set must match the evaluation distribution (test bug,
fixed).** Frame posteriors were stored grouped by noise level, so a sequential
50/50 split fitted `T` on clean audio and evaluated it on noisy audio. Under
that shift temperature scaling *worsened* ECE (0.066 → 0.080). Shuffling before
splitting fixed it. Temperature scaling minimises NLL, not ECE, and offers no
guarantee under distribution shift.

**9.5.3 Degradation appears as deletions, not substitutions.** Adding noise did
not make the recogniser produce *wrong* tokens so much as *no* tokens: the
usable band was narrow (clean below ~0.05, collapsed above ~0.12), and levels
chosen by intuition (0.3–1.5) yielded 18 tokens instead of 90. Statistics are
now accumulated by repeating draws inside the informative band rather than by
increasing severity — a reminder to characterise a perturbation empirically
before building an evaluation on it.

## 10. Stage 4 — the composite objective and freeze-first

```
L = L_ASR + λ_c·L_contrast + λ_b·L_boundary + λ_cal·L_Brier
```

The central risk in a multi-term loss is a term that is *summed in but inert* —
mis-shaped, mis-wired, or gradient-blocked — while the total still falls because
the other terms carry it. Every term is therefore judged by **its own metric**
through ablation, never by the total.

### 10.1 Boundary supervision

Per-frame word-start targets come from a CTC forced alignment of the known
transcript, computed under `no_grad`: the alignment is a *target*, not a
differentiable path — letting gradients flow through it would let the model move
the goalposts rather than fit them.

Boundaries are rare (≈`L` positives among `T` frames, often under 5%), so an
unweighted BCE is minimised almost perfectly by predicting "never a boundary":
it looks converged while detecting nothing. The loss is class-balanced by
`pos_weight = #neg/#pos` (capped), and a test asserts the degenerate all-negative
predictor is properly penalised.

### 10.2 Freeze-first

Asymmetric risk motivates the schedule: a pretrained encoder is the most
valuable, least replaceable component, and a large gradient through it while a
randomly-initialised head still emits nonsense can destroy representations that
cost enormous compute. Phase 1 trains only adapters and new heads; phase 2
releases the top `k` blocks at `encoder_lr_scale ×` the base LR.

The optimiser is **rebuilt** at the transition rather than extended with
`add_param_group`: newly released parameters must start with clean moment
estimates, not inherit phase-1 momentum. `encoder_lr_scale ≤ 1` is enforced —
the pretrained encoder should never train faster than the adapters.

### 10.3 Stage 4 findings

**10.3.1 LoRA injection silently broke the model (real bug, fixed).**
`inject_lora` replaced `out_proj` inside `nn.MultiheadAttention`, which PyTorch
reaches **attribute-wise** (`self.out_proj.weight`). The wrapper type-checks,
passes a structural test, and then raises `AttributeError` on the first forward
pass. The Stage 1 test missed it because **it never ran a forward pass after
injection** — it only asserted which modules were replaced and which parameters
were trainable. Injection now skips such parents by default
(`allow_unsafe_parents` to override), and the Stage 1 test executes the adapted
model. Lesson: a structural assertion about a model is not a test of the model.

**10.3.2 The boundary term works (positive result).**
Ablation with identical seed and data, differing only in `λ_b`:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| with `λ_b = 0.5` | 0.533 | 1.000 | 0.525 |
| without (`λ_b = 0`) | 0.034 | 0.068 | 0.042 |

(train-fit boundary F1). Decisive on every seed. Held-out generalisation is much
noisier at this scale (one seed collapsed to 0.0), so the test asserts the
reproducible claim — the term fits its signal — not a generalisation claim.

**10.3.3 The Brier term shows no measurable ECE benefit (negative result).**
Mean frame ECE over three seeds: **0.0376 without** the term, **0.0395 with**
the full objective — indistinguishable. A single-seed run was worse still
(0.035 → 0.169). Two plausible causes: the Brier targets are derived from a
forced alignment of the model's *own* posteriors, so the term partly trains
toward its own beliefs; and blank-dominated frames leave it little to correct.

This is recorded rather than tuned away. Stage 3's temperature scaling remains
the mechanism that demonstrably improves calibration, and **no claim of ECE
improvement from `λ_cal` is made anywhere in the codebase**. A test pins the
observation so it cannot be quietly forgotten, and deliberately does *not*
assert `ece_with < ece_without`, because that is not reproducible.

**10.3.4 InfoNCE is computed, not believed.**
Per the source document, a low contrastive loss is not evidence of semantic
equivalence: batch-level retrieval can be solved by speaker, channel or duration
cues. `speech_sign_retrieval` reports recall@k as the *necessary but
insufficient* check the document asks for; the downstream ablation in Stage 5
is what would actually decide the question.

## 11. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| **1** | Front-end: log-Mel, prosody, streaming+latency, resampler+gated projection, LoRA | **done** |
| **2** | CTC beam search → N-best/lattice, word timestamps, committed/uncommitted revision | **done** |
| **3** | Confidence + calibration (Brier, temperature scaling, ECE) + fail-closed policy | **done** |
| **4** | Training objective: boundary + Brier + symmetric InfoNCE; freeze-first schedule | **done** |
| 5 | Evaluation harness: WER/CER, timestamp error, ECE, revision rate, latency percentiles, condition ablations | planned |

Stage 1 is deliberately the signal-processing foundation: every later stage
consumes its output, and it is the layer whose correctness can be *proved*
(Parseval, partition of unity, scale invertibility, streaming equivalence,
analytic latency) rather than merely smoke-tested.
