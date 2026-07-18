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

## 7. Staged roadmap

| Stage | Content | Status |
|---|---|---|
| **1** | Front-end: log-Mel, prosody, streaming+latency, resampler+gated projection, LoRA | **this round** |
| 2 | CTC beam search → N-best/lattice, word timestamps, committed/uncommitted revision | planned |
| 3 | Confidence + calibration (Brier, temperature scaling, ECE) + fail-closed policy | planned |
| 4 | Training objective: boundary + Brier + symmetric InfoNCE; freeze-first schedule | planned |
| 5 | Evaluation harness: WER/CER, timestamp error, ECE, revision rate, latency percentiles, condition ablations | planned |

Stage 1 is deliberately the signal-processing foundation: every later stage
consumes its output, and it is the layer whose correctness can be *proved*
(Parseval, partition of unity, scale invertibility, streaming equivalence,
analytic latency) rather than merely smoke-tested.
