# Stage S2 design: masked decoding over d2

Written 2026-10-31, 18:07 UTC, from 8540fea. Nothing is declared; nothing assumes the C20 result. Numbers are on the 1,000 unless marked.

## 1. The one question

**Does removing the history path let text reach the codes of unseen sentences, or does this data hold too little generalising text-to-motion information for any decoder?**

By the chain rule, a calibrated model's total text gain equals I(text; codes) in any decoding order. Masking cannot create information; it can only stop the autoregressive (AR) prior underfitting it. P1 falsifies cheaply: if a history-free head gains no more than AR on the same clips, S2 closes untrained.

S1's reading is thinner than the plan states. Only about 0.2 nats per stage-1 code is available (D20: 3.3 tokens/s × about 6 nats ÷ 96 codes/s), and readable content needs about 7 nats/s (half the tokens at ln 64). Gain on the true code: flan-T5 +1.42 nats/s, byte +0.77, W14b −0.51. W14b moves codes most (KL 3.20 nats/s) but the wrong way, which looks like overfitting.

**Stored reads that change the plan:**
- S1's second condition fails below stage 4. Share of d2's BLEU gap: stage 1 0.25, stages 1–2 0.62, stages 1–4 0.93 (bar 0.8).
- The bar ρ_faith < 0.8 is void: on DTW, blank reads 0.778, W14b 0.788. Time-averaged ρ works: blank 1.004, W14b 0.984, stages 1–4 0.192.
- Likelihood minus blank: stages 1–4 +0.019, all stages +0.055, composition +0.055 to +0.080. A faithful decode cannot beat composition on likelihood.

## 2. Options, ranked

Owner exclusions: retrieval hints, gloss. Continuous flow would discard d2.

| Rank | Option | Codes/s | Codes in 5.3 s | Feedback depth |
|---|---|---|---|---|
| 1 | Masked stage 1, then 3 text-conditioned parallel passes for stages 2–4 | 96 + 288 | 2,042 | 10 + 3 |
| 2 | Option 1 on 12 Hz keyframes plus infill, if P2 passes it | 48 + 144 | 1,021 | about 15 |
| 3 | Joint masked, 16 heads: residual stages depend on earlier ones, so parallel heads are incoherent | 384 | 2,042 | 10 |
| 4 | AR, stages 1–4, flan-T5: control only (rounds 4, 16, 19 null) | 384 | 2,042 | 128 |
| now | W14b | 768 | 4,083 | 128 |

**Error accumulation.** AR injects 2.33 nats × 96 = about 224 nats/s of sampling noise into its own history, against 1.42 nats/s of text. Masking cuts feedback depth from 128 to 13, which bounds drift, but wrong anchors persist.

**The bottleneck.** Fully masked, only text and clock are inputs, so the copy path is gone. A 10-iteration cosine schedule commits about 6 of 510 stage-1 codes first, where text can concentrate. Frozen multi-token flan-T5-base (existing `text_model` path) is our best held-out reader. No paper measures text reliance by decoder; G2P-DDM's 5.94 → 6.98 is closed-domain.

**Curriculum.** `round14b-mixed-v1` train, W14b's mix: Sem-Lex 26,698 records (9.9 h), RIT 5,295 (2.7), ASLLVD 5,155 (2.5), WLASL 1,045 (0.6), ASLLRP 1,652 (1.9), How2Sign 18,289 (33.9), YouTube 99,918 (151.2); total 158,052 (202.7 h). Never ASL Citizen or STEM Wiki; 2M-Flores stays held out. By tokens: words (≤ 1) 33,452; phrases (2–8) 56,826, of which YouTube 47,743, How2Sign 3,698, ASLLRP 391; sentences (≥ 9) 67,774. `round19-mixed-v1` (YouTube 173,261, 264.5 h) is a later arm only.

## 3. Probes first

Bars are declared before each run. Times are estimates.

| Probe | Pass bar | Where, script |
|---|---|---|
| **P1, text without history.** Small masked head (3.3M parameters) on cached stage-1 codes with flan-T5 tokens; chain-rule gain over empty text, coarse-to-fine reveal, D20 codes, length given to both branches | Gain ≥ **2.84 nats/s** and ≥ 2× flan-T5 AR on the same clips; shuffled ≤ 0.2; each signer group > 0. Own text among 100, 86 short clips: top-5 ≥ 15%. Word-clip 64-way top-1 ≥ 0.3 on held-out signers, else void | Mac: about 8k c02 clips, under 30 min. A fail is confirmed on one pod, about 1 GPU-h. New `s2_text_info_probe.py`; `s1_text_kl.py` gains a D20 list |
| **P2, code noise.** D20 stages 1–4: clean; i.i.d. 10, 25, 50%; 0.25 s blocks 25, 50%; ±0.25 s jitter; 12 and 6 Hz keyframes | Blocks 25%: BLEU ≥ 2.328 (D20 composition), lower bound above blank. Jitter within 0.3 of clean. Keyframes ≥ 0.9× stages 1–4 share, else option 2 drops | Mac: BLEU and ΔNLL, 5 processes, under 15 min. v2 on a pod. New `s2_noise_curve.py` |
| **P3, length.** Real resampled to predicted lengths, D20 | Pooled ratio in [0.95, 1.10]; paired ΔNLL lower bound > −0.03 | Mac, 5 min. Extend `fit_duration_model` |
| **P4, amplitude.** Real rotations ×0.8, 0.9, 1.1, 1.27, D20 | Not a stop. If ×0.8 has ΔNLL lower bound > 0, motion bands bind and v1-only gains do not count | Mac, 10 min. New `s2_amplitude_probe.py` |
| **B0, baselines.** W14b end-to-end on D20, gains and no gains, reference and predicted lengths, PMI ranks | Baseline | GPU evaluation pod, about $5. `s0_ceilings.py` |

**Local checks run for this design (descriptive; no val instrument read).** On 600 c02 clips, wrist speed over real: stages 1–4 0.97×, so not damped at the wrist; stage 1 0.84×; 25% block noise 1.26×, so code noise alone does not produce 2.6×; 12 Hz keyframes 1.00×. v1 reads 0.31 s/clip on Mac CPU. On D20 metadata, a c02 length fit overshoots 1.21× (val signers 0.28–0.33 s/token against 0.366); a signer-balanced c02 + c04 fit reads 1.08×, median error 25%. That read precedes P3's bar, so the ratio is not blind.

## 4. Evidence test for the owner's rule

**No training unless all hold on D20:**
1. P1's gain ≥ 2.84 nats/s, against flan-T5 AR's 1.42 and W14b's −0.51.
2. P2's curve, read at the DTW-aligned latent error of P1-head samples, predicts BLEU at least 0.2 above B0's W14b no-gains read (2.000 on the 1,000), and ΔNLL at least 0.10 above W14b's.
3. Those samples' time-averaged ρ is below W14b's, lower bound above 0.

R19 (1.601) was the last training round; W14b no-gains is stricter. My judgement, not a measurement: about one chance in three that P1 passes.

## 5. Risks and where this could be wrong

- **Too little information.** Three retrieval encoders stall at R@3 0.26 of 32 (real 0.29) on about 20k clips. If that is the limit, P1 fails cheaply.
- **Data.** YouTube is blocked, the UWB lever closed; more pairs were null in Hong & Košecká and R19.
- **Instruments.** v1 reads arms and hands only, and cannot separate real ASLLRP from blank on BLEU (0.708, 0.637). Dropping gains adds likelihood +0.033 on real round trips, yet a 6 Hz low-pass changes nothing, so the damped preference may be amplitude (P4). F12 reads R@3 0.266 (real 0.294) while v1 reads it at blank.
- **Ceiling.** Perfect stages 1–4 read 3.263 with 6% of the likelihood gap. S2 cannot reach 3.8; S5 stays a precondition.
- **Proxy.** A valid sample with other timing has high latent error, so P2 may understate.
- **Length.** Slower output loses BLEU (real at 0.8×: −0.234), so compared arms share lengths.

## 6. Declaration-ready pilot gates

**Arm `S2P`:** option 1, d2 frozen, no gains, predicted lengths, no best-of-N in sentences. Words and phrases 10–20 GPU-h; sentences 50–100 GPU-h after the phrase gate.

**Comparators:** W14b regenerated at the same lengths; `a4-w14b-n16-r10` (D20: 2.328, +0.065); `a7-nogains-r10` (2.186, +0.043).

**Sets:** D20's 500 sentences and 438 labelled words; phrases are D20's 86 clips ≤ 8 tokens plus 200 YouTube monitor clips ≤ 8 tokens. Never C20 or the 1,000. The at most 562 fresh clips stay unread, for one final claim (±0.44).

**Controls:** real, blank, shuffled, mismatched text, stages 1–4 of the same clips, length ratio, wrist speed.

**Rules, paired 95%; a failed rung stops the pilot:**

| Rung | Must hold |
|---|---|
| Words, recogniser v3d | All-samples top-1 lower bound > 0 against W14b (0.659), point ≥ R16 (0.719). Best of 16 lower bound > −0.02 against W14b (0.938) |
| Words, composed | Against `a7`: BLEU and ΔNLL points > 0 on v1, same signs on v2; recogniser v2 spotting lower bound > −0.02; wrist speed below 2.36× |
| Phrases | PMI-MRR and time-averaged ρ lower bounds > 0 against W14b, v1 and v2. 286 clips is thin: a null is not proof |
| Sentences | Against W14b no-gains: BLEU, ΔNLL, PMI-MRR lower bounds > 0 on v1, same signs on v2. BLEU lower bound > 0 against blank. Wrist speed 0.8–1.25×, spread ≥ 0.85×, length ratio ≤ 1.51 |
| Replacing composition | BLEU lower bound > 0 against `a4-w14b-n16-r10`, v2 agreeing. Likelihood reported, not gated (section 1); this exemption must be disclosed with any claim |

If v2 is not accepted, reads are provisional and no progress is claimed. Passing words enter composition on their own merit.
