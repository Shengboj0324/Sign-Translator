# After S2 closes: design plan

Written 2026-11-01, 00:04 UTC, from f721ba9. **Design only: nothing is declared, nothing here ran on a pod, nothing was written to R2.** Every "gate" below is declaration-ready text, not a rule. New numbers in section 1 come from the stored D20 reports (`round20/phase-a/`, downloaded read-only, read on the Mac at 1 thread) with the repository's `corpus_bleu`; it reproduces real 4.495, blank 1.68, a1 2.219, a4 2.328, a9 2.306. Numbers for the 1,000 and C20 are taken from the plan's records, not re-read. "Share" means (form − blank) / (real − blank) on the same set and scale.

## 1. The key question: how much can a better tokenizer alone raise composition?
**Answer: about 0.03 to 0.28 BLEU on D20, which is 1 to 10 points of its 2.815 gap, by point estimates with wide intervals (the upper bound is +0.70), and most likely near the bottom of that range.** Real word clips with no tokenizer in the path read about what generated words through d2 read. The path is confirmed clean: `oracle_stage2` runs with no checkpoint (`round15_compose_eval.py` line 38, report `checkpoint_key: null`), and "oracle words have no decoder" (line 1074). The bigger limits are which words get signed, and how the reader behaves on stitched input.

| Layer | Evidence (D20 unless marked) | What it limits |
|---|---|---|
| **Tokenizer + gains** | a1 (no tokenizer) − a7 (same words generated, d2, no gains): +0.034 [−0.434, +0.50]. a1 − a0 (with gains): +0.276 [−0.156, +0.696]. On the 1,000, oracle 2.17 → `@recon` 1.90, gains on (amendment 8; the plan gives no paired CI) | ≤ 10 points, and removing gains already recovers +0.242 (a7 − a0). On real sentences d2 keeps 66% of the gap (3.364 on the 1,000); that cap binds only once composition nears it |
| **Word form and choice** | a4 (best of 16) − a0 +0.385 [+0.014, +0.791]. Stacked on a perfect word source: 2.219 + 0.385 ≈ 2.60, 33% of the gap (additivity assumed) | About 14 points, already mostly taken |
| **Word coverage** | Half of the references' content tokens are never signed under r10 (3,881 of 7,829). Simulated reader, below: 33% [26, 42] of the gap under r10, 61% [53, 70] under r14b. 86% of real's 4-gram matches contain a word r10 does not sign | **Largest single limit: about 67 points at r10** |
| **Reader loops on stitching** | Loop share (a 3-gram repeated ≥ 3 times): a1 54%, a4 47%, real 28%, blank 23%. Length ratio a1 1.64, real 1.32. With repeats collapsed in the text (real 5.42, blank 1.77): a1 44% [29, 62], a4 47% [30, 66], a9 50% [31, 71], against raw 19%, 23%, 22% | Up to about 25 points, **if** loops are an instrument effect (section 1.2) |
| **Timing** | Tempo oracles span 1.972 to 2.350 against a1's 2.219; best +0.131 [−0.201, +0.486]. Wrist speed barely moves (2.26–2.31×) | −9 to +5 points on BLEU. Binds the release bar, not this score |
| **Grammar, non-manual** | v1 reads arm and hand points only, so face forms equal real on v1. On ASLLRP, English-order words read 0.932, gloss-order 0.675, real 0.708 (audit, 400 utterances) | **Invisible on v1, by construction and by that read.** Only v2 (face) and Deaf raters can credit it |

BLEU here rests on thin counts: over all 500 sentences, 4-gram matches are real 179, a1 85, a9 74, a4 72, blank 33.

### 1.1 The coverage read (a simulation)
The simulated reader takes real's clipped n-gram matches when every token is signed, a stop word or a contraction piece, and blank's matches otherwise. So it reads signed material as well as real and unsigned material only as well as blank. Adding the most frequent unsigned tokens by D20 count raises it:

| Added tokens | 0 | 10 | 30 | 100 | 300 | 1,000 |
|---|---|---|---|---|---|---|
| Content tokens covered | 50% | 68% | 74% | 80% | 87% | 97% |
| Share of gap [95%] | 33 [26, 42] | 62 [52, 69] | 77 [69, 83] | 87 [81, 93] | 90 [84, 95] | 99 [96, 100] |

The top ten are you 365, it 224, your 146, i 138, this 100, so 96, going 89, or 64, just 63, as 49. **None of these is in r10's lists**; r14b has you, it, your, going and or, but not i, this, so, just or as.

Limits of the read: it is not a hard bound. Treating pronouns as free raises r10 to 56%; a1 itself beats blank on 4-grams with unsigned tokens (72 against 33), so context does help; ranks use D20's own counts. It also does not say coverage is sufficient: r14b's real-word oracle stayed at 2.213 (hands tracked 0.58×), and generated r14b fell to 1.700 at wrist 3.70×, because 12.2 words instead of 9.1 were squeezed into the same time. **Coverage is blocked by timing, so they must be tested together.**

### 1.2 Does the instrument under-credit stitched signing?
**At word level, no.** Of the reference content words composition signs, the hypothesis repeats real 13.8%, a4 13.7%, a1 12.0%, blank 3.2%. Lift over blank relative to real: a1 0.83 [0.66, 1.06], a4 0.985 [0.79, 1.22], a9 1.14 [0.93, 1.40]. For unsigned content words, real lifts 6.0 points and a1 almost none (0.03 of real). This is a new read and it also shows how weak v1 is: only 14% even on real.

**At sentence level, possibly.** Stitched input loops about twice as often as real, which costs precision. Collapsing repeats is a text edit, not a declared metric, and it can't replace BLEU. Shuffled words rise on it too (34%, and a1 − shuffled is +0.361 [−0.175, +0.925]), so what it recovers is mainly word presence, not order. On that collapsed scale composition sits at the r10 simulated limit (38% [31, 47]), which says stitching costs little beyond coverage and loops. The evidence against a large under-credit: likelihood, which cannot loop, gives a1 11% and a9 29% of its gap; and in Jiang et al., sign.mt got 29% of the reference from Deaf raters against 23% in raw BLEU ratio (arXiv 2510.07453, Table 5; no blank floor; the fetch tool flagged that row's column alignment as uncertain), so raters and back-translation roughly agreed there.

**What would show it, cheapest first:**
1. Seam ablation on real motion (section 5): if linear bridges in real signing raise its loops to composition's level, loops are caused by seams.
2. Loop share and word echo on v2, on the same forms, computed from its stored hypotheses.
3. Deaf raters with **a1 beside a4 and real**. Raters reading a1 well above its 19% would be direct proof. About $900 per added system at the rater plan's assumed $40/h.

## 2. Order the evidence supports
The owner leaned toward tokenizer, then timing, then learned layer. The evidence reorders the first two and adds a fourth item:

1. **Timing, motion and coverage (A):** no training, about $5, required by the release bar, and coverage is the largest limit.
2. **Tokenizer probes (B):** run in parallel with A because they cost about $5; they are needed for 3.8 either way. **Training waits.** A new tokenizer alone can't raise composition, and it costs about 53 pod-hours plus the risk of losing W14b's words.
3. **Learned layer (C):** the seam ablation decides whether it starts. It can't add unsigned words, so it can't lift the coverage limit.
3.8 still needs a tokenizer: d2's 3.364 cap holds for anything decoded through it. Sequence matters, not need.

## 3. Workstream A: timing, motion, coverage (no generation unless noted)
**A1. Wrist-speed test from the stored `a9` cache.** Diagnosis figures at predicted timing (ledger 22:58; a9 wrist): baseline 1.86×; margin 0 1.55×; margin 0 + left arm ×0.5 1.25×; margin 0 + both arms ×0.7 1.20×. `--core-margin` and `--predicted-duration` exist; no arm-scale flag exists (searched `scripts/`), so that is a few lines in composition, plus tests. Read each arm against `a9@predicted` from phase a10 (running) with its `blank@predicted`: BLEU, likelihood, v2 spotting, wrist, spread, length, plus loop share and word echo.
- **Rule, declaration-ready:** both paired points > 0, v2 spotting lower bound > −0.02, wrist ≤ 1.25×, spread ≥ 0.8× a9's.
- **Control:** real D20 with the same arm scales. Scaling can pass the band by shrinking motion, and real at 0.8× tempo loses 0.234. If scaling real costs more than 10 points, the arm fails however composition reads.
- **Cost:** a10 is estimated at 55 GPU-min for two arms, so about 2.5 GPU-h, $2.70.

**A2. Why is the left wrist 2.70× and the right 1.77×?** Local, stored cache only, before any left-arm scale is adopted: per-entry left-arm speed on one-handed entries, by corpus and by core against bridge. Scaling could hide a defect (for example, a resting arm travelling to a bridge target). Mac, about 1 h.

**A3. Coverage with pace (reopens a closed route only by a new declaration).** r10 plus the top 10–30 tokens, using well-tracked clips, at predicted timing with margin 0. The reason it differs from the closed r14b read: that one added 388 Sem-Lex entries with 0.58× hand tracking at true length. Real-word oracle first (CPU or GPU, about 1 h); if both points beat a1, generate only the new entries (16 samples × 30 entries, under 15 GPU-min at the D20 cache's rate). ASL `i`, `you`, `your`, `this` are largely pointing and possessive handshapes, so check clip sources first (ASLLRP, Sem-Lex) and ask Deaf reviewers whether a citation point reads. **Not a gloss path:** English word to clip, as today.

**A4. Real-clip best of 16 (the true word ceiling).** a1 uses one draw per word, with hands tracked at 1.03× real against generated words' 1.19×. Choose real clips the way a4 chooses samples, by v3d. If this reads well above a1, word source still has room and the tokenizer bound in section 1 widens; re-state it then. About 1 GPU-h.

**Known limit, for the owner:** real signing itself misses the wrist band against How2Sign: YouTube 1.45×, ASLLRP 1.52×, RIT 1.37× (How2Sign train 1.06–1.09×). Only the owner changes the bar; nothing here assumes it moves.

## 4. Workstream B: tokenizer v3
**d2 today** (`run.json`): 4 parts, 8 stages, 512 codes, no time downsampling, 24 × 32 = 768 codes/s, nominally 6,912 bits/s; warm-started from s8 and stopped at step 15,000, best 5,000. Without gains: hand speed 0.84–0.90, hand variation 0.76–0.79, reliable-joint body speed 0.86.

**Math from stored reads (the 1,000):**
- **Rate is not the BLEU limit.** Share at stages 1, 2, 4, 8: 17, 41, 62, 66%. The last doubling buys 4.5 points. An exponential fit to 4 and 8 stages saturates at about 66%, so more stages of this design won't reach 85%.
- **Nor can rate alone close likelihood.** −0.119, −0.054, +0.019, +0.055; the last steps are about +0.036 per doubling, so reaching +0.306 would take about 7 more doublings.
- **So suspect the decoder or the training target** (damped hands). The first probe tests that directly.

### Probes (D20 reads; fits on c02 only; never c00; no training of the network)
| Probe | Forms | Reading | Cost |
|---|---|---|---|
| **B1 no quantiser** | d2's encoder → `decode_latents`, no codes | Near round trip: the decoder is the limit. Near real: the quantiser is | 0.2 GPU-h |
| **B2 per part** | Round trip on one part with the rest real, and real on one part with the rest round trip: body, left, right, fingers vs wrist | Which part holds the 34% loss (BLEU) and the 82% (likelihood) | 1 GPU-h |
| **B3 face ablation** | Mean face, no face. **v2 only:** v1 is blind to face | The Route report's retained-without-face figure is 81.6% (SignDino, not re-checked) | 0.5 GPU-h, after v2 |
| **B4 rate per part** | Stage grids per part by truncation; hands extended by 4 stages post hoc (k-means on residuals) | Which parts have spare rate; whether extra hand rate helps | 1.2 GPU-h |
| **B5 codebook sweep** | Post hoc K ∈ {256, 1024, 4096} on stored latents; a post-hoc FSQ grid at about 10 bits | Whether K or the quantiser form matters | 1.5 GPU-h |
| **B6 words** | v3d top-5 on decoded RIT and ASLLVD words for each surviving form | The word instrument agrees or not | 0.3 GPU-h |

Total about 4.7 GPU-h, $5. Every form reports BLEU, likelihood, loop share, length, wrist, hand variation. The S5 row's pass condition stands: one part, or the face, explains most of the loss.

### Candidates, in the order to try
| Id | Design | Math and evidence | What it invalidates |
|---|---|---|---|
| **V3-D** | **Same encoder and codes, new decoder**: fine-tune from d2, heavier hand velocity and acceleration terms, no gains | Runs if B1 shows the decoder limits. Data processing: it can't add information beyond the codes, only stop wasting it. Warm start is the only path known stable here | **Least: W14b's codes stay valid, no prior retrain.** Word caches must regenerate, since `round20_word_cache.py` stores states |
| **V3-H** | Per-part stages (Q_h, Q_b, Q_f), warm start | Codes/s = 24(2Q_h + Q_b + Q_f). (8, 4, 2) gives 528, −31%; compounding error scales about T², so −53% on that bound. (12, 6, 4) gives 816 if hands need rate. Runs if B4 shows unequal returns. No rate cut for hands' time: t2x/t4x reached hand speed 0.45 | Priors, caches, codes, recon indices |
| **V3-F** | Residual FSQ, levels [8, 6, 5, 5] (1,200 codes, 10.2 bits) per stage | Removes EMA, re-seeding and commitment (2309.15505). Our own d4 stop shows codebook dynamics drive the spikes: weights frozen, norm still rose to 17,892. Naive residual FSQ decays per stage, so use invertible layer norm (2508.15860). Motion: AnyMo uses it (2605.29488); VersatileMotion MPJPE FSQ 0.1216, RVQ 0.1352, VQ 0.1600 (2411.17335, Table 8); M3T face utilisation 78.8 → 99.0% (2603.23617). **No residual FSQ against residual VQ head-to-head on sign data found** | Everything V3-H does; the depth prior's heads |
| **V3-T** | Add a translation loss from a separate translator, never v1 or v2; variable rate only later | T2S-GPT 8.39 → 11.87 on PHOENIX14T (2406.07119); its How2Sign reimplementation fell well below SOKE (hand DTW 6.39 vs 2.35, 2411.17799, via the Route report). A new translator is itself a training job | As above, plus that translator |

From-scratch training is the last resort: t2x, t2xb, t2xc and d4 all became unstable between 8.9k and 31k steps.

### Gate, declaration-ready (no gains; D20 to select; the 1,000 once to confirm against stored d2, real clips only)
| Check | Bar | d2 today |
|---|---|---|
| **Round trip, v1** | **≥ 92% of the gap** (D20 4.27; the 1,000 3.954), paired lower bound against d2 > 0. Why 92: S2's stages 1–4 keep 0.93 of d2's share, and 0.85 / 0.93 = 0.914 | 66%; 3.364 |
| Tempo guard | Length ratio within 5% of real's; wrist 0.9–1.1×. BLEU rises with speed alone (+0.615 at 1.25×) | 1.32 vs 1.31 |
| **v2** | ≥ 85% of v2's gap, same sign as v1. Disclose: v2 trained on d2 copies, so it favours d2 | not read |
| **Likelihood gap, v1** | ≥ 50%, paired lower bound against d2 > 0. A judgement; composition already reaches 29% on D20 | 18% |
| MRR gap | ≥ 85% | 59% |
| **Hands** | Speed and variation each in [0.9, 1.1], jitter ≤ 1.2×; v3d decoded top-5 within 0.02 of real on RIT and ASLLVD | 0.84–0.90; 0.76–0.79 |
| **Code usage** | Stage-1 perplexity ≥ 0.9× d2 per part (body 146, face 132, left 58, right 99); for FSQ, utilisation ≥ 90% and the same perplexity bars | 163, 147, 64, 110 |
| **Prior load** | Codes/s ≤ 768 unless declared; stage-1 repeat share ≤ 39%; a small word head's free-run v3d top-1 lower bound ≥ d2's − 0.02 (2–4 GPU-h) | 768; 34% |
| Stability | Full run with no spike-guard stop; 2 seeds within 5% validation loss | d2 fine-tuned only |
| Domain, per signer | Round 3d's amendment 2 rule (1.5×); c01, c03, c05 each above d2; brow flaw (c05 browInnerUp 45×) no worse | passes |
| Hygiene | v1 and v2 in no loss, selection or reranking; no c00 | |
| **Downstream** | W14b's recipe on v3: all-samples top-1 lower bound > −0.02 vs 0.659, best of 16 vs 0.938. Composed on D20 vs a9: both points > 0 | |

### What a new tokenizer invalidates, and the bill (V3-H or V3-F; from job logs at the ledger's $1.09/h)
| Item | Record | Pod-h |
|---|---|---|
| 3 tokenizer candidates | d2 log 5.2 steps/s, so 60,000 steps 3.2 h, plus about 1.3 h of checks | 13.5 |
| Priors (F12 → W14b) | F12 about 12.5 h, 60,000 steps; W14b 03:22–08:13, 30,000 steps | 17.4 |
| Best-of-16 caches | D20 4.0 h, the 1,000 5.2 h, 562 clips about 3 h (estimate) | 12.2 |
| Composition reads | a4 about 30 min; the 1,000 about 1 h | 2.5 |
| Recogniser v3d: recon index and retrain | Recon pods 0.7 + 1.5 h; training 3–5 h declared | 7 |
| **Total, excluding v2** | **About 53 pod-hours, $57, about a week with reviews** (estimate) | 52.6 |

Also bound to d2, by key in 11 scripts (`round17_pt*`, `round18/19_train`, `state_back_translator.sh`, `s1_text_kl`, `build_recon_states`, `build_refiner_pairs`, `s0_ceilings`, `s2_lib`): S0/S1 baselines, S2's code caches and heads, plan codes. **v2's recon index** (`recon-d2-v1`, `recon_p` 0.4) needs a rebuild and retrain, hours unknown because v2 has not trained, or v2 stays frozen and is read as biased toward d2. **v1 is untouched** (2D keypoints, not trained on d2). Sets, r10 and timing code survive. Renders redo.

**Can word quality regress? Yes.** W14b's words took rounds 11–14b to reach; top-1 0.659, best of 16 0.938 are on d2. More hand stages add free-running decisions per word. v3d trained on d2 copies (G4) and would under-read v3 until retrained, while best of 16 relies on it, so read both old and new v3d. Keep W14b + d2 as comparator until the downstream row passes. V3-D avoids most of this, which is why it goes first.

## 5. Workstream C: a learned layer from stitched words to continuous signing
**The record first.** Refiners r1 and r2 failed (amendment 8: refined 1.62, rest-only input 1.61, so it ignored its input; amendment 9: source gain 0.014 < 0.05). Composed and real stage-1 codes agree at chance, 0.49% vs 0.42%. SignRR shows the same regime: refined 10.96 from retrieved motion, 1.01 from a periodic input (2608.28568). **A refiner can only polish content already present, and it can't sign a word that isn't there.**

**Probe C0, seam ablation (training-free, decides whether any of this starts).** On real D20: cut at activity minima at each sentence's word count; replace frames with linear bridges to fill 15, 30, 45% of frames (composition: 45% of frames, 41% of path). Second form: squeeze segments 1.96× with bridges filling the length. Control: same frame counts at random positions. Read BLEU, likelihood, loop share, word echo, v2 when accepted. No learned transition can do better than real ones, so **real minus this form bounds what perfect transitions could buy.**
- Keeps ≥ 85% of real's gap: seams aren't the limit; **no transition model trains.**
- Falls, and loop share climbs toward 54%: seams cause loops; L1 has its evidence. Precedent in-domain: ASLLRP own cuts keep 72% of the likelihood gap at true times, 55% on a fitted plan.
- About 1 GPU-h, $1. Local script, reviewed first.

| Id | Design | Trains on | Why it can't invent signs | Evidence to start | Hybrid or gloss structural change? |
|---|---|---|---|---|---|
| **L2 pace model** | Predicts each junction's frames and word durations from boundary joints and text; emits numbers only. Spoken2Sign's 4-layer MLP gave +3.5 dev, +4.64 test on PHOENIX14T (2401.04730) | ASLLRP gaps (gloss alignment as timing labels); How2Sign c02 pace | Emits no motion | Beats fitted durations on held-out ASLLRP error, then A1's rule on D20. Mac, hours | **No.** Gloss labels in training only; no gloss at inference. Say so to the owner anyway |
| **L1 bridge inpainter** | Masked in-betweening (CondMDI, 2405.11126; blended encodings, FlowMDM, 2402.15509). Word cores copied exactly; only bridges up to 0.6 s are generated. No text input | Real How2Sign c02 and YouTube sentences, random masks at activity minima, real as target. **Not composition pairs**, which is what sank r1 and r2 | Cores are bit-exact by construction. Guard: recogniser false spots inside bridges ≤ linear bridges, plus a mismatched-words control | C0 shows ≥ 15 points lost, and a local overfit test on 200 clips beats linear on held-out c04. Pilot about 10–20 GPU-h (estimate) | **Not a gloss path, and adds no retrieval.** It is a new learned stage after composition, so the owner decides; ledger 23:45 already flagged it |
| **L4 junction infill in codes** | Resample codes around junctions with a masked head, core codes clamped | Same as S2's head | Clamped core codes | Only if S2's P1 passes; bound by the tokenizer's cap | **No.** Same family as S2 |
| **L3 retrieve and refine** | SignRR's part RVQ refiner; random per-gloss concatenation → real sentence | ASLLRP only (How2Sign has no public gloss) | Weakly: its output stays near its input | S3's bar: ≥ 80% coverage reaches ≥ 50% of the gap. r10 covers 50%. Plan closed this route on evidence | **Yes, both.** Do not start without the owner |
| **L5 grammar layer** | Text → sign order, drops, non-manual marks (negation head shake, question brows). SIGNER's temporal gloss conditioning is the nearest method (2506.07460) | ASLLRP, Deaf review | Order and marks only, but a wrong plan changes meaning | **Raters only.** v1 can't credit it and prefers English order. Release bar needs negation (26 of 80 negated sentences have no cached negation word) | **Yes for reordering (gloss).** A narrow rule, head shake on existing negation words, is not; judged only by raters |

Owner's standing rule: no hybrid or gloss structural change without evidence. Today none exists for L3 or L5. A rater read of ASL-order against English-order real clips would be the first evidence.

## 6. First 72 hours after S2 closes (T0)
Assumes reviewed scripts are ready (section 7). Pods are L40S at $1.09/h unless noted; at most 5, including any v2 pods still running; none above $2/h; smoke first; delete when verified.

| Hours | Pods | Work | Cost | Decision at the end |
|---|---|---|---|---|
| 0–2 | 0 | Record S2's verdict, ledger, push notice. Compute loop share and echo on B0's stored forms | $0 | Which branch below |
| 2–8 | 3 | **P1:** A1 wrist arms and real control. **P2:** B1, B2, B4. **P3:** C0, A4 | $8 | Read each with skepticism; independent audit before any number is relied on |
| 8–12 | 0 | Independent review of the reads; A2 on the Mac; write the amended section 1 | $0 | Adopt or drop arm scale; B outcome (decoder, part, rate, or none) |
| 12–24 | 2 | **P4:** v2 reads of a1, a4, a9, B-forms and C0, if v2 is accepted, else recorded as provisional. **P5:** B5, B6, A3 oracle | $5 | Coverage route: declare or drop. L1: start or close |
| 24–48 | 2–3 | If B1 and B2 justify it: V3-D, 2 arms, about 4.5 h each. L2 on the Mac | $10 | V3-D gate rows, read on D20 |
| 48–72 | 1–2 | If V3-D passes: regenerate D20 words (4 h), compose against a9. Otherwise no tokenizer training; write up. Owner memo: L1, raters with a1, tokenizer bill | $5 | Owner decisions |

**Total about $28 (about 26 GPU-h), roughly 2.5% of the $1,146 spent so far.** All estimates.

**Branches.** *S2 fails:* end-to-end closes on this data, as agreed with the owner at 23:45; everything above proceeds, composition takes the compute. *S2 passes its evidence test:* its pilot runs on its own gates (words and phrases 10–20 GPU-h, $11–22) in one slot; S5 becomes more urgent because S2's ceiling is 3.263; L4 becomes eligible; V3-D is still preferred because W14b's and S2's codes stay valid.

## 7. What can start before S2 closes, changing nothing
- **Local, stored files:** rebuild section 1 as a reviewed script; apply loop and echo reads to a10's stored forms when they land. A2.
- **Code, on branches, unmerged:** arm-scale flags, C0 forms, B1–B6 forms, each with tests and review. No reads.
- **Add to v2's eval as report-only, offline:** loop share and word echo from its hypotheses, alongside the wrong-word read already committed at 21:57.
- **A1 itself:** the ledger already lists the wrist test as next, to declare. It can run when a slot is free, on its own declaration; no a10 or S2 result depends on it.
- **Owner:** whether the rater pilot adds a1. Rater build stays under review.
- **Not before S2 closes:** tokenizer training, any learned layer, the 562 clips.

## 8. What a new M1-style claim needs, with C20 spent
C20 is closed for composition. The only fresh data is at most 562 clips: **one read, ever, half-width about ±0.44** (minus-blank about ±0.45, scaled from C20's ±0.34).
- **Blank moves between sets** (1,000: 1.859; C20: 2.053; D20: 1.68), and shares for this arm read 31%, 17.5%, 23%. So the claim should be a **share of the gap with its own real and blank**, not an absolute: 2.55 was 30% on the 1,000.
- **Proposed claim:** share ≥ 30% at point, minus-blank lower bound > 0, on v1, v2 agreeing in sign, **at predicted timing** (only that form is deployable), with loop share, wrist and length disclosed, and a new amendment recording this change.
- **Spend condition:** D20 share ≥ 42% against `blank@predicted`, before the 562 are touched. Why: the SE is about 8.6 points on D20 and about 12 on 562 (C20's ±0.34 over its 1.98 gap, scaled), so about 14.5 on the difference, and 0.84 × 14.5 ≈ 12 points gives roughly 80% that the claim carries (normal approximation, ignoring selection on D20, which makes it optimistic). The last arm read 23% on D20.
- **Selection stays on D20.** The 1,000 has been read too often to count; optional, disclosed, never decisive. 2M-Flores is the separate cross-domain check. The test partition stays for the exit gate, once.
- No retry; arm and threshold are recorded before any cache is built for those clips.

## 9. The owner's rule, stage by stage
| Stage | Research | Local simulation or math | Previous round to beat | Trains? |
|---|---|---|---|---|
| A1, A3, A4, B1–B6, C0 | Section 3–5 | Training-free by design | Not applicable | **No** |
| L2 pace | 2401.04730 | Held-out ASLLRP error first, Mac | Fitted durations | Small, Mac, only after the error read |
| V3-D | B1, B2 outcome; T2S-GPT | B1 near round trip; decoder-only overfit on 200 c02 clips beats d2 on hand variation | d2: 66% BLEU, 18% likelihood | Only if both hold |
| V3-H | MoMask depth sweep; 2603.23617 | B4 returns per part; codes/s math; word head proxy | d2 | Only if B4 shows unequal returns and V3-D falls short |
| V3-F | 2309.15505, 2508.15860, 2605.29488 | B5 post-hoc FSQ at least equal to VQ at equal bits; stability on 20k warm-started steps | d2, d4's instability | Only if B5 holds and the prior can be rebuilt |
| Prior on v3 | Rounds 11–14b recipe | Gate's downstream proxy; composition math says tokenizer alone adds ≤ 10 points, so **needs a separate reason**, namely 3.8's cap | W14b words 0.659, 0.938 | Only after the gate, and only when composition or S2 nears d2's cap, or by owner decision |
| L1 inpainter | 2405.11126, 2402.15509 | C0 ≥ 15 points; 200-clip overfit beats linear on c04 | a9 on D20 | Only if both hold |
| L3, L5 | 2608.28568, 2506.07460 | No simulation can show it on v1 | a9, by raters | **No**, until the owner rules on evidence |

## 10. Odds (my judgement, not measurements)
| Event | Odds |
|---|---|
| An A1 arm passes its rule at ≤ 1.25× without the real control vetoing | 35% |
| A3 coverage oracle beats a1 on both points at predicted timing | 35% |
| C0 shows ≥ 15 points from seams | 40% |
| One part or the decoder explains most of the round-trip loss | 40% |
| V3-D reaches 85% on v1 / the 92% bar with likelihood ≥ 50% | 25% / 10% |
| W14b-recipe words on V3-H or V3-F no worse than d2's, first try | 50% |
| Composition, all levers, earns a fresh 30% share claim on the 562 | 25% |
| 3.8 on v1 with v2 agreeing, any route, within 3 months | 5–10% |
| Release bar met within 3 months | under 5% |

## 11. Not verified
- **Not run:** S2's outcome, a10, v2's acceptance, every probe here. Section 1's simulated reader is arithmetic on hypotheses, not a reader; its ranks use D20; a1 best of 16 could widen the tokenizer bound.
- **Taken from records, not re-read (hard limits or scope):** all figures on the 1,000 and C20; `@recon`'s −0.27 (no paired CI in the plan, gains on, unofficial shards read 2.14 → 1.82); motion figures (ledger; a1's 2.17× did reproduce); in-domain ASLLRP reads; L40S price; rater cost at an assumed $40/h.
- **Unknown:** v2's training hours; A2's cause; whether collapsed-text loops reflect the motion or v1's decoder; whether citation pointing reads in a stitched sentence.
- **References:** each arXiv ID below loaded a matching abstract or HTML page today, except where noted. Fetched text is what the tool returned; I did not read full papers. Excluded: SONIC's FSQ result (2511.07820; a search summary claimed it, the abstract does not show it). PoseStitch-SLT's abstract says 1.97 → 4.56 on How2Sign; the Route report's table says 2.62, so that number needs re-checking there. SignDino's 81.6% and SOKE's hand DTW are via the Route report, unchecked here.
- **Research reports** live untracked in `~/Desktop/Sign-Translator/reports/`, not on main.

## Sources
**Tokenizers:** FSQ [2309.15505](https://arxiv.org/abs/2309.15505); RFSQ [2508.15860](https://arxiv.org/abs/2508.15860); AnyMo [2605.29488](https://arxiv.org/abs/2605.29488); VersatileMotion [2411.17335](https://arxiv.org/html/2411.17335v2); M3T [2603.23617](https://arxiv.org/abs/2603.23617); MoMask [2312.00063](https://arxiv.org/abs/2312.00063); T2S-GPT [2406.07119](https://arxiv.org/html/2406.07119v1); SOKE [2411.17799](https://arxiv.org/html/2411.17799).
**Transition and refinement:** SignRR [2608.28568](https://arxiv.org/html/2608.28568v3); Spoken2Sign [2401.04730](https://arxiv.org/html/2401.04730); Sign Stitching [2405.07663](https://arxiv.org/abs/2405.07663); Signing at Scale [2203.15354](https://arxiv.org/pdf/2203.15354) (search result only); CondMDI [2405.11126](https://arxiv.org/abs/2405.11126v2); FlowMDM [2402.15509](https://arxiv.org/abs/2402.15509); SIGNER [2506.07460](https://arxiv.org/abs/2506.07460).
**Evaluation and data:** Jiang et al. [2510.07453](https://arxiv.org/html/2510.07453); Hong & Košecká [2606.01643](https://arxiv.org/abs/2606.01643); PoseStitch-SLT [2511.00270](https://arxiv.org/abs/2511.00270).
**Project:** `docs/NEXT_PHASE_PLAN.md` (S0–S5 table; rounds 3d, 10, 15 amendments 8–9, 17, 20; C20 verdict; release bar), `reports/s2_design.md`, `reports/rater_package_plan.md`, `docs/HUMAN_EVALUATION_PROTOCOL.md` §6, `signtranslator/generation/motion_rvq.py`, `round20/phase-a/*.json`, `training-runs/tokens-rvq-d0b-r3d-d2/`, job logs in `~/.sign-translator-ops`.
