# Sign-Translator

Research software toward an industrially deployable English-to-American Sign Language (ASL) 3D translator. **This checkout is not a validated translator or deployed product.** Its runnable end-to-end path is a synthetic-data baseline; the richer linguistic, motion, avatar, governance, and deployment components are mostly separately tested research modules. Mathematical consistency and passing unit tests cannot establish ASL intelligibility, real-time reliability, or commercial rights.

This README describes implemented contracts, unresolved interfaces, and evidence requirements. The authoritative stage status is in the [roadmap](Sign%20Translator%20Stage%20Documentation/06_IMPLEMENTATION_ROADMAP.md) and [Phase 3C evidence register](Sign%20Translator%20Stage%20Documentation/12_PHASE_3C_EVIDENCE_INTEGRATION.md).

![Research architecture](docs/figures/architecture.svg)

## 1. The representation boundary

The target pipeline is `English speech/text → semantic plan → ASL linguistic structure → synchronized multichannel 3D motion → avatar`, plus a distinct sign-to-language analysis branch. The executable [`signtranslator/run.py`](signtranslator/run.py) does **not** traverse that full path. It links synthetic acoustic features, a compact CTC model, a source-token planner, ST-GCN motion recognition, contrastive alignment, and conditional diffusion over **27 Cartesian joints with three coordinates per joint**. Its generated output is a motion tensor, not rendered or human-approved ASL.

The synthetic corpus uses a fixed source/target concept-vocabulary bijection; its field named `gloss_tokens` is an API convention, **not** authentic ASL gloss. Real local How2Sign material is external to the repository and its frontal OpenPose observations are **2D image-plane coordinates plus confidence** for 137 landmarks (25 body, 21 per hand, 70 face). Neither English transcripts nor 2D landmarks can be relabeled as reviewed gloss, temporal ASL structure, depth, metric 3D, 6D joint rotations, gaze, or rig motion. The [architecture audit](Sign%20Translator%20Stage%20Documentation/02_CURRENT_ARCHITECTURE.md) tracks this mismatch.

| Type | Needed evidence | Forbidden substitution |
|---|---|---|
| Authentic gloss | Source-native sign annotations, convention, time alignment, provenance | English text, uppercase lemmas, generated candidate strings |
| Sign Intermediate Representation (SIR) | Qualified human mapping/adjudication of typed linguistic events and scopes | A transcript or an unchecked model prediction |
| 2D observation | Camera coordinates, confidence, timestamp, explicit validity mask | Metric 3D pose or anatomical joint rotation |
| Canonical 3D motion | Synchronized body/hands/face/eyes, coordinate system, timing, validated rig mapping | Fusing unrelated 2D, lexical, and face datasets into fictional paired samples |
| Production authorization | Exact-action rights, consent/identity treatment, asset lineage | A download link, research-only license, or code license |

## 2. Mathematics of the executable core

Notation: `N` batch size, `T` motion frames, `V` joints, `C` coordinate channels, `L` token length, and `d` latent width. The default synthetic corpus has `V=27`, `C=3`, `T=32` (see [`signtranslator/data/corpus.py`](signtranslator/data/corpus.py)). These are experimental dimensions, not a full ASL skeleton. The five branch losses below train against synthetic paired arrays, not qualified ASL supervision.

### Acoustic CTC: sequence likelihood, not linguistic meaning

The acoustic branch consumes a feature tensor `S ∈ ℝ^(N×T_audio×F)`, applies temporal convolutional subsampling, a Transformer encoder, and a framewise softmax over source tokens plus a blank. It minimizes

```text
L_speech = −log p(y|S),
p(y|S) = Σ_{π : B(π)=y} ∏_t p_t(π_t|S).
```

`B` removes blanks and collapses repeated token runs. A target `y` requires at least `|y| + Σ_i 1[y_i=y_(i+1)]` output frames: adjacent identical targets need an intervening blank. The code checks lengths, class IDs, and finite log-probabilities before `CTCLoss(zero_infinity=False)`; impossible alignment is not silently assigned zero loss. Greedy decoding is an algorithmic baseline, not calibrated ASR confidence. The separate [`speech/`](signtranslator/speech/) package studies STFT `X(m,ω)=Σ_n x[n]w[n−mH]e^(−iωn)`, Mel/log scaling, YIN pitch, CTC beam search, forced alignment, and Brier/ECE/temperature calibration, but is not the active microphone front end. See [`models/speech.py`](signtranslator/models/speech.py) and [speech design](docs/SPEECH_FOUNDATION.md).

### Planner: syntactic interface versus ASL grammar

A causal Transformer attends to encoded source tokens and minimizes teacher-forced target cross-entropy,

```text
L_plan = −Σ_(i,t) 1[y_(i,t)≠PAD] log p_θ(y_(i,t)|y_(i,<t), source_i).
```

The runnable experiment learns the synthetic vocabulary mapping, not English-to-ASL grammatical transformation. At inference an empty decoded target raises `TranslationAbstainedError` before motion generation. Independent [`planning/`](signtranslator/planning/) modules implement typed serialization, constrained decoding, lexicon/provenance checks, and Phase 3 evidence contracts; a schema-valid plan is still not a qualified linguistic judgment. See [`models/planner.py`](signtranslator/models/planner.py) and [semantic-planner design](docs/SEMANTIC_PLANNER.md).

### Graph motion encoding and the shared manifold

The reverse branch receives pose `P ∈ ℝ^(N×C×T×V)`. A spatiotemporal graph convolution aggregates graph neighbors and filters through time. Schematically, for adjacency partitions `A_k`,

```text
H^(ℓ+1) = TemporalConv(σ(Σ_k A_k H^ℓ W_k)) + residual(H^ℓ).
```

The 27-node adjacency is not the real How2Sign 137-node graph. Per-frame features drive sign-side CTC; time-pooled features drive a motion projection head. A separate target-token encoder drives a language head. Both heads L2-normalize their outputs. For paired unit vectors `u_i,v_i`, learnable scale `s=exp(clamp(log_scale,max=log 100))` gives symmetric in-batch InfoNCE:

```text
L_align = −1/(2N) Σ_i [ log(e^(s u_i·v_i)/Σ_j e^(s u_i·v_j))
                       + log(e^(s v_i·u_i)/Σ_j e^(s v_i·u_j)) ].
```

This is a retrieval objective, not proof that `u` or `v` encode ASL semantics. Source, signer, duration, and background can be shortcuts; the independent hard-negative and leakage probes exist to challenge them. A shared latent space alone does not make the forward and reverse directions linguistically inverse. See [`models/stgcn.py`](signtranslator/models/stgcn.py), [`models/alignment.py`](signtranslator/models/alignment.py), and [pretraining design](docs/PRETRAINING.md).

### Conditional Cartesian diffusion

For `β_k∈(0,1)`, `α_k=1−β_k`, `ā_k=∏_(j≤k)α_j`, the exact forward marginal is

```text
q(x_k|x_0)=N(√ā_k x_0,(1−ā_k)I),
x_k=√ā_k x_0+√(1−ā_k)ε,  ε~N(0,I).
```

The generator denoises **Cartesian coordinates**, conditioned on token memory through cross-attention. Configuration selects `ε` or `x_0` prediction; `x_0` mode can add a finite-difference velocity penalty `||Δ_t x̂_0−Δ_t x_0||²`. A configured high-noise timestep mixture changes the sampling of training steps; it is not a physical motion prior. The DDPM posterior uses

```text
β̃_k=β_k(1−ā_(k−1))/(1−ā_k),
μ̃_k=[√ā_(k−1)β_k/(1−ā_k)]x_0
    +[√α_k(1−ā_(k−1))/(1−ā_k)]x_k.
```

Ancestral and DDIM sampling and classifier-free guidance are implemented. Guidance `ε_g=ε_u+g(ε_c−ε_u)` has `g=1` at the conditional estimate; larger `g` extrapolates and can degrade diversity/quality. Low synthetic denoising error does not establish anatomical plausibility or intelligible signing. See [`models/diffusion.py`](signtranslator/models/diffusion.py), [`models/guided_diffusion.py`](signtranslator/models/guided_diffusion.py), and [diffusion design](docs/DIFFUSION_GEN.md).

### Joint loss and reproducibility

For a complete batch, [`BidirectionalSignTranslator.training_step`](signtranslator/models/pipeline.py) computes

```text
L = 1.0 L_generation + 0.3 L_alignment + 2.0 L_plan
  + 1.0 L_sign_CTC + 1.0 L_speech_CTC.
```

These are current synthetic weights, not a universal optimum. Missing task fields omit their corresponding branch. One ST-GCN pass is shared by recognition and contrastive pooling; per-token generation conditioning and pooled alignment use separate target encoders to reduce destructive objective interference. Under its documented CPU/epoch-boundary constraints, joint training can resume exactly from `last`: best/last checkpoints bind model, corpus, configuration and code identities, SHA-256 sidecars, optimizer/scheduler/history, and Python/NumPy/Torch RNG states. Generator-only fine-tune and polish are **not** state-complete resume modes, so checkpointing with either is refused. See [`training/`](signtranslator/training/).

## 3. Specialized mathematical stack and integration status

These modules implement separately testable primitives, **not** one integrated production graph. Each row gives the mathematical purpose and the evidence it still lacks. Unit tests verify encoded properties; they are not proofs of linguistic or empirical adequacy.

| Layer and documentation | Mathematics/design philosophy | Present limit |
|---|---|---|
| [01 Speech](docs/SPEECH_FOUNDATION.md) | STFT/log-Mel and pitch extract time-frequency evidence; CTC prefix search and forced alignment retain alternative paths; calibration/pause policy distinguishes probability estimates from action. | Compact acoustic CTC is active; no deployed ASR/audio calibration. |
| [02 Planning](docs/SEMANTIC_PLANNER.md) | Typed sign plans, a finite-state constrained serializer, versioned lexicon, and sequence preference objectives make structural errors detectable. | No qualified English→ASL plan corpus or automatic authenticity certificate. |
| [03 Grammar/SIR](docs/GRAMMAR_SIR.md) | Temporal graph `G=(V,E)` carries typed manual/non-manual events `[s_i,e_i]`; e.g. scope containment penalty `max(0,s_scope−s_sign)+max(0,e_sign−e_scope)`. Structural validity and temporal consistency are distinct. | SIR code exists, but English labels are not SIR annotations. |
| [04 Human pose](docs/HUMAN_REPRESENTATION.md) | 6D rotations are orthonormalized with Gram–Schmidt; `d_SO(3)(R,R̂)=arccos(clamp((tr(RᵀR̂)−1)/2,−1,1))`. SMPL-X/LBS maps shape, pose, and expression into mesh coordinates. | Licensed assets and approved continuous metric 3D data absent; 2D cannot be promoted to 6D rotation. |
| [05 Hand graph](docs/HAND_GRAPH.md) | Heterogeneous message passing `h'_i=Σ_rΣ_(j∈N_r(i))W_r h_j/c_(i,r)` retains edge types; wrist-relative local coordinates reduce global translation dependence. | Independent graph components, not a validated continuous motion model. |
| [06 Motion transformer](docs/MOTION_TRANSFORMER.md) | Residual VQ uses `r_(m+1)=r_m−c_m` and one straight-through gradient path; spectral penalties challenge temporal oversmoothing; causal chunks/SLERP address continuity. | No learned ASL-preserving token vocabulary. |
| [07 Advanced diffusion](docs/DIFFUSION_GEN.md) | VP parameterization triangle `x_k=a x_0+b ε`, `v=a ε−b x_0`, `a²+b²=1`; temporal DiT, adaLN-Zero, inpainting, few-step distillation. | Separate from the active Cartesian generator; no trained high-fidelity ASL result. |
| [08 Avatar](docs/AVATAR_RENDER.md) | Skinning, Kabsch retargeting with `det R=+1`, rendering, and appearance scoring are separated. Dual-quaternion skinning can mitigate some LBS twisting; it is **not** a general volume-preservation guarantee. | No sign-qualified rig, licensed production asset chain, or integrated renderer. |
| [09 Facial/NMM](docs/FACIAL_NMM.md) | Independent Bernoulli channels allow simultaneous brow, eye, mouth, head and torso signals; temporal scope links them to manual events. A softmax would impose false exclusivity. | No synchronized, reviewed facial/eye linguistic supervision. |
| [10 Data](docs/DATA_ENGINEERING.md) | Canonical schema, immutable hashing, source groups, and rights gates preserve evidence. On irregular clocks, `v_(t−1/2)=(x_t−x_(t−1))/Δt` and `a_t=2(v_(t+1/2)−v_(t−1/2))/(Δt_next+Δt_prev)`; `1.4826·MAD` is reported without dividing by zero in constant-MAD cases. | QC flags review, not anatomical/linguistic truth; real-source failures remain explicit. |
| [11 Pretraining](docs/PRETRAINING.md) | 2D masked reconstruction scores only *originally valid and artificially hidden* observations; point/span/region masks are compared with interpolation, last-value and mean baselines. Hard negatives test shortcuts. | Quarantined gloss-independent experiment, not 3D translation. |
| [12 Evaluation](docs/EVALUATION_FRAMEWORK.md) | Metric contracts form a conjunction with mandatory caveats; paired tests and source-cluster bootstrap quantify uncertainty. SignBLEU/text similarity cannot substitute for human comprehension. | Qualified independent ASL study not yet performed. |
| [13 Deployment](docs/DEPLOYMENT.md) | Display-commit monotonicity requires committed prefix `C_t` to remain a prefix of `C_(t+1)`; latency includes buffering, lookahead, computation, rendering, queueing. Bounded queues require explicit arrival/service assumptions. Numerical optimization gates check equivalence and quality. | No measured end-to-end SLA or deployed service. |

The intended multichannel target must eventually co-observe manual articulation, body, face, eye aperture/gaze, spatial reference, and variable timing. Concatenating unrelated datasets cannot manufacture that paired observation.

### Why a temporal SIR, rather than a list of sign words?

A sequence of lexical identifiers can preserve order but cannot directly express the fact that a brow raise overlaps an entire question, that a headshake negates a scoped clause, or that a pointing locus refers back to a discourse entity. The SIR therefore treats intervals and typed edges as part of the representation. If event `a` must precede `b` with margin `m`, a soft violation is `max(0,e_a+m−s_b)`; if a non-manual event must contain manual event `b`, the two-sided hinge in the table measures both early and late boundary violations. Such losses are only meaningful **after** a qualified annotator defines the relation: minimizing them cannot discover what ASL grammar ought to be. The serialized graph, relation validator, and differentiable temporal penalties are separate layers of evidence; a valid graph can still have bad timing. See [`grammar/`](signtranslator/grammar/) and [SIR design](docs/GRAMMAR_SIR.md).

### Why rotations and masks must remain typed

For 6D rotation vectors `a,b`, Gram–Schmidt computes `r₁=a/||a||`, `r₂=(b−(r₁·b)r₁)/||b−(r₁·b)r₁||`, `r₃=r₁×r₂`, then `R=[r₁ r₂ r₃]∈SO(3)` when both denominators are nonzero. Degenerate inputs must be rejected or explicitly treated; an image-plane wrist/elbow configuration does not determine a unique depth or joint rotation. Likewise, 2D confidence thresholding defines an observation mask, not a synthetic coordinate. The audit keeps invalid/out-of-frame/missing observations distinguishable and calculates derivatives only where the actual timestamps and required landmarks are present. Torso-relative quantities are dimensionless only if the anchor scale is nondegenerate. Threshold-grid coverage is monotone by construction (`M_(τ₂)⊆M_(τ₁)` for `τ₂>τ₁`); a coverage curve is **not** a learned optimum. See [`pose/`](signtranslator/pose/), [`data_engineering/how2sign_audit.py`](signtranslator/data_engineering/how2sign_audit.py), and [data mathematics](docs/DATA_ENGINEERING.md).

### Why hierarchical motion modelling does not imply expressiveness

In the advanced motion-token branch, a residual quantizer selects `c_m` from codebook `m` against residual `r_m`, sums `z_q=Σ_m c_m`, and subtracts `r_(m+1)=r_m−c_m`. A single straight-through estimator over the *whole sum* allows the encoder to receive the reconstruction gradient without accidentally detaching later stages. But low reconstruction error can coexist with an oversmoothed hand trajectory: spectral penalties compare temporal frequency content, and hand/face-region metrics must be reported separately. A generated coordinate trajectory still needs feasible joint rotations, contact, gaze, non-manual scope, and intelligibility checks before retargeting. In the render layer, `det(R)=+1` excludes a reflection from Kabsch alignment; PSNR/SSIM and frame rate quantify appearance/throughput, never signing comprehension. See [motion-transformer](docs/MOTION_TRANSFORMER.md) and [avatar](docs/AVATAR_RENDER.md) design.

### Why evaluation is conjunctive and streaming is monotone

An evaluation contract has a direction (`≤` for error, `≥` for coverage), threshold, population, and caveat. The readiness chain is a logical **AND** over required contracts: a favorable mean score cannot offset a failed comprehension or non-manual test. Report sample counts, confidence intervals, source/signer partitioning, and preregistered comparisons; paired permutation/sign tests can quantify differences, but they do not manufacture an independent ASL reference set. For deployment, distinguish first-output latency (buffer/lookahead plus critical-path computation) from throughput (steady-state bottleneck) and queueing. A committed display prefix `C_t` cannot be revised in a later output; only an uncommitted suffix may change. If arrival rate `λ` exceeds service rate `μ`, backlog grows without bound absent admission control. A capacity `B` plus throttling can bound queue occupancy, but any latency bound additionally requires a stated service-rate lower bound and scheduling assumptions. See [evaluation](docs/EVALUATION_FRAMEWORK.md) and [deployment](docs/DEPLOYMENT.md) design.

## 4. Weak-label and Phase 3 evidence controls

[`pseudo_gloss/`](signtranslator/pseudo_gloss/) is an **offline** candidate generator, not a gloss oracle and not connected to the production loader. It combines an order-aware text proposal and transcript-independent 137-node video CTC evidence over a finite lattice. Its ranking is schematically

```text
q(G|X,V) ∝ q_ψ(G|X)^α p_φ(G|V)^β exp[−λC(G,V)],
```

with candidate sequence `G`, English `X`, video `V`, and explicit constraint cost `C`. This does **not** solve identifiability: correlated text/video errors, transcript leakage, vocabulary collapse, and self-training confirmation bias remain possible. CTC path feasibility, source-disjoint calibration, abstention, model/input hashes, Unicode/size validation, provenance, and blank/shuffled-video interventions are evidence controls. Without independent qualified reference and activation approval, outputs remain `weak_gloss_candidates`; they are not promoted to authentic `gloss_tokens`. See the [research dossier](Sign%20Translator%20Stage%20Documentation/09_PSEUDO_GLOSS_MODEL_RESEARCH.md).

Phase 3A ingests publisher-native linguistic records; 3B governs project-human mapping/adjudication; 3C binds reviewed lexemes and forms to exact canonical Phase-2 motion payloads and authorization evidence. Lookup returns exactly `resolved`, `unknown_lexeme`, `ambiguous_form`, or `action_not_authorized`, never an English spelling or nearest-motion fallback. The **pre-artifact Phase 3C software boundary is complete**; governed corpus population, direct paired training/results, nonempty reviewed motion library, qualified validation, and commercial rights are not. See [Phase 3A](Sign%20Translator%20Stage%20Documentation/10_PHASE_3A_LINGUISTIC_REFERENCE_INGESTION.md), [3B](Sign%20Translator%20Stage%20Documentation/11_PHASE_3B_GOVERNED_MAPPING_AND_ADJUDICATION.md), and [3C](Sign%20Translator%20Stage%20Documentation/12_PHASE_3C_EVIDENCE_INTEGRATION.md).

## 5. Data sources: a primary-source acquisition map

This is a broad **ASL-relevant, task-directed** search, not a claim to list every global corpus. The [2026 ACL survey](https://aclanthology.org/2026.acl-long.1928/) indexes 120 resources across 35 sign languages; other signed languages are methodological comparisons, not interchangeable ASL labels. For every candidate, inspect the exact release, participant consent/identity policy, access, retention/withdrawal, model-derivative rights, and commercial training/deployment rights. A downloadable file or repository code license grants neither commercial media use nor linguistic ground truth.

| Primary source | What it supplies | Why it is not sufficient by itself |
|---|---|---|
| [How2Sign](https://how2sign.github.io/) / [CVPR paper](https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html) | >80 hours continuous ASL, English/speech alignment, multiple modalities; local frontal RGB/2D OpenPose lineage. | Site publishes **CC BY-NC 4.0**; local English is not authentic gloss, frontal tracks are not 3D, and commercial rights are absent. |
| [SignAvatars](https://github.com/ZhengdiYu/SignAvatars) / [ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00653.pdf) | Requested SMPL-X/hand/face annotations for isolated and continuous subsets, including How2Sign-aligned material. | Noncommercial request process, fitted rather than marker-observed 3D, no original video redistribution; inspect exact channels/text tier. |
| [RIT/CUNY MoCap](https://latlab.ist.rit.edu/downloads.html) | 98 unscripted stories/3 signers with timed gloss, spatial reference, English, BVH/FBX, and front/side/face MOV. | Strong co-observed inquiry; access by email, modest scale, exact schema and commercial rights unresolved. |
| [2026 ASL STEM dialogue MoCap](https://aclanthology.org/2026.lrec-1.669/) | Natural dialogue/lecture motion and linguistic timing. | Very small research source; access, capture channels, and permission need verification. |
| [3D-LEX/SAPA](https://github.com/OlineRanum/SAPA) | 1,000 isolated ASL signs plus NGT; Vicon body, StretchSense hands, ARKit face. | Publisher states **CC BY 4.0** for dataset and asks for access contact; isolated signs/limited ASL-signer coverage cannot supply sentence grammar. |
| [ASLLRP/ASLLVD](https://www.bu.edu/asllrp/indexright.html) and [ASL Signbank](https://aslsignbank.com/) | Source-native gloss conventions, multiview signing, lexical/continuous linguistic tiers. | Access and commercial terms vary; not a metric 3D corpus or How2Sign gloss release. |
| [Cokely parallel corpus](https://encompass.eku.edu/cokely_videos/) | Small ASL/English examples and EAF annotation for ingest tests. | CC BY-NC-SA 4.0 collection, narrow domain, no canonical 3D. |
| [CARD/SLAASh](https://sites.google.com/gallaudet.edu/card/data) | ELAN templates and linguistic annotation conventions. | Annotation infrastructure, not human-reviewed labels for these videos. |
| [ASL-LEX 2.0](https://asl-lex.org/download.html) | Lexical/phonological database of 2,723 signs. | Database CC BY-NC 4.0; reference **videos are excluded** and cannot be saved/reused without permission; no sentence timing. |
| [ASL STEM Wiki](https://www.microsoft.com/en-us/research/project/asl-stem-wiki/dataset-description/) / [license](https://www.microsoft.com/en-us/research/project/asl-stem-wiki/dataset-license/) | 64,266 sentence-aligned STEM videos, ~316 hours, 37 professional interpreters, consented collection. | Noncommercial terms; English alignment rather than full gloss/SIR; no metric 3D; published archive ~187 GB. |
| [YouTube-ASL](https://github.com/google-research/google-research/blob/master/youtube_asl/README.md) | Large video-ID/English-caption research index. | IDs are not owned media, consent, gloss, or a commercial grant; links can disappear. |
| [OpenASL](https://github.com/chevalierNoir/OpenASL) | Continuous ASL→English online-video research baseline. | CC BY-NC-ND 4.0 repository, third-party media rights, no 3D and wrong direction for motion supervision. |
| [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) / [license](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/) | Community isolated-sign visual recognition data. | Noncommercial research; isolated labels do not encode continuous grammar. |
| [PopSign v2.1](https://signdata.cc.gatech.edu/view/datasets/popsign_v2_1/index.html) | 200,686 isolated clips/562 signs/47 signers; device/viewpoint robustness. | CC BY 4.0, ~1.1 TB; its data card expressly excludes continuous translation as a suitable use. |
| [NVIDIA ASL 1000](https://registry.opendata.aws/asl_1000/) / [license](https://github.com/NVIDIA/Trustworthy-AI/blob/main/ASL%20Developer%20Community/NVIDIA%20Data%20License%20for%20ASL%20Project.pdf) | Controlled-access video and corrected 2D landmark evidence. | Purpose-limited, revocable terms; commercial/distribution scope must be resolved in writing. |
| [WLASL](https://github.com/dxli94/WLASL/blob/master/index.md), [MS-ASL](https://www.microsoft.com/en-us/download/details.aspx?id=100121), [SignAvatar/ASL3DWord](https://github.com/dongludeeplearning/SignAvatar) | Isolated recognition and word-level 3D auxiliary references. | Separate media/access licenses; none establishes continuous English→ASL 3D grammar. |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/modellicense.html), [MakeHuman](https://static.makehumancommunity.org/about/license.html) | Body/hand/face parameterization and possible deployment rig assets. | Assets are not signing observations; SMPL-X standard license is noncommercial, and exact MakeHuman asset dependencies/ASL fidelity require audit. |

Acquisition priority follows *co-observed evidence*, not maximum bytes: inquire about RIT/CUNY and STEM dialogue linguistic+kinematic files; request 3D-LEX as an isolated 3D engineering reference; obtain ASLLRP conventions and governed human annotation; use How2Sign, SignAvatars, and ASL STEM Wiki only within their current research rights. No identified source alone provides sufficiently diverse, continuous, synchronized body/hands/face/eyes, authentic linguistic timing, qualified review, and explicit commercial rights. A Deaf/ASL-partnered consented capture/adjudication program may be needed. The detailed status and correspondence priorities are in [data strategy](Sign%20Translator%20Stage%20Documentation/04_DATA_ENGINEERING_AND_CORPUS.md).

## 6. Distinctive hypothesis, not an unearned novelty claim

Prior work already has [text-to-3D sign diffusion with graph-based avatars](https://openaccess.thecvf.com/content/CVPR2024/html/Baltatzis_Neural_Sign_Actors_A_Diffusion_Model_for_3D_Sign_Language_CVPR_2024_paper.html), a [large 3D holistic sign-motion benchmark](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00653.pdf), [word-level 3D generation](https://github.com/dongludeeplearning/SignAvatar), and [multichannel evaluation](https://aclanthology.org/2024.lrec-main.1289/). Thus **3D, diffusion, graph networks, shared embeddings, and gloss-free learning are not novel merely by appearing here**. No head-to-head evaluation establishes this repository as more accurate or deployable.

| Published line of work | Established contribution | Relationship to this checkout |
|---|---|---|
| [Neural Sign Actors](https://openaccess.thecvf.com/content/CVPR2024/html/Baltatzis_Neural_Sign_Actors_A_Diffusion_Model_for_3D_Sign_Language_CVPR_2024_paper.html) | Text-conditioned diffusion on a 3D signing-avatar representation, with perceptual evaluation. | Direct prior art for 3D diffusion; our active synthetic Cartesian generator has **not** shown parity. The proposed differentiator is explicit typed linguistic/evidence gating, still unvalidated end-to-end. |
| [SignAvatars](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00653.pdf) | Holistic fitted 3D motion corpus and multi-prompt production benchmark. | Potential research data/benchmark, not a missing production license or a proof that our 2D tracks are 3D. |
| [SignAvatar](https://github.com/dongludeeplearning/SignAvatar) | Word-level 3D motion reconstruction and generation. | Useful lexical/retargeting comparator; isolated-word success does not test continuous discourse or scoped non-manual grammar. |
| [SignBLEU](https://aclanthology.org/2024.lrec-main.1289/) | Automatic multi-channel translation metric. | Relevant metric design, but no metric substitutes for qualified-ASL comprehension and error analysis. |

The project's proposed contribution is the **specific coupling** of typed temporal ASL structure, continuous multichannel motion, explicit observation-versus-inference types, action-scoped provenance/rights, weak-label abstention, and evaluation contracts that cannot average a linguistic failure into a pass. That combination is an engineering/research hypothesis. Scientific uniqueness requires a frozen comparison protocol, strong baselines, source-/signer-disjoint tests, ablations and video counterfactuals, and independent qualified Deaf/ASL comprehension review. Until then, this is a design philosophy, not a demonstrated advantage.

The mathematical ancestry is explicit rather than proprietary: [CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf) supplies unsegmented sequence marginalization; [ST-GCN](https://aaai.org/papers/12328-spatial-temporal-graph-convolutional-networks-for-skeleton-based-action-recognition/) supplies skeleton-temporal convolution; [CLIP](https://proceedings.mlr.press/v139/radford21a.html) motivates symmetric cross-modal contrastive training; [DDPM](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) supplies Gaussian diffusion; [continuous 6D rotations](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html) and [SMPL-X](https://openaccess.thecvf.com/content_CVPR_2019/html/Pavlakos_Expressive_Body_Capture_3D_Hands_Face_and_Body_From_a_CVPR_2019_paper.html) motivate the proposed 3D representation; [SoundStream](https://research.google/pubs/soundstream-an-end-to-end-neural-audio-codec/) is one origin of residual VQ. These papers establish methods, **not** the performance of this particular implementation or their suitability for ASL without evaluation.

## 7. Reproducibility and bounded execution

Python `>=3.12,<3.13` is required; [`pyproject.toml`](pyproject.toml) declares dependency ranges, while [`requirements.lock`](requirements.lock) is the lock snapshot. The CLI defaults to a long **30 joint + 175 generator-only + 16 polish epochs**. Override both secondary stages for a bounded integration run. Use an empty corpus directory; synthetic generation refuses a nonempty one unless deliberately overridden.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -c "from signtranslator.run import run_pipeline; r = run_pipeline('./corpus', epochs=1, regenerate=True, do_analyze=False); assert r['readiness'].passed"
.venv/bin/python -W error -m pytest
```

The one-epoch API smoke run intentionally skips the model's performance analysis; it checks data readiness and training integration, not translation quality. It must target a **new empty** `./corpus` directory. For the full CLI, run `python -m signtranslator.run --help` and set `--gen-finetune-epochs 0 --polish-epochs 0` explicitly for bounded work. For epoch-boundary joint-training checkpoint/resume, pass `--ckpt artifacts/run.pt` with those secondary stages at zero, then later use `--resume` with the same corpus/configuration/code identities. `--overwrite-synthetic` can overwrite a synthetic output directory and is not part of the example.

Tests cover numerical identities, finite gradients, CTC feasibility, shape errors, provenance tampering, resume integrity, source grouping, strict parsing, and readiness gates. They establish behavior relative to encoded contracts, **not absolute flawlessness**. Historical test counts in stage documents are snapshots; rerun the suite for a current verdict. No documentation edit or green synthetic test approves Stage B, Phase 2, Phase 3 empirical exit, Stage C, or production use.

## 8. Stop rules and rights

Do not train the intended 3D ASL path by forcing English into gloss or 2D into rotations. First acquire source-native annotation, synchronized canonical motion, authoritative signer/source grouping, qualified-ASL adjudication, and permission for each research/commercial action. Then freeze leakage-safe evaluation, test video dependence and linguistic minimal pairs, measure comprehension and non-manual errors with qualified reviewers, compare strong published baselines, and only afterward integrate a licensed rig, renderer, latency budget, monitoring, and rollback. See [evaluation design](docs/EVALUATION_FRAMEWORK.md) and [deployment design](docs/DEPLOYMENT.md).

Repository code is [Apache-2.0](LICENSE); that does **not** relicense datasets, human likenesses, trained derivatives, model weights, or third-party rig assets. Commercial deployment requires a separately documented and verified data/asset lineage.
