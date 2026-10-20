# Computational Sign-Language Production (SLP) with Grammar and Spatial Reference: Systems and Evaluation

Scope: how published SLP systems represent and generate spatial grammar (loci, agreement or indicating verbs, classifier or depicting constructions, role shift), non-manual markers (NMMs) and gaze, and which metrics exist for grammatical and spatial correctness. The target is an English-to-ASL text-conditioned diffusion or motion-token generator over continuous 3D motion: body, hand and head rotations, torso and head translation, eye gaze and facial blendshapes. ASL comes first; DGS, BSL and LSF work is included where it adds spatial-grammar methods. Preprints are marked [preprint]. Research date: 2026-10-20.

## Q1. Rule- and notation-based avatar systems with spatial grammar (SiGML/HamNoSys/JASigning, Paula, EMBR, ViSiCAST/eSIGN, Dicta-Sign, Huenerfauth/Lu): how loci, agreement paths, role shift and NMMs are represented, and what Deaf-user studies found

### Takeaway
The rule-based and notation-based systems are the only ones that explicitly model spatial grammar. Loci appear in three forms:
- Discrete slots: the 7 arc positions in Huenerfauth and Lu; DePaul Paula's 4 radial loci at fixed angles; the eSIGN editor's positions, rounded to what HamNoSys can express.
- Continuous 3D placeholders: Huenerfauth's classifier-predicate scene model, and the numeric SiGML locations added in Dicta-Sign.
- A signer-anchored coordinate frame: LIMSI's pelvis-anchored frame.

Agreement verbs are treated as a lexical motion template parameterized by subject and object loci. Role shift is a procedural rotation layered over the spine, neck and eyes, with gaze leading. Deaf-user studies consistently found that adding spatial reference, verb inflection and grammatical facial expressions significantly improves comprehension. Absolute comprehension of avatars is still often low; one 2025 DGS avatar study found less than 50% of sentence units understood.

### Cited Findings

**Huenerfauth and Lu: spatial reference points and inflecting verbs (ASL, CUNY/RIT)**
- Linguistic framing used by the system: after an entity is mentioned, the signer may point to a 3D location around the body and later point back to it. Some linguists place loci on a semicircular arc at chest height; others place them at varying heights and distances. Lu and Huenerfauth note there is "an infinite number of locations", so a finite fixed lexicon is insufficient for inflecting verbs. — [Lu & Huenerfauth 2012, SLPAT](https://aclanthology.org/W12-2909.pdf)
- Locus representation: 7 discrete positions on an arc around the signer (front and top views in their Fig. 2). Training data covered every non-reflexive subject/object combination: 42 examples for subject-and-object verbs (GIVE, MEET) and 7 for object-only verbs (TELL, SCOLD, ASK). Five verbs were modeled in total. — [Lu & Huenerfauth 2012](https://aclanthology.org/W12-2909.pdf)
- "Point" model, from prior work: keyframe hand positions were extracted, giving 504 values for GIVE (42 examples × 2 keyframes × 2 hands × xyz). A third-order polynomial was fit per dimension and keyframe, parameterized on the subject and object arc locations. Synthesis then predicts the keyframe hand targets and uses inverse kinematics plus interpolation. This allows new combinations not seen in training. — [Lu & Huenerfauth 2012](https://aclanthology.org/W12-2909.pdf)
- "Vector" model (2012): models only the hand movement vector (direction and magnitude) as a function of the subject and object loci. Starting locations come from a 3D Gaussian mixture model of hand location per verb, with 6 components, searched over a grid. Data came from 3 signers, cross-validated on 42 verb instances from the held-out signers. The error is the mean Euclidean start/end distance for both hands, in units where shoulder width = 1.0. The vector model had lower error than the point model; inter-signer variation was plotted as the baseline. — [Lu & Huenerfauth 2012](https://aclanthology.org/W12-2909.pdf)
- Deaf-user results:
  - In prior studies, short ASL stories using model-synthesized verbs showed no significant difference in comprehension or subjective scores from verbs made by a human animator. — [Lu & Huenerfauth 2012](https://aclanthology.org/W12-2909.pdf)
  - In a study with 12 native signers, model-generated animations were rated similar in quality to a human animator's. — [SLTAT 2011 abstract](https://vhg.cmp.uea.ac.uk/demo/SLTAT2011Dundee/10.html)
  - Native signers rated animations with spatially inflected verbs and spatially associated entities as easier to understand than animations without them (Huenerfauth & Lu 2012). — [Lu & Huenerfauth 2012](https://aclanthology.org/W12-2909.pdf)
- Lu's dissertation: the model-generated verbs were comparable in understandability to animations made by a human animator or driven by motion capture, using both subjective ratings and comprehension questions with Deaf native signers. — [Lu 2014, CUNY dissertation](https://academicworks.cuny.edu/gc_etds/1418)
- A secondary summary reports that in a study with 18 native signers, model-synthesized inflected verbs scored significantly higher on comprehension than uninflected (citation-form) verbs and matched human-animated inflected verbs. Data-driven synthesis of new verb instances is also reported in the SLPAT 2015 paper (Huenerfauth, Lu, Kacorri). — [SLPAT 2015 W15-5105](https://preview.aclanthology.org/fix_video/W15-5105.pdf). Caveat: this came from a search snippet; the PDF returned 404 when fetched, so the 18-participant figure was not checked in the primary text.
- CUNY ASL motion-capture corpus, first release: 3 signers, 98 stories; access requires contacting the lab. — [RIT LATLab corpus page](https://latlab.ist.rit.edu/corpus/). The elicitation protocol was designed to draw out pronominal spatial reference while minimizing classifier predicates. — [Huenerfauth & Lu, LREC 2010](https://www.sign-lang.uni-hamburg.de/lrec/pub/10030.html)

**Huenerfauth: classifier-predicate generator (ASL, 2006–2008)**
- Input is an explicit 3D scene model of "placeholders" for discourse entities, each with a center-of-mass location (x, y, z), an orientation (rotation angles), and bits for the allowed classifier handshapes. The model is mapped onto a volume in front of the signer's torso. In a full system, the scene would come from VR scene-visualization software. — [Huenerfauth, Zhao, Gu & Allbeck 2007](https://aclanthology.org/W07-1907.pdf)
- Planning operators coordinate handshape, hand location, eye gaze, head tilt and brow raise. Gaze is computed from the 3D model locations and supports gaze at a locus, smooth pursuit of a moving locus, and blinking. The head tilts when a gaze shift exceeds a threshold, and blink rate drops while gaze tracks the hands. Example: HOUSE with raised brows and gaze to the audience, then a spread-C classifier with gaze to the house locus. — [Huenerfauth et al. 2007](https://aclanthology.org/W07-1907.pdf)
- Evaluation: 15 native ASL signers, screened for ASL since birth, Deaf parents, or residential school (8 met all three criteria). Ratings on 1–10 scales for grammaticality, understandability and naturalness, plus a matching task in which participants picked which 3D animation of objects matched the sentence. The system beat a Signed English baseline on all four measures (α = 0.05, pairwise Mann-Whitney U with Bonferroni correction). — [Huenerfauth et al. 2007](https://aclanthology.org/W07-1907.pdf)
- Qualitative feedback from those 15 participants: 8 wanted more facial expression (4 named nose and mouth), 2 wanted brows raised higher, and 3 wanted more eye-gaze movement. One suggested the non-dominant hand should hold the house locus during the car's movement. — [Huenerfauth et al. 2007](https://aclanthology.org/W07-1907.pdf)
- The journal version reports that classifier-predicate animations conveyed information content more effectively to native signers. — [Huenerfauth et al. 2008, ACM TACCESS (SIGACCESS summary)](https://www.sigaccess.org/?p=317)

**DePaul Paula (Wolfe, McDonald et al.)**
- Loci for indexing and verb conjugation: "four key referential points" spaced radially around the avatar, at about 15° and 30° on the strong side and 30° and 45° on the weak side of the midsagittal plane (Toro 2004). The weak side gets extra angle to compensate for the torso twist when reaching across the body. If protagonists are not explicitly indexed, the system infers them from speaking order and assigns contrastive strong- and weak-side positions. — [McDonald, Wolfe, Moncrief & Baowidan, LREC 2014 workshop](https://www.sign-lang.uni-hamburg.de/lrec/pub/14019.pdf)
- Role-shift synthesis: procedurally generated keys are layered over sparse keyframe sign data. This transforms signs from the fixed referential framework to the shifted one. The authors state it can layer over motion-capture base animation too, given separate spine, neck and eye controls. — [McDonald et al. 2014](https://www.sign-lang.uni-hamburg.de/lrec/pub/14019.pdf)
  - The avatar does not need to face the protagonist's locus exactly; a contrastive shift toward it is enough.
  - Joints move in a cascade: eyes, then neck, then pelvis, then lumbar and thoracic spine, then sternoclavicular joints.
  - Eyes turn to the addressee's locus up to half a second before the body rotates.
  - Because referents are indicated by direction rather than position, the gaze yaw is computed as the angle between the addressee locus and the midsagittal plane, using the dominant-shoulder position and ignoring height.
- NMM inventory for role shift used by DePaul (after Quer 2005): a slight body shift toward the protagonist's locus, a gaze change to the purported addressee, a head-orientation change, and the protagonist's facial expression. In DGS only the gaze change is obligatory; body and head shifts are optional (Herrmann & Steinbach 2012). — [McDonald et al. 2014](https://www.sign-lang.uni-hamburg.de/lrec/pub/14019.pdf)
- Annotation schemes relevant to role shift, as surveyed in the same paper: ECHO and Auslan guidelines have a role tier with start and end times plus the assumed character. Cormier & Smith (BSL, 2011) use 8 tiers: gaze, head, face, torso, dominant arm, non-dominant arm, Role1 and Role2. ECHO gaze codes are discrete directions (r-90, r, l-90, l, u, d, a, s, p). — [McDonald et al. 2014](https://www.sign-lang.uni-hamburg.de/lrec/pub/14019.pdf)
- Other systems in the same survey: LIMSI (Braffort & Dalle 2008) placed referent loci in avatar-centered coordinates anchored on the pelvis so role shifts can rotate the pelvis. SignCom (LSF; Duarte & Gibet) builds role-shift transitions between motion-capture segments. — [McDonald et al. 2014](https://www.sign-lang.uni-hamburg.de/lrec/pub/14019.pdf)
- Recent Paula work:
  - A two-parameter representation for emotional NMMs, controlled through the EASIER notation, intended to specify emotional facial expression more coherently than earlier methods. — [McDonald, Wolfe & Nunnari 2025, arXiv 2508.07937 [preprint]](https://arxiv.org/abs/2508.07937v1)
  - NMMs are still the hardest parameter for avatars to display legibly; DePaul built facial shading for legibility at fast render times. — [Johnson & Wolfe, LREC 2020](https://www.sign-lang.uni-hamburg.de/lrec/pub/20019.html)
  - Related papers include adverbs of manner (Moncrief 2020) and complex shape deployments (Filhol & McDonald 2020). — [sign-lang@LREC Paula tool page](https://www.sign-lang.uni-hamburg.de/lrec/tool/paula.html)
  - DePaul states Paula can portray all linguistic parameters of ASL. — [DePaul ASL Avatar Project](https://asl.cs.depaul.edu/project_info.html)

**HamNoSys / SiGML / JASigning, ViSiCAST/eSIGN, Dicta-Sign (UEA and Hamburg; BSL, DGS, NGT and others)**
- HamNoSys is a phonetic notation for handshape, orientation, location and movement. It drives avatars through SiGML, its XML form. JASigning's Animgen module turns an avatar description plus a SiGML script into frame-by-frame animation. — [UEA Virtual Humans Group](https://vh.cmp.uea.ac.uk/index.php/Virtual_Humans_Group); [Marshall & Sáfár 2003](https://www.cs.jhu.edu/~yarowsky/acl03cd/posterdemo/pdfs/Marshall.pdf)
- Directional verbs: their start and end positions carry syntactic and semantic information. — [Marshall & Sáfár 2003](https://www.cs.jhu.edu/~yarowsky/acl03cd/posterdemo/pdfs/Marshall.pdf). The eSIGN editor had users set source and goal positions by clicking, and each click was rounded to the nearest position HamNoSys can express. Loci were therefore discrete notation slots. — [eSIGN annual report 2003, editor: source/goal](https://www.sign-lang.uni-hamburg.de/esign/annualreport2003/editor2003/srcgol.html)
- Dicta-Sign extended SiGML with continuous attribute values in place of HamNoSys's discrete ones, so signing-space locations can be given numerically. It also added explicit timing to support comparing avatar signing with human video. — [Dicta-Sign Deliverable D3.3](https://cordis.europa.eu/docs/projects/cnect/5/231135/080/deliverables/001-DICTASIGNDeliverableD33Ares2012696081.pdf); [Elliott et al., "Extending the SiGML Notation", SLTAT 2011](https://vhg.cmp.uea.ac.uk/demo/SLTAT2011Dundee/8.pdf)
- Practitioner view: SiGML allows more precision than HamNoSys, but authoring signs in it is "complicated, hard and time-consuming". — search-result summary of an applied paper ([IJEECS](https://ijeecs.iaescore.com/index.php/IJEECS/article/download/22518/14493)); not checked in full text.

**EMBR / EMBRScript (Heloir & Kipp, DFKI; DGS)**
- Kipp, Heloir & Nguyen (IVA 2011) extended the EMBR character-animation system with a gloss-based animation tool built with two deaf experts. They introduced "delta testing", which compares avatar comprehension against human-signer video, and claimed state-of-the-art comprehensibility. — [DFKI publication page](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/5469); [Heloir, Nguyen & Kipp, SLTAT 2011](https://vhg.cmp.uea.ac.uk/demo/SLTAT2011Dundee/5.html)
- EMBR was later used as the rendering back end for Kacorri and Huenerfauth's ASL facial-expression work. Facial expressions were an EMBR PoseSequence with one pose every 133 ms, driven from MPEG-4 features. — [Kacorri & Huenerfauth 2016, ACL](https://terpconnect.umd.edu/~hernisa/pub/2016-acl-hkacorri-huenerfauth.pdf)
- 2025 DFKI DGS avatar on HoloLens 2 (AVASAG project, 6 public-transport sentences). 8 of 9 participants completed it; 7 were Deaf. On average fewer than 50% of sentence units were understood, and adjustability of avatar size and position made no difference. Missing mouthings caused confusions ("Hannover" read as BLUE, "13" read as "3"). Participants disliked the static facial expression. — [Wasserroth et al., IVA 2025, arXiv 2508.05358](https://arxiv.org/html/2508.05358v1)

### Inferences
- For a continuous 3D generator, the clearest lessons are as follows. Huenerfauth and Lu show that agreement verbs are well modeled as a citation form plus a low-dimensional, locus-conditioned displacement (start and end vectors). That displacement generalizes across signers better when expressed as a movement vector than as absolute endpoints. DePaul shows that role shift can be a residual layer on spine, neck and eye rotations, keyed to the protagonist and addressee loci, with gaze leading the body by up to 0.5 s. Both map naturally onto conditioning a diffusion model on locus embeddings, as continuous yaw angles or 3D points in a chest-centered frame, plus a role-shift flag or target.
- Discrete locus slots (7 arc positions, 4 DePaul angles, HamNoSys positions) are a practical label set for building conditioning labels and automatic checks, even if generation is continuous. Converting them to angles relative to the midsagittal plane, as DePaul does, gives a signer-scale-invariant representation.
- Deaf-user results show that the biggest comprehension gains come from adding spatial reference or verb inflection, and grammatical NMMs, rather than from smoother motion. This supports making locus and NMM conditioning first-class rather than emergent.

### Gaps
- Exact numeric means for grammaticality, understandability, naturalness and matching-task scores in Huenerfauth et al. 2007 are only in a bar chart and could not be extracted. The 2008 TACCESS numbers and the Huenerfauth & Lu UAHCI 2011 and UAIS 2012 percentages for the effect of spatial reference were not retrieved.
- Kipp et al. 2011's EMBR comprehension percentages were not found. The EMBRScript specification of NMMs (gaze and face keyframes) was not checked in primary text.
- No primary source was found for ViSiCAST/eSIGN user-study comprehension numbers, or for how JASigning handles gaze toward loci or role shift.
- The DePaul brow-modeling paper (Wolfe et al. 2011, *Sign Language & Linguistics* 14:1), on layering linguistic and affective brow motion, could not be fetched (publisher server refused connection), so its method and results are not reported here.

## Q2. Data-driven SLP (Progressive Transformers, adversarial/MDN, Text2Sign, SignDiff, SignLLM/Prompt2Sign, MS2SL, SignAvatars, Neural Sign Actors, T2S-GPT, G2P-DDM, SOKE): do any model spatial agreement, loci, role shift or NMM grammar, and what representations and conditioning do they use?

### Takeaway
None of the data-driven systems examined explicitly model loci, agreement, role shift or NMM grammar. They condition on sentence text (CLIP or mBART) or gloss sequences, generate 2D/3D keypoints or SMPL-X parameters (sometimes via VQ motion tokens), and treat facial expression as just more output dimensions. The one partial exception is Zhang et al. (CHI 2025): an LLM predicts sentence-type NMM labels (yes/no question, wh-question, conditional, negation) that adjust eyebrows in a retrieval-and-blend pipeline. It still has no locus model.

### Cited Findings
- **Progressive Transformers** (Saunders, Camgoz & Bowden, ECCV 2020): gloss-to-pose and text-to-pose with a counter-based progressive decoder. It introduced back-translation evaluation using a pretrained SLT model on PHOENIX14T. — [arXiv 2004.14874](https://arxiv.org/pdf/2004.14874). Reported BLEU-4 on the PHOENIX14T test set is 10.43 (gloss-to-pose) and 10.51 (text-to-pose); dev is 11.93 and 11.82. These figures come from tables in later papers, not the original. — [Mixed SIGNals, arXiv 2107.11317](https://arxiv.org/pdf/2107.11317)
- **Adversarial multi-channel SLP** (Saunders et al. 2020): a transformer generator with a text-conditioned discriminator. It explicitly includes non-manual channels (facial features and mouthing) alongside manual features, but treats them as appearance channels, not grammatical markers. — [arXiv 2008.12405 [preprint page]](https://arxiv.org/abs/2008.12405)
- **G2P-DDM** (Xie et al., AAAI 2024): Pose-VQVAE discretizes pose, and a discrete denoising diffusion model with a CodeUNet generates pose tokens from gloss, using clustering-based length prediction. — [arXiv 2208.09141](https://arxiv.org/abs/2208.09141). A 2026 retrieval-and-refine method (SignRR) reports 10.96 test BLEU-4 on PHOENIX14T and claims to beat G2P-DDM on all four metrics. — [SignRR, arXiv 2608.28568 [preprint]](https://arxiv.org/pdf/2608.28568)
- **T2S-GPT** (Yin et al., ACL 2024): DVA-VAE dynamic-length vector quantization, where code length follows information density. A GPT-style model generates codes and durations from text. It was evaluated on PHOENIX14T, and the authors released PHOENIX-News (486 hours). — [ACL 2024.acl-long.183](https://preview.aclanthology.org/setup/2024.acl-long.183); [arXiv 2406.07119](https://arxiv.org/html/2406.07119v1)
- **Neural Sign Actors** (Baltatzis et al., arXiv 2312.02702; venue not confirmed in the fetched page) — [arXiv 2312.02702](https://arxiv.org/html/2312.02702)
  - Model: diffusion over SMPL-X pose and expression parameters on a canonical shape. The denoiser is an anatomically inspired GNN over the kinematic tree; expression goes through an MLP with learnable tokens. Text is conditioned through a CLIP ViT-L/14 sentence embedding with gating, and the authors explicitly use no gloss or intermediate representation. How2Sign was annotated with SMPL-X.
  - How2Sign test results: body MPJPE 35.87 mm, BLEU-4 13.12, ROUGE 47.55, body DTW 7.83, FID 1.56. The best baseline (Stoll et al.) scored BLEU-4 7.51 and ROUGE 33.17.
  - User study with 15 ASL-fluent raters: text-alignment scores of 8.7 for ground truth, 8.1 for fittings and 5.8 for NSA. Top-1 translation-pick accuracy was 40%, top-2 was 80%.
  - Ablation: replacing CLIP with learned word embeddings raised body MPVPE from 31.47 to 69.12 mm.
  - No discussion of loci, agreement or role shift, and no NMM-specific evaluation.
- **SOKE / Signs as Tokens** (Zuo et al., ICCV 2025) — [arXiv 2411.17799v3](https://arxiv.org/html/2411.17799v3); [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zuo_Signs_as_Tokens_A_Retrieval-Enhanced_Multilingual_Sign_Language_Generator_ICCV_2025_paper.html)
  - Representation: a decoupled tokenizer with three VQ-VAEs (upper body, left hand, right hand; codebooks of 96, 192 and 192). The 133-D SMPL-X vector includes 10 expression parameters, but expression is not tokenized separately.
  - Generation: mBART-large-cc25 with multi-head decoding (one head per body part) and language tokens <ASL>, <CSL> and <DGS>. Dictionary signs are retrieved and their motion tokens appended to the prompt, which cut DTW error by about 20–24%.
  - How2Sign results: DTW-PA-JPE body/hand 6.82/2.35 and BLEU-4 14.48, against 11.23/4.39 and 11.45 for S-MotionGPT.
  - User study: 5 ASL and 4 CSL professional signers rated text-sign alignment at 5.32 (How2Sign) and 6.45 (CSL-Daily).
  - No NMM or spatial-grammar modeling.
- **SignAvatars** (Yu et al., ECCV 2024): about 70K videos, 153 signers and 8.34M frames, with automated 3D holistic body, hand and face annotations. Conditioning includes HamNoSys, spoken-language text and words. It is a benchmark for 3D holistic production; NMMs and spatial grammar are not discussed. — [arXiv 2310.20436](https://arxiv.org/abs/2310.20436)
- **SignLLM / Prompt2Sign** (Fang et al.): an 8-language pose dataset in a unified compressed format. Two modes are offered (MLSF from text; Prompt2LangGloss from prompts), with an RL loss and a "Priority Learning Channel". The abstract claims state of the art across 8 sign languages but gives no grammar modeling. — [arXiv 2405.10718 [preprint, v3 Apr 2025]](https://arxiv.org/abs/2405.10718)
- **Zhang et al., "Towards AI-driven Sign Language Generation with Non-manual Markers"** (CHI 2025; Apple, UW, Gallaudet, Tel Aviv) — [arXiv 2502.05661](https://arxiv.org/html/2502.05661v1)
  - Pipeline: GPT-4o few-shot English→gloss, constrained to a dictionary, plus zero-shot NMM labels (yes/no question, wh-question, conditional, negation). Motion matching over 12,681 pose clips favors "economy of motion" transitions, followed by expression blending (mainly eyebrows) and a pose-to-video U-Net.
  - Results: gloss BLEU-4 0.276 (previous 0.191); NMM detection precision 0.91 and recall 0.97; FVD 173.78 against a 2024.42 baseline.
  - 30 DHH participants rated signing below real signers but often understood the content.
- **Diverse Signer Avatars** (Lakhal & Bowden 2025): a latent diffusion avatar renderer with a sign-feature aggregation module that models face (non-manual) and hand features separately. It targets appearance and emotion, not grammatical NMMs. — [arXiv 2508.15988 [preprint]](https://arxiv.org/abs/2508.15988)

### Inferences
- The best-reported text-to-3D systems (NSA, SOKE) reach only about 13–15 back-translation BLEU-4 on How2Sign, with text-alignment ratings from Deaf raters around 5–6 out of 10. The back-translators differ, so the BLEU numbers are not strictly comparable across papers. Grammatical phenomena are likely to be among the first things missing at that level, though this is untested.
- The SOKE design (per-part tokens plus retrieved dictionary-sign tokens) and the NSA design (anatomical GNN plus CLIP text) give natural slots to add (a) a head, gaze and face token stream or channel, and (b) explicit conditioning tokens for locus assignment, verb agreement and NMM spans. None of these systems has done either.
- The Zhang et al. CHI 2025 approach (an LLM predicting sentence-type NMM labels with high precision and recall, then applying parametric brow curves) is a cheap, verifiable way to produce NMM-span conditioning labels from English. It could produce labels for a learned generator.

### Gaps
- Text2Sign (Stoll et al.), SignDiff and MS2SL were not examined in primary text; their representations and metrics are not reported here. MS2SL (arXiv 2407.12842) was only located.
- Exact G2P-DDM and T2S-GPT metric values were not retrieved.
- No data-driven 2024–2026 system was found that models loci, agreement or role shift. That absence is from my searches, not a confirmed survey result.

## Q3. Non-manual marker generation: grammatical facial expressions and head movement (DePaul brow and headshake work, Kacorri & Huenerfauth)

### Takeaway
Kacorri and Huenerfauth have the most rigorous data-driven pipeline for grammatical NMMs:
- 5 syntactic types (yes/no question, wh-question, rhetorical question, topic, negation) represented as MPEG-4 facial action parameters (FAPs) plus head pose, extracted from video.
- Synthesis by picking a DTW "centroid" recording, later improved with Continuous Profile Models (CPMs), aligned to the clause span.
- Evaluation with Deaf participants against a static-face baseline, using released stimuli.

DePaul treats NMMs procedurally and layered (brows, emotion parameters, role-shift spine and gaze). In both groups, Deaf participants repeatedly asked for more and stronger facial expression and gaze.

### Cited Findings
- Grammatical facial expression definitions used for synthesis: — [Kacorri & Huenerfauth 2016, ACL](https://terpconnect.umd.edu/~hernisa/pub/2016-acl-hkacorri-huenerfauth.pdf)
  - YesNo: brows raised, head tilted forward.
  - WhQuestion: brows furrowed, head tilted forward.
  - Rhetorical: brows raised, head tilted back and to the side.
  - Topic: brows raised, head back, on a clause-initial phrase.
  - Negative: headshake during the verb phrase, often with NOT.
- Context-dependent subgroups: each type is split in two. YesNo is split by whether a brow-raise immediately precedes it (9 vs 10 videos). Wh and rhetorical are split by single-word versus phrase scope (4/8 and 2/8). Topic is split the same way (29/15). Negative is split by preceding brow-raise (16/25). Feature dimension is 14. — [Kacorri & Huenerfauth 2016](https://terpconnect.umd.edu/~hernisa/pub/2016-acl-hkacorri-huenerfauth.pdf)
- Features: Visage Face Tracker MPEG-4 FAPs. These are head orientation (FAP47–49), vertical eyebrow displacements (FAP30–35, 6 parameters) and horizontal eyebrow displacements (FAP36–37), plus head position. The authors chose MPEG-4 because FAPs are normalized by facial scaling factors and so transfer between a human face and an avatar. — [Kacorri & Huenerfauth 2015, SLPAT](https://aclanthology.org/W15-5106.pdf)
- Synthesis methods:
  - "Centroid": a multivariate DTW picks the training recording with minimal pairwise normalized DTW distance to the others.
  - CPM: a latent trace at about twice the temporal resolution, learned from multiple recordings, which removes idiosyncratic variation.
  - Both are applied only within the NMM's syntactic span, resampled with cubic interpolation to the animation's duration, with face and head neutral elsewhere. Rendering is through EMBR.
  - — [Kacorri & Huenerfauth 2016](https://terpconnect.umd.edu/~hernisa/pub/2016-acl-hkacorri-huenerfauth.pdf)
- CPM evaluation:
  - Metric evaluation: the CPM latent trace was closer to a gold-standard male signer than the centroid in an inter-signer DTW evaluation; the training recordings were from a female signer.
  - User study: 17 participants (14 Deaf, 3 hard of hearing), between-subjects, with three conditions: neutral face, centroid, and CPM. Ratings used 1–10 scales for grammar, understandability and naturalness, averaged. Participants rated CPM significantly higher (Kruskal-Wallis), even though the differences between conditions were subtle.
  - — [Kacorri & Huenerfauth 2016](https://terpconnect.umd.edu/~hernisa/pub/2016-acl-hkacorri-huenerfauth.pdf)
- Methodology contributions:
  - Released stimuli with comprehension and subjective questions for evaluating ASL facial expressions. — [Huenerfauth & Kacorri, LREC 2014 workshop](https://www.sign-lang.uni-hamburg.de/lrec/pub/14010.html)
  - Comprehension questions designed to test whether viewers perceived and understood linguistic facial expressions. — [Kacorri, Lu & Huenerfauth, UAHCI 2013](https://terpconnect.umd.edu/~hernisa/pub/2013-uahci-kacorri-lu-huenerfauth.pdf)
  - Standardized screening and technology-attitude surveys (MediaSharing, AnimationAttitude) to report with every study; in the CPM study the means were 4.3 and 3.8. — [Kacorri & Huenerfauth 2016](https://terpconnect.umd.edu/~hernisa/pub/2016-acl-hkacorri-huenerfauth.pdf)
  - Eye tracking: native signers' proportional fixation time on the upper versus lower face varied significantly with animation quality. This makes it a candidate implicit evaluation measure. — [Kacorri, Harper & Huenerfauth, UAHCI 2014](https://terpconnect.umd.edu/~hernisa/pub/2014-uahci-kacorri-harper-huenerfauth.pdf)
  - Kacorri's dissertation studies how experiment design affects outcomes when evaluating sign-language animation with facial expressions. — [Kacorri 2016, CUNY dissertation](https://academicworks.cuny.edu/gc_etds/1375)
- An earlier comprehension finding: including grammatical facial expressions in ASL animation significantly improved user comprehension. — [search summary of Huenerfauth group work, SIGACCESS](https://www.sigaccess.org/?p=423). Not checked in full text.
- DePaul: role-shift NMMs are synthesized procedurally (spine, neck and eye cascade; see Q1). Emotional NMMs use a two-parameter representation via EASIER. Face shading targets NMM legibility. — [McDonald et al. 2014](https://www.sign-lang.uni-hamburg.de/lrec/pub/14019.pdf); [arXiv 2508.07937 [preprint]](https://arxiv.org/abs/2508.07937v1); [Johnson & Wolfe 2020](https://www.sign-lang.uni-hamburg.de/lrec/pub/20019.html)

### Inferences
- For a blendshape-based generator, the Kacorri feature set (brow vertical and horizontal, head pitch, yaw and roll) is a minimal grammatical-NMM subspace that is useful for targeted losses and metrics.
- The split by context (preceded by brow-raise; single-word versus phrase scope) shows that NMM shape depends on neighboring NMMs and span length. A generator should be conditioned on NMM spans with scope, not just sentence-level labels.
- Averaging across signers (the CPM latent trace) beat using a single real exemplar. This suggests that reducing idiosyncratic variation helps grammatical legibility, which matters for diffusion sampling-temperature and guidance choices.

### Gaps
- The DePaul brow paper (Wolfe et al. 2011) and any DePaul headshake model could not be retrieved; no numbers are available.
- Exact mean ratings per condition in the CPM study, and per-type comprehension effects, were not extracted beyond significance.

## Q4. Evaluation metrics: back-translation, DTW-MJE, FID-like scores, Deaf-user studies, and automatic metrics for spatial consistency, agreement or NMM correctness

### Takeaway
Standard SLP evaluation is back-translation (BLEU, ROUGE, chrF, BLEURT, likelihood), keypoint or joint DTW errors, FID/FVD, and small Deaf-rater studies (typically 1–10 alignment or grammar, understandability and naturalness ratings, plus comprehension questions). The only validated NMM-specific automatic metric found is Kacorri & Huenerfauth's phase-segmented DTW over MPEG-4 brow and head features, with Spearman ρ of −0.38 against comprehension and −0.33 against noticing the expression. No validated automatic metric for locus consistency or agreement correctness was found. The nearest are Lu & Huenerfauth's endpoint-distance error against signer data, the 2026 Surrey indexing-expert framework (recognition-side), and BackTranslation2.0's "grammatical correctness" dimension (BSL; details unverified).

### Cited Findings
- **Phase-segmented DTW scoring for facial expressions** — [Kacorri & Huenerfauth 2015, SLPAT](https://aclanthology.org/W15-5106.pdf)
  - The gold and candidate MPEG-4 FAP streams are split into 4 phases by 5 milestones tied to sign timing (for example, Negation M3 is the start of the second word).
  - DTW distance is normalized per phase and combined.
  - Study: 18 participants (15 learned ASL before age 9); stimuli contained Negation, wh-question and yes/no-question NMMs.
  - Correlations: Spearman ρ = −0.38 (p < 0.001) with comprehension-question scores, and ρ = −0.33 (p < 0.001) with "noticed the facial expression" scores. The authors call these moderate and suggest LCSS as a variant.
- **Meta-evaluation of pose-based SLP metrics** (Jiang et al. 2025) — [arXiv 2510.07453 [preprint]](https://arxiv.org/html/2510.07453v1)
  - Setup: 13 native deaf raters (DSGS, LSF, LIS); 11,471 ratings over 2,650 examples. Inter-annotator κ = 0.36 and intra-annotator κ = 0.49.
  - Segment-level Spearman correlations (approximate, read from an extracted table): tuned DTWpp ≈ 0.76, close to the human inter-rater ≈ 0.77; chrF ≈ 0.42; back-translation likelihood ≈ 0.42; BLEURT ≈ 0.36; BLEU-4 ≈ 0.21.
  - Implementation details of distance metrics matter a great deal. The authors call back-translation likelihood "the most consistent metric" and call for open, standardized pose-to-text models.
  - No spatial-consistency metric. Non-manuals are mentioned only as a possible keypoint selection.
- Back-translation depends on an unstandardized sign-to-text model, and DTW mean joint error can penalize valid linguistic variation (which motivates SiLVERScore's semantic embedding). — [SiLVERScore, arXiv 2509.03791 [preprint]](https://arxiv.org/pdf/2509.03791)
- **BackTranslation2.0** (Cory, Ivashechkin, …, Bowden; ECCV 2026 per arXiv comments): an agentic pipeline of specialized tools scoring grammatical correctness, phonological accuracy, motion fluency and generation fidelity, with LLM comparison modules and deterministic weighted formulas. Validated on BSL with a human-rater study against 6 baselines, reporting "strong correlation with human judgements across all dimensions". — [arXiv 2606.28673 [preprint]](https://arxiv.org/abs/2606.28673)
- **SLRTP 2025 challenge** (CVPR workshop, PHOENIX14T text-to-pose): metrics were BLEU-1 back-translation and DTW-MJE. 33 participants submitted 231 solutions. The winner (retrieval plus a pretrained LM) scored BLEU-1 31.40 and DTW-MJE 0.0574. The organizers released the evaluation network as a consistent baseline and noted that missing standard metrics hamper comparison. — [arXiv 2508.06951](https://arxiv.org/abs/2508.06951)
- Typical 3D SLP metric bundle: MPVPE and MPJPE per body part, FID, DTW, BLEU-1–4, ROUGE, plus a small Deaf-rater alignment study and translation-pick accuracy. — [Neural Sign Actors](https://arxiv.org/html/2312.02702). Variants: DTW-PA-JPE and DTW-JPE by part, plus BLEU-4. — [SOKE](https://arxiv.org/html/2411.17799v3). FVD for video. — [Zhang et al. CHI 2025](https://arxiv.org/html/2502.05661v1)
- **Huenerfauth's Deaf-user methodology** — [Huenerfauth et al. 2007](https://aclanthology.org/W07-1907.pdf); [Kacorri & Huenerfauth 2016](https://terpconnect.umd.edu/~hernisa/pub/2016-acl-hkacorri-huenerfauth.pdf); [Kacorri & Huenerfauth 2015](https://aclanthology.org/W15-5106.pdf)
  - Screen for native or early signers (ASL at home, Deaf parents, residential school).
  - Give instructions in ASL with a native signer present.
  - Use 1–10 scales for grammar ("perfect ASL grammar" versus "mixed-up/English-like"), understandability and naturalness ("like a robot" versus "like a real person").
  - Ask comprehension questions about story content, a "notice" question for specific NMMs, and matching tasks for spatial or classifier content.
  - Compare against lower baselines (neutral face, Signed English, uninflected verbs) and upper baselines (human animator, motion-capture driven).
- **Spatial or agreement evaluation proxies:**
  - Lu & Huenerfauth: mean Euclidean error of predicted start and end hand positions against held-out signers' inflected-verb data, in shoulder-width units, compared with inter-signer variation. — [Lu & Huenerfauth 2012](https://aclanthology.org/W12-2909.pdf)
  - The classifier "matching task", in which participants pick which 3D object animation the sentence describes. — [Huenerfauth et al. 2007](https://aclanthology.org/W07-1907.pdf)
  - DePaul proposes the linguistic inventory of role-shift visual cues as "a metric against which we can evaluate" synthesis. — [McDonald et al. 2014](https://www.sign-lang.uni-hamburg.de/lrec/pub/14019.pdf)
- **Indexing:** current sign recognition models recover pointing-based indexing poorly, even though it makes up about 10–15% of signing content. Ranum, Hadfield & Bowden propose indexing experts (index detection plus discourse-entity linking) as a baseline for index-aware modeling. The abstract does not propose a generation metric. — [arXiv 2606.08056 [preprint, June 2026]](https://arxiv.org/abs/2606.08056)

### Inferences
- A practical automatic grammar and space test suite for an ASL diffusion model could combine five existing pieces:
  - Kacorri-style phase-aligned DTW on brow, head and blendshape channels within annotated NMM spans.
  - Lu-style endpoint and vector error for agreement verbs relative to the conditioned subject and object loci, in shoulder-width units, with inter-signer variation as the floor.
  - Pointing-direction and gaze-direction angular error against the assigned locus (DePaul's midsagittal-angle formulation), plus a consistency check that re-indexing the same referent hits the same angle.
  - Back-translation likelihood with an open pose-to-text model, as Jiang et al. recommend.
  - A small Deaf-rater study with comprehension and notice questions.
  These would need validation against human judgments, which none of them have for spatial features.
- Given human inter-rater κ of only about 0.36, any spatial or NMM metric should be validated at the segment level with multiple raters. System-level claims need several systems.

### Gaps
- BackTranslation2.0's grammatical-correctness sub-metric and its correlation numbers were not available on the abstract page.
- No source was found for an automatic metric that scores locus reuse or coreference consistency, agreement-verb direction, or role-shift presence in generated motion, beyond the proxies above.
- The Huenerfauth & Kacorri CSUN 2015 "best practices" paper was located only as a reference in [Kacorri & Huenerfauth 2015](https://aclanthology.org/W15-5106.pdf), not read.

## Q5. The 2024–2026 state of the art and open problems in grammar-aware SLP

### Takeaway
From 2024 to 2026, SLP moved to large 3D SMPL-X datasets (How2Sign SMPL-X fits, SignAvatars, CSL-Daily 3D), diffusion and LLM token generators (NSA, SOKE, T2S-GPT, G2P-DDM, SignLLM), retrieval-augmented decoding, and better evaluation (DTWpp, back-translation likelihood, BackTranslation2.0). Grammar-aware generation of spatial reference, agreement and role shift is still essentially absent from neural SLP. The open problems the field itself names are non-lexical and productive constructions such as indexing, NMM legibility, metric validity, and low absolute comprehension.

### Cited Findings
- SOTA text-to-3D on How2Sign: SOKE BLEU-4 14.48 (ICCV 2025) and NSA BLEU-4 13.12 (arXiv 2312.02702), each with its own back-translator. Deaf-rater text alignment is 5.32/10 for SOKE and 5.8/10 for NSA, against 8.7 for ground truth in the NSA study. — [SOKE](https://arxiv.org/html/2411.17799v3); [NSA](https://arxiv.org/html/2312.02702)
- Models trained on glosses or text "under-model non-lexical, productive constructions". Spatial indexing is about 10–15% of signing and is poorly recovered by SLR models. — [Ranum, Hadfield & Bowden 2026 [preprint]](https://arxiv.org/abs/2606.08056)
- Agreement between automatic metrics and human judgments had not been formally validated before 2025 meta-evaluation work; inter-rater agreement is low (κ ≈ 0.36). — [Jiang et al. 2025 [preprint]](https://arxiv.org/html/2510.07453v1)
- Linguistically decomposed metrics (grammar, phonology, fluency, fidelity) appeared in 2026. — [BackTranslation2.0 [preprint]](https://arxiv.org/abs/2606.28673)
- NMMs are now generated from LLM-predicted sentence types (brows), but users still rate AI signing well below real signers. — [Zhang et al. CHI 2025](https://arxiv.org/html/2502.05661v1)
- Avatar comprehension in a deployed 2025 DGS study was below 50% of sentence units, with missing mouthing and static faces cited. — [Wasserroth et al. 2025](https://arxiv.org/html/2508.05358v1)
- Role shift with explicit signing-space reassignment and torso rotation has appeared in a 2025 robotics thesis (Unitree G1 humanoid), evaluated in simulation and in a case study with an ASL-fluent collaborator. It is not neural. — [UNR thesis 2025](https://scholarwolf.unr.edu/entities/thesis/c0aed129-7d1d-4228-a905-2a88877610d7)
- Field-level reviews: the 2021 SLP review and the 2024 JCLR comprehensive review (Tuveri). — [SLP review 2021, arXiv 2103.15910](https://arxiv.org/pdf/2103.15910); [Tuveri 2024, JCLR](https://polipapers.upv.es/index.php/jclr/article/view/20983)

### Inferences
- Open problems most relevant to an English-to-ASL continuous generator:
  1. No conditioning interface for discourse-level spatial state (which referent is at which locus) in any neural SLP system found.
  2. No datasets with locus, agreement or role-shift annotation at How2Sign scale. The CUNY mocap corpus (3 signers, 98 stories) and the ECHO/Cormier-style annotated corpora are small.
  3. Facial expression is carried as SMPL-X expression parameters without grammatical supervision.
  4. Gaze is absent from SMPL-X-based pipelines, while DePaul and the role-shift literature say gaze is the obligatory role-shift cue (DGS) and leads body rotation.
  5. There is no validated automatic metric for spatial or NMM correctness, apart from the moderate-correlation Kacorri DTW.
- A plausible differentiator, inferred and not tested by any source, is to combine rule-based locus and agreement structure with learned motion. Use LLM or rules to assign loci, agreement endpoints and NMM spans, as Zhang et al. do for NMM labels. Pass them as conditioning to a diffusion or token model, as NSA and SOKE do for text. Evaluate with the Q4 proxy suite plus Kacorri-style Deaf-user studies.

### Gaps
- No 2024–2026 neural SLP paper explicitly generating agreement verbs, locus reuse or role shift was found. This may reflect search limits; a direct arXiv listing sweep for "sign language production" in 2025–2026 was not done.
- No head-to-head comparison of 2024–2026 neural systems with rule-based avatars (Paula, JASigning) on the same Deaf-user comprehension protocol was found.
