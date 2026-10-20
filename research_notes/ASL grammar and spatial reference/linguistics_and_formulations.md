# ASL Grammar and Spatial Reference as Measurable Properties of 3D Motion (Linguistics and Computational Formulations)

Scope note: ASL unless stated. Evidence from BSL, DGS, ISL, HKSL, NGT, FinSL and Auslan is labelled. Data target: per-frame body/hand joint rotations, torso/head translation (m, body/session frame), eye-gaze unit vectors, 52 MediaPipe/ARKit blendshapes, 24–30 fps. At 30 fps one frame is about 33 ms, and at 24 fps about 42 ms. All frame conversions below are my arithmetic, not values from the sources.

Coordinate convention used in the Inferences (my choice, not from the literature): body frame B with origin at the sternum/mid-shoulder point. +x is toward the signer's dominant (ipsilateral) side, after mirroring left-handers. +y is up and +z is forward (toward the addressee/camera). Azimuth φ = atan2(x, z): 0° is straight ahead and positive is ipsilateral. Elevation θ = atan2(y, √(x²+z²)).

---

## 1. Referent loci and indexing (pointing): number, placement, establishment, re-reference; discrete vs continuous

### Takeaway
ASL signers associate discourse referents with locations ("loci" or "spatial reference points", SRPs) in signing space by pointing (IX) and later point back to them. In practice a discourse usually holds only a few loci, typically a contrastive ipsilateral/contralateral pair. The literature offers no corpus-validated "typical number" for ASL. Theorists disagree on whether loci are a finite, listable set (an arc at chest height, or a few heights) or are continuous and gesturally determined (Liddell). Generation systems have used about 6 discrete locations, or 7 points on a chest-height arc, as a workable approximation.

### Cited Findings
- **Establishment and re-reference.** After first mentioning an entity (e.g., "Mary"), the signer may point to a location to create an SRP for her. Later references point to the same location. When the subject or object is not mentioned overtly, eye gaze or head tilt aimed at the location may be the only identifying cue. Signers may also set up a left and a right SRP and then orient the torso toward one side when contrasting them — [Lu & Huenerfauth, CUNY ASL corpus article (preprint)](https://huenerfauth.ist.rit.edu/pubs/lu-huenerfauth-2013-csl-preprint.pdf)
- **Corpus annotation.** The CUNY corpus records when each SRP is established (aligned to the establishing sign or phrase), which entity it denotes, each later reference ("r"), and verbs spatially inflected for it. In the worked example (a Mac vs PC comparison), one two-handed pointing sign IX-1-p:1,2 establishes 2 SRPs, which are later referenced by IX-1-s:1 and IX-1-s:2. Torso movements toward locations were not yet annotated ("planned for future work"). The corpus held 242 passages from 8 signers over 11 sessions (212 minutes) at the time of writing — [Lu & Huenerfauth preprint](https://huenerfauth.ist.rit.edu/pubs/lu-huenerfauth-2013-csl-preprint.pdf). Another summary reports a released portion of 246 passages from 9 signers, likely a later release — [CUNY dissertation record](https://academicworks.cuny.edu/gc_etds/1418) / [search summary of SLTAT 2011](https://vhg.cmp.uea.ac.uk/demo/SLTAT2011Dundee/10.html). The two counts differ.
- **Placement models.**
  - Some linguists (McBurney 2002; Meier 1990) hold that signers pick locations "on a semi-circular arc floating at chest height in front of their torso". Others (Liddell 2003) hold that loci vary in height and distance from the body. In either case "there are an infinite number of locations where entities may be associated" — [Lu & Huenerfauth 2012, SLPAT](https://aclanthology.org/W12-2909.pdf)
  - Lu & Huenerfauth sidestep the debate: the corpus stores raw 3D motion so that loci can be analysed as arc positions or as 3D points — [Lu & Huenerfauth preprint](https://huenerfauth.ist.rit.edu/pubs/lu-huenerfauth-2013-csl-preprint.pdf)
- **Discrete generator inventories.** A BSL animation generator (Marshall & Safar 2005) associated entities with "approximately 6" fixed locations and did not allow arbitrary placement. Lu & Huenerfauth used 7 positions on an arc around the signer, with fractional arc coordinates such as −0.6, 0.3, 0.6 and 0.9. In their mocap lab the arc positions were marked by 10 cm coloured paper targets on the walls at precise angles from the seated signer — [Lu & Huenerfauth 2012](https://aclanthology.org/W12-2909.pdf)
- **Eye-tracking task design.** Thompson et al. used a picture story with three look-alike boys plus a teacher, designed so that the only practical way to distinguish characters was distinct spatial loci. All 10 signers adopted this strategy. In task 3 the experimenter placed "Jack" on the signer's left and "Jill" on the right at roughly equal heights. Two subjects later moved them to new locations — [Thompson, Emmorey & Kluender 2006, NLLT](https://crl.ucsd.edu/newsletter/Publications/Thompson_NLLT_06.pdf)
- **Contrastive loci.** Loci in a contrastive discourse sit "typically on contra- and ipsi-lateral sides of signing space" (notated a/b). Horizontal space without a marked locus is treated as neutral — [Schlenker, Semantics & Pragmatics (via search summary)](https://semprag.org/index.php/sp/article/download/sp.15.1/pdf/10787)
- **Height as an iconic dimension.**
  - Schlenker, Lamberton & Santoro (2013, *Linguistics & Philosophy*) argue that loci are both formal variables and simplified pictures of their denotations. High vs low position is preserved by interpretation, e.g., a tall or powerful referent gets a high locus — [Schlenker et al. 2013 (Springer)](https://link.springer.com/article/10.1007/s11049-015-9323-7); [Schlenker "Iconic Features" abstract](https://www.leibniz-zas.de/fileadmin/media/Dokumente/Events/Workshops/2014-06_SPE7/abstracts/Schlenker_abstract.pdf)
  - Zucchi observes that the original data "only are compatible with a simple three-way distinction among high, neuter and low loci" — [Zucchi 2018](https://sites.unimi.it/zucchi/NuoviFile/Zucchi18b.pdf)
  - Continuous height has also been argued to allow gradient domain widening and narrowing — [semprag sp.15.1](https://semprag.org/index.php/sp/article/download/sp.15.1/pdf/10787)
- **Discrete vs gestural debate.**
  - Liddell (2003, 2011) argues that directional signs move between locations tied to referents. Since the points are infinite, the forms are unlistable and therefore partly gestural ("indicating"). Lillo-Martin & Meier (2011, *Theoretical Linguistics* 37) treat directionality as grammatical person marking, but concede that real-world referent locations are not part of grammar, so the language must interface closely with gesture. Quer (2011) argues Liddell's reasoning is flawed — [Wilbur 2013 summary (SLL 16:2)](https://www.jbe-platform.com/content/journals/10.1075/sll.16.2.05wil); [Lillo-Martin & Meier 2011 reply PDF](https://lillomartin.linguistics.uconn.edu/wp-content/uploads/sites/1112/2016/11/Lillo-Martin_Meier_2011b_f.pdf)
  - Van Hoek argues that loci carry conceptual information beyond referent identity — [Nordic J. Linguistics abstract](https://www.cambridge.org/core/journals/nordic-journal-of-linguistics/article/abs/conceptual-spaces-and-pronominal-reference-in-american-sign-language/E12479DA41238C0B47F4F2E358C669BE)
- **BSL corpus evidence (Cormier, Fenlon & Schembri 2015, *Open Linguistics*).** Conversational data suggest a "predominantly motivated rather than arbitrary use of space" with indicating verbs. Placement often reflects real or imagined topography rather than an arbitrary abstract arc — [Fenlon et al. record / search summary](https://researchportal.hw.ac.uk/en/publications/modification-of-indicating-verbs-in-british-sign-language-a-corpu/)
- **Analogue perception of space.** ASL signers perceive locations used in spatial descriptions in an analogue manner (Emmorey & Herzig 2003, cited second-hand) — [HSE classifier overview](https://ling.hse.ru/data/2019/12/08/1521178288/Classifiers%20in%20SLs.pdf)

### Inferences
- **Generator design.** The generator should treat loci as continuous 3D points with a soft prior, not as fixed slots.
  - Lateral prior: a mixture over azimuth with modes at roughly ±(30–60)° (ipsilateral/contralateral) plus 0° (neutral). The angle values are my guess; no source gives angles.
  - Height prior: three modes (high/neutral/low), justified by Zucchi's three-way observation. Neutral height is near the chest/lower sternum, per the "chest height arc".
  - Radius prior: within arm reach. For pointing to non-present referents, about 0.3–0.5 m forward of the sternum (my estimate from anatomy; no ASL source quantifies it).
  - Present referents (addressee, objects in the room) override the prior with their real direction (Liddell/Cormier "motivated" placement).
- **Locus assignment policy (algorithmic).** Allocate a new locus when a new third-person referent is introduced and will be re-mentioned. Place the first two contrasting referents on opposite sides, maximising angular separation, and reuse the same coordinates on every re-reference. Use height only for iconic reasons (tall/high status/physically above).
- **Per-discourse locus estimation.** Detect index-finger pointing frames: extended index, other fingers flexed, a short hold, and the wrist away from the body. Compute the pointing ray r_i(s) = p_i + s·d_i with p_i the index tip and d_i the unit vector MCP→tip, both in frame B.
  - If several pointings refer to one entity, the least-squares nearest point to their rays is x* = (Σ_i A_i)⁻¹ Σ_i A_i p_i with A_i = I − d_i d_iᵀ.
  - A simpler alternative is to intersect each ray with a sphere of radius R ≈ arm reach centred on the shoulder, or to use the fingertip endpoint itself.
- **Clustering.** Cluster endpoints/ray intersections over a discourse with:
  - a von Mises–Fisher mixture on the direction (φ, θ), or
  - DBSCAN with an angular eps of about 10–15° (a tuning guess), or
  - a Dirichlet-process/CRP mixture, since the number of loci K is unknown and discourse-specific.
- **Online re-reference.** Model the discourse as a Chinese-restaurant process. A new pointing joins an existing locus k with probability ∝ n_k·vMF(d | μ_k, κ), or opens a new one with probability ∝ α·uniform. Constrain with the gloss stream if available, since the gloss index (IX:1, IX:2) gives the identity directly in CUNY-style annotation.
- **Discreteness as an empirical question.** Discreteness can be tested on the user's own data. Pool locus centroids across discourses after per-signer normalisation by shoulder width. A multimodal distribution in φ/θ (BIC-selected K small) supports the discrete-arc view. A diffuse distribution plus a correlation with real-world referent geometry supports the Liddell/Cormier view.

### Gaps
- No peer-reviewed corpus figure was found for the typical number of simultaneously active loci per ASL discourse, or for their azimuth/elevation/radius distribution in degrees or metres. CUNY Figure 6 (SRPs per passage) exists but its values were not in the extractable text.
- No source gave typical locus distance from the body in metres.
- I could not retrieve primary texts of Liddell 2003, Lillo-Martin & Klima 1990, or Cormier, Schembri & Woll 2013 (*Lingua*, pronouns and pointing); their positions are reported second-hand above.

---

## 2. Agreement / indicating verbs: path and orientation between loci; regular, backwards and spatial verbs; single vs double agreement; formalising pointing–verb consistency

### Takeaway
Padden's (1988) three-way typology separates:
- **plain verbs**: no spatial modification;
- **agreeing (indicating) verbs**: path from subject locus to object locus and/or hand facing toward the object, marking person/number;
- **spatial verbs**: path between locative source and goal.

Backwards verbs (e.g., BORROW, TAKE) move from object to subject, which Meir explains by thematic source→goal mapping, with facing marking the syntactic object. Some verbs agree with the object only (TELL, ASK, SCOLD); others with subject and object (GIVE, MEET). Mocap work shows that the movement vector (direction and magnitude) is more stable across signers than absolute start and end points. BSL corpus data show that modification is not obligatory.

### Cited Findings
- **Padden's classes.** Agreement verbs mark person and number through the start and end locations of the sign. Spatial verbs modify the path to show source and goal locations of a moving theme. Plain verbs show neither — [SLL 00042 abstract](https://www.jbe-platform.com/content/journals/10.1075/sll.00042.kwo). Padden (1988) treats agreement as lexically specified — [Lourenço & Wilbur 2018 (FEAST)](https://repositorio.ufmg.br//bitstreams/bf7530a6-68bd-40c4-a4bf-e003bb17bffa/download)
- **Meir's thematic analysis (ISL).** Meir (1998, 2002) distinguishes two phonological tools: path direction, which marks thematic source→goal, and hand facing, which marks the syntactic object (as distinct from mere hand orientation). Backwards verbs move from object to subject while the syntax is unchanged — [Meir 1998, SLL](https://signlab.haifa.ac.il/wp-content/uploads/2013/01/Meir_SLL_1998.pdf); [Meir et al. LSA 2009 slides](https://signlab.haifa.ac.il/wp-content/uploads/2013/01/LSA_2009_Meir_et_al.pdf). Backwards verbs challenge any theory in which the path maps source and goal directly onto subject/object — [Quadros & Quer](https://signwriting.org/archive/docs14/Desktop/Academia/more_papers_by_ronice_quadros/Back_to_back_wards_and_moving_on_on_agre.pdf)
- **Single vs double agreement (Lu & Huenerfauth's modelled verbs).**
  - Subject+object: ASK (index finger moves from asker to askee, bending 1→X); GIVE (two flat-O hands move from giver toward recipient); MEET (two index fingers move toward each other and meet in the middle).
  - Object only: SCOLD (index wags at the object); TELL (index from mouth/chin toward the object).
  - Some verbs inflect for object location only (Liddell 2003; Padden 1988) — [Lu & Huenerfauth 2012](https://aclanthology.org/W12-2909.pdf)
- **Polynomial "point" model.**
  - Data: 7 arc positions; all 42 non-reflexive subject/object combinations (7×6) for double-agreement verbs; 7 examples for object-only verbs. For a two-handed double-agreement verb: 42 examples × 2 keyframes × 2 hands × 3 coords = 504 values.
  - Model: third-order polynomials per coordinate per keyframe, parameterised on subject and object arc positions.
  - Evaluation: no significant differences in comprehension or evaluation scores against animations made by a human animator — [Lu & Huenerfauth 2012](https://aclanthology.org/W12-2909.pdf). The SLTAT 2011 work reports a study with 12 native signers judging synthesized verbs as similar in quality to human-animator versions; it covers 8 verbs (ASK, TELL, SCOLD; GIVE, MEET, COPY, EMAIL, SEND) — [SLTAT 2011 page](https://vhg.cmp.uea.ac.uk/demo/SLTAT2011Dundee/10.html)
- **"Vector" model (Lu & Huenerfauth 2012).**
  - Inter-signer Euclidean distance between hand start/end points is large. The distance between movement vectors (end − start) is "much smaller".
  - Each verb example is modelled as (Δx, Δy, Δz) per hand: 3 parameters one-handed, 6 two-handed, fitted with the same polynomial method on subject/object arc positions.
  - Start location: a per-verb 3D Gaussian mixture of hand occupancy (6 components for SCOLD/GIVE/ASK/MEET, 5 for TELL), followed by a grid search for the start point whose path has maximum likelihood under the GMM.
  - Units: the coordinate scale sets signer shoulder width = 1.0.
  - Results: the vector model had lower cross-signer error than the point model, including when trained on 22 mixed examples from two signers and tested on 42 from a third.
  - Mocap system: Intersense IS-900 (hand/head/torso position + roll/pitch/yaw), 3 native signers, 90-minute sessions — [Lu & Huenerfauth 2012](https://aclanthology.org/W12-2909.pdf)
- **Later work.** Huenerfauth, Lu & Kacorri (2015) address context-dependent sign movement and synthesis of novel verb instances — [W15-5105](https://preview.aclanthology.org/fix_video/W15-5105.pdf)
- **BSL corpus (Fenlon, Schembri & Cormier 2018, *Language* 94).** In 1,436 indicating-verb tokens, "modification is not obligatory", and patient (object) modification is conditioned by factors including constructed action. The authors read this as support for indicating verbs fusing morphemic and gestural elements — [Fenlon et al. 2018 record](https://researchportal.hw.ac.uk/en/publications/modification-of-indicating-verbs-in-british-sign-language-a-corpu/). The exact proportion modified was not available in what I could retrieve. Note: one UCL brief lists different volume/pages (94(3), 543–574) than the institutional records (94(1), 84–118).
- **Gaze is part of agreement morphology.** Thompson et al. (2006) argue that manual and gaze agreement form one morpheme, like a circumfix: gaze begins "about 160 ms before" verb onset, while manual agreement is simultaneous with the verb (details in §4) — [Thompson et al. 2006](https://crl.ucsd.edu/newsletter/Publications/Thompson_NLLT_06.pdf)

### Inferences
- **Generator rule.** Each verb lexicon entry needs a flag: plain / agreeing (regular | backwards) / spatial, plus an agreement arity (object-only | subject+object).
  - Source locus L_src: the subject for regular verbs, the object for backwards verbs, the locative source for spatial verbs.
  - Goal locus L_goal: the object for regular verbs, the subject for backwards verbs, the locative goal for spatial verbs.
  - Facing target: always the syntactic object (Meir).
  - First person: the locus is the signer's chest (sternum point). Second person: the addressee direction (+z).
- **Consistency metrics** (for evaluation or as a training loss) between a verb's motion and the loci established earlier by IX. Let h(t) be the dominant-hand reference point (palm centre or fingertip) in frame B, and t₀/t₁ the verb's movement start and end.
  - Direction consistency: c_dir = ⟨Δh/‖Δh‖, u⟩ with Δh = h(t₁) − h(t₀) and u = (L_goal − L_src)/‖L_goal − L_src‖. Angular error e_dir = arccos(c_dir).
  - Endpoint alignment in azimuth/elevation: e_end = ∠(h(t₁) − o, L_goal − o), with o the sternum or shoulder. Because of reach, the hand rarely reaches the locus, so compare directions, not distances. The analogous start term e_start = ∠(h(t₀) − o, L_src − o) applies only for double agreement; for object-only verbs, h(t₀) is lexically fixed (e.g., chin for TELL).
  - Facing: f = ⟨n_palm(t₁) or d_fingertip(t₁), (L_obj − h(t₁))/‖…‖⟩; require f > cos(τ_f).
  - Training loss: L_agr = λ₁(1 − c_dir) + λ₂ e_end² + λ₃ (1 − f), masked to frames inside agreeing-verb spans. The vector-model finding above suggests weighting direction (λ₁) more than absolute endpoints (λ₂).
- **Spatial verbs.** The goal endpoint should sit lower than the gaze target for object agreement (see §4), and the path may carry iconic manner, which cannot be reduced to a straight vector.
- **Backwards-verb test.** In regular verbs the direction and facing vectors are roughly parallel; in backwards verbs they are roughly antiparallel. This sign can classify regular vs backwards behaviour in data.
- **BSL non-obligatoriness.** The generator should not force maximal modification everywhere. Let the modification magnitude scale with discourse factors (stronger under constructed action), and treat c_dir as a soft rather than hard constraint.

### Gaps
- No ASL study found quantifies typical path length (m), peak wrist speed, or duration of agreeing verbs vs plain verbs.
- The proportion of modified vs citation-form agreeing verbs in ASL corpora was not found. The BSL proportion exists in Fenlon et al. 2018 but I could not access the full text.
- Lu & Huenerfauth's error values (Fig. 7, 9, 10) and the exact arc angles (Fig. 2) are only in figures, not in the extractable text.

---

## 3. Role shift and constructed action: correlates, angles, durations

### Takeaway
Role shift (RS) is signalled by:
- eye-gaze shift away from the addressee,
- head turn,
- body lean/torso shift,
- character-appropriate facial expression.

Attitude RS (quotation/reported thought) is distinguished from action RS (constructed action, CA). CA is graded and forms a continuum with ordinary narration. In FinSL/LSFB mocap the head and eyes lead and the chest and dominant hand join last. I found no published torso-rotation angles or durations for ASL role shift.

### Cited Findings
- **Markers and analysis.** RS reports speech or thought from the attitude holder's viewpoint and is signalled by non-manual markers: eye-gaze shift, body leans, head turns. It has been analysed as context shift akin to quotation (Lillo-Martin 1995; Quer 2005; Herrmann & Steinbach 2012; Schlenker 2017) — [Koulidobrova & Davidson, SuB](https://www.uni-goettingen.de/de/document/download/373344eaab36d3c3d336367b011ad472-en.pdf/KoulidobrovaDavidson-SuB-attitudes.pdf)
- **Quer (LSC).** RS structures "do not constitute a uniform class" and serve both direct and indirect reported discourse — [Quer PDF](https://www.ruhr-uni-bochum.de/phil-lang/quotation/mam/pdf/quer.pdf)
- **Attitude vs action RS.** A *Semantics & Pragmatics* article (vol. 10, art. 9) separates Attitude Role Shift from Action Role Shift (constructed action). It argues the latter is not amenable to a quotational analysis, and that ASL and LSF obey the constraint that indexicals "shift together" — [semprag sp.10.9](https://semprag.org/index.php/sp/article/download/sp.10.9/pdf/7407)
- **Counter-evidence (NGT, RSL, HKSL).** A NGT corpus study (Blunier & Khristoforova) suggests RS non-manuals may mark a contrast between perspectives. RSL and HKSL show shifted indexicals without the matching non-manuals — [unige.ch](https://unige.ch/lettres/linguistique/download_file/view/93/355) (via search summary)
- **ASL constructed action.** A University of Chicago dissertation (2021) describes CA as enactment marked by facial expression, eye gaze or torso shifts, and reports that classifier constructions are more amenable to role shift than other verb types — [UChicago record](https://knowledge.uchicago.edu/record/3634)
- **BSL (Cormier, Smith & Sevcikova-Sehyr 2015, SLL 18(2):167–204).** CA is an enactment strategy in which hands and other body parts show a character's actions, thoughts and speech. It is classified into overt, reduced and subtle CA by the number of enacting articulators and the intensity of character perspective — [UCL Discovery record](https://discovery-pp.ucl.ac.uk/id/eprint/1447790); [search summary of SignCafe2 poster](https://pure.unamur.be/ws/files/108286985/SignCafe2_poster.pdf)
- **FinSL mocap (Jantunen, De Weerdt, Burger & Puupponen 2021, *Gesture* 19(1):76–101).** Head and upper-torso movement in CA vs regular narration supports a continuum between them, not the three-way overt/reduced/subtle split. The authors propose strong vs weak CA. The title states the core relation: "the more you move, the more action you construct" — [JYX record](https://jyx.jyu.fi/jyx/Record/jyx_123456789_75961)
- **Articulator onset order (FinSL/LSFB mocap).** When narration shifts into overt CA, "the head and the eyes tend to take the leading role, and the chest and the dominant hand tend to start acting last" — [SignCafe2 poster (UNamur)](https://pure.unamur.be/ws/files/108286985/SignCafe2_poster.pdf)
- **Thompson et al. (2006).** Bahan (1996) claims body lean toward a locus can replace eye gaze as agreement marking. Thompson et al. observed no body lean toward the object location in their relevant null-object sentences. They also did not analyse head tilt because the degree of tilt could not be measured accurately from video — [Thompson et al. 2006](https://crl.ucsd.edu/newsletter/Publications/Thompson_NLLT_06.pdf)
- **Head-pose tooling.** Head-pose estimation (MediaPipe, OpenFace, 6DRepNet) was validated against optical mocap on FinSL. All three measured yaw well, MediaPipe best; pitch and roll were worse — [Sargano, Vandenitte, Jantunen & Kimmelman 2024, IEEE ICIT](https://jyx.jyu.fi/jyx/Record/jyx_123456789_100521/Description)
- **Synthesis precedents.**
  - SLTAT 2022: a deep model computing avatar torso rotation from wrist positions — [ACL 2022.sltat-1.2](https://aclanthology.org/2022.sltat-1.2)
  - UNR 2025 thesis: robotic ASL coordinating torso rotation with arm motion for role shift, using referent-specific "signer's squares"; no angles in the abstract — [UNR thesis](https://scholarwolf.unr.edu/entities/thesis/c0aed129-7d1d-4228-a905-2a88877610d7)
  - MMS Player: parameterises head yaw and torso rotation — [MMS Player arXiv 2507.16463](https://arxiv.org/pdf/2507.16463)

### Inferences
- **Per-frame RS evidence features** (all computable from the user's data):
  - torso yaw ψ_T(t) and lean (pitch/roll, plus lateral translation of the upper torso in metres) relative to a per-signer narrator baseline;
  - head yaw ψ_H(t) and roll;
  - gaze–addressee angle γ_A(t) = ∠(g(t), +z_head-or-camera);
  - gaze–locus angles γ_k(t) to each active locus (see §4);
  - affect blendshapes departing from baseline (e.g., mouthSmile*, mouthFrown*, browDown*, eyeWide*, noseSneer*).
- **Narrator vs shifted.** A narrator state shows γ_A small, ψ_T ≈ ψ_H ≈ 0 and neutral affect. A shifted state shows gaze broken from the addressee (γ_A large), plus head/torso rotated or translated toward one side, plus character affect.
- **Change-point formulation.** On x(t) = [ψ_T, ψ_H, Δtorso_x, γ_A, affect-PCA₁…₃], run Bayesian online change-point detection (Adams & MacKay) or PELT with a Gaussian mean/variance cost. Then classify segments into {narrator, RS→locus k} with an HMM or HSMM whose state durations are learned.
- **Shift onset.** Because the head and eyes lead (FinSL/LSFB), take the earliest change in γ_A or ψ_H as RS onset and expect the torso to follow. In generation, schedule gaze/head first, then torso, then the hands.
- **Degree of CA.** Following Jantunen et al., model CA as a continuous intensity, e.g., a weighted norm of head/torso angular displacement and velocity relative to narration, rather than discrete classes. Threshold it into strong vs weak if a binary label is needed.
- **Direction of shift.** The common descriptive account has the signer shift toward the character's locus and direct gaze toward the locus of the character's interlocutor. I could not retrieve a primary source with measurements for this (see Gaps). Treat the direction rule as a hypothesis to estimate from data: correlate sign(ψ_T) during RS with the azimuth of the quoted character's locus and with the interlocutor's locus.

### Gaps
- **No published values found** for ASL (or other SL) torso rotation angles (degrees), head yaw amplitudes, or durations of role-shift episodes. Lillo-Martin and Quer give qualitative descriptions. The Cormier et al. 2015 and Jantunen et al. 2021 full texts (which may contain mocap magnitudes) were not accessible (403 / restricted).
- No verified primary-source statement was found on the exact direction rule (body toward the character's locus vs gaze toward the interlocutor's locus) for ASL.

---

## 4. Eye-gaze functions: referential gaze, gaze as agreement, gaze in role shift and turn-taking

### Takeaway
Head-mounted eye-tracking of 10 native ASL signers (Thompson et al. 2006) gives the key numbers. Gaze accompanies agreeing verbs toward the object locus about 74% of the time, but plain verbs only 11%. For spatial verbs, gaze goes to the locative (about 72% transitive, 68% intransitive). Locative gaze is lower in signing space than object gaze. Gaze starts about 160 ms (about 5 frames at 30 fps) before the verb. Gaze is therefore an agreement marker only when manual agreement is present. It does not mark person (Thompson et al. 2013), and DGS shows it less systematically.

### Cited Findings
**Thompson, Emmorey & Kluender (2006), NLLT 24(2):571–604** — [PDF](https://crl.ucsd.edu/newsletter/Publications/Thompson_NLLT_06.pdf)
- **Participants and equipment.**
  - 10 native signers (4 men, 6 women; mean age 28.6; all from Deaf families: 9 Deaf, 1 hearing native).
  - SMI iView head-mounted tracker with a scene camera and an eye camera, time-locked to a video of the signer. The addressee sat 6 ft away.
  - Gaze was coded on a monitor grid of one-inch cells (the addressee's head covered about 4 cells). Inter-rater reliability was 91%.
  - About 11% of verbs were discarded (unclear gaze, e.g., blinks).
- **Tokens.** 290 agreeing, 251 plain and 210 spatial verbs with clear gaze.
- **Agreeing verbs.** Gaze to the object location averaged 73.8% (74/75/73% across tasks 1–3). Gaze went to the indirect object in 98.4% of ditransitives.
- **Plain verbs.** Gaze to the object 11.1%, to the addressee 40.71%, to "other" 44.88% (mostly above the addressee's head or at the addressee's body). t(9) = 10.1, p < 0.0001 for agreeing vs plain object gaze.
- **Backwards verbs.** Gaze went to the syntactic object 82.5% and to the semantic goal 0%. For BORROW, the extracted text reads "13 signers directed their eye gaze toward the" syntactic object. This conflicts with N = 10 participants, so it is probably a token count or an extraction error; check the original.
- **Transitive spatial verbs.** Locative 72.22% (SD 24.09), object 2.62%, subject 2.5%, addressee 13.58%, other 9.01%. Gaze went to the locative and "moved away before the hand arrived at that location"; it did not track the hand.
- **Intransitive spatial verbs.** Subject/locative 68.15% (SD 15.18), addressee 23.07%, other 8.78%. For WALK-TO-type verbs where subject and locative separate (N = 22): locative 54%, subject 0%.
- **Gaze height.** Locative-agreement fixations were significantly lower than object-agreement fixations by about two one-inch grid cells (t(9) = 8.06, p < 0.0001). This was driven by verb type, not by referent placement: Jack and Jill stayed at the same height, yet gaze was lower for "Jack flew over to Jill" than for "Jack bothered Jill".
- **Timing.** Gaze "usually begin[s] about 160 ms before" verb onset. Manual agreement is concurrent with the verb.
- **Rejected claim.** The data reject the Neidle et al. (2000) "Boston Group" claim that gaze independently checks agreement for all verbs. With null objects and plain verbs, gaze went to the addressee. Seven native signers judged Neidle et al.'s example (2a) (*JOHN LOVE pro_j with gaze_j) ungrammatical; it is acceptable only with a topicalised object.

**Later and cross-linguistic work**
- Thompson, Emmorey, Kluender & Langdon (2013, *Lingua*) conclude ASL does not use gaze to mark person (no 2nd/3rd person distinction). Thompson (2009, *Bilingualism: Language and Cognition*) studies how learners acquire gaze agreement — [Thompson et al. 2013 PDF](https://vl2.gallaudet.edu/pdf/thompson-emmorey-luender-langdon-2013.pdf); [Learning to look](https://resolve-he.cambridge.org/core/journals/bilingualism-language-and-cognition/article/learning-to-look-the-acquisition-of-eye-gaze-agreement-during-the-production-of-asl-verbs/9B54F58E48ACDCA02AD0D5C0D64DC22D)
- **DGS.** Gaze with agreeing verbs is less systematic than in ASL (Hosemann, SLL 14:1) — [Hosemann, SLL 14(1)](https://www.jbe-platform.com/content/journals/10.1075/sll.14.1.05hos)
- **Default gaze.** In the absence of grammatical gaze, the default direction is the addressee (Siple 1978, per Thompson et al.) — [Thompson et al. 2006](https://crl.ucsd.edu/newsletter/Publications/Thompson_NLLT_06.pdf). Thompson et al. also mention gaze functions in regulating turn-taking and checking addressee comprehension.
- **Gaze as sole cue.** Gaze or head tilt toward a locus may be the only cue to the identity of an unexpressed subject/object (Neidle et al. 2000, per Lu & Huenerfauth) — [Lu & Huenerfauth preprint](https://huenerfauth.ist.rit.edu/pubs/lu-huenerfauth-2013-csl-preprint.pdf)
- **CUNY mocap** recorded "a vector representing the eye-gaze aim" alongside body joints — [Lu & Huenerfauth preprint](https://huenerfauth.ist.rit.edu/pubs/lu-huenerfauth-2013-csl-preprint.pdf)

### Inferences
- **Gaze–locus alignment angle.** With eye-centre position e(t) in frame B (head translation plus a fixed head-to-eye offset) and gaze unit vector g(t):
  - α_k(t) = arccos(⟨g(t), (L_k − e(t))/‖L_k − e(t)‖⟩) for each locus k;
  - α_A(t) is the same quantity for the addressee direction.
  - Label frame t as "gaze to locus k" if α_k < τ and α_k is the smallest among loci and the addressee. Thompson et al.'s 1-inch-cell coding does not translate into degrees, so τ ≈ 8–15° is a tuning value; MediaPipe iris-based gaze noise is likely several degrees.
- **Generation targets** per verb type (as probabilities, for stochastic realism):
  - agreeing: object locus p ≈ 0.74, else addressee/other;
  - backwards: syntactic object (the path's source) p ≈ 0.83;
  - transitive spatial: locative goal p ≈ 0.72, with gaze lower than the locus height;
  - intransitive spatial: locative p ≈ 0.68;
  - plain: addressee p ≈ 0.41, other p ≈ 0.45, object p ≈ 0.11.
- **Gaze timing in generation.** Gaze onset ≈ verb onset − 160 ms (≈ 4–5 frames at 24–30 fps). For spatial verbs, release gaze before the hand reaches the goal.
- **Lowered gaze for locatives.** Add a negative elevation offset Δθ_loc to the gaze target for spatial verbs. The magnitude is unknown in degrees ("two one-inch cells" on a monitor); estimate it from the user's data as median(θ_gaze | locative) − median(θ_gaze | object).
- **Gaze consistency loss.** L_gaze = Σ_t∈W w(t)·α_target(t)², where W = [t_verb_on − 5 frames, t_verb_off] and the target is chosen by verb type. Do not penalise gaze on plain verbs except toward the addressee.
- **Turn-taking and RS.** Gaze returning to the addressee (α_A drops below τ) is a candidate marker of RS offset and of turn-yielding at sentence ends. This is a hypothesis to validate; no quantitative source was found.

### Gaps
- No source found gives gaze–locus angular deviations in degrees, gaze offset times, or fixation durations, other than the 160 ms onset lead.
- No quantitative ASL data were found for gaze in role shift (e.g., proportion of RS episodes with gaze break, or timing relative to the torso) or for turn-taking gaze.
- The DGS proportions from Hosemann were not retrieved.

---

## 5. Non-manual grammatical markers: brow raise, brow furrow, headshake, nod, mouth morphemes; scope, timing, blendshape mapping

### Takeaway
- **Yes/no questions:** brow raise (100% in ASL) with upper-lid raise and the head forward (70%).
- **Wh-questions:** brow furrow (100%), the head forward and intensifying toward the phrase end, and cheek raise (79% in ASL).
- **Topics:** brow raise (95–98%) with a raised head (over 80%).
- **Conditional if-clauses:** brow raise (100%) with a raised head across the clause (85%).
- **Negation:** headshake, which optionally spreads over the c-command domain of Neg and must cover the VP if manual NOT is absent (Neidle et al.).

Brow raise does not spread like wh-furrow and headshake. NMMs align with the manual constituent's edges and change sharply at intonational-phrase boundaries. Brow-raise timing, on vs just before a sign, changes its function.

### Cited Findings
**Dachkovsky, Healy & Sandler (2013), *Phonology*, "Visual intonation in two sign languages" (ASL vs ISL)** — [PDF](https://signlab.haifa.ac.il/images/wendy%20articles/Visual_Intonation_in_two_sign_languages.pdf); [Cambridge abstract](https://www.cambridge.org/core/journals/phonology/article/abs/visual-intonation-in-two-sign-languages/C08E71864BC04B710560470FC0D3F4FE)
- **Method.** 6 ASL and 5 ISL native signers, 52 elicited sentences, FACS action-unit coding with scope. ASL timing came from ELAN time code; the comparison was frame-accurate. Inter-coder reliability 87%.
- **Constituent edges.** The timing of manual signals (hold, reduplication or pause at the final boundary) "delimits the scope of the constituent", and the facial/head intonation aligns with it. NMMs "tend to change dramatically at intonational phrase boundaries".
- **Yes/no questions.** Brow raise (AU1+2) 100% in ASL (95% ISL), usually with upper-lid raise (AU5). Head forward (AU57) in 70% of ASL (90% ISL), often combined with head down (AU54).
- **Wh-questions.**
  - Brow furrow (AU4) in all wh-questions in both languages.
  - The head moves forward (AU57) and "intensifies toward the end of the intonational phrase by moving further forward". The head is neutral or slightly raised vertically.
  - ASL shows cheek raise (AU6) in 79%; upper-lid raise (AU5) is rare in ASL (3%) vs 75% in ISL.
  - All 6 ASL signers produced WHERE BOOK? (wh-sign first).
- **Topics.**
  - Sentence-initial ASL topics were marked by raised brows: 95% in the results section, 98% in the summary (an inconsistency within the paper).
  - Over 80% of ASL topics had a raised head position over the entire topic. Squint appears on about 45% of ASL topics.
  - Topics were separated from the comment by a timing break in 76% of ASL (98% ISL).
- **Conditionals.** All if-clauses had raised brows. In 85% of ASL if-clauses a raised head position is held across the whole clause. ISL instead shows a gradual change to head forward (95%).
- **Squint.** ASL "squint" is achieved by cheek raise (AU6) plus upper-lip raise (AU10). ISL uses lower-lid tightening (AU7) plus AU11. ASL squint is often affective and less grammaticalised than in ISL.

**Other findings**
- **ISL background.** Brow raise signals continuation/dependency (yes/no questions, if-clauses); squint signals information that the addressee knows but cannot readily access (Dachkovsky & Sandler 2009) — [Language & Speech 2009](https://signlab.haifa.ac.il/images/wendy%20articles/Language_and_Speech-2009-Dachkovsky-287-314.pdf). Neutral ISL conditionals take raised brows; counterfactuals take raised brows plus squint — [CNLSE record](https://cnlse.es/es/recursos/biblioteca/facial-expression-intonation-israeli-sign-language-case-neutral-and-counterfactual)
- **Brow raise timing (Schlenker & Lamberton, *Glossa*, ASL with LSF replication).** Brow raise "on" an expression can mark focus/information structure or intensify gradable constructions. Brow raise "right before" an expression preferentially has the information-theoretic (focus) function. In their examples, brow raise on a pronoun lasted about 0.8 s and 1 s; brow raise right before the pronoun lasted about 0.2 s and 0.3 s — [Schlenker & Lamberton, lingbuzz 005080](https://lingbuzz.net/lingbuzz/005080/v5.pdf)
- **Spreading and scope (Neidle, MacLaughlin, Bahan & Kegl 1996 report).** Non-manual markings associate with heads of functional projections and "optionally spread over the c-command domain" of that node; in some cases spreading is obligatory — [Neidle et al. 1996 (UPenn mirror)](https://ling.upenn.edu/courses/Fall_1997/ling001/nmbk96.pdf); [BU OpenBU copy](https://open.bu.edu/server/api/core/bitstreams/9478b16d-0f74-42e9-b15e-4cc402708ce7/content)
- **ASL negation (S-Neg-V-O).** Manual NOT is optional. Without NOT, the headshake must accompany the entire VP, the c-command domain of Neg (Neidle et al. 2000, per Pfau/Oomen) — [Pfau talk](https://www.uni-goettingen.de/de/document/download/e56672c6baed01bfcce8384892845ebf.pdf/Roland%20Pfau-LING3.pdf). Veinberg & Wilbur (1990) found a strong preference for headshake with negative signs, but the headshake is optional when a manual negative sign is present — [SLL article via search](https://www.jbe-platform.com/docserver/fulltext/sll.20012.bem.pdf)
- **Differential spreading (Wilbur).** Negative headshake and [+wh] brow lowering spread over their c-command domain; brow raise does not, and its domain is harder to characterise — [Wilbur chapter (bct.53.08wil)](https://www.jbe-platform.com/content/books/9789027271747-bct.53.08wil) (via search summary; attribution not verified against the full text)
- **Headshake kinematics (NGT, ERC/CORDIS project).** Duration, amplitude and velocity of negative headshakes vary with whether the headshake spreads and which signs it covers. A cross-linguistic NGT/DGS/Libras study uses head-pose estimation to measure rotation — [CORDIS 101039378](https://cordis.europa.eu/project/id/101039378/reporting/fr); [Kimmelman seminar, Stockholm](https://www.su.se/institutionen-for-lingvistik/kalender/högre-seminarium-vadim-kimmelman-university-of-bergen-1.762079). NGT corpus data show considerable variation in headshake spreading (Oomen & Pfau) — [Oomen & Pfau poster](https://www.uni-goettingen.de/de/document/download/09aa1f06d2e680cd3a136d80fba0f615.pdf/Oomen%20&%20Pfau%20Poster%20GöNeg.pdf)
- **Auslan (Johnston 2018).** Clauses are almost always negated manually, and headshakes occur in "just over half" of manually negated clauses; the position and spreading of headshakes are not linguistically constrained — [Macquarie record](https://researchers.mq.edu.au/en/publications/a-corpus-based-study-of-the-role-of-headshaking-in-negation-in-au/)
- **Nods (FinSL).** Nods, nodding, head thrusts and head pulls were studied by form and function (Puupponen, Wainio, Burger & Jantunen 2015, SLL 18(1):41–89), with nod–syntax alignment compared across FinSL and SSL — [Puupponen dissertation](https://jyx.jyu.fi/handle/123456789/63646). Head nods are listed among boundary edge markers in the prosody literature — [Wilbur search summary / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2837852)
- **Brow height vs emotion (Weast 2008).** 6 native Deaf signers produced yes/no questions, wh-questions and statements in neutral, happy, sad, surprise and angry states; over 3,500 eyebrow measurements across 270 sentences, measured as pixel distance from mid-brow to the eye. Consistent sentence-type distinctions persist across emotions, but they depend on the timing and spread of brow raising/lowering — [Weast dissertation](https://uta-ir.tdl.org/uta-ir/handle/10106/986)
- **Onset vs core (Liu et al., FG 2013, Neidle/Metaxas group).** The linguistically significant part of a raised or lowered brow gesture begins after an anticipatory transitional phase ("onset"); the core is aligned with the manual signing — [FG13 (search summary)](https://webpages.charlotte.edu/szhang16/paper/FG13-ASL.pdf). The full text was not retrievable (404).
- **Mouth morphemes (Bickford & Fraychineaud, "Mouth morphemes in ASL: A closer look").** STUDY + puffed cheeks = "study a lot" (degree/intensity); WRITE + 'th' = "write sloppily". Mouth morphemes also mark size, quantity and distance, and their categories are not always clean — [Bickford & Fraychineaud PDF](https://www.dentonisd.org/cms/lib/TX21000245/Centricity/Domain/622/Mouth%20Morphemes%20Article.pdf). Liddell (1980) gave 'th' the meaning lack of control or inattention; a BSL study found a broader range of meanings and separated adverbial use from enaction and echo phonology — [LINGUIST List 25.2309](https://linguistlist.org/issues/25/2309/). For MM ("normal, regular, with ease") and CS ("close in time/space"), the sources I found were teaching resources only, not peer-reviewed — [Lifeprint NMM page](https://www.lifeprint.com/asl101/pages-layout/nonmanualmarkers.htm)
- **Blendshape name inventories.** ARKit defines 52 coefficients, e.g., browInnerUp, browDownLeft/Right, browOuterUpLeft/Right, cheekPuff, cheekSquintLeft/Right, eyeBlinkLeft/Right, eyeSquintLeft/Right, eyeWideLeft/Right, eyeLook{Up,Down,In,Out}{Left,Right}, jawOpen, jawForward, jawLeft/Right, mouthClose, mouthFunnel, mouthPucker, mouthPressLeft/Right, mouthRollUpper/Lower, mouthShrugUpper/Lower, mouthStretchLeft/Right, mouthSmileLeft/Right, mouthFrownLeft/Right, mouthDimpleLeft/Right, mouthLowerDownLeft/Right, mouthUpperUpLeft/Right, mouthLeft/Right, noseSneerLeft/Right, tongueOut — [Apple ARKit BlendShapeLocation docs](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation). MediaPipe Face Landmarker outputs 52 blendshape scores using largely the same names — [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (whether MediaPipe includes `_neutral` and omits tongueOut should be checked against the model's label file; not verified here).

**Computational recognition (ASLLRP: Neidle, Metaxas et al.)**
- **Neidle, Michael, Nash & Metaxas (2009, ESSLLI).**
  - Pipeline: ASM face tracker → 3D head pose (pitch, yaw, tilt) per frame → eye/brow region → pyramidal histograms of SIFT → SVM per frame → majority vote per sequence.
  - Headshake: detected from the smoothed yaw derivative with a 2-level temporal pyramid, exploiting the roughly equal number of frames turning each way.
  - Results: negation 95%; wh-question 95% (SIFT), 73% (head pitch only), 100% stacked SVM.
  - Data were tiny: 94 wh / non-wh sequences (44 test) and 42 neg / non-neg sequences (20 test), split by signer — [Neidle et al. 2009 PDF](https://www.bu.edu/asllrp/papers/Neidle_ESSLLI_2009.pdf)
- **Michael, Neidle & Metaxas (2010).** Signer-independent continuous recognition of wh-questions, negation and topics with HMMs over facial shape/appearance — [sign-lang@LREC 10029](https://www.sign-lang.uni-hamburg.de/lrec/pub/10029.html)
- **Metaxas et al. (2012, LREC).** 2D–3D deformable face tracking corrected for head pose; detects grammatical NMMs in continuous signing and separates subtly different expressions — [LREC 2012 L12-1123](https://www.sign-lang.uni-hamburg.de/lrec/pub/L12-1123.html)
- **Liu et al. (2014).** Component events (raised/lowered brows, periodic nods and shakes) recognised from linguistically annotated corpora — [OpenBU](https://open.bu.edu/items/a4d49622-abbc-44e8-b510-6abee3663137); [Neidle et al. 2014](https://open.bu.edu/items/d7cf033d-00b5-4f54-b8ac-fc0f4be0ef63)

**Generation-side work**
- SMPL-FX pairs FLAME expression with SMPL-X. It scores 58.3% vs 49.0% for the best pose baseline on NMFs-CSL, a dataset where signs differ only by non-manuals — [M3T, arXiv 2603.23617](https://arxiv.org/pdf/2603.23617)
- Sentiment/semantics-conditioned facial synthesis with a Fréchet Expression Distance metric, on How2Sign and PHOENIX14T — [Azevedo et al. 2024, arXiv 2408.15159](https://arxiv.org/pdf/2408.15159)
- Grammatical facial signals and emotion overlap on the face with different timing and must be blended — [arXiv 2508.07937](https://arxiv.org/pdf/2508.07937)

### Inferences
- **Blendshape mapping** (my mapping from FACS AUs used in the literature to ARKit/MediaPipe names; validate per signer).

  | Marker | Literature correlates | Blendshape features |
  |---|---|---|
  | Brow raise: yes/no, topic, conditional, focus | AU1+AU2 (+AU5) | BR = mean(browInnerUp, browOuterUpLeft, browOuterUpRight) + optional eyeWideLeft/Right (AU5). Head: pitch up (topic/conditional) or forward translation/pitch down (yes/no) |
  | Brow furrow: wh | AU4 (+AU6 in ASL) | BF = mean(browDownLeft, browDownRight) + cheekSquintLeft/Right (AU6). browInnerUp may co-activate (AU1+4 "worried" brow), so use BF − k·BR or a learned linear model. Head: forward translation (+z) increasing toward phrase end |
  | ASL "squint" | AU6 + AU10 | cheekSquint*, eyeSquint*, mouthUpperUp*, noseSneer* (weak) |
  | Negation headshake | — | head yaw oscillation; no blendshape. Often co-occurs with browDown*, mouthFrown*, mouthPress* (inference, unsourced) |
  | Nod | — | head pitch down-up pulse(s) |
  | MM | lips together, slightly protruded | mouthClose, mouthPress*, mouthPucker (low), mouthShrugUpper/Lower, jawOpen ≈ 0 |
  | CS | inference, unsourced | mouthStretch* (one side or both), cheekSquint*, head tilt toward the dominant shoulder (roll) |
  | TH | tongue protrusion | tongueOut (ARKit; may be absent in MediaPipe), jawOpen (small), mouthLowerDown* |
  | PUFF (puffed cheeks) | — | cheekPuff (+ mouthClose) |
  | OO / "pursed" (small/thin size descriptors) | — | mouthPucker, mouthFunnel |

- **Detection on blendshape time series.**
  1. Per-signer baseline: the median of each blendshape over frames labelled neutral, or the 20th percentile over the session. Normalise with robust z-scores.
  2. Smooth with a Savitzky–Golay or low-pass filter (≈ 3–4 Hz cut-off at 30 fps; a tuning choice).
  3. Hysteresis thresholding: enter a state at z > θ_on, exit at z < θ_off, with a minimum duration of 150–200 ms (≈ 5–6 frames). This minimum keeps the "right before" brow raise of ≈ 0.2 s (Schlenker & Lamberton) detectable.
  4. Better: a linear-chain CRF or an HSMM over states {neutral, onset, core, offset} × marker type. Use gloss-boundary features (sign onsets/offsets, holds) as observations, so that NMM edges snap to manual-sign edges as Dachkovsky et al. report. The onset/core split follows Liu et al. (FG 2013).
- **Headshake detector.** Compute yaw ψ_H(t) and its derivative. Detect ≥ 2 sign changes of dψ/dt with amplitude above A_min and roughly symmetric excursions, mirroring Neidle et al.'s yaw-derivative symmetry. Equivalent alternatives: band-limited energy of ψ_H (e.g., 1–4 Hz, my guess; not from the literature), or a continuous wavelet transform. MediaPipe yaw is the most reliable head angle (Sargano et al. 2024).
- **Scope rules for generation.**
  - wh-furrow and headshake span from the onset of the first sign in their c-command domain to the offset of the last sign (optional spreading). Without a manual NOT, the headshake must cover the VP.
  - Brow raise covers exactly the topic/if-clause/yes-no clause and ends at its I-phrase boundary, with a sharp change there. Hold the raised head across the whole if-clause (85% ASL).
  - Wh head-forward amplitude ramps up toward the phrase end.
  - NMM onset should anticipate the first manual sign slightly (Liu et al.'s anticipatory onset); the exact lead in ms is not quantified in what I retrieved.
- **Emotion vs grammar layering.** Model brow height as a grammatical component plus an affective component (additive in blendshape space, clipped to [0, 1]). Weast found grammatical distinctions persist across emotions through timing and spread.

### Gaps
- No ASL source retrieved gives NMM onset/offset lead or lag times in ms relative to manual signs. Baker-Shenk 1983 and Liddell 1980 contain timing observations but were not accessible.
- No source gave brow-raise amplitudes in physical units, or headshake frequency/amplitude in degrees for ASL. The NGT kinematics project reports that these vary but its numbers were not retrieved.
- Mouth morpheme meanings for MM and CS rest on teaching sources. No peer-reviewed ASL inventory with frequencies or timing was retrieved.
- No published mapping of ASL NMMs to ARKit/MediaPipe blendshapes was found. The table above is my derivation.

---

## 6. Classifier / depicting constructions: what is spatially iconic and how to represent it

### Takeaway
Classifier (depicting) predicates combine a categorical handshape with analogue location, movement and orientation that map space iconically:
- **entity classifiers**: the hand stands for the whole object;
- **handling classifiers**: the hand shows how the object is held;
- **size-and-shape specifiers (SASS)**: the hands trace extent.

Signers perceive these locations in an analogue (gradient) way. In ASL, handling handshapes for MOVE/PUT cluster on flat-O, S and C.

### Cited Findings
- **Classifier types.** The usual three-way split is entity classifiers (the whole entity), size-and-shape specifiers (two hands show size or extent) and handling classifiers. Handshapes are iconic but categorical, while movement shape, speed and location vary continuously — [HSE overview](https://ling.hse.ru/data/2019/12/08/1521178288/Classifiers%20in%20SLs.pdf); [UCLA course notes](https://linguistics.ucla.edu/people/zuraw/251_2020/03_classifiers.pdf)
- **Analogue perception.** Emmorey & Herzig (2003) showed ASL signers perceive locations in spatial descriptions in an analogue manner (second-hand) — [HSE overview](https://ling.hse.ru/data/2019/12/08/1521178288/Classifiers%20in%20SLs.pdf)
- **Terminology.** "Classifier predicates" (Liddell 1977) = "verbs of motion and location" (Supalla 1978). Liddell's 2003 book frames them as depicting verbs mixing lexical and gestural components — [CNLSE record](https://cnlse.es/es/recursos/biblioteca/sources-meaning-asl-classifier-predicates)
- **ASL MOVE/PUT (2025, *Open Linguistics*).** Two signers mainly used three handling handshapes (flat-O, S, C), chosen iconically from entity properties; flat-O may act as a generic handling classifier — [Open Linguistics 2025-0047](https://www.degruyterbrill.com/document/doi/10.1515/opli-2025-0047/html)
- **RSL handling predicates.** These encode two subevents, handling and movement, that are not necessarily causally linked — [NLLT 2019](https://link.springer.com/article/10.1007/s11049-019-09448-9)
- **Interaction with RS.** Classifier constructions are more amenable to role shift than other verb types (ASL dissertation) — [UChicago](https://knowledge.uchicago.edu/record/3634)
- **Prosody.** Classifier constructions often form their own prosodic phrases (P-Phrases) in picture descriptions (HKSL) — [Tang, Brentari, González & Sze 2010](https://ling.cuhk.edu.hk/people/gladys/doc/Crosslinguistic%20variation.pdf)
- **Corpus design.** The CUNY corpus deliberately limited classifier predicates to focus on pronominal spatial reference — [Huenerfauth & Lu 2010 LREC](https://huenerfauth.ist.rit.edu/pubs/huenerfauth-lu-2010-lrec.pdf) (search summary)

### Inferences
- **Representation.** Treat a depicting construction as a small scene graph: a set of entities, each with a classifier handshape class (categorical), a 3D pose trajectory in a "depiction frame", and a ground entity (often the non-dominant hand as a held reference).
  - The depiction frame is the signer's frontal space, scaled to a region of about 0.3–0.5 m, with a mapping from scene coordinates (top-down "diagrammatic" or life-size "viewer" space).
  - Generation is retargeting: place the hand(s) so that palm normal and finger direction encode object orientation, and so that the trajectory is a scaled copy of the scene path (shape, speed profile, contact or stop points).
  - The non-dominant hand holds its pose (a weak-hand hold) while the dominant hand moves relative to it. Express the dominant-hand trajectory in the non-dominant-hand frame for consistency.
- **Iconicity metric.** Given a target 2D/3D path P_scene(s) and generated path h(t) in the depiction frame, minimise a similarity-invariant shape distance, e.g., Procrustes distance after uniform scaling, or discrete Fréchet distance. Keep separate terms for orientation (geodesic distance on SO(3) between intended object orientation and hand orientation) and for the manner speed profile.
- **Locus interaction.** An entity's classifier location should coincide with its locus if one exists, so the locus model of §1 and the depiction frame must share coordinates. Discrepancy can be measured with the same azimuth/elevation errors.

### Gaps
- No study retrieved quantifies the metric scale of depiction space (e.g., hand displacement per unit scene distance) or typical classifier movement speeds.
- No study retrieved quantifies the ASL handshape inventory by frequency in corpora.

---

## 7. Prosody and timing: final holds, phrase-final lengthening, boundary blinks, sentence-final pointing

### Takeaway
- **Lengthening.** Intonational-phrase (I-Phrase) boundaries are marked by phrase-final lengthening: ASL sentence-final signs are about twice their phrase-internal duration. Phrase-internal signs last about 250–550 ms depending on movement type.
- **Final holds** also mark boundaries but are less reliable than overall duration.
- **Blinks.** Boundary blinks are short, overlap a sign by less than 75%, and mark the right edge of I-Phrases. In Tang et al.'s ASL sample every I-Phrase right edge carried a blink (48/48).
- **Other cues.** Nods and changes of head/brow position at I-Phrase edges.

### Cited Findings
- **Wilbur (1994), "Eyeblinks and ASL phrase structure", *Sign Language Studies* 84:221–240.** Formulated a generalisation for where inhibited periodic eyeblinks occur, tested its predictions, and revisited Baker & Padden's (1978) observation that signers do not blink after a conditional clause before a question (with 5 deaf and 6 hearing signers) — [CNLSE record](https://cnlse.es/es/recursos/biblioteca/eyeblinks-and-asl-phrase-structure)
- **Two blink types (Wilbur 1994a, per Tang et al.).**
  - Lexical blinks: longer, overlapping ≥ 75% of the co-occurring sign.
  - Periodic inhibited ("boundary") blinks: shorter, overlapping < 75%.
  - Baker & Padden (1978) found signers' blinks more tied to linguistic structure than speakers' blinks — [Tang, Brentari, González & Sze 2010, in Brentari (ed.) *Sign Languages*, CUP](https://ling.cuhk.edu.hk/people/gladys/doc/Crosslinguistic%20variation.pdf)
- **Tang et al. (2010) blink taxonomy** (after Sze 2008):
  - boundary/prosodic: at the left/right edge of an I-Phrase, ≤ 75% overlap;
  - lexical;
  - physiological: hands near the face, head movement, gaze change;
  - hesitation/self-correction.
- **Tang et al. (2010) I-Phrase criteria.** A candidate counts as an I-Phrase only if ≥ 2 of these cues are present:
  - lengthening ×2 vs the phrase-internal form;
  - change in head position (nod, or return to neutral from a tilt);
  - change in brow position;
  - blink.
- **Phrase-final lengthening.** In Brentari et al. (cited in Tang et al.), ASL signers' sentence-final productions were twice the length of the same signs phrase-internally. Holds were mainly at the end of single movements and "less reliable than sign duration" as a lengthening indicator — [Tang et al. 2010](https://ling.cuhk.edu.hk/people/gladys/doc/Crosslinguistic%20variation.pdf)
- **Phrase-internal sign durations (Tang et al. Table 23.4, pooled data).**

  | Movement | Single (1σ) | Repeated (2σ) |
  |---|---|---|
  | Path | 300 ms | 350 ms |
  | Trilled | 300 ms | 350 ms |
  | Local | 250 ms | 300 ms |
  | Complex (path + other) | — | 550 ms |

  — [Tang et al. 2010](https://ling.cuhk.edu.hk/people/gladys/doc/Crosslinguistic%20variation.pdf)
- **Blink profiles (Tang et al. Table 23.5).**

  | Language | Total blinks | IPs | IP right edges blinked | Blinks at IP right edges | Blinks at IP left edges | "Other" blinks |
  |---|---|---|---|---|---|---|
  | ASL | 57 | 48 | 48/48 (1.00) | 84% | 16% (all utterance-initial) | 0% |
  | JSL | 83 | 61 | 61/61 (1.00) | — | — | 2/83 |
  | DSGS | 48 | 46 | 42/46 (0.96) | — | — | 4/48 |
  | HKSL | 132 | 71 | 63/71 (0.89) | — | — | 53/132 (40%) |

  Blinks favour right edges, and left-edge marking implied right-edge marking. In HKSL, blinks also mark smaller P-Phrases — [Tang et al. 2010](https://ling.cuhk.edu.hk/people/gladys/doc/Crosslinguistic%20variation.pdf)
- **HKSL detail (Tang et al. Study 3, not ASL).** I-Phrases blinked: 89% / 74% for two signers. Lengthening at I-Phrase boundaries: 74% / 67%. Both lengthening and blink: 71% / 53%. Mean duration of the I-Phrase-final sign: 977 / 1,043 ms. Example final signs ran 880–1,540 ms, with final holds of 630–870 ms — [Tang et al. 2010](https://ling.cuhk.edu.hk/people/gladys/doc/Crosslinguistic%20variation.pdf)
- **Earlier HKSL finding (Sze 2008).** Only 59% of I-Phrases carried a blink, and 41% of boundary blinks fell at other constituents — [Tang et al. 2010](https://ling.cuhk.edu.hk/people/gladys/doc/Crosslinguistic%20variation.pdf)
- **DGS.** Blinks are not obligatory boundary markers, with a constant 70%/30% ratio of prosodic to non-prosodic blinks — [search summary, Potsdam SFB1287 "Prosodic boundary phenomena"](https://www.sfb1287.uni-potsdam.de/wp-content/uploads/Prosodic-boundary-phenomena.pdf)
- **Edge markers in general.** Lengthening, final holds and nonmanual edge markers (head nods, eye blinks) mark constituent boundaries — [PMC 2837852](https://pmc.ncbi.nlm.nih.gov/articles/PMC2837852)
- **ASL vs ISL topic breaks.** ASL separates topics from the comment by a timing break less consistently than ISL (76% vs 98%), but still shows NMM changes at I-Phrase boundaries — [Dachkovsky, Healy & Sandler 2013](https://signlab.haifa.ac.il/images/wendy%20articles/Visual_Intonation_in_two_sign_languages.pdf)
- **Bimodal bilinguals.** Experience with co-speech gesture can influence sign prosodic cues — [Cambridge BLC abstract](https://www.cambridge.org/core/journals/bilingualism-language-and-cognition/article/abs/can-experience-with-cospeech-gesture-influence-the-prosody-of-a-sign-language-sign-language-prosodic-cues-in-bimodal-bilinguals/98E889E5B174DB6876B6ACB386F9BA2B)

### Inferences
- **Frame budgets** (≈ 33 ms per frame at 30 fps):

  | Item | Duration | Frames at 30 fps |
  |---|---|---|
  | Phrase-internal path sign | ≈ 300 ms | ≈ 9 |
  | Phrase-internal local sign | ≈ 250 ms | ≈ 7–8 |
  | I-Phrase-final sign (×2) | ≈ 600–1,100 ms | ≈ 18–33 |
  | Final hold component (HKSL examples) | 0.6–0.9 s | 18–27 |

  A boundary blink should start in the second half of the final sign or in the transition to rest. It must overlap that sign by < 75%, and lexical blinks should be avoided inside signs.
- **Prosodic boundary detector** on the user's data. Per frame, compute:
  1. normalised hand speed ‖v_wrist‖ / shoulder width (a hold is speed < ε for ≥ k frames);
  2. local sign-duration ratio vs the signer's phrase-internal mean for that movement type;
  3. blink events from eyeBlinkLeft/Right (both > 0.5 for 2–6 frames; tuning guess);
  4. |Δ| of brow features (BR, BF) and head pitch/yaw across a window.

  Combine with a logistic or CRF boundary classifier, or use the Tang et al. rule "≥ 2 cues ⇒ I-Phrase boundary".
- **Generation.** At each predicted I-Phrase right edge: stretch the final sign ×≈2 (time-warp the movement plus add a hold), insert a boundary blink, and reset brow/head NMMs to neutral or switch to the next NMM. Allow topics to omit the timing break (ASL 76%) while keeping the NMM change.
- **Sentence-final pointing.** Clause-final pronoun copies or indexes (IX pointing back to the subject locus, Padden 1988) are commonly described, but I retrieved no source for their frequency or timing (see Gaps). If used, they should reuse the subject's locus (§1) and fall inside the final I-Phrase before the boundary blink.

### Gaps
- Wilbur 1994's own blink durations and percentages were not retrievable (only the abstract). The ASL figures above come from Tang et al.'s smaller sample (one dataset with 48 I-Phrases).
- No ASL source retrieved quantified the frequency, duration or timing of sentence-final pointing / pronoun copy. Padden 1988 and Neidle et al. 2000 describe it but were not accessible.
- No source retrieved quantified absolute final-hold durations for ASL specifically (the HKSL values above are only examples).

---

## 8. Mathematical and algorithmic formulations for 3D motion data (from the literature, plus derivations)

### Takeaway
The literature provides several concrete models:
- Lu & Huenerfauth's polynomial "point" and "vector" models of agreeing verbs, parameterised on arc locus positions, with a GMM path prior;
- Neidle/Michael/Metaxas head-pose-derivative and appearance-feature SVM/HMM recognisers for NMMs;
- the Thompson et al. gaze-category proportions and 160 ms lead as targets;
- the Tang et al. multi-cue boundary rule.

Most of the rest must be derived: locus clustering, gaze–locus angles, role-shift change-point detection and blendshape NMM HSMMs/CRFs. Summary formulas follow.

### Cited Findings
- **Verb synthesis models.** Third-order polynomials for hand keyframe position on (subject arc position, object arc position). The vector model predicts (Δx, Δy, Δz) per hand and chooses the start point by maximum-likelihood path under a per-verb 3D GMM (5–6 components), in shoulder-width-normalised coordinates. It has lower inter-signer error than the point model — [Lu & Huenerfauth 2012](https://aclanthology.org/W12-2909.pdf)
- **NMM recognition.**
  - ASM tracking → head pose → SIFT spatial pyramid on the eye/brow region + yaw-derivative temporal pyramid → per-frame SVM / stacked SVM → majority vote: neg 95%, wh 100% (small test sets) — [Neidle et al. 2009](https://www.bu.edu/asllrp/papers/Neidle_ESSLLI_2009.pdf)
  - HMMs for continuous wh / neg / topic recognition — [Michael et al. 2010](https://www.sign-lang.uni-hamburg.de/lrec/pub/10029.html)
  - 2D+3D face tracking with geometric + texture features — [Metaxas et al. 2012](https://www.sign-lang.uni-hamburg.de/lrec/pub/L12-1123.html)
  - Onset vs core segmentation of brow gestures — [Liu et al. FG 2013](https://webpages.charlotte.edu/szhang16/paper/FG13-ASL.pdf)
- **Gaze targets and timing** — see §4 — [Thompson et al. 2006](https://crl.ucsd.edu/newsletter/Publications/Thompson_NLLT_06.pdf)
- **Head-pose reliability.** MediaPipe yaw is reliable; pitch and roll less so — [Sargano et al. 2024](https://jyx.jyu.fi/jyx/Record/jyx_123456789_100521/Description)
- **Torso from wrists.** Torso rotation can be predicted from wrist positions with a learned model — [SLTAT 2022](https://aclanthology.org/2022.sltat-1.2)
- **Non-manual-sensitive evaluation.** NMFs-CSL is a benchmark where signs differ only in non-manuals (the SMPL-FX result is in §5) — [M3T 2026](https://arxiv.org/pdf/2603.23617)

### Inferences (derived formulations; notation as in the header)
1. **Body frame.** Torso translation T(t) and rotation R_T(t) are given. Map any world point p to B by p_B = R_T(t)ᵀ (p − T(t) − o_sternum). Normalise lengths by shoulder width s_w, which matches Lu & Huenerfauth's scale. Mirror x for left-dominant signers.
2. **Pointing ray and locus.** Pointing frames are those where index extension e_idx > θ, other fingers flexed, wrist speed < ε for ≥ 3 frames, and wrist forward distance z > z_min. The ray is r(s) = p_tip + s·d, with d = (p_tip − p_MCP)/‖·‖ (or the shoulder→tip direction, which some argue better reflects the intended direction; untested). Estimate the locus as the multi-ray least-squares point x* = (Σ A_i)⁻¹ Σ A_i p_i (A_i = I − d_i d_iᵀ). Or use the direction only: μ = normalise(Σ d_i) as a vMF mean.
3. **Locus inventory.** Fit a DP-vMF mixture to pointing directions per discourse: K̂ = number of loci, μ_k = centroids. Re-reference is the posterior assignment. For generation, sample new locus directions from a learned prior p(φ, θ) (e.g., a GMM over pooled μ_k), and penalise proximity to existing loci: choose φ_new = argmax min_k ∠(φ, φ_k).
4. **Verb–locus consistency.** Use e_dir = arccos(⟨Δh/‖Δh‖, u⟩), e_end, and the facing cosine f from §2. Report per-verb-type distributions. The backwards-verb signature is that the path runs object→subject while facing points to the object.
5. **Gaze–locus.** α_k(t) = arccos⟨g(t), unit(L_k − e(t))⟩ and α_A(t) for the addressee. Assign gaze target k*(t) = argmin over {loci, addressee} if the minimum is below τ, else "other". Verb-level agreement-gaze indicator: G = 1 if α_obj < τ for ≥ 50% of frames in [t_on − 160 ms, t_on + ½·dur]. Expected rates per verb type are the Thompson et al. proportions.
6. **Role-shift detection.** On y(t) = [ψ_T, ψ_H, x_torso, α_A, PCA_affect(1..3)] (z-scored per signer), apply Bayesian online change-point detection (hazard 1/λ with λ ≈ the expected segment length in frames) or PELT (penalty by BIC). Label segments with an HSMM over {Narrator, RS_k} with emissions N(μ_state, Σ_state); RS_k means are tied to locus k's azimuth sign. The "head/eyes lead" ordering lets the RS onset be defined as the earliest change among α_A and ψ_H. CA intensity is I_CA(t) = ‖[Δψ_H, Δψ_T, Δx_torso, dψ_H/dt]‖_W.
7. **NMM detection.**
   - Features: robust z-scored blendshape groups (BR, BF, squint, cheekPuff, mouthPucker, tongueOut, …) plus head pose (pitch, yaw, roll) and its derivatives.
   - Model: a linear-chain CRF or BiLSTM-CRF with labels {O, B-yn, I-yn, B-wh, I-wh, B-top, I-top, B-cond, I-cond, B-neg, I-neg, B-MM, …}. Overlapping markers (e.g., topic + wh) need multi-label tracks: one chain per marker family (brow, head, mouth).
   - Lightweight baseline: hysteresis thresholds plus a minimum duration of ≈ 5 frames, then snap edges to the nearest sign boundary within ±k frames.
   - Headshake: the count of sign alternations of the smoothed dψ_H/dt with amplitude above a threshold, as in Neidle et al. 2009.
8. **Scope constraint as a structured loss for generation.**
   - Given a syntactic span [i, j] of signs carrying marker m, the target activation a_m(t) = 1 on [onset(sign_i) − δ_lead, offset(sign_j)] and 0 elsewhere.
   - Use a soft loss L_m = Σ_t (â_m(t) − a_m(t))² plus a boundary-sharpness term at I-Phrase edges (Dachkovsky et al.: NMMs change sharply at IP boundaries).
   - δ_lead is not quantified for ASL brows. For gaze use 160 ms; for brows, estimate it from data.
9. **Prosodic boundary rule.** B(t_i) = 1 if ≥ 2 of {dur_i ≥ 2·dur_internal(type_i), blink in the right half of sign_i or the following transition, |Δbrow| > θ_b, |Δhead| > θ_h} (Tang et al. operationalisation). Generation inverts this: at a predicted boundary, set dur ×2, insert a blink, and reset NMMs.
10. **Evaluation.** Report the metrics above on held-out real data as reference ranges:
    - c_dir and e_end distributions;
    - gaze-target proportions vs Thompson et al.;
    - NMM span IoU with gold annotations;
    - boundary F1;
    - RS segment F1.

    Ground-truth locus identity can come from CUNY-style SRP annotation where available.

### Gaps
- No published algorithm found does unsupervised locus clustering on 3D ASL pointing data, so §8.2–8.3 are my derivations. CUNY's SRP annotations would allow validation.
- No published change-point or HSMM role-shift detector for ASL with reported accuracy was found.
- Metaxas-group accuracies use video features on small test sets (20–44 sequences). No published accuracy figures were found for MediaPipe/ARKit-blendshape-based NMM recognition on ASL.
- Several primary sources could not be retrieved in full: Neidle et al. 2000, Sandler & Lillo-Martin 2006, Liddell 2003, Baker-Shenk 1983, Cormier et al. 2015, Jantunen et al. 2021, Fenlon et al. 2018. Their quantitative content, if any, is not reflected here.
