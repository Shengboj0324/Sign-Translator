# Sign-language datasets with grammatical and spatial annotation: inventory, licences and availability (as of October 2026)

Scope: datasets for training and validating an ASL grammar and spatial-reference layer (NMMs with onset and offset, loci and indexing, agreement-verb inflection, role shift, eye gaze, mouthing, sentence types, negation, topics), plus non-ASL corpora for method transfer. No data was downloaded for these notes. Licence names are quoted as each source states them. Where sources disagree, both are given.

## ASLLRP / Boston University resources (NCSLGR, ASLLRP SignStream 3 Corpus, ASLLRP Sign Bank, DAI / DAI 2)

### Takeaway
The BU ASLLRP corpora are the only openly searchable ASL data with frame-level onset and offset for both anatomical NMMs and grammatical NMMs. The anatomical fields cover head position and movement, eye brows, eye gaze, eye aperture, nose, mouth, cheeks, shoulders and body lean. The grammatical fields cover negative, wh question, yes-no question, rhq, topic/focus, cond/when, rel. clause, role shift, and subj agr / obj agr. Spatial indices are also encoded in the gloss strings (e.g. `i:GIVE:j`, `POSS-3p:i`). The data is available for "research and education purposes" only, "cannot be redistributed without permission", and commercial use needs explicit permission. NCSLGR XML and SignStream files are linked from a public BU page. ASLLRP SignStream 3 data and Sign Bank downloads go through DAI 2, where the Sign Bank downloads sit behind a login.

### Cited Findings
**Corpus sizes**
- As of the January 2022 report, DAI 2 hosts the BU ASLLRP SignStream® 3 Corpus with 2,048 utterances and 16,659 sign tokens (1,867 distinct signs or variants) from 4 native ASL signers. It also hosts the older BU NCSLGR SignStream® 2 Corpus with 1,002 utterances and 10,719 sign tokens. DawnSignPress contributed 527 more sentences from 15 native signers. Most utterances have 3 synchronized views: front, side, and a close-up of the face — [Neidle, Opoku & Metaxas 2022, "ASL Video Corpora & Sign Bank", arXiv 2201.07899](https://www.arxiv.org/pdf/2201.07899)
- The live DAI 2 page now lists larger counts: "ASLLRP sentences (39,174)" and "RIT sentences (3,882)". Per-participant counts are Ben 6,921, Cory 12,946, Jonathan 7,439, Rachel 11,868, P02 3,077 and P03 805. These look like the sign or utterance counts the search page uses, not the same unit as the 2022 report — [DAI 2](https://dai.cs.rutgers.edu/dai/s/dai)
- The ASLLRP site says "lots of new data" has been added to the corpus and the Sign Bank, and that new options to search and download Sign Bank data have arrived. It gives no new totals. SignStream® 3.5.1 was released in May 2025 — [ASLLRP index](https://www.bu.edu/asllrp/indexright.html)
- The Sign Bank has "almost 6,000 sign entries" and 41,830 lexical sign examples. It also holds about 300 gestures, more than 1,000 fingerspelled signs and 475 classifier examples — [arXiv 2201.07899](https://www.arxiv.org/abs/2201.07899v1)
- ASLLVD, which is included in the Sign Bank, has "almost 10,000 examples of about 3,000 distinct signs", each produced by 1 to 6 signers — [ASLLRP index](https://www.bu.edu/asllrp/indexright.html)
- The NCSLGR download page counts 1,866 distinct canonical signs (including gestures and classifiers) and 11,854 tokens. Without gestures and classifiers, the figures are 1,278 and 10,719. There are 1,002 utterances from short narratives and 885 elicited utterances. "Most of these data are from four native signers of ASL." — [NCSLGR download info](https://www.bu.edu/asllrp/ncslgr-for-download/download-info.html)

**What is annotated (Annotation Schema 3.0, Reports 11 and 13)**
- The conventions are documented in Report No. 11 (Neidle 2002) and Report No. 13 (Neidle 2007, an addendum). Together they define ASLLRP Annotation Schema 3.0. A 2025 "SignStream®3 Collection File XML Documentation" (Dimitriadis) is also available — [ASLLRP reports](https://www.bu.edu/asllrp/reports.html)
- Appendix A of Report 11 lists these anatomical fields:
  - head pos: tilt fr/bk, turn, tilt side, jut
  - head mvmt: nod, shake, side to side, jut
  - body lean, body mvmt, shoulders
  - eye brows, eye gaze, eye aperture
  - nose, mouth, English mouthing, cheeks, neck
- Appendix A also lists these grammatical fields: negative, wh question, yes-no question, rhetorical question (rhq), topic/focus, conditional/when (cond/when), relative clause (rel. clause), role shift, subject agreement (subj agr), object agreement (obj agr), adverbial (adv), POS, POS2, main gloss, non-dominant hand gloss, and English translation — [Report 11, Neidle 2002](http://www.bu.edu/asllrp/asllrpr11.pdf)
- Values carry start (s) and end (e) points plus graded intensity. Examples: eye brows can be raised, -raised, +raised, lwrd, -lwrd or +lwrd. Eye gaze directions include left, right, up, down, up/lf and so on, plus "watch hands", "track hd" and "to addressee". Eye aperture values include blink, squint, lowered lid, wide and closed. Mouth values include oo, mm, th, cha, sta, puh, pow and cs — [Report 11, Appendix A](http://www.bu.edu/asllrp/asllrpr11.pdf)
- Loci and agreement: spatial locations used for reference get "unique indexes that are indicated as subscripts". For example, `JOHN i:GIVE:j MARY` versus `neu:GIVE:j` for neutral subject agreement, `indef` for indefinite object agreement, and `POSS-3p:i` and `POSS-[person]-[numeral]:i/j/k` for possessives and plurals — [Report 11 §Reference, agreement, and locatives](http://www.bu.edu/asllrp/asllrpr11.pdf)
- Non-manual agreement: "an eye gaze to the same location, designated by the index 'i'" is glossed the same way. Definite agreement (gaze to a point) is distinguished from indefinite agreement (gaze over a region), and the same applies to head tilt — [Report 11](http://www.bu.edu/asllrp/asllrpr11.pdf)
- Role shift is annotated as `rs:John`. The "non-manual correlates of agreement or adverbials" were deliberately not labelled in the grammatical fields, although the underlying behaviours are annotated in the anatomical fields — [Report 11 §Role Shift / "But wait, there's more"](http://www.bu.edu/asllrp/asllrpr11.pdf)
- DAI 2 utterance search can filter on "the presence of a specific grammatical marking and/or specific anatomical non-manual actions (e.g., wh-question, negation, raised eyebrows, head nod)". A search for negative marking returns 532 utterances — [arXiv 2201.07899](https://www.arxiv.org/pdf/2201.07899)
- DAI 2 sign-level search also covers sign types (lexical, loan, fingerspelled, number, gesture), handshapes at start and end for both hands, and classifier types. The DAI 2 page shows these classifier counts: Body 14, Body Part 18, Descriptive 221, Instrument 26, Locative 14, Plural 12, Semantic 133 — [DAI 2](https://dai.cs.rutgers.edu/dai/s/dai)
- BU and Rutgers published non-manual tracking visualizations for seven categories: Conditional/When, Negation, Relative Clauses, Rhetorical Questions, Topic/Focus, Wh-Questions and Yes/No Questions. They plot eyebrow height, eye aperture, head rotation and upper-body position — [ASLLRP NM visualizations](https://www.bu.edu/asllrp/av/NM)

**Formats and download mechanics**
- NCSLGR comes in several forms:
  - A SignStream 2.2.2 archive (`Archive-ssdb-2-2-12.zip`), which needs Classic Mac SignStream.
  - An XML export from about 2011 that matches the DAI data, with the DTD in `old-dtd.xml`.
  - A Python parser (`signstream-xmlparser.zip`, by Christian Vogler).
  - A video index (`video_index-20120129.zip`, CSV and Excel) that maps annotations to video files.
- NCSLGR videos are viewed through the original DAI. "Some of this data was lost as a result of hardware failures", and some compressed files "play at half-speed". The time conversion is QuickTime time/1000×30, "assuming 30 fps". The page lists no licence text — [NCSLGR download info](https://www.bu.edu/asllrp/ncslgr-for-download/download-info.html)
- ASLLRP SignStream 3 annotations "are available in an XML export format", and corpus data "can be downloaded" from DAI 2 using a download cart — [arXiv 2201.07899](https://www.arxiv.org/pdf/2201.07899)
- The Sign Bank page offers two login-protected downloads: citation-form sign data, and segmented signs from the continuous corpora. ASLLRP Report 24 documents the segmented-sign CSV columns (video ID, main entry gloss, variant gloss, frame boundaries) — [DAI 2 Sign Bank](https://dai.cs.rutgers.edu/dai/s/signbank); [ASLLRP Report 24](https://www.bu.edu/asllrp/rpt24/asllrpr24.pdf)

**Licence**
- The DAI Terms of Use say:
  - "The data available from these pages can be used for research and education purposes" and "cannot be redistributed without permission."
  - "Commercial use, without explicit permission, is not allowed, nor are any patents and copyrights based on this material."
  - Users must cite the project and list the DAI and ASLLRP URLs.
  - No registration requirement is stated in the terms.
  — [DAI terms](http://www.bu.edu/asllrp/dai-terms.html)
- The sign-lang@LREC anthology lists the corpus licence as "ASLLRP License" — [sign-lang@LREC data: ASLLRP](https://www.sign-lang.uni-hamburg.de/lrec/data/asllrp.html)

**Published converted versions**
- Hugging Face `ncslgr/ncslgr` (contributed by @AmitMY) has configs `entire_dataset` and `annotations`. It yields `eaf` (ELAN) paths, `videos`, and gloss/text sentence pairs. Its card lists the licence as "mit". The loading script downloads from a third-party DePaul mirror: `http://asl.cs.depaul.edu/corpus/elanBUcorpus.zip` and `.../video.zip` — [HF dataset card](https://huggingface.co/datasets/ncslgr/ncslgr); [HF loading script](https://huggingface.co/datasets/ncslgr/ncslgr/blob/main/ncslgr.py)
- The TensorFlow community catalog for the same dataset shows one train split of 870 examples and "license unknown" (v0.7.0) — [TFDS community catalog](https://tensorflow.google.cn/datasets/community_catalog/huggingface/ncslgr?hl=it)
- EmoSign (`catfang/emosign`) has 200 ASLLRP clips with sentiment and emotion labels plus free-text descriptions of NMM cues. It is labelled "cc-by-sa-4.0", and the videos must be obtained from BU — [EmoSign HF](https://huggingface.co/datasets/catfang/emosign); [EmoSign paper](https://arxiv.org/pdf/2505.17090)

### Inferences
- The "mit" tag on the HF NCSLGR mirror conflicts with the BU terms ("cannot be redistributed without permission"). Treat it as non-authoritative. The safe route is the BU-hosted XML plus the DAI video index, used under the BU research and education terms.
- For a grammar layer, ASLLRP provides direct supervision targets: NMM spans for sentence type, negation, topic, conditional, relative clause and role shift. It also provides locus indices on agreeing verbs, pronouns and possessives, and gaze/head-tilt agreement in the anatomical tiers. Its 3-view video (including a face close-up) suits gaze and brow modelling.
- Aligning to How2Sign is a matter of convention only: the two corpora share no video. Glosses need mapping between ASLLRP Sign Bank glosses and whatever How2Sign gloss convention is used. NCSLGR assumes 30 fps, and How2Sign green-screen video is also recorded at 30 fps (see the How2Sign section), so frame-level transfer of NMM span models is straightforward once pose is re-extracted with the same pipeline used on How2Sign.

### Gaps
- Exact current (2026) totals for the ASLLRP SignStream 3 corpus. The DAI 2 counts and the 2022 report use different units, and no 2025–26 total was found.
- How to obtain a DAI 2 login (self-registration or request) is not stated on the pages fetched.
- Whether the DePaul ELAN conversion was authorised by BU was not found.

## ASL-LEX 2.0 (phonology)

### Takeaway
ASL-LEX 2.0 covers 2,723 signs with 16 phonological properties: sign type, major and minor location, contact, selected fingers, flexion, spread, thumb position and contact, handshape, non-dominant handshape, path movement, flexion and spread change, ulnar rotation, and repeated movement. It also includes frequency, iconicity and neighbourhood measures. The data is a public, freely downloadable OSF CSV with no registration. The sources disagree on the licence: OSF says CC-BY 4.0, while the website footer says CC BY-NC 4.0.

### Cited Findings
- 2,723 signs and "60+ properties". The download is at https://osf.io/zpha4/, and there are cross-references to ASL Signbank, SLAAASh and ASL-CDI 2.0 — [ASL-LEX download](https://asl-lex.org/download.html)
- The phonological properties are: Sign Type, Major Location, Minor Location (first and second), Contact, Selected Fingers, Flexion, Spread, Thumb Position, Thumb Contact, Handshape, Non-dominant Handshape, Path Movement, Flexion Change, Spread Change, Ulnar Rotation, and Repeated Movement — [Sehyr et al. 2021, JDSDE (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7977685/)
- Other properties include subjective frequency, iconicity, transparency (430 signs), lexical class, sign duration, neighbourhood density, phonotactic probability, and complexity. The data is distributed as .csv files on OSF — [PMC7977685](https://pmc.ncbi.nlm.nih.gov/articles/PMC7977685/)
- The OSF node licence is "CC-By Attribution 4.0 International" and the node is public — [OSF API](https://api.osf.io/v2/nodes/zpha4/?embed=license)
- The website download page says the database (excluding sign videos) is under "Creative Commons Attribution 4.0 International License", but its footer says "ASL-LEX database and visualization CC BY-NC 4.0". The about page also says CC BY-NC 4.0, and the "website and videos are © ASL-LEX.org" — [ASL-LEX download](https://asl-lex.org/download.html); [ASL-LEX about](https://asl-lex.org/about.html)
- The journal article itself is CC BY-NC-ND 4.0 — [PMC7977685](https://pmc.ncbi.nlm.nih.gov/articles/PMC7977685/)
- The Sem-Lex Benchmark has more than 84k isolated-sign videos aligned by experts with ASL-LEX, SignBank and ASL Citizen. The catalogue lists it as CC BY-NC-SA 4.0, and download instructions are at github.com/leekezar/SemLex — [sign-lang@LREC: Sem-Lex](https://www.sign-lang.uni-hamburg.de/lrec/data/semlexbenchmark.html); [Sem-Lex (NSF PAR)](https://par.nsf.gov/biblio/10481698)

### Inferences
- ASL-LEX gives no grammatical or NMM labels. Its value for the spatial layer is lexicon-level priors: which signs are body-anchored and which use neutral space (Major Location), and movement type. These help distinguish citation-form location from locus-driven modification on agreeing verbs.
- The licence conflict should be resolved by contacting the maintainers before any commercial use. For research, the OSF CC-BY 4.0 data file is the most permissive reading.

### Gaps
- No explicit verb-agreement or "directional verb" flag in ASL-LEX was confirmed.

## How2Sign: its annotations and later community annotation

### Takeaway
How2Sign was released under CC BY-NC 4.0 with direct Google Drive downloads and a download script. The released material is 80+ hours of front and side RGB video, sentence-level English, and front-view 2D OpenPose keypoints. Glosses were collected in ELAN by ASL linguists for part of the data but have never been released. In October 2022 the maintainers said gloss collection was "suspended", and issues requesting glosses remain open as of July 2026. The Panoptic 3D (about 3 hours) and depth modalities are also unreleased. No NMM annotation exists.

### Cited Findings
- More than 80 hours, "more than 35k sentences", a vocabulary of 16k English words and 2,456 How2 videos. 11 signers appear (5 hearing, 4 Deaf, 2 hard-of-hearing). Green-screen recording used a depth camera and an HD frontal camera plus a lateral HD camera, all at 1280×720 and 30 fps. A subset of about 3 hours from the val and test splits was re-recorded in the Panoptic studio for 3D keypoints — [How2Sign paper, arXiv 2008.08143](https://arxiv.org/pdf/2008.08143)
- "We collected gloss annotations by employing ASL linguists … using ELAN". Annotating took on average one hour per 90 seconds of video. The paper notes that glosses "do not indicate grammatical uses of facial expressions (for example, raising the eyebrows is used in yes/no questions)" — [arXiv 2008.08143](https://arxiv.org/pdf/2008.08143)
- The website lists RGB videos and clips (front and side), English translations (original and re-aligned) and B-F-H 2D keypoints (front view) for download from Google Drive. Gloss, depth and Panoptic 3D are named in the header but have no download links ("We will be releasing the other modalities soon!"). The licence is "Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)" and the dataset is "publicly available for research purposes only" — [how2sign.github.io](https://how2sign.github.io/)
- The official `download_how2sign.sh` has a `## TODO` section listing "# Gloss annotations" and "# Panoptic Studio data". The side-view 2D keypoints option is commented out — [download_how2sign.sh (GitHub)](https://github.com/how2sign/how2sign.github.io/blob/main/download_how2sign.sh)
- Maintainer statements on GitHub:
  - 2021-08-12: "the gloss annotations are still not available yet" — [issue #3](https://github.com/how2sign/how2sign.github.io/issues/3)
  - 2022-02-25 (xavigiro): "planning to release a first part of gloss dataset during March 2022" — [issue #5](https://github.com/how2sign/how2sign.github.io/issues/5)
  - 2022-10-14 (how2sign): the annotated part is still being cleaned, and for the full corpus "The gloss annotation is suspended by now until we can find more annotators" — [issue #5 comment](https://github.com/how2sign/how2sign.github.io/issues/5#issuecomment-1278618505)
  - A request opened 2026-07-30 is still open: "Request for official How2Sign gloss annotation release" — [issue #28](https://github.com/how2sign/how2sign.github.io/issues/28)
- A 2025 paper states "How2Sign does not provide gloss annotations" and uses LLM-generated pseudo-glosses instead — [Guo, Li & Cohn 2025, arXiv 2505.15438](https://arxiv.org/pdf/2505.15438). An earlier UPC thesis also produced pseudo-glosses for How2Sign — [UPC: SLT with pseudo-glosses](https://upcommons.upc.edu/handle/2117/373981)
- The `sign-language-processing/datasets` TFDS library includes a `how2sign` loader (videos plus OpenPose, v1.0.0) — [sign-language-processing/datasets](https://github.com/sign-language-processing/datasets)

### Inferences
- No released How2Sign layer carries grammar or NMM supervision. NMM and locus labels for How2Sign would have to come from transfer: models trained on ASLLRP NMM spans, applied to How2Sign face and pose, and validated on a small hand-annotated How2Sign subset.
- How2Sign's CC BY-NC 4.0 allows redistributing derived annotations non-commercially with attribution. So a project-made How2Sign NMM/gloss layer could be shared, but ASLLRP-derived training labels could not be bundled with it.

### Gaps
- I found no published third-party gloss or NMM annotation of How2Sign, only pseudo-gloss generation papers.

## Other ASL resources (ASL Citizen, WLASL, CUNY mocap, OpenASL, YouTube-ASL, ASL STEM Wiki, Signbanks, Signing Savvy, ASL-HW-RGBD)

### Takeaway
Of these, only the CUNY ASL Motion-Capture Corpus has explicit spatial-reference annotation. It annotates spatial reference points (SRPs), pronoun and verb references to each SRP, spatially inflected verbs, role shift, negation, wh and yes/no questions, topics, conditionals and rhetorical questions, all in SignStream. It is available only by emailing Matt Huenerfauth, and no licence is stated. The other datasets in this section are lexical (ASL Citizen, WLASL, Sem-Lex, Signbank) or sentence-level translation corpora without grammar labels (OpenASL, YouTube-ASL, ASL STEM Wiki).

### Cited Findings
**CUNY ASL Motion-Capture Corpus (Lu & Huenerfauth)**
- Equipment: "two Immersion CyberGloves®, an Applied Science Labs H6 head-mounted eye-tracker, an Intersense IS-900 inertial/acoustic tracker (for the head), and magnetic/inertial sensors on an Animazoo IGS-190 bodysuit", plus three HD cameras (front, side, face) — [Huenerfauth & Lu 2010, LREC](https://huenerfauth.ist.rit.edu/pubs/huenerfauth-lu-2010-lrec.pdf)
- Annotation is done in SignStream. It covers sign glosses, part of speech, syntactic bracketing, and "non-manual marking of role shift, negation, who/what/where/when/why/how questions, yes-no questions, topicalization, conditionals, and rhetorical questions". It also records "when SRPs are established, which discourse entity is associated with each, when referring expressions indicate each SRP, and when any verbs are spatially inflected to indicate each SRP". Each SRP gets an index number on parallel timeline tracks, and classifier predicates are marked — [Huenerfauth & Lu 2010](https://huenerfauth.ist.rit.edu/pubs/huenerfauth-lu-2010-lrec.pdf)
- First release: 98 unscripted single-signer multi-sentence stories from 3 signers, each 30 s to 4 min. It includes glosses with start and end keyframes, a non-dominant-hand gloss file for some videos, a referents list, English translation, BVH and FBX mocap, and MOV video in front, side and face views. Access: "Please send email to matt at cs.qc.cuny.edu to inquire about accessing the corpus." — [LATLab corpus page](https://latlab.ist.rit.edu/corpus/)
- The RIT downloads page gives the request address as matt (dot) huenerfauth (at) rit.edu. No licence is stated, only a citation request. It also links supplements for spatially inflected verb models (SLPAT 2012, TACCESS 2011) — [LATLab downloads](https://latlab.ist.rit.edu/downloads.html)
- The LREC 2012 release paper describes 246 passages from 9 signers in the wider collection, plus a "CUNY ASL Mo-Cap Sub-Corpus of Inflected Verbs". The paper itself is CC BY-NC — [sign-lang@LREC 12005](https://www.sign-lang.uni-hamburg.de/lrec/pub/12005.html); [sign-lang@LREC sub-corpus entry](https://www.sign-lang.uni-hamburg.de/lrec/data/cunysubcorpus.html)

**ASL Citizen**
- 83,399 videos of 2,731 signs from 52 signers, released under "MICROSOFT RESEARCH LICENSE TERMS". Use is "solely for non-commercial, non-revenue generating, research purposes", and "you may not distribute the data or your modifications to the data". It is a direct download from the Microsoft Download Center (ID 105253, ASL_Citizen.zip plus Demographics.zip, 42.8 GB), and wget is supported — [ASL Citizen project](https://www.microsoft.com/en-us/research/project/asl-citizen/); [ASL Citizen licence](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/); [Download Center](https://www.microsoft.com/download/details.aspx?id=105253)
- It is lexical (isolated signs) only.

**ASL STEM Wiki**
- 254 Wikipedia STEM articles, about 300 hours, interpreted by professional signers including CDIs, collected with consent under IRB. It is a direct download (Download Center ID 106253, wget). The page refers to a "Dataset License" tab and directs commercial enquiries to ASL_Citizen@microsoft.com — [ASL STEM Wiki project](https://www.microsoft.com/en-us/research/project/asl-stem-wiki/); [arXiv 2411.05783](https://arxiv.org/abs/2411.05783v1)

**WLASL**
- Released "under the Computational Use of Data Agreement (C-UDA)". "All the WLASL data is intended for academic and computational use only", and commercial use is not allowed. The release is a JSON of source URLs, frame spans, gloss, bbox, signer_id and variation_id. Videos are fetched from source URLs, and missing videos are available through a request form that requires agreeing to the terms. It is lexical only — [WLASL GitHub](https://github.com/dxli94/WLASL)

**OpenASL**
- 288 hours, "over 200 signers", YouTube news and VLOG content — [Shi et al. 2022, arXiv 2205.12870](https://arxiv.org/pdf/2205.12870)
- The release is a TSV plus a bbox JSON and download and crop scripts (yt-dlp), under "Creative Commons BY-NC-ND 4.0 License" — [OpenASL GitHub](https://github.com/chevalierNoir/OpenASL)
- The ASL STEM Wiki paper notes "No signer consent" for OpenASL — [arXiv 2411.05783](https://arxiv.org/pdf/2411.05783)

**YouTube-ASL**
- 11,093 videos, 984 hours, 610,193 English captions. Only video IDs are released, via a GCS bucket (`gresearch/youtube-asl`). No licence text appears on the README — [google-research/youtube_asl](https://github.com/google-research/google-research/tree/master/youtube_asl); [Uthus et al. 2023](https://arxiv.org/html/2306.15162v1)

**ASL Signbank (Haskins/Gallaudet, Hochgesang)**
- 3,702 signs under "CC BY-NC-SA 4.0". "Public access to 2848 signs via browsable homepage", while "More detailed information and all signs require confirmed registration" — [Compendium: ASL Signbank](https://www.sign-lang.uni-hamburg.de/lr/compendium/lex/aslsignbank.html)
- The copyright page says "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License". The site recommends its ECV link in ELAN — [ASL Signbank copyright](https://aslsignbank.haskins.yale.edu/about/copyright); [conditions](https://aslsignbank.haskins.yale.edu/about/conditions/)
- An older Figshare note says online registration was not working and to email Julie Hochgesang (Gallaudet) — [Figshare](https://figshare.com/articles/online_resource/ASL_Signbank/9741788)

**Signing Savvy**
- I could not retrieve its terms of use (URLs returned 404 and search found nothing). It is a commercial subscription dictionary, which is likely relevant to scraping, but this is unverified.

**ASL-Homework-RGBD**
- 45 fluent and non-fluent signers recorded on Kinect v2 RGB-D, "annotated for grammatical features and non-manual markers". The arXiv paper is CC BY-NC-ND 4.0, and no download route was found — [arXiv 2207.04021](https://arxiv.org/abs/2207.04021)

### Inferences
- CUNY is the best match for spatial reference and agreement-verb inflection: SRP index tracks plus 3D wrist and hand trajectories and eye-tracker gaze. It is also the only dataset found that combines mocap with grammatical annotation for ASL. Expect an email negotiation with no standard licence.
- ASL Citizen, Sem-Lex and ASL-LEX can be joined through the Sem-Lex alignment to give a citation-form lexicon. A How2Sign gloss or recognition layer could use this lexicon to tell lexical location apart from spatial modification.
- OpenASL and YouTube-ASL are only useful as unlabeled pretraining for a grammar layer. OpenASL's ND clause and lack of signer consent argue against redistributing derived labels.

### Gaps
- Signing Savvy licence and terms of use.
- Whether ASL-HW-RGBD was ever publicly released, and which specific NMMs it labels.
- The ASL STEM Wiki licence name (it is on a tab that was not captured).
- The YouTube-ASL licence.
- A formal licence for the CUNY corpus.

## Non-ASL corpora with rich spatial and NMM annotation (DGS-Korpus, BSL Corpus, MEDIAPI-SKEL, Dicta-Sign-LSF-v2, NGT, Auslan, Dicta-Sign)

### Takeaway
Two of these can be downloaded automatically without registration: the Public DGS Corpus (custom "DGS Corpus License", linguistic research only) and the BSL Corpus open subsets (CC BY-SA 4.0, after accepting terms). The Corpus NGT open subset (CC BY-NC-SA) is available through The Language Archive. MEDIAPI-SKEL and Dicta-Sign-LSF-v2 require an Ortolang account. Auslan annotation files are only available on a data-sharing request, but they have the richest spatial and grammatical tiers: pointing with gaze and palm orientation, constructed action (role shift), and spatial modification of verbs.

### Cited Findings
**DGS-Korpus / Public DGS Corpus**
- The full corpus has 560 hours recorded, 657,000 tokens annotated and 90.9 hours annotated, from 330 participants. It is in iLex format, and the open subset is also available as ELAN and SRT. Access is tiered: "Open: 50 hours of video, annotation, and translation", while "Restricted: … individual license agreement". The licence is the "DGS Corpus License" — [Compendium: DGS Corpus](https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/dgscorpus.html)
- Release 4 (LREC-COLING 2024) brings the corpus to 52.4 hours. It adds MediaPipe and Apple Vision pose outputs alongside OpenPose, "including representations with 3D estimates for keypoint locations", and introduces the MY DGS – iLex portal. ANNIS search results can be exported as CSV. A second data collection with 46 new participants was scheduled for 2024–2025 — [Konrad et al. 2024, "Corpus à la carte"](https://www.sign-lang.uni-hamburg.de/lrec/pub/24050.pdf)
- Each transcript page offers direct links to iLex (.ilex), ELAN (.eaf), MP4 views, English SRT, CMDI metadata and OpenPose (.json.gz), with no visible login — [MY DGS – annotated transcript list](https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start-name_en.html)
- The licence page is titled "License Conditions for Movies, Pictures, Annotations, and Metadata on this Site". Material is available "for linguistic research", and you may download and use it "for this sole purpose". Other uses, higher-resolution video and bird's-eye views go through info@dgs-korpus.de. It names no Creative Commons licence — [DGS licence](https://www.sign-lang.uni-hamburg.de/meinedgs/ling/license_en.html)
- Mouthings and mouth gestures share a tier, with mouth gestures marked "[MG]" — [DGS-Korpus annotation conventions AP03](https://www.sign-lang.uni-hamburg.de/dgs-korpus/arbeitspapiere/DGS-Korpus_AP03-2018-01v04_en.pdf)
- The `dgs_corpus` TFDS loader (v3.0.0, videos plus OpenPose and Holistic) exists — [sign-language-processing/datasets](https://github.com/sign-language-processing/datasets)

**BSL Corpus**
- 125 hours recorded, 249 Deaf participants and 70,000 tokens annotated, in ELAN with IMDI metadata. The licence is "CC BY-SA 4.0" for narrative and lexical elicitation data, while a "User license" applies to conversation and interview data, which "requires confirmed registration" — [Compendium: BSL Corpus](https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/bslcorpus.html)
- Through CAVA, open data can be viewed without login, but downloading means "agreeing to the terms and conditions". Restricted data needs an End User Licence, and approval may take several weeks. Lexical annotations exist only for some regions — [BSL Corpus CAVA](https://www.bslcorpusproject.org/cava/)

**Corpus NGT**
- 72 hours, 100 Deaf signers, 150,000 tokens, in ELAN with CMDI metadata. The licences are "Openly accessible videos under CC BY-NC-SA 3.0 NL" and annotations "under CC BY-NC-SA 4.0". Open access is through The Language Archive, and restricted parts need an individual licence agreement — [Compendium: Corpus NGT](https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/corpusngt.html)
- The public release includes gloss, mouth action and sentence translation tiers. A study-specific scheme separates mouthing and mouth-gesture tiers — [Crasborn & Bank, sign-lang@LREC 14034](https://www.sign-lang.uni-hamburg.de/lrec/pub/14034.pdf)

**Auslan Corpus**
- 300 hours, 100 Deaf native and near-native signers, ELAN, 357 annotated movies. No licence is listed. The compendium notes "Annotation restricted to Aesop's fables", and files are behind registration and confirmed registration on ELAR — [Compendium: Auslan Corpus](https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/auslancorpus.html)
- Annotation types include:
  - ID glosses
  - gaze direction and palm orientation during points
  - "periods of constructed action ('role shift')"
  - part of speech
  - clause boundaries, verb arguments, macro-role and semantic-role tags
  - "presence or absence of spatial modification"
  - free and literal translations
- The annotation files "are not publicly available". Access is "on a data-sharing and data-enrichment basis", conditional on returning enriched files. Auslan Signbank is CC BY-NC-ND 4.0. The guidelines are at v2.0, November 2024 — [Auslan annotations](https://auslan.org.au/about/annotations/); [Guidelines 2024 v2.0](https://media.auslan.org.au/django-summernote/2024-11-29/AuslanCorpusAnnotationGuidelines_2024v2.0.pdf)

**MEDIAPI-SKEL (LSF)**
- 27 hours, more than 100 signers, about 17,000 subtitle tokens. Skeletons come as OpenPose JSON, with MediaPipe Holistic added in v2, and subtitles as WebVTT. Skeletons and subtitles are "CC BY-NC 4.0", limited to non-commercial research and teaching, and redistribution is only allowed via Ortolang. Original videos are available "Upon written agreement with Média'Pi". Video consultation through Ortolang requires registration. It has no grammatical annotation — [Compendium: MEDIAPI-SKEL](https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/mediapiskel.html)

**Dicta-Sign-LSF-v2**
- 11 hours, 35,000 tokens. There are 16 participants per the compendium metadata, but its description says 18 signers. It is "Fully annotated (100%)" for "lexical sign, illustrative sign, pointing, tag, number, typing, and gesture", with the data listed as OpenPose. The licence is "CC BY-NC-SA 4.0". "Access to videos, elicitation material and partial annotations via Ortolang requires registration". The handle is hdl.handle.net/11403/dicta-sign-lsf-v2/v1 — [Compendium: Dicta-Sign-LSF-v2](https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/dictasignlsfv2.html)
- The LREC 2020 paper (Belissen, Braffort & Gouiffès) lists "Depicting Signs" in its keywords and trains a model to detect iconic (depicting) structures — [ACL Anthology 2020.lrec-1.740](https://aclanthology.org/2020.lrec-1.740)

**Dicta-Sign (BSL/DGS/GSL/LSF)**
- 25 hours, 14–16 signers per language. Annotations, in iLex, cover "segmentation, glosses, gaze, clause boundaries, and HamNoSys". The licence is "Individual license agreement for researchers", with "Open access to subset of videos and elicitation material" — [Compendium: Dicta-Sign](https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/dictasign.html)

### Inferences
- For method transfer on a spatial-reference layer, the most useful corpora are:
  - Auslan, for pointing with gaze, constructed action and spatial modification of verbs. It is closest in label design to locus and agreement modelling.
  - Dicta-Sign-LSF-v2, for pointing versus depicting versus lexical segmentation.
  - DGS, for scale, mouthing, and multi-model pose with 3D estimates.
- None of these use ASL signs, so they help with segmenters and NMM or pointing detectors, not with ASL-specific glossing.
- The Public DGS licence is not Creative Commons and is limited to "linguistic research". Training a commercial or production model on it would need permission from info@dgs-korpus.de.

### Gaps
- The exact tier names for eye gaze and brow in the BSL Corpus and Corpus NGT guidelines were not retrieved.
- The Auslan Corpus licence name.
- Whether Public DGS ELAN files include an eye-gaze tier (none was found).

## Datasets combining 3D motion capture with grammatical annotation

### Takeaway
Only the CUNY ASL Motion-Capture Corpus clearly combines full-body, glove and eye-tracker mocap with grammatical and spatial-reference annotation. It is available on request. The LSF mocap corpora (MOCAP1, LSF-ANIMAL) have mocap with sign segmentation but little grammar. The DGS mocap capture found is small and isolated-sign. Dicta-Sign is video only. DGS Release 4 and How2Sign Panoptic offer estimated or multi-view 3D pose, not mocap. How2Sign Panoptic is unreleased.

### Cited Findings
- CUNY: CyberGloves, ASL H6 eye-tracker, IS-900 head tracker and IGS-190 bodysuit, released as BVH and FBX, with SignStream annotation of SRPs, inflected verbs and NMM grammar — [Huenerfauth & Lu 2010](https://huenerfauth.ist.rit.edu/pubs/huenerfauth-lu-2010-lrec.pdf); [LATLab corpus](https://latlab.ist.rit.edu/corpus/)
- MOCAP1 (LSF): mocap plus HD camera, 4 signers on several tasks, with annotation to "segment and classify the signs". The paper is listed CC BY-NC 4.0 — [sign-lang@LREC 16029](https://www.sign-lang.uni-hamburg.de/lrec/pub/16029.html); [HAL](https://hal.archives-ouvertes.fr/hal-01633625)
- LSF-ANIMAL is an LSF mocap corpus designed for avatar animation — [HAL](https://hal.archives-ouvertes.fr/hal-03005767)
- A DGS Vicon capture of one native signer covers 57 isolated signs (about 16,600 frames), made for avatar reconstruction, and is not a public corpus — [Forte et al. 2023, arXiv 2304.10482](https://arxiv.org/pdf/2304.10482)
- Public DGS Release 4 adds MediaPipe and Apple Vision outputs with 3D keypoint estimates — [Konrad et al. 2024](https://www.sign-lang.uni-hamburg.de/lrec/pub/24050.pdf)
- How2Sign Panoptic covers about 3 hours with more than 500 cameras, but is not released — [arXiv 2008.08143](https://arxiv.org/pdf/2008.08143); [download script TODO](https://github.com/how2sign/how2sign.github.io/blob/main/download_how2sign.sh)

### Inferences
- For validating 3D locus geometry (e.g. whether a verb's endpoint matches a previously established SRP), CUNY is the only ground truth with measured 3D trajectories. Everything else would require monocular or two-view 3D lifting, as the project already does for How2Sign.

### Gaps
- MOCAP1 and LSF-ANIMAL licence and Ortolang access specifics were not confirmed.
- No 2024–2026 DGS mocap corpus with grammar annotation was found.

## Practical access constraints summary

### Takeaway
Directly downloadable without an account:
- How2Sign: CC BY-NC 4.0.
- ASL-LEX: OSF CC-BY 4.0, though the site says CC BY-NC 4.0.
- ASL Citizen and ASL STEM Wiki: Microsoft Research License Terms, non-commercial, no redistribution.
- Public DGS Corpus: linguistic-research licence.
- NCSLGR XML: BU research and education terms, no redistribution.
- BSL Corpus open data: CC BY-SA 4.0, after accepting terms.
- OpenASL, YouTube-ASL and WLASL: ID or URL lists. OpenASL is CC BY-NC-ND 4.0 and WLASL is C-UDA.

Need an account, registration or agreement:
- DAI 2 Sign Bank downloads (login).
- BSL restricted data (End User Licence).
- Corpus NGT restricted data and DGS restricted data (individual licence).
- MEDIAPI-SKEL and Dicta-Sign-LSF-v2 (Ortolang registration).
- Auslan annotations (data-sharing request).
- CUNY mocap (email request).
- ASL Signbank full data (confirmed registration).

### Cited Findings
- How2Sign: CC BY-NC 4.0, Google Drive links and a script — [how2sign.github.io](https://how2sign.github.io/)
- ASL-LEX: OSF CC-By 4.0 — [OSF API](https://api.osf.io/v2/nodes/zpha4/?embed=license)
- BU/ASLLRP: research and education only, no redistribution or commercial use without permission — [DAI terms](http://www.bu.edu/asllrp/dai-terms.html). The Sign Bank downloads are login-protected — [DAI 2 Sign Bank](https://dai.cs.rutgers.edu/dai/s/signbank)
- ASL Citizen: Microsoft Research License Terms, non-commercial, no distribution — [licence](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/)
- Public DGS: linguistic research "sole purpose" — [licence](https://www.sign-lang.uni-hamburg.de/meinedgs/ling/license_en.html)
- BSL: CC BY-SA 4.0 for open data and a user licence for restricted data — [Compendium](https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/bslcorpus.html)
- NGT: CC BY-NC-SA 3.0 NL for video and CC BY-NC-SA 4.0 for annotations — [Compendium](https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/corpusngt.html)
- MEDIAPI-SKEL: CC BY-NC 4.0 with Ortolang registration — [Compendium](https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/mediapiskel.html)
- Dicta-Sign-LSF-v2: CC BY-NC-SA 4.0 with Ortolang registration — [Compendium](https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/dictasignlsfv2.html)
- Auslan: request only — [Auslan annotations](https://auslan.org.au/about/annotations/)
- CUNY: email request — [LATLab](https://latlab.ist.rit.edu/corpus/)
- OpenASL: CC BY-NC-ND 4.0 — [GitHub](https://github.com/chevalierNoir/OpenASL)
- WLASL: C-UDA — [GitHub](https://github.com/dxli94/WLASL)

### Inferences
- No ASL dataset with grammatical or NMM annotation is under a plain CC licence. Every grammar-bearing ASL source carries a bespoke non-commercial or no-redistribution licence (BU), or no stated licence (CUNY).
- A shippable grammar layer should therefore keep ASLLRP- and CUNY-derived labels as internal training and validation assets. Releasable artefacts would be How2Sign-based annotations, which can be shared under CC BY-NC 4.0, together with model weights, subject to BU's "no … patents and copyrights based on this material" clause. That clause should be clarified with Prof. Neidle before releasing model weights trained on ASLLRP.

### Gaps
- Whether BU considers trained model weights "redistribution" or "copyrights based on this material" is not addressed in the terms.
- Signing Savvy and YouTube-ASL licence terms are unconfirmed.
