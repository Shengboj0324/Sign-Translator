# Model card: sign-translator-research 2026-10-22-v2b-planner-v2

## Model details
- **System:** English text -> ASL sign plan (gloss planner) -> 3D signing motion (body, hands, head, face blendshapes, gaze) -> avatar or skeleton rendering.
- **Generator:** `training-runs/tokens-rvq-d0b-prior-loci-subset-v1-r2/checkpoints/step-00005000.pt` (SHA-256 `7b18e84783bf6a4b...`), conditioned on the English sentence; guidance 1.0, per-part temperatures {'body': 1.2, 'left_hand': 1.2, 'right_hand': 1.2, 'face': 1.0}.
- **Planner:** `training-runs/gloss-planner-asllrp-v2/checkpoints/step-00008000.pt` (SHA-256 `7e63b03416c56669...`).
- **Timing:** 24.0 fps, 0.57 s per planned sign.
- **Release SHA-256:** `7e594d3b0bdd882fd47b11e0da7650f01136fcfd4e0db99bce7a0a0c17ce4fa3`.

## Intended use
- Research on sign-language generation and its evaluation; demonstrations to and evaluation by Deaf signers.
- **Not** a replacement for a qualified interpreter. **Not** for medical, legal, emergency or other safety-critical communication. **Not** for identifying signers.

## Training data and licences

**ASLLRP attribution.** This model was trained in part on data from the American Sign Language Linguistic Research Project (ASLLRP), Boston University, obtained through the Data Access Interface (DAI 2, https://dai.cs.rutgers.edu/dai/s/dai) and the ASLLRP ASL Sign Bank (https://dai.cs.rutgers.edu/dai/s/signbank), under the ASLLRP Sign Bank Data Terms of Use. We thank Carol Neidle, Dimitris Metaxas and the ASLLRP team. No data, video or annotation is redistributed. Please cite C. Neidle, A. Opoku and D. Metaxas (2022), arXiv:2201.07899; C. Neidle, A. Thangali and S. Sclaroff (2012), LREC 2012 workshop (ASLLVD); and each DAI dataset used (the ASLLRP continuous signing corpus and the RIT sentence collection) as its DAI page specifies. *This wording awaits ASLLRP's review before distribution.*

**YouTube-ASL / YouTube-SL-25 terms (authors' guidance, 2026-10-29).** The video-ID lists and annotations of YouTube-ASL and YouTube-SL-25 are released under CC BY 4.0; the underlying videos and captions remain subject to the YouTube Terms of Service. A build trained on them may release its weights with attribution, but no raw videos, captions or full extracted feature sets are distributed with them. Cite Uthus, Tanzer and Georg (2023), arXiv:2306.15162, and/or Tanzer and Zhang (2024), arXiv:2407.11144 (YouTube-SL-25 includes YouTube-ASL). The authors note that the corpora's domain variance (many signers, recording set-ups and registers) can reduce generation consistency, that the static YouTube splits are unstable as videos are removed (How2Sign and FLEURS-ASL are the stable benchmarks), and that this guidance is practical, not legal advice. *This release contains no YouTube-ASL data; the terms above apply to any later build that does.*

| Corpus | Licence / terms | Research only | Non-commercial | Release permission | Citation |
|---|---|---|---|---|---|
| How2Sign (two-view subset) | CC BY-NC 4.0 | no | yes | not needed (CC BY-NC 4.0; courtesy notice sent 2026-10-22) | Duarte et al., How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language, CVPR 2021. |
| ASLLRP / DAI 2 continuous (planner) | ASLLRP Sign Bank Data Terms of Use | yes | no | granted 2026-10-23 (email from Carol Neidle, BU, cc Dimitris Metaxas, Rutgers): non-commercial release of weights trained in part on ASLLRP/DAI 2; condition: attribution wording reviewed by ASLLRP before release | C. Neidle, A. Opoku and D. Metaxas (2022), 'ASL Video Corpora & Sign Bank: Resources Available through the American Sign Language Linguistic Research Project (ASLLRP)', arXiv:2201.07899; Data Access Interface (DAI 2): https://dai.cs.rutgers.edu/dai/s/dai; ASLLRP ASL Sign Bank: https://dai.cs.rutgers.edu/dai/s/signbank; plus the citation given on each dataset's DAI page. |

- **Public release allowed by the licence gate:** True; **commercial use allowed:** False.
- Research-only corpora are released only with the data owners' written permission, recorded in the release provenance.

## Evaluation

**Human evaluation:** pending (reviewer packets built; no sheets returned yet).

All automatic scores are relative: real recorded motion is the ceiling of each instrument and blank (duration-only) motion its floor. They do not measure comprehension by Deaf signers.

## Limitations
- Motion is estimated from monocular or two-view video, not motion capture; fingers and depth are noisier than body.
- Spatial grammar (placing referents in signing space) is weak: the planner's locus head is at the per-gloss baseline.
- Fingerspelling, name signs and classifier constructions are under-represented.
- Trained mostly on How2Sign instructional signing; other registers and dialects are less covered.
- Signer demographics and regional variants are skewed towards the signers who appear in the source corpora (few signers in How2Sign; web video for YouTube-ASL builds).
- A research artifact: it is **not** ready for accessibility deployment without evaluation with Deaf and Hard of Hearing users.

## Ethical considerations
- Built with and for Deaf signers' review; outputs must be presented as machine-generated.
- Training videos belong to their signers and creators; no video, keypoints or captions are released.

