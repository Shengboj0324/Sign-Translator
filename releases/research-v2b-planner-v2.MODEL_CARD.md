# Model card: sign-translator-research 2026-10-22-v2b-planner-v2

## Model details
- **System:** English text -> ASL sign plan (gloss planner) -> 3D signing motion (body, hands, head, face blendshapes, gaze) -> avatar or skeleton rendering.
- **Generator:** `training-runs/tokens-rvq-d0b-prior-loci-subset-v1-r2/checkpoints/step-00005000.pt` (SHA-256 `7b18e84783bf6a4b...`), conditioned on the English sentence; guidance 1.0, per-part temperatures {'body': 1.2, 'left_hand': 1.2, 'right_hand': 1.2, 'face': 1.0}.
- **Planner:** `training-runs/gloss-planner-asllrp-v2/checkpoints/step-00008000.pt` (SHA-256 `7e63b03416c56669...`).
- **Timing:** 24.0 fps, 0.57 s per planned sign.
- **Release SHA-256:** `a0ba2473b4a7e70af5934b64ac95d17d9bc10b0cc59a78cf2c775bbdc8b9e65a`.

## Intended use
- Research on sign-language generation and its evaluation; demonstrations to and evaluation by Deaf signers.
- **Not** a replacement for a qualified interpreter. **Not** for medical, legal, emergency or other safety-critical communication. **Not** for identifying signers.

## Training data and licences
| Corpus | Licence / terms | Research only | Non-commercial | Release permission | Citation |
|---|---|---|---|---|---|
| How2Sign (two-view subset) | CC BY-NC 4.0 | no | yes | not needed (CC BY-NC 4.0; courtesy notice sent 2026-10-22) | Duarte et al., How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language, CVPR 2021. |
| ASLLRP / DAI 2 continuous (planner) | ASLLRP Sign Bank Data Terms of Use | yes | no | requested 2026-10-22 (email to carol@bu.edu) | Neidle et al., ASLLRP / DAI 2 and the ASLLRP Sign Bank, Boston University (dai.cs.rutgers.edu). |

- **Public release allowed by the licence gate:** False; **commercial use allowed:** False.
- Research-only corpora are released only with the data owners' written permission, recorded in the release provenance.

## Evaluation

**Human evaluation:** pending (reviewer packets built; no sheets returned yet).

All automatic scores are relative: real recorded motion is the ceiling of each instrument and blank (duration-only) motion its floor. They do not measure comprehension by Deaf signers.

## Limitations
- Motion is estimated from monocular or two-view video, not motion capture; fingers and depth are noisier than body.
- Spatial grammar (placing referents in signing space) is weak: the planner's locus head is at the per-gloss baseline.
- Fingerspelling, name signs and classifier constructions are under-represented.
- Trained mostly on How2Sign instructional signing; other registers and dialects are less covered.

## Ethical considerations
- Built with and for Deaf signers' review; outputs must be presented as machine-generated.
- Training videos belong to their signers and creators; no video, keypoints or captions are released.

