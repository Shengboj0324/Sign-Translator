**Research and acceptance review dated October 5, 2026.** Financial constraints are set aside. Technical capacity, rights, consent, source quality, reviewer availability and statistical independence still affect whether the implementation can work.

The most useful next action is to request one complete continuous ASL motion sample from RIT/CUNY, while opening ASLLRP annotation access and recruiting qualified Deaf/ASL collaborators. These are complementary requirements. No researched source is yet verified to satisfy the entire current co-observed motion gate. Large RGB collections, isolated signs and fitted 3D are useful at other stages but do not remove that gap.

**Acceptance result:** 2,056 software tests passed. The current source portfolio remains ineligible for both research and commercial use. Completing the nine manual interventions does not alone make the full translator execute: source adapters, synchronized retargeting, canonical-state training integration and qualified empirical acceptance still require engineering and validation. The steps below distinguish supplied inputs, executable contract checks and actual phase acceptance.

[Return to the full project tracker](https://chatgpt.com/space/page_6abc723bf82081918e70d95f54b20d86)

## What the code actually requires

The policy in `signtranslator/data_engineering/phase2_policy.py` requires four bundles. One source must satisfy every capability within a bundle; different sources may satisfy different bundles. Taking hands from one independent recording and face or gaze from another cannot manufacture a co-observed sample.

| Bundle | Required evidence | Current implication |
|---|---|---|
| Co-observed motion | Continuous ASL, 3D body and both hands, facial expression, head pose, eye gaze, eyelid state, stable signer identity and text alignment; locally verified | A dataset title containing “3D” or “holistic” is insufficient |
| Linguistic reference | Continuous ASL, governed reference, manual timing, handshape, nonmanual, head movement and gaze annotations, qualified ASL annotation | English captions or automatic glosses do not close this bundle |
| Render rig | Body, hand, face and eye rig; locally verified | Asset acquisition and real-signing articulation acceptance are separate |
| Review governance | Qualified ASL review, Deaf co-design and accessible consent; qualified human evidence | A laboratory contact or template is not a recruited reviewer or completed review |

The canonical state in `signtranslator/pose/multichannel.py` has 11 required channel records: root translation; body, left-hand and right-hand rotations; head rotation and translation; facial coefficients; left and right gaze; blink; contact. Unavailable channels must remain explicitly masked, not fabricated.

Translations use metres; rotations use the declared 6D column convention; gaze vectors must be unit length; blink/contact coefficients are bounded. Each channel requires labels, coordinate frame, units, source identity, source hash and convention. Masks distinguish valid from observed values. Valid inferred values require an inference method. Invalid entries are zero-filled, including confidence. Timestamps use a finite, strictly increasing float64 seconds clock. Joint counts, blendshape bases and calibration cannot be guessed.

The authorization boundary binds an actual local evidence file and SHA-256 to the exact source. Motion and reference ingestion require derivative creation and model training permission for the selected scope. Research eligibility does not grant commercial use, redistribution, public deployment or identity use. The checker validates recorded claims and bytes; people must still establish the issuer's authority and the meaning of the permission.

## Research coverage and access labels

Coverage spans motion capture, fitted 3D, continuous discourse, isolated vocabulary, fingerspelling, linguistic annotation, facial comprehension evaluation, rig assets, collaboration and later speech evaluation. Discovery used primary laboratory and publisher pages, ACL/LREC, CVF, author repositories and official dataset portals. The [Hamburg resource compendium](https://www.sign-lang.uni-hamburg.de/lr/compendium/), [LREC dataset index](https://www.sign-lang.uni-hamburg.de/lrec/data/index.html) and [2026 sign language proceedings](https://aclanthology.org/volumes/2026.signlang-1/) provide further discovery routes.

This is a broad, requirement-driven survey, not a claim to have visited every scholarly website. “Direct archive” means an official download target was found, not that its complete contents were downloaded and validated. “Account/form” requires the owner's actions. “Request” means the publisher gives a contact route; there is no verified public archive. All capabilities below remain candidates until local files and their documentation are inspected.

## Continuous motion and 3D acquisition priorities

### RIT and CUNY ASL motion corpus

**First continuous-motion inquiry.** [Official access and release description](https://latlab.ist.rit.edu/downloads.html#corpus). Request access from **matt.huenerfauth@rit.edu**. The described release has 98 multi-sentence stories from three signers, with BVH/FBX, video views, gloss timing, translations and referents. Ask for one fully documented sample first.

Confirm the exported hand detail, gaze calibration and synchronization, facial representation, eyelid channels, stable signer identifiers and permitted derivative/training actions. A description of capture equipment does not establish that every captured channel is in the released files. Three signers also limit generalization. This is the strongest immediate inquiry for the continuous-motion requirement, not an accepted complete source.

### 3D LEX through SAPA

[Author repository and access instructions](https://github.com/OlineRanum/SAPA), [project](https://olineranum.github.io/SAPA/), [primary paper](https://aclanthology.org/2024.signlang-1.33/). Request the ASL subset from **o.ranum@surrey.ac.uk**. The work describes 1,000 ASL and 1,000 NGT isolated signs, body capture, instrumented hands and facial capture. The repository distinguishes MIT code from the dataset's stated CC BY 4.0 terms and directs users to the authors for data.

Useful for hand/body/face adapter and articulation development. Ask for timestamps, rig/rest definitions, raw versus processed channels, coefficient basis and actual eye/eyelid exports. Isolated ASL does not satisfy continuous discourse; NGT must not be relabelled as ASL.

### STEM ASL dialogue motion capture

[2026 LREC paper and PDF](https://aclanthology.org/2026.lrec-1.669/), [BRIDGE research programme](https://www.bridgeproject.net/research-in-progress). The paper describes a small dialogue/isolated-term study with two Deaf signers, motion capture and ELAN annotations. Access is a researcher agreement route, not a verified public download.

Contact the paper's authors through their listed institutional routes and request the continuous segment, synchronized source videos, marker/joint definitions and annotation release terms. It is a promising dialogue pilot, but small participant coverage and unverified face/gaze/eyelid availability prevent treating it as a complete training or evaluation corpus.

### SignAvatars

[Official repository](https://github.com/ZhengdiYu/SignAvatars), [dataset request form](https://docs.google.com/forms/d/e/1FAIpQLSc6xQJJMf_R4xJ1sIwDL6FBIYw4HbVVv_HUgCqeiguWX5XGPg/viewform). Request the ASL How2Sign subset explicitly. The collection spans multiple signed languages and contains fitted SMPL-X motion; it is not wholly ASL or direct motion-capture ground truth. Source RGB must be obtained separately.

The documented 182-value layout includes pose, hands, jaw, shape, expression and camera translation, but does not establish explicit gaze/blink channels. Camera translation must not be assumed to be calibrated root translation in metres. Preserve inference status, validity masks and source lineage. The dataset agreement and SMPL-X model terms are separate.

### How2Sign

[Official downloads and metadata](https://how2sign.github.io/). RGB, 2D keypoints and alignment downloads are available; the project also describes a Panoptic capture subset. Ask **amanda.duarte@upc.edu** which actual 3D/calibration releases are obtainable. A paper's modalities are not proof that all modalities are downloadable today.

Use the corrected alignment release where applicable and preserve source clocks. Existing local files should first be inventoried instead of downloading the same training release again. RGB and 2D keypoints cannot directly satisfy metric rotations, gaze or a governed linguistic reference.

### SignAvatar ASL3DWord

[Official repository](https://github.com/dongludeeplearning/SignAvatar). This singular project name is distinct from SignAvatars. Its README gives **ludong@buffalo.edu** as the request route and asks for identity, institution, research purpose and the requested data/checkpoints. It concerns word-level 3D reconstruction/generation. Useful as an auxiliary engineering or comparison candidate, not continuous ASL acceptance. Obtain the compatible SMPL-X assets through their official licensing route.

## Linguistic references and annotation access

### ASLLRP Data Access Interface

[Corpus interface](https://dai.cs.rutgers.edu/dai/s/dai), [account request](https://dai.cs.rutgers.edu/dai/s/request), [ASLLRP SignBank](https://dai.cs.rutgers.edu/dai/s/signbank), [resource overview](https://www.bu.edu/asllrp/ASL-SignBank-and-other-Resources.html). The interface and registration form were verified in the browser. Registration requests email, name, affiliation, credentials and acceptance of terms; the owner should complete those steps.

Ask for the exact continuous-corpus release, native annotations, media, tier documentation and stable signer/source identifiers. Keep [DAI terms](https://www.bu.edu/asllrp/dai-terms.html) separate from [SignBank terms](https://www.bu.edu/asllrp/signbank-terms.pdf); the DAI terms fetch returned a service error during this review. **carol@bu.edu** is a published project contact. Native linguistic annotation is useful input, but qualified mapping to the project's temporal representation remains necessary.

### ASL Signbank and CARD

[Current ASL Signbank](https://aslsignbank.com/), [login](https://aslsignbank.com/accounts/login/), [ELAN external controlled vocabulary](https://aslsignbank.com/static/ecv/asl.ecv). The current domain works and explicitly reports its move. Registration requires subsequent manual approval; contact **julie.hochgesang@gallaudet.edu** if needed. The old Yale host produced a certificate error and should not be used for this workflow.

[CARD data](https://sites.google.com/gallaudet.edu/card/data), [collections](https://sites.google.com/gallaudet.edu/card/data/collections), [SLAASh version 4 conventions](https://figshare.com/articles/online_resource/ASL_Signbank_ID_glossing_and_SLAASh_Annotation_Conventions_Version_4_0/30582434), [ELAN template DOI](https://doi.org/10.6084/m9.figshare.29205617.v1), [ELAN software](https://archive.mpi.nl/tla/elan). Select collection-specific media and EAF together. A shared lexicon or annotation template does not supply missing human judgements, time alignment or motion. Figshare access was not fully retrievable through the research tool; use the linked official routes manually.

### ASL LEX

[Download and terms page](https://asl-lex.org/download.html), [OSF data](https://osf.io/zpha4/). ASL-LEX supplies lexical and phonological properties useful for vocabulary coverage and pilot stratification. Its database terms do not automatically include permission to download or reuse the reference videos. Contact **asllexproject@gmail.com** for those uses. OSF returned an access error in this review. This source is neither continuous motion nor sentence-level timing.

### Dennis Cokely Parallel Corpus

[Collection](https://encompass.eku.edu/cokely_videos/), [I Have a Dream item](https://encompass.eku.edu/cokely_videos/1). The collection provides translated speeches with media and annotation resources under its published terms. It is valuable for a bounded linguistic pilot, not a large independent population or 3D source.

The existing local binding specifically expects `2_I_Have_a_Dream_720_CokelyAFSParallelCorpus_v1_0.mp4`, 650,953,008 bytes, SHA-256 `8eb18b6b2f01a5a179100b0acd84a639b5f2a0ee95fb0195d477b946217c6bb3`. That primary file is currently absent. The preserved publisher page identifies item 1. Recover that exact authorized payload or create a separately versioned replacement and revalidate its clock. Do not substitute the SD video under the HD hash.

## Larger datasets for supporting tasks

| Source and direct route | Useful scale or role | Boundary for this project |
|---|---|---|
| [ASL Citizen official page](https://www.microsoft.com/en-us/research/project/asl-citizen/) and [direct ZIP](https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip) | About 84,000 isolated videos and 2,700 signs; vocabulary and signer variation | RGB isolated signs; not continuous 3D. [Dataset license](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/) limits the stated research use. The old tracker access label is stale: a direct link now exists |
| [ASL STEM Wiki download centre](https://www.microsoft.com/en-us/download/details.aspx?id=106253) and [direct ZIP](https://download.microsoft.com/download/4/c/f/4cfec788-7478-4e47-9a15-ace9b6a96198/ASL_STEM_Wiki.zip) | 64,266 videos, 316 hours, 37 interpreters; technical-domain continuous content | Roughly 187 GB archive; translation pairs are not canonical SIR or calibrated motion. Read [dataset terms](https://www.microsoft.com/en-us/research/project/asl-stem-wiki/dataset-license/) |
| [PopSign 2.1](https://signdata.cc.gatech.edu/view/datasets/popsign_v2_1/index.html) and [portal](https://signdata.cc.gatech.edu) | 200,686 videos, 562 signs, 47 signers; mobile isolated-sign variation | Approximately 1.1 TB; useful robustness data, not discourse or complete motion channels |
| [NVIDIA ASL1000 listing](https://registry.opendata.aws/asl_1000/) and [request form](https://www.nvidia.com/en-us/gated-resources/trustworthy-ai-american-sign-language/dataset/) | Video and landmark data for recognition/representation experiments | Gated access; raw and derived modalities must be distinguished. An AWS bucket listing is not anonymous download authorization. Inspect the [project license and docs](https://github.com/NVIDIA/Trustworthy-AI/tree/main/ASL%20Developer%20Community) |
| [YouTube ASL official repository](https://github.com/google-research/google-research/blob/master/youtube_asl/README.md) | Large continuous video/caption discovery pool | Video IDs and captions do not confer rights, consent or complete motion; availability can decay |
| [OpenASL](https://github.com/chevalierNoir/OpenASL) | Continuous video/English pairs and preparation scripts | Source links may disappear; inspect media and annotation terms separately; cannot assume derivative permission or ASL generation readiness |
| [WLASL](https://github.com/dxli94/WLASL) | 2,000-sign isolated vocabulary benchmark | Third-party video and use-agreement constraints; no continuous multichannel state |
| [MS ASL](https://www.microsoft.com/en-us/download/details.aspx?id=100121) | Isolated-sign benchmark metadata | Supporting recognition benchmark; not complete continuous motion or qualified annotation |
| [ChicagoFSWild and Plus](https://home.ttic.edu/~klivescu/ChicagoFSWild.htm) | 7,304 and 55,232 fingerspelling sequences; official archive links on page | Fingerspelling only; inspect original-video lineage and signer splits before use |
| [FSboard dataset](https://www.kaggle.com/datasets/googleai/fsboard) and [primary paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Georg_FSboard_Over_3_Million_Characters_of_ASL_Fingerspelling_Collected_via_CVPR_2025_paper.pdf) | 147 consenting Deaf signers and over three million fingerspelled characters | Mobile fingerspelling task, not full ASL translation. The landing page was located; archive transfer was not tested |
| [SignNet 1M repository](https://github.com/openhe-hub/SignNet-1M/blob/main/README.md) | Synthetic multiview augmentation; released How2Sign-derived portion | Planned portions are not released evidence. Synthetic identities and camera views do not add independent human signers or source units |
| [Finnish Sign Language motion corpus index](https://www.sign-lang.uni-hamburg.de/lrec/data/finslmocapcorpus.html) | Methods reference for synchronized acquisition | Different language; not ASL supervision or a substitute for the current gate |
| [LibriSpeech](https://www.openslr.org/12) | Later audio front-end testing on English read speech | Not paired speech-to-ASL data and not sufficient for spontaneous speech, noise or accent acceptance |

## Rig assets and evaluation resources

[MakeHuman licensing](https://static.makehumancommunity.org/about/license.html) is relevant to the already acquired rig assets. Preserve the pinned asset inventory and inspect the exact asset's terms. Downloading another mesh does not resolve contact, palm orientation or expression fidelity.

[SMPL-X](https://smpl-x.is.tue.mpg.de/), [official model downloads](https://smpl-x.is.tue.mpg.de/download.php), [model terms](https://smpl-x.is.tue.mpg.de/modellicense.html). Register through the official site and obtain the precise model required by the chosen dataset. Research model access and commercial licensing are separate. Public contacts include **smplx@tue.mpg.de** for project questions and **smpl@max-planck-innovation.de** for commercial licensing.

[RIT facial-expression stimuli](https://latlab.ist.rit.edu/downloads.html#stimuli) provide a request route for facial comprehension materials, including questions and facial parameter data. Use them to design a qualified nonmanual pilot; they cannot be attached to unrelated motion and called co-observed data.

[SignBLEU](https://github.com/eq4all-projects/SignBLEU) can support multichannel annotation scoring. Matching its input format requires channel-wise gloss intervals. A score is not a substitute for Deaf participant comprehension or source-to-render review.

[Neural Sign Actors repository](https://github.com/baltatzisv/neural-sign-actors) was a project-page repository at review time. Do not schedule it as a ready-to-run training baseline without locating and validating executable code, assets and terms.

[Gallaudet Motion Light Lab](https://gallaudet.edu/visual-language-visual-learning/ml2/) is a relevant potential collaboration route: **motionlightlab@gallaudet.edu**. RIT LATLab and CARD/ASLLRP are other relevant professional contacts. Their existence does not establish availability, agreement or independence for this project.

## Manual action 1 Obtain the continuous co observed sample

Owner: project lead and source custodian. Blocks B01, B05 and B15.

1. Request a small RIT/CUNY continuous sample first; make a parallel researcher inquiry to the STEM dialogue authors. Request 3D-LEX separately for isolated articulation development.

2. Send a channel checklist: body and both hands in 3D; head pose; face expression; gaze and eyelids; stable signer IDs; text alignment; common clock; calibration; rig/rest state; observed versus fitted/inferred status. Include contact labels if available and accept an explicit unavailable response.

3. Ask for the exact release/version, source video, motion file, annotation file, channel documentation and checksum manifest. A useful first packet is one continuous passage plus the required calibration and reference files, not the largest available archive.

4. Request a list of channels that were recorded but are not included in the release. Confirm whether gaze is a vector, eye rotation, screen point or a linguistic annotation.

5. Keep received raw files immutable; hash them; record the acquisition date, signer/source IDs and every transformation separately.

6. Engineering then implements the source-specific parser, synchronization and mapping. Validate a sample before broad conversion.

**Acceptance:** local byte inventory, actual channel inspection, clock/calibration audit and policy eligibility. Missing face/gaze/eyelid evidence keeps this intervention open. A custom co-designed capture is the fallback if no release meets the complete contract; it requires a separate protocol and pilot, not invented channels.

## Manual action 2 Bind permission to actual actions

Owner: project lead and authorized rights holder. Blocks B06, B09, B59 and B60.

1. Identify the exact dataset, release, model assets and derivative files being requested.

2. Save the publisher license and any supplemental permission with issuer identity, date, covered versions and restrictions.

3. Obtain explicit coverage for ingestion, derivative creation, model training, rendered demonstrations and the intended sharing. Separate actions that remain disallowed.

4. Establish participant/identity coverage, retention, withdrawal handling and whether consent is directly obtained or supplied through a publisher's release. Do not turn indirect evidence into a direct-consent claim.

5. Record evidence URI, SHA-256, licensor, license identifier/URL, permitted uses/actions, personality-rights status, attribution and limitations in the existing authorization schema.

6. Have the responsible owner assess the scope; run the code's evidence checks afterwards. Store source-bound evidence rather than setting a broad “permitted” flag.

**Acceptance:** the exact source passes the chosen action-specific research gate with verifiable local evidence. This does not grant commercial use or prove legal interpretation automatically. No account agreements or permissions have been accepted on the user's behalf.

## Manual action 3 Recruit qualified creation review and adjudication

Owner: project lead and qualified Deaf/ASL collaborators. Blocks B02, B03, B04, B14, B48 and B51.

1. Contact candidate collaborators through the published institutional routes above with the project's ASL scope and the actual tasks.

2. Define three responsibilities: creation/annotation, independent review and adjudication. Record who can do each task, their relevant qualifications and any conflicts.

3. Confirm availability and weekly review capacity. Financial negotiation is outside this review, but reviewer time is still a scheduling dependency.

4. Supply accessible project information and consent/withdrawal materials.

5. Run a small common pilot. Collect individual judgements before adjudication; preserve disagreement rather than silently replacing one review.

6. Store reviewer identity, role, versioned rubric, item/source hashes, decision, timestamp and rationale in the governed workflow.

**Acceptance:** actual completed pilot records and adjudications tied to source versions. Contacting a lab or adding a name to the tracker does not close the requirement.

## Manual action 4 Accept the pilot scope and QC rubric

Owner: linguistic lead with engineering support. Blocks B10, B13 and B14.

1. Choose the initial domain, target users and supported phenomena: lexical content, fingerspelling, two-handed signing, spatial reference, negation/questions, head/gaze and contact where relevant.

2. Define visibility and occlusion strata and distinguish intended contact/overlap from collision or tracking failure.

3. Adopt a versioned timing and annotation convention using the appropriate source's native tiers and controlled vocabulary.

4. Document unknown, unobservable, ambiguous, excluded and adjudication-needed states. Do not fill gaps with English text or synthetic timing.

5. Use development items to calibrate QC thresholds and review examples. Reserve final-test material before tuning.

6. Review exclusions and disagreements, then sign off the pilot scope and rubric version.

**Acceptance:** an agreed rubric plus qualified decisions on real pilot items. It is permissible for a bounded pilot to exclude unsupported phenomena, but that must not be represented as completing the broader project requirement.

## Manual action 5 Establish independent units and the study design

Owner: statistical lead and source custodian. Blocks B11, B12, B37, B38, B39 and B68.

1. Build signer, recording, source, prompt and derivative lineage. Link duplicates, repeated views, crops and regenerated avatars to their original units.

2. Declare the target population and the unit on which each scientific claim is based.

3. Freeze train, development and final-test groups at that level. Keep test access separate from threshold calibration.

4. Define primary estimands, practically useful effects, precision/power targets, multiplicity handling and missingness/exclusion rules.

5. Use a pilot to estimate variability and dependency; choose the required population size from that design. Do not infer power from a large clip count.

6. Preregister the final comparison and analyze independent held-out units with uncertainty.

**Acceptance:** auditable split lineage and a feasible study plan, followed later by the actual empirical result. In the current exact paired sign-test calculation, four independent non-tied test units have a best possible two-sided p-value of 0.125. Six give 0.03125. Seven give 0.015625, enough only for resolution at 0.025 for two Bonferroni-adjusted primary endpoints. Seven is not a power recommendation. More seeds, windows or synthetic identities do not change the independent sample size.

## Manual action 6 Qualify the rig against real signing

Owner: rig specialist and independent ASL reviewer. Blocks B08, B46, B47, B48 and B69.

1. Bind the exact rig version, rest geometry, joint hierarchy, units, local/global conventions, face basis and eye controls.

2. Convert one accepted continuous source with explicit calibration and inverse transforms.

3. Render synchronized source-versus-avatar views at several relevant angles. Preserve the source timeline and show visibility limitations.

4. Review palm orientation, finger articulation, face, head/gaze, transitions and intended contact. Classify hidden geometry separately from visible penetration.

5. Record semantic and geometric failures by source interval, repair the mapping and rerun the same cases.

6. Obtain independent linguistic review and retain accepted failure scope.

**Acceptance:** a real source-to-state-to-rig round-trip with quantified transform/timing checks and qualified review. Static rig views and synthetic stream tests do not close this intervention.

## Manual action 7 Recover Cokely only if used

Owner: project lead or source custodian. Blocks B67 for this source only.

1. Open the I Have a Dream publisher item linked above and select its primary HD asset.

2. Obtain an authorized copy and compare its byte size and SHA-256 with the recorded binding.

3. If it differs, keep it as a new version; inspect codec, frame timestamps and annotation offsets before rebinding.

4. Re-run media/reference integrity and timing checks. Preserve unknown annotation endpoints.

5. Record the disposition. If the exact file cannot be recovered, exclude this source or qualify the new version explicitly.

**Acceptance:** exact payload restored or separately reviewed replacement. Other eligible sources can proceed independently; this recovery is not a universal project dependency.

## Manual action 8 Decide commercial lineage separately

Owner: project lead and relevant rights holders. Blocks B07 and commercial release.

1. Choose whether the current milestone is research-only or includes commercial training/deployment.

2. For commercial scope, inspect every dataset, base model, checkpoint, derivative and rig asset in the dependency chain.

3. Obtain explicit commercial and distribution coverage where required, with version-bound evidence. A free download or research agreement is insufficient.

4. Record incompatible items and select qualified replacements if necessary.

5. Run the commercial policy separately from research and preserve separate release decisions.

**Acceptance:** a complete commercial lineage for the actual shipped chain. Research development can proceed under valid research scope while this remains open. Ignoring financial constraints does not remove license restrictions.

## Manual action 9 Confirm capacity and target hardware

Owner: project lead, engineering lead and reviewers. Blocks B22, B34, B56, B62 and B65.

1. Name the initial device/platform, offline or streaming mode, input/output format and intended deployment environment.

2. Record available CPU/GPU or accelerator, RAM/VRAM, storage and network capacity without treating these as monetary constraints.

3. Measure one representative source conversion and one training/inference pilot: wall time, peak memory, failures, throughput and p50/p95 latency.

4. Reserve productive engineering hours and actual reviewer sessions each week.

5. Estimate full conversion/training from the pilot, retaining raw, derived and checkpoint storage separately.

6. Replan after measured throughput. Do not turn acquisition lead time into coding time.

**Acceptance:** a hardware-specific benchmark record and staffed calendar. A generic GPU recommendation or an assumed 40-hour week is not measured capacity.

## Ready to send request outlines

These are drafts for the project lead; no external messages have been sent.

**Motion custodian request:** “We are developing an ASL research pipeline and would like to inspect one continuous sample before requesting a larger release. Can you provide the exact version, source video, body and both-hand motion, head/face/gaze/eyelid availability, observed-versus-fitted status, timestamps, calibration/rest frames, signer identifiers, annotations and file checksums? Please identify unavailable channels and the actions permitted for research training, derivatives and rendered evaluation. We will preserve missingness and will not treat unrelated captures as synchronized.”

**Annotation or reviewer request:** “We need qualified ASL creation, independent review and adjudication for a bounded real-data pilot. Can we discuss relevant qualifications, role independence, accessible consent, weekly availability and a versioned review procedure? The pilot will include linguistic timing, nonmanual information, ambiguity and source-versus-avatar fidelity. Please indicate which tasks you can support and which require another collaborator.”

**Source follow-up:** request a channel dictionary and sample packet before bulk access; record each unanswered item separately. No response is not permission, and a publisher's willingness to discuss access is not data delivery.

## Executed acceptance evidence and what it proves

The October 5 full regression command was:

```sh
.venv/bin/python -m pytest -o addopts='' -q -W error
```

Result: **2,056 passed in 72.14 seconds**. A focused run covering multichannel state, scoped authorization, adjudication/evidence and renderer integration passed **147 tests**. The working tree retains the pre-existing edits to the renderer and its tests; the verification record includes code hashes rather than describing the checkout as clean.

The test `test_scoped_ingestion_requires_real_bytes_and_never_approves_phase` rejects absent authorization, rejects changed source bytes before creating an output, then exports and reloads a well-formed synthetic state with a verified archive hash. It explicitly asserts that phase exit remains false. The permissions and motion in that positive test are fictional fixtures.

Local evidence directory: `/Users/jiangshengbo/Desktop/Sign-Translator/Sign Translator Stage Documentation/evidence/source-research-2026-10-05/`. It contains `full-regression.txt` and `acceptance-evidence.json`, including code/log hashes, current research/commercial decisions, exact statistical resolution and the missing-Cokely check. These local paths are references, not uploaded evidence files.

| Acceptance stage | Evidence needed | Current status |
|---|---|---|
| Structural software contracts | Positive export/reload plus rejection of tampering, invalid state and absent authorization | Verified with synthetic fixtures |
| Real source receipt and action eligibility | Exact local source and permission bytes, all required capabilities, source identity | Not satisfied by current portfolio |
| Real adapter and synchronization | Native channel mapping, calibrated units, clocks, masks and inverse-transform evidence | Not established for an accepted complete source |
| Real rig round-trip | Source and render interval comparison, geometry and semantic review | Not accepted |
| Training integration | Active path consumes governed temporal reference and canonical state; real training and inference run | Incomplete; canonical export is not an active trainer integration |
| Statistical and human acceptance | Independent held-out units, declared analysis and qualified comprehension/fidelity evidence | Not established |
| Product execution | Target hardware, real input/output, measured latency and explicit failure handling | Not proved by the regression suite |

## Engineering that remains after manual inputs arrive

Manual completion is necessary but not sufficient. The following work must appear in the implementation plan.

1. Implement and validate the chosen native source adapter; preserve physical calibration and observed/inferred distinctions.

2. Connect canonical multichannel state to the actual training/data-loader path. The present export boundary is not evidence of this connection.

3. Connect governed temporal linguistic reference to the planner/training objective. The legacy exporter requires token fields and cannot silently substitute for the new representation.

4. Integrate inverse transforms and source-to-rig retargeting; test real hands, face and gaze through the render path.

5. Run real training/inference, compare meaningful baselines, validate uncertainty/calibration and implement the declared abstention or degraded behaviour.

6. Complete the independent empirical study and hardware-specific runtime acceptance.

A future claim that “the implementation executes after the inputs are supplied” must be backed by a recorded run through these stages using the supplied files. The present evidence supports the narrower claim that the existing ingestion contracts can export and reload a structurally valid, correctly bound test state and reject specified invalid inputs.

## Conditional work plan after receipt

Estimates below are engineering planning allowances, not measured completion promises. One workday means six productive hours. External access and reviewer scheduling are separate elapsed-time dependencies. The schedule starts only after a usable sample, relevant permission and reviewer availability are confirmed.

| Workdays | Engineering workload | Reviewer or owner work | Exit evidence |
|---|---|---|---|
| 1 to 2 | 12 hours: inventory, permission bindings, schema/channel audit and initial split freeze | Source custodian resolves channel questions | Source packet decision and reserved held-out groups |
| 3 to 6 | 24 hours: native adapter and synchronization | 2 to 4 hours: source conventions review | Valid canonical state and clock/unit report |
| 7 to 9 | 18 hours: inverse transforms and rig integration | 3 to 5 hours: articulation review | Recorded real round-trip with failures classified |
| 10 to 12 | 18 hours: governed reference and loader integration | 4 to 6 hours: annotation/adjudication pilot | Real training-ready governed example |
| 13 to 15 | 18 hours: real training/inference smoke run and defect repair | 2 to 3 hours: pilot output review | Reproducible bounded pipeline run |
| 16 to 18 | 18 hours: calibration, held-out split audit and baseline preparation | 3 to 5 hours: statistical/linguistic sign-off | Final analysis protocol locked before held-out evaluation |
| 19 to 22 | 24 hours: debugging reserve and regression | 3 to 5 hours: rerun affected review cases | Closed critical defects or explicit remaining blockers |
| 23 to 25 | 18 hours: measured runtime, acceptance packet and handoff | 2 to 4 hours: acceptance review | Bounded pilot decision, not general product approval |

Total planning envelope: **150 engineering hours plus approximately 19 to 32 specialist hours**. Approximately six workdays across the plan are for debugging and repeated validation. This is a conditional 25-workday pilot envelope; it does not include a newly commissioned capture, a fully powered study, large-scale model convergence or production deployment. Expand it if the sample lacks required channels, the adapter needs substantial reconstruction, or pilot throughput invalidates the assumptions.

## Immediate decision order

1. Request the continuous sample and channel dictionary; avoid buying time with bulk RGB downloads that do not meet the motion gate.

2. Open ASLLRP access and select an appropriate native annotation pilot.

3. Secure qualified roles and accept the pilot rubric.

4. Supply source-bound action evidence and independent-unit lineage.

5. Build and verify the actual adapter and round-trip before scaling.

6. Use larger lexical, fingerspelling and RGB corpora only for declared supporting objectives.

7. Keep research acceptance, commercial lineage and product release as separate decisions.

The nine interventions remain open until their actual handoffs and acceptance records exist. This guide improves access and makes the acceptance path concrete; it does not confer source rights, reviewer approval or empirical readiness.

