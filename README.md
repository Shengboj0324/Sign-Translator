# Sign-Translator

A research codebase for **bidirectional neural sign-language translation** built
around a single core idea: treat sign language as a **continuous 3D language
space** rather than a sequence of isolated gestures. English/gloss meaning
embeddings and 3D human-motion embeddings are mathematically aligned into one
shared latent manifold, and continuous signing motion is produced by a
conditional diffusion generator sampling from that manifold.

This repository implements and **verifies the mathematics of that core** in
pure PyTorch. It builds, trains on synthetic data, and is fully unit-tested on
CPU with no large downloads.

---

## Scope and honesty about maturity

The long-term system envisioned is a multimodal neuro-symbolic framework
(speech foundation models → LLM semantic planner → shared manifold → 3D motion
diffusion → rendered avatar). That full system is a multi-year research and
data effort. **This repository is the rigorously-implemented novel core, not a
production-deployed product.** Concretely:

| Component | Status in this repo |
|---|---|
| Skeleton graph + ST-GCN motion encoder | **Implemented & tested** |
| Keypoint preprocessing + augmentation (invariance-tested) | **Implemented & tested** |
| CLIP-style contrastive motion↔language alignment | **Implemented & tested** |
| Gaussian diffusion (DDPM/DDIM) motion generator | **Implemented & tested** |
| Cross-modal attention denoiser + classifier-free guidance | **Implemented & tested** |
| Continuous sign recognition (sign→gloss, CTC) | **Implemented & tested** |
| Semantic planner (English→gloss seq2seq) | **Implemented & tested** |
| Bidirectional pipeline (speech→gloss→motion, motion→gloss) | **Implemented & tested** |
| Evaluation metrics (recall@k, MPJPE, WER, accuracy) | **Implemented & tested** |
| End-to-end joint training | **Implemented & tested** |
| Speech / text encoders | **Interface + lightweight Transformer stub** (swap in Whisper/wav2vec2/LLM) |
| Speech front-end (audio features → spoken tokens, CTC) | **Implemented & tested** |
| Full audio → spoken tokens → gloss → 3D motion path | **Implemented & tested** |
| On-disk data ingestion + corpus pipeline | **Implemented & tested** (synthetic corpus; real corpora plug into the same schema) |
| Data quality inspection + cleaning (NaN/dropped/outlier/frozen/duplicate) | **Implemented & tested** |
| Training-readiness audit (coverage, balance, split leakage, CTC feasibility) | **Implemented & tested** |
| Keypoint adapters (MediaPipe Holistic, OpenPose → skeleton) | **Implemented & tested** |
| Adaptive graph refinement (CTR-GCN / 2s-AGCN style learnable adjacency) | **Implemented & tested** |
| Preference optimisation (Diffusion-DPO + naturalness proxies) | **Implemented & tested** |
| Unified multi-branch trainer (epochs, LR schedule, checkpoints) | **Implemented & tested** |
| Analysis/report stage with pass-gated metrics | **Implemented & tested** |
| Real sign-language data (How2Sign, PHOENIX-2014T) | **Not included** — synthetic corpus only (same on-disk schema) |
| Avatar rendering (NeRF / Gaussian Splatting / SMPL-X) | **Not implemented** (out of scope for this core) |

The stubs are deliberate: every heavy foundation model sits behind a small
interface (`TextEncoder`, `SpeechEncoder`) so it can be replaced without
touching the manifold or generator.

---

## Architecture

```
        speech ──▶ SpeechEncoder ┐                    (swappable: Whisper/wav2vec2)
                                 ├─▶ language feature ─▶ language projection ┐
        text/gloss ─▶ TextEncoder┘                                          │
                                                                            ▼
                                                        ┌────────── shared manifold ──────────┐
                                                        │   InfoNCE contrastive alignment      │
                                                        ▼                                      ▲
        3D pose seq ─▶ ST-GCN encoder ─▶ motion feature ─▶ motion projection ─────────────────┘
                                                        │
                                              language latent c
                                                        │
                                                        ▼
                                   Conditional diffusion (DDPM/DDIM) ─▶ generated 3D motion
```

- **Perception** (`signtranslator/skeleton`, `models/stgcn.py`): the upper body
  and both hands are a graph of 27 keypoints. ST-GCN performs spatial graph
  convolution using spatial-configuration partitioning (self / centripetal /
  centrifugal) plus temporal convolution, mapping a pose clip to a motion
  embedding.
- **Alignment** (`models/alignment.py`): projection heads map motion and
  language features onto a unit hypersphere; a symmetric InfoNCE loss with a
  learnable temperature aligns paired examples.
- **Generation** (`models/diffusion.py`, `models/denoiser.py`): a Gaussian
  diffusion model with a Transformer denoiser generates continuous 3D motion
  conditioned on the language latent.

The mathematics of each stage is documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
and [`docs/MATH.md`](docs/MATH.md).

---

## Installation

```bash
python -m pip install -r requirements.txt
# or, as a package (editable):
python -m pip install -e .
```

Only `torch` and `numpy` are required for the core. Optional real backends:

```bash
python -m pip install -e ".[foundation]"   # transformers, soundfile
```

## Quick start

Train the joint objective on the synthetic dataset (CPU-friendly):

```bash
python -m signtranslator.train --steps 300 --batch-size 16
```

Generate a motion clip from gloss tokens in Python:

```python
import torch
from signtranslator import ModelConfig, DiffusionConfig
from signtranslator.models import SignTranslator

model = SignTranslator(ModelConfig(), DiffusionConfig())
tokens = torch.randint(1, 4096, (2, 6))         # (batch, gloss length)
motion = model.generate(tokens, num_frames=64)  # (2, 3, 64, 27) = (N, xyz, T, joints)
```

Run the **full bidirectional** system (speech→gloss→sign, and sign→gloss):

```python
import torch
from signtranslator import ModelConfig, DiffusionConfig
from signtranslator.models import BidirectionalSignTranslator

model = BidirectionalSignTranslator(ModelConfig(), DiffusionConfig(),
                                    src_vocab=256, gloss_vocab=128, num_glosses=64)

# speech/text tokens -> gloss reordering -> 3D signing motion (with CFG)
src = torch.randint(3, 256, (2, 8))
out = model.translate_speech_to_sign(src, num_frames=64, guidance_scale=2.0)
out["gloss"]    # decoded gloss id sequences
out["motion"]   # (2, 3, 64, 27) generated motion

# sign -> gloss recognition (CTC greedy decode)
pose = torch.randn(2, 3, 64, 27)
model.recognize(pose)
```

## End-to-end pipeline: ingest → train → analyze

A single command generates an on-disk corpus, trains all branches jointly, and
prints a pass-gated analysis report:

```bash
python -m signtranslator.run --corpus-dir ./corpus --epochs 28 --lr 4e-3
```

Measured result on a held-out validation split:

```
Analysis report
================================================
  recognition_wer              0.0000  PASS
  planner_token_accuracy       0.9727  PASS
  recall_at_1                  0.9531  PASS
  recall_at_5                  1.0000
  generation_val_loss          0.1778  PASS
  cycle_consistency_wer        0.0000  PASS
  speech_wer                   0.0000  PASS
------------------------------------------------
  OVERALL (gated metrics): PASS
```

Every metric is **gated** — each measures that one branch works end to end:
audio→spoken tokens (speech WER), sign→gloss recognition (CTC WER),
spoken→gloss translation (token accuracy), motion↔gloss manifold retrieval
(recall@1), conditional generation (diffusion objective), and the strictest of
all, **cycle-consistency**: generate 3D motion from gloss, feed it back to the
recogniser, and require the gloss to come back. At 0.0000 WER, generated motion
is recognised *as accurately as ground-truth motion*.

### Training schedule (three phases)

Getting cycle-consistency to pass required more than tuning; see
[`docs/MATH.md`](docs/MATH.md) for the derivations.

1. **Joint** — all branches together. Discriminative branches converge in a few
   hundred steps.
2. **Generator fine-tune** — the diffusion generator needs far more updates than
   the rest, so it is trained alone (with its private conditioning encoder) for
   many cheap steps.
3. **Polish** — a short low-LR joint pass re-converges every branch together.

The generator additionally uses **x₀-prediction**, a **velocity loss**, pose
**standardisation**, and **high-noise timestep emphasis**. Without the last of
these the model learns only to denoise and never to synthesise from the
conditioning — which is precisely where sampling starts.

### Data ingestion

Corpora live on disk as `manifest.json` + `<split>.npz` (see
`signtranslator/data/corpus.py`), carrying pose, gloss/spoken concepts, and
acoustic features. `generate_corpus` writes a synthetic corpus whose spoken
tokens, gloss tokens, 3D motion, audio features, and CTC targets are mutually
consistent. `validate_corpus` checks the schema. Real corpora (How2Sign,
PHOENIX-2014T) export into the same schema and are read by `SignDataset`
without code changes.

### Data cleaning and readiness

```python
from signtranslator.data import inspect_pose, clean_pose, assess_corpus

print(inspect_pose(pose).summary())      # NaN/Inf, dropped keypoints, outliers,
                                         # dead joints, frozen frames, duplicates
clean, kept, rep = clean_pose(pose)      # interpolate gaps, clip spikes, drop
                                         # unrecoverable samples (auditable)
print(assess_corpus("./corpus").summary())
```

`assess_corpus` gates training on structural fitness — sample counts, per-class
coverage in **both** splits, class balance, **split leakage** (byte-identical
samples across train/val), CTC length feasibility, and normalisation sanity. The
`run.py` pipeline refuses to train on a corpus that fails these checks.

### Ingesting real pose estimators

```python
from signtranslator.data import mediapipe_holistic_adapter, clean_pose

adapter = mediapipe_holistic_adapter(conf_threshold=0.3)
res = adapter(body, right_hand, left_hand, body_conf=conf)  # -> 27-joint skeleton
pose = res.pose; pose[res.missing] = float("nan")           # low-confidence -> missing
pose, _, _ = clean_pose(pose.unsqueeze(0))                  # interpolated
```

### Preference optimisation (RLHF-style)

```python
from signtranslator.training import DiffusionDPO, naturalness_score, build_preference_pairs

pref, rej = build_preference_pairs(candidates, naturalness_score)  # or human ratings
dpo = DiffusionDPO(model.diffusion, beta=0.1)
dpo.step(optimizer, pref, rej, cond=model.gloss_memory(gloss))
```

Naturalness proxies are minimum-jerk smoothness and bone-length consistency;
swap in human ratings for true RLHF. A frozen reference policy keeps the tuned
model from drifting away from the supervised solution.

## Testing

```bash
python -m pytest      # 97 tests
```

The suite verifies (among other things): adjacency normalisation bounds the
graph-convolution spectral radius; the diffusion forward process matches its
analytic mean/variance; `predict_start_from_noise` exactly inverts `q_sample`;
InfoNCE matches a manual cross-entropy and rewards correct pairing; and the full
model **reduces its loss** when trained on structured synthetic data.

## Repository layout

```
signtranslator/
  config.py             typed dataclass configs
  skeleton/graph.py     skeleton graph + partitioned normalised adjacency
  models/
    stgcn.py            ST-GCN motion encoder (clip + per-frame outputs)
    encoders.py         text/speech encoder interfaces + stubs
    alignment.py        projection heads + symmetric InfoNCE
    denoiser.py         Transformer noise predictor + cross-modal denoiser
    diffusion.py        Gaussian diffusion (DDPM/DDIM) math
    guided_diffusion.py classifier-free guidance (condition dropout + guided sampling)
    recognition.py      CTC continuous sign recognition (sign→gloss)
    speech.py           acoustic front-end (audio→spoken tokens, CTC)
    planner.py          seq2seq semantic planner (English→gloss)
    pipeline.py         SignTranslator + BidirectionalSignTranslator
  data/
    synthetic.py        in-memory paired (motion, gloss) dataset
    preprocess.py       keypoint normalisation + augmentation
    corpus.py           on-disk corpus: schema, generator, ingestion, collate
    quality.py          defect inspection + cleaning pipeline
    readiness.py        training-readiness audit (gated)
    adapters.py         MediaPipe/OpenPose → skeleton keypoint adapters
  eval/metrics.py       recall@k, MPJPE, WER, top-1 accuracy
  training/
    trainer.py          unified multi-branch trainer + generator fine-tune
    preference.py       Diffusion-DPO + naturalness proxies
  analysis/report.py    pass-gated evaluation report
  train.py              single-branch training loop / CLI
  run.py                end-to-end ingest -> train -> analyze CLI
tests/                  pytest suite (math, shapes, gradients, learning, invariances,
                        ingestion, training, analysis)
docs/                   architecture & math notes
```

## References

- Yan et al., *Spatial Temporal Graph Convolutional Networks*, AAAI 2018.
- Radford et al., *Learning Transferable Visual Models From Natural Language
  Supervision* (CLIP), 2021.
- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
- Nichol & Dhariwal, *Improved DDPM*, 2021.
- Song et al., *Denoising Diffusion Implicit Models* (DDIM), 2021.
- Tevet et al., *Human Motion Diffusion Model* (MDM), 2023.

## License

Apache-2.0.
