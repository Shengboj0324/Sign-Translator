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
| Real sign-language data (How2Sign, PHOENIX-2014T) | **Not included** — synthetic dataset only |
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

## Testing

```bash
python -m pytest
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
    planner.py          seq2seq semantic planner (English→gloss)
    pipeline.py         SignTranslator + BidirectionalSignTranslator
  data/
    synthetic.py        paired (motion, gloss) synthetic dataset
    preprocess.py       keypoint normalisation + augmentation
  eval/metrics.py       recall@k, MPJPE, WER, top-1 accuracy
  train.py              training loop / CLI
tests/                  pytest suite (math, shapes, gradients, learning, invariances)
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
