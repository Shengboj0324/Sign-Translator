# Architecture

## Design goals

1. **Novel core first.** Implement and *verify* the shared motion-language
   manifold and diffusion motion generator — the genuinely new research
   direction — rather than a shallow wrapper around many unverified models.
2. **Swappable foundation models.** Speech and language backends sit behind
   narrow interfaces so Whisper / wav2vec2 / an LLM planner can replace the
   stubs without changing the manifold or generator.
3. **Buildable and testable anywhere.** No GPU, no multi-GB downloads, and no
   external data are required to build, train (synthetic), and test the core.

## Data flow

A training example is a pair `(pose, gloss tokens)`:

- `pose`: `(C=3, T, V=27)` — 3D coordinates of 27 upper-body/hand joints over
  `T` frames.
- `tokens`: a gloss/word id sequence.

```
pose  ─▶ STGCNEncoder ───────────▶ motion feature  f^m ∈ R^{256}
tokens ─▶ TextEncoder (stub/LLM) ─▶ language feature f^l ∈ R^{256}

f^m, f^l ─▶ ContrastiveAligner ─▶ z^m, z^l on unit sphere,  L_InfoNCE
                                     │
                              language latent z^l
                                     ▼
pose (x_0) ─▶ GaussianMotionDiffusion(denoiser, cond=z^l) ─▶ L_diff
```

At inference (`generate`), only the text path runs: tokens → language latent →
diffusion sampling → `(N, 3, T, 27)` motion.

## Component responsibilities

### `skeleton/graph.py` — `SkeletonGraph`
Builds the 27-joint tree, computes hop distances by BFS, partitions edges into
self/centripetal/centrifugal subsets, and symmetrically normalises each. Output:
a `(3, 27, 27)` float32 adjacency tensor. The adjacency is a *fixed* buffer, not
a learned parameter.

### `models/stgcn.py` — `STGCNEncoder`
Data batch-norm → stacked `STGCNBlock`s (graph conv + temporal conv + residual)
→ global average pool over time and joints. Produces one motion embedding per
clip. The graph convolution realises the `K` partition weight matrices with a
single `1×1` conv and an `einsum` contraction against the adjacency.

### `models/encoders.py` — `TextEncoder` / `SpeechEncoder`
Abstract interfaces returning `(N, embed_dim)`. Default implementations are
small Transformers with sinusoidal position encoding and masked mean pooling.
Replace by subclassing and returning the same shape.

### `models/alignment.py` — `ContrastiveAligner`
Two `ProjectionHead`s (MLP + L2 norm) and a learnable log-temperature. Computes
the symmetric InfoNCE loss and exposes the projected embeddings.

### `models/denoiser.py` — `MotionDenoiser`
Transformer over per-frame joint-feature tokens, conditioned additively on a
sinusoidal timestep embedding and the language latent. Output projection is
zero-initialised so the model begins as the zero-noise predictor.

### `models/diffusion.py` — `GaussianMotionDiffusion`
Owns the schedule buffers and all forward/reverse math: `q_sample`,
`predict_start_from_noise`, `q_posterior_mean_variance`, `p_losses`, ancestral
`sample`, and `ddim_sample`.

### `models/pipeline.py` — `SignTranslator`
Wires the above into one module, exposes the joint training `forward` and the
`generate` inference path.

## Extending toward the full system

- **Real perception:** feed MediaPipe Holistic / MMPose / SMPL-X keypoints into
  `STGCNEncoder` (adjust `num_joints` and the edge list in `skeleton/graph.py`).
- **Real language/speech:** implement `TextEncoder` / `SpeechEncoder` wrapping
  Whisper, wav2vec2, SeamlessM4T, or an LLM semantic planner (English → gloss
  reordering) and pass it to `SignTranslator(text_encoder=...)`.
- **Richer generation:** replace the pooled-latent conditioning with
  cross-attention to the full language token sequence; add a variational or
  latent-diffusion stage; attach an SMPL-X / Gaussian-splatting renderer to turn
  generated joint trajectories into a photorealistic avatar.
- **Preference optimisation:** add an RLHF / DPO stage on top of the generator to
  improve naturalness and grammatical faithfulness.

These are all additive: none require changing the manifold or diffusion core.
