# 01 — Project Overview

## 1. Problem statement

Sign-Translator aims to support bidirectional translation between spoken or
written language and continuous sign-language motion. A complete forward system
would accept raw speech or text, recover its meaning, express that meaning using
the grammar of a specified sign language, generate coordinated manual and
non-manual motion, and render the result through an avatar. A reverse system
would recognize signing motion and return a linguistic representation.

This problem is not equivalent to replacing words with gestures. Sign languages
encode meaning through:

- handshape, orientation, location, and movement;
- facial and mouth activity;
- head and torso motion;
- timing and simultaneity;
- spatial loci and coreference;
- discourse structure and role shift;
- language- and dialect-specific grammar.

Any system that omits these channels may generate movement while failing to
communicate the intended proposition.

## 2. Project thesis

The codebase is built around three research commitments:

1. **Linguistic structure should be explicit.** A temporal sign representation
   should describe manual events, non-manual scope, spatial reference, and
   grammatical relations.
2. **Language and motion should share a measurable representation.** Contrastive
   alignment places paired linguistic and motion examples in a common latent
   space.
3. **Motion generation should be constrained and testable.** Diffusion,
   kinematics, temporal modeling, and rendering should be evaluated separately
   rather than hidden behind one aggregate loss.

These principles are valuable. The current limitation is not the absence of
ideas; it is the absence of one real-data path connecting all required ideas.

## 3. Intended end-to-end capability

The intended forward direction is:

```text
raw audio or text
    -> speech/language representation
    -> semantic sign plan
    -> grammatical temporal representation
    -> body, hand, and facial motion
    -> avatar rendering
```

The intended reverse direction is:

```text
video or fitted motion
    -> pose and non-manual features
    -> sign recognition
    -> gloss or structured linguistic output
    -> spoken/written language
```

The current executable system implements a smaller path:

```text
synthetic acoustic features
    -> compact CTC recognizer
    -> token-to-gloss Transformer
    -> Cartesian 27-joint diffusion
```

and:

```text
Cartesian 27-joint motion
    -> ST-GCN
    -> CTC gloss recognition
```

## 4. What is genuinely present

The repository contains substantial implementations for:

- log-Mel speech features, CTC decoding, alignment, calibration, and revision;
- constrained semantic-plan serialization and decoding;
- temporal sign graphs and interval relations;
- rotation geometry, toy SMPL-X-shaped forward kinematics, and reprojection;
- hand and body graph operations;
- motion tokenization, Transformer decoding, and streaming interpolation;
- diffusion schedules, parameterizations, guidance, and constraints;
- rendering mathematics for skinning, Gaussian compositing, and NeRF rays;
- facial/non-manual channels and scope losses;
- data provenance, consent gates, grouped splitting, and quality mathematics;
- self-supervised objectives and hard-negative tests;
- statistical evaluation contracts;
- latency, backpressure, quantization, and runtime-control primitives.

The breadth is real, and many primitives are tested carefully. However, the
presence of these packages does not mean the active model trains or deploys all
of them.

## 5. Current positioning

The correct description is:

> A modular, mathematically explicit research core for studying sign-language
> translation components, supported by synthetic data and extensive adversarial
> tests.

The following descriptions would currently be inaccurate:

- a trained sign-language translator;
- a full speech-to-avatar pipeline;
- a validated ASL or other sign-language model;
- a production-ready accessibility system;
- a system proven understandable by Deaf or fluent signers.

## 6. Success criteria for the next stage

The next stage succeeds only when the team can:

1. name the exact target sign language, dialect, task, and intended use;
2. ingest licensed real samples without replacing them with synthetic data;
3. produce one versioned batch that includes valid motion, language, timing,
   confidence, and provenance fields;
4. overfit a tiny real subset for the right reasons;
5. generalize to unseen signers or source recordings;
6. render outputs that qualified signers can evaluate;
7. reproduce the result from a complete checkpoint and data manifest.

