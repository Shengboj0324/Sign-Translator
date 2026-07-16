"""Sign-Translator: a shared motion-language manifold for sign-language translation.

The package implements the *novel core* of a bidirectional sign-language
translation system:

* ``skeleton``  - biomechanical skeleton graph definition.
* ``models.stgcn``     - spatio-temporal graph conv encoder (pose -> motion embedding).
* ``models.encoders``  - swappable text / speech encoders (light stubs + real backends).
* ``models.alignment`` - CLIP-style contrastive alignment into a shared latent manifold.
* ``models.diffusion`` - Gaussian DDPM for conditional 3D motion generation.
* ``models.pipeline``  - end-to-end model tying the pieces together.

Heavy foundation models (Whisper, wav2vec2, LLM planners, NeRF/Gaussian avatars)
are intentionally kept behind interfaces so the core builds, trains on synthetic
data, and is fully unit-tested without large downloads or a GPU.
"""

from .config import ModelConfig, DiffusionConfig, TrainConfig

__all__ = ["ModelConfig", "DiffusionConfig", "TrainConfig", "__version__"]
__version__ = "0.1.0"
