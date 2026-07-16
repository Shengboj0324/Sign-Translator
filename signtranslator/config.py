"""Typed configuration objects.

Using dataclasses keeps configuration explicit and serializable while avoiding a
heavyweight config framework. All dimensions are validated at construction time so
that shape mismatches surface early rather than deep inside a forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """Architecture hyper-parameters for the shared-manifold model."""

    # Skeleton / motion representation
    num_joints: int = 27          # number of body/hand keypoints in the graph
    in_channels: int = 3          # (x, y, z) per joint
    num_frames: int = 64          # temporal length of a motion clip

    # ST-GCN encoder
    stgcn_channels: tuple = (64, 128, 256)
    stgcn_temporal_kernel: int = 9

    # Language / speech encoders
    vocab_size: int = 4096        # gloss / token vocabulary for the stub text encoder
    text_embed_dim: int = 256
    text_layers: int = 4
    text_heads: int = 4
    speech_input_dim: int = 80    # e.g. log-mel bins for the stub speech encoder

    # Shared contrastive manifold
    latent_dim: int = 256

    def __post_init__(self) -> None:
        assert self.in_channels >= 2, "need at least 2 spatial channels"
        assert self.num_joints > 0 and self.num_frames > 0
        assert self.stgcn_temporal_kernel % 2 == 1, "temporal kernel must be odd"
        assert self.latent_dim > 0

    @property
    def motion_feature_dim(self) -> int:
        """Dimension of the ST-GCN encoder output before projection."""
        return self.stgcn_channels[-1]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiffusionConfig:
    """Gaussian-diffusion schedule configuration for motion generation."""

    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "cosine"      # {"linear", "cosine"}
    denoiser_dim: int = 256
    denoiser_layers: int = 4
    denoiser_heads: int = 4

    def __post_init__(self) -> None:
        assert self.num_timesteps > 0
        assert 0.0 < self.beta_start < self.beta_end < 1.0
        assert self.schedule in {"linear", "cosine"}


@dataclass
class TrainConfig:
    """Optimization / loss-weighting configuration."""

    lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    max_steps: int = 1000
    grad_clip: float = 1.0

    # Loss weights for the joint objective.
    w_contrastive: float = 1.0
    w_diffusion: float = 1.0

    contrastive_temperature: float = 0.07
    seed: int = 0
    device: str = "cpu"

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    """Configuration for the unified multi-branch :class:`Trainer`."""

    epochs: int = 12
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1        # fraction of total steps for LR warmup
    min_lr_frac: float = 0.05       # cosine floor as a fraction of peak lr
    grad_clip: float = 1.0
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "generation": 1.0, "alignment": 0.5, "planner": 1.0, "recognition": 1.0,
    })
    val_every: int = 1
    seed: int = 0
    device: str = "cpu"
    ckpt_path: Optional[str] = None

    def __post_init__(self) -> None:
        assert self.epochs > 0 and self.batch_size > 0
        assert 0.0 <= self.warmup_frac < 1.0
        assert 0.0 <= self.min_lr_frac <= 1.0
