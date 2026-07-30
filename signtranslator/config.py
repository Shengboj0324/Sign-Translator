"""Typed configuration objects.

Using dataclasses keeps configuration explicit and serializable while avoiding a
heavyweight config framework. All dimensions are validated at construction time so
that shape mismatches surface early rather than deep inside a forward pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar


CONFIG_SCHEMA_VERSION = 1
_ConfigT = TypeVar("_ConfigT", bound="SerializableConfig")


class SerializableConfig:
    """Strict, versioned serialization shared by every executable config."""

    schema_version: ClassVar[int] = CONFIG_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_type": type(self).__name__,
            "schema_version": self.schema_version,
            "values": asdict(self),
        }

    @classmethod
    def from_dict(cls: Type[_ConfigT], payload: Dict[str, Any]) -> _ConfigT:
        if not isinstance(payload, dict):
            raise TypeError("configuration payload must be a dictionary")
        if payload.get("config_type") != cls.__name__:
            raise ValueError(
                f"expected config_type={cls.__name__!r}, got "
                f"{payload.get('config_type')!r}")
        if payload.get("schema_version") != cls.schema_version:
            raise ValueError(
                f"unsupported {cls.__name__} schema version "
                f"{payload.get('schema_version')!r}; expected {cls.schema_version}")
        values = payload.get("values")
        if not isinstance(values, dict):
            raise TypeError("configuration 'values' must be a dictionary")
        allowed = {item.name for item in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
        values = dict(values)
        if cls is ModelConfig and "stgcn_channels" in values:
            values["stgcn_channels"] = tuple(values["stgcn_channels"])
        return cls(**values)


@dataclass
class ModelConfig(SerializableConfig):
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
        if self.in_channels < 2:
            raise ValueError("need at least 2 spatial channels")
        if self.num_joints <= 0 or self.num_frames <= 0:
            raise ValueError("num_joints and num_frames must be positive")
        if not self.stgcn_channels or any(c <= 0 for c in self.stgcn_channels):
            raise ValueError("stgcn_channels must contain positive widths")
        if self.stgcn_temporal_kernel <= 0 or self.stgcn_temporal_kernel % 2 != 1:
            raise ValueError("temporal kernel must be positive and odd")
        if self.text_embed_dim <= 0 or self.text_layers <= 0 or self.text_heads <= 0:
            raise ValueError("text dimensions and layer counts must be positive")
        if self.text_embed_dim % self.text_heads != 0:
            raise ValueError("text_embed_dim must be divisible by text_heads")
        if self.latent_dim <= 0 or self.vocab_size <= 0 or self.speech_input_dim <= 0:
            raise ValueError("latent, vocabulary, and speech dimensions must be positive")

    @property
    def motion_feature_dim(self) -> int:
        """Dimension of the ST-GCN encoder output before projection."""
        return self.stgcn_channels[-1]

@dataclass
class DiffusionConfig(SerializableConfig):
    """Gaussian-diffusion schedule configuration for motion generation."""

    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "cosine"      # {"linear", "cosine"}
    denoiser_dim: int = 256
    denoiser_layers: int = 4
    denoiser_heads: int = 4
    # "x0" prediction + a velocity term is the higher-fidelity setting for
    # motion (cf. MDM); "eps" is the classic DDPM objective.
    parameterization: str = "x0"
    velocity_weight: float = 1.0
    # Emphasise high-noise timesteps during training so the model learns to
    # synthesise from the conditioning (which is what sampling from t=T needs),
    # not merely to denoise a partially-visible signal.
    high_t_frac: float = 0.65
    high_t_start: float = 0.85

    def __post_init__(self) -> None:
        if self.num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
        if self.schedule not in {"linear", "cosine"}:
            raise ValueError("schedule must be 'linear' or 'cosine'")
        if self.parameterization not in {"eps", "x0"}:
            raise ValueError("parameterization must be 'eps' or 'x0'")
        if self.velocity_weight < 0.0:
            raise ValueError("velocity_weight must be non-negative")
        if self.denoiser_dim <= 0 or self.denoiser_layers <= 0 or self.denoiser_heads <= 0:
            raise ValueError("denoiser dimensions and layer counts must be positive")
        if self.denoiser_dim % self.denoiser_heads != 0:
            raise ValueError("denoiser_dim must be divisible by denoiser_heads")
        if not 0.0 <= self.high_t_frac <= 1.0:
            raise ValueError("high_t_frac must be in [0, 1]")
        if not 0.0 <= self.high_t_start <= 1.0:
            raise ValueError("high_t_start must be in [0, 1]")


@dataclass
class TrainConfig(SerializableConfig):
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

    def __post_init__(self) -> None:
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("lr must be positive and weight_decay non-negative")
        if self.batch_size <= 0 or self.max_steps <= 0:
            raise ValueError("batch_size and max_steps must be positive")
        if self.grad_clip <= 0 or self.contrastive_temperature <= 0:
            raise ValueError("grad_clip and contrastive_temperature must be positive")


@dataclass
class TrainerConfig(SerializableConfig):
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
        "speech": 1.0,
    })
    val_every: int = 1
    seed: int = 0
    device: str = "cpu"
    ckpt_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.lr <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("lr and grad_clip must be positive; weight_decay non-negative")
        if not 0.0 <= self.warmup_frac < 1.0:
            raise ValueError("warmup_frac must be in [0, 1)")
        if not 0.0 <= self.min_lr_frac <= 1.0:
            raise ValueError("min_lr_frac must be in [0, 1]")
        if self.val_every <= 0:
            raise ValueError("val_every must be positive")
        if not self.loss_weights or any(v < 0 for v in self.loss_weights.values()):
            raise ValueError("loss_weights must be non-empty and non-negative")
