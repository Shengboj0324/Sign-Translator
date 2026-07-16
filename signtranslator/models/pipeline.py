"""End-to-end model: perception + alignment + conditional motion generation.

``SignTranslator`` ties the components into one ``nn.Module`` exposing the two
capabilities the project targets:

Recognition / understanding direction
    pose sequence --ST-GCN--> motion feature --proj--> shared manifold
    (aligned against language via contrastive loss).

Generation direction (speech/text -> sign)
    tokens --text encoder--> language feature --proj--> manifold latent c,
    c conditions the diffusion denoiser which generates a 3D motion clip.

The joint training loss is
    L = w_contrastive * L_InfoNCE + w_diffusion * L_DDPM .

The two losses cooperate: contrastive alignment shapes a semantically
meaningful conditioning latent, and the diffusion loss teaches the decoder to
realise that latent as continuous motion.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import ModelConfig, DiffusionConfig
from ..skeleton.graph import SkeletonGraph
from .stgcn import STGCNEncoder
from .encoders import StubTextEncoder, StubSpeechEncoder, TextEncoder
from .alignment import ContrastiveAligner
from .denoiser import MotionDenoiser
from .diffusion import GaussianMotionDiffusion


class SignTranslator(nn.Module):
    def __init__(self, model_cfg: ModelConfig, diff_cfg: DiffusionConfig,
                 graph: Optional[SkeletonGraph] = None,
                 text_encoder: Optional[TextEncoder] = None) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.diff_cfg = diff_cfg
        self.graph = graph or SkeletonGraph(num_nodes=model_cfg.num_joints)
        if self.graph.num_nodes != model_cfg.num_joints:
            raise ValueError("graph joints != config joints")
        adjacency = self.graph.adjacency()

        # Perception: pose -> motion feature.
        self.motion_encoder = STGCNEncoder(
            in_channels=model_cfg.in_channels, adjacency=adjacency,
            channels=model_cfg.stgcn_channels,
            temporal_kernel=model_cfg.stgcn_temporal_kernel,
            num_joints=model_cfg.num_joints,
        )

        # Language: tokens -> language feature (swappable).
        self.text_encoder = text_encoder or StubTextEncoder(
            vocab_size=model_cfg.vocab_size, embed_dim=model_cfg.text_embed_dim,
            num_layers=model_cfg.text_layers, num_heads=model_cfg.text_heads,
        )
        self.speech_encoder = StubSpeechEncoder(
            input_dim=model_cfg.speech_input_dim, embed_dim=model_cfg.text_embed_dim,
            num_layers=model_cfg.text_layers, num_heads=model_cfg.text_heads,
        )

        # Shared contrastive manifold.
        self.aligner = ContrastiveAligner(
            motion_dim=self.motion_encoder.out_dim,
            language_dim=self.text_encoder.embed_dim,
            latent_dim=model_cfg.latent_dim,
        )

        # Conditional motion generator.
        denoiser = MotionDenoiser(
            num_joints=model_cfg.num_joints, in_channels=model_cfg.in_channels,
            cond_dim=model_cfg.latent_dim, hidden_dim=diff_cfg.denoiser_dim,
            num_layers=diff_cfg.denoiser_layers, num_heads=diff_cfg.denoiser_heads,
            max_frames=max(model_cfg.num_frames, 128),
        )
        self.diffusion = GaussianMotionDiffusion(
            denoiser, num_timesteps=diff_cfg.num_timesteps, schedule=diff_cfg.schedule,
            beta_start=diff_cfg.beta_start, beta_end=diff_cfg.beta_end,
        )

    # -- feature extraction -------------------------------------------------
    def encode_motion(self, pose: torch.Tensor) -> torch.Tensor:
        return self.motion_encoder(pose)

    def encode_text(self, tokens: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.text_encoder(tokens, mask)

    # -- training -----------------------------------------------------------
    def forward(self, pose: torch.Tensor, tokens: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None,
                w_contrastive: float = 1.0, w_diffusion: float = 1.0) -> dict:
        """Joint contrastive + diffusion objective on a paired batch.

        Args:
            pose:   (N, C, T, V) ground-truth motion clips.
            tokens: (N, L) gloss/word token ids paired with ``pose``.
        """
        motion_feat = self.encode_motion(pose)
        language_feat = self.encode_text(tokens, text_mask)

        align = self.aligner(motion_feat, language_feat)
        # Condition generation on the (detached-graph-free) language latent so the
        # decoder learns to realise language semantics as motion.
        cond = align["z_language"]
        diff_loss = self.diffusion(pose, cond=cond)

        total = w_contrastive * align["loss"] + w_diffusion * diff_loss
        return {
            "loss": total,
            "contrastive_loss": align["loss"],
            "diffusion_loss": diff_loss,
            "logits": align["logits"],
        }

    # -- inference: text -> motion -----------------------------------------
    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, num_frames: Optional[int] = None,
                 text_mask: Optional[torch.Tensor] = None,
                 use_ddim: bool = True, ddim_steps: int = 50) -> torch.Tensor:
        """Generate 3D motion clips conditioned on text/gloss tokens."""
        self.eval()
        language_feat = self.encode_text(tokens, text_mask)
        cond = self.aligner.language_head(language_feat)
        n = tokens.shape[0]
        T = num_frames or self.model_cfg.num_frames
        shape = (n, self.model_cfg.in_channels, T, self.model_cfg.num_joints)
        device = next(self.parameters()).device
        if use_ddim:
            return self.diffusion.ddim_sample(shape, cond=cond, num_steps=ddim_steps, device=device)
        return self.diffusion.sample(shape, cond=cond, device=device)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
