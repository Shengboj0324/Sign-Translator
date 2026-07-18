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
import torch.nn.functional as F

from ..config import ModelConfig, DiffusionConfig
from ..skeleton.graph import SkeletonGraph
from .stgcn import STGCNEncoder
from .encoders import StubTextEncoder, StubSpeechEncoder, TextEncoder
from .alignment import ContrastiveAligner
from .denoiser import MotionDenoiser, CrossModalDenoiser
from .diffusion import GaussianMotionDiffusion
from .guided_diffusion import GuidedMotionDiffusion
from .recognition import SignRecognizer
from .speech import SpeechRecognizer
from .planner import GlossPlanner, BOS, EOS
from ..data.corpus import CONTENT_OFFSET


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


class BidirectionalSignTranslator(nn.Module):
    """Full bidirectional system tying together every branch.

    Directions:
      * **speech/text -> sign**: ``GlossPlanner`` reorders spoken-language tokens
        into gloss tokens; a ``StubTextEncoder`` produces per-token memory; a
        cross-modal ``GuidedMotionDiffusion`` generates 3D motion conditioned on
        that memory with classifier-free guidance.
      * **sign -> gloss**: a ``SignRecognizer`` (ST-GCN + CTC) reads a pose clip.

    Losses are exposed separately so a trainer can weight them; ``training_step``
    returns their sum for convenience.
    """

    def __init__(self, model_cfg: ModelConfig, diff_cfg: DiffusionConfig,
                 src_vocab: int = 256, gloss_vocab: int = 128, num_glosses: int = 64,
                 cond_drop_prob: float = 0.1, planner_layers: Optional[int] = None,
                 graph: Optional[SkeletonGraph] = None) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.diff_cfg = diff_cfg
        self.gloss_vocab = gloss_vocab
        self.num_glosses = num_glosses
        self.graph = graph or SkeletonGraph(num_nodes=model_cfg.num_joints)
        adjacency = self.graph.adjacency()

        # speech/text -> gloss (the planner can be deepened independently since
        # sequence reordering benefits from more decoder depth than pooling tasks)
        self.planner = GlossPlanner(src_vocab=src_vocab, tgt_vocab=gloss_vocab,
                                    d_model=model_cfg.text_embed_dim,
                                    nhead=model_cfg.text_heads,
                                    num_layers=planner_layers or model_cfg.text_layers)

        # Two task-specific gloss encoders.
        #
        # `gloss_encoder` feeds the contrastive manifold (pooled sentence-level
        # semantics). `cond_encoder` feeds the generator's cross-attention
        # (per-token features driving motion synthesis). Sharing one encoder for
        # both causes destructive interference: fine-tuning it for generation
        # collapses retrieval, while freezing it starves the generator. Keeping
        # them separate lets each objective converge without harming the other.
        self.gloss_encoder = StubTextEncoder(
            vocab_size=gloss_vocab, embed_dim=model_cfg.text_embed_dim,
            num_layers=model_cfg.text_layers, num_heads=model_cfg.text_heads,
        )
        self.cond_encoder = StubTextEncoder(
            vocab_size=gloss_vocab, embed_dim=model_cfg.text_embed_dim,
            num_layers=model_cfg.text_layers, num_heads=model_cfg.text_heads,
        )

        # conditional motion generator (cross-modal + CFG)
        denoiser = CrossModalDenoiser(
            num_joints=model_cfg.num_joints, in_channels=model_cfg.in_channels,
            context_dim=model_cfg.text_embed_dim, hidden_dim=diff_cfg.denoiser_dim,
            num_layers=diff_cfg.denoiser_layers, num_heads=diff_cfg.denoiser_heads,
            max_frames=max(model_cfg.num_frames, 128),
        )
        self.diffusion = GuidedMotionDiffusion(
            denoiser, num_timesteps=diff_cfg.num_timesteps, schedule=diff_cfg.schedule,
            beta_start=diff_cfg.beta_start, beta_end=diff_cfg.beta_end,
            cond_drop_prob=cond_drop_prob,
            parameterization=diff_cfg.parameterization,
            velocity_weight=diff_cfg.velocity_weight,
            high_t_frac=diff_cfg.high_t_frac,
            high_t_start=diff_cfg.high_t_start,
        )

        # sign -> gloss recognition
        recog_encoder = STGCNEncoder(
            in_channels=model_cfg.in_channels, adjacency=adjacency,
            channels=model_cfg.stgcn_channels,
            temporal_kernel=model_cfg.stgcn_temporal_kernel,
            num_joints=model_cfg.num_joints,
        )
        self.recognizer = SignRecognizer(recog_encoder, num_glosses=num_glosses)

        # Shared contrastive manifold: the *same* ST-GCN encoder that feeds CTC
        # recognition also produces a clip embedding aligned with the gloss
        # encoder's pooled embedding. This ties the "novel core" manifold into
        # the bidirectional system with full parameter sharing.
        self.aligner = ContrastiveAligner(
            motion_dim=recog_encoder.out_dim,
            language_dim=model_cfg.text_embed_dim,
            latent_dim=model_cfg.latent_dim,
        )

        # Acoustic front-end: audio features -> spoken tokens (CTC). Stands in
        # for a speech foundation model (Whisper / wav2vec 2.0); swap by feeding
        # that model's hidden states as `speech` features of matching width.
        self.speech_recognizer = SpeechRecognizer(
            input_dim=model_cfg.speech_input_dim,
            num_tokens=num_glosses,
            hidden_dim=model_cfg.text_embed_dim,
            num_layers=model_cfg.text_layers,
            num_heads=model_cfg.text_heads,
        )

    # -- conditioning helper ------------------------------------------------
    def gloss_memory(self, gloss_tokens: torch.Tensor):
        """Per-token conditioning memory for the generator's cross-attention."""
        return self.cond_encoder.encode_sequence(gloss_tokens)

    # -- per-branch losses --------------------------------------------------
    def planner_loss(self, src: torch.Tensor, gloss: torch.Tensor) -> torch.Tensor:
        return self.planner.loss(src, gloss)

    def generation_loss(self, pose: torch.Tensor, gloss_tokens: torch.Tensor) -> torch.Tensor:
        cond = self.gloss_memory(gloss_tokens)
        return self.diffusion(pose, cond=cond)

    def recognition_loss(self, pose: torch.Tensor, targets: torch.Tensor,
                         target_lengths: torch.Tensor) -> torch.Tensor:
        return self.recognizer.loss(pose, targets, target_lengths)

    def speech_loss(self, speech: torch.Tensor, targets: torch.Tensor,
                    target_lengths: torch.Tensor) -> torch.Tensor:
        """CTC loss for the acoustic branch (audio -> spoken tokens)."""
        return self.speech_recognizer.loss(speech, targets, target_lengths)

    @torch.no_grad()
    def recognize_speech(self, speech: torch.Tensor):
        """Decode audio features to spoken token ids (1..K)."""
        return self.speech_recognizer.decode(speech)

    def alignment_loss(self, pose: torch.Tensor, gloss_tokens: torch.Tensor) -> torch.Tensor:
        motion_feat = self.recognizer.encoder(pose)                 # (N, D) pooled
        lang_feat = self.gloss_encoder(gloss_tokens)                # (N, D) pooled
        return self.aligner(motion_feat, lang_feat)["loss"]

    def _encode_pose_shared(self, pose: torch.Tensor):
        """Single ST-GCN pass reused by recognition (CTC) and alignment.

        The clip embedding equals the time-mean of the per-frame features
        (both are joint+time global averages), so recognition log-probs and the
        pooled motion embedding are derived from one forward pass.
        """
        seq = self.recognizer.encoder(pose, return_sequence=True)   # (N, T, D)
        pooled = seq.mean(dim=1)                                    # (N, D) clip embedding
        logprobs = F.log_softmax(self.recognizer.classifier(seq), dim=-1)
        return logprobs, pooled

    @torch.no_grad()
    def embed_motion(self, pose: torch.Tensor) -> torch.Tensor:
        """Unit-norm motion embedding on the shared manifold (for retrieval)."""
        return self.aligner.motion_head(self.recognizer.encoder(pose))

    @torch.no_grad()
    def embed_gloss(self, gloss_tokens: torch.Tensor) -> torch.Tensor:
        """Unit-norm gloss embedding on the shared manifold (for retrieval)."""
        return self.aligner.language_head(self.gloss_encoder(gloss_tokens))

    def training_step(self, batch: dict, weights: Optional[dict] = None) -> dict:
        """Compute all applicable branch losses from a batch dict.

        Expected keys (any subset): ``pose``, ``gloss_tokens`` (generation +
        alignment), ``src``+``gloss_seq`` (planner), ``ctc_targets``+
        ``ctc_lengths`` (recognition). ``weights`` optionally scales each branch
        in the returned ``total`` (default weight 1.0).
        """
        w = weights or {}
        losses = {}
        pose = batch.get("pose")

        # Share one ST-GCN pass across recognition + alignment when both apply.
        need_align = pose is not None and "gloss_tokens" in batch
        need_recog = pose is not None and "ctc_targets" in batch
        logprobs = pooled = None
        if need_align or need_recog:
            logprobs, pooled = self._encode_pose_shared(pose)

        if pose is not None and "gloss_tokens" in batch:
            losses["generation"] = self.generation_loss(pose, batch["gloss_tokens"])
            lang_feat = self.gloss_encoder(batch["gloss_tokens"])
            losses["alignment"] = self.aligner(pooled, lang_feat)["loss"]
        if "src" in batch and "gloss_seq" in batch:
            losses["planner"] = self.planner_loss(batch["src"], batch["gloss_seq"])
        if need_recog:
            n, t, _ = logprobs.shape
            input_lengths = torch.full((n,), t, dtype=torch.long, device=logprobs.device)
            losses["recognition"] = self.recognizer.ctc(
                logprobs.permute(1, 0, 2), batch["ctc_targets"],
                input_lengths, batch["ctc_lengths"])
        if "speech" in batch and "speech_ctc_targets" in batch:
            losses["speech"] = self.speech_loss(
                batch["speech"], batch["speech_ctc_targets"],
                batch["speech_ctc_lengths"])

        losses["total"] = sum(w.get(k, 1.0) * v for k, v in losses.items())
        return losses

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def generate_from_gloss(self, gloss_tokens: torch.Tensor, num_frames: Optional[int] = None,
                            guidance_scale: float = 2.0, ddim_steps: int = 50) -> torch.Tensor:
        self.eval()
        cond = self.gloss_memory(gloss_tokens)
        n = gloss_tokens.shape[0]
        T = num_frames or self.model_cfg.num_frames
        shape = (n, self.model_cfg.in_channels, T, self.model_cfg.num_joints)
        device = next(self.parameters()).device
        return self.diffusion.ddim_sample(shape, cond=cond, num_steps=ddim_steps,
                                          guidance_scale=guidance_scale, device=device)

    @torch.no_grad()
    def translate_speech_to_sign(self, src_tokens: torch.Tensor, num_frames: Optional[int] = None,
                                 guidance_scale: float = 2.0, ddim_steps: int = 50,
                                 max_gloss_len: int = 16) -> dict:
        """End-to-end: spoken-language tokens -> gloss -> 3D signing motion."""
        self.eval()
        device = next(self.parameters()).device
        gloss_lists = self.planner.greedy_decode(src_tokens, max_len=max_gloss_len)
        # Pad decoded gloss id lists into a batch tensor for the encoder.
        max_len = max((len(g) for g in gloss_lists), default=1) or 1
        gloss = torch.zeros(len(gloss_lists), max_len, dtype=torch.long, device=device)
        for i, g in enumerate(gloss_lists):
            if g:
                gloss[i, :len(g)] = torch.tensor(g, dtype=torch.long, device=device)
        motion = self.generate_from_gloss(gloss, num_frames=num_frames,
                                          guidance_scale=guidance_scale, ddim_steps=ddim_steps)
        return {"gloss": gloss_lists, "motion": motion}

    @torch.no_grad()
    def translate_audio_to_sign(self, speech: torch.Tensor,
                                num_frames: Optional[int] = None,
                                guidance_scale: float = 1.0, ddim_steps: int = 20,
                                max_gloss_len: int = 16) -> dict:
        """Full acoustic path: audio features -> spoken tokens -> gloss -> motion."""
        self.eval()
        device = next(self.parameters()).device
        spoken = self.recognize_speech(speech.to(device))     # ids in 1..K
        # Spoken CTC ids (1..K) -> planner source tokens (concept + offset).
        max_len = max((len(s) for s in spoken), default=1) or 1
        src = torch.zeros(len(spoken), max_len, dtype=torch.long, device=device)
        for i, s in enumerate(spoken):
            if s:
                ids = torch.tensor([tok - 1 + CONTENT_OFFSET for tok in s],
                                   dtype=torch.long, device=device)
                src[i, :len(s)] = ids
        out = self.translate_speech_to_sign(src, num_frames=num_frames,
                                            guidance_scale=guidance_scale,
                                            ddim_steps=ddim_steps,
                                            max_gloss_len=max_gloss_len)
        out["spoken_tokens"] = spoken
        return out

    @torch.no_grad()
    def recognize(self, pose: torch.Tensor):
        return self.recognizer.decode(pose)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
