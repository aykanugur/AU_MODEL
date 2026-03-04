"""
model/transformer.py — RMSNorm, TransformerBlock, AUModel.

Contains:
  - RMSNorm:         Root-mean-square normalisation (replaces LayerNorm).
  - TransformerBlock: One decoder layer with pre-norm + residuals.
  - AUModel:         Full decoder-only transformer (~749.5M params by default).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import compute_freqs_cis
from .attention import Attention
from .feedforward import FeedForward


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation.

    Forward: x_norm = x / sqrt(mean(x²) + eps) * weight

    Replaces LayerNorm throughout the model (constitution invariant).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise x.

        Args:
            x: Tensor of any shape with last dim = dim.

        Returns:
            Normalised tensor of the same shape.
        """
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """One decoder layer: pre-norm attention + residual, pre-norm FFN + residual."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass through one decoder layer.

        Args:
            x:         Input tensor of shape (B, T, d_model).
            freqs_cis: RoPE frequencies, shape (T, head_dim // 2).
            past_kv:   Optional KV cache tuple for inference.

        Returns:
            (output, new_kv) with output shape (B, T, d_model).
        """
        attn_out, new_kv = self.attn(self.norm1(x), freqs_cis, past_kv)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# AUModel
# ---------------------------------------------------------------------------

class AUModel(nn.Module):
    """AUModel — ~749.5M parameter decoder-only transformer.

    Architecture highlights:
      - 24 TransformerBlocks with GQA (12/6 heads), RoPE, SwiGLU, RMSNorm
      - Tied token embedding and LM-head weights
      - RoPE frequencies registered as a non-trainable buffer
      - No torch.compile() call — the trainer (Epic 4) does that

    Constitution-locked defaults (see ModelConfig):
      vocab_size=64000, d_model=1536, num_heads=12, num_kv_heads=6,
      num_layers=24, ffn_hidden_dim=4352, max_seq_len=4096, rope_theta=500000
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer layers
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        # Final norm + LM head
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: lm_head shares weights with the embedding table
        self.lm_head.weight = self.embed.weight

        # Precomputed RoPE frequencies — non-trainable, moves with model.to(device)
        self.register_buffer(
            "freqs_cis",
            compute_freqs_cis(config.max_seq_len, config.head_dim, config.rope_theta),
        )

        # Apply weight initialisation after all sub-modules are created
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise all weights following LLaMA-style practice.

        Rules:
          - All nn.Embedding weights:          N(0, 0.02)
          - All nn.Linear weights:             N(0, 0.02)
          - Residual-branch projections only   N(0, 0.02 / sqrt(2 * num_layers))
            (attn.wo and ffn.w2) — prevents activation scale from growing with depth.
          - All bias terms:                    zeros (there are none — bias=False everywhere)
          - RMSNorm weights:                   already 1.0 from nn.Parameter(torch.ones(...))
          - lm_head.weight:                    tied to embed.weight — initialised once above

        The embedding std=0.02 ensures that logits (= hidden @ embed.weight.T) have
        variance ≈ d_model × 0.02² ≈ 0.6, keeping the initial softmax near-uniform and
        CE loss close to ln(vocab_size) ≈ 11.07.
        """
        std = 0.02
        residual_std = std / math.sqrt(2 * self.config.num_layers)

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Down-scale residual-branch output projections
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=residual_std)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            tokens:  Long tensor of shape (B, T) with 0 <= id < vocab_size
                     and T <= max_seq_len.
            targets: Optional long tensor of shape (B, T) for next-token
                     prediction loss. Pass None during inference.

        Returns:
            (logits, loss) where:
              - logits: Float tensor of shape (B, T, vocab_size).
              - loss:   Scalar cross-entropy loss if targets provided; else None.

        Raises:
            ValueError: If T > max_seq_len.
        """
        _B, T = tokens.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"seq_len {T} exceeds max_seq_len {self.config.max_seq_len}"
            )

        # Embed tokens
        x = self.embed(tokens)  # (B, T, d_model)

        # Slice RoPE buffer to current sequence length
        freqs_cis: torch.Tensor = self.freqs_cis[:T]  # (T, head_dim//2)

        # Pass through all transformer blocks
        for block in self.blocks:
            x, _ = block(x, freqs_cis)

        # Final normalisation + project to vocabulary
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets are provided
        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    def get_num_params(self) -> int:
        """Return total number of trainable parameters.

        Note: Because embedding weights are tied to lm_head, they are counted
        only once (nn.Parameter deduplication via set()).

        Returns:
            Integer count of trainable parameters.
        """
        return sum(p.numel() for p in set(self.parameters()))
