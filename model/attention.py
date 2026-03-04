"""
model/attention.py — Grouped Query Attention with RoPE.

Implements multi-head causal self-attention with:
  - Grouped Query Attention (GQA): num_kv_heads K/V heads shared across groups
  - Rotary Position Embeddings (RoPE) applied to Q and K
  - F.scaled_dot_product_attention (auto-dispatches to Flash Attention 2 on CUDA)
  - Optional KV cache (past_kv) for inference

All linear layers have bias=False per constitution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import apply_rope


class Attention(nn.Module):
    """Multi-head causal self-attention with GQA and RoPE.

    Shapes (B=batch, T=seq_len, H=num_heads, Hkv=num_kv_heads, D=head_dim):
        Input x:          (B, T, d_model)
        Q after wq:       (B, T, H, D)
        K/V after wk/wv:  (B, T, Hkv, D)
        K/V expanded:     (B, T, H, D)   via repeat_interleave
        SDPA input:       (B, H, T, D)   after transpose
        Output:           (B, T, d_model)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.kv_groups = config.kv_groups

        self.wq = nn.Linear(config.d_model, config.num_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.num_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.num_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.num_heads * config.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass.

        Args:
            x:         Input tensor of shape (B, T, d_model).
            freqs_cis: Complex64 RoPE frequencies of shape (T, head_dim // 2).
                       Must be sliced to current sequence length before calling.
            past_kv:   Optional (past_keys, past_values) for KV caching during
                       inference. Both tensors have shape (B, Hkv, T_past, D).
                       Pass None during training.

        Returns:
            (output, new_kv) where:
              - output:  Tensor of shape (B, T, d_model).
              - new_kv:  (keys, values) tuple if past_kv was provided; else None.
                         Both tensors have shape (B, Hkv, T_total, D).
        """
        B, T, _ = x.shape

        # Project to Q, K, V and reshape to head format
        q = self.wq(x).view(B, T, self.num_heads, self.head_dim)      # (B, T, H, D)
        k = self.wk(x).view(B, T, self.num_kv_heads, self.head_dim)   # (B, T, Hkv, D)
        v = self.wv(x).view(B, T, self.num_kv_heads, self.head_dim)   # (B, T, Hkv, D)

        # Apply RoPE to Q and K
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Transpose to (B, H, T, D) layout for SDPA
        q = q.transpose(1, 2)   # (B, H, T, D)
        k = k.transpose(1, 2)   # (B, Hkv, T, D)
        v = v.transpose(1, 2)   # (B, Hkv, T, D)

        # KV cache: concatenate past K/V if provided
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)   # (B, Hkv, T_total, D)
            v = torch.cat([past_v, v], dim=2)
            new_kv: tuple[torch.Tensor, torch.Tensor] | None = (k, v)
            is_causal = False
        else:
            new_kv = None
            is_causal = True

        # GQA: expand K and V from Hkv heads to H heads via interleaved repeat
        # repeat_interleave produces [h0,h0,h1,h1,...] so Q groups align correctly
        k = k.repeat_interleave(self.kv_groups, dim=1)  # (B, H, T_total, D)
        v = v.repeat_interleave(self.kv_groups, dim=1)  # (B, H, T_total, D)

        # Scaled dot-product attention (Flash Attention 2 on CUDA)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)  # (B, H, T, D)

        # Merge heads and project output
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, H*D)
        out = self.wo(out)                                       # (B, T, d_model)

        return out, new_kv
