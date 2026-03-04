"""
model/feedforward.py — SwiGLU Feed-Forward Network.

Implements FFN(x) = W2(SiLU(W1(x)) ⊙ W3(x)).
All linear layers have bias=False per constitution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class FeedForward(nn.Module):
    """SwiGLU feed-forward block.

    Architecture:
        gate  = SiLU(W1(x))          # shape: (B, T, ffn_hidden_dim)
        value = W3(x)                # shape: (B, T, ffn_hidden_dim)
        out   = W2(gate * value)     # shape: (B, T, d_model)

    All projection layers have bias=False (constitution invariant).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
