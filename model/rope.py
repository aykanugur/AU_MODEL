"""
model/rope.py — Rotary Position Embeddings (RoPE).

Provides a pure function that precomputes complex-valued rotation factors
using torch.polar. No nn.Module — AUModel registers the result as a buffer.
"""
import torch


def compute_freqs_cis(
    seq_len: int,
    head_dim: int,
    theta: float = 500_000.0,
) -> torch.Tensor:
    """Precompute RoPE frequency tensor using complex rotation (torch.polar).

    Args:
        seq_len:  Number of positions to precompute (typically max_seq_len).
        head_dim: Size of each attention head (d_model // num_heads).
                  Must be even.
        theta:    Base frequency (constitution default: 500_000.0).

    Returns:
        Complex64 tensor of shape (seq_len, head_dim // 2).
        Each entry is e^{i * m * theta_k} for position m and frequency index k.
    """
    # Inverse frequencies: theta_k = 1 / (theta ^ (2k / head_dim))
    # Shape: (head_dim // 2,)
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )

    # Position indices: shape (seq_len,)
    positions = torch.arange(seq_len, dtype=torch.float32)

    # Outer product → angles: shape (seq_len, head_dim // 2)
    angles = torch.outer(positions, freqs)

    # Convert to complex unit vectors: e^{i*angle}
    # torch.polar(magnitude=1, angle) → complex64 of same shape
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    return freqs_cis  # (seq_len, head_dim // 2), dtype=complex64


def apply_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE rotations to a query or key tensor.

    Args:
        x:          Real tensor of shape (B, T, num_heads, head_dim).
        freqs_cis:  Complex64 tensor of shape (T, head_dim // 2),
                    already sliced to the current sequence length.

    Returns:
        Real tensor of the same shape as x, with RoPE applied.
    """
    # Reshape x to complex: (B, T, num_heads, head_dim//2) complex
    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

    # freqs_cis: (T, head_dim//2) → broadcast shape (1, T, 1, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Multiply (rotation), then convert back to real
    x_rotated = torch.view_as_real(x_complex * freqs_cis)

    # Flatten last two dims: (B, T, num_heads, head_dim//2, 2) → (B, T, num_heads, head_dim)
    return x_rotated.flatten(-2)
