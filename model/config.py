"""
model/config.py — ModelConfig dataclass.

Single source of truth for all architecture hyperparameters.
All default values are frozen by the project constitution.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Architecture and training hyperparameters for AUModel.

    All default values are constitution-locked and must not be changed
    without a formal constitution amendment.
    """

    vocab_size: int = 64_000
    d_model: int = 1_536
    num_heads: int = 12
    num_kv_heads: int = 6
    num_layers: int = 24
    ffn_hidden_dim: int = 4_352
    max_seq_len: int = 4_096
    dropout: float = 0.0
    rope_theta: float = 500_000.0

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.num_heads

    @property
    def kv_groups(self) -> int:
        """Number of query heads per KV head (GQA ratio)."""
        return self.num_heads // self.num_kv_heads
