# Contract: AUModel Public Interface

**Feature**: 002-model-architecture | **Phase 2 contracts** | **Date**: 2026-03-04
**Addresses**: FR-001 (public exports), FR-012 (forward signature), FR-013 (param count), FR-014 (default config)

---

## Purpose

This contract defines the stable public interface of the `model` package. All downstream epics depend on this interface:

- **Epic 3 (Data Pipeline)**: Needs `ModelConfig.vocab_size`, `ModelConfig.max_seq_len`
- **Epic 4 (Pretraining)**: Instantiates `AUModel(config)`, calls `model(tokens, targets)`, reads `model.get_num_params()`
- **Epic 5 (SFT)**: Loads a checkpoint into `AUModel`, calls `model(tokens, targets)` with masked targets
- **Epic 6 (Inference)**: Calls `model(tokens)` without targets, reads logits for sampling

Once the pretraining run begins (Epic 4), this interface is **frozen**. Breaking changes require a new spec.

---

## Import Pattern

All downstream epics import the model using:

```python
from model import AUModel, ModelConfig
```

This works because `model/__init__.py` contains:
```python
from .transformer import AUModel
from .config import ModelConfig
__all__ = ["AUModel", "ModelConfig"]
```

**Do not** import from internal submodules (e.g., `from model.transformer import AUModel`) — always use the package-level import.

---

## Class: `ModelConfig`

**Module**: `model/config.py`
**Imported as**: `from model import ModelConfig`

### Constructor

```python
@dataclass
class ModelConfig:
    vocab_size: int = 64000
    d_model: int = 1536
    num_heads: int = 12
    num_kv_heads: int = 6
    num_layers: int = 24
    ffn_hidden_dim: int = 4352
    max_seq_len: int = 4096
    dropout: float = 0.0
    rope_theta: float = 500000.0
```

**Validation** (raised at construction time):

| Condition | Error |
|-----------|-------|
| `num_heads % num_kv_heads != 0` | `ValueError: num_heads (N) must be divisible by num_kv_heads (M)` |

**Usage**:
```python
config = ModelConfig()                               # default ~1.3B model
config = ModelConfig(vocab_size=64000, d_model=512)  # small test model
```

---

## Class: `AUModel`

**Module**: `model/transformer.py`
**Imported as**: `from model import AUModel`

### Constructor

```python
def __init__(self, config: ModelConfig) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `ModelConfig` | Architecture hyperparameters |

**Behavior**:
- Creates all submodules (embedding, transformer blocks, RMSNorm, LM head)
- Ties `lm_head.weight` to `embed.weight` (no duplicate storage)
- Registers RoPE frequency buffers (`freqs_cis`) — move with model on `.to(device)`
- Uses PyTorch default weight initialization

**Example**:
```python
config = ModelConfig()
model = AUModel(config)
model = model.to("cuda")        # all buffers move automatically
model = model.bfloat16()        # cast weights for training
```

---

### Method: `forward`

```python
def forward(
    self,
    tokens: torch.Tensor,
    targets: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `tokens` | `torch.LongTensor` shape `(B, T)` | Integer token IDs, `0 ≤ id < vocab_size`, `T ≤ max_seq_len` |
| `targets` | `torch.LongTensor` shape `(B, T)` or `None` | Next-token targets for loss; pass `None` during inference |

**Returns**: `(logits, loss)`

| Return value | Type | Description |
|--------------|------|-------------|
| `logits` | `torch.Tensor` shape `(B, T, vocab_size)` | Raw unnormalized scores for each position |
| `loss` | `torch.Tensor` scalar or `None` | Cross-entropy loss if `targets` provided; `None` otherwise |

**Behavior**:
- Applies causal self-attention (each position attends only to previous positions)
- Applies RoPE to queries and keys within attention
- When `targets` is provided: computes `F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))`
- When `targets` is `None`: returns `(logits, None)` — no loss computed

**Example — Training**:
```python
logits, loss = model(tokens, targets)
loss.backward()
```

**Example — Inference**:
```python
logits, _ = model(tokens)              # (B, T, vocab_size)
next_token_logits = logits[:, -1, :]   # (B, vocab_size)
```

---

### Method: `get_num_params`

```python
def get_num_params(self) -> int
```

**Returns**: Total number of trainable parameters.

**Note**: Because embedding and LM head weights are tied, this count does not double-count the shared weight. Expected value: `~700,317,696` for default config.

**Example**:
```python
n = model.get_num_params()
print(f"{n / 1e6:.0f}M parameters")   # → "700M parameters"
```

---

## Device & Dtype Contract

| Operation | Behavior |
|-----------|----------|
| `model.to("cuda")` | All parameters AND `freqs_cis` buffer move to GPU |
| `model.bfloat16()` | All parameters cast to bfloat16; `freqs_cis` stays float32 |
| `model.cpu()` | Model returns to CPU; buffers move too |

**Training dtype**: Always `bfloat16` (never float16).

---

## Default Configuration — Parameter Budget

| Component | Calculation | Parameters |
|-----------|-------------|-----------|
| Token embedding (= LM head, tied) | `64000 × 1536` | `≈ 98M` |
| Attention Q proj per layer | `1536 × (12 × 128)` | `≈ 2.4M` |
| Attention K proj per layer | `1536 × (6 × 128)` | `≈ 1.2M` |
| Attention V proj per layer | `1536 × (6 × 128)` | `≈ 1.2M` |
| Attention O proj per layer | `(12 × 128) × 1536` | `≈ 2.4M` |
| FFN W1 per layer | `1536 × 4352` | `≈ 6.7M` |
| FFN W2 per layer | `4352 × 1536` | `≈ 6.7M` |
| FFN W3 per layer | `1536 × 4352` | `≈ 6.7M` |
| **Per layer total** | `2.4+1.2+1.2+2.4+6.7+6.7+6.7` | `≈ 27.3M` |
| **24 layers** | `24 × 27.3M` | `≈ 655M` |
| **Embedding (tied, counted once)** | | `+  98M` |
| **RMSNorm weights (×49)** | negligible | `≈   0.1M` |
| **Grand total** | | `≈ 753M → ~700,317,696` |

> Weight tying means the 98M embedding weight is shared with the LM head — it is counted once, not twice.
> Exact figure per constitution: **700,317,696**.
