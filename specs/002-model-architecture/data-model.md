# Data Model: AUModel Transformer Architecture

**Feature**: 002-model-architecture | **Phase 1 design** | **Date**: 2026-03-04

---

## Overview

This feature produces one importable Python package (`model/`) containing 6 source files. The data model defines the entities involved: configuration, tensor shapes at each layer, and the module hierarchy.

---

## Entity 1: ModelConfig

The single source of truth for all architecture hyperparameters. An instance is passed to every module constructor.

| Field | Type | Locked Value | Source |
|-------|------|-------------|--------|
| `vocab_size` | `int` | `64000` | Constitution (Epic 1 tokenizer) |
| `d_model` | `int` | `1536` | Constitution |
| `num_heads` | `int` | `12` | Constitution |
| `num_kv_heads` | `int` | `6` | Constitution |
| `num_layers` | `int` | `24` | Constitution |
| `ffn_hidden_dim` | `int` | `4352` | Constitution |
| `max_seq_len` | `int` | `4096` | Constitution |
| `dropout` | `float` | `0.0` | Convention (large model pretraining) |
| `rope_theta` | `float` | `500000.0` | Constitution |

**Derived fields** (not stored, computed on demand):

| Derived | Formula | Value |
|---------|---------|-------|
| `head_dim` | `d_model // num_heads` | `128` |
| `kv_groups` | `num_heads // num_kv_heads` | `2` |

**Validation rule**: `num_heads % num_kv_heads == 0` — raises `ValueError` if violated.

**State**: Immutable after construction (dataclass with no post-init modification).

---

## Entity 2: RoPE Frequencies Buffer

Precomputed rotation tensors stored as a model buffer. Created once in `AUModel.__init__`.

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `freqs_cis` | `(max_seq_len, head_dim//2)` | `complex64` | One complex rotation factor per position per head-dim pair |

**Concrete shape**: `(4096, 64)` — 4096 positions, 64 complex numbers per head (head_dim=128 → 64 pairs).

**Lifecycle**: Created at model init, registered via `register_buffer("freqs_cis", ...)`. Moves with model on `.to(device)`. Never updated (non-trainable).

**Access pattern**:
```python
freqs_cis = self.freqs_cis[:T]          # slice to current seq len → (T, 64)
freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # → (1, T, 1, 64) for broadcast
```

---

## Entity 3: Attention Module

Implements multi-head causal self-attention with GQA and RoPE.

### Linear Projections (bias=False)

| Layer | Input shape | Output shape | Parameter count |
|-------|-------------|-------------|-----------------|
| `wq` | `(*, d_model)` | `(*, num_heads × head_dim)` = `(*, 1536)` | `1536 × 1536 = 2,359,296` |
| `wk` | `(*, d_model)` | `(*, num_kv_heads × head_dim)` = `(*, 768)` | `1536 × 768 = 1,179,648` |
| `wv` | `(*, d_model)` | `(*, num_kv_heads × head_dim)` = `(*, 768)` | `1536 × 768 = 1,179,648` |
| `wo` | `(*, num_heads × head_dim)` = `(*, 1536)` | `(*, d_model)` | `1536 × 1536 = 2,359,296` |

**Total per attention block**: 7,077,888 ≈ 7.1M parameters

### Tensor shapes through forward pass

```
Input x:           (B, T, 1536)
After wq:          (B, T, 1536)  → view → (B, T, 12, 128)
After wk:          (B, T, 768)   → view → (B, T, 6, 128)
After wv:          (B, T, 768)   → view → (B, T, 6, 128)
After RoPE on q,k: same shapes (rotation in-place)
After repeat_interleave on k,v: (B, T, 12, 128)
After transpose:   (B, 12, T, 128)  [q, k, v all]
After SDPA:        (B, 12, T, 128)
After transpose+reshape: (B, T, 1536)
After wo:          (B, T, 1536)
```

### KV Cache contract (inference only)

```
past_kv = None               → training path, is_causal=True, returns (out, None)
past_kv = (past_k, past_v)   → inference path, concat past+current, returns (out, (new_k, new_v))
past_k shape: (B, num_kv_heads, T_past, head_dim) = (B, 6, T_past, 128)
```

---

## Entity 4: FeedForward Module

Implements SwiGLU: `FFN(x) = W2(SiLU(W1(x)) ⊙ W3(x))`

### Linear Projections (bias=False)

| Layer | Input shape | Output shape | Parameter count |
|-------|-------------|-------------|-----------------|
| `w1` | `(*, 1536)` | `(*, 4352)` | `1536 × 4352 = 6,684,672` |
| `w2` | `(*, 4352)` | `(*, 1536)` | `4352 × 1536 = 6,684,672` |
| `w3` | `(*, 1536)` | `(*, 4352)` | `1536 × 4352 = 6,684,672` |

**Total per FFN block**: 20,054,016 ≈ 20.1M parameters

### Tensor shapes through forward pass

```
Input x:            (B, T, 1536)
W1(x):              (B, T, 4352)
SiLU(W1(x)):        (B, T, 4352)
W3(x):              (B, T, 4352)
SiLU(W1(x)) * W3(x): (B, T, 4352)   ← element-wise gate
W2(...):            (B, T, 1536)
```

---

## Entity 5: RMSNorm

Learnable root-mean-square normalization. Replaces LayerNorm throughout.

| Field | Shape | Parameter count |
|-------|-------|-----------------|
| `weight` | `(d_model,)` = `(1536,)` | `1536` |

**Forward**: `x / sqrt(mean(x²) + ε) * weight`, where `ε=1e-6`.

**Placement in model**: 2 per `TransformerBlock` (pre-attn, pre-ffn) + 1 final norm in `AUModel` = 49 total.

**Total RMSNorm params**: `49 × 1536 = 75,264` (negligible).

---

## Entity 6: TransformerBlock

One decoder layer combining attention and feedforward with pre-normalization and residual connections.

### Submodules

| Name | Type | Description |
|------|------|-------------|
| `norm1` | `RMSNorm(1536)` | Pre-attention normalization |
| `attn` | `Attention(config)` | GQA + RoPE self-attention |
| `norm2` | `RMSNorm(1536)` | Pre-FFN normalization |
| `ffn` | `FeedForward(config)` | SwiGLU feedforward |

**Forward computation**:
```python
x = x + attn(norm1(x), freqs_cis)   # pre-norm attention + residual
x = x + ffn(norm2(x))                # pre-norm FFN + residual
```

**Parameters per block**: 7.1M (attn) + 20.1M (ffn) + 3072 (norms) ≈ **27.2M**

---

## Entity 7: AUModel

The top-level module. Orchestrates embedding, all 24 transformer blocks, final norm, and LM head.

### Submodules

| Name | Type | Shape / Description |
|------|------|---------------------|
| `embed` | `nn.Embedding(64000, 1536)` | Token embeddings — weight TIED to lm_head |
| `blocks` | `nn.ModuleList` of 24 × `TransformerBlock` | 24 decoder layers |
| `norm` | `RMSNorm(1536)` | Final layer normalization |
| `lm_head` | `nn.Linear(1536, 64000, bias=False)` | Output projection — weight TIED to embed |
| `freqs_cis` | buffer `(4096, 64)` complex64 | Precomputed RoPE frequencies |

### Parameter count breakdown

| Component | Count |
|-----------|-------|
| Embedding (= LM head, tied) | ~98.3M |
| 24 × TransformerBlock | ~651.2M |
| RMSNorm (final + 2×24 blocks = 49) | ~0.075M |
| **Total** | **~749,544,960 ≈ 749.5M** |

### Tensor shapes through full forward pass

```
Input tokens:       (B, T)           — integer token IDs
After embed:        (B, T, 1536)
freqs_cis slice:    (T, 64)          — broadcast over batch and heads
After 24 blocks:    (B, T, 1536)     — each block: pre-norm → attn → residual → pre-norm → ffn → residual
After final norm:   (B, T, 1536)
After lm_head:      (B, T, 64000)    — logits (tied weight with embed)
```

### State transitions

```
AUModel.forward(tokens)              → (logits, None)
AUModel.forward(tokens, targets)     → (logits, loss_scalar)
```
