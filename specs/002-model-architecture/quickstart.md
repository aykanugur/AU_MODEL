# Quickstart: AUModel Transformer Architecture

**Feature**: 002-model-architecture | **Phase 1** | **Date**: 2026-03-04

---

## Prerequisites

```bash
# Python 3.11, PyTorch ≥ 2.1
pip install torch>=2.1

# Verify
python -c "import torch; print(torch.__version__)"  # must be ≥ 2.1
```

**Repository**: `git clone https://github.com/aykanugur/AU_MODEL.git`
**Branch**: `002-model-architecture`

---

## 1. Import the Model

```python
from model import AUModel, ModelConfig

config = ModelConfig()  # default ~700M config (all constitution values)
print(config)
# ModelConfig(vocab_size=64000, d_model=1536, num_heads=12, num_kv_heads=6,
#             num_layers=24, ffn_hidden_dim=4352, max_seq_len=4096,
#             dropout=0.0, rope_theta=500000.0)
```

---

## 2. Instantiate and Inspect

```python
import torch

model = AUModel(config)
print(f"Parameters: {model.get_num_params() / 1e6:.1f}M")
# → Parameters: 700.3M

# Move to GPU for training
model = model.to("cuda").bfloat16()
```

---

## 3. Run a Forward Pass

```python
B, T = 2, 128  # batch size, sequence length
tokens = torch.randint(0, config.vocab_size, (B, T))

logits, _ = model(tokens)
print(logits.shape)   # → torch.Size([2, 128, 64000])
```

---

## 4. Compute Loss (Training Path)

```python
tokens  = torch.randint(0, config.vocab_size, (B, T))
targets = torch.randint(0, config.vocab_size, (B, T))

logits, loss = model(tokens, targets)
print(f"Loss: {loss.item():.4f}")   # → ~10.77 (log(64000)) for untrained model
```

---

## 5. Sanity Check Script (CLI)

Runs all 4 automated checks. Exit code 0 = all pass.

```bash
python model/sanity_check.py

# Expected output:
# [CHECK 1] Import ......................... PASS
# [CHECK 2] Forward pass shape ............. PASS  logits=(2, 128, 64000)
# [CHECK 3] Parameter count ................ PASS  700,317,696
# [CHECK 4] Initial loss ................... PASS  10.76 (expected ~10.77)
# All 4 checks passed.
```

**Exit codes**:
- `0` — all checks passed
- `1` — one or more checks failed (stderr shows which)

---

## 6. Colab Notebook (GPU Validation)

Open `colab/02_model.ipynb` in Google Colab:

1. **Section 1** — environment setup (mount Drive, install deps)
2. **Section 2** — instantiate model on H100
3. **Section 3** — run 4 sanity checks (same as CLI script)
4. **Section 4** — overfit single batch test (50 steps, loss → < 0.1)

**Expected overfit test output**:
```
Step  0 | loss: 10.77
Step 10 | loss:  7.32
Step 20 | loss:  3.81
Step 30 | loss:  1.24
Step 40 | loss:  0.28
Step 50 | loss:  0.04  ← target: < 0.1 ✓
```

---

## 7. Custom Configuration (Small Test Model)

For quick local testing without a GPU:

```python
config = ModelConfig(
    vocab_size=64000,
    d_model=256,
    num_heads=4,
    num_kv_heads=2,
    num_layers=2,
    ffn_hidden_dim=512,
    max_seq_len=512,
)
model = AUModel(config)
print(f"Params: {model.get_num_params() / 1e6:.2f}M")   # → ~20M
```

> **Warning**: Only use custom configs for debugging. The production model MUST use default `ModelConfig()` values exactly as specified in the constitution.

---

## File Reference

| File | Purpose |
|------|---------|
| `model/__init__.py` | `from model import AUModel, ModelConfig` |
| `model/config.py` | `ModelConfig` dataclass |
| `model/rope.py` | `precompute_freqs_cis()`, `apply_rotary_emb()` |
| `model/attention.py` | `Attention` (GQA + RoPE + optional KV cache) |
| `model/feedforward.py` | `FeedForward` (SwiGLU) |
| `model/transformer.py` | `RMSNorm`, `TransformerBlock`, `AUModel` |
| `model/sanity_check.py` | CLI: runs 4 checks, exits 0 on pass |
| `colab/02_model.ipynb` | Colab: 4 checks + overfit test on GPU |
