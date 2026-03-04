# Epic 2 — Model Architecture

| Field | Value |
|-------|-------|
| **Branch** | `epic/02-model-architecture` |
| **Base branch** | `main` |
| **Merge target** | `main` |
| **PRD refs** | F-02, G2, M2 |
| **Depends on** | Epic 1 (needs `vocab_size=64000` confirmed) |
| **Status** | ⬜ Not started |
| **Output files** | `model/config.py`, `model/rope.py`, `model/attention.py`, `model/feedforward.py`, `model/transformer.py`, `model/__init__.py` |

---

## Goal

Implement a 700M-parameter LLaMA 3-style transformer from scratch in pure PyTorch. No external LLM libraries — only `torch` and `torch.nn`.

**Locked config:**
```
d_model       = 1536
num_heads     = 12
num_kv_heads  = 6
num_layers    = 24
ffn_hidden    = 4352
max_seq_len   = 4096
vocab_size    = 64000
rope_theta    = 500000
norm_eps      = 1e-5
bias          = False  (on all Linear layers)
```

---

## Tasks

- [ ] **`model/config.py`** — `@dataclass ModelConfig`: all fields above plus `dropout=0.0`; include `from_json(path) → ModelConfig` and `to_json(path)` classmethods for saving/loading config alongside checkpoints
- [ ] **`model/rope.py`** — `precompute_freqs_cis(head_dim, max_seq_len, theta=500000) → (cos, sin)` tensors of shape `(max_seq_len, head_dim // 2)`; `apply_rotary_emb(xq, xk, cos, sin) → (xq_rot, xk_rot)` — rotate using complex number multiplication
- [ ] **`model/attention.py`** — `GroupedQueryAttention(nn.Module)`: `Wq = nn.Linear(d_model, num_heads * head_dim, bias=False)`, `Wk = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)`, `Wv = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)`, `Wo = nn.Linear(d_model, d_model, bias=False)`; repeat K/V from 6 heads to 12 heads via `torch.repeat_interleave`; call `F.scaled_dot_product_attention(q, k, v, is_causal=True)` (dispatches Flash Attention on H100)
- [ ] **`model/feedforward.py`** — `SwiGLUFFN(nn.Module)`: `W1 = nn.Linear(d_model, ffn_hidden, bias=False)`, `W2 = nn.Linear(ffn_hidden, d_model, bias=False)`, `W3 = nn.Linear(d_model, ffn_hidden, bias=False)`; `forward: x → W2(F.silu(W1(x)) * W3(x))`
- [ ] **`model/transformer.py`** — `RMSNorm(nn.Module)`: `forward: x / sqrt(mean(x^2) + eps) * weight`; `TransformerBlock`: pre-RMSNorm → GQA → residual add → pre-RMSNorm → SwiGLUFFN → residual add; `AUModel`: `nn.Embedding(64000, 1536)` → 24× `TransformerBlock` → final `RMSNorm` → `nn.Linear(1536, 64000, bias=False)` with `lm_head.weight = tok_embedding.weight` (weight tying)
- [ ] **Param count verification** — `assert sum(p.numel() for p in model.parameters()) == 700_317_696` (run in a test script before merging)
- [ ] **Initial loss check** — `x = torch.randint(0, 64000, (1, 128))`; compute cross-entropy loss; `assert 10.57 <= loss.item() <= 11.57` (log(64000) = 11.07 ± 0.5)

---

## Exit Criteria

| Check | Pass Condition |
|-------|---------------|
| Param count | 700,317,696 ± 5% (665M – 735M) |
| Initial loss | 11.07 ± 0.5 on random input |
| Memory at (B=8, T=4096), BF16, H100 | Peak GPU memory ≤ 22 GB (no optimizer) |
| Forward pass speed | ≥ 50,000 tokens/sec on H100 at batch=(8, 4096) |
| No bias parameters | `assert all('bias' not in n for n, p in model.named_parameters())` |
| Weight tying | `assert model.lm_head.weight.data_ptr() == model.tok_embedding.weight.data_ptr()` |

---

## Unlocks

- **Epic 4** (Pretraining) — needs `AUModel` class and `ModelConfig`

---

_Last updated: 4 Mart 2026_
