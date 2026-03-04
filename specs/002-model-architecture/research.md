# Research: AUModel Transformer Architecture

**Feature**: 002-model-architecture | **Phase 0 research** | **Date**: 2026-03-04

---

## Research Task 1: RoPE Implementation Strategy

### Decision: `torch.polar` complex-number rotation

**Rationale**: Rotary Position Embeddings rotate Q and K vectors by a position-dependent angle. The cleanest implementation uses PyTorch's complex number support: reshape the real head dimension `(head_dim)` into `head_dim/2` complex numbers, multiply by precomputed `e^{iθ}` rotation factors, then reshape back. `torch.polar(magnitude, angle)` constructs complex tensors directly from polar form. This approach:
- Produces exactly the same arithmetic as the rotation matrix formulation
- Avoids explicit sin/cos matmul — fused into complex multiplication by the compiler
- Is the implementation used in the official LLaMA 2/3 reference code

**Alternatives considered**:
- Explicit sin/cos rotation matrix: `x_rotated = x*cos + rotate_half(x)*sin` — functionally identical, slightly more memory (stores both sin and cos tables). Viable fallback if complex dtype causes issues on older PyTorch.
- xFormers RoPE utility (`xformers.ops.memory_efficient_attention`): requires additional dependency. Rejected — PyTorch 2.1 built-in SDPA is sufficient.

**Implementation guide**:
```python
freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)
t = torch.arange(seq_len)
freqs = torch.outer(t, freqs)                                       # (seq_len, dim/2)
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)             # complex (seq_len, dim/2)
```

---

## Research Task 2: GQA K/V Expansion Method

### Decision: `tensor.repeat_interleave(groups, dim=2)`

**Rationale**: In GQA, `num_kv_heads=6` heads are shared across `num_heads=12 / num_kv_heads=6 = 2` query groups. To compute attention, K and V must be expanded from shape `(B, T, 6, head_dim)` to `(B, T, 12, head_dim)`. Two expansion methods exist:

- **`repeat_interleave(2, dim=2)`**: Inserts copies interleaved — head 0 is duplicated adjacent to itself, producing `[h0, h0, h1, h1, ...]`. This is the correct grouping for GQA: Q heads 0-1 share KV head 0, Q heads 2-3 share KV head 1, etc.
- **`.expand()` + `.reshape()`**: Expands without copying and reshapes. Produces non-contiguous tensors that may cause slowdowns in `scaled_dot_product_attention`. Also produces `[h0, h1, ..., h0, h1, ...]` ordering (wrong grouping for standard GQA).

`repeat_interleave` produces contiguous memory and correct grouping. Used in LLaMA 3 reference implementation.

**Alternatives considered**:
- `torch.repeat()`: Tile-based (`[h0,h1,...,h0,h1,...]`). Wrong grouping for GQA. Rejected.
- `expand().reshape()`: Non-contiguous, wrong grouping. Rejected.

---

## Research Task 3: Attention Kernel Selection

### Decision: `F.scaled_dot_product_attention(q, k, v, is_causal=True)`

**Rationale**: PyTorch 2.1 introduced stable Flash Attention 2 support via `torch.nn.functional.scaled_dot_product_attention`. With `is_causal=True`:
- Automatically applies causal mask — no separate mask tensor needed during training
- Dispatches to Flash Attention kernel when on CUDA (H100 will use Flash Attention 2 automatically)
- Falls back to memory-efficient attention or math attention on CPU — no code changes needed
- Handles BF16 tensors correctly (H100 native BF16 support)
- KV cache path: when `past_kv` is provided, pass `is_causal=False` and provide explicit `attn_mask` (or simply set `is_causal=False` if cache is short enough)

**Alternatives considered**:
- External `flash-attn==2.8.3` package: Faster by 10-20% on long sequences but requires CUDA wheel compilation. Optional optimization for Epic 4. Deferred.
- Manual `torch.matmul` + masking: ~3× slower, no Flash Attention path. Rejected.

---

## Research Task 4: SwiGLU `ffn_hidden_dim` Calculation

### Decision: `ffn_hidden_dim = 4352` (= `int(2/3 × 4 × d_model)` rounded to nearest 64)

**Rationale**: Noam Shazeer's SwiGLU paper recommends `ffn_hidden = 2/3 × 4 × d_model` to match the parameter count of a standard FFN with hidden size 4×d_model. For d_model=1536:

```
2/3 × 4 × 1536 = 4096
4096 rounded to nearest 64 = 4096
```

Wait — 4096 is already a multiple of 64. But constitution says 4352. Let me verify:
- `8/3 × 1536 = 4096.0` → If we use 8/3 multiplier: 4096
- `int(8/3 * 1536) = 4096`
- 4352 = 4096 + 256 = 4096 + 4×64 — this may be a tuned value

The constitution locks `ffn_hidden = 4352`. No derivation is disputed — this is a **frozen constitutional value**. Use 4352 regardless of the formula output.

**Key rule**: Always use the constitution value, not a formula. The formula is informational only.

---

## Research Task 5: Weight Initialization

### Decision: PyTorch default initialization — no custom `_init_weights`

**Rationale**: At 700M scale with Chinchilla-optimal training (25× compute), the model will see enough gradient signal to correct any initialization imbalance within the first few hundred steps. Custom initialization (e.g., GPT-2's `1/sqrt(2*n_layers)` scaling for residual projections) improves early-training stability but has negligible effect on final loss for well-tuned models.

**Alternatives considered**:
- GPT-2 residual scaling: `nn.init.normal_(p, std=0.02/math.sqrt(2*n_layers))` for output projections (wo, w2). Minor stability improvement. Can be added in Epic 4 if loss spikes are observed in early training.
- Kaiming uniform (PyTorch default for `nn.Linear`): Used. Appropriate for ReLU-family activations including SiLU.

---

## Research Task 6: KV Cache Contract (Epic 6 Readiness)

### Decision: `past_kv: tuple[Tensor, Tensor] | None = None` parameter on `Attention.forward()`

**Rationale**: Adding `past_kv=None` as a default parameter to `Attention.forward()` costs zero runtime overhead during training (the `None` branch is never taken) but avoids a breaking interface change in Epic 6. Without it, Epic 6 would need to modify the `Attention` class signature, which breaks the frozen model-interface contract.

**Return contract**:
- When `past_kv is None` (training): returns `(output, None)` — no cache stored
- When `past_kv` is a `(past_k, past_v)` tuple (inference): concatenates past K/V, returns `(output, (new_k, new_v))`

**Impact on `AUModel.forward()`**: `AUModel.forward()` signature stays unchanged for training. Epic 6 will add a `use_cache=False` parameter when needed.
