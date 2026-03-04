# Feature Specification: AUModel Transformer Architecture

**Feature Branch**: `002-model-architecture`
**Created**: 2026-03-04
**Status**: Draft
**Input**: Implement LLaMA-style transformer model architecture in PyTorch for AUModel (~1.3B parameters)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Instantiate and Run Forward Pass (Priority: P1)

A researcher imports AUModel, creates it with the default configuration, feeds a batch of integer token IDs, and receives logits of the correct shape with no runtime errors.

**Why this priority**: Every downstream phase — pretraining, SFT, inference — depends on the model being importable, instantiable, and producing a valid forward pass. Nothing else can be tested or built until this works.

**Independent Test**: Can be fully tested by running `from model import AUModel, ModelConfig; m = AUModel(ModelConfig()); logits, _ = m(tokens)` on a fresh environment. Delivers a complete, working model object.

**Acceptance Scenarios**:

1. **Given** a default `ModelConfig`, **When** `AUModel(config)` is called, **Then** the model is created with no runtime errors and all submodules are accessible.
2. **Given** an instantiated model and a batch of shape `(2, 128)` integer token IDs, **When** `model(tokens)` is called, **Then** output logits have shape `(2, 128, vocab_size)`.
3. **Given** an instantiated model, **When** `model.get_num_params()` is called, **Then** the reported count is between 680M and 720M.
4. **Given** a `ModelConfig` where `num_kv_heads` does not evenly divide `num_heads`, **When** `AUModel(config)` is constructed, **Then** a clear `ValueError` is raised before any computation.

---

### User Story 2 - Verify Correct Initial Loss (Priority: P2)

A researcher confirms the untrained model produces cross-entropy loss close to `log(vocab_size)` on random data, validating that weight initialization is unbiased and the model has no systematic predictions at initialization.

**Why this priority**: An incorrect initial loss signals broken initialization, broken weight tying, or a normalization bug — all of which corrupt every pretraining gradient from step 0.

**Independent Test**: Feed random token sequences and random targets; confirm mean cross-entropy loss is within 5% of the theoretical maximum-entropy baseline.

**Acceptance Scenarios**:

1. **Given** an untrained model with `vocab_size=64000`, **When** a forward pass is run with random integer targets, **Then** mean cross-entropy loss is in `[10.0, 11.0]` (expected: `log(64000) ≈ 10.77`).
2. **Given** 100 different random batches evaluated on the untrained model, **When** losses are collected, **Then** variance across batches is below 0.5, confirming stable initialization.

---

### User Story 3 - Overfit a Single Batch (Priority: P3)

A researcher trains the model on a single fixed batch for 50 gradient steps and observes loss decreasing from ~10.77 to near zero, confirming the model has sufficient capacity and correct gradient flow through all layers.

**Why this priority**: Overfitting a single batch is the canonical "model is learning" check. It validates end-to-end gradient flow before committing to a 100-hour pretraining run.

**Independent Test**: Run 50 AdamW steps on one fixed batch; confirm loss at step 50 is below 0.1.

**Acceptance Scenarios**:

1. **Given** a single fixed batch of 8 sequences, **When** trained for 50 AdamW steps, **Then** training loss drops from ~10.77 at step 0 to below 0.1 at step 50.
2. **Given** the overfit test passes, **When** the model is evaluated on a different unseen batch, **Then** loss remains near 10.77, confirming memorization of the training batch (expected behavior).

---

### Edge Cases

- What happens when input `seq_len` equals `max_seq_len` exactly? (RoPE frequency table must cover the full length without index error.)
- What if the model is moved between devices (CPU → GPU) after instantiation? (RoPE buffers registered via `register_buffer` must move automatically.)
- What happens when `targets=None` is passed to `forward()`? (Model must return `(logits, None)` without computing loss — required for inference.)
- What if `vocab_size` in config differs from the actual tokenizer vocabulary? (No runtime guard needed here; documented as a user responsibility.)
- What is the return type of `Attention.forward()` when `past_kv` is provided? Returns `(output, (new_keys, new_values))` tuple; when `past_kv=None` returns `(output, None)` — caller decides whether to cache.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The model package MUST expose `AUModel` and `ModelConfig` as public names importable from `model`.
- **FR-002**: `ModelConfig` MUST be a dataclass defining: `vocab_size`, `d_model`, `num_heads`, `num_kv_heads`, `num_layers`, `ffn_hidden_dim`, `max_seq_len`, `dropout`, `rope_theta`. Default values MUST match constitution exactly: `vocab_size=64000, d_model=1536, num_heads=12, num_kv_heads=6, num_layers=24, ffn_hidden_dim=4352, max_seq_len=4096, dropout=0.0, rope_theta=500000`.
- **FR-003**: `ModelConfig` MUST validate at construction that `num_heads % num_kv_heads == 0`, raising a `ValueError` with a descriptive message if not.
- **FR-004**: The attention mechanism MUST apply Rotary Position Embeddings (RoPE) to query and key tensors before the attention dot product.
- **FR-005**: The attention mechanism MUST implement Grouped Query Attention (GQA): `num_kv_heads` K/V heads shared across `num_heads / num_kv_heads` query head groups.
- **FR-006**: All linear projection layers throughout the model MUST have `bias=False`.
- **FR-007**: The model MUST use pre-normalization: apply normalization to the input of each sub-layer (attention, feedforward), not to the output.
- **FR-008**: The normalization layer MUST be RMSNorm (not LayerNorm or BatchNorm).
- **FR-009**: The feedforward layer MUST use SwiGLU activation: two parallel up-projections gated by SiLU, fed into a single down-projection.
- **FR-010**: The token embedding weights and LM head weights MUST be tied (same parameter tensor).
- **FR-011**: RoPE frequency tensors MUST be stored as a non-trainable model buffer so they transfer automatically with the model across devices.
- **FR-012**: The `forward()` method MUST accept an optional `targets` tensor; when provided, compute and return cross-entropy loss; when absent, return `(logits, None)`.
- **FR-013**: The model MUST expose a `get_num_params()` method returning the total trainable parameter count.
- **FR-014**: Default `ModelConfig` values MUST produce a model with approximately 750M trainable parameters (~749,544,960 per analysis — see §Constitution Gate — Violation Record for correction note).
- **FR-015**: A standalone sanity check script (`model/sanity_check.py`) MUST exist that runs all 4 checks (import, forward shape, param count, initial loss) and exits with code 0 on pass, non-zero on failure.
- **FR-016**: A Colab notebook (`colab/02_model.ipynb`) MUST exist that runs the same 4 sanity checks plus the single-batch overfit test on GPU.
- **FR-017**: `Attention.forward()` MUST accept an optional `past_kv` parameter (default `None`). When `None`, standard causal self-attention is used (`is_causal=True`). When provided, it is a tuple `(past_keys, past_values)` that is concatenated with current K/V before the attention dot product, and the updated cache is returned.

### Key Entities

- **ModelConfig**: Dataclass; single source of truth for all architecture and training hyperparameters.
- **AUModel**: Full decoder-only transformer. Input: integer token IDs `(B, T)`. Output: logits `(B, T, vocab_size)` and optional loss scalar.
- **TransformerBlock**: One decoder layer — pre-norm → attention → residual → pre-norm → feedforward → residual.
- **Attention**: Multi-head causal self-attention with GQA and RoPE.
- **FeedForward**: SwiGLU two-gate feedforward block (`W1`, `W2`, `W3`).
- **RMSNorm**: Root-mean-square normalization layer with a learnable scale parameter.
- **RoPE frequencies**: Precomputed rotation tensors stored as a model buffer, indexed by sequence position.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Model instantiation with default config completes in under 30 seconds on a machine with at least 8 GB RAM.
- **SC-002**: Forward pass on a batch of 8 sequences of length 512 completes in under 10 seconds on CPU.
- **SC-003**: Total trainable parameter count is between 730M and 770M for the default configuration (~749,544,960 analytically confirmed; constitution figure of ~700M was an estimation error — see §Constitution Gate — Violation Record).
- **SC-004**: Initial cross-entropy loss on random data is within 5% of `log(vocab_size)` (~10.77 for vocab size 64,000).
- **SC-005**: The model overfits a single batch to loss < 0.1 within 50 gradient steps.
- **SC-006**: Running `python model/sanity_check.py` exits with code 0 and prints PASS for all 4 checks in a clean environment with no manual intervention.

## Assumptions

- **Vocab size**: The trained tokenizer from Phase 1 has `vocab_size=64000`; all default config values use this (DESIGN.md shows 32000, which predates the actual training run).
- **BF16 training**: Matrix multiplications use bfloat16 on GPU; sanity checks may run in float32 on CPU.
- **No dropout**: `dropout=0.0` is the default; standard practice for large pretraining runs.
- **Flash Attention**: `F.scaled_dot_product_attention(is_causal=True)` is used; PyTorch falls back automatically if Flash Attention kernel is unavailable.
- **Weight initialization**: PyTorch default initialization is used; no custom `_init_weights` required for this phase.

## ⚠ Constitution Gate — Violation Record

**Detected**: 2026-03-04 during `/speckit.plan` gate check.
**Resolution**: Spec corrected. DESIGN.md contains stale 1.3B values from an earlier design iteration. PRD v1.3 and constitution both specify 700M nominal target. Constitution architectural invariants win — all spec values updated to match.

**Detected (2)**: 2026-03-04 during `/speckit.implement` parameter count verification.
**Resolution**: The constitution’s “~700,317,696” param count estimate is arithmetically incorrect for the given hyperparameters. Analytical and runtime verification both give **749,544,960** (≈749.5M). SC-003 bounds corrected from [680M, 720M] to [730M, 770M]. The frozen hyperparameters are NOT changed. A formal constitution amendment to update the “~700,317,696” line is pending.

| Parameter | Original Spec (stale) | Corrected (constitution) |
|-----------|----------------------|-------------------------|
| `d_model` | 2048 | 1536 |
| `num_heads` | 16 | 12 |
| `num_kv_heads` | 8 | 6 |
| `ffn_hidden_dim` | 5504 | 4352 |
| `rope_theta` | 10000.0 | 500000 |
| Total params | ~1.3B | ~700M |

**Action required on DESIGN.md**: Should be updated to reflect 700M values in a separate task (non-blocking for this epic).
- **`torch.compile`**: `torch.compile(model)` is called by the trainer (Epic 4), not inside `AUModel`. Sanity checks and inference run on the uncompiled model.
- **PyTorch version**: Minimum PyTorch 2.1. `F.scaled_dot_product_attention` is used for attention (built-in, no extra install). External `flash-attn` package is optional and deferred to Epic 4.

## Clarifications

### Session 2026-03-04

- Q: Should `AUModel` include checkpoint save/load methods? → A: No — `AUModel` has no checkpoint I/O methods. All save/load logic (including Google Drive path management and Colab resume) belongs to `training/checkpoint.py` (Epic 4). `AUModel` exposes only the standard PyTorch `state_dict()` / `load_state_dict()` interface inherited from `nn.Module`.
- Q: Where do the sanity checks run? → A: Both — `model/sanity_check.py` (standalone CLI script for local/CI use) and `colab/02_model.ipynb` (Colab notebook for step-by-step GPU validation). Consistent with Epic 1 pattern.
- Q: Who calls `torch.compile()`? → A: The trainer (Epic 4) calls `torch.compile(model)` after instantiation. `AUModel` itself does not call compile — the model class stays pure. This avoids slowdown during sanity checks and decouples compile from model definition.
- Q: Should `Attention` support KV cache? → A: Yes — `Attention.forward()` accepts an optional `past_kv` parameter (default `None`). Training uses `past_kv=None` with `is_causal=True`; inference (Epic 6) passes accumulated K/V tensors. This avoids breaking the interface in Epic 6.
- Q: Minimum PyTorch version? → A: PyTorch ≥ 2.1. `F.scaled_dot_product_attention` (built-in Flash Attention) and `torch.compile` are both stable from 2.1. External `flash-attn` package (latest: 2.8.3) is optional — can be added in Epic 4 if throughput bottleneck is observed.
