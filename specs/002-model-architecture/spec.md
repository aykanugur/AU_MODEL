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
3. **Given** an instantiated model, **When** `model.get_num_params()` is called, **Then** the reported count is between 1.25B and 1.35B.
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

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The model package MUST expose `AUModel` and `ModelConfig` as public names importable from `model`.
- **FR-002**: `ModelConfig` MUST be a dataclass defining: `vocab_size`, `d_model`, `num_heads`, `num_kv_heads`, `num_layers`, `ffn_hidden_dim`, `max_seq_len`, `dropout`, `rope_theta`.
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
- **FR-014**: Default `ModelConfig` values MUST produce a model with approximately 1.3B trainable parameters.

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
- **SC-003**: Total trainable parameter count is between 1.25B and 1.35B for the default configuration.
- **SC-004**: Initial cross-entropy loss on random data is within 5% of `log(vocab_size)` (~10.77 for vocab size 64,000).
- **SC-005**: The model overfits a single batch to loss < 0.1 within 50 gradient steps.
- **SC-006**: All sanity checks (import, forward shape, param count, initial loss) pass with zero manual intervention in a clean environment.

## Assumptions

- **Vocab size**: The trained tokenizer from Phase 1 has `vocab_size=64000`; all default config values use this (DESIGN.md shows 32000, which predates the actual training run).
- **BF16 training**: Matrix multiplications use bfloat16 on GPU; sanity checks may run in float32 on CPU.
- **No dropout**: `dropout=0.0` is the default; standard practice for large pretraining runs.
- **Flash Attention**: `F.scaled_dot_product_attention(is_causal=True)` is used; PyTorch falls back automatically if Flash Attention kernel is unavailable.
- **Weight initialization**: PyTorch default initialization is used; no custom `_init_weights` required for this phase.
