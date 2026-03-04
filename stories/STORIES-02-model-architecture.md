# Stories — Epic 2: Model Architecture

**Epic ref:** `EPIC-02-model-architecture.md`
**Branch:** `epic/02-model-architecture`
**Persona:** Developer
**Total stories:** 7

---

## ST-02-01 — Model Configuration

**As a developer,**
I want all model hyperparameters stored in a single loadable config object,
So that I can change or inspect the model dimensions in one place and all modules read from it consistently.

### Acceptance Criteria

- A `ModelConfig` dataclass holds all locked values: `d_model=1536`, `num_heads=12`, `num_kv_heads=6`, `num_layers=24`, `ffn_hidden=4352`, `max_seq_len=4096`, `vocab_size=64000`, `rope_theta=500000`, `norm_eps=1e-5`, `dropout=0.0`, and no bias flags.
- The config can be saved to a JSON file and re-loaded back into the same dataclass without data loss.
- Saving and loading the config alongside a checkpoint produces identical field values.

---

## ST-02-02 — Rotary Position Embeddings

**As a developer,**
I want rotary positional embeddings pre-computed with `rope_theta=500000`,
So that the model can attend to position information for sequences up to 4,096 tokens using the LLaMA 3 long-context configuration.

### Acceptance Criteria

- Pre-computed cosine and sine frequency tensors have shape `(max_seq_len, head_dim // 2)`.
- The computation uses `rope_theta=500000`, not the older default of 10,000.
- Applying the rotary embeddings to query and key tensors returns tensors of the same shape as the input.
- The rotation is computed using complex-number multiplication (not additive).

---

## ST-02-03 — Grouped Query Attention

**As a developer,**
I want a Grouped Query Attention module with 12 query heads and 6 key-value heads,
So that the model uses less memory for K/V cache while maintaining full-head expressiveness for queries.

### Acceptance Criteria

- Query projection maps `d_model` to `num_heads × head_dim` (12 heads).
- Key and value projections each map `d_model` to `num_kv_heads × head_dim` (6 heads).
- Key and value tensors are expanded from 6 heads to 12 heads before attention computation (each KV head serves 2 query heads).
- Attention uses `is_causal=True` — future tokens are never visible.
- No bias terms exist on any of the four projection matrices.

---

## ST-02-04 — SwiGLU Feed-Forward Network

**As a developer,**
I want a SwiGLU feed-forward block with a hidden dimension of 4,352,
So that the model uses the LLaMA 3-style gated activation for better training efficiency.

### Acceptance Criteria

- The block has three weight matrices: gate projection, down projection, and up projection.
- The gate-up product uses the SiLU activation (not ReLU or GELU).
- No bias terms exist on any of the three projection matrices.
- Input and output both have dimension `d_model=1536`.
- Hidden dimension is 4,352.

---

## ST-02-05 — Full Transformer Assembly

**As a developer,**
I want the complete AUModel to be assembled as 24 stacked transformer blocks with weight-tied embeddings,
So that I have a single `AUModel` class I can instantiate, forward-pass, and checkpoint.

### Acceptance Criteria

- The model contains an embedding layer, 24 transformer blocks, a final RMSNorm, and an output projection.
- Each transformer block follows the pre-norm pattern: RMSNorm before attention, RMSNorm before feed-forward.
- The output projection (LM head) shares its weight tensor with the input embedding — they are the same object in memory.
- The model produces logits of shape `(batch, seq_len, vocab_size)` for any valid input.
- No bias parameters exist anywhere in the model.

---

## ST-02-06 — Parameter Count Verification

**As a developer,**
I want to confirm the total number of trainable parameters is within the accepted 700M range,
So that I know the model is the intended size before starting any expensive training run.

### Acceptance Criteria

- Total parameter count is between 665,000,000 and 735,000,000 (700M ± 5%).
- A verification script prints the exact count and an explicit PASS or FAIL result.
- The script is runnable without a GPU (CPU inference on random input).

---

## ST-02-07 — Initial Loss Sanity Check

**As a developer,**
I want to verify that an untrained model produces a loss close to the theoretical random baseline,
So that I know the model is initialised correctly and the forward pass is numerically sound before training.

### Acceptance Criteria

- Running a forward pass on random integer inputs with untrained weights produces a cross-entropy loss between 10.57 and 11.57.
- The expected baseline is `log(64000) ≈ 11.07`.
- The check uses a short sequence (e.g., length 128) so it runs quickly on any machine.
- A verification script prints the loss value and an explicit PASS or FAIL result.

---

_Epic complete when all 7 stories pass their acceptance criteria._
_Last updated: 4 Mart 2026_
