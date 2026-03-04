# Project Constitution: AUModel

**Project**: AUModel — 700M parameter Turkish-native language model
**Version**: 1.0 | **Date**: 2026-03-04 | **Source**: PRD v1.3 + memory-bank

This constitution defines MUST-level invariants that cannot be changed without an explicit constitution amendment. All feature specs, plans, tasks, and implementation must comply.

---

## Architecture Invariants

These values are FROZEN after Epic 1 (tokenizer). Changing any of them invalidates all downstream checkpoints.

| Parameter | Value | Locked after |
|-----------|-------|-------------|
| `vocab_size` | 64,000 | Epic 1 (tokenizer training) |
| `d_model` | 1,536 | Epic 2 (model architecture) |
| `num_heads` | 12 | Epic 2 |
| `num_kv_heads` | 6 | Epic 2 (GQA: 2:1 ratio) |
| `num_layers` | 24 | Epic 2 |
| `ffn_hidden` | 4,352 | Epic 2 |
| `max_seq_len` | 4,096 | Epic 2 |
| `rope_theta` | 500,000 | Epic 2 |
| Total params | ~700,317,696 | Epic 2 |

**MUST**: These values MUST NOT be changed in any feature spec, plan, or implementation task without an explicit constitution amendment.

---

## Training Invariants

- **MUST use BF16** for all model training. FP16 is FORBIDDEN (gradient instability on Turkish morphology).
- **MUST set** `torch.backends.cuda.matmul.allow_tf32 = True` before any training run.
- **MUST call** `torch.compile(model)` after model initialization.
- **MUST use** `bias=False` on all `nn.Linear` layers in the model.
- **MUST use** `ignore_index=-100` in SFT cross-entropy loss (mask prompt tokens).
- **MUST use** RoPE (Rotary Position Embeddings). Absolute positional embeddings are FORBIDDEN.
- **MUST use** RMSNorm. LayerNorm is FORBIDDEN.

---

## Special Token Invariants

These IDs are FROZEN after tokenizer training (Epic 1). Any downstream epic that references special token IDs MUST use these values.

| Token | String | ID |
|-------|--------|----|
| PAD | `<pad>` | 0 |
| UNK | `<unk>` | 1 |
| BOS | `<s>` | 2 |
| EOS | `</s>` | 3 |
| SYSTEM | `[SYSTEM]` | 4 |
| USER | `[USER]` | 5 |
| ASSISTANT | `[ASSISTANT]` | 6 |
| SEP | `[SEP]` | 7 |

**MUST**: Token string order in `user_defined_symbols` MUST be `[SYSTEM],[USER],[ASSISTANT],[SEP]`. Changing this order changes IDs 4-7 and invalidates all SFT checkpoints.

---

## Tokenizer Invariants

- **MUST** train tokenizer exclusively on Turkish text (no multilingual or English pre-training).
- **MUST** use `normalization_rule_name="identity"` (NFC applied upstream).
- **MUST** use `byte_fallback=True` (guarantees zero `<unk>` at inference).
- **MUST** use `random_seed=42` for reproducibility.
- **MUST** use `input_sentence_size=10_000_000` (OOM protection).
- **MUST** achieve fertility <= 1.4 tokens/word before any downstream epic begins.
- Output paths are LOCKED: `tokenizer/turkish_bpe.model` and `tokenizer/turkish_bpe.vocab`.

---

## Infrastructure Invariants (Colab)

- **MUST** save all checkpoints to Google Drive (`/content/drive/MyDrive/AUModel/`).
- **MUST** support seamless resume after Colab session disconnect.
- Sessions can disconnect at any time — resume MUST require no manual intervention beyond re-running a cell.

---

## Code Quality Invariants

- **MUST NOT** use `assert` statements in production code — they are stripped by `-O` in Colab.
- Use `if/raise ValueError(...)` instead of `assert`.
- **MUST** use type hints on all function signatures.
- **MUST** add `bias=False` to all `nn.Linear` calls in model code.

---

## Amendment Process

To change any MUST in this constitution:
1. Create a separate spec entry titled "Constitution Amendment: <change>"
2. Document the reason, affected epics, and migration path
3. Update this file with the new value and a changelog entry
4. Re-validate all downstream specs/plans that reference the changed value

---

## Changelog

| Date | Change | Reason |
|------|--------|--------|
| 2026-03-04 | Initial constitution written | Governance formalization pre-implementation |
