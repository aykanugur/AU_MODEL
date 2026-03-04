# Data Model: Turkish Native Tokenizer

**Feature**: 001-turkish-tokenizer | **Phase 1 design** | **Date**: 2026-03-04

---

## Overview

This feature produces one binary artifact (`turkish_bpe.model`) and one human-readable artifact (`turkish_bpe.vocab`). The data model defines the entities involved in corpus construction, training configuration, and the runtime wrapper.

---

## Entity 1: CorpusDocument

Represents one document during corpus filtering (in memory only — not persisted).

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | NFC-normalized document text (UTF-8) |
| `char_count` | `int` | Length in Unicode code points (not bytes) |
| `source` | `str` | Dataset name, e.g. `"wikipedia"`, `"oscar"`, `"custom"` |
| `kept` | `bool` | `True` if passed filter criteria (char_count ≥ 200) |

**Validation rules**:
- `char_count >= 200` to keep (per FR-008 minimum document length)
- `text` must be NFC-normalized before entry (not enforced inside entity — caller responsibility)
- HTML tags, URLs, and boilerplate footers stripped before NFC normalization

**State transitions**: `raw_document → filtered → written_to_corpus_txt`

---

## Entity 2: TrainingConfig

Represents the locked SentencePiece training parameters. Not a runtime class — documented here for reference during implementation.

| Field | Type | Locked Value | Source |
|-------|------|-------------|--------|
| `model_type` | `str` | `"bpe"` | FR-001 |
| `vocab_size` | `int` | `64000` | FR-002 (locked) |
| `character_coverage` | `float` | `0.9999` | FR-001 |
| `byte_fallback` | `bool` | `True` | research.md |
| `normalization_rule_name` | `str` | `"identity"` | research.md |
| `input_sentence_size` | `int` | `10_000_000` | research.md |
| `shuffle_input_sentence` | `bool` | `True` | research.md |
| `random_seed` | `int` | `42` | research.md |
| `pad_id` | `int` | `0` | FR-006 (locked) |
| `unk_id` | `int` | `1` | FR-006 (locked) |
| `bos_id` | `int` | `2` | FR-006 (locked) |
| `eos_id` | `int` | `3` | FR-006 (locked) |
| `user_defined_symbols` | `list[str]` | `["[SYSTEM]","[USER]","[ASSISTANT]","[SEP]"]` | FR-007 |

**Note**: `user_defined_symbols` are added in order — they receive IDs 4, 5, 6, 7 (the next available slots after BOS/EOS). The order in the list determines the ID assignment. This order is locked.

---

## Entity 3: SpecialTokenTable

The canonical mapping between special token names and integer IDs. This is the authoritative reference for all downstream code.

| Name | Token String | ID | Source | Notes |
|------|-------------|-----|--------|-------|
| PAD | `<pad>` | 0 | FR-006 (locked) | SPM control symbol, set via `pad_id=0` |
| UNK | `<unk>` | 1 | FR-006 (locked) | SPM control symbol, set via `unk_id=1` |
| BOS | `<s>` | 2 | FR-006 (locked) | SPM control symbol, set via `bos_id=2` |
| EOS | `</s>` | 3 | FR-006 (locked) | SPM control symbol, set via `eos_id=3` |
| SYSTEM | `[SYSTEM]` | 4 | FR-007 (locked) | First `user_defined_symbols` slot |
| USER | `[USER]` | 5 | FR-007 (locked) | Second `user_defined_symbols` slot |
| ASSISTANT | `[ASSISTANT]` | 6 | FR-007 (locked) | Third `user_defined_symbols` slot |
| SEP | `[SEP]` | 7 | FR-007 (locked) | Fourth `user_defined_symbols` slot |

**Critical**: IDs 4-7 depend on the order of `user_defined_symbols` in the training call. The order `[SYSTEM], [USER], [ASSISTANT], [SEP]` is locked. Changing the order changes all chat token IDs and breaks all SFT checkpoints.

---

## Entity 4: ValidationResult

Runtime record of one validation check result.

| Field | Type | Description |
|-------|------|-------------|
| `check_name` | `str` | Human-readable check name, e.g. `"fertility"`, `"round_trip"` |
| `passed` | `bool` | `True` if check passed |
| `measured_value` | `float \| None` | Actual measured value (if numeric check) |
| `threshold` | `float \| None` | Validation threshold (if numeric check) |
| `message` | `str` | Single-line human-readable result string |

**Validation checks** (4 total, all required):

| Check ID | `check_name` | Condition to PASS | Threshold | Related FR |
|---------|-------------|-------------------|-----------|-----------|
| V-001 | `"fertility"` | `avg_tokens_per_word <= 1.4` | `1.4` | FR-005, FR-011 |
| V-002 | `"round_trip"` | `decode(encode(s)) == s` for all test strings | 100% (100/100) | FR-004 |
| V-003 | `"turkish_chars"` | All 12 chars in vocabulary as direct tokens | No fallback | FR-003 |
| V-004 | `"special_tokens"` | All 8 special token IDs distinct and in range | IDs 0-7 distinct | FR-006, FR-007 |

---

## Entity 5: TrainedTokenizer (file artifacts)

The two output files produced by the training script.

| Property | Value |
|----------|-------|
| `model_path` | `tokenizer/turkish_bpe.model` (relative to repo root) |
| `vocab_path` | `tokenizer/turkish_bpe.vocab` (relative to repo root) |
| `model_format` | SentencePiece Protocol Buffer binary |
| `vocab_format` | TSV: `piece<TAB>score` per line, 64,000 lines |
| `vocab_size` | Exactly 64,000 |
| `approximate_model_size` | ~5 MB |
| `approximate_vocab_size` | ~3 MB |

**Vocab file format** (human-readable, for inspection):
```
<unk>	0
<s>	0
</s>	0
...
bir	-3.14159
ve	-3.21...
...
```

---

## Transition: from Artifact to Runtime

```
tokenizer/turkish_bpe.model
        │
        ▼
Tokenizer("tokenizer/turkish_bpe.model")   ← class instantiation
        │
        ▼
Tokenizer._sp                               ← SentencePieceProcessor instance
        │          │
        ▼          ▼
    .encode()    @property special IDs
    .decode()    .pad_id, .bos_id, etc.
```

All downstream epics (model code, data pipeline, inference) instantiate `Tokenizer` once at startup and pass the instance through their pipeline. No epic should call `SentencePieceProcessor` directly.
