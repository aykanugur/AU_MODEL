# Contract: Tokenizer Wrapper Interface

**Feature**: 001-turkish-tokenizer | **Phase 1 contracts** | **Date**: 2026-03-04  
**Addresses**: CHK006 (wrapper interface), CHK016 (method signatures), CHK017 (downstream import pattern)

---

## Purpose

This contract defines the stable public interface of `tokenizer/tokenizer.py`. All downstream epics (Epic 2 model, Epic 3 data pipeline, Epic 5 SFT, Epic 6 inference) depend on this interface. Once `impt` completes and the model is trained, this interface is **frozen** — breaking changes require a new spec.

---

## Import Pattern

All downstream epics import the tokenizer using:

```python
from tokenizer import Tokenizer

tok = Tokenizer("tokenizer/turkish_bpe.model")
```

This works because `tokenizer/__init__.py` contains:
```python
from .tokenizer import Tokenizer
__all__ = ["Tokenizer"]
```

**Do not** use `from tokenizer.tokenizer import Tokenizer` in downstream code — always use the package-level import.

---

## Class: `Tokenizer`

**Module**: `tokenizer/tokenizer.py`  
**Imported as**: `from tokenizer import Tokenizer`

### Constructor

```python
def __init__(self, model_path: str | Path) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str \| Path` | Path to `turkish_bpe.model` file |

**Behavior**:
- Loads the SentencePiece model file into memory
- Raises `FileNotFoundError` if model file does not exist
- Raises `RuntimeError` if the model file is corrupted or not a valid SPM model
- Does NOT download or train the model (training is `train_tokenizer.py`'s job)

**Example**:
```python
tok = Tokenizer("tokenizer/turkish_bpe.model")
```

---

### Method: `encode`

```python
def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | — | Input text to encode |
| `add_bos` | `bool` | `False` | Prepend BOS token (ID 2) to output |
| `add_eos` | `bool` | `False` | Append EOS token (ID 3) to output |

**Returns**: `list[int]` — token IDs in vocabulary range `[0, vocab_size)`

**Behavior**:
- Returns `[]` for empty string
- Never returns `unk_id` (1) for Turkish text — `byte_fallback=True` ensures coverage
- Does not modify `text` (NFC normalization was applied at corpus build time)

**Examples**:
```python
tok.encode("merhaba")               # → [5432, 7891]  (example IDs)
tok.encode("merhaba", add_bos=True) # → [2, 5432, 7891]
tok.encode("merhaba", add_eos=True) # → [5432, 7891, 3]
tok.encode("")                      # → []
```

---

### Method: `decode`

```python
def decode(self, ids: list[int], skip_special: bool = True) -> str
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ids` | `list[int]` | — | Token IDs to decode |
| `skip_special` | `bool` | `True` | If `True`, silently strip all special token IDs before decoding |

**Returns**: `str` — decoded text

**Behavior**:
- `skip_special=True` (default): Removes IDs `{0, 2, 3, 4, 5, 6, 7}` from `ids` before decoding. UNK (1) is NOT stripped (would indicate an encoding bug worth seeing).
- `skip_special=False`: Passes all IDs to SPM decoder as-is (special tokens appear as literal strings, e.g. `"<pad>"`)
- Returns `""` for empty list
- Round-trip guarantee: `decode(encode(s)) == s` for all `s` in Turkish text corpus (SC-005)

**Examples**:
```python
tok.decode([5432, 7891])                         # → "merhaba"
tok.decode([2, 5432, 7891, 3])                   # → "merhaba"  (skip_special=True, default)
tok.decode([2, 5432, 7891, 3], skip_special=False) # → "<s>merhaba</s>"
```

---

### Property: `vocab_size`

```python
@property
def vocab_size(self) -> int
```

**Returns**: `64000` (always — verified during training)

---

### Properties: Special Token IDs

All IDs delegate to `sp.piece_to_id()` — authoritative from the loaded model file.

```python
@property
def pad_id(self) -> int        # 0  — <pad>
@property
def unk_id(self) -> int        # 1  — <unk>
@property
def bos_id(self) -> int        # 2  — <s>
@property
def eos_id(self) -> int        # 3  — </s>
@property
def system_id(self) -> int     # 4  — [SYSTEM]
@property
def user_id(self) -> int       # 5  — [USER]
@property
def assistant_id(self) -> int  # 6  — [ASSISTANT]
@property
def sep_id(self) -> int        # 7  — [SEP]
```

**Note**: IDs 4-7 are determined by the order of `user_defined_symbols` in the training call. These values are locked per FR-007. Do not assume IDs without checking data-model.md's SpecialTokenTable.

---

## Usage Patterns for Downstream Epics

### Epic 2 (Model Architecture)
```python
from tokenizer import Tokenizer
tok = Tokenizer("tokenizer/turkish_bpe.model")
vocab_size = tok.vocab_size     # 64000 → used to set nn.Embedding size
pad_id = tok.pad_id             # 0 → used for attention mask
```

### Epic 3 (Data Pipeline)
```python
from tokenizer import Tokenizer
tok = Tokenizer("tokenizer/turkish_bpe.model")
# Encode document with BOS/EOS for causal LM training
tokens = tok.encode(document, add_bos=True, add_eos=True)
```

### Epic 5 (SFT)
```python
from tokenizer import Tokenizer
tok = Tokenizer("tokenizer/turkish_bpe.model")
# Chat format
user_tokens = [tok.user_id] + tok.encode(user_message) + [tok.sep_id]
assistant_tokens = [tok.assistant_id] + tok.encode(response) + [tok.eos_id]
```

### Epic 6 (Inference)
```python
from tokenizer import Tokenizer
tok = Tokenizer("tokenizer/turkish_bpe.model")
input_ids = tok.encode(prompt, add_bos=True)
output_text = tok.decode(generated_ids)  # skip_special=True by default
```

---

## What This Contract Does NOT Cover

- `train_tokenizer.py` (offline training CLI — not part of the runtime interface)
- `validate_tokenizer.py` (one-shot validation script — not an importable module)
- Internal SPM API calls (implementation detail of `tokenizer.py`)
- Batch encoding / padding (handled by the data pipeline, not the tokenizer wrapper)
