# Research: Turkish Native Tokenizer

**Feature**: 001-turkish-tokenizer | **Phase 0 research** | **Date**: 2026-03-04

---

## Research Task 1: SentencePiece BPE Training Parameters

### Decision: `normalization_rule_name = "identity"`

**Rationale**: The corpus preprocessing pipeline applies NFC normalization before writing `tokenizer_corpus.txt`. Passing pre-normalized text to SPM with `normalization_rule_name="identity"` means SPM stores text as-is — no further transformation. The alternative `nmt_nfkc_cf` (default) would decompose characters like `ğ`, `ş`, `ç`, `ı` into base+diacritic sequences, then case-fold them, making the tokenizer unable to distinguish case. `identity` is the correct choice for a case-preserving Turkish model.

**Alternatives considered**:
- `nmt_nfkc_cf` (default): Case-folds + NFKC. Would break Turkish case distinction (ğ/Ğ merged). Rejected.
- `nmt_nfkc` (no case-fold): Still applies NFKC which decomposes precomposed characters. Lossy for Turkish. Rejected.
- `nfkc`: Same problem as `nmt_nfkc`. Rejected.

---

### Decision: `byte_fallback = True`

**Rationale**: Even with `character_coverage=0.9999`, 0.01% of Unicode code points won't have dedicated tokens. Without byte fallback, these become `<unk>`, breaking encode/decode round-trip. With `byte_fallback=True`, SPM adds 256 byte-level tokens (`<0x00>` through `<0xFF>`) that can represent any Unicode character as a UTF-8 byte sequence. This guarantees zero `<unk>` at inference.

**Alternatives considered**:
- `byte_fallback=False`: Simpler vocabulary, but allows `<unk>` to appear. Rejected — breaks round-trip guarantee (SC-005) and introduces inference-time unknowns.

**Note**: The 256 byte tokens occupy vocabulary slots but at 64k vocab size (with only ~1,000 control characters likely edge cases), the cost is negligible.

---

### Decision: `input_sentence_size = 10_000_000`

**Rationale**: SPM loads training data into RAM to build initial character frequency tables and BPE merge statistics. A 7.5 GB corpus contains approximately 50-100 million sentences. Loading all of them would require >15 GB RAM — unavailable on Colab's CPU worker during SentencePiece training phase. `10_000_000` (10M sentences) is the value used by LLaMA and mT5 training runs; it provides statistically sufficient BPE merge coverage for a 64k vocabulary without OOM.

**Alternatives considered**:
- No limit (all sentences): OOM on Colab CPU worker. Rejected.
- `5_000_000`: Safer memory-wise but BPE merge statistics may underrepresent rare Turkish morphological forms. Rejected.
- `20_000_000`: Possible OOM risk on smaller Colab instances. Rejected.

---

### Decision: `random_seed = 42`

**Rationale**: SPM shuffles input sentences before training when `shuffle_input_sentence=True` (recommended). Setting a fixed seed makes training reproducible — re-running on the same corpus produces identical `.model` and `.vocab` files. This is critical for debugging vocabulary issues after training.

**Implementation note**: The SPM Python API parameter name is `random_seed=` (integer). Do NOT use `seed_sentencepiece_size=` — that parameter does not exist. Also set `shuffle_input_sentence=True` explicitly.

---

### Decision: Explicit `os.path.exists()` guard before training

**Rationale**: `SentencePieceTrainer.train()` silently overwrites the output `.model` file if it already exists. There is no `--overwrite` flag or error raised. If a developer accidentally re-runs the training script against an existing trained model, it will be replaced without warning. The training script must check `os.path.exists(output_model_path)` before calling `train()` and either abort with a message or require an explicit `--force` flag.

**Alternatives considered**:
- Rely on SPM error behavior: SPM does not error — it silently overwrites. Cannot rely on this. Rejected.
- Always overwrite: Acceptable only if `--force` is explicitly passed by the developer.

---

## Research Task 2: Python Validation Pipeline Exit Codes

### Decision: `sys.exit(1)` for gate failure, `sys.exit(2)` for runtime error

**Rationale**: Exit code `0` = all checks passed. Exit code `1` = one or more validation thresholds exceeded (fertility > 1.4, round-trip failure, missing Turkish chars, etc.). Exit code `2` = script failed to run (model file not found, corpus missing, import error). This is the convention used by `pytest`, `mypy`, `flake8`, and `grep`, making it composable in shell scripts.

**Colab context**: In Colab, the exit code is surfaced as a cell error. A clear distinction between "threshold failed" and "script crashed" aids debugging.

**Alternatives considered**:
- Always exit 1 for any failure: Does not distinguish gate failure from environmental issue. Rejected.
- Custom exit codes per check: Over-engineered for 4 checks. Rejected.

---

### Decision: Print numeric values to `stdout`, prose + remediation hints to `stderr`

**Rationale**: Colab renders stderr in red, immediately attracting attention to failures. Printing numeric values (fertility score, round-trip pass count) to stdout allows `grep`/pipeline capture. Prose like "FAIL: fertility 1.52 > threshold 1.40. Try expanding corpus." goes to stderr.

**Implementation pattern**:
```python
print(f"fertility: {score:.4f}", file=sys.stdout)
if score > FERTILITY_THRESHOLD:
    print(f"FAIL fertility: {score:.4f} > {FERTILITY_THRESHOLD}. Consider expanding corpus.", file=sys.stderr)
```

---

### Decision: Use `if/raise` — never `assert`

**Rationale**: Python strips `assert` statements when run with the `-O` (optimize) flag. Colab cells using `!python -O script.py` would silently skip all assertions. Use `if condition: raise ValueError(...)` for all validation logic.

---

### Decision: Report-all pattern (collect all 4 results, single exit at end)

**Rationale**: Fail-fast (exit on first failure) hides downstream failures. If fertility fails, the developer needs to know whether round-trip and character coverage also fail, because the fix strategy differs. The report-all pattern runs all 4 checks, accumulates `ValidationResult` objects, prints a summary table, then calls `sys.exit(1)` once at the end if any check failed.

**Implementation pattern**:
```python
results = []
results.append(check_fertility(tokenizer, corpus))
results.append(check_roundtrip(tokenizer, test_strings))
results.append(check_turkish_chars(tokenizer))
results.append(check_special_tokens(tokenizer))

# Print summary table
for r in results:
    status = "PASS" if r.passed else "FAIL"
    print(f"[{status}] {r.check_name}: {r.message}")

if any(not r.passed for r in results):
    sys.exit(1)
```

---

## Research Task 3: Python Tokenizer Wrapper Interface Design

### Decision: Separate `tokenizer/tokenizer.py` from `train_tokenizer.py`

**Rationale**: `train_tokenizer.py` imports `sentencepiece.SentencePieceTrainer` (training), datasets, tqdm, etc. — dependencies not needed at inference. `tokenizer.py` imports only `sentencepiece.SentencePieceProcessor` (inference). Keeping them separate means model code, data pipeline, and inference code can do `from tokenizer import Tokenizer` without pulling in training dependencies. This is the standard pattern from LLaMA's `tokenizer.py` and Andrej Karpathy's nanoGPT.

---

### Decision: Class-based `Tokenizer` with `@property` special token IDs

**Rationale**: A class encapsulates the `SentencePieceProcessor` instance (`self._sp`). `@property` accessors delegate to `sp.piece_to_id()` at access time rather than storing IDs at construction — this ensures IDs are always authoritative from the loaded model file and eliminates the risk of stale cached values if the model is somehow reloaded.

**Design rule**: Individual `@property` per special token (not a dict). Downstream code writes `tok.pad_id` not `tok.special_ids["pad"]` — this is explicit, IDE-autocompletable, and statically type-checkable.

```python
@property
def pad_id(self) -> int:
    return self._sp.piece_to_id("<pad>")
```

**Note**: `<pad>` maps to ID 0, which is SPM's `<pad>` control symbol. The spec requires PAD=0 — this is achieved by setting `pad_id=0` in the training call.

---

### Decision: `decode(ids, skip_special=True)` default

**Rationale**: Production behavior (model inference, evaluation) never wants special tokens in decoded text. `skip_special=True` as a default is the safe behavior. Developers debugging token sequences pass `skip_special=False` explicitly.

**Implementation**: SPM does not have a native `skip_special` flag on `decode()`. Must filter `ids` before calling `sp.decode()`:

```python
def decode(self, ids: list[int], skip_special: bool = True) -> str:
    if skip_special:
        special = {self.pad_id, self.bos_id, self.eos_id,
                   self.system_id, self.user_id, self.assistant_id, self.sep_id}
        ids = [i for i in ids if i not in special]
    return self._sp.decode(ids)
```

---

### Decision: `from tokenizer import Tokenizer` import pattern (via `__init__.py`)

**Rationale**: Downstream epics (Epic 2 model, Epic 3 data pipeline, Epic 6 inference) all need to import the tokenizer. A `tokenizer/__init__.py` re-export allows the clean `from tokenizer import Tokenizer` syntax rather than `from tokenizer.tokenizer import Tokenizer`. This is the conventional Python package pattern.

**File**: `tokenizer/__init__.py` contains:
```python
from .tokenizer import Tokenizer

__all__ = ["Tokenizer"]
```

---

## Summary Table

| Research Area | Key Finding | Applies To |
|--------------|-------------|-----------|
| SPM normalization | `identity` (not nmt_nfkc_cf) | `train_tokenizer.py` training call |
| SPM byte fallback | `True` | `train_tokenizer.py` training call |
| SPM sentence cap | `10_000_000` | `train_tokenizer.py` training call |
| SPM reproducibility | `random_seed=42` + `shuffle_input_sentence=True` | `train_tokenizer.py` training call |
| SPM overwrite guard | `os.path.exists()` pre-check | `train_tokenizer.py` guard logic |
| Validation exit codes | 0=pass, 1=gate fail, 2=runtime error | `validate_tokenizer.py` |
| Validation output | numeric→stdout, prose→stderr | `validate_tokenizer.py` |
| Validation strategy | Report-all, single sys.exit at end | `validate_tokenizer.py` |
| Use `if/raise` not assert | assert stripped by -O | All validation code |
| Wrapper: file split | `tokenizer.py` separate from `train_tokenizer.py` | Project structure |
| Wrapper: class design | `class Tokenizer` with `@property` special IDs | `tokenizer/tokenizer.py` |
| Wrapper: decode default | `skip_special=True` | `tokenizer/tokenizer.py` |
| Wrapper: import pattern | `from tokenizer import Tokenizer` via `__init__.py` | `tokenizer/__init__.py` |
