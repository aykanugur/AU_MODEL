# Stories — Epic 1: Tokenizer

**Epic ref:** `EPIC-01-tokenizer.md`
**Branch:** `epic/01-tokenizer`
**Persona:** Developer
**Total stories:** 5

---

## ST-01-01 — Corpus Collection

**As a developer,**
I want the training corpus to be automatically downloaded and cleaned from Turkish Wikipedia and OSCAR 23.01,
So that I start tokenizer training with a consistent, reproducible text base without manually preparing files.

### Acceptance Criteria

- Running the corpus downloader produces a single `data/raw/tokenizer_corpus.txt` file.
- The file contains text from both Turkish Wikipedia and at least 5 GB of OSCAR 23.01 Turkish.
- Every document in the file has been NFC-normalized.
- Documents shorter than 200 characters are not in the file.
- The download can be re-run and produces the same output (deterministic filtering).

---

## ST-01-02 — Tokenizer Training

**As a developer,**
I want a SentencePiece BPE tokenizer to be trainable from a single script call,
So that I get a reproducible `.model` and `.vocab` file without manual SentencePiece configuration.

### Acceptance Criteria

- Running the training script produces `tokenizer/turkish_bpe.model` and `tokenizer/turkish_bpe.vocab`.
- The resulting vocabulary contains exactly 64,000 entries.
- Special token IDs are fixed: PAD=0, UNK=1, BOS=2, EOS=3.
- The model file can be loaded by SentencePiece without errors.
- Training completes without crashing on the full corpus file.

---

## ST-01-03 — Fertility Validation

**As a developer,**
I want to automatically verify that the tokenizer is efficient on Turkish text,
So that I know the tokenizer is suitable for the language before committing to it for all downstream epics.

### Acceptance Criteria

- A validation script tokenizes 10,000 Turkish sentences from a held-out Wikipedia split.
- The script computes and prints average tokens-per-word (fertility ratio).
- The fertility ratio is ≤ 1.4 tokens per word.
- If fertility exceeds 1.4, the script exits with a non-zero status code and a clear error message.

---

## ST-01-04 — Turkish Character Coverage

**As a developer,**
I want to confirm that all 12 Turkish-specific characters have their own token in the vocabulary,
So that Turkish text is never degraded to byte-level fallback for common characters.

### Acceptance Criteria

- A coverage checker script runs over all 12 characters: `ç ğ ı İ ö ş ü Ü Ö Ç Ğ Ş`.
- Each character resolves to a unique token ID that is not the UNK ID.
- The checker prints a pass/fail result per character.
- If any character maps to UNK, the script exits with a non-zero status code naming the failing character.

---

## ST-01-05 — Tokenizer Wrapper

**As a developer,**
I want a `TurkishTokenizer` wrapper that all other epics can import and use uniformly,
So that tokenization logic is defined in one place and every epic uses the same encode/decode interface.

### Acceptance Criteria

- The wrapper exposes `encode(text)` returning a list of integer token IDs.
- The wrapper exposes `decode(ids)` returning a string.
- Decoding an encoded string returns the original string for any valid Turkish Unicode input.
- The wrapper exposes `vocab_size()` returning 64,000.
- The wrapper exposes `special_ids()` returning a dict with at least BOS, EOS, PAD, UNK, system, user, and assistant token IDs.
- Calling `encode("merhaba")` returns exactly 1 token ID (single-token word coverage check).

---

_Epic complete when all 5 stories pass their acceptance criteria._
_Last updated: 4 Mart 2026_
