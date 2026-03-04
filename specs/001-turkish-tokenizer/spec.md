# Feature Specification: Turkish Native Tokenizer

**Feature Branch**: `001-turkish-tokenizer`  
**Created**: 4 Mart 2026  
**Status**: Specification-Complete  
**Input**: User description: "Train a 64k-vocab Turkish-native SentencePiece BPE tokenizer on Wikipedia and OSCAR 23.01, with corpus downloader, fertility validation, Turkish character coverage checks, and TurkishTokenizer wrapper class"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Model builder trains a tokenizer (Priority: P1)

A developer building the AUModel LLM needs a tokenizer trained specifically on Turkish text before any other part of the model can be built. Without this tokenizer, the vocabulary size is unknown and corpus tokenization cannot begin.

**Why this priority**: Every other component (model architecture, data pipeline, pretraining) is blocked until the tokenizer exists and its vocabulary size is confirmed. This is the single entry gate for the entire project.

**Independent Test**: Can be fully tested by running the training pipeline end-to-end and confirming the output model file exists with exactly 64,000 vocabulary entries.

**Acceptance Scenarios**:

1. **Given** a Turkish text corpus is available, **When** the tokenizer training pipeline runs, **Then** a tokenizer model file and vocabulary file are produced with exactly 64,000 entries.
2. **Given** the tokenizer model file exists, **When** any Turkish text is encoded and then decoded, **Then** the decoded text exactly matches the original input.
3. **Given** the tokenizer model file exists, **When** the vocabulary is inspected, **Then** each of the 12 essential Turkish characters (ç, ğ, ı, İ, ö, ş, ü, Ü, Ö, Ç, Ğ, Ş) has its own dedicated entry — it is never represented as a fallback byte sequence.

---

### User Story 2 — Developer verifies Turkish language efficiency (Priority: P2)

A developer needs confidence that the tokenizer is genuinely efficient for Turkish — i.e., it does not fragment common Turkish words into excessive pieces the way a non-Turkish tokenizer would.

**Why this priority**: A tokenizer that wastes tokens on Turkish text reduces effective context length and degrades model quality. This must be validated before committing to a vocabulary.

**Independent Test**: Can be fully tested by running a fertility check on 10,000 Turkish sentences and confirming the average tokens-per-word ratio meets the target.

**Acceptance Scenarios**:

1. **Given** 10,000 Turkish sentences from a held-out Wikipedia set, **When** each sentence is tokenized, **Then** the average number of tokens per word is ≤ 1.4.
2. **Given** the common Turkish word "merhaba" (hello), **When** it is tokenized, **Then** it produces a single token (advisory check — not a hard failure).

---

### User Story 3 — Other pipeline components consume the tokenizer (Priority: P3)

The data pipeline and model architecture components need a stable, well-defined interface to the tokenizer — particularly the vocabulary size and the IDs of special control tokens (padding, start-of-sequence, end-of-sequence, chat role markers).

**Why this priority**: Downstream components must hard-code vocabulary size and special token IDs into their configuration. These values must be finalized before those components begin.

**Independent Test**: Can be fully tested by loading the tokenizer wrapper, calling its interface methods, and confirming all returned values match the expected constants.

**Acceptance Scenarios**:

- **Given** the trained tokenizer is loaded via the wrapper, **When** `tok.vocab_size` is accessed, **Then** it returns exactly 64,000.
- **Given** the trained tokenizer is loaded via the wrapper, **When** `tok.pad_id`, `tok.bos_id`, `tok.eos_id`, `tok.system_id`, `tok.user_id`, `tok.assistant_id`, and `tok.sep_id` are accessed, **Then** they return valid, distinct integer IDs for all eight special tokens.
3. **Given** text with chat-role markers is encoded, **When** decoded, **Then** the role markers are handled cleanly and do not corrupt surrounding text.

---

### Edge Cases

- What happens when the input text is an empty string? Encoding must not crash; decoding an empty list must return an empty string.
- What happens when the input contains numerals, punctuation, or mixed Turkish/Latin scripts? The tokenizer must handle all Unicode without errors.
- What happens when a document in the training corpus is extremely short (< 200 characters)? It must be silently skipped, not produce a training error.
- What happens if a configured corpus source is temporarily unavailable during download? The pipeline must log a warning and continue with whatever data has already been collected; it MUST NOT silently produce a corrupt or empty corpus file.
- What happens when two special tokens have conflicting IDs? The training configuration must prevent ID collisions between pad, unk, bos, and eos.
- What happens when the fertility check fails (ratio > 1.4)? The validation pipeline MUST run all checks to completion (report-all pattern), then halt with a non-zero exit code, log the exact measured ratio and a remediation hint (e.g., "increase corpus size or add more Turkish data"). No automatic retry occurs — the developer must decide the next action.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST produce a tokenizer trained exclusively on Turkish text — no pre-existing English or multilingual tokenizer may be reused or adapted.
- **FR-001b**: The training pipeline MUST write its outputs to exactly two fixed paths: `tokenizer/turkish_bpe.model` (binary model) and `tokenizer/turkish_bpe.vocab` (human-readable vocabulary list).
- **FR-002**: The vocabulary MUST contain exactly 64,000 entries.
- **FR-003**: The tokenizer MUST assign dedicated vocabulary entries to all 12 essential Turkish characters (ç, ğ, ı, İ, ö, ş, ü, Ü, Ö, Ç, Ğ, Ş) — none may fall back to byte-level representation.
- **FR-004**: Encoding followed by decoding MUST reproduce the exact original input for any valid Turkish Unicode string.
- **FR-005**: The average tokens-per-word ratio on a 10,000-sentence Turkish evaluation set MUST be ≤ 1.4.
- **FR-006**: The tokenizer MUST reserve fixed IDs for four base special tokens: padding (ID 0), unknown (ID 1), beginning-of-sequence (ID 2), end-of-sequence (ID 3).
- **FR-007**: The tokenizer MUST include dedicated vocabulary entries for four chat control tokens: system-role marker, user-role marker, assistant-role marker, and turn separator.
- **FR-008**: The training corpus MUST consist of Turkish plain text with NFC Unicode normalization applied to every document. The default starting source is the Turkish Wikipedia dump (20231101.tr); additional sources (e.g., OSCAR, mC4) MAY be added at the developer's discretion before training, and the pipeline MUST accept any plain-text file as input without requiring a specific dataset.
- **FR-009**: Documents shorter than 200 characters MUST be excluded from the training corpus.
- **FR-010**: The tokenizer MUST be accessible through a stable wrapper interface providing at minimum: encode, decode, vocabulary size query, and special token ID query.
- **FR-011**: If the post-training fertility check fails (measured ratio > 1.4), the validation pipeline MUST halt with a non-zero exit code, log the exact measured ratio, and emit a human-readable remediation hint. It MUST NOT silently continue or auto-retry.

### Key Entities

- **Tokenizer Model**: The trained artifact that maps Turkish text to integer sequences. Key attributes: vocabulary size (64,000), model type (subword BPE), language (Turkish-native). Output path: `tokenizer/turkish_bpe.model`. Produced once; consumed by model architecture, data pipeline, and inference.
- **Vocabulary**: The complete list of 64,000 subword pieces with their integer IDs. Output path: `tokenizer/turkish_bpe.vocab`. Includes base language pieces, Turkish character entries, special tokens, and chat control tokens.
- **Training Corpus**: The plain-text Turkish document collection used to learn vocabulary statistics. Default source: Turkish Wikipedia dump (20231101.tr). Additional sources are at developer discretion. All documents are NFC-normalized; short documents excluded.
- **Special Tokens**: A fixed set of control tokens with reserved IDs used to structure model input: PAD (0), UNK (1), BOS (2), EOS (3), plus four chat role tokens with IDs assigned at training time.
- **Tokenizer Wrapper**: A stable interface component that other pipeline stages use to interact with the trained model. Hides all implementation details from callers.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Tokenizer model file exists at `tokenizer/turkish_bpe.model` and vocabulary file exists at `tokenizer/turkish_bpe.vocab` — both load without errors after training completes.
- **SC-002**: Vocabulary contains exactly 64,000 entries — no more, no fewer.
- **SC-003**: All 12 essential Turkish characters have dedicated vocabulary IDs — zero byte-fallback representations among them.
- **SC-004**: Average tokens-per-word on 10,000 Turkish evaluation sentences is ≤ 1.4.
- **SC-005**: Encode → decode round-trip is lossless for 100% of a 100-string Turkish test suite.
- **SC-006**: All eight special token IDs (PAD, UNK, BOS, EOS, plus four chat control tokens) are distinct integers within the 0–63,999 range.
- **SC-007**: The wrapper interface returns correct vocabulary size and special token IDs on first call after loading — no warm-up or re-training required.

## Clarifications

### Session 2026-03-04

- Q: If the fertility check fails (ratio > 1.4), should the pipeline halt manually, auto-retry, or warn-only? → A: Pipeline halts, logs measured ratio + remediation hints; developer decides next action (add data / adjust hyperparams). No auto-retry.
- Q: What is the minimum corpus size if OSCAR is unavailable, and is OSCAR a required source? → A: Dataset sources are flexible and will be determined at training time. Only Turkish Wikipedia is confirmed as the default starting source. No minimum corpus size constraint — developer decides which datasets to use before running training.
- Q: What are the required output file paths for the trained tokenizer? → A: `tokenizer/turkish_bpe.model` (binary model) and `tokenizer/turkish_bpe.vocab` (vocabulary list) at repo root — both paths are fixed and locked.

## Assumptions

- The Turkish Wikipedia dump (20231101.tr) is publicly accessible via the HuggingFace datasets hub and is the default corpus source. Other dataset sources are flexible and will be determined by the developer at training time.
- Training is performed on a machine with internet access and sufficient disk space for a ~7.5 GB plain-text corpus.
- A vocabulary size of 64,000 is the locked target per PRD v1.3; this value is not negotiable.
- The four base special token IDs (0–3) are locked per the project constitution and must not change after training.
- Fertility threshold of ≤ 1.4 tokens/word is a hard gate — the tokenizer must be retrained if this is not met.
- The output files are locked to `tokenizer/turkish_bpe.model` and `tokenizer/turkish_bpe.vocab` at the repo root; downstream components (model architecture, data pipeline, inference) will reference these exact paths.
