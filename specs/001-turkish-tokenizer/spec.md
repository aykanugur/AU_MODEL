# Feature Specification: Turkish Native Tokenizer

**Feature Branch**: `001-turkish-tokenizer`  
**Created**: 4 Mart 2026  
**Status**: Draft  
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

1. **Given** the trained tokenizer is loaded via the wrapper, **When** `vocab_size()` is called, **Then** it returns exactly 64,000.
2. **Given** the trained tokenizer is loaded via the wrapper, **When** `special_ids()` is called, **Then** it returns valid integer IDs for: pad, unk, bos, eos, system, user, assistant, and separator tokens — all distinct from each other.
3. **Given** text with chat-role markers is encoded, **When** decoded, **Then** the role markers are handled cleanly and do not corrupt surrounding text.

---

### Edge Cases

- What happens when the input text is an empty string? Encoding must not crash; decoding an empty list must return an empty string.
- What happens when the input contains numerals, punctuation, or mixed Turkish/Latin scripts? The tokenizer must handle all Unicode without errors.
- What happens when a document in the training corpus is extremely short (< 200 characters)? It must be silently skipped, not produce a training error.
- What happens if the OSCAR dataset is temporarily unavailable during corpus download? The pipeline must continue with Wikipedia-only data and log a warning.
- What happens when two special tokens have conflicting IDs? The training configuration must prevent ID collisions between pad, unk, bos, and eos.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST produce a tokenizer trained exclusively on Turkish text — no pre-existing English or multilingual tokenizer may be reused or adapted.
- **FR-002**: The vocabulary MUST contain exactly 64,000 entries.
- **FR-003**: The tokenizer MUST assign dedicated vocabulary entries to all 12 essential Turkish characters (ç, ğ, ı, İ, ö, ş, ü, Ü, Ö, Ç, Ğ, Ş) — none may fall back to byte-level representation.
- **FR-004**: Encoding followed by decoding MUST reproduce the exact original input for any valid Turkish Unicode string.
- **FR-005**: The average tokens-per-word ratio on a 10,000-sentence Turkish evaluation set MUST be ≤ 1.4.
- **FR-006**: The tokenizer MUST reserve fixed IDs for four base special tokens: padding (ID 0), unknown (ID 1), beginning-of-sequence (ID 2), end-of-sequence (ID 3).
- **FR-007**: The tokenizer MUST include dedicated vocabulary entries for four chat control tokens: system-role marker, user-role marker, assistant-role marker, and turn separator.
- **FR-008**: The training corpus MUST be sourced from Turkish Wikipedia and the OSCAR 23.01 Turkish subset, with NFC Unicode normalization applied to every document.
- **FR-009**: Documents shorter than 200 characters MUST be excluded from the training corpus.
- **FR-010**: The tokenizer MUST be accessible through a stable wrapper interface providing at minimum: encode, decode, vocabulary size query, and special token ID query.

### Key Entities

- **Tokenizer Model**: The trained artifact that maps Turkish text to integer sequences. Key attributes: vocabulary size (64,000), model type (subword BPE), language (Turkish-native). Produced once; consumed by model architecture, data pipeline, and inference.
- **Vocabulary**: The complete list of 64,000 subword pieces with their integer IDs. Includes base language pieces, Turkish character entries, special tokens, and chat control tokens.
- **Training Corpus**: The plain-text Turkish document collection used to learn vocabulary statistics. Sourced from Turkish Wikipedia and OSCAR 23.01 Turkish; NFC-normalized; short documents excluded.
- **Special Tokens**: A fixed set of control tokens with reserved IDs used to structure model input: PAD (0), UNK (1), BOS (2), EOS (3), plus four chat role tokens with IDs assigned at training time.
- **Tokenizer Wrapper**: A stable interface component that other pipeline stages use to interact with the trained model. Hides all implementation details from callers.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Tokenizer model file and vocabulary file exist and load without errors after training completes.
- **SC-002**: Vocabulary contains exactly 64,000 entries — no more, no fewer.
- **SC-003**: All 12 essential Turkish characters have dedicated vocabulary IDs — zero byte-fallback representations among them.
- **SC-004**: Average tokens-per-word on 10,000 Turkish evaluation sentences is ≤ 1.4.
- **SC-005**: Encode → decode round-trip is lossless for 100% of a 100-string Turkish test suite.
- **SC-006**: All five special token IDs (PAD, UNK, BOS, EOS, plus chat tokens) are distinct integers within the 0–63,999 range.
- **SC-007**: The wrapper interface returns correct vocabulary size and special token IDs on first call after loading — no warm-up or re-training required.

## Assumptions

- The Turkish Wikipedia dump (20231101.tr) and OSCAR 23.01 Turkish subset are publicly accessible via the HuggingFace datasets hub during training.
- Training is performed on a machine with internet access and sufficient disk space for a ~7.5 GB plain-text corpus.
- A vocabulary size of 64,000 is the locked target per PRD v1.3; this value is not negotiable.
- The four base special token IDs (0–3) are locked per the project constitution and must not change after training.
- Fertility threshold of ≤ 1.4 tokens/word is a hard gate — the tokenizer must be retrained if this is not met.

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - [Brief Title] (Priority: P1)

[Describe this user journey in plain language]

**Why this priority**: [Explain the value and why it has this priority level]

**Independent Test**: [Describe how this can be tested independently - e.g., "Can be fully tested by [specific action] and delivers [specific value]"]

**Acceptance Scenarios**:

1. **Given** [initial state], **When** [action], **Then** [expected outcome]
2. **Given** [initial state], **When** [action], **Then** [expected outcome]

---

### User Story 2 - [Brief Title] (Priority: P2)

[Describe this user journey in plain language]

**Why this priority**: [Explain the value and why it has this priority level]

**Independent Test**: [Describe how this can be tested independently]

**Acceptance Scenarios**:

1. **Given** [initial state], **When** [action], **Then** [expected outcome]

---

### User Story 3 - [Brief Title] (Priority: P3)

[Describe this user journey in plain language]

**Why this priority**: [Explain the value and why it has this priority level]

**Independent Test**: [Describe how this can be tested independently]

**Acceptance Scenarios**:

1. **Given** [initial state], **When** [action], **Then** [expected outcome]

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right edge cases.
-->

- What happens when [boundary condition]?
- How does system handle [error scenario]?

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: System MUST [specific capability, e.g., "allow users to create accounts"]
- **FR-002**: System MUST [specific capability, e.g., "validate email addresses"]  
- **FR-003**: Users MUST be able to [key interaction, e.g., "reset their password"]
- **FR-004**: System MUST [data requirement, e.g., "persist user preferences"]
- **FR-005**: System MUST [behavior, e.g., "log all security events"]

*Example of marking unclear requirements:*

- **FR-006**: System MUST authenticate users via [NEEDS CLARIFICATION: auth method not specified - email/password, SSO, OAuth?]
- **FR-007**: System MUST retain user data for [NEEDS CLARIFICATION: retention period not specified]

### Key Entities *(include if feature involves data)*

- **[Entity 1]**: [What it represents, key attributes without implementation]
- **[Entity 2]**: [What it represents, relationships to other entities]

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: [Measurable metric, e.g., "Users can complete account creation in under 2 minutes"]
- **SC-002**: [Measurable metric, e.g., "System handles 1000 concurrent users without degradation"]
- **SC-003**: [User satisfaction metric, e.g., "90% of users successfully complete primary task on first attempt"]
- **SC-004**: [Business metric, e.g., "Reduce support tickets related to [X] by 50%"]
