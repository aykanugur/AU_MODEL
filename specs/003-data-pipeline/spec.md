# Feature Specification: Turkish Pretraining Data Pipeline

**Feature Branch**: `003-data-pipeline`
**Created**: 4 Mart 2026
**Status**: Draft

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Single-Source Download, Clean & Shard (Priority: P1)

A developer runs the preparation script targeting one dataset (e.g. Turkish Wikipedia) and receives a set of verified binary shard files on Google Drive, with a printed summary of total token count and shard count. This is the MVP: a fully working end-to-end pipeline for one source that proves every step works before adding more data.

**Why this priority**: Without a working single-source pipeline, nothing else can proceed. Wikipedia is the fastest source to download and the cleanest text, making it the safest first validation target.

**Independent Test**: Run `python scripts/prepare_data.py --source wikipedia` on a Colab session. Confirm shard files appear in the Drive output directory, total token count is printed, and `ShardedDataset` can read back the shards without errors.

**Acceptance Scenarios**:

1. **Given** the tokenizer model file exists and the Drive output directory is mounted, **When** the script runs with `--source wikipedia`, **Then** at least one shard file (`shard_0000.bin`) is written to the Drive directory and the script prints a token count summary.
2. **Given** the script has already produced shards from a previous run, **When** the script is re-run with `--source wikipedia`, **Then** it skips already-completed shards and only processes remaining ones.
3. **Given** a document contains only HTML tags or fewer than 100 tokens after cleaning, **When** the cleaning step processes it, **Then** that document is silently discarded and not written to any shard.

---

### User Story 2 — Full Multi-Source Corpus (Priority: P2)

A developer runs the pipeline across all configured sources (Wikipedia, OSCAR, mC4) and the combined output reaches ≥ 30 billion tokens across ≥ 1,500 shard files, ready for streaming pretraining.

**Why this priority**: The minimum token count is a hard prerequisite for reaching the 25× Chinchilla ratio at 749.5M parameters. But the single-source pipeline must work first.

**Independent Test**: Run `python scripts/prepare_data.py --source all`. Confirm total token count reported is ≥ 30B and the shard directory contains ≥ 1,500 files.

**Acceptance Scenarios**:

1. **Given** all three sources are configured and `HF_TOKEN` is set, **When** `--source all` is run to completion, **Then** total tokens ≥ 30,000,000,000 and shard count ≥ 1,500.
2. **Given** a source is temporarily unavailable mid-run, **When** a network error occurs, **Then** the script retries up to 3 times with exponential backoff before skipping that source and continuing with others.
3. **Given** the Drive output directory has less than 50 GB free, **When** the script detects insufficient space before writing a new shard, **Then** it prints a warning and stops without corrupting existing shards.

---

### User Story 3 — `ShardedDataset` for Training (Priority: P3)

The `training/dataset.py` `ShardedDataset` class reads the shard files and yields correctly-shifted `(input_ids, target_ids)` tensor pairs of length `max_seq_len=4096` for use in the training loop.

**Why this priority**: The training loop (Epic 4) depends on this, but the data must exist first. Can be implemented and unit-tested against any single shard before the full corpus is ready.

**Independent Test**: Instantiate `ShardedDataset` on any single shard file, fetch 10 items, and verify that `target_ids[t] == input_ids[t+1]` holds for all positions within each item.

**Acceptance Scenarios**:

1. **Given** at least one valid shard file exists, **When** `ShardedDataset` is instantiated and iterated, **Then** each item is a tuple of two `torch.LongTensor` shapes `(4096,)` where `target[t] == input[t+1]` for all `t < 4095`.
2. **Given** a shard file is corrupted (wrong size or unreadable), **When** `ShardedDataset` encounters it, **Then** it raises a `ValueError` with the shard path in the message.
3. **Given** the dataset is used with `DataLoader(num_workers=4)`, **When** multiple workers access different shards simultaneously, **Then** no data race or incorrect output occurs.

---

### Edge Cases

- What if the HuggingFace token is missing or invalid when accessing a gated dataset (OSCAR)?
- What if a shard file is partially written due to Drive disconnect mid-write?
- What if all documents from a source are filtered out after cleaning (zero output)?
- What if a tokenised document produces token IDs outside [0, 64000)?
- What if available Drive space runs out mid-shard, leaving a partial `.bin` file?

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The pipeline MUST support a `--source` flag accepting: `wikipedia`, `oscar`, `mc4`, `cc100`, `all`
- **FR-002**: The pipeline MUST read the HuggingFace API token from the `HF_TOKEN` environment variable (loaded from `.env`); credentials MUST NOT be hardcoded
- **FR-003**: Each source dataset MUST be loaded in **streaming mode** to avoid materialising the full dataset before processing begins
- **FR-004**: Each document MUST pass through a cleaning step that: removes HTML tags, normalises Unicode to NFC, and discards documents with fewer than 100 tokens after tokenisation
- **FR-005**: The pipeline MUST deduplicate documents using MD5 hash of cleaned text; duplicate documents are silently skipped
- **FR-006**: The pipeline MUST tokenise cleaned text using the trained BPE model at `tokenizer/turkish_bpe.model`, prepending BOS (`id=1`) and appending EOS (`id=2`) to each document
- **FR-007**: Tokenised documents MUST be concatenated into a flat uint16 numpy array and written to binary shard files of approximately 1 GB each, named `shard_NNNN.bin`, in the configured output directory
- **FR-008**: The pipeline MUST support resumable execution: on restart it detects existing complete shards and skips them, continuing from the next incomplete position
- **FR-009**: On completion the pipeline MUST print: number of shards written, total token count, token count per source, and total elapsed time
- **FR-010**: Each completed shard MUST be validated (file size > 0, readable as uint16) before the next shard begins; corrupted shards are deleted and re-written
- **FR-011**: `training/dataset.py` MUST implement `ShardedDataset(shard_paths, seq_len)` returning `(input_ids, target_ids)` pairs of shape `(seq_len,)` where `target_ids` is `input_ids` shifted left by one position
- **FR-012**: `ShardedDataset` MUST raise `ValueError` if any shard file is missing, empty, or has a byte size not divisible by 2 (uint16 alignment requirement)

### Key Entities

- **Document**: A single unit of raw text from a source dataset. Key attributes: raw text, source name, cleaned text, token count.
- **Shard**: A binary file of uint16 token IDs, approximately 1 GB. Named `shard_NNNN.bin`. Represents a contiguous flat slice of the full token stream.
- **Corpus**: The complete collection of shards. Attributes: total token count, shard count, per-source token breakdown.
- **Source**: One of Wikipedia, OSCAR, mC4, CC-100. Each yields documents as plain text strings via a streaming iterator. The text field name varies per source (`"text"` for Wikipedia/OSCAR/mC4).

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The full pipeline (`--source all`) produces a corpus of **≥ 30 billion tokens** across **≥ 1,500 shard files**
- **SC-002**: The Wikipedia-only pipeline (`--source wikipedia`) completes end-to-end on a Colab CPU session in **≤ 3 hours**
- **SC-003**: Each shard file is between **900 MB and 1,100 MB** (1 GB ± 10%)
- **SC-004**: `ShardedDataset` correctly shifts all token pairs: for every item, `target_ids[t] == input_ids[t+1]` for all `t` in `[0, seq_len-2]`
- **SC-005**: The pipeline is resumable: restarting after an interruption produces the same final corpus as an uninterrupted run
- **SC-006**: A spot-check of 100 randomly selected shards confirms all token IDs are in **[0, 64000)**

### Assumptions

- The trained tokenizer model file exists at `tokenizer/turkish_bpe.model` (produced in Epic 1)
- Google Drive is mounted at `/content/drive` during Colab execution; default output path is `/content/drive/MyDrive/aumodel_checkpoints/data/`
- HuggingFace OSCAR and mC4 datasets require a valid `HF_TOKEN` for streaming access
- Token IDs fit in uint16 (max value 65,535 > vocab_size 64,000 ✓)
- CC-100 is an optional fourth source; the 30B token target is achievable with Wikipedia + OSCAR + mC4 alone
