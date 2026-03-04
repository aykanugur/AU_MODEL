# Stories — Epic 3: Data Pipeline

**Epic ref:** `EPIC-03-data-pipeline.md`
**Branch:** `epic/03-data-pipeline`
**Persona:** Developer
**Total stories:** 6

---

## ST-03-01 — Multi-Source Corpus Download

**As a developer,**
I want all four Turkish corpora (Wikipedia, OSCAR 23.01, mC4, CC-100) to be downloadable with a single script,
So that I can bootstrap the full training dataset from scratch on any machine without manually sourcing files.

### Acceptance Criteria

- Running the download script fetches streaming data from all four sources.
- Turkish Wikipedia is pulled via the HuggingFace `datasets` library.
- OSCAR 23.01 Turkish and mC4 Turkish are pulled via the HuggingFace `datasets` library.
- CC-100 Turkish is downloaded from the `data.statmt.org` URL as a compressed file.
- The script completes without crashing on a fresh environment with internet access.

---

## ST-03-02 — Document Cleaning and Deduplication

**As a developer,**
I want every document in the corpus to be cleaned and deduplicated before tokenization,
So that the training data is free of HTML, encoding inconsistencies, near-empty documents, and exact duplicates.

### Acceptance Criteria

- Every document is NFC-normalized before storage.
- HTML tags are stripped from every document.
- Documents shorter than 200 characters are discarded.
- Documents where fewer than 50% of characters are alphabetic Turkish text are discarded.
- Each document's SHA-256 hash is computed and duplicate documents are dropped — only the first occurrence is kept.
- The pipeline logs how many documents were discarded at each filter step.

---

## ST-03-03 — Tokenization and Binary Sharding

**As a developer,**
I want the cleaned corpus tokenized and written to fixed-size binary shard files,
So that the training loop can memory-map shards sequentially without loading the entire dataset into RAM.

### Acceptance Criteria

- Each shard contains exactly 20,000,000 token IDs stored as `uint16` values.
- Shard files are named `shard_0000.bin`, `shard_0001.bin`, and so on with zero-padded four-digit indices.
- The total token count across all shards is ≥ 17.5 billion.
- The maximum token ID in any shard is less than 64,000 (no index-out-of-range values).
- A partial final shard is flushed at the end if remaining tokens are fewer than 20M.
- The pipeline prints the total token count on completion.

---

## ST-03-04 — Validation Split Isolation

**As a developer,**
I want a dedicated validation file that is guaranteed to never overlap with training shards,
So that validation perplexity measures generalisation and not memorisation.

### Acceptance Criteria

- The first 50,000,000 tokens from the Wikipedia stream are reserved to `data/processed/val.bin` before any training shard is written.
- The `val.bin` file contains exactly 50,000,000 token IDs (file size = 100 MB as `uint16`).
- None of the document hashes from the validation split appear in any training shard — spot-checked against 1,000 random document hashes.
- The validation file is written before training shard writing begins, ensuring no cross-contamination regardless of streaming ordering.

---

## ST-03-05 — Curriculum Shard Ordering

**As a developer,**
I want training shards ordered so that Phase 1 (Wikipedia-only) comes before Phase 2 (mixed-corpus),
So that the model trains on clean, well-structured Turkish text first before encountering noisier web data.

### Acceptance Criteria

- After all shards are written, Wikipedia-only shards are numbered 0000–0436.
- Mixed-corpus shards (OSCAR + mC4 + CC-100) are numbered 0437–0874.
- A `shard_manifest.json` file exists mapping each shard index to its source corpus name.
- Inspecting the manifest for any shard in 0000–0436 shows "wikipedia" as the source.
- Inspecting the manifest for any shard in 0437–0874 shows a non-Wikipedia source.

---

## ST-03-06 — Dataset Loader for Training

**As a developer,**
I want a PyTorch dataset class that can stream shard files in the correct curriculum order,
So that the training loop receives `(input_ids, target_ids)` batches at shape `(batch_size, 4096)` without implementing shard logic in the trainer.

### Acceptance Criteria

- The dataset accepts `shard_dir`, `split` (`'train'` or `'val'`), and `phase` (1 = shards 0–436, 2 = shards 437–874, 0 = all) as constructor arguments.
- Each item returned is a pair of tensors of shape `(4096,)` with dtype `torch.long`.
- The target tensor is the input tensor shifted by one position, with EOS appended at the end.
- Wrapping the dataset in a DataLoader at `batch_size=16` returns batches of shape `(16, 4096)`.
- The dataset reads shards via memory-mapping — it does not load entire shard files into RAM.

---

_Epic complete when all 6 stories pass their acceptance criteria._
_Last updated: 4 Mart 2026_
