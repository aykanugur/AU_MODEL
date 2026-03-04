# Data Model: Turkish Pretraining Data Pipeline

**Date**: 2026-03-04  
**Branch**: `003-data-pipeline`

---

## Entities

### 1. Document

A single unit of raw text from a source dataset. The atomic processing unit.

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Raw text from HF dataset (field name varies per source) |
| `source_name` | `str` | One of `wikipedia`, `oscar`, `mc4`, `cc100` |
| `cleaned_text` | `str` | After HTML removal + NFC normalization |
| `doc_hash` | `str` | MD5 hex digest of `cleaned_text` — used for Bloom filter |
| `token_ids` | `list[int]` | SentencePiece IDs including BOS/EOS |
| `token_count` | `int` | `len(token_ids)` |

**Validation rules**:
- `token_count >= 100` after tokenization (enforce in cleaning step)
- All token IDs must be in `[0, 64000)` (vocab_size)
- `doc_hash` must not exist in the Bloom filter (dedup check)

**State transitions**:
```
raw → cleaned → deduplicated → tokenized → written_to_shard
                     ↓ (duplicate)
                  discarded
                     ↓ (< 100 tokens)
                  discarded
```

---

### 2. Source

Represents one HF dataset source with its access configuration.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Short identifier: `wikipedia`, `oscar`, `mc4`, `cc100` |
| `hf_id` | `str` | HuggingFace dataset ID |
| `config` | `str` | Dataset config/language code |
| `split` | `str` | Usually `"train"` |
| `text_field` | `str` | Field name for document text in the dataset record |
| `requires_auth` | `bool` | Whether `HF_TOKEN` is required |

**Pinned values**:

| `name` | `hf_id` | `config` | `text_field` | `requires_auth` |
|--------|---------|----------|-------------|-----------------|
| `wikipedia` | `wikimedia/wikipedia` | `20231101.tr` | `"text"` | `False` |
| `oscar` | `oscar-corpus/OSCAR-2301` | `tr` | `"content"` | `True` |
| `mc4` | `allenai/c4` | `tr` | `"text"` | `False` |
| `cc100` | `cc100` | `tr` | `"text"` | `False` |

---

### 3. Shard

A binary file on disk containing a flat array of `uint16` token IDs.

| Field | Type | Description |
|-------|------|-------------|
| `filename` | `str` | `shard_NNNN.bin` (zero-padded 4-digit index) |
| `path` | `str` | Absolute path on Drive |
| `token_count` | `int` | Number of uint16 values in the file |
| `byte_size` | `int` | `token_count * 2` |
| `source_breakdown` | `dict[str, int]` | Tokens contributed per source |
| `written_at` | `str` | ISO 8601 timestamp |

**Invariants**:
- `byte_size % 2 == 0` (uint16 alignment)
- `token_count` ≈ 500,000,000 (except the final shard which may be smaller)
- File must be readable as `np.fromfile(path, dtype=np.uint16)`

---

### 4. ManifestEntry (shards_manifest.json)

JSON file tracking pipeline state for resumability.

```json
{
  "version": 1,
  "bloom_filter_path": "/content/drive/.../bloom.pkl",
  "shards": [
    {
      "filename": "shard_0000.bin",
      "token_count": 498234112,
      "sources": {"wikipedia": 498234112},
      "written_at": "2026-03-04T14:22:11Z"
    }
  ],
  "total_tokens": 498234112,
  "source_totals": {"wikipedia": 498234112}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `version` | `int` | Schema version (currently `1`) |
| `bloom_filter_path` | `str` | Path to pickled Bloom filter checkpoint |
| `shards` | `list[ShardRecord]` | All completed shards in order |
| `total_tokens` | `int` | Sum of all shard token counts |
| `source_totals` | `dict[str, int]` | Per-source token totals |

**Shard record fields**: `filename`, `token_count`, `sources`, `written_at`

---

### 5. BloomFilterCheckpoint

Persisted Bloom filter state file.

| Field | Type | Description |
|-------|------|-------------|
| `path` | `str` | `<output_dir>/bloom.pkl` |
| `capacity` | `int` | `100_000_000` (100M unique docs) |
| `error_rate` | `float` | `0.01` (1% FPR) |
| `size_bytes` | `int` | ~120 MB in memory/on disk |

Written after each completed shard. Path recorded in manifest.

---

### 6. Corpus (logical aggregate)

Not a file; the logical view of the complete dataset.

| Property | Target Value |
|----------|-------------|
| `total_tokens` | ≥ 30,000,000,000 |
| `shard_count` | ≥ 1,500 |
| `token_id_range` | `[0, 64000)` |
| `shard_size_range` | 900 MB – 1,100 MB (except final) |

---

## Data Flow

```
HF Streams (4 sources)
        │
        ▼
  [Cleaning Step]
  - strip HTML
  - NFC normalize
  - min-length filter (< 100 tokens → discard)
        │
        ▼
  [Dedup Step]
  - MD5(cleaned_text) → Bloom filter check
  - duplicate → discard
        │
        ▼
  [Tokenize Step]  ← multiprocessing.Pool (≤32 workers)
  - SentencePiece.EncodeAsIds(cleaned_text)
  - prepend BOS (id=2), append EOS (id=3)
        │
        ▼
  [Shard Buffer]   uint16 numpy array, 500M slots
  - fill until full → write shard_NNNN.bin
  - validate shard
  - update manifest + Bloom checkpoint
        │
        ▼
  Drive: shard_0000.bin, shard_0001.bin, ..., shard_NNNN.bin
         shards_manifest.json
         bloom.pkl
```

---

## ShardedDataset

Used by the training loop (Epic 4).

```python
class ShardedDataset(Dataset):
    shard_paths: list[str]  # sorted list of shard_NNNN.bin paths
    seq_len: int            # 4096 (from config)
    
    # Each item: (input_ids, target_ids) both shape (seq_len,)
    # target_ids[t] == input_ids[t+1] for t in [0, seq_len-2]
    # target_ids[seq_len-1] is the token AFTER input_ids[seq_len-1]
    #   i.e. from the concatenated flat token stream
```

**Indexing**: Items are non-overlapping windows of `seq_len` tokens from the flat concatenated token stream across all shards. Item `i` starts at token `i * seq_len`.

**Validation on init**: Each shard must exist, have `byte_size > 0`, and `byte_size % 2 == 0`.
