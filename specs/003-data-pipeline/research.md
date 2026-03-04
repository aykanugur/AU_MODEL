# Research: Turkish Pretraining Data Pipeline

**Date**: 2026-03-04  
**Branch**: `003-data-pipeline`  
**Purpose**: Resolve all NEEDS CLARIFICATION items from the spec and document key technical decisions with rationale.

---

## 1. Deduplication Strategy

**Decision**: Cross-source Bloom filter on `MD5(cleaned_text)`.

**Rationale**: A plain hash set holding 100M hashes would require ~3–4 GB RAM (40 bytes/entry in Python set). A Bloom filter can hold the same capacity at ~1% FPR in ~120 MB. Cross-source dedup prevents model from seeing the same document twice even if it appears in both OSCAR and mC4 (internet crawls overlap significantly).

**Library chosen**: `pybloom-live`
- Simple `BloomFilter(capacity, error_rate)` API
- Supports pickle to disk → enables checkpoint/resume of dedup state
- `BloomFilter(capacity=100_000_000, error_rate=0.01)` → ~120 MB RAM

```python
from pybloom_live import BloomFilter
bloom = BloomFilter(capacity=100_000_000, error_rate=0.01)
import hashlib, pickle
doc_hash = hashlib.md5(cleaned_text.encode()).hexdigest()
if doc_hash in bloom:
    continue  # skip duplicate
bloom.add(doc_hash)
# Checkpoint:
with open(bloom_path, "wb") as f:
    pickle.dump(bloom, f)
```

**Alternatives considered**:
- `datasketch` MinHashLSH — fuzzy dedup (near-duplicate). Rejected: overkill, significantly more complex, slower.
- Plain `set()` — too much RAM.
- Per-source dedup only — rejected in clarification: worse for model accuracy.

---

## 2. Dataset Versions (researched 2026-03-04)

**Decision**: Fixed pinned versions.

| Source | HuggingFace ID | Config | Text Field | Size (Turkish) |
|--------|---------------|--------|------------|----------------|
| Wikipedia | `wikimedia/wikipedia` | `20231101.tr` | `"text"` | ~500K articles |
| OSCAR | `oscar-corpus/OSCAR-2301` | `tr` (dedup) | `"content"` | 73.7 GB |
| mC4 | `allenai/c4` | `tr` | `"text"` | ~6B tr tokens |
| CC-100 | `cc100` | `tr` | `"text"` | Optional fallback |

**Important**: OSCAR-2301 uses field name `"content"`, not `"text"`. Other sources use `"text"`.

**OSCAR access status**: Dataset is currently **gated** with access temporarily suspended (copyright review). HF account approval request submitted. If access is not granted, CC-100 (`cc100`, config `tr`) is the fallback which covers similar Common Crawl data.

**Rationale for pinning**: Reproducibility. If OSCAR or mC4 is updated between runs, the token index into the corpus would shift, making checkpoint resume impossible. Fixed versions guarantee identical shards across machines.

---

## 3. Shard Format & Size

**Decision**: `uint16` numpy binary files, ~1 GB each.

**Math**:
- 1 GB = 1,073,741,824 bytes
- uint16 = 2 bytes/token
- Tokens per shard = 1,073,741,824 / 2 = 536,870,912 ≈ 500M tokens

**Implementation**:
```python
TOKENS_PER_SHARD = 500_000_000  # 500M tokens ≈ 1 GB

buffer = np.empty(TOKENS_PER_SHARD, dtype=np.uint16)
buffer_pos = 0
shard_idx = 0

doc_tokens = [BOS_ID] + token_ids + [EOS_ID]
# BOS_ID = 2  (<s>)
# EOS_ID = 3  (</s>)
# Per constitution special token table
```

**Validation**: After writing each shard → `os.path.getsize(path) > 0` and `np.fromfile(path, dtype=np.uint16)` (no exception).

---

## 4. Resumable Execution via Manifest

**Decision**: `shards_manifest.json` in the output directory.

**Format**:
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

**Resume logic**:
1. If manifest exists → load completed shard filenames as a set
2. Load Bloom filter from `bloom_filter_path` (if file exists)
3. Skip all completed shards; restore buffer position to 0 (start fresh shard)
4. Continue streaming from the next unprocessed document

**Limitation**: On resume, we cannot know where we left off in the HF stream — so we must re-stream and skip already-hashed docs via the Bloom filter. The Bloom filter itself acts as the dedup memory, so re-streaming is safe.

---

## 5. Parallelization Strategy

**Decision**: `multiprocessing.Pool` with `spawn`, `num_workers = min(cpu_count // 2, 32)`.

**Architecture** (producer-consumer):
```
HF Stream → doc buffer (1000 docs) → Pool.map(tokenize_batch) → token lists → shard writer
```

**Why spawn, not fork**: HF `datasets` streaming iterators hold file handles and internal state that don't survive `fork` safely. `spawn` creates clean child processes with only the tokenizer model loaded.

**Tokenizer loading in workers**: SentencePiece is loaded once per worker via `initializer` argument to `Pool`.

```python
import sentencepiece as spm
from multiprocessing import Pool, cpu_count

_sp = None
def _init_worker(model_path: str):
    global _sp
    _sp = spm.SentencePieceProcessor()
    _sp.Load(model_path)

def _tokenize_doc(text: str) -> list[int]:
    return _sp.EncodeAsIds(text)

workers = min(cpu_count() // 2, 32)
with Pool(workers, initializer=_init_worker, initargs=(TOKENIZER_PATH,)) as pool:
    for batch in iter_batches(stream, batch_size=1000):
        results = pool.map(_tokenize_doc, [doc["text"] for doc in batch])
```

---

## 6. Progress Reporting

**Decision**: Two-level output (C).

- **Level 1** (every 10,000 docs): `[wikipedia]  10000 docs |   8.2M tokens`
- **Level 2** (every shard): `[Shard 0042/----] 500M tokens | 12.3 GB written | elapsed 00:14:22`
- **Final**: Table with per-source token counts + total elapsed

---

## 7. Special Token IDs (Constitution Verified)

The spec originally stated BOS=1, EOS=2. **This is incorrect per the constitution.**

| Token | ID (per constitution) |
|-------|-----------------------|
| BOS `<s>` | **2** |
| EOS `</s>` | **3** |

All implementation MUST use BOS=2 and EOS=3. Spec corrected in this planning session.

---

## 8. Python Dependencies (new for this epic)

| Package | Version | Purpose |
|---------|---------|---------|
| `datasets` | ≥2.14 | HF streaming |
| `sentencepiece` | ≥0.1.99 | Tokenizer |
| `numpy` | ≥1.24 | uint16 array ops |
| `pybloom-live` | ≥4.0 | Bloom filter dedup |
| `mmh3` | ≥3.0 | Fast hashing (pybloom dep) |
| `python-dotenv` | ≥1.0 | `.env` loading |
| `torch` | ≥2.0 | ShardedDataset tensors |

---

## Summary Table

| Question | Decision | Rationale |
|----------|----------|-----------|
| Dedup scope | Cross-source Bloom filter (`pybloom-live`) | Best model accuracy; < 1 GB RAM |
| Dataset versions | Pinned: Wikipedia `20231101.tr`, OSCAR-2301 `tr`, mC4 `tr` | Reproducibility |
| Shard size | 500M tokens ≈ 1 GB | Clean GB boundary, easy Drive management |
| Resume mechanism | `shards_manifest.json` + Bloom pickle | Colab disconnect resilience |
| Parallelism | `multiprocessing.Pool(spawn)`, ≤32 workers | H100 48–128 cores, SentencePiece-safe |
| Progress | Dual: per-10k-docs + per-shard | Visibility without log spam |
| BOS/EOS | BOS=2, EOS=3 (constitution) | Frozen after Epic 1 |
