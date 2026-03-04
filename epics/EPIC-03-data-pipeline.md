# Epic 3 — Data Pipeline

| Field | Value |
|-------|-------|
| **Branch** | `epic/03-data-pipeline` |
| **Base branch** | `epic/01-tokenizer` |
| **Merge target** | `main` |
| **PRD refs** | F-03, G3, M3 |
| **Depends on** | Epic 1 (needs trained `turkish_bpe.model` to tokenize corpus) |
| **Status** | ⬜ Not started |
| **Output files** | `scripts/prepare_data.py`, `training/dataset.py`, `data/processed/shard_*.bin`, `data/processed/val.bin` |

---

## Goal

Build an end-to-end pipeline that downloads OSCAR 23.01 Turkish, mC4 Turkish, CC-100 Turkish, and Turkish Wikipedia; cleans and deduplicates at document level; tokenizes with the 64k tokenizer; packs tokens into `uint16` binary shards of exactly 20M tokens each; and orders shards for curriculum training. Output: ≥ 875 shards totalling ≥ 17.5B tokens plus a separate 50M-token validation split.

---

## Curriculum Shard Order

```
Shards 0000 – 0436  →  Phase 1: Wikipedia-only tokens  (~8.75B tokens, steps 0–16,500)
Shards 0437 – 0874  →  Phase 2: OSCAR + mC4 + CC-100  (~8.75B tokens, steps 16,500–33,000)
data/processed/val.bin  →  first 50M Wikipedia tokens (held out, never in training shards)
```

---

## Tasks

- [ ] **Download module** — stream Turkish Wikipedia via `datasets.load_dataset('wikimedia/wikipedia', '20231101.tr', streaming=True)`; stream OSCAR 23.01 via `datasets.load_dataset('oscar-corpus/OSCAR-2301', 'tr', streaming=True)`; stream mC4 via `datasets.load_dataset('mc4', 'tr', streaming=True)`; download CC-100 TR from `data.statmt.org/cc-100/tr.txt.xz` via `urllib.request`
- [ ] **Cleaning module** — per document: (1) `unicodedata.normalize('NFC', text)`, (2) strip HTML `re.sub(r'<[^>]+>', '', text)`, (3) discard if `len(text) < 200`, (4) discard if `sum(c.isalpha() and (ord(c) < 128 or c in 'çğışöüÇĞİŞÖÜ') for c in text) / len(text) < 0.50`, (5) compute `hashlib.sha256(text.encode()).hexdigest()` and skip if already seen (in-memory set)
- [ ] **Tokenization + sharding module** — tokenize each clean document with `TurkishTokenizer.encode(text)`; append token IDs to a buffer; when buffer reaches 20,000,000 tokens write as `numpy.array(buffer, dtype=numpy.uint16).tofile(f'data/processed/shard_{N:04d}.bin')`; flush partial shard at end; log total token count
- [ ] **Validation split** — before writing any training shard, reserve the first 50,000,000 tokens from Wikipedia stream to `data/processed/val.bin`; these tokens must never appear in any `shard_*.bin`
- [ ] **Curriculum shard renamer** — after all shards are written, rename/reorder so Wikipedia-only shards are numbered 0000–0436 and mixed-corpus shards are numbered 0437–0874; write a `data/processed/shard_manifest.json` mapping shard ID → source corpus
- [ ] **`training/dataset.py`** — `PretrainDataset(torch.utils.data.IterableDataset)`: constructor accepts `shard_dir: str`, `split: str` (`'train'`/`'val'`), `phase: int` (1 = shards 0–436, 2 = shards 437–874, 0 = all); opens shard files sequentially via `numpy.memmap(dtype=numpy.uint16)`; yields `(input_ids, target_ids)` both of shape `(4096,)` as `torch.long` tensors; `target_ids = input_ids[1:] cat [EOS_ID]`

---

## Exit Criteria

| Check | Pass Condition |
|-------|---------------|
| Total training tokens | ≥ 17.5B (count shards × 20M) |
| Phase 1 shards | Shards 0000–0436 contain only Wikipedia-sourced tokens (verify via `shard_manifest.json`) |
| `val.bin` isolation | No overlap with any training shard — SHA-256 spot-check on 1,000 random document hashes |
| DataLoader output | `next(iter(loader))` shapes = `(16, 4096)` and `(16, 4096)` at `batch_size=16` |
| dtype | `numpy.frombuffer(open(shard,'rb').read(), dtype=numpy.uint16).max() < 64000` |
| `val.bin` size | Exactly 50,000,000 token IDs (file size = `50_000_000 × 2 = 100 MB`) |

---

## Unlocks

- **Epic 4** (Pretraining) — needs shards + `PretrainDataset`

---

_Last updated: 4 Mart 2026_
