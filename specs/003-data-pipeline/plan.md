# Implementation Plan: Turkish Pretraining Data Pipeline

**Branch**: `003-data-pipeline` | **Date**: 2026-03-04 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `/specs/003-data-pipeline/spec.md`

---

## Summary

Build a single CLI script (`scripts/prepare_data.py`) that streams Turkish text from 3–4 HuggingFace sources (Wikipedia, OSCAR, mC4, optional CC-100), cleans and deduplicates documents cross-source via Bloom filter, tokenises with the trained SentencePiece BPE model in parallel, and writes a corpus of ≥30B tokens into flat uint16 binary shards of ~1 GB each. Resumable via `shards_manifest.json` + Bloom filter checkpoint. Also implements `training/dataset.py::ShardedDataset` for streaming during pretraining.

---

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: `datasets` ≥2.14, `sentencepiece` ≥0.1.99, `numpy` ≥1.24, `pybloom-live` ≥4.0, `python-dotenv` ≥1.0, `torch` ≥2.0  
**Storage**: Google Drive (Colab) — flat binary `.bin` shards + `shards_manifest.json` + `bloom.pkl`  
**Testing**: `pytest` — unit tests for cleaning, tokenization, ShardedDataset  
**Target Platform**: H100 server (48–128 CPU cores); also runnable on Colab CPU  
**Project Type**: CLI data pipeline + PyTorch dataset class  
**Performance Goals**: Full corpus (≥30B tokens) completed in reasonable time on H100; Wikipedia-only ≤3 hours on Colab CPU  
**Constraints**: Bloom filter memory <1 GB; streaming mode (no full materialisation); resume without data loss after disconnect  
**Scale/Scope**: ≥30B tokens, ≥1,500 shards, 3–4 sources, up to 100M unique documents

---

## Constitution Check

*GATE: Must pass before implementation begins.*

| Gate | Status | Notes |
|------|--------|-------|
| `vocab_size=64000` — token IDs in `[0, 63999]` | ✅ PASS | uint16 max 65535 > 64000 |
| BOS=2 (`<s>`), EOS=3 (`</s>`) per constitution | ✅ FIXED | Spec originally said BOS=1, EOS=2 — **corrected in this plan** |
| `max_seq_len=4096` for ShardedDataset | ✅ PASS | seq_len=4096 in all references |
| No `assert` in production code | ✅ PLAN | Use `if/raise ValueError(...)` everywhere |
| Type hints on all function signatures | ✅ PLAN | Enforced in implementation |
| Drive checkpoint path: `/content/drive/MyDrive/AUModel/` | ⚠️ NOTE | Spec uses `aumodel_checkpoints/data/` — minor path variant, not a blocker |
| No model code changes | ✅ N/A | This epic touches only data pipeline |

**No gate violations.** BOS/EOS was corrected before implementation phase.

---

## Project Structure

### Documentation (this feature)

```text
specs/003-data-pipeline/
├── spec.md              # Feature spec (clarified, BOS/EOS fixed)
├── plan.md              # This file
├── research.md          # Phase 0 — technical decisions
├── data-model.md        # Phase 1 — entity definitions + data flow
├── quickstart.md        # Phase 1 — developer quickstart
├── contracts/
│   └── shard-format.md  # Binary file format + ShardedDataset API
└── checklists/
    └── requirements.md
```

### Source Code

```text
scripts/
└── prepare_data.py      # CLI entry point — download, clean, dedup, tokenize, shard

training/
└── dataset.py           # ShardedDataset(shard_paths, seq_len) for training loop

tokenizer/
└── turkish_bpe.model    # Pre-existing — produced in Epic 1 (READ ONLY)

.env                     # HF_TOKEN (gitignored)
.env.example             # Template (committed)
```

---

## Phase 0 — Research (COMPLETE)

See [research.md](research.md). All decisions resolved:

- Bloom filter: `pybloom-live` — 100M capacity, 1% FPR, ~120 MB RAM
- Dataset versions pinned (see research.md §2)
- OSCAR `text_field = "content"` (not `"text"`)
- Shard size: 500M tokens ≈ 1 GB
- Resume: manifest + Bloom pickle
- Parallelism: `Pool(spawn)`, ≤32 workers
- BOS=2, EOS=3 (per constitution)

---

## Phase 1 — Design (COMPLETE)

Artifacts generated:
- ✅ [data-model.md](data-model.md) — Document, Source, Shard, Manifest, Corpus entities
- ✅ [contracts/shard-format.md](contracts/shard-format.md) — Binary format + ShardedDataset API
- ✅ [quickstart.md](quickstart.md) — Developer quickstart

---

## Implementation Roadmap

### Task Group 1: Core Pipeline (`scripts/prepare_data.py`)

| # | Task | FR | File |
|---|------|----|------|
| 1.1 | CLI arg parsing (`--source`, `--output`, `--tokenizer`) | FR-001 | `prepare_data.py` |
| 1.2 | `.env` loading + `HF_TOKEN` validation | FR-002 | `prepare_data.py` |
| 1.3 | Source config table (name → hf_id, config, text_field) | FR-001 | `prepare_data.py` |
| 1.4 | HF streaming iterator per source | FR-003 | `prepare_data.py` |
| 1.5 | Cleaning function: strip HTML + NFC normalize | FR-004 | `prepare_data.py` |
| 1.6 | Bloom filter init / load from checkpoint | FR-005 | `prepare_data.py` |
| 1.7 | `multiprocessing.Pool(spawn)` tokenizer worker | FR-003b, FR-006 | `prepare_data.py` |
| 1.8 | Shard buffer + write logic (`shard_NNNN.bin`) | FR-007 | `prepare_data.py` |
| 1.9 | Manifest read/write for resume | FR-008 | `prepare_data.py` |
| 1.10 | Shard validation after write | FR-010 | `prepare_data.py` |
| 1.11 | Dual-level progress reporting | FR-009 | `prepare_data.py` |
| 1.12 | Drive space check before each shard write | US2 AC3 | `prepare_data.py` |
| 1.13 | Network retry with exponential backoff | US2 AC2 | `prepare_data.py` |

### Task Group 2: ShardedDataset (`training/dataset.py`)

| # | Task | FR | File |
|---|------|----|------|
| 2.1 | `ShardedDataset.__init__` — validate + memmap all shards | FR-011, FR-012 | `dataset.py` |
| 2.2 | `__len__` — total windows across all shards | FR-011 | `dataset.py` |
| 2.3 | `__getitem__` — (input_ids, target_ids) with left shift | FR-011 | `dataset.py` |
| 2.4 | Multi-worker safety (each worker opens its own memmap) | US3 AC3 | `dataset.py` |

### Task Group 3: Tests

| # | Task | Covers |
|---|------|--------|
| 3.1 | Unit test: `clean_document()` — HTML strip, NFC, min-length | FR-004 |
| 3.2 | Unit test: Bloom filter — duplicate detection, checkpoint round-trip | FR-005 |
| 3.3 | Unit test: `ShardedDataset` — shift invariant, ValueError on bad shard | FR-011, FR-012 |
| 3.4 | Integration test: fake 1000-doc stream → 1 shard — verify token count | SC-006 |
| 3.5 | Resume test: interrupt mid-shard → restart → same final output | SC-005 |

---

## Special Token Reference (Constitution-Locked)

| Token | String | ID |
|-------|--------|----|  
| PAD | `<pad>` | 0 |
| UNK | `<unk>` | 1 |
| BOS | `<s>` | **2** |
| EOS | `</s>` | **3** |

All shard writing code MUST use `BOS_ID = 2` and `EOS_ID = 3`.

---

## Key Constants

```python
TOKENIZER_PATH  = "tokenizer/turkish_bpe.model"
TOKENS_PER_SHARD = 500_000_000   # ≈ 1 GB as uint16
MIN_DOC_TOKENS  = 100
BATCH_SIZE      = 1_000          # docs per pool task
BLOOM_CAPACITY  = 100_000_000
BLOOM_ERROR     = 0.01
BOS_ID          = 2              # per constitution
EOS_ID          = 3              # per constitution
VOCAB_SIZE      = 64_000
MAX_WORKERS     = min(os.cpu_count() // 2, 32)
```

---

## Dependency Notes

```bash
# New packages required for this epic
pip install pybloom-live mmh3
# Already present from prior epics
pip install datasets sentencepiece numpy torch python-dotenv
```

---

## Constitution Re-check (Post-Design)

All constitution invariants satisfied:
- ✅ BOS=2, EOS=3 used throughout design and contracts
- ✅ Token IDs in `[0, 63999]` — enforced via `if token_id >= VOCAB_SIZE: raise ValueError`
- ✅ `if/raise` instead of `assert` throughout
- ✅ Type hints on all function signatures
- ✅ Drive paths support `/content/drive/MyDrive/...`
- ✅ Resume works after disconnect (manifest + Bloom pickle)

