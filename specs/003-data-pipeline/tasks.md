# Tasks: Turkish Pretraining Data Pipeline

**Feature**: `003-data-pipeline`  
**Branch**: `003-data-pipeline`  
**Input**: [plan.md](plan.md), [spec.md](spec.md), [data-model.md](data-model.md), [contracts/shard-format.md](contracts/shard-format.md)

---

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files or independent concerns)
- **[US1/US2/US3]**: User story scope
- All paths relative to repo root

---

## Phase 1: Setup

**Purpose**: Install new dependencies and scaffold empty source files.

- [ ] T001 Install new dependencies: `pip install pybloom-live mmh3` and verify import in Python 3.11
- [ ] T002 [P] Scaffold `scripts/prepare_data.py` with module-level constants: `BOS_ID=2`, `EOS_ID=3`, `TOKENS_PER_SHARD=500_000_000`, `MIN_DOC_TOKENS=100`, `BATCH_SIZE=1_000`, `BLOOM_CAPACITY=100_000_000`, `BLOOM_ERROR=0.01`, `VOCAB_SIZE=64_000`

---

## Phase 2: Foundational

**Purpose**: Shared building blocks required by ALL user stories. Must complete before US1/US2/US3.

**⚠️ CRITICAL**: US1 and US2 cannot proceed without T003–T004.

- [ ] T003 Add `SOURCES` config dict in `scripts/prepare_data.py` mapping each source name to `{hf_id, config, split, text_field, requires_auth}` per the pinned versions in [data-model.md](data-model.md): wikipedia=`wikimedia/wikipedia`/config=`20231101.tr`/split=`"train"`/`"text"`, oscar=`oscar-corpus/OSCAR-2301`/config=`tr`/split=`"train"`/`"content"`, mc4=`allenai/c4`/config=`tr`/split=`"train"`/`"text"`, cc100=`cc100`/config=`tr`/split=`"train"`/`"text"`
- [ ] T004 Implement `load_hf_token() -> str | None` in `scripts/prepare_data.py`: load `.env` via `python-dotenv`, return `HF_TOKEN` value; raise `ValueError` with clear message if a source requires auth and token is absent

**Checkpoint**: Foundation ready — US1, US2, US3 can now begin (US3 is fully independent).

---

## Phase 3: User Story 1 — Single-Source Pipeline (Priority: P1) 🎯 MVP

**Goal**: Run `python scripts/prepare_data.py --source wikipedia --output <dir>` end-to-end, produce verified shard files, print token summary.

**Independent Test**: `python scripts/prepare_data.py --source wikipedia --output /tmp/test_shards/` → at least one `shard_0000.bin` written, token count printed, `ShardedDataset` reads it back without errors.

- [ ] T005 [US1] Implement CLI arg parsing in `scripts/prepare_data.py`: `--source` (choices: wikipedia, oscar, mc4, cc100, all), `--output` (str, default `"/content/drive/MyDrive/AUModel/data/"` per constitution checkpoint path), `--tokenizer` (str, default `tokenizer/turkish_bpe.model`)
- [ ] T006 [US1] Implement `clean_document(text: str) -> str | None` in `scripts/prepare_data.py`: strip HTML tags with regex, apply `unicodedata.normalize("NFC", text)`, return `None` if result is empty string (short-circuit before tokenization)
- [ ] T007 [US1] Implement `init_bloom(output_dir: str) -> BloomFilter` in `scripts/prepare_data.py`: load from `bloom.pkl` if it exists in output_dir (resume), else create new `BloomFilter(BLOOM_CAPACITY, BLOOM_ERROR)`; implement `save_bloom(bloom, path: str) -> None` using `pickle.dump`
- [ ] T008 [US1] Implement `load_manifest(output_dir: str) -> dict` and `save_manifest(manifest: dict, output_dir: str) -> None` in `scripts/prepare_data.py`: read/write `shards_manifest.json` per the schema in [data-model.md](data-model.md); return empty manifest dict if file does not exist
- [ ] T009 [US1] Implement `_init_worker(model_path: str) -> None` and `_tokenize_doc(text: str) -> list[int]` module-level functions in `scripts/prepare_data.py` for `multiprocessing.Pool(spawn)`: worker loads SentencePiece once per process; `_tokenize_doc` returns `[BOS_ID] + sp.EncodeAsIds(text) + [EOS_ID]`
- [ ] T010 [US1] Implement `tokenize_batch(texts: list[str], pool: Pool) -> list[list[int]]` in `scripts/prepare_data.py`: calls `pool.map(_tokenize_doc, texts)`, filters results where `len(ids) < MIN_DOC_TOKENS`, validates all IDs in `[0, VOCAB_SIZE)` raising `ValueError` if any are out of range
- [ ] T011 [US1] Implement `ShardWriter` class in `scripts/prepare_data.py`: holds a `np.ndarray(TOKENS_PER_SHARD, dtype=np.uint16)` buffer and `pos` counter; `write_tokens(ids)` fills buffer; `flush(output_dir, shard_idx, source_counts)` writes `shard_NNNN.bin` via `np.ndarray.tofile`, then validates (size > 0, `np.fromfile` round-trip), updates manifest and saves Bloom checkpoint; deletes and re-raises on validation failure
- [ ] T012 [US1] Implement `stream_source(source_name: str, hf_token: str | None) -> Iterator[str]` in `scripts/prepare_data.py`: calls `load_dataset(hf_id, config, split="train", streaming=True, token=hf_token)`, yields `record[text_field]` for each record
- [ ] T013 [US1] Implement `run_pipeline(sources: list[str], output_dir: str, tokenizer_path: str)` main loop in `scripts/prepare_data.py`: initialize Bloom + manifest + Pool(spawn, initializer=_init_worker); for each source stream documents → `clean_document` → dedup via Bloom → accumulate batch of `BATCH_SIZE` → `tokenize_batch` → `ShardWriter.write_tokens`; flush final partial shard on completion; after processing each source print `[WARNING] Source <name> produced 0 tokens — all documents filtered or source empty` if that source contributed zero tokens
- [ ] T014 [US1] Add dual-level progress reporting inside `run_pipeline` in `scripts/prepare_data.py`: print `[<source>] <N> docs | <M>M tokens` every 10,000 docs; print `[Shard NNNN/----] <tokens>M tokens | <GB> GB written | elapsed HH:MM:SS` after each `ShardWriter.flush`
- [ ] T015 [US1] Add final summary print at end of `run_pipeline` in `scripts/prepare_data.py`: total shards, total tokens, per-source token breakdown table, total elapsed time; add `if __name__ == "__main__":` entry point calling `run_pipeline`

---

## Phase 4: User Story 2 — Full Multi-Source Corpus (Priority: P2)

**Goal**: `--source all` streams all 4 sources sequentially, reaches ≥30B tokens across ≥1,500 shards, handles network failures and Drive space exhaustion gracefully.

**Independent Test**: Run `python scripts/prepare_data.py --source all --output <dir>` to completion. Confirm `len(shards) >= 1500` and `total_tokens >= 30_000_000_000` in manifest.

**Dependency**: US1 must be complete. `stream_source`, `run_pipeline`, `ShardWriter` from US1 are extended here.

- [ ] T016 [US2] Extend `run_pipeline` in `scripts/prepare_data.py` to accept `sources=["wikipedia","oscar","mc4","cc100"]` when `--source all`; iterate sources in order, tracking per-source token counts in the manifest `source_totals` field throughout the run
- [ ] T017 [P] [US2] Implement `retry_stream(source_name, hf_token, max_retries=3)` decorator/wrapper in `scripts/prepare_data.py`: on `requests.exceptions.ConnectionError` or `datasets.exceptions.DatasetGenerationError`, wait `2**attempt` seconds and retry; after 3 failures log warning and skip source
- [ ] T018 [P] [US2] Implement `check_drive_space(output_dir: str, min_gb: float = 50.0) -> None` in `scripts/prepare_data.py`: call `shutil.disk_usage(output_dir).free`; if free bytes < `min_gb * 1e9` print warning and `raise SystemExit(1)` without corrupting existing shards; call this inside `ShardWriter.flush` before writing
- [ ] T019 [US2] Update `ShardWriter.flush` and manifest logic in `scripts/prepare_data.py` to track per-source token contribution per shard (`source_counts` dict) and accumulate into `manifest["source_totals"]`; update final summary print to use real per-source totals

---

## Phase 5: User Story 3 — ShardedDataset for Training (Priority: P3)

**Goal**: `ShardedDataset(shard_paths, seq_len=4096)` yields `(input_ids, target_ids)` pairs of shape `(4096,)` with correct left-shift, safe for `DataLoader(num_workers=4)`.

**Independent Test**: Instantiate on any single `shard_0000.bin`, fetch 10 items, assert `target_ids[t] == input_ids[t+1]` for all `t < 4095`.

**Dependency**: None (can be implemented before US1/US2 complete; test against any fake shard).

- [ ] T020 [P] [US3] Implement `ShardedDataset.__init__(self, shard_paths: list[str], seq_len: int)` in `training/dataset.py`: for each path raise `ValueError` if missing, empty, or `byte_size % 2 != 0`; store shard sizes (token counts) and compute cumulative offset array for O(1) index lookup; do NOT memmap at init (open per-access for worker safety)
- [ ] T021 [P] [US3] Implement `ShardedDataset.__len__(self) -> int` in `training/dataset.py`: return `total_tokens // seq_len` where `total_tokens` is sum of all shard token counts
- [ ] T022 [US3] Implement `ShardedDataset.__getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.LongTensor]` in `training/dataset.py`: compute global token offset `idx * seq_len`, map to shard via cumulative offset array, open shard as `np.memmap(..., dtype="uint16", mode="r")`, slice `seq_len + 1` tokens (spanning shard boundary if needed), split into `input_ids[0:seq_len]` and `target_ids[1:seq_len+1]`, return as `torch.LongTensor` pair
- [ ] T023 [P] [US3] Verify `DataLoader(dataset, num_workers=4)` safety in `training/dataset.py`: since each `__getitem__` opens its own memmap, no shared state exists between workers; add comment documenting this; write a quick `assert` check in `__init__` that each shard file is readable (no lock)

---

## Phase 6: Polish & Cross-Cutting

- [ ] T024 Add `requirements-data.txt` at repo root listing new dependencies: `pybloom-live>=4.0`, `mmh3>=3.0`, `datasets>=2.14`, `sentencepiece>=0.1.99`; update [quickstart.md](quickstart.md) install command to reference it
- [ ] T025 Add token ID range guard inside `_tokenize_doc` in `scripts/prepare_data.py`: after `sp.EncodeAsIds`, iterate IDs and `raise ValueError(f"Token ID {id} out of range [0, {VOCAB_SIZE})")` if any ID >= VOCAB_SIZE; inner guard for early detection — outer batch-level check is in T010

---

## Phase 7: Tests

**Purpose**: Verify correctness of cleaning, dedup, tokenization, sharding, and resume. Covers SCs and plan.md Task Group 3.

- [ ] T026 [P] Unit test `clean_document()` in `tests/test_prepare_data.py`: assert HTML tags stripped, NFC normalization applied, returns `None` for empty string short-circuit; covers FR-004
- [ ] T027 [P] Unit test Bloom filter in `tests/test_prepare_data.py`: assert duplicate doc hash is detected on second call, Bloom checkpoint round-trips correctly via `pickle.dump`/`load`; covers FR-005
- [ ] T028 [P] [US3] Unit test `ShardedDataset` in `tests/test_dataset.py`: assert shift invariant `target_ids[t] == input_ids[t+1]` for all `t < seq_len-1`; assert `ValueError` raised on missing shard, empty shard, and odd byte-size shard; covers FR-011, FR-012, SC-004
- [ ] T029 Integration test in `tests/test_pipeline.py`: generate a synthetic 1,000-doc stream → run `run_pipeline` → confirm ≥1 shard written, token count > 0, manifest is valid JSON with correct `total_tokens`; covers plan task 3.4
- [ ] T030 Resume test in `tests/test_pipeline.py`: run pipeline against synthetic stream until 1 shard completes, then interrupt (inject exception), restart → confirm final corpus is identical to an uninterrupted run (same shard checksum); covers SC-005, plan task 3.5
- [ ] T031 [P] SC-006 spot-check script `tests/check_token_range.py`: given a shard directory, sample up to 100 random `.bin` files, load each as `np.fromfile(dtype=np.uint16)`, assert all values in `[0, VOCAB_SIZE)`, print pass/fail per shard; standalone — run manually after full corpus build; covers SC-006

---

## Dependencies

```
Phase 1 (Setup) → Phase 2 (Foundational) → Phase 3 (US1) → Phase 4 (US2)
                                          ↘                        ↘
                                           Phase 5 (US3)            Phase 7 (Tests)
                                                    ↘              ↗
                                                     ← independent
```

- **US2 requires US1 complete**: `stream_source`, `ShardWriter`, `run_pipeline` must exist before extending them
- **US3 is fully independent**: `training/dataset.py` only reads `.bin` files; can develop and test against a fake shard
- **Phase 7 can begin after Phase 3**: T026/T027/T028 can be written alongside US1; T029/T030 need US1 complete; T031 needs full corpus
- **T017 ∥ T018**: Both are US2 tasks touching different functions in `prepare_data.py` — can be done in parallel by different developers
- **T020 ∥ T021 ∥ T023**: All in `training/dataset.py`, touching different methods — parallelizable
- **T026 ∥ T027 ∥ T028 ∥ T031**: All test tasks in different files — fully parallelizable

---

## Parallel Execution Examples

### Developer A + Developer B (US1 in parallel with US3)

After Phase 2 is complete:

```
Dev A: T005 → T006 → T007 → T009 → T010 → T011 → T012 → T013 → T014 → T015
Dev B: T020 → T021 → T022 → T023
```

Dev B can work entirely on `training/dataset.py` with a synthetic 2MB test shard — no dependency on Dev A.

### US2 Extensions (after US1)

```
Dev A: T016 → T019
Dev B: T017 ∥ T018 (parallel — different functions in same file)
```

---

## Implementation Strategy — MVP First

**Suggested delivery order**:
1. **T001–T004** (Phase 1+2) — ~30 min setup
2. **T005–T015** (US1) — Wikipedia pipeline working end-to-end ← **verify this before proceeding**
3. **T020–T023** (US3) — `ShardedDataset` ready for training loop ← **Epic 4 unblocked here**
4. **T026–T028** (Tests) — unit tests for clean, bloom, ShardedDataset (can run alongside step 2-3)
5. **T016–T019** (US2) — Multi-source scale-out
6. **T029–T030** (Tests) — integration + resume tests
7. **T024–T025** (Polish)
8. **T031** (SC-006 spot-check) — run after full corpus build

---

## Task Summary

| Phase | Tasks | Story | Count |
|-------|-------|-------|-------|
| Phase 1: Setup | T001–T002 | — | 2 |
| Phase 2: Foundational | T003–T004 | — | 2 |
| Phase 3: US1 | T005–T015 | [US1] | 11 |
| Phase 4: US2 | T016–T019 | [US2] | 4 |
| Phase 5: US3 | T020–T023 | [US3] | 4 |
| Phase 6: Polish | T024–T025 | — | 2 |
| Phase 7: Tests | T026–T031 | — | 6 |
| **Total** | | | **31** |

**Parallel opportunities**: 12 tasks marked [P] across US1/US2/US3/Tests phases  
**Independent test criteria**: Each story has its own runnable verification (see phase headers)  
**MVP scope**: Phases 1–3 (T001–T015) — Wikipedia pipeline working end-to-end
