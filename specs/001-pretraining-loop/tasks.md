# Tasks: AUModel Pretraining Loop

**Branch**: `001-pretraining-loop`  
**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)  
**Generated**: 4 Mart 2026  
**Total tasks**: 20

## Format: `- [ ] [ID] [P?] [Story?] Description with file path`

- **[P]**: Can run in parallel (different files, no incomplete-task dependencies)
- **[US1–US4]**: User story label (phases 3–6 only)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the synthetic test fixture needed by all test phases.

- [ ] T001 Create synthetic test fixture shard in tests/fixtures/data/wikipedia/shard_00000.bin — generate as `np.random.randint(0, 64000, size=(100_352,), dtype=np.uint16)` (explicitly random-uniform so SC-001 initial-loss test computes ≈ ln(64000) ≈ 11.07; a constant-token fixture would produce loss ≈ 0); save with `arr.tofile(path)`

---

## Phase 2: Foundational (Blocking Prerequisite)

**Purpose**: The LR scheduler is a pure function consumed by every other story. MUST be complete before Phase 3 begins.

⚠️ **CRITICAL**: No user story implementation can start until T002 is complete.

- [ ] T002 Implement `get_lr(step, warmup_steps, max_steps, max_lr, min_lr) -> float` in training/lr_scheduler.py — linear warmup, cosine decay, min_lr hold; clamp at step<0 and step>max_steps; full type hints; no assert statements

**Checkpoint**: `get_lr` importable and returns finite float for all input ranges

---

## Phase 3: User Story 1 — Run a Complete Pretraining Session (P1) 🎯 MVP

**Goal**: The trainer starts from scratch, runs for N steps, logs metrics every `log_interval` steps, saves a checkpoint, and exits cleanly.

**Independent Test**: `python -m pytest tests/test_lr_scheduler.py tests/test_trainer.py::test_smoke tests/test_trainer.py::test_initial_loss -v` — all must pass on local CPU.

### Tests for User Story 1

- [ ] T003 [P] [US1] Write tests/test_lr_scheduler.py — SC-005: test warmup region (step<2000 → proportional), decay region (step midpoint → between min/max), hold region (step>100000 → exactly min_lr), boundary values (get_lr(0)=0, get_lr(2000)=max_lr, get_lr(100000)=min_lr, get_lr(100001)=min_lr), negative step clamped

### Implementation for User Story 1

- [ ] T004 [US1] Implement `TrainingConfig` dataclass + `parse_args()` CLI in training/trainer.py — all fields from data-model.md (micro_batch_size=32, grad_accum_steps=4, seq_len=4096, max_steps=100000, warmup_steps=2000, max_lr=3e-4, min_lr=3e-5, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, grad_clip=1.0, log_interval=10, checkpoint_interval=1000, checkpoint_keep=10, val_interval=1000, use_wandb=True, gradient_checkpointing=False); emit `warnings.warn` (NOT ValueError) when `micro_batch_size × grad_accum_steps ≠ 128` — this preserves test configs (e.g. 2×2=4) while flagging non-H100 defaults; raise ValueError only for clearly invalid fields (micro_batch_size<1, grad_accum_steps<1, max_lr≤min_lr, warmup_steps≥max_steps); full type hints
- [ ] T005 [P] [US1] Implement `save_checkpoint(path, state)` with atomic write (write to .tmp then os.rename) in training/checkpoint.py — state dict includes model_state, optimizer_state, step, config, loader_state, best_val_loss, created_at; full type hints
- [ ] T006 [US1] Implement `SourceState` dataclass + `InterleavedShardLoader(IterableDataset)` in training/trainer.py — weighted-random source selection, infinite cycling with per-cycle reshuffle (seed=base_seed+cycle×1000), per-source shard_idx/sample_idx/cycle_count tracking, state_dict()/load_state_dict() methods for resume; wraps `ShardedDataset` from training/dataset.py; default weights: wikipedia=0.20, oscar=0.30, mc4=0.30, cc100=0.20
- [ ] T007 [US1] Implement `estimate_mfu(tokens_per_sec, model_params, seq_len, peak_flops, num_layers, num_heads, d_model) -> float` in training/trainer.py — formula: `(6×N + 12×L×T×H×d_head) × tokens_per_sec / peak_flops`; defaults: model_params=749_544_960, seq_len=4096, peak_flops=989e12, num_layers=24, num_heads=12, d_model=1536
- [ ] T008 [US1] Implement core `train(cfg: TrainingConfig) -> None` in training/trainer.py — device setup (cuda:0 or cpu), TF32 flags, SDPA backend flags (enable_flash_sdp/enable_mem_efficient_sdp), model = AUModel(ModelConfig()) loaded and compiled with torch.compile, AdamW optimiser, BF16 autocast context for forward+loss, compute loss via `F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))` (no ignore_index — all tokens active in pretraining), grad accum loop (loss /= grad_accum_steps per micro-step), grad clip max_norm=1.0, LR scheduler step each optimiser update, every checkpoint_interval steps: call `save_checkpoint(ckpt_path, state)` **then immediately** call `rotate_checkpoints(cfg.output_dir, keep=cfg.checkpoint_keep)` (FR-007: auto-delete oldest — these two calls are always paired), basic console print of step+loss+lr; gradient_checkpointing applied if cfg.gradient_checkpointing; __main__ entry point calling parse_args()+train()
- [ ] T009 [US1] Write tests/test_trainer.py — smoke test: run train() for 50 steps with `micro_batch_size=2, grad_accum_steps=2, seq_len=64, checkpoint_interval=10, log_interval=10, val_interval=10000, no_wandb`, fixture shard; confirm exit code 0 and at least one checkpoint file (`step_000010.pt`) exists in output_dir (SC-004); **note**: `checkpoint_interval=10` is required so the 50-step run actually triggers a save — the default of 1000 would never fire; initial loss test: assert step-1 loss ∈ [10.5, 11.6] confirming correct init (SC-001); `estimate_mfu` unit test: `assert estimate_mfu(0.0) == 0.0`, `assert 0.10 <= estimate_mfu(100_000) <= 0.70` (hardware-agnostic sanity range — catches wrong formula constants without requiring H100)

**Checkpoint**: US1 independently functional. Run `python -m pytest tests/test_lr_scheduler.py tests/test_trainer.py::test_smoke tests/test_trainer.py::test_initial_loss -v` — all pass.

---

## Phase 4: User Story 2 — Resume Training After Interruption (P1)

**Goal**: After a session is killed mid-training, relaunching detects the latest checkpoint and continues from the next step with no data repeated.

**Independent Test**: `python -m pytest tests/test_checkpoint.py tests/test_trainer.py::test_resume -v` — all pass on local CPU.

### Tests for User Story 2

- [ ] T010 [P] [US2] Write tests/test_checkpoint.py — SC-006: save checkpoint, reload with `load_latest_checkpoint`, assert model produces identical logits for same input; rotate test: save 11 checkpoints, assert only 10 remain and the oldest (step_00001) is deleted; loader state test: capture `loader.state_dict()` before `save_checkpoint`, reload via `load_latest_checkpoint`, assert restored loader state equals captured state field-by-field (shard_idx, sample_idx, cycle_count) — confirms no token repetition on resume
- [ ] T011 [P] [US2] Add `test_resume` to tests/test_trainer.py — run 20 steps with `checkpoint_interval=10, log_interval=5, val_interval=10000` so that a checkpoint is saved at step 10 and step 20; record the loss at step 20; kill the process (return early); restart with same output_dir and same config, assert **first logged step is 21** (loader restores from step_000020.pt), assert loss at step 21 is within 0.01 of non-interrupted baseline (SC-003); **note**: `checkpoint_interval=10` is mandatory — without it no checkpoint is saved within the 20-step run and the restart would start from step 0

### Implementation for User Story 2

- [ ] T012 [US2] Implement `load_latest_checkpoint(output_dir) -> dict | None` + `rotate_checkpoints(ckpt_dir, keep=10)` in training/checkpoint.py — glob `step_*.pt`, sort by step number, load highest; delete oldest when count > keep; return None if directory empty (with info print)
- [ ] T013 [US2] Wire resume into `train()` in training/trainer.py: at startup call `load_latest_checkpoint`, restore model+optimizer state_dict, restore InterleavedShardLoader via load_state_dict, set step=checkpoint["step"]+1; print `[Resume] Loaded step_N.pt — resuming from step N+1` or `[Resume] No checkpoint found — starting from step 0`

**Checkpoint**: US2 independently functional. Run `python -m pytest tests/test_checkpoint.py tests/test_trainer.py::test_resume -v` — all pass.

---

## Phase 5: User Story 3 — Monitor Training Progress (P2)

**Goal**: Every `log_interval` steps, a complete metrics line is printed to console and (optionally) synced to WandB. Validation loss appears every `val_interval` steps.

**Independent Test**: `python -m pytest tests/test_trainer.py::test_logging tests/test_trainer.py::test_val_loss -v` — verify 4 log lines for 20-step run and SC-007 val_loss is finite.

### Tests for User Story 3

- [ ] T014 [P] [US3] Add `test_logging` and `test_val_loss` to tests/test_trainer.py — log test: run 20 steps with log_interval=5, capture stdout, assert exactly 4 lines each containing all **7 fields** (step, loss, val_loss, lr, grad_norm, tok/s, mfu); in this log-format test assert `val_loss=-` (dash, not numeric) because val_interval defaults to 1000 and will not fire within 20 steps; assert mfu is finite and >0; val_loss test: use val_interval=10 (or mock the validation call), run enough steps to trigger it, assert val_loss is a finite float and |val_loss − train_loss| < 2.0 (SC-007)

### Implementation for User Story 3

- [ ] T015 [P] [US3] Implement `Logger` class in training/trainer.py — try/except wandb import; check WANDB_API_KEY; `__init__(enabled, project, run_name)` calls wandb.init if available; `log(metrics: dict, step: int)` calls wandb.log or prints formatted line; `finish()` calls wandb.finish; graceful fallback to console-only on any wandb failure
- [ ] T016 [US3] Implement validation loss pass in `train()` in training/trainer.py — every val_interval steps: load val shard via ShardedDataset, iterate with torch.no_grad(), compute mean CE loss across val batches, store in best_val_loss, pass val_loss to Logger; skip with warning if cfg.val_shard is None
- [ ] T017 [US3] Replace basic console print in train() with full Logger log line in training/trainer.py — format: `step=N  loss=F  val_loss=F|-  lr=F  grad_norm=F  tok/s=F  mfu=F%  elapsed=Fs`; wire estimate_mfu() and elapsed time tracking; replace stub Logger with full Logger class from T015

**Checkpoint**: US3 independently functional. Run `python -m pytest tests/test_trainer.py::test_logging tests/test_trainer.py::test_val_loss -v` — all pass.

---

## Phase 6: User Story 4 — Launch Training from Colab Notebook (P2)

**Goal**: `colab/02_pretrain.ipynb` runs cell-by-cell on a fresh H100 Colab session, mounts Drive, installs all dependencies, configures hardware, and starts or resumes training.

**Independent Test**: Manual test on Colab H100 — run all cells top-to-bottom, final cell prints first training log line without errors.

### Implementation for User Story 4

- [ ] T018 [US4] Create colab/02_pretrain.ipynb with 7 cells: (1) Drive mount + path config, (2) `pip install flash-attn --no-build-isolation wandb tqdm -q`, (3) TF32 + GPU info + SDPA backend config, (4) git clone/pull + sys.path setup, (5) model sanity check (`model/sanity_check.py`), (6) `train(parse_args())` call with Drive shard-dir + output-dir defaults, (7) WandB run URL display; each cell includes a descriptive markdown header; resume is automatic (no separate resume cell needed)

**Checkpoint**: US4 complete — notebook ready for H100. Verify manually on Colab or confirm cells are syntactically valid locally.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T019 [P] Update training/__init__.py to export `TrainingConfig` and `get_lr` for downstream use in SFT epic
- [ ] T020 Run full test suite `python -m pytest tests/test_lr_scheduler.py tests/test_checkpoint.py tests/test_trainer.py -v` and confirm all automated SCs pass: SC-001 (init loss), SC-003 (resume delta), SC-004 (smoke 5min), SC-005 (LR boundaries), SC-006 (checkpoint round-trip + loader state), SC-007 (val_loss finite); fix any failures; **manual verification on H100 required for**: SC-002 (MFU ≥35%, throughput ≥100k tok/s) and SC-008 (VRAM ≥70 GB with micro_batch_size=32) — document results in a comment in this task after the Colab run

---

## Dependencies

```
T001 (fixture) → T009, T011, T013 (all tests that use fixture shards)
T002 (get_lr)  → T003 (tests can be written before, but need impl to pass), T004+
T004 (TrainingConfig) → T006, T007, T008
T005 (save_checkpoint) → T008 (called in train loop), T010 (round-trip test)
T006 (InterleavedShardLoader) → T008, T011, T013
T007 (estimate_mfu) → T008, T014
T008 (core train loop) → T009, T012, T013, T016, T017
T012 (load_latest_checkpoint) → T013 (resume logic wired into train())
T015 (Logger class) → T017 (replaces stub logger in train())
T016 (val pass) → T017 (val_loss fed into log line)
```

## Parallel Execution Examples Per Story

**US1 (Phase 3)**: 3 agents simultaneously:
- Agent A: T003 (test_lr_scheduler.py)
- Agent B: T004 (TrainingConfig) → T006 (InterleavedShardLoader) → T007 (estimate_mfu) → T008 (core loop) → T009 (tests)
- Agent C: T005 (checkpoint.py save)

**US2 (Phase 4)**: 2 agents simultaneously:
- Agent A: T010 (test_checkpoint.py) + T011 (test_resume)
- Agent B: T012 (load+rotate) → T013 (resume logic)

**US3 (Phase 5)**: 2 agents simultaneously:
- Agent A: T014 (logging tests) + T015 (Logger class)
- Agent B: T016 (val pass) → T017 (wire full log line, depends on T015+T016)

## Implementation Strategy

**MVP (minimum to deliver value)**: Phase 1 + Phase 2 + Phase 3 only (T001–T009). This alone gives a working, testable pretraining loop that runs on CPU and H100.

**Full delivery**: All 7 phases. Each phase adds an independently verifiable capability on top of the MVP.

**Colab workflow**: Implement and test Phases 1–6 locally using fixture shards (CPU); only T018 (notebook) and SC-002/SC-008 (hardware metrics) require actual H100.
