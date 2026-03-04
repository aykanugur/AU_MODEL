# Implementation Plan: AUModel Pretraining Loop

**Branch**: `001-pretraining-loop` | **Date**: 4 Mart 2026 | **Spec**: [spec.md](spec.md)

## Summary

Build the pretraining training loop for AUModel (749.5M parameter Turkish LLM). The implementation consists of three new Python modules — `training/trainer.py` (main loop), `training/lr_scheduler.py` (cosine LR with warmup), and `training/checkpoint.py` (save/load/rotate) — plus `colab/02_pretrain.ipynb` for H100 execution. The trainer uses BF16 mixed precision, gradient accumulation (micro_batch=32 × grad_accum=4 = effective batch 128), interleaved multi-source shard loading (anti-forgetting), validation loss evaluation every 1000 steps, and seamless Drive-backed resume. `torch.compile` is applied after model init; flash attention is activated automatically via PyTorch SDPA backend flags.

## Technical Context

**Language/Version**: Python 3.11 (Colab H100 runtime); Python 3.14.2 (.venv, local dev/test)  
**Primary Dependencies**: `torch>=2.1`, `numpy`, `tqdm`, `wandb` (optional), `pytest` (test)  
**Storage**: Google Drive (`/content/drive/MyDrive/AUModel/checkpoints/`) for checkpoints; local disk for shard files (read-only `np.memmap`)  
**Testing**: `pytest` — unit tests for LR scheduler, checkpoint round-trip, MFU function, InterleavedShardLoader; integration smoke test (50 steps, synthetic shard)  
**Target Platform**: CUDA (H100 SXM5 80 GB) on Colab; CPU for local dev and CI  
**Project Type**: CLI training script (`python -m training.trainer`) + Jupyter notebook  
**Performance Goals**: MFU ≥ 35% on H100 (target 38–48%); throughput ≥ 100k tokens/sec; VRAM utilisation ≥ 70 GB  
**Constraints**: BF16 only (no FP16); single GPU cuda:0 (no DDP/FSDP); `torch.compile` required; `allow_tf32=True` required; all checkpoints to Drive  
**Scale/Scope**: 100,000 optimiser steps; ~30B tokens; 749.5M parameters; 4,096 token sequences

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Rule | Status | Notes |
|------|--------|-------|
| MUST use BF16 — FP16 forbidden | ✅ PASS | FR-005, `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` |
| MUST call `torch.compile(model)` after init | ✅ PASS | FR-006, called once in trainer startup |
| MUST set `allow_tf32 = True` | ✅ PASS | Set in trainer preamble and Colab notebook |
| MUST save checkpoints to Drive | ✅ PASS | FR-007/FR-008; output_dir defaults to Drive path |
| MUST support seamless resume | ✅ PASS | FR-008; auto-detects latest checkpoint |
| MUST NOT use `assert` in production | ✅ PASS | Use `if/raise ValueError` throughout |
| MUST use type hints on all functions | ✅ PASS | All public signatures annotated |
| `bias=False` on nn.Linear | ✅ N/A | Already enforced in `model/` (Epic 2) — no new linear layers in trainer |
| RoPE not absolute PE | ✅ N/A | Already in `model/` (Epic 2) |
| RMSNorm not LayerNorm | ✅ N/A | Already in `model/` (Epic 2) |
| `ignore_index=-100` | ✅ N/A | SFT only (Epic 5) — not applicable to pretraining |
| BOS=2, EOS=3 | ✅ N/A | Already embedded in shards by Epic 3 pipeline |

**Constitution verdict**: All applicable rules PASS. No violations. No complexity tracking required.

## Post-Design Constitution Re-check

After Phase 1 design:

| Additional check | Status |
|-----------------|--------|
| `TrainingConfig.micro_batch_size × grad_accum_steps = 128` (effective batch invariant) | ✅ PASS (32×4) |
| `InterleavedShardLoader` uses ONLY `ShardedDataset` from Epic 3 — no new shard format | ✅ PASS |
| Checkpoint write uses atomic rename — no partial writes on Drive | ✅ PASS |
| `estimate_mfu()` uses total params=749,544,960, not 700M from constitution typo | ✅ NOTE — 749.5M is the verified count from Epic 2 sanity_check; constitution's "~700M" is a known rounding error committed in constitution but already corrected in Epic 2 data-model |

## Project Structure

### Documentation (this feature)

```text
specs/001-pretraining-loop/
├── plan.md                   ← this file
├── research.md               ← Phase 0: MFU formula, shard interleaving, flash attn, WandB, checkpoint rotation
├── data-model.md             ← Phase 1: TrainingConfig, TrainingState, TrainingMetrics, LRSchedule, SourceState, InterleavedShardLoader
├── quickstart.md             ← Phase 1: Colab setup, local smoke test, checkpoint verification
├── contracts/
│   └── trainer-cli.md        ← Phase 1: CLI argument contract, log format, resume behaviour
├── checklists/
│   └── requirements.md       ← spec quality checklist (done)
└── tasks.md                  ← Phase 2 output (speckit.tasks — not yet created)
```

### Source Code (repository root)

```text
training/                          # existing package
├── __init__.py                    # existing
├── dataset.py                     # existing (Epic 3 — ShardedDataset, do NOT modify)
├── lr_scheduler.py                # EMPTY STUB → implement: get_lr()
├── checkpoint.py                  # EMPTY STUB → implement: save_checkpoint(), load_latest_checkpoint(), rotate_checkpoints()
└── trainer.py                     # EMPTY STUB → implement: TrainingConfig, Logger, InterleavedShardLoader, estimate_mfu(), train()

colab/
├── 01_tokenizer.ipynb             # existing (Epic 1)
├── 02_model.ipynb                 # existing (Epic 2)
└── 02_pretrain.ipynb              # NEW → Colab H100 training notebook

tests/
├── test_prepare_data.py           # existing (Epic 3)
├── test_dataset.py                # existing (Epic 3)
├── test_pipeline.py               # existing (Epic 3)
├── check_token_range.py           # existing (Epic 3)
├── test_lr_scheduler.py           # NEW → unit tests for get_lr() (SC-005)
├── test_checkpoint.py             # NEW → unit tests for save/load/rotate (SC-006)
└── test_trainer.py                # NEW → smoke test, MFU unit test, InterleavedShardLoader (SC-004)

tests/fixtures/
└── data/                          # NEW → synthetic shards for local testing
    └── wikipedia/
        └── shard_00000.bin        # small synthetic shard (e.g. 50k tokens)
```
