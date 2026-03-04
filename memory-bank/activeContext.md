# Active Context

> **Update this file at the start and end of every work session.**

## Current Focus

**Phase:** Phase 5 — SFT (Instruction Tuning)
**Current task:** Epic 4 (pretraining loop) code complete — ready to run trainer tests on Colab H100 and start Epic 5 speckit
**Last updated:** 5 Mart 2026

## What We Just Did

- Completed speckit.implement for Epic 4 (001-pretraining-loop), commit `fd8d21a`:
  - T001: test fixtures (training 100k tok + val 2048 tok uint16 shards)
  - T002: `get_lr()` cosine LR schedule with linear warmup in `training/lr_scheduler.py`
  - T003: `tests/test_lr_scheduler.py` — 14/14 PASS locally
  - T004: `TrainingConfig` dataclass + `parse_args()` in `training/trainer.py`
  - T005: `save_checkpoint()` (atomic .tmp→rename) + `rotate_checkpoints()` in `training/checkpoint.py`
  - T006: `SourceState` + `InterleavedShardLoader(IterableDataset)` — weighted, resumable, partial-source aware
  - T007: `estimate_mfu()` — FLOPs formula: `(6N + 12LTHd_head) × tps / peak_flops`
  - T008: `train()` — grad accum, BF16 autocast, grad clip, checkpoint+rotate, LR schedule
  - T009–T014: `tests/test_trainer.py` — smoke, resume, logging, val_loss tests written (need H100 to run)
  - T010: `tests/test_checkpoint.py` — 12/12 PASS locally (logit round-trip, rotation, loader state)
  - T012: `load_latest_checkpoint()` in `training/checkpoint.py`
  - T013: resume wired into `train()` — restores model+opt+loader, sets step=ckpt["step"]+1
  - T015: `Logger` class — W&B with graceful fallback
  - T016: validation pass — `ShardedDataset([val_shard]) → DataLoader → no_grad forward`
  - T017: full log line `step=N  loss=F  val_loss=F|-  lr=F  grad_norm=F  tok/s=F  mfu=F%  elapsed=Fs`
  - T018: `colab/02_pretrain.ipynb` (7 cells, H100-ready, auto-resume)
  - T019: `training/__init__.py` now exports `TrainingConfig` + `get_lr`
  - **Local results**: lr_scheduler 14/14 ✅, checkpoint 12/12 ✅; trainer tests deferred to H100

## What We Decided (Key Decisions Log)

| Decision | Reasoning |
|----------|-----------|
| vocab_size=64000 (not 32000) | Constitution amendment: richer Turkish coverage |
| d_model=1536, num_layers=24, num_heads=12, num_kv_heads=6 | ~749.5M params at H100 budget |
| ffn_hidden_dim=4352 | SwiGLU constraint: must be multiple of 64 |
| rope_theta=500000 | Extended context (YaRN-style); handles 4096 seq_len |
| AUModel._init_weights() | LLaMA-style init essential: without it, initial loss > 1000 |
| Param count 749.5M (not "700M") | Arithmetic verification confirmed; spec was wrong |
| bias=False on all nn.Linear | Standard LLaMA practice; reduces params + improves training |
| Weight tying (embed ↔ lm_head) | Reduces 98M params counted twice; standard for decoder LLMs |
| get_num_params() uses set() | Correctly deduplicates tied parameters |

## What Needs To Be Done Next

**Immediate next step: Run trainer tests on Colab H100, then start Epic 5 (SFT)**

```
Colab H100 — run before starting Epic 5:
1. Open colab/02_pretrain.ipynb
2. Run tests/test_trainer.py (smoke, resume, logging, val_loss)
3. Verify SC-002 (MFU ≥35%), SC-008 (VRAM ≥70 GB, micro_batch_size=32)
4. Mark T020 in specs/001-pretraining-loop/tasks.md with actual H100 numbers

Epic 5 — SFT (can spec now while waiting for H100):
1. speckit.specify for Epic 5 (epics/EPIC-05-sft.md already written)
2. Implement sft/sft_dataset.py — chat format loader, loss masking (ignore_index=-100)
3. Implement sft/sft_trainer.py — SFT loop (reuse TrainingConfig pattern)
4. Collect/translate instruction data: Alpaca-52k + Dolly-15k → Turkish
```

## Active Questions / Blockers

| Question | Status |
|----------|--------|
| Constitution amendment for param count (700M→750M) | Pending — low priority |
| ffn_hidden_dim field name drift (constitution uses `ffn_hidden`) | Pending |
| Colab H100 availability | Open — needed for trainer test_trainer.py + SC-002/SC-008 |
| REPO_URL in colab/02_pretrain.ipynb Cell 4 | Must be set before Colab run |

## Recent Context (AI should know this)

- **Frozen hyperparameters** (do NOT change): vocab_size=64000, d_model=1536, num_heads=12, num_kv_heads=6, num_layers=24, ffn_hidden_dim=4352, max_seq_len=4096, rope_theta=500000
- **Param count**: 749,544,960 (verified analytically + by running model)
- **AUModel.forward API**: `(logits, loss) = model(tokens, targets)` — loss computed INSIDE model via `F.cross_entropy`; never recompute loss in SFT trainer either
- **SFT loss masking**: use `ignore_index=-100` in cross_entropy — prompt tokens must be masked
- `model/sanity_check.py` is the standard validation script — run it after any model change
- `training/` package is complete: TrainingConfig, get_lr, save/load checkpoint, InterleavedShardLoader, train()
- Training dtype: **BF16 only** (never FP16)
- All Colab checkpoints → `/content/drive/MyDrive/aumodel_checkpoints/`
- SFT chat format: `<|system|>\n{sys}\n<|user|>\n{instr}\n<|assistant|>\n{resp}`

## Notes for Next Session

- `model/` and `training/` are complete — do not modify unless there is a bug
- SFT epic: implement `sft/sft_dataset.py` first (chat format + masking), then `sft/sft_trainer.py`
- SFT trainer should reuse `TrainingConfig` from `training/` and `save_checkpoint/rotate_checkpoints` from `training/checkpoint.py`
- The tokenizer model file is at `tokenizer/turkish_bpe.model` (and in Drive)
