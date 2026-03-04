# Epic 4 — Pretraining

| Field | Value |
|-------|-------|
| **Branch** | `epic/04-pretraining` |
| **Base branch** | `main` |
| **Merge target** | `main` |
| **PRD refs** | F-04, F-05, G4, M4 |
| **Depends on** | Epic 2 (model), Epic 3 (data shards + `PretrainDataset`) |
| **Status** | ⬜ Not started |
| **Output files** | `training/lr_scheduler.py`, `training/checkpoint.py`, `training/trainer.py`, `scripts/run_training.py`, `colab/train.ipynb` |

---

## Goal

Pretrain AUModel 700M on ≥ 17.5B Turkish tokens in ≤ 18 H100 hours using curriculum training. Target: validation perplexity ≤ 22 at step 33,000. All Colab sessions must be resumable with zero data loss — checkpoint every 1,000 steps to Google Drive.

---

## Hyperparameters (Locked)

```
optimizer        = AdamW(β1=0.9, β2=0.95, weight_decay=0.1, eps=1e-8)
lr               = 3e-4  →  cosine decay to 3e-5
warmup_steps     = 2000
max_steps        = 33000
batch_size       = 16
grad_accum       = 8
effective_batch  = 16 × 8 × 4096 = 524,288 tokens/step
grad_clip        = 1.0
dtype            = bfloat16
curriculum       = Phase 1: steps 0–16,500 (Wikipedia shards 0000–0436)
                   Phase 2: steps 16,500–33,000 (mixed shards 0437–0874)
```

---

## Tasks

- [ ] **`training/lr_scheduler.py`** — `CosineWithWarmup`: linear warmup from `lr=0` to `lr=3e-4` over steps 0–2,000; cosine decay from `3e-4` to `lr_min=3e-5` over steps 2,000–33,000; `get_lr(step: int) → float`
- [ ] **`training/checkpoint.py`** — `save_checkpoint(model, optimizer, step, val_loss, path: str)`: `torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, 'val_loss': val_loss}, path)`; `load_latest_checkpoint(ckpt_dir: str) → (state_dict, step)`: glob `ckpt_dir/ckpt_*.pt`, parse step from filename, return highest; delete all but 3 most recent checkpoints after each save
- [ ] **`training/trainer.py`** — main train loop: `model = torch.compile(model)`; `torch.autocast('cuda', dtype=torch.bfloat16)` context; inner grad-accum loop of 8 micro-steps; `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`; call `scheduler.get_lr(step)` and update param groups each step; switch `DataLoader` from Phase 1 to Phase 2 dataset exactly at step 16,500; log `step / train_loss / lr / tokens_per_sec` every 100 steps via `print()`; compute `val_loss` on full `val.bin` every 1,000 steps; call `save_checkpoint()` every 1,000 steps
- [ ] **`scripts/run_training.py`** — CLI via `argparse`: `--checkpoint_dir` (default `checkpoints/`), `--data_dir` (default `data/processed/`), `--batch_size` (default 16), `--resume` (flag); calls `load_latest_checkpoint()` if `--resume` set
- [ ] **`colab/train.ipynb`** — 5 cells: (1) `!pip install torch==2.2.0 sentencepiece datasets`, (2) mount Google Drive at `/content/drive`, (3) `!git clone https://github.com/aykanugur/AUModel && cd AUModel`, (4) `!python scripts/run_training.py --checkpoint_dir /content/drive/MyDrive/AUModel/checkpoints --resume`, (5) loss curve plot via `matplotlib`; every cell is idempotent (safe to re-run)

---

## Exit Criteria

| Check | Pass Condition |
|-------|---------------|
| Training reaches step 33,000 | `ckpt_033000.pt` exists in checkpoint dir |
| Val perplexity at step 33,000 | `exp(val_loss) ≤ 22` |
| Training loss at step 33,000 | `train_loss ≤ 2.9` |
| Peak GPU memory | `torch.cuda.max_memory_allocated() ≤ 30 GB` |
| Checkpoint resume | Load `ckpt_010000.pt`, train 5 steps, loss matches logged value from original run `± 0.01` |
| Tokens/sec | ≥ 200,000 tok/sec sustained (measured over steps 5,000–6,000) |
| Curriculum phase switch | Loss spike of ≤ 0.3 when switching phase at step 16,500 |

---

## Unlocks

- **Epic 5** (SFT) — needs `ckpt_033000.pt`

---

_Last updated: 4 Mart 2026_
