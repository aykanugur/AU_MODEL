# Quickstart: AUModel Pretraining Loop

**Branch**: `001-pretraining-loop`

---

## Prerequisites

Before running the trainer, ensure the following are complete:

- [ ] Epic 1 done: `tokenizer/turkish_bpe.model` exists (vocab_size=64000)
- [ ] Epic 2 done: `model/` package is fully implemented and `model/sanity_check.py` exits 0
- [ ] Epic 3 done: training shards exist at `/content/drive/MyDrive/AUModel/data/<source>/shard_*.bin`
- [ ] Google Drive is mounted (Colab only)
- [ ] Python 3.11+ with `torch>=2.1`, `numpy`, `tqdm`, `wandb` (optional) installed

---

## Colab H100 — Full Pretraining (Primary Path)

Open `colab/02_pretrain.ipynb` and run cells top-to-bottom.

### Cell 1: Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Install Dependencies
```bash
%%bash
pip install flash-attn --no-build-isolation -q   # H100 flash attention
pip install wandb tqdm -q
```

### Cell 3: Hardware Setup
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Cell 4: Clone / Update Repo
```bash
%%bash
# If not already cloned:
git clone https://github.com/aykanugur/AU_MODEL.git /content/AU_MODEL
cd /content/AU_MODEL && git pull
```

### Cell 5: Start / Resume Training
```bash
%%bash
cd /content/AU_MODEL
python -m training.trainer \
  --shard-dir  /content/drive/MyDrive/AUModel/data \
  --val-shard  /content/drive/MyDrive/AUModel/data/validation/shard_00000.bin \
  --output-dir /content/drive/MyDrive/AUModel/checkpoints \
  --max-steps  100000 \
  --wandb-project au-model
```

The trainer automatically detects and resumes from the latest checkpoint if one exists.

---

## Local CPU — Development & Testing

For unit tests and smoke tests only (no H100 required):

```bash
# Install deps
cd /Users/aykanugur/Desktop/git/AU_MODEL
.venv/bin/pip install torch numpy tqdm wandb pytest -q

# Run unit tests
.venv/bin/python -m pytest tests/test_lr_scheduler.py tests/test_checkpoint.py tests/test_trainer.py -v

# Smoke test (50 steps on synthetic shard)
.venv/bin/python -m training.trainer \
  --shard-dir  tests/fixtures/data \
  --output-dir /tmp/au_model_smoke \
  --max-steps  50 \
  --micro-batch-size 2 \
  --grad-accum-steps 2 \
  --no-wandb
```

Expected output for smoke test (last lines):
```
step=10  loss=11.07  val_loss=-  lr=1.50e-05  grad_norm=...  tok/s=...  mfu=...%  elapsed=...s
...
step=50  loss=...  val_loss=-  lr=7.50e-05  ...
[Train] Done. Total steps: 50  Best val: N/A
```

---

## Verifying a Checkpoint

```python
import torch
from model import AUModel, ModelConfig

ckpt = torch.load("/path/to/step_001000.pt", map_location="cpu")
print("Saved at step:", ckpt["step"])

cfg = ModelConfig()
model = AUModel(cfg)
model.load_state_dict(ckpt["model_state"])
model.eval()

x = torch.zeros(1, 16, dtype=torch.long)
with torch.no_grad():
    logits, _ = model(x)
print("Logits shape:", logits.shape)   # → (1, 16, 64000) ✓
```

---

## Monitoring on WandB

1. Set your API key: `export WANDB_API_KEY=<your_key>` (or set it in Colab secrets)
2. The trainer initialises a WandB run automatically if `--no-wandb` is not passed
3. Navigate to `https://wandb.ai/aykanugur/au-model` to see live metrics

Tracked metrics: `step`, `train_loss`, `val_loss`, `lr`, `grad_norm`, `tok/s`, `mfu`

---

## Expected Loss Curve

| Step | Expected Train Loss |
|------|---------------------|
| 1 | ~11.07 (≈ ln 64000) |
| 1000 | ~7.0–8.5 |
| 10000 | ~3.5–5.0 |
| 100000 | ~2.0–3.0 |

If loss at step 1 is outside [10.5, 11.6], check model initialisation (`model/sanity_check.py`).  
If loss is NaN at any step, check BF16 is active and gradient clipping is applied.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUDA out of memory` | Batch too large | Reduce `--micro-batch-size` to 16, enable `--gradient-checkpointing` |
| `NaN loss` immediately | FP16 instead of BF16, or bad init | Check `torch.autocast(dtype=torch.bfloat16)` is used; run `model/sanity_check.py` |
| MFU < 30% | Flash attention not active | Verify `torch.backends.cuda.flash_sdp_enabled()` returns True |
| Resume starts from step 0 | checkpoint filename mismatch | Check `--output-dir` matches the path used in the previous run |
| WandB not logging | No API key | Set `WANDB_API_KEY` or pass `--no-wandb` to use console logging |
