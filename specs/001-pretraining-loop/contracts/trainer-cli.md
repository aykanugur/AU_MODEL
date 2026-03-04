# CLI Contract: training/trainer.py

**Date**: 4 Mart 2026  
**Branch**: `001-pretraining-loop`

This document specifies the command-line interface contract for `training/trainer.py`. Any implementation that satisfies all entries in this contract is compliant.

---

## Invocation

```bash
python -m training.trainer \
  --shard-dir   <path>      \
  --val-shard   <path>      \
  --output-dir  <path>      \
  [options...]
```

Or via Colab cell:
```python
%run training/trainer.py --shard-dir ... --output-dir ...
```

---

## Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--shard-dir` | `str` | Directory containing `.bin` training shard files |
| `--output-dir` | `str` | Directory to write checkpoint files (must be writable; created if absent) |

---

## Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--val-shard` | `str` | `None` | Path to a single `.bin` validation shard; if omitted, validation is skipped with a warning |
| `--max-steps` | `int` | `100000` | Total optimiser update steps |
| `--warmup-steps` | `int` | `2000` | LR linear warmup steps |
| `--max-lr` | `float` | `3e-4` | Peak learning rate |
| `--min-lr` | `float` | `3e-5` | Minimum learning rate (LR floor after decay) |
| `--micro-batch-size` | `int` | `32` | Sequences per micro-step |
| `--grad-accum-steps` | `int` | `4` | Micro-steps per optimiser update |
| `--log-interval` | `int` | `10` | Log metrics every N optimiser steps |
| `--checkpoint-interval` | `int` | `1000` | Save checkpoint every N steps |
| `--checkpoint-keep` | `int` | `10` | Maximum checkpoints to retain |
| `--val-interval` | `int` | `1000` | Run validation every N steps |
| `--wandb-project` | `str` | `"au-model"` | WandB project name |
| `--run-name` | `str` | auto | WandB run name |
| `--no-wandb` | flag | off | Disable WandB; use console-only logging |
| `--gradient-checkpointing` | flag | off | Enable activation checkpointing (trades compute for memory) |
| `--source-weights` | `str` | see below | JSON string of per-source sampling weights |

**Default `--source-weights`**:
```json
{"wikipedia": 0.20, "oscar": 0.30, "mc4": 0.30, "cc100": 0.20}
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Training completed to `max_steps` without error |
| `1` | Fatal error (shard directory not found, invalid config, OOM without recovery) |

---

## stdout Log Format

Each log event (every `log_interval` steps) emits one line:

```
step=<N>  loss=<F>  val_loss=<F|->  lr=<F>  grad_norm=<F>  tok/s=<F>  mfu=<F>%  elapsed=<F>s
```

Where `val_loss` is printed as numeric at `val_interval` steps and `-` otherwise.

**Example**:
```
step=1000  loss=8.3421  val_loss=8.5104  lr=3.00e-04  grad_norm=0.92  tok/s=48312  mfu=30.8%  elapsed=241.3s
step=1010  loss=8.2897  val_loss=-       lr=3.00e-04  grad_norm=0.88  tok/s=48501  mfu=30.9%  elapsed=251.7s
```

---

## Resume Behaviour

When `--output-dir` already contains checkpoint files matching `step_*.pt`, the trainer performs automatic resume:
1. Identifies the highest-numbered `step_*.pt` file.
2. Loads model weights, optimiser state, loader state, and step counter.
3. Prints: `[Resume] Loaded checkpoint step_<N>.pt — resuming from step <N+1>`.
4. Continues training from step `N+1`.

If no checkpoint exists:
- Prints: `[Resume] No checkpoint found in <output_dir> — starting from step 0.`
- Initialises model from scratch.

---

## Shard File Convention

Training shards must match the pattern `<shard_dir>/<source>/shard_?????.bin` where `<source>` is one of `wikipedia`, `oscar`, `mc4`, `cc100`.

Example layout expected by default source-weight keys:
```
/content/drive/MyDrive/AUModel/data/
├── wikipedia/
│   ├── shard_00000.bin
│   └── shard_00001.bin
├── oscar/
│   ├── shard_00000.bin
│   ├── ...
├── mc4/
│   └── ...
└── cc100/
    └── ...
```

If only one source directory is present, the trainer runs on that source only (no interleaving).
