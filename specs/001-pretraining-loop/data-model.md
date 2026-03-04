# Data Model: AUModel Pretraining Loop

**Date**: 4 Mart 2026  
**Branch**: `001-pretraining-loop`

---

## Entity 1: TrainingConfig

Immutable configuration object assembled at startup from CLI arguments and project constants. Passed to all components.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `shard_dir` | `str` | required | Path to directory containing training shard `.bin` files |
| `val_shard` | `str \| None` | `None` | Path to a single held-out validation shard; if None, validation is skipped |
| `output_dir` | `str` | required | Destination directory for checkpoints (Google Drive path in Colab) |
| `max_steps` | `int` | `100_000` | Total optimiser update steps |
| `warmup_steps` | `int` | `2_000` | Linear LR warmup duration |
| `max_lr` | `float` | `3e-4` | Peak learning rate (after warmup) |
| `min_lr` | `float` | `3e-5` | Floor LR (after full cosine decay) |
| `beta1` | `float` | `0.9` | AdamW β₁ |
| `beta2` | `float` | `0.95` | AdamW β₂ |
| `eps` | `float` | `1e-8` | AdamW ε |
| `weight_decay` | `float` | `0.1` | AdamW weight decay |
| `grad_clip` | `float` | `1.0` | Gradient clipping max norm |
| `micro_batch_size` | `int` | `32` | Sequences per micro-step (H100 80GB default) |
| `grad_accum_steps` | `int` | `4` | Micro-steps per optimiser update; effective batch = 128 |
| `seq_len` | `int` | `4_096` | Tokens per sequence (locked — constitution) |
| `log_interval` | `int` | `10` | Log metrics every N optimiser steps |
| `checkpoint_interval` | `int` | `1_000` | Save checkpoint every N steps |
| `checkpoint_keep` | `int` | `10` | Maximum checkpoints to retain on disk |
| `val_interval` | `int` | `1_000` | Evaluate validation loss every N steps |
| `use_wandb` | `bool` | `True` | Enable WandB logging (falls back to console if unavailable) |
| `wandb_project` | `str` | `"au-model"` | WandB project name |
| `run_name` | `str \| None` | `None` | WandB run name; auto-generated if None |
| `gradient_checkpointing` | `bool` | `False` | Trade compute for memory (enables `torch.utils.checkpoint`) |
| `source_weights` | `dict[str, float]` | see below | Per-source sampling weights for interleaving |

**Default source weights** (normalised to 1.0):
```python
{"wikipedia": 0.20, "oscar": 0.30, "mc4": 0.30, "cc100": 0.20}
```

**Validation rules**:
- `micro_batch_size × grad_accum_steps` SHOULD equal 128 (H100 80 GB default); if ≠ 128, emit `warnings.warn(f"Effective batch {micro_batch_size*grad_accum_steps} ≠ 128 (H100 default)")` but do NOT raise — this allows reduced test configs (e.g. 2×2=4) without crashing the test suite
- `micro_batch_size ≥ 1` and `grad_accum_steps ≥ 1` — raise `ValueError` if violated
- `max_lr > min_lr > 0`
- `warmup_steps < max_steps`
- `log_interval ≤ checkpoint_interval`

---

## Entity 2: TrainingState (Checkpoint contents)

Mutable state saved and loaded at every checkpoint interval. Enables exact resume.

| Field | Type | Description |
|-------|------|-------------|
| `step` | `int` | Current optimiser step (0-indexed at first save = 1000) |
| `model_state` | `dict` | `model.state_dict()` |
| `optimizer_state` | `dict` | `optimizer.state_dict()` |
| `config` | `dict` | Serialised `TrainingConfig` (for audit) |
| `loader_state` | `dict` | `InterleavedShardLoader` state — per-source shard_idx, sample_idx, cycle_count, rng_state |
| `best_val_loss` | `float \| None` | Best validation loss seen so far |
| `created_at` | `str` | ISO-8601 timestamp of when this checkpoint was saved |

**File naming convention**: `step_{step:06d}.pt`  
Example: `step_001000.pt`, `step_002000.pt`, …

**Rotation rule**: After saving, delete the oldest checkpoint when total count > `checkpoint_keep` (default 10).

**Write protocol**: Write to `step_{step:06d}.tmp`, then atomically rename to `step_{step:06d}.pt` — prevents corrupt checkpoints on Drive disconnect.

---

## Entity 3: TrainingMetrics

The set of scalar values emitted at each log event.

| Field | Type | Description |
|-------|------|-------------|
| `step` | `int` | Optimiser step number |
| `train_loss` | `float` | Mean training loss over the last `log_interval` steps |
| `val_loss` | `float \| None` | Validation loss (only populated at val_interval steps; None otherwise) |
| `lr` | `float` | Learning rate at this step (post-scheduler) |
| `grad_norm` | `float` | Pre-clip global gradient L2 norm |
| `tokens_per_sec` | `float` | Effective token throughput: `micro_batch_size × seq_len × grad_accum_steps / elapsed_sec` |
| `mfu` | `float` | Model Flops Utilisation in [0, 1] (`tokens_per_sec × 6.31B / 989T` for H100) |
| `elapsed_sec` | `float` | Wall-clock time since training start |

---

## Entity 4: LRSchedule

Stateless function mapping step → learning rate.

**State transitions**:

```
step ∈ [0, warmup_steps)    →  lr = max_lr × (step / warmup_steps)    # linear warmup
step ∈ [warmup_steps,
         max_steps]          →  lr = min_lr + 0.5 × (max_lr - min_lr)
                                    × (1 + cos(π × decay_ratio))       # cosine decay
step > max_steps             →  lr = min_lr                             # hold at floor
```

where `decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)`.

**Boundary values** (verified by unit test):
- `get_lr(0, ...)` = 0 (zero start)
- `get_lr(warmup_steps, ...)` = `max_lr`
- `get_lr(max_steps, ...)` = `min_lr`
- `get_lr(max_steps + 1, ...)` = `min_lr` (holds, no exception)

---

## Entity 5: SourceState

Per-data-source position tracker inside `InterleavedShardLoader`. Saved as part of `TrainingState.loader_state`.

| Field | Type | Description |
|-------|------|-------------|
| `shard_paths` | `list[str]` | Canonical sorted shard paths for this source |
| `shuffled_paths` | `list[str]` | Current epoch ordering (re-shuffled each cycle) |
| `shard_idx` | `int` | Index into `shuffled_paths` — which shard is active |
| `sample_idx` | `int` | Position within the active shard (which seq_len window) |
| `cycle_count` | `int` | How many full passes over this source's shards |
| `tokens_consumed` | `int` | Cumulative tokens seen from this source (diagnostic only) |

**Exhaustion handling**: When `shard_idx ≥ len(shuffled_paths)`, increment `cycle_count`, reshuffle with `seed = base_seed + cycle_count × 1000`, reset `shard_idx = 0`.

---

## Entity 6: InterleavedShardLoader

An `IterableDataset` that wraps multiple `ShardedDataset` sources and yields `(input_ids, target_ids)` tuples drawn according to per-source sampling weights.

| Property | Value |
|---------|-------|
| Type | `torch.utils.data.IterableDataset` |
| Outputs | `tuple[torch.Tensor, torch.Tensor]` — shape `(seq_len,)` each |
| Iteration | Infinite (training loop terminates via `max_steps`) |
| Source selection | Weighted-random per step using `random.choices()` |
| Resumable | Yes — via `state_dict()` / `load_state_dict()` methods |
| Source weights | Default: wikipedia=0.20, oscar=0.30, mc4=0.30, cc100=0.20 |

---

## File-to-Entity Mapping

| File | Primary Entity |
|------|---------------|
| `training/lr_scheduler.py` | LRSchedule |
| `training/checkpoint.py` | TrainingState (save/load/rotate) |
| `training/dataset.py` (existing) | ShardedDataset (per-shard source) |
| `training/trainer.py` | TrainingConfig, TrainingMetrics, InterleavedShardLoader, orchestration |
| `colab/02_pretrain.ipynb` | TrainingConfig instantiation + trainer invocation |
