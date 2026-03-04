# Feature Specification: AUModel Pretraining Loop

**Feature Branch**: `001-pretraining-loop`  
**Created**: 4 Mart 2026  
**Status**: Draft

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Run a Complete Pretraining Session (Priority: P1)

A researcher mounts their prepared token shards (from the data pipeline) and launches the trainer for the first time. The trainer initialises the model, configures the optimiser and learning-rate schedule, and begins iterating over the shards. Every N steps it prints a training log line (step, loss, learning rate, throughput). Every 1000 steps it saves a checkpoint to Google Drive. The run completes or is interrupted cleanly.

**Why this priority**: Nothing else (SFT, inference) is possible without a trained checkpoint. This is the single most critical deliverable of the phase.

**Independent Test**: Can be fully tested by running the trainer with a small synthetic shard for 50 steps; it must log metrics at each step interval and save correctly, then exit with code 0.

**Acceptance Scenarios**:

1. **Given** token shards exist and the model config is valid, **When** the trainer is launched, **Then** training starts, step 1 logs a finite loss value, and execution continues without error.
2. **Given** 1000 steps have elapsed, **When** the checkpoint interval fires, **Then** a checkpoint file appears in the output directory containing model weights, optimiser state, and step number.
3. **Given** gradient accumulation is configured, **When** a gradient update fires, **Then** the cumulative loss has been divided by the accumulation count before the update.
4. **Given** BF16 mixed precision is active, **When** the forward pass runs, **Then** no overflow or NaN appears in loss for at least the first 5000 steps on a well-initialised model.

---

### User Story 2 — Resume Training After Interruption (Priority: P1)

A Colab session disconnects at step 3,742. The researcher relaunches the same notebook cell. The trainer discovers the latest checkpoint on Drive, restores model and optimiser state, and resumes from step 3,743 without loss spike or data repetition.

**Why this priority**: Colab sessions disconnect unpredictably; seamless resume is essential for a 100-hour H100 run.

**Independent Test**: Run trainer for 100 steps, kill the process, relaunch with `--resume`; verify the first logged step is 101 and the loss curve is continuous.

**Acceptance Scenarios**:

1. **Given** a checkpoint at step N exists, **When** the trainer is relaunched, **Then** the first training step is N+1 and model + optimiser states match the saved checkpoint exactly.
2. **Given** resume is active, **When** the data loader is reconstructed, **Then** no tokens already consumed before step N are fed again.
3. **Given** a checkpoint directory is empty or absent, **When** the trainer is launched with resume, **Then** training starts from step 0 with an informational message.

---

### User Story 3 — Monitor Training Progress (Priority: P2)

A researcher wants to track whether the model is learning during a long Colab run. At each logging interval they see step, loss, learning rate, tokens per second, and model-flops utilisation (MFU). The same metrics are mirrored to a remote experiment dashboard.

**Why this priority**: Without loss and MFU visibility, hardware under-utilisation or training instabilities cannot be caught early.

**Independent Test**: Launch trainer for 20 steps with logging interval = 5; verify exactly 4 log lines appear, each containing all five metric fields with finite values.

**Acceptance Scenarios**:

1. **Given** logging interval is N, **When** steps N, 2N, 3N … fire, **Then** each log line contains step, loss, lr, tok/s, and mfu — all finite and non-NaN.
2. **Given** WandB integration is enabled, **When** a log event fires, **Then** the metric appears in the WandB run within the next sync cycle.
3. **Given** step 1 runs with a freshly initialised model, **When** loss is logged, **Then** the value is approximately ln(vocab_size) ≈ 11.07.

---

### User Story 4 — Launch Training from Colab Notebook (Priority: P2)

A researcher opens `colab/02_pretrain.ipynb` on a fresh H100 Colab session. Running cells in order: mounts Drive, installs dependencies (including the accelerated attention library), enables hardware optimisations, loads the model from config, and starts or resumes pretraining.

**Why this priority**: All actual pretraining happens on Colab; the notebook is the primary entrypoint.

**Independent Test**: On a Colab H100 instance, run all notebook cells top-to-bottom; the final cell must print the first training log line without raising any import or hardware error.

**Acceptance Scenarios**:

1. **Given** a fresh Colab H100 session, **When** all cells run in order, **Then** the model loads without OOM and training begins.
2. **Given** Drive contains a previous checkpoint, **When** the training cell runs, **Then** training resumes from the latest checkpoint automatically.
3. **Given** hardware optimisations are enabled in the notebook, **When** training runs for 10 steps, **Then** GPU utilisation exceeds 70%.

---

### Edge Cases

- What happens when a shard file is corrupted mid-run? Trainer must raise a descriptive error naming the file, not silently produce NaN loss.
- What happens when Drive space is exhausted during checkpoint save? Trainer must detect the write failure, log an error, and continue training without crashing.
- What happens when `get_lr()` is called with a step value below 0 or above `max_steps`? It must return a clamped valid value without raising an exception.
- What happens when gradient accumulation steps is 1? Trainer must behave identically to accumulation > 1 but update every step.
- What happens when WandB is unavailable (no internet, no API key)? Trainer must fall back to console-only logging and continue without crashing.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The training loop MUST process tokens in fixed-length windows of `max_seq_len` (4096) tokens, assembling input and target sequences as a one-position shift.
- **FR-002**: The training loop MUST accumulate gradients over exactly `grad_accum_steps` (16) micro-steps before each optimiser update, dividing loss by `grad_accum_steps` at each micro-step.
- **FR-003**: The training loop MUST clip the global gradient norm to `max_norm=1.0` before each optimiser update.
- **FR-004**: The learning-rate scheduler MUST implement linear warm-up from 0 to `max_lr=3e-4` over the first 2000 steps, followed by cosine decay to `min_lr=3e-5` over `max_steps=100,000` steps, then hold at `min_lr`.
- **FR-005**: The trainer MUST use BF16 mixed precision for the forward pass; FP16 is not permitted.
- **FR-006**: The trainer MUST compile the model (graph optimisation) once, immediately after initialisation and before the first forward pass.
- **FR-007**: Checkpoints MUST be saved every 1000 steps to the configured output directory and MUST include model weights, optimiser state, current step number, and training configuration.
- **FR-008**: The trainer MUST support seamless resume: at startup it discovers the highest-numbered checkpoint in the output directory, restores model + optimiser state, and continues from the next step.
- **FR-009**: The trainer MUST log the following at each logging interval: step, training loss, current learning rate, tokens per second, and MFU.
- **FR-010**: WandB metric logging MUST be optional; if unavailable or disabled, the trainer MUST fall back to console-only logging without crashing.
- **FR-011**: The Colab notebook MUST mount Google Drive, install all required dependencies, enable TF32 hardware optimisations, load the model from `ModelConfig`, and launch or resume the training loop — all in sequential cells runnable top-to-bottom.
- **FR-012**: The data loader MUST stream token windows from pre-built binary shard files and MUST NOT load all shards into RAM simultaneously.
- **FR-013**: The AdamW optimiser MUST be configured with `beta1=0.9`, `beta2=0.95`, `weight_decay=0.1`, and `eps=1e-8`.
- **FR-014**: The trainer MUST accept `--shard-dir` and `--output-dir` as CLI arguments specifying shard source and checkpoint destination.

### Key Entities

- **TrainingConfig**: Encapsulates all hyperparameters — `max_steps`, `warmup_steps`, `max_lr`, `min_lr`, `weight_decay`, `beta1`, `beta2`, `grad_accum_steps`, `batch_size`, `seq_len`, `log_interval`, `checkpoint_interval`, `output_dir`, `shard_dir`, `wandb_project`.
- **Checkpoint**: Snapshot of training state — model weights, optimiser state, step number, training config. One file per checkpoint interval.
- **TrainingMetrics**: Per-step observable state — `step`, `loss`, `lr`, `tokens_per_sec`, `mfu`, `grad_norm`.
- **LRSchedule**: Function mapping `step → learning_rate` under the cosine-with-warmup rule.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Training loss at step 1 is within ±0.5 of ln(vocab_size) ≈ 11.07, confirming correct model initialisation.
- **SC-002**: Training throughput on the target accelerator reaches at least 100,000 tokens per second with model-flops utilisation ≥ 35%, confirming hardware is not being wasted.
- **SC-003**: After interrupting and resuming from a checkpoint at step N, the training loss at step N+1 differs from the non-interrupted baseline by less than 0.01.
- **SC-004**: A 50-step smoke-test run with a synthetic shard completes in under 5 minutes on a local CPU without raising any exception.
- **SC-005**: The LR scheduler returns `max_lr` at exactly step 2000 and `min_lr` at `max_steps`, verified by a unit test covering all three regions (warmup, decay, hold).
- **SC-006**: A checkpoint saved to disk is loadable and produces identical logits for the same input batch as the in-memory model (round-trip test).

## Assumptions

- Token shards have already been built by the Epic 3 data pipeline and stored at `/content/drive/MyDrive/AUModel/data/`.
- The tokenizer is at `tokenizer/turkish_bpe.model` with `vocab_size=64000`, `BOS_ID=2`, `EOS_ID=3`.
- `AUModel` and `ModelConfig` from `model/` are stable and will not change during this epic.
- `ShardedDataset` from `training/dataset.py` is complete and tested.
- Flash Attention will be installed at H100 runtime; local/CPU development must degrade gracefully to standard attention.
- All checkpoints target Google Drive; local paths are for temporary files only.
- Frozen hyperparameters (must not change): `vocab_size=64000`, `d_model=1536`, `num_heads=12`, `num_kv_heads=6`, `num_layers=24`, `ffn_hidden_dim=4352`, `max_seq_len=4096`, `rope_theta=500000`.
