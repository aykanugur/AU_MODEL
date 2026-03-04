# Stories — Epic 4: Pretraining

**Epic ref:** `EPIC-04-pretraining.md`
**Branch:** `epic/04-pretraining`
**Persona:** Developer
**Total stories:** 5

---

## ST-04-01 — Learning Rate Scheduler

**As a developer,**
I want a learning rate scheduler that warms up and then decays following a cosine curve,
So that training starts stably and learning rate is reduced exactly as specified without manual LR adjustments each step.

### Acceptance Criteria

- The scheduler linearly increases the learning rate from 0 to `3e-4` over steps 0–2,000.
- After warmup, the learning rate follows a cosine decay from `3e-4` to `3e-5` over steps 2,000–33,000.
- Querying `get_lr(0)` returns approximately 0.
- Querying `get_lr(2000)` returns `3e-4`.
- Querying `get_lr(33000)` returns `3e-5`.
- The scheduler returns a value in the valid range for any step between 0 and 33,000.

---

## ST-04-02 — Checkpoint Save and Resume

**As a developer,**
I want training to save checkpoints every 1,000 steps and resume from the latest checkpoint with a flag,
So that a Colab session disconnect never causes me to lose more than 1,000 steps of training progress.

### Acceptance Criteria

- A checkpoint file is written every 1,000 steps containing model weights, optimizer state, current step number, and validation loss.
- Checkpoint filenames include the step number (e.g., `ckpt_010000.pt`).
- On startup with the `--resume` flag, the trainer loads the checkpoint with the highest step number automatically.
- After resuming from a saved checkpoint and training 5 more steps, training loss matches the value that was logged at that step in the original run (within ±0.01).
- Only the 3 most recent checkpoints are kept on disk — older ones are deleted automatically after each save.

---

## ST-04-03 — Training Loop with Curriculum Phase Switch

**As a developer,**
I want the training loop to automatically switch from Phase 1 to Phase 2 data at step 16,500,
So that the curriculum ordering (Wikipedia-first) is enforced without human intervention during a long training run.

### Acceptance Criteria

- Steps 0–16,499 draw exclusively from Phase 1 shards (Wikipedia, indices 0000–0436).
- At step 16,500 the DataLoader switches to Phase 2 shards (mixed corpus, indices 0437–0874) without trainer restart.
- The training loss spike at the phase switch is ≤ 0.3 above the running average at that point.
- Gradient clipping is applied at every step with a max norm of 1.0.
- The effective batch size is 524,288 tokens per step (batch=16 × grad_accum=8 × seq_len=4096).
- Validation loss is computed on the full `val.bin` every 1,000 steps and printed alongside the step and training loss.
- Tokens-per-second throughput is logged every 100 steps.

---

## ST-04-04 — CLI Training Script

**As a developer,**
I want a command-line script that starts or resumes a training run without editing source files,
So that I can kick off training on any machine by passing paths and flags at the command line.

### Acceptance Criteria

- The script accepts `--checkpoint_dir`, `--data_dir`, `--batch_size`, and `--resume` arguments.
- Using `--resume` causes the trainer to pick up from the latest saved checkpoint automatically.
- Omitting `--resume` starts training from step 0 with freshly initialised weights.
- Default values for `--checkpoint_dir` (`checkpoints/`) and `--data_dir` (`data/processed/`) allow the script to run without specifying paths on a standard layout.

---

## ST-04-05 — Colab Training Notebook

**As a developer,**
I want a Colab notebook that mounts Google Drive and runs training end-to-end from any fresh session,
So that I can resume training on H100 Colab hardware with a single "Run all cells" action even after session resets.

### Acceptance Criteria

- The notebook has exactly 5 cells: install dependencies, mount Google Drive, clone the repository, run the training script with `--resume`, and plot the loss curve.
- Every cell is idempotent — running it a second time does not cause an error or re-download already present files.
- Checkpoints are saved to Google Drive so data persists across Colab session restarts.
- The loss curve cell plots training loss and validation loss against step number using matplotlib.
- On completion of a full run, `ckpt_033000.pt` exists in the Drive checkpoint directory.

---

_Epic complete when all 5 stories pass their acceptance criteria._
_Validation perplexity target: `exp(val_loss) ≤ 22` at step 33,000._
_Last updated: 4 Mart 2026_
