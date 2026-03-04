"""AU-Model pretraining loop.

Public API
----------
    TrainingConfig  — all hyper-parameters in one place
    train(cfg)      — main entry-point; call from Colab or CLI
    estimate_mfu    — model-FLOPs-utilisation helper
    parse_args      — build a TrainingConfig from sys.argv

Internal modules used
---------------------
    training/lr_scheduler.py   → get_lr
    training/checkpoint.py     → save_checkpoint, rotate_checkpoints,
                                  load_latest_checkpoint
    training/dataset.py        → ShardedDataset
    model/__init__.py           → AUModel, ModelConfig
"""
from __future__ import annotations

import argparse
import math
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from model import AUModel, ModelConfig
from training.checkpoint import (
    load_latest_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
)
from training.dataset import ShardedDataset
from training.lr_scheduler import get_lr


# ─────────────────────────────────────────────────────────────────────────────
# T004  TrainingConfig
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    """All hyper-parameters for a pretraining run.

    Required positional arguments (no defaults):
        shard_dir   — root dir containing <source>/shard_*.bin sub-directories
        output_dir  — where checkpoints are written

    The effective batch size is micro_batch_size × grad_accum_steps.
    For production this must equal 128; during testing 2×2=4 is accepted with
    a warning (not an error) so that unit tests can run cheaply.
    """

    # ── Required ──────────────────────────────────────────────────────────
    shard_dir: str
    output_dir: str

    # ── Optional ──────────────────────────────────────────────────────────
    val_shard: str | None = None

    max_steps: int = 100_000
    warmup_steps: int = 2_000

    max_lr: float = 3e-4
    min_lr: float = 3e-5

    # AdamW
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Batching
    micro_batch_size: int = 32
    grad_accum_steps: int = 4
    seq_len: int = 4_096

    # Timing / checkpointing / logging
    log_interval: int = 10
    checkpoint_interval: int = 1_000
    checkpoint_keep: int = 10
    val_interval: int = 1_000

    # W&B
    use_wandb: bool = True
    wandb_project: str = "au-model"
    run_name: str | None = None

    # Misc
    gradient_checkpointing: bool = False

    # Data mixture weights  (absent sources are silently skipped)
    source_weights: dict[str, float] = field(
        default_factory=lambda: {
            "wikipedia": 0.20,
            "oscar": 0.30,
            "mc4": 0.30,
            "cc100": 0.20,
        }
    )

    def __post_init__(self) -> None:
        # Hard errors —————————————————————————————————————
        if self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be ≥ 1")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be ≥ 1")
        if self.max_lr <= self.min_lr:
            raise ValueError("max_lr must be > min_lr")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be < max_steps")

        # Soft warning ————————————————————————————————————
        eff = self.micro_batch_size * self.grad_accum_steps
        if eff != 128:
            warnings.warn(
                f"Effective batch size is {eff} (micro_batch_size={self.micro_batch_size} "
                f"× grad_accum_steps={self.grad_accum_steps}), expected 128. "
                "Use production values for actual pretraining.",
                stacklevel=2,
            )


# ─────────────────────────────────────────────────────────────────────────────
# T004  parse_args
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> TrainingConfig:
    """Build a :class:`TrainingConfig` from ``sys.argv``."""
    p = argparse.ArgumentParser(description="AU-Model pretraining")

    # Required
    p.add_argument("--shard_dir", required=True)
    p.add_argument("--output_dir", required=True)

    # Optional with defaults mirroring TrainingConfig
    p.add_argument("--val_shard", default=None)
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--warmup_steps", type=int, default=2_000)
    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--micro_batch_size", type=int, default=32)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=4_096)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--checkpoint_interval", type=int, default=1_000)
    p.add_argument("--checkpoint_keep", type=int, default=10)
    p.add_argument("--val_interval", type=int, default=1_000)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="au-model")
    p.add_argument("--run_name", default=None)
    p.add_argument("--gradient_checkpointing", action="store_true")

    args = p.parse_args()
    return TrainingConfig(
        shard_dir=args.shard_dir,
        output_dir=args.output_dir,
        val_shard=args.val_shard,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        seq_len=args.seq_len,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_keep=args.checkpoint_keep,
        val_interval=args.val_interval,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        gradient_checkpointing=args.gradient_checkpointing,
    )


# ─────────────────────────────────────────────────────────────────────────────
# T006  SourceState + InterleavedShardLoader
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SourceState:
    """Mutable per-source iterator state used by InterleavedShardLoader."""

    shard_paths: list[str]        # all shard files for this source (sorted)
    shuffled_paths: list[str]     # current shuffled order of shards
    shard_idx: int                # index into shuffled_paths (current shard)
    sample_idx: int               # sample offset within the current shard
    cycle_count: int              # how many full passes have been completed
    tokens_consumed: int          # total tokens yielded from this source

    def state_dict(self) -> dict:
        return {
            "shard_paths": list(self.shard_paths),
            "shuffled_paths": list(self.shuffled_paths),
            "shard_idx": self.shard_idx,
            "sample_idx": self.sample_idx,
            "cycle_count": self.cycle_count,
            "tokens_consumed": self.tokens_consumed,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "SourceState":
        return cls(
            shard_paths=list(d["shard_paths"]),
            shuffled_paths=list(d["shuffled_paths"]),
            shard_idx=int(d["shard_idx"]),
            sample_idx=int(d["sample_idx"]),
            cycle_count=int(d["cycle_count"]),
            tokens_consumed=int(d["tokens_consumed"]),
        )


class InterleavedShardLoader(IterableDataset):
    """Infinitely-cycling interleaved data loader.

    Scans ``<shard_dir>/<source>/shard_*.bin`` for each source named in
    *source_weights*.  Sources whose directory does not exist or contains no
    shards are silently skipped and weights are renormalised.

    Each call to ``__iter__`` resumes from the current internal state, which
    makes the loader resumable after a checkpoint restore.

    Args:
        shard_dir:      Root directory containing per-source sub-directories.
        seq_len:        Sequence length passed to :class:`ShardedDataset`.
        source_weights: Dict mapping source name → unnormalised sampling weight.
        base_seed:      RNG seed for shard shuffling.
    """

    def __init__(
        self,
        shard_dir: str,
        seq_len: int,
        source_weights: dict[str, float],
        base_seed: int = 42,
    ) -> None:
        super().__init__()
        self._shard_dir = Path(shard_dir)
        self._seq_len = seq_len
        self._base_seed = base_seed

        # ── Discover available sources ────────────────────────────────────
        available: dict[str, list[str]] = {}
        for source, _ in source_weights.items():
            src_dir = self._shard_dir / source
            if not src_dir.exists():
                continue
            shards = sorted(str(p) for p in src_dir.glob("shard_*.bin"))
            if not shards:
                continue
            available[source] = shards

        if not available:
            raise ValueError(
                f"No shard files found under {self._shard_dir} for sources "
                f"{list(source_weights.keys())}"
            )

        # Renormalise weights to discovered sources
        raw_total = sum(source_weights[s] for s in available)
        self._source_names: list[str] = list(available.keys())
        self._weights: list[float] = [
            source_weights[s] / raw_total for s in self._source_names
        ]

        # ── Initialise SourceState for each source ────────────────────────
        self._states: dict[str, SourceState] = {}
        for source in self._source_names:
            paths = available[source]
            shuffled = self._shuffle_paths(paths, seed=base_seed)
            self._states[source] = SourceState(
                shard_paths=paths,
                shuffled_paths=shuffled,
                shard_idx=0,
                sample_idx=0,
                cycle_count=0,
                tokens_consumed=0,
            )

        # Cache ShardedDataset instances (invalidated when shard_idx changes)
        self._shard_cache: dict[str, tuple[int, ShardedDataset]] = {}

    # ── Shuffle helper ────────────────────────────────────────────────────

    def _shuffle_paths(self, paths: list[str], seed: int) -> list[str]:
        rng = random.Random(seed)
        result = list(paths)
        rng.shuffle(result)
        return result

    # ── Per-source sample fetching ────────────────────────────────────────

    def _current_dataset(self, source: str) -> ShardedDataset:
        """Return (possibly cached) ShardedDataset for the active shard."""
        state = self._states[source]
        shard_path = state.shuffled_paths[state.shard_idx]
        cached = self._shard_cache.get(source)
        if cached is None or cached[0] != state.shard_idx:
            ds = ShardedDataset([shard_path], self._seq_len)
            self._shard_cache[source] = (state.shard_idx, ds)
        return self._shard_cache[source][1]

    def _next_sample(
        self, source: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance state and return next (input_ids, target_ids) for *source*."""
        state = self._states[source]

        # Get current dataset
        ds = self._current_dataset(source)
        n = len(ds)

        # If current shard is exhausted (or empty), advance to next shard
        while n == 0 or state.sample_idx >= n:
            state.shard_idx += 1
            if state.shard_idx >= len(state.shuffled_paths):
                # Completed one full pass through all shards — reshuffle
                state.cycle_count += 1
                new_seed = self._base_seed + state.cycle_count * 1000
                state.shuffled_paths = self._shuffle_paths(
                    state.shard_paths, seed=new_seed
                )
                state.shard_idx = 0
            state.sample_idx = 0
            # Invalidate cache
            self._shard_cache.pop(source, None)
            ds = self._current_dataset(source)
            n = len(ds)

        inp, tgt = ds[state.sample_idx]
        state.sample_idx += 1
        state.tokens_consumed += self._seq_len
        return inp, tgt

    # ── IterableDataset interface ─────────────────────────────────────────

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (input_ids, target_ids) infinitely, sampling sources by weight."""
        while True:
            source = random.choices(self._source_names, weights=self._weights, k=1)[0]
            yield self._next_sample(source)

    # ── Checkpoint resume ─────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            source: state.state_dict()
            for source, state in self._states.items()
        }

    def load_state_dict(self, sd: dict) -> None:
        for source, state_d in sd.items():
            if source in self._states:
                self._states[source] = SourceState.from_state_dict(state_d)
                # Invalidate dataset cache for this source
                self._shard_cache.pop(source, None)


# ─────────────────────────────────────────────────────────────────────────────
# T007  estimate_mfu
# ─────────────────────────────────────────────────────────────────────────────


def estimate_mfu(
    tokens_per_sec: float,
    model_params: int = 749_544_960,
    seq_len: int = 4_096,
    peak_flops: float = 989e12,
    num_layers: int = 24,
    num_heads: int = 12,
    d_model: int = 1_536,
) -> float:
    """Return Model FLOPs Utilisation as a fraction in [0, 1].

    FLOPs per token formula:
        param_flops = 6 * N        (forward + backward ≈ 3× forward)
        attn_flops  = 12 * L * T * H * d_head
        total_flops_per_token = param_flops + attn_flops
        MFU = total_flops_per_token * tokens_per_sec / peak_flops

    Args:
        tokens_per_sec:  Measured throughput.
        model_params:    Total trainable parameter count.
        seq_len:         Sequence length T.
        peak_flops:      Hardware peak FLOP/s (default: H100 SXM 989 TFLOP/s BF16).
        num_layers:      Transformer depth L.
        num_heads:       Number of query heads H.
        d_model:         Model width; d_head = d_model // num_heads.
    """
    d_head = d_model // num_heads
    param_flops = 6 * model_params
    attn_flops = 12 * num_layers * seq_len * num_heads * d_head
    return tokens_per_sec * (param_flops + attn_flops) / peak_flops


# ─────────────────────────────────────────────────────────────────────────────
# T015  Logger
# ─────────────────────────────────────────────────────────────────────────────


class Logger:
    """Thin wrapper around W&B with graceful fallback to no-op.

    If ``cfg.use_wandb`` is True but wandb is unavailable or init fails, a
    warning is emitted and the logger silently becomes a no-op for the rest
    of the run.
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        self._wb = None
        if not cfg.use_wandb:
            return
        try:
            import wandb  # type: ignore[import]

            wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name,
                config={k: v for k, v in vars(cfg).items() if not callable(v)},
            )
            self._wb = wandb
        except Exception as exc:
            warnings.warn(
                f"W&B initialisation failed ({exc}); running without W&B logging.",
                stacklevel=2,
            )

    def log(self, step: int, data: dict) -> None:
        """Forward *data* to W&B at *step* (no-op if W&B unavailable)."""
        if self._wb is None:
            return
        try:
            self._wb.log(data, step=step)
        except Exception:
            pass

    def finish(self) -> None:
        """Close the W&B run (no-op if W&B unavailable)."""
        if self._wb is None:
            return
        try:
            self._wb.finish()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# T017  Log line formatting
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_log_line(
    step: int,
    loss: float,
    val_loss: float | None,
    lr: float,
    grad_norm: float,
    toks_per_sec: float,
    mfu: float,
    elapsed: float,
) -> str:
    """Return a fixed-width log line with 8 space-separated key=value fields.

    Format::

        step=N  loss=F  val_loss=F|-  lr=F  grad_norm=F  tok/s=F  mfu=F%  elapsed=Fs
    """
    val_str = f"{val_loss:.4f}" if val_loss is not None else "-"
    return (
        f"step={step:<6d}  "
        f"loss={loss:.4f}  "
        f"val_loss={val_str}  "
        f"lr={lr:.4e}  "
        f"grad_norm={grad_norm:.4f}  "
        f"tok/s={toks_per_sec:.0f}  "
        f"mfu={100 * mfu:.2f}%  "
        f"elapsed={elapsed:.1f}s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T016  Validation pass
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _validate(
    model: AUModel,
    cfg: TrainingConfig,
    device: torch.device,
) -> float | None:
    """Run one pass over the validation shard and return mean cross-entropy loss.

    Returns ``None`` if no val_shard is configured or if the dataset is empty.
    """
    if cfg.val_shard is None:
        return None

    val_ds = ShardedDataset([cfg.val_shard], cfg.seq_len)
    if len(val_ds) == 0:
        warnings.warn(
            f"Validation shard {cfg.val_shard!r} produced an empty dataset "
            f"(seq_len={cfg.seq_len}); skipping validation.",
            stacklevel=2,
        )
        return None

    val_dl = DataLoader(val_ds, batch_size=cfg.micro_batch_size, num_workers=0)

    model.eval()
    total_loss = 0.0
    n_batches = 0
    for inp, tgt in val_dl:
        inp = inp.to(device)
        tgt = tgt.to(device)
        _, loss = model(inp, tgt)
        total_loss += loss.item()
        n_batches += 1
    model.train()

    return total_loss / n_batches if n_batches > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# T008 + T013  train()
# ─────────────────────────────────────────────────────────────────────────────


def train(cfg: TrainingConfig) -> None:
    """Run the pretraining loop described by *cfg*.

    The loop supports seamless resume: if a checkpoint exists in
    ``cfg.output_dir``, training continues from the next step.

    Gradient accumulation pattern (one *step*):
        for _ in range(grad_accum_steps):
            inp, tgt = next(data_iter)
            _, loss = model(inp, tgt)   # model computes loss internally
            (loss / grad_accum_steps).backward()
        clip → optimiser.step() → zero_grad
    """
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = ModelConfig()
    model = AUModel(model_cfg).to(device)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore[attr-defined]

    # TF32 on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.max_lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        fused=False,
    )

    # ── Data loader ───────────────────────────────────────────────────────
    loader = InterleavedShardLoader(
        shard_dir=cfg.shard_dir,
        seq_len=cfg.seq_len,
        source_weights=cfg.source_weights,
    )

    # ── W&B logger ────────────────────────────────────────────────────────
    logger = Logger(cfg)

    # ── Resume from checkpoint (T013) ─────────────────────────────────────
    start_step = 1
    ckpt = load_latest_checkpoint(cfg.output_dir)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "loader_state" in ckpt and ckpt["loader_state"]:
            loader.load_state_dict(ckpt["loader_state"])
        start_step = int(ckpt["step"]) + 1
        print(f"Resumed from checkpoint at step {ckpt['step']}; continuing from step {start_step}")

    # ── DataLoader wrapping the IterableDataset ────────────────────────────
    # num_workers=0 is mandatory for deterministic resume (IterableDataset
    # fork-safety and reproducibility are not guaranteed with workers > 0).
    dl = DataLoader(loader, batch_size=cfg.micro_batch_size, num_workers=0)
    data_iter = iter(dl)

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_run_start = time.time()
    t_step_start = time.time()
    val_loss: float | None = None

    for step in range(start_step, cfg.max_steps + 1):
        optimizer.zero_grad()

        # ── Gradient accumulation ─────────────────────────────────────────
        accumulated_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            inp, tgt = next(data_iter)
            inp = inp.to(device)
            tgt = tgt.to(device)
            _, loss = model(inp, tgt)   # ← model computes loss internally
            scaled = loss / cfg.grad_accum_steps
            scaled.backward()
            accumulated_loss += loss.item()

        mean_loss = accumulated_loss / cfg.grad_accum_steps

        # ── Gradient clip → optimiser ─────────────────────────────────────
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()

        current_lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.max_lr, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        optimizer.step()

        # ── Validation ────────────────────────────────────────────────────
        if step % cfg.val_interval == 0:
            val_loss = _validate(model, cfg, device)

        # ── Logging ───────────────────────────────────────────────────────
        if step % cfg.log_interval == 0:
            now = time.time()
            step_time = now - t_step_start
            t_step_start = now

            tokens_per_step = cfg.micro_batch_size * cfg.grad_accum_steps * cfg.seq_len
            toks_per_sec = (tokens_per_step * cfg.log_interval) / step_time
            mfu = estimate_mfu(toks_per_sec, seq_len=cfg.seq_len)
            elapsed = now - t_run_start

            line = _fmt_log_line(
                step=step,
                loss=mean_loss,
                val_loss=val_loss,
                lr=current_lr,
                grad_norm=grad_norm,
                toks_per_sec=toks_per_sec,
                mfu=mfu,
                elapsed=elapsed,
            )
            print(line)

            logger.log(
                step,
                {
                    "train/loss": mean_loss,
                    "train/val_loss": val_loss,
                    "train/lr": current_lr,
                    "train/grad_norm": grad_norm,
                    "train/tok_per_sec": toks_per_sec,
                    "train/mfu": mfu,
                },
            )

        # ── Checkpoint ────────────────────────────────────────────────────
        if step % cfg.checkpoint_interval == 0:
            ckpt_path = output_dir / f"step_{step:06d}.pt"
            state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "config": cfg,
                "loader_state": loader.state_dict(),
                "best_val_loss": val_loss,
                "created_at": time.time(),
            }
            save_checkpoint(ckpt_path, state)
            rotate_checkpoints(output_dir, keep=cfg.checkpoint_keep)

    logger.finish()
