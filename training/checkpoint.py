"""Checkpoint utilities: save, rotate, and load training snapshots.

Checkpoint filename convention:  step_{step:06d}.pt
Atomic writes (write .tmp → os.rename) guard against corrupt files on crash.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_CKPT_PATTERN = re.compile(r"step_(\d+)\.pt$")


def _sorted_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Return all step_*.pt files in *ckpt_dir* sorted by step number (ascending)."""
    matches: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("step_*.pt"):
        m = _CKPT_PATTERN.search(p.name)
        if m:
            matches.append((int(m.group(1)), p))
    matches.sort(key=lambda x: x[0])
    return [p for _, p in matches]


# ──────────────────────────────────────────────────────────────────────────────
# T005 — save + rotate
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically save *state* to *path*.

    The checkpoint state dict is expected to contain at minimum:
        model_state, optimizer_state, step, config,
        loader_state, best_val_loss, created_at

    The write is made atomic by first writing to a sibling ``.tmp`` file and
    then using :func:`os.rename` (POSIX atomic on the same filesystem).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.rename(tmp, path)


def rotate_checkpoints(ckpt_dir: Path, keep: int = 10) -> None:
    """Delete the oldest checkpoints in *ckpt_dir* so that at most *keep* remain.

    Files are identified by the ``step_NNNNNN.pt`` naming convention and sorted
    ascending by step number; the oldest (lowest step) are removed first.
    """
    ckpt_dir = Path(ckpt_dir)
    all_ckpts = _sorted_checkpoints(ckpt_dir)
    to_delete = all_ckpts[: max(0, len(all_ckpts) - keep)]
    for p in to_delete:
        try:
            p.unlink()
        except FileNotFoundError:
            pass  # already gone — not an error


# ──────────────────────────────────────────────────────────────────────────────
# T012 — load latest
# ──────────────────────────────────────────────────────────────────────────────

def load_latest_checkpoint(output_dir: str) -> dict[str, Any] | None:
    """Load the most-recent checkpoint from *output_dir*.

    Returns the state dict if a checkpoint is found, or ``None`` if the
    directory is empty / does not exist.
    """
    ckpt_dir = Path(output_dir)
    if not ckpt_dir.exists():
        return None
    all_ckpts = _sorted_checkpoints(ckpt_dir)
    if not all_ckpts:
        return None
    latest = all_ckpts[-1]
    return torch.load(latest, map_location="cpu", weights_only=False)
