"""
ShardedDataset — PyTorch Dataset for streaming pretraining.
============================================================
Reads flat uint16 binary shard files produced by scripts/prepare_data.py
and yields (input_ids, target_ids) tensor pairs of shape (seq_len,).

Contract (from contracts/shard-format.md):
  - Shard files: flat little-endian uint16, no header.
  - target_ids[t] == input_ids[t+1]  for all t in [0, seq_len-2].
  - Each shard byte_size must be > 0 and divisible by 2.

Multi-worker DataLoader safety:
  __getitem__ opens its own np.memmap on every call — no shared file handles
  or mutable state exist between workers. This makes the dataset safe for
  DataLoader(num_workers=N) with any N without locking.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class ShardedDataset(Dataset):
    """
    Reads a collection of uint16 binary shard files as a flat token stream
    and yields (input_ids, target_ids) windows of length seq_len.

    Args:
        shard_paths: Sorted list of absolute paths to shard_NNNN.bin files.
        seq_len: Sequence length. 4096 for pretraining.

    Raises:
        ValueError: If any shard is missing, empty, or has odd byte size.
    """

    def __init__(self, shard_paths: list[str], seq_len: int) -> None:
        if not shard_paths:
            raise ValueError("shard_paths must not be empty.")

        self._seq_len = seq_len
        self._shard_paths: list[str] = list(shard_paths)
        self._shard_token_counts: list[int] = []

        for path in self._shard_paths:
            # Existence check
            if not os.path.exists(path):
                raise ValueError(f"Shard file not found: {path}")

            byte_size = os.path.getsize(path)
            if byte_size == 0:
                raise ValueError(f"Shard file is empty: {path}")
            if byte_size % 2 != 0:
                raise ValueError(
                    f"Shard file has odd byte size ({byte_size}), "
                    f"not uint16-aligned: {path}"
                )

            # Verify file is actually readable (no lock / corruption check)
            # Opens and closes immediately — safe to call from any worker.
            _ = np.memmap(path, dtype="uint16", mode="r")

            token_count = byte_size // 2
            self._shard_token_counts.append(token_count)

        # Cumulative token offsets for O(1) shard lookup in __getitem__
        self._cumulative: list[int] = []
        running = 0
        for count in self._shard_token_counts:
            self._cumulative.append(running)
            running += count
        self._total_tokens: int = running

    # ------------------------------------------------------------------
    # Dataset interface (T021)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of non-overlapping seq_len windows across all shards."""
        # Requires seq_len+1 tokens per window (one extra for target shift)
        if self._total_tokens < self._seq_len + 1:
            return 0
        return (self._total_tokens - 1) // self._seq_len

    # ------------------------------------------------------------------
    # Item access (T022)
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Return the idx-th (input_ids, target_ids) pair.

        Global token offset = idx * seq_len.
        We read seq_len + 1 consecutive tokens from the flat stream, then:
          input_ids  = tokens[0 : seq_len]
          target_ids = tokens[1 : seq_len + 1]

        Handles the case where the window spans a shard boundary by reading
        from two consecutive shards and concatenating.

        Args:
            idx: Window index in [0, len(self)).

        Returns:
            Tuple of (input_ids, target_ids), both torch.LongTensor of shape
            (seq_len,).
        """
        global_start = idx * self._seq_len
        need = self._seq_len + 1  # +1 for the shifted target

        tokens = self._read_tokens(global_start, need)
        input_ids = torch.tensor(tokens[:self._seq_len], dtype=torch.long)
        target_ids = torch.tensor(tokens[1 : self._seq_len + 1], dtype=torch.long)
        return input_ids, target_ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shard_for_offset(self, global_offset: int) -> int:
        """
        Return the shard index that contains global_offset (0-based).
        Uses a linear scan over cumulative offsets; fast enough for
        the typical number of shards (~1500).
        """
        shard_idx = 0
        for i in range(len(self._cumulative) - 1, -1, -1):
            if self._cumulative[i] <= global_offset:
                shard_idx = i
                break
        return shard_idx

    def _read_tokens(self, global_start: int, count: int) -> list[int]:
        """
        Read `count` tokens from the flat token stream starting at
        global_start. Spans shard boundaries transparently.

        Opens its own np.memmap per call — safe for DataLoader workers.
        """
        tokens: list[int] = []
        remaining = count
        offset = global_start

        while remaining > 0:
            shard_idx = self._shard_for_offset(offset)
            shard_start = self._cumulative[shard_idx]
            shard_len = self._shard_token_counts[shard_idx]
            local_offset = offset - shard_start
            available = shard_len - local_offset
            take = min(remaining, available)

            mm = np.memmap(
                self._shard_paths[shard_idx], dtype="uint16", mode="r"
            )
            tokens.extend(mm[local_offset : local_offset + take].tolist())
            del mm  # release handle immediately

            offset += take
            remaining -= take

            # If we've consumed this shard and still need more, move to next
            if remaining > 0 and shard_idx + 1 >= len(self._shard_paths):
                # Not enough tokens left in the dataset — pad with EOS (id=3)
                tokens.extend([3] * remaining)
                break

        return tokens

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ShardedDataset("
            f"shards={len(self._shard_paths)}, "
            f"total_tokens={self._total_tokens:,}, "
            f"seq_len={self._seq_len}, "
            f"len={len(self):,})"
        )
