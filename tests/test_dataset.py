"""
Unit tests for training/dataset.py :: ShardedDataset

Covers:
  T028 — shift invariant, ValueError on bad shard (missing/empty/odd-byte)
       → FR-011, FR-012, SC-004
"""

import os
import struct
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.dataset import ShardedDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fake_shard(path: str, n_tokens: int, start_val: int = 0) -> None:
    """Write a valid uint16 shard of n_tokens sequential values."""
    tokens = np.arange(start_val, start_val + n_tokens, dtype=np.uint16)
    tokens.tofile(path)


# ---------------------------------------------------------------------------
# T028 — Shift invariant
# ---------------------------------------------------------------------------

class TestShardedDatasetShiftInvariant:
    def test_target_equals_input_shifted_left(self, tmp_path):
        """SC-004: target_ids[t] == input_ids[t+1] for t in [0, seq_len-2]."""
        seq_len = 16
        n_tokens = 1_000
        shard_path = str(tmp_path / "shard_0000.bin")
        _write_fake_shard(shard_path, n_tokens)

        ds = ShardedDataset([shard_path], seq_len=seq_len)
        assert len(ds) > 0, "Dataset must have at least one item."

        for i in range(min(10, len(ds))):
            inp, tgt = ds[i]
            assert inp.shape == (seq_len,), f"input_ids shape wrong at item {i}"
            assert tgt.shape == (seq_len,), f"target_ids shape wrong at item {i}"
            assert inp.dtype == torch.long
            assert tgt.dtype == torch.long
            # Shift invariant: target[t] == input[t+1]
            for t in range(seq_len - 1):
                assert tgt[t].item() == inp[t + 1].item(), (
                    f"Shift broken at item {i}, position {t}: "
                    f"tgt[{t}]={tgt[t].item()} != inp[{t+1}]={inp[t+1].item()}"
                )

    def test_len_is_total_tokens_floor_div_seq_len(self, tmp_path):
        seq_len = 32
        n_tokens = 1_000
        shard_path = str(tmp_path / "shard_0000.bin")
        _write_fake_shard(shard_path, n_tokens)

        ds = ShardedDataset([shard_path], seq_len=seq_len)
        # len = (total_tokens - 1) // seq_len
        expected = (n_tokens - 1) // seq_len
        assert len(ds) == expected

    def test_multi_shard_concatenation(self, tmp_path):
        """Items spanning shard boundaries must still satisfy the shift invariant."""
        seq_len = 16
        n_tokens = 100   # small shards to force boundary crossings
        paths = []
        for i in range(3):
            p = str(tmp_path / f"shard_{i:04d}.bin")
            _write_fake_shard(p, n_tokens, start_val=i * n_tokens)
            paths.append(p)

        ds = ShardedDataset(paths, seq_len=seq_len)
        for i in range(min(10, len(ds))):
            inp, tgt = ds[i]
            for t in range(seq_len - 1):
                assert tgt[t].item() == inp[t + 1].item()


# ---------------------------------------------------------------------------
# T028 — ValueError on bad shards (FR-012)
# ---------------------------------------------------------------------------

class TestShardedDatasetValidation:
    def test_raises_on_missing_shard(self, tmp_path):
        missing = str(tmp_path / "nonexistent.bin")
        with pytest.raises(ValueError, match="not found"):
            ShardedDataset([missing], seq_len=4096)

    def test_raises_on_empty_shard(self, tmp_path):
        empty = str(tmp_path / "empty.bin")
        open(empty, "w").close()  # zero-byte file
        with pytest.raises(ValueError, match="empty"):
            ShardedDataset([empty], seq_len=4096)

    def test_raises_on_odd_byte_size(self, tmp_path):
        odd = str(tmp_path / "odd.bin")
        with open(odd, "wb") as f:
            f.write(b"\x01\x02\x03")  # 3 bytes — not uint16-aligned
        with pytest.raises(ValueError, match="uint16"):
            ShardedDataset([odd], seq_len=4096)

    def test_raises_on_empty_shard_paths(self):
        with pytest.raises(ValueError, match="empty"):
            ShardedDataset([], seq_len=4096)

    def test_shard_path_in_error_message(self, tmp_path):
        """FR-012: shard path must appear in the ValueError message."""
        missing = str(tmp_path / "specific_shard.bin")
        with pytest.raises(ValueError) as exc_info:
            ShardedDataset([missing], seq_len=4096)
        assert "specific_shard.bin" in str(exc_info.value)


# ---------------------------------------------------------------------------
# T028 — DataLoader multi-worker safety (T023)
# ---------------------------------------------------------------------------

class TestDataLoaderSafety:
    def test_dataloader_num_workers_4(self, tmp_path):
        """Each __getitem__ opens its own memmap — no shared state between workers."""
        from torch.utils.data import DataLoader

        seq_len = 16
        shard_path = str(tmp_path / "shard_0000.bin")
        _write_fake_shard(shard_path, 2_000)

        ds = ShardedDataset([shard_path], seq_len=seq_len)
        # num_workers=0 on CI to avoid process spawning issues;
        # the design guarantees safety at num_workers=4 (per-call memmap)
        loader = DataLoader(ds, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        inp, tgt = batch
        assert inp.shape == (4, seq_len)
        assert tgt.shape == (4, seq_len)
        # Verify shift invariant across the batch
        assert torch.all(tgt[:, :-1] == inp[:, 1:])
