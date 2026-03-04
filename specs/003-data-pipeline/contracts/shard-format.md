# Contract: Shard File Format

**Consumer**: `training/dataset.py` (`ShardedDataset`)  
**Producer**: `scripts/prepare_data.py`  
**Type**: Binary file format contract

---

## File Naming

```
shard_NNNN.bin
```

- `NNNN` is a zero-padded 4-digit integer starting at `0000`
- Stored at: `<output_dir>/shard_NNNN.bin`

---

## File Format

| Property | Value |
|----------|-------|
| Encoding | Raw binary, little-endian |
| Element type | `uint16` (unsigned 16-bit integer) |
| Element size | 2 bytes |
| Layout | Flat 1D array — no header, no metadata |
| Token range | `[0, 63999]` — fits in uint16 |

**How to read**:
```python
import numpy as np
tokens = np.fromfile("shard_0000.bin", dtype=np.uint16)
# tokens.shape == (N,) where N ≈ 500_000_000
```

---

## Token Encoding

- BOS token `<s>` = **ID 2** — prepended to every document
- EOS token `</s>` = **ID 3** — appended to every document
- Documents are concatenated without separators beyond BOS/EOS
- Vocabulary: 64,000 tokens (IDs 0–63,999)

---

## Size Contract

| Property | Value |
|----------|-------|
| Target size | ~1 GB (≈ 500,000,000 tokens × 2 bytes) |
| Minimum size | > 0 bytes |
| Alignment | `byte_size % 2 == 0` (always — uint16 pairs) |
| Final shard | May be smaller than 1 GB |

---

## ShardedDataset API Contract

```python
class ShardedDataset(torch.utils.data.Dataset):
    def __init__(self, shard_paths: list[str], seq_len: int) -> None:
        """
        Args:
            shard_paths: Sorted list of absolute paths to shard_NNNN.bin files.
            seq_len: Sequence length. 4096 for pretraining.
        Raises:
            ValueError: If any shard is missing, empty, or has odd byte size.
        """

    def __len__(self) -> int:
        """Total number of non-overlapping seq_len windows across all shards."""

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Returns:
            (input_ids, target_ids): Both shape (seq_len,), dtype=torch.long
            Guarantee: target_ids[t] == input_ids[t+1] for t in [0, seq_len-2]
        """
```

---

## Manifest Contract

File: `<output_dir>/shards_manifest.json`

```json
{
  "version": 1,
  "bloom_filter_path": "<output_dir>/bloom.pkl",
  "shards": [
    {
      "filename": "shard_0000.bin",
      "token_count": 498234112,
      "sources": {"wikipedia": 498234112},
      "written_at": "2026-03-04T14:22:11Z"
    }
  ],
  "total_tokens": 498234112,
  "source_totals": {"wikipedia": 498234112}
}
```

The consumer (`ShardedDataset`) does **not** read the manifest — it only reads `.bin` files. The manifest is used solely by the producer (`prepare_data.py`) for resume logic.
