# Quickstart: Turkish Pretraining Data Pipeline

**Branch**: `003-data-pipeline`  
**Target environment**: H100 server (or Google Colab with Drive mounted)

---

## Prerequisites

1. Epic 1 complete — tokenizer at `tokenizer/turkish_bpe.model`
2. HF token in `.env` (OSCAR requires approval; use CC-100 fallback if not approved)
3. Google Drive mounted at `/content/drive` (Colab) or Drive path configured

```bash
# Install dependencies
pip install datasets sentencepiece numpy pybloom-live mmh3 python-dotenv torch
```

---

## Run — Single Source (MVP / Testing)

```bash
# Wikipedia only (~30 min on H100, ~3 hrs on Colab CPU)
python scripts/prepare_data.py --source wikipedia --output /content/drive/MyDrive/aumodel_checkpoints/data/
```

Expected output:
```
[wikipedia]  10000 docs |   8.2M tokens
[wikipedia]  20000 docs |  16.4M tokens
...
[Shard 0000/----] 500M tokens |  1.0 GB written | elapsed 00:28:14
...
=== DONE ===
Shards: 2 | Total tokens: 812,345,678 | Sources: wikipedia=812,345,678
Elapsed: 00:45:11
```

---

## Run — Full Corpus

```bash
python scripts/prepare_data.py --source all --output /content/drive/MyDrive/aumodel_checkpoints/data/
```

---

## Resume After Interruption

Re-run the **exact same command**. The pipeline reads `shards_manifest.json` and `bloom.pkl` from the output directory and continues from where it stopped:

```bash
# Same command — automatically resumes
python scripts/prepare_data.py --source all --output /content/drive/MyDrive/aumodel_checkpoints/data/
```

---

## Verify Corpus

```python
import numpy as np, os, glob

shard_dir = "/content/drive/MyDrive/aumodel_checkpoints/data/"
shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.bin")))

total_tokens = sum(np.fromfile(s, dtype=np.uint16).size for s in shards)
print(f"Shards: {len(shards)} | Total tokens: {total_tokens:,}")
# Expected: ≥1500 shards | ≥30,000,000,000 tokens
```

---

## Use in Training (Epic 4)

```python
from training.dataset import ShardedDataset
from torch.utils.data import DataLoader
import glob

shard_paths = sorted(glob.glob("/content/drive/MyDrive/aumodel_checkpoints/data/shard_*.bin"))
dataset = ShardedDataset(shard_paths, seq_len=4096)
loader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True)

for input_ids, target_ids in loader:
    # input_ids: (8, 4096)  target_ids: (8, 4096)
    pass
```

---

## Environment Variables

| Variable | Example | Required |
|----------|---------|---------|
| `HF_TOKEN` | `hf_abc123...` | For OSCAR (gated). Fallback to CC-100 if not set. |

Set in `.env` file (gitignored):
```
HF_TOKEN=hf_your_token_here
```
