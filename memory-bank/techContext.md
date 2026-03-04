# Technical Context

## Development Environment

| Item | Value |
|------|-------|
| OS (local) | macOS |
| Training environment | Google Colab Pro |
| GPU | NVIDIA H100 80GB SXM5 |
| Python | 3.10+ |
| PyTorch | 2.2+ |
| CUDA | 12.x |

## Hardware Specs & Performance

### H100 80GB
- BF16 TFLOPS: 756
- Memory bandwidth: 3.35 TB/s
- Effective throughput for 1.3B model: ~155,000 tokens/sec
- Memory usage (1.3B, batch=8): ~38GB / 80GB

### Training Budget
```
100 hours × 3600 sec × 155,000 tok/s = ~55.8B tokens reachable
Target: 32B tokens (conservative, 25× Chinchilla)
```

## Tech Stack

### Core
| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.2+ | Model, training, Flash Attention dispatch |
| `sentencepiece` | 0.1.99+ | Turkish BPE tokenizer |
| `numpy` | 1.26+ | Data shards (uint16 memmap) |

### Data
| Library | Version | Purpose |
|---------|---------|---------|
| `datasets` | 2.18+ | OSCAR, Wikipedia, mC4 from HuggingFace |
| `tqdm` | 4.66+ | Progress bars |

### Optional
| Library | Purpose |
|---------|---------|
| `wandb` | Training metrics dashboard |
| `flash-attn` | Manual Flash Attn install (auto via torch SDPA) |

## Project File Structure

```
AUModel/
├── memory-bank/          ← AI memory system (this folder)
│   ├── projectbrief.md
│   ├── productContext.md
│   ├── systemPatterns.md
│   ├── techContext.md
│   ├── activeContext.md
│   └── progress.md
│
├── data/
│   ├── raw/              ← Downloaded corpora
│   ├── processed/        ← shard_XXXX.bin (uint16 token IDs)
│   └── instruction/      ← SFT JSONL files
│
├── tokenizer/
│   ├── train_tokenizer.py
│   ├── turkish_bpe.model  (generated)
│   └── turkish_bpe.vocab  (generated)
│
├── model/
│   ├── config.py          ← ModelConfig dataclass
│   ├── rope.py            ← Rotary Position Embeddings
│   ├── attention.py       ← GQA attention
│   ├── feedforward.py     ← SwiGLU FFN
│   ├── transformer.py     ← Full AUModel
│   └── __init__.py
│
├── training/
│   ├── dataset.py         ← Shard streaming DataLoader
│   ├── lr_scheduler.py    ← Cosine + warmup
│   ├── checkpoint.py      ← Save/load/resume
│   ├── trainer.py         ← Main training loop
│   └── __init__.py
│
├── sft/
│   ├── sft_dataset.py     ← Chat format dataset
│   ├── sft_trainer.py     ← Instruction tuning loop
│   └── __init__.py
│
├── inference/
│   ├── generate.py        ← Sampling (temp, top-k, top-p)
│   ├── chat.py            ← Terminal chat interface
│   └── __init__.py
│
├── scripts/
│   ├── prepare_data.py    ← Download + clean + tokenize + shard
│   └── run_training.py    ← CLI entry point
│
├── colab/
│   ├── 01_tokenizer.ipynb
│   ├── 02_pretrain.ipynb
│   └── 03_sft.ipynb
│
├── DESIGN.md              ← Full technical design document
├── PRD_TEMPLATE.md        ← Product Requirements template
└── requirements.txt
```

## Data Sources

| Source | Size (Turkish) | Quality | HF Dataset ID |
|--------|---------------|---------|---------------|
| Turkish Wikipedia | ~1GB | High | `wikipedia` `20231101.tr` |
| OSCAR 23.01 | ~50GB | Medium | `oscar-corpus/OSCAR-2301` |
| mC4 Turkish | ~100GB | Medium | `mc4` language=`tr` |
| CC-100 Turkish | ~25GB | Medium | `cc100` language=`tr` |

Target total after dedup/filtering: **~30–50B tokens**

## Critical Colab Notes

1. **Always save to Google Drive** — Colab local storage resets on session end
2. **Sessions disconnect every ~8–12 hrs** — auto-resume must work from checkpoint
3. Enable TF32: `torch.backends.cuda.matmul.allow_tf32 = True`
4. Use `torch.compile(model)` — ~30% throughput gain on H100
5. BF16 only: never use FP16 for LLM pretraining (numerical instability)
6. `pip install flash-attn --no-build-isolation` at session start if needed

## Special Tokens (must match across tokenizer + model)

| Token | ID | Purpose |
|-------|----|---------|
| `[PAD]` | 0 | Padding |
| `[UNK]` | 1 | Unknown |
| `[BOS]` | 2 | Begin of sequence |
| `[EOS]` | 3 | End of sequence |
| `<\|system\|>` | TBD | Chat system role |
| `<\|user\|>` | TBD | Chat user role |
| `<\|assistant\|>` | TBD | Chat assistant role |
| `<\|endoftext\|>` | TBD | End of turn |

*TBD IDs are assigned after tokenizer training — update `model/config.py` after training.*
