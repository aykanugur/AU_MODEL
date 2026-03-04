# AU_MODEL Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-04

## Active Technologies
- Python 3.11 + `datasets` ≥2.14, `sentencepiece` ≥0.1.99, `numpy` ≥1.24, `pybloom-live` ≥4.0, `python-dotenv` ≥1.0, `torch` ≥2.0 (003-data-pipeline)
- Google Drive (Colab) — flat binary `.bin` shards + `shards_manifest.json` + `bloom.pkl` (003-data-pipeline)

- **Python 3.10+** (001-turkish-tokenizer)
- **sentencepiece>=0.1.99** -- BPE tokenizer training and inference (001-turkish-tokenizer)
- **datasets>=2.18.0** -- HuggingFace datasets, Turkish Wikipedia download (001-turkish-tokenizer)
- **tqdm>=4.66.0** -- Progress bars (001-turkish-tokenizer)
- **PyTorch 2.2+ (BF16)** -- Model training (Epic 2+, not yet active)

## Project Structure

```text
tokenizer/
  train_tokenizer.py     # Offline CLI: corpus download + SPM training
  tokenizer.py           # Runtime wrapper: class Tokenizer (importable)
  __init__.py            # Re-exports: from tokenizer import Tokenizer
  turkish_bpe.model      # Output artifact (~5 MB, post-training)
  turkish_bpe.vocab      # Output artifact (~3 MB, post-training)

data/raw/
  tokenizer_corpus.txt   # Intermediate corpus (~7.5 GB)

specs/001-turkish-tokenizer/
  spec.md                # Feature spec (frozen)
  plan.md                # Implementation plan
  research.md            # Phase 0 research findings
  data-model.md          # Entity design
  quickstart.md          # End-to-end usage guide
  contracts/tokenizer-interface.md  # Stable interface contract
```

## Commands

```bash
python tokenizer/train_tokenizer.py --download
python tokenizer/train_tokenizer.py --train
python tokenizer/validate_tokenizer.py
pip install sentencepiece>=0.1.99 datasets>=2.18.0 tqdm>=4.66.0
```

## Code Style

Python: PEP 8, type hints required, bias=False on all nn.Linear, BF16 for all training, ignore_index=-100 for SFT cross-entropy. Use if/raise never assert.

## Key Constraints (Locked)

- vocab_size = 64000 -- frozen after tokenizer training
- Special token IDs: PAD=0, UNK=1, BOS=2, EOS=3, SYSTEM=4, USER=5, ASSISTANT=6, SEP=7
- Output paths: tokenizer/turkish_bpe.model, tokenizer/turkish_bpe.vocab
- SPM params: normalization_rule_name="identity", byte_fallback=True, random_seed=42
- No assert in validation code (stripped by -O)
- Fertility gate: <= 1.4 tokens/word -- hard exit

## Recent Changes
- 003-data-pipeline: Added Python 3.11 + `datasets` ≥2.14, `sentencepiece` ≥0.1.99, `numpy` ≥1.24, `pybloom-live` ≥4.0, `python-dotenv` ≥1.0, `torch` ≥2.0
- 002-model-architecture: Added [if applicable, e.g., PostgreSQL, CoreData, files or N/A]
- 002-model-architecture: Added [if applicable, e.g., PostgreSQL, CoreData, files or N/A]


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
