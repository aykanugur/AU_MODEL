# Quickstart: Turkish Native Tokenizer

**Feature**: 001-turkish-tokenizer | **Phase 1 quickstart** | **Date**: 2026-03-04

---

## Prerequisites

| Requirement | Value |
|-------------|-------|
| Python | 3.10+ |
| `sentencepiece` | `>=0.1.99` |
| `datasets` | `>=2.18.0` |
| `tqdm` | `>=4.66.0` |
| Disk space | ~10 GB free (7.5 GB corpus + ~500 MB intermediate files + ~128 MB artifacts) |
| RAM | ≥12 GB recommended (SPM training with 10M sentence cap) |
| Time | ~30-40 minutes on Colab CPU for SPM training pass |

Install dependencies:
```bash
pip install sentencepiece>=0.1.99 datasets>=2.18.0 tqdm>=4.66.0
```

---

## Step 1: Build the Corpus

```bash
python tokenizer/train_tokenizer.py --download
```

This step:
1. Downloads the Turkish Wikipedia dump via HuggingFace `datasets` library
2. Applies NFC normalization to each document
3. Strips HTML tags, URLs, and boilerplate footers
4. Filters documents with fewer than 200 characters
5. Writes the filtered corpus to `data/raw/tokenizer_corpus.txt` (UTF-8 plain text, one document per line)

**Expected output**:
```
Downloading wikipedia/20231101.tr ... 100%|████████████████████| 7.5GB
Filtering documents ... 100%|████████████████████| 1.23M docs
Kept: 1.18M documents (95.9%)
Corpus written to data/raw/tokenizer_corpus.txt
Total size: 7.4 GB
```

**Expected duration**: ~10-15 minutes on Colab (network dependent)

**If you have additional corpus sources** (OSCAR, CC-100, custom text), concatenate them into `data/raw/tokenizer_corpus.txt` before Step 2:
```bash
cat data/raw/oscar_tr.txt >> data/raw/tokenizer_corpus.txt
```

---

## Step 2: Train the Tokenizer

```bash
python tokenizer/train_tokenizer.py --train
```

This step:
1. Checks that `data/raw/tokenizer_corpus.txt` exists (aborts if missing)
2. Checks that `tokenizer/turkish_bpe.model` does NOT already exist (aborts without `--force` flag)
3. Calls `SentencePieceTrainer.train()` with locked parameters:
   - `vocab_size=64000`, `model_type=bpe`, `character_coverage=0.9999`
   - `normalization_rule_name=identity`, `byte_fallback=True`
   - `input_sentence_size=10_000_000`, `shuffle_input_sentence=True`, `random_seed=42`
   - `pad_id=0, unk_id=1, bos_id=2, eos_id=3`
   - `user_defined_symbols=[SYSTEM],[USER],[ASSISTANT],[SEP]`
4. Writes `tokenizer/turkish_bpe.model` (~5 MB)
5. Writes `tokenizer/turkish_bpe.vocab` (~3 MB)

**Expected output**:
```
Training SentencePiece BPE tokenizer...
  vocab_size=64000, character_coverage=0.9999
  input_sentence_size=10000000, random_seed=42
Training complete.
Model saved: tokenizer/turkish_bpe.model (4.8 MB)
Vocab saved: tokenizer/turkish_bpe.vocab (2.9 MB)
```

**Expected duration**: ~25-35 minutes on Colab CPU  
**Expected duration**: ~5-8 minutes on macOS M2

**Re-run protection**: If `tokenizer/turkish_bpe.model` already exists, the script will refuse to overwrite it:
```
ERROR: tokenizer/turkish_bpe.model already exists. Use --force to overwrite.
```
To retrain (e.g. after corpus expansion), pass `--force`:
```bash
python tokenizer/train_tokenizer.py --train --force
```

---

## Step 3: Validate

```bash
python tokenizer/validate_tokenizer.py
```

This step runs 4 checks in report-all mode:

| Check | Threshold | Test |
|-------|-----------|------|
| Fertility | ≤ 1.4 tokens/word | 10,000 random sentences from corpus |
| Round-trip | 100% (100/100 strings) | 100 diverse Turkish strings |
| Turkish chars | All 12 chars as direct tokens | ğ Ğ ş Ş ç Ç ı I ö Ö ü Ü |
| Special tokens | All 8 IDs distinct, range 0-7 | BOS/EOS/PAD/UNK + 4 chat tokens |

**Expected output (all passing)**:
```
[PASS] fertility:       0.9823 (threshold: 1.4)
[PASS] round_trip:      100/100 strings lossless
[PASS] turkish_chars:   12/12 characters have direct tokens
[PASS] special_tokens:  IDs 0-7 all distinct

All checks passed. Tokenizer ready for use.
```

**If fertility check fails**:
```
[FAIL] fertility:       1.52 (threshold: 1.4)

Remediation: Fertility > 1.4 indicates insufficient Turkish text in the corpus.
Options:
  1. Add more Turkish text (OSCAR, CC-100, news corpora) and retrain
  2. Verify the corpus has at least 200M words of natural Turkish text
  3. Check that normalization did not mangle characters

Exiting with code 1 (gate failure).
```

**Exit codes**:
- `0` — all checks passed; tokenizer is training-ready
- `1` — one or more checks failed; **do not proceed to Epic 2**
- `2` — runtime error (model file missing, import error, etc.)

---

## Step 4: (Colab) Copy to Google Drive

If running on Colab, copy the artifacts to Google Drive immediately after successful validation:

```python
import shutil
from google.colab import drive

drive.mount("/content/drive")
base = "/content/drive/MyDrive/AUModel/tokenizer"

import os; os.makedirs(base, exist_ok=True)
shutil.copy("tokenizer/turkish_bpe.model", f"{base}/turkish_bpe.model")
shutil.copy("tokenizer/turkish_bpe.vocab", f"{base}/turkish_bpe.vocab")
print("Artifacts saved to Google Drive.")
```

**Why**: Colab sessions disconnect after idle timeout. The model file (~5 MB) takes no meaningful time to copy. If the session disconnects before copying, you must retrain (~35 minutes).

---

## Using the Tokenizer (Post-Training)

```python
from tokenizer import Tokenizer

tok = Tokenizer("tokenizer/turkish_bpe.model")

# Encode
ids = tok.encode("Türkçe bir model eğitiyoruz.")
print(ids)          # [4521, 112, 6789, ...]

# Decode
text = tok.decode(ids)
print(text)         # "Türkçe bir model eğitiyoruz."

# Special tokens
print(tok.vocab_size)     # 64000
print(tok.pad_id)         # 0
print(tok.bos_id)         # 2
print(tok.system_id)      # 4
print(tok.user_id)        # 5
print(tok.assistant_id)   # 6
print(tok.sep_id)         # 7
```

---

## Colab Notebook

The full workflow is also available as a Colab notebook:

```
colab/01_tokenizer.ipynb
```

Open in Colab, mount Drive, run all cells in order.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Training OOM | Corpus too large for RAM | Already handled: `input_sentence_size=10_000_000` caps training sentences |
| Fertility > 1.4 | Insufficient Turkish text | Add more corpus — OSCAR Turkish, CC-100-tr, or custom |
| Round-trip failure | NFC not applied upstream | Check corpus builder normalization step |
| Turkish chars have `<unk>` | `character_coverage < 1.0` for specific char | Verify `character_coverage=0.9999` and `byte_fallback=True` in training call |
| `FileNotFoundError` on `Tokenizer()` | Model path wrong | Use path relative to repo root, or pass absolute path |
| SPM silently overwrote model | Forgot `--force` check | The script guards against this; if it happened, corpus was identical |
