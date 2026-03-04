# Implementation Plan: Turkish Native Tokenizer

**Branch**: `001-turkish-tokenizer` | **Date**: 2026-03-04 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-turkish-tokenizer/spec.md`

---

## Summary

Train a 64,000-vocabulary Turkish-native SentencePiece BPE tokenizer on a plain-text Turkish corpus (default: Turkish Wikipedia dump, additional sources at developer discretion). The output artifacts — `tokenizer/turkish_bpe.model` and `tokenizer/turkish_bpe.vocab` — are the single gate blocking all downstream epics (model architecture, data pipeline, pretraining). A second output file, `tokenizer/tokenizer.py`, provides the stable wrapper interface consumed at runtime by every other pipeline component.

Technical approach: SentencePiece BPE, `vocab_size=64000`, `character_coverage=0.9999`, `byte_fallback=True`, `normalization_rule_name=identity` (NFC applied upstream), `input_sentence_size=10_000_000`, `random_seed=42`. Validation runs report-all (4 checks), exits non-zero on any failure.

---

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: `sentencepiece>=0.1.99`, `datasets>=2.18.0`, `tqdm>=4.66.0`
**Storage**: Local disk (~7.5 GB corpus, ~128 MB model artifacts); Google Drive for Colab persistence
**Testing**: Manual validation script (no pytest -- one-shot offline training artifact)
**Target Platform**: Local macOS (development) + Google Colab Pro H100 (execution)
**Project Type**: CLI training script + importable Python module
**Performance Goals**: Training completes in <= 40 min on Colab CPU; validation script runs in <= 5 min
**Constraints**: Vocabulary fixed at 64,000 -- cannot change after this epic; PAD/UNK/BOS/EOS IDs 0-3 locked; output paths locked to `tokenizer/turkish_bpe.*`
**Scale/Scope**: Single training run; produces one .model file consumed by all downstream epics

---

## Constitution Check

*Note: `.specify/memory/constitution.md` content sourced from PRD v1.3 and memory-bank constraints.*

| Gate | Rule (source) | Status |
|------|--------------|--------|
| BF16 training only | PRD -- applies to model training, not tokenizer | N/A for this epic |
| No bias in nn.Linear | PRD -- model architecture constraint | N/A for this epic |
| vocab_size = 64,000 locked | PRD v1.3 -- must not change after this epic | PASS -- enforced in FR-002 |
| Output paths locked | spec.md FR-001b | PASS -- enforced |
| No pre-existing tokenizer reuse | spec.md FR-001 | PASS -- enforced |
| Checkpoint to Google Drive | PRD Colab constraint | PASS -- covered in quickstart |
| Fertility <= 1.4, hard gate | spec.md FR-005, FR-011 | PASS -- exits 1 on failure |
| All 12 Turkish chars covered | spec.md FR-003 | PASS -- enforced |
| Round-trip lossless | spec.md FR-004 | PASS -- enforced |

**Constitution Check: PASS** -- no violations.

---

## Project Structure

### Documentation (this feature)

```
specs/001-turkish-tokenizer/
  plan.md                       <- This file
  research.md                   <- Phase 0 output
  data-model.md                 <- Phase 1 output
  quickstart.md                 <- Phase 1 output
  contracts/
    tokenizer-interface.md      <- Phase 1 output
  checklists/
    requirements.md
    spec.md
```

### Source Code (repository root)

```
tokenizer/
  train_tokenizer.py    <- Offline script: corpus download + SPM training
  tokenizer.py          <- Runtime wrapper: Tokenizer class
  __init__.py           <- Re-exports: from tokenizer import Tokenizer
  turkish_bpe.model     <- Output artifact: SentencePiece binary model
  turkish_bpe.vocab     <- Output artifact: human-readable vocabulary (TSV)

data/
  raw/
    tokenizer_corpus.txt   <- Intermediate: plain-text training corpus (~7.5 GB)

colab/
  01_tokenizer.ipynb    <- End-to-end execution notebook
```

**Structure Decision**: Script + module split. `train_tokenizer.py` is the offline CLI (run once). `tokenizer.py` is the importable runtime wrapper reused by every downstream epic. Separate files so `SentencePieceTrainer` is never imported at inference time. Standard LLaMA/nanoGPT pattern.

---

## Phase 0 -- Research

*See [research.md](research.md) for full findings and alternatives evaluated.*

| Decision | Chosen | Rationale |
|----------|--------|-----------|
| normalization_rule_name | identity | NFC applied upstream; NFKC would be lossy second pass |
| byte_fallback | True | Guarantees zero unk at inference |
| input_sentence_size | 10_000_000 | Prevents OOM; consistent with LLaMA/mT5 practice |
| random_seed | 42 | Reproducible runs; SPM param name is random_seed= |
| Overwrite guard | os.path.exists() pre-check | SPM silently overwrites; must protect existing artifacts |
| Validation exit codes | 1=gate fail, 2=runtime error | Shell-distinguishable; follows pytest/mypy convention |
| Validation strategy | Report-all 4 checks, single sys.exit | Fail-fast hides downstream failures; report-all shows full picture |
| Wrapper design | Class with @property special IDs | Properties delegate to SPM; always authoritative |
| decode() default | skip_special=True | Production behavior; False for debugging |
| File split | train_tokenizer.py (offline) + tokenizer.py (runtime) | LLaMA/nanoGPT convention |

---

## Phase 1 -- Design & Contracts

*See [data-model.md](data-model.md), [contracts/tokenizer-interface.md](contracts/tokenizer-interface.md), [quickstart.md](quickstart.md).*

### Interface Contract Summary

`tokenizer/tokenizer.py` exposes `class Tokenizer`:

```python
class Tokenizer:
    def __init__(self, model_path: str | Path) -> None: ...
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...
    def decode(self, ids: list[int], skip_special: bool = True) -> str: ...
    @property def vocab_size(self) -> int: ...    # 64000
    @property def pad_id(self) -> int: ...        # 0
    @property def unk_id(self) -> int: ...        # 1
    @property def bos_id(self) -> int: ...        # 2
    @property def eos_id(self) -> int: ...        # 3
    @property def system_id(self) -> int: ...     # 4
    @property def user_id(self) -> int: ...       # 5
    @property def assistant_id(self) -> int: ...  # 6
    @property def sep_id(self) -> int: ...        # 7
```

Downstream import: `from tokenizer import Tokenizer`

### Checklist Blockers Addressed

| ID | Blocker | Resolution |
|----|---------|-----------|
| CHK002 | SPM overwrite behavior | os.path.exists() guard + --force flag |
| CHK006 | Wrapper interface undefined | Defined in contracts/tokenizer-interface.md |
| CHK007 | Chat token strings not locked | [SYSTEM],[USER],[ASSISTANT],[SEP] locked in FR-007 |
| CHK012 | Validators need non-zero exit | sys.exit(1)/2 pattern in T009 |
| CHK016 | Method signatures missing | Defined in contracts/tokenizer-interface.md |
| CHK017 | Downstream import pattern | from tokenizer import Tokenizer via __init__.py |

---

## Complexity Tracking

*No constitution violations -- section not applicable.*
