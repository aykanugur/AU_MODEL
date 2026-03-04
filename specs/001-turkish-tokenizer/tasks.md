# Tasks: Turkish Native Tokenizer

**Feature**: 001-turkish-tokenizer | **Branch**: `001-turkish-tokenizer`  
**Input**: specs/001-turkish-tokenizer/plan.md, spec.md, data-model.md, contracts/tokenizer-interface.md  
**Generated**: 2026-03-04

---

## Format

- `- [ ] TXXX` -- required task ID
- `[P]` -- parallelizable (different file, no dependency on incomplete sibling tasks)
- `[USn]` -- maps to User Story n from spec.md

---

## Phase 1: Setup

**Purpose**: Create the package skeleton and directory structure so all downstream files have a home.

- [x] T001 Create package skeleton: `tokenizer/__init__.py` (empty placeholder — T011 will add re-export content, do not add logic here), `data/raw/.gitkeep` (ensures raw data directory is tracked in git)

---

## Phase 2: User Story 1 — Model Builder Trains a Tokenizer (P1)

**Story goal**: A developer can run `python tokenizer/train_tokenizer.py --download && python tokenizer/train_tokenizer.py --train` and get valid `tokenizer/turkish_bpe.model` and `tokenizer/turkish_bpe.vocab` files.

**Independent test**: After running the two CLI commands successfully, `tokenizer/turkish_bpe.model` and `tokenizer/turkish_bpe.vocab` exist and load without errors. `sentencepiece.SentencePieceProcessor().load("tokenizer/turkish_bpe.model"); assert sp.get_piece_size() == 64000` passes.

- [x] T002 [US1] Implement `download_corpus()` in `tokenizer/train_tokenizer.py`: Stream Turkish Wikipedia 20231101.tr via HuggingFace `datasets`, apply NFC normalization to each document, skip documents < 200 chars, write one document per line to `data/raw/tokenizer_corpus.txt` with progress bar (tqdm); after download, verify the output file exists and is > 1 MB — raise `RuntimeError` and refuse to return if file is missing or empty (guard against partial/failed download per edge case in spec)
- [x] T003 [US1] Implement `run_spm_training()` in `tokenizer/train_tokenizer.py`: Check `os.path.exists()` guard (refuse to overwrite without `--force`), call `SentencePieceTrainer.train()` with locked params: `model_type=bpe, vocab_size=64000, character_coverage=0.9999, normalization_rule_name=identity, byte_fallback=True, input_sentence_size=10_000_000, shuffle_input_sentence=True, random_seed=42, pad_id=0, unk_id=1, bos_id=2, eos_id=3, user_defined_symbols=[SYSTEM],[USER],[ASSISTANT],[SEP]`, write to `tokenizer/turkish_bpe.model` and `tokenizer/turkish_bpe.vocab`
- [x] T004 [US1] Implement `main()` CLI entry point in `tokenizer/train_tokenizer.py`: `argparse` with `--download` (calls `download_corpus()`), `--train` (calls `run_spm_training()`), `--corpus <path>` optional flag (skips download and uses an existing plain-text file as corpus, per FR-008: pipeline MUST accept any plain-text input), `--force` flag (bypasses overwrite guard for existing model file), `if __name__ == "__main__": main()` guard

---

## Phase 3: User Story 2 — Developer Verifies Turkish Language Efficiency (P2)

**Story goal**: A developer can run `python tokenizer/validate_tokenizer.py` and see a pass/fail summary for all 4 checks. Exit code 0 = all pass; exit code 1 = gate failure(s); exit code 2 = runtime error.

**Independent test**: Running `python tokenizer/validate_tokenizer.py` against a valid trained model exits with code 0 and prints `[PASS]` for all 4 lines. Running against a model with fertility > 1.4 exits with code 1 and prints `[FAIL] fertility`.

- [x] T005 [US2] Create `tokenizer/validate_tokenizer.py`: Define `ValidationResult` dataclass (`check_name: str, passed: bool, measured_value: float | None, threshold: float | None, message: str`), implement `check_fertility(sp, corpus_path, n=10_000)` that samples `n` sentences, tokenizes each, computes avg tokens-per-word ratio, returns `ValidationResult` (threshold: 1.4, V-001)
- [x] T006 [US2] Add `check_roundtrip(sp, test_strings: list[str]) -> ValidationResult` to `tokenizer/validate_tokenizer.py`: For each string call `sp.decode(sp.encode(s)) == s`; collect all failures without stopping early (report-all); return `ValidationResult` with `passed=(failures==0)` and count in message (V-002); test strings are 100 items hardcoded inline: 25 common Turkish words (merhaba, güzel, çalışmak, şehir, öğrenci...), 25 Turkish sentences, 25 strings with numerals/punctuation, 25 edge cases (empty string, single char, all-ASCII, mixed Turkish/Latin)
- [x] T007 [US2] Add `check_turkish_chars(sp) -> ValidationResult` to `tokenizer/validate_tokenizer.py`: For each of the 12 exact Unicode chars `['ç','ğ','ı','İ','ö','ş','ü','Ü','Ö','Ç','Ğ','Ş']` (from FR-003), verify `sp.id_to_piece(sp.piece_to_id(char)) == char` and the piece does NOT match `r'^<0x[0-9A-Fa-f]+>$'` (not a byte fallback); collect all failures without stopping; return `ValidationResult` with `passed=(failures==0)` (V-003)
- [x] T008 [US2] Add `check_special_tokens(sp)` to `tokenizer/validate_tokenizer.py`: Verify IDs for `<pad>`=0, `<unk>`=1, `<s>`=2, `</s>`=3, `[SYSTEM]`=4, `[USER]`=5, `[ASSISTANT]`=6, `[SEP]`=7; check all 8 IDs are distinct and in range 0-63999; return `ValidationResult` (V-004)
- [x] T009 [US2] Implement `main()` report-all loop in `tokenizer/validate_tokenizer.py`: Parse model path arg (default `tokenizer/turkish_bpe.model`), wrap load in try/except (sys.exit(2) on error), run all 4 check functions collecting `ValidationResult` list, print summary table with `[PASS]`/`[FAIL]` status for each check, print numeric values to stdout and prose + remediation hints to stderr, call `sys.exit(1)` once at end if any check failed

---

## Phase 4: User Story 3 — Other Pipeline Components Consume the Tokenizer (P3)

**Story goal**: Any downstream epic can do `from tokenizer import Tokenizer; tok = Tokenizer("tokenizer/turkish_bpe.model")` and call `tok.encode()`, `tok.decode()`, `tok.vocab_size`, `tok.pad_id`, etc. with correct results.

**Independent test**: `from tokenizer import Tokenizer; tok = Tokenizer("tokenizer/turkish_bpe.model"); assert tok.vocab_size == 64000; assert tok.pad_id == 0; assert tok.bos_id == 2; assert tok.decode(tok.encode("merhaba")) == "merhaba"` all pass.

- [x] T010 [P] [US3] Implement complete `Tokenizer` class in `tokenizer/tokenizer.py`: `__init__(self, model_path: str | Path)` loads SPM (raises `FileNotFoundError` if missing), `encode(text: str, add_bos=False, add_eos=False) -> list[int]`, `decode(ids: list[int], skip_special=True) -> str` (filters special IDs `{pad_id, bos_id, eos_id, system_id, user_id, assistant_id, sep_id}` before decoding when `skip_special=True`), `@property vocab_size -> int`, `@property` for each of 8 special IDs: `pad_id`, `unk_id`, `bos_id`, `eos_id`, `system_id`, `user_id`, `assistant_id`, `sep_id` (all delegate to `self._sp.piece_to_id()`)
- [x] T011 [P] [US3] Implement `tokenizer/__init__.py` re-export: `from .tokenizer import Tokenizer` and `__all__ = ["Tokenizer"]` so downstream code uses `from tokenizer import Tokenizer`

---

## Phase 5: Polish & Cross-Cutting Concerns

- [x] T012 [P] Create `colab/01_tokenizer.ipynb`: 4-section notebook following quickstart.md — (1) install deps + mount Drive, (2) `--download` corpus build, (3) `--train` SPM training, (4) `validate_tokenizer.py` + copy artifacts to Drive; include expected outputs and timing estimates per cell
- [x] T013 Update `memory-bank/progress.md` to mark Epic 1 tokenizer tasks complete; update `specs/001-turkish-tokenizer/plan.md` status from Draft to Implementation-Ready

---

## Dependencies

```
T001 (setup)
  └─> T002 [US1: download corpus]
        └─> T003 [US1: train SPM]
              └─> T004 [US1: CLI]
                    └─> T005 [US2: fertility check]  ─────────────────┐
                          └─> T006 [US2: roundtrip]                   │
                                └─> T007 [US2: turkish chars]          │
                                      └─> T008 [US2: special tokens]  │
                                            └─> T009 [US2: main()]    │
                                                                        │
T001 (setup)                                                            │
  └─> T010 [US3: Tokenizer class]  <── can code in parallel with US1/US2
  └─> T011 [US3: __init__.py]      <── can code in parallel with US1/US2

T009 + T011 both complete
  └─> T012 [Polish: Colab notebook]
        └─> T013 [Polish: progress.md update]
```

**Coding dependency note**: T010 and T011 can be written before the tokenizer model file exists — they only require the model file at *test time*. This means US3 code can be implemented in parallel with US1 during the `impt` phase.

---

## Parallel Execution Examples

### US1 + US3 in parallel (different files, no runtime dependency)

```
Worker A: T002 → T003 → T004   (tokenizer/train_tokenizer.py)
Worker B: T010 → T011           (tokenizer/tokenizer.py + __init__.py)
```

### US2 checks (sequential — same file, but logically independent)

```
T005 → T006 → T007 → T008 → T009   (tokenizer/validate_tokenizer.py)
```

---

## Implementation Strategy

**MVP scope (deliver first)**: US1 only — train the tokenizer. All other epics are blocked by this.
1. T001 → T002 → T003 → T004 → validate manually that `tokenizer/turkish_bpe.model` loads with 64k vocab
2. Then US2 (T005-T009) to get the formal validation gate
3. Then US3 (T010-T011) to unlock downstream epics

**Suggested `impt` order**: T001 → T002 → T003 → T004 || T010 → T011 → T005 → T006 → T007 → T008 → T009 → T012 → T013

---

## Summary

| Metric | Count |
|--------|-------|
| Total tasks | 13 |
| US1 tasks (P1) | 3 |
| US2 tasks (P2) | 5 |
| US3 tasks (P3) | 2 |
| Setup tasks | 1 |
| Polish tasks | 2 |
| Parallelizable [P] tasks | 3 (T010, T011, T012) |
| MVP scope (US1 only) | T001, T002, T003, T004 |
