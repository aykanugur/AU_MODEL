# Stories — Epic 5: Supervised Fine-Tuning (SFT)

**Epic ref:** `EPIC-05-sft.md`
**Branch:** `epic/05-sft`
**Persona:** Developer
**Total stories:** 5

---

## ST-05-01 — Instruction Pair Translation

**As a developer,**
I want Alpaca-52k and Dolly-15k instruction pairs automatically translated to Turkish,
So that I have a large-scale Turkish instruction dataset without manually translating 67,000 examples.

### Acceptance Criteria

- Running the translation pipeline produces `data/instruction/alpaca_tr.jsonl` and `data/instruction/dolly_tr.jsonl`.
- Each line in both files contains exactly two keys: `"instruction"` and `"response"`, both in Turkish.
- The pipeline reads the DeepL API key from the `DEEPL_API_KEY` environment variable — no hardcoded credentials.
- The total entry count across the two files is ≥ 67,000 pairs (52k Alpaca + 15k Dolly).
- The pipeline logs the count of translated pairs and the count of any pairs that failed or were skipped.

---

## ST-05-02 — Quality Filtering

**As a developer,**
I want translated pairs with very short responses automatically removed,
So that the model does not train on near-empty answers that would degrade output quality.

### Acceptance Criteria

- Any response whose Turkish word count is fewer than 20 is discarded from the dataset.
- The filter logs the number of discarded pairs per file.
- After filtering, 500 random pairs from each file are printed to stdout for manual review.
- The combined dataset is still ≥ 66,000 pairs after filtering (reasonable discard tolerance).

---

## ST-05-03 — Manual Turkish Instruction Pairs

**As a developer,**
I want a verified set of ≥ 5,000 manually curated Turkish Q&A pairs written to a JSONL file,
So that the model trains on high-quality, culturally accurate Turkish content that machine translation cannot reliably produce.

### Acceptance Criteria

- `data/instruction/manual_tr.jsonl` contains ≥ 5,000 entries.
- Each entry has `"instruction"` and `"response"` keys in Turkish.
- Domain breakdown is met: ≥ 1,000 Turkish geography/history, ≥ 500 Turkish grammar explanations, ≥ 500 math word problems in Turkish, ≥ 500 coding help in Turkish, ≥ 2,500 general factual Q&A.
- Every entry in the file is valid JSON (parseable without errors).

---

## ST-05-04 — SFT Dataset with Loss Masking

**As a developer,**
I want the SFT dataset to mask loss on all prompt tokens so the model only learns to predict responses,
So that fine-tuning teaches the model what to say, not how to repeat the instruction it was given.

### Acceptance Criteria

- Every training example is formatted using the exact chat template:
  ```
  <|system|>
  Sen yardımcı bir Türkçe yapay zeka asistanısın.
  <|user|>
  {instruction}
  <|assistant|>
  {response}
  <|endoftext|>
  ```
- The `labels` tensor uses `-100` for every token from the start through and including the `<|assistant|>` token.
- The `labels` tensor uses real token IDs for the response and `<|endoftext|>` tokens only.
- On average, more than 30% of label positions per batch are `-100` (prompts are the majority of tokens).
- Sequences longer than 2,048 tokens are truncated from the left, not the right.

---

## ST-05-05 — SFT Trainer and Checkpoint

**As a developer,**
I want an SFT training loop that loads the pretrained checkpoint, trains for 3 epochs on ≥ 72,000 instruction pairs, and saves the best checkpoint,
So that I have a single `sft_best.pt` file that the inference and DPO epics can build upon.

### Acceptance Criteria

- The trainer loads `ckpt_033000.pt` as the starting point.
- Training runs for 3 epochs over all instruction pairs with batch_size=8 and grad_accum=4.
- The cross-entropy loss is computed only on positions where `labels != -100`.
- After each epoch, validation loss is computed on a held-out 10% split of the instruction data.
- The checkpoint with the lowest validation loss across all epochs is saved as `checkpoints/sft_best.pt`.
- `sft_best.pt` is loadable after training completes.
- A human spot-check of 20 randomly sampled model outputs rates average quality ≥ 4.0 out of 5.0.
- Turkish grammar errors appear in ≤ 10% of 20 sampled outputs.

---

_Epic complete when all 5 stories pass their acceptance criteria._
_Total instruction dataset target: ≥ 72,000 pairs across all three JSONL files._
_Last updated: 4 Mart 2026_
