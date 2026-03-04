# Epic 5 — Supervised Fine-Tuning (SFT)

| Field | Value |
|-------|-------|
| **Branch** | `epic/05-sft` |
| **Base branch** | `epic/04-pretraining` |
| **Merge target** | `main` |
| **PRD refs** | F-06, F-07, G5, M5 |
| **Depends on** | Epic 4 — needs `ckpt_033000.pt` |
| **Status** | ⬜ Not started |
| **Output files** | `sft/sft_dataset.py`, `sft/sft_trainer.py`, `sft/__init__.py`, `data/instruction/alpaca_tr.jsonl`, `data/instruction/dolly_tr.jsonl`, `data/instruction/manual_tr.jsonl`, `checkpoints/sft_best.pt` |

---

## Goal

Translate Alpaca-52k and Dolly-15k to Turkish, manually curate ≥ 5,000 Turkish Q&A pairs, then fine-tune AUModel 700M for 3 epochs on ≥ 72,000 instruction pairs. Loss is masked to assistant response tokens only — the model must never learn to predict the prompt.

---

## Chat Format

Every training example uses this exact template:
```
<|system|>
Sen yardımcı bir Türkçe yapay zeka asistanısın.
<|user|>
{instruction}
<|assistant|>
{response}
<|endoftext|>
```
`labels` tensor: `-100` for all tokens through `<|assistant|>` (inclusive); real token IDs for `{response}<|endoftext|>`.

---

## SFT Hyperparameters

```
base_checkpoint  = ckpt_033000.pt
optimizer        = AdamW(lr=5e-5, weight_decay=0.0, β1=0.9, β2=0.999)
epochs           = 3
batch_size       = 8
grad_accum       = 4   (effective batch = 32)
dtype            = bfloat16
save_strategy    = after each epoch, keep best by SFT val loss
```

---

## Tasks

- [ ] **Translation pipeline** — download `tatsu-lab/alpaca` and `databricks/dolly-15k` from HuggingFace; translate `instruction` + `output` fields to Turkish via DeepL API (`deepl.Translator(api_key=os.environ['DEEPL_API_KEY'])`); save as `data/instruction/alpaca_tr.jsonl` and `data/instruction/dolly_tr.jsonl`; each line: `{"instruction": str, "response": str}`
- [ ] **Quality filter** — discard pairs where Turkish `response` word count < 20; random-sample 500 pairs from each file and print for manual spot-check; log discard count
- [ ] **Manual curation** — write ≥ 5,000 Turkish Q&A pairs to `data/instruction/manual_tr.jsonl`; domain breakdown: Turkish geography/history ≥ 1,000, Turkish grammar explanations ≥ 500, math word problems in Turkish ≥ 500, coding help asked in Turkish ≥ 500, general factual Q&A ≥ 2,500
- [ ] **`sft/sft_dataset.py`** — `SFTDataset(Dataset)`: load all three JSONL files; format each pair using the chat template above; tokenize full string with `TurkishTokenizer.encode()`; build `labels` tensor with `-100` masking up to and including the `<|assistant|>` token; return `(input_ids, labels)` both of length ≤ 2048 (truncate from the left if longer)
- [ ] **`sft/sft_trainer.py`** — load `ckpt_033000.pt` into `AUModel`; training loop with above hyperparameters; cross-entropy loss computed only on positions where `labels != -100`; evaluate on 10% held-out split after each epoch; save checkpoint if val loss is best seen; final checkpoint saved as `checkpoints/sft_best.pt`

---

## Exit Criteria

| Check | Pass Condition |
|-------|---------------|
| Dataset size | ≥ 72,000 pairs total across all three JSONL files |
| Labels masking | `(labels == -100).float().mean() > 0.3` for every batch (prompt tokens are majority) |
| SFT val loss | Lower than pretrain baseline on instruction val split |
| Human eval spot-check | ≥ 4.0 / 5.0 average on 20 manually reviewed outputs (author-rated) |
| Grammar error rate | ≤ 10% on 20 sampled outputs (manual check) |
| `sft_best.pt` exists | File is present and loadable via `torch.load()` |

---

## Unlocks

- **Epic 6** (Inference & Chat) — needs `sft_best.pt`
- **Epic 8** (DPO) — needs `sft_best.pt` as base and reference model

---

_Last updated: 4 Mart 2026_
