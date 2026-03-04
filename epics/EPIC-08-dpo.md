# Epic 8 — DPO & Preference Collection

| Field | Value |
|-------|-------|
| **Branch** | `epic/08-dpo` |
| **Base branch** | `epic/06-inference` |
| **Merge target** | `main` |
| **PRD refs** | G9 |
| **Depends on** | Epic 5 (`sft_best.pt` as base and reference), Epic 6 (chat REPL for A/B display) |
| **Status** | ⬜ Not started |
| **Output files** | `sft/dpo_dataset.py` (new), `sft/dpo_trainer.py` (new), updated `inference/chat.py`, `data/preference/preference_log.jsonl`, `checkpoints/dpo_best.pt` |

---

## Goal

On every 2nd assistant turn in the terminal chat, generate two candidate responses (Response A conservative / Response B creative) and ask the user which is better. Log each choice as a `(prompt, chosen, rejected)` triplet. Once ≥ 10,000 preference pairs accumulate, run DPO fine-tuning (`β=0.1`) on top of `sft_best.pt` to produce `dpo_best.pt`.

---

## A/B Sampling Parameters

| Variant | `temperature` | `top_p` | Role |
|---------|--------------|---------|------|
| Response A | 0.7 | 0.9 | Conservative, lower variance |
| Response B | 1.1 | 0.85 | Creative, higher variance |

On odd turns: use Response A parameters only (no A/B).

---

## DPO Hyperparameters

```
base_checkpoint  = sft_best.pt
reference_model  = sft_best.pt (frozen, requires_grad=False)
β (beta)         = 0.1
optimizer        = AdamW(lr=1e-6, weight_decay=0.0)
batch_size       = 4
grad_accum       = 8   (effective batch = 32)
epochs           = 1
min_dataset_size = 10,000 preference pairs (hard gate)
```

---

## Tasks

- [ ] **Update `inference/chat.py`** — add `turn_counter: int = 0` to `ChatSession`; on `respond()`: if `turn_counter % 2 == 0`: call `generate()` twice with A-params then B-params; print `\n[A] {response_a}\n\n[B] {response_b}\n`; prompt `"Hangisi daha iyi? (A/B): "`; read single char with `sys.stdin.read(1).strip().upper()`; determine `chosen`/`rejected` based on user input; append `{"prompt": context, "chosen": chosen_text, "rejected": rejected_text, "timestamp": datetime.utcnow().isoformat()}` to `data/preference/preference_log.jsonl`; increment `turn_counter`; on odd turns: use A-params, skip A/B display
- [ ] **`sft/dpo_dataset.py`** — `DPODataset(Dataset)`: load `preference_log.jsonl`; `assert len(data) >= 10000`, raise `ValueError('Need ≥ 10,000 preference pairs to run DPO. Currently: {len(data)}')` if not met; format `prompt` as full chat template up to `<|assistant|>`; tokenize `prompt + chosen` and `prompt + rejected`; pad both to `max_len=2048` with `pad_id=0`; return `(prompt_ids, chosen_ids, rejected_ids, chosen_mask, rejected_mask)`
- [ ] **`sft/dpo_trainer.py`** — load two copies of `sft_best.pt`: policy `π_θ` (trainable) and reference `π_ref` (all `requires_grad=False`); for each batch compute:
  ```
  chosen_logprob   = sum(log_softmax(π_θ(chosen_ids)) * chosen_mask)
  rejected_logprob = sum(log_softmax(π_θ(rejected_ids)) * rejected_mask)
  ref_chosen       = sum(log_softmax(π_ref(chosen_ids)) * chosen_mask).detach()
  ref_rejected     = sum(log_softmax(π_ref(rejected_ids)) * rejected_mask).detach()

  log_ratio = (chosen_logprob - ref_chosen) - (rejected_logprob - ref_rejected)
  loss = -F.logsigmoid(β * log_ratio).mean()
  ```
  log loss every 50 steps; save `checkpoints/dpo_best.pt` at end of epoch

---

## Exit Criteria

| Check | Pass Condition |
|-------|---------------|
| DPO training guard | `DPODataset.__init__` raises `ValueError` if `len(data) < 10,000` |
| Reference model frozen | `assert not any(p.requires_grad for p in ref_model.parameters())` |
| JSONL schema | Every entry in `preference_log.jsonl` has keys `prompt`, `chosen`, `rejected`, `timestamp` |
| DPO loss | Decreases monotonically over 1 epoch (no spike > 0.5 above running average) |
| Head-to-head eval | `dpo_best.pt` wins ≥ 60% of 20 manual side-by-side comparisons vs `sft_best.pt` |
| `dpo_best.pt` exists | File loadable via `torch.load('checkpoints/dpo_best.pt')` |

---

## Unlocks

- **Epic 9** (Web Deployment) — needs `dpo_best.pt` as the final model

---

_Last updated: 4 Mart 2026_
