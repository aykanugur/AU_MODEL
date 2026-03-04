# Epic 6 — Inference & Terminal Chat

| Field | Value |
|-------|-------|
| **Branch** | `epic/06-inference` |
| **Base branch** | `epic/05-sft` |
| **Merge target** | `main` |
| **PRD refs** | F-08, F-09, G6, M6 |
| **Depends on** | Epic 5 — needs `checkpoints/sft_best.pt` |
| **Status** | ⬜ Not started |
| **Output files** | `inference/generate.py`, `inference/chat.py`, `inference/__init__.py` |

---

## Goal

Build a `generate()` function with autoregressive sampling (temperature, top-k, top-p, repetition penalty) and a terminal REPL that maintains multi-turn conversation history within the 4096-token context window. No external generation libraries — pure PyTorch.

---

## Tasks

- [ ] **`inference/generate.py`** — `generate(model: AUModel, tokenizer: TurkishTokenizer, prompt_ids: List[int], max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.1) → List[int]`:
  - run under `torch.inference_mode()`
  - loop: `model(input_ids)` → logits of shape `(1, seq_len, vocab_size)` → take last position → divide by `temperature`
  - apply repetition penalty: divide logits at positions of already-generated token IDs by `repetition_penalty`
  - top-k filter: zero out all logits except top-k
  - top-p nucleus filter: sort descending, cumsum softmax, zero out tokens past `top_p` threshold
  - `torch.multinomial(probs, 1)` → next token ID
  - stop if next token == `EOS_ID` or `len(generated) >= max_new_tokens`
  - return list of new token IDs only (not including prompt)

- [ ] **`inference/chat.py`** — `ChatSession`:
  - `history: List[Dict[str, str]]` — list of `{"role": "user"|"assistant"|"system", "content": str}`
  - `format_context() → str` — concatenate turns as full chat template; if `len(tokenizer.encode(context)) > 3800` drop the oldest non-system turn and retry until ≤ 3800 tokens
  - `respond(user_input: str) → str` — add user turn to history, call `generate()`, decode result, add assistant turn to history, return decoded string
  - REPL loop: `while True`: `input("Sen: ")` → `respond()` → `print(f"AUModel: {response}")`; handle `KeyboardInterrupt` with `print("\nGörüşmek üzere!")` and clean exit

---

## Exit Criteria

| Check | Pass Condition |
|-------|---------------|
| 200-token response latency (H100) | ≤ 10 seconds (measured with `time.perf_counter()`) |
| 5-turn context retention | Response to turn 5 correctly references content from turn 1 (manual test) |
| Context truncation | When conversation exceeds 3,800 tokens, oldest user/assistant turn is dropped; session does not crash |
| EOS stop | Generation stops at `EOS_ID` and returns; does not generate further tokens after EOS |
| Repetition penalty | Same 20-word phrase does not appear twice in a 200-token output with `repetition_penalty=1.1` |
| Top-p sum | `probs[kept_tokens].sum() >= top_p - 0.01` after nucleus filter |

---

## Unlocks

- **Epic 7** (MCP Web Search) — needs working `generate()` loop to intercept search tokens
- **Epic 8** (DPO) — needs working chat REPL to display A/B responses

---

_Last updated: 4 Mart 2026_
