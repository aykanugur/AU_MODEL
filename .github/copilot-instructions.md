# Copilot / AI Assistant Instructions

## Memory Bank Protocol

This project uses a **memory bank** — a set of markdown files in `/memory-bank/` that serve as persistent project context across sessions.

**At the start of every session, read these files in order:**

```
1. memory-bank/projectbrief.md     — What we're building, hard constraints
2. memory-bank/productContext.md   — Why, for whom, success criteria
3. memory-bank/systemPatterns.md   — Architecture decisions + patterns
4. memory-bank/techContext.md      — Tech stack, file structure, environment
5. memory-bank/activeContext.md    — Current focus, recent decisions, next steps
6. memory-bank/progress.md         — What's done, what's not, known issues
```

**Also reference:**
- `DESIGN.md` — Detailed pseudocode for every module

---

## How to Update the Memory Bank

When completing a task, always update the relevant memory bank files:

| Event | Update |
|-------|--------|
| Task completed | `progress.md` — mark task ✅ |
| New decision made | `activeContext.md` — add to decisions log |
| Architecture changed | `systemPatterns.md` — update patterns |
| New file/dependency added | `techContext.md` — update file structure |
| Phase completed | `progress.md` — update phase status bar |
| Starting a new session | `activeContext.md` — update "Current Focus" |
| Ending a session | `activeContext.md` — update "What We Just Did" + "What Needs To Be Done Next" |

---

## Coding Guidelines

### Always
- Check `DESIGN.md` before implementing any file — pseudocode is there
- Verify `model/config.py` values are consistent with tokenizer `vocab_size`
- Use `BF16` (never FP16) for all training
- Add `bias=False` to all `nn.Linear` layers in the model
- Use `ignore_index=-100` in SFT cross-entropy loss

### Never
- Auto-implement files unless the user explicitly asks
- Change locked hyperparameters without discussing tradeoffs
- Use absolute positional embeddings (we use RoPE)
- Use LayerNorm (we use RMSNorm)
- Use FP16 for training

### Colab-Specific
- All checkpoints must go to Google Drive paths
- Sessions can disconnect — resume must be seamless
- `torch.compile(model)` must be called after model init
- Always set `torch.backends.cuda.matmul.allow_tf32 = True`

---

## Project Phases (Quick Reference)

```
Phase 0: Design        ✅ DONE
Phase 1: Tokenizer     ← START HERE
Phase 2: Model Code
Phase 3: Data Pipeline
Phase 4: Pretraining   (100hrs H100)
Phase 5: SFT
Phase 6: Inference/Chat
```

---

## Quick Parameter Reference

```python
vocab_size    = 32000
d_model       = 2048
num_heads     = 16
num_kv_heads  = 8
num_layers    = 24
ffn_hidden    = 5504
max_seq_len   = 4096
# ~1.3B params

batch_size    = 8
grad_accum    = 16     # effective batch = 128 seqs
lr            = 3e-4
lr_min        = 3e-5
warmup        = 2000 steps
max_steps     = 100_000
dtype         = bfloat16
```
