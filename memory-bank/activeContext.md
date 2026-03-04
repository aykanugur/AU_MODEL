# Active Context

> **Update this file at the start and end of every work session.**

## Current Focus

**Phase:** Pre-implementation  
**Current task:** Memory bank created, ready to begin coding  
**Last updated:** 4 Mart 2026

## What We Just Did

- Designed full LLaMA-style 1.3B architecture for Turkish LLM
- Locked in all hyperparameters for H100 100-hour budget
- Created project scaffold (all directories + empty `.py` files)
- Created `DESIGN.md` with full pseudocode for every module
- Created memory bank system for AI-assisted SDLC

## What We Decided (Key Decisions Log)

| Decision | Reasoning |
|----------|-----------|
| 1.3B params (not 700M) | H100 makes 1.3B feasible in 100hrs (25× Chinchilla) |
| H100 (not A100) | 2.8× faster → changes optimal model size entirely |
| max_seq_len=4096 | Chatbot needs longer context than 2048 |
| GQA num_kv_heads=8 | Halves KV cache, no quality loss |
| SentencePiece BPE 32k | Turkish chars need dedicated tokens (not byte fallback) |
| From scratch (not fine-tune) | True Turkish tokenizer + architecture ownership |

## What Needs To Be Done Next

**Immediate next step: Start with Phase 1 — Tokenizer**

```
1. Collect Turkish corpus (Wikipedia dump is fastest to start)
2. Implement tokenizer/train_tokenizer.py
3. Train tokenizer, validate fertility ratio (target: 1.3–1.8 tok/word)
4. Update special token IDs in model/config.py after training
```

## Active Questions / Blockers

| Question | Status |
|----------|--------|
| Colab H100 availability — is it consistently available? | Open |
| Turkish Wikipedia dump size — enough to train tokenizer? | Open — ~1GB, should be OK |
| OSCAR 23.01 access — HuggingFace login needed? | Open |
| SFT dataset — how many Turkish instruction pairs available? | Open |

## Recent Context (AI should know this)

- All `.py` files are currently **empty** — implementation not started
- `DESIGN.md` contains detailed pseudocode for every file — always reference it
- `requirements.txt` is populated
- The user wants to **write the code themselves** — AI should guide, review, explain — not auto-implement unless asked
- Training will run on **Colab** — every file must be importable from Drive mount

## Notes for Next Session

- Check `DESIGN.md` for implementation details before writing any module
- Always validate `model/config.py` values are consistent with tokenizer vocab_size after tokenizer training
- Run `model/transformer.py` sanity check (forward pass + parameter count) before starting data pipeline
