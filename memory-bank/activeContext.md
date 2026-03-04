# Active Context

> **Update this file at the start and end of every work session.**

## Current Focus

**Phase:** Phase 3 — Data Pipeline (next)
**Current task:** Epic 2 (model architecture) complete — ready to start Epic 3
**Last updated:** 4 Mart 2026

## What We Just Did

- Completed speckit.implement for Epic 2 (002-model-architecture):
  - Wrote all 6 model source files from scratch (config, rope, feedforward, attention, transformer, __init__)
  - Discovered and corrected spec arithmetic error: param count is 749,544,960 (~749.5M) not "700M"
  - Added LLaMA-style `_init_weights()` to AUModel (N(0,0.02) + scaled residual projections)
  - Wrote `model/sanity_check.py` — all 4 checks PASS, exits 0
  - Created `colab/02_model.ipynb` (7 cells)
  - SC-002 CPU benchmark: forward (8,512) = 5.98s < 10s ✅
  - AST type-hints audit: all public signatures annotated ✅
  - Updated spec.md, tasks.md, data-model.md, plan.md, contracts/model-interface.md with correct 749.5M count
  - All 15 tasks in tasks.md marked [X]
  - Committed as `7f956d0`

## What We Decided (Key Decisions Log)

| Decision | Reasoning |
|----------|-----------|
| vocab_size=64000 (not 32000) | Constitution amendment: richer Turkish coverage |
| d_model=1536, num_layers=24, num_heads=12, num_kv_heads=6 | ~749.5M params at H100 budget |
| ffn_hidden_dim=4352 | SwiGLU constraint: must be multiple of 64 |
| rope_theta=500000 | Extended context (YaRN-style); handles 4096 seq_len |
| AUModel._init_weights() | LLaMA-style init essential: without it, initial loss > 1000 |
| Param count 749.5M (not "700M") | Arithmetic verification confirmed; spec was wrong |
| bias=False on all nn.Linear | Standard LLaMA practice; reduces params + improves training |
| Weight tying (embed ↔ lm_head) | Reduces 98M params counted twice; standard for decoder LLMs |
| get_num_params() uses set() | Correctly deduplicates tied parameters |

## What Needs To Be Done Next

**Immediate next step: Start Epic 3 — Data Pipeline**

```
1. Run speckit.specify for Epic 3 (data pipeline)
2. Implement scripts/prepare_data.py (download + tokenize + shard)
3. Target: 30B+ tokens from Turkish Wikipedia + OSCAR + mC4
4. Shards to /content/drive/MyDrive/aumodel_checkpoints/data/
```

## Active Questions / Blockers

| Question | Status |
|----------|--------|
| Constitution amendment for param count (700M→750M) | Pending — low priority |
| ffn_hidden_dim field name drift (constitution uses `ffn_hidden`) | Pending |
| Colab H100 availability | Open |

## Recent Context (AI should know this)

- **Frozen hyperparameters** (do NOT change): vocab_size=64000, d_model=1536, num_heads=12, num_kv_heads=6, num_layers=24, ffn_hidden_dim=4352, max_seq_len=4096, rope_theta=500000
- **Param count**: 749,544,960 (verified analytically + by running model)
- **model/ package**: fully implemented and tested at commit `7f956d0`
- `model/sanity_check.py` is the standard validation script — run it after any model change
- Training will use `torch.compile(model)` — do this in trainer, NOT in AUModel
- All Colab checkpoints → `/content/drive/MyDrive/aumodel_checkpoints/`

## Notes for Next Session

- `model/` is complete — do not modify unless there is a bug
- Before starting data pipeline, reference DESIGN.md for `scripts/prepare_data.py` pseudocode
- The tokenizer model file is at `tokenizer/turkish_bpe.model` (and in Drive)
- Epic 3 needs tokenizer package import: `from tokenizer import Tokenizer`
