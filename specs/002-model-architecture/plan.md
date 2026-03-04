# Implementation Plan: AUModel Transformer Architecture

**Branch**: `002-model-architecture` | **Date**: 2026-03-04 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-model-architecture/spec.md`

## Summary

Implement a LLaMA-3-style decoder-only transformer in PyTorch as the `model/` package. The model uses Grouped Query Attention (GQA, 12 query heads / 6 KV heads), Rotary Position Embeddings (RoPE via `torch.polar` complex rotation, `rope_theta=500000`), SwiGLU feedforward, RMSNorm (no LayerNorm), and tied embeddings. Default config produces **~700,317,696 parameters** (constitution-locked). Delivered as an importable package `from model import AUModel, ModelConfig` with a standalone CLI sanity check (`model/sanity_check.py`) and a Colab GPU validation notebook (`colab/02_model.ipynb`).

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: PyTorch ≥ 2.1 (uses `F.scaled_dot_product_attention` stable API and `torch.compile`); no extra packages required at this phase  
**Storage**: N/A (no persistence in this module; checkpoints handled by Epic 4 `training/checkpoint.py`)  
**Testing**: Manual CLI (`python model/sanity_check.py`); Colab notebook cells as integration tests; no pytest at this phase  
**Target Platform**: Linux/CUDA (Google Colab H100/A100) for training; macOS CPU for development/sanity checks  
**Project Type**: Python library (importable package)  
**Performance Goals**: Instantiation < 30s on ≥8 GB RAM machine; forward pass `(8, 512)` < 10s on CPU (≥8 cores, ≥16 GB RAM)  
**Constraints**: BF16 on GPU (FP16 FORBIDDEN); `bias=False` on all `nn.Linear`; no training logic inside `AUModel`; `torch.compile` called by trainer (Epic 4), not model; MUST NOT use `assert` (use `if/raise ValueError` per constitution)  
**Scale/Scope**: Single GPU / single node; ~700M params in BF16 ≈ 1.4 GB GPU memory for weights alone

## Constitution Check

*GATE: Re-checked after `/speckit.plan` Phase 1 design — **PASS***

| Invariant | Spec Value | Status |
|-----------|-----------|--------|
| `vocab_size` | 64,000 | ✅ |
| `d_model` | 1,536 | ✅ |
| `num_heads` | 12 | ✅ |
| `num_kv_heads` | 6 | ✅ |
| `num_layers` | 24 | ✅ |
| `ffn_hidden_dim` | 4,352 | ✅ |
| `max_seq_len` | 4,096 | ✅ |
| `rope_theta` | 500,000 | ✅ |
| Total params | ~700,317,696 | ✅ |
| BF16 training | Required (FP16 forbidden) | ✅ |
| `bias=False` on all Linear | Required | ✅ |
| RoPE (no absolute PE) | Required | ✅ |
| RMSNorm (no LayerNorm) | Required | ✅ |
| Type hints on all signatures | Required | ✅ |
| MUST NOT use `assert` | Required | ✅ |

**Violation record**: During gate check, spec originally contained stale 1.3B values (`d_model=2048`, `num_heads=16`, etc.) from an outdated `DESIGN.md`. All values corrected to match PRD v1.3 and constitution. See `§Constitution Gate — Violation Record` in spec.md.

## Project Structure

### Documentation (this feature)

```text
specs/002-model-architecture/
├── plan.md              ← This file
├── spec.md              ← Feature spec (17 FRs, 6 SCs, 3 user stories)
├── research.md          ← Phase 0: 6 architectural decisions
├── data-model.md        ← Phase 1: 7 entities with tensor shapes
├── quickstart.md        ← Phase 1: 7-step usage guide
├── tasks.md             ← Phase 2: 15 tasks across 6 phases
└── contracts/
    └── model-interface.md  ← Stable public API for Epics 3–6
```

### Source Code (repository root)

```text
model/
├── __init__.py          ← Public exports: AUModel, ModelConfig
├── config.py            ← ModelConfig dataclass + GQA validation
├── rope.py              ← compute_freqs_cis() pure function (torch.polar RoPE)
├── attention.py         ← Attention: GQA + RoPE + F.scaled_dot_product_attention
├── feedforward.py       ← FeedForward: SwiGLU (W1/W2/W3, bias=False)
├── transformer.py       ← RMSNorm, TransformerBlock, AUModel
└── sanity_check.py      ← CLI: 4 checks, exits 0 on pass (FR-015)

colab/
└── 02_model.ipynb       ← GPU sanity check + single-batch overfit test (FR-016)
```

**Structure Decision**: Single-project layout, no `src/` wrapper. `model/` is importable directly from repo root, matching the convention of Epic 1's `tokenizer/` package. Verification via CLI script and Colab notebook; no pytest at this phase.

## Complexity Tracking

No constitution violations remain. All architecture choices match invariants.

| Phase 0 Decision | Rationale |
|-----------------|-----------|
| RoPE via `torch.polar` | Fused complex multiplication; avoids explicit sin/cos tables |
| GQA via `repeat_interleave(groups, dim=2)` | Correct interleaved grouping; produces contiguous memory |
| Attention via `F.scaled_dot_product_attention(is_causal=True)` | Auto-dispatches to Flash Attention 2 on CUDA; no extra install |
| `ffn_hidden_dim=4352` | Frozen constitution value (≈ 8/3 × d_model rounded up to multiple of 64) |
| PyTorch default weight init | Sufficient at 700M scale; custom init optional in Epic 4 |
| `past_kv=None` on `Attention.forward()` | Zero training overhead; avoids breaking interface change in Epic 6 |
