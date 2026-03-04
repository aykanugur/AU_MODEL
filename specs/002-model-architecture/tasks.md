# Tasks: AUModel Transformer Architecture

**Input**: Design documents from `/specs/002-model-architecture/`
**Prerequisites**: spec.md ✅ | plan.md ✅ | research.md ✅ | data-model.md ✅ | contracts/model-interface.md ✅
**Constitution values**: `vocab_size=64000, d_model=1536, num_heads=12, num_kv_heads=6, num_layers=24, ffn_hidden_dim=4352, max_seq_len=4096, rope_theta=500000, ~700M params`

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no unsatisfied dependencies)
- **[Story]**: User story this task belongs to (US1, US2, US3)
- Exact file paths included in every task description

---

## Phase 1: Setup

**Purpose**: Create the `model/` package skeleton and `colab/` directory so all downstream tasks have a home.

- [ ] T001 Create `model/` package with empty `model/__init__.py` and `colab/` directory
- [ ] T002 Add `model/` module to `.gitignore` exclusions audit — confirm no `__pycache__` tracked (depends on T001)

**Checkpoint**: `model/` and `colab/` directories exist; `python -c "import model"` executes without ImportError (empty package).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core building blocks used by every user story. No US work begins until this phase is complete.

**⚠️ CRITICAL**: Attention, FeedForward, and AUModel ALL depend on these three modules.

- [ ] T003 Implement `ModelConfig` dataclass with `__post_init__` validation (`num_heads % num_kv_heads == 0` → `ValueError`) in `model/config.py`
- [ ] T004 [P] Implement `compute_freqs_cis(seq_len: int, head_dim: int, theta: float) -> torch.Tensor` pure function (returns complex64 `(seq_len, head_dim//2)` via `torch.polar`; no `nn.Module`, no `register_buffer`) in `model/rope.py`
- [ ] T005 [P] Implement `RMSNorm` layer (forward: `x / sqrt(mean(x²)+1e-6) * weight`, learnable `weight (d_model,)`) in `model/transformer.py`

**Checkpoint**: Foundation ready — all three modules importable. `ModelConfig()` raises `ValueError` on bad GQA ratio. `compute_freqs_cis(128, 64)` returns complex64 tensor of shape `(128, 64)`.

---

## Phase 3: User Story 1 — Instantiate and Run Forward Pass (Priority: P1) 🎯 MVP

**Goal**: A researcher runs `from model import AUModel, ModelConfig; m = AUModel(ModelConfig()); logits, _ = m(tokens)` and gets logits shaped `(B, T, 64000)` with no errors.

**Independent Test**: `python -c "import torch; from model import AUModel, ModelConfig; m = AUModel(ModelConfig()); t = torch.randint(0, 64000, (2, 128)); logits, _ = m(t); expected = (2, 128, 64000); (print('US1 PASS') if logits.shape == torch.Size(expected) else (_ for _ in ()).throw(SystemExit(f'BAD SHAPE: {tuple(logits.shape)} != {expected}')))"`

### Implementation for User Story 1

- [ ] T006 [P] [US1] Implement `FeedForward` module (SwiGLU: `W2(SiLU(W1(x)) * W3(x))`, all `nn.Linear` with `bias=False`, hidden=4352) in `model/feedforward.py`
- [ ] T007 [P] [US1] Implement `Attention` module (GQA via `repeat_interleave(groups, dim=2)`, apply RoPE via complex view-multiply, `F.scaled_dot_product_attention(is_causal=True)`, `past_kv=None` parameter returning `(out, None)` or `(out, (new_k, new_v))`) in `model/attention.py`
- [ ] T008 [US1] Implement `TransformerBlock` (pre-norm pattern: `x = x + attn(norm1(x), freqs_cis)` → `x = x + ffn(norm2(x))`) in `model/transformer.py`
- [ ] T009 [US1] Implement `AUModel` (`nn.Embedding(64000, 1536)`, 24 `TransformerBlock`s, final `RMSNorm`, `nn.Linear(1536, 64000, bias=False)` LM head, weight tying `lm_head.weight = embed.weight`, call `self.register_buffer("freqs_cis", compute_freqs_cis(...))` in `__init__`, `forward(tokens: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]` with guard `if tokens.shape[1] > self.config.max_seq_len: raise ValueError(f"seq_len {tokens.shape[1]} exceeds max_seq_len {self.config.max_seq_len}")` before embed, returning `(logits, loss|None)`, `get_num_params() -> int`) in `model/transformer.py`
- [ ] T010 [US1] Wire public exports in `model/__init__.py` (`from .transformer import AUModel; from .config import ModelConfig; __all__ = ["AUModel", "ModelConfig"]`)

**Checkpoint**: US1 fully functional. `AUModel(ModelConfig())` instantiates in < 30s. Forward pass on `(2, 128)` returns `(2, 128, 64000)`. `model.get_num_params()` returns value between 680M–720M. Bad GQA config raises `ValueError`.

---

## Phase 4: User Story 2 — Verify Correct Initial Loss (Priority: P2)

**Goal**: Confirm untrained model produces cross-entropy loss ∈ [10.0, 11.0] (expected: `log(64000) ≈ 10.77`), validating unbiased initialization and correct weight tying.

**Independent Test**: `python model/sanity_check.py` exits with code 0 and prints `PASS` for all 4 checks.

### Implementation for User Story 2

- [ ] T011 [US2] Implement `model/sanity_check.py` CLI script with 4 sequential checks: (1) import+instantiate `AUModel(ModelConfig())`, (2) forward shape check `(2,128) → (2,128,64000)`, (3) param count in [680M, 720M] via `get_num_params()`, (4) initial loss check: mean cross-entropy on 10 random batches ∈ [10.0, 11.0]; print `[PASS]`/`[FAIL]` per check; `sys.exit(0)` on all pass, `sys.exit(1)` on any failure. **Constitution**: use `if/raise ValueError(...)` NOT `assert` anywhere in this file

**Checkpoint**: `python model/sanity_check.py` exits with code 0, all 4 lines print `[PASS]`. Satisfies FR-015 and SC-006.

---

## Phase 5: User Story 3 — Overfit a Single Batch (Priority: P3)

**Goal**: 50 AdamW gradient steps on one fixed batch drive training loss from ~10.77 → < 0.1, confirming correct gradient flow through all 24 layers.

**Independent Test**: Run the overfit cell in `colab/02_model.ipynb` on GPU; confirm loss at step 50 < 0.1 and clean-batch loss stays near 10.77.

### Implementation for User Story 3

- [ ] T012 [US3] Create `colab/02_model.ipynb` with 6 cells: (1) GPU/BF16 setup + dependency check (PyTorch ≥ 2.1), (2) model import + `AUModel(ModelConfig()).to("cuda").bfloat16()`, (3) 4-check sanity block (shape, params, initial loss — mirrors `sanity_check.py`), (4) single-batch overfit loop (8 seqs × 512 tokens, 50 AdamW steps `lr=1e-3`, log loss every 10 steps), (5) clean-batch inference check (loss stays ≈ 10.77 on unseen batch), (6) results summary table

**Checkpoint**: Notebook runs end-to-end on Colab T4/A100 GPU. Loss at step 50 < 0.1. Satisfies FR-016 and SC-005.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final verification and memory-bank update.

- [ ] T013 Run `python model/sanity_check.py` in a clean Python environment (only PyTorch installed) and confirm it exits 0 — validates SC-006 independence; additionally time a forward pass on batch `(8, 512)` and confirm it completes in < 10s on >=8-core CPU (SC-002)
- [ ] T014 [P] Update `memory-bank/progress.md` to mark Epic 2 (model architecture) as ✅ complete, recording the actual date on which T011 and T012 both pass
- [ ] T015 [P] Add type annotations to all public function and method signatures across `model/config.py`, `model/rope.py`, `model/attention.py`, `model/feedforward.py`, `model/transformer.py`, `model/sanity_check.py` (constitution MUST: "type hints on all function signatures")

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — **BLOCKS Phases 3–5**
- **Phase 3 (US1)**: Depends on Phase 2 completion. T006 and T007 are parallel; T008 depends on T006+T007; T009 depends on T008; T010 depends on T009
- **Phase 4 (US2)**: Depends on Phase 3 completion (needs `AUModel.forward(tokens, targets)` working)
- **Phase 5 (US3)**: Depends on Phase 4 (sanity_check.py provides the overfit baseline)
- **Phase 6 (Polish)**: Depends on all desired stories complete. T013, T014, T015 are parallel within this phase.

### User Story Dependencies

```
T001 → T002 (sequential: T002 audits what T001 creates)
         ↓
T003 + T004 + T005  (foundational, parallelizable)
         ↓
T006 + T007  (parallel, both need T003–T005)
         ↓
T008  (needs T006 + T007)
         ↓
T009  (needs T008)
         ↓
T010  (needs T009)
         ↓
T011  (needs T010 — forward with targets)
         ↓
T012  (needs T011 — sanity check as reference)
         ↓
T013 + T014 + T015  (parallel, polish)
```

### Within Each User Story

- T006 (FeedForward) and T007 (Attention) are parallel — different files
- T003 (ModelConfig), T004 (RoPE pure function), T005 (RMSNorm) are parallel — different scopes
- T004 provides `compute_freqs_cis()` used by T009 (`AUModel.__init__` calls `self.register_buffer(..., compute_freqs_cis(...))`)
- T013, T014, and T015 are parallel — different concerns

---

## Parallel Execution Examples

### Phase 2: Foundational (3-way parallel)

```text
T003: model/config.py       — ModelConfig dataclass + validation
T004: model/rope.py         — compute_freqs_cis() pure function (no nn.Module)
T005: model/transformer.py  — RMSNorm class only
```

### Phase 3 partial parallel (T006 + T007)

```text
T006: model/feedforward.py  — FeedForward (SwiGLU)
T007: model/attention.py    — Attention (GQA + RoPE + SDPA)
→ then T008 (TransformerBlock, needs both)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup)
2. Complete Phase 2 (Foundational) — CRITICAL
3. Complete Phase 3 (US1) — forward pass works
4. **STOP and VALIDATE**: `python -c "from model import AUModel, ModelConfig; ..."` passes
5. Continue to US2/US3 if model checks out

### Incremental Delivery

1. **After T010**: Working model importable with correct shapes → Epic 4 can begin integration testing
2. **After T011**: `sanity_check.py` provides CI-ready verification → run in every future Epic
3. **After T012**: Colab notebook enables GPU validation before 100-hour pretraining
4. **After T015**: All public signatures type-annotated → constitution compliance confirmed

### Key Architectural Constraints (from research.md — do not deviate)

| Constraint | Value |
|-----------|-------|
| RoPE implementation | `torch.polar` complex multiplication |
| GQA expansion | `repeat_interleave(kv_groups, dim=2)` |
| Attention kernel | `F.scaled_dot_product_attention(is_causal=True)` |
| All `nn.Linear` | `bias=False` |
| Normalization | `RMSNorm` only (no LayerNorm, no BatchNorm) |
| Activation | SwiGLU (`SiLU(W1(x)) * W3(x)` → `W2(...)`) |
| Weight tying | `lm_head.weight = embed.weight` |
| `torch.compile` | NOT inside AUModel — trainer calls it (Epic 4) |
| Training dtype | BF16 (never FP16) |
| Weight init | PyTorch default (no custom `_init_weights`) |
