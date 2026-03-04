# Research: AUModel Pretraining Loop

**Date**: 4 Mart 2026  
**Branch**: `001-pretraining-loop`  
**Status**: Complete — all NEEDS CLARIFICATION resolved

---

## R-001: MFU Calculation (Model Flops Utilisation)

**Decision**: Use the combined PaLM/Chinchilla formula that accounts for both parameter FLOPs and attention matrix FLOPs.

**Formula**:

```
flops_per_token = 6 × N + 12 × L × T × H × d_head

where:
  N = total model parameters (749,544,960)
  L = num_layers (24)
  T = seq_len (4,096)
  H = num_heads / query heads (12)
  d_head = d_model / num_heads = 1536 / 12 = 128

→ param_flops  = 6 × 749.5M   =  4,497,000,000
→ attn_flops   = 12 × 24 × 4096 × 12 × 128 = 1,811,939,328
→ flops_per_token ≈ 6,309,000,000 (6.31 B)

MFU = tokens_per_sec × flops_per_token / peak_hardware_flops_per_sec
    = tokens_per_sec × 6.31e9 / 989e12         (H100 SXM5 BF16 peak)
```

**Rationale**: Attention term (~29% of total) is non-trivial at 4K sequence length; ignoring it underestimates compute by ~29%.

**Python implementation**:
```python
def estimate_mfu(
    tokens_per_sec: float,
    model_params: int = 749_544_960,
    seq_len: int = 4096,
    peak_flops: float = 989e12,
    num_layers: int = 24,
    num_heads: int = 12,
    d_model: int = 1536,
) -> float:
    d_head = d_model // num_heads
    param_flops = 6 * model_params
    attn_flops = 12 * num_layers * seq_len * num_heads * d_head
    return tokens_per_sec * (param_flops + attn_flops) / peak_flops
```

**Expected MFU on H100 SXM5 + BF16 + flash attention + torch.compile**:
- Target: ≥ 38%
- Healthy: 38–48%
- Outstanding: 48–55%
- Below 35% → investigate: CPU data bottleneck, missing torch.compile, flash not dispatching, batch too small

**Alternatives considered**: Using `6N` only (ignores attention) — rejected because at seq_len=4096 the error is ~29%.

---

## R-002: Interleaved Multi-Source Shard Loading

**Decision**: Weighted-random interleaving with per-source infinite cycling.

**Rationale**: Industry standard (LLaMA 1/2, Mistral, GPT-NeoX) uses proportional/weighted random sampling — not round-robin — because it respects data quality differences across sources. For a small Turkish LLM (749M params), high-quality sources (Turkish Wikipedia) must be upsampled relative to their raw token count to prevent catastrophic forgetting.

**Recommended sampling weights**:
```
wikipedia : 0.20  (upsampled ~2× its natural proportion)
oscar     : 0.30
mc4       : 0.30
cc100     : 0.20
```
These follow the LLaMA 1 Appendix A pattern for low-resource language upsampling of high-quality corpora.

**Shard exhaustion**: When all shards for a source are consumed, reshuffle and cycle from the start. Never stop — training step count (not dataset size) controls termination. Reshuffling uses `seed = base_seed + cycle_count × 1000` to prevent memorising shard order.

**Architecture**: `InterleavedShardLoader(IterableDataset)` wraps multiple `ShardedDataset` instances. Per-source `SourceState` tracks `shard_idx`, `sample_idx`, and `cycle_count` for resume.

**Resume state**: Checkpoint must save the `InterleavedShardLoader` state dict (per-source shard_idx, sample_idx, cycle_count, rng_state) so that after resume, no tokens are repeated and interleaving position is exact.

**Alternatives considered**:
- Round-robin — rejected: equal representation ignores data quality; Wikipedia would be underrepresented.
- Stop when smallest source exhausts — rejected: causes catastrophic forgetting for the remainder of training.

---

## R-003: Flash Attention Dispatch (No `flash_attn` Package Required)

**Decision**: Use PyTorch 2.0+ SDPA backend flags — no `flash_attn` package import anywhere in trainer source.

**Rationale**: `torch.nn.functional.scaled_dot_product_attention` (already used in `model/attention.py`) automatically dispatches to Flash Attention 2 on H100 when `flash_sdp` is enabled. This is zero-model-code-change: existing SDPA calls benefit automatically. On CPU, `enable_flash_sdp(True)` is a silent no-op, so there is no conditional needed.

**Implementation**:
```python
def configure_attention_backends(device: torch.device) -> None:
    if device.type != "cuda":
        return  # SDPA auto-uses math backend on CPU, nothing to configure
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)  # fallback
```

**Constraints**:
- dtype must be bf16 or fp16 (satisfied — FR-005 mandates bf16)
- `d_head` must be in {16, 32, 64, 128, 256} (satisfied — `d_head=128`)
- `dropout_p=0.0` during eval (natural, no dropout in this model)

Call `configure_attention_backends(device)` immediately after device assignment, before `model.to(device)` and before `torch.compile(model)`.

**Alternatives considered**: Installing `flash_attn` package at H100 time with `pip install flash-attn --no-build-isolation` — retained as optional enhancement in the Colab notebook; but trainer source has no hard dependency on it.

---

## R-004: Optional WandB Integration

**Decision**: Thin `Logger` wrapper class with graceful fallback.

**Rationale**: WandB is unavailable on disconnected sessions, CI environments, and during local unit tests. Hard-importing wandb causes import errors in these environments.

**Implementation pattern**:
```python
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

class Logger:
    def __init__(self, enabled: bool, project: str, run_name: str | None = None): ...
    def log(self, metrics: dict, step: int) -> None: ...
    def finish(self) -> None: ...
```

Falls back to `print()` on: wandb not installed, `WANDB_API_KEY` not set, init failure, or `enabled=False`.

**Alternatives considered**: Using `wandb.init(mode="disabled")` — rejected because it requires wandb to be installed.

---

## R-005: Checkpoint Rotation (Keep Last N)

**Decision**: Atomic write via temp file + `os.rename`, followed by sorted glob of `step_*.pt` files and deletion of oldest when count exceeds 10.

**Pattern**:
```python
def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.rename(path)           # atomic on same filesystem

def rotate_checkpoints(ckpt_dir: Path, keep: int = 10) -> None:
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    for old in checkpoints[:-keep]:
        old.unlink()
```

**Rationale**: Atomic rename prevents corrupt checkpoints on Drive if Colab disconnects mid-write. Sorted glob ensures oldest (lowest step number) are deleted first.

**Alternatives considered**: Keeping all checkpoints — rejected: 100 × ~3 GB = 300 GB exceeds reasonable Drive budget; user specified 10.
