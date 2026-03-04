# System Patterns

## Architecture: LLaMA-style Decoder-Only Transformer

```
[Turkish Token IDs]
        ↓
[Embedding Layer]  (vocab_size=32000, d_model=2048)
        ↓
[24 × TransformerBlock]
  ├─ RMSNorm (pre-norm)
  ├─ Grouped Query Attention (GQA) + RoPE
  └─ SwiGLU FeedForward
        ↓
[RMSNorm]
        ↓
[LM Head]  (weight-tied with embedding)
        ↓
[Logits: vocab_size=32000]
```

## Key Design Decisions & Why

### 1. RMSNorm (not LayerNorm)
- Simpler: no mean subtraction, no bias
- As effective as LayerNorm in practice
- Used in: LLaMA, Mistral, Gemma

### 2. Pre-Norm (not Post-Norm)
- Normalize input BEFORE each sublayer, apply residual after
- More training stable than post-norm
- Prevents gradient explosion in deep networks

### 3. RoPE — Rotary Position Embeddings
- Encodes position by rotating Q and K vectors in complex space
- No extra parameters (vs. learned positional embeddings)
- Naturally extrapolates to longer sequences
- Paper: arxiv.org/abs/2104.09864

### 4. Grouped Query Attention (GQA)
- `num_heads=16` Q heads, but only `num_kv_heads=8` K/V heads
- Each K/V head serves 2 Q heads
- Reduces KV cache memory ~2× with minimal quality loss
- Paper: arxiv.org/abs/2305.13245

### 5. SwiGLU Activation
- `FFN(x) = W2 * (SiLU(W1·x) ⊙ W3·x)`
- Two gate projections, one down projection
- Consistently outperforms GELU/ReLU FFN
- `ffn_hidden_dim = round(2/3 × 4 × d_model / 64) × 64 = 5504`
- Paper: arxiv.org/abs/2002.05202

### 6. Weight Tying
- LM Head shares weights with Token Embedding
- Saves ~65M parameters
- Improves training: embedding space = output space

### 7. No Bias Terms
- Linear layers have `bias=False`
- Cleaner, marginally better, standard in modern LLMs

### 8. Flash Attention 2
- Used via `torch.nn.functional.scaled_dot_product_attention`
- PyTorch 2.0+ auto-dispatches to Flash Attention on H100
- ~2-4× memory efficient vs. standard attention

## Model Hyperparameters (Final / Locked)

```python
vocab_size    = 32000   # Turkish BPE
d_model       = 2048
num_heads     = 16
num_kv_heads  = 8       # GQA
num_layers    = 24
ffn_hidden    = 5504    # SwiGLU
max_seq_len   = 4096    # chatbot context
head_dim      = 128     # d_model / num_heads
# Total: ~1.3B parameters
```

## Training Patterns

### Gradient Accumulation
- `batch_size=8` micro-batch × `grad_accum=16` = effective 128 sequences
- 128 × 4096 = 524,288 tokens per effective batch
- Required because single micro-batch doesn't fill H100

### BF16 Mixed Precision
- NOT FP16 — BF16 has wider dynamic range, no inf/nan explosions
- `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`
- No loss scaling needed (unlike FP16)

### Cosine LR with Warmup
- 0 → `3e-4` over 2000 warmup steps
- `3e-4` → `3e-5` cosine decay over 100k steps

### Checkpoint Strategy (Colab)
- Save every 1000 steps to Google Drive
- Keep last 3 checkpoints only (storage)
- Auto-resume on session restart

## SFT (Instruction Tuning) Patterns

### Chat Template
```
<|system|>
{system message}
<|user|>
{user message}
<|assistant|>
{assistant reply}<|endoftext|>
```

### Loss Masking
- Labels = `-100` for all non-assistant tokens
- CrossEntropy `ignore_index=-100` skips these
- Model ONLY learns to predict assistant responses

## Data Format Patterns

### Pretraining Shards
- Binary `uint16` files: `shard_0000.bin`, `shard_0001.bin`, ...
- Each shard: ~20M tokens (~40MB)
- Documents separated by BOS/EOS tokens

### SFT Format (JSONL)
```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```
