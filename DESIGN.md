# AUModel — Turkish LLM Chatbot: Technical Design

> **Target:** 1.3B parameter Turkish-native chatbot  
> **Hardware:** Google Colab Pro, NVIDIA H100 80GB  
> **Architecture:** LLaMA-style Decoder-Only Transformer  
> **Training Budget:** ~100 hours → ~32B tokens (25× Chinchilla)

---

## Project Structure

```
AUModel/
├── data/
│   ├── raw/                    # Downloaded raw corpora
│   ├── processed/              # Cleaned, tokenized shards
│   └── instruction/            # SFT chat pairs (JSONL)
│
├── tokenizer/
│   ├── train_tokenizer.py      # Train Turkish BPE
│   ├── tokenizer.model         # Output: SentencePiece model
│   └── tokenizer.vocab         # Output: vocabulary file
│
├── model/
│   ├── config.py               # ModelConfig dataclass
│   ├── attention.py            # Multi-head attention + RoPE + GQA
│   ├── feedforward.py          # SwiGLU FFN
│   ├── transformer.py          # Full model (blocks + LM head)
│   └── rope.py                 # Rotary Position Embeddings
│
├── training/
│   ├── dataset.py              # DataLoader for token shards
│   ├── trainer.py              # Training loop
│   ├── lr_scheduler.py         # Cosine scheduler with warmup
│   └── checkpoint.py           # Save / load checkpoints
│
├── sft/
│   ├── sft_dataset.py          # Chat format dataset
│   └── sft_trainer.py          # Instruction tuning loop
│
├── inference/
│   ├── generate.py             # Text generation + sampling
│   └── chat.py                 # Chat interface
│
├── scripts/
│   ├── download_data.sh        # Pull Turkish corpora
│   ├── prepare_data.py         # Clean + tokenize + shard
│   └── run_training.py         # Entry point
│
├── colab/
│   ├── 01_tokenizer.ipynb      # Colab: train tokenizer
│   ├── 02_pretrain.ipynb       # Colab: pretraining
│   └── 03_sft.ipynb            # Colab: instruction tuning
│
└── requirements.txt
```

---

## Phase 1 — Tokenizer

### Goal
Train a Turkish-native BPE tokenizer. Never reuse GPT/LLaMA tokenizers — they byte-fallback on Turkish characters (ç, ğ, ı, ö, ş, ü) and waste tokens.

### File: `tokenizer/train_tokenizer.py`

```python
# Pseudocode — implement this yourself

# 1. Collect plain text Turkish corpus (~5-10GB minimum)
#    Sources: Turkish Wikipedia dump, OSCAR 23.01 Turkish subset
#    Save as one big utf-8 text file: corpus.txt

# 2. Train SentencePiece BPE
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="data/raw/corpus.txt",
    model_prefix="tokenizer/turkish_bpe",
    vocab_size=32000,
    character_coverage=0.9999,   # must cover ç ğ ı ö ş ü
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="[PAD]",
    unk_piece="[UNK]",
    bos_piece="[BOS]",
    eos_piece="[EOS]",
)

# 3. Test it
sp = spm.SentencePieceProcessor(model_file="tokenizer/turkish_bpe.model")
tokens = sp.encode("Merhaba dünya, bu bir test cümlesidir.", out_type=str)
print(tokens)  # should NOT produce many <0x...> byte tokens
```

### Validation Criteria
- Fertility ratio (tokens/word) should be ~1.3–1.8 for Turkish
- Turkish-specific chars must have dedicated token IDs (not byte fallbacks)
- `vocab_size=32000` is the target

---

## Phase 2 — Model Architecture

### File: `model/config.py`

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 32000
    d_model: int = 2048
    num_heads: int = 16
    num_kv_heads: int = 8        # GQA: fewer key/value heads
    num_layers: int = 24
    ffn_hidden_dim: int = 5504   # SwiGLU: int(2/3 * 4 * d_model) rounded to multiple of 64
    max_seq_len: int = 4096      # chatbot-friendly, H100 has headroom
    dropout: float = 0.0         # 0 for large models
    rope_theta: float = 10000.0  # RoPE base frequency

    # Training  (tuned for H100 80GB)
    batch_size: int = 8          # H100 handles larger batch vs A100
    grad_accumulation_steps: int = 16   # effective batch = 128 seqs = 524k tokens
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    dtype: str = "bfloat16"

# Parameter count estimate:
# Embedding:  vocab_size * d_model                    = 32k * 2048 = ~65M
# Per layer:  attention + ffn                         = ~50M
# 24 layers:  24 * 50M                               = ~1.2B
# Total:      ~1.3B params
#
# H100 throughput:   ~155k tok/s effective
# 100 hours budget:  100 × 3600 × 155k ≈ 32B tokens
# Chinchilla ratio:  32B / 1.3B = 25×  ✓ well-trained
```

---

### File: `model/rope.py`

```python
# Rotary Position Embeddings (RoPE)
# Replace sinusoidal absolute positions with rotary encoding
# Applied directly to Q and K in attention

# Implementation steps:
# 1. Precompute cos/sin tables for positions 0..max_seq_len
# 2. For each head dimension pair (x, y), rotate by angle θ_i * position
# 3. Apply BEFORE the attention dot product

import torch

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # dim = head_dim (d_model / num_heads)
    # Returns: complex tensor of shape (seq_len, dim//2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)           # (seq_len, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x shape: (batch, seq_len, num_heads, head_dim)
    # Reshape last dim into complex numbers, rotate, reshape back
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.shape[1]]     # trim to seq_len
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # broadcast
    x_rotated = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).reshape(x.shape).type_as(x)
```

---

### File: `model/attention.py`

```python
# Multi-Head Attention with GQA + RoPE
# Key concept: Grouped Query Attention (GQA)
#   - num_heads Q heads, but only num_kv_heads K and V heads
#   - Each group of (num_heads / num_kv_heads) Q heads shares one K,V head
#   - Reduces KV cache memory significantly

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rotary_emb, precompute_freqs_cis

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.groups = config.num_heads // config.num_kv_heads  # heads per KV group

        # Projections — no bias
        self.wq = nn.Linear(config.d_model, config.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.num_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.num_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Expand K,V to match Q heads (GQA)
        # k: (B, T, num_kv_heads, head_dim) → (B, T, num_heads, head_dim)
        k = k.repeat_interleave(self.groups, dim=2)
        v = v.repeat_interleave(self.groups, dim=2)

        # Transpose for matmul: (B, num_heads, T, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Use Flash Attention if available
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
        # is_causal=True applies causal mask automatically — no manual mask needed

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
```

---

### File: `model/feedforward.py`

```python
# SwiGLU Feed-Forward Network
# Formula: FFN(x) = (Swish(W1·x) ⊙ W3·x) · W2
# Two gate projections (W1, W3) and one down projection (W2)
# ffn_hidden_dim = int(2/3 * 4 * d_model), rounded to multiple of 64

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # No bias on any projection
        self.w1 = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=False)  # down
        self.w3 = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)  # up

    def forward(self, x):
        # SwiGLU: element-wise product of SiLU gate and linear projection
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

---

### File: `model/transformer.py`

```python
# Full Model: Embedding → N×Block → RMSNorm → LM Head
# Pre-norm architecture (norm before attention/ffn, not after)

import torch
import torch.nn as nn
from .config import ModelConfig
from .attention import Attention
from .feedforward import FeedForward
from .rope import precompute_freqs_cis

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Normalize by root mean square
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / norm * self.weight).type_as(x)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)   # pre-attention norm
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.d_model)   # pre-FFN norm
        self.ffn = FeedForward(config)

    def forward(self, x, freqs_cis, mask=None):
        # Pre-norm + residual connection (critical — do NOT put norm after)
        x = x + self.attn(self.norm1(x), freqs_cis, mask)
        x = x + self.ffn(self.norm2(x))
        return x

class AUModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.d_model)    # final norm
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding and LM head weights (saves ~65M params)
        self.lm_head.weight = self.embed.weight

        # Precompute RoPE frequencies once
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.d_model // config.num_heads, config.max_seq_len)
        )

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        x = self.embed(tokens)                 # (B, T, d_model)
        freqs_cis = self.freqs_cis[:T]

        for block in self.blocks:
            x = block(x, freqs_cis)

        x = self.norm(x)
        logits = self.lm_head(x)               # (B, T, vocab_size)

        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        return logits, None

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
```

---

## Phase 3 — Data Pipeline

### File: `scripts/prepare_data.py`

```python
# Steps to implement:

# STEP 1: Download Turkish data
# Sources:
#   - OSCAR 23.01 Turkish: huggingface.co/datasets/oscar-corpus/OSCAR-2301
#   - Turkish Wikipedia: dumps.wikimedia.org/trwiki/latest/
#   - mC4 Turkish: huggingface.co/datasets/mc4 (language="tr")
#   - CC-100 Turkish: data.statmt.org/cc-100/

# STEP 2: Clean each source
# For each document:
#   - Remove HTML tags
#   - Deduplicate (MinHash or exact hash)
#   - Filter by language score (fasttext langid)
#   - Filter short docs (< 100 tokens)
#   - Normalize unicode (NFC)
#   - Remove documents with too many non-Turkish chars

# STEP 3: Tokenize + shard into binary files
# Target: ~100 files of ~1GB each (easy to shuffle and stream)
# Format: raw uint16 token IDs, saved with numpy memmap
#
#   tokens = tokenizer.encode(text)
#   tokens_array = np.array(tokens, dtype=np.uint16)
#   # Prepend BOS, append EOS per document
#   # Concatenate all tokens into large flat array
#   # Save as: data/processed/shard_0000.bin, shard_0001.bin, ...

# STEP 4: Verify
# Total tokens target: 30B tokens minimum
# Check distribution: print token frequency histogram
```

### File: `training/dataset.py`

```python
# DataLoader that reads binary shards
# Key: during training, slice windows of max_seq_len from flat token array

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ShardedDataset(Dataset):
    def __init__(self, shard_paths: list[str], seq_len: int):
        # Load all shards into memory (or use memmap for large datasets)
        # Each item: input tokens [0..T-1], target tokens [1..T] (shifted by 1)
        ...

    def __getitem__(self, idx):
        # Return (input_ids, target_ids) both of shape (seq_len,)
        # target is input shifted left by 1 position
        ...
```

---

## Phase 4 — Training Loop

### File: `training/trainer.py`

```python
# Training loop — key implementation details:

# 1. Mixed precision (BF16, not FP16 — more stable for LLMs)
#    scaler = torch.cuda.amp.GradScaler()  # not needed for BF16
#    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#        logits, loss = model(x, y)

# 2. Gradient accumulation (simulate large batch)
#    for micro_step in range(grad_accumulation_steps):
#        loss = loss / grad_accumulation_steps
#        loss.backward()
#    optimizer.step()
#    optimizer.zero_grad()

# 3. Gradient clipping (prevent exploding gradients)
#    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. Cosine LR schedule with warmup
#    See: training/lr_scheduler.py

# 5. Logging (every N steps)
#    Log: step, loss, lr, tokens/sec, MFU (model flops utilization)

# 6. Checkpointing (every 1000 steps)
#    torch.save({
#        'step': step,
#        'model': model.state_dict(),
#        'optimizer': optimizer.state_dict(),
#        'config': config,
#    }, f'checkpoints/step_{step:06d}.pt')
```

### File: `training/lr_scheduler.py`

```python
import math

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    # After max_steps: hold at min_lr
    if step > max_steps:
        return min_lr
    # Cosine decay between warmup and max_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# Typical values:
# max_lr = 3e-4
# min_lr = max_lr * 0.1 = 3e-5
# warmup_steps = 2000
# max_steps = 100000
```

---

## Phase 5 — Instruction Tuning (SFT)

### Chat Format

Every example must follow a consistent template. Use this:

```
<|system|>
Sen yardımcı bir Türkçe yapay zeka asistanısın.
<|user|>
{kullanıcı mesajı}
<|assistant|>
{model yanıtı}
```

Add special tokens to vocabulary: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|endoftext|>`

### File: `data/instruction/` — Data Format

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "Türkiye'nin başkenti neresidir?"}, {"role": "assistant", "content": "Türkiye'nin başkenti Ankara'dır."}]}
{"messages": [{"role": "user", "content": "Merhaba, nasılsın?"}, {"role": "assistant", "content": "Teşekkür ederim, iyiyim! Size nasıl yardımcı olabilirim?"}]}
```

### Data Sources for SFT
- Translate Alpaca/Dolly datasets to Turkish (use Google Translate API or DeepL)
- Turkish Stack Exchange / Quora (scrape Q&A pairs)
- Turkish Reddit-like forums (Ekşi Sözlük etc.)
- Manual curation: ~1k high-quality examples

### File: `sft/sft_trainer.py`

```python
# SFT differs from pretraining in one key way:
# Only compute loss on ASSISTANT tokens, not on user/system tokens
# This prevents the model from "memorizing" user inputs

# Implementation:
# labels = token_ids.clone()
# labels[mask_of_non_assistant_tokens] = -100  # ignore_index in cross_entropy
# loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
```

---

## Phase 6 — Inference & Chat

### File: `inference/generate.py`

```python
# Key generation parameters to implement:

# Temperature: float (0.0 = greedy, 1.0 = sampling, >1.0 = more random)
# Top-p (nucleus): only sample from top tokens summing to probability p
# Top-k: only sample from top k tokens
# Repetition penalty: penalize tokens that already appeared

# Generation loop:
# while len(tokens) < max_new_tokens:
#     logits = model(tokens[-max_seq_len:])  # always truncate to context window
#     logits = logits[:, -1, :]             # last position only
#     logits = apply_temperature(logits, temperature)
#     logits = apply_top_p(logits, top_p)
#     next_token = sample(logits)
#     if next_token == EOS_ID: break
#     tokens.append(next_token)
```

---

## Colab Training Setup

### `colab/02_pretrain.ipynb` — Key Setup

```python
# Always run at the start of each Colab session:

# 1. Mount Drive (for checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# 2. Install Flash Attention
# !pip install flash-attn --no-build-isolation

# 3. Enable TF32 (faster matmul on H100)
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 4. Enable gradient checkpointing (optional on H100 — you have headroom)
# Still recommended to be safe:
model.gradient_checkpointing_enable()

# 5. Compile model (PyTorch 2.0+, ~30% speedup)
model = torch.compile(model)

# 6. Save checkpoint every 1000 steps to Drive
# Never rely on Colab's local storage — it resets
```

### Memory Budget (H100 80GB)

```
Model weights (BF16):    1.3B × 2 bytes   =  2.6 GB
Optimizer (AdamW):       1.3B × 8 bytes   = 10.4 GB
Activations (batch=8):   ~20 GB (w/ grad checkpointing)
Gradients:               ~2.6 GB
CUDA overhead:           ~2 GB
──────────────────────────────────────────────────────
Total:                   ~38 GB  ✓ (plenty of headroom on 80GB)

Without grad checkpointing: ~55GB → still fits on H100
```

### H100 vs A100 Comparison

```
               A100 80GB     H100 80GB
Throughput:    ~55k tok/s    ~155k tok/s    (2.8× faster)
Tokens/100hrs: ~20B          ~32B
Chinchilla:    15× (weak)    25× (good)     for 1.3B model
Memory:        80GB          80GB
BF16 TFLOPS:   312           756
```

---

## Implementation Order

```
Week 1:  [ ] Tokenizer training (train_tokenizer.py)
         [ ] Data download (Turkish Wikipedia + OSCAR TR)
         [ ] Data preparation script (prepare_data.py)

Week 2:  [ ] ModelConfig (config.py)
         [ ] RoPE (rope.py)
         [ ] Attention (attention.py)
         [ ] FeedForward (feedforward.py)
         [ ] Transformer (transformer.py)
         [ ] Sanity check: forward pass, count params

Week 3:  [ ] Dataset loader (dataset.py)
         [ ] LR scheduler (lr_scheduler.py)
         [ ] Trainer (trainer.py)
         [ ] Run on small data first (overfit 1 batch as sanity check)

Week 4+: [ ] Full pretraining on Colab (long-running)
         [ ] SFT data collection
         [ ] SFT trainer

Final:   [ ] Inference / chat interface
         [ ] Evaluation on Turkish benchmarks
```

---

## Key Hyperparameters Summary

```python
# Model
vocab_size       = 32000
d_model          = 2048
num_heads        = 16
num_kv_heads     = 8       # GQA
num_layers       = 24
ffn_hidden_dim   = 5504    # SwiGLU: round(2/3 * 4 * 2048 / 64) * 64
max_seq_len      = 4096    # chatbot context
total_params     ≈ 1.3B

# Training  (H100 80GB)
batch_size       = 8       # per GPU
grad_accum       = 16      # effective batch = 128 sequences = 524k tokens
learning_rate    = 3e-4
lr_min           = 3e-5
warmup_steps     = 2000
max_steps        = 100000
weight_decay     = 0.1
grad_clip        = 1.0
dtype            = bfloat16
optimizer        = AdamW(β1=0.9, β2=0.95)

# Tokens
target_tokens    = 32B     # ~100 H100-hours  (25× Chinchilla)
```

---

## Resources

| Topic | Resource |
|-------|----------|
| LLaMA architecture paper | arxiv.org/abs/2302.13971 |
| RoPE paper | arxiv.org/abs/2104.09864 |
| GQA paper | arxiv.org/abs/2305.13245 |
| SwiGLU paper | arxiv.org/abs/2002.05202 |
| Flash Attention | arxiv.org/abs/2205.14135 |
| nanoGPT (reference impl) | github.com/karpathy/nanoGPT |
| litgpt (clean LLaMA impl) | github.com/Lightning-AI/litgpt |
| OSCAR Turkish data | huggingface.co/datasets/oscar-corpus/OSCAR-2301 |
| Turkish Wikipedia dump | dumps.wikimedia.org/trwiki/latest/ |
