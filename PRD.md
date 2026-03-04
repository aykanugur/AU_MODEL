# Product Requirements Document

> **Project:** AUModel — Turkish Native LLM Chatbot  
> **Author:** Aykan Ugur  
> **Status:** Draft  
> **Version:** 1.3  
> **Date:** 4 Mart 2026  
> **Reviewers:** —

---

## 1. Overview

### 1.1 Problem Statement

Turkish speakers have no access to a high-quality open-source LLM that natively understands Turkish. Every major model availaGooble today (GPT-4, LLaMA 3, Mistral, Gemma) tokenizes Turkish using byte-level fallbacks because their vocabularies were built primarily on English corpora. A single Turkish word like *"yapamayacağım"* (I cannot do it) costs 5–8 tokens in LLaMA's tokenizer versus 1–2 tokens in a Turkish-native tokenizer. This 3–5× token inefficiency means:

- The model's effective context window shrinks 3–5× for Turkish input
- Turkish generation quality is lower because the model has seen far less Turkish *meaning* per token
- No open model exists that a Turkish developer can fine-tune, study, or deploy without these fundamental limitations

### 1.2 Product Vision

AUModel is a 700 million parameter Turkish-native language model trained from scratch — with a custom Turkish BPE tokenizer, 17.5B tokens of Turkish pretraining data, and instruction fine-tuning for conversational use — deployable first locally and then as a public chatbot on a personal website.

### 1.3 Success Metrics

| Metric | Baseline (GPT-2 Turkish fine-tune) | Target | Measurement Method |
|--------|-------------------------------------|--------|--------------------|
| Turkish token fertility | 3.0–5.0 tokens/word | 1.1–1.4 tokens/word (64k vocab vs. 32k vocab improvement) | Run tokenizer on 10,000 Turkish sentences, compute tokens/word |
| Training loss (pretrain) | N/A | ≤ 2.9 after 17.5B tokens | Logged during training every 100 steps |
| Validation perplexity | N/A | ≤ 25 on Turkish Wikipedia held-out set | Computed at each 1,000-step checkpoint |
| SFT response quality | N/A | ≥ 4.0/5.0 average on 100 manual Turkish Q&A evals | Human evaluation by author |
| Grammatical correctness | N/A | ≥ 90% of responses contain no Turkish grammar errors | Manual review of 100 generated outputs |
| Tokenizer coverage | N/A | 100% of Turkish chars (ç,ğ,ı,İ,ö,ş,ü,Ü,Ö,Ç,Ğ,Ş) have dedicated token IDs in the **64k** vocabulary | Automated check post-training |
| Web search grounding | N/A | ≥ 80% of search-triggered responses cite no factually wrong claim | Human review of 50 search-grounded outputs |
| DPO preference agreement | N/A | DPO-tuned model wins ≥ 60% of head-to-head comparisons vs. SFT-only model | Eval on 200 held-out preference pairs after DPO fine-tune |

---

## 2. Background & Context

### 2.1 Current State

There is no open-source Turkish-native LLM trained from scratch as of Mart 2026. Existing approaches fall into one of three categories:

1. **English-centric models with Turkish fine-tuning:** LLaMA 3, Mistral 7B fine-tuned on Turkish data. Core limitation: tokenizer is not Turkish-aware; the model never develops native Turkish morphological understanding.
2. **Multilingual models:** mBERT, mT5, BLOOM. These are encoder-only or seq2seq — not optimized for generation or chat.
3. **Closed models:** GPT-4o Turbo, Claude 3.5 — Turkish support is good but proprietary, not deployable, not open.

AUModel fills the gap: a decoder-only generative model with a Turkish-native tokenizer, trained from scratch on Turkish corpora.

### 2.2 Prior Art

| Work | Description | Relevance |
|------|-------------|-----------|
| LLaMA 2/3 (Meta, 2023/2024) | 7B–70B decoder-only, RoPE+GQA+SwiGLU | Architecture reference — we replicate at 700M |
| Phi-3-mini (Microsoft, 2024) | 3.8B, beats 7B models via quality data | Confirms small high-quality models compete with larger ones |
| SmolLM2 (HuggingFace, 2024) | 360M–1.7B production chatbot | Confirms 700M is viable for chatbot use; SmolLM2-1.7B is the closest reference point |
| Chinchilla (DeepMind, 2022) | Compute-optimal scaling: 20× tokens per param | Informs our 17.5B token target for 700M params (25×) |
| RoPE (Su et al., 2021) | Rotary position embeddings | Used in attention module |
| GQA (Ainslie et al., 2023) | Grouped query attention | Used to reduce KV cache memory 2× |
| SwiGLU (Noam Shazeer, 2020) | Gated FFN activation | Used in feed-forward blocks |
| OSCAR 23.01 | Large multilingual web corpus | Primary Turkish pretraining data source |

### 2.3 Assumptions & Constraints

**Assumptions:**
- Google Colab Pro provides consistent H100 80GB access for single-session durations of 8–12 hours
- OSCAR 23.01 Turkish subset (~50GB raw) yields ≥ 20B cleaned tokens after deduplication and filtering
- mC4 Turkish and Turkish Wikipedia contribute ≥ 10B additional tokens
- Turkish Wikipedia (~1GB raw text) + OSCAR Turkish sample (5GB) provides sufficient diversity to train a 64k SentencePiece BPE tokenizer; 64k vocab requires ~128 MB extra embedding memory (negligible on H100)
- A manually curated + machine-translated instruction dataset of ≥ 50,000 Turkish pairs is achievable

**Constraints:**
- Single developer (no team)
- Compute: 1× H100 80GB on Google Colab Pro — no multi-GPU training
- Training budget: ≤ 100 compute hours total for pretraining
- No budget for cloud GPU rental beyond Colab Pro subscription
- Model must fit in 80GB GPU memory during training (weights + optimizer + activations)
- No proprietary data — all training data must be freely available or public domain
- Google Drive storage limit (~15GB free, or purchased) — checkpoints must be managed carefully

---

## 3. Goals & Non-Goals

### 3.1 Goals

- **G1:** Train a Turkish BPE tokenizer with vocabulary size **64,000** that achieves 1.1–1.4 tokens-per-word fertility on Turkish text, with dedicated token IDs for all Turkish-specific characters — 64k vocab reduces fertility by ~25% vs. 32k at the cost of only 128 MB extra embedding memory
- **G2:** Implement a 700M parameter LLaMA 3-style transformer model (RoPE with `theta=500,000`, GQA, SwiGLU, RMSNorm, pre-norm, weight-tied LM head) from scratch in PyTorch — exact config: `d_model=1536`, `num_heads=12`, `num_kv_heads=6`, `num_layers=24`, `ffn_hidden_dim=4352`, `max_seq_len=4096`, `vocab_size=64000`, `rope_theta=500000`
- **G3:** Build a data pipeline that downloads, cleans, deduplicates, tokenizes, and shards Turkish corpora into uint16 binary files totalling ≥ 17.5B tokens; shards are **ordered for curriculum training**: Phase 1 shards (steps 0–16,500) contain only Turkish Wikipedia (~300M tokens cycled), Phase 2 shards (steps 16,500–33,000) contain the full mixed corpus (OSCAR + mC4 + CC-100)
- **G4:** Pretrain the model for ≤ 18 H100 hours to process ≥ 17.5B tokens (25× Chinchilla), achieving validation perplexity ≤ 22 on held-out Turkish Wikipedia — full 100-hour budget provides headroom for failed runs, extended training, and SFT
- **G5:** Fine-tune the pretrained model using supervised instruction tuning on ≥ 50,000 Turkish conversation pairs (machine-translated + manually curated), with loss masking limited to assistant tokens
- **G6:** Deploy a working terminal-based chat interface locally, then publish the chatbot on a personal website
- **G7:** Open-source the training code, model weights (if performance meets quality bar), and tokenizer
- **G8:** Implement an MCP (Model Context Protocol) web search tool that the model triggers autonomously at inference time — when the model determines it needs live information, it emits a structured `<|search|>query<|/search|>` token sequence, the MCP server calls a search API (Brave Search or SerpAPI), injects the top-3 result snippets as `<|search_results|>...</|search_results|>` context, and the model generates its final answer grounded in those results
- **G9:** Implement a preference data collection loop and DPO fine-tuning phase — every 2nd assistant turn during live chat, generate 2 candidate responses (Response A: temperature=0.7, Response B: temperature=1.1 + nucleus top-p=0.85) and display both to the user; the user clicks "A" or "B" to indicate the preferred response; each selection is logged as a `(prompt, chosen, rejected)` triplet in `data/preference/preference_log.jsonl`; once ≥ 10,000 preference pairs are collected, run a DPO fine-tuning pass (β=0.1, LR=1e-6, 1 epoch) on top of the SFT checkpoint

### 3.2 Non-Goals

- **Not G:** Multi-language support — model will be Turkish-only; English or other language outputs are not a requirement
- **Not G:** Retrieval-Augmented Generation (RAG) with a local vector store or document index — web search via MCP covers live information retrieval without a persistent vector database
- **Not G:** RLHF with a separate reward model — preference signals will be collected during live chat and used for DPO fine-tuning only (DPO requires no reward model)
- **Not G:** Real-time streaming output in v1 (token-by-token streaming in web UI)
- **Not G:** ≥ 1B parameter model — 700M is the target; scaling up requires re-profiling compute budget against H100 throughput
- **Not G:** Mobile or native app — web deployment only
- **Not G:** Multi-turn memory beyond the 4096-token context window — no external memory store

---

## 4. Users & Stakeholders

### 4.1 Target Users

| Persona | Description | Primary Need |
|---------|-------------|--------------|
| Turkish speaker (general) | Native Turkish speaker looking for a chatbot that gives accurate, fluent Turkish answers | Ask general questions and get correct, fluent Turkish responses |
| Turkish developer | Developer wanting a base model to fine-tune for Turkish NLP tasks (classification, summarization, etc.) | Download weights, tokenizer, and reproduce training |
| NLP researcher | Researcher studying low-resource language model training | Access to training code, design decisions, and benchmark results |

### 4.2 Stakeholders

| Stakeholder | Role | Involvement |
|-------------|------|-------------|
| Aykan Ugur | Author, sole developer | All decisions, all implementation |
| Website visitors | End users of the chatbot | Testing and using the deployed chatbot |

---

## 5. Model & Data Requirements

### 5.1 Model Objectives

**Task:** Autoregressive next-token prediction (causal language modeling) during pretraining; instruction-following chat during SFT.

The model must:
1. Generate grammatically correct, semantically coherent Turkish text
2. Follow Turkish-language instructions given in the `<|user|>` chat role
3. Respond in the `<|assistant|>` role with accurate, relevant Turkish answers
4. Handle conversations up to 4096 tokens total (prompt + response)

### 5.2 Input / Output Specification

| Field | Type | Max Length | Description | Example |
|-------|------|------------|-------------|---------|
| System prompt | String (Turkish) | 512 tokens | Defines assistant behavior | `"Sen yardımcı bir Türkçe yapay zeka asistanısın."` |
| User message | String (Turkish) | Up to context window | User's question or instruction | `"Türkiye'nin başkenti neresidir?"` |
| Assistant response | String (Turkish) | Up to 2048 new tokens | Model-generated Turkish reply | `"Türkiye'nin başkenti Ankara'dır."` |
| Total context | Token sequence | 4096 tokens max | Full conversation history | — |

### 5.3 Data Requirements

**Pretraining Data Sources:**

| Source | Raw Size | Est. Cleaned Tokens | Access Method |
|--------|----------|---------------------|---------------|
| Turkish Wikipedia (20231101) | ~1 GB | ~300M tokens | HuggingFace `datasets` |
| OSCAR 23.01 Turkish | ~50 GB | ~18B tokens | HuggingFace `datasets` (streaming) |
| mC4 Turkish | ~100 GB | ~12B tokens | HuggingFace `datasets` (streaming) |
| CC-100 Turkish | ~25 GB | ~6B tokens | statmt.org direct download |
| **Total target** | — | **≥ 17.5B tokens** | — |

**Cleaning Requirements (per document, before tokenization):**
- NFC unicode normalization (mandatory for Turkish ç,ğ,ı,İ,ö,ş,ü)
- Remove all HTML/XML tags
- Remove documents shorter than 200 characters
- Remove documents where < 50% of characters are Latin-script + Turkish diacritics (filters non-Turkish content)
- Deduplicate at document level using exact SHA-256 hash

**SFT Data Sources:**

| Source | Volume | Type | Method |
|--------|--------|------|--------|
| Alpaca 52k → Turkish | 52,000 pairs | General instruction following | Machine-translate via DeepL or Google Translate |
| Dolly 15k → Turkish | 15,000 pairs | Open-domain Q&A | Machine-translate |
| Manually curated Turkish Q&A | ≥ 5,000 pairs | Turkish-specific factual Q&A | Manual authorship |
| **Total** | **≥ 72,000 pairs** | — | — |

**Quality Requirements for SFT:**
- No machine-translation artifacts (reviews of 500 random samples to spot-check)
- No factually incorrect answers in manually curated set
- All assistant turns must be ≥ 20 words (filters degenerate short answers)

**Privacy / PII Constraints:**
- No personal names, addresses, phone numbers, or Turkish ID numbers in training data
- OSCAR and mC4 are already filtered for personal data by their publishers

### 5.4 Model Performance Requirements

| Metric | Minimum Acceptable | Target | Notes |
|--------|--------------------|--------|-------|
| Pretraining val perplexity | ≤ 28 | ≤ 22 | Measured on 50M token held-out Turkish Wikipedia |
| Training loss at 17.5B tokens | ≤ 3.2 | ≤ 2.9 | Cross-entropy on next-token prediction |
| Initial training loss | ≈ log(64000) = 11.07 | 11.07 ± 0.5 | Sanity check: random init must produce this |
| SFT human eval score | ≥ 3.5 / 5.0 | ≥ 4.0 / 5.0 | Author rates 100 outputs on fluency + accuracy |
| Turkish grammar error rate | ≤ 20% of outputs | ≤ 10% of outputs | Manual review of 100 generated responses |
| Token fertility | ≤ 2.0 tok/word | 1.3–1.8 tok/word | Measured on 10,000 Turkish sentences |

### 5.5 Evaluation Strategy

**Pretraining:**
- Validation loss logged every 1,000 training steps on a fixed 50M token held-out split from Turkish Wikipedia
- Loss curve must be monotonically decreasing in the first 10,000 steps — any spike > 0.5 above rolling average triggers investigation
- Overfit sanity check before full training: loss on a single batch of 128 sequences must reach < 0.1 within 100 gradient steps

**SFT / Chatbot Quality:**
- **Human eval set:** 100 manually authored Turkish questions across 5 domains: factual (geography, history, science), conversational greetings, mathematical reasoning, Turkish grammar/language, coding help in Turkish
- **Scoring rubric:** 1 = wrong/incoherent, 2 = partially correct with major errors, 3 = correct but unnatural Turkish, 4 = correct and natural Turkish, 5 = excellent
- **Automatic grammar check:** Use `zemberek-python` or equivalent Turkish NLP library to flag morphological errors in 100 sampled outputs
- **Comparison baseline:** Same 100 questions answered by `google/mt5-base` fine-tuned on Turkish — AUModel must score ≥ 0.5 points higher on average

---

## 6. Functional Requirements

### 6.1 Core Features

| ID | Feature | Description | Priority |
|----|---------|-------------|----------|
| F-01 | Turkish BPE Tokenizer | Train 64k-vocab SentencePiece BPE tokenizer trained on Turkish Wikipedia + 5GB OSCAR sample (`character_coverage=0.9999`, `vocab_size=64000`), with dedicated token IDs for all Turkish chars (ç,ğ,ı,İ,ö,ş,ü) | P0 |
| F-02 | 700M Transformer Model | LLaMA 3-style architecture: 24 layers, d_model=1536, 12 heads, GQA (6 KV heads), SwiGLU FFN (ffn_hidden=4352), RMSNorm (eps=1e-5), RoPE (`theta=500,000`), vocab_size=64000, weight-tied LM head | P0 |
| F-03 | Data Pipeline | Download OSCAR/Wikipedia/mC4 Turkish, clean, deduplicate, tokenize, save as uint16 binary shards of 20M tokens each | P0 |
| F-04 | Pretraining Loop | BF16 autocast, batch_size=16, grad_accum=8 (effective batch 128 seqs × 4096 tokens = 524k tokens/step), cosine LR schedule (lr=3e-4 → 3e-5, warmup=2000 steps, max_steps=33000), gradient clipping at 1.0, auto-resume from checkpoint; **curriculum order**: steps 0–16,500 on Wikipedia-only shards, steps 16,500–33,000 on full mixed corpus | P0 |
| F-05 | Checkpoint Management | Save model + optimizer state to Google Drive every 1,000 steps, keep last 3 checkpoints, auto-resume on Colab session restart | P0 |
| F-06 | SFT Dataset | Convert Turkish instruction pairs to chat format with `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>` tokens; mask loss on non-assistant tokens (-100) | P0 |
| F-07 | SFT Training Loop | Fine-tune pretrained model for 3 epochs on ≥ 72,000 Turkish instruction pairs, LR = 5e-5 | P0 |
| F-08 | Text Generation | Autoregressive sampling with configurable temperature, top-k, top-p, and repetition penalty | P1 |
| F-09 | Terminal Chat Interface | Multi-turn conversation loop, history management within 4096-token window, graceful truncation of old turns | P1 |
| F-10 | Web Chat Interface | Embed chatbot on personal website; input field, response display, conversation history visible to user | P1 |

### 6.2 User Stories

**Story 1 — Turkish Speaker**
```
As a Turkish speaker,
I want to ask questions in Turkish and receive accurate, grammatically correct answers,
So that I can use a chatbot that actually understands Turkish naturally.

Acceptance Criteria:
- [ ] Response is in Turkish (no unexpected English words)
- [ ] Response answers the question asked (not a hallucinated non-sequitur)
- [ ] Response contains no obvious Turkish grammar errors (correct verb conjugation, proper suffix agglutination)
- [ ] Response is generated within the context window (no truncation mid-sentence)
```

**Story 2 — Turkish Developer**
```
As a Turkish NLP developer,
I want to download the model weights and tokenizer,
So that I can fine-tune AUModel on my domain-specific Turkish dataset.

Acceptance Criteria:
- [ ] Model weights published as a single .pt checkpoint file
- [ ] Tokenizer .model file published alongside weights
- [ ] ModelConfig dataclass documented so architecture can be recreated
- [ ] Loading and running inference requires only PyTorch + SentencePiece (no proprietary dependencies)
```

**Story 3 — Website Visitor**
```
As a visitor to the author's website,
I want to open a chat window and start a conversation in Turkish,
So that I can experience the capabilities of AUModel.

Acceptance Criteria:
- [ ] Chat input field is visible without logging in
- [ ] First response arrives within 30 seconds (no timeout error)
- [ ] Conversation history is displayed in the chat window for the current session
- [ ] Long responses do not get cut off mid-word
```

---

## 7. Non-Functional Requirements

| Category | Requirement | Target | Notes |
|----------|-------------|--------|-------|
| Accuracy | Factual correctness on Turkish Q&A | ≥ 4.0 / 5.0 human eval | More important than response speed |
| Latency (local) | Response time for 200-token output | ≤ 10 seconds | H100 local inference |
| Latency (web) | Response time for 200-token output on website | ≤ 30 seconds | Acceptable for demo/personal site |
| Memory (inference) | GPU memory for inference | ≤ 2 GB (BF16 weights only) | 700M × 2 bytes = 1.4 GB; fits any 4 GB+ consumer GPU |
| Memory (training) | GPU memory during pretraining | ≤ 30 GB | 700M BF16 + optimizer + activations (batch=16) ≈ 27 GB — well within H100 80GB |
| Reproducibility | Training is deterministic | Fixed random seed must reproduce same loss curve ± 0.01 | Required for debugging |
| Portability | Model runs on CUDA GPU ≥ 4 GB | Any NVIDIA GPU with 4 GB VRAM can run inference | 700M BF16 = 1.4 GB weights; with KV cache budget to spare |
| Data privacy | No PII in training data | 0 known PII records | Verified by upstream dataset publishers (OSCAR, mC4) |
| Code quality | All modules have docstrings and inline comments | Every function has a docstring | Required for open-source release |
| Checkpoint safety | No training run can lose > 1,000 steps of work | Auto-save every 1,000 steps to Drive | Colab session limit mitigation |

---

## 8. System Design

### 8.1 Architecture Overview

**Training Pipeline:**
```
[Raw Text Files (OSCAR, Wikipedia, mC4)]
        ↓  scripts/prepare_data.py
[Cleaned + Tokenized uint16 Binary Shards]  data/processed/shard_XXXX.bin
        ↓  training/dataset.py (streaming DataLoader)
[Batches: (B=8, T=4096) token tensors]
        ↓  training/trainer.py
[AUModel 700M forward pass — BF16 autocast]
        ↓  CrossEntropy loss → backward → AdamW step
[Checkpoint every 1000 steps → Google Drive]
```

**Inference Pipeline:**
```
[User text input (Turkish)]
        ↓  tokenizer/turkish_bpe.model
[Token IDs: [BOS] + sp.encode(prompt)]
        ↓  AUModel.forward() — inference mode
[Logits → temperature → top-p → sample]
        ↓  sp.decode(new_token_ids)
[Turkish text response]
```

**Chat Format (SFT + Inference):**
```
<|system|>
Sen yardımcı bir Türkçe yapay zeka asistanısın.
<|user|>
{user message}
<|assistant|>
{model generates from here}
```

### 8.2 Key Components

| Component | Technology | Responsibility |
|-----------|-----------|----------------|
| Tokenizer | SentencePiece BPE | Encode/decode Turkish text ↔ token IDs |
| Token Embedding | `nn.Embedding(64000, 1536)` | Map token IDs to dense vectors; +128 MB memory vs. 32k vocab |
| RMSNorm | Custom `nn.Module` | Pre-normalize each transformer sublayer |
| Attention | Custom GQA + RoPE | Causal self-attention, 12 Q heads / 6 KV heads |
| FeedForward | Custom SwiGLU | 3-projection gated FFN (W1, W2, W3) |
| LM Head | `nn.Linear(1536, 64000, bias=False)` | Project hidden state to vocab logits (weight-tied with Token Embedding) |
| Training loop | Pure PyTorch | BF16 autocast, gradient accumulation, cosine LR |
| Data loader | `torch.utils.data.IterableDataset` | Stream uint16 shards, yield (input, target) pairs |
| Checkpointing | `torch.save` / `torch.load` | Persist model+optimizer to Google Drive |
| Inference | `torch.inference_mode()` | Autoregressive generation with sampling |
| Web deployment | TBD (FastAPI + simple HTML or Next.js) | Serve chatbot on personal website |

### 8.3 Integrations

| Integration | Purpose | Required By |
|------------|---------|-------------|
| Google Drive (Colab mount) | Persistent checkpoint storage across Colab sessions | Training |
| HuggingFace `datasets` library | Stream OSCAR 23.01, mC4, Wikipedia training data | Data pipeline |
| `sentencepiece` library | Tokenizer training and inference | Tokenizer + all modules |
| `torch.nn.functional.scaled_dot_product_attention` | Flash Attention 2 dispatch on H100 | Attention module |
| Personal website hosting (TBD) | Public chatbot deployment | Web interface |

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Colab H100 unavailable / pre-empted mid-session | High | High | Save checkpoint every 1,000 steps; auto-resume logic in trainer |
| Training loss diverges (NaN/Inf) | Medium | High | Use BF16 (not FP16), gradient clipping at 1.0, monitor loss every 100 steps |
| Turkish corpus < 30B tokens after cleaning | Medium | High | Increase mC4 sampling; add CC-100 Turkish as fallback source |
| Tokenizer fertility > 1.8 tokens/word on 64k vocab | Very Low | Medium | 64k vocab is already ~2× standard; if fertility fails, audit training corpus for non-Turkish content and retrain tokenizer |
| Model underfits at 17.5B tokens (loss > 3.2) | Low | Medium | Extend training within 100-hour budget; 700M at H100 trains fast — 18 hrs for 17.5B, leaving 82 hrs of headroom |
| Google Drive storage full from checkpoints | Medium | Low | Keep only last 3 checkpoints; each checkpoint ≈ 11 GB |
| SFT instruction data quality too low | Medium | High | Manually review 500 random translated pairs before training SFT |
| Web inference too slow for visitors (> 60s) | High | Medium | Quantize model to INT8 for web deployment (4-bit if needed) |
| Web hosting cost too high for inference | Medium | Medium | Start with CPU inference on cheap VPS; use batch timeout |

---

## 10. Milestones & Timeline

No hard deadline — quality-first. Estimated durations given compute and single-developer constraint.

| Milestone | Deliverable | Success Criteria | Est. Duration |
|-----------|-------------|-----------------|---------------|
| M1 | Tokenizer trained and validated | Fertility ≤ 1.8, all Turkish chars native | 3–5 days |
| M2 | Model architecture implemented | Forward pass runs, param count = 700M ± 5% (verify: `sum(p.numel() for p in model.parameters())` = 700,317,696), initial loss = 10.37 ± 0.5 | 5–7 days |
| M3 | Data pipeline complete | ≥ 32B tokens in uint16 shards, valset created | 5–10 days |
| M4 | Pretraining complete | ≤ 18 H100-hours, ≥ 17.5B tokens processed, val perplexity ≤ 22, checkpoint saved to Drive | ~18 compute hours |
| M5 | SFT complete | Human eval ≥ 4.0/5.0 on 100 Turkish Q&A, grammar error rate ≤ 10% | 3–7 days |
| M6 | Local chatbot working | Terminal chat produces correct fluent Turkish in multi-turn conversation | 1–2 days |
| M7 | Web deployment | Chatbot live on personal website, responds within 30 seconds | 1–2 weeks |

---

## 11. Open Questions

| # | Question | Owner | Resolution Needed By |
|---|----------|-------|----------------------|
| 1 | Is H100 consistently available on Google Colab Pro, or does availability vary? | Aykan | Before M4 (pretraining) |
| 2 | What are the actual cleaned token counts from OSCAR 23.01 Turkish after dedup? | Aykan | Before M3 (data pipeline) |
| 3 | Does HuggingFace OSCAR 23.01 require authentication / license agreement? | Aykan | Before M3 |
| 4 | What web framework will be used for the chatbot website? (FastAPI + HTML, Next.js, etc.) | Aykan | Before M7 |
| 5 | Will model weights be public (HuggingFace Hub) or private? | Aykan | Before M7 |
| 6 | What quantization method for web inference — INT8 `bitsandbytes` or GGUF `llama.cpp`? | Aykan | Before M7 |
| 7 | What are the special token IDs for `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`, `<\|endoftext\|>` after tokenizer training? | Auto-resolved | After M1 (tokenizer) |

---

## 12. Appendix

### 12.1 Glossary

| Term | Definition |
|------|------------|
| BPE | Byte Pair Encoding — a subword tokenization algorithm that merges frequent character pairs |
| Fertility | Average number of tokens produced per word by a tokenizer — lower is more efficient |
| GQA | Grouped Query Attention — Q heads share K/V heads to reduce KV cache memory |
| RoPE | Rotary Position Embeddings — encodes position by rotating Q and K vectors |
| RMSNorm | Root Mean Square Normalization — simpler alternative to LayerNorm |
| SwiGLU | Swish-Gated Linear Unit — a gated feedforward activation outperforming GELU |
| SFT | Supervised Fine-Tuning — training on labeled input-output instruction pairs after pretraining |
| Chinchilla ratio | Tokens-per-parameter ratio; Chinchilla (2022) defines 20× as compute-optimal minimum |
| Perplexity | Exponentiation of cross-entropy loss — lower is better; measures how "surprised" the model is by held-out text |
| BF16 | BFloat16 — a 16-bit float format with wide dynamic range; preferred over FP16 for LLM training |
| KV cache | Key-Value cache — stores attention K and V tensors during generation to avoid recomputation |
| uint16 | Unsigned 16-bit integer — used for token ID storage in binary shards (supports vocab up to 65,535; 64k vocab comfortably fits) |
| RoPE θ (theta) | Base frequency for Rotary Position Embeddings; LLaMA 2 uses 10,000, LLaMA 3 uses 500,000 — higher theta improves long-context position discrimination |
| Curriculum training | Training data ordering strategy: expose the model to clean, formal text (Wikipedia) in Phase 1, then introduce noisier web text (OSCAR, mC4) in Phase 2 — accelerates convergence on small token budgets |

### 12.2 References

| Reference | URL |
|-----------|-----|
| LLaMA 2 paper | arxiv.org/abs/2307.09288 |
| Chinchilla scaling laws | arxiv.org/abs/2203.15556 |
| RoPE paper | arxiv.org/abs/2104.09864 |
| GQA paper | arxiv.org/abs/2305.13245 |
| SwiGLU paper | arxiv.org/abs/2002.05202 |
| Flash Attention 2 | arxiv.org/abs/2307.08691 |
| OSCAR 23.01 Turkish | huggingface.co/datasets/oscar-corpus/OSCAR-2301 |
| Turkish Wikipedia dump | dumps.wikimedia.org/trwiki/latest/ |
| mC4 Turkish | huggingface.co/datasets/mc4 |
| SentencePiece | github.com/google/sentencepiece |
| nanoGPT (reference impl) | github.com/karpathy/nanoGPT |

---

_Last updated: 4 Mart 2026_
