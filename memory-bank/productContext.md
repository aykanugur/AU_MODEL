# Product Context

## Problem Being Solved

Turkish NLP is severely underserved:
- All major LLMs (GPT-4, LLaMA, Mistral, Gemma) are primarily English-trained
- Turkish characters (ç, ğ, ı, İ, ö, ş, ü) are byte-fallback tokens in existing tokenizers
- A single Turkish word can cost 3–5 tokens in models like GPT-2/LLaMA — up to 40% token waste
- No high-quality open-source Turkish-native LLM exists as of 2026

## Target Users

| Persona | Need |
|---------|------|
| Turkish speakers | A chatbot that actually understands Turkish fluently |
| Turkish developers | A base model to fine-tune for Turkish domain tasks |
| Researchers | A reproducible, open Turkish LLM training pipeline |

## Success Looks Like

1. Model generates grammatically correct, fluent Turkish
2. Chatbot answers Turkish questions accurately and helpfully
3. Token fertility ratio: 1.3–1.8 tokens/word (vs. 3–5 for existing models)
4. Training pipeline is reproducible on a single H100

## User Experience Goals

- Chat interface responds in natural Turkish
- Context window of 4096 tokens (enough for long conversations)
- Inference latency < 500ms for typical chat responses (after deployment)
- Model doesn't hallucinate Turkish place names / cultural references

## Why Not Fine-Tune an Existing Model?

| Approach | Issue |
|----------|-------|
| Fine-tune LLaMA | Turkish still tokenized poorly — model "thinks in English" |
| Fine-tune mBERT | Encoder-only, can't generate |
| Fine-tune mT5 | Seq2seq, not optimized for chat |
| **Train from scratch** | True Turkish model, custom tokenizer, full control |
