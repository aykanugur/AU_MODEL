# Project Brief

> **Core document — read this first every session.**

## What We Are Building

**AUModel** — A Turkish-native 1.3B parameter Large Language Model trained from scratch, fine-tuned as a conversational chatbot.

## Non-Negotiable Requirements

| Requirement | Value |
|-------------|-------|
| Language | Turkish (native, not translated) |
| Architecture | LLaMA-style decoder-only transformer |
| Parameters | ~1.3B |
| Training hardware | Google Colab Pro, NVIDIA H100 80GB |
| Training budget | ~100 hours |
| Token target | ~32B Turkish tokens (25× Chinchilla) |
| Tokenizer | Custom Turkish BPE (SentencePiece, 32k vocab) |
| Final product | Turkish-language chatbot |

## Why From Scratch

- Existing models (LLaMA, Mistral) use byte-fallback for Turkish characters (ç, ğ, ı, ö, ş, ü) — wasting ~40% of tokens
- A Turkish-native tokenizer with dedicated vocab entries makes the model fundamentally more efficient and accurate
- We want full ownership of the architecture and weights

## Project Phase

Current phase: **Pre-implementation** — design complete, code scaffolded, no implementation yet.

## Owner

Single developer — Aykan Ugur
