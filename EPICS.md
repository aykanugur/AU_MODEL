# AUModel — EPICS Index

> **Project:** AUModel — Turkish Native LLM Chatbot  
> **Author:** Aykan Ugur  
> **Version:** 2.0  
> **Date:** 4 Mart 2026  
> **Linked PRD:** [PRD.md](PRD.md)  
> **Branch convention:** `epic/NN-name`

---

## Execution Order & Dependencies

```
Epic 1 (Tokenizer)
    │
    ├──► Epic 2 (Model Architecture)
    │
    └──► Epic 3 (Data Pipeline)
              │
              └──► Epic 4 (Pretraining)
                        │
                        └──► Epic 5 (SFT)
                                  │
                                  └──► Epic 6 (Inference & Chat)
                                            │
                                            ├──► Epic 7 (MCP Web Search)
                                            │
                                            └──► Epic 8 (DPO)
                                                      │
                                            Both ─────► Epic 9 (Web Deployment)
```

---

## Epics

| # | Epic | Branch | Status | File |
|---|------|--------|--------|------|
| 1 | Tokenizer | `epic/01-tokenizer` | ⬜ Not started | [EPIC-01-tokenizer.md](epics/EPIC-01-tokenizer.md) |
| 2 | Model Architecture | `epic/02-model-architecture` | ⬜ Not started | [EPIC-02-model-architecture.md](epics/EPIC-02-model-architecture.md) |
| 3 | Data Pipeline | `epic/03-data-pipeline` | ⬜ Not started | [EPIC-03-data-pipeline.md](epics/EPIC-03-data-pipeline.md) |
| 4 | Pretraining | `epic/04-pretraining` | ⬜ Not started | [EPIC-04-pretraining.md](epics/EPIC-04-pretraining.md) |
| 5 | Supervised Fine-Tuning (SFT) | `epic/05-sft` | ⬜ Not started | [EPIC-05-sft.md](epics/EPIC-05-sft.md) |
| 6 | Inference & Terminal Chat | `epic/06-inference` | ⬜ Not started | [EPIC-06-inference.md](epics/EPIC-06-inference.md) |
| 7 | MCP Web Search | `epic/07-mcp-web-search` | ⬜ Not started | [EPIC-07-mcp-web-search.md](epics/EPIC-07-mcp-web-search.md) |
| 8 | DPO & Preference Collection | `epic/08-dpo` | ⬜ Not started | [EPIC-08-dpo.md](epics/EPIC-08-dpo.md) |
| 9 | Web Deployment | `epic/09-web-deployment` | ⬜ Not started | [EPIC-09-web-deployment.md](epics/EPIC-09-web-deployment.md) |

---

## Key Artifacts Per Epic

| Epic | Input | Output |
|------|-------|--------|
| 1 | Raw Turkish text | `tokenizer/turkish_bpe.model` (64k vocab) |
| 2 | `vocab_size=64000` | `model/*.py` — 700M param `AUModel` class |
| 3 | `turkish_bpe.model` | `data/processed/shard_*.bin` ≥ 17.5B tokens, curriculum-ordered |
| 4 | Model + shards | `checkpoints/ckpt_033000.pt` — pretrained weights |
| 5 | `ckpt_033000.pt` | `checkpoints/sft_best.pt` — instruction-tuned weights |
| 6 | `sft_best.pt` | `inference/generate.py` + `inference/chat.py` — working REPL |
| 7 | Working `generate()` | `inference/mcp_search.py` + search tokens in vocab |
| 8 | `sft_best.pt` + chat REPL | `checkpoints/dpo_best.pt` + `data/preference/preference_log.jsonl` |
| 9 | `dpo_best.pt` + MCP | `web/app.py` + Docker — live website chatbot |

---

_Last updated: 4 Mart 2026_
