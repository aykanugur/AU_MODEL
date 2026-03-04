# Epic 7 — MCP Web Search

| Field | Value |
|-------|-------|
| **Branch** | `epic/07-mcp-web-search` |
| **Base branch** | `epic/06-inference` |
| **Merge target** | `main` |
| **PRD refs** | G8 |
| **Depends on** | Epic 6 — needs working `generate()` loop |
| **Status** | ⬜ Not started |
| **Output files** | `inference/mcp_search.py` (new), updated `inference/generate.py`, updated `inference/chat.py`, 500 search SFT examples added to `data/instruction/manual_tr.jsonl` |

---

## Goal

Add 4 special tokens for search to the vocabulary. At inference, when generation produces `<|search|>query<|/search|>`, intercept the loop, call Brave Search API, inject top-3 result snippets as `<|search_results|>...<|/search_results|>` into the context, and resume generation. The model must learn when to trigger search via ≥ 500 SFT examples.

---

## Special Token IDs

| Token | ID |
|-------|----|
| `<\|search\|>` | 64001 |
| `<\|/search\|>` | 64002 |
| `<\|search_results\|>` | 64003 |
| `<\|/search_results\|>` | 64004 |

---

## Tasks

- [ ] **Extend tokenizer** — call `spm.SentencePieceTrainer.train(..., user_defined_symbols=['<|search|>','<|/search|>','<|search_results|>','<|/search_results|>'])` to add 4 tokens; update `TurkishTokenizer.special_ids()` to include them
- [ ] **Resize model embeddings** — `model.tok_embedding = nn.Embedding(64004, 1536)`: copy existing 64000 rows, initialize 4 new rows with `torch.nn.init.normal_(std=0.02)`; resize `model.lm_head` weight to `(64004, 1536)` identically; re-apply weight tying
- [ ] **`inference/mcp_search.py`** — `MCPSearchClient`: `search(query: str) → str`: `requests.get('https://api.search.brave.com/res/v1/web/search', params={'q': query, 'count': 3}, headers={'Accept': 'application/json', 'X-Subscription-Token': os.environ['BRAVE_API_KEY']}, timeout=5)`; parse `results[*].description`; return top-3 joined with `\n---\n`; on any exception return `"[Arama başarısız oldu]"`
- [ ] **Update `inference/generate.py`** — after appending each new token, check if the last generated sequence ends with `[..., ID(search), ..., ID(/search)]` pattern; if yes: extract all token IDs between `<|search|>` and `<|/search|>`, decode to query string; call `MCPSearchClient.search(query)`; encode `<|search_results|>{snippets}<|/search_results|>` and append to `input_ids`; resume generation from there
- [ ] **Update `inference/chat.py`** — when search is triggered, print `[Aranıyor: "{query}"]` to stdout before showing the response; include the full `<|search_results|>` span in history so the model sees its own past searches in multi-turn context
- [ ] **Add 500 search SFT examples** — append to `data/instruction/manual_tr.jsonl`: ≥ 500 pairs where the correct `response` begins with `<|search|>{relevant query}<|/search|>`; query types: current weather (≥ 100), today's news/sports (≥ 100), exchange rates/prices (≥ 100), recent events post-2024 (≥ 100), live data requests (≥ 100); re-run SFT fine-tune step (`sft/sft_trainer.py`) with the expanded dataset to produce updated `sft_best.pt`

---

## Exit Criteria

| Check | Pass Condition |
|-------|---------------|
| Token ID roundtrip | `tokenizer.decode([64001]) == '<|search|>'` and same for all 4 tokens |
| Model accepts 64004 vocab | `model(torch.tensor([[64001]])).shape == (1, 1, 64004)` (no index error) |
| Search trigger rate | ≥ 70% on 20 manually tested time-sensitive Turkish prompts (e.g., "Bugün dolar kaç?") |
| Search API latency | ≤ 5 seconds per query (p50 over 10 requests) |
| Factual accuracy | ≥ 80% of 20 search-grounded responses contain no wrong factual claim (manual check) |
| Graceful failure | `MCPSearchClient.search()` with invalid API key returns fallback string; `generate()` continues |

---

## Unlocks

- **Epic 9** (Web Deployment) — MCP search needed for full feature set in web chat

---

_Last updated: 4 Mart 2026_
