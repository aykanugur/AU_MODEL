# Stories — Epic 7: MCP Web Search

**Epic ref:** `EPIC-07-mcp-web-search.md`
**Branch:** `epic/07-mcp-web-search`
**Persona:** Mixed — Developer (token integration, API client) / End-user (search-aware chat experience)
**Total stories:** 6

---

## ST-07-01 — Search Special Tokens

**As a developer,**
I want four dedicated special tokens registered in the vocabulary for web search control flow,
So that the model can express a search intent as a native part of the token sequence, not as a post-hoc parsing hack.

### Acceptance Criteria

- Four tokens are added to the tokenizer: `<|search|>` (ID 64001), `<|/search|>` (ID 64002), `<|search_results|>` (ID 64003), `<|/search_results|>` (ID 64004).
- Encoding any of these four strings returns the correct single-token ID.
- Decoding any of these four IDs returns the original string exactly.
- `TurkishTokenizer.special_ids()` returns all four IDs alongside the existing special token IDs.

---

## ST-07-02 — Model Embedding Resize

**As a developer,**
I want the model's embedding and output layers extended to cover the 4 new token IDs without losing existing weights,
So that the model can generate and receive all 64,004 distinct token IDs without an index out-of-range error.

### Acceptance Criteria

- The embedding layer is resized to accommodate 64,004 tokens.
- All 64,000 original weight rows are preserved exactly after the resize.
- The 4 new embedding rows are initialised with small random values (`std=0.02`), not zeros.
- The output (LM head) layer is resized identically and weight tying with the embedding is re-applied.
- A forward pass with any of the 4 new token IDs as input does not raise an error.
- The model's output logits have dimension 64,004.

---

## ST-07-03 — Brave Search API Client

**As a developer,**
I want a search client that queries Brave Search and returns the top-3 result snippets as plain text,
So that the generation loop can inject concrete, up-to-date web content without the model needing to know how the API works.

### Acceptance Criteria

- The client reads the API key from the `BRAVE_API_KEY` environment variable — no hardcoded credentials.
- A valid search query returns the top 3 result descriptions joined by `\n---\n`.
- The request has a 5-second timeout.
- If the API call fails for any reason (invalid key, network error, timeout), the client returns the Turkish fallback string `"[Arama başarısız oldu]"` instead of raising an exception.
- The client does not crash on queries containing non-ASCII Turkish characters.

---

## ST-07-04 — Search Interception in Generation Loop

**As a developer,**
I want the generation loop to detect when the model produces a `<|search|>...<|/search|>` span and inject real search results before continuing,
So that the model's web-search capability is fully automatic and transparent to the caller.

### Acceptance Criteria

- After each token is appended, the generation loop checks whether the latest tokens form a complete `<|search|>query<|/search|>` sequence.
- When detected, the query text between the search tokens is decoded and passed to the search client.
- The returned snippets are encoded as `<|search_results|>{snippets}<|/search_results|>` and appended to the input context.
- Generation resumes from the token immediately after `<|/search_results|>`.
- If the search client returns the fallback string, generation still resumes — the loop does not stall.

---

## ST-07-05 — Search SFT Training Examples

**As a developer,**
I want at least 500 labelled examples added to the SFT dataset where the correct response begins with a search invocation,
So that the model reliably learns when to trigger search instead of answering from outdated pretrained knowledge.

### Acceptance Criteria

- ≥ 500 instruction pairs are appended to `data/instruction/manual_tr.jsonl` where the response starts with `<|search|>{query}<|/search|>`.
- Example categories covered: current weather (≥ 100), today's news/sports (≥ 100), exchange rates or prices (≥ 100), recent events after 2024 (≥ 100), live data requests (≥ 100).
- Each example is valid JSON on a single line.
- After adding these examples, SFT fine-tuning is re-run to produce an updated `sft_best.pt` that has seen the search examples.

---

## ST-07-06 — End-User Search Experience in Chat

**As a Turkish speaker,**
I want the model to automatically search the web when I ask about current events or live data, and show me what it searched for,
So that I get up-to-date answers in Turkish without manually going to a search engine.

### Acceptance Criteria

- When I ask a time-sensitive question in Turkish (e.g., current exchange rate, today's news), the model triggers a search at least 70% of the time across 20 test prompts.
- The terminal prints `[Aranıyor: "{query}"]` before displaying the final response, so I know a live search happened.
- The final response incorporates the retrieved information — not a hallucinated answer.
- If the search fails, the model still gives a response (acknowledging it could not retrieve current data) rather than crashing or staying silent.
- Search results added to context appear in subsequent turns so the model can reference them in follow-up questions.

---

_Epic complete when all 6 stories pass their acceptance criteria._
_Search trigger target: ≥ 70% on 20 manually tested time-sensitive Turkish prompts._
_Last updated: 4 Mart 2026_
