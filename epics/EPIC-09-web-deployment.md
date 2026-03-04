# Epic 9 — Web Deployment

| Field | Value |
|-------|-------|
| **Branch** | `epic/09-web-deployment` |
| **Base branch** | `main` (after Epic 7 + Epic 8 merged) |
| **Merge target** | `main` |
| **PRD refs** | F-10, G6, M7 |
| **Depends on** | Epic 7 (`dpo_best.pt` with MCP tokens), Epic 8 (`dpo_best.pt` DPO-tuned) |
| **Status** | ⬜ Not started |
| **Output files** | `web/quantize.py`, `web/app.py`, `web/static/index.html`, `web/static/style.css`, `Dockerfile` |

---

## Goal

Quantize `dpo_best.pt` to INT8 (~0.7 GB), serve it via a FastAPI backend, and embed a Turkish chat UI on a personal website. Every 2nd web response shows both A and B with preference buttons to continue DPO data collection from real users. First response arrives within ≤ 30 seconds.

---

## Tasks

- [ ] **`web/quantize.py`** — load `dpo_best.pt`; `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`; run 10 spot-check Turkish prompts and manually verify responses still make sense (≥ 3.5/5.0 author rating); save to `checkpoints/dpo_int8.pt`; print memory before/after: `BF16 ≈ 1.4 GB → INT8 ≈ 0.7 GB`
- [ ] **`web/app.py`** — FastAPI app:
  - load `dpo_int8.pt` and tokenizer once at startup into module-level globals
  - `POST /chat` — body: `{session_id: str, history: List[{role, content}], message: str}` → returns `{response_a: str, response_b: str | null, is_ab_turn: bool, turn_index: int}`: use `session_id` keyed in-memory dict for `turn_counter`; on even turns generate A (temp=0.7) and B (temp=1.1); on odd turns generate A only and set `response_b=null`; wrap `generate()` in `asyncio.wait_for(..., timeout=55.0)` to ensure response before 60s client timeout
  - `POST /feedback` — body: `{session_id: str, prompt: str, chosen: str, rejected: str}` → append to `data/preference/preference_log.jsonl` and return `{"status": "ok"}`
  - `GET /health` → `{"status": "ok", "model": "AUModel-700M-DPO-INT8"}`
- [ ] **`web/static/index.html`** — chat UI:
  - text input at bottom, send on Enter or button click
  - message bubbles: user messages right-aligned, AUModel messages left-aligned
  - on `is_ab_turn=true`: show `[A]` and `[B]` response in two separate colored bubbles; show "A daha iyi" / "B daha iyi" buttons below; on click: `POST /feedback` then hide buttons and show the chosen response as the canonical message
  - show `🔍 Aranıyor: "{query}"` loading indicator when search is running
  - auto-scroll to bottom after each message
- [ ] **`web/static/style.css`** — minimal styling: white background, `font-family: system-ui`, bubbles with `border-radius: 12px`; A-bubble `background: #e8f0fe`, B-bubble `background: #fce8e6`; preference buttons `background: #1a73e8 color: white`
- [ ] **`Dockerfile`** — `FROM python:3.10-slim`; `pip install torch==2.2.0+cpu sentencepiece fastapi uvicorn`; `COPY checkpoints/dpo_int8.pt /app/checkpoints/`; `COPY tokenizer/ /app/tokenizer/`; `COPY web/ /app/web/`; `EXPOSE 8000`; `CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]`

---

## Exit Criteria

| Check | Pass Condition |
|-------|---------------|
| Cold-start latency | Server loads model and serves first response within ≤ 30 seconds from `docker run` |
| INT8 memory footprint | `dpo_int8.pt` file size ≤ 750 MB; server RSS ≤ 2 GB under load |
| 5-turn session integrity | History sent on request 5 includes turns 1–4; model references earlier content correctly |
| A/B buttons appear on turn 2 | `is_ab_turn=true` on turn index 2, 4, 6, ... — verified by browser inspection |
| `/feedback` writes to JSONL | `tail -1 data/preference/preference_log.jsonl` shows new entry after button click |
| Input edge cases | Empty string, 5000-char input, Arabic/emoji input all return HTTP 200 with a valid Turkish response |
| `/health` endpoint | `curl /health` returns `{"status": "ok"}` within 1 second |

---

## Deployment Notes

- Start with CPU-only Docker image on a cheap VPS (≥ 2 GB RAM, e.g., Hetzner CX21 €4/month)
- `uvicorn` runs single-threaded with 1 worker — concurrent users queue; acceptable for personal site traffic
- If traffic > 5 concurrent users: switch to GPU instance or add `gunicorn` with multiple workers
- `data/preference/preference_log.jsonl` must be mounted as a persistent volume in Docker so DPO data survives container restarts

---

## Unlocks

Nothing — this is the final epic.

---

_Last updated: 4 Mart 2026_
