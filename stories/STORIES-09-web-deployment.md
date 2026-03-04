# Stories — Epic 9: Web Deployment

**Epic ref:** `EPIC-09-web-deployment.md`
**Branch:** `epic/09-web-deployment`
**Persona:** Mixed — End-user (chat web interface) / Developer (quantization, API, Docker)
**Total stories:** 7

---

## ST-09-01 — Model Quantization

**As a developer,**
I want the DPO model quantized to INT8 and verified to still give coherent responses,
So that the server runs on a cheap VPS with ≤ 2 GB RAM without sacrificing the overall conversation quality.

### Acceptance Criteria

- Running the quantization script converts `dpo_best.pt` to INT8 and saves it as `checkpoints/dpo_int8.pt`.
- The script prints memory usage before and after: BF16 baseline ≈ 1.4 GB, INT8 target ≈ 0.7 GB.
- `dpo_int8.pt` file size is ≤ 750 MB on disk.
- After quantization, 10 Turkish spot-check prompts are run and results are printed for manual review.
- Author rates the spot-check outputs ≥ 3.5 / 5.0 on average — response quality is still acceptable.

---

## ST-09-02 — Chat API Backend

**As a developer,**
I want a FastAPI server that handles chat requests and A/B turns from any client,
So that the web frontend and any future clients can query the model over HTTP without touching Python inference code directly.

### Acceptance Criteria

- `POST /chat` accepts a session ID, conversation history, and a new user message.
- On even turns (2, 4, 6, …) the response includes both `response_a` and `response_b` with `is_ab_turn: true`.
- On odd turns the response includes only `response_a` with `response_b: null` and `is_ab_turn: false`.
- `POST /feedback` accepts a prompt, chosen response, and rejected response, appends them to `preference_log.jsonl`, and returns a success status.
- `GET /health` returns `{"status": "ok"}` and the model name within 1 second.
- Each `/chat` request is wrapped in a 55-second timeout — the server returns a failure response rather than hanging indefinitely.
- The model and tokenizer are loaded once at server startup into memory — not reloaded per request.

---

## ST-09-03 — Web Chat Interface

**As a Turkish speaker,**
I want to open a web page and chat with the model in Turkish from my browser without installing anything,
So that I can use the model as a day-to-day assistant from any device.

### Acceptance Criteria

- Visiting the page shows a chat input at the bottom and a message area above.
- I can type a message and press Enter or a send button to submit.
- My messages appear right-aligned; model messages appear left-aligned.
- After I send a message, a visible loading state (e.g., spinner or "..." indicator) appears until the response arrives.
- The page auto-scrolls to the newest message after each turn.
- The page works on both desktop and mobile screen sizes.

---

## ST-09-04 — A/B Preference Buttons in Web UI

**As a Turkish speaker,**
I want to see two response options on every second turn with buttons to choose my favourite,
So that my preference feedback continues to be collected even when I use the model through the website.

### Acceptance Criteria

- On even turns, two response bubbles appear: one with a blue tint (`[A]`) and one with a red tint (`[B]`).
- Below the two bubbles, two buttons appear: `A daha iyi` and `B daha iyi`.
- Clicking a button sends my preference to `/feedback`, hides the buttons, and shows only the chosen response as the final message.
- On odd turns no buttons appear — only a single response bubble is shown.
- Button clicks are not duplicated if clicked more than once (buttons are disabled after first click).

---

## ST-09-05 — Web Search Indicator

**As a Turkish speaker,**
I want to see a visible indicator when the model is performing a web search,
So that I understand why a response is taking longer and what the model is looking up on my behalf.

### Acceptance Criteria

- When the model generates a search token during response generation, the UI shows `🔍 Aranıyor: "{query}"` before the final response appears.
- The indicator disappears once the response is displayed.
- The indicator text is in Turkish regardless of browser language settings.

---

## ST-09-06 — Docker Deployment

**As a developer,**
I want the complete server packaged as a Docker image that runs with a single command,
So that I can deploy the model to any VPS without managing Python environments or installing dependencies manually.

### Acceptance Criteria

- The Dockerfile builds a runnable image from `python:3.10-slim`.
- The image includes the quantized model, tokenizer files, and web application code.
- Running `docker run -p 8000:8000` serves the API on port 8000.
- The server loads the model and serves the first `/health` response within 30 seconds of container start.
- `preference_log.jsonl` is written to a path that can be mounted as a persistent Docker volume so data survives container restarts.
- The Docker image runs on CPU only — no GPU required on the deployment host.

---

## ST-09-07 — Resilience Against Bad Input

**As a Turkish speaker,**
I want the web chat to handle unusual or edge-case input gracefully without crashing or showing an error page,
So that unusual messages don't interrupt my session or leave the server in a broken state.

### Acceptance Criteria

- Sending an empty message returns an HTTP 200 with a valid Turkish response (e.g., asking me to type something).
- Sending a 5,000-character message returns HTTP 200 with a valid Turkish response (context is truncated, not rejected).
- Sending text in Arabic script or with emoji characters returns HTTP 200 with a valid Turkish response.
- None of these inputs cause a server 5xx error or an unhandled exception in the logs.

---

_Epic complete when all 7 stories pass their acceptance criteria._
_Deployment target: CPU-only VPS, ≤ 2 GB RSS, first response ≤ 30 seconds after `docker run`._
_Last updated: 4 Mart 2026_
