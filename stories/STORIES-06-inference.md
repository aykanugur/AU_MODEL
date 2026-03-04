# Stories — Epic 6: Inference & Terminal Chat

**Epic ref:** `EPIC-06-inference.md`
**Branch:** `epic/06-inference`
**Persona:** Mixed — Developer (generation engine) / End-user (chat experience)
**Total stories:** 6

---

## ST-06-01 — Autoregressive Token Generation

**As a developer,**
I want a `generate()` function that produces tokens one at a time using temperature, top-k, and top-p sampling,
So that all other epics and the chat interface can produce varied, controllable outputs by passing sampling parameters.

### Acceptance Criteria

- The function accepts `max_new_tokens`, `temperature`, `top_p`, `top_k`, and `repetition_penalty` as parameters.
- Sampling runs under inference mode — no gradients are tracked.
- Temperature is applied by dividing logits before softmax.
- Top-k filtering retains only the highest-k logits; the rest are set to negative infinity.
- Top-p (nucleus) filtering keeps the smallest subset of tokens whose cumulative probability meets or exceeds `top_p`.
- The repetition penalty reduces the logit probability of any token that already appeared in the generated sequence.
- Generation stops when the EOS token ID is produced or `max_new_tokens` is reached, whichever comes first.
- The function returns only the newly generated token IDs, not the original prompt IDs.

---

## ST-06-02 — Generation Speed

**As a developer,**
I want the generation function to produce 200 tokens within 10 seconds on H100,
So that the system is fast enough for an interactive chat experience without noticeable lag.

### Acceptance Criteria

- Generating 200 new tokens from a prompt of typical chat length completes in ≤ 10 seconds on H100 hardware.
- Measurement uses wall-clock time, not CPU time.
- Test is repeatable — running the same prompt three consecutive times each stays under 10 seconds.

---

## ST-06-03 — Multi-Turn Context Management

**As a developer,**
I want the chat session to maintain a history of all turns and trim the oldest turns when context is full,
So that multi-turn conversations are coherent without crashing due to exceeding the 4,096-token window.

### Acceptance Criteria

- The session stores every user and assistant turn as structured history (role + content pairs).
- Before each generation, the full history is formatted into the chat template and the token count is checked.
- If the formatted context exceeds 3,800 tokens, the oldest non-system turn pair is dropped and the count is rechecked.
- The trimming loop repeats until the context fits within 3,800 tokens.
- The system prompt (if present) is never dropped during trimming.

---

## ST-06-04 — Stopping at End-of-Sequence

**As a developer,**
I want generation to stop immediately when the EOS token is produced,
So that the model never outputs trailing garbage tokens after a natural sentence ending.

### Acceptance Criteria

- If the EOS token ID is the next predicted token, generation stops and the EOS token is not included in the returned list.
- Generation never continues past EOS even if `max_new_tokens` has not been reached.
- This behaviour is verifiable by inspecting the decoded output — no tokens appear after the final sentence ends.

---

## ST-06-05 — Terminal Chat Session

**As a Turkish speaker,**
I want to chat with the model in a terminal loop where I type in Turkish and see a Turkish response,
So that I can have a back-and-forth conversation with the model on my machine without any web browser or setup beyond running a script.

### Acceptance Criteria

- The REPL prompts me with `Sen:` and waits for my input each turn.
- After I press Enter, the model generates and prints a response prefixed with `AUModel:`.
- The conversation continues turn after turn without me restarting the script.
- Pressing `Ctrl+C` ends the session cleanly and prints `Görüşmek üzere!` before exiting.
- The session does not crash on an empty input or very long input.

---

## ST-06-06 — 5-Turn Contextual Coherence

**As a Turkish speaker,**
I want responses in later turns to reflect information I shared in earlier turns,
So that the conversation feels connected — the model doesn't treat each message as an isolated question.

### Acceptance Criteria

- In a 5-turn test conversation, when I reference something I mentioned in turn 1 during turn 5, the model's response correctly incorporates that earlier information.
- This test is evaluated manually by the author.
- The test conversation is conducted in Turkish.
- The model does not contradict itself between turns 1 and 5 on the same topic.

---

_Epic complete when all 6 stories pass their acceptance criteria._
_Last updated: 4 Mart 2026_
