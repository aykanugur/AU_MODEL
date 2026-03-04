# Stories — Epic 8: DPO & Preference Collection

**Epic ref:** `EPIC-08-dpo.md`
**Branch:** `epic/08-dpo`
**Persona:** Mixed — End-user (A/B choice experience) / Developer (DPO training pipeline)
**Total stories:** 5

---

## ST-08-01 — A/B Response Display in Chat

**As a Turkish speaker,**
I want to see two alternative responses every other turn and choose which one I prefer,
So that I can give direct feedback that improves the model's future outputs without filling out any external form.

### Acceptance Criteria

- On every second assistant turn (turn 2, 4, 6, …), the terminal displays two options: `[A]` and `[B]`.
- Response A is generated with conservative settings (lower temperature, higher precision).
- Response B is generated with creative settings (higher temperature, more variation).
- Both responses are in Turkish and are complete sentences.
- The terminal then prompts `Hangisi daha iyi? (A/B):` and waits for my input.
- On odd turns (turn 1, 3, 5, …), only a single response is shown with no A/B prompt.

---

## ST-08-02 — Preference Logging

**As a developer,**
I want every user preference choice written immediately to a JSONL log file,
So that preference data accumulates persistently across sessions and can be fed into DPO training when enough pairs exist.

### Acceptance Criteria

- Every A/B choice appends one entry to `data/preference/preference_log.jsonl`.
- Each entry contains exactly four keys: `prompt` (the full conversation context up to the response), `chosen` (the preferred response text), `rejected` (the non-preferred response text), and `timestamp` (UTC ISO format).
- The file is appended to, not overwritten — data from previous sessions is preserved.
- If the user enters anything other than `A` or `B`, the session prompts again without logging a malformed entry.
- The file is valid JSONL — each line is independently parseable as JSON.

---

## ST-08-03 — Minimum Data Gate for DPO

**As a developer,**
I want DPO training to refuse to start if fewer than 10,000 preference pairs exist,
So that the DPO model is never trained on insufficient data that would degrade rather than improve quality.

### Acceptance Criteria

- When the DPO dataset is initialised, it counts the number of entries in `preference_log.jsonl`.
- If the count is below 10,000, the dataset raises an error with a message stating the current count and the required minimum.
- If the count is ≥ 10,000, the dataset loads all pairs without error.
- This check happens before any model weights are loaded, so the error is fast and clear.

---

## ST-08-04 — DPO Fine-Tuning

**As a developer,**
I want DPO training to fine-tune the SFT checkpoint using collected preference pairs and produce `dpo_best.pt`,
So that the final model has been aligned to human preference judgements rather than just trained on curated examples.

### Acceptance Criteria

- Training loads two instances of `sft_best.pt`: one as the trainable policy and one as the frozen reference model.
- The reference model's parameters are never updated during training.
- The DPO loss is computed as the log-sigmoid of a scaled log-ratio between policy and reference log-probabilities on chosen vs. rejected responses.
- Loss is logged every 50 steps.
- Training runs for exactly 1 epoch over all preference pairs.
- `checkpoints/dpo_best.pt` is saved at the end of the epoch.
- `dpo_best.pt` is loadable without errors after training.

---

## ST-08-05 — DPO Quality Improvement

**As a Turkish speaker,**
I want the DPO-trained model to produce better responses than the SFT-only model in a side-by-side comparison,
So that my preference feedback has had a measurable positive effect on the model's output quality.

### Acceptance Criteria

- In a manual side-by-side evaluation of 20 prompt pairs, `dpo_best.pt` is preferred over `sft_best.pt` in ≥ 60% of comparisons.
- Evaluation is conducted by the author in Turkish.
- Both models receive the same prompts under the same sampling parameters.
- Evaluator does not know which response comes from which model during rating.

---

_Epic complete when all 5 stories pass their acceptance criteria._
_Minimum data threshold: 10,000 preference pairs before DPO training is permitted._
_Last updated: 4 Mart 2026_
