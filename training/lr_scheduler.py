"""Cosine learning-rate schedule with linear warm-up.

get_lr() is a pure function — no state, no PyTorch dependency — so it is
trivially testable and usable from CLI tooling without importing the trainer.
"""
import math


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Return the learning rate for *step* given the schedule parameters.

    Regions:
        step < 0               → clamped to 0, returns 0.0
        0 ≤ step < warmup_steps → linear warm-up  [0 … max_lr)
        warmup_steps ≤ step ≤ max_steps → cosine annealing [max_lr … min_lr]
        step > max_steps       → hold at min_lr

    No assertions are raised for out-of-range inputs; callers may pass any
    integer step (including values beyond max_steps during a resumed run).
    """
    # Clamp negative steps — treat them as step=0
    if step < 0:
        step = 0

    # 1 ── Linear warm-up
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # 2 ── Hold floor after schedule end
    if step > max_steps:
        return min_lr

    # 3 ── Cosine decay
    decay_ratio: float = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff: float = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
