"""Tests for get_lr() — SC-005 boundary-value suite."""
import math

import pytest

from training.lr_scheduler import get_lr

# ---------------------------------------------------------------------------
# Default schedule parameters (match production defaults)
# ---------------------------------------------------------------------------
WARMUP = 2_000
MAX_STEPS = 100_000
MAX_LR = 3e-4
MIN_LR = 3e-5


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------

class TestGetLrBoundaries:
    def test_step_zero_returns_zero(self):
        """Very first step: warm-up factor is 0/warmup → 0.0."""
        lr = get_lr(0, WARMUP, MAX_STEPS, MAX_LR, MIN_LR)
        assert lr == 0.0

    def test_end_of_warmup_returns_max_lr(self):
        """Step == warmup_steps transitions out of warm-up → decay region."""
        lr = get_lr(WARMUP, WARMUP, MAX_STEPS, MAX_LR, MIN_LR)
        assert abs(lr - MAX_LR) < 1e-12, f"Expected {MAX_LR}, got {lr}"

    def test_end_of_schedule_returns_min_lr(self):
        """Step == max_steps → cosine is fully decayed → min_lr."""
        lr = get_lr(MAX_STEPS, WARMUP, MAX_STEPS, MAX_LR, MIN_LR)
        assert abs(lr - MIN_LR) < 1e-12, f"Expected {MIN_LR}, got {lr}"

    def test_beyond_max_steps_holds_at_min_lr(self):
        """Step > max_steps → hold floor."""
        lr = get_lr(MAX_STEPS + 1, WARMUP, MAX_STEPS, MAX_LR, MIN_LR)
        assert lr == MIN_LR

    def test_negative_step_clamped_no_exception(self):
        """Negative steps must NOT raise; they are clamped to 0."""
        lr = get_lr(-1, WARMUP, MAX_STEPS, MAX_LR, MIN_LR)
        assert lr == 0.0

    def test_large_negative_step_clamped(self):
        lr = get_lr(-99_999, WARMUP, MAX_STEPS, MAX_LR, MIN_LR)
        assert lr == 0.0


# ---------------------------------------------------------------------------
# Warmup region
# ---------------------------------------------------------------------------

class TestWarmupRegion:
    def test_linear_proportionality(self):
        """LR at midpoint of warm-up should be ≈ max_lr / 2."""
        mid = WARMUP // 2
        lr = get_lr(mid, WARMUP, MAX_STEPS, MAX_LR, MIN_LR)
        expected = MAX_LR * mid / WARMUP
        assert abs(lr - expected) < 1e-12

    def test_monotonically_increasing_during_warmup(self):
        samples = [get_lr(s, WARMUP, MAX_STEPS, MAX_LR, MIN_LR) for s in range(0, WARMUP, 100)]
        for a, b in zip(samples, samples[1:]):
            assert b >= a, "LR should be non-decreasing during warm-up"

    def test_one_step_before_end_of_warmup(self):
        lr = get_lr(WARMUP - 1, WARMUP, MAX_STEPS, MAX_LR, MIN_LR)
        expected = MAX_LR * (WARMUP - 1) / WARMUP
        assert abs(lr - expected) < 1e-12


# ---------------------------------------------------------------------------
# Cosine decay region
# ---------------------------------------------------------------------------

class TestCosineDecayRegion:
    def test_midpoint_is_between_min_and_max(self):
        mid = (WARMUP + MAX_STEPS) // 2
        lr = get_lr(mid, WARMUP, MAX_STEPS, MAX_LR, MIN_LR)
        assert MIN_LR < lr < MAX_LR

    def test_midpoint_formula(self):
        """At the exact midpoint of decay the cosine gives 0.5*(max+min)."""
        mid = (WARMUP + MAX_STEPS) // 2
        lr = get_lr(mid, WARMUP, MAX_STEPS, MAX_LR, MIN_LR)
        decay_ratio = (mid - WARMUP) / (MAX_STEPS - WARMUP)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        expected = MIN_LR + coeff * (MAX_LR - MIN_LR)
        assert abs(lr - expected) < 1e-12

    def test_monotonically_decreasing_during_decay(self):
        steps = range(WARMUP, MAX_STEPS + 1, 1_000)
        samples = [get_lr(s, WARMUP, MAX_STEPS, MAX_LR, MIN_LR) for s in steps]
        for a, b in zip(samples, samples[1:]):
            assert b <= a + 1e-15, "LR should be non-increasing during cosine decay"

    def test_lr_stays_above_min_lr_during_decay(self):
        steps = range(WARMUP, MAX_STEPS + 1, 1_000)
        for s in steps:
            lr = get_lr(s, WARMUP, MAX_STEPS, MAX_LR, MIN_LR)
            assert lr >= MIN_LR - 1e-12, f"lr={lr} fell below min_lr at step {s}"


# ---------------------------------------------------------------------------
# Hold region
# ---------------------------------------------------------------------------

class TestHoldRegion:
    def test_holds_at_various_steps_beyond_max(self):
        for extra in [1, 100, 10_000, 999_999]:
            lr = get_lr(MAX_STEPS + extra, WARMUP, MAX_STEPS, MAX_LR, MIN_LR)
            assert lr == MIN_LR, f"Expected {MIN_LR} at step {MAX_STEPS + extra}, got {lr}"
