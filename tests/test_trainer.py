"""Trainer integration tests.

T009 — smoke test (50 steps, checkpoint + loss range + MFU)
T011 — resume test (20 + 1 steps, step numbering + loss delta)
T014 — logging format + validation loss test
"""
from __future__ import annotations

import io
import pathlib
import re
import sys
import warnings

import pytest
import torch

from training.trainer import TrainingConfig, estimate_mfu, train

# ---------------------------------------------------------------------------
# Common fixture paths
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).parent
_FIXTURE_SHARD_DIR = str(_HERE / "fixtures/data")
_VAL_SHARD = str(_HERE / "fixtures/data/val/shard_00000.bin")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_train(cfg: TrainingConfig) -> str:
    """Run train(cfg) and return everything printed to stdout."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train(cfg)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


def _parse_log_lines(output: str) -> list[dict[str, str]]:
    """Parse 'key=value' fields from each non-empty log line."""
    result = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line or line.startswith("Resumed"):
            continue
        fields = line.split()
        row = {}
        for f in fields:
            if "=" in f:
                k, v = f.split("=", 1)
                row[k] = v
        if row:
            result.append(row)
    return result


# ---------------------------------------------------------------------------
# T009  Smoke test — 50 steps
# ---------------------------------------------------------------------------


class TestSmokeTraining:
    """US-1 — basic training loop runs without error and produces checkpoints."""

    def test_checkpoint_file_exists(self, tmp_path):
        """step_000010.pt must be created within the first 10 steps."""
        cfg = TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=str(tmp_path),
            max_steps=50,
            warmup_steps=10,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            checkpoint_interval=10,
            log_interval=10,
            val_interval=10_000,
            use_wandb=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _capture_train(cfg)

        assert (tmp_path / "step_000010.pt").exists(), "step_000010.pt not found"

    def test_first_logged_loss_in_expected_range(self, tmp_path):
        """First logged loss (step 10) should be in [10.5, 11.6] for a random model.

        Expected CE for a random model ≈ ln(64_000) ≈ 11.07.
        """
        cfg = TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=str(tmp_path),
            max_steps=50,
            warmup_steps=10,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            checkpoint_interval=10,
            log_interval=10,
            val_interval=10_000,
            use_wandb=False,
        )
        output = _capture_train(cfg)
        rows = _parse_log_lines(output)

        assert rows, "No log lines found in training output"
        first_row = rows[0]
        assert "loss" in first_row, f"'loss' field missing from: {first_row}"

        loss_val = float(first_row["loss"])
        assert 10.5 <= loss_val <= 11.6, (
            f"First logged loss {loss_val:.4f} is outside expected range [10.5, 11.6]"
        )

    def test_first_logged_step_is_10(self, tmp_path):
        """With log_interval=10 the very first logged step must be step 10."""
        cfg = TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=str(tmp_path),
            max_steps=50,
            warmup_steps=10,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            checkpoint_interval=10,
            log_interval=10,
            val_interval=10_000,
            use_wandb=False,
        )
        output = _capture_train(cfg)
        rows = _parse_log_lines(output)

        assert rows
        assert int(rows[0]["step"]) == 10, (
            f"Expected first logged step=10, got {rows[0]['step']}"
        )

    def test_all_expected_checkpoints_exist(self, tmp_path):
        """At 50 steps with checkpoint_interval=10, 5 checkpoints must exist."""
        cfg = TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=str(tmp_path),
            max_steps=50,
            warmup_steps=10,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            checkpoint_interval=10,
            log_interval=10,
            val_interval=10_000,
            use_wandb=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _capture_train(cfg)

        ckpts = sorted(tmp_path.glob("step_*.pt"))
        assert len(ckpts) == 5, f"Expected 5 checkpoints, found {len(ckpts)}"


# ---------------------------------------------------------------------------
# T009  estimate_mfu boundary tests
# ---------------------------------------------------------------------------


class TestEstimateMfu:
    def test_zero_tokens_per_sec_returns_zero(self):
        assert estimate_mfu(0.0) == 0.0

    def test_reasonable_throughput_in_valid_range(self):
        """100_000 tok/s on H100 should give a plausible MFU fraction."""
        mfu = estimate_mfu(100_000)
        assert 0.10 <= mfu <= 0.70, (
            f"estimate_mfu(100_000)={mfu:.4f} is outside [0.10, 0.70]"
        )

    def test_explicit_seq_len_overrides_default(self):
        """Smaller seq_len means lower attention FLOPs → lower MFU."""
        mfu_64 = estimate_mfu(100_000, seq_len=64)
        mfu_4096 = estimate_mfu(100_000, seq_len=4096)
        assert mfu_64 < mfu_4096

    def test_mfu_proportional_to_throughput(self):
        """MFU should scale linearly with tokens_per_sec."""
        mfu_1x = estimate_mfu(50_000)
        mfu_2x = estimate_mfu(100_000)
        assert abs(mfu_2x - 2 * mfu_1x) < 1e-10


# ---------------------------------------------------------------------------
# T011  Resume test — 20 + 1 steps
# ---------------------------------------------------------------------------


class TestResumeTraining:
    """US-2 — training resumes seamlessly after checkpoint restore."""

    def _base_cfg(self, output_dir: str) -> TrainingConfig:
        return TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=output_dir,
            max_steps=21,
            warmup_steps=5,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            checkpoint_interval=10,
            log_interval=1,
            val_interval=10_000,
            use_wandb=False,
        )

    def test_first_logged_step_after_resume_is_21(self, tmp_path):
        """After running 20 steps, a restart must begin logging at step=21."""
        # ── First run: 20 steps
        cfg = self._base_cfg(str(tmp_path))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg.max_steps = 20
            _capture_train(cfg)

        assert (tmp_path / "step_000020.pt").exists(), "step_000020.pt not written"

        # ── Second run: resume
        cfg2 = self._base_cfg(str(tmp_path))
        cfg2.max_steps = 21  # one more step
        output = _capture_train(cfg2)

        rows = _parse_log_lines(output)
        assert rows, "No log lines after resume"
        first_step = int(rows[0]["step"])
        assert first_step == 21, (
            f"Expected first logged step after resume to be 21, got {first_step}"
        )

    def test_resumed_loss_close_to_continuous_run(self, tmp_path):
        """Loss at step 21 must be nearly identical whether resumed or continuous."""
        # ── Continuous 21-step run
        out_a = tmp_path / "continuous"
        cfg_a = TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=str(out_a),
            max_steps=21,
            warmup_steps=5,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            checkpoint_interval=25,    # no checkpoint during the run
            log_interval=1,
            val_interval=10_000,
            use_wandb=False,
        )
        output_a = _capture_train(cfg_a)
        rows_a = _parse_log_lines(output_a)
        loss_a_21 = float(next(r["loss"] for r in rows_a if int(r["step"]) == 21))

        # ── Resumed run: 20 steps → checkpoint → 1 more step
        out_b = tmp_path / "resumed"
        cfg_b = TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=str(out_b),
            max_steps=20,
            warmup_steps=5,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            checkpoint_interval=10,
            log_interval=1,
            val_interval=10_000,
            use_wandb=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _capture_train(cfg_b)

        cfg_b2 = TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=str(out_b),
            max_steps=21,
            warmup_steps=5,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            checkpoint_interval=10,
            log_interval=1,
            val_interval=10_000,
            use_wandb=False,
        )
        output_b = _capture_train(cfg_b2)
        rows_b = _parse_log_lines(output_b)
        loss_b_21 = float(rows_b[0]["loss"])  # first (and likely only) line in resumed output

        assert abs(loss_b_21 - loss_a_21) < 0.01, (
            f"Resumed loss at step 21 ({loss_b_21:.4f}) diverges from continuous "
            f"loss ({loss_a_21:.4f}) by more than 0.01"
        )


# ---------------------------------------------------------------------------
# T014  Logging format + validation loss
# ---------------------------------------------------------------------------


class TestLogFormat:
    """US-3 — log lines have correct format (8 fields, correct step numbers)."""

    def test_exactly_4_log_lines(self, tmp_path):
        """20 steps with log_interval=5 → 4 log lines (steps 5, 10, 15, 20)."""
        cfg = TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=str(tmp_path),
            max_steps=20,
            warmup_steps=5,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            log_interval=5,
            checkpoint_interval=10,
            val_interval=10_000,
            use_wandb=False,
        )
        output = _capture_train(cfg)
        rows = _parse_log_lines(output)
        assert len(rows) == 4, f"Expected 4 log lines, got {len(rows)}: {rows}"

    def test_log_line_has_8_fields(self, tmp_path):
        """Each log line must have exactly 8 key=value fields."""
        cfg = TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=str(tmp_path),
            max_steps=20,
            warmup_steps=5,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            log_interval=5,
            checkpoint_interval=10,
            val_interval=10_000,
            use_wandb=False,
        )
        output = _capture_train(cfg)
        for raw in output.splitlines():
            line = raw.strip()
            if not line or line.startswith("Resumed"):
                continue
            fields = line.split()
            key_value = [f for f in fields if "=" in f]
            assert len(key_value) == 8, (
                f"Expected 8 key=value fields, got {len(key_value)} in: {line!r}"
            )

    def test_val_loss_is_dash_when_no_validation(self, tmp_path):
        """With val_interval=10_000 all val_loss entries should be '-'."""
        cfg = TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=str(tmp_path),
            max_steps=20,
            warmup_steps=5,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            log_interval=5,
            checkpoint_interval=10,
            val_interval=10_000,
            use_wandb=False,
        )
        output = _capture_train(cfg)
        rows = _parse_log_lines(output)
        for row in rows:
            assert row.get("val_loss") == "-", (
                f"Expected val_loss='-', got val_loss={row.get('val_loss')!r} in {row}"
            )

    def test_step_numbers_are_correct(self, tmp_path):
        """Log lines must appear at steps 5, 10, 15, 20 with log_interval=5."""
        cfg = TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=str(tmp_path),
            max_steps=20,
            warmup_steps=5,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            log_interval=5,
            checkpoint_interval=10,
            val_interval=10_000,
            use_wandb=False,
        )
        output = _capture_train(cfg)
        rows = _parse_log_lines(output)
        steps = [int(r["step"]) for r in rows]
        assert steps == [5, 10, 15, 20], f"Expected steps [5,10,15,20], got {steps}"


class TestValidationLoss:
    """US-3 (continued) — validation loss is computed and logged correctly."""

    def test_val_loss_is_finite_at_val_interval(self, tmp_path):
        """With val_interval=10 and 15 steps, at step 10 val_loss must be a finite float."""
        cfg = TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=str(tmp_path),
            val_shard=_VAL_SHARD,
            max_steps=15,
            warmup_steps=5,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            log_interval=5,
            checkpoint_interval=20,
            val_interval=10,
            use_wandb=False,
        )
        output = _capture_train(cfg)
        rows = _parse_log_lines(output)

        # Find the row for step 10 where val_loss should be computed
        row_10 = next((r for r in rows if int(r["step"]) == 10), None)
        assert row_10 is not None, f"No log line for step=10. Lines: {rows}"

        val_loss_str = row_10.get("val_loss", "-")
        assert val_loss_str != "-", (
            f"val_loss was '-' at step=10, expected a finite float. Row: {row_10}"
        )
        val_loss = float(val_loss_str)
        import math
        assert math.isfinite(val_loss), f"val_loss is not finite: {val_loss}"

    def test_val_loss_close_to_train_loss(self, tmp_path):
        """Validation loss at step 10 should be close to training loss (|delta| < 2.0)."""
        cfg = TrainingConfig(
            shard_dir=_FIXTURE_SHARD_DIR,
            output_dir=str(tmp_path),
            val_shard=_VAL_SHARD,
            max_steps=15,
            warmup_steps=5,
            micro_batch_size=2,
            grad_accum_steps=2,
            seq_len=64,
            log_interval=5,
            checkpoint_interval=20,
            val_interval=10,
            use_wandb=False,
        )
        output = _capture_train(cfg)
        rows = _parse_log_lines(output)

        row_10 = next((r for r in rows if int(r["step"]) == 10), None)
        assert row_10 is not None

        train_loss = float(row_10["loss"])
        val_loss = float(row_10["val_loss"])

        assert abs(val_loss - train_loss) < 2.0, (
            f"|val_loss - train_loss| = |{val_loss:.4f} - {train_loss:.4f}| = "
            f"{abs(val_loss - train_loss):.4f} ≥ 2.0"
        )
