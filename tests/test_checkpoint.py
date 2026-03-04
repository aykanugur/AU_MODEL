"""Tests for save_checkpoint / rotate_checkpoints / load_latest_checkpoint.

Covers T010 (SC-001, SC-003):
    1. Logit round-trip after save+load at seq_len=64
    2. Rotation: save 11 stubs → assert 10 remain, oldest deleted
    3. Loader state field-by-field survival across save/load
"""
import pathlib
import time

import pytest
import torch

from model import AUModel, ModelConfig
from training.checkpoint import (
    load_latest_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
)
from training.trainer import InterleavedShardLoader, SourceState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEQ_LEN = 64
FIXTURE_SHARD = str(
    pathlib.Path(__file__).parent / "fixtures/data/wikipedia/shard_00000.bin"
)


@pytest.fixture()
def small_model() -> AUModel:
    """Return a tiny CPU AUModel for testing."""
    cfg = ModelConfig()
    return AUModel(cfg)


# ---------------------------------------------------------------------------
# Logit round-trip
# ---------------------------------------------------------------------------

class TestLogitRoundTrip:
    def test_logits_identical_after_checkpoint(self, small_model, tmp_path):
        """Logits from the original model must match logits after save+load."""
        model = small_model
        model.eval()

        # Reference forward pass
        tokens = torch.randint(0, 64_000, (1, SEQ_LEN))
        with torch.no_grad():
            logits_before, _ = model(tokens, tokens)

        # Save checkpoint
        ckpt_path = tmp_path / "step_000001.pt"
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "step": 1,
            "config": None,
            "loader_state": None,
            "best_val_loss": None,
            "created_at": time.time(),
        }
        save_checkpoint(ckpt_path, state)

        # Load and compare
        loaded = load_latest_checkpoint(str(tmp_path))
        assert loaded is not None

        model2 = AUModel(ModelConfig())
        model2.load_state_dict(loaded["model_state"])
        model2.eval()
        with torch.no_grad():
            logits_after, _ = model2(tokens, tokens)

        assert torch.allclose(logits_before, logits_after, atol=1e-6), (
            "Logits diverged after checkpoint round-trip"
        )

    def test_checkpoint_writes_tmp_then_renames(self, tmp_path):
        """A .tmp file must NOT be left behind after save_checkpoint."""
        path = tmp_path / "step_000001.pt"
        save_checkpoint(path, {"step": 1})
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# Rotate checkpoints
# ---------------------------------------------------------------------------

class TestRotateCheckpoints:
    def _write_stub(self, path: pathlib.Path, step: int) -> pathlib.Path:
        p = path / f"step_{step:06d}.pt"
        save_checkpoint(p, {"step": step})
        return p

    def test_11_saves_leaves_10(self, tmp_path):
        """After saving 11 stubs and rotating with keep=10, exactly 10 remain."""
        for s in range(1, 12):
            self._write_stub(tmp_path, s)
        rotate_checkpoints(tmp_path, keep=10)
        remaining = sorted(tmp_path.glob("step_*.pt"))
        assert len(remaining) == 10

    def test_oldest_deleted(self, tmp_path):
        """step_000001.pt must be deleted when 11 checkpoints exist and keep=10."""
        for s in range(1, 12):
            self._write_stub(tmp_path, s)
        rotate_checkpoints(tmp_path, keep=10)
        assert not (tmp_path / "step_000001.pt").exists()

    def test_newest_preserved(self, tmp_path):
        """step_000011.pt must survive rotation."""
        for s in range(1, 12):
            self._write_stub(tmp_path, s)
        rotate_checkpoints(tmp_path, keep=10)
        assert (tmp_path / "step_000011.pt").exists()

    def test_keep_more_than_existing_is_noop(self, tmp_path):
        for s in range(1, 4):
            self._write_stub(tmp_path, s)
        rotate_checkpoints(tmp_path, keep=10)
        assert len(list(tmp_path.glob("step_*.pt"))) == 3

    def test_empty_dir_is_noop(self, tmp_path):
        rotate_checkpoints(tmp_path, keep=10)  # must not raise


# ---------------------------------------------------------------------------
# Loader state round-trip
# ---------------------------------------------------------------------------

class TestLoaderStateRoundTrip:
    def test_state_dict_fields_survive_save_load(self, tmp_path):
        """SourceState fields (shard_idx, sample_idx, cycle_count) must round-trip."""
        loader = InterleavedShardLoader(
            shard_dir=str(pathlib.Path(FIXTURE_SHARD).parent.parent),
            seq_len=SEQ_LEN,
            source_weights={"wikipedia": 1.0},
        )

        # Advance the loader a bit so state is non-trivial
        dl_iter = iter(
            torch.utils.data.DataLoader(loader, batch_size=1, num_workers=0)
        )
        for _ in range(5):
            next(dl_iter)

        original_state = loader.state_dict()

        # Save and reload
        ckpt_path = tmp_path / "step_000001.pt"
        save_checkpoint(ckpt_path, {"step": 1, "loader_state": original_state,
                                    "model_state": {}, "optimizer_state": {},
                                    "config": None, "best_val_loss": None,
                                    "created_at": time.time()})

        loaded = load_latest_checkpoint(str(tmp_path))
        assert loaded is not None
        recovered_state = loaded["loader_state"]

        # Field-by-field comparison for the wikipedia source
        orig_src = original_state["wikipedia"]
        recov_src = recovered_state["wikipedia"]

        assert recov_src["shard_idx"] == orig_src["shard_idx"]
        assert recov_src["sample_idx"] == orig_src["sample_idx"]
        assert recov_src["cycle_count"] == orig_src["cycle_count"]
        assert recov_src["tokens_consumed"] == orig_src["tokens_consumed"]
        assert recov_src["shard_paths"] == orig_src["shard_paths"]
        assert recov_src["shuffled_paths"] == orig_src["shuffled_paths"]

    def test_load_state_dict_restores_position(self, tmp_path):
        """Loader restored from a state_dict must continue from the same sample."""
        fixture_dir = str(pathlib.Path(FIXTURE_SHARD).parent.parent)

        def _make_loader():
            return InterleavedShardLoader(
                shard_dir=fixture_dir,
                seq_len=SEQ_LEN,
                source_weights={"wikipedia": 1.0},
            )

        loader_a = _make_loader()
        dl_a = iter(torch.utils.data.DataLoader(loader_a, batch_size=1, num_workers=0))

        # Advance 3 samples, capture state
        for _ in range(3):
            next(dl_a)
        saved_state = loader_a.state_dict()
        sample_4_a, _ = next(dl_a)

        # Create fresh loader, restore state, check next sample matches
        loader_b = _make_loader()
        loader_b.load_state_dict(saved_state)
        dl_b = iter(torch.utils.data.DataLoader(loader_b, batch_size=1, num_workers=0))
        sample_4_b, _ = next(dl_b)

        assert torch.equal(sample_4_a, sample_4_b), (
            "Sample after resume does not match original sequence"
        )


# ---------------------------------------------------------------------------
# load_latest_checkpoint edge cases
# ---------------------------------------------------------------------------

class TestLoadLatestCheckpoint:
    def test_returns_none_for_missing_dir(self, tmp_path):
        result = load_latest_checkpoint(str(tmp_path / "nonexistent"))
        assert result is None

    def test_returns_none_for_empty_dir(self, tmp_path):
        result = load_latest_checkpoint(str(tmp_path))
        assert result is None

    def test_loads_highest_step(self, tmp_path):
        for step in [5, 10, 3]:
            save_checkpoint(tmp_path / f"step_{step:06d}.pt", {"step": step})
        loaded = load_latest_checkpoint(str(tmp_path))
        assert loaded is not None
        assert loaded["step"] == 10
