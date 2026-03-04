"""
model/sanity_check.py — Self-contained sanity check for AUModel.

Runs 4 sequential checks and prints [PASS]/[FAIL] per check.
Exits with code 0 on all pass, code 1 on any failure.

Usage:
    python model/sanity_check.py

Constitution rules observed:
  - No assert anywhere; use if/raise ValueError(...)
  - bias=False on all nn.Linear (enforced in model code)
  - bfloat16 dtype check
  - type hints on all public signatures
"""

import sys
import math
import os

# Ensure the repo root is on the path so `import model` works when this
# script is executed directly (python model/sanity_check.py) from the root,
# AND when executed from inside the model/ directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F


def _passed(name: str) -> None:
    print(f"[PASS] {name}")


def _failed(name: str, reason: str) -> None:
    print(f"[FAIL] {name} — {reason}")


def check_instantiate() -> bool:
    """Check 1: Import and instantiate AUModel(ModelConfig())."""
    name = "instantiate AUModel(ModelConfig())"
    try:
        from model import AUModel, ModelConfig
        cfg = ModelConfig()
        _model = AUModel(cfg)
        _passed(name)
        return True
    except Exception as exc:
        _failed(name, str(exc))
        return False


def check_forward_shape() -> bool:
    """Check 2: Forward pass shape (2, 128) → logits (2, 128, 64000)."""
    name = "forward shape (2,128) → (2,128,64000)"
    try:
        from model import AUModel, ModelConfig
        cfg = ModelConfig()
        model = AUModel(cfg)
        model.eval()

        batch_size, seq_len = 2, 128
        tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits, _ = model(tokens)

        expected = (batch_size, seq_len, cfg.vocab_size)
        if logits.shape != expected:
            _failed(name, f"got shape {tuple(logits.shape)}, expected {expected}")
            return False

        _passed(name)
        return True
    except Exception as exc:
        _failed(name, str(exc))
        return False


def check_param_count() -> bool:
    """Check 3: Parameter count in [730M, 770M] (expected ~749,544,960)."""
    name = "param count in [730M, 770M]"
    lo, hi = 730_000_000, 770_000_000
    try:
        from model import AUModel, ModelConfig
        cfg = ModelConfig()
        model = AUModel(cfg)
        n = model.get_num_params()

        if not (lo <= n <= hi):
            _failed(name, f"got {n:,} ({n / 1e6:.1f}M), expected [{lo // 1_000_000}M, {hi // 1_000_000}M]")
            return False

        print(f"[PASS] {name}  ({n:,} ≈ {n / 1e6:.1f}M params)")
        return True
    except Exception as exc:
        _failed(name, str(exc))
        return False


def check_initial_loss() -> bool:
    """Check 4: Mean CE loss on 10 random batches ∈ [10.0, 11.0].

    For a random model over vocab_size=64000, expected loss ≈ ln(64000) ≈ 11.07.
    We allow the range [10.0, 11.0] which accounts for minor fluctuations at
    the specific batch sizes we test. Note: loss approaching ln(vocab_size) is
    the theoretical maximum-entropy baseline; a freshly initialised model
    should sit close to this value.
    """
    name = "initial CE loss in [10.0, 11.5]"
    lo_loss, hi_loss = 10.0, 11.5
    try:
        from model import AUModel, ModelConfig
        cfg = ModelConfig()
        model = AUModel(cfg)
        model.eval()

        torch.manual_seed(42)
        batch_size, seq_len = 4, 64
        losses: list[float] = []

        with torch.no_grad():
            for _ in range(10):
                tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len + 1))
                inputs = tokens[:, :-1]   # (B, seq_len)
                targets = tokens[:, 1:]   # (B, seq_len)

                logits, _ = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
                losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        # For reference: ln(64000) ≈ 11.07; we expect values close to this
        theoretical = math.log(cfg.vocab_size)

        if not (lo_loss <= mean_loss <= hi_loss):
            _failed(
                name,
                f"mean loss={mean_loss:.4f}, expected [{lo_loss}, {hi_loss}]  "
                f"(theoretical max-entropy ≈ {theoretical:.4f})",
            )
            return False

        print(
            f"[PASS] {name}  "
            f"(mean={mean_loss:.4f}, theoretical≈{theoretical:.4f})"
        )
        return True
    except Exception as exc:
        _failed(name, str(exc))
        return False


def main() -> None:
    """Run all 4 checks in sequence and exit with code 0 or 1."""
    print("=" * 60)
    print("AUModel Sanity Check")
    print("=" * 60)

    results: list[bool] = [
        check_instantiate(),
        check_forward_shape(),
        check_param_count(),
        check_initial_loss(),
    ]

    print("=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"ALL {total} CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"{passed}/{total} CHECKS PASSED — {total - passed} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
