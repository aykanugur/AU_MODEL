#!/usr/bin/env python3
"""
SC-006 Spot-check: Token ID Range Verification
=================================================
Samples up to 100 random shard files from a shard directory and verifies
that all token IDs are in the valid range [0, VOCAB_SIZE).

Usage (standalone — run manually after full corpus build):
    python tests/check_token_range.py --shard-dir /content/drive/MyDrive/AUModel/data/

Covers: SC-006 — spot-check of 100 randomly selected shards confirms all
        token IDs are in [0, 64000).
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Constants (must match prepare_data.py)
# ---------------------------------------------------------------------------
VOCAB_SIZE = 64_000
MAX_SHARDS_TO_CHECK = 100


def check_shard(shard_path: str) -> tuple[bool, str]:
    """
    Load a shard and verify all token IDs are in [0, VOCAB_SIZE).

    Returns:
        (passed, message)
    """
    try:
        tokens = np.fromfile(shard_path, dtype=np.uint16)
    except Exception as exc:
        return False, f"FAIL  {shard_path}  — unreadable: {exc}"

    if len(tokens) == 0:
        return False, f"FAIL  {shard_path}  — empty file"

    oob_mask = tokens >= VOCAB_SIZE
    n_oob = int(oob_mask.sum())
    if n_oob > 0:
        bad_ids = tokens[oob_mask][:5].tolist()
        return False, (
            f"FAIL  {shard_path}  — "
            f"{n_oob} out-of-range IDs (first 5: {bad_ids})"
        )

    return True, f"PASS  {shard_path}  — {len(tokens):,} tokens in [0, {VOCAB_SIZE})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Spot-check token ID ranges in shard files (SC-006)."
    )
    parser.add_argument(
        "--shard-dir",
        required=True,
        help="Directory containing shard_NNNN.bin files.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=MAX_SHARDS_TO_CHECK,
        help=f"Number of shards to sample (default: {MAX_SHARDS_TO_CHECK}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.shard_dir, "shard_*.bin")
    all_shards = sorted(glob.glob(pattern))

    if not all_shards:
        print(f"[ERROR] No shard files found in: {args.shard_dir}")
        return 1

    print(f"Found {len(all_shards):,} shard files.")

    rng = random.Random(args.seed)
    sample = rng.sample(all_shards, min(args.max_shards, len(all_shards)))
    sample.sort()

    print(f"Checking {len(sample)} shards (seed={args.seed})...")
    print()

    passed = 0
    failed = 0
    for shard_path in sample:
        ok, msg = check_shard(shard_path)
        print(msg)
        if ok:
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} PASS / {failed} FAIL / {len(sample)} checked")
    if failed == 0:
        print("✓ SC-006 PASS — all sampled shards contain valid token IDs.")
        return 0
    else:
        print(f"✗ SC-006 FAIL — {failed} shard(s) contain out-of-range IDs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
