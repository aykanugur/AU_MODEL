"""
Turkish Pretraining Data Pipeline
==================================
Downloads, cleans, deduplicates, tokenises and shards Turkish text from
HuggingFace sources (Wikipedia, OSCAR, mC4, CC-100) into flat uint16 binary
shard files for pretraining.

Usage:
    python scripts/prepare_data.py --source wikipedia --output /tmp/shards/
    python scripts/prepare_data.py --source all

Constitution-locked constants:
    BOS_ID = 2   (<s>)
    EOS_ID = 3   (</s>)
    VOCAB_SIZE = 64_000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import time
import unicodedata
from datetime import datetime, timezone
from multiprocessing import Pool
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants (T002) — constitution-locked values
# ---------------------------------------------------------------------------
BOS_ID: int = 2                   # <s>  — per constitution
EOS_ID: int = 3                   # </s> — per constitution
TOKENS_PER_SHARD: int = 500_000_000  # ~1 GB as uint16
MIN_DOC_TOKENS: int = 100
BATCH_SIZE: int = 1_000           # documents per pool.map call
BLOOM_CAPACITY: int = 100_000_000
BLOOM_ERROR: float = 0.01
VOCAB_SIZE: int = 64_000
MAX_WORKERS: int = min((os.cpu_count() or 2) // 2, 32)

# ---------------------------------------------------------------------------
# Source config (T003) — pinned dataset versions
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict] = {
    "wikipedia": {
        "hf_id": "wikimedia/wikipedia",
        "config": "20231101.tr",
        "split": "train",
        "text_field": "text",
        "requires_auth": False,
    },
    "oscar": {
        "hf_id": "oscar-corpus/OSCAR-2301",
        "config": "tr",
        "split": "train",
        "text_field": "content",   # OSCAR uses "content", not "text"
        "requires_auth": True,
    },
    "mc4": {
        "hf_id": "allenai/c4",
        "config": "tr",
        "split": "train",
        "text_field": "text",
        "requires_auth": False,
    },
    "cc100": {
        "hf_id": "cc100",
        "config": "tr",
        "split": "train",
        "text_field": "text",
        "requires_auth": False,
    },
}

# ---------------------------------------------------------------------------
# HF token loading (T004)
# ---------------------------------------------------------------------------

def load_hf_token() -> str | None:
    """Load HF_TOKEN from .env file. Returns None if not set."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("HF_TOKEN")


def require_hf_token(source_name: str) -> str:
    """Return HF token or raise ValueError if source requires auth."""
    token = load_hf_token()
    if token is None:
        raise ValueError(
            f"Source '{source_name}' requires authentication but HF_TOKEN is "
            "not set. Add HF_TOKEN=<your_token> to .env or set the environment "
            "variable before running."
        )
    return token


# ---------------------------------------------------------------------------
# CLI arg parsing (T005)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Turkish pretraining corpus as uint16 binary shards."
    )
    parser.add_argument(
        "--source",
        choices=["wikipedia", "oscar", "mc4", "cc100", "all"],
        required=True,
        help="Dataset source to process.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/content/drive/MyDrive/AUModel/data/",
        help="Output directory for shard files and manifest.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer/turkish_bpe.model",
        help="Path to trained SentencePiece .model file.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Document cleaning (T006)
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_document(text: str) -> str | None:
    """
    Clean a raw document string.

    Steps:
      1. Strip HTML tags.
      2. Normalise unicode to NFC.
      3. Collapse whitespace.
      4. Return None for empty results (short-circuit before tokenization).

    Args:
        text: Raw document text.

    Returns:
        Cleaned text, or None if result is empty.
    """
    if not text:
        return None
    text = _HTML_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    if not text:
        return None
    return text


# ---------------------------------------------------------------------------
# Bloom filter init / checkpoint (T007)
# ---------------------------------------------------------------------------

def init_bloom(output_dir: str):  # -> BloomFilter
    """Load Bloom filter from checkpoint or create a fresh one."""
    from pybloom_live import BloomFilter

    bloom_path = os.path.join(output_dir, "bloom.pkl")
    if os.path.exists(bloom_path):
        print(f"[resume] Loading Bloom filter from {bloom_path}")
        with open(bloom_path, "rb") as f:
            return pickle.load(f)
    return BloomFilter(capacity=BLOOM_CAPACITY, error_rate=BLOOM_ERROR)


def save_bloom(bloom, path: str) -> None:
    """Pickle Bloom filter to disk."""
    with open(path, "wb") as f:
        pickle.dump(bloom, f)


# ---------------------------------------------------------------------------
# Manifest read/write (T008)
# ---------------------------------------------------------------------------

_MANIFEST_VERSION = 1
_MANIFEST_FILENAME = "shards_manifest.json"


def load_manifest(output_dir: str) -> dict:
    """
    Load shards_manifest.json from output_dir.
    Returns an empty manifest skeleton if file does not exist.
    """
    path = os.path.join(output_dir, _MANIFEST_FILENAME)
    if not os.path.exists(path):
        return {
            "version": _MANIFEST_VERSION,
            "bloom_filter_path": os.path.join(output_dir, "bloom.pkl"),
            "shards": [],
            "total_tokens": 0,
            "source_totals": {},
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict, output_dir: str) -> None:
    """Write manifest atomically (write-then-rename)."""
    path = os.path.join(output_dir, _MANIFEST_FILENAME)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Multiprocessing tokenizer worker (T009 + T025)
# ---------------------------------------------------------------------------

# Module-level state for spawned workers
_worker_sp = None
_worker_model_path: str | None = None


def _init_worker(model_path: str) -> None:
    """Initialise SentencePiece model once per worker process."""
    global _worker_sp, _worker_model_path
    import sentencepiece as spm
    _worker_sp = spm.SentencePieceProcessor()
    _worker_sp.Load(model_path)
    _worker_model_path = model_path


def _tokenize_doc(text: str) -> list[int]:
    """
    Tokenize a single document in a worker process.
    Returns [BOS_ID] + token_ids + [EOS_ID].

    Inner token-range guard (T025): raises ValueError early if any ID is OOB.
    """
    ids = _worker_sp.EncodeAsIds(text)
    # Inner guard — early detection per worker (outer batch check is in tokenize_batch)
    for tok_id in ids:
        if tok_id >= VOCAB_SIZE:
            raise ValueError(
                f"Token ID {tok_id} out of range [0, {VOCAB_SIZE}) in worker"
            )
    return [BOS_ID] + ids + [EOS_ID]


# ---------------------------------------------------------------------------
# Batch tokenisation (T010)
# ---------------------------------------------------------------------------

def tokenize_batch(texts: list[str], pool: Pool) -> list[list[int]]:
    """
    Tokenise a batch of cleaned documents in parallel.

    Filters documents shorter than MIN_DOC_TOKENS (after BOS/EOS).
    Validates all token IDs are in [0, VOCAB_SIZE).

    Args:
        texts: Pre-cleaned document strings.
        pool: Active multiprocessing.Pool with _init_worker initialised.

    Returns:
        List of token-id lists (filtered).

    Raises:
        ValueError: If any token ID is outside [0, VOCAB_SIZE).
    """
    results: list[list[int]] = pool.map(_tokenize_doc, texts)
    filtered: list[list[int]] = []
    for ids in results:
        if len(ids) < MIN_DOC_TOKENS:
            continue
        # Outer batch-level validation
        for tok_id in ids:
            if tok_id >= VOCAB_SIZE:
                raise ValueError(
                    f"Token ID {tok_id} out of range [0, {VOCAB_SIZE}) — "
                    "check tokenizer vocab_size setting."
                )
        filtered.append(ids)
    return filtered


# ---------------------------------------------------------------------------
# Drive space check (T018)
# ---------------------------------------------------------------------------

def check_drive_space(output_dir: str, min_gb: float = 50.0) -> None:
    """
    Verify sufficient free space before writing a shard.

    Args:
        output_dir: Directory to check.
        min_gb: Minimum free gigabytes required (default 50 GB).

    Raises:
        SystemExit(1): If free space is below threshold.
    """
    free = shutil.disk_usage(output_dir).free
    free_gb = free / 1e9
    if free_gb < min_gb:
        print(
            f"[WARNING] Only {free_gb:.1f} GB free in {output_dir}. "
            f"Need ≥{min_gb:.0f} GB. Stopping to avoid corruption."
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Retry wrapper for streaming (T017)
# ---------------------------------------------------------------------------

def _stream_source_once(source_name: str, hf_token: str | None) -> Iterator[str]:
    """Single-attempt stream over one HF source. Yields cleaned texts."""
    from datasets import load_dataset

    cfg = SOURCES[source_name]
    ds = load_dataset(
        cfg["hf_id"],
        cfg["config"],
        split=cfg["split"],
        streaming=True,
        token=hf_token,
        trust_remote_code=True,
    )
    field = cfg["text_field"]
    for record in ds:
        yield record[field]


def stream_source(source_name: str, hf_token: str | None) -> Iterator[str]:
    """
    Stream raw text from a HuggingFace source with retry logic (T012 + T017).

    On ConnectionError or DatasetGenerationError retries up to 3 times with
    exponential backoff (2^attempt seconds). After 3 failures, skips source.

    Args:
        source_name: One of "wikipedia", "oscar", "mc4", "cc100".
        hf_token: HuggingFace API token (may be None for public sources).

    Yields:
        Raw document strings from the source.
    """
    import requests

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            yield from _stream_source_once(source_name, hf_token)
            return
        except (requests.exceptions.ConnectionError, Exception) as exc:
            # Catch DatasetGenerationError by name to avoid import at module level
            exc_type = type(exc).__name__
            if exc_type not in ("ConnectionError", "DatasetGenerationError") and \
               not isinstance(exc, requests.exceptions.ConnectionError):
                raise
            if attempt < max_retries:
                wait = 2 ** attempt
                print(
                    f"[WARNING] Network error on {source_name} "
                    f"(attempt {attempt + 1}/{max_retries}): {exc}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                print(
                    f"[WARNING] {source_name} failed after {max_retries} retries. "
                    "Skipping source."
                )
                return


# ---------------------------------------------------------------------------
# Shard writer (T011 + T018 + T019)
# ---------------------------------------------------------------------------

class ShardWriter:
    """
    Buffers tokenised uint16 token IDs and flushes complete shard files.

    Each flush:
      1. Checks available Drive space (T018).
      2. Writes shard_NNNN.bin as a flat uint16 binary file.
      3. Validates the written file (size > 0, round-trip read).
      4. Updates the manifest and saves the Bloom filter checkpoint.
      5. Tracks per-source token counts (T019).
    """

    def __init__(self) -> None:
        self._buffer = np.empty(TOKENS_PER_SHARD, dtype=np.uint16)
        self._pos: int = 0
        self._source_counts: dict[str, int] = {}

    @property
    def remaining(self) -> int:
        """Slots left in the current buffer."""
        return TOKENS_PER_SHARD - self._pos

    @property
    def has_data(self) -> bool:
        return self._pos > 0

    def write_tokens(self, ids: list[int], source_name: str) -> bool:
        """
        Write token IDs into the buffer. Accumulates per-source counts.

        Returns True if the buffer is now full and should be flushed.
        """
        n = len(ids)
        if n > self.remaining:
            # Fill to capacity; caller flushes then calls again with remainder
            take = self.remaining
            self._buffer[self._pos : self._pos + take] = ids[:take]
            self._pos += take
            self._source_counts[source_name] = (
                self._source_counts.get(source_name, 0) + take
            )
            return True  # full
        self._buffer[self._pos : self._pos + n] = ids
        self._pos += n
        self._source_counts[source_name] = (
            self._source_counts.get(source_name, 0) + n
        )
        return self._pos >= TOKENS_PER_SHARD

    def flush(
        self,
        output_dir: str,
        shard_idx: int,
        manifest: dict,
        bloom,
        start_time: float,
    ) -> tuple[int, str]:
        """
        Write the buffered tokens to disk as shard_NNNN.bin.

        Args:
            output_dir: Directory to write into.
            shard_idx: Zero-based shard index.
            manifest: Manifest dict (mutated in-place).
            bloom: Active Bloom filter (checkpointed).
            start_time: Pipeline start time for elapsed display.

        Returns:
            (tokens_written, shard_path)

        Raises:
            RuntimeError: If shard validation fails (file is deleted first).
        """
        check_drive_space(output_dir)

        tokens_written = self._pos
        shard_filename = f"shard_{shard_idx:04d}.bin"
        shard_path = os.path.join(output_dir, shard_filename)

        # Write
        data_to_write = self._buffer[:tokens_written]
        data_to_write.tofile(shard_path)

        # Validate (T010 / FR-010)
        file_size = os.path.getsize(shard_path)
        if file_size == 0:
            os.remove(shard_path)
            raise RuntimeError(f"Shard {shard_path} was written with 0 bytes — deleted.")
        try:
            check = np.fromfile(shard_path, dtype=np.uint16)
            if len(check) != tokens_written:
                raise RuntimeError(
                    f"Shard validation failed: expected {tokens_written} tokens, "
                    f"got {len(check)}."
                )
        except Exception as exc:
            if os.path.exists(shard_path):
                os.remove(shard_path)
            raise RuntimeError(f"Shard {shard_path} failed validation: {exc}") from exc

        # Update manifest (T008 / T019)
        bloom_path = os.path.join(output_dir, "bloom.pkl")
        save_bloom(bloom, bloom_path)
        manifest["bloom_filter_path"] = bloom_path

        shard_record = {
            "filename": shard_filename,
            "token_count": tokens_written,
            "sources": dict(self._source_counts),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest["shards"].append(shard_record)
        manifest["total_tokens"] = manifest.get("total_tokens", 0) + tokens_written

        # Accumulate per-source totals (T019)
        for src, cnt in self._source_counts.items():
            manifest["source_totals"][src] = (
                manifest["source_totals"].get(src, 0) + cnt
            )

        save_manifest(manifest, output_dir)

        # Progress line (T014)
        elapsed = time.time() - start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        gb_written = file_size / 1e9
        print(
            f"[Shard {shard_idx:04d}/----] "
            f"{tokens_written / 1e6:.1f}M tokens | "
            f"{gb_written:.2f} GB written | "
            f"elapsed {h:02d}:{m:02d}:{s:02d}"
        )

        # Reset buffer for next shard
        self._pos = 0
        self._source_counts = {}

        return tokens_written, shard_path


# ---------------------------------------------------------------------------
# Main pipeline loop (T013 + T014 + T015 + T016)
# ---------------------------------------------------------------------------

def run_pipeline(
    sources: list[str],
    output_dir: str,
    tokenizer_path: str,
) -> None:
    """
    Main data pipeline.

    For each source:
      - Streams documents via HF datasets in streaming mode.
      - Cleans + deduplicates with Bloom filter.
      - Tokenises in parallel batches via multiprocessing.Pool(spawn).
      - Writes into uint16 binary shard files via ShardWriter.

    Supports resume:
      - Reads shards_manifest.json to skip already-written shards.
      - Restores Bloom filter from bloom.pkl checkpoint.

    Args:
        sources: List of source names to process.
        output_dir: Destination directory for shards and manifest.
        tokenizer_path: Path to turkish_bpe.model.
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}. "
            "Run Epic 1 (train_tokenizer.py) first."
        )

    os.makedirs(output_dir, exist_ok=True)

    hf_token = load_hf_token()

    # Validate auth requirements upfront
    for src in sources:
        if SOURCES[src]["requires_auth"]:
            hf_token = require_hf_token(src)
            break

    # Load (or create) manifest and Bloom filter
    manifest = load_manifest(output_dir)
    completed_shards = {s["filename"] for s in manifest["shards"]}
    shard_idx = len(completed_shards)

    bloom = init_bloom(output_dir)

    print(f"[init] Output dir: {output_dir}")
    print(f"[init] Tokenizer:  {tokenizer_path}")
    print(f"[init] Sources:    {sources}")
    print(f"[init] Resume:     {shard_idx} shards already complete")
    print(f"[init] Workers:    {MAX_WORKERS}")

    writer = ShardWriter()
    start_time = time.time()
    total_docs_processed = 0

    ctx = __import__("multiprocessing").get_context("spawn")
    with ctx.Pool(
        processes=MAX_WORKERS,
        initializer=_init_worker,
        initargs=(tokenizer_path,),
    ) as pool:
        for source_name in sources:
            source_start_tokens = manifest["source_totals"].get(source_name, 0)
            source_docs = 0
            source_tokens = 0
            batch_texts: list[str] = []

            print(f"\n[{source_name}] Starting stream...")

            for raw_text in stream_source(source_name, hf_token):
                cleaned = clean_document(raw_text)
                if cleaned is None:
                    continue

                # Deduplication via Bloom filter (FR-005)
                doc_hash = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
                if doc_hash in bloom:
                    continue
                bloom.add(doc_hash)

                batch_texts.append(cleaned)

                if len(batch_texts) < BATCH_SIZE:
                    continue

                # Process a full batch
                token_lists = tokenize_batch(batch_texts, pool)
                batch_texts = []

                for ids in token_lists:
                    # Handle shard boundary (ids may span two shards)
                    remaining_ids = ids
                    while remaining_ids:
                        take = min(len(remaining_ids), writer.remaining)
                        full = writer.write_tokens(remaining_ids[:take], source_name)
                        remaining_ids = remaining_ids[take:]
                        if full and writer._pos == TOKENS_PER_SHARD:
                            shard_idx += 1
                            writer.flush(output_dir, shard_idx - 1, manifest, bloom, start_time)

                    source_tokens += len(ids)
                    source_docs += 1
                    total_docs_processed += 1

                # Per-10k-docs progress (T014)
                if total_docs_processed % 10_000 == 0:
                    print(
                        f"[{source_name}] {total_docs_processed:,} docs | "
                        f"{source_tokens / 1e6:.1f}M tokens"
                    )

            # Flush remaining batch for this source
            if batch_texts:
                token_lists = tokenize_batch(batch_texts, pool)
                batch_texts = []
                for ids in token_lists:
                    remaining_ids = ids
                    while remaining_ids:
                        take = min(len(remaining_ids), writer.remaining)
                        full = writer.write_tokens(remaining_ids[:take], source_name)
                        remaining_ids = remaining_ids[take:]
                        if full and writer._pos == TOKENS_PER_SHARD:
                            shard_idx += 1
                            writer.flush(output_dir, shard_idx - 1, manifest, bloom, start_time)
                    source_tokens += len(ids)
                    source_docs += 1
                    total_docs_processed += 1

            # Zero-output warning (M5)
            if source_tokens == 0:
                print(
                    f"[WARNING] Source {source_name} produced 0 tokens — "
                    "all documents filtered or source empty."
                )
            else:
                print(
                    f"[{source_name}] Done: {source_docs:,} docs | "
                    f"{source_tokens / 1e6:.1f}M tokens"
                )

        # Flush final partial shard (T013)
        if writer.has_data:
            writer.flush(output_dir, shard_idx, manifest, bloom, start_time)
            shard_idx += 1

    # Final summary (T015)
    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    total_tokens = manifest["total_tokens"]
    total_shards = len(manifest["shards"])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total shards : {total_shards:,}")
    print(f"Total tokens : {total_tokens:,}  ({total_tokens / 1e9:.2f}B)")
    print(f"Elapsed time : {h:02d}:{m:02d}:{s:02d}")
    print("\nPer-source breakdown:")
    print(f"  {'Source':<15}  {'Tokens':>15}  {'%':>6}")
    print(f"  {'-'*15}  {'-'*15}  {'-'*6}")
    for src, cnt in manifest["source_totals"].items():
        pct = 100.0 * cnt / total_tokens if total_tokens > 0 else 0.0
        print(f"  {src:<15}  {cnt:>15,}  {pct:>5.1f}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.source == "all":
        selected_sources = ["wikipedia", "oscar", "mc4", "cc100"]
    else:
        selected_sources = [args.source]

    run_pipeline(
        sources=selected_sources,
        output_dir=args.output,
        tokenizer_path=args.tokenizer,
    )
