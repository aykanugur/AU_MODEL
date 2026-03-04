"""
Integration and resume tests for scripts/prepare_data.py :: run_pipeline

Covers:
  T029 — Integration: fake 1000-doc stream → shard written, manifest valid
  T030 — Resume: interrupt mid-run → restart → same corpus as uninterrupted run

These tests use a monkey-patched stream_source so no HuggingFace network
calls are made. A real SentencePiece model is needed; if not present the
tests are automatically skipped.
"""

import hashlib
import json
import os
import sys
import tempfile
from typing import Iterator
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tokenizer", "turkish_bpe.model"
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(TOKENIZER_PATH),
    reason="tokenizer/turkish_bpe.model not found — run Epic 1 first",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 200-word Turkish sentence repeated to generate enough tokens
_SAMPLE_TEXT = (
    "Türkiye, Asya ve Avrupa kıtaları arasında yer alan, iki kıtayı birbirine "
    "bağlayan stratejik konumuyla önemli bir ülkedir. Başkenti Ankara olup, "
    "en büyük şehri ve ekonomik merkezi İstanbul'dur. Türkçe resmi dil olarak "
    "konuşulmakta olup, nüfusu 85 milyonu aşmaktadır. Tarihî ve kültürel "
    "zenginlikleriyle dünyada önemli bir yere sahip olan ülke, Cumhuriyet "
    "rejimi ile yönetilmektedir. "
)


def _make_fake_docs(n: int) -> list[str]:
    """Return n distinct fake Turkish documents, each long enough to tokenize."""
    docs = []
    for i in range(n):
        docs.append(f"[DOC {i}] " + _SAMPLE_TEXT * 3)
    return docs


def _fake_stream(docs: list[str]):
    """Return a generator that yields document strings."""
    def _gen(source_name: str, hf_token) -> Iterator[str]:
        yield from docs
    return _gen


def _shard_checksum(path: str) -> str:
    """MD5 of a shard file for determinism comparison."""
    tokens = np.fromfile(path, dtype=np.uint16)
    return hashlib.md5(tokens.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# T029 — Integration test
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_wikipedia_produces_shard_and_valid_manifest(self, tmp_path):
        """
        Fake 1000-doc stream → run_pipeline → ≥1 shard written, manifest valid.
        Covers plan.md task 3.4 / FR-007, FR-008.
        """
        from scripts.prepare_data import run_pipeline

        docs = _make_fake_docs(200)   # 200 docs — enough for at least some tokens

        with mock.patch("scripts.prepare_data.stream_source", side_effect=_fake_stream(docs)):
            run_pipeline(
                sources=["wikipedia"],
                output_dir=str(tmp_path),
                tokenizer_path=TOKENIZER_PATH,
            )

        # At least one shard or a partial shard must exist
        bin_files = sorted(tmp_path.glob("shard_*.bin"))
        assert len(bin_files) >= 1, "No shard files written."

        # Manifest must exist and be valid JSON
        manifest_path = tmp_path / "shards_manifest.json"
        assert manifest_path.exists(), "shards_manifest.json not written."

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["version"] == 1
        assert manifest["total_tokens"] > 0, "total_tokens is 0."
        assert len(manifest["shards"]) >= 1
        assert "wikipedia" in manifest["source_totals"]
        assert manifest["source_totals"]["wikipedia"] > 0

    def test_shard_is_readable_as_uint16(self, tmp_path):
        """Written shards must be loadable as flat uint16 arrays."""
        from scripts.prepare_data import run_pipeline

        docs = _make_fake_docs(200)

        with mock.patch("scripts.prepare_data.stream_source", side_effect=_fake_stream(docs)):
            run_pipeline(
                sources=["wikipedia"],
                output_dir=str(tmp_path),
                tokenizer_path=TOKENIZER_PATH,
            )

        for shard_path in tmp_path.glob("shard_*.bin"):
            tokens = np.fromfile(str(shard_path), dtype=np.uint16)
            assert len(tokens) > 0
            assert tokens.dtype == np.uint16

    def test_zero_output_source_does_not_crash(self, tmp_path, capsys):
        """If all docs are filtered, pipeline prints warning and exits cleanly."""
        from scripts.prepare_data import run_pipeline

        # All-HTML docs will be stripped to empty → min-token filter → 0 tokens
        empty_docs = ["<html></html>"] * 50

        with mock.patch(
            "scripts.prepare_data.stream_source",
            side_effect=_fake_stream(empty_docs),
        ):
            run_pipeline(
                sources=["wikipedia"],
                output_dir=str(tmp_path),
                tokenizer_path=TOKENIZER_PATH,
            )

        captured = capsys.readouterr()
        assert "WARNING" in captured.out and "0 tokens" in captured.out


# ---------------------------------------------------------------------------
# T030 — Resume test (SC-005)
# ---------------------------------------------------------------------------

class TestPipelineResume:
    def test_resume_produces_same_shard_as_uninterrupted(self, tmp_path):
        """
        SC-005: Restarting after interruption produces the same final corpus
        as an uninterrupted run.

        Strategy:
          1. Run pipeline with 300 docs → completes some shards.
          2. Inject an exception after first successful flush (simulates disconnect).
          3. Restart pipeline with same 300 docs.
          4. Compare shard checksums — must be identical.
        """
        from scripts.prepare_data import run_pipeline, ShardWriter

        docs = _make_fake_docs(300)
        out_interrupted = tmp_path / "interrupted"
        out_clean = tmp_path / "clean"
        out_interrupted.mkdir()
        out_clean.mkdir()

        # --- Reference: clean run ---
        with mock.patch(
            "scripts.prepare_data.stream_source", side_effect=_fake_stream(docs)
        ):
            run_pipeline(
                sources=["wikipedia"],
                output_dir=str(out_clean),
                tokenizer_path=TOKENIZER_PATH,
            )

        # --- Interrupted run: inject exception after first flush ---
        flush_count = {"n": 0}
        original_flush = ShardWriter.flush

        def patched_flush(self, output_dir, shard_idx, manifest, bloom, start_time):
            result = original_flush(self, output_dir, shard_idx, manifest, bloom, start_time)
            flush_count["n"] += 1
            if flush_count["n"] == 1:
                raise RuntimeError("Simulated Colab disconnect after first shard!")
            return result

        try:
            with mock.patch.object(ShardWriter, "flush", patched_flush), \
                 mock.patch("scripts.prepare_data.stream_source", side_effect=_fake_stream(docs)):
                run_pipeline(
                    sources=["wikipedia"],
                    output_dir=str(out_interrupted),
                    tokenizer_path=TOKENIZER_PATH,
                )
        except RuntimeError as exc:
            if "Simulated Colab disconnect" not in str(exc):
                raise

        # --- Resume run ---
        with mock.patch(
            "scripts.prepare_data.stream_source", side_effect=_fake_stream(docs)
        ):
            run_pipeline(
                sources=["wikipedia"],
                output_dir=str(out_interrupted),
                tokenizer_path=TOKENIZER_PATH,
            )

        # --- Compare shards ---
        clean_shards = sorted(out_clean.glob("shard_*.bin"))
        resumed_shards = sorted(out_interrupted.glob("shard_*.bin"))

        assert len(clean_shards) > 0, "Clean run produced no shards."
        assert len(resumed_shards) > 0, "Resumed run produced no shards."
        assert len(clean_shards) == len(resumed_shards), (
            f"Shard count mismatch: clean={len(clean_shards)}, "
            f"resumed={len(resumed_shards)}"
        )

        for c, r in zip(clean_shards, resumed_shards):
            assert _shard_checksum(str(c)) == _shard_checksum(str(r)), (
                f"Shard checksum mismatch: {c.name} vs {r.name}"
            )
