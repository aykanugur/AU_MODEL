"""
Unit tests for scripts/prepare_data.py utility functions.

Covers:
  T026 — clean_document(): HTML stripping, NFC normalisation, empty guard
  T027 — Bloom filter: duplicate detection, pickle checkpoint round-trip
"""

import hashlib
import os
import pickle
import sys
import tempfile

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.prepare_data import clean_document, BLOOM_CAPACITY, BLOOM_ERROR


# ===========================================================================
# T026 — clean_document() tests
# ===========================================================================

class TestCleanDocument:
    def test_strips_html_tags(self):
        result = clean_document("<p>Hello <b>world</b></p>")
        assert result == "Hello world"

    def test_strips_complex_html(self):
        result = clean_document('<a href="http://example.com">link</a> text')
        assert "link" in result
        assert "<a" not in result
        assert "href" not in result

    def test_nfc_normalization(self):
        # NFD composed character (e + combining acute) → NFC single codepoint
        nfd_text = "cafe\u0301"          # e + combining acute accent
        nfc_text = "caf\u00e9"           # é as single codepoint
        result = clean_document(nfd_text)
        assert result == nfc_text

    def test_returns_none_for_empty_string(self):
        assert clean_document("") is None

    def test_returns_none_for_whitespace_only(self):
        assert clean_document("   \n\t  ") is None

    def test_returns_none_for_html_only(self):
        result = clean_document("<div><span></span></div>")
        # After stripping tags and collapsing whitespace, result is empty
        assert result is None

    def test_collapses_whitespace(self):
        result = clean_document("hello   world\n\nfoo")
        assert "  " not in result
        assert result == "hello world foo"

    def test_returns_none_for_none_like_empty(self):
        # None input should be handled gracefully
        assert clean_document("") is None

    def test_plain_text_unchanged(self):
        text = "Türkçe metin örneği."
        result = clean_document(text)
        assert result == text

    def test_mixed_html_and_text(self):
        result = clean_document("<h1>Başlık</h1><p>İçerik buraya.</p>")
        assert "Başlık" in result
        assert "İçerik buraya." in result
        assert "<h1>" not in result
        assert "<p>" not in result


# ===========================================================================
# T027 — Bloom filter tests
# ===========================================================================

class TestBloomFilter:
    def _make_bloom(self):
        from pybloom_live import BloomFilter
        return BloomFilter(capacity=1_000, error_rate=0.01)

    def test_adds_and_detects(self):
        bloom = self._make_bloom()
        text = "Türkçe bir cümle örneği."
        doc_hash = hashlib.md5(text.encode()).hexdigest()
        bloom.add(doc_hash)
        assert doc_hash in bloom

    def test_duplicate_detected_on_second_call(self):
        bloom = self._make_bloom()
        text = "Bir cümle."
        doc_hash = hashlib.md5(text.encode()).hexdigest()

        assert doc_hash not in bloom   # first time: not duplicate
        bloom.add(doc_hash)
        assert doc_hash in bloom       # second time: duplicate detected

    def test_different_docs_not_duplicate(self):
        bloom = self._make_bloom()
        h1 = hashlib.md5("cümle bir".encode()).hexdigest()
        h2 = hashlib.md5("cümle iki".encode()).hexdigest()
        bloom.add(h1)
        assert h2 not in bloom

    def test_pickle_checkpoint_round_trip(self):
        bloom = self._make_bloom()
        text = "Kayıt edilecek belge."
        doc_hash = hashlib.md5(text.encode()).hexdigest()
        bloom.add(doc_hash)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name

        try:
            # Save
            with open(tmp_path, "wb") as f:
                pickle.dump(bloom, f)

            # Load
            with open(tmp_path, "rb") as f:
                restored = pickle.load(f)

            # Hash must still be present after restore
            assert doc_hash in restored

            # A hash not in original must not appear in restored
            other_hash = hashlib.md5("başka bir şey".encode()).hexdigest()
            assert other_hash not in restored
        finally:
            os.unlink(tmp_path)

    def test_init_bloom_creates_fresh_if_no_checkpoint(self, tmp_path):
        from scripts.prepare_data import init_bloom
        bloom = init_bloom(str(tmp_path))
        # Brand-new Bloom should be empty
        test_hash = hashlib.md5(b"test").hexdigest()
        assert test_hash not in bloom

    def test_init_bloom_restores_from_checkpoint(self, tmp_path):
        from scripts.prepare_data import init_bloom, save_bloom
        bloom = self._make_bloom()
        test_hash = hashlib.md5(b"checkpoint_test").hexdigest()
        bloom.add(test_hash)

        bloom_path = os.path.join(str(tmp_path), "bloom.pkl")
        save_bloom(bloom, bloom_path)

        restored = init_bloom(str(tmp_path))
        assert test_hash in restored
