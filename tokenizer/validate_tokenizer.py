"""
validate_tokenizer.py — Post-training quality gate for the Turkish BPE tokenizer.

Runs 4 validation checks and exits with:
  0  — all checks passed
  1  — one or more checks failed (gate failure)
  2  — runtime error (model load failed, etc.)

Usage:
    python tokenizer/validate_tokenizer.py
    python tokenizer/validate_tokenizer.py --model tokenizer/turkish_bpe.model
    python tokenizer/validate_tokenizer.py --corpus data/raw/tokenizer_corpus.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = Path("tokenizer/turkish_bpe.model")
DEFAULT_CORPUS_PATH = Path("data/raw/tokenizer_corpus.txt")

FERTILITY_THRESHOLD = 1.4          # average tokens per word (V-001)
FERTILITY_SAMPLE_SIZE = 10_000     # sentences to sample

# Exact Unicode Turkish chars (FR-003, T007 spec)
TURKISH_CHARS = [
    "\u00e7",  # c with cedilla (ç)
    "\u011f",  # g with breve (ğ)
    "\u0131",  # dotless i (ı)
    "\u0130",  # I with dot above (İ)
    "\u00f6",  # o with diaeresis (ö)
    "\u015f",  # s with cedilla (ş)
    "\u00fc",  # u with diaeresis (ü)
    "\u00dc",  # U with diaeresis (Ü)
    "\u00d6",  # O with diaeresis (Ö)
    "\u00c7",  # C with cedilla (Ç)
    "\u011e",  # G with breve (Ğ)
    "\u015e",  # S with cedilla (Ş)
]

BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]+>$")

# Special token expectations (T008)
EXPECTED_SPECIAL = {
    "<pad>": 0,
    "<unk>": 1,
    "<s>": 2,
    "</s>": 3,
    "[SYSTEM]": 4,
    "[USER]": 5,
    "[ASSISTANT]": 6,
    "[SEP]": 7,
}

# ---------------------------------------------------------------------------
# T005 — ValidationResult dataclass
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    check_name: str
    passed: bool
    measured_value: Optional[float]
    threshold: Optional[float]
    message: str


# ---------------------------------------------------------------------------
# T005 (cont.) — check_fertility
# ---------------------------------------------------------------------------
def check_fertility(sp, corpus_path: Path, n: int = FERTILITY_SAMPLE_SIZE) -> ValidationResult:
    """Sample *n* sentences from *corpus_path*, compute average tokens-per-word.

    Threshold: <= FERTILITY_THRESHOLD (1.4) — V-001.
    """
    if not corpus_path.exists():
        return ValidationResult(
            check_name="fertility",
            passed=False,
            measured_value=None,
            threshold=FERTILITY_THRESHOLD,
            message=f"Corpus file not found: {corpus_path}. Cannot evaluate fertility.",
        )

    sentences: list[str] = []
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                sentences.append(line)
                if len(sentences) >= n:
                    break

    if not sentences:
        return ValidationResult(
            check_name="fertility",
            passed=False,
            measured_value=None,
            threshold=FERTILITY_THRESHOLD,
            message=f"Corpus is empty or contains no non-blank lines: {corpus_path}",
        )

    total_tokens = 0
    total_words = 0
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        tokens = sp.encode(sentence, out_type=int)
        total_tokens += len(tokens)
        total_words += len(words)

    if total_words == 0:
        return ValidationResult(
            check_name="fertility",
            passed=False,
            measured_value=None,
            threshold=FERTILITY_THRESHOLD,
            message="No words found in sampled sentences.",
        )

    avg = total_tokens / total_words
    passed = avg <= FERTILITY_THRESHOLD
    msg = (
        f"avg tokens/word = {avg:.4f} (threshold {FERTILITY_THRESHOLD}, "
        f"sampled {len(sentences):,} sentences, {total_words:,} words)"
    )
    if not passed:
        msg += (
            "\n  REMEDIATION: Vocab size may be too small, or corpus quality is low. "
            "Try increasing vocab_size or improving corpus filtering."
        )
    return ValidationResult(
        check_name="fertility",
        passed=passed,
        measured_value=avg,
        threshold=FERTILITY_THRESHOLD,
        message=msg,
    )


# ---------------------------------------------------------------------------
# T006 — check_roundtrip
# ---------------------------------------------------------------------------
# 100 test strings: 25 common Turkish words, 25 Turkish sentences,
# 25 strings with numerals/punctuation, 25 edge cases
_ROUNDTRIP_TEST_STRINGS: list[str] = [
    # --- 25 common Turkish words ---
    "merhaba",
    "guzel",
    "calisma",
    "sehir",
    "ogrenci",
    "turkiye",
    "dil",
    "kitap",
    "ev",
    "araba",
    "insan",
    "zaman",
    "yer",
    "gun",
    "yil",
    "su",
    "hava",
    "yemek",
    "para",
    "is",
    "\u00e7ali\u015fmak",   # çalışmak
    "\u00f6\u011frenci",    # öğrenci
    "\u015fehir",           # şehir
    "g\u00fczel",           # güzel
    "\u00e7ocuk",           # çocuk
    # --- 25 Turkish sentences ---
    "Merhaba, nas\u0131ls\u0131n\u0131z?",
    "T\u00fcrkiye g\u00fczel bir \u00fclkedir.",
    "\u0130stanbul'da ya\u015f\u0131yorum.",
    "Kitap okumay\u0131 \u00e7ok seviyorum.",
    "\u00d6\u011frenciler \u00e7al\u0131\u015fkan ve ba\u015far\u0131l\u0131d\u0131r.",
    "Hava bug\u00fcn \u00e7ok g\u00fczel.",
    "\u00c7ay i\u00e7er misiniz?",
    "Bu \u015fehir \u00e7ok kalabal\u0131k.",
    "Yemek haz\u0131r m\u0131?",
    "Ankara T\u00fcrkiye'nin ba\u015fkentidir.",
    "Sabah erkenden uyand\u0131m.",
    "\u00d6\u011fretmen s\u0131n\u0131fta ders anlat\u0131yor.",
    "K\u0131\u015f geldi, hava so\u011fukla\u015ft\u0131.",
    "Annem yemek pi\u015firiyor.",
    "Babam her sabah kahve i\u00e7er.",
    "Arkada\u015flar\u0131m benimle geldi.",
    "\u015eiir yazmay\u0131 seviyorum.",
    "Deniz \u00e7ok g\u00fczel g\u00f6r\u00fcn\u00fcyor.",
    "Spor yapmak sa\u011fl\u0131kl\u0131d\u0131r.",
    "Bu filmi izlediniz mi?",
    "Sabah kahvalt\u0131s\u0131 \u00f6nemlidir.",
    "Kedi ve k\u00f6pek iyi arkada\u015f olamaz.",
    "Yaz tatili yakla\u015f\u0131yor.",
    "Bilgisayar\u0131m bozuldu.",
    "Telefon ald\u0131m, \u00e7ok memnunum.",
    # --- 25 strings with numerals/punctuation ---
    "123456789",
    "3.14159",
    "2024-01-01",
    "100%",
    "50 TL",
    "abc123",
    "Hello, World!",
    "x = y + z",
    "https://example.com",
    "user@email.com",
    "(555) 123-4567",
    "A, B, C",
    "1. Madde",
    "#hashtag",
    "@mention",
    "C++",
    "Python 3.10",
    "v1.2.3",
    "---",
    "...",
    "!!!",
    "?!?",
    '"T\u0131rnak i\u00e7inde"',
    "'tek t\u0131rnak'",
    "say\u0131: 42",
    # --- 25 edge cases ---
    "",                             # empty string
    " ",                            # single space
    "a",                            # single ASCII char
    "\u00e7",                       # single Turkish char (ç)
    "A",
    "0",
    "  \t  ",                       # whitespace only
    "aaa",
    "\u00e7\u00e7\u00e7",           # repeated Turkish char
    "TurkeyTurkiye",
    "12 34 56",
    "test.",
    "BUYUK HARF",                   # all uppercase
    "k\u00fc\u00e7\u00fck harf",    # all lowercase (küçük harf)
    "\u00c7\u011e\u0130\u00d6\u015e\u00dc",  # uppercase Turkish
    "\u00e7\u011f\u0131\u00f6\u015f\u00fc",  # lowercase Turkish
    "mixed \u00e7ali\u015fma 123",
    "a" * 50,                       # long repetition
    "T" * 50,
    "\u00e7" * 20,                  # long Turkish repetition
    "hello world foo bar",
    "bir iki \u00fc\u00e7 d\u00f6rt be\u015f",  # bir iki üç dört beş
    "\n",                           # newline (will be present in IDs, decoded to " " or "")
    "\r\n",
    "tab\there",
]


def check_roundtrip(sp, test_strings: list[str] = _ROUNDTRIP_TEST_STRINGS) -> ValidationResult:
    """Verify encode->decode round-trip for all test strings (V-002).

    Collects ALL failures without stopping early (report-all).
    """
    failures: list[str] = []
    for s in test_strings:
        try:
            ids = sp.encode(s, out_type=int)
            decoded = sp.decode(ids)
            if decoded != s:
                failures.append(
                    f"  Input:   {repr(s)}\n"
                    f"  Decoded: {repr(decoded)}"
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"  Input: {repr(s)} -> EXCEPTION: {exc}")

    passed = len(failures) == 0
    if passed:
        msg = f"All {len(test_strings)} strings round-trip correctly."
    else:
        msg = (
            f"{len(failures)}/{len(test_strings)} strings failed round-trip:\n"
            + "\n".join(failures[:10])
        )
        if len(failures) > 10:
            msg += f"\n  ... and {len(failures) - 10} more failures."
        msg += (
            "\n  REMEDIATION: Check normalization settings. "
            "identity normalization should preserve all characters exactly."
        )
    return ValidationResult(
        check_name="roundtrip",
        passed=passed,
        measured_value=float(len(test_strings) - len(failures)),
        threshold=float(len(test_strings)),
        message=msg,
    )


# ---------------------------------------------------------------------------
# T007 — check_turkish_chars
# ---------------------------------------------------------------------------
def check_turkish_chars(sp) -> ValidationResult:
    """Verify all 12 Turkish chars have dedicated vocabulary entries (V-003).

    A char passes if:
      - sp.piece_to_id(char) is NOT the UNK id
      - sp.id_to_piece(sp.piece_to_id(char)) == char  (exact round-trip)
      - the piece does NOT match the byte-fallback pattern <0xXX>
    """
    failures: list[str] = []
    unk_id = sp.unk_id()

    for char in TURKISH_CHARS:
        piece_id = sp.piece_to_id(char)
        if piece_id == unk_id:
            failures.append(f"  '{char}' (U+{ord(char):04X}) -> mapped to UNK")
            continue
        piece = sp.id_to_piece(piece_id)
        if BYTE_FALLBACK_RE.match(piece):
            failures.append(
                f"  '{char}' (U+{ord(char):04X}) -> byte-fallback piece '{piece}' (ID {piece_id})"
            )
        elif piece != char:
            failures.append(
                f"  '{char}' (U+{ord(char):04X}) -> piece mismatch: got '{piece}' (ID {piece_id})"
            )

    passed = len(failures) == 0
    if passed:
        msg = f"All {len(TURKISH_CHARS)} Turkish chars have dedicated vocabulary entries."
    else:
        msg = (
            f"{len(failures)}/{len(TURKISH_CHARS)} Turkish chars failed:\n"
            + "\n".join(failures)
            + "\n  REMEDIATION: Increase character_coverage (currently 0.9999) "
            "or verify the corpus contains sufficient Turkish text."
        )
    return ValidationResult(
        check_name="turkish_chars",
        passed=passed,
        measured_value=float(len(TURKISH_CHARS) - len(failures)),
        threshold=float(len(TURKISH_CHARS)),
        message=msg,
    )


# ---------------------------------------------------------------------------
# T008 — check_special_tokens
# ---------------------------------------------------------------------------
def check_special_tokens(sp) -> ValidationResult:
    """Verify all 8 special token IDs match expected values (V-004).

    Checks: distinct IDs, correct values, all in range [0, vocab_size).
    """
    failures: list[str] = []
    vocab_size = sp.get_piece_size()
    seen_ids: dict[int, str] = {}

    for piece, expected_id in EXPECTED_SPECIAL.items():
        actual_id = sp.piece_to_id(piece)
        # Check value
        if actual_id != expected_id:
            failures.append(
                f"  '{piece}': expected ID={expected_id}, got ID={actual_id}"
            )
        # Check range
        if not (0 <= actual_id < vocab_size):
            failures.append(
                f"  '{piece}': ID {actual_id} is out of range [0, {vocab_size})"
            )
        # Check uniqueness
        if actual_id in seen_ids:
            failures.append(
                f"  '{piece}' and '{seen_ids[actual_id]}' share the same ID={actual_id}"
            )
        seen_ids[actual_id] = piece

    passed = len(failures) == 0
    if passed:
        id_summary = ", ".join(f"{p}={EXPECTED_SPECIAL[p]}" for p in EXPECTED_SPECIAL)
        msg = f"All 8 special tokens verified: {id_summary}"
    else:
        msg = (
            f"{len(failures)} special token check(s) failed:\n"
            + "\n".join(failures)
            + "\n  REMEDIATION: Verify pad_id, unk_id, bos_id, eos_id and "
            "user_defined_symbols order in SentencePieceTrainer.train() call."
        )
    return ValidationResult(
        check_name="special_tokens",
        passed=passed,
        measured_value=float(len(EXPECTED_SPECIAL) - len(failures)),
        threshold=float(len(EXPECTED_SPECIAL)),
        message=msg,
    )


# ---------------------------------------------------------------------------
# T009 — main (report-all loop)
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a trained Turkish BPE tokenizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0  All checks passed
  1  One or more checks failed (gate failure)
  2  Runtime error (model could not be loaded)
""",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        metavar="PATH",
        help="Path to the trained model file. Default: %(default)s",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        metavar="PATH",
        help="Path to the corpus used for fertility check. Default: %(default)s",
    )
    args = parser.parse_args()

    # --- Load model ---
    try:
        import sentencepiece as spm  # type: ignore
    except ImportError:
        print("[ERROR] sentencepiece not installed.", file=sys.stderr)
        print("Install with: pip install sentencepiece>=0.1.99", file=sys.stderr)
        sys.exit(2)

    sp = spm.SentencePieceProcessor()
    try:
        sp.load(str(args.model))
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load model '{args.model}': {exc}", file=sys.stderr)
        print(
            "  REMEDIATION: Run training first: "
            "python tokenizer/train_tokenizer.py --download --train",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"[INFO] Model loaded: {args.model} (vocab_size={sp.get_piece_size():,})")
    print()

    # --- Run all 4 checks (report-all, never short-circuit) ---
    results: list[ValidationResult] = [
        check_fertility(sp, corpus_path=args.corpus),
        check_roundtrip(sp),
        check_turkish_chars(sp),
        check_special_tokens(sp),
    ]

    # --- Print summary table to stdout ---
    col_w = 16
    print(f"{'Check':<{col_w}}  {'Status':<8}  {'Value':>10}  {'Threshold':>10}")
    print("-" * (col_w + 34))
    for r in results:
        status = "[PASS]" if r.passed else "[FAIL]"
        val_str = f"{r.measured_value:.4f}" if r.measured_value is not None else "N/A"
        thr_str = f"{r.threshold:.4f}" if r.threshold is not None else "N/A"
        print(f"{r.check_name:<{col_w}}  {status:<8}  {val_str:>10}  {thr_str:>10}")

    print()

    # --- Print detailed messages to stderr for failures ---
    has_failure = any(not r.passed for r in results)
    for r in results:
        if not r.passed:
            print(f"[FAIL] {r.check_name}: {r.message}", file=sys.stderr)
            print(file=sys.stderr)

    # Single exit at the end (report-all pattern)
    if has_failure:
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
