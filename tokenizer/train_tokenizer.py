"""
train_tokenizer.py — Download Turkish corpus and train a SentencePiece BPE tokenizer.

Usage:
    python tokenizer/train_tokenizer.py --download
    python tokenizer/train_tokenizer.py --train
    python tokenizer/train_tokenizer.py --download --train
    python tokenizer/train_tokenizer.py --train --corpus /path/to/corpus.txt
    python tokenizer/train_tokenizer.py --train --force   # overwrite existing model
"""

from __future__ import annotations

import argparse
import os
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (locked — do not modify without updating spec + constitution)
# ---------------------------------------------------------------------------
VOCAB_SIZE = 64_000
MODEL_TYPE = "bpe"
CHARACTER_COVERAGE = 0.9999
NORMALIZATION_RULE = "identity"
BYTE_FALLBACK = True
INPUT_SENTENCE_SIZE = 10_000_000
SHUFFLE_INPUT = True
RANDOM_SEED = 42

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
USER_DEFINED_SYMBOLS = "[SYSTEM],[USER],[ASSISTANT],[SEP]"

CORPUS_PATH = Path("data/raw/tokenizer_corpus.txt")
MODEL_PREFIX = "tokenizer/turkish_bpe"
MODEL_FILE = Path("tokenizer/turkish_bpe.model")
VOCAB_FILE = Path("tokenizer/turkish_bpe.vocab")

HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.tr"
MIN_DOC_CHARS = 200
MIN_CORPUS_BYTES = 1_024 * 1_024  # 1 MB


# ---------------------------------------------------------------------------
# T002 — download_corpus
# ---------------------------------------------------------------------------
def download_corpus(output_path: Path = CORPUS_PATH) -> None:
    """Stream Turkish Wikipedia via HuggingFace datasets, NFC-normalise, write
    one document per line to *output_path*.

    Raises:
        RuntimeError: if output file is missing or smaller than MIN_CORPUS_BYTES
            after writing (guards against partial / failed downloads).
    """
    try:
        from datasets import load_dataset  # type: ignore
        from tqdm import tqdm  # type: ignore
    except ImportError as exc:
        print(f"[ERROR] Missing dependency: {exc}", file=sys.stderr)
        print("Install with: pip install datasets tqdm", file=sys.stderr)
        sys.exit(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading {HF_DATASET} ({HF_CONFIG}) ...")

    dataset = load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True)
    written = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as fh:
        for doc in tqdm(dataset, desc="Writing corpus", unit=" docs"):
            text: str = doc.get("text", "") or ""
            # NFC normalisation
            text = unicodedata.normalize("NFC", text)
            # Skip short documents
            if len(text) < MIN_DOC_CHARS:
                skipped += 1
                continue
            fh.write(text.replace("\n", " ") + "\n")
            written += 1

    print(f"[INFO] Wrote {written:,} documents ({skipped:,} skipped) -> {output_path}")

    # Guard: verify output exists and is large enough
    if not output_path.exists():
        raise RuntimeError(
            f"Corpus file not found after download: {output_path}\n"
            "This may indicate a failed or interrupted download."
        )
    size = output_path.stat().st_size
    if size < MIN_CORPUS_BYTES:
        raise RuntimeError(
            f"Corpus file is only {size:,} bytes (< {MIN_CORPUS_BYTES:,} bytes minimum).\n"
            "Download may have failed or dataset may be empty. "
            f"Delete {output_path} and retry."
        )
    print(f"[INFO] Corpus size: {size / 1_024 / 1_024:.1f} MB [OK]")


# ---------------------------------------------------------------------------
# T003 — run_spm_training
# ---------------------------------------------------------------------------
def run_spm_training(
    corpus_path: Path = CORPUS_PATH,
    force: bool = False,
) -> None:
    """Train a SentencePiece BPE tokenizer on *corpus_path*.

    Args:
        corpus_path: Path to the plain-text training corpus (one sentence per line).
        force: If True, overwrite existing model files. If False, raises if they
               already exist.

    Raises:
        FileNotFoundError: if *corpus_path* does not exist.
        FileExistsError: if model already exists and *force* is False.
    """
    try:
        import sentencepiece as spm  # type: ignore
    except ImportError:
        print("[ERROR] sentencepiece not installed.", file=sys.stderr)
        print("Install with: pip install sentencepiece>=0.1.99", file=sys.stderr)
        sys.exit(2)

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}\n"
            "Run with --download first, or pass --corpus <path> to supply your own."
        )

    # Overwrite guard
    if MODEL_FILE.exists() and not force:
        raise FileExistsError(
            f"Model already exists: {MODEL_FILE}\n"
            "Use --force to overwrite, or delete the file manually."
        )

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Training SentencePiece BPE tokenizer (vocab_size={VOCAB_SIZE:,}) ...")
    print(f"[INFO] Input corpus: {corpus_path}")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=MODEL_PREFIX,
        model_type=MODEL_TYPE,
        vocab_size=VOCAB_SIZE,
        character_coverage=CHARACTER_COVERAGE,
        normalization_rule_name=NORMALIZATION_RULE,
        byte_fallback=BYTE_FALLBACK,
        input_sentence_size=INPUT_SENTENCE_SIZE,
        shuffle_input_sentence=SHUFFLE_INPUT,
        seed_sentencepiece_size=RANDOM_SEED,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        user_defined_symbols=USER_DEFINED_SYMBOLS,
    )

    print(f"[INFO] Model saved: {MODEL_FILE} [OK]")
    print(f"[INFO] Vocab saved: {VOCAB_FILE} [OK]")

    # Quick sanity check
    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_FILE))
    assert sp.get_piece_size() == VOCAB_SIZE, (
        f"Vocab size mismatch: expected {VOCAB_SIZE}, got {sp.get_piece_size()}"
    )
    print(f"[INFO] Vocab size verified: {sp.get_piece_size():,} [OK]")


# ---------------------------------------------------------------------------
# T004 — main CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Turkish BPE SentencePiece tokenizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: download corpus then train
  python tokenizer/train_tokenizer.py --download --train

  # Download only
  python tokenizer/train_tokenizer.py --download

  # Train only (corpus already downloaded)
  python tokenizer/train_tokenizer.py --train

  # Train on a custom corpus (FR-008: accepts any plain-text file)
  python tokenizer/train_tokenizer.py --train --corpus /path/to/corpus.txt

  # Overwrite an existing model
  python tokenizer/train_tokenizer.py --train --force
""",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and prepare the Turkish Wikipedia corpus.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the SentencePiece BPE model.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_PATH,
        metavar="PATH",
        help=(
            "Path to a plain-text corpus file (one document per line). "
            "Skips download when provided. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model files without prompting.",
    )

    args = parser.parse_args()

    if not args.download and not args.train:
        parser.print_help()
        sys.exit(0)

    try:
        if args.download:
            download_corpus(output_path=CORPUS_PATH)

        if args.train:
            corpus = args.corpus
            run_spm_training(corpus_path=corpus, force=args.force)

    except (FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
