"""
tokenizer.py — Stable wrapper around the trained SentencePiece BPE model.

Downstream usage (all epics):
    from tokenizer import Tokenizer
    tok = Tokenizer("tokenizer/turkish_bpe.model")
    ids = tok.encode("Merhaba dunya", add_bos=True)
    text = tok.decode(ids)

This interface is frozen once the model is trained. See:
    specs/001-turkish-tokenizer/contracts/tokenizer-interface.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


class Tokenizer:
    """Thin, stable wrapper around a trained SentencePiece BPE model.

    All token ID properties delegate to the underlying SPM model so they
    reflect the actual trained vocabulary rather than hard-coded constants.

    Special token ID mapping (locked by training config):
        <pad>       -> 0
        <unk>       -> 1
        <s>         -> 2   (BOS)
        </s>        -> 3   (EOS)
        [SYSTEM]    -> 4
        [USER]      -> 5
        [ASSISTANT] -> 6
        [SEP]       -> 7
    """

    def __init__(self, model_path: Union[str, Path]) -> None:
        """Load the SentencePiece model from *model_path*.

        Args:
            model_path: Absolute or relative path to ``turkish_bpe.model``.

        Raises:
            FileNotFoundError: if the model file does not exist.
            RuntimeError: if the file exists but is not a valid SPM model.
        """
        try:
            import sentencepiece as spm  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentencepiece is required. Install with: "
                "pip install sentencepiece>=0.1.99"
            ) from exc

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Tokenizer model not found: {model_path}\n"
                "Train the model first: "
                "python tokenizer/train_tokenizer.py --download --train"
            )

        self._sp = spm.SentencePieceProcessor()
        try:
            self._sp.load(str(model_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load SentencePiece model '{model_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode *text* to a list of token IDs.

        Args:
            text: Input string. Returns ``[]`` for empty strings.
            add_bos: Prepend BOS token (ID 2) when ``True``.
            add_eos: Append EOS token (ID 3) when ``True``.

        Returns:
            List of integer token IDs in ``[0, vocab_size)``.
        """
        if not text:
            ids: list[int] = []
            if add_bos:
                ids.insert(0, self.bos_id)
            if add_eos:
                ids.append(self.eos_id)
            return ids

        ids = self._sp.encode(text, out_type=int)
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        ids: list[int],
        skip_special: bool = True,
    ) -> str:
        """Decode a list of token IDs back to a string.

        Args:
            ids: Token IDs to decode.
            skip_special: When ``True``, special token IDs are silently removed
                before decoding so they do not appear as literal strings in the
                output.

        Returns:
            Decoded string. Returns ``""`` for an empty or all-special list.
        """
        if skip_special:
            special = {
                self.pad_id,
                self.bos_id,
                self.eos_id,
                self.system_id,
                self.user_id,
                self.assistant_id,
                self.sep_id,
            }
            ids = [i for i in ids if i not in special]
        if not ids:
            return ""
        return self._sp.decode(ids)

    # ------------------------------------------------------------------
    # Vocabulary properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary (should be 64,000)."""
        return self._sp.get_piece_size()

    # ------------------------------------------------------------------
    # Special token ID properties
    # ------------------------------------------------------------------

    @property
    def pad_id(self) -> int:
        """ID of the padding token ``<pad>`` (should be 0)."""
        return self._sp.pad_id()

    @property
    def unk_id(self) -> int:
        """ID of the unknown token ``<unk>`` (should be 1)."""
        return self._sp.unk_id()

    @property
    def bos_id(self) -> int:
        """ID of the beginning-of-sequence token ``<s>`` (should be 2)."""
        return self._sp.bos_id()

    @property
    def eos_id(self) -> int:
        """ID of the end-of-sequence token ``</s>`` (should be 3)."""
        return self._sp.eos_id()

    @property
    def system_id(self) -> int:
        """ID of the ``[SYSTEM]`` chat control token (should be 4)."""
        return self._sp.piece_to_id("[SYSTEM]")

    @property
    def user_id(self) -> int:
        """ID of the ``[USER]`` chat control token (should be 5)."""
        return self._sp.piece_to_id("[USER]")

    @property
    def assistant_id(self) -> int:
        """ID of the ``[ASSISTANT]`` chat control token (should be 6)."""
        return self._sp.piece_to_id("[ASSISTANT]")

    @property
    def sep_id(self) -> int:
        """ID of the ``[SEP]`` separator token (should be 7)."""
        return self._sp.piece_to_id("[SEP]")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size})"
