"""
tokenizer — Turkish BPE tokenizer package.

Downstream usage:
    from tokenizer import Tokenizer
    tok = Tokenizer("tokenizer/turkish_bpe.model")
"""

from .tokenizer import Tokenizer

__all__ = ["Tokenizer"]
