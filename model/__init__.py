"""
model/__init__.py — Public API for the AUModel package.

Usage:
    from model import AUModel, ModelConfig

Do not import from internal submodules directly.
"""
from .transformer import AUModel
from .config import ModelConfig

__all__ = ["AUModel", "ModelConfig"]
