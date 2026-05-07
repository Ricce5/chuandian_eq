"""Model builder registry package."""

from .registry import ModelBuilder
from . import clf as _clf_builders  # noqa: F401
from . import reg as _reg_builders  # noqa: F401
from . import tpp as _tpp_builders  # noqa: F401

__all__ = ["ModelBuilder"]
