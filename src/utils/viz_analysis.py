"""Clear-named entrypoints for model analysis helpers.

This module provides a stable alias layer for analysis utilities while keeping
the canonical implementation in :mod:`src.utils.analysis`.
"""

from .analysis import predict_all, tsne_scatter

__all__ = [
    "predict_all",
    "tsne_scatter",
]

