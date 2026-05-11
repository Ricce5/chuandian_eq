"""Backward-compatible aliases for sliding-window b-value updater.

This module preserves historical import paths used in notebooks/tests:
`from src.data.slidewindow_b_updater import SlidingWindowBUpdater`.
"""

from src.models.updaters.slidewindow_b_updater import SlidingWindowBUpdater

__all__ = ["SlidingWindowBUpdater"]

