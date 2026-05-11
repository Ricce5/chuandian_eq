"""Updater registry entrypoint.

Core contracts and sampling helpers must remain importable even when optional
dependencies for specific updater implementations are unavailable.
"""

from .base import BValueUpdaterBase
from .bayesian_b_updater import BayesianGRBUpdater
from .slidewindow_b_updater import FixedTimeWindowGRB as SlidingWindowBUpdater
from .sampling_wrapper import (
    UpdaterSamplingWrapper,
    build_sampling_updater,
    make_sampling_updater,
    resolve_sampling_updater,
)


def _safe_import(module_name: str) -> None:
    try:
        __import__(f"{__name__}.{module_name}")
    except Exception:
        return


for _module_name in (
    "pyro_kf_b_updater",
):
    _safe_import(_module_name)


def bayesian_b_updater(**kwargs):
    """Backward-compatible factory alias for Bayesian GR b-value updater."""
    return BayesianGRBUpdater(**kwargs)


def slidewindow_b_updater(**kwargs):
    """Backward-compatible factory alias for sliding-window b-value updater."""
    return SlidingWindowBUpdater(**kwargs)


bayesian_gr_updater = bayesian_b_updater


__all__ = [
    "BValueUpdaterBase",
    "UpdaterSamplingWrapper",
    "build_sampling_updater",
    "make_sampling_updater",
    "resolve_sampling_updater",
    "BayesianGRBUpdater",
    "SlidingWindowBUpdater",
    "bayesian_b_updater",
    "bayesian_gr_updater",
    "slidewindow_b_updater",
]
