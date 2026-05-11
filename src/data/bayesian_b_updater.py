"""Backward-compatible aliases for Bayesian b-value updater.

This module preserves historical import paths used in notebooks/tests:
`from src.data.bayesian_b_updater import BayesianGRBUpdater`.
"""

from src.models.updaters.bayesian_b_updater import BayesianGRBUpdater

__all__ = ["BayesianGRBUpdater"]

