"""Data preparation facade.

This module preserves the historical import surface while delegating concrete
pipelines to focused submodules for lower coupling and clearer structure.
"""

from src.data.preparation_modules.common import (  # split/test helpers kept for compatibility
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    _build_outer_split_indices_for_lstm,
    _build_train_subset_local_indices,
    _ensure_args_defaults,
    _loader_runtime_kwargs,
    _maybe_apply_train_subset,
    _resolve_subset_base_indices,
    _resolve_t_elaps_mode,
    _resolve_train_subset_ratio,
    _split_config,
    _split_lstm_indices_aligned_to_outer_windows,
    _split_with_event_loader,
    _subset_dataset_by_local_indices,
)
from src.data.preparation_modules.core import prepare_data
from src.data.preparation_modules.lstm import prepare_data_lstm
from src.data.preparation_modules.rf import prepare_data_rf, print_sample_distribution
from src.data.preparation_modules.tpp import (
    _auto_configure_global_bg_time_bounds,
    _drop_tpp_sequence_events,
    _maybe_wrap_tpp_train_dataset,
    prepare_data_tpp,
)

__all__ = [
    "prepare_data",
    "prepare_data_rf",
    "prepare_data_lstm",
    "prepare_data_tpp",
    "print_sample_distribution",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "_split_config",
    "_build_outer_split_indices_for_lstm",
]
