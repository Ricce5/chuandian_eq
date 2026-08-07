"""Backward-compatible shim for sliding forecast helpers."""

from .forecast_eval import SlidingWindowEvaluationRange
from .forecast_eval_helpers import (
    build_post_step,
    compute_display_upper_cap,
    format_time_axis,
    infer_step_timedelta,
    resolve_catalog_time_reference,
    resolve_time_bounds,
    resolve_view_mode,
    run_sliding_window_forecast,
    style_axes,
    to_absolute_time_axis,
)

__all__ = [
    "SlidingWindowEvaluationRange",
    "run_sliding_window_forecast",
    "to_absolute_time_axis",
    "format_time_axis",
    "style_axes",
    "resolve_catalog_time_reference",
    "compute_display_upper_cap",
    "infer_step_timedelta",
    "resolve_time_bounds",
    "build_post_step",
    "resolve_view_mode",
]
