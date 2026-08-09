from __future__ import annotations

import hashlib
import inspect
import json
import re
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import cycler
from matplotlib import ticker as mticker


DEFAULT_PLOT_COLORS: dict[str, str] = {
    "true": "#1f77b4",
    "mean": "#ff7f0e",
    "pi": "#4c78a8",
    "err": "#d62728",
    "covered": "#2ca02c",
    "below": "#e45756",
    "above": "#9467bd",
}

DEFAULT_SLIDING_PLOT_FIGSIZES: dict[str, tuple[float, float]] = {
    "counts": (10.5, 4.4),
    "error": (10.5, 3.8),
    "coverage": (10.5, 3.2),
    "count_scatter": (6.8, 6.2),
    "mag_max": (10.5, 4.2),
    "mag_max_scatter": (6.8, 6.2),
}


@dataclass(frozen=True, slots=True)
class SlidingWindowEvaluationRange:
    """Relative-time bounds for the target portion of a sliding evaluation."""

    name: str = "full"
    start: float | None = None
    end: float | None = None
    grid_anchor: float | None = None


@dataclass(frozen=True, slots=True)
class SlidingWindowForecastConfig:
    duration: float = 12
    slide_step: float = 12
    quantiles: tuple[float, float] = (2.5, 97.5)
    samples_per_batch: int = 1000
    predict_b: bool | None = None
    bg_cache_seq: Any | None = None
    return_sim_count_matrix: bool = False
    compute_mag_max: bool = False
    return_sim_mag_max_matrix: bool = False
    precomputed_counts: np.ndarray | None = None
    precomputed_mag_max: np.ndarray | None = None
    include_truncated_final_window: bool = False
    evaluation_range: SlidingWindowEvaluationRange | None = None


@dataclass(slots=True)
class SlidingWindowForecastResult:
    t_forecast: np.ndarray
    counts: np.ndarray
    quantiles: np.ndarray
    mean: np.ndarray
    sim_count_matrix: np.ndarray | None = None
    mag_max: np.ndarray | None = None
    mag_max_quantiles: np.ndarray | None = None
    mag_max_mean: np.ndarray | None = None
    sim_mag_max_matrix: np.ndarray | None = None
    t_window_end: np.ndarray | None = None

    def __iter__(self):
        yield from self.as_legacy_tuple()

    def __len__(self):
        return 9 if self.mag_max is not None else (5 if self.sim_count_matrix is not None else 4)

    def __getitem__(self, index):
        return self.as_legacy_tuple()[index]

    def as_legacy_tuple(self):
        if self.mag_max is not None:
            return (
                self.t_forecast,
                self.counts,
                self.quantiles,
                self.mean,
                self.sim_count_matrix,
                self.mag_max,
                self.mag_max_quantiles,
                self.mag_max_mean,
                self.sim_mag_max_matrix,
            )
        if self.sim_count_matrix is not None:
            return (
                self.t_forecast,
                self.counts,
                self.quantiles,
                self.mean,
                self.sim_count_matrix,
            )
        return (self.t_forecast, self.counts, self.quantiles, self.mean)


def _stack_or_empty(rows, *, cols: int, dtype):
    """Stack row arrays or return an empty 2D array with a fixed column size."""
    if rows:
        return np.stack(rows, axis=0)
    return np.empty((0, cols), dtype=dtype)


def _sequence_observation_bounds(seq) -> tuple[float, float]:
    """Return the complete observed sequence bounds in relative time."""
    start = float(getattr(seq, "t_start", 0.0))
    fallback_end = float(seq.arrival_times[-1].item())
    end = float(getattr(seq, "t_end", fallback_end))
    return start, end


def resolve_sliding_window_evaluation_range(
    seq,
    evaluation_range: SlidingWindowEvaluationRange | None = None,
) -> SlidingWindowEvaluationRange:
    """Resolve and validate a target range against a sequence.

    The default preserves the previous evaluator coverage: it starts at the
    first event and ends at the final event. An explicit range may extend to
    ``seq.t_end`` so that a final zero-event interval can be evaluated.
    """
    first_event = float(seq.arrival_times[0].item())
    last_event = float(seq.arrival_times[-1].item())
    sequence_start, sequence_end = _sequence_observation_bounds(seq)

    if evaluation_range is None:
        name = "full"
        start = first_event
        end = last_event
        grid_anchor = start
        grid_anchor_follows_start = True
    else:
        name = str(evaluation_range.name).strip() or "custom"
        start = (
            first_event
            if evaluation_range.start is None
            else float(evaluation_range.start)
        )
        end = (
            last_event
            if evaluation_range.end is None
            else float(evaluation_range.end)
        )
        grid_anchor = (
            start
            if evaluation_range.grid_anchor is None
            else float(evaluation_range.grid_anchor)
        )
        grid_anchor_follows_start = evaluation_range.grid_anchor is None

    if not (
        np.isfinite(start)
        and np.isfinite(end)
        and np.isfinite(grid_anchor)
    ):
        raise ValueError(
            "Sliding evaluation range bounds and grid_anchor must be finite."
        )

    bound_tolerance = 1e-9 * max(
        1.0,
        abs(sequence_start),
        abs(sequence_end),
        abs(start),
        abs(end),
    )
    if start < sequence_start:
        if np.isclose(
            start,
            sequence_start,
            rtol=0.0,
            atol=bound_tolerance,
        ):
            start = sequence_start
        else:
            raise ValueError(
                "evaluation_range.start="
                f"{start} must be >= seq.t_start={sequence_start}."
            )
    if end > sequence_end:
        if np.isclose(
            end,
            sequence_end,
            rtol=0.0,
            atol=bound_tolerance,
        ):
            end = sequence_end
        else:
            raise ValueError(
                f"evaluation_range.end={end} must be <= seq.t_end={sequence_end}."
            )
    if grid_anchor_follows_start:
        grid_anchor = start
    if not start < end:
        raise ValueError(
            f"evaluation_range must satisfy start < end, got start={start}, end={end}."
        )

    return SlidingWindowEvaluationRange(
        name=name,
        start=start,
        end=end,
        grid_anchor=grid_anchor,
    )


def _first_grid_start_at_or_after(
    lower_bound: float,
    grid_anchor: float,
    slide_step: float,
) -> float:
    """Return the first grid point at or after lower_bound."""
    if lower_bound <= grid_anchor:
        return grid_anchor

    offset = (lower_bound - grid_anchor) / slide_step
    steps = int(np.ceil(offset - 1e-12))
    candidate = grid_anchor + steps * slide_step
    if candidate < lower_bound and not np.isclose(candidate, lower_bound):
        candidate += slide_step
    return candidate


def _build_sliding_window_bounds(
    seq,
    *,
    duration: float,
    slide_step: float,
    evaluation_range: SlidingWindowEvaluationRange | None = None,
    include_truncated_final_window: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build forecast starts and target ends for one sliding evaluation range."""
    duration = float(duration)
    slide_step = float(slide_step)
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}.")
    if slide_step <= 0:
        raise ValueError(f"slide_step must be positive, got {slide_step}.")

    resolved_range = resolve_sliding_window_evaluation_range(seq, evaluation_range)
    first_event = float(seq.arrival_times[0].item())
    lower_bound = max(float(resolved_range.start), first_event + duration)
    first_start = _first_grid_start_at_or_after(
        lower_bound,
        float(resolved_range.grid_anchor),
        slide_step,
    )
    candidate_starts = np.arange(first_start, float(resolved_range.end), slide_step)
    candidate_ends = np.minimum(
        candidate_starts + duration,
        float(resolved_range.end),
    )

    if include_truncated_final_window:
        return candidate_starts, candidate_ends

    full_window_mask = candidate_starts + duration <= float(resolved_range.end)
    return candidate_starts[full_window_mask], candidate_ends[full_window_mask]


def _build_sliding_window_starts(
    start: float,
    end: float,
    duration: float,
    slide_step: float,
    *,
    include_truncated_final_window: bool = False,
) -> np.ndarray:
    """Backward-compatible standalone wrapper for sliding start construction."""
    class _SequenceBounds:
        def __init__(self, sequence_start: float, sequence_end: float):
            self.t_start = float(sequence_start)
            self.t_end = float(sequence_end)
            self.arrival_times = torch.tensor(
                [sequence_start, sequence_end],
                dtype=torch.float64,
            )

    seq = _SequenceBounds(start, end)
    starts, _ = _build_sliding_window_bounds(
        seq,
        duration=duration,
        slide_step=slide_step,
        evaluation_range=SlidingWindowEvaluationRange(
            start=start,
            end=end,
            grid_anchor=start,
        ),
        include_truncated_final_window=include_truncated_final_window,
    )
    return starts


def _cache_background_sequence(model: Any, bg_cache_seq: Any | None) -> None:
    if bg_cache_seq is None:
        return
    bg_model = getattr(model, "bg_model", None)
    if bg_model is None:
        return
    time_series = getattr(bg_cache_seq, "time_series", None)
    time_series_times = getattr(bg_cache_seq, "time_series_times", None)
    if time_series is None or time_series_times is None:
        return
    cached = getattr(bg_model, "ts_batch_cache", None)
    cached_times = getattr(cached, "time_series_times", None)
    cached_series = getattr(cached, "time_series", None)
    source_times = torch.as_tensor(time_series_times).reshape(-1)
    source_series = torch.as_tensor(time_series)
    if cached_times is not None:
        cached_times = torch.as_tensor(cached_times).reshape(-1)
        if (
            cached_times.shape == source_times.shape
            and cached_series is not None
            and torch.as_tensor(cached_series).shape == source_series.shape
            and torch.allclose(
                cached_times.detach().cpu(),
                source_times.detach().cpu(),
                rtol=0.0,
                atol=0.0,
            )
            and torch.allclose(
                torch.as_tensor(cached_series).detach().cpu(),
                source_series.detach().cpu(),
                rtol=0.0,
                atol=0.0,
            )
        ):
            return
    bg_model.cache_batch(
        time_series=time_series.unsqueeze(0),
        time_series_times=time_series_times.unsqueeze(0),
    )


def _background_cache_time_bounds(model: Any) -> tuple[float, float, float] | None:
    """Return cached background time bounds and a small endpoint tolerance."""
    bg_model = getattr(model, "bg_model", None)
    if bg_model is None:
        return None
    cached = getattr(bg_model, "ts_batch_cache", None)
    if cached is None:
        return None
    time_series_times = getattr(cached, "time_series_times", None)
    if time_series_times is None:
        return None

    cache_time_tensor = torch.as_tensor(time_series_times).detach().cpu()
    ts_times = cache_time_tensor.numpy()
    ts_times = np.asarray(ts_times, dtype=np.float64).reshape(-1)
    if ts_times.size == 0 or not np.all(np.isfinite(ts_times)):
        return None

    positive_steps = np.diff(ts_times)
    positive_steps = positive_steps[positive_steps > 0.0]
    median_step = (
        float(np.median(positive_steps))
        if positive_steps.size > 0
        else 0.0
    )
    cache_start = float(ts_times[0])
    cache_end = float(ts_times[-1])
    scale = max(1.0, abs(cache_start), abs(cache_end))
    dtype_tolerance = 0.0
    if cache_time_tensor.is_floating_point():
        dtype_tolerance = 4.0 * float(
            torch.finfo(cache_time_tensor.dtype).eps
        ) * scale
    grid_endpoint_tolerance = median_step if median_step > 0.0 else 0.0
    endpoint_tolerance = max(
        1e-9,
        1e-9 * scale,
        dtype_tolerance,
        grid_endpoint_tolerance,
    )
    return cache_start, cache_end, endpoint_tolerance


def _limit_range_to_background_cache(
    model: Any,
    evaluation_range: SlidingWindowEvaluationRange,
) -> SlidingWindowEvaluationRange:
    """Numerically guard sliding target bounds against background cache edges."""
    cache_bounds = _background_cache_time_bounds(model)
    if cache_bounds is None:
        return evaluation_range

    cache_start, cache_end, endpoint_tolerance = cache_bounds
    start = float(evaluation_range.start)
    end = float(evaluation_range.end)
    grid_anchor = float(evaluation_range.grid_anchor)
    adjusted = False

    if start < cache_start:
        if cache_start - start <= endpoint_tolerance:
            start = cache_start
            if np.isclose(
                grid_anchor,
                evaluation_range.start,
                rtol=0.0,
                atol=endpoint_tolerance,
            ):
                grid_anchor = start
            adjusted = True
        else:
            raise ValueError(
                "Sliding evaluation range starts before cached background "
                f"range: start={start:.10g}, bg_cache_start={cache_start:.10g}. "
                "Pass a background cache sequence covering the forecast interval "
                "or restrict --eval-start."
            )

    if end > cache_end:
        if end - cache_end <= endpoint_tolerance:
            end = cache_end
            adjusted = True
        else:
            raise ValueError(
                "Sliding evaluation range ends after cached background range: "
                f"end={end:.10g}, bg_cache_end={cache_end:.10g}. "
                "Pass a background cache sequence covering the forecast interval "
                "or restrict --eval-end."
            )

    if not start < end:
        raise ValueError(
            "Sliding evaluation range has no positive duration after applying "
            f"background cache bounds: start={start}, end={end}."
        )

    if adjusted:
        warnings.warn(
            "Clamped sliding evaluation range to cached background time bounds "
            f"[{cache_start:.10g}, {cache_end:.10g}] within endpoint tolerance "
            f"{endpoint_tolerance:.3g}; no background feature values were "
            "extrapolated.",
            RuntimeWarning,
            stacklevel=2,
        )

    return SlidingWindowEvaluationRange(
        name=evaluation_range.name,
        start=start,
        end=end,
        grid_anchor=grid_anchor,
    )


def _as_1d_array(name: str, value):
    """Convert value to a NumPy array and enforce 1D shape.

    Raises:
        ValueError: If the converted array is not 1-dimensional.
    """
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}.")
    return arr


def _as_2d_array(name: str, value):
    """Convert value to a NumPy array and enforce 2D shape.

    Raises:
        ValueError: If the converted array is not 2-dimensional.
    """
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    return arr


_SLIDING_RESULT_REQUIRED_KEYS = {"t_forecast", "counts", "quantiles", "mean"}
_SLIDING_RESULT_ALLOWED_KEYS = {
    "t_forecast",
    "t_window_end",
    "counts",
    "quantiles",
    "mean",
    "sim_count_matrix",
    "mag_max",
    "mag_max_quantiles",
    "mag_max_mean",
    "sim_mag_max_matrix",
}


def _sliding_result_to_mapping(result: SlidingWindowForecastResult) -> dict[str, Any]:
    """Normalize a SlidingWindowForecastResult into a dict payload."""
    result_mapping = {
        "t_forecast": result.t_forecast,
        "counts": result.counts,
        "quantiles": result.quantiles,
        "mean": result.mean,
        "sim_count_matrix": result.sim_count_matrix,
        "mag_max": result.mag_max,
        "mag_max_quantiles": result.mag_max_quantiles,
        "mag_max_mean": result.mag_max_mean,
        "sim_mag_max_matrix": result.sim_mag_max_matrix,
    }
    if result.t_window_end is not None:
        result_mapping["t_window_end"] = result.t_window_end
    return result_mapping


def _tuple_sliding_result_to_mapping(result: tuple) -> dict[str, Any] | None:
    """Convert legacy tuple outputs to a dict payload.

    Supported tuple shapes are 4, 5, and 9 elements.
    Returns None when the tuple shape is not recognized.
    """
    match result:
        case (t_forecast, counts, quantiles, mean):
            return {
                "t_forecast": t_forecast,
                "counts": counts,
                "quantiles": quantiles,
                "mean": mean,
            }
        case (t_forecast, counts, quantiles, mean, sim_count_matrix):
            return {
                "t_forecast": t_forecast,
                "counts": counts,
                "quantiles": quantiles,
                "mean": mean,
                "sim_count_matrix": sim_count_matrix,
            }
        case (
            t_forecast,
            counts,
            quantiles,
            mean,
            sim_count_matrix,
            mag_max,
            mag_max_quantiles,
            mag_max_mean,
            sim_mag_max_matrix,
        ):
            return {
                "t_forecast": t_forecast,
                "counts": counts,
                "quantiles": quantiles,
                "mean": mean,
                "sim_count_matrix": sim_count_matrix,
                "mag_max": mag_max,
                "mag_max_quantiles": mag_max_quantiles,
                "mag_max_mean": mag_max_mean,
                "sim_mag_max_matrix": sim_mag_max_matrix,
            }
    return None


def _build_sliding_result(
    *,
    t_forecast,
    t_window_end=None,
    counts,
    quantiles,
    mean,
    sim_count_matrix=None,
    mag_max=None,
    mag_max_quantiles=None,
    mag_max_mean=None,
    sim_mag_max_matrix=None,
):
    """Build a validated SlidingWindowForecastResult from raw payload fields.

    This function centralizes shape checks and cross-field consistency checks.
    """
    t_forecast_arr = _as_1d_array("t_forecast", t_forecast)
    counts_arr = _as_1d_array("counts", counts)
    mean_arr = _as_1d_array("mean", mean)
    quantiles_arr = _as_2d_array("quantiles", quantiles)

    n_bins = int(t_forecast_arr.shape[0])
    if counts_arr.shape[0] != n_bins:
        raise ValueError("counts and t_forecast must have the same length.")
    if mean_arr.shape[0] != n_bins:
        raise ValueError("mean and t_forecast must have the same length.")
    if quantiles_arr.shape[0] != n_bins:
        raise ValueError("quantiles and t_forecast must have the same number of rows.")
    if quantiles_arr.shape[1] < 2:
        raise ValueError("quantiles must have at least 2 columns.")

    t_window_end_arr = None
    if t_window_end is not None:
        t_window_end_arr = _as_1d_array("t_window_end", t_window_end)
        if t_window_end_arr.shape[0] != n_bins:
            raise ValueError(
                "t_window_end and t_forecast must have the same length."
            )

    sim_count_arr = None
    if sim_count_matrix is not None:
        sim_count_arr = _as_2d_array("sim_count_matrix", sim_count_matrix)
        if sim_count_arr.shape[0] != n_bins:
            raise ValueError("sim_count_matrix rows must match t_forecast length.")

    mag_max_arr = None
    mag_max_quantiles_arr = None
    mag_max_mean_arr = None
    sim_mag_max_arr = None

    if mag_max is not None or mag_max_quantiles is not None or mag_max_mean is not None or sim_mag_max_matrix is not None:
        if mag_max is None or mag_max_quantiles is None or mag_max_mean is None:
            raise ValueError("mag_max, mag_max_quantiles, and mag_max_mean must be provided together.")

        mag_max_arr = _as_1d_array("mag_max", mag_max)
        mag_max_quantiles_arr = _as_2d_array("mag_max_quantiles", mag_max_quantiles)
        mag_max_mean_arr = _as_1d_array("mag_max_mean", mag_max_mean)

        if mag_max_arr.shape[0] != n_bins:
            raise ValueError("mag_max and t_forecast must have the same length.")
        if mag_max_mean_arr.shape[0] != n_bins:
            raise ValueError("mag_max_mean and t_forecast must have the same length.")
        if mag_max_quantiles_arr.shape[0] != n_bins:
            raise ValueError("mag_max_quantiles and t_forecast must have the same number of rows.")
        if mag_max_quantiles_arr.shape[1] < 2:
            raise ValueError("mag_max_quantiles must have at least 2 columns.")

        if sim_mag_max_matrix is not None:
            sim_mag_max_arr = _as_2d_array("sim_mag_max_matrix", sim_mag_max_matrix)
            if sim_mag_max_arr.shape[0] != n_bins:
                raise ValueError("sim_mag_max_matrix rows must match t_forecast length.")

    return SlidingWindowForecastResult(
        t_forecast=t_forecast_arr,
        counts=counts_arr,
        quantiles=quantiles_arr,
        mean=mean_arr,
        sim_count_matrix=sim_count_arr,
        mag_max=mag_max_arr,
        mag_max_quantiles=mag_max_quantiles_arr,
        mag_max_mean=mag_max_mean_arr,
        sim_mag_max_matrix=sim_mag_max_arr,
        t_window_end=t_window_end_arr,
    )


def _coerce_sliding_window_forecast_mapping(result: Mapping[str, Any]):
    """Validate a mapping payload and convert it to SlidingWindowForecastResult.

    Required keys: t_forecast, counts, quantiles, mean.
    Optional keys include simulation matrices and magnitude statistics.
    """
    keys = set(result.keys())
    missing = sorted(_SLIDING_RESULT_REQUIRED_KEYS - keys)
    if missing:
        raise ValueError(f"Missing required keys in sliding result mapping: {missing}.")

    unknown = sorted(keys - _SLIDING_RESULT_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in sliding result mapping: {unknown}.")

    return _build_sliding_result(
        t_forecast=result["t_forecast"],
        t_window_end=result.get("t_window_end"),
        counts=result["counts"],
        quantiles=result["quantiles"],
        mean=result["mean"],
        sim_count_matrix=result.get("sim_count_matrix"),
        mag_max=result.get("mag_max"),
        mag_max_quantiles=result.get("mag_max_quantiles"),
        mag_max_mean=result.get("mag_max_mean"),
        sim_mag_max_matrix=result.get("sim_mag_max_matrix"),
    )


def _coerce_sliding_window_forecast_result(result):
    """Coerce heterogeneous forecast outputs into SlidingWindowForecastResult.

    Accepted inputs:
        - SlidingWindowForecastResult
        - Mapping with named fields
        - Legacy tuple (4/5/9 elements)
    """
    if isinstance(result, SlidingWindowForecastResult):
        result = _sliding_result_to_mapping(result)

    if isinstance(result, Mapping):
        return _coerce_sliding_window_forecast_mapping(result)

    if isinstance(result, tuple):
        mapping_payload = _tuple_sliding_result_to_mapping(result)
        if mapping_payload is not None:
            return _coerce_sliding_window_forecast_mapping(mapping_payload)

    raise TypeError("run_sliding_window_forecast must return a compatible tuple, a mapping, or SlidingWindowForecastResult.")


def apply_publication_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.1,
            "lines.markersize": 4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.prop_cycle": cycler(
                color=[
                    "#4E79A7",
                    "#F28E2B",
                    "#59A14F",
                    "#E15759",
                    "#76B7B2",
                    "#B07AA1",
                    "#EDC948",
                    "#9C755F",
                    "#BAB0AC",
                ]
            ),
        }
    )


def save_pub_figure(fig, output_path, dpi: int = 300, file_format: str = "pdf") -> Path:
    output_path = Path(output_path)
    file_format = str(file_format).lower().lstrip(".")
    if file_format not in {"pdf", "png"}:
        raise ValueError("file_format must be 'pdf' or 'png'")
    target_path = output_path.with_suffix(f".{file_format}")
    save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
    if file_format == "png":
        save_kwargs["dpi"] = dpi
    fig.savefig(target_path, format=file_format, **save_kwargs)
    return target_path


def style_current_figure(*, title: str | None = None):
    fig = plt.gcf()
    if title is not None and fig.axes:
        fig.axes[0].set_title(title)
    for ax in fig.axes:
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
    fig.tight_layout()
    return fig



def run_sliding_window_forecast(
    model,
    seq,
    device,
    *,
    config: SlidingWindowForecastConfig | None = None,
    duration: float = 12,
    slide_step: float = 12,
    include_truncated_final_window: bool = False,
    evaluation_range: SlidingWindowEvaluationRange | None = None,
    quantiles: tuple[float, float] = (2.5, 97.5),
    samples_per_batch: int = 1000,
    predict_b: bool | None = None,
    bg_cache_seq: Any | None = None,
    return_sim_count_matrix: bool = False,
    compute_mag_max: bool = False,
    return_sim_mag_max_matrix: bool = False,
):
    """
    Run a sliding window forecast over a sequence and compute statistics.

    Parameters:
        model: Forecasting model with a `.sample()` method.
        seq: Sequence object with `arrival_times` and `.get_subsequence()` method.
        device: Torch device to run the model on.
        duration: Forecast window length.
        slide_step: Step size to slide the forecast window.
        include_truncated_final_window: If True, include grid-aligned forecast
            windows that extend beyond the evaluation range end and clip them.
        evaluation_range: Relative-time target interval that limits forecast
            starts and ends without restricting the historical past context.
        quantiles: Lower and upper percentiles for prediction intervals.
        samples_per_batch: Number of simulated samples per forecast window.
        predict_b: Optional explicit b-prediction mode passed to model.sample.
        return_sim_count_matrix: If True, return matrix of simulated counts.
        compute_mag_max: If True, compute maximum magnitude per forecast sample.
        return_sim_mag_max_matrix: If True, return matrix of simulated max magnitudes.

    Returns:
        A SlidingWindowForecastResult object. It supports attribute access
        and legacy tuple unpacking.
    """
    if config is not None:
        duration = float(config.duration)
        slide_step = float(config.slide_step)
        include_truncated_final_window = bool(config.include_truncated_final_window)
        evaluation_range = config.evaluation_range
        quantiles = tuple(config.quantiles)
        samples_per_batch = int(config.samples_per_batch)
        predict_b = config.predict_b
        bg_cache_seq = config.bg_cache_seq
        return_sim_count_matrix = bool(config.return_sim_count_matrix)
        compute_mag_max = bool(config.compute_mag_max)
        return_sim_mag_max_matrix = bool(config.return_sim_mag_max_matrix)
        precomputed_counts = config.precomputed_counts
        precomputed_mag_max = config.precomputed_mag_max
    else:
        precomputed_counts = None
        precomputed_mag_max = None

    model.eval()
    _cache_background_sequence(model, bg_cache_seq)
    evaluation_range = _limit_range_to_background_cache(
        model,
        resolve_sliding_window_evaluation_range(seq, evaluation_range),
    )

    t_forecast_list, t_window_end_list = _build_sliding_window_bounds(
        seq,
        duration=duration,
        slide_step=slide_step,
        evaluation_range=evaluation_range,
        include_truncated_final_window=include_truncated_final_window,
    )
    precomputed_counts_arr = (
        np.asarray(precomputed_counts, dtype=np.int64).reshape(-1)
        if precomputed_counts is not None else None
    )
    precomputed_mag_max_arr = (
        np.asarray(precomputed_mag_max, dtype=np.float64).reshape(-1)
        if precomputed_mag_max is not None else None
    )
    if precomputed_counts_arr is not None and precomputed_counts_arr.shape[0] != t_forecast_list.shape[0]:
        raise ValueError("precomputed_counts length must match the number of sliding windows.")
    if precomputed_mag_max_arr is not None and precomputed_mag_max_arr.shape[0] != t_forecast_list.shape[0]:
        raise ValueError("precomputed_mag_max length must match the number of sliding windows.")

    counts_list: list[int] = []
    q_list: list[np.ndarray] = []
    mean_list: list[float] = []

    mag_max_list: list[float] = []
    q_mag_max_list: list[np.ndarray] = []
    mean_mag_max_list: list[float] = []

    sim_count_rows: list[np.ndarray] = []
    sim_mag_max_rows: list[np.ndarray] = []

    try:
        sample_params = inspect.signature(model.sample).parameters
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sample_params.values()
        )
        supports_predict_b = "predict_b" in sample_params or accepts_var_kwargs
        supports_bg_cache_seq = "bg_cache_seq" in sample_params or accepts_var_kwargs
    except Exception:
        supports_predict_b = False
        supports_bg_cache_seq = False

    past_start, _ = _sequence_observation_bounds(seq)
    for window_index, (t_forecast, t_end) in enumerate(
        zip(t_forecast_list, t_window_end_list, strict=True)
    ):

        # Extract past and observed subsequences
        past_seq = seq.get_subsequence(
            past_start,
            t_forecast,
            reset_t_nll_to_end=True,
        ).to(device)
        # ``Sequence`` stores float32 inter-times, so reconstructing a
        # subsequence can drift from the requested grid boundary. Keep the
        # model's forecast origin aligned with the evaluator's exact start.
        if hasattr(past_seq, "t_end"):
            past_seq.t_end = float(t_forecast)
        observed_seq = None
        if precomputed_counts_arr is None or (compute_mag_max and precomputed_mag_max_arr is None):
            observed_seq = seq.get_subsequence(t_forecast, t_end, reset_t_nll_to_end=True)

        # Generate forecast samples
        sample_kwargs = {
            "batch_size": samples_per_batch,
            "duration": (t_end - t_forecast),
            "past_seq": past_seq,
            "return_sequences": True,
        }
        if predict_b is not None and supports_predict_b:
            sample_kwargs["predict_b"] = bool(predict_b)
        if bg_cache_seq is not None and supports_bg_cache_seq:
            sample_kwargs["bg_cache_seq"] = bg_cache_seq
        with torch.inference_mode():
            forecasts = model.sample(**sample_kwargs)

        fc_counts = np.fromiter((len(fc) for fc in forecasts), dtype=np.int32)
        observed_count = (
            int(precomputed_counts_arr[window_index])
            if precomputed_counts_arr is not None
            else len(observed_seq)
        )
        counts_list.append(observed_count)
        q_list.append(np.percentile(fc_counts, quantiles))
        mean_list.append(float(fc_counts.mean()))
        if return_sim_count_matrix:
            sim_count_rows.append(fc_counts)

        if compute_mag_max:
            if precomputed_mag_max_arr is not None:
                obs_mag_max = float(precomputed_mag_max_arr[window_index])
            else:
                obs_mag = getattr(observed_seq, "mag", None)
                if obs_mag is None:
                    raise AttributeError("observed_seq is missing 'mag'; cannot compute observed max magnitude.")
                obs_mag_t = torch.as_tensor(obs_mag)
                obs_mag_max = float(obs_mag_t.max().item()) if obs_mag_t.numel() > 0 else np.nan

            fc_mag_max = np.fromiter(
                (fc.mag.max().item() if len(fc) > 0 else np.nan for fc in forecasts),
                dtype=np.float32
            )
            mag_max_list.append(obs_mag_max)
            finite_fc_mag_mask = np.isfinite(fc_mag_max)
            if np.any(finite_fc_mag_mask):
                q_mag_max_list.append(np.nanpercentile(fc_mag_max, quantiles))
                mean_mag_max_list.append(float(np.nanmean(fc_mag_max)))
            else:
                q_mag_max_list.append(np.full(len(quantiles), np.nan, dtype=np.float64))
                mean_mag_max_list.append(np.nan)
            if return_sim_mag_max_matrix:
                sim_mag_max_rows.append(fc_mag_max)

    t_forecast_arr = np.asarray(t_forecast_list, dtype=np.float64)
    t_window_end_arr = np.asarray(t_window_end_list, dtype=np.float64)
    counts_arr = np.array(counts_list)
    q_arr = np.vstack(q_list) if q_list else np.empty((0, 2), dtype=np.float64)
    mean_arr = np.array(mean_list)

    if compute_mag_max:
        mag_max_arr = np.array(mag_max_list)
        q_mag_max_arr = np.vstack(q_mag_max_list) if q_mag_max_list else np.empty((0, 2), dtype=np.float64)
        mean_mag_max_arr = np.array(mean_mag_max_list)

        sim_mag_max_matrix = (
            _stack_or_empty(sim_mag_max_rows, cols=int(samples_per_batch), dtype=np.float32)
            if return_sim_mag_max_matrix else None
        )

        sim_count_matrix = (
            _stack_or_empty(sim_count_rows, cols=int(samples_per_batch), dtype=np.int32)
            if return_sim_count_matrix else None
        )

        return SlidingWindowForecastResult(
            t_forecast=t_forecast_arr,
            counts=counts_arr,
            quantiles=q_arr,
            mean=mean_arr,
            sim_count_matrix=sim_count_matrix,
            mag_max=mag_max_arr,
            mag_max_quantiles=q_mag_max_arr,
            mag_max_mean=mean_mag_max_arr,
            sim_mag_max_matrix=sim_mag_max_matrix,
            t_window_end=t_window_end_arr,
        )

    sim_count_matrix = (
        _stack_or_empty(sim_count_rows, cols=int(samples_per_batch), dtype=np.int32)
        if return_sim_count_matrix else None
    )

    return SlidingWindowForecastResult(
        t_forecast=t_forecast_arr,
        counts=counts_arr,
        quantiles=q_arr,
        mean=mean_arr,
        sim_count_matrix=sim_count_matrix,
        t_window_end=t_window_end_arr,
    )

def compute_mean_nb_log_prob(obs_counts, sim_count_matrix, *, eps: float = 1e-10):
    """
    Fit a per-bin negative binomial (method of moments) and compute
    LP_NB = mean_i log P(N_obs^i | theta_hat_i).
    """
    obs = np.asarray(obs_counts, dtype=np.int64).reshape(-1)
    sim = np.asarray(sim_count_matrix, dtype=np.float64)

    if sim.ndim != 2:
        raise ValueError("sim_count_matrix must be a 2D array with shape (n_bins, n_samples).")
    if sim.shape[0] != obs.shape[0]:
        raise ValueError("obs_counts and sim_count_matrix must have the same number of bins.")
    if sim.shape[1] == 0:
        raise ValueError("sim_count_matrix must contain at least one simulated sample per bin.")

    mu = np.mean(sim, axis=1)
    var = np.var(sim, axis=1, ddof=1) if sim.shape[1] > 1 else mu.copy()

    log_prob_bins = np.full(obs.shape[0], np.nan, dtype=np.float64)

    zero_mu = mu <= eps
    obs_zero = obs == 0
    log_prob_bins[zero_mu & obs_zero] = 0.0
    log_prob_bins[zero_mu & (~obs_zero)] = -np.inf

    active = ~zero_mu
    if np.any(active):
        mu_active = np.clip(mu[active], eps, None)
        var_active = np.maximum(var[active], mu_active + eps)
        total_count = np.clip((mu_active * mu_active) / (var_active - mu_active), eps, 1e12)

        k_t = torch.as_tensor(obs[active], dtype=torch.float64)
        mu_t = torch.as_tensor(mu_active, dtype=torch.float64)
        total_count_t = torch.as_tensor(total_count, dtype=torch.float64)

        # NB parameterized by mean mu and total_count r:
        # log PMF = lgamma(k+r)-lgamma(r)-lgamma(k+1)
        #           + r*log(r/(r+mu)) + k*log(mu/(r+mu))
        log_prob_active = (
            torch.lgamma(k_t + total_count_t)
            - torch.lgamma(total_count_t)
            - torch.lgamma(k_t + 1.0)
            + total_count_t * (torch.log(total_count_t) - torch.log(total_count_t + mu_t))
            + k_t * (torch.log(mu_t) - torch.log(total_count_t + mu_t))
        )
        log_prob_bins[active] = log_prob_active.cpu().numpy()

    lp_nb = float(np.mean(log_prob_bins))
    return lp_nb, log_prob_bins


def compute_mean_crps(obs_counts, sim_count_matrix):
    """
    Compute per-bin and mean CRPS for empirical forecast samples.

    For each bin i with observation y_i and simulated counts x_{i,1:m},
    CRPS_i = E|X - y_i| - 0.5 * E|X - X'|.
    """
    obs = np.asarray(obs_counts, dtype=np.float64).reshape(-1)
    sim = np.asarray(sim_count_matrix, dtype=np.float64)

    if sim.ndim != 2:
        raise ValueError("sim_count_matrix must be a 2D array with shape (n_bins, n_samples).")
    if sim.shape[0] != obs.shape[0]:
        raise ValueError("obs_counts and sim_count_matrix must have the same number of bins.")
    if sim.shape[1] == 0:
        raise ValueError("sim_count_matrix must contain at least one simulated sample per bin.")

    n_samples = int(sim.shape[1])
    abs_to_obs = np.mean(np.abs(sim - obs[:, None]), axis=1)

    sim_sorted = np.sort(sim, axis=1)
    order = np.arange(1, n_samples + 1, dtype=np.float64)
    coeff = (2.0 * order - n_samples - 1.0)[None, :]
    half_pairwise_abs = np.sum(coeff * sim_sorted, axis=1) / float(n_samples * n_samples)

    crps_bins = abs_to_obs - half_pairwise_abs
    crps = float(np.mean(crps_bins))
    return crps, crps_bins


def to_absolute_time_axis(rel_times, base_ts, freq_td):
    rel = np.asarray(rel_times, dtype=float).reshape(-1)
    delta = pd.to_timedelta(rel * freq_td.total_seconds(), unit="s")
    return pd.DatetimeIndex(pd.Timestamp(base_ts) + delta)


def format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelrotation=0)


def format_days_since_axis(ax, *, reference_ts):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    reference_num = mdates.date2num(pd.Timestamp(reference_ts).to_pydatetime())

    def _fmt_days_since(value, _position):
        days_since = float(value - reference_num)
        if np.isclose(days_since, round(days_since)):
            return f"{int(round(days_since))}"
        return f"{days_since:.1f}"

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_days_since))
    ax.tick_params(axis="x", labelrotation=0)


def _format_compact_thousands(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value >= 1000.0:
        k_value = value / 1000.0
        if np.isclose(k_value, round(k_value)):
            return f"{int(round(k_value))}k"
        return f"{k_value:.1f}k"
    if np.isclose(value, round(value)):
        return f"{int(round(value))}"
    return f"{value:g}"


def style_axes(
    ax,
    *,
    xlabel: str = "Days since start",
    ylabel: str | None = None,
    integer_y: bool = False,
    compact_y_thousands: bool = False,
):
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if integer_y:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def resolve_catalog_time_reference(catalog_ds):
    md = getattr(catalog_ds, "metadata", {})
    if isinstance(md, dict):
        start_val = md.get("start_ts")
        freq_val = md.get("freq")
    else:
        start_val = getattr(md, "start_ts", None)
        freq_val = getattr(md, "freq", None)

    if start_val is None or freq_val is None:
        raise KeyError("catalog_ds.metadata must contain 'start_ts' and 'freq'.")

    return pd.to_datetime(start_val), pd.to_timedelta(freq_val)


def compute_display_upper_cap(q_low, q_high, counts, mean, pct: float = 80):
    q_ref = np.asarray(q_high, dtype=float).reshape(-1)
    robust_cap = float(np.nanpercentile(q_ref, pct))
    anchor = float(np.nanmax(np.concatenate([q_low.reshape(-1), counts.reshape(-1), mean.reshape(-1)])))
    return max(robust_cap, anchor * 1.10, 1.0)


def infer_step_timedelta(ts_index, fallback_td):
    ts_index = pd.DatetimeIndex(ts_index)
    fallback_td = pd.to_timedelta(fallback_td)
    if len(ts_index) >= 2:
        deltas = pd.Series(ts_index).diff().dropna()
        deltas = deltas[deltas > pd.Timedelta(0)]
        if not deltas.empty:
            return deltas.median()
    return fallback_td if fallback_td > pd.Timedelta(0) else pd.Timedelta(days=1)


def resolve_time_bounds(ts_index, fallback_td):
    ts_index = pd.DatetimeIndex(ts_index)
    step_td = infer_step_timedelta(ts_index, fallback_td)

    if len(ts_index) == 0:
        now = pd.Timestamp.today().normalize()
        return now, now + step_td, step_td

    left = ts_index.min() - step_td * 0.35
    right = ts_index.max() + step_td * 0.65
    if right <= left:
        right = left + step_td
    return left, right, step_td


def build_post_step(ts_index, values, step_td):
    ts_index = pd.DatetimeIndex(ts_index)
    values = np.asarray(values, dtype=float).reshape(-1)

    if len(ts_index) == 0:
        return ts_index, values
    if len(ts_index) == 1:
        ts_step = pd.DatetimeIndex([ts_index[0], ts_index[0] + step_td])
        v_step = np.array([values[0], values[0]], dtype=float)
        return ts_step, v_step

    ts_step = ts_index.append(pd.DatetimeIndex([ts_index[-1] + step_td]))
    v_step = np.r_[values, values[-1]]
    return ts_step, v_step


def resolve_view_mode(config_value, *, auto_use_zoom: bool):
    mode = str(config_value).strip().lower()
    if mode not in {"zoomed", "full", "auto"}:
        mode = "auto"
    if mode == "auto":
        return "zoomed" if auto_use_zoom else "full"
    return mode


LEGACY_FULL_RANGE_SLIDING_CACHE_VERSION = 2
RANGE_AWARE_SLIDING_CACHE_VERSION = 4
SLIDING_CACHE_VERSION = RANGE_AWARE_SLIDING_CACHE_VERSION
SUPPORTED_SLIDING_CACHE_VERSIONS = frozenset(
    {
        LEGACY_FULL_RANGE_SLIDING_CACHE_VERSION,
        RANGE_AWARE_SLIDING_CACHE_VERSION,
    }
)
DEFAULT_SLIDING_CACHE_FILENAME = "sliding_window_cache.npz"


def _cache_predict_b_mode(predict_b: bool | None) -> int:
    if predict_b is None:
        return -1
    return int(bool(predict_b))


def build_default_sliding_cache_filename(
    metadata: Mapping[str, Any],
    *,
    prefix: str = "sliding_window_cache",
) -> str:
    """Build a range-specific cache filename from validated metadata."""
    range_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "-",
        str(metadata.get("evaluation_name", "full")).strip() or "custom",
    ).strip("-") or "custom"
    identity_keys = (
        "duration",
        "slide_step",
        "include_truncated_final_window",
        "evaluation_name",
        "evaluation_start",
        "evaluation_end",
        "evaluation_grid_anchor",
        "quantile_low",
        "quantile_high",
        "samples_per_batch",
        "predict_b_mode",
        "sampling_seed",
        "seq_start",
        "seq_end",
        "seq_t_start",
        "seq_t_end",
    )
    identity = {
        key: metadata[key]
        for key in identity_keys
        if key in metadata
    }
    encoded_identity = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded_identity).hexdigest()[:12]
    return f"{prefix}_{range_name}_{digest}.npz"


def build_sliding_cache_metadata(
    seq,
    *,
    duration: float,
    slide_step: float,
    quantiles: tuple[float, float],
    samples_per_batch: int,
    include_truncated_final_window: bool = False,
    evaluation_range: SlidingWindowEvaluationRange | None = None,
    predict_b: bool | None = None,
    sampling_seed: int | None = None,
):
    resolved_range = resolve_sliding_window_evaluation_range(seq, evaluation_range)
    sequence_t_start, sequence_t_end = _sequence_observation_bounds(seq)
    return {
        "cache_version": int(SLIDING_CACHE_VERSION),
        "duration": float(duration),
        "slide_step": float(slide_step),
        "include_truncated_final_window": int(bool(include_truncated_final_window)),
        "evaluation_name": str(resolved_range.name),
        "evaluation_start": float(resolved_range.start),
        "evaluation_end": float(resolved_range.end),
        "evaluation_grid_anchor": float(resolved_range.grid_anchor),
        "quantile_low": float(quantiles[0]),
        "quantile_high": float(quantiles[1]),
        "samples_per_batch": int(samples_per_batch),
        "predict_b_mode": int(_cache_predict_b_mode(predict_b)),
        "sampling_seed": -1 if sampling_seed is None else int(sampling_seed),
        "seq_start": float(seq.arrival_times[0].item()),
        "seq_end": float(seq.arrival_times[-1].item()),
        "seq_t_start": sequence_t_start,
        "seq_t_end": sequence_t_end,
    }


def save_sliding_window_cache(
    cache_path,
    *,
    metadata,
    t_forecast_list,
    t_window_end_list=None,
    counts_list,
    q_list,
    mean_list,
    sim_count_matrix,
    mag_max=None,
    mag_max_quantiles=None,
    mag_max_mean=None,
    sim_mag_max_matrix=None,
):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "cache_version": np.int64(metadata["cache_version"]),
        "duration": np.float64(metadata["duration"]),
        "slide_step": np.float64(metadata["slide_step"]),
        "include_truncated_final_window": np.int8(
            metadata.get("include_truncated_final_window", 0)
        ),
        "evaluation_name": np.asarray(metadata.get("evaluation_name", "full")),
        "evaluation_start": np.float64(metadata["evaluation_start"]),
        "evaluation_end": np.float64(metadata["evaluation_end"]),
        "evaluation_grid_anchor": np.float64(metadata["evaluation_grid_anchor"]),
        "quantile_low": np.float64(metadata["quantile_low"]),
        "quantile_high": np.float64(metadata["quantile_high"]),
        "samples_per_batch": np.int64(metadata["samples_per_batch"]),
        "predict_b_mode": np.int64(metadata["predict_b_mode"]),
        "sampling_seed": np.int64(metadata.get("sampling_seed", -1)),
        "seq_start": np.float64(metadata["seq_start"]),
        "seq_end": np.float64(metadata["seq_end"]),
        "seq_t_start": np.float64(metadata["seq_t_start"]),
        "seq_t_end": np.float64(metadata["seq_t_end"]),
        "t_forecast_list": np.asarray(t_forecast_list, dtype=np.float64),
        "counts_list": np.asarray(counts_list, dtype=np.int64),
        "q_list": np.asarray(q_list, dtype=np.float64),
        "mean_list": np.asarray(mean_list, dtype=np.float64),
        "sim_count_matrix": np.asarray(sim_count_matrix, dtype=np.int32),
    }

    if t_window_end_list is not None:
        payload["t_window_end_list"] = np.asarray(t_window_end_list, dtype=np.float64)

    mag_payload = (mag_max, mag_max_quantiles, mag_max_mean, sim_mag_max_matrix)
    if any(item is not None for item in mag_payload):
        if mag_max is None or mag_max_quantiles is None or mag_max_mean is None:
            raise ValueError(
                "mag_max, mag_max_quantiles, and mag_max_mean must be saved together."
            )
        payload["mag_max"] = np.asarray(mag_max, dtype=np.float64)
        payload["mag_max_quantiles"] = np.asarray(mag_max_quantiles, dtype=np.float64)
        payload["mag_max_mean"] = np.asarray(mag_max_mean, dtype=np.float64)
        if sim_mag_max_matrix is not None:
            payload["sim_mag_max_matrix"] = np.asarray(
                sim_mag_max_matrix,
                dtype=np.float32,
            )

    np.savez_compressed(cache_path, **payload)
    return cache_path


_SLIDING_CACHE_PAYLOAD_KEYS = {
    "t_forecast_list",
    "counts_list",
    "q_list",
    "mean_list",
    "sim_count_matrix",
}
_SLIDING_CACHE_V2_REQUIRED_KEYS = _SLIDING_CACHE_PAYLOAD_KEYS | {
    "cache_version",
    "duration",
    "slide_step",
    "quantile_low",
    "quantile_high",
    "samples_per_batch",
    "predict_b_mode",
    "seq_start",
    "seq_end",
}
_SLIDING_CACHE_V4_REQUIRED_KEYS = _SLIDING_CACHE_V2_REQUIRED_KEYS | {
    "include_truncated_final_window",
    "evaluation_name",
    "evaluation_start",
    "evaluation_end",
    "evaluation_grid_anchor",
    "sampling_seed",
    "seq_t_start",
    "seq_t_end",
}
_SLIDING_CACHE_V2_FLOAT_KEYS = (
    "duration",
    "slide_step",
    "quantile_low",
    "quantile_high",
    "seq_start",
    "seq_end",
)
_SLIDING_CACHE_V4_FLOAT_KEYS = _SLIDING_CACHE_V2_FLOAT_KEYS + (
    "evaluation_start",
    "evaluation_end",
    "evaluation_grid_anchor",
    "seq_t_start",
    "seq_t_end",
)
_SLIDING_CACHE_MAG_KEYS = {
    "mag_max",
    "mag_max_quantiles",
    "mag_max_mean",
}
_SLIDING_CACHE_OPTIONAL_MAG_KEYS = _SLIDING_CACHE_MAG_KEYS | {
    "sim_mag_max_matrix",
}


def _cache_scalar(data, key: str):
    return np.asarray(data[key]).item()


def _cache_metadata_matches(
    data,
    metadata: Mapping[str, Any],
    *,
    required_keys: set[str],
    int_keys: tuple[str, ...] = (),
    float_keys: tuple[str, ...] = (),
    str_keys: tuple[str, ...] = (),
    legacy_full_range_only: bool = False,
    atol: float = 1e-9,
) -> bool:
    if not required_keys.issubset(data.files):
        return False

    if legacy_full_range_only:
        # v2 has no range/truncation/seed fields; only reuse it for full windows.
        if str(metadata.get("evaluation_name", "full")) != "full":
            return False
        if int(metadata.get("include_truncated_final_window", 0)) != 0:
            return False

    for key in int_keys:
        if int(_cache_scalar(data, key)) != int(metadata[key]):
            return False
    for key in str_keys:
        if str(_cache_scalar(data, key)) != str(metadata[key]):
            return False
    for key in float_keys:
        cache_val = float(_cache_scalar(data, key))
        if not np.isclose(cache_val, float(metadata[key]), atol=atol, rtol=0.0):
            return False
    return True


def _cache_matches_supported_schema(
    data,
    metadata: Mapping[str, Any],
    *,
    atol: float,
) -> bool:
    version = int(_cache_scalar(data, "cache_version"))
    if version == LEGACY_FULL_RANGE_SLIDING_CACHE_VERSION:
        return _cache_metadata_matches(
            data,
            metadata,
            required_keys=_SLIDING_CACHE_V2_REQUIRED_KEYS,
            int_keys=("samples_per_batch", "predict_b_mode"),
            float_keys=_SLIDING_CACHE_V2_FLOAT_KEYS,
            legacy_full_range_only=True,
            atol=atol,
        )
    if version == RANGE_AWARE_SLIDING_CACHE_VERSION:
        return _cache_metadata_matches(
            data,
            metadata,
            required_keys=_SLIDING_CACHE_V4_REQUIRED_KEYS,
            int_keys=(
                "samples_per_batch",
                "predict_b_mode",
                "sampling_seed",
                "include_truncated_final_window",
            ),
            str_keys=("evaluation_name",),
            float_keys=_SLIDING_CACHE_V4_FLOAT_KEYS,
            atol=atol,
        )
    return False


def _load_sliding_cache_payload(data) -> dict[str, Any] | None:
    payload: dict[str, Any] = {
        "t_forecast": np.asarray(data["t_forecast_list"]),
        "counts": np.asarray(data["counts_list"]),
        "quantiles": np.asarray(data["q_list"]),
        "mean": np.asarray(data["mean_list"]),
        "sim_count_matrix": np.asarray(data["sim_count_matrix"]),
    }

    if "t_window_end_list" in data.files:
        payload["t_window_end"] = np.asarray(data["t_window_end_list"])

    mag_keys_in_cache = _SLIDING_CACHE_OPTIONAL_MAG_KEYS.intersection(data.files)
    if mag_keys_in_cache:
        if not _SLIDING_CACHE_MAG_KEYS.issubset(data.files):
            return None
        payload["mag_max"] = np.asarray(data["mag_max"])
        payload["mag_max_quantiles"] = np.asarray(data["mag_max_quantiles"])
        payload["mag_max_mean"] = np.asarray(data["mag_max_mean"])
        if "sim_mag_max_matrix" in data.files:
            payload["sim_mag_max_matrix"] = np.asarray(data["sim_mag_max_matrix"])

    return payload


def load_sliding_window_cache_if_compatible(cache_path, *, metadata, atol: float = 1e-9):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if "cache_version" not in data.files:
                return None

            if not _cache_matches_supported_schema(data, metadata, atol=atol):
                return None

            payload = _load_sliding_cache_payload(data)
            if payload is None:
                return None
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        return None

    try:
        return _coerce_sliding_window_forecast_mapping(payload)
    except (TypeError, ValueError):
        return None


def _precompute_observed_window_stats(seq, t_forecast_list, t_window_end_list):
    event_times = torch.as_tensor(seq.arrival_times).detach().cpu().numpy()
    event_times = np.asarray(event_times, dtype=np.float64)
    starts = np.asarray(t_forecast_list, dtype=np.float64)
    ends = np.asarray(t_window_end_list, dtype=np.float64)
    if starts.shape != ends.shape:
        raise ValueError("t_forecast_list and t_window_end_list must have the same shape.")

    left_idx = np.searchsorted(event_times, starts, side="left")
    right_idx = np.searchsorted(event_times, ends, side="right")
    counts = (right_idx - left_idx).astype(np.int64, copy=False)

    mag_values = getattr(seq, "mag", None)
    if mag_values is None:
        return counts, None

    mag_arr = torch.as_tensor(mag_values).detach().cpu().numpy()
    if mag_arr.ndim != 1 or mag_arr.shape[0] != event_times.shape[0]:
        return counts, None

    mag_max = np.full(starts.shape[0], np.nan, dtype=np.float64)
    for index, (left, right) in enumerate(zip(left_idx, right_idx, strict=True)):
        if right > left:
            mag_max[index] = float(np.nanmax(mag_arr[left:right]))
    return counts, mag_max


def evaluate_sliding_window_forecast_plots(
    *,
    model,
    seq,
    device,
    catalog_ds,
    checkpoint_dir,
    sliding_duration: float,
    sliding_step: float,
    sliding_quantiles: tuple[float, float],
    samples_per_batch: int,
    include_truncated_final_window: bool = False,
    evaluation_range: SlidingWindowEvaluationRange | None = None,
    predict_b: bool | None = None,
    bg_cache_seq: Any | None = None,
    sliding_view_mode: str = "auto",
    load_sliding_cache: bool = True,
    force_recompute_sliding: bool = False,
    sliding_cache_filename: str | None = None,
    sampling_seed: int | None = None,
    plot_colors: Mapping[str, str] | None = None,
    plot_figsizes: Mapping[str, tuple[float, float]] | None = None,
    save_plots: bool = True,
    plot_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    colors = dict(DEFAULT_PLOT_COLORS)
    if plot_colors is not None:
        colors.update(plot_colors)
    figsizes = dict(DEFAULT_SLIDING_PLOT_FIGSIZES)
    if plot_figsizes is not None:
        for name, figsize in plot_figsizes.items():
            if name not in figsizes:
                available = ", ".join(sorted(figsizes))
                raise KeyError(f"Unknown plot_figsizes key {name!r}. Available keys: {available}")
            if len(figsize) != 2:
                raise ValueError(f"plot_figsizes[{name!r}] must be a (width, height) tuple.")
            width, height = float(figsize[0]), float(figsize[1])
            if width <= 0 or height <= 0:
                raise ValueError(f"plot_figsizes[{name!r}] values must be positive.")
            figsizes[name] = (width, height)

    _cache_background_sequence(model, bg_cache_seq)
    resolved_range = resolve_sliding_window_evaluation_range(
        seq,
        evaluation_range,
    )
    resolved_range = _limit_range_to_background_cache(model, resolved_range)
    cache_meta = build_sliding_cache_metadata(
        seq,
        duration=sliding_duration,
        slide_step=sliding_step,
        include_truncated_final_window=include_truncated_final_window,
        evaluation_range=resolved_range,
        quantiles=sliding_quantiles,
        samples_per_batch=samples_per_batch,
        predict_b=predict_b,
        sampling_seed=sampling_seed,
    )
    if sliding_cache_filename is None:
        sliding_cache_filename = build_default_sliding_cache_filename(cache_meta)
    checkpoint_dir = Path(checkpoint_dir)
    sliding_cache_path = checkpoint_dir / str(sliding_cache_filename)

    loaded_from_cache = False
    cached_payload = None
    if load_sliding_cache and not force_recompute_sliding:
        cached_payload = load_sliding_window_cache_if_compatible(
            sliding_cache_path,
            metadata=cache_meta,
        )

    if cached_payload is not None:
        sliding_result = _coerce_sliding_window_forecast_result(cached_payload)
        loaded_from_cache = True
    else:
        t_forecast_preview, t_window_end_preview = _build_sliding_window_bounds(
            seq,
            duration=sliding_duration,
            slide_step=sliding_step,
            evaluation_range=resolved_range,
            include_truncated_final_window=include_truncated_final_window,
        )
        observed_counts, observed_mag_max = _precompute_observed_window_stats(
            seq,
            t_forecast_preview,
            t_window_end_preview,
        )
        sliding_result = run_sliding_window_forecast(
            model=model,
            seq=seq,
            device=device,
            config=SlidingWindowForecastConfig(
                duration=sliding_duration,
                slide_step=sliding_step,
                include_truncated_final_window=include_truncated_final_window,
                evaluation_range=resolved_range,
                quantiles=sliding_quantiles,
                samples_per_batch=samples_per_batch,
                predict_b=predict_b,
                bg_cache_seq=bg_cache_seq,
                return_sim_count_matrix=True,
                compute_mag_max=True,
                return_sim_mag_max_matrix=True,
                precomputed_counts=observed_counts,
                precomputed_mag_max=observed_mag_max,
            ),
        )
        if load_sliding_cache:
            save_sliding_window_cache(
                sliding_cache_path,
                metadata=cache_meta,
                t_forecast_list=sliding_result.t_forecast,
                t_window_end_list=sliding_result.t_window_end,
                counts_list=sliding_result.counts,
                q_list=sliding_result.quantiles,
                mean_list=sliding_result.mean,
                sim_count_matrix=sliding_result.sim_count_matrix,
                mag_max=sliding_result.mag_max,
                mag_max_quantiles=sliding_result.mag_max_quantiles,
                mag_max_mean=sliding_result.mag_max_mean,
                sim_mag_max_matrix=sliding_result.sim_mag_max_matrix,
            )

    t_forecast_list = sliding_result.t_forecast
    result_window_ends = getattr(sliding_result, "t_window_end", None)
    if result_window_ends is None:
        t_window_end_list = np.minimum(
            np.asarray(t_forecast_list, dtype=np.float64) + float(sliding_duration),
            float(resolved_range.end),
        )
    else:
        t_window_end_list = np.asarray(result_window_ends, dtype=np.float64)
        if t_window_end_list.shape != np.asarray(t_forecast_list).shape:
            raise ValueError(
                "Sliding result t_window_end must match t_forecast shape."
            )
    counts_list = sliding_result.counts
    q_list = sliding_result.quantiles
    mean_list = sliding_result.mean
    sim_count_matrix = sliding_result.sim_count_matrix
    mag_max_mean = getattr(sliding_result, "mag_max_mean", None)
    mag_max = getattr(sliding_result, "mag_max", None)

    mag_max_mae = None
    if mag_max_mean is not None and mag_max is not None:
        mag_max_arr = np.asarray(mag_max, dtype=np.float64)
        mag_max_mean_arr = np.asarray(mag_max_mean, dtype=np.float64)
        if mag_max_arr.shape == mag_max_mean_arr.shape and mag_max_arr.size > 0:
            finite_mask_for_mae = np.isfinite(mag_max_arr) & np.isfinite(mag_max_mean_arr)
            if np.any(finite_mask_for_mae):
                mag_max_mae = float(np.mean(np.abs(mag_max_arr[finite_mask_for_mae] - mag_max_mean_arr[finite_mask_for_mae])))
            else:
                mag_max_mae = None

    if len(t_forecast_list) == 0:
        return {
            "status": "empty",
            "message": "No sliding windows available with current duration/step settings.",
            "evaluation_range": {
                "name": resolved_range.name,
                "start": resolved_range.start,
                "end": resolved_range.end,
                "grid_anchor": resolved_range.grid_anchor,
            },
            "t_forecast_list": t_forecast_list,
            "t_window_end_list": t_window_end_list,
            "sliding_cache_path": str(sliding_cache_path),
            "sliding_loaded_from_cache": loaded_from_cache,
        }

    q_low, q_high = q_list[:, 0], q_list[:, 1]
    covered = (counts_list >= q_low) & (counts_list <= q_high)
    coverage = float(covered.mean())
    mae = float(np.mean(np.abs(counts_list - mean_list)))
    rmse = float(np.sqrt(np.mean((counts_list - mean_list) ** 2)))
    w95 = float(np.mean(q_high - q_low))
    lp_nb, _ = compute_mean_nb_log_prob(counts_list, sim_count_matrix)
    crps, _ = compute_mean_crps(counts_list, sim_count_matrix)

    result_payload = {
        "status": "ok",
        "evaluation_range": {
            "name": resolved_range.name,
            "start": resolved_range.start,
            "end": resolved_range.end,
            "grid_anchor": resolved_range.grid_anchor,
        },
        "t_forecast_list": t_forecast_list,
        "t_window_end_list": t_window_end_list,
        "counts_list": counts_list,
        "q_list": q_list,
        "mean_list": mean_list,
        "coverage": coverage,
        "mae": mae,
        "rmse": rmse,
        "crps": crps,
        "w95": w95,
        "lp_nb": lp_nb,
        "mag_max_mae": mag_max_mae,
        "sliding_cache_path": str(sliding_cache_path),
        "sliding_loaded_from_cache": loaded_from_cache,
    }

    if not save_plots:
        return result_payload

    plot_output_dir = Path(plot_output_dir) if plot_output_dir is not None else checkpoint_dir
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    base_start_ts, freq_td_local = resolve_catalog_time_reference(catalog_ds)
    base_start_ts = pd.Timestamp(base_start_ts)
    time_xlabel = f"Days since {base_start_ts.strftime('%Y-%m-%d')}"
    seq_start_rel = float(getattr(seq, "t_start", 0.0))
    seq_start_ts = base_start_ts + pd.to_timedelta(
        seq_start_rel * freq_td_local.total_seconds(),
        unit="s",
    )
    t_forecast_ts = to_absolute_time_axis(t_forecast_list, seq_start_ts, freq_td_local)

    default_step_td = pd.to_timedelta(
        max(float(sliding_step), 1e-6) * freq_td_local.total_seconds(),
        unit="s",
    )
    plot_x_left, plot_x_right, inferred_step_td = resolve_time_bounds(
        t_forecast_ts,
        default_step_td,
    )

    # 1) Counts over time
    display_cap = compute_display_upper_cap(q_low, q_high, counts_list, mean_list, pct=80)
    base_bottom = min(0.0, float(np.nanmin(q_low)) * 1.05)
    zoom_top = display_cap * 1.05
    full_top = max(float(np.nanmax(q_high)) * 1.05, zoom_top)
    counts_mode = resolve_view_mode(sliding_view_mode, auto_use_zoom=(full_top > zoom_top * 1.8))

    if counts_mode == "zoomed":
        q_high_plot = np.minimum(q_high, display_cap)
        clipped_mask = q_high > display_cap
        counts_ylim_top = zoom_top
        counts_title = "Counts: true vs forecast (zoomed)"
        cap_note = f" | clipped={int(np.count_nonzero(clipped_mask))}" if np.any(clipped_mask) else ""
    else:
        q_high_plot = q_high
        clipped_mask = np.zeros_like(q_high, dtype=bool)
        counts_ylim_top = full_top
        counts_title = "Counts: true vs forecast (full range)"
        cap_note = ""

    fig, ax = plt.subplots(figsize=figsizes["counts"])
    ax.plot(
        t_forecast_ts,
        counts_list,
        label="true counts",
        color=colors["true"],
        linewidth=1.1,
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        markeredgewidth=0.8,
    )
    ax.plot(
        t_forecast_ts,
        mean_list,
        label="forecast mean",
        color=colors["mean"],
        linewidth=1.1,
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        markeredgewidth=0.8,
    )
    ax.fill_between(
        t_forecast_ts,
        q_low,
        q_high_plot,
        alpha=0.14,
        color=colors["pi"],
        label="forecast PI (2.5-97.5)",
    )
    if np.any(clipped_mask):
        ax.scatter(
            t_forecast_ts[clipped_mask],
            np.full(int(clipped_mask.sum()), display_cap),
            marker="^",
            s=20,
            color=colors["pi"],
            label="PI upper clipped (display)",
            zorder=4,
        )
    style_axes(ax, xlabel=time_xlabel, ylabel="count in next window", integer_y=True)
    format_days_since_axis(ax, reference_ts=base_start_ts)
    ax.set_xlim(plot_x_left, plot_x_right)
    ax.set_ylim(base_bottom, counts_ylim_top)
    ax.set_title(counts_title)
    ax.text(
        0.01,
        0.02,
        f"Coverage: {coverage:.2%} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | CRPS: {crps:.2f} | W95: {w95:.2f} | LP_NB: {lp_nb:.4f}{cap_note}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    save_pub_figure(fig, plot_output_dir / "forecast_counts_over_time.png")
    plt.show()

    # 2) Error over time
    err = counts_list - mean_list
    bias = float(np.mean(err))
    err_abs = np.abs(err)
    err_zoom_lim = max(float(np.nanpercentile(err_abs, 80)) * 1.2, 1.0)
    err_full_lim = max(float(np.max(err_abs)) * 1.05, err_zoom_lim)
    error_mode = resolve_view_mode(sliding_view_mode, auto_use_zoom=(err_full_lim > err_zoom_lim * 1.8))

    clip_lim = err_zoom_lim if error_mode == "zoomed" else None
    if clip_lim is None:
        err_plot = err
        clip_hi = np.zeros(err.shape, dtype=bool)
        clip_lo = np.zeros(err.shape, dtype=bool)
        err_title = "Forecast error over time (full range)"
        err_ylim = err_full_lim
    else:
        err_plot = np.clip(err, -clip_lim, clip_lim)
        clip_hi = err > clip_lim
        clip_lo = err < -clip_lim
        err_title = "Forecast error over time (zoomed)"
        err_ylim = clip_lim

    fig, ax = plt.subplots(figsize=figsizes["error"])
    ax.plot(
        t_forecast_ts,
        err_plot,
        label="true - mean",
        color=colors["err"],
        linewidth=1.0,
        marker="o",
        markersize=2.6,
        markerfacecolor="white",
        markeredgewidth=0.8,
    )
    if clip_lim is not None and np.any(clip_hi):
        ax.scatter(
            t_forecast_ts[clip_hi],
            np.full(int(clip_hi.sum()), clip_lim),
            marker="^",
            s=20,
            color=colors["err"],
            zorder=4,
        )
    if clip_lim is not None and np.any(clip_lo):
        ax.scatter(
            t_forecast_ts[clip_lo],
            np.full(int(clip_lo.sum()), -clip_lim),
            marker="v",
            s=20,
            color=colors["err"],
            zorder=4,
        )

    clipped_err_points = int(np.count_nonzero(clip_hi | clip_lo))
    clip_note = f" | clipped={clipped_err_points}" if clipped_err_points > 0 else ""
    ax.axhline(0, linewidth=0.9, color="gray", linestyle="--")
    style_axes(ax, xlabel=time_xlabel, ylabel="error")
    format_days_since_axis(ax, reference_ts=base_start_ts)
    ax.set_xlim(plot_x_left, plot_x_right)
    ax.set_ylim(-err_ylim, err_ylim)
    ax.set_title(err_title)
    ax.text(
        0.01,
        0.02,
        f"Bias: {bias:.2f} | RMSE: {rmse:.2f}{clip_note}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_pub_figure(fig, plot_output_dir / "forecast_error_over_time.png")
    plt.show()

    # 3) Coverage over time
    covered_i = covered.astype(int)
    coverage_t_step, coverage_v_step = build_post_step(t_forecast_ts, covered_i, inferred_step_td)
    fig, ax = plt.subplots(figsize=figsizes["coverage"])
    ax.step(
        coverage_t_step,
        coverage_v_step,
        where="post",
        label=f"covered (avg={coverage:.3f})",
        color=colors["covered"],
        linewidth=1.0,
    )
    ax.scatter(t_forecast_ts, covered_i, s=16, color=colors["covered"], zorder=3)
    ax.fill_between(
        coverage_t_step,
        np.zeros_like(coverage_v_step),
        coverage_v_step,
        step="post",
        alpha=0.12,
        color=colors["covered"],
    )
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_xlim(plot_x_left, plot_x_right)
    ax.set_title("Coverage over time")
    style_axes(ax, xlabel=time_xlabel, ylabel="in PI")
    format_days_since_axis(ax, reference_ts=base_start_ts)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_pub_figure(fig, plot_output_dir / "forecast_pi_coverage_over_time.png")
    plt.show()

    # 4) Observed vs forecast scatter
    out_of_lower = counts_list < q_low
    out_of_upper = counts_list > q_high
    scatter_min = min(float(counts_list.min()), float(mean_list.min()))
    scatter_max = max(float(counts_list.max()), float(mean_list.max()))
    scatter_zoom_max = float(np.nanpercentile(np.concatenate([counts_list, mean_list]), 85)) * 1.10
    scatter_zoom_max = max(scatter_zoom_max, 1.0)
    scatter_mode = resolve_view_mode(
        sliding_view_mode,
        auto_use_zoom=(scatter_max > scatter_zoom_max * 1.8),
    )

    def draw_scatter(ax, title, xlim=None, ylim=None):
        ax.scatter(
            counts_list,
            mean_list,
            color=colors["true"],
            alpha=0.7,
            s=28,
            label="true vs mean",
        )
        ax.scatter(
            counts_list[out_of_lower],
            mean_list[out_of_lower],
            color=colors["below"],
            marker="v",
            s=40,
            label="below PI",
        )
        ax.scatter(
            counts_list[out_of_upper],
            mean_list[out_of_upper],
            color=colors["above"],
            marker="^",
            s=40,
            label="above PI",
        )
        min_v = scatter_min if xlim is None else min(xlim[0], scatter_min)
        max_v = scatter_max if xlim is None else max(xlim[1], scatter_max)
        ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="gray", linewidth=1.0, label="ideal y=x")
        style_axes(ax, xlabel="Observed count", ylabel="Forecast mean", integer_y=True)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    fig, ax = plt.subplots(figsize=figsizes["count_scatter"])
    if scatter_mode == "zoomed":
        zoom_low = scatter_min - max(0.03 * scatter_zoom_max, 0.5)
        zoom_high = scatter_zoom_max
        draw_scatter(
            ax,
            f"Observed vs Forecast (zoomed) | Coverage: {coverage:.2%}",
            xlim=(zoom_low, zoom_high),
            ylim=(zoom_low, zoom_high),
        )
    else:
        draw_scatter(ax, f"Observed vs Forecast Counts (full range) | Coverage: {coverage:.2%}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_pub_figure(fig, plot_output_dir / "obs_vs_forecast_scatter.png")
    plt.show()

    # 5) Max magnitude: true vs forecast (if available)
    if mag_max_mean is not None and mag_max is not None:
        mag_max_arr = np.asarray(mag_max, dtype=np.float64)
        mag_max_mean_arr = np.asarray(mag_max_mean, dtype=np.float64)
        mag_max_quantiles = getattr(sliding_result, "mag_max_quantiles", None)
        paired_finite_mask = np.isfinite(mag_max_arr) & np.isfinite(mag_max_mean_arr)
        obs_finite_mask = np.isfinite(mag_max_arr)
        pred_finite_mask = np.isfinite(mag_max_mean_arr)
        if np.any(obs_finite_mask) or np.any(pred_finite_mask):
            # Time series of max magnitude
            fig, ax = plt.subplots(figsize=figsizes["mag_max"])
            if np.any(obs_finite_mask):
                ax.plot(
                    t_forecast_ts[obs_finite_mask],
                    mag_max_arr[obs_finite_mask],
                    label="true max mag",
                    color=colors["true"],
                    linewidth=1.1,
                    marker="o",
                    markersize=2.8,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                )
            if np.any(pred_finite_mask):
                ax.plot(
                    t_forecast_ts[pred_finite_mask],
                    mag_max_mean_arr[pred_finite_mask],
                    label="forecast max mag mean",
                    color=colors["mean"],
                    linewidth=1.1,
                    marker="o",
                    markersize=2.8,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                )
            if mag_max_quantiles is not None and mag_max_quantiles.shape[0] == len(t_forecast_list):
                q_low_m = mag_max_quantiles[:, 0]
                q_high_m = mag_max_quantiles[:, 1]
                pi_finite_mask = np.isfinite(q_low_m) & np.isfinite(q_high_m)
                if np.any(pi_finite_mask):
                    ax.fill_between(
                        t_forecast_ts[pi_finite_mask],
                        q_low_m[pi_finite_mask],
                        q_high_m[pi_finite_mask],
                        alpha=0.14,
                        color=colors["pi"],
                        label="forecast PI (2.5-97.5)",
                    )
            style_axes(ax, xlabel=time_xlabel, ylabel="max magnitude")
            format_days_since_axis(ax, reference_ts=base_start_ts)
            ax.set_xlim(plot_x_left, plot_x_right)
            ax.set_title("Max magnitude: true vs forecast")
            if np.any(paired_finite_mask):
                try:
                    mag_max_mae_display = float(np.mean(np.abs(mag_max_arr[paired_finite_mask] - mag_max_mean_arr[paired_finite_mask])))
                except Exception:
                    mag_max_mae_display = None
            else:
                mag_max_mae_display = None
            mae_text = f"MAE: {mag_max_mae_display:.2f}" if mag_max_mae_display is not None and np.isfinite(mag_max_mae_display) else "MAE: N/A"
            ax.text(
                0.01,
                0.02,
                mae_text,
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=False, loc="upper right")
            fig.tight_layout()
            save_pub_figure(fig, plot_output_dir / "forecast_mag_max_over_time.png")
            plt.show()

            # Scatter plot observed vs forecast max magnitude
            if np.any(paired_finite_mask):
                fig, ax = plt.subplots(figsize=figsizes["mag_max_scatter"])
                ax.scatter(
                    mag_max_arr[paired_finite_mask],
                    mag_max_mean_arr[paired_finite_mask],
                    color=colors["true"],
                    alpha=0.7,
                    s=28,
                    label="true vs mean",
                )
                combo = np.concatenate([mag_max_arr[paired_finite_mask], mag_max_mean_arr[paired_finite_mask]])
                scatter_min = float(np.nanmin(combo))
                scatter_max = float(np.nanmax(combo))
                ax.plot([scatter_min, scatter_max], [scatter_min, scatter_max], linestyle="--", color="gray", linewidth=1.0, label="ideal y=x")
                style_axes(ax, xlabel="Observed max mag", ylabel="Forecast max mag mean")
                ax.set_title(f"Observed vs Forecast Max Magnitude | MAE: {mag_max_mae_display:.2f}" if mag_max_mae_display is not None and np.isfinite(mag_max_mae_display) else "Observed vs Forecast Max Magnitude")
                ax.set_aspect("equal", adjustable="box")
                ax.legend(frameon=False, loc="upper right")
                fig.tight_layout()
                save_pub_figure(fig, plot_output_dir / "obs_vs_forecast_mag_max_scatter.png")
                plt.show()

    return result_payload
