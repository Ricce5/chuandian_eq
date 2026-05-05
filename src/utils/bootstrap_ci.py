"""Structured bootstrap utilities for classification/regression evaluation.

This module provides:

- A configuration object (:class:`BootstrapConfig`) for reusable bootstrap setup.
- Metric-level bootstrap CI for both classification and regression tasks.
- Curve-level bootstrap CI for ROC/PR curves.
- Paired multi-model bootstrap (same resample indices across models) for
  fair delta estimation at both metric and curve level.
- Backward-compatible wrappers used by existing notebooks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_curve,
)

TaskType = Literal["classification", "regression"]
SamplingMethod = Literal["iid", "block"]
CurveType = Literal["roc", "pr"]
MetricName = Literal["auc", "ap", "mse", "rmse", "mae", "r2"]
MetricLike = MetricName | str | Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class BootstrapConfig:
    """Configuration for bootstrap resampling.

    Attributes
    ----------
    n_resamples:
        Number of bootstrap resamples.
    ci:
        Percentile interval, e.g. ``(2.5, 97.5)`` for a 95% interval.
    seed:
        Random seed for reproducibility. ``None`` uses non-deterministic RNG.
    sampling:
        Sampling strategy: ``"iid"`` or ``"block"``.
    block_size:
        Block length for block bootstrap. If ``None``, a default based on
        sample size is used.
    circular_block:
        Whether to use circular/wrap-around block bootstrap.
    """

    n_resamples: int = 800
    ci: tuple[float, float] = (2.5, 97.5)
    seed: int | None = 42
    sampling: SamplingMethod = "iid"
    block_size: int | None = None
    circular_block: bool = True


@dataclass
class BootstrappedMetric:
    """Result container for scalar metric bootstrap."""

    metric_name: str
    point_estimate: float
    ci_low: float
    ci_high: float
    samples: np.ndarray
    valid_resamples: int
    total_resamples: int


@dataclass
class BootstrappedCurve:
    """Result container for curve bootstrap envelopes on a common grid."""

    curve: CurveType
    grid: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    valid_resamples: int
    total_resamples: int


@dataclass
class MultiModelBootstrapResult:
    """Paired bootstrap results for multiple aligned models."""

    model_results: dict[str, BootstrappedMetric]
    pairwise_deltas: dict[tuple[str, str], BootstrappedMetric]


@dataclass
class MultiModelCurveBootstrapResult:
    """Paired bootstrap curve-envelope results for aligned models."""

    curve: CurveType
    grid: np.ndarray
    model_results: dict[str, BootstrappedCurve]
    valid_resamples: int
    total_resamples: int


MetricFn = Callable[[np.ndarray, np.ndarray], float]

__all__ = [
    "BootstrapConfig",
    "BootstrappedMetric",
    "BootstrappedCurve",
    "MultiModelBootstrapResult",
    "MultiModelCurveBootstrapResult",
    "bootstrap_curve",
    "bootstrap_metric",
    "bootstrap_multi_model_curve_ci",
    "bootstrap_multi_model_metric_ci",
    "bootstrap_curve_ci",
    "bootstrap_metric_ci",
]


def _prepare_vectors(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and reshape target/prediction vectors."""

    y_true_arr = np.asarray(y_true).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"y_true and y_pred length mismatch: {y_true_arr.shape[0]} vs {y_pred_arr.shape[0]}"
        )
    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true and y_pred must be non-empty.")
    return y_true_arr, y_pred_arr


def _prepare_multi_model_predictions(
    y_true: Sequence[float] | np.ndarray,
    model_predictions: Mapping[str, Sequence[float] | np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Validate aligned labels/predictions for multi-model bootstrap."""

    if not model_predictions:
        raise ValueError("model_predictions must be non-empty.")

    y_true_arr = np.asarray(y_true).reshape(-1)
    n_obs = y_true_arr.shape[0]
    if n_obs == 0:
        raise ValueError("y_true must be non-empty.")

    pred_map: dict[str, np.ndarray] = {}
    for model_name, pred in model_predictions.items():
        arr = np.asarray(pred, dtype=np.float64).reshape(-1)
        if arr.shape[0] != n_obs:
            raise ValueError(
                f"Model '{model_name}' prediction length mismatch: {arr.shape[0]} vs y_true {n_obs}"
            )
        pred_map[str(model_name)] = arr

    return y_true_arr, pred_map


def _classification_sample_valid(y_true: np.ndarray) -> bool:
    """Check whether a bootstrap sample contains both classes."""

    return np.unique(y_true.astype(int)).size >= 2


def _metric_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC metric implementation used by the metric registry."""

    fpr, tpr, _ = roc_curve(y_true.astype(int), y_score)
    return float(auc(fpr, tpr))


def _metric_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision metric implementation."""

    return float(average_precision_score(y_true.astype(int), y_score))


def _metric_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE metric implementation."""

    return float(mean_squared_error(y_true, y_pred))


def _metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE metric implementation."""

    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE metric implementation."""

    return float(mean_absolute_error(y_true, y_pred))


def _metric_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R2 metric implementation."""

    return float(r2_score(y_true, y_pred))


CLASSIFICATION_METRICS: dict[str, MetricFn] = {
    "auc": _metric_auc,
    "ap": _metric_ap,
}
REGRESSION_METRICS: dict[str, MetricFn] = {
    "mse": _metric_mse,
    "rmse": _metric_rmse,
    "mae": _metric_mae,
    "r2": _metric_r2,
}


def _resolve_metric_fn(
    metric: MetricLike,
    task: TaskType | None,
) -> tuple[str, TaskType, MetricFn]:
    """Resolve metric name/task/function from flexible metric input."""

    if callable(metric):
        metric_name = getattr(metric, "__name__", "custom_metric")
        if task is None:
            raise ValueError("task must be provided when metric is a callable.")
        return metric_name, task, metric

    metric_name = str(metric).lower()
    if task is None:
        if metric_name in CLASSIFICATION_METRICS:
            task = "classification"
        elif metric_name in REGRESSION_METRICS:
            task = "regression"
        else:
            raise ValueError(
                f"Cannot infer task for metric={metric_name!r}. "
                f"Known classification metrics={sorted(CLASSIFICATION_METRICS)}, "
                f"regression metrics={sorted(REGRESSION_METRICS)}."
            )

    if task == "classification":
        if metric_name not in CLASSIFICATION_METRICS:
            raise ValueError(
                f"Unsupported classification metric {metric_name!r}. "
                f"Supported={sorted(CLASSIFICATION_METRICS)}."
            )
        return metric_name, task, CLASSIFICATION_METRICS[metric_name]

    if task == "regression":
        if metric_name not in REGRESSION_METRICS:
            raise ValueError(
                f"Unsupported regression metric {metric_name!r}. "
                f"Supported={sorted(REGRESSION_METRICS)}."
            )
        return metric_name, task, REGRESSION_METRICS[metric_name]

    raise ValueError(f"Unsupported task {task!r}.")


def _make_iid_sampler(n_obs: int) -> Callable[[np.random.Generator], np.ndarray]:
    """Create IID bootstrap sampler that returns resampled indices."""

    def _sample(rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, n_obs, size=n_obs, dtype=np.int64)

    return _sample


def _make_block_sampler(
    n_obs: int,
    *,
    block_size: int,
    circular_block: bool,
) -> Callable[[np.random.Generator], np.ndarray]:
    """Create block-bootstrap sampler that returns resampled indices."""

    if block_size <= 0:
        raise ValueError("block_size must be positive for block bootstrap.")
    if not circular_block and block_size > n_obs:
        raise ValueError(
            f"block_size={block_size} cannot exceed n_obs={n_obs} when circular_block=False."
        )

    n_blocks = int(math.ceil(n_obs / block_size))
    local_offsets = np.arange(block_size, dtype=np.int64)

    if circular_block:
        def _sample(rng: np.random.Generator) -> np.ndarray:
            starts = rng.integers(0, n_obs, size=n_blocks, dtype=np.int64)
            stacked = ((starts[:, None] + local_offsets[None, :]) % n_obs).reshape(-1)
            return stacked[:n_obs]
    else:
        max_start = n_obs - block_size + 1

        def _sample(rng: np.random.Generator) -> np.ndarray:
            starts = rng.integers(0, max_start, size=n_blocks, dtype=np.int64)
            stacked = (starts[:, None] + local_offsets[None, :]).reshape(-1)
            return stacked[:n_obs]

    return _sample


def _make_index_sampler(
    n_obs: int,
    config: BootstrapConfig,
) -> Callable[[np.random.Generator], np.ndarray]:
    """Dispatch to the proper index sampler according to config."""

    if config.sampling == "iid":
        return _make_iid_sampler(n_obs)
    if config.sampling == "block":
        block_size = config.block_size or max(1, int(round(math.sqrt(n_obs))))
        return _make_block_sampler(
            n_obs,
            block_size=int(block_size),
            circular_block=bool(config.circular_block),
        )
    raise ValueError(f"Unsupported sampling method {config.sampling!r}.")


def _percentile_ci(samples: np.ndarray, ci: tuple[float, float]) -> tuple[float, float]:
    """Compute percentile CI from bootstrap samples."""

    if samples.size == 0:
        return float("nan"), float("nan")
    low, high = ci
    lo, hi = np.nanpercentile(samples, [low, high])
    return float(lo), float(hi)


def _curve_to_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    curve: CurveType,
    grid: np.ndarray,
) -> np.ndarray:
    """Project one ROC/PR curve to a fixed grid for envelope aggregation."""

    if curve == "roc":
        fpr, tpr, _ = roc_curve(y_true.astype(int), y_prob)
        y_interp = np.interp(grid, fpr, tpr)
        y_interp[0] = 0.0
        y_interp[-1] = 1.0
        return y_interp

    if curve == "pr":
        precision, recall, _ = precision_recall_curve(y_true.astype(int), y_prob)
        if recall[0] > recall[-1]:
            recall = recall[::-1]
            precision = precision[::-1]
        y_interp = np.interp(grid, recall, precision)
        return np.clip(y_interp, 0.0, 1.0)

    raise ValueError(f"Unknown curve type: {curve!r}")


def bootstrap_metric(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    metric: MetricLike = "auc",
    task: TaskType | None = None,
    config: BootstrapConfig | None = None,
) -> BootstrappedMetric:
    """Compute bootstrap CI for a scalar metric.

    Supports classification and regression metrics using a common interface.
    For classification, degenerate resamples (single class only) are skipped.
    """
    cfg = config or BootstrapConfig()
    if cfg.n_resamples <= 0:
        raise ValueError("config.n_resamples must be positive.")

    y_true_arr, y_pred_arr = _prepare_vectors(y_true, y_pred)
    metric_name, task_name, metric_fn = _resolve_metric_fn(metric, task)
    rng = np.random.default_rng(cfg.seed)
    sampler = _make_index_sampler(y_true_arr.shape[0], cfg)
    point_estimate = float(metric_fn(y_true_arr, y_pred_arr))

    sample_values: list[float] = []
    for _ in range(cfg.n_resamples):
        idx = sampler(rng)
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]
        if task_name == "classification" and not _classification_sample_valid(yt):
            continue
        try:
            sample_values.append(float(metric_fn(yt, yp)))
        except Exception:
            continue

    sample_arr = np.asarray(sample_values, dtype=np.float64)
    ci_low, ci_high = _percentile_ci(sample_arr, cfg.ci)
    return BootstrappedMetric(
        metric_name=metric_name,
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        samples=sample_arr,
        valid_resamples=int(sample_arr.shape[0]),
        total_resamples=int(cfg.n_resamples),
    )


def bootstrap_curve(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    curve: CurveType = "roc",
    config: BootstrapConfig | None = None,
    grid_n: int = 201,
) -> BootstrappedCurve:
    """Compute pointwise bootstrap CI for ROC/PR curves."""
    cfg = config or BootstrapConfig(n_resamples=400)
    if cfg.n_resamples <= 0:
        raise ValueError("config.n_resamples must be positive.")
    if grid_n <= 1:
        raise ValueError("grid_n must be greater than 1.")

    y_true_arr, y_prob_arr = _prepare_vectors(y_true, y_prob)
    y_true_arr = y_true_arr.astype(int)
    rng = np.random.default_rng(cfg.seed)
    sampler = _make_index_sampler(y_true_arr.shape[0], cfg)
    grid = np.linspace(0.0, 1.0, grid_n)
    curves: list[np.ndarray] = []

    for _ in range(cfg.n_resamples):
        idx = sampler(rng)
        yt = y_true_arr[idx]
        yp = y_prob_arr[idx]
        if not _classification_sample_valid(yt):
            continue
        try:
            curves.append(_curve_to_grid(yt, yp, curve=curve, grid=grid))
        except Exception:
            continue

    if not curves:
        return BootstrappedCurve(
            curve=curve,
            grid=grid,
            ci_low=np.full_like(grid, np.nan),
            ci_high=np.full_like(grid, np.nan),
            valid_resamples=0,
            total_resamples=int(cfg.n_resamples),
        )

    curve_arr = np.asarray(curves, dtype=np.float64)
    q_low, q_high = cfg.ci
    ci_low = np.nanpercentile(curve_arr, q_low, axis=0)
    ci_high = np.nanpercentile(curve_arr, q_high, axis=0)
    return BootstrappedCurve(
        curve=curve,
        grid=grid,
        ci_low=ci_low,
        ci_high=ci_high,
        valid_resamples=int(curve_arr.shape[0]),
        total_resamples=int(cfg.n_resamples),
    )


def bootstrap_multi_model_curve_ci(
    y_true: Sequence[int] | np.ndarray,
    model_predictions: Mapping[str, Sequence[float] | np.ndarray],
    *,
    curve: CurveType = "roc",
    config: BootstrapConfig | None = None,
    grid_n: int = 201,
) -> MultiModelCurveBootstrapResult:
    """Paired bootstrap curve CI for multiple models on aligned samples.

    Each accepted resample uses the same sampled indices across all models,
    so ROC/PR curve envelopes are directly comparable model-to-model.
    """
    cfg = config or BootstrapConfig(n_resamples=400)
    if cfg.n_resamples <= 0:
        raise ValueError("config.n_resamples must be positive.")
    if grid_n <= 1:
        raise ValueError("grid_n must be greater than 1.")

    y_true_arr, pred_map = _prepare_multi_model_predictions(y_true, model_predictions)
    y_true_arr = y_true_arr.astype(int)
    rng = np.random.default_rng(cfg.seed)
    sampler = _make_index_sampler(y_true_arr.shape[0], cfg)
    grid = np.linspace(0.0, 1.0, grid_n)

    model_names = list(pred_map.keys())
    curves_by_model: dict[str, list[np.ndarray]] = {name: [] for name in model_names}
    valid_resamples = 0

    for _ in range(cfg.n_resamples):
        idx = sampler(rng)
        yt = y_true_arr[idx]
        if not _classification_sample_valid(yt):
            continue

        batch_curves: dict[str, np.ndarray] = {}
        failed = False
        for name in model_names:
            yp = pred_map[name][idx]
            try:
                batch_curves[name] = _curve_to_grid(yt, yp, curve=curve, grid=grid)
            except Exception:
                failed = True
                break
        if failed:
            continue

        for name, curve_values in batch_curves.items():
            curves_by_model[name].append(curve_values)
        valid_resamples += 1

    model_results: dict[str, BootstrappedCurve] = {}
    for name in model_names:
        curves = curves_by_model[name]
        if not curves:
            model_results[name] = BootstrappedCurve(
                curve=curve,
                grid=grid,
                ci_low=np.full_like(grid, np.nan),
                ci_high=np.full_like(grid, np.nan),
                valid_resamples=0,
                total_resamples=int(cfg.n_resamples),
            )
            continue

        curve_arr = np.asarray(curves, dtype=np.float64)
        q_low, q_high = cfg.ci
        ci_low = np.nanpercentile(curve_arr, q_low, axis=0)
        ci_high = np.nanpercentile(curve_arr, q_high, axis=0)
        model_results[name] = BootstrappedCurve(
            curve=curve,
            grid=grid,
            ci_low=ci_low,
            ci_high=ci_high,
            valid_resamples=int(curve_arr.shape[0]),
            total_resamples=int(cfg.n_resamples),
        )

    return MultiModelCurveBootstrapResult(
        curve=curve,
        grid=grid,
        model_results=model_results,
        valid_resamples=int(valid_resamples),
        total_resamples=int(cfg.n_resamples),
    )


def bootstrap_multi_model_metric_ci(
    y_true: Sequence[float] | np.ndarray,
    model_predictions: Mapping[str, Sequence[float] | np.ndarray],
    *,
    metric: MetricLike = "auc",
    task: TaskType | None = None,
    config: BootstrapConfig | None = None,
) -> MultiModelBootstrapResult:
    """Paired bootstrap for multiple models on aligned samples.

    All models share the same bootstrap indices per resample, enabling direct
    pairwise delta CI estimation.
    """
    if not model_predictions:
        raise ValueError("model_predictions must be non-empty.")

    cfg = config or BootstrapConfig()
    if cfg.n_resamples <= 0:
        raise ValueError("config.n_resamples must be positive.")

    y_true_arr, pred_map = _prepare_multi_model_predictions(y_true, model_predictions)
    n_obs = y_true_arr.shape[0]

    metric_name, task_name, metric_fn = _resolve_metric_fn(metric, task)
    rng = np.random.default_rng(cfg.seed)
    sampler = _make_index_sampler(n_obs, cfg)

    model_names = list(pred_map.keys())
    point_values = {
        name: float(metric_fn(y_true_arr, pred_map[name]))
        for name in model_names
    }
    samples_by_model: dict[str, list[float]] = {name: [] for name in model_names}

    for _ in range(cfg.n_resamples):
        idx = sampler(rng)
        yt = y_true_arr[idx]
        if task_name == "classification" and not _classification_sample_valid(yt):
            continue

        batch_values: dict[str, float] = {}
        failed = False
        for name in model_names:
            yp = pred_map[name][idx]
            try:
                batch_values[name] = float(metric_fn(yt, yp))
            except Exception:
                failed = True
                break
        if failed:
            continue

        for name, value in batch_values.items():
            samples_by_model[name].append(value)

    model_results: dict[str, BootstrappedMetric] = {}
    for name in model_names:
        sample_arr = np.asarray(samples_by_model[name], dtype=np.float64)
        ci_low, ci_high = _percentile_ci(sample_arr, cfg.ci)
        model_results[name] = BootstrappedMetric(
            metric_name=metric_name,
            point_estimate=point_values[name],
            ci_low=ci_low,
            ci_high=ci_high,
            samples=sample_arr,
            valid_resamples=int(sample_arr.shape[0]),
            total_resamples=int(cfg.n_resamples),
        )

    pairwise_deltas: dict[tuple[str, str], BootstrappedMetric] = {}
    for left_name, right_name in combinations(model_names, 2):
        left_samples = model_results[left_name].samples
        right_samples = model_results[right_name].samples
        min_len = min(left_samples.shape[0], right_samples.shape[0])
        if min_len == 0:
            delta_samples = np.asarray([], dtype=np.float64)
        else:
            delta_samples = left_samples[:min_len] - right_samples[:min_len]

        ci_low, ci_high = _percentile_ci(delta_samples, cfg.ci)
        pairwise_deltas[(left_name, right_name)] = BootstrappedMetric(
            metric_name=f"{metric_name}_delta",
            point_estimate=point_values[left_name] - point_values[right_name],
            ci_low=ci_low,
            ci_high=ci_high,
            samples=delta_samples,
            valid_resamples=int(delta_samples.shape[0]),
            total_resamples=int(cfg.n_resamples),
        )

    return MultiModelBootstrapResult(
        model_results=model_results,
        pairwise_deltas=pairwise_deltas,
    )


def bootstrap_curve_ci(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    curve: CurveType = "roc",
    n_boot: int = 400,
    seed: int = 42,
    grid_n: int = 201,
    ci: tuple[float, float] = (2.5, 97.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper returning ``(grid, ci_low, ci_high)``.

    This keeps legacy notebook code stable while internally using
    :func:`bootstrap_curve`.
    """
    result = bootstrap_curve(
        y_true,
        y_prob,
        curve=curve,
        config=BootstrapConfig(
            n_resamples=n_boot,
            seed=seed,
            ci=ci,
            sampling="iid",
        ),
        grid_n=grid_n,
    )
    return result.grid, result.ci_low, result.ci_high


def bootstrap_metric_ci(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    metric: MetricLike = "auc",
    n_boot: int = 800,
    seed: int = 42,
    ci: tuple[float, float] = (2.5, 97.5),
    task: TaskType | None = None,
) -> tuple[float, float]:
    """Backward-compatible wrapper returning ``(ci_low, ci_high)``.

    This keeps legacy notebook code stable while internally using
    :func:`bootstrap_metric`.
    """
    result = bootstrap_metric(
        y_true,
        y_pred,
        metric=metric,
        task=task,
        config=BootstrapConfig(
            n_resamples=n_boot,
            seed=seed,
            ci=ci,
            sampling="iid",
        ),
    )
    return result.ci_low, result.ci_high
