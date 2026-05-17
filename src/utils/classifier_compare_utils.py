"""Reusable helpers for classifier comparison notebooks.

This module keeps notebook code focused on experiment setup and plotting while
moving repeated data-processing logic into importable functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from src.utils.bootstrap_ci import (
    MultiModelBootstrapResult,
    SeedAggregationMethod,
    TaskType,
    bootstrap_multi_model_metric_ci,
    bootstrap_multi_model_metric_ci_hierarchical,
)

DEFAULT_CLASSIFICATION_METRIC_KEYS: tuple[str, ...] = (
    "threshold",
    "f1",
    "recall",
    "precision",
    "R",
    "auc",
    "pr_auc",
    "fpr",
    "tpr",
    "conf",
)


def round_optional(value: Any, decimals: int = 4) -> float | None:
    """Round numeric value, returning ``None`` for missing/non-numeric input."""

    if value is None:
        return None
    try:
        return round(float(value), int(decimals))
    except Exception:
        return None


def mf_token(mf: float) -> str:
    """Convert magnitude window float to token used in run/cache names."""

    return str(float(mf)).replace(".", "p")


def run_name(mf: float, tfore: int, seed: int) -> str:
    """Build standard run name used by classifier experiment checkpoints."""

    return f"tf_{int(tfore)}_mf_{mf_token(float(mf))}_seed_{int(seed)}"


def mean_seed_metrics(
    seed_metrics_rows: Sequence[Mapping[str, Any]],
    *,
    metric_keys: Sequence[str] = DEFAULT_CLASSIFICATION_METRIC_KEYS,
) -> dict[str, float | None]:
    """Compute per-key mean across seed-wise metric dicts."""

    if len(seed_metrics_rows) == 0:
        raise ValueError("seed_metrics_rows must be non-empty.")

    out: dict[str, float | None] = {}
    for key in metric_keys:
        values = [row.get(key) for row in seed_metrics_rows if row.get(key) is not None]
        if len(values) == 0:
            out[key] = None
        else:
            out[key] = float(np.mean(np.asarray(values, dtype=np.float64)))
    return out


def checkpoint_cache_path(
    checkpoint_path: Path,
    *,
    project_root: Path,
    disk_cache_dir: Path,
    cache_suffix: str = ".pred_cache.npz",
) -> Path:
    """Map checkpoint path to a deterministic disk-cache file path."""

    rel = checkpoint_path.resolve().relative_to(project_root.resolve())
    return disk_cache_dir / rel.with_suffix(cache_suffix)


def load_npz_prediction_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(y_true, y_prob)`` vectors from a compressed npz cache file."""

    payload = np.load(cache_path)
    y_true = np.asarray(payload["y_true"]).astype(int).reshape(-1)
    y_prob = np.asarray(payload["y_prob"], dtype=np.float64).reshape(-1)
    return y_true, y_prob


def save_npz_prediction_cache(
    cache_path: Path,
    *,
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> None:
    """Persist ``(y_true, y_prob)`` vectors to a compressed npz cache file."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        y_true=np.asarray(y_true).astype(int).reshape(-1),
        y_prob=np.asarray(y_prob, dtype=np.float64).reshape(-1),
    )


def ensure_identical_labels(
    y_true_left: Sequence[int] | np.ndarray,
    y_true_right: Sequence[int] | np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Validate paired labels and return normalized integer vector."""

    left = np.asarray(y_true_left).astype(int).reshape(-1)
    right = np.asarray(y_true_right).astype(int).reshape(-1)

    if left.shape[0] != right.shape[0] or not np.array_equal(left, right):
        raise ValueError(f"Label mismatch for paired comparison: {context}")
    return left


def build_seed_prediction_map_from_models(
    model_rows: Mapping[str, Mapping[str, Any]],
    *,
    seed_probs_key: str = "seed_probs",
) -> dict[str, np.ndarray]:
    """Extract ``{model: [n_seeds, n_obs]}`` map from notebook model rows."""

    seed_pred_map: dict[str, np.ndarray] = {}
    for name, row in model_rows.items():
        if seed_probs_key not in row:
            raise KeyError(f"Missing key '{seed_probs_key}' in model row: {name}")
        arr = np.asarray(row[seed_probs_key], dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(
                f"Seed predictions for model '{name}' must be 1D/2D. Got ndim={arr.ndim}."
            )
        seed_pred_map[str(name)] = arr
    return seed_pred_map


def paired_metric_bootstrap(
    *,
    y_true: Sequence[int] | np.ndarray,
    metric_name: str,
    metric_config,
    use_hierarchical_seed_sample: bool,
    seed_aggregation: SeedAggregationMethod,
    baseline_model: str | None = None,
    model_predictions: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    model_seed_predictions: Mapping[str, Sequence[Sequence[float] | np.ndarray] | np.ndarray]
    | None = None,
    task: TaskType = "classification",
) -> MultiModelBootstrapResult:
    """Run paired metric bootstrap in either ensemble or hierarchical mode."""

    y_true_arr = np.asarray(y_true).astype(int).reshape(-1)

    if use_hierarchical_seed_sample:
        if not model_seed_predictions:
            raise ValueError("model_seed_predictions is required in hierarchical mode.")
        return bootstrap_multi_model_metric_ci_hierarchical(
            y_true=y_true_arr,
            model_seed_predictions=model_seed_predictions,
            metric=metric_name,
            task=task,
            config=metric_config,
            baseline_model=baseline_model,
            seed_aggregation=seed_aggregation,
        )

    if not model_predictions:
        raise ValueError("model_predictions is required in ensemble mode.")

    return bootstrap_multi_model_metric_ci(
        y_true=y_true_arr,
        model_predictions=model_predictions,
        metric=metric_name,
        task=task,
        config=metric_config,
        baseline_model=baseline_model,
    )


def metric_ylim(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    low_key: str,
    high_key: str,
    pad_ratio: float = 0.12,
    min_span: float = 0.08,
) -> tuple[float, float]:
    """Compute stable y-limits for metric-with-CI scatter plots."""

    lows = [float(row[low_key]) for row in rows]
    highs = [float(row[high_key]) for row in rows]
    values = [float(row[value_key]) for row in rows]

    lower = min(lows + values)
    upper = max(highs + values)
    span = upper - lower

    if span < min_span:
        center = 0.5 * (lower + upper)
        lower = center - 0.5 * min_span
        upper = center + 0.5 * min_span
        span = upper - lower

    pad = max(0.01, span * pad_ratio)
    lower = max(0.0, lower - pad)
    upper = min(1.0, upper + pad)

    if (upper - lower) < min_span:
        center = 0.5 * (lower + upper)
        lower = max(0.0, center - 0.5 * min_span)
        upper = min(1.0, center + 0.5 * min_span)

    lower = max(0.0, np.floor(lower * 20.0) / 20.0)
    upper = min(1.0, np.ceil(upper * 20.0) / 20.0)

    if (upper - lower) < 0.05:
        upper = min(1.0, lower + 0.05)

    return float(lower), float(upper)


def style_metric_axis(ax, *, ylabel: str, ylim: tuple[float, float], style_axes_fn, xlabel: str | None = None) -> None:
    """Apply consistent axis style for notebook metric panels."""

    style_axes_fn(ax, xlabel=xlabel if xlabel is not None else "", ylabel=ylabel)
    ax.set_ylim(*ylim)

    span = ylim[1] - ylim[0]
    if span <= 0.18:
        major_step = 0.02
    elif span <= 0.40:
        major_step = 0.05
    else:
        major_step = 0.10

    ax.yaxis.set_major_locator(MultipleLocator(major_step))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(direction="in", length=3.2, width=0.8, top=False, right=False)

