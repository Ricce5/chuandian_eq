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
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
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
SeedAggregationMethod = Literal["mean", "median"]
PointEstimateMode = Literal["ensemble", "mean"]


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
    "aggregate_seed_predictions",
    "bootstrap_curve",
    "bootstrap_metric",
    "resolve_checkpoint_path",
    "resolve_seed_checkpoint_paths",
    "bootstrap_multi_model_curve_ci",
    "bootstrap_multi_model_metric_ci",
    "bootstrap_multi_model_metric_ci_hierarchical",
    "PointEstimateMode",
    "bootstrap_curve_ci",
    "bootstrap_metric_ci",
]

_SEED_DIR_PATTERN = re.compile(r"^(?P<prefix>.+)_seed_(?P<seed>\d+)$")


def _normalize_root(path_like: str | Path | None) -> Path | None:
    """Convert an optional root path input to :class:`Path`."""

    if path_like is None:
        return None
    return Path(path_like).expanduser()


def _path_dedupe_key(path: Path) -> str:
    """Build a stable de-duplication key for path candidates."""

    return str(path.expanduser().resolve())


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    """Keep first-seen unique paths while preserving order."""

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = _path_dedupe_key(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def _candidate_base_paths(
    path_like: str | Path,
    *,
    project_root: Path | None,
    checkpoints_root: Path | None,
) -> list[Path]:
    """Enumerate candidate base paths for checkpoint resolution."""

    raw = Path(path_like).expanduser()
    if raw.is_absolute():
        return [raw]

    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(project_root / raw)
    else:
        candidates.append(raw)
    if checkpoints_root is not None:
        candidates.append(checkpoints_root / raw)
    return _dedupe_paths(candidates)


def _seed_key_for_sort(path: Path) -> tuple[int, str]:
    """Sort checkpoint paths by ``seed`` suffix when available."""

    run_name = path.parent.name if path.suffix.lower() == ".pth" else path.name
    match = _SEED_DIR_PATTERN.fullmatch(run_name)
    if match is None:
        return math.inf, str(path)
    return int(match.group("seed")), str(path)


def resolve_checkpoint_path(
    path_like: str | Path,
    *,
    project_root: str | Path | None = None,
    checkpoints_root: str | Path | None = None,
    checkpoint_filename: str = "best_model_1.pth",
) -> Path:
    """Resolve a single checkpoint path from flexible input.

    The input can be:
    - a checkpoint file path (absolute or relative);
    - a run directory containing ``checkpoint_filename``;
    - a path relative to ``project_root`` or ``checkpoints_root``.
    """

    if not checkpoint_filename:
        raise ValueError("checkpoint_filename must be non-empty.")

    project_root_path = _normalize_root(project_root)
    checkpoints_root_path = _normalize_root(checkpoints_root)
    if checkpoints_root_path is None and project_root_path is not None:
        checkpoints_root_path = project_root_path / "checkpoints"

    base_candidates = _candidate_base_paths(
        path_like,
        project_root=project_root_path,
        checkpoints_root=checkpoints_root_path,
    )
    file_candidates: list[Path] = []
    for base in base_candidates:
        if base.suffix.lower() == ".pth":
            file_candidates.append(base)
        else:
            file_candidates.append(base / checkpoint_filename)

    file_candidates = _dedupe_paths(file_candidates)
    for candidate in file_candidates:
        if candidate.is_file():
            return candidate.expanduser().resolve()

    tried = ", ".join(str(path) for path in file_candidates)
    raise FileNotFoundError(
        f"Checkpoint not found for path_like={path_like!r}. Tried: {tried}"
    )


def resolve_seed_checkpoint_paths(
    path_like: str | Path | Sequence[str | Path],
    *,
    project_root: str | Path | None = None,
    checkpoints_root: str | Path | None = None,
    checkpoint_filename: str = "best_model_1.pth",
    sort_by_seed: bool = True,
) -> list[Path]:
    """Resolve one or more checkpoints for seed-based runs.

    Accepted forms:
    - single run directory/file path;
    - seed run directory like ``.../tf_90_mf_5p5_seed_1`` (collect siblings);
    - non-seed base directory like ``.../tf_90_mf_5p5`` (collect ``*_seed_*``);
    - sequence of any of the above.
    """

    if isinstance(path_like, (str, Path)):
        raw_entries: list[str | Path] = [path_like]
    else:
        raw_entries = list(path_like)
        if not raw_entries:
            raise ValueError("path_like sequence must be non-empty.")

    project_root_path = _normalize_root(project_root)
    checkpoints_root_path = _normalize_root(checkpoints_root)
    if checkpoints_root_path is None and project_root_path is not None:
        checkpoints_root_path = project_root_path / "checkpoints"

    collected: list[Path] = []
    for entry in raw_entries:
        resolved_from_entry: list[Path] = []
        base_candidates = _candidate_base_paths(
            entry,
            project_root=project_root_path,
            checkpoints_root=checkpoints_root_path,
        )

        # 1) Direct file input.
        direct_file_candidates = [
            base for base in base_candidates if base.suffix.lower() == ".pth" and base.is_file()
        ]
        if direct_file_candidates:
            resolved_from_entry.extend(direct_file_candidates)
        else:
            # 2) Collect seed siblings when directory name follows *_seed_<int>.
            for base in base_candidates:
                if base.suffix:
                    continue
                match = _SEED_DIR_PATTERN.fullmatch(base.name)
                if match is None:
                    continue
                prefix = match.group("prefix")
                if not base.parent.exists():
                    continue
                for sibling in base.parent.iterdir():
                    if not sibling.is_dir():
                        continue
                    sibling_match = _SEED_DIR_PATTERN.fullmatch(sibling.name)
                    if sibling_match is None:
                        continue
                    if sibling_match.group("prefix") != prefix:
                        continue
                    candidate = sibling / checkpoint_filename
                    if candidate.is_file():
                        resolved_from_entry.append(candidate)

            # 3) Collect seed children by non-seed prefix or fallback to single run.
            if not resolved_from_entry:
                for base in base_candidates:
                    if base.suffix:
                        continue
                    if base.parent.exists():
                        glob_pat = f"{base.name}_seed_*"
                        for sibling in base.parent.glob(glob_pat):
                            if not sibling.is_dir():
                                continue
                            if _SEED_DIR_PATTERN.fullmatch(sibling.name) is None:
                                continue
                            candidate = sibling / checkpoint_filename
                            if candidate.is_file():
                                resolved_from_entry.append(candidate)
                    candidate_single = base / checkpoint_filename
                    if candidate_single.is_file():
                        resolved_from_entry.append(candidate_single)

        if not resolved_from_entry:
            raise FileNotFoundError(
                f"No checkpoints found for entry={entry!r} with checkpoint_filename={checkpoint_filename!r}."
            )

        collected.extend(resolved_from_entry)

    deduped = _dedupe_paths([path.expanduser().resolve() for path in collected])
    if sort_by_seed:
        deduped = sorted(deduped, key=_seed_key_for_sort)
    return deduped


def aggregate_seed_predictions(
    seed_predictions: Sequence[Sequence[float] | np.ndarray],
    *,
    method: SeedAggregationMethod = "mean",
) -> np.ndarray:
    """Aggregate per-seed prediction vectors into one probability vector."""

    if not seed_predictions:
        raise ValueError("seed_predictions must be non-empty.")

    vectors: list[np.ndarray] = []
    for index, pred in enumerate(seed_predictions):
        arr = np.asarray(pred, dtype=np.float64).reshape(-1)
        if arr.shape[0] == 0:
            raise ValueError(f"seed_predictions[{index}] must be non-empty.")
        vectors.append(arr)

    n_obs = vectors[0].shape[0]
    for index, arr in enumerate(vectors[1:], start=1):
        if arr.shape[0] != n_obs:
            raise ValueError(
                "All seed prediction vectors must share the same length. "
                f"Got {arr.shape[0]} at index={index}, expected {n_obs}."
            )

    stacked = np.stack(vectors, axis=0)
    if method == "mean":
        return np.nanmean(stacked, axis=0)
    if method == "median":
        return np.nanmedian(stacked, axis=0)
    raise ValueError(f"Unsupported aggregation method: {method!r}.")


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


def _prepare_multi_model_seed_predictions(
    y_true: Sequence[float] | np.ndarray,
    model_seed_predictions: Mapping[
        str,
        Sequence[Sequence[float] | np.ndarray] | np.ndarray,
    ],
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Validate aligned per-seed predictions for multi-model bootstrap.

    Parameters
    ----------
    y_true:
        Aligned ground-truth labels.
    model_seed_predictions:
        Mapping from model name to per-seed prediction vectors. Each value can
        be a list-like ``[n_seeds, n_obs]`` or an ``ndarray`` with equivalent
        shape.
    """

    if not model_seed_predictions:
        raise ValueError("model_seed_predictions must be non-empty.")

    y_true_arr = np.asarray(y_true).reshape(-1)
    n_obs = y_true_arr.shape[0]
    if n_obs == 0:
        raise ValueError("y_true must be non-empty.")

    seed_pred_map: dict[str, np.ndarray] = {}
    n_seeds_ref: int | None = None
    for model_name, seed_pred in model_seed_predictions.items():
        arr = np.asarray(seed_pred, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(
                f"Model '{model_name}' seed predictions must be 1D/2D. "
                f"Got ndim={arr.ndim}."
            )

        if arr.shape[1] != n_obs:
            if arr.shape[0] == n_obs and arr.shape[1] != n_obs:
                arr = arr.T
            else:
                raise ValueError(
                    f"Model '{model_name}' seed prediction shape mismatch: "
                    f"{arr.shape} vs expected (*, {n_obs})."
                )

        if arr.shape[1] != n_obs:
            raise ValueError(
                f"Model '{model_name}' seed prediction shape mismatch after transpose attempt: "
                f"{arr.shape} vs expected (*, {n_obs})."
            )

        n_seeds = int(arr.shape[0])
        if n_seeds <= 0:
            raise ValueError(f"Model '{model_name}' must provide at least one seed.")

        if n_seeds_ref is None:
            n_seeds_ref = n_seeds
        elif n_seeds != n_seeds_ref:
            raise ValueError(
                "All models must provide the same number of seed predictions. "
                f"Model '{model_name}' has {n_seeds}, expected {n_seeds_ref}."
            )

        seed_pred_map[str(model_name)] = arr

    if n_seeds_ref is None:
        raise ValueError("No valid model seed predictions provided.")

    return y_true_arr, seed_pred_map, int(n_seeds_ref)


def _aggregate_seed_prediction_matrix(
    seed_pred_matrix: np.ndarray,
    *,
    method: SeedAggregationMethod,
) -> np.ndarray:
    """Aggregate one ``[n_seeds, n_obs]`` matrix to ``[n_obs]``."""

    if seed_pred_matrix.ndim != 2:
        raise ValueError(
            f"seed_pred_matrix must be 2D. Got shape={seed_pred_matrix.shape}."
        )
    if method == "mean":
        return np.nanmean(seed_pred_matrix, axis=0)
    if method == "median":
        return np.nanmedian(seed_pred_matrix, axis=0)
    raise ValueError(f"Unsupported aggregation method: {method!r}.")


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
    baseline_model: str | None = None,
) -> MultiModelBootstrapResult:
    """Paired bootstrap for multiple models on aligned samples.

    All models share the same bootstrap indices per resample, enabling direct
    pairwise delta CI estimation.

    Delta behavior:
    - If ``baseline_model`` is provided: compute only ``(model - baseline)``
      for all non-baseline models.
    - If ``baseline_model`` is not provided and number of models is 2:
      compute the single pairwise delta between the two models.
    - If ``baseline_model`` is not provided and number of models is >= 3:
      raise ``ValueError`` and require explicit baseline selection.
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
    if baseline_model is not None and baseline_model not in pred_map:
        raise ValueError(
            f"baseline_model={baseline_model!r} not found in model_predictions. "
            f"Available={model_names}."
        )
    if baseline_model is None and len(model_names) >= 3:
        raise ValueError(
            "baseline_model must be provided when model_predictions has 3 or more models."
        )

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
    if baseline_model is None:
        delta_pairs = list(combinations(model_names, 2))
    else:
        delta_pairs = [
            (name, baseline_model)
            for name in model_names
            if name != baseline_model
        ]

    for left_name, right_name in delta_pairs:
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


def bootstrap_multi_model_metric_ci_hierarchical(
    y_true: Sequence[float] | np.ndarray,
    model_seed_predictions: Mapping[
        str,
        Sequence[Sequence[float] | np.ndarray] | np.ndarray,
    ],
    *,
    metric: MetricLike = "auc",
    task: TaskType | None = None,
    config: BootstrapConfig | None = None,
    baseline_model: str | None = None,
    seed_aggregation: SeedAggregationMethod = "mean",
    point_estimate_mode: PointEstimateMode = "ensemble",
) -> MultiModelBootstrapResult:
    """Hierarchical paired bootstrap over ``seed + sample`` for aligned models.

    Workflow per resample:
    1) Sample seed indices with replacement (same indices shared by all models).
    2) Sample observation indices according to ``config.sampling``.
    3) Compute per-model metric according to ``point_estimate_mode``:
       - ``"ensemble"``: aggregate per-seed predictions with ``seed_aggregation``
         then evaluate metric once;
       - ``"mean"``: evaluate each seed prediction separately and average metrics.
    4) Compute pairwise deltas on the paired resample.

    This captures both seed-level variability and sample-level variability.
    """

    if not model_seed_predictions:
        raise ValueError("model_seed_predictions must be non-empty.")

    cfg = config or BootstrapConfig()
    if cfg.n_resamples <= 0:
        raise ValueError("config.n_resamples must be positive.")
    if point_estimate_mode not in ("ensemble", "mean"):
        raise ValueError(
            f"Unsupported point_estimate_mode: {point_estimate_mode!r}. "
            "Expected 'ensemble' or 'mean'."
        )

    y_true_arr, seed_pred_map, n_seeds = _prepare_multi_model_seed_predictions(
        y_true,
        model_seed_predictions,
    )
    n_obs = y_true_arr.shape[0]

    metric_name, task_name, metric_fn = _resolve_metric_fn(metric, task)
    rng = np.random.default_rng(cfg.seed)
    obs_sampler = _make_index_sampler(n_obs, cfg)

    model_names = list(seed_pred_map.keys())
    if baseline_model is not None and baseline_model not in seed_pred_map:
        raise ValueError(
            f"baseline_model={baseline_model!r} not found in model_seed_predictions. "
            f"Available={model_names}."
        )
    if baseline_model is None and len(model_names) >= 3:
        raise ValueError(
            "baseline_model must be provided when model_seed_predictions has 3 or more models."
        )

    point_values: dict[str, float] = {}
    for name in model_names:
        if point_estimate_mode == "ensemble":
            agg_full = _aggregate_seed_prediction_matrix(
                seed_pred_map[name],
                method=seed_aggregation,
            )
            point_values[name] = float(metric_fn(y_true_arr, agg_full))
        else:
            seed_values = [
                float(metric_fn(y_true_arr, seed_pred_map[name][seed_idx]))
                for seed_idx in range(n_seeds)
            ]
            point_values[name] = float(np.mean(np.asarray(seed_values, dtype=np.float64)))

    samples_by_model: dict[str, list[float]] = {name: [] for name in model_names}
    for _ in range(cfg.n_resamples):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds, dtype=np.int64)
        obs_idx = obs_sampler(rng)
        yt = y_true_arr[obs_idx]
        if task_name == "classification" and not _classification_sample_valid(yt):
            continue

        batch_values: dict[str, float] = {}
        failed = False
        for name in model_names:
            seed_matrix = seed_pred_map[name][seed_idx][:, obs_idx]
            try:
                if point_estimate_mode == "ensemble":
                    yp = _aggregate_seed_prediction_matrix(seed_matrix, method=seed_aggregation)
                    batch_values[name] = float(metric_fn(yt, yp))
                else:
                    seed_metric_values = [
                        float(metric_fn(yt, seed_matrix[row_idx]))
                        for row_idx in range(seed_matrix.shape[0])
                    ]
                    batch_values[name] = float(
                        np.mean(np.asarray(seed_metric_values, dtype=np.float64))
                    )
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
    if baseline_model is None:
        delta_pairs = list(combinations(model_names, 2))
    else:
        delta_pairs = [
            (name, baseline_model)
            for name in model_names
            if name != baseline_model
        ]

    for left_name, right_name in delta_pairs:
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
