"""Reusable helpers for classifier comparison notebooks.

This module keeps notebook code focused on experiment setup and plotting while
moving repeated data-processing logic into importable functions.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from src.utils.bootstrap_ci import (
    MultiModelCurveBootstrapResult,
    MultiModelBootstrapResult,
    PointEstimateMode,
    SeedAggregationMethod,
    TaskType,
    bootstrap_multi_model_curve_ci,
    bootstrap_multi_model_curve_ci_hierarchical,
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


@dataclass(frozen=True)
class ClassificationBootstrapExportArtifacts:
    """Artifacts produced by classifier bootstrap export pipeline."""

    metrics_rows: list[dict[str, Any]]
    delta_rows: list[dict[str, Any]]
    summary_by_model: dict[str, dict[str, float | None]]
    metrics_csv_path: Path
    deltas_csv_path: Path | None
    metrics_json_path: Path
    json_payload: dict[str, Any]


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
    point_estimate_mode: PointEstimateMode = "ensemble",
    baseline_model: str | None = None,
    model_predictions: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    model_seed_predictions: Mapping[str, Sequence[Sequence[float] | np.ndarray] | np.ndarray]
    | None = None,
    task: TaskType = "classification",
) -> MultiModelBootstrapResult:
    """Run paired metric bootstrap in either ensemble or hierarchical mode."""

    if task == "classification":
        y_true_arr = np.asarray(y_true).astype(int).reshape(-1)
    else:
        y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)

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
            point_estimate_mode=point_estimate_mode,
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


def _default_window_model_rows(
    window_row: Mapping[str, Any],
    *,
    left_model_name: str = "RF",
    right_model_name: str = "EM-EQF",
) -> dict[str, dict[str, Any]]:
    """Normalize per-window notebook row to ``{model_name: model_payload}``.

    Supports both:
    1) modern shape with ``window_row["models"]``;
    2) legacy two-model shape used by baseline notebook.
    """

    if "models" in window_row:
        model_rows = window_row["models"]
        if not isinstance(model_rows, Mapping):
            raise TypeError("window_row['models'] must be a mapping.")
        out: dict[str, dict[str, Any]] = {}
        for name, payload in model_rows.items():
            if not isinstance(payload, Mapping):
                raise TypeError(f"window_row['models'][{name!r}] must be a mapping.")
            out[str(name)] = dict(payload)
        return out

    required = (
        "y_test_rf",
        "y_prob_rf",
        "rf_seed_probs",
        "rf_metrics_mean",
        "y_test_em",
        "y_prob_em",
        "em_seed_probs",
        "em_metrics_mean",
    )
    missing = [key for key in required if key not in window_row]
    if missing:
        raise KeyError(
            "window_row must contain either 'models' or legacy RF/EM keys. "
            f"Missing keys: {missing}"
        )

    return {
        str(left_model_name): {
            "metrics": dict(window_row["rf_metrics_mean"]),
            "y_true": window_row["y_test_rf"],
            "y_prob": window_row["y_prob_rf"],
            "seed_probs": window_row["rf_seed_probs"],
            "seed_metrics": window_row.get("rf_seed_metrics", []),
        },
        str(right_model_name): {
            "metrics": dict(window_row["em_metrics_mean"]),
            "y_true": window_row["y_test_em"],
            "y_prob": window_row["y_prob_em"],
            "seed_probs": window_row["em_seed_probs"],
            "seed_metrics": window_row.get("em_seed_metrics", []),
        },
    }


def _optional_delta(left: float | None, right: float | None) -> float | None:
    """Safely compute ``left-right`` for optional values."""

    if left is None or right is None:
        return None
    return float(left) - float(right)


def build_classification_bootstrap_export_rows(
    *,
    multi_results: Sequence[Mapping[str, Any]],
    bootstrap_preset,
    round_fn: Callable[[Any], float | None],
    threshold_optimize_metric: str,
    use_hierarchical_seed_sample: bool,
    seed_aggregation: SeedAggregationMethod = "mean",
    point_estimate_mode: PointEstimateMode = "ensemble",
    baseline_model: str | None = None,
    sort_export_by: Sequence[str] = ("Mf", "Tfore", "model"),
    sort_delta_by: Sequence[str] = ("Mf", "Tfore", "comparison"),
    auc_seed_base: int = 9100,
    ap_seed_base: int = 9300,
    window_model_rows_getter: Callable[[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build metrics/delta bootstrap export rows for classifier notebooks.

    Parameters are aligned with baseline/pretrain notebook usage and keep all
    exported fields stable while reducing repeated bootstrap loop code.
    """

    ci_low_pct, ci_high_pct = bootstrap_preset.ci
    metrics_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    for i, window_row in enumerate(multi_results):
        model_rows = (
            dict(window_model_rows_getter(window_row))
            if window_model_rows_getter is not None
            else _default_window_model_rows(window_row)
        )
        model_names = list(model_rows.keys())
        if len(model_names) == 0:
            continue

        effective_baseline = baseline_model or model_names[0]
        if effective_baseline not in model_rows:
            raise ValueError(
                f"baseline_model={effective_baseline!r} not found in model rows. "
                f"Available={model_names}"
            )

        y_true_ref = np.asarray(model_rows[effective_baseline]["y_true"]).astype(int).reshape(-1)
        for model_name, model_payload in model_rows.items():
            y_true_ref = ensure_identical_labels(
                model_payload["y_true"],
                y_true_ref,
                context=(
                    f"window (Mf={window_row.get('Mf')}, Tfore={window_row.get('Tfore')}), "
                    f"baseline={effective_baseline}, current={model_name}"
                ),
            )

        model_predictions = {
            name: np.asarray(payload["y_prob"], dtype=np.float64).reshape(-1)
            for name, payload in model_rows.items()
        }

        model_seed_predictions = build_seed_prediction_map_from_models(model_rows)

        auc_result = paired_metric_bootstrap(
            y_true=y_true_ref,
            metric_name="auc",
            metric_config=bootstrap_preset.metric_config(seed=int(auc_seed_base + i)),
            use_hierarchical_seed_sample=use_hierarchical_seed_sample,
            seed_aggregation=seed_aggregation,
            point_estimate_mode=point_estimate_mode,
            baseline_model=effective_baseline,
            model_predictions=model_predictions,
            model_seed_predictions=model_seed_predictions,
            task="classification",
        )
        ap_result = paired_metric_bootstrap(
            y_true=y_true_ref,
            metric_name="ap",
            metric_config=bootstrap_preset.metric_config(seed=int(ap_seed_base + i)),
            use_hierarchical_seed_sample=use_hierarchical_seed_sample,
            seed_aggregation=seed_aggregation,
            point_estimate_mode=point_estimate_mode,
            baseline_model=effective_baseline,
            model_predictions=model_predictions,
            model_seed_predictions=model_seed_predictions,
            task="classification",
        )

        seed_metric_std_by_model: dict[str, dict[str, float | None]] = {}
        for model_name in model_names:
            seed_metrics = model_rows[model_name].get("seed_metrics", [])
            if len(seed_metrics) > 0:
                seed_metric_std_by_model[model_name] = {
                    "auc_seed_std": round_fn(
                        np.std(
                            [float(m["auc"]) for m in seed_metrics],
                            ddof=1,
                        )
                        if len(seed_metrics) > 1
                        else 0.0
                    ),
                    "pr_auc_seed_std": round_fn(
                        np.std(
                            [float(m["pr_auc"]) for m in seed_metrics],
                            ddof=1,
                        )
                        if len(seed_metrics) > 1
                        else 0.0
                    ),
                }
            else:
                seed_metric_std_by_model[model_name] = {
                    "auc_seed_std": None,
                    "pr_auc_seed_std": None,
                }

        point_metrics_by_model: dict[str, dict[str, float | None]] = {}
        for model_name in model_names:
            metrics = model_rows[model_name].get("metrics", {})
            point_metrics_by_model[model_name] = {
                "threshold": round_fn(metrics.get("threshold")),
                "f1": round_fn(metrics.get("f1")),
                "recall": round_fn(metrics.get("recall")),
                "precision": round_fn(metrics.get("precision")),
                "R": round_fn(metrics.get("R")),
                "auc": round_fn(metrics.get("auc")),
                    "pr_auc": round_fn(metrics.get("pr_auc")),
                }

        window_id = int(window_row.get("window_id", i))
        mf_value = float(window_row["Mf"])
        tfore_value = int(window_row["Tfore"])

        for model_name in model_names:
            point_metrics = point_metrics_by_model[model_name]
            auc_ci = auc_result.model_results[model_name]
            ap_ci = ap_result.model_results[model_name]

            metrics_rows.append(
                {
                    "window_id": window_id,
                    "Mf": mf_value,
                    "Tfore": tfore_value,
                    "model": str(model_name),
                    "n_test": int(y_true_ref.shape[0]),
                    "positive_rate": round_fn(np.mean(y_true_ref)),
                    "threshold_optimize_metric": threshold_optimize_metric,
                    "threshold": point_metrics["threshold"],
                    "threshold_ci_low": None,
                    "threshold_ci_high": None,
                    "f1": point_metrics["f1"],
                    "f1_ci_low": None,
                    "f1_ci_high": None,
                    "recall": point_metrics["recall"],
                    "recall_ci_low": None,
                    "recall_ci_high": None,
                    "precision": point_metrics["precision"],
                    "precision_ci_low": None,
                    "precision_ci_high": None,
                    "r": point_metrics["R"],
                    "r_ci_low": None,
                    "r_ci_high": None,
                    "auc": point_metrics["auc"],
                    "auc_ci_low": round_fn(auc_ci.ci_low),
                    "auc_ci_high": round_fn(auc_ci.ci_high),
                    "ap": point_metrics["pr_auc"],
                    "auc_seed_std": seed_metric_std_by_model.get(model_name, {}).get("auc_seed_std"),
                    "pr_auc_seed_std": seed_metric_std_by_model.get(model_name, {}).get("pr_auc_seed_std"),
                    "auc_point_estimate_std": (
                        round_fn(auc_ci.point_estimate_std)
                        if auc_ci.point_estimate_std is not None
                        else None
                    ),
                    "ap_point_estimate_std": (
                        round_fn(ap_ci.point_estimate_std)
                        if ap_ci.point_estimate_std is not None
                        else None
                    ),
                    "ap_ci_low": round_fn(ap_ci.ci_low),
                    "ap_ci_high": round_fn(ap_ci.ci_high),
                    "ci_low_percentile": round_fn(ci_low_pct),
                    "ci_high_percentile": round_fn(ci_high_pct),
                    "bootstrap_sampling": str(bootstrap_preset.sampling),
                    "bootstrap_n_resamples": int(bootstrap_preset.n_resamples_metric),
                    "bootstrap_block_size": (
                        int(bootstrap_preset.block_size)
                        if (
                            bootstrap_preset.sampling == "block"
                            and bootstrap_preset.block_size is not None
                        )
                        else None
                    ),
                    "bootstrap_circular_block": bool(bootstrap_preset.circular_block),
                    "threshold_valid_resamples": None,
                    "f1_valid_resamples": None,
                    "recall_valid_resamples": None,
                    "precision_valid_resamples": None,
                    "r_valid_resamples": None,
                    "auc_valid_resamples": int(auc_ci.valid_resamples),
                    "ap_valid_resamples": int(ap_ci.valid_resamples),
                }
            )

        for model_name in model_names:
            if model_name == effective_baseline:
                continue

            delta_key = (model_name, effective_baseline)
            if (
                delta_key not in auc_result.pairwise_deltas
                or delta_key not in ap_result.pairwise_deltas
            ):
                continue

            baseline_metrics = point_metrics_by_model[effective_baseline]
            model_metrics = point_metrics_by_model[model_name]
            auc_delta = auc_result.pairwise_deltas[delta_key]
            ap_delta = ap_result.pairwise_deltas[delta_key]

            auc_delta_point = _optional_delta(model_metrics["auc"], baseline_metrics["auc"])
            ap_delta_point = _optional_delta(model_metrics["pr_auc"], baseline_metrics["pr_auc"])

            delta_rows.append(
                {
                    "window_id": window_id,
                    "Mf": mf_value,
                    "Tfore": tfore_value,
                    "comparison": f"{model_name}_minus_{effective_baseline}",
                    "threshold_delta": round_fn(_optional_delta(model_metrics["threshold"], baseline_metrics["threshold"])),
                    "threshold_delta_ci_low": None,
                    "threshold_delta_ci_high": None,
                    "f1_delta": round_fn(_optional_delta(model_metrics["f1"], baseline_metrics["f1"])),
                    "f1_delta_ci_low": None,
                    "f1_delta_ci_high": None,
                    "recall_delta": round_fn(_optional_delta(model_metrics["recall"], baseline_metrics["recall"])),
                    "recall_delta_ci_low": None,
                    "recall_delta_ci_high": None,
                    "precision_delta": round_fn(_optional_delta(model_metrics["precision"], baseline_metrics["precision"])),
                    "precision_delta_ci_low": None,
                    "precision_delta_ci_high": None,
                    "r_delta": round_fn(_optional_delta(model_metrics["R"], baseline_metrics["R"])),
                    "r_delta_ci_low": None,
                    "r_delta_ci_high": None,
                    "auc_delta": round_fn(auc_delta_point),
                    "auc_delta_ci_low": round_fn(auc_delta.ci_low),
                    "auc_delta_ci_high": round_fn(auc_delta.ci_high),
                    "ap_delta": round_fn(ap_delta_point),
                    "ap_delta_ci_low": round_fn(ap_delta.ci_low),
                    "ap_delta_ci_high": round_fn(ap_delta.ci_high),
                    "ci_low_percentile": round_fn(ci_low_pct),
                    "ci_high_percentile": round_fn(ci_high_pct),
                    "valid_resamples": int(min(auc_delta.valid_resamples, ap_delta.valid_resamples)),
                }
            )

    metrics_rows = sorted(
        metrics_rows,
        key=lambda row: tuple(row[key] for key in sort_export_by),
    )
    delta_rows = sorted(
        delta_rows,
        key=lambda row: tuple(row[key] for key in sort_delta_by),
    )
    return metrics_rows, delta_rows


def summarize_classification_metric_rows(
    metrics_rows: Sequence[Mapping[str, Any]],
    *,
    round_fn: Callable[[Any], float | None],
) -> dict[str, dict[str, float | None]]:
    """Summarize per-model averages/std from exported metric rows."""

    if len(metrics_rows) == 0:
        return {}

    summary_by_model: dict[str, dict[str, float | None]] = {}
    model_names = sorted({str(row["model"]) for row in metrics_rows})

    def _numeric_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
        out: list[float] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            out.append(float(value))
        return out

    def _safe_mean(values: list[float]) -> float | None:
        if len(values) == 0:
            return None
        return round_fn(np.mean(values))

    def _safe_std(values: list[float]) -> float | None:
        if len(values) == 0:
            return None
        if len(values) == 1:
            return round_fn(0.0)
        return round_fn(np.std(values, ddof=1))

    for model_name in model_names:
        rows_i = [row for row in metrics_rows if str(row["model"]) == model_name]
        threshold_values = _numeric_values(rows_i, "threshold")
        f1_values = _numeric_values(rows_i, "f1")
        recall_values = _numeric_values(rows_i, "recall")
        precision_values = _numeric_values(rows_i, "precision")
        r_values = _numeric_values(rows_i, "r")
        auc_values = _numeric_values(rows_i, "auc")
        ap_values = _numeric_values(rows_i, "ap")

        summary_by_model[model_name] = {
            "mean_threshold": _safe_mean(threshold_values),
            "mean_f1": _safe_mean(f1_values),
            "mean_recall": _safe_mean(recall_values),
            "mean_precision": _safe_mean(precision_values),
            "mean_r": _safe_mean(r_values),
            "mean_auc": _safe_mean(auc_values),
            "std_auc": _safe_std(auc_values),
            "mean_ap": _safe_mean(ap_values),
            "std_ap": _safe_std(ap_values),
            "num_windows": int(len(rows_i)),
        }

    return summary_by_model


def _sort_export_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sort_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Sort export rows with robust coercion for common classifier keys."""

    if len(rows) == 0:
        return []

    def _coerce(key: str, value: Any) -> Any:
        if value is None:
            return ""
        if key == "Mf":
            return float(value)
        if key == "Tfore":
            return int(value)
        if key in {"model", "comparison"}:
            return str(value)
        return value

    return sorted(
        [dict(row) for row in rows],
        key=lambda row: tuple(_coerce(key, row.get(key)) for key in sort_keys),
    )


def run_classification_bootstrap_export_pipeline(
    *,
    multi_results: Sequence[Mapping[str, Any]],
    bootstrap_preset,
    round_fn: Callable[[Any], float | None],
    threshold_optimize_metric: str,
    use_hierarchical_seed_sample: bool,
    export_dir: str | Path,
    export_stem: str,
    export_label: str,
    round_decimals: int,
    seed_aggregation: SeedAggregationMethod = "mean",
    point_estimate_mode: PointEstimateMode = "ensemble",
    baseline_model: str | None = None,
    sort_export_by: Sequence[str] = ("Mf", "Tfore", "model"),
    sort_delta_by: Sequence[str] = ("Mf", "Tfore", "comparison"),
    window_model_rows_getter: Callable[[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]] | None = None,
    bootstrap_metrics: Sequence[str] = ("auc", "ap"),
    metadata: Mapping[str, Any] | None = None,
    ensure_ascii: bool = False,
) -> ClassificationBootstrapExportArtifacts:
    """Run classifier bootstrap export flow and persist CSV/JSON artifacts."""

    ci_low_pct, ci_high_pct = bootstrap_preset.ci
    metrics_rows, delta_rows = build_classification_bootstrap_export_rows(
        multi_results=multi_results,
        bootstrap_preset=bootstrap_preset,
        round_fn=round_fn,
        threshold_optimize_metric=threshold_optimize_metric,
        use_hierarchical_seed_sample=use_hierarchical_seed_sample,
        seed_aggregation=seed_aggregation,
        point_estimate_mode=point_estimate_mode,
        baseline_model=baseline_model,
        sort_export_by=sort_export_by,
        sort_delta_by=sort_delta_by,
        window_model_rows_getter=window_model_rows_getter,
    )
    metrics_rows = _sort_export_rows(metrics_rows, sort_keys=sort_export_by)
    delta_rows = _sort_export_rows(delta_rows, sort_keys=sort_delta_by)

    if len(metrics_rows) == 0:
        raise RuntimeError("No metrics rows were generated.")

    export_dir_path = Path(export_dir)
    export_dir_path.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = export_dir_path / f"{export_stem}_long_{export_label}.csv"
    deltas_csv_path = export_dir_path / f"{export_stem}_pairwise_delta_{export_label}.csv"
    metrics_json_path = export_dir_path / f"{export_stem}_long_{export_label}.json"

    metric_fieldnames = list(metrics_rows[0].keys())
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    delta_csv_out: Path | None = None
    if len(delta_rows) > 0:
        delta_fieldnames = list(delta_rows[0].keys())
        with deltas_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=delta_fieldnames)
            writer.writeheader()
            writer.writerows(delta_rows)
        delta_csv_out = deltas_csv_path

    summary_by_model = summarize_classification_metric_rows(
        metrics_rows,
        round_fn=round_fn,
    )

    json_payload: dict[str, Any] = dict(metadata or {})
    json_payload.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "num_windows": int(len(multi_results)),
            "baseline_model": str(baseline_model) if baseline_model is not None else None,
            "seed_aggregation_method": str(seed_aggregation),
            "bootstrap_point_estimate_mode": str(point_estimate_mode),
            "round_decimals": int(round_decimals),
            "sort_by": list(sort_export_by),
            "pairwise_sort_by": list(sort_delta_by),
            "threshold_optimize_metric": str(threshold_optimize_metric),
            "bootstrap": {
                "sampling": str(bootstrap_preset.sampling),
                "ci_percentiles": [round_fn(ci_low_pct), round_fn(ci_high_pct)],
                "n_resamples_metric": int(bootstrap_preset.n_resamples_metric),
                "bootstrap_metrics": [str(name) for name in bootstrap_metrics],
                "block_size": (
                    int(bootstrap_preset.block_size)
                    if bootstrap_preset.block_size is not None
                    else None
                ),
                "circular_block": bool(bootstrap_preset.circular_block),
            },
            "hierarchical_seed_sample": {
                "enabled": bool(use_hierarchical_seed_sample),
                "seed_resampling": "iid_with_replacement",
                "sample_resampling": str(bootstrap_preset.sampling),
            },
            "metrics_rows": metrics_rows,
            "pairwise_delta_rows": delta_rows,
            "summary_by_model": summary_by_model,
        }
    )
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=bool(ensure_ascii))

    print(f"Bootstrap long CSV exported: {metrics_csv_path}")
    if delta_csv_out is not None:
        print(f"Bootstrap pairwise delta CSV exported: {delta_csv_out}")
    print(f"Bootstrap JSON exported: {metrics_json_path}")
    print(f"Exported rows: metrics={len(metrics_rows)}, deltas={len(delta_rows)}")

    return ClassificationBootstrapExportArtifacts(
        metrics_rows=metrics_rows,
        delta_rows=delta_rows,
        summary_by_model=summary_by_model,
        metrics_csv_path=metrics_csv_path,
        deltas_csv_path=delta_csv_out,
        metrics_json_path=metrics_json_path,
        json_payload=json_payload,
    )


def render_classification_bootstrap_export(
    artifacts: ClassificationBootstrapExportArtifacts,
    *,
    display_columns: Sequence[str] = ("Mf", "Tfore", "model", "auc", "auc_point_estimate_std", "ap", "ap_point_estimate_std", "auc_ci_low", "auc_ci_high", "ap_ci_low", "ap_ci_high"),
) -> None:
    """Display export tables for notebook use."""

    import pandas as pd
    from IPython.display import display

    if len(artifacts.metrics_rows) == 0:
        raise RuntimeError("No metrics rows were generated.")

    display(pd.DataFrame(artifacts.metrics_rows)[list(display_columns)])


def build_rf_em_window_ci_cache(
    plot_results: Sequence[Mapping[str, Any]],
    *,
    bootstrap_preset,
    use_hierarchical_seed_sample: bool,
    seed_aggregation: SeedAggregationMethod = "mean",
    point_estimate_mode: PointEstimateMode = "ensemble",
    left_model_name: str = "RF",
    right_model_name: str = "EM-EQF",
    roc_seed_base: int = 100,
    pr_seed_base: int = 300,
    auc_seed_base: int = 500,
    ap_seed_base: int = 700,
) -> list[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[float, float]]]:
    """Build CI cache for RF/EM-style faceted ROC/PR notebook plots.

    Each returned element contains:
    - ``roc_rf``, ``roc_em``, ``pr_rf``, ``pr_em``: ``(grid, ci_low, ci_high)``
    - ``auc_ci_rf``, ``auc_ci_em``, ``ap_ci_rf``, ``ap_ci_em``: ``(low, high)``
    """

    ci_cache: list[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[float, float]]] = []

    for i, row in enumerate(plot_results):
        model_rows = _default_window_model_rows(
            row,
            left_model_name=left_model_name,
            right_model_name=right_model_name,
        )

        if left_model_name not in model_rows or right_model_name not in model_rows:
            raise KeyError(
                f"Expected both models in row: {left_model_name!r}, {right_model_name!r}. "
                f"Available={list(model_rows.keys())}"
            )

        y_true_left = np.asarray(model_rows[left_model_name]["y_true"]).astype(int).reshape(-1)
        y_true_right = np.asarray(model_rows[right_model_name]["y_true"]).astype(int).reshape(-1)
        y_true = ensure_identical_labels(
            y_true_left,
            y_true_right,
            context=f"window (Mf={row.get('Mf')}, Tfore={row.get('Tfore')})",
        )

        model_predictions = {
            left_model_name: np.asarray(model_rows[left_model_name]["y_prob"], dtype=np.float64).reshape(-1),
            right_model_name: np.asarray(model_rows[right_model_name]["y_prob"], dtype=np.float64).reshape(-1),
        }
        model_seed_predictions = {
            left_model_name: np.asarray(model_rows[left_model_name]["seed_probs"], dtype=np.float64),
            right_model_name: np.asarray(model_rows[right_model_name]["seed_probs"], dtype=np.float64),
        }

        if use_hierarchical_seed_sample:
            roc_result: MultiModelCurveBootstrapResult = (
                bootstrap_multi_model_curve_ci_hierarchical(
                    y_true=y_true,
                    model_seed_predictions=model_seed_predictions,
                    curve="roc",
                    config=bootstrap_preset.curve_config(seed=int(roc_seed_base + i)),
                )
            )
            pr_result: MultiModelCurveBootstrapResult = (
                bootstrap_multi_model_curve_ci_hierarchical(
                    y_true=y_true,
                    model_seed_predictions=model_seed_predictions,
                    curve="pr",
                    config=bootstrap_preset.curve_config(seed=int(pr_seed_base + i)),
                )
            )
        else:
            roc_result = bootstrap_multi_model_curve_ci(
                y_true=y_true,
                model_predictions=model_predictions,
                curve="roc",
                config=bootstrap_preset.curve_config(seed=int(roc_seed_base + i)),
            )
            pr_result = bootstrap_multi_model_curve_ci(
                y_true=y_true,
                model_predictions=model_predictions,
                curve="pr",
                config=bootstrap_preset.curve_config(seed=int(pr_seed_base + i)),
            )

        auc_result = paired_metric_bootstrap(
            y_true=y_true,
            metric_name="auc",
            metric_config=bootstrap_preset.metric_config(seed=int(auc_seed_base + i)),
            use_hierarchical_seed_sample=use_hierarchical_seed_sample,
            seed_aggregation=seed_aggregation,
            point_estimate_mode=point_estimate_mode,
            baseline_model=right_model_name,
            model_predictions=model_predictions,
            model_seed_predictions=model_seed_predictions,
            task="classification",
        )
        ap_result = paired_metric_bootstrap(
            y_true=y_true,
            metric_name="ap",
            metric_config=bootstrap_preset.metric_config(seed=int(ap_seed_base + i)),
            use_hierarchical_seed_sample=use_hierarchical_seed_sample,
            seed_aggregation=seed_aggregation,
            point_estimate_mode=point_estimate_mode,
            baseline_model=right_model_name,
            model_predictions=model_predictions,
            model_seed_predictions=model_seed_predictions,
            task="classification",
        )

        ci_cache.append(
            {
                "roc_rf": (
                    roc_result.grid,
                    roc_result.model_results[left_model_name].ci_low,
                    roc_result.model_results[left_model_name].ci_high,
                ),
                "roc_em": (
                    roc_result.grid,
                    roc_result.model_results[right_model_name].ci_low,
                    roc_result.model_results[right_model_name].ci_high,
                ),
                "pr_rf": (
                    pr_result.grid,
                    pr_result.model_results[left_model_name].ci_low,
                    pr_result.model_results[left_model_name].ci_high,
                ),
                "pr_em": (
                    pr_result.grid,
                    pr_result.model_results[right_model_name].ci_low,
                    pr_result.model_results[right_model_name].ci_high,
                ),
                "auc_ci_rf": (
                    float(auc_result.model_results[left_model_name].ci_low),
                    float(auc_result.model_results[left_model_name].ci_high),
                ),
                "auc_ci_em": (
                    float(auc_result.model_results[right_model_name].ci_low),
                    float(auc_result.model_results[right_model_name].ci_high),
                ),
                "ap_ci_rf": (
                    float(ap_result.model_results[left_model_name].ci_low),
                    float(ap_result.model_results[left_model_name].ci_high),
                ),
                "ap_ci_em": (
                    float(ap_result.model_results[right_model_name].ci_low),
                    float(ap_result.model_results[right_model_name].ci_high),
                ),
            }
        )

    return ci_cache


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
