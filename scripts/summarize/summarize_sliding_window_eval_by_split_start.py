#!/usr/bin/env python3
"""Summarize sliding-window metrics from validation/test split starts.

This script reuses existing per-run ``sliding_window_eval_metrics.json`` files.
It filters each run's saved sliding windows by ``t_forecast >= val_start_t`` and
``t_forecast >= test_start_t`` from the corresponding catalog metadata, then
writes per-run and grouped summary CSV/JSON files.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import re
import csv
import sys
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_METRICS_FILENAME = "sliding_window_eval_metrics.json"
DEFAULT_OUTPUT_PREFIX = "sliding_window_eval_metrics_by_split_start"
METRIC_KEYS = ("coverage", "mae", "rmse", "crps", "w95", "lp_nb", "num_windows")
SEED_PATTERN = re.compile(r"^(?P<base>.+)_seed_(?P<seed>\d+)$")
DATASET_DEFAULT_SPLIT_CONFIGS = {
    "CB_HAB4": {
        "val_start_ts": "2012-11-28 12:00:00",
        "test_start_ts": "2012-11-30 00:00:00",
    },
    "St1-2018": {
        "end_ts": "2018-08-21T23:59:59",
        "val_start_ts": "2018-07-14T00:00:00",
        "test_start_ts": "2018-07-20T00:00:00",
    },
}
FALLBACK_SPLIT_CONFIGS = {
    "PNR_1z": {
        "start_ts": "2018-10-15 08:00:00",
        "end_ts": "2018-12-18 13:00:00",
        "train_start_ts": "2018-10-22",
        "val_start_ts": "2018-11-22",
        "test_start_ts": "2018-12-14",
    },
    "PNR_2": {
        "start_ts": "2019-08-13 15:00:00",
        "end_ts": "2019-10-02 06:00:00",
        "train_start_ts": "2019-08-20",
        "val_start_ts": "2019-08-24",
        "test_start_ts": "2019-09-25",
    },
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value):
    if value is None:
        return None
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(float_value):
        return None
    return float_value


def mean_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    valid_values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not valid_values:
        return None, None
    if len(valid_values) == 1:
        return valid_values[0], 0.0
    mean_value = sum(valid_values) / len(valid_values)
    variance = sum((value - mean_value) ** 2 for value in valid_values) / (len(valid_values) - 1)
    return mean_value, math.sqrt(max(variance, 0.0))


def group_by_key(rows: Sequence[Dict[str, Any]], key_fn):
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        key = key_fn(row)
        if key is None:
            continue
        grouped.setdefault(key, []).append(row)
    return grouped


def build_group_stats_rows(
    grouped: Mapping[Any, Sequence[Dict[str, Any]]],
    metric_keys: Sequence[str],
    base_row_builder,
    sort_key_fn=None,
) -> List[Dict[str, Any]]:
    sorted_items = sorted(grouped.items(), key=sort_key_fn) if sort_key_fn else sorted(grouped.items())
    out_rows: List[Dict[str, Any]] = []
    for group_key_value, items in sorted_items:
        out = dict(base_row_builder(group_key_value, items))
        out["n_runs"] = len(items)
        out["n_success"] = sum(row.get("metrics_found") == 1 for row in items)
        for metric_key in metric_keys:
            values = [row.get(metric_key) for row in items if row.get(metric_key) is not None]
            mean_value, std_value = mean_std(values)
            out[f"{metric_key}_mean"] = mean_value
            out[f"{metric_key}_std"] = std_value
        out_rows.append(out)
    return out_rows


def build_group_best_rows(
    grouped: Mapping[Any, Sequence[Dict[str, Any]]],
    metric: str,
    maximize: bool,
    base_row_builder,
    sort_key_fn=None,
) -> List[Dict[str, Any]]:
    sorted_items = sorted(grouped.items(), key=sort_key_fn) if sort_key_fn else sorted(grouped.items())
    out_rows: List[Dict[str, Any]] = []
    for group_key_value, items in sorted_items:
        candidates = [row for row in items if row.get(metric) is not None]
        if not candidates:
            row = dict(base_row_builder(group_key_value, items, None))
            row["best_metric_name"] = metric
            row["best_metric_value"] = None
            row["best_run"] = None
            row["best_seed"] = None
            out_rows.append(row)
            continue
        best = max(candidates, key=lambda row: float(row[metric])) if maximize else min(candidates, key=lambda row: float(row[metric]))
        row = dict(base_row_builder(group_key_value, items, best))
        row["best_metric_name"] = metric
        row["best_metric_value"] = best.get(metric)
        row["best_run"] = best.get("run")
        row["best_seed"] = best.get("seed")
        out_rows.append(row)
    return out_rows


def split_seed_suffix(run_name: str) -> tuple[str, int | None]:
    matched = SEED_PATTERN.match(str(run_name or ""))
    if matched is None:
        return str(run_name or ""), None
    return matched.group("base"), int(matched.group("seed"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-compute validation/test-start sliding-window metrics from "
            "existing per-run sliding_window_eval_metrics.json files."
        )
    )
    parser.add_argument(
        "exp_dir",
        nargs="?",
        default="experiments/etas_multi_ds_bg_norm_0.2",
        help="Experiment directory containing runs/. Defaults to ETAS multi-dataset experiment.",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Runs directory. Defaults to <exp_dir>/runs. If set, exp_dir is its parent.",
    )
    parser.add_argument(
        "--metrics-filename",
        default=DEFAULT_METRICS_FILENAME,
        help=f"Metrics JSON filename under each run. Default: {DEFAULT_METRICS_FILENAME}.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory containing dataset folders. Default: data.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output report directory. Defaults to <exp_dir>/reports.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"Output filename prefix. Default: {DEFAULT_OUTPUT_PREFIX}.",
    )
    parser.add_argument(
        "--best-metric",
        default="test_start_rmse",
        help="Metric used for group-best rows. Default: test_start_rmse.",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="Maximize --best-metric instead of minimizing it.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include runs with missing metrics/config in the per-run table.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read run config YAML files.") from exc
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def compute_mean_nb_log_prob(obs_counts, sim_count_matrix, *, eps: float = 1e-10):
    """Compute the same method-of-moments NB log probability as forecast_eval."""
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
        obs_active = obs[active].astype(np.float64)
        mu_active = np.clip(mu[active], eps, None)
        var_active = np.maximum(var[active], mu_active + eps)
        total_count = np.clip((mu_active * mu_active) / (var_active - mu_active), eps, 1e12)
        log_prob_bins[active] = (
            np.asarray(
                [
                    math.lgamma(float(k + r)) - math.lgamma(float(r)) - math.lgamma(float(k + 1.0))
                    for k, r in zip(obs_active, total_count)
                ],
                dtype=np.float64,
            )
            + total_count * (np.log(total_count) - np.log(total_count + mu_active))
            + obs_active * (np.log(mu_active) - np.log(total_count + mu_active))
        )

    return float(np.mean(log_prob_bins)), log_prob_bins


def compute_mean_crps(obs_counts, sim_count_matrix):
    """Compute empirical CRPS using the same formula as forecast_eval."""
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
    return float(np.mean(crps_bins)), crps_bins


def discover_run_dirs(exp_dir: Path, runs_dir: Path) -> List[Path]:
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs directory not found: {runs_dir}")
    return sorted(
        [candidate.resolve() for candidate in runs_dir.iterdir() if candidate.is_dir()],
        key=lambda path: path.name,
    )


def variant_and_seed_from_run_name(run_name: str) -> tuple[str, int | None]:
    variant_name, seed = split_seed_suffix(run_name)
    if seed is not None:
        return variant_name, seed
    matched = SEED_PATTERN.match(run_name)
    if matched is None:
        return run_name, None
    return matched.group("base"), int(matched.group("seed"))


def find_config_path(run_dir: Path) -> Path | None:
    for filename in ("config_input.yaml", "config.yaml"):
        candidate = run_dir / filename
        if candidate.exists():
            return candidate
    return None


def dataset_and_catalog_cfg(run_dir: Path) -> tuple[str | None, Dict[str, Any], Path | None]:
    config_path = find_config_path(run_dir)
    if config_path is None:
        return None, {}, None
    config = load_yaml_mapping(config_path)
    dataset = config.get("dataset")
    catalog_cfg = config.get("catalog_cfg")
    return (
        str(dataset) if dataset is not None else None,
        catalog_cfg if isinstance(catalog_cfg, dict) else {},
        config_path,
    )


def coerce_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def clip_timestamp(ts: pd.Timestamp, low: pd.Timestamp, high: pd.Timestamp) -> pd.Timestamp:
    if ts < low:
        return low
    if ts > high:
        return high
    return ts


def resolve_split_metadata_from_summary(
    dataset: str,
    catalog_cfg: Mapping[str, Any],
    *,
    data_root: Path,
) -> Dict[str, Any]:
    summary_path = data_root / dataset / "processed" / f"{dataset}_summary.json"
    if not summary_path.exists():
        fallback = FALLBACK_SPLIT_CONFIGS.get(dataset)
        if fallback is None:
            return {
                "registry_name": "",
                "val_start_t": None,
                "test_start_t": None,
                "split_status": "missing_dataset_summary",
                "split_source_path": str(summary_path),
            }
        summary = {
            "start_time_iso": fallback["start_ts"],
            "end_time_iso": fallback["end_ts"],
        }
        effective_catalog_cfg = {**fallback, **dict(catalog_cfg)}
        summary_source_path = f"fallback:{dataset}"
    else:
        summary = read_json(summary_path)
        effective_catalog_cfg = {
            **DATASET_DEFAULT_SPLIT_CONFIGS.get(dataset, {}),
            **dict(catalog_cfg),
        }
        summary_source_path = str(summary_path)

    if not isinstance(summary, dict):
        return {
            "registry_name": "",
            "val_start_t": None,
            "test_start_t": None,
            "split_status": "invalid_dataset_summary",
            "split_source_path": summary_source_path,
        }

    try:
        start_ts = coerce_timestamp(summary["start_time_iso"])
        summary_end_ts = coerce_timestamp(summary["end_time_iso"])
        end_ts = summary_end_ts
        if effective_catalog_cfg.get("end_ts") is not None:
            clipped_end = clip_timestamp(coerce_timestamp(effective_catalog_cfg.get("end_ts")), start_ts, summary_end_ts)
            if clipped_end <= start_ts:
                raise ValueError(f"end_ts={clipped_end} must be after start_ts={start_ts}")
            end_ts = clipped_end

        duration = end_ts - start_ts
        train_start_ts = coerce_timestamp(effective_catalog_cfg.get("train_start_ts", start_ts))
        val_start_ts = coerce_timestamp(effective_catalog_cfg.get("val_start_ts", start_ts + duration * 0.70))
        test_start_ts = coerce_timestamp(effective_catalog_cfg.get("test_start_ts", start_ts + duration * 0.85))
        train_start_ts = clip_timestamp(train_start_ts, start_ts, end_ts)
        val_start_ts = clip_timestamp(val_start_ts, train_start_ts, end_ts)
        test_start_ts = clip_timestamp(test_start_ts, val_start_ts, end_ts)

        unit_td = pd.Timedelta(str(effective_catalog_cfg.get("freq", "1h")))
        if unit_td <= pd.Timedelta(0):
            raise ValueError(f"freq must be positive, got {effective_catalog_cfg.get('freq')!r}")

        val_start_t = float((val_start_ts - start_ts) / unit_td)
        test_start_t = float((test_start_ts - start_ts) / unit_td)
    except Exception as exc:
        return {
            "registry_name": "",
            "val_start_t": None,
            "test_start_t": None,
            "split_status": f"split_error:{type(exc).__name__}",
            "split_source_path": summary_source_path,
        }

    return {
        "registry_name": f"{dataset}-Standard",
        "val_start_t": val_start_t,
        "test_start_t": test_start_t,
        "val_start_ts": val_start_ts.isoformat(),
        "test_start_ts": test_start_ts.isoformat(),
        "split_status": "ok",
        "split_source_path": summary_source_path,
    }


def split_metadata_for_run(
    run_dir: Path,
    *,
    data_root: Path,
    cache: Dict[tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    dataset, catalog_cfg, config_path = dataset_and_catalog_cfg(run_dir)
    if not dataset:
        return {
            "config_path": str(config_path) if config_path else "",
            "dataset": None,
            "split_status": "missing_dataset",
        }

    cache_key = (dataset, json.dumps(catalog_cfg, sort_keys=True, default=str))
    if cache_key not in cache:
        cache[cache_key] = resolve_split_metadata_from_summary(
            dataset,
            catalog_cfg=catalog_cfg,
            data_root=data_root,
        )

    metadata = cache[cache_key]
    return {
        "config_path": str(config_path) if config_path else "",
        "dataset": dataset,
        "registry_name": metadata.get("registry_name"),
        "val_start_t": to_float(metadata.get("val_start_t")),
        "test_start_t": to_float(metadata.get("test_start_t")),
        "val_start_ts": metadata.get("val_start_ts"),
        "test_start_ts": metadata.get("test_start_ts"),
        "split_status": metadata.get("split_status"),
        "split_source_path": metadata.get("split_source_path"),
    }


def finite_float(value: Any) -> float | None:
    number = to_float(value)
    if number is None or not math.isfinite(number):
        return None
    return float(number)


def rows_from_windows(metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    windows = metrics.get("windows")
    if not isinstance(windows, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in windows:
        if not isinstance(item, dict):
            continue
        t_forecast = finite_float(item.get("t_forecast"))
        count = finite_float(item.get("count"))
        forecast_mean = finite_float(item.get("forecast_mean"))
        q_low = finite_float(item.get("q_low"))
        q_high = finite_float(item.get("q_high"))
        if (
            t_forecast is None
            or count is None
            or forecast_mean is None
            or q_low is None
            or q_high is None
        ):
            continue
        rows.append(
            {
                "t_forecast": t_forecast,
                "count": count,
                "forecast_mean": forecast_mean,
                "q_low": q_low,
                "q_high": q_high,
            }
        )
    return rows


def cache_path_from_metrics(metrics: Mapping[str, Any], metrics_path: Path) -> Path | None:
    raw_path = metrics.get("sliding_cache_path")
    candidates: List[Path] = []
    if raw_path:
        raw = Path(str(raw_path)).expanduser()
        candidates.append(metrics_path.parent / raw.name)
        candidates.append(raw if raw.is_absolute() else metrics_path.parent / raw)

    cache_filename = None
    settings = metrics.get("settings")
    if isinstance(settings, dict):
        cache_filename = settings.get("cache_filename")
    if cache_filename:
        candidates.append(metrics_path.parent / str(cache_filename))

    candidates.extend(sorted(metrics_path.parent.glob("sliding_window_cache*.npz")))

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    return None


def rows_from_cache(metrics: Mapping[str, Any], metrics_path: Path) -> List[Dict[str, Any]]:
    """Recover window-level fields from a compatible sliding-window cache."""
    cache_path = cache_path_from_metrics(metrics, metrics_path)
    if cache_path is None:
        return []

    required_keys = {"t_forecast_list", "counts_list", "q_list", "mean_list"}
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            if not required_keys.issubset(payload.files):
                return []
            t_forecast = np.asarray(payload["t_forecast_list"], dtype=np.float64)
            counts = np.asarray(payload["counts_list"], dtype=np.float64)
            quantiles = np.asarray(payload["q_list"], dtype=np.float64)
            mean = np.asarray(payload["mean_list"], dtype=np.float64)
    except Exception:
        return []

    n_windows = t_forecast.shape[0] if t_forecast.ndim == 1 else 0
    if (
        n_windows == 0
        or counts.ndim != 1
        or mean.ndim != 1
        or quantiles.ndim != 2
        or quantiles.shape != (n_windows, 2)
        or counts.shape[0] != n_windows
        or mean.shape[0] != n_windows
    ):
        return []

    values = np.column_stack((t_forecast, counts, mean, quantiles))
    if not np.all(np.isfinite(values)):
        return []

    return [
        {
            "t_forecast": float(t_forecast[index]),
            "count": float(counts[index]),
            "forecast_mean": float(mean[index]),
            "q_low": float(quantiles[index, 0]),
            "q_high": float(quantiles[index, 1]),
        }
        for index in range(n_windows)
    ]


def load_cache_subset(metrics: Mapping[str, Any], metrics_path: Path, mask: np.ndarray):
    cache_path = cache_path_from_metrics(metrics, metrics_path)
    if cache_path is None:
        return None, None, ""
    try:
        payload = np.load(cache_path, allow_pickle=False)
    except Exception:
        return None, None, str(cache_path)
    if "sim_count_matrix" not in payload.files:
        return None, None, str(cache_path)
    sim = np.asarray(payload["sim_count_matrix"])
    if sim.ndim != 2 or sim.shape[0] != mask.shape[0]:
        return None, None, str(cache_path)
    return sim[mask], payload, str(cache_path)


def compute_subset_metrics(
    metrics: Mapping[str, Any],
    metrics_path: Path,
    *,
    start_t: float | None,
) -> Dict[str, Any]:
    if start_t is None:
        return {
            "status": "missing_split_start",
            "num_windows": 0,
        }
    rows = rows_from_windows(metrics)
    if not rows:
        rows = rows_from_cache(metrics, metrics_path)
        if not rows:
            return {
                "status": "missing_windows",
                "num_windows": 0,
            }

    t_forecast = np.asarray([row["t_forecast"] for row in rows], dtype=np.float64)
    counts = np.asarray([row["count"] for row in rows], dtype=np.float64)
    mean = np.asarray([row["forecast_mean"] for row in rows], dtype=np.float64)
    q_low = np.asarray([row["q_low"] for row in rows], dtype=np.float64)
    q_high = np.asarray([row["q_high"] for row in rows], dtype=np.float64)

    mask = t_forecast >= float(start_t)
    if not bool(np.any(mask)):
        return {
            "status": "empty",
            "start_t": float(start_t),
            "num_windows": 0,
        }

    obs = counts[mask]
    pred = mean[mask]
    low = q_low[mask]
    high = q_high[mask]
    errors = obs - pred

    out: Dict[str, Any] = {
        "status": "ok",
        "start_t": float(start_t),
        "num_windows": int(mask.sum()),
        "first_t_forecast": float(t_forecast[mask][0]),
        "last_t_forecast": float(t_forecast[mask][-1]),
        "coverage": float(np.mean((obs >= low) & (obs <= high))),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "w95": float(np.mean(high - low)),
    }

    sim_subset, _cache_payload, cache_path = load_cache_subset(metrics, metrics_path, mask)
    out["sliding_cache_path"] = cache_path
    out["cache_found"] = int(sim_subset is not None)
    if sim_subset is not None:
        try:
            lp_nb, _ = compute_mean_nb_log_prob(obs, sim_subset)
            out["lp_nb"] = lp_nb
        except Exception:
            out["lp_nb"] = None
        try:
            crps, _ = compute_mean_crps(obs, sim_subset)
            out["crps"] = crps
        except Exception:
            out["crps"] = None
    else:
        out["lp_nb"] = None
        out["crps"] = None

    return out


def prefixed_subset_fields(prefix: str, subset: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        f"{prefix}_status": subset.get("status"),
        f"{prefix}_start_t": subset.get("start_t"),
        f"{prefix}_first_t_forecast": subset.get("first_t_forecast"),
        f"{prefix}_last_t_forecast": subset.get("last_t_forecast"),
        f"{prefix}_cache_found": subset.get("cache_found", 0),
    }
    for metric_key in METRIC_KEYS:
        fields[f"{prefix}_{metric_key}"] = subset.get(metric_key)
    return fields


def collect_rows(
    run_dirs: Sequence[Path],
    *,
    metrics_filename: str,
    data_root: Path,
    include_missing: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    catalog_cache: Dict[tuple[str, str], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        metrics_path = run_dir / metrics_filename
        metrics = read_json(metrics_path) if metrics_path.exists() else {}
        metrics_found = int(isinstance(metrics, dict) and bool(metrics))
        if not metrics_found and not include_missing:
            continue

        run_name = run_dir.name
        variant_name, seed = variant_and_seed_from_run_name(run_name)
        split_meta = split_metadata_for_run(
            run_dir,
            data_root=data_root,
            cache=catalog_cache,
        )
        row: Dict[str, Any] = {
            "run": run_name,
            "variant_name": variant_name,
            "seed": seed,
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path) if metrics_path.exists() else "",
            "metrics_found": metrics_found,
            "source_status": metrics.get("status") if isinstance(metrics, dict) else None,
            **split_meta,
        }

        if isinstance(metrics, dict) and metrics:
            val_subset = compute_subset_metrics(
                metrics,
                metrics_path,
                start_t=split_meta.get("val_start_t"),
            )
            test_subset = compute_subset_metrics(
                metrics,
                metrics_path,
                start_t=split_meta.get("test_start_t"),
            )
        else:
            val_subset = {"status": "missing_metrics", "num_windows": 0}
            test_subset = {"status": "missing_metrics", "num_windows": 0}

        row.update(prefixed_subset_fields("val_start", val_subset))
        row.update(prefixed_subset_fields("test_start", test_subset))
        rows.append(row)

        for split_name, subset in (("val_start", val_subset), ("test_start", test_subset)):
            detail_row = {
                "run": run_name,
                "variant_name": variant_name,
                "seed": seed,
                "split": split_name,
                "dataset": split_meta.get("dataset"),
                "registry_name": split_meta.get("registry_name"),
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path) if metrics_path.exists() else "",
            }
            detail_row.update(subset)
            detail_rows.append(detail_row)

    rows.sort(
        key=lambda item: (
            str(item.get("variant_name") or ""),
            int(item["seed"]) if item.get("seed") is not None else 10**9,
            str(item.get("run") or ""),
        )
    )
    detail_rows.sort(
        key=lambda item: (
            str(item.get("variant_name") or ""),
            int(item["seed"]) if item.get("seed") is not None else 10**9,
            str(item.get("split") or ""),
        )
    )
    return rows, detail_rows


def group_key(row: Mapping[str, Any]):
    variant_name = str(row.get("variant_name") or "").strip()
    return variant_name or None


def numeric_metric_keys(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    keys: List[str] = []
    for prefix in ("val_start", "test_start"):
        for metric_key in METRIC_KEYS:
            key = f"{prefix}_{metric_key}"
            values = [row.get(key) for row in rows]
            if any(to_float(value) is not None for value in values):
                keys.append(key)
    return keys


def build_group_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = group_by_key(rows, group_key)
    metric_keys = numeric_metric_keys(rows)

    def base_row_builder(group_value, _items: Sequence[Dict[str, Any]]):
        return {"variant_name": str(group_value)}

    return build_group_stats_rows(
        grouped=grouped,
        metric_keys=metric_keys,
        base_row_builder=base_row_builder,
        sort_key_fn=lambda item: str(item[0]),
    )


def build_best_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    best_metric: str,
    maximize: bool,
) -> List[Dict[str, Any]]:
    grouped = group_by_key(rows, group_key)

    def base_row_builder(group_value, _items: Sequence[Dict[str, Any]], best):
        row = {"variant_name": str(group_value)}
        if best is not None:
            for prefix in ("val_start", "test_start"):
                for metric_key in METRIC_KEYS:
                    key = f"{prefix}_{metric_key}"
                    row[key] = best.get(key)
        return row

    return build_group_best_rows(
        grouped=grouped,
        metric=best_metric,
        maximize=maximize,
        base_row_builder=base_row_builder,
        sort_key_fn=lambda item: str(item[0]),
    )


def summary_payload(rows: Sequence[Dict[str, Any]], detail_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "num_runs": len(rows),
        "num_detail_rows": len(detail_rows),
        "num_metrics_found": int(sum(row.get("metrics_found") == 1 for row in rows)),
        "num_val_start_ok": int(sum(row.get("val_start_status") == "ok" for row in rows)),
        "num_test_start_ok": int(sum(row.get("test_start_status") == "ok" for row in rows)),
        "datasets": sorted({str(row.get("dataset")) for row in rows if row.get("dataset")}),
    }


def main() -> int:
    args = parse_args()
    exp_dir = resolve_path(args.exp_dir)
    runs_dir = resolve_path(args.runs_dir) if args.runs_dir else exp_dir / "runs"
    if args.runs_dir:
        exp_dir = runs_dir.parent
    output_dir = resolve_path(args.output_dir) if args.output_dir else exp_dir / "reports"
    data_root = resolve_path(args.data_root)

    run_dirs = discover_run_dirs(exp_dir, runs_dir)
    rows, detail_rows = collect_rows(
        run_dirs,
        metrics_filename=args.metrics_filename,
        data_root=data_root,
        include_missing=bool(args.include_missing),
    )
    group_rows = build_group_rows(rows)
    best_rows = build_best_rows(
        rows,
        best_metric=args.best_metric,
        maximize=bool(args.maximize),
    )
    summary = summary_payload(rows, detail_rows)

    prefix = args.output_prefix
    per_run_path = output_dir / f"{prefix}_per_run.csv"
    detail_path = output_dir / f"{prefix}_long.csv"
    group_path = output_dir / f"{prefix}_group_mean_std.csv"
    best_path = output_dir / f"{prefix}_group_best.csv"
    summary_path = output_dir / f"{prefix}_summary.json"

    write_csv(per_run_path, rows)
    write_csv(detail_path, detail_rows)
    write_csv(group_path, group_rows)
    write_csv(best_path, best_rows)
    write_json(summary_path, summary)

    print(f"Runs scanned: {len(run_dirs)}")
    print(f"Rows written: {len(rows)}")
    print(f"Val-start OK: {summary['num_val_start_ok']}")
    print(f"Test-start OK: {summary['num_test_start_ok']}")
    print(f"Wrote: {per_run_path}")
    print(f"Wrote: {detail_path}")
    print(f"Wrote: {group_path}")
    print(f"Wrote: {best_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
