#!/usr/bin/env python3
"""Summarize sliding-window forecast metrics.

This script scans an experiment directory or a single checkpoint/run directory,
collects a configurable per-run sliding-window metrics JSON file, and exports:
1) a per-run table,
2) a grouped mean/std table,
3) a grouped best table.

The output filename prefix defaults to the metrics filename stem.  Therefore
the default metrics file keeps the historical report names, while a file such
as ``sliding_window_eval_metrics_test_trunc.json`` gets a separate report set.
"""

from __future__ import annotations

from pathlib import Path
import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


import argparse
import copy
import math
import re
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from automation import (
    build_group_best_rows,
    build_group_stats_rows,
    group_by_key,
    read_json,
    resolve_best_metric_name,
    split_seed_suffix,
    to_float,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_FILENAME = "sliding_window_eval_metrics.json"
DEFAULT_FIELD_KEYS = (
    "coverage",
    "mae",
    "rmse",
    "crps",
    "w95",
    "lp_nb",
    "mag_max_mae",
    "num_windows",
    "summary_filter.name",
    "summary_filter.source_num_windows",
    "summary_filter.num_windows_excluded",
    "summary_filter.recomputed_from_cache",
    "settings.duration",
    "settings.step",
    "settings.eval_range",
    "settings.include_truncated_final_window",
    "settings.samples_per_batch",
    "settings.predict_b",
    "evaluation_range.name",
    "evaluation_range.start",
    "evaluation_range.end",
    "metadata.dataset",
    "metadata.registry_name",
    "metadata.seed",
    "sliding_loaded_from_cache",
)
SEED_PATTERN = re.compile(r"^(?P<base>.+)_seed_(?P<seed>\d+)$")


def parse_csv(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def default_output_prefix(metrics_filename: str | Path) -> str:
    """Return a stable report prefix derived from a metrics filename."""
    stem = Path(str(metrics_filename)).stem.strip()
    return stem or "sliding_window_eval_metrics"


def lookup_path(mapping: Dict[str, Any], key_path: str):
    current: Any = mapping
    for part in str(key_path).split("."):
        if not isinstance(current, dict):
            return None
        if part in current:
            current = current[part]
            continue

        part_lower = part.lower()
        matched_key = None
        for key in current:
            if str(key).lower() == part_lower:
                matched_key = key
                break
        if matched_key is None:
            return None
        current = current[matched_key]
    return current


def normalize_value(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return str(value)
    return value


def metric_value(metrics: Dict[str, Any], key_path: str):
    value = lookup_path(metrics, key_path)
    normalized = normalize_value(value)
    numeric_value = to_float(normalized)
    if numeric_value is not None:
        return numeric_value
    return normalized


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def path_exists(path: Path) -> bool:
    """Return whether *path* exists without failing on an inaccessible parent."""
    try:
        return path.exists()
    except OSError:
        return False


def path_is_dir(path: Path) -> bool:
    """Return whether *path* is a directory without propagating access errors."""
    try:
        return path.is_dir()
    except OSError:
        return False


def read_json_mapping(path: Path | None) -> Dict[str, Any]:
    if path is None or not path_exists(path):
        return {}
    try:
        payload = read_json(path)
    except OSError:
        return {}
    return payload if isinstance(payload, dict) else {}


def status_ok_from_metrics(metrics: Dict[str, Any]) -> int:
    if not metrics:
        return 0
    status = str(metrics.get("status", "")).strip().lower()
    if status:
        return int(status == "ok")
    return 1


def variant_and_seed_from_run_name(run_name: str):
    variant_name, seed = split_seed_suffix(run_name)
    if seed is not None:
        return variant_name, seed
    matched = SEED_PATTERN.match(run_name)
    if matched is None:
        return run_name, None
    return matched.group("base"), int(matched.group("seed"))


def cache_path_from_metrics(metrics: Mapping[str, Any], metrics_path: Path) -> Path | None:
    """Resolve the sliding-window cache referenced by a metrics JSON file."""
    candidates: List[Path] = []

    raw_path = metrics.get("sliding_cache_path")
    if raw_path:
        raw = Path(str(raw_path)).expanduser()
        # Prefer a same-directory cache with the recorded filename. Absolute
        # cache paths in metrics JSON can point to the machine that generated
        # the experiment, while the run directory itself may have been copied.
        candidates.append(metrics_path.parent / raw.name)
        candidates.append(raw if raw.is_absolute() else metrics_path.parent / raw)

    settings = metrics.get("settings")
    if isinstance(settings, Mapping):
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
        total_count = np.clip(
            (mu_active * mu_active) / (var_active - mu_active),
            eps,
            1e12,
        )
        log_prob_bins[active] = (
            np.asarray(
                [
                    math.lgamma(float(k + r))
                    - math.lgamma(float(r))
                    - math.lgamma(float(k + 1.0))
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


def _status_payload_for_truncated_filter(
    metrics: Mapping[str, Any],
    status: str,
    *,
    cache_path: Path | None = None,
) -> Dict[str, Any]:
    source_num_windows = to_float(metrics.get("num_windows"))
    payload = copy.deepcopy(dict(metrics))
    payload["status"] = status
    payload["num_windows"] = 0
    for key in ("coverage", "mae", "rmse", "crps", "w95", "lp_nb", "mag_max_mae"):
        payload[key] = None
    payload["summary_filter"] = {
        "name": "exclude_truncated_final_window",
        "source_num_windows": source_num_windows,
        "num_windows_excluded": None,
        "recomputed_from_cache": 0,
        "cache_path": str(cache_path) if cache_path is not None else "",
    }
    return payload


def _window_end_from_metrics(
    metrics: Mapping[str, Any],
    *,
    expected_windows: int,
) -> np.ndarray | None:
    windows = metrics.get("windows")
    if not isinstance(windows, list) or len(windows) != expected_windows:
        return None

    values: List[float] = []
    for item in windows:
        if not isinstance(item, Mapping):
            return None
        end_value = to_float(item.get("t_window_end"))
        if end_value is None:
            return None
        values.append(float(end_value))
    return np.asarray(values, dtype=np.float64)


def metrics_excluding_truncated_final_window(
    metrics: Mapping[str, Any],
    metrics_path: Path,
) -> Dict[str, Any]:
    """Return metrics recomputed after dropping final clipped windows.

    This is a reporting-time filter.  It never rewrites the original metrics
    JSON or cache and relies on the cache's per-window simulation matrix for
    CRPS/LP_NB recomputation.
    """
    cache_path = cache_path_from_metrics(metrics, metrics_path)
    if cache_path is None:
        return _status_payload_for_truncated_filter(
            metrics,
            "missing_cache_for_exclude_truncated_final_window",
        )

    required = {"t_forecast_list", "counts_list", "q_list", "mean_list", "sim_count_matrix"}
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            if not required.issubset(set(payload.files)):
                return _status_payload_for_truncated_filter(
                    metrics,
                    "incompatible_cache_for_exclude_truncated_final_window",
                    cache_path=cache_path,
                )
            t_forecast = np.asarray(payload["t_forecast_list"], dtype=np.float64)
            counts = np.asarray(payload["counts_list"], dtype=np.float64)
            q_list = np.asarray(payload["q_list"], dtype=np.float64)
            mean = np.asarray(payload["mean_list"], dtype=np.float64)
            sim = np.asarray(payload["sim_count_matrix"], dtype=np.float64)
            duration = (
                float(np.asarray(payload["duration"]).item())
                if "duration" in payload.files
                else to_float(lookup_path(dict(metrics), "settings.duration"))
            )
            evaluation_end = (
                float(np.asarray(payload["evaluation_end"]).item())
                if "evaluation_end" in payload.files
                else to_float(lookup_path(dict(metrics), "evaluation_range.end"))
            )
    except Exception:
        return _status_payload_for_truncated_filter(
            metrics,
            "cache_read_error_for_exclude_truncated_final_window",
            cache_path=cache_path,
        )

    n_windows = int(t_forecast.shape[0]) if t_forecast.ndim == 1 else 0
    if (
        n_windows == 0
        or counts.ndim != 1
        or mean.ndim != 1
        or q_list.ndim != 2
        or sim.ndim != 2
        or counts.shape[0] != n_windows
        or mean.shape[0] != n_windows
        or q_list.shape[0] != n_windows
        or q_list.shape[1] < 2
        or sim.shape[0] != n_windows
        or duration is None
        or evaluation_end is None
    ):
        return _status_payload_for_truncated_filter(
            metrics,
            "incompatible_cache_for_exclude_truncated_final_window",
            cache_path=cache_path,
        )

    duration = float(duration)
    evaluation_end = float(evaluation_end)
    if not (math.isfinite(duration) and duration > 0 and math.isfinite(evaluation_end)):
        return _status_payload_for_truncated_filter(
            metrics,
            "invalid_bounds_for_exclude_truncated_final_window",
            cache_path=cache_path,
        )

    window_end = _window_end_from_metrics(metrics, expected_windows=n_windows)
    if window_end is None:
        window_end = np.minimum(t_forecast + duration, evaluation_end)

    tolerance = 1e-9 * max(
        1.0,
        abs(duration),
        abs(evaluation_end),
        float(np.nanmax(np.abs(t_forecast))) if t_forecast.size else 0.0,
    )
    window_duration = window_end - t_forecast
    keep_mask = window_duration >= (duration - tolerance)
    excluded = int(n_windows - int(np.sum(keep_mask)))

    out = copy.deepcopy(dict(metrics))
    out["summary_filter"] = {
        "name": "exclude_truncated_final_window",
        "source_num_windows": n_windows,
        "num_windows_excluded": excluded,
        "recomputed_from_cache": 1,
        "cache_path": str(cache_path),
    }

    if not bool(np.any(keep_mask)):
        out["status"] = "empty_after_exclude_truncated_final_window"
        out["num_windows"] = 0
        for key in ("coverage", "mae", "rmse", "crps", "w95", "lp_nb", "mag_max_mae"):
            out[key] = None
        out["windows"] = []
        return out

    obs = counts[keep_mask]
    pred = mean[keep_mask]
    low = q_list[keep_mask, 0]
    high = q_list[keep_mask, 1]
    sim_subset = sim[keep_mask]
    errors = obs - pred

    out["status"] = "ok"
    out["num_windows"] = int(obs.shape[0])
    out["coverage"] = float(np.mean((obs >= low) & (obs <= high)))
    out["mae"] = float(np.mean(np.abs(errors)))
    out["rmse"] = float(np.sqrt(np.mean(errors ** 2)))
    out["w95"] = float(np.mean(high - low))
    try:
        out["lp_nb"], _ = compute_mean_nb_log_prob(obs.astype(np.int64), sim_subset)
    except Exception:
        out["lp_nb"] = None
    try:
        out["crps"], _ = compute_mean_crps(obs, sim_subset)
    except Exception:
        out["crps"] = None

    kept_indices = np.flatnonzero(keep_mask)
    out["windows"] = [
        {
            "t_forecast": float(t_forecast[index]),
            "t_window_end": float(window_end[index]),
            "window_duration": float(window_duration[index]),
            "count": int(counts[index]),
            "forecast_mean": float(mean[index]),
            "q_low": float(q_list[index, 0]),
            "q_high": float(q_list[index, 1]),
            "covered": bool(q_list[index, 0] <= counts[index] <= q_list[index, 1]),
            "error": float(counts[index] - mean[index]),
            "abs_error": float(abs(counts[index] - mean[index])),
        }
        for index in kept_indices
    ]
    return out


def load_summary_rows_if_available(exp_dir: Path) -> List[Dict[str, Any]]:
    summary_path = exp_dir / "summary.json"
    if not path_exists(summary_path):
        return []
    payload = read_json(summary_path)
    if not isinstance(payload, list):
        raise ValueError(f"summary.json must be a list: {summary_path}")
    return [item for item in payload if isinstance(item, dict)]


def discover_run_dirs(exp_dir: Path, metrics_filename: str) -> List[tuple[str, Path, Dict[str, Any]]]:
    runs_dir = exp_dir / "runs"
    local_run_dirs = (
        {
            candidate.name: candidate.resolve()
            for candidate in runs_dir.iterdir()
            if path_is_dir(candidate)
        }
        if path_is_dir(runs_dir)
        else {}
    )

    summary_rows = load_summary_rows_if_available(exp_dir)
    if summary_rows:
        discovered: List[tuple[str, Path, Dict[str, Any]]] = []
        discovered_names = set()
        for item in summary_rows:
            run_name = item.get("run")
            if not run_name:
                continue
            run_name = str(run_name)
            if run_name in discovered_names:
                continue

            local_run_dir = local_run_dirs.get(run_name)
            raw_run_dir = item.get("run_dir")
            # A local run directory is authoritative. The matching entry in
            # summary.json may be an accessible-but-unreadable directory from
            # the machine that created the experiment.
            if local_run_dir is not None:
                run_dir = local_run_dir
            elif raw_run_dir:
                run_dir = Path(str(raw_run_dir)).expanduser()
                if not run_dir.is_absolute():
                    run_dir = exp_dir / run_dir
            else:
                run_dir = runs_dir / run_name
            discovered.append((run_name, run_dir.resolve(), item))

            discovered_names.add(run_name)

        # summary.json may predate recently completed runs. Include every
        # local run directory so the report reflects the experiment on disk.
        discovered.extend(
            (run_name, run_dir, {})
            for run_name, run_dir in sorted(local_run_dirs.items())
            if run_name not in discovered_names
        )
        return discovered

    if local_run_dirs:
        return [(run_name, run_dir, {}) for run_name, run_dir in sorted(local_run_dirs.items())]

    if path_exists(exp_dir / metrics_filename):
        return [(exp_dir.name, exp_dir.resolve(), {})]

    return []


def collect_rows(
    exp_dir: Path,
    *,
    field_keys: Sequence[str],
    metrics_filename: str,
    include_missing: bool,
    exclude_truncated_final_window: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_name, run_dir, summary_item in discover_run_dirs(exp_dir, metrics_filename):
        metrics_path = run_dir / metrics_filename
        metrics = read_json_mapping(metrics_path)
        if not metrics and not include_missing:
            continue
        source_metrics_found = bool(metrics)
        if metrics and exclude_truncated_final_window:
            metrics = metrics_excluding_truncated_final_window(metrics, metrics_path)

        variant_name, seed_from_name = variant_and_seed_from_run_name(run_name)
        row_seed = summary_item.get("seed", seed_from_name)

        row: Dict[str, Any] = {
            "run": run_name,
            "variant_name": variant_name,
            "seed": row_seed,
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path) if path_exists(metrics_path) else "",
            "status": metrics.get("status") if metrics else None,
            "status_ok": status_ok_from_metrics(metrics),
            "metrics_found": int(source_metrics_found),
        }

        for key in field_keys:
            row[key] = metric_value(metrics, key) if metrics else None

        rows.append(row)

    rows.sort(
        key=lambda row: (
            str(row.get("variant_name") or ""),
            int(row["seed"]) if row.get("seed") is not None else 10**9,
            str(row.get("run") or ""),
        )
    )
    return rows


def numeric_field_keys(rows: Sequence[Dict[str, Any]], field_keys: Sequence[str]) -> List[str]:
    numeric_keys: List[str] = []
    for key in field_keys:
        has_value = False
        all_numeric = True
        for row in rows:
            value = row.get(key)
            if value is None or value == "":
                continue
            has_value = True
            if to_float(value) is None:
                all_numeric = False
                break
        if has_value and all_numeric:
            numeric_keys.append(key)
    return numeric_keys


def group_key(row: Dict[str, Any]):
    variant_name = str(row.get("variant_name") or "").strip()
    return variant_name or None


def group_rows(rows: Sequence[Dict[str, Any]], field_keys: Sequence[str]) -> List[Dict[str, Any]]:
    metric_keys = numeric_field_keys(rows, field_keys)
    grouped = group_by_key(rows, group_key)

    def base_row_builder(group_value, _items: Sequence[Dict[str, Any]]):
        return {"variant_name": str(group_value)}

    return build_group_stats_rows(
        grouped=grouped,
        metric_keys=metric_keys,
        base_row_builder=base_row_builder,
        sort_key_fn=lambda item: str(item[0]),
    )


def group_best_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    best_metric: str,
    maximize: bool,
) -> List[Dict[str, Any]]:
    grouped = group_by_key(rows, group_key)

    def base_row_builder(group_value, _items: Sequence[Dict[str, Any]], best: Dict[str, Any] | None):
        row = {"variant_name": str(group_value)}
        if best is not None:
            row.update(
                {
                    "run_dir": best.get("run_dir"),
                    "metrics_path": best.get("metrics_path"),
                    "status_ok": best.get("status_ok"),
                }
            )
        return row

    return build_group_best_rows(
        grouped=grouped,
        metric=best_metric,
        maximize=maximize,
        base_row_builder=base_row_builder,
        sort_key_fn=lambda item: str(item[0]),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize per-run sliding-window metrics JSON files."
    )
    parser.add_argument(
        "--exp_dir",
        "--exp-dir",
        dest="exp_dir",
        type=str,
        required=True,
        help=(
            "Experiment directory, runs directory parent, or a single "
            "checkpoint/run directory containing the selected metrics JSON."
        ),
    )
    parser.add_argument(
        "--out_dir",
        "--out-dir",
        dest="out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <exp_dir>/reports",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default=",".join(DEFAULT_FIELD_KEYS),
        help=(
            "Comma-separated fields to collect. Nested fields use dot paths, "
            "e.g. settings.duration."
        ),
    )
    parser.add_argument(
        "--metrics_filename",
        "--metrics-filename",
        dest="metrics_filename",
        type=str,
        default=DEFAULT_METRICS_FILENAME,
        help="Metrics JSON filename inside each run directory.",
    )
    parser.add_argument(
        "--output_prefix",
        "--output-prefix",
        dest="output_prefix",
        type=str,
        default=None,
        help=(
            "Prefix for generated report files. Default: metrics filename "
            "stem, e.g. sliding_window_eval_metrics_test_trunc."
        ),
    )
    parser.add_argument(
        "--best_metric",
        "--best-metric",
        dest="best_metric",
        type=str,
        default="rmse",
        help="Metric name used to select best run in each variant group.",
    )
    parser.add_argument(
        "--best_mode",
        "--best-mode",
        dest="best_mode",
        type=str,
        choices=["max", "min"],
        default="min",
        help="Whether best metric is maximized or minimized.",
    )
    parser.add_argument(
        "--include_missing",
        "--include-missing",
        dest="include_missing",
        action="store_true",
        help="Include runs without metrics JSON as empty rows.",
    )
    parser.add_argument(
        "--exclude_truncated_final_window",
        "--exclude-truncated-final-window",
        dest="exclude_truncated_final_window",
        action="store_true",
        help=(
            "Recompute each run's summary from its sliding-window cache after "
            "dropping windows whose target duration is shorter than "
            "settings.duration. This is useful for metrics files generated "
            "with --include-truncated-final-window."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    exp_dir = resolve_path(args.exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    out_dir = resolve_path(args.out_dir) if args.out_dir else exp_dir / "reports"
    field_keys = parse_csv(args.fields)
    if not field_keys:
        raise ValueError("--fields must contain at least one field")

    rows = collect_rows(
        exp_dir,
        field_keys=field_keys,
        metrics_filename=args.metrics_filename,
        include_missing=bool(args.include_missing),
        exclude_truncated_final_window=bool(args.exclude_truncated_final_window),
    )
    group_stats = group_rows(rows, field_keys)
    best_metric = resolve_best_metric_name(rows, args.best_metric)
    best_rows = group_best_rows(
        rows,
        best_metric=best_metric,
        maximize=(args.best_mode == "max"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = (
        str(args.output_prefix).strip()
        if args.output_prefix is not None
        else default_output_prefix(args.metrics_filename)
    )
    if args.output_prefix is None and args.exclude_truncated_final_window:
        output_prefix = f"{output_prefix}_no_truncated_final_window"
    if not output_prefix:
        raise ValueError("--output_prefix must not be empty")
    output_prefix = Path(output_prefix).name
    if output_prefix in {".", ".."}:
        raise ValueError("--output_prefix must contain a filename prefix")

    per_run_csv = out_dir / f"{output_prefix}_per_run.csv"
    group_csv = out_dir / f"{output_prefix}_group_mean_std.csv"
    best_csv = out_dir / f"{output_prefix}_group_best.csv"
    summary_json = out_dir / f"{output_prefix}_summary.json"

    write_csv(per_run_csv, rows)
    write_csv(group_csv, group_stats)
    write_csv(best_csv, best_rows)

    n_metrics_found = sum(int(row.get("metrics_found", 0)) for row in rows)
    n_success = sum(int(row.get("status_ok", 0)) for row in rows)
    payload = {
        "exp_dir": str(exp_dir),
        "out_dir": str(out_dir),
        "n_runs": len(rows),
        "n_metrics_found": n_metrics_found,
        "n_success": n_success,
        "field_keys": field_keys,
        "best_metric": args.best_metric,
        "resolved_best_metric": best_metric,
        "best_mode": args.best_mode,
        "metrics_filename": args.metrics_filename,
        "output_prefix": output_prefix,
        "exclude_truncated_final_window": bool(args.exclude_truncated_final_window),
        "files": {
            "per_run_csv": str(per_run_csv),
            "group_mean_std_csv": str(group_csv),
            "group_best_csv": str(best_csv),
        },
    }
    write_json(summary_json, payload)

    print(f"[OK] per-run rows: {len(rows)}")
    print(f"[OK] metrics found: {n_metrics_found}")
    print(f"[OK] successful metrics: {n_success}")
    print(f"[OK] group rows: {len(group_stats)}")
    print(f"[OK] best rows: {len(best_rows)}")
    print(f"[OUT] {per_run_csv}")
    print(f"[OUT] {group_csv}")
    print(f"[OUT] {best_csv}")
    print(f"[OUT] {summary_json}")


if __name__ == "__main__":
    main()
