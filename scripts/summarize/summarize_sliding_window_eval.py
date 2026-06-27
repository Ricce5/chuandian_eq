#!/usr/bin/env python3
"""Summarize sliding-window forecast metrics.

This script scans an experiment directory or a single checkpoint/run directory,
collects `sliding_window_eval_metrics.json`, and exports:
1) per-run table,
2) grouped mean/std table,
3) grouped best table.
"""

from __future__ import annotations

from pathlib import Path
import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


import argparse
import re
from typing import Any, Dict, List, Sequence

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
    "num_windows",
    "settings.duration",
    "settings.step",
    "settings.samples_per_batch",
    "settings.predict_b",
    "sliding_loaded_from_cache",
)
SEED_PATTERN = re.compile(r"^(?P<base>.+)_seed_(?P<seed>\d+)$")


def parse_csv(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


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


def read_json_mapping(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
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


def load_summary_rows_if_available(exp_dir: Path) -> List[Dict[str, Any]]:
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        return []
    payload = read_json(summary_path)
    if not isinstance(payload, list):
        raise ValueError(f"summary.json must be a list: {summary_path}")
    return [item for item in payload if isinstance(item, dict)]


def discover_run_dirs(exp_dir: Path, metrics_filename: str) -> List[tuple[str, Path, Dict[str, Any]]]:
    summary_rows = load_summary_rows_if_available(exp_dir)
    if summary_rows:
        discovered: List[tuple[str, Path, Dict[str, Any]]] = []
        for item in summary_rows:
            run_name = item.get("run")
            if not run_name:
                continue
            run_name = str(run_name)
            raw_run_dir = item.get("run_dir")
            if raw_run_dir:
                run_dir = Path(str(raw_run_dir)).expanduser()
                if not run_dir.is_absolute():
                    run_dir = exp_dir / run_dir
            else:
                run_dir = exp_dir / "runs" / run_name
            discovered.append((run_name, run_dir.resolve(), item))
        return discovered

    runs_dir = exp_dir / "runs"
    if runs_dir.is_dir():
        return [
            (candidate.name, candidate.resolve(), {})
            for candidate in sorted(runs_dir.iterdir(), key=lambda path: path.name)
            if candidate.is_dir()
        ]

    if (exp_dir / metrics_filename).exists():
        return [(exp_dir.name, exp_dir.resolve(), {})]

    return []


def collect_rows(
    exp_dir: Path,
    *,
    field_keys: Sequence[str],
    metrics_filename: str,
    include_missing: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_name, run_dir, summary_item in discover_run_dirs(exp_dir, metrics_filename):
        metrics_path = run_dir / metrics_filename
        metrics = read_json_mapping(metrics_path)
        if not metrics and not include_missing:
            continue

        variant_name, seed_from_name = variant_and_seed_from_run_name(run_name)
        row_seed = summary_item.get("seed", seed_from_name)

        row: Dict[str, Any] = {
            "run": run_name,
            "variant_name": variant_name,
            "seed": row_seed,
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path) if metrics_path.exists() else "",
            "status": metrics.get("status") if metrics else None,
            "status_ok": status_ok_from_metrics(metrics),
            "metrics_found": int(bool(metrics)),
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
        description="Summarize sliding_window_eval_metrics.json files."
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help=(
            "Experiment directory, runs directory parent, or a single "
            "checkpoint/run directory containing sliding_window_eval_metrics.json."
        ),
    )
    parser.add_argument(
        "--out_dir",
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
        type=str,
        default=DEFAULT_METRICS_FILENAME,
        help="Metrics JSON filename inside each run directory.",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="rmse",
        help="Metric name used to select best run in each variant group.",
    )
    parser.add_argument(
        "--best_mode",
        type=str,
        choices=["max", "min"],
        default="min",
        help="Whether best metric is maximized or minimized.",
    )
    parser.add_argument(
        "--include_missing",
        action="store_true",
        help="Include runs without metrics JSON as empty rows.",
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
    )
    group_stats = group_rows(rows, field_keys)
    best_metric = resolve_best_metric_name(rows, args.best_metric)
    best_rows = group_best_rows(
        rows,
        best_metric=best_metric,
        maximize=(args.best_mode == "max"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_csv = out_dir / "sliding_window_eval_metrics_per_run.csv"
    group_csv = out_dir / "sliding_window_eval_metrics_group_mean_std.csv"
    best_csv = out_dir / "sliding_window_eval_metrics_group_best.csv"
    summary_json = out_dir / "sliding_window_eval_metrics_summary.json"

    write_csv(per_run_csv, rows)
    write_csv(group_csv, group_stats)
    write_csv(best_csv, best_rows)

    payload = {
        "exp_dir": str(exp_dir),
        "out_dir": str(out_dir),
        "n_runs": len(rows),
        "field_keys": field_keys,
        "best_metric": args.best_metric,
        "resolved_best_metric": best_metric,
        "best_mode": args.best_mode,
        "metrics_filename": args.metrics_filename,
        "files": {
            "per_run_csv": str(per_run_csv),
            "group_mean_std_csv": str(group_csv),
            "group_best_csv": str(best_csv),
        },
    }
    write_json(summary_json, payload)

    print(f"[OK] per-run rows: {len(rows)}")
    print(f"[OK] group rows: {len(group_stats)}")
    print(f"[OK] best rows: {len(best_rows)}")
    print(f"[OUT] {per_run_csv}")
    print(f"[OUT] {group_csv}")
    print(f"[OUT] {best_csv}")
    print(f"[OUT] {summary_json}")


if __name__ == "__main__":
    main()
