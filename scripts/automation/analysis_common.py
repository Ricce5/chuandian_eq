#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as file:
            file.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
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


def mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    valid_values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not valid_values:
        return None, None
    if len(valid_values) == 1:
        return valid_values[0], 0.0
    mean_value = sum(valid_values) / len(valid_values)
    var = sum((x - mean_value) ** 2 for x in valid_values) / (len(valid_values) - 1)
    return mean_value, math.sqrt(max(var, 0.0))


def lookup_metric_value(metrics: Dict, metric_name: str):
    if metric_name in metrics:
        return metrics.get(metric_name)
    metric_name_lower = str(metric_name).lower()
    for key, value in metrics.items():
        if str(key).lower() == metric_name_lower:
            return value
    return None


def resolve_best_metric_name(rows: Sequence[Dict], metric_name: str) -> str:
    if not rows:
        return metric_name
    row_keys = list(rows[0].keys())
    if metric_name in row_keys:
        return metric_name
    metric_name_lower = str(metric_name).lower()
    for key in row_keys:
        if str(key).lower() == metric_name_lower:
            return str(key)
    return metric_name


def find_metrics_file(run_dir: Path, ckpt_select: str = "auto") -> Optional[Path]:
    select_mode = str(ckpt_select).strip().lower()
    if select_mode in {"best", "last"}:
        candidates = sorted(run_dir.glob(f"metrics_test_{select_mode}_*.json"))
        if candidates:
            return candidates[0]

    candidates = sorted(run_dir.glob("metrics_test_*.json"))
    if not candidates:
        return None
    return candidates[0]


def resolve_run_dir(exp_dir: Path, run_name: str, run_dir_raw, ckpt_select: str = "auto") -> Path:
    local_run_dir = (exp_dir / "runs" / str(run_name)).resolve()
    candidates: List[Path] = [local_run_dir]

    if run_dir_raw:
        raw = Path(str(run_dir_raw)).expanduser()
        if raw.is_absolute():
            candidates.append(raw.resolve())
        else:
            candidates.append((exp_dir / raw).resolve())

    uniq_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        uniq_candidates.append(candidate)

    for candidate in uniq_candidates:
        if find_metrics_file(candidate, ckpt_select=ckpt_select) is not None:
            return candidate
    for candidate in uniq_candidates:
        if candidate.exists():
            return candidate
    return local_run_dir


def iter_summary_rows(summary_data: Iterable[Dict]) -> Iterable[Dict]:
    for row in summary_data:
        if isinstance(row, dict):
            yield row


def load_summary_rows(exp_dir: Path) -> List[Dict]:
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    summary_data = read_json(summary_path)
    if not isinstance(summary_data, list):
        raise ValueError(f"summary.json must be a list: {summary_path}")
    return list(iter_summary_rows(summary_data))


def group_by_key(rows: Sequence[Dict], key_fn):
    grouped = {}
    for row in rows:
        key = key_fn(row)
        if key is None:
            continue
        grouped.setdefault(key, []).append(row)
    return grouped


def build_group_stats_rows(
    grouped: Dict,
    metric_keys: Sequence[str],
    base_row_builder,
    sort_key_fn=None,
) -> List[Dict]:
    if sort_key_fn is None:
        sorted_items = sorted(grouped.items())
    else:
        sorted_items = sorted(grouped.items(), key=sort_key_fn)

    group_stats: List[Dict] = []
    for group_key, items in sorted_items:
        out = dict(base_row_builder(group_key, items))
        out["n_runs"] = len(items)
        out["n_success"] = sum(int(r.get("status_ok", 0)) for r in items)
        out["n_metrics_found"] = sum(int(r.get("metrics_found", 0)) for r in items)
        for metric_key in metric_keys:
            values = [r.get(metric_key) for r in items if r.get(metric_key) is not None]
            mean_value, std_value = mean_std(values)
            out[f"{metric_key}_mean"] = mean_value
            out[f"{metric_key}_std"] = std_value
        group_stats.append(out)
    return group_stats


def build_group_best_rows(
    grouped: Dict,
    metric: str,
    maximize: bool,
    base_row_builder,
    sort_key_fn=None,
) -> List[Dict]:
    if sort_key_fn is None:
        sorted_items = sorted(grouped.items())
    else:
        sorted_items = sorted(grouped.items(), key=sort_key_fn)

    best_rows: List[Dict] = []
    for group_key, items in sorted_items:
        candidates = [r for r in items if r.get(metric) is not None]
        if not candidates:
            row = dict(base_row_builder(group_key, items, None))
            row["best_metric_name"] = metric
            row["best_metric_value"] = None
            row["best_run"] = None
            row["best_seed"] = None
            best_rows.append(row)
            continue

        if maximize:
            best = max(candidates, key=lambda r: float(r[metric]))
        else:
            best = min(candidates, key=lambda r: float(r[metric]))

        row = dict(base_row_builder(group_key, items, best))
        row["best_metric_name"] = metric
        row["best_metric_value"] = best.get(metric)
        row["best_run"] = best.get("run")
        row["best_seed"] = best.get("seed")
        best_rows.append(row)
    return best_rows
