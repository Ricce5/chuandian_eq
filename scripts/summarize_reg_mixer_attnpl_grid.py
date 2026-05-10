#!/usr/bin/env python3
"""Summarize metrics from run_reg_mixer_attnpl_grid.py experiments.

This script scans one experiment directory (e.g. experiments/reg_mixer_attnpl_grid_a23_loadcmp),
collects per-run test metrics, and exports:
1) per-run table (one row per seed/run),
2) grouped mean/std table by (attn_layer_idx, load_strategy),
3) grouped best table by (attn_layer_idx, load_strategy) with a target metric.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRIC_KEYS = (
    "RMSE",
    "MAE",
    "MSE",
    "MAPE",
    "R2",
    "PearsonR",
    "PearsonR_diff",
    "DA",
    "SpearmanR",
    "SpearmanR_diff",
    "DTW",
    "DTW_normalized",
)


def _lookup_metric_value(metrics: Dict, metric_name: str):
    if metric_name in metrics:
        return metrics.get(metric_name)
    metric_name_lower = str(metric_name).lower()
    for key, value in metrics.items():
        if str(key).lower() == metric_name_lower:
            return value
    return None


def _resolve_best_metric_name(rows: Sequence[Dict], metric_name: str) -> str:
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


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    mean_v = sum(vals) / len(vals)
    var = sum((x - mean_v) ** 2 for x in vals) / (len(vals) - 1)
    return mean_v, math.sqrt(max(var, 0.0))


def _find_metrics_file(run_dir: Path, ckpt_select: str = "best") -> Optional[Path]:
    select_mode = str(ckpt_select).strip().lower()
    if select_mode in {"best", "last"}:
        candidates = sorted(run_dir.glob(f"metrics_test_{select_mode}_*.json"))
        if candidates:
            return candidates[0]

    candidates = sorted(run_dir.glob("metrics_test_*.json"))
    if not candidates:
        return None
    return candidates[0]


def _resolve_run_dir(exp_dir: Path, run_name: str, run_dir_raw) -> Path:
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
        if _find_metrics_file(candidate, ckpt_select="auto") is not None:
            return candidate
    for candidate in uniq_candidates:
        if candidate.exists():
            return candidate
    return local_run_dir


def _iter_summary_rows(summary_data: Iterable[Dict]) -> Iterable[Dict]:
    for row in summary_data:
        if not isinstance(row, dict):
            continue
        yield row


def collect_per_run_rows(exp_dir: Path, metric_keys: Sequence[str], ckpt_select: str = "best") -> List[Dict]:
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    summary_data = _read_json(summary_path)
    if not isinstance(summary_data, list):
        raise ValueError(f"summary.json must be a list: {summary_path}")

    rows: List[Dict] = []
    for item in _iter_summary_rows(summary_data):
        run_name = item.get("run")
        if not run_name:
            continue

        run_dir = _resolve_run_dir(exp_dir, str(run_name), item.get("run_dir"))
        metrics_path = _find_metrics_file(run_dir, ckpt_select=ckpt_select)
        metrics = {}
        if metrics_path and metrics_path.exists():
            loaded = _read_json(metrics_path)
            if isinstance(loaded, dict):
                metrics = loaded

        row = {
            "run": run_name,
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path) if metrics_path else "",
            "attn_layer_idx": item.get("attn_layer_idx"),
            "load_strategy": item.get("load_strategy"),
            "load_strategy_value": item.get("load_strategy_value"),
            "seed": item.get("seed"),
            "cuda_id": item.get("cuda_id"),
            "train_returncode": item.get("train_returncode"),
            "test_returncode": item.get("test_returncode"),
            "test_skipped_reason": item.get("test_skipped_reason"),
            "status_ok": int((item.get("train_returncode") in (None, 0)) and (item.get("test_returncode") in (None, 0))),
            "metrics_found": int(bool(metrics)),
        }
        for key in metric_keys:
            row[key] = _to_float(_lookup_metric_value(metrics, key))
        rows.append(row)

    rows.sort(
        key=lambda r: (
            int(r["attn_layer_idx"]) if r.get("attn_layer_idx") is not None else 10**9,
            str(r.get("load_strategy") or ""),
            int(r["seed"]) if r.get("seed") is not None else 10**9,
            str(r["run"]),
        )
    )
    return rows


def group_rows(rows: Sequence[Dict], metric_keys: Sequence[str]) -> List[Dict]:
    grouped: Dict[Tuple[int, str], List[Dict]] = {}
    for row in rows:
        attn_layer_idx = row.get("attn_layer_idx")
        load_strategy = row.get("load_strategy")
        if attn_layer_idx is None or load_strategy is None:
            continue
        grouped.setdefault((int(attn_layer_idx), str(load_strategy)), []).append(row)

    group_stats: List[Dict] = []
    for (attn_layer_idx, load_strategy), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        out = {
            "attn_layer_idx": attn_layer_idx,
            "load_strategy": load_strategy,
            "load_strategy_value": items[0].get("load_strategy_value"),
            "n_runs": len(items),
            "n_success": sum(int(r.get("status_ok", 0)) for r in items),
            "n_metrics_found": sum(int(r.get("metrics_found", 0)) for r in items),
        }
        for key in metric_keys:
            vals = [r.get(key) for r in items if r.get(key) is not None]
            mean_v, std_v = _mean_std(vals)
            out[f"{key}_mean"] = mean_v
            out[f"{key}_std"] = std_v
        group_stats.append(out)
    return group_stats


def group_best_rows(
    rows: Sequence[Dict],
    metric: str,
    maximize: bool,
) -> List[Dict]:
    grouped: Dict[Tuple[int, str], List[Dict]] = {}
    for row in rows:
        attn_layer_idx = row.get("attn_layer_idx")
        load_strategy = row.get("load_strategy")
        if attn_layer_idx is None or load_strategy is None:
            continue
        grouped.setdefault((int(attn_layer_idx), str(load_strategy)), []).append(row)

    best_by_group: List[Dict] = []
    for (attn_layer_idx, load_strategy), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        candidates = [r for r in items if r.get(metric) is not None]
        if not candidates:
            best_by_group.append(
                {
                    "attn_layer_idx": attn_layer_idx,
                    "load_strategy": load_strategy,
                    "best_metric_name": metric,
                    "best_metric_value": None,
                    "best_run": None,
                    "best_seed": None,
                }
            )
            continue

        if maximize:
            best = max(candidates, key=lambda r: float(r[metric]))
        else:
            best = min(candidates, key=lambda r: float(r[metric]))

        best_by_group.append(
            {
                "attn_layer_idx": attn_layer_idx,
                "load_strategy": load_strategy,
                "best_metric_name": metric,
                "best_metric_value": best.get(metric),
                "best_run": best.get("run"),
                "best_seed": best.get("seed"),
                "run_dir": best.get("run_dir"),
                "metrics_path": best.get("metrics_path"),
                "status_ok": best.get("status_ok"),
            }
        )
    return best_by_group


def _write_csv(path: Path, rows: Sequence[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize metrics for reg_mixer_attnpl_grid experiments.")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment directory produced by run_reg_mixer_attnpl_grid.py",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <exp_dir>/reports",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRIC_KEYS),
        help="Comma-separated metric keys to collect and aggregate.",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="RMSE",
        help="Metric name used to select best run in each (attn_layer_idx, load_strategy) group.",
    )
    parser.add_argument(
        "--best_mode",
        type=str,
        choices=["max", "min"],
        default="min",
        help="Whether best metric is maximized or minimized.",
    )
    parser.add_argument(
        "--ckpt_select",
        type=str,
        choices=["best", "last", "auto"],
        default="best",
        help=(
            "Which test metrics file to summarize: "
            "best -> metrics_test_best_*.json, "
            "last -> metrics_test_last_*.json, "
            "auto -> fallback to first metrics_test_*.json."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = (ROOT / exp_dir).resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else exp_dir / "reports"
    metric_keys = [x.strip() for x in str(args.metrics).split(",") if x.strip()]
    if not metric_keys:
        raise ValueError("--metrics must contain at least one key")

    per_run_rows = collect_per_run_rows(exp_dir, metric_keys, ckpt_select=args.ckpt_select)
    group_stats = group_rows(per_run_rows, metric_keys)
    best_rows = group_best_rows(
        per_run_rows,
        metric=_resolve_best_metric_name(per_run_rows, args.best_metric),
        maximize=(args.best_mode == "max"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_csv = out_dir / "reg_grid_metrics_per_run.csv"
    group_csv = out_dir / "reg_grid_metrics_group_mean_std.csv"
    best_csv = out_dir / "reg_grid_metrics_group_best.csv"
    summary_json = out_dir / "reg_grid_metrics_summary.json"

    _write_csv(per_run_csv, per_run_rows)
    _write_csv(group_csv, group_stats)
    _write_csv(best_csv, best_rows)

    payload = {
        "exp_dir": str(exp_dir),
        "out_dir": str(out_dir),
        "n_runs": len(per_run_rows),
        "metric_keys": metric_keys,
        "best_metric": args.best_metric,
        "best_mode": args.best_mode,
        "ckpt_select": args.ckpt_select,
        "files": {
            "per_run_csv": str(per_run_csv),
            "group_mean_std_csv": str(group_csv),
            "group_best_csv": str(best_csv),
        },
    }
    _write_json(summary_json, payload)

    print(f"[OK] per-run rows: {len(per_run_rows)}")
    print(f"[OK] group rows: {len(group_stats)}")
    print(f"[OK] best rows: {len(best_rows)}")
    print(f"[OUT] {per_run_csv}")
    print(f"[OUT] {group_csv}")
    print(f"[OUT] {best_csv}")
    print(f"[OUT] {summary_json}")


if __name__ == "__main__":
    main()
