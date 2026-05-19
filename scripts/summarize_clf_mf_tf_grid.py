#!/usr/bin/env python3
"""Summarize metrics from run_clf_mf_tf_grid.py experiments.

This script scans one experiment directory (e.g. experiments/clf_grid_2),
collects per-run test metrics, and exports:
1) per-run table (one row per seed/run),
2) grouped mean/std table by (Tfore, Mf),
3) grouped best table by (Tfore, Mf) with a target metric.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from automation import (
    build_group_best_rows,
    build_group_stats_rows,
    find_metrics_file,
    group_by_key,
    load_summary_rows,
    read_json,
    resolve_run_dir,
    to_float,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRIC_KEYS = (
    "precision",
    "recall",
    "f1",
    "auc",
    "pr_auc",
    "R",
    "threshold",
    "conf",
    "fpr",
    "tpr",
)


def _build_tf_mf_group_key(row: Dict):
    tf = row.get("Tfore")
    mf = row.get("Mf")
    if tf is None or mf is None:
        return None
    return int(tf), float(mf)


def _group_sort_key(item):
    (tf, mf), _ = item
    return tf, mf


def _resolve_cfg_path(run_dir: Path, cfg_path_raw) -> Path | None:
    candidates: List[Path] = []

    if cfg_path_raw:
        raw = Path(str(cfg_path_raw)).expanduser()
        if raw.is_absolute():
            candidates.append(raw.resolve())
        else:
            candidates.append((run_dir / raw).resolve())

    candidates.extend(
        [
            run_dir / "config_input.yaml",
            run_dir / "config.yaml",
        ]
    )

    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return path
    return None


def _dedupe_summary_rows(summary_rows: Sequence[Dict]) -> List[Dict]:
    deduped: Dict[str, Dict] = {}
    extras: List[Dict] = []

    for item in summary_rows:
        run_name = item.get("run")
        if not run_name:
            extras.append(item)
            continue
        deduped[str(run_name)] = item

    return list(deduped.values()) + extras


def collect_per_run_rows(exp_dir: Path, metric_keys: Sequence[str]) -> List[Dict]:
    summary_rows = _dedupe_summary_rows(load_summary_rows(exp_dir))

    rows: List[Dict] = []
    for item in summary_rows:
        run_name = item.get("run")
        if not run_name:
            continue

        run_dir = resolve_run_dir(exp_dir, str(run_name), item.get("run_dir"), ckpt_select="auto")
        cfg_path = _resolve_cfg_path(run_dir, item.get("cfg_path"))
        metrics_path = find_metrics_file(run_dir, ckpt_select="auto")
        metrics = {}
        if metrics_path and metrics_path.exists():
            loaded = read_json(metrics_path)
            if isinstance(loaded, dict):
                metrics = loaded

        row = {
            "run": run_name,
            "run_dir": str(run_dir),
            "cfg_path": str(cfg_path) if cfg_path else "",
            "metrics_path": str(metrics_path) if metrics_path else "",
            "Twindow": item.get("Twindow"),
            "Tfore": item.get("Tfore"),
            "Mf": item.get("Mf"),
            "seed": item.get("seed"),
            "time_bias_type": item.get("time_bias_type"),
            "criterion_alpha": item.get("criterion_alpha"),
            "cuda_id": item.get("cuda_id"),
            "train_returncode": item.get("train_returncode"),
            "test_returncode": item.get("test_returncode"),
            "test_skipped_reason": item.get("test_skipped_reason"),
            "status_ok": int(
                (item.get("train_returncode") in (None, 0))
                and (item.get("test_returncode") in (None, 0))
            ),
            "metrics_found": int(bool(metrics)),
        }
        for key in metric_keys:
            row[key] = to_float(metrics.get(key))
        rows.append(row)

    rows.sort(
        key=lambda r: (
            int(r["Tfore"]) if r.get("Tfore") is not None else 10**9,
            float(r["Mf"]) if r.get("Mf") is not None else float("inf"),
            int(r["seed"]) if r.get("seed") is not None else 10**9,
            str(r["run"]),
        )
    )
    return rows


def group_rows(rows: Sequence[Dict], metric_keys: Sequence[str]) -> Tuple[List[Dict], List[Dict]]:
    grouped = group_by_key(rows, _build_tf_mf_group_key)

    def _base_row_builder(group_key: Tuple[int, float], _items: Sequence[Dict]):
        tf, mf = group_key
        return {
            "Tfore": tf,
            "Mf": mf,
        }

    group_stats = build_group_stats_rows(
        grouped=grouped,
        metric_keys=metric_keys,
        base_row_builder=_base_row_builder,
        sort_key_fn=_group_sort_key,
    )
    return group_stats, []


def group_best_rows(rows: Sequence[Dict], metric: str, maximize: bool) -> List[Dict]:
    grouped = group_by_key(rows, _build_tf_mf_group_key)

    def _base_row_builder(group_key: Tuple[int, float], _items: Sequence[Dict], best: Dict | None):
        tf, mf = group_key
        row = {
            "Tfore": tf,
            "Mf": mf,
        }
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
        metric=metric,
        maximize=maximize,
        base_row_builder=_base_row_builder,
        sort_key_fn=_group_sort_key,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize metrics for clf_mf_tf_grid experiments.")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment directory produced by run_clf_mf_tf_grid.py, e.g. experiments/clf_grid_2",
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
        default="R",
        help="Metric name used to select best run in each (Tfore, Mf) group.",
    )
    parser.add_argument(
        "--best_mode",
        type=str,
        choices=["max", "min"],
        default="max",
        help="Whether best metric is maximized or minimized.",
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

    per_run_rows = collect_per_run_rows(exp_dir, metric_keys)
    group_stats, _ = group_rows(per_run_rows, metric_keys)
    best_rows = group_best_rows(
        per_run_rows,
        metric=args.best_metric,
        maximize=(args.best_mode == "max"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_csv = out_dir / "clf_grid_metrics_per_run.csv"
    group_csv = out_dir / "clf_grid_metrics_group_mean_std.csv"
    best_csv = out_dir / "clf_grid_metrics_group_best.csv"
    summary_json = out_dir / "clf_grid_metrics_summary.json"

    write_csv(per_run_csv, per_run_rows)
    write_csv(group_csv, group_stats)
    write_csv(best_csv, best_rows)

    payload = {
        "exp_dir": str(exp_dir),
        "out_dir": str(out_dir),
        "n_runs": len(per_run_rows),
        "metric_keys": metric_keys,
        "best_metric": args.best_metric,
        "best_mode": args.best_mode,
        "files": {
            "per_run_csv": str(per_run_csv),
            "group_mean_std_csv": str(group_csv),
            "group_best_csv": str(best_csv),
        },
    }
    write_json(summary_json, payload)

    print(f"[OK] per-run rows: {len(per_run_rows)}")
    print(f"[OK] group rows: {len(group_stats)}")
    print(f"[OK] best rows: {len(best_rows)}")
    print(f"[OUT] {per_run_csv}")
    print(f"[OUT] {group_csv}")
    print(f"[OUT] {best_csv}")
    print(f"[OUT] {summary_json}")


if __name__ == "__main__":
    main()
