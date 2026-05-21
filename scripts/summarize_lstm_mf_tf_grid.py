#!/usr/bin/env python3
"""Summarize metrics from run_lstm_mf_tf_grid.py experiments.

This script scans one experiment directory (e.g. experiments/lstm_seed),
collects per-run test metrics, and exports:
1) per-run table (one row per seed/run),
2) grouped mean/std table by (Tfore, Mf),
3) grouped best table by (Tfore, Mf) with a target metric.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from automation import (
    build_tf_mf_group_key,
    build_variant_group_key,
    build_group_best_rows,
    build_group_stats_rows,
    collect_per_run_summary_rows,
    group_by_key,
    resolve_best_metric_name,
    resolve_tf_mf_group_mode,
    sort_tf_mf_group_key,
    write_csv,
    write_json,
)


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


def _build_tf_mf_group_key(row: Dict):
    return build_tf_mf_group_key(row)


def _group_sort_key(item):
    return sort_tf_mf_group_key(item)


def _build_variant_group_key(row: Dict):
    return build_variant_group_key(row)


def _variant_group_sort_key(item):
    variant, _ = item
    return str(variant)


def collect_per_run_rows(
    exp_dir: Path,
    metric_keys: Sequence[str],
    ckpt_select: str = "best",
) -> List[Dict]:
    def _row_extra_builder(**kwargs):
        item = kwargs["item"]
        run_name = kwargs["run_name"]
        return {
            "variant_name": build_variant_group_key({"run": run_name}),
            "Twindow": item.get("Twindow"),
            "Tfore": item.get("Tfore"),
            "Mf": item.get("Mf"),
            "seed": item.get("seed"),
            "criterion_name": item.get("criterion_name"),
            "criterion_beta": item.get("criterion_beta"),
            "cuda_id": item.get("cuda_id"),
            "train_returncode": item.get("train_returncode"),
            "test_returncode": item.get("test_returncode"),
            "test_skipped_reason": item.get("test_skipped_reason"),
            "test_ckpt_selects": item.get("test_ckpt_selects"),
        }

    return collect_per_run_summary_rows(
        exp_dir=exp_dir,
        metric_keys=metric_keys,
        ckpt_select=ckpt_select,
        row_extra_builder=_row_extra_builder,
        sort_key_fn=lambda row: (
            int(row["Tfore"]) if row.get("Tfore") is not None else 10**9,
            float(row["Mf"]) if row.get("Mf") is not None else float("inf"),
            int(row["seed"]) if row.get("seed") is not None else 10**9,
            str(row["run"]),
        ),
    )


def group_rows(rows: Sequence[Dict], metric_keys: Sequence[str], group_mode: str) -> List[Dict]:
    if group_mode == "variant":
        grouped = group_by_key(rows, _build_variant_group_key)

        def _base_row_builder(group_key: str, _items: Sequence[Dict]):
            return {"variant_name": str(group_key)}

        return build_group_stats_rows(
            grouped=grouped,
            metric_keys=metric_keys,
            base_row_builder=_base_row_builder,
            sort_key_fn=_variant_group_sort_key,
        )

    grouped = group_by_key(rows, _build_tf_mf_group_key)

    def _base_row_builder(group_key: Tuple[int, float], _items: Sequence[Dict]):
        tf, mf = group_key
        return {
            "Tfore": tf,
            "Mf": mf,
        }

    return build_group_stats_rows(
        grouped=grouped,
        metric_keys=metric_keys,
        base_row_builder=_base_row_builder,
        sort_key_fn=_group_sort_key,
    )


def group_best_rows(rows: Sequence[Dict], metric: str, maximize: bool, group_mode: str) -> List[Dict]:
    if group_mode == "variant":
        grouped = group_by_key(rows, _build_variant_group_key)

        def _base_row_builder(
            group_key: str,
            _items: Sequence[Dict],
            best: Optional[Dict],
        ):
            row = {"variant_name": str(group_key)}
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
            sort_key_fn=_variant_group_sort_key,
        )

    grouped = group_by_key(rows, _build_tf_mf_group_key)

    def _base_row_builder(
        group_key: Tuple[int, float],
        _items: Sequence[Dict],
        best: Optional[Dict],
    ):
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
    parser = argparse.ArgumentParser(description="Summarize metrics for lstm_mf_tf_grid experiments.")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment directory produced by run_lstm_mf_tf_grid.py, e.g. experiments/lstm_seed",
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
        help="Metric name used to select best run in each (Tfore, Mf) group.",
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
    parser.add_argument(
        "--group_by",
        type=str,
        choices=["auto", "tf_mf", "variant"],
        default="auto",
        help=(
            "Grouping key for summary tables. "
            "auto: choose variant when only one (Tfore,Mf) but multiple variants, else tf_mf."
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
    group_mode = resolve_tf_mf_group_mode(per_run_rows, args.group_by, _build_variant_group_key)
    group_stats = group_rows(per_run_rows, metric_keys, group_mode=group_mode)
    best_rows = group_best_rows(
        per_run_rows,
        metric=resolve_best_metric_name(per_run_rows, args.best_metric),
        maximize=(args.best_mode == "max"),
        group_mode=group_mode,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_csv = out_dir / "lstm_grid_metrics_per_run.csv"
    group_csv = out_dir / "lstm_grid_metrics_group_mean_std.csv"
    best_csv = out_dir / "lstm_grid_metrics_group_best.csv"
    summary_json = out_dir / "lstm_grid_metrics_summary.json"

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
        "ckpt_select": args.ckpt_select,
        "group_by": group_mode,
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
