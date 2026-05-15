#!/usr/bin/env python3
"""Summarize metrics from run_reg_mixer_attnpl_grid.py experiments.

This script scans one experiment directory (e.g. experiments/reg_mixer_attnpl_grid_a23_loadcmp),
collects per-run test metrics, and exports:
1) per-run table (one row per seed/run),
2) grouped mean/std table by (attn_layer_idx, load_strategy),
3) grouped best table by (attn_layer_idx, load_strategy) with a target metric.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from automation import (
    build_group_best_rows,
    build_group_stats_rows,
    find_metrics_file,
    group_by_key,
    load_summary_rows,
    lookup_metric_value,
    read_json,
    resolve_best_metric_name,
    resolve_run_dir,
    to_float,
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
SEED_SUFFIX_PATTERN = re.compile(r"^(?P<base>.+)_seed_(?P<seed>\d+)$")


def _split_seed_suffix(run_name: str) -> Tuple[str, Optional[int]]:
    text = str(run_name or "")
    matched = SEED_SUFFIX_PATTERN.match(text)
    if not matched:
        return text, None
    base = matched.group("base")
    try:
        seed_value = int(matched.group("seed"))
    except (TypeError, ValueError):
        seed_value = None
    return base, seed_value


def _resolve_group_key(row: Dict):
    variant_name = str(row.get("variant_name") or "").strip()
    if variant_name:
        return "variant", variant_name
    attn_layer_idx = row.get("attn_layer_idx")
    load_strategy = row.get("load_strategy")
    if attn_layer_idx is None or load_strategy is None:
        return None
    return "matrix", f"attn_l{int(attn_layer_idx)}__{str(load_strategy)}"


def _group_sort_key(item):
    (group_type, group_name), _ = item
    return group_type, group_name


def collect_per_run_rows(exp_dir: Path, metric_keys: Sequence[str], ckpt_select: str = "best") -> List[Dict]:
    summary_rows = load_summary_rows(exp_dir)

    rows: List[Dict] = []
    for item in summary_rows:
        run_name = item.get("run")
        if not run_name:
            continue

        run_name = str(run_name)
        variant_name, seed_from_run_name = _split_seed_suffix(run_name)
        variant_source = item.get("variant_source")
        if not isinstance(variant_source, dict):
            variant_source = {}

        run_dir = resolve_run_dir(exp_dir, run_name, item.get("run_dir"), ckpt_select="auto")
        metrics_path = find_metrics_file(run_dir, ckpt_select=ckpt_select)
        metrics = {}
        if metrics_path and metrics_path.exists():
            loaded = read_json(metrics_path)
            if isinstance(loaded, dict):
                metrics = loaded

        row_seed = item.get("seed")
        if row_seed is None:
            row_seed = seed_from_run_name

        row = {
            "run": run_name,
            "variant_name": variant_name if seed_from_run_name is not None else "",
            "group_type": "variant" if seed_from_run_name is not None else "matrix",
            "source_run": variant_source.get("run"),
            "source_profile": variant_source.get("profile"),
            "source_trial": variant_source.get("trial"),
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path) if metrics_path else "",
            "attn_layer_idx": item.get("attn_layer_idx"),
            "load_strategy": item.get("load_strategy"),
            "load_strategy_value": item.get("load_strategy_value"),
            "seed": row_seed,
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
            row[key] = to_float(lookup_metric_value(metrics, key))
        rows.append(row)

    rows.sort(
        key=lambda row: (
            str(row.get("variant_name") or ""),
            int(row["attn_layer_idx"]) if row.get("attn_layer_idx") is not None else 10**9,
            str(row.get("load_strategy") or ""),
            int(row["seed"]) if row.get("seed") is not None else 10**9,
            str(row["run"]),
        )
    )
    return rows


def group_rows(rows: Sequence[Dict], metric_keys: Sequence[str]) -> List[Dict]:
    grouped = group_by_key(rows, _resolve_group_key)

    def _base_row_builder(_group_key, items: Sequence[Dict]):
        first = items[0]
        return {
            "group_type": first.get("group_type"),
            "group_name": _group_key[1],
            "variant_name": first.get("variant_name"),
            "source_run": first.get("source_run"),
            "source_profile": first.get("source_profile"),
            "source_trial": first.get("source_trial"),
            "attn_layer_idx": first.get("attn_layer_idx"),
            "load_strategy": first.get("load_strategy"),
            "load_strategy_value": first.get("load_strategy_value"),
        }

    return build_group_stats_rows(
        grouped=grouped,
        metric_keys=metric_keys,
        base_row_builder=_base_row_builder,
        sort_key_fn=_group_sort_key,
    )


def group_best_rows(rows: Sequence[Dict], metric: str, maximize: bool) -> List[Dict]:
    grouped = group_by_key(rows, _resolve_group_key)

    def _base_row_builder(_group_key, items: Sequence[Dict], best: Dict | None):
        first = items[0]
        row = {
            "group_type": first.get("group_type"),
            "group_name": _group_key[1],
            "variant_name": first.get("variant_name"),
            "source_run": first.get("source_run"),
            "source_profile": first.get("source_profile"),
            "source_trial": first.get("source_trial"),
            "attn_layer_idx": first.get("attn_layer_idx"),
            "load_strategy": first.get("load_strategy"),
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
        metric=resolve_best_metric_name(per_run_rows, args.best_metric),
        maximize=(args.best_mode == "max"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_csv = out_dir / "reg_grid_metrics_per_run.csv"
    group_csv = out_dir / "reg_grid_metrics_group_mean_std.csv"
    best_csv = out_dir / "reg_grid_metrics_group_best.csv"
    summary_json = out_dir / "reg_grid_metrics_summary.json"

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
