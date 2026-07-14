#!/usr/bin/env python3
"""Summarize metrics from run_reg_mixer_attnpl_grid.py experiments.

This script scans one experiment directory (e.g. experiments/reg_mixer_attnpl_grid_a23_loadcmp),
collects per-run test metrics, and exports:
1) per-run table (one row per seed/run),
2) grouped mean/std table by (attn_layer_idx, load_strategy),
3) grouped best table by (attn_layer_idx, load_strategy) with a target metric.
"""

from pathlib import Path
import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))



import argparse
from typing import Dict, List, Sequence

from automation import (
    build_variant_or_matrix_group_key,
    build_group_best_rows,
    build_group_stats_rows,
    group_by_key,
    resolve_best_metric_name,
    safe_mapping,
    split_seed_suffix,
    sort_group_type_name_key,
    collect_per_run_summary_rows,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
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


def _parse_csv_text(raw: str) -> List[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _to_seed_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_target_seeds(seed_value, seeds_csv: str | None) -> List[int] | None:
    raw_tokens: List[str] = []
    if seeds_csv:
        raw_tokens.extend(_parse_csv_text(seeds_csv))
    if seed_value is not None:
        raw_tokens.append(str(seed_value))
    if not raw_tokens:
        return None

    out: List[int] = []
    seen = set()
    for token in raw_tokens:
        seed_int = _to_seed_int(token)
        if seed_int is None:
            raise ValueError(f"Invalid seed value: {token!r}")
        if seed_int in seen:
            continue
        seen.add(seed_int)
        out.append(seed_int)
    return sorted(out)


def _filter_rows_by_seeds(rows: Sequence[Dict], seeds: Sequence[int] | None) -> List[Dict]:
    if seeds is None:
        return list(rows)
    seed_set = {int(seed) for seed in seeds}
    return [
        row
        for row in rows
        if _to_seed_int(row.get("seed")) in seed_set
    ]


def _seed_report_dir_name(seeds: Sequence[int] | None) -> str | None:
    if seeds is None:
        return None
    if len(seeds) == 1:
        return f"seed_{int(seeds[0])}"
    return "seeds_" + "_".join(str(int(seed)) for seed in seeds)


def _resolve_group_key(row: Dict):
    return build_variant_or_matrix_group_key(row)


def _group_sort_key(item):
    return sort_group_type_name_key(item)


def collect_per_run_rows(exp_dir: Path, metric_keys: Sequence[str], ckpt_select: str = "best") -> List[Dict]:
    def _row_extra_builder(**kwargs):
        item = kwargs["item"]
        run_name = kwargs["run_name"]
        variant_name, seed_from_run_name = split_seed_suffix(run_name)
        variant_source = safe_mapping(item.get("variant_source"))
        row_seed = item.get("seed")
        if row_seed is None:
            row_seed = seed_from_run_name
        return {
            "variant_name": variant_name if seed_from_run_name is not None else "",
            "group_type": "variant" if seed_from_run_name is not None else "matrix",
            "source_run": variant_source.get("run"),
            "source_profile": variant_source.get("profile"),
            "source_trial": variant_source.get("trial"),
            "attn_layer_idx": item.get("attn_layer_idx"),
            "load_strategy": item.get("load_strategy"),
            "load_strategy_value": item.get("load_strategy_value"),
            "seed": row_seed,
            "cuda_id": item.get("cuda_id"),
            "train_returncode": item.get("train_returncode"),
            "test_returncode": item.get("test_returncode"),
            "test_skipped_reason": item.get("test_skipped_reason"),
        }

    return collect_per_run_summary_rows(
        exp_dir=exp_dir,
        metric_keys=metric_keys,
        ckpt_select=ckpt_select,
        row_extra_builder=_row_extra_builder,
        sort_key_fn=lambda row: (
            str(row.get("variant_name") or ""),
            int(row["attn_layer_idx"]) if row.get("attn_layer_idx") is not None else 10**9,
            str(row.get("load_strategy") or ""),
            int(row["seed"]) if row.get("seed") is not None else 10**9,
            str(row["run"]),
        ),
    )


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
        "--seed",
        type=int,
        default=None,
        help="Single seed to summarize, e.g. 0. If omitted with --seeds, all seeds are included.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to summarize, e.g. 0,1,2. Can be combined with --seed.",
    )
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
        help=(
            "Output directory. Default: <exp_dir>/reports, or "
            "<exp_dir>/reports/seed_<n> / seeds_<...> when --seed/--seeds is provided."
        ),
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
    target_seeds = _resolve_target_seeds(args.seed, args.seeds)

    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = (ROOT / exp_dir).resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        seed_dir_name = _seed_report_dir_name(target_seeds)
        out_dir = exp_dir / "reports" / seed_dir_name if seed_dir_name else exp_dir / "reports"

    metric_keys = [x.strip() for x in str(args.metrics).split(",") if x.strip()]
    if not metric_keys:
        raise ValueError("--metrics must contain at least one key")

    per_run_rows = collect_per_run_rows(exp_dir, metric_keys, ckpt_select=args.ckpt_select)
    per_run_rows = _filter_rows_by_seeds(per_run_rows, target_seeds)
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
        "seed": int(target_seeds[0]) if target_seeds is not None and len(target_seeds) == 1 else None,
        "seeds": target_seeds,
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

    if target_seeds is not None:
        print(f"[OK] seeds: {target_seeds}")
    print(f"[OK] per-run rows: {len(per_run_rows)}")
    print(f"[OK] group rows: {len(group_stats)}")
    print(f"[OK] best rows: {len(best_rows)}")
    print(f"[OUT] {per_run_csv}")
    print(f"[OUT] {group_csv}")
    print(f"[OUT] {best_csv}")
    print(f"[OUT] {summary_json}")


if __name__ == "__main__":
    main()
