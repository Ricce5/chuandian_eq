#!/usr/bin/env python3
"""Extract and compare clf-grid metrics for one/multiple seeds across experiments.

This script is designed for cross-experiment comparison such as:
  - experiments/clf_grid_r_2
  - experiments/clf_grid_r_2_AZDX_minus_b
  - experiments/clf_grid_r_2_SCEDC
  - experiments/clf_grid_r_2_SCEDC_minus_b
  - experiments/clf_grid_r_2_scratch

Outputs:
1) long table: one row per run (filtered by selected seeds),
2) wide table: one row per (Tfore, Mf, seed), columns split by experiment,
3) mean/std table by (experiment, Tfore, Mf) across selected seeds,
4) summary json with output paths and counts.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from automation import (
    find_metrics_file,
    load_summary_rows,
    lookup_metric_value,
    mean_std,
    read_json,
    resolve_run_dir,
    to_float,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIRS = (
    "experiments/clf_grid_r_2",
    "experiments/clf_grid_r_2_AZDX_minus_b",
    "experiments/clf_grid_r_2_SCEDC",
    "experiments/clf_grid_r_2_SCEDC_minus_b",
    "experiments/clf_grid_r_2_scratch",
)
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


def _parse_csv_text(raw: str) -> List[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _resolve_exp_dir(raw_path: str) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _compute_status_ok(item: Dict) -> int:
    train_ok = item.get("train_returncode") in (None, 0)
    test_returncode = item.get("test_returncode")
    if test_returncode is not None:
        test_ok = test_returncode == 0
    else:
        test_ok = True
        test_returncodes = item.get("test_returncodes")
        if isinstance(test_returncodes, dict):
            valid_codes = [value for value in test_returncodes.values() if value is not None]
            if valid_codes:
                test_ok = all(int(value) == 0 for value in valid_codes)
    return int(train_ok and test_ok)


def _to_seed_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_target_seeds(seed_value, seeds_csv: str | None) -> List[int]:
    raw_tokens: List[str] = []
    if seeds_csv:
        raw_tokens.extend(_parse_csv_text(seeds_csv))
    if seed_value is not None:
        raw_tokens.append(str(seed_value))
    if not raw_tokens:
        raise ValueError("Please provide --seed or --seeds.")

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


def collect_rows_for_seeds(
    exp_dirs: Sequence[Path],
    seeds: Sequence[int],
    metric_keys: Sequence[str],
    ckpt_select: str,
) -> List[Dict]:
    rows: List[Dict] = []
    seed_set = {int(seed) for seed in seeds}

    for exp_dir in exp_dirs:
        summary_rows = load_summary_rows(exp_dir)
        exp_name = exp_dir.name

        for item in summary_rows:
            row_seed = _to_seed_int(item.get("seed"))
            if row_seed not in seed_set:
                continue

            run_name = item.get("run")
            if not run_name:
                continue

            run_dir = resolve_run_dir(exp_dir, str(run_name), item.get("run_dir"), ckpt_select="auto")
            metrics_path = find_metrics_file(run_dir, ckpt_select=ckpt_select)
            metrics = {}
            if metrics_path and metrics_path.exists():
                loaded = read_json(metrics_path)
                if isinstance(loaded, dict):
                    metrics = loaded

            row = {
                "experiment": exp_name,
                "exp_dir": str(exp_dir),
                "run": run_name,
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path) if metrics_path else "",
                "seed": row_seed,
                "Twindow": item.get("Twindow"),
                "Tfore": item.get("Tfore"),
                "Mf": item.get("Mf"),
                "time_bias_type": item.get("time_bias_type"),
                "criterion_alpha": item.get("criterion_alpha"),
                "cuda_id": item.get("cuda_id"),
                "train_returncode": item.get("train_returncode"),
                "test_returncode": item.get("test_returncode"),
                "test_skipped_reason": item.get("test_skipped_reason"),
                "test_ckpt_selects": item.get("test_ckpt_selects"),
                "status_ok": _compute_status_ok(item),
                "metrics_found": int(bool(metrics)),
            }
            for key in metric_keys:
                row[key] = to_float(lookup_metric_value(metrics, key))
            rows.append(row)

    rows.sort(
        key=lambda row: (
            int(row["Tfore"]) if row.get("Tfore") is not None else 10**9,
            float(row["Mf"]) if row.get("Mf") is not None else float("inf"),
            str(row.get("experiment") or ""),
            str(row.get("run") or ""),
        )
    )
    return rows


def _group_key(row: Dict) -> Tuple:
    return (
        int(row["Tfore"]) if row.get("Tfore") is not None else 10**9,
        float(row["Mf"]) if row.get("Mf") is not None else float("inf"),
        int(row["seed"]) if row.get("seed") is not None else 10**9,
    )


def build_wide_rows(rows: Sequence[Dict], exp_names: Sequence[str], metric_keys: Sequence[str]) -> List[Dict]:
    grouped: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    wide_rows: List[Dict] = []
    for group_key in sorted(grouped.keys()):
        items = grouped[group_key]
        first = items[0]
        row = {
            "seed": first.get("seed"),
            "Twindow": first.get("Twindow"),
            "Tfore": first.get("Tfore"),
            "Mf": first.get("Mf"),
        }

        by_exp: Dict[str, Dict] = {}
        for item in items:
            exp = str(item.get("experiment") or "")
            previous = by_exp.get(exp)
            if previous is None:
                by_exp[exp] = item
                continue
            prev_ok = int(previous.get("metrics_found", 0)), int(previous.get("status_ok", 0))
            now_ok = int(item.get("metrics_found", 0)), int(item.get("status_ok", 0))
            if now_ok > prev_ok:
                by_exp[exp] = item

        present_experiments: List[str] = []
        for exp_name in exp_names:
            item = by_exp.get(exp_name)
            row[f"{exp_name}__run"] = item.get("run") if item else None
            row[f"{exp_name}__status_ok"] = item.get("status_ok") if item else 0
            row[f"{exp_name}__metrics_found"] = item.get("metrics_found") if item else 0
            for metric_key in metric_keys:
                row[f"{exp_name}__{metric_key}"] = item.get(metric_key) if item else None
            if item is not None:
                present_experiments.append(exp_name)

        row["n_experiments_present"] = len(present_experiments)
        row["present_experiments"] = ",".join(present_experiments)
        wide_rows.append(row)

    return wide_rows


def _mean_std_group_key(row: Dict) -> Tuple:
    return (
        str(row.get("experiment") or ""),
        int(row["Tfore"]) if row.get("Tfore") is not None else 10**9,
        float(row["Mf"]) if row.get("Mf") is not None else float("inf"),
    )


def build_mean_std_rows(rows: Sequence[Dict], metric_keys: Sequence[str]) -> List[Dict]:
    grouped: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(_mean_std_group_key(row), []).append(row)

    mean_rows: List[Dict] = []
    for key in sorted(grouped.keys(), key=lambda item: (item[1], item[2], item[0])):
        items = grouped[key]
        first = items[0]
        seeds_present = sorted({int(item["seed"]) for item in items if item.get("seed") is not None})
        out = {
            "experiment": first.get("experiment"),
            "Twindow": first.get("Twindow"),
            "Tfore": first.get("Tfore"),
            "Mf": first.get("Mf"),
            "n_runs": len(items),
            "n_success": sum(int(item.get("status_ok", 0)) for item in items),
            "n_metrics_found": sum(int(item.get("metrics_found", 0)) for item in items),
            "n_seeds_present": len(seeds_present),
            "seeds_present": ",".join(str(seed) for seed in seeds_present),
        }
        for metric_key in metric_keys:
            values = [item.get(metric_key) for item in items if item.get(metric_key) is not None]
            mean_value, std_value = mean_std(values)
            out[f"{metric_key}_mean"] = mean_value
            out[f"{metric_key}_std"] = std_value
        mean_rows.append(out)
    return mean_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and compare clf-grid metrics for one/multiple seeds across experiment dirs."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed to extract, e.g. 0. Backward-compatible with old usage.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds, e.g. 0,1,2. Can be used with/without --seed.",
    )
    parser.add_argument(
        "--exp_dirs",
        type=str,
        default=",".join(DEFAULT_EXPERIMENT_DIRS),
        help="Comma-separated experiment directories.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRIC_KEYS),
        help="Comma-separated metric keys.",
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
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: experiments/reports/clf_seed_compare_seed_<seed> or ..._seeds_<s0_s1_...>",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    exp_dir_values = _parse_csv_text(args.exp_dirs)
    if not exp_dir_values:
        raise ValueError("--exp_dirs is empty")

    exp_dirs = [_resolve_exp_dir(path) for path in exp_dir_values]
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")
        if not (exp_dir / "summary.json").exists():
            raise FileNotFoundError(f"summary.json not found: {exp_dir / 'summary.json'}")

    metric_keys = _parse_csv_text(args.metrics)
    if not metric_keys:
        raise ValueError("--metrics must contain at least one key")

    target_seeds = _resolve_target_seeds(args.seed, args.seeds)
    if args.out_dir:
        out_dir = _resolve_exp_dir(args.out_dir)
    else:
        if len(target_seeds) == 1:
            out_dir = ROOT / "experiments" / "reports" / f"clf_seed_compare_seed_{int(target_seeds[0])}"
        else:
            seed_tag = "_".join(str(seed) for seed in target_seeds)
            out_dir = ROOT / "experiments" / "reports" / f"clf_seed_compare_seeds_{seed_tag}"

    per_run_rows = collect_rows_for_seeds(
        exp_dirs=exp_dirs,
        seeds=target_seeds,
        metric_keys=metric_keys,
        ckpt_select=args.ckpt_select,
    )
    exp_names = [path.name for path in exp_dirs]
    wide_rows = build_wide_rows(per_run_rows, exp_names=exp_names, metric_keys=metric_keys)
    mean_rows = build_mean_std_rows(per_run_rows, metric_keys=metric_keys)

    out_dir.mkdir(parents=True, exist_ok=True)
    long_csv = out_dir / "clf_seed_metrics_long.csv"
    wide_csv = out_dir / "clf_seed_metrics_wide.csv"
    mean_std_csv = out_dir / "clf_seed_metrics_mean_std.csv"
    summary_json = out_dir / "clf_seed_metrics_summary.json"

    write_csv(long_csv, per_run_rows)
    write_csv(wide_csv, wide_rows)
    write_csv(mean_std_csv, mean_rows)

    runs_by_exp = {name: 0 for name in exp_names}
    for row in per_run_rows:
        runs_by_exp[str(row.get("experiment"))] += 1

    payload = {
        "seed": int(target_seeds[0]) if len(target_seeds) == 1 else None,
        "seeds": target_seeds,
        "exp_dirs": [str(path) for path in exp_dirs],
        "metric_keys": metric_keys,
        "ckpt_select": args.ckpt_select,
        "n_long_rows": len(per_run_rows),
        "n_wide_rows": len(wide_rows),
        "n_mean_std_rows": len(mean_rows),
        "runs_by_experiment": runs_by_exp,
        "files": {
            "long_csv": str(long_csv),
            "wide_csv": str(wide_csv),
            "mean_std_csv": str(mean_std_csv),
            "summary_json": str(summary_json),
        },
    }
    write_json(summary_json, payload)

    print(f"[OK] seeds: {target_seeds}")
    print(f"[OK] experiments: {len(exp_dirs)}")
    print(f"[OK] long rows: {len(per_run_rows)}")
    print(f"[OK] wide rows: {len(wide_rows)}")
    print(f"[OK] mean/std rows: {len(mean_rows)}")
    print(f"[OUT] {long_csv}")
    print(f"[OUT] {wide_csv}")
    print(f"[OUT] {mean_std_csv}")
    print(f"[OUT] {summary_json}")


if __name__ == "__main__":
    main()
