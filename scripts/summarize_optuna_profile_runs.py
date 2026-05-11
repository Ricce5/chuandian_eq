#!/usr/bin/env python3
"""Summarize and compare multiple Optuna profile run directories.

Expected run directory layout:
  <run_dir>/
    profile_optuna_summary.json
    runs/<profile>/optuna_summary_<profile>.json

This script aggregates per-profile best results and trial test metrics
(`best` / `last` checkpoint metrics), then exports CSV/JSON comparison files.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence


TEST_METRIC_KEYS = (
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

LOWER_IS_BETTER_FIELDS = {
    "best_value",
    "test_best_RMSE",
    "test_last_RMSE",
    "test_best_MAE",
    "test_last_MAE",
    "test_best_MSE",
    "test_last_MSE",
    "test_best_MAPE",
    "test_last_MAPE",
    "test_best_DTW",
    "test_last_DTW",
    "test_best_DTW_normalized",
    "test_last_DTW_normalized",
}

PARAM_KEYS = (
    "scheduler_type",
    "learning_rate",
    "encoder_learning_rate",
    "weight_decay",
    "warmup_ratio",
    "step_lr_step_size_ratio",
    "step_lr_gamma",
    "batch_size",
    "criterion_cfg.beta",
    "mlp_dropout",
)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _resolve_summary_path(run_dir: Path, row: Dict) -> Path:
    raw = row.get("summary_path")
    if raw:
        p = Path(str(raw))
        if p.is_absolute():
            return p
        return (run_dir / p).resolve()

    profile = row.get("profile")
    if profile:
        return (run_dir / "runs" / str(profile) / f"optuna_summary_{profile}.json").resolve()
    return run_dir / "missing_summary.json"


def _load_best_params(item: Dict, best_trial: Dict) -> Dict:
    params = best_trial.get("params")
    if isinstance(params, dict) and params:
        return params

    params_json = item.get("best_params_json")
    if isinstance(params_json, str) and params_json.strip():
        try:
            parsed = json.loads(params_json)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _extract_profile_row(run_dir: Path, run_name: str, item: Dict) -> Dict:
    profile = str(item.get("profile") or "")
    status = item.get("status")
    return_code = item.get("return_code")

    profile_summary_path = _resolve_summary_path(run_dir, item)
    profile_summary = {}
    if profile_summary_path.exists():
        loaded = _read_json(profile_summary_path)
        if isinstance(loaded, dict):
            profile_summary = loaded

    best_trial = profile_summary.get("best_trial", {}) if isinstance(profile_summary, dict) else {}
    user_attrs = best_trial.get("user_attrs", {}) if isinstance(best_trial, dict) else {}
    if not isinstance(user_attrs, dict):
        user_attrs = {}
    trial_test_metrics = user_attrs.get("trial_test_metrics", {})
    if not isinstance(trial_test_metrics, dict):
        trial_test_metrics = {}

    best_metrics = trial_test_metrics.get("best", {})
    last_metrics = trial_test_metrics.get("last", {})
    if not isinstance(best_metrics, dict):
        best_metrics = {}
    if not isinstance(last_metrics, dict):
        last_metrics = {}

    params = _load_best_params(item, best_trial)
    n_trials_total = item.get("n_trials_total")
    n_trials_completed = item.get("n_trials_completed")
    completion_rate = None
    if n_trials_total not in (None, 0) and n_trials_completed is not None:
        completion_rate = _to_float(n_trials_completed) / _to_float(n_trials_total)

    row = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "profile": profile,
        "status": status,
        "return_code": return_code,
        "study_name": item.get("study_name"),
        "summary_path": str(profile_summary_path),
        "summary_exists": int(profile_summary_path.exists()),
        "n_trials_total": n_trials_total,
        "n_trials_completed": n_trials_completed,
        "completion_rate": completion_rate,
        "best_trial_number": item.get("best_trial_number"),
        "best_value": _to_float(item.get("best_value")),
        "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
    }

    for key in PARAM_KEYS:
        row[f"param_{key}"] = params.get(key)

    for metric in TEST_METRIC_KEYS:
        row[f"test_best_{metric}"] = _to_float(best_metrics.get(metric))
        row[f"test_last_{metric}"] = _to_float(last_metrics.get(metric))
    return row


def collect_rows(run_dirs: Sequence[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "profile_optuna_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"profile_optuna_summary.json not found: {summary_path}")
        summary = _read_json(summary_path)
        if not isinstance(summary, dict):
            raise ValueError(f"Invalid summary format: {summary_path}")

        run_name = run_dir.name
        results = summary.get("results", [])
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            rows.append(_extract_profile_row(run_dir=run_dir, run_name=run_name, item=item))

    rows.sort(key=lambda r: (str(r.get("profile")), str(r.get("run_name"))))
    return rows


def compare_rows(rows: Sequence[Dict], run_order: Sequence[str], baseline_run: str) -> List[Dict]:
    by_profile: Dict[str, Dict[str, Dict]] = {}
    for row in rows:
        profile = str(row.get("profile"))
        run_name = str(row.get("run_name"))
        by_profile.setdefault(profile, {})[run_name] = row

    compared_rows: List[Dict] = []
    metric_fields = ["best_value"] + [f"test_best_{m}" for m in TEST_METRIC_KEYS] + [f"test_last_{m}" for m in TEST_METRIC_KEYS]

    for profile in sorted(by_profile.keys()):
        run_map = by_profile[profile]
        if baseline_run not in run_map:
            continue
        base = run_map[baseline_run]
        for run_name in run_order:
            if run_name == baseline_run or run_name not in run_map:
                continue
            cur = run_map[run_name]
            out = {
                "profile": profile,
                "baseline_run": baseline_run,
                "target_run": run_name,
            }
            for field in metric_fields:
                base_v = _to_float(base.get(field))
                cur_v = _to_float(cur.get(field))
                out[f"{field}_baseline"] = base_v
                out[f"{field}_target"] = cur_v
                delta = None
                better = None
                if base_v is not None and cur_v is not None:
                    delta = cur_v - base_v
                    if field in LOWER_IS_BETTER_FIELDS:
                        better = int(delta < 0)
                    else:
                        better = int(delta > 0)
                out[f"{field}_delta_target_minus_baseline"] = delta
                out[f"{field}_target_better"] = better
            compared_rows.append(out)

    compared_rows.sort(key=lambda r: (str(r["profile"]), str(r["target_run"])))
    return compared_rows


def build_run_overview(rows: Sequence[Dict], run_order: Sequence[str]) -> List[Dict]:
    run_map: Dict[str, List[Dict]] = {}
    for row in rows:
        run_map.setdefault(str(row["run_name"]), []).append(row)

    out: List[Dict] = []
    for run_name in run_order:
        items = run_map.get(run_name, [])
        best_items = [r for r in items if r.get("best_value") is not None]
        best_profile = None
        best_value = None
        if best_items:
            picked = min(best_items, key=lambda r: float(r["best_value"]))
            best_profile = picked.get("profile")
            best_value = picked.get("best_value")
        out.append(
            {
                "run_name": run_name,
                "n_profiles": len(items),
                "n_profile_ok": sum(1 for r in items if str(r.get("status")) == "ok"),
                "best_profile": best_profile,
                "best_value": best_value,
            }
        )
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize and compare multiple Optuna profile run directories.")
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="Run directories to compare, each containing profile_optuna_summary.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for summary files (default: <first_run_dir>/comparison_<baseline>_vs_others).",
    )
    parser.add_argument(
        "--baseline_run",
        type=str,
        default=None,
        help="Baseline run folder name; default uses the first run in --run_dirs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dirs = [Path(x).expanduser().resolve() for x in args.run_dirs]
    run_order = [p.name for p in run_dirs]
    baseline_run = args.baseline_run or run_order[0]
    if baseline_run not in run_order:
        raise ValueError(f"baseline_run '{baseline_run}' not found in run_dirs: {run_order}")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = (run_dirs[0] / f"comparison_{baseline_run}_vs_others").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(run_dirs)
    run_overview = build_run_overview(rows, run_order)
    comparisons = compare_rows(rows, run_order=run_order, baseline_run=baseline_run)

    _write_csv(output_dir / "per_profile_summary.csv", rows)
    _write_json(output_dir / "per_profile_summary.json", rows)
    _write_csv(output_dir / "run_overview.csv", run_overview)
    _write_json(output_dir / "run_overview.json", run_overview)
    _write_csv(output_dir / "pairwise_comparison.csv", comparisons)
    _write_json(output_dir / "pairwise_comparison.json", comparisons)

    print(f"[OK] Wrote summary files to: {output_dir}")
    print(f"[OK] Per-profile rows: {len(rows)}")
    print(f"[OK] Pairwise comparisons: {len(comparisons)}")
    print("[Run overview]")
    for row in run_overview:
        print(
            f"  - {row['run_name']}: "
            f"n_profiles={row['n_profiles']}, n_ok={row['n_profile_ok']}, "
            f"best_profile={row['best_profile']}, best_value={row['best_value']}"
        )


if __name__ == "__main__":
    main()
