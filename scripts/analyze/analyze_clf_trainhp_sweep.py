#!/usr/bin/env python3
"""Aggregate analysis for clf_trainhp sweep experiments.

This script scans all sub-experiments under a root folder (default:
`experiments/clf_trainhp`), merges run-level metrics, and exports:

1) Per-run table across all hp_* sub-experiments.
2) Grouped mean/std by (experiment, hp, Tfore, Mf).
3) Grouped overall mean/std by experiment/hyperparameter.
4) Best hyperparameter setting per (Tfore, Mf) by target metric.
5) Best hyperparameter setting overall by target metric.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
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
HP_FIELDS = (
    "learning_rate",
    "weight_decay",
    "scheduler_type",
    "warmup_ratio",
    "batch_size",
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


def _to_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _to_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    mean_v = sum(vals) / len(vals)
    var = sum((x - mean_v) ** 2 for x in vals) / (len(vals) - 1)
    return mean_v, math.sqrt(max(var, 0.0))


def _parse_numeric_token(token: str):
    raw = str(token).strip()
    if not raw:
        return None
    replaced = raw.replace("p", ".")
    if re.fullmatch(r"-?\d+(\.\d+)?", replaced):
        if "." in replaced:
            return float(replaced)
        return int(replaced)
    return None


def _find_metrics_file(run_dir: Path, prefer: str = "best") -> Optional[Path]:
    candidates = sorted(run_dir.glob("metrics_test_*.json"))
    if not candidates:
        return None
    if prefer in {"best", "last"}:
        preferred = [p for p in candidates if f"_{prefer}_" in p.name]
        if preferred:
            return preferred[0]
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

    seen = set()
    uniq_candidates = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        uniq_candidates.append(candidate)

    for candidate in uniq_candidates:
        if _find_metrics_file(candidate) is not None:
            return candidate
    for candidate in uniq_candidates:
        if candidate.exists():
            return candidate
    return local_run_dir


def _iter_summary_rows(summary_data: Iterable[Dict]) -> Iterable[Dict]:
    for row in summary_data:
        if isinstance(row, dict):
            yield row


def _parse_hp_from_name(exp_name: str) -> Dict:
    out = {
        "learning_rate": None,
        "weight_decay": None,
        "scheduler_type": None,
        "warmup_ratio": None,
        "batch_size": None,
    }
    pattern = re.compile(
        r"_lr(?P<lr>[^_]+)_wd(?P<wd>[^_]+)_(?P<scheduler>step_warmup|warmup_linear_decay)_wr(?P<wr>[^_]+)_bs(?P<bs>[^_]+)$"
    )
    m = pattern.search(exp_name)
    if not m:
        return out

    out["learning_rate"] = _to_float(_parse_numeric_token(m.group("lr")))
    out["weight_decay"] = _to_float(_parse_numeric_token(m.group("wd")))
    out["scheduler_type"] = m.group("scheduler")
    out["warmup_ratio"] = _to_float(_parse_numeric_token(m.group("wr")))
    out["batch_size"] = _to_int(_parse_numeric_token(m.group("bs")))
    return out


def _load_hp_info(exp_dir: Path) -> Dict:
    hp = {
        "learning_rate": None,
        "weight_decay": None,
        "scheduler_type": None,
        "warmup_ratio": None,
        "batch_size": None,
    }

    plan_path = exp_dir / "plan.json"
    if plan_path.exists():
        try:
            plan = _read_json(plan_path)
        except Exception:
            plan = None
        if isinstance(plan, dict):
            extra = plan.get("extra_overrides")
            if isinstance(extra, dict):
                hp["learning_rate"] = _to_float(extra.get("learning_rate"))
                hp["weight_decay"] = _to_float(extra.get("weight_decay"))
                hp["scheduler_type"] = extra.get("scheduler_type")
                hp["warmup_ratio"] = _to_float(extra.get("warmup_ratio"))
                hp["batch_size"] = _to_int(extra.get("batch_size"))

    fallback = _parse_hp_from_name(exp_dir.name)
    for key in HP_FIELDS:
        if hp.get(key) is None:
            hp[key] = fallback.get(key)
    return hp


def _compute_status_ok(item: Dict) -> int:
    train_ok = item.get("train_returncode") in (None, 0)

    test_returncode = item.get("test_returncode")
    if test_returncode is not None:
        test_ok = test_returncode == 0
    else:
        test_ok = True
        test_returncodes = item.get("test_returncodes")
        if isinstance(test_returncodes, dict):
            vals = [v for v in test_returncodes.values() if v is not None]
            if vals:
                test_ok = all(int(v) == 0 for v in vals)

    return int(train_ok and test_ok)


def collect_all_runs(exp_root: Path, metric_keys: Sequence[str], metrics_prefer: str) -> Tuple[List[Dict], List[Dict]]:
    exp_dirs = sorted([p for p in exp_root.iterdir() if p.is_dir()])
    rows: List[Dict] = []
    exp_index_rows: List[Dict] = []

    for exp_dir in exp_dirs:
        summary_path = exp_dir / "summary.json"
        if not summary_path.exists():
            continue

        hp_info = _load_hp_info(exp_dir)

        try:
            summary_data = _read_json(summary_path)
        except Exception as e:
            exp_index_rows.append(
                {
                    "exp_name": exp_dir.name,
                    "exp_dir": str(exp_dir),
                    **hp_info,
                    "summary_ok": 0,
                    "summary_rows": 0,
                    "error": f"failed to read summary.json: {e}",
                }
            )
            continue

        if not isinstance(summary_data, list):
            exp_index_rows.append(
                {
                    "exp_name": exp_dir.name,
                    "exp_dir": str(exp_dir),
                    **hp_info,
                    "summary_ok": 0,
                    "summary_rows": 0,
                    "error": "summary.json is not a list",
                }
            )
            continue

        local_count = 0
        local_metrics_found = 0
        local_success = 0
        for item in _iter_summary_rows(summary_data):
            run_name = item.get("run")
            if not run_name:
                continue
            local_count += 1

            run_dir = _resolve_run_dir(exp_dir, str(run_name), item.get("run_dir"))
            metrics_path = _find_metrics_file(run_dir, prefer=metrics_prefer)
            metrics = {}
            if metrics_path and metrics_path.exists():
                loaded = _read_json(metrics_path)
                if isinstance(loaded, dict):
                    metrics = loaded

            status_ok = _compute_status_ok(item)
            local_success += status_ok
            local_metrics_found += int(bool(metrics))

            row = {
                "exp_name": exp_dir.name,
                "exp_dir": str(exp_dir),
                **hp_info,
                "run": run_name,
                "run_dir": str(run_dir),
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
                "status_ok": status_ok,
                "metrics_found": int(bool(metrics)),
            }
            for key in metric_keys:
                row[key] = _to_float(metrics.get(key))
            rows.append(row)

        exp_index_rows.append(
            {
                "exp_name": exp_dir.name,
                "exp_dir": str(exp_dir),
                **hp_info,
                "summary_ok": 1,
                "summary_rows": local_count,
                "success_rows": local_success,
                "metrics_found_rows": local_metrics_found,
                "error": "",
            }
        )

    rows.sort(
        key=lambda r: (
            str(r.get("exp_name")),
            int(r["Tfore"]) if r.get("Tfore") is not None else 10**9,
            float(r["Mf"]) if r.get("Mf") is not None else float("inf"),
            int(r["seed"]) if r.get("seed") is not None else 10**9,
            str(r.get("run")),
        )
    )

    exp_index_rows.sort(key=lambda r: str(r.get("exp_name")))
    return rows, exp_index_rows


def _group_rows(rows: Sequence[Dict], group_fields: Sequence[str], metric_keys: Sequence[str]) -> List[Dict]:
    grouped: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        key = tuple(row.get(f) for f in group_fields)
        grouped.setdefault(key, []).append(row)

    out_rows: List[Dict] = []
    for key, items in grouped.items():
        out = {field: value for field, value in zip(group_fields, key)}
        out["n_runs"] = len(items)
        out["n_success"] = sum(int(r.get("status_ok", 0)) for r in items)
        out["n_metrics_found"] = sum(int(r.get("metrics_found", 0)) for r in items)
        out["success_rate"] = (
            (out["n_success"] / out["n_runs"]) if out["n_runs"] > 0 else None
        )

        for metric in metric_keys:
            vals = [r.get(metric) for r in items if r.get(metric) is not None]
            mean_v, std_v = _mean_std(vals)
            out[f"{metric}_mean"] = mean_v
            out[f"{metric}_std"] = std_v

        out_rows.append(out)

    def _sort_key(row):
        key_parts = []
        for f in group_fields:
            v = row.get(f)
            if isinstance(v, (int, float)):
                key_parts.append((0, float(v)))
            elif v is None:
                key_parts.append((2, ""))
            else:
                key_parts.append((1, str(v)))
        return tuple(key_parts)

    out_rows.sort(key=_sort_key)
    return out_rows


def _compute_hp_overall_rows(
    per_run_rows: Sequence[Dict],
    hp_tf_mf_rows: Sequence[Dict],
    metric_keys: Sequence[str],
) -> List[Dict]:
    hp_key_fields = ("exp_name", *HP_FIELDS)

    grouped_runs: Dict[Tuple, List[Dict]] = {}
    for row in per_run_rows:
        key = tuple(row.get(f) for f in hp_key_fields)
        grouped_runs.setdefault(key, []).append(row)

    grouped_pair: Dict[Tuple, List[Dict]] = {}
    for row in hp_tf_mf_rows:
        key = tuple(row.get(f) for f in hp_key_fields)
        grouped_pair.setdefault(key, []).append(row)

    out_rows: List[Dict] = []
    for key, run_items in grouped_runs.items():
        out = {field: value for field, value in zip(hp_key_fields, key)}
        out["n_runs"] = len(run_items)
        out["n_success"] = sum(int(r.get("status_ok", 0)) for r in run_items)
        out["n_metrics_found"] = sum(int(r.get("metrics_found", 0)) for r in run_items)
        out["success_rate"] = (out["n_success"] / out["n_runs"]) if out["n_runs"] > 0 else None

        pair_items = grouped_pair.get(key, [])
        out["n_tf_mf_pairs"] = len(pair_items)

        for metric in metric_keys:
            run_vals = [r.get(metric) for r in run_items if r.get(metric) is not None]
            run_mean, run_std = _mean_std(run_vals)
            out[f"{metric}_run_mean"] = run_mean
            out[f"{metric}_run_std"] = run_std

            pair_vals = [r.get(f"{metric}_mean") for r in pair_items if r.get(f"{metric}_mean") is not None]
            pair_mean, pair_std = _mean_std(pair_vals)
            out[f"{metric}_pair_mean"] = pair_mean
            out[f"{metric}_pair_std"] = pair_std

        out_rows.append(out)

    out_rows.sort(
        key=lambda r: (
            str(r.get("exp_name")),
            float(r.get("learning_rate") or float("inf")),
            float(r.get("weight_decay") or float("inf")),
            str(r.get("scheduler_type") or ""),
            float(r.get("warmup_ratio") or float("inf")),
            int(r.get("batch_size") or 10**9),
        )
    )
    return out_rows


def _select_best_by_tf_mf(
    hp_tf_mf_rows: Sequence[Dict], metric: str, maximize: bool
) -> List[Dict]:
    grouped: Dict[Tuple[int, float], List[Dict]] = {}
    metric_key = f"{metric}_mean"
    for row in hp_tf_mf_rows:
        tf = row.get("Tfore")
        mf = row.get("Mf")
        if tf is None or mf is None:
            continue
        grouped.setdefault((int(tf), float(mf)), []).append(row)

    out_rows: List[Dict] = []
    for (tf, mf), items in sorted(grouped.items()):
        candidates = [r for r in items if r.get(metric_key) is not None]
        if not candidates:
            out_rows.append(
                {
                    "Tfore": tf,
                    "Mf": mf,
                    "best_metric_name": metric,
                    "best_metric_column": metric_key,
                    "best_metric_value": None,
                    "best_exp_name": None,
                    "learning_rate": None,
                    "weight_decay": None,
                    "scheduler_type": None,
                    "warmup_ratio": None,
                    "batch_size": None,
                    "n_seed_runs": None,
                    "success_rate": None,
                }
            )
            continue

        if maximize:
            best = max(candidates, key=lambda r: float(r[metric_key]))
        else:
            best = min(candidates, key=lambda r: float(r[metric_key]))

        out_rows.append(
            {
                "Tfore": tf,
                "Mf": mf,
                "best_metric_name": metric,
                "best_metric_column": metric_key,
                "best_metric_value": best.get(metric_key),
                "best_exp_name": best.get("exp_name"),
                "learning_rate": best.get("learning_rate"),
                "weight_decay": best.get("weight_decay"),
                "scheduler_type": best.get("scheduler_type"),
                "warmup_ratio": best.get("warmup_ratio"),
                "batch_size": best.get("batch_size"),
                "n_seed_runs": best.get("n_runs"),
                "success_rate": best.get("success_rate"),
            }
        )
    return out_rows


def _select_best_overall(
    hp_overall_rows: Sequence[Dict],
    metric: str,
    metric_source: str,
    maximize: bool,
) -> List[Dict]:
    metric_key = f"{metric}_{metric_source}"
    candidates = [r for r in hp_overall_rows if r.get(metric_key) is not None]
    if not candidates:
        return [
            {
                "best_metric_name": metric,
                "best_metric_source": metric_source,
                "best_metric_column": metric_key,
                "best_metric_value": None,
                "best_exp_name": None,
                "learning_rate": None,
                "weight_decay": None,
                "scheduler_type": None,
                "warmup_ratio": None,
                "batch_size": None,
                "n_runs": None,
                "success_rate": None,
            }
        ]

    if maximize:
        best = max(candidates, key=lambda r: float(r[metric_key]))
    else:
        best = min(candidates, key=lambda r: float(r[metric_key]))

    return [
        {
            "best_metric_name": metric,
            "best_metric_source": metric_source,
            "best_metric_column": metric_key,
            "best_metric_value": best.get(metric_key),
            "best_exp_name": best.get("exp_name"),
            "learning_rate": best.get("learning_rate"),
            "weight_decay": best.get("weight_decay"),
            "scheduler_type": best.get("scheduler_type"),
            "warmup_ratio": best.get("warmup_ratio"),
            "batch_size": best.get("batch_size"),
            "n_runs": best.get("n_runs"),
            "success_rate": best.get("success_rate"),
        }
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate analysis for experiments/clf_trainhp sweep outputs."
    )
    parser.add_argument(
        "--exp_root",
        type=str,
        default="experiments/clf_trainhp",
        help="Root directory that contains hp_* experiment subfolders.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <exp_root>/reports",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRIC_KEYS),
        help="Comma-separated metrics to collect and aggregate.",
    )
    parser.add_argument(
        "--metrics_prefer",
        type=str,
        choices=["best", "last", "first"],
        default="best",
        help="Which checkpoint metrics file to prefer when multiple metrics_test_*.json exist.",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="R",
        help="Metric name used for selecting best hyperparameter settings.",
    )
    parser.add_argument(
        "--best_mode",
        type=str,
        choices=["max", "min"],
        default="max",
        help="Whether best metric is maximized or minimized.",
    )
    parser.add_argument(
        "--overall_metric_source",
        type=str,
        choices=["pair_mean", "run_mean"],
        default="pair_mean",
        help=(
            "Metric source for selecting best overall hyperparameter: "
            "pair_mean (equal weight to each Tfore-Mf pair) or run_mean."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    exp_root = Path(args.exp_root)
    if not exp_root.is_absolute():
        exp_root = (ROOT / exp_root).resolve()
    if not exp_root.exists():
        raise FileNotFoundError(f"Experiment root not found: {exp_root}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else exp_root / "reports"
    metric_keys = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    if not metric_keys:
        raise ValueError("--metrics must contain at least one key")

    per_run_rows, exp_index_rows = collect_all_runs(
        exp_root=exp_root,
        metric_keys=metric_keys,
        metrics_prefer=args.metrics_prefer,
    )

    hp_tf_mf_rows = _group_rows(
        per_run_rows,
        group_fields=(
            "exp_name",
            "learning_rate",
            "weight_decay",
            "scheduler_type",
            "warmup_ratio",
            "batch_size",
            "Tfore",
            "Mf",
        ),
        metric_keys=metric_keys,
    )
    hp_overall_rows = _compute_hp_overall_rows(per_run_rows, hp_tf_mf_rows, metric_keys)

    best_by_tf_mf = _select_best_by_tf_mf(
        hp_tf_mf_rows,
        metric=args.best_metric,
        maximize=(args.best_mode == "max"),
    )
    best_overall = _select_best_overall(
        hp_overall_rows,
        metric=args.best_metric,
        metric_source=args.overall_metric_source,
        maximize=(args.best_mode == "max"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    exp_index_csv = out_dir / "clf_trainhp_experiments_index.csv"
    per_run_csv = out_dir / "clf_trainhp_metrics_per_run.csv"
    hp_tf_mf_csv = out_dir / "clf_trainhp_metrics_hp_tf_mf_mean_std.csv"
    hp_overall_csv = out_dir / "clf_trainhp_metrics_hp_overall.csv"
    best_pair_csv = out_dir / "clf_trainhp_best_hp_by_tf_mf.csv"
    best_overall_csv = out_dir / "clf_trainhp_best_hp_overall.csv"
    summary_json = out_dir / "clf_trainhp_metrics_summary.json"

    _write_csv(exp_index_csv, exp_index_rows)
    _write_csv(per_run_csv, per_run_rows)
    _write_csv(hp_tf_mf_csv, hp_tf_mf_rows)
    _write_csv(hp_overall_csv, hp_overall_rows)
    _write_csv(best_pair_csv, best_by_tf_mf)
    _write_csv(best_overall_csv, best_overall)

    payload = {
        "exp_root": str(exp_root),
        "out_dir": str(out_dir),
        "n_sub_experiments": len(exp_index_rows),
        "n_runs": len(per_run_rows),
        "metric_keys": metric_keys,
        "metrics_prefer": args.metrics_prefer,
        "best_metric": args.best_metric,
        "best_mode": args.best_mode,
        "overall_metric_source": args.overall_metric_source,
        "files": {
            "experiments_index_csv": str(exp_index_csv),
            "per_run_csv": str(per_run_csv),
            "hp_tf_mf_mean_std_csv": str(hp_tf_mf_csv),
            "hp_overall_csv": str(hp_overall_csv),
            "best_hp_by_tf_mf_csv": str(best_pair_csv),
            "best_hp_overall_csv": str(best_overall_csv),
        },
    }
    _write_json(summary_json, payload)

    print(f"[OK] experiments: {len(exp_index_rows)}")
    print(f"[OK] per-run rows: {len(per_run_rows)}")
    print(f"[OK] hp-(tf,mf) rows: {len(hp_tf_mf_rows)}")
    print(f"[OK] hp-overall rows: {len(hp_overall_rows)}")
    print(f"[OUT] {exp_index_csv}")
    print(f"[OUT] {per_run_csv}")
    print(f"[OUT] {hp_tf_mf_csv}")
    print(f"[OUT] {hp_overall_csv}")
    print(f"[OUT] {best_pair_csv}")
    print(f"[OUT] {best_overall_csv}")
    print(f"[OUT] {summary_json}")


if __name__ == "__main__":
    main()
