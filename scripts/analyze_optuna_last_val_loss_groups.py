#!/usr/bin/env python3
"""Analyze which Optuna trial groups have lower last-checkpoint val loss.

Inputs:
  - optuna_trials directory (contains trial_* subdirectories)
  - each trial directory should contain:
      - last_model_1.pth
      - metrics_optuna_val.json

Outputs (under --out-dir):
  - trial_last_val_loss_rows.csv
  - low_group_trials.csv
  - high_group_trials.csv
  - last_val_loss_group_analysis.json
"""

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


DEFAULT_PARAM_KEYS = (
    "scheduler_type",
    "warmup_ratio",
    "step_lr_step_size_ratio",
    "step_lr_gamma",
    "learning_rate",
    "weight_decay",
    "encoder_learning_rate",
    "batch_size",
    "criterion_cfg.beta",
    "mlp_dropout",
)

TRIAL_DIR_PATTERN = re.compile(r"^trial_(\d+)(?:_.+)?$")


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
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    low_idx = int(math.floor(pos))
    high_idx = int(math.ceil(pos))
    if low_idx == high_idx:
        return float(sorted_values[low_idx])
    weight = pos - low_idx
    return float(sorted_values[low_idx] * (1.0 - weight) + sorted_values[high_idx] * weight)


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _counter_dict(values: Sequence[Any]) -> Dict[str, int]:
    normalized = []
    for value in values:
        if value is None:
            normalized.append("__MISSING__")
        else:
            normalized.append(str(value))
    counter = Counter(normalized)
    return dict(counter)


def _param_col_name(key: str) -> str:
    return f"param_{key.replace('.', '__')}"


def _resolve_group_size(n_rows: int, fraction: float, k: Optional[int]) -> int:
    if n_rows <= 0:
        return 0
    if k is not None:
        return max(1, min(n_rows, int(k)))
    size = int(round(n_rows * fraction))
    return max(1, min(n_rows, size))


def collect_trial_rows(
    trials_dir: Path,
    *,
    last_model_file: str,
    metrics_file: str,
    param_keys: Sequence[str],
) -> Tuple[List[Dict], List[str]]:
    rows: List[Dict] = []
    warnings: List[str] = []

    for trial_dir in sorted(p for p in trials_dir.iterdir() if p.is_dir()):
        match = TRIAL_DIR_PATTERN.match(trial_dir.name)
        if not match:
            continue
        trial_number = int(match.group(1))
        checkpoint_path = trial_dir / last_model_file
        if not checkpoint_path.exists():
            warnings.append(f"missing checkpoint: {checkpoint_path}")
            continue

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            warnings.append(f"failed loading checkpoint: {checkpoint_path} ({exc})")
            continue

        if not isinstance(checkpoint, dict):
            warnings.append(f"invalid checkpoint format (not dict): {checkpoint_path}")
            continue

        last_val_loss = _to_float(checkpoint.get("val_loss"))
        if last_val_loss is None:
            warnings.append(f"missing val_loss in checkpoint: {checkpoint_path}")
            continue

        metrics_path = trial_dir / metrics_file
        sampled_params: Dict[str, Any] = {}
        best_val_loss = None
        objective_score = None
        profile = None

        if metrics_path.exists():
            try:
                metrics = _read_json(metrics_path)
            except Exception as exc:
                warnings.append(f"failed loading metrics: {metrics_path} ({exc})")
                metrics = {}
            if isinstance(metrics, dict):
                profile = metrics.get("profile")
                maybe_sampled_params = metrics.get("sampled_params")
                if isinstance(maybe_sampled_params, dict):
                    sampled_params = maybe_sampled_params
                best_val_loss = _to_float(metrics.get("best_val_loss"))
                objective_score = _to_float(metrics.get("objective_score"))
        else:
            warnings.append(f"missing metrics file: {metrics_path}")

        if best_val_loss is None:
            best_val_loss = objective_score

        row = {
            "trial": trial_number,
            "trial_dir_name": trial_dir.name,
            "trial_dir": str(trial_dir),
            "profile": profile,
            "last_val_loss": last_val_loss,
            "best_val_loss": best_val_loss,
            "last_minus_best": None if best_val_loss is None else float(last_val_loss - best_val_loss),
            "params_json": json.dumps(sampled_params, ensure_ascii=False, sort_keys=True),
        }
        for key in param_keys:
            row[_param_col_name(key)] = sampled_params.get(key)
        rows.append(row)

    rows.sort(key=lambda r: (r["last_val_loss"], r["trial"]))
    return rows, warnings


def analyze_groups(
    rows: Sequence[Dict],
    *,
    param_keys: Sequence[str],
    low_fraction: float,
    high_fraction: float,
    low_k: Optional[int],
    high_k: Optional[int],
) -> Dict:
    n_rows = len(rows)
    if n_rows == 0:
        return {
            "n_rows": 0,
            "message": "No valid trial rows found.",
            "low_group": {},
            "high_group": {},
            "global_stats": {},
            "param_analysis": {},
        }

    low_size = _resolve_group_size(n_rows, low_fraction, low_k)
    high_size = _resolve_group_size(n_rows, high_fraction, high_k)
    low_group = list(rows[:low_size])
    high_group = list(rows[-high_size:])

    loss_values = [float(r["last_val_loss"]) for r in rows]
    global_stats = {
        "min": min(loss_values),
        "p25": _percentile(loss_values, 0.25),
        "median": _percentile(loss_values, 0.50),
        "p75": _percentile(loss_values, 0.75),
        "max": max(loss_values),
        "mean": _safe_mean(loss_values),
    }

    low_losses = [float(r["last_val_loss"]) for r in low_group]
    high_losses = [float(r["last_val_loss"]) for r in high_group]
    low_gaps = [_to_float(r.get("last_minus_best")) for r in low_group]
    high_gaps = [_to_float(r.get("last_minus_best")) for r in high_group]
    low_gaps = [v for v in low_gaps if v is not None]
    high_gaps = [v for v in high_gaps if v is not None]

    param_analysis: Dict[str, Dict[str, Any]] = {}
    for key in param_keys:
        col_name = _param_col_name(key)
        all_values = [r.get(col_name) for r in rows]
        low_values = [r.get(col_name) for r in low_group]
        high_values = [r.get(col_name) for r in high_group]

        numeric_all = [_to_float(v) for v in all_values]
        numeric_low = [_to_float(v) for v in low_values]
        numeric_high = [_to_float(v) for v in high_values]

        numeric_all = [v for v in numeric_all if v is not None]
        numeric_low = [v for v in numeric_low if v is not None]
        numeric_high = [v for v in numeric_high if v is not None]

        payload = {
            "low_counts": _counter_dict(low_values),
            "high_counts": _counter_dict(high_values),
            "all_counts": _counter_dict(all_values),
        }
        if len(numeric_all) >= max(2, int(0.8 * len(all_values))):
            payload["numeric"] = {
                "low_mean": _safe_mean(numeric_low),
                "low_median": _safe_median(numeric_low),
                "high_mean": _safe_mean(numeric_high),
                "high_median": _safe_median(numeric_high),
                "all_mean": _safe_mean(numeric_all),
                "all_median": _safe_median(numeric_all),
            }
        param_analysis[key] = payload

    overlap_trials = sorted(
        set(int(r["trial"]) for r in low_group).intersection(int(r["trial"]) for r in high_group)
    )

    return {
        "n_rows": n_rows,
        "low_group": {
            "size": low_size,
            "fraction": low_size / n_rows,
            "max_last_val_loss": max(low_losses),
            "trial_numbers": [int(r["trial"]) for r in low_group],
            "mean_last_val_loss": _safe_mean(low_losses),
            "mean_last_minus_best": _safe_mean(low_gaps),
        },
        "high_group": {
            "size": high_size,
            "fraction": high_size / n_rows,
            "min_last_val_loss": min(high_losses),
            "trial_numbers": [int(r["trial"]) for r in high_group],
            "mean_last_val_loss": _safe_mean(high_losses),
            "mean_last_minus_best": _safe_mean(high_gaps),
        },
        "groups_overlap_trial_numbers": overlap_trials,
        "global_stats": global_stats,
        "top_low_trials": low_group,
        "top_high_trials": high_group,
        "param_analysis": param_analysis,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze characteristics of trial groups with low last-checkpoint val loss."
    )
    parser.add_argument(
        "--trials-dir",
        type=Path,
        required=True,
        help="Path to optuna_trials directory (e.g., .../runs/base/optuna_trials).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for analysis files. Default: <trials-dir>/../analysis_last_val_loss",
    )
    parser.add_argument(
        "--low-fraction",
        type=float,
        default=0.25,
        help="Fraction for low-loss group (used when --low-k is not set).",
    )
    parser.add_argument(
        "--high-fraction",
        type=float,
        default=0.25,
        help="Fraction for high-loss group (used when --high-k is not set).",
    )
    parser.add_argument(
        "--low-k",
        type=int,
        default=None,
        help="Absolute size of low-loss group. Overrides --low-fraction.",
    )
    parser.add_argument(
        "--high-k",
        type=int,
        default=None,
        help="Absolute size of high-loss group. Overrides --high-fraction.",
    )
    parser.add_argument(
        "--last-model-file",
        type=str,
        default="last_model_1.pth",
        help="Checkpoint filename inside each trial directory.",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="metrics_optuna_val.json",
        help="Metrics JSON filename inside each trial directory.",
    )
    parser.add_argument(
        "--param-keys",
        nargs="+",
        default=list(DEFAULT_PARAM_KEYS),
        help="Parameter keys to compare between groups.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    trials_dir = args.trials_dir.expanduser().resolve()
    if not trials_dir.exists() or not trials_dir.is_dir():
        raise FileNotFoundError(f"trials dir not found: {trials_dir}")

    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else (trials_dir.parent / "analysis_last_val_loss").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, warnings = collect_trial_rows(
        trials_dir,
        last_model_file=args.last_model_file,
        metrics_file=args.metrics_file,
        param_keys=args.param_keys,
    )
    analysis = analyze_groups(
        rows,
        param_keys=args.param_keys,
        low_fraction=float(args.low_fraction),
        high_fraction=float(args.high_fraction),
        low_k=args.low_k,
        high_k=args.high_k,
    )
    analysis_payload = {
        "trials_dir": str(trials_dir),
        "out_dir": str(out_dir),
        "warnings": warnings,
        "analysis": analysis,
    }

    low_rows = analysis.get("top_low_trials", []) if isinstance(analysis, dict) else []
    high_rows = analysis.get("top_high_trials", []) if isinstance(analysis, dict) else []

    _write_csv(out_dir / "trial_last_val_loss_rows.csv", rows)
    _write_csv(out_dir / "low_group_trials.csv", low_rows)
    _write_csv(out_dir / "high_group_trials.csv", high_rows)
    _write_json(out_dir / "last_val_loss_group_analysis.json", analysis_payload)

    print(f"[OK] Loaded trials: {len(rows)}")
    if warnings:
        print(f"[WARN] Issues: {len(warnings)} (see JSON report)")

    low = analysis.get("low_group", {}) if isinstance(analysis, dict) else {}
    high = analysis.get("high_group", {}) if isinstance(analysis, dict) else {}
    print(
        "[LOW] size={size} threshold<={thr}".format(
            size=low.get("size"),
            thr=low.get("max_last_val_loss"),
        )
    )
    print(
        "[HIGH] size={size} threshold>={thr}".format(
            size=high.get("size"),
            thr=high.get("min_last_val_loss"),
        )
    )
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()
