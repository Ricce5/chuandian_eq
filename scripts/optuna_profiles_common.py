#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


def parse_profiles(raw_profiles: str, config_path: Path):
    raw = str(raw_profiles).strip().lower()
    if raw not in {"", "auto"}:
        return [x.strip() for x in raw_profiles.split(",") if x.strip()]

    cfg = OmegaConf.load(str(config_path))
    optuna_cfg = cfg.get("optuna", {})
    profiles_cfg = optuna_cfg.get("profiles", {})
    profiles = list(profiles_cfg.keys())
    if not profiles:
        raise ValueError("No optuna profiles found in config. Please pass --profiles explicitly.")
    return profiles


def run_cmd(cmd, cwd: Path):
    print("[CMD]", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd), check=False)


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def extract_best_from_summary(summary_path: Path):
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    best_trial = summary.get("best_trial", {})
    best_user_attrs = best_trial.get("user_attrs", {}) or {}
    best_metrics = best_user_attrs.get("val_metrics", {}) or {}
    return {
        "study_name": summary.get("study_name"),
        "n_trials_total": summary.get("n_trials_total"),
        "n_trials_completed": summary.get("n_trials_completed"),
        "best_trial_number": best_trial.get("number"),
        "best_value": safe_float(best_trial.get("value")),
        "best_rmse": safe_float(best_metrics.get("RMSE")),
        "best_dtw": safe_float(best_metrics.get("DTW")),
        "best_dtw_normalized": safe_float(best_metrics.get("DTW_normalized")),
        "best_params": best_trial.get("params", {}),
    }


def build_profiles_runner_parser(*, description: str, default_model: str, default_config: str, default_out_dir: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--profiles", type=str, default="auto", help="Comma-separated profile names, or 'auto' to read from config.")
    parser.add_argument("--out_dir", type=str, default=default_out_dir)
    parser.add_argument("--run_name", type=str, default=None, help="Optional run folder name; default uses timestamp.")
    parser.add_argument("--optuna_trials", type=int, default=None, help="Override number of Optuna trials.")
    parser.add_argument("--optuna_sampler_seed", type=int, default=None, help="Override Optuna sampler seed.")
    parser.add_argument("--optuna_storage", type=str, default=None, help="Override Optuna storage URI/path.")
    parser.add_argument("--optuna_n_jobs", type=int, default=None, help="Override parallel trial workers per process.")
    parser.add_argument(
        "--optuna_trial_test_ckpt_selects",
        type=str,
        default=None,
        help="Comma-separated checkpoint selectors to auto-test per trial, e.g. 'best,last'.",
    )
    parser.add_argument(
        "--optuna_trial_test_top_k",
        type=int,
        default=None,
        help="If >0, defer trial auto-test to post-hoc and run only top-k trials per profile.",
    )
    parser.add_argument("--optuna_dry_run", action="store_true", help="Dry-run Optuna sampling only (no training).")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable path.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop immediately when one profile run fails.")
    return parser


def run_optuna_profiles(args):
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    profiles = parse_profiles(args.profiles, config_path)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = Path(args.out_dir) / run_name
    runs_dir = run_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    for profile in profiles:
        profile_dir = runs_dir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python,
            "main.py",
            "--model",
            args.model,
            "--mode",
            "optuna",
            "--config",
            str(config_path),
            "--checkpoint_dir",
            str(profile_dir),
            "--optuna_profile",
            profile,
        ]
        if args.optuna_trials is not None:
            cmd.extend(["--optuna_trials", str(args.optuna_trials)])
        if args.optuna_sampler_seed is not None:
            cmd.extend(["--optuna_sampler_seed", str(args.optuna_sampler_seed)])
        if args.optuna_storage is not None:
            cmd.extend(["--optuna_storage", str(args.optuna_storage)])
        if args.optuna_n_jobs is not None:
            cmd.extend(["--optuna_n_jobs", str(args.optuna_n_jobs)])
        if args.optuna_trial_test_ckpt_selects is not None:
            cmd.extend(["--optuna_trial_test_ckpt_selects", str(args.optuna_trial_test_ckpt_selects)])
        if args.optuna_trial_test_top_k is not None:
            cmd.extend(["--optuna_trial_test_top_k", str(args.optuna_trial_test_top_k)])
        if args.optuna_dry_run:
            cmd.append("--optuna_dry_run")

        ret = run_cmd(cmd, repo_root)
        summary_path = profile_dir / f"optuna_summary_{profile}.json"
        parsed = extract_best_from_summary(summary_path)

        row = {
            "profile": profile,
            "status": "ok" if ret.returncode == 0 else "failed",
            "return_code": int(ret.returncode),
            "run_dir": str(profile_dir),
            "summary_path": str(summary_path),
            "study_name": None,
            "n_trials_total": None,
            "n_trials_completed": None,
            "best_trial_number": None,
            "best_value": None,
            "best_rmse": None,
            "best_dtw": None,
            "best_dtw_normalized": None,
            "best_params_json": None,
        }
        if parsed is not None:
            row.update(
                {
                    "study_name": parsed["study_name"],
                    "n_trials_total": parsed["n_trials_total"],
                    "n_trials_completed": parsed["n_trials_completed"],
                    "best_trial_number": parsed["best_trial_number"],
                    "best_value": parsed["best_value"],
                    "best_rmse": parsed["best_rmse"],
                    "best_dtw": parsed["best_dtw"],
                    "best_dtw_normalized": parsed["best_dtw_normalized"],
                    "best_params_json": json.dumps(parsed["best_params"], ensure_ascii=False),
                }
            )

        detail_rows.append(row)
        if ret.returncode != 0 and args.fail_fast:
            break

    ok_rows = [r for r in detail_rows if r["status"] == "ok" and r["best_value"] is not None]
    best_overall = min(ok_rows, key=lambda x: float(x["best_value"])) if ok_rows else None

    report = {
        "model": args.model,
        "config": str(config_path),
        "profiles": profiles,
        "run_root": str(run_root),
        "best_overall": best_overall,
        "results": detail_rows,
    }

    detail_json = run_root / "profile_optuna_detail.json"
    detail_csv = run_root / "profile_optuna_detail.csv"
    summary_json = run_root / "profile_optuna_summary.json"

    with open(detail_json, "w", encoding="utf-8") as f:
        json.dump(detail_rows, f, ensure_ascii=False, indent=2)

    fields = [
        "profile",
        "status",
        "return_code",
        "run_dir",
        "summary_path",
        "study_name",
        "n_trials_total",
        "n_trials_completed",
        "best_trial_number",
        "best_value",
        "best_rmse",
        "best_dtw",
        "best_dtw_normalized",
        "best_params_json",
    ]
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow({k: row.get(k) for k in fields})

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Run root: {run_root}")
    print(f"Done. Detail CSV: {detail_csv}")
    print(f"Done. Summary JSON: {summary_json}")
    if best_overall is None:
        print("No successful profile result with valid objective value.")
    else:
        print(
            "Best overall: "
            f"profile={best_overall['profile']} "
            f"best_value={best_overall['best_value']:.6f} "
            f"RMSE={best_overall['best_rmse']:.6f} "
            f"DTW={best_overall['best_dtw']:.6f}"
        )
